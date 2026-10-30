"""State hulls and exact quadratic/perspective objectives in funded coordinates."""

from __future__ import annotations

from dataclasses import dataclass, field
import cvxpy as cp
import numpy as np

from .domain import compile_states, domain_bounds
from .types import check_psd

SCALE = 1e4


def diagonal_split(matrix):
    """Return a conservative verified diagonal split, including the singular case."""
    diagonal = np.diag(matrix)
    active = diagonal > 0
    d = np.zeros(len(matrix))
    if active.any():
        root = np.sqrt(diagonal[active])
        corr = matrix[np.ix_(active, active)] / np.outer(root, root)
        alpha = 0.5 * max(0.0, float(np.linalg.eigvalsh(corr).min()))
        d[active] = alpha * diagonal[active]
    check_psd(matrix - np.diag(d))
    return d


@dataclass
class ConicModel:
    """A CVXPY graph plus explicit variables for verified native start lifting."""

    optimisation: cp.Problem
    weights: cp.Expression
    displacement_bp: cp.Variable | None
    indices: np.ndarray
    reference: np.ndarray
    selectors: dict
    contributions: dict
    states: tuple
    linear_constraints: list
    occupancy: dict
    funded_covariance: np.ndarray
    split: np.ndarray
    constant: float
    gradient: np.ndarray
    direction: np.ndarray
    qp_diagnostics: dict = field(default_factory=dict, init=False)

    def initial_values(self, weights):
        """Map an audited original-space vector into the complete state assignment."""
        initial = {}
        if self.displacement_bp is not None:
            initial[self.displacement_bp.id] = SCALE * (
                weights[self.indices] - self.reference[self.indices]
            )
        for i, selector in self.selectors.items():
            row = self.states[i]
            # Prefer the closest admissible branch, with stable state ties.
            errors = [max(s.lower - weights[i], weights[i] - s.upper, 0.0) for s in row]
            j = int(np.argmin(errors))
            z = np.zeros(len(row))
            z[j] = 1.0
            t = z * weights[i] * SCALE
            initial[selector.id] = z
            initial[self.contributions[i].id] = t
        return initial


def _interval_rows(values, directions, constant, lower, upper, reduce_fixed):
    """Omit only satisfied constant intervals from a fixed-state sizing graph.

    Zero coefficients are tested exactly, without a numerical threshold. Keep
    every nonconstant row and every constant contradiction. This removes barrier
    degeneracy after fixed-coordinate elimination without enlarging the domain.
    """
    if not reduce_fixed:
        return [values >= lower, values <= upper]
    keep = (
        np.any(directions != 0.0, axis=1)
        | (constant < lower)
        | (constant > upper)
    )
    if not keep.any():
        return []
    return [values[keep] >= lower[keep], values[keep] <= upper[keep]]


def build_model(
    problem, formulation="CORRIDOR", perspective=False, continuous=False, fixed_states=None
):
    """Build a state hull or fixed-state QP using the declared group audit envelope.

    All coordinates fixed by original bounds are eliminated algebraically. Cash
    is eliminated through exact funding, while its original bounds remain rows.
    Group rows admit only the existing original-space audit tolerance. Original
    limits remain immutable and every delivered vector still requires that audit.
    Fixed-state QPs omit satisfied constant bound/group rows; contradictory rows
    remain. Integer hulls balance only the quadratic cone's coordinates; their
    linear rows, fixed-state sizing and continuous relaxation graphs stay unchanged.
    """
    states = compile_states(problem, formulation)
    lower, upper = domain_bounds(problem, formulation)
    n = len(lower)
    lo, hi = lower.copy(), upper.copy()
    if fixed_states is not None:
        for i, j in fixed_states.items():
            lo[i], hi[i] = states[i][j].lower, states[i][j].upper
    else:
        for i, row in enumerate(states):
            if len(row) == 1:
                lo[i], hi[i] = row[0].lower, row[0].upper
    indices = np.flatnonzero((lo != hi) & (np.arange(n) != problem.cash))
    r = problem.base.copy()
    fixed = (lo == hi) & (np.arange(n) != problem.cash)
    r[fixed] = lo[fixed]
    r[problem.cash] = problem.investment - r[np.arange(n) != problem.cash].sum()
    u = np.zeros((n, len(indices)))
    u[indices, np.arange(len(indices))] = 1.0
    u[problem.cash, :] = -1.0
    x = cp.Variable(len(indices), name="funded_bp") if len(indices) else None
    wbp = cp.Constant(SCALE * r) if x is None else SCALE * r + u @ x
    w = wbp / SCALE
    rows = _interval_rows(
        wbp, u, SCALE * r, SCALE * lo, SCALE * hi, fixed_states is not None
    )
    if len(problem.group_lower):
        # Compile the same per-row residual envelope for fixed QPs, integer
        # models and continuous relaxations. Inward rounding avoids exceeding
        # the declared allowance merely through endpoint subtraction/addition.
        group_lo = np.nextafter(
            problem.group_lower - problem.rules.audit_tolerance, problem.group_lower
        )
        group_hi = np.nextafter(
            problem.group_upper + problem.rules.audit_tolerance, problem.group_upper
        )
        rows += _interval_rows(
            problem.groups @ wbp, problem.groups @ u, problem.groups @ (SCALE * r),
            SCALE * group_lo, SCALE * group_hi, fixed_states is not None
        )
    selectors, contributions, occupancy = {}, {}, {}
    tickets, optional = [], []
    for i, row in enumerate(states):
        if i == problem.cash:
            continue
        if not row:
            rows.append(cp.Constant(0.0) >= 1.0)
            continue
        if fixed_states is not None or len(row) == 1:
            state = row[fixed_states[i]] if fixed_states is not None else row[0]
            # Already represented exactly in the vector bound rows above.
            z = cp.Constant(float(state.ticket))
        else:
            selector = cp.Variable(len(row), boolean=not continuous, name=f"states_{i}")
            contribution = cp.Variable(len(row), name=f"endpoints_{i}")
            selectors[i], contributions[i] = selector, contribution
            rows += [
                selector >= 0,
                selector <= 1,
                cp.sum(selector) == 1,
                cp.sum(contribution) == wbp[i],
                contribution >= cp.multiply(SCALE * np.array([s.lower for s in row]), selector),
                contribution <= cp.multiply(SCALE * np.array([s.upper for s in row]), selector),
            ]
            z = np.array([float(s.ticket) for s in row]) @ selector
        occupancy[i] = z
        tickets.append(z)
        if not problem.mandatory[i]:
            optional.append(z)
    if problem.rules.ticket_cap is not None:
        rows.append(cp.sum(cp.hstack(tickets)) <= problem.rules.ticket_cap)
    if problem.rules.discretionary_allowance is not None:
        rows.append(
            cp.sum(cp.hstack(optional or [cp.Constant(0.0)]))
            <= problem.rules.discretionary_allowance
        )
    linear = list(rows)
    # QIS RiskModel has no CVXPY funded-coordinate objective compiler.
    active = r - problem.target
    constant = float(active @ problem.matvec(active))
    gradient = u.T @ problem.matvec(active)
    if problem.covariance is not None:
        funded = u.T @ problem.covariance @ u
    else:
        b = u.T @ problem.factor_loadings
        funded = b @ problem.factor_covariance @ b.T + u.T @ (
            problem.residual_variance[:, None] * u
        )
    split = diagonal_split(funded) if perspective and len(indices) else np.zeros(len(indices))
    objective = cp.Constant(SCALE * constant)
    if x is not None:
        objective += 2 * gradient @ x
        if selectors and not continuous:
            # Algebraically identical to quad_form(x, Q)/SCALE. Balancing the
            # integer model's conic epigraph avoids false native MIP bounds;
            # basis-point hard rows and the incumbent/audit mapping are unchanged.
            objective += cp.quad_form(x / 100.0, cp.psd_wrap(funded - np.diag(split)))
        else:
            objective += cp.quad_form(x, cp.psd_wrap(funded - np.diag(split))) / SCALE
        if np.any(split):
            epigraph = cp.Variable(len(indices), name="perspective_epigraph")
            for j, i in enumerate(indices):
                z = cp.Constant(1.0) if problem.mandatory[i] else occupancy[i]
                # Keep bp-scaled hard rows, but balance the conic coordinates
                # in percentage-point units. The objective remains identical.
                rows.append(cp.SOC(epigraph[j] + z, cp.hstack([2 * x[j] / 100.0, epigraph[j] - z])))
            objective += split @ epigraph
    return ConicModel(
        cp.Problem(cp.Minimize(objective), rows),
        w,
        x,
        indices,
        r,
        selectors,
        contributions,
        states,
        linear,
        occupancy,
        funded,
        split,
        constant,
        gradient,
        u,
    )


def solve_qp(model, solver="CLARABEL"):
    """Return raw sizing weights, with one equivalent CLARABEL numerical retry.

    Automatic equilibration can stall on a heavily fixed funded-coordinate QP.
    A SolverError permits one fresh solve without that scaling, using the same
    graph and tolerances. Native infeasibility is not retried. Callers must still
    audit the returned weights in original space; no failure implies feasibility.
    """
    kwargs = (
        dict(tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10) if solver == "CLARABEL" else {}
    )
    model.qp_diagnostics.clear()
    model.qp_diagnostics.update(attempts=1, equilibration_retry=False)
    try:
        model.optimisation.solve(solver=solver, **kwargs)
    except cp.SolverError as error:
        model.qp_diagnostics["initial_error"] = str(error)
        if solver != "CLARABEL":
            raise
        model.qp_diagnostics.update(attempts=2, equilibration_retry=True)
        # A fresh native instance is required: equilibration is an immutable
        # CLARABEL setting on a cached solver. No graph coefficient is changed.
        model.optimisation.solve(
            solver=solver, warm_start=False, equilibrate_enable=False, **kwargs
        )
    return None if model.weights.value is None else np.asarray(model.weights.value).reshape(-1)
