"""State hulls and exact quadratic/perspective objectives in funded coordinates."""

from __future__ import annotations

from dataclasses import dataclass
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


def build_model(
    problem, formulation="CORRIDOR", perspective=False, continuous=False, fixed_states=None
):
    """Build the exact state hull or a fixed-state QP with no hard-rule relaxation.

    All coordinates fixed by original bounds are eliminated algebraically. Cash
    is eliminated through exact funding, while its original bounds remain rows.
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
    rows = [wbp >= SCALE * lo, wbp <= SCALE * hi]
    if len(problem.group_lower):
        rows += [
            problem.groups @ wbp >= SCALE * problem.group_lower,
            problem.groups @ wbp <= SCALE * problem.group_upper,
        ]
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
    """Solve a fixed-state graph with explicit tolerances and return raw weights."""
    kwargs = (
        dict(tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10) if solver == "CLARABEL" else {}
    )
    model.optimisation.solve(solver=solver, **kwargs)
    return None if model.weights.value is None else np.asarray(model.weights.value).reshape(-1)
