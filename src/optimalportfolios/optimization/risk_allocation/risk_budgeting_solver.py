"""
Scale-consistent constrained risk budgeting with joint portfolio bounds.

For a positive scale s = sum(y), portfolio weights are w = y / s. Solve

    min_y  0.5 y' Q y - lambda sum_i b_i log(y_i)
    s.t.   lo s <= y <= hi s, C y <= d s.

Q is covariance rescaled by its largest diagonal entry; lambda is a positive
numerical scale, not a portfolio-policy parameter. All constraint faces are
homogeneous, so changing either scale leaves w unchanged. Normalization enforces
full investment WITHOUT changing the instrument or group bounds.

Eliminating s gives the normalized objective log(sigma(w)) - sum_i b_i log(w_i).
Without binding constraints its first-order conditions reproduce the requested
risk budgets. With binding bounds, budgets are preferences, not exact attainable
risk contributions. This is a deliberate correction of the absolute-bound
lambda search: that search can lack a root on a feasible fully invested set, and
can distinguish a group floor from the equivalent complementary group cap.
See the scaling-compatibility issue in Richard & Roncalli (2019), section 5.2,
https://arxiv.org/abs/1902.05710. Their absolute-bound constrained table values are
not numerical references for this homogeneous formulation.

CCD solves the quadratic/log proximal objective by exact coordinate updates.
ADMM projects onto the homogeneous instrument/group faces via quadprog.
Only a converged feasible iterate is returned; no SLSQP fallback or mandate
relaxation is used. The public entry point remains opt_risk_budgeting.

Covariance, asset eligibility, budgets, rebalancing and reporting are supplied by
callers and are not changed here. This module performs no annualisation.
"""
import math
from typing import Optional, Tuple

import numpy as np
import quadprog

# Squared residuals in the dimensionless quadratic/log lift. The inner solve
# must be more accurate than ADMM so a small proximal step is not false convergence.
CCD_TOL = 1e-20
ADMM_TOL = 1e-18
MAX_CCD_CYCLES = 5000
MAX_ADMM_ITERS = 5000
PINNED_BOX_ATOL = 1e-9
FEASIBILITY_ATOL = 1e-8

ADMM_PENALTY_RATIO = 10.0
ADMM_PENALTY_SCALE = 2.0
ADMM_PENALTY_MIN = 1e-6
ADMM_PENALTY_MAX = 1e6


def _validate_inputs(covar: np.ndarray,
                     budgets: np.ndarray,
                     bounds: Optional[np.ndarray],
                     c_rows: Optional[np.ndarray],
                     c_lhs: Optional[np.ndarray]
                     ) -> None:
    """validate raw solver inputs (before any normalisation or slicing),
    raising ValueError with the offending value."""
    n = covar.shape[0]
    if covar.ndim != 2 or covar.shape[0] != covar.shape[1]:
        raise ValueError(f"covar must be square: got shape {covar.shape!r}")
    if not np.all(np.isfinite(covar)):
        raise ValueError(f"covar contains non-finite values: "
                         f"n_nonfinite={int(np.sum(~np.isfinite(covar)))}")
    diag = np.diag(covar)
    if np.any(diag <= 0.0):
        raise ValueError(f"covar diagonal must be strictly positive: got {diag!r}")
    if budgets.shape != (n,):
        raise ValueError(f"budgets must have shape ({n},): got {budgets.shape!r}")
    if not np.all(np.isfinite(budgets)) or np.any(budgets < 0.0):
        raise ValueError(f"budgets must be finite and non-negative: got {budgets!r}")
    if np.sum(budgets) <= 0.0:
        raise ValueError(f"budgets must have a positive sum: got sum={np.sum(budgets)!r}")
    if bounds is not None:
        if bounds.ndim != 2 or bounds.shape != (n, 2):
            raise ValueError(f"bounds must have shape ({n}, 2): got {bounds.shape!r}")
        lower_bounds = bounds[:, 0]
        upper_bounds = bounds[:, 1]
        if np.any(lower_bounds > upper_bounds):
            bad = int(np.argmax(lower_bounds - upper_bounds))
            raise ValueError(f"lower bound exceeds upper bound at index {bad}: "
                             f"lo={lower_bounds[bad]!r}, hi={upper_bounds[bad]!r}")
        if np.any(lower_bounds < 0.0):
            bad = int(np.argmin(lower_bounds))
            raise ValueError(f"lower bounds must be non-negative (log-barrier domain): "
                             f"got lo[{bad}]={lower_bounds[bad]!r}")
    if (c_rows is None) != (c_lhs is None):
        raise ValueError(f"c_rows and c_lhs must both be given or both be None: "
                         f"got c_rows={c_rows!r}, c_lhs={c_lhs!r}")
    if c_rows is not None:
        if c_rows.ndim != 2 or c_rows.shape[1] != n:
            raise ValueError(f"c_rows must have shape (p, {n}): got {c_rows.shape!r}")
        if c_lhs.shape != (c_rows.shape[0],):
            raise ValueError(f"c_lhs must have shape ({c_rows.shape[0]},): "
                             f"got {c_lhs.shape!r}")


def _ccd_solve(covar: np.ndarray,
               budgets: np.ndarray,
               lambda_log: float,
               x0: np.ndarray,
               varphi: float = 0.0,
               v_x: Optional[np.ndarray] = None) -> np.ndarray:
    """Solve the nonnegative quadratic/log proximal objective by exact CCD.

    Each coordinate solves a*x_i**2 + beta*x_i - lambda*b_i = 0.
    The rationalized positive root avoids cancellation for very small budgets.
    Zero budgets contribute no logarithm and permit an exact zero position.
    """
    var = np.diag(covar)
    v_x = np.zeros_like(x0) if v_x is None else v_x
    lam_b = lambda_log * budgets
    x = x0.copy()
    for _cycle in range(MAX_CCD_CYCLES):
        previous = x.copy()
        s_x = covar @ x
        for i in range(len(x)):
            alpha = var[i] + varphi
            beta = s_x[i] - var[i] * x[i] - varphi * v_x[i]
            discriminant = math.sqrt(beta * beta + 4.0 * alpha * lam_b[i])
            if beta > 0.0:
                value = 2.0 * lam_b[i] / (discriminant + beta)
            else:
                value = (discriminant - beta) / (2.0 * alpha)
            s_x += (value - x[i]) * covar[i]
            x[i] = value
        step = float(np.sum((x - previous) ** 2))
        if step <= CCD_TOL:
            return x
    raise ValueError(
        f"CCD did not converge after {MAX_CCD_CYCLES} cycles: squared_step={step:.6g}")


def _project_polyhedron(v: np.ndarray,
                        rows: np.ndarray,
                        lhs: np.ndarray,
                        full_investment: bool = False) -> np.ndarray:
    """Project onto inequality faces, optionally with sum(w) = 1 as an equality."""
    if full_investment:
        rows = np.vstack([-np.ones(len(v)), rows])
        lhs = np.hstack([-1.0, lhs])
    return quadprog.solve_qp(
        np.eye(len(v)), np.ascontiguousarray(v, dtype=float),
        np.ascontiguousarray(-rows.T), -lhs, int(full_investment))[0]


def _admm_ccd_solve(covar: np.ndarray,
                    budgets: np.ndarray,
                    rows: np.ndarray,
                    lambda_log: float,
                    x0: np.ndarray) -> np.ndarray:
    """Solve the homogeneous constrained lift and return its feasible ADMM leg."""
    varphi = 1.0
    x = x0.copy()
    z = x.copy()
    u = np.zeros_like(x)
    lhs = np.zeros(len(rows))
    for _iteration in range(MAX_ADMM_ITERS):
        z_prev = z
        x_prev = x
        x = _ccd_solve(covar, budgets, lambda_log, x, varphi, z - u)
        z = _project_polyhedron(x + u, rows, lhs)
        r = x - z
        s = varphi * (z - z_prev)
        u += r
        primal_err = float(np.sum(r ** 2))
        dual_err = float(np.sum(s ** 2))
        cvg = max(float(np.sum((x - x_prev) ** 2)), primal_err, dual_err)
        if cvg <= ADMM_TOL:
            return z
        if primal_err > ADMM_PENALTY_RATIO * dual_err and varphi < ADMM_PENALTY_MAX:
            varphi *= ADMM_PENALTY_SCALE
            u /= ADMM_PENALTY_SCALE
        elif dual_err > ADMM_PENALTY_RATIO * primal_err and varphi > ADMM_PENALTY_MIN:
            varphi /= ADMM_PENALTY_SCALE
            u *= ADMM_PENALTY_SCALE
    raise ValueError(
        f"ADMM did not converge after {MAX_ADMM_ITERS} iterations: "
        f"convergence_metric={cvg:.6g}, primal_residual_sq={primal_err:.6g}, "
        f"dual_residual_sq={dual_err:.6g}, penalty={varphi:.6g}")


def solve_constrained_risk_budgeting(covar: np.ndarray,
                                     budgets: np.ndarray = None,
                                     bounds: np.ndarray = None,
                                     c_rows: np.ndarray = None,
                                     c_lhs: np.ndarray = None) -> Tuple[np.ndarray, float]:
    """Compute risk-budgeted weights on the jointly fully invested feasible set.

    Args:
        covar: Finite covariance matrix with strictly positive diagonal.
        budgets: Nonnegative risk budgets; None selects equal budgets.
        bounds: Per-instrument [lower, upper] portfolio weights; None is long-only.
        c_rows: Optional linear inequality matrix C in C w <= d.
        c_lhs: Optional right-hand side d.

    Returns:
        Fully invested weights and portfolio volatility (the normalized
        volatility/log multiplier); NaN multiplier for a completely pinned box.

    Raises:
        ValueError: Invalid inputs, infeasible joint constraints or nonconvergence.
    """
    covar = np.ascontiguousarray(covar, dtype=float)
    n = covar.shape[0]
    budgets = np.ones(n) / n if budgets is None else np.asarray(budgets, dtype=float)
    bounds = None if bounds is None else np.asarray(bounds, dtype=float)
    c_rows = None if c_rows is None else np.asarray(c_rows, dtype=float)
    c_lhs = None if c_lhs is None else np.asarray(c_lhs, dtype=float)
    _validate_inputs(covar, budgets, bounds, c_rows, c_lhs)
    budgets = budgets / budgets.sum()
    lower = np.zeros(n) if bounds is None else bounds[:, 0]
    # A fully invested long-only portfolio cannot put more than 100% in one asset.
    upper = np.ones(n) if bounds is None else np.minimum(bounds[:, 1], 1.0)
    if lower.sum() > 1.0 + FEASIBILITY_ATOL:
        raise ValueError(f"infeasible box: sum of lower bounds exceeds 1: got {lower.sum()!r}")
    if upper.sum() < 1.0 - FEASIBILITY_ATOL:
        raise ValueError(f"infeasible box: sum of upper bounds is below 1: got {upper.sum()!r}")

    eye = np.eye(n)
    rows = np.vstack([-eye, eye])
    lhs = np.hstack([-lower, upper])
    if c_rows is not None:
        rows = np.vstack([rows, c_rows])
        lhs = np.hstack([lhs, c_lhs])
    inv_vol = 1.0 / np.sqrt(np.diag(covar))
    start = inv_vol / inv_vol.sum()
    # Check the ACTUAL weight feasible set, including 100%, before a cone solve.
    try:
        feasible = _project_polyhedron(start, rows, lhs, full_investment=True)
    except ValueError as error:
        raise ValueError(f"infeasible fully invested risk-budget constraints: {error}") from error
    if np.all(np.abs(lower - upper) <= PINNED_BOX_ATOL):
        return feasible, np.nan

    # Fixed zero positions have no free risk budget, as in zero-budget exclusion.
    # Do not fabricate positive positions to make the logarithm finite.
    budgets = np.where(upper == 0.0, 0.0, budgets)
    if budgets.sum() == 0.0:
        raise ValueError("no positive risk budget remains outside zero-pinned positions")
    budgets /= budgets.sum()
    scaled_covar = covar / np.max(np.diag(covar))
    lambda_log = float(start @ scaled_covar @ start)
    if np.all(lower == 0.0) and np.all(upper == 1.0) and c_rows is None:
        solution = _ccd_solve(scaled_covar, budgets, lambda_log, start)
    else:
        # A w <= d, w = y/sum(y)  <=>  (A - d 1') y <= 0.
        # Both instrument AND group bounds must scale; post-normalizing a solve
        # with absolute bounds would change the original portfolio constraints.
        cone_rows = rows - lhs[:, None]
        norms = np.linalg.norm(cone_rows, axis=1)
        cone_rows = cone_rows[norms > 0.0] / norms[norms > 0.0, None]
        solution = _admm_ccd_solve(
            scaled_covar, budgets, cone_rows, lambda_log, feasible)
    total = float(solution.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError(f"degenerate risk-budgeting scale: sum(y)={total!r}")
    weights = solution / total
    # This projection removes roundoff only; material violations remain failures.
    if np.max(rows @ weights - lhs) > FEASIBILITY_ATOL:
        raise ValueError("risk-budgeting solution violates the original portfolio bounds")
    weights = _project_polyhedron(weights, rows, lhs, full_investment=True)
    return weights, float(np.sqrt(weights @ covar @ weights))
