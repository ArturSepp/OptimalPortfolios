"""
Constrained risk budgeting solver.

Implements the logarithmic-barrier formulation of Richard J-C. and Roncalli T.
(2019), "Constrained Risk Budgeting Portfolios" (SSRN 3331184), for the
volatility risk measure R(x) = sqrt(x' Σ x):

    x*(λ) = argmin R(x) - λ Σ_i b_i ln(x_i)   s.t.  lo <= x <= hi, C x <= d
    λ* such that Σ_i x*_i(λ*) = 1

Algorithms:
    - Cyclical coordinate descent (CCD) with per-coordinate box projection
      (paper Algorithm 3) when only box bounds are present. Each coordinate
      update is the positive root of α_i x² + β_i x + γ_i = 0 with
          α_i = σ_i² + φ σ(x)
          β_i = ((Σx)_i - x_i σ_i²) - φ v_i σ(x)
          γ_i = -λ b_i σ(x)
      where φ = 0 and v = 0 outside the ADMM x-update.
    - ADMM with CCD x-update (paper Algorithm 4) when linear inequality rows
      C x <= d are present. The z-update is the Euclidean projection onto
      {z: C z <= d, lo <= z <= hi}, solved as a QP via quadprog. The penalty
      φ adapts by the He-Yang-Wang scheme (paper Appendix A.7).
    - Brent root-finding on λ with the bracket seeded by the scaling identity
      of the unconstrained problem, x*(tλ) = t·x*(λ): one solve at
      λ_0 = R(x_inverse_vol) (paper Remark 7) gives λ_est = λ_0 / Σx(λ_0),
      exact when no constraint binds. Each λ evaluation solves from the same
      cold start, so f(λ) = Σ_i x*_i(λ) - 1 is deterministic and the
      root-finding is noise-free.

This module replaces the vendored pyrb package (github.com/jcrichard/pyrb,
MIT licence), whose ConstrainedRiskBudgeting it reproduces to within ADMM
tolerance (~5e-5 in weight space) with no numba dependency; the CCD inner loop
uses rank-1 updates of Σx and σ²(x), so one cycle is O(n²). Parity against
frozen pyrb baselines and against the paper's published tables is pinned in
optimization/tests/risk_budgeting_solver_test.py.

The module is internal to optimalportfolios: the public entry point is
``opt_risk_budgeting`` in optimization/general/risk_budgeting.py.
"""
# packages
import math
import numpy as np
import quadprog
from scipy.optimize import brentq
from typing import Optional, Tuple

# algorithm tolerances (values mirror the retired pyrb defaults for parity)
CCD_TOL = 1e-10  # convergence on the squared CCD step ||x_k - x_{k-1}||²
ADMM_TOL = 1e-10  # convergence on max of squared ADMM step / primal / dual residuals
MAX_CCD_CYCLES = 5000
MAX_ADMM_ITERS = 5000
ROOT_XTOL = 2e-12  # convergence on λ in the Brent root-finding
SUM_ABS_TOL = 1e-10  # |Σx - 1| at which a λ iterate is accepted outright
BRACKET_SEED_WIDTH = 1.25  # initial bracket [λ_est / w, λ_est · w] around the scaling estimate
MAX_BRACKET_EXPANSIONS = 60
DEFAULT_MAX_WEIGHT = 1e3  # upper bound when bounds is None (effectively unbounded)
PINNED_BOX_ATOL = 1e-9  # lo == hi detection for fully pinned boxes
FEASIBILITY_ATOL = 1e-6  # tolerance on sum(lo) <= 1 <= sum(hi)

# ADMM adaptive penalisation, He, Yang and Wang (2000) / paper Appendix A.7
ADMM_PENALTY_RATIO = 10.0  # residual imbalance ratio triggering a φ update
ADMM_PENALTY_SCALE = 2.0  # multiplicative φ step τ
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
               lower_bounds: np.ndarray,
               upper_bounds: np.ndarray,
               lambda_log: float,
               x0: np.ndarray,
               varphi: float = 0.0,  # ADMM penalty φ; 0.0 outside the ADMM x-update
               v_x: Optional[np.ndarray] = None,  # ADMM anchor v = z - u; None -> zeros
               ) -> np.ndarray:
    """cyclical coordinate descent for the box-constrained log-barrier problem.

    Solves min_x sqrt(x'Σx) + (φ/2)||x - v||²_σ-scaled - λ Σ_i b_i ln(x_i)
    over lo <= x <= hi by coordinate-wise exact minimisation (paper Algorithm 3;
    with φ > 0 it is the x-update of Algorithm 4). Σx and σ²(x) are maintained
    by rank-1 updates per coordinate (O(n) each) and refreshed by a full O(n²)
    recomputation once per cycle to bound floating-point drift.

    Parameters
    ----------
    covar : np.ndarray, shape (n, n)
        Covariance matrix Σ.
    budgets : np.ndarray, shape (n,)
        Risk budgets b, non-negative.
    lower_bounds, upper_bounds : np.ndarray, shape (n,)
        Box bounds lo, hi.
    lambda_log : float
        Log-barrier multiplier λ.
    x0 : np.ndarray, shape (n,)
        Starting point.
    varphi : float
        ADMM penalty φ; 0.0 outside the ADMM x-update.
    v_x : np.ndarray, shape (n,), optional
        ADMM anchor v = z - u; zeros when None.

    Returns
    -------
    x : np.ndarray, shape (n,)
        The CCD fixed point (not normalised to sum one).
    """
    n = covar.shape[0]
    var = np.ascontiguousarray(np.diag(covar))
    if v_x is None:
        v_x = np.zeros(n)
    lam_b = lambda_log * budgets  # hoisted γ coefficients
    x = x0.copy()
    s_x = covar @ x  # Σx
    sigma2 = float(x @ s_x)  # σ²(x)
    for _cycle in range(MAX_CCD_CYCLES):
        x_prev = x.copy()
        for i in range(n):
            sigma_x = math.sqrt(sigma2) if sigma2 > 0.0 else 0.0
            var_i = var[i]
            x_i = x[i]
            alpha = var_i + varphi * sigma_x
            beta = (s_x[i] - x_i * var_i) - varphi * v_x[i] * sigma_x
            gamma = -lam_b[i] * sigma_x
            # positive root of α x² + β x + γ = 0; γ <= 0 so the discriminant >= β²
            x_new = (-beta + math.sqrt(beta * beta - 4.0 * alpha * gamma)) / (2.0 * alpha)
            x_new = min(max(x_new, lower_bounds[i]), upper_bounds[i])
            delta = x_new - x_i
            if delta != 0.0:
                sigma2 += 2.0 * delta * s_x[i] + delta * delta * var_i
                s_x += delta * covar[i]  # symmetric: row i == column i, contiguous access
                x[i] = x_new
        # refresh Σx and σ² against rank-1 drift once per cycle
        s_x = covar @ x
        sigma2 = float(x @ s_x)
        if float(np.sum((x - x_prev) ** 2)) <= CCD_TOL:
            break
    return x


def _project_polyhedron(v: np.ndarray,
                        c_rows: np.ndarray,
                        c_lhs: np.ndarray,
                        lower_bounds: np.ndarray,
                        upper_bounds: np.ndarray
                        ) -> np.ndarray:
    """euclidean projection of v onto {z: C z <= d, lo <= z <= hi} via quadprog.

    quadprog solves min (1/2) z'Pz - q'z s.t. G'z >= h; the projection uses
    P = I, q = v, and stacks the inequality rows with both box faces.
    """
    n = len(v)
    eye = np.eye(n)
    g_mat = np.vstack([c_rows, -eye, eye])
    h_vec = np.hstack([c_lhs, -lower_bounds, upper_bounds])
    return quadprog.solve_qp(eye, np.ascontiguousarray(v, dtype=float),
                             -g_mat.T, -h_vec, 0)[0]


def _admm_ccd_solve(covar: np.ndarray,
                    budgets: np.ndarray,
                    lower_bounds: np.ndarray,
                    upper_bounds: np.ndarray,
                    c_rows: np.ndarray,
                    c_lhs: np.ndarray,
                    lambda_log: float,
                    x0: np.ndarray
                    ) -> np.ndarray:
    """ADMM with CCD x-update for linear inequality constraints (paper Algorithm 4).

    Iterates x-update (CCD on the φ-augmented problem, warm-started at the
    current x), z-update (projection onto the polyhedron), and dual update
    u += x - z, with adaptive penalty φ (paper Appendix A.7). Stops when the
    max of squared step, primal residual ||x - z||² and dual residual
    ||φ(z - z_prev)||² drops below ADMM_TOL. The returned x satisfies the box
    exactly (enforced inside CCD) and C x <= d up to the primal residual,
    i.e. O(sqrt(ADMM_TOL)) per row.
    """
    varphi = 1.0
    x = x0.copy()
    z = x.copy()
    u = np.zeros_like(x)
    for _it in range(MAX_ADMM_ITERS):
        z_prev = z
        x_prev = x
        x = _ccd_solve(covar=covar, budgets=budgets,
                       lower_bounds=lower_bounds, upper_bounds=upper_bounds,
                       lambda_log=lambda_log, x0=x,
                       varphi=varphi, v_x=z - u)
        z = _project_polyhedron(v=x + u, c_rows=c_rows, c_lhs=c_lhs,
                                lower_bounds=lower_bounds, upper_bounds=upper_bounds)
        r = x - z  # primal residual
        s = varphi * (z - z_prev)  # dual residual
        u = u + r
        cvg = max(float(np.sum((x - x_prev) ** 2)),
                  float(np.sum(r ** 2)),
                  float(np.sum(s ** 2)))
        if cvg <= ADMM_TOL:
            break
        # adaptive penalisation: rebalance primal vs dual residuals
        primal_err = float(np.sum(r ** 2))
        dual_err = float(np.sum(s ** 2))
        if primal_err > ADMM_PENALTY_RATIO * dual_err and varphi < ADMM_PENALTY_MAX:
            varphi *= ADMM_PENALTY_SCALE
            u = u / ADMM_PENALTY_SCALE
        elif dual_err > ADMM_PENALTY_RATIO * primal_err and varphi > ADMM_PENALTY_MIN:
            varphi /= ADMM_PENALTY_SCALE
            u = u * ADMM_PENALTY_SCALE
    return x


def solve_constrained_risk_budgeting(covar: np.ndarray,
                                     budgets: np.ndarray = None,  # risk budgets b; None -> equal
                                     bounds: np.ndarray = None,  # (n, 2) [lower, upper] per asset
                                     c_rows: np.ndarray = None,  # (p, n) inequality matrix C
                                     c_lhs: np.ndarray = None,  # (p,) right-hand side d in C x <= d
                                     ) -> Tuple[np.ndarray, float]:
    """constrained risk budgeting portfolio with sum(x) = 1 via root-finding on λ.

    Finds x*(λ*) with Σ_i x*_i = 1 where x*(λ) solves the log-barrier problem
    min sqrt(x'Σx) - λ Σ_i b_i ln(x_i) over {lo <= x <= hi, C x <= d}. Routes
    to plain CCD when c_rows is None and to ADMM-CCD otherwise. λ* is found by
    Brent's method on a bracket seeded by the scaling identity
    λ_est = λ_0 / Σx(λ_0) with λ_0 = σ(x_inverse_vol) (paper Remark 7),
    expanding geometrically when the seed does not straddle the root; each
    f(λ) evaluation is solved from the same cold start, so the root-finding is
    deterministic.

    A fully pinned box (lo == hi elementwise, Σ lo == 1) is short-circuited to
    the pinned vector: f(λ) has no sign change in that case and the pinned
    weights are the unique feasible point.

    Parameters
    ----------
    covar : np.ndarray, shape (n, n)
        Covariance matrix Σ.
    budgets : np.ndarray, shape (n,), optional
        Risk budgets b (normalised internally). Equal budgets when None.
    bounds : np.ndarray, shape (n, 2), optional
        Per-asset [lower, upper] bounds; [0, 1e3] when None. This is the
        first element of ``Constraints.set_pyrb_constraints``.
    c_rows : np.ndarray, shape (p, n), optional
        Linear inequality matrix C in C x <= d (group constraints). Second
        element of ``Constraints.set_pyrb_constraints``.
    c_lhs : np.ndarray, shape (p,), optional
        Right-hand side d. Third element of ``Constraints.set_pyrb_constraints``.

    Returns
    -------
    (x, lambda_star) : Tuple[np.ndarray, float]
        Optimal weights (sum one to root-finding precision) and the barrier
        multiplier λ* = R(x*) at the optimum; λ* is NaN for a pinned box.

    Raises
    ------
    ValueError
        On invalid inputs, an infeasible box (Σ lo > 1 or Σ hi < 1), or a
        bracket failure. The caller decides the fallback.
    """
    covar = np.ascontiguousarray(covar, dtype=float)
    n = covar.shape[0]
    if budgets is None:
        budgets = np.ones(n) / n
    else:
        budgets = np.asarray(budgets, dtype=float)
    if bounds is not None:
        bounds = np.asarray(bounds, dtype=float)
    if c_rows is not None:
        c_rows = np.asarray(c_rows, dtype=float)
    if c_lhs is not None:
        c_lhs = np.asarray(c_lhs, dtype=float)
    # validate raw inputs before any normalisation or slicing so that every
    # invalid input raises ValueError (the caller's fallback contract)
    _validate_inputs(covar=covar, budgets=budgets, bounds=bounds,
                     c_rows=c_rows, c_lhs=c_lhs)
    budgets = budgets / np.sum(budgets)
    if bounds is None:
        lower_bounds = np.zeros(n)
        upper_bounds = np.full(n, DEFAULT_MAX_WEIGHT)
    else:
        lower_bounds = np.ascontiguousarray(bounds[:, 0])
        upper_bounds = np.ascontiguousarray(bounds[:, 1])

    # box feasibility for sum(x) = 1; a fully pinned box short-circuits the solve
    sum_lower = float(np.sum(lower_bounds))
    sum_upper = float(np.sum(upper_bounds))
    if sum_lower > 1.0 + FEASIBILITY_ATOL:
        raise ValueError(f"infeasible box: sum of lower bounds exceeds 1: got {sum_lower!r}")
    if sum_upper < 1.0 - FEASIBILITY_ATOL:
        raise ValueError(f"infeasible box: sum of upper bounds is below 1: got {sum_upper!r}")
    if np.allclose(lower_bounds, upper_bounds, atol=PINNED_BOX_ATOL):
        return lower_bounds.copy(), np.nan

    inv_vol = 1.0 / np.sqrt(np.diag(covar))
    x_cold = inv_vol / np.sum(inv_vol) / 100.0  # cold start for every λ evaluation

    cache = {'lam': None, 'x': None}

    def solve_at(lambda_log: float) -> np.ndarray:
        if c_rows is None:
            x = _ccd_solve(covar=covar, budgets=budgets,
                           lower_bounds=lower_bounds, upper_bounds=upper_bounds,
                           lambda_log=lambda_log, x0=x_cold)
        else:
            x = _admm_ccd_solve(covar=covar, budgets=budgets,
                                lower_bounds=lower_bounds, upper_bounds=upper_bounds,
                                c_rows=c_rows, c_lhs=c_lhs,
                                lambda_log=lambda_log, x0=x_cold)
        cache['lam'] = lambda_log
        cache['x'] = x
        return x

    def f(lambda_log: float) -> float:
        return float(np.sum(solve_at(lambda_log))) - 1.0

    # seed λ from the scaling identity of the unconstrained problem,
    # x*(tλ) = t·x*(λ): after one solve at λ_0, λ_est = λ_0 / Σx(λ_0) is exact
    # when no box or group constraint binds, and a tight starting guess otherwise
    x_start = inv_vol / np.sum(inv_vol)
    lam_0 = float(np.sqrt(x_start @ covar @ x_start))  # σ(x_inverse_vol), paper Remark 7
    sum_0 = float(np.sum(solve_at(lam_0)))
    if sum_0 <= 0.0:
        raise ValueError(f"risk-budgeting solve degenerate at lambda={lam_0!r}: "
                         f"sum(x)={sum_0!r}; check covar and budgets")
    lam_est = lam_0 / sum_0
    f_est = f(lam_est)
    if abs(f_est) <= SUM_ABS_TOL:
        return cache['x'], lam_est

    # bracket λ* around the estimate; f(λ) = Σx(λ) - 1 is increasing in λ
    lam_lo = lam_est / BRACKET_SEED_WIDTH
    lam_hi = lam_est * BRACKET_SEED_WIDTH
    if f_est < 0.0:
        lam_lo, f_lo = lam_est, f_est
        f_hi = f(lam_hi)
        for _ in range(MAX_BRACKET_EXPANSIONS):
            if f_hi >= 0.0:
                break
            lam_lo, f_lo = lam_hi, f_hi
            lam_hi *= 2.0
            f_hi = f(lam_hi)
    else:
        lam_hi, f_hi = lam_est, f_est
        f_lo = f(lam_lo)
        for _ in range(MAX_BRACKET_EXPANSIONS):
            if f_lo <= 0.0:
                break
            lam_hi, f_hi = lam_lo, f_lo
            lam_lo *= 0.5
            f_lo = f(lam_lo)
    if f_lo > 0.0 or f_hi < 0.0:
        raise ValueError(f"cannot bracket lambda_star: f({lam_lo!r})={f_lo!r}, "
                         f"f({lam_hi!r})={f_hi!r}; check constraint feasibility")

    # Brent root-finding on the bracket; f is deterministic (cold start per λ)
    lambda_star = float(brentq(f, lam_lo, lam_hi, xtol=ROOT_XTOL))
    if cache['lam'] == lambda_star:
        x_final = cache['x']
    else:
        x_final = solve_at(lambda_star)
    return x_final, lambda_star
