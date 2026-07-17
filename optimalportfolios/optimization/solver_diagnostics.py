"""Solver diagnostics: post-solve feasibility validation and pre-solve
covariance conditioning warnings for the CVXPY optimisers.

Motivation
----------
Every CVXPY solver in ``optimalportfolios`` (max_sharpe, quadratic, the two
SAA solvers, the two TAA tracking-error solvers, maximise_alpha_with_target_yield)
shares the same two gaps:

1. The covariance is passed through ``cvx.psd_wrap`` — which *asserts* PSD to
   CVXPY rather than enforcing it, and in doing so also suppresses CVXPY's DCP
   convexity check (``problem.is_dcp()`` returns True even for an indefinite Σ).
2. The only post-solve check is ``if optimal_weights is None``. ``problem.status``
   is never inspected, and the returned ``w.value`` is never tested for
   feasibility. A solver that terminates ``optimal_inaccurate`` /
   ``*_inaccurate`` on an ill-conditioned KKT system returns a *populated but
   grossly infeasible* iterate (e.g. ``sum(w) ~ 1.5e6``), which the
   ``is None`` check happily accepts. ``weights[np.isinf]=0`` does not help —
   the bad iterate is finite.

This is exactly the GROWM (tre=100, turnover=0.2) blow-up: a near-collinear
private-asset block (two proxies at corr 1.00) made Σ rank-deficient
(cond ~5e12); CLARABEL returned a non-optimal iterate summing to 1.5e6; the
``is None`` check accepted it; one 2021 quarter then poisoned every
second-moment backtest statistic.

The failure cannot be reliably pre-empted by provoking the solver (modern
CLARABEL stays feasible even at cond ~6e14 on small problems) nor by checking
convexity (psd_wrap defeats the DCP check). The robust defence is to validate
the *output* unconditionally, on every solve. That is ``validate_solution``.

``check_covar_conditioning`` is a *diagnostic-only* early warning: it logs when
Σ is near-singular or indefinite (which is what would have flagged 2021-04
before the solve) and never modifies the matrix — its thresholds are logging
knobs, not optimisation parameters, so they introduce no estimation/objective
parameter and cannot change any result.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from optimalportfolios.optimization.constraints import Constraints, RelaxationRecord

logger = logging.getLogger(__name__)


# CVXPY status strings. Anything not in OK / INACCURATE is treated as a hard
# failure (this includes None and any unknown/solver-specific string).
_STATUS_OPTIMAL = "optimal"
_STATUS_INACCURATE = {"optimal_inaccurate"}
_STATUS_HARD_FAIL = {
    "infeasible",
    "unbounded",
    "infeasible_inaccurate",
    "unbounded_inaccurate",
    "infeasible_or_unbounded",
    "solver_error",
}


def _as_np(s: Optional[pd.Series], index: Sequence[str]) -> Optional[np.ndarray]:
    """Reindex an optional pd.Series to ``index`` and return as float ndarray."""
    if s is None:
        return None
    if isinstance(s, pd.Series):
        return s.reindex(index=pd.Index(index)).to_numpy(dtype=float)
    return np.asarray(s, dtype=float)


def _compute_fallback(constraints: Constraints, tickers: Sequence[str], n: int) -> Tuple[np.ndarray, str]:
    """Pick a guaranteed-finite fallback weight vector.

    Order: drifted ``weights_0`` → ``benchmark_weights`` → zeros. The rolling
    loop already treats an all-zero vector as 'skip this rebalance', so zeros
    is a safe terminal fallback. Each candidate is accepted only if finite.
    """
    w0 = _as_np(getattr(constraints, "weights_0", None), tickers)
    if w0 is not None and w0.shape == (n,) and np.all(np.isfinite(w0)):
        return np.array(w0, dtype=float), "weights_0"
    bench = _as_np(getattr(constraints, "benchmark_weights", None), tickers)
    if bench is not None and bench.shape == (n,) and np.all(np.isfinite(bench)):
        return np.array(bench, dtype=float), "benchmark_weights"
    return np.zeros(n), "zeros"


def _budget_target(constraints: Constraints) -> Tuple[float, float, bool]:
    """Return (min_exposure, max_exposure, is_equality) for the budget test."""
    max_exp = float(getattr(constraints, "max_exposure", 1.0))
    min_exp = float(getattr(constraints, "min_exposure", max_exp))
    return min_exp, max_exp, np.isclose(min_exp, max_exp)


def _derive_tickers(constraints: Constraints, n: int) -> List[Any]:
    """Best-effort asset labels (for logging / Series alignment)."""
    for _attr in ("benchmark_weights", "max_weights", "min_weights", "weights_0"):
        _s = getattr(constraints, _attr, None)
        if _s is not None and hasattr(_s, "index"):
            return list(_s.index)
    return list(range(n))


def _validate_weight_vector(w: np.ndarray, constraints: Constraints, n: int,
                            tickers: Sequence[str], budget_atol: float,
                            bound_atol: float) -> Tuple[bool, str]:
    """Solver-agnostic feasibility check on a weight vector → (ok, reason).

    Checks, in order: finiteness/shape, budget (full-investment or exposure
    band), long-only, per-asset max_weights, per-asset min_weights. Shared by
    the CVXPY, scipy and risk-budgeting validators so the accept/reject logic
    is identical across backends.
    """
    if w.shape != (n,) or not np.all(np.isfinite(w)):
        return False, (f"non-finite or wrong-shape weights (shape={w.shape}, "
                       f"n_nonfinite={int(np.sum(~np.isfinite(w)))})")
    min_exp, max_exp, is_eq = _budget_target(constraints)
    s = float(np.sum(w))
    if is_eq:
        if abs(s - max_exp) > budget_atol:
            return False, (f"budget violated: sum(w)={s:.6g} vs target {max_exp:.6g} "
                           f"(atol={budget_atol:g})")
    else:
        if s < min_exp - budget_atol or s > max_exp + budget_atol:
            return False, (f"exposure band violated: sum(w)={s:.6g} not in "
                           f"[{min_exp:.6g}, {max_exp:.6g}]")
    if getattr(constraints, "is_long_only", False):
        if w.min() < -bound_atol:
            return False, f"long-only violated: min(w)={w.min():.6g}"
    max_w = _as_np(getattr(constraints, "max_weights", None), tickers)
    if max_w is not None:
        over = float(np.max(w - max_w))
        if over > bound_atol:
            j = int(np.argmax(w - max_w))
            return False, (f"max_weight violated by {over:.6g} at index {j} "
                           f"(w={w[j]:.6g}, cap={max_w[j]:.6g})")
    min_w = _as_np(getattr(constraints, "min_weights", None), tickers)
    if min_w is not None:
        under = float(np.min(w - min_w))
        if under < -bound_atol:
            j = int(np.argmin(w - min_w))
            return False, (f"min_weight violated by {under:.6g} at index {j} "
                           f"(w={w[j]:.6g}, floor={min_w[j]:.6g})")
    return True, ""


@dataclass(frozen=True)
class SolverDiagnostic:
    """Structured per-solve outcome, attached to the log record under
    ``extra={"solver_diag": ...}`` so handlers can aggregate the *data* of a
    solve rather than parse its message string.

    Attributes:
        context: caller label (e.g. ``"GROWM 2021-04-30"``).
        solver: solver name.
        status: solver status string (CVXPY) or ``None``.
        outcome: ``"accepted"``, ``"accepted_inaccurate"`` or ``"rejected"``.
        accepted: True iff the solver output itself was used (not a fallback).
        reason: rejection reason (empty when accepted).
        fallback_source: which fallback was returned on rejection
            (``"weights_0"`` / ``"benchmark_weights"`` / ``"zeros"``), else None.
        severity: the logging level the event was emitted at.
        n_assets: problem size.
        sum_w: sum of the solver weights, or NaN if missing / non-finite.
        budget_residual: signed distance of sum_w outside the budget band
            (0.0 inside it; NaN if missing / non-finite).
        max_box_violation: worst box / long-only breach magnitude (0.0 if none;
            NaN if missing / non-finite).
    """
    context: str
    solver: str
    status: Optional[str]
    outcome: str
    accepted: bool
    reason: str
    fallback_source: Optional[str]
    severity: int
    n_assets: int
    sum_w: float
    budget_residual: float
    max_box_violation: float


_NAN_METRICS = {"sum_w": float("nan"), "budget_residual": float("nan"),
                "max_box_violation": float("nan")}


def _weight_metrics(w: Optional[np.ndarray], constraints: Constraints,
                    tickers: Sequence[str]) -> Dict[str, float]:
    """Raw (unthresholded) telemetry for the diagnostic: sum, budget residual,
    worst box / long-only breach. Independent of the accept/reject decision in
    ``_validate_weight_vector``. NaN when the weights are missing or non-finite.
    """
    if w is None or w.shape != (len(tickers),) or not np.all(np.isfinite(w)):
        return dict(_NAN_METRICS)
    s = float(np.sum(w))
    min_exp, max_exp, is_eq = _budget_target(constraints)
    if is_eq or s > max_exp:
        budget_residual = s - max_exp
    elif s < min_exp:
        budget_residual = s - min_exp
    else:
        budget_residual = 0.0
    viol = 0.0
    if getattr(constraints, "is_long_only", False):
        viol = max(viol, float(-np.min(w)))
    max_w = _as_np(getattr(constraints, "max_weights", None), tickers)
    if max_w is not None:
        viol = max(viol, float(np.max(w - max_w)))
    min_w = _as_np(getattr(constraints, "min_weights", None), tickers)
    if min_w is not None:
        viol = max(viol, float(np.max(min_w - w)))
    return {"sum_w": s, "budget_residual": float(budget_residual),
            "max_box_violation": max(0.0, viol)}


def _emit_diag(level: int, message: str, context: str, solver: str,
               status: Optional[str], outcome: str, accepted: bool, reason: str,
               fallback_source: Optional[str], w: Optional[np.ndarray],
               constraints: Constraints, tickers: Sequence[str], n: int) -> None:
    """Log ``message`` at ``level`` with a structured ``SolverDiagnostic`` on the
    record (``extra={"solver_diag": ...}``) for ``SolverRejectionSummary``."""
    metrics = _weight_metrics(w, constraints, tickers)
    diag = SolverDiagnostic(
        context=context, solver=solver, status=status, outcome=outcome,
        accepted=accepted, reason=reason, fallback_source=fallback_source,
        severity=level, n_assets=n, sum_w=metrics["sum_w"],
        budget_residual=metrics["budget_residual"],
        max_box_violation=metrics["max_box_violation"])
    logger.log(level, message, extra={"solver_diag": diag})


def validate_solution(
    optimal_weights: Optional[np.ndarray],
    problem_status: Optional[str],
    constraints: Constraints,
    n: int,
    solver: str = "",
    context: str = "",
    budget_atol: float = 1e-4,
    bound_atol: float = 1e-6,
    accept_inaccurate: bool = True,
) -> Tuple[np.ndarray, bool]:
    """Validate a CVXPY solver result and return safe weights.

    Runs on every solve. Rejects a result that (a) is missing, (b) has a hard
    failure status, (c) is non-finite, (d) violates the budget, or (e) violates
    the box / long-only bounds — and on rejection returns the
    ``_compute_fallback`` vector (drifted weights_0 → benchmark → zeros) instead
    of the bad iterate. An ``optimal_inaccurate`` result is accepted *iff* it
    still passes the feasibility checks (and ``accept_inaccurate`` is True),
    with a warning; otherwise it is rejected like a hard failure.

    All rejections and degraded acceptances are logged via ``logger`` (a single
    channel — no ``warnings.warn``) with the solver, status, ``context`` (pass the
    rebalance date / mandate), and the offending magnitudes. Severity is
    graded: a solver that honestly reports no solution (None / infeasible /
    unbounded) logs at WARNING (a benign fallback to prior weights), whereas a
    result the solver calls optimal/inaccurate but which violates the budget or
    box bounds — the numerical blow-up case — logs at ERROR. Configure handlers
    in the application (see ``configure_run_logging``) to route these to a file
    and keep the console to ERROR.

    Args:
        optimal_weights: ``w.value`` from the solved problem (may be None).
        problem_status: ``problem.status``.
        constraints: the (ticker-aligned) Constraints used for the solve.
            Read for is_long_only, min/max exposure, min/max weights,
            benchmark_weights, weights_0.
        n: number of assets in the solve.
        solver: solver name, for logging.
        context: free-form label (e.g. ``f"{mandate} {date:%Y-%m-%d}"``).
        budget_atol: absolute tolerance on the full-investment / exposure-band
            constraint. The 1.5e6 incident fails this by ~6 orders of magnitude.
        bound_atol: absolute tolerance on the box / long-only bounds.
        accept_inaccurate: if True, an ``optimal_inaccurate`` status that is
            otherwise feasible is accepted (with a warning) rather than
            discarded — useful so a marginally-imprecise but sane solve is not
            needlessly replaced by the previous weights.

    Returns:
        (weights, is_valid). ``weights`` is always a finite ndarray of length
        ``n`` — either the accepted solver output or the fallback. ``is_valid``
        is True only when the solver output itself was accepted.
    """
    tickers = _derive_tickers(constraints, n)
    tag = f"[{context}] " if context else ""

    def _reject(reason: str, level: int = logging.ERROR,
                w: Optional[np.ndarray] = None) -> Tuple[np.ndarray, bool]:
        fallback, source = _compute_fallback(constraints, tickers, n)
        msg = (f"{tag}solver={solver} status={problem_status}: REJECTED solution "
               f"({reason}); falling back to {source}.")
        _emit_diag(level, msg, context, solver, problem_status, "rejected", False,
                   reason, source, w, constraints, tickers, n)
        return fallback, False

    # (1) missing solution
    if optimal_weights is None:
        return _reject("w.value is None", logging.WARNING)

    w = np.array(optimal_weights, dtype=float).ravel()  # copy -> writable (pandas 3.0 / downstream in-place edits)

    # (2) hard failure status (or unknown/None status string)
    if problem_status is None or (
        problem_status != _STATUS_OPTIMAL and problem_status not in _STATUS_INACCURATE
    ):
        # any non-optimal, non-inaccurate status (infeasible/unbounded/error/unknown)
        return _reject(f"hard solver status '{problem_status}'", logging.WARNING, w)

    # (3-5) finiteness, budget, and box / long-only feasibility
    ok, reason = _validate_weight_vector(
        w, constraints, n, tickers, budget_atol=budget_atol, bound_atol=bound_atol)
    if not ok:
        return _reject(reason, w=w)
    s = float(np.sum(w))

    # (6) inaccurate-but-feasible: accept with a warning, or reject if disallowed
    if problem_status in _STATUS_INACCURATE:
        if accept_inaccurate:
            _emit_diag(
                logging.WARNING,
                f"{tag}solver={solver} status={problem_status}: accepting "
                f"feasible-but-imprecise solution (sum(w)={s:.6g}, "
                f"max(w)={w.max():.6g}).",
                context, solver, problem_status, "accepted_inaccurate", True,
                "", None, w, constraints, tickers, n)
            return w, True
        return _reject(f"status '{problem_status}' and accept_inaccurate=False",
                       logging.WARNING, w)

    # (7) clean optimal + feasible
    _emit_diag(logging.DEBUG,
               f"{tag}solver={solver} status={problem_status}: accepted "
               f"(sum(w)={s:.6g}, max(w)={w.max():.6g}).",
               context, solver, problem_status, "accepted", True,
               "", None, w, constraints, tickers, n)
    return w, True


def validate_scipy_solution(
    optimal_weights: Optional[np.ndarray],
    res: Any,
    constraints: Constraints,
    n: int,
    solver: str = "SLSQP",
    context: str = "",
    budget_atol: float = 1e-4,
    bound_atol: float = 1e-6,
) -> Tuple[np.ndarray, bool]:
    """Validate a ``scipy.optimize`` result — the scipy analogue of
    ``validate_solution``.

    For the scipy backends (risk-budgeting SLSQP, CARA / CARA-mixture). Unlike
    CVXPY there is no ``problem.status``; the convergence signal is
    ``res.success`` plus ``res.status`` / ``res.message``. Rejects when scipy
    reports non-convergence, the weights are missing/non-finite, or the
    budget / box constraints are violated, and on rejection falls back drifted
    weights_0 → benchmark → zeros. scipy's status/message are logged.

    Args:
        optimal_weights: ``res.x`` (passed explicitly so the caller controls
            any pre-shaping).
        res: the ``OptimizeResult`` (read for ``success``/``status``/``message``).
        constraints: the ticker-aligned Constraints used for the solve.
        n: number of assets.
        solver: solver label for logging (e.g. ``"SLSQP"``).
        context: free-form label (rebalance date / mandate).
        budget_atol, bound_atol: feasibility tolerances.

    Returns:
        (weights, is_valid) — ``weights`` is always finite and length ``n``.
    """
    tickers = _derive_tickers(constraints, n)
    tag = f"[{context}] " if context else ""
    success = getattr(res, "success", None)
    status = getattr(res, "status", None)
    message = str(getattr(res, "message", "") or "")

    def _reject(reason: str, level: int = logging.ERROR,
                w: Optional[np.ndarray] = None) -> Tuple[np.ndarray, bool]:
        fallback, source = _compute_fallback(constraints, tickers, n)
        msg = (f"{tag}solver={solver} success={success} status={status}: "
               f"REJECTED solution ({reason}; scipy: {message}); "
               f"falling back to {source}.")
        _emit_diag(level, msg, context, solver, str(status), "rejected", False,
                   reason, source, w, constraints, tickers, n)
        return fallback, False

    if optimal_weights is None:
        return _reject("scipy returned no weights", logging.WARNING)
    w = np.array(optimal_weights, dtype=float).ravel()  # copy -> writable (pandas 3.0 / downstream in-place edits)
    if success is False:
        return _reject("scipy reported non-convergence", logging.WARNING, w)
    ok, reason = _validate_weight_vector(
        w, constraints, n, tickers, budget_atol=budget_atol, bound_atol=bound_atol)
    if not ok:
        return _reject(reason, w=w)
    _emit_diag(logging.DEBUG,
               f"{tag}solver={solver} success={success} status={status}: accepted "
               f"(sum(w)={float(np.sum(w)):.6g}).",
               context, solver, str(status), "accepted", True,
               "", None, w, constraints, tickers, n)
    return w, True


def validate_rb_solution(
    optimal_weights: Optional[np.ndarray],
    constraints: Constraints,
    n: int,
    solver: str = "risk_budgeting",
    context: str = "",
    converged: Optional[bool] = None,
    budget_atol: float = 1e-4,
    bound_atol: float = 1e-6,
    c_rows: Optional[np.ndarray] = None,
    c_lhs: Optional[np.ndarray] = None,
    group_atol: float = 1e-4,
) -> Tuple[np.ndarray, bool]:
    """Validate a risk-budgeting solver result (internal CCD / ADMM-CCD).

    The solver raises on failure, so the caller passes ``None`` for a failed
    solve; ``None`` / NaN / infeasible weights are rejected; an optional
    ``converged`` flag is honoured when the caller can derive one. Budget and
    box feasibility are checked against ``constraints`` (risk-budgeting solves
    are fully invested, ``sum(w)=max_exposure``); the group inequality rows
    ``c_rows @ w <= c_lhs`` are checked at ``group_atol`` when provided —
    ``group_atol`` is set to the ADMM primal-residual scale (~1e-4), not
    ``bound_atol``, because the ADMM z-projection enforces group rows only up
    to its primal residual while the box is enforced exactly inside CCD.
    Falls back drifted weights_0 → benchmark → zeros on rejection.

    Args:
        optimal_weights: solver weights, or ``None`` on solver failure.
        constraints: the ticker-aligned Constraints used for the solve.
        n: number of assets.
        solver: solver label for logging.
        context: free-form label (rebalance date / mandate).
        converged: optional convergence flag; ``False`` forces a reject.
        budget_atol, bound_atol: feasibility tolerances.
        c_rows, c_lhs: optional group inequality rows ``C w <= d`` as produced
            by ``Constraints.set_pyrb_constraints``.
        group_atol: absolute tolerance on the group inequality rows.

    Returns:
        (weights, is_valid) — ``weights`` is always finite and length ``n``.
    """
    tickers = _derive_tickers(constraints, n)
    tag = f"[{context}] " if context else ""

    def _reject(reason: str, level: int = logging.ERROR,
                w: Optional[np.ndarray] = None) -> Tuple[np.ndarray, bool]:
        fallback, source = _compute_fallback(constraints, tickers, n)
        msg = (f"{tag}solver={solver}: REJECTED solution ({reason}); "
               f"falling back to {source}.")
        _emit_diag(level, msg, context, solver, None, "rejected", False,
                   reason, source, w, constraints, tickers, n)
        return fallback, False

    if optimal_weights is None:
        return _reject("risk-budgeting solver returned no solution", logging.WARNING)
    w = np.array(optimal_weights, dtype=float).ravel()  # copy -> writable (pandas 3.0 / downstream in-place edits)
    if converged is False:
        return _reject("risk-budgeting solver reported non-convergence", logging.WARNING, w)
    ok, reason = _validate_weight_vector(
        w, constraints, n, tickers, budget_atol=budget_atol, bound_atol=bound_atol)
    if not ok:
        return _reject(reason, w=w)
    if c_rows is not None and c_lhs is not None:
        group_residuals = np.asarray(c_rows, dtype=float) @ w - np.asarray(c_lhs, dtype=float)
        breach = float(np.max(group_residuals))
        if breach > group_atol:
            j = int(np.argmax(group_residuals))
            return _reject(f"group constraint row {j} violated by {breach:.6g} "
                           f"(atol={group_atol:g})", w=w)
    _emit_diag(logging.DEBUG,
               f"{tag}solver={solver}: accepted (sum(w)={float(np.sum(w)):.6g}).",
               context, solver, None, "accepted", True,
               "", None, w, constraints, tickers, n)
    return w, True


def check_covar_conditioning(
    pd_covar: pd.DataFrame,
    cond_warn: float = 1e12,
    neg_eig_atol: float = 1e-10,
    report_worst_pair: bool = True,
    context: str = "",
    emit: bool = True,
) -> Dict[str, Any]:
    """Diagnostic-only conditioning check for a covariance matrix.

    Logs a warning when Σ is (a) indefinite beyond ``neg_eig_atol`` — i.e. a
    negative eigenvalue is leaking to a ``psd_wrap``'d solve — or (b)
    ill-conditioned beyond ``cond_warn`` (near-collinear assets, the GROWM
    root condition). Does NOT modify ``pd_covar``: it changes no result and adds
    no optimisation parameter; ``cond_warn`` / ``neg_eig_atol`` are logging
    thresholds only.

    Args:
        pd_covar: covariance matrix (square, matching index/columns).
        cond_warn: condition-number threshold above which a warning is logged.
        neg_eig_atol: a min eigenvalue below ``-neg_eig_atol`` triggers an
            indefiniteness warning.
        report_worst_pair: if True, also identify the most collinear asset pair
            (largest off-diagonal correlation) to name the likely culprit.
        context: free-form label for the log line.
        emit: if True (default), log the indefinite / ill-conditioned warnings.
            The pre-solve input contract calls this with ``emit=False`` to get
            the numbers without the WARNING, then logs them itself at INFO and
            folds them into the run-level input-contract tally.

    Returns:
        dict with ``min_eig``, ``cond``, ``rank_deficient`` (bool), and — when
        requested — ``worst_pair`` and ``worst_corr``.
    """
    tag = f"[{context}] " if context else ""
    m = pd_covar.to_numpy(dtype=float)
    m = (m + m.T) / 2.0
    eig = np.linalg.eigvalsh(m)
    min_eig, max_eig = float(eig[0]), float(eig[-1])
    cond = float(max_eig / min_eig) if min_eig > 0 else np.inf
    out = {"min_eig": min_eig, "cond": cond,
           "rank_deficient": bool(min_eig <= neg_eig_atol)}

    worst_pair = None
    worst_corr = np.nan
    if report_worst_pair and m.shape[0] > 1:
        d = np.sqrt(np.clip(np.diag(m), 1e-300, None))
        corr = m / np.outer(d, d)
        np.fill_diagonal(corr, 0.0)
        k = int(np.argmax(np.abs(corr)))
        i, j = np.unravel_index(k, corr.shape)
        worst_corr = float(corr[i, j])
        worst_pair = (str(pd_covar.index[i]), str(pd_covar.columns[j]))
        out["worst_pair"] = worst_pair
        out["worst_corr"] = worst_corr

    if emit and min_eig < -neg_eig_atol:
        logger.warning(
            f"{tag}covariance is INDEFINITE (min_eig={min_eig:.3e}); a "
            f"psd_wrap'd solve will treat a non-convex problem as convex.")
    if emit and cond > cond_warn:
        extra = (f" most collinear pair {worst_pair} corr={worst_corr:.4f}"
                 if worst_pair is not None else "")
        logger.warning(
            f"{tag}covariance is ill-conditioned (cond={cond:.3e} > "
            f"{cond_warn:.0e}, min_eig={min_eig:.3e}); optimiser may be "
            f"numerically unstable.{extra}")
    return out


class SolverRejectionSummary(logging.Handler):
    """Logging handler that tallies per-solve outcomes over a run from the
    structured ``SolverDiagnostic`` records the validators attach.

    It reads ``record.solver_diag`` rather than parsing message text, so it is
    robust to message wording. Every solve is counted (accepted solves are
    emitted at DEBUG, rejections at WARNING / ERROR), which is what makes the
    *fallback fraction* meaningful: ERROR = a result the solver called
    optimal/inaccurate but which violated the budget/box (numerical blow-up);
    WARNING = the solver reported no / infeasible / unbounded solution and we
    fell back to prior weights.

    For the accepted (DEBUG) records to reach this handler, the
    ``optimalportfolios.optimization.solver_diagnostics`` logger must be at
    DEBUG; ``configure_run_logging(attach_summary=True)`` sets that. Call
    ``.summary()`` for a one-line tally and ``.check_fallback_gate()`` to fail
    the run loud when too many rebalances fell back.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)   # DEBUG so accepted solves are tallied too
        self.records: List[SolverDiagnostic] = []

    def emit(self, record: logging.LogRecord) -> None:
        diag = getattr(record, "solver_diag", None)
        if isinstance(diag, SolverDiagnostic):
            self.records.append(diag)

    @property
    def n_total(self) -> int:
        return len(self.records)

    @property
    def n_accepted(self) -> int:
        return sum(1 for d in self.records if d.accepted)

    @property
    def n_rejected(self) -> int:
        return sum(1 for d in self.records if not d.accepted)

    @property
    def n_blowup(self) -> int:
        return sum(1 for d in self.records
                   if not d.accepted and d.severity >= logging.ERROR)

    @property
    def n_infeasible_fallback(self) -> int:
        return sum(1 for d in self.records
                   if not d.accepted and d.severity < logging.ERROR)

    @property
    def fallback_fraction(self) -> float:
        """Rejected solves / total solves (0.0 when no solves were recorded)."""
        return self.n_rejected / self.n_total if self.n_total else 0.0

    def summary(self) -> str:
        return (f"solver outcomes this run: {self.n_total} solves, "
                f"{self.n_rejected} rejected "
                f"({self.n_blowup} ERROR / numerical blow-up, "
                f"{self.n_infeasible_fallback} WARNING / infeasible-fallback); "
                f"fallback rate {self.fallback_fraction:.1%}")

    def check_fallback_gate(self, max_fraction: float = 0.05,
                            raise_on_breach: bool = False) -> bool:
        """Run-level gate on the fallback fraction.

        A run that falls back on a large minority of rebalances is partly
        held-prior and should not be treated as a clean track. When the fraction
        exceeds ``max_fraction`` this logs at ERROR (so it stands out even on a
        quiet console) and, if ``raise_on_breach``, raises ``RuntimeError`` to
        abort a production pipeline before the output is consumed.

        Returns True when within the gate (or no solves were recorded), False on
        a breach that was logged but not raised.
        """
        if self.n_total == 0:
            return True
        frac = self.fallback_fraction
        if frac > max_fraction:
            msg = (f"FALLBACK GATE breached: {self.n_rejected}/{self.n_total} solves "
                   f"fell back ({frac:.1%} > {max_fraction:.1%}); the run's output "
                   f"is partly held-prior and is not a clean track.")
            logger.error(msg)
            if raise_on_breach:
                raise RuntimeError(msg)
            return False
        logger.info("fallback gate ok: %.1f%% <= %.1f%% (%d/%d solves fell back).",
                    frac * 100.0, max_fraction * 100.0, self.n_rejected, self.n_total)
        return True


class RelaxationSummary(logging.Handler):
    """Logging handler that tallies frozen-overhang group-bound relaxations over
    a run from the structured ``RelaxationRecord`` records ``constraints.py``
    attaches (``extra={"relaxation": ...}``).

    The per-rebalance relaxation detail is emitted at INFO (so it lands in the
    file but does not flood the console); this handler aggregates it into one
    run-level line via ``.summary()`` — how many rebalances relaxed, which group
    bounds relaxed most often, the largest single relaxation, and how many
    breached the budget or the magnitude tolerance.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: List[RelaxationRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        rec = getattr(record, "relaxation", None)
        if isinstance(rec, RelaxationRecord):
            self.records.append(rec)

    @property
    def n_rebalances_relaxed(self) -> int:
        return len(self.records)

    @property
    def n_breached_budget(self) -> int:
        return sum(1 for r in self.records if r.breached_budget)

    @property
    def n_breached_tol(self) -> int:
        return sum(1 for r in self.records if r.breached_tol)

    @property
    def max_relaxation(self) -> float:
        return max((r.max_relaxation for r in self.records), default=0.0)

    def bound_frequency(self) -> Dict[Tuple[str, str], int]:
        """How many rebalances each (group, kind) bound was relaxed on."""
        freq: Dict[Tuple[str, str], int] = {}
        for r in self.records:
            for group, kind, _old, _new in r.items:
                freq[(group, kind)] = freq.get((group, kind), 0) + 1
        return freq

    def summary(self) -> str:
        if not self.records:
            return "group-bound relaxations this run: none"
        freq = self.bound_frequency()
        top = sorted(freq.items(), key=lambda kv: -kv[1])[:3]
        top_str = "; ".join(f"{g} {k} ({c})" for (g, k), c in top)
        extra = ""
        if self.n_breached_budget or self.n_breached_tol:
            extra = (f"; {self.n_breached_budget} breached budget, "
                     f"{self.n_breached_tol} breached tolerance")
        return (f"group-bound relaxations this run: {self.n_rebalances_relaxed} "
                f"rebalances; most-relaxed: {top_str}; max single relaxation "
                f"{self.max_relaxation:.4f}{extra}")


@dataclass(frozen=True)
class InputContractRecord:
    """Structured record of one pre-solve input-contract evaluation.

    Attached to the log record under ``extra={"input_contract": ...}`` so a
    handler can aggregate the per-rebalance findings into one line per category
    instead of flooding the console. ``groups`` is a tuple of (group, kind) with
    kind in {"floor_unreachable", "cap_too_low"}; ``benchmarks`` a tuple of
    (index, kind) with kind in {"cap_exceeded", "below_floor", "sum_out_of_band"};
    ``structural`` carries box-vs-budget infeasibilities; ``covar_issues`` carries
    hard covariance-integrity problems and asymmetry.
    """
    context: str
    ok: bool
    ill_conditioned: bool = False
    cond: float = float("nan")
    min_eig: float = float("nan")
    collinear_pair: Optional[Tuple[str, str]] = None
    groups: Tuple[Tuple[str, str], ...] = ()
    benchmarks: Tuple[Tuple[int, str], ...] = ()
    structural: Tuple[str, ...] = ()
    covar_issues: Tuple[str, ...] = ()


class InputContractSummary(logging.Handler):
    """Logging handler that tallies the pre-solve input contract over a run from
    the structured ``InputContractRecord`` records ``validate_solver_inputs``
    attaches.

    The per-rebalance contract detail is emitted at INFO (file, not console
    spam); this handler aggregates it into one line per category — how many
    rebalances were ill-conditioned and the most frequent collinear pair, how
    many had a benchmark outside its box (and which index), how many had an
    unreachable group bound (and which group), structural box-budget failures,
    and hard covariance-integrity failures.
    """

    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: List[InputContractRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        rec = getattr(record, "input_contract", None)
        if isinstance(rec, InputContractRecord):
            self.records.append(rec)

    @property
    def n_solves(self) -> int:
        return len(self.records)

    @property
    def n_ill_conditioned(self) -> int:
        return sum(1 for r in self.records if r.ill_conditioned)

    @property
    def n_benchmark(self) -> int:
        return sum(1 for r in self.records if r.benchmarks)

    @property
    def n_group(self) -> int:
        return sum(1 for r in self.records if r.groups)

    @property
    def n_structural(self) -> int:
        return sum(1 for r in self.records if r.structural)

    @property
    def n_covar(self) -> int:
        return sum(1 for r in self.records if r.covar_issues)

    @property
    def worst_min_eig(self) -> float:
        eigs = [r.min_eig for r in self.records if r.ill_conditioned and r.min_eig == r.min_eig]
        return min(eigs) if eigs else float("nan")

    def collinear_frequency(self) -> Dict[Tuple[str, str], int]:
        freq: Dict[Tuple[str, str], int] = {}
        for r in self.records:
            if r.collinear_pair is not None:
                key = tuple(sorted(r.collinear_pair))
                freq[key] = freq.get(key, 0) + 1
        return freq

    def group_frequency(self) -> Dict[Tuple[str, str], int]:
        freq: Dict[Tuple[str, str], int] = {}
        for r in self.records:
            for item in r.groups:
                freq[item] = freq.get(item, 0) + 1
        return freq

    def benchmark_frequency(self) -> Dict[Tuple[int, str], int]:
        freq: Dict[Tuple[int, str], int] = {}
        for r in self.records:
            for item in r.benchmarks:
                freq[item] = freq.get(item, 0) + 1
        return freq

    def summary(self) -> str:
        m = self.n_solves
        if m == 0:
            return "input contract: no solves recorded"
        lines: List[str] = []
        if self.n_ill_conditioned:
            freq = self.collinear_frequency()
            pair_str = ""
            if freq:
                (a, b), c = max(freq.items(), key=lambda kv: kv[1])
                pair_str = f"; most frequent collinear pair {a!r}-{b!r} ({c})"
            lines.append(f"  conditioning: ill-conditioned on {self.n_ill_conditioned}/{m} "
                         f"rebalances{pair_str}; worst min_eig {self.worst_min_eig:.2e}")
        if self.n_benchmark:
            top = sorted(self.benchmark_frequency().items(), key=lambda kv: -kv[1])[:2]
            top_str = "; ".join(f"index {i} {k} ({c})" for (i, k), c in top)
            lines.append(f"  benchmark: outside box on {self.n_benchmark}/{m} "
                         f"rebalances; {top_str}")
        if self.n_group:
            top = sorted(self.group_frequency().items(), key=lambda kv: -kv[1])[:3]
            top_str = "; ".join(f"{g} {k} ({c})" for (g, k), c in top)
            lines.append(f"  group reachability: unreachable on {self.n_group}/{m} "
                         f"rebalances; {top_str}")
        if self.n_structural:
            lines.append(f"  structural: box-vs-budget infeasible on "
                         f"{self.n_structural}/{m} rebalances")
        if self.n_covar:
            lines.append(f"  covariance integrity: {self.n_covar}/{m} "
                         f"rebalances had a hard covariance problem")
        if not lines:
            return f"input contract: no issues across {m} rebalances"
        return "input contract findings:\n" + "\n".join(lines)


class RunDiagnostics:
    """Bundle of the run-level diagnostic handlers returned by
    ``configure_run_logging(attach_summary=True)``.

    Exposes the rejection, relaxation and input-contract summaries together;
    ``.summary()`` combines all three and ``.check_fallback_gate(...)`` delegates
    to the rejection summary so existing call sites keep working.
    """

    def __init__(self, rejections: SolverRejectionSummary,
                 relaxations: RelaxationSummary,
                 contract: "Optional[InputContractSummary]" = None) -> None:
        self.rejections = rejections
        self.relaxations = relaxations
        self.contract = contract

    def summary(self) -> str:
        parts = [self.rejections.summary(), self.relaxations.summary()]
        if self.contract is not None:
            parts.append(self.contract.summary())
        return "\n".join(parts)

    def check_fallback_gate(self, max_fraction: float = 0.05,
                            raise_on_breach: bool = False) -> bool:
        return self.rejections.check_fallback_gate(max_fraction, raise_on_breach)


def log_environment(config_hash: Optional[str] = None) -> None:
    """Log a one-line environment banner (solver/library versions) for
    reproducibility. The 1.5e6 blow-up and the pandas-3.0 read-only crash were
    both version-specific; recording the versions makes a run bisectable.

    Emitted at INFO so it lands in the run-log file. Pass ``config_hash`` to also
    record the config that produced the run.
    """
    import importlib
    import platform

    def _ver(mod_name: str) -> str:
        try:
            return importlib.import_module(mod_name).__version__
        except Exception:
            return "n/a"

    parts = [f"python={platform.python_version()}"]
    for name in ("clarabel", "cvxpy", "numpy", "pandas", "scipy"):
        parts.append(f"{name}={_ver(name)}")
    if config_hash is not None:
        parts.append(f"config={config_hash}")
    logger.info("run environment: %s", ", ".join(parts))


def configure_run_logging(
    log_path: Optional[str] = None,
    console_level: int = logging.WARNING,
    file_level: int = logging.INFO,
    capture_warnings: bool = True,
    attach_summary: bool = False,
) -> Optional["RunDiagnostics"]:
    """Convenience logging setup for a backtest / production run.

    Call ONCE from the application entry point (not from library code). Wires a
    console handler (``console_level`` — default WARNING) and, if ``log_path`` is
    given, a file handler (``file_level`` — default INFO) that writes the same
    records, timestamped, as a persistent run log. With ``capture_warnings`` any
    remaining ``warnings.warn`` notices are routed through logging too. A
    one-line environment banner (solver/library versions) is logged at INFO for
    reproducibility. Set ``console_level=logging.ERROR`` for a quiet console with
    the detail in the file only.

    Returns a :class:`RunDiagnostics` bundle when ``attach_summary`` is True
    (else ``None``), holding the rejection and relaxation summaries. Enabling it
    lifts the solver-diagnostics logger to DEBUG (so accepted solves are counted)
    and the constraints logger to INFO (so per-rebalance relaxations are tallied)
    — neither is printed unless you lower ``console_level``. Call
    ``diagnostics.summary()`` for the combined report and
    ``diagnostics.check_fallback_gate(max_fraction=..., raise_on_breach=...)`` to
    fail loud when too large a fraction of rebalances fell back.
    """
    root = logging.getLogger()
    root.setLevel(min(console_level, file_level))
    fmt = logging.Formatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s",
                            "%Y-%m-%d %H:%M:%S")
    ch = logging.StreamHandler()
    ch.setLevel(console_level)
    ch.setFormatter(fmt)
    root.addHandler(ch)
    if log_path is not None:
        fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        fh.setLevel(file_level)
        fh.setFormatter(fmt)
        root.addHandler(fh)
    if capture_warnings:
        logging.captureWarnings(True)
    log_environment()
    diagnostics = None
    if attach_summary:
        rejections = SolverRejectionSummary()
        diag_logger = logging.getLogger(logger.name)
        # accepted solves are emitted at DEBUG; lift this logger to DEBUG so they
        # reach the summary handler. The console/file handlers keep their own
        # (higher) levels, so DEBUG records are tallied but not printed.
        diag_logger.setLevel(logging.DEBUG)
        diag_logger.addHandler(rejections)

        relaxations = RelaxationSummary()
        # per-rebalance relaxations are emitted at INFO on the constraints logger;
        # lift it to INFO so the summary tallies them (console keeps its level).
        cons_logger = logging.getLogger("optimalportfolios.optimization.constraints")
        cons_logger.setLevel(logging.INFO)
        cons_logger.addHandler(relaxations)

        # the input contract emits per-rebalance records on the solver-diagnostics
        # logger (DEBUG when clean, INFO when it finds issues); this handler tallies
        # them into one line per category. Same logger as the rejection summary;
        # each handler filters by its own record attribute.
        contract = InputContractSummary()
        diag_logger.addHandler(contract)

        diagnostics = RunDiagnostics(rejections=rejections, relaxations=relaxations,
                                     contract=contract)
    return diagnostics


# =============================================================================
# infeasibility diagnosis — turn "infeasible" into "which constraints must give,
# and by how much". Enabled via OptimiserConfig.diagnose_infeasibility (default
# True), threaded to the TAA solvers as the ``diagnose`` argument; a *rejected*
# solve then triggers a second, cheap analysis logged on the same channel:
#   * an infeasible solve runs an elastic (minimum-violation) LP and reports
#     which box / group bounds must relax, and by how much, to make that
#     rebalance solvable while holding full investment and long-only fixed;
#   * a result the solver called optimal/inaccurate but which we rejected (the
#     numerical blow-up), or a numerical_error, runs a covariance-conditioning
#     report instead — because the feasible region was non-empty.
# Costs one extra LP per rejected rebalance (not per rebalance).
# =============================================================================

_INFEASIBLE_TOKENS = ("infeasible",)


def diagnose_solver_failure(problem_status: Optional[str], constraints: Constraints,
                            covar: Optional[np.ndarray] = None,
                            solver: str = "CLARABEL", context: str = "") -> None:
    """Route a rejected solve to the right diagnostic and log the outcome.

    ``infeasible`` / ``infeasible_inaccurate`` → elastic re-solve (constraints).
    Anything else we rejected (optimal/inaccurate blow-up, numerical_error) →
    covariance conditioning, since the feasible region was non-empty.
    """
    status = str(problem_status or "").lower()
    if any(tok in status for tok in _INFEASIBLE_TOKENS):
        diagnose_infeasibility(constraints, covar=covar, solver=solver, context=context)
    elif covar is not None:
        tag = f"[{context}] " if context else ""
        idx = _diag_asset_index(constraints)
        if idx is not None and np.asarray(covar).shape[0] == len(idx):
            logger.warning("%ssolver=%s status=%s: constraints were feasible; "
                           "running covariance conditioning check.",
                           tag, solver, problem_status)
            check_covar_conditioning(pd.DataFrame(np.asarray(covar, dtype=float),
                                                  index=idx, columns=idx),
                                     context=context)


def _diag_asset_index(constraints: Constraints) -> Optional[pd.Index]:
    for fld in ("min_weights", "max_weights", "benchmark_weights"):
        s = getattr(constraints, fld, None)
        if isinstance(s, pd.Series):
            return s.index
    return None


def diagnose_infeasibility(constraints: Constraints, covar: Optional[np.ndarray] = None,
                           solver: str = "CLARABEL", context: str = "",
                           slack_tol: float = 1e-6,
                           max_report: int = 12) -> Dict[str, float]:
    """Elastic (minimum-violation) re-solve of the constraint set.

    Keeps long-only and full investment HARD and puts a non-negative slack on
    every box and group bound, minimising the total slack (a linear program —
    no Σ, so far more robust than the original QP). The non-zero slacks are the
    constraints that must give, sized by how much. If every slack is ~0 the
    constraints were satisfiable and the infeasible status was numerical, not
    structural — in that case the covariance conditioning is reported instead.

    Returns a dict {label: violation} (empty if none / not applicable). Logs a
    formatted summary on the ``logger`` channel.
    """
    import cvxpy as cvx  # lazy: only needed when diagnosis actually runs

    tag = f"[{context}] " if context else ""
    idx = _diag_asset_index(constraints)
    if idx is None:
        logger.warning("%sinfeasibility diagnosis skipped: constraints carry no "
                       "indexed bounds.", tag)
        return {}
    assets = [str(a) for a in idx]
    n = len(assets)

    nonneg = bool(getattr(constraints, "is_long_only", True))
    w = cvx.Variable(n, nonneg=nonneg)

    # HARD: full investment (so the question is "which bounds must give to keep
    # full investment", not "drop full investment"). long-only is hard via nonneg.
    max_exp = float(getattr(constraints, "max_exposure", 1.0) or 1.0)
    min_exp = float(getattr(constraints, "min_exposure", 1.0) or 1.0)
    hard = [cvx.sum(w) == max_exp] if max_exp == min_exp else \
           [cvx.sum(w) <= max_exp, cvx.sum(w) >= min_exp]

    elastic = []
    slack_vars = []
    items = []  # (kind, name, var, elem_or_None, ref_value)

    maxw = getattr(constraints, "max_weights", None)
    if isinstance(maxw, pd.Series):
        mw = maxw.reindex(idx).to_numpy(dtype=float)
        s = cvx.Variable(n, nonneg=True)
        elastic.append(w <= mw + s)
        slack_vars.append(s)
        for i in range(n):
            items.append(("box_max", assets[i], s, i, float(mw[i])))

    minw = getattr(constraints, "min_weights", None)
    if isinstance(minw, pd.Series):
        mw = minw.reindex(idx).to_numpy(dtype=float)
        s = cvx.Variable(n, nonneg=True)
        elastic.append(w >= mw - s)
        slack_vars.append(s)
        for i in range(n):
            items.append(("box_min", assets[i], s, i, float(mw[i])))

    gluc = getattr(constraints, "group_lower_upper_constraints", None)
    if gluc is not None and getattr(gluc, "group_loadings", None) is not None:
        loadings = gluc.group_loadings.reindex(index=idx).fillna(0.0)
        gmin = getattr(gluc, "group_min_allocation", None)
        gmax = getattr(gluc, "group_max_allocation", None)
        for group in loadings.columns:
            gl = loadings[group].to_numpy(dtype=float)
            if not np.any(~np.isclose(gl, 0.0)):
                continue
            if gmin is not None and group in gmin.index and not np.isnan(gmin.loc[group]):
                s = cvx.Variable(nonneg=True)
                elastic.append(gl @ w >= float(gmin.loc[group]) - s)
                slack_vars.append(s)
                items.append(("group_min", str(group), s, None, float(gmin.loc[group])))
            if gmax is not None and group in gmax.index and not np.isnan(gmax.loc[group]):
                s = cvx.Variable(nonneg=True)
                elastic.append(gl @ w <= float(gmax.loc[group]) + s)
                slack_vars.append(s)
                items.append(("group_max", str(group), s, None, float(gmax.loc[group])))

    if not slack_vars:
        logger.warning("%sinfeasibility diagnosis: no box/group bounds to test.", tag)
        return {}

    try:
        prob = cvx.Problem(cvx.Minimize(sum(cvx.sum(v) for v in slack_vars)),
                           hard + elastic)
        prob.solve(solver=solver)
    except Exception as exc:  # diagnosis must never crash the run
        logger.warning("%sinfeasibility diagnosis failed to solve: %s", tag, exc)
        return {}

    if prob.status not in ("optimal", "optimal_inaccurate") or any(
            v.value is None for v in slack_vars):
        logger.warning("%sinfeasibility diagnosis inconclusive (elastic status=%s).",
                       tag, prob.status)
        return {}

    def _val(var: Any, elem: Optional[int]) -> float:
        return float(var.value if elem is None else var.value[elem])

    violations = {}
    rows = []
    max_slack = 0.0
    for kind, name, var, elem, ref in items:
        v = _val(var, elem)
        max_slack = max(max_slack, v)
        if v <= slack_tol:
            continue
        if kind == "box_max":
            rows.append((v, f"  {name} max {ref:.4f} \u2192 {ref + v:.4f} (over {v:.4f})"))
            violations[f"box_max:{name}"] = v
        elif kind == "box_min":
            rows.append((v, f"  {name} min {ref:.4f} \u2192 {ref - v:.4f} (short {v:.4f})"))
            violations[f"box_min:{name}"] = v
        elif kind == "group_min":
            rows.append((v, f"  group '{name}' floor {ref:.4f} \u2192 {ref - v:.4f} (short {v:.4f})"))
            violations[f"group_min:{name}"] = v
        elif kind == "group_max":
            rows.append((v, f"  group '{name}' cap {ref:.4f} \u2192 {ref + v:.4f} (over {v:.4f})"))
            violations[f"group_max:{name}"] = v

    if not rows:
        logger.warning(
            "%sinfeasibility diagnosis: constraints are satisfiable (max slack "
            "%.2e \u2264 tol); the infeasible status was numerical, not "
            "structural \u2014 checking covariance conditioning.", tag, max_slack)
        if covar is not None and np.asarray(covar).shape[0] == n:
            check_covar_conditioning(pd.DataFrame(np.asarray(covar, dtype=float),
                                                  index=idx, columns=idx),
                                     context=context)
        return {}

    rows.sort(key=lambda r: -r[0])
    shown = rows[:max_report]
    total = sum(violations.values())
    body = "\n".join(line for _, line in shown)
    more = "" if len(rows) <= max_report else f"\n  (+{len(rows) - max_report} more)"
    logger.warning(
        "%sinfeasibility diagnosis \u2014 solvable only if these bounds give "
        "(minimum total violation %.4f):\n%s%s", tag, total, body, more)
    return violations


# =============================================================================
# pre-solve input contract — validate the covariance, constraint self-consistency
# and benchmark *before* the solve, so a broken/infeasible input is flagged at
# entry rather than discovered as a failed solve. Reuses check_covar_conditioning
# and (optionally) diagnose_infeasibility as proactive pre-flight checks.
# =============================================================================

@dataclass(frozen=True)
class InputContractResult:
    """Outcome of the pre-solve input contract.

    Attributes:
        ok: False iff a *hard* problem was found (non-finite / wrong-shape
            covariance) that will make the solve produce garbage; structural
            infeasibility and benchmark issues are surfaced as ``issues`` but do
            not set ``ok`` False (the solve will degrade, not crash).
        issues: human-readable findings (each also logged at its own severity).
        n_assets: problem size implied by the constraints.
        n_dropped: assets removed (NaN / zero-variance) before the solve.
    """
    ok: bool
    issues: List[str]
    n_assets: int
    n_dropped: int


def validate_solver_inputs(pd_covar: pd.DataFrame, constraints: Constraints,
                           context: str = "", n_dropped: int = 0,
                           check_conditioning: bool = True,
                           check_feasibility: bool = True,
                           deep_feasibility: bool = False,
                           solver: str = "CLARABEL",
                           symmetry_atol: float = 1e-8,
                           bound_atol: float = 1e-6) -> InputContractResult:
    """Validate solver inputs at the wrapper entry, before the solve.

    Cheap, deterministic checks (run on every solve when enabled via
    ``OptimiserConfig.validate_inputs``):

    * covariance integrity — finite, square, dimension matches the constraints,
      symmetric to ``symmetry_atol``;
    * a note when ``n_dropped`` assets were removed (NaN / zero variance);
    * conditioning (``check_conditioning``) — reuses ``check_covar_conditioning``
      so an indefinite or ill-conditioned Σ is flagged at entry, not only after
      a rejection;
    * structural feasibility (``check_feasibility``) — necessary conditions that
      the constraint set can be satisfied at all: box caps reach full investment
      (Σ max_weights ≥ budget) and floors fit within it (Σ min_weights ≤ budget),
      each group bound is reachable given the box, and the benchmark lies within
      its own box / budget.

    Optional heavyweight check:

    * ``deep_feasibility`` — runs the elastic min-violation LP
      (``diagnose_infeasibility``) as a definitive pre-flight, reporting exactly
      which bounds must give. Off by default (one extra LP per solve).

    Findings are returned in :class:`InputContractResult` and emitted as ONE
    structured :class:`InputContractRecord` per solve (covar integrity at ERROR;
    conditioning / benchmark / group-reachability folded into a single INFO line
    so an :class:`InputContractSummary` can tally them instead of flooding the
    console). Never raises and never mutates the inputs.
    """
    tag = f"[{context}] " if context else ""
    issues: List[str] = []
    groups: List[Tuple[str, str]] = []
    benchmarks: List[Tuple[int, str]] = []
    structural: List[str] = []
    covar_issues: List[str] = []
    ill_conditioned = False
    cond_v = float("nan")
    mineig_v = float("nan")
    collinear_pair: Optional[Tuple[str, str]] = None

    idx = _diag_asset_index(constraints)
    n = len(idx) if idx is not None else (pd_covar.shape[0] if pd_covar is not None else 0)

    def _record(ok: bool) -> InputContractRecord:
        return InputContractRecord(
            context=context, ok=ok, ill_conditioned=ill_conditioned,
            cond=cond_v, min_eig=mineig_v, collinear_pair=collinear_pair,
            groups=tuple(groups), benchmarks=tuple(benchmarks),
            structural=tuple(structural), covar_issues=tuple(covar_issues))

    # --- covariance integrity (hard -> ERROR + early return) ---
    if pd_covar is None:
        issues.append("covariance is None")
        covar_issues.append("none")
        logger.error("%sinput contract: covariance is None.", tag,
                     extra={"input_contract": _record(False)})
        return InputContractResult(ok=False, issues=issues, n_assets=n, n_dropped=n_dropped)

    m = pd_covar.to_numpy(dtype=float)
    hard = False
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        issues.append(f"covariance is not square (shape={m.shape})")
        covar_issues.append("not_square")
        hard = True
    if not np.all(np.isfinite(m)):
        n_bad = int(np.sum(~np.isfinite(m)))
        issues.append(f"covariance has {n_bad} non-finite entries")
        covar_issues.append("non_finite")
        hard = True
    if idx is not None and m.ndim == 2 and m.shape[0] != len(idx):
        issues.append(f"covariance dim {m.shape[0]} != n_constraints {len(idx)}")
        covar_issues.append("dim_mismatch")
        hard = True
    if (not hard) and m.shape[0] == m.shape[1] and np.all(np.isfinite(m)):
        asym = float(np.max(np.abs(m - m.T))) if m.size else 0.0
        if asym > symmetry_atol:
            issues.append(f"covariance asymmetric (max |Σ-Σ'|={asym:.2e})")
            covar_issues.append("asymmetric")

    if n_dropped > 0:
        logger.info("%sinput contract: %d asset(s) dropped (NaN / zero variance) "
                    "before solve.", tag, n_dropped)

    if hard:
        logger.error("%sinput contract (%d issue%s): %s", tag, len(issues),
                     "" if len(issues) == 1 else "s", "; ".join(issues),
                     extra={"input_contract": _record(False)})
        return InputContractResult(ok=False, issues=issues, n_assets=n, n_dropped=n_dropped)

    # --- conditioning (computed silently; folded into the contract record) ---
    if check_conditioning:
        condd = check_covar_conditioning(pd_covar, context=context, emit=False)
        cond_v = float(condd.get("cond", float("nan")))
        mineig_v = float(condd.get("min_eig", float("nan")))
        pair = condd.get("worst_pair")
        if bool(condd.get("rank_deficient")) or (cond_v == cond_v and cond_v > 1e12):
            ill_conditioned = True
            collinear_pair = tuple(pair) if pair is not None else None
            pair_str = f", pair {pair[0]!r}/{pair[1]!r}" if pair is not None else ""
            issues.append(f"covariance ill-conditioned (cond={cond_v:.2e}, "
                          f"min_eig={mineig_v:.2e}{pair_str})")

    # --- cheap structural feasibility ---
    if check_feasibility and idx is not None:
        min_exp, max_exp, _is_eq = _budget_target(constraints)
        max_w = _as_np(getattr(constraints, "max_weights", None), idx)
        min_w = _as_np(getattr(constraints, "min_weights", None), idx)
        if max_w is not None and float(np.sum(max_w)) < min_exp - bound_atol:
            msg = (f"box caps sum to {float(np.sum(max_w)):.4f} < budget "
                   f"{min_exp:.4f}: full investment is infeasible")
            issues.append(msg)
            structural.append(msg)
        if min_w is not None and float(np.sum(min_w)) > max_exp + bound_atol:
            msg = (f"box floors sum to {float(np.sum(min_w)):.4f} > budget "
                   f"{max_exp:.4f}: constraints are infeasible")
            issues.append(msg)
            structural.append(msg)

        # per-group reachability given the box
        gluc = getattr(constraints, "group_lower_upper_constraints", None)
        if gluc is not None and getattr(gluc, "group_loadings", None) is not None:
            loadings = gluc.group_loadings.reindex(index=idx).fillna(0.0)
            gmin = getattr(gluc, "group_min_allocation", None)
            gmax = getattr(gluc, "group_max_allocation", None)
            cap = max_w if max_w is not None else np.ones(len(idx))
            floor = min_w if min_w is not None else np.zeros(len(idx))
            for group in loadings.columns:
                gl = loadings[group].to_numpy(dtype=float)
                if not np.any(gl > 0):
                    continue
                reach_max = float(np.sum(gl * cap))
                reach_min = float(np.sum(gl * floor))
                if (gmin is not None and group in gmin.index
                        and not np.isnan(gmin.loc[group])
                        and float(gmin.loc[group]) > reach_max + bound_atol):
                    issues.append(f"group '{group}' floor {float(gmin.loc[group]):.4f} "
                                  f"> max reachable {reach_max:.4f} given box caps")
                    groups.append((str(group), "floor_unreachable"))
                if (gmax is not None and group in gmax.index
                        and not np.isnan(gmax.loc[group])
                        and float(gmax.loc[group]) < reach_min - bound_atol):
                    issues.append(f"group '{group}' cap {float(gmax.loc[group]):.4f} "
                                  f"< min forced {reach_min:.4f} given box floors")
                    groups.append((str(group), "cap_too_low"))

        # benchmark within its own bounds
        bench = _as_np(getattr(constraints, "benchmark_weights", None), idx)
        if bench is not None:
            if max_w is not None and float(np.max(bench - max_w)) > bound_atol:
                j = int(np.argmax(bench - max_w))
                issues.append(f"benchmark weight {bench[j]:.4f} at index {j} exceeds "
                              f"its cap {max_w[j]:.4f}")
                benchmarks.append((j, "cap_exceeded"))
            if min_w is not None and float(np.max(min_w - bench)) > bound_atol:
                j = int(np.argmax(min_w - bench))
                issues.append(f"benchmark weight {bench[j]:.4f} at index {j} below "
                              f"its floor {min_w[j]:.4f}")
                benchmarks.append((j, "below_floor"))
            b_sum = float(np.sum(bench))
            if b_sum < min_exp - bound_atol or b_sum > max_exp + bound_atol:
                issues.append(f"benchmark sums to {b_sum:.4f}, outside budget band "
                              f"[{min_exp:.4f}, {max_exp:.4f}]")
                benchmarks.append((-1, "sum_out_of_band"))

    # --- definitive elastic pre-flight (opt-in; diagnose_infeasibility logs WARNING) ---
    if deep_feasibility:
        covar_np = pd_covar.to_numpy(dtype=float)
        violations = diagnose_infeasibility(constraints, covar=covar_np,
                                            solver=solver, context=context)
        if violations:
            issues.append(f"elastic pre-flight: {len(violations)} bound(s) must relax")

    # --- one consolidated emission: INFO when issues found, DEBUG when clean ---
    if issues:
        logger.info("%sinput contract (%d issue%s): %s", tag, len(issues),
                    "" if len(issues) == 1 else "s", "; ".join(issues),
                    extra={"input_contract": _record(True)})
    else:
        logger.debug("%sinput contract: ok", tag,
                     extra={"input_contract": _record(True)})
    return InputContractResult(ok=True, issues=issues, n_assets=n, n_dropped=n_dropped)
