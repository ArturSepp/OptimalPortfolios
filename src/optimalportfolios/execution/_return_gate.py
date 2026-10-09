"""Opt-in return selection from a fixed branch search under a common audit.

The caller supplies its original-domain weight audit. Search may explore states
that fail that audit, but this gate never returns them. The gate adds no
projections. Separate, explicitly activated precision recovery can add one
bounded projection without changing the original-baseline return guards.
"""
from collections.abc import Callable
from dataclasses import dataclass, replace

import cvxpy as cvx
import numpy as np
import pandas as pd

from optimalportfolios.covar_estimation.risk_model_adapter import build_risk_model
from optimalportfolios.execution import schema as s, solver as execution_solver
from optimalportfolios.execution.branches import ExecutionBranchConfig, ExecutionBranchSearchResult
from optimalportfolios.execution.guarded import _metrics, _preference, _qualification
from optimalportfolios.execution.solver import ExecutionOptimizationResult, RELAXED_CORRIDOR_SOLVE
from optimalportfolios.execution.types import ResolvedExecutionProblem
from optimalportfolios.optimization.constraints import cvx_covar_variance
from optimalportfolios.optimization.covar_factorization import factorize_covariance
from optimalportfolios.optimization.solver_diagnostics import validate_solution


@dataclass(frozen=True)
class _AuditedExecutionReturn:
    """Optional valid ``result``, per-checkpoint ``audit`` and return ``summary``.

    ``result=None`` explicitly means there is no audited eligible incumbent;
    callers must not substitute the rejected baseline or unfiltered result.
    """

    result: ExecutionOptimizationResult | None
    audit: pd.DataFrame
    summary: dict


def _gate_branched_return(
    problem: ResolvedExecutionProblem,
    search: ExecutionBranchSearchResult,
    controls: ExecutionBranchConfig,
    accept_weights: Callable[[pd.Series], bool],
) -> _AuditedExecutionReturn:
    """Choose the best common-audited checkpoint without resetting return guards.

    The original baseline anchors risk, counts and turnover even if it fails
    the common audit. A valid alternative must still qualify against it.
    The baseline is a fallback only when its own raw weights pass. Exact ties
    retain the earlier checkpoint, with baseline considered first. The audit
    receives a detached weight vector and must return a boolean. Exceptions
    propagate rather than masquerading as an infeasible portfolio.
    """
    baseline = search.checkpoints['baseline']
    rows = []
    winner = None
    selected = None
    best = None
    if not (baseline.accepted and baseline.compliant):
        return _AuditedExecutionReturn(None, pd.DataFrame(), dict(
            accepted=False, stop_reason='baseline_not_accepted', selected_checkpoint=None))
    risk = build_risk_model({pd.Timestamp('2000-01-01'): problem.covariance})
    reference = _metrics(problem, baseline, risk, controls.ticket_size_bp)
    checkpoints = [('baseline', baseline)]+[
        (name, value) for name, value in search.checkpoints.items() if name != 'baseline']
    for name, candidate in checkpoints:
        row = dict(checkpoint=name, common_audit_passed=False, eligible=False,
                   retained=False, reason='candidate_not_accepted')
        if candidate.accepted and candidate.compliant:
            if candidate.trade_table[RELAXED_CORRIDOR_SOLVE].any():
                row['reason'] = 'candidate_relaxed_corridors'
            else:
                values = _metrics(problem, candidate, risk, controls.ticket_size_bp)
                passed = accept_weights(candidate.weights.copy(deep=True))
                if not isinstance(passed, (bool, np.bool_)):
                    raise TypeError('accept_weights must return a boolean')
                reason = ('baseline' if name == 'baseline' else
                          _qualification(reference, values, controls))
                eligible = bool(passed and reason in (
                    'baseline', 'tracking_improvement', 'ticket_saving_within_allowance'))
                row.update(values, common_audit_passed=bool(passed), eligible=eligible,
                           reason=reason if passed else 'common_audit_failed')
                if eligible and (best is None or _preference(values) < _preference(best)):
                    winner, best, selected = candidate, values, name
                    row['retained'] = True
        rows.append(row)
    return _AuditedExecutionReturn(winner, pd.DataFrame(rows), dict(
        accepted=winner is not None, baseline=reference, final=best,
        selected_checkpoint=selected,
        unfiltered_checkpoint=search.summary.get('selected_checkpoint'),
        changed_return=selected != search.summary.get('selected_checkpoint'),
        audited_checkpoints=len(rows),
        common_audit_rejections=sum(not row['common_audit_passed'] for row in rows),
        stop_reason='audited_return' if winner is not None else 'no_common_audited_incumbent'))


def _precision_trade_table(table, weights):
    """Refresh weight-dependent execution diagnostics after an audited correction.

    The native solver has no result-enrichment API independent of its solve.
    Keep its lifecycle/selection flags and recompute its weight/funding columns.
    """
    out = table.copy(deep=True)
    cash = out[s.SETTLEMENT_CASH]
    mandatory = out[s.MANDATORY_TRADE]
    out[execution_solver.PROPOSED_WEIGHT] = weights
    out[execution_solver.EXECUTED_TRADE] = weights-out[s.CURRENT_WEIGHT]
    out[execution_solver.MANDATORY_EXECUTED_TRADE] = out[s.BASE_WEIGHT]-out[s.CURRENT_WEIGHT]
    out[execution_solver.DISCRETIONARY_EXECUTED_TRADE] = weights-out[s.BASE_WEIGHT]
    out.loc[mandatory, execution_solver.MANDATORY_EXECUTED_TRADE] = (
        weights-out[s.CURRENT_WEIGHT]).loc[mandatory]
    out.loc[mandatory, execution_solver.DISCRETIONARY_EXECUTED_TRADE] = 0.
    out[execution_solver.TARGET_GAP] = weights-out[s.EFFECTIVE_MODEL_WEIGHT]
    corridor = out[execution_solver.MANDATORY_CORRIDOR]
    out[execution_solver.MANDATORY_CORRIDOR_EXTRA_TRADE] = (
        weights-out[s.BASE_WEIGHT]).where(corridor, 0.)
    out[execution_solver.CORRIDOR_RELAXATION] = (
        (out[execution_solver.STRICT_CORRIDOR_MIN]-weights).clip(lower=0.)
        +(weights-out[execution_solver.STRICT_CORRIDOR_MAX]).clip(lower=0.)
    ).where(out[s.SELECTED_TRADE], 0.)
    sell_down = out.get(s.CUTOFF_SELL_DOWN, pd.Series(False, index=out.index))
    out[execution_solver.CUTOFF_SELL_DOWN_RETENTION] = weights.where(sell_down)
    out[execution_solver.CUTOFF_SELL_DOWN_RETAINED_FRACTION] = weights.div(
        out[s.CURRENT_WEIGHT].replace(0., np.nan)).where(sell_down)
    funding = float((weights-out[s.BASE_WEIGHT]).loc[~cash].sum())
    out[execution_solver.REWEIGHT_FUNDING] = 0.
    out.loc[cash, execution_solver.REWEIGHT_FUNDING] = funding
    out[execution_solver.EXPECTED_CASH_WEIGHT] = np.nan
    out.loc[cash, execution_solver.EXPECTED_CASH_WEIGHT] = float(
        out.loc[cash, s.CURRENT_WEIGHT].iloc[0]
        -out.loc[~cash, s.MANDATORY_FUNDING].sum()-funding)
    out['precision_recovery_applied'] = True
    return out


# wrapper_minimise_tracking_error cannot express a local correction trust region,
# frozen ticket support and noncash turnover together. Reuse OP's constraint
# compiler, covariance atom and original-constraint audit for this auxiliary QP.
def _recover_precision_return(problem, baseline, controls, accept_weights):
    """Try one bounded correction, keeping the common audit and original guards.

    Activate only when no ordinary audited return exists. Movable corridors,
    group and budget inequalities receive 8e-8 room, strictly inside the existing
    1e-7 audit tolerance. Exact lifecycle pins and hard instrument bounds remain
    fixed. Each weight moves by at most 1e-5 NAV (0.1 bp). No new registered or
    sized ticket is admitted. Acceptance still requires the caller's unchanged
    audit and the original baseline's complete improvement qualification.
    This is tolerance-feasibility recovery, not an exact-arithmetic certificate.
    """
    summary = dict(accepted=False, projection_calls=0, audit_tolerance=1e-7,
                   internal_residual_allowance=8e-8, max_weight_correction=1e-5)

    def finish(result, reason, residuals=None):
        """Keep rejected solves explicit and attach the acceptance reason."""
        summary.update(accepted=result is not None, stop_reason=reason)
        return _AuditedExecutionReturn(
            result, pd.DataFrame() if residuals is None else residuals, summary)

    if not (baseline.accepted and baseline.compliant):
        return finish(None, 'baseline_not_accepted')
    if baseline.trade_table[RELAXED_CORRIDOR_SOLVE].any():
        return finish(None, 'baseline_relaxed_corridors')
    passed = accept_weights(baseline.weights.copy(deep=True))
    if not isinstance(passed, (bool, np.bool_)):
        raise TypeError('accept_weights must return a boolean')
    if passed:
        return finish(baseline, 'already_audited')
    table = baseline.trade_table
    original = execution_solver.build_selected_execution_constraints(
        problem.constraints, table, partition_groups=problem.partition_groups).copy(
            group_lower_upper_constraints=problem.constraints.group_lower_upper_constraints,
            benchmark_weights=problem.target[s.RAW_MODEL_WEIGHT])
    hard_lo, hard_hi = execution_solver._build_selected_execution_bound_series(
        problem.constraints, table, True)
    lo, hi = original.min_weights.copy(), original.max_weights.copy()
    current, initial = table[s.CURRENT_WEIGHT], baseline.weights
    cash = table[s.SETTLEMENT_CASH]
    changes = (initial-current).abs()
    movable = hi.gt(lo) & (changes.gt(s.WEIGHT_ZERO_TOLERANCE) | cash)
    pad, step = summary['internal_residual_allowance'], summary['max_weight_correction']
    lo.loc[movable] = np.maximum(lo-pad, hard_lo).loc[movable]
    hi.loc[movable] = np.minimum(hi+pad, hard_hi).loc[movable]
    lo, hi = np.maximum(lo, initial-step), np.minimum(hi, initial+step)
    stationary = ~cash & changes.le(s.WEIGHT_ZERO_TOLERANCE)
    lo.loc[stationary] = initial.loc[stationary]
    hi.loc[stationary] = initial.loc[stationary]
    unsized = ~cash & changes.le(controls.ticket_size_bp/1e4)
    lo.loc[unsized] = np.maximum(lo, current-controls.ticket_size_bp/1e4).loc[unsized]
    hi.loc[unsized] = np.minimum(hi, current+controls.ticket_size_bp/1e4).loc[unsized]
    if lo.gt(hi).any():
        return finish(None, 'empty_correction_interval')
    groups = original.group_lower_upper_constraints
    if groups is not None:
        groups = replace(groups,
            group_min_allocation=None if groups.group_min_allocation is None
            else groups.group_min_allocation-pad,
            group_max_allocation=None if groups.group_max_allocation is None
            else groups.group_max_allocation+pad)
    numerical = original.copy(min_weights=lo, max_weights=hi,
        group_lower_upper_constraints=groups, min_exposure=original.min_exposure-pad,
        max_exposure=original.max_exposure+pad)
    factor = factorize_covariance(problem.covariance.to_numpy())
    risk = build_risk_model({pd.Timestamp('2000-01-01'): problem.covariance})
    reference = _metrics(problem, baseline, risk, controls.ticket_size_bp)
    summary['baseline'] = reference
    delta = cvx.Variable(len(initial))
    w = initial.to_numpy()+delta/1e5
    restrictions = numerical.set_cvx_all_constraints(
        w=w, covar=factor.covar, covar_factorization=factor)
    noncash = np.flatnonzero(~cash)
    cash_position = int(np.flatnonzero(cash)[0])
    restrictions += [w[cash_position] == float(current.loc[cash].iloc[0]
        -table.loc[~cash, s.MANDATORY_FUNDING].sum()
        +table.loc[~cash, s.BASE_WEIGHT].sum())-cvx.sum(w[noncash])]
    restrictions += [cvx.norm1(w[noncash]-current.iloc[noncash].to_numpy())
                     <= reference['gross_turnover_bp']/1e4-1e-10]
    variance = cvx_covar_variance(w-table[s.RAW_MODEL_WEIGHT].to_numpy(),
                                  factor.covar, covar_factorization=factor)
    program = cvx.Problem(cvx.Minimize(1e6*variance), restrictions)
    summary['projection_calls'] = 1
    try:
        program.solve(solver='CLARABEL', tol_feas=1e-11, tol_gap_abs=1e-11,
                      tol_gap_rel=1e-11, max_iter=500, warm_start=False)
    except cvx.error.SolverError:
        return finish(None, 'solver_error')
    summary['solver_status'] = str(program.status)
    if program.status != cvx.OPTIMAL or w.value is None:
        return finish(None, 'no_optimal_correction')
    weights = pd.Series(np.asarray(w.value).ravel(), index=initial.index, name=initial.name)
    pins = lo.eq(hi)
    weights.loc[pins] = lo.loc[pins]
    weights.loc[cash] = float(current.loc[cash].iloc[0]
        -table.loc[~cash, s.MANDATORY_FUNDING].sum()
        -(weights-table[s.BASE_WEIGHT]).loc[~cash].sum())
    summary['actual_max_weight_correction'] = float((weights-initial).abs().max())
    if summary['actual_max_weight_correction'] > step+1e-10:
        return finish(None, 'correction_too_large')
    outcome = validate_solution(weights.to_numpy(), program.status, original, len(weights),
        solver='CLARABEL', context=problem.context+' precision recovery',
        budget_atol=1e-7, bound_atol=1e-7, constraint_atol=1e-7,
        covar=factor.covar, covar_factorization=factor)
    if not (outcome.accepted and outcome.compliant):
        return finish(None, 'original_constraint_audit_failed', outcome.residuals_frame())
    passed = accept_weights(weights.copy(deep=True))
    if not isinstance(passed, (bool, np.bool_)):
        raise TypeError('accept_weights must return a boolean')
    if not passed:
        return finish(None, 'common_audit_failed', outcome.residuals_frame())
    result = ExecutionOptimizationResult(weights, _precision_trade_table(table, weights), outcome)
    values = _metrics(problem, result, risk, controls.ticket_size_bp)
    summary['final'] = values
    reason = _qualification(reference, values, controls)
    if reason not in ('tracking_improvement', 'ticket_saving_within_allowance'):
        return finish(None, reason, outcome.residuals_frame())
    return finish(result, reason, outcome.residuals_frame())
