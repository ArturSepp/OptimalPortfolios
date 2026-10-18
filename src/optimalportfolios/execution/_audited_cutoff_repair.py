"""Opt-in support-consistent numerical repair with original hard portfolio bounds."""
from dataclasses import replace

import cvxpy as cvx
import numpy as np
import pandas as pd

from optimalportfolios.execution import schema as s
from optimalportfolios.execution.api import solve_ranked_execution
from optimalportfolios.execution._constraint_scope import _original_group_bounds_scope
from optimalportfolios.execution._cutoff_repair import (
    _CutoffRepairFailure, _CutoffRepairResult, _repair_domain,
)
from optimalportfolios.execution._return_gate import _precision_trade_table
from optimalportfolios.execution.solver import (
    ExecutionOptimizationResult, ExecutionSolverInfeasibility,
    _build_selected_execution_bound_series, _get_mandatory_corridor_masks,
    build_selected_execution_constraints,
)
from optimalportfolios.optimization.constraints import cvx_covar_variance
from optimalportfolios.optimization.constraints.analytics import evaluate_constraint_residuals
from optimalportfolios.optimization.covar_factorization import factorize_covariance
from optimalportfolios.optimization.solver_diagnostics import OptimizationOutcome, validate_solution


def _provided_endpoint_baseline(problem, original_error):
    """Audit an existing endpoint without claiming a solver optimum or exact feasibility.

    A pinned entry can miss a group floor by less than the common 1e-7 audit,
    making the exact convex model infeasible. The original resolved endpoint
    may nevertheless meet the declared audit. Preserve its pins, snap omitted
    optional dust to base and fund cash exactly; never normalize the book.
    """
    table = problem.target.copy(deep=True)
    optional = (table[s.RULE4_TRADE] & table[s.REBALANCE_CADENCE_ELIGIBLE]
                & ~table[s.MANDATORY_TRADE] & ~table[s.SETTLEMENT_CASH])
    desired = table[s.EFFECTIVE_MODEL_WEIGHT]-table[s.BASE_WEIGHT]
    table[s.SELECTED_TRADE] = optional & desired.abs().gt(s.WEIGHT_ZERO_TOLERANCE)
    table[s.FEASIBILITY_RESCUE_TRADE] = table[s.SELECTED_TRADE]
    with _original_group_bounds_scope():
        hard = build_selected_execution_constraints(problem.constraints, table,
            partition_groups=problem.partition_groups)
    weights = table[s.EFFECTIVE_MODEL_WEIGHT].copy()
    pins = hard.min_weights.eq(hard.max_weights)
    weights.loc[pins] = hard.min_weights.loc[pins]
    cash = table[s.SETTLEMENT_CASH]
    weights.loc[cash] = float(table[s.CURRENT_WEIGHT].sum()-weights.loc[~cash].sum())
    if not np.isfinite(weights).all():
        raise _CutoffRepairFailure('provided_endpoint', 'nonfinite weights')
    factor = factorize_covariance(problem.covariance.to_numpy())
    residuals = evaluate_constraint_residuals(weights.to_numpy(), hard,
        covar=factor.covar, covar_factorization=factor, tolerance=1e-7)
    if not all(r.passed for r in residuals if r.hard):
        raise _CutoffRepairFailure('provided_endpoint', 'original hard audit failed')
    outcome = OptimizationOutcome(weights=weights.to_numpy().copy(), accepted=True,
        solver='NONE', status='audited_feasible_endpoint', context=problem.context,
        reason='Provided endpoint passed original audit; no optimality or exact-arithmetic claim',
        fallback_source='effective_model_endpoint', constraint_residuals=tuple(residuals),
        covar_factorization=factor, constraints=hard)
    buy, sell = _get_mandatory_corridor_masks(table)
    table['mandatory_corridor'] = buy | sell
    table['strict_corridor_min'], table['strict_corridor_max'] = hard.min_weights, hard.max_weights
    table['relaxed_corridor_solve'] = False
    enriched = _precision_trade_table(table, weights)
    enriched['precision_recovery_applied'] = False
    enriched['audited_provided_endpoint'] = True
    result = ExecutionOptimizationResult(weights, enriched, outcome)
    detail = pd.DataFrame(dict(repair_eligible=optional,
        original_effective_target=problem.target[s.EFFECTIVE_MODEL_WEIGHT],
        original_corridor_min=hard.min_weights, original_corridor_max=hard.max_weights,
        repaired_endpoint=weights, endpoint_widening=0.,
        executed_weight=weights, actual_widening=0.))
    return _CutoffRepairResult(result, problem, weights, detail, dict(
        minimum_widening=None, minimum_scope='provided endpoint; no optimisation claim',
        endpoint_widening=0., actual_widening=0., numerical_endpoint_retry=True,
        additional_endpoint_projection_calls=1, feasible_endpoint_baseline=True,
        provided_endpoint=True, original_failure=str(original_error),
        raw_model_unchanged=True, hard_bounds_relaxed=False, excluded_entries_reopened=False,
        group_bounds='original_supplied', audit_tolerance=1e-7))


def _endpoint_solve(program, stage, linear=False, fallback_solver=None):
    """Use scaled LP/QP solves, accepting only an optimal status."""
    try:
        if linear:
            program.solve(solver='SCIPY', warm_start=False, scipy_options={
                'method': 'highs', 'primal_feasibility_tolerance': 1e-9,
                'dual_feasibility_tolerance': 1e-9})
            solver = 'SCIPY/HiGHS'
        else:
            program.solve(solver='CLARABEL', warm_start=False, tol_feas=1e-10,
                          tol_gap_abs=1e-10, tol_gap_rel=1e-10, max_iter=1000)
            solver = 'CLARABEL'
    except cvx.error.SolverError as error:
        # A hard volatility constraint makes phase one conic rather than linear.
        if linear and not program.is_qp():
            return _endpoint_solve(program, stage, fallback_solver=fallback_solver)
        if not linear and fallback_solver == 'MOSEK':
            try:
                program.solve(solver='MOSEK', warm_start=False, mosek_params={
                    'MSK_IPAR_NUM_THREADS': 1, 'MSK_DPAR_INTPNT_CO_TOL_PFEAS': 1e-10,
                    'MSK_DPAR_INTPNT_CO_TOL_DFEAS': 1e-10,
                    'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': 1e-10})
                solver = 'MOSEK'
            except cvx.error.SolverError as retry_error:
                raise _CutoffRepairFailure(stage, 'both_endpoint_solvers_failed') from retry_error
        else:
            raise _CutoffRepairFailure(stage, 'solver_error') from error
    if program.status != cvx.OPTIMAL:
        raise _CutoffRepairFailure(stage, str(program.status))
    return solver


# wrapper_minimise_tracking_error cannot express the joint minimum corridor-
# widening budget or admission freezes. Reuse OP's compiler, covariance atom,
# lifecycle bounds and auditor; the ordinary ranking/sizing stays in OP.
def _repair_audited_cutoff_target(problem, original_error, widening_atol=1e-8):
    """Retry a failed cutoff repair on an endpoint its admitted trades can reach.

    Phase one retains the original aggregate-widening objective. Phase two uses
    the same extra 1e-8 allowance, with basis-point variables. Before sizing,
    optional moves at or below the unchanged zero threshold are frozen and the
    endpoint is resolved. No product/group bound, lifecycle pin, exclusion,
    covariance, raw-model benchmark or ticket threshold is relaxed.
    """
    if not np.isfinite(widening_atol) or not 0 <= widening_atol <= 1e-6:
        raise ValueError('widening_atol must be finite and between zero and 1e-6')
    table = problem.target.copy(deep=True)
    if s.MATERIAL_TRADE_THRESHOLD not in table:
        raise ValueError('cutoff repair requires the resolved material_trade_threshold')
    threshold = pd.to_numeric(table[s.MATERIAL_TRADE_THRESHOLD], errors='raise')
    if not np.isfinite(threshold).all() or threshold.lt(0).any():
        raise ValueError('invalid material trade threshold')
    domain = _repair_domain(table)
    if not domain.any():
        raise _CutoffRepairFailure('audited_domain', 'no eligible repair coordinates')
    opened = table.copy(deep=True)
    opened[s.SELECTED_TRADE] = domain
    opened[s.FEASIBILITY_RESCUE_TRADE] = False
    strict_lo, strict_hi = _build_selected_execution_bound_series(
        problem.constraints, opened, False)
    with _original_group_bounds_scope():
        hard = build_selected_execution_constraints(problem.constraints, opened,
            relax_selected_corridors=True, partition_groups=problem.partition_groups)
    hard = hard.copy(
        group_lower_upper_constraints=problem.constraints.group_lower_upper_constraints,
        benchmark_weights=table[s.RAW_MODEL_WEIGHT])
    factor = factorize_covariance(problem.covariance.to_numpy())
    x = cvx.Variable(len(table), name='audited_endpoint_bp')
    w = x / 1e4
    rows = hard.set_cvx_all_constraints(w=w, covar=factor.covar,
                                      covar_factorization=factor)
    positions = np.flatnonzero(domain.to_numpy())
    excursion_bp = cvx.sum(cvx.pos(1e4*strict_lo.iloc[positions].to_numpy()-x[positions])
                          + cvx.pos(x[positions]-1e4*strict_hi.iloc[positions].to_numpy()))
    first = cvx.Problem(cvx.Minimize(excursion_bp), rows)
    try:
        solvers = [_endpoint_solve(first, 'audited_minimum_widening', linear=True,
                                  fallback_solver=problem.optimiser_config.solver)]
    except _CutoffRepairFailure as error:
        try:
            return _provided_endpoint_baseline(problem, error)
        except (ExecutionSolverInfeasibility, _CutoffRepairFailure):
            raise error
    minimum = max(float(first.value)/1e4, 0.)
    original_minimum = minimum
    minimum_history = [minimum]
    budget = minimum+widening_atol
    variance = cvx_covar_variance(1e3*(w-table[s.RAW_MODEL_WEIGHT].to_numpy()),
                                factor.covar, covar_factorization=factor)
    pins = hard.min_weights.eq(hard.max_weights)
    witness = pd.Series(np.asarray(w.value).ravel(), index=table.index)
    witness.loc[pins] = hard.min_weights.loc[pins]
    witness_move = witness-table[s.BASE_WEIGHT]
    locked = pd.Series(False, index=table.index)
    activated = pd.Series(0., index=table.index)
    refinements = 0

    def qualify(weights, label, bounds=hard):
        """Re-audit original rules and the minimum-widening budget."""
        outcome = validate_solution(weights.to_numpy(), 'optimal', bounds, len(table),
            solver='CLARABEL', context=problem.context+' audited cutoff '+label,
            budget_atol=1e-7, bound_atol=1e-7, constraint_atol=1e-7,
            covar=factor.covar, covar_factorization=factor)
        if not (outcome.accepted and outcome.compliant):
            raise _CutoffRepairFailure('audited_'+label, outcome.reason)
        widening = ((strict_lo-weights).clip(lower=0.)
                    +(weights-strict_hi).clip(lower=0.)).where(domain, 0.)
        if float(widening.sum()) > budget+1e-9:
            raise _CutoffRepairFailure('audited_'+label, 'widening budget exceeded')
        return outcome, widening

    while True:
        frozen = np.flatnonzero(locked.to_numpy())
        support_rows = list(rows)
        if len(frozen):
            support_rows += [x[frozen] == 1e4*table[s.BASE_WEIGHT].iloc[frozen].to_numpy()]
        for direction in (-1., 1.):
            active = np.flatnonzero((direction*activated).gt(0.).to_numpy())
            if len(active):
                support_rows += [
                    direction*(x[active]-1e4*table[s.BASE_WEIGHT].iloc[active].to_numpy())
                    >= 1e4*activated.iloc[active].abs().to_numpy()]
        if refinements:
            # The all-coordinate minimum can rely on dust that admission omits.
            # Recompute the minimum on the actual support, recording its entire
            # extra cost; only the same 1e-8 lexicographic allowance is added.
            admitted_first = cvx.Problem(cvx.Minimize(excursion_bp), support_rows)
            solvers.append(_endpoint_solve(admitted_first, 'audited_admitted_minimum', linear=True,
                                          fallback_solver=problem.optimiser_config.solver))
            minimum = max(float(admitted_first.value)/1e4, 0.)
            minimum_history.append(minimum)
            budget = minimum+widening_atol
            witness = pd.Series(np.asarray(w.value).ravel(), index=table.index)
            witness.loc[pins] = hard.min_weights.loc[pins]
            witness_move = witness-table[s.BASE_WEIGHT]
        second = cvx.Problem(cvx.Minimize(variance), support_rows+[excursion_bp <= 1e4*budget])
        solvers.append(_endpoint_solve(second, 'audited_admission_endpoint',
                                      fallback_solver=problem.optimiser_config.solver))
        endpoint = pd.Series(np.asarray(w.value).ravel(), index=table.index)
        endpoint.loc[pins] = hard.min_weights.loc[pins]
        endpoint.loc[locked] = table.loc[locked, s.BASE_WEIGHT]
        _, endpoint_widening = qualify(endpoint, 'endpoint')
        micro = domain & ~locked & (endpoint-table[s.BASE_WEIGHT]).abs().le(s.WEIGHT_ZERO_TOLERANCE)
        if not micro.any():
            break
        if refinements >= 8:
            raise _CutoffRepairFailure('audited_admission_endpoint', 'refinement limit')
        # Freezing every tiny move can destroy a feasible buying/funding set.
        # Preserve the sparse LP witness's admitted directions, concentrating
        # the correction into real tickets instead of relying on discarded dust.
        protect = micro & witness_move.abs().gt(1.01*s.WEIGHT_ZERO_TOLERANCE)
        activated.loc[protect] = np.sign(witness_move.loc[protect])*np.minimum(
            witness_move.loc[protect].abs(), 2*s.WEIGHT_ZERO_TOLERANCE)
        locked |= micro & ~protect
        refinements += 1

    target = table.copy(deep=True)
    target[s.EFFECTIVE_MODEL_WEIGHT] = endpoint
    desired = endpoint-target[s.BASE_WEIGHT]
    target[s.DESIRED_REWEIGHT] = desired
    target[s.RULE4_TRADE] = domain
    target[s.MATERIAL_TRADE] = desired.abs().ge(threshold)
    target[s.TRADE_CANDIDATE] = (
        domain & target[s.MATERIAL_TRADE] & desired.abs().gt(s.WEIGHT_ZERO_TOLERANCE))
    revised = replace(problem, target=target, allow_corridor_relaxation=False,
                      context=problem.context+' audited cutoff endpoint')
    fallback = False
    native_failure = None
    with _original_group_bounds_scope():
        try:
            result = solve_ranked_execution(revised)
            if not (result.accepted and result.compliant):
                raise _CutoffRepairFailure('audited_ranked', result.outcome.reason)
            qualify(result.weights, 'ranked')
        except (ExecutionSolverInfeasibility, _CutoffRepairFailure) as error:
            # The qualified endpoint is an explicit feasibility baseline. It is
            # never silently substituted for an invalid search improvement.
            native_failure = str(error)
            fallback = True
            admitted = target.copy(deep=True)
            admitted[s.SELECTED_TRADE] = domain & desired.abs().gt(s.WEIGHT_ZERO_TOLERANCE)
            admitted[s.FEASIBILITY_RESCUE_TRADE] = admitted[s.SELECTED_TRADE]
            lo, hi = _build_selected_execution_bound_series(problem.constraints, admitted, False)
            exact = hard.copy(min_weights=lo, max_weights=hi)
            outcome, _ = qualify(endpoint, 'fallback', exact)
            buy, sell = _get_mandatory_corridor_masks(admitted)
            admitted['mandatory_corridor'] = buy | sell
            admitted['strict_corridor_min'], admitted['strict_corridor_max'] = lo, hi
            admitted['relaxed_corridor_solve'] = False
            enriched = _precision_trade_table(admitted, endpoint)
            enriched['precision_recovery_applied'] = False
            result = ExecutionOptimizationResult(endpoint, enriched, outcome)
    _, actual = qualify(result.weights, 'final')
    cash = table[s.SETTLEMENT_CASH]
    funding_error = abs(float(result.weights.loc[cash].iloc[0])
                        -float(table[s.CURRENT_WEIGHT].sum()-result.weights.loc[~cash].sum()))
    if funding_error > 1e-7:
        raise _CutoffRepairFailure('audited_final', 'cash funding identity')
    audit = pd.DataFrame(dict(
        repair_eligible=domain, original_effective_target=table[s.EFFECTIVE_MODEL_WEIGHT],
        original_corridor_min=strict_lo, original_corridor_max=strict_hi,
        repaired_endpoint=endpoint, endpoint_widening=endpoint_widening,
        executed_weight=result.weights, actual_widening=actual, admission_frozen=locked))
    enriched = result.trade_table.copy(deep=True)
    for name in audit:
        enriched['cutoff_repair_'+name] = audit[name]
    enriched['cutoff_repair_applied'] = True
    result = replace(result, trade_table=enriched)
    summary = dict(minimum_widening=original_minimum, admitted_minimum_widening=minimum,
        minimum_widening_history=minimum_history,
        support_widening_charge=max(minimum-original_minimum, 0.), widening_budget=budget,
        minimum_scope='minimum on final admitted support; all-coordinate lower bound retained',
        endpoint_widening=float(endpoint_widening.sum()), actual_widening=float(actual.sum()),
        numerical_endpoint_retry=True, endpoint_solver=solvers,
        endpoint_variable_scale=1e4, additional_endpoint_projection_calls=len(solvers),
        admission_refinements=refinements, microscopic_moves_frozen=int(locked.sum()),
        lp_witness_admissions=int(activated.ne(0.).sum()),
        feasible_endpoint_baseline=fallback, native_failure=native_failure,
        original_failure=str(original_error), raw_model_unchanged=True,
        hard_bounds_relaxed=False, excluded_entries_reopened=False,
        group_bounds='original_supplied', phase_one_status=first.status,
        phase_two_status=second.status)
    return _CutoffRepairResult(result, revised, endpoint, audit, summary)
