"""Opt-in repair of infeasible cutoff targets without reopening excluded entries.

This experimental fallback changes the execution corridor domain. It does not
change production dispatch, the raw model risk benchmark, resolved lifecycle
instructions, cadence, or hard portfolio bounds. Call only after the ordinary
consumer lifecycle and its same-date fallback have failed.
"""
from dataclasses import dataclass, replace

import cvxpy as cvx
import numpy as np
import pandas as pd

from optimalportfolios.execution import schema as s
from optimalportfolios.execution.api import solve_ranked_execution
from optimalportfolios.execution.solver import (
    ExecutionOptimizationResult,
    _build_selected_execution_bound_series,
    build_selected_execution_constraints,
)
from optimalportfolios.execution.types import ResolvedExecutionProblem
from optimalportfolios.optimization.constraints import cvx_covar_variance
from optimalportfolios.optimization.covar_factorization import factorize_covariance
from optimalportfolios.optimization.solver_diagnostics import validate_solution


@dataclass(frozen=True)
class _CutoffRepairResult:
    """Return execution, the revised problem, endpoint and domain-change audit."""

    result: ExecutionOptimizationResult
    revised_problem: ResolvedExecutionProblem
    endpoint: pd.Series
    audit: pd.DataFrame
    summary: dict


class _CutoffRepairFailure(ValueError):
    """Preserve an unsuccessful stage without authorizing any fallback weights."""

    def __init__(self, stage: str, status: str):
        """Identify whether broad-domain feasibility, optimization or auditing failed."""
        super().__init__(f'{stage}: {status}')
        self.stage = stage
        self.status = status


def _repair_domain(table: pd.DataFrame) -> pd.Series:
    """Permit existing Rule 4 holdings, including previously immaterial moves."""
    domain = (table[s.RULE4_TRADE] & table[s.REBALANCE_CADENCE_ELIGIBLE]
              & ~table[s.MANDATORY_TRADE] & ~table[s.SETTLEMENT_CASH])
    # A cutoff repair cannot use a previously excluded entry to satisfy a floor.
    for flag in (s.LOW_WEIGHT_EXCLUDED, s.MODEL_ENTRY, s.MODEL_EXIT,
                 s.CUTOFF_EXIT, s.PINNED_INSTRUCTION, s.CUTOFF_SELL_DOWN):
        if flag in table:
            domain &= ~table[flag].astype(bool)
    return domain.astype(bool)


def _solve_stage(program, problem, stage):
    """Require an optimal convex solve; never use a solver's fallback vector."""
    solver = problem.optimiser_config.solver
    options = {'max_iter': 1000} if solver.upper() == 'CLARABEL' else {}
    try:
        program.solve(solver=solver, verbose=problem.optimiser_config.verbose,
                      warm_start=False, **options)
    except cvx.error.SolverError as error:
        raise _CutoffRepairFailure(stage, 'solver_error') from error
    if program.status != cvx.OPTIMAL:
        raise _CutoffRepairFailure(stage, str(program.status))


# wrapper_minimise_tracking_error cannot express a joint L1 corridor-expansion
# budget. Reuse OP's hard-constraint compiler, covariance atom and solution audit
# for that auxiliary convex endpoint problem; execution sizing stays in OP.
def _repair_cutoff_target(problem: ResolvedExecutionProblem,
                          widening_atol: float = 1e-8) -> _CutoffRepairResult:
    """Find minimum aggregate widening, then least-risk feasible endpoint.

    Phase one minimizes the sum of non-cash weight excursions beyond the
    original all-eligible strict corridors. Phase two minimizes OP covariance
    tracking variance to the unchanged raw model, allowing at most
    ``widening_atol`` additional aggregate widening. The ordinary ranked solver
    then selects and sizes trades inside base-to-repaired-endpoint corridors.
    Its discretionary allowance and feasibility-expansion convention remain.

    Existing exact pins, one-sided mandatory corridors and cutoff sell-down
    retention intervals stay under the native execution compiler. Zero desired
    moves may become feasibility trades, but excluded entries and non-cadence
    Rule 4 holdings cannot be reopened. Material thresholds govern initial
    ranking, not the native feasibility-rescue domain. A successful endpoint
    alone is insufficient: the final ranked portfolio must pass the original
    hard bounds and the recorded aggregate-widening budget.
    """
    if not np.isfinite(widening_atol) or not 0. <= widening_atol <= 1e-6:
        raise ValueError('widening_atol must be finite and between zero and 1e-6')
    snapshot = replace(problem)
    table = snapshot.target.copy(deep=True)
    if s.MATERIAL_TRADE_THRESHOLD not in table:
        raise ValueError('cutoff repair requires the resolved material_trade_threshold')
    threshold = pd.to_numeric(table[s.MATERIAL_TRADE_THRESHOLD], errors='raise')
    if not np.isfinite(threshold).all() or threshold.lt(0.).any():
        raise ValueError('material_trade_threshold must be finite and nonnegative')
    domain = _repair_domain(table)
    if not domain.any():
        raise _CutoffRepairFailure('domain', 'no eligible repair coordinates')
    opened = table.copy(deep=True)
    opened[s.SELECTED_TRADE] = domain
    opened[s.FEASIBILITY_RESCUE_TRADE] = False
    strict_lo, strict_hi = _build_selected_execution_bound_series(
        snapshot.constraints, opened, False)
    hard = build_selected_execution_constraints(
        snapshot.constraints, opened, relax_selected_corridors=True,
        partition_groups=snapshot.partition_groups)
    # The auxiliary repair enforces the supplied group bounds, rather than the
    # legacy compiler's optional 0.5 bp group-endpoint normalization.
    hard = hard.copy(
        group_lower_upper_constraints=snapshot.constraints.group_lower_upper_constraints,
        benchmark_weights=table[s.RAW_MODEL_WEIGHT])
    factor = factorize_covariance(snapshot.covariance.to_numpy())
    w = cvx.Variable(len(table))
    restrictions = hard.set_cvx_all_constraints(
        w=w, covar=factor.covar, covar_factorization=factor)
    positions = np.flatnonzero(domain.to_numpy())
    excursion = cvx.sum(cvx.pos(strict_lo.iloc[positions].to_numpy() - w[positions])
                       + cvx.pos(w[positions] - strict_hi.iloc[positions].to_numpy()))
    phase_one = cvx.Problem(cvx.Minimize(excursion), restrictions)
    _solve_stage(phase_one, snapshot, 'minimum_widening')
    minimum = max(float(phase_one.value), 0.)
    budget = minimum + widening_atol
    risk = cvx_covar_variance(w - table[s.RAW_MODEL_WEIGHT].to_numpy(),
                             factor.covar, covar_factorization=factor)
    phase_two = cvx.Problem(cvx.Minimize(risk), restrictions + [excursion <= budget])
    _solve_stage(phase_two, snapshot, 'endpoint_risk')
    endpoint = pd.Series(np.asarray(w.value).ravel(), index=table.index, name='repaired_endpoint')
    # Native equality pins can leave solver-scale noise. Snap only those pins;
    # re-audit the resulting portfolio before it supplies any execution corridor.
    pins = hard.min_weights.eq(hard.max_weights)
    endpoint.loc[pins] = hard.min_weights.loc[pins]
    endpoint_audit = validate_solution(
        optimal_weights=endpoint.to_numpy(), problem_status=phase_two.status,
        constraints=hard, n=len(table), solver=snapshot.optimiser_config.solver,
        context=snapshot.context + ' cutoff endpoint audit',
        budget_atol=1e-7, bound_atol=1e-7, constraint_atol=1e-7,
        covar=factor.covar, covar_factorization=factor)
    if not endpoint_audit.accepted or not endpoint_audit.compliant:
        raise _CutoffRepairFailure('endpoint_audit', endpoint_audit.reason)
    endpoint_widening = ((strict_lo-endpoint).clip(lower=0.)
                         + (endpoint-strict_hi).clip(lower=0.)).where(domain, 0.)
    if float(endpoint_widening.sum()) > budget + 1e-7:
        raise _CutoffRepairFailure('endpoint_audit', 'widening budget exceeded')

    target = table.copy(deep=True)
    target[s.EFFECTIVE_MODEL_WEIGHT] = endpoint
    target[s.DESIRED_REWEIGHT] = endpoint - target[s.BASE_WEIGHT]
    material = target[s.DESIRED_REWEIGHT].abs().ge(threshold)
    nonzero = target[s.DESIRED_REWEIGHT].abs().gt(s.WEIGHT_ZERO_TOLERANCE)
    target[s.MATERIAL_TRADE] = material
    target[s.TRADE_CANDIDATE] = domain & material & nonzero
    # Freeze Rule 4 movement on all excluded/non-cadence rows even if a caller
    # supplied inconsistent descriptive flags. Lifecycle bounds remain intact.
    target[s.RULE4_TRADE] = domain
    revised = replace(snapshot, target=target, allow_corridor_relaxation=False,
                      context=snapshot.context + ' cutoff target repair')
    solved = solve_ranked_execution(revised)
    if not solved.accepted or not solved.compliant:
        raise _CutoffRepairFailure('ranked_projection', solved.outcome.reason)
    original_audit = validate_solution(
        optimal_weights=solved.weights.reindex(table.index).to_numpy(), problem_status='optimal',
        constraints=hard, n=len(table), solver=snapshot.optimiser_config.solver,
        context=snapshot.context + ' original hard-bound audit',
        budget_atol=1e-7, bound_atol=1e-7, constraint_atol=1e-7,
        covar=factor.covar, covar_factorization=factor)
    if not original_audit.accepted or not original_audit.compliant:
        raise _CutoffRepairFailure('final_audit', original_audit.reason)
    actual_widening = ((strict_lo-solved.weights).clip(lower=0.)
                       + (solved.weights-strict_hi).clip(lower=0.)).where(domain, 0.)
    if float(actual_widening.sum()) > budget + 2e-7:
        raise _CutoffRepairFailure('final_audit', 'widening budget exceeded')
    audit = pd.DataFrame({
        'repair_eligible': domain, 'original_effective_target': table[s.EFFECTIVE_MODEL_WEIGHT],
        'original_corridor_min': strict_lo, 'original_corridor_max': strict_hi,
        'repaired_endpoint': endpoint, 'endpoint_widening': endpoint_widening,
        'executed_weight': solved.weights, 'actual_widening': actual_widening})
    enriched = solved.trade_table.copy(deep=True)
    for name in audit:
        enriched['cutoff_repair_' + name] = audit[name]
    enriched['cutoff_repair_applied'] = True
    solved = replace(solved, trade_table=enriched)
    summary = dict(minimum_widening=minimum, widening_budget=budget,
        endpoint_widening=float(endpoint_widening.sum()),
        actual_widening=float(actual_widening.sum()),
        widened_instruments=int(actual_widening.gt(1e-7).sum()),
        maximum_instrument_widening=float(actual_widening.max()),
        phase_one_status=phase_one.status, phase_two_status=phase_two.status,
        raw_model_unchanged=True, hard_bounds_relaxed=False, excluded_entries_reopened=False,
        expansion_scope='existing cadence-eligible Rule 4 corridors; original pins retained',
        covariance_objective='minimum raw-model tracking variance within widening tolerance')
    return _CutoffRepairResult(solved, revised, endpoint, audit, summary)
