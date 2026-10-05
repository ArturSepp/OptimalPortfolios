"""Compress fixed ranking branches under one shared original-incumbent budget."""
from dataclasses import dataclass, replace
from time import perf_counter

import pandas as pd

from optimalportfolios.covar_estimation.risk_model_adapter import build_risk_model
from optimalportfolios.execution import schema as s
from optimalportfolios.execution._budget import (
    _ProjectionBudget, _ProjectionBudgetExhausted, _projection_scope,
)
from optimalportfolios.execution.guarded import (
    ExecutionGuardConfig, _canonical_priority, _metrics, _preference, _qualification,
)
from optimalportfolios.execution.improvement import (
    ExecutionSearchResult, _eligible_domain, _solve, score_execution_candidates,
)
from optimalportfolios.execution.solver import (
    ExecutionOptimizationResult, ExecutionSolverInfeasibility, RELAXED_CORRIDOR_SOLVE,
)
from optimalportfolios.execution.types import ResolvedExecutionProblem


@dataclass(frozen=True)
class ExecutionBranchConfig(ExecutionGuardConfig):
    """Inherit original-baseline guards with a shared 52-projection search ceiling.

    ``max_projection_calls`` must be finite and includes all post-baseline
    proposal/rescue and removal attempts. ``max_removal_trials`` also caps total
    deletions across all branches, never a separate allowance for each branch.
    Other fields have the meaning defined by ``ExecutionGuardConfig``. Optional
    operational guards constrain return candidates, not exploratory branch seeds.
    """

    max_removal_trials: int = 52
    max_projection_calls: int = 52

    def __post_init__(self) -> None:
        """Require an explicit finite projection ceiling for multi-branch search."""
        super().__post_init__()
        if self.max_projection_calls is None:
            raise ValueError('branched search requires max_projection_calls')


@dataclass(frozen=True)
class ExecutionBranchSearchResult(ExecutionSearchResult):
    """Add audited ``checkpoints`` for baseline and every retained branch state.

    The final ``result`` satisfies the original-reference return guard. Internal
    checkpoints may exceed optional operational caps and are not execution orders.
    ``attempts`` identifies each parent and checkpoint and distinguishes branch
    advancement from replacement of the eligible returned incumbent.
    """

    checkpoints: dict[str, ExecutionOptimizationResult]


@dataclass
class _Branch:
    """Keep an independent support path and its pending deletion ordering."""

    name: str
    result: ExecutionOptimizationResult
    metrics: dict
    checkpoint: str
    remaining: list


def _removable(problem, result):
    """Order only optional eligible selected rows by original-holdings trade size."""
    table = result.trade_table
    names = table.index[table[s.SELECTED_TRADE] & _eligible_domain(table)]
    return sorted(names, key=lambda name: (
        abs(result.weights.at[name]-problem.target.at[name, s.CURRENT_WEIGHT]), str(name)))


def solve_branched_execution(
    problem: ResolvedExecutionProblem,
    config: ExecutionBranchConfig = ExecutionBranchConfig(),
) -> ExecutionBranchSearchResult:
    """Explore legacy and fixed ranking branches, then return the best guarded state.

    Solve the unchanged production baseline outside the search budget. Build each
    strict, hard-audited proposal seed even if it fails optional operational caps.
    After seeding, take one deletion attempt per active branch in round-robin order:
    legacy first, then configured methods. A retained deletion must reduce at least
    one ticket count, improve its branch's lexicographic preference and remain
    below original O0 TE plus the single allowance. Rescue is disabled for deletions.

    Every replacement of the returned incumbent independently passes all original
    O0 guards. Branch states never reset risk or turnover references. Save each
    retained branch state for independent path audits. The shared projection budget
    counts every post-baseline attempt, including rescue and structural failure.
    Failed/relaxed baselines return unchanged; structural baseline and unexpected
    programming errors propagate. Time limits remain cooperative between solves.
    """
    snapshot, controls = replace(problem), replace(config)
    if controls.max_projection_calls is None:
        raise ValueError('branched search requires max_projection_calls')
    started = perf_counter()
    baseline = _solve(snapshot, score_execution_candidates(snapshot), rescue=True)
    columns = ['trial_id', 'stage', 'branch', 'removed', 'parent_checkpoint',
               'checkpoint', 'status', 'branch_retained', 'return_retained', 'qualified',
               'branch_reason', 'return_reason', 'te_bp', 'tickets', 'sized_tickets',
               'gross_turnover_bp', 'projection_calls', 'seconds']
    checkpoints = {'baseline': baseline}
    if not (baseline.accepted and baseline.compliant):
        return ExecutionBranchSearchResult(baseline, pd.DataFrame(columns=columns),
            {'stop_reason': 'baseline_not_accepted'}, checkpoints)
    if baseline.trade_table[RELAXED_CORRIDOR_SOLVE].any():
        return ExecutionBranchSearchResult(baseline, pd.DataFrame(columns=columns),
            {'stop_reason': 'baseline_relaxed_corridors'}, checkpoints)
    risk = build_risk_model({pd.Timestamp('2000-01-01'): snapshot.covariance})
    reference = _metrics(snapshot, baseline, risk, controls.ticket_size_bp)
    incumbent, best, selected, selected_checkpoint = baseline, reference, 'legacy', 'baseline'
    branches = [_Branch('legacy', baseline, reference, 'baseline', _removable(snapshot, baseline))]
    proposal_problem = replace(snapshot, ranking_config=replace(snapshot.ranking_config,
        minimum_trade_score=None, sequential_greedy_selection=False))
    priority = _canonical_priority(snapshot.target.index) if controls.canonical_ties else None
    budget = _ProjectionBudget(controls.max_projection_calls)
    trace, removal_trials, proposal_trials = [], 0, 0
    search_started, time_exhausted = perf_counter(), False

    def stopped():
        """Check cooperative time and projection ceilings before any new work."""
        nonlocal time_exhausted
        time_exhausted = (controls.max_search_seconds is not None
                          and perf_counter()-search_started >= controls.max_search_seconds)
        return time_exhausted or budget.exhausted

    def attempt(branch, table, stage, removed=None):
        """Advance one exploratory path, independently qualifying any returned result."""
        nonlocal incumbent, best, selected, selected_checkpoint
        before, trial_started = budget.used, perf_counter()
        row = dict(trial_id=len(trace)+1, stage=stage, branch=branch.name, removed=removed,
                   parent_checkpoint=branch.checkpoint, checkpoint='', branch_retained=False,
                   return_retained=False, qualified=False, return_reason='not_audited')
        try:
            with _projection_scope(budget):
                candidate = _solve(proposal_problem if stage == 'proposal' else snapshot,
                                   table, rescue=stage == 'proposal')
        except _ProjectionBudgetExhausted:
            row.update(status='projection_budget', branch_reason='search_projection_budget')
        except ExecutionSolverInfeasibility as error:
            row.update(status='interval_infeasible', branch_reason=str(error))
        else:
            row['status'] = str(candidate.outcome.status)
            if not (candidate.accepted and candidate.compliant):
                row['branch_reason'] = 'candidate_not_accepted'
            elif candidate.trade_table[RELAXED_CORRIDOR_SOLVE].any():
                row['branch_reason'] = 'candidate_relaxed_corridors'
            else:
                values = _metrics(snapshot, candidate, risk, controls.ticket_size_bp)
                row.update(values)
                qualification = _qualification(reference, values, controls)
                row['qualified'] = qualification in (
                    'tracking_improvement', 'ticket_saving_within_allowance')
                row['return_reason'] = qualification
                advances = stage == 'proposal' or (
                    values['te_bp'] <= reference['te_bp']+controls.te_allowance_bp
                    and _preference(values) < _preference(branch.metrics)
                    and (values['tickets'] < branch.metrics['tickets']
                         or values['sized_tickets'] < branch.metrics['sized_tickets']))
                row['branch_reason'] = 'branch_advanced' if advances else 'branch_not_improved'
                if advances:
                    checkpoint = f'checkpoint_{row["trial_id"]:05d}'
                    checkpoints[checkpoint] = candidate
                    branch.result, branch.metrics, branch.checkpoint = candidate, values, checkpoint
                    branch.remaining = _removable(snapshot, candidate)
                    row.update(branch_retained=True, checkpoint=checkpoint)
                    if row['qualified'] and _preference(values) < _preference(best):
                        incumbent, best = candidate, values
                        selected, selected_checkpoint = branch.name, checkpoint
                        row['return_retained'] = True
                    elif row['qualified']:
                        row['return_reason'] = 'incumbent_preferred'
        row.update(projection_calls=budget.used-before, seconds=perf_counter()-trial_started)
        trace.append(row)
        return row['branch_retained']

    for method in controls.methods:
        if stopped():
            break
        branch = _Branch(method.value, baseline, reference, 'baseline', [])
        table = score_execution_candidates(proposal_problem, method, tie_priority=priority)
        proposal_trials += 1
        if attempt(branch, table, 'proposal'):
            branches.append(branch)
    while (removal_trials < controls.max_removal_trials
           and any(branch.remaining for branch in branches) and not stopped()):
        for branch in branches:
            if removal_trials >= controls.max_removal_trials or stopped():
                break
            if not branch.remaining:
                continue
            name = branch.remaining.pop(0)
            table = branch.result.trade_table.copy(deep=True)
            table.at[name, s.SELECTED_TRADE] = False
            table.at[name, s.FEASIBILITY_RESCUE_TRADE] = False
            removal_trials += 1
            attempt(branch, table, 'removal', name)
    summary = dict(baseline=reference, final=best, selected_method=selected,
                   selected_checkpoint=selected_checkpoint, proposal_trials=proposal_trials,
                   removal_trials=removal_trials, search_projection_calls=budget.used,
                   projection_budget=controls.max_projection_calls,
                   removal_budget_exhausted=bool(controls.max_removal_trials)
                   and removal_trials == controls.max_removal_trials,
                   stop_reason=('search_time_budget' if time_exhausted else
                                'search_projection_budget' if budget.exhausted else
                                'bounded_search_complete'),
                   branches={branch.name: branch.checkpoint for branch in branches},
                   seconds=perf_counter()-started,
                   timing_scope=('baseline plus branched search; '
                                 'projection ceiling excludes common baseline'))
    return ExecutionBranchSearchResult(
        incumbent, pd.DataFrame(trace, columns=columns), summary, checkpoints)
