"""Compress fixed ranking branches under one shared original-incumbent budget."""
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, replace
from time import perf_counter

import numpy as np
import pandas as pd

from optimalportfolios.covar_estimation.risk_model_adapter import build_risk_model
from optimalportfolios.execution import schema as s
from optimalportfolios.execution._budget import (
    _ProjectionBudget, _ProjectionBudgetExhausted, _projection_scope,
)
from optimalportfolios.execution._projection_cache import _ProjectionCache, _projection_reuse_scope
from optimalportfolios.execution._swaps import (
    _exact_swap, _support, _swap_candidates, _turnover_capped_problem,
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
    ``allow_support_plateaus`` permits an exact optional-support deletion with
    unchanged registered and sized counts under the original risk allowance.
    ``preserve_guarded_path`` adds a guard-respecting continuation from the best
    eligible seed, scheduled first each round under the same shared work ceiling.
    ``reuse_projection_solves`` reuses identical successful resolved projections
    within this search, retaining logical projection charges and the search order.
    It requires ``max_search_seconds=None``; wall-clock stopping is not invariant
    to faster solves. ``reuse_projection_programs`` additionally reuses compatible
    box-parameterized problem graphs with cold native solves; it requires
    ``reuse_projection_solves=True``. These reuse switches default to False.
    ``max_swap_trials=0`` disables support swaps. A positive limit reserves that
    many positions of the same shared ceiling from deletion work, then explores
    strict one-for-one optional-support edits from the eligible incumbent.
    ``swap_candidate_limit`` bounds incoming names per outgoing name;
    ``swap_depth`` and ``swap_beam_width`` bound continuation levels/frontier size.
    Swap states satisfy original risk/count/turnover caps but may be provisional;
    the unchanged original-baseline qualification alone authorizes a return.
    ``reserve_swap_budget=True`` retains the original up-front reservation.
    False completes the swap-disabled deletion path first, then uses only spare
    shared projections for swaps. With no wall-clock stopping, that continuation
    cannot displace a return from the completed fixed-work deletion path.
    """

    max_removal_trials: int = 52
    max_projection_calls: int = 52
    allow_support_plateaus: bool = False
    preserve_guarded_path: bool = False
    reuse_projection_solves: bool = False
    reuse_projection_programs: bool = False
    max_swap_trials: int = 0
    swap_candidate_limit: int = 8
    swap_depth: int = 1
    swap_beam_width: int = 1
    reserve_swap_budget: bool = True

    def __post_init__(self) -> None:
        """Require an explicit finite projection ceiling for multi-branch search."""
        super().__post_init__()
        for name in ('allow_support_plateaus', 'preserve_guarded_path', 'reuse_projection_solves',
                     'reserve_swap_budget',
                     'reuse_projection_programs'):
            if not isinstance(getattr(self, name), (bool, np.bool_)):
                raise ValueError(f'{name} must be boolean')
        if self.max_projection_calls is None:
            raise ValueError('branched search requires max_projection_calls')
        if self.reuse_projection_solves and self.max_search_seconds is not None:
            raise ValueError('projection reuse requires max_search_seconds=None')
        if self.reuse_projection_programs and not self.reuse_projection_solves:
            raise ValueError('program reuse requires reuse_projection_solves=True')
        for name in ('max_swap_trials', 'swap_candidate_limit', 'swap_depth', 'swap_beam_width'):
            value = getattr(self, name)
            minimum = 0 if name == 'max_swap_trials' else 1
            if (isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer))
                    or value < minimum):
                raise ValueError(f'{name} must be an integer of at least {minimum}')
        if self.max_swap_trials > self.max_projection_calls:
            raise ValueError('max_swap_trials cannot exceed max_projection_calls')


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



def _support_plateau(branch, candidate, values, removed):
    """Require equal counted trades and exactly one eligible selected-support deletion."""
    if (values['tickets'] != branch.metrics['tickets']
            or values['sized_tickets'] != branch.metrics['sized_tickets']):
        return False
    parent, child = branch.result.trade_table, candidate.trade_table
    if (not parent.index.equals(child.index) or removed not in parent.index
            or not parent.at[removed, s.SELECTED_TRADE]
            or not _eligible_domain(parent).at[removed]):
        return False
    expected = parent[s.SELECTED_TRADE].copy()
    expected.at[removed] = False
    return child[s.SELECTED_TRADE].equals(expected)


def solve_branched_execution(
    problem: ResolvedExecutionProblem,
    config: ExecutionBranchConfig = ExecutionBranchConfig(),
    *,
    on_incumbent: Callable[[str, ExecutionOptimizationResult], None] | None = None,
) -> ExecutionBranchSearchResult:
    """Explore legacy and fixed ranking branches, then return the best guarded state.

    Solve the unchanged production baseline outside the search budget. Build each
    strict, hard-audited proposal seed even if it fails optional operational caps.
    After seeding, take one deletion attempt per active branch in round-robin order:
    legacy first, then configured methods. A retained deletion must reduce at least
    one ticket count, improve its branch's lexicographic preference and remain
    below original O0 TE plus the single allowance. Opt-in support plateaus permit
    flat counts only when exactly one eligible selected coordinate disappears;
    their internal preference may worsen. Opt-in guarded continuation starts at
    the best eligible seed and gets the first turn of each shared-budget round.
    Rescue is disabled for deletions. Final return preference is unchanged.

    Every replacement of the returned incumbent independently passes all original
    O0 guards. Branch states never reset risk or turnover references. Save each
    retained branch state for independent path audits. The shared projection budget
    counts every post-baseline attempt, including rescue and structural failure.
    Failed/relaxed baselines return unchanged; structural baseline and unexpected
    programming errors propagate. Time limits remain cooperative between solves.

    Optional ``on_incumbent(checkpoint_id, result)`` receives a detached copy of
    the strict accepted baseline and each replacement of the eligible incumbent.
    Exploratory states are never delivered. The synchronous callback's errors
    propagate; its work consumes elapsed time but no projection allowance. It can
    persist progress for an external supervisor, but does not enforce a deadline
    or guarantee a feasible result before the baseline completes.
    """
    if on_incumbent is not None and not callable(on_incumbent):
        raise TypeError('on_incumbent must be callable or None')
    snapshot, controls = replace(problem), replace(config)
    if controls.max_projection_calls is None:
        raise ValueError('branched search requires max_projection_calls')
    started = perf_counter()
    with _projection_reuse_scope(None):
        baseline = _solve(snapshot, score_execution_candidates(snapshot), rescue=True)
    columns = ['trial_id', 'stage', 'branch', 'removed', 'parent_checkpoint',
               'checkpoint', 'status', 'branch_retained', 'return_retained', 'qualified',
               'branch_reason', 'return_reason', 'te_bp', 'tickets', 'sized_tickets',
               'gross_turnover_bp', 'projection_calls', 'seconds']
    if controls.max_swap_trials:
        columns += ['added', 'swap_depth']
    cache = _ProjectionCache() if controls.reuse_projection_solves else None
    if controls.reuse_projection_programs:
        cache.enable_program_reuse()
    if cache is not None:
        columns += ['numerical_projection_calls', 'projection_cache_hits']
    checkpoints = {'baseline': baseline}
    if not (baseline.accepted and baseline.compliant):
        return ExecutionBranchSearchResult(baseline, pd.DataFrame(columns=columns),
            {'stop_reason': 'baseline_not_accepted'}, checkpoints)
    if baseline.trade_table[RELAXED_CORRIDOR_SOLVE].any():
        return ExecutionBranchSearchResult(baseline, pd.DataFrame(columns=columns),
            {'stop_reason': 'baseline_relaxed_corridors'}, checkpoints)
    if on_incumbent is not None:
        on_incumbent('baseline', deepcopy(baseline))
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

    def attempt(branch, table, stage, removed=None, added=None, depth=None):
        """Advance one exploratory path, independently qualifying any returned result."""
        nonlocal incumbent, best, selected, selected_checkpoint
        before, trial_started = budget.used, perf_counter()
        numerical_before = cache.numerical_calls if cache is not None else 0
        hits_before = cache.hits if cache is not None else 0
        row = dict(trial_id=len(trace)+1, stage=stage, branch=branch.name, removed=removed,
                   parent_checkpoint=branch.checkpoint, checkpoint='', branch_retained=False,
                   return_retained=False, qualified=False, return_reason='not_audited')
        if controls.max_swap_trials:
            row.update(added=added, swap_depth=depth)
        try:
            with _projection_scope(budget), _projection_reuse_scope(cache):
                sizing_problem = (_turnover_capped_problem(snapshot, table, reference, controls)
                                  if stage == 'swap' else
                                  proposal_problem if stage == 'proposal' else snapshot)
                candidate = _solve(sizing_problem, table, rescue=stage == 'proposal')
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
                plateau = (stage == 'removal' and controls.allow_support_plateaus
                           and values['te_bp'] <= reference['te_bp']+controls.te_allowance_bp
                           and _support_plateau(branch, candidate, values, removed))
                advances = advances or plateau
                row['branch_reason'] = ('support_plateau' if plateau else
                                        'branch_advanced' if advances else 'branch_not_improved')
                if stage == 'swap':
                    advances = (
                        values['tickets'] <= branch.metrics['tickets']
                        and values['te_bp'] <= reference['te_bp']+controls.te_allowance_bp
                        and qualification not in (
                            'more_registered_tickets', 'more_sized_tickets', 'more_turnover')
                        and _exact_swap(
                            branch.result.trade_table, candidate.trade_table, removed, added))
                    row['branch_reason'] = 'swap_bounded' if advances else 'swap_rejected'
                if branch.name == 'guarded':
                    guarded = row['qualified'] or (plateau and qualification not in (
                        'more_registered_tickets', 'more_sized_tickets', 'more_turnover'))
                    if advances and not guarded:
                        advances = False
                        row['branch_reason'] = 'guarded_return_rejected'
                if advances:
                    checkpoint = f'checkpoint_{row["trial_id"]:05d}'
                    checkpoints[checkpoint] = candidate
                    branch.result, branch.metrics, branch.checkpoint = candidate, values, checkpoint
                    branch.remaining = [] if stage == 'swap' else _removable(snapshot, candidate)
                    row.update(branch_retained=True, checkpoint=checkpoint)
                    if row['qualified'] and _preference(values) < _preference(best):
                        incumbent, best = candidate, values
                        selected, selected_checkpoint = branch.name, checkpoint
                        row['return_retained'] = True
                    elif row['qualified']:
                        row['return_reason'] = 'incumbent_preferred'
        row.update(projection_calls=budget.used-before, seconds=perf_counter()-trial_started)
        if cache is not None:
            row.update(numerical_projection_calls=cache.numerical_calls-numerical_before,
                       projection_cache_hits=cache.hits-hits_before)
        trace.append(row)
        if row['return_retained'] and on_incumbent is not None:
            on_incumbent(selected_checkpoint, deepcopy(incumbent))
        return row['branch_retained']

    for method in controls.methods:
        if stopped():
            break
        branch = _Branch(method.value, baseline, reference, 'baseline', [])
        table = score_execution_candidates(proposal_problem, method, tie_priority=priority)
        proposal_trials += 1
        if attempt(branch, table, 'proposal'):
            branches.append(branch)
    guarded_seed = None
    if controls.preserve_guarded_path:
        guarded_seed = selected_checkpoint
        branches.insert(0, _Branch('guarded', incumbent, best, selected_checkpoint,
                                   _removable(snapshot, incumbent)))
    removal_phase_limit = controls.max_projection_calls-(
        controls.max_swap_trials if controls.reserve_swap_budget else 0)
    def reserved():
        """Stop deletions at the optional swap reservation, without changing the total ceiling."""
        return bool(controls.max_swap_trials and budget.used >= removal_phase_limit)
    while (removal_trials < controls.max_removal_trials
           and any(branch.remaining for branch in branches) and not stopped() and not reserved()):
        for branch in branches:
            if removal_trials >= controls.max_removal_trials or stopped() or reserved():
                break
            if not branch.remaining:
                continue
            name = branch.remaining.pop(0)
            table = branch.result.trade_table.copy(deep=True)
            table.at[name, s.SELECTED_TRADE] = False
            table.at[name, s.FEASIBILITY_RESCUE_TRADE] = False
            removal_trials += 1
            attempt(branch, table, 'removal', name)
    swap_trials, swap_levels, swap_candidates, swap_seed = 0, 0, 0, None
    frontier = []
    if controls.max_swap_trials and not stopped():
        swap_seed = selected_checkpoint
        frontier = [_Branch('swap', incumbent, best, selected_checkpoint, [])]
        visited = {_support(incumbent.trade_table)}
        for depth in range(1, controls.swap_depth+1):
            if not frontier or swap_trials >= controls.max_swap_trials or stopped():
                break
            swap_levels = depth
            pool = []
            for parent in frontier:
                for gain, removed, added in _swap_candidates(
                        snapshot, parent.result, controls.swap_candidate_limit,
                        controls.canonical_ties):
                    pool.append((gain, parent, removed, added))
            pool.sort(key=lambda pair: (-pair[0], pair[1].checkpoint))
            swap_candidates += len(pool)
            remaining_levels = controls.swap_depth-depth+1
            allowance = (controls.max_swap_trials-swap_trials+remaining_levels-1)//remaining_levels
            children, level_trials = [], 0
            for _, parent, removed, added in pool:
                if (level_trials >= allowance or swap_trials >= controls.max_swap_trials
                        or stopped()):
                    break
                table = parent.result.trade_table.copy(deep=True)
                table.at[removed, s.SELECTED_TRADE] = False
                table.at[removed, s.FEASIBILITY_RESCUE_TRADE] = False
                table.at[added, s.SELECTED_TRADE] = True
                table.at[added, s.FEASIBILITY_RESCUE_TRADE] = False
                signature = _support(table)
                if signature in visited:
                    continue
                visited.add(signature)
                child = _Branch('swap', parent.result, parent.metrics, parent.checkpoint, [])
                swap_trials += 1
                level_trials += 1
                if attempt(child, table, 'swap', removed, added, depth):
                    children.append(child)
            children.sort(key=lambda branch: (_preference(branch.metrics), branch.checkpoint))
            frontier = children[:controls.swap_beam_width]
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
    if controls.preserve_guarded_path:
        summary['guarded_seed_checkpoint'] = guarded_seed
    if controls.max_swap_trials:
        summary.update(swap_trials=swap_trials, swap_levels=swap_levels,
                       swap_candidates=swap_candidates, swap_seed_checkpoint=swap_seed,
                       swap_budget_exhausted=swap_trials == controls.max_swap_trials,
                       removal_phase_projection_limit=removal_phase_limit,
                       swap_frontier=[branch.checkpoint for branch in frontier])
    if cache is not None:
        summary.update(search_numerical_projection_calls=cache.numerical_calls,
                       search_projection_cache_hits=cache.hits,
                       search_covariance_factorizations=cache.preparations.factorizations,
                       search_covariance_cache_hits=cache.preparations.hits)
        if cache.programs is not None:
            summary.update(search_program_builds=cache.programs.builds,
                           search_program_cache_hits=cache.programs.hits,
                           search_program_bypasses=cache.programs.bypasses)
    return ExecutionBranchSearchResult(
        incumbent, pd.DataFrame(trace, columns=columns), summary, checkpoints)
