"""Bounded proposal selection and compression anchored to legacy execution.

The operational thresholds are explicit research controls. They count saved
weight changes without rounding orders or imposing a minimum executable size.
"""
from dataclasses import dataclass, replace
from time import perf_counter
from typing import Optional

import numpy as np
import pandas as pd

from optimalportfolios.covar_estimation.risk_model_adapter import build_risk_model
from optimalportfolios.execution import schema as s
from optimalportfolios.execution._budget import (
    _ProjectionBudget, _ProjectionBudgetExhausted, _projection_scope,
)
from optimalportfolios.execution.improvement import (
    ExecutionScoreMethod, ExecutionSearchResult, _eligible_domain, _solve,
    score_execution_candidates,
)
from optimalportfolios.execution.solver import RELAXED_CORRIDOR_SOLVE, ExecutionSolverInfeasibility
from optimalportfolios.execution.types import ResolvedExecutionProblem


@dataclass(frozen=True)
class ExecutionGuardConfig:
    """Original-incumbent limits and deterministic proposal/compression budgets.

    ``methods`` fixes proposal order; duplicate methods are invalid. Canonical
    ties use instrument-label order only for exact score/efficiency ties.
    The total ``te_allowance_bp`` is always measured from original legacy TE.
    A candidate must improve TE by ``min_te_improvement_bp`` or save a registered
    ticket within that allowance. It may never add registered noncash tickets.
    ``ticket_size_bp`` counts changes strictly above that NAV threshold.
    ``max_sized_ticket_increase`` and ``max_turnover_increase_bp`` bound changes
    from original legacy execution; None disables the respective extra guard.
    Gross noncash turnover is the sum of absolute changes, without division by two.
    ``max_removal_trials`` counts projections, including rejected attempts.
    ``max_search_seconds`` checks elapsed search time between solves, after the
    baseline. It cannot interrupt an in-flight solver and is not a hard deadline.
    ``max_projection_calls`` optionally caps all post-baseline projection attempts,
    including proposal rescues and structural failures; None preserves unlimited
    proposal/rescue work under the other configured stage limits.
    """

    methods: tuple[ExecutionScoreMethod, ...] = (
        ExecutionScoreMethod.PARTIAL_RISK, ExecutionScoreMethod.SEQUENTIAL_FULL_RISK)
    canonical_ties: bool = True
    te_allowance_bp: float = 1.0
    min_te_improvement_bp: float = 1e-4
    ticket_size_bp: float = 1.0
    max_sized_ticket_increase: Optional[int] = 0
    max_turnover_increase_bp: Optional[float] = 0.0
    max_removal_trials: int = 50
    max_search_seconds: Optional[float] = None
    max_projection_calls: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate controls before executing any numerical work."""
        methods = tuple(ExecutionScoreMethod(method) for method in self.methods)
        if len(set(methods)) != len(methods) or ExecutionScoreMethod.LEGACY in methods:
            raise ValueError('proposal methods must be unique nonlegacy methods')
        object.__setattr__(self, 'methods', methods)
        if not isinstance(self.canonical_ties, (bool, np.bool_)):
            raise ValueError('canonical_ties must be boolean')
        for name in ('max_removal_trials', 'max_sized_ticket_increase', 'max_projection_calls'):
            value = getattr(self, name)
            if value is None and name in ('max_sized_ticket_increase', 'max_projection_calls'):
                continue
            if (isinstance(value, (bool, np.bool_))
                    or not isinstance(value, (int, np.integer)) or value < 0):
                raise ValueError(f'{name} must be a nonnegative integer')
        for name in ('te_allowance_bp', 'min_te_improvement_bp', 'ticket_size_bp',
                     'max_turnover_increase_bp', 'max_search_seconds'):
            value = getattr(self, name)
            if value is None and name in ('max_turnover_increase_bp', 'max_search_seconds'):
                continue
            if isinstance(value, (bool, np.bool_)) or not np.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
            if name in ('min_te_improvement_bp', 'ticket_size_bp') and value == 0:
                raise ValueError(f'{name} must be strictly positive')


def _canonical_priority(index):
    """Order exact ties by unique string labels without reordering matrix inputs."""
    labels = index.map(str)
    if not labels.is_unique:
        raise ValueError('canonical ties require distinct instrument string labels')
    order = sorted(range(len(index)), key=lambda pos: labels[pos])
    return pd.Series(np.arange(len(index)), index=index[order]).reindex(index)


def _metrics(problem, result, risk, ticket_size_bp):
    """Use QIS TE and original holdings for every candidate and removal trial."""
    if not result.weights.index.equals(problem.target.index):
        raise ValueError('guarded weights must retain the original instrument order')
    if not np.isfinite(result.weights).all():
        raise ValueError('guarded weights must be finite')
    date = pd.Timestamp('2000-01-01')
    te = float(1e4*risk.compute_tre_at_date(
        problem.target[s.RAW_MODEL_WEIGHT], result.weights, date))
    if not np.isfinite(te) or te < 0:
        raise ValueError('guarded tracking error must be finite and nonnegative')
    changes = (result.weights-problem.target[s.CURRENT_WEIGHT]).abs().loc[
        ~problem.target[s.SETTLEMENT_CASH]]
    return dict(te_bp=te, tickets=int(changes.gt(s.WEIGHT_ZERO_TOLERANCE).sum()),
                sized_tickets=int(changes.gt(ticket_size_bp/1e4).sum()),
                gross_turnover_bp=float(1e4*changes.sum()))


def _qualification(reference, candidate, controls):
    """Apply every limit to the same original incumbent, never a later candidate."""
    if candidate['tickets'] > reference['tickets']:
        return 'more_registered_tickets'
    if (controls.max_sized_ticket_increase is not None
            and candidate['sized_tickets'] > reference['sized_tickets']
            + controls.max_sized_ticket_increase):
        return 'more_sized_tickets'
    # This 1e-8 bp comparison tolerance is far below the registered ticket threshold.
    if (controls.max_turnover_increase_bp is not None
            and candidate['gross_turnover_bp'] > reference['gross_turnover_bp']
            + controls.max_turnover_increase_bp + 1e-8):
        return 'more_turnover'
    if candidate['te_bp'] <= reference['te_bp']-controls.min_te_improvement_bp:
        return 'tracking_improvement'
    if (candidate['tickets'] < reference['tickets']
            and candidate['te_bp'] <= reference['te_bp']+controls.te_allowance_bp):
        return 'ticket_saving_within_allowance'
    return 'no_qualifying_improvement'


def _preference(metrics):
    """Prefer fewer registered tickets, then TE, sized tickets and turnover."""
    return tuple(metrics[name] for name in (
        'tickets', 'te_bp', 'sized_tickets', 'gross_turnover_bp'))


def solve_guarded_execution(
    problem: ResolvedExecutionProblem,
    config: ExecutionGuardConfig = ExecutionGuardConfig(),
) -> ExecutionSearchResult:
    """Compare a fixed proposal bank, then compress under one original risk allowance.

    The legacy solve retains all original controls. Nonlegacy proposals explicitly
    disable the legacy score cutoff and saved sequential flag, as score units differ.
    Joint sizing, rescue, pins and hard constraints use the existing OP solver.
    Only accepted, compliant, strict-corridor proposals are eligible. Selection is
    lexicographic: registered tickets, TE, sized tickets, gross noncash turnover.
    Exact ties keep the incumbent. Every removal must reduce at least one ticket
    count, improve this ordering and still pass all original-incumbent guards.

    Failed legacy results are retained; structural legacy failures propagate.
    Rejected proposals retain the incumbent. A soft time budget stops new work
    between projections and returns the last audited incumbent; it does not kill
    an in-flight solve. No execution rounding, transaction-cost model, hard ticket
    cap, global support optimality or production deadline guarantee is introduced.
    """
    snapshot, controls = replace(problem), replace(config)
    started = perf_counter()
    baseline = _solve(snapshot, score_execution_candidates(snapshot), rescue=True)
    columns = ['stage', 'method', 'removed', 'status', 'retained', 'reason',
               'te_bp', 'tickets', 'sized_tickets', 'gross_turnover_bp', 'seconds',
               'projection_calls']
    if not (baseline.accepted and baseline.compliant):
        return ExecutionSearchResult(baseline, pd.DataFrame(columns=columns),
                                     {'stop_reason': 'baseline_not_accepted'})
    if baseline.trade_table[RELAXED_CORRIDOR_SOLVE].any():
        return ExecutionSearchResult(baseline, pd.DataFrame(columns=columns),
                                     {'stop_reason': 'baseline_relaxed_corridors'})
    date = pd.Timestamp('2000-01-01')
    risk = build_risk_model({date: snapshot.covariance})
    reference = _metrics(snapshot, baseline, risk, controls.ticket_size_bp)
    incumbent, current, source = baseline, reference, 'legacy'
    proposal_problem = replace(snapshot, ranking_config=replace(snapshot.ranking_config,
                               minimum_trade_score=None, sequential_greedy_selection=False))
    priority = _canonical_priority(snapshot.target.index) if controls.canonical_ties else None
    trace, removal_trials, proposal_trials = [], 0, 0
    budget = _ProjectionBudget(controls.max_projection_calls)
    search_started = perf_counter()
    time_exhausted = False

    def out_of_time():
        """Stop new projections after the cooperative search budget has elapsed."""
        nonlocal time_exhausted
        time_exhausted = (controls.max_search_seconds is not None
                          and perf_counter()-search_started >= controls.max_search_seconds)
        return time_exhausted or budget.exhausted

    def attempt(table, stage, method, removed=None):
        """Record a proposed support, qualify it against O0 and retain the best result."""
        nonlocal incumbent, current, source
        trial_started = perf_counter()
        before_calls = budget.used
        row = dict(stage=stage, method=method, removed=removed, retained=False)
        try:
            with _projection_scope(budget):
                candidate = _solve(proposal_problem if stage == 'proposal' else snapshot,
                                   table, rescue=stage == 'proposal')
        except _ProjectionBudgetExhausted:
            row.update(status='projection_budget', reason='search_projection_budget')
        except ExecutionSolverInfeasibility as error:
            row.update(status='interval_infeasible', reason=str(error))
        else:
            row['status'] = str(candidate.outcome.status)
            if not (candidate.accepted and candidate.compliant):
                row['reason'] = 'candidate_not_accepted'
            elif candidate.trade_table[RELAXED_CORRIDOR_SOLVE].any():
                row['reason'] = 'candidate_relaxed_corridors'
            else:
                metrics = _metrics(snapshot, candidate, risk, controls.ticket_size_bp)
                row.update(metrics)
                reason = _qualification(reference, metrics, controls)
                if reason in ('tracking_improvement', 'ticket_saving_within_allowance'):
                    if _preference(metrics) >= _preference(current):
                        reason = 'incumbent_preferred'
                    elif (stage == 'removal' and metrics['tickets'] >= current['tickets']
                          and metrics['sized_tickets'] >= current['sized_tickets']):
                        reason = 'no_ticket_removed'
                    else:
                        incumbent, current = candidate, metrics
                        source = method if stage == 'proposal' else source
                        row['retained'] = True
                row['reason'] = reason
        row['seconds'] = perf_counter()-trial_started
        row['projection_calls'] = budget.used-before_calls
        trace.append(row)
        return row['retained']

    for method in controls.methods:
        if out_of_time():
            break
        table = score_execution_candidates(proposal_problem, method, tie_priority=priority)
        proposal_trials += 1
        attempt(table, 'proposal', method.value)
    while removal_trials < controls.max_removal_trials and not out_of_time():
        table = incumbent.trade_table
        removable = table.index[table[s.SELECTED_TRADE] & _eligible_domain(table)]
        names = sorted(removable, key=lambda name: (
            abs(incumbent.weights.at[name]-snapshot.target.at[name, s.CURRENT_WEIGHT]), str(name)))
        changed = False
        for name in names:
            if removal_trials >= controls.max_removal_trials or out_of_time():
                break
            trial = incumbent.trade_table.copy(deep=True)
            trial.at[name, s.SELECTED_TRADE] = False
            trial.at[name, s.FEASIBILITY_RESCUE_TRADE] = False
            removal_trials += 1
            if attempt(trial, 'removal', source, name):
                changed = True
                break
        if not changed:
            break
    summary = dict(baseline=reference, final=current, selected_method=source,
                   proposal_trials=proposal_trials, removal_trials=removal_trials,
                   stop_reason=('search_time_budget' if time_exhausted else
                                'search_projection_budget' if budget.exhausted else
                                'bounded_search_complete'),
                   search_projection_calls=budget.used,
                   removal_budget_exhausted=bool(controls.max_removal_trials)
                   and removal_trials == controls.max_removal_trials,
                   seconds=perf_counter()-started,
                   timing_scope='baseline plus guarded search; time budget checks between solves')
    return ExecutionSearchResult(incumbent, pd.DataFrame(trace, columns=columns), summary)
