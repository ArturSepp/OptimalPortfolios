"""Opt-in scoring experiments and bounded, audited execution-support search.

These methods consume the same resolved inputs as legacy Ranked Execution.
They never widen corridors, change lifecycle instructions, or impose a new
production policy. QIS owns reported risk; the funded quadratic is used only
as an algorithmic selection kernel, which QIS does not expose.
"""
from dataclasses import dataclass, replace
from enum import Enum
from time import perf_counter
from typing import Optional

import numpy as np
import pandas as pd

from optimalportfolios.covar_estimation.risk_model_adapter import build_risk_model
from optimalportfolios.execution import schema as s
from optimalportfolios.execution.ranking import score_execution_trades, resolve_effective_max_trades
from optimalportfolios.execution.solver import (
    ExecutionOptimizationResult, ExecutionSolverInfeasibility, RELAXED_CORRIDOR_SOLVE,
    _build_selected_execution_bound_series, solve_feasible_execution_portfolio,
    solve_selected_execution_portfolio,
)
from optimalportfolios.execution.types import ResolvedExecutionProblem


class ExecutionScoreMethod(str, Enum):
    """Explicit experimental scores; LEGACY retains every saved ranking control."""

    LEGACY = 'legacy'
    NO_ALPHA = 'no_alpha'
    FULL_RISK = 'full_risk'
    PARTIAL_RISK = 'partial_risk'
    SEQUENTIAL_FULL_RISK = 'sequential_full_risk'


@dataclass(frozen=True)
class ExecutionSearchConfig:
    """Deterministic search budgets and a total initial-TE allowance in basis points.

    ``max_removal_trials`` and ``max_exchange_trials`` count attempted projections,
    including structural failures. Zero disables the corresponding stage.
    ``te_allowance_bp`` is measured against the initial portfolio, never accumulated
    per deletion. Exchanges require ``min_exchange_improvement_bp`` improvement
    versus the current incumbent. ``exchange_candidate_limit`` bounds the incoming
    candidates in each exchange pass, ranked by the supplied score.
    """

    max_removal_trials: int = 100
    max_exchange_trials: int = 100
    te_allowance_bp: float = 1.0
    min_exchange_improvement_bp: float = 1e-4
    exchange_candidate_limit: int = 20

    def __post_init__(self) -> None:
        """Reject nonfinite tolerances and malformed or negative work budgets."""
        for name in ('max_removal_trials', 'max_exchange_trials', 'exchange_candidate_limit'):
            value = getattr(self, name)
            minimum = 1 if name == 'exchange_candidate_limit' else 0
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
                raise ValueError(f'{name} must be an integer')
            if value < minimum:
                raise ValueError(f'{name} must be at least {minimum}')
        for name in ('te_allowance_bp', 'min_exchange_improvement_bp'):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if self.min_exchange_improvement_bp == 0:
            raise ValueError('min_exchange_improvement_bp must be strictly positive')


@dataclass(frozen=True)
class ExecutionSearchResult:
    """Final audited ``result``, per-trial ``attempts`` and aggregate ``summary``."""

    result: ExecutionOptimizationResult
    attempts: pd.DataFrame
    summary: dict


@dataclass(frozen=True)
class ExecutionCorridorDiagnostic:
    """Continuous-domain ``status``, ``bridges``, optional ``result`` and ``reason``."""

    status: str
    bridges: pd.DataFrame
    result: Optional[ExecutionOptimizationResult]
    reason: str


def _eligible_domain(table: pd.DataFrame) -> pd.Series:
    """Include the saved rescue domain, preserving mandatory and cadence exclusions."""
    return (table[s.RULE4_TRADE].astype(bool)
            & table[s.REBALANCE_CADENCE_ELIGIBLE].astype(bool)
            & ~table[s.SETTLEMENT_CASH].astype(bool)
            & ~table[s.MANDATORY_TRADE].astype(bool)
            & table[s.DESIRED_REWEIGHT].abs().gt(s.WEIGHT_ZERO_TOLERANCE))


def _risk_gains(covariance, active, cash, displacement, lower=None, upper=None):
    """Vectorise exact funded variance gains, optionally optimising each interval."""
    covar_active = covariance @ active
    gradient = covar_active-covar_active[cash]
    variance = np.diag(covariance)-2*covariance[:, cash]+covariance[cash, cash]
    if np.any(variance < -1e-10):
        raise ValueError('funded-trade variance is negative')
    variance = np.maximum(variance, 0.)
    moves = np.array(displacement, dtype=float, copy=True)
    if lower is not None:
        positive = variance > 0
        optimum = np.zeros_like(moves)
        np.divide(-gradient, variance, out=optimum, where=positive)
        optimum[~positive & (gradient > 0)] = lower[~positive & (gradient > 0)]
        optimum[~positive & (gradient < 0)] = upper[~positive & (gradient < 0)]
        moves = np.clip(optimum, lower, upper)
    gains = -(2*moves*gradient+moves*moves*variance)
    return gains, moves


def score_execution_candidates(
    problem: ResolvedExecutionProblem,
    method: ExecutionScoreMethod = ExecutionScoreMethod.LEGACY,
    *,
    tie_priority: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Score and select a copied resolved decision with an explicit research method.

    Nonlegacy methods require a disabled minimum-score filter and one-shot saved
    configuration: cutoffs cannot silently cross score units. Sequential full-risk
    selection is requested by its own enum. Alpha removal preserves class weights;
    full/partial risk uses uniform variance-gain scores. Partial risk uses the actual
    solver's effective corridors, including product limits. Sizing and rescue are
    unchanged. Additional columns label the score method and scored displacement.
    Optional ``tie_priority`` supplies unique finite priorities for every instrument;
    lower values break exact score/efficiency ties, including sequential ties.
    It never permutes numerical inputs. None preserves the inherited input order.
    Legacy scoring does not accept a tie override.
    """
    snapshot = replace(problem)
    method = ExecutionScoreMethod(method)
    config = snapshot.ranking_config
    if tie_priority is not None:
        if method == ExecutionScoreMethod.LEGACY:
            raise ValueError('legacy scoring does not accept a tie override')
        if (not tie_priority.index.is_unique
                or set(tie_priority.index) != set(snapshot.target.index)):
            raise ValueError('tie priorities must cover exactly the execution instruments')
        tie_priority = tie_priority.reindex(snapshot.target.index).astype(float)
        if not np.isfinite(tie_priority).all() or not tie_priority.is_unique:
            raise ValueError('tie priorities must be finite and unique')
    if method != ExecutionScoreMethod.LEGACY and (
        config.minimum_trade_score is not None or config.sequential_greedy_selection
    ):
        raise ValueError('research scores require no minimum score and no saved sequential flag')
    table = score_execution_trades(
        snapshot.target, snapshot.alphas, snapshot.covariance, snapshot.asset_classes,
        config, snapshot.asset_class_tre_weights,
    )
    if method == ExecutionScoreMethod.LEGACY:
        return table
    candidate = table[s.TRADE_CANDIDATE].astype(bool)
    desired = table[s.DESIRED_REWEIGHT].to_numpy(dtype=float)
    active = (table[s.BASE_WEIGHT]-table[s.RAW_MODEL_WEIGHT]).to_numpy(dtype=float, copy=True)
    cov = snapshot.covariance.to_numpy()
    cash = int(np.flatnonzero(table[s.SETTLEMENT_CASH].to_numpy())[0])
    lower = upper = None
    if method == ExecutionScoreMethod.PARTIAL_RISK:
        opened = table.copy()
        opened[s.SELECTED_TRADE] = _eligible_domain(table)
        lo, hi = _build_selected_execution_bound_series(snapshot.constraints, opened, False)
        lower = (lo-table[s.BASE_WEIGHT]).to_numpy(copy=True)
        upper = (hi-table[s.BASE_WEIGHT]).to_numpy(copy=True)
        # Rescue may use sub-material Rule 4 rows, so score that full saved domain.
        mask = _eligible_domain(table).to_numpy()
        if np.any(lower[mask] > 0) or np.any(upper[mask] < 0):
            raise ValueError('partial scoring intervals must contain zero')
        lower[~mask], upper[~mask] = 0., 0.
    gains, moves = _risk_gains(cov, active, cash, desired, lower, upper)
    scores = (-table[s.TRE_WEIGHT]*table[s.MARGINAL_TRE]).to_numpy() \
        if method == ExecutionScoreMethod.NO_ALPHA else gains
    table[s.TRADE_SCORE] = scores
    table[s.SCORE_PER_TURNOVER] = table[s.TRADE_SCORE].div(
        table[s.DESIRED_REWEIGHT].abs().where(table[s.MATERIAL_TRADE]))
    table['execution_score_method'] = method.value
    table['scored_displacement'] = moves
    order_frame = table.loc[candidate, [s.TRADE_SCORE, s.SCORE_PER_TURNOVER]].copy()
    order_columns, ascending = [s.TRADE_SCORE, s.SCORE_PER_TURNOVER], [False, False]
    if tie_priority is not None:
        # Assign only candidate rows: a full Series expands an empty pandas frame.
        order_frame['tie_priority'] = tie_priority.reindex(order_frame.index)
        order_columns.append('tie_priority')
        ascending.append(True)
        table['selection_tie_priority'] = tie_priority
    ordered = order_frame.sort_values(order_columns, ascending=ascending, kind='stable').index
    table[s.TRADE_RANK] = pd.Series(pd.NA, index=table.index, dtype='Int64')
    table.loc[ordered, s.TRADE_RANK] = np.arange(1, len(ordered)+1)
    mandatory = int(table[s.MANDATORY_TRADE].sum())
    total = resolve_effective_max_trades(config, mandatory)
    capacity = len(ordered) if total is None else max(total-mandatory, 0)
    table[s.SELECTION_SCORE] = np.nan
    table[s.SELECTION_MARGINAL_TRE] = np.nan
    table[s.SEQUENTIAL_SELECTION_ORDER] = pd.Series(pd.NA, index=table.index, dtype='Int64')
    remaining = list(ordered)
    chosen = []
    for step in range(min(capacity, len(remaining))):
        if method == ExecutionScoreMethod.SEQUENTIAL_FULL_RISK:
            live_gains, _ = _risk_gains(cov, active, cash, desired)
            # Final tie uses original input order, not an earlier changed ranking.
            remaining.sort(key=lambda name: (
                -live_gains[table.index.get_loc(name)],
                -live_gains[table.index.get_loc(name)]/abs(table.at[name, s.DESIRED_REWEIGHT]),
                table.index.get_loc(name) if tie_priority is None else tie_priority.at[name]))
        else:
            live_gains = scores
        name = remaining.pop(0)
        pos = table.index.get_loc(name)
        chosen.append(name)
        table.at[name, s.SELECTION_SCORE] = live_gains[pos]
        table.at[name, s.SEQUENTIAL_SELECTION_ORDER] = step+1
        if method == ExecutionScoreMethod.SEQUENTIAL_FULL_RISK:
            active[pos] += desired[pos]
            active[cash] -= desired[pos]
    table[s.SELECTED_TRADE] = table.index.isin(chosen)
    table[s.REQUESTED_TRADE] = table[s.SELECTED_TRADE]
    table[s.FEASIBILITY_RESCUE_TRADE] = False
    return table


def _solve(problem, table, rescue):
    """Dispatch copied trial tables through the existing audited OP projection."""
    args = dict(trade_table=table, base_constraints=problem.constraints,
                covariance=problem.covariance, optimiser_config=problem.optimiser_config,
                context=problem.context, partition_groups=problem.partition_groups)
    if rescue:
        return solve_feasible_execution_portfolio(
            **args, expand_for_feasibility=problem.expand_for_feasibility,
            sign_directed_rescue=problem.sign_directed_rescue,
            allow_corridor_relaxation=problem.allow_corridor_relaxation)
    return solve_selected_execution_portfolio(**args)


def diagnose_execution_corridors(problem: ResolvedExecutionProblem) -> ExecutionCorridorDiagnostic:
    """Open all saved eligible rescue coordinates and solve the continuous corridor problem.

    This diagnostic never relaxes a corridor or changes desk pins. An accepted
    solution does not certify a strict ticket/minimum-size problem. A returned
    solver infeasibility remains solver-reported; an interval bridge identifies
    only this declared domain, not an unrestricted mandate conflict.
    """
    snapshot = replace(problem)
    table = score_execution_candidates(snapshot)
    table[s.SELECTED_TRADE] = _eligible_domain(table)
    try:
        result = _solve(snapshot, table, rescue=False)
    except ExecutionSolverInfeasibility as error:
        return ExecutionCorridorDiagnostic('interval_infeasible', error.bridges, None, str(error))
    status = 'accepted_continuous' if result.accepted and result.compliant else \
        ('solver_reported_infeasible' if str(result.outcome.status) == 'infeasible'
         else 'unresolved_or_rejected')
    return ExecutionCorridorDiagnostic(status, pd.DataFrame(), result, str(result.outcome.reason))


def improve_ranked_execution(
    problem: ResolvedExecutionProblem,
    method: ExecutionScoreMethod = ExecutionScoreMethod.LEGACY,
    config: ExecutionSearchConfig = ExecutionSearchConfig(),
) -> ExecutionSearchResult:
    """Compress a feasible ranked selection, then try bounded one-for-one exchanges.

    Every trial uses joint sizing and the existing hard audit, with rescue disabled
    so removed coordinates cannot be silently readmitted. Mandatory coordinates,
    cadence and all corridor rules are preserved. Removal never increases actual
    noncash tickets and stays inside the TOTAL initial-TE allowance. Exchanges
    improve current TE and do not increase its actual ticket count. Trial limits
    bound solve counts, not wall-clock time; no global minimum-ticket or full
    neighbourhood optimality claim is made. Failed baselines are never repaired
    by changing their hard rules. Relaxed-corridor incumbents are unsupported.
    """
    snapshot, controls = replace(problem), replace(config)
    table = score_execution_candidates(snapshot, method)
    incumbent = _solve(snapshot, table, rescue=True)
    columns = ['stage', 'removed', 'added', 'status', 'retained', 'te_bp', 'tickets', 'seconds']
    if not (incumbent.accepted and incumbent.compliant):
        return ExecutionSearchResult(incumbent, pd.DataFrame(columns=columns),
                                     {'stop_reason': 'baseline_not_accepted'})
    if incumbent.trade_table[RELAXED_CORRIDOR_SOLVE].any():
        raise NotImplementedError('support improvement requires a strict-corridor incumbent')
    date = pd.Timestamp('2000-01-01')  # A label for one supplied covariance, not a sampled date.
    risk = build_risk_model({date: snapshot.covariance})
    noncash = ~snapshot.target[s.SETTLEMENT_CASH]

    def metrics(result):
        """Use QIS risk and original pre-trade holdings for comparable trial measures."""
        te = 1e4*risk.compute_tre_at_date(
            snapshot.target[s.RAW_MODEL_WEIGHT], result.weights, date)
        if not np.isfinite(te):
            raise ValueError('trial tracking error is unavailable')
        tickets = int((result.weights-snapshot.target[s.CURRENT_WEIGHT]).loc[noncash]
                      .abs().gt(s.WEIGHT_ZERO_TOLERANCE).sum())
        return float(te), tickets

    initial_te, initial_tickets = metrics(incumbent)
    current_te, current_tickets = initial_te, initial_tickets
    initial_selected = incumbent.trade_table[s.SELECTED_TRADE].copy()
    initial_rescue = incumbent.trade_table[s.FEASIBILITY_RESCUE_TRADE].copy()
    trace = []
    counts = {'removal': 0, 'exchange': 0}

    def attempt(removed, added, stage):
        """Audit a single support edit, retaining the incumbent on every rejected trial."""
        nonlocal incumbent, current_te, current_tickets
        trial = incumbent.trade_table.copy(deep=True)
        trial.at[removed, s.SELECTED_TRADE] = False
        trial.at[removed, s.FEASIBILITY_RESCUE_TRADE] = False
        if added is not None:
            trial.at[added, s.SELECTED_TRADE] = True
        started = perf_counter()
        row = dict(stage=stage, removed=removed, added=added, retained=False,
                   te_bp=np.nan, tickets=np.nan)
        counts[stage] += 1
        try:
            candidate = _solve(snapshot, trial, rescue=False)
        except ExecutionSolverInfeasibility as error:
            row['status'] = f'interval_infeasible: {error}'
        else:
            row['status'] = str(candidate.outcome.status)
            if candidate.accepted and candidate.compliant:
                te, tickets = metrics(candidate)
                row.update(te_bp=te, tickets=tickets)
                acceptable_risk = (
                    te <= initial_te+controls.te_allowance_bp if stage == 'removal'
                    else te <= current_te-controls.min_exchange_improvement_bp)
                if acceptable_risk and tickets <= current_tickets:
                    incumbent, current_te, current_tickets = candidate, te, tickets
                    row['retained'] = True
        row['seconds'] = perf_counter()-started
        trace.append(row)
        return row['retained']

    def removable():
        """Try small actual optional tickets first, preserving every mandatory row."""
        frame = incumbent.trade_table
        mask = frame[s.SELECTED_TRADE] & _eligible_domain(frame)
        names = list(frame.index[mask])
        names.sort(key=lambda name: (abs(incumbent.weights[name]-frame.at[name, s.CURRENT_WEIGHT]),
                                     frame.index.get_loc(name)))
        return names

    for stage, limit in [('removal', controls.max_removal_trials),
                         ('exchange', controls.max_exchange_trials)]:
        while counts[stage] < limit:
            changed = False
            frame = incumbent.trade_table
            incoming = frame.loc[_eligible_domain(frame) & ~frame[s.SELECTED_TRADE]].sort_values(
                [s.TRADE_SCORE, s.SCORE_PER_TURNOVER], ascending=False, kind='stable').index
            additions = [None] if stage == 'removal' else list(
                incoming[:controls.exchange_candidate_limit])
            for removed in removable():
                for added in additions:
                    if counts[stage] >= limit:
                        break
                    if attempt(removed, added, stage):
                        changed = True
                        break
                if changed or counts[stage] >= limit:
                    break
            if not changed:
                break
    enriched = incumbent.trade_table.copy()
    enriched['initial_selected_trade'] = initial_selected
    enriched['initial_rescue_trade'] = initial_rescue
    enriched['improvement_removed'] = initial_selected & ~enriched[s.SELECTED_TRADE]
    enriched['improvement_added'] = ~initial_selected & enriched[s.SELECTED_TRADE]
    incumbent = replace(incumbent, trade_table=enriched)
    summary = dict(initial_te_bp=initial_te, final_te_bp=current_te,
                   initial_tickets=initial_tickets, final_tickets=current_tickets,
                   removal_trials=counts['removal'], exchange_trials=counts['exchange'],
                   stop_reason='bounded_search_complete',
                   removal_budget_exhausted=bool(controls.max_removal_trials) and
                   counts['removal'] == controls.max_removal_trials,
                   exchange_budget_exhausted=bool(controls.max_exchange_trials) and
                   counts['exchange'] == controls.max_exchange_trials)
    return ExecutionSearchResult(incumbent, pd.DataFrame(trace, columns=columns), summary)
