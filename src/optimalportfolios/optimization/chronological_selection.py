"""Bounded chronological portfolio selection through the canonical QIS execution engine.

Inner folds are independent cash-start research experiments. Only their mean net simple
return selects a configuration. One untouched outer period is a sequential implemented
portfolio, never a concatenation of research paths. Fit callbacks must construct every
transform, prior and estimator inside the detached training panel; captured external data
cannot be sandboxed by this API. Fixed observation horizons are conservative purge bounds,
not support for arbitrary overlapping event labels or combinatorial cross validation.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from numbers import Integral
from typing import Callable

import numpy as np
import pandas as pd
import qis

from optimalportfolios.optimization.executed_state import (
    ExecutedRollingResult, backtest_with_qis_state,
)


@dataclass(frozen=True)
class ChronologicalFold:
    """An expanding training cutoff followed by an evaluation calendar.

    Attributes:
        train_end: Requested last observed training close, before the first decision.
        decision_dates: Exact observed evaluation closes, ordered and unique.
        evaluation_end: Last observed valuation close, after every possible execution.
    """

    train_end: pd.Timestamp
    decision_dates: pd.DatetimeIndex
    evaluation_end: pd.Timestamp


@dataclass(frozen=True)
class PortfolioSelectionResult:
    """Selection evidence and a separately scored untouched implemented portfolio.

    Attributes:
        selected: Winning candidate name; insertion order breaks exact score ties.
        scores: Candidate-indexed configuration, mean net return, rank and selection flag.
        fold_scores: Independent candidate/fold net simple returns; no stitched NAV.
        provenance: Actual training cutoffs, label-end bounds and held-out timestamps.
        outer_score: Outer net simple return including initial entry costs.
        outer: One L2 result, with executed holdings propagated across outer decisions.
    """

    selected: str
    scores: pd.DataFrame
    fold_scores: pd.DataFrame
    provenance: pd.DataFrame
    outer_score: float
    outer: ExecutedRollingResult


def _position(index: pd.DatetimeIndex, date) -> int:
    """Require a single exact observed close without silently aligning timestamps."""
    pos = index.get_indexer([date])[0]
    if pos < 0:
        raise ValueError('fold boundaries must be exact observed closes')
    return int(pos)


def _prepare(prices: pd.DataFrame, fold: ChronologicalFold, horizon: int, embargo: int):
    """Resolve a detached fold and conservatively remove label overlap plus the gap."""
    if not isinstance(fold, ChronologicalFold):
        raise TypeError('folds must be ChronologicalFold objects')
    dates = fold.decision_dates
    if (not isinstance(dates, pd.DatetimeIndex) or dates.empty or dates.hasnans
            or not dates.is_unique or not dates.is_monotonic_increasing
            or dates.tz != prices.index.tz or not dates.isin(prices.index).all()):
        raise ValueError('fold decisions must be ordered unique observed closes')
    start = _position(prices.index, dates[0])
    requested = _position(prices.index, fold.train_end)
    end = _position(prices.index, fold.evaluation_end)
    if requested >= start or end <= _position(prices.index, dates[-1]):
        raise ValueError('training must precede evaluation, which must include a later close')
    cutoff = min(requested, start - horizon - embargo - 1)
    if cutoff < 1:
        raise ValueError('purging and embargo must leave at least two training observations')
    details = dict(requested_train_end=prices.index[requested], train_start=prices.index[0],
                   train_end=prices.index[cutoff], label_end=prices.index[cutoff + horizon],
                   purged_rows=requested - cutoff, embargo_rows=embargo,
                   evaluation_start=dates[0], evaluation_end=prices.index[end])
    return cutoff, prices.iloc[start:end + 1].copy(deep=True), dates.copy(), details


def _net_return(result: ExecutedRollingResult, initial_nav: float) -> float:
    """Score through QIS with an explicit initial-capital anchor including entry fees."""
    anchors = pd.Series([initial_nav, result.portfolio.nav.iloc[-1]],
                        index=result.portfolio.nav.index[[0, -1]])
    score = float(qis.compute_total_return(anchors))
    if not np.isfinite(score):
        raise ValueError('portfolio score must be finite')
    return score


def select_portfolio_model(
        prices: pd.DataFrame,
        candidates: dict[str, dict],
        fit: Callable,
        inner_folds: tuple[ChronologicalFold, ...],
        outer_fold: ChronologicalFold,
        *,
        label_horizon: int = 0,
        embargo: int = 0,
        implementation_lag: int = 1,
        transaction_costs: float = 0.0,
        initial_nav: float = 100.0,
        ) -> PortfolioSelectionResult:
    """Fit fresh policies, select on past research folds, evaluate once on held-out prices.

    Requires QIS >=5.37 through L2. Scores maximize the equally weighted arithmetic mean
    of inner-fold net simple total returns, without annualisation or a risk adjustment.
    Compare candidates on the same folds; unequal fold durations remain equally weighted.
    All folds start in cash, with the same execution lag and costs. An outer policy is fitted
    once after selection; its sequential L2 decisions use actual executed state.

    Args:
        prices: Positive common-currency total-return prices on ordered unique closes.
            NaN quotes follow L2's freeze contract. Observation frequency is caller supplied.
        candidates: One to 32 insertion-ordered names mapped to configuration dictionaries,
            for example covariance span and alpha strength. Values are detached per fit.
        fit: Factory ``fit(training_prices, configuration) -> decide(history, state)``.
            Create preprocessing, priors, estimates and any FactorLasso CV inside each call.
            Fit sees only the purged expanding training prefix, never scores or future rows.
            The returned policy receives evaluation history through the decision and L2 state.
            Its retained training history is the factory's responsibility. No sklearn adapter
            or protection against data captured by user callbacks is promised.
        inner_folds: One to ten chronological, nonoverlapping independent research windows.
            Their entire evaluations must precede the actual outer training cutoff.
        outer_fold: One untouched forward evaluation. Its scores never enter selection.
        label_horizon: Maximum forward-label span in price observations. Training is capped
            so the last possible label end strictly precedes evaluation after the embargo.
            Callbacks must build labels entirely within their supplied panel; this additional
            exclusion is conservative. Irregular event-dependent horizons need another API.
        embargo: Additional observed closes after the label-end bound and before evaluation.
            This is a pre-evaluation gap; no later blocks are used for training.
        implementation_lag: L2 price-observation execution lag, applied once.
        transaction_costs: L2 proportional trade fee, included in selection and outer scoring.
        initial_nav: Positive initial cash for every independent fold and the outer portfolio.

    Returns:
        Selection tables, auditable fold cutoffs and a separately labelled outer L2 result.

    Raises:
        ValueError: On invalid grids, calendars, gaps, scores or L2 accounting inputs.
        TypeError: On invalid folds or a factory that does not return a callable policy.
        RuntimeError: If L2's QIS version requirement is not satisfied.
    """
    if (not isinstance(prices, pd.DataFrame) or not isinstance(prices.index, pd.DatetimeIndex)
            or prices.index.empty or prices.index.hasnans or not prices.index.is_unique
            or not prices.index.is_monotonic_increasing):
        raise ValueError('prices need ordered unique observed datetimes')
    prices = prices.copy(deep=True)
    if (not isinstance(candidates, dict) or not 1 <= len(candidates) <= 32
            or any(not isinstance(k, str) or not k or not isinstance(v, dict)
                   for k, v in candidates.items())):
        raise ValueError('supply one to 32 named configuration dictionaries')
    candidates = deepcopy(candidates)
    if not 1 <= len(inner_folds) <= 10:
        raise ValueError('supply one to ten independent inner folds')
    for gap in (label_horizon, embargo):
        if isinstance(gap, (bool, np.bool_)) or not isinstance(gap, Integral) or gap < 0:
            raise ValueError('label_horizon and embargo must be nonnegative integers')
    outer = _prepare(prices, outer_fold, label_horizon, embargo)
    inner = [_prepare(prices, f, label_horizon, embargo) for f in inner_folds]
    for i, plan in enumerate(inner):
        if (plan[3]['evaluation_end'] > prices.index[outer[0]]
                or (i and plan[3]['evaluation_start'] <= inner[i - 1][3]['evaluation_end'])):
            raise ValueError('inner evaluations must be ordered, disjoint '
                             'and inside outer training')

    def evaluate(name, plan):
        """Build a fresh isolated model and delegate sequential accounting to L2."""
        cutoff, evaluation, decisions, _ = plan
        policy = fit(prices.iloc[:cutoff + 1].copy(deep=True), deepcopy(candidates[name]))
        if not callable(policy):
            raise TypeError('fit must return a callable decision policy')
        result = backtest_with_qis_state(
            evaluation, decisions, policy, implementation_lag=implementation_lag,
            transaction_costs=transaction_costs, initial_nav=initial_nav)
        return result, _net_return(result, initial_nav)

    rows = []
    for name in candidates:
        for i, plan in enumerate(inner):
            _, score = evaluate(name, plan)
            rows.append(dict(candidate=name, fold=i, net_return=score))
    fold_scores = pd.DataFrame(rows)
    means = fold_scores.groupby('candidate', sort=False).net_return.mean()
    selected = str(means.idxmax())
    scores = pd.DataFrame({'configuration': list(candidates.values()),
                           'mean_net_return': means.reindex(candidates).to_numpy()},
                          index=pd.Index(candidates, name='candidate'))
    scores['rank'] = scores.mean_net_return.rank(method='first', ascending=False).astype(int)
    scores['selected'] = scores.index == selected
    outer_result, outer_score = evaluate(selected, outer)
    provenance = pd.DataFrame([dict(role='independent_inner', fold=i, **p[3])
                               for i, p in enumerate(inner)]
                              + [dict(role='sequential_outer', fold=0, **outer[3])])
    return PortfolioSelectionResult(selected, scores, fold_scores, provenance,
                                    outer_score, outer_result)
