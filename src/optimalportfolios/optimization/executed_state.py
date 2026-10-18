"""Optional chronological decisions fed by canonical QIS executed holdings.

This bounded cash-security workflow replays accepted targets through QIS on each observed
prefix. OP owns decisions and policy; QIS alone advances units, cash and trade costs. Existing
rolling solvers keep their decision-date drift approximation. No outstanding order queue,
funding, management fees, short positions or initial external holdings are supported here.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from importlib.metadata import version as _distribution_version
import re
from numbers import Integral
from typing import Callable

import numpy as np
import pandas as pd
import qis

from optimalportfolios.optimization.solver_diagnostics import OptimizationOutcome


@dataclass(frozen=True)
class ExecutedState:
    """Detached pre-decision snapshot from a QIS replay.

    Attributes:
        date: Observed decision close, before the new order.
        units: Actual held cash-security units, aligned by asset label.
        weights: Actual position values divided by post-cost NAV; never renormalised.
        cash: Residual currency balance, including cost funding debits.
        nav: Positive marked portfolio NAV in the common valuation currency.
        quoted: Whether each asset has a current execution quote at this close.
        eligible: Point-in-time policy eligibility supplied by the caller.
    """

    date: pd.Timestamp
    units: pd.Series
    weights: pd.Series
    cash: float
    nav: float
    quoted: pd.Series
    eligible: pd.Series


@dataclass(frozen=True)
class ExecutedRollingResult:
    """Rolling decisions and the canonical reporting object.

    Attributes:
        portfolio: QIS PortfolioData including actual units, NAV, costs and turnover methods.
        targets: Accepted target rows indexed by decision close, in price-column order.
        states: Pre-decision snapshots, including skipped decisions.
        events: Decision-indexed execution dates, status and reason; skipped orders never trade.
    """

    portfolio: qis.PortfolioData
    targets: pd.DataFrame
    states: tuple[ExecutedState, ...]
    events: pd.DataFrame


def _dates(index: pd.Index, name: str, timezone) -> None:
    """Reject ambiguous, unordered or timezone-incompatible observation calendars."""
    if (not isinstance(index, pd.DatetimeIndex) or index.empty or index.hasnans
            or not index.is_unique or not index.is_monotonic_increasing
            or index.tz != timezone):
        raise ValueError(f'{name} must be ordered unique datetimes with the price timezone')


def _replay(prices: pd.DataFrame, schedule: dict, initial_nav: float,
            transaction_costs: float) -> qis.PortfolioData:
    """Delegate every state transition to QIS, including a zero-position opening sentinel."""
    rows = {prices.index[0]: pd.Series(0.0, index=prices.columns)} | schedule
    weights = pd.DataFrame.from_dict(rows, orient='index').reindex(columns=prices.columns)
    return qis.backtest_model_portfolio(
        prices=prices, weights=weights, initial_nav=initial_nav,
        rebalancing_freq=None, weight_implementation_lag=0,
        rebalancing_costs=transaction_costs, ticker='Executed-state decisions')


def backtest_with_qis_state(
        prices: pd.DataFrame,
        decision_dates: pd.DatetimeIndex,
        decide: Callable,
        *,
        implementation_lag: int = 1,
        transaction_costs: float = 0.0,
        initial_nav: float = 100.0,
        eligibility: pd.DataFrame | None = None,
        ) -> ExecutedRollingResult:
    """Call a policy with actual holdings before each optional close-price trade.

    Requires QIS 5.37 or later for its frozen-holdings and marked-cash contract.

    Args:
        prices: Positive common-currency total-return cash-security prices; NaNs are unquoted.
            The fixed column superset can include late entrants and discontinued assets.
        decision_dates: Exact observed closes. Each previous execution must strictly precede
            the next decision; every execution needs a subsequent price observation.
        decide: Callable ``decide(history, state)`` receiving detached observed prices through
            this close and an ExecutedState. Return a labelled long-only Series with sum at
            most one, None to hold, or the existing single-date wrapper's ``(weights, outcome)``.
            Rejected/noncompliant OptimizationOutcome means hold, ignoring its fallback vector.
            The caller owns causal estimation, covariance units and all financial constraints.
        implementation_lag: Nonnegative observation count, resolved once before QIS replay.
            Zero lag assumes observation, decision and trade at the same model close.
        transaction_costs: Nonnegative fractional proportional costs paid from QIS cash.
        initial_nav: Positive initial cash in the common price currency.
        eligibility: Binary point-in-time asset policy, forward-filled from earlier dated rows.
            Quoted ineligible assets require a zero target; unquoted assets must retain their
            actual weight. Future quote availability is never supplied to the policy.

    Returns:
        ExecutedRollingResult with canonical reporting, accepted decisions and audited states.
        State uses only earlier accepted executions. Costs can make cash negative and actual
        risky weights sum above one. No borrowing limit or cost-funding rescaling is inferred.

    Raises:
        ValueError: On ambiguous inputs, overlapping execution windows or invalid targets.
        TypeError: If a policy returns neither a Series, None nor a weights/outcome tuple.
        RuntimeError: If the installed QIS predates the required 5.37 execution-state contract.
    """
    engine_version = re.match(r'(\d+)\.(\d+)', _distribution_version('qis'))
    if engine_version is None or tuple(map(int, engine_version.groups())) < (5, 37):
        raise RuntimeError('backtest_with_qis_state requires qis>=5.37; upgrade QIS for '
                           'this opt-in workflow. Existing OP backtests retain their core floor.')
    if not isinstance(prices, pd.DataFrame):
        raise ValueError('prices must be a DataFrame')
    prices = prices.copy(deep=True)
    _dates(prices.index, 'prices', prices.index.tz if isinstance(
        prices.index, pd.DatetimeIndex) else None)
    _dates(decision_dates, 'decision_dates', prices.index.tz)
    if prices.columns.empty or not prices.columns.is_unique:
        raise ValueError('price asset labels must be nonempty and unique')
    values = prices.to_numpy(dtype=float)
    if np.isinf(values).any() or (values[np.isfinite(values)] <= 0).any():
        raise ValueError('quoted prices must be positive and finite; NaNs mean unquoted')
    if (isinstance(implementation_lag, (bool, np.bool_))
            or not isinstance(implementation_lag, Integral) or implementation_lag < 0):
        raise ValueError('implementation_lag must be a nonnegative integer')
    if (not np.isfinite(initial_nav) or initial_nav <= 0
            or not np.isfinite(transaction_costs) or transaction_costs < 0):
        raise ValueError('initial_nav must be positive and costs nonnegative, both finite')
    if not decision_dates.isin(prices.index).all():
        raise ValueError('decisions must be exact price observations')
    positions = prices.index.get_indexer(decision_dates)
    executions = positions + implementation_lag
    if (executions >= len(prices) - 1).any():
        raise ValueError('every execution needs a subsequent observation')
    if (executions[:-1] >= positions[1:]).any():
        raise ValueError('previous execution must precede next decision; no pending orders')
    eligible = pd.DataFrame(True, index=decision_dates, columns=prices.columns)
    if eligibility is not None:
        _dates(eligibility.index, 'eligibility', prices.index.tz)
        if (not eligibility.columns.is_unique
                or set(eligibility.columns) != set(prices.columns)):
            raise ValueError('eligibility needs exactly the price assets')
        eligible = eligibility.reindex(columns=prices.columns).reindex(
            index=decision_dates, method='ffill')
        if not eligible.isin([0, 1, False, True]).to_numpy().all():
            raise ValueError('eligibility needs known binary values at each decision')
        eligible = eligible.astype(bool)

    schedule, targets, states, events = {}, {}, [], []
    for date, execution_position in zip(decision_dates, executions):
        history = prices.loc[:date].copy(deep=True)
        previous = _replay(history, schedule, initial_nav, transaction_costs)
        nav = float(previous.nav.iloc[-1])
        units = previous.units.iloc[-1].copy()
        marks = history.ffill().fillna(0.0).iloc[-1]
        if not np.isfinite(nav) or nav <= 0:
            raise ValueError('executed state requires positive finite NAV')
        # An accounting identity from QIS output, never a separate cash transition engine.
        state = ExecutedState(date, units, previous.weights.iloc[-1].copy(),
                              nav - float(units @ marks), nav,
                              history.iloc[-1].notna(), eligible.loc[date].copy())
        states.append(deepcopy(state))
        candidate = decide(history, deepcopy(state))
        reason = 'policy returned no order'
        if isinstance(candidate, tuple):
            if len(candidate) != 2 or not isinstance(candidate[1], OptimizationOutcome):
                raise TypeError('expected (labelled weights, OptimizationOutcome)')
            target, outcome = candidate
            reason = outcome.reason or outcome.status or 'solver rejected or noncompliant'
            candidate = target if outcome.accepted and outcome.compliant else None
        execution = prices.index[execution_position]
        if candidate is None:
            events.append(dict(decision=date, execution=pd.NaT, status='held', reason=reason))
            continue
        if not isinstance(candidate, pd.Series):
            raise TypeError('policy must return a labelled Series, wrapper tuple or None')
        if not candidate.index.is_unique or set(candidate.index) != set(prices.columns):
            raise ValueError('target labels must match every price asset exactly')
        target = candidate.reindex(prices.columns).astype(float).copy()
        if (not np.isfinite(target.to_numpy()).all() or (target < 0).any()
                or target.sum() > 1.0 + 1e-8):
            raise ValueError('targets must be finite long-only with sum at most one')
        frozen = ~state.quoted
        if not np.allclose(target[frozen], state.weights[frozen], rtol=0, atol=1e-8):
            raise ValueError('unquoted assets must retain their actual weight')
        if (target[state.quoted & ~state.eligible].abs() > 1e-8).any():
            raise ValueError('quoted ineligible assets require zero targets')
        targets[date] = target
        schedule[execution] = target
        events.append(dict(decision=date, execution=execution, status='scheduled', reason=''))
    portfolio = _replay(prices, schedule, initial_nav, transaction_costs)
    return ExecutedRollingResult(
        portfolio, pd.DataFrame.from_dict(targets, orient='index').reindex(columns=prices.columns),
        tuple(states), pd.DataFrame(events).set_index('decision'))
