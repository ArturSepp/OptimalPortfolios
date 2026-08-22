"""
pin the quantified claims of the JOSS manuscript (``paper.md``).

The paper's statement of need quotes one measured effect on the wheel-shipped fixture: a
quarterly minimum-variance backtest with a hard per-rebalance L1 turnover budget of 3%, where
the optimiser measures turnover against the previous *target* weights
(``use_drifted_weights_0=False``), reports compliance at every rebalance while the executed
trades breach the budget at about 71% of rebalances, by up to 2.4 times the budget. Under the
drift-aware default (``use_drifted_weights_0=True``) the executed trades never exceed it.

This file recomputes that experiment end to end and fails when the manuscript's numbers stop
being true. Executed turnover is verified by two independent routes (the D4 second pass):

    the ``qis`` simulator
        ``qis.backtest_model_portfolio`` holds units between rebalancings and reports the
        traded fraction of NAV from its own state; it knows nothing of
        ``apply_drift_to_weights_0``, the code whose effect is under test.

    an inline drifted-L1 recomputation
        target weights are drifted from first principles (price relatives over the holding
        period, renormalised by portfolio growth) and the L1 distance to the next target is
        taken directly. Written out here rather than calling ``apply_drift_to_weights_0``
        precisely because that function sits on the code path under test.

Assertions use bands rather than exact pins where a floating solver version could move the
third decimal (breach share, peak ratio); the paper quotes the values measured on the locked
primary CI cell, and the bands are tight enough that the quoted "71%" and "2.4 times" remain
honest anywhere inside them. The rebalance count is pinned exactly: it depends only on the
frozen fixture, so a change is a reviewed fixture change, not solver noise.
"""
# packages
import numpy as np
import pandas as pd
import pytest
from typing import Dict, Tuple
# qis / project
import qis
from optimalportfolios import (Constraints, EwmaCovarEstimator, OptimiserConfig,
                               PortfolioObjective, rolling_quadratic_optimisation)
from optimalportfolios.tests.data.multiasset import load_multiasset_data

TURNOVER_BUDGET = 0.03  # hard L1 budget per quarterly rebalance, fraction of NAV
NUMERICAL_TOL = 1.0e-3  # relative slack on "respects the budget" for solver tolerance
ROUTE_AGREEMENT_TOL = 1.0e-4  # max |qis route - inline route| per rebalance


def _drifted_l1_turnover(weights: pd.DataFrame,
                         prices: pd.DataFrame,
                         ) -> pd.Series:
    """recompute executed turnover from first principles, independently of the package.

    Each previous target is drifted over the holding period using price relatives and
    renormalised by realised portfolio growth; the executed trade is the L1 distance from the
    drifted weights to the new target. ``apply_drift_to_weights_0`` is deliberately not called:
    it is the implementation whose effect this file verifies.

    Args:
        weights: target weights by rebalance date, columns of instruments.
        prices: total-return price panel covering the weight dates.

    Returns:
        Executed L1 turnover per rebalance date; the first date is NaN.
    """
    executed = pd.Series(index=weights.index, dtype=float)
    prev_date, prev_w = None, None
    for date, w_new in weights.iterrows():
        if prev_w is None:
            prev_date, prev_w = date, w_new
            continue
        p0 = prices.loc[:prev_date].ffill().iloc[-1].reindex(w_new.index)
        p1 = prices.loc[:date].ffill().iloc[-1].reindex(w_new.index)
        ratio = (p1 / p0.where(p0 > 0.0)).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        nav_growth = 1.0 + float((prev_w * (ratio - 1.0)).sum())
        drifted = prev_w if nav_growth < 1e-12 else prev_w * ratio / nav_growth
        executed.loc[date] = float((w_new - drifted).abs().sum())
        prev_date, prev_w = date, w_new
    return executed


def _run_policy(prices: pd.DataFrame,
                covar_dict: Dict[pd.Timestamp, pd.DataFrame],
                use_drifted_weights_0: bool,
                ) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """run one rolling min-variance backtest and measure turnover three ways.

    Args:
        prices: fixture price panel.
        covar_dict: rolling covariances shared by both policies.
        use_drifted_weights_0: the ``OptimiserConfig`` drift policy under test.

    Returns:
        Apparent turnover (L1 between consecutive targets), executed turnover from the ``qis``
        simulator, and executed turnover from the inline recomputation, each per rebalance
        date with the first date dropped.
    """
    constraints = Constraints(is_long_only=True, turnover_constraint=TURNOVER_BUDGET)
    weights = rolling_quadratic_optimisation(
        prices=prices,
        constraints=constraints,
        covar_dict=covar_dict,
        portfolio_objective=PortfolioObjective.MIN_VARIANCE,
        optimiser_config=OptimiserConfig(use_drifted_weights_0=use_drifted_weights_0),
    )
    apparent = weights.diff().abs().sum(axis=1).iloc[1:]
    # zero costs and no implementation lag so the simulator's traded fraction of NAV at each
    # rebalance date is directly comparable to the inline drifted-L1 quantity
    portfolio_data = qis.backtest_model_portfolio(prices=prices,
                                                  weights=weights,
                                                  rebalancing_costs=0.0,
                                                  ticker='policy')
    simulator = portfolio_data.get_turnover(roll_period=None, freq=None)['policy']
    executed_qis = simulator.reindex(apparent.index)
    executed_inline = _drifted_l1_turnover(weights, prices).iloc[1:]
    return apparent, executed_qis, executed_inline


@pytest.fixture(scope='module')
def turnover_measurements() -> Dict[bool, Tuple[pd.Series, pd.Series, pd.Series]]:
    """measure both drift policies once on the shipped fixture, shared across tests.

    Returns:
        Mapping from ``use_drifted_weights_0`` to the three per-rebalance turnover series
        of :func:`_run_policy`.
    """
    data = load_multiasset_data()
    prices = data.prices
    time_period = qis.TimePeriod(prices.index[0], prices.index[-1])
    covar_dict = EwmaCovarEstimator(returns_freq='ME', span=24,
                                    rebalancing_freq='QE').fit_rolling_covars(
        prices=prices, time_period=time_period)
    return {policy: _run_policy(prices=prices, covar_dict=covar_dict,
                                use_drifted_weights_0=policy)
            for policy in (False, True)}


def test_experiment_shape_is_the_frozen_fixture(
        turnover_measurements: Dict[bool, Tuple[pd.Series, pd.Series, pd.Series]]) -> None:
    """the experiment runs on 95 post-initial quarterly rebalances of the frozen fixture.

    Args:
        turnover_measurements: per-policy turnover series from the module fixture.
    """
    for policy, (apparent, executed_qis, executed_inline) in turnover_measurements.items():
        assert len(apparent) == 95, (
            f"policy {policy}: expected 95 rebalance intervals on the frozen fixture, "
            f"got {len(apparent)} — the fixture or schedule changed, re-derive the paper numbers"
        )
        assert executed_qis.notna().all(), f"policy {policy}: simulator turnover has NaNs"
        assert executed_inline.notna().all(), f"policy {policy}: inline turnover has NaNs"


def test_both_executed_turnover_routes_agree(
        turnover_measurements: Dict[bool, Tuple[pd.Series, pd.Series, pd.Series]]) -> None:
    """the qis-simulator route and the inline drifted-L1 route give the same executed trades.

    Args:
        turnover_measurements: per-policy turnover series from the module fixture.
    """
    for policy, (_, executed_qis, executed_inline) in turnover_measurements.items():
        gap = (executed_qis - executed_inline).abs().max()
        assert gap < ROUTE_AGREEMENT_TOL, (
            f"policy {policy}: executed-turnover routes disagree by {gap:.2e}"
        )


def test_legacy_policy_breaches_budget_as_quoted(
        turnover_measurements: Dict[bool, Tuple[pd.Series, pd.Series, pd.Series]]) -> None:
    """under ``use_drifted_weights_0=False`` the paper's breach numbers hold.

    The optimiser's apparent turnover respects the budget at every rebalance, while executed
    trades breach it at about 71% of rebalances (band 60-80%) with a peak of about 2.4 times
    the budget (band 2.0-2.8) and a mean above the budget.

    Args:
        turnover_measurements: per-policy turnover series from the module fixture.
    """
    apparent, executed_qis, _ = turnover_measurements[False]
    budget_ceiling = TURNOVER_BUDGET * (1.0 + NUMERICAL_TOL)
    assert apparent.max() <= budget_ceiling, (
        f"apparent turnover {apparent.max():.4f} exceeds the budget the optimiser enforces"
    )
    breach_share = float((executed_qis > budget_ceiling).mean())
    assert 0.60 <= breach_share <= 0.80, (
        f"executed-breach share {breach_share:.3f} left the quoted band around 71%"
    )
    peak_ratio = float(executed_qis.max() / TURNOVER_BUDGET)
    assert 2.0 <= peak_ratio <= 2.8, (
        f"peak executed/budget ratio {peak_ratio:.2f} left the quoted band around 2.4"
    )
    assert float(executed_qis.mean()) > TURNOVER_BUDGET, (
        "mean executed turnover no longer exceeds the budget under the legacy policy"
    )


def test_drift_aware_policy_respects_budget(
        turnover_measurements: Dict[bool, Tuple[pd.Series, pd.Series, pd.Series]]) -> None:
    """under the drift-aware default the executed trades never exceed the budget.

    Args:
        turnover_measurements: per-policy turnover series from the module fixture.
    """
    _, executed_qis, _ = turnover_measurements[True]
    budget_ceiling = TURNOVER_BUDGET * (1.0 + NUMERICAL_TOL)
    assert float(executed_qis.max()) <= budget_ceiling, (
        f"executed turnover {executed_qis.max():.4f} exceeds the budget under the "
        "drift-aware default"
    )
