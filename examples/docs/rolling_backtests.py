"""Canonical script of docs/rolling_backtests.md.

The page's two Python blocks are excerpts of ``main`` and run here in the same order; every
number and property the page states is asserted after them against a reference computed a
different way: a hand-held ledger of units and cash, currency positions for the drift identity,
an explicit EWMA recursion over monthly log returns, an independent SciPy SLSQP minimum-variance
solve at every decision date, and a shock to later prices. ``main`` runs offline after
``pip install optimalportfolios`` and needs no data file or random seed:

    python -m examples.docs.rolling_backtests

``exhibit`` draws the page's figure from a seeded synthetic panel;
``tools/docs_analytics/teaching.py`` calls it with the constants below and records their values.
"""
from contextlib import contextmanager
import inspect
from unittest.mock import patch
import warnings

import numpy as np
import pandas as pd

# The exhibit's synthetic panel: business-day prices with these annual log drifts, volatilities
# and correlations, and the page's minimum-variance path traded quarterly with lag one.
EXHIBIT_ASSETS = ['Equity', 'Bonds', 'Diversifier']
EXHIBIT_DRIFTS = [0.08, 0.02, 0.04]
EXHIBIT_VOLS = [0.18, 0.05, 0.10]
EXHIBIT_CORR = [[1.0, -0.2, 0.3],
                [-0.2, 1.0, 0.1],
                [0.3, 0.1, 1.0]]
EXHIBIT_DATES = ['2019-01-01', '2021-12-31']
EXHIBIT_DECISIONS = ['2019-12-01', '2021-12-31']
EXHIBIT_RETURNS_FREQ = 'W-WED'
EXHIBIT_SPAN = 26
EXHIBIT_MAX_WEIGHT = 0.60
EXHIBIT_LAG = 1
SEED = 3


def ledger(prices: pd.DataFrame, targets: pd.DataFrame, lag: int, rate: float,
           initial_nav: float = 100.0) -> tuple:
    """Hold units and cash by hand, trading each target ``lag`` observations after its date.

    A target dated between two price observations maps to the next one before the lag is
    added. Each trade is sized on the NAV just before it and pays ``rate`` times the absolute
    traded notional out of cash.

    Returns:
        Units, NAV, cash and costs, all on the price index.
    """
    execution = prices.index[prices.index.searchsorted(targets.index) + lag]
    orders = dict(zip(execution, targets[prices.columns].to_numpy()))
    units, cash = np.zeros(prices.shape[1]), initial_nav
    held, navs, cashes, costs = [], [], [], []
    for date, price in zip(prices.index, prices.to_numpy()):
        cost = np.zeros(prices.shape[1])
        if date in orders:
            nav_before = cash + units @ price
            traded = nav_before * orders[date] / price
            cost = rate * np.abs(traded - units) * price
            cash = nav_before - traded @ price - cost.sum()
            units = traded
        held.append(units)
        navs.append(cash + units @ price)
        cashes.append(cash)
        costs.append(cost)
    return (pd.DataFrame(held, index=prices.index, columns=prices.columns),
            pd.Series(navs, index=prices.index), pd.Series(cashes, index=prices.index),
            pd.DataFrame(costs, index=prices.index, columns=prices.columns))


def drift_by_positions(weights: pd.Series, start: pd.Series, end: pd.Series) -> pd.Series:
    """Drift weights by holding currency positions and uninvested cash between two price rows."""
    values = weights / start * end
    return values / (1.0 - weights.sum() + values.sum())


def ewma_covariance_reference(prices: pd.DataFrame, span: int, periods_per_year: int) -> dict:
    """Annualised EWMA covariance of EWMA-demeaned log returns, from the explicit recursion.

    The mean is pandas ``ewm(adjust=False)``, seeded with the first return; the first demeaned
    return, which is zero, is dropped; the covariance starts at zero.
    """
    log_returns = np.log(prices).diff().iloc[1:]
    deviations = (log_returns - log_returns.ewm(span=span, adjust=False).mean()).iloc[1:]
    decay = 1.0 - 2.0 / (span + 1.0)
    covar = np.zeros((prices.shape[1], prices.shape[1]))
    covars = {}
    for date, row in deviations.iterrows():
        covar = decay * covar + (1.0 - decay) * np.outer(row, row)
        covars[date] = pd.DataFrame(periods_per_year * covar, index=prices.columns,
                                    columns=prices.columns)
    return covars


def min_variance_reference(covar: pd.DataFrame, max_weight: float) -> np.ndarray:
    """Minimise w' Sigma w over fully invested weights in [0, max_weight] with SciPy SLSQP."""
    from scipy.optimize import minimize

    sigma = covar.to_numpy() / np.max(np.diag(covar))  # rescaled so ftol is meaningful
    n = len(sigma)
    result = minimize(lambda w: w @ sigma @ w, np.full(n, 1.0 / n),
                      jac=lambda w: 2.0 * sigma @ w, bounds=[(0.0, max_weight)] * n,
                      constraints=[{'type': 'eq', 'fun': lambda w: w.sum() - 1.0}],
                      method='SLSQP', options={'ftol': 1e-15, 'maxiter': 500})
    assert result.success, result.message
    return result.x


@contextmanager
def recorded_solves():
    """Record the baseline ``weights_0`` and the outcome of every single-date quadratic solve."""
    from optimalportfolios.optimization.general import quadratic

    original = quadratic.wrapper_quadratic_optimisation
    records = []

    def record(*args, **kwargs):
        """Forward the real call and keep its baseline and its returned outcome."""
        weights, outcome = original(*args, **kwargs)
        records.append((kwargs.get('weights_0'), outcome))
        return weights, outcome

    with patch.object(quadratic, 'wrapper_quadratic_optimisation', record):
        yield records


@contextmanager
def expected_warning(match: str):
    """Fail unless the enclosed code emits a UserWarning whose message contains ``match``."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        yield
    assert any(issubclass(w.category, UserWarning) and match in str(w.message)
               for w in caught), f'expected a warning containing {match!r}'


def assert_raises(error: type, match: str, function, *args, **kwargs) -> None:
    """Fail unless ``function(*args, **kwargs)`` raises ``error`` with ``match`` in its text."""
    try:
        function(*args, **kwargs)
    except error as exc:
        assert match in str(exc), str(exc)
        return
    raise AssertionError(f'expected {error.__name__}')


def exhibit_prices() -> pd.DataFrame:
    """Return the exhibit's seeded business-day panel of three total-return prices."""
    dates = pd.bdate_range(*EXHIBIT_DATES)
    vols = np.array(EXHIBIT_VOLS)
    daily_covar = np.outer(vols, vols) * np.array(EXHIBIT_CORR) / 260.0
    shocks = np.random.default_rng(SEED).standard_normal((len(dates), len(vols)))
    log_returns = (np.array(EXHIBIT_DRIFTS) / 260.0 - 0.5 * np.diag(daily_covar)
                   + shocks @ np.linalg.cholesky(daily_covar).T)
    log_returns[0] = 0.0
    return pd.DataFrame(100.0 * np.exp(np.cumsum(log_returns, axis=0)), index=dates,
                        columns=EXHIBIT_ASSETS)


def drift_path() -> dict:
    """Run the page's path on the exhibit panel and tabulate targets against held weights.

    The targets come from ``compute_rolling_optimal_weights`` and the holdings from a cost-free
    ``backtest_rolling_optimal_portfolio`` with lag one. The target in force at a date is the
    last executed target.

    Returns:
        The prices, dated targets, portfolio, execution dates and the plotted table.
    """
    import qis
    import optimalportfolios as opt

    prices = exhibit_prices()
    estimator = opt.EwmaCovarEstimator(returns_freq=EXHIBIT_RETURNS_FREQ, span=EXHIBIT_SPAN,
                                       rebalancing_freq='QE')
    covar_dict = estimator.fit_rolling_covars(prices=prices,
                                              time_period=qis.TimePeriod(*EXHIBIT_DECISIONS))
    constraints = opt.Constraints(is_long_only=True,
                                  max_weights=pd.Series(EXHIBIT_MAX_WEIGHT, index=prices.columns))
    targets = opt.compute_rolling_optimal_weights(
        prices=prices, constraints=constraints, covar_dict=covar_dict,
        portfolio_objective=opt.PortfolioObjective.MIN_VARIANCE)
    portfolio = opt.backtest_rolling_optimal_portfolio(
        prices=prices, constraints=constraints, covar_dict=covar_dict,
        portfolio_objective=opt.PortfolioObjective.MIN_VARIANCE, rebalancing_costs=0.0,
        weight_implementation_lag=EXHIBIT_LAG)
    executions = portfolio.is_rebalancing[portfolio.is_rebalancing].index
    in_force = pd.DataFrame(targets.to_numpy(), index=executions, columns=prices.columns)
    in_force = in_force.reindex(portfolio.weights.index, method='ffill').loc[executions[0]:]
    held = portfolio.weights.loc[executions[0]:]
    table = pd.concat([in_force.add_prefix('target: '), held.add_prefix('held: '),
                       (held - in_force).abs().sum(axis=1).rename('distance')], axis=1)
    return {'prices': prices, 'targets': targets, 'portfolio': portfolio,
            'executions': executions, 'table': table}


def ledger_weights(path: dict) -> pd.DataFrame:
    """Held weights of the exhibit path from the hand-held ledger: units times prices over NAV."""
    prices = path['prices'].loc[path['targets'].index[0]:]
    units, nav, _, _ = ledger(prices, path['targets'], lag=EXHIBIT_LAG, rate=0.0)
    return (units * prices).div(nav, axis=0)


def helper_weights(path: dict) -> pd.DataFrame:
    """Held weights of the exhibit path from ``apply_drift_to_weights_0`` anchored at each trade.

    Every date from the first trade on drifts the last executed target from its execution date.
    """
    import optimalportfolios as opt

    executions, prices, targets = path['executions'], path['prices'], path['targets']
    dates = path['table'].index
    last = executions.searchsorted(dates, side='right') - 1
    return pd.DataFrame([opt.apply_drift_to_weights_0(targets.iloc[k], prices, executions[k], date)
                         for k, date in zip(last, dates)], index=dates)


def main() -> None:
    """Run the page's blocks in order and assert every number and property it states."""
    # The second block's import makes np local to main; the checks after the first block need it.
    import numpy as np

    import pandas as pd
    import qis
    import optimalportfolios as opt

    toy_dates = pd.bdate_range("2024-01-02", periods=4)
    toy_prices = pd.DataFrame(
        {"A": [100.0, 110.0, 121.0, 133.1], "B": [100.0, 100.0, 100.0, 100.0]},
        index=toy_dates,
    )
    toy_targets = pd.DataFrame(
        [[0.5, 0.5], [0.5, 0.5]], index=toy_dates[[0, 2]], columns=toy_prices.columns
    )
    toy_portfolio = qis.backtest_model_portfolio(
        prices=toy_prices, weights=toy_targets, initial_nav=100.0,
        weight_implementation_lag=1, rebalancing_costs=0.0,
    )
    decision_baseline = opt.apply_drift_to_weights_0(
        weights_0=toy_targets.iloc[0], prices=toy_prices,
        prev_date=toy_dates[0], date=toy_dates[2],
    )

    # A rises 10% each observation and B is flat; targets are 50/50 on 2 and 4 January.
    np.testing.assert_allclose(toy_prices["A"].pct_change().iloc[1:], 0.10, rtol=0, atol=1e-12)
    assert (toy_prices["B"] == 100.0).all()
    assert toy_targets.index.strftime("%Y-%m-%d").tolist() == ["2024-01-02", "2024-01-04"]
    # Units: nothing on 2 January, 50/110 of A and 50/100 of B from 3 January, and a second trade
    # on 5 January sized on NAV 110.5, against the hand-held ledger and explicit fractions.
    units, nav, cash, costs = ledger(toy_prices, toy_targets, lag=1, rate=0.0)
    expected_units = np.array([[0, 0], [50 / 110, 50 / 100], [50 / 110, 50 / 100],
                               [55.25 / 133.1, 55.25 / 100]])
    np.testing.assert_allclose(toy_portfolio.units, expected_units, rtol=0, atol=1e-12)
    np.testing.assert_allclose(units, expected_units, rtol=0, atol=1e-12)
    np.testing.assert_allclose(toy_portfolio.nav, nav, rtol=0, atol=1e-12)
    assert toy_portfolio.is_rebalancing.tolist() == [False, True, False, True]
    pd.testing.assert_frame_equal(toy_portfolio.prices, toy_prices)
    assert (toy_portfolio.realized_costs == 0.0).all().all() and (costs == 0.0).all().all()
    # The event table: its dates and NAV; with no trade, NAV is B + sum(u P) with cash B = 0 here.
    displayed_nav = {"2024-01-02": 100.0, "2024-01-03": 100.0, "2024-01-04": 105.0,
                     "2024-01-05": 110.5}
    assert toy_dates.strftime("%Y-%m-%d").tolist() == list(displayed_nav)
    np.testing.assert_allclose(toy_portfolio.nav, list(displayed_nav.values()), rtol=0, atol=1e-12)
    np.testing.assert_allclose(nav, cash + (units * toy_prices).sum(axis=1), rtol=0, atol=1e-12)
    assert abs(cash.iloc[1]) < 1e-12 and cash.iloc[0] == 100.0
    # On 4 January the positions are worth 55 and 50: A's realised weight is 55/105. The
    # decision-date baseline assumes entry at A = 100 and gives 60.5/110.5, from currency
    # positions and from the NAV-growth identity.
    np.testing.assert_allclose(units.loc["2024-01-04"] * toy_prices.loc["2024-01-04"], [55, 50],
                               rtol=0, atol=1e-12)
    holdings = toy_portfolio.weights.loc["2024-01-04"]
    np.testing.assert_allclose(holdings, [55 / 105, 50 / 105], rtol=0, atol=1e-12)
    np.testing.assert_allclose(decision_baseline, [60.5 / 110.5, 50 / 110.5], rtol=0, atol=1e-12)
    np.testing.assert_allclose(
        decision_baseline,
        drift_by_positions(toy_targets.iloc[0], toy_prices.iloc[0], toy_prices.iloc[2]),
        rtol=0, atol=1e-12)
    returns = toy_prices.iloc[2] / toy_prices.iloc[0] - 1.0
    identity = toy_targets.iloc[0] * (1 + returns) / (1 + (toy_targets.iloc[0] * returns).sum())
    np.testing.assert_allclose(decision_baseline, identity, rtol=0, atol=1e-15)
    displayed = {"Decision-date drift baseline": (0.547511, 0.452489),
                 "Realised holdings with lag one": (0.523810, 0.476190)}
    np.testing.assert_allclose(displayed["Decision-date drift baseline"], decision_baseline,
                               rtol=0, atol=0.5e-6)
    np.testing.assert_allclose(displayed["Realised holdings with lag one"], holdings,
                               rtol=0, atol=0.5e-6)
    assert decision_baseline["A"] - holdings["A"] > 0.02
    # Anchored at the execution date, 3 January, the same helper reproduces the held weights.
    assert toy_portfolio.is_rebalancing.loc["2024-01-03"] and toy_dates[1] == pd.Timestamp(
        "2024-01-03")
    np.testing.assert_allclose(
        opt.apply_drift_to_weights_0(weights_0=toy_targets.iloc[0], prices=toy_prices,
                                     prev_date=toy_dates[1], date=toy_dates[2]),
        holdings, rtol=0, atol=1e-15)
    # The first 10% rise of A is not earned: NAV is still 100 on 3 January. A constant 50/50
    # weighted-return calculation from 3 January compounds to 110.25, not 110.5.
    assert toy_portfolio.nav.loc["2024-01-03"] == 100.0
    constant_mix = 100.0 * (1 + 0.5 * toy_prices["A"].pct_change().loc["2024-01-04":]).prod()
    assert abs(constant_mix - 110.25) < 1e-12 and abs(toy_portfolio.nav.iloc[-1] - 110.5) < 1e-12
    # The NAV-growth denominator keeps cash and short exposure: currency positions plus residual
    # cash reproduce the helper for a fully invested, a half-invested and a long-short prior.
    for prior in ([0.6, 0.4], [0.3, 0.2], [0.8, -0.2]):
        prior = pd.Series(prior, index=toy_prices.columns)
        np.testing.assert_allclose(
            opt.apply_drift_to_weights_0(prior, toy_prices, toy_dates[0], toy_dates[2]),
            drift_by_positions(prior, toy_prices.iloc[0], toy_prices.iloc[2]), rtol=0, atol=1e-12)
    # Dividing by the sum of risky positions instead would change a half-invested prior.
    half = pd.Series([0.3, 0.2], index=toy_prices.columns)
    grown = half * toy_prices.iloc[2] / toy_prices.iloc[0]
    assert abs((grown / grown.sum())["A"]
               - opt.apply_drift_to_weights_0(half, toy_prices, toy_dates[0], toy_dates[2])["A"]
               ) > 0.2
    # Drift fallbacks: no previous date or target, a zero target, an anchor before the prices
    # and NAV collapse return the prior unchanged; an asset without prices is treated as flat.
    prior = pd.Series([0.6, 0.4], index=toy_prices.columns)
    early = pd.Timestamp("2023-12-29")
    for weights_0, prev_date in ((prior, None), (None, toy_dates[0]), (prior * 0.0, toy_dates[0]),
                                 (prior, early)):
        unchanged = opt.apply_drift_to_weights_0(weights_0, toy_prices, prev_date, early)
        assert unchanged is weights_0
    crashed = toy_prices.assign(A=[100.0, 1.0, 1.0, 1.0])
    short = pd.Series([2.0, -1.0], index=toy_prices.columns)
    assert opt.apply_drift_to_weights_0(short, crashed, toy_dates[0], toy_dates[2]) is short
    unpriced = pd.Series([0.5, 0.3, 0.2], index=["A", "B", "Cash-like"])
    drifted = opt.apply_drift_to_weights_0(unpriced, toy_prices, toy_dates[0], toy_dates[2])
    np.testing.assert_allclose(drifted, np.array([0.5 * 1.21, 0.3, 0.2]) / (1 + 0.5 * 0.21),
                               rtol=0, atol=1e-15)
    # A missing price at an anchor is forward-filled from earlier history, never from later.
    stale = toy_prices.astype(float)
    stale.loc[toy_dates[2], "A"] = np.nan
    np.testing.assert_allclose(
        opt.apply_drift_to_weights_0(prior, stale, toy_dates[0], toy_dates[2]),
        drift_by_positions(prior, toy_prices.iloc[0], toy_prices.iloc[1]), rtol=0, atol=1e-15)
    # The toggle: use_drifted_weights_0=False returns the prior target unchanged.
    assert opt.apply_drift_to_weights_0(prior, toy_prices, toy_dates[0], toy_dates[2],
                                        use_drifted_weights_0=False) is prior
    assert opt.OptimiserConfig().use_drifted_weights_0 is True
    # Execution timing: a Saturday decision maps to Monday, then lag one trades on Tuesday.
    # None means zero lag.
    weekend = pd.DataFrame({"A": [100.0, 110.0, 121.0], "B": 100.0},
                           index=pd.to_datetime(["2024-01-05", "2024-01-08", "2024-01-09"]))
    saturday = pd.DataFrame([[0.5, 0.5]], index=pd.to_datetime(["2024-01-06"]),
                            columns=weekend.columns)
    mapped = qis.backtest_model_portfolio(prices=weekend, weights=saturday,
                                          weight_implementation_lag=1, rebalancing_costs=0.0)
    assert mapped.is_rebalancing.tolist() == [False, False, True]
    assert (mapped.nav == 100.0).all()
    no_lag = qis.backtest_model_portfolio(prices=toy_prices, weights=toy_targets,
                                          weight_implementation_lag=None)
    zero_lag = qis.backtest_model_portfolio(prices=toy_prices, weights=toy_targets,
                                            weight_implementation_lag=0)
    pd.testing.assert_series_equal(no_lag.nav, zero_lag.nav)
    assert no_lag.is_rebalancing.tolist() == [True, False, True, False]
    # Execution-grid boundaries: a target traded past the panel is dropped with a warning; no
    # executable target raises; two targets resolving to one observation raise.
    tail = pd.concat([toy_targets, toy_targets.iloc[[0]].set_axis([toy_dates[-1]])])
    with expected_warning("trade past the end"):
        dropped = qis.backtest_model_portfolio(prices=toy_prices, weights=tail,
                                               weight_implementation_lag=1)
    assert dropped.is_rebalancing.tolist() == [False, True, False, True]
    assert_raises(ValueError, "no weight date is traded", qis.backtest_model_portfolio,
                  prices=toy_prices, weights=tail.iloc[[-1]], weight_implementation_lag=1)
    assert_raises(ValueError, "resolve to", qis.backtest_model_portfolio,
                  prices=toy_prices.iloc[[0, 3]], weights=tail.iloc[[1, 2]])
    # A missing execution price leaves that target weight in cash; an interior hole in a held
    # asset's history removes its value from that observation's NAV.
    unpriced_entry = toy_prices.astype(float)
    unpriced_entry.loc[toy_dates[1], "B"] = np.nan
    with expected_warning("have no price on their traded date"):
        stuck = qis.backtest_model_portfolio(prices=unpriced_entry, weights=toy_targets.iloc[[0]],
                                             weight_implementation_lag=1)
    assert stuck.units.loc[toy_dates[1], "B"] == 0.0 and stuck.nav.loc[toy_dates[1]] == 100.0
    holed = toy_prices.astype(float)
    holed.loc[toy_dates[2], "B"] = np.nan
    with expected_warning("inside the reported history"):
        gapped = qis.backtest_model_portfolio(prices=holed, weights=toy_targets.iloc[[0]],
                                              weight_implementation_lag=1)
    assert abs(gapped.nav.loc[toy_dates[2]] - 55.0) < 1e-12

    import numpy as np
    import pandas as pd
    import qis
    import optimalportfolios as opt

    dates = pd.date_range("2020-01-31", periods=24, freq="ME")
    monthly_returns = np.array([
        [0.010, 0.004, 0.002], [0.015, -0.003, 0.003],
        [-0.008, 0.006, 0.002], [0.012, 0.001, -0.001],
    ] * 6)
    prices = pd.DataFrame(
        100.0 * np.cumprod(1.0 + monthly_returns, axis=0),
        index=dates,
        columns=["Equity", "Bonds", "Diversifier"],
    )
    estimator = opt.EwmaCovarEstimator(
        returns_freq="ME", span=6, rebalancing_freq="QE"
    )
    covar_dict = estimator.fit_rolling_covars(
        prices=prices,
        time_period=qis.TimePeriod("31Dec2020", "30Sep2021"),
    )
    constraints = opt.Constraints(
        is_long_only=True,
        max_weights=pd.Series(0.80, index=prices.columns),
    )
    weights = opt.compute_rolling_optimal_weights(
        prices=prices,
        constraints=constraints,
        covar_dict=covar_dict,
        portfolio_objective=opt.PortfolioObjective.MIN_VARIANCE,
    )
    portfolio = opt.backtest_rolling_optimal_portfolio(
        prices=prices,
        constraints=constraints,
        covar_dict=covar_dict,
        portfolio_objective=opt.PortfolioObjective.MIN_VARIANCE,
        rebalancing_costs=0.0003,  # 3 bp of traded notional
        weight_implementation_lag=1,
        ticker="Minimum variance",
    )

    # 24 month ends from simple returns that repeat a four-month pattern; the first decision
    # has 11 monthly returns of history behind it, and the panel runs through December 2021.
    assert len(prices) == 24 and prices.index[-1] == pd.Timestamp("2021-12-31")
    np.testing.assert_allclose(prices.pct_change().iloc[1:], monthly_returns[1:], rtol=0,
                               atol=1e-14)
    assert (monthly_returns[4:] == monthly_returns[:-4]).all()
    assert len(prices.loc[:"2020-12-31"]) - 1 == 11
    # Four quarterly decisions, each an EWMA of EWMA-demeaned monthly log returns with span six,
    # annualised by 12; the matrices are keyed by the displayed decision dates.
    decision_dates = ["2020-12-31", "2021-03-31", "2021-06-30", "2021-09-30"]
    assert [date.strftime("%Y-%m-%d") for date in covar_dict] == decision_dates
    reference = ewma_covariance_reference(prices, span=6, periods_per_year=12)
    for date, covar in covar_dict.items():
        pd.testing.assert_frame_equal(covar, reference[date], rtol=0, atol=1e-15)
    # One target row per decision date and one column per asset: fully invested, long-only,
    # within the 80% cap, and at the minimum variance of an independent SLSQP solve. These
    # covariances are of order 1e-5, so the solver's default tolerances stop within 3e-4 of the
    # SLSQP weights while matching their variance to about one part in a million.
    pd.testing.assert_index_equal(weights.index, pd.DatetimeIndex(list(covar_dict)))
    pd.testing.assert_index_equal(weights.columns, prices.columns)
    np.testing.assert_allclose(weights.sum(axis=1), 1.0, rtol=0, atol=1e-8)
    assert (weights >= -1e-8).all().all() and (weights <= 0.80 + 1e-8).all().all()
    for date, covar in covar_dict.items():
        exact, sigma = min_variance_reference(covar, 0.80), covar.to_numpy()
        np.testing.assert_allclose(weights.loc[date], exact, rtol=0, atol=3e-4)
        solved = weights.loc[date].to_numpy() @ sigma @ weights.loc[date].to_numpy()
        assert 0.0 <= solved / (exact @ sigma @ exact) - 1.0 < 1e-5
    # Computing weights and then calling the convenience backtester solves twice: four real
    # solves each, all accepted and compliant, with no fallback and no floored eigenvalue.
    # The rolling solver's baseline is the previous target drifted between the two decision
    # dates, not the executed holdings; with the toggle off it is the previous target.
    with recorded_solves() as solves:
        rerun = opt.compute_rolling_optimal_weights(
            prices=prices, constraints=constraints, covar_dict=covar_dict,
            portfolio_objective=opt.PortfolioObjective.MIN_VARIANCE)
        opt.backtest_rolling_optimal_portfolio(
            prices=prices, constraints=constraints, covar_dict=covar_dict,
            portfolio_objective=opt.PortfolioObjective.MIN_VARIANCE, rebalancing_costs=0.0003,
            weight_implementation_lag=1)
    assert len(solves) == 8
    assert all(outcome.accepted and outcome.compliant and outcome.fallback_source is None
               and outcome.covar_factorization.n_eigenvalues_floored == 0
               for _, outcome in solves)
    # The convention card's solver: CVXPY with CLARABEL on the eigendecomposed covariance, the
    # OptimiserConfig defaults.
    assert opt.OptimiserConfig().solver == "CLARABEL" and opt.OptimiserConfig().factorize_covar
    assert all(outcome.solver == "CLARABEL" and outcome.covar_factorization is not None
               for _, outcome in solves)
    pd.testing.assert_frame_equal(rerun, weights)
    baselines = [baseline for baseline, _ in solves[:4]]
    assert baselines[0] is None
    for k in range(1, 4):
        previous, current = weights.index[k - 1], weights.index[k]
        np.testing.assert_allclose(
            baselines[k], drift_by_positions(weights.iloc[k - 1], prices.loc[previous],
                                             prices.loc[current]), rtol=0, atol=1e-14)
    with recorded_solves() as undrifted:
        toggled = opt.compute_rolling_optimal_weights(
            prices=prices, constraints=constraints, covar_dict=covar_dict,
            portfolio_objective=opt.PortfolioObjective.MIN_VARIANCE,
            optimiser_config=opt.OptimiserConfig(apply_total_to_good_ratio=True,
                                                 use_drifted_weights_0=False))
    for k in range(1, 4):
        pd.testing.assert_series_equal(undrifted[k][0], weights.iloc[k - 1], check_names=False)
    np.testing.assert_allclose(toggled, weights, rtol=0, atol=1e-10)
    # The dispatcher's returns_freq and span do not recompute a supplied covariance, and the
    # maximum-CARA-mixture solver takes no covariance dictionary.
    np.testing.assert_allclose(
        opt.compute_rolling_optimal_weights(
            prices=prices, constraints=constraints, covar_dict=covar_dict,
            portfolio_objective=opt.PortfolioObjective.MIN_VARIANCE, returns_freq="QE", span=2),
        weights, rtol=0, atol=1e-12)
    from optimalportfolios.optimization.general.carra_mixture import (
        rolling_maximize_cara_mixture)
    assert "covar_dict" not in inspect.signature(rolling_maximize_cara_mixture).parameters
    # A monthly lag of one observation trades one month later, at the displayed dates.
    assert isinstance(portfolio, qis.PortfolioData) and hasattr(portfolio, "get_turnover")
    executions = portfolio.is_rebalancing[portfolio.is_rebalancing].index
    assert executions.strftime("%Y-%m-%d").tolist() == [
        "2021-01-31", "2021-04-30", "2021-07-31", "2021-10-31"]
    assert (executions == prices.index[prices.index.searchsorted(weights.index) + 1]).all()
    # The backtest starts in cash at the first target date with NAV 100; entry in January costs
    # 3 bp of the 100 traded, leaving NAV 99.97, and earns nothing from December to January.
    assert portfolio.nav.index[0] == pd.Timestamp("2020-12-31") and portfolio.nav.iloc[0] == 100.0
    assert (portfolio.units.iloc[0] == 0.0).all()
    assert abs(portfolio.nav.loc["2021-01-31"] - 99.97) < 1e-10
    assert abs(portfolio.realized_costs.loc["2021-01-31"].sum() - 0.03) < 1e-12
    pd.testing.assert_frame_equal(portfolio.prices, prices.loc[portfolio.prices.index])
    # Units, NAV and costs over the whole path from the hand-held ledger, costs from the
    # absolute units traded at execution prices; qis sizes on pre-cost NAV, so post-cost
    # weights right after entry are the targets scaled by 100 / 99.97.
    units, nav, cash, costs = ledger(prices.loc["2020-12-31":], weights, lag=1, rate=0.0003)
    np.testing.assert_allclose(portfolio.units, units, rtol=0, atol=1e-12)
    np.testing.assert_allclose(portfolio.nav, nav, rtol=0, atol=1e-10)
    np.testing.assert_allclose(portfolio.realized_costs, costs, rtol=0, atol=1e-14)
    traded = portfolio.units.diff().fillna(portfolio.units).abs() * portfolio.prices * 0.0003
    np.testing.assert_allclose(portfolio.realized_costs, traded, rtol=0, atol=1e-14)
    np.testing.assert_allclose(portfolio.weights.loc["2021-01-31"],
                               weights.iloc[0] * 100.0 / 99.97, rtol=0, atol=1e-12)
    assert abs(portfolio.weights.loc["2021-01-31"] - weights.iloc[0]).max() > 1e-6
    # The October target is carried to December with fixed units.
    assert (portfolio.units.loc["2021-10-31":].nunique() == 1).all()
    # Targets already in hand go straight to qis with the same result.
    direct = qis.backtest_model_portfolio(prices=prices.loc["2020-12-31":], weights=weights,
                                          rebalancing_costs=0.0003, weight_implementation_lag=1)
    np.testing.assert_allclose(direct.nav, portfolio.nav, rtol=0, atol=1e-12)
    # Later prices cannot move earlier estimates or targets.
    shocked = prices.copy()
    later = shocked.index > pd.Timestamp("2021-06-30")
    shocked.loc[later, "Equity"] *= np.linspace(1.5, 4.0, later.sum())
    shocked_covars = estimator.fit_rolling_covars(
        shocked, qis.TimePeriod("31Dec2020", "30Sep2021"))
    for date in decision_dates[:3]:
        pd.testing.assert_frame_equal(shocked_covars[pd.Timestamp(date)],
                                      covar_dict[pd.Timestamp(date)], rtol=0, atol=1e-12)
    shocked_weights = opt.compute_rolling_optimal_weights(
        prices=shocked, constraints=constraints, covar_dict=shocked_covars,
        portfolio_objective=opt.PortfolioObjective.MIN_VARIANCE)
    np.testing.assert_allclose(shocked_weights.iloc[:3], weights.iloc[:3], rtol=0, atol=1e-10)
    assert abs(shocked_weights.iloc[3] - weights.iloc[3]).max() > 1e-3
    # perf_time_period filters targets before simulation: the backtest restarts in cash at the
    # first retained target and trades on 30 April, while the full path already holds all three
    # assets on 31 March at NAV 100.41.
    filtered = opt.backtest_rolling_optimal_portfolio(
        prices=prices, constraints=constraints, covar_dict=covar_dict,
        perf_time_period=qis.TimePeriod("31Mar2021", "30Sep2021"),
        portfolio_objective=opt.PortfolioObjective.MIN_VARIANCE,
        rebalancing_costs=0.0003, weight_implementation_lag=1)
    assert filtered.nav.index[0] == pd.Timestamp("2021-03-31") and filtered.nav.iloc[0] == 100.0
    assert (filtered.units.iloc[0] == 0.0).all()
    assert filtered.is_rebalancing[filtered.is_rebalancing].index[0] == pd.Timestamp("2021-04-30")
    assert round(portfolio.nav.loc["2021-03-31"], 2) == 100.41
    assert (portfolio.units.loc["2021-03-31"] > 0.0).all()
    # round_weights_to_pct: an equal three-asset allocation rounds naively to 33.33% each, which
    # sums to 99.99%; the largest-remainder method reports 33.34%, 33.33% and 33.33%. Every
    # target row of the monthly example, rounded, sums to 100 and matches floors to whole basis
    # points with the shortfall handed to the largest remainders.
    equal = pd.Series(1.0 / 3.0, index=prices.columns)
    assert (100.0 * equal).round(2).tolist() == [33.33] * 3
    assert round((100.0 * equal).round(2).sum(), 2) == 99.99
    assert opt.round_weights_to_pct(equal).tolist() == [33.34, 33.33, 33.33]
    for _, row in pd.concat([weights, equal.to_frame().T]).iterrows():
        percent = opt.round_weights_to_pct(row)
        assert round(percent.sum(), 10) == 100.0
        points = 10_000.0 * row.to_numpy()
        basis = np.floor(points)
        order = np.argsort(-(points - basis), kind="stable")
        basis[order[:int(round(10_000 - basis.sum()))]] += 1
        np.testing.assert_allclose(percent, basis / 100.0, rtol=0, atol=1e-12)

    # The toy insight: anchored at the decision date, the baseline overstates A by 2.4 points.
    assert round(100.0 * (decision_baseline["A"] - holdings["A"]), 1) == 2.4

    # The figure: two years of business days from January 2020, eight quarterly trades, each
    # one observation after its decision, Bonds at its 60% cap throughout, cost-free holdings
    # equal to the targets at every trade, and a largest distance of 5.1 points on 6 January
    # 2021, the day before a trade. Held weights come from the backtester and, independently,
    # from the hand-held ledger; the drift helper anchored at each trade reproduces them on
    # every day in between.
    path = drift_path()
    table, executions, targets = path["table"], path["executions"], path["targets"]
    figure_prices = path["prices"]
    distance = table["distance"]
    held = table.filter(like="held: ").set_axis(EXHIBIT_ASSETS, axis=1)
    in_force = table.filter(like="target: ").set_axis(EXHIBIT_ASSETS, axis=1)
    assert len(executions) == 8 and executions[0] == pd.Timestamp("2020-01-02")
    assert table.index[-1] == pd.Timestamp("2021-12-31")
    # Each decision is the first Wednesday of the weekly return grid on or after a quarter end.
    quarter_ends = pd.DatetimeIndex([pd.offsets.QuarterEnd().rollback(date)
                                     for date in targets.index])
    assert (targets.index.dayofweek == 2).all() and ((targets.index - quarter_ends).days < 7).all()
    assert quarter_ends.strftime("%Y-%m-%d").tolist() == [
        "2019-12-31", "2020-03-31", "2020-06-30", "2020-09-30", "2020-12-31", "2021-03-31",
        "2021-06-30", "2021-09-30"]
    assert (executions == figure_prices.index[
        figure_prices.index.searchsorted(targets.index) + 1]).all()
    assert (targets["Bonds"] - EXHIBIT_MAX_WEIGHT).abs().max() < 1e-6
    independent = ledger_weights(path)
    np.testing.assert_allclose(independent, path["portfolio"].weights, rtol=0, atol=1e-12)
    np.testing.assert_allclose(independent.loc[executions], targets, rtol=0, atol=1e-12)
    np.testing.assert_allclose(helper_weights(path), held, rtol=0, atol=1e-12)
    reference_distance = (independent.loc[table.index] - in_force).abs().sum(axis=1)
    np.testing.assert_allclose(distance, reference_distance, rtol=0, atol=1e-12)
    assert (distance.loc[executions] < 1e-12).all() and (distance.drop(executions) > 0).all()
    assert round(100.0 * distance.max(), 1) == 5.1
    assert distance.idxmax() == pd.Timestamp("2021-01-06")
    assert figure_prices.index[figure_prices.index.get_loc("2021-01-06") + 1] in executions
    print("rolling_backtests: all page statements verified.")


def exhibit(path) -> dict:
    """Draw the page's figure: targets against drifted holdings, and the distance between them.

    Args:
        path: PNG file to write.

    Returns:
        The plotted table and the checks the figure illustrates.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    run = drift_path()
    prices, targets, portfolio = run['prices'], run['targets'], run['portfolio']
    executions, table = run['executions'], run['table']
    held = table.filter(like='held: ').set_axis(EXHIBIT_ASSETS, axis=1)
    target = table.filter(like='target: ').set_axis(EXHIBIT_ASSETS, axis=1)
    distance = table['distance']

    ink, muted, grid, surface = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
    colours = dict(zip(EXHIBIT_ASSETS, ['#2a78d6', '#eb6834', '#1baf7a']))
    plt.rcParams.update({'font.size': 11, 'axes.edgecolor': grid, 'axes.labelcolor': muted,
                         'xtick.color': muted, 'ytick.color': muted})
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.0, 4.4), facecolor=surface,
                                      gridspec_kw={'width_ratios': [1.45, 1.0]})
    for axis in (left, right):
        for date in executions:
            axis.axvline(date, color='#b9b8b3', linewidth=0.8, zorder=0)
    for asset in EXHIBIT_ASSETS:
        left.plot(target.index, 100.0 * target[asset], color=colours[asset], linewidth=1.2,
                  linestyle='--', drawstyle='steps-post')
        left.plot(held.index, 100.0 * held[asset], color=colours[asset], linewidth=1.6)
        left.text(held.index[-1] + pd.Timedelta(days=12), 100.0 * held[asset].iloc[-1], asset,
                  color=ink, fontsize=9, va='center')
    left.plot([], [], color=muted, linewidth=1.6, label='held (drifted)')
    left.plot([], [], color=muted, linewidth=1.2, linestyle='--', label='target in force')
    left.plot([], [], color='#b9b8b3', linewidth=0.8, label='quarterly trade')
    left.legend(loc='center left', bbox_to_anchor=(0.0, 0.68), fontsize=9, labelcolor=ink,
                facecolor=surface, edgecolor='none', framealpha=1.0)
    left.set_ylabel('weight, % of NAV')
    left.set_title('Target and held weights', loc='left', color=ink)
    # Start just before the first trade so that its grey line shows.
    first = held.index[0] - pd.Timedelta(days=20)
    left.set_xlim(first, held.index[-1] + pd.Timedelta(days=110))

    # One summary series, not an asset: drawn in ink so it is not read as a fourth entity.
    right.plot(distance.index, 100.0 * distance, color=ink, linewidth=1.2)
    peak = distance.idxmax()
    right.text(peak - pd.Timedelta(days=12), 100.0 * distance.max(),
               f'{100.0 * distance.max():.1f} points on {peak.day} {peak:%b %Y},\n'
               'the day before a trade',
               color=ink, fontsize=9, ha='right', va='top',
               bbox={'facecolor': surface, 'edgecolor': 'none', 'pad': 1.0})
    right.set_ylabel('sum of |held - target|, % points')
    right.set_title('Distance from target', loc='left', color=ink)
    right.set_ylim(bottom=0.0)
    right.set_xlim(first, distance.index[-1])
    for axis in (left, right):
        axis.set_facecolor(surface)
        axis.grid(axis='y', color=grid, linewidth=0.8)
        axis.set_axisbelow(True)
        axis.xaxis.set_major_locator(matplotlib.dates.MonthLocator(bymonth=(1, 7)))
        axis.xaxis.set_major_formatter(matplotlib.dates.DateFormatter('%b %y'))
        for side in ('top', 'right'):
            axis.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=surface)
    plt.close(fig)

    units, _, _, _ = ledger(prices.loc[targets.index[0]:], targets, lag=EXHIBIT_LAG, rate=0.0)
    independent = ledger_weights(run)
    checks = {
        'ledger_units_match_backtester': bool(np.allclose(units, portfolio.units, rtol=0.0,
                                                          atol=1e-10)),
        'weights_from_units_and_prices_match_holdings': bool(np.allclose(
            independent, portfolio.weights, rtol=0.0, atol=1e-12)),
        'weights_from_units_and_prices_equal_targets_after_each_trade': bool(np.allclose(
            independent.loc[executions], targets, rtol=0.0, atol=1e-12)),
        'drift_helper_anchored_at_each_trade_matches_holdings': bool(np.allclose(
            helper_weights(run), held, rtol=0.0, atol=1e-12)),
        'each_trade_one_observation_after_its_decision': bool(
            (executions == prices.index[prices.index.searchsorted(targets.index)
                                        + EXHIBIT_LAG]).all()),
        'distance_zero_at_trades_and_positive_between': bool(
            (distance.loc[executions] < 1e-12).all() and (distance.drop(executions) > 0.0).all()),
        'peak_distance_on_the_day_before_a_trade': bool(
            prices.index[prices.index.get_loc(distance.idxmax()) + 1] in executions),
    }
    return {'table': table, 'checks': checks}


if __name__ == '__main__':
    main()
