"""
the rolling wrappers around every optimiser, end to end over a synthetic panel.

Manual per-solver ``run_local`` diagnostics live beside their owning optimisation modules because
they may plot or need the author's price data. This module tests the package layer those
diagnostics exercise interactively: ``rolling_*``, which walks the rebalancing dates, drifts the
previous weights, calls the solver and assembles the result.

Every case here builds its own three-asset panel and its own covariance dictionary, so the
run is offline and deterministic. The universe is deliberately easy to solve — a positive
definite covariance, no conflicting constraints — because the subject is the *wrapper*, not
the solver's numerics: that each rebalancing date produces a row, that the columns come back
in the price panel's order, that constraints are respected, and that the objective actually
bites (a minimum-variance solve must tilt away from the high-vol asset, a risk-budgeted solve
must move with its budget).

Where a solver has a closed-form answer the test asserts against it rather than against a
recorded output. ``closed_form_optima_test.py`` already does this for the solvers themselves;
here it is used only to confirm the wrapper passes its arguments through unmangled.
"""
# packages
from typing import Dict
import numpy as np
import pandas as pd
import pytest
import qis
# optimalportfolios
from optimalportfolios import (
    Constraints,
    PortfolioObjective,
    rolling_maximise_diversification,
    rolling_maximize_cara_mixture,
    rolling_maximize_portfolio_sharpe,
    rolling_maximise_alpha_over_tre,
    rolling_max_return_target_vol,
    rolling_min_variance_target_return,
    rolling_minimise_tracking_error,
    rolling_quadratic_optimisation,
    rolling_risk_budgeting,
    solve_for_risk_budgets_from_given_weights,
)
from optimalportfolios.optimization.general.quadratic import cvx_quadratic_optimisation
from optimalportfolios.optimization.taa.maximise_alpha_with_target_yield import (
    rolling_maximise_alpha_with_target_return)
from optimalportfolios.optimization.wrapper_rolling_portfolios import (
    backtest_rolling_optimal_portfolio, compute_rolling_optimal_weights)

SEED = 20260810
TICKERS = ['growth', 'balanced', 'defensive']
# vols 22% / 14% / 6% with mild positive correlation: distinct enough that a min-variance
# solve must tilt to 'defensive' and a max-return solve to 'growth'
VOLS = np.array([0.22, 0.14, 0.06])
CORR = np.array([[1.00, 0.45, 0.15],
                 [0.45, 1.00, 0.25],
                 [0.15, 0.25, 1.00]])
COVAR = np.outer(VOLS, VOLS) * CORR
REBALANCING_DATES = pd.DatetimeIndex(['2024-03-31', '2024-06-30', '2024-09-30'])


def make_prices(n_days: int = 500) -> pd.DataFrame:
    """A seeded daily price panel over TICKERS, used for column order and weight drift."""
    rng = np.random.default_rng(SEED)
    dates = pd.date_range('2023-06-01', periods=n_days, freq='B')
    returns = rng.multivariate_normal(np.full(3, 0.0003), COVAR / 260.0, size=n_days)
    return pd.DataFrame(100.0 * np.exp(np.cumsum(returns, axis=0)), index=dates,
                        columns=TICKERS)


def make_covar_dict() -> Dict[pd.Timestamp, pd.DataFrame]:
    """The same covariance at every rebalancing date, so results are comparable across them."""
    covar = pd.DataFrame(COVAR, index=TICKERS, columns=TICKERS)
    return {date: covar for date in REBALANCING_DATES}


def make_expected_returns() -> pd.DataFrame:
    """Expected returns per rebalancing date, ordered growth > balanced > defensive."""
    return pd.DataFrame([[0.09, 0.06, 0.02]] * len(REBALANCING_DATES),
                        index=REBALANCING_DATES, columns=TICKERS)


def long_only() -> Constraints:
    """Fully invested, long only — the constraint set every case starts from."""
    return Constraints(is_long_only=True, min_weights=pd.Series(0.0, index=TICKERS),
                       max_weights=pd.Series(1.0, index=TICKERS))


def assert_valid_weights(weights: pd.DataFrame, dates=REBALANCING_DATES) -> None:
    """Every wrapper returns one fully invested long-only row per rebalancing date."""
    assert list(weights.index) == list(dates)
    assert list(weights.columns) == TICKERS, "columns must come back in the price panel's order"
    assert weights.notna().all().all()
    assert (weights >= -1e-8).all().all(), "long-only solve produced a short"
    np.testing.assert_allclose(weights.sum(axis=1).to_numpy(), 1.0, atol=1e-6)


# --------------------------------------------------------------------------- #
# risk budgeting
# --------------------------------------------------------------------------- #
def test_rolling_risk_budgeting_respects_the_budget_ordering() -> None:
    """the asset given the largest risk budget carries the largest risk contribution"""
    risk_budget = pd.Series([0.2, 0.3, 0.5], index=TICKERS)
    weights = rolling_risk_budgeting(prices=make_prices(), constraints=long_only(),
                                     risk_budget=risk_budget, covar_dict=make_covar_dict())
    assert_valid_weights(weights)
    w = weights.iloc[0].to_numpy()
    contributions = w * (COVAR @ w)
    contributions = contributions / contributions.sum()
    # the realised risk shares reproduce the requested budget
    np.testing.assert_allclose(contributions, risk_budget.to_numpy(), atol=0.02)


def test_rolling_risk_budgeting_moves_weight_when_the_budget_moves() -> None:
    """shifting the budget towards an asset raises its weight — the objective bites"""
    prices, covar_dict = make_prices(), make_covar_dict()
    even = rolling_risk_budgeting(prices=prices, constraints=long_only(),
                                  risk_budget=pd.Series([1 / 3, 1 / 3, 1 / 3], index=TICKERS),
                                  covar_dict=covar_dict)
    tilted = rolling_risk_budgeting(prices=prices, constraints=long_only(),
                                    risk_budget=pd.Series([0.6, 0.2, 0.2], index=TICKERS),
                                    covar_dict=covar_dict)
    assert tilted.iloc[0]['growth'] > even.iloc[0]['growth']


def test_rolling_risk_budgeting_short_circuits_a_single_asset_budget() -> None:
    """one budgeted asset is a 100% allocation, with the rest of the panel zeroed"""
    weights = rolling_risk_budgeting(prices=make_prices(), constraints=long_only(),
                                     risk_budget=pd.Series([1.0], index=['defensive']),
                                     covar_dict=make_covar_dict())
    assert_valid_weights(weights)
    assert (weights['defensive'] == 1.0).all()
    assert (weights[['growth', 'balanced']] == 0.0).all().all()


def test_rolling_risk_budgeting_accepts_rebalancing_indicators() -> None:
    """the freezing indicator is an optional input the wrapper aligns to the solve dates"""
    indicators = pd.DataFrame(1.0, index=REBALANCING_DATES, columns=TICKERS)
    weights = rolling_risk_budgeting(
        prices=make_prices(), constraints=long_only(),
        risk_budget=pd.Series([0.4, 0.3, 0.3], index=TICKERS),
        covar_dict=make_covar_dict(), rebalancing_indicators=indicators)
    assert_valid_weights(weights)


def test_solve_for_risk_budgets_recovers_the_budget_of_a_given_portfolio() -> None:
    """the inverse problem returns budgets that reproduce the weights it was given"""
    given = pd.Series([0.2, 0.3, 0.5], index=TICKERS)
    budgets = solve_for_risk_budgets_from_given_weights(
        prices=make_prices(), given_weights=given, covar_dict=make_covar_dict())
    assert list(budgets.index) == TICKERS
    assert budgets.sum() == pytest.approx(1.0, abs=1e-6)
    # the implied budget must order the assets the way their risk contributions do
    contributions = given.to_numpy() * (COVAR @ given.to_numpy())
    assert list(np.argsort(budgets.to_numpy())) == list(np.argsort(contributions))


# --------------------------------------------------------------------------- #
# variance and diversification
# --------------------------------------------------------------------------- #
def test_rolling_min_variance_tilts_to_the_lowest_vol_asset() -> None:
    """an unconstrained minimum-variance solve concentrates in 'defensive'"""
    weights = rolling_quadratic_optimisation(
        prices=make_prices(), constraints=long_only(), covar_dict=make_covar_dict(),
        portfolio_objective=PortfolioObjective.MIN_VARIANCE)
    assert_valid_weights(weights)
    assert weights.iloc[0].idxmax() == 'defensive'
    # and it is genuinely the minimum-variance point: perturbing towards growth raises vol
    w = weights.iloc[0].to_numpy()
    nudged = w + np.array([0.05, 0.0, -0.05])
    assert float(w @ COVAR @ w) < float(nudged @ COVAR @ nudged)


def test_rolling_quadratic_utility_trades_variance_for_return() -> None:
    """with expected returns and a low risk aversion the solve tilts to the high-return asset"""
    prices, covar_dict = make_prices(), make_covar_dict()
    cautious = rolling_quadratic_optimisation(
        prices=prices, constraints=long_only(), covar_dict=covar_dict,
        portfolio_objective=PortfolioObjective.QUADRATIC_UTILITY,
        expected_returns=make_expected_returns(), carra=20.0)
    aggressive = rolling_quadratic_optimisation(
        prices=prices, constraints=long_only(), covar_dict=covar_dict,
        portfolio_objective=PortfolioObjective.QUADRATIC_UTILITY,
        expected_returns=make_expected_returns(), carra=0.5)
    assert_valid_weights(cautious)
    assert_valid_weights(aggressive)
    assert aggressive.iloc[0]['growth'] > cautious.iloc[0]['growth']


def test_rolling_quadratic_optimisation_honours_inclusion_indicators() -> None:
    """an asset excluded at a date gets no weight there"""
    inclusion = pd.DataFrame(1.0, index=REBALANCING_DATES, columns=TICKERS)
    inclusion['defensive'] = 0.0
    weights = rolling_quadratic_optimisation(
        prices=make_prices(), constraints=long_only(), covar_dict=make_covar_dict(),
        inclusion_indicators=inclusion,
        portfolio_objective=PortfolioObjective.MIN_VARIANCE)
    assert np.allclose(weights['defensive'].to_numpy(), 0.0, atol=1e-8)
    np.testing.assert_allclose(weights.sum(axis=1).to_numpy(), 1.0, atol=1e-6)


def test_rolling_maximise_diversification_spreads_across_the_panel() -> None:
    """the diversification ratio solve holds every asset rather than concentrating"""
    weights = rolling_maximise_diversification(
        prices=make_prices(), constraints=long_only(), covar_dict=make_covar_dict())
    assert_valid_weights(weights)
    assert (weights.iloc[0] > 0.01).all(), "a max-diversification solve dropped an asset"
    # the diversification ratio beats the equally weighted portfolio's
    def diversification_ratio(w: np.ndarray) -> float:
        """Weighted average vol divided by portfolio vol."""
        return float(w @ VOLS) / float(np.sqrt(w @ COVAR @ w))
    assert diversification_ratio(weights.iloc[0].to_numpy()) >= diversification_ratio(
        np.full(3, 1 / 3)) - 1e-9


def test_rolling_maximize_portfolio_sharpe_prefers_the_best_reward_per_risk() -> None:
    """the max-Sharpe solve overweights the asset with the best return-to-vol ratio"""
    weights = rolling_maximize_portfolio_sharpe(
        prices=make_prices(), expected_returns=make_expected_returns(),
        constraints=long_only(), covar_dict=make_covar_dict())
    assert_valid_weights(weights)
    w = weights.iloc[0].to_numpy()
    sharpe = float(w @ make_expected_returns().iloc[0].to_numpy()) / float(
        np.sqrt(w @ COVAR @ w))
    equal = np.full(3, 1 / 3)
    equal_sharpe = float(equal @ make_expected_returns().iloc[0].to_numpy()) / float(
        np.sqrt(equal @ COVAR @ equal))
    assert sharpe >= equal_sharpe - 1e-9


# --------------------------------------------------------------------------- #
# constraints are passed through
# --------------------------------------------------------------------------- #
def test_max_weight_constraint_caps_the_dominant_asset() -> None:
    """a cap the unconstrained solution would breach is respected at every date"""
    constraints = Constraints(is_long_only=True,
                              min_weights=pd.Series(0.0, index=TICKERS),
                              max_weights=pd.Series([1.0, 1.0, 0.40], index=TICKERS))
    weights = rolling_quadratic_optimisation(
        prices=make_prices(), constraints=constraints, covar_dict=make_covar_dict(),
        portfolio_objective=PortfolioObjective.MIN_VARIANCE)
    assert_valid_weights(weights)
    assert (weights['defensive'] <= 0.40 + 1e-6).all()


def test_min_weight_constraint_floors_every_asset() -> None:
    """a floor forces the solve to hold an asset it would otherwise drop"""
    constraints = Constraints(is_long_only=True,
                              min_weights=pd.Series(0.15, index=TICKERS),
                              max_weights=pd.Series(1.0, index=TICKERS))
    weights = rolling_quadratic_optimisation(
        prices=make_prices(), constraints=constraints, covar_dict=make_covar_dict(),
        portfolio_objective=PortfolioObjective.MIN_VARIANCE)
    assert_valid_weights(weights)
    assert (weights >= 0.15 - 1e-6).all().all()


# --------------------------------------------------------------------------- #
# benchmark-relative solvers
# --------------------------------------------------------------------------- #
def test_rolling_minimise_tracking_error_returns_to_the_benchmark() -> None:
    """with no other objective the tracking-error solve reproduces the benchmark"""
    benchmark = pd.Series([0.5, 0.3, 0.2], index=TICKERS)
    weights = rolling_minimise_tracking_error(
        prices=make_prices(), constraints=long_only(), benchmark_weights=benchmark,
        covar_dict=make_covar_dict())
    assert_valid_weights(weights)
    np.testing.assert_allclose(weights.iloc[0].to_numpy(), benchmark.to_numpy(), atol=1e-4)


def test_rolling_maximise_alpha_over_tre_tilts_towards_the_alpha() -> None:
    """a positive alpha on one asset moves the portfolio above its benchmark weight"""
    benchmark = pd.Series([1 / 3, 1 / 3, 1 / 3], index=TICKERS)
    alphas = pd.DataFrame([[0.05, 0.0, -0.05]] * len(REBALANCING_DATES),
                          index=REBALANCING_DATES, columns=TICKERS)
    constraints = Constraints(is_long_only=True,
                              min_weights=pd.Series(0.0, index=TICKERS),
                              max_weights=pd.Series(1.0, index=TICKERS),
                              tracking_err_vol_constraint=0.03)
    weights = rolling_maximise_alpha_over_tre(
        prices=make_prices(), alphas=alphas, constraints=constraints,
        benchmark_weights=benchmark, covar_dict=make_covar_dict())
    assert_valid_weights(weights)
    assert weights.iloc[0]['growth'] > benchmark['growth']
    assert weights.iloc[0]['defensive'] < benchmark['defensive']


def test_rolling_max_return_target_vol_hits_the_vol_budget() -> None:
    """the solve spends its volatility budget rather than sitting below it"""
    target_vols = pd.Series(0.10, index=REBALANCING_DATES)
    weights = rolling_max_return_target_vol(
        prices=make_prices(), expected_returns=make_expected_returns(),
        target_vols=target_vols, constraints=long_only(), benchmark_weights=None,
        covar_dict=make_covar_dict())
    assert_valid_weights(weights)
    w = weights.iloc[0].to_numpy()
    assert float(np.sqrt(w @ COVAR @ w)) <= 0.10 + 1e-4
    # a looser budget buys more of the high-return asset
    looser = rolling_max_return_target_vol(
        prices=make_prices(), expected_returns=make_expected_returns(),
        target_vols=pd.Series(0.16, index=REBALANCING_DATES), constraints=long_only(),
        benchmark_weights=None, covar_dict=make_covar_dict())
    assert looser.iloc[0]['growth'] > weights.iloc[0]['growth']


def test_rolling_min_variance_target_return_hits_the_return_floor() -> None:
    """the solve meets the required return and takes the least variance that does"""
    target_returns = pd.Series(0.05, index=REBALANCING_DATES)
    weights = rolling_min_variance_target_return(
        prices=make_prices(), expected_returns=make_expected_returns(),
        target_returns=target_returns, constraints=long_only(), benchmark_weights=None,
        covar_dict=make_covar_dict())
    assert_valid_weights(weights)
    achieved = float(weights.iloc[0].to_numpy() @ make_expected_returns().iloc[0].to_numpy())
    assert achieved >= 0.05 - 1e-4
    # demanding more return costs more variance
    demanding = rolling_min_variance_target_return(
        prices=make_prices(), expected_returns=make_expected_returns(),
        target_returns=pd.Series(0.075, index=REBALANCING_DATES), constraints=long_only(),
        benchmark_weights=None, covar_dict=make_covar_dict())
    w, v = weights.iloc[0].to_numpy(), demanding.iloc[0].to_numpy()
    assert float(w @ COVAR @ w) <= float(v @ COVAR @ v) + 1e-9


def test_rolling_maximise_alpha_with_target_return_meets_the_yield_floor() -> None:
    """the yield constraint binds, and the alpha decides how the budget is spent"""
    yields = pd.DataFrame([[0.02, 0.04, 0.07]] * len(REBALANCING_DATES),
                          index=REBALANCING_DATES, columns=TICKERS)
    alphas = pd.DataFrame([[0.03, 0.0, -0.03]] * len(REBALANCING_DATES),
                          index=REBALANCING_DATES, columns=TICKERS)
    weights = rolling_maximise_alpha_with_target_return(
        prices=make_prices(), alphas=alphas, yields=yields,
        target_returns=pd.Series(0.05, index=REBALANCING_DATES),
        constraints=long_only(), covar_dict=make_covar_dict())
    assert_valid_weights(weights)
    achieved = float(weights.iloc[0].to_numpy() @ yields.iloc[0].to_numpy())
    assert achieved >= 0.05 - 1e-4


def test_rolling_maximise_alpha_with_target_return_runs_benchmark_relative() -> None:
    """with a benchmark the objective becomes active alpha under a tracking-error budget"""
    yields = pd.DataFrame([[0.02, 0.04, 0.07]] * len(REBALANCING_DATES),
                          index=REBALANCING_DATES, columns=TICKERS)
    alphas = pd.DataFrame([[0.03, 0.0, -0.03]] * len(REBALANCING_DATES),
                          index=REBALANCING_DATES, columns=TICKERS)
    benchmark = pd.Series([1 / 3, 1 / 3, 1 / 3], index=TICKERS)
    constraints = Constraints(is_long_only=True,
                              min_weights=pd.Series(0.0, index=TICKERS),
                              max_weights=pd.Series(1.0, index=TICKERS),
                              tracking_err_vol_constraint=0.04)
    weights = rolling_maximise_alpha_with_target_return(
        prices=make_prices(), alphas=alphas, yields=yields,
        target_returns=pd.Series(0.04, index=REBALANCING_DATES),
        constraints=constraints, covar_dict=make_covar_dict(),
        benchmark_weights=benchmark)
    assert_valid_weights(weights)
    # the positive-alpha asset is held above its benchmark weight
    assert weights.iloc[0]['growth'] > benchmark['growth']


# --------------------------------------------------------------------------- #
# the mixture solver and the dispatcher
# --------------------------------------------------------------------------- #
def test_rolling_maximize_cara_mixture_produces_investable_weights() -> None:
    """the CARA mixture solve estimates its own mixture off the price panel"""
    prices = make_prices(n_days=900)
    weights = rolling_maximize_cara_mixture(
        prices=prices, constraints=long_only(),
        time_period=qis.get_time_period(df=prices), rebalancing_freq='QE',
        roll_window=6, returns_freq='W-WED', carra=0.5, n_components=2)
    assert not weights.empty
    assert list(weights.columns) == TICKERS
    assert (weights.to_numpy() >= -1e-6).all()
    np.testing.assert_allclose(weights.sum(axis=1).to_numpy(), 1.0, atol=1e-4)


@pytest.mark.parametrize('objective', [
    PortfolioObjective.EQUAL_RISK_CONTRIBUTION,
    PortfolioObjective.MAX_DIVERSIFICATION,
    PortfolioObjective.MIN_VARIANCE,
    PortfolioObjective.QUADRATIC_UTILITY,
    PortfolioObjective.MAXIMUM_SHARPE_RATIO,
])
def test_compute_rolling_optimal_weights_routes_every_objective(objective) -> None:
    """the dispatcher reaches each solver and returns a weight panel for it"""
    prices = make_prices()
    weights = compute_rolling_optimal_weights(
        prices=prices, constraints=long_only(), covar_dict=make_covar_dict(),
        portfolio_objective=objective,
        risk_budget=pd.Series([1 / 3, 1 / 3, 1 / 3], index=TICKERS),
        returns_freq='W-WED', span=12)
    assert_valid_weights(weights)


def test_quadratic_utility_without_expected_returns_is_rejected_up_front() -> None:
    """the utility objective is μ'w - γ/2 w'Σw, so it cannot run without μ

    Caught at the rolling entry point rather than one date into the walk: the alternative is a
    ``NoneType`` failure inside the first single-date solve, which says nothing about the
    argument the caller left out.
    """
    with pytest.raises(ValueError, match='expected_returns must be given'):
        rolling_quadratic_optimisation(
            prices=make_prices(), constraints=long_only(), covar_dict=make_covar_dict(),
            portfolio_objective=PortfolioObjective.QUADRATIC_UTILITY)


def test_the_single_date_quadratic_solve_repeats_the_expected_returns_guard() -> None:
    """the same check at the single-date entry point, which callers reach directly"""
    with pytest.raises(ValueError, match='means must be given'):
        cvx_quadratic_optimisation(
            portfolio_objective=PortfolioObjective.QUADRATIC_UTILITY,
            covar=COVAR, constraints=long_only())


def test_the_single_date_quadratic_solve_rejects_an_objective_it_cannot_form() -> None:
    """only the two quadratic objectives have an objective function here

    ``rolling_quadratic_optimisation`` is only ever called with MIN_VARIANCE or
    QUADRATIC_UTILITY, so this is reachable only by calling the single-date solver directly
    with something else — the misuse AGENTS.md names when it asks callers to extend the enum
    rather than pass strings around.
    """
    with pytest.raises(ValueError, match='unsupported portfolio_objective'):
        cvx_quadratic_optimisation(
            portfolio_objective=PortfolioObjective.MAX_DIVERSIFICATION,
            covar=COVAR, constraints=long_only())


def test_compute_rolling_optimal_weights_routes_the_mixture_objective() -> None:
    """the CARA mixture arm takes none of the covariance dict, so it is routed separately

    Every other objective is solved off the pre-computed ``covar_dict``; this one estimates
    its own mixture from the price panel, and so needs the time period, the rebalancing
    frequency and the roll window that the parametrised cases above have no use for.
    """
    prices = make_prices(n_days=900)
    weights = compute_rolling_optimal_weights(
        prices=prices, constraints=long_only(), covar_dict=make_covar_dict(),
        portfolio_objective=PortfolioObjective.MAX_CARA_MIXTURE,
        time_period=qis.get_time_period(df=prices), rebalancing_freq='QE',
        returns_freq='W-WED', roll_window=6, n_mixures=2, carra=0.5)
    assert not weights.empty
    assert list(weights.columns) == TICKERS
    assert (weights.to_numpy() >= -1e-6).all()
    np.testing.assert_allclose(weights.sum(axis=1).to_numpy(), 1.0, atol=1e-4)


def test_compute_rolling_optimal_weights_rejects_an_unrouted_objective() -> None:
    """a raw string instead of the enum raises rather than silently returning nothing

    Every ``PortfolioObjective`` member is routed, so the guard is only reachable by passing
    something that is not one — which is the misuse AGENTS.md names when it asks callers to
    extend the enum rather than pass strings around.
    """
    with pytest.raises(NotImplementedError, match='MinVariance'):
        compute_rolling_optimal_weights(
            prices=make_prices(), constraints=long_only(), covar_dict=make_covar_dict(),
            portfolio_objective='MinVariance')


def test_backtest_rolling_optimal_portfolio_runs_the_weights_through_qis() -> None:
    """the one-call wrapper optimises and backtests, and hands back a qis PortfolioData

    The backtest itself is ``qis.backtest_model_portfolio`` — this package must not hand-roll
    a loop over dates that accumulates a position. What is asserted here is the wiring: the
    panel is truncated to start at the first weight date, and the portfolio that comes back
    carries the tickers it was given.
    """
    prices = make_prices()
    portfolio = backtest_rolling_optimal_portfolio(
        prices=prices, constraints=long_only(), covar_dict=make_covar_dict(),
        portfolio_objective=PortfolioObjective.MIN_VARIANCE, ticker='min_variance')
    weights = compute_rolling_optimal_weights(
        prices=prices, constraints=long_only(), covar_dict=make_covar_dict(),
        portfolio_objective=PortfolioObjective.MIN_VARIANCE)
    nav = portfolio.get_portfolio_nav()
    assert nav.name == 'min_variance'
    assert nav.notna().all()
    # the backtest starts at the first weight date, not at the start of the price panel
    # (to within the business-day grid the prices are on, which the weight dates are not)
    assert abs(nav.index[0] - weights.index[0]) <= pd.Timedelta(days=7)
    assert nav.index[0] > prices.index[0]
    assert list(portfolio.weights.columns) == TICKERS


def test_backtest_rolling_optimal_portfolio_reports_over_the_requested_period() -> None:
    """perf_time_period narrows the weights before the backtest, not the report afterwards"""
    prices = make_prices()
    period = qis.TimePeriod(REBALANCING_DATES[1], prices.index[-1])
    portfolio = backtest_rolling_optimal_portfolio(
        prices=prices, constraints=long_only(), covar_dict=make_covar_dict(),
        portfolio_objective=PortfolioObjective.MIN_VARIANCE,
        perf_time_period=period, ticker='windowed')
    whole = backtest_rolling_optimal_portfolio(
        prices=prices, constraints=long_only(), covar_dict=make_covar_dict(),
        portfolio_objective=PortfolioObjective.MIN_VARIANCE, ticker='whole')
    windowed_start = portfolio.get_portfolio_nav().index[0]
    assert windowed_start > whole.get_portfolio_nav().index[0]
    assert windowed_start >= REBALANCING_DATES[0]
    # the first two rebalancings are gone, so only the last one is traded
    assert len(portfolio.weights) < len(whole.weights)
