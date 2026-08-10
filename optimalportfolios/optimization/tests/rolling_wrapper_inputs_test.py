"""
the per-date argument plumbing the three benchmark-relative rolling wrappers share.

``rolling_max_return_target_vol``, ``rolling_min_variance_target_return`` and
``rolling_maximise_alpha_over_tre`` each walk the rebalancing schedule and slice their
per-date arguments out of a panel before calling the single-date wrapper. That slicing is
where a stale or misaligned row would enter a backtest silently: a benchmark taken from the
wrong date, an eligibility flag that never arrived, a drift chain that keeps running after a
rebalancing produced nothing. Nothing raises when any of it goes wrong — the weights are
merely someone else's.

The cases here pin that layer rather than the numerics, which the closed-form and utility
suites already cover:

* a *static* benchmark given as a Series must be broadcast to every rebalancing date and give
  the same answer as passing the equivalent panel;
* ``rebalancing_indicators`` reach the single-date solve, so a frozen asset stays frozen;
* a date whose solve returns nothing must reset the drift chain rather than drift a zero
  portfolio into the next date;
* ``apply_total_to_good_ratio`` rescales the constraint set when assets are dropped;
* the benchmark-beta constraint takes its loadings per date, and demands them when the
  constraint is set.

The universe is the three-asset panel the other rolling-solver files use, so a result here is
comparable with theirs.
"""
# packages
from typing import Dict
import numpy as np
import pandas as pd
import pytest
# optimalportfolios
from optimalportfolios import Constraints
from optimalportfolios.optimization.config import OptimiserConfig
from optimalportfolios.optimization.constraints import (
    BenchmarkBetaConstraint, ConstraintEnforcementType, GroupLowerUpperConstraints)
from optimalportfolios.optimization.saa.max_return_target_vol import (
    rolling_max_return_target_vol)
from optimalportfolios.optimization.saa.min_variance_target_return import (
    rolling_min_variance_target_return, wrapper_min_variance_target_return)
from optimalportfolios.optimization.taa.maximise_alpha_over_tre import (
    rolling_maximise_alpha_over_tre)

SEED = 20260810
TICKERS = ['growth', 'balanced', 'defensive']
VOLS = np.array([0.22, 0.14, 0.06])
CORR = np.array([[1.00, 0.45, 0.15],
                 [0.45, 1.00, 0.25],
                 [0.15, 0.25, 1.00]])
COVAR = np.outer(VOLS, VOLS) * CORR
ALPHAS = np.array([0.09, 0.06, 0.02])
BENCHMARK = pd.Series([0.40, 0.35, 0.25], index=TICKERS)
REBALANCING_DATES = pd.DatetimeIndex(['2024-03-31', '2024-06-30', '2024-09-30'])


def make_prices(n_days: int = 400) -> pd.DataFrame:
    """A seeded daily price panel, used for column alignment and weight drift."""
    rng = np.random.default_rng(SEED)
    dates = pd.date_range('2023-06-01', periods=n_days, freq='B')
    returns = rng.multivariate_normal(np.full(3, 0.0003), COVAR / 260.0, size=n_days)
    return pd.DataFrame(100.0 * np.exp(np.cumsum(returns, axis=0)), index=dates,
                        columns=TICKERS)


def make_covar_dict(covar: np.ndarray = COVAR) -> Dict[pd.Timestamp, pd.DataFrame]:
    """The same covariance at every rebalancing date.

    The array is copied rather than wrapped. ``pd.DataFrame(ndarray)`` shares the buffer under
    pandas 2, so a frame built here and written to later would reach back into the module-level
    ``COVAR`` — see ``covar_with_a_dead_asset``.
    """
    frame = pd.DataFrame(np.array(covar, copy=True), index=TICKERS, columns=TICKERS)
    return {date: frame for date in REBALANCING_DATES}


def make_expected_returns() -> pd.DataFrame:
    """Expected returns per rebalancing date, ordered growth > balanced > defensive."""
    return pd.DataFrame([list(ALPHAS)] * len(REBALANCING_DATES),
                        index=REBALANCING_DATES, columns=TICKERS)


def varying_expected_returns() -> pd.DataFrame:
    """Expected returns that reorder across the schedule, so each date has its own optimum.

    With one constant forecast every rebalancing solves to the same portfolio, and a case
    about *which* date was solved cannot tell a re-solve from a hold.
    """
    return pd.DataFrame([[0.09, 0.06, 0.02],
                         [0.02, 0.06, 0.09],
                         [0.06, 0.09, 0.02]],
                        index=REBALANCING_DATES, columns=TICKERS)


def benchmark_panel() -> pd.DataFrame:
    """The static benchmark written out as one identical row per rebalancing date."""
    return pd.DataFrame([BENCHMARK.to_numpy()] * len(REBALANCING_DATES),
                        index=REBALANCING_DATES, columns=TICKERS)


def long_only(**overrides) -> Constraints:
    """Fully invested, long only — the constraint set every case starts from."""
    kwargs = dict(is_long_only=True, min_weights=pd.Series(0.0, index=TICKERS),
                  max_weights=pd.Series(1.0, index=TICKERS))
    kwargs.update(overrides)
    return Constraints(**kwargs)


def target_vols() -> pd.Series:
    """A constant volatility budget across the schedule."""
    return pd.Series(0.12, index=REBALANCING_DATES)


def target_returns() -> pd.Series:
    """A return target inside the achievable range of ALPHAS."""
    return pd.Series(0.05, index=REBALANCING_DATES)


def test_the_fixtures_do_not_write_into_the_shared_covariance() -> None:
    """building a covariance frame must not reach back into the module-level array

    Under pandas 2 ``pd.DataFrame(ndarray)`` wraps the buffer instead of copying it, so a
    fixture that mutates its frame silently poisons ``COVAR`` for every test that runs after
    it — which surfaced as ``multivariate_normal`` failing with "SVD did not converge" in
    eight unrelated cases, on the 3.10 leg only. Asserted here because the corruption is
    invisible in the test that causes it and fatal in the ones that follow.
    """
    covar_with_a_dead_asset()
    make_covar_dict()
    assert np.isfinite(COVAR).all(), 'a fixture wrote NaN into the shared covariance'
    np.testing.assert_allclose(COVAR, np.outer(VOLS, VOLS) * CORR)


# --------------------------------------------------------------------------- #
# a static benchmark is broadcast, not silently dropped
# --------------------------------------------------------------------------- #
def test_max_return_accepts_a_static_benchmark_as_a_series() -> None:
    """a Series benchmark is the same statement as a panel repeating it every date

    A caller with a fixed strategic benchmark passes it once as a Series. The wrapper widens
    it to the rebalancing grid; if it instead kept only the first date the later solves would
    run benchmark-free, which changes the objective from active to absolute return without
    reporting anything.
    """
    prices = make_prices()
    kwargs = dict(prices=prices, expected_returns=make_expected_returns(),
                  target_vols=target_vols(), constraints=long_only(),
                  covar_dict=make_covar_dict())
    from_series = rolling_max_return_target_vol(benchmark_weights=BENCHMARK, **kwargs)
    from_panel = rolling_max_return_target_vol(benchmark_weights=benchmark_panel(), **kwargs)
    pd.testing.assert_frame_equal(from_series, from_panel, atol=1e-8)
    assert list(from_series.index) == list(REBALANCING_DATES)


def test_min_variance_accepts_a_static_benchmark_as_a_series() -> None:
    """the same broadcast rule in the minimum-variance wrapper"""
    prices = make_prices()
    kwargs = dict(prices=prices, expected_returns=make_expected_returns(),
                  target_returns=target_returns(), constraints=long_only(),
                  covar_dict=make_covar_dict())
    from_series = rolling_min_variance_target_return(benchmark_weights=BENCHMARK, **kwargs)
    from_panel = rolling_min_variance_target_return(benchmark_weights=benchmark_panel(),
                                                    **kwargs)
    pd.testing.assert_frame_equal(from_series, from_panel, atol=1e-8)


def test_alpha_over_tre_accepts_a_time_varying_benchmark_panel() -> None:
    """the TRE wrapper takes a per-date benchmark, and a moving one moves the answer"""
    prices = make_prices()
    alphas = make_expected_returns()
    constraints = long_only(tracking_err_vol_constraint=0.03)
    kwargs = dict(prices=prices, alphas=alphas, constraints=constraints,
                  covar_dict=make_covar_dict())
    static = rolling_maximise_alpha_over_tre(benchmark_weights=BENCHMARK, **kwargs)

    moving = benchmark_panel()
    moving.iloc[-1] = [0.10, 0.10, 0.80]     # the last date tracks a different portfolio
    varying = rolling_maximise_alpha_over_tre(benchmark_weights=moving, **kwargs)

    pd.testing.assert_series_equal(static.iloc[0], varying.iloc[0], atol=1e-6)
    assert not np.allclose(static.iloc[-1].to_numpy(), varying.iloc[-1].to_numpy(), atol=1e-3)


# --------------------------------------------------------------------------- #
# rebalancing indicators reach the single-date solve
# --------------------------------------------------------------------------- #
def frozen_indicators() -> pd.DataFrame:
    """Freeze every asset on the middle rebalancing date and trade on the others."""
    indicators = pd.DataFrame(1.0, index=REBALANCING_DATES, columns=TICKERS)
    indicators.loc[REBALANCING_DATES[1]] = 0.0
    return indicators


def test_max_return_freezes_the_positions_it_is_told_to_freeze() -> None:
    """a zero rebalancing indicator holds the prior weights instead of re-solving"""
    prices = make_prices()
    kwargs = dict(prices=prices, expected_returns=varying_expected_returns(),
                  target_vols=target_vols(), constraints=long_only(),
                  benchmark_weights=BENCHMARK, covar_dict=make_covar_dict())
    traded = rolling_max_return_target_vol(**kwargs)
    frozen = rolling_max_return_target_vol(rebalancing_indicators=frozen_indicators(),
                                           **kwargs)
    assert list(frozen.index) == list(REBALANCING_DATES)
    # the frozen date holds a portfolio drifted from the first solve, not the free optimum
    assert not np.allclose(frozen.iloc[1].to_numpy(), traded.iloc[1].to_numpy(), atol=1e-4)


def test_min_variance_freezes_the_positions_it_is_told_to_freeze() -> None:
    """the same indicator plumbing in the minimum-variance wrapper"""
    prices = make_prices()
    kwargs = dict(prices=prices, expected_returns=varying_expected_returns(),
                  target_returns=target_returns(), constraints=long_only(),
                  benchmark_weights=BENCHMARK, covar_dict=make_covar_dict())
    traded = rolling_min_variance_target_return(**kwargs)
    frozen = rolling_min_variance_target_return(rebalancing_indicators=frozen_indicators(),
                                                **kwargs)
    assert not np.allclose(frozen.iloc[1].to_numpy(), traded.iloc[1].to_numpy(), atol=1e-4)


def test_alpha_over_tre_freezes_the_positions_it_is_told_to_freeze() -> None:
    """and in the tracking-error wrapper"""
    prices = make_prices()
    kwargs = dict(prices=prices, alphas=varying_expected_returns(),
                  constraints=long_only(tracking_err_vol_constraint=0.03),
                  benchmark_weights=BENCHMARK, covar_dict=make_covar_dict())
    traded = rolling_maximise_alpha_over_tre(**kwargs)
    frozen = rolling_maximise_alpha_over_tre(rebalancing_indicators=frozen_indicators(),
                                             **kwargs)
    assert not np.allclose(frozen.iloc[1].to_numpy(), traded.iloc[1].to_numpy(), atol=1e-4)


# --------------------------------------------------------------------------- #
# a rebalancing that produced nothing must not be drifted forward
# --------------------------------------------------------------------------- #
def unreachable_box() -> Constraints:
    """Box caps summing to 0.6, so full investment is impossible at every date."""
    return Constraints(is_long_only=True, min_weights=pd.Series(0.0, index=TICKERS),
                       max_weights=pd.Series(0.2, index=TICKERS))


@pytest.mark.parametrize('wrapper', ['max_return', 'min_variance', 'alpha_over_tre'])
def test_a_rebalancing_that_solved_to_nothing_resets_the_drift_chain(wrapper: str) -> None:
    """zero weights mean "no portfolio", and drifting one forward would invent holdings

    With no prior portfolio and no benchmark to fall back to, a rejected solve returns zeros.
    The wrapper reads that as a date with no position and clears both ``weights_0`` and
    ``prev_date``; keeping them would hand the next date a zero starting portfolio drifted
    over a price path, which is a holding nobody ever had.
    """
    prices, covar_dict = make_prices(), make_covar_dict()
    if wrapper == 'max_return':
        weights = rolling_max_return_target_vol(
            prices=prices, expected_returns=make_expected_returns(),
            target_vols=target_vols(), constraints=unreachable_box(),
            benchmark_weights=None, covar_dict=covar_dict)
    elif wrapper == 'min_variance':
        weights = rolling_min_variance_target_return(
            prices=prices, expected_returns=make_expected_returns(),
            target_returns=target_returns(), constraints=unreachable_box(),
            benchmark_weights=None, covar_dict=covar_dict)
    else:
        # the TRE wrapper always has a benchmark to fall back to, so the zero case is the
        # absolute mandate: tracking error measured against cash rather than an index
        weights = rolling_maximise_alpha_over_tre(
            prices=prices, alphas=make_expected_returns(),
            constraints=unreachable_box(),
            benchmark_weights=pd.Series(0.0, index=TICKERS), covar_dict=covar_dict)
    assert list(weights.index) == list(REBALANCING_DATES)
    assert np.isfinite(weights.to_numpy()).all()
    np.testing.assert_allclose(weights.to_numpy(), 0.0, atol=1e-10)


# --------------------------------------------------------------------------- #
# rescaling when assets drop out
# --------------------------------------------------------------------------- #
def covar_with_a_dead_asset() -> Dict[pd.Timestamp, pd.DataFrame]:
    """A covariance whose third asset is all-NaN, so the filter drops it before the solve.

    The copy is load-bearing. ``pd.DataFrame(ndarray)`` wraps the buffer without copying it
    under pandas 2 (pandas 3 copies on write), so the two assignments below wrote straight
    into the module-level ``COVAR`` — every later ``make_prices`` then drew from a covariance
    full of NaN and ``multivariate_normal`` failed inside LAPACK with "SVD did not converge",
    in eight tests that have nothing to do with this fixture.
    """
    frame = pd.DataFrame(np.array(COVAR, copy=True), index=TICKERS, columns=TICKERS)
    frame.loc[TICKERS[2], :] = np.nan
    frame.loc[:, TICKERS[2]] = np.nan
    return {date: frame for date in REBALANCING_DATES}


@pytest.mark.parametrize('wrapper', ['max_return', 'min_variance', 'alpha_over_tre'])
def test_dropping_an_asset_rescales_the_constraints_when_asked(wrapper: str) -> None:
    """apply_total_to_good_ratio widens the surviving assets' bounds by N_total / N_valid

    An asset dropped for a NaN covariance takes its share of the budget with it. Left alone,
    the remaining bounds are stated against a universe that no longer exists and the solve is
    squeezed into a smaller box than the caller asked for. The rescaling is off by default,
    so this path only runs when a caller opts in — and then nothing checks it.
    """
    config = OptimiserConfig(apply_total_to_good_ratio=True)
    prices, covar_dict = make_prices(), covar_with_a_dead_asset()
    if wrapper == 'max_return':
        # a looser vol budget than the other cases: with the third asset gone, the least
        # volatile portfolio available is 14% and a 12% budget would simply be infeasible
        weights = rolling_max_return_target_vol(
            prices=prices, expected_returns=make_expected_returns(),
            target_vols=pd.Series(0.20, index=REBALANCING_DATES), constraints=long_only(),
            benchmark_weights=None, covar_dict=covar_dict, optimiser_config=config)
    elif wrapper == 'min_variance':
        weights = rolling_min_variance_target_return(
            prices=prices, expected_returns=make_expected_returns(),
            target_returns=target_returns(), constraints=long_only(),
            benchmark_weights=None, covar_dict=covar_dict, optimiser_config=config)
    else:
        weights = rolling_maximise_alpha_over_tre(
            prices=prices, alphas=make_expected_returns(),
            constraints=long_only(tracking_err_vol_constraint=0.03),
            benchmark_weights=BENCHMARK, covar_dict=covar_dict, optimiser_config=config)
    # the dropped asset gets nothing, and the survivors still carry the whole budget
    assert np.allclose(weights[TICKERS[2]].to_numpy(), 0.0, atol=1e-8)
    assert weights.to_numpy().sum(axis=1).min() > 0.0


# --------------------------------------------------------------------------- #
# the benchmark-beta constraint takes its loadings per date
# --------------------------------------------------------------------------- #
def test_alpha_over_tre_demands_loadings_for_a_beta_constraint() -> None:
    """the beta bound is static but the loadings are not, so they must be supplied

    Without loadings the constraint has nothing to apply and would be silently skipped, so
    the run would report a beta-constrained portfolio that was never beta-constrained.
    """
    constraints = long_only(tracking_err_vol_constraint=0.03,
                            benchmark_beta_constraint=BenchmarkBetaConstraint(beta_max=0.9))
    with pytest.raises(ValueError, match='benchmark_beta_loadings must be given'):
        rolling_maximise_alpha_over_tre(
            prices=make_prices(), alphas=make_expected_returns(), constraints=constraints,
            benchmark_weights=BENCHMARK, covar_dict=make_covar_dict())


def test_alpha_over_tre_injects_this_date_s_beta_loadings() -> None:
    """a beta cap bites: the constrained portfolio carries less beta than the free one

    The benchmark itself has a beta of 0.945 under these loadings, so the cap is a genuine
    tilt away from it rather than a bound the benchmark already satisfies.
    """
    loadings = pd.DataFrame([[1.30, 1.00, 0.30]] * len(REBALANCING_DATES),
                            index=REBALANCING_DATES, columns=TICKERS)
    kwargs = dict(prices=make_prices(), alphas=make_expected_returns(),
                  benchmark_weights=BENCHMARK, covar_dict=make_covar_dict())
    free = rolling_maximise_alpha_over_tre(
        constraints=long_only(tracking_err_vol_constraint=0.06), **kwargs)
    capped = rolling_maximise_alpha_over_tre(
        constraints=long_only(tracking_err_vol_constraint=0.06,
                              benchmark_beta_constraint=BenchmarkBetaConstraint(beta_max=0.85)),
        benchmark_beta_loadings=loadings, **kwargs)
    row = loadings.iloc[0].to_numpy()
    assert capped.iloc[0].to_numpy() @ row <= 0.85 + 1e-5
    assert free.iloc[0].to_numpy() @ row > capped.iloc[0].to_numpy() @ row


def test_alpha_over_tre_runs_without_alphas_at_all() -> None:
    """with no alphas the objective is the penalty terms alone, not an error

    ``alphas=None`` is how a caller asks for the closest feasible portfolio to the benchmark
    under a set of constraints — a rebalancing with no view. The wrapper then skips the NaN
    filtering that is otherwise keyed on the alphas.

    Note this runs the penalised formulation: ``cvx_maximise_alpha_over_tre``, the forced one,
    dereferences ``alphas`` unconditionally, so ``alphas=None`` reaches it only as an
    ``AttributeError`` despite the ``Optional`` annotation on the rolling entry point.
    """
    weights = rolling_maximise_alpha_over_tre(
        prices=make_prices(), alphas=None,
        constraints=long_only(
            tracking_err_vol_constraint=0.03,
            constraint_enforcement_type=ConstraintEnforcementType.UTILITY_CONSTRAINTS),
        benchmark_weights=BENCHMARK, covar_dict=make_covar_dict())
    assert list(weights.index) == list(REBALANCING_DATES)
    np.testing.assert_allclose(weights.sum(axis=1).to_numpy(), 1.0, atol=1e-4)


# --------------------------------------------------------------------------- #
# the utility formulation carries group bounds as hard constraints
# --------------------------------------------------------------------------- #
def group_constraints() -> GroupLowerUpperConstraints:
    """Cap the two risky assets at 50% of the portfolio as a group."""
    loadings = pd.DataFrame({'risky': [1.0, 1.0, 0.0], 'safe': [0.0, 0.0, 1.0]},
                            index=TICKERS)
    return GroupLowerUpperConstraints(
        group_loadings=loadings,
        group_min_allocation=pd.Series([0.0, 0.0], index=['risky', 'safe']),
        group_max_allocation=pd.Series([0.5, 1.0], index=['risky', 'safe']))


@pytest.mark.parametrize('wrapper', ['max_return', 'min_variance'])
def test_group_bounds_stay_hard_in_the_penalised_formulation(wrapper: str) -> None:
    """the utility formulation penalises risk and turnover, but not group membership

    Everything else in ``UTILITY_CONSTRAINTS`` becomes a penalty a solve can pay its way out
    of. Group bounds do not: they are handed to the solver as constraints in the penalised
    path too, so a mandate limit stays a limit.
    """
    constraints = long_only(
        constraint_enforcement_type=ConstraintEnforcementType.UTILITY_CONSTRAINTS,
        group_lower_upper_constraints=group_constraints())
    prices, covar_dict = make_prices(), make_covar_dict()
    if wrapper == 'max_return':
        weights = rolling_max_return_target_vol(
            prices=prices, expected_returns=make_expected_returns(),
            target_vols=target_vols(), constraints=constraints,
            benchmark_weights=None, covar_dict=covar_dict)
    else:
        weights = rolling_min_variance_target_return(
            prices=prices, expected_returns=make_expected_returns(),
            target_returns=target_returns(), constraints=constraints,
            benchmark_weights=None, covar_dict=covar_dict)
    risky = weights[TICKERS[:2]].sum(axis=1)
    assert risky.max() <= 0.5 + 1e-5, 'a group cap was paid off rather than respected'


# --------------------------------------------------------------------------- #
# the minimum-variance wrapper's two input guards
# --------------------------------------------------------------------------- #
def test_min_variance_clamps_a_target_return_no_asset_can_reach() -> None:
    """an unreachable target is clamped with a warning rather than solved as infeasible

    A target above every asset's expected return makes the return constraint unsatisfiable,
    and the whole rebalance would be rejected. Clamping to the best available asset return
    keeps the date investable and says so.
    """
    with pytest.warns(UserWarning, match='exceeds max asset return'):
        weights = rolling_min_variance_target_return(
            prices=make_prices(), expected_returns=make_expected_returns(),
            target_returns=pd.Series(0.50, index=REBALANCING_DATES),
            constraints=long_only(), benchmark_weights=None,
            covar_dict=make_covar_dict())
    # clamped to the top asset return, so the solve concentrates there
    assert weights.iloc[0][TICKERS[0]] > 0.9


def test_min_variance_zero_fills_an_asset_with_no_expected_return() -> None:
    """a NaN expected return is set to zero in the return constraint, with a warning

    Left as NaN it would propagate into the constraint row and make the whole solve
    unsolvable for every asset, not only the one with the missing forecast.

    The single-date wrapper is called directly: the rolling entry point zero-fills the
    forecast panel before it gets here, so this guard is only reachable by a caller that
    solves one date at a time.
    """
    expected_returns = pd.Series(ALPHAS, index=TICKERS)
    expected_returns[TICKERS[1]] = np.nan
    with pytest.warns(UserWarning, match='NaN expected returns'):
        weights, outcome = wrapper_min_variance_target_return(
            pd_covar=pd.DataFrame(COVAR, index=TICKERS, columns=TICKERS),
            expected_returns=expected_returns, target_return=0.05,
            benchmark_weights=None, constraints=long_only())
    assert outcome.accepted
    assert np.isfinite(weights.to_numpy()).all()
    assert weights.sum() == pytest.approx(1.0, abs=1e-4)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
