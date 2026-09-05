"""
the utility layer: the NaN filter, and the portfolio statistics built on it.

Two modules, both reached on every rebalancing of every backtest and neither previously tested.
`filter_nans` decides which assets enter an optimisation; `portfolio_funcs` computes the numbers
reported beside the result. A defect in either is quiet: the optimisation still runs, and the
statistics still look like statistics.

Each formula is checked against its definition, with `Σ` the covariance, `w` the weights and
`σ = √diag(Σ)`:

    portfolio variance          w'Σw
    portfolio volatility        √(w'Σw)
    risk contributions          w ⊙ Σw / √(w'Σw), summing to the volatility by Euler's theorem
    diversification ratio       w'σ / √(w'Σw)
    tracking error              √((w - w_b)'Σ(w - w_b))
    turnover                    Σ|w - w₀|

The filter removes every asset with non-positive or NaN variance. A positive variance is retained
unchanged unless the caller explicitly supplies a floor, in which case smaller surviving diagonal
entries are raised to it. The committed fixture's Cash instrument has an annualised volatility of
about 5bps, so it exercises that opt-in floor without weakening the invalid-asset rule.
"""
# packages
from typing import Tuple

import numpy as np
import pandas as pd
import pytest
import qis

# optimalportfolios
import optimalportfolios.utils.portfolio_funcs as portfolio_funcs
from optimalportfolios.covar_estimation.ewma_covar_estimator import EwmaCovarEstimator
from optimalportfolios.tests.data.multiasset import load_multiasset_data
from optimalportfolios.utils.filter_nans import filter_covar_and_vectors_for_nans

TICKERS = ['A', 'B', 'C', 'D']
VOLS = np.array([0.10, 0.15, 0.20, 0.25])
VARIANCE_FLOOR = 0.001 ** 2


def _universe() -> Tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """
    a small well-conditioned covariance with equal pairwise correlation.

    Returns:
        the covariance as a frame, as an array, and a weight vector summing to one
    """
    corr = np.full((len(TICKERS), len(TICKERS)), 0.30)
    np.fill_diagonal(corr, 1.0)
    sigma = np.outer(VOLS, VOLS) * corr
    weights = np.array([0.40, 0.30, 0.20, 0.10])
    return pd.DataFrame(sigma, index=TICKERS, columns=TICKERS), sigma, weights


# ───────────────────────────────────────────────────────────────────────────────
# The NaN filter
# ───────────────────────────────────────────────────────────────────────────────


def test_nan_variance_assets_are_removed_and_vectors_follow() -> None:
    """
    an asset with no variance estimate cannot be optimised over, and its alpha must go with it.

    A covariance and a mean vector that disagree on the universe is the failure this prevents:
    the shapes still conform if the vector is merely reindexed, and the optimiser then pairs each
    asset with its neighbour's expected return.
    """
    covar, sigma, _ = _universe()
    broken = sigma.copy()
    broken[2, 2] = np.nan
    means = pd.Series([1.0, 2.0, 3.0, 4.0], index=TICKERS)

    filtered, vectors = filter_covar_and_vectors_for_nans(
        pd_covar=pd.DataFrame(broken, index=TICKERS, columns=TICKERS), vectors={'means': means})

    assert list(filtered.index) == ['A', 'B', 'D']
    assert list(filtered.columns) == ['A', 'B', 'D']
    assert list(vectors['means'].index) == ['A', 'B', 'D']
    assert vectors['means'].to_dict() == {'A': 1.0, 'B': 2.0, 'D': 4.0}


def test_non_finite_vector_assets_are_removed_from_every_aligned_input() -> None:
    """NaN or infinite solver vectors exclude their asset before reaching a backend."""
    covar, _, _ = _universe()
    means = pd.Series([1.0, np.nan, 3.0, 4.0], index=TICKERS)
    alphas = pd.Series([0.1, 0.2, np.inf, 0.4], index=TICKERS)

    filtered, vectors = filter_covar_and_vectors_for_nans(
        pd_covar=covar, vectors={'means': means, 'alphas': alphas},
        drop_non_finite_vectors=True)

    assert list(filtered.index) == ['A', 'D']
    assert vectors['means'].to_dict() == {'A': 1.0, 'D': 4.0}
    assert vectors['alphas'].to_dict() == {'A': 0.1, 'D': 0.4}


def test_tiny_positive_variances_are_unchanged_without_an_explicit_floor() -> None:
    """A positive variance remains valid and is not modified unless the caller requests a floor."""
    covar, sigma, _ = _universe()
    nearly_riskless = sigma.copy()
    nearly_riskless[3, 3] = 1e-10

    filtered, _ = filter_covar_and_vectors_for_nans(
        pd_covar=pd.DataFrame(nearly_riskless, index=TICKERS, columns=TICKERS))

    assert list(filtered.index) == TICKERS
    assert filtered.loc['D', 'D'] == pytest.approx(1e-10)


def test_an_explicit_variance_floor_clamps_tiny_positive_variances() -> None:
    """A supplied floor changes small positive diagonals without removing their assets."""
    covar, sigma, _ = _universe()
    nearly_riskless = sigma.copy()
    nearly_riskless[3, 3] = 1e-10

    filtered, _ = filter_covar_and_vectors_for_nans(
        pd_covar=pd.DataFrame(nearly_riskless, index=TICKERS, columns=TICKERS),
        variance_floor=VARIANCE_FLOOR,
    )

    assert list(filtered.index) == TICKERS
    assert filtered.loc['D', 'D'] == pytest.approx(VARIANCE_FLOOR)


def test_clamping_preserves_positive_semi_definiteness() -> None:
    """
    the claim the implementation makes in a comment, checked.

    Raising a diagonal entry adds a non-negative multiple of e_i e_i' to the matrix, so it cannot
    create a negative eigenvalue. Worth asserting because the optimiser's behaviour on a non-PSD
    matrix is unbounded rather than merely wrong.

    The input has to be PSD for the claim to mean anything — it is about *preserving* the
    property, not creating it. An earlier version of this test set a single diagonal entry to
    1e-12 while leaving that asset's covariances untouched, which is not a covariance matrix at
    all: it already had an eigenvalue of -0.0084 before the filter saw it, and the test failed on
    its own input. The tiny variance here comes from a consistent volatility vector instead.
    """
    vols = np.array([0.10, 0.15, 0.20, 1e-6])          # the fourth asset is genuinely riskless
    corr = np.full((len(TICKERS), len(TICKERS)), 0.30)
    np.fill_diagonal(corr, 1.0)
    nearly_riskless = np.outer(vols, vols) * corr
    assert np.linalg.eigvalsh(nearly_riskless).min() >= 0.0, 'the test input is not PSD'
    assert nearly_riskless[3, 3] < VARIANCE_FLOOR, 'the test input does not trigger the clamp'

    filtered, _ = filter_covar_and_vectors_for_nans(
        pd_covar=pd.DataFrame(nearly_riskless, index=TICKERS, columns=TICKERS),
        variance_floor=VARIANCE_FLOOR,
    )
    assert filtered.loc['D', 'D'] == pytest.approx(VARIANCE_FLOOR)
    assert np.linalg.eigvalsh(filtered.to_numpy()).min() >= -1e-14


def test_inclusion_indicators_remove_assets() -> None:
    """a zero indicator excludes an asset even when its variance is perfectly good."""
    covar, _, _ = _universe()
    indicators = pd.Series([1.0, 0.0, 1.0, 1.0], index=TICKERS)
    filtered, _ = filter_covar_and_vectors_for_nans(pd_covar=covar, inclusion_indicators=indicators)
    assert list(filtered.index) == ['A', 'C', 'D']


def test_invalid_strict_vectors_are_rejected_and_none_is_ignored() -> None:
    """Strict filtering ignores None but rejects unaligned or non-numeric vectors."""
    covar, _, _ = _universe()
    with pytest.raises(TypeError):
        filter_covar_and_vectors_for_nans(pd_covar=covar, vectors={'means': np.array([1.0, 2.0])})
    with pytest.raises(TypeError, match="vector must be pd.Series"):
        filter_covar_and_vectors_for_nans(
            pd_covar=covar, vectors={'means': np.array([1.0, 2.0])},
            drop_non_finite_vectors=True)
    with pytest.raises(TypeError, match="must contain numeric values"):
        filter_covar_and_vectors_for_nans(
            pd_covar=covar, vectors={'means': pd.Series(['a', 'b', 'c', 'd'], index=TICKERS)},
            drop_non_finite_vectors=True)

    filtered, vectors = filter_covar_and_vectors_for_nans(
        pd_covar=covar, vectors={'means': None}, drop_non_finite_vectors=True)
    assert list(filtered.index) == TICKERS
    assert vectors == {}


@pytest.mark.parametrize('variance_floor', [None, VARIANCE_FLOOR])
def test_zero_variance_assets_are_always_dropped(variance_floor: float | None) -> None:
    """An optional floor applies only after zero-variance assets have been removed."""
    covar, sigma, _ = _universe()
    degenerate = sigma.copy()
    degenerate[1, 1] = 0.0
    filtered, _ = filter_covar_and_vectors_for_nans(
        pd_covar=pd.DataFrame(degenerate, index=TICKERS, columns=TICKERS),
        variance_floor=variance_floor,
    )
    assert list(filtered.index) == ['A', 'C', 'D']


def test_the_fixtures_cash_instrument_survives_an_explicit_floor() -> None:
    """
    the clamp on real data rather than a constructed matrix.

    Cash in the fixture's Liquidity group dips under the 10bps floor on 9 of the 77 rebalancing
    dates, between 2014-06 and 2016-06. It must still be investable on those dates rather than
    disappearing from the universe for two years.

    The date is found rather than assumed: an earlier version took the last rebalancing date,
    where Cash estimates at 31bps and is comfortably above the floor, so the test passed without
    exercising the clamp at all.
    """
    prices = load_multiasset_data().prices
    estimator = EwmaCovarEstimator(returns_freq='ME', span=24, rebalancing_freq='QE')
    covars = estimator.fit_rolling_covars(
        prices=prices, time_period=qis.TimePeriod(prices.index[60], prices.index[-1]))
    assert 'Cash' in next(iter(covars.values())).index, (
        'the fixture no longer has a Cash instrument; update this test')

    below_floor = {date: matrix for date, matrix in covars.items()
                   if float(matrix.loc['Cash', 'Cash']) < VARIANCE_FLOOR}
    assert below_floor, (
        'Cash never estimates below the variance floor on this fixture, so this test is not '
        'exercising the clamp. Either the estimator or the fixture has changed')

    for date, covar in below_floor.items():
        filtered, _ = filter_covar_and_vectors_for_nans(
            pd_covar=covar,
            variance_floor=VARIANCE_FLOOR,
        )
        assert 'Cash' in filtered.index, f'Cash was dropped rather than clamped at {date.date()}'
        assert filtered.loc['Cash', 'Cash'] == pytest.approx(VARIANCE_FLOOR)
        assert len(filtered.index) == len(covar.index), (
            f'the filter changed the universe size at {date.date()}')


# ───────────────────────────────────────────────────────────────────────────────
# Portfolio statistics
# ───────────────────────────────────────────────────────────────────────────────


def test_portfolio_variance_and_volatility_agree_with_their_definitions() -> None:
    """w'Σw, and its square root."""
    covar, sigma, weights = _universe()
    assert portfolio_funcs.compute_portfolio_variance(weights, sigma) == pytest.approx(
        weights @ sigma @ weights)
    assert portfolio_funcs.compute_portfolio_vol(sigma, weights) == pytest.approx(
        np.sqrt(weights @ sigma @ weights))


def test_portfolio_volatility_is_the_same_for_arrays_and_frames() -> None:
    """
    the labelled and unlabelled paths must not diverge.

    ``compute_portfolio_vol`` accepts either, and a caller mixing them across a codebase should
    not have to know which one a given call site used.
    """
    covar, sigma, weights = _universe()
    from_arrays = portfolio_funcs.compute_portfolio_vol(sigma, weights)
    from_frames = portfolio_funcs.compute_portfolio_vol(covar, pd.Series(weights, index=TICKERS))
    assert from_arrays == pytest.approx(from_frames)


def test_diversification_ratio_matches_its_definition() -> None:
    """w'σ / √(w'Σw), and it exceeds one whenever the assets are not perfectly correlated."""
    covar, sigma, weights = _universe()
    ratio = portfolio_funcs.calculate_diversification_ratio(weights, sigma)
    assert ratio == pytest.approx(weights @ VOLS / np.sqrt(weights @ sigma @ weights))
    assert ratio > 1.0


def test_diversification_ratio_is_one_under_perfect_correlation() -> None:
    """
    with every correlation at one there is nothing to diversify.

    The degenerate case the ratio is defined against; a ratio above one here would mean the
    denominator is not the portfolio volatility.
    """
    sigma = np.outer(VOLS, VOLS)          # all correlations equal to one
    weights = np.array([0.40, 0.30, 0.20, 0.10])
    assert portfolio_funcs.calculate_diversification_ratio(weights, sigma) == pytest.approx(1.0)


def test_rounding_weights_preserves_the_budget() -> None:
    """
    rounding must not lose or invent allocation.

    Rounding each weight independently is the obvious implementation and the wrong one: three
    weights of 33.333% round to 99.99%. The result is in percent, so it totals 100.
    """
    weights = pd.Series([1 / 3, 1 / 3, 1 / 3, 0.0], index=TICKERS)
    rounded = portfolio_funcs.round_weights_to_pct(weights, decimals=2)
    assert float(rounded.sum()) == pytest.approx(100.0, abs=1e-9)
    assert list(rounded.index) == TICKERS


def test_tracking_error_and_turnover_match_their_definitions() -> None:
    """√((w - w_b)'Σ(w - w_b)) and Σ|w - w₀|, with the portfolio and benchmark volatilities."""
    covar, sigma, weights = _universe()
    weights_series = pd.Series(weights, index=TICKERS)
    benchmark = pd.Series(0.25, index=TICKERS)
    previous = pd.Series([0.30, 0.30, 0.20, 0.20], index=TICKERS)
    alphas = pd.Series([0.01, 0.02, -0.01, 0.00], index=TICKERS)

    te_vol, turnover, port_alpha, port_vol, benchmark_vol = \
        portfolio_funcs.compute_tre_turnover_stats(covar=sigma, benchmark_weights=benchmark,
                                                   weights=weights_series, weights_0=previous,
                                                   alphas=alphas)
    active = weights - benchmark.to_numpy()
    assert te_vol == pytest.approx(np.sqrt(active @ sigma @ active))
    assert turnover == pytest.approx(np.abs(weights - previous.to_numpy()).sum())
    assert port_alpha == pytest.approx(alphas.to_numpy() @ weights)
    assert port_vol == pytest.approx(np.sqrt(weights @ sigma @ weights))
    assert benchmark_vol == pytest.approx(
        np.sqrt(benchmark.to_numpy() @ sigma @ benchmark.to_numpy()))


def test_tracking_error_is_zero_against_the_portfolio_itself() -> None:
    """the degenerate case, and a sharp check that the active weights are differenced correctly."""
    covar, sigma, weights = _universe()
    weights_series = pd.Series(weights, index=TICKERS)
    te_vol, turnover, _, _, _ = portfolio_funcs.compute_tre_turnover_stats(
        covar=sigma, benchmark_weights=weights_series, weights=weights_series,
        weights_0=weights_series)
    assert te_vol == pytest.approx(0.0, abs=1e-12)
    assert turnover == pytest.approx(0.0, abs=1e-12)


def test_tre_turnover_stats_match_public_risk_model_on_seeded_psd_covariances() -> None:
    """Pin solver-hot-path risk statistics to the canonical qis formulas."""
    # Seed 20260810; portfolio and benchmark vol are TE against a zero benchmark.
    rng = np.random.default_rng(20260810)
    tickers = pd.Index([f'asset_{idx}' for idx in range(7)])
    date = pd.Timestamp('2026-06-30')
    zero_weights = pd.Series(0.0, index=tickers)

    for sample in range(8):
        root = rng.normal(scale=0.12, size=(len(tickers), len(tickers)))
        covar_values = root @ root.T + np.diag(rng.uniform(0.001, 0.004, len(tickers)))
        covar = pd.DataFrame(covar_values, index=tickers, columns=tickers)
        benchmark = pd.Series(rng.dirichlet(np.ones(len(tickers))), index=tickers)
        if sample == 0:
            weights = pd.Series(
                [0.75, -0.35, 0.40, -0.20, 0.25, -0.05, 0.20], index=tickers)
            assert (weights < 0.0).any(), 'the property set must include a long-short portfolio'
        else:
            weights = pd.Series(rng.dirichlet(np.ones(len(tickers))), index=tickers)
        previous = pd.Series(rng.dirichlet(np.ones(len(tickers))), index=tickers)

        te_vol, _, _, port_vol, benchmark_vol = portfolio_funcs.compute_tre_turnover_stats(
            covar=covar_values,
            benchmark_weights=benchmark,
            weights=weights,
            weights_0=previous,
        )
        risk_model = qis.RiskModel(covar={date: covar})
        np.testing.assert_allclose(
            [te_vol, port_vol, benchmark_vol],
            [
                risk_model.compute_tre_at_date(benchmark, weights, date),
                risk_model.compute_tre_at_date(zero_weights, weights, date),
                risk_model.compute_tre_at_date(zero_weights, benchmark, date),
            ],
            rtol=1e-12,
            atol=1e-16,
        )


def test_filter_drops_a_bad_asset_from_the_companion_vectors_too() -> None:
    """A vector supplied alongside the covariance is filtered to the same surviving assets.

    The vectors are expected returns, alpha or weight bounds indexed by asset. If the covariance
    loses an asset and a vector does not, every downstream ``covar @ vector`` misaligns by one
    position, which still computes and still returns a plausible-looking number.
    """
    tickers = pd.Index(['A', 'B', 'C'])
    covar = np.diag([0.04, np.nan, 0.09])
    vectors = {'means': pd.Series([0.1, 0.2, 0.3], index=tickers)}

    clean_covar, good_vectors = filter_covar_and_vectors_for_nans(
        pd_covar=pd.DataFrame(covar, index=tickers, columns=tickers),
        vectors=vectors,
    )

    assert list(clean_covar.columns) == ['A', 'C']
    assert list(good_vectors['means'].index) == ['A', 'C']
    pd.testing.assert_series_equal(
        good_vectors['means'], vectors['means'].loc[['A', 'C']],
    )


def test_risk_contribution_table_defaults_the_budget_to_zeros() -> None:
    """Omitting ``risk_budget`` yields a zero column rather than a missing one."""
    tickers = pd.Index(['A', 'B'])
    clean_covar = pd.DataFrame(np.diag([0.04, 0.09]), index=tickers, columns=tickers)
    weights = pd.Series([0.6, 0.4], index=tickers)

    table = portfolio_funcs.compute_portfolio_risk_contribution_outputs(
        weights=weights, clean_covar=clean_covar,
    )

    assert (table['Risk Budget'] == 0.0).all()
    assert list(table.index) == list(tickers)
