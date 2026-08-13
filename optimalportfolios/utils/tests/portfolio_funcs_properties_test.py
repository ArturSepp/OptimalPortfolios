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

The filter's contract is subtler and is where the NaN-aware claim in `AGENTS.md` is either kept
or not: an asset with NaN variance is **removed**, while an asset with a genuine but tiny variance
is **clamped** to a floor and kept. The distinction matters for real universes — the committed
fixture's Cash instrument has an annualised volatility of about 5bps, below the 10bps floor, so it
is exactly the case the clamp exists for. Dropping it instead would silently shrink the investable
universe by one every time a low-volatility instrument appeared.
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
from optimalportfolios.utils.filter_nans import (filter_covar_and_vectors,
                                                 filter_covar_and_vectors_for_nans)

TICKERS = ['A', 'B', 'C', 'D']
VOLS = np.array([0.10, 0.15, 0.20, 0.25])
VARIANCE_FLOOR = 0.001 ** 2       # the module default: about 10bps of annualised volatility


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


def test_tiny_variances_are_clamped_and_kept_not_dropped() -> None:
    """
    the documented distinction: NaN is removed, near-zero is floored.

    Removing a low-volatility instrument would drop cash from a multi-asset universe whenever its
    realised volatility dipped, changing the investable set without anything reporting it.
    """
    covar, sigma, _ = _universe()
    nearly_riskless = sigma.copy()
    nearly_riskless[3, 3] = 1e-10

    filtered, _ = filter_covar_and_vectors_for_nans(
        pd_covar=pd.DataFrame(nearly_riskless, index=TICKERS, columns=TICKERS))

    assert list(filtered.index) == TICKERS, 'the near-zero variance asset was dropped, not clamped'
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
        pd_covar=pd.DataFrame(nearly_riskless, index=TICKERS, columns=TICKERS))
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


def test_zero_variance_assets_are_dropped_by_the_positional_filter() -> None:
    """
    ``filter_covar_and_vectors`` is the stricter sibling: it drops zero variance as well as NaN.

    Both are exported and they differ in exactly this, which is worth pinning so a caller reaching
    for one does not get the other's behaviour.
    """
    covar, sigma, _ = _universe()
    degenerate = sigma.copy()
    degenerate[1, 1] = 0.0
    filtered, _ = filter_covar_and_vectors(covar=degenerate, tickers=pd.Index(TICKERS))
    assert list(filtered.index) == ['A', 'C', 'D']


def test_the_fixtures_cash_instrument_survives_the_filter() -> None:
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
        filtered, _ = filter_covar_and_vectors_for_nans(pd_covar=covar)
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


def test_risk_contributions_sum_to_the_portfolio_volatility() -> None:
    """
    Euler's theorem: volatility is homogeneous of degree one in the weights, so the marginal
    contributions add up to it exactly. A decomposition that does not sum to the whole is not a
    decomposition, and it is reported beside the portfolio as though it were.
    """
    covar, sigma, weights = _universe()
    contributions = portfolio_funcs.compute_portfolio_risk_contributions(weights, sigma)
    assert contributions.sum() == pytest.approx(np.sqrt(weights @ sigma @ weights))


def test_normalised_risk_contributions_sum_to_one() -> None:
    """the labelled variant reports shares, so they total one and carry the tickers."""
    covar, sigma, weights = _universe()
    contributions = portfolio_funcs.compute_risk_contributions(pd.Series(weights, index=TICKERS),
                                                               covar)
    assert list(contributions.index) == TICKERS
    assert float(contributions.sum()) == pytest.approx(1.0)


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


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
