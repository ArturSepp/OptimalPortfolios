"""EWMA covariance structure, annualisation, schedules and current/rolling agreement."""
from functools import lru_cache
from typing import Dict
import numpy as np
import pandas as pd
import pytest
import qis
from optimalportfolios.covar_estimation.ewma_covar_estimator import EwmaCovarEstimator
from optimalportfolios.tests.data.multiasset import load_multiasset_data

RETURNS_FREQ = 'ME'
SPAN = 24
WARMUP = 60
PSD_TOL = -1e-12


@lru_cache(maxsize=1)
def _prices() -> pd.DataFrame:
    """the committed offline fixture."""
    return load_multiasset_data().prices


@lru_cache(maxsize=4)
def _rolling(rebalancing_freq: str = 'QE', demean: bool = True) -> Dict[pd.Timestamp, pd.DataFrame]:
    """rolling EWMA covariances over the fixture, cached per configuration."""
    prices = _prices()
    estimator = EwmaCovarEstimator(returns_freq=RETURNS_FREQ, span=SPAN,
                                   rebalancing_freq=rebalancing_freq, demean=demean)
    return estimator.fit_rolling_covars(
        prices=prices, time_period=qis.TimePeriod(prices.index[WARMUP], prices.index[-1]))


def test_rolling_covariances_are_symmetric_and_psd() -> None:
    """
    a covariance matrix that is not PSD admits a portfolio with negative variance.

    The optimiser will find it: a negative-variance direction is unboundedly attractive to a
    minimum-variance objective, so this fails as a nonsensical allocation rather than an error.
    """
    covars = _rolling()
    assert len(covars) > 20, f'only {len(covars)} matrices; the fixture window is too short'
    for date, matrix in covars.items():
        values = matrix.to_numpy()
        np.testing.assert_allclose(values, values.T, atol=1e-14,
                                   err_msg=f'covariance at {date.date()} is not symmetric')
        assert np.linalg.eigvalsh(values).min() >= PSD_TOL, (
            f'covariance at {date.date()} has a negative eigenvalue')


def test_rolling_covariances_are_labelled_and_finite() -> None:
    """every matrix carries the universe on both axes and contains no NaN or Inf."""
    tickers = list(_prices().columns)
    for date, matrix in _rolling().items():
        assert list(matrix.index) == tickers, f'row labels differ at {date.date()}'
        assert list(matrix.columns) == tickers, f'column labels differ at {date.date()}'
        assert np.all(np.isfinite(matrix.to_numpy())), f'non-finite entry at {date.date()}'


def test_volatilities_are_finite_and_strictly_positive() -> None:
    """
    every diagonal entry is a real variance.

    The bound is deliberately loose at the bottom. The fixture's Liquidity group contains Cash,
    whose annualised volatility is about 0.05%; an earlier version of this test asserted a 0.5%
    floor and failed on it, which was the test being wrong rather than the estimator. A zero or
    negative variance is the defect worth catching here, and annualisation is checked separately
    below.
    """
    for date, matrix in _rolling().items():
        variances = np.diag(matrix.to_numpy())
        assert np.all(np.isfinite(variances)), f'non-finite variance at {date.date()}'
        assert variances.min() > 0.0, f'non-positive variance at {date.date()}'
        assert np.sqrt(variances).max() < 1.5, (
            f'implausible volatility {np.sqrt(variances).max():.4f} at {date.date()}')


def test_the_covariance_is_annualised() -> None:
    """
    the diagonal is an annual variance, not a monthly one.

    A missing annualisation factor is invisible in a covariance matrix and changes every reported
    risk number by √12. It is caught by scale rather than by a plausibility band: the most
    volatile instrument in this fixture reaches 40% annualised, and 11.7% un-annualised, so a
    threshold between the two separates the cases and nothing else does.
    """
    peak = max(np.sqrt(np.diag(m.to_numpy())).max() for m in _rolling().values())
    assert peak > 0.20, (
        f'the highest volatility across the sample is {peak:.4f}. Annualised it should reach '
        f'about 0.40 on this fixture; {peak:.4f} is consistent with a monthly figure that was '
        f'never scaled by sqrt(12)')


@pytest.mark.parametrize('rebalancing_freq, min_gap, max_gap', [('QE', 80, 100), ('YE', 350, 380)])
def test_rebalancing_schedule_matches_the_request(rebalancing_freq: str,
                                                  min_gap: int, max_gap: int) -> None:
    """the matrices are sampled at the frequency asked for, not at the return frequency."""
    dates = sorted(_rolling(rebalancing_freq=rebalancing_freq))
    gaps = pd.Series(dates).diff().dropna().dt.days
    assert gaps.min() >= min_gap, f'{rebalancing_freq}: shortest gap {gaps.min()} days'
    assert gaps.max() <= max_gap, f'{rebalancing_freq}: longest gap {gaps.max()} days'


def test_fit_current_covar_equals_the_last_rolling_matrix() -> None:
    """
    the two entry points agree on the same data.

    This is the check the source-adjacent ``ewma_covar_estimator_run`` prints as "EXACT MATCH" or
    "MISMATCH — investigate" and nobody reads. Monthly rebalancing puts the last rolling date on
    the last return date, so the two see identical data and must agree exactly; with a coarser
    rebalancing frequency they legitimately differ, because the rolling matrix is taken at the
    rebalancing date and the current one uses everything.
    """
    prices = _prices()
    estimator = EwmaCovarEstimator(returns_freq=RETURNS_FREQ, span=SPAN, rebalancing_freq='ME')
    rolling = estimator.fit_rolling_covars(
        prices=prices, time_period=qis.TimePeriod(prices.index[WARMUP], prices.index[-1]))
    current = estimator.fit_current_covar(prices=prices)
    last = max(rolling)
    assert last == prices.index[-1], (
        f'last rolling date {last.date()} is not the last price date {prices.index[-1].date()}, '
        f'so the two are not seeing the same data and this comparison is not the intended one')
    np.testing.assert_allclose(current.to_numpy(), rolling[last].to_numpy(), atol=1e-14)


def test_demean_changes_the_estimate() -> None:
    """
    a flag that changes nothing is either dead or being dropped before it is read.

    Not a claim about which setting is right: only that the argument reaches the estimator.
    """
    demeaned = _rolling(demean=True)
    raw = _rolling(demean=False)
    assert set(demeaned) == set(raw)
    difference = max(np.abs(demeaned[k].to_numpy() - raw[k].to_numpy()).max() for k in demeaned)
    assert difference > 1e-8, 'demean=False produced the same matrices as demean=True'
