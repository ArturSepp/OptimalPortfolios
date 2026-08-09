"""
properties the covariance estimators must have, asserted rather than printed.

Both estimators had diagnostic scripts and no tests. `ewma_covar_estimator_local.py` computes the
agreement between `fit_current_covar` and the last matrix of `fit_rolling_covars`, prints either
"EXACT MATCH" or "MISMATCH — investigate", and returns; nothing reads the output, so a mismatch
has always been a line of console text rather than a failure. That comparison is the first test
here, and it holds exactly.

What is checked:

    structure          symmetric, positive semi-definite, correctly labelled, finite
    internal agreement fit_current_covar equals the last rolling matrix on the same data
    the flags matter   an argument that changes nothing is either dead or ignored
    factor identity    Σ_y = β Σ_x β' + diag(residual variance), exactly
    the floor          n_clusters above the universe size clamps instead of raising, which is
                       the factorlasso 0.10.1 fix this package's floor exists to require
    the defaults       PEARSON and ONE_MINUS_RHO, which the 6.3.0 and 6.4.0 changelog entries
                       promise reproduce the earlier clustering and which nothing checked

The panel is the committed `multiasset` fixture. Estimation is cached per configuration so the
whole module runs in a few seconds; `factorlasso` is a mandatory dependency of this package, so
the factor tests need no skip guard and must pass on a core install.
"""
# packages
from functools import lru_cache
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pytest
import qis

# optimalportfolios
from optimalportfolios import LassoModel, LassoModelType, DependenceMeasure, DistanceTransform
from optimalportfolios.covar_estimation.ewma_covar_estimator import EwmaCovarEstimator
from optimalportfolios.covar_estimation.factor_covar_estimator import FactorCovarEstimator
from optimalportfolios.examples.data.multiasset import load_multiasset_data

RETURNS_FREQ = 'ME'
SPAN = 24
WARMUP = 60
N_FACTORS = 4
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


@lru_cache(maxsize=2)
def _factor_inputs() -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, pd.DataFrame]]:
    """
    split the fixture into risk factors and assets.

    The first four instruments stand in as the factor block and the remaining fifteen as the
    assets; the split is arbitrary but fixed, and the identity being checked does not depend on
    which columns play which role.
    """
    prices = _prices()
    factors, assets = prices.iloc[:, :N_FACTORS], prices.iloc[:, N_FACTORS:]
    asset_returns = {RETURNS_FREQ: qis.to_returns(assets, freq=RETURNS_FREQ,
                                                  is_log_returns=True, drop_first=True)}
    return factors, assets, asset_returns


@lru_cache(maxsize=2)
def _current_factor_covars(n_clusters: int = None):
    """fit the factor model once; ``n_clusters`` selects the clustering model type."""
    factors, assets, asset_returns = _factor_inputs()
    if n_clusters is None:
        lasso_model = LassoModel(model_type=LassoModelType.LASSO, reg_lambda=1e-5, span=36)
    else:
        lasso_model = LassoModel(model_type=LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO,
                                 reg_lambda=1e-5, span=36, n_clusters=n_clusters)
    estimator = FactorCovarEstimator(lasso_model=lasso_model, factor_returns_freq=RETURNS_FREQ,
                                     factor_covar_span=SPAN)
    return estimator.fit_current_factor_covars(risk_factor_prices=factors,
                                               asset_returns_dict=asset_returns,
                                               assets=assets.columns)


# ───────────────────────────────────────────────────────────────────────────────
# EWMA structure
# ───────────────────────────────────────────────────────────────────────────────


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

    This is the check `ewma_covar_estimator_local.py` prints as "EXACT MATCH" or
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


# ───────────────────────────────────────────────────────────────────────────────
# Factor model
# ───────────────────────────────────────────────────────────────────────────────


def test_factor_covariance_reconstructs_from_its_parts() -> None:
    """
    Σ_y = β Σ_x β' + diag(residual variance).

    The identity the factor model exists to impose. If it fails, the assembled covariance is not
    the model's covariance and the risk decomposition reported beside it describes a different
    matrix from the one the optimiser used.
    """
    data = _current_factor_covars()
    betas = data.y_betas.to_numpy()
    factor_covar = data.x_covar.to_numpy()
    residual_var = data.y_variances['residual_var'].to_numpy()
    reconstructed = betas @ factor_covar @ betas.T + np.diag(residual_var)
    np.testing.assert_allclose(data.get_y_covar().to_numpy(), reconstructed, atol=1e-12)


def test_factor_covariance_off_diagonal_is_purely_systematic() -> None:
    """
    residual risk is diagonal by construction, so every off-diagonal entry is β Σ_x β'.

    Sharper than the identity above: it isolates the modelling assumption from the diagonal, where
    a residual term could hide a reconstruction error.
    """
    data = _current_factor_covars()
    betas = data.y_betas.to_numpy()
    systematic = betas @ data.x_covar.to_numpy() @ betas.T
    y_covar = data.get_y_covar().to_numpy()
    off_diagonal = ~np.eye(len(y_covar), dtype=bool)
    np.testing.assert_allclose(y_covar[off_diagonal], systematic[off_diagonal], atol=1e-12)


def test_factor_and_asset_covariances_are_symmetric_and_psd() -> None:
    """both the factor block and the assembled asset covariance are usable by an optimiser."""
    data = _current_factor_covars()
    for name, matrix in (('x_covar', data.x_covar.to_numpy()),
                         ('y_covar', data.get_y_covar().to_numpy())):
        np.testing.assert_allclose(matrix, matrix.T, atol=1e-12,
                                   err_msg=f'{name} is not symmetric')
        assert np.linalg.eigvalsh(matrix).min() >= PSD_TOL, f'{name} is not PSD'


def test_n_clusters_above_the_universe_size_clamps_instead_of_raising() -> None:
    """
    the defect that set this package's factorlasso floor at 0.10.1.

    factorlasso 0.10.0 raised when ``n_clusters`` exceeded the number of assets. A count
    calibrated on the full sample therefore failed at earlier dates holding fewer instruments,
    which is precisely a rolling estimation over a growing universe. 0.10.1 clamps to the universe
    size, which is what scipy's 'maxclust' criterion does natively. This test fails against a
    floor that slips back below 0.10.1.
    """
    _, assets, _ = _factor_inputs()
    n_assets = len(assets.columns)
    data = _current_factor_covars(n_clusters=n_assets + 10)
    clusters = data.clusters
    labels = set(clusters.values()) if isinstance(clusters, dict) else set(np.ravel(clusters))
    assert 0 < len(labels) <= n_assets, (
        f'asked for {n_assets + 10} clusters over {n_assets} assets and got {len(labels)}')


def test_clustering_defaults_are_the_documented_ones() -> None:
    """
    the defaults the 6.3.0 and 6.4.0 changelog entries promise, and nothing checked.

    Both entries state that the new ``distance_transform`` and ``dependence_measure`` parameters
    default to the pre-existing behaviour, so results computed before them are unchanged. That
    promise is only kept while the defaults stay put; changing either silently re-clusters every
    factor covariance in the stack.
    """
    model = LassoModel(model_type=LassoModelType.LASSO, reg_lambda=1e-5, span=36)
    assert model.distance_transform == DistanceTransform.ONE_MINUS_RHO
    assert model.dependence_measure == DependenceMeasure.PEARSON
    assert model.n_clusters is None, (
        'n_clusters now defaults to a value, so the fractional-distance cut is no longer the '
        'default clustering and the 6.4.0 changelog claim no longer holds')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
