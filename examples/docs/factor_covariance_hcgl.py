"""Canonical script of docs/factor_covariance_hcgl.md.

The page's Python blocks are excerpts of this file and run here in the same order; every number
and property the page states is asserted after them against a reference computed a different
way: the factor covariance as an explicit EWMA weighted sum of factor log returns, residual
variances as EWMA-weighted squared residuals of the fitted loadings, the empirical residual
correlation from quarterly sums of the stored residuals, separate single-cadence fits for the
cadence penalties, and current fits on inputs sliced through each rolling date. The script runs
offline after ``pip install optimalportfolios`` and draws its panel from a fixed seed:

    python -m examples.docs.factor_covariance_hcgl

``exhibit`` draws the page's figure; ``tools/docs_analytics/teaching.py`` calls it with the
constants below and records their values.
"""
from dataclasses import fields, replace
from importlib.metadata import version
import re

import numpy as np
import pandas as pd
import qis
from factorlasso import CurrentFactorCovarData, LassoModel, LassoModelType, VarianceColumns

import optimalportfolios as op

SEED = 5
FACTORS = ['Equity', 'Rates', 'Credit', 'Commodities']
MONTHLY = ['Govt', 'Credit', 'DM equity', 'EM equity', 'Commodities']
QUARTERLY = ['Private equity', 'Real estate', 'Infrastructure']
ASSETS = MONTHLY + QUARTERLY
# Generating monthly loadings of the eight assets (rows) on the four factors (columns).
LOADINGS = [
    [0.0, 1.0, 0.0, 0.0],
    [0.1, 0.5, 0.7, 0.0],
    [1.0, 0.0, 0.0, 0.0],
    [1.1, 0.0, 0.2, 0.2],
    [0.2, 0.0, 0.0, 1.0],
    [0.8, 0.0, 0.3, 0.0],
    [0.4, 0.4, 0.2, 0.0],
    [0.3, 0.3, 0.0, 0.2],
]
FACTOR_VOLS = [0.045, 0.02, 0.02, 0.06]  # monthly
FACTOR_CORR = [
    [1.0, -0.2, 0.5, 0.3],
    [-0.2, 1.0, 0.1, -0.1],
    [0.5, 0.1, 1.0, 0.2],
    [0.3, -0.1, 0.2, 1.0],
]
RESIDUAL_VOLS = [0.003, 0.006, 0.01, 0.02, 0.02, 0.008, 0.006, 0.005]  # monthly
PRIVATE_SHOCK_VOL = 0.006  # a monthly residual shock common to the three private assets
SPANS = {'ME': 36, 'QE': 12}
AS_OF = '2023-12-31'
PERIODS_PER_YEAR = {'ME': 12.0, 'QE': 4.0}
ROLLING_KEYS = ['2023-12-31', '2024-03-31', '2024-06-30', '2024-09-30', '2024-12-31']


def installed_version(package: str) -> tuple:
    """Return the first three numeric parts of an installed distribution's version."""
    return tuple(int(part) for part in re.findall(r'\d+', version(package))[:3])


def simulated_panel(seed: int) -> tuple:
    """Simulate month-end factor prices and asset log returns on a monthly and a quarterly grid."""
    dates = pd.date_range('2004-12-31', '2024-12-31', freq='ME')
    rng = np.random.default_rng(seed)
    factor_covar = np.outer(FACTOR_VOLS, FACTOR_VOLS) * np.array(FACTOR_CORR)
    factors = rng.multivariate_normal(np.full(4, 0.004), factor_covar, size=len(dates) - 1)
    residuals = rng.normal(0.0, RESIDUAL_VOLS, size=(len(dates) - 1, len(ASSETS)))
    residuals[:, 5:] += rng.normal(0.0, PRIVATE_SHOCK_VOL, size=(len(dates) - 1, 1))
    monthly = pd.DataFrame(factors @ np.array(LOADINGS).T + residuals,
                           index=dates[1:], columns=ASSETS)
    factor_prices = pd.DataFrame(
        100.0 * np.exp(np.vstack([np.zeros(4), np.cumsum(factors, axis=0)])),
        index=dates, columns=FACTORS)
    buckets = {'ME': monthly[MONTHLY], 'QE': monthly[QUARTERLY].resample('QE').sum()}
    return factor_prices, buckets


def hcgl_model(reg_lambda: float = 1e-5) -> LassoModel:
    """Return a fresh HCGL model: the article's penalty, 36 months and 12 quarters of span."""
    return LassoModel(model_type=LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO,
                      reg_lambda=reg_lambda, span_freq_dict={'ME': 36, 'QE': 12})


def ewma_second_moment(values: np.ndarray, span: float) -> np.ndarray:
    """Zero-seeded EWMA second moment at the last row, written as a finite weighted sum."""
    decay = 1 - 2 / (span + 1)
    weights = (1 - decay) * decay ** np.arange(len(values) - 1, -1, -1)
    return np.einsum('t,ti,tj->ij', weights, values, values)


def factor_covar_reference(prices: pd.DataFrame, span: float,
                           periods_per_year: float) -> np.ndarray:
    """Annual factor covariance: EWMA-demeaned log returns in an explicit weighted sum."""
    returns = pd.DataFrame(np.diff(np.log(prices.to_numpy()), axis=0))
    centred = (returns - returns.ewm(span=span, adjust=False).mean()).iloc[1:]
    return periods_per_year * ewma_second_moment(centred.to_numpy(), span)


def residual_variance_reference(prices: pd.DataFrame, returns: pd.DataFrame,
                                betas: pd.DataFrame, span: float,
                                periods_per_year: float) -> pd.Series:
    """Annual residual variance: AN times the EWMA-weighted mean squared demeaned residual.

    The first factor return is missing, so its EWMA mean starts from zero, as in FactorLasso.
    """
    factors = np.log(prices.reindex(returns.index, method='ffill')).diff().fillna(0.0)
    x = (factors - factors.ewm(span=span, adjust=False).mean()).iloc[1:].to_numpy()
    y = (returns - returns.ewm(span=span, adjust=False).mean()).iloc[1:].to_numpy()
    decay = 1 - 2 / (span + 1)
    weights = decay ** np.arange(len(y) - 1, -1, -1)
    squared = (y - x @ betas.loc[returns.columns].to_numpy().T) ** 2
    return pd.Series(periods_per_year * (weights / weights.sum()) @ squared,
                     index=returns.columns)


def residual_correlation_reference(residuals: pd.DataFrame, scale: pd.Series,
                                   span: float) -> pd.DataFrame:
    """Quarterly residual correlation: undo the annual scale, sum complete quarters, EWMA."""
    raw = residuals / scale
    sums = raw.resample('QE').sum()
    counts = raw.notna().resample('QE').sum()
    needed = pd.Series({asset: 3 if asset in MONTHLY else 1 for asset in raw.columns})
    complete = sums.where(counts.eq(needed, axis=1)).dropna()
    centred = (complete - complete.ewm(span=span, adjust=False).mean()).iloc[1:]
    moment = ewma_second_moment(centred.to_numpy(), span)
    vols = np.sqrt(np.diag(moment))
    return pd.DataFrame(moment / np.outer(vols, vols), index=raw.columns, columns=raw.columns)


def systematic(data: CurrentFactorCovarData) -> np.ndarray:
    """Return beta Sigma_F beta' of a fitted decomposition as an explicit matrix product."""
    beta = data.y_betas.to_numpy()
    return np.einsum('if,fg,jg->ij', beta, data.x_covar.to_numpy(), beta)


def residual_vars(data: CurrentFactorCovarData) -> pd.Series:
    """Return the annual residual variances, the diagonal of D."""
    return data.y_variances[VarianceColumns.RESIDUAL_VARS.value]


def history_through(factor_prices: pd.DataFrame, buckets: dict, date) -> dict:
    """Slice the factor prices and every return bucket through ``date``."""
    return {'risk_factor_prices': factor_prices.loc[:date],
            'asset_returns_dict': {freq: r.loc[:date] for freq, r in buckets.items()},
            'estimation_date': pd.Timestamp(date)}


def private_weights() -> np.ndarray:
    """Equal weights in the three private assets and none in the liquid ones."""
    return np.array([0.0] * len(MONTHLY) + [1.0 / len(QUARTERLY)] * len(QUARTERLY))


def residual_block(data: CurrentFactorCovarData, rho: float) -> np.ndarray:
    """S[(1 - rho) I + rho R]S from the stored variances and the prepared correlation."""
    vols = np.sqrt(residual_vars(data).to_numpy())
    corr = data.residual_correlation.correlation.loc[ASSETS, ASSETS].to_numpy()
    return vols[:, None] * ((1 - rho) * np.eye(len(vols)) + rho * corr) * vols[None, :]


def check_empirical_residuals(factor_prices: pd.DataFrame, buckets: dict, data,
                              empirical: op.FactorCovarEstimator, empirical_data,
                              blended: pd.DataFrame) -> None:
    """Assert the page's statements on empirical residual covariance."""
    history = history_through(factor_prices, buckets, AS_OF)
    as_of = history['estimation_date']
    prepared = empirical_data.residual_correlation
    # The default grid is the lowest native frequency with its beta span, QE and 12; the
    # estimate records its last complete period and its fit date.
    assert (prepared.frequency, prepared.span) == ('QE', 12)
    assert prepared.observation_date == prepared.estimation_date == as_of
    # The stored residuals carry the multiplier AN_i: 12 monthly and 4 quarterly.
    metadata = empirical_data.residual_metadata.loc[ASSETS]
    assert metadata['residual_scale'].tolist() == [12.0] * 5 + [4.0] * 3
    assert metadata['annualisation_factor'].tolist() == metadata['residual_scale'].tolist()
    # R is the EWMA correlation of complete quarterly sums of the raw residuals, span 12.
    reference = residual_correlation_reference(empirical_data.residuals,
                                               metadata['residual_scale'], 12)
    np.testing.assert_allclose(prepared.correlation.loc[ASSETS, ASSETS],
                               reference.loc[ASSETS, ASSETS], rtol=0, atol=1e-12)
    # The empirical fit has the loadings and residual variances of the orthogonal one.
    np.testing.assert_allclose(empirical_data.y_betas, data.y_betas, rtol=0, atol=1e-12)
    np.testing.assert_allclose(residual_vars(empirical_data), residual_vars(data),
                               rtol=1e-12, atol=0)
    # The shared method assembles beta Sigma_F beta' + S[(1 - rho) I + rho R]S at rho = 0.5.
    np.testing.assert_allclose(blended, systematic(data) + residual_block(empirical_data, 0.5),
                               rtol=1e-10, atol=1e-15)
    # At every rho the diagonal is the fitted residual variances; rho = 0 is the orthogonal
    # model; D is positive semidefinite.
    orthogonal = data.get_y_covar().to_numpy()
    for rho in (0.0, 0.5, 1.0):
        matrix = empirical_data.get_y_covar(residual_type='empirical', residual_corr_weight=rho)
        np.testing.assert_allclose(np.diag(matrix), np.diag(orthogonal), rtol=1e-14, atol=0)
        assert np.linalg.eigvalsh(residual_block(empirical_data, rho)).min() > -1e-15
    np.testing.assert_allclose(
        empirical_data.get_y_covar(residual_type='empirical', residual_corr_weight=0.0),
        orthogonal, rtol=0, atol=1e-16)
    # residual_var_weight = 0 removes exactly D, off-diagonal entries included, and a weight
    # scales the whole residual block.
    no_residual = replace(empirical, lasso_model=hcgl_model()).fit_current_covar(
        **history, residual_var_weight=0.0)
    np.testing.assert_allclose(no_residual, systematic(data), rtol=1e-10, atol=1e-15)
    np.testing.assert_allclose(blended - no_residual, residual_block(empirical_data, 0.5),
                               rtol=1e-9, atol=1e-15)
    block = empirical_data.get_residual_covar(residual_type='empirical')
    offdiagonal = block.to_numpy() - np.diag(np.diag(block))
    assert np.abs(offdiagonal).max() > 1e-4
    np.testing.assert_allclose(
        empirical_data.get_residual_covar(residual_type='empirical', residual_var_weight=0.35),
        0.35 * block, rtol=1e-14, atol=0)
    # Insight: equal weights in the private assets have residual variance 0.00019 with
    # orthogonal residuals and 0.00040 with empirical ones, while no asset variance moves.
    w = private_weights()
    orthogonal_residual = w @ np.diag(residual_vars(data)) @ w
    empirical_residual = w @ residual_block(empirical_data, 1.0) @ w
    assert round(orthogonal_residual, 5) == 0.00019 and round(empirical_residual, 5) == 0.00040
    assert empirical_residual > 2 * orthogonal_residual
    assert round(w @ residual_block(empirical_data, 0.5) @ w, 5) == 0.00030
    # A correlation is never served before its availability date.
    try:
        prepared.get_corr(as_of - pd.Timedelta(days=1))
    except ValueError:
        pass
    else:
        raise AssertionError('a correlation was served before its availability date')
    # Decomposition getters default to orthogonal residuals, while the shared estimator method
    # uses the configured type.
    pd.testing.assert_frame_equal(empirical_data.get_y_covar(), data.get_y_covar(),
                                  check_exact=False, rtol=1e-12, atol=1e-16)
    # Empirical current fits truncate every input at estimation_date.
    whole = replace(empirical, lasso_model=hcgl_model()).fit_current_covar(
        risk_factor_prices=factor_prices, asset_returns_dict=buckets, estimation_date=as_of)
    np.testing.assert_allclose(whole, blended, rtol=1e-10, atol=1e-15)
    # A trailing incomplete quarter is excluded: a fit at the end of February 2024 observes
    # the quarter to December 2023 and is available from February.
    february = replace(empirical, lasso_model=hcgl_model()).fit_current_factor_covars(
        **history_through(factor_prices, buckets, '2024-02-29')).residual_correlation
    assert february.observation_date == as_of
    assert february.estimation_date == pd.Timestamp('2024-02-29')
    # An internal gap fails instead of being filled.
    gapped = {'ME': buckets['ME'].copy(), 'QE': buckets['QE']}
    gapped['ME'].iloc[100, 0] = np.nan
    try:
        replace(empirical, lasso_model=hcgl_model()).fit_current_factor_covars(
            **history_through(factor_prices, gapped, AS_OF))
    except ValueError:
        pass
    else:
        raise AssertionError('an internal residual gap was accepted')
    # residual_covar_span counts common periods; an explicitly coarser residual_covar_freq
    # converts the quarterly decay to the new grid; a monthly-only universe stays monthly.
    spanned = replace(empirical, lasso_model=hcgl_model(), residual_covar_span=8)
    assert spanned.fit_current_factor_covars(**history).residual_correlation.span == 8
    yearly = replace(empirical, lasso_model=hcgl_model(), residual_covar_freq='YE')
    annual = yearly.fit_current_factor_covars(**history).residual_correlation
    decay = (1 - 2 / 13) ** 4
    assert annual.frequency == 'YE' and np.isclose(annual.span, (1 + decay) / (1 - decay))
    monthly = replace(empirical, lasso_model=hcgl_model()).fit_current_factor_covars(
        risk_factor_prices=history['risk_factor_prices'],
        asset_returns_dict={'ME': history['asset_returns_dict']['ME']},
        estimation_date=as_of).residual_correlation
    assert (monthly.frequency, monthly.span) == ('ME', 36)
    # The retention weight defaults to 1, lies in [0, 1] and applies only to empirical residuals.
    assert op.FactorCovarEstimator().residual_corr_weight == 1.0
    for bad in ({'residual_type': 'empirical', 'residual_corr_weight': 1.5},
                {'residual_type': 'orthogonal', 'residual_corr_weight': 0.5}):
        try:
            op.FactorCovarEstimator(lasso_model=hcgl_model(), **bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'{bad} was accepted')
    # Rolling fits at month ends hold the quarterly correlation between complete quarters,
    # while the loadings update; half retention halves the empirical off-diagonal dependence.
    rolling = replace(empirical, lasso_model=hcgl_model(), rebalancing_freq='ME') \
        .fit_rolling_factor_covars(risk_factor_prices=factor_prices, asset_returns_dict=buckets,
                                   time_period=qis.TimePeriod('2023-12-31', '2024-12-31'))
    dates = list(rolling.get_y_covars())
    assert len(dates) == 13 and all(date.is_month_end for date in dates)
    vintages = rolling.get_residual_correlations()
    assert list(vintages) == list(pd.to_datetime(ROLLING_KEYS))
    assert rolling.get_beta('Equity')['DM equity'].nunique() == len(dates)
    assert rolling.get_residual_vars()['DM equity'].nunique() == len(dates)
    orthogonal_covars = rolling.get_y_covars()
    half = rolling.get_y_covars(residual_type='empirical', residual_corr_weight=0.5)
    full = rolling.get_y_covars(residual_type='empirical')
    for date, covar in orthogonal_covars.items():
        np.testing.assert_allclose(half[date] - covar, 0.5 * (full[date] - covar),
                                   rtol=1e-10, atol=1e-16)
    # An as-of query before the first fit is refused rather than backdated.
    try:
        rolling.get_y_covars(dates=pd.DatetimeIndex(['2023-11-30']))
    except ValueError:
        pass
    else:
        raise AssertionError('a query before the first fit was answered')


def check_reporting(factor_prices: pd.DataFrame, buckets: dict, data,
                    rolling) -> None:
    """Assert what the reporting functions return, with a non-interactive backend."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # One snapshot gives four figures: factor correlations, asset correlations, clusters and
    # the betas with their fit statistics.
    figures = op.plot_current_covar_data(data)
    assert len(figures) == 4
    plt.close('all')
    # The report refits the rolling decompositions and returns one snapshot table per date.
    estimator = op.FactorCovarEstimator(lasso_model=hcgl_model(), factor_returns_freq='ME',
                                        factor_covar_span=36)
    figures, tables = op.run_rolling_covar_report(
        risk_factor_prices=factor_prices, prices=None, covar_estimator=estimator,
        time_period=qis.TimePeriod('2023-12-31', '2024-12-31'), asset_returns_dict=buckets,
        assets=ASSETS, is_plot=False)
    assert figures == []
    assert list(tables) == [pd.Timestamp(key).strftime('%d%b%Y') for key in ROLLING_KEYS]
    for key, snapshot in rolling.get_snapshot().items():
        pd.testing.assert_frame_equal(tables[key.strftime('%d%b%Y')], snapshot,
                                      check_exact=False, rtol=1e-8, atol=1e-12)


def main() -> None:
    """Run the page's blocks in order and assert every number and property it states."""
    factor_prices, buckets = simulated_panel(SEED)

    # 241 month-end factor prices; 240 monthly log returns of five liquid assets and 80
    # quarterly log returns, sums of three months, of three private assets.
    assert factor_prices.shape == (241, 4) and buckets['ME'].shape == (240, 5)
    assert buckets['QE'].shape == (80, 3)
    assert buckets['QE'].index[-1] == pd.Timestamp('2024-12-31')
    assert set(buckets['ME'].columns).isdisjoint(buckets['QE'].columns)

    estimator = op.FactorCovarEstimator(
        lasso_model=hcgl_model(), factor_returns_freq='ME', factor_covar_span=36,
        rebalancing_freq='QE')
    as_of = pd.Timestamp('2023-12-31')
    history = {'risk_factor_prices': factor_prices.loc[:as_of],
               'asset_returns_dict': {freq: r.loc[:as_of] for freq, r in buckets.items()},
               'estimation_date': as_of}
    data = estimator.fit_current_factor_covars(**history)
    covar = data.get_y_covar()
    factor_only = data.get_y_covar(residual_var_weight=0.0)

    # Eight asset rows, four factor columns, CLARABEL; clusters are labelled by cadence.
    assert data.estimation_date == as_of and data.y_betas.shape == (8, 4)
    assert data.y_betas.index.tolist() == ASSETS and data.y_betas.columns.tolist() == FACTORS
    assert estimator.lasso_model.solver == 'CLARABEL'
    assert data.clusters.loc[MONTHLY].str.startswith('ME:').all()
    assert data.clusters.loc[QUARTERLY].str.startswith('QE:').all()
    assert hcgl_model().linkage_method == 'ward'
    assert {'residual_var', 'r2', 'insample_alpha'} <= set(data.y_variances.columns)
    # Sigma_F is the demeaned monthly EWMA of factor log returns at span 36, times 12.
    factor_covar = factor_covar_reference(factor_prices.loc[:as_of], 36, 12)
    np.testing.assert_allclose(data.x_covar, factor_covar, rtol=1e-10, atol=1e-14)
    # D is diagonal: AN times the EWMA-weighted mean squared residual of the fitted loadings
    # on EWMA-demeaned returns, with each bucket's span, 12 monthly and 4 quarterly.
    variances = pd.concat([
        residual_variance_reference(factor_prices.loc[:as_of], returns, data.y_betas,
                                    SPANS[freq], PERIODS_PER_YEAR[freq])
        for freq, returns in history['asset_returns_dict'].items()])
    np.testing.assert_allclose(residual_vars(data).loc[ASSETS], variances.loc[ASSETS],
                               rtol=1e-8, atol=1e-14)
    # The assembled covariance is beta Sigma_F beta' + D; residual weight 0 removes exactly D.
    beta = data.y_betas.to_numpy()
    expected = np.einsum('if,fg,jg->ij', beta, factor_covar, beta)
    np.testing.assert_allclose(covar, expected + np.diag(variances.loc[ASSETS]),
                               rtol=1e-8, atol=1e-14)
    np.testing.assert_allclose(factor_only, expected, rtol=1e-10, atol=1e-15)
    np.testing.assert_allclose(covar - factor_only, np.diag(residual_vars(data)),
                               rtol=0, atol=1e-15)
    assert np.linalg.eigvalsh(covar).min() > 0
    # The shared method returns the same assembly, and its residual_var_weight scales D.
    fresh = replace(estimator, lasso_model=hcgl_model())
    np.testing.assert_allclose(fresh.fit_current_covar(**history, residual_var_weight=0.35),
                               expected + 0.35 * np.diag(residual_vars(data)),
                               rtol=1e-10, atol=1e-15)
    # The function behind the current fit gives the same decomposition.
    direct = op.estimate_lasso_factor_covar_data(
        risk_factor_prices=history['risk_factor_prices'],
        asset_returns_dict=history['asset_returns_dict'], lasso_model=hcgl_model(),
        factor_returns_freq='ME', factor_covar_span=36, estimation_date=as_of)
    np.testing.assert_allclose(direct.get_y_covar(), covar, rtol=1e-10, atol=1e-15)
    # The top-level demean field is not read: demean=False reproduces the fit.
    undemeaned = replace(estimator, lasso_model=hcgl_model(), demean=False) \
        .fit_current_factor_covars(**history)
    np.testing.assert_allclose(undemeaned.x_covar, data.x_covar, rtol=0, atol=1e-15)
    np.testing.assert_allclose(undemeaned.y_betas, data.y_betas, rtol=0, atol=1e-12)
    # is_apply_vol_normalised_returns selects the normalised qis kernel for Sigma_F.
    normalised = replace(estimator, lasso_model=hcgl_model(), is_apply_vol_normalised_returns=True)
    normalised_covar = normalised.fit_current_factor_covars(**history).x_covar
    np.testing.assert_allclose(
        normalised_covar,
        op.estimate_current_ewma_covar(factor_prices.loc[:as_of], returns_freq='ME', span=36,
                                       is_apply_vol_normalised_returns=True),
        rtol=1e-10, atol=1e-15)
    assert np.abs(normalised_covar - data.x_covar).max().max() > 1e-4
    # A supplied annual factor covariance is used as given, not multiplied by 12 again.
    supplied = data.x_covar * 1.7
    pd.testing.assert_frame_equal(
        replace(estimator, lasso_model=hcgl_model()).fit_current_factor_covars(
            **history, x_covar=supplied).x_covar, supplied)
    # The stored residuals are AN times the returns minus the factor contribution, with no
    # intercept subtracted; the first row of each cadence has no earlier factor price.
    for freq, returns in history['asset_returns_dict'].items():
        prices = factor_prices.loc[:as_of].reindex(returns.index)
        contribution = np.log(prices).diff() @ data.y_betas.loc[returns.columns].T
        scaled = PERIODS_PER_YEAR[freq] * (returns - contribution)
        stored = data.residuals.loc[returns.index, returns.columns]
        np.testing.assert_allclose(stored.iloc[1:], scaled.iloc[1:], rtol=0, atol=1e-14)
        assert stored.iloc[0].isna().all()
    # The fit leaves the final bucket's state on the supplied LassoModel.
    assert estimator.lasso_model.estimated_betas.index.tolist() == QUARTERLY
    # An asset without history, added through assets, gets zero loadings and zero residual
    # variance, so it looks riskless.
    padded = replace(estimator, lasso_model=hcgl_model()).fit_current_covar(
        **history, assets=ASSETS + ['Hedge funds'])
    assert (padded.loc['Hedge funds'] == 0.0).all() and (padded['Hedge funds'] == 0.0).all()
    # In-sample residual variances of the private assets are below those of the generating
    # model, 12 times the sum of the idiosyncratic and common monthly residual variances.
    generating = 12 * (np.array(RESIDUAL_VOLS[5:]) ** 2 + PRIVATE_SHOCK_VOL ** 2)
    assert (residual_vars(data).loc[QUARTERLY].to_numpy() < generating).all()
    # Defaults: weekly factor returns at span 52, quarter ends, orthogonal residuals, no
    # references and no cadence penalties.
    default = op.FactorCovarEstimator()
    assert (default.factor_returns_freq, default.factor_covar_span) == ('W-WED', 52)
    assert (default.rebalancing_freq, default.residual_type) == ('QE', 'orthogonal')
    assert default.lasso_model is None and default.demean is True
    assert default.is_apply_vol_normalised_returns is False
    assert default.include_factors_in_clustering is False
    assert default.factor_clustering_freqs is None and default.reg_lambda_freq_dict is None
    assert default.residual_covar_freq is None and default.residual_covar_span is None

    # Empirical residual correlation needs FactorLasso 0.19.0 or newer; older releases serve
    # only the orthogonal default.
    supported = 'residual_correlation' in {f.name for f in fields(CurrentFactorCovarData)}
    assert supported == (installed_version('factorlasso') >= (0, 19, 0))
    if supported:
        empirical = op.FactorCovarEstimator(
            lasso_model=hcgl_model(), factor_returns_freq='ME', factor_covar_span=36,
            residual_type='empirical', residual_corr_weight=0.5)
        empirical_data = empirical.fit_current_factor_covars(**history)
        blended = empirical.fit_current_covar(**history)

        check_empirical_residuals(factor_prices, buckets, data, empirical, empirical_data,
                                  blended)

    penalised = op.FactorCovarEstimator(
        lasso_model=hcgl_model(), factor_returns_freq='ME', factor_covar_span=36,
        reg_lambda_freq_dict={'ME': 1e-5, 'QE': 1e-4})
    penalised_data = penalised.fit_current_factor_covars(**history)

    # The monthly bucket keeps its fit; the quarterly one equals a separate quarterly fit with
    # the model's penalty set to 1e-4, which shrinks Real estate's rates loading from 0.46 to
    # 0.16. The model's own penalty is restored after the fit.
    np.testing.assert_allclose(penalised_data.y_betas.loc[MONTHLY], data.y_betas.loc[MONTHLY],
                               rtol=0, atol=1e-10)
    separate = op.estimate_lasso_factor_covar_data(
        risk_factor_prices=history['risk_factor_prices'],
        asset_returns_dict={'QE': history['asset_returns_dict']['QE']},
        lasso_model=hcgl_model(reg_lambda=1e-4), factor_returns_freq='ME',
        factor_covar_span=36)
    np.testing.assert_allclose(penalised_data.y_betas.loc[QUARTERLY], separate.y_betas,
                               rtol=0, atol=1e-10)
    shrink = (data.y_betas - penalised_data.y_betas).loc[QUARTERLY]
    assert round(shrink.abs().max().max(), 2) == 0.30
    assert round(data.y_betas.at['Real estate', 'Rates'], 2) == 0.46
    assert round(penalised_data.y_betas.at['Real estate', 'Rates'], 2) == 0.16
    assert penalised.lasso_model.reg_lambda == 1e-5
    # Every fitted cadence needs a penalty, and a penalty must be finite and non-negative.
    for bad, error in (({'ME': 1e-5}, KeyError), ({'ME': 1e-5, 'QE': -1e-4}, ValueError)):
        try:
            replace(penalised, lasso_model=hcgl_model(), reg_lambda_freq_dict=bad) \
                .fit_current_factor_covars(**history)
        except error:
            pass
        else:
            raise AssertionError(f'{bad} was accepted')

    referenced = op.FactorCovarEstimator(
        lasso_model=hcgl_model(), factor_returns_freq='ME', factor_covar_span=36,
        include_factors_in_clustering=True, factor_clustering_freqs=['QE'])
    referenced_data = referenced.fit_current_factor_covars(**history)

    # The references are not responses: loadings, clusters, residuals and the covariance
    # cover the eight assets only.
    assert referenced_data.y_betas.index.tolist() == ASSETS
    assert referenced_data.clusters.index.tolist() == ASSETS
    assert referenced_data.residuals.columns.tolist() == ASSETS
    np.testing.assert_allclose(
        referenced_data.get_y_covar(),
        systematic(referenced_data) + np.diag(residual_vars(referenced_data)),
        rtol=1e-10, atol=1e-15)
    # The monthly bucket, not listed in factor_clustering_freqs, is unchanged. With the four
    # factors as references the three private assets form one cluster instead of three, and
    # their loadings move by up to 0.09.
    np.testing.assert_allclose(referenced_data.y_betas.loc[MONTHLY], data.y_betas.loc[MONTHLY],
                               rtol=0, atol=1e-10)
    pd.testing.assert_series_equal(referenced_data.clusters.loc[MONTHLY],
                                   data.clusters.loc[MONTHLY])
    # The reported quarterly tree has the two merges of three assets, without the references.
    assert referenced_data.linkages.index.str.startswith('QE:').sum() == 2
    assert data.clusters.loc[QUARTERLY].nunique() == 3
    assert referenced_data.clusters.loc[QUARTERLY].nunique() == 1
    moved = (referenced_data.y_betas - data.y_betas).loc[QUARTERLY].abs().max().max()
    assert round(moved, 2) == 0.09
    # With references a current fit truncates its inputs at estimation_date.
    whole_referenced = replace(referenced, lasso_model=hcgl_model()).fit_current_covar(
        risk_factor_prices=factor_prices, asset_returns_dict=buckets, estimation_date=as_of)
    np.testing.assert_allclose(whole_referenced, referenced_data.get_y_covar(),
                               rtol=1e-10, atol=1e-15)
    # References need HCGL or FCGL, and a nonempty list of cadences.
    for bad in ({'lasso_model': LassoModel(model_type=LassoModelType.LASSO)},
                {'lasso_model': hcgl_model(), 'factor_clustering_freqs': []}):
        try:
            op.FactorCovarEstimator(include_factors_in_clustering=True, **bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f'{bad} was accepted')

    period = qis.TimePeriod('2023-12-31', '2024-12-31')
    rolling = estimator.fit_rolling_factor_covars(
        risk_factor_prices=factor_prices, asset_returns_dict=buckets, time_period=period)
    covars = rolling.get_y_covars()

    # Five calendar quarter ends; each rolling matrix equals a current fit on the inputs
    # sliced through its date, and the shared method returns the same matrices.
    assert list(covars) == list(pd.to_datetime(ROLLING_KEYS))
    for date, matrix in covars.items():
        current = replace(estimator, lasso_model=hcgl_model()).fit_current_covar(
            **history_through(factor_prices, buckets, date))
        np.testing.assert_allclose(matrix, current, rtol=1e-10, atol=1e-15)
    shared = replace(estimator, lasso_model=hcgl_model()).fit_rolling_covars(
        risk_factor_prices=factor_prices, asset_returns_dict=buckets, time_period=period)
    for date, matrix in covars.items():
        np.testing.assert_allclose(shared[date], matrix, rtol=1e-10, atol=1e-15)
    # An as-of query between fits is answered by the latest snapshot available then; checked
    # with the FactorLasso releases that serve empirical residuals.
    if supported:
        asof = rolling.get_y_covars(dates=pd.DatetimeIndex(['2024-02-15']))
        np.testing.assert_allclose(asof[pd.Timestamp('2024-02-15')],
                                   covars[pd.Timestamp(AS_OF)], rtol=0, atol=0)
    # Later inputs change no earlier estimate: after 30 June 2024 the Equity factor price
    # doubles and two assets gain 10%, which moves only the last two matrices.
    cutoff = pd.Timestamp('2024-06-30')
    later_prices = factor_prices.copy()
    later_prices.loc[later_prices.index > cutoff, 'Equity'] *= 2.0
    later_buckets = {freq: r.copy() for freq, r in buckets.items()}
    later_buckets['ME'].loc[later_buckets['ME'].index > cutoff, 'DM equity'] += 0.10
    later_buckets['QE'].loc[later_buckets['QE'].index > cutoff, 'Real estate'] += 0.10
    changed = replace(estimator, lasso_model=hcgl_model()).fit_rolling_factor_covars(
        risk_factor_prices=later_prices, asset_returns_dict=later_buckets,
        time_period=period).get_y_covars()
    for date, matrix in covars.items():
        if date <= cutoff:
            np.testing.assert_allclose(changed[date], matrix, rtol=1e-10, atol=1e-15)
        else:
            assert np.abs(changed[date] - matrix).max().max() > 1e-3
    # Pitfall: estimation_date alone does not truncate an ordinary current fit. The whole
    # history labelled 31 December 2023 gives Commodities a volatility of 23.9%, against
    # 20.4% from the inputs sliced through that date.
    whole = replace(estimator, lasso_model=hcgl_model()).fit_current_covar(
        risk_factor_prices=factor_prices, asset_returns_dict=buckets, estimation_date=as_of)
    assert round(np.sqrt(covar.at['Commodities', 'Commodities']), 3) == 0.204
    assert round(np.sqrt(whole.at['Commodities', 'Commodities']), 3) == 0.239
    # The rolling wrapper refuses a date at which a bucket has fewer rows than warmup_period.
    try:
        replace(estimator, lasso_model=hcgl_model()).fit_rolling_factor_covars(
            risk_factor_prices=factor_prices, asset_returns_dict=buckets,
            time_period=qis.TimePeriod('2005-03-31', '2006-06-30'))
    except ValueError:
        pass
    else:
        raise AssertionError('a fit before the warm-up was accepted')

    check_reporting(factor_prices, buckets, data, rolling)
    print('factor_covariance_hcgl: all page statements verified.')


def exhibit(path) -> dict:
    """Draw the page's figure: the variance split by asset and a portfolio's residual variance.

    Args:
        path: PNG file to write.

    Returns:
        The plotted table and the checks the figure illustrates.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    factor_prices, buckets = simulated_panel(SEED)
    history = history_through(factor_prices, buckets, AS_OF)
    data = op.FactorCovarEstimator(
        lasso_model=hcgl_model(), factor_returns_freq='ME', factor_covar_span=36,
        residual_type='empirical').fit_current_factor_covars(**history)
    systematic_vars = np.diag(systematic(data))
    residual = residual_vars(data).loc[ASSETS].to_numpy()
    totals = {kind: np.diag(data.get_y_covar(residual_type=kind)) for kind in
              ('orthogonal', 'empirical')}
    w = private_weights()
    diagonal_part = w @ np.diag(residual) @ w
    retained = {}
    for rho in (0.0, 0.5, 1.0):
        block = residual_block(data, rho)
        retained[rho] = w @ (block - np.diag(np.diag(block))) @ w
    rows = [('asset', asset, 'systematic', value) for asset, value in zip(ASSETS, systematic_vars)]
    rows += [('asset', asset, 'residual', value) for asset, value in zip(ASSETS, residual)]
    rows += [('private portfolio', f'rho = {rho:g}', 'residual variances', diagonal_part)
             for rho in retained]
    rows += [('private portfolio', f'rho = {rho:g}', 'residual covariances', value)
             for rho, value in retained.items()]
    table = pd.DataFrame(rows, columns=['panel', 'bar', 'component', 'variance'])

    ink, muted, grid, surface = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
    blue, orange, aqua = '#2a78d6', '#eb6834', '#1baf7a'
    plt.rcParams.update({'font.size': 11, 'axes.edgecolor': grid, 'axes.labelcolor': muted,
                         'xtick.color': muted, 'ytick.color': muted})
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.0, 4.4), facecolor=surface,
                                      gridspec_kw={'width_ratios': [1.6, 1.0]})
    x = np.arange(len(ASSETS))
    left.bar(x, systematic_vars, 0.62, color=blue,
             label=r'Systematic, diagonal of $\beta \Sigma_F \beta^{T}$')
    left.bar(x, residual, 0.62, bottom=systematic_vars, color=orange,
             label='Residual, diagonal of $D$')
    left.set_xticks(x, ASSETS, rotation=35, ha='right', fontsize=10)
    left.set_title('Variance of each asset,\nidentical under both residual types', loc='left',
                   color=ink)
    left.legend(frameon=False, loc='upper left', fontsize=10, labelcolor=ink)
    left.set_ylim(0.0, 1.3 * (systematic_vars + residual).max())
    bars = np.arange(len(retained))
    right.bar(bars, [diagonal_part] * len(retained), 0.58, color=orange,
              label='Residual variances')
    right.bar(bars, list(retained.values()), 0.58, bottom=diagonal_part, color=aqua,
              label='Residual covariances')
    right.set_xticks(bars, ['Orthogonal\nor ρ = 0', 'Empirical\nρ = 0.5', 'Empirical\nρ = 1'],
                     fontsize=10)
    right.set_title('Residual variance of equal\nweights in the private assets', loc='left',
                    color=ink)
    right.legend(frameon=False, loc='upper left', fontsize=10, labelcolor=ink)
    right.set_ylim(0.0, 1.35 * (diagonal_part + retained[1.0]))
    for axis in (left, right):
        axis.set_ylabel('Annual variance')
        axis.set_facecolor(surface)
        axis.grid(axis='y', color=grid, linewidth=0.8)
        axis.set_axisbelow(True)
        for side in ('top', 'right'):
            axis.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=surface)
    plt.close(fig)
    checks = {
        'systematic_plus_residual_is_total_orthogonal': bool(np.allclose(
            systematic_vars + residual, totals['orthogonal'], rtol=1e-12, atol=0)),
        'systematic_plus_residual_is_total_empirical': bool(np.allclose(
            systematic_vars + residual, totals['empirical'], rtol=1e-12, atol=0)),
        'orthogonal_has_no_residual_covariance': bool(retained[0.0] == 0.0),
        'retention_is_linear_in_rho': bool(np.isclose(retained[0.5], 0.5 * retained[1.0],
                                                      rtol=1e-12, atol=0)),
        'empirical_more_than_doubles_private_residual_variance': bool(
            diagonal_part + retained[1.0] > 2 * diagonal_part),
    }
    return {'table': table, 'checks': checks}


if __name__ == '__main__':
    main()
