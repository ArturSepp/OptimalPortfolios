"""Canonical script of docs/universe_data_and_unsmoothing.md.

The page shows excerpts of this file; every number it quotes is asserted here against a
reference computed a different way. The script runs offline after ``pip install
optimalportfolios`` and needs no data file:

    python -m examples.docs.universe_data_and_unsmoothing

``exhibit`` draws the page's figure; ``tools/docs_analytics/teaching.py`` calls it with the
constants below and records their values.
"""
import dataclasses
import inspect
import tempfile
import warnings
from enum import Enum
from unittest.mock import patch

import numpy as np
import pandas as pd
import qis

import optimalportfolios as op

TICKERS = ['Govt', 'Credit', 'Equity', 'PE']
NAMES = ['Government bonds', 'Credit', 'Listed equity', 'Private equity']
ASSET_CLASSES = ['Bonds', 'Bonds', 'Equities', 'Equities']
LIQUIDITY = ['Liquid', 'Liquid', 'Liquid', 'Illiquid']
# Annual means, volatilities and correlations of the true quarterly log returns.
MEANS = [0.03, 0.04, 0.07, 0.08]
VOLS = [0.05, 0.08, 0.16, 0.20]
CORR = [[1.0, 0.5, -0.2, -0.1],
        [0.5, 1.0, 0.4, 0.4],
        [-0.2, 0.4, 1.0, 0.7],
        [-0.1, 0.4, 0.7, 1.0]]
PHI = 0.6  # appraisal smoothing: weight of the previous reported return
START = '1989-12-31'
END = '2024-12-31'
SEED = 19
SPAN = 40  # EWMA span in quarters, the unsmoothing default
ILLIQUID_CAP = 0.15
UNIVERSE_FIELDS = ['prices', 'metadata', 'metadata_fields', 'group_loadings_level1',
                   'group_loadings_level2', 'liquidity_ac_id', 'equity_ac_id', 'bond_ac_id',
                   'pe_asset_id', 'validate_on_init']
# Defaults the page quotes for the unsmoothing copy (besides mean_adj_type=EWMA), the return
# construction and the qis price unsmoother.
COPY_DEFAULTS = {'freq': 'QE', 'unsmooth_span': 40, 'warmup_period': 8,
                 'max_value_for_beta': 0.75, 'is_log_returns': True}
RETURNS_DEFAULTS = {'returns_freq': 'ME', 'demean': True, 'drop_first': True,
                    'is_first_zero': False, 'is_log_returns': True, 'span': 52}
QIS_DEFAULTS = {'ar_order': 2, 'min_value_for_beta': -0.25}
RETURNS_IN_ESTIMATOR = ('optimalportfolios.covar_estimation.ewma_covar_estimator.'
                        'compute_returns_from_prices')


def simulated_panel(seed: int) -> tuple:
    """Simulate quarterly prices whose private-equity column is appraisal-smoothed with PHI."""
    dates = pd.date_range(START, END, freq='QE')
    vols = np.array(VOLS) / 2.0  # quarterly volatilities
    rng = np.random.default_rng(seed)
    # The draws of Generator.multivariate_normal, whose SVD factor has signs that differ between
    # LAPACK builds, with each singular vector's largest entry made positive so that the same
    # seed gives the same returns on every platform.
    _, singular_values, vh = np.linalg.svd(np.outer(vols, vols) * np.array(CORR))
    vh = vh * np.sign(vh[np.arange(len(vh)), np.abs(vh).argmax(axis=1)])[:, None]
    shocks = rng.standard_normal((len(dates) - 1, len(TICKERS)))
    true = np.array(MEANS) / 4.0 + shocks @ (np.sqrt(singular_values)[:, None] * vh)
    reported = true.copy()
    pe = TICKERS.index('PE')
    for t in range(1, len(reported)):
        reported[t, pe] = PHI * reported[t - 1, pe] + (1.0 - PHI) * true[t, pe]
    log_prices = np.vstack([np.zeros(len(TICKERS)), np.cumsum(reported, axis=0)])
    prices = pd.DataFrame(100.0 * np.exp(log_prices), index=dates, columns=TICKERS)
    return prices, pd.DataFrame(true, index=dates[1:], columns=TICKERS)


def example_metadata() -> pd.DataFrame:
    """Return the metadata table of the four-asset universe."""
    return pd.DataFrame({'name': NAMES, 'asset_class': ASSET_CLASSES, 'currency': 'USD'},
                        index=TICKERS)


def example_universes(seed: int) -> tuple:
    """Return the reported universe, its unsmoothed copy and the true returns."""
    prices, true_returns = simulated_panel(seed=seed)
    universe = op.UniverseData(prices=prices, metadata=example_metadata(), pe_asset_id='PE')
    flag = pd.Series(universe.metadata.index == universe.pe_asset_id,
                     index=universe.metadata.index)
    unsmoothed = op.copy_universe_data_with_unsmoothed_prices(universe_data=universe,
                                                              assets_for_unsmoothing=flag)
    return universe, unsmoothed, true_returns


def round_trip(universe: op.UniverseData) -> op.UniverseData:
    """Save a universe with both loading levels to a temporary folder and load it back."""
    with tempfile.TemporaryDirectory() as folder, warnings.catch_warnings():
        # qis parses every CSV index as dates; the metadata and loadings indexes are not.
        warnings.simplefilter('ignore', UserWarning)
        universe.save(file_name='universe', local_path=folder)
        return op.UniverseData.load(
            file_name='universe', local_path=folder,
            group_loadings_keys=['group_loadings_level1', 'group_loadings_level2'])


def defaults(function) -> dict:
    """Return the default values of a function's keyword parameters."""
    return {name: parameter.default
            for name, parameter in inspect.signature(function).parameters.items()
            if parameter.default is not inspect.Parameter.empty}


def construction_error(**arguments) -> str:
    """Construct a UniverseData and return the ValueError message, or '' if none is raised."""
    try:
        op.UniverseData(**arguments)
    except ValueError as error:
        return str(error)
    return ''


def lag1_autocorrelation(x: np.ndarray) -> float:
    """Return the Pearson correlation of consecutive observations."""
    return float(np.corrcoef(x[1:], x[:-1])[0, 1])


def annualised_vol(quarterly: np.ndarray) -> float:
    """Return the sample standard deviation of quarterly returns times two."""
    return float(np.std(quarterly, ddof=1) * 2.0)


def smoothing_moments(phi: float, terms: int = 2000) -> tuple:
    """Return the variance ratio and lag-1 autocorrelation of the smoothing filter by summation."""
    weights = (1.0 - phi) * phi ** np.arange(terms)  # impulse response of x on r
    variance_ratio = float(np.sum(weights ** 2))
    autocorrelation = float(np.sum(weights[1:] * weights[:-1]) / variance_ratio)
    return variance_ratio, autocorrelation


def demeaned_reference(x: np.ndarray, span: int) -> np.ndarray:
    """Return lambda times the deviation from the previous EWMA mean, by explicit recursion."""
    lam = 1.0 - 2.0 / (span + 1.0)
    mean = x[0].copy()
    out = []
    for t in range(1, len(x)):
        out.append(lam * (x[t] - mean))
        mean = lam * mean + (1.0 - lam) * x[t]
    return np.array(out)


def risk_shares(weights: pd.Series, covar: pd.DataFrame) -> np.ndarray:
    """Return each asset's share of portfolio variance, w_i (Sigma w)_i / w'Sigma w."""
    w = weights.to_numpy()
    sigma = covar.to_numpy()
    return w * (sigma @ w) / (w @ sigma @ w)


def main() -> None:
    """Run the worked example of the page and assert every quoted number."""
    # A four-asset quarterly universe whose private-equity column is appraisal-smoothed.
    prices, true_returns = simulated_panel(seed=SEED)
    metadata = pd.DataFrame({'name': NAMES, 'asset_class': ASSET_CLASSES, 'currency': 'USD'},
                            index=TICKERS)
    level1 = qis.set_group_loadings(group_data=metadata['asset_class'])
    level2 = qis.set_group_loadings(group_data=pd.Series(LIQUIDITY, index=TICKERS))
    universe = op.UniverseData(prices=prices, metadata=metadata, group_loadings_level1=level1,
                               group_loadings_level2=level2, pe_asset_id='PE')
    assert [f.name for f in dataclasses.fields(op.UniverseData)] == UNIVERSE_FIELDS
    try:
        universe.pe_asset_id = 'Equity'
        raise AssertionError('UniverseData must be frozen')
    except dataclasses.FrozenInstanceError:
        pass
    assert universe.n_assets == 4 and universe.assets == TICKERS and len(universe.prices) == 141
    assert universe.date_range == (pd.Timestamp(START), pd.Timestamp(END))
    assert list(level1.columns) == ['Bonds', 'Equities']
    assert list(level2.columns) == ['Liquid', 'Illiquid']
    assert (level1.sum(axis=1) == 1.0).all() and (level2.sum(axis=1) == 1.0).all()
    assert level2.loc['PE', 'Illiquid'] == 1.0 and level2['Illiquid'].sum() == 1.0
    assert (universe.metadata_fields is op.MetadataField and universe.validate_on_init
            and (universe.liquidity_ac_id, universe.equity_ac_id, universe.bond_ac_id)
            == ('Liquidity', 'Equities', 'Bonds'))
    assert [field.value for field in op.MetadataField] == ['name', 'asset_class', 'currency']
    assert op.MetadataField.ASSET_CLASS == 'asset_class'  # a str enum compares to its value
    assert universe.asset_class.tolist() == ASSET_CLASSES and universe.name.tolist() == NAMES
    assert universe.currency.eq('USD').all()
    assert universe.get_hedge_ratio(hedged_acs=['Bonds']).tolist() == [1.0, 1.0, 0.0, 0.0]
    # The reported PE log returns follow x_t = phi x_{t-1} + (1 - phi) r_t exactly, and the
    # other columns are the true returns.
    reported = np.diff(np.log(prices.to_numpy()), axis=0)
    true = true_returns.to_numpy()
    assert np.allclose(reported[1:, 3] - PHI * reported[:-1, 3], (1.0 - PHI) * true[1:, 3],
                       atol=1e-12)
    assert np.allclose(reported[:, :3], true[:, :3], atol=1e-12)

    # Proposition 1: variance ratio (1 - phi) / (1 + phi) and autocorrelation phi, checked
    # against the impulse-response sums; with phi = 0.6 reported volatility is half the true one.
    variance_ratio, autocorrelation = smoothing_moments(PHI)
    assert abs(np.sum((1.0 - PHI) * PHI ** np.arange(2000)) - 1.0) < 1e-12  # mean unchanged
    assert abs(variance_ratio - (1.0 - PHI) / (1.0 + PHI)) < 1e-12
    assert abs(autocorrelation - PHI) < 1e-12
    assert abs(np.sqrt(variance_ratio) - 0.5) < 1e-12
    # Proposition 2: the reverse filter with the known phi recovers the true returns.
    recovered = (reported[1:, 3] - PHI * reported[:-1, 3]) / (1.0 - PHI)
    assert np.allclose(recovered, true[1:, 3], atol=1e-12)

    # Construction validates alignment, required fields, duplicates, nulls and loadings.
    invalid = {
        'metadata row missing': dict(prices=prices, metadata=metadata.drop(index='PE')),
        'required column missing': dict(prices=prices, metadata=metadata.drop(columns='name')),
        'duplicate price column': dict(prices=prices[TICKERS + ['PE']], metadata=metadata),
        'null in required column': dict(
            prices=prices, metadata=metadata.assign(currency=['USD', 'USD', 'USD', None])),
        'loadings on other assets': dict(prices=prices, metadata=metadata,
                                         group_loadings_level2=level2.drop(index='PE')),
    }
    errors = {case: construction_error(**arguments) for case, arguments in invalid.items()}
    assert errors['metadata row missing'].startswith('Asset mismatch')
    assert errors['required column missing'] == "Metadata missing required columns: {'name'}"
    assert errors['duplicate price column'] == "Duplicate asset names in prices: ['PE']"
    assert errors['null in required column'] == \
        "Null values in required metadata columns: ['currency']"
    assert errors['loadings on other assets'] == \
        "group_loadings_level2 index doesn't match price columns"
    # A duplicated metadata row is caught by the same check.
    assert construction_error(prices=prices, metadata=metadata.loc[TICKERS + ['PE']]) == \
        "Duplicate asset names in metadata: ['PE']"
    # validate_on_init=False defers the checks to an explicit validate().
    deferred = op.UniverseData(**invalid['metadata row missing'], validate_on_init=False)
    try:
        deferred.validate()
        raise AssertionError('validate() must raise on a misaligned universe')
    except ValueError as error:
        assert str(error).startswith('Asset mismatch')
    # What construction does not check: row order, price values, loading values, identifiers.
    reordered = op.UniverseData(prices=prices, metadata=metadata.iloc[::-1])
    assert reordered.metadata.index.tolist() == TICKERS[::-1]
    gaps = prices.copy()
    gaps.iloc[:4, 3] = np.nan
    gaps.iloc[5, 0] = -1.0
    assert construction_error(prices=gaps, metadata=metadata) == ''
    assert construction_error(prices=prices, metadata=metadata,
                              group_loadings_level1=0.5 * level1) == ''
    assert construction_error(prices=prices, metadata=metadata, pe_asset_id='not an asset',
                              equity_ac_id='not a class') == ''
    # Further metadata columns are allowed and unchecked, even with missing values.
    assert construction_error(prices=prices,
                              metadata=metadata.assign(region=[None, 'US', 'US', 'US'])) == ''

    # A custom MetadataField enum changes the required columns; the properties still read
    # the default column names.
    RiskField = Enum('RiskField', {'ASSET_CLASS': 'asset_class', 'REGION': 'region'})
    regional = metadata.drop(columns='name').assign(region=['US', 'US', 'US', 'Global'])
    custom = op.UniverseData(prices=prices, metadata=regional, metadata_fields=RiskField)
    assert custom.asset_class.tolist() == ASSET_CLASSES
    try:
        custom.name
        raise AssertionError('the name property reads the default name column')
    except KeyError:
        pass
    assert construction_error(prices=prices, metadata=metadata,
                              metadata_fields=RiskField) == \
        "Metadata missing required columns: {'region'}"

    # Unsmooth the private-equity column named by pe_asset_id.
    flag = pd.Series(universe.metadata.index == universe.pe_asset_id,
                     index=universe.metadata.index)
    unsmoothed = op.copy_universe_data_with_unsmoothed_prices(universe_data=universe,
                                                              assets_for_unsmoothing=flag)
    others = ['Govt', 'Credit', 'Equity']
    assert flag.tolist() == [False, False, False, True]
    assert unsmoothed.prices.index.equals(universe.prices.index)
    assert unsmoothed.prices[others].equals(universe.prices[others])
    assert np.array_equal(unsmoothed.prices[others].to_numpy(), prices[others].to_numpy())
    assert unsmoothed.metadata is universe.metadata
    assert unsmoothed.group_loadings_level1 is level1
    assert unsmoothed.group_loadings_level2 is level2
    assert unsmoothed.metadata_fields is op.MetadataField and unsmoothed.pe_asset_id == 'PE'
    # The replaced column is the qis NAV: missing for 17 quarters, 1.0 at 1994-03-31, then
    # compounded from the first unsmoothed return at 1994-06-30.
    pe = unsmoothed.prices['PE']
    assert pe.iloc[:17].isna().all() and pe.first_valid_index() == pd.Timestamp('1994-03-31')
    assert pe.loc['1994-03-31'] == 1.0 and pe.iloc[17:].notna().all()
    # The Pitfall: the wrapper does not pass ar_order, so qis applies its default AR(2).
    ar2, _, _, _ = qis.compute_ar_unsmoothed_prices(
        prices=prices[['PE']], ar_order=2, freq='QE', span=40,
        mean_adj_type=qis.MeanAdjType.EWMA, warmup_period=8, max_value_for_beta=0.75,
        is_log_returns=True)
    ar1, _, _, _ = qis.compute_ar_unsmoothed_prices(
        prices=prices[['PE']], ar_order=1, freq='QE', span=40,
        mean_adj_type=qis.MeanAdjType.EWMA, warmup_period=8, max_value_for_beta=0.75,
        is_log_returns=True)
    assert pe.equals(ar2['PE'].reindex(prices.index).ffill())
    assert not pe.equals(ar1['PE'].reindex(prices.index).ffill())
    common = pe.index[pe.notna() & ar1['PE'].reindex(prices.index).notna()]
    assert np.abs(np.log(pe[common] / ar1['PE'][common])).max() > 0.01
    # The wrapper's defaults, the qis defaults it leaves in place, and pe_asset_id unread.
    copy_defaults = defaults(op.copy_universe_data_with_unsmoothed_prices)
    assert {name: copy_defaults[name] for name in COPY_DEFAULTS} == COPY_DEFAULTS
    assert copy_defaults['mean_adj_type'] == qis.MeanAdjType.EWMA
    qis_defaults = defaults(qis.compute_ar_unsmoothed_prices)
    assert {name: qis_defaults[name] for name in QIS_DEFAULTS} == QIS_DEFAULTS
    unnamed = op.UniverseData(prices=prices, metadata=metadata)
    assert unnamed.pe_asset_id is None
    assert op.copy_universe_data_with_unsmoothed_prices(unnamed, flag).prices.equals(
        unsmoothed.prices)
    # With no flagged asset, the same object comes back.
    none = pd.Series(False, index=universe.metadata.index)
    assert op.copy_universe_data_with_unsmoothed_prices(universe, none) is universe
    # The flag must match the metadata index in order, not only as a set.
    try:
        op.copy_universe_data_with_unsmoothed_prices(reordered, flag)
        raise AssertionError('a flag in another order must raise')
    except ValueError as error:
        assert str(error).startswith('assets_for_unsmoothing index does not match universe')
    try:
        op.copy_universe_data_with_unsmoothed_prices(
            universe, flag, freq=pd.Series('QE', index=TICKERS[::-1]))
        raise AssertionError('a freq Series in another order must raise')
    except ValueError as error:
        assert str(error) == 'freq Series index does not match universe assets'
    # The copy keeps equity_ac_id, bond_ac_id, pe_asset_id and validate_on_init, but not
    # liquidity_ac_id.
    labelled = op.UniverseData(prices=prices, metadata=metadata, liquidity_ac_id='Cash',
                               equity_ac_id='Equity', bond_ac_id='Fixed income',
                               pe_asset_id='PE', validate_on_init=False)
    copied = op.copy_universe_data_with_unsmoothed_prices(labelled, flag)
    assert (copied.equity_ac_id, copied.bond_ac_id, copied.pe_asset_id) == \
        ('Equity', 'Fixed income', 'PE')
    assert copied.validate_on_init is False and unsmoothed.validate_on_init is True
    assert copied.liquidity_ac_id == 'Liquidity'
    # from_selection and rename_index return the default identifiers.
    selected = op.UniverseData.from_selection(prices=prices, metadata=metadata, assets=TICKERS)
    renamed = labelled.rename_index()
    for other in (selected, renamed):
        assert (other.liquidity_ac_id, other.equity_ac_id, other.bond_ac_id,
                other.pe_asset_id) == ('Liquidity', 'Equities', 'Bonds', None)
    assert renamed.assets == NAMES
    # On a monthly grid the quarterly unsmoothed column is carried between quarter ends: the
    # log-interpolated monthly panel moves every month, the unsmoothed column only at quarter ends.
    monthly = np.exp(np.log(prices.resample('ME').last()).interpolate())
    monthly_universe = op.UniverseData(prices=monthly, metadata=metadata)
    monthly_copy = op.copy_universe_data_with_unsmoothed_prices(monthly_universe, flag)
    assert monthly['PE'].diff().iloc[1:].ne(0.0).all()
    moves = monthly_copy.prices['PE'].dropna().diff().iloc[1:].ne(0.0)
    assert moves[moves.index.is_quarter_end].all() and not moves[~moves.index.is_quarter_end].any()
    assert (~moves.index.is_quarter_end).sum() == 2 * moves.index.is_quarter_end.sum()
    # from_selection keeps the order of the selected assets.
    pair = op.UniverseData.from_selection(prices=prices, metadata=metadata, assets=['PE', 'Govt'],
                                          group_loadings_level2=level2)
    assert pair.assets == ['PE', 'Govt'] and pair.metadata.index.tolist() == ['PE', 'Govt']
    assert pair.group_loadings_level2.index.tolist() == ['PE', 'Govt']
    # save and load round-trip; load without metadata_fields requires every metadata column.
    loaded = round_trip(op.UniverseData(
        prices=prices, metadata=metadata.assign(region=['US', 'US', 'US', 'Global']),
        group_loadings_level1=level1, group_loadings_level2=level2))
    assert np.allclose(loaded.prices.to_numpy(), prices.to_numpy(), rtol=1e-12)
    assert loaded.group_loadings_level2.equals(level2)
    assert [field.value for field in loaded.metadata_fields] == ['name', 'asset_class',
                                                                  'currency', 'region']

    # Lag-1 autocorrelation and volatility of the private-equity returns, before and after.
    before = op.compute_returns_from_prices(universe.prices, returns_freq='QE', demean=False)
    after = op.compute_returns_from_prices(unsmoothed.prices, returns_freq='QE', demean=False)
    window = after['PE'].dropna().index
    stats = pd.DataFrame({
        'lag-1 autocorrelation': [before.loc[window, 'PE'].autocorr(lag=1),
                                  after.loc[window, 'PE'].autocorr(lag=1)],
        'annualised volatility': [before.loc[window, 'PE'].std() * 2.0,
                                  after.loc[window, 'PE'].std() * 2.0]},
        index=['smoothed', 'unsmoothed'])
    assert len(window) == 123 and window[0] == pd.Timestamp('1994-06-30')
    assert before[others].equals(after[others])
    # Reference: numpy log differences of the price arrays, from the same first price.
    rows = slice(prices.index.get_loc(window[0]) - 1, None)
    x_before = np.diff(np.log(prices['PE'].to_numpy()[rows]))
    x_after = np.diff(np.log(pe.to_numpy()[rows]))
    x_true = true_returns.loc[window, 'PE'].to_numpy()
    assert np.allclose(before.loc[window, 'PE'], x_before, atol=1e-12)
    assert np.allclose(after.loc[window, 'PE'], x_after, atol=1e-12)
    reference = pd.DataFrame({
        'lag-1 autocorrelation': [lag1_autocorrelation(x_before), lag1_autocorrelation(x_after)],
        'annualised volatility': [annualised_vol(x_before), annualised_vol(x_after)]},
        index=['smoothed', 'unsmoothed'])
    assert np.allclose(stats.to_numpy(), reference.to_numpy(), atol=1e-12)
    assert stats.round(2)['lag-1 autocorrelation'].tolist() == [0.61, -0.04]
    assert stats.round(3)['annualised volatility'].tolist() == [0.102, 0.205]
    assert round(annualised_vol(x_true), 3) == 0.204
    assert round(lag1_autocorrelation(x_true), 2) == -0.02
    # The Insight: reported volatility is half the true one in the sample, as for phi = 0.6,
    # so the reported variance is about a quarter of the true one.
    assert round(annualised_vol(x_before) / annualised_vol(x_true), 2) == 0.50
    assert round((annualised_vol(x_true) / annualised_vol(x_before)) ** 2) == 4
    # The coefficient sum that qis estimates stays inside its clipping bounds.
    _, _, theta_sum, _ = qis.compute_ar_unsmoothed_prices(prices=prices[['PE']])
    assert round(theta_sum['PE'].min(), 2) == 0.45 and round(theta_sum['PE'].max(), 2) == 0.67
    # Another seed over-corrects: unsmoothed volatility well above the true one.
    _, other_unsmoothed, other_true = example_universes(seed=3)
    other_after = op.compute_returns_from_prices(other_unsmoothed.prices, returns_freq='QE',
                                                 demean=False)['PE'].dropna()
    assert round(annualised_vol(other_after.to_numpy()), 3) == 0.307
    assert round(annualised_vol(other_true.loc[other_after.index, 'PE'].to_numpy()), 3) == 0.195

    # compute_returns_from_prices with its EWMA demeaning: lambda times the deviation from
    # the previous mean, the first zero row dropped.
    demeaned = op.compute_returns_from_prices(universe.prices, returns_freq='QE', span=SPAN)
    log_returns = np.diff(np.log(prices.to_numpy()), axis=0)
    assert len(demeaned) == len(prices) - 2 == 139
    assert np.allclose(demeaned.to_numpy(), demeaned_reference(log_returns, SPAN), atol=1e-12)
    assert abs((1.0 - 2.0 / (SPAN + 1.0)) - 39.0 / 41.0) < 1e-15
    assert round(39.0 / 41.0, 3) == 0.951
    assert defaults(op.compute_returns_from_prices) == RETURNS_DEFAULTS
    # Both EWMA covariance entry points build their returns with it.
    with patch(RETURNS_IN_ESTIMATOR, wraps=op.compute_returns_from_prices) as spy:
        op.estimate_current_ewma_covar(prices=prices, returns_freq='QE', span=SPAN)
        assert spy.call_count == 1
        op.EwmaCovarEstimator(returns_freq='QE', span=SPAN, rebalancing_freq='QE') \
            .fit_rolling_covars(prices=prices, time_period=qis.TimePeriod(START, END))
        assert spy.call_count == 2

    # Group budgets from the level-1 loadings, a cap from the level-2 loadings, and one
    # risk-budgeting allocation on each universe.
    budgets = op.compute_group_risk_budgets(groups=universe.group_loadings_level1.idxmax(axis=1))
    illiquid_cap = op.GroupLowerUpperConstraints(
        group_loadings=universe.group_loadings_level2, group_min_allocation=None,
        group_max_allocation=pd.Series({'Liquid': 1.0, 'Illiquid': ILLIQUID_CAP}))
    constraints = op.Constraints(is_long_only=True, group_lower_upper_constraints=illiquid_cap)
    start = unsmoothed.prices['PE'].first_valid_index()
    covars, weights = {}, {}
    for label, data in {'smoothed': universe, 'unsmoothed': unsmoothed}.items():
        covars[label] = op.estimate_current_ewma_covar(prices=data.prices.loc[start:],
                                                       returns_freq='QE', span=SPAN)
        weights[label] = op.wrapper_risk_budgeting(pd_covar=covars[label],
                                                   constraints=constraints, risk_budget=budgets)
    # Two groups of two assets with equal group budgets give 1/4 to each asset.
    assert np.allclose(budgets.to_numpy(), 1.0 / (2 * 2), atol=1e-15)
    assert illiquid_cap.group_loadings.equals(level2)
    # The other group constraints take the same group_loadings field.
    for group_constraint in (op.GroupTrackingErrorConstraint, op.GroupTurnoverConstraint):
        assert 'group_loadings' in {f.name for f in dataclasses.fields(group_constraint)}
    # The covariance is the quarterly estimate times 4.
    quarterly = op.estimate_current_ewma_covar(prices=universe.prices.loc[start:],
                                               returns_freq='QE', span=SPAN,
                                               apply_an_factor=False)
    assert np.allclose(covars['smoothed'].to_numpy(), 4.0 * quarterly.to_numpy(), rtol=1e-12)
    # Without the cap, smoothed data give private equity 22% of capital for a quarter of the
    # risk; with the cap it binds at 15%.
    uncapped = op.wrapper_risk_budgeting(pd_covar=covars['smoothed'],
                                         constraints=op.Constraints(is_long_only=True),
                                         risk_budget=budgets)
    assert np.allclose(risk_shares(uncapped, covars['smoothed']), 0.25, atol=1e-4)
    assert round(uncapped['PE'], 2) == 0.22
    assert abs(weights['smoothed']['PE'] - ILLIQUID_CAP) < 1e-6
    assert abs(weights['smoothed'].sum() - 1.0) < 1e-6
    # Unsmoothed data give private equity 10%: the cap is slack and every budget is met.
    assert round(weights['unsmoothed']['PE'], 2) == 0.10
    assert weights['unsmoothed']['PE'] < ILLIQUID_CAP - 0.04
    assert np.allclose(risk_shares(weights['unsmoothed'], covars['unsmoothed']), 0.25,
                       atol=1e-4)
    assert np.sqrt(covars['unsmoothed'].loc['PE', 'PE']) > 2.0 * np.sqrt(
        covars['smoothed'].loc['PE', 'PE'])
    print('universe_data_and_unsmoothing: all page statements verified.')


def exhibit(path) -> dict:
    """Draw the page's figure: reported and unsmoothed paths, and their autocorrelation and vol.

    Args:
        path: PNG file to write.

    Returns:
        The plotted table and the checks the figure illustrates.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    universe, unsmoothed, true_returns = example_universes(SEED)
    before = op.compute_returns_from_prices(universe.prices, returns_freq='QE', demean=False)
    after = op.compute_returns_from_prices(unsmoothed.prices, returns_freq='QE', demean=False)
    window = after['PE'].dropna().index
    series = {'reported (smoothed)': before.loc[window, 'PE'],
              'unsmoothed': after.loc[window, 'PE'],
              'true (simulated)': true_returns.loc[window, 'PE']}
    table = pd.DataFrame({label: {'lag1_autocorrelation': lag1_autocorrelation(s.to_numpy()),
                                  'annualised_volatility': annualised_vol(s.to_numpy())}
                          for label, s in series.items()}).T
    others = [ticker for ticker in TICKERS if ticker != 'PE']

    ink, muted, grid, surface = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
    blue, orange, grey = '#2a78d6', '#eb6834', '#b9b8b3'
    colours = [blue, orange, grey]
    plt.rcParams.update({'font.size': 11, 'axes.edgecolor': grid, 'axes.labelcolor': muted,
                         'xtick.color': muted, 'ytick.color': muted})
    # Autocorrelation and volatility are different measures, so each has its own axis.
    fig, (left, middle, right) = plt.subplots(1, 3, figsize=(10.0, 4.4), facecolor=surface,
                                              gridspec_kw={'width_ratios': [2.0, 1.0, 1.0]})
    anchor = window[0] - pd.offsets.QuarterEnd(1)
    for (label, s), colour in zip(series.items(), colours):
        path_ = pd.concat([pd.Series([0.0], index=[anchor]), s.cumsum()])
        width = 1.2 if colour == grey else 1.8
        left.plot(path_.index, path_.to_numpy(), color=colour, linewidth=width, label=label,
                  zorder=1 if colour == grey else 2)
    left.set_title('Cumulative log return of private equity', loc='left', color=ink)
    left.set_ylabel('Cumulative log return')
    left.xaxis.set_major_locator(mdates.YearLocator(6))
    left.legend(frameon=False, loc='upper left', fontsize=10, labelcolor=ink)
    x = np.arange(len(series))
    panels = [(middle, 'lag1_autocorrelation', 'Lag-1 autocorrelation', '{:.2f}', (-0.12, 0.75)),
              (right, 'annualised_volatility', 'Annualised volatility', '{:.1%}', (0.0, 0.25))]
    for axis, metric, title, style, limits in panels:
        values = table.loc[list(series), metric].to_numpy()
        axis.bar(x, values, 0.7, color=colours)
        offset = 0.015 * (limits[1] - limits[0])
        for k, value in enumerate(values):
            axis.text(k, value + (offset if value >= 0 else -offset), style.format(value),
                      ha='center', va='bottom' if value >= 0 else 'top', color=ink, fontsize=10)
        axis.axhline(0.0, color=muted, linewidth=0.8)
        axis.set_xticks(x, ['Reported', 'Unsmoothed', 'True'], rotation=30, ha='right',
                        rotation_mode='anchor')
        axis.set_ylim(*limits)
        axis.set_title(title, loc='left', color=ink)
    right.yaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f'{value:.0%}'))
    for axis in (left, middle, right):
        axis.set_facecolor(surface)
        axis.grid(axis='y', color=grid, linewidth=0.8)
        axis.set_axisbelow(True)
        for side in ('top', 'right'):
            axis.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=surface)
    plt.close(fig)
    smoothed, unsmooth = table.loc['reported (smoothed)'], table.loc['unsmoothed']
    checks = {
        'autocorrelation_falls': bool(unsmooth['lag1_autocorrelation']
                                      < smoothed['lag1_autocorrelation']),
        'volatility_rises': bool(unsmooth['annualised_volatility']
                                 > smoothed['annualised_volatility']),
        'other_columns_unchanged': bool(unsmoothed.prices[others].equals(
            universe.prices[others])),
    }
    return {'table': table, 'checks': checks}


if __name__ == '__main__':
    main()
