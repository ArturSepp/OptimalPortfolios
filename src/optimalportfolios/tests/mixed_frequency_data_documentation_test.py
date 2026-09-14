"""Execute mixed-frequency documentation and verify cadence, scoring, units and timing."""

from dataclasses import replace
import hashlib
import re
import runpy

import numpy as np
import pandas as pd
import pytest

LEGACY_ANCHORS = {
    'mixed-frequency-data', 'per-asset-return-frequencies', 'signal-horizons',
    'risk-estimation-and-rebalancing', 'failure-modes', 'see-also',
}
ORIGINAL_BLOCK_HASHES = [
    '35eabb5c204fc64955869a0bc2503fd28297f18a663f6288debc454162cc2b88',
    'dadf9fa6ff5e44fab1a1dbfe3fff7bcd922cbc60d041f0d42656bf948a4b95a5',
    'ebd4810fb2a697dfd81005b33e817d4431c75ecdc8b345264dc557c7b8919758',
]


@pytest.fixture(scope='module')
def article(root):
    """Read the canonical article only when a repository checkout is available."""
    return (root / 'docs/mixed_frequency_data.md').read_text(encoding='utf-8')


@pytest.fixture(scope='module')
def examples(article):
    """Execute all six canonical blocks in their documented order."""
    blocks = re.findall(r'^```python([^\n]*)\n(.*?)^```', article, re.M | re.S)
    assert len(blocks) == 6 and all(not options.strip() for options, _ in blocks)
    state = {'__name__': '__mixed_frequency_article__'}
    for index, (_, code) in enumerate(blocks, start=1):
        exec(compile(code, f'mixed_frequency_data.md (block {index})', 'exec'), state)
    return state


def _fresh_estimator(examples, **overrides):
    """Return an estimator with fresh LASSO state and the article's selected settings."""
    import optimalportfolios as opt

    configured = examples['factor_estimator']
    model = opt.LassoModel(**configured.lasso_model.get_params())
    return replace(configured, lasso_model=model, **overrides)


def test_structure_and_original_examples(article, root):
    """Keep portable methodology structure, source links, six anchors and all old code."""
    checker = runpy.run_path(str(root / 'tools/check_docs.py'))
    assert not checker['check_document'](article, methodology=True)
    assert not checker['check_local_links'](article, root / 'docs/mixed_frequency_data.md', root)
    headings = re.findall(r'^#{1,6} (.+)$', article, re.M)
    anchors = {re.sub(r'[^\w -]', '', heading).lower().replace(' ', '-') for heading in headings}
    assert LEGACY_ANCHORS <= anchors
    blocks = re.findall(r'^```python\n(.*?)^```', article, re.M | re.S)
    preserved = [blocks[1], blocks[2].split('ewma_scores,')[0],
                 blocks[3].split('classic_scores,')[0]]
    actual_hashes = [hashlib.sha256(code.encode()).hexdigest() for code in preserved]
    assert actual_hashes == ORIGINAL_BLOCK_HASHES
    assert not (root / 'docs/mixed_frequency_data.rst').exists()
    assert 'https://github.com/ArturSepp/FactorLasso/blob/main/CITATION.cff' in article


@pytest.mark.parametrize('frequency,assets,rows', [
    ('ME', ['Global Equity', 'Government Bonds'], 73),
    ('QE', ['Private Assets'], 25),
])
def test_bucket_returns_match_observed_endpoint_ratios(examples, frequency, assets, rows):
    """Derive bucket returns directly from genuine price endpoints and retain the initial zero."""
    observed = examples['prices'][assets].dropna()
    expected = np.log(observed / observed.shift())
    expected.iloc[0] = 0.0
    actual = examples['returns_by_frequency'][frequency]
    assert len(actual) == rows
    pd.testing.assert_frame_equal(actual, expected, check_freq=False, rtol=1e-12, atol=1e-15)
    pd.testing.assert_frame_equal(examples['log_by_frequency'][frequency], actual)
    np.testing.assert_allclose(examples['arithmetic_by_frequency'][frequency],
                               np.expm1(expected), rtol=1e-12, atol=1e-15)


# The public momentum function is under test; use endpoint ratios, not its rolling-sum helper.
@pytest.mark.parametrize('asset,lookback', [
    ('Global Equity', 12), ('Government Bonds', 12), ('Private Assets', 4),
])
def test_classic_signal_equals_included_window_endpoint_ratio(examples, asset, lookback):
    """One skipped observation changes the endpoint by one native period, not one month."""
    observed = examples['prices'][asset].dropna()
    expected = np.log(observed.shift(1) / observed.shift(lookback + 1))
    expected = expected.reindex(examples['classic_raw'].index).ffill()
    pd.testing.assert_series_equal(examples['classic_raw'][asset], expected,
                                   check_freq=False, rtol=1e-11, atol=1e-14)


def test_worked_table_matches_calculation_and_calendar_endpoints(article, examples):
    """All displayed numbers come from the executed raw signal and independently dated windows."""
    rows = re.findall(r'^\| (202[34]-\d\d-\d\d) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \|$',
                      article, re.M)
    assert len(rows) == 4
    dates = pd.to_datetime([row[0] for row in rows])
    values = np.array([[float(value) for value in row[1:]] for row in rows])
    np.testing.assert_allclose(values, examples['classic_raw'].loc[dates], rtol=0, atol=0.5e-6)
    prices = examples['prices']
    monthly = np.log(prices.loc['2023-11-30'].iloc[:2] / prices.loc['2022-11-30'].iloc[:2])
    quarterly = np.log(prices.at[pd.Timestamp('2023-09-30'), 'Private Assets']
                       / prices.at[pd.Timestamp('2022-09-30'), 'Private Assets'])
    np.testing.assert_allclose(values[0], [*monthly, quarterly], rtol=0, atol=0.5e-6)


@pytest.mark.parametrize('method', ['classic', 'ewma'])
def test_scoring_is_per_cadence_and_quarterly_updates_are_carried(examples, method):
    """A singleton quarterly raw signal remains useful, but its cross-sectional score is NaN."""
    raw, scores = examples[f'{method}_raw'], examples[f'{method}_scores']
    assert raw.columns.equals(examples['prices'].columns)
    assert scores['Private Assets'].isna().all()
    private = raw.loc['2023-12-31':'2024-02-29', 'Private Assets']
    assert private.notna().all() and private.nunique() == 1
    assert raw.at[pd.Timestamp('2024-03-31'), 'Private Assets'] != private.iloc[0]
    monthly = raw.iloc[:, :2].clip(-5, 5)
    expected = monthly.sub(monthly.mean(axis=1), axis=0).div(monthly.std(axis=1, ddof=0), axis=0)
    np.testing.assert_allclose(scores.iloc[:, :2], expected, rtol=1e-10, atol=1e-13, equal_nan=True)


@pytest.mark.parametrize('method', ['classic', 'ewma'])
def test_signal_history_is_unchanged_by_future_prices(examples, method):
    """Changing later prices must leave all earlier formation-date signals and scores unchanged."""
    opt = examples['opt']
    prices = examples['prices'].copy()
    cutoff = examples['cutoff']
    mask = prices.index > cutoff
    prices.loc[mask] *= np.linspace(1.2, 2.8, mask.sum())[:, None]
    if method == 'classic':
        changed = opt.compute_classic_momentum_alpha(
            prices, returns_freq=examples['return_frequencies'],
            lookback_periods={'ME': 12, 'QE': 4}, skip_periods={'ME': 1, 'QE': 1})
    else:
        changed = opt.compute_momentum_alpha(
            prices, returns_freq=examples['return_frequencies'],
            long_span={'ME': 12, 'QE': 4}, vol_span={'ME': 13, 'QE': 4})
    for name, actual in zip(['scores', 'raw'], changed):
        pd.testing.assert_frame_equal(
            actual.loc[:cutoff], examples[f'{method}_{name}'].loc[:cutoff])


def test_factor_variances_use_each_asset_cadence(examples):
    """Fit native bucket regressions independently and multiply monthly by 12, quarterly by 4."""
    from factorlasso import VarianceColumns

    data = examples['factor_data']
    cutoff = examples['cutoff']
    for frequency, annualization, span in [('ME', 12, 24), ('QE', 4, 8)]:
        returns = examples['returns_by_frequency'][frequency].loc[:cutoff]
        prices = examples['factor_prices'].loc[returns.index]
        factor_returns = np.log(prices).diff()
        model = examples['opt'].LassoModel(
            model_type=examples['opt'].LassoModelType.LASSO,
            reg_lambda=1e-5, span=span, warmup_period=8, demean=True, solver='CLARABEL')
        model.fit(x=factor_returns, y=returns)
        np.testing.assert_allclose(data.y_betas.loc[returns.columns],
                                   model.estimated_betas, rtol=1e-7, atol=1e-9)
        expected = annualization * model.estimation_result_.ss_res
        np.testing.assert_allclose(
            data.y_variances.loc[returns.columns, VarianceColumns.RESIDUAL_VARS.value],
            expected, rtol=1e-7, atol=1e-12)


# The owning QIS EWMA is under test; compute a finite weighted sum after pandas mean adjustment.
def test_factor_covariance_uses_monthly_span_and_annual_scale(examples):
    """Independently reconstruct monthly factor covariance, then assemble the asset matrix."""
    from factorlasso import VarianceColumns

    prices = examples['factor_prices'].loc[:examples['cutoff']]
    returns = np.log(prices).diff().iloc[1:]
    adjusted = (returns - returns.ewm(span=24, adjust=False).mean()).iloc[1:]
    decay = 1 - 2 / 25
    weights = (1 - decay) * decay ** np.arange(len(adjusted) - 1, -1, -1)
    expected = 12 * np.einsum('t,ti,tj->ij', weights, adjusted, adjusted)
    data = examples['factor_data']
    np.testing.assert_allclose(data.x_covar, expected, rtol=1e-10, atol=1e-14)
    betas = data.y_betas.to_numpy()
    residual = data.y_variances[VarianceColumns.RESIDUAL_VARS.value].to_numpy()
    assembled = betas @ expected @ betas.T + np.diag(residual)
    np.testing.assert_allclose(examples['annual_covar'], assembled, rtol=1e-10, atol=1e-14)


def test_current_cutoff_and_rolling_dates_are_separate_from_sampling(examples):
    """The canonical current result equals the last prefix-only rolling fit."""
    expected_dates = list(pd.to_datetime(['2023-06-30', '2023-09-30', '2023-12-31']))
    assert list(examples['rolling_covars']) == expected_dates
    np.testing.assert_allclose(examples['annual_covar'],
                               examples['rolling_covars'][examples['cutoff']], atol=1e-12)
    monthly = _fresh_estimator(examples, rebalancing_freq='ME').fit_rolling_factor_covars(
        risk_factor_prices=examples['factor_prices'],
        asset_returns_dict=examples['returns_by_frequency'],
        assets=examples['prices'].columns,
        time_period=examples['qis'].TimePeriod('2023-06-30', '2023-12-31'),
    ).get_y_covars()
    assert len(monthly) == 7
    for date in expected_dates:
        np.testing.assert_allclose(monthly[date], examples['rolling_covars'][date], atol=1e-12)


def test_rolling_mixed_factor_fit_ignores_future_prices(examples):
    """Perturb both later factors and asset returns, leaving earlier rolling estimates unchanged."""
    factors = examples['factor_prices'].copy()
    mask = factors.index > examples['cutoff']
    factors.loc[mask] *= np.linspace(1.2, 2.5, mask.sum())[:, None]
    buckets = {}
    for frequency, original in examples['returns_by_frequency'].items():
        changed = original.copy()
        changed.loc[changed.index > examples['cutoff']] += 0.3
        buckets[frequency] = changed
    actual = _fresh_estimator(examples).fit_rolling_factor_covars(
        risk_factor_prices=factors, asset_returns_dict=buckets,
        assets=examples['prices'].columns,
        time_period=examples['qis'].TimePeriod('2023-06-30', '2023-12-31'),
    ).get_y_covars()
    for date, expected in examples['rolling_covars'].items():
        pd.testing.assert_frame_equal(actual[date], expected)


def test_incomplete_mapping_is_not_a_uniform_validation_contract(examples):
    """QIS can omit an unmapped price column, while the signal wrapper requires it."""
    incomplete = examples['return_frequencies'].drop('Private Assets')
    buckets = examples['qis'].compute_asset_returns_dict(
        examples['prices'], returns_freqs=incomplete, is_log_returns=True)
    assert set(buckets) == {'ME'}
    with pytest.raises(KeyError):
        examples['opt'].compute_classic_momentum_alpha(
            examples['prices'], returns_freq=incomplete)
    with pytest.raises(ValueError, match='QE'):
        examples['opt'].compute_classic_momentum_alpha(
            examples['prices'], returns_freq=examples['return_frequencies'],
            lookback_periods={'ME': 12})


def test_missing_quarter_endpoint_can_become_a_stale_zero_return(examples):
    """A filled NAV can yield a finite return without a genuine observation."""
    prices = examples['prices'].copy()
    missing_date = pd.Timestamp('2023-12-31')
    prices.loc[missing_date, 'Private Assets'] = np.nan
    bucket = examples['qis'].compute_asset_returns_dict(
        prices, returns_freqs=examples['return_frequencies'], is_log_returns=True)['QE']
    assert bucket.at[missing_date, 'Private Assets'] == 0.0
    assert examples['returns_by_frequency']['QE'].at[missing_date, 'Private Assets'] != 0.0

