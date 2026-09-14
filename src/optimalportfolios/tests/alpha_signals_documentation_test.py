"""Execute the alpha guide and independently check signal, scoring and timing contracts."""

from dataclasses import fields, replace
import hashlib
import inspect
import re
import runpy

import numpy as np
import pandas as pd
import pytest

LEGACY_HEADINGS = {
    'Alpha signals — `optimalportfolios.alphas`', 'Architecture', 'Naming Conventions',
    'Signal Matrix', 'Signal Functions', 'Momentum', 'Low Beta', 'Residual Momentum',
    'Managers Alpha', 'Cluster-Based Scoring', 'Motivation', 'How Clusters Are Derived',
    'Scoring Logic', 'Cluster Signal Usage', 'Empirical Findings', 'Mixed-Frequency Support',
    'Within-Group Scoring (Fixed Groups)', 'Signal Comparison', 'AlphasData',
    'AlphasData Fields', 'References',
}
ORIGINAL_HASHES = [
    '27e8fb30bce268404b14e834411bda69569157d90194a2a2b3bacf528e1fa60e',
    'e3f0e421a2809bca7bf291097ece2fe214087ddbb560827778b5760fa021be2c',
    'a906daa349f684535b0604d91a720a6027d43d6f2e6a421f21c09efae3c147ae',
    'b1a3c01b875b3abb29690d427b4bc73547f7bf289fd91cfeb43de86fb25f27fb',
    'aea36deb27af468cfe201ca4a2a85fa3e88fc702c5afb9716e6f9b8c7c77c948',
    'dd411197cc4cf9ba7ecc4c1c5728710748cb01f1a380ad12804aa44f885f0cc3',
    'c8b99d31c6f45cec4bd790ac5ee2310b1df1d6d82e30952a104361170be5daf1',
    'd9cc8d23b4452eb01fe249f6e32e6fbbab187da7b0ca01259e7d4fdb78c910dc',
]
FAMILIES = ['momentum', 'classic_momentum', 'low_beta', 'residual_momentum',
            'residual_reversal', 'ra_carry']
RAW_NAMES = {
    'momentum': 'raw_momentum', 'classic_momentum': 'raw_classic', 'low_beta': 'raw_beta',
    'residual_momentum': 'raw_residual', 'residual_reversal': 'raw_reversal',
    'ra_carry': 'raw_carry',
}
SCORE_NAMES = {
    'momentum': 'mom_score', 'classic_momentum': 'classic_score', 'low_beta': 'beta_score',
    'residual_momentum': 'res_score', 'residual_reversal': 'reversal_score',
    'ra_carry': 'carry_score',
}


@pytest.fixture(scope='module')
def article(root):
    """Read the authoritative guide only when its repository source is available."""
    return (root / 'docs/alphas_module_readme.md').read_text(encoding='utf-8')


@pytest.fixture(scope='module')
def examples(article):
    """Execute every published block with the core stack and network access denied."""
    blocks = re.findall(r'^```python([^\n]*)\n(.*?)^```', article, re.M | re.S)
    assert len(blocks) == 15 and all(not options.strip() for options, _ in blocks)
    state = {'__name__': '__alpha_article__'}

    def reject_network(*args, **kwargs):
        """Fail immediately if an executable documentation block attempts a network call."""
        raise AssertionError('Alpha documentation examples must remain offline')

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr('socket.create_connection', reject_network)
        patch.setattr('socket.socket.connect', reject_network)
        for i, (_, code) in enumerate(blocks):
            exec(compile(code, f'alphas_module_readme.md (block {i})', 'exec'), state)
    return state


# QIS EWM/filter kernels are under test; explicit geometric weights provide a separate reference.
def _ewm_reference(values, span, seed_first=False):
    """Apply finite-input geometric weights, with a zero or first-observation initial state."""
    values = np.asarray(values, dtype=float)
    count = len(values)
    decay = 1 - 2 / (span + 1)
    age = np.arange(count)[:, None] - np.arange(count)[None, :]
    weights = np.where(age >= 0, (1 - decay) * decay ** np.maximum(age, 0), 0.0)
    if seed_first:
        weights[:, 0] = decay ** np.arange(count)
    return weights @ values


def _log_returns(prices):
    """Derive returns directly from adjacent observed levels, including the initial zero."""
    result = np.log(prices / prices.shift())
    result.iloc[0] = 0.0
    return result


def _standard_score(raw):
    """Use explicit clipping and population moments, independent of the public QIS scorer."""
    clipped = raw.clip(-5.0, 5.0)
    return clipped.sub(clipped.mean(axis=1), axis=0).div(clipped.std(axis=1, ddof=0), axis=0)


def _constructor(examples, family, cluster=False):
    """Return the appropriate public constructor and the common monthly example settings."""
    suffix = '_cluster_alpha' if cluster else '_alpha'
    function = getattr(examples['signals'], 'compute_' + family + suffix)
    kwargs = {'returns_freq': 'ME'}
    if family not in ['classic_momentum', 'ra_carry']:
        kwargs['benchmark_price'] = examples['benchmark']
    if family == 'ra_carry':
        kwargs['carry'] = examples['carry']
    if cluster:
        kwargs['rolling_clusters'] = examples['rolling_clusters']
    return function, kwargs


def test_structure_and_preserved_examples(article, root):
    """Keep old headings and eight concrete examples; replace four pseudocode blocks."""
    checker = runpy.run_path(str(root / 'tools/check_docs.py'))
    assert not checker['check_document'](article, methodology=True)
    path = root / 'docs/alphas_module_readme.md'
    assert not checker['check_local_links'](article, path, root)
    assert LEGACY_HEADINGS <= set(re.findall(r'^#{1,4} (.+)$', article, re.M))
    blocks = re.findall(r'^```python\n(.*?)^```', article, re.M | re.S)
    original = [
        blocks[1].split('mom_score, raw_momentum =')[0],
        blocks[2].split('beta_score =')[0],
        blocks[3].split('res_score =')[0],
        blocks[6].split('mgr_score =')[0],
        blocks[7].replace('assets=prices.columns.tolist()',
                          'assets=universe_data.get_taa_prices().columns.tolist()'),
        blocks[8],
        blocks[9].split('mom_cluster_score, raw_momentum_cluster =')[0],
        blocks[13][blocks[13].index('from optimalportfolios.alphas import AlphasData'):],
    ]
    assert [hashlib.sha256(code.encode()).hexdigest() for code in original] == ORIGINAL_HASHES
    assert '55% → 60%' not in article


def test_signal_matrix_matches_public_paired_constructors(article, examples):
    """Check all 13 constructor names and the deliberately different carry export/return paths."""
    matrix = article.split('### Signal Matrix\n', 1)[1].split('### AlphasData Fields', 1)[0]
    documented = set(re.findall(r'`(compute_\w+_alpha)`', matrix))
    exported = {name for name, value in vars(examples['signals']).items()
                if name.startswith('compute_') and name.endswith('_alpha') and callable(value)}
    assert documented == exported and len(exported) == 13
    assert not hasattr(examples['alphas'], 'compute_ra_carry_alpha')
    pd.testing.assert_frame_equal(examples['legacy_carry_score'], examples['carry_score'])
    for family in FAMILIES:
        function, _ = _constructor(examples, family)
        if family != 'ra_carry':
            assert inspect.signature(function).parameters['returns_freq'].default == 'ME'
    assert inspect.signature(examples['signals'].compute_ra_carry_alpha).parameters[
        'returns_freq'].default == 'W-WED'


@pytest.mark.parametrize('family', FAMILIES)
def test_standard_scores_match_their_raw_characteristic(examples, family):
    """All six standard families use clipped population scores; low beta reverses only the score."""
    raw = examples[RAW_NAMES[family]]
    expected = _standard_score(raw)
    if family == 'low_beta':
        expected = -expected
    np.testing.assert_allclose(examples[SCORE_NAMES[family]], expected,
                               rtol=1e-11, atol=1e-12, equal_nan=True)
    assert raw.shape == (97, 8)


def test_momentum_default_has_contemporaneous_vol_and_unit_variance_filter(examples):
    """Derive the default signal from squared-return EWM weights and a separate geometric sum."""
    relative = _log_returns(examples['prices']).sub(_log_returns(examples['benchmark']), axis=0)
    variance = _ewm_reference(relative.to_numpy() ** 2, span=13)
    adjusted = np.divide(relative, np.sqrt(variance), out=np.zeros_like(variance),
                         where=variance > 0)
    expected = np.sqrt(12) * _ewm_reference(adjusted, span=12)
    np.testing.assert_allclose(examples['raw_momentum'].iloc[24:], expected[24:],
                               rtol=1e-11, atol=1e-12)


@pytest.mark.parametrize('short_span', [None, 3])
def test_unbenchmarked_momentum_uses_own_returns_and_an_exponential_filter(examples, short_span):
    """No benchmark means no subtraction; a short leg changes filter weights rather than skips."""
    _, actual = examples['signals'].compute_momentum_alpha(
        examples['prices'], benchmark_price=None, vol_span=None, short_span=short_span)
    returns = _log_returns(examples['prices']).to_numpy()
    long_decay = 1 - 2 / 13
    if short_span is None:
        expected = np.sqrt(12) * _ewm_reference(returns, 12)
    else:
        short_decay = 1 - 2 / (short_span + 1)
        normalizer = np.sqrt(1 / (1 - long_decay ** 2) + 1 / (1 - short_decay ** 2)
                             - 2 / (1 - long_decay * short_decay))
        expected = (_ewm_reference(returns, 12) / (1 - long_decay)
                    - _ewm_reference(returns, short_span) / (1 - short_decay)) / normalizer
    np.testing.assert_allclose(actual.iloc[24:], expected[24:], rtol=1e-11, atol=1e-12)


def test_classic_momentum_matches_price_window_endpoints(examples):
    """The default includes exactly twelve returns and excludes the latest observation."""
    prices = examples['prices']
    expected = np.log(prices.shift(1) / prices.shift(13))
    np.testing.assert_allclose(examples['raw_classic'].iloc[13:], expected.iloc[13:],
                               rtol=1e-11, atol=1e-14)
    returns = np.log(prices / prices.shift())  # the price constructor keeps its first NaN
    direct = examples['signals'].compute_classic_momentum_from_returns(
        returns, lookback_periods=12, skip_periods=1)
    pd.testing.assert_frame_equal(direct, examples['raw_classic'], check_freq=False,
                                  rtol=1e-11, atol=1e-14)


def test_low_beta_recovers_known_proportional_return_loadings(examples):
    """Recover known betas from proportional returns without duplicating the regression engine."""
    benchmark = examples['benchmark']
    true_betas = np.array([0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 6.0])
    prices = pd.DataFrame(
        (benchmark.to_numpy()[:, None] / benchmark.iloc[0]) ** true_betas,
        index=benchmark.index, columns=examples['prices'].columns)
    score, raw = examples['signals'].compute_low_beta_alpha(prices, benchmark_price=benchmark)
    np.testing.assert_allclose(raw.iloc[30:], np.broadcast_to(true_betas, raw.iloc[30:].shape),
                               rtol=1e-10, atol=1e-11)
    assert score.iloc[-1, 0] > score.iloc[-1, -1]
    assert raw.iloc[-1, -1] > 5  # clipping is a scoring step, not a raw-beta mutation


def test_residuals_use_previous_beta_and_reversal_changes_sign(examples):
    """An unnormalised one-period filter exposes the lagged regression residual directly."""
    function = examples['signals'].compute_residual_momentum_alpha
    kwargs = dict(prices=examples['prices'], benchmark_price=examples['benchmark'],
                  beta_span=12, long_span=1, vol_span=None)
    score, actual = function(**kwargs)
    returns = _log_returns(examples['prices'])
    benchmark_returns = _log_returns(examples['benchmark'])
    expected = returns - examples['raw_beta'].shift(1).mul(benchmark_returns, axis=0)
    np.testing.assert_allclose(actual.iloc[30:], expected.iloc[30:], rtol=1e-10, atol=1e-12)
    reversed_score, reversed_raw = examples['signals'].compute_residual_reversal_alpha(**kwargs)
    np.testing.assert_allclose(reversed_raw, -actual, equal_nan=True, atol=1e-13)
    np.testing.assert_allclose(reversed_score, -score, equal_nan=True, atol=1e-12)
    assert not np.allclose(examples['raw_reversal'].iloc[30:],
                           -examples['raw_residual'].iloc[30:])


@pytest.mark.parametrize('span', [13, None])
def test_carry_uses_annual_volatility_even_when_span_is_none(examples, span):
    """Use annual volatility; None selects decay 0.94 rather than disabling normalisation."""
    _, actual = examples['signals'].compute_ra_carry_alpha(
        examples['prices'], carry=examples['carry'], returns_freq='ME', vol_span=span)
    returns = _log_returns(examples['prices']).to_numpy()
    effective_span = span if span is not None else 2 / (1 - 0.94) - 1
    variance = 12 * _ewm_reference(returns ** 2, effective_span)
    expected = examples['carry'].to_numpy()[1:] / np.sqrt(variance[1:])
    np.testing.assert_allclose(actual.iloc[1:], expected, rtol=1e-11, atol=1e-12)


def test_manager_alphas_use_prior_snapshots_annual_scaling_and_no_centering(examples):
    """Assemble residuals from observed endpoints and independently smooth the annual amounts."""
    prices, factors = examples['asset_prices'], examples['factor_prices']
    returns = np.log(prices / prices.shift())
    factor_returns = np.log(factors / factors.shift())
    beta_history = examples['rolling_data'].get_y_betas()
    residuals = {}
    for previous, current in zip(prices.index[1:-1], prices.index[2:]):
        eligible = [date for date in beta_history if date <= previous]
        if eligible:
            betas = beta_history[max(eligible)].loc[prices.columns, factors.columns]
            contributions = betas.mul(factor_returns.loc[current], axis=1).sum(axis=1)
            residuals[current] = returns.loc[current] - contributions
    residuals = pd.DataFrame.from_dict(residuals, orient='index')
    expected = 12 * _ewm_reference(residuals.to_numpy(), 12, seed_first=True)
    pd.testing.assert_index_equal(examples['raw_alpha'].index, residuals.index)
    np.testing.assert_allclose(examples['raw_alpha'], expected, rtol=1e-10, atol=1e-12)
    expected_score = expected / np.std(expected, axis=1, keepdims=True, ddof=0)
    np.testing.assert_allclose(examples['mgr_score'], expected_score, rtol=1e-10, atol=1e-11)
    assert np.abs(examples['mgr_score'].mean(axis=1)).max() > 0.1
    _, period_raw = examples['signals'].compute_managers_alpha(
        prices, factors, beta_history, returns_freq='ME', alpha_span=12, annualise=False)
    np.testing.assert_allclose(period_raw * 12, examples['raw_alpha'], rtol=1e-10, atol=1e-12)


@pytest.mark.parametrize('family', FAMILIES)
def test_cluster_and_standard_variants_share_raw_inputs(examples, family):
    """An explicit benchmark and one cadence leave each pair's raw signal unchanged."""
    function, kwargs = _constructor(examples, family, cluster=True)
    _, raw = function(examples['prices'], **kwargs)
    np.testing.assert_allclose(raw, examples[RAW_NAMES[family]], rtol=1e-11, atol=1e-12,
                               equal_nan=True)


def test_displayed_scoring_table_matches_both_independent_formulas(article, examples):
    """Distinguish clipping and population moments from unbounded sample cluster statistics."""
    rows = re.findall(r'^\| ([A-F]) \| (-?\d+) \| (-?[\d.]+) \| (-?[\d.]+) \|$', article, re.M)
    assert len(rows) == 6
    displayed = np.array([[float(v) for v in row[1:]] for row in rows])
    raw = examples['raw_probe'].iloc[0]
    sample = (raw - raw.mean()) / raw.std(ddof=1)
    sample.iloc[:4] = (raw.iloc[:4] - raw.iloc[:4].mean()) / raw.iloc[:4].std(ddof=1)
    expected = np.column_stack((raw, _standard_score(examples['raw_probe']).iloc[0], sample))
    np.testing.assert_allclose(displayed, expected, rtol=0, atol=0.5e-6)
    np.testing.assert_allclose(examples['score_comparison'], expected, rtol=1e-12, atol=1e-12)


def test_cluster_threshold_empty_groups_and_unassigned_assets(examples):
    """The threshold is inclusive; empty mappings, missing labels and prehistory differ."""
    function = examples['signals'].score_within_clusters
    raw = examples['raw_probe']
    labels = examples['probe_clusters']
    all_global = function(raw, labels, min_cluster_size=4)
    expected = raw.sub(raw.mean(axis=1), axis=0).div(raw.std(axis=1, ddof=1), axis=0)
    np.testing.assert_allclose(all_global, expected, rtol=1e-12)
    np.testing.assert_allclose(function(raw, {}), _standard_score(raw), rtol=1e-12)
    later = raw.index[0] + pd.Timedelta(days=1)
    np.testing.assert_allclose(function(raw, {later: next(iter(labels.values()))}), 0.0)
    incomplete = next(iter(labels.values())).drop(index='F')
    result = function(raw, {raw.index[0]: incomplete})
    assert result.at[raw.index[0], 'F'] == 0.0
    fallback = raw.iloc[0, :5]
    assert result.at[raw.index[0], 'E'] == pytest.approx(
        (fallback['E'] - fallback.mean()) / fallback.std(ddof=1))


def test_cluster_extraction_keeps_estimator_memberships(examples):
    """Read labels stored by HCGL; the extraction helper must not derive clusters from betas."""
    rolling = examples['rolling_data']
    assert list(examples['rolling_clusters']) == list(rolling.dates)
    for date, labels in examples['rolling_clusters'].items():
        expected = rolling[date].clusters.reindex(examples['prices'].columns)
        pd.testing.assert_series_equal(labels, expected,
                                       check_names=False)
        assert labels.astype(str).str.startswith('ME:').all()


def test_mixed_cadence_and_fixed_groups_use_the_stated_comparison_sets(examples):
    """Quarterly endpoint windows and per-group population scores agree with independent data."""
    for assets, lookback in [(list('ABCDEF'), 12), (list('GH'), 4)]:
        prices = examples['mixed_prices'][assets].dropna()
        expected = np.log(prices.shift(1) / prices.shift(lookback + 1))
        actual = examples['mixed_raw'].loc[prices.index, assets]
        np.testing.assert_allclose(actual.iloc[lookback + 1:], expected.iloc[lookback + 1:],
                                   rtol=1e-11, atol=1e-14)
    quarterly = examples['mixed_raw'].loc['2024-09-30':'2024-11-30', ['G', 'H']]
    assert (quarterly.nunique() == 1).all()
    pd.testing.assert_frame_equal(examples['group_raw'], examples['raw_momentum'])
    for assets in [list('ABCD'), list('EFGH')]:
        np.testing.assert_allclose(examples['group_score'][assets],
                                   _standard_score(examples['raw_momentum'][assets]),
                                   rtol=1e-11, atol=1e-12, equal_nan=True)


@pytest.mark.parametrize('family,mean_adjustment', [
    pytest.param(family, 'default', marks=pytest.mark.xfail(
        strict=True, reason='QIS MEAN initialisation uses future data; numerical fix is pending'))
    if family in ['low_beta', 'residual_momentum', 'residual_reversal']
    else pytest.param(family, 'default')
    for family in FAMILIES
] + [(family, 'NONE') for family in ['low_beta', 'residual_momentum', 'residual_reversal']])
@pytest.mark.parametrize('cluster', [False, True])
def test_future_observations_do_not_change_earlier_signals(
        examples, family, mean_adjustment, cluster):
    """Change later prices, benchmark and carry without altering any earlier raw signal or score."""
    function, kwargs = _constructor(examples, family, cluster)
    if mean_adjustment == 'NONE':
        kwargs['mean_adj_type'] = examples['qis'].MeanAdjType.NONE
    cutoff = pd.Timestamp('2023-06-30')
    original = function(examples['prices'], **kwargs)
    prices = examples['prices'].copy()
    future = prices.index > cutoff
    prices.loc[future, 'A'] *= np.linspace(1.1, 2.0, future.sum())
    if 'benchmark_price' in kwargs:
        benchmark = kwargs['benchmark_price'].copy()
        benchmark.loc[future] *= np.linspace(1.03, 1.3, future.sum())
        kwargs['benchmark_price'] = benchmark
    if 'carry' in kwargs:
        carry = kwargs['carry'].copy()
        carry.loc[future, 'A'] *= 2
        kwargs['carry'] = carry
    if cluster:
        labels = dict(kwargs['rolling_clusters'])
        labels[pd.Timestamp('2024-01-31')] = pd.Series('future', index=prices.columns)
        kwargs['rolling_clusters'] = labels
    changed = function(prices, **kwargs)
    for before, after in zip(original, changed):
        np.testing.assert_allclose(before.loc[:cutoff], after.loc[:cutoff],
                                   rtol=1e-11, atol=1e-12, equal_nan=True)
    assert not np.allclose(original[1].iloc[-6:], changed[1].iloc[-6:], equal_nan=True)


def test_manager_current_snapshot_cannot_residualise_current_period(examples):
    """An extreme loading dated at the return endpoint may affect only later return periods."""
    history = examples['rolling_data'].get_y_betas()
    changed = {date: betas.copy() for date, betas in history.items()}
    last_date = max(changed)
    changed[last_date] *= 100
    scores, raw = examples['signals'].compute_managers_alpha(
        examples['asset_prices'], examples['factor_prices'], changed)
    pd.testing.assert_frame_equal(raw, examples['raw_alpha'])
    pd.testing.assert_frame_equal(scores, examples['mgr_score'])


def test_container_keeps_supplied_panels_and_exposes_snapshot_fallback(examples):
    """No CDF is applied; a missing component date currently falls back to its last row."""
    data = examples['data']
    expected = 0.5 * examples['mom_score'] + 0.5 * examples['beta_score']
    pd.testing.assert_frame_equal(data.alpha_scores, expected)
    assert examples['snapshot'].shape == (8, 16)
    assert len(fields(type(data))) == len(examples['output']) == 16
    assert examples['output']['alpha_scores'] is data.alpha_scores
    earlier = pd.Timestamp('2024-06-30')
    late_component = pd.DataFrame(17.0, index=[pd.Timestamp('2025-01-31')],
                                  columns=data.alpha_scores.columns)
    misaligned = replace(data, momentum_score=late_component)
    assert (misaligned.get_alphas_snapshot(earlier)['Momentum Score'] == 17.0).all()
    with pytest.raises(KeyError):
        data.get_alphas_snapshot(pd.Timestamp('2000-01-31'))


def test_profiler_diagnostics_and_rolling_means_preserve_their_distinct_outputs(examples):
    """Check QIS-backed evaluation outputs and annual log means against geometric weights."""
    assert len(examples['profiles'].portfolio_datas) == 3
    assert len(examples['component_panels']) == 8
    assert len(examples['diagnostics'].horizon_labels) == 2
    returns = _log_returns(examples['prices']).iloc[1:]
    expected = pd.DataFrame(
        12 * _ewm_reference(returns, span=12, seed_first=True),
        index=returns.index, columns=returns.columns,
    ).loc[examples['mean_dates']]
    pd.testing.assert_frame_equal(examples['annual_log_means'], expected, check_freq=False,
                                  rtol=1e-11, atol=1e-12)
    assert expected.shape == (8, 8)


def test_rank_selection_ties_and_nonmissing_only_mask(examples):
    """The selection rule uses ceil, column-order ties and nonmissing rather than finite checks."""
    raw = pd.DataFrame([[3., 3., 2., 1., 0., -1., -2., -3.]],
                       index=[examples['probe_date']], columns=examples['prices'].columns)
    prices = examples['prices'].loc[raw.index]
    function = examples['alphas'].compute_top_quantile_equal_weights
    weights = function(raw, prices, quantile=0.125)
    np.testing.assert_allclose(weights.iloc[0], [1, 0, 0, 0, 0, 0, 0, 0])
    raw.iloc[0, -1] = np.inf
    prices = prices.copy()
    prices.iloc[0, -1] = -1.0
    weights = function(raw, prices, quantile=0.125)
    assert weights.iloc[0, -1] == 1.0

