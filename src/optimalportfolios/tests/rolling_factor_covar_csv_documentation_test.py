"""Execute the six-CSV guide and verify FX economics, persisted inputs and rolling risk."""

from dataclasses import replace
import hashlib
from io import StringIO
import re
import runpy
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

LEGACY_ANCHORS = {
    'rolling-factor-risk-model-from-csv', 'pipeline-at-a-glance', 'install-and-run',
    'the-six-file-csv-contract', 'yahoo-demonstration-basis', 'fetch-and-persist',
    'reload-every-input', 'convert-asset-returns', 'fit-the-rolling-decomposition',
    'numerical-reconstruction-check', 'replace-yahoo-with-delivered-matf-data',
    'persistence-boundary', 'see-also',
}
ORIGINAL_BLOCK_HASHES = [
    'aea81f34652384f65e16c22763b288eb34e7b83b30516b3025516aaf6aeadc94',
    '90aad50d78400f25521170c3161ca9358e08ffb9e48bf7fd5c8df73190549901',
    '9e9452d662a7b5f8f506b0d36015c071d467d5bbdf7e1e9986c8541e88fd8c01',
    '8d4b1ae37d97a4ce0467bee12a99e1f3ec140828bc6b2981fa8324ee6fea3cde',
    'edfb2e42784ddc7db3e48836b7ed1d3985193dce6964f5ff9401ab5e2bbbcef1',
    'e533fd3cd7e04ba7376f2ae3468b09b82d8a4be18f1bbe7946f49cedb450fb8e',
    '6fd61452e2ab1d3b98120c60d310e343cc9b1929d526eb37958f88c2b6c13cdb',
]
FILES = {
    'futures_risk_factors.csv', 'fx_hedging_data_fx_spots.csv',
    'fx_hedging_data_domestic_rates.csv', 'asset_prices.csv',
    'asset_metadata.csv', 'risk_model_settings.csv',
}


@pytest.fixture(scope='module')
def article(root):
    """Read the authoritative guide only from a source checkout."""
    return (root / 'docs/rolling_factor_covar_from_csv.md').read_text(encoding='utf-8')


@pytest.fixture(scope='module')
def examples(article, root, tmp_path_factory):
    """Execute seven offline blocks with isolated files and blocked network access."""
    blocks = re.findall(r'^```python([^\n]*)\n(.*?)^```', article, re.M | re.S)
    assert len(blocks) == 8
    assert [i for i, (option, _) in enumerate(blocks) if option.strip()] == [6]
    assert blocks[6][0].strip() == '+SKIP'
    data_dir = tmp_path_factory.mktemp('factor-csv-docs')
    state = {'__name__': '__factor_csv_article__'}

    def fresh_directory(**kwargs):
        """Place the documented temporary bundle in pytest's isolated output tree."""
        assert kwargs == {'prefix': 'op-factor-csv-'}
        return str(data_dir)

    def reject_network(*args, **kwargs):
        """Make accidental data acquisition fail before opening any connection."""
        raise AssertionError('The documented load walkthrough must remain offline')

    with pytest.MonkeyPatch.context() as patch:
        patch.syspath_prepend(str(root))
        patch.setattr('socket.create_connection', reject_network)
        patch.setattr('socket.socket.connect', reject_network)
        from examples.covar_estimation import rolling_factor_covar_from_csv as csv_example
        patch.setattr(csv_example, '_download_close', reject_network)
        # Patch only while the first block creates its explicitly named directory.
        with pytest.MonkeyPatch.context() as paths:
            paths.setattr(tempfile, 'mkdtemp', fresh_directory)
            exec(compile(blocks[0][1], 'CSV article (fixture)', 'exec'), state)
        state['original_inputs'] = {
            name: state[name].copy()
            for name in ['factor_prices', 'asset_prices', 'metadata', 'fx_spots', 'domestic_rates']
        }
        state['original_settings'] = state['settings']
        state['bundle_hashes'] = _hashes(data_dir)
        for index in range(1, 6):
            exec(compile(blocks[index][1], f'CSV article (block {index})', 'exec'), state)
        state['manual_rolling'] = state['rolling']
        exec(compile(blocks[7][1], 'CSV article (convenience loader)', 'exec'), state)
        state['csv_example'] = csv_example
        yield state


def _hashes(directory):
    """Measure every bundle file so estimation cannot silently alter persisted inputs."""
    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in directory.iterdir()}


def _copy_bundle(examples, destination):
    """Copy the exact worked input files for one validation or timing experiment."""
    for name in FILES:
        shutil.copyfile(examples['data_dir'] / name, destination / name)
    return destination


def _rewrite(directory, filename, change):
    """Apply a controlled edit while preserving CSV index and header conventions."""
    path = directory / filename
    frame = pd.read_csv(path, index_col=0)
    change(frame).to_csv(path)


def test_structure_anchors_and_original_examples(article, root):
    """Keep all old section anchors and code, apart from two documented path substitutions."""
    checker = runpy.run_path(str(root / 'tools/check_docs.py'))
    assert not checker['check_document'](article, methodology=True)
    path = root / 'docs/rolling_factor_covar_from_csv.md'
    assert not checker['check_local_links'](article, path, root)
    headings = re.findall(r'^#{1,6} (.+)$', article, re.M)
    anchors = {re.sub(r'[^\w -]', '', h).lower().replace(' ', '-') for h in headings}
    assert LEGACY_ANCHORS <= anchors
    assert not path.with_suffix('.rst').exists()
    blocks = re.findall(r'^```python[^\n]*\n(.*?)^```', article, re.M | re.S)
    original = [blocks[i] for i in [6, 1, 2, 3, 4, 5, 7]]
    original[1] = original[1].replace(
        'load_inputs_from_csv(data_dir)', 'load_inputs_from_csv(Path("path/to/risk_model_inputs"))')
    original[6] = original[6].replace(
        '    data_dir\n', '    Path("path/to/risk_model_inputs")\n')
    assert [hashlib.sha256(c.encode()).hexdigest() for c in original] == ORIGINAL_BLOCK_HASHES


def test_default_calibration_table(article, examples):
    """The 15 documented Yahoo settings agree with the canonical example defaults."""
    tables = re.findall(r'^```text\n(.*?)^```', article, re.M | re.S)
    calibration = next(t for t in tables if t.startswith('setting,value\n'))
    documented = pd.read_csv(StringIO(calibration), index_col=0, dtype=str)
    default = examples['csv_example'].RiskModelSettings.yahoo_demo()
    assert len(documented) == 15
    assert examples['csv_example'].RiskModelSettings.from_frame(documented) == default
    assert examples['settings'] == replace(
        default, factor_names=('Equity', 'Rates'), estimation_end=pd.Timestamp('2022-12-31'))


def test_all_six_files_round_trip_without_output_artifacts(examples):
    """Prices, FX, metadata and typed settings all survive a fresh CSV load unchanged."""
    inputs = examples['inputs']
    before = examples['original_inputs']
    assert set(_hashes(examples['data_dir'])) == FILES
    assert _hashes(examples['data_dir']) == examples['bundle_hashes']
    for name, actual in [
        ('factor_prices', inputs.factors_data.get_prices()), ('asset_prices', inputs.asset_prices),
        ('metadata', inputs.asset_metadata), ('fx_spots', inputs.fx_rates_data.fx_spots),
        ('domestic_rates', inputs.fx_rates_data.domestic_rates),
    ]:
        pd.testing.assert_frame_equal(actual, before[name], check_freq=False,
                                      check_names=False, rtol=1e-13, atol=1e-14)
    assert inputs.settings == examples['original_settings']


@pytest.mark.parametrize('asset', ['Growth', 'Income', 'Domestic'])
def test_converted_returns_equal_endpoint_wealth(examples, asset):
    """Rebuild reference-currency wealth using price/spot endpoints and contracted forwards."""
    data = examples['original_inputs']
    prices, spots, rates = data['asset_prices'], data['fx_spots'], data['domestic_rates']
    currency = data['metadata'].at[asset, 'currency']
    hedge = data['metadata'].at[asset, 'hedge_ratio']
    spot = spots[currency] / spots['CHF']
    price_gross, spot_gross = prices[asset] / prices[asset].shift(), spot / spot.shift()
    forward_ratio = (1 + rates['CHF'].shift() / 12) / (1 + rates[currency].shift() / 12)
    wealth = price_gross * spot_gross + hedge * (forward_ratio - spot_gross)
    expected = np.log(wealth)
    actual = examples['asset_returns_dict']['ME'][asset]
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-14, equal_nan=True)
    assert actual.iloc[0:1].isna().all()


def test_excess_returns_subtract_starting_reference_cash_in_log_space(examples):
    """Vary reference cash to distinguish starting-period accrual from terminal rates."""
    inputs, metadata = examples['inputs'], examples['metadata']
    rates = inputs.fx_rates_data.domestic_rates.copy()
    rates['CHF'] = 0.01 + 0.006 * np.sin(np.arange(len(rates)) * 0.7)
    fx_data = examples['qis'].FxRatesData(
        fx_spots=inputs.fx_rates_data.fx_spots.copy(), domestic_rates=rates)
    kwargs = dict(prices=inputs.asset_prices, hedge_ratios=metadata['hedge_ratio'],
                  local_ccys=metadata['currency'], reference_ccy='CHF',
                  freq=metadata['return_frequency'], is_log_returns=True)
    total = fx_data.compute_fx_adjusted_returns(**kwargs, is_excess_returns=False)['ME']
    actual = fx_data.compute_fx_adjusted_returns(**kwargs, is_excess_returns=True)['ME']
    cash = np.log1p(rates['CHF'].shift() / 12)
    expected = total.sub(cash, axis=0)
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-14, equal_nan=True)
    terminal_cash = total.sub(np.log1p(rates['CHF'] / 12), axis=0)
    assert not np.allclose(actual, terminal_cash, rtol=1e-11, atol=1e-14, equal_nan=True)


def test_genuine_zero_returns_are_missing_in_the_fx_wrapper(examples):
    """Characterize the documented exact-zero policy without changing the numerical engine."""
    inputs, metadata = examples['inputs'], examples['metadata']
    prices = inputs.asset_prices.copy()
    prices.iloc[25, 2] = prices.iloc[24, 2]
    result = inputs.fx_rates_data.compute_fx_adjusted_returns(
        prices=prices, hedge_ratios=metadata['hedge_ratio'], local_ccys=metadata['currency'],
        reference_ccy='CHF', freq=metadata['return_frequency'], is_log_returns=True)['ME']
    assert result.iloc[24, 2] != 0 and pd.isna(result.iloc[25, 2])
    assert np.isfinite(result.iloc[26, 2])


def test_displayed_sizes_dates_and_all_snapshot_covariances(article, examples):
    """Check all structural results and assemble covariance by sums of labelled factor terms."""
    rolling = examples['rolling']
    dates = pd.date_range('2019-12-31', '2022-12-31', freq='YE')
    pd.testing.assert_index_equal(rolling.dates, dates, check_names=False)
    assert '| CSV files | 6 |' in article
    assert '| Price observations per time series | 73 month ends |' in article
    assert '| Factor / asset count | 2 / 3 |' in article
    assert '| Betas / factor covariance / asset covariance | 3 × 2 / 2 × 2 / 3 × 3 |' in article
    assert '| Snapshot dates | ' + ', '.join(dates.strftime('%Y-%m-%d')) + ' |' in article
    assert examples['asset_prices'].shape == (73, 3)
    assert examples['factor_prices'].shape == (73, 2)
    assert set(examples['asset_returns_dict']) == {'ME'}
    for date, snapshot in rolling.data.items():
        betas, factor = snapshot.y_betas, snapshot.x_covar
        assert betas.shape == (3, 2) and factor.shape == (2, 2)
        assert snapshot.estimation_date == date
        residual = snapshot.y_variances['residual_var']
        expected = pd.DataFrame(0.0, index=betas.index, columns=betas.index)
        for a in betas.index:
            for b in betas.index:
                expected.at[a, b] = sum(
                    betas.at[a, f] * factor.at[f, g] * betas.at[b, g]
                    for f in factor.index for g in factor.columns)
                if a == b:
                    expected.at[a, b] += residual[a]
        actual = snapshot.get_y_covar()
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-14)
        assert np.isfinite(actual.to_numpy()).all()
        assert np.linalg.eigvalsh(actual.to_numpy()).min() >= -1e-10
        pd.testing.assert_frame_equal(actual, examples['manual_rolling'][date].get_y_covar())


def test_residual_history_is_annual_scaled_and_has_no_intercept_subtraction(examples):
    """Distinguish the stored residual panel from monthly regression errors and annual D."""
    snapshot = examples['rolling'].get_latest()
    factor_returns = np.log(examples['factor_prices'] / examples['factor_prices'].shift())
    asset_returns = examples['asset_returns_dict']['ME']
    expected = 12 * (asset_returns - factor_returns @ snapshot.y_betas.T)
    actual = snapshot.residuals
    np.testing.assert_allclose(actual.loc[expected.index[2:]], expected.iloc[2:],
                               rtol=1e-11, atol=1e-14)
    assert snapshot.y_variances['insample_alpha'].abs().max() > 1e-5
    assert snapshot.y_variances['r2'].between(0, 1).all()


def test_adapter_factor_exposures_equal_weighted_loadings(examples):
    """QIS exposure reporting uses the same dated beta matrix, including nonuniform weights."""
    weights = pd.Series([0.2, 0.3, 0.5], index=examples['asset_prices'].columns)
    for date, snapshot in examples['rolling'].data.items():
        actual = examples['risk_model'].compute_exposures_at_date(weights, date=date)
        expected = snapshot.y_betas.mul(weights, axis=0).sum()
        pd.testing.assert_series_equal(actual, expected, check_names=False, rtol=1e-12)


def test_load_cli_is_csv_only_in_a_fresh_process(examples, root):
    """Deny yfinance import and socket connections; the real load CLI must still complete."""
    guard = r"""
import builtins, runpy, socket, sys
def audit(event, args):
    if event == 'import' and args[0].split('.')[0] == 'yfinance':
        raise AssertionError('CSV-only load imported yfinance')
    if event in ('socket.connect', 'socket.getaddrinfo'):
        raise AssertionError('CSV-only load attempted network access')
sys.addaudithook(audit)
sys.path.insert(0, sys.argv[1])
sys.argv = ['rolling_factor_covar_from_csv', 'load', '--data-dir', sys.argv[2]]
runpy.run_module('examples.covar_estimation.rolling_factor_covar_from_csv', run_name='__main__')
"""
    result = subprocess.run([sys.executable, '-c', guard, str(root), str(examples['data_dir'])],
                            cwd=root, text=True, capture_output=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'Rolling snapshots: 4' in result.stdout
    assert 'Latest snapshot: 2022-12-31' in result.stdout
    assert 'Maximum covariance reconstruction error:' in result.stdout
    assert _hashes(examples['data_dir']) == examples['bundle_hashes']


def test_future_csv_inputs_do_not_change_prior_risk_snapshots(examples, tmp_path):
    """Perturb factors, assets, spots and rates beyond 2020 and compare earlier fitted outputs."""
    directory = _copy_bundle(examples, tmp_path)
    cutoff = '2020-12-31'
    for filename in ['futures_risk_factors.csv', 'asset_prices.csv',
                     'fx_hedging_data_fx_spots.csv', 'fx_hedging_data_domestic_rates.csv']:
        path = directory / filename
        frame = pd.read_csv(path, index_col=0)
        mask = frame.index > cutoff
        column = 'CHF' if 'fx_spots' in filename else frame.columns[0]
        frame.loc[mask, column] *= np.linspace(1.1, 2.0, mask.sum())
        frame.to_csv(path)
    changed, _ = examples['csv_example'].fit_rolling_risk_model_from_csv(directory)
    for date, snapshot in examples['rolling'].data.items():
        if date <= pd.Timestamp(cutoff):
            pd.testing.assert_frame_equal(snapshot.y_betas, changed[date].y_betas,
                                          rtol=1e-10, atol=1e-13)
            pd.testing.assert_frame_equal(snapshot.get_y_covar(), changed[date].get_y_covar(),
                                          rtol=1e-10, atol=1e-13)
    assert not np.allclose(changed.get_latest().get_y_covar(),
                           examples['rolling'].get_latest().get_y_covar())


@pytest.mark.parametrize('defect,message', [
    ('missing_file', 'incomplete'), ('factor_order', 'ordered factor_names'),
    ('duplicate_dates', 'sorted and unique'), ('unsorted_factors', 'sorted and unique'),
    ('nonpositive_nav', 'positive NAVs'), ('nonfinite_asset', 'finite numeric'),
    ('missing_metadata', 'same assets'), ('hedge_bounds', 'hedge_ratio'),
    ('bad_frequency', 'return_frequency'), ('missing_currency', 'currencies'),
    ('bad_boolean', 'boolean'), ('simple_returns', 'log factor returns'),
    ('missing_setting', 'missing'), ('future_end', 'estimation_end'),
])
def test_loader_rejects_documented_invalid_bundles(examples, tmp_path, defect, message):
    """Break one part of the documented input contract and require an actionable load error."""
    directory = _copy_bundle(examples, tmp_path)
    if defect == 'missing_file':
        (directory / 'asset_prices.csv').unlink()
    elif defect == 'factor_order':
        _rewrite(directory, 'futures_risk_factors.csv', lambda f: f.iloc[:, ::-1])
    elif defect == 'duplicate_dates':
        _rewrite(directory, 'asset_prices.csv', lambda f: pd.concat([f, f.tail(1)]))
    elif defect == 'unsorted_factors':
        _rewrite(directory, 'futures_risk_factors.csv', lambda f: f.iloc[::-1])
    elif defect == 'nonpositive_nav':
        _rewrite(directory, 'futures_risk_factors.csv', lambda f: f.assign(Equity=0.0))
    elif defect == 'nonfinite_asset':
        _rewrite(directory, 'asset_prices.csv', lambda f: f.assign(Growth=np.inf))
    elif defect == 'missing_metadata':
        _rewrite(directory, 'asset_metadata.csv', lambda f: f.iloc[1:])
    elif defect == 'hedge_bounds':
        _rewrite(directory, 'asset_metadata.csv', lambda f: f.assign(hedge_ratio=1.1))
    elif defect == 'bad_frequency':
        _rewrite(directory, 'asset_metadata.csv', lambda f: f.assign(return_frequency='INVALID'))
    elif defect == 'missing_currency':
        _rewrite(directory, 'asset_metadata.csv', lambda f: f.assign(currency='EUR'))
    elif defect == 'missing_setting':
        _rewrite(directory, 'risk_model_settings.csv', lambda f: f.drop(index='solver'))
    else:
        setting, value = {
            'bad_boolean': ('demean', 'maybe'), 'simple_returns': ('is_log_returns', 'False'),
            'future_end': ('estimation_end', '2023-12-31'),
        }[defect]
        path = directory / 'risk_model_settings.csv'
        frame = pd.read_csv(path, index_col=0, dtype=str)
        frame.at[setting, 'value'] = value
        frame.to_csv(path)
    with pytest.raises((ValueError, FileNotFoundError), match=message):
        examples['csv_example'].load_inputs_from_csv(directory)


def test_loader_repairs_and_unchecked_fields_match_stated_limits(examples, tmp_path):
    """Characterize sorting, FX filling, count truncation and deferred model validation."""
    directory = _copy_bundle(examples, tmp_path)
    _rewrite(directory, 'asset_prices.csv', lambda f: f.iloc[::-1])
    _rewrite(directory, 'asset_metadata.csv', lambda f: f.iloc[::-1])
    _rewrite(directory, 'fx_hedging_data_domestic_rates.csv', lambda f: f.iloc[:-12])
    spot_path = directory / 'fx_hedging_data_fx_spots.csv'
    spots = pd.read_csv(spot_path, index_col=0)
    spots.iloc[10, 1] = np.nan
    spots['USD'] = 2.0
    spots.to_csv(spot_path)
    path = directory / 'risk_model_settings.csv'
    settings = pd.read_csv(path, index_col=0, dtype=str)
    settings.at['beta_span', 'value'] = '36.9'
    settings.at['lasso_model_type', 'value'] = 'UNKNOWN'
    settings.loc['extra_setting'] = 'accepted'
    settings.to_csv(path)
    (directory / 'delivery-note.txt').write_text('An additional file is permitted.')
    loaded = examples['csv_example'].load_inputs_from_csv(directory)
    assert loaded.settings.beta_span == 36
    assert loaded.asset_prices.index.is_monotonic_increasing
    assert loaded.asset_metadata.index.equals(loaded.asset_prices.columns)
    rates = loaded.fx_rates_data.domestic_rates
    assert rates.index[-1] == pd.Timestamp('2022-12-31')
    pd.testing.assert_series_equal(rates.iloc[-1], rates.iloc[-13], check_names=False)
    assert loaded.fx_rates_data.fx_spots.iloc[10, 1] == spots.iloc[9, 1]
    assert (loaded.fx_rates_data.fx_spots['USD'] == 2.0).all()
    with pytest.raises(ValueError, match='UNKNOWN'):
        examples['csv_example'].fit_rolling_risk_model_from_csv(directory)

