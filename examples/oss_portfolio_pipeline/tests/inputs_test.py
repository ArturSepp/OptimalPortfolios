"""Offline composed checks for the public example's inputs.

Run with pytest against this file. Synthetic held-unit and EWMA references deliberately
compute the accounting independently of FM/QIS so a plausible numerical regression fails.
"""
from dataclasses import replace
import hashlib
import json
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from examples.oss_portfolio_pipeline.config import (
    PipelineConfig, all_tickers, benchmark_weights, universe_table,
)
from examples.oss_portfolio_pipeline.data import download_prices
from examples.oss_portfolio_pipeline.factors import (
    build_factors, factor_model, month_end_dates,
)
from examples.oss_portfolio_pipeline.pipeline import build_inputs, save_inputs

pytestmark = pytest.mark.factor_integration


@pytest.fixture(scope='module')
def config():
    """Specify a completed synthetic backtest interval with ample construction prehistory."""
    return PipelineConfig(end='2020-07-01')


@pytest.fixture(scope='module')
def prices():
    """Create distinct, positive deterministic ETF paths and a modest bill return."""
    dates = pd.bdate_range('2014-12-01', '2020-06-30')
    x = np.arange(len(dates), dtype=float)
    columns = all_tickers()
    shocks = np.column_stack([
        .00015 + .004 * np.sin(.23 * x + .17 * i)
        + .0015 * np.cos(.07 * x + .7 * i)
        for i in range(len(columns))
    ])
    levels = 100 * np.exp(np.cumsum(shocks, axis=0))
    frame = pd.DataFrame(levels, index=dates.rename('Date'), columns=columns)
    frame['BIL'] = 100 * np.exp(.00008 * x + .000005 * np.sin(.1 * x))
    return frame


@pytest.fixture(scope='module')
def factors(prices, config):
    """Build the common full-history factor panel once for accounting assertions."""
    return build_factors(prices, config)


@pytest.fixture(scope='module')
def inputs(prices, config):
    """Assemble the full example from synthetic inputs without any Yahoo request."""
    return build_inputs(prices, config)


def test_agreed_universe_benchmark_and_financed_factor_definitions(config):
    """Freeze the approved 36 assets, 12 strategic holdings and 11 funded risk baskets."""
    table = universe_table()
    strategic = benchmark_weights()
    assert len(all_tickers()) == len(set(all_tickers())) == 40
    assert len(table) == len(strategic) == 36
    assert strategic.index.equals(table.index)
    assert (strategic > 0).sum() == 12
    assert strategic.sum() == pytest.approx(1.)
    expected_benchmark = {
        'SPY': .25, 'EFA': .15, 'EEM': .10, 'GOVT': .15,
        'IGOV': .05, 'TIP': .05, 'LQD': .05, 'HYG': .03,
        'EMB': .02, 'GLD': .05, 'DBC': .05, 'BIL': .05,
    }
    assert strategic[strategic > 0].to_dict() == expected_benchmark
    assert table.loc[['VNQ', 'VNQI', 'IGF'], 'economic_group'].eq('Equities').all()
    expected = {
        'Energy': (.15, {'DBE': 1., 'BIL': -1.}),
        'Agriculture': (.15, {'DBA': 1., 'BIL': -1.}),
        'Base metals': (.15, {'DBB': 1., 'BIL': -1.}),
        'Precious metals': (.15, {'DBP': 1., 'BIL': -1.}),
        'DM equity': (.15, {'URTH': 1., 'BIL': -1.}),
        'EM vs DM equity': (.15, {'EEM': 1., 'URTH': -1.}),
        'Nominal vs TIPS': (.075, {'GOVT': 1., 'TIP': -1.}),
        'Global government bonds': (.075, {'GOVT': .5, 'IGOV': .5, 'BIL': -1.}),
        'USD vs DM': (.075, {'UUP': 1., 'BIL': -1.}),
        'EM FX stress vs DM': (.075, {'BIL': 2., 'CEW': -1., 'UUP': -1.}),
        'HY vs IG credit': (.05, {'HYG': 1., 'LQD': -1.}),
    }
    model = factor_model(config)
    assert [member.value for member in model.factors] == list(expected)
    for factor in model.baskets.values():
        target, basket = expected[factor.name]
        assert factor.resolved_vol_target == target
        assert dict(zip(factor.signed_tickers(), factor.signed_weights())) == basket
        assert sum(factor.signed_weights()) == pytest.approx(0.)
        assert factor.require_all_constituents


@pytest.mark.parametrize(('dates', 'end', 'expected'), [
    (['2021-01-28', '2021-01-29', '2021-02-01', '2021-02-02'],
     '2021-02-03', ['2021-01-29']),
    (['2024-03-27', '2024-03-28', '2024-04-01', '2024-04-02',
      '2024-04-29', '2024-04-30', '2024-05-01', '2024-05-08'],
     '2024-05-09', ['2024-03-28', '2024-04-30']),
    (['2017-12-27', '2017-12-28', '2017-12-29'],
     '2018-01-01', ['2017-12-29']),
])
def test_completed_month_ends_use_last_session_not_next_month(dates, end, expected):
    """Weekend/Good-Friday anchors trade backward and a partial final month adds no trade."""
    actual = month_end_dates(pd.DatetimeIndex(dates), end)
    pd.testing.assert_index_equal(actual, pd.DatetimeIndex(expected))


@pytest.mark.parametrize('name', ['Energy', 'EM FX stress vs DM'])
def test_factor_holding_period_matches_independent_units_and_cash(prices, factors, name):
    """Financed factors earn signed simple P&L while units remain fixed within the month."""
    targets = factors.target_weights[name]
    active = targets.index[targets.abs().sum(axis=1) > 0]
    entry, next_rebalance = active[:2]
    legs = prices.loc[entry:next_rebalance, targets.columns]
    weights = targets.loc[entry].to_numpy()
    relative_prices = legs.div(legs.iloc[0])
    # QIS owns production accounting; this endpoint identity is the independent unit oracle.
    expected = 1. + (relative_prices - 1.).to_numpy() @ weights
    actual = factors.navs.loc[entry:next_rebalance, name]
    np.testing.assert_allclose(actual / actual.iloc[0], expected, rtol=2e-12, atol=2e-12)
    wrong_log_basket = np.exp(np.log(relative_prices.to_numpy()) @ weights)
    assert np.max(np.abs(expected - wrong_log_basket)) > 1e-5


def test_first_monthly_leverage_matches_independent_weekly_ewma(prices, config, factors):
    """The signed excess basket, including BIL, determines annualised target leverage."""
    targets = factors.target_weights['Energy']
    entry = targets.index[targets.abs().sum(axis=1).gt(0)][0]
    Wednesdays = pd.date_range(prices.index[0], entry, freq='W-WED')
    weekly_prices = prices[['DBE', 'BIL']].reindex(Wednesdays, method='ffill')
    # Independent sample-plus-recursion reference detects cash omission or wrong annualisation.
    weekly = np.log(weekly_prices).diff().dropna()
    basket = weekly['DBE'].to_numpy() - weekly['BIL'].to_numpy()
    warmup = config.vol_warmup_period
    variance = np.var(basket[:warmup], ddof=1)
    decay = 1. - 2. / (config.vol_span + 1.)
    for value in basket[warmup:]:
        variance = decay * variance + (1. - decay) * value ** 2
    leverage = .15 / np.sqrt(52. * variance)
    np.testing.assert_allclose(targets.loc[entry, ['DBE', 'BIL']],
                               [leverage, -leverage], rtol=2e-11, atol=2e-11)


def test_volatility_warmup_has_no_exposure_or_cash_charge(prices, config, factors):
    """No factor earns or pays returns before its thirteen-week risk sample is known."""
    Wednesday_dates = pd.date_range(prices.index[0], prices.index[-1], freq='W-WED')
    seed_date = Wednesday_dates[config.vol_warmup_period]
    history = factors.navs.loc[:seed_date]
    np.testing.assert_allclose(history, np.tile(history.iloc[0], (len(history), 1)))
    for targets in factors.target_weights.values():
        assert targets.loc[targets.index < seed_date].eq(0.).all().all()
    assert factors.log_returns.iloc[0].isna().all()
    pd.testing.assert_frame_equal(factors.log_returns, np.log(factors.navs).diff())


def test_future_prices_and_cash_cannot_change_past_factors(prices, config, factors):
    """Future ETF and cash changes cannot leak into risk targets or the historical NAV."""
    cutoff = pd.Timestamp('2019-12-31')
    changed = prices.copy()
    future = changed.index > cutoff
    changed.loc[future, 'BIL'] *= 1.005 + .00001 * np.arange(future.sum())
    changed.loc[future, 'DBE'] *= 1.02
    changed_factors = build_factors(changed, config)
    prefix = build_factors(prices.loc[:cutoff], replace(config, end='2020-01-01'))
    pd.testing.assert_frame_equal(changed_factors.navs.loc[:cutoff], factors.navs.loc[:cutoff])
    pd.testing.assert_frame_equal(prefix.navs, factors.navs.loc[:cutoff])
    for name, expected in factors.target_weights.items():
        pd.testing.assert_frame_equal(changed_factors.target_weights[name].loc[:cutoff],
                                      expected.loc[:cutoff])
        pd.testing.assert_frame_equal(prefix.target_weights[name], expected.loc[:cutoff])


def test_benchmark_holds_units_and_preserves_the_requested_start(inputs):
    """The static strategy resets monthly instead of silently rebalancing every day."""
    assert inputs.benchmark_targets.index[0] == pd.Timestamp('2019-12-31')
    assert pd.Timestamp('2020-02-28') in inputs.benchmark_targets.index
    assert pd.Timestamp('2020-03-02') not in inputs.benchmark_targets.index
    expected_targets = np.tile(inputs.benchmark_weights, (len(inputs.benchmark_targets), 1))
    np.testing.assert_allclose(inputs.benchmark_targets, expected_targets)
    first, following = inputs.benchmark_targets.index[:2]
    prices = inputs.prices.loc[first:following]
    relatives = prices.div(prices.iloc[0])
    expected = relatives.to_numpy() @ inputs.benchmark_weights.to_numpy()
    nav = inputs.benchmark.get_portfolio_nav().loc[first:following]
    np.testing.assert_allclose(nav / nav.iloc[0], expected, rtol=2e-12, atol=2e-12)
    wrong_daily = (1. + prices.pct_change(fill_method=None).fillna(0.)
                   @ inputs.benchmark_weights).cumprod()
    assert np.max(np.abs(expected - wrong_daily.to_numpy())) > 1e-5
    assert inputs.factor_navs.index[0] == pd.Timestamp('2017-12-29')
    np.testing.assert_allclose(inputs.factor_navs.iloc[0], 100.)
    assert inputs.factor_returns.iloc[0].isna().all()
    pd.testing.assert_frame_equal(inputs.factor_returns, np.log(inputs.factor_navs).diff())


@pytest.mark.parametrize('bad_price', [np.nan, np.inf, 0., -1.])
def test_missing_or_invalid_cash_quotes_fail_without_filling(prices, config, bad_price):
    """A broken financing leg must reject the panel rather than alter its economic meaning."""
    broken = prices.copy()
    broken.loc[pd.Timestamp('2019-12-31'), 'BIL'] = bad_price
    with pytest.raises(ValueError, match='BIL'):
        build_inputs(broken, config)


@pytest.mark.parametrize('ticker_first', [False, True])
def test_yahoo_options_cache_roundtrip_and_checksum(monkeypatch, tmp_path, ticker_first):
    """A mocked download records adjustments, reuses exact bytes, and rejects cache tampering."""
    tickers = ('SPY', 'BIL')
    dates = pd.DatetimeIndex(['2020-01-02', '2020-01-03'], name='Date')
    adjusted = pd.DataFrame([[100.1234567890123, 99.01], [101.2345678901234, 99.02]],
                            index=dates, columns=tickers)
    raw = pd.concat({'Open': adjusted * .99, 'Close': adjusted}, axis=1)
    if ticker_first:
        raw = raw.swaplevel(axis=1)
    calls, cache_locations = [], []

    def fake_download(**kwargs):
        """Record the exact public request and return an in-memory Yahoo-shaped panel."""
        calls.append(kwargs)
        return raw.copy()

    def fake_tz_cache(path):
        """Record where yfinance would place its ancillary cache."""
        cache_locations.append(path)

    fake_yahoo = SimpleNamespace(download=fake_download, set_tz_cache_location=fake_tz_cache,
                                 __version__='offline-fixture')
    monkeypatch.setitem(sys.modules, 'yfinance', fake_yahoo)
    first = download_prices(tickers, '2020-01-01', '2020-01-06', cache_dir=tmp_path)
    pd.testing.assert_frame_equal(first, adjusted)
    request = calls[0]
    assert request['tickers'] == list(tickers)
    assert request['start'] == '2020-01-01' and request['end'] == '2020-01-06'
    assert request['auto_adjust'] is True and request['repair'] is False
    assert request['interval'] == '1d' and request['keepna'] is True
    assert request['threads'] is False and request['ignore_tz'] is True
    assert cache_locations == [str(tmp_path / 'yfinance')]
    cached = download_prices(tickers, '2020-01-01', '2020-01-06', cache_dir=tmp_path)
    pd.testing.assert_frame_equal(cached, first, check_exact=True)
    assert len(calls) == 1
    manifest = next(tmp_path.glob('*.json'))
    metadata = json.loads(manifest.read_text(encoding='utf-8'))
    assert metadata['yfinance_version'] == 'offline-fixture'
    assert first.attrs['yahoo_download'] == cached.attrs['yahoo_download'] == metadata
    csv_path = next(tmp_path.glob('*.csv'))
    csv_path.write_bytes(csv_path.read_bytes() + b'corrupt\n')
    with pytest.raises(ValueError, match='invalid Yahoo cache'):
        download_prices(tickers, '2020-01-01', '2020-01-06', cache_dir=tmp_path)
    assert len(calls) == 1
    repaired = download_prices(tickers, '2020-01-01', '2020-01-06',
                               cache_dir=tmp_path, refresh=True)
    pd.testing.assert_frame_equal(repaired, first, check_exact=True)
    assert len(calls) == 2


def test_saved_bundle_retains_all_source_prices_and_hashes(inputs, config, tmp_path):
    """A saved run can identify and rebuild its factors after the download cache changes."""
    output = save_inputs(inputs, config, tmp_path)
    manifest = json.loads((output / 'manifest.json').read_text(encoding='utf-8'))
    source = pd.read_csv(output / 'source_adjusted_prices.csv', index_col=0,
                         parse_dates=True, dtype=str).astype(float)
    # CSV preserves dates and values, but does not encode pandas' optional frequency hint.
    pd.testing.assert_frame_equal(source, inputs.source_prices, check_exact=True, check_freq=False)
    assert tuple(source.columns) == all_tickers()
    for filename, expected in manifest['file_sha256'].items():
        assert hashlib.sha256((output / filename).read_bytes()).hexdigest() == expected
    assert manifest['config']['end'] == config.end
    assert manifest['source_download'] is None  # synthetic caller-supplied panel
