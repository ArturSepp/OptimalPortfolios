"""Offline checks of report dates, executed turnover, funding cash and saved provenance."""
from dataclasses import replace
import hashlib
import json

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import pytest

from examples.oss_portfolio_pipeline import backtest_validation_local as backtest_checks
from examples.oss_portfolio_pipeline.backtest import BacktestConfig, run_backtest
from examples.oss_portfolio_pipeline.reporting import (
    calendar_month_labels, create_report_figures, prepare_report_tables, save_reports,
)


@pytest.fixture(scope='module')
def report_case():
    """Reuse real FM/FL/OP short synthetic execution with a weekend month-end."""
    prepared = backtest_checks.prepared.__wrapped__()
    result = backtest_checks.completed.__wrapped__(prepared)
    return prepared, result


def test_risk_dates_and_missing_warmup(report_case):
    """A Friday model belongs to Saturday ME and cannot backfill unobserved realised risk."""
    (inputs, risk, _), result = report_case
    observed = pd.Series([np.nan, np.nan, .021],
                         index=pd.date_range('2019-12-31', periods=3, freq='ME'))
    tables = prepare_report_tables(inputs, replace(result, realised_tracking_error=observed))
    risk_table = tables['tracking_error']
    assert risk_table.index.tolist() == observed.index.tolist()
    pd.testing.assert_series_equal(risk_table['Realised net (36m EWMA)'], observed,
                                   check_names=False)
    for date, weights in result.executed.targets.iterrows():
        active = (weights - inputs.benchmark_weights).to_numpy()
        # Independent test-only quadratic-form reference checks both units and alignment.
        expected = np.sqrt(active @ risk.covariances[date].to_numpy() @ active)
        label = date.to_period('M').to_timestamp('M')
        assert risk_table.loc[label, 'Target vs fixed benchmark'] == pytest.approx(expected)
        held = (result.executed.portfolio.weights.loc[date]
                - result.benchmark.weights.loc[date]).to_numpy()
        expected_held = np.sqrt(held @ risk.covariances[date].to_numpy() @ held)
        assert risk_table.loc[label, 'Held vs held benchmark'] == pytest.approx(expected_held)


@pytest.mark.parametrize('lag', [0, 1])
def test_executed_turnover_costs_and_partial_month(report_case, lag):
    """Compare executed unit changes, not target differences; exclude funding even with lag."""
    (inputs, risk, config), result = report_case
    if lag:
        result = run_backtest(inputs, risk, config, BacktestConfig(
            solver='CLARABEL', group_band=.02, implementation_lag=lag))
    tables = prepare_report_tables(inputs, result)
    for name, portfolio in (('Active (net)', result.executed.portfolio),
                            ('Benchmark (net)', result.benchmark)):
        expected = (portfolio.units.diff().abs() * portfolio.prices).sum(axis=1, min_count=1)
        expected /= portfolio.nav
        funded = portfolio.units.abs().sum(axis=1).gt(0.)
        expected.loc[:funded[funded].index[0]] = np.nan
        pd.testing.assert_series_equal(tables['turnover_daily'][name], expected, check_names=False)
        monthly = expected.resample('ME').sum(min_count=1).loc[:'2020-02-29']
        pd.testing.assert_series_equal(tables['turnover_monthly'][name], monthly, check_names=False)
        costs = portfolio.realized_costs.sum(axis=1) / portfolio.nav
        monthly_cost = costs.resample('ME').sum().loc[:'2020-02-29']
        pd.testing.assert_series_equal(tables['costs_monthly'][name], monthly_cost,
                                       check_names=False)
    assert tables['turnover_12m'].isna().all().all()
    assert tables['turnover_monthly'].index[-1] == pd.Timestamp('2020-02-29')
    assert tables['allocation_active'].index[-1] == pd.Timestamp('2020-03-03')


def test_allocations_and_no_input_mutation(report_case):
    """Retain all ETFs and the cash accounting identity without relabelling original objects."""
    (inputs, _, _), result = report_case
    original_name = result.executed.portfolio.ticker
    original_nav = result.executed.portfolio.nav.copy()
    original_groups = result.executed.portfolio.group_data.copy()
    tables = prepare_report_tables(inputs, result)
    for key, portfolio in (('active', result.executed.portfolio), ('benchmark', result.benchmark)):
        weights = tables[f'allocation_{key}']
        np.testing.assert_allclose(weights.sum(axis=1), 1., atol=1e-12)
        cash = portfolio.nav - (portfolio.units * portfolio.prices).sum(axis=1)
        np.testing.assert_allclose(weights['Funding cash'], cash / portfolio.nav, atol=1e-12)
    snapshot = tables['allocation_snapshot']
    assert snapshot.index.tolist() == inputs.universe.index.tolist() + ['Funding cash']
    pd.testing.assert_series_equal(snapshot['Fixed target'].iloc[:-1], inputs.benchmark_weights,
                                   check_names=False)
    assert result.executed.portfolio.ticker == original_name
    pd.testing.assert_series_equal(result.executed.portfolio.nav, original_nav)
    pd.testing.assert_series_equal(result.executed.portfolio.group_data, original_groups)


def test_calendar_collision_rejected():
    """Daily observations cannot silently become ambiguous monthly estimates."""
    values = pd.Series([1., 2.], index=pd.to_datetime(['2020-01-30', '2020-01-31']))
    with pytest.raises(ValueError, match='at most one'):
        calendar_month_labels(values)


def test_allocation_chart_retains_daily_dates(report_case):
    """A year-formatted label must not collapse many daily holdings onto the same x coordinate."""
    (inputs, _, _), result = report_case
    figures = create_report_figures(inputs, result)
    try:
        for ax in figures['allocations'].axes[:2]:
            vertices = ax.collections[0].get_paths()[0].vertices
            dates = np.unique(vertices[:, 0])
            expected = mdates.date2num(result.navs.index.to_pydatetime())
            np.testing.assert_array_equal(dates, expected)
    finally:
        for fig in figures.values():
            plt.close(fig)


def test_pdf_tables_and_manifest(report_case, tmp_path):
    """A short-history report renders missing statistics and hashes every saved page/table."""
    (inputs, _, _), result = report_case
    manifest = tmp_path / 'backtest_manifest.json'
    manifest.write_text('{"test_bundle": true}', encoding='utf-8')
    before = plt.get_fignums()
    pdf = save_reports(inputs, result, tmp_path)
    assert pdf.read_bytes().startswith(b'%PDF-')
    saved = json.loads((pdf.parent / 'report_manifest.json').read_text())
    assert len(saved['pages']) == 5
    assert saved['backtest_manifest_sha256'] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    for name, checksum in saved['file_sha256'].items():
        assert hashlib.sha256((pdf.parent / name).read_bytes()).hexdigest() == checksum
    assert plt.get_fignums() == before
    assert saved['completed_trading_month'] == '2020-02-29'
    assert len(list(pdf.parent.glob('*.png'))) == 5


def test_missing_backtest_manifest(report_case, tmp_path):
    """Reporting must link to a persisted backtest bundle."""
    (inputs, _, _), result = report_case
    with pytest.raises(ValueError, match='Save the portfolio backtest'):
        save_reports(inputs, result, tmp_path)
