"""Offline independent references for signal timing, breadth and linked attribution."""
from dataclasses import replace
import hashlib
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from examples.oss_portfolio_pipeline.tests import covariance_test as covariance_checks
from examples.oss_portfolio_pipeline.backtest import BacktestConfig, run_backtest
from examples.oss_portfolio_pipeline.covariance import estimate_covariances
from examples.oss_portfolio_pipeline.diagnostics import (
    prepare_investment_diagnostics, save_investment_reports,
)
from examples.oss_portfolio_pipeline.factsheets import comparison_runs, save_native_factsheets
from examples.oss_portfolio_pipeline.pipeline import build_inputs

pytestmark = pytest.mark.factor_integration


@pytest.fixture(scope='module')
def prepared():
    """Fit real FM/FL and execute two signal variants across fourteen synthetic test months."""
    inputs, config = covariance_checks.prepared.__wrapped__()
    dates = pd.bdate_range('2020-03-02', '2021-03-03')
    rng = np.random.default_rng(736)
    returns = rng.normal(.0002, .004, (len(dates), len(inputs.source_prices.columns)))
    extra = pd.DataFrame(np.exp(returns.cumsum(axis=0)), index=dates,
                         columns=inputs.source_prices.columns) * inputs.source_prices.iloc[-1]
    extra['BIL'] = inputs.source_prices.BIL.iloc[-1] * np.exp(.00008 * np.arange(1, len(dates)+1))
    config = replace(config, end='2021-03-04')
    inputs = build_inputs(pd.concat([inputs.source_prices, extra]), config)
    risk = estimate_covariances(inputs, config, solver='CLARABEL')
    settings = BacktestConfig(solver='CLARABEL', group_band=.02)
    full = run_backtest(inputs, risk, config, settings)
    longer = run_backtest(inputs, risk, config, replace(settings, momentum_span=24))
    return inputs, risk, full, longer


@pytest.fixture(scope='module')
def study(prepared):
    """Compute QIS diagnostics once without downloading or changing the completed portfolios."""
    inputs, risk, full, longer = prepared
    return prepare_investment_diagnostics(inputs, risk, full, [longer])


def test_signal_pairs_against_price_endpoints(prepared, study):
    """Recover every multi-month forward return and prior signal from observed price endpoints."""
    inputs, _, full, longer = prepared
    for run in (full, longer):
        for kind, scores in (('raw', run.raw_alpha), ('scored', run.alpha_scores)):
            for horizon in (1, 3, 6):
                key = f'signal_{run.settings.momentum_span}m_{kind}_pairs_{horizon}m'
                pairs = study.tables[key]
                dates = pairs.date.drop_duplicates()
                assert dates.iloc[0] == pd.Timestamp('2020-01-31')
                assert np.all(np.diff(dates.dt.to_period('M').astype('int64')) == horizon)
                for row in pairs.itertuples():
                    start = inputs.prices.index.asof(row.date - pd.offsets.MonthEnd(1))
                    end = inputs.prices.index.asof(row.forward_window_end)
                    expected = np.log(inputs.prices.loc[end, row.asset]
                                      / inputs.prices.loc[start, row.asset])
                    expected -= np.log(inputs.prices.loc[end, 'BIL']
                                       / inputs.prices.loc[start, 'BIL'])
                    assert row.r == pytest.approx(expected, abs=1e-12)
                    assert row.signal_observation_date == start
                    assert row.z == scores.loc[start, row.asset]
                    assert end <= full.executed.targets.index[-1]
    assert pd.Timestamp('2020-02-28') in study.tables[
        'signal_12m_scored_pairs_1m'].signal_observation_date.values


def test_rank_ic_reference(study):
    """Recompute rank correlations directly and the annualisation from non-overlapping dates."""
    for name, result in study.signals.items():
        for horizon, pairs in result.pairs.items():
            ic = []
            for _, group in pairs.groupby('date'):
                ic.append(np.corrcoef(group.z.rank(), group.r.rank())[0, 1])
            expected = np.mean(ic) / np.std(ic, ddof=1) * np.sqrt(12 / int(horizon))
            row = study.ic_summary.loc[(name, horizon)]
            assert row.n_dates == len(ic)
            assert row.mean_IC == pytest.approx(np.mean(ic), abs=1e-12)
            assert row.IC_IR_an == pytest.approx(expected, abs=1e-12)


def test_completed_pairs_are_prefix_invariant(prepared, study):
    """A shorter study keeps all fully realised pairs; later returns never enter early windows."""
    inputs, risk, full, longer = prepared
    cutoff = pd.Timestamp('2020-08-31')
    shortened = replace(full, executed=replace(
        full.executed, targets=full.executed.targets.loc[:cutoff]))
    earlier = prepare_investment_diagnostics(inputs, risk, shortened)
    for kind in ('raw', 'scored'):
        for horizon in (1, 3, 6):
            key = f'signal_12m_{kind}_pairs_{horizon}m'
            expected = study.tables[key]
            expected = expected.loc[expected.forward_window_end <= cutoff].reset_index(drop=True)
            pd.testing.assert_frame_equal(earlier.tables[key], expected)


def test_breadth_against_full_covariance(prepared, study):
    """Recover effective capital/risk counts with direct full-covariance Euler calculations."""
    inputs, risk, full, _ = prepared
    for name, result in study.breadth.items():
        assert result.covariance_source == 'provided'
        assert 'BIL' not in result.absolute_weight_shares
        np.testing.assert_array_equal(result.covariance_dates, full.executed.targets.index)
        for date in full.executed.targets.index:
            w = (full.executed.targets.loc[date] if name == 'Active'
                 else inputs.benchmark_weights).drop('BIL')
            w = w.where(w.abs() > 1e-4, 0.)
            sigma = risk.covariances[date].loc[w.index, w.index]
            capital = w.abs() / w.abs().sum()
            absolute_euler = np.abs(w.to_numpy() * (sigma.to_numpy() @ w.to_numpy()))
            shares = absolute_euler / absolute_euler.sum()
            row = result.metrics.loc[date]
            assert row['Effective capital assets'] == pytest.approx(1 / (capital**2).sum())
            assert row['Effective risk contributors'] == pytest.approx(1 / (shares**2).sum())
            assert row['Invested assets'] == w.ne(0).sum()
            assert row['Capital utilisation'] == pytest.approx(
                row['Selection coverage'] * row['Sizing evenness'])


def test_brinson_independent_linking_and_costs(prepared, study):
    """Verify Frongello by a separate recurrence and the native unit/cost return identity."""
    inputs, _, full, _ = prepared
    for basis, a, b in (('Net', full.executed.portfolio, full.benchmark),
                         ('Gross', full.gross_portfolio, full.gross_benchmark)):
        contributions, weights = [], []
        for p in (a, b):
            pnl = p.units.shift(1) * p.prices.diff()
            if basis == 'Net':
                pnl -= p.realized_costs
            contributions.append(pnl.div(p.nav.shift(1), axis=0).iloc[1:])
            weights.append(p.weights.shift(1).iloc[1:])
        pa, pb = contributions
        wa, wb = weights
        linked = np.zeros((3, inputs.universe.economic_group.nunique()))
        wealth = 1.
        for date in pa.index:
            grouped_a = pa.loc[date].groupby(inputs.universe.economic_group).sum()
            grouped_b = pb.loc[date].groupby(inputs.universe.economic_group).sum()
            ga = wa.loc[date].groupby(inputs.universe.economic_group).sum()
            gb = wb.loc[date].groupby(inputs.universe.economic_group).sum()
            ra = grouped_a.div(ga).where(ga != 0, 0.)
            rb = grouped_b.div(gb).where(gb != 0, 0.)
            allocation = (ga-gb) * rb
            selection = gb * (ra-rb)
            interaction = grouped_a-grouped_b-allocation-selection
            linked = linked * (1+pb.loc[date].sum()) + wealth * np.array(
                [allocation, selection, interaction])
            wealth *= 1+pa.loc[date].sum()
        if basis == 'Net':
            linked[1] += linked[2]
            linked[2] = 0.
        for i, effect in enumerate(study.brinson[basis][2:]):
            np.testing.assert_allclose(effect[ga.index].sum(), linked[i], atol=1e-10)
        nav_active = a.nav.iloc[-1]/a.nav.iloc[0] - b.nav.iloc[-1]/b.nav.iloc[0]
        assert linked.sum() == pytest.approx(nav_active, abs=1e-10)
        assert study.tables[f'brinson_{basis.lower()}_reconciliation'][
            'Reconciliation error'].abs().max() < 1e-10


def test_comparison_guardrails(prepared):
    """Reject incomparable histories and ambiguous named configurations."""
    _, _, full, longer = prepared
    with pytest.raises(ValueError, match='NAV observations'):
        comparison_runs(full, [replace(longer, navs=longer.navs.iloc[1:])])
    with pytest.raises(ValueError, match='Ambiguous'):
        comparison_runs(full, [longer, longer])


def test_native_and_investment_report_exports(prepared, study, tmp_path):
    """Exercise all six PDF generators and verify hashes, regimes, inputs and figure ownership."""
    inputs, _, full, longer = prepared
    for name in ('backtest_manifest.json', 'covariance_manifest.json'):
        (tmp_path/name).write_text('{}')
    prior = plt.figure()
    nav_before = full.executed.portfolio.nav.copy()
    targets_before = full.executed.targets.copy()
    paths = save_native_factsheets(inputs, full, tmp_path, [longer])
    paths.update(save_investment_reports(study, tmp_path))
    assert len(paths) == 6
    assert plt.get_fignums() == [prior.number]
    plt.close(prior)
    for path in paths.values():
        assert path.read_bytes().startswith(b'%PDF-')
    for directory in {p.parent for p in paths.values()}:
        manifest = json.loads((directory/'manifest.json').read_text())
        for name, digest in manifest['file_sha256'].items():
            assert hashlib.sha256((directory/name).read_bytes()).hexdigest() == digest
        for name, digest in manifest['source_sha256'].items():
            assert hashlib.sha256((tmp_path/name).read_bytes()).hexdigest() == digest
    native = json.loads((paths['strategy_factsheet'].parent/'manifest.json').read_text())
    assert 'ME benchmark' in native['regimes']
    assert len(native['strategies']) == 2
    pd.testing.assert_series_equal(full.executed.portfolio.nav, nav_before)
    pd.testing.assert_frame_equal(full.executed.targets, targets_before)
