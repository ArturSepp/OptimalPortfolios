"""Verify the six real covariance estimators and the complete offline analytics bundle."""

from copy import deepcopy
import importlib
import json

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope='module')
def producer(root):
    """Import repository-only tooling after installed-wheel checks have skipped."""
    module = importlib.import_module('tools.docs_analytics.covariance_comparison')
    assert module.reports.ROOT.resolve() == root.resolve()
    return module


@pytest.fixture(scope='module')
def analytics(producer):
    """Compute the six real cases with network access disabled."""
    with producer.reports.offline():
        return producer.build_analytics(producer.configuration())


def test_configuration_inventory_and_native_grid(root, producer, analytics):
    """Retain 32 actual factor fits and 48 allocations on eight identical dates."""
    config = producer.reports.load_registry(root)['producers'][producer.NAME]['configuration']
    assert config == producer.configuration() == analytics['configuration']
    assert set(config['tables']) == set(analytics['tables'])
    assert len(config['tables']) == 23
    assert all(analytics['checks'].values())
    records = analytics['diagnostics']['solves']
    assert len(records) == len({item['context'] for item in records}) == 80
    assert sum(item['stage'] == 'factor_fit' for item in records) == 32
    assert sum(item['stage'] == 'allocation' for item in records) == 48
    assert all(item['accepted'] and item['compliant'] and item['fallback_source'] is None
               and item['solver'] == 'CLARABEL' and item['raw_status'] == 'optimal'
               for item in records)
    dates = analytics['tables']['decision_schedule'].index
    assert len(dates) == 8
    assert dates[[0, -1]].tolist() == [pd.Timestamp('2024-01-03'), pd.Timestamp('2025-10-01')]


@pytest.mark.parametrize('section,key,value', [
    ('parameters', 'span', 26),
    ('parameters', 'warmup', 10),
    ('parameters', 'cost_rate', 0),
    ('parameters', 'objective', 'MAX_DIVERSIFICATION'),
    ('parameters', 'factorize_covar', True),
    ('parameters', 'group_assignments', [0] * 8),
    ('parameters', 'group_penalty', 'unnormalized'),
    ('conventions', 'seed', 1),
    ('conventions', 'returns', 'simple'),
    ('conventions', 'implementation_lag', 0),
])
def test_configuration_cannot_be_ignored(producer, section, key, value):
    """Reject unsupported simulation, estimation and execution changes before producing output."""
    config = producer.configuration()
    config[section][key] = value
    with pytest.raises(ValueError, match='Unsupported covariance-comparison configuration'):
        producer.build_analytics(config)


def test_fixture_preserves_numpy_state_and_price_conversion(producer, analytics):
    """The legacy generator cannot reset the caller's RNG; initial price replaces one increment."""
    before = np.random.get_state()
    tables = producer._fixture(producer.configuration())
    after = np.random.get_state()
    assert before[0] == after[0] and before[2:] == after[2:]
    np.testing.assert_array_equal(before[1], after[1])
    pd.testing.assert_frame_equal(tables['prices'], analytics['tables']['prices'])
    for prices, returns in [('prices', 'simulated_asset_returns'),
                            ('factor_prices', 'simulated_factor_returns')]:
        np.testing.assert_allclose(tables[prices].iloc[0], 100)
        expected = 100 * np.exp(tables[returns].iloc[1:].cumsum())
        np.testing.assert_allclose(tables[prices].iloc[1:], expected, rtol=1e-12)


def test_known_covariance_uses_daily_260_scale(analytics):
    """Reconstruct the annual covariance using recorded loadings and daily idiosyncratic scales."""
    tables = analytics['tables']
    beta = tables['true_betas'].to_numpy()
    systematic = np.einsum('ki,kl,lj->ij', beta, tables['true_factor_covariance'], beta)
    residual = np.diag(np.square(tables['true_idio_volatility']['daily_volatility']) * 260)
    np.testing.assert_allclose(tables['true_asset_covariance'], systematic + residual, rtol=1e-12)


@pytest.mark.parametrize('label', ['EWMA', 'EWMA vol norm'])
def test_ewma_annualization_against_explicit_weekly_kernels(analytics, label):
    """Rebuild demeaning and each covariance kernel outside the OptimalPortfolios estimator."""
    import qis

    date = analytics['tables']['decision_schedule'].index[-1]
    prices = analytics['tables']['prices'].loc[:date]
    weekly = qis.to_returns(prices, freq='W-WED', is_log_returns=True,
                            is_first_zero=False, drop_first=True)
    demeaned = (weekly - qis.compute_ewm(weekly, span=52)).iloc[1:].to_numpy()
    if label == 'EWMA':
        tensor = qis.compute_ewm_covar_tensor(
            a=demeaned, span=52, nan_backfill=qis.NanBackfill.ZERO_FILL)
    else:
        tensor, _, _ = qis.compute_ewm_covar_tensor_vol_norm_returns(
            a=demeaned, span=52, nan_backfill=qis.NanBackfill.ZERO_FILL)
    np.testing.assert_allclose(
        analytics['estimates'][label]['covars'][date], 52 * tensor[-1], rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize('label', [
    'Lasso', 'Lasso factor vol norm', 'Group Lasso', 'Group Lasso factor vol norm',
])
def test_factor_components_and_residual_scale_against_direct_fit(producer, analytics, label):
    """Check beta/factor covariance assembly and annual residual variances against a raw fit."""
    import factorlasso as fl
    import qis

    tables = analytics['tables']
    date = tables['decision_schedule'].index[-1]
    components = analytics['estimates'][label]['components']
    beta = components['estimated_betas'].xs(date).to_numpy()
    matrix = components['estimated_factor_covariances'].xs(date).to_numpy()
    residual = components['estimated_residual_variances'].xs(date).iloc[:, 0].to_numpy()
    expected = np.einsum('ik,kl,jl->ij', beta, matrix, beta) + np.diag(residual)
    np.testing.assert_allclose(
        analytics['estimates'][label]['covars'][date], expected, rtol=1e-12, atol=1e-14)
    asset_returns = qis.compute_asset_returns_dict(
        prices=tables['prices'], is_log_returns=True, returns_freqs='W-WED')['W-WED'].loc[:date]
    factors = tables['factor_prices'].loc[:date].reindex(asset_returns.index, method='ffill')
    x = qis.to_returns(factors, is_log_returns=True, is_first_zero=False,
                       drop_first=False, freq=None)
    case = producer.CASES[label]
    model = fl.LassoModel(
        model_type=getattr(fl.LassoModelType, case['model']),
        group_data=tables['group_assignments']['group'], reg_lambda=case['reg_lambda'],
        span=52, warmup_period=52, solver='CLARABEL')
    model.fit(x=x, y=asset_returns, verbose=False)
    np.testing.assert_allclose(beta, model.estimated_betas, rtol=1e-12, atol=1e-14)
    np.testing.assert_allclose(residual, 52 * model.estimation_result_.ss_res, rtol=1e-12)


@pytest.mark.parametrize('label', [
    'EWMA', 'EWMA vol norm', 'Lasso', 'Lasso factor vol norm',
    'Group Lasso', 'Group Lasso factor vol norm',
])
def test_estimates_ignore_future_prices_and_known_truth(producer, analytics, label):
    """Future observations and altered truth cannot affect earlier estimates."""
    tables = deepcopy({name: analytics['tables'][name]
                       for name in producer.configuration()['input_tables']})
    cutoff = pd.Timestamp('2024-12-16')
    for name in ('prices', 'factor_prices'):
        panel = tables[name]
        future = panel.index > cutoff
        panel.loc[future] *= np.linspace(0.2, 5, future.sum())[:, None]
    tables['true_betas'] *= 100
    tables['true_asset_covariance'] *= 100
    dates = analytics['tables']['decision_schedule'].index
    dates = list(dates[dates <= cutoff])
    result = producer._estimate_case(
        label, producer.CASES[label], tables, dates, producer.configuration())
    for date, matrix in result['covars'].items():
        pd.testing.assert_frame_equal(
            matrix, analytics['estimates'][label]['covars'][date],
            check_names=False, rtol=1e-12, atol=1e-14)


def test_costs_against_fixed_units_between_trades(analytics):
    """Validate entry NAV, static units before the next trade, caps and costs for every case."""
    tables = analytics['tables']
    first, second = tables['decision_schedule']['implementation_date'].iloc[:2]
    for item in analytics['portfolios'].values():
        portfolio, weights = item['portfolio'], item['tables']['target_weights']
        np.testing.assert_allclose(weights.sum(axis=1), 1, atol=1e-10, rtol=0)
        assert weights.min().min() >= -1e-8 and weights.max().max() <= 0.35 + 1e-8
        units = 100 * weights.iloc[0] / portfolio.prices.loc[first]
        holding_dates = portfolio.prices.loc[first:second].index[:-1]
        expected = portfolio.prices.loc[holding_dates].dot(units) - 0.1
        np.testing.assert_allclose(portfolio.nav.loc[holding_dates], expected, rtol=1e-12)
        cost = portfolio.units.diff().fillna(0).abs() * portfolio.prices * 0.001
        np.testing.assert_allclose(portfolio.realized_costs, cost, rtol=1e-12, atol=1e-12)


def test_cvxpy_recorder_restores_method_on_success_and_exception(producer):
    """Scoped instrumentation must not leak into later estimators or portfolio construction."""
    import cvxpy as cp

    original = cp.Problem.solve
    with producer._capture_factor_solve('test') as records:
        x = cp.Variable()
        problem = cp.Problem(cp.Minimize(cp.square(x - 1)))
        problem.solve(solver='CLARABEL', verbose=False)
    assert cp.Problem.solve is original
    assert len(records) == 1 and records[0]['accepted']
    with pytest.raises(RuntimeError, match='injected'):
        with producer._capture_factor_solve('failure'):
            raise RuntimeError('injected')
    assert cp.Problem.solve is original


@pytest.mark.parametrize('defect', [
    'missing_fit', 'missing_allocation', 'fallback_fit', 'covariance_reuse', 'wrong_dates',
])
def test_bad_estimates_and_solver_evidence_fail_closed(producer, analytics, monkeypatch, defect):
    """Corrupt cached real results to exercise the comparison guard without changing solvers."""
    def estimate(label, case, tables, dates, config):
        """Return recorded estimates with one explicit provenance or estimator defect."""
        item = deepcopy(analytics['estimates'][label])
        if label != 'Group Lasso factor vol norm':
            return item
        if defect == 'missing_fit':
            item['solves'].pop()
        elif defect == 'fallback_fit':
            item['solves'][0]['fallback_source'] = 'secondary_solver'
        elif defect == 'covariance_reuse':
            item['covars'] = deepcopy(analytics['estimates']['EWMA']['covars'])
        elif defect == 'wrong_dates':
            date = next(iter(item['covars']))
            item['covars'][date - pd.Timedelta(days=1)] = item['covars'].pop(date)
        return item

    def portfolio(label, covars, prices, config):
        """Preserve real backtests while selectively removing one allocation record."""
        item = deepcopy(analytics['portfolios'][label])
        if defect == 'missing_allocation' and label == 'EWMA':
            item['solves'].pop()
        return item

    monkeypatch.setattr(producer, '_estimate_case', estimate)
    monkeypatch.setattr(producer, '_portfolio', portfolio)
    with pytest.raises(ValueError):
        producer.build_analytics(producer.configuration())


def test_complete_real_bundle_repeats_and_validates(root, producer, tmp_path, monkeypatch):
    """All four real families produce identical six-image/62-table bundles without publication."""
    runner = importlib.import_module('tools.docs_analytics.run')
    validator = importlib.import_module('tools.docs_analytics.validate')
    monkeypatch.setenv('AGENT_LOCAL_ROOT', str(tmp_path))
    plan = runner.build_plan(root)
    assert plan['generation_ready'] and not plan['generation_blockers']
    published = root / 'examples/figures/analytics_manifest.json'
    published_before = published.read_bytes() if published.exists() else None
    paths = [runner.generate(tmp_path / name, root) for name in ('first', 'second')]
    manifests = [json.loads((path / 'analytics_manifest.json').read_text(encoding='utf-8'))
                 for path in paths]
    a, b = manifests
    assert len(a['outputs']) == 68
    assert a['outputs'] == b['outputs']
    assert a['source'] == b['source']
    assert a['producers'] == b['producers']
    assert a['environment'] == b['environment']
    assert not a['publication_ready'] and a['review_status'] == 'pending'
    assert sum(len(item['diagnostics']['solves']) for item in a['producers'].values()) == 467
    for path in paths:
        validator.validate_bundle(path, root)
    assert (published.read_bytes() if published.exists() else None) == published_before
