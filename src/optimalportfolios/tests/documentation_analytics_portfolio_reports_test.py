"""Verify the real synthetic portfolio family, independently of controlled bundle producers."""

from dataclasses import replace
import importlib
import json

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope='module')
def producer(root):
    """Import repository-only tooling after installed-wheel checks have skipped."""
    module = importlib.import_module('tools.docs_analytics.portfolio_reports')
    assert module.ROOT.resolve() == root.resolve()
    return module


@pytest.fixture(scope='module')
def analytics(producer):
    """Compute the real teaching baseline once with network access disabled."""
    with producer.offline():
        return producer.build_analytics(producer.configuration())


def test_registry_matches_executed_configuration(root, producer, analytics):
    """The real producer's configuration and seven checks must match the registry contract."""
    registry = producer.load_registry(root)
    config = registry['producers'][producer.NAME]['configuration']
    assert config == producer.configuration() == analytics['configuration']
    assert set(analytics['tables']) == set(config['tables'])
    assert all(analytics['checks'].values())
    assert len(analytics['diagnostics']['solves']) == 43
    weights = analytics['tables']['target_weights']
    assert weights.index[[0, -1]].tolist() == [
        pd.Timestamp('2015-04-01'), pd.Timestamp('2025-10-01')]


@pytest.mark.parametrize('section,key,value', [
    ('parameters', 'cost_rate', 0),
    ('parameters', 'span', 26),
    ('parameters', 'apply_quirks', True),
    ('conventions', 'seed', 1),
    ('conventions', 'implementation_lag', 0),
    ('conventions', 'annualization', 260),
    ('conventions', 'returns', 'simple'),
    ('fixture', 'factory', 'unrecorded fixture'),
    ('rendering', 'dpi', 100),
])
def test_configuration_changes_cannot_be_silently_ignored(producer, section, key, value):
    """A manifest must never echo a requested option that the producer did not apply."""
    config = producer.configuration()
    config[section][key] = value
    with pytest.raises(ValueError, match='Unsupported portfolio-report configuration'):
        producer.build_analytics(config)


def test_weights_and_risk_against_direct_reference(analytics):
    """Check budget/caps and Euler volatility contributions with independent NumPy arithmetic."""
    tables = analytics['tables']
    weights = tables['target_weights']
    # SLSQP is configured with ftol=1e-8; roundoff varies across BLAS platforms.
    np.testing.assert_allclose(weights.sum(axis=1), 1, rtol=0, atol=1e-8)
    assert weights.min().min() >= -1e-10
    assert weights.max().max() <= 0.35 + 1e-10
    w = weights.iloc[-1].to_numpy()
    covariance = tables['last_covariance'].to_numpy()
    variance = np.einsum('i,ij,j->', w, covariance, w)
    expected = w * np.matmul(covariance, w) / np.sqrt(variance)
    np.testing.assert_allclose(tables['risk_contributions'].iloc[:, 0], expected, rtol=1e-12)
    np.testing.assert_allclose(expected.sum(), np.sqrt(variance), rtol=1e-12)


def test_covariance_scale_against_explicit_weekly_kernel(analytics):
    """Reconstruct the last covariance through qis kernels with an explicit factor of 52."""
    import qis

    date = analytics['tables']['target_weights'].index[-1]
    prices = analytics['tables']['prices'].loc[:date]
    weekly = qis.to_returns(prices, freq='W-WED', is_log_returns=True,
                            is_first_zero=False, drop_first=True)
    demeaned = (weekly - qis.compute_ewm(weekly, span=52)).iloc[1:]
    expected = 52 * qis.compute_ewm_covar_tensor(
        a=demeaned.to_numpy(), span=52, nan_backfill=qis.NanBackfill.ZERO_FILL)[-1]
    np.testing.assert_allclose(analytics['tables']['last_covariance'], expected,
                               rtol=1e-12, atol=1e-14)


def test_future_prices_do_not_change_prior_covariances_or_targets(producer, analytics):
    """Perturb future prices, then truncate history and rerun production rolling construction."""
    import optimalportfolios as op
    import qis

    prices = analytics['tables']['prices']
    config = producer.configuration()
    cutoff = pd.Timestamp('2020-12-16')
    modified = prices.copy()
    future = modified.index > cutoff
    modified.loc[future] *= np.linspace(0.5, 3, future.sum())[:, None]
    estimator = producer._estimator(config)
    covars = estimator.fit_rolling_covars(modified, qis.TimePeriod('2015-01-01', str(cutoff)))
    for date, covar in covars.items():
        pd.testing.assert_frame_equal(covar, analytics['covars'][date], rtol=1e-12, atol=1e-14)
    constraints = op.Constraints(max_weights=pd.Series(0.35, index=prices.columns))
    prefix_weights, records = producer._solve(prices.loc[:cutoff], covars, constraints, config)
    pd.testing.assert_frame_equal(
        prefix_weights, analytics['tables']['target_weights'].loc[prefix_weights.index],
        check_names=False, rtol=1e-10, atol=1e-12)
    assert all(record.accepted for record in records)


def test_holdings_and_entry_cost_against_buy_and_hold_reference(analytics):
    """Before the second trade, compare the NAV to fixed units bought at the first entry price."""
    portfolio = analytics['portfolio']
    targets = analytics['tables']['target_weights']
    first, second = analytics['tables']['decision_schedule']['implementation_date'].iloc[:2]
    assert portfolio.nav.iloc[0] == 100
    assert (portfolio.units.iloc[0] == 0).all()
    expected_units = 100 * targets.iloc[0] / portfolio.prices.loc[first]
    holding_dates = portfolio.prices.loc[first:second].index[:-1]
    # Keep the residual cash and actual entry cost when the solver's budget is
    # within tolerance rather than exactly one; retain the strict NAV comparison.
    residual_cash = 100 * (1 - targets.iloc[0].sum())
    entry_cost = 100 * targets.iloc[0].abs().sum() * 0.001
    expected_nav = portfolio.prices.loc[holding_dates].dot(expected_units) + residual_cash - entry_cost
    np.testing.assert_allclose(portfolio.nav.loc[holding_dates], expected_nav, rtol=1e-12)
    changes = portfolio.units.diff().fillna(0)
    assert (changes.loc[~portfolio.is_rebalancing] == 0).all().all()
    reference_cost = changes.abs() * portfolio.prices * 0.001
    np.testing.assert_allclose(portfolio.realized_costs, reference_cost, rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('defect', ['fallback', 'missing_record', 'wrong_context', 'bad_weights'])
def test_solver_failures_cannot_become_successful_previews(producer, monkeypatch, defect):
    """Inject a realistic rejected solve, missing diagnostic or noncompliant output."""
    original = producer._solve

    def broken(*args, **kwargs):
        """Corrupt one actual solver record or weight without changing the numerical engine."""
        weights, records = original(*args, **kwargs)
        if defect == 'fallback':
            records[0] = replace(records[0], accepted=False, fallback_source='weights_0')
        elif defect == 'missing_record':
            records.pop()
        elif defect == 'wrong_context':
            records[0] = replace(records[0], context='1900-01-01')
        else:
            weights.iloc[0, 0] = 0.8
        return weights, records

    monkeypatch.setattr(producer, '_solve', broken)
    with producer.offline(), pytest.raises(ValueError):
        producer.build_analytics(producer.configuration())


def test_point_in_time_check_detects_a_forward_looking_estimate(producer, monkeypatch):
    """The production check rejects a covariance that accidentally includes future observations."""
    estimator = producer._estimator(producer.configuration())
    original = estimator.fit_rolling_covars

    def future_covars(prices, time_period):
        """Replace the first decision's matrix with a later matrix."""
        covars = original(prices, time_period)
        covars[next(iter(covars))] = list(covars.values())[1].copy()
        return covars

    monkeypatch.setattr(estimator, 'fit_rolling_covars', future_covars)
    def supplied_estimator(config):
        """Supply the deliberately forward-looking estimator only for this regression."""
        return estimator

    monkeypatch.setattr(producer, '_estimator', supplied_estimator)
    with producer.offline(), pytest.raises(ValueError, match='failed numerical checks'):
        producer.build_analytics(producer.configuration())


def test_two_offline_previews_have_identical_images_tables_and_provenance(
        root, producer, tmp_path, monkeypatch):
    """Exercise the real family writer twice and compare every PNG/CSV fingerprint."""
    monkeypatch.setenv('AGENT_LOCAL_ROOT', str(tmp_path))
    first = producer.generate_preview(tmp_path / 'first', root)
    second = producer.generate_preview(tmp_path / 'second', root)
    records = [json.loads((path / 'family_manifest.json').read_text(encoding='utf-8'))
               for path in (first, second)]
    a, b = records
    assert a['outputs'] == b['outputs']
    assert len(a['outputs']) == 15
    assert a['source'] == b['source']
    assert a['environment'] == b['environment']
    assert a['producers'] == b['producers']
    assert a['kind'] == 'documentation_analytics_family_preview'
    assert a['publication_ready'] is False
    assert a['review_status'] == 'pending'
    assert not (first / 'analytics_manifest.json').exists()
    for name, record in a['outputs'].items():
        assert producer.file_record(first / name) == record
    validator = importlib.import_module('tools.docs_analytics.validate')
    with pytest.raises(ValueError):
        validator.validate_bundle(first, root)


def test_failed_preview_keeps_only_failure_record(root, producer, tmp_path, monkeypatch):
    """A failed family never leaves a completion marker or a final output directory."""
    runner = importlib.import_module('tools.docs_analytics.run')
    monkeypatch.setenv('AGENT_LOCAL_ROOT', str(tmp_path))

    def fail(*args):
        """Represent a producer failure before saving any figures."""
        raise RuntimeError('injected failure')

    monkeypatch.setattr(runner, '_produce', fail)
    with pytest.raises(RuntimeError, match='injected failure'):
        producer.generate_preview(tmp_path / 'failed', root)
    assert not (tmp_path / 'failed').exists()
    staging, = tmp_path.glob('.failed-building-*')
    assert (staging / 'FAILED.json').exists()
    assert not (staging / 'family_manifest.json').exists()
