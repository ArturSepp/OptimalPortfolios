"""Validate the real span comparison, shared portfolio calculation and preview publication gate."""

from copy import deepcopy
import importlib
import json

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope='module')
def producer(root):
    """Import repository-only tooling after installed-wheel checks have skipped."""
    module = importlib.import_module('tools.docs_analytics.span_sensitivity')
    assert module.reports.ROOT.resolve() == root.resolve()
    return module


@pytest.fixture(scope='module')
def analytics(producer):
    """Compute all five real spans with the shared offline guard active."""
    with producer.reports.offline():
        return producer.build_analytics(producer.configuration())


def test_actual_span_grid_and_solver_records(root, producer, analytics):
    """Every declared span has 43 accepted, compliant, correctly dated solves."""
    registry = producer.reports.load_registry(root)
    config = registry['producers'][producer.NAME]['configuration']
    assert config == producer.configuration() == analytics['configuration']
    assert config['parameters']['spans'] == [5, 13, 26, 52, 104]
    assert set(analytics['tables']) == set(config['tables'])
    assert all(analytics['checks'].values())
    solves = analytics['diagnostics']['solves']
    assert len(solves) == len({item['context'] for item in solves}) == 215
    expected_dates = analytics['tables']['decision_schedule'].index.strftime('%Y-%m-%d').tolist()
    for span in producer.SPANS:
        selected = [item for item in solves if item['span'] == span]
        assert [item['decision_date'] for item in selected] == expected_dates
        assert all(item['accepted'] and item['compliant'] and item['fallback_source'] is None
                   and item['raw_status'] == '0' for item in selected)


@pytest.mark.parametrize('section,key,value', [
    ('parameters', 'spans', [5, 13, 26, 52]),
    ('parameters', 'spans', [5, 13, 26, 52, 52]),
    ('parameters', 'spans', [104, 52, 26, 13, 5]),
    ('parameters', 'cost_rate', 0),
    ('conventions', 'seed', 1),
    ('conventions', 'implementation_lag', 0),
    ('conventions', 'returns', 'simple'),
])
def test_configuration_cannot_silently_change(producer, section, key, value):
    """Reject unsupported grids and any attempt to vary a second comparison dimension."""
    config = producer.configuration()
    config[section][key] = value
    with pytest.raises(ValueError, match='Unsupported span-sensitivity configuration'):
        producer.build_analytics(config)


@pytest.mark.parametrize('span', [True, 1, 0, -1, 2.5])
def test_shared_helper_rejects_invalid_span(producer, span):
    """An override must be an integer EWMA span of at least two observations."""
    with pytest.raises(ValueError, match='sensitivity span'):
        producer.reports.build_analytics(producer.reports.configuration(), span=span)


def test_span_override_does_not_mutate_the_caller_or_change_the_52_week_baseline(
        producer, analytics):
    """The shared extension preserves default report results and records the actual override."""
    config = producer.reports.configuration()
    original = deepcopy(config)
    baseline = producer.reports.build_analytics(config)
    assert config == original
    for name, table in baseline['tables'].items():
        pd.testing.assert_frame_equal(table, analytics['cases']['52 weeks']['tables'][name])
    for span, case in zip(producer.SPANS, analytics['cases'].values()):
        expected = deepcopy(original)
        expected['parameters']['span'] = span
        assert case['configuration'] == expected


@pytest.mark.parametrize('span', [5, 13, 26, 52, 104])
def test_each_span_covariance_matches_explicit_weekly_kernels(producer, analytics, span):
    """Verify both smoothing stages and the annualization scale independently of the wrapper."""
    import qis

    case = analytics['cases'][f'{span} weeks']
    date = case['tables']['target_weights'].index[-1]
    prices = case['tables']['prices'].loc[:date]
    weekly = qis.to_returns(prices, freq='W-WED', is_log_returns=True,
                            is_first_zero=False, drop_first=True)
    demeaned = (weekly - qis.compute_ewm(weekly, span=span)).iloc[1:]
    expected = 52 * qis.compute_ewm_covar_tensor(
        a=demeaned.to_numpy(), span=span, nan_backfill=qis.NanBackfill.ZERO_FILL)[-1]
    np.testing.assert_allclose(case['tables']['last_covariance'], expected,
                               rtol=1e-12, atol=1e-14)


@pytest.mark.parametrize('span', [5, 104])
def test_fast_and_slow_spans_are_point_in_time(producer, analytics, span):
    """Future price changes and truncated reruns must leave earlier decisions unchanged."""
    import optimalportfolios as op
    import qis

    case = analytics['cases'][f'{span} weeks']
    prices = case['tables']['prices']
    cutoff = pd.Timestamp('2020-12-16')
    modified = prices.copy()
    future = modified.index > cutoff
    modified.loc[future] *= np.linspace(0.4, 4, future.sum())[:, None]
    covars = producer.reports._estimator(case['configuration']).fit_rolling_covars(
        modified, qis.TimePeriod('2015-01-01', str(cutoff)))
    for date, matrix in covars.items():
        pd.testing.assert_frame_equal(matrix, case['covars'][date], rtol=1e-12, atol=1e-14)
    constraints = op.Constraints(max_weights=pd.Series(0.35, index=prices.columns))
    weights, records = producer.reports._solve(
        prices.loc[:cutoff], covars, constraints, case['configuration'])
    pd.testing.assert_frame_equal(
        weights, case['tables']['target_weights'].loc[weights.index],
        check_names=False, rtol=1e-10, atol=1e-12)
    assert all(record.accepted for record in records)


def test_cost_summary_against_traded_units_reference(analytics):
    """Reconstruct displayed totals from unit changes, trade prices and same-day NAV."""
    summary = analytics['tables']['cost_summary']
    for label, case in analytics['cases'].items():
        portfolio = case['portfolio']
        traded_notional = (portfolio.units.diff().fillna(0).abs() * portfolio.prices).sum(axis=1)
        expected_turnover = (traded_notional / portfolio.nav).sum()
        assert summary.loc[label, 'summed_gross_turnover'] == pytest.approx(
            expected_turnover, rel=1e-12)
        assert summary.loc[label, 'summed_cost_fraction'] == pytest.approx(
            0.001 * expected_turnover, rel=1e-12)


@pytest.mark.parametrize('defect', [
    'prices', 'benchmark_prices', 'schedule', 'sample', 'checks',
    'fallback', 'missing_solve', 'span_reuse',
])
def test_inconsistent_cases_are_rejected(producer, monkeypatch, defect):
    """Comparison-level gates detect drift or a lost span even when a case returns normally."""
    original = producer.reports.build_analytics

    def corrupt(config, *, span=None):
        """Alter one shared-case result after its own checks to exercise the comparison guards."""
        case = original(config, span=span)
        if span != 104:
            return case
        if defect in {'prices', 'benchmark_prices'}:
            case['tables'][defect].iloc[0, 0] *= 1.01
        elif defect == 'schedule':
            case['tables']['decision_schedule'].iloc[0, 0] += pd.Timedelta(days=1)
        elif defect == 'sample':
            case['tables']['navs'] = case['tables']['navs'].iloc[1:]
        elif defect == 'checks':
            case['checks'].clear()
        elif defect == 'fallback':
            case['diagnostics']['solves'][0]['fallback_source'] = 'weights_0'
        elif defect == 'missing_solve':
            case['diagnostics']['solves'].pop()
        else:
            case = original(config, span=52)
        return case

    monkeypatch.setattr(producer.reports, 'build_analytics', corrupt)
    with producer.reports.offline(), pytest.raises(ValueError):
        producer.build_analytics(producer.configuration())


def test_repeated_previews_are_identical_and_cannot_be_published(
        root, producer, tmp_path, monkeypatch):
    """Use the shared writer twice; compare all outputs and keep complete publication gated."""
    monkeypatch.setenv('AGENT_LOCAL_ROOT', str(tmp_path))
    first = producer.reports.generate_preview(tmp_path / 'first', root, family_name=producer.NAME)
    second = producer.reports.generate_preview(tmp_path / 'second', root, family_name=producer.NAME)
    a, b = [json.loads((path / 'family_manifest.json').read_text(encoding='utf-8'))
            for path in (first, second)]
    assert a['outputs'] == b['outputs']
    assert len(a['outputs']) == 14
    assert a['source'] == b['source']
    assert a['environment'] == b['environment']
    assert a['producers'] == b['producers']
    assert a['family'] == producer.NAME
    assert a['publication_ready'] is False
    assert a['review_status'] == 'pending'
    assert not (first / 'analytics_manifest.json').exists()
    for name, record in a['outputs'].items():
        assert producer.reports.file_record(first / name) == record
    validator = importlib.import_module('tools.docs_analytics.validate')
    with pytest.raises(ValueError):
        validator.validate_bundle(first, root)


@pytest.mark.parametrize('name', ['unknown', 'covariance_comparison'])
def test_shared_writer_refuses_unknown_or_pending_families(
        root, producer, tmp_path, monkeypatch, name):
    """Selecting a family does not bypass readiness, including an injected pending registry."""
    if name == 'covariance_comparison':
        registry = deepcopy(producer.reports.load_registry(root))
        registry['producers'][name]['status'] = 'pending'

        def pending_registry(_root):
            """Return an explicitly pending fixture without modifying source files."""
            return registry

        monkeypatch.setattr(producer.reports, 'load_registry', pending_registry)
    output = tmp_path / 'refused'
    with pytest.raises(ValueError):
        producer.reports.generate_preview(output, root, family_name=name)
    assert not output.exists()


def test_shared_writer_checks_the_selected_family(root, producer, tmp_path, monkeypatch):
    """A failure in the selected span producer cannot leave a family completion marker."""
    runner = importlib.import_module('tools.docs_analytics.run')
    monkeypatch.setenv('AGENT_LOCAL_ROOT', str(tmp_path))

    def fail(name, spec, source):
        """Assert exact dispatch before simulating a producer failure."""
        assert name == producer.NAME
        assert spec['configuration'] == producer.configuration()
        raise RuntimeError('injected span failure')

    monkeypatch.setattr(runner, '_produce', fail)
    with pytest.raises(RuntimeError, match='injected span failure'):
        producer.reports.generate_preview(tmp_path / 'failed', root, family_name=producer.NAME)
    assert not (tmp_path / 'failed').exists()
    staging, = tmp_path.glob('.failed-building-*')
    assert (staging / 'FAILED.json').exists()
    assert not (staging / 'family_manifest.json').exists()
