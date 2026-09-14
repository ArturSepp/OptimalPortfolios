"""Exercise the covariance-only optimizer comparison and its independently checked results."""

from copy import deepcopy
from dataclasses import replace
import importlib
import json

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope='module')
def producer(root):
    """Import repository-only tooling after installed-wheel checks have skipped."""
    module = importlib.import_module('tools.docs_analytics.optimiser_comparison')
    assert module.reports.ROOT.resolve() == root.resolve()
    return module


@pytest.fixture(scope='module')
def analytics(producer):
    """Compute the three real objectives once with the offline guard active."""
    with producer.reports.offline():
        return producer.build_analytics(producer.configuration())


def test_registry_grid_and_native_solver_records(root, producer, analytics):
    """Each objective has 43 accepted and compliant native solves on the common dates."""
    config = producer.reports.load_registry(root)['producers'][producer.NAME]['configuration']
    assert config == producer.configuration() == analytics['configuration']
    assert set(analytics['tables']) == set(config['tables'])
    assert all(analytics['checks'].values())
    solves = analytics['diagnostics']['solves']
    assert len(solves) == len({item['context'] for item in solves}) == 129
    expected = analytics['tables']['decision_schedule'].index.strftime('%Y-%m-%d').tolist()
    backends = {
        'MIN_VARIANCE': ('CLARABEL', 'optimal'),
        'MAX_DIVERSIFICATION': ('SLSQP', '0'),
        'EQUAL_RISK_CONTRIBUTION': ('risk_budgeting', None),
    }
    for objective, (backend, status) in backends.items():
        selected = [item for item in solves if item['objective'] == objective]
        assert [item['decision_date'] for item in selected] == expected
        assert all(item['solver'] == backend and item['raw_status'] == status
                   and item['accepted'] and item['compliant'] and item['fallback_source'] is None
                   and item['covariance_stabilization']['factorized'] is False
                   for item in selected)
    np.testing.assert_allclose(analytics['tables']['risk_budgets']['target_fraction'], 1 / 6)


@pytest.mark.parametrize('section,key,value', [
    ('parameters', 'objectives', ['MIN_VARIANCE']),
    ('parameters', 'objectives', ['MAXIMUM_SHARPE_RATIO']),
    ('parameters', 'cost_rate', 0),
    ('parameters', 'span', 26),
    ('parameters', 'factorize_covar', True),
    ('parameters', 'risk_budget', 'unequal'),
    ('conventions', 'seed', 1),
    ('conventions', 'implementation_lag', 0),
])
def test_unsupported_configuration_is_rejected(producer, section, key, value):
    """Reject silently ignored choices, mean-based substitutions and a second changed dimension."""
    config = producer.configuration()
    config[section][key] = value
    with pytest.raises(ValueError, match='Unsupported optimizer-comparison configuration'):
        producer.build_analytics(config)


@pytest.mark.parametrize('objective', ['', 'MAXIMUM_SHARPE_RATIO', 'MAX_CARA_MIXTURE', True])
def test_shared_helper_only_allows_declared_objectives(producer, objective):
    """The shared extension cannot implicitly enable unrelated production objectives."""
    with pytest.raises(ValueError, match='Unsupported covariance-only objective'):
        producer.reports.build_analytics(producer.reports.configuration(), objective=objective)


def test_shared_helper_refuses_two_overrides(producer):
    """A sensitivity case cannot mix a changed objective with a changed EWMA span."""
    with pytest.raises(ValueError, match='one comparison dimension'):
        producer.reports.build_analytics(
            producer.reports.configuration(), objective='MIN_VARIANCE', span=26)


def test_max_diversification_preserves_the_portfolio_baseline(producer, analytics):
    """The default pipeline is unchanged and every case records its actual configuration."""
    base = producer.reports.configuration()
    original = deepcopy(base)
    default = producer.reports.build_analytics(base)
    assert base == original
    for name, table in default['tables'].items():
        pd.testing.assert_frame_equal(
            table, analytics['cases']['Max diversification']['tables'][name])
    for objective, label in producer.OBJECTIVES.items():
        assert analytics['cases'][label]['configuration'] == (
            producer.reports.case_configuration(base, objective))


@pytest.mark.parametrize('position', [0, 21, 42])
def test_minimum_variance_against_independent_slsqp(analytics, position):
    """Compare the convex solver to a scaled SciPy formulation on first, middle and last dates."""
    from scipy.optimize import minimize

    case = analytics['cases']['Minimum variance']
    date = case['tables']['target_weights'].index[position]
    matrix = case['covars'][date].to_numpy()
    scaled = matrix / np.diag(matrix).max()

    def objective(weights):
        """Compute the quadratic reference objective directly, only inside the test."""
        return np.einsum('i,ij,j->', weights, scaled, weights)

    def budget(weights):
        """Require the reference weights to be fully invested."""
        return weights.sum() - 1

    reference = minimize(objective, np.full(6, 1 / 6), method='SLSQP',
                         bounds=[(0, 0.35)] * 6,
                         constraints={'type': 'eq', 'fun': budget},
                         options={'ftol': 1e-12, 'maxiter': 1000})
    assert reference.success
    actual = case['tables']['target_weights'].loc[date].to_numpy()
    np.testing.assert_allclose(actual, reference.x, atol=2e-5, rtol=0)


@pytest.mark.parametrize('position', [0, 21, 42])
def test_risk_budgeting_against_independent_log_barrier(analytics, position):
    """Compare ADMM-CCD to a SciPy cone-constrained log-barrier problem, then normalize."""
    from scipy.optimize import minimize

    case = analytics['cases']['Equal risk budgets']
    date = case['tables']['target_weights'].index[position]
    matrix = case['covars'][date].to_numpy()
    scaled = matrix / np.diag(matrix).max()

    def objective(positions):
        """Use scale-invariant risk-budget coordinates rather than a weight-deviation penalty."""
        return (0.5 * np.einsum('i,ij,j->', positions, scaled, positions)
                - np.log(positions).mean())

    def caps(positions):
        """Represent 35% weight caps in positive, unnormalized cone coordinates."""
        return 0.35 * positions.sum() - positions

    reference = minimize(objective, np.ones(6), method='SLSQP', bounds=[(1e-10, None)] * 6,
                         constraints={'type': 'ineq', 'fun': caps},
                         options={'ftol': 1e-12, 'maxiter': 1000})
    assert reference.success
    expected = reference.x / reference.x.sum()
    np.testing.assert_allclose(
        case['tables']['target_weights'].loc[date], expected, atol=2e-5, rtol=0)


@pytest.mark.parametrize('objective', ['MIN_VARIANCE', 'EQUAL_RISK_CONTRIBUTION'])
def test_future_prices_do_not_change_prior_targets(producer, analytics, objective):
    """A changed future and a truncated history preserve each added objective's earlier targets."""
    import optimalportfolios as op
    import qis

    case = analytics['cases'][producer.OBJECTIVES[objective]]
    prices = case['tables']['prices']
    cutoff = pd.Timestamp('2020-12-16')
    modified = prices.copy()
    future = modified.index > cutoff
    modified.loc[future] *= np.linspace(0.4, 4, future.sum())[:, None]
    covars = producer.reports._estimator(case['configuration']).fit_rolling_covars(
        modified, qis.TimePeriod('2015-01-01', str(cutoff)))
    constraints = op.Constraints(max_weights=pd.Series(0.35, index=prices.columns))
    weights, records = producer.reports._solve(
        modified.loc[:cutoff], covars, constraints, case['configuration'])
    pd.testing.assert_frame_equal(
        weights, case['tables']['target_weights'].loc[weights.index],
        check_names=False, rtol=1e-10, atol=1e-12)
    assert all(record.accepted for record in records)


def test_cost_summary_against_traded_units(analytics):
    """Reconstruct all displayed cost totals from unit changes and execution prices."""
    for label, case in analytics['cases'].items():
        portfolio = case['portfolio']
        notional = (portfolio.units.diff().fillna(0).abs() * portfolio.prices).sum(axis=1)
        expected = (notional / portfolio.nav).sum()
        assert analytics['tables']['cost_summary'].loc[label, 'summed_cost_fraction'] == (
            pytest.approx(0.001 * expected, rel=1e-12))


@pytest.mark.parametrize('defect', [
    'prices', 'covariance', 'schedule', 'sample', 'checks', 'configuration',
    'fallback', 'missing_solve', 'objective_reuse', 'floor',
])
def test_inconsistent_cases_fail_closed(producer, analytics, monkeypatch, defect):
    """Inject an inconsistent shared-case result and require the comparison validator to stop."""
    def corrupt(config, *, objective=None):
        """Copy real results and corrupt the selected comparison invariant."""
        label = producer.OBJECTIVES[objective]
        case = deepcopy(analytics['cases'][label])
        if defect == 'floor':
            for matrix in case['covars'].values():
                matrix.iloc[:, :] = np.eye(6) * 1e-8
            case['tables']['last_covariance'].iloc[:, :] = np.eye(6) * 1e-8
            return case
        if objective != 'EQUAL_RISK_CONTRIBUTION':
            return case
        if defect == 'prices':
            case['tables']['prices'].iloc[0, 0] *= 1.01
        elif defect == 'covariance':
            date = next(iter(case['covars']))
            case['covars'][date].iloc[0, 0] = 1e-8
        elif defect == 'schedule':
            case['tables']['decision_schedule'].iloc[0, 0] += pd.Timedelta(days=1)
        elif defect == 'sample':
            case['tables']['navs'] = case['tables']['navs'].iloc[1:]
        elif defect == 'checks':
            case['checks'].clear()
        elif defect == 'configuration':
            case['configuration']['parameters']['objective'] = 'MIN_VARIANCE'
        elif defect == 'fallback':
            case['diagnostics']['solves'][0]['fallback_source'] = 'weights_0'
        elif defect == 'missing_solve':
            case['diagnostics']['solves'].pop()
        else:
            case['tables']['target_weights'] = (
                analytics['cases']['Minimum variance']['tables']['target_weights'].copy())
        return case

    monkeypatch.setattr(producer.reports, 'build_analytics', corrupt)
    with pytest.raises(ValueError):
        producer.build_analytics(producer.configuration())


@pytest.mark.parametrize('backend,status', [
    ('CLARABEL', 'infeasible'), ('risk_budgeting', 'nonconverged'), ('unknown', '0'),
])
def test_native_failure_status_cannot_be_normalized_to_success(
        producer, analytics, backend, status):
    """A true accepted flag alone is insufficient without a recognized native success status."""
    import optimalportfolios as op

    case = analytics['cases']['Minimum variance']
    prices, covars = case['tables']['prices'], case['covars']
    constraints = op.Constraints(max_weights=pd.Series(0.35, index=prices.columns))
    weights, records = producer.reports._solve(prices, covars, constraints, case['configuration'])
    records[0] = replace(records[0], solver=backend, status=status)
    evidence = producer.reports._diagnostics(weights, covars, constraints, records)
    assert evidence['solves'][0]['status'] == 'rejected'


def test_repeated_previews_are_identical_and_not_publishable(
        root, producer, tmp_path, monkeypatch):
    """Run the shared writer twice, compare every output and preserve the complete-bundle gate."""
    monkeypatch.setenv('AGENT_LOCAL_ROOT', str(tmp_path))
    paths = [producer.reports.generate_preview(tmp_path / name, root, family_name=producer.NAME)
             for name in ('first', 'second')]
    a, b = [json.loads((path / 'family_manifest.json').read_text()) for path in paths]
    assert len(a['outputs']) == 15
    assert a['outputs'] == b['outputs']
    assert a['source'] == b['source']
    assert a['environment'] == b['environment']
    assert a['producers'] == b['producers']
    assert a['publication_ready'] is False
    assert a['review_status'] == 'pending'
    assert not (paths[0] / 'analytics_manifest.json').exists()
    validator = importlib.import_module('tools.docs_analytics.validate')
    with pytest.raises(ValueError):
        validator.validate_bundle(paths[0], root)
