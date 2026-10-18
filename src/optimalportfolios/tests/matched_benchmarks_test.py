"""Validate B0 references, convention mismatches and optional competitor adapters offline."""
import importlib
import importlib.util

import numpy as np
import pytest


@pytest.fixture(scope='module')
def benchmark(root):
    """Load repository-only tooling after wheel users have skipped via the root fixture."""
    return importlib.import_module('tools.matched_benchmarks')


@pytest.fixture(scope='module')
def reference(benchmark):
    """Load independent oracles from the selected source checkout."""
    return importlib.import_module('tools.matched_benchmark_references')


def test_optimizer_reference_and_budget_rejection(benchmark, reference):
    """The unique KKT reference and feasibility guard reject a plausible wrong allocation."""
    covariance, means, gamma, cap, _ = benchmark.optimizer_inputs(10, 'quadratic_utility')
    weights = reference.optimizer_oracle(covariance, means, gamma)
    assert reference.verify_optimizer(weights, covariance, means, gamma, cap)['unique']
    with pytest.raises(AssertionError):
        reference.verify_optimizer(weights * .99, covariance, means, gamma, cap)


@pytest.mark.parametrize('objective', ['minimum_variance', 'quadratic_utility'])
def test_op_solver_matches_independent_kkt(benchmark, objective):
    """Check actual OP solves and observed common solver tolerances, without a timing gate."""
    with benchmark.offline(), benchmark.matched_solver() as records:
        result = benchmark.fit_optimizer(10, 'optimalportfolios', objective)
    assert result['max_weight_error'] < 2e-5
    assert len(records) == 1 and records[0]['options'] == benchmark.SOLVER_OPTIONS


def test_cash_ledger_cost_lag_and_freeze(benchmark):
    """Verify all six QIS cases against its independent owned cash ledger."""
    with benchmark.offline():
        cases = benchmark.accounting_cases(include_skfolio=False)
    assert len(cases) == 6 and all(case['ledger_passed'] for case in cases)
    assert cases[2]['qis_nav'][0] == 99
    assert cases[4]['qis_nav'][:2] == [100, 100]


def test_forecast_loss_and_variance_calibration(reference):
    """Known diagonal moments have zero covariance loss and exactly calibrated risk."""
    heldout = np.array([[1., 0], [-1., 0], [0, 1.], [0, -1.]])
    score = reference.forecast_scores(np.eye(2) / 2, heldout)
    assert score['relative_squared_frobenius_loss'] == 0
    assert score['stein_covariance_loss'] == 0
    assert score['forecast_equal_weight_variance'] == .25
    assert score['realized_over_forecast_variance'] == 1


def test_factor_fixture_chronology_and_rng_preserved(benchmark):
    """The frozen simulator cannot advance caller RNG or mix held-out observations into fit."""
    before = np.random.get_state()
    factors, responses, heldout = benchmark.factor_inputs(10)
    after = np.random.get_state()
    np.testing.assert_array_equal(before[1], after[1])
    assert before[2:] == after[2:]
    assert factors.shape == (504, 4) and responses.shape == (504, 10)
    assert heldout.shape == (252, 10)
    assert responses.index.max() < heldout.index.min()
    assert benchmark.array_hash(*benchmark.factor_inputs(10)) == benchmark.array_hash(
        factors, responses, heldout)


def test_residual_denominator_difference_is_diagonal(benchmark, reference):
    """An independently augmented OLS design isolates the two residual denominators."""
    x, y, _ = benchmark.factor_inputs(10)
    beta0, covariance0 = reference.factor_oracle(x, y, 0)
    beta1, covariance1 = reference.factor_oracle(x, y, 1)
    np.testing.assert_array_equal(beta0, beta1)
    difference = covariance1 - covariance0
    np.testing.assert_allclose(difference, np.diag(np.diag(difference)), atol=1e-16)
    assert (np.diag(difference) > 0).all()


@pytest.mark.parametrize('objective', ['minimum_variance', 'quadratic_utility'])
def test_optional_skfolio_matches_supplied_moments(benchmark, objective):
    """Optional comparator checks are explicit skips in the core source/wheel environment."""
    if importlib.util.find_spec('skfolio') is None:
        pytest.skip('Optional B0 comparator is absent; install the pinned benchmark tool')
    with benchmark.offline(), benchmark.matched_solver():
        op = benchmark.fit_optimizer(10, 'optimalportfolios', objective)
        skfolio = benchmark.fit_optimizer(10, 'skfolio', objective)
    assert op['input_hash'] == skfolio['input_hash']
    assert abs(op['objective'] - skfolio['objective']) < 2e-8


def test_optional_native_accounting_semantics(benchmark):
    """Equal gross intervals reconcile; native cost/freeze conventions must remain different."""
    if importlib.util.find_spec('skfolio') is None:
        pytest.skip('Optional B0 comparator is absent')
    with benchmark.offline():
        cases = benchmark.accounting_cases()
    assert sum(case['expected_match'] for case in cases) == 3
    assert all(case['max_nav_difference'] > 0 for case in cases if not case['expected_match'])


@pytest.mark.parametrize('method', [
    pytest.param('factorlasso', marks=pytest.mark.factor_integration), 'skfolio'])
def test_factor_fits_match_independent_covariance(benchmark, method):
    """Real estimators agree with the independently reconstructed native normalisation."""
    if method == 'skfolio' and importlib.util.find_spec('skfolio') is None:
        pytest.skip('Optional B0 comparator is absent')
    with benchmark.offline(), benchmark.matched_solver():
        result = benchmark.fit_factor(10, method)
    assert result['beta_max_error'] < 2e-6
    assert result['covariance_max_error'] < 2e-10


@pytest.mark.factor_integration
def test_benchmark_records_imported_reference_implementations(benchmark):
    """The manifest identifies installed owners rather than unexecuted checkout examples."""
    import inspect
    from pathlib import Path

    fl = importlib.import_module('factorlasso')
    ledger = importlib.import_module('qis.portfolio.tests.backtester_missing_prices_test')
    record = benchmark.benchmark_source()
    paths = [Path(inspect.getfile(fl.CurrentFactorCovarData)).resolve(),
             Path(ledger.__file__).resolve()]
    for path in paths:
        assert record['benchmark_and_owned_references'][str(path)] == benchmark.content_record(path)
    assert set(record['source_versions']) == {'optimalportfolios', 'qis', 'factorlasso'}
    assert all(record['source_versions'].values())


@pytest.mark.parametrize('repeats,sizes', [(1, (10,)), (5, (1000,)), (5, (10, 10))])
def test_unreviewed_workloads_fail_before_output(benchmark, tmp_path, repeats, sizes):
    """Reject undersampled timings and unsupported workload expansions before computation."""
    output = tmp_path / 'must-not-exist'
    with pytest.raises(ValueError, match='reviewed 10/100'):
        benchmark.run_bundle(output, repeats=repeats, sizes=sizes)
    assert not output.exists()


def test_bundle_output_tampering_is_rejected(benchmark, tmp_path):
    """The persisted manifest cannot silently vouch for changed numeric results."""
    data = tmp_path / 'result.csv'
    data.write_text('asset,value\nA,1\n', encoding='utf-8')
    benchmark.write_json(tmp_path / 'bundle.json', {
        'stage': 'B0', 'status': 'verified',
        'outputs': {'result.csv': benchmark.content_record(data)}})
    benchmark.verify_bundle(tmp_path)
    data.write_text('asset,value\nA,2\n', encoding='utf-8')
    with pytest.raises(ValueError, match='output changed'):
        benchmark.verify_bundle(tmp_path)


@pytest.mark.factor_integration
def test_future_outcomes_change_scores_but_not_the_fit(benchmark):
    """Perturbing only future observations must leave the trained covariance unchanged."""
    x, y, heldout = benchmark.factor_inputs(10)
    with benchmark.offline(), benchmark.matched_solver():
        original = benchmark.fit_factor(10, 'factorlasso', (x, y, heldout))
        changed = benchmark.fit_factor(10, 'factorlasso', (x, y, heldout * 1.5))
    np.testing.assert_allclose(original['covariance'], changed['covariance'], atol=1e-14)
    np.testing.assert_allclose(changed['realized_equal_weight_variance'],
                               2.25 * original['realized_equal_weight_variance'])
    assert original['input_hash'] != changed['input_hash']
