"""Execute the solver guide and check its numerical, routing and outcome contracts."""

from dataclasses import asdict, fields, replace
import hashlib
import importlib
import inspect
import re
import runpy

import numpy as np
import pandas as pd
import pytest

LEGACY_HEADINGS = [
    'Optimization Module',
    'Architecture',
    'Submodule roles',
    'Dispatch flow',
    'Three-layer solver pattern',
    'OptimiserConfig',
    'Solver reference',
    'Constraint system',
    'Why constraints are shared but objectives are not',
    'Solver backends',
    '`Constraints` — the main container',
    'Constraint enforcement types',
    'Backend and enforcement capabilities',
    'Constraint classes',
    '`GroupLowerUpperConstraints`',
    '`BenchmarkDeviationConstraints`',
    '`GroupTrackingErrorConstraint`',
    '`GroupTurnoverConstraint`',
    'Feasibility validation',
    'NaN handling and universe filtering',
    'Structured constraint inspection',
    'Test pattern',
    'Constraint test files',
    'References',
]
PRESERVED_HASHES = [
    '944a38317e815c37dde3cc287e04ad1ff48d32cf0745551dc109e3951e7a4dd1',
    '3f077e20907743890aa3d56596711034ebac0c2e51b6159ba616b509cee66a44',
    '5a4ae5847080c8a07027498a21b943fc577f9156e27f17efefca31c3aabb8cb6',
    '28e3f8e417dfff8c0347922383d3391f20041ac09f5e4d78985a3acbd5e298a8',
]
ROUTES = {
    'EQUAL_RISK_CONTRIBUTION': 'rolling_risk_budgeting',
    'MAX_DIVERSIFICATION': 'rolling_maximise_diversification',
    'MIN_VARIANCE': 'rolling_quadratic_optimisation',
    'QUADRATIC_UTILITY': 'rolling_quadratic_optimisation',
    'MAXIMUM_SHARPE_RATIO': 'rolling_maximize_portfolio_sharpe',
    'MAX_CARA_MIXTURE': 'rolling_maximize_cara_mixture',
}


@pytest.fixture(scope='module')
def article(root):
    """Read the authoritative guide only in a source checkout."""
    return (root / 'docs/optimization_module_readme.md').read_text(encoding='utf-8')


@pytest.fixture(scope='module')
def examples(article):
    """Execute all published examples together with network access prohibited."""
    blocks = re.findall(r'^```python([^\n]*)\n(.*?)^```', article, re.M | re.S)
    assert len(blocks) == 11 and all(not options.strip() for options, _ in blocks)
    state = {'__name__': '__optimization_guide__'}

    def reject_network(*args, **kwargs):
        """Reject any network-dependent example."""
        raise AssertionError('The optimization guide must execute offline')

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr('socket.create_connection', reject_network)
        patch.setattr('socket.socket.connect', reject_network)
        for i, (_, code) in enumerate(blocks):
            exec(compile(code, f'optimization_module_readme.md (block {i})', 'exec'), state)
    return state


def test_structure_legacy_headings_and_examples(article, root):
    """Keep the utility standard and original anchors while retiring the stale dataclass."""
    checker = runpy.run_path(str(root / 'tools/check_docs.py'))
    assert not checker['check_document'](article, methodology=False)
    assert not checker['check_local_links'](
        article, root / 'docs/optimization_module_readme.md', root)
    assert set(LEGACY_HEADINGS) <= set(re.findall(r'^#{1,4} (.+)$', article, re.M))
    blocks = re.findall(r'^```python\n(.*?)^```', article, re.M | re.S)
    preserved = blocks.copy()
    backend = blocks[5]
    start = backend.index('constraints.set_cvx_all_constraints')
    end = backend.index('\ncvx_rows')
    preserved.append(backend[start:end])
    hashes = {hashlib.sha256(code.strip().encode()).hexdigest() for code in preserved}
    assert set(PRESERVED_HASHES) <= hashes
    assert 'class OptimiserConfig:' not in article


def test_config_table_covers_all_fields_and_dataclass_defaults(article, examples):
    """Tie the eight documented defaults to the real frozen dataclass."""
    config = examples['default_config']
    expected = {
        'solver': 'CLARABEL', 'verbose': False, 'apply_total_to_good_ratio': False,
        'use_drifted_weights_0': True, 'diagnose_infeasibility': True,
        'validate_inputs': True, 'max_constraint_relaxation': None, 'factorize_covar': True,
    }
    assert asdict(config) == examples['configuration'] == expected
    assert config.__dataclass_params__.frozen
    for field in fields(config):
        assert f'| `{field.name}` |' in article
    assert not examples['legacy_drift_config'].use_drifted_weights_0


def test_entry_point_defaults_are_distinct(examples):
    """Do not silently replace dispatcher defaults with dataclass or direct-solver defaults."""
    opt = examples['opt']
    dispatch = inspect.signature(opt.compute_rolling_optimal_weights).parameters
    direct = inspect.signature(opt.wrapper_quadratic_optimisation).parameters
    backtest = inspect.signature(opt.backtest_rolling_optimal_portfolio).parameters
    assert dispatch['portfolio_objective'].default == opt.PortfolioObjective.MAX_DIVERSIFICATION
    assert dispatch['optimiser_config'].default.apply_total_to_good_ratio
    assert direct['optimiser_config'].default.apply_total_to_good_ratio
    assert not inspect.signature(opt.wrapper_minimise_tracking_error).parameters[
        'optimiser_config'].default.apply_total_to_good_ratio
    assert dispatch['carra'].default == 0.5 and direct['carra'].default == 1.0
    assert dispatch['roll_window'].default == 20 and backtest['roll_window'].default == 312
    assert backtest['rebalancing_costs'].default == 0.0010
    assert backtest['weight_implementation_lag'].default is None


def test_return_contracts_and_accepted_outcomes(examples):
    """Separate structured numerical outcomes, labelled weights and reporting containers."""
    from optimalportfolios.optimization.solver_diagnostics import OptimizationOutcome

    for name, value in examples.items():
        if name.endswith('outcome'):
            assert isinstance(value, OptimizationOutcome)
            assert value.accepted and value.compliant
            assert value.covar_factorization is not None
            assert value.weights.shape == (5,)
    for name in ['weights', 'utility_weights', 'sharpe_weights', 'tracking_weights',
                 'risk_weights', 'diversification_weights', 'mixture_weights',
                 'return_weights', 'vol_weights', 'tactical_weights', 'yield_weights',
                 'soft_weights', 'group_weights']:
        weights = examples[name]
        assert isinstance(weights, pd.Series) and list(weights.index) == examples['tickers']
        assert np.isfinite(weights).all() and (weights >= -1e-7).all()
        assert weights.sum() == pytest.approx(1.0, abs=1e-6)
    assert isinstance(examples['portfolio'], examples['qis'].PortfolioData)


# Independent analytic references test the numerical contracts, not a new production risk layer.
def test_minimum_variance_and_displayed_weights_use_inverse_variance(article, examples):
    """Check the displayed numbers against the diagonal-covariance closed form."""
    variance = examples['annual_vols'].to_numpy() ** 2
    expected = (1 / variance) / np.sum(1 / variance)
    np.testing.assert_allclose(examples['weights'], expected, atol=2e-6)
    np.testing.assert_allclose(examples['raw_outcome'].weights, expected, atol=2e-6)
    rows = re.findall(r'^\| (Equity [AB]|Bond [AB]|Gold) \| ([\d.]+) \| ([\d.]+) \|$',
                      article, re.M)
    assert [row[0] for row in rows] == examples['tickers']
    np.testing.assert_allclose([float(row[2]) for row in rows], expected, rtol=0, atol=0.5e-6)


def test_quadratic_utility_has_the_half_gamma_coefficient(examples):
    """Use the equality-constrained closed form with an interior long-only solution."""
    inverse = 1 / examples['annual_vols'].to_numpy() ** 2
    means = examples['expected_returns'].to_numpy()
    multiplier = (np.sum(inverse * means) - 5.0) / np.sum(inverse)
    expected = inverse * (means - multiplier) / 5.0
    assert np.min(expected) > 0 and np.max(expected) < 1
    np.testing.assert_allclose(examples['utility_weights'], expected, atol=3e-6)


def test_sharpe_solution_uses_the_supplied_means(examples):
    """For positive diagonal inputs the tangency solution is proportional to mean/variance."""
    expected = examples['expected_returns'] / examples['annual_vols'] ** 2
    np.testing.assert_allclose(examples['sharpe_weights'], expected / expected.sum(), atol=3e-6)


@pytest.mark.parametrize('name', ['risk_weights', 'diversification_weights'])
def test_risk_and_diversification_diagonal_examples_use_inverse_volatility(examples, name):
    """Distinguish the two inverse-volatility examples from minimum variance."""
    inverse = 1 / examples['annual_vols']
    np.testing.assert_allclose(examples[name], inverse / inverse.sum(), atol=3e-5)


def test_minimum_tracking_error_returns_the_feasible_benchmark(examples):
    """A feasible benchmark has zero active risk and is the unique positive-definite optimum."""
    np.testing.assert_allclose(examples['tracking_weights'], examples['benchmark'], atol=2e-6)


def test_return_floor_uses_the_two_equality_closed_form(examples):
    """An active floor and exposure equality determine this interior minimum-variance solution."""
    means = examples['expected_returns'].to_numpy()
    matrix = np.column_stack([np.ones(5), means])
    inverse = np.diag(1 / examples['annual_vols'].to_numpy() ** 2)
    expected = inverse @ matrix @ np.linalg.solve(matrix.T @ inverse @ matrix, [1, 0.055])
    assert np.min(expected) > 0 and np.max(expected) < 1
    np.testing.assert_allclose(examples['return_weights'], expected, atol=3e-5)
    assert means @ examples['return_weights'] == pytest.approx(0.055, abs=1e-6)


def test_volatility_budget_is_in_covariance_units(examples):
    """Check the active 12% annual budget against the known diagonal fixture."""
    actual = np.linalg.norm(examples['vol_weights'] * examples['annual_vols'])
    assert actual == pytest.approx(0.12, abs=2e-6)


def test_hard_alpha_te_matches_the_ellipsoid_reference(examples):
    """Resolve the sum-zero active direction analytically, independently of CVXPY."""
    inverse = 1 / examples['annual_vols'].to_numpy() ** 2
    alpha = examples['alphas'].to_numpy()
    multiplier = np.sum(inverse * alpha) / np.sum(inverse)
    direction = inverse * (alpha - multiplier)
    scale = 0.03 / np.linalg.norm(direction * examples['annual_vols'])
    expected = examples['benchmark'].to_numpy() + scale * direction
    assert np.min(expected) > 0 and np.max(expected) < 1
    np.testing.assert_allclose(examples['tactical_weights'], expected, atol=3e-5)


def test_utility_alpha_te_does_not_retain_the_hard_cap(examples):
    """The penalty example can exceed the 3% budget while its hard rows remain compliant."""
    active = examples['soft_weights'] - examples['benchmark']
    assert np.linalg.norm(active * examples['annual_vols']) > 0.03 + 1e-3
    assert examples['soft_outcome'].accepted and examples['soft_outcome'].compliant


def test_yield_floor_and_hard_te_remain_enforced(examples):
    """The target-return-named function consumes the supplied yield vector and TE budget."""
    weights = examples['yield_weights']
    assert weights @ examples['expected_returns'] >= 0.05 - 1e-6
    active = weights - examples['benchmark']
    assert np.linalg.norm(active * examples['annual_vols']) <= 0.03 + 2e-6


def test_mixture_example_improves_on_simple_feasible_allocations(examples):
    """Evaluate fixed-mixture exponential utility independently at competing portfolios."""
    means = examples['expected_returns'].to_numpy()
    variance = examples['annual_vols'].to_numpy() ** 2

    def loss(weights):
        """Compute negative expected CARA utility for the two stated mixture components."""
        variance_term = np.sum(variance * np.asarray(weights) ** 2)
        return (0.75 * np.exp(-5 * (means @ weights) + 12.5 * variance_term)
                + 0.25 * np.exp(-2.5 * (means @ weights) + 18.75 * variance_term))

    fitted = loss(examples['mixture_weights'].to_numpy())
    alternatives = [examples['benchmark'].to_numpy(), *np.eye(5)]
    assert all(fitted <= loss(candidate) + 1e-8 for candidate in alternatives)


def test_group_and_deviation_examples_enforce_both_sets(examples):
    """Compute absolute group allocations and active deviations from the published loadings."""
    weights = examples['group_weights']
    group = examples['gluc']
    allocation = group.group_loadings.T @ weights
    assert (allocation >= group.group_min_allocation - 1e-6).all()
    assert (allocation <= group.group_max_allocation + 1e-6).all()
    deviation = examples['bdc']
    actual = deviation.factor_loading_mat.T @ (weights - examples['benchmark'])
    assert (actual.abs() <= deviation.factor_max_deviation + 1e-6).all()


def test_filtered_outcome_has_different_labels_and_ratio_policy(examples):
    """The full-universe Series and the aligned outcome differ after an invalid asset is removed."""
    opt = examples['opt']
    names = examples['tickers'] + ['Unavailable']
    covar = examples['pd_covar'].reindex(index=names, columns=names, fill_value=0.0)
    constraints = replace(
        examples['constraints'], min_weights=pd.Series(0.0, index=names),
        max_weights=pd.Series(0.25, index=names))
    weights, explicit = opt.wrapper_quadratic_optimisation(
        covar, constraints, optimiser_config=examples['config'])
    _, default = opt.wrapper_quadratic_optimisation(covar, constraints)
    assert weights.shape == (6,) and explicit.weights.shape == (5,)
    assert weights['Unavailable'] == 0.0
    assert explicit.constraints.max_weights.tolist() == [0.25] * 5
    np.testing.assert_allclose(default.constraints.max_weights, 0.25 * 6 / 5)
    np.testing.assert_allclose(weights.loc[examples['tickers']], explicit.weights)


def test_rejected_fallback_is_not_certified_compliant(examples):
    """A finite prior allocation does not meet the deliberately impossible new caps."""
    rejected = examples['rejected']
    assert not rejected.accepted and not rejected.compliant
    assert rejected.fallback_source == 'weights_0'
    np.testing.assert_allclose(examples['fallback_weights'], examples['benchmark'])
    assert any(r.hard and not r.passed for r in rejected.constraint_residuals)
    assert examples['hard_breaches'] == []
    assert not examples['outcome'].residuals_frame().empty


def test_original_compiler_examples_have_executable_inputs(examples):
    """All three original compiler calls execute, without claiming they solve a portfolio."""
    assert isinstance(examples['cvx_rows'], list) and examples['cvx_rows']
    assert isinstance(examples['scipy_rows'], list) and examples['scipy_rows']
    assert examples['pyrb_bounds'] is not None


def test_rolling_examples_keep_the_supplied_schedule_and_forecasts(examples):
    """Compare rolling allocations with the separately verified single-date solutions."""
    pairs = [('rolling_weights', 'weights'), ('rolling_utility', 'utility_weights')]
    for panel, reference in pairs:
        actual = examples[panel]
        assert actual.shape == (8, 5)
        pd.testing.assert_index_equal(actual.index, examples['decision_dates'], check_names=False)
        np.testing.assert_allclose(actual, np.broadcast_to(examples[reference], actual.shape),
                                   atol=3e-6)
    actual_dates = examples['portfolio'].is_rebalancing
    traded = actual_dates[actual_dates].index
    expected = examples['decision_dates'] + pd.offsets.MonthEnd(1)
    pd.testing.assert_index_equal(traded, expected, check_names=False)


@pytest.mark.parametrize('objective', ['MIN_VARIANCE', 'QUADRATIC_UTILITY'])
def test_future_prices_do_not_change_earlier_dispatch_results(examples, objective):
    """Keep covariance fixed and change only future price observations."""
    opt = examples['opt']
    kwargs = dict(
        constraints=examples['constraints'], covar_dict=examples['covar_dict'],
        portfolio_objective=getattr(opt.PortfolioObjective, objective),
        returns_freq='ME', span=12, optimiser_config=examples['config'])
    before = opt.compute_rolling_optimal_weights(examples['prices'], **kwargs)
    changed = examples['prices'].copy()
    cutoff = pd.Timestamp('2023-12-31')
    mask = changed.index > cutoff
    changed.loc[mask, 'Equity A'] *= np.linspace(1.1, 2, mask.sum())
    after = opt.compute_rolling_optimal_weights(changed, **kwargs)
    np.testing.assert_allclose(before.loc[:cutoff], after.loc[:cutoff], atol=2e-7)


@pytest.mark.parametrize('objective,target', list(ROUTES.items()))
def test_dispatch_routes_and_parameter_scope(examples, monkeypatch, objective, target):
    """Check every enum route, including CARA's distinct inputs and annual mean estimation."""
    opt = examples['opt']
    module = importlib.import_module('optimalportfolios.optimization.wrapper_rolling_portfolios')
    calls, mean_calls = [], []
    sentinel = pd.DataFrame([[1.0]], index=[pd.Timestamp('2024-12-31')], columns=['sentinel'])

    def target_spy(**kwargs):
        """Record the single routed call."""
        calls.append(kwargs)
        return sentinel

    def means_spy(**kwargs):
        """Record the forecast convention forwarded by the dispatcher."""
        mean_calls.append(kwargs)
        return examples['return_forecasts']

    for name in set(ROUTES.values()):
        def wrong_route(_name=name, **kwargs):
            """Fail on any solver route other than the documented one."""
            raise AssertionError(f'Unexpected route: {_name}')
        monkeypatch.setattr(module, name, wrong_route)
    monkeypatch.setattr(module, target, target_spy)
    monkeypatch.setattr(module, 'estimate_rolling_ewma_means', means_spy)
    period = examples['qis'].TimePeriod('2024-01-01', '2024-12-31')
    result = module.compute_rolling_optimal_weights(
        examples['prices'], examples['constraints'], examples['covar_dict'],
        portfolio_objective=getattr(opt.PortfolioObjective, objective), time_period=period,
        returns_freq='ME', rebalancing_freq='YE', roll_window=17, n_mixures=2,
        optimiser_config=examples['config'])
    assert result is sentinel and len(calls) == 1
    call = calls[0]
    assert call['optimiser_config'] is examples['config']
    if objective == 'MAX_CARA_MIXTURE':
        assert 'covar_dict' not in call
        assert call['time_period'] is period and call['rebalancing_freq'] == 'YE'
        assert call['n_components'] == 2 and call['roll_window'] == 17
    else:
        assert call['covar_dict'] is examples['covar_dict']
        assert 'time_period' not in call and 'rebalancing_freq' not in call
    if objective in ['QUADRATIC_UTILITY', 'MAXIMUM_SHARPE_RATIO']:
        assert call['expected_returns'] is examples['return_forecasts']
        assert len(mean_calls) == 1 and mean_calls[0]['annualize'] is True
        assert mean_calls[0]['returns_freq'] == 'ME'
    else:
        assert mean_calls == []


def test_dispatch_has_no_minimum_tracking_error_enum_route(examples):
    """A direct minimum-TE function must not be confused with a seventh enum member."""
    opt = examples['opt']
    assert {objective.name for objective in opt.PortfolioObjective} == set(ROUTES)
    with pytest.raises(NotImplementedError):
        opt.compute_rolling_optimal_weights(
            examples['prices'], examples['constraints'], examples['covar_dict'],
            portfolio_objective='MINIMUM_TRACKING_ERROR')


@pytest.mark.parametrize('variable_exposure', [False, True])
def test_sharpe_selects_the_documented_backend(examples, monkeypatch, variable_exposure):
    """The fixed/variable exposure choice takes precedence over a CVXPY-style function name."""
    module = importlib.import_module('optimalportfolios.optimization.general.max_sharpe')
    used = []
    sentinel = object()

    def cvx_spy(**kwargs):
        """Record the transformed fixed-exposure route."""
        used.append('CVXPY')
        return sentinel

    def scipy_spy(**kwargs):
        """Record the direct ratio route."""
        used.append('SLSQP')
        return sentinel

    monkeypatch.setattr(module, '_cvx_maximize_sharpe_charnes_cooper', cvx_spy)
    monkeypatch.setattr(module, '_scipy_maximize_sharpe', scipy_spy)
    constraints = replace(examples['constraints'], min_exposure=0.5 if variable_exposure else 1.0)
    result = module.cvx_maximize_portfolio_sharpe(
        examples['pd_covar'].to_numpy(), examples['expected_returns'].to_numpy(), constraints)
    assert result is sentinel
    assert used == (['SLSQP'] if variable_exposure else ['CVXPY'])
