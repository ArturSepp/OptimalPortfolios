"""Compatibility contracts for the portfolio-constraint module boundary.

The constraint implementation is intentionally split behind
``optimalportfolios.optimization.constraints``.  These tests freeze the import aliases and
constructor/method shapes that callers see, so moving an implementation cannot silently turn a
structural refactor into an API change.
"""
import inspect
import pickle
from dataclasses import fields
from pathlib import Path

import pytest

import optimalportfolios as op
import optimalportfolios.optimization as optimization
import optimalportfolios.optimization.constraints as constraint_module
import optimalportfolios.optimization.constraints.analytics as constraint_analytics
import optimalportfolios.optimization.solver_diagnostics as solver_diagnostics
from optimalportfolios.optimization.constraints import ConstraintEnforcementType, Constraints


ROOT_REEXPORTS = (
    'ConstraintEnforcementType',
    'ConstraintResidual',
    'Constraints',
    'GroupLowerUpperConstraints',
    'GroupTrackingErrorConstraint',
    'GroupTurnoverConstraint',
    'compute_eligible_rebalancing_bounds',
    'evaluate_constraint_residuals',
    'merge_group_lower_upper_constraints',
)

DIRECT_MODULE_API = (
    'BenchmarkBetaConstraint',
    'BenchmarkDeviationConstraints',
    'ConstraintEnforcementType',
    'ConstraintResidual',
    'Constraints',
    'DroppedGroupRecord',
    'GroupLowerUpperConstraints',
    'GroupTrackingErrorConstraint',
    'GroupTurnoverConstraint',
    'RelaxationRecord',
    'add_term_to_objective_function',
    'compute_benchmark_beta_loadings',
    'compute_benchmark_beta_loadings_from_covar',
    'compute_eligible_rebalancing_bounds',
    'cvx_covar_variance',
    'evaluate_constraint_residuals',
    'long_only_constraint',
    'make_max_constraint',
    'make_min_constraint',
    'merge_group_lower_upper_constraints',
    'total_weight_constraint',
)

EXPECTED_CONSTRAINT_FIELDS = (
    ('is_long_only', True),
    ('min_weights', None),
    ('max_weights', None),
    ('max_exposure', 1.0),
    ('min_exposure', 1.0),
    ('benchmark_weights', None),
    ('tracking_err_vol_constraint', None),
    ('weights_0', None),
    ('turnover_constraint', None),
    ('turnover_costs', None),
    ('target_return', None),
    ('asset_returns', None),
    ('max_target_portfolio_vol_an', None),
    ('constraint_enforcement_type', ConstraintEnforcementType.FORCED_CONSTRAINTS),
    ('tre_utility_weight', 1.0),
    ('turnover_utility_weight', 0.40),
    ('group_lower_upper_constraints', None),
    ('group_tracking_error_constraint', None),
    ('group_turnover_constraint', None),
    ('sector_deviation_constraints', None),
    ('style_deviation_constraints', None),
    ('benchmark_beta_constraint', None),
)

EXPECTED_METHOD_PARAMETERS = {
    'update_with_valid_tickers': (
        'self', 'valid_tickers', 'total_to_good_ratio', 'weights_0', 'asset_returns',
        'benchmark_weights', 'target_return', 'rebalancing_indicators', 'context',
        'max_relaxation_tol', 'relax_frozen_group_bounds',
    ),
    'set_cvx_exposure_constraints': ('self', 'w', 'exposure_scaler'),
    'set_cvx_all_constraints': (
        'self', 'w', 'covar', 'exposure_scaler', 'covar_factorization',
    ),
    'set_cvx_utility_objective_constraints': (
        'self', 'w', 'alphas', 'covar', 'exposure_scaler', 'covar_factorization',
    ),
    'set_scipy_bounds': ('self', 'covar'),
    'set_scipy_constraints': ('self', 'covar'),
    'set_pyrb_constraints': ('self', 'covar'),
}

EXPECTED_RESIDUAL_FIELDS = (
    'constraint_type',
    'name',
    'actual',
    'lower',
    'upper',
    'violation',
    'tolerance',
    'hard',
    'passed',
)

EXPECTED_EVALUATOR_PARAMETERS = (
    'weights',
    'constraints',
    'covar',
    'covar_factorization',
    'tolerance',
)


@pytest.mark.parametrize('name', ROOT_REEXPORTS)
def test_constraint_root_reexports_are_the_canonical_objects(name: str) -> None:
    """The module, optimisation namespace, and package root expose one object."""
    expected = getattr(constraint_module, name)
    assert getattr(optimization, name) is expected
    assert getattr(op, name) is expected


@pytest.mark.parametrize('name', DIRECT_MODULE_API)
def test_direct_constraint_module_api_remains_available(name: str) -> None:
    """Every established direct-module symbol remains bound after extraction."""
    assert getattr(constraint_module, name) is not None


def test_constraints_dataclass_contract_is_stable() -> None:
    """Field order, defaults, and immutability remain part of construction semantics."""
    actual = tuple((field.name, field.default) for field in fields(Constraints))
    assert actual == EXPECTED_CONSTRAINT_FIELDS
    assert Constraints.__dataclass_params__.frozen is True


def test_legacy_canonical_constraint_pickle_still_loads() -> None:
    """Payloads written through the former facade resolve to the canonical class."""
    implementation_module = Constraints.__module__
    try:
        Constraints.__module__ = 'optimalportfolios.optimization.constraints'
        payload = pickle.dumps(Constraints())
    finally:
        Constraints.__module__ = implementation_module

    restored = pickle.loads(payload)

    assert type(restored) is Constraints
    assert restored == Constraints()


def test_constraint_enforcement_enum_contract_is_stable() -> None:
    """Stored configurations may rely on both enforcement names and values."""
    actual = tuple((member.name, member.value) for member in ConstraintEnforcementType)
    assert actual == (('FORCED_CONSTRAINTS', 1), ('UTILITY_CONSTRAINTS', 2))


@pytest.mark.parametrize('name', ('ConstraintResidual', 'evaluate_constraint_residuals'))
def test_constraint_analytics_reexports_are_the_canonical_objects(name: str) -> None:
    """Every supported analytics path resolves to its package-owned implementation."""
    expected = getattr(constraint_analytics, name)
    assert getattr(constraint_module, name) is expected
    assert getattr(solver_diagnostics, name) is expected
    assert getattr(optimization, name) is expected
    assert getattr(op, name) is expected


def test_constraint_residual_dataclass_contract_is_stable() -> None:
    """Residual field order and immutability survive the owner-module move."""
    residual_type = constraint_analytics.ConstraintResidual
    assert tuple(field.name for field in fields(residual_type)) == EXPECTED_RESIDUAL_FIELDS
    assert residual_type.__dataclass_params__.frozen is True


def test_constraint_residual_evaluator_signature_is_stable() -> None:
    """The evaluator retains its established positional order and defaults."""
    signature = inspect.signature(constraint_analytics.evaluate_constraint_residuals)
    assert tuple(signature.parameters) == EXPECTED_EVALUATOR_PARAMETERS
    assert signature.parameters['weights'].default is inspect.Parameter.empty
    assert signature.parameters['constraints'].default is inspect.Parameter.empty
    assert signature.parameters['covar'].default is None
    assert signature.parameters['covar_factorization'].default is None
    assert signature.parameters['tolerance'].default == 1e-4


@pytest.mark.parametrize(('method_name', 'expected'), EXPECTED_METHOD_PARAMETERS.items())
def test_constraints_method_parameter_order_is_stable(
        method_name: str,
        expected: tuple[str, ...],
) -> None:
    """Backend delegation retains the public method signatures callers introspect."""
    actual = tuple(inspect.signature(getattr(Constraints, method_name)).parameters)
    assert actual == expected


def test_constraint_events_use_the_canonical_logger() -> None:
    """Run diagnostics listen to the original constraint logger by exact name."""
    assert constraint_module.logger.name == 'optimalportfolios.optimization.constraints'


def test_constraint_implementation_uses_the_package_layout() -> None:
    """Constraint owners live below one package with no legacy root modules."""
    optimization_dir = Path(optimization.__file__).resolve().parent
    expected_init = optimization_dir / 'constraints' / '__init__.py'
    actual_file = Path(constraint_module.__file__).resolve()
    private_siblings = tuple(
        sorted(path.name for path in optimization_dir.glob('_constraint_*.py'))
    )
    violations = []
    if constraint_module.__spec__.submodule_search_locations is None:
        violations.append('optimalportfolios.optimization.constraints is not a package')
    if actual_file != expected_init.resolve():
        violations.append(f'constraint facade is {actual_file}, expected {expected_init}')
    if (optimization_dir / 'constraints.py').exists():
        violations.append('legacy optimization/constraints.py still exists')
    if private_siblings:
        violations.append(f'legacy private constraint modules still exist: {private_siblings}')
    assert not violations, '\n'.join(violations)
