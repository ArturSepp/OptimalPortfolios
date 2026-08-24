"""Compatibility contracts for the portfolio-constraint module boundary.

The constraint implementation is intentionally split behind
``optimalportfolios.optimization.constraints``.  These tests freeze the import aliases and
constructor/method shapes that callers see, so moving an implementation cannot silently turn a
structural refactor into an API change.
"""
import inspect
from dataclasses import fields

import pytest

import optimalportfolios as op
import optimalportfolios.optimization as optimization
import optimalportfolios.optimization.constraints as constraint_module
from optimalportfolios.optimization.constraints import ConstraintEnforcementType, Constraints


ROOT_REEXPORTS = (
    'ConstraintEnforcementType',
    'Constraints',
    'GroupLowerUpperConstraints',
    'GroupTrackingErrorConstraint',
    'GroupTurnoverConstraint',
    'compute_eligible_rebalancing_bounds',
    'merge_group_lower_upper_constraints',
)

DIRECT_MODULE_API = (
    'BenchmarkBetaConstraint',
    'BenchmarkDeviationConstraints',
    'ConstraintEnforcementType',
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


def test_constraint_enforcement_enum_contract_is_stable() -> None:
    """Stored configurations may rely on both enforcement names and values."""
    actual = tuple((member.name, member.value) for member in ConstraintEnforcementType)
    assert actual == (('FORCED_CONSTRAINTS', 1), ('UTILITY_CONSTRAINTS', 2))


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
