"""Tests for factorized covariance use in hard and utility TRE optimization."""
from __future__ import annotations

import inspect
import cvxpy as cvx
import numpy as np
import pandas as pd
import pytest

import optimalportfolios.optimization.taa.maximise_alpha_over_tre as tre_solver
import optimalportfolios.optimization.general.max_sharpe as max_sharpe_solver
import optimalportfolios.optimization.general.quadratic as quadratic_solver
import optimalportfolios.optimization.saa.max_return_target_vol as max_return_solver
import optimalportfolios.optimization.saa.min_variance_target_return as min_variance_solver
import optimalportfolios.optimization.taa.maximise_alpha_with_target_yield as target_yield_solver
from optimalportfolios.config import PortfolioObjective
from optimalportfolios.optimization.config import OptimiserConfig
from optimalportfolios.optimization.constraints import (
    ConstraintEnforcementType,
    Constraints,
    GroupTrackingErrorConstraint,
)
from optimalportfolios.optimization.covar_factorization import (
    DEFAULT_EIGENVALUE_FLOOR,
    factorize_covariance,
)
from optimalportfolios.optimization.solver_diagnostics import OptimizationOutcome


TICKERS = pd.Index(['A', 'B', 'C', 'D'])


def _covar() -> np.ndarray:
    vols = np.array([0.20, 0.12, 0.08, 0.05])
    corr = np.array([
        [1.00, 0.30, 0.10, 0.00],
        [0.30, 1.00, 0.20, 0.10],
        [0.10, 0.20, 1.00, 0.25],
        [0.00, 0.10, 0.25, 1.00],
    ])
    return np.outer(vols, vols) * corr


def _group_loadings() -> pd.DataFrame:
    return pd.DataFrame({
        'Growth': [1.0, 1.0, 0.0, 0.0],
        'Defensive': [0.0, 0.0, 1.0, 1.0],
    }, index=TICKERS)


def _constraints(enforcement: ConstraintEnforcementType) -> Constraints:
    benchmark = pd.Series(0.25, index=TICKERS)
    kwargs = {}
    if enforcement == ConstraintEnforcementType.FORCED_CONSTRAINTS:
        kwargs['group_tracking_error_constraint'] = GroupTrackingErrorConstraint(
            group_loadings=_group_loadings(),
            group_tre_vols=pd.Series({'Growth': 0.10, 'Defensive': 0.10}),
        )
    else:
        kwargs['group_tracking_error_constraint'] = GroupTrackingErrorConstraint(
            group_loadings=_group_loadings(),
            group_tre_utility_weights=pd.Series({'Growth': 25.0, 'Defensive': 25.0}),
        )
    return Constraints(
        min_weights=pd.Series(0.0, index=TICKERS),
        max_weights=pd.Series(0.80, index=TICKERS),
        benchmark_weights=benchmark,
        weights_0=benchmark,
        constraint_enforcement_type=enforcement,
        turnover_utility_weight=None,
        **kwargs,
    )


def test_factorization_reconstructs_stabilized_near_singular_covariance() -> None:
    covar = _covar()
    covar[-1, :] = covar[-2, :]
    covar[:, -1] = covar[:, -2]

    result = factorize_covariance(covar)

    np.testing.assert_allclose(
        result.factor @ result.factor.T, result.covar, rtol=1e-10, atol=1e-12)
    assert float(np.linalg.eigvalsh(result.covar).min()) >= (
        DEFAULT_EIGENVALUE_FLOOR * 0.999)
    assert result.n_eigenvalues_floored >= 1
    assert result.stabilized_min_eigenvalue >= DEFAULT_EIGENVALUE_FLOOR * 0.999
    assert np.isfinite(result.stabilized_condition_number)


def test_wrapper_can_return_a_structured_auditable_outcome() -> None:
    constraints = _constraints(ConstraintEnforcementType.FORCED_CONSTRAINTS)
    weights, outcome = tre_solver.wrapper_maximise_alpha_over_tre(
        pd_covar=pd.DataFrame(_covar(), index=TICKERS, columns=TICKERS),
        alphas=pd.Series([0.03, 0.01, -0.01, 0.0], index=TICKERS),
        benchmark_weights=constraints.benchmark_weights,
        constraints=constraints,
        weights_0=constraints.weights_0,
        optimiser_config=OptimiserConfig(validate_inputs=False),
    )

    assert isinstance(outcome, OptimizationOutcome)
    assert outcome.accepted
    assert outcome.compliant
    assert outcome.constraints is not None
    assert outcome.covar_factorization is not None
    assert isinstance(weights, pd.Series)
    assert any(r.constraint_type == 'group_tracking_error'
               for r in outcome.constraint_residuals)


def test_factorization_accepts_roundoff_but_rejects_material_indefiniteness() -> None:
    roundoff = np.diag([1.0, -5e-12])
    result = factorize_covariance(roundoff)
    assert float(np.linalg.eigvalsh(result.covar).min()) > 0.0

    with pytest.raises(ValueError, match='materially indefinite'):
        factorize_covariance(np.diag([1.0, -1e-5]))


def test_factorized_and_legacy_utility_solutions_are_equivalent() -> None:
    constraints = _constraints(ConstraintEnforcementType.UTILITY_CONSTRAINTS)
    alphas = np.array([0.03, 0.01, -0.01, 0.0])

    factorized = tre_solver.cvx_maximise_tre_utility(
        covar=_covar(),
        constraints=constraints,
        alphas=alphas,
        factorize_covar=True,
    )
    legacy = tre_solver.cvx_maximise_tre_utility(
        covar=_covar(),
        constraints=constraints,
        alphas=alphas,
        factorize_covar=False,
    )

    np.testing.assert_allclose(
        factorized.weights, legacy.weights, rtol=1e-6, atol=1e-7)


def test_factorized_group_utility_preserves_quadratic_value() -> None:
    group_constraint = _constraints(
        ConstraintEnforcementType.UTILITY_CONSTRAINTS
    ).group_tracking_error_constraint
    factorization = factorize_covariance(_covar())
    benchmark = pd.Series(0.25, index=TICKERS)
    weights = np.array([0.40, 0.20, 0.25, 0.15])
    w = cvx.Variable(len(TICKERS))

    legacy = group_constraint.set_cvx_group_tre_utility(
        w=w, benchmark_weights=benchmark, covar=cvx.psd_wrap(_covar()))
    factorized = group_constraint.set_cvx_group_tre_utility(
        w=w,
        benchmark_weights=benchmark,
        covar=cvx.psd_wrap(factorization.covar),
        covar_factorization=factorization,
    )
    w.value = weights

    assert float(factorized.value) == pytest.approx(float(legacy.value), abs=1e-10)


@pytest.mark.parametrize('enforcement', [
    ConstraintEnforcementType.FORCED_CONSTRAINTS,
    ConstraintEnforcementType.UTILITY_CONSTRAINTS,
])
def test_solve_factorizes_once_with_multiple_groups(
        monkeypatch,
        enforcement: ConstraintEnforcementType,
) -> None:
    calls = 0
    original = tre_solver.factorize_covariance

    def counted(covar):
        nonlocal calls
        calls += 1
        return original(covar)

    monkeypatch.setattr(tre_solver, 'factorize_covariance', counted)
    constraints = _constraints(enforcement)
    weights, _ = tre_solver.wrapper_maximise_alpha_over_tre(
        pd_covar=pd.DataFrame(_covar(), index=TICKERS, columns=TICKERS),
        alphas=pd.Series([0.03, 0.01, -0.01, 0.0], index=TICKERS),
        benchmark_weights=constraints.benchmark_weights,
        constraints=constraints,
        weights_0=constraints.weights_0,
        optimiser_config=OptimiserConfig(
            factorize_covar=True, validate_inputs=False),
    )

    assert calls == 1
    assert np.isfinite(weights.to_numpy()).all()
    assert float(weights.sum()) == pytest.approx(1.0, abs=1e-6)


def test_factorization_is_enabled_by_default_and_can_be_disabled(monkeypatch) -> None:
    assert OptimiserConfig().factorize_covar is True

    def unexpected(_covar):
        raise AssertionError('factorize_covariance must not be called')

    monkeypatch.setattr(tre_solver, 'factorize_covariance', unexpected)
    constraints = _constraints(ConstraintEnforcementType.UTILITY_CONSTRAINTS)
    weights, _ = tre_solver.wrapper_maximise_alpha_over_tre(
        pd_covar=pd.DataFrame(_covar(), index=TICKERS, columns=TICKERS),
        alphas=pd.Series([0.03, 0.01, -0.01, 0.0], index=TICKERS),
        benchmark_weights=constraints.benchmark_weights,
        constraints=constraints,
        weights_0=constraints.weights_0,
        optimiser_config=OptimiserConfig(
            factorize_covar=False, validate_inputs=False),
    )

    assert np.isfinite(weights.to_numpy()).all()


def _basic_constraints(**kwargs) -> Constraints:
    values = dict(
        min_weights=pd.Series(0.0, index=TICKERS),
        max_weights=pd.Series(0.80, index=TICKERS),
    )
    values.update(kwargs)
    return Constraints(**values)


def _run_quadratic(optimiser_config: OptimiserConfig) -> pd.Series:
    return quadratic_solver.wrapper_quadratic_optimisation(
        pd_covar=pd.DataFrame(_covar(), index=TICKERS, columns=TICKERS),
        constraints=_basic_constraints(),
        portfolio_objective=PortfolioObjective.MIN_VARIANCE,
        optimiser_config=optimiser_config,
    )[0]


def _run_max_sharpe(optimiser_config: OptimiserConfig) -> pd.Series:
    return max_sharpe_solver.wrapper_maximize_portfolio_sharpe(
        pd_covar=pd.DataFrame(_covar(), index=TICKERS, columns=TICKERS),
        means=pd.Series([0.08, 0.06, 0.04, 0.02], index=TICKERS),
        constraints=_basic_constraints(),
        optimiser_config=optimiser_config,
    )[0]


def _run_min_variance_target_return(
        optimiser_config: OptimiserConfig,
) -> pd.Series:
    return min_variance_solver.wrapper_min_variance_target_return(
        pd_covar=pd.DataFrame(_covar(), index=TICKERS, columns=TICKERS),
        expected_returns=pd.Series([0.08, 0.06, 0.04, 0.02], index=TICKERS),
        target_return=0.04,
        constraints=_basic_constraints(),
        optimiser_config=optimiser_config,
    )[0]


def _run_max_return_target_vol(optimiser_config: OptimiserConfig) -> pd.Series:
    return max_return_solver.wrapper_max_return_target_vol(
        pd_covar=pd.DataFrame(_covar(), index=TICKERS, columns=TICKERS),
        expected_returns=pd.Series([0.08, 0.06, 0.04, 0.02], index=TICKERS),
        target_vol=0.12,
        constraints=_basic_constraints(),
        optimiser_config=optimiser_config,
    )[0]


def _run_alpha_target_return(optimiser_config: OptimiserConfig) -> pd.Series:
    return target_yield_solver.wrapper_maximise_alpha_with_target_return(
        pd_covar=pd.DataFrame(_covar(), index=TICKERS, columns=TICKERS),
        alphas=pd.Series([0.03, 0.01, -0.01, 0.0], index=TICKERS),
        yields=pd.Series([0.08, 0.06, 0.04, 0.02], index=TICKERS),
        target_return=0.04,
        constraints=_basic_constraints(max_target_portfolio_vol_an=0.15),
        optimiser_config=optimiser_config,
    )[0]


@pytest.mark.parametrize(('solver_module', 'run_wrapper'), [
    (quadratic_solver, _run_quadratic),
    (max_sharpe_solver, _run_max_sharpe),
    (min_variance_solver, _run_min_variance_target_return),
    (max_return_solver, _run_max_return_target_vol),
    (target_yield_solver, _run_alpha_target_return),
], ids=[
    'quadratic',
    'max_sharpe_fixed_exposure',
    'min_variance_target_return',
    'max_return_target_vol',
    'alpha_target_return',
])
@pytest.mark.parametrize(('enabled', 'expected_calls'), [(True, 1), (False, 0)])
def test_all_other_supported_solves_factorize_once_and_honour_flag(
        monkeypatch,
        solver_module,
        run_wrapper,
        enabled: bool,
        expected_calls: int,
) -> None:
    calls = 0
    original = solver_module.factorize_covariance

    def counted(covar):
        nonlocal calls
        calls += 1
        return original(covar)

    monkeypatch.setattr(solver_module, 'factorize_covariance', counted)
    weights = run_wrapper(OptimiserConfig(factorize_covar=enabled))

    assert calls == expected_calls
    assert np.isfinite(weights.to_numpy()).all()
    assert float(weights.sum()) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.parametrize('solver_function', [
    tre_solver.cvx_maximise_alpha_over_tre,
    tre_solver.cvx_maximise_tre_utility,
    quadratic_solver.cvx_quadratic_optimisation,
    max_sharpe_solver.cvx_maximize_portfolio_sharpe,
    max_sharpe_solver._cvx_maximize_sharpe_charnes_cooper,
    min_variance_solver.cvx_min_variance_target_return,
    min_variance_solver.cvx_min_variance_target_return_utility,
    max_return_solver.cvx_max_return_target_vol,
    max_return_solver.cvx_max_return_target_vol_utility,
    target_yield_solver.cvx_maximise_alpha_with_target_return,
])
def test_cvx_solver_api_does_not_accept_precomputed_factorization(
        solver_function,
) -> None:
    parameters = inspect.signature(solver_function).parameters
    assert 'factorize_covar' in parameters
    assert 'covar_factorization' not in parameters
