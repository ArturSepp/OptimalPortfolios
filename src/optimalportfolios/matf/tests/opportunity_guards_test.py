"""Opportunity input, risk attribution and solver failure contracts."""
from dataclasses import replace
from types import SimpleNamespace

import cvxpy as cvx
import numpy as np
import pandas as pd
import pytest

from optimalportfolios.matf import opportunity as op
from optimalportfolios.matf.tests.opportunity_test import model
from optimalportfolios.matf.tests.uncertainty_test import model as uncertainty_model
from optimalportfolios.optimization.constraints import (
    ConstraintEnforcementType, Constraints, LinearConstraints,
)


@pytest.mark.parametrize('field', ['betas', 'residual_variances', 'factor_covariance', 'alpha'])
def test_model_requires_labelled_inputs(field):
    """Unlabelled moments cannot silently acquire another asset or factor order."""
    m = model()
    with pytest.raises(ValueError, match=field + ' must be a labelled'):
        replace(m, **{field: getattr(m, field).to_numpy()})


def test_model_rejects_missing_date():
    """Risk snapshots cannot be keyed by an unavailable date."""
    with pytest.raises(ValueError, match='date must be finite'):
        replace(model(), date=pd.NaT)


@pytest.mark.parametrize('premium', [.05, 0.])
def test_factor_access_matches_direct_covariance_solution(premium):
    """A constant-beta model has a closed-form spectrum and systematic Sharpe."""
    m = model()
    summary, factors, spectrum = op.factor_access_metrics(
        m, pd.Series([premium], index=m.betas.columns))
    g = 3 * .09 / .04
    assert spectrum.g.iloc[0] == pytest.approx(g)
    assert spectrum.access_fraction.iloc[0] == pytest.approx(g / (1 + g))
    assert summary['factor_ceiling2'] == pytest.approx(premium**2 / .09)
    assert summary['systematic_sharpe2'] == pytest.approx(3 * premium**2 / (.04 + 3 * .09))
    assert summary['access_loss'] == pytest.approx(
        premium**2 / .09 - 3 * premium**2 / (.04 + 3 * .09))
    assert factors.tangency_direction.iloc[0] == pytest.approx(premium / .09)
    assert factors.premium_sensitivity.iloc[0] == pytest.approx(2 * premium / .09)
    assert factors.conditional_variance.iloc[0] == pytest.approx(.09)
    assert factors.drop_one_gain.iloc[0] == pytest.approx(premium**2 / .09)
    assert factors.hurdle.iloc[0] == pytest.approx(0.)
    if premium:
        assert summary['FPIR'] == pytest.approx(g / (1 + g))
    else:
        assert np.isnan(summary['FPIR'])
    no_premia, empty_factors, same_spectrum = op.factor_access_metrics(m)
    assert no_premia == {} and empty_factors.empty
    pd.testing.assert_frame_equal(spectrum, same_spectrum)


@pytest.mark.parametrize('variance', [0., -.01, np.nan, np.inf])
def test_candidate_gain_rejects_nonpositive_or_nonfinite_residual_risk(variance):
    """A candidate with undefined precision cannot enter the addition calculation."""
    m = model()
    premia = pd.Series(.05, index=m.betas.columns)
    with pytest.raises(ValueError, match='positive residual variance'):
        op.candidate_access_gain(m, premia, pd.Series(.6, index=premia.index), variance)


def test_omitted_reference_reports_absolute_saved_book_risk():
    """The reporting default is a zero book, preserving actual funding and absolute risk."""
    m = model()
    weights = pd.Series([.2, .3, .7], index=m.alpha.index)
    rows, exposures, holdings = op.portfolio_opportunity_metrics(m, weights)
    expected_variance = .09 * weights.sum()**2 + .04 * (weights @ weights)
    assert rows['net_exposure'] == pytest.approx(1.2)
    assert rows['reference_net_exposure'] == 0.
    assert rows['total_te']**2 == pytest.approx(expected_variance)
    assert rows['total_te'] == pytest.approx(rows['portfolio_vol'])
    assert rows['alpha_increment'] == pytest.approx(rows['alpha_level'])
    np.testing.assert_array_equal(holdings.reference, np.zeros(3))
    np.testing.assert_array_equal(exposures.exposure, exposures.active_exposure)


@pytest.mark.parametrize('tolerance', [0., -.001, np.nan, np.inf])
def test_frontier_tolerance_must_be_finite_and_positive(tolerance):
    """Invalid audit tolerances cannot turn a frontier into an accepted portfolio."""
    m = model()
    with pytest.raises(ValueError, match='tolerance must be finite and positive'):
        op.solve_opportunity(m, pd.Series(1 / 3, index=m.alpha.index), tolerance=tolerance)


@pytest.mark.parametrize('case', [
    'covariance_only', 'radius_only', 'negative_radius', 'nan_radius', 'infinite_radius',
    'risk_objective', 'unlabelled_covariance', 'reordered_covariance',
])
def test_uncertainty_frontier_requires_a_joint_aligned_alpha_policy(case):
    """Calibration inputs stay paired, finite and tied to the exact alpha asset axis."""
    m = model()
    covariance = pd.DataFrame(np.eye(3) * .01, index=m.alpha.index, columns=m.alpha.index)
    kwargs = dict(alpha_covariance=covariance, uncertainty_radius=1.)
    if case == 'covariance_only':
        kwargs.pop('uncertainty_radius')
    elif case == 'radius_only':
        kwargs.pop('alpha_covariance')
    elif case in ('negative_radius', 'nan_radius', 'infinite_radius'):
        kwargs['uncertainty_radius'] = {
            'negative_radius': -1., 'nan_radius': np.nan, 'infinite_radius': np.inf}[case]
    elif case == 'risk_objective':
        kwargs['objective'] = 'min_total_te'
    elif case == 'unlabelled_covariance':
        kwargs['alpha_covariance'] = covariance.to_numpy()
    else:
        kwargs['alpha_covariance'] = covariance.iloc[::-1]
    with pytest.raises(ValueError):
        op.solve_opportunity(m, pd.Series(1 / 3, index=m.alpha.index), **kwargs)


def test_orthogonal_uncertainty_penalty_uses_the_joint_projection():
    """Independent Euclidean projection gives the standard error of neutral active alpha."""
    m = uncertainty_model()
    reference = pd.Series(.25, index=m.alpha.index)
    covariance = pd.DataFrame(np.diag([.001, .002, .003, .004]),
                              index=m.alpha.index, columns=m.alpha.index)
    result = op.solve_opportunity(m, reference, objective='orthogonal_upper',
                                 alpha_covariance=covariance, uncertainty_radius=.05)
    assert result.accepted
    design = np.column_stack([np.ones(4), m.betas])
    projector = np.eye(4) - design @ np.linalg.pinv(design)
    delta = (result.weights - reference).to_numpy()
    se = np.sqrt(delta @ projector @ covariance.to_numpy() @ projector.T @ delta)
    assert result.metrics['alpha_estimation_standard_error'] == pytest.approx(se, abs=1e-12)
    assert result.metrics['alpha_uncertainty_penalty'] == pytest.approx(.05 * se, abs=1e-12)
    expected = (projector @ m.alpha.to_numpy()) @ delta - .05 * se
    assert result.objective_value == pytest.approx(expected, abs=1e-10)


@pytest.mark.parametrize('cap', ['total_te', 'factor_te'])
@pytest.mark.parametrize('value', [-.01, np.nan, np.inf])
def test_frontier_risk_caps_are_finite_and_nonnegative(cap, value):
    """Invalid caps must be rejected before any solver can supply holdings."""
    m = model()
    with pytest.raises(ValueError, match=cap + ' must be finite and nonnegative'):
        op.solve_opportunity(m, pd.Series(1 / 3, index=m.alpha.index), **{cap: value})


@pytest.mark.parametrize('spec', [
    Constraints(is_long_only=False), Constraints(min_exposure=0.),
    Constraints(constraint_enforcement_type=ConstraintEnforcementType.UTILITY_CONSTRAINTS),
])
def test_frontier_requires_hard_long_only_unit_funding(spec):
    """A soft or differently funded mandate cannot inherit the frontier certificate."""
    m = model()
    with pytest.raises(ValueError):
        op.solve_opportunity(m, pd.Series(1 / 3, index=m.alpha.index), constraints=spec)


def test_nested_policy_loadings_keep_asset_order_and_enforce_the_bound():
    """A labelled cap binds correctly; reversed nested axes are never treated positionally."""
    m = model()
    reference = pd.Series(1 / 3, index=m.alpha.index)
    loading = pd.DataFrame({'c_cap': [0., 0., 1.]}, index=m.alpha.index)
    bound = pd.Series([.4], index=['c_cap'])
    spec = Constraints(linear_constraints=LinearConstraints(loading, upper=bound))
    result = op.solve_opportunity(m, reference, constraints=spec)
    assert result.accepted and result.weights['c'] == pytest.approx(.4, abs=1e-7)
    misaligned = Constraints(linear_constraints=LinearConstraints(loading.iloc[::-1], upper=bound))
    with pytest.raises(ValueError, match='order'):
        op.solve_opportunity(m, reference, constraints=misaligned)


def test_investability_requires_boolean_values():
    """Integer labels are not interpreted implicitly as a boolean investment policy."""
    m = model()
    with pytest.raises(ValueError, match='labelled boolean Series'):
        op.solve_opportunity(m, pd.Series(1 / 3, index=m.alpha.index),
                             investable=pd.Series([1, 0, 1], index=m.alpha.index))


@pytest.mark.parametrize('objective,metric', [
    ('min_factor_te', 'factor_te'), ('min_total_te', 'total_te'),
])
def test_risk_minimizing_frontiers_attain_the_feasible_zero_risk_reference(objective, metric):
    """A feasible reference proves zero as the global lower bound on a risk norm."""
    m = model()
    result = op.solve_opportunity(m, pd.Series(1 / 3, index=m.alpha.index), objective=objective)
    assert result.accepted
    assert result.objective_value == pytest.approx(0., abs=1e-8)
    assert result.metrics[metric] == pytest.approx(0., abs=1e-8)


def test_score_frontier_reports_its_separate_dimensionless_objective():
    """The simplex optimum places all mass on the largest score and reports its increment."""
    m = model()
    reference = pd.Series(1 / 3, index=m.alpha.index)
    scores = pd.Series([-1., 0., 1.], index=m.alpha.index)
    result = op.solve_opportunity(m, reference, objective='score', scores=scores)
    assert result.accepted
    np.testing.assert_allclose(result.weights, [0., 0., 1.], atol=1e-7)
    assert result.metrics['score_level'] == pytest.approx(1., abs=1e-7)
    assert result.metrics['score_increment'] == pytest.approx(1., abs=1e-7)
    assert result.objective_value == pytest.approx(result.metrics['score_increment'])
    with pytest.raises(ValueError, match='unknown objective'):
        op.solve_opportunity(m, reference, objective='unregistered')


def test_solver_error_returns_no_fallback_portfolio():
    """An unavailable numerical backend produces an explicit failure with no holdings."""
    m = model()
    result = op.solve_opportunity(m, pd.Series(1 / 3, index=m.alpha.index),
                                 solver='OP_TEST_UNAVAILABLE_SOLVER')
    assert not result.accepted and result.weights is None and result.metrics == {}
    assert result.status.startswith('solver_error:')


def test_optimal_status_without_primal_values_cannot_be_accepted(monkeypatch):
    """A solver status alone does not provide an investment portfolio."""
    def status_only(problem, **kwargs):
        """Simulate an optimal status without assigning any primal solution."""
        problem._status = cvx.OPTIMAL
    monkeypatch.setattr(cvx.Problem, 'solve', status_only)
    m = model()
    result = op.solve_opportunity(m, pd.Series(1 / 3, index=m.alpha.index))
    assert result.status == cvx.OPTIMAL
    assert not result.accepted and result.weights is None and result.metrics == {}


def test_post_solve_rejection_never_returns_holdings(monkeypatch):
    """A rejected independent audit cannot be hidden behind an optimal solver status."""
    monkeypatch.setattr(op, 'validate_solution', lambda *args, **kwargs:
                        SimpleNamespace(accepted=False))
    m = model()
    result = op.solve_opportunity(m, pd.Series(1 / 3, index=m.alpha.index))
    assert result.status == 'constraint_violation'
    assert not result.accepted and result.weights is None
    assert result.metrics['net_exposure'] == pytest.approx(1.)
