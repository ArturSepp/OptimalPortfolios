"""Independent uncertainty propagation and unchanged opportunity identities."""
import numpy as np
import pandas as pd
import pytest
from optimalportfolios.matf.opportunity import OpportunityModel, alpha_dispersion, solve_opportunity
from optimalportfolios.matf.uncertainty import (
    alpha_uncertainty_metrics, removal_uncertainty, fixed_portfolio_alpha_interval,
)


def model():
    """A four-asset model with a nontrivial funding/factor projector."""
    ids = pd.Index(['a', 'b', 'c', 'd'])
    return OpportunityModel(pd.DataFrame([.5, 1., 1., 1.5], index=ids, columns=['f']),
                            pd.Series(.04, index=ids),
                            pd.DataFrame([[.09]], index=['f'], columns=['f']),
                            pd.Series([-.02, .01, .03, .015], index=ids))


def test_projection_covariance_and_trace_noise():
    """Compare with an independently calculated ordinary projector for constant D."""
    m = model()
    v = pd.DataFrame(np.eye(4)*.0001, index=m.alpha.index, columns=m.alpha.index)
    result = alpha_uncertainty_metrics(m, v, draws=400, seed=7)
    design = np.column_stack([np.ones(4), m.betas])
    p = np.eye(4)-design @ np.linalg.pinv(design)
    np.testing.assert_allclose(result['projected_covariance'], p @ v @ p.T, atol=1e-16)
    assert result['summary'].loc['K', 'noise'] == pytest.approx(.005)
    assert result['contributors'].K_contribution.sum() == pytest.approx(
        alpha_dispersion(m)[0]['K'])
    for name in ['A', 'J', 'S_h2', 'K']:
        assert result['summary'].loc[name, 'observed'] == pytest.approx(
            alpha_dispersion(m)[0][name])


def test_long_only_capacity_bound_uses_full_error_norm_and_not_selected_support():
    """Independently reconcile positive-part points and correlated error radii."""
    from factorlasso import gaussian_quadratic_quantile
    m = model()
    v = pd.DataFrame(.0001*(.6*np.ones((4, 4))+.4*np.eye(4)),
                     index=m.alpha.index, columns=m.alpha.index)
    result = alpha_uncertainty_metrics(m, v, draws=3, quadratic_method='weighted_chi2')
    row = result['summary'].loc['K_star']
    d = m.residual_variances.to_numpy()
    expected = np.sum(np.maximum(m.alpha, 0.)**2/d)
    eigenvalues = np.linalg.eigvalsh(v.to_numpy()/np.sqrt(np.outer(d, d)))
    radius = np.sqrt(gaussian_quadratic_quantile(eigenvalues))
    assert row.observed == pytest.approx(expected)
    assert row.error_norm_radius == pytest.approx(radius)
    assert row.lower == pytest.approx(max(0., np.sqrt(expected)-radius)**2)
    assert row.upper == pytest.approx((np.sqrt(expected)+radius)**2)
    assert np.isnan(row.noise_adjusted)
    np.testing.assert_array_equal(result['scenario_metrics'].K_star,
                                  result['scenario_metrics'].A_plus)
    diagonal = v*0.+np.diag(np.diag(v))
    other = alpha_uncertainty_metrics(m, diagonal, draws=3, quadratic_method='weighted_chi2')
    assert not np.isclose(row.error_norm_radius,
                          other['summary'].loc['K_star', 'error_norm_radius'])


@pytest.mark.parametrize('level', [-.02, 0., .02])
def test_common_alpha_and_zero_uncertainty_have_exact_capacity_bounds(level):
    """K removes a common level; K_star includes its positive part and never shorts."""
    m = model()
    m = OpportunityModel(m.betas, m.residual_variances, m.factor_covariance,
                          pd.Series(level, index=m.alpha.index))
    v = pd.DataFrame(0., index=m.alpha.index, columns=m.alpha.index)
    result = alpha_uncertainty_metrics(m, v, draws=2, quadratic_method='weighted_chi2')
    rows = result['summary']
    assert rows.loc['K', 'observed'] == pytest.approx(0., abs=1e-15)
    assert rows.loc['K_star', 'observed'] == pytest.approx(4*max(0., level)**2/.04)
    np.testing.assert_allclose(rows.lower, rows.observed, atol=1e-15)
    np.testing.assert_allclose(rows.upper, rows.observed, atol=1e-15)


def test_capacity_intervals_known_covariance_gaussian_coverage():
    """Independent Monte Carlo checks the norm geometry, including sign crossings."""
    from scipy.linalg import null_space
    from factorlasso import gaussian_quadratic_quantile
    m = model()
    d = m.residual_variances.to_numpy()
    design = np.column_stack([np.ones(4), m.betas])
    basis = null_space(design.T/np.sqrt(d))
    metric = (basis @ basis.T)/np.sqrt(np.outer(d, d))
    v = .0002*(.5*np.ones((4, 4))+.5*np.eye(4))
    root = np.linalg.cholesky(v)
    rng = np.random.default_rng(2026100501)
    errors = rng.normal(size=(50000, 4)) @ root.T
    radii = {}
    for label, q in [('K', metric), ('K_star', np.diag(1/d))]:
        eig = np.maximum(np.linalg.eigvalsh(root.T @ q @ root), 0.)
        eig = eig[eig > 1e-14]
        radii[label] = np.sqrt(gaussian_quadratic_quantile(eig))
    for alpha in [np.zeros(4), np.array([-.04, 0., .01, .06]), np.full(4, -.04)]:
        estimates = alpha+errors
        for label in radii:
            if label == 'K':
                observed = np.sum((estimates/np.sqrt(d) @ basis)**2, axis=1)
                target = np.sum((alpha/np.sqrt(d) @ basis)**2)
            else:
                observed = np.sum(np.maximum(estimates, 0.)**2/d, axis=1)
                target = np.sum(np.maximum(alpha, 0.)**2/d)
            lower = np.maximum(0., np.sqrt(observed)-radii[label])**2
            upper = (np.sqrt(observed)+radii[label])**2
            assert np.mean((lower <= target+1e-14) & (target <= upper+1e-14)) >= .945


def test_paired_removal_and_null_bounds():
    """Paired scenario removal agrees with direct subset evaluation, including rank changes."""
    m = model()
    v = pd.DataFrame(np.eye(4)*.0001, index=m.alpha.index, columns=m.alpha.index)
    result = alpha_uncertainty_metrics(m, v, draws=20, seed=8)
    removal = removal_uncertainty(m, v, result['alpha_scenarios'])
    for row in removal.itertuples():
        ids = m.alpha.index[m.alpha.index != row.asset]
        sub = OpportunityModel(m.betas.loc[ids], m.residual_variances.loc[ids],
                               m.factor_covariance, m.alpha.loc[ids])
        assert row.observed == pytest.approx(alpha_dispersion(m)[0]['K']
                                            - alpha_dispersion(sub)[0]['K'], abs=1e-13)
    assert (removal.lower >= 0).all()
    zero = OpportunityModel(m.betas, m.residual_variances, m.factor_covariance, m.alpha*0)
    assert alpha_uncertainty_metrics(zero, v, draws=5)['summary'].loc['K', 'lower'] == 0


def test_fixed_portfolio_interval_matches_linear_propagation():
    """Uncertainty is for active alpha, with reference and book kept at unit exposure."""
    m = model()
    v = pd.DataFrame(np.eye(4)*.0001, index=m.alpha.index, columns=m.alpha.index)
    w = pd.Series([.2, .2, .4, .2], index=m.alpha.index)
    w0 = pd.Series(.25, index=m.alpha.index)
    result = fixed_portfolio_alpha_interval(m, v, w, w0)
    delta = w-w0
    assert result['estimate'] == pytest.approx(delta @ m.alpha)
    assert result['standard_error'] == pytest.approx(np.sqrt(delta @ v @ delta))
    with pytest.raises(ValueError):
        alpha_uncertainty_metrics(m, v.iloc[::-1], draws=5)


def test_conservative_frontier_reduces_to_baseline_and_can_prefer_reference():
    """A joint uncertainty penalty preserves feasibility and suppresses noisy active bets."""
    m = model()
    w0 = pd.Series(.25, index=m.alpha.index)
    v = pd.DataFrame(np.eye(4)*.01, index=m.alpha.index, columns=m.alpha.index)
    baseline = solve_opportunity(m, w0, total_te=.03, factor_te=.005)
    zero = solve_opportunity(m, w0, total_te=.03, factor_te=.005,
                             alpha_covariance=v, uncertainty_radius=0.)
    np.testing.assert_allclose(zero.weights, baseline.weights, atol=1e-7)
    robust = solve_opportunity(m, w0, total_te=.03, factor_te=.005,
                               alpha_covariance=v, uncertainty_radius=3.)
    assert robust.accepted
    assert robust.weights.sum() == pytest.approx(1., abs=1e-8)
    assert robust.weights.min() >= 0
    np.testing.assert_allclose(robust.weights, w0, atol=2e-7)
    assert robust.metrics['alpha_uncertainty_penalty'] >= 0


def test_fixed_policy_scenarios_preserve_exact_coupling():
    """Passing paired scenarios preserves joint admission-policy comparisons exactly."""
    m = model()
    v = pd.DataFrame(np.eye(4)*.0001, index=m.alpha.index, columns=m.alpha.index)
    raw = alpha_uncertainty_metrics(m, v, draws=20, seed=1)
    weights = np.array([0., .5, 1., 1.])
    admitted = OpportunityModel(m.betas, m.residual_variances, m.factor_covariance, m.alpha*weights)
    paired = raw['alpha_scenarios']*weights
    result = alpha_uncertainty_metrics(admitted, v*np.outer(weights, weights),
                                       draws=20, scenarios=paired)
    np.testing.assert_array_equal(result['alpha_scenarios'], paired)


def test_non_normal_intervals_route_to_all_reported_endpoints():
    """Externally calibrated asymmetric intervals survive OP accounting unchanged."""
    from factorlasso import linear_confidence_intervals
    m = model()
    v = pd.DataFrame(np.eye(4)*.0001, index=m.alpha.index, columns=m.alpha.index)
    baseline = alpha_uncertainty_metrics(m, v, draws=5)
    intervals = {}
    for key, mean, covariance in [
            ('alpha', m.alpha, v),
            ('projected', baseline['contributors'].h_joint, baseline['projected_covariance'])]:
        values = linear_confidence_intervals(mean, covariance)
        values['lower'] = np.asarray(mean)-.123
        values['upper'] = np.asarray(mean)+.456
        values['interval_method'] = 'synthetic_calibration'
        intervals[key] = pd.DataFrame(values, index=m.alpha.index)
    result = alpha_uncertainty_metrics(m, v, draws=5, intervals=intervals,
                                       quadratic_method='weighted_chi2')
    np.testing.assert_array_equal(result['contributors'].alpha_lower, intervals['alpha'].lower)
    np.testing.assert_array_equal(result['contributors'].projected_upper,
                                   intervals['projected'].upper)
    assert result['contributors'].projected_interval_method.eq('synthetic_calibration').all()
    with pytest.raises(ValueError):
        alpha_uncertainty_metrics(m, v, draws=5,
                                  intervals={'alpha': intervals['alpha'].iloc[::-1]})
