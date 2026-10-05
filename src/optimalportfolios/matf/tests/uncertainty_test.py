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
