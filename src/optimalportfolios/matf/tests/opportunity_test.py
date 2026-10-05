"""Independent numerical contracts for statistical-alpha opportunity analytics."""
import numpy as np
import pandas as pd
import pytest
from optimalportfolios.matf.opportunity import (
    OpportunityModel, alpha_dispersion, candidate_access_gain, solve_opportunity,
)
from optimalportfolios.matf import achievable_sharpe2


def model():
    """A small model with an exact self-financing, factor-neutral direction."""
    names = pd.Index(['a', 'b', 'c'])
    return OpportunityModel(
        pd.DataFrame([1., 1., 1.], index=names, columns=['market']),
        pd.Series(0.04, index=names),
        pd.DataFrame([[0.09]], index=['market'], columns=['market']),
        pd.Series([-0.02, 0., 0.02], index=names),
    )


def test_joint_dispersion_and_affine_invariance():
    """Compare K with an independent whitened SVD and remove affine alpha."""
    m = model()
    result, detail = alpha_dispersion(m)
    c = np.column_stack([np.ones(3), m.betas]) / np.sqrt(m.residual_variances.to_numpy())[:, None]
    u, s, _ = np.linalg.svd(c, full_matrices=False)
    rank = np.sum(s > s[0] * 1e-12)
    aw = m.alpha.to_numpy() / np.sqrt(m.residual_variances)
    residual = aw - u[:, :rank] @ (u[:, :rank].T @ aw)
    assert result['K'] == pytest.approx(residual @ residual, abs=1e-12)
    assert result['K'] == pytest.approx(0.02)
    assert result['K'] <= min(result['J'], result['S_h2']) + 1e-12
    shifted = OpportunityModel(m.betas, m.residual_variances, m.factor_covariance, m.alpha + 0.3)
    assert alpha_dispersion(shifted)[0]['K'] == pytest.approx(result['K'])
    assert np.max(np.abs(c.T @ (detail.h_joint / np.sqrt(m.residual_variances)))) < 1e-12


def test_frontier_matches_unrestricted_bound_when_interior():
    """An interior simplex solution must attain tau sqrt(K) exactly."""
    m = model()
    w0 = pd.Series(1 / 3, index=m.betas.index)
    tau = 0.03
    result = solve_opportunity(m, w0, total_te=tau, factor_te=0., objective='upper')
    assert result.accepted
    assert result.metrics['alpha_increment'] == pytest.approx(tau * np.sqrt(0.02), abs=2e-8)
    assert result.metrics['total_te'] <= tau + 1e-7
    assert result.weights.sum() == pytest.approx(1., abs=1e-8)
    assert result.weights.min() >= -1e-8
    lower = solve_opportunity(m, w0, total_te=tau, factor_te=0., objective='lower')
    assert lower.metrics['alpha_increment'] == pytest.approx(
        -result.metrics['alpha_increment'], abs=2e-8)


def test_frontier_mask_infeasibility_and_no_reference_normalization():
    """A reference-only asset is retained in risk while excluded from holdings."""
    m = model()
    w0 = pd.Series([1., 0., 0.], index=m.betas.index)
    allowed = pd.Series([False, True, True], index=m.betas.index)
    result = solve_opportunity(m, w0, investable=allowed, total_te=0.001, factor_te=0.)
    assert not result.accepted
    assert result.weights is None
    with pytest.raises(ValueError, match='unit'):
        solve_opportunity(m, w0 * 0.9)


def test_candidate_gain_matches_direct_refit_free_difference():
    """Sherman-Morrison gain agrees with the canonical expanded-universe value."""
    m = model()
    premia = pd.Series([0.05], index=m.betas.columns)
    candidate = pd.Series([0.6], index=m.betas.columns)
    gain = candidate_access_gain(m, premia, candidate, 0.02)
    expanded_b = pd.concat([m.betas, pd.DataFrame([candidate], index=['new'])])
    expanded_d = pd.concat([m.residual_variances, pd.Series([0.02], index=['new'])])
    direct = achievable_sharpe2(premia, m.factor_covariance, expanded_b, expanded_d)
    direct -= achievable_sharpe2(premia, m.factor_covariance, m.betas, m.residual_variances)
    assert gain == pytest.approx(direct, abs=1e-12)


def test_labels_variances_and_factor_covariance_are_not_repaired():
    """Input failures cannot silently change the investment model."""
    m = model()
    with pytest.raises(ValueError):
        OpportunityModel(m.betas, m.residual_variances.iloc[::-1], m.factor_covariance, m.alpha)
    with pytest.raises(ValueError):
        OpportunityModel(m.betas, m.residual_variances * 0, m.factor_covariance, m.alpha)
    with pytest.raises(ValueError):
        OpportunityModel(m.betas, m.residual_variances, m.factor_covariance * 0, m.alpha)


def test_orthogonal_frontier_uses_joint_funding_projection():
    """Under loose factor limits O5 optimizes h_C, not the uncentered h_B."""
    m = model()
    b = m.betas.copy()
    b.iloc[:, 0] = [0., 1., 2.]
    m = OpportunityModel(b, m.residual_variances, m.factor_covariance,
                         pd.Series([1., 2., 4.], index=b.index))
    w0 = pd.Series(1 / 3, index=b.index)
    _, detail = alpha_dispersion(m)
    result = solve_opportunity(m, w0, objective='orthogonal_upper')
    assert result.accepted
    expected = detail.h_joint.max() - detail.h_joint @ w0
    assert result.objective_value == pytest.approx(expected, abs=1e-7)


def test_factor_risk_uses_covariance_and_portfolio_attribution_adds():
    """Audit risk independently with a correlated two-factor quadratic form."""
    from optimalportfolios.matf.opportunity import portfolio_opportunity_metrics
    names = pd.Index(['a', 'b', 'c', 'd'])
    factors = pd.Index(['equity', 'rates'])
    b = pd.DataFrame([[1., 0.], [0., 1.], [.8, .4], [.2, .1]], index=names, columns=factors)
    f = pd.DataFrame([[.04, -.012], [-.012, .01]], index=factors, columns=factors)
    m = OpportunityModel(b, pd.Series(.01, index=names), f,
                         pd.Series([.03, -.02, .08, 0.], index=names))
    w0 = pd.Series(.25, index=names)
    solved = solve_opportunity(m, w0, total_te=.06, factor_te=.004)
    assert solved.accepted
    delta = (solved.weights - w0).to_numpy()
    df = b.to_numpy().T @ delta
    factor_variance = float(df @ f @ df)
    residual_variance = float(.01 * (delta @ delta))
    assert solved.metrics['factor_te']**2 == pytest.approx(factor_variance, abs=1e-12)
    assert solved.metrics['total_te']**2 == pytest.approx(
        factor_variance + residual_variance, abs=1e-12)
    rows, exposures, _ = portfolio_opportunity_metrics(m, solved.weights, w0)
    assert rows['factor_te'] <= .004 + 1e-7
    assert rows['alpha_increment'] == pytest.approx(
        rows['orthogonal_alpha_increment'] + rows['factor_aligned_alpha_increment'])
    assert exposures.active_variance_contribution.sum() == pytest.approx(factor_variance)


def test_mandate_caps_and_nested_universe_monotonicity():
    """Existing OP position caps bind; adding investability cannot reduce the optimum."""
    from optimalportfolios.optimization.constraints import Constraints
    m = model()
    w0 = pd.Series(1 / 3, index=m.betas.index)
    cap = pd.Series([1., 1., .4], index=m.betas.index)
    spec = Constraints(max_weights=cap)
    small = solve_opportunity(m, w0, constraints=spec,
                              investable=pd.Series([True, True, False], index=cap.index))
    large = solve_opportunity(m, w0, constraints=spec)
    assert small.accepted and large.accepted
    assert large.weights['c'] == pytest.approx(.4, abs=1e-6)
    assert large.metrics['alpha_increment'] >= small.metrics['alpha_increment'] - 1e-7
    with pytest.raises(ValueError, match='order'):
        solve_opportunity(m, w0, constraints=Constraints(max_weights=cap.iloc[::-1]))


def test_all_negative_alpha_and_zero_budget():
    """The positive part is not a signed optimum; zero TE preserves the reference."""
    m = model()
    m = OpportunityModel(m.betas, m.residual_variances, m.factor_covariance, m.alpha - .1)
    rows, detail = alpha_dispersion(m)
    assert rows['A_plus'] == 0.
    assert rows['residual_reference_signed_sharpe'] == pytest.approx(-.4)
    assert detail.residual_reference_weight['c'] == 1.
    w0 = pd.Series(1 / 3, index=m.betas.index)
    solved = solve_opportunity(m, w0, total_te=0.)
    assert solved.accepted
    np.testing.assert_allclose(solved.weights, w0, atol=1e-8)


def test_normalized_capacity_and_common_level_dispersion():
    """Replicating comparable independent modeled assets scales totals, not RMS."""
    m = model()
    base, _ = alpha_dispersion(m)
    b = pd.concat([m.betas, m.betas], ignore_index=True)
    doubled = OpportunityModel(b, pd.concat([m.residual_variances] * 2, ignore_index=True),
                               m.factor_covariance, pd.concat([m.alpha] * 2, ignore_index=True))
    other, _ = alpha_dispersion(doubled)
    for name in ('A', 'J', 'K'):
        assert other[name] == pytest.approx(2 * base[name])
        assert other[name + '_rms'] == pytest.approx(base[name + '_rms'])
    constant = OpportunityModel(m.betas, m.residual_variances, m.factor_covariance,
                                pd.Series(.02, index=m.betas.index))
    result, _ = alpha_dispersion(constant)
    assert result['A'] > 0
    assert result['J'] == pytest.approx(0., abs=1e-14)
    assert result['J_weighted_alpha_std'] == pytest.approx(0., abs=1e-14)
