"""Independent numerical contracts for constrained supplied-input opportunities."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from optimalportfolios.optimization.opportunity import (
    OpportunityModel, portfolio_opportunity_metrics, solve_opportunity,
)


def model():
    """A supplied three-asset covariance with a neutral transfer direction."""
    ids = pd.Index(['a', 'b', 'c'])
    b = pd.DataFrame(1., index=ids, columns=['market'])
    f = pd.DataFrame([[.09]], index=b.columns, columns=b.columns)
    residual = pd.DataFrame(np.eye(3)*.04, index=ids, columns=ids)
    alpha = pd.Series([-.02, 0., .02], index=ids)
    return OpportunityModel(b @ f @ b.T + residual, alpha,
                            b, f, residual, alpha.copy())


def test_frontier_matches_unrestricted_bound_when_interior():
    """An interior simplex solution attains the independent tau sqrt(.02) bound."""
    m = model()
    w0 = pd.Series(1/3, index=m.alpha.index)
    tau = .03
    result = solve_opportunity(m, w0, total_te=tau, factor_te=0.)
    assert result.accepted
    assert result.metrics['alpha_increment'] == pytest.approx(tau*np.sqrt(.02), abs=2e-8)
    assert result.metrics['total_te'] <= tau + 1e-7
    assert result.weights.sum() == pytest.approx(1., abs=1e-8)
    assert result.weights.min() >= -1e-8
    lower = solve_opportunity(m, w0, total_te=tau, factor_te=0., objective='lower')
    assert lower.metrics['alpha_increment'] == pytest.approx(
        -result.metrics['alpha_increment'], abs=2e-8)


def test_covariance_only_has_identical_weights_and_reports_available_risk():
    """A full covariance needs no factor decomposition for an alpha/TE frontier."""
    m = model()
    direct = OpportunityModel(m.covariance, m.alpha)
    reference = pd.Series(1/3, index=m.alpha.index)
    factor = solve_opportunity(m, reference, total_te=.03)
    plain = solve_opportunity(direct, reference, total_te=.03)
    assert factor.accepted and plain.accepted
    np.testing.assert_allclose(factor.weights, plain.weights, atol=1e-10)
    delta = plain.weights-reference
    assert plain.metrics['total_te']**2 == pytest.approx(delta @ m.covariance @ delta)
    assert 'factor_te' not in plain.metrics and 'residual_te' not in plain.metrics
    assert portfolio_opportunity_metrics(direct, plain.weights)[1].empty
    with pytest.raises(ValueError, match='require supplied factor'):
        solve_opportunity(direct, reference, factor_te=0.)
    with pytest.raises(ValueError, match='require supplied orthogonal_alpha'):
        solve_opportunity(direct, reference, objective='orthogonal_upper')


def test_frontier_mask_infeasibility_and_no_reference_normalization():
    """Reference-only assets remain in risk without normalizing the saved book."""
    m = model()
    reference = pd.Series([1., 0., 0.], index=m.alpha.index)
    allowed = pd.Series([False, True, True], index=m.alpha.index)
    result = solve_opportunity(m, reference, investable=allowed, total_te=.001, factor_te=0.)
    assert not result.accepted and result.weights is None
    with pytest.raises(ValueError, match='unit'):
        solve_opportunity(m, reference*.9)


def test_prepared_orthogonal_alpha_is_used_without_refitting():
    """The supplied vector defines the objective even when unlike a local GLS fit."""
    m = model()
    prepared = pd.Series([.1, -.2, .4], index=m.alpha.index)
    m = replace(m, orthogonal_alpha=prepared)
    reference = pd.Series(1/3, index=m.alpha.index)
    result = solve_opportunity(m, reference, objective='orthogonal_upper')
    assert result.accepted
    assert result.objective_value == pytest.approx(prepared.max()-prepared @ reference, abs=1e-7)


def test_factor_and_full_residual_risk_match_independent_quadratics():
    """Correlated factor covariance and residual covariance are both retained."""
    ids = pd.Index(['a', 'b', 'c', 'd'])
    factors = pd.Index(['equity', 'rates'])
    b = pd.DataFrame([[1., 0.], [0., 1.], [.8, .4], [.2, .1]],
                     index=ids, columns=factors)
    f = pd.DataFrame([[.04, -.012], [-.012, .01]], index=factors, columns=factors)
    residual = pd.DataFrame(.01*(.4*np.ones((4, 4))+.6*np.eye(4)),
                            index=ids, columns=ids)
    m = OpportunityModel(b @ f @ b.T + residual,
        pd.Series([.03, -.02, .08, 0.], index=ids), b, f, residual)
    reference = pd.Series(.25, index=ids)
    result = solve_opportunity(m, reference, total_te=.06, factor_te=.004)
    assert result.accepted
    delta = result.weights-reference
    df = b.T @ delta
    fv, rv = float(df @ f @ df), float(delta @ residual @ delta)
    assert result.metrics['factor_te']**2 == pytest.approx(fv, abs=1e-12)
    assert result.metrics['residual_te']**2 == pytest.approx(rv, abs=1e-12)
    assert result.metrics['total_te']**2 == pytest.approx(fv+rv, abs=1e-12)
    _, exposures, _ = portfolio_opportunity_metrics(m, result.weights, reference)
    assert exposures.active_variance_contribution.sum() == pytest.approx(fv)


def test_mandate_caps_and_nested_universe_monotonicity():
    """Existing OP mandate caps bind and a larger admissible universe cannot hurt."""
    from optimalportfolios.optimization.constraints import Constraints
    m = model()
    reference = pd.Series(1/3, index=m.alpha.index)
    cap = pd.Series([1., 1., .4], index=m.alpha.index)
    spec = Constraints(max_weights=cap)
    small = solve_opportunity(m, reference, constraints=spec,
                              investable=pd.Series([True, True, False], index=cap.index))
    large = solve_opportunity(m, reference, constraints=spec)
    assert small.accepted and large.accepted
    assert large.weights['c'] == pytest.approx(.4, abs=1e-6)
    assert large.metrics['alpha_increment'] >= small.metrics['alpha_increment']-1e-7
    with pytest.raises(ValueError, match='order'):
        solve_opportunity(m, reference, constraints=Constraints(max_weights=cap.iloc[::-1]))


def test_zero_budget_preserves_reference_with_negative_alpha():
    """An exact zero-risk budget cannot be relaxed to chase a different alpha."""
    m = replace(model(), alpha=model().alpha-.1)
    reference = pd.Series(1/3, index=m.alpha.index)
    solved = solve_opportunity(m, reference, total_te=0.)
    assert solved.accepted
    np.testing.assert_allclose(solved.weights, reference, atol=1e-8)


def test_prepared_uncertainty_penalty_preserves_reference():
    """The same solver uses supplied estimation covariance without fitting it."""
    m = model()
    reference = pd.Series(1/3, index=m.alpha.index)
    v = pd.DataFrame(np.eye(3)*.01, index=m.alpha.index, columns=m.alpha.index)
    baseline = solve_opportunity(m, reference, total_te=.03)
    zero = solve_opportunity(m, reference, total_te=.03,
                             alpha_covariance=v, uncertainty_radius=0.)
    np.testing.assert_allclose(zero.weights, baseline.weights, atol=1e-7)
    robust = solve_opportunity(m, reference, total_te=.03,
                               alpha_covariance=v, uncertainty_radius=3.)
    assert robust.accepted
    np.testing.assert_allclose(robust.weights, reference, atol=2e-7)


@pytest.mark.parametrize('field', ['covariance', 'residual_covariance'])
def test_nonconvex_or_misaligned_inputs_are_rejected(field):
    """Neither asset identity nor numerical convexity can be silently repaired."""
    m = model()
    with pytest.raises(ValueError):
        replace(m, **{field: getattr(m, field).iloc[::-1]})
    with pytest.raises(ValueError, match='indefinite'):
        replace(m, **{field: -getattr(m, field)})


def test_component_reconciliation_and_owned_inputs():
    """Optional decomposition must describe the supplied covariance and owns copies."""
    m = model()
    with pytest.raises(ValueError, match='reconcile'):
        replace(m, residual_covariance=m.residual_covariance*2)
    with pytest.raises(ValueError, match='together'):
        replace(m, factor_covariance=None)
    covariance = m.covariance.copy()
    alpha = m.alpha.copy()
    plain = OpportunityModel(covariance, alpha)
    covariance.iloc[0, 0] += 100
    alpha.iloc[0] += 100
    pd.testing.assert_frame_equal(plain.covariance, m.covariance)
    pd.testing.assert_series_equal(plain.alpha, m.alpha)
