"""Return-floor correctness across transformed and direct Sharpe solvers."""
import cvxpy as cvx
import numpy as np
import pandas as pd
import pytest

from optimalportfolios import Constraints, cvx_maximize_portfolio_sharpe


def _inputs():
    """Return a fixed core and two overlays with an attainable signed floor."""
    names = pd.Index(['core', 'growth', 'hedge'])
    covar = np.array([[0.01, 0.002, -0.004],
                      [0.002, 0.0225, 0.0], [-0.004, 0.0, 0.01]])
    means = np.array([0.035, 0.06, -0.02])
    coefficients = pd.Series([-0.08, -0.04, 0.12], index=names)
    base = Constraints(min_weights=pd.Series([1.0, 0.0, 0.0], index=names),
                       max_weights=pd.Series(1.0, index=names),
                       min_exposure=2.0, max_exposure=2.0)
    return covar, means, coefficients, base


@pytest.mark.parametrize('floor', [-0.048, 0.0, 0.005])
def test_direct_floor_matches_independent_scaled_problem(floor):
    """Both signs of a nonzero bound must solve the intended economic problem."""
    covar, means, a, base = _inputs()
    outcome = cvx_maximize_portfolio_sharpe(
        covar, means, base.copy(asset_returns=a, target_return=floor))
    y = cvx.Variable(3)
    k = cvx.Variable(nonneg=True)
    oracle = cvx.Problem(cvx.Minimize(cvx.quad_form(y, covar)), [
        y >= k * base.min_weights.to_numpy(), y <= k * base.max_weights.to_numpy(),
        cvx.sum(y) == 2.0 * k, means @ y == 2.0, a.to_numpy() @ y >= floor * k,
    ])
    oracle.solve(solver='CLARABEL')
    assert outcome.accepted
    np.testing.assert_allclose(outcome.weights, y.value / k.value, atol=1e-6)
    homogeneous = cvx_maximize_portfolio_sharpe(
        covar, means, base.copy(asset_returns=a - floor / 2.0, target_return=0.0))
    np.testing.assert_allclose(outcome.weights, homogeneous.weights, atol=1e-7)


def test_floor_is_invariant_to_return_units():
    """Decimal and percentage returns encode the same portfolio policy."""
    covar, means, a, base = _inputs()
    weights = []
    for scale in (1.0, 100.0):
        outcome = cvx_maximize_portfolio_sharpe(
            covar * scale ** 2, means * scale,
            base.copy(asset_returns=a * scale, target_return=-0.048 * scale))
        assert outcome.accepted
        weights.append(outcome.weights)
    np.testing.assert_allclose(*weights, atol=1e-6)


@pytest.mark.parametrize('floor', [-0.048, 0.0, 0.005])
def test_exposure_band_enforces_return_floor(floor):
    """SLSQP must enforce the floor rather than rely on post-solve rejection."""
    covar, means, a, base = _inputs()
    outcome = cvx_maximize_portfolio_sharpe(
        covar, means, base.copy(min_exposure=1.8, asset_returns=a, target_return=floor))
    assert outcome.accepted
    assert a.to_numpy() @ outcome.weights >= floor - 1e-7


def test_scipy_floor_requires_coefficients():
    """A configured floor cannot silently vanish when its inputs are absent."""
    with pytest.raises(ValueError, match='asset_returns'):
        Constraints(target_return=0.02).set_scipy_constraints(np.eye(2))
