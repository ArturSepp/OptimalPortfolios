"""Full investment and absolute portfolio bounds must describe one feasible set.

The old absolute-bound lambda search has no root for the two-asset hedge below,
although its fully invested feasible set is nonempty. Independent conic solves
use the volatility/log lift, not ADMM's quadratic/log coordinate updates.
"""
import cvxpy as cp
import numpy as np
import pytest

import optimalportfolios.optimization.risk_allocation.risk_budgeting_solver as rb_solver
from optimalportfolios.optimization.risk_allocation.risk_budgeting_solver import (
    solve_constrained_risk_budgeting,
)


def conic_reference(covar, budgets, bounds, c_rows=None, c_lhs=None):
    """Independently solve the homogeneous volatility/log formulation with CLARABEL."""
    y = cp.Variable(len(budgets))
    total = cp.sum(y)
    root = np.linalg.cholesky(covar / np.max(np.diag(covar))).T
    positive = budgets > 0.0
    constraints = [y >= bounds[:, 0] * total, y <= bounds[:, 1] * total]
    if c_rows is not None:
        constraints.append(c_rows @ y <= c_lhs * total)
    objective = cp.norm(root @ y) - cp.sum(cp.multiply(
        budgets[positive] / budgets.sum(), cp.log(y[positive])))
    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve(solver='CLARABEL', tol_gap_abs=1e-11, tol_gap_rel=1e-11,
                  tol_feas=1e-11, max_iter=500)
    assert problem.status in ('optimal', 'optimal_inaccurate')
    return y.value / y.value.sum()


@pytest.mark.parametrize('as_group', [False, True])
def test_negative_correlation_floor_has_a_fully_invested_solution(as_group):
    """A min-risk endpoint at 150% must not reject the feasible 60/40 portfolio."""
    covar = np.array([[0.04, -0.015], [-0.015, 0.01]])
    bounds = np.array([[0.6, 0.9], [0.1, 0.9]])
    rows, lhs = None, None
    if as_group:
        bounds[0, 0] = 0.0
        rows, lhs = np.array([[-1.0, 0.0]]), np.array([-0.6])
    weights, _ = solve_constrained_risk_budgeting(
        covar, bounds=bounds, c_rows=rows, c_lhs=lhs)
    np.testing.assert_allclose(weights, [0.6, 0.4], atol=1e-8)
    assert weights.sum() == pytest.approx(1.0, abs=1e-12)


def test_complementary_group_bounds_define_the_same_portfolio():
    """A group's floor and its complement's cap are equivalent under full investment."""
    covar = np.diag([0.01, 0.02, 0.09, 0.16])
    floor, _ = solve_constrained_risk_budgeting(
        covar, c_rows=np.array([[0., 0., -1., -1.]]), c_lhs=np.array([-.6]))
    cap, _ = solve_constrained_risk_budgeting(
        covar, c_rows=np.array([[1., 1., 0., 0.]]), c_lhs=np.array([.4]))
    np.testing.assert_allclose(floor, cap, atol=1e-8)


@pytest.mark.parametrize('scale', [1e-6, 1.0, 1e6])
def test_constrained_weights_match_independent_conic_reference(scale):
    """Covariance units must not alter the normalized bounded portfolio."""
    covar = np.array([[.04, -.015, .003], [-.015, .01, -.001],
                      [.003, -.001, .0225]])
    budgets = np.array([.55, .35, .10])
    bounds = np.array([[.40, .7], [.10, .5], [.05, .3]])
    rows, lhs = np.array([[1., 0., 1.]]), np.array([.70])
    expected = conic_reference(covar, budgets, bounds, rows, lhs)
    weights, _ = solve_constrained_risk_budgeting(
        scale * covar, budgets, bounds, rows, lhs)
    np.testing.assert_allclose(weights, expected, atol=2e-6)
    assert weights.sum() == pytest.approx(1.0, abs=1e-12)
    assert np.all(weights >= bounds[:, 0] - 1e-12)
    assert np.all(weights <= bounds[:, 1] + 1e-12)
    assert np.all(rows @ weights <= lhs + 1e-12)


def test_inferred_budgets_reproduce_target_with_slack_bands():
    """Analytical diagonal-covariance inverse budgets recover their feasible target."""
    target = np.array([.50, .25, .15, .10])
    variances = np.array([.04, .01, .0225, .0004])
    # Independent analytical identity for diagonal covariance, not a portfolio analytics layer.
    budgets = target**2 * variances
    budgets /= budgets.sum()
    bounds = np.column_stack([target - .03, target + .03])
    weights, _ = solve_constrained_risk_budgeting(
        np.diag(variances), budgets, bounds,
        np.array([[1., 1., 0., 0.], [-1., -1., 0., 0.]]), np.array([.78, -.72]))
    np.testing.assert_allclose(weights, target, atol=2e-5)


def test_group_infeasibility_is_detected_with_full_investment():
    """A valid box does not excuse group caps that cannot reach 100%."""
    with pytest.raises(ValueError, match='infeasible'):
        solve_constrained_risk_budgeting(
            np.eye(3), c_rows=np.ones((1, 3)), c_lhs=np.array([.9]))


def test_pinned_box_still_checks_group_bounds():
    """The unique box point is not feasible if it violates a group cap."""
    with pytest.raises(ValueError, match='infeasible'):
        solve_constrained_risk_budgeting(
            np.eye(2), bounds=np.array([[.6, .6], [.4, .4]]),
            c_rows=np.array([[1., 0.]]), c_lhs=np.array([.5]))


def test_instrument_floors_can_exhaust_full_investment():
    """Minimums totalling 100% define a feasible ray, not a missing lambda bracket."""
    weights, _ = solve_constrained_risk_budgeting(
        np.eye(3), bounds=np.array([[.4, .8], [.3, .8], [.3, .8]]))
    np.testing.assert_allclose(weights, [.4, .3, .3], atol=1e-12)


def test_partial_freeze_including_a_zero_position_is_preserved():
    """A zero-pinned asset stays zero and a positive frozen weight does not rescale."""
    weights, _ = solve_constrained_risk_budgeting(
        np.diag([.04, .01, .09, .16]),
        bounds=np.array([[0., 0.], [.3, .3], [.1, .5], [.1, .6]]))
    np.testing.assert_allclose(weights[:2], [0., .3], atol=1e-12)
    assert weights.sum() == pytest.approx(1.0, abs=1e-12)


def test_all_budgets_on_zero_pinned_assets_raise():
    """A forbidden budget cannot be reassigned implicitly to an unbudgeted universe."""
    with pytest.raises(ValueError, match='no positive risk budget'):
        solve_constrained_risk_budgeting(
            np.eye(3), budgets=np.array([1., 0., 0.]),
            bounds=np.array([[0., 0.], [0., 1.], [0., 1.]]))


def test_ccd_iteration_exhaustion_is_not_accepted(monkeypatch):
    """Unfinished coordinate updates must not be returned as an optimum."""
    monkeypatch.setattr(rb_solver, 'MAX_CCD_CYCLES', 1)
    monkeypatch.setattr(rb_solver, 'CCD_TOL', -1.0)
    with pytest.raises(ValueError, match='CCD did not converge after 1 cycles'):
        solve_constrained_risk_budgeting(np.diag([.04, .01]))


@pytest.mark.parametrize('candidate', [np.zeros(2), np.full(2, np.nan)])
def test_invalid_solver_scale_is_rejected(monkeypatch, candidate):
    """A numerical failure cannot become normalized weights or reach the projection."""
    monkeypatch.setattr(rb_solver, '_ccd_solve', lambda *args: candidate)
    with pytest.raises(ValueError, match='degenerate risk-budgeting scale'):
        solve_constrained_risk_budgeting(np.eye(2))


def test_projection_does_not_repair_a_material_solver_violation(monkeypatch):
    """Only roundoff, not a failed optimizer allocation, may be projected at the end."""
    monkeypatch.setattr(rb_solver, '_admm_ccd_solve', lambda *args: np.array([.9, .1]))
    with pytest.raises(ValueError, match='violates the original portfolio bounds'):
        solve_constrained_risk_budgeting(np.eye(2), bounds=np.array([[0., .6], [0., .6]]))
