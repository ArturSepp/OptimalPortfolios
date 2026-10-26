"""Regression contracts for coordinated, cash-funded optional trade changes."""

from dataclasses import replace

import cvxpy as cp
import numpy as np
import pytest

from optimalportfolios.execution.v2 import (
    ExecutionProblem, ExecutionRules, RankedConfig, audit_execution,
    compile_states, wrapper_ranked_execution,
)
from optimalportfolios.execution.v2.ranked import _pair_gain, ranked_selection


def coordinated_problem(cap=2, fixed_cash=False):
    """Two correlated names need opposite trades, neither profitable alone."""
    current = np.array([0.05, 0.05, 0.10])
    return ExecutionProblem(
        names=("buy", "sell", "cash"), cash=2, current=current, base=current,
        target=np.array([0.07, 0.03, 0.10]),
        lower=np.array([0.05, 0.03, 0.10 if fixed_cash else 0.0]),
        upper=np.array([0.07, 0.05, 0.10 if fixed_cash else 0.2]),
        full_lower=np.zeros(3), full_upper=np.full(3, 0.2),
        groups=np.zeros((0, 3)), group_lower=np.zeros(0), group_upper=np.zeros(0),
        mandatory=np.zeros(3, bool), categories=("",) * 3,
        eligible=np.ones(3, bool), frozen=np.zeros(3, bool), investment=0.2,
        rules=ExecutionRules(ticket_cap=cap, minimum_ticket=0.01, holding_floor=0.005),
        covariance=np.array([[1., .99, 0.], [.99, 1., 0.], [0., 0., 0.]]),
    )


@pytest.mark.parametrize("fixed_cash", [False, True])
def test_coordinated_pair_recovers_stalled_greedy_search(fixed_cash):
    """Jointly admit opposite trades even when both scalar gains are negative."""
    p = coordinated_problem(fixed_cash=fixed_cash)
    selected, _, trace, _ = ranked_selection(p, compile_states(p), 2)
    assert selected == {} and trace == []
    out = wrapper_ranked_execution(p)
    assert audit_execution(p, out.weights)["passed"]
    # Independent reference: the target is feasible and zero variance is the
    # global lower bound of this positive-semidefinite quadratic.
    assert out.diagnostics["variance"] < 1e-10
    assert np.allclose(out.weights, p.target, atol=1e-4, rtol=0)
    assert out.diagnostics["audit"]["tickets"] == 2


@pytest.mark.parametrize("cap", [0, 1])
def test_coordinated_search_preserves_hard_capacity(cap):
    """Two useful trades cannot be smuggled through a one-ticket allowance."""
    p = coordinated_problem(cap=cap)
    out = wrapper_ranked_execution(p)
    assert np.array_equal(out.weights, p.base)
    assert audit_execution(p, out.weights)["passed"]


@pytest.mark.parametrize("config", [RankedConfig(exchange=False), RankedConfig(rescue_trials=0)])
def test_coordinated_search_can_be_disabled(config):
    """Existing bounded-search controls still permit the initial sizing alone."""
    p = coordinated_problem()
    assert np.array_equal(wrapper_ranked_execution(p, config).weights, p.base)


def test_coordinated_dense_and_structured_covariance_agree():
    """The pair score and final sizing retain exact structured covariance semantics."""
    p = coordinated_problem()
    eig, vec = np.linalg.eigh(p.covariance)
    structured = replace(p, covariance=None, factor_loadings=vec,
                         factor_covariance=np.diag(eig), residual_variance=np.zeros(3))
    dense = wrapper_ranked_execution(p)
    other = wrapper_ranked_execution(structured)
    assert dense.diagnostics["variance"] < 1e-10
    assert np.allclose(dense.weights, other.weights, atol=1e-7, rtol=0)


def test_pair_priority_and_deduplication_respect_trial_budget():
    """A single extra trial tests the joint gain before either losing scalar trade."""
    p = coordinated_problem()
    out = wrapper_ranked_execution(p, RankedConfig(rescue_trials=1))
    assert out.diagnostics["variance"] < 1e-10
    assert out.diagnostics["coordinated_candidates"] == 3
    assert out.diagnostics["coordinated_search_truncated"]
    assert out.diagnostics["qp_counts"]["coordinated"] == 1
    assert out.diagnostics["visited_supports"] == 2


@pytest.mark.parametrize("mode", ["failure", "unaudited"])
def test_failed_pair_cannot_replace_audited_incumbent(monkeypatch, mode):
    """Neither a native exception nor an invalid returned vector weakens the gate."""
    import optimalportfolios.execution.v2.ranked as module

    p = coordinated_problem()

    def invalid_sizing(*args, **kwargs):
        """Inject failure or a fully invested vector outside the supplied corridors."""
        if mode == "failure":
            raise cp.SolverError("injected coordinated sizing failure")
        return np.array([0.2, 0., 0.])

    monkeypatch.setattr(module, "solve_qp", invalid_sizing)
    out = wrapper_ranked_execution(p)
    assert np.array_equal(out.weights, p.base)
    assert out.diagnostics["qp_counts"]["coordinated"] > 0


def test_pair_score_matches_independent_convex_box_solve():
    """Verify edge, interior, singular and zero-curvature scores with a QP backend."""
    cases = [
        (np.eye(2), np.array([-.2, .3])),
        (np.array([[1., .99], [.99, 1.]]), np.array([-2., .5])),
        (np.ones((2, 2)), np.array([.2, .2])),
        (np.zeros((2, 2)), np.array([-1., 2.])),
    ]
    for hessian, gradient in cases:
        x = cp.Variable(2)
        qp = cp.Problem(cp.Minimize(cp.quad_form(x, hessian) + 2 * gradient @ x),
                        [x >= -.4, x <= .6])
        qp.solve(solver="CLARABEL", tol_gap_abs=1e-10, tol_feas=1e-10, tol_gap_rel=1e-10)
        gain = _pair_gain(*gradient, hessian[0, 0], hessian[1, 1], hessian[0, 1],
                          -.4, .6, -.4, .6)
        assert abs(gain + qp.value) < 1e-8
