"""Check algebraic fixed-row reduction without changing the feasible domain."""

from dataclasses import replace

import cvxpy as cp
import numpy as np
import pytest

from optimalportfolios.execution.v2 import audit_execution, compile_states
from optimalportfolios.execution.v2.domain import variance
from optimalportfolios.execution.v2.model import _interval_rows, build_model, solve_qp
from optimalportfolios.execution.v2.tests.formulation_test import independent_kkt
from optimalportfolios.execution.v2.tests.group_audit_test import pinned_group


def one_trade(problem):
    """Select one funded trade and retain every other original position."""
    states = compile_states(problem)
    fixed = {i: next(j for j, state in enumerate(row) if state.ticket == (i == 2))
             for i, row in enumerate(states[:-1])}
    return fixed, states


def test_arbitrarily_small_nonzero_direction_is_preserved():
    """A small coefficient is not a constant-row licence or a widened upper bound."""
    x = cp.Variable(1)
    rows = _interval_rows(1e-30*x, np.array([[1e-30]]), np.zeros(1),
                          np.zeros(1), np.array([1e-30]), True)
    x.value = np.array([2.])
    assert len(rows) == 2
    assert rows[1].violation().item() == pytest.approx(1e-30, rel=1e-14, abs=0.)


@pytest.mark.parametrize("side", ["lower", "upper"])
def test_tight_constant_rows_do_not_stall_fixed_state_sizing(side):
    """A near-saturated constant group must not obstruct an independent scalar optimum."""
    p = pinned_group(side, (1.0 - 1e-8) * 1e-7)
    fixed, states = one_trade(p)
    model = build_model(p, fixed_states=fixed)
    data = model.optimisation.get_problem_data(cp.CLARABEL)[0]
    zero_row = np.asarray((data["A"] != 0).sum(axis=1)).reshape(-1) == 0
    # Exact fixed-coordinate identities and tiny positive constant slacks add
    # no feasible-set information but can stall an interior-point iteration.
    assert not np.any(zero_row & (np.abs(data["b"]) < 1e-8))
    weights = solve_qp(model)
    assert model.optimisation.status == cp.OPTIMAL
    assert audit_execution(p, weights)["passed"]
    reference = replace(p, group_lower=np.array([-1.]), group_upper=np.array([1.]))
    optimum = independent_kkt(reference, {
        i: (states[i][j].lower, states[i][j].upper) for i, j in fixed.items()})
    assert abs(variance(p, weights) - optimum) < 1e-12


@pytest.mark.parametrize("side", ["lower", "upper"])
def test_constant_group_contradiction_is_retained(side):
    """A constant breach just beyond the same audit envelope stays infeasible."""
    p = pinned_group(side, 1.01e-7)
    fixed, _ = one_trade(p)
    model = build_model(p, fixed_states=fixed)
    assert solve_qp(model) is None
    assert model.optimisation.status == cp.INFEASIBLE
    assert "groups" in audit_execution(p, p.base)["violations"]


def test_fixed_cash_contradiction_is_retained():
    """Eliminating noncash coordinates must not erase an impossible cash bound."""
    p = pinned_group("upper", 0.)
    fixed = {i: 0 for i in range(len(p.names)-1)}
    lo, hi = p.lower.copy(), p.upper.copy()
    lo[:-1] = hi[:-1] = p.current[:-1]
    lo[-1] = hi[-1] = .01
    p = replace(p, lower=lo, upper=hi, group_lower=np.array([-1.]),
                group_upper=np.array([1.]))
    model = build_model(p, fixed_states=fixed)
    solve_qp(model)
    assert model.optimisation.status == cp.INFEASIBLE


def test_feasible_constant_problem_keeps_original_funding():
    """An entirely fixed feasible decision needs no artificial variable or bound slack."""
    p = pinned_group("upper", 0.)
    lo, hi = p.current.copy(), p.current.copy()
    lo[-1], hi[-1] = 0., .1
    p = replace(p, lower=lo, upper=hi)
    model = build_model(p, fixed_states={i: 0 for i in range(len(p.names)-1)})
    weights = solve_qp(model)
    np.testing.assert_allclose(weights, p.current, atol=1e-16, rtol=0.)
    assert model.optimisation.status == cp.OPTIMAL
    assert audit_execution(p, weights)["passed"]
