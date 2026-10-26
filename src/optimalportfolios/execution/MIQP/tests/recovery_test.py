"""Offline regression checks for audited fixed-selection numerical recovery."""

import cvxpy as cvx
import numpy as np
import pytest
import optimalportfolios.execution.MIQP.solver as exact
from optimalportfolios.execution.v1.policy import CURRENT_WEIGHT, RAW_MODEL_WEIGHT, BASE_WEIGHT
from optimalportfolios.execution.MIQP.tests.solver_test import _exact_config, _exact_inputs


def _case(monkeypatch, residue=2e-7):
    """Inject an optimal MIP exit with a residual balance and known binary decisions."""
    config = _exact_config(max_trades=2, material_trade_threshold=0.005)
    table, constraints, covariance = _exact_inputs(config)
    table[CURRENT_WEIGHT] = [0.03, 0.1, 0.87]
    table[BASE_WEIGHT] = [0.03, 0.1, 0.87]
    table[RAW_MODEL_WEIGHT] = [0.0, 0.3, 0.7]

    def mip(problem, exact_config):
        """Supply a MIP incumbent with a numerically imperfect full exit."""
        problem._status = cvx.OPTIMAL
        for variable in problem.variables():
            if variable.shape == (3,):
                variable.value = [residue, 0.3, 0.7 - residue]
            elif variable.name() == "exact_hold":
                variable.value = [0, 1]
            elif variable.name() == "exact_buy":
                variable.value = [0, 1]
            elif variable.name() == "exact_sell":
                variable.value = [1, 0]
            elif variable.name() == "exact_trade":
                variable.value = [1, 1]
        return exact.ExactExecutionSolverAudit(integer_solutions=1, termination="gap")

    def continuous(problem, exact_config):
        """Solve the unchanged fixed pattern without a commercial test dependency."""
        assert not problem.is_mixed_integer()
        problem.solve(solver="CLARABEL", tol_gap_abs=1e-11, tol_feas=1e-11, tol_gap_rel=1e-11)

    monkeypatch.setattr(exact, "_solve_mosek_with_audit", mip)
    monkeypatch.setattr(exact, "_solve_exact_continuous_problem", continuous, raising=False)
    return dict(
        trade_table=table,
        base_constraints=constraints,
        covariance=covariance,
        config=config,
        context="exit residue",
    )


def test_numerical_exit_is_recovered_with_fixed_selections(monkeypatch):
    """A tiny exit residue is repaired without changing tickets or mandate limits."""
    result = exact.solve_exact_execution_portfolio(**_case(monkeypatch))
    assert result.accepted and result.compliant
    # The target itself is feasible and has zero objective: an independent reference.
    np.testing.assert_allclose(result.weights, [0.0, 0.3, 0.7], atol=1e-7, rtol=0.0)
    assert result.trade_table["exact_numerical_recovery"].all()
    assert all(r.passed for r in result.outcome.constraint_residuals if r.hard)


def test_substantive_holding_shortfall_is_not_numerical_recovery(monkeypatch):
    """Recovery cannot move a material 2% residue to zero and call it numerical."""
    result = exact.solve_exact_execution_portfolio(**_case(monkeypatch, residue=0.02))
    assert not result.accepted
    assert "A(weight=0.02" in result.outcome.reason
    assert "holding_floor=0.1" in result.outcome.reason
    assert not result.trade_table["exact_numerical_recovery"].any()


def test_failed_polishing_keeps_detailed_rejection(monkeypatch):
    """A recovery solver failure cannot turn a rejected MIP solution into a pass."""
    args = _case(monkeypatch)

    def unavailable(problem, exact_config):
        """Simulate the continuous solver rejecting the fixed pattern."""
        raise cvx.error.SolverError("fixed pattern unavailable")

    monkeypatch.setattr(exact, "_solve_exact_continuous_problem", unavailable)
    result = exact.solve_exact_execution_portfolio(**args)
    assert not result.accepted
    assert "A(weight=2e-07" in result.outcome.reason
    assert "ticket=" in result.outcome.reason
    assert "tolerance=" in result.outcome.reason


def test_successful_mip_does_not_call_recovery(monkeypatch):
    """Already compliant portfolios retain their original weights and solver path."""
    args = _case(monkeypatch, residue=0.0)

    def forbidden(problem, exact_config):
        """Fail if a successful original incumbent is unnecessarily re-solved."""
        pytest.fail("unexpected recovery")

    monkeypatch.setattr(exact, "_solve_exact_continuous_problem", forbidden)
    result = exact.solve_exact_execution_portfolio(**args)
    assert result.accepted and result.compliant
    np.testing.assert_array_equal(result.weights, [0.0, 0.3, 0.7])


def test_numerical_model_zero_does_not_block_full_exit(monkeypatch):
    """A numerical model zero must not forbid the zero-holding side of the MIP."""
    args = _case(monkeypatch)
    args["trade_table"][exact.EXACT_HISTORY_CANDIDATE] = True
    args["trade_table"][RAW_MODEL_WEIGHT] = [2e-7, 0.3, 0.7 - 2e-7]
    result = exact.solve_exact_execution_portfolio(**args)
    assert result.accepted and result.compliant
    assert abs(result.weights["A"]) < 1e-7


def test_polished_result_must_pass_the_original_constraints(monkeypatch):
    """A solver status alone cannot accept a still-invalid polished portfolio."""
    args = _case(monkeypatch)

    def invalid_polish(problem, exact_config):
        """Return the same residual exit while falsely reporting an optimal solve."""
        problem._status = cvx.OPTIMAL
        for variable in problem.variables():
            if variable.name() == "exact_weights":
                variable.value = [2e-7, 0.3, 0.7 - 2e-7]

    monkeypatch.setattr(exact, "_solve_exact_continuous_problem", invalid_polish)
    result = exact.solve_exact_execution_portfolio(**args)
    assert not result.accepted
    assert not result.trade_table["exact_numerical_recovery"].any()
