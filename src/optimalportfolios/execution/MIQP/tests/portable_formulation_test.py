"""Portable complete-formulation and untrusted-incumbent contracts."""

from dataclasses import replace
import cvxpy as cvx
import numpy as np
import pandas as pd
import pytest
from optimalportfolios import GroupLowerUpperConstraints
from optimalportfolios.execution.MIQP import solver as e
from optimalportfolios.execution.MIQP.tests.solver_test import _exact_config, _exact_inputs
from optimalportfolios.execution.MIQP.tests.recovery_test import _case


@pytest.mark.parametrize("domain", ["live", "history", "inception", "soft"])
def test_full_formulation_accepts_independently_feasible_current_holdings(monkeypatch, domain):
    """The known feasible zero-trade incumbent passes each domain's complete audit."""
    config = _exact_config(max_trades=2)
    config = replace(
        config,
        exact_execution=replace(
            config.exact_execution,
            alpha_tradeoff=0.01,
            sub_asset_class_penalty=100 if domain == "soft" else None,
        ),
    )
    table, constraints, covariance = _exact_inputs(config)
    kwargs = {}
    if domain == "history":
        table[e.EXACT_HISTORY_CANDIDATE] = True
        table[e.RAW_MODEL_WEIGHT] = table[e.CURRENT_WEIGHT]
    if domain == "inception":
        table[e.EXACT_INCEPTION_CANDIDATE] = True
    if domain == "soft":
        constraints = constraints.copy(
            group_lower_upper_constraints=GroupLowerUpperConstraints(
                pd.DataFrame({"Soft": [1.0, 0.0, 0.0]}, index=table.index),
                pd.Series({"Soft": 0.5}),
                None,
            )
        )
        kwargs["soft_group_names"] = ("Soft",)

    def solve(problem, exact_config):
        """Provide a feasible fixed point without relying on a commercial backend."""
        problem._status = cvx.OPTIMAL
        for variable in problem.variables():
            if variable.shape == (3,):
                variable.value = table[e.CURRENT_WEIGHT].to_numpy()
            elif variable.name() == "exact_hold":
                variable.value = [1, 1]
            else:
                variable.value = np.zeros(variable.shape)
        assert all(np.max(c.violation()) < 1e-10 for c in problem.constraints)
        return e.ExactExecutionSolverAudit(integer_solutions=1, termination="gap")

    monkeypatch.setattr(e, "_solve_mosek_with_audit", solve)
    result = e.solve_exact_execution_portfolio(table, constraints, covariance, config, **kwargs)
    assert result.accepted and result.compliant
    np.testing.assert_array_equal(result.weights, table[e.CURRENT_WEIGHT])


@pytest.mark.parametrize("inception", [False, True])
def test_native_rejection_never_becomes_an_accepted_hold(monkeypatch, inception):
    """A native failure remains rejected and retains its domain-specific cause."""
    config = _exact_config()
    table, constraints, covariance = _exact_inputs(config)
    if inception:
        table[e.EXACT_INCEPTION_CANDIDATE] = True

    def fail(problem, exact_config):
        """Return a failed native solve without an integer incumbent."""
        problem._status = cvx.INFEASIBLE if inception else "solver_error"
        return e.ExactExecutionSolverAudit()

    monkeypatch.setattr(e, "_solve_mosek_with_audit", fail)
    result = e.solve_exact_execution_portfolio(table, constraints, covariance, config)
    assert not result.accepted


def test_missing_binary_pattern_cannot_be_polished(monkeypatch):
    """Numerical recovery requires all integer choices, not just candidate weights."""
    args = _case(monkeypatch)
    original = e._solve_mosek_with_audit

    def incomplete(problem, exact_config):
        """Emulate an incumbent receipt with one missing binary selector."""
        audit = original(problem, exact_config)
        for variable in problem.variables():
            if variable.name() == "exact_hold":
                variable.value = None
        return audit

    monkeypatch.setattr(e, "_solve_mosek_with_audit", incomplete)
    result = e.solve_exact_execution_portfolio(**args)
    assert not result.accepted and not result.trade_table["exact_numerical_recovery"].any()


@pytest.mark.parametrize("domain", ["frozen", "forced_zero"])
def test_nontradeable_coordinates_are_constrained_in_the_native_problem(monkeypatch, domain):
    """Cash-funded changes to frozen or excluded assets violate the built formulation."""
    config = _exact_config(max_trades=2)
    table, constraints, covariance = _exact_inputs(config)
    if domain == "frozen":
        table.loc["A", e.REBALANCE_CADENCE_ELIGIBLE] = False
    else:
        table[e.EXACT_INCEPTION_CANDIDATE] = [False, True, True]
        for column in (e.CURRENT_WEIGHT, e.BASE_WEIGHT, e.RAW_MODEL_WEIGHT):
            table[column] = [0.0, 0.1, 0.9]
    current = table[e.CURRENT_WEIGHT].to_numpy(copy=True)

    def solve(problem, exact_config):
        """Check both an independent feasible point and a funded forbidden movement."""
        problem._status = cvx.OPTIMAL
        weights = None
        for variable in problem.variables():
            if variable.name() == "exact_weights":
                weights = variable
                variable.value = current
            elif variable.name() == "exact_hold":
                variable.value = np.ones(variable.shape)
            else:
                variable.value = np.zeros(variable.shape)
        assert all(np.max(c.violation()) < 1e-10 for c in problem.constraints)
        forbidden = current.copy()
        forbidden[0] += 0.01
        forbidden[2] -= 0.01
        weights.value = forbidden
        assert any(np.max(c.violation()) > 1e-4 for c in problem.constraints)
        weights.value = current
        return e.ExactExecutionSolverAudit(integer_solutions=1, termination="gap")

    monkeypatch.setattr(e, "_solve_mosek_with_audit", solve)
    result = e.solve_exact_execution_portfolio(table, constraints, covariance, config)
    assert result.accepted and result.compliant
    np.testing.assert_array_equal(result.weights, current)


@pytest.mark.parametrize("corrupt_cleanup", [False, True])
def test_accepted_weight_dust_is_cleaned_and_independently_reaudited(
        monkeypatch, caplog, corrupt_cleanup):
    """Return cleaned cash-funded weights only after the original hard constraints pass."""
    config = _exact_config(max_trades=2)
    table, constraints, covariance = _exact_inputs(config)
    current = table[e.CURRENT_WEIGHT].to_numpy(copy=True)
    dusty = current + np.array([5e-8, 0.0, -5e-8])
    receipts = []
    original_validate = e.validate_solution
    original_cleanup = e._clean_exact_weight_dust

    def solve(problem, exact_config):
        """Supply a tolerance-compliant incumbent with one tiny funded trade."""
        problem._status = cvx.OPTIMAL
        for variable in problem.variables():
            if variable.name() == "exact_weights":
                variable.value = dusty
            elif variable.name() == "exact_hold":
                variable.value = np.ones(variable.shape)
            else:
                variable.value = np.zeros(variable.shape)
        return e.ExactExecutionSolverAudit(integer_solutions=1, termination="gap")

    def cleanup(**kwargs):
        """Optionally corrupt only the cleanup result, preserving its real count."""
        weights, count = original_cleanup(**kwargs)
        assert count == 1
        if corrupt_cleanup:
            weights[0] += 0.7
            weights[2] -= 0.7
        return weights, count

    def validate(**kwargs):
        """Observe calls while retaining the real independent constraint checker."""
        receipts.append((kwargs["context"], np.array(kwargs["optimal_weights"], copy=True)))
        return original_validate(**kwargs)

    monkeypatch.setattr(e, "_solve_mosek_with_audit", solve)
    monkeypatch.setattr(e, "_clean_exact_weight_dust", cleanup)
    monkeypatch.setattr(e, "validate_solution", validate)
    with caplog.at_level("INFO", logger=e.__name__):
        result = e.solve_exact_execution_portfolio(table, constraints, covariance, config)
    assert any("cleaned numerical zeros" in context for context, _ in receipts)
    assert "exact_weight_dust_cleaned instruments=1" in caplog.text
    assert result.accepted == (not corrupt_cleanup)
    # A rejected attempt returns the still-compliant held portfolio, without acceptance.
    assert result.compliant
    np.testing.assert_allclose(result.weights, current, rtol=0.0, atol=1e-15)
    assert result.weights.sum() == pytest.approx(dusty.sum(), rel=0.0, abs=1e-15)
    if corrupt_cleanup:
        assert "long-only violated" in result.outcome.reason
