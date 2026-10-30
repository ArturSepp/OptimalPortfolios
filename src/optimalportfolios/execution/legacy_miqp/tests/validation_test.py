"""Exact preflight, infeasibility evidence and optional-backend contracts."""

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock
import builtins
import sys

import cvxpy as cvx
import numpy as np
import pandas as pd
import pytest

from optimalportfolios import GroupLowerUpperConstraints
from optimalportfolios.execution.legacy_miqp import solver as e
from optimalportfolios.execution.legacy_miqp.tests.solver_test import _exact_config, _exact_inputs


@pytest.mark.parametrize(
    "column,value,match",
    [
        (e.CURRENT_WEIGHT, np.nan, "finite"),
        (e.CURRENT_WEIGHT, 0.9, "sum to one"),
        (e.PRODUCT_MIN_WEIGHT, 1.1, "lower bound"),
        (e.EFFECTIVE_SMALL_TRADE_CUTOFF, -0.1, "non-negative"),
        (e.SETTLEMENT_CASH, True, "exactly one"),
    ],
)
def test_invalid_preflight_table(column, value, match):
    """Invalid rows fail before any backend or incumbent is consulted."""
    config = _exact_config()
    table, _, _ = _exact_inputs(config)
    table.loc["A", column] = value
    with pytest.raises(ValueError, match=match):
        e._prepare_exact_problem(table, config)


def test_required_schema_and_complete_configuration():
    """Revalidate deserialized configuration and table contracts at the solver boundary."""
    config = _exact_config()
    table, constraints, covar = _exact_inputs(config)
    with pytest.raises(ValueError, match="missing columns"):
        e._prepare_exact_problem(table.drop(columns=[e.ALPHA]), config)
    duplicate = pd.concat([table, table.iloc[:1]])
    with pytest.raises(ValueError, match="unique"):
        e._prepare_exact_problem(duplicate, config)
    data = e._prepare_exact_problem(table, config)
    object.__setattr__(config, "exact_execution", None)
    with pytest.raises(ValueError, match="ExactExecutionConfig"):
        e._prepare_exact_problem(table, config)
    with pytest.raises(ValueError, match="complete configuration"):
        e._audit_exact_solution(data.current, data, config)
    assert diagnose(data, constraints, config)[0] == ("invalid_exact_configuration",)
    object.__setattr__(config, "exact_execution", e.ExactExecutionConfig())
    object.__setattr__(config, "max_trades", None)
    with pytest.raises(ValueError, match="finite max_trades"):
        e._prepare_exact_problem(table, config)


@pytest.mark.parametrize(
    "case,match",
    [
        ("excluded_target", "positive exact"),
        ("excluded_min", "positive product"),
        ("off_cadence", "off-cadence exact"),
        ("holding", "sub-floor holdings"),
        ("frozen", "held off-cadence"),
    ],
)
def test_exact_domains_reject_impossible_requirements(case, match):
    """Candidate screens, cadence and exact targets may not contradict hard requirements."""
    config = _exact_config(max_trades=2)
    table, _, _ = _exact_inputs(config)
    if case in ("excluded_target", "excluded_min"):
        table[e.EXACT_INCEPTION_CANDIDATE] = [False, True, True]
        if case == "excluded_target":
            table.loc["A", e.FIXED_INSTRUCTION] = True
        else:
            table.loc["A", e.PRODUCT_MIN_WEIGHT] = 0.1
    elif case == "off_cadence":
        table[e.EXACT_HISTORY_CANDIDATE] = [False, True, True]
        table.loc["A", e.FIXED_INSTRUCTION] = True
        table.loc["A", e.REBALANCE_CADENCE_ELIGIBLE] = False
        table.loc["A", e.BASE_WEIGHT] = 0.2
    elif case == "holding":
        table.loc["A", e.FIXED_INSTRUCTION] = True
        table.loc["A", e.BASE_WEIGHT] = 0.01
    else:
        config = replace(config, exact_execution=replace(config.exact_execution, max_positions=1))
        table.loc[["A", "B"], e.REBALANCE_CADENCE_ELIGIBLE] = False
    with pytest.raises(ValueError, match=match):
        e._prepare_exact_problem(table, config)


def test_preflight_rejects_unresolved_effective_ticket_budget(monkeypatch):
    """A missing budget from a supplied configuration cannot create an uncapped model."""
    config = _exact_config()
    table, _, _ = _exact_inputs(config)
    monkeypatch.setattr(e, "resolve_effective_max_trades", lambda **kwargs: None)
    with pytest.raises(ValueError, match="finite max_trades"):
        e._prepare_exact_problem(table, config)


@pytest.mark.parametrize("kind", ["missing", "asymmetric"])
def test_covariance_rejected_before_native_solve(kind):
    """A missing or asymmetric covariance cannot reach mixed integer optimisation."""
    config = _exact_config()
    table, constraints, covar = _exact_inputs(config)
    if kind == "missing":
        covar = covar.drop(index="A")
    else:
        covar.loc["A", "B"] = 1
    with pytest.raises(ValueError, match="covariance"):
        e.solve_legacy_miqp_portfolio(table, constraints, covar, config)


def test_exact_constraints_require_resolved_beta():
    """Neither live nor inception compilation may infer missing beta loadings."""
    config = _exact_config()
    table, constraints, _ = _exact_inputs(config)
    constraints = constraints.copy(benchmark_beta_constraint=SimpleNamespace(beta_loadings=None))
    data = e._prepare_exact_problem(table, config)
    for build in (e._build_exact_constraints, e._build_exact_inception_constraints):
        with pytest.raises(ValueError, match="resolved benchmark beta"):
            build(constraints, data)


def test_soft_groups_are_explicit_and_nonempty():
    """Only the named subgroups can be softened, with at least one finite bound."""
    config = _exact_config()
    _, constraints, _ = _exact_inputs(config)
    with pytest.raises(ValueError, match="named sub-asset-class"):
        e._partition_soft_group_constraints(constraints, None)
    groups = GroupLowerUpperConstraints(
        pd.DataFrame({"G": [1.0, 0.0, 0.0]}, index=["A", "B", "Cash"]),
        group_min_allocation=pd.Series({"G": np.nan}),
        group_max_allocation=None,
    )
    constraints = constraints.copy(group_lower_upper_constraints=groups)
    with pytest.raises(ValueError, match="no sub-asset-class"):
        e._partition_soft_group_constraints(constraints, ("unknown",))
    with pytest.raises(ValueError, match="all blank"):
        e._soft_group_objective(cvx.Variable(3), groups)
    groups = replace(groups, group_min_allocation=pd.Series({"G": 0.2}))
    w = cvx.Variable(3)
    w.value = [0.1, 0.1, 0.8]
    assert e._soft_group_objective(w, groups).value == pytest.approx(0.01)


def test_exact_failure_diagnostics_distinguish_binding_causes():
    """Independent interval audits distinguish funding, group, cash and integer conflicts."""
    config = _exact_config()
    table, constraints, _ = _exact_inputs(config)
    data = e._prepare_exact_problem(table, config)
    assert "joint_integer_constraint_intersection" in diagnose(data, constraints, config)[0]
    bad = replace(data, effective_lower=np.array([0.8, 0.8, 0.8]))
    assert "product_funding_interval" in diagnose(bad, constraints, config)[0]
    group = GroupLowerUpperConstraints(
        pd.DataFrame({"Low": [1.0, 0.0, 0.0], "High": [0.0, 1.0, 0.0]}, index=table.index),
        group_min_allocation=pd.Series({"Low": 0.6, "High": np.nan}),
        group_max_allocation=pd.Series({"Low": np.nan, "High": 0.05}),
    )
    codes, detail = diagnose(
        replace(data, tradeable=np.array([], dtype=int), frozen=np.array([0, 1])),
        constraints.copy(group_lower_upper_constraints=group),
        config,
    )
    assert "group_ticket_bridge" in codes and "Low" in detail and "High" in detail
    bad = replace(
        data, true_lower=np.array([0.0, 0.0, 0.95]), effective_lower=np.array([0.0, 0.0, 0.95])
    )
    assert "cash_ticket_bridge" in diagnose(bad, constraints, config)[0]


def test_mip_stop_categories_are_conservative():
    """An unproved stop is not reported as a certified optimum."""
    c = e.ExactExecutionConfig(max_branches=3)
    assert e._classify_mip_termination(e.ExactExecutionSolverAudit(branches=3), c) == "branch_limit"
    assert e._classify_mip_termination(e.ExactExecutionSolverAudit(), c) == "other"


def test_backend_errors_are_actionable(monkeypatch):
    """Missing bindings, unavailable solver and native errors propagate distinctly."""
    c = e.ExactExecutionConfig()
    problem = Mock()
    monkeypatch.setattr(cvx, "installed_solvers", lambda: [])
    with pytest.raises(RuntimeError, match="licensed MOSEK"):
        e._solve_mosek_with_audit(problem, c)
    monkeypatch.setattr(cvx, "installed_solvers", lambda: ["MOSEK"])
    original = builtins.__import__

    def blocked(name, *args, **kwargs):
        """Simulate a base install without the optional native binding."""
        if name == "mosek":
            raise ImportError("not installed")
        return original(name, *args, **kwargs)

    with monkeypatch.context() as m:
        m.setattr(builtins, "__import__", blocked)
        with pytest.raises(RuntimeError, match="optional MOSEK"):
            e._solve_mosek_with_audit(problem, c)
    monkeypatch.setitem(sys.modules, "mosek", SimpleNamespace())
    problem.get_problem_data.side_effect = RuntimeError("native failure")
    with pytest.raises(RuntimeError, match="MOSEK solve failed: native failure"):
        e._solve_mosek_with_audit(problem, c)


def test_continuous_recovery_fixes_all_discrete_choices():
    """Recovery refuses integer variables and preserves the exact time limit."""
    c = e.ExactExecutionConfig(max_solve_seconds=3)
    problem = Mock()
    problem.is_mixed_integer.return_value = True
    with pytest.raises(ValueError, match="fix every integer"):
        e._solve_exact_continuous_problem(problem, c)
    problem.is_mixed_integer.return_value = False
    e._solve_exact_continuous_problem(problem, c)
    assert problem.solve.call_args.kwargs["mosek_params"]["MSK_DPAR_OPTIMIZER_MAX_TIME"] == 3


def diagnose(data, constraints, config):
    """Supply a rejected native outcome to the independent interval diagnostic."""
    return e._diagnose_exact_infeasibility(
        data,
        constraints,
        config,
        e.ExactExecutionSolverAudit(),
        SimpleNamespace(status="infeasible", reason="test"),
    )
