"""Counterexamples for bounded search, admission gates and malformed domains."""

from dataclasses import replace
from types import SimpleNamespace

import cvxpy as cp
import numpy as np
import pandas as pd
import pytest

from optimalportfolios.execution.v2 import (
    ExecutionRules,
    RankedConfig,
    audit_execution,
    compile_states,
    compute_perspective_lower_bound,
    from_resolved_problem,
    wrapper_exact_execution,
    wrapper_ranked_execution,
)
from optimalportfolios.execution.v2.domain import result
from optimalportfolios.execution.v2.types import checked_cap, frozen_array
from optimalportfolios.execution.v2.tests.formulation_test import example


@pytest.mark.parametrize("value", [-1, True, 1.2])
def test_bad_capacity(value):
    """A soft or negative capacity cannot enter the hard ticket contract."""
    with pytest.raises(ValueError):
        checked_cap(value)


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(mandatory_count_in_cap=False),
        dict(minimum_ticket=0.0),
        dict(minimum_ticket=1e-10),
        dict(audit_tolerance=np.nan),
    ],
)
def test_bad_rules(kwargs):
    """Reject conventions that would erase mandatory tickets or their audit."""
    with pytest.raises(ValueError):
        ExecutionRules(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [dict(rescue_trials=None), dict(mandatory_state_trials=0), dict(objective_tolerance=-1.0)],
)
def test_bad_rank_controls(kwargs):
    """Unbounded or negative search controls are not accepted."""
    with pytest.raises(ValueError):
        RankedConfig(**kwargs)


def test_array_and_schema_validation():
    """Reject nonfinite inputs, incompatible labels, masks and covariance conventions."""
    for value, shape, dtype in [([np.nan], None, float), ([2], None, bool), ([1], (2,), float)]:
        with pytest.raises(ValueError):
            frozen_array(value, shape, dtype)
    p = example()
    cases = [
        dict(names=("same",) * 5),
        dict(annualisation="daily"),
        dict(full_lower=np.ones(5)),
        dict(base=p.base + 1e-4),
        dict(rules=ExecutionRules(ticket_cap=2, discretionary_allowance=1)),
        dict(factor_loadings=np.zeros((5, 1))),
        dict(covariance=None),
        dict(covariance=np.triu(np.ones((5, 5)))),
        dict(frozen=np.ones(5, bool)),
        dict(preparation_seconds=-1),
        dict(group_upper=np.array([0.01])),
    ]
    for changes in cases:
        with pytest.raises(ValueError):
            replace(p, **changes)
    q = p.base.copy()
    q[-1] += 0.001
    with pytest.raises(ValueError, match="funding"):
        replace(p, base=q)
    with pytest.raises(ValueError, match="residual"):
        replace(
            p,
            covariance=None,
            factor_loadings=np.zeros((5, 1)),
            factor_covariance=np.ones((1, 1)),
            residual_variance=-np.ones(5),
        )
    meta = replace(p, provenance={"currency": "USD", "inputs": {"frozen": True}, "labels": ["a"]})
    assert meta.provenance["inputs"]["frozen"] is True
    with pytest.raises(TypeError):
        meta.provenance["inputs"]["frozen"] = False


def test_audit_and_result_gate():
    """Independent admission rejects invalid vectors, floors and discretionary overuse."""
    p = example(0)
    assert not audit_execution(p, [1.0, 2.0])["passed"]
    with pytest.raises(ValueError):
        compile_states(p, "UNKNOWN")
    w = p.base.copy()
    w[0], w[-1] = 0.001, w[-1] + 0.039
    assert "holding_floor" in audit_execution(p, w)["violations"]
    with pytest.raises(ValueError, match="unaudited"):
        result(p, w, "wrong", {})
    p = replace(p, rules=ExecutionRules(ticket_cap=0, discretionary_allowance=0))
    w = p.base.copy()
    w[0] += 0.01
    w[-1] -= 0.01
    assert "discretionary_allowance" in audit_execution(p, w)["violations"]
    with pytest.raises(ValueError):
        wrapper_exact_execution(p, "CORRIDOR", mip_start=w)
    with pytest.raises(ValueError):
        wrapper_exact_execution(p, "CORRIDOR", time_limit_s=0)


def test_rescue_and_sizing_error_preserve_common_incumbent(monkeypatch):
    """An infeasible greedy support triggers bounded coupled-constraint search."""
    import optimalportfolios.execution.v2.ranked as module

    p = replace(
        example(2),
        group_lower=np.array([0.07]),
        group_upper=np.array([0.07]),
        covariance=np.diag([1.0, 0.0001, 0.0001, 0.9, 0.0]),
    )
    output = wrapper_ranked_execution(p)
    assert output.weights is not None
    assert output.diagnostics["qp_counts"]["rescue"] > 0
    original = module.solve_qp

    def failure(*args, **kwargs):
        """Simulate an actual numerical backend error without changing the common audit."""
        raise cp.SolverError("injected sizing failure")

    monkeypatch.setattr(module, "solve_qp", failure)
    failed = wrapper_ranked_execution(p)
    assert np.array_equal(failed.weights, p.base)
    assert failed.diagnostics["qp_counts"]["rescue"] <= RankedConfig().rescue_trials
    monkeypatch.setattr(module, "solve_qp", original)


def test_mandatory_holding_alternatives_and_empty_domain():
    """A compulsory sale preserves both valid floor states and reports truncation."""
    p = example()
    q, lo, hi = p.base.copy(), p.lower.copy(), p.upper.copy()
    q[0], q[-1], lo[0], hi[0] = 0.02, 0.06, 0.0, 0.03
    fl, fu = p.full_lower.copy(), p.full_upper.copy()
    fl[0], fu[0] = 0.0, 0.03
    mask = np.array([True, False, False, False, False])
    p = replace(
        p,
        base=q,
        lower=lo,
        upper=hi,
        full_lower=fl,
        full_upper=fu,
        mandatory=mask,
        categories=("product_max_sell", "", "", "", ""),
        rules=ExecutionRules(exemption_mask=mask, holding_floor=0.005, ticket_cap=2),
    )
    output = wrapper_ranked_execution(p, RankedConfig(mandatory_state_trials=1))
    assert output.diagnostics["mandatory_state_truncated"]
    assert output.weights is not None
    hi[0] = 0.003
    lo[0] = 0.001
    empty = replace(p, lower=lo, upper=hi, full_lower=lo, full_upper=hi)
    assert not compile_states(empty)[0]
    assert wrapper_ranked_execution(empty).status == "domain_conflict"


def test_constant_bounds_and_solver_failure(monkeypatch):
    """A fully fixed graph and a missing conic backend have explicit bound results."""
    p = example()
    fixed = replace(p, lower=p.base, upper=p.base, full_lower=p.base, full_upper=p.base)
    bounds = compute_perspective_lower_bound(fixed)
    assert all("constant" in item["certificate"] for item in bounds.values())
    out = wrapper_exact_execution(fixed, "CORRIDOR", solver="CLARABEL")
    assert out.weights is not None

    def failure(*args, **kwargs):
        """Exercise the documented unavailable-certificate and solver-error paths."""
        raise cp.SolverError("injected backend error")

    monkeypatch.setattr(cp.Problem, "solve", failure)
    bounds = compute_perspective_lower_bound(p)
    assert all(item["lower_bound"] == 0 and "certificate_error" in item for item in bounds.values())
    out = wrapper_exact_execution(fixed, "CORRIDOR", solver="CLARABEL")
    assert out.diagnostics["native_status"] == "solver_error"
    assert np.array_equal(out.weights, fixed.base)


def test_adapter_retains_final_state_and_rejects_extra_rules():
    """Adapting old labels cannot reset c, rebuild a lifecycle state or lose a hard rule."""
    p = example()
    prepared = {
        key: getattr(p, key)
        for key in (
            "names",
            "cash",
            "current",
            "base",
            "lower",
            "upper",
            "mandatory",
            "categories",
            "groups",
            "group_lower",
            "group_upper",
            "investment",
        )
    }
    old = SimpleNamespace(
        target=pd.DataFrame(
            {"current_weight": p.current, "raw_model_weight": p.target}, index=p.names
        ),
        covariance=pd.DataFrame(p.covariance, index=p.names, columns=p.names),
        constraints=SimpleNamespace(),
    )
    adapted = from_resolved_problem(
        old, p.rules, prepared=prepared, full_lower=p.full_lower, full_upper=p.full_upper
    )
    assert np.array_equal(adapted.covariance, p.covariance)
    old.constraints.is_long_only = False
    with pytest.raises(ValueError, match="long-only"):
        from_resolved_problem(
            old, p.rules, prepared=prepared, full_lower=p.full_lower, full_upper=p.full_upper
        )
    old.constraints.is_long_only = True
    old.constraints.group_lower_upper_constraints = SimpleNamespace(
        group_loadings=pd.DataFrame(p.groups.T, index=p.names, columns=["group"]),
        group_min_allocation=pd.Series(p.group_lower, index=["group"]),
        group_max_allocation=pd.Series(p.group_upper, index=["group"]),
    )
    from_resolved_problem(
        old, p.rules, prepared=prepared, full_lower=p.full_lower, full_upper=p.full_upper
    )
    with pytest.raises(ValueError, match="group rule"):
        from_resolved_problem(
            old,
            p.rules,
            prepared=prepared | {"group_upper": [0.1]},
            full_lower=p.full_lower,
            full_upper=p.full_upper,
        )
    for change in ({"names": ("wrong",) * 5}, {"current": np.zeros(5)}):
        with pytest.raises(ValueError):
            from_resolved_problem(
                old,
                p.rules,
                prepared=prepared | change,
                full_lower=p.full_lower,
                full_upper=p.full_upper,
            )
    old.constraints.turnover_constraint = 0.1
    with pytest.raises(ValueError, match="Unsupported"):
        from_resolved_problem(
            old, p.rules, prepared=prepared, full_lower=p.full_lower, full_upper=p.full_upper
        )
