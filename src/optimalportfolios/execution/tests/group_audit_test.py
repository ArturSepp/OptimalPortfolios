"""Keep native group rows and the independent audit on one numerical contract."""

from dataclasses import replace

import cvxpy as cp
import numpy as np
import pytest

from optimalportfolios.execution import audit_execution, compile_states, wrapper_ranked_execution
from optimalportfolios.execution.domain import variance
from optimalportfolios.execution.model import build_model, solve_qp
from optimalportfolios.execution.tests.formulation_test import example, independent_kkt


def pinned_group(side, gap, tolerance=1e-7):
    """Pin a signed fractional group while leaving an independently funded trade."""
    p = example()
    lo, hi = p.lower.copy(), p.upper.copy()
    fl, fu = p.full_lower.copy(), p.full_upper.copy()
    lo[:2] = hi[:2] = fl[:2] = fu[:2] = p.current[:2]
    groups = np.array([[0.5, -0.25, 0.0, 0.0, 0.0]])
    endpoint = (groups @ p.current).item()
    lower, upper = endpoint - 0.01, endpoint + 0.01
    if side == "lower":
        lower = endpoint + gap
    else:
        upper = endpoint - gap
    return replace(p, lower=lo, upper=hi, full_lower=fl, full_upper=fu, groups=groups,
                   covariance=np.eye(5),
                   group_lower=np.array([lower]), group_upper=np.array([upper]),
                   rules=replace(p.rules, audit_tolerance=tolerance))


@pytest.mark.parametrize("side", ["lower", "upper"])
@pytest.mark.parametrize("formulation", ["CORRIDOR", "FULL"])
@pytest.mark.parametrize("fraction", [0.00583172087, 0.99])
def test_audit_admitted_group_conflict_sizes_same_trade(side, formulation, fraction):
    """Tiny literal contradictions must not suppress an otherwise admissible trade."""
    p = pinned_group(side, fraction * 1e-7)
    original = (p.group_lower.copy(), p.group_upper.copy())
    states = compile_states(p, formulation)
    fixed = {i: next(j for j, s in enumerate(row) if s.ticket == (i == 2)
                    and (i != 2 or s.lower <= p.target[i] <= s.upper))
             for i, row in enumerate(states[:-1])}
    model = build_model(p, formulation, fixed_states=fixed)
    weights = solve_qp(model)
    assert weights is not None, "An audit-admitted pinned group blocked joint sizing"
    assert audit_execution(p, weights, formulation)["passed"]
    # The group is constant: independently remove its redundant row and enumerate KKT sets.
    reference = replace(p, group_lower=np.array([-1.]), group_upper=np.array([1.]))
    expected = independent_kkt(reference, {
        i: (states[i][j].lower, states[i][j].upper) for i, j in fixed.items()})
    assert abs(variance(p, weights) - expected) < 1e-12
    assert variance(p, weights) < variance(p, p.base) - 1e-6
    assert np.array_equal(p.group_lower, original[0])
    assert np.array_equal(p.group_upper, original[1])
    assert model.optimisation.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)


@pytest.mark.parametrize("side", ["lower", "upper"])
@pytest.mark.parametrize("fraction", [1.01, 100.])
def test_group_breach_beyond_existing_tolerance_stays_infeasible(side, fraction):
    """Native sizing and the independent gate both reject a genuine group breach."""
    p = pinned_group(side, fraction * 1e-7)
    model = build_model(p, continuous=True)
    assert solve_qp(model) is None
    assert model.optimisation.status == cp.INFEASIBLE
    assert "groups" in audit_execution(p, p.base)["violations"]
    assert wrapper_ranked_execution(p).weights is None


@pytest.mark.parametrize("perspective", [False, True])
def test_integer_and_continuous_graphs_use_same_group_envelope(perspective):
    """Exact, its relaxation and ranked sizing share identical group rows."""
    p = pinned_group("lower", 5.83e-10)
    graphs = [build_model(p, perspective=perspective, continuous=c) for c in (False, True)]
    for graph in graphs:
        initial = graph.initial_values(p.base)
        for variable in graph.optimisation.variables():
            if variable.id in initial:
                variable.value = initial[variable.id]
        assert max(float(np.max(row.violation())) for row in graph.linear_constraints) < 1e-10
    assert solve_qp(graphs[1]) is not None


def test_configured_tolerance_and_return_gate_remain_authoritative(monkeypatch):
    """Use the declared tolerance, and reject a native candidate that exceeds it."""
    p = pinned_group("lower", 5e-9, tolerance=1e-9)
    assert solve_qp(build_model(p, continuous=True)) is None
    p = pinned_group("lower", 5e-9, tolerance=1e-8)
    assert solve_qp(build_model(p, continuous=True)) is not None
    import optimalportfolios.execution.ranked_execution as ranked

    def bad_candidate(model, solver):
        """Inject a funding-preserving but group-invalid native result."""
        w = p.base.copy()
        w[0] -= 1e-5
        w[-1] += 1e-5
        return w

    monkeypatch.setattr(ranked, "solve_qp", bad_candidate)
    result = wrapper_ranked_execution(p)
    assert np.array_equal(result.weights, p.base)
    assert all(not t["audit"]["passed"] for t in result.diagnostics["trials"])
