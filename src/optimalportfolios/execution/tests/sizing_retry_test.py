"""Verify a bounded, constraint-preserving retry after CLARABEL numerical errors."""

from dataclasses import replace

import cvxpy as cp
import numpy as np
import pytest

from optimalportfolios.execution import audit_execution, compile_states, wrapper_ranked_execution
from optimalportfolios.execution.domain import variance
from optimalportfolios.execution.model import build_model, solve_qp
from optimalportfolios.execution.tests.formulation_test import example, independent_kkt


def fixed_model(problem):
    """Select two valid trades and freeze the remaining synthetic names."""
    states = compile_states(problem)
    fixed = {i: next(j for j, s in enumerate(row) if s.ticket == (i < 2))
             for i, row in enumerate(states[:-1])}
    return build_model(problem, fixed_states=fixed), fixed, states


def test_retry_preserves_graph_tolerances_and_independent_optimum(monkeypatch):
    """A fresh unscaled solve recovers the same QP and independent active-set optimum."""
    p = example()
    model, fixed, states = fixed_model(p)
    original, calls = cp.Problem.solve, []

    def fail_first(self, **kwargs):
        """Fail only the first call and execute the genuine numerical retry."""
        calls.append((self, kwargs))
        if len(calls) == 1:
            raise cp.SolverError("InsufficientProgress")
        return original(self, **kwargs)

    monkeypatch.setattr(cp.Problem, "solve", fail_first)
    weights = solve_qp(model)
    assert len(calls) == 2 and calls[0][0] is calls[1][0] is model.optimisation
    assert calls[1][1] == dict(calls[0][1], warm_start=False, equilibrate_enable=False)
    assert calls[0][1] == dict(solver="CLARABEL", tol_gap_abs=1e-10,
                              tol_gap_rel=1e-10, tol_feas=1e-10)
    reference = independent_kkt(
        p, {i: (states[i][j].lower, states[i][j].upper) for i, j in fixed.items()})
    assert abs(variance(p, weights) - reference) < 1e-12
    assert audit_execution(p, weights)["passed"]
    assert model.qp_diagnostics["attempts"] == 2
    assert model.qp_diagnostics["initial_error"] == "InsufficientProgress"


@pytest.mark.parametrize("solver, expected", [("CLARABEL", 2), ("UNAVAILABLE", 1)])
def test_retry_is_bounded_and_errors_propagate(monkeypatch, solver, expected):
    """Repeated failure is never treated as a solution or an unbounded retry loop."""
    model, _, _ = fixed_model(example())
    calls = []

    def failed(self, **kwargs):
        """Reject every attempt to verify the retry limit and backend isolation."""
        calls.append(kwargs)
        raise cp.SolverError("persistent failure")

    monkeypatch.setattr(cp.Problem, "solve", failed)
    with pytest.raises(cp.SolverError, match="persistent failure"):
        solve_qp(model, solver)
    assert len(calls) == expected


@pytest.mark.parametrize("breach", ["cap", "group"])
def test_true_infeasibility_does_not_retry(monkeypatch, breach):
    """Ticket overuse and impossible group rows remain genuinely infeasible."""
    p = example()
    if breach == "cap":
        p = replace(p, rules=replace(p.rules, ticket_cap=0))
    else:
        p = replace(p, group_lower=np.array([0.15]), group_upper=np.array([0.16]))
    model, _, _ = fixed_model(p)
    original, calls = cp.Problem.solve, []

    def counted(self, **kwargs):
        """Count native calls without altering the problem or solver result."""
        calls.append(kwargs)
        return original(self, **kwargs)

    monkeypatch.setattr(cp.Problem, "solve", counted)
    assert solve_qp(model) is None
    assert model.optimisation.status == cp.INFEASIBLE
    assert len(calls) == 1 and model.qp_diagnostics["attempts"] == 1


def test_retry_output_still_requires_original_space_audit(monkeypatch):
    """A bad retry vector cannot bypass the wrapper's mandatory return gate."""
    from optimalportfolios.execution._sizing import _CachedSizingModel
    p = example()
    calls = {}

    def unaudited(self, **kwargs):
        """Inject a first-call failure followed by an out-of-bounds displacement."""
        calls[self] = calls.get(self, 0) + 1
        if calls[self] == 1:
            raise cp.SolverError("numerical failure")
        graph = self.template[0].optimisation if isinstance(self, _CachedSizingModel) else self
        for variable in graph.variables():
            variable.value = np.full(variable.shape, 1e6)
        graph._status = cp.OPTIMAL

    monkeypatch.setattr(cp.Problem, "solve", unaudited)
    monkeypatch.setattr(_CachedSizingModel, "solve", unaudited)
    out = wrapper_ranked_execution(p)
    assert np.array_equal(out.weights, p.base)
    # The restart can also size the empty support: its weights are constants,
    # so corrupting solver variables cannot alter that legitimate baseline.
    corrupt = [t for t in out.diagnostics["trials"] if t["support"]]
    assert corrupt and all(not t["audit"]["passed"] for t in corrupt)
    assert all(not t["support"] for t in out.diagnostics["trials"] if t["audit"]["passed"])
    assert all(t["sizing_solver"]["attempts"] == 2 for t in out.diagnostics["trials"])
