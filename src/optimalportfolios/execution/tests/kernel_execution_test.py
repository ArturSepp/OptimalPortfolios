"""Check Kernel domains, cost-aware incumbents, stages and native delivery."""

from dataclasses import replace

import cvxpy as cp
import numpy as np
import pytest

from optimalportfolios.execution import KernelConfig, wrapper_kernel_execution
from optimalportfolios.execution import kernel_execution as kernel
from optimalportfolios.execution.domain import audit_execution, objective, variance
from optimalportfolios.execution.tests.formulation_test import example
from optimalportfolios.execution.tests.scored_execution_test import (
    enumerate_reference, scalar_problem, six_name_problem,
)


@pytest.mark.parametrize("field,value", [
    ("time_limit_s", 0.), ("time_limit_s", -1.), ("time_limit_s", np.inf),
    ("time_limit_s", np.nan), ("minimum_kernel", 0), ("minimum_kernel", True),
    ("minimum_kernel", 1.5), ("kernel_per_ticket", -1),
])
def test_invalid_configuration(field, value):
    """Reject unusable allowances and ambiguous integer kernel settings."""
    with pytest.raises(ValueError):
        KernelConfig(**{field: value})


def test_partition_and_pins():
    """Occupancy, name ties, bucket sizes and pins preserve declared hard rows."""
    p = example()
    config = KernelConfig(minimum_kernel=1, kernel_per_ticket=1)
    first, buckets = kernel._search_partition([0, 1, 2, 3], {2: .9}, 2, p.names, config)
    assert first == {0, 2} and buckets == [[1, 3]]
    assert kernel._search_partition([], {}, 2, p.names, config) == (set(), [])
    restricted = kernel._restricted_problem(p, first)
    np.testing.assert_array_equal(restricted.lower[[1, 3]], p.base[[1, 3]])
    np.testing.assert_array_equal(restricted.upper[[1, 3]], p.base[[1, 3]])
    np.testing.assert_array_equal(restricted.groups, p.groups)
    assert restricted.rules is p.rules and restricted.lower[p.cash] == p.lower[p.cash]


def test_cost_cutoff_and_incumbent_use_full_objective(monkeypatch):
    """A lower-variance but more expensive trade fails the F cutoff and retention."""
    p = replace(scalar_problem(), trade_cost=np.array([.003, 0.]))
    candidate = p.target.copy()
    assert variance(p, candidate) < variance(p, p.base)
    assert objective(p, candidate) > objective(p, p.base)

    def native(problem, model, initial, deadline, threads, gap, retain):
        """Inspect the actual cutoff and inject independently audited candidates."""
        assert threads == 1 and gap == 1e-4
        np.testing.assert_array_equal(initial, p.base)
        cutoff = model.optimisation.constraints[-1]
        assert float(cutoff.args[1].value) == pytest.approx(
            kernel.SCALE * (objective(p, p.base) + 1e-12), rel=0, abs=1e-12)
        retain(np.array([np.nan, np.nan]), "invalid")
        retain(candidate, "lower_variance_higher_F")
        return {"native_status": "injected"}

    monkeypatch.setattr(kernel, "native_mosek", native)
    solved = wrapper_kernel_execution(p)
    np.testing.assert_array_equal(solved.weights, p.base)
    events = solved.diagnostics["incumbent_events"]
    assert events[0]["scored_objective"] is None
    assert "improved" not in events[1]
    assert solved.diagnostics["bounds_scope"] == "restricted_subproblems_only"


def test_bucket_rows_and_restricted_audit(monkeypatch):
    """Reject original-feasible trades outside the kernel or without bucket use."""
    p = six_name_problem(0.)
    config = KernelConfig(minimum_kernel=1, kernel_per_ticket=1)
    stages = []

    def guidance_failure(self, **kwargs):
        """Exercise documented fallback to identifier order."""
        raise cp.SolverError("guidance unavailable")

    def native(problem, model, initial, deadline, threads, gap, retain):
        """Expose the bucket requirement and both independent audits."""
        stages.append(problem)
        if len(stages) == 1:
            assert initial is not None
            outside = p.base.copy()
            outside[4] += .01
            outside[-1] -= .01
            retain(outside, "outside_kernel")
        else:
            assert initial is None
            retain(p.base, "unused_bucket")
        weights = p.base.copy()
        i = [0, 2, 4][len(stages) - 1]
        weights[i] += .01
        weights[-1] -= .01
        retain(weights, "used_bucket")
        return {"native_status": "injected"}

    monkeypatch.setattr(cp.Problem, "solve", guidance_failure)
    monkeypatch.setattr(kernel, "native_mosek", native)
    solved = wrapper_kernel_execution(p, config)
    assert len(stages) == 3
    events = solved.diagnostics["incumbent_events"]
    assert events[0]["audit"]["passed"] and not events[0]["restricted_audit"]["passed"]
    assert not events[2]["bucket_used"] and events[3]["bucket_used"]
    assert 2 in solved.diagnostics["subproblems"][2]["kernel"]


def test_later_stage_cutoff_includes_incumbent_cost(monkeypatch):
    """After a costly improvement the next cutoff must retain its nonzero cost."""
    p = six_name_problem(1.)
    incumbent = p.base.copy()
    incumbent[0] += .01
    incumbent[-1] -= .01
    assert variance(p, incumbent) < objective(p, incumbent) < objective(p, p.base)
    calls = []

    def fail_guidance(self, **kwargs):
        """Fix name ordering independently of the relaxation numerics."""
        raise cp.SolverError("name-order fixture")

    def native(problem, model, initial, deadline, threads, gap, retain):
        """Check the second stage's actual graph, not just its reported value."""
        calls.append(initial)
        if len(calls) == 1:
            retain(incumbent, "costly_improvement")
        else:
            cutoff = model.optimisation.constraints[-1]
            assert float(cutoff.args[1].value) == pytest.approx(
                kernel.SCALE * (objective(p, incumbent) + 1e-12), rel=0, abs=1e-12)
        return {"native_status": "injected"}

    monkeypatch.setattr(cp.Problem, "solve", fail_guidance)
    monkeypatch.setattr(kernel, "native_mosek", native)
    solved = wrapper_kernel_execution(p, KernelConfig(minimum_kernel=1, kernel_per_ticket=1))
    assert len(calls) == 3
    np.testing.assert_array_equal(solved.weights, incumbent)


@pytest.mark.parametrize("find_feasible", [False, True])
def test_invalid_base_requires_audited_incumbent(monkeypatch, find_feasible):
    """An invalid base cannot become a delivery merely because a backend fails."""
    p = replace(scalar_problem(), groups=np.array([[1., 0.]]),
                group_lower=np.array([.3]), group_upper=np.array([.5]))

    def native(problem, model, initial, deadline, threads, gap, retain):
        """Either produce a feasible result or fail without one."""
        assert initial is None
        if find_feasible:
            retain(p.target, "first_feasible")
        else:
            raise RuntimeError("backend unavailable")
        return {"native_status": "injected"}

    monkeypatch.setattr(kernel, "native_mosek", native)
    solved = wrapper_kernel_execution(p)
    assert (solved.weights is not None) == find_feasible
    if find_feasible:
        assert solved.diagnostics["first_audited_feasible_seconds"] is not None
    else:
        assert solved.status == "abstention"
        assert "error" in solved.diagnostics["subproblems"][0]


def test_guidance_consumes_allowance(monkeypatch):
    """No native stage starts after the charged relaxation exhausts the allowance."""
    clock = [0.]

    def consume(self, **kwargs):
        """Advance only the injected search clock during guidance."""
        clock[0] = 2.
        raise cp.SolverError("deadline")

    monkeypatch.setattr(kernel, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(cp.Problem, "solve", consume)
    p = example()
    solved = wrapper_kernel_execution(p, KernelConfig(time_limit_s=1.))
    np.testing.assert_array_equal(solved.weights, p.base)
    assert not solved.diagnostics["subproblems"]


def test_no_candidate_continuous_stage():
    """A pinned book has no buckets and retains its independently audited base."""
    p = scalar_problem()
    p = replace(p, lower=p.base, upper=p.base)
    solved = wrapper_kernel_execution(p)
    np.testing.assert_array_equal(solved.weights, p.base)
    assert solved.diagnostics["candidate_instruments"] == 0
    assert solved.diagnostics["total_buckets"] == 0
    assert solved.diagnostics["subproblems"][0]["native"]["native_status"] == cp.OPTIMAL


@pytest.mark.skipif("MOSEK" not in cp.installed_solvers(), reason="Optional native backend")
@pytest.mark.parametrize("cost", [0., 1.])
def test_full_kernel_matches_independent_reference_and_exact(cost):
    """Full-kernel F matches enumerated supports and native Exact at the target gap."""
    from optimalportfolios.execution import wrapper_exact_execution

    p = six_name_problem(cost)
    reference, _ = enumerate_reference(p)
    solved = wrapper_kernel_execution(p, KernelConfig(time_limit_s=10.))
    if any("license" in s.get("error", "").lower()
           for s in solved.diagnostics["subproblems"]):
        pytest.skip("Optional native licence unavailable")
    exact = wrapper_exact_execution(p, "CORRIDOR", time_limit_s=10., threads=1)
    assert audit_execution(p, solved.weights)["passed"]
    assert objective(p, solved.weights) == pytest.approx(reference, rel=1e-4)
    assert objective(p, solved.weights) == pytest.approx(objective(p, exact.weights), rel=1e-4)
    assert solved.diagnostics["total_buckets"] == 0
