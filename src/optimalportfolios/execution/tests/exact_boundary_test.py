"""Qualify conditional Exact sizing without weakening the original return audit."""

from dataclasses import replace
import numpy as np
import pytest

from optimalportfolios.execution import audit_execution, compile_states
from optimalportfolios.execution import exact_execution as exact
from optimalportfolios.execution.domain import variance
from optimalportfolios.execution.tests.formulation_test import example, independent_kkt


def boundary_case(side):
    """Supply a signed fractional row and an otherwise valid native boundary point."""
    p = example(coupled=False)
    target = p.current.copy()
    target[0] += .02 if side == "upper" else -.02
    target[-1] -= .02 if side == "upper" else -.02
    lo, hi = np.minimum(p.current, target), np.maximum(p.current, target)
    lo[-1], hi[-1] = 0., .1
    group = np.array([[.5, 0., -.25, 0., 0.]])
    endpoint = .01 if side == "upper" else 0.
    p = replace(p, target=target, lower=lo, upper=hi, full_lower=lo, full_upper=hi,
                covariance=np.eye(5), groups=group,
                group_lower=np.array([-.1 if side == "upper" else endpoint]),
                group_upper=np.array([endpoint if side == "upper" else .1]))
    raw = p.current.copy()
    raw[0] = (.05 if side == "upper" else .03) + (
        (1 if side == "upper" else -1) * (p.rules.audit_tolerance + 1e-15) / .5)
    raw[-1] += p.current[0] - raw[0]
    assert audit_execution(p, raw)["violations"] == ["groups"]
    return p, raw


def inject_native(monkeypatch, raw, source="native_return"):
    """Inject one native rejection without invoking a licensed integer backend."""
    def native(problem, model, initial, deadline, threads, gap_rel, retain):
        """Expose a native status independently from the return audit."""
        retain(raw, source)
        return dict(native_status="solsta.integer_optimal")
    monkeypatch.setattr(exact.cp, "installed_solvers", lambda: ["MOSEK", "CLARABEL"])
    monkeypatch.setattr(exact, "native_mosek", native)


@pytest.mark.parametrize("side", ["lower", "upper"])
@pytest.mark.parametrize("formulation", ["CORRIDOR", "FULL"])
def test_rejected_boundary_recovers_by_same_state_sizing(monkeypatch, side, formulation):
    """Recover a trade while retaining its raw rejection and unchanged constraints."""
    p, raw = boundary_case(side)
    inject_native(monkeypatch, raw)
    result = exact.wrapper_exact_execution(p, formulation, True, time_limit_s=10, threads=1)
    assert variance(p, result.weights) < variance(p, p.base) - 1e-5, (
        "Exact retained its inferior start after a repairable native boundary rejection")
    assert audit_execution(p, result.weights, formulation)["passed"]
    assert result.diagnostics["native_status"] == "solsta.integer_optimal"
    events = result.diagnostics["incumbent_events"]
    assert not next(e for e in events if e["source"] == "native_return")["audit"]["passed"]
    assert next(e for e in events if e["source"] == "boundary_refinement")["audit"]["passed"]
    assert result.diagnostics["first_native_audited_feasible_seconds"] is None
    refined = result.diagnostics["boundary_refinement"]
    assert refined["attempted"] and refined["status"] == "optimal"
    np.testing.assert_array_equal(refined["input_weights"], raw)
    assert refined["max_weight_change"] < 1e-7
    assert 0 < refined["native_time_limit_s"] <= 10
    states = compile_states(p, formulation)
    fixed = {i:(states[i][j].lower, states[i][j].upper)
             for i,j in refined["fixed_states"]}
    reference = replace(p, group_lower=p.group_lower-p.rules.audit_tolerance,
                        group_upper=p.group_upper+p.rules.audit_tolerance)
    assert abs(variance(p, result.weights)-independent_kkt(reference, fixed)) < 1e-11


def test_genuine_group_infeasibility_remains_rejected(monkeypatch):
    """An impossible fixed-state group cannot become an accepted refined portfolio."""
    p, raw = boundary_case("upper")
    p = replace(p, group_lower=np.array([.08]), group_upper=np.array([.1]))
    inject_native(monkeypatch, raw)
    result = exact.wrapper_exact_execution(p, "CORRIDOR", time_limit_s=10)
    assert result.weights is None
    assert result.diagnostics["boundary_refinement"]["status"] == "infeasible"
    assert all(not e["audit"]["passed"] for e in result.diagnostics["incumbent_events"])


def test_other_violations_do_not_trigger_refinement(monkeypatch):
    """The refinement is not a general repair of invalid native portfolios."""
    p, raw = boundary_case("upper")
    raw[-1] += .001
    inject_native(monkeypatch, raw)
    result = exact.wrapper_exact_execution(p, "CORRIDOR", time_limit_s=10)
    assert np.array_equal(result.weights, p.base)
    assert not result.diagnostics["boundary_refinement"]["requested"]


def test_deadline_exhausted_by_integer_solve_prevents_refinement(monkeypatch):
    """Do not give a failed integer solve an additional refinement time budget."""
    p, raw = boundary_case("upper")
    clock = [0.]
    monkeypatch.setattr(exact, "perf_counter", lambda: clock[0])
    inject_native(monkeypatch, raw)
    original = exact.native_mosek

    def consume(*args):
        """Finish the native phase after its declared end-to-end deadline."""
        result = original(*args)
        clock[0] = 11.
        return result

    monkeypatch.setattr(exact, "native_mosek", consume)
    result = exact.wrapper_exact_execution(p, "CORRIDOR", time_limit_s=10)
    assert np.array_equal(result.weights, p.base)
    assert result.diagnostics["boundary_refinement"]["status"] == "deadline"
    assert not result.diagnostics["boundary_refinement"]["attempted"]


def test_refinement_graph_construction_is_charged(monkeypatch):
    """A graph that exhausts the shared deadline must not start a native QP."""
    p, raw = boundary_case("upper")
    clock = [0.]
    monkeypatch.setattr(exact, "perf_counter", lambda: clock[0])
    original = exact.build_model

    def slow_graph(*args, **kwargs):
        """Consume time without modifying the sizing problem."""
        model = original(*args, **kwargs)
        clock[0] = 11.
        return model

    monkeypatch.setattr(exact, "build_model", slow_graph)
    weights, detail = exact._refine_group_candidate(p, raw, "CORRIDOR", 10.)
    assert weights is None and not detail["attempted"] and detail["status"] == "deadline"


@pytest.mark.parametrize("mode", ["empty", "outside"])
def test_refinement_requires_an_original_state(monkeypatch, mode):
    """Never invent a new interval to accommodate a native coordinate."""
    p, raw = boundary_case("upper")
    if mode == "empty":
        states = compile_states(p)
        monkeypatch.setattr(exact, "compile_states", lambda *args: ((), *states[1:]))
    else:
        raw[0] = .09
    weights, detail = exact._refine_group_candidate(p, raw, "CORRIDOR", np.inf)
    assert weights is None and detail["status"] == "no_matching_state"


def test_failed_refinement_preserves_native_status_and_incumbent(monkeypatch):
    """A refinement failure cannot overwrite raw native evidence or valid delivery."""
    p, raw = boundary_case("upper")
    inject_native(monkeypatch, raw)

    def failed(*args, **kwargs):
        """Simulate a numerical sizing error after the native rejection."""
        raise exact.cp.SolverError("injected refinement failure")

    monkeypatch.setattr(exact.cp.Problem, "solve", failed)
    result = exact.wrapper_exact_execution(p, "CORRIDOR", time_limit_s=10)
    assert np.array_equal(result.weights, p.base)
    assert result.diagnostics["native_status"] == "solsta.integer_optimal"
    assert result.diagnostics["boundary_refinement"]["status"] == "solver_error"


def test_refined_vector_still_requires_original_audit(monkeypatch):
    """An invalid QP result must stay rejected without recursively refining again."""
    p, raw = boundary_case("upper")
    inject_native(monkeypatch, raw)
    calls = []

    def invalid(*args):
        """Return an unchanged group breach while claiming optimal sizing."""
        calls.append(args)
        return raw, dict(attempted=True, status="optimal")

    monkeypatch.setattr(exact, "_refine_group_candidate", invalid)
    result = exact.wrapper_exact_execution(p, "CORRIDOR", time_limit_s=10)
    assert len(calls) == 1 and np.array_equal(result.weights, p.base)
    assert not result.diagnostics["boundary_refinement"]["output_audit"]["passed"]


def test_nonimproving_candidate_needs_no_refinement(monkeypatch):
    """A risk-indifferent invalid native candidate cannot displace a valid incumbent."""
    p, raw = boundary_case("upper")
    p = replace(p, covariance=np.zeros((5, 5)))
    inject_native(monkeypatch, raw)
    result = exact.wrapper_exact_execution(p, "CORRIDOR", time_limit_s=10)
    assert result.diagnostics["boundary_refinement"]["status"] == "incumbent_dominates"
    assert np.array_equal(result.weights, p.base)


def test_only_best_rejected_native_candidate_is_refined(monkeypatch):
    """Multiple callbacks cause one recorded sizing call, with stable candidate choice."""
    p, raw = boundary_case("upper")
    better = raw.copy()
    better[0] += 1e-5
    better[-1] -= 1e-5
    inject_native(monkeypatch, raw)

    def native(problem, model, initial, deadline, threads, gap_rel, retain):
        """Report a better early candidate followed by a worse final native return."""
        retain(better, "native_callback")
        retain(raw, "native_return")
        return dict(native_status="solsta.integer_optimal")

    monkeypatch.setattr(exact, "native_mosek", native)
    result = exact.wrapper_exact_execution(p, "CORRIDOR", time_limit_s=10)
    refinement = result.diagnostics["boundary_refinement"]
    np.testing.assert_array_equal(refinement["input_weights"], better)
    assert refinement["source"] == "native_callback"
    assert result.diagnostics["delivered_source"] == "boundary_refinement"
    assert audit_execution(p, result.weights)["passed"]
    assert sum(e["source"] == "boundary_refinement"
               for e in result.diagnostics["incumbent_events"]) == 1
