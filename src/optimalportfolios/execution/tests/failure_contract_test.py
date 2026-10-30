"""Qualify failure contracts without upgrading or assuming optional solver support."""

from dataclasses import replace
from time import perf_counter
from types import SimpleNamespace

import cvxpy as cp
import numpy as np
import pytest

from optimalportfolios.execution import (
    audit_execution,
    compile_states,
    compute_perspective_lower_bound,
    wrapper_ranked_execution,
    wrapper_exact_execution,
)
from optimalportfolios.execution.model import build_model, solve_qp
from optimalportfolios.execution.tests.formulation_test import example


def test_structured_sizing_and_required_category():
    """Equivalent exact risk inputs give equal sizing and unknown mandates fail closed."""
    p = example()
    eig, vec = np.linalg.eigh(p.covariance)
    structured = replace(
        p,
        covariance=None,
        factor_loadings=vec,
        factor_covariance=np.diag(eig),
        residual_variance=np.zeros(5),
    )
    assert np.allclose(structured.dense_covariance(), p.dense_covariance(), atol=1e-15)
    dense = wrapper_ranked_execution(p)
    other = wrapper_ranked_execution(structured)
    assert np.allclose(dense.weights, other.weights, atol=1e-8)
    q = p.base.copy()
    q[0] += 0.001
    q[-1] -= 0.001
    mask = np.array([True, False, False, False, False])
    with pytest.raises(ValueError, match="Mandatory interval"):
        replace(p, base=q, mandatory=mask, categories=("unregistered", "", "", "", ""))
    nested = replace(p, provenance={"original": {"ids": ["a", "b"]}})
    assert replace(nested, investment=0.2).provenance["original"]["ids"] == ("a", "b")


def test_invalid_zero_choice_is_not_a_feasible_incumbent():
    """An optional dust holding without an admissible exit requires an actual ticket."""
    p = example(0, coupled=False)
    q = p.base.copy()
    q[0], q[-1] = 0.003, 0.077
    lo, hi = p.lower.copy(), p.upper.copy()
    lo[0], hi[0] = 0.003, 0.008
    p = replace(p, current=q, base=q, target=q, lower=lo, upper=hi)
    assert all(s.ticket for s in compile_states(p)[0])
    out = wrapper_ranked_execution(p)
    assert out.weights is None and out.status == "bounded_abstention"
    # There is budget for admission, but the risk-only greedy choice does not
    # request it. The constraint rescue must consider admitting this instrument.
    permitted = replace(p, rules=replace(p.rules, ticket_cap=2))
    rescued = wrapper_ranked_execution(permitted)
    assert rescued.weights is not None and audit_execution(permitted, rescued.weights)["passed"]
    empty = replace(p, upper=np.array([0.004, *hi[1:]]))
    assert not compile_states(empty)[0]
    model = build_model(empty, continuous=True)
    assert solve_qp(model) is None


def test_excess_proposed_support_is_rejected(monkeypatch):
    """A defective selector cannot bypass the capacity guard at the sizing boundary."""
    import optimalportfolios.execution.ranked_execution as module

    p = example(0)
    states = compile_states(p)

    def overfull(*args):
        """Inject an inadmissible greedy proposal to verify independent admission."""
        chosen = {0: next(j for j, s in enumerate(states[0]) if s.ticket)}
        return chosen, p.base.copy(), [], 0.0

    monkeypatch.setattr(module, "ranked_selection", overfull)
    assert np.array_equal(wrapper_ranked_execution(p).weights, p.base)


def test_bound_certificate_fails_closed_on_bad_lp_or_mapping(monkeypatch):
    """Missing duals and unexpected canonical variables cannot become positive bounds."""
    import optimalportfolios.execution.bounds as module

    p = example()
    original = module.linprog

    def failed_lp(*args, **kwargs):
        """Simulate a certificate LP that did not close successfully."""
        return SimpleNamespace(success=False, message="injected LP failure")

    monkeypatch.setattr(module, "linprog", failed_lp)
    assert all(
        v["lower_bound"] == 0 and "LP failed" in v["certificate_error"]
        for v in compute_perspective_lower_bound(p).values()
    )
    monkeypatch.setattr(module, "linprog", original)
    original_data = cp.Problem.get_problem_data

    def changed_canonical(self, solver, *args, **kwargs):
        """Introduce an unrecognised canonical coordinate as a future adapter change."""
        data, chain, inverse = original_data(self, solver, *args, **kwargs)
        if solver == "SCIPY":
            data = dict(data, c=np.r_[data["c"], 0.0])
        return data, chain, inverse

    monkeypatch.setattr(cp.Problem, "get_problem_data", changed_canonical)
    assert all(
        v["lower_bound"] == 0 and "Unmapped" in v["certificate_error"]
        for v in compute_perspective_lower_bound(p).values()
    )


def test_requested_start_continuous_path_and_scip_dispatch(monkeypatch):
    """Record supplied starts honestly on continuous and unsupported dispatch paths."""
    p = example()
    fixed = replace(p, lower=p.base, upper=p.base, full_lower=p.base, full_upper=p.base)
    available = cp.installed_solvers()
    if "MOSEK" in available:
        result = wrapper_exact_execution(fixed, "CORRIDOR", mip_start=p.base)
        assert result.weights is not None
        assert result.diagnostics["initial_start"]["supplied"] is False
    monkeypatch.setattr(cp, "installed_solvers", lambda: ["SCIP"])
    seen = {}

    def unavailable(self, **kwargs):
        """Simulate a detected backend rejecting the actual requested conic program."""
        seen.update(kwargs)
        raise cp.SolverError("unsupported conic integer capability")

    monkeypatch.setattr(cp.Problem, "solve", unavailable)
    result = wrapper_exact_execution(p, "CORRIDOR", solver="SCIP", time_limit_s=3)
    assert seen["scip_params"]["limits/time"] <= 3
    assert result.diagnostics["native_status"] == "solver_error"
    assert np.array_equal(result.weights, p.base)


@pytest.mark.skipif(
    "MOSEK" not in cp.installed_solvers(), reason="Optional native adapter unavailable"
)
def test_native_version_deadline_and_callback_failure(monkeypatch):
    """Actual native paths preserve the external incumbent on unsupported or aborted solves."""
    import optimalportfolios.execution.exact_execution as module

    p = example()
    version = cp.__version__
    monkeypatch.setattr(cp, "__version__", "unqualified")
    unsupported = wrapper_exact_execution(p, "CORRIDOR")
    assert unsupported.diagnostics["native_status"] == "solver_error"
    assert unsupported.weights is not None
    monkeypatch.setattr(cp, "__version__", version)
    import mosek

    native_version = mosek.Env.getversion
    monkeypatch.setattr(mosek.Env, "getversion", staticmethod(lambda: (99, 0, 0)))
    assert wrapper_exact_execution(p, "CORRIDOR").diagnostics["native_status"] == "solver_error"
    monkeypatch.setattr(mosek.Env, "getversion", native_version)
    expired = wrapper_exact_execution(p, "CORRIDOR", time_limit_s=1e-12)
    assert expired.diagnostics["native_status"] == "preparation_deadline"
    assert module.lift_start(None, None, None, None, -1)["supplied"] is False
    clock = perf_counter
    elapsed = [False]

    def spent_lift(*args):
        """Simulate a charged conic lift consuming the remaining budget."""
        elapsed[0] = True
        return dict(requested=True, supplied=False, received=None)

    monkeypatch.setattr(module, "lift_start", spent_lift)
    monkeypatch.setattr(module, "perf_counter", lambda: clock() + (10000 if elapsed[0] else 0))
    expired = wrapper_exact_execution(p, "CORRIDOR", time_limit_s=1)
    assert expired.diagnostics["native_status"] == "preparation_deadline"
    monkeypatch.undo()

    def failed_inverse(*args, **kwargs):
        """Inject a native callback mapping failure, retaining the external return gate."""
        raise ValueError("injected canonical mapping error")

    monkeypatch.setattr(module, "invert_native", failed_inverse)
    failed = wrapper_exact_execution(p, "CORRIDOR", threads=1, time_limit_s=10)
    if "license" in failed.diagnostics.get("error", "").lower():
        pytest.skip("Optional native licence unavailable")
    assert failed.diagnostics["callback_errors"]
    assert failed.weights is not None
