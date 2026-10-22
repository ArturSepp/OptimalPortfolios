"""Exercise optional adapter contracts without a proprietary solver or licence.

The native task double checks starts, callbacks, budget accounting and canonical
inversion. Real MOSEK formulation tests remain separate numerical qualification.
"""

from dataclasses import replace
from types import SimpleNamespace
import sys

import cvxpy as cp
import numpy as np
import pytest

from optimalportfolios.execution.v2 import audit_execution, wrapper_exact_execution
from optimalportfolios.execution.v2 import exact
from optimalportfolios.execution.v2.model import SCALE, build_model
from optimalportfolios.execution.v2.tests.formulation_test import example


@pytest.fixture
def native(monkeypatch):
    """Provide an explicit native protocol double with independently chosen vectors."""
    from cvxpy.reductions.cone2cone.affine2direct import FREE
    from cvxpy.reductions.solvers.conic_solvers.mosek_conif import MOSEK

    p = example()
    state = SimpleNamespace(clock=0.0, tasks=[], receipt=True, bad_callback=False,
                            lift_exhausts_budget=False, mappings=[])
    vector = np.array([50., 0., 0., 0., 1., 0., 0.])
    direction = np.vstack([np.eye(4), -np.ones(4)])
    data = dict(bool_vars_idx=[4, 5], int_vars_idx=[6],
                param_prob=SimpleNamespace(var_id_to_col={7: 0, 8: 4}))

    def initial_values(weights):
        """Map the independently supplied portfolio and one known selector pair."""
        return {7: SCALE * (weights[:4] - p.base[:4]), 8: np.array([1., 0.])}

    def invert(payload, inverse):
        """Keep callback and final native mappings distinct, with an objective offset."""
        state.mappings.append(payload)
        if "sol" in payload:
            if state.bad_callback:
                raise ValueError("deliberately invalid callback map")
            raw = payload["sol"]
            assert np.array_equal(raw.primal_vars[FREE], vector[:4])
            return SimpleNamespace(primal_vars={7: raw.primal_vars[FREE]},
                                   opt_val=raw.opt_val + 17.)
        return SimpleNamespace(status=cp.OPTIMAL, primal_vars={7: vector[:4]})

    chain = SimpleNamespace(invert=invert)
    inverse = [{"K_dir": {FREE: 4}}]
    model = SimpleNamespace(
        reference=p.base, direction=direction, constant=0.0123, selectors={0: object()},
        displacement_bp=SimpleNamespace(id=7), initial_values=initial_values,
        optimisation=SimpleNamespace(get_problem_data=lambda solver: (data, chain, inverse)),
    )

    class Task:
        """Record the adapter's native calls without solving an optimisation problem."""

        def __init__(self, original=None):
            """Distinguish a fixed-start lift from the main task."""
            self.original, self.fixed, self.timers = original, {}, {}
            self.callback, self.supplied, self.converted = None, None, None
            state.tasks.append(self)

        def __enter__(self):
            """Expose the resource used by the real context-manager protocol."""
            return self

        def __exit__(self, *args):
            """Accept cleanup without hiding an adapter exception."""
            return False

        def putvartypelist(self, indices, types):
            """Record exact conversion of both Boolean and integer variables."""
            self.converted = (indices, types)

        def putvarbound(self, index, kind, lower, upper):
            """Require genuine fixed endpoints in the continuous start lift."""
            assert kind == "fixed" and lower == upper
            self.fixed[index] = lower

        def putdouparam(self, key, value):
            """Record remaining native time limits."""
            self.timers[key] = value

        def optimize(self):
            """Emit one incumbent event, or consume the lift budget on request."""
            if self.original is not None and state.lift_exhausts_budget:
                state.clock = 20.
            if self.callback is not None:
                state.callback_return = self.callback("new_integer", {"objective": 23.}, [], [])
            return "terminated"

        def getsolsta(self, kind):
            """Return a deterministic status for the lift or integer solve."""
            return "optimal" if kind == "continuous" else "integer_optimal"

        def getxx(self, kind):
            """Respect the fixed start values while returning a distinct final vector."""
            answer = vector.copy()
            for index, value in self.fixed.items():
                answer[index] = value
            return answer

        def putxx(self, kind, values):
            """Record the complete submitted start without claiming solver receipt."""
            assert kind == "integer"
            self.supplied = np.array(values)

        def set_InfoCallback(self, callback):
            """Install the adapter's actual incumbent callback."""
            self.callback = callback

        def getprosta(self, kind):
            """Return diagnostic problem status only."""
            return "primal_feasible"

        def getintinf(self, name):
            """Keep reported start receipt separate from successful submission."""
            return int(state.receipt) if name == "start_received" else 1

        def getdouinf(self, name):
            """Return diagnostic native bounds, not certified lower bounds."""
            return 12. if name == "bound" else .01

    api = SimpleNamespace(
        Task=Task, Env=SimpleNamespace(getversion=lambda: (11, 2, 3)),
        variabletype=SimpleNamespace(type_cont="continuous"), boundkey=SimpleNamespace(fx="fixed"),
        dparam=SimpleNamespace(optimizer_max_time="wall", mio_max_time="mip"),
        soltype=SimpleNamespace(itr="continuous", itg="integer"),
        solsta=SimpleNamespace(optimal="optimal", prim_feas="feasible"),
        callbackcode=SimpleNamespace(new_int_mio="new_integer"),
        dinfitem=SimpleNamespace(mio_obj_int="objective", mio_obj_bound="bound",
                                mio_obj_rel_gap="gap"),
        iinfitem=SimpleNamespace(mio_num_branch="nodes", mio_num_int_solutions="solutions",
                                mio_initial_feasible_solution="start_received",
                                mio_obj_bound_defined="bound_defined"),
    )
    monkeypatch.setitem(sys.modules, "mosek", api)
    monkeypatch.setattr(cp, "__version__", "1.9.2")
    monkeypatch.setattr(cp, "installed_solvers", lambda: ["MOSEK"])
    monkeypatch.setattr(exact, "perf_counter", lambda: state.clock)
    monkeypatch.setattr(MOSEK, "handle_options",
                        staticmethod(lambda task, verbose, options: options))
    monkeypatch.setattr(MOSEK, "_build_slack_task", staticmethod(lambda task, data: None))
    state.problem, state.model, state.api = p, model, api
    state.chain, state.inverse, state.vector, state.data = chain, inverse, vector, data
    return state


@pytest.mark.parametrize("receipt", [True, False])
def test_native_start_receipt_and_callback_mapping_without_licence(native, monkeypatch, receipt):
    """Submission is not receipt, and native bounds never become valid certificates."""
    native.receipt = receipt
    monkeypatch.setattr(exact, "build_model", lambda *args: native.model)
    result = wrapper_exact_execution(native.problem, "CORRIDOR", threads=2, time_limit_s=10)
    start = result.diagnostics["initial_start"]
    assert start["supplied"] is True and start["received"] is receipt
    task, lift = native.tasks
    assert lift.converted == ([4, 5, 6], ["continuous"] * 3)
    assert set(lift.fixed) == set(range(6))
    assert np.array_equal(task.supplied[:4], np.zeros(4))
    assert task.timers == {"wall": 10., "mip": 10.}
    assert audit_execution(native.problem, result.weights)["passed"]
    sources = {row["source"] for row in result.diagnostics["incumbent_events"]}
    assert {"native_callback", "native_return"} <= sources
    assert result.diagnostics["native_bound"] == pytest.approx(.0123 + 12. / SCALE)
    assert result.diagnostics["bound"] is None
    weights, objective = exact.invert_native(
        native.model, native.chain, native.inverse, native.vector, 23.)
    assert np.allclose(weights, native.problem.base + np.array([.005, 0., 0., 0., -.005]))
    assert objective == 40. / SCALE


def test_native_callback_failure_is_reported_and_final_return_is_audited(native, monkeypatch):
    """A broken callback map cannot bypass the audit or discard the common incumbent."""
    native.bad_callback = True
    monkeypatch.setattr(exact, "build_model", lambda *args: native.model)
    result = wrapper_exact_execution(native.problem, "FULL")
    assert native.callback_return == 1
    assert "deliberately invalid" in result.diagnostics["callback_errors"][0]
    assert audit_execution(native.problem, result.weights, "FULL")["passed"]


def test_native_version_and_charged_deadline_guards_without_licence(native, monkeypatch):
    """Unsupported canonicalisation and spent start budgets stop before native solving."""
    monkeypatch.setattr(exact, "build_model", lambda *args: native.model)
    monkeypatch.setattr(cp, "__version__", "unsupported")
    result = wrapper_exact_execution(native.problem, "CORRIDOR")
    assert result.diagnostics["native_status"] == "solver_error"
    monkeypatch.setattr(cp, "__version__", "1.9.2")
    monkeypatch.setattr(native.api.Env, "getversion", lambda: (12, 0, 0))
    result = wrapper_exact_execution(native.problem, "CORRIDOR")
    assert result.diagnostics["native_status"] == "solver_error"
    monkeypatch.setattr(native.api.Env, "getversion", lambda: (11, 2, 3))
    assert exact.lift_start(None, None, None, None, -1)["supplied"] is False
    record = exact.native_mosek(None, native.model, None, -1, 1, .001, lambda *args: None)
    assert record["native_status"] == "preparation_deadline" and not native.tasks
    native.lift_exhausts_budget = True
    result = wrapper_exact_execution(native.problem, "CORRIDOR", time_limit_s=10)
    assert result.diagnostics["native_status"] == "preparation_deadline"
    assert np.array_equal(result.weights, native.problem.base)


@pytest.mark.parametrize("formulation", ["FULL", "CORRIDOR"])
def test_real_initial_state_assignment_preserves_linear_rows_and_objective(formulation):
    """The real canonical start assignments satisfy the original model before a lift."""
    problem = example()
    model = build_model(problem, formulation)
    values = model.initial_values(problem.base)
    variables = {variable.id: variable for variable in model.optimisation.variables()}
    for key, value in values.items():
        variables[key].value = value
    assert max(float(np.max(row.violation(), initial=0)) for row in model.linear_constraints) < 1e-9
    active = problem.base - problem.target
    assert model.optimisation.objective.value / SCALE == pytest.approx(
        active @ problem.covariance @ active)


def test_fixed_portfolio_dispatch_does_not_claim_native_start_receipt(native, monkeypatch):
    """A continuous fixed-state MOSEK path records its budget and supplied-only semantics."""
    p = native.problem
    fixed = replace(p, lower=p.base, upper=p.base, full_lower=p.base, full_upper=p.base)
    calls = []

    def solve_constant(self, **kwargs):
        """Provide a constant-objective native return without a licence."""
        calls.append(kwargs)
        self._status = cp.OPTIMAL

    monkeypatch.setattr(cp.Problem, "solve", solve_constant)
    result = wrapper_exact_execution(fixed, "CORRIDOR", mip_start=p.base, time_limit_s=10)
    assert calls[0]["mosek_params"]["MSK_DPAR_OPTIMIZER_MAX_TIME"] == 10.
    assert result.diagnostics["initial_start"]["supplied"] is False
    assert result.diagnostics["initial_start"]["received"] is None
    assert np.array_equal(result.weights, p.base)
