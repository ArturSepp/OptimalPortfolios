"""Matched conic integer formulations with verified native MOSEK start receipt."""

from __future__ import annotations

from copy import copy
from time import perf_counter
import cvxpy as cp
import numpy as np

from .domain import audit_execution, compile_states, result, variance
from .model import SCALE, build_model
from .types import ExecutionProblem, ExecutionResult, RankedConfig

_NATIVE_FEASIBILITY_TOLERANCE = 1e-9


def invert_native(model, chain, inverse, native_x, native_objective):
    """Invert the primal canonical map and objective offset using the solver chain."""
    from cvxpy.reductions.solution import Solution
    from cvxpy.reductions.cone2cone.affine2direct import FREE

    length = inverse[-1]["K_dir"][FREE]
    raw = Solution(
        cp.OPTIMAL, float(native_objective), {FREE: np.asarray(native_x[:length])}, None, {}
    )
    solved = chain.invert({"sol": raw}, inverse)
    displacement = solved.primal_vars[model.displacement_bp.id]
    return model.reference + model.direction @ displacement / SCALE, solved.opt_val / SCALE


def lift_start(task, data, model, weights, deadline):
    """Solve a continuous fixed-portfolio lift before supplying every native variable."""
    import mosek

    tick = perf_counter()
    record = dict(requested=True, supplied=False, received=None, lift_status="not_started")
    if deadline <= tick:
        return record
    with mosek.Task(task) as lift:
        integers = data["bool_vars_idx"] + data["int_vars_idx"]
        lift.putvartypelist(integers, [mosek.variabletype.type_cont] * len(integers))
        offsets = data["param_prob"].var_id_to_col
        initial = model.initial_values(weights)
        record["variable_map"] = {str(key): int(offsets[key]) for key in initial}
        for identity, values in initial.items():
            for position, value in enumerate(values):
                lift.putvarbound(
                    offsets[identity] + position, mosek.boundkey.fx, float(value), float(value)
                )
        lift.putdouparam(mosek.dparam.optimizer_max_time, max(0.001, deadline - perf_counter()))
        lift.optimize()
        status = lift.getsolsta(mosek.soltype.itr)
        record["lift_status"] = str(status)
        if status in (mosek.solsta.optimal, mosek.solsta.prim_feas):
            values = lift.getxx(mosek.soltype.itr)
            task.putxx(mosek.soltype.itg, values)
            record.update(supplied=True, variables=len(values))
    record["seconds"] = perf_counter() - tick
    return record


def native_mosek(problem, model, initial, deadline, threads, gap_rel, retain):
    """Use the explicitly qualified CVXPY canonical interface without upgrading it."""
    import mosek
    from cvxpy.reductions.solvers.conic_solvers.mosek_conif import MOSEK

    if cp.__version__ != "1.9.2":
        raise RuntimeError("Native start adapter requires qualified CVXPY 1.9.2")
    if mosek.Env.getversion()[:2] != (11, 2):
        raise RuntimeError("Native start adapter requires the MOSEK 11.2 API")
    tick = perf_counter()
    data, chain, inverse = model.optimisation.get_problem_data("MOSEK")
    # CVXPY omits this constant from the native task and adds it on inversion.
    # Restore it before solving so the relative stopping gap uses the original
    # variance. The copied inverse map must then avoid adding it a second time.
    offset = float(inverse[-1][cp.settings.OBJ_OFFSET])
    native_last = copy(inverse[-1])
    native_last.inverse_data = {**inverse[-1].inverse_data, cp.settings.OBJ_OFFSET: 0.}
    native_inverse = [*inverse[:-1], native_last]
    record = dict(
        native_status="not_run",
        initial_start=dict(requested=initial is not None, supplied=False, received=None),
        canonicalisation_seconds=perf_counter() - tick,
        native_bound=None,
        native_objective_offset=offset / SCALE,
        native_objective_origin="original_variance",
        native_mip_feasibility_tolerance=_NATIVE_FEASIBILITY_TOLERANCE,
        bound=None,
        bound_reason="Native global bound retained as diagnostic, certificate not validated",
    )
    if deadline <= perf_counter():
        record["native_status"] = "preparation_deadline"
        return record
    with mosek.Task() as task:
        options = MOSEK.handle_options(
            task,
            False,
            dict(
                accept_unknown=True,
                mosek_params={
                    "MSK_IPAR_NUM_THREADS": threads,
                    "MSK_IPAR_MIO_SEED": 0,
                    # The old 1e-4 permits false optimal termination on the
                    # public overlapping-group case. Tighten native numerics;
                    # the mathematical graph and original-unit audit stay fixed.
                    "MSK_DPAR_MIO_TOL_FEAS": _NATIVE_FEASIBILITY_TOLERANCE,
                    "MSK_DPAR_MIO_TOL_ABS_RELAX_INT": 1e-9,
                    "MSK_DPAR_INTPNT_CO_TOL_PFEAS": 1e-10,
                    "MSK_DPAR_INTPNT_CO_TOL_DFEAS": 1e-10,
                    "MSK_DPAR_INTPNT_CO_TOL_REL_GAP": 1e-10,
                    "MSK_DPAR_MIO_TOL_REL_GAP": gap_rel,
                    "MSK_DPAR_MIO_TOL_ABS_GAP": 1e-8,
                },
            ),
        )
        MOSEK._build_slack_task(task, data)
        task.putcfix(offset)
        if initial is not None:
            record["initial_start"] = lift_start(task, data, model, initial, deadline)
        callback_errors = []

        def callback(code, doubles, integers, longs):
            """Read primal vectors only at the documented integer-incumbent event."""
            try:
                if code == mosek.callbackcode.new_int_mio:
                    x = task.getxx(mosek.soltype.itg)
                    w, _ = invert_native(
                        model, chain, native_inverse, x, doubles[mosek.dinfitem.mio_obj_int]
                    )
                    retain(w, "native_callback")
            except Exception as error:
                callback_errors.append(repr(error))
                return 1
            return int(perf_counter() >= deadline)

        task.set_InfoCallback(callback)
        if deadline <= perf_counter():
            record["native_status"] = "preparation_deadline"
            return record
        remaining = max(0.001, deadline - perf_counter())
        task.putdouparam(mosek.dparam.optimizer_max_time, remaining)
        task.putdouparam(mosek.dparam.mio_max_time, remaining)
        termination = task.optimize()
        record.update(
            native_status=str(task.getsolsta(mosek.soltype.itg)),
            termination=str(termination),
            native_problem_status=str(task.getprosta(mosek.soltype.itg)),
            nodes=task.getintinf(mosek.iinfitem.mio_num_branch),
            integer_solutions=task.getintinf(mosek.iinfitem.mio_num_int_solutions),
            callback_errors=callback_errors,
        )
        record["initial_start"]["received"] = bool(
            task.getintinf(mosek.iinfitem.mio_initial_feasible_solution)
        )
        if task.getintinf(mosek.iinfitem.mio_obj_bound_defined):
            bound = task.getdouinf(mosek.dinfitem.mio_obj_bound)
            record["native_bound_canonical"] = (bound - offset) / SCALE
            record["native_bound"] = bound / SCALE
            record["native_incumbent"] = task.getdouinf(mosek.dinfitem.mio_obj_int) / SCALE
            record["native_gap_relative"] = task.getdouinf(mosek.dinfitem.mio_obj_rel_gap)
        solution = chain.invert({"task": task, "solver_options": options}, native_inverse)
        record["canonical_status"] = solution.status
        if solution.primal_vars and model.displacement_bp.id in solution.primal_vars:
            w = (
                model.reference
                + model.direction @ solution.primal_vars[model.displacement_bp.id] / SCALE
            )
            retain(w, "native_return")
    return record


def _refine_group_candidate(problem, weights, formulation, deadline):
    """Re-solve one rejected candidate's nearest states inside the original budget.

    The native vector remains rejected. This fresh fixed-state QP uses the same
    objective, group envelope and all other constraints, with the existing sizing
    tolerances. It neither rounds weights nor changes the independent audit.
    """
    record = dict(attempted=False, status="deadline", input_weights=weights.tolist())
    if perf_counter() >= deadline:
        return None, record
    try:
        states = compile_states(problem, formulation)
        fixed = {}
        for i, row in enumerate(states):
            if i == problem.cash:
                continue
            errors = [max(s.lower - weights[i], weights[i] - s.upper, 0.) for s in row]
            if not errors or min(errors) > problem.rules.audit_tolerance:
                record["status"] = "no_matching_state"
                return None, record
            fixed[i] = int(np.argmin(errors))
        record["fixed_states"] = sorted(fixed.items())
        model = build_model(problem, formulation, fixed_states=fixed)
        remaining = deadline - perf_counter()
        if remaining <= 0:
            return None, record
        record.update(attempted=True, native_time_limit_s=remaining)
        # solve_qp has no end-to-end deadline: this single refinement deliberately
        # uses its same CLARABEL tolerances with the remaining Exact time limit.
        model.optimisation.solve(
            solver="CLARABEL", tol_gap_abs=1e-10, tol_gap_rel=1e-10,
            tol_feas=1e-10, time_limit=remaining,
        )
        record["status"] = model.optimisation.status
        if model.weights.value is None:
            return None, record
        refined = np.asarray(model.weights.value).reshape(-1)
        record["max_weight_change"] = float(np.max(np.abs(refined - weights)))
        return refined, record
    except Exception as error:
        record.update(status="solver_error", error=repr(error))
        return None, record


def wrapper_exact_execution(
    problem: ExecutionProblem,
    formulation: str,
    perspective: bool = False,
    mip_start: np.ndarray | None = None,
    time_limit_s: float = 600,
    threads: int = 4,
    solver: str = "MOSEK",
    gap_rel: float = 1e-4,
) -> ExecutionResult:
    """Solve FULL or CORRIDOR and externally retain the same audited common incumbent.

    The time limit includes graph construction, start lifting and at most one
    fixed-state refinement of a group-rejected native candidate. Refinement is
    separately audited and attributed; raw native statuses are preserved. Native
    bound certification is separate from delivery. In-progress work can overrun.
    """
    if not np.isfinite(time_limit_s) or time_limit_s <= 0 or threads < 1 or gap_rel < 0:
        raise ValueError("Invalid Exact budget or solver controls")
    started = perf_counter()
    deadline = started + time_limit_s
    best, best_value = None, np.inf
    best_source, rejected_group = None, None
    diagnostics = dict(
        formulation=formulation,
        perspective=perspective,
        solver=solver,
        threads=threads,
        seed=0,
        time_limit_s=time_limit_s,
        gap_rel=gap_rel,
        first_audited_feasible_seconds=None,
        first_native_audited_feasible_seconds=None,
        incumbent_events=[],
        bound=None,
        initial_start=dict(requested=False, supplied=False, received=None),
        boundary_refinement=dict(requested=False, attempted=False, status="not_needed"),
    )

    def retain(weights, source):
        """Retain only audited incumbents, including externally supplied witnesses."""
        nonlocal best, best_value, best_source, rejected_group
        audit = audit_execution(problem, weights, formulation)
        diagnostics["incumbent_events"].append(
            dict(source=source, audit=audit, seconds=perf_counter() - started)
        )
        if audit["passed"]:
            if (
                source.startswith("native")
                and diagnostics["first_native_audited_feasible_seconds"] is None
            ):
                diagnostics["first_native_audited_feasible_seconds"] = perf_counter() - started
            if diagnostics["first_audited_feasible_seconds"] is None:
                diagnostics["first_audited_feasible_seconds"] = perf_counter() - started
            value = variance(problem, weights)
            if value < best_value:
                best, best_value = np.asarray(weights).copy(), value
                best_source = source
        elif source.startswith("native") and audit["violations"] == ["groups"]:
            value = variance(problem, weights)
            if rejected_group is None or value < rejected_group[0]:
                rejected_group = (value, np.asarray(weights).copy(), source, audit)

    retain(problem.base, "common_prepared_incumbent")
    if mip_start is not None:
        if not audit_execution(problem, mip_start, formulation)["passed"]:
            raise ValueError("Requested MIP start fails independent audit")
        retain(mip_start, "requested_mip_start")
    initial = best if mip_start is None else np.asarray(mip_start).copy()
    diagnostics["initial_start"]["requested"] = initial is not None
    if solver not in cp.installed_solvers():
        diagnostics["native_status"] = "solver_unavailable"
        diagnostics["initial_start"] = dict(
            requested=initial is not None, supplied=False, received=None
        )
    else:
        try:
            model = build_model(problem, formulation, perspective)
            if solver == "MOSEK" and model.selectors:
                diagnostics.update(
                    native_mosek(problem, model, initial, deadline, threads, gap_rel, retain)
                )
            else:
                kwargs = {}
                if solver == "SCIP":
                    kwargs["scip_params"] = {
                        "limits/time": max(0.001, deadline - perf_counter()),
                        "limits/gap": gap_rel,
                    }
                elif solver == "MOSEK":
                    kwargs["mosek_params"] = {
                        "MSK_DPAR_OPTIMIZER_MAX_TIME": max(0.001, deadline - perf_counter()),
                        "MSK_IPAR_NUM_THREADS": threads,
                    }
                model.optimisation.solve(solver=solver, **kwargs)
                diagnostics["native_status"] = model.optimisation.status
                diagnostics["initial_start"] = dict(
                    requested=initial is not None,
                    supplied=False,
                    received=None,
                    reason="No qualified native start interface for this path",
                )
                if model.weights.value is not None:
                    retain(np.asarray(model.weights.value), "native_return")
        except Exception as error:
            diagnostics.update(native_status="solver_error", error=repr(error))
    if rejected_group is not None:
        value, weights, source, audit = rejected_group
        detail = dict(requested=True, attempted=False, status="incumbent_dominates",
                      source=source, input_variance=value, input_audit=audit)
        if value < best_value:
            tick = perf_counter()
            refined, refinement = _refine_group_candidate(problem, weights, formulation, deadline)
            detail.update(refinement, seconds=perf_counter() - tick)
            if refined is not None:
                retain(refined, "boundary_refinement")
                detail["output_audit"] = diagnostics["incumbent_events"][-1]["audit"]
        diagnostics["boundary_refinement"] = detail
    diagnostics["delivered_source"] = best_source
    native_bound = diagnostics.get("native_bound")
    diagnostics["native_bound_check"] = None
    if native_bound is not None and best is not None:
        # This uses the existing default objective comparison scale, never the
        # portfolio feasibility tolerance. Preserve the raw bound and status;
        # consistency with one witness cannot certify a global lower bound.
        tolerance = RankedConfig().objective_tolerance
        diagnostics["native_bound_check"] = dict(
            consistent_with_delivery=bool(
                np.isfinite(native_bound) and native_bound <= best_value + tolerance
            ),
            objective_tolerance=tolerance,
            excess_over_delivery=float(native_bound - best_value),
        )
    diagnostics["total_seconds"] = perf_counter() - started
    return result(
        problem,
        best,
        "audited_incumbent" if best is not None else "no_audited_incumbent",
        diagnostics,
        formulation,
    )
