"""Matched conic integer formulations with verified native MOSEK start receipt."""

from __future__ import annotations

from time import perf_counter
import cvxpy as cp
import numpy as np

from .domain import audit_execution, result, variance
from .model import SCALE, build_model
from .types import ExecutionProblem, ExecutionResult


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
    record = dict(
        native_status="not_run",
        initial_start=dict(requested=initial is not None, supplied=False, received=None),
        canonicalisation_seconds=perf_counter() - tick,
        native_bound=None,
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
                    "MSK_DPAR_MIO_TOL_FEAS": 0.0001,
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
        if initial is not None:
            record["initial_start"] = lift_start(task, data, model, initial, deadline)
        callback_errors = []

        def callback(code, doubles, integers, longs):
            """Read primal vectors only at the documented integer-incumbent event."""
            try:
                if code == mosek.callbackcode.new_int_mio:
                    x = task.getxx(mosek.soltype.itg)
                    w, _ = invert_native(
                        model, chain, inverse, x, doubles[mosek.dinfitem.mio_obj_int]
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
            record["native_bound_canonical"] = task.getdouinf(mosek.dinfitem.mio_obj_bound) / SCALE
            record["native_bound"] = record["native_bound_canonical"] + model.constant
            record["native_gap_relative"] = task.getdouinf(mosek.dinfitem.mio_obj_rel_gap)
        solution = chain.invert({"task": task, "solver_options": options}, inverse)
        record["canonical_status"] = solution.status
        if solution.primal_vars and model.displacement_bp.id in solution.primal_vars:
            w = (
                model.reference
                + model.direction @ solution.primal_vars[model.displacement_bp.id] / SCALE
            )
            retain(w, "native_return")
    return record


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

    The time limit includes graph construction and start lifting, with unavoidable
    in-progress canonicalisation recorded. Native bound certification is separate
    from incumbent delivery. A failed solve never becomes an infeasibility claim.
    """
    if not np.isfinite(time_limit_s) or time_limit_s <= 0 or threads < 1 or gap_rel < 0:
        raise ValueError("Invalid Exact budget or solver controls")
    started = perf_counter()
    deadline = started + time_limit_s
    best, best_value = None, np.inf
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
    )

    def retain(weights, source):
        """Retain only audited incumbents, including externally supplied witnesses."""
        nonlocal best, best_value
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
    diagnostics["total_seconds"] = perf_counter() - started
    return result(
        problem,
        best,
        "audited_incumbent" if best is not None else "no_audited_incumbent",
        diagnostics,
        formulation,
    )
