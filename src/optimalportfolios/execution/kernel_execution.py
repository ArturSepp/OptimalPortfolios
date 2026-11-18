"""Kernel Execution on the supplied, audited execution domain.

Adapted from the Kernel Search framework of Guastaroba and Speranza (2012),
doi:10.1016/j.ejor.2011.09.004, and their general scheme (2014),
doi:10.1016/j.ejor.2014.04.007, Section 2, Algorithm 1. This convex-quadratic
adaptation orders fractional ticket occupancy instead of LP reduced costs and
makes one pass of equal-size buckets. It is not their index-tracking MILP.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from time import perf_counter

import cvxpy as cp
import numpy as np

from .domain import audit_execution, compile_states, objective, result, variance
from .exact_execution import native_mosek
from .model import SCALE, build_model
from .types import ExecutionProblem, ExecutionResult


@dataclass(frozen=True)
class KernelConfig:
    """One-pass search allowance and kernel size, without changing hard rules.

    ``time_limit_s`` includes continuous guidance and model construction.
    The initial kernel has min(candidates, max(minimum_kernel,
    kernel_per_ticket * discretionary_budget)) names. Native subproblems use
    one thread and a relative gap target of 1e-4.
    """

    time_limit_s: float = 60.0
    minimum_kernel: int = 24
    kernel_per_ticket: int = 2

    def __post_init__(self):
        """Reject invalid allowances and nonpositive integer kernel settings."""
        if not np.isfinite(self.time_limit_s) or self.time_limit_s <= 0:
            raise ValueError("Positive finite Kernel Execution allowance required")
        for value in (self.minimum_kernel, self.kernel_per_ticket):
            if (isinstance(value, (bool, np.bool_))
                    or not isinstance(value, (int, np.integer)) or value <= 0):
                raise ValueError("Kernel sizes must be positive integers")


def _restricted_problem(problem, allowed):
    """Pin excluded optional names without changing risk, target or hard rows."""
    lower, upper = problem.lower.copy(), problem.upper.copy()
    for i in range(len(lower)):
        if i != problem.cash and not problem.mandatory[i] and i not in allowed:
            lower[i] = upper[i] = problem.base[i]
    return replace(problem, lower=lower, upper=upper)


def _search_partition(candidates, occupancies, budget, names, config):
    """Order fractional occupancy with stable name ties and disjoint buckets."""
    order = sorted(candidates, key=lambda i: (-occupancies.get(i, 0.0), names[i]))
    size = min(len(order), max(config.minimum_kernel, config.kernel_per_ticket * budget))
    buckets = [order[i:i + size] for i in range(size, len(order), size)] if size else []
    return set(order[:size]), buckets


def wrapper_kernel_execution(
    problem: ExecutionProblem, config: KernelConfig = KernelConfig(),
) -> ExecutionResult:
    """Search one pass of restricted mixed-integer problems, auditing every delivery.

    Requires the optional MOSEK backend and its licence, as Exact Execution does.
    The caller supplies a prepared CORRIDOR problem. Clarabel guidance is charged
    to the allowance and failure falls back to name order. Every native candidate
    must pass both the original and restricted audits. Incumbents and stage
    cutoffs use the full scored objective F, including proportional trade costs.
    Bounds describe restricted subproblems only, not a global certificate.
    A backend failure is recorded and preserves an audited base, if available.
    """
    started = perf_counter()
    deadline = started + config.time_limit_s
    best = problem.base.copy() if audit_execution(problem, problem.base)["passed"] else None
    value = objective(problem, best) if best is not None else np.inf
    diagnostics = dict(
        method="Kernel Execution", bounds_scope="restricted_subproblems_only",
        configuration=dict(
            kernel_size="min(candidates,max(minimum_kernel,kernel_per_ticket*B))",
            minimum_kernel=config.minimum_kernel, kernel_per_ticket=config.kernel_per_ticket,
            bucket_size="initial_kernel_size", passes=1,
            selection="fractional_ticket_occupancy_then_name",
        ),
        guidance={}, subproblems=[], incumbent_events=[],
        first_audited_feasible_seconds=0.0 if best is not None else None,
    )
    states = compile_states(problem)
    candidates = [
        i for i, row in enumerate(states)
        if i != problem.cash and not problem.mandatory[i] and any(s.ticket for s in row)
    ]
    budget = problem.rules.discretionary_allowance
    if budget is None:
        budget = (len(candidates) if problem.rules.ticket_cap is None
                  else max(0, problem.rules.ticket_cap - problem.mandatory_count))
    occupancies = {}
    guidance_start = perf_counter()
    try:
        relaxation = build_model(problem, perspective=True, continuous=True)
        relaxation.optimisation.solve(
            solver="CLARABEL", time_limit=max(.001, deadline - perf_counter()),
            tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10, max_iter=200,
        )
        diagnostics["guidance"]["status"] = relaxation.optimisation.status
        if relaxation.optimisation.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            occupancies = {
                i: float(relaxation.occupancy[i].value) for i in candidates
                if relaxation.occupancy[i].value is not None
            }
    except (ValueError, RuntimeError, cp.SolverError) as error:
        diagnostics["guidance"]["error"] = repr(error)
    diagnostics["guidance"]["seconds"] = perf_counter() - guidance_start
    kernel, buckets = _search_partition(candidates, occupancies, budget, problem.names, config)
    for stage, bucket in enumerate([[]] + buckets):
        if perf_counter() >= deadline:
            break
        available = kernel | set(bucket)
        restricted = _restricted_problem(problem, available)
        model_start = perf_counter()
        model = build_model(restricted)
        rows = list(model.optimisation.constraints)
        if bucket:
            rows.append(cp.sum(cp.hstack([model.occupancy[i] for i in bucket])) >= 1)
        if np.isfinite(value):
            rows.append(model.optimisation.objective.expr <= SCALE * (value + 1e-12))
        model.optimisation = cp.Problem(model.optimisation.objective, rows)
        record = dict(stage=stage, kernel=sorted(kernel), bucket=bucket,
                      construction_seconds=perf_counter() - model_start)
        stage_best, stage_value = None, np.inf

        def retain(weights, source):
            """Check both domains and bucket participation before comparing F."""
            nonlocal best, value, stage_best, stage_value
            audit = audit_execution(problem, weights)
            local = audit_execution(restricted, weights)
            bucket_used = not bucket or any(
                abs(weights[i] - problem.current[i]) > problem.rules.zero_tolerance
                for i in bucket
            )
            event = dict(source=source, seconds=perf_counter() - started, audit=audit,
                         restricted_audit=local, bucket_used=bucket_used)
            invalid = audit.get("violations") == ["invalid_weights"]
            candidate = None if invalid else objective(problem, weights)
            event["scored_objective"] = candidate
            event["variance"] = None if invalid else variance(problem, weights)
            if audit["passed"] and local["passed"] and bucket_used:
                if candidate < stage_value:
                    stage_best, stage_value = np.asarray(weights).copy(), candidate
                if candidate < value:
                    best, value = np.asarray(weights).copy(), candidate
                    event["improved"] = True
                if diagnostics["first_audited_feasible_seconds"] is None:
                    diagnostics["first_audited_feasible_seconds"] = perf_counter() - started
            diagnostics["incumbent_events"].append(event)

        remaining = deadline - perf_counter()
        stages_left = len(buckets) + 1 - stage
        sub_deadline = min(deadline, perf_counter() + max(.001, remaining / stages_left))
        initial = (best if not bucket and best is not None
                   and audit_execution(restricted, best)["passed"] else None)
        try:
            if model.selectors:
                record["native"] = native_mosek(
                    restricted, model, initial, sub_deadline, 1, 1e-4, retain,
                )
            else:
                model.optimisation.solve(
                    solver="CLARABEL", time_limit=max(.001, sub_deadline - perf_counter()),
                    tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10,
                )
                record["native"] = {"native_status": model.optimisation.status}
                if model.weights.value is not None:
                    retain(np.asarray(model.weights.value), "continuous_return")
        except (ValueError, RuntimeError, cp.SolverError) as error:
            record["error"] = repr(error)
        if stage_best is not None:
            selected = {
                i for i in candidates
                if abs(stage_best[i] - problem.current[i]) > problem.rules.zero_tolerance
            }
            # One pass retains the original kernel and the selected bucket names.
            # Ageing or removal would change the evaluated research algorithm.
            kernel |= selected
        record["seconds"] = perf_counter() - model_start
        diagnostics["subproblems"].append(record)
    diagnostics.update(
        seconds=perf_counter() - started,
        completed_buckets=max(0, len(diagnostics["subproblems"]) - 1),
        total_buckets=len(buckets),
    )
    return result(problem, best, "audited_incumbent" if best is not None else "abstention",
                  diagnostics)
