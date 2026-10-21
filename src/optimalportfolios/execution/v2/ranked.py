"""Cash-funded marginal ranking with audited joint sizing and bounded search."""

from __future__ import annotations

from itertools import islice, product
from dataclasses import asdict
from time import perf_counter
import cvxpy as cp
import numpy as np

from .domain import audit_execution, compile_states, result, variance
from .model import build_model, solve_qp
from .types import ExecutionProblem, ExecutionResult, RankedConfig


def cash_geometry(problem):
    """Compute funded curvature without allocating synthetic dense covariance."""
    cash = problem.cash
    if problem.covariance is not None:
        cov = problem.covariance
        return np.diag(cov) - 2 * cov[:, cash] + cov[cash, cash]
    b = problem.factor_loadings - problem.factor_loadings[cash]
    diagonal = np.einsum("ij,ij->i", b @ problem.factor_covariance, b)
    diagonal += problem.residual_variance + problem.residual_variance[cash]
    diagonal[cash] = 0.0
    return diagonal


def state_gain(gradient, curvature, lower, upper):
    """Minimise the one-dimensional funded quadratic over its actual interval."""
    if curvature > 0:
        step = float(np.clip(-gradient / curvature, lower, upper))
    else:
        step = float(lower if gradient >= 0 else upper)
    return -(2 * gradient * step + curvature * step * step), step


def ranked_selection(problem, states, limit):
    """Greedily rank unique instruments with exact incremental covariance updates."""
    started = perf_counter()
    work = problem.base.copy()
    sigma_active = problem.matvec(work - problem.target)
    curvature = cash_geometry(problem)
    remaining = {
        i
        for i, row in enumerate(states)
        if i != problem.cash and not problem.mandatory[i] and any(s.ticket for s in row)
    }
    selected, trace = {}, []
    while remaining and len(selected) < limit:
        options = []
        for i in remaining:
            g = sigma_active[i] - sigma_active[problem.cash]
            for j, state in enumerate(states[i]):
                if state.ticket:
                    gain, step = state_gain(
                        g, curvature[i], state.lower - work[i], state.upper - work[i]
                    )
                    options.append((-gain, problem.names[i], j, i, step))
        negative_gain, _, j, i, step = min(options)
        if negative_gain >= 0:
            break
        selected[i] = j
        remaining.remove(i)
        work[i] += step
        work[problem.cash] -= step
        if problem.covariance is not None:
            sigma_active += step * (problem.covariance[:, i] - problem.covariance[:, problem.cash])
        else:
            b = problem.factor_loadings
            sigma_active += step * (b @ (problem.factor_covariance @ (b[i] - b[problem.cash])))
            sigma_active[i] += step * problem.residual_variance[i]
            sigma_active[problem.cash] -= step * problem.residual_variance[problem.cash]
        trace.append(dict(instrument=problem.names[i], state=j, gain=-negative_gain, step=step))
    return selected, work, trace, perf_counter() - started


def wrapper_ranked_execution(
    problem: ExecutionProblem, config: RankedConfig = RankedConfig()
) -> ExecutionResult:
    """Return the best audited incumbent after bounded ranking, sizing and search.

    No search failure certifies infeasibility. The supplied prepared baseline is
    retained when feasible, and no trial can widen a corridor or create an exemption.
    """
    started = perf_counter()
    states = compile_states(problem)
    base_audit = audit_execution(problem, problem.base)
    best = problem.base.copy() if base_audit["passed"] else None
    best_value = variance(problem, best) if best is not None else np.inf
    diagnostics = dict(
        config=asdict(config),
        qp_counts=dict(sizing=0, rescue=0, drop=0, exchange=0, state=0),
        trials=[],
        gain_trace=[],
        mandatory_state_truncated=False,
        times=dict(selection=0.0, sizing=0.0, rescue=0.0, search=0.0, bound=0.0),
    )
    cap = problem.rules.ticket_cap
    limit = len(problem.names) if cap is None else max(0, cap - problem.mandatory_count)
    if problem.rules.discretionary_allowance is not None:
        limit = min(limit, problem.rules.discretionary_allowance)
    if any(not row for row in states) or (cap is not None and cap < problem.mandatory_count):
        return result(problem, best, "domain_conflict", diagnostics)
    mandatory = [i for i in range(len(states)) if problem.mandatory[i]]
    choices = [
        sorted(
            range(len(states[i])),
            key=lambda j: (
                max(states[i][j].lower - problem.base[i], problem.base[i] - states[i][j].upper, 0),
                j,
            ),
        )
        for i in mandatory
    ]
    total_branches = int(np.prod([len(x) for x in choices], dtype=object))
    mandatory_branches = list(islice(product(*choices), config.mandatory_state_trials))
    diagnostics["mandatory_state_truncated"] = total_branches > len(mandatory_branches)
    visited = set()
    best_support = {}

    def size(support, phase):
        """Enumerate a bounded mandatory state set and retain independently audited improvements."""
        nonlocal best, best_value, best_support
        if len(support) > limit:
            return False
        key = tuple(sorted(support.items()))
        if key in visited:
            return False
        visited.add(key)
        accepted = False
        for branch in mandatory_branches:
            fixed = dict(zip(mandatory, branch))
            fixed.update(support)
            valid = True
            for i, row in enumerate(states):
                if i == problem.cash or i in fixed:
                    continue
                zero = [j for j, state in enumerate(row) if not state.ticket]
                if not zero:
                    valid = False
                    break
                fixed[i] = zero[0]
            if not valid:
                continue
            tick = perf_counter()
            diagnostics["qp_counts"][phase] += 1
            try:
                model = build_model(problem, fixed_states=fixed)
                weights = solve_qp(model, config.qp_solver)
                audit = None if weights is None else audit_execution(problem, weights)
                native = model.optimisation.status
            except (cp.SolverError, ValueError) as error:
                native, weights, audit = str(error), None, None
            bucket = "search" if phase in ("drop", "exchange", "state") else phase
            diagnostics["times"][bucket] += perf_counter() - tick
            diagnostics["trials"].append(
                dict(phase=phase, support=key, native_status=native, audit=audit)
            )
            if audit is not None and audit["passed"]:
                accepted = True
                objective = variance(problem, weights)
                if objective < best_value - config.objective_tolerance:
                    best, best_value, best_support = weights.copy(), objective, dict(support)
        return accepted

    selected, working, trace, elapsed = ranked_selection(problem, states, limit)
    diagnostics["gain_trace"] = trace
    diagnostics["times"]["selection"] = elapsed
    sized = size(selected, "sizing")
    if not sized:
        # Rank rescue admissions by reduction of the current coupled violations,
        # then by identifier. Every candidate remains inside its compiled state.
        group_value = problem.groups @ working
        pressure = np.maximum(problem.group_lower - group_value, 0) - np.maximum(
            group_value - problem.group_upper, 0
        )
        directions = pressure @ problem.groups if len(pressure) else np.zeros(len(working))
        candidates = []
        for i, row in enumerate(states):
            if i == problem.cash or problem.mandatory[i]:
                continue
            for j, state in enumerate(row):
                if state.ticket:
                    movement = (state.lower + state.upper) / 2 - working[i]
                    benefit = (directions[i] - directions[problem.cash]) * movement
                    candidates.append((-benefit, problem.names[i], i, j))
        trials = []
        for i in sorted(selected, key=lambda x: problem.names[x]):
            trials.append({k: v for k, v in selected.items() if k != i})
        for _, _, i, j in sorted(candidates):
            if i in selected:
                proposal = dict(selected)
                proposal[i] = j
                trials.append(proposal)
            elif len(selected) < limit:
                trials.append(selected | {i: j})
            else:
                for old in sorted(selected, key=lambda x: problem.names[x]):
                    trials.append({k: v for k, v in selected.items() if k != old} | {i: j})
        for support in trials[: config.rescue_trials]:
            size(support, "rescue")
    initial = sorted(best_support, key=lambda i: problem.names[i])
    if config.drop:
        for old in initial:
            size({i: j for i, j in best_support.items() if i != old}, "drop")
    if config.exchange:
        for old in initial:
            support = {i: j for i, j in best_support.items() if i != old}
            point = best if best is not None else problem.base
            active = problem.matvec(point - problem.target)
            curvature = cash_geometry(problem)
            candidates = []
            for i, row in enumerate(states):
                if i == problem.cash or problem.mandatory[i] or i in support:
                    continue
                for j, state in enumerate(row):
                    if state.ticket and (i, j) != (old, best_support.get(old)):
                        gain, _ = state_gain(
                            active[i] - active[problem.cash],
                            curvature[i],
                            state.lower - point[i],
                            state.upper - point[i],
                        )
                        candidates.append((-gain, problem.names[i], i, j))
            if candidates:
                _, _, i, j = min(candidates)
                size(support | {i: j}, "state" if i == old else "exchange")
    if config.compute_bound:
        from .bounds import compute_perspective_lower_bound

        tick = perf_counter()
        diagnostics["bounds"] = compute_perspective_lower_bound(problem)
        diagnostics["times"]["bound"] = perf_counter() - tick
    diagnostics["times"]["total"] = perf_counter() - started
    diagnostics["support"] = {problem.names[i]: j for i, j in best_support.items()}
    diagnostics["visited_supports"] = len(visited)
    status = "audited_incumbent" if best is not None else "bounded_abstention"
    return result(problem, best, status, diagnostics)
