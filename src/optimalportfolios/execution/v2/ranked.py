"""Cash-funded marginal ranking with audited joint sizing and bounded search."""

from __future__ import annotations

from itertools import combinations, islice, product
from heapq import nsmallest
from dataclasses import asdict, replace
from time import perf_counter
import cvxpy as cp
import numpy as np

from .domain import audit_execution, compile_states, result, variance
from .model import build_model, solve_qp
from .types import ExecutionProblem, ExecutionResult, RankedConfig
from ._sizing import _FixedStateSizer


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


def _pair_gain(g1, g2, h1, h2, cross, lo1, hi1, lo2, hi2):
    """Minimise a PSD funded quadratic on a box by its edges and stationary point.

    This is a search priority, not a feasible portfolio or a bound certificate:
    cash limits, groups and the remaining movable coordinates are handled by sizing.
    """
    points = []
    for x in (lo1, hi1):
        _, y = state_gain(g2 + cross * x, h2, lo2, hi2)
        points.append((x, y))
    for y in (lo2, hi2):
        _, x = state_gain(g1 + cross * y, h1, lo1, hi1)
        points.append((x, y))
    determinant = h1 * h2 - cross * cross
    if determinant > 0:
        x = (cross * g2 - h2 * g1) / determinant
        y = (cross * g1 - h1 * g2) / determinant
        if lo1 <= x <= hi1 and lo2 <= y <= hi2:
            points.append((x, y))
    return -min(2 * (g1 * x + g2 * y) + h1 * x * x + 2 * cross * x * y
                + h2 * y * y for x, y in points)


def _constrained_pair_gain(problem, point, i, k, gradient, curvature, cross, bounds):
    """Score a funded two-coordinate slice inside the original cash/group envelope.

    This analytic priority is not QIS tracking-error reporting or a new portfolio
    acceptance rule. Polygon clipping retains the declared audit envelope; the
    unchanged full sizing QP can subsequently move all selected coordinates.
    An empty slice falls back to the box score: joint sizing can move the other
    retained coordinates, so a fixed-coordinate slice cannot reject its support.
    """
    lo1, hi1, lo2, hi2 = bounds
    polygon = [(lo1, lo2), (hi1, lo2), (hi1, hi2), (lo1, hi2)]
    cash = problem.cash
    rows = [(1., 1., point[cash] - problem.lower[cash]),
            (-1., -1., problem.upper[cash] - point[cash])]
    exposure = problem.groups @ point
    for row, lower, upper, value in zip(
        problem.groups, problem.group_lower, problem.group_upper, exposure
    ):
        a, b = row[i] - row[cash], row[k] - row[cash]
        rows.extend(((a, b, upper + problem.rules.audit_tolerance - value),
                     (-a, -b, value - lower + problem.rules.audit_tolerance)))
    for a, b, bound in rows:
        clipped = []
        for left, right in zip(polygon, polygon[1:] + polygon[:1]):
            fl = a * left[0] + b * left[1] - bound
            fr = a * right[0] + b * right[1] - bound
            if fl <= 0:
                clipped.append(left)
            if (fl <= 0) != (fr <= 0):
                t = fl / (fl - fr)
                clipped.append((left[0] + t * (right[0] - left[0]),
                                left[1] + t * (right[1] - left[1])))
        polygon = clipped
        if not polygon:
            return _pair_gain(*gradient, *curvature, cross, lo1, hi1, lo2, hi2)
    g1, g2 = gradient
    h1, h2 = curvature
    points = list(polygon)
    for (x, y), (next_x, next_y) in zip(polygon, polygon[1:] + polygon[:1]):
        dx, dy = next_x - x, next_y - y
        slope = (g1 + h1 * x + cross * y) * dx + (g2 + cross * x + h2 * y) * dy
        curve = h1 * dx * dx + 2 * cross * dx * dy + h2 * dy * dy
        _, t = state_gain(slope, curve, 0., 1.)
        points.append((x + t * dx, y + t * dy))
    determinant = h1 * h2 - cross * cross
    if determinant > 0:
        x = (cross * g2 - h2 * g1) / determinant
        y = (cross * g1 - h1 * g2) / determinant
        if (lo1 <= x <= hi1 and lo2 <= y <= hi2
                and all(a * x + b * y <= bound for a, b, bound in rows)):
            points.append((x, y))
    return -min(2 * (g1 * x + g2 * y) + h1 * x * x + 2 * cross * x * y
                + h2 * y * y for x, y in points)


def _coordinated_supports(problem, states, support, point, limit, visited, expanded=True):
    """Yield deterministic priorities for edits and capacity-neutral pair replacements.

    Include single/pair edits and two admissions coupled to one or two removals.
    Removed instruments return to their no-ticket base; the funded quadratic then
    scores the admitted pair conditional on those removals. Full joint sizing,
    including every group and cash row, still decides acceptance.
    Negative scalar or pair gains do not suppress a trial, since joint sizing can
    also move mandatory and already selected instruments.
    """
    active = problem.matvec(point - problem.target)
    gradient = active - active[problem.cash]
    curvature = cash_geometry(problem)
    changes = [(i, j) for i, row in enumerate(states)
               if i != problem.cash and not problem.mandatory[i]
               for j, state in enumerate(row)
               if (state.ticket or i in support) and j != support.get(i)]
    changes.sort(key=lambda x: (problem.names[x[0]], x[1]))

    def proposal(edits):
        """Construct a capacity-valid, previously unvisited support and tie key."""
        candidate = dict(support)
        for i, j in edits:
            if states[i][j].ticket:
                candidate[i] = j
            else:
                candidate.pop(i, None)
        key = tuple(sorted(candidate.items()))
        if len(candidate) > limit or key in visited:
            return None
        tie = tuple(sorted((problem.names[i], j) for i, j in candidate.items()))
        return tie, key

    for i, j in changes:
        candidate = proposal(((i, j),))
        if candidate is not None:
            state = states[i][j]
            gain = _constrained_pair_gain(
                problem, point, i, i, (gradient[i], 0.), (curvature[i], 0.), 0.,
                (state.lower - point[i], state.upper - point[i], 0., 0.),
            ) if expanded else state_gain(gradient[i], curvature[i],
                                           state.lower - point[i], state.upper - point[i])[0]
            yield (-gain, *candidate)
    for (i, j), (k, ell) in combinations(changes, 2):
        if i == k:
            continue
        candidate = proposal(((i, j), (k, ell)))
        if candidate is None:
            continue
        cross = _funded_cross(problem, i, k)
        first, second = states[i][j], states[k][ell]
        gain = _constrained_pair_gain(
            problem, point, i, k, (gradient[i], gradient[k]), (curvature[i], curvature[k]),
            cross, (first.lower - point[i], first.upper - point[i],
                    second.lower - point[k], second.upper - point[k]),
        ) if expanded else _pair_gain(
            gradient[i], gradient[k], curvature[i], curvature[k], cross,
            first.lower - point[i], first.upper - point[i],
            second.lower - point[k], second.upper - point[k],
        )
        yield (-gain, *candidate)

    if not expanded:
        return
    admissions = [(i, j) for i, row in enumerate(states)
                  if i != problem.cash and not problem.mandatory[i] and i not in support
                  for j, state in enumerate(row) if state.ticket]
    removable = [(i, next(j for j, state in enumerate(states[i]) if not state.ticket))
                 for i in support if any(not state.ticket for state in states[i])]
    for count in (1, 2):
        for removals in combinations(removable, count):
            delta = np.zeros(len(point))
            for old, _ in removals:
                delta[old] = problem.base[old] - point[old]
            delta[problem.cash] = -delta.sum()
            shifted = problem.matvec(delta)
            cost = 2 * float(active @ delta) + float(delta @ shifted)
            for first, second in combinations(admissions, 2):
                i, j = first
                k, ell = second
                if i == k:
                    continue
                candidate = proposal((*removals, first, second))
                if candidate is None:
                    continue
                cross = _funded_cross(problem, i, k)
                left, right = states[i][j], states[k][ell]
                gain = _constrained_pair_gain(
                    problem, point + delta, i, k,
                    (gradient[i] + shifted[i] - shifted[problem.cash],
                     gradient[k] + shifted[k] - shifted[problem.cash]),
                    (curvature[i], curvature[k]), cross,
                    (left.lower - point[i], left.upper - point[i],
                     right.lower - point[k], right.upper - point[k]),
                ) - cost
                yield (-gain, *candidate)


def _support_possible(problem, states, support):
    """Reject only supports whose interval envelope cannot satisfy funding or groups.

    Selected coordinates and cash remain free over their original intervals;
    mandatory coordinates use the union envelope of all their states. Independent
    row bounds are necessary, not sufficient. The existing coordinate and row
    audit allowances make this an outer screen, never a new acceptance domain.
    Every surviving support still requires unchanged full joint sizing and audit.
    """
    lower = problem.base.copy()
    upper = lower.copy()
    lower[problem.cash], upper[problem.cash] = (
        problem.lower[problem.cash], problem.upper[problem.cash]
    )
    for i, row in enumerate(states):
        if problem.mandatory[i]:
            lower[i], upper[i] = min(s.lower for s in row), max(s.upper for s in row)
    for i, j in support.items():
        lower[i], upper[i] = states[i][j].lower, states[i][j].upper
    tolerance = problem.rules.audit_tolerance
    lower = np.nextafter(lower - tolerance, -np.inf)
    upper = np.nextafter(upper + tolerance, np.inf)
    if (lower.sum() > problem.investment + tolerance
            or upper.sum() < problem.investment - tolerance):
        return False
    groups = problem.groups
    minimum = np.where(groups >= 0, groups * lower, groups * upper).sum(axis=1)
    maximum = np.where(groups >= 0, groups * upper, groups * lower).sum(axis=1)
    return bool(np.all(minimum <= problem.group_upper + tolerance)
                and np.all(maximum >= problem.group_lower - tolerance))


def _funded_cross(problem, i, k):
    """Return the exact cross curvature of two distinct cash-funded coordinates."""
    cash = problem.cash
    if problem.covariance is not None:
        cov = problem.covariance
        return cov[i, k] - cov[i, cash] - cov[cash, k] + cov[cash, cash]
    loadings = problem.factor_loadings
    return ((loadings[i] - loadings[cash]) @ problem.factor_covariance
            @ (loadings[k] - loadings[cash]) + problem.residual_variance[cash])


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


def _guided_support(problem, states, limit, solver, decision=None):
    """Round an existing perspective relaxation to a support, never to output weights.

    Fractional state occupancy supplies a group-aware alternative to marginal
    ranking when tickets are scarce. The original integer-state sizing and audit
    must accept it before it can replace an incumbent. An optional state decision
    constrains only this continuous guidance graph, never the sizing problem.
    This is not an Exact solve or a separately certified lower bound.
    """
    model = build_model(problem, continuous=True, perspective=True)
    if decision is not None:
        i, j, enabled = decision
        selector = model.selectors[i][j]
        model.optimisation = cp.Problem(
            model.optimisation.objective,
            model.optimisation.constraints + [selector == float(enabled)],
        )
    weights = solve_qp(model, solver)
    if weights is None or not np.isfinite(weights).all():
        return None, str(model.optimisation.status)
    candidates = []
    for i, occupancy in model.occupancy.items():
        if problem.mandatory[i]:
            continue
        tickets = [j for j, state in enumerate(states[i]) if state.ticket
                   and (decision is None or i != decision[0]
                        or (j == decision[1]) == decision[2])]
        if not tickets:
            continue
        if i not in model.selectors:
            candidates.append((-float(occupancy.value), problem.names[i], i, tickets[0]))
            continue
        values = np.asarray(model.selectors[i].value, dtype=float)
        if not np.isfinite(values).all() or not np.isfinite(occupancy.value):
            return None, "nonfinite_occupancy"
        j = min(tickets, key=lambda j: (-values[j], j))
        # A forced admission must survive rounding even if native occupancy is
        # marginally below one. Explicitly excluded states cannot re-enter it.
        priority = 1.0 if decision is not None and i == decision[0] and decision[2] else 0.0
        candidates.append((-priority - float(occupancy.value), problem.names[i], i, j))
    return {i: j for _, _, i, j in sorted(candidates)[:limit]}, str(model.optimisation.status)


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
        qp_counts=dict(sizing=0, rescue=0, drop=0, exchange=0, state=0, coordinated=0,
                       guidance=0, exchange_sweep=0, restart=0, restart_guidance=0,
                       conditional_guidance=0),
        trials=[],
        gain_trace=[],
        mandatory_state_truncated=False,
        coordinated_candidates=0,
        coordinated_search_truncated=False,
        coordinated_passes=[],
        scarce_passes=[],
        coordinated_seed_trials=0,
        guidance_status="not_needed",
        restarts=[],
        conditional_guidance=[],
        conditional_restarts=[],
        completion_passes=[],
        completion_screened=0,
        completion_trial_limit=config.rescue_trials ** 2,
        bridge_restarts=[],
        bridge_completion_passes=[],
        bridge_restart_limit=config.rescue_trials,
        sizing_cache_hits=0,
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
    sizing_cache = {}
    best_support = {}
    sizer = _FixedStateSizer(problem, states, build_model, config.qp_solver)
    diagnostics["sizing_compilation"] = sizer.diagnostics

    def size(support, phase):
        """Enumerate a bounded mandatory state set and retain independently audited improvements."""
        nonlocal best, best_value, best_support
        if len(support) > limit:
            return False
        key = tuple(sorted(support.items()))
        if key in visited:
            return False
        visited.add(key)
        if key in sizing_cache:
            diagnostics["sizing_cache_hits"] += 1
            cached = sizing_cache[key]
            if cached is not None and cached[0] < best_value - config.objective_tolerance:
                best_value, best = cached[0], cached[1].copy()
                best_support = dict(support)
            return cached is not None
        sizing_cache[key] = None
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
            sizing_solver = {}
            try:
                model = sizer(fixed)
                sizing_solver = model.qp_diagnostics
                weights = solve_qp(model, config.qp_solver)
                audit = None if weights is None else audit_execution(problem, weights)
                native = model.optimisation.status
            except (cp.SolverError, ValueError) as error:
                native, weights, audit = str(error), None, None
            bucket = "search" if phase in (
                "drop", "exchange", "state", "coordinated", "exchange_sweep", "restart"
            ) else phase
            diagnostics["times"][bucket] += perf_counter() - tick
            diagnostics["trials"].append(
                dict(phase=phase, support=key, native_status=native, audit=audit,
                     sizing_solver=sizing_solver)
            )
            if audit is not None and audit["passed"]:
                accepted = True
                objective = variance(problem, weights)
                cached = sizing_cache[key]
                if cached is None or objective < cached[0]:
                    sizing_cache[key] = objective, weights.copy()
                if objective < best_value - config.objective_tolerance:
                    best, best_value, best_support = weights.copy(), objective, dict(support)
        return accepted

    selected, working, trace, elapsed = ranked_selection(problem, states, limit)
    diagnostics["gain_trace"] = trace
    diagnostics["times"]["selection"] = elapsed
    # Mandatory corridors can improve risk without any optional ticket. The
    # retained feasible base is not generally their conditional optimum, and
    # dropping from an empty best_support would never test this alternative.
    movable_mandatory = any(
        len(states[i]) > 1 or states[i][0].lower != states[i][0].upper for i in mandatory
    )
    mandatory_sized = size({}, "sizing") if movable_mandatory else False
    sized = mandatory_sized if not selected and mandatory_sized else size(selected, "sizing")
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
    def single_passes():
        """Try the existing drop and exchange passes from the current accepted support."""
        initial = sorted(best_support, key=lambda i: problem.names[i])
        if config.drop:
            for old in initial:
                size({i: j for i, j in best_support.items() if i != old}, "drop")
        if not config.exchange:
            return
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
    single_passes()
    def improve(expanded, remaining_trials, records, screen=False):
        """Run bounded support passes, preserving the earlier incumbent and its search path."""
        for pass_index in range(config.improvement_passes):
            if not remaining_trials:
                break
            tick = perf_counter()
            anchor_value = best_value
            candidate_count = diagnostics["coordinated_candidates"]
            if expanded:
                # Full joint sizing can move every retained coordinate. A
                # two-coordinate score cannot reliably rank a replacement when
                # that recourse is essential, so screen every one-for-one swap.
                anchor = dict(best_support)
                for old in sorted(anchor, key=lambda i: problem.names[i]):
                    for i in sorted(range(len(states)), key=lambda i: problem.names[i]):
                        if i == problem.cash or problem.mandatory[i] or i in anchor:
                            continue
                        for j, state in enumerate(states[i]):
                            if state.ticket:
                                size({k: v for k, v in anchor.items() if k != old} | {i: j},
                                     "exchange_sweep")
                tick = perf_counter()

            def priorities():
                """Count the whole neighbourhood while keeping only a bounded shortlist."""
                for candidate in _coordinated_supports(
                    problem, states, best_support,
                    best if best is not None else problem.base, limit, visited, expanded
                ):
                    if screen and not _support_possible(problem, states, dict(candidate[2])):
                        diagnostics["completion_screened"] += 1
                        continue
                    diagnostics["coordinated_candidates"] += 1
                    yield candidate

            shortlist = nsmallest(remaining_trials, priorities())
            truncated = diagnostics["coordinated_candidates"] - candidate_count > len(shortlist)
            diagnostics["coordinated_search_truncated"] |= truncated
            pass_record = dict(anchor=sorted(best_support.items()), trials=0,
                               shortlisted=len(shortlist), truncated=truncated)
            records.append(pass_record)
            diagnostics["times"]["search"] += perf_counter() - tick
            for _, _, key in shortlist:
                size(dict(key), "coordinated")
                pass_record["trials"] += 1
                remaining_trials -= 1
                # Re-anchor immediately after an improvement. Spending the entire
                # shortlist first starves later anchors when it fills the budget.
                if (expanded and best_value < anchor_value - config.objective_tolerance
                        and pass_index + 1 < config.improvement_passes):
                    break
            if best_value >= anchor_value - config.objective_tolerance:
                break
    if config.exchange:
        improve(False, config.rescue_trials, diagnostics["coordinated_passes"])
    optional_count = sum(any(s.ticket for s in row) for i, row in enumerate(states)
                         if i != problem.cash and not problem.mandatory[i])
    if (config.exchange and config.improvement_passes and config.rescue_trials
            and 0 < limit < optional_count):
        # Retain the completed original search. Changing its order can discard
        # useful paths even when every individual update improves the incumbent.
        tick = perf_counter()
        diagnostics["qp_counts"]["guidance"] += 1
        try:
            guided, native = _guided_support(problem, states, limit, config.qp_solver)
        except (cp.SolverError, ValueError) as error:
            guided, native = None, str(error)
        diagnostics["guidance_status"] = native
        diagnostics["times"]["search"] += perf_counter() - tick
        retained = best, best_value, best_support
        if guided is not None and tuple(sorted(guided.items())) not in visited:
            # An independently audited seed can open a useful search path even
            # when initially inferior. Keep the completed original incumbent
            # separately, and compare the two only after the bounded branch.
            best, best_value, best_support = None, np.inf, {}
            size(guided, "coordinated")
            diagnostics["coordinated_seed_trials"] = 1
            if best is None:
                best, best_value, best_support = retained
            else:
                single_passes()
        improve(True, config.rescue_trials - diagnostics["coordinated_seed_trials"],
                diagnostics["scarce_passes"])
        if best_value >= retained[1] - config.objective_tolerance:
            best, best_value, best_support = retained
        anchor = dict(best_support)

        def restart(seed, label, native=None, records=None):
            """Explore a separate seed, including one-swap rescue of an infeasible rounding.

            Cache only outcomes of the unchanged original-domain sizing QP. A
            branch has its own visited set so an earlier audited candidate can
            supply an intermediate point without another native solve.
            """
            nonlocal best, best_value, best_support, visited
            previous = best, best_value, best_support
            visited = set()
            best, best_value, best_support = None, np.inf, {}
            size(seed, "restart")
            record = dict(label=label, guidance_status=native, seed=sorted(seed.items()), passes=[])
            (diagnostics["restarts"] if records is None else records).append(record)
            for _ in range(config.improvement_passes):
                before = best_value
                support = dict(best_support) if best is not None else dict(seed)
                record["passes"].append(sorted(support.items()))
                for old in sorted(support, key=lambda i: problem.names[i]):
                    for i in sorted(range(len(states)), key=lambda i: problem.names[i]):
                        if i == problem.cash or problem.mandatory[i] or i in support:
                            continue
                        for j, state in enumerate(states[i]):
                            if state.ticket:
                                size({k: v for k, v in support.items() if k != old} | {i: j},
                                     "restart")
                if best_value >= before - config.objective_tolerance:
                    break
            record["audited"] = best is not None
            record["improved"] = best_value < previous[1] - config.objective_tolerance
            if not record["improved"]:
                best, best_value, best_support = previous

        # A final pair can make a replacement useful after the last earlier
        # sweep. Revisit the completed incumbent before exploring other seeds.
        restart(anchor, "completed_incumbent")
        seeds = {tuple(sorted(anchor.items()))}
        if guided is not None and tuple(sorted(guided.items())) not in seeds:
            seeds.add(tuple(sorted(guided.items())))
            restart(guided, "unrestricted_guidance", diagnostics["guidance_status"])
        for i in sorted(anchor, key=lambda i: problem.names[i])[:config.rescue_trials]:
            zero = next((state for state in states[i] if not state.ticket), None)
            if zero is None:
                continue
            lower, upper = problem.lower.copy(), problem.upper.copy()
            lower[i], upper[i] = zero.lower, zero.upper
            restricted = replace(problem, lower=lower, upper=upper)
            tick = perf_counter()
            diagnostics["qp_counts"]["restart_guidance"] += 1
            try:
                seed, native = _guided_support(
                    restricted, compile_states(restricted), limit, config.qp_solver
                )
            except (cp.SolverError, ValueError) as error:
                seed, native = None, str(error)
            diagnostics["times"]["search"] += perf_counter() - tick
            key = None if seed is None else tuple(sorted(seed.items()))
            if key is not None and key not in seeds:
                seeds.add(key)
                restart(seed, problem.names[i], native)
            else:
                diagnostics["restarts"].append(dict(label=problem.names[i],
                    guidance_status=native, skipped="failed" if seed is None else "duplicate"))
        # Empty incumbents have no exclusions to diversify. A tentative relaxed
        # support still supplies useful conditional decisions even when its
        # rounded seed is infeasible. Include as well as exclude those states;
        # every resulting executable support is sized on the original problem.
        conditional = diagnostics["conditional_restarts"]
        proposals = sorted(set(best_support.items()) | set((guided or {}).items()),
                           key=lambda item: (problem.names[item[0]], item[1]))
        decisions = ((i, j, enabled) for i, j in proposals if len(states[i]) > 1
                     for enabled in (True, False))
        for decision in islice(decisions, config.rescue_trials):
            i, j, enabled = decision
            tick = perf_counter()
            diagnostics["qp_counts"]["conditional_guidance"] += 1
            try:
                seed, native = _guided_support(
                    problem, states, limit, config.qp_solver, decision
                )
            except (cp.SolverError, ValueError) as error:
                seed, native = None, str(error)
            diagnostics["times"]["search"] += perf_counter() - tick
            key = None if seed is None else tuple(sorted(seed.items()))
            record = dict(instrument=problem.names[i], state=j, included=enabled,
                          native_status=native, seed=key)
            diagnostics["conditional_guidance"].append(record)
            if key is not None and key not in seeds:
                seeds.add(key)
                restart(seed, f"{'include' if enabled else 'exclude'}:{problem.names[i]}:{j}",
                        native, conditional)
            else:
                record["skipped"] = "failed" if seed is None else "duplicate"
        if conditional:
            restart(dict(best_support), "conditional_completed", records=conditional)
        # Two-coordinate priorities hold retained positions fixed, whereas the
        # sizing QP can resize them. Complete a wider, screened neighbourhood
        # after the earlier search: a low priority is not a rejection certificate.
        # Reuse all cached outcomes; the extra pair budget is bounded by R^2,
        # shared across the existing number of incumbent anchors.
        visited = set(sizing_cache)
        improve(True, config.rescue_trials ** 2, diagnostics["completion_passes"], screen=True)
        # Improving-only descent can strand a feasible three-name replacement.
        # Reuse the best audited alternatives already sized, in deterministic
        # objective/name order. A separate branch may cross an inferior seed;
        # restart() retains the incumbent unless the completed branch improves.
        alternatives = nsmallest(config.rescue_trials, (
            (value[0], tuple(sorted((problem.names[i], j) for i, j in key)), key)
            for key, value in sizing_cache.items() if value is not None
            and key != tuple(sorted(best_support.items()))
        ))
        for seed_value, _, key in alternatives:
            incumbent_value = best_value
            restart(dict(key), "cached_alternative", records=diagnostics["bridge_restarts"])
            record = diagnostics["bridge_restarts"][-1]
            record.update(seed_value=seed_value, incumbent_value=incumbent_value,
                          completed_value=best_value)
            if record["improved"]:
                # Complete the newly reached basin once, then stop. This uses a
                # separate R^2 pair budget and the same bounded anchor count.
                visited = set(sizing_cache)
                improve(True, config.rescue_trials ** 2,
                        diagnostics["bridge_completion_passes"], screen=True)
                break
    if config.compute_bound:
        from .bounds import compute_perspective_lower_bound

        tick = perf_counter()
        diagnostics["bounds"] = compute_perspective_lower_bound(problem)
        diagnostics["times"]["bound"] = perf_counter() - tick
    diagnostics["times"]["total"] = perf_counter() - started
    diagnostics["support"] = {problem.names[i]: j for i, j in best_support.items()}
    diagnostics["visited_supports"] = len(sizing_cache)
    status = "audited_incumbent" if best is not None else "bounded_abstention"
    return result(problem, best, status, diagnostics)
