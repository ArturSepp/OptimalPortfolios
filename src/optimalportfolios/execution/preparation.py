"""Risk-neutral common preparation before ranked or matched Exact execution.

Preparation preserves the supplied original book and hard domain. It first
repairs inside the corridor; a hard-domain attempt explicitly minimises corridor
excursion before turnover and additional repair tickets. Tied repair supports
are ordered by instrument identity before fixed-state sizing. Genuine new repairs
are pinned, counted and declared before any tracking-risk search. Numerical base
anchors inside the supplied corridor retain its discretionary trade intervals.
"""
from dataclasses import replace
from time import perf_counter

import cvxpy as cp
import numpy as np

from .domain import audit_execution, domain_bounds, floor_intervals

_SCALE = 1e4
_OBJECTIVE_GUARD_BP = 1e-7


def _holding_intervals(problem, i, lower, upper, floor):
    """Retain an audit-zero incoming hold inside a strictly positive corridor."""
    intervals = floor_intervals(lower, upper, floor)
    base = problem.base[i]
    # The audit's zero convention does not require a floor-sized purchase for
    # positive numerical dust. Keep its exact hold; never lower the supplied bound.
    if (not problem.mandatory[i] and 0 < lower <= base <= upper
            and base <= problem.rules.zero_tolerance and base < floor):
        intervals.insert(0, (base, base))
    return intervals


def _no_repair_bounds(problem, lo, hi):
    """Represent numerical corridor movements without opening a material repair.

    Only a whole numerical corridor, or the nearest legal endpoint, may replace
    the fixed base. Every point stays within the existing zero convention of both
    base and original holdings; all coupled rows still need a constrained solve.
    """
    anchors = problem.base.copy()
    upper_anchors = anchors.copy()
    floor = np.broadcast_to(problem.rules.holding_floor, anchors.shape)
    for i in range(len(anchors)):
        if i == problem.cash or problem.mandatory[i]:
            continue
        # FULL still minimises corridor excursion first. A numerical corridor
        # correction must not become a free material repair merely because the
        # fallback domain has wider hard bounds.
        domains = ((max(lo[i], problem.lower[i]), min(hi[i], problem.upper[i])),
                   (lo[i], hi[i]))
        for lower, upper in domains:
            intervals = _holding_intervals(problem, i, lower, upper, floor[i])
            if not intervals:
                continue
            candidates = [min(max(anchors[i], low), high) for low, high in intervals]
            endpoint = min(candidates, key=lambda value: abs(value-anchors[i]))
            if (abs(endpoint-anchors[i]) <= problem.rules.zero_tolerance
                    and abs(endpoint-problem.current[i]) <= problem.rules.zero_tolerance):
                anchors[i] = upper_anchors[i] = endpoint
                for low, high in intervals:
                    if (low <= endpoint <= high
                            and high-low <= problem.rules.zero_tolerance
                            and max(abs(low-problem.base[i]), abs(high-problem.base[i]),
                                    abs(low-problem.current[i]), abs(high-problem.current[i]))
                            <= problem.rules.zero_tolerance):
                        anchors[i], upper_anchors[i] = low, high
                        break
                break
    return anchors, upper_anchors


def _graph(problem, mode, choices=None):
    """Compile risk-neutral floor and repair-support disjunctions in bp units."""
    lo, hi = (x.copy() for x in domain_bounds(problem, mode))
    n, cash = len(lo), problem.cash
    noncash = np.flatnonzero(np.arange(n) != cash)
    floor = np.broadcast_to(problem.rules.holding_floor, (n,))
    no_lower, no_upper = _no_repair_bounds(problem, lo, hi)
    optional = noncash[~problem.mandatory[noncash] & (lo[noncash] != hi[noncash])]
    disjunctive = []
    unheld = np.zeros(n)
    for i in noncash:
        intervals = _holding_intervals(problem, i, lo[i], hi[i], floor[i])
        if not intervals:
            raise ValueError(f'No holding state for {problem.names[i]}')
        if len(intervals) == 1:
            lo[i], hi[i] = intervals[0]
        else:
            disjunctive.append(i)
            unheld[i] = intervals[0][0]
    if choices is not None:
        for i, held in zip(disjunctive, choices['held']):
            if held:
                lo[i] = max(lo[i], floor[i])
            else:
                lo[i] = hi[i] = unheld[i]
        for i, moved in zip(optional, choices['repair']):
            if not moved:
                lo[i], hi[i] = no_lower[i], no_upper[i]
        for i, endpoint in choices.get('anchors', {}).items():
            lo[i] = hi[i] = endpoint
        if np.any(lo > hi):
            raise ValueError('Fixed repair support contradicts a holding state')
    movable = np.flatnonzero((lo != hi) & (np.arange(n) != cash))
    reference = lo.copy()
    reference[cash] = problem.investment-reference[noncash].sum()
    direction = np.zeros((n, len(movable)))
    direction[movable, np.arange(len(movable))] = 1.
    direction[cash] = -1.
    step = cp.Variable(len(movable)) if len(movable) else None
    weights = cp.Constant(reference*_SCALE)
    if step is not None:
        weights = weights + direction@step
    rows = [weights >= lo*_SCALE, weights <= hi*_SCALE]
    tolerance = problem.rules.audit_tolerance
    if len(problem.group_lower):
        # A conservative interior margin on movable rows keeps native rounding
        # inside the unchanged original-space audit. Fixed rows retain their
        # full audit allowance; no numerical repair can change those exposures.
        movable_groups = np.any(problem.groups@direction != 0., axis=1)
        allowance = tolerance-np.where(movable_groups, min(1e-11, tolerance/1000), 0.)
        lower = np.nextafter(problem.group_lower-allowance, problem.group_lower)
        upper = np.nextafter(problem.group_upper+allowance, problem.group_upper)
        rows += [problem.groups@weights >= lower*_SCALE,
                 problem.groups@weights <= upper*_SCALE]
    variables = {}
    if choices is None and disjunctive:
        held = cp.Variable(len(disjunctive), boolean=True)
        variables['held'] = (held, disjunctive)
        anchors = unheld[disjunctive]*_SCALE
        rows += [weights[disjunctive] >= anchors
                 + cp.multiply(floor[disjunctive]*_SCALE-anchors, held),
                 weights[disjunctive] <= anchors
                 + cp.multiply(hi[disjunctive]*_SCALE-anchors, held)]
    if choices is None and len(optional):
        repair = cp.Variable(len(optional), boolean=True)
        variables['repair'] = (repair, optional)
        rows += [weights[optional] >= no_lower[optional]*_SCALE
                 + cp.multiply((lo[optional]-no_lower[optional])*_SCALE, repair),
                 weights[optional] <= no_upper[optional]*_SCALE
                 + cp.multiply((hi[optional]-no_upper[optional])*_SCALE, repair)]
        count = cp.sum(repair)
    else:
        count = cp.Constant(sum(choices['repair']) if choices is not None else 0.)
    # Fixed and one-sided rows need no absolute-value epigraph. Eliminating
    # them prevents per-row native residuals accumulating in lexicographic caps.
    up = noncash[lo[noncash] >= problem.base[noncash]]
    down = noncash[(lo[noncash] < problem.base[noncash])
                   & (hi[noncash] <= problem.base[noncash])]
    two = noncash[(lo[noncash] < problem.base[noncash])
                  & (hi[noncash] > problem.base[noncash])]
    turnover = cp.Constant(0.)
    for indices, sign in ((up, 1.), (down, -1.)):
        if len(indices):
            turnover += sign*cp.sum(weights[indices]-problem.base[indices]*_SCALE)
    if len(two):
        turnover += cp.norm1(weights[two]-problem.base[two]*_SCALE)
    excursion = cp.Constant(0.)
    for i in range(n):
        if hi[i] <= problem.lower[i]:
            excursion += problem.lower[i]*_SCALE-weights[i]
        elif lo[i] < problem.lower[i]:
            excursion += cp.pos(problem.lower[i]*_SCALE-weights[i])
        if lo[i] >= problem.upper[i]:
            excursion += weights[i]-problem.upper[i]*_SCALE
        elif hi[i] > problem.upper[i]:
            excursion += cp.pos(weights[i]-problem.upper[i]*_SCALE)
    return weights, rows, variables, count, turnover, excursion, reference, movable


def _solve(objective, rows, solver, deadline, log, label):
    """Spend only the remaining common budget and preserve every native status."""
    remaining = deadline-perf_counter()
    if remaining <= 0:
        log.append(dict(stage=label, status='time_limit'))
        return False
    model = cp.Problem(cp.Minimize(objective), rows)
    started = perf_counter()
    remaining = deadline-started
    if remaining <= 0:
        log.append(dict(stage=label, status='construction_time_limit'))
        return False
    options = {}
    if solver == 'MOSEK':
        options['mosek_params'] = {
            'MSK_DPAR_MIO_MAX_TIME': remaining, 'MSK_DPAR_OPTIMIZER_MAX_TIME': remaining,
            'MSK_IPAR_NUM_THREADS': 1, 'MSK_IPAR_MIO_SEED': 0,
            'MSK_DPAR_MIO_TOL_REL_GAP': 0., 'MSK_DPAR_MIO_TOL_ABS_GAP': 1e-9,
            'MSK_DPAR_MIO_TOL_FEAS': 1e-9,
            'MSK_DPAR_MIO_TOL_ABS_RELAX_INT': 1e-9,
        }
    elif solver in ('SCIPY', 'SCIPY_REPAIR'):
        options['scipy_options'] = dict(time_limit=remaining, mip_rel_gap=0.)
        if solver == 'SCIPY_REPAIR':
            # HiGHS' independent numerical retry preserves the mathematical
            # graph. Disable presolve and tighten native precision only; the
            # original-space audit and objective guards are unchanged.
            options['scipy_options'].update(
                presolve=False, mip_feasibility_tolerance=1e-10,
                primal_feasibility_tolerance=1e-10, dual_feasibility_tolerance=1e-10)
            solver = 'SCIPY'
    elif solver == 'CLARABEL':
        options.update(time_limit=remaining, tol_gap_abs=1e-10,
                       tol_gap_rel=1e-10, tol_feas=1e-10)
    try:
        model.solve(solver=solver, **options)
        log.append(dict(stage=label, status=str(model.status), solver=solver,
                        seconds=perf_counter()-started, objective=model.value))
        return model.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE)
    except cp.error.SolverError as error:
        log.append(dict(stage=label, status='solver_error', solver=solver,
                        seconds=perf_counter()-started, error=str(error)))
        return False


def _binary_values(variable, log, label):
    """Validate a native discrete choice before using it to restrict the graph."""
    value = np.asarray(variable.value, dtype=float)
    if not np.isfinite(value).all() or np.max(np.abs(value-np.rint(value))) > 1e-7:
        log.append(dict(stage=label, status='invalid_integrality'))
        return None
    return np.rint(value).astype(int)


def _canonical_choices(problem, variables, rows, count, solver, deadline, log, mode):
    """Choose lexicographic repair identities, then prefer retained holding states.

    The primary objective caps are already in ``rows``. Selecting the next
    smallest repair identity needs one bounded-coefficient MILP per repair,
    rather than exponential binary weights or one solve per universe asset.
    Every tie-break must prove its integer optimum within the shared deadline.
    """
    choices = {'held': [], 'repair': []}
    for name, (variable, _) in variables.items():
        value = _binary_values(variable, log, mode)
        if value is None:
            return None
        choices[name] = value.tolist()
    selected = set()
    if 'repair' in variables:
        repair, indices = variables['repair']
        needed = sum(choices['repair'])
        rows = rows + [count == needed]
        remaining = sorted(range(len(indices)), key=lambda j: problem.names[indices[j]])
        choices['repair'] = [0]*len(indices)
        for slot in range(needed):
            first = cp.Variable(len(remaining), boolean=True)
            trial_rows = rows + [cp.sum(first) == 1, first <= repair[remaining]]
            label = f'{mode}:repair_identity:{slot}'
            if not _solve(np.arange(1, len(remaining)+1)@first, trial_rows,
                          solver, deadline, log, label):
                return None
            value = _binary_values(first, log, label)
            if value is None or log[-1]['status'] != cp.OPTIMAL:
                return None
            position = int(np.argmax(value))
            chosen = remaining[position]
            if position:
                rows = rows + [repair[remaining[:position]] == 0]
            rows = rows + [repair[chosen] == 1]
            choices['repair'][chosen] = 1
            selected.add(int(indices[chosen]))
            remaining = remaining[position+1:]
            log[-1]['selected_name'] = problem.names[indices[chosen]]
        # Exact fixed support also removes irrelevant inactive floor branches.
        rows = rows + [repair == choices['repair']]
    if 'held' in variables:
        held, indices = variables['held']
        no_lower, _ = _no_repair_bounds(problem, *domain_bounds(problem, mode))
        floor = np.broadcast_to(problem.rules.holding_floor, problem.base.shape)
        for j in sorted(range(len(indices)), key=lambda j: problem.names[indices[j]]):
            i = indices[j]
            if not problem.mandatory[i] and i not in selected:
                choice = int(no_lower[i] >= floor[i])
            else:
                label = f'{mode}:holding_identity:{problem.names[i]}'
                retained = (problem.base[i] > problem.rules.zero_tolerance
                            or problem.base[i] >= floor[i])
                objective = -held[j] if retained else held[j]
                if not _solve(objective, rows, solver, deadline, log, label):
                    return None
                value = _binary_values(held, log, label)
                if value is None or log[-1]['status'] != cp.OPTIMAL:
                    return None
                choice = int(value[j])
            choices['held'][j] = choice
            rows = rows + [held[j] == choice]
    return choices


def _retry_repair_choices(problem, mode, limits, deadline, log):
    """Re-prove a contradictory discrete optimum within unchanged primary caps.

    Near-integral native flags can admit tiny flows that disappear when a
    canonical prefix is fixed. A fresh graph removes that unproved count cap,
    then minimises the count again under the original excursion and turnover
    caps. The same deadline, identity order and final sizing/audit still apply.
    """
    _, rows, variables, count, turnover, excursion, _, _ = _graph(problem, mode)
    caps = {'turnover': limits['turnover']+_OBJECTIVE_GUARD_BP}
    rows += [turnover <= caps['turnover']]
    if 'excursion' in limits:
        caps['excursion'] = limits['excursion']+_OBJECTIVE_GUARD_BP
        rows += [excursion <= caps['excursion']]
    log.append(dict(stage=mode, status='discrete_retry', solver='SCIPY', primary_caps=caps))
    if (not _solve(count, rows, 'SCIPY_REPAIR', deadline, log, f'{mode}:retry_repair_count')
            or log[-1]['status'] != cp.OPTIMAL):
        return None
    return _canonical_choices(
        problem, variables, rows, count, 'SCIPY_REPAIR', deadline, log, mode)


def _repair(problem, mode, solver, deadline, log):
    """Minimise primary repairs, canonically select tied states, then size them."""
    try:
        weights, rows, variables, count, turnover, excursion, _, _ = _graph(problem, mode)
    except ValueError as error:
        log.append(dict(stage=mode, status='empty_domain', error=str(error)))
        return None
    levels = [('excursion', excursion)] if mode == 'FULL' else []
    levels += [('turnover', turnover), ('repair_count', count)]
    limits = {}
    for label, objective in levels:
        if not _solve(objective, rows, solver, deadline, log, f'{mode}:{label}'):
            return None
        value = float(objective.value)
        limits[label] = value
        rows = rows + [objective <= value+_OBJECTIVE_GUARD_BP]
    choices = _canonical_choices(problem, variables, rows, count, solver, deadline, log, mode)
    if (choices is None and log[-1]['status'] in (cp.INFEASIBLE, cp.INFEASIBLE_INACCURATE)
            and ':repair_identity:' in log[-1]['stage']):
        choices = _retry_repair_choices(problem, mode, limits, deadline, log)
    if choices is None:
        return None
    for attempt in range(2):
        weights, rows, _, _, turnover, excursion, reference, movable = _graph(
            problem, mode, choices)
        rows += [turnover <= limits['turnover']+_OBJECTIVE_GUARD_BP]
        if 'excursion' in limits:
            rows += [excursion <= limits['excursion']+_OBJECTIVE_GUARD_BP]
        # Percentage-point scaling changes no minimiser in this risk-neutral tie break.
        objective = cp.sum_squares((weights-problem.base*_SCALE)/100.)
        if not _solve(objective, rows, 'CLARABEL', deadline, log, f'{mode}:fixed_sizing:{attempt}'):
            return None
        candidate = reference.copy()
        candidate[movable] = np.asarray(weights.value, dtype=float)[movable]/_SCALE
        # Eliminate declared constants exactly; never round solved free weights.
        noncash = np.arange(len(candidate)) != problem.cash
        candidate[problem.cash] = problem.investment-candidate[noncash].sum()
        anchors = {}
        domain_lo, domain_hi = domain_bounds(problem, mode)
        floor = np.broadcast_to(problem.rules.holding_floor, candidate.shape)
        for i in movable:
            intervals = _holding_intervals(problem, i, domain_lo[i], domain_hi[i], floor[i])
            valid = any(low <= candidate[i] <= high for low, high in intervals)
            if not valid:
                endpoints = [x for interval in intervals for x in interval]
                endpoint = min(endpoints, key=lambda x: abs(x-candidate[i]))
                if abs(endpoint-candidate[i]) > problem.rules.zero_tolerance:
                    return None
                anchors[int(i)] = float(endpoint)
        if not anchors:
            return candidate
        log.append(dict(stage=mode, status='endpoint_resolve', anchors=anchors))
        # A fresh constrained solve validates boundary anchors and all original
        # rows together. Invalid native endpoints are never clipped for delivery.
        choices['anchors'] = anchors
    return None


def _declare(problem, weights, mode):
    """Pin genuine interventions while retaining legal numerical-anchor intervals."""
    lo, hi = problem.lower.copy(), problem.upper.copy()
    full_lo, full_hi = problem.full_lower.copy(), problem.full_upper.copy()
    categories = list(problem.categories)
    mandatory = problem.mandatory.copy()
    exempt = np.broadcast_to(problem.rules.exemption_mask, weights.shape).copy()
    base = problem.base.copy()
    repairs, numerical, exceptions = [], [], []
    for i in range(len(weights)):
        if i == problem.cash:
            base[i] = weights[i]
        elif mandatory[i]:
            base[i] = weights[i]
        elif weights[i] != base[i]:
            endpoint = weights[i]
            if endpoint < full_lo[i] or endpoint > full_hi[i]:
                raise ValueError(f'Repair endpoint outside unchanged hard interval: '
                                 f'{problem.names[i]} {endpoint} [{full_lo[i]}, {full_hi[i]}]')
            row = dict(index=i, name=problem.names[i], before=float(base[i]),
                       endpoint=float(endpoint))
            outside_corridor = endpoint < lo[i] or endpoint > hi[i]
            if outside_corridor:
                exceptions.append(dict(row, original_lower=float(lo[i]),
                                       original_upper=float(hi[i])))
            material = abs(endpoint-problem.current[i]) > problem.rules.zero_tolerance
            base[i] = endpoint
            # A numerical anchor is a valid hold state, not a compulsory trade.
            # Retain its supplied buy/sell capacity. An out-of-corridor anchor
            # still needs the existing fixed, explicitly declared exception.
            if material or outside_corridor:
                lo[i] = hi[i] = full_lo[i] = full_hi[i] = endpoint
            if material:
                mandatory[i] = exempt[i] = True
                categories[i] = 'compulsory_saa_funding_repair'
                repairs.append(row)
            else:
                numerical.append(dict(row, bounds_preserved=not outside_corridor))
    rules = problem.rules
    if rules.discretionary_allowance is not None:
        rules = replace(rules, ticket_cap=int(mandatory.sum())+rules.discretionary_allowance)
    rules = replace(rules, exemption_mask=exempt)
    prepared = replace(problem, base=base, lower=lo, upper=hi, full_lower=full_lo,
                       full_upper=full_hi, mandatory=mandatory, categories=tuple(categories),
                       rules=rules)
    return prepared, dict(mode=mode, repairs=repairs, numerical_pins=numerical,
                          corridor_exceptions=exceptions)


def prepare_common_execution(problem, *, time_limit_s=30., solver='MOSEK'):
    """Return a common audited problem and preparation record, or explicit failure.

    The same preparation belongs to ranked and Exact callers. It uses no target,
    covariance or alpha to choose repairs. A fixed total cap is never expanded;
    an explicitly declared discretionary allowance remains additional to the new
    mandatory reservation. Genuine hard breaches are retained as failures.
    """
    if not np.isfinite(time_limit_s) or time_limit_s <= 0:
        raise ValueError('Preparation budget must be finite and positive')
    started = perf_counter()
    log = []
    record = dict(status='already_audited', stages=log,
                  original_audit=audit_execution(problem, problem.base))
    if record['original_audit']['passed']:
        record['seconds'] = perf_counter()-started
        return problem, record
    for mode in ('CORRIDOR', 'FULL'):
        weights = _repair(problem, mode, solver, started+time_limit_s, log)
        if weights is None:
            continue
        try:
            prepared, declaration = _declare(problem, weights, mode)
            audit = audit_execution(prepared, prepared.base)
        except ValueError as error:
            log.append(dict(stage=mode, status='declaration_rejected', error=str(error)))
            continue
        record.update(declaration, prepared_audit=audit, seconds=perf_counter()-started)
        if not audit['passed']:
            record['status'] = 'prepared_base_rejected'
            return None, record
        record['status'] = 'prepared'
        prepared = replace(
            prepared, preparation_seconds=problem.preparation_seconds+record['seconds'])
        return prepared, record
    record.update(status='preparation_failed', seconds=perf_counter()-started)
    return None, record
