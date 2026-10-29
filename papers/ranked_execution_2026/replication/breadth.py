"""Frozen broader-domain ablation with separate, time-controlled Exact starts."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import hashlib
import importlib.metadata
from itertools import combinations, product
import json
from pathlib import Path
import platform
from time import perf_counter

import cvxpy as cp
import numpy as np
import optimalportfolios
from optimalportfolios.execution.v2 import (
    ExecutionProblem, ExecutionRules, RankedConfig, compile_states,
    prepare_common_execution, wrapper_ranked_execution, wrapper_exact_execution,
    compute_perspective_lower_bound,
)
from .benchmark import ARRAYS, independent_audit, tracking_error_bp, sha256
from .run_benchmark import write_json

PROTOCOL = Path(__file__).with_name('breadth_protocol.json')
METHODS = ('Ranked', 'OneAnchor', 'ExactBase10', 'ExactBase60', 'ExactFull60', 'ExactOne60')


def case_grid(spec):
    """Declare every broad and tiny case independently of any solver result."""
    fields = ('optional_assets', 'budget', 'groups', 'regime', 'seed')
    broad = [dict(zip(fields, values), sample='broad')
             for values in product(*(spec['grid'][k] for k in fields))]
    tiny = [dict(optional_assets=spec['tiny']['optional_assets'],
                 budget=spec['tiny']['budget'], groups=group,
                 regime=spec['tiny']['regime'], seed=seed, sample='tiny')
            for group, seed in product(spec['tiny']['groups'], spec['tiny']['seed'])]
    return [dict(case, id=f'breadth_{i:03}') for i, case in enumerate(broad+tiny)]


def generated_case(case, spec):
    """Generate heterogeneous holdings and correlated risk without estimated/private data."""
    optional = case['optional_assets']
    n, cash = optional+3, optional+2
    rng = np.random.default_rng(np.random.SeedSequence([case['seed'], optional]))
    relative = np.exp(rng.normal(0., .45, n-1))
    current_floor = spec['current_minimum']
    current = np.r_[current_floor+(.94-current_floor*(n-1))*relative/relative.sum(), .06]
    base = current.copy()
    base[cash] += base[:2].sum()
    base[:2] = 0.
    desired = current[2:cash]*np.exp(rng.normal(0., 1., optional))
    target = np.r_[0., 0., .98*desired/desired.sum(), .02]
    sector = np.zeros((3, n))
    sector[np.arange(n-1) % 3, np.arange(n-1)] = 1.
    baskets = np.c_[rng.uniform(.05, .95, (3, n-1)), np.zeros(3)]
    groups = sector if case['groups'] == 'disjoint' else np.vstack([sector, baskets])
    # This is a known synthetic PSD construction, not covariance estimation.
    loadings = np.c_[rng.uniform(.04, .12, n), .10*sector.T,
                     rng.normal(0., .04, (n, 2))]
    loadings[cash] = 0.
    residual = np.r_[rng.uniform(.04, .09, n-1)**2, .001**2]
    covariance = loadings@loadings.T+np.diag(residual)
    lower, upper = np.zeros(n), np.full(n, .35)
    upper[:2] = 0.
    regime = spec['regimes'][case['regime']]
    if 'cash_half_width' in regime:
        lower[cash], upper[cash] = base[cash]-regime['cash_half_width'], base[cash]+regime['cash_half_width']
    else:
        lower[cash], upper[cash] = regime['cash_lower'], regime['cash_upper']
    lo, hi = np.minimum(base, target), np.minimum(np.maximum(base, target), upper)
    lo[cash], hi[cash] = lower[cash], upper[cash]
    exposure, width = groups@base, regime['group_half_width']
    mandatory = np.arange(n) < 2
    return ExecutionProblem(names=tuple([f'Asset{i:03}' for i in range(n-1)]+['Cash']),
        cash=cash, current=current, base=base, target=target, lower=lo, upper=hi,
        full_lower=lower, full_upper=upper, groups=groups,
        group_lower=np.maximum(0., exposure-width), group_upper=exposure+width,
        mandatory=mandatory, categories=('model_exit', 'model_exit')+('',)*(n-2),
        eligible=np.ones(n, bool), frozen=np.zeros(n, bool), covariance=covariance,
        rules=ExecutionRules(minimum_ticket=spec['minimum_ticket'], holding_floor=spec['holding_floor'],
            ticket_cap=2+case['budget'], discretionary_allowance=case['budget'],
            audit_tolerance=spec['audit_tolerance'], zero_tolerance=spec['zero_tolerance']),
        provenance={'benchmark': 'breadth-generated', 'seed': case['seed']})


def fingerprint(problem):
    """Hash canonical array bytes and all solver-relevant declarations without timestamps."""
    digest = hashlib.sha256()
    for name in ARRAYS:
        value = np.asarray(getattr(problem, name))
        digest.update(name.encode()+str(value.shape).encode()+value.dtype.str.encode()+value.tobytes())
    rules = asdict(problem.rules)
    for key in ('holding_floor', 'exemption_mask'):
        rules[key] = np.asarray(rules[key]).tolist()
    digest.update(json.dumps(dict(rules=rules, names=problem.names, cash=problem.cash,
        categories=problem.categories, investment=problem.investment), sort_keys=True).encode())
    return digest.hexdigest()


def serial_arms(problem, spec, index=0, preparation_seconds=0.):
    """Run independent starts and record their distinct generation costs and failures."""
    records = {}
    heuristics = ['Ranked', 'OneAnchor']
    independent = ['ExactBase10', 'ExactBase60']
    started = ['ExactFull60', 'ExactOne60']
    order = sum([list(reversed(x)) if index % 2 else x for x in
                 (heuristics, independent, started)], [])
    original = fingerprint(problem)
    for method in order:
        tick = perf_counter()
        source = {'ExactFull60': 'Ranked', 'ExactOne60': 'OneAnchor'}.get(method)
        supplied = records.get(source, {})
        start = supplied.get('weights') if supplied.get('audit', {}).get('passed') else None
        try:
            if method.startswith('Exact'):
                result = wrapper_exact_execution(problem, 'CORRIDOR', mip_start=start,
                    time_limit_s=spec['exact_seconds'][0 if method == 'ExactBase10' else 1],
                    threads=spec['exact_threads'], gap_rel=spec['exact_gap_rel'])
            else:
                config = RankedConfig()
                if method == 'OneAnchor':
                    config = replace(config, improvement_passes=1)
                result = wrapper_ranked_execution(problem, config)
            item = dict(status=result.status, weights=result.weights,
                        diagnostics=result.diagnostics, seconds=perf_counter()-tick,
                        audit=independent_audit(problem, result.weights))
            if item['audit']['passed']:
                item['te_bp'] = tracking_error_bp(problem, result.weights)
        except (ValueError, RuntimeError, cp.SolverError) as error:
            item = dict(status='exception', error=repr(error), weights=None,
                        seconds=perf_counter()-tick, audit=independent_audit(problem, None))
        item.update(start_source=source if start is not None else 'common_base',
                    requested_start_source=source, start_generation_seconds=supplied.get('seconds', 0.))
        item['end_to_end_seconds'] = item['seconds']+preparation_seconds+item['start_generation_seconds']
        if fingerprint(problem) != original:
            raise ValueError('An arm mutated the shared problem')
        records[method] = item
    return records, order


# OP build_model is deliberately not used: this tiny reference must independently
# construct the full-weight QP and generated monotone trade intervals.
def enumerate_tiny(problem):
    """Enumerate every support in the generated one-direction, positive-target tiny domain.

    Numerical statuses and rejected vectors remain explicit. A conic primal is
    not a certified lower bound; unresolved supports prevent a completeness claim.
    """
    optional = [i for i in range(len(problem.names)) if i != problem.cash and not problem.mandatory[i]]
    if len(optional) > 8 or np.any(problem.target[optional] <= 0):
        raise ValueError('Independent enumeration is restricted to positive-target tiny cases')
    tau, minimum = problem.rules.audit_tolerance, problem.rules.minimum_ticket
    floor = np.broadcast_to(problem.rules.holding_floor, problem.current.shape)
    intervals = {}
    for i in optional:
        if problem.target[i] > problem.current[i]:
            lo, hi = max(problem.lower[i], problem.current[i]+minimum, floor[i]), problem.upper[i]
        else:
            lo, hi = max(problem.lower[i], floor[i]), min(problem.upper[i], problem.current[i]-minimum)
        if lo <= hi:
            intervals[i] = lo, hi
    capacity = min(problem.rules.discretionary_allowance,
                   problem.rules.ticket_cap-problem.mandatory_count)
    best, best_te, trials = None, np.inf, []
    tick = perf_counter()
    for count in range(capacity+1):
        for support in combinations(intervals, count):
            lower, upper = problem.base.copy(), problem.base.copy()
            lower[problem.cash], upper[problem.cash] = problem.lower[problem.cash], problem.upper[problem.cash]
            for i in support:
                lower[i], upper[i] = intervals[i]
            w_bp = cp.Variable(len(lower))
            rows = [cp.sum(w_bp) == 1e4*problem.investment,
                    w_bp >= 1e4*lower, w_bp <= 1e4*upper,
                    problem.groups@w_bp >= 1e4*np.nextafter(problem.group_lower-tau, np.inf),
                    problem.groups@w_bp <= 1e4*np.nextafter(problem.group_upper+tau, -np.inf)]
            # Full weights and TE are expressed in bp, then audited in NAV units.
            # This invertible scaling changes neither the objective minimiser nor rows.
            qp = cp.Problem(cp.Minimize(cp.quad_form(w_bp-1e4*problem.target,
                                                       cp.psd_wrap(problem.covariance))), rows)
            record = dict(support=list(support))
            try:
                qp.solve(solver='CLARABEL', tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10)
                weights = None if w_bp.value is None else np.asarray(w_bp.value)/1e4
                audit = independent_audit(problem, weights)
                record.update(status=qp.status, audit=audit)
                if audit['passed']:
                    te = tracking_error_bp(problem, weights)
                    record['te_bp'] = te
                    if te < best_te:
                        best, best_te = weights.copy(), te
            except cp.SolverError as error:
                record.update(status='exception', error=repr(error), audit=independent_audit(problem, None))
            trials.append(record)
    unresolved = [t for t in trials if t['status'] != cp.INFEASIBLE and
                  not (t['status'] == cp.OPTIMAL and t['audit']['passed'])]
    return dict(weights=best, te_bp=best_te if best is not None else None,
                audit=independent_audit(problem, best), complete=not unresolved,
                unresolved=len(unresolved), trials=trials, seconds=perf_counter()-tick)


def run(output, protocol_path=PROTOCOL, limit=None):
    """Freeze source/configuration and all generated input bytes before the first solve."""
    output = output.resolve()
    if 'onedrive' in str(output).lower():
        raise ValueError('Generated studies must be outside OneDrive')
    output.mkdir(parents=True, exist_ok=False)
    spec = json.loads(protocol_path.read_text())
    cases = case_grid(spec)
    inputs = output/'inputs'
    inputs.mkdir()
    declared = []
    for case in cases:
        problem = generated_case(case, spec)
        file = inputs/(case['id']+'.npz')
        np.savez_compressed(file, **{name: getattr(problem, name) for name in ARRAYS})
        declared.append(dict(case, input_sha256=sha256(file), fingerprint=fingerprint(problem)))
    package = Path(optimalportfolios.__file__).parent
    write_json(output/'protocol.json', dict(specification=spec, cases=declared,
        frozen_utc=datetime.now(timezone.utc).isoformat(), specification_sha256=sha256(protocol_path),
        ranked_config=asdict(RankedConfig()), python=platform.python_version(),
        versions={name: importlib.metadata.version(name) for name in
                  ('optimalportfolios','qis','numpy','pandas','scipy','cvxpy','Mosek','clarabel')},
        source_sha256={p.relative_to(package).as_posix(): sha256(p)
                       for p in sorted((package/'execution').rglob('*.py'))
                       if 'tests' not in p.parts and 'run_local' not in p.parts},
        runner_sha256=sha256(Path(__file__)), audit_sha256=sha256(Path(__file__).with_name('benchmark.py'))))
    for index, case in enumerate(declared[:limit]):
        raw = generated_case(case, spec)
        if fingerprint(raw) != case['fingerprint']:
            raise ValueError('Generated inputs changed after freeze')
        tick = perf_counter()
        problem, prep = prepare_common_execution(raw, time_limit_s=spec['preparation_seconds'], solver='MOSEK')
        elapsed = perf_counter()-tick
        record = dict(case, preparation=prep, preparation_seconds=elapsed, methods={})
        if problem is not None:
            record['base_audit'] = independent_audit(problem, problem.base)
            record['prepared_fingerprint'] = fingerprint(problem)
            record['optional_candidates'] = sum(any(s.ticket for s in row)
                for i, row in enumerate(compile_states(problem)) if i != problem.cash and not problem.mandatory[i])
            record['methods'], record['method_order'] = serial_arms(problem, spec, index, elapsed)
            record['relaxations'] = compute_perspective_lower_bound(problem)
            if case['sample'] == 'tiny':
                record['enumeration'] = enumerate_tiny(problem)
        write_json(output/(case['id']+'.json'), record)
        write_json(output/'progress.json', dict(completed=index+1, total=len(declared[:limit]), last=case['id']))
        print(case['id'], 'complete', flush=True)
    write_json(output/'complete.json', dict(cases=len(declared[:limit]), full_sample=limit is None))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=PROTOCOL)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    run(args.output, args.protocol, args.limit)
