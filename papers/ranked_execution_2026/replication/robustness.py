"""Frozen public-input sensitivity and generated scarce-ticket experiments.

No production application, fund workbook, private replay or price feed is imported.
The immutable execution problem and independent audit are the experiment boundary.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import importlib.metadata
import itertools
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import optimalportfolios
from optimalportfolios.execution.v2 import (
    ExecutionProblem, ExecutionRules, RankedConfig, compile_states,
    prepare_common_execution, wrapper_ranked_execution, wrapper_exact_execution,
)
from .benchmark import verify_bundle, load_case, independent_audit, tracking_error_bp, sha256
from .run_benchmark import write_json

PROTOCOL = Path(__file__).with_name('robustness_protocol.json')


def perturb(problem, seed, target_size, risk_size, variant):
    """Change only supplied target and/or covariance with funding and PSD preserved."""
    rng = np.random.default_rng(seed)
    direction = rng.uniform(-1., 1., len(problem.current))
    volatility = rng.uniform(-1., 1., len(problem.current))
    target = problem.target.copy()
    noncash = np.arange(len(target)) != problem.cash
    if variant in ('target', 'joint'):
        tilted = target[noncash]*np.exp(direction[noncash])
        if tilted.sum() > 0.:
            tilted *= target[noncash].sum()/tilted.sum()
        delta = tilted-target[noncash]
        size = .5*np.abs(delta).sum()
        target[noncash] += delta*min(1., target_size/size) if size > 0. else 0.
    covariance = problem.dense_covariance().copy()
    if variant in ('covariance', 'joint'):
        scale = np.exp(risk_size*volatility)
        covariance = scale[:, None]*covariance*scale[None, :]
    return replace(problem, target=target, covariance=covariance,
                   factor_loadings=None, factor_covariance=None, residual_variance=None)


def rebuild_corridor(raw, changed):
    """Recompute optional target corridors inside unchanged lifecycle and product bounds."""
    lower, upper = raw.lower.copy(), raw.upper.copy()
    optional = ~raw.mandatory & ~raw.frozen & raw.eligible
    optional[raw.cash] = False
    lower[optional] = np.maximum(raw.full_lower[optional],
                                np.minimum(raw.base[optional], changed.target[optional]))
    upper[optional] = np.minimum(raw.full_upper[optional],
                                np.maximum(raw.base[optional], changed.target[optional]))
    return replace(raw, lower=lower, upper=upper, target=changed.target,
                   covariance=changed.dense_covariance(), factor_loadings=None,
                   factor_covariance=None, residual_variance=None)


def synthetic_case(optional, budget, width, cash_band, seed):
    """Generate a funded PSD problem with two forced exits and many optional trades."""
    rng = np.random.default_rng(seed + optional)
    n, cash = optional+3, optional+2
    current = np.full(n, .94/(n-1))
    current[cash] = .06
    base = current.copy()
    base[:2] = 0.
    base[cash] += current[:2].sum()
    target = np.zeros(n)
    desired = np.exp(rng.normal(0., .85, optional))
    target[2:cash] = .98*desired/desired.sum()
    target[cash] = .02
    full_lower, full_upper = np.zeros(n), np.full(n, .35)
    full_upper[:2] = 0.
    if cash_band == 'narrow':
        full_lower[cash], full_upper[cash] = base[cash]-.002, base[cash]+.002
    else:
        full_lower[cash], full_upper[cash] = .005, .5
    lower, upper = np.minimum(base, target), np.minimum(np.maximum(base, target), full_upper)
    lower[cash], upper[cash] = full_lower[cash], full_upper[cash]
    groups = np.zeros((3, n))
    for i in range(n-1):
        groups[i % 3, i] = 1.
    exposure = groups@base
    factors = rng.normal(0., .08, (n, 4))
    factors[cash] = 0.
    covariance = factors@factors.T + np.diag(np.r_[np.full(n-1, .08**2), .001**2])
    mandatory = np.zeros(n, dtype=bool)
    mandatory[:2] = True
    return ExecutionProblem(names=tuple([f'Asset{i:03}' for i in range(n-1)]+['Cash']),
        cash=cash, current=current, base=base, target=target, lower=lower, upper=upper,
        full_lower=full_lower, full_upper=full_upper, groups=groups,
        group_lower=np.maximum(exposure-width, 0.), group_upper=exposure+width,
        mandatory=mandatory, categories=('model_exit', 'model_exit')+('',)*(n-2),
        eligible=np.ones(n, dtype=bool), frozen=np.zeros(n, dtype=bool),
        rules=ExecutionRules(minimum_ticket=.005, holding_floor=.0025,
                             ticket_cap=2+budget, discretionary_allowance=budget),
        covariance=covariance, provenance={'benchmark': 'synthetic-scarcity', 'seed': seed})


def prepare(raw):
    """Run identical production common preparation with its unchanged time limit."""
    start = perf_counter()
    problem, diagnostics = prepare_common_execution(raw, time_limit_s=30., solver='MOSEK')
    return problem, dict(diagnostics=diagnostics, seconds=perf_counter()-start)


def solve(problem, method='Ranked', start=None):
    """Retain all solver outcomes and an independent original-unit audit."""
    tick = perf_counter()
    try:
        if method.startswith('Exact'):
            result = wrapper_exact_execution(problem, 'CORRIDOR', mip_start=start,
                                              time_limit_s=10., threads=1, gap_rel=1e-4)
        else:
            config = RankedConfig()
            if method == 'Greedy':
                config = replace(config, drop=False, exchange=False, rescue_trials=0)
            elif method == 'OneAnchor':
                config = replace(config, improvement_passes=1)
            result = wrapper_ranked_execution(problem, config)
        item = dict(status=result.status, weights=result.weights, diagnostics=result.diagnostics,
                    seconds=perf_counter()-tick, audit=independent_audit(problem, result.weights))
        if result.weights is not None:
            item['te_bp'] = tracking_error_bp(problem, result.weights)
        return item
    except (ValueError, RuntimeError) as error:
        return dict(status='exception', error=f'{type(error).__name__}: {error}', weights=None,
                    seconds=perf_counter()-tick, audit=independent_audit(problem, None))


def changed_metrics(reference, baseline, problem, result):
    """Compare allocations, support and risk against one unchanged economic reference."""
    if not baseline['audit']['passed'] or not result['audit']['passed']:
        return {}
    w0, w1 = np.asarray(baseline['weights']), np.asarray(result['weights'])
    noncash = np.arange(len(w0)) != reference.cash
    before = (np.abs(w0-reference.current) > reference.rules.zero_tolerance) & noncash
    after = (np.abs(w1-reference.current) > reference.rules.zero_tolerance) & noncash
    return dict(half_l1=float(.5*np.abs(w1-w0).sum()),
                support_symmetric_difference=int(np.count_nonzero(before != after)),
                mandatory_added=int(np.count_nonzero(problem.mandatory & ~reference.mandatory)),
                mandatory_removed=int(np.count_nonzero(reference.mandatory & ~problem.mandatory)),
                common_reference_te_bp=tracking_error_bp(reference, w1),
                baseline_reference_te_bp=tracking_error_bp(reference, w0),
                noncash_gross_turnover_change=result['audit']['turnover']-baseline['audit']['turnover'])


def run_sensitivity(bundle, output, spec, limit=None):
    """Run fixed-domain and rebuilt-corridor perturbations on predetermined public cases."""
    manifest = verify_bundle(bundle)
    positions = np.linspace(0, len(manifest['cases'])-1, spec['sensitivity']['case_count'], dtype=int)
    for count, position in enumerate(positions[:limit], 1):
        entry = manifest['cases'][position]
        raw = load_case(bundle, entry)
        reference, preparation = prepare(raw)
        record = dict(id=entry['id'], preparation=preparation, variants=[])
        if reference is not None:
            baseline = solve(reference)
            record['baseline'] = baseline
            for level, (target_size, risk_size) in enumerate(zip(
                    spec['sensitivity']['target_half_l1'], spec['sensitivity']['log_volatility_shock'])):
                for variant, domain in itertools.product(spec['sensitivity']['variants'],
                                                         spec['sensitivity']['domains']):
                    changed = perturb(reference, spec['seed']+int(position), target_size, risk_size, variant)
                    if domain == 'fixed':
                        problem, prep = changed, {'seconds': 0., 'diagnostics': 'fixed baseline preparation'}
                    else:
                        problem, prep = prepare(rebuild_corridor(raw, changed))
                    row = dict(level=level, variant=variant, domain=domain, preparation=prep,
                               target_half_l1=float(.5*np.abs(changed.target-reference.target).sum()))
                    if problem is not None:
                        row['result'] = solve(problem)
                        row['metrics'] = changed_metrics(reference, baseline, problem, row['result'])
                    record['variants'].append(row)
        write_json(output/f"sensitivity_{entry['id']}.json", record)
        print(f'sensitivity {count}/{len(positions[:limit])} {entry["id"]}', flush=True)


def run_scarcity(output, spec, limit=None):
    """Run all five matched arms serially, with explicit incumbent-generation costs."""
    grid = list(itertools.product(*(spec['scarcity'][key] for key in (
        'optional_assets', 'discretionary_allowance', 'group_half_width', 'cash_band', 'seeds'))))
    for index, (optional, budget, width, cash_band, seed) in enumerate(grid[:limit]):
        raw = synthetic_case(optional, budget, width, cash_band, seed)
        problem, prep = prepare(raw)
        record = dict(id=f'scarcity_{index:03}', optional_assets=optional, budget=budget,
                      width=width, cash_band=cash_band, seed=seed, preparation=prep, methods={})
        if problem is not None:
            record['base_audit'] = independent_audit(problem, problem.base)
            record['optional_candidates'] = sum(i != problem.cash and not problem.mandatory[i]
                and any(s.ticket for s in states) for i, states in enumerate(compile_states(problem)))
            ranked = None
            for method in spec['scarcity']['methods']:
                item = solve(problem, method, ranked if method == 'ExactRanked' else None)
                if method == 'Ranked' and item['audit']['passed']:
                    ranked = item['weights']
                item['end_to_end_seconds'] = item['seconds']+prep['seconds']
                if method == 'ExactRanked':
                    item['end_to_end_seconds'] += record['methods']['Ranked']['seconds']
                item['allowance_binding'] = (item['audit'].get('optional_tickets') == budget)
                record['methods'][method] = item
        write_json(output/f'scarcity_{index:03}.json', record)
        print(f'scarcity {index+1}/{len(grid[:limit])}', flush=True)


def run(bundle, output, study, limit=None, protocol_path=PROTOCOL):
    """Freeze configuration and installed execution hashes before the first solve."""
    output = output.resolve()
    if 'onedrive' in str(output).lower():
        raise ValueError('Generated studies must be outside OneDrive')
    output.mkdir(parents=True, exist_ok=False)
    spec = json.loads(protocol_path.read_text(encoding='utf-8'))
    package = Path(optimalportfolios.__file__).resolve().parent
    write_json(output/'protocol.json', dict(specification=spec,
        specification_sha256=sha256(protocol_path),
        benchmark_sha256=sha256(bundle/'manifest.json'), study=study, limit=limit,
        ranked_config=asdict(RankedConfig()), python=platform.python_version(),
        versions={name: importlib.metadata.version(name) for name in
                  ('optimalportfolios', 'qis', 'numpy', 'pandas', 'scipy', 'cvxpy', 'Mosek', 'clarabel')},
        source_sha256={p.relative_to(package).as_posix(): sha256(p)
                       for p in sorted((package/'execution').rglob('*.py'))
                       if 'tests' not in p.parts and 'run_local' not in p.parts},
        runner_sha256=sha256(Path(__file__))))
    if study in ('sensitivity', 'all'):
        run_sensitivity(bundle, output, spec, limit)
    if study in ('scarcity', 'all'):
        run_scarcity(output, spec, limit)
    write_json(output/'complete.json', dict(study=study, full_sample=limit is None))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--study', choices=('sensitivity', 'scarcity', 'all'), default='all')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--protocol', type=Path, default=PROTOCOL)
    args = parser.parse_args()
    run(args.bundle, args.output, args.study, args.limit, args.protocol)
