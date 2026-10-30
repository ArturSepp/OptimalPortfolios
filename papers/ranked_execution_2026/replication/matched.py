"""Frozen public comparisons of released MIQP, V2 and the audited primary policy."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from datetime import datetime, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import patch

import cvxpy as cp
import numpy as np
import optimalportfolios
from optimalportfolios.execution.v2 import (
    MatchedExecutionPolicyConfig, RankedConfig, compile_states,
    prepare_common_execution, wrapper_exact_execution, wrapper_matched_execution,
    wrapper_ranked_execution,
)
from .benchmark import ARRAYS, independent_audit, load_case, sha256, tracking_error_bp, verify_bundle
from .breadth import case_grid, enumerate_tiny, fingerprint, generated_case
from .run_benchmark import write_json

PROTOCOL = Path(__file__).with_name('matched_protocol.json')


def public_sources():
    """Identify installed numerical code and public runners without personal paths."""
    package = Path(optimalportfolios.__file__).resolve().parent
    return dict(execution={p.relative_to(package).as_posix(): sha256(p)
        for p in sorted((package/'execution').rglob('*.py'))
        if 'tests' not in p.parts and 'run_local' not in p.parts},
        replication={name: sha256(Path(__file__).with_name(name)) for name in
                     ('matched.py', 'benchmark.py', 'breadth.py', 'run_benchmark.py')})


def holdout_grid(spec, legacy):
    """Declare fresh seeds and target variants before looking at any solver output."""
    design = dict(legacy, grid=spec['holdout']['grid'], tiny=spec['holdout']['tiny'])
    cases = []
    for case in case_grid(design):
        patterns = ['diffuse'] if case['sample'] == 'tiny' else spec['holdout']['target_patterns']
        for pattern in patterns:
            cases.append(dict(case, target_pattern=pattern, id=f'holdout_{len(cases):03}'))
    return design, cases


def holdout_case(case, design):
    """Vary supplied targets while retaining every hard row and execution threshold."""
    problem = generated_case(case, design)
    if case['target_pattern'] == 'diffuse':
        return problem
    optional = np.flatnonzero(~problem.mandatory & (np.arange(len(problem.names)) != problem.cash))
    rng = np.random.default_rng(np.random.SeedSequence([case['seed'], len(optional), 920]))
    target = problem.target.copy()
    mass = float(target[optional].sum())
    target[rng.choice(optional, len(optional)//2, replace=False)] = 0.
    target[optional] *= mass/target[optional].sum()
    lower, upper = problem.lower.copy(), problem.upper.copy()
    lower[optional] = np.maximum(problem.full_lower[optional], np.minimum(problem.base[optional], target[optional]))
    upper[optional] = np.minimum(problem.full_upper[optional], np.maximum(problem.base[optional], target[optional]))
    return replace(problem, target=target, lower=lower, upper=upper)


def serialize_case(root, problem, description):
    """Write complete typed public inputs in the existing portable bundle format."""
    name = description['id']+'.npz'
    rules = asdict(problem.rules)
    arrays = {key: getattr(problem, key) for key in ARRAYS}
    for key in ('holding_floor', 'exemption_mask'):
        arrays[key] = rules.pop(key)
    np.savez_compressed(root/name, **arrays)
    return dict(description, array_file=name, names=problem.names, cash=problem.cash,
                categories=problem.categories, rules=rules, fingerprint=fingerprint(problem))


def freeze(bundle, output, wheel, protocol=PROTOCOL):
    """Freeze all inputs and source identities before any preparation or solver call."""
    output = output.resolve()
    if 'onedrive' in str(output).lower():
        raise ValueError('Use a fresh C-local output directory')
    spec = json.loads(protocol.read_text())
    if importlib.metadata.version('optimalportfolios') != spec['package_version']:
        raise ValueError('Run against released OP 9.2.0')
    if sha256(wheel) != spec['wheel_sha256']:
        raise ValueError('Published wheel hash mismatch')
    # An installed version label alone cannot exclude a patched or editable package.
    import zipfile
    package = Path(optimalportfolios.__file__).resolve().parent
    with zipfile.ZipFile(wheel) as archive:
        import hashlib
        for name in archive.namelist():
            if name.startswith('optimalportfolios/') and name.endswith('.py'):
                installed = package/Path(name).relative_to('optimalportfolios')
                if hashlib.sha256(archive.read(name)).hexdigest() != sha256(installed):
                    raise ValueError('Installed source differs from published wheel')
    original = verify_bundle(bundle)
    if sha256(bundle/'manifest.json') != spec['etf_manifest_sha256']:
        raise ValueError('The approved ETF input manifest changed')
    output.mkdir(parents=True, exist_ok=False)
    inputs = output/'inputs'
    inputs.mkdir()
    cases = []
    for row in original['cases']:
        raw = load_case(bundle, row)
        if raw.rules.audit_tolerance != spec['audit_tolerance'] or raw.rules.zero_tolerance != spec['zero_tolerance']:
            raise ValueError('ETF audit thresholds differ from the declared protocol')
        shutil.copy2(bundle/row['array_file'], inputs/row['array_file'])
        cases.append(dict(row, cohort='ETF', sample='empirical', fingerprint=fingerprint(raw)))
    legacy = json.loads(Path(__file__).with_name('joint_protocol.json').read_text())
    for index, row in enumerate(case_grid(legacy)):
        cases.append(serialize_case(inputs, generated_case(row, legacy),
                                    dict(row, id=f'legacy_{index:03}', cohort='Legacy')))
    design, holdout = holdout_grid(spec, legacy)
    for row in holdout:
        cases.append(serialize_case(inputs, holdout_case(row, design), dict(row, cohort='Holdout')))
    files = {row['array_file']: sha256(inputs/row['array_file']) for row in cases}
    write_json(inputs/'manifest.json', dict(schema='ranked-execution-benchmark-1', cases=cases, files=files))
    write_json(output/'protocol.json', dict(specification=spec, legacy_design=legacy,
        holdout_design=design, cases=[{key: row[key] for key in
          ('id', 'cohort', 'sample', 'fingerprint')} for row in cases],
        input_manifest_sha256=sha256(inputs/'manifest.json'),
        specification_sha256=sha256(protocol), sources=public_sources(),
        frozen_utc=datetime.now(timezone.utc).isoformat(),
        python=platform.python_version(), platform=platform.platform(),
        versions={name: importlib.metadata.version(name) for name in
            ('optimalportfolios','qis','numpy','pandas','scipy','cvxpy','Mosek','clarabel')},
        thread_environment={name: os.environ.get(name) for name in
            ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS')},
        installed_distributions=sorted((d.metadata['Name'], d.version) for d in importlib.metadata.distributions()),
        ranked_config=asdict(RankedConfig())))
    print('FROZEN', len(cases), 'cases', flush=True)


def method_names(spec, cohort):
    """Keep historical budget controls as secondary arms on the existing grid."""
    names = list(spec['methods'])
    if cohort == 'Legacy':
        names += [f'{start}{seconds:g}' for seconds in spec['legacy_extra_seconds']
                  for start in ('MIQP', 'MIQPStartV2')]
    return names


def policy_config(spec=None):
    """Supply a neutral class declaration; prepared problems own all hard rules."""
    spec = spec or json.loads(PROTOCOL.read_text())
    return MatchedExecutionPolicyConfig(tre_weight_by_asset_class={'All': 1.},
        primary_time_limit_s=spec['primary_seconds'], primary_threads=spec['threads'],
        primary_gap_rel=spec['gap_rel'])


def serial_arms(problem, spec, cohort, index, preparation_seconds):
    """Call real published entry points without cross-arm incumbent contamination."""
    primary = ['V2', 'MIQP', 'Policy']
    shift = index % len(primary)
    order = primary[shift:]+primary[:shift]+['MIQPStartV2']
    extra = spec['legacy_extra_seconds'][::(-1 if index % 2 else 1)]
    if cohort == 'Legacy':
        order += [f'{start}{seconds:g}' for seconds in extra for start in ('MIQP', 'MIQPStartV2')]
    identity, records = fingerprint(problem), {}
    for method in order:
        started_from_v2 = method.startswith('MIQPStartV2')
        source = records.get('V2', {}) if started_from_v2 else {}
        incumbent = source.get('weights') if source.get('audit', {}).get('passed') else None
        generation = source.get('seconds', 0.) if started_from_v2 else 0.
        suffix = method.removeprefix('MIQPStartV2' if started_from_v2 else 'MIQP')
        seconds = float(suffix) if suffix and method.startswith('MIQP') else spec['primary_seconds']
        tick = perf_counter()
        try:
            if method == 'V2':
                solved = wrapper_ranked_execution(problem, RankedConfig())
            elif method == 'Policy':
                solved = wrapper_matched_execution(problem, policy_config(spec))
            else:
                solved = wrapper_exact_execution(problem, 'CORRIDOR', time_limit_s=seconds,
                    threads=spec['threads'], gap_rel=spec['gap_rel'],
                    mip_start=None if incumbent is None else np.asarray(incumbent).copy())
            item = dict(status=solved.status, weights=solved.weights, diagnostics=solved.diagnostics,
                        seconds=perf_counter()-tick, audit=independent_audit(problem, solved.weights))
            if item['audit']['passed']:
                item['te_bp'] = tracking_error_bp(problem, solved.weights)
        except (ValueError, RuntimeError, cp.SolverError) as error:
            item = dict(status='exception', error=repr(error), weights=None,
                        seconds=perf_counter()-tick, audit=independent_audit(problem, None))
        item.update(start_generation_seconds=generation,
                    supplied_start='V2' if incumbent is not None else 'common_base',
                    allowance_seconds=seconds if method != 'V2' else None)
        item['end_to_end_seconds'] = item['seconds']+preparation_seconds+generation
        if fingerprint(problem) != identity:
            raise ValueError('An arm mutated the shared prepared domain')
        records[method] = item
    return records, order


def load_frozen(output):
    """Fail closed on changed numerical source, protocol or input bytes."""
    protocol = json.loads((output/'protocol.json').read_text())
    if protocol['sources'] != public_sources():
        raise ValueError('Frozen numerical source changed')
    if sha256(output/'inputs/manifest.json') != protocol['input_manifest_sha256']:
        raise ValueError('Frozen input manifest changed')
    manifest = verify_bundle(output/'inputs')
    identities = [{key: row[key] for key in ('id','cohort','sample','fingerprint')} for row in manifest['cases']]
    if identities != protocol['cases']:
        raise ValueError('Frozen case identities changed')
    return protocol, manifest


def solve(output):
    """Resume only completed, fingerprinted cases and retain all failed attempts."""
    protocol, manifest = load_frozen(output)
    spec = protocol['specification']
    for index, entry in enumerate(manifest['cases']):
        destination = output/(entry['id']+'.json')
        if destination.exists():
            saved = json.loads(destination.read_text())
            if saved['fingerprint'] != entry['fingerprint']:
                raise ValueError('Saved case has another input identity')
            continue
        raw = load_case(output/'inputs', entry)
        if fingerprint(raw) != entry['fingerprint']:
            raise ValueError('Serialized case differs from its frozen input')
        tick = perf_counter()
        try:
            problem, prep = prepare_common_execution(raw, time_limit_s=spec['preparation_seconds'], solver='MOSEK')
        except (ValueError, RuntimeError, cp.SolverError) as error:
            problem, prep = None, dict(status='exception', error=repr(error))
        elapsed = perf_counter()-tick
        record = dict(entry, preparation=prep, preparation_seconds=elapsed, methods={})
        if problem is not None:
            record['base_audit'] = independent_audit(problem, problem.base)
            if not record['base_audit']['passed']:
                raise ValueError('Prepared base failed independent audit')
            record['prepared_fingerprint'] = fingerprint(problem)
            record['optional_candidates'] = sum(any(s.ticket for s in row)
                for i, row in enumerate(compile_states(problem)) if i != problem.cash and not problem.mandatory[i])
            record['methods'], record['method_order'] = serial_arms(problem, spec, entry['cohort'], index, elapsed)
            if entry['sample'] == 'tiny':
                record['enumeration'] = enumerate_tiny(problem)
        write_json(destination, record)
        write_json(output/'progress.json', dict(completed=index+1, total=len(manifest['cases']), last=entry['id']))
        print(entry['id'], 'complete', flush=True)
    write_json(output/'complete.json', dict(cases=len(manifest['cases']), full_sample=True))


def injected_primary(problem, fault):
    """Produce an explicitly labelled primary failure without changing any domain."""
    if fault == 'native_error':
        raise cp.SolverError('injected native failure after common preparation')
    weights = problem.base.copy()
    diagnostics = dict(native_status='test', incumbent_events=[dict(source='native_test', audit={'passed':True})])
    if fault == 'invalid_weights':
        weights[problem.cash] += .01
    elif fault == 'inconsistent_bound':
        diagnostics['native_bound_check'] = {'consistent_with_delivery':False}
    else:
        raise ValueError('Unknown injected failure')
    return SimpleNamespace(weights=weights, diagnostics=diagnostics, status='injected')


def fault_controls(output):
    """Run genuine V2 fallback after timed studies, with primary faults clearly separate."""
    protocol, manifest = load_frozen(output)
    cases = [next(row for row in manifest['cases'] if row['cohort'] == cohort)
             for cohort in ('ETF','Legacy','Holdout')]
    records = []
    for entry in cases:
        problem, _ = prepare_common_execution(load_case(output/'inputs', entry),
            time_limit_s=protocol['specification']['preparation_seconds'], solver='MOSEK')
        for fault in protocol['specification']['faults']:
            tick, identity = perf_counter(), fingerprint(problem)
            def failed_primary(p, *args, **kwargs):
                """Inject only into this local process and restore at context exit."""
                return injected_primary(p, fault)
            try:
                with patch('optimalportfolios.execution.v2.primary.wrapper_exact_execution', failed_primary):
                    result = wrapper_matched_execution(problem, policy_config(protocol['specification']))
                audit = independent_audit(problem, result.weights)
                item = dict(case=entry['id'], fault=fault, audit=audit,
                    engine=result.diagnostics['engine'], reason=result.diagnostics['fallback_reason'],
                    weights=result.weights, te_bp=tracking_error_bp(problem, result.weights))
            except (ValueError, RuntimeError, cp.SolverError) as error:
                item = dict(case=entry['id'], fault=fault, audit={'passed':False}, error=repr(error))
            if fingerprint(problem) != identity:
                raise ValueError('Fault control mutated constraints')
            records.append(dict(item, seconds=perf_counter()-tick))
    write_json(output/'fault_controls.json', dict(injected=True, records=records))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['freeze','solve','faults'])
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--bundle', type=Path)
    parser.add_argument('--wheel', type=Path)
    args = parser.parse_args()
    if args.action == 'freeze':
        freeze(args.bundle, args.output, args.wheel)
    elif args.action == 'solve':
        solve(args.output)
    else:
        fault_controls(args.output)
