"""Compare the frozen V2 heuristic with the complete matched mixed-integer model."""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
from time import perf_counter

import cvxpy as cp
import numpy as np
import optimalportfolios
from optimalportfolios.execution.v2 import (
    RankedConfig, compile_states, compute_perspective_lower_bound,
    prepare_common_execution, wrapper_exact_execution, wrapper_ranked_execution,
)
from .benchmark import ARRAYS, independent_audit, sha256, tracking_error_bp
from .breadth import case_grid, enumerate_tiny, fingerprint, generated_case
from .run_benchmark import write_json

PROTOCOL = Path(__file__).with_name('joint_protocol.json')


def methods(spec):
    """Name all separately solved budget/start combinations in stable report order."""
    return ['Ranked'] + [f'Joint{start}{seconds:g}' for seconds in spec['exact_seconds']
                         for start in ('Base', 'Ranked')]


def serial_arms(problem, spec, index=0, preparation_seconds=0.):
    """Keep objective/domain fixed and avoid incumbent transfers between native calls."""
    records = {}
    budgets = list(spec['exact_seconds'])
    if index % 2:
        budgets.reverse()
    order = [('Ranked', None, None)]
    for offset, seconds in enumerate(budgets):
        starts = ['Base', 'Ranked']
        if (index + offset) % 2:
            starts.reverse()
        order.extend((f'Joint{start}{seconds:g}', seconds, start) for start in starts)
    identity = fingerprint(problem)
    for name, seconds, start_kind in order:
        source = records.get('Ranked', {}) if start_kind == 'Ranked' else {}
        start = source.get('weights') if source.get('audit', {}).get('passed') else None
        generation = source.get('seconds', 0.) if start_kind == 'Ranked' else 0.
        tick = perf_counter()
        try:
            if name == 'Ranked':
                solved = wrapper_ranked_execution(problem, RankedConfig())
            else:
                solved = wrapper_exact_execution(problem, 'CORRIDOR',
                    mip_start=None if start is None else np.asarray(start).copy(),
                    time_limit_s=seconds, threads=spec['exact_threads'],
                    gap_rel=spec['exact_gap_rel'])
            item = dict(status=solved.status, weights=solved.weights,
                        diagnostics=solved.diagnostics, seconds=perf_counter()-tick,
                        audit=independent_audit(problem, solved.weights))
            if item['audit']['passed']:
                item['te_bp'] = tracking_error_bp(problem, solved.weights)
        except (ValueError, RuntimeError, cp.SolverError) as error:
            item = dict(status='exception', error=repr(error), weights=None,
                        seconds=perf_counter()-tick, audit=independent_audit(problem, None))
        item.update(time_limit_seconds=seconds,
                    requested_start='Ranked' if start_kind == 'Ranked' else 'common_base',
                    supplied_start='Ranked' if start is not None else 'common_base',
                    start_generation_seconds=generation)
        item['end_to_end_seconds'] = item['seconds'] + preparation_seconds + generation
        if fingerprint(problem) != identity:
            raise ValueError('An arm mutated the shared prepared problem')
        records[name] = item
    return records, [name for name, _, _ in order]


def run(output, protocol_path=PROTOCOL, limit=None):
    """Freeze every input and numerical source identity before the first primary solve."""
    output = output.resolve()
    if 'onedrive' in str(output).lower():
        raise ValueError('Generated studies must be outside OneDrive')
    output.mkdir(parents=True, exist_ok=False)
    spec = json.loads(protocol_path.read_text())
    if spec['exact_seconds'] != [10., 60., 600.] or spec['exact_threads'] != 1:
        raise ValueError('Use the declared three-budget single-thread protocol')
    inputs = output/'inputs'
    inputs.mkdir()
    declared = []
    for position, case in enumerate(case_grid(spec)):
        case = dict(case, id=f'joint_{position:03}')
        problem = generated_case(case, spec)
        path = inputs/(case['id']+'.npz')
        np.savez_compressed(path, **{name: getattr(problem, name) for name in ARRAYS})
        declared.append(dict(case, input_sha256=sha256(path), fingerprint=fingerprint(problem)))
    package = Path(optimalportfolios.__file__).resolve().parent
    write_json(output/'protocol.json', dict(specification=spec, cases=declared,
        methods=methods(spec), frozen_utc=datetime.now(timezone.utc).isoformat(),
        specification_sha256=sha256(protocol_path), ranked_config=asdict(RankedConfig()),
        python=platform.python_version(), versions={name: importlib.metadata.version(name)
            for name in ('optimalportfolios','qis','numpy','pandas','scipy','cvxpy','Mosek','clarabel')},
        source_sha256={p.relative_to(package).as_posix():sha256(p)
            for p in sorted((package/'execution').rglob('*.py'))
            if 'tests' not in p.parts and 'run_local' not in p.parts},
        replication_sha256={p.name:sha256(p) for p in
            (Path(__file__), Path(__file__).with_name('breadth.py'),
             Path(__file__).with_name('benchmark.py'))}))
    for index, case in enumerate(declared[:limit]):
        raw = generated_case(case, spec)
        if fingerprint(raw) != case['fingerprint']:
            raise ValueError('Generated input changed after freeze')
        tick = perf_counter()
        problem, prep = prepare_common_execution(raw,
            time_limit_s=spec['preparation_seconds'], solver='MOSEK')
        elapsed = perf_counter()-tick
        record = dict(case, preparation=prep, preparation_seconds=elapsed, methods={})
        if problem is not None:
            record['base_audit'] = independent_audit(problem, problem.base)
            if not record['base_audit']['passed']:
                raise ValueError('Prepared incumbent failed independent audit')
            record['prepared_fingerprint'] = fingerprint(problem)
            record['optional_candidates'] = sum(any(s.ticket for s in row)
                for i, row in enumerate(compile_states(problem))
                if i != problem.cash and not problem.mandatory[i])
            record['methods'], record['method_order'] = serial_arms(problem, spec, index, elapsed)
            record['relaxations'] = compute_perspective_lower_bound(problem)
            if case['sample'] == 'tiny':
                record['enumeration'] = enumerate_tiny(problem)
        write_json(output/(case['id']+'.json'), record)
        write_json(output/'progress.json', dict(completed=index+1,
            total=len(declared[:limit]), last=case['id']))
        print(case['id'], 'complete', flush=True)
    write_json(output/'complete.json', dict(cases=len(declared[:limit]), full_sample=limit is None))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--protocol', type=Path, default=PROTOCOL)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    run(args.output, args.protocol, args.limit)
