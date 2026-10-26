"""Run the frozen five-arm ETF decision benchmark; preserve every failed case."""
from __future__ import annotations

import argparse
from dataclasses import asdict, replace
import importlib.metadata
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
import optimalportfolios

from optimalportfolios.execution.v2 import (
    RankedConfig, prepare_common_execution, wrapper_exact_execution,
    wrapper_ranked_execution, compile_states,
)
from .benchmark import verify_bundle, load_case, independent_audit, tracking_error_bp, sha256


def json_value(value):
    """Convert diagnostics without writing non-standard JSON numeric values."""
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)):
        return float(value) if np.isfinite(value) else None
    return value


def write_json(path, value):
    """Publish a completed JSON record atomically."""
    temporary = path.with_suffix('.partial')
    temporary.write_text(json.dumps(json_value(value), indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def run(bundle, output, *, limit=None):
    """Compare identical prepared domains under a predeclared serial protocol."""
    output = output.resolve()
    if not output.is_absolute() or 'onedrive' in str(output).lower():
        raise ValueError('Use an absolute output directory outside OneDrive')
    output.mkdir(parents=True, exist_ok=False)
    manifest = verify_bundle(bundle)
    package = Path(optimalportfolios.__file__).resolve().parent
    protocol = dict(schema=1, benchmark_sha256=sha256(bundle/'manifest.json'),
        exact_seconds=10., exact_threads=1, exact_gap_rel=1e-4,
        preparation_seconds=30., configuration=asdict(RankedConfig()),
        methods=['Ranked', 'Greedy', 'OneAnchor', 'ExactBase', 'ExactRanked'],
        python=platform.python_version(), platform=platform.platform(),
        versions={p: importlib.metadata.version(p) for p in
                  ('optimalportfolios','qis','numpy','pandas','scipy','cvxpy','Mosek','clarabel')},
        execution_source_sha256={p.relative_to(package).as_posix(): sha256(p)
            for p in sorted((package/'execution').rglob('*.py'))
            if 'tests' not in p.parts and 'run_local' not in p.parts},
        timing='Serial; preparation shared and reported separately; start generation added '
               'to ExactRanked end-to-end time. Imports and input reading excluded.',
        sample='All cases in order; limit is a smoke run, never the headline experiment.',
        limit=limit)
    write_json(output/'protocol.json', protocol)
    records = []
    for position, entry in enumerate(manifest['cases'][:limit], 1):
        raw = load_case(bundle, entry)
        started = perf_counter()
        prepared, prep = prepare_common_execution(raw, time_limit_s=30., solver='MOSEK')
        record = dict(id=entry['id'], preparation=prep, methods={},
                      preparation_wall_seconds=perf_counter()-started)
        if prepared is not None:
            record['base_audit'] = independent_audit(prepared, prepared.base)
            assert record['base_audit']['passed'], record
            record['base_te_bp'] = tracking_error_bp(prepared, prepared.base)
            states = compile_states(prepared)
            record['optional_candidates'] = sum(
                i != prepared.cash and not prepared.mandatory[i] and any(s.ticket for s in row)
                for i, row in enumerate(states))
            record['mandatory'] = int(prepared.mandatory.sum())
            record['cap'] = prepared.rules.ticket_cap
            ranked = None
            for method in protocol['methods']:
                tick = perf_counter()
                if method == 'Ranked':
                    solved = wrapper_ranked_execution(prepared, RankedConfig())
                    ranked = solved.weights
                elif method == 'Greedy':
                    solved = wrapper_ranked_execution(prepared, RankedConfig(
                        drop=False, exchange=False, rescue_trials=0))
                elif method == 'OneAnchor':
                    solved = wrapper_ranked_execution(prepared,
                        replace(RankedConfig(), improvement_passes=1))
                else:
                    solved = wrapper_exact_execution(prepared, 'CORRIDOR',
                        mip_start=ranked if method == 'ExactRanked' else None,
                        time_limit_s=10., threads=1, gap_rel=1e-4)
                seconds = perf_counter()-tick
                audit = independent_audit(prepared, solved.weights)
                item = dict(status=solved.status, seconds=seconds, audit=audit,
                            weights=solved.weights, diagnostics=solved.diagnostics)
                item['end_to_end_seconds'] = seconds+record['preparation_wall_seconds']
                if method == 'ExactRanked':
                    item['end_to_end_seconds'] += record['methods']['Ranked']['seconds']
                if solved.weights is not None:
                    assert audit['passed'], (entry['id'], method, audit)
                    item['te_bp'] = tracking_error_bp(prepared, solved.weights)
                record['methods'][method] = item
        write_json(output/(entry['id']+'.json'), record)
        records.append(record)
        write_json(output/'progress.json', dict(completed=position,
            total=len(manifest['cases'][:limit]), last=entry['id'],
            prepared=prepared is not None))
        print(entry['id'], 'prepared' if prepared is not None else 'preparation_failed',
              flush=True)
    write_json(output/'complete.json', dict(cases=len(records), full_sample=limit is None))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    run(args.bundle, args.output, limit=args.limit)
