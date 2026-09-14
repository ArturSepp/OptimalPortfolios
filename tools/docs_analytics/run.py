"""Inspect, generate and validate C-local documentation analytics.

--all generates and validates a complete bundle once every producer is implemented.
Plans are never publishable bundles; successful generation still needs visual review.
Boundary and fingerprint ideas adapt QuantInvestStrats/tools/docs_analytics/run.py (MIT).
"""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
from importlib import metadata
import json
from pathlib import Path
import platform
import sys
import types
import uuid

from tools.docs_analytics.registry import ROOT, load_registry, source_file
from tools.docs_analytics.validate import (
    MANIFEST, check_result, dependencies_for, digest, environment_identity, environment_snapshot,
    expected_files, file_record, generation_blockers, input_files, offline, output_boundary,
    producer_contract, rendering_record, source_fingerprint, validate_bundle, write_json,
)


DISTRIBUTIONS = (
    'optimalportfolios', 'qis', 'factorlasso', 'numpy', 'pandas', 'scipy',
    'matplotlib', 'cvxpy', 'clarabel', 'quadprog',
)


def build_plan(root: Path = ROOT) -> dict:
    """Create a reproducible plan without importing producers or numerical libraries.

    Args:
        root: Repository checkout or source export whose current images are inventoried.

    Returns:
        Planning record with coverage, effective source and legacy-image identities.
        It is not an execution manifest and does not certify image reproducibility.
    """
    registry = load_registry(root)
    versions = {}
    for name in DISTRIBUTIONS:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    legacy = {asset['path']: {'sha256': digest(source_file(root, asset['path'])),
                              'bytes': source_file(root, asset['path']).stat().st_size}
              for asset in registry['assets']}
    blockers = generation_blockers(registry, root)
    return {
        'schema_version': 1,
        'kind': 'documentation_analytics_plan',
        'status': 'planned',
        'generation_ready': not blockers,
        'publication_ready': False,
        'generation_blockers': blockers,
        'registry': registry,
        'source': source_fingerprint(root),
        'legacy_images': legacy,
        'environment': {
            'python': platform.python_version(),
            'platform': platform.system(),
            'installed_distributions': versions,
            'identity': 'Distribution metadata only; no analytical package was imported.',
        },
        'notes': [
            'Legacy hashes identify current files, not their generation data or source.',
            'Fixture candidates are proposals; configuration=None means not yet frozen.',
            'No prices are downloaded, analytics executed, images generated or files published.',
            'Execution records actual inputs, dependencies, conventions and solver diagnostics.',
        ],
    }


def write_plan(plan: dict, output: Path, root: Path = ROOT) -> Path:
    """Write a plan into a new permitted directory without overwriting prior output."""
    output = output_boundary(output, root)
    output.mkdir(parents=True, exist_ok=False)
    path = output / 'run_plan.json'
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(plan, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    return path


def _produce(name: str, spec: dict, root: Path) -> dict:
    """Execute the verified local module bytes, avoiding stale producer bytecode."""
    module_name = spec['module']
    path = source_file(root, f'tools/docs_analytics/{name}.py')
    module = types.ModuleType(module_name)
    module.__file__ = str(path)
    module.__package__ = 'tools.docs_analytics'
    previous = sys.modules.get(module_name)
    sys.modules[module_name] = module
    try:
        exec(compile(path.read_bytes(), str(path), 'exec'), module.__dict__)
        result = getattr(module, spec['callable'])(deepcopy(spec))
        if not isinstance(result, dict):
            raise ValueError(f'{name}: producer must return a dictionary')
        return result
    finally:
        if previous is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous


def _save_outputs(name: str, result: dict, registry: dict, staging: Path) -> dict:
    """Require exact figure/table sets, save lossless CSV snapshots and check diagnostics."""
    from matplotlib.figure import Figure
    import pandas as pd

    config = registry['producers'][name]['configuration']
    figures = result.pop('figures', None)
    tables = result.pop('tables', None)
    wanted = {Path(asset['path']).name for asset in registry['assets']
              if asset['producer'] == name}
    if not isinstance(figures, dict) or set(figures) != wanted:
        raise ValueError(f'{name}: incomplete or unexpected figures')
    if not isinstance(tables, dict) or set(tables) != set(config['tables']):
        raise ValueError(f'{name}: incomplete or unexpected tables')
    check_result(result, config)
    for filename, figure in figures.items():
        if not isinstance(figure, Figure):
            raise ValueError(f'{name}: expected a matplotlib Figure')
        figure.savefig(staging / 'images' / filename, dpi=config['rendering']['dpi'],
                       format='png', metadata={'Software': 'OptimalPortfolios docs analytics'})
    directory = staging / 'tables' / name
    directory.mkdir(parents=True)
    for label, table in tables.items():
        if not isinstance(table, pd.DataFrame) or table.empty:
            raise ValueError(f'{name}: expected a nonempty DataFrame')
        headers = list(table.index.names) + list(table.columns)
        if (any(not isinstance(value, str) or not value for value in headers)
                or len(headers) != len(set(headers))):
            raise ValueError(f'{name}: index and column labels must be unique nonempty strings')
        table.to_csv(directory / f'{label}.csv', float_format='%.17g', lineterminator='\n')
    return result


def generate(output: Path, root: Path = ROOT) -> Path:
    """Generate all registered producers and finalize only after complete bundle validation.

    Args:
        output: New directory below AGENT_LOCAL_ROOT, separate from the source tree.
        root: Source checkout/export supplying the registry and producer modules.

    Returns:
        Completed C-local bundle directory. Its figures still require visual review.

    Raises:
        ValueError: If preflight, producer contracts, output or provenance checks fail.
        RuntimeError: If a producer attempts network access or execution fails.
    """
    registry = load_registry(root)
    blockers = generation_blockers(registry, root)
    if blockers:
        raise ValueError('Generation unavailable: ' + '; '.join(blockers))
    dependencies = dependencies_for(registry)
    output = output_boundary(output, root)
    source = source_fingerprint(root)
    inputs = input_files(registry, root)
    staging = output.with_name(f'.{output.name}-building-{uuid.uuid4().hex}')
    output_boundary(staging, root)
    staging.mkdir(parents=True, exist_ok=False)
    (staging / 'images').mkdir()
    try:
        with offline():
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            environment = environment_snapshot(dependencies)
            if 'optimalportfolios' in environment['libraries']:
                imported = Path(environment['libraries']['optimalportfolios']['module_path'])
                if imported != (root / 'src/optimalportfolios/__init__.py').resolve():
                    raise ValueError('Import optimalportfolios from the selected source export')
            results, rendering = {}, {}
            for name, spec in registry['producers'].items():
                config = producer_contract(spec, root)
                rendering[name] = rendering_record(config['rendering'])
                old_figures = set(plt.get_fignums())
                try:
                    with matplotlib.rc_context({
                            'font.family': config['rendering']['font_family'],
                            'figure.dpi': config['rendering']['dpi'],
                            'savefig.dpi': config['rendering']['dpi']}):
                        result = _produce(name, spec, root)
                        results[name] = _save_outputs(name, result, registry, staging)
                finally:
                    for number in set(plt.get_fignums()) - old_figures:
                        plt.close(number)
            if source_fingerprint(root) != source or load_registry(root) != registry:
                raise ValueError('Source changed during generation')
            if input_files(registry, root) != inputs:
                raise ValueError('Fixture input files changed during generation')
            after = environment_snapshot(dependencies)
            if environment_identity(after) != environment_identity(environment):
                raise ValueError('Imported dependency source changed during generation')
            outputs = {name: file_record(staging / name)
                       for name in sorted(expected_files(registry))}
            input_names = {f'tables/{name}/{label}.csv'
                           for name, spec in registry['producers'].items()
                           for label in spec['configuration']['input_tables']}
            record = {
                'schema_version': 1, 'kind': 'documentation_analytics_bundle',
                'status': 'complete', 'review_status': 'pending', 'publication_ready': False,
                'generated_at_utc': datetime.now(timezone.utc).isoformat(),
                'registry': registry, 'source': source, 'input_files': inputs,
                'input_tables': {name: outputs[name] for name in sorted(input_names)},
                'environment': environment, 'rendering': rendering,
                'producers': results, 'outputs': outputs,
            }
            write_json(staging / MANIFEST, record)
            validate_bundle(staging, root)
        output_boundary(output, root)
        staging.rename(output)
    except BaseException as error:
        # A failed staging tree must never retain a completion marker.
        manifest = staging / MANIFEST
        if manifest.is_file():
            manifest.unlink()
        if staging.is_dir():
            write_json(staging / 'FAILED.json',
                       {'status': 'failed', 'error': f'{type(error).__name__}: {error}'})
        raise
    return output


def main(argv: list[str] | None = None) -> int:
    """List coverage, inspect/write a plan, or report why generation is unavailable.

    Args:
        argv: Command-line arguments without the program name.

    Returns:
        Zero for successful inspection/planning; two for invalid input or unavailable generation.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument('--list', action='store_true', help='Check and list all preview coverage.')
    actions.add_argument('--plan', action='store_true', help='Print or save a refresh plan.')
    actions.add_argument('--all', action='store_true',
                        help='Generate and validate every implemented producer.')
    parser.add_argument(
        '--output-root', type=Path, help='New C-local task directory for a plan/run.')
    args = parser.parse_args(argv)
    if args.list and args.output_root:
        parser.error('--output-root applies to --plan or --all')
    if args.all and not args.output_root:
        parser.error('--all requires --output-root')
    try:
        if args.list:
            registry = load_registry()
            for asset in registry['assets']:
                producer = asset['producer']
                print(f'{asset["id"]}: {producer} [{registry["producers"][producer]["status"]}]'
                      f' -> {asset["path"]}')
            print(f'{len(registry["assets"])} legacy previews; '
                  f'{len(registry["producers"])} producer families; '
                  f'{len(registry["non_analytics"])} non-analytics images.')
            return 0
        plan = build_plan()
        if args.all:
            if plan['generation_blockers']:
                print('Generation unavailable:\n- ' + '\n- '.join(plan['generation_blockers']),
                      file=sys.stderr)
                return 2
            print(f'Bundle generated and validated: {generate(args.output_root)}')
            print('Visual review remains pending; no files have been published.')
            return 0
        if args.output_root:
            print(f'Plan written: {write_plan(plan, args.output_root)}')
        else:
            print(json.dumps(plan, indent=2, sort_keys=True, allow_nan=False))
        return 0
    except (ValueError, OSError, KeyError, TypeError, RuntimeError,
            ImportError) as error:
        print(f'Analytics run failed: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
