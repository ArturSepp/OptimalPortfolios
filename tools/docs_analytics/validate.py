"""Validate complete C-local documentation analytics bundles before review/publication.

This integrity contract adapts the MIT-licensed qis documentation pipeline locally.
It does not recompute every statistic or authenticate deliberately rewritten provenance.
"""

import argparse
from contextlib import contextmanager
import csv
from datetime import date, datetime, timezone
import hashlib
import importlib
from importlib import metadata
import json
import os
from pathlib import Path
import math
import platform
import re
import socket
import sys

from tools.docs_analytics.registry import ROOT, load_registry, relative_path, source_file


CONVENTIONS = {
    'data_kind', 'sample_start', 'sample_end', 'seed', 'universe', 'missing_data', 'returns',
    'observation_frequency', 'estimation_frequency', 'annualization', 'warmup',
    'rebalance_frequency', 'implementation_lag', 'transaction_costs', 'sharpe_convention',
}
BASE_DEPENDENCIES = {'matplotlib': 'matplotlib', 'pandas': 'pandas', 'numpy': 'numpy',
                     'PIL': 'Pillow'}
MANIFEST = 'analytics_manifest.json'


def digest(path: Path) -> str:
    """Return a file's SHA-256 content digest."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_fingerprint(root: Path) -> dict:
    """Identify effective source content, including dirty edits, without relying on Git."""
    paths = set()
    for directory in ('src/optimalportfolios', 'examples', 'tools/docs_analytics'):
        paths.update((root / directory).rglob('*.py'))
    paths.add(source_file(root, 'tools/docs_analytics/registry.json'))
    paths.add(source_file(root, 'pyproject.toml'))
    for name in ('uv.lock', 'tools/docs_inventory.json', 'AGENTS.md'):
        if (root / name).is_file():
            paths.add(source_file(root, name))
    files = {
        path.relative_to(root).as_posix(): digest(
            source_file(root, path.relative_to(root).as_posix()))
        for path in sorted(paths) if path.is_file()
    }
    fingerprint = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()
    return {'sha256': fingerprint, 'files': files,
            'identity': 'Effective source bytes; includes uncommitted source-export edits.'}


def output_boundary(output: Path, root: Path, *, new: bool = True) -> Path:
    """Require a new task directory under local storage, outside OneDrive and source trees."""
    setting = os.environ.get('AGENT_LOCAL_ROOT')
    if not setting:
        raise ValueError('Set AGENT_LOCAL_ROOT through the repository setup first')
    local = Path(setting)
    if not local.is_absolute() or not output.is_absolute():
        raise ValueError('AGENT_LOCAL_ROOT and output-root must be absolute paths')
    for path in (local, output):
        for parent in (path, *path.parents):
            if parent.is_symlink() or (hasattr(parent, 'is_junction') and parent.is_junction()):
                raise ValueError(f'Linked output path is not allowed: {parent}')
        if any(part.lower().startswith('onedrive') for part in path.parts):
            raise ValueError('Generated output must be outside OneDrive')
    local, output, root = local.resolve(), output.resolve(), root.resolve()
    if os.name == 'nt' and local.drive.upper() != 'C:':
        raise ValueError('AGENT_LOCAL_ROOT must be on C:')
    if output == local or not output.is_relative_to(local):
        raise ValueError('Output must be a task directory below AGENT_LOCAL_ROOT')
    if output.is_relative_to(root) or root.is_relative_to(output):
        raise ValueError('Output must be outside and separate from the source tree')
    if new and output.exists():
        raise ValueError(f'Refusing to overwrite existing output: {output}')
    if not new and not output.is_dir():
        raise ValueError(f'Bundle directory does not exist: {output}')
    return output


def json_safe(value: object) -> None:
    """Reject non-JSON values and NaN/infinite numbers in configurations and records."""
    json.dumps(value, allow_nan=False)


def _bad_constant(value: str) -> None:
    """Reject JSON's nonstandard NaN and Infinity tokens."""
    raise ValueError(f'Non-finite JSON constant: {value}')


def _unique_pairs(pairs: list) -> dict:
    """Reject duplicate keys in a stored execution record."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate JSON key: {key}')
        result[key] = value
    return result


def read_json(path: Path) -> dict:
    """Read a strict JSON object, never silently replacing duplicate fields."""
    result = json.loads(path.read_text(encoding='utf-8'), object_pairs_hook=_unique_pairs,
                        parse_constant=_bad_constant)
    if not isinstance(result, dict):
        raise ValueError(f'Expected a JSON object: {path}')
    return result


def write_json(path: Path, value: dict) -> None:
    """Write a new JSON record without replacing any existing file."""
    json_safe(value)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def _names(value: object, field: str) -> None:
    """Require a nonempty unique list of portable table/check identifiers."""
    if (not isinstance(value, list) or not value
            or any(not isinstance(item, str) or not re.fullmatch(r'[a-z][a-z0-9_]*', item)
                   for item in value) or len(set(value)) != len(value)):
        raise ValueError(f'Invalid {field}')


def producer_contract(spec: dict, root: Path) -> dict:
    """Require explicit data, numerical, rendering and dependency conventions before execution."""
    config = spec['configuration']
    required = {'fixture', 'parameters', 'conventions', 'tables', 'input_tables',
                'checks', 'rendering', 'solver', 'dependencies'}
    if not isinstance(config, dict) or set(config) != required:
        raise ValueError(f'Configuration requires exactly {sorted(required)}')
    json_safe(config)
    fixture = config['fixture']
    if (not isinstance(fixture, dict) or set(fixture) != {'factory', 'source_files'}
            or not isinstance(fixture['factory'], str) or not fixture['factory'].strip()
            or not isinstance(fixture['source_files'], list)):
        raise ValueError('Fixture requires a factory description and source_files list')
    if len(set(fixture['source_files'])) != len(fixture['source_files']):
        raise ValueError('Duplicate fixture source file')
    for name in fixture['source_files']:
        source_file(root, name)
    if not isinstance(config['parameters'], dict):
        raise ValueError('Parameters must be an explicit dictionary')
    conventions = config['conventions']
    if not isinstance(conventions, dict) or set(conventions) != CONVENTIONS:
        raise ValueError(f'Conventions require exactly {sorted(CONVENTIONS)}')
    for name in CONVENTIONS - {'seed', 'universe', 'annualization', 'implementation_lag'}:
        if not isinstance(conventions[name], str) or not conventions[name].strip():
            raise ValueError(f'Missing convention: {name}')
    if date.fromisoformat(conventions['sample_start']) > date.fromisoformat(
            conventions['sample_end']):
        raise ValueError('Sample interval is reversed')
    if conventions['returns'] not in {'simple', 'log', 'not_applicable'}:
        raise ValueError('State the return convention explicitly')
    if conventions['seed'] is not None and type(conventions['seed']) is not int:
        raise ValueError('Seed must be an integer or explicit null for deterministic inputs')
    universe = conventions['universe']
    if (not isinstance(universe, list) or not universe
            or any(not isinstance(item, str) or not item for item in universe)
            or len(universe) != len(set(universe))):
        raise ValueError('Universe must be a nonempty unique list')
    annualization = conventions['annualization']
    if annualization is not None and (
            type(annualization) not in (int, float) or annualization <= 0):
        raise ValueError('Annualization must be positive or explicitly not applicable (null)')
    lag = conventions['implementation_lag']
    if type(lag) is not int or lag < 0:
        raise ValueError('Implementation lag must be a nonnegative observation count')
    for name in ('tables', 'input_tables', 'checks'):
        _names(config[name], name)
    if not set(config['input_tables']) <= set(config['tables']):
        raise ValueError('Input tables must be included among saved tables')
    render = config['rendering']
    if (not isinstance(render, dict) or set(render) != {'dpi', 'font_family'}
            or type(render['dpi']) is not int or not 1 <= render['dpi'] <= 600
            or not isinstance(render['font_family'], str) or not render['font_family']):
        raise ValueError('Rendering requires dpi (1..600) and an explicit font family')
    solver = config['solver']
    if (not isinstance(solver, dict) or set(solver) != {'required', 'backends'}
            or type(solver['required']) is not bool or not isinstance(solver['backends'], dict)
            or (solver['required'] and not solver['backends'])
            or any(not isinstance(name, str) or not name or not isinstance(options, dict)
                   for name, options in solver['backends'].items())):
        raise ValueError('Solver requires an explicit requirement and backend/options mapping')
    dependencies = config['dependencies']
    if not isinstance(dependencies, dict):
        raise ValueError('Dependencies must map import names to distributions')
    for module, distribution in dependencies.items():
        if (not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*', module)
                or not isinstance(distribution, str) or not distribution):
            raise ValueError('Dependencies require top-level import names and distribution names')
    return config


def generation_blockers(registry: dict, root: Path) -> list[str]:
    """List pending or invalid producer contracts without importing analytical code."""
    blockers = []
    for name, spec in registry['producers'].items():
        if spec['status'] != 'implemented':
            blockers.append(f'{name}: offline producer and fixed configuration are pending.')
        else:
            try:
                producer_contract(spec, root)
            except (ValueError, TypeError, KeyError) as error:
                blockers.append(f'{name}: {error}')
    return blockers


def dependencies_for(registry: dict) -> dict:
    """Combine producer dependencies and reject ambiguous distribution ownership."""
    dependencies = dict(BASE_DEPENDENCIES)
    for spec in registry['producers'].values():
        for module, distribution in spec['configuration']['dependencies'].items():
            if module in dependencies and dependencies[module] != distribution:
                raise ValueError(f'Conflicting dependency distribution: {module}')
            dependencies[module] = distribution
    return dependencies


def input_files(registry: dict, root: Path) -> dict:
    """Fingerprint declared fixture files, separately from generated input-table snapshots."""
    return {name: digest(source_file(root, name)) for name in sorted({
        name for spec in registry['producers'].values()
        for name in spec['configuration']['fixture']['source_files']
    })}


@contextmanager
def offline():
    """Block Python network entry points during imports, execution and validation.

    This is an accidental-network guard for trusted in-process producers, not an OS sandbox
    for hostile code or arbitrary subprocess/native-library traffic.
    """
    def denied(*args, **kwargs):
        """Reject DNS, connected sockets and datagram traffic."""
        raise RuntimeError('Documentation analytics must run offline')

    targets = [(socket.socket, name) for name in
               ('connect', 'connect_ex', 'sendto', 'sendmsg') if hasattr(socket.socket, name)]
    targets += [(socket, name) for name in
                ('create_connection', 'getaddrinfo', 'gethostbyname', 'gethostbyname_ex',
                 'gethostbyaddr')]
    originals = [(owner, name, getattr(owner, name)) for owner, name in targets]
    try:
        for owner, name, _ in originals:
            setattr(owner, name, denied)
        yield
    finally:
        for owner, name, value in originals:
            setattr(owner, name, value)


def environment_snapshot(dependencies: dict) -> dict:
    """Record actual imports, distribution versions and first-party Python source identity."""
    libraries = {}
    for name, distribution in sorted(dependencies.items()):
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        tree = None
        if name in {'qis', 'factorlasso', 'optimalportfolios'}:
            hashes = {p.relative_to(path.parent).as_posix(): digest(p)
                      for p in sorted(path.parent.rglob('*.py'))}
            tree = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
        libraries[name] = {'distribution': distribution, 'version': metadata.version(distribution),
                           'module_path': str(path), 'module_sha256': digest(path),
                           'python_tree_sha256': tree}
    return {'python': platform.python_version(), 'platform': platform.platform(),
            'libraries': libraries}


def environment_identity(record: dict) -> dict:
    """Compare content/version identity while retaining import paths as separate provenance."""
    return {'python': record['python'], 'platform': record['platform'],
            'libraries': {name: {key: value for key, value in entry.items()
                                 if key != 'module_path'}
                          for name, entry in record['libraries'].items()}}


def rendering_record(config: dict) -> dict:
    """Resolve and fingerprint the actual font used for each producer's figures."""
    from matplotlib import font_manager
    path = Path(font_manager.findfont(config['font_family'], fallback_to_default=False))
    return {**config, 'font_file': str(path.resolve()), 'font_sha256': digest(path)}


def expected_files(registry: dict) -> set[str]:
    """Return the complete image/table inventory, excluding the completion manifest."""
    paths = {f'images/{Path(asset["path"]).name}' for asset in registry['assets']}
    for name, spec in registry['producers'].items():
        paths.update(f'tables/{name}/{label}.csv' for label in spec['configuration']['tables'])
    return paths


def file_record(path: Path) -> dict:
    """Read back each PNG or CSV and record content, dimensions or table shape."""
    record = {'sha256': digest(path), 'bytes': path.stat().st_size}
    if not record['bytes']:
        raise ValueError(f'Empty output: {path.name}')
    if path.suffix == '.PNG':
        from PIL import Image
        with Image.open(path) as picture:
            if picture.format != 'PNG':
                raise ValueError('Preview must be a PNG image')
            picture.verify()
        with Image.open(path) as picture:
            picture.load()
            extrema = picture.convert('RGB').getextrema()
            if min(picture.size) < 100 or all(low == high for low, high in extrema):
                raise ValueError(f'Blank or undersized preview: {path.name}')
            record.update(width=picture.width, height=picture.height)
    elif path.suffix == '.csv':
        with path.open(encoding='utf-8', newline='') as stream:
            rows = list(csv.reader(stream, strict=True))
        if (len(rows) < 2 or not rows[0] or not all(rows[0])
                or len(rows[0]) != len(set(rows[0]))
                or any(len(row) != len(rows[0]) for row in rows[1:])):
            raise ValueError(f'Malformed or empty table: {path.name}')
        record.update(rows=len(rows) - 1, columns=rows[0])
    else:
        raise ValueError(f'Unregistered output format: {path.name}')
    return record


def check_result(result: dict, config: dict) -> None:
    """Validate actual configuration, named numerical checks and solver audit records."""
    json_safe(result)
    if set(result) != {'configuration', 'checks', 'diagnostics'}:
        raise ValueError('Producer result requires configuration, checks and diagnostics')
    if result['configuration'] != config:
        raise ValueError('Producer used a different configuration')
    checks = result['checks']
    if (not isinstance(checks, dict) or set(checks) != set(config['checks'])
            or not all(value is True for value in checks.values())):
        raise ValueError('Missing or failed numerical checks')
    diagnostics = result['diagnostics']
    if (not isinstance(diagnostics, dict) or 'solves' not in diagnostics
            or not isinstance(diagnostics['solves'], list)):
        raise ValueError('Diagnostics must include the actual solves list')
    solves = diagnostics['solves']
    explanation = diagnostics.get('not_applicable')
    if not solves and (config['solver']['required'] or not isinstance(explanation, str)
                       or not explanation.strip()):
        raise ValueError('Missing solver diagnostics or not-applicable explanation')
    required = {'solver', 'context', 'status', 'accepted', 'compliant', 'fallback_source',
                'constraint_residuals', 'covariance_stabilization'}
    for solve in solves:
        if (not isinstance(solve, dict) or not required <= set(solve)
                or solve['solver'] not in config['solver']['backends']
                or solve['status'] not in {'optimal', 'optimal_inaccurate', 'success'}
                or not isinstance(solve['context'], str) or not solve['context']
                or solve['accepted'] is not True or solve['compliant'] is not True
                or solve['fallback_source'] is not None):
            raise ValueError('Rejected, noncompliant or fallback solver result')
        if (not isinstance(solve['constraint_residuals'], list)
                or (not solve['constraint_residuals'] and (
                    not isinstance(solve.get('constraints_not_applicable'), str)
                    or not solve['constraints_not_applicable'].strip()))):
            raise ValueError('Missing constraint residual records')
        for residual in solve['constraint_residuals']:
            if (not isinstance(residual, dict) or type(residual.get('hard')) is not bool
                    or type(residual.get('passed')) is not bool
                    or not isinstance(residual.get('name'), str) or not residual['name']
                    or type(residual.get('violation')) not in (int, float)
                    or type(residual.get('tolerance')) not in (int, float)
                    or not math.isfinite(residual['violation']) or residual['violation'] < 0
                    or not math.isfinite(residual['tolerance']) or residual['tolerance'] < 0
                    or residual['passed'] != (residual['violation'] <= residual['tolerance'])
                    or (residual['hard'] and not residual['passed'])):
                raise ValueError('Invalid or failed hard-constraint residual')
        covariance = solve['covariance_stabilization']
        if (not isinstance(covariance, dict) or type(covariance.get('factorized')) is not bool
                or (covariance['factorized'] and (
                    type(covariance.get('n_floored')) is not int or covariance['n_floored'] < 0))
                or (not covariance['factorized'] and (
                    not isinstance(covariance.get('not_applicable'), str)
                    or not covariance['not_applicable'].strip()))):
            raise ValueError('Missing covariance stabilization record')


def _bundle_files(bundle: Path, wanted: set[str]) -> None:
    """Reject incomplete bundles, extra files/directories and symlink/junction escapes."""
    directories = {parent.as_posix() for name in wanted
                   for parent in Path(name).parents if parent != Path('.')}
    actual = set()
    for base, dirs, files in os.walk(bundle, followlinks=False):
        for name in dirs + files:
            path = Path(base) / name
            if path.is_symlink() or (hasattr(path, 'is_junction') and path.is_junction()):
                raise ValueError(f'Linked bundle path: {path}')
            relative = path.relative_to(bundle).as_posix()
            if name in dirs:
                if relative not in directories:
                    raise ValueError(f'Unregistered bundle directory: {relative}')
            else:
                actual.add(relative)
    if actual != wanted | {MANIFEST}:
        raise ValueError('Incomplete bundle or unregistered output')


def validate_bundle(bundle: Path, root: Path = ROOT) -> dict:
    """Check a bundle against current source, input files, environment and producer contracts.

    Args:
        bundle: Existing C-local run directory.
        root: Source checkout/export whose registry and bytes must match.

    Returns:
        Verified execution record. Visual review and publication are separate steps.

    Raises:
        ValueError: If completeness, provenance, readability or declared checks fail.
    """
    bundle = output_boundary(bundle, root, new=False)
    registry = load_registry(root)
    blockers = generation_blockers(registry, root)
    if blockers:
        raise ValueError('Generation unavailable: ' + '; '.join(blockers))
    _bundle_files(bundle, expected_files(registry))
    record = read_json(bundle / MANIFEST)
    if (record.get('schema_version') != 1
            or record.get('kind') != 'documentation_analytics_bundle'
            or record.get('status') != 'complete'
            or record.get('review_status') != 'pending'
            or record.get('publication_ready') is not False):
        raise ValueError('Not a complete unreviewed analytics bundle')
    generated = datetime.fromisoformat(record['generated_at_utc'])
    if generated.utcoffset() != timezone.utc.utcoffset(generated):
        raise ValueError('Generation timestamp must be UTC')
    if record['registry'] != registry or record['source'] != source_fingerprint(root):
        raise ValueError('Bundle registry or source is stale')
    if record['input_files'] != input_files(registry, root):
        raise ValueError('Bundle fixture input files are stale')
    if set(record['producers']) != set(registry['producers']):
        raise ValueError('Missing or unexpected producer result')
    with offline():
        current_environment = environment_snapshot(dependencies_for(registry))
        if environment_identity(record['environment']) != environment_identity(current_environment):
            raise ValueError('Bundle execution environment differs')
        for name, spec in registry['producers'].items():
            config = producer_contract(spec, root)
            check_result(record['producers'][name], config)
            if record['rendering'][name] != rendering_record(config['rendering']):
                raise ValueError('Bundle rendering/font identity differs')
    wanted = expected_files(registry)
    if set(record['outputs']) != wanted:
        raise ValueError('Incomplete output provenance')
    for name in sorted(wanted):
        if file_record(bundle / relative_path(name)) != record['outputs'][name]:
            raise ValueError(f'Output content/shape mismatch: {name}')
    expected_inputs = {f'tables/{name}/{label}.csv'
                       for name, spec in registry['producers'].items()
                       for label in spec['configuration']['input_tables']}
    if record['input_tables'] != {name: record['outputs'][name] for name in expected_inputs}:
        raise ValueError('Input-table snapshots are incomplete or mismatched')
    return record


def main(argv: list[str] | None = None) -> int:
    """Validate one existing run; return zero only for a complete, consistent bundle."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = validate_bundle(args.run_root)
        print(f'Validated {len(result["outputs"])} outputs; visual review remains pending.')
        return 0
    except (ValueError, OSError, KeyError, TypeError, RuntimeError,
            ImportError, csv.Error) as error:
        print(f'Bundle validation failed: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
