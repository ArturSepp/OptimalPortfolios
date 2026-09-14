"""Exercise complete analytics bundles with controlled producers, never financial baselines."""

from copy import deepcopy
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from optimalportfolios.tests.documentation_analytics_registry_test import (
    replica as replica,
    tooling as tooling,
)


PRODUCER = '''
"""Controlled offline producer for executor contract tests."""
from pathlib import Path
import socket

def produce(spec):
    """Return labelled test figures/tables; injected modes exercise failure boundaries."""
    from copy import deepcopy
    import matplotlib.pyplot as plt
    import pandas as pd
    config = deepcopy(spec['configuration'])
    mode = config['parameters']['mode']
    table = pd.DataFrame({'value': [0.1, 0.2]}, index=pd.Index(['a', 'b'], name='row'))
    figures = {}
    for filename in config['parameters']['figures']:
        figure, axis = plt.subplots(figsize=(3, 2))
        axis.plot([0, 1], [1, 2])
        axis.set_title('Controlled test only')
        figures[filename] = figure
    result = {'figures': figures, 'tables': {'inputs': table, 'summary': table.copy()},
              'configuration': config, 'checks': {'finite': True},
              'diagnostics': {'solves': [], 'not_applicable': 'No solver in this test.'}}
    if mode == 'missing_figure':
        figures.pop(next(iter(figures)))
    elif mode == 'missing_table':
        result['tables'].pop('summary')
    elif mode == 'extra_table':
        result['tables']['extra'] = table
    elif mode == 'false_check':
        result['checks']['finite'] = False
    elif mode == 'missing_check':
        result['checks'] = {}
    elif mode == 'config_drift':
        config['conventions']['seed'] = 999
    elif mode == 'empty_input':
        result['tables']['inputs'] = table.iloc[:0]
    elif mode == 'duplicate_header':
        table.index.name = 'value'
    elif mode == 'blank_png':
        for figure in figures.values():
            figure.clear()
    elif mode == 'crash':
        raise RuntimeError('Injected producer failure')
    elif mode == 'network':
        socket.getaddrinfo('example.org', 443)
    elif mode == 'input_drift':
        (Path(__file__).parents[2] / 'input.txt').write_text('changed', encoding='utf-8')
    elif mode == 'source_drift':
        path = Path(__file__)
        path.write_text(path.read_text(encoding='utf-8') + '\\n# changed\\n', encoding='utf-8')
    return result
'''


@pytest.fixture
def executable(replica, tooling, root, monkeypatch, tmp_path):
    """Supply all four families with controlled entry points and explicit configurations."""
    repo, registry = replica
    for path in (root / 'tools/docs_analytics').glob('*.py'):
        shutil.copyfile(path, repo / 'tools/docs_analytics' / path.name)
    (repo / 'input.txt').write_text('fixed test input', encoding='utf-8')
    for name, spec in registry['producers'].items():
        figures = [Path(asset['path']).name for asset in registry['assets']
                   if asset['producer'] == name]
        spec['status'] = 'implemented'
        spec['configuration'] = {
            'fixture': {'factory': 'controlled test only', 'source_files': ['input.txt']},
            'parameters': {'figures': figures, 'mode': 'ok'},
            'conventions': {
                'data_kind': 'controlled test', 'sample_start': '2020-01-01',
                'sample_end': '2020-01-02', 'seed': None, 'universe': ['A'],
                'missing_data': 'none', 'returns': 'not_applicable',
                'observation_frequency': 'not applicable', 'estimation_frequency': 'not applicable',
                'annualization': None, 'warmup': 'none', 'rebalance_frequency': 'none',
                'implementation_lag': 0, 'transaction_costs': 'not applicable',
                'sharpe_convention': 'not applicable',
            },
            'tables': ['inputs', 'summary'], 'input_tables': ['inputs'], 'checks': ['finite'],
            'rendering': {'dpi': 100, 'font_family': 'DejaVu Sans'},
            'solver': {'required': False, 'backends': {}}, 'dependencies': {},
        }
        (repo / f'tools/docs_analytics/{name}.py').write_text(PRODUCER, encoding='utf-8')
    save(repo, registry)
    monkeypatch.setenv('AGENT_LOCAL_ROOT', str(tmp_path))
    validator = importlib.import_module('tools.docs_analytics.validate')
    return repo, registry, tooling[1], validator


def save(repo, value):
    """Persist a deliberate test registry mutation after rereading it."""
    path = repo / 'tools/docs_analytics/registry.json'
    path.read_bytes()
    path.write_text(json.dumps(value), encoding='utf-8')


@pytest.fixture
def complete(executable, tmp_path):
    """Generate a validated candidate once for each independent tampering test."""
    repo, registry, runner, validator = executable
    bundle = runner.generate(tmp_path / 'bundle', repo)
    return repo, registry, runner, validator, bundle


def test_all_producers_repeat_and_capture_inputs_and_environment(complete, tmp_path):
    """Six real asset names complete twice with identical test data, PNGs and numerical tables."""
    repo, registry, runner, validator, first = complete
    second = runner.generate(tmp_path / 'repeat', repo)
    a, b = validator.validate_bundle(first, repo), validator.validate_bundle(second, repo)
    assert len(a['outputs']) == 14
    assert a['outputs'] == b['outputs']
    assert a['producers'] == b['producers']
    assert len(a['input_tables']) == 4
    assert a['input_files']['input.txt'] == validator.digest(repo / 'input.txt')
    assert a['environment']['libraries']['matplotlib']['module_path']
    assert a['rendering']['portfolio_reports']['font_sha256']
    assert a['review_status'] == 'pending' and not a['publication_ready']
    assert runner.build_plan(repo)['generation_ready']
    assert not runner.build_plan(repo)['publication_ready']
    assert not list(tmp_path.glob('.*-building-*'))
    assert {p.name for p in (first / 'images').iterdir()} == {
        Path(asset['path']).name for asset in registry['assets']
    }


@pytest.mark.parametrize('mode', [
    'missing_figure', 'missing_table', 'extra_table', 'false_check', 'missing_check',
    'config_drift', 'empty_input', 'duplicate_header', 'blank_png', 'crash',
    'network', 'input_drift', 'source_drift',
])
def test_failed_producers_leave_only_failed_staging(executable, tmp_path, mode):
    """Incomplete, drifting, networked and crashed runs never create the requested destination."""
    repo, registry, runner, validator = executable
    registry['producers']['portfolio_reports']['configuration']['parameters']['mode'] = mode
    save(repo, registry)
    with pytest.raises((ValueError, RuntimeError)):
        runner.generate(tmp_path / 'failed', repo)
    assert not (tmp_path / 'failed').exists()
    staging, = tmp_path.glob('.failed-building-*')
    assert (staging / 'FAILED.json').is_file()
    assert not (staging / validator.MANIFEST).exists()
    import matplotlib.pyplot as plt
    assert not plt.get_fignums()


@pytest.mark.parametrize('defect', [
    'missing_image', 'missing_table', 'extra_image', 'extra_directory', 'changed_png',
    'changed_csv', 'input_file', 'source_file', 'registry', 'environment', 'font',
    'producer', 'input_snapshot', 'check', 'configuration', 'pending_status',
    'publication_flag', 'timestamp', 'duplicate_json', 'nonfinite_json', 'csv_shape',
    'blank_image',
])
def test_validator_rejects_tampered_or_stale_bundle(complete, defect):
    """Reject output and provenance drift, including rehashed malformed content."""
    repo, _, _, validator, bundle = complete
    manifest = bundle / validator.MANIFEST
    record = validator.read_json(manifest)
    image_name = next(name for name in record['outputs'] if name.startswith('images/'))
    table_name = next(name for name in record['outputs'] if name.endswith('/summary.csv'))
    if defect == 'missing_image':
        (bundle / image_name).unlink()
    elif defect == 'missing_table':
        (bundle / table_name).unlink()
    elif defect == 'extra_image':
        (bundle / 'images/extra.PNG').write_bytes(b'extra')
    elif defect == 'extra_directory':
        (bundle / 'extra').mkdir()
    elif defect == 'changed_png':
        path = bundle / image_name
        path.write_bytes(path.read_bytes() + b'drift')
    elif defect == 'changed_csv':
        (bundle / table_name).write_text('row,value\nx,0\n', encoding='utf-8')
    elif defect == 'input_file':
        (repo / 'input.txt').write_text('stale', encoding='utf-8')
    elif defect == 'source_file':
        path = repo / 'tools/docs_analytics/portfolio_reports.py'
        path.write_text(path.read_text(encoding='utf-8') + '\n# drift\n', encoding='utf-8')
    elif defect == 'registry':
        record['registry']['producers']['portfolio_reports']['purpose'] = 'drift'
    elif defect == 'environment':
        record['environment']['libraries']['pandas']['version'] = '0'
    elif defect == 'font':
        record['rendering']['portfolio_reports']['font_sha256'] = 'drift'
    elif defect == 'producer':
        record['producers'].pop('portfolio_reports')
    elif defect == 'input_snapshot':
        record['input_tables'] = {}
    elif defect == 'check':
        record['producers']['portfolio_reports']['checks']['finite'] = False
    elif defect == 'configuration':
        record['producers']['portfolio_reports']['configuration']['conventions']['seed'] = 9
    elif defect == 'pending_status':
        record['status'] = 'building'
    elif defect == 'publication_flag':
        record['publication_ready'] = True
    elif defect == 'timestamp':
        record['generated_at_utc'] = '2020-01-01T12:00:00'
    elif defect == 'csv_shape':
        (bundle / table_name).write_text('row,value\nx,1,2\n', encoding='utf-8')
        record['outputs'][table_name]['sha256'] = validator.digest(bundle / table_name)
    elif defect == 'blank_image':
        from PIL import Image
        Image.new('RGB', (200, 200), 'white').save(bundle / image_name)
        record['outputs'][image_name]['sha256'] = validator.digest(bundle / image_name)
    manifest.read_bytes()
    content = json.dumps(record)
    if defect == 'duplicate_json':
        content = content[:-1] + ', "status": "complete"}'
    elif defect == 'nonfinite_json':
        content = content[:-1] + ', "unexpected": NaN}'
    manifest.write_text(content, encoding='utf-8')
    with pytest.raises(ValueError):
        validator.validate_bundle(bundle, repo)


@pytest.mark.parametrize('defect', ['configuration', 'tables', 'dependency', 'lag', 'dates'])
def test_contract_preflight_creates_no_staging(executable, tmp_path, defect):
    """Reject incomplete conventions and conflicting dependencies before importing producers."""
    repo, registry, runner, _ = executable
    config = registry['producers']['portfolio_reports']['configuration']
    if defect == 'configuration':
        config.pop('conventions')
    elif defect == 'tables':
        config['input_tables'] = ['unknown']
    elif defect == 'dependency':
        config['dependencies'] = {'numpy': 'not-numpy'}
    elif defect == 'lag':
        config['conventions']['implementation_lag'] = -1
    elif defect == 'dates':
        config['conventions']['sample_start'] = '2021-01-01'
    save(repo, registry)
    with pytest.raises(ValueError):
        runner.generate(tmp_path / 'invalid', repo)
    assert not list(tmp_path.glob('.invalid-building-*'))
    assert not (tmp_path / 'invalid').exists()


def solver_result(config):
    """Construct a declared solver audit record without executing a financial solver."""
    config['solver'] = {'required': True, 'backends': {'controlled': {'tolerance': 1e-8}}}
    return {'configuration': config, 'checks': {'finite': True}, 'diagnostics': {'solves': [{
        'solver': 'controlled', 'context': 'test solve', 'status': 'optimal',
        'accepted': True, 'compliant': True, 'fallback_source': None,
        'constraint_residuals': [
            {'name': 'test_bound', 'hard': True, 'passed': True,
             'violation': 0.0, 'tolerance': 1e-8}],
        'covariance_stabilization': {'factorized': True, 'n_floored': 0},
    }]}}


@pytest.mark.parametrize('defect', [
    'missing_solve', 'fallback', 'noncompliant', 'rejected', 'infeasible', 'backend',
    'hard_residual', 'dishonest_pass', 'missing_covariance', 'negative_floor',
])
def test_solver_failures_cannot_pass_declared_checks(executable, defect):
    """Solver success flags cannot conceal fallback use or failing hard residuals."""
    _, registry, _, validator = executable
    config = deepcopy(registry['producers']['portfolio_reports']['configuration'])
    result = solver_result(config)
    validator.check_result(result, config)
    solve = result['diagnostics']['solves'][0]
    if defect == 'missing_solve':
        result['diagnostics']['solves'] = []
    elif defect == 'fallback':
        solve['fallback_source'] = 'previous_weights'
    elif defect == 'noncompliant':
        solve['compliant'] = False
    elif defect == 'rejected':
        solve['accepted'] = False
    elif defect == 'infeasible':
        solve['status'] = 'infeasible'
    elif defect == 'backend':
        solve['solver'] = 'undeclared'
    elif defect in {'hard_residual', 'dishonest_pass'}:
        solve['constraint_residuals'][0]['violation'] = 1.0
        solve['constraint_residuals'][0]['passed'] = defect == 'dishonest_pass'
    elif defect == 'missing_covariance':
        solve['covariance_stabilization'] = {}
    elif defect == 'negative_floor':
        solve['covariance_stabilization']['n_floored'] = -1
    with pytest.raises(ValueError):
        validator.check_result(result, config)


def test_validator_failure_removes_completion_marker(executable, tmp_path, monkeypatch):
    """A late read-back failure retains diagnostics but cannot leave a complete manifest."""
    repo, _, runner, validator = executable

    def reject(*args, **kwargs):
        """Inject failure after the manifest and all candidate files have been written."""
        raise ValueError('Injected validation failure')

    monkeypatch.setattr(runner, 'validate_bundle', reject)
    with pytest.raises(ValueError, match='Injected validation'):
        runner.generate(tmp_path / 'late', repo)
    stage, = tmp_path.glob('.late-building-*')
    assert not (stage / validator.MANIFEST).exists()
    assert (stage / 'FAILED.json').is_file()


def test_bundle_symlink_rejected_without_following_it(complete, monkeypatch):
    """Mock the platform link flag to exercise the guard without Windows symlink privileges."""
    repo, _, _, validator, bundle = complete
    target = next((bundle / 'images').iterdir())
    original = Path.is_symlink

    def linked(path):
        """Report one output as a symlink while preserving all other checks."""
        return path == target or original(path)

    monkeypatch.setattr(Path, 'is_symlink', linked)
    with pytest.raises(ValueError, match='Linked bundle'):
        validator.validate_bundle(bundle, repo)


def test_cli_generation_and_validation_require_complete_bundle(executable, tmp_path):
    """Use fresh processes to prove both command-line success and nonzero validation failure."""
    repo, _, _, validator = executable
    environment = dict(os.environ, PYTHONPATH=str(repo))
    output = tmp_path / 'cli'

    def run(module, *args):
        """Execute repository-only tools from the controlled source export."""
        return subprocess.run([sys.executable, '-m', module, *map(str, args)], cwd=repo,
                              env=environment, text=True, capture_output=True, timeout=90)

    generated = run('tools.docs_analytics.run', '--all', '--output-root', output)
    assert generated.returncode == 0, generated.stderr
    checked = run('tools.docs_analytics.validate', '--run-root', output)
    assert checked.returncode == 0, checked.stderr
    assert 'visual review remains pending' in checked.stdout
    repeated = run('tools.docs_analytics.run', '--all', '--output-root', output)
    assert repeated.returncode == 2 and 'overwrite' in repeated.stderr
    assert (output / validator.MANIFEST).is_file()
    next((output / 'images').iterdir()).unlink()
    broken = run('tools.docs_analytics.validate', '--run-root', output)
    assert broken.returncode == 2 and 'Incomplete' in broken.stderr


def test_import_origin_must_match_the_selected_export(executable, tmp_path):
    """Installed/source imports from another tree cannot stand in for this export."""
    repo, registry, runner, _ = executable
    registry['producers']['portfolio_reports']['configuration']['dependencies'] = {
        'optimalportfolios': 'optimalportfolios',
    }
    save(repo, registry)
    with pytest.raises(ValueError, match='selected source export'):
        runner.generate(tmp_path / 'wrong-import', repo)
    assert not (tmp_path / 'wrong-import').exists()


def test_a_plan_is_never_accepted_as_an_execution_bundle(executable, tmp_path):
    """Even an execution-ready plan contains no validated output."""
    repo, _, runner, validator = executable
    plan = runner.write_plan(runner.build_plan(repo), tmp_path / 'plan', repo)
    with pytest.raises(ValueError, match='Incomplete'):
        validator.validate_bundle(plan.parent, repo)
