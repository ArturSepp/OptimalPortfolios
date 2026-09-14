"""Verify offline analytics planning without claiming generation or publication readiness."""

from copy import deepcopy
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import pytest


@pytest.fixture(scope='module')
def tooling(root):
    """Import repository-only tools after root has skipped installed-wheel checks."""
    with patch.object(sys, 'path', [str(root), *sys.path]):
        registry = importlib.import_module('tools.docs_analytics.registry')
        runner = importlib.import_module('tools.docs_analytics.run')
    assert Path(registry.__file__).resolve().is_relative_to(root.resolve())
    return registry, runner


@pytest.fixture
def replica(root, tooling, tmp_path):
    """Create minimal C-local source stand-ins; legacy bytes are never decoded as new output."""
    registry, _ = tooling
    spec = deepcopy(registry.load_registry(root))
    repo = tmp_path / 'source'
    repo.mkdir()
    (repo / 'README.md').write_bytes((root / 'README.md').read_bytes())
    (repo / 'docs').mkdir()
    names = {'pyproject.toml', 'uv.lock', 'AGENTS.md', 'src/optimalportfolios/example.py'}
    names.update(path for producer in spec['producers'].values()
                 for path in producer['legacy_sources'])
    names.update(asset['path'] for asset in spec['assets'])
    names.update(f"tools/docs_analytics/{name}.py" for name, producer in spec['producers'].items()
                 if producer['status'] == 'implemented')
    for name in names:
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f'Fixture content for {name}\n', encoding='utf-8')
    path = repo / registry.REGISTRY_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec), encoding='utf-8')
    return repo, spec


def save_registry(repo, spec):
    """Write a test ledger after rereading the existing file."""
    path = repo / 'tools/docs_analytics/registry.json'
    assert path.is_file()
    path.read_bytes()
    path.write_text(json.dumps(spec), encoding='utf-8')


def test_inventory_covers_the_six_legacy_previews_and_eight_badges(root, tooling):
    """Freeze public asset paths and account for every current displayed image."""
    registry, _ = tooling
    spec = registry.load_registry(root)
    assert {asset['path'] for asset in spec['assets']} == {
        'examples/figures/example_portfolio_factsheet1.PNG',
        'examples/figures/example_portfolio_factsheet2.PNG',
        'examples/figures/example_customised_report.PNG',
        'examples/figures/max_diversification_span.PNG',
        'examples/figures/multi_optimisers_backtest.PNG',
        'examples/figures/MinVariance_multi_covar_estimator_backtest.PNG',
    }
    assert len(spec['producers']) == 4
    assert len(spec['non_analytics']) == 8
    implemented = {
        'portfolio_reports', 'span_sensitivity', 'optimiser_comparison', 'covariance_comparison'}
    assert all(spec['producers'][name]['status'] == 'implemented'
               and spec['producers'][name]['configuration'] is not None for name in implemented)
    assert all(producer['status'] == 'pending' and producer['configuration'] is None
               for name, producer in spec['producers'].items() if name not in implemented)


@pytest.mark.parametrize('reference', [
    '![new](../examples/figures/unregistered.PNG)',
    '<img src="../examples/figures/unregistered.PNG">',
    '```{image} ../examples/figures/unregistered.PNG\n```',
    '.. image:: ../examples/figures/unregistered.PNG',
    '.. figure:: ../examples/figures/unregistered.PNG',
    '.. |new| image:: ../examples/figures/unregistered.PNG',
    '![named][new]\n\n[new]: ../examples/figures/unregistered.PNG',
    '![new](https://example.org/unregistered.PNG)',
])
def test_new_displayed_image_requires_registration(replica, tooling, reference):
    """New Markdown, HTML, MyST and RST images must not escape coverage."""
    repo, _ = replica
    registry, _ = tooling
    suffix = '.rst' if reference.startswith('.. ') else '.md'
    (repo / 'docs' / f'new{suffix}').write_text(reference + '\n', encoding='utf-8')
    with pytest.raises(ValueError, match='unregistered'):
        registry.load_registry(repo)


@pytest.mark.parametrize('reference', [
    '```python\n![hidden](unregistered.PNG)\n```',
    '~~~markdown\n![hidden](unregistered.PNG)\n~~~',
    '<!-- ![hidden](unregistered.PNG) -->',
    'Use `![hidden](unregistered.PNG)` as an example.',
    '.. code-block:: text\n\n    .. image:: unregistered.PNG\n\nVisible text.\n',
])
def test_code_and_comment_examples_are_not_displayed_images(tooling, reference):
    """Do not mistake illustration syntax in literal examples for image consumers."""
    registry, _ = tooling
    assert registry.image_references(reference) == []


@pytest.mark.parametrize('path', [
    '../escape.PNG', '/absolute.PNG', 'C:/escape.PNG', 'examples\\figures\\bad.PNG',
    'examples//figures/bad.PNG', 'examples/./figures/bad.PNG', '', ' bad.PNG',
])
def test_nonportable_registry_paths_are_rejected(tooling, path):
    """Reject traversal, alternate separators and ambiguous path spellings."""
    registry, _ = tooling
    with pytest.raises(ValueError):
        registry.relative_path(path)


@pytest.mark.parametrize('defect', ['duplicate_id', 'duplicate_path', 'unknown_producer',
                                     'missing_consumer', 'unused_badge', 'removed_asset',
                                     'wrong_case', 'unsupported_status'])
def test_stale_or_ambiguous_registry_is_rejected(replica, tooling, defect):
    """Catch coverage drift and invalid ownership before any source import or output write."""
    repo, spec = replica
    registry, _ = tooling
    if defect == 'duplicate_id':
        spec['assets'][1]['id'] = spec['assets'][0]['id']
    elif defect == 'duplicate_path':
        spec['assets'][1]['path'] = spec['assets'][0]['path']
    elif defect == 'unknown_producer':
        spec['assets'][0]['producer'] = 'missing'
    elif defect == 'missing_consumer':
        spec['assets'][0]['documents'] = ['docs/missing.md']
    elif defect == 'unused_badge':
        spec['non_analytics'].append({
            'document': 'README.md', 'url': 'https://example.org/badge.svg', 'reason': 'badge'})
    elif defect == 'removed_asset':
        spec['assets'].pop()
    elif defect == 'wrong_case':
        spec['assets'][0]['path'] = 'examples/figures/EXAMPLE_portfolio_factsheet1.PNG'
    else:
        spec['producers']['portfolio_reports']['status'] = 'complete'
    save_registry(repo, spec)
    with pytest.raises(ValueError):
        registry.load_registry(repo)


def test_duplicate_json_keys_are_rejected(replica, tooling):
    """The final JSON value must not silently replace an earlier schema declaration."""
    repo, _ = replica
    registry, _ = tooling
    path = repo / registry.REGISTRY_PATH
    text = path.read_text(encoding='utf-8')
    path.write_text(text.replace('"schema_version": 1',
                                 '"schema_version": 0, "schema_version": 1'), encoding='utf-8')
    with pytest.raises(ValueError, match='Duplicate registry key'):
        registry.load_registry(repo)


def test_plans_are_stable_and_distinguish_source_edits_from_legacy_bytes(replica, tooling):
    """Content identities must change when either source or a legacy file changes."""
    repo, spec = replica
    _, runner = tooling
    first = runner.build_plan(repo)
    assert first == runner.build_plan(repo)
    assert first['status'] == 'planned'
    assert first['generation_ready'] and not first['publication_ready']
    assert not first['generation_blockers']
    path = repo / 'src/optimalportfolios/example.py'
    text = path.read_text(encoding='utf-8')
    path.write_text(text + '# source changed\n', encoding='utf-8')
    source_change = runner.build_plan(repo)
    assert source_change['source']['sha256'] != first['source']['sha256']
    assert source_change['legacy_images'] == first['legacy_images']
    image_path = repo / spec['assets'][0]['path']
    image_path.write_bytes(image_path.read_bytes() + b'changed image')
    image_change = runner.build_plan(repo)
    assert image_change['legacy_images'] != source_change['legacy_images']
    assert image_change['source'] == source_change['source']


def test_readiness_requires_actual_source_and_configuration(replica, tooling):
    """An implemented label alone cannot make a producer real."""
    repo, spec = replica
    registry, _ = tooling
    producer = spec['producers']['portfolio_reports']
    producer['status'] = 'implemented'
    producer['configuration'] = None
    (repo / 'tools/docs_analytics/portfolio_reports.py').unlink()
    save_registry(repo, spec)
    with pytest.raises(ValueError, match='lacks configuration'):
        registry.load_registry(repo)
    producer['configuration'] = {'seed': 1}
    save_registry(repo, spec)
    with pytest.raises(ValueError, match='Missing source file'):
        registry.load_registry(repo)


@pytest.mark.parametrize('destination', ['local_root', 'source', 'relative', 'onedrive', 'outside'])
def test_output_boundary_rejects_unsafe_destinations(replica, tooling, monkeypatch,
                                                    tmp_path, destination):
    """No plan may land in the checkout, OneDrive, an ambiguous path or an unrelated directory."""
    repo, _ = replica
    _, runner = tooling
    monkeypatch.setenv('AGENT_LOCAL_ROOT', str(tmp_path))
    paths = {
        'local_root': tmp_path,
        'source': repo / 'output',
        'relative': Path('output'),
        'onedrive': tmp_path / 'OneDrive - example' / 'output',
        'outside': tmp_path.parent / 'outside-output',
    }
    with pytest.raises(ValueError):
        runner.output_boundary(paths[destination], repo)


def test_output_requires_setup_and_refuses_overwrites(replica, tooling, monkeypatch, tmp_path):
    """A new plan directory must be explicit and existing bytes must survive a retry."""
    repo, _ = replica
    _, runner = tooling
    output = tmp_path / 'new-plan'
    monkeypatch.delenv('AGENT_LOCAL_ROOT', raising=False)
    with pytest.raises(ValueError, match='AGENT_LOCAL_ROOT'):
        runner.write_plan({'status': 'planned'}, output, repo)
    assert not output.exists()
    monkeypatch.setenv('AGENT_LOCAL_ROOT', str(tmp_path))
    path = runner.write_plan({'status': 'planned'}, output, repo)
    original = path.read_bytes()
    assert json.loads(original) == {'status': 'planned'}
    with pytest.raises(ValueError, match='overwrite'):
        runner.write_plan({'status': 'different'}, output, repo)
    assert path.read_bytes() == original
    assert {item.name for item in output.iterdir()} == {'run_plan.json'}


def test_linked_output_is_rejected(replica, tooling, monkeypatch, tmp_path):
    """Exercise the link guard portably without requiring Windows symlink privileges."""
    repo, _ = replica
    _, runner = tooling
    monkeypatch.setenv('AGENT_LOCAL_ROOT', str(tmp_path))
    output = tmp_path / 'linked' / 'plan'
    original = Path.is_symlink

    def linked(path):
        """Represent a linked ancestor while retaining normal behavior elsewhere."""
        return path == output.parent or original(path)

    monkeypatch.setattr(Path, 'is_symlink', linked)
    with pytest.raises(ValueError, match='Linked output'):
        runner.output_boundary(output, repo)


def test_source_escape_image_is_rejected(replica, tooling):
    """An encoded traversal cannot turn a displayed image into an outside file read."""
    repo, _ = replica
    registry, _ = tooling
    (repo / 'docs/new.md').write_text('![escape](%2e%2e/%2e%2e/escape.PNG)\n', encoding='utf-8')
    with pytest.raises(ValueError, match='escapes source tree'):
        registry.load_registry(repo)


def test_cli_plans_with_only_the_standard_library_and_all_fails_without_writing(
        root, tooling, tmp_path, replica):
    """A pending registry fixture refuses generation without analytics imports or network."""
    _, runner = tooling
    original_root = root
    root, spec = replica
    for name in ('run.py', 'registry.py', 'validate.py'):
        relative = Path('tools/docs_analytics') / name
        (root / relative).write_bytes((original_root / relative).read_bytes())
    spec['producers']['covariance_comparison']['status'] = 'pending'
    spec['producers']['covariance_comparison']['configuration'] = None
    save_registry(root, spec)
    output = tmp_path / 'refused-generation'
    plan_output = tmp_path / 'cli-plan'
    script = r'''
import importlib.abc
import json
from pathlib import Path
import socket
import sys

class RejectAnalytics(importlib.abc.MetaPathFinder):
    """Reject legacy producers and analytical libraries before their import side effects."""

    def find_spec(self, fullname, path=None, target=None):
        """Fail immediately if inspection tries to load analytical code."""
        blocked = {"qis", "optimalportfolios", "factorlasso", "numpy", "pandas", "matplotlib",
                   "yfinance", "examples"}
        if fullname.split(".")[0] in blocked:
            raise AssertionError(f"Unexpected analytical import: {fullname}")

def no_network(*args, **kwargs):
    """Forbid both name resolution and connection attempts in this planning test."""
    raise AssertionError("Planning attempted network access")

sys.meta_path.insert(0, RejectAnalytics())
socket.create_connection = no_network
socket.getaddrinfo = no_network
socket.socket.connect = no_network
from tools.docs_analytics.run import main
assert main(["--list"]) == 0
assert main(["--plan", "--output-root", sys.argv[1]]) == 0
plan = json.loads((Path(sys.argv[1]) / "run_plan.json").read_text(encoding="utf-8"))
assert plan["kind"] == "documentation_analytics_plan"
assert not plan["generation_ready"] and not plan["publication_ready"]
assert main(["--all", "--output-root", sys.argv[2]]) == 2
assert not Path(sys.argv[2]).exists()
print("STDLIB_ONLY_PLANNING_PASSED")
'''
    env = dict(os.environ, PYTHONPATH=str(root), AGENT_LOCAL_ROOT=str(tmp_path),
               PYTHONDONTWRITEBYTECODE='1')
    result = subprocess.run(
        [sys.executable, '-S', '-c', script, str(plan_output), str(output)],
        cwd=root, env=env, text=True, capture_output=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'STDLIB_ONLY_PLANNING_PASSED' in result.stdout
    assert 'Generation unavailable' in result.stderr
    assert not output.exists()
    assert set(item.name for item in plan_output.iterdir()) == {'run_plan.json'}
    assert runner is not None
