"""Verify publication boundaries and recovery with controlled images in disposable repositories."""

from datetime import datetime, timezone
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from optimalportfolios.tests.documentation_analytics_bundle_test import (
    complete as complete,
    executable as executable,
)
from optimalportfolios.tests.documentation_analytics_registry_test import (
    replica as replica,
    tooling as tooling,
)


@pytest.fixture
def candidate(complete, tmp_path):
    """Publish from a source-export bundle into a separate identical test checkout."""
    source, _, _, validator, bundle = complete
    root = tmp_path / 'target'
    shutil.copytree(source, root)
    publisher = importlib.import_module('tools.docs_analytics.publish')
    review = publisher.prepare_review(bundle, root, tmp_path / 'visual-review')
    return publisher, validator, root, bundle, review


@pytest.fixture
def reviewed(candidate):
    """Attest only controlled test figures, never the real financial previews."""
    publisher, validator, root, bundle, path = candidate
    record = validator.read_json(path)
    record.update(status='reviewed', reviewer='Controlled test fixture',
                  reviewed_at_utc=datetime.now(timezone.utc).isoformat())
    for image in record['images'].values():
        image.update(full_resolution=True, article_width=True)
    rewrite(path, record)
    return publisher, validator, root, bundle, path


def rewrite(path, record):
    """Re-read a test record immediately before an intentional mutation."""
    path.read_bytes()
    path.write_text(json.dumps(record), encoding='utf-8')


def snapshot(root):
    """Compare every repository file byte, including unrelated files and manifest absence."""
    return {path.relative_to(root).as_posix(): path.read_bytes()
            for path in root.rglob('*') if path.is_file()}


def backups(bundle):
    """List only backup directories created for this independent test run."""
    return sorted(bundle.parent.glob('publication-backup-*'))


def test_review_template_does_not_approve_the_bundle(candidate, monkeypatch):
    """Preparation leaves an immutable candidate and explicitly incomplete visual-review record."""
    publisher, validator, root, bundle, path = candidate
    before = snapshot(root)
    original_bundle = snapshot(bundle)
    attempted = []
    install = publisher._install

    def record_writes(content, destination, workspace):
        """Record any transient destination replacement, including one later rolled back."""
        if destination.is_relative_to(root):
            attempted.append(destination)
        install(content, destination, workspace)

    monkeypatch.setattr(publisher, '_install', record_writes)
    template = validator.read_json(path)
    assert template['status'] == 'pending' and template['reviewer'] == ''
    assert len(template['images']) == 6
    assert all(not image['full_resolution'] and not image['article_width']
               for image in template['images'].values())
    with pytest.raises(ValueError, match='review'):
        publisher.publish_bundle(bundle, root, path)
    assert snapshot(root) == before
    assert snapshot(bundle) == original_bundle
    assert not attempted
    assert not backups(bundle)


@pytest.mark.parametrize('prior_manifest', [False, True])
def test_publish_verify_and_manual_rollback_restore_exact_bytes(reviewed, prior_manifest):
    """Copy only the six PNGs and receipt; preserve tables locally and restore prior absence too."""
    publisher, validator, root, bundle, review = reviewed
    if prior_manifest:
        (root / publisher.PROVENANCE).write_bytes(b'previous provenance bytes')
    (root / 'unrelated.txt').write_text('retain this file', encoding='utf-8')
    previous = snapshot(root)
    original_bundle = snapshot(bundle)
    receipt = publisher.publish_bundle(bundle, root, review)
    assert publisher.verify_published(root) == receipt
    after = snapshot(root)
    changed = {name for name in after.keys() | previous.keys()
               if after.get(name) != previous.get(name)}
    assert changed == set(publisher.TARGETS)
    assert snapshot(bundle) == original_bundle
    assert 'tables' not in {path.name for path in root.iterdir()}
    backup, = backups(bundle)
    assert validator.read_json(backup / 'transaction.json')['status'] == 'complete'
    journal = publisher.rollback(backup, root)
    assert journal['status'] == 'rolled_back'
    assert snapshot(root) == previous
    assert publisher.rollback(backup, root)['status'] == 'rolled_back'


@pytest.mark.parametrize('defect', [
    'review_hash', 'image_hash', 'reviewer', 'review_scope', 'full_resolution', 'article_width',
    'review_timestamp', 'changed_png', 'missing_png', 'missing_table', 'extra_file',
    'source_drift', 'fixture_drift', 'manifest_directory',
])
def test_invalid_publication_changes_no_destination(reviewed, defect):
    """All candidate, review and destination failures occur before any repository write."""
    publisher, validator, root, bundle, review_path = reviewed
    review = validator.read_json(review_path)
    name = publisher.PREVIEW_PATHS[0]
    if defect == 'review_hash':
        review['bundle_sha256'] = 'stale'
    elif defect == 'image_hash':
        review['images'][name]['sha256'] = 'stale'
    elif defect == 'reviewer':
        review['reviewer'] = ''
    elif defect == 'review_scope':
        review['images'].pop(name)
    elif defect in {'full_resolution', 'article_width'}:
        review['images'][name][defect] = False
    elif defect == 'review_timestamp':
        review['reviewed_at_utc'] = '2020-01-01T00:00:00'
    elif defect == 'changed_png':
        image = bundle / 'images' / Path(name).name
        image.write_bytes(image.read_bytes() + b'drift')
    elif defect == 'missing_png':
        (bundle / 'images' / Path(name).name).unlink()
    elif defect == 'missing_table':
        next((bundle / 'tables').rglob('*.csv')).unlink()
    elif defect == 'extra_file':
        (bundle / 'unexpected.PNG').write_bytes(b'not registered')
    elif defect == 'source_drift':
        source = root / 'tools/docs_analytics/portfolio_reports.py'
        source.write_bytes(source.read_bytes() + b'\n# drift\n')
    elif defect == 'fixture_drift':
        (root / 'input.txt').write_text('drift', encoding='utf-8')
    elif defect == 'manifest_directory':
        (root / publisher.PROVENANCE).mkdir()
    rewrite(review_path, review)
    before = snapshot(root)
    with pytest.raises(ValueError):
        publisher.publish_bundle(bundle, root, review_path)
    assert snapshot(root) == before
    assert not backups(bundle)


@pytest.mark.parametrize('position', [0, 2, 6])
@pytest.mark.parametrize('after_replace', [False, True])
def test_mid_copy_failure_rolls_back_even_after_replace_raises(
        reviewed, monkeypatch, position, after_replace):
    """Recovery inspects bytes rather than trusting that an interrupted replace returned."""
    publisher, validator, root, bundle, review = reviewed
    (root / publisher.PROVENANCE).write_bytes(b'old manifest')
    before = snapshot(root)
    install = publisher._install
    counter, fired, order = 0, False, []

    def fail_once(content, destination, workspace):
        """Inject one failure before/after a selected target replacement, then allow recovery."""
        nonlocal counter, fired
        if destination.is_relative_to(root) and not fired:
            order.append(destination.relative_to(root).as_posix())
            index = counter
            counter += 1
            if index == position:
                fired = True
                if after_replace:
                    install(content, destination, workspace)
                raise OSError('Injected copy failure')
        install(content, destination, workspace)

    monkeypatch.setattr(publisher, '_install', fail_once)
    with pytest.raises(OSError, match='Injected'):
        publisher.publish_bundle(bundle, root, review)
    assert snapshot(root) == before
    backup, = backups(bundle)
    assert validator.read_json(backup / 'transaction.json')['status'] == 'rolled_back'
    if position == 6:
        assert order == list(publisher.TARGETS)


def test_verification_failure_also_rolls_back(reviewed, monkeypatch):
    """A late failure after the manifest is installed restores every original byte."""
    publisher, _, root, bundle, review = reviewed
    before = snapshot(root)

    def reject(*args, **kwargs):
        """Simulate a final receipt/image consistency failure."""
        raise ValueError('Injected post-publication validation failure')

    monkeypatch.setattr(publisher, 'verify_published', reject)
    with pytest.raises(ValueError, match='Injected'):
        publisher.publish_bundle(bundle, root, review)
    assert snapshot(root) == before


def test_rollback_failure_retains_recoverable_backup(reviewed, monkeypatch):
    """Report failed restoration honestly; retry after its external cause is removed."""
    publisher, validator, root, bundle, review = reviewed
    before = snapshot(root)
    install = publisher._install
    writes = 0

    def interrupted(content, destination, workspace):
        """Allow one target replacement, then simulate persistent filesystem failure."""
        nonlocal writes
        if destination.is_relative_to(root):
            writes += 1
            if writes > 1:
                raise OSError('Injected filesystem failure')
        install(content, destination, workspace)

    monkeypatch.setattr(publisher, '_install', interrupted)
    with pytest.raises(RuntimeError, match='rollback incomplete'):
        publisher.publish_bundle(bundle, root, review)
    backup, = backups(bundle)
    assert validator.read_json(backup / 'transaction.json')['status'] == 'rollback_failed'
    monkeypatch.setattr(publisher, '_install', install)
    publisher.rollback(backup, root)
    assert snapshot(root) == before


@pytest.mark.parametrize('defect', ['concurrent_edit', 'corrupt_backup', 'wrong_root'])
def test_recovery_preflight_prevents_destructive_restoration(reviewed, tmp_path, defect):
    """Reject conflicting or damaged recovery material before restoring any path."""
    publisher, _, root, bundle, review = reviewed
    publisher.publish_bundle(bundle, root, review)
    backup, = backups(bundle)
    if defect == 'concurrent_edit':
        (root / publisher.PREVIEW_PATHS[0]).write_bytes(b'another writer owns these bytes')
    elif defect == 'corrupt_backup':
        path = backup / 'before' / Path(publisher.PREVIEW_PATHS[0]).name
        path.write_bytes(b'corrupt')
    else:
        other = tmp_path / 'different-target'
        shutil.copytree(root, other)
        root = other
    before = snapshot(root)
    with pytest.raises(ValueError):
        publisher.rollback(backup, root)
    assert snapshot(root) == before


def test_linked_destination_is_rejected(reviewed, monkeypatch):
    """Exercise the Windows/POSIX link guard without requiring symlink privileges."""
    publisher, _, root, bundle, review = reviewed
    target = root / publisher.PROVENANCE
    original = publisher._linked

    def linked(path):
        """Mark the target receipt as linked while preserving the rest of preflight."""
        return path == target or original(path)

    monkeypatch.setattr(publisher, '_linked', linked)
    before = snapshot(root)
    with pytest.raises(ValueError, match='Linked publication'):
        publisher.publish_bundle(bundle, root, review)
    assert snapshot(root) == before


def test_lock_blocks_concurrent_publication(reviewed):
    """The same-machine OS lock rejects another publication and releases on context exit."""
    publisher, _, root, bundle, review = reviewed
    before = snapshot(root)
    with publisher._lock(root):
        with pytest.raises(ValueError, match='repository lock'):
            publisher.publish_bundle(bundle, root, review)
    assert snapshot(root) == before
    publisher.publish_bundle(bundle, root, review)


@pytest.mark.parametrize('defect', ['image', 'review', 'paths'])
def test_published_verification_rejects_drift(reviewed, defect):
    """Verify images and review bindings without claiming source freshness."""
    publisher, validator, root, bundle, review = reviewed
    publisher.publish_bundle(bundle, root, review)
    if defect == 'image':
        (root / publisher.PREVIEW_PATHS[0]).write_bytes(b'corrupt')
    else:
        path = root / publisher.PROVENANCE
        record = validator.read_json(path)
        if defect == 'review':
            record['review']['bundle_sha256'] = 'stale'
        else:
            record['paths'].pop()
        rewrite(path, record)
    with pytest.raises((ValueError, OSError)):
        publisher.verify_published(root)


def test_published_record_remains_a_dated_snapshot(reviewed):
    """Unrelated source updates do not falsely invalidate unchanged historical preview bytes."""
    publisher, _, root, bundle, review = reviewed
    receipt = publisher.publish_bundle(bundle, root, review)
    path = root / 'tools/docs_analytics/portfolio_reports.py'
    path.write_bytes(path.read_bytes() + b'\n# later source revision\n')
    assert publisher.verify_published(root) == receipt


def test_killed_process_can_be_recovered_by_cli(reviewed):
    """A real process exit mid-copy leaves a journal that a fresh process can roll back."""
    publisher, validator, root, bundle, review = reviewed
    before = snapshot(root)
    code = """
import os
from pathlib import Path
import sys
from tools.docs_analytics import publish
root = Path(sys.argv[2])
original = publish._install
counter = 0
def interrupted(content, destination, workspace):
    global counter
    original(content, destination, workspace)
    if destination.is_relative_to(root):
        counter += 1
        if counter == 2:
            os._exit(73)
publish._install = interrupted
publish.publish_bundle(Path(sys.argv[1]), root, Path(sys.argv[3]))
"""
    environment = dict(os.environ, PYTHONPATH=str(root))
    killed = subprocess.run([sys.executable, '-B', '-c', code, str(bundle), str(root), str(review)],
                            cwd=root, env=environment, capture_output=True, text=True, timeout=90)
    assert killed.returncode == 73, killed.stdout + killed.stderr
    assert snapshot(root) != before
    backup, = backups(bundle)
    assert validator.read_json(backup / 'transaction.json')['status'] == 'publishing'
    restored = subprocess.run([
        sys.executable, '-B', '-m', 'tools.docs_analytics.publish', '--repo-root', str(root),
        '--rollback', str(backup)], cwd=root, env=environment, capture_output=True, text=True,
        timeout=90)
    assert restored.returncode == 0, restored.stdout + restored.stderr
    assert snapshot(root) == before


def test_cli_publish_and_verify(reviewed):
    """Fresh CLI processes report success only after complete reviewed publication."""
    publisher, _, root, bundle, review = reviewed
    environment = dict(os.environ, PYTHONPATH=str(root))

    def command(*args):
        """Run one publication CLI action against the disposable target."""
        return subprocess.run([sys.executable, '-m', 'tools.docs_analytics.publish',
                               '--repo-root', str(root), *map(str, args)],
                              cwd=root, env=environment, capture_output=True, text=True, timeout=90)

    result = command('--run-root', bundle, '--review-file', review)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'Published and verified 6' in result.stdout
    assert command('--verify').returncode == 0
    (root / publisher.PREVIEW_PATHS[0]).unlink()
    assert command('--verify').returncode == 2


def test_review_template_must_stay_outside_candidate(candidate):
    """Preparing review inside a valid bundle must not invalidate that bundle."""
    publisher, _, root, bundle, _ = candidate
    before = snapshot(bundle)
    with pytest.raises(ValueError, match='separate'):
        publisher.prepare_review(bundle, root, bundle / 'review')
    assert snapshot(bundle) == before


def test_registry_cannot_widen_publication_paths(reviewed):
    """A valid changed ledger cannot silently authorize a seventh/different preview location."""
    publisher, validator, root, bundle, review = reviewed
    ledger = root / 'tools/docs_analytics/registry.json'
    registry = validator.read_json(ledger)
    old = registry['assets'][0]['path']
    replacement = 'examples/figures/not_approved.PNG'
    (root / replacement).write_bytes((root / old).read_bytes())
    readme = root / 'README.md'
    updated = readme.read_text(encoding='utf-8').replace(old, replacement)
    readme.write_text(updated, encoding='utf-8')
    registry['assets'][0]['path'] = replacement
    rewrite(ledger, registry)
    before = snapshot(root)
    with pytest.raises(ValueError, match='approved preview'):
        publisher.publish_bundle(bundle, root, review)
    assert snapshot(root) == before
    assert not backups(bundle)


def test_source_change_during_copy_rolls_back_only_deliverables(reviewed, monkeypatch):
    """An input edit during copying aborts publication and survives preview restoration."""
    publisher, _, root, bundle, review = reviewed
    before = snapshot(root)
    install = publisher._install
    changed = False

    def edit_input(content, destination, workspace):
        """Simulate a separate source/input editor immediately after the first PNG copy."""
        nonlocal changed
        install(content, destination, workspace)
        if destination.is_relative_to(root) and not changed:
            changed = True
            (root / 'input.txt').write_bytes(b'a newer fixture edit')

    monkeypatch.setattr(publisher, '_install', edit_input)
    with pytest.raises(ValueError, match='fixture changed'):
        publisher.publish_bundle(bundle, root, review)
    before['input.txt'] = b'a newer fixture edit'
    assert snapshot(root) == before


def test_destination_edit_during_copy_is_not_overwritten_by_recovery(reviewed, monkeypatch):
    """A competing writer's bytes stop copying and recovery until the conflict is reconciled."""
    publisher, _, root, bundle, review = reviewed
    before = snapshot(root)
    install = publisher._install
    conflicting = root / publisher.PREVIEW_PATHS[1]
    changed = False

    def competing_edit(content, destination, workspace):
        """Inject another writer's update to the second preview after the first copy."""
        nonlocal changed
        install(content, destination, workspace)
        if destination.is_relative_to(root) and not changed:
            changed = True
            conflicting.write_bytes(b'concurrent preview update')

    monkeypatch.setattr(publisher, '_install', competing_edit)
    with pytest.raises(RuntimeError, match='rollback incomplete'):
        publisher.publish_bundle(bundle, root, review)
    assert conflicting.read_bytes() == b'concurrent preview update'
    monkeypatch.setattr(publisher, '_install', install)
    conflicting.write_bytes(before[conflicting.relative_to(root).as_posix()])
    backup, = backups(bundle)
    publisher.rollback(backup, root)
    assert snapshot(root) == before


def test_wrong_case_manifest_destination_is_rejected(reviewed):
    """Avoid replacing a differently cased manifest name on case-insensitive filesystems."""
    publisher, _, root, bundle, review = reviewed
    (root / 'examples/figures/Analytics_Manifest.json').write_bytes(b'wrong case')
    before = snapshot(root)
    with pytest.raises(ValueError, match='wrong case'):
        publisher.publish_bundle(bundle, root, review)
    assert snapshot(root) == before
