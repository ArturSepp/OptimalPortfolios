"""Publish reviewed documentation previews with C-local backups and recoverable rollback.

The small publication workflow adapts the MIT-licensed qis documentation tooling locally.
Copies of seven files are not a transaction; the manifest is written last in both directions.
"""

import argparse
from contextlib import contextmanager
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import uuid

from tools.docs_analytics.registry import load_registry
from tools.docs_analytics.validate import (
    check_result, digest, expected_files, file_record, input_files, output_boundary,
    read_json, source_fingerprint, validate_bundle, write_json,
)


PREVIEW_PATHS = tuple(sorted(
    'examples/figures/' + name for name in (
        'example_portfolio_factsheet1.PNG', 'example_portfolio_factsheet2.PNG',
        'example_customised_report.PNG', 'max_diversification_span.PNG',
        'multi_optimisers_backtest.PNG', 'MinVariance_multi_covar_estimator_backtest.PNG',
    )
))
PROVENANCE = 'examples/figures/analytics_manifest.json'
TARGETS = (*PREVIEW_PATHS, PROVENANCE)


def _encoded(value: dict) -> bytes:
    """Serialize records consistently without NaN or platform-dependent line endings."""
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n').encode('utf-8')


def _hash(content: bytes) -> str:
    """Hash immutable bytes captured for review, installation or recovery."""
    return hashlib.sha256(content).hexdigest()


def _linked(path: Path) -> bool:
    """Recognize symbolic links and Windows directory junctions."""
    return path.is_symlink() or (hasattr(path, 'is_junction') and path.is_junction())


def _root(root: Path) -> Path:
    """Require an explicit existing repository directory without linked ancestors."""
    if not root.is_absolute() or '..' in root.parts or not root.is_dir():
        raise ValueError('Repository root must be an absolute existing directory')
    if any(_linked(path) for path in (root, *root.parents)):
        raise ValueError('Linked repository root is not allowed')
    return root.resolve()


def _destination(root: Path, name: str) -> Path:
    """Accept only the seven reviewed deliverables, with exact case and no path links."""
    if name not in TARGETS:
        raise ValueError(f'Publication path is not allowlisted: {name}')
    path = root
    for part in Path(name).parts:
        if path.is_dir():
            actual = [entry.name for entry in path.iterdir()
                      if entry.name.casefold() == part.casefold()]
            if actual and actual != [part]:
                raise ValueError(f'Publication destination has wrong case: {name}')
        path = path / part
        if _linked(path):
            raise ValueError(f'Linked publication destination: {name}')
    if not path.parent.is_dir() or (path.exists() and not path.is_file()):
        raise ValueError(f'Invalid publication destination: {name}')
    if not path.resolve().is_relative_to(root):
        raise ValueError(f'Publication destination escapes repository: {name}')
    return path


def _registry(root: Path) -> dict:
    """Require exact agreement with the six stable preview paths, never a widened allowlist."""
    registry = load_registry(root)
    if tuple(sorted(asset['path'] for asset in registry['assets'])) != PREVIEW_PATHS:
        raise ValueError('Registry differs from the six approved preview paths')
    return registry


def _utc(value: str) -> None:
    """Require an explicitly UTC timestamp rather than an ambiguous local date."""
    stamp = datetime.fromisoformat(value)
    if stamp.utcoffset() != timezone.utc.utcoffset(stamp):
        raise ValueError('Review/publication timestamp must be UTC')


def _check_review(review: dict, bundle: dict) -> None:
    """Bind every visual-review assertion to the exact candidate record and six images."""
    required = {'schema_version', 'kind', 'status', 'bundle_sha256', 'reviewer',
                'reviewed_at_utc', 'images'}
    if (set(review) != required or review['schema_version'] != 1
            or review['kind'] != 'documentation_analytics_review'
            or review['status'] != 'reviewed'
            or review['bundle_sha256'] != _hash(_encoded(bundle))
            or not isinstance(review['reviewer'], str) or not review['reviewer'].strip()
            or set(review['images']) != set(PREVIEW_PATHS)):
        raise ValueError('Missing, incomplete or stale visual review')
    _utc(review['reviewed_at_utc'])
    for name in PREVIEW_PATHS:
        expected = bundle['outputs'][f'images/{Path(name).name}']['sha256']
        if review['images'][name] != {
                'sha256': expected, 'full_resolution': True, 'article_width': True}:
            raise ValueError(f'Missing or stale visual review: {name}')
        if any(type(review['images'][name][key]) is not bool
               for key in ('full_resolution', 'article_width')):
            raise ValueError(f'Visual review requires Boolean checks: {name}')


def prepare_review(bundle: Path, root: Path, output: Path) -> Path:
    """Write a pending review template; this command never claims an image was inspected.

    Args:
        bundle: Complete C-local analytics bundle.
        root: Checkout/export whose source and registry must match.
        output: New C-local directory for the separate review record.

    Returns:
        Path to a pending review.json template.
    """
    root = _root(root)
    _registry(root)
    record = validate_bundle(bundle, root)
    review = {
        'schema_version': 1, 'kind': 'documentation_analytics_review', 'status': 'pending',
        'bundle_sha256': _hash(_encoded(record)), 'reviewer': '', 'reviewed_at_utc': None,
        'images': {name: {'sha256': record['outputs'][f'images/{Path(name).name}']['sha256'],
                          'full_resolution': False, 'article_width': False}
                   for name in PREVIEW_PATHS},
    }
    output = output_boundary(output, root)
    if output.is_relative_to(bundle.resolve()) or bundle.resolve().is_relative_to(output):
        raise ValueError('Review output must be separate from the candidate bundle')
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'review.json', review)
    return output / 'review.json'


@contextmanager
def _lock(root: Path):
    """Serialize publishers/recovery on this machine; OS locks release after process exit."""
    local = Path(os.environ.get('AGENT_LOCAL_ROOT', ''))
    directory = local / 'publication-locks'
    # Use the existing boundary checks without requiring this shared directory to be new.
    if directory.exists():
        output_boundary(directory, root, new=False)
    else:
        output_boundary(directory, root)
        directory.mkdir(parents=True, exist_ok=True)
    path = directory / (_hash(os.path.normcase(str(root)).encode('utf-8')) + '.lock')
    if _linked(path):
        raise ValueError('Linked publication lock is not allowed')
    with path.open('a+b') as stream:
        if path.stat().st_size == 0:
            stream.write(b'\0')
            stream.flush()
        stream.seek(0)
        try:
            if os.name == 'nt':
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            raise ValueError('Another publication or rollback holds the repository lock') from error
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == 'nt':
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _install(content: bytes, destination: Path, workspace: Path) -> None:
    """Stage and flush bytes on C, then atomically replace one destination on the same volume."""
    staged = workspace / f'write-{uuid.uuid4().hex}.tmp'
    with staged.open('xb') as stream:
        stream.write(content)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(staged, destination)


def _journal(backup: Path, record: dict) -> None:
    """Update a C-local recovery journal atomically."""
    _install(_encoded(record), backup / 'transaction.json', backup)


def _capture(root: Path) -> dict:
    """Read all allowlisted destinations after rechecking each path."""
    previous = {}
    for name in TARGETS:
        destination = _destination(root, name)
        previous[name] = destination.read_bytes() if destination.exists() else None
    return previous


def _assert_source(root: Path, record: dict) -> None:
    """Reject source or fixture drift while preparing/copying the publication."""
    if (_registry(root) != record['registry'] or source_fingerprint(root) != record['source']
            or input_files(record['registry'], root) != record['input_files']):
        raise ValueError('Source or fixture changed during publication')


def _publication(record: dict, review: dict) -> dict:
    """Wrap unchanged generation provenance with a separate reviewed publication receipt."""
    return {'schema_version': 1, 'kind': 'documentation_analytics_publication',
            'status': 'complete', 'published_at_utc': datetime.now(timezone.utc).isoformat(),
            'paths': list(PREVIEW_PATHS), 'bundle': record, 'review': review}


def verify_published(root: Path) -> dict:
    """Verify displayed PNGs against dated publication provenance, without rerunning analytics.

    Args:
        root: Repository containing the six displayed images and their publication manifest.

    Returns:
        Verified publication receipt. This does not certify freshness against current source.
    """
    root = _root(root)
    registry = _registry(root)
    record = read_json(_destination(root, PROVENANCE))
    if (set(record) != {'schema_version', 'kind', 'status', 'published_at_utc',
                       'paths', 'bundle', 'review'} or record['schema_version'] != 1
            or record['kind'] != 'documentation_analytics_publication'
            or record['status'] != 'complete' or record['paths'] != list(PREVIEW_PATHS)):
        raise ValueError('Incomplete publication provenance')
    _utc(record['published_at_utc'])
    bundle = record['bundle']
    if (bundle['registry'] != registry or bundle['schema_version'] != 1
            or bundle['kind'] != 'documentation_analytics_bundle'
            or bundle['status'] != 'complete' or bundle['review_status'] != 'pending'
            or bundle['publication_ready'] is not False
            or set(bundle['outputs']) != expected_files(registry)
            or set(bundle['producers']) != set(registry['producers'])):
        raise ValueError('Published bundle differs from the coverage registry')
    _check_review(record['review'], bundle)
    for name, spec in registry['producers'].items():
        check_result(bundle['producers'][name], spec['configuration'])
    for name in PREVIEW_PATHS:
        if file_record(_destination(root, name)) != bundle['outputs'][f'images/{Path(name).name}']:
            raise ValueError(f'Published image does not match provenance: {name}')
    return record


def _stored(backup: Path, phase: str, name: str, expected: str) -> bytes:
    """Read and verify one recovery payload without following links inside its backup."""
    directory = backup / phase
    path = directory / Path(name).name
    if _linked(directory) or _linked(path) or not path.resolve().is_relative_to(backup):
        raise ValueError('Linked or escaping recovery payload')
    content = path.read_bytes()
    if _hash(content) != expected:
        raise ValueError(f'Corrupted publication backup: {name}')
    return content


def _recover(backup: Path, root: Path) -> dict:
    """Restore a fully preflighted backup; leave another writer's conflicting bytes untouched."""
    backup = output_boundary(backup, root, new=False)
    journal_path = backup / 'transaction.json'
    if _linked(journal_path):
        raise ValueError('Linked recovery journal')
    record = read_json(journal_path)
    if (record.get('schema_version') != 1
            or record.get('kind') != 'documentation_analytics_publication_backup'
            or record.get('repo_root') != str(root)
            or record.get('status') not in {
                'prepared', 'publishing', 'complete', 'rolled_back', 'rollback_failed'}
            or set(record['before']) != set(TARGETS) or set(record['after']) != set(TARGETS)
            or any(record['before'][name] is None for name in PREVIEW_PATHS)):
        raise ValueError('Invalid or mismatched publication backup')
    before, after = {}, {}
    for name in TARGETS:
        old_hash = record['before'][name]
        before[name] = None if old_hash is None else _stored(backup, 'before', name, old_hash)
        after[name] = _stored(backup, 'after', name, record['after'][name])
    current = _capture(root)
    for name in TARGETS:
        if current[name] not in (before[name], after[name]):
            raise ValueError(f'Rollback would overwrite a concurrent edit: {name}')
    try:
        # Restore image bytes first, with the prior manifest (or its absence) last.
        for name in TARGETS:
            destination = _destination(root, name)
            observed = destination.read_bytes() if destination.exists() else None
            if observed == before[name]:
                continue
            if observed != after[name]:
                raise ValueError(f'Rollback would overwrite a concurrent edit: {name}')
            if before[name] is None:
                destination.unlink()
            else:
                _install(before[name], destination, backup)
        if _capture(root) != before:
            raise ValueError('Rollback verification failed')
        record['status'] = 'rolled_back'
        record.pop('error', None)
        _journal(backup, record)
    except BaseException as error:
        record['status'] = 'rollback_failed'
        record['error'] = f'{type(error).__name__}: {error}'
        _journal(backup, record)
        raise
    return record


def rollback(backup: Path, root: Path) -> dict:
    """Recover an interrupted publication or undo a completed one without replacing later edits.

    Args:
        backup: C-local backup directory printed by publication.
        root: The exact checkout/export recorded in that backup.

    Returns:
        Recovery journal with status rolled_back, after every original byte is verified.
    """
    root = _root(root)
    with _lock(root):
        return _recover(backup, root)


def publish_bundle(bundle: Path, root: Path, review_file: Path) -> dict:
    """Validate and publish the seven allowlisted deliverables with recoverable rollback.

    Args:
        bundle: Complete C-local generation directory.
        root: Destination checkout with the same effective source and registry.
        review_file: Separate C-local JSON record attesting review of every candidate image.

    Returns:
        Publication receipt after verifying all displayed previews.

    Raises:
        ValueError: If preflight, review, identity, paths or output validation fails.
        RuntimeError: If publication and automatic rollback both fail; use the retained backup.
    """
    root = _root(root)
    _registry(root)
    record = validate_bundle(bundle, root)
    output_boundary(review_file.parent, root, new=False)
    if _linked(review_file) or not review_file.is_file():
        raise ValueError('Visual review must be an ordinary C-local JSON file')
    review = read_json(review_file)
    _check_review(review, record)
    receipt = _publication(record, review)
    writes = {}
    for name in PREVIEW_PATHS:
        candidate = bundle / 'images' / Path(name).name
        content = candidate.read_bytes()
        if _hash(content) != record['outputs'][f'images/{candidate.name}']['sha256']:
            raise ValueError('Candidate changed after validation')
        writes[name] = content
    writes[PROVENANCE] = _encoded(receipt)
    with _lock(root):
        _assert_source(root, record)
        previous = _capture(root)
        backup = output_boundary(bundle.parent / f'publication-backup-{uuid.uuid4().hex}', root)
        backup.mkdir()
        if backup.stat().st_dev != root.stat().st_dev:
            raise ValueError('Publication requires C-local staging on the destination volume')
        for phase, payloads in (('before', previous), ('after', writes)):
            directory = backup / phase
            directory.mkdir()
            for name, content in payloads.items():
                if content is not None:
                    with (directory / Path(name).name).open('xb') as stream:
                        stream.write(content)
                        stream.flush()
                        os.fsync(stream.fileno())
        journal = {
            'schema_version': 1, 'kind': 'documentation_analytics_publication_backup',
            'repo_root': str(root), 'status': 'prepared',
            'before': {name: None if value is None else _hash(value)
                       for name, value in previous.items()},
            'after': {name: _hash(value) for name, value in writes.items()},
        }
        _journal(backup, journal)
        print(f'Publication backup: {backup}', flush=True)
        try:
            _assert_source(root, record)
            if _capture(root) != previous:
                raise ValueError('Publication destinations changed during preflight')
            journal['status'] = 'publishing'
            _journal(backup, journal)
            for name in TARGETS:
                destination = _destination(root, name)
                observed = destination.read_bytes() if destination.exists() else None
                if observed != previous[name]:
                    raise ValueError(f'Publication destination changed: {name}')
                if name == PROVENANCE:
                    _assert_source(root, record)
                    for preview in PREVIEW_PATHS:
                        if digest(_destination(root, preview)) != _hash(writes[preview]):
                            raise ValueError('Preview changed before manifest publication')
                _install(writes[name], destination, backup)
            _assert_source(root, record)
            verify_published(root)
            journal['status'] = 'complete'
            _journal(backup, journal)
        except BaseException as error:
            try:
                _recover(backup, root)
            except BaseException as recovery_error:
                journal['status'] = 'rollback_failed'
                journal['error'] = f'{type(recovery_error).__name__}: {recovery_error}'
                _journal(backup, journal)
                raise RuntimeError(
                    f'Publication failed ({error}); rollback incomplete. '
                    f'Preserve the backup and recover from {backup}: {recovery_error}'
                ) from recovery_error
            raise
    return receipt


def main(argv: list[str] | None = None) -> int:
    """Prepare visual review, publish, verify or recover only the explicit repository target."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo-root', type=Path, required=True)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument('--run-root', type=Path)
    actions.add_argument('--verify', action='store_true')
    actions.add_argument('--rollback', type=Path, metavar='BACKUP')
    review = parser.add_mutually_exclusive_group()
    review.add_argument('--review-file', type=Path)
    review.add_argument('--prepare-review', type=Path, metavar='NEW_DIRECTORY')
    args = parser.parse_args(argv)
    if bool(args.run_root) != bool(args.review_file or args.prepare_review):
        parser.error('--run-root requires --review-file or --prepare-review, and vice versa')
    try:
        if args.prepare_review:
            path = prepare_review(args.run_root, args.repo_root, args.prepare_review)
            print(f'Pending review template: {path}. Inspect every image before completing it.')
        elif args.run_root:
            record = publish_bundle(args.run_root, args.repo_root, args.review_file)
            print(f'Published and verified {len(record["paths"])} reviewed previews.')
        elif args.verify:
            record = verify_published(args.repo_root)
            print(f'Verified {len(record["paths"])} published previews against dated provenance.')
        else:
            rollback(args.rollback, args.repo_root)
            print('Rollback verified; all previous destination bytes restored.')
        return 0
    except (ValueError, OSError, KeyError, TypeError, RuntimeError,
            ImportError, csv.Error) as error:
        print(f'Publication error: {error}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
