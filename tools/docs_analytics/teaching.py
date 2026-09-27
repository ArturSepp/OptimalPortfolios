"""Generate, publish and verify the teaching exhibits of the documentation articles.

A teaching exhibit is a synthetic figure drawn by the canonical script of the page that shows it,
``examples/docs/<page>.py``. The exhibits are registered in ``tools/docs_analytics/teaching.json``,
separately from the six README previews of ``registry.json``, whose publication receipt embeds
that registry verbatim. For each exhibit the registry names the script, the function that draws
it, the pages that display it, and the constants of the script that fix its inputs.

``--all`` runs every exhibit into a new C-local directory. It loads each script without running its
``__main__`` block, compares the registered parameters with the script's constants, calls the
drawing function, which returns its plotted table and the numerical checks it asserts, and writes
the PNG, a CSV of the table and a manifest with the hashes of the image, table and script and the
software versions. ``--publish`` copies a complete bundle whose checks all pass into ``docs/images``
together with the manifest and a review note, after the images have been inspected.
``--verify`` checks that every committed image matches the committed manifest and that the
manifest was produced from the current scripts, so an edited script requires a regenerated
exhibit. It reads files only and imports nothing from the numerical stack.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path, PurePosixPath
import platform
import re
import runpy
import shutil
import sys
from typing import Optional, Sequence

from tools.docs_analytics.registry import ROOT, relative_path, source_file

TEACHING_PATH = 'tools/docs_analytics/teaching.json'
IMAGES_DIR = PurePosixPath('docs/images')
MANIFEST = 'docs/images/analytics_manifest.json'
PACKAGES = ('optimalportfolios', 'qis', 'factorlasso', 'numpy', 'pandas', 'scipy', 'cvxpy',
            'matplotlib')


def sha256(path: Path) -> str:
    """Return the hex SHA-256 of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def text_sha256(path: Path) -> str:
    """Return the SHA-256 of a text file with CRLF normalised to LF.

    Scripts are stored with LF, but a checkout may convert line endings; the hash of a script
    must not depend on the platform that verifies it.
    """
    return hashlib.sha256(path.read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def load_teaching(root: Path = ROOT) -> list[dict]:
    """Read and validate the teaching-exhibit registry.

    Args:
        root: Repository checkout or C-local source export.

    Returns:
        The exhibit records, or an empty list when the registry file does not exist.

    Raises:
        ValueError: If an exhibit record, path, script or consumer is invalid.
    """
    path = root / TEACHING_PATH
    if not path.is_file():
        return []
    registry = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(registry, dict) or registry.get('schema_version') != 1:
        raise ValueError('Unsupported teaching registry schema')
    exhibits = registry.get('exhibits')
    if not isinstance(exhibits, list):
        raise ValueError('The teaching registry needs an exhibits list')
    ids, paths = set(), set()
    for exhibit in exhibits:
        if (not isinstance(exhibit, dict) or not isinstance(exhibit.get('id'), str)
                or not re.fullmatch(r'[a-z][a-z0-9_]*', exhibit['id'])):
            raise ValueError('Invalid teaching exhibit identity')
        image = relative_path(exhibit.get('path'))
        if image.parent != IMAGES_DIR or image.name != f"{exhibit['id']}.png":
            raise ValueError(f'A teaching exhibit is docs/images/<id>.png: {image}')
        script = relative_path(exhibit.get('script'))
        if script.parent != PurePosixPath('examples/docs') or script.suffix != '.py':
            raise ValueError(f'A teaching exhibit is drawn by an examples/docs script: {script}')
        source_file(root, str(script))
        if not re.fullmatch(r'[a-z_][a-z0-9_]*', str(exhibit.get('function', ''))):
            raise ValueError(f'Invalid drawing function: {exhibit["id"]}')
        if not isinstance(exhibit.get('parameters'), dict) or not exhibit['parameters']:
            raise ValueError(f'Register the fixed inputs of the exhibit: {exhibit["id"]}')
        consumers = exhibit.get('documents')
        if (not isinstance(consumers, list) or not consumers
                or len(consumers) != len(set(consumers))):
            raise ValueError(f'Invalid consumer list: {exhibit["id"]}')
        for document in consumers:
            source_file(root, document)
        for field in ('question', 'sample'):
            if not str(exhibit.get(field, '')).strip():
                raise ValueError(f'The exhibit needs a {field}: {exhibit["id"]}')
        if exhibit['id'] in ids or str(image) in paths:
            raise ValueError(f'Duplicate teaching exhibit: {exhibit["id"]}')
        ids.add(exhibit['id'])
        paths.add(str(image))
    return exhibits


def _plain(value):
    """Convert numpy scalars and arrays to plain Python values for comparison."""
    if hasattr(value, 'tolist'):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    return value


def check_parameters(exhibit: dict, namespace: dict) -> None:
    """Require each registered parameter to equal the constant of the same name in the script."""
    for name, expected in exhibit['parameters'].items():
        if name not in namespace:
            raise ValueError(f'{exhibit["id"]}: the script defines no constant {name}')
        if _plain(namespace[name]) != expected:
            raise ValueError(f'{exhibit["id"]}: {name} differs from the registered value')


def versions() -> dict:
    """Return the installed versions of the packages an exhibit depends on."""
    found = {'python': platform.python_version()}
    for package in PACKAGES:
        try:
            found[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            found[package] = None
    return found


def generate(output: Path, root: Path = ROOT) -> Path:
    """Draw every registered exhibit into a new directory and write its manifest.

    Args:
        output: A directory that does not exist yet.
        root: Repository checkout or C-local source export.

    Returns:
        The manifest path.

    Raises:
        ValueError: If the output exists, a parameter differs, or a check fails.
    """
    exhibits = load_teaching(root)
    if output.exists():
        raise ValueError(f'Refusing to reuse an existing output directory: {output}')
    (output / 'images').mkdir(parents=True)
    (output / 'tables').mkdir()
    import matplotlib

    records = {}
    sys.path.insert(0, str(root))
    for exhibit in exhibits:
        script = root / exhibit['script']
        namespace = runpy.run_path(str(script), run_name='__docs_exhibit__')
        check_parameters(exhibit, namespace)
        image = output / 'images' / f'{exhibit["id"]}.png'
        # Each exhibit starts from the same style: settings one exhibit changes are restored
        # before the next, so an image does not depend on the registry order.
        with matplotlib.rc_context():
            result = namespace[exhibit['function']](image)
        checks = {name: bool(value) for name, value in result['checks'].items()}
        if not checks or not all(checks.values()):
            raise ValueError(f'{exhibit["id"]}: failed checks {checks}')
        table = output / 'tables' / f'{exhibit["id"]}.csv'
        result['table'].to_csv(table, lineterminator='\n')
        records[exhibit['id']] = {
            'path': exhibit['path'], 'script': exhibit['script'],
            'script_sha256': text_sha256(script), 'image_sha256': sha256(image),
            'table_sha256': sha256(table), 'parameters': exhibit['parameters'],
            'checks': checks, 'documents': exhibit['documents'],
        }
    manifest = {
        'schema_version': 1, 'kind': 'documentation_teaching_exhibits',
        'generated_at_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
        'platform': platform.platform(), 'versions': versions(), 'exhibits': records,
        'review': None,
    }
    path = output / 'analytics_manifest.json'
    path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8', newline='\n')
    return path


def publish(bundle: Path, review: str, root: Path = ROOT) -> Path:
    """Copy a complete, passing bundle into docs/images with the review note.

    Args:
        bundle: Directory written by ``generate``.
        review: What was inspected, by whom, at which resolutions.
        root: Repository checkout to update.

    Returns:
        The published manifest path.
    """
    if not review.strip():
        raise ValueError('Publication needs a review note')
    manifest = json.loads((bundle / 'analytics_manifest.json').read_text(encoding='utf-8'))
    exhibits = {exhibit['id']: exhibit for exhibit in load_teaching(root)}
    if set(manifest['exhibits']) != set(exhibits):
        raise ValueError('The bundle does not cover exactly the registered exhibits')
    for key, record in manifest['exhibits'].items():
        image = bundle / 'images' / f'{key}.png'
        if (sha256(image) != record['image_sha256']
                or record['script_sha256'] != text_sha256(root / exhibits[key]['script'])
                or not all(record['checks'].values())):
            raise ValueError(f'{key}: the bundle is stale, altered or failing')
    target = root / IMAGES_DIR
    target.mkdir(parents=True, exist_ok=True)
    for key in manifest['exhibits']:
        shutil.copyfile(bundle / 'images' / f'{key}.png', target / f'{key}.png')
    manifest['review'] = {
        'note': review.strip(),
        'published_at_utc': datetime.now(timezone.utc).isoformat(timespec='seconds'),
    }
    path = root / MANIFEST
    path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8', newline='\n')
    return path


def verify(root: Path = ROOT) -> dict:
    """Check committed exhibits against their manifest and the current scripts.

    Args:
        root: Repository checkout or C-local source export.

    Returns:
        The committed manifest.

    Raises:
        ValueError: If an image, script or registry entry differs from the manifest.
    """
    exhibits = {exhibit['id']: exhibit for exhibit in load_teaching(root)}
    path = root / MANIFEST
    if not exhibits:
        if path.exists():
            raise ValueError('A teaching manifest exists without registered exhibits')
        return {}
    manifest = json.loads(path.read_text(encoding='utf-8'))
    if (manifest.get('kind') != 'documentation_teaching_exhibits'
            or set(manifest.get('exhibits', {})) != set(exhibits)
            or not isinstance(manifest.get('review'), dict)):
        raise ValueError('The teaching manifest does not cover the registered exhibits')
    for key, record in manifest['exhibits'].items():
        exhibit = exhibits[key]
        image = source_file(root, exhibit['path'])
        if sha256(image) != record['image_sha256']:
            raise ValueError(f'{key}: the committed image differs from its manifest')
        if text_sha256(root / exhibit['script']) != record['script_sha256']:
            raise ValueError(f'{key}: the script changed since the exhibit was drawn; regenerate')
        if record['parameters'] != exhibit['parameters'] or not all(record['checks'].values()):
            raise ValueError(f'{key}: registered parameters or checks differ from the manifest')
    images = {p.name for p in (root / IMAGES_DIR).glob('*.png')}
    if images != {f'{key}.png' for key in exhibits}:
        raise ValueError(f'Unregistered or missing images in docs/images: {sorted(images)}')
    return manifest


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the command-line interface."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--repo-root', type=Path, default=ROOT)
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument('--list', action='store_true', help='Validate and list the exhibits.')
    actions.add_argument('--all', action='store_true', help='Draw every exhibit.')
    actions.add_argument('--publish', type=Path, metavar='RUN_ROOT',
                         help='Publish a reviewed bundle into docs/images.')
    actions.add_argument('--verify', action='store_true', help='Verify committed exhibits.')
    parser.add_argument('--output-root', type=Path, help='New directory for --all.')
    parser.add_argument('--review', default='', help='Review note for --publish.')
    args = parser.parse_args(argv)
    root = args.repo_root.resolve()
    if args.list:
        for exhibit in load_teaching(root):
            print(f'{exhibit["id"]}: {exhibit["script"]} -> {exhibit["path"]}')
        print(f'{len(load_teaching(root))} teaching exhibits.')
    elif args.all:
        if args.output_root is None:
            parser.error('--all needs --output-root')
        print(f'Wrote {generate(args.output_root, root)}')
    elif args.publish:
        print(f'Published {publish(args.publish, args.review, root)}')
    else:
        manifest = verify(root)
        print(f'Verified {len(manifest.get("exhibits", {}))} teaching exhibits against their '
              'manifest and current scripts.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
