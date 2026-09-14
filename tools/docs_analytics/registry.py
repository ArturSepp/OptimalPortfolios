"""Read the documentation image registry without importing analytical code.

Coverage parsing adapts QuantInvestStrats/tools/docs_analytics/run.py (MIT, Artur Sepp).
This local implementation also scans README, RST and source-adjacent contributor notes.
"""

from html.parser import HTMLParser
import json
from pathlib import Path, PurePosixPath
import re
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = Path('tools/docs_analytics/registry.json')


def relative_path(value: str) -> PurePosixPath:
    """Require a canonical repository-relative path with portable separators."""
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f'Expected a nonempty relative path: {value!r}')
    path = PurePosixPath(value)
    if (path.is_absolute() or '\\' in value or ':' in value
            or any(ord(char) < 32 for char in value)
            or any(part in ('', '.', '..') for part in value.split('/'))):
        raise ValueError(f'Unsafe relative path: {value!r}')
    return path


def source_file(root: Path, value: str) -> Path:
    """Resolve an existing file with exact case, rejecting source-tree escapes."""
    relative = relative_path(value)
    path = root
    for part in relative.parts:
        if not path.is_dir() or part not in {entry.name for entry in path.iterdir()}:
            raise ValueError(f'Missing source file or incorrect case: {value}')
        path = path / part
    if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError(f'Missing file or source-tree escape: {value}')
    return path


def image_references(text: str) -> list[str]:
    """Read common Markdown, HTML, MyST and RST image references outside code examples."""
    text = re.sub(r'<!--.*?-->', '', text, flags=re.S)
    text = re.sub(
        r'(?m)^\.\. (?:code-block|code)::[^\n]*\n(?:[ \t]+[^\n]*\n|\n)*', '', text)
    visible, references = [], []
    fence = None
    for line in text.splitlines():
        match = re.match(r'^\s*(\x60{3,}|~{3,})(.*)$', line)
        if fence:
            if match and match[1][0] == fence[0] and len(match[1]) >= len(fence):
                fence = None
            continue
        if match:
            fence = match[1]
            directive = re.match(r'\{(?:image|figure)\}\s+(\S+)', match[2])
            if directive:
                references.append(directive[1])
            continue
        directive = re.match(
            r'^\s*(?::{3,}\{(?:image|figure)\}|\.\. (?:\|[^|]+\| )?(?:image|figure)::)'
            r'\s+(\S+)', line)
        if directive:
            references.append(directive[1])
        visible.append(line)
    prose = '\n'.join(visible)
    prose = re.sub(r'(\x60+)[^\n]*?\1', '', prose)
    references += re.findall(r'!\[[^\]]*\]\(\s*<?([^\s)>]+)>?(?:\s+[^)]*)?\)', prose)

    class ImageParser(HTMLParser):
        """Collect HTML image sources, including self-closing tags."""

        def handle_starttag(self, tag, attrs):
            """Append each visible img source."""
            if tag.lower() == 'img' and dict(attrs).get('src'):
                references.append(dict(attrs)['src'])

    ImageParser().feed(prose)
    definitions = {
        key.strip().casefold(): url for key, url in re.findall(
            r'^\s*\[([^\]]+)\]:\s*<?([^\s>]+)>?', prose, flags=re.M)
    }
    for match in re.finditer(r'!\[([^\]]+)\](?:\[([^\]]*)\])?(?!\()', prose):
        label = (match[2] or match[1]).strip().casefold()
        if label not in definitions:
            raise ValueError(f'Unresolved image reference: {label}')
        references.append(definitions[label])
    return references


def documents(root: Path) -> list[Path]:
    """Discover human documentation while excluding generated Sphinx directories."""
    paths = {source_file(root, 'README.md')}
    excluded = {'generated', '_templates', '_build'}
    for path in (root / 'docs').rglob('*'):
        relative = path.relative_to(root / 'docs')
        if (path.is_file() and path.suffix in {'.md', '.rst'}
                and not any(part in excluded for part in relative.parts)):
            paths.add(path)
    paths.update((root / 'src').rglob('README.md'))
    return sorted(paths)


def check_coverage(registry: dict, root: Path = ROOT) -> None:
    """Require explicit classification for every displayed image and reject unused records."""
    expected = {(document, asset['path'])
                for asset in registry['assets'] for document in asset['documents']}
    expected |= {(item['document'], item['url']) for item in registry['non_analytics']}
    observed = set()
    for document in documents(root):
        name = document.relative_to(root).as_posix()
        for url in image_references(document.read_text(encoding='utf-8')):
            parsed = urlsplit(url)
            if parsed.scheme or parsed.netloc:
                target = url
            else:
                if parsed.path.startswith(('/', '\\')):
                    raise ValueError(f'Absolute image path: {name}: {url}')
                resolved = (document.parent / unquote(parsed.path)).resolve()
                if not resolved.is_relative_to(root.resolve()):
                    raise ValueError(f'Image escapes source tree: {name}: {url}')
                # Preserve link spelling on Windows so capitalization drift remains visible.
                target = resolved.relative_to(root.resolve()).as_posix()
            observed.add((name, target))
    if observed != expected:
        raise ValueError(
            f'Image coverage mismatch: unregistered={sorted(observed - expected)}, '
            f'unused={sorted(expected - observed)}')


def _unique_object(pairs: list) -> dict:
    """Reject duplicate JSON keys instead of silently taking the final value."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'Duplicate registry key: {key}')
        result[key] = value
    return result


def load_registry(root: Path = ROOT) -> dict:
    """Validate the coverage ledger without loading legacy or future producers.

    Args:
        root: Repository checkout or C-local source export.

    Returns:
        Parsed registry with verified local paths and complete image coverage.

    Raises:
        ValueError: If the registry, asset ownership or image coverage is invalid.
    """
    registry = json.loads(source_file(root, REGISTRY_PATH.as_posix()).read_text(
        encoding='utf-8'), object_pairs_hook=_unique_object)
    if not isinstance(registry, dict) or registry.get('schema_version') != 1:
        raise ValueError('Unsupported analytics registry schema')
    producers = registry.get('producers')
    assets = registry.get('assets')
    badges = registry.get('non_analytics')
    if (not isinstance(producers, dict) or not producers or not isinstance(assets, list)
            or not assets or not isinstance(badges, list)):
        raise ValueError('Registry needs producers, assets and non_analytics collections')
    for name, spec in producers.items():
        if (not re.fullmatch(r'[a-z][a-z0-9_]*', name) or not isinstance(spec, dict)
                or spec.get('module') != f'tools.docs_analytics.{name}'
                or spec.get('callable') != 'produce'):
            raise ValueError(f'Invalid producer entry point: {name}')
        if spec.get('status') not in {'pending', 'implemented'}:
            raise ValueError(f'Invalid producer status: {name}')
        config = spec.get('configuration')
        if config is not None and (not isinstance(config, dict) or not config):
            raise ValueError(f'Invalid producer configuration: {name}')
        if spec['status'] == 'implemented':
            if config is None:
                raise ValueError(f'Implemented producer lacks configuration: {name}')
            source_file(root, f'tools/docs_analytics/{name}.py')
        if not spec.get('legacy_sources') or not spec.get('fixture_candidate'):
            raise ValueError(f'Missing legacy source or fixture decision: {name}')
        for legacy in spec['legacy_sources']:
            source_file(root, legacy)
    ids, paths, used = set(), set(), set()
    for asset in assets:
        if (not isinstance(asset, dict) or not isinstance(asset.get('id'), str)
                or not re.fullmatch(r'[a-z][a-z0-9_]*', asset['id'])):
            raise ValueError('Invalid asset identity')
        path = relative_path(asset.get('path'))
        if path.parent != PurePosixPath('examples/figures') or path.suffix != '.PNG':
            raise ValueError(f'Asset outside the preview paths: {path}')
        if asset['id'] in ids or str(path).casefold() in paths:
            raise ValueError(f'Duplicate asset identity/path: {path}')
        ids.add(asset['id'])
        paths.add(str(path).casefold())
        if asset.get('producer') not in producers or asset.get('status') != 'legacy':
            raise ValueError(f'Invalid producer or legacy asset status: {path}')
        used.add(asset['producer'])
        consumers = asset.get('documents')
        if (not isinstance(consumers, list) or not consumers
                or len(consumers) != len(set(consumers))):
            raise ValueError(f'Invalid consumer list: {path}')
        source_file(root, str(path))
        for document in consumers:
            source_file(root, document)
    if used != set(producers):
        raise ValueError('Registry contains an unused producer')
    classified = set()
    for item in badges:
        if (not isinstance(item, dict) or not item.get('reason')
                or not isinstance(item.get('url'), str)):
            raise ValueError('Non-analytics images need a URL and reason')
        parsed = urlsplit(item['url'])
        if parsed.scheme != 'https' or not parsed.netloc:
            raise ValueError('Non-analytics classification requires an HTTPS image URL')
        source_file(root, item['document'])
        key = (item['document'], item['url'])
        if key in classified:
            raise ValueError('Duplicate non-analytics classification')
        classified.add(key)
    check_coverage(registry, root)
    return registry
