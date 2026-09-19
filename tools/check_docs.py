"""Check portable OptimalPortfolios documentation without importing the numerical stack.

Adapted from QuantInvestStrats/tools/check_docs.py (Artur Sepp, MIT).
Default: validate adopted pages and report pending migration. --files checks a batch;
--source-all checks all human sources without adopting them; --all requires complete adoption.
Renderer, numerical and external-link checks are separate.
"""

import argparse
from datetime import date
import json
import re
from pathlib import Path
from urllib.parse import unquote, urlsplit
from typing import NamedTuple, Optional, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_URL = 'https://github.com/ArturSepp/OptimalPortfolios'
CITATION_URL = f'{PROJECT_URL}/blob/main/CITATION.cff'
ARTICLE_HEADINGS = (
    'Overview',
    'Inputs, notation, and assumptions',
    'Methodology',
    'Worked example',
    'Implementation in optimalportfolios',
    'Interpretation and limitations',
    'See also',
    'References',
)
FENCE = re.compile(r'^ {0,3}(`{3,}|~{3,})(.*)$')
HEADING = re.compile(r'^(#{1,6})\s+(.+?)\s*#*\s*$')
LINK = re.compile(r'(?<!!)\[[^\]\n]+\]\((https://[^\s)]+)\)')
BYLINE = re.compile(
    r'^\*Author: \[[^\]\n]+\]\(https://github\.com/[A-Za-z0-9-]+\)'
    r'(?: / First recorded: \[(?P<date>\d{4}-\d{2}-\d{2})\]'
    rf'\({re.escape(PROJECT_URL)}/commit/[0-9a-f]{{40}}\))?\*$'
)


class Issue(NamedTuple):
    """One source-level documentation problem.

    Attributes:
        line: One-based source line, or one for a document-wide problem.
        message: Explanation of the violated convention.
    """

    line: int
    message: str


def prose_lines(text: str) -> tuple[list[tuple[int, str]], list[Issue], str]:
    """Extract prose while respecting YAML front matter and nested example fences.

    Args:
        text: Markdown source; no files are modified.

    Returns:
        Visible lines, syntax issues, and the front-matter body.
    """
    lines = text.splitlines()
    issues = []
    metadata = ''
    first = 0
    if lines and lines[0] == '---':
        closing = next((i for i in range(1, len(lines)) if lines[i] == '---'), None)
        if closing is None:
            return [], [Issue(1, 'Unclosed YAML front matter.')], ''
        metadata = '\n'.join(lines[1:closing])
        first = closing + 1
    visible = []
    fence = ''
    fence_line = 0
    math_line = 0
    comment = False
    for index in range(first, len(lines)):
        line = lines[index]
        number = index + 1
        matched = FENCE.match(line)
        if fence:
            if (matched and matched[1][0] == fence[0] and len(matched[1]) >= len(fence)
                    and not matched[2].strip()):
                fence = ''
            continue
        if math_line:
            if line.strip() == '$$':
                math_line = 0
                if index + 1 < len(lines) and lines[index + 1].strip():
                    issues.append(Issue(number, 'Put a blank line after display mathematics.'))
            continue
        # HTML comments must not satisfy a missing byline, section, or citation.
        if comment:
            if '-->' in line:
                comment = False
                line = line.split('-->', 1)[1]
            else:
                continue
        while '<!--' in line:
            before, after = line.split('<!--', 1)
            if '-->' in after:
                line = before + after.split('-->', 1)[1]
            else:
                line = before
                comment = True
        matched = FENCE.match(line)
        if matched:
            fence, fence_line = matched[1], number
            if re.match(r'^(?:\{math\}|math)(?:\s|$)', matched[2].strip()):
                issues.append(Issue(number, 'Use standalone $$ display blocks, not math fences.'))
            continue
        if line.startswith(('    ', '\t', '>')):
            continue  # indented code and quoted examples are not article structure
        # Inline code can demonstrate a delimiter without being a display expression.
        for code in re.finditer(r'(`+).*?\1', line):
            if re.search(r'\{(?:math|eq)\}$', line[:code.start()]):
                issues.append(Issue(number, 'Use dollar math and ordinary equation-section links.'))
        without_code = re.sub(r'(`+).*?\1', '', line)
        if '$$' in without_code:
            if line.strip() == '$$':
                math_line = number
                if index > 0 and lines[index - 1].strip():
                    issues.append(Issue(number, 'Put a blank line before display mathematics.'))
            else:
                issues.append(Issue(number, 'Put each display $$ delimiter on its own line.'))
            continue
        if re.search(r'\\[\[\]()]', without_code):
            issues.append(Issue(number, 'Use dollar delimiters for portable mathematics.'))
        if len(re.findall(r'(?<!\\)\$', without_code)) % 2:
            issues.append(Issue(number, 'Unclosed inline mathematics; pair dollars on one line.'))
        visible.append((number, line))
    if fence:
        issues.append(Issue(fence_line, 'Unclosed code fence.'))
    if math_line:
        issues.append(Issue(math_line, 'Unclosed display mathematics.'))
    if comment:
        issues.append(Issue(len(lines), 'Unclosed HTML comment.'))
    return visible, issues, metadata


def valid_byline(line: str) -> bool:
    """Check linked attribution and an optional, calendar-valid repository-evidence date."""
    match = BYLINE.fullmatch(line)
    if match is None:
        return False
    if match['date'] is not None:
        try:
            date.fromisoformat(match['date'])
        except ValueError:
            return False
    return True


def check_document(text: str, *, methodology: bool) -> list[Issue]:
    """Check one article's metadata, structure, byline, references, and math source.

    Args:
        text: Complete Markdown source.
        methodology: Whether the full methodology section order is required.

    Returns:
        Source issues. An empty list means these source checks passed, not that TeX rendered.
    """
    visible, issues, metadata = prose_lines(text)
    description = re.search(r'(?m)^[ \t]+description:[ \t]*([^\n]*)', metadata)
    if not description or 'html_meta:' not in metadata or 'myst:' not in metadata:
        issues.append(Issue(1, 'Provide a myst.html_meta.description in front matter.'))
    elif description[1].strip() in ('', "''", '\"\"'):
        issues.append(Issue(1, 'The page description must not be empty.'))
    elif description[1].strip() in ('>', '>-', '|', '|-'):
        following = metadata[description.end():].strip()
        if not following:
            issues.append(Issue(1, 'The page description must not be empty.'))
    headings = [(number, len(match[1]), match[2]) for number, line in visible
                if (match := HEADING.match(line))]
    titles = [heading for heading in headings if heading[1] == 1]
    if len(titles) != 1 or not headings or headings[0][1] != 1:
        issues.append(Issue(1, 'Start with exactly one H1 title.'))
    previous = 0
    for number, level, _ in headings:
        if level > previous + 1:
            issues.append(Issue(number, 'Do not skip heading levels.'))
        previous = level
    title_line = titles[0][0] if titles else 0
    opening = [line for number, line in visible if title_line < number <= title_line + 12]
    if not any(valid_byline(line) for line in opening):
        issues.append(Issue(title_line or 1,
                            'Use a linked author byline; any date needs a valid commit link.'))
    prose = '\n'.join(line for _, line in visible)
    links = set(LINK.findall(re.sub(r'(`+).*?\1', '', prose)))
    if PROJECT_URL not in links and PROJECT_URL + '/' not in links:
        issues.append(Issue(1, 'Link to the OptimalPortfolios project repository in prose.'))
    if CITATION_URL not in links:
        issues.append(Issue(1, 'Link to the canonical OptimalPortfolios CITATION.cff in prose.'))
    if methodology:
        sections = [title for _, level, title in headings if level == 2]
        if sections != list(ARTICLE_HEADINGS):
            issues.append(Issue(1, 'Use each required methodology H2 once, in the standard order.'))
    return issues



def check_local_links(text: str, path: Path, root: Path) -> list[Issue]:
    """Check local inline and reference-style Markdown file links.

    Fragment targets, external URLs, and HTML attributes require Sphinx or a viewer.
    Code examples and comments are excluded in the same way as article structure.
    """
    visible, _, _ = prose_lines(text)
    issues = []
    for number, line in visible:
        prose = re.sub(r'(`+).*?\1', '', line)
        links = re.findall(r'\[[^\]\n]*\]\((<[^>\n]+>|[^\s)]+)(?:\s+[^)]*)?\)', prose)
        definition = re.match(r'^\s{0,3}\[[^\]^]+\]:\s*(<[^>\n]+>|\S+)', prose)
        if definition:
            links.append(definition[1])
        for link in links:
            try:
                target = urlsplit(link.strip('<>'))
            except ValueError:
                issues.append(Issue(number, f'Invalid link target: {link}'))
                continue
            if target.scheme or target.netloc or not target.path:
                continue
            decoded = unquote(target.path)
            destination = (root / decoded.lstrip('/') if decoded.startswith('/')
                           else path.parent / decoded).resolve()
            if not destination.is_relative_to(root.resolve()):
                issues.append(Issue(number, f'Local link leaves the repository: {link}'))
                continue
            candidates = [destination]
            if destination.suffix == '.html':
                candidates.extend(destination.with_suffix(suffix) for suffix in ('.md', '.rst'))
            if not any(candidate.exists() for candidate in candidates):
                issues.append(Issue(number, f'Missing local link target: {link}'))
    return issues


def load_inventory(root: Path) -> tuple[dict, list[str]]:
    """Read and validate explicit page ownership; never infer adoption from page content."""
    path = root / 'tools' / 'docs_inventory.json'
    try:
        inventory = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        return {}, [f'{path}:1: Cannot read documentation inventory: {exc}']
    if not isinstance(inventory, dict) or inventory.get('schema_version') != 1:
        return {}, ['tools/docs_inventory.json:1: Expected inventory schema_version 1.']
    pages = inventory.get('pages')
    excluded = inventory.get('excluded_doc_roots')
    if not isinstance(pages, dict) or not pages or not isinstance(excluded, dict):
        return {}, ['tools/docs_inventory.json:1: Provide nonempty pages and excluded_doc_roots.']
    errors = []
    allowed_exclusions = {'docs/generated', 'docs/_templates', 'docs/_build'}
    if set(excluded) != allowed_exclusions or not all(excluded.values()):
        errors.append('tools/docs_inventory.json:1: Keep explicit generated/template/build roots.')
    for name, entry in pages.items():
        relative = Path(name)
        if (relative.is_absolute() or '..' in relative.parts or '\\' in name
                or relative.suffix not in {'.md', '.rst'}):
            errors.append(f'{name}:1: Inventory paths must be repository-relative Markdown or RST.')
            continue
        if not isinstance(entry, dict):
            errors.append(f'{name}:1: Invalid page ownership record.')
            continue
        pair = entry.get('form'), entry.get('status')
        if pair not in {
            ('methodology', 'pending'), ('methodology', 'adopted'),
            ('utility', 'pending'), ('utility', 'adopted'), ('api', 'api'),
        }:
            errors.append(f'{name}:1: Invalid page form/adoption status.')
        if entry.get('status') == 'api' and name != 'docs/api.rst':
            errors.append(f'{name}:1: Only the autosummary entry docs/api.rst has API ownership.')
        if entry.get('status') == 'adopted' and relative.suffix != '.md':
            errors.append(f'{name}:1: Adopted human pages must use portable Markdown.')
        if not (root / relative).is_file():
            errors.append(f'{name}:1: Missing documentation page.')
    return inventory, errors


def discover_pages(root: Path) -> set[str]:
    """Find reader-facing docs and contributor READMEs, excluding known generated sources."""
    found = {'README.md'} if (root / 'README.md').is_file() else set()
    excluded = {'generated', '_templates', '_build'}
    for path in (root / 'docs').rglob('*'):
        if path.is_file() and path.suffix in {'.md', '.rst'}:
            relative = path.relative_to(root / 'docs')
            if relative.parts[0] not in excluded:
                found.add(path.relative_to(root).as_posix())
    for path in (root / 'src').rglob('README.md'):
        found.add(path.relative_to(root).as_posix())
    return found


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Validate selected sources, retaining --all as the complete-adoption gate.

    Args:
        argv: Optional command-line arguments, excluding the program name.

    Returns:
        Zero for a passing selection, one for failed checks; invalid CLI paths raise SystemExit.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument('--files', nargs='+', type=Path, help='Human pages to validate now.')
    selection.add_argument('--all', action='store_true', help='Require complete adoption.')
    selection.add_argument('--source-all', action='store_true',
                           help='Check all human sources without claiming adoption.')
    args = parser.parse_args(argv)
    inventory, errors = load_inventory(REPO_ROOT)
    if errors:
        for error in errors:
            print(error)
        return 1
    pages = inventory['pages']
    discovered = discover_pages(REPO_ROOT)
    for name in sorted(discovered - pages.keys()):
        errors.append(f'{name}:1: Add this page to the explicit documentation inventory.')
    for name in sorted(pages.keys() - discovered):
        errors.append(f'{name}:1: Inventory entry is outside the reader-facing discovery scope.')
    stems = {}
    for name in sorted(discovered):
        if name.startswith('docs/'):
            stem = str(Path(name).with_suffix(''))
            if stem in stems:
                errors.append(f'{name}:1: Duplicate Sphinx source basename with {stems[stem]}.')
            stems[stem] = name
    human = {name for name, entry in pages.items() if entry['status'] != 'api'}
    adopted = {name for name in human if pages[name]['status'] == 'adopted'}
    pending = human - adopted
    if args.files:
        selected = set()
        for requested in args.files:
            path = (REPO_ROOT / requested).resolve()
            if not path.is_relative_to(REPO_ROOT.resolve()):
                parser.error(f'Expected a repository-relative documentation page: {requested}')
            name = path.relative_to(REPO_ROOT.resolve()).as_posix()
            if name not in human:
                parser.error(f'Expected an inventoried human documentation page: {requested}')
            selected.add(name)
    else:
        selected = human if args.all or args.source_all else adopted
    if args.all:
        for name in sorted(pending):
            errors.append(f'{name}:1: Pending migration; mark adopted only after verification.')
    for name in sorted(selected):
        path = REPO_ROOT / name
        if path.suffix != '.md':
            errors.append(f'{name}:1: Migrate human RST to Markdown before adoption.')
            continue
        source = path.read_text(encoding='utf-8')
        issues = check_document(source, methodology=pages[name]['form'] == 'methodology')
        issues.extend(check_local_links(source, path, REPO_ROOT))
        for issue in issues:
            errors.append(f'{name}:{issue.line}: {issue.message}')
    for error in errors:
        print(error)
    print(f'{"FAIL" if errors else "PASS"}: {len(selected)} selected human pages; '
          f'{len(errors)} issues; {len(pages) - len(human)} API entry.')
    if pending:
        print(f'PENDING (not adopted): {len(pending)} pages: {", ".join(sorted(pending))}')
    return 1 if errors else 0


if __name__ == '__main__':
    raise SystemExit(main())
