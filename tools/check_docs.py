"""Check portable OptimalPortfolios documentation without importing the numerical stack.

Adapted from QuantInvestStrats/tools/check_docs.py (Artur Sepp, MIT); the ownership, case-study,
excerpt and paper-ledger rules follow FactorLasso/tools/check_docs.py.
Default: validate adopted pages and report pending migration. --files checks a batch;
--source-all checks all human sources without adopting them; --all requires complete adoption.
Every mode also checks the repository-wide contracts of the inventory: each public object and
each configuration field of the mapped dataclasses has exactly one owning page, and no retired
paper title appears in reader-facing text. The public surface and the dataclass fields are read
from the package source with ``ast``, so the checker still imports nothing from the stack.
Renderer, numerical and external-link checks are separate.
"""

import argparse
import ast
from datetime import date
import json
import re
import textwrap
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
CASE_STUDY_HEADINGS = (
    'Overview',
    'Study design and data',
    'Configuration',
    'Results',
    'What the study does and does not show',
    'Reproduce',
    'See also',
    'References',
)
CONVENTION_SECTION = 'Inputs, notation, and assumptions'
CONVENTION_HEADER = ('Convention', 'This article')
CONVENTION_ROWS = (
    'Return basis',
    'Estimation grid',
    'Rebalancing grid',
    'Covariance units',
    'Expected returns',
    'Weight state',
    'Solver',
)
FRAGMENT_MARKER = '<!-- fragment -->'
PACKAGE = 'optimalportfolios'
PAGE_STATES = {
    ('methodology', 'pending'), ('methodology', 'adopted'),
    ('case_study', 'pending'), ('case_study', 'adopted'),
    ('utility', 'pending'), ('utility', 'adopted'),
    ('api', 'api'),
}
PLANNED_FORMS = {'methodology', 'case_study', 'utility'}
EXCLUDED_DOC_ROOTS = {'docs/generated', 'docs/_generated', 'docs/_templates', 'docs/_build'}
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


def check_document(text: str, *, methodology: bool = False, form: Optional[str] = None,
                   card: bool = False) -> list[Issue]:
    """Check one article's metadata, structure, byline, references, and math source.

    Args:
        text: Complete Markdown source.
        methodology: Whether the full methodology section order is required; kept for callers
            that predate ``form``.
        form: Page form from the inventory: ``methodology``, ``case_study`` or ``utility``.
            Overrides ``methodology`` when given.
        card: Whether the convention card must open the inputs section.

    Returns:
        Source issues. An empty list means these source checks passed, not that TeX rendered.
    """
    if form is None:
        form = 'methodology' if methodology else 'utility'
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
    sections = [title for _, level, title in headings if level == 2]
    if form == 'methodology' and sections != list(ARTICLE_HEADINGS):
        issues.append(Issue(1, 'Use each required methodology H2 once, in the standard order.'))
    if form == 'case_study' and sections != list(CASE_STUDY_HEADINGS):
        issues.append(Issue(1, 'Use each required case-study H2 once, in the standard order.'))
    if card:
        issues.extend(check_convention_card(visible))
    return issues


def table_cells(line: str) -> list[str]:
    """Split one Markdown table row into stripped cell texts."""
    return [cell.strip() for cell in line.strip().strip('|').split('|')]


def check_convention_card(visible: list[tuple[int, str]]) -> list[Issue]:
    """Require the seven-row convention card as the first table of the inputs section.

    Args:
        visible: Numbered prose lines from ``prose_lines``.

    Returns:
        One issue naming the first deviation, or none.
    """
    start = next((number for number, line in visible
                  if (match := HEADING.match(line)) and len(match[1]) == 2
                  and match[2] == CONVENTION_SECTION), None)
    message = (f'Open "{CONVENTION_SECTION}" with the convention card: '
               f'| Convention | This article | and the rows {", ".join(CONVENTION_ROWS)}.')
    if start is None:
        return [Issue(1, message)]
    table = []
    for number, line in visible:
        if number <= start:
            continue
        match = HEADING.match(line)
        if match and len(match[1]) <= 2:
            break
        if line.lstrip().startswith('|'):
            table.append((number, line))
        elif table:
            break
    if len(table) < 2 or tuple(table_cells(table[0][1])) != CONVENTION_HEADER:
        return [Issue(table[0][0] if table else start, message)]
    rows = [table_cells(line) for _, line in table[2:]]
    if [row[0] for row in rows] != list(CONVENTION_ROWS):
        return [Issue(table[0][0], message)]
    empty = [row[0] for row in rows if len(row) < 2 or not row[1]]
    if empty:
        return [Issue(table[0][0], f'Fill every convention-card row; empty: {", ".join(empty)}.')]
    return []


def python_fences(text: str) -> list[tuple[int, str, bool]]:
    """Return the top-level Python fences of a page.

    Args:
        text: Markdown source.

    Returns:
        ``(line, body, fragment)`` per fence, where ``fragment`` records a
        ``<!-- fragment -->`` marker on the nearest preceding non-blank line.
    """
    lines = text.splitlines()
    found = []
    fence = ''
    language = ''
    start = 0
    for index, line in enumerate(lines):
        matched = FENCE.match(line)
        if fence:
            if (matched and matched[1][0] == fence[0] and len(matched[1]) >= len(fence)
                    and not matched[2].strip()):
                if language == 'python':
                    previous = next((lines[i].strip() for i in range(start - 1, -1, -1)
                                     if lines[i].strip()), '')
                    found.append((start + 1, '\n'.join(lines[start + 1:index]),
                                  previous == FRAGMENT_MARKER))
                fence = ''
            continue
        if matched:
            fence, start = matched[1], index
            info = matched[2].strip().split()
            language = info[0].strip('{}') if info else ''
    return found


def check_excerpts(text: str, script: str) -> list[Issue]:
    """Require each unmarked Python block to be a contiguous excerpt of the canonical script.

    Both sides are compared after removing their common indentation and trailing spaces, so a
    block may excerpt the body of a function. A block that is not meant to run carries
    ``<!-- fragment -->`` on the line before its fence.

    Args:
        text: Markdown source of the page.
        script: Source of the page's canonical example script.

    Returns:
        One issue per block that is not an excerpt.
    """
    script_lines = [line.rstrip() for line in script.splitlines()]
    issues = []
    for number, body, fragment in python_fences(text):
        if fragment:
            continue
        block = textwrap.dedent('\n'.join(line.rstrip() for line in body.splitlines())).strip('\n')
        size = len(block.splitlines())
        found = bool(block) and any(
            textwrap.dedent('\n'.join(script_lines[i:i + size])).strip('\n') == block
            for i in range(len(script_lines) - size + 1)
        )
        if not found:
            issues.append(Issue(number, 'Python block is not a verbatim excerpt of the canonical '
                                        f'script; mark a non-runnable block {FRAGMENT_MARKER}.'))
    return issues


def module_source(root: Path, dotted: str) -> Optional[tuple[str, Path]]:
    """Locate a module of the package in the source tree.

    Args:
        root: Repository root.
        dotted: Absolute module name, such as ``optimalportfolios.utils.__init__``.

    Returns:
        The package-normalised module name and its file, or None for a module outside the
        package, which is then an external dependency.
    """
    parts = dotted.split('.')
    if parts[0] != PACKAGE:
        return None
    if parts[-1] == '__init__':
        parts = parts[:-1]
    base = root.joinpath('src', *parts)
    for candidate in (base / '__init__.py', base.with_suffix('.py')):
        if candidate.is_file():
            return '.'.join(parts), candidate
    return None


def imported_module(dotted: str, path: Path, node: ast.ImportFrom) -> str:
    """Resolve the absolute source module of a possibly relative ``from ... import``."""
    if not node.level:
        return node.module or ''
    package = dotted.split('.') if path.name == '__init__.py' else dotted.split('.')[:-1]
    package = package[:len(package) - node.level + 1]
    return '.'.join(package + ([node.module] if node.module else []))


def parse_module(root: Path, dotted: str) -> Optional[tuple[str, Path, ast.Module]]:
    """Parse a package module, returning its normalised name, file and syntax tree."""
    located = module_source(root, dotted)
    if located is None:
        return None
    name, path = located
    return name, path, ast.parse(path.read_text(encoding='utf-8'))


def exported_names(root: Path, dotted: str) -> list[str]:
    """Return the non-underscore, non-module names a star import of a module binds.

    Honours a literal ``__all__``; otherwise collects top-level definitions and ``from``
    imports, following star imports. Plain ``import`` statements bind modules and are skipped.
    """
    parsed = parse_module(root, dotted)
    if parsed is None:
        return []
    name, path, tree = parsed
    for node in tree.body:
        if (isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == '__all__'
                for target in node.targets)):
            return [str(value) for value in ast.literal_eval(node.value)]
    names = []
    for node in tree.body:
        if isinstance(node, ast.ImportFrom):
            source = imported_module(name, path, node)
            for alias in node.names:
                if alias.name == '*':
                    names.extend(exported_names(root, source))
                else:
                    names.append(alias.asname or alias.name)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names.append(node.name)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names.extend(target.id for target in targets if isinstance(target, ast.Name))
    return list(dict.fromkeys(name for name in names if not name.startswith('_')))


def locate(root: Path, dotted: str, name: str, seen: frozenset = frozenset()
           ) -> tuple[str, Optional[str], Optional[ast.AST]]:
    """Follow imports to the module that defines a name.

    Returns:
        ``(package, module, node)``: the defining top-level package; for names defined in this
        package, the defining module and its top-level definition node (None for an assignment).
    """
    parsed = parse_module(root, dotted)
    if parsed is None:
        return dotted.split('.')[0], None, None
    module, path, tree = parsed
    if (module, name) in seen:
        return PACKAGE, module, None
    seen = seen | {(module, name)}
    stars = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == name:
                return PACKAGE, module, node
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            if any(isinstance(target, ast.Name) and target.id == name for target in targets):
                return PACKAGE, module, None
        elif isinstance(node, ast.ImportFrom):
            source = imported_module(module, path, node)
            for alias in node.names:
                if alias.name == '*':
                    stars.append(source)
                elif (alias.asname or alias.name) == name:
                    return locate(root, source, alias.name, seen)
    for source in stars:
        if name in exported_names(root, source):
            return locate(root, source, name, seen)
    return PACKAGE, module, None


def public_surface(root: Path) -> dict[str, str]:
    """Map each public, non-module name of the package root to its defining package."""
    return {name: locate(root, PACKAGE, name)[0] for name in exported_names(root, PACKAGE)}


def dataclass_fields(root: Path, name: str) -> list[str]:
    """Return the fields of a public dataclass in declaration order, bases first.

    Reads annotated class-body assignments, skipping ``ClassVar`` and ``InitVar``, and follows
    base classes defined in the package, as ``dataclasses.fields`` orders them.
    """
    package, _, node = locate(root, PACKAGE, name)
    if package != PACKAGE or not isinstance(node, ast.ClassDef):
        raise ValueError(f'{name} is not a class defined in {PACKAGE}')
    return dataclass_fields_of(root, PACKAGE, name)


def dataclass_fields_of(root: Path, module: str, name: str) -> list[str]:
    """Return the dataclass fields of a class named in a given module's namespace."""
    package, defining, node = locate(root, module, name)
    if package != PACKAGE or not isinstance(node, ast.ClassDef):
        return []
    fields = []
    for base in node.bases:
        if isinstance(base, ast.Name):
            fields.extend(dataclass_fields_of(root, defining, base.id))
    for statement in node.body:
        if isinstance(statement, ast.AnnAssign) and isinstance(statement.target, ast.Name):
            annotation = ast.unparse(statement.annotation)
            if not re.match(r'^(?:typing\.|dataclasses\.)?(?:ClassVar|InitVar)\b', annotation):
                if statement.target.id not in fields:
                    fields.append(statement.target.id)
    return fields


def owned_names(text: str) -> set[str]:
    """Return the identifiers that a page names in inline code outside examples and comments."""
    visible, _, _ = prose_lines(text)
    names = set()
    for _, line in visible:
        for code in re.findall(r'`([^`\n]+)`', line):
            names.update(re.findall(r'[A-Za-z_][A-Za-z0-9_]*', code))
    return names


def check_ownership(inventory: dict, root: Path) -> list[str]:
    """Check that each public object and mapped dataclass field has exactly one owning page.

    Applies only to a tree that contains the package source; fixture trees without it return
    no errors. Owning pages must be inventoried or planned, and an adopted page must name, in
    inline code, every object and field it owns.

    Args:
        inventory: Parsed documentation inventory.
        root: Repository root.

    Returns:
        ``path:line: message`` errors.
    """
    if module_source(root, PACKAGE) is None:
        return []
    errors = []
    for key in ('symbols', 'external_symbols', 'parameters'):
        if not isinstance(inventory.get(key), dict):
            errors.append(f'tools/docs_inventory.json:1: Provide the {key} section.')
    if errors:
        return errors
    pages, planned = inventory['pages'], inventory.get('planned', {})
    owners_allowed = set(pages) | set(planned)
    surface = public_surface(root)
    own = {name for name, package in surface.items() if package == PACKAGE}
    external = {name: package for name, package in surface.items() if package != PACKAGE}
    seen = {}
    for page, names in inventory['symbols'].items():
        if page not in owners_allowed:
            errors.append(f'{page}:1: Symbol owner is neither an inventoried nor a planned page.')
        for name in names:
            if name in seen:
                errors.append(f'{page}:1: `{name}` is also owned by {seen[name]}.')
            seen[name] = page
            if name in external:
                errors.append(f'{page}:1: `{name}` is re-exported from {external[name]}; '
                              'list it under external_symbols.')
            elif name not in own:
                errors.append(f'{page}:1: `{name}` is not a public object of {PACKAGE}.')
    for name in sorted(own - set(seen)):
        errors.append('tools/docs_inventory.json:1: Public object without an owning page: '
                      f'`{name}`.')
    declared = inventory['external_symbols']
    for name in sorted(set(external) - set(declared)):
        errors.append('tools/docs_inventory.json:1: Re-export without an external owner: '
                      f'`{name}`.')
    for name, entry in declared.items():
        if name not in external:
            errors.append(f'tools/docs_inventory.json:1: `{name}` is not a re-exported name.')
        elif (not isinstance(entry, dict) or entry.get('package') != external[name]
              or not str(entry.get('url', '')).startswith('https://')):
            errors.append(f'tools/docs_inventory.json:1: `{name}` needs package '
                          f'{external[name]} and an https owner url.')
    field_owner = {}
    for class_name, owners in inventory['parameters'].items():
        try:
            fields = dataclass_fields(root, class_name)
        except ValueError as exc:
            errors.append(f'tools/docs_inventory.json:1: {exc}.')
            continue
        claimed = {}
        for page, names in owners.items():
            if page not in owners_allowed:
                errors.append(f'{page}:1: Parameter owner is neither an inventoried nor a '
                              'planned page.')
            for name in names:
                if name in claimed:
                    errors.append(f'{page}:1: `{class_name}.{name}` is also owned by '
                                  f'{claimed[name]}.')
                claimed[name] = page
                if name not in fields:
                    errors.append(f'{page}:1: `{class_name}.{name}` is not a field.')
                field_owner[(class_name, name)] = page
        for name in fields:
            if name not in claimed:
                errors.append(f'tools/docs_inventory.json:1: Field without an owning page: '
                              f'`{class_name}.{name}`.')
    for page, entry in pages.items():
        if entry.get('status') != 'adopted' or not (root / page).is_file():
            continue
        named = owned_names((root / page).read_text(encoding='utf-8'))
        required = [name for name, owner in seen.items() if owner == page]
        required += [field for (_, field), owner in field_owner.items() if owner == page]
        for name in sorted(set(required) - named):
            errors.append(f'{page}:1: Name the owned `{name}` in inline code on this page.')
    return errors


def normalised(text: str) -> str:
    """Collapse whitespace and case so that wrapped titles compare equal."""
    return re.sub(r'\s+', ' ', text).strip().casefold()


def check_papers(inventory: dict, root: Path) -> list[str]:
    """Check the paper ledger: complete entries, no retired title in reader-facing text.

    Only public papers belong in the ledger. The research-papers page, when inventoried, must
    carry each ledger title.
    """
    papers = inventory.get('papers', {})
    if not isinstance(papers, dict):
        return ['tools/docs_inventory.json:1: The papers ledger must be a mapping.']
    errors = []
    for key, entry in papers.items():
        if not isinstance(entry, dict) or not all(
                isinstance(entry.get(field), str) and entry[field].strip()
                for field in ('title', 'authors', 'status', 'citation')):
            errors.append(f'tools/docs_inventory.json:1: Paper {key} needs title, authors, '
                          'status and citation.')
        elif not isinstance(entry.get('retired_titles', []), list):
            errors.append(f'tools/docs_inventory.json:1: Paper {key} retired_titles is a list.')
    retired = [(key, title) for key, entry in papers.items() if isinstance(entry, dict)
               for title in entry.get('retired_titles', []) if isinstance(title, str)]
    scope = [root / 'README.md', root / 'CITATION.cff']
    scope += [path for path in (root / 'docs').rglob('*.md')
              if path.relative_to(root / 'docs').parts[0] not in {'generated', '_generated',
                                                                  '_build', '_templates'}]
    scope += sorted((root / 'papers').glob('*/README.md'))
    for path in scope:
        if not path.is_file():
            continue
        text = normalised(path.read_text(encoding='utf-8'))
        for key, title in retired:
            if normalised(title) in text:
                errors.append(f'{path.relative_to(root).as_posix()}:1: Retired title of paper '
                              f'{key}: "{title}".')
    page = root / 'docs' / 'research_papers.md'
    if 'docs/research_papers.md' in inventory.get('pages', {}) and page.is_file():
        text = normalised(page.read_text(encoding='utf-8'))
        for key, entry in papers.items():
            if isinstance(entry, dict) and normalised(entry.get('title', '')) not in text:
                errors.append(f'docs/research_papers.md:1: Ledger title of paper {key} missing.')
    return errors



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
    if not isinstance(inventory, dict) or inventory.get('schema_version') != 2:
        return {}, ['tools/docs_inventory.json:1: Expected inventory schema_version 2.']
    pages = inventory.get('pages')
    excluded = inventory.get('excluded_doc_roots')
    if not isinstance(pages, dict) or not pages or not isinstance(excluded, dict):
        return {}, ['tools/docs_inventory.json:1: Provide nonempty pages and excluded_doc_roots.']
    errors = []
    if set(excluded) != EXCLUDED_DOC_ROOTS or not all(excluded.values()):
        errors.append('tools/docs_inventory.json:1: Keep explicit generated/template/build roots.')
    planned = inventory.get('planned', {})
    if not isinstance(planned, dict):
        errors.append('tools/docs_inventory.json:1: planned must map page paths to records.')
        planned = {}
    for name, entry in planned.items():
        if (not isinstance(entry, dict) or entry.get('form') not in PLANNED_FORMS
                or not str(entry.get('title', '')).strip()):
            errors.append(f'{name}:1: A planned page needs a form and a title.')
        if name in pages:
            errors.append(f'{name}:1: A page is either inventoried or planned, not both.')
        if (root / name).exists():
            errors.append(f'{name}:1: Planned page exists; move it to pages.')
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
        if pair not in PAGE_STATES:
            errors.append(f'{name}:1: Invalid page form/adoption status.')
        example = entry.get('example')
        if example is not None and not (isinstance(example, str) and (root / example).is_file()
                                        and example.startswith('examples/docs/')):
            errors.append(f'{name}:1: The example must be an existing examples/docs/ script.')
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
    excluded = {Path(name).name for name in EXCLUDED_DOC_ROOTS}
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
    errors.extend(check_ownership(inventory, REPO_ROOT))
    errors.extend(check_papers(inventory, REPO_ROOT))
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
        for name in sorted(inventory.get('planned', {})):
            errors.append(f'{name}:1: Planned page not yet written.')
    for name in sorted(selected):
        path = REPO_ROOT / name
        if path.suffix != '.md':
            errors.append(f'{name}:1: Migrate human RST to Markdown before adoption.')
            continue
        source = path.read_text(encoding='utf-8')
        entry = pages[name]
        # The convention card is required from adoption, or earlier for a page written to the
        # canonical-script contract; pending legacy articles gain it when they are migrated.
        card = entry['form'] == 'methodology' and (
            entry['status'] == 'adopted' or 'example' in entry)
        issues = check_document(source, form=entry['form'], card=card)
        issues.extend(check_local_links(source, path, REPO_ROOT))
        if 'example' in entry and (REPO_ROOT / entry['example']).is_file():
            issues.extend(check_excerpts(
                source, (REPO_ROOT / entry['example']).read_text(encoding='utf-8')))
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
