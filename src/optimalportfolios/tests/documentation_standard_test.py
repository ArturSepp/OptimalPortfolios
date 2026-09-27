"""Source-standard regressions that HTML compilation alone cannot detect.

The checker is repository tooling. Installed-wheel runs skip this module; a checkout missing
its checker or inventory fails instead of silently dropping the documentation gate.
"""

import json
from pathlib import Path
import runpy

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
if not (REPO_ROOT / 'pyproject.toml').is_file():
    pytest.skip('Documentation tooling is repository-only.', allow_module_level=True)
CHECKER_PATH = REPO_ROOT / 'tools' / 'check_docs.py'
assert CHECKER_PATH.is_file(), 'A checkout must contain its documentation checker.'
CHECKER = runpy.run_path(str(CHECKER_PATH))
CHECK = CHECKER['check_document']
PROJECT = 'https://github.com/ArturSepp/OptimalPortfolios'
AUTHOR_BYLINE = '*Author: [Artur Sepp](https://github.com/ArturSepp)*'
DATED_BYLINE = (AUTHOR_BYLINE[:-1] + ' / First recorded: [2026-09-06]('
                + PROJECT + '/commit/' + 'a' * 40 + ')*')
HEADER = f'''---
myst:
  html_meta:
    description: >-
      An analytical method implemented in OptimalPortfolios.
---

# An analytical method

{DATED_BYLINE}

Implemented in [OptimalPortfolios]({PROJECT}).
Software citation: [CITATION.cff]({PROJECT}/blob/main/CITATION.cff).
'''
SECTIONS = r'''
## Overview

Define the method before its implementation.

## Inputs, notation, and assumptions

Let $b_i$ be a fractional budget.

## Methodology

$$
\sum_i b_i = 1.
$$

The budgets sum to one.

## Worked example

Four equal budgets are each one quarter.

## Implementation in optimalportfolios

Use the verified public entry point.

## Interpretation and limitations

Target budgets alone do not determine portfolio weights.

## See also

Related construction methods.

## References

The software citation appears above.
'''


def test_complete_article_and_short_utility_page_pass():
    """Require the full section contract only for methodology pages."""
    assert not CHECK(HEADER + SECTIONS, methodology=True)
    assert not CHECK(HEADER, methodology=False)
    assert CHECK(HEADER, methodology=True)


@pytest.mark.parametrize('before,after,message', [
    (DATED_BYLINE, '', 'byline'),
    (f'[OptimalPortfolios]({PROJECT})', 'OptimalPortfolios', 'project repository'),
    (f'[CITATION.cff]({PROJECT}/blob/main/CITATION.cff)', 'CITATION.cff', 'CITATION.cff'),
    ('    description: >-\n      An analytical method implemented in OptimalPortfolios.',
     '', 'description'),
    ('    description: >-\n      An analytical method implemented in OptimalPortfolios.',
     '    description: ""', 'description must not be empty'),
    ('## Methodology', '#### Methodology', 'heading levels'),
    ('## Worked example', '# Another title', 'one H1'),
    ('## Worked example', '## Overview', 'required methodology'),
    ('## Overview', '## Worked example', 'required methodology'),
    ('## Worked example', '## Extra section\n\n## Worked example', 'required methodology'),
    ('$$\n\\sum_i b_i = 1.\n$$', '```{math}\n\\sum_i b_i = 1.\n```', 'math fences'),
    ('$$\n\\sum_i b_i = 1.\n$$', '```{math} label\nx = 1\n```', 'math fences'),
    ('$$\n\\sum_i b_i = 1.\n$$', '$$x = 1$$', 'own line'),
    ('$$\n\nThe budgets', '$$\nThe budgets', 'blank line after'),
    ('## Methodology\n\n$$', '## Methodology\n$$', 'blank line before'),
    ('$b_i$', '{math}`b_i`', 'ordinary equation-section links'),
    ('$b_i$', '\\(b_i\\)', 'portable mathematics'),
    ('$b_i$', '$b_i', 'Unclosed inline'),
])
def test_article_defects_are_rejected(before, after, message):
    """Inject distinct reader-visible defects and require a diagnostic for each."""
    source = HEADER + SECTIONS
    assert before in source, 'The injected defect must actually change the fixture.'
    issues = CHECK(source.replace(before, after, 1), methodology=True)
    assert any(message in issue.message for issue in issues), issues


def test_code_and_comments_cannot_supply_attribution_or_structure():
    """Hidden examples and comments must not impersonate a real article's metadata."""
    hidden = '````markdown\n' + HEADER + '\n````\n'
    issues = CHECK('# Real title\n\n' + hidden + SECTIONS, methodology=True)
    assert any('byline' in issue.message for issue in issues)
    assert any('project repository' in issue.message for issue in issues)
    issues = CHECK(HEADER + '<!--\n' + SECTIONS + '\n-->', methodology=True)
    assert any('required methodology' in issue.message for issue in issues)
    source = HEADER.replace(f'[OptimalPortfolios]({PROJECT})',
                            f'`[OptimalPortfolios]({PROJECT})`')
    assert any('project repository' in issue.message
               for issue in CHECK(source + SECTIONS, methodology=True))


def test_teaching_source_is_not_checked_as_article_math():
    """Respect nested fences, tilde fences, inline source and escaped currency."""
    example = '\n````markdown\n# Example\n```{math}\nx = 1\n```\n$$x$$\n````\n'
    assert not CHECK(HEADER + SECTIONS + example, methodology=True)
    example = '\n~~~markdown\n$$x$$\n~~~\nUse `$...$`, `{math}` and `\\(x\\)`.\n'
    assert not CHECK(HEADER + SECTIONS + example, methodology=True)
    assert not CHECK(HEADER + SECTIONS + '\nA literal \\$ sign.\n', methodology=True)


@pytest.mark.parametrize('ending,message', [
    ('\n```python\nx = 1\n', 'Unclosed code'),
    ('\n$$\nx = 1\n', 'Unclosed display'),
    ('\n<!-- hidden\n', 'Unclosed HTML'),
])
def test_unclosed_source_blocks_are_rejected(ending, message):
    """Unterminated source blocks cannot hide the remainder of a page."""
    issues = CHECK(HEADER + SECTIONS + ending, methodology=True)
    assert any(message in issue.message for issue in issues)


@pytest.mark.parametrize('faulty,fixed,message', [
    ('A scale $\\sigma \\, C$.', 'A scale $\\sigma C$.', 'Spell TeX commands'),
    ('A set $\\{w\\}$.', 'A set $\\lbrace w \\rbrace$.', 'Spell TeX commands'),
    ('$$\n\\lVert w \\rVert = \\left\\|w\\right\\|\n$$', '$$\n\\lVert w \\rVert\n$$',
     'Spell TeX commands'),
    ('Positive $w_i > 0$ weights.', 'Positive $w_i \\gt 0$ weights.', '\\lt and \\gt'),
    ('An $n$-by-$n$ matrix.', 'An $n \\times n$ matrix.', 'space or an opening parenthesis'),
    ('The $i$th asset.', 'Asset $i$.', 'letter or digit'),
    ('$$\nx = a\n+ b\n$$', '$$\nx = a +\nb\n$$', 'display-math line'),
    ('> **Insight.** Weights $w_i > 0$.', '> **Insight.** Weights $w_i \\gt 0$.',
     '\\lt and \\gt'),
    ('Returns $\\hat{\\mu}_t$ and\nholdings $w_{t^-}$.',
     'Returns $\\hat\\mu_t$ and\nholdings $w_{t^-}$.', 'as emphasis'),
    ('| $\\hat{\\mu}_t$ | $w_{t^-}$ |', '| $\\hat{\\mu}_t$ | $w_{t^-}$ |', None),
    ('- $\\hat{\\mu}_t$ returns\n- $w_{t^-}$ holdings',
     '- $\\hat{\\mu}_t$ returns\n- $w_{t^-}$ holdings', None),
])
def test_math_that_github_renders_wrongly_is_rejected(faulty, fixed, message):
    """Each GitHub-only math fault fails, and its portable spelling passes.

    The last two cases pass as written: emphasis never crosses a table cell or a list item.
    """
    source = HEADER + SECTIONS + '\n' + faulty + '\n'
    issues = CHECK(source, methodology=True)
    if message is None:
        assert not issues, issues
    else:
        assert any(message in issue.message for issue in issues), issues
    assert not CHECK(HEADER + SECTIONS + '\n' + fixed + '\n', methodology=True)


def test_local_links_check_files_without_interpreting_examples(tmp_path):
    """Resolve encoded and reference-style files, excluding code and remote URLs."""
    docs = tmp_path / 'docs'
    docs.mkdir()
    (docs / 'target file.md').write_text('# Target\n', encoding='utf-8')
    path = docs / 'article.md'
    source = HEADER + '''
[Target](target%20file.md#fragment)
[Reference][ref]
[ref]: <target file.md>
[Site](target%20file.html)
[Remote](https://example.invalid/anything)
`[Illustration](missing.py)`
<!-- [Hidden](missing.py) -->
```markdown
[Teaching source](missing.py)
```
'''
    assert not CHECKER['check_local_links'](source, path, tmp_path)
    for link in ('missing.py', '../../outside.md'):
        issues = CHECKER['check_local_links'](HEADER + f'\n[Source]({link})\n', path, tmp_path)
        assert len(issues) == 1
        assert ('Missing local' in issues[0].message
                or 'leaves the repository' in issues[0].message)
    # Image alt text that wraps lines is still resolved on the line that closes the label.
    wrapped = '\n![Left: a long description that\ncontinues here](images/{}.png)\n'
    (docs / 'images').mkdir()
    (docs / 'images' / 'present.png').write_bytes(b'png')
    assert not CHECKER['check_local_links'](HEADER + wrapped.format('present'), path, tmp_path)
    issues = CHECKER['check_local_links'](HEADER + wrapped.format('absent'), path, tmp_path)
    assert len(issues) == 1 and 'images/absent.png' in issues[0].message


@pytest.fixture
def inventory_tree(tmp_path, monkeypatch):
    """Create an isolated adopted/pending/API inventory for CLI behavior tests."""
    (tmp_path / 'docs').mkdir()
    (tmp_path / 'tools').mkdir()
    (tmp_path / 'docs/adopted.md').write_text(HEADER, encoding='utf-8')
    (tmp_path / 'docs/legacy.rst').write_text('Legacy\n======\n', encoding='utf-8')
    (tmp_path / 'docs/api.rst').write_text('API\n===\n', encoding='utf-8')
    inventory = {
        'schema_version': 2,
        'excluded_doc_roots': {
            'docs/generated': 'autosummary', 'docs/_generated': 'API body',
            'docs/_templates': 'templates', 'docs/_build': 'build output',
        },
        'pages': {
            'docs/adopted.md': {'form': 'utility', 'status': 'adopted'},
            'docs/legacy.rst': {'form': 'utility', 'status': 'pending'},
            'docs/api.rst': {'form': 'api', 'status': 'api'},
        },
    }
    save_inventory(tmp_path, inventory)
    monkeypatch.setitem(CHECKER['main'].__globals__, 'REPO_ROOT', tmp_path)
    return tmp_path, inventory


def save_inventory(root, inventory):
    """Persist a fixture inventory without affecting the repository copy."""
    (root / 'tools/docs_inventory.json').write_text(json.dumps(inventory), encoding='utf-8')


def test_cli_reports_pending_and_requires_adoption(inventory_tree, capsys):
    """A partial default pass must not be reported as completed migration."""
    root, inventory = inventory_tree
    assert CHECKER['main']([]) == 0
    assert 'PENDING (not adopted): 1' in capsys.readouterr().out
    assert CHECKER['main'](['--all']) == 1
    assert 'Pending migration' in capsys.readouterr().out
    (root / 'docs/legacy.rst').unlink()
    (root / 'docs/legacy.md').write_text(HEADER, encoding='utf-8')
    inventory['pages']['docs/legacy.md'] = inventory['pages'].pop('docs/legacy.rst')
    save_inventory(root, inventory)
    assert CHECKER['main'](['--files', 'docs/legacy.md']) == 0
    assert CHECKER['main'](['--all']) == 1
    inventory['pages']['docs/legacy.md']['status'] = 'adopted'
    save_inventory(root, inventory)
    assert CHECKER['main'](['--all']) == 0


@pytest.mark.parametrize('name', ['new_topic.md', 'new_topic.rst'])
def test_unknown_pages_fail_even_in_partial_mode(inventory_tree, name, capsys):
    """New Markdown and RST pages cannot evade ownership through the migration baseline."""
    root, _ = inventory_tree
    (root / 'docs' / name).write_text('# New\n', encoding='utf-8')
    assert CHECKER['main']([]) == 1
    assert 'explicit documentation inventory' in capsys.readouterr().out


def test_missing_pending_source_and_duplicate_basename_fail(inventory_tree, capsys):
    """The pending list still checks existence, and Sphinx source names stay unique."""
    root, inventory = inventory_tree
    (root / 'docs/legacy.rst').unlink()
    assert CHECKER['main']([]) == 1
    assert 'Missing documentation page' in capsys.readouterr().out
    (root / 'docs/legacy.rst').write_text('Legacy\n======\n', encoding='utf-8')
    (root / 'docs/adopted.rst').write_text('Duplicate\n=========\n', encoding='utf-8')
    inventory['pages']['docs/adopted.rst'] = {'form': 'utility', 'status': 'pending'}
    save_inventory(root, inventory)
    assert CHECKER['main']([]) == 1
    assert 'Duplicate Sphinx source basename' in capsys.readouterr().out


def test_only_named_api_entry_is_exempt(inventory_tree, capsys):
    """Arbitrary pages cannot opt out by declaring API ownership."""
    root, inventory = inventory_tree
    inventory['pages']['docs/legacy.rst'] = {'form': 'api', 'status': 'api'}
    save_inventory(root, inventory)
    assert CHECKER['main']([]) == 1
    assert 'Only the autosummary entry' in capsys.readouterr().out


def test_generated_roots_are_excluded_but_new_nested_prose_is_not(inventory_tree, capsys):
    """Generated API files are exempt while new authored subdirectories remain visible."""
    root, _ = inventory_tree
    for directory in ('generated', '_generated', '_templates', '_build'):
        (root / 'docs' / directory).mkdir()
        (root / 'docs' / directory / 'machine.rst').write_text('Generated\n', encoding='utf-8')
    assert CHECKER['main']([]) == 0
    (root / 'docs/guides').mkdir()
    (root / 'docs/guides/new.md').write_text(HEADER, encoding='utf-8')
    assert CHECKER['main']([]) == 1
    assert 'explicit documentation inventory' in capsys.readouterr().out


def test_selected_paths_cannot_escape_inventory(inventory_tree):
    """Reject unrelated files, API sources and paths outside the repository."""
    for path in ('../outside.md', 'docs/api.rst', 'docs/unknown.md'):
        with pytest.raises(SystemExit):
            CHECKER['main'](['--files', path])


def test_cli_rejects_missing_local_target(inventory_tree, capsys):
    """The actual CLI must invoke local-link checking on adopted pages."""
    root, _ = inventory_tree
    (root / 'docs/adopted.md').write_text(HEADER + '\n[Source](missing.py)\n', encoding='utf-8')
    assert CHECKER['main']([]) == 1
    assert 'Missing local link target' in capsys.readouterr().out


def test_adopted_repository_pages_pass():
    """Exercise the committed inventory rather than only synthetic checker fixtures."""
    assert CHECKER['main']([]) == 0



def test_source_all_checks_pending_pages_without_adopting_them(inventory_tree, capsys):
    """CI checks every article while preserving independent adoption evidence."""
    root, inventory = inventory_tree
    assert CHECKER['main'](['--source-all']) == 1
    assert 'Migrate human RST' in capsys.readouterr().out
    (root / 'docs/legacy.rst').unlink()
    path = root / 'docs/legacy.md'
    path.write_text(HEADER, encoding='utf-8')
    inventory['pages']['docs/legacy.md'] = inventory['pages'].pop('docs/legacy.rst')
    save_inventory(root, inventory)
    before = (root / 'tools/docs_inventory.json').read_bytes()
    assert CHECKER['main'](['--source-all']) == 0
    output = capsys.readouterr().out
    assert 'PASS: 2 selected human pages' in output
    assert 'PENDING (not adopted): 1' in output
    assert (root / 'tools/docs_inventory.json').read_bytes() == before
    assert CHECKER['main'](['--all']) == 1
    assert 'Pending migration' in capsys.readouterr().out
    path.write_text(HEADER + '\n$$broken$$\n', encoding='utf-8')
    assert CHECKER['main']([]) == 0
    capsys.readouterr()
    assert CHECKER['main'](['--source-all']) == 1
    assert 'own line' in capsys.readouterr().out
    path.write_text(HEADER + '\n[Source](missing.py)\n', encoding='utf-8')
    assert CHECKER['main'](['--source-all']) == 1
    assert 'Missing local link target' in capsys.readouterr().out


def test_source_all_cannot_be_combined_with_narrower_selections(inventory_tree):
    """A mixed invocation must not silently narrow the all-source CI gate."""
    for arguments in (['--source-all', '--all'],
                      ['--source-all', '--files', 'docs/adopted.md']):
        with pytest.raises(SystemExit) as error:
            CHECKER['main'](arguments)
        assert error.value.code == 2


def test_all_repository_sources_pass_without_claiming_adoption():
    """Check pending prose too, retaining the installed-wheel module-level skip."""
    assert CHECKER['main'](['--source-all']) == 0


@pytest.mark.parametrize('byline', [
    AUTHOR_BYLINE,
    DATED_BYLINE,
    DATED_BYLINE.replace('2026-09-06', '2024-02-29'),
])
def test_linked_author_and_evidenced_date_pass(byline):
    """Allow confirmed authors without inventing an affiliation or uncommitted date."""
    assert not CHECK(HEADER.replace(DATED_BYLINE, byline), methodology=False)


@pytest.mark.parametrize('byline', [
    '*[author / affiliation / date — placeholder]*',
    AUTHOR_BYLINE.replace('[Artur Sepp](https://github.com/ArturSepp)', 'Artur Sepp'),
    AUTHOR_BYLINE.replace('https://github.com/', 'https://example.com/'),
    DATED_BYLINE.replace('2026-09-06', '2026-02-30'),
    DATED_BYLINE.replace('a' * 40, 'abcdef'),
    DATED_BYLINE.replace(PROJECT + '/commit/', PROJECT + '/tree/'),
    DATED_BYLINE.replace(PROJECT + '/commit/', 'https://example.com/commit/'),
])
def test_invalid_author_metadata_is_rejected(byline):
    """Reject placeholders, broken attribution and dates without a valid evidence link."""
    issues = CHECK(HEADER.replace(DATED_BYLINE, byline), methodology=False)
    assert any('byline' in issue.message for issue in issues)


CARD = '''| Convention | This article |
|---|---|
| Return basis | Simple returns |
| Estimation grid | Weekly, span 52 |
| Rebalancing grid | Quarter ends |
| Covariance units | Annualised by the estimator |
| Expected returns | None |
| Weight state | Target weights |
| Solver | CLARABEL through CVXPY |

'''
WITH_CARD = SECTIONS.replace('Let $b_i$ be a fractional budget.', CARD + 'Let $b_i$ be a budget.')


def test_convention_card_is_required_when_requested():
    """The card must open the inputs section with the seven rows in order and filled in."""
    assert not CHECK(HEADER + WITH_CARD, form='methodology', card=True)
    assert not CHECK(HEADER + SECTIONS, form='methodology', card=False)
    for source, message in (
            (HEADER + SECTIONS, 'convention card'),
            (HEADER + WITH_CARD.replace('| Solver |', '| Backend |'), 'convention card'),
            (HEADER + WITH_CARD.replace('| Return basis | Simple returns |\n', ''),
             'convention card'),
            (HEADER + WITH_CARD.replace('| Expected returns | None |', '| Expected returns |  |'),
             'Fill every convention-card row'),
            (HEADER + WITH_CARD.replace('| Convention | This article |', '| Row | Value |'),
             'convention card')):
        issues = CHECK(source, form='methodology', card=True)
        assert any(message in issue.message for issue in issues), issues


CASE_STUDY = ''.join(f'\n## {heading}\n\nText.\n' for heading in CHECKER['CASE_STUDY_HEADINGS'])


def test_case_study_form_requires_its_sections_in_order():
    """A case study uses its own eight sections, not the methodology sections."""
    assert not CHECK(HEADER + CASE_STUDY, form='case_study')
    swapped = CASE_STUDY.replace('## Results', '## Swap').replace('## Configuration', '## Results')
    issues = CHECK(HEADER + swapped.replace('## Swap', '## Configuration'), form='case_study')
    assert any('case-study H2' in issue.message for issue in issues)
    assert any('case-study H2' in issue.message
               for issue in CHECK(HEADER + SECTIONS, form='case_study'))


SCRIPT = '''"""Canonical script."""
import numpy as np


def main():
    weights = np.full(4, 0.25)
    assert abs(weights.sum() - 1.0) < 1e-12
    return weights


if __name__ == "__main__":
    main()
'''


@pytest.mark.parametrize('block,passes', [
    ('weights = np.full(4, 0.25)\nassert abs(weights.sum() - 1.0) < 1e-12', True),
    ('import numpy as np', True),
    ('weights = np.full(4, 0.30)', False),
    ('weights = np.full(4, 0.25)\nreturn weights', False),
])
def test_python_blocks_must_be_excerpts_of_the_script(block, passes):
    """Blocks compare after dedenting; an edited or non-contiguous block fails."""
    page = f'{HEADER}\n```python\n{block}\n```\n'
    issues = CHECKER['check_excerpts'](page, SCRIPT)
    assert (not issues) is passes, issues
    marked = f'{HEADER}\n{CHECKER["FRAGMENT_MARKER"]}\n```python\n{block}\n```\n'
    assert not CHECKER['check_excerpts'](marked, SCRIPT)


def test_python_fences_skip_teaching_source_inside_outer_fences():
    """A Python fence shown inside a Markdown teaching block is not an executable block."""
    page = '````markdown\n```python\nnot_in_script()\n```\n````\n\n```text\nx\n```\n'
    assert CHECKER['python_fences'](page) == []


@pytest.fixture(scope='module')
def repository_inventory():
    """The committed inventory of this checkout."""
    return json.loads((REPO_ROOT / 'tools/docs_inventory.json').read_text(encoding='utf-8'))


def test_static_public_surface_matches_the_runtime_package():
    """The ast reading of the package root equals the imported package's public objects."""
    import inspect
    import optimalportfolios as op
    surface = CHECKER['public_surface'](REPO_ROOT)
    runtime = {name for name in dir(op)
               if not name.startswith('_') and not inspect.ismodule(getattr(op, name))}
    assert set(surface) == runtime
    for name, package in surface.items():
        module = getattr(getattr(op, name), '__module__', '') or ''
        assert module.split('.')[0] == package, (name, module, package)


def test_static_dataclass_fields_match_runtime(repository_inventory):
    """The ast reading of each mapped dataclass equals dataclasses.fields, in order."""
    import dataclasses
    import optimalportfolios as op
    for name in repository_inventory['parameters']:
        expected = [field.name for field in dataclasses.fields(getattr(op, name))]
        assert CHECKER['dataclass_fields'](REPO_ROOT, name) == expected, name


def test_repository_ownership_is_complete(repository_inventory):
    """Every public object and mapped field of this checkout has exactly one owning page."""
    assert CHECKER['check_ownership'](repository_inventory, REPO_ROOT) == []
    assert CHECKER['check_papers'](repository_inventory, REPO_ROOT) == []


def _mutated(inventory, change):
    """Return a deep copy of the inventory after applying one change to it."""
    copy = json.loads(json.dumps(inventory))
    change(copy)
    return copy


def _move_to_adopted_page(inventory, name):
    """Give an object to the adopted standard page, which does not name it."""
    for names in inventory['symbols'].values():
        if name in names:
            names.remove(name)
    inventory['symbols']['docs/documentation_standard.md'] = [name]


@pytest.mark.parametrize('change,message', [
    (lambda inv: inv['symbols']['docs/constraints.md'].remove('Constraints'),
     'Public object without an owning page: `Constraints`'),
    (lambda inv: inv['symbols']['docs/rolling_backtests.md'].append('Constraints'),
     '`Constraints` is also owned'),
    (lambda inv: inv['symbols']['docs/rolling_backtests.md'].append('not_a_public_name'),
     'is not a public object'),
    (lambda inv: inv['symbols']['docs/rolling_backtests.md'].append('LassoModel'),
     'list it under external_symbols'),
    (lambda inv: inv['external_symbols'].pop('LassoModel'),
     'Re-export without an external owner: `LassoModel`'),
    (lambda inv: inv['symbols'].setdefault('docs/unknown_page.md', []),
     'neither an inventoried nor a planned page'),
    (lambda inv: inv['parameters']['Constraints']['docs/constraints.md'].remove('weights_0'),
     'Field without an owning page: `Constraints.weights_0`'),
    (lambda inv: inv['parameters']['Constraints']['docs/constraints.md'].append('not_a_field'),
     'is not a field'),
    (lambda inv: _move_to_adopted_page(inv, 'round_weights_to_pct'),
     'Name the owned `round_weights_to_pct`'),
])
def test_ownership_defects_are_rejected(repository_inventory, change, message):
    """Each ownership defect of the real inventory produces its diagnostic."""
    errors = CHECKER['check_ownership'](_mutated(repository_inventory, change), REPO_ROOT)
    assert any(message in error for error in errors), errors


def test_retired_paper_titles_and_ledger_titles_are_checked(tmp_path):
    """A retired title fails in reader-facing text; the papers page must carry each title."""
    (tmp_path / 'docs').mkdir()
    (tmp_path / 'docs/page.md').write_text('See *An Old Working\nTitle* here.', encoding='utf-8')
    (tmp_path / 'docs/research_papers.md').write_text('Nothing yet.', encoding='utf-8')
    inventory = {'pages': {'docs/research_papers.md': {}}, 'papers': {'paper': {
        'title': 'The Current Title', 'authors': 'Sepp, A.', 'status': 'Working paper',
        'citation': 'Sepp, A. (2026). The Current Title.',
        'retired_titles': ['An old working title']}}}
    errors = CHECKER['check_papers'](inventory, tmp_path)
    assert any('docs/page.md:1: Retired title' in error for error in errors), errors
    assert any('Ledger title of paper paper missing' in error for error in errors), errors
    inventory['papers']['paper'].pop('citation')
    assert any('needs title, authors' in error
               for error in CHECKER['check_papers'](inventory, tmp_path))


def test_planned_pages_block_complete_adoption(inventory_tree, capsys):
    """A planned page is not a written page: it fails --all and must not already exist."""
    root, inventory = inventory_tree
    (root / 'docs/legacy.rst').unlink()
    (root / 'docs/legacy.md').write_text(HEADER, encoding='utf-8')
    inventory['pages'].pop('docs/legacy.rst')
    inventory['pages']['docs/legacy.md'] = {'form': 'utility', 'status': 'adopted'}
    inventory['planned'] = {'docs/future.md': {'form': 'methodology', 'title': 'Future'}}
    save_inventory(root, inventory)
    assert CHECKER['main']([]) == 0
    capsys.readouterr()
    assert CHECKER['main'](['--all']) == 1
    assert 'Planned page not yet written' in capsys.readouterr().out
    (root / 'docs/future.md').write_text(HEADER, encoding='utf-8')
    assert CHECKER['main']([]) == 1
    assert 'Planned page exists' in capsys.readouterr().out


def test_example_key_enables_card_and_excerpt_checks(inventory_tree, capsys):
    """A page written to the script contract gets the card and excerpt rules before adoption."""
    root, inventory = inventory_tree
    (root / 'examples/docs').mkdir(parents=True)
    (root / 'examples/docs/method.py').write_text(SCRIPT, encoding='utf-8')
    page = HEADER + WITH_CARD + '\n```python\nweights = np.full(4, 0.25)\n```\n'
    (root / 'docs/method.md').write_text(page, encoding='utf-8')
    inventory['pages']['docs/method.md'] = {'form': 'methodology', 'status': 'pending',
                                            'example': 'examples/docs/method.py'}
    save_inventory(root, inventory)
    assert CHECKER['main'](['--files', 'docs/method.md']) == 0
    (root / 'docs/method.md').write_text(page.replace('0.25', '0.5').replace(CARD, ''),
                                         encoding='utf-8')
    assert CHECKER['main'](['--files', 'docs/method.md']) == 1
    output = capsys.readouterr().out
    assert 'not a verbatim excerpt' in output and 'convention card' in output
    inventory['pages']['docs/method.md']['example'] = 'examples/docs/missing.py'
    save_inventory(root, inventory)
    assert CHECKER['main'](['--files', 'docs/method.md']) == 1
    assert 'existing examples/docs/ script' in capsys.readouterr().out
