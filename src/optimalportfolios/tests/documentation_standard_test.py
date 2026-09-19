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


@pytest.fixture
def inventory_tree(tmp_path, monkeypatch):
    """Create an isolated adopted/pending/API inventory for CLI behavior tests."""
    (tmp_path / 'docs').mkdir()
    (tmp_path / 'tools').mkdir()
    (tmp_path / 'docs/adopted.md').write_text(HEADER, encoding='utf-8')
    (tmp_path / 'docs/legacy.rst').write_text('Legacy\n======\n', encoding='utf-8')
    (tmp_path / 'docs/api.rst').write_text('API\n===\n', encoding='utf-8')
    inventory = {
        'schema_version': 1,
        'excluded_doc_roots': {
            'docs/generated': 'autosummary', 'docs/_templates': 'templates',
            'docs/_build': 'build output',
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
    for directory in ('generated', '_templates', '_build'):
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
