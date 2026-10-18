"""Run the canonical script of every documentation page.

Each page written to the canonical-script contract shows excerpts of one script under
``examples/docs/``, and that script asserts every number the page quotes. This module runs each
script's ``main`` in process, so the whole test suite fails when a page's statements no longer
hold. It replaces the per-page harnesses that executed the Markdown blocks. The offline lane of
``examples.yml`` also runs the scripts, as separate processes.
"""

from pathlib import Path
import runpy
import sys
from unittest.mock import patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = sorted((REPO_ROOT / 'examples' / 'docs').glob('*.py'))


def test_every_script_is_collected(root):
    """The parametrised run below sees the checkout's scripts, not an empty directory."""
    assert SCRIPTS, 'examples/docs holds no canonical script.'
    assert {path.name for path in SCRIPTS} == {
        path.name for path in (root / 'examples' / 'docs').glob('*.py')}


def _script_cases():
    """Mark composed examples through the same import-closure classifier used by CI."""
    if not SCRIPTS:
        return []
    classifier = runpy.run_path(str(REPO_ROOT / '.github/scripts/run_examples.py'))
    return [pytest.param(path, id=path.stem,
                         marks=pytest.mark.factor_integration
                         if classifier['external_dependencies'](path)
                         & {'tradablefactors', 'factorlasso'} else ())
            for path in SCRIPTS]


@pytest.mark.parametrize('script', _script_cases())
def test_canonical_script_asserts_its_page(root, script):
    """Run the script as ``__main__``; any failed assertion fails the test."""
    with patch.object(sys, 'path', [str(root), *sys.path]):
        runpy.run_path(str(script), run_name='__main__')
