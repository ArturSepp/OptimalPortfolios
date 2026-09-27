"""Check the build versions that the Sphinx configuration puts in every page footer.

Pages carry no hand-written version stamps; ``docs/conf.py`` reads the versions of the build
environment instead. The configuration reads ``pyproject.toml`` with ``tomllib``, so the test
needs Python 3.11 or later, as the documentation build does.
"""

from importlib.metadata import version
import runpy
import sys
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.skipif(sys.version_info < (3, 11),
                                reason='docs/conf.py reads pyproject.toml with tomllib.')


def test_footer_names_the_versions_of_the_build(root):
    """The footer text names the source release and every installed stack package."""
    with patch.object(sys, 'path', list(sys.path)):
        conf = runpy.run_path(str(root / 'docs' / 'conf.py'))
    text = conf['html_context']['build_versions']
    assert text.startswith(f"Built from optimalportfolios {conf['release']} with ")
    assert text.endswith('.')
    for distribution, label in conf['BUILD_PACKAGES'].items():
        assert f'{label} {version(distribution)}' in text
    template = (root / 'docs' / '_templates' / 'page.html').read_text(encoding='utf-8')
    assert '{% extends "!page.html" %}' in template and '{{ super() }}' in template
    assert 'build_versions' in template
