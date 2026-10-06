"""Check the build versions in every page footer and the page titles of the documentation site.

Pages carry no hand-written version stamps; ``docs/conf.py`` reads the versions of the build
environment instead. The configuration reads ``pyproject.toml`` with ``tomllib``, so the test
needs Python 3.11 or later, as the documentation build does.
"""

from importlib.metadata import version
import re
import runpy
import subprocess
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


def test_base_template_shortens_page_titles(root):
    """The base template replaces the long ``html_title`` suffix with the project name."""
    template = (root / 'docs' / '_templates' / 'base.html').read_text(encoding='utf-8')
    assert '{%- block htmltitle -%}' in template
    assert 'pagename != master_doc' in template and '{{ super() }}' in template
    assert '{{ project|striptags|e }}</title>' in template


def test_built_pages_carry_short_titles(root, tmp_path):
    """A Furo build titles the homepage with ``html_title`` and other pages with the project.

    Furo otherwise ends every title with the full ``html_title``, which search results cut off.
    Runs only where the ``docs`` extra is installed.
    """
    pytest.importorskip('furo')
    with patch.object(sys, 'path', list(sys.path)):
        conf = runpy.run_path(str(root / 'docs' / 'conf.py'))
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'conf.py').write_text(
        f"templates_path = [{str(root / 'docs' / '_templates')!r}]\n"
        "html_theme = 'furo'\n"
        f"project = {conf['project']!r}\n"
        f"html_title = {conf['html_title']!r}\n", encoding='utf-8')
    (source / 'index.rst').write_text('Home\n====\n\n.. toctree::\n\n   method\n',
                                      encoding='utf-8')
    (source / 'method.rst').write_text('Risk budgeting\n==============\n\nA method.\n',
                                       encoding='utf-8')
    output = tmp_path / 'html'
    result = subprocess.run(
        [sys.executable, '-m', 'sphinx', '-W', '-q', '-b', 'html', str(source), str(output)],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    def head_titles(name):
        """Return the ``<title>`` elements of a built page's head, not the theme's icon titles."""
        html = (output / f'{name}.html').read_text(encoding='utf-8')
        return re.findall(r'<title>(.*?)</title>', html.split('</head>')[0])

    assert head_titles('index') == [conf['html_title']]
    assert head_titles('method') == [f"Risk budgeting - {conf['project']}"]
