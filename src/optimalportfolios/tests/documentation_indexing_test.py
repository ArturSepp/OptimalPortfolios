"""Documentation discovery contracts using the existing Sphinx extension."""
import runpy
from pathlib import Path
from types import SimpleNamespace

import pytest

EXTENSION = Path(__file__).resolve().parents[3] / 'docs' / '_ext' / 'optimalportfolios_indexing.py'
if not EXTENSION.is_file():
    pytest.skip('Documentation sources are not shipped in the wheel.', allow_module_level=True)
SEO = runpy.run_path(str(EXTENSION))
BASE = 'https://optimalportfolios.readthedocs.io/en/stable/'


def make_app(tmp_path, *, base=BASE, builder='html'):
    """Make a minimal builder and complete document inventory."""
    return SimpleNamespace(
        config=SimpleNamespace(html_baseurl=base),
        builder=SimpleNamespace(name=builder, get_target_uri=lambda name: name + '.html'),
        env=SimpleNamespace(found_docs={'index', 'api', 'quickstart', 'search'}),
        outdir=str(tmp_path),
    )

PACKAGE = 'optimalportfolios'


def test_generated_navigation_uses_real_exports_and_is_deterministic(tmp_path):
    """Reject invented exports and keep generated navigation stable across reruns."""
    package = __import__(PACKAGE)
    names = SEO['public_api_names']()
    assert names and names == sorted(set(names))
    assert all(hasattr(package, name) for name in names)
    app = make_app(tmp_path)
    name = names[0]
    app.config.release = '1.2.3'
    app.env.titles = {}
    app.env.get_domain = lambda domain: SimpleNamespace(get_objects=lambda: [
        (PACKAGE + '.' + name, name, 'class', 'index', 'real-anchor', 1),
        (PACKAGE + '.invented_export', 'invented_export', 'class', 'index', 'fake', 1),
    ])
    SEO['write_llms'](app, None)
    initial = (tmp_path / 'llms.txt').read_bytes()
    api = (tmp_path / 'llms-api.txt').read_text(encoding='utf-8')
    assert 'invented_export' not in api
    assert PACKAGE + '.' + name in api and '#real-anchor' in api
    assert '_modules/' not in initial.decode() and 'search.html' not in initial.decode()
    assert 'llms-api.txt' in initial.decode()
    SEO['write_llms'](app, None)
    assert (tmp_path / 'llms.txt').read_bytes() == initial
    assert (tmp_path / 'llms-api.txt').read_text(encoding='utf-8') == api


@pytest.mark.parametrize('builder,base,exception', [
    ('latex', BASE, None), ('html', '', None), ('html', BASE, RuntimeError('failed')),
])
def test_failed_and_non_html_builds_publish_no_llm_index(tmp_path, builder, base, exception):
    """Publish indexes only for successful HTML builds with an absolute base URL."""
    SEO['write_llms'](make_app(tmp_path, builder=builder, base=base), exception)
    assert not (tmp_path / 'llms.txt').exists()
    assert not (tmp_path / 'llms-api.txt').exists()


@pytest.mark.parametrize('channel,label', [('stable', 'Released'), ('latest', 'Development')])
def test_release_navigation_keeps_channel_roots_separate(tmp_path, channel, label):
    """Link channel roots rather than guessing that the same page exists in both."""
    base = BASE.replace('/stable/', '/' + channel + '/').replace('/latest/', '/' + channel + '/')
    app = make_app(tmp_path, base=base)
    app.config.release = '<unsafe>'
    context = {'body': '<p>Page</p>'}
    SEO['set_release_navigation'](app, 'index', 'page.html', context, None)
    assert label in context['body'] and '&lt;unsafe&gt;' in context['body']
    assert '/en/stable/' in context['body'] and '/en/latest/' in context['body']
    assert '/en/stable/index' not in context['body']
    assert context['body'].endswith('<p>Page</p>')


def test_stale_api_inventory_fails_closed(monkeypatch):
    """A misspelled or removed declared export must stop index generation."""
    import json

    package = __import__(PACKAGE)
    if PACKAGE == 'optimalportfolios':
        inventory = {'symbols': {'docs/index.md': ['invented_export']}, 'external_symbols': {}}
        monkeypatch.setattr(json, 'loads', lambda *args, **kwargs: inventory)
    else:
        monkeypatch.setattr(package, '__all__', [*package.__all__, 'invented_export'])
    with pytest.raises(ValueError, match='invented_export'):
        SEO['public_api_names']()
