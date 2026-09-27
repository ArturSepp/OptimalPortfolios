"""Verify the teaching-exhibit registry, generation, publication and verification.

The teaching exhibits are drawn by the canonical scripts of the documentation pages. These tests
use the committed exhibits, and a toy script in a temporary tree for the failure cases, so no
figure of a real page is drawn here.
"""

import importlib
import json
import shutil
import sys
from unittest.mock import patch

import pytest


@pytest.fixture(scope='module')
def teaching(root):
    """Import the repository-only tool after root has skipped installed-wheel checks."""
    with patch.object(sys, 'path', [str(root), *sys.path]):
        module = importlib.import_module('tools.docs_analytics.teaching')
        registry = importlib.import_module('tools.docs_analytics.registry')
    return module, registry


def test_committed_exhibits_verify(root, teaching):
    """Every committed teaching exhibit matches its manifest and its current script."""
    module, registry = teaching
    exhibits = module.load_teaching(root)
    assert exhibits, 'The teaching registry must list the committed exhibits.'
    manifest = module.verify(root)
    assert set(manifest['exhibits']) == {exhibit['id'] for exhibit in exhibits}
    assert manifest['review']['note']
    pairs = registry.teaching_pairs(root)
    for exhibit in exhibits:
        for document in exhibit['documents']:
            assert (document, exhibit['path']) in pairs


TOY_SCRIPT = '''"""Toy canonical script."""
import pandas as pd

SIZE = 3
PASS = True


def exhibit(path):
    """Write a placeholder image and report the checks."""
    path.write_bytes(b'toy image ' + str(SIZE).encode())
    return {'table': pd.DataFrame({'x': range(SIZE)}), 'checks': {'ok': PASS}}
'''


@pytest.fixture
def toy(tmp_path):
    """A minimal tree with one registered toy exhibit and its consumer page."""
    (tmp_path / 'examples/docs').mkdir(parents=True)
    (tmp_path / 'examples/docs/toy.py').write_text(TOY_SCRIPT, encoding='utf-8')
    (tmp_path / 'docs').mkdir()
    (tmp_path / 'docs/page.md').write_text('![toy](images/toy_exhibit.png)\n', encoding='utf-8')
    (tmp_path / 'tools/docs_analytics').mkdir(parents=True)
    registry = {'schema_version': 1, 'exhibits': [{
        'id': 'toy_exhibit', 'path': 'docs/images/toy_exhibit.png',
        'script': 'examples/docs/toy.py', 'function': 'exhibit', 'parameters': {'SIZE': 3},
        'documents': ['docs/page.md'], 'question': 'Toy?', 'sample': 'None.'}]}
    save(tmp_path, registry)
    return tmp_path, registry


def save(root, registry):
    """Write a fixture teaching registry."""
    (root / 'tools/docs_analytics/teaching.json').write_text(json.dumps(registry),
                                                            encoding='utf-8')


def test_generate_publish_verify_round_trip(toy, teaching):
    """A passing bundle publishes, and verification then catches image and script drift."""
    root, _ = toy
    module, _ = teaching
    bundle = root / 'bundle'
    module.generate(bundle, root)
    with pytest.raises(ValueError, match='existing output'):
        module.generate(bundle, root)
    with pytest.raises(ValueError, match='review note'):
        module.publish(bundle, ' ', root)
    module.publish(bundle, 'Inspected the toy image.', root)
    assert module.verify(root)['review']['note'] == 'Inspected the toy image.'
    image = root / 'docs/images/toy_exhibit.png'
    image.write_bytes(image.read_bytes() + b'!')
    with pytest.raises(ValueError, match='differs from its manifest'):
        module.verify(root)
    shutil.copyfile(bundle / 'images/toy_exhibit.png', image)
    module.verify(root)
    script = root / 'examples/docs/toy.py'
    script.write_text(TOY_SCRIPT.replace('"""Toy', '"""Edited toy'), encoding='utf-8')
    with pytest.raises(ValueError, match='script changed'):
        module.verify(root)
    with pytest.raises(ValueError, match='stale'):
        module.publish(bundle, 'Inspected again.', root)
    (root / 'docs/images/extra.png').write_bytes(b'x')
    script.write_text(TOY_SCRIPT, encoding='utf-8')
    with pytest.raises(ValueError, match='Unregistered or missing'):
        module.verify(root)


@pytest.mark.parametrize('change,message', [
    (lambda s: s.replace('SIZE = 3', 'SIZE = 4'), 'differs from the registered value'),
    (lambda s: s.replace('PASS = True', 'PASS = False'), 'failed checks'),
])
def test_generation_refuses_drift_and_failing_checks(toy, teaching, change, message):
    """A constant that differs from the registry, or a failing check, stops generation."""
    root, _ = toy
    module, _ = teaching
    script = root / 'examples/docs/toy.py'
    script.write_text(change(TOY_SCRIPT), encoding='utf-8')
    with pytest.raises(ValueError, match=message):
        module.generate(root / 'bundle', root)


@pytest.mark.parametrize('field,value,message', [
    ('path', 'examples/figures/toy_exhibit.png', 'docs/images/<id>.png'),
    ('script', 'examples/toy.py', 'examples/docs script'),
    ('parameters', {}, 'fixed inputs'),
    ('documents', [], 'consumer list'),
    ('question', '', 'needs a question'),
])
def test_registry_rejects_invalid_records(toy, teaching, field, value, message):
    """Each malformed record is rejected with its diagnostic."""
    root, registry = toy
    module, _ = teaching
    registry['exhibits'][0][field] = value
    save(root, registry)
    with pytest.raises(ValueError, match=message):
        module.load_teaching(root)


def test_script_hash_ignores_line_endings(toy, teaching):
    """A checkout that converts a script to CRLF still verifies against an LF-built manifest."""
    root, _ = toy
    module, _ = teaching
    script = root / 'examples/docs/toy.py'
    script.write_bytes(TOY_SCRIPT.encode('utf-8'))
    lf = module.text_sha256(script)
    script.write_bytes(TOY_SCRIPT.encode('utf-8').replace(b'\n', b'\r\n'))
    assert module.text_sha256(script) == lf
    assert module.sha256(script) != lf


def test_absent_registry_means_no_exhibits(tmp_path, teaching):
    """A tree without teaching exhibits verifies trivially, unless a stray manifest exists."""
    module, _ = teaching
    assert module.load_teaching(tmp_path) == []
    assert module.verify(tmp_path) == {}
    (tmp_path / 'docs/images').mkdir(parents=True)
    (tmp_path / 'docs/images/analytics_manifest.json').write_text('{}', encoding='utf-8')
    with pytest.raises(ValueError, match='without registered exhibits'):
        module.verify(tmp_path)
