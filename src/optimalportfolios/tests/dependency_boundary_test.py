"""Guard OP's supplied-input boundary in source and installed distributions."""
import ast
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

import optimalportfolios


def test_library_has_no_factor_package_imports():
    """Reject even lazy imports: fitting and statistical preparation belong upstream."""
    package = Path(optimalportfolios.__file__).parent
    violations = []
    inspected = 0
    for path in package.rglob('*.py'):
        relative = path.relative_to(package)
        if 'tests' in relative.parts or 'run_local' in relative.parts:
            continue
        inspected += 1
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
            modules = ([name.name for name in node.names] if isinstance(node, ast.Import)
                       else [node.module or ''] if isinstance(node, ast.ImportFrom) else [])
            for module in modules:
                if module.split('.')[0] in {'factorlasso', 'tradablefactors'}:
                    violations.append(f'{relative}:{node.lineno}: {module}')
    assert inspected > 50
    assert not violations, violations


def test_removed_modules_are_not_shipped():
    """The clean migration cannot retain old statistical modules as forwarding shims."""
    package = Path(optimalportfolios.__file__).parent
    removed = ('covar_estimation/risk_labelling.py', 'covar_estimation/factor_covar_estimator.py',
               'covar_estimation/risk_model_adapter.py',
               'matf/uncertainty.py', 'matf/universe_diagnostics.py', 'matf/gls_diagnostic.py',
               'matf/sharpe_accounting.py', 'matf/cma_diagnostics.py')
    assert not [name for name in removed if (package / name).exists()]


def test_missing_factor_installation_fails_required_lane(monkeypatch):
    """An explicitly required composed lane must fail rather than silently skip everything."""
    hooks = importlib.import_module('optimalportfolios.conftest')
    monkeypatch.setattr(hooks.importlib.util, 'find_spec', lambda name: None)
    monkeypatch.setenv('OP_REQUIRE_FACTOR_INTEGRATION', '1')
    with pytest.raises(pytest.UsageError, match='requires tradablefactors'):
        hooks.pytest_collection_modifyitems(None, [])


@pytest.mark.parametrize('installed', [False, True])
def test_base_checks_are_never_skipped_by_factor_lane(monkeypatch, installed):
    """Only explicitly composed cases can be skipped; required base cases always execute."""
    hooks = importlib.import_module('optimalportfolios.conftest')
    monkeypatch.setattr(hooks.importlib.util, 'find_spec',
                        lambda name: object() if installed else None)
    monkeypatch.delenv('OP_REQUIRE_FACTOR_INTEGRATION', raising=False)
    base_marks, composed_marks = [], []
    base = SimpleNamespace(get_closest_marker=lambda name: None, add_marker=base_marks.append)
    composed = SimpleNamespace(
        get_closest_marker=lambda name: True if name == 'factor_integration' else None,
        add_marker=composed_marks.append,
    )
    hooks.pytest_collection_modifyitems(None, [base, composed])
    assert not base_marks
    assert len(composed_marks) == (0 if installed else 1)
