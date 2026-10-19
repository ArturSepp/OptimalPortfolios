"""Keep the optional quadprog backend outside OP's ordinary import boundary."""
import builtins
import importlib
import importlib.metadata
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest

import optimalportfolios as op
from optimalportfolios.optimization.risk_allocation.risk_budgeting_solver import (
    _project_polyhedron,
    solve_constrained_risk_budgeting,
)


def test_quadprog_is_only_a_riskbudget_extra():
    """Installed metadata must not make the GPL backend a base requirement."""
    metadata = importlib.metadata.metadata("optimalportfolios")
    requirements = [r for r in metadata.get_all("Requires-Dist", [])
                    if r.lower().startswith("quadprog")]
    assert len(requirements) == 1
    assert requirements[0].replace(" ", "") == 'quadprog>=0.1.11;extra=="riskbudget"'
    assert "riskbudget" in metadata.get_all("Provides-Extra", [])


def test_ordinary_import_does_not_attempt_quadprog(tmp_path):
    """Reject an eager backend import even in an environment that has the extra."""
    script = '''
import importlib.abc
import sys
class NoQuadprog(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] == 'quadprog':
            raise AssertionError('OP ordinary import attempted quadprog')
        return None
sys.meta_path.insert(0, NoQuadprog())
import optimalportfolios
assert 'quadprog' not in sys.modules
'''
    result = subprocess.run([sys.executable, "-c", script], cwd=tmp_path,
                            capture_output=True, text=True, timeout=180, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("entry", ["projection", "solver", "public_wrapper"])
def test_missing_quadprog_has_actionable_error(monkeypatch, entry):
    """The missing extra propagates through the public API rather than returning fallbacks."""
    original = builtins.__import__

    def absent_backend(name, *args, **kwargs):
        """Model precisely the unavailable optional module, regardless of installed extras."""
        if name == "quadprog":
            raise ModuleNotFoundError("No module named 'quadprog'", name="quadprog")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", absent_backend)
    with pytest.raises(ImportError, match=r'pip install "optimalportfolios\[riskbudget\]"'):
        if entry == "projection":
            _project_polyhedron(np.array([.5, .5]), -np.eye(2), np.zeros(2))
        elif entry == "solver":
            solve_constrained_risk_budgeting(np.eye(2))
        else:
            op.opt_risk_budgeting(np.eye(2), op.Constraints(is_long_only=True))


def test_broken_backend_dependency_is_not_reported_as_missing_extra(monkeypatch):
    """An installed but broken backend retains its original dependency diagnostic."""
    original = builtins.__import__

    def broken_backend(name, *args, **kwargs):
        """Simulate a different missing module during backend import."""
        if name == "quadprog":
            raise ModuleNotFoundError("missing backend dependency", name="backend_dependency")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken_backend)
    with pytest.raises(ModuleNotFoundError) as caught:
        _project_polyhedron(np.array([.5, .5]), -np.eye(2), np.zeros(2))
    assert caught.value.name == "backend_dependency"


def test_required_riskbudget_lane_rejects_missing_extra(monkeypatch):
    """A required backend lane must fail rather than report an all-skipped success."""
    hooks = importlib.import_module("optimalportfolios.conftest")
    monkeypatch.setattr(hooks.importlib.util, "find_spec", lambda name: None)
    monkeypatch.delenv("OP_REQUIRE_FACTOR_INTEGRATION", raising=False)
    monkeypatch.setenv("OP_REQUIRE_RISKBUDGET", "1")
    with pytest.raises(pytest.UsageError, match="requires optimalportfolios"):
        hooks.pytest_collection_modifyitems(None, [])


@pytest.mark.parametrize("installed", [False, True])
def test_only_marked_backend_cases_are_skipped(monkeypatch, installed):
    """Base assertions remain meaningful without a backend installation."""
    hooks = importlib.import_module("optimalportfolios.conftest")
    monkeypatch.setattr(hooks.importlib.util, "find_spec",
                        lambda name: object() if installed else None)
    monkeypatch.delenv("OP_REQUIRE_FACTOR_INTEGRATION", raising=False)
    monkeypatch.delenv("OP_REQUIRE_RISKBUDGET", raising=False)
    base_marks, backend_marks = [], []
    base = SimpleNamespace(get_closest_marker=lambda name: None, add_marker=base_marks.append)
    backend = SimpleNamespace(
        get_closest_marker=lambda name: True if name == "riskbudget" else None,
        add_marker=backend_marks.append,
    )
    hooks.pytest_collection_modifyitems(None, [base, backend])
    assert not base_marks
    assert len(backend_marks) == (0 if installed else 1)
