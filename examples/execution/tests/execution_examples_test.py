"""Protect the supplied domain and honest native reporting in the teaching examples."""

from types import SimpleNamespace

import cvxpy as cp
import numpy as np
import pytest

from examples.execution import native_evidence, pair_problem, portfolio_problem
from examples.execution import compare_methods, coordinated_trades, policy


@pytest.mark.parametrize("problem", [portfolio_problem(), pair_problem(2)])
def test_prepared_corridors(problem):
    """Require real current-to-target corridors rather than relabelled hard bounds."""
    noncash = np.arange(len(problem.names)) != problem.cash
    np.testing.assert_array_equal(
        problem.lower[noncash], np.minimum(problem.current, problem.target)[noncash])
    np.testing.assert_array_equal(
        problem.upper[noncash], np.maximum(problem.current, problem.target)[noncash])
    assert np.all(problem.full_lower <= problem.lower)
    assert np.all(problem.full_upper >= problem.upper)


def test_core_examples_without_native(monkeypatch, capsys):
    """Exercise default commands with MOSEK and SCIP explicitly absent."""
    monkeypatch.setattr(cp, "installed_solvers", lambda: ["CLARABEL", "SCS", "OSQP"])
    policy.main()
    assert "Engine: ranked_audited_fallback" in capsys.readouterr().out
    coordinated_trades.main()
    assert "Ranked K=2" in capsys.readouterr().out
    monkeypatch.setattr("sys.argv", ["compare_methods"])
    compare_methods.main()
    assert "Exact: NOT RUN. Kernel: NOT RUN." in capsys.readouterr().out


@pytest.mark.parametrize("method", ["Exact", "Kernel"])
def test_prepared_incumbent_is_not_native_success(method):
    """Passing the portfolio audit alone must never qualify a native comparison."""
    info = {"native_status": "solver_unavailable", "subproblems": [],
            "incumbent_events": [{"source": "common_prepared_incumbent",
                                  "audit": {"passed": True}}]}
    assert not native_evidence(method, SimpleNamespace(diagnostics=info))["native_usable"]


@pytest.mark.parametrize("method", ["Exact", "Kernel"])
def test_native_error_rejects_audited_candidate(method):
    """An audited native witness does not erase a backend error."""
    info = {"native_status": "solver_error", "error": "unavailable backend",
            "subproblems": [{"error": "unavailable backend"}],
            "incumbent_events": [{"source": "native_return", "audit": {"passed": True}}]}
    assert not native_evidence(method, SimpleNamespace(diagnostics=info))["native_usable"]
