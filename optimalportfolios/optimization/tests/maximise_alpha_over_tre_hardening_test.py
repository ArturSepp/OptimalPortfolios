"""End-to-end integration tests for the validate_solution wiring in the
TRE utility solver (``cvx_maximise_tre_utility`` / ``wrapper_maximise_alpha_over_tre``).

Unlike ``solver_diagnostics_test.py`` (which unit-tests the validator with
synthetic outputs), these run the *actually wired* solver:

* ``test_happy_path_*`` runs a real CLARABEL solve at the GROWM corner
  (tre_utility_weight=100, turnover_utility_weight=0.2) on a near-singular
  covariance (a corr-1.00 private-asset block) and asserts the returned weights
  are finite, fully invested, and in-box — i.e. the wiring does not disturb the
  normal path.

* ``test_finite_garbage_*`` monkeypatches the CVXPY solve to return the exact
  GROWM failure signature — a *finite* iterate that sums to ~5e5 with no proper
  status — and asserts the wired solver discards it and falls back to weights_0.
  This is the case the pre-fix ``if optimal_weights is None`` check accepted.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import cvxpy as cvx

from optimalportfolios.optimization.constraints import (
    Constraints, ConstraintEnforcementType, GroupLowerUpperConstraints,
)
from optimalportfolios.optimization.config import OptimiserConfig
from optimalportfolios.optimization.taa.maximise_alpha_over_tre import (
    wrapper_maximise_alpha_over_tre,
    cvx_maximise_tre_utility,
)

TICKERS = ["EQ", "BOND", "GOLD", "PE", "HF"]


def _near_singular_covar():
    """Covariance with a corr-1.00 PE/HF block → rank-deficient (the GROWM cause)."""
    vols = np.array([0.18, 0.05, 0.15, 0.10, 0.10])
    corr = np.eye(5)
    corr[0, 1] = corr[1, 0] = 0.10
    corr[0, 3] = corr[3, 0] = 0.40
    corr[0, 4] = corr[4, 0] = 0.40
    corr[3, 4] = corr[4, 3] = 1.00     # PE and HF identical → singular block
    cov = np.outer(vols, vols) * corr
    return pd.DataFrame((cov + cov.T) / 2, index=TICKERS, columns=TICKERS)


def _utility_constraints(weights_0=True):
    idx = pd.Index(TICKERS)
    bench = pd.Series([0.45, 0.35, 0.15, 0.05, 0.0], index=idx)
    w0 = pd.Series([0.44, 0.34, 0.16, 0.06, 0.0], index=idx) if weights_0 else None
    return Constraints(
        is_long_only=True,
        min_weights=pd.Series(0.0, index=idx),
        max_weights=pd.Series(0.40, index=idx),
        benchmark_weights=bench,
        weights_0=w0,
        constraint_enforcement_type=ConstraintEnforcementType.UTILITY_CONSTRAINTS,
        tre_utility_weight=100.0,        # the GROWM corner
        turnover_utility_weight=0.2,
    )


def test_happy_path_near_singular_covar_is_feasible():
    """Real solve on a rank-deficient covar at (100, 0.2) → feasible output."""
    pd_covar = _near_singular_covar()
    alphas = pd.Series([0.01, 0.005, 0.01, 0.04, 0.04], index=TICKERS)
    constraints = _utility_constraints()
    w = wrapper_maximise_alpha_over_tre(
        pd_covar=pd_covar,
        alphas=alphas,
        benchmark_weights=constraints.benchmark_weights,
        constraints=constraints,
        weights_0=constraints.weights_0,
        optimiser_config=OptimiserConfig(solver="CLARABEL"),
        context="GROWM 2021-04-30",
    )
    assert np.all(np.isfinite(w.to_numpy()))
    assert abs(w.sum() - 1.0) < 1e-4               # fully invested
    assert w.min() >= -1e-6                          # long-only
    assert w.max() <= 0.40 + 1e-6                    # box respected


def test_finite_garbage_solver_return_is_caught(monkeypatch, caplog):
    """Inject the GROWM signature — a finite iterate summing to ~5e5 with no
    valid status — and assert the wired solver discards it for weights_0."""
    idx = pd.Index(TICKERS)
    constraints = Constraints(
        is_long_only=True,
        min_weights=pd.Series(0.0, index=idx),
        max_weights=pd.Series(0.40, index=idx),
        benchmark_weights=pd.Series([0.45, 0.35, 0.15, 0.05, 0.0], index=idx),
        weights_0=pd.Series([0.44, 0.34, 0.16, 0.06, 0.0], index=idx),
        constraint_enforcement_type=ConstraintEnforcementType.UTILITY_CONSTRAINTS,
        tre_utility_weight=100.0,
        turnover_utility_weight=0.2,
    )
    covar = _near_singular_covar().to_numpy()
    alphas = np.array([0.01, 0.005, 0.01, 0.04, 0.04])

    # Make the "solver" return a finite blow-up iterate (sum ~5e5) and no status.
    def fake_solve(self, *args, **kwargs):
        for v in self.variables():
            v.value = np.full(v.shape[0], 1.0e5)
        return None

    monkeypatch.setattr(cvx.Problem, "solve", fake_solve)

    import logging
    with caplog.at_level(logging.WARNING,
                         logger="optimalportfolios.optimization.solver_diagnostics"):
        w = cvx_maximise_tre_utility(
            covar=covar, constraints=constraints, alphas=alphas,
            solver="CLARABEL", context="GROWM 2021-04-30")
    assert any("REJECTED" in r.getMessage() for r in caplog.records)

    # the 5e5 garbage was rejected; we fell back to the (feasible) weights_0
    assert np.all(np.isfinite(w))
    assert abs(float(np.sum(w)) - 1.0) < 1e-4
    np.testing.assert_allclose(w, constraints.weights_0.to_numpy())


def test_diagnose_flag_routes_rejected_solve_to_infeasibility_report(caplog):
    """The OptimiserConfig.diagnose_infeasibility path: box caps that sum to
    0.75 < 1.0 make full investment infeasible, so the solve is rejected. With
    diagnose=True the elastic diagnosis fires and names the bounds that must
    give; with diagnose=False only the one-line rejection is logged."""
    import logging
    idx = pd.Index(TICKERS)
    constraints = Constraints(
        is_long_only=True,
        min_weights=pd.Series(0.0, index=idx),
        max_weights=pd.Series(0.15, index=idx),      # 5 * 0.15 = 0.75 < 1 -> infeasible
        benchmark_weights=pd.Series(0.20, index=idx),
        weights_0=pd.Series(0.20, index=idx),
        constraint_enforcement_type=ConstraintEnforcementType.UTILITY_CONSTRAINTS,
        tre_utility_weight=100.0,
        turnover_utility_weight=0.2,
    )
    covar = _near_singular_covar().to_numpy()
    alphas = np.array([0.01, 0.005, 0.01, 0.04, 0.04])

    with caplog.at_level(logging.WARNING,
                         logger="optimalportfolios.optimization.solver_diagnostics"):
        cvx_maximise_tre_utility(covar=covar, constraints=constraints, alphas=alphas,
                                 solver="CLARABEL", context="t", diagnose=True)
    msgs = [r.getMessage() for r in caplog.records]
    assert any("REJECTED" in m for m in msgs)
    assert any("infeasibility diagnosis" in m for m in msgs)

    caplog.clear()
    with caplog.at_level(logging.WARNING,
                         logger="optimalportfolios.optimization.solver_diagnostics"):
        cvx_maximise_tre_utility(covar=covar, constraints=constraints, alphas=alphas,
                                 solver="CLARABEL", context="t", diagnose=False)
    msgs = [r.getMessage() for r in caplog.records]
    assert any("REJECTED" in m for m in msgs)
    assert not any("infeasibility diagnosis" in m for m in msgs)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
