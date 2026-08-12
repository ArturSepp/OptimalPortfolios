"""Regression tests for alpha maximisation with a target yield."""

import numpy as np
import pandas as pd

from optimalportfolios.optimization.constraints import Constraints
from optimalportfolios.optimization.taa.maximise_alpha_with_target_yield import (
    cvx_maximise_alpha_with_target_return,
)


TICKERS = pd.Index(["A", "B", "C", "D"])
VOLS = np.array([0.20, 0.12, 0.08, 0.05])
CORR = np.array([
    [1.0, 0.3, 0.1, 0.0],
    [0.3, 1.0, 0.2, 0.1],
    [0.1, 0.2, 1.0, 0.25],
    [0.0, 0.1, 0.25, 1.0],
])
COVAR = np.outer(VOLS, VOLS) * CORR
ALPHAS = np.array([0.04, 0.02, -0.01, -0.02])
YIELDS = pd.Series([0.01, 0.02, 0.04, 0.05], index=TICKERS)
BENCHMARK = pd.Series(0.25, index=TICKERS)


def _constraints(
        hard_te: float | None,
        turnover: float | None = None,
) -> Constraints:
    """Create the issue-49 constraint geometry."""
    return Constraints(
        is_long_only=True,
        min_weights=pd.Series(0.0, index=TICKERS),
        max_weights=pd.Series(1.0, index=TICKERS),
        min_exposure=1.0,
        max_exposure=1.0,
        asset_returns=YIELDS,
        target_return=0.045,
        benchmark_weights=BENCHMARK,
        weights_0=BENCHMARK if turnover is not None else None,
        turnover_constraint=turnover,
        tracking_err_vol_constraint=hard_te,
        tre_utility_weight=10.0,
    )


def test_soft_tracking_error_ignores_a_populated_hard_budget() -> None:
    """Soft TE accepts the solved geometry even when a hard budget is populated."""
    with_hard_budget = cvx_maximise_alpha_with_target_return(
        covar=COVAR,
        alphas=ALPHAS,
        constraints=_constraints(hard_te=0.0001),
        soft_tracking_error=True,
    )
    utility_only = cvx_maximise_alpha_with_target_return(
        covar=COVAR,
        alphas=ALPHAS,
        constraints=_constraints(hard_te=None),
        soft_tracking_error=True,
    )

    assert with_hard_budget.status == "optimal"
    assert with_hard_budget.accepted
    assert with_hard_budget.fallback_source is None
    assert float(YIELDS.to_numpy() @ with_hard_budget.weights) >= 0.045 - 1e-8
    active_weights = with_hard_budget.weights - BENCHMARK.to_numpy()
    tracking_error = float(np.sqrt(active_weights @ COVAR @ active_weights))
    assert tracking_error > 0.0001
    np.testing.assert_allclose(with_hard_budget.weights, utility_only.weights, atol=1e-7)
    assert with_hard_budget.constraints is not None
    assert with_hard_budget.constraints.tracking_err_vol_constraint is None


def test_hard_tracking_error_still_rejects_the_infeasible_geometry() -> None:
    """The ordinary hard-TE path retains the conflicting hard budget."""
    outcome = cvx_maximise_alpha_with_target_return(
        covar=COVAR,
        alphas=ALPHAS,
        constraints=_constraints(hard_te=0.0001),
        soft_tracking_error=False,
    )

    assert not outcome.accepted
    assert outcome.fallback_source == "benchmark_weights"
    np.testing.assert_allclose(outcome.weights, BENCHMARK.to_numpy())


def test_soft_tracking_error_keeps_total_turnover_hard() -> None:
    """Softening TE does not soften the separately re-added turnover budget."""
    outcome = cvx_maximise_alpha_with_target_return(
        covar=COVAR,
        alphas=ALPHAS,
        constraints=_constraints(hard_te=0.0001, turnover=1.0),
        soft_tracking_error=True,
    )

    assert outcome.accepted
    turnover = float(np.abs(outcome.weights - BENCHMARK.to_numpy()).sum())
    assert turnover <= 1.0 + 1e-4
    turnover_residuals = [
        residual
        for residual in outcome.constraint_residuals
        if residual.constraint_type == "turnover"
    ]
    assert len(turnover_residuals) == 1
    assert turnover_residuals[0].hard
    assert turnover_residuals[0].passed
