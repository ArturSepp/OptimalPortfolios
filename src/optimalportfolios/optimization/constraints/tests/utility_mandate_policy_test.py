"""Policy tests for hard benchmark-relative mandates in utility solves."""
from __future__ import annotations

import cvxpy as cvx
import numpy as np
import pandas as pd
import pytest

from optimalportfolios.optimization.constraints import (
    BenchmarkDeviationConstraints,
    ConstraintEnforcementType,
    Constraints,
    evaluate_constraint_residuals,
)
from optimalportfolios.optimization.saa.max_return_target_vol import (
    cvx_max_return_target_vol_utility,
)


@pytest.mark.parametrize("solver_path", ["shared_builder", "saa_benchmark_relative"])
def test_utility_solve_keeps_sector_and_style_mandates_hard(solver_path: str) -> None:
    """Utility enforcement softens penalties, not sector or style mandate caps."""
    tickers = pd.Index(["asset_0", "asset_1", "asset_2"])
    covariance = np.eye(3)
    benchmark = pd.Series([1.0 / 3.0] * 3, index=tickers)
    alphas = np.array([10.0, 0.0, 0.0])
    sector_loadings = pd.DataFrame(
        {"favoured_sector": [1.0, 0.0, 0.0]}, index=tickers
    )
    style_loadings = pd.DataFrame(
        {"favoured_style": [1.0, -1.0, 0.0]}, index=tickers
    )
    sector_limits = pd.Series({"favoured_sector": 0.05})
    style_limits = pd.Series({"favoured_style": 0.05})
    constraint_spec = Constraints(
        benchmark_weights=benchmark,
        constraint_enforcement_type=ConstraintEnforcementType.UTILITY_CONSTRAINTS,
        tre_utility_weight=0.0,
        turnover_utility_weight=None,
        sector_deviation_constraints=BenchmarkDeviationConstraints(
            factor_loading_mat=sector_loadings,
            factor_max_deviation=sector_limits,
        ),
        style_deviation_constraints=BenchmarkDeviationConstraints(
            factor_loading_mat=style_loadings,
            factor_max_deviation=style_limits,
        ),
    )

    if solver_path == "shared_builder":
        w = cvx.Variable(len(tickers))
        objective, hard_constraints = (
            constraint_spec.set_cvx_utility_objective_constraints(
                w=w,
                alphas=alphas,
                covar=cvx.psd_wrap(covariance),
            )
        )
        problem = cvx.Problem(cvx.Maximize(objective), hard_constraints)
        problem.solve(solver=cvx.CLARABEL)
        assert problem.status == cvx.OPTIMAL
        weights = np.asarray(w.value, dtype=float)
    else:
        outcome = cvx_max_return_target_vol_utility(
            covar=covariance,
            alphas=alphas,
            constraints=constraint_spec,
            has_benchmark=True,
            factorize_covar=False,
        )
        weights = np.asarray(outcome.weights, dtype=float)

    active_weights = weights - benchmark.to_numpy()
    direct_checks = {
        "sector_deviation": (
            abs(float(sector_loadings["favoured_sector"].to_numpy() @ active_weights)),
            float(sector_limits["favoured_sector"]),
        ),
        "style_deviation": (
            abs(float(style_loadings["favoured_style"].to_numpy() @ active_weights)),
            float(style_limits["favoured_style"]),
        ),
    }
    residuals = evaluate_constraint_residuals(
        weights=weights,
        constraints=constraint_spec,
        covar=covariance,
    )
    mandate_residuals = {
        residual.constraint_type: residual
        for residual in residuals
        if residual.constraint_type in direct_checks
    }

    failures = []
    for kind, (actual, limit) in direct_checks.items():
        if actual > limit + 1e-6:
            failures.append(f"{kind} actual={actual:.6f} exceeds limit={limit:.6f}")
        residual = mandate_residuals.get(kind)
        if residual is None:
            failures.append(f"{kind} residual is missing")
        elif not residual.hard or not residual.passed:
            failures.append(
                f"{kind} residual hard={residual.hard}, passed={residual.passed}"
            )
    assert not failures, "; ".join(failures)
