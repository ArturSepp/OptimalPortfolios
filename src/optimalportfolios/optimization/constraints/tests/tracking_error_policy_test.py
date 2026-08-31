"""Numerical policy tests for combined tracking-error constraints."""
from __future__ import annotations

import cvxpy as cvx
import numpy as np
import pandas as pd
import pytest

from optimalportfolios.optimization.constraints import (
    Constraints,
    GroupTrackingErrorConstraint,
)
from optimalportfolios.optimization.covar_factorization import factorize_covariance


@pytest.mark.parametrize(
    "factorize_covar",
    [pytest.param(False, id="quadratic"), pytest.param(True, id="factorized")],
)
def test_group_and_total_tracking_error_caps_are_additive(factorize_covar: bool) -> None:
    """A loose group cap must not disable an independently tight total TE cap."""
    tickers = pd.Index(["asset_0", "asset_1"])
    covariance = np.eye(2)
    benchmark = pd.Series([0.5, 0.5], index=tickers)
    group_loadings = pd.DataFrame(np.eye(2), index=tickers, columns=tickers)
    group_limits = pd.Series(1.0, index=group_loadings.columns)
    total_limit = 0.10
    constraint_spec = Constraints(
        benchmark_weights=benchmark,
        tracking_err_vol_constraint=total_limit,
        group_tracking_error_constraint=GroupTrackingErrorConstraint(
            group_loadings=group_loadings,
            group_tre_vols=group_limits,
        ),
    )
    factorization = factorize_covariance(covariance) if factorize_covar else None
    w = cvx.Variable(len(tickers))
    compiled_constraints = constraint_spec.set_cvx_all_constraints(
        w=w,
        covar=cvx.psd_wrap(covariance),
        covar_factorization=factorization,
    )
    problem = cvx.Problem(cvx.Maximize(w[0]), compiled_constraints)

    problem.solve(solver=cvx.CLARABEL)

    assert problem.status == cvx.OPTIMAL
    active_weights = np.asarray(w.value, dtype=float) - benchmark.to_numpy()
    for group, group_limit in group_limits.items():
        group_active = active_weights * group_loadings[group].to_numpy()
        group_tracking_error = float(
            np.sqrt(group_active @ covariance @ group_active)
        )
        assert group_tracking_error <= group_limit + 1e-6
    total_tracking_error = float(
        np.sqrt(active_weights @ covariance @ active_weights)
    )
    assert total_tracking_error <= total_limit + 1e-6
