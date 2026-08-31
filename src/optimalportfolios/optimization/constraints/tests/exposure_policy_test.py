"""Policy tests for exact exposure equality versus exposure bands."""
from __future__ import annotations

from types import SimpleNamespace

import cvxpy as cvx
import numpy as np
import pytest
from cvxpy.constraints.nonpos import Inequality
from cvxpy.constraints.zero import Equality

from optimalportfolios.optimization.constraints import Constraints
from optimalportfolios.optimization.solver_diagnostics import validate_scipy_solution


@pytest.mark.parametrize("with_exposure_scaler", [False, True])
def test_exposure_compiler_uses_stored_limit_identity(with_exposure_scaler: bool) -> None:
    """Only exactly equal stored limits compile to a single equality row."""
    w = cvx.Variable(2)
    exposure_scaler = cvx.Variable(nonneg=True) if with_exposure_scaler else None
    exact = Constraints(
        is_long_only=False,
        min_exposure=1.0,
        max_exposure=1.0,
    )
    band = Constraints(
        is_long_only=False,
        min_exposure=1.0,
        max_exposure=1.0 + 5e-6,
    )

    exact_rows = exact.set_cvx_exposure_constraints(
        w=w, exposure_scaler=exposure_scaler
    )
    band_rows = band.set_cvx_exposure_constraints(
        w=w, exposure_scaler=exposure_scaler
    )

    assert len(exact_rows) == 1
    assert isinstance(exact_rows[0], Equality)
    assert len(band_rows) == 2
    assert all(isinstance(row, Inequality) for row in band_rows)


def test_nearby_distinct_exposure_limits_remain_a_band_in_validation() -> None:
    """A vector on either edge of a stored band is feasible at strict tolerance."""
    constraint_spec = Constraints(
        min_exposure=1.0,
        max_exposure=1.0 + 5e-6,
    )
    candidate = np.array([0.6, 0.4])
    result = SimpleNamespace(
        success=True,
        status=0,
        message="Optimization terminated successfully",
    )

    returned_weights, accepted = validate_scipy_solution(
        optimal_weights=candidate,
        res=result,
        constraints=constraint_spec,
        n=len(candidate),
        budget_atol=1e-8,
    )

    assert accepted and np.allclose(returned_weights, candidate), (
        "validator collapsed distinct exposure limits into an equality: "
        f"accepted={accepted}, returned={returned_weights.tolist()}"
    )
