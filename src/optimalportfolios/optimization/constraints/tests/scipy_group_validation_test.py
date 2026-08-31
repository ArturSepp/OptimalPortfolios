"""Acceptance-policy tests for SciPy group allocation constraints."""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from optimalportfolios.optimization.constraints import (
    Constraints,
    GroupLowerUpperConstraints,
)
from optimalportfolios.optimization.constraints.analytics import _iter_finite_group_bounds
from optimalportfolios.optimization.solver_diagnostics import validate_scipy_solution


def test_scipy_validator_rejects_a_compiled_group_cap_breach() -> None:
    """A successful SciPy status cannot override a violated emitted group row."""
    tickers = pd.Index(["asset_0", "asset_1", "asset_2"])
    candidate = np.array([0.7, 0.2, 0.1])
    fallback = pd.Series([0.2, 0.2, 0.6], index=tickers)
    group_loading = pd.DataFrame({"risky": [1.0, 1.0, 0.0]}, index=tickers)
    group_cap = 0.5
    constraint_spec = Constraints(
        min_weights=pd.Series(0.0, index=tickers),
        max_weights=pd.Series(0.8, index=tickers),
        weights_0=fallback,
        group_lower_upper_constraints=GroupLowerUpperConstraints(
            group_loadings=group_loading,
            group_min_allocation=None,
            group_max_allocation=pd.Series({"risky": group_cap}),
        ),
    )
    scipy_constraints, bounds = constraint_spec.set_scipy_constraints(
        covar=np.eye(len(tickers))
    )

    assert candidate.sum() == pytest.approx(1.0)
    assert np.all(candidate >= bounds[:, 0])
    assert np.all(candidate <= bounds[:, 1])
    group_actual = float(group_loading["risky"].to_numpy() @ candidate)
    breach = group_actual - group_cap
    scalar_callback_values = [
        float(value)
        for item in scipy_constraints
        if np.asarray(value := item["fun"](candidate)).ndim == 0
    ]
    group_callback_value = min(scalar_callback_values)
    assert breach == pytest.approx(0.4)
    assert group_callback_value == pytest.approx(-breach)

    result = SimpleNamespace(
        success=True,
        status=0,
        message="Optimization terminated successfully",
    )
    returned_weights, accepted = validate_scipy_solution(
        optimal_weights=candidate,
        res=result,
        constraints=constraint_spec,
        n=len(tickers),
    )

    rejected_with_fallback = (
        accepted is False
        and np.allclose(returned_weights, fallback.to_numpy())
    )
    assert rejected_with_fallback, (
        f"validator accepted={accepted} and returned {returned_weights.tolist()} "
        f"despite emitted group callback={group_callback_value:.6f}"
    )


def test_group_bound_iteration_ignores_a_container_without_loadings() -> None:
    """An incomplete optional group block contributes no auditable rows."""
    group_constraints = SimpleNamespace(
        group_loadings=None,
        group_min_allocation=pd.Series({"unused": 0.0}),
        group_max_allocation=pd.Series({"unused": 1.0}),
    )

    assert list(_iter_finite_group_bounds(group_constraints)) == []


def test_scipy_validator_ignores_an_all_zero_group_loading() -> None:
    """A structurally empty group cannot reject an otherwise valid candidate."""
    tickers = pd.Index(["asset_0", "asset_1"])
    candidate = np.array([0.4, 0.6])
    constraints = Constraints(
        weights_0=pd.Series(candidate, index=tickers),
        group_lower_upper_constraints=GroupLowerUpperConstraints(
            group_loadings=pd.DataFrame({"empty": [1.0, 0.0]}, index=tickers),
            group_min_allocation=None,
            group_max_allocation=pd.Series({"empty": 0.5}),
        ),
    )
    object.__setattr__(
        constraints.group_lower_upper_constraints,
        "group_loadings",
        pd.DataFrame({"empty": [0.0, 0.0]}, index=tickers),
    )
    result = SimpleNamespace(success=True, status=0, message="ok")

    returned_weights, accepted = validate_scipy_solution(
        optimal_weights=candidate,
        res=result,
        constraints=constraints,
        n=len(tickers),
    )

    assert accepted is True
    assert np.array_equal(returned_weights, candidate)
