"""Characterization tests for CVXPY backend composition and numerical values."""
from __future__ import annotations

import cvxpy as cvx
import numpy as np
import pandas as pd
import pytest

from optimalportfolios.optimization.constraints import (
    BenchmarkBetaConstraint,
    BenchmarkDeviationConstraints,
    Constraints,
    GroupLowerUpperConstraints,
    GroupTrackingErrorConstraint,
    GroupTurnoverConstraint,
)
from optimalportfolios.optimization.covar_factorization import factorize_covariance


TICKERS = pd.Index(["asset_a", "asset_b", "asset_c"])
COVAR = np.array([
    [0.0400, 0.0060, 0.0000],
    [0.0060, 0.0225, 0.0030],
    [0.0000, 0.0030, 0.0100],
])
BENCHMARK = pd.Series([0.40, 0.35, 0.25], index=TICKERS)
WEIGHTS_0 = pd.Series([0.35, 0.30, 0.35], index=TICKERS)
PROBE_WEIGHTS = np.array([0.52, 0.21, 0.12])
GROUP_LOADINGS = pd.DataFrame(
    {
        "Growth": [1.0, 1.0, 0.0],
        "Defensive": [0.0, 0.0, 1.0],
    },
    index=TICKERS,
)
STYLE_LOADINGS = pd.DataFrame(
    {"Momentum": [1.0, -0.5, 0.25]},
    index=TICKERS,
)
MIN_WEIGHTS = pd.Series([0.00, 0.05, 0.00], index=TICKERS)
MAX_WEIGHTS = pd.Series([0.70, 0.60, 0.50], index=TICKERS)
ASSET_RETURNS = pd.Series([0.08, 0.04, 0.02], index=TICKERS)
TURNOVER_COSTS = pd.Series([2.0, 0.5, 1.5], index=TICKERS)
BETA_LOADINGS = pd.Series([1.2, 0.8, 0.4], index=TICKERS)


def _full_constraint_spec() -> Constraints:
    """Return a specification exercising every hard CVXPY compiler block."""
    return Constraints(
        is_long_only=True,
        min_weights=MIN_WEIGHTS,
        max_weights=MAX_WEIGHTS,
        max_exposure=1.10,
        min_exposure=0.80,
        benchmark_weights=BENCHMARK,
        tracking_err_vol_constraint=0.03,
        weights_0=WEIGHTS_0,
        turnover_constraint=0.22,
        turnover_costs=TURNOVER_COSTS,
        target_return=0.045,
        asset_returns=ASSET_RETURNS,
        max_target_portfolio_vol_an=0.30,
        group_lower_upper_constraints=GroupLowerUpperConstraints(
            group_loadings=GROUP_LOADINGS,
            group_min_allocation=pd.Series(
                {"Growth": 0.35, "Defensive": 0.10}),
            group_max_allocation=pd.Series(
                {"Growth": 0.85, "Defensive": 0.45}),
        ),
        group_tracking_error_constraint=GroupTrackingErrorConstraint(
            group_loadings=GROUP_LOADINGS,
            group_tre_vols=pd.Series(
                {"Growth": 0.12, "Defensive": 0.07}),
        ),
        group_turnover_constraint=GroupTurnoverConstraint(
            group_loadings=GROUP_LOADINGS,
            group_max_turnover=pd.Series(
                {"Growth": 0.18, "Defensive": 0.11}),
        ),
        sector_deviation_constraints=BenchmarkDeviationConstraints(
            factor_loading_mat=GROUP_LOADINGS,
            factor_max_deviation=pd.Series(
                {"Growth": 0.06, "Defensive": 0.04}),
        ),
        style_deviation_constraints=BenchmarkDeviationConstraints(
            factor_loading_mat=STYLE_LOADINGS,
            factor_max_deviation=pd.Series({"Momentum": 0.09}),
        ),
        benchmark_beta_constraint=BenchmarkBetaConstraint(
            beta_min=0.65,
            beta_max=1.05,
            beta_loadings=BETA_LOADINGS,
        ),
    )


def _expected_hard_rows(
        weights: np.ndarray,
) -> list[tuple[str, object, object]]:
    """Compute every expected hard-row argument directly with NumPy."""
    weight_sum = float(np.sum(weights))
    weight_change = weights - WEIGHTS_0.to_numpy()
    active_weights = weights - BENCHMARK.to_numpy()
    growth = GROUP_LOADINGS["Growth"].to_numpy()
    defensive = GROUP_LOADINGS["Defensive"].to_numpy()
    momentum = STYLE_LOADINGS["Momentum"].to_numpy()
    growth_active = growth * active_weights
    defensive_active = defensive * active_weights
    beta = float(BETA_LOADINGS.to_numpy() @ weights)

    return [
        ("long_only", np.zeros(len(TICKERS)), weights),
        ("maximum_exposure", weight_sum, 1.10),
        ("minimum_exposure", 0.80, weight_sum),
        ("minimum_weights", MIN_WEIGHTS.to_numpy(), weights),
        ("maximum_weights", weights, MAX_WEIGHTS.to_numpy()),
        ("target_return", 0.045, float(ASSET_RETURNS.to_numpy() @ weights)),
        ("maximum_volatility", float(weights @ COVAR @ weights), 0.30 ** 2),
        (
            "growth_turnover",
            float(np.sum(np.abs(growth * weight_change))),
            0.18,
        ),
        (
            "defensive_turnover",
            float(np.sum(np.abs(defensive * weight_change))),
            0.11,
        ),
        (
            "total_turnover",
            float(np.sum(np.abs(TURNOVER_COSTS.to_numpy() * weight_change))),
            0.22,
        ),
        (
            "growth_tracking_error",
            float(growth_active @ COVAR @ growth_active),
            0.12 ** 2,
        ),
        (
            "defensive_tracking_error",
            float(defensive_active @ COVAR @ defensive_active),
            0.07 ** 2,
        ),
        (
            "total_tracking_error",
            float(active_weights @ COVAR @ active_weights),
            0.03 ** 2,
        ),
        ("growth_minimum", 0.35, float(growth @ weights)),
        ("growth_maximum", float(growth @ weights), 0.85),
        ("defensive_minimum", 0.10, float(defensive @ weights)),
        ("defensive_maximum", float(defensive @ weights), 0.45),
        (
            "growth_deviation",
            float(np.abs(growth @ active_weights)),
            0.06,
        ),
        (
            "defensive_deviation",
            float(np.abs(defensive @ active_weights)),
            0.04,
        ),
        (
            "momentum_deviation",
            float(np.abs(momentum @ active_weights)),
            0.09,
        ),
        ("minimum_beta", 0.65, beta),
        ("maximum_beta", beta, 1.05),
    ]


def _assert_row_arguments(
        rows: list,
        expected: list[tuple[str, object, object]],
) -> None:
    """Assert CVXPY inequality order and evaluated left/right arguments."""
    assert len(rows) == len(expected)
    for row, (label, expected_left, expected_right) in zip(rows, expected):
        assert row.__class__.__name__ == "Inequality", label
        assert len(row.args) == 2, label
        np.testing.assert_allclose(
            np.asarray(row.args[0].value, dtype=float),
            np.asarray(expected_left, dtype=float),
            rtol=0.0,
            atol=1e-12,
            err_msg=f"unexpected left argument for {label}",
        )
        np.testing.assert_allclose(
            np.asarray(row.args[1].value, dtype=float),
            np.asarray(expected_right, dtype=float),
            rtol=0.0,
            atol=1e-12,
            err_msg=f"unexpected right argument for {label}",
        )


def _expected_utility_hard_rows(
        weights: np.ndarray,
) -> list[tuple[str, object, object]]:
    """Select the established hard-row subset and ordering used by utility compilation."""
    utility_labels = {
        "long_only",
        "maximum_exposure",
        "minimum_exposure",
        "minimum_weights",
        "maximum_weights",
        "target_return",
        "growth_minimum",
        "growth_maximum",
        "defensive_minimum",
        "defensive_maximum",
        "growth_deviation",
        "defensive_deviation",
        "momentum_deviation",
        "minimum_beta",
        "maximum_beta",
    }
    return [row for row in _expected_hard_rows(weights) if row[0] in utility_labels]


def test_hard_compiler_preserves_complete_row_order_and_values() -> None:
    """All hard blocks retain exact row order, including additive group and total TE."""
    constraint_spec = _full_constraint_spec()
    weights = cvx.Variable(len(TICKERS))

    rows = constraint_spec.set_cvx_all_constraints(
        w=weights,
        covar=cvx.psd_wrap(COVAR),
    )
    weights.value = PROBE_WEIGHTS

    _assert_row_arguments(rows, _expected_hard_rows(PROBE_WEIGHTS))


@pytest.mark.parametrize(
    "factorize_covar",
    [pytest.param(False, id="quadratic"), pytest.param(True, id="factorized")],
)
def test_total_utility_objective_and_risk_match_direct_numpy(
        factorize_covar: bool,
) -> None:
    """Total turnover and TRE utility terms equal direct NumPy values on both risk paths."""
    constraint_spec = _full_constraint_spec().copy(
        group_tracking_error_constraint=None,
        group_turnover_constraint=None,
        tre_utility_weight=3.5,
        turnover_utility_weight=1.25,
    )
    alphas = np.array([0.030, 0.010, -0.015])
    factorization = factorize_covariance(COVAR) if factorize_covar else None
    risk_covar = factorization.covar if factorization is not None else COVAR
    weights = cvx.Variable(len(TICKERS))

    objective, rows = constraint_spec.set_cvx_utility_objective_constraints(
        w=weights,
        alphas=alphas,
        covar=cvx.psd_wrap(risk_covar),
        covar_factorization=factorization,
    )
    weights.value = PROBE_WEIGHTS

    active_weights = PROBE_WEIGHTS - BENCHMARK.to_numpy()
    weight_change = PROBE_WEIGHTS - WEIGHTS_0.to_numpy()
    alpha_reward = float(alphas @ active_weights)
    turnover_penalty = 1.25 * float(
        np.sum(np.abs(TURNOVER_COSTS.to_numpy() * weight_change)))
    direct_risk = float(active_weights @ risk_covar @ active_weights)
    expected_objective = alpha_reward - turnover_penalty - 3.5 * direct_risk

    assert float(objective.value) == pytest.approx(expected_objective, abs=1e-12)
    compiled_risk = (
        alpha_reward - turnover_penalty - float(objective.value)
    ) / 3.5
    assert compiled_risk == pytest.approx(direct_risk, abs=1e-12)
    _assert_row_arguments(rows, _expected_utility_hard_rows(PROBE_WEIGHTS))


def test_group_utility_terms_keep_precedence_and_match_direct_numpy() -> None:
    """Group penalties suppress total penalties and retain the utility hard-row subset."""
    group_turnover_weights = pd.Series(
        {"Growth": 1.7, "Defensive": 2.3})
    group_tre_weights = pd.Series(
        {"Growth": 4.0, "Defensive": 6.0})
    constraint_spec = _full_constraint_spec().copy(
        group_turnover_constraint=GroupTurnoverConstraint(
            group_loadings=GROUP_LOADINGS,
            group_max_turnover=pd.Series(
                {"Growth": 0.18, "Defensive": 0.11}),
            group_turnover_utility_weights=group_turnover_weights,
        ),
        group_tracking_error_constraint=GroupTrackingErrorConstraint(
            group_loadings=GROUP_LOADINGS,
            group_tre_vols=pd.Series(
                {"Growth": 0.12, "Defensive": 0.07}),
            group_tre_utility_weights=group_tre_weights,
        ),
        turnover_utility_weight=97.0,
        tre_utility_weight=103.0,
    )
    alphas = np.array([0.030, 0.010, -0.015])
    weights = cvx.Variable(len(TICKERS))

    objective, rows = constraint_spec.set_cvx_utility_objective_constraints(
        w=weights,
        alphas=alphas,
        covar=cvx.psd_wrap(COVAR),
    )
    weights.value = PROBE_WEIGHTS

    active_weights = PROBE_WEIGHTS - BENCHMARK.to_numpy()
    weight_change = PROBE_WEIGHTS - WEIGHTS_0.to_numpy()
    expected_objective = float(alphas @ active_weights)
    for group in GROUP_LOADINGS.columns:
        loading = GROUP_LOADINGS[group].to_numpy()
        group_active = loading * active_weights
        expected_objective -= group_turnover_weights.loc[group] * float(
            np.sum(np.abs(loading * weight_change)))
        expected_objective -= group_tre_weights.loc[group] * float(
            group_active @ COVAR @ group_active)

    assert float(objective.value) == pytest.approx(expected_objective, abs=1e-12)
    _assert_row_arguments(rows, _expected_utility_hard_rows(PROBE_WEIGHTS))
