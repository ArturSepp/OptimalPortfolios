"""Cash-funded selected-trade execution optimisation tests."""

from optimalportfolios import OptimiserConfig as _PortableOptimiserConfig


import numpy as np


import pandas as pd


import pytest


from optimalportfolios import Constraints, GroupLowerUpperConstraints, OptimizationOutcome


from optimalportfolios.execution.v1 import solver as optimiser_module


from optimalportfolios.execution.v1.solver import (
    CORRIDOR_RELAXATION,
    DISCRETIONARY_EXECUTED_TRADE,
    EXECUTED_TRADE,
    EXPECTED_CASH_WEIGHT,
    MANDATORY_CORRIDOR,
    MANDATORY_CORRIDOR_EXTRA_TRADE,
    MANDATORY_CORRIDOR_MAX,
    MANDATORY_CORRIDOR_MIN,
    MANDATORY_EXECUTED_TRADE,
    PROPOSED_WEIGHT,
    RELAXED_CORRIDOR_SOLVE,
    _get_sign_directed_rescue_candidates,
    build_selected_execution_constraints,
    solve_feasible_execution_portfolio,
    solve_selected_execution_portfolio,
)


from optimalportfolios.execution.v1.policy import (
    BASE_WEIGHT,
    CURRENT_WEIGHT,
    DESIRED_REWEIGHT,
    EFFECTIVE_MODEL_WEIGHT,
    FEASIBILITY_RESCUE_TRADE,
    MANDATORY_FUNDING,
    MANDATORY_CATEGORY,
    MANDATORY_TRADE,
    POLICY_MAX_WEIGHT,
    POLICY_MIN_WEIGHT,
    RAW_MODEL_WEIGHT,
    REBALANCE_CADENCE_ELIGIBLE,
    REQUESTED_TRADE,
    RULE4_TRADE,
    SCORE_PER_TURNOVER,
    SELECTED_TRADE,
    SETTLEMENT_CASH,
    TRADE_SCORE,
)


ASSETS = pd.Index(["A", "B", "Cash"])


ASSET_CLASSES = ("Liquidity", "Fixed Income", "Equity", "Alternatives")


def _base_constraints(group_max: float = 0.8) -> Constraints:
    """Return full product constraints for the three-asset reference."""
    group = GroupLowerUpperConstraints(
        group_loadings=pd.DataFrame({"Risk Assets": [1.0, 1.0, 0.0]}, index=ASSETS),
        group_min_allocation=pd.Series({"Risk Assets": 0.0}),
        group_max_allocation=pd.Series({"Risk Assets": group_max}),
    )
    return Constraints(
        min_weights=pd.Series(0.0, index=ASSETS),
        max_weights=pd.Series(1.0, index=ASSETS),
        tracking_err_vol_constraint=0.001,
        turnover_constraint=0.001,
        tre_utility_weight=25.0,
        turnover_utility_weight=10.0,
        group_lower_upper_constraints=group,
    )


def _trade_table(
    current: list[float],
    model: list[float],
    selected: list[bool],
    base: list[float] | None = None,
    mandatory: list[bool] | None = None,
) -> pd.DataFrame:
    """Build a resolved trade table independently of target preparation."""
    current_s = pd.Series(current, index=ASSETS)
    model_s = pd.Series(model, index=ASSETS)
    base_s = current_s.copy() if base is None else pd.Series(base, index=ASSETS)
    mandatory_s = pd.Series([False, False, False] if mandatory is None else mandatory, index=ASSETS)
    rule4 = ~mandatory_s & pd.Series([True, True, False], index=ASSETS)
    desired = (model_s - base_s).where(rule4, 0.0)
    return pd.DataFrame(
        {
            CURRENT_WEIGHT: current_s,
            RAW_MODEL_WEIGHT: model_s,
            EFFECTIVE_MODEL_WEIGHT: model_s,
            BASE_WEIGHT: base_s,
            DESIRED_REWEIGHT: desired,
            MANDATORY_FUNDING: (base_s - current_s).where(
                ~pd.Series([False, False, True], index=ASSETS), -(base_s - current_s).iloc[:2].sum()
            ),
            MANDATORY_TRADE: mandatory_s,
            RULE4_TRADE: rule4,
            POLICY_MIN_WEIGHT: 0.0,
            POLICY_MAX_WEIGHT: 1.0,
            REBALANCE_CADENCE_ELIGIBLE: True,
            SETTLEMENT_CASH: [False, False, True],
            SELECTED_TRADE: selected,
            TRADE_SCORE: [3.0, 2.0, 0.0],
            SCORE_PER_TURNOVER: [3.0, 2.0, 0.0],
        },
        index=ASSETS,
    )


def test_minimum_tre_solve_freezes_unselected_and_removes_tre_turnover() -> None:
    """Selected assets reach the model while old TRE/turnover limits are removed."""
    table = _trade_table(
        current=[0.4, 0.2, 0.4],
        model=[0.3, 0.3, 0.4],
        selected=[True, True, False],
    )
    covariance = pd.DataFrame(np.eye(3) * 0.04, index=ASSETS, columns=ASSETS)
    result = solve_selected_execution_portfolio(
        optimiser_config=_PortableOptimiserConfig(solver="CLARABEL"),
        trade_table=table,
        base_constraints=_base_constraints(),
        covariance=covariance,
        context="test execution",
    )
    assert result.accepted and result.compliant
    pd.testing.assert_series_equal(
        result.weights,
        pd.Series([0.3, 0.3, 0.4], index=ASSETS, name="minimum_tracking_error"),
        atol=2e-4,
        check_exact=False,
    )
    solved = result.outcome.constraints
    assert solved.tracking_err_vol_constraint is None
    assert solved.turnover_constraint is None
    assert solved.tre_utility_weight is None
    assert solved.turnover_utility_weight is None
    assert result.trade_table[PROPOSED_WEIGHT].equals(result.weights)
    assert result.trade_table.loc["Cash", EXPECTED_CASH_WEIGHT] == pytest.approx(
        result.weights.loc["Cash"]
    )
    assert not result.trade_table[RELAXED_CORRIDOR_SOLVE].any()
    assert result.trade_table[CORRIDOR_RELAXATION].max() == pytest.approx(0.0)


def test_hard_group_minimum_uses_minimum_selected_corridor_relaxation() -> None:
    """A selected asset may exceed its model target only to restore compliance."""
    table = _trade_table(
        current=[0.890, 0.0, 0.110],
        model=[0.898, 0.0, 0.102],
        selected=[True, False, False],
    )
    group = GroupLowerUpperConstraints(
        group_loadings=pd.DataFrame({"Risk Assets": [1.0, 1.0, 0.0]}, index=ASSETS),
        group_min_allocation=pd.Series({"Risk Assets": 0.9}),
        group_max_allocation=pd.Series({"Risk Assets": 1.0}),
    )
    constraints = Constraints(
        min_weights=pd.Series(0.0, index=ASSETS),
        max_weights=pd.Series(1.0, index=ASSETS),
        group_lower_upper_constraints=group,
    )
    with pytest.raises(ValueError, match="0.8980.*0.9000"):
        build_selected_execution_constraints(
            base_constraints=constraints,
            partition_groups=ASSET_CLASSES,
            trade_table=table,
        )

    covariance = pd.DataFrame(np.eye(3) * 0.04, index=ASSETS, columns=ASSETS)
    result = solve_selected_execution_portfolio(
        optimiser_config=_PortableOptimiserConfig(solver="CLARABEL"),
        trade_table=table,
        base_constraints=constraints,
        partition_groups=ASSET_CLASSES,
        covariance=covariance,
        context="test hard-feasibility corridor relaxation",
        relax_selected_corridors=True,
    )

    assert result.accepted and result.compliant
    assert result.weights.at["A"] == pytest.approx(0.9, abs=2e-4)
    assert result.weights.at["B"] == pytest.approx(0.0, abs=2e-4)
    assert result.weights.at["Cash"] == pytest.approx(0.1, abs=2e-4)
    assert result.trade_table[RELAXED_CORRIDOR_SOLVE].all()
    assert result.trade_table.at["A", CORRIDOR_RELAXATION] == pytest.approx(0.002, abs=2e-4)
    assert result.trade_table.at["B", CORRIDOR_RELAXATION] == pytest.approx(0.0)


def test_hard_group_minimum_cannot_move_mandatory_model_entry() -> None:
    """A mandatory model entry remains pinned even in relaxed-corridor mode."""
    table = _trade_table(
        current=[0.88, 0.008, 0.112],
        model=[0.89, 0.008, 0.102],
        selected=[True, False, False],
        mandatory=[False, True, False],
    )
    table[MANDATORY_CATEGORY] = ["", "model_entry", ""]
    group = GroupLowerUpperConstraints(
        group_loadings=pd.DataFrame({"Risk Assets": [1.0, 1.0, 0.0]}, index=ASSETS),
        group_min_allocation=pd.Series({"Risk Assets": 0.9}),
        group_max_allocation=pd.Series({"Risk Assets": 1.0}),
    )
    constraints = Constraints(
        min_weights=pd.Series(0.0, index=ASSETS),
        max_weights=pd.Series([0.89, 1.0, 1.0], index=ASSETS),
        group_lower_upper_constraints=group,
    )
    covariance = pd.DataFrame(np.eye(3) * 0.04, index=ASSETS, columns=ASSETS)

    with pytest.raises(ValueError, match="0.8980.*0.9000"):
        solve_selected_execution_portfolio(
            optimiser_config=_PortableOptimiserConfig(solver="CLARABEL"),
            trade_table=table,
            base_constraints=constraints,
            partition_groups=ASSET_CLASSES,
            covariance=covariance,
            context="test pinned model-entry relaxation",
            relax_selected_corridors=True,
        )


def test_income_cash_bootstrap_keeps_exact_model_entries() -> None:
    """A 2% cash-bootstrap model allocation cannot satisfy a 1.5% group cap."""
    model = [0.018319, 0.001681, 0.980000]
    table = _trade_table(
        current=[0.0, 0.0, 1.0],
        base=model,
        model=model,
        selected=[False, False, False],
        mandatory=[True, True, True],
    )
    table[MANDATORY_CATEGORY] = ["model_entry", "model_entry", "mandatory_cash_funding"]
    constraints = Constraints(
        min_weights=pd.Series(0.0, index=ASSETS),
        max_weights=pd.Series(1.0, index=ASSETS),
        group_lower_upper_constraints=GroupLowerUpperConstraints(
            group_loadings=pd.DataFrame({"Commodities ex-Precious": [1.0, 1.0, 0.0]}, index=ASSETS),
            group_min_allocation=pd.Series({"Commodities ex-Precious": 0.0}),
            group_max_allocation=pd.Series({"Commodities ex-Precious": 0.015}),
        ),
    )
    covariance = pd.DataFrame(np.eye(3) * 0.04, index=ASSETS, columns=ASSETS)

    with pytest.raises(ValueError, match="0.0200.*0.0150"):
        solve_selected_execution_portfolio(
            optimiser_config=_PortableOptimiserConfig(solver="CLARABEL"),
            trade_table=table,
            base_constraints=constraints,
            partition_groups=ASSET_CLASSES,
            covariance=covariance,
            context="test Income cash bootstrap",
        )


def test_feasible_exact_cash_model_survives_numerical_projection_failure(monkeypatch) -> None:
    """An already feasible, pinned zero-TRE model is the exact bootstrap optimum."""
    model = [0.4, 0.4, 0.2]
    table = _trade_table(
        current=[0, 0, 1], model=model, base=model, selected=[False] * 3, mandatory=[True] * 3
    )
    table[MANDATORY_CATEGORY] = ["model_entry", "model_entry", "mandatory_cash_funding"]
    constraints = _base_constraints(group_max=0.8)
    covariance = pd.DataFrame(
        [[0.04, 0.04, 0], [0.04, 0.04, 0], [0, 0, 0]], index=ASSETS, columns=ASSETS
    )

    def rejected(**kwargs):
        """Simulate a numerical projection failure without changing its inputs."""
        current = kwargs["weights_0"].to_numpy()
        return kwargs["weights_0"], OptimizationOutcome(
            weights=current,
            accepted=False,
            solver="MOSEK",
            status="infeasible",
            context="synthetic numerical failure",
            constraints=kwargs["constraints"],
        )

    monkeypatch.setattr(optimiser_module, "wrapper_minimise_tracking_error", rejected)
    result = solve_selected_execution_portfolio(
        trade_table=table,
        base_constraints=constraints,
        partition_groups=ASSET_CLASSES,
        covariance=covariance,
    )
    assert result.accepted and result.compliant
    np.testing.assert_allclose(result.weights.to_numpy(), model, atol=1e-15, rtol=0)
    assert result.outcome.constraint_residuals
    assert result.trade_table[RELAXED_CORRIDOR_SOLVE].eq(False).all()


def test_cash_model_recovery_does_not_accept_product_group_rounding(monkeypatch) -> None:
    """A numerical fallback must not grandfather even a sub-half-bp group breach."""
    model = [0.4, 0.4, 0.2]
    table = _trade_table(
        current=[0, 0, 1], model=model, base=model, selected=[False] * 3, mandatory=[True] * 3
    )
    table[MANDATORY_CATEGORY] = ["model_entry", "model_entry", "mandatory_cash_funding"]
    constraints = _base_constraints(group_max=0.8 - 1e-6)
    covariance = pd.DataFrame(np.eye(3) * 0.04, index=ASSETS, columns=ASSETS)

    def rejected(**kwargs):
        """Simulate a numerical projection failure without changing its inputs."""
        return kwargs["weights_0"], OptimizationOutcome(
            weights=kwargs["weights_0"].to_numpy(),
            accepted=False,
            solver="MOSEK",
            status="infeasible",
            context="synthetic numerical failure",
            constraints=kwargs["constraints"],
        )

    monkeypatch.setattr(optimiser_module, "wrapper_minimise_tracking_error", rejected)
    result = solve_selected_execution_portfolio(
        trade_table=table,
        base_constraints=constraints,
        partition_groups=ASSET_CLASSES,
        covariance=covariance,
    )
    assert not result.accepted


def test_exact_cash_model_recovery_reconciles_only_numerical_dust(monkeypatch) -> None:
    """Sub-zero-tolerance entries and a tiny budget excess fund back to cash."""
    model = [0.8, 1e-9, 0.2000000126]
    table = _trade_table(
        current=[0, 0, 1],
        model=model,
        base=[0.8, 0, 0.2000000126],
        selected=[False] * 3,
        mandatory=[True] * 3,
    )
    table[MANDATORY_CATEGORY] = ["model_entry", "model_entry", "mandatory_cash_funding"]
    constraints = _base_constraints(group_max=0.8)
    covariance = pd.DataFrame(np.eye(3) * 0.04, index=ASSETS, columns=ASSETS)

    def rejected(**kwargs):
        """Simulate a numerical projection failure without changing its inputs."""
        return kwargs["weights_0"], OptimizationOutcome(
            weights=kwargs["weights_0"].to_numpy(),
            accepted=False,
            solver="MOSEK",
            status="infeasible",
            context="numerical budget/dust",
            constraints=kwargs["constraints"],
        )

    monkeypatch.setattr(optimiser_module, "wrapper_minimise_tracking_error", rejected)
    result = solve_selected_execution_portfolio(
        trade_table=table,
        base_constraints=constraints,
        partition_groups=ASSET_CLASSES,
        covariance=covariance,
    )
    assert result.accepted and result.compliant
    np.testing.assert_allclose(result.weights, [0.8, 0, 0.2], atol=1e-12, rtol=0)
    assert result.weights.sum() == pytest.approx(1, abs=1e-12)


@pytest.mark.parametrize(
    ("category", "current", "base", "model", "group_min", "group_max"),
    [
        (
            "product_min_buy",
            [0.01, 0.04, 0.95],
            [0.03, 0.04, 0.93],
            [0.04, 0.04, 0.92],
            0.08,
            1.0,
        ),
        (
            "product_max_sell",
            [0.08, 0.04, 0.88],
            [0.05, 0.04, 0.91],
            [0.04, 0.04, 0.92],
            0.0,
            0.08,
        ),
    ],
)
def test_one_sided_mandatory_repair_remains_movable_for_group_feasibility(
    category: str,
    current: list[float],
    base: list[float],
    model: list[float],
    group_min: float,
    group_max: float,
) -> None:
    """A mandatory boundary repair is an interval, not a frozen target."""
    table = _trade_table(
        current=current,
        base=base,
        model=model,
        selected=[False, False, False],
        mandatory=[True, False, False],
    )
    table[MANDATORY_CATEGORY] = [category, "", "mandatory_cash_funding"]
    table.loc["A", POLICY_MIN_WEIGHT] = 0.03 if category.endswith("min_buy") else 0.0
    table.loc["A", POLICY_MAX_WEIGHT] = 0.05 if category.endswith("max_sell") else 0.10
    group = GroupLowerUpperConstraints(
        group_loadings=pd.DataFrame({"Risk Assets": [1.0, 1.0, 0.0]}, index=ASSETS),
        group_min_allocation=pd.Series({"Risk Assets": group_min}),
        group_max_allocation=pd.Series({"Risk Assets": group_max}),
    )
    constraints = Constraints(
        min_weights=pd.Series(0.0, index=ASSETS),
        max_weights=pd.Series([0.10, 1.0, 1.0], index=ASSETS),
        group_lower_upper_constraints=group,
    )
    covariance = pd.DataFrame(np.eye(3) * 0.04, index=ASSETS, columns=ASSETS)

    result = solve_selected_execution_portfolio(
        optimiser_config=_PortableOptimiserConfig(solver="CLARABEL"),
        trade_table=table,
        base_constraints=constraints,
        partition_groups=ASSET_CLASSES,
        covariance=covariance,
        context=f"test mandatory corridor {category}",
    )

    assert result.accepted and result.compliant
    assert result.weights.at["A"] == pytest.approx(0.04, abs=2e-4)
    assert result.weights.at["B"] == pytest.approx(0.04, abs=2e-4)
    assert result.weights.at["Cash"] == pytest.approx(0.92, abs=2e-4)
    assert result.trade_table.at["A", MANDATORY_CORRIDOR]
    expected_min = 0.03 if category.endswith("min_buy") else 0.0
    expected_max = 0.10 if category.endswith("min_buy") else 0.05
    assert result.trade_table.at["A", MANDATORY_CORRIDOR_MIN] == pytest.approx(expected_min)
    assert result.trade_table.at["A", MANDATORY_CORRIDOR_MAX] == pytest.approx(expected_max)
    assert result.trade_table.at["A", MANDATORY_CORRIDOR_EXTRA_TRADE] == pytest.approx(
        model[0] - base[0], abs=2e-4
    )
    assert result.trade_table.at["A", MANDATORY_EXECUTED_TRADE] == pytest.approx(
        model[0] - current[0], abs=2e-4
    )
    assert result.trade_table.at["A", DISCRETIONARY_EXECUTED_TRADE] == pytest.approx(0.0)


def test_custom1_style_mandatory_sell_corridor_removes_frozen_group_bridge() -> None:
    """The former EWT 2% workaround is unnecessary with a sell corridor."""
    current = [0.021338, 0.050692, 0.927970]
    base = [0.020000, 0.050692, 0.929308]
    model = [0.011173, 0.050692, 0.938135]
    table = _trade_table(
        current=current,
        base=base,
        model=model,
        selected=[False, False, False],
        mandatory=[True, False, False],
    )
    table[MANDATORY_CATEGORY] = ["instruction_max_sell", "", "mandatory_cash_funding"]
    table.loc["A", POLICY_MAX_WEIGHT] = 0.02
    constraints = Constraints(
        min_weights=pd.Series(0.0, index=ASSETS),
        max_weights=pd.Series([0.02, 1.0, 1.0], index=ASSETS),
        group_lower_upper_constraints=GroupLowerUpperConstraints(
            group_loadings=pd.DataFrame({"Asia ex-Japan": [1.0, 1.0, 0.0]}, index=ASSETS),
            group_min_allocation=pd.Series({"Asia ex-Japan": 0.0}),
            group_max_allocation=pd.Series({"Asia ex-Japan": 0.064566}),
        ),
    )
    covariance = pd.DataFrame(np.eye(3) * 0.04, index=ASSETS, columns=ASSETS)

    result = solve_selected_execution_portfolio(
        optimiser_config=_PortableOptimiserConfig(solver="CLARABEL"),
        trade_table=table,
        base_constraints=constraints,
        partition_groups=ASSET_CLASSES,
        covariance=covariance,
        context="test Custom1 mandatory corridor",
    )

    assert result.accepted and result.compliant
    assert result.weights.at["A"] == pytest.approx(model[0], abs=2e-4)
    assert result.weights.at["A"] < 0.02
    assert result.weights.at["A"] + result.weights.at["B"] <= 0.064566 + 1e-7


def test_selected_corridor_contains_the_frozen_base() -> None:
    """Selection is a relaxation and never removes the frozen zero point."""
    table = _trade_table(
        current=[0.0, 0.6, 0.4],
        model=[0.02, 0.58, 0.4],
        selected=[True, False, False],
    )
    constraints = build_selected_execution_constraints(
        base_constraints=_base_constraints(),
        trade_table=table,
    )
    assert constraints.min_weights.to_dict() == pytest.approx({"A": 0.0, "B": 0.6, "Cash": 0.0})
    assert constraints.max_weights.to_dict() == pytest.approx({"A": 0.02, "B": 0.6, "Cash": 1.0})


@pytest.mark.parametrize(
    ("group_min", "group_max", "expected_min", "expected_max", "direction"),
    [
        (0.300024, 1.0, 0.3, 1.0, "minimum"),
        (0.0, 0.299976, 0.0, 0.3, "maximum"),
    ],
)
def test_group_bound_rounding_snaps_sub_half_bp_frozen_bridge(
    group_min: float,
    group_max: float,
    expected_min: float,
    expected_max: float,
    direction: str,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A frozen group bridge below 0.5 bp is snapped to its executable endpoint."""
    table = _trade_table(
        current=[0.1, 0.2, 0.7],
        model=[0.1, 0.2, 0.7],
        selected=[False, False, False],
    )
    group = GroupLowerUpperConstraints(
        group_loadings=pd.DataFrame({"Risk Assets": [1.0, 1.0, 0.0]}, index=ASSETS),
        group_min_allocation=pd.Series({"Risk Assets": group_min}),
        group_max_allocation=pd.Series({"Risk Assets": group_max}),
    )
    constraints = Constraints(
        min_weights=pd.Series(0.0, index=ASSETS),
        max_weights=pd.Series(1.0, index=ASSETS),
        group_lower_upper_constraints=group,
    )

    with caplog.at_level("INFO", logger=solve_selected_execution_portfolio.__module__):
        rounded = build_selected_execution_constraints(
            base_constraints=constraints,
            partition_groups=ASSET_CLASSES,
            trade_table=table,
        )

    rounded_group = rounded.group_lower_upper_constraints
    assert rounded_group.group_min_allocation["Risk Assets"] == pytest.approx(expected_min)
    assert rounded_group.group_max_allocation["Risk Assets"] == pytest.approx(expected_max)
    assert group.group_min_allocation["Risk Assets"] == pytest.approx(group_min)
    assert group.group_max_allocation["Risk Assets"] == pytest.approx(group_max)
    assert direction in caplog.text
    assert "0.2400 bp" in caplog.text


@pytest.mark.parametrize(
    ("group_min", "group_max"),
    [(0.300051, 1.0), (0.0, 0.299949)],
)
def test_group_bound_rounding_rejects_bridge_above_half_bp(
    group_min: float,
    group_max: float,
) -> None:
    """A group bridge above 0.5 bp remains a hard infeasibility."""
    table = _trade_table(
        current=[0.1, 0.2, 0.7],
        model=[0.1, 0.2, 0.7],
        selected=[False, False, False],
    )
    constraints = Constraints(
        min_weights=pd.Series(0.0, index=ASSETS),
        max_weights=pd.Series(1.0, index=ASSETS),
        group_lower_upper_constraints=GroupLowerUpperConstraints(
            group_loadings=pd.DataFrame({"Risk Assets": [1.0, 1.0, 0.0]}, index=ASSETS),
            group_min_allocation=pd.Series({"Risk Assets": group_min}),
            group_max_allocation=pd.Series({"Risk Assets": group_max}),
        ),
    )

    with pytest.raises(ValueError, match="Group 'Risk Assets'"):
        build_selected_execution_constraints(
            base_constraints=constraints,
            partition_groups=ASSET_CLASSES,
            trade_table=table,
        )


def test_group_bound_rounding_snaps_below_solver_zero_tolerance() -> None:
    """A positive rounding bridge is snapped even below the solver zero tolerance."""
    table = _trade_table(
        current=[0.1, 0.2, 0.7],
        model=[0.1, 0.2, 0.7],
        selected=[False, False, False],
    )
    constraints = Constraints(
        min_weights=pd.Series(0.0, index=ASSETS),
        max_weights=pd.Series(1.0, index=ASSETS),
        group_lower_upper_constraints=GroupLowerUpperConstraints(
            group_loadings=pd.DataFrame({"Risk Assets": [1.0, 1.0, 0.0]}, index=ASSETS),
            group_min_allocation=pd.Series({"Risk Assets": 0.300000005}),
            group_max_allocation=pd.Series({"Risk Assets": 1.0}),
        ),
    )

    rounded = build_selected_execution_constraints(
        base_constraints=constraints,
        partition_groups=ASSET_CLASSES,
        trade_table=table,
    )

    assert rounded.group_lower_upper_constraints.group_min_allocation[
        "Risk Assets"
    ] == pytest.approx(0.3, abs=1e-12)


def test_mandatory_and_reweight_funding_reconcile_to_cash() -> None:
    """Mandatory proceeds and Rule 4 funding reproduce the solved cash weight."""
    table = _trade_table(
        current=[0.4, 0.2, 0.4],
        model=[0.3, 0.3, 0.4],
        base=[0.3, 0.2, 0.5],
        mandatory=[True, False, False],
        selected=[False, True, False],
    )
    covariance = pd.DataFrame(np.eye(3) * 0.04, index=ASSETS, columns=ASSETS)
    result = solve_selected_execution_portfolio(
        optimiser_config=_PortableOptimiserConfig(solver="CLARABEL"),
        trade_table=table,
        base_constraints=_base_constraints(),
        covariance=covariance,
    )
    assert result.accepted and result.compliant
    assert result.weights.at["A"] == pytest.approx(0.3)
    assert result.weights.at["B"] == pytest.approx(0.3, abs=2e-4)
    assert result.weights.at["Cash"] == pytest.approx(0.4, abs=2e-4)
    assert result.trade_table.at["A", EXECUTED_TRADE] == pytest.approx(-0.1)
    assert result.trade_table.at["Cash", EXPECTED_CASH_WEIGHT] == pytest.approx(
        result.weights.at["Cash"]
    )


def test_small_accepted_solver_cash_residual_is_reconciled(monkeypatch) -> None:
    """An accepted rounding residual is funded in cash and re-audited."""
    from dataclasses import replace

    from optimalportfolios import wrapper_minimise_tracking_error
    from optimalportfolios.execution.v1 import solver as optimiser

    table = _trade_table(
        current=[0.4, 0.2, 0.4],
        model=[0.3, 0.3, 0.4],
        selected=[True, True, False],
    )
    covariance = pd.DataFrame(np.eye(3) * 0.04, index=ASSETS, columns=ASSETS)

    def solve(**kwargs):
        """Model a numerically accepted solver result with excess cash."""
        weights, outcome = wrapper_minimise_tracking_error(**kwargs)
        weights = weights.copy()
        weights.at["Cash"] += 1.5e-7
        solver_index = outcome.constraints.min_weights.index
        return weights, replace(
            outcome, weights=weights.reindex(solver_index).to_numpy(dtype=float)
        )

    monkeypatch.setattr(optimiser, "wrapper_minimise_tracking_error", solve)
    result = solve_selected_execution_portfolio(
        table,
        _base_constraints(),
        covariance,
        optimiser_config=_PortableOptimiserConfig(solver="CLARABEL"),
    )

    assert result.accepted and result.compliant
    expected_cash = table[CURRENT_WEIGHT].sum() - result.weights.drop("Cash").sum()
    assert result.weights.at["Cash"] == pytest.approx(expected_cash, abs=1e-12)
    assert result.trade_table.at["Cash", EXPECTED_CASH_WEIGHT] == pytest.approx(
        expected_cash, abs=1e-12
    )
    solver_index = result.outcome.constraints.min_weights.index
    np.testing.assert_allclose(result.outcome.weights, result.weights.loc[solver_index])


def test_subthreshold_base_repairs_are_included_in_cash_funding() -> None:
    """Many individually tiny base repairs still require cash funding in total."""
    table = _trade_table(
        current=[0.4, 0.2, 0.4],
        model=[0.3, 0.3, 0.4],
        selected=[True, True, False],
    )
    tiny_repair = 5e-9
    count = 30
    for number in range(count):
        row = table.loc["B"].copy()
        row[[CURRENT_WEIGHT, RAW_MODEL_WEIGHT, EFFECTIVE_MODEL_WEIGHT]] = 0.0
        row[[BASE_WEIGHT, MANDATORY_FUNDING]] = tiny_repair
        row[[DESIRED_REWEIGHT, TRADE_SCORE, SCORE_PER_TURNOVER]] = 0.0
        row[[MANDATORY_TRADE, RULE4_TRADE, SELECTED_TRADE]] = False
        table.loc[f"Tiny{number}"] = row
    table.at["Cash", BASE_WEIGHT] -= count * tiny_repair
    table.at["Cash", MANDATORY_FUNDING] = -count * tiny_repair
    assets = table.index
    constraints = Constraints(
        min_weights=pd.Series(0.0, index=assets),
        max_weights=pd.Series(1.0, index=assets),
    )
    covariance = pd.DataFrame(np.eye(len(assets)) * 0.04, index=assets, columns=assets)

    result = solve_selected_execution_portfolio(
        table, constraints, covariance, optimiser_config=_PortableOptimiserConfig(solver="CLARABEL")
    )

    assert result.accepted and result.compliant
    assert result.weights.at["Cash"] == pytest.approx(
        result.trade_table.at["Cash", EXPECTED_CASH_WEIGHT], abs=1e-7
    )
    old_mandatory_funding = table.loc[table[MANDATORY_TRADE], MANDATORY_FUNDING].sum()
    noncash = assets != "Cash"
    old_expected = (
        table.at["Cash", CURRENT_WEIGHT]
        - old_mandatory_funding
        - (result.weights.loc[noncash] - table.loc[noncash, BASE_WEIGHT]).sum()
    )
    assert abs(result.weights.at["Cash"] - old_expected) > 1e-7


def test_material_solver_cash_residual_is_not_silently_reconciled(monkeypatch) -> None:
    """A material solver funding mismatch remains a hard execution failure."""
    from dataclasses import replace

    from optimalportfolios import wrapper_minimise_tracking_error
    from optimalportfolios.execution.v1 import solver as optimiser

    table = _trade_table(
        current=[0.4, 0.2, 0.4],
        model=[0.3, 0.3, 0.4],
        selected=[True, True, False],
    )
    covariance = pd.DataFrame(np.eye(3) * 0.04, index=ASSETS, columns=ASSETS)

    def solve(**kwargs):
        """Model an accepted solver result with a material cash error."""
        weights, outcome = wrapper_minimise_tracking_error(**kwargs)
        weights = weights.copy()
        weights.at["Cash"] += 2e-6
        solver_index = outcome.constraints.min_weights.index
        return weights, replace(
            outcome, weights=weights.reindex(solver_index).to_numpy(dtype=float)
        )

    monkeypatch.setattr(optimiser, "wrapper_minimise_tracking_error", solve)
    with pytest.raises(RuntimeError, match="cash rounding adjustment exceeds limit"):
        solve_selected_execution_portfolio(
            table,
            _base_constraints(),
            covariance,
            optimiser_config=_PortableOptimiserConfig(solver="CLARABEL"),
        )


def test_feasibility_expansion_adds_smallest_ranked_rule4_prefix() -> None:
    """Hard group feasibility adds only the first required unselected trade."""
    table = _trade_table(
        current=[0.6, 0.2, 0.2],
        model=[0.3, 0.2, 0.5],
        selected=[False, False, False],
    )
    covariance = pd.DataFrame(np.eye(3) * 0.04, index=ASSETS, columns=ASSETS)
    result = solve_feasible_execution_portfolio(
        optimiser_config=_PortableOptimiserConfig(solver="CLARABEL"),
        trade_table=table,
        base_constraints=_base_constraints(group_max=0.5),
        covariance=covariance,
        expand_for_feasibility=True,
        allow_corridor_relaxation=True,
        context="test feasibility expansion",
    )
    assert result.accepted and result.compliant
    assert result.trade_table[REQUESTED_TRADE].tolist() == [False, False, False]
    assert result.trade_table[FEASIBILITY_RESCUE_TRADE].tolist() == [True, False, False]
    assert result.trade_table[SELECTED_TRADE].tolist() == [True, False, False]
    assert not result.trade_table[RELAXED_CORRIDOR_SOLVE].any()


def test_sign_directed_rescue_keeps_group_restoring_trade_only() -> None:
    """A group-max breach admits sells but excludes group-increasing buys."""
    table = _trade_table(
        current=[0.6, 0.2, 0.2],
        model=[0.4, 0.4, 0.2],
        selected=[False, False, False],
    )
    eligible = pd.Series([True, True, False], index=ASSETS)

    directed = _get_sign_directed_rescue_candidates(
        trade_table=table,
        base_constraints=_base_constraints(group_max=0.7),
        rescue_eligible=eligible,
    )

    assert directed.to_dict() == {"A": True, "B": False, "Cash": False}


@pytest.mark.parametrize("factorize", [True, False])
@pytest.mark.parametrize("max_vol", [None, 0.2498])
def test_dust_cleanup_keeps_filtered_solver_order(factorize, max_vol) -> None:
    """Missing-history assets do not break dust cleanup or corrupt its audit."""
    from optimalportfolios import OptimiserConfig, evaluate_constraint_residuals
    from optimalportfolios.execution.v1.policy import (
        CUTOFF_SELL_DOWN,
        CUTOFF_SELL_DOWN_DUST_ROUNDED,
        SELL_DOWN_DUST_THRESHOLD,
    )

    table = _trade_table(
        current=[0.5, 0.001, 0.499],
        model=[0.5, 0.001, 0.499],
        selected=[False, True, False],
    )
    table.loc["Missing"] = table.loc["B"]
    table.loc[
        "Missing",
        [CURRENT_WEIGHT, RAW_MODEL_WEIGHT, EFFECTIVE_MODEL_WEIGHT, BASE_WEIGHT, DESIRED_REWEIGHT],
    ] = 0.0
    table.loc["Missing", SELECTED_TRADE] = False
    table[CUTOFF_SELL_DOWN] = table.index == "B"
    table[SELL_DOWN_DUST_THRESHOLD] = 0.0025
    constraints = Constraints(
        min_weights=pd.Series(0.0, index=table.index),
        max_weights=pd.Series(1.0, index=table.index),
        max_target_portfolio_vol_an=max_vol,
    )
    order = pd.Index(["Cash", "Missing", "B", "A"])
    # NaN denotes missing history; zero variance may be retained by older OP releases.
    covariance = pd.DataFrame(np.diag([0.25, np.nan, 0.0001, 0.0001]), index=order, columns=order)
    result = solve_selected_execution_portfolio(
        trade_table=table,
        base_constraints=constraints,
        covariance=covariance,
        optimiser_config=OptimiserConfig(solver="CLARABEL", factorize_covar=factorize),
    )

    assert result.accepted and result.compliant
    assert bool(result.trade_table.at["B", CUTOFF_SELL_DOWN_DUST_ROUNDED]) == (max_vol is None)
    assert result.weights["Missing"] == 0.0
    if max_vol is None:
        assert result.weights["B"] == 0.0
        assert result.weights["Cash"] == pytest.approx(0.5, abs=1e-7)
    else:
        assert result.weights["B"] > 0.0005
    aligned = result.outcome.constraints.min_weights.index
    assert aligned.tolist() == ["Cash", "B", "A"]
    np.testing.assert_allclose(result.outcome.weights, result.weights.loc[aligned])
    residuals = evaluate_constraint_residuals(
        result.outcome.weights,
        result.outcome.constraints,
        covar=covariance.loc[aligned, aligned].to_numpy(),
        covar_factorization=result.outcome.covar_factorization,
    )
    assert all(r.passed for r in residuals if r.hard)


def test_dust_cleanup_respects_original_unfiltered_group_floor(monkeypatch) -> None:
    """A solver's reduced constraints cannot erase the execution group floor."""
    from optimalportfolios import wrapper_minimise_tracking_error
    from optimalportfolios.execution.v1 import solver as optimiser
    from optimalportfolios.execution.v1.policy import (
        CUTOFF_SELL_DOWN,
        CUTOFF_SELL_DOWN_DUST_ROUNDED,
        SELL_DOWN_DUST_THRESHOLD,
    )
    from dataclasses import replace

    table = _trade_table(
        current=[0.5, 0.001, 0.499], model=[0.5, 0.001, 0.499], selected=[False, True, False]
    )
    table[CUTOFF_SELL_DOWN] = [False, True, False]
    table[SELL_DOWN_DUST_THRESHOLD] = 0.0025
    # Signed loading makes the simple positive-loading slack shortcut insufficient:
    # transferring B to cash reduces this exposure by twice the transferred weight.
    group = GroupLowerUpperConstraints(
        group_loadings=pd.DataFrame({"Signed": [0.0, 1.0, -1.0]}, index=ASSETS),
        group_min_allocation=pd.Series({"Signed": -0.4995}),
        group_max_allocation=None,
    )
    constraints = Constraints(
        min_weights=pd.Series(0.0, index=ASSETS),
        max_weights=pd.Series(1.0, index=ASSETS),
        group_lower_upper_constraints=group,
    )
    covariance = pd.DataFrame(np.eye(3) * 0.04, index=ASSETS, columns=ASSETS)

    def solve(**kwargs):
        """Model a reduced solver constraint set, preserving its accepted weights."""
        weights, outcome = wrapper_minimise_tracking_error(**kwargs)
        return weights, replace(
            outcome, constraints=outcome.constraints.copy(group_lower_upper_constraints=None)
        )

    monkeypatch.setattr(optimiser, "wrapper_minimise_tracking_error", solve)
    result = solve_selected_execution_portfolio(
        table, constraints, covariance, optimiser_config=_PortableOptimiserConfig(solver="CLARABEL")
    )
    assert result.accepted and result.compliant
    assert not result.trade_table.at["B", CUTOFF_SELL_DOWN_DUST_ROUNDED]
    assert result.weights["B"] > 0.0005


@pytest.mark.parametrize("diagnose", [True, False])
def test_exhausted_rescue_retains_final_attempt_and_diagnoses_once(monkeypatch, diagnose):
    """Terminal diagnostics describe the final expanded problem, never each retry."""
    from optimalportfolios import OptimizationOutcome, OptimiserConfig
    from optimalportfolios.execution.v1 import solver as optimiser

    table = _trade_table(
        current=[0.4, 0.2, 0.4], model=[0.3, 0.3, 0.4], selected=[True, False, False]
    )
    constraints = _base_constraints()
    covariance = pd.DataFrame(np.eye(3) * 0.04, index=ASSETS, columns=ASSETS)
    calls = []
    diagnoses = []

    def reject(**kwargs):
        """Reject initial and expanded selections with distinguishable reasons."""
        calls.append(kwargs["trade_table"].copy())
        outcome = OptimizationOutcome(
            weights=table[CURRENT_WEIGHT].to_numpy(),
            accepted=False,
            solver="CLARABEL",
            status="infeasible" if len(calls) == 1 else "solver_error",
            context=kwargs["context"],
            reason=f"attempt {len(calls)}",
            constraints=constraints,
        )
        return optimiser.ExecutionOptimizationResult(
            weights=table[CURRENT_WEIGHT], trade_table=kwargs["trade_table"], outcome=outcome
        )

    def diagnose_once(**kwargs):
        """Return a diagnostic suggestion without changing any hard bounds."""
        diagnoses.append(kwargs)
        return {"group_min:Group2": 0.001}

    monkeypatch.setattr(optimiser, "solve_selected_execution_portfolio", reject)
    monkeypatch.setattr(optimiser, "diagnose_infeasibility", diagnose_once, raising=False)
    result = solve_feasible_execution_portfolio(
        table,
        constraints,
        covariance,
        expand_for_feasibility=True,
        optimiser_config=OptimiserConfig(solver="CLARABEL", diagnose_infeasibility=diagnose),
    )
    assert not result.accepted
    assert len(calls) == 2
    assert result.outcome.status == "solver_error"
    assert result.trade_table.at["B", FEASIBILITY_RESCUE_TRADE]
    assert "attempt 2" in result.outcome.reason
    assert len(diagnoses) == int(diagnose)
    if diagnose:
        assert "group_min:Group2" in result.outcome.reason
        assert "not applied" in result.outcome.reason
    assert constraints.group_lower_upper_constraints.group_min_allocation.iloc[0] == 0.0


@pytest.mark.parametrize("max_risk", [0.945, 0.95])
def test_liquidity_group_capacity_includes_full_investment(max_risk):
    """Liquidity includes a cash-like holding as well as settlement cash."""
    import cvxpy as cvx
    from optimalportfolios.execution.v1.solver import (
        ExecutionSolverInfeasibility,
        _build_selected_execution_bound_series,
        _compute_selected_execution_bridges,
    )

    table = _trade_table(
        current=[0.945, 0.03, 0.025],
        model=[max_risk, 0.03, 0.97 - max_risk],
        selected=[True, False, False],
    )
    group = GroupLowerUpperConstraints(
        group_loadings=pd.DataFrame({"Liquidity": [0.0, 1.0, 1.0]}, index=ASSETS),
        group_min_allocation=pd.Series({"Liquidity": 0.0}),
        group_max_allocation=pd.Series({"Liquidity": 0.05}),
    )
    constraints = Constraints(
        min_weights=pd.Series(0.0, index=ASSETS),
        max_weights=pd.Series(1.0, index=ASSETS),
        group_lower_upper_constraints=group,
    )
    bridges = _compute_selected_execution_bridges(
        constraints, table, partition_groups=ASSET_CLASSES
    )
    assert bridges.at["Liquidity", "e_min"] == pytest.approx(1.0 - max_risk)
    assert bridges.at["Liquidity", "bridge_max"] == pytest.approx(max(0.95 - max_risk, 0))
    # Independently compile the original constraints into a feasibility LP.
    lower, upper = _build_selected_execution_bound_series(constraints, table, False)
    compiled = constraints.copy(min_weights=lower, max_weights=upper)
    w = cvx.Variable(3)
    problem = cvx.Problem(cvx.Minimize(0), compiled.set_cvx_all_constraints(w=w))
    problem.solve(solver="CLARABEL")
    if max_risk < 0.95:
        assert problem.status == "infeasible"
        with pytest.raises(ExecutionSolverInfeasibility, match="Liquidity"):
            build_selected_execution_constraints(constraints, table, partition_groups=ASSET_CLASSES)
    else:
        assert problem.status == "optimal"
        build_selected_execution_constraints(constraints, table, partition_groups=ASSET_CLASSES)


def test_liquidity_bridge_admits_cash_funded_risk_buy():
    """Buying outside Liquidity repairs its cap through settlement cash funding."""
    table = _trade_table(
        current=[0.945, 0.03, 0.025], model=[0.96, 0.03, 0.01], selected=[False, False, False]
    )
    group = GroupLowerUpperConstraints(
        group_loadings=pd.DataFrame({"Liquidity": [0.0, 1.0, 1.0]}, index=ASSETS),
        group_min_allocation=pd.Series({"Liquidity": 0.0}),
        group_max_allocation=pd.Series({"Liquidity": 0.05}),
    )
    constraints = Constraints(
        min_weights=pd.Series(0.0, index=ASSETS),
        max_weights=pd.Series(1.0, index=ASSETS),
        group_lower_upper_constraints=group,
    )
    covariance = pd.DataFrame(np.eye(3) * 0.04, index=ASSETS, columns=ASSETS)
    result = solve_feasible_execution_portfolio(
        table,
        constraints,
        covariance,
        optimiser_config=_PortableOptimiserConfig(solver="CLARABEL"),
        expand_for_feasibility=True,
    )
    assert result.accepted and result.compliant
    assert result.trade_table.at["A", FEASIBILITY_RESCUE_TRADE]
    assert result.weights[["B", "Cash"]].sum() <= 0.05 + 1e-7


def test_partition_caps_tighten_liquidity_capacity():
    """An Alternatives cap reduces the joint capacity outside Liquidity."""
    from optimalportfolios.execution.v1.solver import (
        ExecutionSolverInfeasibility,
        _compute_selected_execution_bridges,
    )

    table = _trade_table(
        current=[0.59, 0.14, 0.05], model=[0.6, 0.14, 0.06], selected=[True, False, False]
    )
    table.loc["Alt"] = table.loc["B"]
    table.loc["Alt", [CURRENT_WEIGHT, BASE_WEIGHT]] = 0.22
    table.loc["Alt", [RAW_MODEL_WEIGHT, EFFECTIVE_MODEL_WEIGHT]] = 0.20
    table.loc["Alt", DESIRED_REWEIGHT] = -0.02
    table.loc["Alt", SELECTED_TRADE] = True
    group = GroupLowerUpperConstraints(
        group_loadings=pd.DataFrame(
            {
                "Equity": [1.0, 0.0, 0.0, 0.0],
                "Fixed Income": [0.0, 1.0, 0.0, 0.0],
                "Liquidity": [0.0, 0.0, 1.0, 0.0],
                "Alternatives": [0.0, 0.0, 0.0, 1.0],
            },
            index=table.index,
        ),
        group_min_allocation=pd.Series(
            {"Equity": 0.0, "Fixed Income": 0.0, "Liquidity": 0.0, "Alternatives": 0.0}
        ),
        group_max_allocation=pd.Series(
            {"Equity": 1.0, "Fixed Income": 1.0, "Liquidity": 0.05, "Alternatives": 0.20}
        ),
    )
    constraints = Constraints(
        min_weights=pd.Series(0.0, index=table.index),
        max_weights=pd.Series(1.0, index=table.index),
        group_lower_upper_constraints=group,
    )
    bridges = _compute_selected_execution_bridges(
        constraints, table, partition_groups=ASSET_CLASSES
    )
    assert bridges.at["Liquidity", "e_min"] == pytest.approx(0.06)
    assert bridges.at["Liquidity", "bridge_max"] == pytest.approx(0.01)
    with pytest.raises(ExecutionSolverInfeasibility, match="Liquidity"):
        build_selected_execution_constraints(constraints, table, partition_groups=ASSET_CLASSES)


@pytest.mark.parametrize("group_name", ["Equity", "Liquidity"])
def test_budget_bridge_rescue_admits_zero_exposure_funding_counterpart(group_name):
    """A buy outside Equity can fund its selected sale when cash is capped."""
    table = _trade_table(
        current=[0.65, 0.30, 0.05], model=[0.60, 0.35, 0.05], selected=[True, False, False]
    )
    group = GroupLowerUpperConstraints(
        group_loadings=pd.DataFrame(
            {group_name: ([1.0, 0.0, 0.0] if group_name == "Equity" else [0.0, 1.0, 1.0])},
            index=ASSETS,
        ),
        group_min_allocation=pd.Series({group_name: 0.0 if group_name == "Equity" else 0.40}),
        group_max_allocation=pd.Series({group_name: 0.60 if group_name == "Equity" else 1.0}),
    )
    constraints = Constraints(
        min_weights=pd.Series(0.0, index=ASSETS),
        max_weights=pd.Series([1.0, 1.0, 0.05], index=ASSETS),
        group_lower_upper_constraints=group,
    )
    covariance = pd.DataFrame(np.eye(3) * 0.04, index=ASSETS, columns=ASSETS)
    result = solve_feasible_execution_portfolio(
        table,
        constraints,
        covariance,
        optimiser_config=_PortableOptimiserConfig(solver="CLARABEL"),
        expand_for_feasibility=True,
    )
    assert result.accepted and result.compliant
    assert result.trade_table.at["B", FEASIBILITY_RESCUE_TRADE]
    np.testing.assert_allclose(result.weights.loc[ASSETS], [0.60, 0.35, 0.05], atol=1e-7)
