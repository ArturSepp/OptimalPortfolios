"""Sell-down, fallback and constraint-directed rescue regressions."""

import numpy as np


import pandas as pd


import pytest


from optimalportfolios import (
    Constraints,
    OptimiserConfig,
    GroupLowerUpperConstraints,
)


from optimalportfolios.execution.v1.feasibility import compute_group_bound_bridges


from optimalportfolios.execution.v1.solver import (
    EXECUTED_TRADE,
    ExecutionSolverInfeasibility,
    RESCUE_REASON,
    solve_feasible_execution_portfolio,
)


from optimalportfolios.execution.v1.policy import (
    BASE_WEIGHT,
    CUTOFF_SELL_DOWN,
    CUTOFF_SELL_DOWN_RETAINED_FRACTION,
    CUTOFF_SELL_DOWN_RETENTION,
    CURRENT_WEIGHT,
    DESIRED_REWEIGHT,
    EFFECTIVE_MODEL_WEIGHT,
    FEASIBILITY_RESCUE_TRADE,
    LOW_WEIGHT_EXCLUDED,
    MANDATORY_FUNDING,
    MANDATORY_TRADE,
    POLICY_MAX_WEIGHT,
    POLICY_MIN_WEIGHT,
    RAW_MODEL_WEIGHT,
    REBALANCE_CADENCE_ELIGIBLE,
    RULE4_TRADE,
    SCORE_PER_TURNOVER,
    SELECTED_TRADE,
    SETTLEMENT_CASH,
    TRADE_SCORE,
    PINNED_INSTRUCTION,
    ExecutionPolicyConfig,
    MandatoryTrades,
    build_execution_target,
    score_execution_trades,
)


def test_compute_group_bound_bridges_handles_signed_loadings() -> None:
    """Interval arithmetic chooses bounds from each loading's sign."""
    assets = pd.Index(["A", "B", "C", "D"])
    lower = pd.Series([0.1, 0.2, 0.0, 0.0], index=assets)
    upper = pd.Series([0.3, 0.4, 0.5, 0.2], index=assets)
    loadings = pd.DataFrame(
        {
            "Positive": [1.0, 1.0, 0.0, 0.0],
            "Signed": [1.0, -2.0, 0.5, 0.0],
        },
        index=assets,
    )
    bridges = compute_group_bound_bridges(
        lower_bounds=lower,
        upper_bounds=upper,
        group_loadings=loadings,
        group_min=pd.Series({"Positive": 0.8, "Signed": 0.2}),
        group_max=pd.Series({"Positive": 0.25, "Signed": -0.8}),
    )

    assert bridges.loc["Positive", ["e_min", "e_max"]].tolist() == pytest.approx([0.3, 0.7])
    assert bridges.loc["Signed", ["e_min", "e_max"]].tolist() == pytest.approx([-0.7, 0.15])
    assert bridges.at["Positive", "bridge_min"] == pytest.approx(0.1)
    assert bridges.at["Positive", "bridge_max"] == pytest.approx(0.05)
    assert bridges.at["Signed", "bridge_min"] == pytest.approx(0.05)
    assert bridges.at["Signed", "bridge_max"] == pytest.approx(0.1)


def _score_target(
    target: pd.DataFrame,
    model: pd.Series,
    asset_classes: pd.Series,
    config: ExecutionPolicyConfig,
) -> pd.DataFrame:
    """Add deterministic scores to a synthetic execution target."""
    covariance = pd.DataFrame(np.eye(len(model)) * 0.04, index=model.index, columns=model.index)
    return score_execution_trades(
        target=target,
        alphas=pd.Series(0.0, index=model.index),
        covariance=covariance,
        asset_classes=asset_classes,
        config=config,
    )


def test_redemption_notice_pin_survives_cutoff_and_counts_for_group_capacity() -> None:
    """A Current Min=Max=current alternative is held with no trade or ticket."""
    assets = pd.Index(["Redeeming Alternative", "Equity", "Cash"])
    current = pd.Series([0.06, 0.84, 0.10], index=assets)
    model = pd.Series([0.004, 0.896, 0.10], index=assets)
    classes = pd.Series(["Alternatives", "Equity", "Liquidity"], index=assets)
    loadings = pd.DataFrame({"Alternatives": [1.0, 0.0, 0.0]}, index=assets)
    group = GroupLowerUpperConstraints(
        group_loadings=loadings,
        group_min_allocation=pd.Series({"Alternatives": 0.05}),
        group_max_allocation=pd.Series({"Alternatives": 0.20}),
    )
    constraints = Constraints(
        min_weights=pd.Series(0.0, index=assets),
        max_weights=pd.Series(1.0, index=assets),
        group_lower_upper_constraints=group,
    )
    covariance = pd.DataFrame(np.eye(3) * 0.04, index=assets, columns=assets)
    config = ExecutionPolicyConfig(
        tre_weight_by_asset_class={
            k: 1.0 for k in ("Liquidity", "Fixed Income", "Equity", "Alternatives")
        },
        max_trades=1,
    )
    target = build_execution_target(
        current_weights=current,
        model_weights=model,
        previous_model_weights=model,
        config=config,
        settlement_cash_asset="Cash",
        required_min_weights=constraints.min_weights,
        required_max_weights=constraints.max_weights,
        mandatory_trades=MandatoryTrades(hard_target={"Redeeming Alternative": 0.06}),
        instruction_audit=pd.DataFrame(
            {
                "Instruction": ["pin"],
                "Comment": ["Current Min=0.06; Current Max=0.06"],
                "Source": ["current_min_max"],
                "allow_current_pin": [True],
                "model_max_repair": [True],
            },
            index=["Redeeming Alternative"],
        ),
        asset_classes=classes,
        group_loadings=loadings,
        group_min=group.group_min_allocation,
        group_max=group.group_max_allocation,
    )
    table = _score_target(target, model, classes, config)

    pinned = "Redeeming Alternative"
    assert table.at[pinned, PINNED_INSTRUCTION]
    assert table.at[pinned, POLICY_MIN_WEIGHT] == pytest.approx(current[pinned])
    assert table.at[pinned, POLICY_MAX_WEIGHT] == pytest.approx(current[pinned])
    assert not table.at[pinned, MANDATORY_TRADE]
    assert not table.at[pinned, RULE4_TRADE]
    assert not table.at[pinned, LOW_WEIGHT_EXCLUDED]
    assert not table.at[pinned, CUTOFF_SELL_DOWN]

    unpinned = build_execution_target(
        current_weights=current,
        model_weights=model,
        previous_model_weights=model,
        config=config,
        settlement_cash_asset="Cash",
        required_min_weights=constraints.min_weights,
        required_max_weights=constraints.max_weights,
        asset_classes=classes,
    )
    assert unpinned.at[pinned, LOW_WEIGHT_EXCLUDED]
    assert unpinned.at[pinned, MANDATORY_TRADE]

    result = solve_feasible_execution_portfolio(
        trade_table=table,
        base_constraints=constraints,
        covariance=covariance,
        optimiser_config=OptimiserConfig(solver="CLARABEL"),
    )
    assert result.accepted and result.compliant
    assert result.weights[pinned] == pytest.approx(current[pinned], abs=2e-5)
    assert abs(result.trade_table.at[pinned, EXECUTED_TRADE]) <= 1e-8
    group_exposure = float((loadings["Alternatives"] * result.weights).sum())
    assert group_exposure == pytest.approx(0.06, abs=2e-5)


def test_sell_down_corridor_recovers_cash_that_frozen_waiver_loses() -> None:
    """The convex corridor retains the FI minimum and sells the cash surplus."""
    assets = pd.Index(["Held FI", "Equity", "New Asset", "Cash"])
    current = pd.Series([0.15, 0.70, 0.00, 0.15], index=assets)
    model = pd.Series([0.004, 0.746, 0.10, 0.15], index=assets)
    previous_model = pd.Series([0.004, 0.846, 0.00, 0.15], index=assets)
    asset_classes = pd.Series(["Fixed Income", "Equity", "Equity", "Liquidity"], index=assets)
    loadings = pd.DataFrame({"Fixed Income": [1.0, 0.0, 0.0, 0.0]}, index=assets)
    group = GroupLowerUpperConstraints(
        group_loadings=loadings,
        group_min_allocation=pd.Series({"Fixed Income": 0.10}),
        group_max_allocation=pd.Series({"Fixed Income": 1.0}),
    )
    constraints = Constraints(
        min_weights=pd.Series({"Held FI": 0.0, "Equity": 0.0, "New Asset": 0.0, "Cash": 0.10}),
        max_weights=pd.Series(1.0, index=assets),
        group_lower_upper_constraints=group,
    )
    covariance = pd.DataFrame(np.eye(4) * 0.04, index=assets, columns=assets)
    config = ExecutionPolicyConfig(
        tre_weight_by_asset_class={
            k: 1.0 for k in ("Liquidity", "Fixed Income", "Equity", "Alternatives")
        },
        small_trade_cutoff=0.005,
        small_trade_cutoff_by_asset_class=None,
    )
    target = build_execution_target(
        current_weights=current,
        model_weights=model,
        previous_model_weights=previous_model,
        config=config,
        settlement_cash_asset="Cash",
        asset_classes=asset_classes,
        required_min_weights=constraints.min_weights,
        required_max_weights=constraints.max_weights,
        group_loadings=loadings,
        group_min=group.group_min_allocation,
        group_max=group.group_max_allocation,
    )
    table = _score_target(target, model, asset_classes, config)

    assert table.at["Held FI", CUTOFF_SELL_DOWN]
    assert table.at["Held FI", BASE_WEIGHT] == pytest.approx(0.15)
    assert not table.at["Held FI", LOW_WEIGHT_EXCLUDED]
    assert not table.at["Held FI", MANDATORY_TRADE]
    result = solve_feasible_execution_portfolio(
        trade_table=table,
        base_constraints=constraints,
        covariance=covariance,
        optimiser_config=OptimiserConfig(solver="CLARABEL"),
    )

    assert result.accepted and result.compliant
    assert result.weights["Held FI"] == pytest.approx(0.10, abs=2e-5)
    assert result.weights["Cash"] == pytest.approx(0.10, abs=2e-5)
    assert result.trade_table.at["Held FI", CUTOFF_SELL_DOWN_RETENTION] == pytest.approx(
        0.10, abs=2e-5
    )
    assert result.trade_table.at["Held FI", CUTOFF_SELL_DOWN_RETAINED_FRACTION] == pytest.approx(
        2.0 / 3.0, abs=2e-4
    )

    frozen = table.copy()
    frozen[CUTOFF_SELL_DOWN] = False
    with pytest.raises(ExecutionSolverInfeasibility, match="Group 'cash'"):
        solve_feasible_execution_portfolio(
            trade_table=frozen,
            base_constraints=constraints,
            covariance=covariance,
            optimiser_config=OptimiserConfig(solver="CLARABEL"),
        )


def test_fi_zero_cutoff_can_buy_small_position_needed_by_group_minimum() -> None:
    """Uniform cutoff proves infeasible; the FI override admits the Rule-4 buy."""
    assets = pd.Index(["Small FI", "Equity", "Cash"])
    current = pd.Series([0.0, 0.95, 0.05], index=assets)
    model = pd.Series([0.004, 0.946, 0.05], index=assets)
    classes = pd.Series(["Fixed Income", "Equity", "Liquidity"], index=assets)
    loadings = pd.DataFrame({"Fixed Income": [1.0, 0.0, 0.0]}, index=assets)
    group = GroupLowerUpperConstraints(
        group_loadings=loadings,
        group_min_allocation=pd.Series({"Fixed Income": 0.004}),
        group_max_allocation=pd.Series({"Fixed Income": 1.0}),
    )
    constraints = Constraints(
        min_weights=pd.Series(0.0, index=assets),
        max_weights=pd.Series(1.0, index=assets),
        group_lower_upper_constraints=group,
    )
    covariance = pd.DataFrame(np.eye(3) * 0.04, index=assets, columns=assets)

    def build(config: ExecutionPolicyConfig) -> pd.DataFrame:
        """Prepare the cutoff scenario with an explicit class-level policy."""
        target = build_execution_target(
            current_weights=current,
            model_weights=model,
            previous_model_weights=model,
            config=config,
            settlement_cash_asset="Cash",
            asset_classes=classes,
            group_loadings=loadings,
            group_min=group.group_min_allocation,
            group_max=group.group_max_allocation,
        )
        return _score_target(target, model, classes, config)

    uniform = ExecutionPolicyConfig(
        tre_weight_by_asset_class={
            k: 1.0 for k in ("Liquidity", "Fixed Income", "Equity", "Alternatives")
        },
        small_trade_cutoff=0.005,
        small_trade_cutoff_by_asset_class=None,
    )
    with pytest.raises(ExecutionSolverInfeasibility):
        solve_feasible_execution_portfolio(
            trade_table=build(uniform),
            base_constraints=constraints,
            covariance=covariance,
            optimiser_config=OptimiserConfig(solver="CLARABEL"),
        )

    fi0 = ExecutionPolicyConfig(
        tre_weight_by_asset_class={
            k: 1.0 for k in ("Liquidity", "Fixed Income", "Equity", "Alternatives")
        },
        small_trade_cutoff=0.005,
        small_trade_cutoff_by_asset_class={"Fixed Income": 0.0},
    )
    result = solve_feasible_execution_portfolio(
        trade_table=build(fi0),
        base_constraints=constraints,
        covariance=covariance,
        optimiser_config=OptimiserConfig(solver="CLARABEL"),
    )
    assert result.accepted and result.compliant
    assert result.weights["Small FI"] == pytest.approx(0.004, abs=2e-5)


def test_constraint_directed_rescue_ignores_score_filter_and_wrong_directions() -> None:
    """A negative-score group-restoring trade is admitted without higher-score noise."""
    assets = pd.Index(["Required Sell", "Wrong Buy", "Irrelevant", "Cash"])
    current = pd.Series([0.6, 0.1, 0.1, 0.2], index=assets)
    model = pd.Series([0.4, 0.3, 0.2, 0.1], index=assets)
    cash = pd.Series([False, False, False, True], index=assets)
    rule4 = ~cash
    desired = (model - current).where(rule4, 0.0)
    table = pd.DataFrame(
        {
            CURRENT_WEIGHT: current,
            RAW_MODEL_WEIGHT: model,
            EFFECTIVE_MODEL_WEIGHT: model,
            BASE_WEIGHT: current,
            DESIRED_REWEIGHT: desired,
            MANDATORY_FUNDING: 0.0,
            MANDATORY_TRADE: False,
            RULE4_TRADE: rule4,
            POLICY_MIN_WEIGHT: 0.0,
            POLICY_MAX_WEIGHT: 1.0,
            REBALANCE_CADENCE_ELIGIBLE: True,
            SETTLEMENT_CASH: cash,
            SELECTED_TRADE: False,
            TRADE_SCORE: [-1.0, 9.0, 10.0, 0.0],
            SCORE_PER_TURNOVER: [-5.0, 45.0, 100.0, 0.0],
        },
        index=assets,
    )
    group = GroupLowerUpperConstraints(
        group_loadings=pd.DataFrame({"Risk Assets": [1.0, 1.0, 0.0, 0.0]}, index=assets),
        group_min_allocation=pd.Series({"Risk Assets": 0.0}),
        group_max_allocation=pd.Series({"Risk Assets": 0.5}),
    )
    constraints = Constraints(
        min_weights=pd.Series(0.0, index=assets),
        max_weights=pd.Series(1.0, index=assets),
        group_lower_upper_constraints=group,
    )
    covariance = pd.DataFrame(np.eye(4) * 0.04, index=assets, columns=assets)

    result = solve_feasible_execution_portfolio(
        trade_table=table,
        base_constraints=constraints,
        covariance=covariance,
        optimiser_config=OptimiserConfig(solver="CLARABEL"),
        expand_for_feasibility=True,
        context="constraint-directed rescue test",
    )

    assert result.accepted and result.compliant
    assert result.trade_table.loc[
        ["Required Sell", "Wrong Buy", "Irrelevant"], FEASIBILITY_RESCUE_TRADE
    ].tolist() == [True, False, False]
    assert result.trade_table.at["Required Sell", RESCUE_REASON] == "Risk Assets:max"
