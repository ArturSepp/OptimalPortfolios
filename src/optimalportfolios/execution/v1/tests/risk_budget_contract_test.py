"""Target-band risk-budget-ranked execution tests."""

from dataclasses import replace


import numpy as np


import pandas as pd


import pytest


from optimalportfolios import (
    OptimiserConfig,
    ConstraintEnforcementType,
    Constraints,
    GroupTrackingErrorConstraint,
    GroupTurnoverConstraint,
)


from optimalportfolios.execution.v1 import RiskBudgetRankedExecutionPolicyConfig


from optimalportfolios.execution.v1.solver import (
    MANDATORY_CORRIDOR,
    build_selected_execution_constraints,
    solve_feasible_execution_portfolio,
)


from optimalportfolios.execution.v1.policy import (
    ABS_DISTANCE_TO_BAND,
    ALPHA_MISSING,
    ASSET_CLASS_RANK,
    BASE_WEIGHT,
    FEASIBILITY_RESCUE_TRADE,
    GROUP_K_RISK,
    GROUP_SELECTION_STATUS,
    GROUP_TRE_BUDGET,
    GROUP_TRE_WEIGHT_NORMALIZED,
    MANDATORY_CATEGORY,
    MANDATORY_TRADE,
    MATERIAL_TRADE,
    MODEL_ENTRY,
    MODEL_EXIT,
    RAW_MODEL_WEIGHT,
    RANKING_ELIGIBLE,
    RISK_BUDGET_SELECTED,
    RULE4_TRADE,
    SELECTED_TRADE,
    TARGET_BAND_LOWER,
    TARGET_BAND_UPPER,
    TARGET_BAND_WIDTH,
    TRADE_RANK,
    build_execution_target,
)


from optimalportfolios.execution.v1.risk_budget_ranker import (
    build_risk_budget_group_summary,
    build_risk_budget_prefix_curve,
    build_risk_budget_ranked_trade_table,
    finalize_risk_budget_diagnostics,
)


ASSETS = pd.Index(
    [
        "Upper",
        "Entry",
        "Exit",
        "Equity Far",
        "FI In Band",
        "FI Sub Material",
        "Alt Off Cadence",
        "Cash",
    ]
)


ASSET_CLASSES = ("Liquidity", "Fixed Income", "Equity", "Alternatives")


def _config() -> RiskBudgetRankedExecutionPolicyConfig:
    """Return one explicit test policy."""
    return RiskBudgetRankedExecutionPolicyConfig(
        tre_weight_by_asset_class={k: 1.0 for k in ASSET_CLASSES},
        small_trade_cutoff=0.01,
        small_trade_cutoff_by_asset_class={asset_class: 0.01 for asset_class in ASSET_CLASSES},
        material_trade_threshold=0.01,
        execution_tre_budget=0.005,
        target_band_tight_by_asset_class={asset_class: 0.005 for asset_class in ASSET_CLASSES},
        target_band_wide_by_asset_class={asset_class: 0.02 for asset_class in ASSET_CLASSES},
    )


def _trade_table(
    config: RiskBudgetRankedExecutionPolicyConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Return a cross-section covering adjustment, transition and rank states."""
    current = pd.Series(
        [0.20, 0.00, 0.06, 0.05, 0.10, 0.10, 0.10, 0.39],
        index=ASSETS,
    )
    model = pd.Series(
        [0.15, 0.08, 0.00, 0.15, 0.105, 0.125, 0.20, 0.19],
        index=ASSETS,
    )
    maximum = pd.Series(1.0, index=ASSETS)
    maximum.loc["Upper"] = 0.10
    groups = pd.Series(
        [
            "Equity",
            "Equity",
            "Equity",
            "Equity",
            "Fixed Income",
            "Fixed Income",
            "Alternatives",
            "Liquidity",
        ],
        index=ASSETS,
    )
    allowed = pd.Series(True, index=ASSETS)
    allowed.loc["Alt Off Cadence"] = False
    config = _config() if config is None else config
    scaffold = build_execution_target(
        current_weights=current,
        model_weights=model,
        previous_model_weights=current,
        settlement_cash_asset="Cash",
        config=replace(
            config,
            instruction_correction_tolerance=0.0,
            sell_down_dust_threshold=0.0,
        ),
        rebalancing_allowed=allowed,
        required_min_weights=pd.Series(0.0, index=ASSETS),
        required_max_weights=maximum,
        asset_classes=groups,
    )
    covariance = pd.DataFrame(
        np.eye(len(ASSETS)) * 0.04,
        index=ASSETS,
        columns=ASSETS,
    )
    alphas = pd.Series(
        [0.0, 0.0, 0.0, 1.0, np.nan, 0.0, 0.0, 0.0],
        index=ASSETS,
    )
    table = build_risk_budget_ranked_trade_table(
        scaffold=scaffold,
        alphas=alphas,
        covariance=covariance,
        asset_classes=groups,
        asset_class_tre_weights=pd.Series(1.0, index=ASSET_CLASSES),
        config=config,
    )
    return table, covariance, groups


def test_current_model_classification_and_complete_class_ranks() -> None:
    """Classification ignores previous model and ranks every overlap holding."""
    table, _, _ = _trade_table()

    assert table.loc["Upper", MANDATORY_CATEGORY] == "product_max_sell"
    assert table.loc["Upper", BASE_WEIGHT] == pytest.approx(0.10)
    assert table.loc["Entry", MODEL_ENTRY]
    assert table.loc["Exit", MODEL_EXIT]
    assert table.loc["FI Sub Material", RULE4_TRADE]
    assert table.loc["FI Sub Material", ASSET_CLASS_RANK] == 1
    assert table.loc["FI In Band", ASSET_CLASS_RANK] == 2
    assert table.loc["Alt Off Cadence", TRADE_RANK] == 1
    assert not table.loc["FI In Band", MATERIAL_TRADE]
    assert table.loc["FI Sub Material", MATERIAL_TRADE]
    assert table.loc["FI Sub Material", RANKING_ELIGIBLE]
    assert not table.loc["Alt Off Cadence", RANKING_ELIGIBLE]
    assert table.loc["Equity Far", RISK_BUDGET_SELECTED]
    assert table.loc["Equity Far", SELECTED_TRADE]
    assert table.loc["Equity Far", GROUP_K_RISK] == 1


def test_alpha_strength_bands_and_missing_alpha_audit() -> None:
    """Strong alpha tightens the band and missing alpha uses the wide band."""
    table, _, _ = _trade_table()

    assert table.loc["Equity Far", TARGET_BAND_WIDTH] == pytest.approx(0.005)
    assert table.loc["FI In Band", TARGET_BAND_WIDTH] == pytest.approx(0.02)
    assert table.loc["FI In Band", ALPHA_MISSING]
    assert table.loc["Entry", TARGET_BAND_LOWER] <= 0.08
    assert table.loc["Entry", TARGET_BAND_LOWER] >= 0.01
    assert table.loc["Entry", TARGET_BAND_UPPER] >= 0.08
    assert table.loc["FI In Band", ABS_DISTANCE_TO_BAND] == pytest.approx(0.0)


def test_group_budgets_normalize_and_select_only_eligible_prefixes() -> None:
    """Each class gets one quarter of B and only eligible prefixes can be used."""
    table, _, _ = _trade_table()

    assert table.loc["Equity Far", GROUP_TRE_WEIGHT_NORMALIZED] == pytest.approx(0.25)
    assert table.loc["Equity Far", GROUP_TRE_BUDGET] == pytest.approx(0.00125)
    assert table.loc["FI In Band", GROUP_SELECTION_STATUS] == "unattainable"
    assert not table.loc["FI In Band", RISK_BUDGET_SELECTED]
    assert table.loc["FI Sub Material", RISK_BUDGET_SELECTED]

    summary = build_risk_budget_group_summary(table)
    prefixes = build_risk_budget_prefix_curve(table)
    assert summary.index.tolist() == list(ASSET_CLASSES)
    assert set(prefixes.index.get_level_values("Asset Class")) == set(ASSET_CLASSES)

    equity_candidates = table.index[
        table[RANKING_ELIGIBLE] & table["asset_class"].eq("Equity")
    ].tolist()
    equity_candidates.sort(key=lambda asset: int(table.loc[asset, TRADE_RANK]))
    cash = table.index[table["settlement_cash"]][0]
    group_mask = table["asset_class"].eq("Equity") & ~table["settlement_cash"]
    for prefix_size in range(len(equity_candidates) + 1):
        hypothetical = table[BASE_WEIGHT].copy()
        selected = equity_candidates[:prefix_size]
        hypothetical.loc[selected] = table.loc[selected, "policy_target_weight"]
        hypothetical.loc[cash] = 1.0 - hypothetical.drop(index=cash).sum()
        active = pd.Series(0.0, index=table.index)
        active.loc[group_mask] = (
            hypothetical.loc[group_mask] - table.loc[group_mask, "effective_model_weight"]
        )
        active.loc[cash] = -active.sum()
        covariance = np.eye(len(table)) * 0.04
        expected = float(np.sqrt(active.to_numpy() @ covariance @ active.to_numpy()))
        assert prefixes.loc[("Equity", prefix_size), "Group TRE"] == pytest.approx(expected)


def test_non_monotone_prefix_curve_uses_smallest_budget_attaining_prefix() -> None:
    """Every prefix is evaluated even when an early ranked trade raises TRE."""
    assets = pd.Index(["Tight Buy", "Wide Sell", "Cash"])
    groups = pd.Series(["Equity", "Equity", "Liquidity"], index=assets)
    current = pd.Series([0.05, 0.20, 0.75], index=assets)
    model = pd.Series([0.10, 0.10, 0.80], index=assets)
    config = replace(
        _config(),
        material_trade_threshold=0.001,
        execution_tre_budget=0.08,
    )
    scaffold = build_execution_target(
        current_weights=current,
        model_weights=model,
        previous_model_weights=current,
        settlement_cash_asset="Cash",
        config=config,
        rebalancing_allowed=pd.Series(True, index=assets),
        required_min_weights=pd.Series(0.0, index=assets),
        required_max_weights=pd.Series(1.0, index=assets),
        asset_classes=groups,
    )
    covariance = pd.DataFrame(
        np.eye(len(assets)) * 0.04,
        index=assets,
        columns=assets,
    )
    table = build_risk_budget_ranked_trade_table(
        scaffold=scaffold,
        alphas=pd.Series([1.0, 0.0, 0.0], index=assets),
        covariance=covariance,
        asset_classes=groups,
        asset_class_tre_weights=pd.Series(1.0, index=ASSET_CLASSES),
        config=config,
    )
    curve = build_risk_budget_prefix_curve(table).loc["Equity"]

    assert curve.loc[1, "Group TRE"] > curve.loc[0, "Group TRE"]
    assert curve.loc[0, "Group TRE"] > curve.loc[0, "TRE Budget"]
    assert curve.loc[2, "Group TRE"] <= curve.loc[2, "TRE Budget"]
    assert table.loc["Tight Buy", TRADE_RANK] == 1
    assert table.loc["Wide Sell", TRADE_RANK] == 2
    assert table.loc["Tight Buy", GROUP_K_RISK] == 2
    assert table.loc["Tight Buy", GROUP_SELECTION_STATUS] == "attained"
    assert table.loc[["Tight Buy", "Wide Sell"], RISK_BUDGET_SELECTED].all()


def test_model_gap_materiality_enforces_directional_minimum_ticket() -> None:
    """A material target gap remains eligible just outside the no-trade band."""
    assets = pd.Index(["Fund", "Cash"])
    groups = pd.Series(["Equity", "Liquidity"], index=assets)
    current = pd.Series([0.10, 0.90], index=assets)
    model = pd.Series([0.114, 0.886], index=assets)
    bands = {asset_class: 0.01 for asset_class in ASSET_CLASSES}
    config = replace(
        _config(),
        material_trade_threshold=0.005,
        execution_tre_budget=0.0001,
        target_band_tight_by_asset_class=bands,
        target_band_wide_by_asset_class=bands,
        max_trades=1,
        add_mandatory_trades_to_max_trades=True,
    )
    scaffold = build_execution_target(
        current_weights=current,
        model_weights=model,
        previous_model_weights=current,
        settlement_cash_asset="Cash",
        config=config,
        rebalancing_allowed=pd.Series(True, index=assets),
        required_min_weights=pd.Series(0.0, index=assets),
        required_max_weights=pd.Series(1.0, index=assets),
        asset_classes=groups,
    )
    covariance = pd.DataFrame(np.eye(len(assets)) * 0.04, index=assets, columns=assets)
    table = build_risk_budget_ranked_trade_table(
        scaffold=scaffold,
        alphas=pd.Series(0.0, index=assets),
        covariance=covariance,
        asset_classes=groups,
        asset_class_tre_weights=pd.Series(1.0, index=ASSET_CLASSES),
        config=config,
    )

    assert table.loc["Fund", MATERIAL_TRADE]
    assert table.loc["Fund", RANKING_ELIGIBLE]
    assert table.loc["Fund", RISK_BUDGET_SELECTED]
    result = solve_feasible_execution_portfolio(
        optimiser_config=OptimiserConfig(solver="CLARABEL"),
        trade_table=table,
        base_constraints=Constraints(
            min_weights=pd.Series(0.0, index=assets),
            max_weights=pd.Series(1.0, index=assets),
        ),
        covariance=covariance,
        expand_for_feasibility=True,
        group_split_asset_classes=groups,
        context="directional minimum-ticket test",
    )

    assert result.accepted and result.compliant
    assert result.weights.loc["Fund"] - current.loc["Fund"] >= 0.005 - 2e-5


def test_model_gap_materiality_enforces_directional_minimum_sell() -> None:
    """A selected sell is bounded away from its post-compulsory base."""
    assets = pd.Index(["Fund", "Cash"])
    groups = pd.Series(["Equity", "Liquidity"], index=assets)
    current = pd.Series([0.114, 0.886], index=assets)
    model = pd.Series([0.10, 0.90], index=assets)
    bands = {asset_class: 0.01 for asset_class in ASSET_CLASSES}
    config = replace(
        _config(),
        material_trade_threshold=0.005,
        execution_tre_budget=0.0001,
        target_band_tight_by_asset_class=bands,
        target_band_wide_by_asset_class=bands,
        max_trades=1,
        add_mandatory_trades_to_max_trades=True,
    )
    scaffold = build_execution_target(
        current_weights=current,
        model_weights=model,
        previous_model_weights=current,
        settlement_cash_asset="Cash",
        config=config,
        rebalancing_allowed=pd.Series(True, index=assets),
        required_min_weights=pd.Series(0.0, index=assets),
        required_max_weights=pd.Series(1.0, index=assets),
        asset_classes=groups,
    )
    covariance = pd.DataFrame(np.eye(len(assets)) * 0.04, index=assets, columns=assets)
    table = build_risk_budget_ranked_trade_table(
        scaffold=scaffold,
        alphas=pd.Series(0.0, index=assets),
        covariance=covariance,
        asset_classes=groups,
        asset_class_tre_weights=pd.Series(1.0, index=ASSET_CLASSES),
        config=config,
    )
    result = solve_feasible_execution_portfolio(
        optimiser_config=OptimiserConfig(solver="CLARABEL"),
        trade_table=table,
        base_constraints=Constraints(
            min_weights=pd.Series(0.0, index=assets),
            max_weights=pd.Series(1.0, index=assets),
        ),
        covariance=covariance,
        expand_for_feasibility=True,
        group_split_asset_classes=groups,
        context="directional minimum-sell test",
    )

    assert result.accepted and result.compliant
    assert current.loc["Fund"] - result.weights.loc["Fund"] >= 0.005 - 2e-5


def test_hard_max_repair_can_continue_toward_clipped_model() -> None:
    """A mandatory limit repair remains movable toward the compliant target."""
    assets = pd.Index(["Upper", "Peer", "Cash"])
    groups = pd.Series(["Equity", "Equity", "Liquidity"], index=assets)
    current = pd.Series([0.20, 0.20, 0.60], index=assets)
    model = pd.Series([0.05, 0.20, 0.75], index=assets)
    maximum = pd.Series([0.10, 1.0, 1.0], index=assets)
    config = replace(
        _config(),
        execution_tre_budget=0.10,
        max_trades=1,
        add_mandatory_trades_to_max_trades=True,
    )
    scaffold = build_execution_target(
        current_weights=current,
        model_weights=model,
        previous_model_weights=current,
        settlement_cash_asset="Cash",
        config=config,
        rebalancing_allowed=pd.Series(True, index=assets),
        required_min_weights=pd.Series(0.0, index=assets),
        required_max_weights=maximum,
        asset_classes=groups,
    )
    covariance = pd.DataFrame(np.eye(len(assets)) * 0.04, index=assets, columns=assets)
    table = build_risk_budget_ranked_trade_table(
        scaffold=scaffold,
        alphas=pd.Series(0.0, index=assets),
        covariance=covariance,
        asset_classes=groups,
        asset_class_tre_weights=pd.Series(1.0, index=ASSET_CLASSES),
        config=config,
    )
    result = solve_feasible_execution_portfolio(
        optimiser_config=OptimiserConfig(solver="CLARABEL"),
        trade_table=table,
        base_constraints=Constraints(
            min_weights=pd.Series(0.0, index=assets),
            max_weights=maximum,
        ),
        covariance=covariance,
        expand_for_feasibility=True,
        group_split_asset_classes=groups,
        context="one-sided hard-repair corridor test",
    )

    assert result.accepted and result.compliant
    assert result.trade_table.loc["Upper", MANDATORY_CORRIDOR]
    assert 0.05 - 2e-5 <= result.weights.loc["Upper"] < 0.10 - 2e-5


def test_hard_min_repair_can_continue_toward_clipped_model() -> None:
    """A mandatory minimum repair remains movable toward the compliant target."""
    assets = pd.Index(["Lower", "Peer", "Cash"])
    groups = pd.Series(["Equity", "Equity", "Liquidity"], index=assets)
    current = pd.Series([0.02, 0.20, 0.78], index=assets)
    model = pd.Series([0.15, 0.20, 0.65], index=assets)
    minimum = pd.Series([0.10, 0.0, 0.0], index=assets)
    maximum = pd.Series(1.0, index=assets)
    config = replace(
        _config(),
        execution_tre_budget=0.10,
        max_trades=1,
        add_mandatory_trades_to_max_trades=True,
    )
    scaffold = build_execution_target(
        current_weights=current,
        model_weights=model,
        previous_model_weights=current,
        settlement_cash_asset="Cash",
        config=config,
        rebalancing_allowed=pd.Series(True, index=assets),
        required_min_weights=minimum,
        required_max_weights=maximum,
        asset_classes=groups,
    )
    covariance = pd.DataFrame(np.eye(len(assets)) * 0.04, index=assets, columns=assets)
    table = build_risk_budget_ranked_trade_table(
        scaffold=scaffold,
        alphas=pd.Series(0.0, index=assets),
        covariance=covariance,
        asset_classes=groups,
        asset_class_tre_weights=pd.Series(1.0, index=ASSET_CLASSES),
        config=config,
    )
    result = solve_feasible_execution_portfolio(
        optimiser_config=OptimiserConfig(solver="CLARABEL"),
        trade_table=table,
        base_constraints=Constraints(
            min_weights=minimum,
            max_weights=maximum,
        ),
        covariance=covariance,
        expand_for_feasibility=True,
        group_split_asset_classes=groups,
        context="one-sided hard-min repair corridor test",
    )

    assert result.accepted and result.compliant
    assert result.trade_table.loc["Lower", MANDATORY_CORRIDOR]
    assert 0.10 + 2e-5 < result.weights.loc["Lower"] <= 0.15 + 2e-5


def test_group_split_solve_uses_bands_and_freezes_unselected_rows() -> None:
    """The final QP moves selected/entry rows and freezes unselected overlaps."""
    table, covariance, groups = _trade_table()
    maximum = pd.Series(1.0, index=ASSETS)
    maximum.loc["Upper"] = 0.10
    result = solve_feasible_execution_portfolio(
        optimiser_config=OptimiserConfig(solver="CLARABEL"),
        trade_table=table,
        base_constraints=Constraints(
            min_weights=pd.Series(0.0, index=ASSETS),
            max_weights=maximum,
        ),
        covariance=covariance,
        expand_for_feasibility=True,
        group_split_asset_classes=groups,
        context="target-band test",
    )

    assert result.accepted and result.compliant
    assert result.weights.loc["Upper"] == pytest.approx(0.10, abs=2e-5)
    assert result.weights.loc["Exit"] == pytest.approx(0.0, abs=2e-5)
    assert (
        table.loc["Equity Far", TARGET_BAND_LOWER] - 2e-5
        <= result.weights.loc["Equity Far"]
        <= table.loc["Equity Far", TARGET_BAND_UPPER] + 2e-5
    )
    assert result.weights.loc["FI In Band"] == pytest.approx(0.10, abs=2e-5)
    assert result.weights.loc["FI Sub Material"] >= 0.11 - 2e-5
    assert (
        table.loc["Entry", TARGET_BAND_LOWER] - 2e-5
        <= result.weights.loc["Entry"]
        <= table.loc["Entry", TARGET_BAND_UPPER] + 2e-5
    )
    assert not result.trade_table[FEASIBILITY_RESCUE_TRADE].any()
    audited = finalize_risk_budget_diagnostics(
        table=result.trade_table,
        proposed_weights=result.weights,
        covariance=covariance,
    )
    summary = build_risk_budget_group_summary(audited)
    assert summary.loc["Equity", "Risk K"] == 1
    assert summary.loc["Equity", "Final K"] == 1


def test_proved_group_tre_infeasibility_admits_next_ranked_candidate() -> None:
    """A hard group-TRE breach admits only its next eligible ranked name."""
    config = replace(
        _config(),
        material_trade_threshold=0.004,
        execution_tre_budget=0.04,
    )
    table, covariance, groups = _trade_table(config=config)
    candidate = "FI Sub Material"
    assert table.loc[candidate, RANKING_ELIGIBLE]
    assert not table.loc[candidate, RISK_BUDGET_SELECTED]

    group_loadings = pd.DataFrame(
        {
            "Fixed Income": groups.eq("Fixed Income").astype(float),
        },
        index=table.index,
    )
    result = solve_feasible_execution_portfolio(
        optimiser_config=OptimiserConfig(solver="CLARABEL"),
        trade_table=table,
        base_constraints=Constraints(
            min_weights=pd.Series(0.0, index=table.index),
            max_weights=pd.Series(1.0, index=table.index),
            group_tracking_error_constraint=GroupTrackingErrorConstraint(
                group_loadings=group_loadings,
                group_tre_vols=pd.Series({"Fixed Income": 0.0045}),
            ),
        ),
        covariance=covariance,
        expand_for_feasibility=True,
        group_split_asset_classes=groups,
        context="hard group-TRE rescue test",
    )

    assert result.accepted and result.compliant
    assert result.trade_table.loc[candidate, FEASIBILITY_RESCUE_TRADE]
    assert result.trade_table.loc[candidate, "rescue_reason"] == ("group_tre:Fixed Income")
    assert (
        table.loc[candidate, TARGET_BAND_LOWER] - 2e-5
        <= result.weights.loc[candidate]
        <= table.loc[candidate, TARGET_BAND_UPPER] + 2e-5
    )
    group_active = (
        (result.weights - table[RAW_MODEL_WEIGHT]) * group_loadings["Fixed Income"]
    ).to_numpy(dtype=float)
    group_tre = float(np.sqrt(group_active @ covariance.to_numpy() @ group_active))
    assert group_tre <= 0.0045 + 1e-7
    assert not result.trade_table.loc["Equity Far", FEASIBILITY_RESCUE_TRADE]


def test_group_split_prunes_utilities_but_retains_true_hard_turnover() -> None:
    """Utility-only group objects cannot become malformed hard constraints."""
    table, _, _ = _trade_table()
    loadings = pd.DataFrame(
        {
            "Equity": table["asset_class"].eq("Equity").astype(float),
            "Other": ~table["asset_class"].eq("Equity"),
        },
        index=table.index,
    ).astype(float)
    costs = pd.Series(np.linspace(1.0, 2.0, len(table)), index=table.index)
    constraints = Constraints(
        min_weights=pd.Series(0.0, index=table.index),
        max_weights=pd.Series(1.0, index=table.index),
        weights_0=table["current_weight"],
        turnover_constraint=2.0,
        turnover_costs=costs,
        constraint_enforcement_type=ConstraintEnforcementType.UTILITY_CONSTRAINTS,
        group_tracking_error_constraint=GroupTrackingErrorConstraint(
            group_loadings=loadings,
            group_tre_utility_weights=pd.Series(1.0, index=loadings.columns),
        ),
        group_turnover_constraint=GroupTurnoverConstraint(
            group_loadings=loadings,
            group_turnover_utility_weights=pd.Series(1.0, index=loadings.columns),
        ),
    )

    selected = build_selected_execution_constraints(
        base_constraints=constraints,
        trade_table=table,
    )

    assert selected.constraint_enforcement_type == (ConstraintEnforcementType.FORCED_CONSTRAINTS)
    assert selected.group_tracking_error_constraint is None
    assert selected.group_turnover_constraint is None
    assert selected.turnover_constraint == pytest.approx(2.0)
    assert selected.turnover_costs.equals(costs)


def test_any_strictly_positive_hard_minimum_prevents_model_exit() -> None:
    """A tiny positive product minimum cannot be overwritten by an exit to zero."""
    assets = pd.Index(["Holding", "Cash"])
    groups = pd.Series(["Equity", "Liquidity"], index=assets)
    config = _config()
    current = pd.Series([0.02, 0.98], index=assets)
    model = pd.Series([0.0, 1.0], index=assets)
    scaffold = build_execution_target(
        current_weights=current,
        model_weights=model,
        previous_model_weights=current,
        settlement_cash_asset="Cash",
        config=config,
        required_min_weights=pd.Series([1e-10, 0.0], index=assets),
        required_max_weights=pd.Series(1.0, index=assets),
        asset_classes=groups,
    )
    table = build_risk_budget_ranked_trade_table(
        scaffold=scaffold,
        alphas=pd.Series(0.0, index=assets),
        covariance=pd.DataFrame(np.eye(2) * 0.04, index=assets, columns=assets),
        asset_classes=groups,
        asset_class_tre_weights=pd.Series(1.0, index=ASSET_CLASSES),
        config=config,
    )

    assert not table.loc["Holding", MODEL_EXIT]
    assert not table.loc["Holding", MANDATORY_TRADE]
    assert table.loc["Holding", BASE_WEIGHT] == pytest.approx(0.02)
    summary = build_risk_budget_group_summary(table)
    assert summary.index.tolist() == list(ASSET_CLASSES)


def test_serialized_source_missing_alpha_uses_the_wide_band() -> None:
    """Replay zero fills retain the source-missing alpha state for audit."""
    table, covariance, groups = _trade_table()
    date = pd.Timestamp("2026-08-31")
    alphas = pd.Series(0.0, index=table.index, name=date)
    alphas.attrs["rosaa_alpha_missing"] = frozenset({(date, "Equity Far")})
    rebuilt = build_risk_budget_ranked_trade_table(
        scaffold=table,
        alphas=alphas,
        covariance=covariance,
        asset_classes=groups,
        asset_class_tre_weights=pd.Series(1.0, index=ASSET_CLASSES),
        config=_config(),
    )

    assert rebuilt.loc["Equity Far", ALPHA_MISSING]
    assert rebuilt.loc["Equity Far", TARGET_BAND_WIDTH] == pytest.approx(0.02)
