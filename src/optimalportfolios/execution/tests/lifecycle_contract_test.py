"""
Execution-policy target, calendar, drift, ranking and output-label tests.

The fixtures are deliberately small and self-contained. They exercise the
policy mechanics without loading a mandate workbook or solving a portfolio.

"""

import numpy as np


import pandas as pd


import pytest


from optimalportfolios.execution.lifecycle import (
    BASE_WEIGHT,
    CUTOFF_EXIT,
    DESIRED_REWEIGHT,
    EFFECTIVE_MODEL_WEIGHT,
    LOW_WEIGHT_EXCLUDED,
    MANDATORY_FUNDING,
    MANDATORY_REASON,
    MANDATORY_CATEGORY,
    MANDATORY_TRADE,
    MARGINAL_ALPHA,
    MARGINAL_TRE,
    MODEL_ENTRY,
    MODEL_EXIT,
    REBALANCE_CADENCE_ELIGIBLE,
    RULE4_TRADE,
    SELECTED_TRADE,
    TRADE_CANDIDATE,
    TRADE_RANK,
    ExecutionPolicyConfig,
    MandatoryTrades,
    build_execution_target,
    build_instrument_rebalancing_indicators,
    drift_execution_weights,
    score_execution_trades,
    resolve_effective_max_trades,
)


ASSET_CLASSES = ("Liquidity", "Fixed Income", "Equity", "Alternatives")


def test_effective_trade_budget_can_add_mandatory_tickets() -> None:
    """Five discretionary tickets plus four mandatory tickets allow nine total."""
    total_cap = ExecutionPolicyConfig(
        tre_weight_by_asset_class={k: 1.0 for k in ASSET_CLASSES}, max_trades=5
    )
    expanded_cap = ExecutionPolicyConfig(
        tre_weight_by_asset_class={k: 1.0 for k in ASSET_CLASSES},
        max_trades=5,
        add_mandatory_trades_to_max_trades=True,
    )

    assert resolve_effective_max_trades(total_cap, mandatory_ticket_count=4) == 5
    assert resolve_effective_max_trades(expanded_cap, mandatory_ticket_count=4) == 9
    assert resolve_effective_max_trades(expanded_cap, mandatory_ticket_count=0) == 5
    assert (
        resolve_effective_max_trades(
            ExecutionPolicyConfig(
                tre_weight_by_asset_class={k: 1.0 for k in ASSET_CLASSES}, max_trades=None
            ),
            mandatory_ticket_count=4,
        )
        is None
    )


def test_policy_config_accepts_strictly_positive_reference_materiality() -> None:
    """An all-trades sweep reference may use a threshold below weight tolerance."""
    config = ExecutionPolicyConfig(
        tre_weight_by_asset_class={k: 1.0 for k in ASSET_CLASSES}, material_trade_threshold=1e-12
    )

    assert config.material_trade_threshold == pytest.approx(1e-12)


def test_execution_target_applies_cutoff_and_live_mandatory_bounds() -> None:
    """Cutoffs remain historical policy; live bounds can override cadence."""
    assets = pd.Index(["Fund A", "Fund B", "Fund C", "Fund D"])
    current = pd.Series([0.01, 0.02, 0.03, 0.94], index=assets)
    model = pd.Series([0.02, 0.0, 0.04, 0.94], index=assets)
    cadence = pd.Series(False, index=assets)
    config = ExecutionPolicyConfig(
        tre_weight_by_asset_class={k: 1.0 for k in ASSET_CLASSES},
        mandatory_trades=MandatoryTrades(hard_buy_min={"Fund A": 0.03}, hard_sell=("Fund C",)),
    )

    target = build_execution_target(
        current_weights=current,
        model_weights=model,
        config=config,
        settlement_cash_asset="Fund D",
        rebalancing_allowed=cadence,
    )

    assert target.at["Fund A", EFFECTIVE_MODEL_WEIGHT] == pytest.approx(0.03)
    assert target.at["Fund C", EFFECTIVE_MODEL_WEIGHT] == pytest.approx(0.0)
    assert target.at["Fund B", EFFECTIVE_MODEL_WEIGHT] == pytest.approx(0.0)
    assert bool(target.at["Fund B", LOW_WEIGHT_EXCLUDED])
    assert not bool(target.at["Fund B", REBALANCE_CADENCE_ELIGIBLE])
    assert not bool(target.at["Fund B", TRADE_CANDIDATE])
    for instrument in ("Fund A", "Fund B", "Fund C"):
        assert bool(target.at[instrument, MANDATORY_TRADE])
        assert not bool(target.at[instrument, REBALANCE_CADENCE_ELIGIBLE])
        assert not bool(target.at[instrument, TRADE_CANDIDATE])
    assert target[BASE_WEIGHT].sum() == pytest.approx(current.sum())


def test_explicit_hard_sell_and_buy_override_the_model() -> None:
    """Explicit live instructions apply even when workbook bounds are disabled."""
    current = pd.Series({"Buy": 0.00, "Sell": 0.04, "Hold": 0.86, "Cash": 0.10})
    model = pd.Series({"Buy": 0.002, "Sell": 0.06, "Hold": 0.838, "Cash": 0.10})
    config = ExecutionPolicyConfig(
        tre_weight_by_asset_class={k: 1.0 for k in ASSET_CLASSES},
        mandatory_trades=MandatoryTrades(
            hard_sell=("Sell",),
            hard_buy_min={"Buy": 0.03},
        ),
    )
    target = build_execution_target(current, model, config, settlement_cash_asset="Cash")
    assert target.at["Buy", EFFECTIVE_MODEL_WEIGHT] == pytest.approx(0.03)
    assert target.at["Sell", EFFECTIVE_MODEL_WEIGHT] == pytest.approx(0.0)
    assert target.loc[["Buy", "Sell"], MANDATORY_TRADE].all()
    assert target.at["Cash", BASE_WEIGHT] == pytest.approx(0.11)


def test_mandatory_trade_ignores_the_materiality_cutoff() -> None:
    """A small hard instruction remains a candidate regardless of trade size."""
    current = pd.Series({"Fund": 0.0100, "Cash": 0.9900})
    model = pd.Series({"Fund": 0.0101, "Cash": 0.9899})
    config = ExecutionPolicyConfig(
        tre_weight_by_asset_class={k: 1.0 for k in ASSET_CLASSES},
        material_trade_threshold=0.005,
        mandatory_trades=MandatoryTrades(hard_buy_min={"Fund": 0.0101}),
    )
    target = build_execution_target(current, model, config)
    assert bool(target.at["Fund", MANDATORY_TRADE])
    assert not bool(target.at["Fund", TRADE_CANDIDATE])


def test_current_product_bound_violation_becomes_a_mandatory_repair() -> None:
    """Small drift outside an ordinary bound bypasses every policy cutoff."""
    current = pd.Series({"Fund": 0.0503, "Cash": 0.9497})
    model = pd.Series({"Fund": 0.0502, "Cash": 0.9498})
    target = build_execution_target(
        current_weights=current,
        model_weights=model,
        config=ExecutionPolicyConfig(
            tre_weight_by_asset_class={k: 1.0 for k in ASSET_CLASSES},
            material_trade_threshold=0.005,
        ),
        rebalancing_allowed=pd.Series(False, index=current.index),
        required_min_weights=pd.Series({"Fund": 0.0, "Cash": 0.0}),
        required_max_weights=pd.Series({"Fund": 0.05, "Cash": 1.0}),
    )
    assert target.at["Fund", EFFECTIVE_MODEL_WEIGHT] == pytest.approx(0.05)
    assert bool(target.at["Fund", MANDATORY_TRADE])
    assert not bool(target.at["Fund", REBALANCE_CADENCE_ELIGIBLE])
    assert not bool(target.at["Fund", TRADE_CANDIDATE])


def test_model_exit_is_not_overwritten_by_current_product_max_repair() -> None:
    """A valid sell-to-zero takes precedence over the current-bound repair."""
    current = pd.Series({"Fund": 0.03, "Cash": 0.97})
    previous_model = pd.Series({"Fund": 0.02, "Cash": 0.98})
    model = pd.Series({"Fund": 0.0, "Cash": 1.0})
    target = build_execution_target(
        current_weights=current,
        model_weights=model,
        previous_model_weights=previous_model,
        config=ExecutionPolicyConfig(
            tre_weight_by_asset_class={k: 1.0 for k in ASSET_CLASSES}, small_trade_cutoff=0.0
        ),
        settlement_cash_asset="Cash",
        required_min_weights=pd.Series({"Fund": 0.0, "Cash": 0.0}),
        required_max_weights=pd.Series({"Fund": 0.02, "Cash": 1.0}),
    )

    assert target.at["Fund", BASE_WEIGHT] == pytest.approx(0.0)
    assert target.at["Fund", EFFECTIVE_MODEL_WEIGHT] == pytest.approx(0.0)
    assert bool(target.at["Fund", MODEL_EXIT])


def test_off_cadence_model_entry_and_exit_are_both_mandatory() -> None:
    """Lifecycle transitions bypass the discretionary rebalance calendar symmetrically."""
    assets = pd.Index(["Entry", "Exit", "Cash"])
    target = build_execution_target(
        current_weights=pd.Series([0.0, 0.02, 0.98], index=assets),
        model_weights=pd.Series([0.02, 0.0, 0.98], index=assets),
        previous_model_weights=pd.Series([0.0, 0.02, 0.98], index=assets),
        config=ExecutionPolicyConfig(
            tre_weight_by_asset_class={k: 1.0 for k in ASSET_CLASSES},
        ),
        settlement_cash_asset="Cash",
        rebalancing_allowed=pd.Series(False, index=assets),
    )
    assert target.loc[["Entry", "Exit"], MANDATORY_TRADE].all()
    assert target.loc["Entry", MANDATORY_CATEGORY] == "model_entry"
    assert target.loc["Exit", MANDATORY_CATEGORY] == "model_exit"
    assert not target.loc[["Entry", "Exit"], REBALANCE_CADENCE_ELIGIBLE].any()


def test_positive_single_cutoff_drives_exclusion_entry_and_exit() -> None:
    """Rules 1-3 use the same inclusive positive model-weight threshold."""
    assets = pd.Index(["Cutoff", "Entry", "Exit", "Cash"])
    target = build_execution_target(
        current_weights=pd.Series([0.004, 0.0, 0.006, 0.990], index=assets),
        model_weights=pd.Series([0.003, 0.006, 0.004, 0.987], index=assets),
        previous_model_weights=pd.Series([0.004, 0.004, 0.006, 0.986], index=assets),
        config=ExecutionPolicyConfig(
            tre_weight_by_asset_class={k: 1.0 for k in ASSET_CLASSES}, small_trade_cutoff=0.005
        ),
        settlement_cash_asset="Cash",
    )

    assert bool(target.at["Cutoff", CUTOFF_EXIT])
    assert bool(target.at["Entry", MODEL_ENTRY])
    assert bool(target.at["Exit", MODEL_EXIT])
    assert target.loc[["Cutoff", "Entry", "Exit"], MANDATORY_TRADE].all()
    assert target.at["Cutoff", EFFECTIVE_MODEL_WEIGHT] == pytest.approx(0.0)
    assert target.at["Entry", EFFECTIVE_MODEL_WEIGHT] == pytest.approx(0.006)
    assert target.at["Exit", EFFECTIVE_MODEL_WEIGHT] == pytest.approx(0.0)


def test_instruction_bounds_do_not_alias_product_repair_inputs() -> None:
    """Mutating instruction bounds cannot invent an ordinary product repair."""
    assets = pd.Index(["Fund", "Cash"])
    target = build_execution_target(
        current_weights=pd.Series([0.01, 0.99], index=assets),
        model_weights=pd.Series([0.02, 0.98], index=assets),
        config=ExecutionPolicyConfig(
            tre_weight_by_asset_class={k: 1.0 for k in ASSET_CLASSES},
            mandatory_trades=MandatoryTrades(hard_target={"Fund": 0.03}),
        ),
        settlement_cash_asset="Cash",
        required_min_weights=pd.Series([0.0, 0.0], index=assets),
        required_max_weights=pd.Series([0.10, 1.0], index=assets),
    )
    assert "product_min_buy" not in target.at["Fund", MANDATORY_REASON]
    assert target.at["Fund", MANDATORY_REASON].endswith("fixed_instruction")


def test_effective_target_continues_from_product_repair_toward_model() -> None:
    """A mandatory boundary repair remains eligible to continue to the model."""
    current = pd.Series({"Fund": 0.08, "Cash": 0.92})
    model = pd.Series({"Fund": 0.04, "Cash": 0.96})
    target = build_execution_target(
        current_weights=current,
        model_weights=model,
        config=ExecutionPolicyConfig(
            tre_weight_by_asset_class={k: 1.0 for k in ASSET_CLASSES}, small_trade_cutoff=0.0
        ),
        settlement_cash_asset="Cash",
        required_min_weights=pd.Series({"Fund": 0.0, "Cash": 0.0}),
        required_max_weights=pd.Series({"Fund": 0.05, "Cash": 1.0}),
    )

    assert bool(target.at["Fund", MANDATORY_TRADE])
    assert target.at["Fund", BASE_WEIGHT] == pytest.approx(0.05)
    assert target.at["Fund", EFFECTIVE_MODEL_WEIGHT] == pytest.approx(0.05)
    assert target.at["Cash", EFFECTIVE_MODEL_WEIGHT] == pytest.approx(0.95)
    assert not bool(target.at["Fund", RULE4_TRADE])
    assert not bool(target.at["Fund", TRADE_CANDIDATE])
    assert target[EFFECTIVE_MODEL_WEIGHT].sum() == pytest.approx(1.0)


def test_effective_target_preserves_numerically_drifted_current_budget() -> None:
    """The settlement-cash leg uses the actual current sum, not a literal one."""
    current = pd.Series({"Fund": 0.08, "Cash": 0.920000000244})
    model = pd.Series({"Fund": 0.04, "Cash": 0.96})
    target = build_execution_target(
        current_weights=current,
        model_weights=model,
        config=ExecutionPolicyConfig(
            tre_weight_by_asset_class={k: 1.0 for k in ASSET_CLASSES}, small_trade_cutoff=0.0
        ),
        settlement_cash_asset="Cash",
    )

    assert target[EFFECTIVE_MODEL_WEIGHT].sum() == pytest.approx(current.sum(), abs=1e-12)


def test_entry_exit_cutoff_and_cash_are_consolidated_before_ranking() -> None:
    """Rules 1-3 produce fixed weights and one self-financing cash leg."""
    assets = pd.Index(["Entry", "Exit", "Cutoff", "Continue", "Cash"])
    current = pd.Series([0.0, 0.08, 0.004, 0.70, 0.216], index=assets)
    previous_model = pd.Series([0.0, 0.08, 0.004, 0.70, 0.216], index=assets)
    model = pd.Series([0.03, 0.0, 0.0, 0.75, 0.22], index=assets)
    target = build_execution_target(
        current_weights=current,
        model_weights=model,
        previous_model_weights=previous_model,
        settlement_cash_asset="Cash",
        config=ExecutionPolicyConfig(
            tre_weight_by_asset_class={k: 1.0 for k in ASSET_CLASSES}, small_trade_cutoff=0.0
        ),
    )

    assert bool(target.at["Entry", MODEL_ENTRY])
    assert bool(target.at["Exit", MODEL_EXIT])
    assert not bool(target.at["Cutoff", CUTOFF_EXIT])
    assert target.loc[["Entry", "Exit", "Cutoff"], MANDATORY_TRADE].all()
    assert target.at["Entry", BASE_WEIGHT] == pytest.approx(0.03)
    assert target.at["Exit", BASE_WEIGHT] == pytest.approx(0.0)
    assert target.at["Cutoff", BASE_WEIGHT] == pytest.approx(0.0)
    assert target.at["Cash", BASE_WEIGHT] == pytest.approx(0.270)
    assert target[BASE_WEIGHT].sum() == pytest.approx(1.0)
    assert target.loc[["Entry", "Exit", "Cutoff"], MANDATORY_FUNDING].sum() == (
        pytest.approx(-0.054)
    )
    assert bool(target.at["Continue", TRADE_CANDIDATE])
    assert target.at["Continue", DESIRED_REWEIGHT] == pytest.approx(0.05)


def test_fixed_current_bounds_override_model_and_remain_active_risk() -> None:
    """Fixed positive and zero desk targets are cash funded and not rebenchmarked."""
    assets = pd.Index(["Fixed Buy", "Fixed Sell", "Cash"])
    current = pd.Series([0.01, 0.04, 0.95], index=assets)
    model = pd.Series([0.005, 0.06, 0.935], index=assets)
    target = build_execution_target(
        current_weights=current,
        model_weights=model,
        settlement_cash_asset="Cash",
        config=ExecutionPolicyConfig(
            tre_weight_by_asset_class={k: 1.0 for k in ASSET_CLASSES},
            mandatory_trades=MandatoryTrades(
                hard_target={"Fixed Buy": 0.03}, hard_sell=("Fixed Sell",)
            ),
        ),
    )
    assert target.at["Fixed Buy", BASE_WEIGHT] == pytest.approx(0.03)
    assert target.at["Fixed Sell", BASE_WEIGHT] == pytest.approx(0.0)
    assert target.at["Cash", BASE_WEIGHT] == pytest.approx(0.97)
    assert target.at["Fixed Buy", BASE_WEIGHT] - model.at["Fixed Buy"] == pytest.approx(0.025)
    assert target.at["Fixed Sell", BASE_WEIGHT] - model.at["Fixed Sell"] == pytest.approx(-0.06)


def test_instrument_calendar_uses_monthly_and_quarterly_frequencies() -> None:
    """Quarterly assets freeze on non-quarterly model decision dates."""
    dates = pd.date_range("2024-01-31", "2024-06-30", freq="ME")
    frequencies = pd.Series({"Monthly": "ME", "Quarterly": "QE"})
    indicators = build_instrument_rebalancing_indicators(dates, frequencies)

    assert indicators["Monthly"].all()
    assert indicators.iloc[0].all()
    assert indicators.loc[pd.Timestamp("2024-02-29"), "Quarterly"] == 0
    assert indicators.loc[pd.Timestamp("2024-03-31"), "Quarterly"] == 1
    assert indicators.loc[pd.Timestamp("2024-06-30"), "Quarterly"] == 1


def test_weight_drift_matches_self_financing_identity() -> None:
    """Execution current weights are prior targets after realised return drift."""
    dates = pd.to_datetime(["2024-01-31", "2024-02-29"])
    prices = pd.DataFrame({"A": [100.0, 110.0], "B": [100.0, 100.0]}, index=dates)
    previous = pd.Series({"A": 0.5, "B": 0.5})
    drifted = drift_execution_weights(
        previous_target=previous,
        prices=prices,
        previous_date=dates[0],
        decision_date=dates[1],
    )
    expected = pd.Series({"A": 0.55 / 1.05, "B": 0.50 / 1.05})
    pd.testing.assert_series_equal(drifted, expected)


def test_trade_scores_use_full_covariance_and_cash_funding_direction() -> None:
    """Marginal alpha and TRE include the opposite settlement-cash leg."""
    assets = pd.Index(["FI", "EQ", "Cash"])
    current = pd.Series([0.60, 0.30, 0.10], index=assets)
    model = pd.Series([0.50, 0.40, 0.10], index=assets)
    config = ExecutionPolicyConfig(
        tre_weight_by_asset_class={k: 1.0 for k in ASSET_CLASSES}, max_trades=2
    )
    target = build_execution_target(current, model, config, settlement_cash_asset="Cash")
    groups = pd.Series(["Fixed Income", "Equity", "Liquidity"], index=assets)
    covariance = pd.DataFrame(
        [
            [0.04, 0.01, 0.002],
            [0.01, 0.09, 0.003],
            [0.002, 0.003, 0.001],
        ],
        index=assets,
        columns=assets,
    )
    scored = score_execution_trades(
        target=target,
        alphas=pd.Series({"FI": 0.01, "EQ": 0.03, "Cash": 0.005}),
        covariance=covariance,
        asset_classes=groups,
        config=config,
    )
    active = target[BASE_WEIGHT] - model
    tre = np.sqrt(float(active @ covariance @ active))
    delta = target.at["FI", DESIRED_REWEIGHT]
    funded = pd.Series({"FI": 1.0, "EQ": 0.0, "Cash": -1.0})
    moved = target[BASE_WEIGHT] + delta * funded
    expected_fi = np.sqrt(float((moved - model) @ covariance @ (moved - model))) - tre
    assert scored.at["FI", MARGINAL_TRE] == pytest.approx(expected_fi)
    assert scored.at["FI", MARGINAL_ALPHA] == pytest.approx(
        (0.01 - 0.005) * target.at["FI", DESIRED_REWEIGHT]
    )
    assert scored[TRADE_RANK].notna().sum() == 2
    assert int(scored[SELECTED_TRADE].sum()) == 2


def test_mandatory_trades_consume_the_trade_count_cutoff() -> None:
    """Mandatory tickets reserve capacity but are never dropped by the cap."""
    assets = pd.Index(["Mandatory", "Discretionary", "Cash"])
    current = pd.Series([0.1, 0.8, 0.1], index=assets)
    model = pd.Series([0.0, 0.9, 0.1], index=assets)
    config = ExecutionPolicyConfig(
        tre_weight_by_asset_class={k: 1.0 for k in ASSET_CLASSES},
        max_trades=1,
        mandatory_trades=MandatoryTrades(hard_sell=("Mandatory",)),
    )
    target = build_execution_target(current, model, config, settlement_cash_asset="Cash")
    covariance = pd.DataFrame(np.eye(3) * 0.04, index=assets, columns=assets)
    scored = score_execution_trades(
        target=target,
        alphas=pd.Series({"Mandatory": 10.0, "Discretionary": 20.0, "Cash": 0.0}),
        covariance=covariance,
        asset_classes=pd.Series("Equity", index=assets),
        config=config,
    )
    assert bool(scored.at["Mandatory", MANDATORY_TRADE])
    assert not bool(scored.at["Mandatory", SELECTED_TRADE])
    assert not bool(scored.at["Discretionary", SELECTED_TRADE])


def test_mandatory_trades_can_leave_the_discretionary_capacity_unchanged() -> None:
    """The optional budget mode keeps one discretionary slot after a hard sell."""
    assets = pd.Index(["Mandatory", "Discretionary", "Cash"])
    current = pd.Series([0.1, 0.8, 0.1], index=assets)
    model = pd.Series([0.0, 0.9, 0.1], index=assets)
    config = ExecutionPolicyConfig(
        tre_weight_by_asset_class={k: 1.0 for k in ASSET_CLASSES},
        max_trades=1,
        add_mandatory_trades_to_max_trades=True,
        mandatory_trades=MandatoryTrades(hard_sell=("Mandatory",)),
    )
    target = build_execution_target(current, model, config, settlement_cash_asset="Cash")
    covariance = pd.DataFrame(np.eye(3) * 0.04, index=assets, columns=assets)
    scored = score_execution_trades(
        target=target,
        alphas=pd.Series({"Mandatory": 10.0, "Discretionary": 20.0, "Cash": 0.0}),
        covariance=covariance,
        asset_classes=pd.Series("Equity", index=assets),
        config=config,
    )

    assert bool(scored.at["Mandatory", MANDATORY_TRADE])
    assert bool(scored.at["Discretionary", SELECTED_TRADE])
