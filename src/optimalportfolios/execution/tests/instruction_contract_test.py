"""Desk-instruction dead-band target and solve regressions."""

from optimalportfolios.execution.production import RankedExecutionPolicyConfig, solve_ranked_execution_portfolio


import numpy as np


import pandas as pd


import pytest


from optimalportfolios import Constraints, GroupLowerUpperConstraints


from optimalportfolios.execution.instrument_bounds import _build_selected_execution_bound_series


from optimalportfolios.execution.lifecycle import (
    BASE_WEIGHT,
    INSTRUCTION_MAX_SELL,
    INSTRUCTION_RETAINED_BREACH,
    INSTRUCTION_WITHIN_TOLERANCE,
    MANDATORY_FUNDING,
    MANDATORY_TRADE,
    PINNED_INSTRUCTION,
    POLICY_MAX_WEIGHT,
    POLICY_MIN_WEIGHT,
    PRODUCT_BOUND_OVERRIDE,
    PRODUCT_BOUND_OVERRIDE_BREACH,
    RULE4_TRADE,
    SELECTED_TRADE,
    TRADE_WEIGHT,
    ExecutionPolicyConfig,
    MandatoryTrades,
    build_execution_target,
)


ASSETS = pd.Index(["Risk", "Cash"])


ASSET_CLASSES = ("Liquidity", "Fixed Income", "Equity", "Alternatives")


def _instruction_audit(kind: str) -> pd.DataFrame:
    """Return one desk-owned instruction provenance row."""
    return pd.DataFrame(
        {
            "Instruction": [kind],
            "Comment": ["test desk bound"],
            "Source": ["current_min_max"],
            "allow_current_pin": [True],
            "model_max_repair": [True],
        },
        index=["Risk"],
    )


def _direct_deadband_reference(
    current: pd.Series,
    instruction_min: float,
    instruction_max: float,
    tolerance: float,
) -> pd.DataFrame:
    """Compute the roadmap reference directly with pure pandas arithmetic.

    This deliberately does not call :func:`build_execution_target`.  It is the
    independent second-pass reference for executable instruction intervals,
    mandatory targets and settlement-cash funding required by the roadmap.
    """
    fixed = np.isclose(instruction_min, instruction_max, atol=1e-8, rtol=0.0)
    low_breach = max(instruction_min - float(current["Risk"]), 0.0)
    high_breach = max(float(current["Risk"]) - instruction_max, 0.0)
    breach = (
        abs(instruction_min - float(current["Risk"])) if fixed else max(low_breach, high_breach)
    )
    within = tolerance > 0.0 and 1e-8 < breach <= tolerance

    executable_min = instruction_min
    executable_max = instruction_max
    if within and fixed:
        executable_min = float(current["Risk"])
        executable_max = float(current["Risk"])
    elif within and low_breach > 0.0:
        executable_min = float(current["Risk"])
    elif within and high_breach > 0.0:
        executable_max = float(current["Risk"])

    risk_base = float(current["Risk"])
    if not within:
        risk_base = min(max(risk_base, executable_min), executable_max)
    funding = risk_base - float(current["Risk"])
    return pd.DataFrame(
        {
            "executable_min": [executable_min, 0.0],
            "executable_max": [executable_max, 1.0],
            "base": [risk_base, float(current["Cash"]) - funding],
            "mandatory_funding": [funding, -funding],
            "within_tolerance": [within, False],
            "retained_breach": [breach if within else 0.0, 0.0],
        },
        index=ASSETS,
    )


def _build_upper_bound_target(tolerance: float) -> pd.DataFrame:
    """Build the representative 3.18 bp Current Max breach fixture."""
    current = pd.Series([0.050318, 0.949682], index=ASSETS)
    return build_execution_target(
        current_weights=current,
        model_weights=pd.Series([0.0445, 0.9555], index=ASSETS),
        config=ExecutionPolicyConfig(
            tre_weight_by_asset_class={k: 1.0 for k in ASSET_CLASSES},
            instruction_correction_tolerance=tolerance,
        ),
        settlement_cash_asset="Cash",
        mandatory_trades=MandatoryTrades(hard_max={"Risk": 0.05}),
        instruction_audit=_instruction_audit("hard_max"),
    )


def test_current_max_deadband_skips_ticket_and_zero_tolerance_reaches_model() -> None:
    """A retained breach has no ticket; zero tolerance reaches model within cap."""
    current = pd.Series([0.050318, 0.949682], index=ASSETS)
    reference = _direct_deadband_reference(current, 0.0, 0.05, 0.005)
    retained = _build_upper_bound_target(tolerance=0.005)

    assert retained.at["Risk", INSTRUCTION_WITHIN_TOLERANCE]
    assert retained.at["Risk", INSTRUCTION_RETAINED_BREACH] == pytest.approx(
        reference.at["Risk", "retained_breach"]
    )
    assert retained.at["Risk", POLICY_MIN_WEIGHT] == pytest.approx(
        reference.at["Risk", "executable_min"]
    )
    assert retained.at["Risk", POLICY_MAX_WEIGHT] == pytest.approx(
        reference.at["Risk", "executable_max"]
    )
    pd.testing.assert_series_equal(retained[BASE_WEIGHT], reference["base"], check_names=False)
    pd.testing.assert_series_equal(
        retained[MANDATORY_FUNDING],
        reference["mandatory_funding"],
        check_names=False,
    )
    assert not retained.at["Risk", MANDATORY_TRADE]

    exact = _build_upper_bound_target(tolerance=0.0)
    assert exact.at["Risk", INSTRUCTION_MAX_SELL]
    assert exact.at["Risk", TRADE_WEIGHT] == pytest.approx(-0.005818)
    assert exact.at["Risk", BASE_WEIGHT] == pytest.approx(0.0445)
    assert exact.at["Risk", MANDATORY_FUNDING] == pytest.approx(-0.005818)
    assert not exact[INSTRUCTION_WITHIN_TOLERANCE].any()

    retained[SELECTED_TRADE] = False
    group = GroupLowerUpperConstraints(
        group_loadings=pd.DataFrame({"Risk": [1.0, 0.0]}, index=ASSETS),
        group_min_allocation=pd.Series({"Risk": 0.0}),
        group_max_allocation=pd.Series({"Risk": 0.050318}),
    )
    lower, upper = _build_selected_execution_bound_series(
        Constraints(min_weights=pd.Series(0., index=ASSETS),
                    max_weights=pd.Series(1., index=ASSETS),
                    group_lower_upper_constraints=group), retained, False)
    assert lower.at["Risk"] == upper.at["Risk"] == pytest.approx(0.050318)



def test_fixed_target_deadband_pins_current_or_fully_corrects() -> None:
    """A fixed target is pinned inside the band and fully repaired outside it."""
    current = pd.Series([0.018642, 0.981358], index=ASSETS)
    reference = _direct_deadband_reference(current, 0.02, 0.02, 0.005)
    retained = build_execution_target(
        current_weights=current,
        model_weights=pd.Series([0.04, 0.96], index=ASSETS),
        config=ExecutionPolicyConfig(
            tre_weight_by_asset_class={k: 1.0 for k in ASSET_CLASSES},
            instruction_correction_tolerance=0.005,
        ),
        settlement_cash_asset="Cash",
        mandatory_trades=MandatoryTrades(hard_target={"Risk": 0.02}),
        instruction_audit=_instruction_audit("hard_target"),
    )

    assert retained.at["Risk", PINNED_INSTRUCTION]
    assert retained.at["Risk", INSTRUCTION_WITHIN_TOLERANCE]
    assert not retained.at["Risk", RULE4_TRADE]
    assert retained.at["Risk", POLICY_MIN_WEIGHT] == pytest.approx(
        reference.at["Risk", "executable_min"]
    )
    assert retained.at["Risk", POLICY_MAX_WEIGHT] == pytest.approx(
        reference.at["Risk", "executable_max"]
    )
    pd.testing.assert_series_equal(retained[BASE_WEIGHT], reference["base"], check_names=False)
    pd.testing.assert_series_equal(
        retained[MANDATORY_FUNDING],
        reference["mandatory_funding"],
        check_names=False,
    )

    corrected = build_execution_target(
        current_weights=current,
        model_weights=pd.Series([0.04, 0.96], index=ASSETS),
        config=ExecutionPolicyConfig(
            tre_weight_by_asset_class={k: 1.0 for k in ASSET_CLASSES},
            instruction_correction_tolerance=0.001,
        ),
        settlement_cash_asset="Cash",
        mandatory_trades=MandatoryTrades(hard_target={"Risk": 0.02}),
        instruction_audit=_instruction_audit("hard_target"),
    )
    assert not corrected.at["Risk", PINNED_INSTRUCTION]
    assert not corrected.at["Risk", INSTRUCTION_WITHIN_TOLERANCE]
    assert corrected.at["Risk", MANDATORY_TRADE]
    assert corrected.at["Risk", TRADE_WEIGHT] == pytest.approx(0.001358)
    assert corrected.at["Risk", MANDATORY_FUNDING] == pytest.approx(0.001358)


def test_retained_interval_breach_remains_rule4_selectable() -> None:
    """A selected retained cap breach may sell to model inside widened bounds."""
    target = _build_upper_bound_target(tolerance=0.005)
    target[SELECTED_TRADE] = [True, False]
    result = solve_ranked_execution_portfolio(
        config=RankedExecutionPolicyConfig(
            tre_weight_by_asset_class={k: 1. for k in ASSET_CLASSES},
            preparation_solver="SCIPY"),
        trade_table=target,
        base_constraints=Constraints(
            min_weights=pd.Series(0.0, index=ASSETS),
            max_weights=pd.Series(1.0, index=ASSETS),
        ),
        covariance=pd.DataFrame(np.eye(2) * 0.04, index=ASSETS, columns=ASSETS),
        context="instruction dead-band selected cap",
    )

    assert target.at["Risk", RULE4_TRADE]
    assert result.accepted and result.compliant
    assert result.weights.at["Risk"] == pytest.approx(0.0445, abs=2e-4)
    assert (
        target.at["Risk", POLICY_MIN_WEIGHT] - 1e-8
        <= result.weights.at["Risk"]
        <= target.at["Risk", POLICY_MAX_WEIGHT] + 1e-8
    )


def test_mandatory_maximum_repair_can_continue_selling_for_group_compliance() -> None:
    """An exact cap repair remains movable below its compulsory boundary."""
    target = _build_upper_bound_target(tolerance=0.0)
    target[SELECTED_TRADE] = False
    group = GroupLowerUpperConstraints(
        group_loadings=pd.DataFrame({"Risk": [1.0, 0.0]}, index=ASSETS),
        group_min_allocation=pd.Series({"Risk": 0.0}),
        group_max_allocation=pd.Series({"Risk": 0.047}),
    )
    result = solve_ranked_execution_portfolio(
        config=RankedExecutionPolicyConfig(
            tre_weight_by_asset_class={k: 1. for k in ASSET_CLASSES},
            preparation_solver="SCIPY"),
        trade_table=target,
        base_constraints=Constraints(
            min_weights=pd.Series(0.0, index=ASSETS),
            max_weights=pd.Series(1.0, index=ASSETS),
            group_lower_upper_constraints=group,
        ),
        covariance=pd.DataFrame(np.eye(2) * 0.04, index=ASSETS, columns=ASSETS),
        context="mandatory maximum corridor",
    )

    assert result.accepted and result.compliant
    assert result.weights.at["Risk"] == pytest.approx(0.0445, abs=2e-4)
    assert result.weights.at["Risk"] <= target.at["Risk", BASE_WEIGHT] + 1e-7
    assert result.weights.at["Risk"] <= .047 + 1e-7
    assert result.trade_table.attrs["execution"]["audit"]["passed"]


def test_current_min_deadband_widens_only_the_retention_side() -> None:
    """A retained floor breach permits buying but prohibits further selling."""
    current = pd.Series([0.0497, 0.9503], index=ASSETS)
    target = build_execution_target(
        current_weights=current,
        model_weights=pd.Series([0.06, 0.94], index=ASSETS),
        config=ExecutionPolicyConfig(
            tre_weight_by_asset_class={k: 1.0 for k in ASSET_CLASSES},
            instruction_correction_tolerance=0.005,
        ),
        settlement_cash_asset="Cash",
        mandatory_trades=MandatoryTrades(hard_buy_min={"Risk": 0.05}),
        instruction_audit=_instruction_audit("hard_buy_min"),
    )

    assert target.at["Risk", INSTRUCTION_WITHIN_TOLERANCE]
    assert target.at["Risk", POLICY_MIN_WEIGHT] == pytest.approx(0.0497)
    assert target.at["Risk", POLICY_MAX_WEIGHT] == pytest.approx(1.0)
    assert target.at["Risk", BASE_WEIGHT] == pytest.approx(0.0497)
    assert target.at["Risk", RULE4_TRADE]
    assert not target.at["Risk", MANDATORY_TRADE]


def test_deadband_never_weakens_hard_sell_product_or_lifecycle_repairs() -> None:
    """Non-desk correction classes remain exact even below the tolerance."""
    current = pd.Series([0.003, 0.997], index=ASSETS)
    config = ExecutionPolicyConfig(
        tre_weight_by_asset_class={k: 1.0 for k in ASSET_CLASSES},
        instruction_correction_tolerance=0.005,
    )

    hard_sell = build_execution_target(
        current_weights=current,
        model_weights=pd.Series([0.02, 0.98], index=ASSETS),
        config=config,
        settlement_cash_asset="Cash",
        mandatory_trades=MandatoryTrades(hard_sell=("Risk",)),
    )
    assert hard_sell.at["Risk", BASE_WEIGHT] == pytest.approx(0.0)
    assert hard_sell.at["Risk", MANDATORY_TRADE]
    assert not hard_sell.at["Risk", INSTRUCTION_WITHIN_TOLERANCE]

    product = build_execution_target(
        current_weights=pd.Series([0.050318, 0.949682], index=ASSETS),
        model_weights=pd.Series([0.0445, 0.9555], index=ASSETS),
        config=config,
        settlement_cash_asset="Cash",
        required_max_weights=pd.Series([0.05, 1.0], index=ASSETS),
    )
    assert product.at["Risk", BASE_WEIGHT] == pytest.approx(0.05)
    assert product.at["Risk", MANDATORY_TRADE]
    assert not product.at["Risk", INSTRUCTION_WITHIN_TOLERANCE]

    lifecycle = build_execution_target(
        current_weights=current,
        model_weights=pd.Series([0.0, 1.0], index=ASSETS),
        previous_model_weights=pd.Series([0.02, 0.98], index=ASSETS),
        config=config,
        settlement_cash_asset="Cash",
    )
    assert lifecycle.at["Risk", BASE_WEIGHT] == pytest.approx(0.0)
    assert lifecycle.at["Risk", MANDATORY_TRADE]
    assert not lifecycle.at["Risk", INSTRUCTION_WITHIN_TOLERANCE]


def test_exact_current_pin_grandfathers_product_breach_with_rounding_margin() -> None:
    """A desk pin rounded within 0.5 bp may retain an existing product breach."""
    current = pd.Series([0.023947, 0.976053], index=ASSETS)
    target = build_execution_target(
        current_weights=current,
        model_weights=pd.Series([0.0, 1.0], index=ASSETS),
        config=ExecutionPolicyConfig(
            tre_weight_by_asset_class={k: 1.0 for k in ASSET_CLASSES}, small_trade_cutoff=0.0
        ),
        settlement_cash_asset="Cash",
        required_min_weights=pd.Series([0.0, 0.0], index=ASSETS),
        required_max_weights=pd.Series([0.02, 1.0], index=ASSETS),
        mandatory_trades=MandatoryTrades(hard_target={"Risk": 0.02395}),
        instruction_audit=_instruction_audit("hard_target"),
    )

    assert target.at["Risk", PRODUCT_BOUND_OVERRIDE]
    assert target.at["Risk", PRODUCT_BOUND_OVERRIDE_BREACH] == pytest.approx(0.003947)
    assert target.at["Risk", POLICY_MIN_WEIGHT] == pytest.approx(current["Risk"])
    assert target.at["Risk", POLICY_MAX_WEIGHT] == pytest.approx(current["Risk"])
    assert target.at["Risk", BASE_WEIGHT] == pytest.approx(current["Risk"])
    assert target.at["Risk", PINNED_INSTRUCTION]
    assert not target.at["Risk", MANDATORY_TRADE]

    target[SELECTED_TRADE] = False
    result = solve_ranked_execution_portfolio(
        config=RankedExecutionPolicyConfig(
            tre_weight_by_asset_class={k: 1. for k in ASSET_CLASSES},
            preparation_solver="SCIPY"),
        trade_table=target,
        base_constraints=Constraints(
            min_weights=pd.Series([0.0, 0.0], index=ASSETS),
            max_weights=pd.Series([0.02, 1.0], index=ASSETS),
        ),
        covariance=pd.DataFrame(np.eye(2) * 0.04, index=ASSETS, columns=ASSETS),
        context="grandfathered exact current pin",
    )
    assert result.accepted and result.compliant
    assert result.weights.at["Risk"] == pytest.approx(current["Risk"])


def test_non_current_or_non_desk_pin_cannot_override_product_bounds() -> None:
    """Only a desk-owned target matching current within 0.5 bp is grandfathered."""
    current = pd.Series([0.023947, 0.976053], index=ASSETS)
    common = dict(
        current_weights=current,
        model_weights=pd.Series([0.0, 1.0], index=ASSETS),
        config=ExecutionPolicyConfig(
            tre_weight_by_asset_class={k: 1.0 for k in ASSET_CLASSES}, small_trade_cutoff=0.0
        ),
        settlement_cash_asset="Cash",
        required_min_weights=pd.Series([0.0, 0.0], index=ASSETS),
        required_max_weights=pd.Series([0.02, 1.0], index=ASSETS),
    )

    with pytest.raises(ValueError, match="do not intersect product bounds"):
        build_execution_target(
            **common,
            mandatory_trades=MandatoryTrades(hard_target={"Risk": 0.02380}),
            instruction_audit=_instruction_audit("hard_target"),
        )
    with pytest.raises(ValueError, match="do not intersect product bounds"):
        build_execution_target(
            **common,
            mandatory_trades=MandatoryTrades(hard_target={"Risk": current["Risk"]}),
        )
