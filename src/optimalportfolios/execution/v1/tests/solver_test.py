"""Cash-funded selected-trade execution optimisation tests."""

from optimalportfolios.optimization.config import OptimiserConfig as _PortableOptimiserConfig
import numpy as np
import pandas as pd
import pytest

from optimalportfolios import Constraints, GroupLowerUpperConstraints
from optimalportfolios.execution.solver import (
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
from optimalportfolios.execution.schema import (
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


def _base_constraints(group_max: float = 0.8) -> Constraints:
    """Construct hard asset/group bounds with inherited model utilities."""
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
    """Construct a resolved, self-financing three-asset legacy trade table."""
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
            trade_table=table,
        )

    covariance = pd.DataFrame(np.eye(3) * 0.04, index=ASSETS, columns=ASSETS)
    result = solve_selected_execution_portfolio(
        optimiser_config=_PortableOptimiserConfig(solver="CLARABEL"),
        trade_table=table,
        base_constraints=constraints,
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
            covariance=covariance,
            context="test Income cash bootstrap",
        )


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
    from optimalportfolios.execution import solver as optimiser

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
    from optimalportfolios.execution import solver as optimiser

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
    from optimalportfolios.execution.schema import (
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
    from optimalportfolios.execution import solver as optimiser
    from optimalportfolios.execution.schema import (
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
    from optimalportfolios.execution import solver as optimiser

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
    from optimalportfolios.execution.solver import (
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
    bridges = _compute_selected_execution_bridges(constraints, table)
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
            build_selected_execution_constraints(constraints, table)
    else:
        assert problem.status == "optimal"
        build_selected_execution_constraints(constraints, table)


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
    from optimalportfolios.execution.solver import (
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
        constraints, table, partition_groups=tuple(group.group_loadings.columns)
    )
    assert bridges.at["Liquidity", "e_min"] == pytest.approx(0.06)
    assert bridges.at["Liquidity", "bridge_max"] == pytest.approx(0.01)
    with pytest.raises(ExecutionSolverInfeasibility, match="Liquidity"):
        build_selected_execution_constraints(
            constraints, table, partition_groups=tuple(group.group_loadings.columns)
        )


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


@pytest.mark.parametrize(
    "case", ["cash_bounds", "cash_filtered", "invested_filtered", "hard_audit", "funding_identity"]
)
def test_accepted_solver_output_cannot_bypass_cash_funding_audit(monkeypatch, case):
    """Accepted numerical outputs still require funded cash and a complete hard audit."""
    from optimalportfolios.execution import solver as optimiser
    from optimalportfolios.optimization.constraints.analytics import ConstraintResidual
    from optimalportfolios.optimization.solver_diagnostics import OptimizationOutcome

    table = _trade_table([0.4, 0.2, 0.4], [0.3, 0.3, 0.4], [True, True, False])
    constraints = _base_constraints()
    weights = pd.Series([0.3, 0.3, 0.4000005], index=ASSETS)
    expected = {
        "cash_bounds": "breaches cash bounds",
        "cash_filtered": "requires cash in the solver universe",
        "invested_filtered": "leaves excluded assets invested",
        "hard_audit": "breaches hard constraints",
        "funding_identity": "cash funding identity failed",
    }
    if case == "cash_bounds":
        constraints = constraints.copy(min_weights=pd.Series([0.0, 0.0, 0.4], index=ASSETS))
        weights = pd.Series([0.3, 0.3000002, 0.4], index=ASSETS)
    elif case == "hard_audit":
        weights = pd.Series([0.2, 0.4, 0.4000005], index=ASSETS)

    def numerical_output(**kwargs):
        """Inject realistic sub-micro cash residuals or inconsistent filtered metadata."""
        solved = kwargs["constraints"]
        if case in ("cash_filtered", "invested_filtered"):
            retained = ASSETS.drop("Cash" if case == "cash_filtered" else "B")
            solved = Constraints(
                min_weights=solved.min_weights.loc[retained],
                max_weights=solved.max_weights.loc[retained],
            )
        residuals = ()
        if case == "funding_identity":
            residuals = (
                ConstraintResidual(
                    constraint_type="group_weight",
                    name="risk",
                    actual=0.9,
                    lower=0.0,
                    upper=0.8,
                    violation=0.1,
                    tolerance=1e-4,
                    hard=True,
                    passed=False,
                ),
            )
        return weights, OptimizationOutcome(
            weights=weights.reindex(solved.min_weights.index).to_numpy(),
            accepted=True,
            solver="CLARABEL",
            status="optimal",
            context="injected audit",
            constraints=solved,
            constraint_residuals=residuals,
        )

    monkeypatch.setattr(optimiser, "wrapper_minimise_tracking_error", numerical_output)
    covariance = pd.DataFrame(np.eye(3) * 0.04, index=ASSETS, columns=ASSETS)
    with pytest.raises(RuntimeError, match=expected[case]):
        solve_selected_execution_portfolio(table, constraints, covariance)


@pytest.mark.parametrize("barrier", ["group_minimum", "filtered_cash"])
def test_dust_cleanup_retains_weight_when_funding_or_group_slack_is_unavailable(
    monkeypatch, barrier
):
    """A dust position is retained when its funded deletion has no audited capacity."""
    from optimalportfolios.execution import solver as optimiser
    from optimalportfolios.execution.schema import (
        CUTOFF_SELL_DOWN,
        CUTOFF_SELL_DOWN_DUST_ROUNDED,
        SELL_DOWN_DUST_THRESHOLD,
    )
    from optimalportfolios.optimization.solver_diagnostics import OptimizationOutcome

    current = [0.5, 0.001, 0.499] if barrier == "group_minimum" else [0.999, 0.001, 0.0]
    table = _trade_table(current, current, [False, False, False])
    table[CUTOFF_SELL_DOWN] = [False, True, False]
    table[SELL_DOWN_DUST_THRESHOLD] = 0.0025
    group = None
    if barrier == "group_minimum":
        group = GroupLowerUpperConstraints(
            group_loadings=pd.DataFrame({"B minimum": [0.0, 1.0, 0.0]}, index=ASSETS),
            group_min_allocation=pd.Series({"B minimum": 0.001}),
            group_max_allocation=None,
        )
    constraints = Constraints(
        min_weights=pd.Series(0.0, index=ASSETS),
        max_weights=pd.Series(1.0, index=ASSETS),
        group_lower_upper_constraints=group,
    )

    def exact_base_output(**kwargs):
        """Return a funded base with an explicitly filtered zero-weight cash row."""
        solved = kwargs["constraints"]
        if barrier == "filtered_cash":
            retained = ASSETS.drop("Cash")
            solved = Constraints(
                min_weights=solved.min_weights.loc[retained],
                max_weights=solved.max_weights.loc[retained],
            )
        weights = table[CURRENT_WEIGHT].copy()
        return weights, OptimizationOutcome(
            weights=weights.reindex(solved.min_weights.index).to_numpy(),
            accepted=True,
            solver="CLARABEL",
            status="optimal",
            context="dust guard",
            constraints=solved,
        )

    monkeypatch.setattr(optimiser, "wrapper_minimise_tracking_error", exact_base_output)
    covariance = pd.DataFrame(np.eye(3) * 0.04, index=ASSETS, columns=ASSETS)
    result = solve_selected_execution_portfolio(table, constraints, covariance)
    assert result.weights.at["B"] == pytest.approx(0.001)
    assert not result.trade_table.at["B", CUTOFF_SELL_DOWN_DUST_ROUNDED]
    assert result.weights.sum() == pytest.approx(1.0)


@pytest.mark.parametrize(
    "status,slack_count", [("user_limit", 0), ("infeasible", 0), ("solver_error", 14)]
)
def test_terminal_diagnostic_preserves_status_and_limits_slack_report(
    monkeypatch, status, slack_count
):
    """Only terminal infeasibility gets advisory slacks; limits retain their original status."""
    from optimalportfolios.execution import solver as optimiser
    from optimalportfolios.optimization.solver_diagnostics import OptimizationOutcome

    table = _trade_table([0.4, 0.2, 0.4], [0.3, 0.3, 0.4], [True, True, False])
    constraints = _base_constraints()
    calls = []

    def rejected(**kwargs):
        """Supply one failed terminal outcome without touching weights or hard bounds."""
        return optimiser.ExecutionOptimizationResult(
            weights=table[CURRENT_WEIGHT],
            trade_table=kwargs["trade_table"],
            outcome=OptimizationOutcome(
                weights=table[CURRENT_WEIGHT].to_numpy(),
                accepted=False,
                solver="CLARABEL",
                status=status,
                context="terminal",
                reason="original",
                constraints=constraints,
            ),
        )

    def diagnose(**kwargs):
        """Record diagnostic calls and return distinguishable ordered slack magnitudes."""
        calls.append(kwargs)
        return {f"bound_{i:02d}": (i + 1) / 10000.0 for i in range(slack_count)}

    monkeypatch.setattr(optimiser, "solve_selected_execution_portfolio", rejected)
    monkeypatch.setattr(optimiser, "diagnose_infeasibility", diagnose)
    covariance = pd.DataFrame(np.eye(3) * 0.04, index=ASSETS, columns=ASSETS)
    result = solve_feasible_execution_portfolio(
        table,
        constraints,
        covariance,
        optimiser_config=_PortableOptimiserConfig(solver="CLARABEL", diagnose_infeasibility=True),
    )
    assert result.outcome.status == status
    pd.testing.assert_series_equal(result.weights, table[CURRENT_WEIGHT])
    if status == "user_limit":
        assert not calls
        assert result.outcome.reason == "original"
    elif slack_count:
        assert len(calls) == 1
        assert "2 additional bounds" in result.outcome.reason
        assert "bound_13=14.0000 bp" in result.outcome.reason
        assert "bound_00=" not in result.outcome.reason
    else:
        assert len(calls) == 1
        assert "cause remains unresolved" in result.outcome.reason


@pytest.mark.parametrize(
    "states,expand,relax,selected,expected_index,expected_error",
    [
        (["structural", "accepted"], False, True, [True, False, False], 1, None),
        (["failed", "failed"], False, True, [True, False, False], 1, None),
        (["structural", "structural"], False, True, [True, False, False], None, 0),
        (["structural"], True, False, [True, True, False], None, 0),
        (["structural"], False, False, [True, False, False], None, 0),
        (["failed", "failed", "failed", "accepted"], True, True, [True, False, False], 3, None),
        (
            ["failed", "structural", "structural", "structural"],
            True,
            True,
            [True, False, False],
            None,
            3,
        ),
    ],
)
def test_rescue_and_relaxation_preserve_attempt_order_and_terminal_evidence(
    monkeypatch,
    states,
    expand,
    relax,
    selected,
    expected_index,
    expected_error,
):
    """Strict and relaxed retries retain the last actual attempt and original failure evidence."""
    from optimalportfolios.execution import solver as optimiser
    from optimalportfolios.optimization.solver_diagnostics import OptimizationOutcome

    table = _trade_table([0.4, 0.2, 0.4], [0.3, 0.3, 0.4], selected)
    constraints = _base_constraints()
    calls = []
    errors = {}

    def scripted_attempt(**kwargs):
        """Return the prescribed numerical status for an independently recorded selection."""
        position = len(calls)
        calls.append((kwargs["trade_table"].copy(), kwargs.get("relax_selected_corridors", False)))
        state = states[position]
        if state == "structural":
            error = optimiser.ExecutionSolverInfeasibility(f"structural attempt {position}")
            errors[position] = error
            raise error
        return optimiser.ExecutionOptimizationResult(
            weights=table[CURRENT_WEIGHT],
            trade_table=kwargs["trade_table"].copy(),
            outcome=OptimizationOutcome(
                weights=table[CURRENT_WEIGHT].to_numpy(),
                accepted=state == "accepted",
                solver="CLARABEL",
                status="optimal" if state == "accepted" else "user_limit",
                context=str(position),
                reason=f"attempt {position}",
                constraints=constraints,
            ),
        )

    monkeypatch.setattr(optimiser, "solve_selected_execution_portfolio", scripted_attempt)
    covariance = pd.DataFrame(np.eye(3) * 0.04, index=ASSETS, columns=ASSETS)
    kwargs = dict(
        trade_table=table,
        base_constraints=constraints,
        covariance=covariance,
        expand_for_feasibility=expand,
        sign_directed_rescue=True,
        allow_corridor_relaxation=relax,
        optimiser_config=_PortableOptimiserConfig(solver="CLARABEL"),
    )
    if expected_error is not None:
        with pytest.raises(optimiser.ExecutionSolverInfeasibility) as captured:
            solve_feasible_execution_portfolio(**kwargs)
        assert captured.value is errors[expected_error]
    else:
        result = solve_feasible_execution_portfolio(**kwargs)
        assert result.outcome.reason == f"attempt {expected_index}"
        pd.testing.assert_frame_equal(result.trade_table, calls[expected_index][0])
    assert len(calls) == len(states)
    assert not calls[0][1]
    if len(states) == 4:
        assert [flag for _, flag in calls] == [False, False, True, True]
        assert [bool(frame.at["B", FEASIBILITY_RESCUE_TRADE]) for frame, _ in calls] == [
            False,
            True,
            False,
            True,
        ]


def test_bridge_rescue_admits_all_required_capacity_before_numerical_retry(monkeypatch):
    """Two compulsory funding directions are admitted before retrying the hard solve."""
    from optimalportfolios.execution import solver as optimiser

    table = _trade_table([0.5, 0.3, 0.2], [0.4, 0.2, 0.4], [False, False, False])
    calls = []
    original = optimiser.solve_selected_execution_portfolio

    def record_attempt(**kwargs):
        """Observe actual selected sets while keeping the independent numerical solve."""
        calls.append(kwargs["trade_table"][SELECTED_TRADE].copy())
        return original(**kwargs)

    monkeypatch.setattr(optimiser, "solve_selected_execution_portfolio", record_attempt)
    covariance = pd.DataFrame(np.eye(3) * 0.04, index=ASSETS, columns=ASSETS)
    result = solve_feasible_execution_portfolio(
        table,
        _base_constraints(group_max=0.6),
        covariance,
        expand_for_feasibility=True,
        optimiser_config=_PortableOptimiserConfig(solver="CLARABEL"),
    )
    assert result.accepted and result.compliant
    assert len(calls) == 2
    assert calls[0].tolist() == [False, False, False]
    assert calls[1].tolist() == [True, True, False]
    assert result.trade_table[FEASIBILITY_RESCUE_TRADE].tolist() == [True, True, False]
    np.testing.assert_allclose(result.weights, [0.4, 0.2, 0.4], atol=1e-6)


@pytest.mark.parametrize("bound", ["minimum", "maximum"])
def test_cash_bridge_rescue_uses_the_opposite_noncash_funding_direction(bound):
    """A breached cash floor requires a sell; a breached cash cap requires a buy."""
    current = [0.4, 0.3, 0.3] if bound == "minimum" else [0.3, 0.3, 0.4]
    model = [0.3, 0.3, 0.4] if bound == "minimum" else [0.4, 0.3, 0.3]
    table = _trade_table(current, model, [False, False, False])
    constraints = Constraints(
        min_weights=pd.Series([0.0, 0.0, 0.4 if bound == "minimum" else 0.0], index=ASSETS),
        max_weights=pd.Series([1.0, 1.0, 0.3 if bound == "maximum" else 1.0], index=ASSETS),
    )
    covariance = pd.DataFrame(np.eye(3) * 0.04, index=ASSETS, columns=ASSETS)
    result = solve_feasible_execution_portfolio(
        table,
        constraints,
        covariance,
        expand_for_feasibility=True,
        optimiser_config=_PortableOptimiserConfig(solver="CLARABEL"),
    )
    assert result.accepted and result.compliant
    assert result.trade_table.at["A", FEASIBILITY_RESCUE_TRADE]
    assert result.trade_table.at["A", "rescue_reason"] == (
        "cash:min" if bound == "minimum" else "cash:max"
    )
    np.testing.assert_allclose(result.weights, model, atol=1e-6)


def test_group_minimum_budget_bridge_requires_an_outside_funding_sale():
    """An eligible outside sale supplies funding for an already selected group purchase."""
    table = _trade_table([0.5, 0.45, 0.05], [0.6, 0.35, 0.05], [True, False, False])
    constraints = Constraints(
        min_weights=pd.Series([0.0, 0.0, 0.05], index=ASSETS),
        max_weights=pd.Series(1.0, index=ASSETS),
        group_lower_upper_constraints=GroupLowerUpperConstraints(
            group_loadings=pd.DataFrame({"A minimum": [1.0, 0.0, 0.0]}, index=ASSETS),
            group_min_allocation=pd.Series({"A minimum": 0.6}),
            group_max_allocation=None,
        ),
    )
    covariance = pd.DataFrame(np.eye(3) * 0.04, index=ASSETS, columns=ASSETS)
    result = solve_feasible_execution_portfolio(
        table,
        constraints,
        covariance,
        expand_for_feasibility=True,
        optimiser_config=_PortableOptimiserConfig(solver="CLARABEL"),
    )
    assert result.accepted and result.compliant
    assert result.trade_table.at["B", FEASIBILITY_RESCUE_TRADE]
    assert "A minimum:min" in result.trade_table.at["B", "rescue_reason"]
    np.testing.assert_allclose(result.weights, [0.6, 0.35, 0.05], atol=1e-6)


def test_unknown_bridge_row_cannot_authorize_a_rescue_trade():
    """A diagnostic row with no corresponding constraint has no trade direction."""
    from optimalportfolios.execution.solver import _get_constraint_directed_rescue_candidates

    table = _trade_table([0.4, 0.2, 0.4], [0.3, 0.3, 0.4], [False, False, False])
    bridges = pd.DataFrame(
        {
            "e_min": [0.0],
            "e_max": [0.1],
            "group_min": [0.2],
            "group_max": [1.0],
            "bridge_min": [0.1],
            "bridge_max": [0.0],
        },
        index=["unknown group"],
    )
    admissible, reasons = _get_constraint_directed_rescue_candidates(
        table,
        _base_constraints(),
        table[RULE4_TRADE],
        bridges,
    )
    assert not admissible.any()
    assert reasons.eq("").all()


def test_exhausted_bridge_rescue_returns_structural_capacity_evidence():
    """A cadence-ineligible funding source cannot be silently admitted to clear a cap."""
    from optimalportfolios.execution.solver import ExecutionSolverInfeasibility

    table = _trade_table([0.5, 0.3, 0.2], [0.4, 0.2, 0.4], [False, False, False])
    table[REBALANCE_CADENCE_ELIGIBLE] = False
    covariance = pd.DataFrame(np.eye(3) * 0.04, index=ASSETS, columns=ASSETS)
    with pytest.raises(ExecutionSolverInfeasibility) as captured:
        solve_feasible_execution_portfolio(
            table,
            _base_constraints(group_max=0.6),
            covariance,
            expand_for_feasibility=True,
            optimiser_config=_PortableOptimiserConfig(solver="CLARABEL"),
        )
    assert captured.value.bridges.at["Risk Assets", "bridge_max"] == pytest.approx(0.2)
    assert not captured.value.trade_table[SELECTED_TRADE].any()
    assert not captured.value.trade_table[FEASIBILITY_RESCUE_TRADE].any()


def test_relaxed_rescue_admits_sales_against_original_model_direction():
    """Full hard intervals permit a necessary sale even when both model changes are buys."""
    from optimalportfolios.execution.solver import ExecutionSolverInfeasibility

    table = _trade_table([0.4, 0.3, 0.3], [0.45, 0.35, 0.2], [False, False, False])
    covariance = pd.DataFrame(np.eye(3) * 0.04, index=ASSETS, columns=ASSETS)
    before = table.copy(deep=True)
    kwargs = dict(trade_table=table, base_constraints=_base_constraints(group_max=0.5),
                  covariance=covariance, expand_for_feasibility=True,
                  optimiser_config=_PortableOptimiserConfig(solver="CLARABEL"))
    with pytest.raises(ExecutionSolverInfeasibility):
        solve_feasible_execution_portfolio(**kwargs, allow_corridor_relaxation=False)
    result = solve_feasible_execution_portfolio(**kwargs, allow_corridor_relaxation=True)
    assert result.accepted and result.compliant
    assert result.trade_table[RELAXED_CORRIDOR_SOLVE].all()
    assert result.weights.loc[["A", "B"]].sum() <= 0.5+1e-6
    assert (result.weights.loc[["A", "B"]] < table.loc[["A", "B"], BASE_WEIGHT]-1e-8).any()
    assert result.trade_table[FEASIBILITY_RESCUE_TRADE].any()
    pd.testing.assert_frame_equal(before, table, check_exact=True)
