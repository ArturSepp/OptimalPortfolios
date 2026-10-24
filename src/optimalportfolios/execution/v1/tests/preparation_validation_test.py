"""Validation, explicit instruction and same-date retry contracts."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from optimalportfolios.execution.v1 import policy as p
from optimalportfolios.execution.v1 import orchestration as o
from optimalportfolios.execution.v1.solver import ExecutionOptimizationResult
from optimalportfolios.execution._common.config import ExactExecutionConfig


def config(**kwargs):
    """Provide one consumer-defined class without a library taxonomy default."""
    return p.ExecutionPolicyConfig(
        **({"tre_weight_by_asset_class": {"Risk": 1.0}, "max_trades": 3} | kwargs)
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"small_trade_cutoff": -1},
        {"small_trade_cutoff_by_asset_class": {"bad": 0.1}},
        {"small_trade_cutoff_by_asset_class": {"Risk": np.nan}},
        {"sell_down_dust_threshold": -1},
        {"material_trade_threshold": 0},
        {"instruction_correction_tolerance": -1},
        {"max_trades": 0},
        {"add_mandatory_trades_to_max_trades": 1},
        {"minimum_trade_score": np.nan},
        {"tre_weight_by_asset_class": {}},
        {"tre_weight_by_asset_class": {"Risk": -1}},
    ],
)
def test_invalid_ranked_controls_are_rejected(kwargs):
    """Invalid controls cannot silently weaken selection and instruction rules."""
    with pytest.raises(ValueError):
        config(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"solver": None},
        {"solver": "CLARABEL"},
        {"max_positions": 0},
        {"alpha_tradeoff": np.nan},
        {"relative_gap_tolerance": -1},
        {"absolute_gap_tolerance": -1},
        {"max_branches": False},
        {"max_solve_seconds": 0},
        {"accept_audited_incumbent": 1},
        {"seed": -1},
        {"sub_asset_class_penalty": 0},
        {"constraint_set": "unknown"},
    ],
)
def test_invalid_exact_controls_are_rejected(kwargs):
    """Reject invalid native solver controls before constructing an MIQP."""
    with pytest.raises(ValueError):
        ExactExecutionConfig(**kwargs)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_trades": None},
        {"max_trades": 1.5},
        {"expand_for_feasibility": True},
        {"minimum_trade_score": 0.1},
        {"sequential_greedy_selection": True},
        {"sign_directed_rescue": True},
        {"allow_corridor_relaxation": True},
    ],
)
def test_exact_rejects_incompatible_ranked_options(kwargs):
    """An exact policy cannot silently accept ranked-only controls."""
    with pytest.raises(ValueError):
        config(exact_execution=ExactExecutionConfig(), **kwargs)


def test_method_labels_and_explicit_class_defaults():
    """Public variants identify their distinct execution contracts."""
    with pytest.raises(ValueError, match="at least one class"):
        p.ExecutionPolicyConfig()
    assert config().execution_method == "ranked"
    assert config(exact_execution=ExactExecutionConfig()).execution_method == "exact"
    assert (
        config(exact_execution=ExactExecutionConfig(sub_asset_class_penalty=1)).execution_method
        == "exact_soft"
    )
    assert (
        p.RankedExecutionPolicyConfig(tre_weight_by_asset_class={"Risk": 1}).execution_method
        == "ranked"
    )
    with pytest.raises(ValueError, match="constraint_set"):
        p.RankedExecutionPolicyConfig(tre_weight_by_asset_class={"Risk": 1}, constraint_set="bad")


def risk_config(**kwargs):
    """Provide a complete single-class target-band policy."""
    defaults = dict(
        max_trades=3,
        tre_weight_by_asset_class={"Risk": 1},
        execution_tre_budget=0.01,
        target_band_tight_by_asset_class={"Risk": 0.01},
        target_band_wide_by_asset_class={"Risk": 0.02},
    )
    return p.RiskBudgetRankedExecutionPolicyConfig(**(defaults | kwargs))


@pytest.mark.parametrize(
    "kwargs",
    [
        {"instruction_correction_tolerance": 0.01},
        {"sell_down_dust_threshold": 0.01},
        {"exact_execution": ExactExecutionConfig()},
        {"execution_tre_budget": 0},
        {"target_band_tight_by_asset_class": {}},
        {"target_band_wide_by_asset_class": {"Risk": np.nan}},
        {"target_band_tight_by_asset_class": {"Risk": 0.03}},
        {"constraint_set": "bad"},
    ],
)
def test_invalid_target_band_controls_are_rejected(kwargs):
    """A risk budget requires complete, ordered, nonnegative class bands."""
    with pytest.raises(ValueError):
        risk_config(**kwargs)
    assert risk_config().execution_method == "risk_budget_ranked"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"hard_sell": ("A", "A")},
        {"hard_sell": ("A",), "hard_buy_min": {"A": 0.1}},
        {"hard_target": {"A": 0.1}, "hard_max": {"A": 0.2}},
        {"hard_buy_min": {"A": np.nan}},
        {"hard_buy_min": {"A": 0.2}, "hard_max": {"A": 0.1}},
    ],
)
def test_conflicting_instructions_fail_closed(kwargs):
    """Instructions with ambiguous precedence are never combined silently."""
    with pytest.raises(ValueError):
        p.MandatoryTrades(**kwargs)


def test_merge_instructions_keeps_the_stricter_interval():
    """Merging is order-independent for bounds and rejects conflicting targets."""
    a = p.MandatoryTrades(
        hard_sell=("Z",), hard_buy_min={"A": 0.1}, hard_max={"B": 0.4}, hard_target={"C": 0.1}
    )
    b = p.MandatoryTrades(
        hard_sell=("Z",), hard_buy_min={"A": 0.2}, hard_max={"B": 0.3}, hard_target={"C": 0.1}
    )
    merged = p.merge_mandatory_trades(a, b)
    assert merged == p.merge_mandatory_trades(b, a)
    assert merged.hard_buy_min == {"A": 0.2} and merged.hard_max == {"B": 0.3}
    assert merged.hard_sell == ("Z",)
    with pytest.raises(ValueError, match="conflicting hard targets"):
        p.merge_mandatory_trades(a, p.MandatoryTrades(hard_target={"C": 0.2}))


def target(**kwargs):
    """Prepare a funded two-asset decision with explicit generic configuration."""
    values = pd.Series([0.4, 0.6], index=["A", "Cash"])
    return p.build_execution_target(
        **(dict(current_weights=values, model_weights=values, config=config()) | kwargs)
    )


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"current_weights": pd.Series([0.5, 0.5], index=["A", "A"])}, "unique"),
        ({"current_weights": pd.Series([np.inf, 0.5], index=["A", "Cash"])}, "finite"),
        ({"settlement_cash_asset": "missing"}, "not in the universe"),
        ({"config": config(small_trade_cutoff_by_asset_class={"Risk": 0.1})}, "asset_classes"),
        ({"asset_classes": pd.Series({"A": "Risk"})}, "missing asset classes"),
        ({"asset_classes": pd.Series({"A": "Unknown", "Cash": "Risk"})}, "unsupported"),
        ({"required_min_weights": pd.Series({"A": np.inf})}, "finite"),
        (
            {
                "required_min_weights": pd.Series({"A": 0.8}),
                "required_max_weights": pd.Series({"A": 0.5}),
            },
            "Min exceeds Max",
        ),
        ({"mandatory_trades": p.MandatoryTrades(hard_sell=("missing",))}, "unknown instruments"),
    ],
)
def test_invalid_prepared_inputs_are_rejected(kwargs, match):
    """Reject malformed supplied inputs before creating a policy trade table."""
    with pytest.raises(ValueError, match=match):
        target(**kwargs)


def test_calendar_and_drift_validation(monkeypatch):
    """Invalid schedules and absent holdings fail before any execution sizing."""
    dates = pd.date_range("2024-01-31", periods=2, freq="ME")
    for bad in (pd.DatetimeIndex([]), dates[::-1], dates[[0, 0]]):
        with pytest.raises(ValueError, match="decision_dates"):
            p.build_instrument_rebalancing_indicators(bad, pd.Series({"A": "ME"}))
    for freq in (pd.Series(dtype=str), pd.Series(["ME", "ME"], index=["A", "A"])):
        with pytest.raises(ValueError, match="unique index"):
            p.build_instrument_rebalancing_indicators(dates, freq)
    with pytest.raises(ValueError, match="missing instrument"):
        p.build_instrument_rebalancing_indicators(dates, pd.Series({"A": None}))
    monkeypatch.setattr(p, "apply_drift_to_weights_0", lambda **kwargs: None)
    with pytest.raises(ValueError, match="previous_target"):
        p.drift_execution_weights(None, pd.DataFrame(), None, dates[0])


def result(table, accepted=True):
    """Supply an audited callback result independently of a numerical solver."""
    return ExecutionOptimizationResult(
        pd.Series([1.0], index=table.index),
        table,
        SimpleNamespace(
            accepted=accepted,
            compliant=accepted,
            status="optimal" if accepted else "infeasible",
            reason="test",
        ),
    )


@pytest.mark.parametrize(
    "failure", ["none", "returned", "raised", "bridges", "cash_only_bridge", "no_table"]
)
def test_retry_preserves_inputs_and_records_fallback(failure):
    """Retry only the group preparation, retaining the exact same holdings and constraints."""
    attempts = []
    original = pd.DataFrame({"cutoff_residual_bridge": [0.02]}, index=["A"])

    def build(enabled):
        """Record which preparation policy the orchestration requested."""
        attempts.append(enabled)
        if failure == "no_table" and enabled:
            raise ValueError("no target")
        return original.copy()

    def solve(table):
        """Exercise accepted, rejected and exception-returning solvers."""
        if attempts[-1] and failure in ("raised", "bridges", "cash_only_bridge"):
            error = RuntimeError("target failure")
            if failure in ("bridges", "cash_only_bridge"):
                error.trade_table = table
                error.bridges = pd.DataFrame(
                    {"bridge_min": [0.03], "bridge_max": [0.01]},
                    index=["cash" if failure == "cash_only_bridge" else "Risk"],
                )
            raise error
        return result(table, not (attempts[-1] and failure == "returned"))

    got = o.solve_with_group_retry(build, solve)
    assert got.accepted and got.compliant
    assert attempts == ([True] if failure == "none" else [True, False])
    assert bool(got.trade_table[o.GROUP_MECHANISM_ABANDONED].iloc[0]) == (failure != "none")
    expected = 0 if failure in ("none", "no_table") else 0.03 if failure == "bridges" else 0.02
    assert got.trade_table[o.GROUP_MECHANISM_RESIDUAL_BRIDGE].iloc[0] == expected
    assert original.columns.tolist() == ["cutoff_residual_bridge"]


def test_both_attempts_failed_preserve_last_diagnostics():
    """A second failure must retain both attempts and its final table and bridges."""
    table = pd.DataFrame({"x": [1]}, index=["A"])

    def fail(table):
        """Expose structural evidence from an unsuccessful callback."""
        e = RuntimeError("hard floor")
        e.trade_table = table
        e.bridges = pd.DataFrame({"bridge_min": [0.1], "bridge_max": [0.0]}, index=["Risk"])
        raise e

    with pytest.raises(o.ExecutionGroupFallbackFailure) as caught:
        o.solve_with_group_retry(lambda enabled: table.copy(), fail)
    assert caught.value.enabled_diagnostic == caught.value.disabled_diagnostic
    pd.testing.assert_frame_equal(caught.value.trade_table, table)
    assert caught.value.bridges.loc["Risk", "bridge_min"] == 0.1
    assert o.ExecutionGroupFallbackFailure("a", "b").bridges.empty
