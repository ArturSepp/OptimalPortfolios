"""Audited recovery refusal and target-band rescue failure contracts."""

from types import SimpleNamespace
import numpy as np
import pandas as pd
import pytest
from optimalportfolios import Constraints, OptimiserConfig, OptimizationOutcome
from optimalportfolios.execution.v1 import solver as s
from optimalportfolios.execution.v1.tests.production_solver_test import (
    _trade_table,
    _base_constraints,
    ASSETS,
)
from optimalportfolios.execution.v1.tests.risk_budget_contract_test import (
    _trade_table as band_table,
)
from optimalportfolios.execution.v1.tests.preparation_validation_test import target, config
from optimalportfolios.execution.v1.policy import MandatoryTrades, build_execution_target


def test_paired_instructions_and_funding_refusal():
    """Explicit ranges preserve both endpoints and require a real funding leg."""
    table = target(mandatory_trades=MandatoryTrades(hard_buy_min={"A": 0.3}, hard_max={"A": 0.5}))
    assert (
        table.loc["A", "instruction_min_weight"] == 0.3
        and table.loc["A", "instruction_max_weight"] == 0.5
    )
    with pytest.raises(ValueError, match="settlement-cash"):
        build_execution_target(
            pd.Series([0.4, 0.6], index=["A", "B"]),
            pd.Series([0.0, 1.0], index=["A", "B"]),
            config=config(),
        )
    with pytest.raises(RuntimeError, match="effective execution target"):
        build_execution_target(
            pd.Series([0.4, 0.6], index=["A", "B"]),
            pd.Series([0.3, 0.3], index=["A", "B"]),
            config=config(),
        )
    with pytest.raises(RuntimeError, match="post-mandatory base"):
        build_execution_target(
            pd.Series([1e16, 1.0, -1e16], index=["A", "B", "Cash"]),
            pd.Series([0.0, 0.5, 0.5], index=["A", "B", "Cash"]),
            config=config(),
        )


@pytest.mark.parametrize("case", ["cash", "model", "movable", "different_target", "cash_residual"])
def test_cash_recovery_refuses_unproved_candidates(case):
    """Only the unique, funded cash-start model is eligible for algebraic recovery."""
    table = _trade_table(
        [0, 0, 1],
        [0.4, 0.3, 0.3],
        [False, False, False],
        base=[0.4, 0.3, 0.3],
        mandatory=[True, True, False],
    )
    base = _base_constraints()
    selected = Constraints(
        min_weights=pd.Series([0.4, 0.3, 0.0], index=ASSETS),
        max_weights=pd.Series([0.4, 0.3, 1.0], index=ASSETS),
    )
    if case == "cash":
        table[s.SETTLEMENT_CASH] = False
    elif case == "model":
        table.loc["A", s.RAW_MODEL_WEIGHT] = np.inf
    elif case == "movable":
        selected = selected.copy(max_weights=pd.Series([0.5, 0.3, 1.0], index=ASSETS))
    elif case == "different_target":
        table.loc["A", s.RAW_MODEL_WEIGHT] = 0.41
    else:
        table.loc["Cash", s.RAW_MODEL_WEIGHT] = 0.31
    assert (
        s._audit_exact_cash_bootstrap_model(
            table, base, selected, pd.DataFrame(np.eye(3), index=ASSETS, columns=ASSETS), "test"
        )
        is None
    )


@pytest.mark.parametrize("case", ["index", "missing", "asymmetry", "solver"])
def test_group_split_rejects_bad_inputs_and_backend_failure(case):
    """Group split sizing preserves validation and native failure outcomes."""
    covar = pd.DataFrame(np.eye(3), index=ASSETS, columns=ASSETS)
    groups = pd.Series(["Risk", "Risk", "Cash"], index=ASSETS)
    if case == "index":
        covar.columns = ["x", "B", "Cash"]
    elif case == "missing":
        groups = groups.drop(index="A")
    elif case == "asymmetry":
        covar.iloc[0, 1] = 0.1
    args = dict(
        covariance=covar,
        benchmark_weights=pd.Series([0.4, 0.3, 0.3], index=ASSETS),
        current_weights=pd.Series([0.4, 0.3, 0.3], index=ASSETS),
        asset_classes=groups,
        constraints=_base_constraints(),
        optimiser_config=OptimiserConfig(solver="MISSING_BACKEND"),
        context="test",
    )
    if case == "solver":
        _, outcome = s._solve_group_split_tracking_error(**args)
        assert not outcome.accepted and outcome.status == "solver_error"
    else:
        with pytest.raises(ValueError):
            s._solve_group_split_tracking_error(**args)


def test_band_direction_and_group_tre_guards():
    """No omitted, nonfinite or already satisfied group budget admits a rescue."""
    table, covar, groups = band_table()
    base = Constraints(
        min_weights=pd.Series(0.0, index=table.index), max_weights=pd.Series(1.0, index=table.index)
    )
    with pytest.raises(ValueError, match="never relaxes target bands"):
        s._build_selected_execution_bound_series(base, table, True)
    directed, _ = s._get_constraint_directed_rescue_candidates(
        table,
        base,
        pd.Series(True, index=table.index),
        pd.DataFrame(columns=["bridge_min", "bridge_max", "group_min", "group_max"]),
    )
    assert not directed.any()
    eligible = pd.Series(True, index=table.index)
    assert not s._get_group_tre_rescue_candidates(table, base, covar, eligible)[0].any()
    loadings = pd.DataFrame(1.0, index=table.index, columns=["omitted", "nan", "satisfied"])
    limits = pd.Series({"nan": np.nan, "satisfied": 100.0})
    base = base.copy(
        group_tracking_error_constraint=SimpleNamespace(
            group_loadings=loadings, group_tre_vols=limits
        )
    )
    assert not s._get_group_tre_rescue_candidates(table, base, covar, eligible)[0].any()
    assert not s._is_proven_infeasible(None)
    assert not s._is_proven_infeasible(SimpleNamespace(outcome=SimpleNamespace(status=None)))


@pytest.mark.parametrize(
    "first,second",
    [
        ("raise", "unused"),
        ("infeasible", "none"),
        ("infeasible", "solver_error"),
        ("infeasible", "infeasible"),
    ],
)
def test_group_rescue_requires_proved_infeasibility(monkeypatch, first, second):
    """Rescue never treats an unknown numerical failure as a hard-budget proof."""
    table, covar, groups = band_table()
    base = Constraints(
        min_weights=pd.Series(0.0, index=table.index), max_weights=pd.Series(1.0, index=table.index)
    )
    table[s.SELECTED_TRADE] = False
    calls = []

    def solve(**kwargs):
        """Model explicit solver statuses after a bridge-free initial rejection."""
        calls.append(kwargs["trade_table"])
        status = first if len(calls) == 1 else second
        if status == "raise":
            raise s.ExecutionSolverInfeasibility("structural", trade_table=kwargs["trade_table"])
        return s.ExecutionOptimizationResult(
            table[s.BASE_WEIGHT],
            kwargs["trade_table"],
            OptimizationOutcome(
                weights=table[s.BASE_WEIGHT].to_numpy(),
                accepted=False,
                status=status,
                solver="CLARABEL",
                reason="test",
                context="test",
            ),
        )

    monkeypatch.setattr(s, "solve_selected_execution_portfolio", solve)
    monkeypatch.setattr(
        s,
        "_compute_selected_execution_bridges",
        lambda *args, **kwargs: pd.DataFrame(columns=["bridge_min", "bridge_max"]),
    )
    if second != "none":

        def candidates(**kwargs):
            """Supply the next independently eligible group risk candidate."""
            return kwargs["rescue_eligible"], pd.Series("group_tre:test", index=table.index)

        monkeypatch.setattr(s, "_get_group_tre_rescue_candidates", candidates)
    try:
        result = s.solve_feasible_execution_portfolio(
            table,
            base,
            covar,
            optimiser_config=OptimiserConfig(solver="CLARABEL", diagnose_infeasibility=False),
            expand_for_feasibility=True,
            group_split_asset_classes=groups,
        )
        assert not result.accepted
    except s.ExecutionSolverInfeasibility as error:
        assert "rescue" in str(error)
    assert calls
