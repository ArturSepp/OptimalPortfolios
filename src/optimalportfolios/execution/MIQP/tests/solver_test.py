"""Exact Execution configuration, formulation and numerical-reference tests."""

import logging
import sys
from types import SimpleNamespace
from dataclasses import FrozenInstanceError, replace
from unittest.mock import Mock
import warnings

import cvxpy as cvx
import numpy as np
import pandas as pd
import pytest

from optimalportfolios import Constraints, GroupLowerUpperConstraints
import optimalportfolios.execution.MIQP.solver as exact_optimiser
from optimalportfolios.execution.MIQP.solver import (
    EXACT_EXECUTION,
    EXACT_HISTORY_CANDIDATE,
    EXACT_INCEPTION_CANDIDATE,
    MIP_RELATIVE_GAP,
    MIP_TERMINATION,
    ExactExecutionSolverAudit,
    solve_exact_execution_portfolio,
)
from optimalportfolios.execution.v1.solver import (
    DIRECTED_RESCUE_CANDIDATE,
    EXECUTED_TRADE,
)
from optimalportfolios.execution.v1.policy import (
    BASE_WEIGHT,
    CURRENT_WEIGHT,
    EFFECTIVE_SMALL_TRADE_CUTOFF,
    FIXED_INSTRUCTION,
    MANDATORY_CATEGORY,
    MANDATORY_REASON,
    MANDATORY_TRADE,
    MandatoryTrades,
    PINNED_INSTRUCTION,
    PRODUCT_MAX_WEIGHT,
    PRODUCT_MIN_WEIGHT,
    RAW_MODEL_WEIGHT,
    REBALANCE_CADENCE_ELIGIBLE,
    SETTLEMENT_CASH,
    ExactExecutionConfig,
    ExecutionPolicyConfig,
    build_execution_target,
    score_execution_trades,
)


ASSET_CLASSES = ("Liquidity", "Fixed Income", "Equity", "Alternatives")


ASSETS = pd.Index(["A", "B", "Cash"])


def _exact_config(
    max_trades: int = 1,
    material_trade_threshold: float = 0.05,
) -> ExecutionPolicyConfig:
    """Return a pure-tracking-error Exact Execution policy for three assets."""
    return ExecutionPolicyConfig(
        tre_weight_by_asset_class={key: 1.0 for key in ASSET_CLASSES},
        small_trade_cutoff=0.10,
        small_trade_cutoff_by_asset_class={key: 0.10 for key in ASSET_CLASSES},
        material_trade_threshold=material_trade_threshold,
        max_trades=max_trades,
        exact_execution=ExactExecutionConfig(
            max_positions=2,
            alpha_tradeoff=0.0,
            relative_gap_tolerance=1e-8,
        ),
    )


def _exact_inputs(
    config: ExecutionPolicyConfig,
) -> tuple[pd.DataFrame, Constraints, pd.DataFrame]:
    """Build an ordinary stage-one table consumed by the exact solver."""
    current = pd.Series([0.4, 0.1, 0.5], index=ASSETS)
    model = pd.Series([0.2, 0.3, 0.5], index=ASSETS)
    covariance = pd.DataFrame(np.diag([1.0, 4.0, 1.0]), index=ASSETS, columns=ASSETS)
    target = build_execution_target(
        current_weights=current,
        model_weights=model,
        previous_model_weights=model,
        settlement_cash_asset="Cash",
        config=config,
        rebalancing_allowed=pd.Series(True, index=ASSETS),
        required_min_weights=pd.Series(0.0, index=ASSETS),
        required_max_weights=pd.Series(1.0, index=ASSETS),
        asset_classes=pd.Series(["Equity", "Equity", "Liquidity"], index=ASSETS),
    )
    table = score_execution_trades(
        target=target,
        alphas=pd.Series(0.0, index=ASSETS),
        covariance=covariance,
        asset_classes=pd.Series(["Equity", "Equity", "Liquidity"], index=ASSETS),
        config=config,
    )
    constraints = Constraints(
        min_weights=pd.Series(0.0, index=ASSETS),
        max_weights=pd.Series(1.0, index=ASSETS),
    )
    return table, constraints, covariance


def _one_ticket_reference_objective(
    current: np.ndarray,
    model: np.ndarray,
    covariance: np.ndarray,
    instrument: int,
) -> tuple[float, np.ndarray]:
    """Solve one cash-funded coordinate analytically, independent of CVXPY."""
    direction = np.zeros(3)
    direction[instrument] = 1.0
    direction[2] = -1.0
    active = current - model
    delta = -float(direction @ covariance @ active) / float(direction @ covariance @ direction)
    delta = float(np.clip(delta, -current[instrument], 1.0 - current[instrument]))
    weights = current + delta * direction
    objective = float((weights - model) @ covariance @ (weights - model))
    return objective, weights


def test_live_exact_candidates_exclude_irrelevant_zero_weight_assets() -> None:
    """Live Exact freezes zero-current/zero-model assets unless they are required."""
    config = _exact_config(max_trades=2)
    table, _, _ = _exact_inputs(config)
    for instrument in ["Model Below Cutoff", "Mandatory", "Fixed", "Minimum", "Irrelevant"]:
        table.loc[instrument] = table.loc["B"]
        table.loc[instrument, [CURRENT_WEIGHT, RAW_MODEL_WEIGHT, BASE_WEIGHT]] = 0.0
        table.loc[instrument, MANDATORY_CATEGORY] = ""
        table.loc[instrument, MANDATORY_TRADE] = False
        table.loc[instrument, FIXED_INSTRUCTION] = False
        table.loc[instrument, PINNED_INSTRUCTION] = False
        table.loc[instrument, PRODUCT_MIN_WEIGHT] = 0.0
        table.loc[instrument, SETTLEMENT_CASH] = False
        table.loc[instrument, EFFECTIVE_SMALL_TRADE_CUTOFF] = 0.10
    table.loc["B", RAW_MODEL_WEIGHT] = 0.25
    table.loc["Model Below Cutoff", RAW_MODEL_WEIGHT] = 0.05
    table.loc["Mandatory", MANDATORY_TRADE] = True
    table.loc["Fixed", FIXED_INSTRUCTION] = True
    table.loc["Minimum", PRODUCT_MIN_WEIGHT] = 0.02

    problem = exact_optimiser._prepare_exact_problem(table, config)
    tradeable = set(table.index[problem.tradeable])
    frozen = set(table.index[problem.frozen])

    assert {"A", "B", "Model Below Cutoff", "Mandatory", "Fixed", "Minimum"} <= tradeable
    assert "Irrelevant" in frozen


def test_live_current_max_repair_is_an_exact_mandatory_ticket() -> None:
    """A live Current Max breach cannot be skipped by the exact MIP."""
    config = replace(
        _exact_config(max_trades=1, material_trade_threshold=0.01),
        small_trade_cutoff=0.01,
        small_trade_cutoff_by_asset_class={key: 0.01 for key in ASSET_CLASSES},
    )
    current = pd.Series([0.10096421, 0.0, 0.89903579], index=ASSETS)
    model = pd.Series([0.09892663, 0.0, 0.90107337], index=ASSETS)
    covariance = pd.DataFrame(np.eye(3), index=ASSETS, columns=ASSETS)
    target = build_execution_target(
        current_weights=current,
        model_weights=model,
        previous_model_weights=model,
        settlement_cash_asset="Cash",
        config=config,
        mandatory_trades=MandatoryTrades(hard_max={"A": 0.10}),
        instruction_audit=pd.DataFrame(
            {
                "Instruction": ["hard_max"],
                "Comment": ["Current Min=0; Current Max=0.10"],
                "Source": ["consumer supplied bounds"],
                "model_max_repair": [True],
                "allow_current_pin": [True],
            },
            index=["A"],
        ),
        asset_classes=pd.Series(["Equity", "Equity", "Liquidity"], index=ASSETS),
    )
    table = score_execution_trades(
        target=target,
        alphas=pd.Series(0.0, index=ASSETS),
        covariance=covariance,
        asset_classes=pd.Series(["Equity", "Equity", "Liquidity"], index=ASSETS),
        config=config,
    )

    problem = exact_optimiser._prepare_exact_problem(table, config)

    assert problem.index[problem.exact].tolist() == ["A"]
    assert problem.index[problem.minimum_ticket_exempt].tolist() == ["A"]
    assert problem.exact_targets[problem.exact[0]] == pytest.approx(0.09892663)
    assert problem.mandatory_ticket_count == 1
    assert problem.effective_max_trades == 1

    solved = problem.current.copy()
    solved[problem.exact] = problem.exact_targets[problem.exact]
    solved[problem.cash_index] = 1.0 - solved[problem.exact].sum()
    minimum_ticket = next(
        residual
        for residual in exact_optimiser._audit_exact_solution(
            weights=solved,
            problem_data=problem,
            config=config,
        )
        if residual.name == "smallest_ticket"
    )
    assert minimum_ticket.passed


def test_exact_execution_config_is_frozen_and_normalises_solver() -> None:
    """The solver controls are immutable and normalize their backend name."""
    config = ExactExecutionConfig(solver="mosek")

    assert config.solver == "MOSEK"
    assert config.alpha_tradeoff == pytest.approx(0.0001)
    assert config.max_branches == 500_000
    assert config.max_solve_seconds is None
    assert config.accept_audited_incumbent is True
    with pytest.raises(FrozenInstanceError):
        config.max_positions = 31


@pytest.mark.parametrize("value", [0.0, -1.0, np.inf, np.nan])
def test_exact_execution_config_rejects_invalid_solve_time(value) -> None:
    """The optional exact-solver wall-clock cap must be finite and positive."""
    with pytest.raises(ValueError, match="max_solve_seconds"):
        ExactExecutionConfig(max_solve_seconds=value)


def test_exact_mosek_solve_enables_warm_start(monkeypatch, mosek_api_stub) -> None:
    """The production Exact solve asks CVXPY to reuse an available MIP start."""
    task = Mock()
    task.getdouinf.return_value = 0.0
    task.getintinf.return_value = 0
    chain = Mock()
    chain.solve_via_data.return_value = {"task": task}
    problem = Mock()
    problem.get_problem_data.return_value = ({}, chain, None)
    monkeypatch.setattr(cvx, "installed_solvers", lambda: ["MOSEK"])

    exact_optimiser._solve_mosek_with_audit(
        problem=problem,
        exact_config=ExactExecutionConfig(),
    )

    solve_kwargs = chain.solve_via_data.call_args.kwargs
    assert solve_kwargs["warm_start"] is True
    assert solve_kwargs["solver_opts"]["mosek_params"]["MSK_IPAR_MIO_MAX_NUM_BRANCHES"] == 500_000
    assert "MSK_DPAR_MIO_MAX_TIME" not in solve_kwargs["solver_opts"]["mosek_params"]


def test_exact_mosek_solve_passes_time_limit_and_reports_termination(
    monkeypatch,
    mosek_api_stub,
) -> None:
    """The MOSEK time cap and inferred stop reason survive task disposal."""
    import mosek

    task = Mock()
    task.getdouinf.side_effect = lambda item: {
        mosek.dinfitem.mio_obj_rel_gap: 0.25,
        mosek.dinfitem.mio_obj_abs_gap: 0.01,
        mosek.dinfitem.mio_obj_int: 0.04,
        mosek.dinfitem.mio_obj_bound: 0.03,
        mosek.dinfitem.optimizer_time: 1_800.1,
    }[item]
    task.getintinf.side_effect = lambda item: {
        mosek.iinfitem.mio_num_branch: 400_000,
        mosek.iinfitem.mio_num_int_solutions: 3,
    }[item]
    chain = Mock()
    chain.solve_via_data.return_value = {"task": task}
    problem = Mock()
    problem.get_problem_data.return_value = ({}, chain, None)
    monkeypatch.setattr(cvx, "installed_solvers", lambda: ["MOSEK"])

    audit = exact_optimiser._solve_mosek_with_audit(
        problem=problem,
        exact_config=ExactExecutionConfig(max_solve_seconds=1_800.0),
    )

    solver_options = chain.solve_via_data.call_args.kwargs["solver_opts"]
    assert solver_options["mosek_params"]["MSK_DPAR_MIO_MAX_TIME"] == 1_800.0
    assert audit.termination == "time_limit"


def test_exact_inaccurate_incumbent_is_accepted_only_after_audit(
    monkeypatch,
) -> None:
    """A branch-limited incumbent remains usable when every hard audit passes."""
    base = _exact_config()
    config = replace(
        base,
        exact_execution=replace(
            base.exact_execution,
            accept_audited_incumbent=True,
        ),
    )
    table, constraints, covariance = _exact_inputs(config)

    def fake_inaccurate_solve(problem, exact_config):
        """Supply a feasible incumbent with an unproved objective bound."""
        problem._status = cvx.OPTIMAL_INACCURATE
        weight_variable = next(
            variable for variable in problem.variables() if variable.shape == (3,)
        )
        weight_variable.value = table[CURRENT_WEIGHT].to_numpy(dtype=float)
        return ExactExecutionSolverAudit(
            relative_gap=0.25,
            branches=exact_config.max_branches,
            integer_solutions=3,
            solve_seconds=1.0,
            termination="branch_limit",
        )

    monkeypatch.setattr(
        exact_optimiser,
        "_solve_mosek_with_audit",
        fake_inaccurate_solve,
    )

    result = solve_exact_execution_portfolio(
        trade_table=table,
        base_constraints=constraints,
        covariance=covariance,
        config=config,
        context="audited incumbent",
    )

    assert result.accepted and result.compliant
    assert result.trade_table[MIP_TERMINATION].iloc[0] == "branch_limit"


def test_exact_weight_dust_cleanup_preserves_total_in_cash() -> None:
    """Tiny accepted long-only negatives are zeroed without changing exposure."""
    current = np.array([0.4, 0.0, 0.6])
    weights = np.array([0.40000005, -2e-7, 0.60000015])

    cleaned, count = exact_optimiser._clean_exact_weight_dust(
        weights=weights,
        current=current,
        cash_index=2,
    )

    assert count == 2
    assert cleaned[0] == pytest.approx(current[0])
    assert cleaned[1] == 0.0
    assert cleaned.sum() == pytest.approx(weights.sum())
    assert cleaned[2] == pytest.approx(0.6)


def test_exact_inception_cleanup_does_not_restore_excluded_raw_weight() -> None:
    """Inception zero exits remain zero even when the raw weight was tiny."""
    current = np.array([0.4, 5e-8, 0.59999995])
    weights = np.array([0.4, 0.0, 0.6])

    cleaned, count = exact_optimiser._clean_exact_weight_dust(
        weights=weights,
        current=current,
        cash_index=2,
        clean_trade_dust=False,
    )

    assert count == 0
    np.testing.assert_array_equal(cleaned, weights)


def test_exact_sub_ticket_failure_logs_target_and_required_ticket(caplog) -> None:
    """A fixed sub-ticket target identifies the instrument and exact conflict."""
    config = _exact_config(material_trade_threshold=0.005)
    table, constraints, covariance = _exact_inputs(config)
    table.loc["A", FIXED_INSTRUCTION] = True
    table.loc["A", BASE_WEIGHT] = 0.401
    table.loc["A", MANDATORY_CATEGORY] = "fixed_instruction"
    table.loc["A", MANDATORY_REASON] = "lifecycle fixed target"

    with caplog.at_level(
        logging.WARNING,
        logger=exact_optimiser.__name__,
    ):
        with pytest.raises(ValueError, match="sub-minimum tickets"):
            solve_exact_execution_portfolio(
                trade_table=table,
                base_constraints=constraints,
                covariance=covariance,
                config=config,
                context="2008-10-31",
            )

    failure = next(
        record.exact_execution_failure
        for record in caplog.records
        if hasattr(record, "exact_execution_failure")
    )
    assert failure.stage == "preflight"
    assert failure.cause_codes == ("exact_target_sub_minimum_ticket",)
    assert "A(current=0.40000000, target=0.40100000" in failure.diagnostic
    assert "required_ticket=0.00100000" in failure.diagnostic
    assert "minimum_ticket=0.00500000" in failure.diagnostic
    assert "category=fixed_instruction" in failure.diagnostic
    assert "reason=lifecycle fixed target" in failure.diagnostic


def test_exact_mandatory_target_overrides_cadence_with_warning(caplog) -> None:
    """A material exact instruction trades off-cycle while other assets stay frozen."""
    config = _exact_config(max_trades=1, material_trade_threshold=0.05)
    table, _, _ = _exact_inputs(config)
    table[REBALANCE_CADENCE_ELIGIBLE] = pd.Series({"A": False, "B": False, "Cash": True})
    table.loc["A", FIXED_INSTRUCTION] = True
    table.loc["A", BASE_WEIGHT] = 0.0
    table.loc["A", PRODUCT_MAX_WEIGHT] = 0.0

    with caplog.at_level(
        logging.WARNING,
        logger=exact_optimiser.__name__,
    ):
        problem = exact_optimiser._prepare_exact_problem(
            trade_table=table,
            config=config,
            context="2026-07-31",
        )

    assert problem.index[problem.tradeable].tolist() == ["A"]
    assert problem.index[problem.exact].tolist() == ["A"]
    assert problem.index[problem.frozen].tolist() == ["B"]
    assert (
        "[2026-07-31] Exact Execution off-cycle override: mandatory exact "
        "targets supersede rebalance cadence for "
        "A(current=0.40000000, target=0.00000000, ticket=0.40000000)"
    ) in caplog.text


def test_exact_mandatory_targets_can_expand_the_total_ticket_cap() -> None:
    """Two exact tickets plus one discretionary slot produce a cap of three."""
    base_config = _exact_config(max_trades=1, material_trade_threshold=0.05)
    table, _, _ = _exact_inputs(base_config)
    table.loc[["A", "B"], FIXED_INSTRUCTION] = True
    table.loc[["A", "B"], BASE_WEIGHT] = [0.0, 0.5]

    with pytest.raises(ValueError, match="effective_max_trades=1"):
        exact_optimiser._prepare_exact_problem(
            trade_table=table,
            config=base_config,
        )

    expanded_config = replace(
        base_config,
        add_mandatory_trades_to_max_trades=True,
    )
    problem = exact_optimiser._prepare_exact_problem(
        trade_table=table,
        config=expanded_config,
    )

    assert problem.mandatory_ticket_count == 2
    assert problem.effective_max_trades == 3


@pytest.mark.skipif("MOSEK" not in cvx.installed_solvers(), reason="MOSEK unavailable")
def test_exact_expanded_cap_accepts_more_mandatory_tickets_than_configured() -> None:
    """The hard audit uses the expanded cap while preserving both exact targets."""
    config = replace(
        _exact_config(max_trades=1, material_trade_threshold=0.05),
        add_mandatory_trades_to_max_trades=True,
    )
    table, constraints, covariance = _exact_inputs(config)
    table.loc[["A", "B"], FIXED_INSTRUCTION] = True
    table.loc[["A", "B"], BASE_WEIGHT] = [0.0, 0.5]

    result = solve_exact_execution_portfolio(
        trade_table=table,
        base_constraints=constraints,
        covariance=covariance,
        config=config,
        context="expanded ticket reference",
    )

    non_cash_trades = result.weights.sub(table[CURRENT_WEIGHT]).drop(index="Cash").abs().gt(1e-7)
    ticket_audit = next(
        residual
        for residual in result.outcome.constraint_residuals
        if residual.name == "non_cash_tickets"
    )
    assert result.accepted and result.compliant
    assert non_cash_trades.sum() == 2
    assert ticket_audit.actual == 2
    assert ticket_audit.upper == 3


def test_exact_infeasible_failure_logs_position_ticket_conflict(
    monkeypatch,
    caplog,
) -> None:
    """An infeasible pinned portfolio reports its position/ticket arithmetic."""
    base_config = _exact_config(max_trades=1)
    config = replace(
        base_config,
        exact_execution=replace(base_config.exact_execution, max_positions=1),
    )
    table, constraints, covariance = _exact_inputs(config)
    table.loc[["A", "B"], FIXED_INSTRUCTION] = True
    table.loc[["A", "B"], BASE_WEIGHT] = table.loc[["A", "B"], CURRENT_WEIGHT]

    def fake_infeasible_solve(problem, exact_config):
        """Mark the CVXPY problem infeasible without requiring MOSEK."""
        problem._status = cvx.INFEASIBLE
        return ExactExecutionSolverAudit(
            branches=17,
            integer_solutions=0,
            solve_seconds=0.25,
        )

    monkeypatch.setattr(
        exact_optimiser,
        "_solve_mosek_with_audit",
        fake_infeasible_solve,
    )
    with caplog.at_level(
        logging.WARNING,
        logger=exact_optimiser.__name__,
    ):
        result = solve_exact_execution_portfolio(
            trade_table=table,
            base_constraints=constraints,
            covariance=covariance,
            config=config,
            context="2009-01-30",
        )

    failure = next(
        record.exact_execution_failure
        for record in caplog.records
        if hasattr(record, "exact_execution_failure")
    )
    assert not result.accepted
    assert failure.stage == "solver"
    assert "position_ticket_budget" in failure.cause_codes
    assert "positions_after_exact:2" in failure.diagnostic
    assert "max_positions:1" in failure.diagnostic
    assert "exits_needed:1" in failure.diagnostic
    assert "closable_positions:0" in failure.diagnostic
    assert "branches:17" in failure.diagnostic


def test_exact_infeasible_failure_logs_exact_product_bound(
    monkeypatch,
    caplog,
) -> None:
    """An exact target outside its product maximum names the conflicting bound."""
    config = _exact_config(max_trades=1)
    table, constraints, covariance = _exact_inputs(config)
    table.loc["A", FIXED_INSTRUCTION] = True
    table.loc["A", BASE_WEIGHT] = 0.60
    table.loc["A", PRODUCT_MAX_WEIGHT] = 0.50

    def fake_infeasible_solve(problem, exact_config):
        """Mark the CVXPY problem infeasible without requiring MOSEK."""
        problem._status = cvx.INFEASIBLE
        return ExactExecutionSolverAudit()

    monkeypatch.setattr(
        exact_optimiser,
        "_solve_mosek_with_audit",
        fake_infeasible_solve,
    )
    with caplog.at_level(
        logging.WARNING,
        logger=exact_optimiser.__name__,
    ):
        result = solve_exact_execution_portfolio(
            trade_table=table,
            base_constraints=constraints,
            covariance=covariance,
            config=config,
            context="2010-02-26",
        )

    failure = next(
        record.exact_execution_failure
        for record in caplog.records
        if hasattr(record, "exact_execution_failure")
    )
    assert not result.accepted
    assert "exact_target_product_bound" in failure.cause_codes
    assert "A:target=0.60000000,product=[0.00000000,0.50000000]" in failure.diagnostic


@pytest.mark.skipif("MOSEK" not in cvx.installed_solvers(), reason="MOSEK unavailable")
def test_exact_solution_matches_independent_one_ticket_reference() -> None:
    """Joint selection matches enumeration plus an analytic continuous solve."""
    config = _exact_config()
    table, constraints, covariance = _exact_inputs(config)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = solve_exact_execution_portfolio(
            trade_table=table,
            base_constraints=constraints,
            covariance=covariance,
            config=config,
            context="three-asset reference",
        )

    current = np.array([0.4, 0.1, 0.5])
    model = np.array([0.2, 0.3, 0.5])
    candidates = [
        _one_ticket_reference_objective(
            current=current,
            model=model,
            covariance=covariance.to_numpy(),
            instrument=instrument,
        )
        for instrument in (0, 1)
    ]
    _, reference = min(candidates, key=lambda item: item[0])

    assert result.accepted and result.compliant
    np.testing.assert_allclose(result.weights.to_numpy(), reference, atol=1e-5)
    non_cash_trades = result.trade_table[EXECUTED_TRADE].abs().gt(1e-8) & ~result.trade_table[
        SETTLEMENT_CASH
    ].astype(bool)
    assert non_cash_trades.sum() == 1
    assert result.trade_table[EXACT_EXECUTION].all()
    assert not result.trade_table[DIRECTED_RESCUE_CANDIDATE].any()
    assert result.trade_table[MIP_RELATIVE_GAP].iloc[0] <= 1e-8
    assert not [
        warning
        for warning in caught
        if "putaijlist64: Incorrect array format causing data to be copied" in str(warning.message)
    ]
    hard_names = {residual.name for residual in result.outcome.constraint_residuals}
    assert {
        "non_cash_tickets",
        "non_cash_positions",
        "smallest_ticket",
        "worst_traded_holding_slack",
        "largest_off_cadence_trade",
        "largest_exact_target_error",
        "largest_true_bound_violation",
    }.issubset(hard_names)


@pytest.mark.skipif("MOSEK" not in cvx.installed_solvers(), reason="MOSEK unavailable")
def test_exact_soft_relaxes_only_named_sub_asset_class_bounds() -> None:
    """Soft Exact penalizes level-2 breaches while preserving other group caps."""
    base = _exact_config(max_trades=1, material_trade_threshold=0.05)
    config = replace(
        base,
        exact_execution=replace(
            base.exact_execution,
            sub_asset_class_penalty=100.0,
        ),
    )
    table, constraints, covariance = _exact_inputs(config)
    loadings = pd.DataFrame(
        {
            "Soft SAC": [1.0, 0.0, 0.0],
            "Hard product cap": [1.0, 0.0, 0.0],
        },
        index=ASSETS,
    )
    constraints = constraints.copy(
        group_lower_upper_constraints=GroupLowerUpperConstraints(
            group_loadings=loadings,
            group_min_allocation=pd.Series(
                {
                    "Soft SAC": 0.70,
                    "Hard product cap": np.nan,
                }
            ),
            group_max_allocation=pd.Series(
                {
                    "Soft SAC": np.nan,
                    "Hard product cap": 0.45,
                }
            ),
        )
    )

    result = solve_exact_execution_portfolio(
        trade_table=table,
        base_constraints=constraints,
        covariance=covariance,
        config=config,
        soft_group_names=("Soft SAC",),
        context="soft level-2 reference",
    )

    soft = next(
        residual for residual in result.outcome.constraint_residuals if residual.name == "Soft SAC"
    )
    hard = next(
        residual
        for residual in result.outcome.constraint_residuals
        if residual.name == "Hard product cap"
    )
    assert result.accepted and result.compliant
    assert result.weights["A"] == pytest.approx(0.45, abs=1e-6)
    assert not soft.hard and soft.violation == pytest.approx(0.25, abs=1e-6)
    assert hard.hard and hard.passed


@pytest.mark.skipif("MOSEK" not in cvx.installed_solvers(), reason="MOSEK unavailable")
def test_exact_inception_candidates_force_excluded_assets_to_zero() -> None:
    """A deterministic inception exclusion removes binaries and exits the asset."""
    config = _exact_config(max_trades=2, material_trade_threshold=1e-8)
    table, constraints, covariance = _exact_inputs(config)
    table[EXACT_INCEPTION_CANDIDATE] = pd.Series({"A": True, "B": False, "Cash": True})

    result = solve_exact_execution_portfolio(
        trade_table=table,
        base_constraints=constraints,
        covariance=covariance,
        config=config,
        context="reduced inception reference",
    )

    assert result.accepted and result.compliant
    assert result.weights["B"] == pytest.approx(0.0, abs=1e-8)
    assert not result.trade_table.at["B", EXACT_INCEPTION_CANDIDATE]


@pytest.mark.skipif("MOSEK" not in cvx.installed_solvers(), reason="MOSEK unavailable")
def test_exact_history_candidates_freeze_exclusions_and_prevent_overshoot() -> None:
    """Historical screening freezes exclusions and trades toward the model."""
    config = _exact_config(max_trades=2, material_trade_threshold=1e-8)
    table, constraints, covariance = _exact_inputs(config)
    table[EXACT_HISTORY_CANDIDATE] = pd.Series({"A": True, "B": False, "Cash": True})

    result = solve_exact_execution_portfolio(
        trade_table=table,
        base_constraints=constraints,
        covariance=covariance,
        config=config,
        context="screened history reference",
    )

    assert result.accepted and result.compliant
    assert result.weights["B"] == pytest.approx(0.1, abs=1e-8)
    assert 0.2 - 1e-8 <= result.weights["A"] <= 0.4 + 1e-8


@pytest.fixture
def mosek_api_stub(monkeypatch):
    """Model only native statistic names so transport tests run without the extra."""
    names = ("mio_obj_rel_gap", "mio_obj_abs_gap", "mio_obj_int", "mio_obj_bound", "optimizer_time")
    module = SimpleNamespace(
        dinfitem=SimpleNamespace(**{n: n for n in names}),
        iinfitem=SimpleNamespace(mio_num_branch="branches", mio_num_int_solutions="solutions"),
    )
    monkeypatch.setitem(sys.modules, "mosek", module)
    return module
