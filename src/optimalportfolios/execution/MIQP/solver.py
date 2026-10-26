"""Exact joint selection of execution positions, tickets and trade sizes.

``Exact Execution`` is the desk-facing name for the convex mixed-integer
quadratic program (MIQP) documented in ``execution_policy_methodology.tex``. The
solver consumes the same stage-one policy table and production
``optimalportfolios.Constraints`` object as the ranked execution path, but it
ignores the pre-ranked selection: position membership, ticket membership and
trade sizes are optimized jointly against the raw model portfolio.

Weights are fractions of NAV and covariance remains in its supplied annualized
variance units. No return conversion, annualization or renormalization occurs
in this module. MOSEK is loaded only inside the solve helper so ordinary ROSAA
imports remain usable without the commercial optional dependency.
"""
from optimalportfolios.execution._common.model import is_model_zero

import logging
from dataclasses import dataclass, replace
from typing import Optional
import warnings

import cvxpy as cvx
import numpy as np
import pandas as pd

from optimalportfolios.optimization.constraints import (
    ConstraintEnforcementType,
    Constraints,
    GroupLowerUpperConstraints,
    cvx_covar_variance,
)
from optimalportfolios.optimization.covar_factorization import factorize_covariance
from optimalportfolios.optimization.solver_diagnostics import (
    ConstraintResidual,
    OptimizationOutcome,
    validate_solution,
)
from optimalportfolios.execution.v1.solver import (
    CORRIDOR_RELAXATION,
    CUTOFF_SELL_DOWN_DUST_ROUNDED,
    CUTOFF_SELL_DOWN_RETAINED_FRACTION,
    CUTOFF_SELL_DOWN_RETENTION,
    DIRECTED_RESCUE_CANDIDATE,
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
    REWEIGHT_FUNDING,
    RESCUE_REASON,
    STRICT_CORRIDOR_MAX,
    STRICT_CORRIDOR_MIN,
    TARGET_GAP,
    ExecutionOptimizationResult,
)
from optimalportfolios.execution.v1.policy import (
    ALPHA,
    BASE_WEIGHT,
    CUTOFF_SELL_DOWN,
    CURRENT_WEIGHT,
    DESK_FUNDING,
    DESIRED_REWEIGHT,
    EFFECTIVE_MODEL_WEIGHT,
    EFFECTIVE_SMALL_TRADE_CUTOFF,
    ENTRY_FUNDING,
    EXIT_FUNDING,
    FEASIBILITY_RESCUE_TRADE,
    FIXED_INSTRUCTION,
    INSTRUCTION_MAX_SELL,
    INSTRUCTION_SOURCE,
    LOW_WEIGHT_EXCLUDED,
    MANDATORY_CATEGORY,
    MANDATORY_FUNDING,
    MANDATORY_REASON,
    MANDATORY_TRADE,
    MATERIAL_TRADE,
    MODEL_ENTRY,
    MODEL_EXIT,
    OTHER_FUNDING,
    PINNED_INSTRUCTION,
    POLICY_TARGET_WEIGHT,
    PRODUCT_BOUND_OVERRIDE,
    PRODUCT_BOUND_OVERRIDE_BREACH,
    PRODUCT_BOUND_OVERRIDE_REASON,
    PRODUCT_MAX_WEIGHT,
    PRODUCT_MIN_WEIGHT,
    RAW_MODEL_WEIGHT,
    MODEL_MAX_REPAIR,
    REBALANCE_CADENCE_ELIGIBLE,
    REQUESTED_TRADE,
    RULE4_TRADE,
    SELECTED_TRADE,
    SETTLEMENT_CASH,
    TRADE_CANDIDATE,
    TRADE_WEIGHT,
    WEIGHT_ZERO_TOLERANCE,
    ExactExecutionConfig,
    ExecutionPolicyConfig,
    resolve_effective_max_trades,
)


EXACT_EXECUTION = 'exact_execution'
EXACT_TARGET = 'exact_target'
EXACT_INCEPTION_CANDIDATE = 'exact_inception_candidate'
EXACT_HISTORY_CANDIDATE = 'exact_history_candidate'
MIP_RELATIVE_GAP = 'mip_relative_gap'
MIP_ABSOLUTE_GAP = 'mip_absolute_gap'
MIP_BRANCHES = 'mip_branches'
MIP_INTEGER_SOLUTIONS = 'mip_integer_solutions'
MIP_INCUMBENT_OBJECTIVE = 'mip_incumbent_objective'
MIP_OBJECTIVE_BOUND = 'mip_objective_bound'
MIP_SOLVE_SECONDS = 'mip_solve_seconds'
MIP_TERMINATION = 'mip_termination'
MIP_ACCEPTANCE_BASIS = 'mip_acceptance_basis'
SUB_ASSET_CLASS_PENALTY = 'sub_asset_class_penalty'

_EXACT_CONSTRAINT_TOLERANCE = 1e-7
_EXACT_OUTPUT_ZERO_TOLERANCE = 1e-6

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExactExecutionSolverAudit:
    """Immutable branch-and-bound diagnostics captured before MOSEK task disposal."""

    relative_gap: float = np.nan
    absolute_gap: float = np.nan
    branches: int = 0
    integer_solutions: int = 0
    incumbent_objective: float = np.nan
    objective_bound: float = np.nan
    solve_seconds: float = np.nan
    termination: str = 'other'


@dataclass(frozen=True)
class _ExactProblemData:
    """Aligned numeric inputs and membership masks for one exact solve."""

    index: pd.Index
    current: np.ndarray
    model: np.ndarray
    alphas: np.ndarray
    true_lower: np.ndarray
    true_upper: np.ndarray
    effective_lower: np.ndarray
    effective_upper: np.ndarray
    holding_floor: np.ndarray
    cash_index: int
    tradeable: np.ndarray
    frozen: np.ndarray
    forced_zero: np.ndarray
    exact: np.ndarray
    exact_targets: np.ndarray
    minimum_ticket_exempt: np.ndarray
    mandatory_ticket_count: int
    effective_max_trades: int


@dataclass(frozen=True)
class ExactExecutionFailureRecord:
    """Structured Exact Execution failure attached to warning log records."""

    context: str
    stage: str
    status: str
    cause_codes: tuple[str, ...]
    diagnostic: str


def _log_exact_failure(
        context: str,
        stage: str,
        status: str,
        cause_codes: tuple[str, ...],
        diagnostic: str,
) -> None:
    """Emit one human-readable and machine-aggregatable failure warning."""
    record = ExactExecutionFailureRecord(
        context=context or 'execution',
        stage=stage,
        status=status,
        cause_codes=cause_codes,
        diagnostic=diagnostic,
    )
    logger.warning(
        '[%s] exact_execution_failure stage=%s status=%s causes=%s; %s',
        record.context,
        stage,
        status,
        '|'.join(cause_codes),
        diagnostic,
        extra={'exact_execution_failure': record},
    )


def _format_sub_ticket_details(
        trade_table: pd.DataFrame,
        locations: np.ndarray,
        current: np.ndarray,
        targets: np.ndarray,
        minimum_ticket: float,
) -> str:
    """Format exact-target tickets that contradict the configured minimum."""
    rows = []
    reason = trade_table.get(
        MANDATORY_REASON,
        pd.Series('', index=trade_table.index, dtype=object),
    ).fillna('').astype(str)
    category = trade_table[MANDATORY_CATEGORY].fillna('').astype(str)
    for location in locations:
        instrument = trade_table.index[location]
        required = abs(float(targets[location] - current[location]))
        rows.append(
            f'{instrument}(current={current[location]:.8f}, '
            f'target={targets[location]:.8f}, required_ticket={required:.8f}, '
            f'minimum_ticket={minimum_ticket:.8f}, '
            f'category={category.iloc[location] or "unspecified"}, '
            f'reason={reason.iloc[location] or "unspecified"})'
        )
    return '; '.join(rows)


def _ticket_limited_effect_range(
        loadings: np.ndarray,
        lower_delta: np.ndarray,
        upper_delta: np.ndarray,
        ticket_budget: int,
        minimum_ticket: float,
) -> tuple[float, float]:
    """Return an optimistic one-constraint effect interval under a ticket cap."""
    lower_effect = np.minimum(loadings * lower_delta, loadings * upper_delta)
    upper_effect = np.maximum(loadings * lower_delta, loadings * upper_delta)
    lower_effect = np.where(
        np.maximum(np.abs(lower_delta), np.abs(upper_delta))
        >= minimum_ticket - _EXACT_CONSTRAINT_TOLERANCE,
        np.minimum(lower_effect, 0.0),
        0.0,
    )
    upper_effect = np.where(
        np.maximum(np.abs(lower_delta), np.abs(upper_delta))
        >= minimum_ticket - _EXACT_CONSTRAINT_TOLERANCE,
        np.maximum(upper_effect, 0.0),
        0.0,
    )
    lower = float(np.sort(lower_effect)[:ticket_budget].sum())
    upper = float(np.sort(upper_effect)[::-1][:ticket_budget].sum())
    return lower, upper


def _diagnose_exact_infeasibility(
        problem_data: _ExactProblemData,
        constraints: Constraints,
        config: ExecutionPolicyConfig,
        audit: ExactExecutionSolverAudit,
        outcome: OptimizationOutcome,
) -> tuple[tuple[str, ...], str]:
    """Identify deterministic ticket, position, product and group conflicts."""
    exact_config = config.exact_execution
    if exact_config is None or config.max_trades is None:
        return ('invalid_exact_configuration',), 'Exact configuration is incomplete'

    current = problem_data.current
    post_exact = current.copy()
    post_exact[problem_data.exact] = problem_data.exact_targets[problem_data.exact]
    cash_index = problem_data.cash_index
    post_exact[cash_index] = current[cash_index] - float(
        (post_exact - current).sum()
    )
    active_exact = np.abs(post_exact - current) > _EXACT_CONSTRAINT_TOLERANCE
    active_exact[cash_index] = False
    exact_tickets = int(active_exact.sum())
    remaining_tickets = max(
        0, problem_data.effective_max_trades - exact_tickets
    )
    held_after_exact = post_exact > _EXACT_CONSTRAINT_TOLERANCE
    held_after_exact[cash_index] = False
    positions_after_exact = int(held_after_exact.sum())
    exits_needed = max(0, positions_after_exact - exact_config.max_positions)

    exact_mask = np.zeros(len(problem_data.index), dtype=bool)
    exact_mask[problem_data.exact] = True
    tradeable_mask = np.zeros(len(problem_data.index), dtype=bool)
    tradeable_mask[problem_data.tradeable] = True
    closable = (
        held_after_exact
        & tradeable_mask
        & ~exact_mask
        & (problem_data.effective_lower <= _EXACT_CONSTRAINT_TOLERANCE)
        & (post_exact >= config.material_trade_threshold
           - _EXACT_CONSTRAINT_TOLERANCE)
    )
    closable_positions = int(closable.sum())

    causes = []
    details = [
        'position_ticket_state='
        f'positions_after_exact:{positions_after_exact},'
        f'max_positions:{exact_config.max_positions},'
        f'exits_needed:{exits_needed},closable_positions:{closable_positions},'
        f'exact_tickets:{exact_tickets},remaining_tickets:{remaining_tickets},'
        f'configured_max_trades:{config.max_trades},'
        f'effective_max_trades:{problem_data.effective_max_trades},'
        'mandatory_added_to_cap:'
        f'{config.add_mandatory_trades_to_max_trades}'
    ]
    if exits_needed > min(remaining_tickets, closable_positions):
        causes.append('position_ticket_budget')

    active_exact_locations = np.flatnonzero(active_exact & exact_mask)
    exact_bound_conflicts = active_exact_locations[
        (
            post_exact[active_exact_locations]
            < problem_data.true_lower[active_exact_locations]
            - _EXACT_CONSTRAINT_TOLERANCE
        )
        | (
            post_exact[active_exact_locations]
            > problem_data.true_upper[active_exact_locations]
            + _EXACT_CONSTRAINT_TOLERANCE
        )
    ]
    if exact_config.repair_on_touch and exact_bound_conflicts.size:
        causes.append('exact_target_product_bound')
        conflicts = ','.join(
            f'{problem_data.index[location]}:'
            f'target={post_exact[location]:.8f},'
            f'product=[{problem_data.true_lower[location]:.8f},'
            f'{problem_data.true_upper[location]:.8f}]'
            for location in exact_bound_conflicts
        )
        details.append(f'exact_product_conflicts={conflicts}')

    feasible_lower = problem_data.effective_lower.copy()
    feasible_upper = problem_data.effective_upper.copy()
    feasible_lower[problem_data.frozen] = current[problem_data.frozen]
    feasible_upper[problem_data.frozen] = current[problem_data.frozen]
    feasible_lower[problem_data.exact] = post_exact[problem_data.exact]
    feasible_upper[problem_data.exact] = post_exact[problem_data.exact]
    total_lower = float(feasible_lower.sum())
    total_upper = float(feasible_upper.sum())
    if (total_lower > constraints.max_exposure + _EXACT_CONSTRAINT_TOLERANCE
            or total_upper < constraints.min_exposure
            - _EXACT_CONSTRAINT_TOLERANCE):
        causes.append('product_funding_interval')
    details.append(
        f'continuous_funding_interval=[{total_lower:.8f},{total_upper:.8f}],'
        f'required=[{constraints.min_exposure:.8f},'
        f'{constraints.max_exposure:.8f}]'
    )

    remaining = problem_data.tradeable[~exact_mask[problem_data.tradeable]]
    lower_delta = problem_data.effective_lower[remaining] - current[remaining]
    upper_delta = problem_data.effective_upper[remaining] - current[remaining]
    bridge_rows = []
    groups = constraints.group_lower_upper_constraints
    if groups is not None:
        loadings = groups.group_loadings.reindex(
            index=problem_data.index, fill_value=0.0
        ).fillna(0.0)
        minimum = groups.group_min_allocation
        maximum = groups.group_max_allocation
        for group in loadings.columns:
            group_loadings = loadings[group].to_numpy(dtype=float)
            base_exposure = float(group_loadings @ post_exact)
            effect_min, effect_max = _ticket_limited_effect_range(
                loadings=group_loadings[remaining],
                lower_delta=lower_delta,
                upper_delta=upper_delta,
                ticket_budget=remaining_tickets,
                minimum_ticket=config.material_trade_threshold,
            )
            reachable_min = base_exposure + effect_min
            reachable_max = base_exposure + effect_max
            if (minimum is not None and group in minimum.index
                    and np.isfinite(minimum.loc[group])
                    and reachable_max < float(minimum.loc[group])
                    - _EXACT_CONSTRAINT_TOLERANCE):
                bridge = float(minimum.loc[group]) - reachable_max
                bridge_rows.append(
                    f'{group}:min required={minimum.loc[group]:.8f},'
                    f'reachable_max={reachable_max:.8f},bridge_bp={10_000 * bridge:.4f}'
                )
            if (maximum is not None and group in maximum.index
                    and np.isfinite(maximum.loc[group])
                    and reachable_min > float(maximum.loc[group])
                    + _EXACT_CONSTRAINT_TOLERANCE):
                bridge = reachable_min - float(maximum.loc[group])
                bridge_rows.append(
                    f'{group}:max required={maximum.loc[group]:.8f},'
                    f'reachable_min={reachable_min:.8f},bridge_bp={10_000 * bridge:.4f}'
                )
    if bridge_rows:
        causes.append('group_ticket_bridge')
        details.append('group_bridges=' + ';'.join(bridge_rows[:10]))

    cash_loading = -np.ones(len(remaining), dtype=float)
    cash_effect_min, cash_effect_max = _ticket_limited_effect_range(
        loadings=cash_loading,
        lower_delta=lower_delta,
        upper_delta=upper_delta,
        ticket_budget=remaining_tickets,
        minimum_ticket=config.material_trade_threshold,
    )
    cash_reachable_min = post_exact[cash_index] + cash_effect_min
    cash_reachable_max = post_exact[cash_index] + cash_effect_max
    cash_min = problem_data.true_lower[cash_index]
    cash_max = problem_data.true_upper[cash_index]
    if (cash_reachable_max < cash_min - _EXACT_CONSTRAINT_TOLERANCE
            or cash_reachable_min > cash_max + _EXACT_CONSTRAINT_TOLERANCE):
        causes.append('cash_ticket_bridge')
        details.append(
            f'cash_required=[{cash_min:.8f},{cash_max:.8f}],'
            f'cash_reachable=[{cash_reachable_min:.8f},{cash_reachable_max:.8f}]'
        )

    if not causes:
        causes.append('joint_integer_constraint_intersection')
    details.append(
        f'solver=status:{outcome.status},reason:{outcome.reason or "unspecified"},'
        f'branches:{audit.branches},integer_solutions:{audit.integer_solutions},'
        f'relative_gap:{audit.relative_gap:.8g},seconds:{audit.solve_seconds:.4f}'
    )
    return tuple(dict.fromkeys(causes)), '; '.join(details)


def _finite_stat(value: float) -> float:
    """Return a finite solver statistic or NaN when MOSEK leaves it undefined."""
    value = float(value)
    return value if np.isfinite(value) else np.nan


def _classify_mip_termination(
        audit: ExactExecutionSolverAudit,
        exact_config: ExactExecutionConfig,
) -> str:
    """Infer the MOSEK stop reason conservatively from gaps and active limits."""
    gap_reached = (
        np.isfinite(audit.relative_gap)
        and audit.relative_gap <= exact_config.relative_gap_tolerance
    ) or (
        np.isfinite(audit.absolute_gap)
        and audit.absolute_gap <= exact_config.absolute_gap_tolerance
    )
    if gap_reached:
        return 'gap'
    if audit.branches >= exact_config.max_branches:
        return 'branch_limit'
    if (
            exact_config.max_solve_seconds is not None
            and np.isfinite(audit.solve_seconds)
            and audit.solve_seconds >= exact_config.max_solve_seconds - 1.0
    ):
        return 'time_limit'
    return 'other'


def _clean_exact_weight_dust(
        weights: np.ndarray,
        current: np.ndarray,
        cash_index: int,
        clean_trade_dust: bool = True,
) -> tuple[np.ndarray, int]:
    """Zero accepted weight/trade dust and preserve total exposure in cash."""
    cleaned = np.asarray(weights, dtype=float).copy()
    negative_dust = (
        (cleaned < 0.0)
        & (cleaned >= -_EXACT_OUTPUT_ZERO_TOLERANCE)
    )
    negative_dust[cash_index] = False
    trade_dust = np.zeros(cleaned.shape, dtype=bool)
    if clean_trade_dust:
        trade_dust = (
            np.abs(cleaned - np.asarray(current, dtype=float))
            <= _EXACT_CONSTRAINT_TOLERANCE
        ) & ~np.isclose(cleaned, current, atol=0.0, rtol=0.0)
    trade_dust[cash_index] = False
    dust = negative_dust | trade_dust
    count = int(dust.sum())
    if count:
        original_total = float(cleaned.sum())
        cleaned[trade_dust] = current[trade_dust]
        cleaned[negative_dust] = 0.0
        cleaned[cash_index] += original_total - float(cleaned.sum())
    return cleaned, count


def _prepare_exact_problem(
        trade_table: pd.DataFrame,
        config: ExecutionPolicyConfig,
        context: str = '',
) -> _ExactProblemData:
    """Validate and align the stage-one policy table for joint selection."""
    required = {
        ALPHA,
        BASE_WEIGHT,
        CURRENT_WEIGHT,
        EFFECTIVE_SMALL_TRADE_CUTOFF,
        FIXED_INSTRUCTION,
        INSTRUCTION_SOURCE,
        MANDATORY_CATEGORY,
        MANDATORY_TRADE,
        PINNED_INSTRUCTION,
        PRODUCT_MAX_WEIGHT,
        PRODUCT_MIN_WEIGHT,
        RAW_MODEL_WEIGHT,
        REBALANCE_CADENCE_ELIGIBLE,
        SETTLEMENT_CASH,
    }
    missing = required.difference(trade_table.columns)
    if missing:
        raise ValueError(
            f'Exact Execution trade table is missing columns {sorted(missing)}'
        )
    if not trade_table.index.is_unique:
        raise ValueError('Exact Execution trade table index must be unique')
    exact_config = config.exact_execution
    if exact_config is None:
        raise ValueError('Exact Execution requires ExactExecutionConfig')
    if config.max_trades is None:
        raise ValueError('Exact Execution requires a finite max_trades')

    index = trade_table.index
    current = trade_table[CURRENT_WEIGHT].to_numpy(dtype=float).copy()
    model = trade_table[RAW_MODEL_WEIGHT].to_numpy(dtype=float).copy()
    alphas = trade_table[ALPHA].to_numpy(dtype=float).copy()
    true_lower = trade_table[PRODUCT_MIN_WEIGHT].to_numpy(dtype=float).copy()
    true_upper = trade_table[PRODUCT_MAX_WEIGHT].to_numpy(dtype=float).copy()
    holding_floor = trade_table[EFFECTIVE_SMALL_TRADE_CUTOFF].to_numpy(
        dtype=float
    ).copy()
    values = np.concatenate([
        current, model, alphas, true_lower, true_upper, holding_floor,
    ])
    if not np.isfinite(values).all():
        raise ValueError('Exact Execution weights, bounds, floors and alphas must be finite')
    if not np.isclose(current.sum(), 1.0, atol=1e-6, rtol=0.0):
        raise ValueError(
            f'Exact Execution current weights must sum to one, got {current.sum():.12f}'
        )
    invalid_bounds = true_lower > true_upper + _EXACT_CONSTRAINT_TOLERANCE
    if invalid_bounds.any():
        raise ValueError(
            'Exact Execution true lower bound exceeds upper bound for '
            f'{index[invalid_bounds].tolist()}'
        )
    if (holding_floor < 0.0).any():
        raise ValueError('Exact Execution holding floors must be non-negative')

    cash_mask = trade_table[SETTLEMENT_CASH].astype(bool).to_numpy()
    if cash_mask.sum() != 1:
        raise ValueError('Exact Execution requires exactly one settlement-cash row')
    cash_index = int(np.flatnonzero(cash_mask)[0])
    is_inception = EXACT_INCEPTION_CANDIDATE in trade_table.columns
    is_history = EXACT_HISTORY_CANDIDATE in trade_table.columns
    if is_inception or is_history:
        candidate_column = (
            EXACT_INCEPTION_CANDIDATE if is_inception else EXACT_HISTORY_CANDIDATE
        )
        candidate_mask = trade_table[candidate_column].fillna(False).astype(bool).to_numpy()
    else:
        # Live Exact may only trade held/model assets or explicit hard requirements.
        candidate_mask = (
            (np.abs(current) > WEIGHT_ZERO_TOLERANCE)
            | (~is_model_zero(model))
            | trade_table[MANDATORY_TRADE].astype(bool).to_numpy()
            | trade_table[FIXED_INSTRUCTION].astype(bool).to_numpy()
            | trade_table[PINNED_INSTRUCTION].astype(bool).to_numpy()
            | (true_lower > WEIGHT_ZERO_TOLERANCE)
            | cash_mask
        )
    current_max_repair_mask = (
        trade_table[MANDATORY_CATEGORY].eq(INSTRUCTION_MAX_SELL)
        & trade_table.get(MODEL_MAX_REPAIR, pd.Series(False, index=trade_table.index)).astype(bool)
    ).to_numpy()
    exact_mask = (
        trade_table[FIXED_INSTRUCTION].astype(bool).to_numpy()
        | trade_table[PINNED_INSTRUCTION].astype(bool).to_numpy()
        | trade_table[MANDATORY_CATEGORY].isin([MODEL_ENTRY, MODEL_EXIT]).to_numpy()
        | current_max_repair_mask
    )
    exact_targets = trade_table[BASE_WEIGHT].to_numpy(dtype=float).copy()
    forced_zero_mask = (
        ~candidate_mask & ~cash_mask
        if is_inception else np.zeros(len(index), dtype=bool)
    )
    tradeable_mask = candidate_mask & ~cash_mask
    if not is_inception:
        cadence_eligible = trade_table[
            REBALANCE_CADENCE_ELIGIBLE
        ].astype(bool).to_numpy()
        off_cycle_exact = (
            exact_mask
            & candidate_mask
            & ~cadence_eligible
            & ~cash_mask
            & (np.abs(exact_targets - current) > _EXACT_CONSTRAINT_TOLERANCE)
        )
        if off_cycle_exact.any():
            details = '; '.join(
                f'{index[location]}(current={current[location]:.8f}, '
                f'target={exact_targets[location]:.8f}, '
                f'ticket={abs(exact_targets[location] - current[location]):.8f})'
                for location in np.flatnonzero(off_cycle_exact)
            )
            logger.warning(
                '[%s] Exact Execution off-cycle override: mandatory exact '
                'targets supersede rebalance cadence for %s',
                context or 'execution',
                details,
            )
        tradeable_mask &= cadence_eligible | off_cycle_exact
    tradeable = np.flatnonzero(tradeable_mask)
    forced_zero = np.flatnonzero(forced_zero_mask)
    frozen = np.flatnonzero(~tradeable_mask & ~forced_zero_mask & ~cash_mask)
    if not is_inception and not is_history:
        logger.info(
            '[%s] exact_live_candidates selected=%d; excluded=%d',
            context or 'execution',
            len(tradeable),
            len(frozen),
        )
    exact = np.flatnonzero(exact_mask & tradeable_mask)
    excluded_exact = np.flatnonzero(
        exact_mask
        & forced_zero_mask
        & (exact_targets > _EXACT_CONSTRAINT_TOLERANCE)
    )
    if excluded_exact.size:
        raise ValueError(
            'Exact Execution inception candidates exclude positive exact targets '
            f'{index[excluded_exact].tolist()}'
        )
    excluded_minimum = np.flatnonzero(
        forced_zero_mask & (true_lower > _EXACT_CONSTRAINT_TOLERANCE)
    )
    if excluded_minimum.size:
        raise ValueError(
            'Exact Execution inception candidates exclude positive product minima '
            f'{index[excluded_minimum].tolist()}'
        )

    off_cadence_exact = np.flatnonzero(
        exact_mask
        & ~tradeable_mask
        & ~cash_mask
        & (np.abs(exact_targets - current) > _EXACT_CONSTRAINT_TOLERANCE)
    )
    if off_cadence_exact.size:
        raise ValueError(
            'Exact Execution cannot move off-cadence exact targets '
            f'{index[off_cadence_exact].tolist()}'
        )
    exact_trades = np.abs(exact_targets[exact] - current[exact])
    active_exact = exact_trades > _EXACT_CONSTRAINT_TOLERANCE
    minimum_ticket_exempt = np.flatnonzero(
        current_max_repair_mask & tradeable_mask
    )
    mandatory_ticket_count = int(active_exact.sum())
    effective_max_trades = resolve_effective_max_trades(
        config=config,
        mandatory_ticket_count=mandatory_ticket_count,
    )
    if effective_max_trades is None:
        raise ValueError('Exact Execution requires a finite max_trades')
    if mandatory_ticket_count > effective_max_trades:
        raise ValueError(
            f'Exact Execution has {mandatory_ticket_count} exact tickets but '
            f'effective_max_trades={effective_max_trades} '
            f'(configured_max_trades={config.max_trades}, '
            'add_mandatory_trades_to_max_trades='
            f'{config.add_mandatory_trades_to_max_trades})'
        )
    sub_ticket = active_exact & (
        exact_trades < config.material_trade_threshold - _EXACT_CONSTRAINT_TOLERANCE
    ) & ~np.isin(exact, minimum_ticket_exempt)
    if sub_ticket.any():
        locations = exact[sub_ticket]
        diagnostic = _format_sub_ticket_details(
            trade_table=trade_table,
            locations=locations,
            current=current,
            targets=exact_targets,
            minimum_ticket=config.material_trade_threshold,
        )
        _log_exact_failure(
            context=context,
            stage='preflight',
            status='invalid',
            cause_codes=('exact_target_sub_minimum_ticket',),
            diagnostic=diagnostic,
        )
        raise ValueError(
            'Exact Execution exact targets require sub-minimum tickets for '
            f'{index[locations].tolist()}; {diagnostic}'
        )
    exact_holdings = exact_targets[exact]
    invalid_holding = (
        active_exact
        & (exact_holdings > _EXACT_CONSTRAINT_TOLERANCE)
        & (exact_holdings < holding_floor[exact] - _EXACT_CONSTRAINT_TOLERANCE)
    )
    if invalid_holding.any():
        raise ValueError(
            'Exact Execution exact targets require sub-floor holdings for '
            f'{index[exact[invalid_holding]].tolist()}'
        )

    effective_lower = np.minimum(true_lower, current)
    effective_upper = np.maximum(true_upper, current)
    # Settlement cash is the unticketed funding residual and must remain inside
    # its true product band; only non-cash retained breaches are grandfathered.
    effective_lower[cash_index] = true_lower[cash_index]
    effective_upper[cash_index] = true_upper[cash_index]
    effective_lower[forced_zero] = 0.0
    effective_upper[forced_zero] = 0.0
    frozen_held = int((current[frozen] > WEIGHT_ZERO_TOLERANCE).sum())
    if frozen_held > exact_config.max_positions:
        raise ValueError(
            f'Exact Execution has {frozen_held} held off-cadence positions but '
            f'max_positions={exact_config.max_positions}'
        )
    return _ExactProblemData(
        index=index,
        current=current,
        model=model,
        alphas=alphas,
        true_lower=true_lower,
        true_upper=true_upper,
        effective_lower=effective_lower,
        effective_upper=effective_upper,
        holding_floor=holding_floor,
        cash_index=cash_index,
        tradeable=tradeable,
        frozen=frozen,
        forced_zero=forced_zero,
        exact=exact,
        exact_targets=exact_targets,
        minimum_ticket_exempt=minimum_ticket_exempt,
        mandatory_ticket_count=mandatory_ticket_count,
        effective_max_trades=effective_max_trades,
    )


def _build_exact_constraints(
        base_constraints: Constraints,
        problem_data: _ExactProblemData,
) -> Constraints:
    """Return the full production constraint object with retained-breach bounds."""
    beta_constraint = base_constraints.benchmark_beta_constraint
    if beta_constraint is not None and beta_constraint.beta_loadings is None:
        raise ValueError('Exact Execution requires resolved benchmark beta loadings')
    return base_constraints.copy(
        min_weights=pd.Series(
            problem_data.effective_lower, index=problem_data.index
        ),
        max_weights=pd.Series(
            problem_data.effective_upper, index=problem_data.index
        ),
        benchmark_weights=pd.Series(problem_data.model, index=problem_data.index),
        weights_0=pd.Series(problem_data.current, index=problem_data.index),
        tracking_err_vol_constraint=None,
        turnover_constraint=None,
        tre_utility_weight=None,
        turnover_utility_weight=None,
        group_tracking_error_constraint=None,
        group_turnover_constraint=None,
        constraint_enforcement_type=ConstraintEnforcementType.FORCED_CONSTRAINTS,
    )


def _build_exact_inception_constraints(
        base_constraints: Constraints,
        problem_data: _ExactProblemData,
) -> Constraints:
    """Return full production constraints with true inception product bands."""
    beta_constraint = base_constraints.benchmark_beta_constraint
    if beta_constraint is not None and beta_constraint.beta_loadings is None:
        raise ValueError('Exact Execution requires resolved benchmark beta loadings')
    return base_constraints.copy(
        min_weights=pd.Series(problem_data.true_lower, index=problem_data.index),
        max_weights=pd.Series(problem_data.true_upper, index=problem_data.index),
        benchmark_weights=pd.Series(problem_data.model, index=problem_data.index),
        weights_0=pd.Series(problem_data.current, index=problem_data.index),
        tracking_err_vol_constraint=None,
        turnover_constraint=None,
        tre_utility_weight=None,
        turnover_utility_weight=None,
        group_tracking_error_constraint=None,
        group_turnover_constraint=None,
        constraint_enforcement_type=ConstraintEnforcementType.FORCED_CONSTRAINTS,
    )


def _partition_soft_group_constraints(
        base_constraints: Constraints,
        soft_group_names: Optional[tuple[object, ...]],
) -> tuple[Constraints, Optional[GroupLowerUpperConstraints]]:
    """Remove only named level-2 bounds from the hard constraint set."""
    group_bounds = base_constraints.group_lower_upper_constraints
    if group_bounds is None or not soft_group_names:
        raise ValueError(
            'exact_soft execution requires named sub-asset-class constraints'
        )
    soft_names = [
        name for name in soft_group_names
        if name in group_bounds.group_loadings.columns
    ]
    if not soft_names:
        raise ValueError(
            'exact_soft execution found no sub-asset-class constraints to soften'
        )
    hard_names = group_bounds.group_loadings.columns.difference(
        soft_names, sort=False
    )

    def subset(names: pd.Index | list[object]) -> GroupLowerUpperConstraints:
        """Return group bounds restricted to the requested columns."""
        return GroupLowerUpperConstraints(
            group_loadings=group_bounds.group_loadings.loc[:, names],
            group_min_allocation=(
                None if group_bounds.group_min_allocation is None
                else group_bounds.group_min_allocation.reindex(names)
            ),
            group_max_allocation=(
                None if group_bounds.group_max_allocation is None
                else group_bounds.group_max_allocation.reindex(names)
            ),
        )

    hard_bounds = subset(hard_names) if len(hard_names) else None
    return (
        base_constraints.copy(group_lower_upper_constraints=hard_bounds),
        subset(soft_names),
    )


def _soft_group_objective(
        w: cvx.Variable,
        group_bounds: GroupLowerUpperConstraints,
) -> cvx.Expression:
    """Return squared level-2 bound violations in portfolio-weight units."""
    exposure = group_bounds.group_loadings.to_numpy(dtype=float).T @ w
    terms = []
    for allocations, sign in (
            (group_bounds.group_min_allocation, -1.0),
            (group_bounds.group_max_allocation, 1.0)):
        if allocations is None:
            continue
        values = allocations.to_numpy(dtype=float)
        locations = np.flatnonzero(np.isfinite(values))
        if locations.size:
            violation = sign * (exposure[locations] - values[locations])
            terms.append(cvx.sum_squares(cvx.pos(violation)))
    if not terms:
        raise ValueError('exact_soft sub-asset-class bounds are all blank')
    return sum(terms)


def _audit_soft_group_solution(
        weights: np.ndarray,
        group_bounds: GroupLowerUpperConstraints,
) -> tuple[ConstraintResidual, ...]:
    """Report each softened level-2 bound without treating it as hard."""
    exposures = group_bounds.group_loadings.to_numpy(dtype=float).T @ weights
    residuals = []
    for location, name in enumerate(group_bounds.group_loadings.columns):
        lower = (
            None if group_bounds.group_min_allocation is None
            or pd.isna(group_bounds.group_min_allocation.iloc[location])
            else float(group_bounds.group_min_allocation.iloc[location])
        )
        upper = (
            None if group_bounds.group_max_allocation is None
            or pd.isna(group_bounds.group_max_allocation.iloc[location])
            else float(group_bounds.group_max_allocation.iloc[location])
        )
        residuals.append(_constraint_residual(
            'soft_sub_asset_class', str(name), float(exposures[location]),
            lower=lower, upper=upper, hard=False,
        ))
    return tuple(residuals)


def _solve_mosek_with_audit(
        problem: cvx.Problem,
        exact_config: ExactExecutionConfig,
) -> ExactExecutionSolverAudit:
    """Solve through CVXPY while retaining MOSEK's MIP gap and branch counters."""
    if exact_config.solver not in cvx.installed_solvers():
        raise RuntimeError(
            'Exact Execution requires a licensed MOSEK installation visible to CVXPY'
        )
    try:
        import mosek
    except ImportError as error:
        raise RuntimeError(
            'Exact Execution requires the optional MOSEK Python package'
        ) from error

    solver_options = {
        'mosek_params': {
            'MSK_DPAR_MIO_TOL_REL_GAP': exact_config.relative_gap_tolerance,
            'MSK_DPAR_MIO_TOL_ABS_GAP': exact_config.absolute_gap_tolerance,
            'MSK_IPAR_MIO_MAX_NUM_BRANCHES': exact_config.max_branches,
            'MSK_IPAR_MIO_SEED': exact_config.seed,
        },
        'accept_unknown': True,
    }
    if exact_config.max_solve_seconds is not None:
        solver_options['mosek_params']['MSK_DPAR_MIO_MAX_TIME'] = (
            exact_config.max_solve_seconds
        )
    try:
        data, chain, inverse_data = problem.get_problem_data(exact_config.solver)
        with warnings.catch_warnings():
            # CVXPY supplies valid sparse index arrays that MOSEK copies into
            # its preferred native format. Suppress only that harmless adapter
            # warning; all solver, licence and numerical warnings remain live.
            warnings.filterwarnings(
                'ignore',
                message=(
                    r'Argument sub[ij] in putaijlist64: Incorrect array '
                    r'format causing data to be copied'
                ),
                category=UserWarning,
            )
            raw_solution = chain.solve_via_data(
                problem=problem,
                data=data,
                warm_start=True,
                verbose=False,
                solver_opts=solver_options,
            )
        task = raw_solution['task']
        audit = ExactExecutionSolverAudit(
            relative_gap=_finite_stat(
                task.getdouinf(mosek.dinfitem.mio_obj_rel_gap)
            ),
            absolute_gap=_finite_stat(
                task.getdouinf(mosek.dinfitem.mio_obj_abs_gap)
            ),
            branches=int(task.getintinf(mosek.iinfitem.mio_num_branch)),
            integer_solutions=int(
                task.getintinf(mosek.iinfitem.mio_num_int_solutions)
            ),
            incumbent_objective=_finite_stat(
                task.getdouinf(mosek.dinfitem.mio_obj_int)
            ),
            objective_bound=_finite_stat(
                task.getdouinf(mosek.dinfitem.mio_obj_bound)
            ),
            solve_seconds=_finite_stat(
                task.getdouinf(mosek.dinfitem.optimizer_time)
            ),
        )
        audit = replace(
            audit,
            termination=_classify_mip_termination(
                audit=audit,
                exact_config=exact_config,
            ),
        )
        solution = chain.invert(raw_solution, inverse_data)
        problem.unpack(solution)
        return audit
    except Exception as error:
        raise RuntimeError(f'Exact Execution MOSEK solve failed: {error}') from error


def _constraint_residual(
        constraint_type: str,
        name: str,
        actual: float,
        lower: Optional[float] = None,
        upper: Optional[float] = None,
        tolerance: float = _EXACT_CONSTRAINT_TOLERANCE,
        hard: bool = True,
) -> ConstraintResidual:
    """Build one independent post-solve constraint audit row."""
    violation = 0.0
    if lower is not None:
        violation = max(violation, float(lower) - float(actual))
    if upper is not None:
        violation = max(violation, float(actual) - float(upper))
    return ConstraintResidual(
        constraint_type=constraint_type,
        name=name,
        actual=float(actual),
        lower=None if lower is None else float(lower),
        upper=None if upper is None else float(upper),
        violation=max(0.0, violation),
        tolerance=float(tolerance),
        hard=hard,
        passed=violation <= tolerance,
    )


def _audit_exact_solution(
        weights: np.ndarray,
        problem_data: _ExactProblemData,
        config: ExecutionPolicyConfig,
) -> tuple[ConstraintResidual, ...]:
    """Audit ticket, holding, cadence, exact-target and repair guarantees."""
    exact_config = config.exact_execution
    if exact_config is None or config.max_trades is None:
        raise ValueError('Exact Execution audit requires complete configuration')
    current = problem_data.current
    cash_index = problem_data.cash_index
    trades = np.abs(weights - current)
    trades[cash_index] = 0.0
    # MOSEK's conic MIP solution may leave a mathematical zero at a few 1e-8.
    # Use the exact hard-audit tolerance so such solver noise is not reported
    # as a sub-floor position while genuine tickets remain orders of magnitude
    # larger under the configured material-trade threshold.
    traded = trades > _EXACT_CONSTRAINT_TOLERANCE
    held = weights > _EXACT_CONSTRAINT_TOLERANCE
    held[cash_index] = False
    traded_held = traded & held
    materiality_trades = traded.copy()
    materiality_trades[problem_data.minimum_ticket_exempt] = False
    minimum_ticket = (
        float(trades[materiality_trades].min())
        if materiality_trades.any() else config.material_trade_threshold
    )
    holding_slack = (
        float((weights[traded_held] - problem_data.holding_floor[traded_held]).min())
        if traded_held.any() else 0.0
    )
    frozen_delta = (
        float(np.abs(weights[problem_data.frozen] - current[problem_data.frozen]).max())
        if problem_data.frozen.size else 0.0
    )
    exact_delta = (
        float(np.abs(
            weights[problem_data.exact]
            - problem_data.exact_targets[problem_data.exact]
        ).max())
        if problem_data.exact.size else 0.0
    )
    touch_violation = 0.0
    if exact_config.repair_on_touch and traded.any():
        touch_violation = float(max(
            np.max(np.maximum(
                problem_data.true_lower[traded] - weights[traded], 0.0
            )),
            np.max(np.maximum(
                weights[traded] - problem_data.true_upper[traded], 0.0
            )),
        ))
    return (
        _constraint_residual(
            'execution_ticket_count', 'non_cash_tickets',
            actual=float(traded.sum()),
            upper=float(problem_data.effective_max_trades),
        ),
        _constraint_residual(
            'execution_position_count', 'non_cash_positions',
            actual=float(held.sum()), upper=float(exact_config.max_positions),
        ),
        _constraint_residual(
            'execution_minimum_ticket', 'smallest_ticket',
            actual=minimum_ticket, lower=config.material_trade_threshold,
        ),
        _constraint_residual(
            'execution_holding_floor', 'worst_traded_holding_slack',
            actual=holding_slack, lower=0.0,
        ),
        _constraint_residual(
            'execution_cadence_freeze', 'largest_off_cadence_trade',
            actual=frozen_delta, upper=0.0,
        ),
        _constraint_residual(
            'execution_exact_target', 'largest_exact_target_error',
            actual=exact_delta, upper=0.0,
        ),
        _constraint_residual(
            'execution_repair_on_touch', 'largest_true_bound_violation',
            actual=touch_violation, upper=0.0,
        ),
    )


def _enrich_exact_trade_table(
        trade_table: pd.DataFrame,
        weights: pd.Series,
        problem_data: _ExactProblemData,
        config: ExecutionPolicyConfig,
        audit: ExactExecutionSolverAudit,
        outcome: OptimizationOutcome,
) -> pd.DataFrame:
    """Return a report-compatible table carrying exact-selection diagnostics."""
    enriched = trade_table.copy()
    index = enriched.index
    current = pd.Series(problem_data.current, index=index)
    model = pd.Series(problem_data.model, index=index)
    exact = pd.Series(False, index=index)
    exact.iloc[problem_data.exact] = True
    executed = weights - current
    selected = executed.abs().gt(WEIGHT_ZERO_TOLERANCE)
    selected.iloc[problem_data.cash_index] = False

    base = current.copy()
    base.loc[exact] = pd.Series(problem_data.exact_targets, index=index).loc[exact]
    cash_name = index[problem_data.cash_index]
    base.loc[cash_name] = current.loc[cash_name] - float(
        (base - current).drop(index=cash_name).sum()
    )
    mandatory_funding = base - current
    mandatory = exact & mandatory_funding.abs().gt(WEIGHT_ZERO_TOLERANCE)
    rule4 = pd.Series(False, index=index)
    rule4.iloc[problem_data.tradeable] = True
    rule4 &= ~exact

    enriched[EXACT_EXECUTION] = True
    enriched[EXACT_TARGET] = exact
    enriched[BASE_WEIGHT] = base
    enriched[EFFECTIVE_MODEL_WEIGHT] = model
    enriched[POLICY_TARGET_WEIGHT] = model
    enriched[TRADE_WEIGHT] = model - current
    enriched[LOW_WEIGHT_EXCLUDED] = False
    enriched[DESIRED_REWEIGHT] = (model - base).where(rule4, 0.0)
    enriched[MANDATORY_TRADE] = mandatory
    enriched[MANDATORY_FUNDING] = mandatory_funding.where(mandatory, 0.0)
    enriched.loc[cash_name, MANDATORY_FUNDING] = -float(
        enriched.loc[exact, MANDATORY_FUNDING].sum()
    )
    enriched.loc[~exact, MANDATORY_CATEGORY] = ''
    enriched.loc[~exact, MANDATORY_REASON] = ''
    enriched[ENTRY_FUNDING] = 0.0
    enriched[EXIT_FUNDING] = 0.0
    enriched[DESK_FUNDING] = 0.0
    enriched[OTHER_FUNDING] = 0.0
    enriched[RULE4_TRADE] = rule4
    enriched[TRADE_CANDIDATE] = rule4
    enriched[SELECTED_TRADE] = selected
    enriched[REQUESTED_TRADE] = selected
    enriched[MATERIAL_TRADE] = selected
    enriched[FEASIBILITY_RESCUE_TRADE] = False
    enriched[DIRECTED_RESCUE_CANDIDATE] = False
    enriched[RESCUE_REASON] = ''
    enriched[CUTOFF_SELL_DOWN] = False
    enriched[CUTOFF_SELL_DOWN_RETENTION] = np.nan
    enriched[CUTOFF_SELL_DOWN_RETAINED_FRACTION] = np.nan
    enriched[CUTOFF_SELL_DOWN_DUST_ROUNDED] = False

    enriched[PROPOSED_WEIGHT] = weights
    enriched[EXECUTED_TRADE] = executed
    enriched[MANDATORY_EXECUTED_TRADE] = executed.where(mandatory, 0.0)
    enriched[DISCRETIONARY_EXECUTED_TRADE] = executed.where(~mandatory, 0.0)
    enriched[TARGET_GAP] = weights - model
    enriched[STRICT_CORRIDOR_MIN] = problem_data.effective_lower
    enriched[STRICT_CORRIDOR_MAX] = problem_data.effective_upper
    enriched[MANDATORY_CORRIDOR] = False
    enriched[MANDATORY_CORRIDOR_MIN] = np.nan
    enriched[MANDATORY_CORRIDOR_MAX] = np.nan
    enriched[MANDATORY_CORRIDOR_EXTRA_TRADE] = 0.0
    enriched[CORRIDOR_RELAXATION] = 0.0
    enriched[RELAXED_CORRIDOR_SOLVE] = False

    reweight_funding = float((weights - base).drop(index=cash_name).sum())
    enriched[REWEIGHT_FUNDING] = 0.0
    enriched.loc[cash_name, REWEIGHT_FUNDING] = reweight_funding
    enriched[EXPECTED_CASH_WEIGHT] = np.nan
    enriched.loc[cash_name, EXPECTED_CASH_WEIGHT] = (
        current.loc[cash_name] - float(executed.drop(index=cash_name).sum())
    )

    lower_breach = pd.Series(problem_data.true_lower, index=index) - weights
    upper_breach = weights - pd.Series(problem_data.true_upper, index=index)
    retained_breach = lower_breach.clip(lower=0.0) + upper_breach.clip(lower=0.0)
    enriched[PRODUCT_BOUND_OVERRIDE_BREACH] = retained_breach
    enriched[PRODUCT_BOUND_OVERRIDE] = retained_breach.gt(
        _EXACT_CONSTRAINT_TOLERANCE
    )
    enriched[PRODUCT_BOUND_OVERRIDE_REASON] = np.where(
        enriched[PRODUCT_BOUND_OVERRIDE], 'exact_execution_retained_breach', ''
    )

    enriched[MIP_RELATIVE_GAP] = audit.relative_gap
    enriched[MIP_ABSOLUTE_GAP] = audit.absolute_gap
    enriched[MIP_BRANCHES] = audit.branches
    enriched[MIP_INTEGER_SOLUTIONS] = audit.integer_solutions
    enriched[MIP_INCUMBENT_OBJECTIVE] = audit.incumbent_objective
    enriched[MIP_OBJECTIVE_BOUND] = audit.objective_bound
    enriched[MIP_SOLVE_SECONDS] = audit.solve_seconds
    enriched[MIP_TERMINATION] = audit.termination
    enriched[MIP_ACCEPTANCE_BASIS] = (
        'proven_optimal'
        if outcome.accepted and outcome.compliant and audit.termination == 'gap'
        else 'audited_incumbent'
        if outcome.accepted and outcome.compliant
        else 'rejected'
    )
    enriched[SUB_ASSET_CLASS_PENALTY] = (
        config.exact_execution.sub_asset_class_penalty
        if config.exact_execution is not None else np.nan
    )
    return enriched


def _solve_exact_continuous_problem(problem, exact_config) -> None:
    """Polish fixed MIP selections with the same solver and tighter primal precision."""
    if problem.is_mixed_integer():
        raise ValueError('Numerical recovery must fix every integer selection')
    params = {
        'MSK_DPAR_INTPNT_CO_TOL_PFEAS': 1e-10,
        'MSK_DPAR_INTPNT_CO_TOL_DFEAS': 1e-10,
        'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': 1e-10,
    }
    if exact_config.max_solve_seconds is not None:
        params['MSK_DPAR_OPTIMIZER_MAX_TIME'] = exact_config.max_solve_seconds
    problem.solve(solver=exact_config.solver, mosek_params=params, verbose=False)


def _format_exact_audit_failure(weights, problem_data, residuals) -> str:
    """Include violated limits and offending holdings in persisted failure reasons."""
    details = [
        f'{r.name}(actual={r.actual:.12g},lower={r.lower},upper={r.upper},'
        f'violation={r.violation:.12g},tolerance={r.tolerance:.12g})'
        for r in residuals if r.hard and not r.passed
    ]
    delta = np.abs(weights - problem_data.current)
    bad_holding = (
        (delta > _EXACT_CONSTRAINT_TOLERANCE)
        & (weights > _EXACT_CONSTRAINT_TOLERANCE)
        & (weights < problem_data.holding_floor - _EXACT_CONSTRAINT_TOLERANCE)
    )
    bad_holding[problem_data.cash_index] = False
    for i in np.flatnonzero(bad_holding):
        details.append(
            f'{problem_data.index[i]}(weight={weights[i]:.12g},'
            f'current={problem_data.current[i]:.12g},ticket={delta[i]:.12g},'
            f'holding_floor={problem_data.holding_floor[i]:.12g})'
        )
    return '; '.join(details)


def solve_exact_execution_portfolio(
        trade_table: pd.DataFrame,
        base_constraints: Constraints,
        covariance: pd.DataFrame,
        config: ExecutionPolicyConfig,
        soft_group_names: Optional[tuple[object, ...]] = None,
        context: str = '',
) -> ExecutionOptimizationResult:
    """Jointly choose holdings and tickets with optional level-2 penalties.

    Args:
        trade_table: Stage-one policy and workbook audit in model-name order.
        base_constraints: Complete production TAA constraint object.
        covariance: Annualized covariance in trade-table order.
        config: Policy with a non-null :class:`ExactExecutionConfig`.
        soft_group_names: Level-2 group names eligible for soft treatment.
        context: Solve label included in structured diagnostics.

    Returns:
        Exact weights, report-compatible trade table and independent hard audit.
    """
    problem_data = _prepare_exact_problem(
        trade_table=trade_table,
        config=config,
        context=context,
    )
    exact_config = config.exact_execution
    aligned_covar = covariance.reindex(
        index=problem_data.index, columns=problem_data.index
    )
    sigma = aligned_covar.to_numpy(dtype=float)
    if not np.isfinite(sigma).all():
        raise ValueError('Exact Execution covariance must be finite and complete')
    if not np.allclose(sigma, sigma.T, atol=1e-10, rtol=0.0):
        raise ValueError('Exact Execution covariance must be symmetric')
    factorization = factorize_covariance(sigma)
    solver_covar = cvx.psd_wrap(factorization.covar)
    is_inception = EXACT_INCEPTION_CANDIDATE in trade_table.columns
    is_history = EXACT_HISTORY_CANDIDATE in trade_table.columns
    constraint_builder = (
        _build_exact_inception_constraints
        if is_inception else _build_exact_constraints
    )
    soft_group_bounds = None
    exact_base_constraints = base_constraints
    if exact_config.sub_asset_class_penalty is not None:
        exact_base_constraints, soft_group_bounds = _partition_soft_group_constraints(
            base_constraints=base_constraints,
            soft_group_names=soft_group_names,
        )
        logger.info(
            '[%s] exact_soft groups=%d; penalty=%g',
            context or 'execution',
            len(soft_group_bounds.group_loadings.columns),
            exact_config.sub_asset_class_penalty,
        )
    constraints = constraint_builder(
        base_constraints=exact_base_constraints,
        problem_data=problem_data,
    )

    selections = {}

    def build_problem(fixed_selections=None):
        """Build the same formulation with binary variables or fixed constants."""
        def selector(name):
            """Retain each MIP decision for a continuous fixed-selection retry."""
            if fixed_selections is not None:
                return cvx.Constant(fixed_selections[name])
            variable = cvx.Variable(
                len(problem_data.tradeable), boolean=True, name=f'exact_{name}'
            )
            selections[name] = variable
            return variable

        n = len(problem_data.index)
        n_tradeable = len(problem_data.tradeable)
        w = cvx.Variable(n, name='exact_weights')
        hold = selector('hold')
        tradeable = problem_data.tradeable
        cash_index = problem_data.cash_index
        ticket_floor = np.full(n_tradeable, config.material_trade_threshold)
        ticket_floor[np.isin(
            tradeable, problem_data.minimum_ticket_exempt
        )] = 0.0

        constraints_ = constraints.set_cvx_all_constraints(
            w=w,
            covar=solver_covar,
            covar_factorization=factorization,
        )
        if problem_data.frozen.size:
            constraints_.append(
                w[problem_data.frozen] == problem_data.current[problem_data.frozen]
            )
        if problem_data.forced_zero.size:
            constraints_.append(w[problem_data.forced_zero] == 0.0)
        if is_inception:
            constraints_ += [
                w[tradeable] <= cvx.multiply(
                    problem_data.true_upper[tradeable], hold
                ),
                w[tradeable] >= cvx.multiply(
                    problem_data.holding_floor[tradeable], hold
                ),
                cvx.sum(hold) <= exact_config.max_positions,
            ]
        elif is_history:
            trade = selector('trade')
            target = problem_data.model.copy()
            target[problem_data.exact] = problem_data.exact_targets[
                problem_data.exact
            ]
            # A continuous-model numerical zero must allow a full exit. Otherwise
            # the target corridor and hold=0 contradict each other below the
            # solver tolerance. Explicit instructions and positive minima survive.
            numerical_zero = (
                (np.abs(target) <= _EXACT_OUTPUT_ZERO_TOLERANCE)
                & (problem_data.true_lower <= 0.0)
            )
            numerical_zero[problem_data.exact] = False
            target[numerical_zero] = 0.0
            direction = np.sign(target[tradeable] - problem_data.current[tradeable])
            maximum_progress = np.abs(
                target[tradeable] - problem_data.current[tradeable]
            )
            progress = cvx.multiply(
                direction,
                w[tradeable] - problem_data.current[tradeable],
            )
            constraints_ += [
                progress >= cvx.multiply(ticket_floor, trade),
                progress <= cvx.multiply(maximum_progress, trade),
                w[tradeable] <= cvx.multiply(
                    problem_data.effective_upper[tradeable], hold
                ),
                w[tradeable] >= cvx.multiply(
                    problem_data.holding_floor[tradeable], hold - (1 - trade)
                ),
                hold <= (
                    (problem_data.current[tradeable] > WEIGHT_ZERO_TOLERANCE).astype(float)
                    + trade
                ),
                cvx.sum(trade) <= problem_data.effective_max_trades,
                cvx.sum(hold) + int((
                    problem_data.current[problem_data.frozen] > WEIGHT_ZERO_TOLERANCE
                ).sum()) <= exact_config.max_positions,
            ]
            zero_direction = np.flatnonzero(
                maximum_progress <= _EXACT_CONSTRAINT_TOLERANCE
            )
            if zero_direction.size:
                constraints_.append(
                    w[tradeable[zero_direction]]
                    == problem_data.current[tradeable[zero_direction]]
                )
            if exact_config.repair_on_touch:
                constraints_ += [
                    w[tradeable] <= problem_data.true_upper[tradeable] + cvx.multiply(
                        np.maximum(
                            problem_data.current[tradeable]
                            - problem_data.true_upper[tradeable],
                            0.0,
                        ),
                        1 - trade,
                    ),
                    w[tradeable] >= problem_data.true_lower[tradeable] - cvx.multiply(
                        np.maximum(
                            problem_data.true_lower[tradeable]
                            - problem_data.current[tradeable],
                            0.0,
                        ),
                        1 - trade,
                    ),
                ]
        else:
            buy = selector('buy')
            sell = selector('sell')
            buy_size = cvx.Variable(n_tradeable, nonneg=True)
            sell_size = cvx.Variable(n_tradeable, nonneg=True)
            tickets = buy + sell
            constraints_ += [
                w[tradeable] - problem_data.current[tradeable] == buy_size - sell_size,
                buy_size <= cvx.multiply(
                    problem_data.effective_upper[tradeable]
                    - problem_data.current[tradeable],
                    buy,
                ),
                sell_size <= cvx.multiply(
                    problem_data.current[tradeable]
                    - problem_data.effective_lower[tradeable],
                    sell,
                ),
                buy_size >= cvx.multiply(ticket_floor, buy),
                sell_size >= cvx.multiply(ticket_floor, sell),
                tickets <= 1,
                w[tradeable] <= cvx.multiply(
                    problem_data.effective_upper[tradeable], hold
                ),
                w[tradeable] >= cvx.multiply(
                    problem_data.holding_floor[tradeable], hold - (1 - tickets)
                ),
                hold <= (
                    (problem_data.current[tradeable] > WEIGHT_ZERO_TOLERANCE).astype(float)
                    + tickets
                ),
                cvx.sum(tickets) <= problem_data.effective_max_trades,
                cvx.sum(hold) + int((
                    problem_data.current[problem_data.frozen] > WEIGHT_ZERO_TOLERANCE
                ).sum()) <= exact_config.max_positions,
            ]
            if exact_config.repair_on_touch:
                constraints_ += [
                    w[tradeable] <= problem_data.true_upper[tradeable] + cvx.multiply(
                        np.maximum(
                            problem_data.current[tradeable]
                            - problem_data.true_upper[tradeable],
                            0.0,
                        ),
                        1 - tickets,
                    ),
                    w[tradeable] >= problem_data.true_lower[tradeable] - cvx.multiply(
                        np.maximum(
                            problem_data.true_lower[tradeable]
                            - problem_data.current[tradeable],
                            0.0,
                        ),
                        1 - tickets,
                    ),
                ]
        if problem_data.exact.size:
            constraints_.append(
                w[problem_data.exact]
                == problem_data.exact_targets[problem_data.exact]
            )

        active = w - problem_data.model
        objective_value = cvx_covar_variance(
            active_weights=active,
            covar=solver_covar,
            covar_factorization=factorization,
        )
        if exact_config.alpha_tradeoff > 0.0:
            cash_alpha = problem_data.alphas[cash_index]
            objective_value -= exact_config.alpha_tradeoff * (
                (problem_data.alphas - cash_alpha) @ (w - problem_data.current)
            )
        if soft_group_bounds is not None:
            objective_value += (
                exact_config.sub_asset_class_penalty
                * _soft_group_objective(w=w, group_bounds=soft_group_bounds)
            )
        return cvx.Problem(cvx.Minimize(objective_value), constraints_), w
    problem, w = build_problem()
    n = len(problem_data.index)
    cash_index = problem_data.cash_index
    audit = _solve_mosek_with_audit(problem=problem, exact_config=exact_config)
    accept_inaccurate = (
        exact_config.accept_audited_incumbent
        and audit.integer_solutions > 0
    )
    outcome: OptimizationOutcome = validate_solution(
        optimal_weights=None if w.value is None else np.asarray(w.value).ravel(),
        problem_status=problem.status,
        constraints=constraints,
        n=n,
        solver=exact_config.solver,
        context=context,
        covar=factorization.covar,
        covar_factorization=factorization,
        accept_inaccurate=accept_inaccurate,
    )
    if outcome.accepted:
        cleaned_weights, cleaned_count = _clean_exact_weight_dust(
            weights=outcome.weights,
            current=problem_data.current,
            cash_index=cash_index,
            clean_trade_dust=not is_inception,
        )
        if cleaned_count:
            logger.info(
                '[%s] exact_weight_dust_cleaned instruments=%d; '
                'zero_tolerance=%.2e',
                context,
                cleaned_count,
                _EXACT_OUTPUT_ZERO_TOLERANCE,
            )
            outcome = validate_solution(
                optimal_weights=cleaned_weights,
                problem_status=problem.status,
                constraints=constraints,
                n=n,
                solver=exact_config.solver,
                context=f'{context} cleaned numerical zeros',
                covar=factorization.covar,
                covar_factorization=factorization,
                accept_inaccurate=accept_inaccurate,
            )
    if not outcome.accepted:
        if (
                not is_inception
                and problem.status in {cvx.INFEASIBLE, cvx.INFEASIBLE_INACCURATE}
        ):
            cause_codes, diagnostic = _diagnose_exact_infeasibility(
                problem_data=problem_data,
                constraints=constraints,
                config=config,
                audit=audit,
                outcome=outcome,
            )
        else:
            cause_codes = (
                ('inception_constraint_intersection',)
                if is_inception else ('solver_rejected',)
            )
            diagnostic = (
                f'solver=status:{outcome.status},'
                f'reason:{outcome.reason or "unspecified"},'
                f'branches:{audit.branches},'
                f'integer_solutions:{audit.integer_solutions},'
                f'relative_gap:{audit.relative_gap:.8g},'
                f'seconds:{audit.solve_seconds:.4f}'
            )
        _log_exact_failure(
            context=context,
            stage='solver',
            status=str(outcome.status or problem.status or 'unknown'),
            cause_codes=cause_codes,
            diagnostic=diagnostic,
        )
    weights = pd.Series(outcome.weights, index=problem_data.index, name='exact_execution')
    exact_residuals = _audit_exact_solution(
        weights=weights.to_numpy(dtype=float),
        problem_data=problem_data,
        config=config,
    )
    numerical_recovery = False
    failed_audit = [r for r in exact_residuals if r.hard and not r.passed]
    if outcome.accepted and failed_audit:
        diagnostic = _format_exact_audit_failure(
            weights.to_numpy(dtype=float), problem_data, failed_audit
        )
        _log_exact_failure(
            context, 'audit', str(outcome.status),
            tuple(r.name for r in failed_audit), diagnostic,
        )
        fixed = {}
        for name, variable in selections.items():
            values = variable.value
            if (values is None or not np.isfinite(values).all()
                    or np.max(np.abs(values - np.rint(values))) > 1e-4):
                break
            fixed[name] = np.rint(values)
        if fixed and len(fixed) == len(selections):
            try:
                retry_problem, retry_weights = build_problem(fixed)
                _solve_exact_continuous_problem(retry_problem, exact_config)
                candidate = validate_solution(
                    optimal_weights=(None if retry_weights.value is None
                                     else np.asarray(retry_weights.value).ravel()),
                    problem_status=retry_problem.status, constraints=constraints,
                    n=n, solver=exact_config.solver, context=f'{context} numerical recovery',
                    covar=factorization.covar, covar_factorization=factorization,
                )
                recovered_audit = _audit_exact_solution(
                    candidate.weights, problem_data, config
                )
                adjustment = float(np.max(np.abs(candidate.weights - weights.to_numpy())))
                formulation_ok = candidate.accepted and all(
                    np.max(np.asarray(constraint.violation())) <= _EXACT_CONSTRAINT_TOLERANCE
                    for constraint in retry_problem.constraints
                )
                # At most 0.1 bp per weight: a numerical repair cannot replace
                # a substantively different trade or relax an execution rule.
                if (formulation_ok and candidate.compliant and adjustment <= 1e-5
                        and all(r.passed for r in recovered_audit if r.hard)):
                    outcome = candidate
                    weights = pd.Series(candidate.weights, index=problem_data.index,
                                        name='exact_execution')
                    exact_residuals = recovered_audit
                    numerical_recovery = True
                    logger.info(
                        '[%s] exact_numerical_recovery accepted; max_weight_adjustment=%.12g',
                        context, adjustment,
                    )
                else:
                    logger.warning(
                        '[%s] exact_numerical_recovery rejected; status=%s; '
                        'max_weight_adjustment=%.12g; formulation_ok=%s',
                        context, retry_problem.status, adjustment, formulation_ok,
                    )
            except (cvx.error.SolverError, ValueError, RuntimeError) as error:
                logger.warning('[%s] exact_numerical_recovery failed: %s', context, error)
    soft_residuals = (
        _audit_soft_group_solution(
            weights=weights.to_numpy(dtype=float),
            group_bounds=soft_group_bounds,
        )
        if soft_group_bounds is not None else ()
    )
    outcome = replace(
        outcome,
        constraint_residuals=(
            outcome.constraint_residuals + exact_residuals + soft_residuals
        ),
    )
    failed_exact_residuals = [
        residual for residual in exact_residuals
        if residual.hard and not residual.passed
    ]
    if outcome.accepted and failed_exact_residuals:
        outcome = replace(
            outcome,
            accepted=False,
            reason=(
                'exact audit failed: '
                + _format_exact_audit_failure(
                    weights.to_numpy(dtype=float), problem_data, failed_exact_residuals
                )
            ),
        )
    if outcome.accepted and outcome.compliant:
        message = (
            'exact_accepted'
            if audit.termination == 'gap' else 'accepted_with_gap'
        )
        logger.log(
            logging.INFO if audit.termination == 'gap' else logging.WARNING,
            '[%s] %s termination=%s relative_gap=%.8g branches=%d '
            'integer_solutions=%d seconds=%.4f',
            context or 'execution',
            message,
            audit.termination,
            audit.relative_gap,
            audit.branches,
            audit.integer_solutions,
            audit.solve_seconds,
        )
    enriched = _enrich_exact_trade_table(
        trade_table=trade_table,
        weights=weights,
        problem_data=problem_data,
        config=config,
        audit=audit,
        outcome=outcome,
    )
    enriched['exact_numerical_recovery'] = numerical_recovery
    return ExecutionOptimizationResult(
        weights=weights,
        trade_table=enriched,
        outcome=outcome,
    )
