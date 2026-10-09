"""Cash-funded legacy ranked execution using resolved caller instructions.

This extraction preserves corridor sizing, constraint-directed rescue, numerical
rounding and final funding audits. The caller owns target construction, cadence
and instruction interpretation. Risk-budget target-band projection is unsupported.
Weights are fractions of NAV; covariance stays in the supplied variance units.
"""

import logging
from dataclasses import dataclass
from dataclasses import replace
from typing import Optional
import numpy as np
import pandas as pd
from optimalportfolios.optimization.config import OptimiserConfig
from optimalportfolios.optimization.constraints import ConstraintEnforcementType, Constraints
from optimalportfolios.optimization.constraints.analytics import evaluate_constraint_residuals
from optimalportfolios.optimization.general.minimum_tracking_error import (
    wrapper_minimise_tracking_error,
)
from optimalportfolios.optimization.solver_diagnostics import (
    OptimizationOutcome,
    diagnose_infeasibility,
)
from optimalportfolios.execution._budget import _consume_projection
from optimalportfolios.execution._projection_cache import _project
from optimalportfolios.execution.feasibility import compute_group_bound_bridges
from optimalportfolios.execution.schema import (
    BASE_WEIGHT,
    CUTOFF_SELL_DOWN,
    CUTOFF_SELL_DOWN_DUST_ROUNDED,
    CUTOFF_SELL_DOWN_RETAINED_FRACTION,
    CUTOFF_SELL_DOWN_RETENTION,
    CURRENT_WEIGHT,
    DESIRED_REWEIGHT,
    EFFECTIVE_MODEL_WEIGHT,
    FEASIBILITY_RESCUE_TRADE,
    MANDATORY_CATEGORY,
    MANDATORY_FUNDING,
    MANDATORY_TRADE,
    POLICY_MAX_WEIGHT,
    POLICY_MIN_WEIGHT,
    PRODUCT_BOUND_OVERRIDE,
    RAW_MODEL_WEIGHT,
    REBALANCE_CADENCE_ELIGIBLE,
    REQUESTED_TRADE,
    RULE4_TRADE,
    SCORE_PER_TURNOVER,
    SELL_DOWN_DUST_THRESHOLD,
    SELECTED_TRADE,
    SETTLEMENT_CASH,
    TRADE_SCORE,
    WEIGHT_ZERO_TOLERANCE,
)

logger = logging.getLogger(__name__)
GROUP_BOUND_ROUNDING_TOLERANCE = 0.5 / 10000.0
CASH_FUNDING_IDENTITY_TOLERANCE = 1e-07
CASH_ROUNDING_RECONCILIATION_LIMIT = 1e-06
PROPOSED_WEIGHT = "proposed_weight"
EXECUTED_TRADE = "executed_trade"
MANDATORY_EXECUTED_TRADE = "mandatory_executed_trade"
DISCRETIONARY_EXECUTED_TRADE = "discretionary_executed_trade"
MANDATORY_CORRIDOR = "mandatory_corridor"
MANDATORY_CORRIDOR_MIN = "mandatory_corridor_min"
MANDATORY_CORRIDOR_MAX = "mandatory_corridor_max"
MANDATORY_CORRIDOR_EXTRA_TRADE = "mandatory_corridor_extra_trade"
TARGET_GAP = "target_gap"
REWEIGHT_FUNDING = "reweight_funding"
EXPECTED_CASH_WEIGHT = "expected_cash_weight"
DIRECTED_RESCUE_CANDIDATE = "directed_rescue_candidate"
RESCUE_REASON = "rescue_reason"
STRICT_CORRIDOR_MIN = "strict_corridor_min"
STRICT_CORRIDOR_MAX = "strict_corridor_max"
CORRIDOR_RELAXATION = "corridor_relaxation"
RELAXED_CORRIDOR_SOLVE = "relaxed_corridor_solve"
MANDATORY_BUY_CORRIDOR_CATEGORIES = frozenset({"product_min_buy", "instruction_min_buy"})
MANDATORY_SELL_CORRIDOR_CATEGORIES = frozenset({"product_max_sell", "instruction_max_sell"})


def _get_mandatory_corridor_masks(trade_table: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Return one-sided mandatory-buy and mandatory-sell corridor masks."""
    index = trade_table.index
    mandatory = trade_table.get(MANDATORY_TRADE, pd.Series(False, index=index)).astype(bool)
    category = (
        trade_table.get(MANDATORY_CATEGORY, pd.Series("", index=index, dtype=object))
        .fillna("")
        .astype(str)
    )
    return (
        mandatory & category.isin(MANDATORY_BUY_CORRIDOR_CATEGORIES),
        mandatory & category.isin(MANDATORY_SELL_CORRIDOR_CATEGORIES),
    )


class ExecutionSolverInfeasibility(ValueError):
    """Strict/relaxed execution bounds are mathematically infeasible.

    Attributes:
        bridges: Unrounded interval diagnostics for each constrained row.
        trade_table: Last audited trade set attempted by feasibility rescue.
    """

    def __init__(
        self,
        message: str,
        bridges: Optional[pd.DataFrame] = None,
        trade_table: Optional[pd.DataFrame] = None,
    ) -> None:
        """Retain structural interval evidence and the last attempted trade set."""
        super().__init__(message)
        self.bridges = pd.DataFrame() if bridges is None else bridges.copy()
        self.trade_table = trade_table


@dataclass(frozen=True)
class ExecutionOptimizationResult:
    """Selected-trade solution and its complete solver audit.

    Attributes:
        weights: Proposed execution weights in internal model-name order.
        trade_table: Policy table plus proposed weights and funding audit.
        outcome: Structured solver status and hard-constraint residuals.
    """

    weights: pd.Series
    trade_table: pd.DataFrame
    outcome: OptimizationOutcome

    @property
    def accepted(self) -> bool:
        """Whether CVXPY returned an accepted portfolio."""
        return self.outcome.accepted

    @property
    def compliant(self) -> bool:
        """Whether every audited hard constraint passed."""
        return self.outcome.compliant


def _build_selected_execution_bound_series(
    base_constraints: Constraints, trade_table: pd.DataFrame, relax_selected_corridors: bool
) -> tuple[pd.Series, pd.Series]:
    """Return executable instrument bounds for one selected trade set."""
    if {"target_band_lower", "target_band_upper"}.issubset(trade_table.columns):
        raise NotImplementedError(
            "target-band risk-budget execution is not part of the legacy ranked solver"
        )
    required = {
        BASE_WEIGHT,
        CURRENT_WEIGHT,
        EFFECTIVE_MODEL_WEIGHT,
        POLICY_MIN_WEIGHT,
        POLICY_MAX_WEIGHT,
        RULE4_TRADE,
        SELECTED_TRADE,
        SETTLEMENT_CASH,
    }
    missing = required.difference(trade_table.columns)
    if missing:
        raise ValueError(f"execution trade table is missing columns {sorted(missing)}")
    index = trade_table.index
    if not index.is_unique:
        raise ValueError("execution trade table index must be unique")
    base = trade_table[BASE_WEIGHT].astype(float)
    target = trade_table[EFFECTIVE_MODEL_WEIGHT].astype(float)
    policy_min = trade_table[POLICY_MIN_WEIGHT].astype(float)
    policy_max = trade_table[POLICY_MAX_WEIGHT].astype(float)
    selected = trade_table[SELECTED_TRADE].astype(bool)
    sell_down = trade_table.get(CUTOFF_SELL_DOWN, pd.Series(False, index=index)).astype(bool)
    rule4 = trade_table[RULE4_TRADE].astype(bool)
    cash = trade_table[SETTLEMENT_CASH].astype(bool)
    if int(cash.sum()) != 1:
        raise ValueError("execution optimisation requires exactly one settlement-cash asset")
    invalid_selection = selected & (~rule4 | cash)
    if invalid_selection.any():
        raise ValueError(
            f"only non-cash Rule 4 trades may be selected: {index[invalid_selection].tolist()}"
        )
    values = pd.concat([base, target, policy_min, policy_max], axis=1, sort=False)
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("execution weights and policy bounds must be finite")
    default_min = 0.0 if base_constraints.is_long_only else -base_constraints.max_exposure
    product_min = (
        base_constraints.min_weights.reindex(index).fillna(default_min).astype(float)
        if base_constraints.min_weights is not None
        else pd.Series(default_min, index=index)
    )
    product_max = (
        base_constraints.max_weights.reindex(index)
        .fillna(base_constraints.max_exposure)
        .astype(float)
        if base_constraints.max_weights is not None
        else pd.Series(base_constraints.max_exposure, index=index)
    )
    product_override = trade_table.get(
        PRODUCT_BOUND_OVERRIDE, pd.Series(False, index=index)
    ).astype(bool)
    product_min.loc[product_override] = trade_table.loc[product_override, CURRENT_WEIGHT]
    product_max.loc[product_override] = trade_table.loc[product_override, CURRENT_WEIGHT]
    hard_min = pd.concat([product_min, policy_min], axis=1, sort=False).max(axis=1)
    hard_max = pd.concat([product_max, policy_max], axis=1, sort=False).min(axis=1)
    invalid_hard = hard_min > hard_max + WEIGHT_ZERO_TOLERANCE
    if invalid_hard.any():
        raise ValueError(
            "execution product and policy bounds do not intersect:\n"
            f"{pd.DataFrame({'min': hard_min, 'max': hard_max}).loc[invalid_hard]}"
        )
    corridor_min = pd.concat([base, target], axis=1, sort=False).min(axis=1)
    corridor_max = pd.concat([base, target], axis=1, sort=False).max(axis=1)
    mandatory_buy, mandatory_sell = _get_mandatory_corridor_masks(trade_table)
    mandatory_corridor = mandatory_buy | mandatory_sell
    movable = selected | sell_down | mandatory_corridor
    if relax_selected_corridors:
        selected_min = hard_min.copy()
        selected_max = hard_max.copy()
    else:
        selected_min = pd.concat([corridor_min, hard_min], axis=1, sort=False).max(axis=1)
        selected_max = pd.concat([corridor_max, hard_max], axis=1, sort=False).min(axis=1)
    selected_min.loc[sell_down] = hard_min.loc[sell_down]
    selected_max.loc[sell_down] = pd.concat(
        [trade_table.loc[sell_down, CURRENT_WEIGHT], hard_max.loc[sell_down]], axis=1, sort=False
    ).min(axis=1)
    selected_min.loc[mandatory_buy] = base.loc[mandatory_buy]
    selected_max.loc[mandatory_buy] = hard_max.loc[mandatory_buy]
    selected_min.loc[mandatory_sell] = hard_min.loc[mandatory_sell]
    selected_max.loc[mandatory_sell] = base.loc[mandatory_sell]
    invalid_selected = movable & (selected_min > selected_max + WEIGHT_ZERO_TOLERANCE)
    if invalid_selected.any():
        details = pd.DataFrame(
            {
                "base": base,
                "target": target,
                "hard_min": hard_min,
                "hard_max": hard_max,
                "effective_min": selected_min,
                "effective_max": selected_max,
            }
        ).loc[invalid_selected]
        raise ValueError(
            f"selected execution corridors do not intersect hard bounds:\n{details.to_string()}"
        )
    if not relax_selected_corridors:
        missing_base = movable & (
            (base < selected_min - WEIGHT_ZERO_TOLERANCE)
            | (base > selected_max + WEIGHT_ZERO_TOLERANCE)
        )
        if missing_base.any():
            details = pd.DataFrame(
                {"base": base, "corridor_min": selected_min, "corridor_max": selected_max}
            ).loc[missing_base]
            raise ValueError(
                f"selected corridors must contain their post-mandatory base:\n{details.to_string()}"
            )
    frozen = ~movable & ~cash
    frozen_violation = frozen & (
        (base < hard_min - WEIGHT_ZERO_TOLERANCE) | (base > hard_max + WEIGHT_ZERO_TOLERANCE)
    )
    if frozen_violation.any():
        details = pd.DataFrame({"base": base, "hard_min": hard_min, "hard_max": hard_max}).loc[
            frozen_violation
        ]
        raise ValueError(f"fixed execution base violates hard bounds:\n{details.to_string()}")
    min_weights = base.copy()
    max_weights = base.copy()
    min_weights.loc[movable] = selected_min.loc[movable]
    max_weights.loc[movable] = selected_max.loc[movable]
    min_weights.loc[cash] = hard_min.loc[cash]
    max_weights.loc[cash] = hard_max.loc[cash]
    return (min_weights, max_weights)


def _compute_selected_execution_bridges(
    base_constraints: Constraints,
    trade_table: pd.DataFrame,
    relax_selected_corridors: bool = False,
    partition_groups: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Diagnose group and funding intervals for one selected trade set.

    Args:
        partition_groups: Optional names of a verified disjoint membership partition.
            The caller supplies these names; no asset-class taxonomy is assumed.
    """
    lower, upper = _build_selected_execution_bound_series(
        base_constraints=base_constraints,
        trade_table=trade_table,
        relax_selected_corridors=relax_selected_corridors,
    )
    group_bounds = base_constraints.group_lower_upper_constraints
    if group_bounds is None:
        bridges = pd.DataFrame(
            columns=("e_min", "e_max", "group_min", "group_max", "bridge_min", "bridge_max")
        )
    else:
        loadings = group_bounds.group_loadings.reindex(index=trade_table.index).fillna(0.0)
        minimum = (
            group_bounds.group_min_allocation
            if group_bounds.group_min_allocation is not None
            else pd.Series(np.nan, index=loadings.columns)
        )
        maximum = (
            group_bounds.group_max_allocation
            if group_bounds.group_max_allocation is not None
            else pd.Series(np.nan, index=loadings.columns)
        )
        bridges = compute_group_bound_bridges(
            lower_bounds=lower,
            upper_bounds=upper,
            group_loadings=loadings,
            group_min=minimum,
            group_max=maximum,
        )
        class_loadings = loadings.reindex(columns=list(partition_groups))
        # Additional complement tightening is valid only for an exhaustive,
        # disjoint partition explicitly supplied by the calling application.
        class_partition = (
            bool(partition_groups)
            and class_loadings.isin([0.0, 1.0]).all().all()
            and class_loadings.sum(axis=1).eq(1.0).all()
        )
        for group in loadings.columns:
            membership = loadings[group]
            if not membership.isin([0.0, 1.0]).all():
                continue
            outside = membership.eq(0.0)
            outside_min = float(lower.loc[outside].sum())
            outside_max = float(upper.loc[outside].sum())
            if class_partition:
                for asset_class in partition_groups:
                    members = class_loadings[asset_class].eq(1.0)
                    if (members & ~outside).any():
                        continue
                    raw_min = float(lower.loc[members].sum())
                    raw_max = float(upper.loc[members].sum())
                    class_min = minimum.get(asset_class, np.nan)
                    class_max = maximum.get(asset_class, np.nan)
                    if np.isfinite(class_min):
                        outside_min += max(raw_min, class_min) - raw_min
                    if np.isfinite(class_max):
                        outside_max += min(raw_max, class_max) - raw_max
            # Full investment couples each binary group to its complement;
            # independent box ranges alone can miss a required funding trade.
            bridges.at[group, "e_min"] = max(
                bridges.at[group, "e_min"], base_constraints.min_exposure - outside_max
            )
            bridges.at[group, "e_max"] = min(
                bridges.at[group, "e_max"], base_constraints.max_exposure - outside_min
            )
        bridges["bridge_min"] = (
            (bridges["group_min"] - bridges["e_max"]).clip(lower=0.0).fillna(0.0)
        )
        bridges["bridge_max"] = (
            (bridges["e_min"] - bridges["group_max"]).clip(lower=0.0).fillna(0.0)
        )
    cash = trade_table[SETTLEMENT_CASH].astype(bool)
    cash_asset = trade_table.index[cash][0]
    noncash = ~cash
    cash_e_min = float(base_constraints.min_exposure - upper.loc[noncash].sum())
    cash_e_max = float(base_constraints.max_exposure - lower.loc[noncash].sum())
    cash_min = float(lower.loc[cash_asset])
    cash_max = float(upper.loc[cash_asset])
    bridges.loc["cash", :] = {
        "e_min": cash_e_min,
        "e_max": cash_e_max,
        "group_min": cash_min,
        "group_max": cash_max,
        "bridge_min": max(cash_min - cash_e_max, 0.0),
        "bridge_max": max(cash_e_min - cash_max, 0.0),
    }
    bridges.index.name = "group"
    return bridges


def _positive_bridges(bridges: pd.DataFrame) -> pd.DataFrame:
    """Return constrained rows with a positive executable-range breach."""
    if bridges.empty:
        return bridges
    return bridges.loc[bridges[["bridge_min", "bridge_max"]].max(axis=1).gt(WEIGHT_ZERO_TOLERANCE)]


def _snap_group_bounds_within_rounding_tolerance(
    base_constraints: Constraints, bridges: pd.DataFrame
) -> Constraints:
    """Snap numerical group-bound bridges to their executable endpoints.

    A strictly positive group bridge no larger than 0.5 bp is treated as a
    representation mismatch between an effective workbook bound and the
    executable interval assembled from instrument bounds. The effective lower
    bound is moved down to ``e_max`` or the effective upper bound up to
    ``e_min`` on a copied constraint object. The strict-positive test is
    deliberately independent of ``WEIGHT_ZERO_TOLERANCE`` so a bridge at that
    numerical boundary is normalized rather than rejected. Cash bounds and
    bridges above 0.5 bp are never changed.

    Args:
        base_constraints: Original product constraints for the decision.
        bridges: Executable group and cash interval diagnostics.

    Returns:
        The original constraints when no group row qualifies, otherwise a
        copy containing only the endpoint-normalized group bounds.
    """
    group_bounds = base_constraints.group_lower_upper_constraints
    if group_bounds is None or bridges.empty:
        return base_constraints
    minimum = (
        None
        if group_bounds.group_min_allocation is None
        else group_bounds.group_min_allocation.copy()
    )
    maximum = (
        None
        if group_bounds.group_max_allocation is None
        else group_bounds.group_max_allocation.copy()
    )
    adjusted = False
    for group, row in bridges.drop(index="cash", errors="ignore").iterrows():
        bridge_min = float(row["bridge_min"])
        if (
            minimum is not None
            and group in minimum.index
            and (0.0 < bridge_min <= GROUP_BOUND_ROUNDING_TOLERANCE)
        ):
            original = float(minimum.loc[group])
            executable = float(row["e_max"])
            minimum.loc[group] = executable
            adjusted = True
            logger.info(
                "execution group-bound rounding: group '%s' minimum %.10f -> %.10f; "
                "bridge=%.4f bp within %.4f bp tolerance",
                group,
                original,
                executable,
                10000.0 * bridge_min,
                10000.0 * GROUP_BOUND_ROUNDING_TOLERANCE,
            )
        bridge_max = float(row["bridge_max"])
        if (
            maximum is not None
            and group in maximum.index
            and (0.0 < bridge_max <= GROUP_BOUND_ROUNDING_TOLERANCE)
        ):
            original = float(maximum.loc[group])
            executable = float(row["e_min"])
            maximum.loc[group] = executable
            adjusted = True
            logger.info(
                "execution group-bound rounding: group '%s' maximum %.10f -> %.10f; "
                "bridge=%.4f bp within %.4f bp tolerance",
                group,
                original,
                executable,
                10000.0 * bridge_max,
                10000.0 * GROUP_BOUND_ROUNDING_TOLERANCE,
            )
    if not adjusted:
        return base_constraints
    rounded_group_bounds = replace(
        group_bounds, group_min_allocation=minimum, group_max_allocation=maximum
    )
    return base_constraints.copy(group_lower_upper_constraints=rounded_group_bounds)


def _format_bridge_failure(bridges: pd.DataFrame) -> str:
    """Format a stable diagnostic while retaining numeric data separately."""
    rows = []
    for group, row in _positive_bridges(bridges).iterrows():
        if row["bridge_min"] > WEIGHT_ZERO_TOLERANCE:
            rows.append(
                f"Group '{group}': executable e_max ({row['e_max']:.4f}) "
                f"< group_min ({row['group_min']:.4f}); bridge={row['bridge_min']:.10f}"
            )
        if row["bridge_max"] > WEIGHT_ZERO_TOLERANCE:
            rows.append(
                f"Group '{group}': executable e_min ({row['e_min']:.4f}) "
                f"> group_max ({row['group_max']:.4f}); bridge={row['bridge_max']:.10f}"
            )
    return "Infeasible execution intervals:\n" + "\n".join(rows)


def build_selected_execution_constraints(
    base_constraints: Constraints,
    trade_table: pd.DataFrame,
    relax_selected_corridors: bool = False,
    partition_groups: tuple[str, ...] = (),
) -> Constraints:
    """Build hard bounds for the selected execution problem.

    Exact mandatory targets and unselected Rule 4 instruments are fixed at the
    post-mandatory base weight. One-sided product and instruction repairs use
    direction-preserving corridors: a minimum buy may increase from its
    compulsory boundary to the combined hard maximum, while a maximum sell may
    decrease from its compulsory boundary to the combined hard minimum. By
    default, selected Rule 4 instruments can move only inside the
    base-to-effective-model corridor, which must retain the base point. When
    ``relax_selected_corridors`` is true, selected instruments can instead use
    their full intersected hard product and policy bounds. Existing model-entry
    instruments remain pinned. The settlement-cash instrument remains free
    inside its hard bounds and funds the movable non-cash positions. Before rejecting a
    structural group infeasibility, a strictly positive bridge no larger than
    0.5 bp is normalized to its executable endpoint on an execution-local
    constraint copy; the bridges are then recomputed. Larger group bridges,
    cash bounds and the original product constraints remain unchanged.

    Args:
        base_constraints: Constraints reconstructed from pipeline input.
        trade_table: Internally indexed scored policy cross-section.
        relax_selected_corridors: Use full hard bounds for selected Rule 4
            instruments. Mandatory corridors, frozen instruments and
            settlement cash are unchanged.
        partition_groups: Optional names of a verified disjoint membership partition.
            The caller supplies these names; no asset-class taxonomy is assumed.

    Returns:
        Forced hard constraints for minimum-tracking-error optimisation.
    """
    min_weights, max_weights = _build_selected_execution_bound_series(
        base_constraints=base_constraints,
        trade_table=trade_table,
        relax_selected_corridors=relax_selected_corridors,
    )
    bridges = _compute_selected_execution_bridges(
        base_constraints=base_constraints,
        trade_table=trade_table,
        relax_selected_corridors=relax_selected_corridors,
        partition_groups=partition_groups,
    )
    effective_constraints = _snap_group_bounds_within_rounding_tolerance(
        base_constraints=base_constraints, bridges=bridges
    )
    if effective_constraints is not base_constraints:
        bridges = _compute_selected_execution_bridges(
            base_constraints=effective_constraints,
            trade_table=trade_table,
            relax_selected_corridors=relax_selected_corridors,
            partition_groups=partition_groups,
        )
    if not _positive_bridges(bridges).empty:
        raise ExecutionSolverInfeasibility(
            _format_bridge_failure(bridges), bridges=bridges, trade_table=trade_table
        )
    beta_constraint = effective_constraints.benchmark_beta_constraint
    if beta_constraint is not None and beta_constraint.beta_loadings is None:
        raise ValueError("execution solve requires resolved benchmark beta loadings")
    try:
        common_updates = {
            "min_weights": min_weights,
            "max_weights": max_weights,
            "weights_0": trade_table[CURRENT_WEIGHT].astype(float),
            "tre_utility_weight": None,
            "turnover_utility_weight": None,
            "constraint_enforcement_type": ConstraintEnforcementType.FORCED_CONSTRAINTS,
        }
        common_updates.update(
            {
                "tracking_err_vol_constraint": None,
                "turnover_constraint": None,
                "turnover_costs": None,
                "group_tracking_error_constraint": None,
                "group_turnover_constraint": None,
            }
        )
        return effective_constraints.copy(**common_updates)
    except ValueError as error:
        if str(error).startswith("Infeasible constraints detected"):
            raise ExecutionSolverInfeasibility(
                str(error), bridges=bridges, trade_table=trade_table
            ) from error
        raise


def _solve_selected_execution_portfolio_once(
    trade_table: pd.DataFrame,
    base_constraints: Constraints,
    covariance: pd.DataFrame,
    optimiser_config: OptimiserConfig = OptimiserConfig(solver="MOSEK"),
    context: str = "",
    relax_selected_corridors: bool = False,
    group_split_asset_classes: Optional[pd.Series] = None,
    partition_groups: tuple[str, ...] = (),
) -> ExecutionOptimizationResult:
    """Run one strict- or relaxed-corridor minimum-TRE solve.

    Args:
        partition_groups: Optional names of a verified disjoint membership partition.
            The caller supplies these names; no asset-class taxonomy is assumed.
    """
    constraints = build_selected_execution_constraints(
        base_constraints=base_constraints,
        trade_table=trade_table,
        relax_selected_corridors=relax_selected_corridors,
        partition_groups=partition_groups,
    )
    weights, outcome = _project(wrapper_minimise_tracking_error,
        pd_covar=covariance,
        benchmark_weights=trade_table[RAW_MODEL_WEIGHT],
        constraints=constraints,
        weights_0=trade_table[CURRENT_WEIGHT],
        optimiser_config=optimiser_config,
        context=context,
    )
    dust_rounded = pd.Series(False, index=trade_table.index)
    if outcome.accepted and outcome.compliant:
        sell_down = trade_table.get(
            CUTOFF_SELL_DOWN, pd.Series(False, index=trade_table.index)
        ).astype(bool)
        threshold = float(
            trade_table[SELL_DOWN_DUST_THRESHOLD].iloc[0]
            if SELL_DOWN_DUST_THRESHOLD in trade_table
            else 0.0
        )
        dust_candidates = trade_table.index[
            sell_down & weights.gt(WEIGHT_ZERO_TOLERANCE) & weights.lt(threshold)
        ]
        cash = trade_table[SETTLEMENT_CASH].astype(bool)
        cash_asset = trade_table.index[cash][0]
        group_bounds = constraints.group_lower_upper_constraints
        cleaned = weights.copy()
        for instrument in dust_candidates:
            weight = float(cleaned.loc[instrument])
            has_group_slack = True
            if group_bounds is not None:
                loadings = group_bounds.group_loadings.reindex(index=trade_table.index).fillna(0.0)
                minimum = group_bounds.group_min_allocation
                exposure = loadings.mul(cleaned, axis=0).sum(axis=0)
                loaded_groups = loadings.columns[loadings.loc[instrument].gt(WEIGHT_ZERO_TOLERANCE)]
                if minimum is not None:
                    for group in loaded_groups:
                        if (
                            group in minimum.index
                            and np.isfinite(minimum.loc[group])
                            and (
                                exposure.loc[group] - minimum.loc[group]
                                < weight - WEIGHT_ZERO_TOLERANCE
                            )
                        ):
                            has_group_slack = False
                            break
            if not has_group_slack:
                continue
            candidate = cleaned.copy()
            candidate.loc[instrument] = 0.0
            candidate.loc[cash_asset] += weight
            solver_constraints = outcome.constraints
            solver_index = solver_constraints.min_weights.index
            excluded = candidate.index.difference(solver_index)
            # The numerical wrapper may have filtered zero-risk instruments.
            # Dust cleanup must pass both original and solver-universe audits.
            if not candidate.loc[excluded].abs().le(WEIGHT_ZERO_TOLERANCE).all():
                continue
            full_residuals = evaluate_constraint_residuals(
                weights=candidate.reindex(trade_table.index).to_numpy(dtype=float),
                constraints=constraints,
            )
            residuals = full_residuals + evaluate_constraint_residuals(
                weights=candidate.reindex(solver_index).to_numpy(dtype=float),
                constraints=solver_constraints,
                covar=covariance.loc[solver_index, solver_index].to_numpy(dtype=float),
                covar_factorization=outcome.covar_factorization,
            )
            if all((residual.passed for residual in residuals if residual.hard)):
                cleaned = candidate
                dust_rounded.loc[instrument] = True
                outcome = replace(
                    outcome,
                    weights=cleaned.reindex(solver_index).to_numpy(dtype=float),
                    constraint_residuals=residuals,
                )
        weights = cleaned
    cash = trade_table[SETTLEMENT_CASH].astype(bool)
    cash_asset = trade_table.index[cash][0]
    if outcome.accepted and outcome.compliant:
        funded_cash = float(trade_table[CURRENT_WEIGHT].sum() - weights.loc[~cash].sum())
        cash_adjustment = funded_cash - float(weights.loc[cash_asset])
        # Reconcile only solver-scale settlement rounding; a material funding
        # residual is an error rather than permission to alter the portfolio.
        if abs(cash_adjustment) > CASH_FUNDING_IDENTITY_TOLERANCE:
            if abs(cash_adjustment) > CASH_ROUNDING_RECONCILIATION_LIMIT:
                raise RuntimeError(
                    "execution cash rounding adjustment exceeds limit: "
                    f"adjustment={cash_adjustment:.10f}"
                )
            corrected = weights.copy()
            corrected.loc[cash_asset] = funded_cash
            if (
                funded_cash < constraints.min_weights.loc[cash_asset] - WEIGHT_ZERO_TOLERANCE
                or funded_cash > constraints.max_weights.loc[cash_asset] + WEIGHT_ZERO_TOLERANCE
            ):
                raise RuntimeError(
                    "execution cash rounding adjustment breaches cash bounds: "
                    f"cash={funded_cash:.10f}"
                )
            solver_constraints = outcome.constraints
            solver_index = solver_constraints.min_weights.index
            if cash_asset not in solver_index:
                raise RuntimeError(
                    "execution cash rounding adjustment requires cash in the solver universe"
                )
            excluded = corrected.index.difference(solver_index)
            if not corrected.loc[excluded].abs().le(WEIGHT_ZERO_TOLERANCE).all():
                raise RuntimeError(
                    "execution cash rounding adjustment leaves excluded assets invested"
                )
            full_residuals = evaluate_constraint_residuals(
                weights=corrected.reindex(trade_table.index).to_numpy(dtype=float),
                constraints=constraints,
            )
            solver_residuals = evaluate_constraint_residuals(
                weights=corrected.reindex(solver_index).to_numpy(dtype=float),
                constraints=solver_constraints,
                covar=covariance.loc[solver_index, solver_index].to_numpy(dtype=float),
                covar_factorization=outcome.covar_factorization,
            )
            residuals = full_residuals + solver_residuals
            if not all((residual.passed for residual in residuals if residual.hard)):
                raise RuntimeError("execution cash rounding adjustment breaches hard constraints")
            weights = corrected
            outcome = replace(
                outcome,
                weights=corrected.reindex(solver_index).to_numpy(dtype=float),
                constraint_residuals=residuals,
            )
            logger.info("%s execution settlement cash rounded by %.10f", context, cash_adjustment)
    enriched = trade_table.copy()
    enriched[PROPOSED_WEIGHT] = weights
    enriched[EXECUTED_TRADE] = weights - enriched[CURRENT_WEIGHT]
    mandatory = enriched[MANDATORY_TRADE].astype(bool)
    enriched[MANDATORY_EXECUTED_TRADE] = enriched[BASE_WEIGHT] - enriched[CURRENT_WEIGHT]
    enriched[DISCRETIONARY_EXECUTED_TRADE] = weights - enriched[BASE_WEIGHT]
    enriched.loc[mandatory, MANDATORY_EXECUTED_TRADE] = enriched.loc[mandatory, EXECUTED_TRADE]
    enriched.loc[mandatory, DISCRETIONARY_EXECUTED_TRADE] = 0.0
    enriched[TARGET_GAP] = weights - enriched[EFFECTIVE_MODEL_WEIGHT]
    strict_min = pd.concat(
        [enriched[BASE_WEIGHT], enriched[EFFECTIVE_MODEL_WEIGHT]], axis=1, sort=False
    ).min(axis=1)
    strict_max = pd.concat(
        [enriched[BASE_WEIGHT], enriched[EFFECTIVE_MODEL_WEIGHT]], axis=1, sort=False
    ).max(axis=1)
    selected = enriched[SELECTED_TRADE].astype(bool)
    sell_down = enriched.get(CUTOFF_SELL_DOWN, pd.Series(False, index=enriched.index)).astype(bool)
    mandatory_buy, mandatory_sell = _get_mandatory_corridor_masks(enriched)
    mandatory_corridor = mandatory_buy | mandatory_sell
    interval_rows = sell_down | mandatory_corridor
    if constraints.min_weights is not None:
        strict_min.loc[interval_rows] = constraints.min_weights.reindex(enriched.index).loc[
            interval_rows
        ]
    if constraints.max_weights is not None:
        strict_max.loc[interval_rows] = constraints.max_weights.reindex(enriched.index).loc[
            interval_rows
        ]
    relaxation_eligible = selected
    enriched[STRICT_CORRIDOR_MIN] = strict_min
    enriched[STRICT_CORRIDOR_MAX] = strict_max
    enriched[MANDATORY_CORRIDOR] = mandatory_corridor
    enriched[MANDATORY_CORRIDOR_MIN] = strict_min.where(mandatory_corridor)
    enriched[MANDATORY_CORRIDOR_MAX] = strict_max.where(mandatory_corridor)
    enriched[MANDATORY_CORRIDOR_EXTRA_TRADE] = (weights - enriched[BASE_WEIGHT]).where(
        mandatory_corridor, 0.0
    )
    enriched[CORRIDOR_RELAXATION] = (
        (strict_min - weights).clip(lower=0.0) + (weights - strict_max).clip(lower=0.0)
    ).where(relaxation_eligible, 0.0)
    enriched[RELAXED_CORRIDOR_SOLVE] = relax_selected_corridors
    enriched[CUTOFF_SELL_DOWN_RETENTION] = weights.where(sell_down)
    retained_fraction = weights.div(enriched[CURRENT_WEIGHT].replace(0.0, np.nan))
    enriched[CUTOFF_SELL_DOWN_RETAINED_FRACTION] = retained_fraction.where(sell_down)
    enriched[CUTOFF_SELL_DOWN_DUST_ROUNDED] = dust_rounded
    cash = enriched[SETTLEMENT_CASH].astype(bool)
    cash_asset = enriched.index[cash][0]
    noncash = ~cash
    mandatory_funding = float(enriched.loc[noncash, MANDATORY_FUNDING].sum())
    reweight_funding = float((weights.loc[noncash] - enriched.loc[noncash, BASE_WEIGHT]).sum())
    expected_cash = float(
        enriched.loc[cash_asset, CURRENT_WEIGHT] - mandatory_funding - reweight_funding
    )
    enriched[REWEIGHT_FUNDING] = 0.0
    enriched.loc[cash_asset, REWEIGHT_FUNDING] = reweight_funding
    enriched[EXPECTED_CASH_WEIGHT] = np.nan
    enriched.loc[cash_asset, EXPECTED_CASH_WEIGHT] = expected_cash
    if outcome.accepted and (
        not np.isclose(
            weights.loc[cash_asset], expected_cash, atol=CASH_FUNDING_IDENTITY_TOLERANCE, rtol=0.0
        )
    ):
        raise RuntimeError(
            f"execution cash funding identity failed: actual={weights.loc[cash_asset]:.10f}, "
            f"expected={expected_cash:.10f}"
        )
    if outcome.accepted and outcome.compliant:
        for instrument in enriched.index[mandatory_corridor]:
            logger.info(
                "%s mandatory corridor: instrument=%s category=%s current=%.8f boundary=%.8f "
                "interval=[%.8f, %.8f] proposed=%.8f extra_from_boundary=%.8f",
                context or "execution",
                instrument,
                enriched.at[instrument, MANDATORY_CATEGORY],
                enriched.at[instrument, CURRENT_WEIGHT],
                enriched.at[instrument, BASE_WEIGHT],
                enriched.at[instrument, MANDATORY_CORRIDOR_MIN],
                enriched.at[instrument, MANDATORY_CORRIDOR_MAX],
                enriched.at[instrument, PROPOSED_WEIGHT],
                enriched.at[instrument, MANDATORY_CORRIDOR_EXTRA_TRADE],
            )
    return ExecutionOptimizationResult(weights=weights, trade_table=enriched, outcome=outcome)


def solve_selected_execution_portfolio(
    trade_table: pd.DataFrame,
    base_constraints: Constraints,
    covariance: pd.DataFrame,
    optimiser_config: OptimiserConfig = OptimiserConfig(solver="MOSEK"),
    context: str = "",
    relax_selected_corridors: bool = False,
    group_split_asset_classes: Optional[pd.Series] = None,
    partition_groups: tuple[str, ...] = (),
) -> ExecutionOptimizationResult:
    """Minimize tracking error to the raw model for a selected trade set.

    By default the selected instruments remain in their strict
    base-to-effective-model corridors. A caller may explicitly request the
    separately reported relaxed-corridor solve.

    Args:
        trade_table: Internal scored table carrying ``selected_trade``.
        base_constraints: Product constraints reconstructed from pipeline data.
        covariance: Full annualised covariance in trade-table order.
        optimiser_config: CVXPY solver and covariance configuration.
        context: Solve label included in diagnostics.
        relax_selected_corridors: Allow selected Rule 4 rows to use their full
            product/instruction bounds instead of their strict corridors.
        group_split_asset_classes: Unsupported compatibility argument. A non-None
            value raises ``NotImplementedError`` in this legacy ranked extraction.
        partition_groups: Optional names of a verified disjoint membership partition.
            The caller supplies these names; no asset-class taxonomy is assumed.

    Returns:
        Proposed weights, enriched trade table and structured solver audit.
    """
    if group_split_asset_classes is not None:
        raise NotImplementedError(
            "group-split risk-budget execution is not part of the legacy ranked solver"
        )
    _consume_projection()
    return _solve_selected_execution_portfolio_once(
        trade_table=trade_table,
        base_constraints=base_constraints,
        covariance=covariance,
        optimiser_config=optimiser_config,
        context=context,
        relax_selected_corridors=relax_selected_corridors,
        group_split_asset_classes=group_split_asset_classes,
        partition_groups=partition_groups,
    )


def _get_constraint_directed_rescue_candidates(
    trade_table: pd.DataFrame,
    base_constraints: Constraints,
    rescue_eligible: pd.Series,
    bridges: pd.DataFrame,
    relax_selected_corridors: bool = False,
) -> tuple[pd.Series, pd.Series]:
    """Return candidates with signed capacity against a positive bridge.

    Score never controls admissibility. It is used only by the caller to
    order the admissible set.
    """
    index = trade_table.index
    eligible = rescue_eligible.reindex(index).fillna(False).astype(bool)
    desired = trade_table[DESIRED_REWEIGHT].astype(float)
    can_buy = desired.gt(WEIGHT_ZERO_TOLERANCE)
    can_sell = desired.lt(-WEIGHT_ZERO_TOLERANCE)
    if relax_selected_corridors:
        opened = trade_table.copy(deep=True)
        opened[SELECTED_TRADE] = opened[SELECTED_TRADE].astype(bool) | eligible
        hard_lower, hard_upper = _build_selected_execution_bound_series(
            base_constraints, opened, True
        )
        base = trade_table[BASE_WEIGHT].astype(float)
        can_buy = hard_upper.gt(base + WEIGHT_ZERO_TOLERANCE)
        can_sell = hard_lower.lt(base - WEIGHT_ZERO_TOLERANCE)
    directed = pd.Series(False, index=index)
    reason_sets = {instrument: set() for instrument in index}
    active = _positive_bridges(bridges)
    group_bounds = base_constraints.group_lower_upper_constraints
    loadings = (
        group_bounds.group_loadings.reindex(index=index).fillna(0.0)
        if group_bounds is not None
        else pd.DataFrame(index=index)
    )
    lower, upper = _build_selected_execution_bound_series(
        base_constraints, trade_table, relax_selected_corridors
    )
    raw_bridges = compute_group_bound_bridges(
        lower, upper, loadings, bridges["group_min"], bridges["group_max"]
    )
    for group, row in active.iterrows():
        if group == "cash":
            if row["bridge_min"] > WEIGHT_ZERO_TOLERANCE:
                matches = can_sell
                reason = "cash:min"
            else:
                matches = can_buy
                reason = "cash:max"
        elif group in loadings.columns:
            cash_asset = trade_table.index[trade_table[SETTLEMENT_CASH].astype(bool)][0]
            # A unit instrument trade is funded by the opposite settlement-cash
            # trade, so its net group exposure uses the difference in loadings.
            loading = loadings[group] - loadings.at[cash_asset, group]
            if row["bridge_min"] > WEIGHT_ZERO_TOLERANCE:
                matches = loading.gt(0.0) & can_buy | loading.lt(0.0) & can_sell
                reason = f"{group}:min"
            else:
                matches = loading.gt(0.0) & can_sell | loading.lt(0.0) & can_buy
                reason = f"{group}:max"
            membership = loadings[group]
            if raw_bridges.at[group, "bridge_min"] > WEIGHT_ZERO_TOLERANCE:
                matches |= membership.gt(0.0) & can_buy | membership.lt(0.0) & can_sell
            if raw_bridges.at[group, "bridge_max"] > WEIGHT_ZERO_TOLERANCE:
                matches |= membership.gt(0.0) & can_sell | membership.lt(0.0) & can_buy
            if membership.isin([0.0, 1.0]).all():
                outside = membership.eq(0.0)
                if (
                    row["bridge_max"] > WEIGHT_ZERO_TOLERANCE
                    and row["e_min"] > raw_bridges.at[group, "e_min"] + WEIGHT_ZERO_TOLERANCE
                ):
                    matches |= outside & can_buy
                if (
                    row["bridge_min"] > WEIGHT_ZERO_TOLERANCE
                    and row["e_max"] < raw_bridges.at[group, "e_max"] - WEIGHT_ZERO_TOLERANCE
                ):
                    matches |= outside & can_sell
        else:
            continue
        matches &= eligible
        directed |= matches
        for instrument in index[matches]:
            reason_sets[instrument].add(reason)
    reasons = pd.Series(
        {instrument: "|".join(sorted(reason_sets[instrument])) for instrument in index},
        index=index,
        dtype=object,
    )
    return (eligible & directed, reasons)


def solve_feasible_execution_portfolio(
    trade_table: pd.DataFrame,
    base_constraints: Constraints,
    covariance: pd.DataFrame,
    optimiser_config: OptimiserConfig = OptimiserConfig(solver="MOSEK"),
    context: str = "",
    expand_for_feasibility: bool = False,
    sign_directed_rescue: bool = False,
    allow_corridor_relaxation: bool = False,
    group_split_asset_classes: Optional[pd.Series] = None,
    partition_groups: tuple[str, ...] = (),
) -> ExecutionOptimizationResult:
    """Solve a requested set and linearly admit constraint-directed rescues.

    Args:
        trade_table: Scored table carrying requested discretionary trades.
        base_constraints: Product constraints reconstructed from pipeline data.
        covariance: Full model covariance.
        optimiser_config: Solver and covariance configuration.
        context: Solve label included in diagnostics.
        expand_for_feasibility: Admit unselected Rule 4 trades until all
            executable bridges are covered and the hard solve is compliant.
        sign_directed_rescue: Retain the legacy base-residual direction filter
            as fallback when no numeric interval bridge is available. Numeric
            bridges are always constraint-directed.
        allow_corridor_relaxation: After strict requested and prefix-expanded
            solves fail, permit an explicitly flagged relaxed-corridor retry.
        group_split_asset_classes: Unsupported compatibility argument. A non-None
            value raises ``NotImplementedError`` in this legacy ranked extraction.
        partition_groups: Optional names of a verified disjoint membership partition.
            The caller supplies these names; no asset-class taxonomy is assumed.

    Returns:
        Accepted repaired solution, or the last failed solver outcome with terminal diagnostics.
    """
    if group_split_asset_classes is not None:
        raise NotImplementedError(
            "group-split risk-budget execution is not part of the legacy ranked solver"
        )
    audited = trade_table.copy()
    requested = audited[SELECTED_TRADE].astype(bool)
    audited[REQUESTED_TRADE] = requested
    audited[FEASIBILITY_RESCUE_TRADE] = False
    audited[DIRECTED_RESCUE_CANDIDATE] = False
    audited[RESCUE_REASON] = ""
    initial_error: Optional[ExecutionSolverInfeasibility] = None
    try:
        initial_result = solve_selected_execution_portfolio(
            trade_table=audited,
            base_constraints=base_constraints,
            covariance=covariance,
            optimiser_config=optimiser_config,
            context=context,
            group_split_asset_classes=group_split_asset_classes,
            partition_groups=partition_groups,
        )
    except ExecutionSolverInfeasibility as error:
        initial_error = error
        initial_result = None
    if initial_result is not None and initial_result.accepted and initial_result.compliant:
        return initial_result
    rescue_eligible = (
        audited[REBALANCE_CADENCE_ELIGIBLE].astype(bool)
        & audited[RULE4_TRADE].astype(bool)
        & audited[DESIRED_REWEIGHT].abs().gt(WEIGHT_ZERO_TOLERANCE)
        & ~requested
    )
    latest_result = initial_result

    def finish_failed(result: ExecutionOptimizationResult) -> ExecutionOptimizationResult:
        """Diagnose only the terminal solve; diagnostic slacks are never applied."""
        if (
            result.accepted
            and result.compliant
            or not optimiser_config.diagnose_infeasibility
            or result.outcome.constraints is None
        ):
            return result
        outcome = result.outcome
        if not any(
            (token in str(outcome.status).lower() for token in ("infeasible", "solver_error"))
        ):
            return result
        aligned = outcome.constraints.min_weights.index
        risk_covar = (
            outcome.covar_factorization.covar
            if outcome.covar_factorization is not None
            else covariance.loc[aligned, aligned].to_numpy(dtype=float)
        )
        slacks = diagnose_infeasibility(
            constraints=outcome.constraints,
            covar=risk_covar,
            solver=optimiser_config.solver,
            context=f"{context} terminal execution failure".strip(),
        )
        if slacks:
            ordered = sorted(slacks.items(), key=lambda item: -item[1])
            detail = "box/group diagnostic slacks (not applied): " + ", ".join(
                (f"{name}={value * 10000:.4f} bp" for name, value in ordered[:12])
            )
            if len(ordered) > 12:
                detail += f"; {len(ordered) - 12} additional bounds in solver log"
        else:
            detail = "box/group diagnostic found no reported slack; cause remains unresolved"
        return replace(
            result, outcome=replace(outcome, reason=f"{outcome.reason}; {detail}".strip("; "))
        )

    def run_linear_rescue(
        start: pd.DataFrame,
        relax_selected_corridors: bool,
        failed_result: Optional[ExecutionOptimizationResult],
    ) -> tuple[
        Optional[ExecutionOptimizationResult], Optional[ExecutionSolverInfeasibility], pd.DataFrame
    ]:
        """Admit ranked feasible directions until hard constraints pass or capacity ends."""
        nonlocal latest_result
        candidate = start.copy()
        latest_error: Optional[ExecutionSolverInfeasibility] = None
        admission_count = int(candidate[FEASIBILITY_RESCUE_TRADE].sum())
        while True:
            bridges = _compute_selected_execution_bridges(
                base_constraints=base_constraints,
                trade_table=candidate,
                relax_selected_corridors=relax_selected_corridors,
                partition_groups=partition_groups,
            )
            active_bridges = _positive_bridges(bridges)
            remaining = rescue_eligible & ~candidate[SELECTED_TRADE].astype(bool)
            if not active_bridges.empty:
                admissible, reasons = _get_constraint_directed_rescue_candidates(
                    trade_table=candidate,
                    base_constraints=base_constraints,
                    rescue_eligible=remaining,
                    bridges=bridges,
                    relax_selected_corridors=relax_selected_corridors,
                )
            else:
                admissible = remaining
                reasons = pd.Series("", index=candidate.index, dtype=object)
                if sign_directed_rescue:
                    admissible = _get_sign_directed_rescue_candidates(
                        trade_table=candidate,
                        base_constraints=base_constraints,
                        rescue_eligible=remaining,
                    )
            candidate.loc[admissible, DIRECTED_RESCUE_CANDIDATE] = True
            ordered = (
                candidate.loc[admissible]
                .sort_values([TRADE_SCORE, SCORE_PER_TURNOVER], ascending=[False, False])
                .index
            )
            if ordered.empty:
                if not active_bridges.empty:
                    latest_error = ExecutionSolverInfeasibility(
                        _format_bridge_failure(bridges), bridges=bridges, trade_table=candidate
                    )
                return (None, latest_error, candidate)
            admitted = ordered[0]
            admission_count += 1
            candidate.loc[admitted, SELECTED_TRADE] = True
            candidate.loc[admitted, FEASIBILITY_RESCUE_TRADE] = True
            reason = reasons.loc[admitted]
            candidate.loc[admitted, RESCUE_REASON] = reason if reason else "solver_residual"
            post_bridges = _compute_selected_execution_bridges(
                base_constraints=base_constraints,
                trade_table=candidate,
                relax_selected_corridors=relax_selected_corridors,
                partition_groups=partition_groups,
            )
            if not _positive_bridges(post_bridges).empty:
                continue
            stage = "relaxed corridor" if relax_selected_corridors else "strict corridor"
            try:
                result = solve_selected_execution_portfolio(
                    trade_table=candidate,
                    base_constraints=base_constraints,
                    covariance=covariance,
                    optimiser_config=optimiser_config,
                    context=(
                        f"{context} {stage} constraint-directed rescue {admission_count}"
                    ).strip(),
                    relax_selected_corridors=relax_selected_corridors,
                    group_split_asset_classes=group_split_asset_classes,
                    partition_groups=partition_groups,
                )
            except ExecutionSolverInfeasibility as error:
                error.trade_table = candidate
                latest_error = error
                continue
            latest_result = result
            latest_error = None
            if result.accepted and result.compliant:
                return (result, None, candidate)

    strict_error = initial_error
    strict_audit = audited
    if expand_for_feasibility:
        strict_result, strict_error, strict_audit = run_linear_rescue(
            start=audited, relax_selected_corridors=False, failed_result=initial_result
        )
        if strict_result is not None:
            return finish_failed(strict_result)
    if not allow_corridor_relaxation:
        if strict_error is not None:
            strict_error.trade_table = strict_audit
            raise strict_error
        if latest_result is None:
            raise initial_error
        return finish_failed(latest_result)
    try:
        relaxed_initial = solve_selected_execution_portfolio(
            trade_table=audited,
            base_constraints=base_constraints,
            covariance=covariance,
            optimiser_config=optimiser_config,
            context=f"{context} hard-feasibility corridor relaxation".strip(),
            relax_selected_corridors=True,
            group_split_asset_classes=group_split_asset_classes,
            partition_groups=partition_groups,
        )
    except ExecutionSolverInfeasibility:
        relaxed_initial = None
    if relaxed_initial is not None and relaxed_initial.accepted and relaxed_initial.compliant:
        return relaxed_initial
    if relaxed_initial is not None:
        latest_result = relaxed_initial
    relaxed_error: Optional[ExecutionSolverInfeasibility] = None
    relaxed_audit = audited
    if expand_for_feasibility:
        relaxed_result, relaxed_error, relaxed_audit = run_linear_rescue(
            start=audited, relax_selected_corridors=True, failed_result=relaxed_initial
        )
        if relaxed_result is not None:
            return finish_failed(relaxed_result)
    if relaxed_error is not None:
        relaxed_error.trade_table = relaxed_audit
        raise relaxed_error
    if latest_result is None:
        raise initial_error
    return finish_failed(latest_result)


def _get_sign_directed_rescue_candidates(
    trade_table: pd.DataFrame, base_constraints: Constraints, rescue_eligible: pd.Series
) -> pd.Series:
    """Return rescue trades that reduce breached base cash/group residuals.

    If the post-mandatory base has no cash or group-bound breach, the original
    eligible set is returned because there is no defensible sign restriction.
    """
    index = trade_table.index
    eligible = rescue_eligible.reindex(index).fillna(False).astype(bool)
    base = trade_table[BASE_WEIGHT].astype(float)
    desired = trade_table[DESIRED_REWEIGHT].astype(float)
    cash = trade_table[SETTLEMENT_CASH].astype(bool)
    if int(cash.sum()) != 1:
        raise ValueError("directed rescue requires exactly one settlement-cash asset")
    cash_asset = index[cash][0]
    directed = pd.Series(False, index=index)
    has_breach = False
    default_min = 0.0 if base_constraints.is_long_only else -base_constraints.max_exposure
    product_min = (
        base_constraints.min_weights.reindex(index).fillna(default_min).astype(float)
        if base_constraints.min_weights is not None
        else pd.Series(default_min, index=index)
    )
    product_max = (
        base_constraints.max_weights.reindex(index)
        .fillna(base_constraints.max_exposure)
        .astype(float)
        if base_constraints.max_weights is not None
        else pd.Series(base_constraints.max_exposure, index=index)
    )
    cash_min = max(
        float(product_min.loc[cash_asset]), float(trade_table.loc[cash_asset, POLICY_MIN_WEIGHT])
    )
    cash_max = min(
        float(product_max.loc[cash_asset]), float(trade_table.loc[cash_asset, POLICY_MAX_WEIGHT])
    )
    if base.loc[cash_asset] < cash_min - WEIGHT_ZERO_TOLERANCE:
        has_breach = True
        directed |= desired.lt(-WEIGHT_ZERO_TOLERANCE)
    if base.loc[cash_asset] > cash_max + WEIGHT_ZERO_TOLERANCE:
        has_breach = True
        directed |= desired.gt(WEIGHT_ZERO_TOLERANCE)
    group_bounds = base_constraints.group_lower_upper_constraints
    if group_bounds is not None:
        loadings = group_bounds.group_loadings.reindex(index=index).fillna(0.0)
        exposure = loadings.mul(base, axis=0).sum(axis=0)
        group_min = group_bounds.group_min_allocation
        group_max = group_bounds.group_max_allocation
        for group in loadings.columns:
            effect = loadings[group] * desired
            if (
                group_max is not None
                and group in group_max.index
                and (exposure.loc[group] > group_max.loc[group] + WEIGHT_ZERO_TOLERANCE)
            ):
                has_breach = True
                directed |= effect.lt(-WEIGHT_ZERO_TOLERANCE)
            if (
                group_min is not None
                and group in group_min.index
                and (exposure.loc[group] < group_min.loc[group] - WEIGHT_ZERO_TOLERANCE)
            ):
                has_breach = True
                directed |= effect.gt(WEIGHT_ZERO_TOLERANCE)
    return eligible & directed if has_breach else eligible
