"""Cash-funded, selected-trade execution optimisation.

The policy layer first resolves mandatory lifecycle and desk boundary repairs
in a base portfolio. This module keeps exact mandatory targets pinned, exposes
one-sided minimum-buy and maximum-sell repairs through direction-preserving
corridors, freezes every unselected Rule 4 instrument, and leaves the
designated settlement-cash asset free to fund the movable instruments. The
escalation order is strict corridor, optional
minimum-prefix expansion, then an explicitly enabled and reported corridor
relaxation for selected Rule 4 instruments. Exact mandatory and unselected
rows stay pinned throughout. Product constraints are never relaxed except for an
explicitly audited exact-current desk pin that grandfathers an existing
instrument-bound breach at zero trade and the fixed numerical group-bound
normalization described below.

Feasibility rescue diagnoses numeric group and cash bridges, admits only
trades with the required signed constraint effect, and uses policy score only
to order that admissible set. Material group constraints remain unchanged;
sub-half-basis-point interval mismatches are snapped to the executable endpoint
and logged as rounding adjustments. See
``docs/implementation_notes/2026-08-11_execution_group_bound_rounding.md``.
"""
import logging
from dataclasses import dataclass
from dataclasses import replace
from typing import Optional

import cvxpy as cvx
import numpy as np
import pandas as pd

from optimalportfolios.optimization import (
    ConstraintEnforcementType,
    Constraints,
    OptimizationOutcome,
    OptimiserConfig,
    evaluate_constraint_residuals,
    wrapper_minimise_tracking_error,
)
from optimalportfolios.optimization.constraints import cvx_covar_variance
from optimalportfolios.optimization.covar_factorization import factorize_covariance
from optimalportfolios.optimization.solver_diagnostics import (
    diagnose_infeasibility,
    validate_solution,
)
from optimalportfolios.execution.v1.feasibility import compute_group_bound_bridges
from optimalportfolios.execution.v1.policy import (
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
    MATERIAL_TRADE_THRESHOLD,
    MODEL_ENTRY,
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
    TARGET_BAND_LOWER,
    TARGET_BAND_UPPER,
    TRADE_RANK,
    TRADE_SCORE,
    WEIGHT_ZERO_TOLERANCE,
)


from optimalportfolios.execution.v1._budget import _consume_projection
from optimalportfolios.execution.v1._constraint_scope import (
    _linear_feasibility_tolerance, _retain_original_group_bounds,
)
from optimalportfolios.execution.v1._projection_cache import _project
from optimalportfolios.execution._common.model import is_model_zero

logger = logging.getLogger(__name__)


GROUP_BOUND_ROUNDING_TOLERANCE = 0.5 / 10_000.0
CASH_FUNDING_IDENTITY_TOLERANCE = 1e-7
CASH_ROUNDING_RECONCILIATION_LIMIT = 1e-6
PROPOSED_WEIGHT = 'proposed_weight'
EXECUTED_TRADE = 'executed_trade'
MANDATORY_EXECUTED_TRADE = 'mandatory_executed_trade'
DISCRETIONARY_EXECUTED_TRADE = 'discretionary_executed_trade'
MANDATORY_CORRIDOR = 'mandatory_corridor'
MANDATORY_CORRIDOR_MIN = 'mandatory_corridor_min'
MANDATORY_CORRIDOR_MAX = 'mandatory_corridor_max'
MANDATORY_CORRIDOR_EXTRA_TRADE = 'mandatory_corridor_extra_trade'
TARGET_GAP = 'target_gap'
REWEIGHT_FUNDING = 'reweight_funding'
EXPECTED_CASH_WEIGHT = 'expected_cash_weight'
DIRECTED_RESCUE_CANDIDATE = 'directed_rescue_candidate'
RESCUE_REASON = 'rescue_reason'
STRICT_CORRIDOR_MIN = 'strict_corridor_min'
STRICT_CORRIDOR_MAX = 'strict_corridor_max'
CORRIDOR_RELAXATION = 'corridor_relaxation'
RELAXED_CORRIDOR_SOLVE = 'relaxed_corridor_solve'

MANDATORY_BUY_CORRIDOR_CATEGORIES = frozenset({
    'product_min_buy', 'instruction_min_buy',
})
MANDATORY_SELL_CORRIDOR_CATEGORIES = frozenset({
    'product_max_sell', 'instruction_max_sell',
})


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
        base_constraints: Constraints,
        trade_table: pd.DataFrame,
        relax_selected_corridors: bool,
) -> tuple[pd.Series, pd.Series]:
    """Return executable instrument bounds for one selected trade set."""
    risk_banded = {
        TARGET_BAND_LOWER, TARGET_BAND_UPPER,
    }.issubset(trade_table.columns)
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
    if risk_banded:
        required.update({MATERIAL_TRADE_THRESHOLD, MODEL_ENTRY})
    missing = required.difference(trade_table.columns)
    if missing:
        if risk_banded:
            raise NotImplementedError(f'incomplete target-band execution inputs: {sorted(missing)}')
        raise ValueError(f'execution trade table is missing columns {sorted(missing)}')
    index = trade_table.index
    if not index.is_unique:
        raise ValueError('execution trade table index must be unique')
    base = trade_table[BASE_WEIGHT].astype(float)
    target = trade_table[EFFECTIVE_MODEL_WEIGHT].astype(float)
    policy_min = trade_table[POLICY_MIN_WEIGHT].astype(float)
    policy_max = trade_table[POLICY_MAX_WEIGHT].astype(float)
    selected = trade_table[SELECTED_TRADE].astype(bool)
    sell_down = trade_table.get(
        CUTOFF_SELL_DOWN, pd.Series(False, index=index)
    ).astype(bool)
    rule4 = trade_table[RULE4_TRADE].astype(bool)
    cash = trade_table[SETTLEMENT_CASH].astype(bool)
    if int(cash.sum()) != 1:
        raise ValueError('execution optimisation requires exactly one settlement-cash asset')
    invalid_selection = selected & (~rule4 | cash)
    if invalid_selection.any():
        raise ValueError(
            'only non-cash Rule 4 trades may be selected: '
            f'{index[invalid_selection].tolist()}'
        )
    values = pd.concat([base, target, policy_min, policy_max], axis=1, sort=False)
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError('execution weights and policy bounds must be finite')

    default_min = 0.0 if base_constraints.is_long_only else -base_constraints.max_exposure
    product_min = (
        base_constraints.min_weights.reindex(index).fillna(default_min).astype(float)
        if base_constraints.min_weights is not None
        else pd.Series(default_min, index=index)
    )
    product_max = (
        base_constraints.max_weights.reindex(index).fillna(
            base_constraints.max_exposure
        ).astype(float)
        if base_constraints.max_weights is not None
        else pd.Series(base_constraints.max_exposure, index=index)
    )
    product_override = trade_table.get(
        PRODUCT_BOUND_OVERRIDE, pd.Series(False, index=index)
    ).astype(bool)
    product_min.loc[product_override] = trade_table.loc[
        product_override, CURRENT_WEIGHT
    ]
    product_max.loc[product_override] = trade_table.loc[
        product_override, CURRENT_WEIGHT
    ]
    hard_min = pd.concat([product_min, policy_min], axis=1, sort=False).max(axis=1)
    hard_max = pd.concat([product_max, policy_max], axis=1, sort=False).min(axis=1)
    invalid_hard = hard_min > hard_max + WEIGHT_ZERO_TOLERANCE
    if invalid_hard.any():
        raise ValueError(
            'execution product and policy bounds do not intersect:\n'
            f'{pd.DataFrame({"min": hard_min, "max": hard_max}).loc[invalid_hard]}'
        )

    if risk_banded:
        if relax_selected_corridors:
            raise ValueError(
                'risk-budget-ranked execution never relaxes target bands'
            )
        mandatory_entry = trade_table[MODEL_ENTRY].astype(bool)
        mandatory_buy, mandatory_sell = _get_mandatory_corridor_masks(
            trade_table
        )
        mandatory_corridor = mandatory_buy | mandatory_sell
        movable = selected | mandatory_entry | mandatory_corridor
        selected_min = pd.concat([
            trade_table[TARGET_BAND_LOWER].astype(float), hard_min
        ], axis=1, sort=False).max(axis=1)
        selected_max = pd.concat([
            trade_table[TARGET_BAND_UPPER].astype(float), hard_max
        ], axis=1, sort=False).min(axis=1)
        minimum_ticket = trade_table[MATERIAL_TRADE_THRESHOLD].astype(float)
        selected_buy = selected & target.gt(base + WEIGHT_ZERO_TOLERANCE)
        selected_sell = selected & target.lt(base - WEIGHT_ZERO_TOLERANCE)
        selected_min.loc[selected_buy] = pd.concat([
            selected_min.loc[selected_buy],
            base.loc[selected_buy] + minimum_ticket.loc[selected_buy],
        ], axis=1, sort=False).max(axis=1)
        selected_max.loc[selected_sell] = pd.concat([
            selected_max.loc[selected_sell],
            base.loc[selected_sell] - minimum_ticket.loc[selected_sell],
        ], axis=1, sort=False).min(axis=1)
        # The boundary repair is compulsory, but paying that ticket makes the
        # same-direction move from the repaired base toward the clipped model
        # available without admitting another trade.
        selected_min.loc[mandatory_buy] = base.loc[mandatory_buy]
        selected_max.loc[mandatory_buy] = target.loc[mandatory_buy]
        selected_min.loc[mandatory_sell] = target.loc[mandatory_sell]
        selected_max.loc[mandatory_sell] = base.loc[mandatory_sell]
    else:
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
        # A sell-down row is a continuous feasibility variable, independent of
        # Rule 4 selection. It may be retained at its pre-trade weight or sold
        # as far as hard product bounds permit, but can never be increased.
        selected_min.loc[sell_down] = hard_min.loc[sell_down]
        selected_max.loc[sell_down] = pd.concat(
            [trade_table.loc[sell_down, CURRENT_WEIGHT], hard_max.loc[sell_down]],
            axis=1, sort=False).min(axis=1)
        # One-sided mandatory instructions define an admissible final interval,
        # not an equality at the first compliant boundary.
        selected_min.loc[mandatory_buy] = base.loc[mandatory_buy]
        selected_max.loc[mandatory_buy] = hard_max.loc[mandatory_buy]
        selected_min.loc[mandatory_sell] = hard_min.loc[mandatory_sell]
        selected_max.loc[mandatory_sell] = base.loc[mandatory_sell]
    invalid_selected = movable & (
        selected_min > selected_max + WEIGHT_ZERO_TOLERANCE
    )
    if invalid_selected.any():
        details = pd.DataFrame({
            'base': base,
            'target': target,
            'hard_min': hard_min,
            'hard_max': hard_max,
            'effective_min': selected_min,
            'effective_max': selected_max,
        }).loc[invalid_selected]
        raise ValueError(
            'selected execution corridors do not intersect hard bounds:\n'
            f'{details.to_string()}'
        )
    if not relax_selected_corridors and not risk_banded:
        missing_base = movable & (
            (base < selected_min - WEIGHT_ZERO_TOLERANCE)
            | (base > selected_max + WEIGHT_ZERO_TOLERANCE)
        )
        if missing_base.any():
            details = pd.DataFrame({
                'base': base,
                'corridor_min': selected_min,
                'corridor_max': selected_max,
            }).loc[missing_base]
            raise ValueError(
                'selected corridors must contain their post-mandatory base:\n'
                f'{details.to_string()}'
            )

    frozen = ~movable & ~cash
    frozen_violation = frozen & (
        (base < hard_min - WEIGHT_ZERO_TOLERANCE)
        | (base > hard_max + WEIGHT_ZERO_TOLERANCE)
    )
    if frozen_violation.any():
        details = pd.DataFrame({
            'base': base,
            'hard_min': hard_min,
            'hard_max': hard_max,
        }).loc[frozen_violation]
        raise ValueError(
            'fixed execution base violates hard bounds:\n'
            f'{details.to_string()}'
        )

    min_weights = base.copy()
    max_weights = base.copy()
    min_weights.loc[movable] = selected_min.loc[movable]
    max_weights.loc[movable] = selected_max.loc[movable]
    min_weights.loc[cash] = hard_min.loc[cash]
    max_weights.loc[cash] = hard_max.loc[cash]
    return min_weights, max_weights


def _solve_group_split_tracking_error(
        covariance: pd.DataFrame,
        benchmark_weights: pd.Series,
        current_weights: pd.Series,
        asset_classes: pd.Series,
        constraints: Constraints,
        optimiser_config: OptimiserConfig,
        context: str,
) -> tuple[pd.Series, OptimizationOutcome]:
    """Minimize the sum of within-asset-class tracking variances.

    ``optimalportfolios`` exposes the covariance atom and complete constraint
    compiler but no wrapper for this deliberately block-split objective, so
    this ROSAA execution method composes those two canonical primitives here.
    """
    index = covariance.index
    if not index.equals(covariance.columns):
        raise ValueError('execution covariance index and columns must match')
    benchmark = benchmark_weights.reindex(index).astype(float)
    current = current_weights.reindex(index).astype(float)
    groups = asset_classes.reindex(index)
    if benchmark.isna().any() or current.isna().any() or groups.isna().any():
        raise ValueError('group-split execution inputs must be complete')
    sigma = covariance.to_numpy(dtype=float)
    if not np.isfinite(sigma).all() or not np.allclose(
            sigma, sigma.T, atol=1e-10, rtol=0.0):
        raise ValueError('group-split execution covariance must be finite and symmetric')
    aligned_constraints = constraints.update_with_valid_tickers(
        valid_tickers=index.tolist(),
        weights_0=current,
        benchmark_weights=benchmark,
        context=context,
        relax_frozen_group_bounds=False,
    )
    factorization = (
        factorize_covariance(sigma)
        if optimiser_config.factorize_covar else None
    )
    solver_covar = (
        factorization.covar if factorization is not None else sigma
    )
    covar_psd = cvx.psd_wrap(solver_covar)
    w = cvx.Variable(len(index), nonneg=aligned_constraints.is_long_only)
    active = w - benchmark.to_numpy(dtype=float)
    objective_terms = []
    for asset_class in dict.fromkeys(asset_classes):
        mask = groups.eq(asset_class).to_numpy(dtype=float)
        if mask.any():
            objective_terms.append(cvx_covar_variance(
                active_weights=cvx.multiply(mask, active),
                covar=covar_psd,
                covar_factorization=factorization,
            ))
    problem = cvx.Problem(
        cvx.Minimize(sum(objective_terms)),
        aligned_constraints.set_cvx_all_constraints(
            w=w,
            covar=covar_psd,
            covar_factorization=factorization,
        ),
    )
    solver_options = (
        {'max_iter': 1000}
        if optimiser_config.solver.upper() == 'CLARABEL' else {}
    )
    try:
        problem.solve(
            solver=optimiser_config.solver,
            verbose=optimiser_config.verbose,
            **solver_options,
        )
        status = problem.status
    except cvx.error.SolverError:
        status = 'solver_error'
    outcome = validate_solution(
        optimal_weights=None if w.value is None else np.asarray(w.value).ravel(),
        problem_status=status,
        constraints=aligned_constraints,
        n=len(index),
        solver=optimiser_config.solver,
        context=context,
        covar=solver_covar,
        covar_factorization=factorization,
    )
    return (
        pd.Series(outcome.weights, index=index, name='group_split_tracking_error'),
        outcome,
    )


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
    return bridges.loc[
        bridges[["bridge_min", "bridge_max"]].max(axis=1).gt(_linear_feasibility_tolerance())
    ]


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
    if _retain_original_group_bounds() or group_bounds is None or bridges.empty:
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
    risk_banded = {
        TARGET_BAND_LOWER, TARGET_BAND_UPPER,
    }.issubset(trade_table.columns)
    effective_constraints = (
        base_constraints
        if risk_banded else _snap_group_bounds_within_rounding_tolerance(
            base_constraints=base_constraints,
            bridges=bridges,
        )
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
            _format_bridge_failure(bridges),
            bridges=bridges,
            trade_table=trade_table,
        )

    beta_constraint = effective_constraints.benchmark_beta_constraint
    if beta_constraint is not None and beta_constraint.beta_loadings is None:
        raise ValueError('execution solve requires resolved benchmark beta loadings')
    try:
        common_updates = {
            'min_weights': min_weights,
            'max_weights': max_weights,
            'weights_0': trade_table[CURRENT_WEIGHT].astype(float),
            'tre_utility_weight': None,
            'turnover_utility_weight': None,
            'constraint_enforcement_type': (
                ConstraintEnforcementType.FORCED_CONSTRAINTS
            ),
        }
        if risk_banded:
            group_tre = effective_constraints.group_tracking_error_constraint
            group_turnover = effective_constraints.group_turnover_constraint
            common_updates.update({
                'group_tracking_error_constraint': (
                    group_tre
                    if group_tre is not None
                    and group_tre.group_tre_vols is not None
                    else None
                ),
                'group_turnover_constraint': (
                    group_turnover
                    if group_turnover is not None
                    and group_turnover.group_max_turnover is not None
                    else None
                ),
            })
        else:
            common_updates.update({
                'tracking_err_vol_constraint': None,
                'turnover_constraint': None,
                'turnover_costs': None,
                'group_tracking_error_constraint': None,
                'group_turnover_constraint': None,
            })
        return effective_constraints.copy(**common_updates)
    except ValueError as error:
        if str(error).startswith('Infeasible constraints detected'):
            raise ExecutionSolverInfeasibility(
                str(error), bridges=bridges, trade_table=trade_table
            ) from error
        raise


def _audit_exact_cash_bootstrap_model(
        trade_table: pd.DataFrame,
        base_constraints: Constraints,
        selected_constraints: Constraints,
        covariance: pd.DataFrame,
        context: str,
) -> Optional[OptimizationOutcome]:
    """Recover a uniquely pinned cash model after independent hard audits.

    Numerical projection can report infeasible when every non-cash weight is
    fixed to an already feasible model. In that narrow case the pinned weights
    and full investment determine the unique candidate. Only
    existing numerical-dust and settlement-cash reconciliation tolerances may
    separate it from the raw model. Other live books and movable corridors
    remain on the normal solver path; product bounds are never rounded here.
    """
    cash = trade_table[SETTLEMENT_CASH].astype(bool)
    if int(cash.sum()) != 1:
        return None
    current = trade_table[CURRENT_WEIGHT].astype(float)
    if (abs(float(current.loc[cash].iloc[0]) - 1.0) > 1e-6
            or float(current.loc[~cash].abs().sum()) > 1e-6):
        return None
    model = trade_table[RAW_MODEL_WEIGHT].astype(float)
    if not np.isfinite(model.to_numpy()).all():
        return None
    minimum = selected_constraints.min_weights.reindex(model.index)
    maximum = selected_constraints.max_weights.reindex(model.index)
    if (not np.allclose(minimum.loc[~cash], maximum.loc[~cash], atol=1e-12, rtol=0.0)
            or not np.isclose(selected_constraints.min_exposure, 1.0, atol=1e-12, rtol=0.0)
            or not np.isclose(selected_constraints.max_exposure, 1.0, atol=1e-12, rtol=0.0)):
        return None
    unchanged = np.isclose(minimum.loc[~cash], model.loc[~cash], atol=1e-12, rtol=0.0)
    dust = (is_model_zero(model.loc[~cash])
            & minimum.loc[~cash].abs().le(1e-12))
    if not (unchanged | dust).all():
        return None
    candidate = model.copy()
    candidate.loc[~cash] = minimum.loc[~cash]
    candidate.loc[cash] = 1.0 - float(candidate.loc[~cash].sum())
    if abs(float((candidate - model).loc[cash].iloc[0])) > CASH_ROUNDING_RECONCILIATION_LIMIT:
        return None
    # Audit the original product ranges as well as the selected intervals.
    # The recovery must not inherit the normal execution-local group snap.
    original_product = selected_constraints.copy(
        min_weights=base_constraints.min_weights,
        max_weights=base_constraints.max_weights,
        group_lower_upper_constraints=base_constraints.group_lower_upper_constraints,
    )
    residuals = ()
    for constraints in (selected_constraints, original_product):
        residuals += evaluate_constraint_residuals(
            weights=candidate.to_numpy(dtype=float),
            constraints=constraints,
            covar=covariance.loc[model.index, model.index].to_numpy(dtype=float),
            tolerance=CASH_FUNDING_IDENTITY_TOLERANCE,
        )
    if not residuals or not all(r.passed for r in residuals if r.hard):
        return None
    logger.warning(
        '%s cash bootstrap recovered the uniquely pinned model after numerical '
        'projection failure; original product bounds and selected intervals audited',
        context,
    )
    return OptimizationOutcome(
        weights=candidate.to_numpy(dtype=float), accepted=True,
        solver='AUDITED_MODEL', status='audited_exact_cash_model', context=context,
        reason='Pinned cash-bootstrap model passed original hard-constraint audits',
        constraint_residuals=residuals, constraints=selected_constraints,
    )


def _solve_selected_execution_portfolio_once(
        trade_table: pd.DataFrame,
        base_constraints: Constraints,
        covariance: pd.DataFrame,
        optimiser_config: OptimiserConfig = OptimiserConfig(solver='MOSEK'),
        context: str = '',
        relax_selected_corridors: bool = False,
        group_split_asset_classes: Optional[pd.Series] = None,
        partition_groups: tuple[str, ...] = (),
) -> ExecutionOptimizationResult:
    """Run one strict- or relaxed-corridor minimum-TRE solve."""
    constraints = build_selected_execution_constraints(
        base_constraints=base_constraints,
        trade_table=trade_table,
        relax_selected_corridors=relax_selected_corridors,
        partition_groups=partition_groups,
    )
    if group_split_asset_classes is None:
        weights, outcome = _project(wrapper_minimise_tracking_error,
            pd_covar=covariance,
            benchmark_weights=trade_table[RAW_MODEL_WEIGHT],
            constraints=constraints,
            weights_0=trade_table[CURRENT_WEIGHT],
            optimiser_config=optimiser_config,
            context=context,
        )
    else:
        weights, outcome = _solve_group_split_tracking_error(
            covariance=covariance.loc[trade_table.index, trade_table.index],
            benchmark_weights=trade_table[RAW_MODEL_WEIGHT],
            current_weights=trade_table[CURRENT_WEIGHT],
            asset_classes=group_split_asset_classes,
            constraints=constraints,
            optimiser_config=optimiser_config,
            context=context,
        )
    if not outcome.accepted and group_split_asset_classes is None:
        audited_model = _audit_exact_cash_bootstrap_model(
            trade_table=trade_table,
            base_constraints=base_constraints,
            selected_constraints=constraints,
            covariance=covariance,
            context=context,
        )
        if audited_model is not None:
            weights = pd.Series(audited_model.weights, index=trade_table.index)
            outcome = audited_model
    dust_rounded = pd.Series(False, index=trade_table.index)
    if (group_split_asset_classes is None
            and outcome.accepted and outcome.compliant):
        sell_down = trade_table.get(
            CUTOFF_SELL_DOWN, pd.Series(False, index=trade_table.index)
        ).astype(bool)
        threshold = float(
            trade_table[SELL_DOWN_DUST_THRESHOLD].iloc[0]
            if SELL_DOWN_DUST_THRESHOLD in trade_table else 0.0
        )
        dust_candidates = trade_table.index[
            sell_down
            & weights.gt(WEIGHT_ZERO_TOLERANCE)
            & weights.lt(threshold)
        ]
        cash = trade_table[SETTLEMENT_CASH].astype(bool)
        cash_asset = trade_table.index[cash][0]
        group_bounds = constraints.group_lower_upper_constraints
        cleaned = weights.copy()
        for instrument in dust_candidates:
            weight = float(cleaned.loc[instrument])
            has_group_slack = True
            if group_bounds is not None:
                loadings = group_bounds.group_loadings.reindex(
                    index=trade_table.index
                ).fillna(0.0)
                minimum = group_bounds.group_min_allocation
                exposure = loadings.mul(cleaned, axis=0).sum(axis=0)
                loaded_groups = loadings.columns[
                    loadings.loc[instrument].gt(WEIGHT_ZERO_TOLERANCE)
                ]
                if minimum is not None:
                    for group in loaded_groups:
                        if (group in minimum.index
                                and np.isfinite(minimum.loc[group])
                                and exposure.loc[group] - minimum.loc[group]
                                < weight - WEIGHT_ZERO_TOLERANCE):
                            has_group_slack = False
                            break
            if not has_group_slack:
                continue
            candidate = cleaned.copy()
            candidate.loc[instrument] = 0.0
            candidate.loc[cash_asset] += weight
            # The wrapper expands its returned Series to the full universe,
            # but the outcome keeps the exact filtered solver order and risk
            # factorization. Audit each contract in its own aligned universe.
            solver_constraints = outcome.constraints
            solver_index = solver_constraints.min_weights.index
            excluded = candidate.index.difference(solver_index)
            if not candidate.loc[excluded].abs().le(WEIGHT_ZERO_TOLERANCE).all():
                # Cleanup cannot allocate to an asset excluded from the solve,
                # including a settlement-cash asset without usable covariance.
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
            if all(residual.passed for residual in residuals if residual.hard):
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
        # The solver accepts a small budget residual. Fund that residual from
        # settlement cash only, then audit the adjusted portfolio in both the
        # full execution universe and the solver's filtered universe.
        funded_cash = float(
            trade_table[CURRENT_WEIGHT].sum() - weights.loc[~cash].sum()
        )
        cash_adjustment = funded_cash - float(weights.loc[cash_asset])
        if abs(cash_adjustment) > CASH_FUNDING_IDENTITY_TOLERANCE:
            if abs(cash_adjustment) > CASH_ROUNDING_RECONCILIATION_LIMIT:
                raise RuntimeError(
                    'execution cash rounding adjustment exceeds limit: '
                    f'adjustment={cash_adjustment:.10f}'
                )
            corrected = weights.copy()
            corrected.loc[cash_asset] = funded_cash
            if (funded_cash < constraints.min_weights.loc[cash_asset]
                    - WEIGHT_ZERO_TOLERANCE
                    or funded_cash > constraints.max_weights.loc[cash_asset]
                    + WEIGHT_ZERO_TOLERANCE):
                raise RuntimeError(
                    'execution cash rounding adjustment breaches cash bounds: '
                    f'cash={funded_cash:.10f}'
                )
            solver_constraints = outcome.constraints
            solver_index = solver_constraints.min_weights.index
            if cash_asset not in solver_index:
                raise RuntimeError(
                    'execution cash rounding adjustment requires cash in the solver universe'
                )
            excluded = corrected.index.difference(solver_index)
            if not corrected.loc[excluded].abs().le(WEIGHT_ZERO_TOLERANCE).all():
                raise RuntimeError(
                    'execution cash rounding adjustment leaves excluded assets invested'
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
            if not all(residual.passed for residual in residuals if residual.hard):
                raise RuntimeError(
                    'execution cash rounding adjustment breaches hard constraints'
                )
            weights = corrected
            outcome = replace(
                outcome,
                weights=corrected.reindex(solver_index).to_numpy(dtype=float),
                constraint_residuals=residuals,
            )
            logger.info(
                '%s execution settlement cash rounded by %.10f',
                context, cash_adjustment,
            )
    enriched = trade_table.copy()
    enriched[PROPOSED_WEIGHT] = weights
    enriched[EXECUTED_TRADE] = weights - enriched[CURRENT_WEIGHT]
    mandatory = enriched[MANDATORY_TRADE].astype(bool)
    enriched[MANDATORY_EXECUTED_TRADE] = (
        enriched[BASE_WEIGHT] - enriched[CURRENT_WEIGHT]
    )
    enriched[DISCRETIONARY_EXECUTED_TRADE] = weights - enriched[BASE_WEIGHT]
    enriched.loc[mandatory, MANDATORY_EXECUTED_TRADE] = enriched.loc[
        mandatory, EXECUTED_TRADE
    ]
    enriched.loc[mandatory, DISCRETIONARY_EXECUTED_TRADE] = 0.0
    enriched[TARGET_GAP] = weights - enriched[EFFECTIVE_MODEL_WEIGHT]
    risk_banded = {
        TARGET_BAND_LOWER, TARGET_BAND_UPPER,
    }.issubset(enriched.columns)
    if risk_banded:
        strict_min = enriched[BASE_WEIGHT].astype(float).copy()
        strict_max = enriched[BASE_WEIGHT].astype(float).copy()
        selected = enriched[SELECTED_TRADE].astype(bool)
        band_movable = (
            selected | enriched[MODEL_ENTRY].astype(bool)
        )
        strict_min.loc[band_movable] = enriched.loc[
            band_movable, TARGET_BAND_LOWER
        ]
        strict_max.loc[band_movable] = enriched.loc[
            band_movable, TARGET_BAND_UPPER
        ]
        minimum_ticket = enriched[MATERIAL_TRADE_THRESHOLD].astype(float)
        target = enriched[EFFECTIVE_MODEL_WEIGHT].astype(float)
        base = enriched[BASE_WEIGHT].astype(float)
        selected_buy = selected & target.gt(base + WEIGHT_ZERO_TOLERANCE)
        selected_sell = selected & target.lt(base - WEIGHT_ZERO_TOLERANCE)
        strict_min.loc[selected_buy] = pd.concat([
            strict_min.loc[selected_buy],
            base.loc[selected_buy] + minimum_ticket.loc[selected_buy],
        ], axis=1, sort=False).max(axis=1)
        strict_max.loc[selected_sell] = pd.concat([
            strict_max.loc[selected_sell],
            base.loc[selected_sell] - minimum_ticket.loc[selected_sell],
        ], axis=1, sort=False).min(axis=1)
        mandatory_buy, mandatory_sell = _get_mandatory_corridor_masks(enriched)
        strict_min.loc[mandatory_buy] = base.loc[mandatory_buy]
        strict_max.loc[mandatory_buy] = target.loc[mandatory_buy]
        strict_min.loc[mandatory_sell] = target.loc[mandatory_sell]
        strict_max.loc[mandatory_sell] = base.loc[mandatory_sell]
    else:
        strict_min = pd.concat(
            [enriched[BASE_WEIGHT], enriched[EFFECTIVE_MODEL_WEIGHT]], axis=1
        , sort=False).min(axis=1)
        strict_max = pd.concat(
            [enriched[BASE_WEIGHT], enriched[EFFECTIVE_MODEL_WEIGHT]], axis=1
        , sort=False).max(axis=1)
    selected = enriched[SELECTED_TRADE].astype(bool)
    sell_down = enriched.get(
        CUTOFF_SELL_DOWN, pd.Series(False, index=enriched.index)
    ).astype(bool)
    if not risk_banded:
        mandatory_buy, mandatory_sell = _get_mandatory_corridor_masks(enriched)
    mandatory_corridor = mandatory_buy | mandatory_sell
    interval_rows = sell_down | mandatory_corridor
    if not risk_banded and constraints.min_weights is not None:
        strict_min.loc[interval_rows] = constraints.min_weights.reindex(
            enriched.index
        ).loc[interval_rows]
    if not risk_banded and constraints.max_weights is not None:
        strict_max.loc[interval_rows] = constraints.max_weights.reindex(
            enriched.index
        ).loc[interval_rows]
    relaxation_eligible = selected
    enriched[STRICT_CORRIDOR_MIN] = strict_min
    enriched[STRICT_CORRIDOR_MAX] = strict_max
    enriched[MANDATORY_CORRIDOR] = mandatory_corridor
    enriched[MANDATORY_CORRIDOR_MIN] = strict_min.where(mandatory_corridor)
    enriched[MANDATORY_CORRIDOR_MAX] = strict_max.where(mandatory_corridor)
    enriched[MANDATORY_CORRIDOR_EXTRA_TRADE] = (
        weights - enriched[BASE_WEIGHT]
    ).where(mandatory_corridor, 0.0)
    enriched[CORRIDOR_RELAXATION] = (
        (strict_min - weights).clip(lower=0.0)
        + (weights - strict_max).clip(lower=0.0)
    ).where(relaxation_eligible, 0.0)
    enriched[RELAXED_CORRIDOR_SOLVE] = relax_selected_corridors
    enriched[CUTOFF_SELL_DOWN_RETENTION] = weights.where(sell_down)
    retained_fraction = weights.div(
        enriched[CURRENT_WEIGHT].replace(0.0, np.nan)
    )
    enriched[CUTOFF_SELL_DOWN_RETAINED_FRACTION] = retained_fraction.where(
        sell_down
    )
    enriched[CUTOFF_SELL_DOWN_DUST_ROUNDED] = dust_rounded

    cash = enriched[SETTLEMENT_CASH].astype(bool)
    cash_asset = enriched.index[cash][0]
    noncash = ~cash
    mandatory_funding = float(
        enriched.loc[noncash, MANDATORY_FUNDING].sum()
    )
    reweight_funding = float(
        (weights.loc[noncash] - enriched.loc[noncash, BASE_WEIGHT]).sum()
    )
    expected_cash = float(
        enriched.loc[cash_asset, CURRENT_WEIGHT]
        - mandatory_funding
        - reweight_funding
    )
    enriched[REWEIGHT_FUNDING] = 0.0
    enriched.loc[cash_asset, REWEIGHT_FUNDING] = reweight_funding
    enriched[EXPECTED_CASH_WEIGHT] = np.nan
    enriched.loc[cash_asset, EXPECTED_CASH_WEIGHT] = expected_cash
    if outcome.accepted and not np.isclose(
            weights.loc[cash_asset], expected_cash,
            atol=CASH_FUNDING_IDENTITY_TOLERANCE, rtol=0.0
    ):
        raise RuntimeError(
            'execution cash funding identity failed: '
            f'actual={weights.loc[cash_asset]:.10f}, expected={expected_cash:.10f}'
        )
    if outcome.accepted and outcome.compliant:
        for instrument in enriched.index[mandatory_corridor]:
            logger.info(
                '%s mandatory corridor: instrument=%s category=%s '
                'current=%.8f boundary=%.8f interval=[%.8f, %.8f] '
                'proposed=%.8f extra_from_boundary=%.8f',
                context or 'execution',
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
        optimiser_config: OptimiserConfig = OptimiserConfig(solver='MOSEK'),
        context: str = '',
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
        group_split_asset_classes: Optional level-0 labels selecting the
            target-band policy's within-class tracking-variance objective.

    Returns:
        Proposed weights, enriched trade table and structured solver audit.
    """
    if group_split_asset_classes is not None and not {
            TARGET_BAND_LOWER, TARGET_BAND_UPPER, MATERIAL_TRADE_THRESHOLD,
    }.issubset(trade_table.columns):
        raise NotImplementedError("group-split execution requires a prepared target-band policy")
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
        preserve_model_directions: bool = False,
) -> tuple[pd.Series, pd.Series]:
    """Return candidates with signed capacity against a positive bridge.

    Score never controls admissibility. It is used only by the caller to
    order the admissible set.
    """
    index = trade_table.index
    eligible = rescue_eligible.reindex(index).fillna(False).astype(bool)
    risk_banded = {
        TARGET_BAND_LOWER, TARGET_BAND_UPPER,
    }.issubset(trade_table.columns)
    if risk_banded:
        base = trade_table[BASE_WEIGHT].astype(float)
        target = trade_table[EFFECTIVE_MODEL_WEIGHT].astype(float)
        ticket = trade_table[MATERIAL_TRADE_THRESHOLD].astype(float)
        can_buy = (
            target.gt(base + WEIGHT_ZERO_TOLERANCE)
            & trade_table[TARGET_BAND_UPPER].astype(float).ge(base + ticket)
        )
        can_sell = (
            target.lt(base - WEIGHT_ZERO_TOLERANCE)
            & trade_table[TARGET_BAND_LOWER].astype(float).le(base - ticket)
        )
    else:
        desired = trade_table[DESIRED_REWEIGHT].astype(float)
        can_buy = desired.gt(WEIGHT_ZERO_TOLERANCE)
        can_sell = desired.lt(-WEIGHT_ZERO_TOLERANCE)
    if relax_selected_corridors and not preserve_model_directions and not risk_banded:
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
        if group_bounds is not None else pd.DataFrame(index=index)
    )
    lower, upper = _build_selected_execution_bound_series(
        base_constraints, trade_table, relax_selected_corridors,
    )
    raw_bridges = compute_group_bound_bridges(
        lower, upper, loadings, bridges['group_min'], bridges['group_max'],
    )
    for group, row in active.iterrows():
        if group == 'cash':
            if row['bridge_min'] > WEIGHT_ZERO_TOLERANCE:
                matches = can_sell
                reason = 'cash:min'
            else:
                matches = can_buy
                reason = 'cash:max'
        elif group in loadings.columns:
            cash_asset = trade_table.index[trade_table[SETTLEMENT_CASH].astype(bool)][0]
            # Every discretionary trade has an opposite settlement-cash leg.
            # In particular, buying a risk asset reduces total Liquidity.
            loading = loadings[group] - loadings.at[cash_asset, group]
            if row['bridge_min'] > WEIGHT_ZERO_TOLERANCE:
                matches = (
                    loading.gt(0.0) & can_buy
                    | loading.lt(0.0) & can_sell
                )
                reason = f'{group}:min'
            else:
                matches = (
                    loading.gt(0.0) & can_sell
                    | loading.lt(0.0) & can_buy
                )
                reason = f'{group}:max'
            membership = loadings[group]
            # Raw interval capacity also matters: a zero net-exposure trade
            # may be the funding counterpart to an already-selected trade.
            if raw_bridges.at[group, 'bridge_min'] > WEIGHT_ZERO_TOLERANCE:
                matches |= (membership.gt(0.0) & can_buy
                            | membership.lt(0.0) & can_sell)
            if raw_bridges.at[group, 'bridge_max'] > WEIGHT_ZERO_TOLERANCE:
                matches |= (membership.gt(0.0) & can_sell
                            | membership.lt(0.0) & can_buy)
            if membership.isin([0.0, 1.0]).all():
                # Budget-complement capacity can require a funding counterpart
                # with zero direct group exposure (e.g. buy outside Equity to
                # fund a selected Equity sale when cash is already capped).
                outside = membership.eq(0.0)
                if (row['bridge_max'] > WEIGHT_ZERO_TOLERANCE
                        and row['e_min'] > raw_bridges.at[group, 'e_min']
                        + WEIGHT_ZERO_TOLERANCE):
                    matches |= outside & can_buy
                if (row['bridge_min'] > WEIGHT_ZERO_TOLERANCE
                        and row['e_max'] < raw_bridges.at[group, 'e_max']
                        - WEIGHT_ZERO_TOLERANCE):
                    matches |= outside & can_sell
        else:
            continue
        matches &= eligible
        directed |= matches
        for instrument in index[matches]:
            reason_sets[instrument].add(reason)
    reasons = pd.Series(
        {
            instrument: '|'.join(sorted(reason_sets[instrument]))
            for instrument in index
        },
        index=index,
        dtype=object,
    )
    return eligible & directed, reasons


def _get_group_tre_rescue_candidates(
        trade_table: pd.DataFrame,
        base_constraints: Constraints,
        covariance: pd.DataFrame,
        rescue_eligible: pd.Series,
) -> tuple[pd.Series, pd.Series]:
    """Return ranked candidates loaded on a breached hard group-TRE constraint."""
    index = trade_table.index
    eligible = rescue_eligible.reindex(index).fillna(False).astype(bool)
    directed = pd.Series(False, index=index)
    reason_sets = {instrument: set() for instrument in index}
    group_tre = base_constraints.group_tracking_error_constraint
    if group_tre is None or group_tre.group_tre_vols is None:
        return directed, pd.Series('', index=index, dtype=object)

    loadings = group_tre.group_loadings.reindex(index=index).fillna(0.0)
    covariance_array = covariance.loc[index, index].to_numpy(dtype=float)
    active = (
        trade_table[BASE_WEIGHT].astype(float)
        - trade_table[RAW_MODEL_WEIGHT].astype(float)
    )
    for group in loadings.columns:
        if group not in group_tre.group_tre_vols.index:
            continue
        limit = float(group_tre.group_tre_vols.loc[group])
        if not np.isfinite(limit):
            continue
        loading = loadings[group].astype(float)
        group_active = (loading * active).to_numpy(dtype=float)
        group_variance = float(group_active @ covariance_array @ group_active)
        group_risk = float(np.sqrt(max(group_variance, 0.0)))
        if group_risk <= limit + WEIGHT_ZERO_TOLERANCE:
            continue
        matches = eligible & loading.abs().gt(WEIGHT_ZERO_TOLERANCE)
        directed |= matches
        reason = f'group_tre:{group}'
        for instrument in index[matches]:
            reason_sets[instrument].add(reason)
    reasons = pd.Series({
        instrument: '|'.join(sorted(reason_sets[instrument]))
        for instrument in index
    }, index=index, dtype=object)
    return directed, reasons


def _is_proven_infeasible(result: Optional[ExecutionOptimizationResult]) -> bool:
    """Return whether the solver explicitly proved the attempted problem infeasible."""
    if result is None or result.outcome.status is None:
        return False
    return 'infeasible' in str(result.outcome.status).lower()


def solve_feasible_execution_portfolio(
        trade_table: pd.DataFrame,
        base_constraints: Constraints,
        covariance: pd.DataFrame,
        optimiser_config: OptimiserConfig = OptimiserConfig(solver='MOSEK'),
        context: str = '',
        expand_for_feasibility: bool = False,
        sign_directed_rescue: bool = False,
        allow_corridor_relaxation: bool = False,
        group_split_asset_classes: Optional[pd.Series] = None,
        partition_groups: tuple[str, ...] = (),
        preserve_model_directions: bool = False,
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
        group_split_asset_classes: Optional level-0 labels selecting the
            target-band policy objective and rank-ordered rescue.

    Returns:
        Accepted repaired solution, or the last failed solver outcome with terminal diagnostics.
    """
    if group_split_asset_classes is not None and not {
            TARGET_BAND_LOWER, TARGET_BAND_UPPER, MATERIAL_TRADE_THRESHOLD,
    }.issubset(trade_table.columns):
        raise NotImplementedError("group-split execution requires a prepared target-band policy")
    audited = trade_table.copy()
    requested = audited[SELECTED_TRADE].astype(bool)
    audited[REQUESTED_TRADE] = requested
    audited[FEASIBILITY_RESCUE_TRADE] = False
    audited[DIRECTED_RESCUE_CANDIDATE] = False
    audited[RESCUE_REASON] = ''

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
    if (initial_result is not None and initial_result.accepted
            and initial_result.compliant):
        return initial_result

    if group_split_asset_classes is None:
        rescue_eligible = (
            audited[REBALANCE_CADENCE_ELIGIBLE].astype(bool)
            & audited[RULE4_TRADE].astype(bool)
            & audited[DESIRED_REWEIGHT].abs().gt(WEIGHT_ZERO_TOLERANCE)
            & ~requested
        )
    else:
        rescue_eligible = (
            audited[REBALANCE_CADENCE_ELIGIBLE].astype(bool)
            & audited[RULE4_TRADE].astype(bool)
            & ~requested
        )

    latest_result = initial_result

    def finish_failed(result: ExecutionOptimizationResult) -> ExecutionOptimizationResult:
        """Diagnose only the terminal solve; diagnostic slacks are never applied."""
        if ((result.accepted and result.compliant)
                or not optimiser_config.diagnose_infeasibility
                or result.outcome.constraints is None):
            return result
        outcome = result.outcome
        if not any(token in str(outcome.status).lower()
                   for token in ('infeasible', 'solver_error')):
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
            context=f'{context} terminal execution failure'.strip(),
        )
        if slacks:
            ordered = sorted(slacks.items(), key=lambda item: -item[1])
            detail = 'box/group diagnostic slacks (not applied): ' + ', '.join(
                f'{name}={value * 10_000:.4f} bp' for name, value in ordered[:12]
            )
            if len(ordered) > 12:
                detail += f'; {len(ordered) - 12} additional bounds in solver log'
        else:
            detail = ('box/group diagnostic found no reported slack; '
                      'cause remains unresolved')
        return replace(result, outcome=replace(
            outcome, reason=f'{outcome.reason}; {detail}'.strip('; '),
        ))

    def run_linear_rescue(
            start: pd.DataFrame,
            relax_selected_corridors: bool,
            failed_result: Optional[ExecutionOptimizationResult],
    ) -> tuple[
        Optional[ExecutionOptimizationResult],
        Optional[ExecutionSolverInfeasibility],
        pd.DataFrame,
    ]:
        """Admit ranked feasible directions until constraints pass or capacity ends."""
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
            remaining = (
                rescue_eligible
                & ~candidate[SELECTED_TRADE].astype(bool)
            )
            if not active_bridges.empty:
                admissible, reasons = _get_constraint_directed_rescue_candidates(
                    trade_table=candidate,
                    base_constraints=base_constraints,
                    rescue_eligible=remaining,
                    bridges=bridges,
                    preserve_model_directions=preserve_model_directions,
                    relax_selected_corridors=relax_selected_corridors,
                )
            elif group_split_asset_classes is not None:
                if not _is_proven_infeasible(failed_result):
                    latest_error = ExecutionSolverInfeasibility(
                        'risk-budget-ranked rescue requires a proved infeasible '
                        'hard solve when no cash/group allocation bridge is present',
                        bridges=bridges,
                        trade_table=candidate,
                    )
                    return None, latest_error, candidate
                admissible, reasons = _get_group_tre_rescue_candidates(
                    trade_table=candidate,
                    base_constraints=base_constraints,
                    covariance=covariance,
                    rescue_eligible=remaining,
                )
                if not admissible.any():
                    latest_error = ExecutionSolverInfeasibility(
                        'risk-budget-ranked rescue found no eligible ranked '
                        'candidate loaded on a breached hard group-TRE constraint',
                        bridges=bridges,
                        trade_table=candidate,
                    )
                    return None, latest_error, candidate
            else:
                admissible = remaining
                reasons = pd.Series('', index=candidate.index, dtype=object)
                if sign_directed_rescue:
                    admissible = _get_sign_directed_rescue_candidates(
                        trade_table=candidate,
                        base_constraints=base_constraints,
                        rescue_eligible=remaining,
                    )
            candidate.loc[admissible, DIRECTED_RESCUE_CANDIDATE] = True
            if group_split_asset_classes is None:
                ordered = candidate.loc[admissible].sort_values(
                    [TRADE_SCORE, SCORE_PER_TURNOVER], ascending=[False, False]
                ).index
            else:
                ordered_frame = candidate.loc[admissible, [TRADE_RANK]].copy()
                ordered_frame['_identifier'] = ordered_frame.index.astype(str)
                ordered = ordered_frame.sort_values(
                    [TRADE_RANK, '_identifier'],
                    ascending=[True, True],
                    kind='mergesort',
                ).index
            if ordered.empty:
                if not active_bridges.empty:
                    latest_error = ExecutionSolverInfeasibility(
                        _format_bridge_failure(bridges),
                        bridges=bridges,
                        trade_table=candidate,
                    )
                return None, latest_error, candidate

            admitted = ordered[0]
            admission_count += 1
            candidate.loc[admitted, SELECTED_TRADE] = True
            candidate.loc[admitted, FEASIBILITY_RESCUE_TRADE] = True
            reason = reasons.loc[admitted]
            candidate.loc[admitted, RESCUE_REASON] = (
                reason if reason else 'solver_residual'
            )

            post_bridges = _compute_selected_execution_bridges(
                base_constraints=base_constraints,
                trade_table=candidate,
                relax_selected_corridors=relax_selected_corridors,
                partition_groups=partition_groups,
            )
            if not _positive_bridges(post_bridges).empty:
                continue
            stage = 'relaxed corridor' if relax_selected_corridors else 'strict corridor'
            try:
                result = solve_selected_execution_portfolio(
                    trade_table=candidate,
                    base_constraints=base_constraints,
                    covariance=covariance,
                    optimiser_config=optimiser_config,
                    context=(
                        f'{context} {stage} constraint-directed rescue '
                        f'{admission_count}'
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
                return result, None, candidate
            if group_split_asset_classes is not None:
                if not _is_proven_infeasible(result):
                    return result, None, candidate
                failed_result = result
            # A solver-only residual has no interval bridge. Continue one
            # candidate at a time in policy-score order and retry each time.

    strict_error = initial_error
    strict_audit = audited
    if expand_for_feasibility and (
            group_split_asset_classes is None
            or initial_error is not None
            or _is_proven_infeasible(initial_result)):
        strict_result, strict_error, strict_audit = run_linear_rescue(
            start=audited,
            relax_selected_corridors=False,
            failed_result=initial_result,
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
            context=f'{context} hard-feasibility corridor relaxation'.strip(),
            relax_selected_corridors=True,
            group_split_asset_classes=group_split_asset_classes,
            partition_groups=partition_groups,
        )
    except ExecutionSolverInfeasibility:
        relaxed_initial = None
    if (relaxed_initial is not None and relaxed_initial.accepted
            and relaxed_initial.compliant):
        return relaxed_initial
    if relaxed_initial is not None:
        latest_result = relaxed_initial
    relaxed_error: Optional[ExecutionSolverInfeasibility] = None
    relaxed_audit = audited
    if expand_for_feasibility:
        relaxed_result, relaxed_error, relaxed_audit = run_linear_rescue(
            start=audited,
            relax_selected_corridors=True,
            failed_result=relaxed_initial,
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
        trade_table: pd.DataFrame,
        base_constraints: Constraints,
        rescue_eligible: pd.Series,
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
        raise ValueError('directed rescue requires exactly one settlement-cash asset')
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
        base_constraints.max_weights.reindex(index).fillna(
            base_constraints.max_exposure
        ).astype(float)
        if base_constraints.max_weights is not None
        else pd.Series(base_constraints.max_exposure, index=index)
    )
    cash_min = max(
        float(product_min.loc[cash_asset]),
        float(trade_table.loc[cash_asset, POLICY_MIN_WEIGHT]),
    )
    cash_max = min(
        float(product_max.loc[cash_asset]),
        float(trade_table.loc[cash_asset, POLICY_MAX_WEIGHT]),
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
            if (group_max is not None and group in group_max.index
                    and exposure.loc[group]
                    > group_max.loc[group] + WEIGHT_ZERO_TOLERANCE):
                has_breach = True
                directed |= effect.lt(-WEIGHT_ZERO_TOLERANCE)
            if (group_min is not None and group in group_min.index
                    and exposure.loc[group]
                    < group_min.loc[group] - WEIGHT_ZERO_TOLERANCE):
                has_breach = True
                directed |= effect.gt(WEIGHT_ZERO_TOLERANCE)
    return eligible & directed if has_breach else eligible
