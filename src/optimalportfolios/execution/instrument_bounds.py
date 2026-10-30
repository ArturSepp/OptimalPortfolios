"""Shared execution result fields and original lifecycle instrument bounds."""
from dataclasses import dataclass
from typing import Optional
import numpy as np
import pandas as pd
from optimalportfolios.optimization import Constraints, OptimizationOutcome
from optimalportfolios.execution.schema import (
    BASE_WEIGHT,
    CUTOFF_SELL_DOWN,
    CURRENT_WEIGHT,
    EFFECTIVE_MODEL_WEIGHT,
    MANDATORY_CATEGORY,
    MANDATORY_TRADE,
    MATERIAL_TRADE_THRESHOLD,
    MODEL_ENTRY,
    POLICY_MAX_WEIGHT,
    POLICY_MIN_WEIGHT,
    PRODUCT_BOUND_OVERRIDE,
    RULE4_TRADE,
    SELECTED_TRADE,
    SETTLEMENT_CASH,
    TARGET_BAND_LOWER,
    TARGET_BAND_UPPER,
    WEIGHT_ZERO_TOLERANCE,
)


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


# These report fields remain explicit re-exports for operational MIQP consumers.
from optimalportfolios.execution.schema import (
    CUTOFF_SELL_DOWN_DUST_ROUNDED as CUTOFF_SELL_DOWN_DUST_ROUNDED,
    CUTOFF_SELL_DOWN_RETAINED_FRACTION as CUTOFF_SELL_DOWN_RETAINED_FRACTION,
    CUTOFF_SELL_DOWN_RETENTION as CUTOFF_SELL_DOWN_RETENTION,
)
