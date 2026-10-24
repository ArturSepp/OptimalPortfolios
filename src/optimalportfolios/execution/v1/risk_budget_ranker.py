"""Target-band ranking and asset-class TRE-budget diagnostics.

The functions in this module are numerical but solver-independent.  They
classify a live current/model cross-section, build alpha-strength target
bands, rank every active-to-active holding within its level-0 asset class and
select the smallest cash-funded prefix that attains each class TRE budget.
"""
from __future__ import annotations

from typing import Optional
from optimalportfolios.execution._common.model import is_model_zero

import numpy as np
import pandas as pd

from optimalportfolios.execution.v1.policy import (
    ABS_DISTANCE_TO_BAND,
    ALPHA,
    ALPHA_MISSING,
    ALPHA_STRENGTH,
    ASSET_CLASS,
    ASSET_CLASS_RANK,
    BASE_WEIGHT,
    COVAR_ACTIVE,
    CUTOFF_EXIT,
    CUTOFF_RESIDUAL_BRIDGE,
    CUTOFF_RESIDUAL_BRIDGE_BP,
    CUTOFF_RESIDUAL_BRIDGE_GROUP,
    CUTOFF_SELL_DOWN,
    CUTOFF_SELL_DOWN_DUST_ROUNDED,
    CUTOFF_SELL_DOWN_REASON,
    CUTOFF_SELL_DOWN_RETAINED_FRACTION,
    CUTOFF_SELL_DOWN_RETENTION,
    CURRENT_WEIGHT,
    DESIRED_REWEIGHT,
    DESK_FUNDING,
    DISTANCE_TO_BAND,
    EFFECTIVE_MODEL_WEIGHT,
    EFFECTIVE_SMALL_TRADE_CUTOFF,
    ENTRY_FUNDING,
    EXIT_FUNDING,
    FEASIBILITY_RESCUE_TRADE,
    FIXED_INSTRUCTION,
    GAP_BAND_SCORE,
    GROUP_K_FINAL,
    GROUP_K_RISK,
    GROUP_SELECTION_STATUS,
    GROUP_TRE_AFTER,
    GROUP_TRE_BEFORE,
    GROUP_TRE_BUDGET,
    GROUP_TRE_BUDGET_DIFFERENCE,
    GROUP_TRE_BUDGET_MET,
    GROUP_TRE_INITIAL,
    GROUP_TRE_PREFIX_CURVE,
    GROUP_TRE_REALIZED,
    GROUP_TRE_WEIGHT_NORMALIZED,
    GROUP_TRE_WEIGHT_RAW,
    INSTRUCTION_MAX_SELL,
    INSTRUCTION_MAX_WEIGHT,
    INSTRUCTION_MIN_BUY,
    INSTRUCTION_MIN_WEIGHT,
    INSTRUCTION_RETAINED_BREACH,
    INSTRUCTION_SOURCE,
    INSTRUCTION_WITHIN_TOLERANCE,
    LOW_WEIGHT_EXCLUDED,
    MANDATORY_CATEGORY,
    MANDATORY_FUNDING,
    MANDATORY_REASON,
    MANDATORY_TRADE,
    MARGINAL_ALPHA,
    MARGINAL_TRE,
    MATERIAL_TRADE,
    MATERIAL_TRADE_THRESHOLD,
    MODEL_ENTRY,
    MODEL_EXIT,
    MODEL_GAP,
    OTHER_FUNDING,
    PINNED_INSTRUCTION,
    POLICY_MAX_WEIGHT,
    POLICY_MIN_WEIGHT,
    POLICY_TARGET_WEIGHT,
    PRODUCT_BOUND_OVERRIDE,
    PRODUCT_BOUND_OVERRIDE_BREACH,
    PRODUCT_BOUND_OVERRIDE_REASON,
    PRODUCT_MAX_WEIGHT,
    PRODUCT_MIN_WEIGHT,
    RAW_MODEL_WEIGHT,
    REBALANCE_CADENCE_ELIGIBLE,
    REQUESTED_TRADE,
    RANKING_ELIGIBLE,
    RISK_BUDGET_SELECTED,
    RULE4_TRADE,
    SCORE_PER_TURNOVER,
    SELECTION_MARGINAL_TRE,
    SELECTION_SCORE,
    SELECTED_TRADE,
    SELL_DOWN_DUST_THRESHOLD,
    SEQUENTIAL_SELECTION_ORDER,
    SETTLEMENT_CASH,
    TARGET_BAND_LOWER,
    TARGET_BAND_UPPER,
    TARGET_BAND_WIDTH,
    TRADE_CANDIDATE,
    TRADE_RANK,
    TRADE_SCORE,
    TRADE_WEIGHT,
    TRE_WEIGHT,
    WEIGHT_ZERO_TOLERANCE,
    RiskBudgetRankedExecutionPolicyConfig,
    resolve_effective_max_trades,
)


_BAND_DIVISION_GUARD = 1e-12
RISK_BUDGET_GROUP_SUMMARY_ATTR = 'rosaa_risk_budget_group_summary'
RISK_BUDGET_PREFIX_CURVE_ATTR = 'rosaa_risk_budget_prefix_curve'
_GROUP_SUMMARY_COLUMNS = (
    'Raw TRE Weight', 'Normalized TRE Weight', 'TRE Budget', 'Overlap Count',
    'Eligible Count', 'Complete Rank Count', 'Initial Group TRE', 'Risk K',
    'Final K', 'Risk-Budget Selected', 'Feasibility Rescue',
    'Realized Group TRE', 'Budget Difference', 'Budget Met (Diagnostic)',
    'Status',
)
_PREFIX_CURVE_COLUMNS = (
    'Candidate Added', 'Group TRE', 'TRE Budget', 'Budget Met',
)


def _scalar_records(frame: pd.DataFrame) -> tuple[tuple[object, ...], ...]:
    """Return equality-safe scalar records for storage in pandas attrs."""
    records = []
    for row in frame.reset_index().itertuples(index=False, name=None):
        records.append(tuple(
            None if pd.isna(value)
            else value.item() if isinstance(value, np.generic)
            else value
            for value in row
        ))
    return tuple(records)


def _validate_covariance(covariance: pd.DataFrame, index: pd.Index) -> np.ndarray:
    """Return one finite symmetric covariance matrix in execution order."""
    missing = set(index).difference(covariance.index).union(
        set(index).difference(covariance.columns)
    )
    if missing:
        raise ValueError(
            f'execution covariance is missing instruments {sorted(missing)}'
        )
    sigma = covariance.loc[index, index].to_numpy(dtype=float)
    if not np.isfinite(sigma).all():
        raise ValueError('execution covariance must be finite')
    if not np.allclose(sigma, sigma.T, atol=1e-10, rtol=0.0):
        raise ValueError('execution covariance must be symmetric')
    return sigma


def _cash_funded_group_tre(
        weights: pd.Series,
        target: pd.Series,
        covariance: np.ndarray,
        group_mask: np.ndarray,
        cash_position: int,
) -> float:
    """Return annualized group residual TRE with its own settlement-cash leg."""
    active = np.zeros(len(weights), dtype=float)
    values = weights.to_numpy(dtype=float) - target.to_numpy(dtype=float)
    active[group_mask] = values[group_mask]
    active[cash_position] = -float(active.sum())
    variance = float(active @ covariance @ active)
    if variance < -1e-10:
        raise ValueError(f'cash-funded group active variance is negative: {variance}')
    return float(np.sqrt(max(variance, 0.0)))


def _classify_adjustment_categories(
        scaffold: pd.DataFrame,
        lower_adjustment: pd.Series,
        upper_adjustment: pd.Series,
) -> pd.Series:
    """Label exact hard-bound repairs by their controlling bound source."""
    index = scaffold.index
    category = pd.Series('', index=index, dtype=object)
    instruction_source = scaffold[INSTRUCTION_SOURCE].fillna('').astype(str)
    fixed = scaffold[FIXED_INSTRUCTION].astype(bool)
    instruction_min = scaffold[INSTRUCTION_MIN_WEIGHT].astype(float)
    instruction_max = scaffold[INSTRUCTION_MAX_WEIGHT].astype(float)
    product_min = scaffold[PRODUCT_MIN_WEIGHT].astype(float)
    product_max = scaffold[PRODUCT_MAX_WEIGHT].astype(float)
    instruction_lower = (
        instruction_source.ne('')
        & instruction_min.ge(product_min - WEIGHT_ZERO_TOLERANCE)
    )
    instruction_upper = (
        instruction_source.ne('')
        & instruction_max.le(product_max + WEIGHT_ZERO_TOLERANCE)
    )
    category.loc[lower_adjustment & ~instruction_lower] = 'product_min_buy'
    category.loc[upper_adjustment & ~instruction_upper] = 'product_max_sell'
    category.loc[lower_adjustment & instruction_lower] = 'instruction_min_buy'
    category.loc[upper_adjustment & instruction_upper] = 'instruction_max_sell'
    category.loc[(lower_adjustment | upper_adjustment) & fixed] = 'fixed_instruction'
    return category


def build_risk_budget_ranked_trade_table(
        scaffold: pd.DataFrame,
        alphas: pd.Series,
        covariance: pd.DataFrame,
        asset_classes: pd.Series,
        asset_class_tre_weights: Optional[pd.Series],
        config: RiskBudgetRankedExecutionPolicyConfig,
) -> pd.DataFrame:
    """Build the complete target-band ranks and risk-selected class prefixes.

    ``scaffold`` is produced by :func:`build_execution_target` solely to reuse
    its audited product/instruction interval construction and provenance.  All
    lifecycle classification, base weights, ranking and selection are rebuilt
    here from the live current and model weights; previous model weights play
    no role.
    """
    required = {
        CURRENT_WEIGHT, RAW_MODEL_WEIGHT, EFFECTIVE_SMALL_TRADE_CUTOFF,
        PRODUCT_MIN_WEIGHT, PRODUCT_MAX_WEIGHT, INSTRUCTION_MIN_WEIGHT,
        INSTRUCTION_MAX_WEIGHT, INSTRUCTION_SOURCE, FIXED_INSTRUCTION,
        SETTLEMENT_CASH, REBALANCE_CADENCE_ELIGIBLE,
    }
    missing = required.difference(scaffold.columns)
    if missing:
        raise ValueError(
            f'risk-budget execution scaffold is missing columns {sorted(missing)}'
        )
    out = scaffold.copy()
    index = out.index
    if not index.is_unique:
        raise ValueError('risk-budget execution index must be unique')
    sigma = _validate_covariance(covariance=covariance, index=index)

    groups = asset_classes.reindex(index)
    if groups.isna().any():
        raise ValueError(
            f'missing asset classes for {groups.index[groups.isna()].tolist()}'
        )
    unsupported = set(groups).difference(tuple(config.tre_weight_by_asset_class))
    if unsupported:
        raise ValueError(f'unsupported level-0 asset classes {sorted(unsupported)}')
    out[ASSET_CLASS] = groups.astype(str)

    cash = out[SETTLEMENT_CASH].astype(bool)
    if int(cash.sum()) != 1:
        raise ValueError(
            'risk-budget-ranked execution requires exactly one settlement-cash asset'
        )
    cash_asset = index[cash][0]
    cash_position = int(index.get_loc(cash_asset))
    noncash = ~cash

    current = out[CURRENT_WEIGHT].astype(float)
    model = out[RAW_MODEL_WEIGHT].astype(float)
    cutoff = out[EFFECTIVE_SMALL_TRADE_CUTOFF].astype(float)
    allowed = out[REBALANCE_CADENCE_ELIGIBLE].astype(bool)
    hard_min = pd.concat([
        out[PRODUCT_MIN_WEIGHT].astype(float),
        out[INSTRUCTION_MIN_WEIGHT].astype(float),
    ], axis=1, sort=False).max(axis=1)
    hard_max = pd.concat([
        out[PRODUCT_MAX_WEIGHT].astype(float),
        out[INSTRUCTION_MAX_WEIGHT].astype(float),
    ], axis=1, sort=False).min(axis=1)
    numeric = pd.concat([current, model, cutoff, hard_min, hard_max], axis=1, sort=False)
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError('risk-budget execution weights and bounds must be finite')
    invalid_hard = hard_min > hard_max + WEIGHT_ZERO_TOLERANCE
    if invalid_hard.any():
        details = pd.DataFrame({'min': hard_min, 'max': hard_max}).loc[invalid_hard]
        raise ValueError(
            'execution product and instruction bounds do not intersect:\n'
            f'{details.to_string()}'
        )

    model_for_rules = model.mask(is_model_zero(model), 0.0)
    target = model_for_rules.clip(lower=hard_min, upper=hard_max)
    raw_alpha = pd.to_numeric(alphas.reindex(index), errors='coerce')
    stored_missing = alphas.attrs.get('rosaa_alpha_missing')
    if isinstance(stored_missing, frozenset):
        alpha_missing = pd.Series(
            ((pd.Timestamp(alphas.name), asset) in stored_missing for asset in index),
            index=index,
        )
    else:
        alpha_missing = raw_alpha.isna()
    alpha_missing = alpha_missing.astype(bool)
    alpha = raw_alpha.fillna(0.0).astype(float)
    if np.isinf(alpha.to_numpy()).any():
        raise ValueError('execution alpha values must be finite or missing')
    invalid_alpha = alpha.abs() > 1.0 + WEIGHT_ZERO_TOLERANCE
    if invalid_alpha.any():
        raise ValueError(
            'risk-budget alpha strength requires alpha in [-1, 1]: '
            f'{invalid_alpha.index[invalid_alpha].tolist()}'
        )
    alpha_strength = alpha.abs().clip(upper=1.0)
    tight = groups.map(config.target_band_tight_by_asset_class).astype(float)
    wide = groups.map(config.target_band_wide_by_asset_class).astype(float)
    band_width = wide - alpha_strength * (wide - tight)
    band_lower = pd.concat([hard_min, target - band_width], axis=1, sort=False).max(axis=1)
    band_upper = pd.concat([hard_max, target + band_width], axis=1, sort=False).min(axis=1)

    lower_adjustment = (
        current.lt(hard_min - WEIGHT_ZERO_TOLERANCE) & noncash
    )
    upper_adjustment = (
        current.gt(hard_max + WEIGHT_ZERO_TOLERANCE) & noncash
    )
    hard_adjustment = lower_adjustment | upper_adjustment
    current_present = current.gt(cutoff)
    target_present = target.gt(cutoff)
    positive_hard_min = hard_min.gt(0.0)
    model_entry = (
        ~hard_adjustment & ~current_present & target_present & noncash
    )
    model_exit = (
        ~hard_adjustment & current_present & ~target_present
        & ~positive_hard_min & noncash
    )
    overlap = (
        ~hard_adjustment & ~model_entry & ~model_exit
        & current_present & target_present & noncash
    )
    band_lower.loc[model_entry] = pd.concat([
        band_lower.loc[model_entry], cutoff.loc[model_entry]
    ], axis=1, sort=False).max(axis=1)
    invalid_band = band_lower > band_upper + WEIGHT_ZERO_TOLERANCE
    if invalid_band.any():
        details = pd.DataFrame({
            'target': target,
            'cutoff': cutoff,
            'hard_min': hard_min,
            'hard_max': hard_max,
            'band_lower': band_lower,
            'band_upper': band_upper,
        }).loc[invalid_band]
        raise ValueError(f'execution target band is empty:\n{details.to_string()}')

    base = current.copy()
    base.loc[lower_adjustment] = hard_min.loc[lower_adjustment]
    base.loc[upper_adjustment] = hard_max.loc[upper_adjustment]
    base.loc[model_entry] = target.loc[model_entry]
    base.loc[model_exit] = 0.0
    noncash_funding = (base - current).where(noncash, 0.0)
    base.loc[cash_asset] = current.loc[cash_asset] - float(noncash_funding.sum())
    if not np.isclose(
            base.sum(), current.sum(), atol=WEIGHT_ZERO_TOLERANCE, rtol=0.0):
        raise RuntimeError('risk-budget post-compulsory base is not self-financing')

    funded_target = target.copy()
    funded_target.loc[cash_asset] = current.sum() - float(target.loc[noncash].sum())
    projection = base.clip(lower=band_lower, upper=band_upper)
    distance = (projection - base).where(overlap, 0.0)
    model_gap = target - base
    score = model_gap.abs().div(band_width.clip(lower=_BAND_DIVISION_GUARD))
    outside_band = distance.abs().gt(WEIGHT_ZERO_TOLERANCE)
    material = model_gap.abs().ge(config.material_trade_threshold)
    eligible = overlap & allowed & outside_band & material

    rank = pd.Series(pd.NA, index=index, dtype='Int64')
    identifier = pd.Series(index.astype(str), index=index)
    for asset_class in tuple(config.tre_weight_by_asset_class):
        members = index[overlap & groups.eq(asset_class)]
        if members.empty:
            continue
        ordering = pd.DataFrame({
            'score': score.loc[members],
            'distance': distance.loc[members].abs(),
            'gap': model_gap.loc[members].abs(),
            'identifier': identifier.loc[members],
        }, index=members).sort_values(
            ['score', 'distance', 'gap', 'identifier'],
            ascending=[False, False, False, True],
            kind='mergesort',
        )
        rank.loc[ordering.index] = pd.array(
            np.arange(1, len(ordering) + 1), dtype='Int64'
        )

    if asset_class_tre_weights is None:
        raise ValueError(
            "risk-budget-ranked execution requires 'TRE weight' in saa_asset_class"
        )
    raw_group_weight = pd.to_numeric(
        asset_class_tre_weights.reindex(tuple(config.tre_weight_by_asset_class)), errors='coerce'
    )
    invalid_group_weight = (
        raw_group_weight.isna()
        | ~np.isfinite(raw_group_weight)
        | raw_group_weight.lt(0.0)
    )
    if invalid_group_weight.any() or float(raw_group_weight.sum()) <= 0.0:
        raise ValueError(
            'saa_asset_class TRE weights must be finite, non-negative and sum positive; '
            f'invalid/missing={raw_group_weight.index[invalid_group_weight].tolist()}'
        )
    normalized_group_weight = raw_group_weight / float(raw_group_weight.sum())

    mandatory = hard_adjustment | model_entry | model_exit
    risk_selected = pd.Series(False, index=index)
    group_tre_before = pd.Series(np.nan, index=index)
    group_tre_after = pd.Series(np.nan, index=index)
    group_values = {
        GROUP_TRE_WEIGHT_RAW: pd.Series(np.nan, index=index),
        GROUP_TRE_WEIGHT_NORMALIZED: pd.Series(np.nan, index=index),
        GROUP_TRE_BUDGET: pd.Series(np.nan, index=index),
        GROUP_TRE_INITIAL: pd.Series(np.nan, index=index),
        GROUP_TRE_PREFIX_CURVE: pd.Series('', index=index, dtype=object),
        GROUP_K_RISK: pd.Series(0, index=index, dtype='Int64'),
        GROUP_K_FINAL: pd.Series(0, index=index, dtype='Int64'),
        GROUP_TRE_REALIZED: pd.Series(np.nan, index=index),
        GROUP_TRE_BUDGET_DIFFERENCE: pd.Series(np.nan, index=index),
        GROUP_TRE_BUDGET_MET: pd.Series(False, index=index),
        GROUP_SELECTION_STATUS: pd.Series('', index=index, dtype=object),
    }
    group_summary_rows = []
    prefix_rows = []
    for asset_class in tuple(config.tre_weight_by_asset_class):
        group_rows = groups.eq(asset_class)
        budget = (
            config.execution_tre_budget
            * float(normalized_group_weight.loc[asset_class])
        )
        group_mask = (
            group_rows.to_numpy(dtype=bool) & noncash.to_numpy(dtype=bool)
        )
        candidates = index[eligible & group_rows].tolist()
        candidates.sort(key=lambda asset: (int(rank.loc[asset]), str(asset)))
        curve = []
        for prefix_size in range(len(candidates) + 1):
            hypothetical = base.copy()
            selected_prefix = candidates[:prefix_size]
            if selected_prefix:
                hypothetical.loc[selected_prefix] = target.loc[selected_prefix]
            hypothetical.loc[cash_asset] = (
                current.sum() - float(hypothetical.loc[noncash].sum())
            )
            curve.append(_cash_funded_group_tre(
                weights=hypothetical,
                target=funded_target,
                covariance=sigma,
                group_mask=group_mask,
                cash_position=cash_position,
            ))
        feasible_prefixes = [
            k for k, value in enumerate(curve)
            if value <= budget + WEIGHT_ZERO_TOLERANCE
        ]
        if curve[0] <= budget + WEIGHT_ZERO_TOLERANCE:
            k_risk = 0
            status = 'already inside'
        elif feasible_prefixes:
            k_risk = min(feasible_prefixes)
            status = 'attained'
        else:
            k_risk = len(candidates)
            status = 'unattainable'
        risk_selected.loc[candidates[:k_risk]] = True
        for position, asset in enumerate(candidates, start=1):
            group_tre_before.loc[asset] = curve[position - 1]
            group_tre_after.loc[asset] = curve[position]
        curve_text = ';'.join(
            f'{position}={value:.10g}' for position, value in enumerate(curve)
        )
        prefix_rows.extend({
            'Asset Class': asset_class,
            'Prefix K': position,
            'Candidate Added': (
                '' if position == 0 else str(candidates[position - 1])
            ),
            'Group TRE': value,
            'TRE Budget': budget,
            'Budget Met': value <= budget + WEIGHT_ZERO_TOLERANCE,
        } for position, value in enumerate(curve))
        group_summary_rows.append({
            'Asset Class': asset_class,
            'Raw TRE Weight': float(raw_group_weight.loc[asset_class]),
            'Normalized TRE Weight': float(
                normalized_group_weight.loc[asset_class]
            ),
            'TRE Budget': budget,
            'Overlap Count': int((overlap & group_rows).sum()),
            'Eligible Count': len(candidates),
            'Complete Rank Count': int(rank.loc[group_rows].notna().sum()),
            'Initial Group TRE': curve[0],
            'Risk K': k_risk,
            'Final K': k_risk,
            'Risk-Budget Selected': k_risk,
            'Feasibility Rescue': 0,
            'Realized Group TRE': np.nan,
            'Budget Difference': np.nan,
            'Budget Met (Diagnostic)': False,
            'Status': status,
        })
        assignments = {
            GROUP_TRE_WEIGHT_RAW: float(raw_group_weight.loc[asset_class]),
            GROUP_TRE_WEIGHT_NORMALIZED: float(
                normalized_group_weight.loc[asset_class]
            ),
            GROUP_TRE_BUDGET: budget,
            GROUP_TRE_INITIAL: curve[0],
            GROUP_TRE_PREFIX_CURVE: curve_text,
            GROUP_K_RISK: k_risk,
            GROUP_K_FINAL: k_risk,
            GROUP_SELECTION_STATUS: status,
        }
        for column, value in assignments.items():
            group_values[column].loc[group_rows] = value

    effective_max_trades = resolve_effective_max_trades(
        config=config,
        mandatory_ticket_count=int(mandatory.sum()),
    )
    if effective_max_trades is not None:
        discretionary_capacity = max(
            effective_max_trades - int(mandatory.sum()), 0
        )
        selected = index[risk_selected]
        if len(selected) > discretionary_capacity:
            ordering = pd.DataFrame({
                'score': score.loc[selected],
                'distance': distance.loc[selected].abs(),
                'gap': model_gap.loc[selected].abs(),
                'identifier': identifier.loc[selected],
            }, index=selected).sort_values(
                ['score', 'distance', 'gap', 'identifier'],
                ascending=[False, False, False, True],
                kind='mergesort',
            )
            retained = ordering.index[:discretionary_capacity]
            risk_selected.loc[:] = False
            risk_selected.loc[retained] = True
            for summary in group_summary_rows:
                asset_class = summary['Asset Class']
                group_rows = groups.eq(asset_class)
                selected_count = int((risk_selected & group_rows).sum())
                if selected_count < int(summary['Risk K']):
                    status = f"{summary['Status']}; ticket cap"
                    summary['Status'] = status
                    group_values[GROUP_SELECTION_STATUS].loc[group_rows] = status
                summary['Final K'] = selected_count
                summary['Risk-Budget Selected'] = selected_count
                group_values[GROUP_K_FINAL].loc[group_rows] = selected_count

    categories = _classify_adjustment_categories(
        scaffold=out,
        lower_adjustment=lower_adjustment,
        upper_adjustment=upper_adjustment,
    )
    categories.loc[model_entry] = 'model_entry'
    categories.loc[model_exit] = 'model_exit'
    reasons = categories.copy()
    reasons.loc[cash_asset] = (
        'mandatory_cash_funding'
        if abs(float(noncash_funding.sum())) > WEIGHT_ZERO_TOLERANCE else ''
    )
    out[POLICY_TARGET_WEIGHT] = target
    out[EFFECTIVE_MODEL_WEIGHT] = funded_target
    out[BASE_WEIGHT] = base
    out[TRADE_WEIGHT] = target - current
    out[LOW_WEIGHT_EXCLUDED] = False
    out[CUTOFF_SELL_DOWN] = False
    out[CUTOFF_SELL_DOWN_REASON] = ''
    out[CUTOFF_SELL_DOWN_RETENTION] = np.nan
    out[CUTOFF_SELL_DOWN_RETAINED_FRACTION] = np.nan
    out[CUTOFF_SELL_DOWN_DUST_ROUNDED] = False
    out[SELL_DOWN_DUST_THRESHOLD] = 0.0
    out[CUTOFF_RESIDUAL_BRIDGE] = 0.0
    out[CUTOFF_RESIDUAL_BRIDGE_BP] = 0.0
    out[CUTOFF_RESIDUAL_BRIDGE_GROUP] = ''
    out[MODEL_ENTRY] = model_entry
    out[MODEL_EXIT] = model_exit
    out[CUTOFF_EXIT] = model_exit
    out[INSTRUCTION_MIN_BUY] = lower_adjustment & categories.eq(
        'instruction_min_buy'
    )
    out[INSTRUCTION_MAX_SELL] = upper_adjustment & categories.eq(
        'instruction_max_sell'
    )
    out[INSTRUCTION_WITHIN_TOLERANCE] = False
    out[INSTRUCTION_RETAINED_BREACH] = 0.0
    out[PINNED_INSTRUCTION] = (
        out[FIXED_INSTRUCTION].astype(bool) & ~hard_adjustment
    )
    out[MANDATORY_REASON] = reasons
    out[MANDATORY_CATEGORY] = categories
    out[MANDATORY_FUNDING] = noncash_funding
    out.loc[cash_asset, MANDATORY_FUNDING] = -float(noncash_funding.sum())
    out[ENTRY_FUNDING] = noncash_funding.where(model_entry, 0.0)
    out[EXIT_FUNDING] = noncash_funding.where(model_exit, 0.0)
    desk_categories = categories.isin([
        'instruction_min_buy', 'instruction_max_sell', 'fixed_instruction'
    ])
    out[DESK_FUNDING] = noncash_funding.where(desk_categories, 0.0)
    out[OTHER_FUNDING] = (
        noncash_funding - out[ENTRY_FUNDING] - out[EXIT_FUNDING]
        - out[DESK_FUNDING]
    )
    out[RULE4_TRADE] = overlap
    out[DESIRED_REWEIGHT] = distance
    out[MANDATORY_TRADE] = mandatory
    out[MATERIAL_TRADE] = material
    out[MATERIAL_TRADE_THRESHOLD] = config.material_trade_threshold
    out[TRADE_CANDIDATE] = eligible
    out[POLICY_MIN_WEIGHT] = hard_min
    out[POLICY_MAX_WEIGHT] = hard_max
    out[PRODUCT_BOUND_OVERRIDE] = False
    out[PRODUCT_BOUND_OVERRIDE_BREACH] = 0.0
    out[PRODUCT_BOUND_OVERRIDE_REASON] = ''

    out[ALPHA] = alpha
    out[ALPHA_MISSING] = alpha_missing
    out[ALPHA_STRENGTH] = alpha_strength
    out[TARGET_BAND_WIDTH] = band_width
    out[TARGET_BAND_LOWER] = band_lower
    out[TARGET_BAND_UPPER] = band_upper
    out[MODEL_GAP] = model_gap
    out[DISTANCE_TO_BAND] = distance
    out[ABS_DISTANCE_TO_BAND] = distance.abs()
    out[GAP_BAND_SCORE] = score
    out[ASSET_CLASS_RANK] = rank
    out[RANKING_ELIGIBLE] = eligible
    out[RISK_BUDGET_SELECTED] = risk_selected
    out[GROUP_TRE_BEFORE] = group_tre_before
    out[GROUP_TRE_AFTER] = group_tre_after
    for column, values in group_values.items():
        out[column] = values

    active = base - model
    out[COVAR_ACTIVE] = sigma @ active.to_numpy(dtype=float)
    out[MARGINAL_ALPHA] = np.nan
    out[MARGINAL_TRE] = np.nan
    out[TRE_WEIGHT] = groups.map(normalized_group_weight).astype(float)
    out[TRADE_SCORE] = score
    out[SCORE_PER_TURNOVER] = np.nan
    out[TRADE_RANK] = rank
    out[SELECTION_SCORE] = score.where(risk_selected)
    out[SELECTION_MARGINAL_TRE] = (
        group_tre_after - group_tre_before
    ).where(risk_selected)
    out[SEQUENTIAL_SELECTION_ORDER] = rank.where(risk_selected)
    out[SELECTED_TRADE] = risk_selected
    out[REQUESTED_TRADE] = risk_selected
    out[FEASIBILITY_RESCUE_TRADE] = False
    group_summary = pd.DataFrame.from_records(group_summary_rows).set_index(
        'Asset Class'
    )
    prefix_curve = pd.DataFrame.from_records(prefix_rows).set_index(
        ['Asset Class', 'Prefix K']
    )
    out.attrs["execution_asset_classes"] = tuple(config.tre_weight_by_asset_class)
    out.attrs[RISK_BUDGET_GROUP_SUMMARY_ATTR] = _scalar_records(group_summary)
    out.attrs[RISK_BUDGET_PREFIX_CURVE_ATTR] = _scalar_records(prefix_curve)
    return out


def finalize_risk_budget_diagnostics(
        table: pd.DataFrame,
        proposed_weights: pd.Series,
        covariance: pd.DataFrame,
) -> pd.DataFrame:
    """Add realized class TRE, final ticket counts and rescue statuses."""
    out = table.copy()
    index = out.index
    sigma = _validate_covariance(covariance=covariance, index=index)
    cash = out[SETTLEMENT_CASH].astype(bool)
    if int(cash.sum()) != 1:
        raise ValueError('realized group TRE requires one settlement-cash asset')
    cash_position = int(np.flatnonzero(cash.to_numpy())[0])
    target = out[EFFECTIVE_MODEL_WEIGHT].astype(float)
    proposed = proposed_weights.reindex(index).astype(float)
    groups = out[ASSET_CLASS].astype(str)
    rescue = out[FEASIBILITY_RESCUE_TRADE].astype(bool)
    selected = out[SELECTED_TRADE].astype(bool)
    eligible = out[RANKING_ELIGIBLE].astype(bool)
    try:
        summary = build_risk_budget_group_summary(out)
    except ValueError:
        summary = None
    for asset_class in _declared_classes(table):
        group_rows = groups.eq(asset_class)
        group_mask = group_rows.to_numpy(dtype=bool) & ~cash.to_numpy(dtype=bool)
        realized = _cash_funded_group_tre(
            weights=proposed,
            target=target,
            covariance=sigma,
            group_mask=group_mask,
            cash_position=cash_position,
        )
        if group_rows.any():
            budget = float(out.loc[group_rows, GROUP_TRE_BUDGET].iloc[0])
        elif summary is not None and asset_class in summary.index:
            budget = float(summary.loc[asset_class, 'TRE Budget'])
        else:
            continue
        k_final = int((group_rows & eligible & selected).sum())
        budget_met = realized <= budget + WEIGHT_ZERO_TOLERANCE
        if group_rows.any():
            out.loc[group_rows, GROUP_K_FINAL] = k_final
            out.loc[group_rows, GROUP_TRE_REALIZED] = realized
            out.loc[group_rows, GROUP_TRE_BUDGET_DIFFERENCE] = realized - budget
            out.loc[group_rows, GROUP_TRE_BUDGET_MET] = budget_met
        rescued = int((group_rows & rescue).sum())
        if rescued:
            out.loc[group_rows, GROUP_SELECTION_STATUS] = (
                'hard-feasibility expansion'
            )
        if summary is not None and asset_class in summary.index:
            summary.loc[asset_class, 'Final K'] = k_final
            summary.loc[asset_class, 'Risk-Budget Selected'] = int((
                group_rows & out[RISK_BUDGET_SELECTED].astype(bool)
            ).sum())
            summary.loc[asset_class, 'Feasibility Rescue'] = rescued
            summary.loc[asset_class, 'Realized Group TRE'] = realized
            summary.loc[asset_class, 'Budget Difference'] = realized - budget
            summary.loc[asset_class, 'Budget Met (Diagnostic)'] = budget_met
            if rescued:
                summary.loc[asset_class, 'Status'] = (
                    'hard-feasibility expansion'
                )
    if summary is not None:
        out.attrs[RISK_BUDGET_GROUP_SUMMARY_ATTR] = _scalar_records(summary)
    return out


def build_risk_budget_group_summary(table: pd.DataFrame) -> pd.DataFrame:
    """Return one stored-diagnostic row for every level-0 asset class."""
    stored = table.attrs.get(RISK_BUDGET_GROUP_SUMMARY_ATTR)
    if isinstance(stored, tuple):
        stored = pd.DataFrame.from_records(
            stored,
            columns=('Asset Class',) + _GROUP_SUMMARY_COLUMNS,
        ).set_index('Asset Class')
        missing_groups = set(_declared_classes(table)).difference(stored.index)
        if missing_groups:
            raise ValueError(
                'risk-budget group summary is missing asset classes '
                f'{sorted(missing_groups)}'
            )
        return stored.reindex(_declared_classes(table)).copy()
    required = {
        ASSET_CLASS, RULE4_TRADE, RANKING_ELIGIBLE, TRADE_RANK,
        RISK_BUDGET_SELECTED, FEASIBILITY_RESCUE_TRADE, SELECTED_TRADE,
        GROUP_TRE_WEIGHT_RAW, GROUP_TRE_WEIGHT_NORMALIZED, GROUP_TRE_BUDGET,
        GROUP_TRE_INITIAL, GROUP_TRE_PREFIX_CURVE, GROUP_K_RISK,
        GROUP_K_FINAL, GROUP_TRE_REALIZED, GROUP_TRE_BUDGET_DIFFERENCE,
        GROUP_TRE_BUDGET_MET, GROUP_SELECTION_STATUS,
    }
    missing = required.difference(table.columns)
    if missing:
        raise ValueError(
            f'risk-budget group summary is missing columns {sorted(missing)}'
        )
    rows = []
    groups = table[ASSET_CLASS].astype(str)
    for asset_class in _declared_classes(table):
        mask = groups.eq(asset_class)
        if not mask.any():
            continue
        first = table.loc[mask].iloc[0]
        rows.append({
            'Asset Class': asset_class,
            'Raw TRE Weight': first[GROUP_TRE_WEIGHT_RAW],
            'Normalized TRE Weight': first[GROUP_TRE_WEIGHT_NORMALIZED],
            'TRE Budget': first[GROUP_TRE_BUDGET],
            'Overlap Count': int(table.loc[mask, RULE4_TRADE].astype(bool).sum()),
            'Eligible Count': int(
                table.loc[mask, RANKING_ELIGIBLE].astype(bool).sum()
            ),
            'Complete Rank Count': int(table.loc[mask, TRADE_RANK].notna().sum()),
            'Initial Group TRE': first[GROUP_TRE_INITIAL],
            'Prefix Curve': first[GROUP_TRE_PREFIX_CURVE],
            'Risk K': int(first[GROUP_K_RISK]),
            'Final K': int(first[GROUP_K_FINAL]),
            'Risk-Budget Selected': int(
                table.loc[mask, RISK_BUDGET_SELECTED].astype(bool).sum()
            ),
            'Feasibility Rescue': int(
                table.loc[mask, FEASIBILITY_RESCUE_TRADE].astype(bool).sum()
            ),
            'Realized Group TRE': first[GROUP_TRE_REALIZED],
            'Budget Difference': first[GROUP_TRE_BUDGET_DIFFERENCE],
            'Budget Met (Diagnostic)': bool(first[GROUP_TRE_BUDGET_MET]),
            'Status': first[GROUP_SELECTION_STATUS],
        })
    return pd.DataFrame.from_records(rows).set_index('Asset Class')


def build_risk_budget_prefix_curve(table: pd.DataFrame) -> pd.DataFrame:
    """Return the stored long-form, all-prefix group TRE diagnostic."""
    stored = table.attrs.get(RISK_BUDGET_PREFIX_CURVE_ATTR)
    if not isinstance(stored, tuple):
        raise ValueError('risk-budget prefix curve is unavailable')
    stored = pd.DataFrame.from_records(
        stored,
        columns=('Asset Class', 'Prefix K') + _PREFIX_CURVE_COLUMNS,
    ).set_index(['Asset Class', 'Prefix K'])
    missing_groups = set(_declared_classes(table)).difference(
        stored.index.get_level_values('Asset Class')
    )
    if missing_groups:
        raise ValueError(
            'risk-budget prefix curve is missing asset classes '
            f'{sorted(missing_groups)}'
        )
    return stored.copy()


def _declared_classes(table: pd.DataFrame) -> tuple[str, ...]:
    """Recover declared class order from current or historical diagnostics."""
    classes = table.attrs.get('execution_asset_classes')
    if classes is not None:
        return tuple(classes)
    stored = table.attrs.get(RISK_BUDGET_GROUP_SUMMARY_ATTR)
    if isinstance(stored, tuple):
        return tuple(row[0] for row in stored)
    return tuple(dict.fromkeys(table[ASSET_CLASS]))
