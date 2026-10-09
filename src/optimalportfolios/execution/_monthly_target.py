"""Monthly model lifecycle for research portfolios without desk instructions.

This private adapter migrates the MAC model-entry, cutoff-retention and cash
funding rules into OP. It deliberately accepts resolved numeric inputs only.
Live instructions and mixed instrument calendars remain consumer concerns.
"""
import numpy as np
import pandas as pd

from optimalportfolios.execution import schema as s
from optimalportfolios.execution.feasibility import compute_group_bound_bridges


def _build_monthly_target(current, model, previous_model, constraints, cutoffs,
                          cash_asset, group_enabled=True, material_threshold=0.0025,
                          dust_threshold=0.0025):
    """Resolve one monthly, fully eligible, instruction-free model cross-section.

    Availability is encoded by dated product caps; held unavailable assets are
    therefore forced out. Fractional group loadings are retained unchanged.
    Cutoff retention is attempted with group inputs, then without by the caller.
    No weight drift, ranking, optimisation or P&L is implemented here.
    """
    idx = current.index
    if not idx.is_unique or cash_asset not in idx:
        raise ValueError('Unique instruments and a settlement cash asset are required')
    vectors = [x.reindex(idx).astype(float) for x in (current, model, previous_model, cutoffs)]
    if not all(np.isfinite(x).all() for x in vectors):
        raise ValueError('Monthly lifecycle inputs must be finite and complete')
    current, model, previous_model, cutoffs = vectors
    if (cutoffs < 0).any() or material_threshold < 0 or dust_threshold < 0:
        raise ValueError('Cutoff and trade thresholds must be nonnegative')
    tol = s.WEIGHT_ZERO_TOLERANCE
    raw = model.mask(model.abs().le(tol), 0.0)
    prior = previous_model.mask(previous_model.abs().le(tol), 0.0)
    held = current.mask(current.abs().le(tol), 0.0)
    cash = pd.Series(idx == cash_asset, index=idx)
    lo = (pd.Series(0., index=idx) if constraints.min_weights is None else
          constraints.min_weights.reindex(idx).fillna(0.).astype(float))
    hi = (pd.Series(1., index=idx) if constraints.max_weights is None else
          constraints.max_weights.reindex(idx).fillna(1.).astype(float))
    if not np.isfinite(lo).all() or not np.isfinite(hi).all() or (lo > hi).any():
        raise ValueError('Invalid product bounds')
    # The historical no-instruction policy intersects product bounds with [0, 1].
    policy_lo, policy_hi = lo.clip(lower=0.), hi.clip(upper=1.)
    if (policy_lo > policy_hi).any():
        raise ValueError('Monthly lifecycle requires long-only product bounds')
    low = raw.le(cutoffs) & ~cash
    policy_target = raw.mask(low, 0.)
    effective = policy_target.clip(lower=policy_lo, upper=policy_hi)
    buy = current.lt(lo - tol)
    sell = current.gt(hi + tol)
    entry = raw.gt(cutoffs) & prior.le(cutoffs) & held.eq(0.) & ~cash
    exit_ = prior.gt(cutoffs) & raw.le(cutoffs) & held.gt(0.) & ~cash
    cutoff_candidate = low & held.gt(0.) & raw.gt(tol)
    retention = pd.Series(False, index=idx)
    reasons = {i: set() for i in idx}
    bridges = pd.Series(0., index=idx)
    bridge_groups = {i: [] for i in idx}
    group = constraints.group_lower_upper_constraints if group_enabled else None
    if group is not None:
        loading = group.group_loadings.reindex(index=idx).fillna(0.).astype(float)
        minimum = (pd.Series(np.nan, index=loading.columns) if
                   group.group_min_allocation is None else
                   group.group_min_allocation.reindex(loading.columns).astype(float))
        maximum = (pd.Series(np.nan, index=loading.columns) if
                   group.group_max_allocation is None else
                   group.group_max_allocation.reindex(loading.columns).astype(float))
        provisional = current.copy()
        for mask, value in ((buy, lo), (sell, hi),
                            (low & policy_lo.le(tol), pd.Series(0., index=idx)),
                            (exit_ & policy_lo.le(tol), pd.Series(0., index=idx)),
                            (entry, effective)):
            active = mask & ~cash
            provisional.loc[active] = value.loc[active]
        mandatory = provisional.sub(current).abs().gt(tol) & ~cash
        candidates = cutoff_candidate & policy_lo.le(tol) & ~sell

        def diagnose(retain):
            """Measure group capacity before and after permitting cutoff retention."""
            upper = current.copy()
            upper.loc[mandatory] = provisional.loc[mandatory]
            excluded = low & policy_lo.le(tol)
            if retain:
                excluded &= ~retention
            upper.loc[excluded] = 0.
            return compute_group_bound_bridges(pd.Series(0., index=idx), upper,
                                               loading, minimum, maximum)

        first = diagnose(False)
        for name in first.index[first['bridge_min'].gt(tol)]:
            affected = candidates & (loading[name] * current).gt(tol)
            retention |= affected
            for instrument in idx[affected]:
                reasons[instrument].add(f'{name}:min')
        remaining = diagnose(True)
        for name, row in remaining.loc[remaining['bridge_min'].gt(tol)].iterrows():
            affected = loading[name].gt(0.)
            bridges.loc[affected] = np.maximum(bridges.loc[affected], float(row['bridge_min']))
            for instrument in idx[affected]:
                bridge_groups[instrument].append(f'{name}:min')
    excluded = low & policy_lo.le(tol) & ~retention
    cutoff_exit = cutoff_candidate & ~retention
    exit_ &= ~retention
    base = current.copy()
    reason = pd.Series('', index=idx, dtype=object)
    category = reason.copy()
    for mask, value, label in (
            (buy, lo, 'product_min_buy'), (sell, hi, 'product_max_sell'),
            (excluded, pd.Series(0., index=idx), 'small_cutoff'),
            (exit_ & policy_lo.le(tol), pd.Series(0., index=idx), 'model_exit'),
            (entry, effective, 'model_entry')):
        active = mask & ~cash
        base.loc[active] = value.loc[active]
        reason.loc[active] = reason.loc[active].where(
            reason.loc[active].eq(''), reason.loc[active] + '|') + label
        category.loc[active] = label
    mandatory = base.sub(current).abs().gt(tol) & ~cash
    funding = base.sub(current).where(~cash, 0.)
    total_funding = float(funding.sum())
    base.loc[cash_asset] = current.loc[cash_asset] - total_funding
    reason.loc[cash_asset] = 'mandatory_cash_funding' if abs(total_funding) > tol else ''
    category.loc[cash_asset] = reason.loc[cash_asset]
    effective.loc[mandatory | retention] = base.loc[mandatory | retention]
    effective.loc[cash_asset] = current.sum() - effective.loc[~cash].sum()
    if not (np.isclose(base.sum(), current.sum(), atol=tol, rtol=0.) and
            np.isclose(effective.sum(), current.sum(), atol=tol, rtol=0.)):
        raise RuntimeError('Monthly target is not self-financing')
    rule4 = raw.gt(cutoffs) & ~mandatory & ~retention & ~cash
    desired = effective - base
    trades = desired.where(~mandatory, base - current)
    trades.loc[cash_asset] = base.loc[cash_asset] - current.loc[cash_asset]
    material = desired.abs().ge(material_threshold)
    entry_funding = funding.where(category.eq('model_entry'), 0.)
    exit_funding = funding.where(category.isin(['small_cutoff', 'model_exit']), 0.)
    table = pd.DataFrame({
        s.CURRENT_WEIGHT: current, s.PREVIOUS_MODEL_WEIGHT: previous_model,
        s.RAW_MODEL_WEIGHT: model, s.POLICY_TARGET_WEIGHT: policy_target,
        s.EFFECTIVE_SMALL_TRADE_CUTOFF: cutoffs, s.EFFECTIVE_MODEL_WEIGHT: effective,
        s.BASE_WEIGHT: base, s.TRADE_WEIGHT: trades, s.LOW_WEIGHT_EXCLUDED: excluded,
        s.CUTOFF_SELL_DOWN: retention,
        s.CUTOFF_SELL_DOWN_REASON: pd.Series({i: '|'.join(sorted(reasons[i])) for i in idx}),
        s.CUTOFF_SELL_DOWN_RETENTION: np.nan, s.CUTOFF_SELL_DOWN_RETAINED_FRACTION: np.nan,
        s.CUTOFF_SELL_DOWN_DUST_ROUNDED: False, s.SELL_DOWN_DUST_THRESHOLD: dust_threshold,
        s.CUTOFF_RESIDUAL_BRIDGE: bridges, s.CUTOFF_RESIDUAL_BRIDGE_BP: 10000. * bridges,
        s.CUTOFF_RESIDUAL_BRIDGE_GROUP: pd.Series(
            {i: '|'.join(sorted(set(bridge_groups[i]))) for i in idx}),
        s.MODEL_ENTRY: entry, s.MODEL_EXIT: exit_, s.CUTOFF_EXIT: cutoff_exit,
        s.MANDATORY_REASON: reason, s.MANDATORY_CATEGORY: category,
        s.MANDATORY_FUNDING: funding.where(~cash, -total_funding),
        s.ENTRY_FUNDING: entry_funding, s.EXIT_FUNDING: exit_funding,
        s.DESK_FUNDING: 0., s.OTHER_FUNDING: funding - entry_funding - exit_funding,
        s.SETTLEMENT_CASH: cash, s.RULE4_TRADE: rule4,
        s.DESIRED_REWEIGHT: desired.where(rule4, 0.), s.MANDATORY_TRADE: mandatory,
        s.REBALANCE_CADENCE_ELIGIBLE: True, s.MATERIAL_TRADE: material,
        s.TRADE_CANDIDATE: rule4 & material, s.POLICY_MIN_WEIGHT: policy_lo,
        s.POLICY_MAX_WEIGHT: policy_hi, s.PRODUCT_MIN_WEIGHT: lo, s.PRODUCT_MAX_WEIGHT: hi,
    })
    for column in (s.INSTRUCTION_MIN_BUY, s.INSTRUCTION_MAX_SELL,
                   s.INSTRUCTION_WITHIN_TOLERANCE, s.FIXED_INSTRUCTION,
                   s.PINNED_INSTRUCTION, s.PRODUCT_BOUND_OVERRIDE):
        table[column] = False
    for column in (s.EXECUTION_INSTRUCTION, s.INSTRUCTION_COMMENT, s.INSTRUCTION_SOURCE,
                   s.PRODUCT_BOUND_OVERRIDE_REASON):
        table[column] = ''
    for column in (s.INSTRUCTION_RETAINED_BREACH, s.INSTRUCTION_MIN_WEIGHT,
                   s.PRODUCT_BOUND_OVERRIDE_BREACH):
        table[column] = 0.
    table[s.INSTRUCTION_MAX_WEIGHT] = 1.
    return table
