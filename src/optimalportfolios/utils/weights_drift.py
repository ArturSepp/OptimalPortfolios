"""
Drift adjustment for prior-period weights used as turnover-penalty baseline.

Rolling backtests carry weights_0 forward from one rebalance date to the next
so that turnover constraints and transaction cost penalties act on a sensible
baseline. The natural choice is the actual portfolio holdings at the rebalance
date, which differ from the previous-period *target* weights by the realised
return drift over the holding period.

This module provides ``apply_drift_to_weights_0``, a constraint-agnostic
helper used by every rolling optimiser to convert the previous-period target
weights into the implied current holdings under the standard self-financing
identity::

    w_drift_i = w_i * (1 + r_i) / (1 + sum_j w_j * r_j)

The denominator is NAV growth over the period. For long-only fully-invested
portfolios this reduces to the conventional ``gross / sum(gross)`` form, but
the NAV-growth divisor remains correct for long-short and variable-exposure
mandates where ``sum w`` is not 1.

The helper is silent: all failure modes (missing prices, missing dates, NaN
returns, zero weights_0, NAV collapse, toggle off) return weights_0 unchanged,
so the caller falls back to passing the previous target without any need for
constraint introspection.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional


def apply_drift_to_weights_0(
        weights_0: Optional[pd.Series],
        prices: Optional[pd.DataFrame],
        prev_date: Optional[pd.Timestamp],
        date: pd.Timestamp,
        use_drifted_weights_0: bool = True,
        eps: float = 1e-12,
) -> Optional[pd.Series]:
    """Drift weights_0 from prev_date to date using realised price returns.

    Implements the self-financing identity
    ``w_drift = w * (1 + r) / (1 + sum w * r)`` where the denominator is
    NAV growth. Constraint-agnostic; works for long-only, long-short, and
    variable-exposure mandates.

    All failure modes silently return weights_0 unchanged so the caller can
    treat this as a transparent passthrough. The function is designed to be
    safe to call unconditionally inside a rolling loop.

    Args:
        weights_0: Prior-period target weights (Series indexed by asset).
            None or all-zero → no drift.
        prices: Asset price panel (DataFrame of total return prices).
            None → no drift. Index must contain timestamps comparable to
            prev_date and date.
        prev_date: Date at which weights_0 were established. None on the
            first rebalance → no drift.
        date: Current rebalance date.
        use_drifted_weights_0: Master toggle. False → no drift (legacy
            behaviour where the previous target is reused as-is).
        eps: Tolerance for zero-checks on weights_0 magnitude and NAV growth.

    Returns:
        Drifted weights as a pd.Series with the same index as weights_0,
        or weights_0 unchanged when any gate fails. NaN values in the
        returned Series are replaced with the corresponding weights_0 entry
        (asset treated as flat over the period).
    """
    # --- gates (silent fallback on failure) ---
    if not use_drifted_weights_0:
        return weights_0
    if weights_0 is None:
        return weights_0
    if prices is None or prev_date is None:
        return weights_0
    # zero / near-zero weights_0 means the prior step was a degenerate
    # fallback; resume cold-start on the next call.
    if float(np.abs(weights_0.to_numpy()).sum()) < eps:
        return weights_0

    # --- locate price anchors with ffill semantics ---
    # ``loc[:date]`` selects the prefix; ffill within the prefix handles
    # intermittent NaN gaps; ``iloc[-1]`` then gives the most recent valid
    # snapshot per asset. Assets with no valid price before the anchor
    # remain NaN and are treated as flat below.
    try:
        prefix_prev = prices.loc[:prev_date]
        prefix_curr = prices.loc[:date]
    except (KeyError, TypeError):
        return weights_0
    if len(prefix_prev) == 0 or len(prefix_curr) == 0:
        return weights_0

    p0 = prefix_prev.ffill().iloc[-1]
    p1 = prefix_curr.ffill().iloc[-1]

    # --- compute per-asset price ratio with NaN/inf hygiene ---
    # divide-by-zero or non-positive p0 → flat; NaN on either side → flat.
    p0_safe = p0.where(p0 > 0.0)
    p_ratio = (p1 / p0_safe).replace([np.inf, -np.inf], np.nan)

    # align to weights_0 index; assets not in prices → flat (ratio = 1).
    p_ratio = p_ratio.reindex(weights_0.index)

    # asset returns; NaN ratios → 0 return so the NAV-growth sum is
    # well-defined regardless of price-panel sparsity.
    asset_returns = (p_ratio - 1.0).fillna(0.0)

    # --- NAV growth and renormalisation ---
    nav_growth = 1.0 + float((weights_0 * asset_returns).sum())
    if not np.isfinite(nav_growth) or nav_growth < eps:
        # NAV collapse or non-finite → undefined drift; fall back.
        return weights_0

    # assets with NaN ratio drift trivially (multiplier 1.0); others scale
    # by (1 + r_i). Then divide everything by NAV growth.
    drift_multiplier = p_ratio.fillna(1.0)
    drifted = weights_0 * drift_multiplier / nav_growth

    # defensive: if anything went non-finite (shouldn't, but be safe),
    # fall back element-wise to the original weight.
    drifted = drifted.where(np.isfinite(drifted), other=weights_0)
    return drifted
