"""Point-in-time alignment and rebalancing policy for portfolio constraints.

This internal module aligns immutable constraint specifications to the valid
solver universe, freezes non-rebalanced positions, and records any one-period
group-bound waiver required by frozen holdings. It does not construct solver
objects or import the public constraints facade at runtime.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, List, Optional, Tuple

import numpy as np
import pandas as pd

from optimalportfolios.optimization._constraint_groups import (
    GroupLowerUpperConstraints,
    _copy_optional_series,
)

if TYPE_CHECKING:
    from optimalportfolios.optimization.constraints import Constraints


logger = logging.getLogger('optimalportfolios.optimization.constraints')


def compute_eligible_rebalancing_bounds(
        current_weights: pd.Series,
        model_weights: pd.Series,
        current_min_weights: pd.Series,
        current_max_weights: pd.Series,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Build instrument bounds for trading a current portfolio toward a model.

    The eligible interval is the current/model corridor, narrowed by projecting
    the supplied current minimum and maximum weights into that corridor. An
    instrument is eligible for rebalancing when either its current or model
    weight is material. Consequently, a proposal can remain at the current
    weight or move toward the model, but cannot overshoot the model or open an
    instrument absent from both portfolios.

    All inputs are aligned to ``current_weights.index`` and missing values are
    treated as zero.

    Args:
        current_weights: Actual implemented portfolio weights.
        model_weights: Target model portfolio weights.
        current_min_weights: Candidate minimum implementation weights.
        current_max_weights: Candidate maximum implementation weights.

    Returns:
        Tuple containing eligible minimum weights, eligible maximum weights,
        and binary rebalancing indicators in that order.

    Raises:
        ValueError: If an aligned current minimum exceeds its current maximum.

    Wide candidate bounds of [0, 1] are narrowed to the current/model corridor, so a proposal can
    hold or move toward the model but never overshoot it. Asset `b` is already at its model weight
    and is pinned; `d` is in neither portfolio, so its indicator is 0 and its corridor is empty:

    >>> import pandas as pd
    >>> assets = ['a', 'b', 'c', 'd']
    >>> current = pd.Series([0.5, 0.3, 0.0, 0.0], index=assets)
    >>> model = pd.Series([0.2, 0.3, 0.5, 0.0], index=assets)
    >>> lower, upper, indicators = compute_eligible_rebalancing_bounds(
    ...     current, model, pd.Series(0.0, index=assets), pd.Series(1.0, index=assets))
    >>> lower.tolist()
    [0.2, 0.3, 0.0, 0.0]
    >>> upper.tolist()
    [0.5, 0.3, 0.5, 0.0]
    >>> indicators.tolist()
    [1, 1, 1, 0]
    """
    index = current_weights.index
    current = current_weights.reindex(index).fillna(0.0).astype(float)
    model = model_weights.reindex(index).fillna(0.0).astype(float)
    current_min = current_min_weights.reindex(index).fillna(0.0).astype(float)
    current_max = current_max_weights.reindex(index).fillna(0.0).astype(float)

    invalid_bounds = current_min > current_max + 1e-12
    if invalid_bounds.any():
        invalid = pd.DataFrame({
            'current_min': current_min.loc[invalid_bounds],
            'current_max': current_max.loc[invalid_bounds],
        })
        raise ValueError(
            "current_min_weights exceeds current_max_weights:\n"
            f"{invalid.to_string()}"
        )

    corridor_min = np.minimum(current.to_numpy(), model.to_numpy())
    corridor_max = np.maximum(current.to_numpy(), model.to_numpy())
    eligible_min = pd.Series(
        np.clip(current_min.to_numpy(), corridor_min, corridor_max),
        index=index,
    )
    eligible_max = pd.Series(
        np.clip(current_max.to_numpy(), corridor_min, corridor_max),
        index=index,
    )
    rebalancing_indicators = (
        (current.abs() > 1e-8) | (model.abs() > 1e-8)
    ).astype(int)

    return eligible_min, eligible_max, rebalancing_indicators


@dataclass(frozen=True)
class RelaxationRecord:
    """Structured record of a frozen-overhang group-bound relaxation.

    Attached to the log record under ``extra={"relaxation": ...}`` so a handler
    can aggregate the per-rebalance relaxations into one run-level tally instead
    of flooding the console. ``items`` is a tuple of (group, kind, old, new)
    where ``kind`` is ``"group_max"`` or ``"group_min"``.
    """
    context: str
    items: Tuple[Tuple[str, str, float, float], ...]
    total_relaxation: float
    max_relaxation: float
    breached_budget: bool
    breached_tol: bool


def _reindex_optional_series(s: Optional[pd.Series], index: pd.Index, fill_value: float = 0.0) -> Optional[pd.Series]:
    """Reindex a Series to align with given index, filling missing values.

    Args:
        s: Series to reindex (may be None).
        index: Target index to align to.
        fill_value: Value for missing entries.

    Returns:
        Reindexed Series or None if input is None.
    """
    if s is None:
        return None
    return s.reindex(index=index, fill_value=fill_value)


def build_valid_ticker_constraint_fields(
        constraint_spec: Constraints,
        valid_tickers: List[str],
        total_to_good_ratio: Optional[float] = None,
        weights_0: pd.Series = None,
        asset_returns: pd.Series = None,
        benchmark_weights: pd.Series = None,
        target_return: float = None,
        rebalancing_indicators: pd.Series = None,
        context: str = '',
        max_relaxation_tol: Optional[float] = None,
        relax_frozen_group_bounds: bool = True,
) -> dict:
    """Build ticker-aligned fields with the existing rebalancing policy.

    All pandas Series fields are reindexed to ``valid_tickers``. Assets with a
    zero rebalancing indicator are frozen at current weights, and group bounds
    may receive the same logged one-period waiver as
    ``Constraints.update_with_valid_tickers``.

    Args:
        constraint_spec: Immutable constraint specification to align.
        valid_tickers: List of tickers to retain.
        total_to_good_ratio: Scaling factor for constrained exposure.
        weights_0: Current portfolio weights.
        asset_returns: Expected asset returns.
        benchmark_weights: Benchmark portfolio weights.
        target_return: Target portfolio return.
        rebalancing_indicators: Binary indicators (1=rebalance, 0=hold fixed).
        context: Rebalance label used in any constraint-relaxation logs.
        max_relaxation_tol: Optional maximum permitted relative relaxation
            when fixed-position constraints must be reconciled.
        relax_frozen_group_bounds: Whether frozen positions may widen group
            allocation bounds. Disable for execution-policy projection, where
            an infeasible selected trade set must remain visible.

    Returns:
        Dictionary of constraint fields aligned to ``valid_tickers``.
    """
    valid_index = pd.Index(valid_tickers)
    self_dict = constraint_spec._to_dict()

    # Update individual weight constraints — aligned to valid_tickers
    if constraint_spec.min_weights is not None:
        self_dict['min_weights'] = constraint_spec.min_weights.reindex(index=valid_index, fill_value=0.0)
    if constraint_spec.max_weights is not None:
        max_w = constraint_spec.max_weights.reindex(index=valid_index, fill_value=0.0)
        if total_to_good_ratio is not None:
            max_w = max_w.where(np.isclose(max_w, 1.0), other=total_to_good_ratio * max_w)
        self_dict['max_weights'] = max_w

    # Update group constraints
    if constraint_spec.group_lower_upper_constraints is not None:
        self_dict['group_lower_upper_constraints'] = \
            constraint_spec.group_lower_upper_constraints.update(valid_tickers=valid_tickers)
    if constraint_spec.group_tracking_error_constraint is not None:
        self_dict['group_tracking_error_constraint'] = \
            constraint_spec.group_tracking_error_constraint.update(valid_tickers=valid_tickers)
    if constraint_spec.group_turnover_constraint is not None:
        self_dict['group_turnover_constraint'] = \
            constraint_spec.group_turnover_constraint.update(valid_tickers=valid_tickers)

    # Update turnover constraints with exposure scaling
    if constraint_spec.turnover_constraint is not None and total_to_good_ratio is not None:
        self_dict['turnover_constraint'] = constraint_spec.turnover_constraint * total_to_good_ratio
    if constraint_spec.turnover_costs is not None:
        self_dict['turnover_costs'] = constraint_spec.turnover_costs.reindex(index=valid_index, fill_value=1.0)

    # Update portfolio universe — all aligned to valid_tickers
    if weights_0 is not None:
        self_dict['weights_0'] = weights_0.reindex(index=valid_index, fill_value=0.0)
    elif constraint_spec.weights_0 is not None:
        self_dict['weights_0'] = constraint_spec.weights_0.reindex(index=valid_index, fill_value=0.0)

    if asset_returns is not None:
        self_dict['asset_returns'] = asset_returns.reindex(index=valid_index, fill_value=0.0)
    elif constraint_spec.asset_returns is not None:
        self_dict['asset_returns'] = constraint_spec.asset_returns.reindex(index=valid_index, fill_value=0.0)

    if benchmark_weights is not None:
        self_dict['benchmark_weights'] = benchmark_weights.reindex(index=valid_index, fill_value=0.0)
    elif constraint_spec.benchmark_weights is not None:
        self_dict['benchmark_weights'] = constraint_spec.benchmark_weights.reindex(index=valid_index, fill_value=0.0)

    if target_return is not None:
        self_dict['target_return'] = target_return

    # Apply rebalancing indicators to freeze certain positions
    resolved_weights_0 = self_dict.get('weights_0')
    if rebalancing_indicators is not None and resolved_weights_0 is not None:
        rebal = rebalancing_indicators.reindex(index=valid_index, fill_value=1.0)
        is_rebalanced = np.isclose(rebal, 1.0)
        # Frozen (non-rebalanced) assets inherit weights_0 as both their
        # lower and upper bound. For a long-only book that bound cannot be
        # negative, but a drifted weights_0 can carry a tiny negative from a
        # prior solve (cvx honours the >= 0 constraint only to ~1e-8), which
        # would set min_weights < 0 here and trip the long-only validation in
        # __post_init__. Floor the frozen bound at 0 for long-only so a lone
        # asset frozen at a numerically-negative weight is pinned to 0.
        frozen_weights_0 = (resolved_weights_0.clip(lower=0.0)
                            if constraint_spec.is_long_only else resolved_weights_0)
        if self_dict['min_weights'] is not None:
            self_dict['min_weights'] = self_dict['min_weights'].where(is_rebalanced, other=frozen_weights_0)
        if self_dict['max_weights'] is not None:
            self_dict['max_weights'] = self_dict['max_weights'].where(is_rebalanced, other=frozen_weights_0)

    # Relax group bounds to accommodate the frozen-position overhang.
    #
    # The freeze step above pins min/max for non-rebalanced assets to
    # ``weights_0``. When ``weights_0`` comes from a drift step (as in
    # the rolling backtest after the use_drifted_weights_0 patch) or
    # from a live portfolio-management system that is slightly out of
    # compliance, the frozen positions can push a group's loading-
    # weighted min above its group_max_allocation (or, symmetrically,
    # push frozen max below group_min_allocation). The optimiser
    # cannot trade frozen assets, so the only feasible resolution is
    # to relax the group bound for this rebalance.
    #
    # We grant a one-period waiver: raise group_max_allocation to the
    # frozen-min sum (or lower group_min_allocation to the frozen-max
    # sum), with a small tolerance. A warning is emitted so the
    # relaxation is visible in logs. The drift-induced overshoot is
    # typically a few tens of basis points; for live-PMS-induced
    # overshoots this is the equivalent of a compliance waiver.
    gluc = self_dict.get('group_lower_upper_constraints')
    min_w = self_dict.get('min_weights')
    max_w = self_dict.get('max_weights')
    if (relax_frozen_group_bounds and gluc is not None
            and (min_w is not None or max_w is not None)):
        loadings = gluc.group_loadings
        gmin = gluc.group_min_allocation
        gmax = gluc.group_max_allocation
        new_gmin = _copy_optional_series(gmin)
        new_gmax = _copy_optional_series(gmax)
        tol = 1e-4
        relax_msgs = []
        relax_items = []
        for group in loadings.columns:
            group_loading = loadings[group]
            members = group_loading.index[group_loading > 0]
            if len(members) == 0:
                continue
            member_loadings = group_loading.loc[members]
            # cap overshoot from frozen min
            if gmax is not None and min_w is not None:
                gmax_val = gmax.get(group, np.nan)
                if not np.isnan(gmax_val):
                    group_min_sum = float(
                        (min_w.reindex(members, fill_value=0.0)
                         * member_loadings).sum())
                    if group_min_sum > gmax_val + tol:
                        new_gmax.loc[group] = group_min_sum + tol
                        relax_msgs.append(
                            f"  group '{group}': group_max_allocation "
                            f"{gmax_val:.4f} → {group_min_sum + tol:.4f} "
                            f"(frozen-min overshoot)")
                        relax_items.append(
                            (str(group), "group_max", float(gmax_val),
                             float(group_min_sum + tol)))
            # floor undershoot from frozen max
            if gmin is not None and max_w is not None:
                gmin_val = gmin.get(group, np.nan)
                if not np.isnan(gmin_val):
                    group_max_sum = float(
                        (max_w.reindex(members, fill_value=1.0)
                         * member_loadings).sum())
                    if group_max_sum < gmin_val - tol:
                        new_gmin.loc[group] = group_max_sum - tol
                        relax_msgs.append(
                            f"  group '{group}': group_min_allocation "
                            f"{gmin_val:.4f} → {group_max_sum - tol:.4f} "
                            f"(frozen-max undershoot)")
                        relax_items.append(
                            (str(group), "group_min", float(gmin_val),
                             float(group_max_sum - tol)))
        if relax_msgs:
            _tag = f"[{context}] " if context else ""
            _msg = (
                _tag + "Constraints.update_with_valid_tickers: relaxing group "
                "bounds for frozen-position overhang (drift or live "
                "PMS state):\n" + "\n".join(relax_msgs))
            max_exposure = self_dict.get('max_exposure', 1.0)
            deltas = [abs(new - old) for _, _, old, new in relax_items]
            total_relaxation = float(sum(deltas))
            max_relax = float(max(deltas)) if deltas else 0.0
            breached_budget = bool(
                new_gmax is not None
                and len(new_gmax[new_gmax > max_exposure + tol]) > 0)
            breached_tol = bool(
                max_relaxation_tol is not None and max_relax > max_relaxation_tol)
            record = RelaxationRecord(
                context=context, items=tuple(relax_items),
                total_relaxation=total_relaxation, max_relaxation=max_relax,
                breached_budget=breached_budget, breached_tol=breached_tol)
            if breached_tol:
                _msg += (f"\n  max single relaxation {max_relax:.4f} exceeds "
                         f"tolerance {max_relaxation_tol:.4f}")
            # Per-rebalance detail at INFO (file); escalate to ERROR when the
            # relaxation magnitude breaches the tolerance or the budget. A
            # run-level RelaxationSummary aggregates these into one line.
            _level = logging.ERROR if (breached_tol or breached_budget) else logging.INFO
            logger.log(_level, _msg, extra={"relaxation": record})
            if breached_budget:
                breached = new_gmax[new_gmax > max_exposure + tol]
                logger.error(
                    _tag + "Constraints.update_with_valid_tickers: frozen "
                    "overhang relaxed group_max above max_exposure "
                    "(%s) for %s; constraints are effectively "
                    "infeasible — the solve output must be "
                    "validated.", max_exposure, breached.to_dict())
            self_dict['group_lower_upper_constraints'] = \
                GroupLowerUpperConstraints(
                    group_loadings=loadings,
                    group_min_allocation=new_gmin,
                    group_max_allocation=new_gmax,
                )

    # Update sector and style deviation constraints
    if constraint_spec.sector_deviation_constraints is not None:
        self_dict["sector_deviation_constraints"] = \
            constraint_spec.sector_deviation_constraints.update(valid_tickers=valid_tickers)
    if constraint_spec.style_deviation_constraints is not None:
        self_dict["style_deviation_constraints"] = \
            constraint_spec.style_deviation_constraints.update(valid_tickers=valid_tickers)

    return self_dict
