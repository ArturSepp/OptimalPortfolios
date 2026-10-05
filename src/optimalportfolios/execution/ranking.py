"""Legacy cash-funded alpha/TRE ranking over an already resolved target.

The per-candidate finite trade calculation is algorithmic scoring, not reported
portfolio analytics. qis.RiskModel reports portfolio risk but does not expose
this selection kernel; retain the reference arithmetic until parity is measured.
"""
from typing import Optional

import numpy as np
import pandas as pd

from optimalportfolios.execution.schema import (
    ALPHA,
    ASSET_CLASS,
    BASE_WEIGHT,
    COVAR_ACTIVE,
    DESIRED_REWEIGHT,
    FEASIBILITY_RESCUE_TRADE,
    MANDATORY_TRADE,
    MARGINAL_ALPHA,
    MARGINAL_TRE,
    MATERIAL_TRADE,
    RAW_MODEL_WEIGHT,
    REQUESTED_TRADE,
    SCORE_PER_TURNOVER,
    SELECTED_TRADE,
    SELECTION_MARGINAL_TRE,
    SELECTION_SCORE,
    SEQUENTIAL_SELECTION_ORDER,
    SETTLEMENT_CASH,
    TRADE_CANDIDATE,
    TRADE_RANK,
    TRADE_SCORE,
    TRE_WEIGHT,
    WEIGHT_ZERO_TOLERANCE,
)
from optimalportfolios.execution.types import ExecutionRankingConfig


def resolve_effective_max_trades(
        config: ExecutionRankingConfig,
        mandatory_ticket_count: int,
) -> Optional[int]:
    """Return the requested per-decision non-cash ticket allowance.

    Args:
        config: Execution policy carrying the configured ticket allowance.
        mandatory_ticket_count: Unique active mandatory non-cash tickets for
            the decision, after instruction and lifecycle overlaps are merged.

    Returns:
        Requested allowance, or None for unrestricted ranked selection. Rescue
        can admit additional trades after ranking.
    """
    if (isinstance(mandatory_ticket_count, (bool, np.bool_))
            or not isinstance(mandatory_ticket_count, (int, np.integer))
            or mandatory_ticket_count < 0):
        raise ValueError('mandatory_ticket_count must be a non-negative integer')
    if config.max_trades is None:
        return None
    if config.add_mandatory_trades_to_max_trades:
        return int(config.max_trades) + int(mandatory_ticket_count)
    return int(config.max_trades)


def _incremental_funded_tre(
        covariance: np.ndarray,
        active: np.ndarray,
        instrument_position: int,
        cash_position: int,
        delta: float,
) -> float:
    """Return the exact TRE change from one cash-funded instrument trade."""
    covar_active = covariance @ active
    variance = float(active @ covar_active)
    if variance < -1e-10:
        raise ValueError(f'active covariance has negative variance {variance}')
    directional_gradient = float(
        covar_active[instrument_position] - covar_active[cash_position]
    )
    funded_trade_variance = float(
        covariance[instrument_position, instrument_position]
        - 2.0 * covariance[instrument_position, cash_position]
        + covariance[cash_position, cash_position]
    )
    if funded_trade_variance < -1e-10:
        raise ValueError(
            'funded-trade variance is negative: '
            f'{funded_trade_variance}'
        )
    post_trade_variance = (
        variance
        + 2.0 * delta * directional_gradient
        + delta * delta * max(funded_trade_variance, 0.0)
    )
    if post_trade_variance < -1e-10:
        raise ValueError(
            f'incremental active variance is negative: {post_trade_variance}'
        )
    return float(
        np.sqrt(max(post_trade_variance, 0.0))
        - np.sqrt(max(variance, 0.0))
    )


def score_execution_trades(
        target: pd.DataFrame,
        alphas: pd.Series,
        covariance: pd.DataFrame,
        asset_classes: pd.Series,
        config: ExecutionRankingConfig,
        asset_class_tre_weights: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Add cash-aware marginal-TRE scores, ranks and selection indicators.

    The active portfolio is the post-mandatory base portfolio relative to the
    raw model. A prospective non-cash trade ``delta_i`` is paired with an
    opposite cash trade. Its TRE contribution is the exact change in portfolio
    TRE after applying that funded trade, including the quadratic trade term.
    Asset-class labels choose the configured base score coefficient and the
    optional caller-supplied class multiplier. Their product is the
    effective marginal-TRE coefficient; asset classes do not truncate the
    covariance matrix.

    Args:
        target: Consumer-resolved table using ``execution.schema`` column names.
        alphas: Model alpha scores at the decision date.
        covariance: Annualised model covariance matrix.
        asset_classes: Caller-defined asset class per instrument.
        config: Execution policy parameters and TRE coefficients.
        asset_class_tre_weights: Optional product TRE weights indexed by
            configured asset class. When supplied, the effective coefficient for
            instrument ``i`` is its configured coefficient multiplied by the
            supplied class multiplier for ``g(i)``. None preserves the
            legacy neutral multiplier of one.

    Returns:
        Copy of ``target`` with score, rank and selected-trade columns.
    """
    required = {
        BASE_WEIGHT, RAW_MODEL_WEIGHT, DESIRED_REWEIGHT, SETTLEMENT_CASH,
        MANDATORY_TRADE, TRADE_CANDIDATE, MATERIAL_TRADE,
    }
    missing = required.difference(target.columns)
    if missing:
        raise ValueError(f"execution target is missing columns {sorted(missing)}")
    index = target.index
    missing_covar = set(index).difference(covariance.index).union(
        set(index).difference(covariance.columns)
    )
    if missing_covar:
        raise ValueError(f"covariance is missing execution instruments {sorted(missing_covar)}")
    aligned_covar = covariance.loc[index, index].to_numpy(dtype=float)
    if not np.isfinite(aligned_covar).all():
        raise ValueError("execution covariance must be finite")
    if not np.allclose(aligned_covar, aligned_covar.T, atol=1e-10):
        raise ValueError("execution covariance must be symmetric")

    groups = asset_classes.reindex(index)
    if groups.isna().any():
        raise ValueError(
            f"missing asset classes for {groups.index[groups.isna()].tolist()}"
        )
    unsupported = set(groups).difference(tuple(config.tre_weight_by_asset_class))
    if unsupported:
        raise ValueError(f"unsupported asset classes {sorted(unsupported)}")

    if asset_class_tre_weights is None:
        product_tre_weights = pd.Series(1.0, index=tuple(config.tre_weight_by_asset_class))
    else:
        product_tre_weights = pd.to_numeric(
            asset_class_tre_weights.reindex(tuple(config.tre_weight_by_asset_class)),
            errors='coerce'
        )
        invalid = (
            product_tre_weights.isna()
            | ~np.isfinite(product_tre_weights)
            | product_tre_weights.lt(0.0)
        )
        if invalid.any():
            raise ValueError(
                'asset_class_tre_weights must define a finite, non-negative '
                "'TRE weight' for every configured asset class; invalid/missing: "
                f'{product_tre_weights.index[invalid].tolist()}'
            )

    out = target.copy()
    out[ASSET_CLASS] = groups
    out[ALPHA] = alphas.reindex(index).fillna(0.0).astype(float)
    if not np.isfinite(out[ALPHA].to_numpy()).all():
        raise ValueError("execution alphas must be finite")
    cash_rows = out[SETTLEMENT_CASH].astype(bool)
    if int(cash_rows.sum()) != 1:
        raise ValueError("execution scoring requires exactly one settlement-cash asset")
    if (cash_rows & out[MANDATORY_TRADE].astype(bool)).any():
        raise ValueError('settlement cash funding is not a mandatory non-cash ticket')
    cash_asset = index[cash_rows][0]
    desired_trade = out[DESIRED_REWEIGHT].astype(float)
    cash_alpha = float(out.loc[cash_asset, ALPHA])
    out[MARGINAL_ALPHA] = (out[ALPHA] - cash_alpha) * desired_trade
    out[COVAR_ACTIVE] = 0.0
    out[MARGINAL_TRE] = 0.0
    base_tre_weights = groups.map(config.tre_weight_by_asset_class).astype(float)
    out[TRE_WEIGHT] = (
        base_tre_weights * groups.map(product_tre_weights).astype(float)
    )

    active = out[BASE_WEIGHT] - out[RAW_MODEL_WEIGHT]
    active_array = active.to_numpy(dtype=float)
    covar_active_full = aligned_covar @ active_array
    variance_full = float(active_array @ covar_active_full)
    if variance_full < -1e-10:
        raise ValueError(f"active covariance has negative variance {variance_full}")
    out[COVAR_ACTIVE] = covar_active_full
    cash_position = int(index.get_loc(cash_asset))
    # QIS reports portfolio risk but does not expose the funded selection kernel.
    # Reuse the common matrix-vector product; the scalar reference remains above.
    deltas = desired_trade.to_numpy(dtype=float)
    moving = np.abs(deltas) > WEIGHT_ZERO_TOLERANCE
    gradients = covar_active_full - covar_active_full[cash_position]
    funded_variances = (np.diag(aligned_covar) - 2.0 * aligned_covar[:, cash_position]
                        + aligned_covar[cash_position, cash_position])
    invalid = moving & (funded_variances < -1e-10)
    if invalid.any():
        raise ValueError(f'funded-trade variance is negative: {funded_variances[invalid][0]}')
    post_variances = (variance_full + 2.0 * deltas * gradients
                      + deltas * deltas * np.maximum(funded_variances, 0.0))
    invalid = moving & (post_variances < -1e-10)
    if invalid.any():
        raise ValueError(f'incremental active variance is negative: {post_variances[invalid][0]}')
    out[MARGINAL_TRE] = np.where(
        moving, np.sqrt(np.maximum(post_variances, 0.0)) - np.sqrt(max(variance_full, 0.0)), 0.0)

    out[TRADE_SCORE] = out[MARGINAL_ALPHA] - out[TRE_WEIGHT] * out[MARGINAL_TRE]
    denominator = desired_trade.abs().where(out[MATERIAL_TRADE])
    out[SCORE_PER_TURNOVER] = out[TRADE_SCORE] / denominator

    candidate = out[TRADE_CANDIDATE].astype(bool)
    out[TRADE_RANK] = pd.Series(pd.NA, index=index, dtype='Int64')
    ranked_index = out.loc[candidate].sort_values(
        [TRADE_SCORE, SCORE_PER_TURNOVER], ascending=[False, False]
    ).index
    out.loc[ranked_index, TRADE_RANK] = pd.array(
        np.arange(1, len(ranked_index) + 1), dtype='Int64'
    )

    mandatory_count = int(out[MANDATORY_TRADE].astype(bool).sum())
    effective_max_trades = resolve_effective_max_trades(
        config=config,
        mandatory_ticket_count=mandatory_count,
    )
    capacity = (
        len(ranked_index) if effective_max_trades is None
        else max(effective_max_trades - mandatory_count, 0)
    )
    out[SELECTION_SCORE] = np.nan
    out[SELECTION_MARGINAL_TRE] = np.nan
    out[SEQUENTIAL_SELECTION_ORDER] = pd.Series(
        pd.NA, index=index, dtype='Int64'
    )
    if config.sequential_greedy_selection:
        sequential_active = active_array.copy()
        remaining = list(ranked_index)
        discretionary_index = []
        for selection_order in range(1, capacity + 1):
            rows = []
            for instrument in remaining:
                position = int(index.get_loc(instrument))
                delta = float(desired_trade.loc[instrument])
                marginal_tre = _incremental_funded_tre(
                    covariance=aligned_covar,
                    active=sequential_active,
                    instrument_position=position,
                    cash_position=cash_position,
                    delta=delta,
                )
                selection_score = float(
                    out.loc[instrument, MARGINAL_ALPHA]
                    - out.loc[instrument, TRE_WEIGHT] * marginal_tre
                )
                score_per_turnover = selection_score / abs(delta)
                if (config.minimum_trade_score is not None
                        and selection_score < config.minimum_trade_score):
                    continue
                rows.append((
                    instrument, selection_score, score_per_turnover,
                    marginal_tre,
                ))
            if not rows:
                break
            rows.sort(key=lambda row: (row[1], row[2]), reverse=True)
            instrument, selection_score, _, marginal_tre = rows[0]
            discretionary_index.append(instrument)
            remaining.remove(instrument)
            out.loc[instrument, SELECTION_SCORE] = selection_score
            out.loc[instrument, SELECTION_MARGINAL_TRE] = marginal_tre
            out.loc[instrument, SEQUENTIAL_SELECTION_ORDER] = selection_order
            position = int(index.get_loc(instrument))
            delta = float(desired_trade.loc[instrument])
            sequential_active[position] += delta
            sequential_active[cash_position] -= delta
        discretionary_index = pd.Index(discretionary_index)
    else:
        discretionary = candidate
        if config.minimum_trade_score is not None:
            discretionary &= out[TRADE_SCORE] >= config.minimum_trade_score
        discretionary_index = out.loc[discretionary].sort_values(
            [TRADE_SCORE, SCORE_PER_TURNOVER], ascending=[False, False]
        ).index[:capacity]
        out.loc[discretionary_index, SELECTION_SCORE] = out.loc[
            discretionary_index, TRADE_SCORE
        ]
        out.loc[discretionary_index, SELECTION_MARGINAL_TRE] = out.loc[
            discretionary_index, MARGINAL_TRE
        ]
        out.loc[discretionary_index, SEQUENTIAL_SELECTION_ORDER] = pd.array(
            np.arange(1, len(discretionary_index) + 1), dtype='Int64'
        )
    selected = pd.Series(False, index=index)
    selected.loc[discretionary_index] = True
    out[SELECTED_TRADE] = selected
    out[REQUESTED_TRADE] = selected
    out[FEASIBILITY_RESCUE_TRADE] = False
    return out
