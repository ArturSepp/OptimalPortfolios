"""
Universe transformations for portfolio optimisation.

This module contains functions that take a UniverseData instance and return
a modified copy, keeping UniverseData itself as a pure data container.
"""
from __future__ import annotations

import pandas as pd
import qis as qis
from typing import Union, Optional

from optimalportfolios.universe.universe_data import UniverseData
from optimalportfolios.utils.returns_unsmoother import compute_ar1_unsmoothed_prices


def copy_universe_data_with_unsmoothed_prices(
        universe_data: UniverseData,
        assets_for_unsmoothing: pd.Series,
        freq: Union[str, pd.Series] = 'QE',
        unsmooth_span: int = 40,
        mean_adj_type: qis.MeanAdjType = qis.MeanAdjType.EWMA,
        warmup_period: Optional[int] = 8,
        max_value_for_beta: Optional[float] = 0.75,
        is_log_returns: bool = True,
) -> UniverseData:
    """Return a new UniverseData with AR(1)-unsmoothed prices for selected assets.

    Illiquid or appraisal-valued assets (e.g. private equity, real estate)
    exhibit serial correlation that understates true volatility. This function
    applies AR(1) unsmoothing to the flagged assets while leaving the rest
    unchanged, then wraps the result in a fresh UniverseData instance.

    Args:
        universe_data: Source universe (not mutated).
        assets_for_unsmoothing: Boolean Series indexed by asset names.
            True = apply unsmoothing to that asset.
        freq: Return frequency for AR(1) estimation.
            Either a single pandas offset string (e.g. 'QE') applied to all
            assets, or a per-asset Series of offset strings.
        unsmooth_span: EWMA span (in periods) for rolling beta estimation.
        mean_adj_type: Method for mean-adjusting returns before estimation.
        warmup_period: Number of initial periods excluded from estimation.
        max_value_for_beta: Upper bound on estimated AR(1) coefficient;
            prevents over-correction when autocorrelation is very high.
        is_log_returns: If True, compute log-returns; otherwise simple returns.

    Returns:
        A new UniverseData with identical metadata and group loadings,
        but with unsmoothed prices for the flagged assets.

    Raises:
        ValueError: If assets_for_unsmoothing or freq index does not match
            the universe assets.
    """
    # --- validate index alignment ------------------------------------------------
    universe_index = universe_data.metadata.index
    if not universe_index.equals(assets_for_unsmoothing.index):
        raise ValueError(
            f"assets_for_unsmoothing index does not match universe: "
            f"missing={set(universe_index) - set(assets_for_unsmoothing.index)}, "
            f"extra={set(assets_for_unsmoothing.index) - set(universe_index)}"
        )
    if isinstance(freq, pd.Series) and not universe_index.equals(freq.index):
        raise ValueError("freq Series index does not match universe assets")

    # --- identify assets to unsmooth ---------------------------------------------
    un_assets = assets_for_unsmoothing[assets_for_unsmoothing].index.tolist()
    if not un_assets:
        return universe_data  # nothing to do

    if isinstance(freq, pd.Series):
        freq = freq.loc[un_assets]

    # --- apply AR(1) unsmoothing to selected columns -----------------------------
    prices = universe_data.prices.copy()
    un_prices, _, _, _ = compute_ar1_unsmoothed_prices(
        prices=prices.reindex(columns=un_assets),
        freq=freq,
        span=unsmooth_span,
        max_value_for_beta=max_value_for_beta,
        warmup_period=warmup_period,
        mean_adj_type=mean_adj_type,
        is_log_returns=is_log_returns
    )
    # align back to the full date index and forward-fill any leading NaNs
    prices[un_assets] = un_prices.reindex(index=prices.index).ffill()

    # --- build a new immutable UniverseData with the same metadata ---------------
    return UniverseData(
        prices=prices,
        metadata=universe_data.metadata,
        metadata_fields=universe_data.metadata_fields,
        group_loadings_level1=universe_data.group_loadings_level1,
        group_loadings_level2=universe_data.group_loadings_level2,
        equity_ac_id=universe_data.equity_ac_id,
        bond_ac_id=universe_data.bond_ac_id,
        pe_asset_id=universe_data.pe_asset_id,
        validate_on_init=universe_data.validate_on_init,
    )
