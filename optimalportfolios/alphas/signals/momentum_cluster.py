"""
Cluster-based momentum alpha computation.

Computes momentum scores (same as momentum.py) but applies
cross-sectional scoring within time-varying statistical clusters
rather than fixed user-defined groups.

Pipeline:
    returns → excess returns (vs benchmark) → EWMA long/short filtered RA returns
            → score within time-varying clusters
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import qis as qis
from typing import Optional, Tuple, Union, Dict

from optimalportfolios.alphas.signals.utils import score_within_clusters


def compute_momentum_cluster_alpha(
        prices: pd.DataFrame,
        benchmark_price: pd.Series = None,
        rolling_clusters: Dict[pd.Timestamp, pd.Series] = None,
        returns_freq: Union[str, pd.Series] = 'ME',
        long_span: int = 12,
        short_span: Optional[int] = None,
        vol_span: Optional[int] = 13,
        mean_adj_type: qis.MeanAdjType = qis.MeanAdjType.NONE,
        min_cluster_size: int = 3,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute momentum scores with time-varying cluster-based scoring.

    Identical to compute_momentum_alpha for the raw signal computation
    (excess returns → EWMA long/short filtered RA returns), but applies
    cross-sectional scoring within statistical clusters that evolve over
    time rather than fixed user-defined groups.

    Args:
        prices: Asset price panel. Index=dates, columns=tickers.
        benchmark_price: Benchmark price series for excess return computation.
            If None, uses raw returns (no benchmark subtraction).
        rolling_clusters: Dict mapping dates to pd.Series (ticker → cluster_id).
            Extracted from RollingFactorCovarData via extract_rolling_clusters().
        returns_freq: Return frequency. String or pd.Series for mixed-freq.
        long_span: EWMA span for the long momentum signal.
        short_span: Optional EWMA span for short-term reversal subtraction.
        vol_span: EWMA span for volatility normalisation. None disables.
        mean_adj_type: Mean adjustment type for EWMA computation.
        min_cluster_size: Minimum cluster size for within-cluster scoring.
            Clusters with size <= min_cluster_size use global statistics.

    Returns:
        Tuple of (momentum_cluster_score, raw_momentum).
    """
    if rolling_clusters is None:
        rolling_clusters = {}

    if isinstance(returns_freq, pd.Series):
        raw_momentum = _compute_raw_momentum_mixed_freq(
            prices=prices, benchmark_price=benchmark_price,
            returns_freqs=returns_freq, long_span=long_span,
            short_span=short_span, vol_span=vol_span,
            mean_adj_type=mean_adj_type)
    else:
        raw_momentum = _compute_raw_momentum_single_freq(
            prices=prices, benchmark_price=benchmark_price,
            returns_freq=returns_freq, long_span=long_span,
            short_span=short_span, vol_span=vol_span,
            mean_adj_type=mean_adj_type)

    # score within time-varying clusters
    momentum_cluster_score = score_within_clusters(
        raw_signal=raw_momentum,
        rolling_clusters=rolling_clusters,
        min_cluster_size=min_cluster_size)

    return momentum_cluster_score, raw_momentum


def _compute_raw_momentum_single_freq(
        prices: pd.DataFrame,
        benchmark_price: pd.Series = None,
        returns_freq: str = 'ME',
        long_span: int = 12,
        short_span: Optional[int] = None,
        vol_span: Optional[int] = 13,
        mean_adj_type: qis.MeanAdjType = qis.MeanAdjType.EWMA
) -> pd.DataFrame:
    """Compute raw EWMA long/short filtered RA excess returns (before scoring)."""
    returns = qis.to_returns(prices, freq=returns_freq, is_log_returns=True)

    if benchmark_price is not None:
        benchmark_price = benchmark_price.reindex(index=prices.index, method='ffill')
        benchmark_returns = qis.to_returns(
            benchmark_price, freq=returns_freq, is_log_returns=True)
        returns = returns.subtract(
            qis.np_array_to_df_columns(
                a=benchmark_returns.to_numpy(), ncols=len(returns.columns)))

    raw_momentum = qis.compute_ewm_long_short_filtered_ra_returns(
        returns=returns, vol_span=vol_span, long_span=long_span,
        short_span=short_span, weight_lag=0, mean_adj_type=mean_adj_type,
        warmup_period=long_span)

    return raw_momentum


def _compute_raw_momentum_mixed_freq(
        prices: pd.DataFrame,
        benchmark_price: pd.Series = None,
        returns_freqs: pd.Series = None,
        long_span: int = 12,
        short_span: Optional[int] = None,
        vol_span: Optional[int] = 13,
        mean_adj_type: qis.MeanAdjType = qis.MeanAdjType.NONE
) -> pd.DataFrame:
    """Mixed-frequency: compute raw momentum per frequency group, merge."""
    returns_freqs = returns_freqs[prices.columns]
    group_freqs = qis.get_group_dict(group_data=returns_freqs)

    all_raw = []
    for freq, asset_tickers in group_freqs.items():
        freq_prices = prices[asset_tickers]
        raw = _compute_raw_momentum_single_freq(
            prices=freq_prices, benchmark_price=benchmark_price,
            returns_freq=freq, long_span=long_span,
            short_span=short_span, vol_span=vol_span,
            mean_adj_type=mean_adj_type)
        all_raw.append(raw)

    return pd.concat(all_raw, axis=1)[prices.columns].ffill()