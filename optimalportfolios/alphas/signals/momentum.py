"""
Momentum alpha computation (standard + cluster scoring).

Computes cross-sectional momentum scores from asset returns relative to a
benchmark. Supports single-frequency and mixed-frequency universes.

Two public entry points share the same raw signal:
    * ``compute_momentum_alpha`` — scores cross-sectionally, optionally
      within fixed user-defined groups (``group_data``).
    * ``compute_momentum_cluster_alpha`` — scores within time-varying
      statistical clusters (``rolling_clusters``) rather than fixed groups.

Pipeline:
    returns → excess returns (vs benchmark) → EWMA long/short filtered RA returns
            → cross-sectional score (global / within groups / within clusters)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import qis as qis
from typing import Optional, Tuple, Union, Dict

from optimalportfolios.alphas.signals.utils import score_within_clusters


# ---------------------------------------------------------------------------
# raw signal — shared by the standard and cluster entry points
# ---------------------------------------------------------------------------
def _compute_raw_momentum_single_freq(prices: pd.DataFrame,
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
        benchmark_returns = qis.to_returns(benchmark_price, freq=returns_freq, is_log_returns=True)
        returns = returns.subtract(
            qis.np_array_to_df_columns(a=benchmark_returns.to_numpy(), ncols=len(returns.columns)))

    raw_momentum = qis.compute_ewm_long_short_filtered_ra_returns(
        returns=returns, vol_span=vol_span, long_span=long_span,
        short_span=short_span, weight_lag=0, mean_adj_type=mean_adj_type,
        warmup_period=long_span)

    return raw_momentum


def _compute_raw_momentum_mixed_freq(prices: pd.DataFrame,
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


# ---------------------------------------------------------------------------
# standard — cross-sectional scoring (optionally within fixed groups)
# ---------------------------------------------------------------------------
def compute_momentum_alpha(prices: pd.DataFrame,
                           benchmark_price: pd.Series = None,
                           returns_freq: Union[str, pd.Series] = 'ME',
                           group_data: Optional[pd.Series] = None,
                           long_span: int = 12,
                           short_span: Optional[int] = None,
                           vol_span: Optional[int] = 13,
                           mean_adj_type: qis.MeanAdjType = qis.MeanAdjType.NONE
                           ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute cross-sectional momentum alpha scores.

    For each asset, computes risk-adjusted cumulative excess return
    (relative to benchmark) using EWMA long/short filtering, then
    converts to a cross-sectional score.

    Handles both single-frequency and mixed-frequency universes:
    - ``returns_freq`` is a string: all assets at the same frequency.
    - ``returns_freq`` is a pd.Series: per-asset frequency.

    When ``group_data`` is provided, cross-sectional scoring is computed
    within each group independently.

    Args:
        prices: Asset price panel. Index=dates, columns=tickers.
        benchmark_price: Benchmark price series for excess return computation.
            If None, uses equal-weight average of asset returns.
        returns_freq: Return frequency. String or pd.Series mapping tickers
            to frequencies for mixed-frequency computation.
        group_data: Optional group labels per asset for within-group scoring.
        long_span: EWMA span for the long momentum signal.
        short_span: Optional EWMA span for short-term reversal subtraction.
        vol_span: EWMA span for volatility normalisation. None disables.
        mean_adj_type: Mean adjustment type for EWMA vol computation.

    Returns:
        Tuple of (momentum_score, raw_momentum).
    """
    if isinstance(returns_freq, pd.Series):
        return _compute_momentum_alpha_mixed_freq(
            prices=prices, benchmark_price=benchmark_price,
            returns_freqs=returns_freq, group_data=group_data,
            long_span=long_span, short_span=short_span,
            vol_span=vol_span, mean_adj_type=mean_adj_type)
    else:
        return _compute_momentum_alpha_single_freq(
            prices=prices, benchmark_price=benchmark_price,
            returns_freq=returns_freq, group_data=group_data,
            long_span=long_span, short_span=short_span,
            vol_span=vol_span, mean_adj_type=mean_adj_type)


def _compute_momentum_alpha_single_freq(prices: pd.DataFrame,
                                        benchmark_price: pd.Series = None,
                                        returns_freq: str = 'ME',
                                        group_data: Optional[pd.Series] = None,
                                        long_span: int = 12,
                                        short_span: Optional[int] = None,
                                        vol_span: Optional[int] = 13,
                                        mean_adj_type: qis.MeanAdjType = qis.MeanAdjType.EWMA
                                        ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Single-frequency momentum: raw signal, then cross-sectional scoring."""
    raw_momentum = _compute_raw_momentum_single_freq(
        prices=prices, benchmark_price=benchmark_price,
        returns_freq=returns_freq, long_span=long_span,
        short_span=short_span, vol_span=vol_span, mean_adj_type=mean_adj_type)

    if group_data is not None:
        grouped_prices = qis.split_df_by_groups(df=prices, group_data=group_data)
        group_scores = []
        for group, gprice in grouped_prices.items():
            group_scores.append(qis.df_to_cross_sectional_score(df=raw_momentum[gprice.columns]))
        momentum_score = pd.concat(group_scores, axis=1)[prices.columns]
    else:
        momentum_score = qis.df_to_cross_sectional_score(df=raw_momentum)

    return momentum_score, raw_momentum


def _compute_momentum_alpha_mixed_freq(prices: pd.DataFrame,
                                       benchmark_price: pd.Series = None,
                                       returns_freqs: pd.Series = None,
                                       group_data: Optional[pd.Series] = None,
                                       long_span: int = 12,
                                       short_span: Optional[int] = None,
                                       vol_span: Optional[int] = 13,
                                       mean_adj_type: qis.MeanAdjType = qis.MeanAdjType.NONE
                                       ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Mixed-frequency momentum: compute per (frequency × group), merge."""
    returns_freqs = returns_freqs[prices.columns]
    group_freqs = qis.get_group_dict(group_data=returns_freqs)

    all_scores = []
    all_momentum = []
    for freq, asset_tickers in group_freqs.items():
        freq_prices = prices[asset_tickers]
        if group_data is not None:
            freq_group_data = group_data.loc[group_data.index.intersection(asset_tickers)]
            grouped_prices = qis.split_df_by_groups(df=freq_prices, group_data=freq_group_data)
        else:
            grouped_prices = {'_': freq_prices}

        for group, gprice in grouped_prices.items():
            score, momentum = _compute_momentum_alpha_single_freq(
                prices=gprice, benchmark_price=benchmark_price,
                returns_freq=freq, group_data=None,
                long_span=long_span, short_span=short_span,
                vol_span=vol_span, mean_adj_type=mean_adj_type)
            all_scores.append(score)
            all_momentum.append(momentum)

    momentum_score = pd.concat(all_scores, axis=1)[prices.columns].ffill()
    raw_momentum = pd.concat(all_momentum, axis=1)[prices.columns].ffill()
    return momentum_score, raw_momentum


# ---------------------------------------------------------------------------
# cluster — scoring within time-varying statistical clusters
# ---------------------------------------------------------------------------
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
