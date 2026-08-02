"""
Residual momentum alpha computation (standard + cluster scoring).

Computes cross-sectional residual momentum scores by stripping out
benchmark beta exposure from asset returns, then filtering with EWMA
long/short risk-adjusted returns (same filter as momentum.py).

Two public entry points share the same raw signal:
    * ``compute_residual_momentum_alpha`` — scores cross-sectionally,
      optionally within fixed user-defined groups (``group_data``).
    * ``compute_residual_momentum_cluster_alpha`` — scores within
      time-varying statistical clusters (``rolling_clusters``).

Pipeline:
    returns → EWMA beta to benchmark (lagged) → residual = r_t - β̂_{t-1} · r_bench_t
            → EWMA long/short filtered RA returns
            → cross-sectional score (global / within groups / within clusters)

References:
    Blitz D., Huij J., Martens M. (2011),
    "Residual Momentum", Journal of Empirical Finance, 18, 506-521.

    Sepp A., Ossa I., Kastenholz M. (2026),
    "Robust Optimization of Strategic and Tactical Asset Allocation
    for Multi-Asset Portfolios", JPM, 52(4), 86-120.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import qis as qis
from typing import Dict, Mapping, Optional, Tuple, Union

from optimalportfolios.alphas.signals.utils import (
    resolve_span,
    extract_rolling_clusters,  # re-exported for back-compat (used to live in residual_momentum_cluster)
    score_within_clusters,
)

# keep ``extract_rolling_clusters`` importable from this module: it used to be
# re-exported from residual_momentum_cluster (now folded in here), and the
# canonical home is utils. Guards external callers that did
# ``from ...residual_momentum import extract_rolling_clusters``.
__all__ = [
    'compute_residual_momentum_alpha',
    'compute_residual_momentum_cluster_alpha',
    'extract_rolling_clusters',
]


# ---------------------------------------------------------------------------
# raw signal — shared by the standard and cluster entry points
# ---------------------------------------------------------------------------
def _compute_raw_residual_momentum_single_freq(
        prices: pd.DataFrame,
        benchmark_price: pd.Series = None,
        returns_freq: str = 'ME',
        beta_span: int = 12,
        long_span: int = 12,
        short_span: Optional[int] = None,
        vol_span: Optional[int] = 13,
        mean_adj_type: qis.MeanAdjType = qis.MeanAdjType.EWMA
) -> pd.DataFrame:
    """Compute raw EWMA long/short filtered RA residual returns (before scoring)."""
    returns = qis.to_returns(prices, freq=returns_freq, is_log_returns=True)

    if benchmark_price is None:
        benchmark_returns = pd.Series(
            np.nanmean(returns.to_numpy(), axis=1), index=returns.index)
    else:
        benchmark_price = benchmark_price.reindex(index=prices.index, method='ffill')
        benchmark_returns = qis.to_returns(
            benchmark_price, freq=returns_freq, is_log_returns=True)

    # EWMA beta estimation
    ewm_linear_model = qis.EwmLinearModel(
        x=benchmark_returns.to_frame('benchmark'), y=returns)
    ewm_linear_model.fit(
        span=beta_span, mean_adj_type=mean_adj_type, is_x_correlated=True, warmup_period=beta_span)
    raw_beta = ewm_linear_model.loadings['benchmark']

    # residual = r_t - beta_{t-1} * r_bench_t  (lagged beta avoids look-ahead)
    lagged_beta = raw_beta.shift(1)
    benchmark_component = lagged_beta.multiply(benchmark_returns, axis=0)
    residuals = returns - benchmark_component

    # EWMA long/short filtered risk-adjusted returns on residuals
    raw_residual_momentum = qis.compute_ewm_long_short_filtered_ra_returns(
        returns=residuals, vol_span=vol_span, long_span=long_span,
        short_span=short_span, weight_lag=0, mean_adj_type=qis.MeanAdjType.NONE,
        warmup_period=long_span)

    return raw_residual_momentum


def _compute_raw_residual_momentum_mixed_freq(
        prices: pd.DataFrame,
        benchmark_price: pd.Series = None,
        returns_freqs: pd.Series = None,
        beta_span: Union[int, Mapping[str, int]] = 12,
        long_span: Union[int, Mapping[str, int]] = 12,
        short_span: Optional[Union[int, Mapping[str, int]]] = None,
        vol_span: Optional[Union[int, Mapping[str, int]]] = 13,
        mean_adj_type: qis.MeanAdjType = qis.MeanAdjType.EWMA
) -> pd.DataFrame:
    """Mixed-frequency: compute raw residuals per frequency group, merge."""
    returns_freqs = returns_freqs[prices.columns]
    group_freqs = qis.get_group_dict(group_data=returns_freqs)

    all_raw = []
    for freq, asset_tickers in group_freqs.items():
        freq_prices = prices[asset_tickers]
        # one horizon per cadence, resolved where the bucket is chosen;
        # _compute_raw_residual_momentum_single_freq takes resolved ints
        bucket_beta_span = resolve_span(beta_span, freq=freq, name='beta_span')
        bucket_long_span = resolve_span(long_span, freq=freq, name='long_span')
        bucket_short_span = resolve_span(short_span, freq=freq, name='short_span')
        bucket_vol_span = resolve_span(vol_span, freq=freq, name='vol_span')
        raw = _compute_raw_residual_momentum_single_freq(
            prices=freq_prices, benchmark_price=benchmark_price,
            returns_freq=freq, beta_span=bucket_beta_span,
            long_span=bucket_long_span, short_span=bucket_short_span,
            vol_span=bucket_vol_span, mean_adj_type=mean_adj_type)
        all_raw.append(raw)

    return pd.concat(all_raw, axis=1, sort=True)[prices.columns].ffill()


# ---------------------------------------------------------------------------
# standard — cross-sectional scoring (optionally within fixed groups)
# ---------------------------------------------------------------------------
def compute_residual_momentum_alpha(
        prices: pd.DataFrame,
        benchmark_price: pd.Series = None,
        returns_freq: Union[str, pd.Series] = 'ME',
        group_data: Optional[pd.Series] = None,
        beta_span: Union[int, Mapping[str, int]] = 12,
        long_span: Union[int, Mapping[str, int]] = 12,
        short_span: Optional[Union[int, Mapping[str, int]]] = None,
        vol_span: Optional[Union[int, Mapping[str, int]]] = 13,
        mean_adj_type: qis.MeanAdjType = qis.MeanAdjType.EWMA
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute cross-sectional residual momentum alpha scores.

    For each asset, estimates EWMA beta to the benchmark, computes the
    residual return (r_t - β̂_{t-1} · r_bench_t), applies EWMA long/short
    filtered risk-adjusted returns, and converts to a cross-sectional score.

    Uses the same EWMA filtering as compute_momentum_alpha but applied
    to beta-stripped residuals rather than total or excess returns.

    Args:
        prices: Asset price panel. Index=dates, columns=tickers.
        benchmark_price: Benchmark price series. If None, uses equal-weight.
        returns_freq: Return frequency. String or pd.Series.
        group_data: Optional group labels for within-group scoring.
        beta_span: EWMA span for benchmark beta estimation.
            Either a scalar applied at every reporting cadence, or a per-cadence mapping
            such as ``{'ME': 12, 'QE': 4}`` giving each cadence the same calendar horizon.
        long_span: EWMA span for the long momentum signal.
            Either a scalar applied at every reporting cadence, or a per-cadence mapping
            such as ``{'ME': 12, 'QE': 4}`` giving each cadence the same calendar horizon.
        short_span: Optional EWMA span for short-term reversal subtraction.
            Either a scalar applied at every reporting cadence, or a per-cadence mapping
            such as ``{'ME': 12, 'QE': 4}`` giving each cadence the same calendar horizon.
        vol_span: EWMA span for volatility normalisation. None disables.
            Either a scalar applied at every reporting cadence, or a per-cadence mapping
            such as ``{'ME': 13, 'QE': 4}`` giving each cadence the same calendar horizon.
        mean_adj_type: Mean adjustment type for EWMA beta regression.

    Returns:
        Tuple of (residual_momentum_score, raw_residual_momentum).
    """
    if isinstance(returns_freq, pd.Series):
        return _compute_residual_momentum_mixed_freq(
            prices=prices, benchmark_price=benchmark_price,
            returns_freqs=returns_freq, group_data=group_data,
            beta_span=beta_span, long_span=long_span,
            short_span=short_span, vol_span=vol_span,
            mean_adj_type=mean_adj_type)
    else:
        return _compute_residual_momentum_single_freq(
            prices=prices, benchmark_price=benchmark_price,
            returns_freq=returns_freq, group_data=group_data,
            beta_span=beta_span, long_span=long_span,
            short_span=short_span, vol_span=vol_span,
            mean_adj_type=mean_adj_type)


def _compute_residual_momentum_single_freq(
        prices: pd.DataFrame,
        benchmark_price: pd.Series = None,
        returns_freq: str = 'ME',
        group_data: Optional[pd.Series] = None,
        beta_span: Union[int, Mapping[str, int]] = 12,
        long_span: Union[int, Mapping[str, int]] = 12,
        short_span: Optional[Union[int, Mapping[str, int]]] = None,
        vol_span: Optional[Union[int, Mapping[str, int]]] = 13,
        mean_adj_type: qis.MeanAdjType = qis.MeanAdjType.EWMA
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Single-frequency residual momentum: raw signal, then cross-sectional scoring."""
    # one horizon per cadence, resolved where the bucket is chosen;
    # _compute_raw_residual_momentum_single_freq takes resolved ints
    beta_span = resolve_span(beta_span, freq=returns_freq, name='beta_span')
    long_span = resolve_span(long_span, freq=returns_freq, name='long_span')
    short_span = resolve_span(short_span, freq=returns_freq, name='short_span')
    vol_span = resolve_span(vol_span, freq=returns_freq, name='vol_span')
    raw_residual_momentum = _compute_raw_residual_momentum_single_freq(
        prices=prices, benchmark_price=benchmark_price,
        returns_freq=returns_freq, beta_span=beta_span,
        long_span=long_span, short_span=short_span,
        vol_span=vol_span, mean_adj_type=mean_adj_type)

    # cross-sectional scoring (within-group if specified)
    if group_data is not None:
        grouped_prices = qis.split_df_by_groups(df=prices, group_data=group_data)
        group_scores = []
        for group, gprice in grouped_prices.items():
            group_cols = [c for c in gprice.columns if c in raw_residual_momentum.columns]
            group_scores.append(
                qis.df_to_cross_sectional_score(df=raw_residual_momentum[group_cols]))
        residual_momentum_score = pd.concat(group_scores, axis=1, sort=True)[prices.columns]
    else:
        residual_momentum_score = qis.df_to_cross_sectional_score(
            df=raw_residual_momentum)

    return residual_momentum_score, raw_residual_momentum


def _compute_residual_momentum_mixed_freq(
        prices: pd.DataFrame,
        benchmark_price: pd.Series = None,
        returns_freqs: pd.Series = None,
        group_data: Optional[pd.Series] = None,
        beta_span: Union[int, Mapping[str, int]] = 12,
        long_span: Union[int, Mapping[str, int]] = 12,
        short_span: Optional[Union[int, Mapping[str, int]]] = None,
        vol_span: Optional[Union[int, Mapping[str, int]]] = 13,
        mean_adj_type: qis.MeanAdjType = qis.MeanAdjType.EWMA
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Mixed-frequency residual momentum: compute per (frequency × group), merge."""
    returns_freqs = returns_freqs[prices.columns]
    group_freqs = qis.get_group_dict(group_data=returns_freqs)

    all_scores = []
    all_raw = []
    for freq, asset_tickers in group_freqs.items():
        freq_prices = prices[asset_tickers]
        if group_data is not None:
            freq_group_data = group_data.loc[
                group_data.index.intersection(asset_tickers)]
            grouped_prices = qis.split_df_by_groups(
                df=freq_prices, group_data=freq_group_data)
        else:
            grouped_prices = {'_': freq_prices}

        for group, gprice in grouped_prices.items():
            score, raw = _compute_residual_momentum_single_freq(
                prices=gprice, benchmark_price=benchmark_price,
                returns_freq=freq, group_data=None,
                beta_span=beta_span, long_span=long_span,
                short_span=short_span, vol_span=vol_span,
                mean_adj_type=mean_adj_type)
            all_scores.append(score)
            all_raw.append(raw)

    residual_momentum_score = pd.concat(all_scores, axis=1, sort=True)[prices.columns].ffill()
    raw_residual_momentum = pd.concat(all_raw, axis=1, sort=True)[prices.columns].ffill()
    return residual_momentum_score, raw_residual_momentum


# ---------------------------------------------------------------------------
# cluster — scoring within time-varying statistical clusters
# ---------------------------------------------------------------------------
def compute_residual_momentum_cluster_alpha(
        prices: pd.DataFrame,
        benchmark_price: pd.Series = None,
        rolling_clusters: Dict[pd.Timestamp, pd.Series] = None,
        returns_freq: Union[str, pd.Series] = 'ME',
        beta_span: Union[int, Mapping[str, int]] = 12,
        long_span: Union[int, Mapping[str, int]] = 12,
        short_span: Optional[Union[int, Mapping[str, int]]] = None,
        vol_span: Optional[Union[int, Mapping[str, int]]] = 13,
        mean_adj_type: qis.MeanAdjType = qis.MeanAdjType.EWMA,
        min_cluster_size: int = 3,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute residual momentum with time-varying cluster-based scoring.

    Identical to compute_residual_momentum_alpha for the raw signal
    computation (EWMA beta → lagged residual → EWMA long/short filtered
    RA returns), but applies cross-sectional scoring within statistical
    clusters that evolve over time.

    Args:
        prices: Asset price panel. Index=dates, columns=tickers.
        benchmark_price: Benchmark price series. If None, uses equal-weight.
        rolling_clusters: Dict mapping dates to pd.Series (ticker → cluster_id).
        returns_freq: Return frequency. String or pd.Series for mixed-freq.
        beta_span: EWMA span for benchmark beta estimation.
            Either a scalar applied at every reporting cadence, or a per-cadence mapping
            such as ``{'ME': 12, 'QE': 4}`` giving each cadence the same calendar horizon.
        long_span: EWMA span for the long momentum signal.
            Either a scalar applied at every reporting cadence, or a per-cadence mapping
            such as ``{'ME': 12, 'QE': 4}`` giving each cadence the same calendar horizon.
        short_span: Optional EWMA span for short-term reversal subtraction.
            Either a scalar applied at every reporting cadence, or a per-cadence mapping
            such as ``{'ME': 12, 'QE': 4}`` giving each cadence the same calendar horizon.
        vol_span: EWMA span for volatility normalisation. None disables.
            Either a scalar applied at every reporting cadence, or a per-cadence mapping
            such as ``{'ME': 13, 'QE': 4}`` giving each cadence the same calendar horizon.
        mean_adj_type: Mean adjustment type for EWMA beta regression.
        min_cluster_size: Minimum cluster size for within-cluster scoring.
            Clusters with size <= min_cluster_size use global statistics.

    Returns:
        Tuple of (residual_momentum_cluster_score, raw_residual_momentum).
    """
    if rolling_clusters is None:
        rolling_clusters = {}

    if isinstance(returns_freq, pd.Series):
        raw_residual_momentum = _compute_raw_residual_momentum_mixed_freq(
            prices=prices, benchmark_price=benchmark_price,
            returns_freqs=returns_freq, beta_span=beta_span,
            long_span=long_span, short_span=short_span,
            vol_span=vol_span, mean_adj_type=mean_adj_type)
    else:
        # one horizon per cadence, resolved where the bucket is chosen;
        # _compute_raw_residual_momentum_single_freq takes resolved ints
        beta_span = resolve_span(beta_span, freq=returns_freq, name='beta_span')
        long_span = resolve_span(long_span, freq=returns_freq, name='long_span')
        short_span = resolve_span(short_span, freq=returns_freq, name='short_span')
        vol_span = resolve_span(vol_span, freq=returns_freq, name='vol_span')
        raw_residual_momentum = _compute_raw_residual_momentum_single_freq(
            prices=prices, benchmark_price=benchmark_price,
            returns_freq=returns_freq, beta_span=beta_span,
            long_span=long_span, short_span=short_span,
            vol_span=vol_span, mean_adj_type=mean_adj_type)

    # score within time-varying clusters
    residual_momentum_score = score_within_clusters(
        raw_signal=raw_residual_momentum,
        rolling_clusters=rolling_clusters,
        min_cluster_size=min_cluster_size)

    return residual_momentum_score, raw_residual_momentum
