"""
Cluster-based residual momentum alpha computation.

Computes residual momentum (same as residual_momentum.py) but applies
cross-sectional scoring within time-varying statistical clusters rather
than fixed user-defined groups.

Clusters are extracted from the HCGL/LASSO covariance estimator
(hierarchical clustering on the factor correlation matrix) and change
at each estimation date as the covariance structure evolves.

Pipeline:
    returns → EWMA beta to benchmark (lagged) → residual = r_t - β̂_{t-1} · r_bench_t
            → EWMA smooth → score within time-varying clusters

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
from typing import Optional, Tuple, Union, Dict, List

from optimalportfolios.alphas.signals.utils import score_within_clusters


def extract_rolling_clusters(
        rolling_covar_data,
        assets: List[str] = None,
) -> Dict[pd.Timestamp, pd.Series]:
    """Extract time-varying cluster assignments from RollingFactorCovarData.

    Merges per-frequency cluster dicts into a single pd.Series per date,
    filtered to the requested asset universe.

    Args:
        rolling_covar_data: RollingFactorCovarData from FactorCovarEstimator.
        assets: Asset tickers to include. If None, includes all.

    Returns:
        Dict mapping estimation dates to pd.Series (ticker → cluster_id).
        Dates where clusters are None or empty are skipped.
    """
    rolling_clusters = {}
    for date, current_data in rolling_covar_data.data.items():
        if current_data.clusters is None:
            continue
        # clusters is Dict[str, pd.Series] keyed by frequency
        # merge all frequencies into one Series
        parts = [s for s in current_data.clusters.values() if s is not None and len(s) > 0]
        if not parts:
            continue
        merged = pd.concat(parts)
        merged = merged[~merged.index.duplicated(keep='last')]
        if assets is not None:
            merged = merged.reindex(assets).dropna()
        if len(merged) > 0:
            rolling_clusters[date] = merged
    return rolling_clusters


def compute_residual_momentum_cluster_alpha(
        prices: pd.DataFrame,
        benchmark_price: pd.Series = None,
        rolling_clusters: Dict[pd.Timestamp, pd.Series] = None,
        returns_freq: Union[str, pd.Series] = 'ME',
        beta_span: int = 12,
        long_span: int = 12,
        short_span: Optional[int] = None,
        vol_span: Optional[int] = 13,
        mean_adj_type: qis.MeanAdjType = qis.MeanAdjType.EWMA
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
        long_span: EWMA span for the long momentum signal.
        short_span: Optional EWMA span for short-term reversal subtraction.
        vol_span: EWMA span for volatility normalisation. None disables.
        mean_adj_type: Mean adjustment type for EWMA beta regression.

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
        raw_residual_momentum = _compute_raw_residual_momentum_single_freq(
            prices=prices, benchmark_price=benchmark_price,
            returns_freq=returns_freq, beta_span=beta_span,
            long_span=long_span, short_span=short_span,
            vol_span=vol_span, mean_adj_type=mean_adj_type)

    # score within time-varying clusters
    residual_momentum_score = score_within_clusters(
        raw_signal=raw_residual_momentum,
        rolling_clusters=rolling_clusters)

    return residual_momentum_score, raw_residual_momentum


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

    ewm_linear_model = qis.EwmLinearModel(
        x=benchmark_returns.to_frame('benchmark'), y=returns)
    ewm_linear_model.fit(
        span=beta_span, mean_adj_type=mean_adj_type, is_x_correlated=True, warmup_period=beta_span)
    raw_beta = ewm_linear_model.loadings['benchmark']

    lagged_beta = raw_beta.shift(1)
    benchmark_component = lagged_beta.multiply(benchmark_returns, axis=0)
    residuals = returns - benchmark_component

    raw_residual_momentum = qis.compute_ewm_long_short_filtered_ra_returns(
        returns=residuals, vol_span=vol_span, long_span=long_span,
        short_span=short_span, weight_lag=0, mean_adj_type=qis.MeanAdjType.NONE,
        warmup_period=long_span)

    return raw_residual_momentum


def _compute_raw_residual_momentum_mixed_freq(
        prices: pd.DataFrame,
        benchmark_price: pd.Series = None,
        returns_freqs: pd.Series = None,
        beta_span: int = 12,
        long_span: int = 12,
        short_span: Optional[int] = None,
        vol_span: Optional[int] = 13,
        mean_adj_type: qis.MeanAdjType = qis.MeanAdjType.EWMA
) -> pd.DataFrame:
    """Mixed-frequency: compute raw residuals per frequency group, merge."""
    returns_freqs = returns_freqs[prices.columns]
    group_freqs = qis.get_group_dict(group_data=returns_freqs)

    all_raw = []
    for freq, asset_tickers in group_freqs.items():
        freq_prices = prices[asset_tickers]
        raw = _compute_raw_residual_momentum_single_freq(
            prices=freq_prices, benchmark_price=benchmark_price,
            returns_freq=freq, beta_span=beta_span,
            long_span=long_span, short_span=short_span,
            vol_span=vol_span, mean_adj_type=mean_adj_type)
        all_raw.append(raw)

    return pd.concat(all_raw, axis=1)[prices.columns].ffill()