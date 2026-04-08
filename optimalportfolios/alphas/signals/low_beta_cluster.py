"""
Cluster-based low-beta alpha computation.

Computes low-beta scores (same as low_beta.py) but applies
cross-sectional scoring within time-varying statistical clusters
rather than fixed user-defined groups.

Pipeline:
    returns → EWMA regression beta → negate → score within time-varying clusters
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import qis as qis
from typing import Optional, Tuple, Union, Dict

from optimalportfolios.alphas.signals.utils import score_within_clusters


def compute_low_beta_cluster_alpha(
        prices: pd.DataFrame,
        benchmark_price: pd.Series = None,
        rolling_clusters: Dict[pd.Timestamp, pd.Series] = None,
        returns_freq: Union[str, pd.Series] = 'ME',
        beta_span: int = 12,
        mean_adj_type: qis.MeanAdjType = qis.MeanAdjType.EWMA,
        min_cluster_size: int = 3,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute low-beta scores with time-varying cluster-based scoring.

    Identical to compute_low_beta_alpha for the raw beta estimation,
    but applies cross-sectional scoring within statistical clusters
    that evolve over time.

    Args:
        prices: Asset price panel. Index=dates, columns=tickers.
        benchmark_price: Benchmark price series. If None, uses equal-weight.
        rolling_clusters: Dict mapping dates to pd.Series (ticker → cluster_id).
            Extracted from RollingFactorCovarData via extract_rolling_clusters().
        returns_freq: Return frequency. String or pd.Series for mixed-freq.
        beta_span: EWMA span for beta estimation.
        mean_adj_type: Mean adjustment type for EWMA regression.
        min_cluster_size: Minimum cluster size for within-cluster scoring.
            Clusters with size <= min_cluster_size use global statistics.

    Returns:
        Tuple of (beta_cluster_score, raw_beta).
    """
    if rolling_clusters is None:
        rolling_clusters = {}

    if isinstance(returns_freq, pd.Series):
        raw_beta = _compute_raw_beta_mixed_freq(
            prices=prices, benchmark_price=benchmark_price,
            returns_freqs=returns_freq, beta_span=beta_span,
            mean_adj_type=mean_adj_type)
    else:
        raw_beta = _compute_raw_beta_single_freq(
            prices=prices, benchmark_price=benchmark_price,
            returns_freq=returns_freq, beta_span=beta_span,
            mean_adj_type=mean_adj_type)

    # negate: low beta → high score
    negated_beta = -1.0 * raw_beta

    # score within time-varying clusters
    beta_cluster_score = score_within_clusters(
        raw_signal=negated_beta,
        rolling_clusters=rolling_clusters,
        min_cluster_size=min_cluster_size)

    return beta_cluster_score, raw_beta


def _compute_raw_beta_single_freq(
        prices: pd.DataFrame,
        benchmark_price: pd.Series = None,
        returns_freq: str = 'ME',
        beta_span: int = 12,
        mean_adj_type: qis.MeanAdjType = qis.MeanAdjType.EWMA
) -> pd.DataFrame:
    """Compute raw EWMA beta to benchmark (before negation and scoring)."""
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
    raw_beta = raw_beta.replace({0.0: np.nan})

    return raw_beta


def _compute_raw_beta_mixed_freq(
        prices: pd.DataFrame,
        benchmark_price: pd.Series = None,
        returns_freqs: pd.Series = None,
        beta_span: int = 12,
        mean_adj_type: qis.MeanAdjType = qis.MeanAdjType.EWMA
) -> pd.DataFrame:
    """Mixed-frequency: compute raw beta per frequency group, merge."""
    returns_freqs = returns_freqs[prices.columns]
    group_freqs = qis.get_group_dict(group_data=returns_freqs)

    all_betas = []
    for freq, asset_tickers in group_freqs.items():
        freq_prices = prices[asset_tickers]
        raw_beta = _compute_raw_beta_single_freq(
            prices=freq_prices, benchmark_price=benchmark_price,
            returns_freq=freq, beta_span=beta_span,
            mean_adj_type=mean_adj_type)
        all_betas.append(raw_beta)

    return pd.concat(all_betas, axis=1)[prices.columns].ffill()