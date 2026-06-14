"""
Low-beta alpha computation (standard + cluster scoring).

Computes cross-sectional low-beta scores from rolling EWMA beta to a
benchmark. Assets with lower beta receive higher scores ("betting against
beta").

Two public entry points share the same raw beta:
    * ``compute_low_beta_alpha`` — scores cross-sectionally, optionally
      within fixed user-defined groups (``group_data``).
    * ``compute_low_beta_cluster_alpha`` — scores within time-varying
      statistical clusters (``rolling_clusters``) rather than fixed groups.

Pipeline:
    returns → EWMA regression beta → negate → cross-sectional score
            (global / within groups / within clusters)
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
def _compute_raw_beta_single_freq(prices: pd.DataFrame,
                                  benchmark_price: pd.Series = None,
                                  returns_freq: str = 'ME',
                                  beta_span: int = 12,
                                  mean_adj_type: qis.MeanAdjType = qis.MeanAdjType.EWMA
                                  ) -> pd.DataFrame:
    """Compute raw EWMA beta to benchmark (before negation and scoring)."""
    returns = qis.to_returns(prices, freq=returns_freq, is_log_returns=True)

    if benchmark_price is None:
        benchmark_returns = pd.Series(np.nanmean(returns.to_numpy(), axis=1), index=returns.index)
    else:
        benchmark_price = benchmark_price.reindex(index=prices.index, method='ffill')
        benchmark_returns = qis.to_returns(benchmark_price, freq=returns_freq, is_log_returns=True)

    ewm_linear_model = qis.EwmLinearModel(x=benchmark_returns.to_frame('benchmark'), y=returns)
    ewm_linear_model.fit(span=beta_span, mean_adj_type=mean_adj_type, is_x_correlated=True, warmup_period=beta_span)
    raw_beta = ewm_linear_model.loadings['benchmark']
    raw_beta = raw_beta.replace({0.0: np.nan})

    return raw_beta


def _compute_raw_beta_mixed_freq(prices: pd.DataFrame,
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


# ---------------------------------------------------------------------------
# standard — cross-sectional scoring (optionally within fixed groups)
# ---------------------------------------------------------------------------
def compute_low_beta_alpha(prices: pd.DataFrame,
                           benchmark_price: pd.Series = None,
                           returns_freq: Union[str, pd.Series] = 'ME',
                           group_data: Optional[pd.Series] = None,
                           beta_span: int = 12,
                           mean_adj_type: qis.MeanAdjType = qis.MeanAdjType.EWMA
                           ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute cross-sectional low-beta alpha scores.

    Estimates EWMA beta to the benchmark, then converts negated beta
    to a cross-sectional score (low beta → high score).

    Handles both single-frequency and mixed-frequency universes.

    Args:
        prices: Asset price panel. Index=dates, columns=tickers.
        benchmark_price: Benchmark price series. If None, uses equal-weight.
        returns_freq: Return frequency. String or pd.Series.
        group_data: Optional group labels for within-group scoring.
        beta_span: EWMA span for beta estimation.
        mean_adj_type: Mean adjustment type for EWMA regression.

    Returns:
        Tuple of (beta_score, raw_beta).
    """
    if isinstance(returns_freq, pd.Series):
        return _compute_low_beta_alpha_mixed_freq(
            prices=prices, benchmark_price=benchmark_price,
            returns_freqs=returns_freq, group_data=group_data,
            beta_span=beta_span, mean_adj_type=mean_adj_type)
    else:
        return _compute_low_beta_alpha_single_freq(
            prices=prices, benchmark_price=benchmark_price,
            returns_freq=returns_freq, group_data=group_data,
            beta_span=beta_span, mean_adj_type=mean_adj_type)


def _compute_low_beta_alpha_single_freq(prices: pd.DataFrame,
                                        benchmark_price: pd.Series = None,
                                        returns_freq: str = 'ME',
                                        group_data: Optional[pd.Series] = None,
                                        beta_span: int = 12,
                                        mean_adj_type: qis.MeanAdjType = qis.MeanAdjType.EWMA
                                        ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Single-frequency low-beta: raw beta, then negated cross-sectional scoring."""
    raw_beta = _compute_raw_beta_single_freq(
        prices=prices, benchmark_price=benchmark_price,
        returns_freq=returns_freq, beta_span=beta_span, mean_adj_type=mean_adj_type)

    if group_data is not None:
        grouped_prices = qis.split_df_by_groups(df=prices, group_data=group_data)
        group_scores = []
        for group, gprice in grouped_prices.items():
            group_scores.append(qis.df_to_cross_sectional_score(df=-1.0 * raw_beta[gprice.columns]))
        beta_score = pd.concat(group_scores, axis=1)[prices.columns]
    else:
        beta_score = qis.df_to_cross_sectional_score(df=-1.0 * raw_beta)

    return beta_score, raw_beta


def _compute_low_beta_alpha_mixed_freq(prices: pd.DataFrame,
                                       benchmark_price: pd.Series = None,
                                       returns_freqs: pd.Series = None,
                                       group_data: Optional[pd.Series] = None,
                                       beta_span: int = 12,
                                       mean_adj_type: qis.MeanAdjType = qis.MeanAdjType.EWMA
                                       ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Mixed-frequency low-beta: compute per (frequency × group), merge."""
    returns_freqs = returns_freqs[prices.columns]
    group_freqs = qis.get_group_dict(group_data=returns_freqs)

    all_scores = []
    all_betas = []
    for freq, asset_tickers in group_freqs.items():
        freq_prices = prices[asset_tickers]
        if group_data is not None:
            freq_group_data = group_data.loc[group_data.index.intersection(asset_tickers)]
            grouped_prices = qis.split_df_by_groups(df=freq_prices, group_data=freq_group_data)
        else:
            grouped_prices = {'_': freq_prices}

        for group, gprice in grouped_prices.items():
            score, beta = _compute_low_beta_alpha_single_freq(
                prices=gprice, benchmark_price=benchmark_price,
                returns_freq=freq, group_data=None,
                beta_span=beta_span, mean_adj_type=mean_adj_type)
            all_scores.append(score)
            all_betas.append(beta)

    beta_score = pd.concat(all_scores, axis=1)[prices.columns].ffill()
    raw_beta = pd.concat(all_betas, axis=1)[prices.columns].ffill()
    return beta_score, raw_beta


# ---------------------------------------------------------------------------
# cluster — scoring within time-varying statistical clusters
# ---------------------------------------------------------------------------
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
