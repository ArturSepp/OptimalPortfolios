"""
Residual reversal alpha computation (standard + cluster scoring).

Computes cross-sectional residual reversal scores by stripping out
benchmark beta exposure from asset returns, then filtering with EWMA
long/short risk-adjusted returns (same filter as residual_momentum.py)
and NEGATING the signal: recent residual losers receive higher scores,
recent residual winners lower scores ("short-term residual reversal").

This is the sign-flipped sibling of compute_residual_momentum_alpha.
Stripping the benchmark beta isolates the asset-specific component and
removes the dynamic benchmark exposure that contaminates conventional
(total-return) reversal.

Two public entry points share the same raw signal:
    * ``compute_residual_reversal_alpha`` — scores cross-sectionally,
      optionally within fixed user-defined groups (``group_data``).
    * ``compute_residual_reversal_cluster_alpha`` — scores within
      time-varying statistical clusters (``rolling_clusters``).

Pipeline:
    returns → EWMA beta to benchmark (lagged) → residual = r_t - β̂_{t-1} · r_bench_t
            → EWMA long/short filtered RA returns → NEGATE
            → cross-sectional score (global / within groups / within clusters)

References:
    Blitz D., Huij J., Lansdorp S., Verbeek M. (2013),
    "Short-Term Residual Reversal", Journal of Financial Markets, 16, 477-504.
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
def _compute_raw_residual_reversal_single_freq(
        prices: pd.DataFrame,
        benchmark_price: pd.Series = None,
        returns_freq: str = 'ME',
        beta_span: int = 12,
        long_span: int = 1,
        short_span: Optional[int] = None,
        vol_span: Optional[int] = 13,
        mean_adj_type: qis.MeanAdjType = qis.MeanAdjType.EWMA
) -> pd.DataFrame:
    """Compute raw NEGATED EWMA long/short filtered RA residual returns (before scoring)."""
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

    # EWMA long/short filtered risk-adjusted residual returns, NEGATED -> reversal
    raw_residual_reversal = -1.0 * qis.compute_ewm_long_short_filtered_ra_returns(
        returns=residuals, vol_span=vol_span, long_span=long_span,
        short_span=short_span, weight_lag=0, mean_adj_type=qis.MeanAdjType.NONE,
        warmup_period=long_span)

    return raw_residual_reversal


def _compute_raw_residual_reversal_mixed_freq(
        prices: pd.DataFrame,
        benchmark_price: pd.Series = None,
        returns_freqs: pd.Series = None,
        beta_span: int = 12,
        long_span: int = 1,
        short_span: Optional[int] = None,
        vol_span: Optional[int] = 13,
        mean_adj_type: qis.MeanAdjType = qis.MeanAdjType.EWMA
) -> pd.DataFrame:
    """Mixed-frequency: compute raw (negated) residuals per frequency group, merge."""
    returns_freqs = returns_freqs[prices.columns]
    group_freqs = qis.get_group_dict(group_data=returns_freqs)

    all_raw = []
    for freq, asset_tickers in group_freqs.items():
        freq_prices = prices[asset_tickers]
        raw = _compute_raw_residual_reversal_single_freq(
            prices=freq_prices, benchmark_price=benchmark_price,
            returns_freq=freq, beta_span=beta_span,
            long_span=long_span, short_span=short_span,
            vol_span=vol_span, mean_adj_type=mean_adj_type)
        all_raw.append(raw)

    return pd.concat(all_raw, axis=1)[prices.columns].ffill()


# ---------------------------------------------------------------------------
# standard — cross-sectional scoring (optionally within fixed groups)
# ---------------------------------------------------------------------------
def compute_residual_reversal_alpha(
        prices: pd.DataFrame,
        benchmark_price: pd.Series = None,
        returns_freq: Union[str, pd.Series] = 'ME',
        group_data: Optional[pd.Series] = None,
        beta_span: int = 12,
        long_span: int = 1,
        short_span: Optional[int] = None,
        vol_span: Optional[int] = 13,
        mean_adj_type: qis.MeanAdjType = qis.MeanAdjType.EWMA
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute cross-sectional residual reversal alpha scores.

    For each asset, estimates EWMA beta to the benchmark, computes the
    residual return (r_t - β̂_{t-1} · r_bench_t), applies EWMA long/short
    filtered risk-adjusted returns, and converts the NEGATED signal to a
    cross-sectional score (recent residual loser → high score → overweight).

    Uses the same beta-stripping and EWMA filtering as
    compute_residual_momentum_alpha; the only difference is the sign flip
    that turns continuation into reversal. ``long_span`` is the reversal
    lookback and is typically short.

    Args:
        prices: Asset price panel. Index=dates, columns=tickers.
        benchmark_price: Benchmark price series. If None, uses equal-weight.
        returns_freq: Return frequency. String or pd.Series.
        group_data: Optional group labels for within-group scoring.
        beta_span: EWMA span for benchmark beta estimation.
        long_span: EWMA span for the residual reversal lookback (short).
        short_span: Optional EWMA span for short-term subtraction.
        vol_span: EWMA span for volatility normalisation. None disables.
        mean_adj_type: Mean adjustment type for EWMA beta regression.

    Returns:
        Tuple of (residual_reversal_score, raw_residual_reversal).
    """
    if isinstance(returns_freq, pd.Series):
        return _compute_residual_reversal_mixed_freq(
            prices=prices, benchmark_price=benchmark_price,
            returns_freqs=returns_freq, group_data=group_data,
            beta_span=beta_span, long_span=long_span,
            short_span=short_span, vol_span=vol_span,
            mean_adj_type=mean_adj_type)
    else:
        return _compute_residual_reversal_single_freq(
            prices=prices, benchmark_price=benchmark_price,
            returns_freq=returns_freq, group_data=group_data,
            beta_span=beta_span, long_span=long_span,
            short_span=short_span, vol_span=vol_span,
            mean_adj_type=mean_adj_type)


def _compute_residual_reversal_single_freq(
        prices: pd.DataFrame,
        benchmark_price: pd.Series = None,
        returns_freq: str = 'ME',
        group_data: Optional[pd.Series] = None,
        beta_span: int = 12,
        long_span: int = 1,
        short_span: Optional[int] = None,
        vol_span: Optional[int] = 13,
        mean_adj_type: qis.MeanAdjType = qis.MeanAdjType.EWMA
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Single-frequency residual reversal: raw signal, then cross-sectional scoring."""
    raw_residual_reversal = _compute_raw_residual_reversal_single_freq(
        prices=prices, benchmark_price=benchmark_price,
        returns_freq=returns_freq, beta_span=beta_span,
        long_span=long_span, short_span=short_span,
        vol_span=vol_span, mean_adj_type=mean_adj_type)

    # cross-sectional scoring (within-group if specified)
    if group_data is not None:
        grouped_prices = qis.split_df_by_groups(df=prices, group_data=group_data)
        group_scores = []
        for group, gprice in grouped_prices.items():
            group_cols = [c for c in gprice.columns if c in raw_residual_reversal.columns]
            group_scores.append(
                qis.df_to_cross_sectional_score(df=raw_residual_reversal[group_cols]))
        residual_reversal_score = pd.concat(group_scores, axis=1)[prices.columns]
    else:
        residual_reversal_score = qis.df_to_cross_sectional_score(
            df=raw_residual_reversal)

    return residual_reversal_score, raw_residual_reversal


def _compute_residual_reversal_mixed_freq(
        prices: pd.DataFrame,
        benchmark_price: pd.Series = None,
        returns_freqs: pd.Series = None,
        group_data: Optional[pd.Series] = None,
        beta_span: int = 12,
        long_span: int = 1,
        short_span: Optional[int] = None,
        vol_span: Optional[int] = 13,
        mean_adj_type: qis.MeanAdjType = qis.MeanAdjType.EWMA
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Mixed-frequency residual reversal: compute per (frequency × group), merge."""
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
            score, raw = _compute_residual_reversal_single_freq(
                prices=gprice, benchmark_price=benchmark_price,
                returns_freq=freq, group_data=None,
                beta_span=beta_span, long_span=long_span,
                short_span=short_span, vol_span=vol_span,
                mean_adj_type=mean_adj_type)
            all_scores.append(score)
            all_raw.append(raw)

    residual_reversal_score = pd.concat(all_scores, axis=1)[prices.columns].ffill()
    raw_residual_reversal = pd.concat(all_raw, axis=1)[prices.columns].ffill()
    return residual_reversal_score, raw_residual_reversal


# ---------------------------------------------------------------------------
# cluster — scoring within time-varying statistical clusters
# ---------------------------------------------------------------------------
def compute_residual_reversal_cluster_alpha(
        prices: pd.DataFrame,
        benchmark_price: pd.Series = None,
        rolling_clusters: Dict[pd.Timestamp, pd.Series] = None,
        returns_freq: Union[str, pd.Series] = 'ME',
        beta_span: int = 12,
        long_span: int = 1,
        short_span: Optional[int] = None,
        vol_span: Optional[int] = 13,
        mean_adj_type: qis.MeanAdjType = qis.MeanAdjType.EWMA,
        min_cluster_size: int = 3,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute residual reversal with time-varying cluster-based scoring.

    Identical to compute_residual_reversal_alpha for the raw signal
    computation (EWMA beta → lagged residual → EWMA long/short filtered
    RA returns → NEGATE), but applies cross-sectional scoring within
    statistical clusters that evolve over time.

    The sign flip lives in the shared raw signal, so the returned
    ``raw_residual_reversal`` matches compute_residual_reversal_alpha and
    is the negation of the corresponding residual momentum raw signal.

    Args:
        prices: Asset price panel. Index=dates, columns=tickers.
        benchmark_price: Benchmark price series. If None, uses equal-weight.
        rolling_clusters: Dict mapping dates to pd.Series (ticker → cluster_id).
            Extracted from RollingFactorCovarData via extract_rolling_clusters().
        returns_freq: Return frequency. String or pd.Series for mixed-freq.
        beta_span: EWMA span for benchmark beta estimation.
        long_span: EWMA span for the residual reversal lookback (short).
        short_span: Optional EWMA span for short-term subtraction.
        vol_span: EWMA span for volatility normalisation. None disables.
        mean_adj_type: Mean adjustment type for EWMA beta regression.
        min_cluster_size: Minimum cluster size for within-cluster scoring.
            Clusters with size <= min_cluster_size use global statistics.

    Returns:
        Tuple of (residual_reversal_cluster_score, raw_residual_reversal).
    """
    if rolling_clusters is None:
        rolling_clusters = {}

    if isinstance(returns_freq, pd.Series):
        raw_residual_reversal = _compute_raw_residual_reversal_mixed_freq(
            prices=prices, benchmark_price=benchmark_price,
            returns_freqs=returns_freq, beta_span=beta_span,
            long_span=long_span, short_span=short_span,
            vol_span=vol_span, mean_adj_type=mean_adj_type)
    else:
        raw_residual_reversal = _compute_raw_residual_reversal_single_freq(
            prices=prices, benchmark_price=benchmark_price,
            returns_freq=returns_freq, beta_span=beta_span,
            long_span=long_span, short_span=short_span,
            vol_span=vol_span, mean_adj_type=mean_adj_type)

    # score within time-varying clusters (raw is already negated)
    residual_reversal_score = score_within_clusters(
        raw_signal=raw_residual_reversal,
        rolling_clusters=rolling_clusters,
        min_cluster_size=min_cluster_size)

    return residual_reversal_score, raw_residual_reversal
