"""
Residual momentum alpha computation.

Computes cross-sectional residual momentum scores by stripping out
benchmark beta exposure from asset returns, then filtering with
EWMA long/short risk-adjusted returns (same as momentum.py).

Pipeline:
    returns → EWMA beta to benchmark (lagged) → residual = r_t - β̂_{t-1} · r_bench_t
            → EWMA long/short filtered RA returns → cross-sectional score

References:
    Blitz D., Huij J., Martens M. (2011),
    "Residual Momentum", Journal of Empirical Finance, 18, 506-521.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import qis as qis
from typing import Optional, Tuple, Union


def compute_residual_momentum_alpha(
        prices: pd.DataFrame,
        benchmark_price: pd.Series = None,
        returns_freq: Union[str, pd.Series] = 'ME',
        group_data: Optional[pd.Series] = None,
        beta_span: int = 12,
        long_span: int = 12,
        short_span: Optional[int] = None,
        vol_span: Optional[int] = 13,
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
        long_span: EWMA span for the long momentum signal.
        short_span: Optional EWMA span for short-term reversal subtraction.
        vol_span: EWMA span for volatility normalisation. None disables.
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
        beta_span: int = 12,
        long_span: int = 12,
        short_span: Optional[int] = None,
        vol_span: Optional[int] = 13,
        mean_adj_type: qis.MeanAdjType = qis.MeanAdjType.EWMA
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Single-frequency residual momentum computation."""
    returns = qis.to_returns(prices, freq=returns_freq, is_log_returns=True)

    # benchmark returns
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

    # cross-sectional scoring (within-group if specified)
    if group_data is not None:
        grouped_prices = qis.split_df_by_groups(df=prices, group_data=group_data)
        group_scores = []
        for group, gprice in grouped_prices.items():
            group_cols = [c for c in gprice.columns if c in raw_residual_momentum.columns]
            group_scores.append(
                qis.df_to_cross_sectional_score(df=raw_residual_momentum[group_cols]))
        residual_momentum_score = pd.concat(group_scores, axis=1)[prices.columns]
    else:
        residual_momentum_score = qis.df_to_cross_sectional_score(
            df=raw_residual_momentum)

    return residual_momentum_score, raw_residual_momentum


def _compute_residual_momentum_mixed_freq(
        prices: pd.DataFrame,
        benchmark_price: pd.Series = None,
        returns_freqs: pd.Series = None,
        group_data: Optional[pd.Series] = None,
        beta_span: int = 12,
        long_span: int = 12,
        short_span: Optional[int] = None,
        vol_span: Optional[int] = 13,
        mean_adj_type: qis.MeanAdjType = qis.MeanAdjType.EWMA
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Mixed-frequency residual momentum: compute per frequency group, merge."""
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

    residual_momentum_score = pd.concat(all_scores, axis=1)[prices.columns].ffill()
    raw_residual_momentum = pd.concat(all_raw, axis=1)[prices.columns].ffill()
    return residual_momentum_score, raw_residual_momentum