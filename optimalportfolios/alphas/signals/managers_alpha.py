"""
Managers regression alpha computation.

Computes cross-sectional alpha scores from rolling factor model residuals.
Uses pre-estimated factor loadings (from FactorCovarEstimator) to strip
out systematic exposure, then scores the residual as manager skill.

Pipeline:
    returns → subtract factor exposure (lagged betas) → EWMA smooth → score
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import qis as qis
from typing import Optional, Tuple, Union, Dict
from qis import get_annualization_factor


def compute_managers_alpha(prices: pd.DataFrame,
                           risk_factor_prices: pd.DataFrame,
                           estimated_betas: Dict[pd.Timestamp, pd.DataFrame],
                           returns_freq: Union[str, pd.Series] = 'ME',
                           alpha_span: int = 12,
                           annualise: bool = True
                           ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute cross-sectional managers alpha scores from factor model residuals.

    For each asset, computes the excess return after removing systematic
    factor exposure using lagged betas:

        excess_return_t = r_t - β_{t-1}' f_t

    The excess returns are smoothed with EWMA and converted to
    cross-sectional scores.

    Args:
        prices: Asset price panel. Index=dates, columns=tickers.
        risk_factor_prices: Factor price panel. Index=dates, columns=factors.
        estimated_betas: Pre-estimated factor loadings. Dict mapping
            estimation dates to DataFrames (N × M).
        returns_freq: Return frequency. String or pd.Series.
        alpha_span: EWMA span for smoothing excess returns.
        annualise: If True, annualise excess returns.

    Returns:
        Tuple of (managers_score, raw_managers_alpha).
    """
    excess_returns = _estimate_rolling_regression_alphas(
        prices=prices,
        risk_factor_prices=risk_factor_prices,
        estimated_betas=estimated_betas,
        rebalancing_freq=returns_freq,
        annualise=annualise)

    raw_managers_alpha = qis.compute_ewm(data=excess_returns, span=alpha_span)

    managers_score = raw_managers_alpha.divide(
        np.nanstd(raw_managers_alpha, axis=1, keepdims=True))

    return managers_score, raw_managers_alpha


def _estimate_rolling_regression_alphas(prices: pd.DataFrame,
                                        risk_factor_prices: pd.DataFrame,
                                        estimated_betas: Dict[pd.Timestamp, pd.DataFrame],
                                        rebalancing_freq: Union[str, pd.Series],
                                        annualise: bool = True
                                        ) -> pd.DataFrame:
    """
    Compute rolling excess returns after removing factor exposure.

    Uses lagged betas (from previous estimation date) to avoid look-ahead.
    """
    estimated_betas_dates = list(estimated_betas.keys())

    def _compute_excess_for_freq(x_: pd.DataFrame, y_: pd.DataFrame, freq: str) -> pd.DataFrame:
        estimation_dates = x_.index
        excess_returns = {}
        for date0, date1 in zip(estimation_dates[:-1], estimation_dates[1:]):
            if date0 in estimated_betas_dates and date1 in y_.index and date1 in x_.index:
                x_t = x_.loc[date1, :]
                y_t = y_.loc[date1, :]
                betas_t0 = estimated_betas[date0].loc[y_.columns, :]
                excess_returns[date1] = y_t - betas_t0 @ x_t
        excess_returns = pd.DataFrame.from_dict(excess_returns, orient='index')
        if annualise:
            an = get_annualization_factor(freq=freq)
            excess_returns *= an
        return excess_returns

    if isinstance(rebalancing_freq, str):
        x = qis.to_returns(prices=risk_factor_prices, is_log_returns=True, drop_first=True, freq=rebalancing_freq)
        y = qis.to_returns(prices=prices, is_log_returns=True, drop_first=True, freq=rebalancing_freq)
        excess_returns = _compute_excess_for_freq(x_=x, y_=y, freq=rebalancing_freq)
    else:
        group_freqs = qis.get_group_dict(group_data=rebalancing_freq.loc[prices.columns])
        excess_parts = []
        for freq, asset_tickers in group_freqs.items():
            y = qis.to_returns(prices=prices[asset_tickers], is_log_returns=True, drop_first=True, freq=freq)
            x = qis.to_returns(prices=risk_factor_prices, is_log_returns=True, drop_first=True, freq=freq)
            excess_parts.append(_compute_excess_for_freq(x_=x, y_=y, freq=freq))
        excess_returns = pd.concat(excess_parts, axis=1)
        excess_returns = excess_returns[prices.columns]

    return excess_returns
