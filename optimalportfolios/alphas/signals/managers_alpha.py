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

    For each asset-return observation r_t (on the asset reporting-frequency
    grid), the systematic part is stripped with the most recent factor loadings
    estimated strictly before t (lagged beta, no look-ahead):

        excess_t = r_t - beta_asof(t-1) f_t

    The applicable beta date is resolved with
    ``qis.find_upto_date_from_datetime_index`` (latest betas date <= the prior
    return date) rather than requiring the return grid to *exactly* equal the
    betas grid. Exact-membership matching (``date0 in estimated_betas_dates``)
    silently produced an EMPTY frame whenever the rebalancing schedule that
    keys ``estimated_betas`` and the resampled return grid did not share
    timestamps -- the sub-monthly-cadence failure mode (e.g. ``2W-WED``), where
    ``generate_dates_schedule`` (start-anchored) and ``to_returns`` (epoch-
    anchored) need not coincide. The as-of match makes the signal robust to any
    such offset. Factor returns are taken over the asset-return periods (factor
    prices reindexed onto the return dates, then differenced), exactly as the
    betas were fit in ``estimate_lasso_factor_covar_data``.
    """
    estimated_betas_dates = list(estimated_betas.keys())

    def _compute_excess_for_freq(asset_prices: pd.DataFrame, freq: str) -> pd.DataFrame:
        # asset returns on this frequency define the spine
        y_ = qis.to_returns(prices=asset_prices, is_log_returns=True, drop_first=True, freq=freq)
        # factor returns over the SAME periods as the asset returns: reindex the
        # factor prices onto the return grid then difference (mirrors the
        # covariance estimator, so the residual uses the same f_t the betas saw)
        x_prices = risk_factor_prices.reindex(index=y_.index, method='ffill').ffill()
        x_ = qis.to_returns(prices=x_prices, is_log_returns=True, is_first_zero=False,
                            drop_first=False, freq=None)

        excess_returns = {}
        for date0, date1 in zip(y_.index[:-1], y_.index[1:]):
            # lagged beta: latest estimate at-or-before the prior return date
            # (=> strictly before date1, no look-ahead). As-of lookup bridges a
            # betas-grid vs return-grid offset instead of demanding exact equality.
            beta_date = qis.find_upto_date_from_datetime_index(estimated_betas_dates, date0)
            if beta_date is None:
                continue  # no beta estimated yet (pre-warmup) -> skip this period
            x_t = x_.loc[date1, :]
            y_t = y_.loc[date1, :]
            if x_t.isna().any() or y_t.isna().any():
                continue  # incomplete period on either grid -> skip, do not fabricate
            betas_t = estimated_betas[beta_date].loc[y_.columns, :]
            excess_returns[date1] = y_t - betas_t @ x_t
        excess_returns = pd.DataFrame.from_dict(excess_returns, orient='index')
        if annualise and not excess_returns.empty:
            excess_returns *= get_annualization_factor(freq=freq)
        return excess_returns

    if isinstance(rebalancing_freq, str):
        excess_returns = _compute_excess_for_freq(asset_prices=prices, freq=rebalancing_freq)
    else:
        group_freqs = qis.get_group_dict(group_data=rebalancing_freq.loc[prices.columns])
        excess_parts = [_compute_excess_for_freq(asset_prices=prices[asset_tickers], freq=freq)
                        for freq, asset_tickers in group_freqs.items()]
        excess_parts = [p for p in excess_parts if not p.empty]
        excess_returns = (pd.concat(excess_parts, axis=1) if excess_parts
                          else pd.DataFrame(index=prices.index))
        # reindex (not [..]) so a frequency block that produced nothing degrades
        # to NaN columns rather than raising an opaque KeyError
        excess_returns = excess_returns.reindex(columns=prices.columns)

    return excess_returns