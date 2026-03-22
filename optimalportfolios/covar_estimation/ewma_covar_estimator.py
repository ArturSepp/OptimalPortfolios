"""
EWMA covariance matrix estimator.

Concrete implementation of CovarEstimator using exponentially weighted
moving average covariance estimation. Supports vol-normalised returns
and shrinkage toward identity.

Usage:
    >>> estimator = EwmaCovarEstimator(returns_freq='W-WED', span=52, rebalancing_freq='QE')
    >>> covar_dict = estimator.fit_rolling_covars(prices=prices, time_period=time_period)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import qis as qis
from typing import Optional, Dict
from dataclasses import dataclass

from optimalportfolios.covar_estimation.covar_estimator import CovarEstimator
from optimalportfolios.covar_estimation.utils import compute_returns_from_prices


def estimate_current_ewma_covar(prices: pd.DataFrame,
                                returns_freq: str = 'W-WED',
                                span: int = 52,
                                is_apply_vol_normalised_returns: bool = False,
                                demean: bool = True,
                                apply_an_factor: bool = True,
                                **kwargs
                                ) -> pd.DataFrame:
    """
    Compute EWMA covariance matrix at the last available date.

    Standalone function for use outside the estimator class (e.g., by
    FactorCovarEstimator for factor covariance estimation).

    Args:
        prices: Asset price panel. Index=dates, columns=tickers.
        returns_freq: Frequency for return computation.
        span: EWMA half-life span in periods.
        is_apply_vol_normalised_returns: If True, normalise returns by rolling vol.
        demean: If True, subtract rolling mean before estimation.
        apply_an_factor: If True, annualise the covariance matrix.

    Returns:
        Covariance matrix (N x N) as pd.DataFrame.
    """
    returns = compute_returns_from_prices(prices=prices, returns_freq=returns_freq, demean=demean, span=span)
    x = returns.to_numpy()
    if is_apply_vol_normalised_returns:
        covar_tensor_txy, _, _ = qis.compute_ewm_covar_tensor_vol_norm_returns(
            a=x, span=span, nan_backfill=qis.NanBackfill.ZERO_FILL)
    else:
        covar_tensor_txy = qis.compute_ewm_covar_tensor(
            a=x, span=span, nan_backfill=qis.NanBackfill.ZERO_FILL)

    covar_t = covar_tensor_txy[-1]
    if apply_an_factor:
        an_factor = qis.infer_annualisation_factor_from_df(data=returns)
    else:
        an_factor = 1.0
    current_covar = pd.DataFrame(an_factor * covar_t, columns=returns.columns, index=returns.columns)
    return current_covar


@dataclass
class EwmaCovarEstimator(CovarEstimator):
    """
    Exponentially weighted covariance matrix estimator.

    Computes EWMA covariance matrices from asset prices, with optional
    vol-normalised returns and shrinkage toward identity.

    Args:
        returns_freq: Frequency for return computation (e.g., 'W-WED', 'ME', 'B').
        span: EWMA half-life span in periods at returns_freq frequency.
        is_apply_vol_normalised_returns: If True, normalise returns by rolling vol
            before covariance estimation (DCC-like effect).
        demean: If True, subtract EWMA rolling mean before estimation.

    Example:
        >>> estimator = EwmaCovarEstimator(returns_freq='W-WED', span=52, rebalancing_freq='QE')
        >>> covar_dict = estimator.fit_rolling_covars(prices=prices, time_period=time_period)
        >>> current = estimator.fit_current_covar(prices=prices)
    """
    returns_freq: str = 'W-WED'
    span: int = 52
    is_apply_vol_normalised_returns: bool = False
    demean: bool = True

    def fit_current_covar(self,
                          prices: pd.DataFrame,
                          ) -> pd.DataFrame:
        """
        Compute annualised EWMA covariance matrix at the last available date.

        Args:
            prices: Asset price panel. Index=dates, columns=tickers.

        Returns:
            Annualised covariance matrix (N x N) as pd.DataFrame.
        """
        return estimate_current_ewma_covar(
            prices=prices,
            returns_freq=self.returns_freq,
            span=self.span,
            is_apply_vol_normalised_returns=self.is_apply_vol_normalised_returns,
            demean=self.demean,
            apply_an_factor=True
        )

    def fit_rolling_covars(self,
                           prices: pd.DataFrame,
                           time_period: qis.TimePeriod,
                           rebalancing_freq: Optional[str] = None,
                           ) -> Dict[pd.Timestamp, pd.DataFrame]:
        """
        Compute rolling EWMA covariance matrices at each rebalancing date.

        Computes the full EWMA covariance tensor in a single O(T) pass,
        then extracts slices at each rebalancing date within the time period.

        Args:
            prices: Asset price panel. Index=dates, columns=tickers.
            time_period: Period over which to generate the rebalancing schedule.
            rebalancing_freq: Override rebalancing frequency. If None, uses self.rebalancing_freq.

        Returns:
            Dict mapping rebalancing dates to annualised covariance matrices.
        """
        freq = rebalancing_freq or self.rebalancing_freq

        returns = compute_returns_from_prices(prices=prices,
                                              returns_freq=self.returns_freq,
                                              demean=self.demean,
                                              span=self.span)
        x = returns.to_numpy()

        if self.is_apply_vol_normalised_returns:
            covar_tensor, _, _ = qis.compute_ewm_covar_tensor_vol_norm_returns(
                a=x, span=self.span, nan_backfill=qis.NanBackfill.ZERO_FILL)
        else:
            covar_tensor = qis.compute_ewm_covar_tensor(
                a=x, span=self.span, nan_backfill=qis.NanBackfill.ZERO_FILL)

        # rebalancing indicator aligned to returns index
        rebalancing_schedule = qis.generate_rebalancing_indicators(df=returns, freq=freq)
        if np.all(rebalancing_schedule == False):
            raise ValueError(
                f"rebalancing schedule is empty for return period "
                f"{qis.get_time_period(df=returns).to_str()} and rebalancing_freq={freq}"
            )

        tickers = prices.columns.to_list()
        an_factor = qis.infer_annualisation_factor_from_df(data=returns)
        start_date = time_period.start.tz_localize(tz=returns.index.tz)

        covars: Dict[pd.Timestamp, pd.DataFrame] = {}
        for idx, (date, is_rebal) in enumerate(rebalancing_schedule.items()):
            if is_rebal and date >= start_date:
                covar_t = covar_tensor[idx]
                covars[date] = pd.DataFrame(an_factor * covar_t, index=tickers, columns=tickers)

        return covars


def estimate_rolling_ewma_covar(prices: pd.DataFrame,
                                time_period: qis.TimePeriod,  # when we start building portfolios
                                returns_freq: str = 'W-WED',
                                rebalancing_freq: str = 'QE',
                                span: int = 52,
                                is_apply_vol_normalised_returns: bool = False,
                                demean: bool = True,
                                apply_an_factor: bool = True,
                                **kwargs
                                ) -> Dict[pd.Timestamp, pd.DataFrame]:
    """
    compute ewma covar matrix: supporting for nans in prices
    output is dict[estimation timestamp, pd.Dataframe(estimated_covar)
    """
    returns = compute_returns_from_prices(prices=prices, returns_freq=returns_freq, demean=demean, span=span)
    x = returns.to_numpy()
    if is_apply_vol_normalised_returns:
        covar_tensor_txy, _, _ = qis.compute_ewm_covar_tensor_vol_norm_returns(a=x, span=span, nan_backfill=qis.NanBackfill.ZERO_FILL)
    else:
        covar_tensor_txy = qis.compute_ewm_covar_tensor(a=x, span=span, nan_backfill=qis.NanBackfill.ZERO_FILL)

    # create rebalancing schedule
    rebalancing_schedule = qis.generate_rebalancing_indicators(df=returns, freq=rebalancing_freq)
    if np.all(rebalancing_schedule == False):
        raise ValueError(f"rebalancing shedule is empty for return period {qis.get_time_period(df=returns).to_str()} "
                         f"and rebalancing_freq={rebalancing_freq}")

    tickers = prices.columns.to_list()
    covars = {}
    if apply_an_factor:
        an_factor = qis.infer_annualisation_factor_from_df(data=returns)
    else:
        an_factor = 1.0
    start_date = time_period.start.tz_localize(tz=returns.index.tz)  # make sure tz is alined with rebalancing_schedule
    for idx, (date, value) in enumerate(rebalancing_schedule.items()):
        if value and date >= start_date:
            covar_t = pd.DataFrame(covar_tensor_txy[idx], index=tickers, columns=tickers)
            covars[date] = an_factor*covar_t
    return covars
