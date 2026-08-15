"""
EWMA covariance matrix estimator.

Concrete implementation of CovarEstimator using exponentially weighted
moving average covariance estimation. Supports vol-normalised returns
and shrinkage toward identity.

Usage:
    >>> import numpy as np
    >>> import pandas as pd
    >>> dates = pd.date_range('2020-01-01', periods=260, freq='W-WED')
    >>> drift = np.exp(0.0004 * np.arange(260))
    >>> prices = pd.DataFrame({'A': 100.0 * drift,
    ...                        'B': 100.0 * drift * (1.0 + 0.02 * np.sin(np.arange(260) / 8.0)),
    ...                        'C': 100.0 / drift}, index=dates)
    >>> estimator = EwmaCovarEstimator(returns_freq='W-WED', span=52, rebalancing_freq='QE')
    >>> covar = estimator.fit_current_covar(prices=prices)
    >>> covar.shape
    (3, 3)
    >>> list(covar.columns)
    ['A', 'B', 'C']
    >>> bool(np.allclose(covar, covar.T))  # a covariance matrix is symmetric
    True
    >>> bool((np.diag(covar) > 0.0).all())  # and carries positive variances
    True

Returns are sampled at ``returns_freq`` and the estimator reports
``Σ_annual = annualisation_factor × Σ_EWMA``; weights are not part of this layer.
Main entry points are ``EwmaCovarEstimator`` and ``estimate_current_ewma_covar``. Boundary:
portfolio objectives, constraints, and performance reporting are not implemented here.
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
        >>> import numpy as np
        >>> import pandas as pd
        >>> import qis
        >>> dates = pd.date_range('2020-01-01', periods=260, freq='W-WED')
        >>> drift = np.exp(0.0004 * np.arange(260))
        >>> prices = pd.DataFrame({'A': 100.0 * drift, 'B': 100.0 / drift}, index=dates)
        >>> estimator = EwmaCovarEstimator(returns_freq='W-WED', span=52, rebalancing_freq='QE')

        One matrix per rebalancing date in the period, keyed by that date:

        >>> time_period = qis.TimePeriod(dates[104], dates[-1])
        >>> covar_dict = estimator.fit_rolling_covars(prices=prices, time_period=time_period)
        >>> len(covar_dict)
        12
        >>> all(covar.shape == (2, 2) for covar in covar_dict.values())
        True

        `fit_current_covar` is the same estimate at the end of the panel only:

        >>> estimator.fit_current_covar(prices=prices).shape
        (2, 2)
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
        if time_period.end is not None:
            end_date = time_period.end.tz_localize(tz=returns.index.tz)
        else:
            end_date = None

        covars: Dict[pd.Timestamp, pd.DataFrame] = {}
        for idx, (date, is_rebal) in enumerate(rebalancing_schedule.items()):
            if is_rebal and date >= start_date and (end_date is None or date <= end_date):
                covar_t = covar_tensor[idx]
                covars[date] = pd.DataFrame(an_factor * covar_t, index=tickers, columns=tickers)

        return covars
# `estimate_rolling_ewma_covar` used to be defined here. It was an independent reimplementation of
# `qis.estimate_rolling_ewma_covar`, which this package already depends on and which qis documents
# in its core API: two same-named estimators with near-identical signatures, one package depending
# on the other, free to drift apart without anything failing. The name is re-exported from qis so
# that callers are unaffected. See CHANGELOG 6.6.0 for the measured difference between the two.
from qis import estimate_rolling_ewma_covar  # noqa: F401,E402
