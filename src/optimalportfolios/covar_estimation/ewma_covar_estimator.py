"""EWMA covariance estimation from sampled log-return prices.

QIS supplies the return, mean and covariance calculations; this module adds
the estimator interface and selects current or scheduled covariance states.
The ordinary kernel uses a zero-seeded exponentially weighted second moment.
Optional volatility normalization selects a different QIS kernel; no
shrinkage-to-identity parameter is implemented.

Span counts return observations and sets decay as 1 - 2 / (span + 1).
It is neither a half-life nor a hard lookback. Class outputs are annualized
using the sampled return index; rebalancing_freq only selects output dates.
The normalized-return kernel initializes volatility from the full supplied
array, so selecting earlier tensor slices is not point-in-time safe.

See docs/covariance_estimators.md in the source checkout for the methodology,
units and timing qualifications. Portfolio weights, objectives and reports
belong to downstream layers.

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
    """Estimate the final QIS EWMA covariance from the supplied price history.

    The helper samples log returns, drops the first price-difference row and,
    when demeaning, subtracts the contemporaneous EWMA mean and drops the first
    zero deviation. The ordinary covariance recursion starts at zero.
    NanBackfill.ZERO_FILL resets non-finite covariance updates to zero; it is
    not equivalent to replacing every missing input return with zero.

    This is a full-panel current fit. Slice prices first for a historical cutoff.
    Volatility normalization uses the QIS kernel whose volatility initialization
    depends on the entire supplied return array.

    Args:
        prices: Date-by-ticker price panel with an ordered DatetimeIndex. Prices
            should represent the intended total-return convention.
        returns_freq: Frequency for sampling log returns, default 'W-WED'.
        span: EWMA span in sampled return observations. Decay is
            1 - 2 / (span + 1); span is not a half-life or a fixed window length.
        is_apply_vol_normalised_returns: Select the QIS normalized-return kernel,
            which reconstructs covariance using EWMA volatilities. This does not
            shrink covariance toward an identity matrix.
        demean: Subtract the current EWMA mean before covariance estimation.
            False retains raw log returns after the initial difference.
        apply_an_factor: Multiply by the annualization factor inferred from the
            sampled return index. False leaves per-observation covariance units.
        **kwargs: Accepted for compatibility but not read or forwarded by this helper.

    Returns:
        Square covariance DataFrame on the price tickers, using the last sampled
        return state. Units are fractional log-return squared, annualized when
        apply_an_factor is True.
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
    """Configure current and rolling EWMA covariance estimates through QIS.

    The ordinary path estimates an exponentially weighted second moment of the
    configured adjusted log returns. It starts covariance at zero, without a
    degrees-of-freedom correction or finite-history weight renormalization.
    The class implements no identity shrinkage.

    Current fits use all supplied prices. Rolling fits compute the full tensor
    and select return-grid observations. The normalized-return option starts
    volatility from the full-array mean square, so future data can affect earlier
    rolling estimates. See docs/covariance_estimators.md for that limitation.

    Attributes:
        rebalancing_freq: Inherited calendar frequency selecting rolling outputs,
            default 'QE'. It does not control return sampling.
        returns_freq: Log-return sampling cadence, default 'W-WED'.
        span: EWMA span in return observations, default 52. Decay is
            1 - 2 / (span + 1); this is not a half-life or hard lookback.
        is_apply_vol_normalised_returns: Use the QIS normalized-return covariance
            kernel instead of the ordinary recursion.
        demean: Subtract the contemporaneous EWMA mean before estimation. The first
            price difference is dropped; demeaning drops one additional zero deviation.

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
        """Return annualized EWMA covariance at the final sampled return date.

        Uses the entire supplied panel with the estimator's return cadence, span,
        demeaning and normalization settings. rebalancing_freq has no effect on
        this current fit. Slice prices explicitly for a historical cutoff.

        Args:
            prices: Ordered date-by-ticker price panel in the intended return convention.

        Returns:
            Square covariance DataFrame with price tickers on both axes, in annual
            fractional log-return-squared units inferred from the sampled return index.
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
        """Select annualized EWMA matrices on the sampled-return rebalance grid.

        The method computes one full covariance tensor before selecting QIS
        rebalancing indicators within time_period, inclusively. History before the
        start still contributes to estimation. The end date filters outputs without
        truncating inputs, so the normalized-return option retains its full-array
        volatility-initialization limitation.

        This schedule uses return-grid observations and need not match the calendar
        keys produced by FactorCovarEstimator. Returned dates do not establish when
        the source observations became available.

        Args:
            prices: Ordered date-by-ticker price history for sampling and estimation.
            time_period: Output-selection bounds, with a required start and optional end.
                Bounds are localized to the sampled return index's timezone.
            rebalancing_freq: Optional override for the inherited output frequency.
                None uses self.rebalancing_freq.

        Returns:
            Dictionary from selected return-grid dates to annual covariance DataFrames
            with price tickers on both axes. If the global schedule exists but no
            selected date falls in time_period, the dictionary is empty.

        Raises:
            ValueError: If the sampled return history has no rebalance indicator at
                the effective frequency.
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
