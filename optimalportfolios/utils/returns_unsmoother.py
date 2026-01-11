"""
returns unsmoothing using AR-1 betas
"""
import numpy as np
import pandas as pd
import qis as qis
from typing import Optional, Tuple


def adjust_returns_with_ar1(returns: pd.DataFrame,
                            span: int = 20,
                            mean_adj_type: qis.MeanAdjType = qis.MeanAdjType.EWMA,
                            warmup_period: Optional[int] = 10,
                            max_value_for_beta: Optional[float] = 0.75,
                            apply_ewma_mean_smoother: bool = True
                            ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    estimate: x_{t} = beta * x_{t-1}
    compute: usmoothed value = (x_{t} - beta * x_{t-1})  / ( 1.0 - beta)
    beta is clipped to make it stationary
    """
    x = returns.shift(1)
    betas, _, _, _, _, ewm_r2 = qis.compute_ewm_beta_alpha_forecast(x_data=x,
                                                                    y_data=returns,
                                                                    mean_adj_type=mean_adj_type,
                                                                    span=span)
    if max_value_for_beta is not None:
        betas = betas.clip(lower=0.0, upper=max_value_for_beta)
    if apply_ewma_mean_smoother:
        betas = qis.compute_ewm(data=betas, span=span)

    if warmup_period is not None:   # set to nan first nonnan in warmup_period and backfill from the first available beta
        betas = qis.set_nans_for_warmup_period(a=betas, warmup_period=warmup_period)
        betas = betas.reindex(index=returns.index).bfill()

    prediction = x.multiply(betas.shift(1))
    unsmoothed = (returns - prediction).divide(1.0-betas.shift(1))

    # adjustment to match the mean
    #mean_true = returns.expanding().mean()
    #mean1 = unsmoothed.expanding().mean()
    #unsmoothed += mean_true - mean1
    # unsmoothed += np.nanmean(returns-unsmoothed, axis=0) #+ 0.5*(np.nanvar(returns, axis=0))

    return unsmoothed, betas, ewm_r2


def compute_ar1_unsmoothed_prices(prices: pd.DataFrame,
                                  freq: str = 'QE',
                                  span: int = 40,
                                  mean_adj_type: qis.MeanAdjType = qis.MeanAdjType.EWMA,
                                  warmup_period: Optional[int] = 8,
                                  max_value_for_beta: Optional[float] = 0.75,
                                  is_log_returns: bool = True
                                  ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = qis.to_returns(prices, freq=freq, drop_first=False, is_log_returns=is_log_returns)
    unsmoothed, betas, ewm_r2 = adjust_returns_with_ar1(returns=y,
                                                        span=span,
                                                        mean_adj_type=mean_adj_type,
                                                        warmup_period=warmup_period,
                                                        max_value_for_beta=max_value_for_beta)
    if is_log_returns:  # back to compounded
        unsmoothed = np.expm1(unsmoothed)
    navs = qis.returns_to_nav(returns=unsmoothed)
    return navs, unsmoothed, betas, ewm_r2
