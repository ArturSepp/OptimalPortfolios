"""
compute alphas
"""
from enum import Enum
from typing import List, Optional

import numpy as np
import pandas as pd
import qis as qis


class AlphaSignal(Enum):
    """
    enumeration of implemented alphas
    """
    MOMENTUM = 'Momentum'
    LOW_BETA = 'LowBeta'
    MOMENTUM_AND_BETA = 'MomentumAndBeta'


def compute_low_beta_alphas(prices: pd.DataFrame,
                            benchmark_price: pd.Series = None,
                            returns_freq: Optional[str] = 'W-WED',
                            beta_span: int = 52,
                            mean_adj_type: qis.MeanAdjType = qis.MeanAdjType.EWMA
                            ) -> pd.DataFrame:
    """
    compute beta to equal weight benchmark
    """
    returns = qis.to_returns(prices, freq=returns_freq, is_log_returns=True)
    if benchmark_price is None:  # use equal weight returns
        benchmark_returns = pd.Series(np.nanmean(returns.to_numpy(), axis=1), index=returns.index)
    else:
        benchmark_price = benchmark_price.reindex(index=prices.index, method='ffill')
        benchmark_returns = qis.to_returns(benchmark_price, freq=returns_freq, is_log_returns=True)

    ewm_linear_model = qis.EwmLinearModel(x=benchmark_returns.to_frame('benchmark'), y=returns)
    ewm_linear_model.fit(span=beta_span, mean_adj_type=mean_adj_type, is_x_correlated=True)
    ewma_betas = ewm_linear_model.loadings['benchmark']
    # set zeros to nans for signal
    ewma_betas = ewma_betas.replace({0.0: np.nan})
    alphas = qis.df_to_cross_sectional_score(df=-1.0 * ewma_betas)
    return alphas


def compute_momentum_alphas(prices: pd.DataFrame,
                            benchmark_price: pd.Series = None,
                            returns_freq: str = 'W-WED',
                            long_span: int = 13,
                            short_span: Optional[int] = None,
                            vol_span: Optional[int] = 13,
                            mean_adj_type: qis.MeanAdjType = qis.MeanAdjType.NONE
                            ) -> pd.DataFrame:
    returns = qis.to_returns(prices, freq=returns_freq, is_log_returns=True)
    if benchmark_price is not None: # adjust
        benchmark_price = benchmark_price.reindex(index=prices.index, method='ffill')
        benchmark_returns = qis.to_returns(benchmark_price, freq=returns_freq, is_log_returns=True)
        returns = returns.subtract(qis.np_array_to_df_columns(a=benchmark_returns.to_numpy(), ncols=len(returns.columns)))

    momentum = qis.compute_ewm_long_short_filtered_ra_returns(returns=returns,
                                                              vol_span=vol_span,
                                                              long_span=long_span,
                                                              short_span=short_span,
                                                              weight_lag=0,
                                                              mean_adj_type=mean_adj_type)
    # momentum = qis.map_signal_to_weight(signals=momentum, loc=0.0, slope_right=0.5, slope_left=0.5, tail_level=3.0)
    alphas = qis.df_to_cross_sectional_score(df=momentum)
    return alphas


def compute_ra_carry_alphas(prices: pd.DataFrame,
                            carry: pd.DataFrame,
                            returns_freq: str = 'W-WED',
                            vol_span: Optional[int] = 13,
                            mean_adj_type: qis.MeanAdjType = qis.MeanAdjType.NONE
                            ) -> pd.DataFrame:
    returns = qis.to_returns(prices, freq=returns_freq, is_log_returns=True)
    ewm_vol = qis.compute_ewm_vol(data=returns,
                                  span=vol_span,
                                  mean_adj_type=mean_adj_type,
                                  annualize=True)
    ra_carry = carry.reindex(index=ewm_vol.index, method='ffill').divide(ewm_vol)
    # momentum = qis.map_signal_to_weight(signals=momentum, loc=0.0, slope_right=0.5, slope_left=0.5, tail_level=3.0)
    alphas = qis.df_to_cross_sectional_score(df=ra_carry)
    return alphas


def compute_grouped_alphas(prices: pd.DataFrame,
                           group_data: pd.Series,
                           group_order: List[str],
                           alpha_signal: AlphaSignal = AlphaSignal.MOMENTUM,
                           mom_long_span: int = 12,
                           mom_short_span: Optional[int] = 1,
                           beta_span: int = 24,
                           returns_freq: Optional[str] = None
                           ) -> pd.DataFrame:
    """
    compute alpha signals by group (asset class)
    """
    if alpha_signal == AlphaSignal.MOMENTUM:
        alphas = compute_grouped_momentum_alphas(prices=prices,
                                                 group_data=group_data,
                                                 group_order=group_order,
                                                 mom_long_span=mom_long_span,
                                                 mom_short_span=mom_short_span,
                                                 returns_freq=returns_freq)

    elif alpha_signal == AlphaSignal.LOW_BETA:
        alphas = compute_grouped_low_beta_alphas(prices=prices,
                                                 group_data=group_data,
                                                 group_order=group_order,
                                                 beta_span=beta_span,
                                                 returns_freq=returns_freq)

    elif alpha_signal == AlphaSignal.MOMENTUM_AND_BETA:
        alphas1 = compute_grouped_momentum_alphas(prices=prices,
                                                  group_data=group_data,
                                                  group_order=group_order,
                                                  mom_long_span=mom_long_span,
                                                  mom_short_span=mom_short_span,
                                                  returns_freq=returns_freq)

        alphas2 = compute_grouped_low_beta_alphas(prices=prices,
                                                  group_data=group_data,
                                                  group_order=group_order,
                                                  beta_span=beta_span,
                                                  returns_freq=returns_freq)
        alphas = alphas1.add(alphas2) / np.sqrt(2)

    else:
        raise NotImplementedError(f"alpha_signal={alpha_signal}")

    return alphas


def compute_grouped_momentum_alphas(prices: pd.DataFrame,
                                    group_data: Optional[pd.Series],
                                    group_order: List[str] = None,
                                    returns_freq: Optional[str] = None,
                                    mom_long_span: int = 3,
                                    mom_short_span: Optional[int] = 1,
                                    vol_span: Optional[int] = 13
                                    ) -> pd.DataFrame:
    if group_data is not None:
        grouped_prices = qis.split_df_by_groups(df=prices, group_data=group_data, group_order=group_order)
    else:
        grouped_prices = {'_': prices}
    groups_momentum_alphas = []
    for group, gprice in grouped_prices.items():
        groups_momentum_alphas_ = compute_momentum_alphas(prices=gprice,
                                                          returns_freq=returns_freq,
                                                          long_span=mom_long_span,
                                                          short_span=mom_short_span,
                                                          vol_span=vol_span)
        groups_momentum_alphas.append(groups_momentum_alphas_)
    groups_momentum_alphas = pd.concat(groups_momentum_alphas, axis=1)[prices.columns]  # align
    return groups_momentum_alphas


def compute_grouped_low_beta_alphas(prices: pd.DataFrame,
                                    group_data: Optional[pd.Series],
                                    group_order: List[str] = None,
                                    returns_freq: Optional[str] = None,
                                    beta_span: int = 12
                                    ) -> pd.DataFrame:
    if group_data is not None:
        grouped_prices = qis.split_df_by_groups(df=prices, group_data=group_data, group_order=group_order)
    else:
        grouped_prices = {'_': prices}
    groups_low_beta_alphas = []
    for group, gprice in grouped_prices.items():
        groups_low_beta_alphas_ = compute_low_beta_alphas(prices=gprice, returns_freq=returns_freq, beta_span=beta_span)
        groups_low_beta_alphas.append(groups_low_beta_alphas_)
    groups_low_beta_alphas = pd.concat(groups_low_beta_alphas, axis=1)[prices.columns]  # align
    return groups_low_beta_alphas


def compute_alpha_long_only_weights(prices: pd.DataFrame,
                                    group_data: Optional[pd.Series] = None,
                                    group_order: List[str] = None,
                                    alpha_signal: AlphaSignal = AlphaSignal.MOMENTUM,
                                    mom_long_span: int = 12,
                                    mom_short_span: Optional[int] = 1,
                                    beta_span: int = 24,
                                    returns_freq: Optional[str] = None
                                    ) -> pd.DataFrame:
    """
    create long only portfolio weights based on alpha signals
    """
    alphas = compute_grouped_alphas(prices=prices,
                                    group_data=group_data,
                                    group_order=group_order,
                                    alpha_signal=alpha_signal,
                                    mom_long_span=mom_long_span,
                                    mom_short_span=mom_short_span,
                                    beta_span=beta_span,
                                    returns_freq=returns_freq)
    signal_weights = qis.df_to_long_only_allocation_sum1(df=alphas)
    return signal_weights
