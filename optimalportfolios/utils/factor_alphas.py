"""
compute alphas
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import qis as qis
from enum import Enum
from typing import List, Optional, Tuple, Dict, Union

from optimalportfolios.lasso.lasso_model_estimator import LassoModel
from qis import get_annualization_factor


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
                            ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    compute beta to benchmark_price
    if benchmark_price is None then compute equal weight benchmark
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
    return alphas, ewma_betas


def compute_low_beta_alphas_different_freqs(prices: pd.DataFrame,
                                            rebalancing_freqs: pd.Series,
                                            benchmark_price: pd.Series = None,
                                            beta_span: int = 52,
                                            mean_adj_type: qis.MeanAdjType = qis.MeanAdjType.EWMA,
                                            group_data: pd.Series = None
                                            ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    compute beta to benchmark_price
    if benchmark_price is None then compute equal weight benchmark
    when group_data is used recommended to have same frequency
    """
    rebalancing_freqs = rebalancing_freqs[prices.columns]
    group_freqs = qis.get_group_dict(group_data=rebalancing_freqs)
    ewma_betas = []
    group_alphas = []
    for freq, asset_tickers in group_freqs.items():
        if group_data is not None:
            grouped_prices = qis.split_df_by_groups(df=prices[asset_tickers], group_data=group_data)
        else:
            grouped_prices = {'_': prices[asset_tickers]}
        for group, gprice in grouped_prices.items():
            alphas_, ewma_betas_ = compute_low_beta_alphas(prices=gprice,
                                                           benchmark_price=benchmark_price,
                                                           returns_freq=freq,
                                                           beta_span=beta_span,
                                                           mean_adj_type=mean_adj_type)
            group_alphas.append(alphas_)
            ewma_betas.append(ewma_betas_)
    ewma_betas = pd.concat(ewma_betas, axis=1)[prices.columns].ffill()
    group_alphas = pd.concat(group_alphas, axis=1)[prices.columns].ffill()
    # global_alphas is joint alphas with different frequencies ignoring groups
    global_alphas = qis.df_to_cross_sectional_score(df=-1.0*ewma_betas)[prices.columns].ffill()
    return group_alphas, global_alphas, ewma_betas


def wrapper_compute_low_beta_alphas(prices: pd.DataFrame,
                                    benchmark_price: pd.Series,
                                    rebalancing_freq: Union[str, pd.Series],
                                    group_data_alphas: pd.Series,
                                    beta_span: int = 12,
                                    momentum_long_span: int = 12
                                    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    use wrapper dependent on frequencies
    """
    if isinstance(rebalancing_freq, pd.Series):
        beta_score, _, beta = compute_low_beta_alphas_different_freqs(prices=prices,
                                                                      benchmark_price=benchmark_price,
                                                                      rebalancing_freqs=rebalancing_freq.loc[prices.columns],
                                                                      beta_span=beta_span,
                                                                      group_data=group_data_alphas)
        momentum_score, _, momentum = compute_momentum_alphas_different_freqs(prices=prices,
                                                                              benchmark_price=benchmark_price,
                                                                              rebalancing_freqs=
                                                                              rebalancing_freq.loc[prices.columns],
                                                                              long_span=momentum_long_span,
                                                                              group_data=group_data_alphas)
    else:  # to do implement groups
        beta_score, beta = compute_low_beta_alphas(prices=prices, benchmark_price=benchmark_price,
                                                   returns_freq=rebalancing_freq, beta_span=beta_span)
        momentum_score, momentum = compute_momentum_alphas(prices=prices, benchmark_price=benchmark_price,
                                                           returns_freq=rebalancing_freq,
                                                           long_span=momentum_long_span)
    #momentum_score = momentum / np.nanstd(momentum, axis=1, keepdims=True)
    #beta_score = qis.df_to_cross_sectional_score(df=beta)
    alpha_scores = beta_score.reindex(index=momentum_score.index, method='ffill')
    alpha_scores = alpha_scores.add(momentum_score) / np.sqrt(2.0)
    return alpha_scores, momentum, beta, momentum_score, beta_score


def compute_momentum_alphas(prices: pd.DataFrame,
                            benchmark_price: pd.Series = None,
                            returns_freq: str = 'W-WED',
                            long_span: int = 13,
                            short_span: Optional[int] = None,
                            vol_span: Optional[int] = 13,
                            mean_adj_type: qis.MeanAdjType = qis.MeanAdjType.NONE
                            ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    compute momentum relative benchmark_price
    if benchmark_price is None then compute equal weight benchmark
    """
    returns = qis.to_returns(prices, freq=returns_freq, is_log_returns=True)
    if benchmark_price is not None:  # adjust
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
    return alphas, momentum


def compute_momentum_alphas_different_freqs(prices: pd.DataFrame,
                                            rebalancing_freqs: pd.Series,
                                            benchmark_price: pd.Series = None,
                                            long_span: int = 13,
                                            short_span: Optional[int] = None,
                                            vol_span: Optional[int] = 13,
                                            mean_adj_type: qis.MeanAdjType = qis.MeanAdjType.NONE,
                                            group_data: pd.Series = None
                                            ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    compute momentum relative benchmark_price
    if benchmark_price is None then compute equal weight benchmark
    rebalancing_freqs is
    """
    rebalancing_freqs = rebalancing_freqs[prices.columns]
    group_freqs = qis.get_group_dict(group_data=rebalancing_freqs)
    group_alphas = []
    group_momentum = []
    for freq, asset_tickers in group_freqs.items():
        if group_data is not None:
            grouped_prices = qis.split_df_by_groups(df=prices[asset_tickers], group_data=group_data)
        else:
            grouped_prices = {'_': prices[asset_tickers]}
        for group, gprice in grouped_prices.items():
            alphas_, momentum_ = compute_momentum_alphas(prices=gprice,
                                                         benchmark_price=benchmark_price,
                                                         returns_freq=freq,
                                                         long_span=long_span,
                                                         short_span=short_span,
                                                         vol_span=vol_span,
                                                         mean_adj_type=mean_adj_type)
            group_alphas.append(alphas_)
            group_momentum.append(momentum_)

    momentum = pd.concat(group_momentum, axis=1)[prices.columns].ffill()
    group_alphas = pd.concat(group_alphas, axis=1)[prices.columns].ffill()
    global_alphas = qis.df_to_cross_sectional_score(df=momentum)
    return group_alphas, global_alphas, momentum


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
        groups_momentum_alphas_, _ = compute_momentum_alphas(prices=gprice,
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
        groups_low_beta_alphas_, _ = compute_low_beta_alphas(prices=gprice, returns_freq=returns_freq, beta_span=beta_span)
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


def estimate_lasso_regression_alphas(prices: pd.DataFrame,
                                     risk_factors_prices: pd.DataFrame,
                                     lasso_model: LassoModel,
                                     rebalancing_freq: str = 'ME'
                                     ) -> pd.DataFrame:
    """
    using new lasso model
    compute alphas = return_t - (factor_returns_t*beta_{t_1}
    """
    y = qis.to_returns(prices=prices, is_log_returns=True, drop_first=True, freq=rebalancing_freq)
    x = qis.to_returns(prices=risk_factors_prices, is_log_returns=True, drop_first=True, freq=rebalancing_freq)

    betas, total_vars, residual_vars, r2_t, clusters = lasso_model.estimate_rolling_betas(x=x, y=y)
    estimation_dates = list(betas.keys())
    excess_returns = {}
    for date0, date1 in zip(estimation_dates[:-1], estimation_dates[1:]):
        x_return1 = x.loc[date1, :]
        y_return1 = y.loc[date1, :]
        betas0 = betas[date0]
        excess_returns[date1] = y_return1 - x_return1 @ betas0
    excess_returns = pd.DataFrame.from_dict(excess_returns, orient='index')
    return excess_returns


def wrapper_estimate_regression_alphas(prices: pd.DataFrame,
                                       risk_factors_prices: pd.DataFrame,
                                       estimated_betas: Dict[pd.Timestamp, pd.DataFrame],
                                       rebalancing_freq: Union[str, pd.Series],
                                       annualise: bool = True,
                                       ) -> pd.DataFrame:
    """
    using estimated factor model
    compute alphas = return_t - (factor_returns_t*beta_{t_1}
    """
    estimated_betas_dates = list(estimated_betas.keys())
    
    def estimate_excess_return(x_: pd.DataFrame, y_: pd.DataFrame, freq: str) -> pd.DataFrame:
        estimation_dates = x.index
        excess_returns = {}
        for date0, date1 in zip(estimation_dates[:-1], estimation_dates[1:]):
            if date0 in estimated_betas_dates and date1 in y_.index and date1 in x_.index:
                x_t = x_.loc[date1, :]
                y_t = y_.loc[date1, :]
                betas0 = estimated_betas[date0].loc[:, y_.columns]
                excess_returns[date1] = y_t - x_t @ betas0
        excess_returns = pd.DataFrame.from_dict(excess_returns, orient='index')
        if annualise:
            an = get_annualization_factor(freq=freq)
            excess_returns *= an
        return excess_returns

    if isinstance(rebalancing_freq, str):
        x = qis.to_returns(prices=risk_factors_prices, is_log_returns=True, drop_first=True, freq=rebalancing_freq)
        y = qis.to_returns(prices=prices, is_log_returns=True, drop_first=True, freq=rebalancing_freq)
        excess_returns = estimate_excess_return(x_=x, y_=y, freq=rebalancing_freq)

    else:
        group_freqs = qis.get_group_dict(group_data=rebalancing_freq.loc[prices.columns])
        excess_returns = []
        for freq, asset_tickers in group_freqs.items():
            y = qis.to_returns(prices=prices[asset_tickers], is_log_returns=True, drop_first=True, freq=freq)
            x = qis.to_returns(prices=risk_factors_prices, is_log_returns=True, drop_first=True, freq=freq)
            excess_returns.append(estimate_excess_return(x_=x, y_=y, freq=freq))
        excess_returns = pd.concat(excess_returns, axis=1)
        excess_returns = excess_returns[prices.columns]
    return excess_returns
