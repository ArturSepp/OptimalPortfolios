"""
for multi-asset portfolios we compute managers alpha with is regression alpha
for other asset classes we compute grouped alpha
"""

import numpy as np
import pandas as pd
import qis as qis
from typing import Union, List, Tuple, Dict, Optional
from dataclasses import dataclass, asdict

from optimalportfolios.utils.lasso import LassoModel
from optimalportfolios.utils.factor_alphas import (compute_low_beta_alphas_different_freqs,
                                                   compute_momentum_alphas_different_freqs,
                                                   compute_low_beta_alphas,
                                                   compute_momentum_alphas,
                                                   estimate_lasso_regression_alphas)


@dataclass
class ManagerAlphas:
    alpha_scores: pd.DataFrame
    beta: Optional[pd.DataFrame]
    momentum: Optional[pd.DataFrame]
    managers_alphas: Optional[pd.DataFrame]

    def get_alphas_snapshot(self, date: pd.Timestamp) -> pd.DataFrame:
        if date not in self.alpha_scores.index:
            raise KeyError(f"{date} is not in {self.alpha_scores.index}")
        snapshot = self.alpha_scores.loc[date, :].to_frame('Alpha Scores')
        if self.beta is not None:
            snapshot = pd.concat([snapshot, self.beta.loc[date, :].to_frame('Beta')], axis=1)
        if self.momentum is not None:
            snapshot = pd.concat([snapshot, self.momentum.loc[date, :].to_frame('Momentum')], axis=1)
        if self.managers_alphas is not None:
            snapshot = pd.concat([snapshot, self.managers_alphas.loc[date, :].to_frame('Managers Alpha')], axis=1)
        return snapshot

    def to_dict(self) -> Dict[str, pd.DataFrame]:
        return asdict(self)


def compute_manager_alphas(prices: pd.DataFrame,
                           benchmark_price: pd.Series,
                           rebalancing_freq: Union[str, pd.Series],
                           risk_factors_prices: pd.DataFrame,
                           lasso_model: LassoModel,
                           group_data: pd.Series,
                           group_data_alphas: pd.Series,
                           managers_groups: Optional[List[str]],
                           beta_span: int = 12,
                           momentum_long_span: int = 12,
                           manager_alphas_rebalancing_freq: str = 'ME',
                           managers_alpha_span: int = 12
                           ) -> ManagerAlphas:
    """
    for multi-asset portfolios we compute managers alpha with is regression alpha
    for other asset classes we compute grouped alpha
    beta_span = 12 for monthly rebalancing_freq
    """
    # compute alphas for manager groups
    if managers_groups is not None:
        prices1 = prices.copy()
        for managers_group in managers_groups:
            # group_tickers = group_data[group_data == managers_group]
            # in_prices = np.isin(group_tickers.index, prices1.columns)
            # tickers_to_drop = group_tickers.loc[in_prices].index.to_list()
            # prices1 = prices1.drop(tickers_to_drop, axis=1)
            group_tickers = group_data[group_data == managers_group].index.to_list()
            prices1 = prices1.drop(group_tickers, axis=1)
    else:
        prices1 = prices

    # if some assets remain their alpha is momntum + beta
    if not prices1.empty:
        if isinstance(rebalancing_freq, pd.Series):
            beta_score, _, beta = compute_low_beta_alphas_different_freqs(prices=prices1, benchmark_price=benchmark_price,
                                                                          rebalancing_freqs=rebalancing_freq.loc[prices1.columns],
                                                                          beta_span=beta_span,
                                                                          group_data=group_data_alphas)
            momentum_score, _, momentum = compute_momentum_alphas_different_freqs(prices=prices1, benchmark_price=benchmark_price,
                                                                                  rebalancing_freqs=rebalancing_freq.loc[prices1.columns],
                                                                                  long_span=momentum_long_span,
                                                                                  group_data=group_data_alphas)
        else:  # to do implement groups
            beta_score, beta = compute_low_beta_alphas(prices=prices1, benchmark_price=benchmark_price,
                                                       returns_freq=rebalancing_freq, beta_span=beta_span)
            momentum_score, momentum = compute_momentum_alphas(prices=prices1, benchmark_price=benchmark_price,
                                                               returns_freq=rebalancing_freq, long_span=momentum_long_span)
        beta_score = beta_score.reindex(index=momentum_score.index, method='ffill')
        alpha_scores = beta_score.add(momentum_score) / np.sqrt(2.0)
    else:
        beta = None
        momentum = None
        alpha_scores = None

    # for hedge funds create group lasso
    if managers_groups is not None:
        managers_alphas = []
        for managers_group in managers_groups:
            group_tickers = group_data[group_data == managers_group].index.to_list()
            lasso_model = lasso_model.copy(kwargs=dict(group_data=pd.Series(managers_group, index=group_tickers)))
            excess_returns = estimate_lasso_regression_alphas(prices=prices[group_tickers],
                                                              risk_factors_prices=risk_factors_prices,
                                                              rebalancing_freq=manager_alphas_rebalancing_freq,
                                                              lasso_model=lasso_model)
            # alphas_ = excess_returns.rolling(managers_alpha_span).sum()
            alphas_ = managers_alpha_span*qis.compute_ewm(data=excess_returns, span=managers_alpha_span)
            managers_alphas.append(alphas_)
            #hf_alphas = qis.df_to_cross_sectional_score(df=hf_alphas)
            # alphas_joint = pd.concat([alphas_joint.drop(group_tickers, axis=1), hf_alphas], axis=1)
        # take joint alphas and compute scores
        managers_alphas = pd.concat(managers_alphas, axis=1)
        managers_scores = qis.df_to_cross_sectional_score(df=managers_alphas)

        if alpha_scores is not None:
            managers_scores = managers_scores.reindex(index=alpha_scores.index).ffill()
            alpha_scores = pd.concat([alpha_scores, managers_scores], axis=1)
        else:
            alpha_scores = managers_scores
    else:
        managers_alphas = None

    alpha_scores = alpha_scores[prices.columns].ffill()
    alphas = ManagerAlphas(alpha_scores=alpha_scores, beta=beta, momentum=momentum, managers_alphas=managers_alphas)
    return alphas
