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
                                                   estimate_lasso_regression_alphas,
                                                   wrapper_compute_low_beta_alphas,
                                                   wrapper_estimate_regression_alphas)


@dataclass
class ManagerAlphas:
    alpha_scores: pd.DataFrame
    beta: Optional[pd.DataFrame]
    momentum: Optional[pd.DataFrame]
    managers_alphas: Optional[pd.DataFrame]
    momentum_score: Optional[pd.DataFrame]
    beta_score: Optional[pd.DataFrame]
    managers_scores: Optional[pd.DataFrame]

    def get_alphas_snapshot(self, date: pd.Timestamp) -> pd.DataFrame:
        if date not in self.alpha_scores.index:
            raise KeyError(f"{date} is not in {self.alpha_scores.index}")
        snapshot = self.alpha_scores.loc[date, :].to_frame('Alpha Scores')

        if self.momentum_score is not None:
            snapshot = pd.concat([snapshot, self.momentum_score.loc[date, :].to_frame('Momentum Score')], axis=1)
        if self.beta_score is not None:
            snapshot = pd.concat([snapshot, self.beta_score.loc[date, :].to_frame('Beta Score')], axis=1)
        if self.managers_scores is not None:
            if date in self.managers_scores.index:
                snapshot = pd.concat([snapshot, self.managers_scores.loc[date, :].to_frame('Managers Score')], axis=1)
            else:
                snapshot = pd.concat([snapshot, self.managers_scores.iloc[-1, :].to_frame('Managers Score')], axis=1)
        if self.momentum is not None:
            snapshot = pd.concat([snapshot, self.momentum.loc[date, :].to_frame('Momentum')], axis=1)
        if self.beta is not None:
            snapshot = pd.concat([snapshot, self.beta.loc[date, :].to_frame('Beta')], axis=1)
        if self.managers_alphas is not None:
            if date in self.managers_alphas.index:
                snapshot = pd.concat([snapshot, self.managers_alphas.loc[date, :].to_frame('Managers Alpha')], axis=1)
            else:
                snapshot = pd.concat([snapshot, self.managers_alphas.iloc[-1, :].to_frame('Managers Alpha')], axis=1)
        return snapshot

    def to_dict(self) -> Dict[str, pd.DataFrame]:
        return asdict(self)


def compute_manager_alphas(prices: pd.DataFrame,
                           benchmark_price: pd.Series,
                           risk_factors_prices: pd.DataFrame,
                           alpha_beta_type: pd.Series,
                           rebalancing_freq: Union[str, pd.Series],
                           estimated_betas: Dict[pd.Timestamp, pd.DataFrame],
                           group_data_alphas: pd.Series,
                           beta_span: int = 12,
                           momentum_long_span: int = 12,
                           managers_alpha_span: int = 12,
                           return_annualisation_freq_dict: Optional[Dict[str, float]] = {'ME': 12.0, 'QE': 4.0}
                           ) -> ManagerAlphas:
    """
    for multi-asset portfolios we compute alpha based on the type:
    1) Beta

    managers alpha with is regression alpha
    for other asset classes we compute grouped alpha
    beta_span = 12 for monthly rebalancing_freq
    """
    # 1. compute momentum and low betas for selected universe
    beta_assets = alpha_beta_type.loc[alpha_beta_type == 'Beta'].index.to_list()
    if len(beta_assets) == 0:
        raise NotImplementedError
    alpha_scores, momentum, beta, momentum_score, beta_score = wrapper_compute_low_beta_alphas(prices=prices[beta_assets],
                                                                                               benchmark_price=benchmark_price,
                                                                                               rebalancing_freq=rebalancing_freq,
                                                                                               group_data_alphas=group_data_alphas.loc[beta_assets],
                                                                                               beta_span=beta_span,
                                                                                               momentum_long_span=momentum_long_span)
    # 2. compute alphas for managers
    alpha_assets = alpha_beta_type.loc[alpha_beta_type == 'Alpha'].index.to_list()
    if len(alpha_assets) == 0:
        raise NotImplementedError

    excess_returns = wrapper_estimate_regression_alphas(prices=prices[alpha_assets],
                                                        risk_factors_prices=risk_factors_prices,
                                                        estimated_betas=estimated_betas,
                                                        rebalancing_freq=rebalancing_freq,
                                                        return_annualisation_freq_dict=return_annualisation_freq_dict)
    # alphas_ = excess_returns.rolling(managers_alpha_span).sum()
    managers_alphas = qis.compute_ewm(data=excess_returns, span=managers_alpha_span)
    # managers_scores = qis.df_to_cross_sectional_score(df=managers_alphas)
    managers_scores = managers_alphas / np.nanstd(managers_alphas, axis=1, keepdims=True)

    # merge
    managers_scores = managers_scores.reindex(index=alpha_scores.index).ffill()
    alpha_scores = pd.concat([alpha_scores, managers_scores], axis=1)
    alpha_scores = alpha_scores[prices.columns].ffill()
    alphas = ManagerAlphas(alpha_scores=alpha_scores,
                           beta=beta,
                           momentum=momentum,
                           managers_alphas=managers_alphas,
                           momentum_score=momentum_score,
                           beta_score=beta_score,
                           managers_scores=managers_scores)
    return alphas
