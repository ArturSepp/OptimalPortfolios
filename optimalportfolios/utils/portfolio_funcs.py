"""
examples of
"""
from __future__ import division

import numpy as np
import pandas as pd
from typing import Tuple, Union, Optional
from numba import njit


@njit
def compute_portfolio_variance(w: np.ndarray, covar: np.ndarray) -> float:
    return w.T @ covar @ w


@njit
def compute_portfolio_risk_contributions(w: np.ndarray, covar: np.ndarray) -> np.ndarray:
    portfolio_vol = np.sqrt(w.T @ covar @ w)
    marginal_risk_contribution = covar @ w.T
    rc = np.multiply(marginal_risk_contribution, w) / portfolio_vol
    return rc


def compute_portfolio_vol(covar: Union[np.ndarray, pd.DataFrame],
                          weights: Union[np.ndarray, pd.Series]
                          ):
    if isinstance(covar, pd.DataFrame):
        covar = covar.to_numpy()
    if isinstance(weights, pd.Series):
        weights = weights.to_numpy()
    return np.sqrt(compute_portfolio_variance(w=weights, covar=covar))


def compute_tre_turnover_stats(covar: np.ndarray,
                               benchmark_weights: pd.Series,
                               weights: pd.Series,
                               weights_0: pd.Series,
                               alphas: pd.Series = None
                               ) -> Tuple[float, float, float, float, float]:
    weight_diff = weights.subtract(benchmark_weights)
    benchmark_vol = np.sqrt(benchmark_weights @ covar @ benchmark_weights.T)
    port_vol = np.sqrt(weights @ covar @ weights.T)
    te_vol = np.sqrt(weight_diff @ covar @ weight_diff.T)
    turnover = np.nansum(np.abs(weights.subtract(weights_0)))
    if alphas is not None:
        port_alpha = alphas @ weights
    else:
        port_alpha = 0.0
    return te_vol, turnover, port_alpha, port_vol, benchmark_vol


def calculate_diversification_ratio(w: np.ndarray, covar: np.ndarray) -> float:
    avg_weighted_vol = np.sqrt(np.diag(covar)) @ w.T
    portfolio_vol = np.sqrt(compute_portfolio_variance(w, covar))
    diversification_ratio = avg_weighted_vol/portfolio_vol
    return diversification_ratio


def compute_portfolio_risk_contribution_outputs(weights: pd.Series,
                                                clean_covar: pd.DataFrame,
                                                risk_budget: Optional[pd.Series] = None
                                                ) -> pd.DataFrame:
    weights = weights.loc[clean_covar.columns]
    asset_rc = compute_portfolio_risk_contributions(weights.to_numpy(), clean_covar.to_numpy())
    asset_rc_ratio = asset_rc / np.nansum(asset_rc)
    if risk_budget is None:
        risk_budget = pd.Series(0.0, index=clean_covar.columns)
    df = pd.concat([pd.Series(weights, index=clean_covar.columns, name='weights'),
                    pd.Series(asset_rc, index=clean_covar.columns, name='risk contribution'),
                    risk_budget.rename('Risk Budget'),
                    pd.Series(asset_rc_ratio, index=clean_covar.columns, name='asset_rc_ratio')
                    ], axis=1)
    return df

