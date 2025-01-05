"""
examples of
"""
from __future__ import division

import numpy as np
import pandas as pd
from typing import Tuple, Union
from numba import njit


@njit
def calculate_portfolio_var(w: np.ndarray, covar: np.ndarray) -> float:
    return w.T @ covar @ w


@njit
def calculate_risk_contribution(w: np.ndarray, covar: np.ndarray) -> np.ndarray:
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
    return np.sqrt(calculate_portfolio_var(w=weights, covar=covar))


def compute_te_turnover(covar: np.ndarray,
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
    portfolio_vol = np.sqrt(calculate_portfolio_var(w, covar))
    diversification_ratio = avg_weighted_vol/portfolio_vol
    return diversification_ratio
