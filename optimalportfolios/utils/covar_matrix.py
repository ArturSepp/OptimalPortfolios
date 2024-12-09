"""
some utilities for estimation of covariance matrices
"""
import pandas as pd
import numpy as np
import qis as qis
from typing import Union, Optional, Dict

from optimalportfolios.utils.lasso import estimate_lasso_covar


def estimate_rolling_ewma_covar(prices: pd.DataFrame,
                                time_period: qis.TimePeriod,  # when we start building portfolios
                                returns_freq: str = 'W-WED',
                                rebalancing_freq: str = 'QE',
                                span: int = 52
                                ) -> Dict[pd.Timestamp, pd.DataFrame]:
    """
    maximise portfolio alpha subject to constraint on tracking tracking error
    """
    # compute ewma covar with fill nans in covar using zeros
    returns = qis.to_returns(prices=prices, is_log_returns=True, drop_first=True, freq=returns_freq)
    returns_np = returns.to_numpy()
    x = returns_np - qis.compute_ewm(returns_np, span=span)
    covar_tensor_txy = qis.compute_ewm_covar_tensor(a=x, span=span, nan_backfill=qis.NanBackfill.ZERO_FILL)
    an_factor = qis.infer_an_from_data(data=returns)

    # create rebalancing schedule
    rebalancing_schedule = qis.generate_rebalancing_indicators(df=returns, freq=rebalancing_freq)

    tickers = prices.columns.to_list()
    y_covars = {}
    for idx, (date, value) in enumerate(rebalancing_schedule.items()):
        if value and date >= time_period.start:
            y_covars[date] = pd.DataFrame(an_factor*covar_tensor_txy[idx], index=tickers, columns=tickers)
    return y_covars


def estimate_rolling_lasso_covar(benchmark_prices: pd.DataFrame,
                                 prices: pd.DataFrame,
                                 time_period: qis.TimePeriod,  # when we start building portfolios
                                 returns_freq: str = 'W-WED',
                                 rebalancing_freq: str = 'QE',
                                 span: int = 52,  # 1y of weekly returns
                                 reg_lambda: float = 1e-8,
                                 squeeze_factor: Optional[float] = None
                                 ) -> Dict[pd.Timestamp, pd.DataFrame]:
    """
    use benchmarks to compute the benchmark covar matrix
    use lasso to estimate betas
    compute covar matrix
    """
    # compute ewma covar with fill nans in covar using zeros
    y_returns = qis.to_returns(prices=prices, is_log_returns=True, drop_first=True, freq=returns_freq)
    x_returns = qis.to_returns(prices=benchmark_prices, is_log_returns=True, drop_first=True, freq=returns_freq)
    x_returns_np = x_returns.to_numpy()
    x = x_returns_np - qis.compute_ewm(x_returns_np, span=span)
    covar_tensor_txy = qis.compute_ewm_covar_tensor(a=x, span=span, nan_backfill=qis.NanBackfill.ZERO_FILL)
    an_factor = qis.infer_an_from_data(data=x_returns)

    # generate rebalancing dates on the returns index
    rebalancing_schedule = qis.generate_rebalancing_indicators(df=x_returns, freq=rebalancing_freq)

    tickers = prices.columns.to_list()
    y_covars = {}
    for idx, (date, value) in enumerate(rebalancing_schedule.items()):
        if value and date >= time_period.start:
            x_covar = squeeze_covariance_matrix(covar_tensor_txy[idx], squeeze_factor=squeeze_factor)
            y_covar = estimate_lasso_covar(x=x_returns.loc[:date, :],
                                           y=y_returns.loc[:date, :],
                                           covar=x_covar,
                                           reg_lambda=reg_lambda,
                                           span=span)
            y_covars[date] = pd.DataFrame(an_factor*y_covar, index=tickers, columns=tickers)
    return y_covars


def squeeze_covariance_matrix(cov_matrix: Union[np.ndarray, pd.DataFrame],
                              squeeze_factor: Optional[float] = 0.5,
                              is_preserve_variance: bool = True
                              ) -> Union[np.ndarray, pd.DataFrame]:
    """
    Adjusts the covariance matrix by applying a squeezing factor to eigenvalues.
    Smaller eigenvalues are reduced to mitigate noise.
    for the methodology see SSRN paper
    Squeezing Financial Noise: A Novel Approach to Covariance Matrix Estimation
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4986939
    """
    if squeeze_factor is None or np.isclose(squeeze_factor, 0.0):
        return cov_matrix

    if isinstance(cov_matrix, pd.DataFrame):
        cov_matrix_np = cov_matrix.to_numpy()
    else:
        cov_matrix_np = cov_matrix

    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix_np)

    # Squeeze smaller eigenvalues (simple threshold-based squeezing)
    squeezed_eigenvalues = np.array([max(eigenvalue, squeeze_factor * max(eigenvalues))
                                     for eigenvalue in eigenvalues])

    # Reconstruct squeezed covariance matrix
    squeezed_cov_matrix = eigenvectors @ np.diag(squeezed_eigenvalues) @ eigenvectors.T

    if is_preserve_variance:
        # adjustment should be applied to off-dioagonal elements too otherwise we may end up with noncosistent matrix
        original_variance = np.diag(cov_matrix)
        squeezed_variance = np.diag(squeezed_cov_matrix)
        adjustment_ratio = np.sqrt(original_variance / squeezed_variance)
        norm = np.outer(adjustment_ratio, adjustment_ratio)
        squeezed_cov_matrix = norm*squeezed_cov_matrix

    if isinstance(cov_matrix, pd.DataFrame):
        squeezed_cov_matrix = pd.DataFrame(squeezed_cov_matrix, index=cov_matrix.index, columns=cov_matrix.columns)

    return squeezed_cov_matrix

