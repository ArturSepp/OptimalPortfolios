from __future__ import annotations

from typing import Union, Optional

import numpy as np
import pandas as pd
import qis as qis


def squeeze_covariance_matrix(covar: Union[np.ndarray, pd.DataFrame],
                              squeeze_factor: Optional[float] = 0.05,
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
        return covar

    # need to create pd.Dataframe for keeping track of good indices
    if isinstance(covar, pd.DataFrame):
        cov_matrix_pd = covar.copy()
    else:
        cov_matrix_pd = pd.DataFrame(covar)

    # filter out nans and zero variances
    variances = np.diag(cov_matrix_pd.to_numpy())
    is_good_asset = np.where(np.logical_and(np.greater(variances, 0.0), np.isnan(variances) == False))
    good_tickers = cov_matrix_pd.columns[is_good_asset]
    clean_covar_pd = cov_matrix_pd.loc[good_tickers, good_tickers]
    clean_covar_np = clean_covar_pd.to_numpy()

    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(clean_covar_np)

    # Squeeze smaller eigenvalues (simple threshold-based squeezing)
    squeezed_eigenvalues = np.array([np.maximum(eigenvalue, squeeze_factor * np.max(eigenvalues))
                                     for eigenvalue in eigenvalues])

    # Reconstruct squeezed covariance matrix
    squeezed_cov_matrix = eigenvectors @ np.diag(squeezed_eigenvalues) @ eigenvectors.T

    if is_preserve_variance:
        # adjustment should be applied to off-dioagonal elements too otherwise we may end up with noncosistent matrix
        original_variance = np.diag(clean_covar_np)
        squeezed_variance = np.diag(squeezed_cov_matrix)
        adjustment_ratio = np.sqrt(original_variance / squeezed_variance)
        norm = np.outer(adjustment_ratio, adjustment_ratio)
        squeezed_cov_matrix = norm*squeezed_cov_matrix

    # now extend back
    squeezed_cov_matrix_pd = pd.DataFrame(squeezed_cov_matrix, index=good_tickers, columns=good_tickers)
    # reindex for all tickers and fill nans with zeros
    all_tickers = cov_matrix_pd.columns
    squeezed_cov_matrix = squeezed_cov_matrix_pd.reindex(index=all_tickers).reindex(columns=all_tickers).fillna(0.0)

    if isinstance(covar, np.ndarray):  # match return to original type
        squeezed_cov_matrix = squeezed_cov_matrix.to_numpy()
    return squeezed_cov_matrix


def compute_returns_from_prices(prices: pd.DataFrame,
                                returns_freq: str = 'ME',
                                demean: bool = True,
                                span: Optional[int] = 52
                                ) -> pd.DataFrame:
    """
    compute returns for covar matrix estimation
    """
    returns = qis.to_returns(prices=prices, is_log_returns=True, drop_first=True, freq=returns_freq)
    if demean:
        returns = returns - qis.compute_ewm(returns, span=span)
        # returns.iloc[0, :] will be zero so shift the period
        returns = returns.iloc[1:, :]
    return returns
