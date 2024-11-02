"""
some utilities for covariance matrices
"""
import pandas as pd
import numpy as np
from typing import Union


def squeeze_covariance_matrix(cov_matrix: Union[np.ndarray, pd.DataFrame],
                              squeeze_factor: float = 0.5,
                              is_preserve_variance: bool = True
                              ) -> Union[np.ndarray, pd.DataFrame]:
    """
    Adjusts the covariance matrix by applying a squeezing factor to eigenvalues.
    Smaller eigenvalues are reduced to mitigate noise.
    for the methodology see SSRN paper
    Squeezing Financial Noise: A Novel Approach to Covariance Matrix Estimation
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4986939
    """
    if np.isclose(squeeze_factor, 0.0):
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
