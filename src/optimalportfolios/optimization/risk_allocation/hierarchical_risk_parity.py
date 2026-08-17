"""Hierarchical risk parity for a caller-supplied clustering linkage."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import leaves_list


def _validate_covar(covar: pd.DataFrame) -> None:
    """Require a finite labelled symmetric covariance with positive diagonal."""
    if not isinstance(covar, pd.DataFrame):
        raise TypeError("covar must be a pandas DataFrame")
    if covar.empty or covar.shape[0] != covar.shape[1]:
        raise ValueError(f"covar must be non-empty and square, got {covar.shape!r}")
    if not covar.index.equals(covar.columns) or not covar.index.is_unique:
        raise ValueError("covar index and columns must be identical unique asset labels")
    values = covar.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("covar must contain only finite values")
    if not np.allclose(values, values.T, atol=1e-12, rtol=1e-12):
        raise ValueError("covar must be symmetric")
    if np.any(np.diag(values) <= 0.0):
        raise ValueError("covar diagonal must be strictly positive")


def _validate_linkage(linkage: np.ndarray, n_assets: int) -> np.ndarray:
    """Return a validated SciPy linkage array for ``n_assets`` leaves."""
    values = np.asarray(linkage, dtype=float)
    expected = (n_assets - 1, 4)
    if values.shape != expected:
        raise ValueError(f"linkage must have shape {expected}, got {values.shape!r}")
    if not np.isfinite(values).all():
        raise ValueError("linkage must contain only finite values")
    if n_assets > 1:
        if np.any(values[:, 2] < 0.0):
            raise ValueError("invalid linkage: distances must be non-negative")
        children = values[:, :2]
        if not np.equal(children, np.floor(children)).all():
            raise ValueError("invalid linkage: child identifiers must be integers")
        child_ids = children.astype(int)
        cluster_sizes = {leaf: 1 for leaf in range(n_assets)}
        for row, (left, right) in enumerate(child_ids):
            next_cluster = n_assets + row
            if left == right or left < 0 or right < 0:
                raise ValueError("invalid linkage: every merge needs two distinct children")
            if left >= next_cluster or right >= next_cluster:
                raise ValueError("invalid linkage: a merge references an unavailable cluster")
            expected_size = cluster_sizes[left] + cluster_sizes[right]
            if values[row, 3] != expected_size:
                raise ValueError("invalid linkage: cluster observation count is inconsistent")
            cluster_sizes[next_cluster] = expected_size
        expected_children = np.arange(2 * n_assets - 2)
        if not np.array_equal(np.sort(child_ids.ravel()), expected_children):
            raise ValueError("invalid linkage: a child is missing or reused")
    return values


def _inverse_variance_weights(covar: pd.DataFrame) -> pd.Series:
    """Return unit-sum inverse-variance weights for one covariance block."""
    inverse = 1.0 / np.diag(covar.to_numpy(dtype=float))
    inverse /= inverse.sum()
    return pd.Series(inverse, index=covar.index, name="weight")


def _cluster_variance(covar: pd.DataFrame, assets: list[object]) -> float:
    """Return the variance of the inverse-variance portfolio over ``assets``."""
    block = covar.loc[assets, assets]
    weights = _inverse_variance_weights(block).to_numpy()
    variance = float(weights @ block.to_numpy(dtype=float) @ weights)
    if not np.isfinite(variance) or variance <= 0.0:
        raise ValueError("HRP cluster variances must be positive and finite")
    return variance


def compute_hierarchical_risk_parity_weights(
        covar: pd.DataFrame,
        linkage: np.ndarray,
        ) -> pd.Series:
    """Compute canonical hierarchical risk parity weights.

    The linkage is supplied by the caller rather than estimated here. This keeps the
    boundary explicit: FactorLasso or another clustering package owns the dependence
    transformation and tree construction, while OptimalPortfolios owns allocation.

    The linkage leaf order quasi-diagonalises the covariance matrix. Ordered blocks are
    then split recursively, with capital divided inversely to the two child-block
    variances. Block variance is measured using the inverse-variance portfolio, matching
    the original volatility-HRP construction.

    Args:
        covar: Labelled covariance matrix in caller-supplied variance units.
        linkage: SciPy-compatible hierarchical linkage over the covariance assets.

    Returns:
        Fully invested long-only HRP weights indexed like ``covar``.
    """
    _validate_covar(covar)
    n_assets = len(covar)
    linkage = _validate_linkage(linkage, n_assets)
    if n_assets == 1:
        return pd.Series(1.0, index=covar.index, name="weight")

    ordered = covar.index[leaves_list(linkage)].tolist()
    weights = pd.Series(1.0, index=covar.index, name="weight")
    blocks: list[list[object]] = [ordered]
    while blocks:
        children = []
        for block in blocks:
            if len(block) <= 1:
                continue
            midpoint = len(block) // 2
            left = block[:midpoint]
            right = block[midpoint:]
            left_variance = _cluster_variance(covar, left)
            right_variance = _cluster_variance(covar, right)
            denominator = left_variance + right_variance
            left_share = right_variance / denominator
            weights.loc[left] *= left_share
            weights.loc[right] *= 1.0 - left_share
            children.extend([left, right])
        blocks = children

    weights /= weights.sum()
    return weights
