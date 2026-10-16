"""Residual-alpha capacity and breadth conditional on a fitted diagonal risk model.

The shared GLS rank rule removes factor loadings and optional class/funding
controls. Residual correlation is a descriptive input, separate from the
covariance of estimated alpha. Neither input is estimated in this module.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from optimalportfolios.matf.gls_diagnostic._inference import (
    _array, _axis, _vector, compute_gls_geometry,
)


@dataclass(frozen=True)
class ResidualOpportunity:
    """Conditional residual opportunity diagnostics.

    Attributes:
        summary (dict):
            Dimension, rank, squared residual-alpha capacity, conditional noise
            floor and raw/repaired effective breadth, with repair diagnostics.
        orthogonal_alpha (ndarray):
            Alpha after removing factor loadings and optional controls.
        neutral_projector (ndarray):
            Symmetric residual projector in variance-whitened asset coordinates.
        residual_correlation (ndarray):
            Supplied correlation, or its explicitly requested one-pass repair.
    """

    summary: dict
    orthogonal_alpha: np.ndarray
    neutral_projector: np.ndarray
    residual_correlation: np.ndarray


def _matrix(value, B, name):
    """Validate an asset-by-asset matrix without changing labels or ordering."""
    labels = B.index if isinstance(B, pd.DataFrame) else None
    _axis(value, labels)
    if isinstance(value, pd.DataFrame):
        _axis(value, value.index if labels is None else labels, axis=1)
    matrix = _array(value, name, 2)
    if matrix.shape != (len(B), len(B)) or not np.allclose(
            matrix, matrix.T, rtol=0, atol=1e-12):
        raise ValueError(f"{name} must be a symmetric asset-by-asset matrix")
    return matrix


def compute_residual_opportunity(B, residual_variances, alpha, *, controls=None,
                                 alpha_covariance=None, residual_correlation=None,
                                 repair_correlation=False, rcond=None):
    """Measure residual capacity, estimation-noise floor and effective breadth.

    Args:
        B (array-like, shape (n, k)):
            Fitted factor loadings, in the exact asset order of all other inputs.
        residual_variances (array-like, shape (n,)):
            Strictly positive residual-return variances D; the risk model is diagonal.
        alpha (array-like, shape (n,)):
            Residual means in units consistent with D. No annualization is applied.
        controls (array-like, shape (n, c), optional):
            Additional columns projected out, such as class indicators or funding.
            No intercept or class centering is added implicitly.
        alpha_covariance (array-like, shape (n, n), optional):
            Positive-semidefinite estimation-error covariance V, not residual-return
            covariance. The conditional floor is trace(M D^-1/2 V D^-1/2).
        residual_correlation (array-like, shape (n, n), optional):
            Unit-diagonal residual-return correlation R; defaults to identity.
            Cadence blocks and missing-history policy are the caller's responsibility.
        repair_correlation (bool, default False):
            Explicitly allow one eigenvalue clip at zero followed by unit-diagonal
            rescaling. No shrinkage or iteration. Otherwise indefinite R is rejected.
        rcond (float or None, default None):
            Relative SVD cutoff passed to compute_gls_geometry.

    Returns:
        ResidualOpportunity:
            Capacity is ||M D^-1/2 alpha||^2. Effective breadth is the participation
            ratio of M R M. R=I gives n-rank; an exhausted subspace gives zero.
            The signed capacity-minus-floor is retained, without truncation or CI.
            These are universe descriptors, not constrained portfolio Sharpe ratios.
    """
    if not isinstance(repair_correlation, (bool, np.bool_)):
        raise ValueError("repair_correlation must be boolean")
    d = _vector(residual_variances, B, "residual_variances")
    a = _vector(alpha, B, "alpha")
    geometry = compute_gls_geometry(B, d, rcond=rcond)
    design = geometry.beta
    if controls is not None:
        _axis(controls, B.index if isinstance(B, pd.DataFrame) else None)
        if isinstance(controls, pd.DataFrame):
            _axis(controls, axis=1)
        extra = _array(controls, "controls", 2)
        if len(extra) != len(B):
            raise ValueError("controls must have one row per asset")
        design = np.column_stack([design, extra])
        geometry = compute_gls_geometry(design, d, rcond=rcond)
    scale = np.sqrt(d)
    n = len(d)
    dimension = n - geometry.rank
    M = geometry.residual_projector / scale[:, None] * scale[None, :]
    if dimension == 0:
        M = np.zeros((n, n))
    whitened = M @ (a / scale)
    capacity = float(whitened @ whitened)
    floor = np.nan
    if alpha_covariance is not None:
        V = _matrix(alpha_covariance, B, "alpha_covariance")
        if np.linalg.eigvalsh(V).min() < -1e-12 * max(1., np.linalg.norm(V, 2)):
            raise ValueError("alpha_covariance must be positive semidefinite")
        floor = float(np.sum(M * (V / np.outer(scale, scale)).T))
    R = np.eye(n) if residual_correlation is None else _matrix(
        residual_correlation, B, "residual_correlation")
    if not np.allclose(np.diag(R), 1., rtol=0, atol=1e-12) or np.max(np.abs(R)) > 1 + 1e-12:
        raise ValueError("residual_correlation must have unit diagonal and entries in [-1, 1]")
    eigenvalues, vectors = np.linalg.eigh(R)
    negative = eigenvalues[eigenvalues < 0.]
    if eigenvalues.min() < -1e-10 and not repair_correlation:
        raise ValueError(
            "indefinite residual_correlation requires explicit repair_correlation=True")
    fixed = R
    if repair_correlation:
        fixed = (vectors * np.maximum(eigenvalues, 0.)) @ vectors.T
        diagonal = np.sqrt(np.diag(fixed))
        fixed = fixed / np.outer(diagonal, diagonal)
    def breadth(correlation):
        """Evaluate the projected-spectrum participation ratio, retaining raw diagnostics."""
        projected = M @ correlation @ M
        denominator = float(np.sum(projected * projected.T))
        numerical_zero = (np.finfo(float).eps * n * np.linalg.norm(correlation, 'fro'))**2
        return (float(np.trace(projected)**2 / denominator)
                if dimension and denominator > numerical_zero else 0.)
    summary = dict(N=n, rank=geometry.rank, neutral_dimension=dimension,
                   capacity=capacity, noise_floor=floor, excess_capacity=capacity-floor,
                   breadth_raw=breadth(R), breadth=breadth(fixed),
                   correlation_repaired=bool(repair_correlation),
                   n_clipped=int(len(negative)) if repair_correlation else 0,
                   sum_clipped=float(-negative.sum()) if repair_correlation else 0.,
                   frobenius_rel=float(np.linalg.norm(fixed-R, 'fro') / np.linalg.norm(R, 'fro')),
                   minimum_correlation_eigenvalue=float(eigenvalues.min()),
                   minimum_repaired_eigenvalue=float(np.linalg.eigvalsh(fixed).min()))
    return ResidualOpportunity(summary, whitened*scale, M, fixed)
