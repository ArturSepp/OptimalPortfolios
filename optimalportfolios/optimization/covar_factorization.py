"""Numerically controlled covariance factorization for CVXPY optimizers.

The optimization layer frequently receives positive-semidefinite factor-model
covariances with tiny negative eigenvalues from floating-point reconstruction.
This module turns such a covariance into a symmetric positive-definite matrix
and an explicit square root ``B`` satisfying ``B @ B.T == covar``.  Passing
``B`` into CVXPY lets tracking-error terms use norms and sums of squares rather
than asking each ``quad_form`` atom to factor the same ill-conditioned matrix.

Each supported low-level CVXPY solver calls :func:`factorize_covariance` once
after its wrapper has filtered the covariance, then reuses the resulting
:class:`CovarianceFactorization` through the objective, constraints, post-solve
validation, and ROSAA reporting. The container also records raw and stabilized
conditioning telemetry for production diagnostics.

The stabilized covariance keeps the input variance units and its factor ``B`` has the
corresponding volatility units; no frequency conversion or weight normalisation occurs here.
Main entry points are ``factorize_covariance`` and ``CovarianceFactorization``. Boundary:
covariance estimation, optimisation objectives, and solver acceptance policy are out of scope.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


DEFAULT_EIGENVALUE_FLOOR = 1e-10
DEFAULT_NEGATIVE_EIGENVALUE_TOLERANCE = 1e-10
DEFAULT_RECONSTRUCTION_RTOL = 1e-10


@dataclass(frozen=True)
class CovarianceFactorization:
    """A stabilized covariance and its explicit square-root factor.

    ``factor`` has shape ``(n_assets, n_risk_directions)`` and satisfies
    ``factor @ factor.T == covar`` to ``reconstruction_rtol``.  The current
    eigenvalue-floor policy keeps every direction, so the factor is square;
    the rectangular contract permits a future exact low-rank representation.

    Attributes:
        covar: Symmetric stabilized covariance used by the solver.
        factor: Matrix ``B`` satisfying ``B @ B.T == covar``.
        raw_min_eigenvalue: Smallest eigenvalue before stabilization.
        raw_condition_number: Condition number before stabilization.
        stabilized_min_eigenvalue: Smallest eigenvalue after flooring.
        stabilized_condition_number: Condition number after flooring.
        n_eigenvalues_floored: Number of eigenvalues raised to the floor.
        max_eigenvalue_adjustment: Largest absolute flooring adjustment.
    """

    covar: np.ndarray
    factor: np.ndarray
    raw_min_eigenvalue: float = float('nan')
    raw_condition_number: float = float('nan')
    stabilized_min_eigenvalue: float = float('nan')
    stabilized_condition_number: float = float('nan')
    n_eigenvalues_floored: int = 0
    max_eigenvalue_adjustment: float = 0.0

    def __post_init__(self) -> None:
        """Validate shape, finiteness and the ``factor @ factor.T == covar`` identity.

        Raises:
            ValueError: If the covariance is not square, the factor has the wrong
                number of rows, either input is non-finite, or the reconstruction
                misses ``covar`` by more than ``DEFAULT_RECONSTRUCTION_RTOL``.
        """
        covar = np.asarray(self.covar, dtype=float)
        factor = np.asarray(self.factor, dtype=float)
        if covar.ndim != 2 or covar.shape[0] != covar.shape[1]:
            raise ValueError(
                f"CovarianceFactorization.covar must be square, got {covar.shape}")
        if factor.ndim != 2 or factor.shape[0] != covar.shape[0]:
            raise ValueError(
                "CovarianceFactorization.factor must have one row per asset, "
                f"got factor={factor.shape}, covar={covar.shape}")
        if not np.all(np.isfinite(covar)) or not np.all(np.isfinite(factor)):
            raise ValueError("CovarianceFactorization inputs must be finite")
        reconstructed = factor @ factor.T
        denominator = max(float(np.linalg.norm(covar, ord='fro')), 1.0)
        relative_error = float(
            np.linalg.norm(reconstructed - covar, ord='fro') / denominator)
        if relative_error > DEFAULT_RECONSTRUCTION_RTOL:
            raise ValueError(
                "CovarianceFactorization.factor does not reconstruct covar: "
                f"relative error {relative_error:.6g} exceeds "
                f"{DEFAULT_RECONSTRUCTION_RTOL:.6g}")
        object.__setattr__(self, 'covar', covar)
        object.__setattr__(self, 'factor', factor)


def factorize_covariance(
        covar: np.ndarray,
        eigenvalue_floor: float = DEFAULT_EIGENVALUE_FLOOR,
        negative_eigenvalue_tolerance: float = DEFAULT_NEGATIVE_EIGENVALUE_TOLERANCE,
) -> CovarianceFactorization:
    """Return a stabilized covariance and factor ``B`` with ``B B.T = Sigma``.

    The input is symmetrized before ``numpy.linalg.eigh``.  Eigenvalues within
    a scale-aware negative tolerance are treated as numerical residue and
    raised to ``eigenvalue_floor``.  A materially negative eigenvalue raises:
    silently making a genuinely indefinite risk model convex would change the
    optimization problem.

    Args:
        covar: Square, finite covariance matrix.
        eigenvalue_floor: Minimum eigenvalue of the covariance used by CVXPY.
        negative_eigenvalue_tolerance: Absolute and relative tolerance for
            accepting floating-point negative eigenvalues.  The effective
            tolerance is this value times ``max(1, max(abs(eigenvalues)))``.
    Returns:
        ``CovarianceFactorization`` containing the stabilized covariance and
        its explicit square-root factor.

    Raises:
        ValueError: If the matrix is empty, non-square, non-finite, materially
            indefinite, or cannot be decomposed/reconstructed to tolerance.
    """
    values = np.asarray(covar, dtype=float)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError(f"covar must be a square matrix, got shape={values.shape}")
    if values.shape[0] == 0:
        raise ValueError("covar must contain at least one asset")
    if not np.all(np.isfinite(values)):
        raise ValueError("covar must contain only finite values")
    if eigenvalue_floor < 0.0:
        raise ValueError("eigenvalue_floor must be non-negative")
    if negative_eigenvalue_tolerance < 0.0:
        raise ValueError("negative_eigenvalue_tolerance must be non-negative")
    symmetric = 0.5 * (values + values.T)
    try:
        eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    except np.linalg.LinAlgError as exc:
        raise ValueError("covariance eigendecomposition did not converge") from exc

    scale = max(1.0, float(np.max(np.abs(eigenvalues))))
    negative_tolerance = negative_eigenvalue_tolerance * scale
    minimum_eigenvalue = float(eigenvalues.min())
    if minimum_eigenvalue < -negative_tolerance:
        raise ValueError(
            "covar is materially indefinite: "
            f"minimum eigenvalue {minimum_eigenvalue:.6g} is below "
            f"-{negative_tolerance:.6g}")

    stabilized_eigenvalues = np.maximum(eigenvalues, eigenvalue_floor)
    factor = eigenvectors * np.sqrt(stabilized_eigenvalues)
    stabilized_covar = (
        eigenvectors @ np.diag(stabilized_eigenvalues) @ eigenvectors.T)
    stabilized_covar = 0.5 * (stabilized_covar + stabilized_covar.T)

    reconstructed = factor @ factor.T
    denominator = max(float(np.linalg.norm(stabilized_covar, ord='fro')), 1.0)
    relative_error = float(
        np.linalg.norm(reconstructed - stabilized_covar, ord='fro') / denominator)
    if (not np.isfinite(relative_error)
            or relative_error > DEFAULT_RECONSTRUCTION_RTOL):
        raise ValueError(
            "covariance factor reconstruction failed: "
            f"relative error {relative_error:.6g} exceeds "
            f"{DEFAULT_RECONSTRUCTION_RTOL:.6g}")

    positive = eigenvalues[eigenvalues > 0.0]
    raw_condition_number = (
        float(eigenvalues.max() / positive.min())
        if len(positive) == len(eigenvalues) and positive.min() > 0.0
        else float('inf')
    )
    stabilized_condition_number = (
        float(stabilized_eigenvalues.max() / stabilized_eigenvalues.min())
        if stabilized_eigenvalues.min() > 0.0 else float('inf')
    )
    adjustment = stabilized_eigenvalues - eigenvalues
    return CovarianceFactorization(
        covar=stabilized_covar,
        factor=factor,
        raw_min_eigenvalue=minimum_eigenvalue,
        raw_condition_number=raw_condition_number,
        stabilized_min_eigenvalue=float(stabilized_eigenvalues.min()),
        stabilized_condition_number=stabilized_condition_number,
        n_eigenvalues_floored=int(np.sum(eigenvalues < eigenvalue_floor)),
        max_eigenvalue_adjustment=float(adjustment.max()),
    )
