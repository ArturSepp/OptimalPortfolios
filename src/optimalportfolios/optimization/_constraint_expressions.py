"""Shared CVXPY expressions used by portfolio constraint compilers.

This module centralises covariance-risk and objective-composition expressions
without defining constraint policy or solver orchestration. Inputs retain their
caller-supplied covariance and volatility units; no resampling, annualisation,
or weight normalisation occurs here.
"""
from __future__ import annotations

from typing import Optional, Union

import cvxpy as cvx
import numpy as np
from cvxpy.atoms.affine.add_expr import AddExpression
from cvxpy.atoms.affine.wraps import psd_wrap

from optimalportfolios.optimization.covar_factorization import CovarianceFactorization


def _cvx_factor_risk(
        active_weights,
        covar_factorization: CovarianceFactorization,
):
    """Map an affine weight vector into covariance-factor risk coordinates.

    Args:
        active_weights: CVXPY affine weight or active-weight vector.
        covar_factorization: Precomputed covariance square root.

    Returns:
        ``factor.T @ active_weights`` in Euclidean risk coordinates.

    Raises:
        ValueError: If the factor and weight-vector dimensions differ.
    """
    n_assets = int(active_weights.shape[0])
    if covar_factorization.factor.shape[0] != n_assets:
        raise ValueError(
            "covar_factorization dimension does not match the weight vector: "
            f"factor={covar_factorization.factor.shape}, assets={n_assets}")
    return covar_factorization.factor.T @ active_weights


def cvx_covar_variance(
        active_weights,
        covar: Union[np.ndarray, psd_wrap],
        covar_factorization: Optional[CovarianceFactorization] = None,
):
    """Return ``active_weights.T @ covar @ active_weights`` as a CVXPY atom.

    When an explicit covariance factor is supplied, the mathematically
    equivalent sum-of-squares form avoids repeated factorization of an
    ill-conditioned matrix inside CVXPY's ``quad_form`` canonicalizer.

    Args:
        active_weights: CVXPY affine vector of portfolio or active weights.
        covar: Covariance matrix or ``psd_wrap`` used when no factorization is
            supplied.
        covar_factorization: Optional precomputed covariance square root.

    Returns:
        A scalar convex CVXPY variance expression.

    Raises:
        ValueError: If neither covariance representation is supplied.
    """
    if covar_factorization is not None:
        return cvx.sum_squares(
            _cvx_factor_risk(active_weights, covar_factorization))
    if covar is None:
        raise ValueError("covar must be supplied when covar_factorization is None")
    return cvx.quad_form(active_weights, covar)


def add_term_to_objective_function(objective_fun: AddExpression, term: AddExpression) -> AddExpression:
    """Safely add a term to CVXPY objective function, handling None cases.

    Args:
        objective_fun: Existing objective function (may be None).
        term: New term to add (may be None).

    Returns:
        Updated objective function.
    """
    if objective_fun is None:
        if term is not None:
            objective_fun = term
    else:
        if term is not None:
            objective_fun += term
    return objective_fun
