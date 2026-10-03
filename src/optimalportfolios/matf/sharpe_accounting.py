"""Annualized excess-Sharpe accounting for any linear factor model.

MATF is the motivating factor set; these functions impose no particular factor names.
Inputs must share units, factor order and asset order. Residual covariance is positive
definite, supplied as a full matrix or a diagonal variance vector. Returned arrays are
in input order. No forecasts, histories, annualization or covariance repairs are inferred.

The functions implement the achievable-Sharpe and GLS identities of Sepp and Kastenholz
(2026). qis.RiskModel owns risk contributions and tracking-error reporting; the closed-form
access, forecast and fixed-book completion identities here are absent from its API.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
import pandas as pd


def _array(value, name, ndim):
    """Validate a nonempty finite real array of the requested dimension."""
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    result = np.asarray(value, dtype=float)
    if result.ndim != ndim or not all(result.shape) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a nonempty finite {ndim}-dimensional array")
    return result


def _align(value, labels, axis=0):
    """Reject duplicated or reordered labelled inputs rather than reindexing."""
    if isinstance(value, (pd.Series, pd.DataFrame)):
        actual = value.index if axis == 0 else value.columns
        if actual.has_duplicates or (labels is not None and not actual.equals(labels)):
            raise ValueError("label/order mismatch or duplicate labels")


def _covariance(value, size, name):
    """Validate an SPD covariance without clipping or regularizing it."""
    result = _array(value, name, 2)
    if result.shape != (size, size) or not np.allclose(result, result.T, rtol=0, atol=1e-12):
        raise ValueError(f"{name} must be a symmetric {size} by {size} matrix")
    try:
        np.linalg.cholesky(result)
    except np.linalg.LinAlgError as error:
        raise ValueError(f"{name} must be positive definite") from error
    return result


def _factors(lam, Sig_F):
    """Validate premia and factor covariance, including labelled axes."""
    labels = lam.index if isinstance(lam, pd.Series) else None
    _align(lam, labels)
    _align(Sig_F, labels)
    _align(Sig_F, Sig_F.index if isinstance(Sig_F, pd.DataFrame) else labels, axis=1)
    premia = _array(lam, "lam", 1)
    return premia, _covariance(Sig_F, len(premia), "Sig_F")


def _loadings(B, Omega):
    """Validate loadings/residual covariance and return whitened loadings."""
    _align(B, None)
    _align(B, None, axis=1)
    assets = B.index if isinstance(B, pd.DataFrame) else None
    _align(Omega, assets)
    _align(Omega, assets, axis=1) if isinstance(Omega, pd.DataFrame) else None
    beta = _array(B, "B", 2)
    if np.iscomplexobj(Omega):
        raise ValueError("Omega must be real")
    residual = np.asarray(Omega, dtype=float)
    if residual.ndim == 1:
        residual = np.diag(_array(Omega, "Omega", 1))
    residual = _covariance(residual, beta.shape[0], "Omega")
    chol = np.linalg.cholesky(residual)
    return beta, residual, chol, np.linalg.solve(chol, beta)


def _model(lam, Sig_F, B, Omega):
    """Validate all factor-model inputs without changing their order."""
    premia, factor = _factors(lam, Sig_F)
    beta, residual, chol, white = _loadings(B, Omega)
    labels = B.columns if isinstance(B, pd.DataFrame) else None
    _align(lam, labels)
    _align(Sig_F, labels)
    _align(Sig_F, labels, axis=1) if isinstance(Sig_F, pd.DataFrame) else None
    if beta.shape[1] != len(premia):
        raise ValueError("B factor dimension does not match lam")
    return premia, factor, beta, residual, chol, white


def _vector(value, B, name):
    """Validate a vector on the loadings' asset axis."""
    _align(value, B.index if isinstance(B, pd.DataFrame) else None)
    result = _array(value, name, 1)
    if len(result) != len(B):
        raise ValueError(f"{name} asset dimension does not match B")
    return result


def factor_ceiling(lam, Sig_F):
    """Return the annualized squared excess Sharpe with unrestricted factor access."""
    premia, factor = _factors(lam, Sig_F)
    return float(premia @ np.linalg.solve(factor, premia))


def achievable_sharpe2(lam, Sig_F, B, Omega):
    """Return systematic squared Sharpe in the asset universe, including deficient rank."""
    premia, factor, _, _, _, white = _model(lam, Sig_F, B, Omega)
    precision = np.linalg.solve(factor, np.eye(len(premia)))
    z = precision @ premia
    return float(premia @ z - z @ np.linalg.solve(precision + white.T @ white, z))


def fpir(lam, Sig_F, B, Omega):
    """Return achievable squared Sharpe divided by the nonzero factor ceiling."""
    ceiling = factor_ceiling(lam, Sig_F)
    if ceiling == 0:
        raise ValueError("FPIR is undefined for zero factor premia")
    return achievable_sharpe2(lam, Sig_F, B, Omega) / ceiling


def gls_projector(B, Omega):
    """Return the rank-safe forecast residual projector in the inverse-Omega metric."""
    beta, _, chol, white = _loadings(B, Omega)
    return np.eye(len(beta)) - beta @ np.linalg.pinv(white) @ np.linalg.solve(
        chol, np.eye(len(beta)))


def gls_split(m, B, Omega):
    """Return minimum-norm GLS factor premia and orthogonal forecast residual."""
    beta, _, chol, white = _loadings(B, Omega)
    mean = _vector(m, B, "m")
    premia = np.linalg.lstsq(white, np.linalg.solve(chol, mean), rcond=None)[0]
    return premia, mean - beta @ premia


def holdings_decomposition(w, B, Omega):
    """Return factor-replicating holdings, factor-neutral holdings and book exposures."""
    beta, _, chol, white = _loadings(B, Omega)
    weights = _vector(w, B, "w")
    exposure = beta.T @ weights
    mimicking = np.linalg.solve(chol.T, np.linalg.pinv(white).T)
    factor_weights = mimicking @ exposure
    return factor_weights, weights - factor_weights, exposure


def hurdle_premia(lam, Sig_F):
    """Return each hurdle premium, conditional variance and drop-one squared-Sharpe gain."""
    premia, factor = _factors(lam, Sig_F)
    precision = np.linalg.solve(factor, np.eye(len(premia)))
    variance = 1.0 / np.diag(precision)
    z = precision @ premia
    return premia - variance * z, variance, variance * z**2


def ceiling_premium_correlation_split(lam, Sig_F):
    """Return additive standalone-premium and correlation contributions by factor."""
    premia, factor = _factors(lam, Sig_F)
    standalone = premia**2 / np.diag(factor)
    return standalone, premia * np.linalg.solve(factor, premia) - standalone


def factor_mimicking_portfolios(B, Omega):
    """Return unit-exposure GLS portfolios; reject factors that cannot all be spanned."""
    _, _, chol, white = _loadings(B, Omega)
    if np.linalg.matrix_rank(white) != white.shape[1]:
        raise ValueError("unit factor exposures are unattainable: B lacks full column rank")
    return np.linalg.solve(chol.T, np.linalg.pinv(white).T)


def fixed_exposure_fpir(e, Sig_F, B, Omega):
    """Return systematic-risk share for an attainable nonzero factor exposure target."""
    beta, residual, chol, white = _loadings(B, Omega)
    labels = B.columns if isinstance(B, pd.DataFrame) else None
    _align(e, labels)
    _align(Sig_F, labels)
    _align(Sig_F, labels, axis=1) if isinstance(Sig_F, pd.DataFrame) else None
    exposure = _array(e, "e", 1)
    factor = _covariance(Sig_F, beta.shape[1], "Sig_F")
    if len(exposure) != beta.shape[1]:
        raise ValueError("e factor dimension does not match B")
    weights = np.linalg.solve(chol.T, np.linalg.pinv(white).T) @ exposure
    if not np.allclose(beta.T @ weights, exposure, rtol=1e-10, atol=1e-12):
        raise ValueError("target exposure is unattainable")
    systematic = float(exposure @ factor @ exposure)
    if systematic == 0:
        raise ValueError("FPIR is undefined for zero exposure")
    return systematic / (systematic + float(weights @ residual @ weights))


def tangency_access_split(lam, Sig_F, B, Omega):
    """Return per-factor tangency contributions split into factor and access risk."""
    premia, factor, beta, residual, chol, white = _model(lam, Sig_F, B, Omega)
    covariance = beta @ factor @ beta.T + residual
    exposure = beta.T @ np.linalg.solve(covariance, beta @ premia)
    mimicking = np.linalg.solve(chol.T, np.linalg.pinv(white).T)
    access = mimicking.T @ residual @ mimicking
    return exposure * (factor @ exposure), exposure * (access @ exposure)


def overlay_value(lam, Sig_F, B, Omega, a):
    """Return full-factor overlay gain and augmented capacity for m = B lam + a."""
    premia, factor, beta, residual, _, white = _model(lam, Sig_F, B, Omega)
    alpha = _vector(a, B, "a")
    z = np.linalg.solve(factor, premia)
    g = beta.T @ np.linalg.solve(residual, alpha)
    h = np.linalg.solve(factor, np.eye(len(premia))) + white.T @ white
    gain = float((z - g) @ np.linalg.solve(h, z - g))
    capacity = float(premia @ z + alpha @ np.linalg.solve(residual, alpha))
    return gain, capacity


def _hedge_indices(hedgeable, size):
    """Resolve unique factor indices or a boolean mask; None means all factors."""
    if hedgeable is None:
        return np.arange(size)
    values = np.asarray(hedgeable)
    if values.ndim != 1:
        raise ValueError("hedgeable must be a one-dimensional index sequence or mask")
    if values.dtype == bool:
        if len(values) != size:
            raise ValueError("hedgeable mask has the wrong length")
        return np.flatnonzero(values)
    if values.size == 0:
        return np.array([], dtype=int)
    if not np.issubdtype(values.dtype, np.integer):
        raise ValueError("hedgeable indices must be integers")
    if np.any(values < 0) or np.any(values >= size) or len(np.unique(values)) != len(values):
        raise ValueError("hedgeable indices are duplicated or outside the factor axis")
    return values.astype(int)


def partial_overlay_value(lam, Sig_F, B, Omega, a, hedgeable):
    """Return overlay gain and augmented capacity when asset weights may be reoptimized."""
    premia, factor, beta, residual, _, _ = _model(lam, Sig_F, B, Omega)
    mean = beta @ premia + _vector(a, B, "a")
    covariance = beta @ factor @ beta.T + residual
    baseline = float(mean @ np.linalg.solve(covariance, mean))
    selected = _hedge_indices(hedgeable, len(premia))
    if not len(selected):
        return 0.0, baseline
    hedge = factor[np.ix_(selected, selected)]
    cross = beta @ factor[:, selected]
    adjusted_mean = mean - cross @ np.linalg.solve(hedge, premia[selected])
    adjusted_covar = covariance - cross @ np.linalg.solve(hedge, cross.T)
    capacity = float(premia[selected] @ np.linalg.solve(hedge, premia[selected])
                     + adjusted_mean @ np.linalg.solve(adjusted_covar, adjusted_mean))
    return capacity - baseline, capacity


class CompletionStatus(str, Enum):
    """Whether fixed-book completion has attainable finite weights or only a bound."""

    FINITE = "finite"
    SUPREMUM = "supremum"
    FIXED_BOOK = "fixed_book"
    NO_POSITIVE_MEAN = "no_positive_mean"
    ZERO_VARIANCE = "zero_variance"


@dataclass(frozen=True)
class CompletionResult:
    """Signed-Sharpe bound, attainment status and optional optimal exposure/overlay.

    ``sharpe2`` is a positive-Sharpe squared bound, not the square of a negative
    book Sharpe; it is None when no positive mean is attainable. ``signed_sharpe``
    may be an unattained supremum. ``hedge_only_sharpe`` refers to cancelling the
    selected factor exposures without premium-seeking leverage. Zero/zero is None.
    """

    signed_sharpe: float | None
    sharpe2: float | None
    status: CompletionStatus
    exposures: np.ndarray | None
    overlays: np.ndarray | None
    hedge_only_sharpe: float | None


def _signed(mean, variance):
    """Return signed Sharpe, infinity for arbitrage and None for zero/zero."""
    if variance == 0:
        return None if mean == 0 else float(np.copysign(np.inf, mean))
    return float(mean / np.sqrt(variance))


def completion_sharpe2(w, m, lam, Sig_F, B, Omega, hedgeable=None):
    """Maximize positive signed Sharpe with the asset book fixed and selected overlays free.

    With positive residual mean and variance the optimum is finite. Nonpositive
    conditional mean and nonzero hedge premia yield an unattained factor-direction
    supremum. Inputs are SPD; a zero book is handled as a separate degenerate case.
    No leverage, funding, margin or trading-cost constraint is inferred.
    """
    premia, factor, beta, residual, _, _ = _model(lam, Sig_F, B, Omega)
    weights, mean = _vector(w, B, "w"), _vector(m, B, "m")
    exposure = beta.T @ weights
    c = float(weights @ (mean - beta @ premia))
    v = float(weights @ residual @ weights)
    selected = _hedge_indices(hedgeable, len(premia))
    hedged = exposure.copy()
    hedged[selected] = 0
    hedge_sharpe = _signed(c + float(hedged @ premia), v + float(hedged @ factor @ hedged))
    if not len(selected):
        signed = _signed(float(weights @ mean), v + float(exposure @ factor @ exposure))
        positive = signed**2 if signed is not None and signed > 0 else None
        return CompletionResult(signed, positive, CompletionStatus.FIXED_BOOK,
                                exposure, np.zeros_like(exposure), hedge_sharpe)
    other = np.setdiff1d(np.arange(len(premia)), selected)
    hedge = factor[np.ix_(selected, selected)]
    cross = factor[np.ix_(selected, other)]
    shift = np.linalg.solve(hedge, cross @ exposure[other])
    conditional_mean = c + float(exposure[other] @ (
        premia[other] - cross.T @ np.linalg.solve(hedge, premia[selected])))
    conditional_var = v + float(exposure[other] @ (
        factor[np.ix_(other, other)] - cross.T @ np.linalg.solve(hedge, cross)) @ exposure[other])
    if conditional_var < 0:
        raise ValueError("negative conditional variance; covariance is numerically unstable")
    direction = np.linalg.solve(hedge, premia[selected])
    ceiling = float(premia[selected] @ direction)
    if conditional_var == 0:
        if conditional_mean != 0:
            raise ValueError("zero conditional variance with nonzero mean is unsupported")
        if ceiling == 0:
            return CompletionResult(None, None, CompletionStatus.ZERO_VARIANCE,
                                    None, None, hedge_sharpe)
        total = exposure.copy()
        total[selected] = direction - shift
        return CompletionResult(np.sqrt(ceiling), ceiling, CompletionStatus.FINITE,
                                total, total - exposure, hedge_sharpe)
    if conditional_mean <= 0:
        if ceiling == 0:
            return CompletionResult(0.0, None, CompletionStatus.NO_POSITIVE_MEAN,
                                    None, None, hedge_sharpe)
        return CompletionResult(np.sqrt(ceiling), ceiling, CompletionStatus.SUPREMUM,
                                None, None, hedge_sharpe)
    optimum = ceiling + conditional_mean**2 / conditional_var
    total = exposure.copy()
    total[selected] = (conditional_var / conditional_mean) * direction - shift
    return CompletionResult(np.sqrt(optimum), optimum, CompletionStatus.FINITE,
                            total, total - exposure, hedge_sharpe)


def ir_bound(m, Sigma):
    """Return squared excess-return capacity of zero-net-investment active holdings."""
    mean, covariance = _factors(m, Sigma)
    precision_mean = np.linalg.solve(covariance, mean)
    precision_one = np.linalg.solve(covariance, np.ones(len(mean)))
    return float(mean @ precision_mean - precision_mean.sum()**2 / precision_one.sum())
