"""Cross-sectional GLS inference of factor premia from asset forecasts.

This is a second-stage projection conditional on estimated loadings, not a
time-series loading estimator. Whitening and a single SVD cutoff determine rank,
the projector, mimicking portfolios and conditional Student-t intervals. No
intercept, covariance repair, annualisation or missing-data substitution is added.

Risk-premia inference belongs to OptimalPortfolios. Historical loading and alpha
estimation remain separate in FactorLasso and its consuming applications.

References
----------
.. [1] Aitken, A. C. (1935). On Least Squares and Linear Combination of
   Observations. Proceedings of the Royal Society of Edinburgh, 55, 42-48.
   doi:10.1017/S0370164600014346.

The finite-sample intervals additionally assume Gaussian forecast errors with
covariance proportional to the supplied residual-return covariance. Estimated
loadings and alpha are held fixed; their estimation uncertainty is not covered.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import t


def _array(value, name, ndim):
    """Require a nonempty finite real array without changing its order."""
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    result = np.asarray(value, dtype=float)
    if result.ndim != ndim or not all(result.shape) or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a nonempty finite {ndim}-dimensional array")
    return result


def _axis(value, labels=None, axis=0):
    """Reject ambiguous pandas labels; never silently reindex inputs."""
    if isinstance(value, (pd.Series, pd.DataFrame)):
        actual = value.columns if axis else value.index
        if actual.has_duplicates or actual.hasnans or (
                labels is not None and not actual.equals(labels)):
            raise ValueError("label/order mismatch or duplicate/missing labels")


def _vector(value, loadings, name):
    """Validate an asset vector against the loading rows."""
    _axis(value, loadings.index if isinstance(loadings, pd.DataFrame) else None)
    result = _array(value, name, 1)
    if len(result) != len(loadings):
        raise ValueError(f"{name} asset dimension does not match B")
    return result


def _scalar(value, name):
    """Require a finite real scalar excluding booleans."""
    if (not isinstance(value, (int, float, np.integer, np.floating))
            or isinstance(value, (bool, np.bool_)) or not np.isfinite(value)):
        raise ValueError(f"{name} must be a finite real scalar")
    return float(value)


@dataclass(frozen=True)
class GlsGeometry:
    """Validated GLS geometry with one numerical rank convention.

    Attributes:
        beta, residual, chol, white (ndarray):
            Loadings, residual covariance, its lower Cholesky factor and whitened loadings.
        singular_values (ndarray):
            Singular values of the whitened loadings (not of their Gram matrix).
        rank (int):
            Number of singular values strictly above ``cutoff``.
        relative_cutoff, cutoff (float):
            Relative and absolute singular-value cutoffs.
        white_pinv, residual_projector (ndarray):
            Whitened pseudoinverse and asset-space residual operator, respectively.
    """

    beta: np.ndarray
    residual: np.ndarray
    chol: np.ndarray
    white: np.ndarray
    singular_values: np.ndarray
    rank: int
    relative_cutoff: float
    cutoff: float
    white_pinv: np.ndarray
    residual_projector: np.ndarray


def compute_gls_geometry(B, Omega, *, rcond=None):
    """Build whitened GLS geometry without squaring the condition number.

    Args:
        B (array-like of shape (n_assets, n_factors)):
            Factor loadings. Pandas axes must be unique, nonmissing and exactly aligned.
        Omega (array-like of shape (n_assets,) or (n_assets, n_assets)):
            Positive residual variances or a positive-definite residual covariance.
        rcond (float or None, default None):
            Relative SVD cutoff in [0, 1). None uses eps * max(n_assets, n_factors).

    Returns:
        GlsGeometry:
            A rank-revealing geometry. Truncating a nonzero direction changes the
            effective model span and can invalidate identities under the original model.
    """
    _axis(B)
    if isinstance(B, pd.DataFrame):
        _axis(B, axis=1)
    assets = B.index if isinstance(B, pd.DataFrame) else None
    _axis(Omega, assets)
    if isinstance(Omega, pd.DataFrame):
        _axis(Omega, Omega.index if assets is None else assets, axis=1)
    beta = _array(B, "B", 2)
    if np.iscomplexobj(Omega):
        raise ValueError("Omega must be real")
    residual = np.asarray(Omega, dtype=float)
    if residual.ndim == 1:
        residual = np.diag(_array(Omega, "Omega", 1))
    residual = _array(residual, "Omega", 2)
    n = len(beta)
    if residual.shape != (n, n) or not np.allclose(residual, residual.T, rtol=0, atol=1e-12):
        raise ValueError(f"Omega must be a symmetric {n} by {n} matrix")
    try:
        chol = np.linalg.cholesky(residual)
    except np.linalg.LinAlgError as error:
        raise ValueError("Omega must be positive definite") from error
    white = np.linalg.solve(chol, beta)
    relative = np.finfo(float).eps * max(white.shape) if rcond is None else _scalar(
        rcond, "rcond")
    if not 0 <= relative < 1:
        raise ValueError("rcond must be a finite scalar in [0, 1)")
    u, singular, vt = np.linalg.svd(white, full_matrices=False)
    cutoff = float(relative * singular[0])
    keep = singular > cutoff
    basis = u[:, keep]
    inverse = (vt[keep].T / singular[keep]) @ basis.T
    projector = chol @ (np.eye(n) - basis @ basis.T) @ np.linalg.solve(chol, np.eye(n))
    return GlsGeometry(beta, residual, chol, white, singular, int(keep.sum()),
                       float(relative), cutoff, inverse, projector)


def gls_projector(B, Omega):
    """Return the asset-space residual operator using the default SVD rank policy.

    Args:
        B, Omega (array-like):
            Loadings and residual covariance as in :func:`compute_gls_geometry`.

    Returns:
        ndarray:
            Residual operator of shape (n_assets, n_assets).
    """
    return compute_gls_geometry(B, Omega).residual_projector


def gls_split(m, B, Omega):
    """Split excess forecasts into minimum-norm premia and orthogonal residuals.

    Args:
        m (array-like of shape (n_assets,)):
            Excess forecasts, with any admitted alpha already subtracted.
        B, Omega (array-like):
            Loadings and residual covariance as in :func:`compute_gls_geometry`.

    Returns:
        premia, residual : ndarray:
            Factor coordinates and asset residual. Coordinates are not uniquely
            identified at deficient rank; use :func:`infer_factor_premia` for flags.
    """
    geometry = compute_gls_geometry(B, Omega)
    mean = _vector(m, B, "m")
    premia = geometry.white_pinv @ np.linalg.solve(geometry.chol, mean)
    return premia, geometry.residual_projector @ mean


def factor_mimicking_portfolios(B, Omega):
    """Return minimum-residual-variance portfolios with unit factor exposures.

    Args:
        B, Omega (array-like):
            Loadings and residual covariance as in :func:`compute_gls_geometry`.

    Returns:
        ndarray:
            Asset-by-factor risky weights Q, with B.T @ Q = I. Weights may be short
            or leveraged and need not sum to one; cash completes their financing.

    Raises:
        ValueError:
            If the loading matrix lacks full column rank.
    """
    geometry = compute_gls_geometry(B, Omega)
    if geometry.rank != geometry.beta.shape[1]:
        raise ValueError("unit factor exposures are unattainable: B lacks full column rank")
    return np.linalg.solve(geometry.chol.T, geometry.white_pinv.T)


@dataclass(frozen=True)
class GlsPremiaResult:
    """Labelled inference, exact forecast reconciliation and explicit assumptions.

    Attributes:
        premia (Series):
            Minimum-norm factor coordinates; only uniquely identified at full rank.
        assets (DataFrame):
            Total forecast, cash, credited alpha, GLS target, fitted return and residual.
        factors (DataFrame):
            Identified estimates, pointwise conditional intervals, VIF and gross weights.
        covariance (DataFrame):
            Conditional premium covariance; NaN when intervals are unavailable.
        mimicking_weights (DataFrame):
            Asset-by-factor unit-exposure weights; NaN at deficient rank.
        sensitivity_bounds (DataFrame):
            Exact marginal forecast-box bounds at full rank, separate from confidence intervals.
        summary (dict):
            Rank, scale, degrees of freedom, interval status and working assumptions.
    """

    premia: pd.Series
    assets: pd.DataFrame
    factors: pd.DataFrame
    covariance: pd.DataFrame
    mimicking_weights: pd.DataFrame
    sensitivity_bounds: pd.DataFrame
    summary: dict


def infer_factor_premia(B, Omega, total_returns, *, cash_rate, alpha=None, alpha_weight=0.,
                       confidence_level=.95, rcond=None, lower_returns=None, upper_returns=None,
                       cash_bounds=None):
    """Infer factor premia from provider CMAs conditional on a fitted risk model.

    Args:
        B, Omega (array-like):
            Estimated loadings and residual-return covariance, in matching asset order.
        total_returns (array-like of shape (n_assets,)):
            Provider arithmetic total-return forecasts in decimal units and one currency.
        cash_rate (float):
            One common cash forecast, in the same return units and horizon convention.
        alpha (array-like of shape (n_assets,) or None, default None):
            Declared alpha vector (including any asset-specific admission weights).
            Supply historical alpha directly only if full historical admission is intended.
        alpha_weight (float, default 0):
            Diagnostic multiplier in [0, 1]. Target = total_returns - cash_rate
            - alpha_weight * alpha. Nonzero weight requires an explicit alpha vector.
        confidence_level (float, default 0.95):
            Pointwise conditional Student-t coverage under the additional working model.
        rcond (float or None, default None):
            SVD rank cutoff passed to :func:`compute_gls_geometry`.
        lower_returns, upper_returns (array-like or None, default None):
            Optional paired total-return box endpoints in the same asset order and units.
        cash_bounds (pair of float or None, default None):
            One shared cash interval for the box. None keeps cash fixed. These sensitivity
            boxes are neither factor quantiles nor statistical confidence intervals.

    Returns:
        GlsPremiaResult:
            Annual outputs when inputs are annual. No annualisation or extra sample-size
            divisor is applied. Coordinate CIs and mimicking portfolios are unavailable
            at deficient rank; zero residual scale and N <= K also suppress CIs.

    Notes:
        Forecast errors are Gaussian with covariance tau squared times Omega. B, Omega,
        cash, alpha and factor definitions are fixed. This assumption is not implied by
        a return risk model and excludes first-stage estimation uncertainty. Data fetching,
        price conversion, loading estimation and report rendering belong to the caller.
    """
    cash = _scalar(cash_rate, "cash_rate")
    weight = _scalar(alpha_weight, "alpha_weight")
    confidence = _scalar(confidence_level, "confidence_level")
    if not 0 <= weight <= 1:
        raise ValueError("alpha_weight must be in [0, 1]")
    if not 0 < confidence < 1:
        raise ValueError("confidence_level must be strictly between zero and one")
    if alpha is None and weight != 0:
        raise ValueError("nonzero alpha_weight requires an explicit alpha vector")
    geometry = compute_gls_geometry(B, Omega, rcond=rcond)
    n, k = geometry.beta.shape
    ids = B.index.copy() if isinstance(B, pd.DataFrame) else pd.RangeIndex(n, name="asset")
    names = B.columns.copy() if isinstance(B, pd.DataFrame) else pd.RangeIndex(k, name="factor")
    total = _vector(total_returns, B, "total_returns")
    admission = np.zeros(n) if alpha is None else _vector(alpha, B, "alpha")
    credited = weight * admission
    target = total - cash - credited
    white_target = np.linalg.solve(geometry.chol, target)
    estimate = geometry.white_pinv @ white_target
    residual = geometry.residual_projector @ target
    white_residual = np.linalg.solve(geometry.chol, residual)
    sse = float(white_residual @ white_residual)
    rank, df = geometry.rank, n - geometry.rank
    unique = rank == k
    status = ("unavailable_rank_deficient" if not unique else
              "unavailable_residual_df" if df <= 0 else
              "unavailable_zero_residual_scale"
              if sse <= max(float(white_target @ white_target), 1e-300) * 1e-24 else
              "conditional_working_model")
    covariance = np.full((k, k), np.nan)
    se, lower, upper = (np.full(k, np.nan) for _ in range(3))
    scale = critical = None
    inverse_gram = geometry.white_pinv @ geometry.white_pinv.T
    if status == "conditional_working_model":
        scale = sse / df
        critical = float(t.isf((1. - confidence) / 2., df))
        covariance = scale * inverse_gram
        se = np.sqrt(np.diag(covariance))
        lower, upper = estimate - critical * se, estimate + critical * se
        if not all(np.isfinite(x).all() for x in (covariance, se, lower, upper)):
            raise ValueError("Conditional GLS intervals overflow; check units and identification")
    q = np.linalg.solve(geometry.chol.T, geometry.white_pinv.T)
    vif = np.sum(geometry.white**2, axis=0) * np.diag(inverse_gram)
    table = pd.DataFrame(dict(estimate=estimate if unique else np.full(k, np.nan),
                             standard_error=se, lower=lower, upper=upper, status=status,
                             variance_inflation=vif if unique else np.full(k, np.nan),
                             gross_weight=np.abs(q).sum(axis=0) if unique else np.full(k, np.nan)),
                         index=names)
    bounds = pd.DataFrame(columns=["lower", "upper", "status"], index=names)
    if (lower_returns is None) != (upper_returns is None):
        raise ValueError("lower_returns and upper_returns must be supplied together")
    if cash_bounds is not None and lower_returns is None:
        raise ValueError("cash_bounds requires return box endpoints")
    if lower_returns is not None:
        lo, hi = (_vector(v, B, name) for v, name in (
            (lower_returns, "lower_returns"), (upper_returns, "upper_returns")))
        if np.any(lo > hi):
            raise ValueError("lower_returns must not exceed upper_returns")
        cash_lo = cash_hi = cash
        if cash_bounds is not None:
            if len(cash_bounds) != 2:
                raise ValueError("cash_bounds must contain two ordered finite endpoints")
            cash_lo, cash_hi = (_scalar(x, "cash_bounds") for x in cash_bounds)
            if cash_lo > cash_hi:
                raise ValueError("cash_bounds must contain two ordered finite endpoints")
        positive, negative, funding = np.maximum(q, 0), np.minimum(q, 0), -q.sum(axis=0)
        low = positive.T @ lo + negative.T @ hi - q.T @ credited
        high = positive.T @ hi + negative.T @ lo - q.T @ credited
        low += np.minimum(funding * cash_lo, funding * cash_hi)
        high += np.maximum(funding * cash_lo, funding * cash_hi)
        bounds = pd.DataFrame(dict(lower=low if unique else np.full(k, np.nan),
                                   upper=high if unique else np.full(k, np.nan),
                                   status="sensitivity_box" if unique else status), index=names)
    assets = pd.DataFrame(dict(total_return=total, cash=cash, alpha=admission,
                               credited_alpha=credited, gls_target=target,
                               fitted=geometry.beta @ estimate, residual=residual), index=ids)
    summary = dict(method="conditional_gls_student_t", confidence_level=confidence,
                   scope="pointwise", status=status, n_assets=n, n_factors=k, rank=rank,
                   residual_df=df, weighted_residual_sse=sse, scale_squared=scale,
                   critical_value=critical, relative_cutoff=geometry.relative_cutoff,
                   alpha_weight=weight, cash_rate=cash,
                   coordinate_convention="unique" if unique else "minimum_norm_nonidentified",
                   assumptions="y = B lambda + e; e ~ N(0, tau^2 Omega). B, Omega, cash, "
                   "alpha and factor definitions are fixed. Forecast-error covariance is "
                   "assumed proportional to residual return covariance. No intercept.",
                   excluded_uncertainty="First-stage loadings, residual-risk and alpha estimation; "
                   "cash and factor definitions; forecast-error covariance misspecification.",
                   interpretation="Conditional pointwise intervals, not simultaneous coverage, "
                   "provider accuracy, adviser dispersion or realized-return prediction.")
    return GlsPremiaResult(pd.Series(estimate, index=names, name="premium"), assets, table,
                           pd.DataFrame(covariance, index=names, columns=names),
                           pd.DataFrame(q if unique else np.full_like(q, np.nan),
                                        index=ids, columns=names), bounds, summary)
