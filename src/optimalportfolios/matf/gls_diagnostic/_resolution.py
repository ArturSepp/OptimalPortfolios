"""GLS geometry and precision diagnostics without altering conditional inference."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from optimalportfolios.matf.gls_diagnostic._inference import (
    _array, _axis, _scalar, compute_gls_geometry,
)


@dataclass(frozen=True)
class GlsResolutionResult:
    """Labelled residual visibility and factor resolution with explicit coordinates."""

    assets: pd.DataFrame
    factors: pd.DataFrame
    summary: dict


def _labels(B):
    """Return the validated loading matrix's asset and factor labels."""
    n, k = np.shape(B)
    return ((B.index.copy(), B.columns.copy()) if isinstance(B, pd.DataFrame) else
            (pd.RangeIndex(n, name="asset"), pd.RangeIndex(k, name="factor")))


def _factor_vector(value, labels, name):
    """Validate factor coordinates without silently relabelling their order."""
    _axis(value, labels)
    result = _array(value, name, 1)
    if len(result) != len(labels):
        raise ValueError(f"{name} factor dimension does not match B")
    return result


def compute_gls_resolution(B, Omega, *, factor_volatility=None, conditional_result=None,
                           annual_precision_target=.01, normalized_precision_target=.25,
                           rcond=None) -> GlsResolutionResult:
    """Expose design resolution and the current interval's precision, without refitting.

    ``sqrt(diag((B' Omega^-1 B)^-1)) / factor_volatility`` is normalized
    resolution per unit forecast-error scale. It is invariant to consistent factor
    rescaling. A value below a threshold is a diagnostic setting, not a rank test.
    Both covariance and premia must use annual decimal return units; no scaling
    or covariance repair is performed. ``conditional_result`` must have been fitted
    with the same B, Omega and SVD rank policy.

    Diagonal Omega identifies whitened coordinates with assets. Dense Omega uses
    ordered lower-Cholesky coordinates: their visibility must not be interpreted
    as the contribution of a single original asset. The original-coordinate fitted
    diagonal is supplied separately and need not lie between zero and one.
    """
    annual = _scalar(annual_precision_target, "annual_precision_target")
    normalized = _scalar(normalized_precision_target, "normalized_precision_target")
    if annual <= 0 or normalized <= 0:
        raise ValueError("precision targets must be positive")
    geometry = compute_gls_geometry(B, Omega, rcond=rcond)
    assets, factors = _labels(B)
    n, k = geometry.beta.shape
    unique = geometry.rank == k
    mapping = np.linalg.solve(geometry.chol.T, geometry.white_pinv.T).T
    white_fit = geometry.white @ geometry.white_pinv
    leverage = np.clip(np.diag(white_fit), 0., 1.)
    diagonal = np.array_equal(geometry.residual, np.diag(np.diag(geometry.residual)))
    asset_table = pd.DataFrame({
        "fitted_leverage": leverage,
        "residual_visibility": 1. - leverage,
        "original_coordinate_fitted_leverage": np.diag(geometry.beta @ mapping),
    }, index=assets)
    variance = (np.sum(geometry.white_pinv ** 2, axis=1) if unique else
                np.full(k, np.nan))
    vol = np.full(k, np.nan)
    if factor_volatility is not None:
        vol = _factor_vector(factor_volatility, factors, "factor_volatility")
        if np.any(vol <= 0):
            raise ValueError("factor_volatility must be positive")
    half_width = np.full(k, np.nan)
    if conditional_result is not None:
        if not all(hasattr(conditional_result, name) for name in
                   ("factors", "covariance", "summary")):
            raise ValueError("conditional_result needs factors, covariance and summary")
        _axis(conditional_result.factors, factors)
        _axis(conditional_result.covariance, factors)
        _axis(conditional_result.covariance, factors, axis=1)
        if (conditional_result.summary["rank"] != geometry.rank or
                conditional_result.summary["relative_cutoff"] != geometry.relative_cutoff):
            raise ValueError("conditional_result geometry does not match B and Omega")
        scale = conditional_result.summary["scale_squared"]
        if scale is not None and not np.allclose(conditional_result.covariance,
                scale * geometry.white_pinv @ geometry.white_pinv.T, rtol=1e-10, atol=1e-14):
            raise ValueError("conditional_result covariance does not match B and Omega")
        half_width = (conditional_result.factors.upper.to_numpy() -
                      conditional_result.factors.lower.to_numpy()) / 2.
    factor_table = pd.DataFrame({
        "identified": unique,
        "factor_volatility": vol,
        "residual_mimicking_variance": variance,
        "normalized_resolution_per_unit_scale": np.sqrt(variance) / vol,
        "gross_mimicking_weight": np.abs(mapping).sum(axis=1) if unique else np.nan,
        "ci_half_width": half_width,
        "normalized_ci_half_width": half_width / vol,
        "annual_precision_met": pd.array(
            [x <= annual if np.isfinite(x) else pd.NA for x in half_width], dtype="boolean"),
        "normalized_precision_met": pd.array(
            [x <= normalized if np.isfinite(x) else pd.NA for x in half_width / vol],
            dtype="boolean"),
    }, index=factors)
    return GlsResolutionResult(asset_table, factor_table, dict(
        method="gls_resolution", n_assets=n, n_factors=k, rank=geometry.rank,
        residual_df=n - geometry.rank, status="identified" if unique else "rank_deficient",
        visibility_basis=("asset_variance_whitened" if diagonal else
                          "ordered_cholesky_coordinates"),
        asset_visibility_interpretation=("Whitened coordinates correspond to individual assets."
            if diagonal else "Dense covariance: each ordered Cholesky coordinate mixes assets; "
            "visibility is not an individual original asset quantity."),
        annual_precision_target=annual, normalized_precision_target=normalized,
        relative_cutoff=geometry.relative_cutoff, units="annual_decimal",
        thresholds="Diagnostic settings, not production constraints or statistical rank tests",
    ))
