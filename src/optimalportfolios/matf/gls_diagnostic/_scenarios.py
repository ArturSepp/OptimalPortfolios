"""Explicit GLS uncertainty scenarios and covariance-aware provider comparisons.

These results supplement the default residual-scale conditional Student-t model.
No uncertainty model is selected from the provider's observed residual score.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm, t

from optimalportfolios.matf.gls_diagnostic._inference import (
    _array, _axis, _scalar, _vector, compute_gls_geometry,
)
from optimalportfolios.matf.gls_diagnostic._resolution import _factor_vector, _labels


@dataclass(frozen=True)
class GlsUncertaintyScenario:
    """Separate factor covariance, interval or scenario bands, and assumptions.

    ``factors`` contains estimate, standard_error, ci_lower and ci_upper. Endpoints
    are absent unless a distribution is explicitly assumed. Adviser disagreement
    endpoints describe a Gaussian dispersion scenario and are not confidence
    intervals for the consensus mean. ``summary`` records this distinction.
    """

    factors: pd.DataFrame
    covariance: pd.DataFrame
    summary: dict


def _text(value, name):
    """Require a nonempty explanatory label or provenance string."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty string")
    return value


def _psd(value, labels, name):
    """Reject invalid covariance; tolerate only floating-point PSD roundoff."""
    _axis(value, labels)
    if isinstance(value, pd.DataFrame):
        _axis(value, labels, axis=1)
    result = _array(value, name, 2)
    if result.shape != (len(labels), len(labels)):
        raise ValueError(f"{name} dimensions do not match labels")
    scale = max(float(np.max(np.abs(result))), np.finfo(float).tiny)
    tolerance = scale * max(1, len(labels)) * np.finfo(float).eps * 100
    if not np.allclose(result, result.T, rtol=0, atol=tolerance):
        raise ValueError(f"{name} must be symmetric")
    if np.linalg.eigvalsh(result).min() < -tolerance:
        raise ValueError(f"{name} must be positive semidefinite")
    return result


def _scenario(estimate, covariance, labels, *, method, distribution, confidence_level,
              reference_degrees_of_freedom, reference_provenance, independent_reference,
              target, units, assumptions, rank, n_factors, extra=None):
    """Construct endpoints only under an explicitly named probability model."""
    _text(method, "method")
    _text(target, "target")
    if units != "annual_decimal":
        raise ValueError("units must be annual_decimal; caller must convert inputs explicitly")
    confidence = _scalar(confidence_level, "confidence_level")
    if not 0 < confidence < 1:
        raise ValueError("confidence_level must be strictly between zero and one")
    if distribution not in ("none", "normal", "reference_student_t"):
        raise ValueError("distribution must be none, normal or reference_student_t")
    if not isinstance(independent_reference, (bool, np.bool_)):
        raise ValueError("independent_reference must be boolean")
    if not isinstance(assumptions, (tuple, list)) or not all(
            isinstance(x, str) for x in assumptions):
        raise ValueError("assumptions must be a sequence of strings")
    df = None
    critical = None
    if reference_provenance is not None:
        _text(reference_provenance, "reference_provenance")
    if distribution == "reference_student_t":
        df = _scalar(reference_degrees_of_freedom, "reference_degrees_of_freedom")
        if df <= 0 or not independent_reference or reference_provenance is None:
            raise ValueError("reference Student-t needs positive df, provenance and independence")
        critical = float(t.ppf((1 + confidence) / 2, df))
    elif reference_degrees_of_freedom is not None or independent_reference:
        raise ValueError("reference degrees of freedom and independence require "
                         "reference_student_t")
    elif distribution == "normal":
        critical = float(norm.ppf((1 + confidence) / 2))
    identified = rank == n_factors
    se = np.sqrt(np.maximum(np.diag(covariance), 0)) if identified else np.full(n_factors, np.nan)
    lower, upper = (np.full(n_factors, np.nan) for _ in range(2))
    status = "unavailable_rank_deficient" if not identified else "covariance_only"
    if identified and critical is not None:
        lower, upper = estimate - critical * se, estimate + critical * se
        status = "explicit_distribution_scenario"
    if not identified:
        estimate = np.full(n_factors, np.nan)
        covariance = np.full((n_factors, n_factors), np.nan)
    elif not all(np.isfinite(x).all() for x in (estimate, covariance, se)) or (
            critical is not None and not (np.isfinite(lower).all() and np.isfinite(upper).all())):
        raise ValueError("GLS scenario overflow; check covariance units and factor identification")
    mandatory = ("B, residual risk, cash, admitted alpha and factor definitions are fixed.",
                 "The target is the GLS projection of the provider's expected forecast, "
                 "which may differ from an unshifted structural premium.")
    if distribution == "normal":
        mandatory += ("The supplied scenario covariance is treated as fixed and "
                      "centered deviations are multivariate Gaussian.",)
    elif distribution == "reference_student_t":
        mandatory += ("A centered Gaussian projection is independent of a reference scale "
                      "estimate with the asserted scaled chi-square law and reference df.",)
    summary = dict(method=method, distribution=distribution, confidence_level=confidence,
                   target=target, units=units, interval_status=status,
                   calibration_status="assumed_scenario_not_empirically_calibrated",
                   reference_provenance=reference_provenance,
                   reference_degrees_of_freedom=df,
                   independent_reference=bool(independent_reference),
                   assumptions=mandatory + tuple(assumptions), critical_value=critical,
                   rank=rank, n_factors=n_factors, simultaneous_coverage=False)
    if extra:
        summary.update(extra)
    return GlsUncertaintyScenario(pd.DataFrame(dict(
        estimate=estimate, standard_error=se, ci_lower=lower, ci_upper=upper), index=labels),
        pd.DataFrame(covariance, index=labels, columns=labels), summary)


def compute_gls_uncertainty(B, Omega, premia, forecast_covariance, *,
                            method="supplied_covariance", distribution="none",
                            confidence_level=.95, reference_degrees_of_freedom=None,
                            reference_provenance=None, independent_reference=False,
                            target="projected_provider_mean", units="annual_decimal",
                            assumptions=(), rcond=None) -> GlsUncertaintyScenario:
    """Propagate an explicit forecast-error covariance as A C A'.

    Inputs use annual decimal returns and their squared units. No horizon or
    adviser-count divisor is applied. A covariance alone provides standard errors;
    normal intervals require ``distribution='normal'``. Estimated reference-scale
    t intervals additionally require its df, provenance and explicit independence.
    A general covariance implies a weighted chi-square residual score; its expected
    value and weights are returned, without inventing a scaled chi-square test.
    """
    geometry = compute_gls_geometry(B, Omega, rcond=rcond)
    assets, factors = _labels(B)
    estimate = _factor_vector(premia, factors, "premia")
    covariance = _psd(forecast_covariance, assets, "forecast_covariance")
    mapping = np.linalg.solve(geometry.chol.T, geometry.white_pinv.T).T
    propagated = mapping @ covariance @ mapping.T
    left_white = np.linalg.solve(geometry.chol, covariance)
    white_covariance = np.linalg.solve(geometry.chol, left_white.T).T
    residual = np.eye(len(assets)) - geometry.white @ geometry.white_pinv
    residual_covariance = residual @ white_covariance @ residual.T
    weights = np.maximum(np.linalg.eigvalsh(residual_covariance), 0.)
    return _scenario(estimate, propagated, factors, method=method, distribution=distribution,
        confidence_level=confidence_level,
        reference_degrees_of_freedom=reference_degrees_of_freedom,
        reference_provenance=reference_provenance, independent_reference=independent_reference,
        target=target, units=units, assumptions=assumptions, rank=geometry.rank,
        n_factors=len(factors), extra=dict(
            n_assets=len(assets), expected_centered_residual_score=float(weights.sum()),
            centered_residual_chi_square_weights=weights.tolist(),
            residual_score_assumption="Weights apply only to centered Gaussian forecast errors; "
            "a nonzero residual mean adds a noncentral component."))


def compute_gls_reference_uncertainty(B, Omega, premia, *, scale_squared,
                                      reference_provenance, reference_degrees_of_freedom=None,
                                      independent_reference=False, confidence_level=.95,
                                      rcond=None) -> GlsUncertaintyScenario:
    """Propagate a declared external proportional scale, never choose it from a test.

    With no df, the scale is asserted known and normal critical values apply. With
    df, it is an independent scaled-chi-square reference estimate and t critical
    values apply. A provider average, median or selected residual scale does not
    automatically have this sampling law.
    """
    scale = _scalar(scale_squared, "scale_squared")
    if scale < 0:
        raise ValueError("scale_squared must be nonnegative")
    _text(reference_provenance, "reference_provenance")
    geometry = compute_gls_geometry(B, Omega, rcond=rcond)
    distribution = "normal" if reference_degrees_of_freedom is None else "reference_student_t"
    result = compute_gls_uncertainty(B, Omega, premia, scale * geometry.residual,
        method="external_reference_scale", distribution=distribution,
        confidence_level=confidence_level,
        reference_degrees_of_freedom=reference_degrees_of_freedom,
        reference_provenance=reference_provenance, independent_reference=independent_reference,
        assumptions=("Forecast-error covariance is scale_squared times Omega; "
                     "the reference must match currency, horizon and target conventions.",),
        rcond=rcond)
    return GlsUncertaintyScenario(result.factors, result.covariance,
                                  dict(result.summary, scale_squared=scale))


def compute_gls_adviser_disagreement(B, Omega, premia, lower_quartile, upper_quartile, *,
                                    correlation=None, source_provenance,
                                    confidence_level=.95, rcond=None) -> GlsUncertaintyScenario:
    """Propagate IQR-derived adviser dispersion under a labelled Gaussian scenario.

    Quartiles must already use the target's annual arithmetic-return convention.
    Fixed common cash/alpha shifts leave the IQR unchanged. No adviser-count or
    forecast-horizon divisor is valid without another, explicit averaging model.
    Scalar correlation gives an equicorrelation scenario; a matrix must be PSD.
    Neither quartiles nor this approximation identify cross-asset dependence.
    """
    _text(source_provenance, "source_provenance")
    lo, hi = (_vector(x, B, name) for x, name in (
        (lower_quartile, "lower_quartile"), (upper_quartile, "upper_quartile")))
    if np.any(lo > hi):
        raise ValueError("lower_quartile must not exceed upper_quartile")
    assets, _ = _labels(B)
    n = len(assets)
    if correlation is None:
        corr, description = np.eye(n), "diagonal cross-asset disagreement scenario"
    elif np.isscalar(correlation):
        rho = _scalar(correlation, "correlation")
        if not -1 <= rho <= 1:
            raise ValueError("correlation must be in [-1,1] and positive semidefinite")
        corr = np.full((n, n), rho)
        np.fill_diagonal(corr, 1.)
        description = f"equicorrelation disagreement scenario: rho={rho:g}"
    else:
        corr, description = correlation, "supplied cross-asset disagreement correlation"
    corr = _psd(corr, assets, "correlation")
    if not np.allclose(np.diag(corr), 1., rtol=0., atol=1e-12):
        raise ValueError("correlation must have unit diagonal")
    conversion = float(norm.ppf(.75) - norm.ppf(.25))
    dispersion = (hi - lo) / conversion
    covariance = corr * np.outer(dispersion, dispersion)
    result = compute_gls_uncertainty(B, Omega, premia, covariance,
        method="adviser_disagreement", distribution="normal", confidence_level=confidence_level,
        reference_provenance=source_provenance, target="projected_adviser_forecast_dispersion",
        assumptions=("Marginal IQR divided by the normal IQR approximates adviser dispersion.",
                     description, "This is not consensus-mean estimation error, provider "
                     "forecast accuracy, or a calibrated confidence interval."), rcond=rcond)
    return GlsUncertaintyScenario(result.factors, result.covariance, dict(result.summary,
        calibration_status="adviser_disagreement_scenario_not_confidence_interval",
        interval_status=("gaussian_disagreement_scenario" if result.summary["rank"] ==
                         result.summary["n_factors"] else "unavailable_rank_deficient"),
        normal_iqr_conversion=conversion, adviser_count_divisor=None,
        correlation_scenario=description))


def compute_gls_provider_contrast(B_left, Omega_left, premia_left,
                                  B_right, Omega_right, premia_right, *,
                                  covariance_left, covariance_right, cross_covariance,
                                  left_provider, right_provider, distribution="none",
                                  confidence_level=.95, assumptions=(), rcond=None
                                  ) -> GlsUncertaintyScenario:
    """Propagate left-minus-right premia using the complete paired covariance.

    Designs and asset universes may differ; factor coordinates and units must be
    identical. Cross covariance rows belong to the left universe and columns to
    the right. Supply zeros explicitly to assert independence. The whole joint
    forecast covariance must be positive semidefinite. Unknown dependence should
    be represented by several declared feasible scenarios rather than an implicit
    significance claim. Only covariance-only or known-Gaussian intervals are allowed.
    """
    _text(left_provider, "left_provider")
    _text(right_provider, "right_provider")
    left = compute_gls_geometry(B_left, Omega_left, rcond=rcond)
    right = compute_gls_geometry(B_right, Omega_right, rcond=rcond)
    assets_left, factors = _labels(B_left)
    assets_right, right_factors = _labels(B_right)
    if not factors.equals(right_factors):
        raise ValueError("provider factor labels/order must match")
    first = _factor_vector(premia_left, factors, "premia_left")
    second = _factor_vector(premia_right, factors, "premia_right")
    c_left = _psd(covariance_left, assets_left, "covariance_left")
    c_right = _psd(covariance_right, assets_right, "covariance_right")
    _axis(cross_covariance, assets_left)
    if isinstance(cross_covariance, pd.DataFrame):
        _axis(cross_covariance, assets_right, axis=1)
    cross = _array(cross_covariance, "cross_covariance", 2)
    if cross.shape != (len(assets_left), len(assets_right)):
        raise ValueError("cross_covariance dimensions must match left and right universes")
    joint = np.block([[c_left, cross], [cross.T, c_right]])
    _psd(joint, pd.RangeIndex(len(joint)), "joint forecast covariance")
    a_left = np.linalg.solve(left.chol.T, left.white_pinv.T).T
    a_right = np.linalg.solve(right.chol.T, right.white_pinv.T).T
    operator = np.concatenate([a_left, -a_right], axis=1)
    covariance = operator @ joint @ operator.T
    return _scenario(first - second, covariance, factors, method="paired_provider_contrast",
        distribution=distribution, confidence_level=confidence_level,
        reference_degrees_of_freedom=None, reference_provenance=None, independent_reference=False,
        target="difference_of_projected_provider_means", units="annual_decimal",
        assumptions=assumptions, rank=min(left.rank, right.rank), n_factors=len(factors),
        extra=dict(
            comparison_label=f"{left_provider} minus {right_provider}",
            left_provider=left_provider, right_provider=right_provider,
            cross_covariance_supplied=True, left_rank=left.rank, right_rank=right.rank))
