"""Conditional factor precision and pair contrasts without pooling the risk model."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import f, t

from optimalportfolios.matf import sharpe_accounting as sa
from optimalportfolios.matf.cma_diagnostics import CmaDiagnosticInputs, CmaDiagnosticSpec, _labelled
from optimalportfolios.matf.cma_uncertainty import compute_conditional_gls_intervals


@dataclass(frozen=True)
class CmaIdentificationResult:
    """Individual precision, pair geometry, linear contrasts and declared diagnostic settings."""

    factors: pd.DataFrame
    contrasts: pd.DataFrame
    directions: pd.DataFrame
    pair: dict
    summary: dict


def _positive_setting(value, name):
    """Accept an optional positive finite scalar threshold, never a boolean or container."""
    if value is not None and (
        not np.isscalar(value)
        or isinstance(value, (str, bool))
        or not np.isfinite(value)
        or value <= 0
    ):
        raise ValueError(f"{name} must be a positive finite scalar or None")


def _contrast(weights, ci, units):
    """Propagate the full covariance, including cross terms, into a fixed linear contrast."""
    estimate = float(weights @ ci.factors.estimate.to_numpy())
    status = ci.summary["status"]
    se = low = high = pvalue = np.nan
    if status == "conditional_working_model":
        variance = float(weights @ ci.covariance.to_numpy() @ weights)
        if not np.isfinite(variance) or variance <= 0:
            raise ValueError("Contrast variance must be finite and positive")
        se = np.sqrt(variance)
        width = ci.summary["critical_value"] * se
        low, high = estimate - width, estimate + width
        pvalue = float(2 * t.sf(abs(estimate / se), ci.summary["residual_df"]))
    return dict(
        estimate=estimate,
        standard_error=se,
        lower=low,
        upper=high,
        p_value=pvalue,
        units=units,
        status=status,
    )


def _pair_geometry(geometry, positions, vol):
    """Residualize a pair against all other columns, then use SVD in fixed risk units."""
    other = [j for j in range(geometry.white.shape[1]) if j not in positions]
    residualized = geometry.white[:, positions].copy()
    if other:
        orthogonal, _ = np.linalg.qr(geometry.white[:, other], mode="reduced")
        residualized -= orthogonal @ (orthogonal.T @ residualized)
    lengths = np.linalg.norm(residualized, axis=0)
    rho = float(np.clip(residualized[:, 0] @ residualized[:, 1] / np.prod(lengths), -1.0, 1.0))
    _, singular, right = np.linalg.svd(residualized * vol, full_matrices=False)
    eigenvalues, directions = singular[::-1] ** 2, right[::-1].T
    for j in range(2):
        directions[:, j] *= 1 if directions[np.argmax(abs(directions[:, j])), j] >= 0 else -1
    mutual_vif = float(np.mean(lengths**2 * np.sum(geometry.white_pinv[positions] ** 2, axis=1)))
    return rho, eigenvalues, directions, mutual_vif


# gls_diagnostic supplies finite-df GLS covariance through the CMA adapter. This
# report layer adds the declared pair contrasts and economic precision settings.
def compute_cma_identification(
    inputs: CmaDiagnosticInputs,
    *,
    factor_pair=("Carry G10", "Carry EM"),
    confidence_level=0.95,
    precision_target=None,
    equivalence_margin=None,
    spec: CmaDiagnosticSpec | None = None,
) -> CmaIdentificationResult:
    """Diagnose fixed-model identification without changing exposures, forecasts or factor count.

    precision_target is a native annual decimal CI half-width, not a significance test.
    equivalence_margin bounds the difference of premia divided by fitted factor volatilities.
    Common is half the sum; spread is first minus second. Both are unrestricted contrasts.
    Optional thresholds default to no assessment. TOST uses level 1-confidence_level and
    a 2*confidence_level-1 interval. All statistical statements condition on B, D and vol.
    """
    _positive_setting(precision_target, "precision_target")
    _positive_setting(equivalence_margin, "equivalence_margin")
    if (
        not isinstance(factor_pair, (tuple, list))
        or len(factor_pair) != 2
        or any(not isinstance(x, str) or not x for x in factor_pair)
        or factor_pair[0] == factor_pair[1]
    ):
        raise ValueError("factor_pair must contain two distinct nonempty labels")
    ci = compute_conditional_gls_intervals(inputs, confidence_level=confidence_level, spec=spec)
    spec = CmaDiagnosticSpec() if spec is None else spec
    geometry = sa._gls_geometry(inputs.betas, inputs.residual_variances, rcond=spec.rcond)
    names = inputs.betas.columns
    k = len(names)
    full_rank = geometry.rank == k
    table = ci.factors.copy()
    raw = np.sum(geometry.white**2, axis=0)
    variance = np.sum(geometry.white_pinv**2, axis=1) if full_rank else np.full(k, np.nan)
    table["raw_loading_information"] = raw
    table["conditional_information"] = 1 / variance
    table["residual_mimicking_variance"] = variance
    table["variance_inflation"] = raw * variance
    table["distinct_loading_fraction"] = 1 / (raw * variance)
    table["ci_half_width"] = (table.upper - table.lower) / 2
    table["precision_target"] = np.nan if precision_target is None else precision_target
    table["precision_status"] = "not_assessed" if precision_target is None else "unavailable"
    if precision_target is not None:
        valid = table.ci_half_width.notna()
        table.loc[valid, "precision_status"] = np.where(
            table.loc[valid, "ci_half_width"] <= precision_target,
            "within_target",
            "weak_for_target",
        )
    vol = None
    if inputs.factor_covariance is not None:
        _labelled(inputs.factor_covariance, pd.DataFrame, names, "factor_covariance")
        sa._align(inputs.factor_covariance, names, axis=1)
        covariance = sa._covariance(inputs.factor_covariance, k, "factor_covariance")
        vol = np.sqrt(np.diag(covariance))
        table["factor_volatility"] = vol
        table["normalized_ci_half_width"] = table.ci_half_width / vol
    pair = dict(
        factors=list(factor_pair),
        status="unavailable_missing_factor",
        geometry_status="unavailable",
        equality_status="unavailable",
        equivalence_status="not_assessed" if equivalence_margin is None else "unavailable",
        equivalence_margin=equivalence_margin,
        interpretation="Unrestricted contrasts; two risk factors retained. "
        "Non-rejection of equality is not evidence of equivalence.",
    )
    summary = dict(
        ci.summary,
        method="conditional_gls_identification",
        precision_target=precision_target,
        equivalence_margin=equivalence_margin,
        factor_pair=list(factor_pair),
        factor_model_changed=False,
        normalization="theta_j = lambda_j / sqrt(factor_covariance[j,j]); vol fixed",
        common_definition="(first + second)/2",
        spread_definition="first - second",
        thresholds="Diagnostic settings, not production constraints or rank tests",
    )
    contrasts, directions = {}, {}
    if all(name in names for name in factor_pair):
        pair["status"] = ci.summary["status"]
        positions = [names.get_loc(name) for name in factor_pair]
        for label, coefficients in [("common", [0.5, 0.5]), ("spread", [1.0, -1.0])]:
            weights = np.zeros(k)
            weights[positions] = coefficients
            contrasts[label] = _contrast(weights, ci, "annual_excess_decimal")
            if vol is not None:
                weights[positions] /= vol[positions]
                contrasts["normalized_" + label] = _contrast(weights, ci, "premium_per_factor_vol")
        if vol is not None and full_rank:
            rho, eigenvalues, vectors, mutual_vif = _pair_geometry(
                geometry, positions, vol[positions]
            )
            pair.update(
                geometry_status="identified",
                partial_loading_cosine=rho,
                mutual_variance_inflation=mutual_vif,
                angular_eigen_ratio=(1 - abs(rho)) / (1 + abs(rho)),
                information_min=float(eigenvalues[0]),
                information_max=float(eigenvalues[1]),
                information_ratio=float(eigenvalues[0] / eigenvalues[1]),
                weak_first=float(vectors[0, 0]),
                weak_second=float(vectors[1, 0]),
                direction_status="not_unique"
                if np.isclose(eigenvalues[0], eigenvalues[1], rtol=1e-8, atol=0.0)
                else "unique",
            )
            for j, label in enumerate(("weak", "strong")):
                weights = np.zeros(k)
                weights[positions] = vectors[:, j] / vol[positions]
                directions[label] = dict(
                    _contrast(weights, ci, "premium_per_factor_vol"),
                    information=float(eigenvalues[j]),
                    first=float(vectors[0, j]),
                    second=float(vectors[1, j]),
                )
        elif vol is None:
            pair["geometry_status"] = "unavailable_factor_covariance"
        if vol is not None and ci.summary["status"] == "conditional_working_model":
            spread = contrasts["normalized_spread"]
            estimate, se = spread["estimate"], spread["standard_error"]
            df = ci.summary["residual_df"]
            statistic = (estimate / se) ** 2
            pair.update(
                equality_status="conditional_working_model",
                equality_f=float(statistic),
                equality_p=float(f.sf(statistic, 1, df)),
                equality_df_num=1,
                equality_df_den=df,
            )
            if equivalence_margin is not None and confidence_level > 0.5:
                critical = t.ppf(confidence_level, df)
                lower, upper = estimate - critical * se, estimate + critical * se
                pvalue = max(
                    t.sf((estimate + equivalence_margin) / se, df),
                    t.cdf((estimate - equivalence_margin) / se, df),
                )
                pair.update(
                    equivalence_status="equivalent"
                    if pvalue < 1 - confidence_level
                    else "not_established",
                    equivalence_p=float(pvalue),
                    equivalence_lower=float(lower),
                    equivalence_upper=float(upper),
                    equivalence_confidence=2 * confidence_level - 1,
                    equivalence_test_level=1 - confidence_level,
                )
    return CmaIdentificationResult(
        table,
        pd.DataFrame.from_dict(contrasts, orient="index"),
        pd.DataFrame.from_dict(directions, orient="index"),
        pair,
        summary,
    )
