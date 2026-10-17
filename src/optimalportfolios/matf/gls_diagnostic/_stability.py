"""Paired loading/alpha stability bands with failures retained in their denominator."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from optimalportfolios.matf.gls_diagnostic._inference import (
    _array, _axis, _scalar, _vector, compute_gls_geometry,
)
from optimalportfolios.matf.gls_diagnostic._resolution import _factor_vector, _labels
from optimalportfolios.matf.gls_diagnostic._scenarios import _text


@dataclass(frozen=True)
class GlsLoadingStabilityResult:
    """Percentile stability bands, retained paired draws and explicit draw quality.

    These bands condition on the finite, full-rank successful subset. They are not
    calibrated confidence intervals for selected or constrained loading estimates.
    Baseline shifts must use the same common-history estimator convention as the
    draws; a native-history production estimate is a separate comparison.
    """

    bands: pd.DataFrame
    draws: pd.DataFrame
    draw_status: pd.DataFrame
    summary: dict


def _draw_vector(value, index, name, *, finite=False):
    """Validate a draw-level vector while retaining unavailable numerical entries."""
    _axis(value, index)
    if np.iscomplexobj(value):
        raise ValueError(f"{name} must be real")
    result = np.asarray(value, dtype=float)
    if result.shape != (len(index),) or (finite and not np.isfinite(result).all()):
        raise ValueError(f"{name} must match the draw index")
    return result


def summarize_gls_loading_stability(premia_draws, baseline, *, ranks, solver_success=None,
                                    minimum_singular_values=None, confidence_level=.95,
                                    alpha_policy, baseline_provenance, seeds=None
                                    ) -> GlsLoadingStabilityResult:
    """Summarize paired full-rank draws, retaining every failure and baseline shift.

    Draw rows must remain paired across providers; factor columns and baseline
    labels must match exactly. Ranks may be NaN for failed geometry. Rank-deficient
    or nonfinite draws never enter percentiles and remain in availability counts.
    Seed-specific endpoints are saved when seeds are supplied, without treating
    repeated seeds as independent evidence of confidence coverage.
    Omitted solver_success flags assume the cached rows succeeded; the returned
    solver_status_provided flag distinguishes this from observed solver metadata.
    """
    _text(alpha_policy, "alpha_policy")
    _text(baseline_provenance, "baseline_provenance")
    confidence = _scalar(confidence_level, "confidence_level")
    if not 0 < confidence < 1:
        raise ValueError("confidence_level must be strictly between zero and one")
    _axis(premia_draws)
    if isinstance(premia_draws, pd.DataFrame):
        _axis(premia_draws, axis=1)
    if np.iscomplexobj(premia_draws):
        raise ValueError("premia_draws must be real")
    values = np.asarray(premia_draws, dtype=float)
    if values.ndim != 2 or not all(values.shape):
        raise ValueError("premia_draws must be a nonempty draw-by-factor matrix")
    n, k = values.shape
    index = (premia_draws.index.copy() if isinstance(premia_draws, pd.DataFrame) else
             pd.RangeIndex(n, name="draw"))
    factors = (premia_draws.columns.copy() if isinstance(premia_draws, pd.DataFrame) else
               baseline.index.copy() if isinstance(baseline, pd.Series) else
               pd.RangeIndex(k, name="factor"))
    point = _factor_vector(baseline, factors, "baseline")
    rank = _draw_vector(ranks, index, "ranks")
    known = np.isfinite(rank)
    if np.isinf(rank).any() or np.any(rank[known] != np.floor(rank[known])) or np.any(
            (rank[known] < 0) | (rank[known] > k)):
        raise ValueError("ranks must be integer values between zero and n_factors or NaN")
    solver = np.ones(n, dtype=bool)
    if solver_success is not None:
        _axis(solver_success, index)
        solver = np.asarray(solver_success)
        if solver.dtype.kind != "b" or solver.shape != (n,):
            raise ValueError("solver_success must be a boolean vector matching draw rows")
    minimum = np.full(n, np.nan)
    if minimum_singular_values is not None:
        minimum = _draw_vector(minimum_singular_values, index, "minimum_singular_values")
        if np.isinf(minimum).any() or np.any(minimum < 0):
            raise ValueError("minimum_singular_values must be nonnegative or NaN")
    finite = np.isfinite(values).all(axis=1)
    available = solver & (rank == k) & finite
    status = np.select([~solver, ~known, rank < k, ~finite],
                       ["solver_failure", "geometry_failure", "rank_deficient", "nonfinite"],
                       default="available")
    quality = pd.DataFrame(dict(rank=rank, solver_success=solver, finite=finite,
        minimum_singular_value=minimum, available=available, status=status), index=index)
    quantile = np.full((3, k), np.nan)
    probabilities = [(1 - confidence) / 2, .5, (1 + confidence) / 2]
    if available.any():
        quantile = np.quantile(values[available], probabilities, axis=0)
    lower, median, upper = quantile
    bands = pd.DataFrame(dict(baseline=point, median=median, lower=lower, upper=upper,
        median_shift=median-point, lower_shift=lower-point, upper_shift=upper-point), index=factors)
    seed_bands = []
    if seeds is not None:
        _axis(seeds, index)
        seed_values = np.asarray(seeds)
        if seed_values.shape != (n,) or pd.isna(seed_values).any():
            raise ValueError("seeds must be nonmissing and match the draw index")
        quality["seed"] = seed_values
        for seed in pd.unique(seed_values):
            use = available & (seed_values == seed)
            endpoints = (np.quantile(values[use], probabilities, axis=0) if use.any() else
                         np.full((3, k), np.nan))
            for j, factor in enumerate(factors):
                seed_bands.append(dict(seed=seed.item() if hasattr(seed, "item") else seed,
                    factor=factor, n_available=int(use.sum()), lower=float(endpoints[0, j]),
                    median=float(endpoints[1, j]), upper=float(endpoints[2, j])))
    summary = dict(method="paired_loading_stability", confidence_level=confidence,
        alpha_policy=alpha_policy, baseline_provenance=baseline_provenance,
        n_draws=n, n_factors=k, n_available=int(available.sum()),
        availability_rate=float(available.mean()),
        rank_failure_count=int(np.sum(status == "rank_deficient")),
        solver_failure_count=int(np.sum(status == "solver_failure")),
        solver_status_provided=solver_success is not None,
        geometry_failure_count=int(np.sum(status == "geometry_failure")),
        nonfinite_count=int(np.sum(status == "nonfinite")),
        calibrated_ci=False, units="annual_decimal", seed_bands=seed_bands,
        status="finite_full_rank_subset" if available.any() else "unavailable_no_valid_draws",
        assumptions=("Provider forecasts are fixed; loading, residual covariance and, when "
                     "declared, admitted alpha are resampled jointly.",
                     "Percentile bands use finite full-rank successful draws only; availability "
                     "is reported against every attempted draw.",
                     "Constrained and selected estimation can invalidate ordinary bootstrap "
                     "confidence coverage. These are stability bands, not calibrated CIs.",
                     "Baseline shifts require the same common-history fitting convention."))
    return GlsLoadingStabilityResult(bands, pd.DataFrame(values.copy(), index=index,
                                                        columns=factors), quality, summary)


def compute_gls_loading_stability(B, Omega, total_returns, *, B_draws, Omega_draws,
                                  cash_rate, alpha=None, alpha_draws=None, alpha_weight=0.,
                                  alpha_policy="fixed_admission", baseline_provenance,
                                  solver_success=None, confidence_level=.95, seeds=None,
                                  rcond=None) -> dict[str, GlsLoadingStabilityResult]:
    """Apply one GLS SVD per draw to all providers, retaining paired alpha policy.

    B/Omega/alpha are the matching baseline. ``total_returns`` is an asset vector
    or asset-by-provider DataFrame. B_draws has shape (draw, asset, factor);
    Omega_draws has shape (draw, asset) or (draw, asset, asset). Draw arrays use
    exactly the baseline's asset/factor order. Alpha_draws contains admitted alpha
    before the common diagnostic alpha_weight and is paired with B and Omega.
    ``fixed_admission`` holds baseline alpha fixed; ``paired_admission`` requires
    alpha_draws when alpha_weight is nonzero. A failed draw remains in each result.
    """
    _text(baseline_provenance, "baseline_provenance")
    if alpha_policy not in ("fixed_admission", "paired_admission"):
        raise ValueError("alpha_policy must be fixed_admission or paired_admission")
    cash = _scalar(cash_rate, "cash_rate")
    weight = _scalar(alpha_weight, "alpha_weight")
    if not 0 <= weight <= 1 or (weight != 0 and alpha is None):
        raise ValueError("alpha_weight must be in [0,1] and nonzero requires baseline alpha")
    base = compute_gls_geometry(B, Omega, rcond=rcond)
    assets, factors = _labels(B)
    n, k = base.beta.shape
    if base.rank != k:
        raise ValueError("bootstrap baseline must have full factor rank")
    _axis(total_returns, assets)
    if isinstance(total_returns, pd.DataFrame):
        _axis(total_returns, axis=1)
        forecasts = _array(total_returns, "total_returns", 2)
        providers = total_returns.columns
    else:
        forecasts = _vector(total_returns, B, "total_returns")[:, None]
        providers = pd.Index(["provider"])
    if np.iscomplexobj(B_draws) or np.iscomplexobj(Omega_draws):
        raise ValueError("B_draws and Omega_draws must be real")
    betas, residuals = np.asarray(B_draws, dtype=float), np.asarray(Omega_draws, dtype=float)
    if betas.ndim != 3 or betas.shape[1:] != (n, k) or not len(betas):
        raise ValueError("B_draws must have shape (draw, asset, factor)")
    z = len(betas)
    if residuals.shape not in ((z, n), (z, n, n)):
        raise ValueError("Omega_draws must have shape (draw, asset) or (draw, asset, asset)")
    admission = np.zeros(n) if alpha is None else _vector(alpha, B, "alpha")
    paired_alpha = None
    if alpha_draws is not None:
        if np.iscomplexobj(alpha_draws):
            raise ValueError("alpha_draws must be real")
        paired_alpha = np.asarray(alpha_draws, dtype=float)
        if paired_alpha.shape != (z, n):
            raise ValueError("alpha_draws must have shape (draw, asset)")
    if alpha_policy == "paired_admission" and weight and paired_alpha is None:
        raise ValueError("paired_admission with nonzero weight requires alpha_draws")
    solver = np.ones(z, dtype=bool) if solver_success is None else np.asarray(solver_success)
    if solver.dtype.kind != "b" or solver.shape != (z,):
        raise ValueError("solver_success must be a boolean vector matching draw rows")
    target = forecasts - cash - weight * admission[:, None]
    baseline = base.white_pinv @ np.linalg.solve(base.chol, target)
    estimates = np.full((z, k, len(providers)), np.nan)
    ranks, minimum = np.full(z, np.nan), np.full(z, np.nan)
    for draw in range(z):
        if not solver[draw]:
            continue
        try:
            geometry = compute_gls_geometry(betas[draw], residuals[draw], rcond=rcond)
        except (ValueError, np.linalg.LinAlgError):
            continue
        ranks[draw] = geometry.rank
        minimum[draw] = geometry.singular_values[-1]
        if geometry.rank != k:
            continue
        draw_target = target
        if alpha_policy == "paired_admission" and weight:
            draw_target = forecasts - cash - weight * paired_alpha[draw, :, None]
        estimates[draw] = geometry.white_pinv @ np.linalg.solve(geometry.chol, draw_target)
    results = {}
    for j, provider in enumerate(providers):
        result = summarize_gls_loading_stability(pd.DataFrame(estimates[:, :, j], columns=factors),
            pd.Series(baseline[:, j], index=factors), ranks=ranks, solver_success=solver,
            minimum_singular_values=minimum, confidence_level=confidence_level,
            alpha_policy=alpha_policy, baseline_provenance=baseline_provenance, seeds=seeds)
        results[provider] = GlsLoadingStabilityResult(
            result.bands, result.draws, result.draw_status,
            dict(result.summary, provider=provider, alpha_weight=weight, cash_rate=cash,
                 relative_cutoff=base.relative_cutoff,
                 solver_status_provided=solver_success is not None))
    return results


def compute_gls_loading_stability_contrast(left, right, *, left_provider, right_provider
                                          ) -> GlsLoadingStabilityResult:
    """Difference aligned provider draws before taking quantiles; never subtract bands."""
    _text(left_provider, "left_provider")
    _text(right_provider, "right_provider")
    if not isinstance(left, GlsLoadingStabilityResult) or not isinstance(
            right, GlsLoadingStabilityResult):
        raise ValueError("contrasts require two GlsLoadingStabilityResult objects")
    if not left.draws.index.equals(right.draws.index) or not left.draws.columns.equals(
            right.draws.columns):
        raise ValueError("paired provider draw and factor labels/order must match")
    keys = ("confidence_level", "alpha_policy", "baseline_provenance")
    if any(left.summary[key] != right.summary[key] for key in keys):
        raise ValueError("paired provider baseline, confidence and alpha policy must match")
    columns = ["rank", "solver_success", "minimum_singular_value"]
    if "seed" in left.draw_status or "seed" in right.draw_status:
        columns.append("seed")
    if any(column not in left.draw_status or column not in right.draw_status for column in columns):
        raise ValueError("paired provider draw quality columns must match")
    if not left.draw_status[columns].equals(right.draw_status[columns]):
        raise ValueError("paired provider draws must use the same model draws and seeds")
    difference = left.draws - right.draws
    result = summarize_gls_loading_stability(difference,
        left.bands.baseline - right.bands.baseline, ranks=left.draw_status["rank"],
        solver_success=left.draw_status.solver_success,
        minimum_singular_values=left.draw_status.minimum_singular_value,
        confidence_level=left.summary["confidence_level"],
        alpha_policy=left.summary["alpha_policy"],
        baseline_provenance=left.summary["baseline_provenance"],
        seeds=left.draw_status.get("seed"))
    return GlsLoadingStabilityResult(result.bands, result.draws, result.draw_status,
        dict(result.summary, method="paired_loading_stability_contrast",
             comparison_label=f"{left_provider} minus {right_provider}",
             left_provider=left_provider, right_provider=right_provider,
             solver_status_provided=bool(left.summary.get("solver_status_provided", False)
                                        and right.summary.get("solver_status_provided", False))))
