"""Independent GLS geometry, covariance propagation and paired stability checks."""

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm, t

from optimalportfolios.matf.gls_diagnostic import (
    compute_gls_resolution, compute_gls_uncertainty, compute_gls_reference_uncertainty,
    compute_gls_adviser_disagreement, compute_gls_provider_contrast,
    summarize_gls_loading_stability, compute_gls_loading_stability,
    compute_gls_loading_stability_contrast, infer_factor_premia,
)


def model():
    """Identified heterogeneous design with three residual dimensions."""
    b = pd.DataFrame([[1., .2], [.4, 1.], [-.3, .8], [.9, -.2], [.6, .7]],
                     index=list("abcde"), columns=["Equity", "Rates"])
    d = pd.Series([.01, .02, .03, .015, .025], index=b.index)
    y = pd.Series([.09, .06, .04, .08, .055], index=b.index)
    vol = pd.Series([.15, .07], index=b.columns)
    design = b.to_numpy()
    a = np.linalg.solve(design.T @ np.diag(1 / d) @ design, design.T @ np.diag(1 / d))
    return b, d, y, vol, a


def test_resolution_normal_equations_qr_rescaling_and_default_ci_preserved():
    """Resolution is geometric, normalized units are invariant and M0 stays identical."""
    b, d, y, vol, a = model()
    old = infer_factor_premia(b, d, y, cash_rate=.02)
    result = compute_gls_resolution(b, d, factor_volatility=vol, conditional_result=old)
    qr, _ = np.linalg.qr(b.to_numpy() / np.sqrt(d.to_numpy())[:, None], mode="reduced")
    np.testing.assert_allclose(result.assets.fitted_leverage, np.sum(qr ** 2, axis=1))
    assert result.assets.residual_visibility.sum() == pytest.approx(3)
    inverse = np.linalg.inv(b.to_numpy().T @ np.diag(1 / d) @ b.to_numpy())
    np.testing.assert_allclose(result.factors.residual_mimicking_variance, np.diag(inverse))
    np.testing.assert_allclose(result.factors.normalized_resolution_per_unit_scale,
                               np.sqrt(np.diag(inverse)) / vol)
    np.testing.assert_allclose(result.factors.gross_mimicking_weight, np.abs(a).sum(axis=1))
    np.testing.assert_allclose(result.factors.normalized_ci_half_width,
                               (old.factors.upper - old.factors.lower) / (2 * vol))
    scaled = compute_gls_resolution(b / [4., .25], d, factor_volatility=vol * [4., .25])
    np.testing.assert_allclose(scaled.factors.normalized_resolution_per_unit_scale,
                               result.factors.normalized_resolution_per_unit_scale)
    another = infer_factor_premia(b, d, y, cash_rate=.02, alpha=y / 5, alpha_weight=1)
    changed = compute_gls_resolution(b, d, factor_volatility=vol, conditional_result=another)
    pd.testing.assert_frame_equal(result.assets, changed.assets)
    repeated = infer_factor_premia(b, d, y, cash_rate=.02)
    pd.testing.assert_frame_equal(old.factors, repeated.factors)
    pd.testing.assert_frame_equal(old.covariance, repeated.covariance)
    assert old.summary == repeated.summary
    with pytest.raises(FrozenInstanceError):
        result.summary = {}


def test_resolution_dense_rank_deficient_and_exhausted_residual_space():
    """Coordinate conventions and unavailable rank are explicit, including N=K."""
    b, d, _, _, _ = model()
    dense = compute_gls_resolution(b, np.diag(d) + .002 * np.ones((5, 5)))
    assert dense.summary["visibility_basis"] == "ordered_cholesky_coordinates"
    assert "not an individual" in dense.summary["asset_visibility_interpretation"]
    assert dense.assets.residual_visibility.between(0, 1).all()
    duplicate = b.copy()
    duplicate.Rates = duplicate.Equity
    deficient = compute_gls_resolution(duplicate, d)
    assert deficient.summary["rank"] == 1
    assert deficient.factors.residual_mimicking_variance.isna().all()
    assert deficient.assets.residual_visibility.sum() == pytest.approx(4)
    square = compute_gls_resolution(np.eye(2), np.ones(2))
    np.testing.assert_allclose(square.assets.residual_visibility, 0, atol=1e-15)
    assert square.summary["residual_df"] == 0


def test_resolution_cma_adapter_contract_and_geometry_mismatch():
    """CMA adapters work structurally and cannot attach unrelated covariance."""
    from optimalportfolios.matf.cma_uncertainty import CmaConditionalGlsResult

    b, d, y, vol, _ = model()
    old = infer_factor_premia(b, d, y, cash_rate=.02)
    adapter = CmaConditionalGlsResult(old.factors, old.covariance, old.summary)
    result = compute_gls_resolution(b, d, conditional_result=adapter)
    assert result.factors.normalized_ci_half_width.isna().all()
    with pytest.raises(ValueError, match="covariance"):
        compute_gls_resolution(b, d * 2, conditional_result=adapter)
    with pytest.raises(ValueError, match="geometry"):
        compute_gls_resolution(b, d, conditional_result=adapter, rcond=1e-6)
    for kwargs in ({"annual_precision_target": 0}, {"normalized_precision_target": -1},
                   {"conditional_result": object()}, {"factor_volatility": vol * 0},
                   {"factor_volatility": [1.]}, {"factor_volatility": vol.iloc[::-1]}):
        with pytest.raises(ValueError):
            compute_gls_resolution(b, d, **kwargs)


@pytest.mark.parametrize("singular", [False, True])
def test_covariance_propagation_matches_direct_equations_and_gaussian_simulation(singular):
    """Known covariance includes cross terms and singular forecast scenarios."""
    b, d, y, _, a = model()
    rng = np.random.default_rng(791)
    root = rng.normal(size=(5, 2 if singular else 5)) * .001
    covariance = root @ root.T
    premium = pd.Series(a @ (y - .02), index=b.columns)
    result = compute_gls_uncertainty(b, d, premium, covariance, distribution="normal")
    expected = a @ covariance @ a.T
    np.testing.assert_allclose(result.covariance, expected, rtol=1e-12, atol=1e-18)
    shocks = rng.normal(size=(60000, root.shape[1])) @ root.T
    np.testing.assert_allclose(np.cov((shocks @ a.T).T), expected, rtol=.025, atol=1e-8)
    np.testing.assert_allclose(result.factors.ci_upper,
                               premium + norm.ppf(.975) * np.sqrt(np.diag(expected)))
    white = b.to_numpy() / np.sqrt(d.to_numpy())[:, None]
    q, _ = np.linalg.qr(white, mode="reduced")
    m = np.eye(5) - q @ q.T
    cwhite = covariance / np.sqrt(np.outer(d, d))
    assert result.summary["expected_centered_residual_score"] == pytest.approx(np.trace(m @ cwhite))
    assert sum(result.summary["centered_residual_chi_square_weights"]) == pytest.approx(
        np.trace(m @ cwhite))
    no_interval = compute_gls_uncertainty(b, d, premium, covariance)
    assert no_interval.factors.ci_lower.isna().all()
    assert no_interval.factors.standard_error.notna().all()


def test_reference_scale_normal_t_and_current_proportional_model():
    """Known and independently estimated reference scales use different quantiles."""
    b, d, y, _, _ = model()
    m0 = infer_factor_premia(b, d, y, cash_rate=.02)
    common = dict(scale_squared=m0.summary["scale_squared"], reference_provenance="fixed source")
    normal = compute_gls_reference_uncertainty(b, d, m0.premia, **common)
    student = compute_gls_reference_uncertainty(b, d, m0.premia, **common,
        reference_degrees_of_freedom=3, independent_reference=True)
    np.testing.assert_allclose(student.covariance, m0.covariance)
    np.testing.assert_allclose(student.factors.ci_lower, m0.factors.lower)
    np.testing.assert_allclose((student.factors.ci_upper - m0.premia) /
        (normal.factors.ci_upper - m0.premia), t.ppf(.975, 3) / norm.ppf(.975))
    with pytest.raises(ValueError, match="independence"):
        compute_gls_reference_uncertainty(b, d, m0.premia, **common,
                                          reference_degrees_of_freedom=3)
    with pytest.raises(ValueError, match="nonnegative"):
        compute_gls_reference_uncertainty(b, d, m0.premia, scale_squared=-1,
                                          reference_provenance="fixed")


def test_span_and_residual_shocks_separate_mean_revisions_from_residual_visibility():
    """Pure factor shifts leave residual scores unchanged and residual shocks leave premia."""
    b, d, y, _, a = model()
    result = infer_factor_premia(b, d, y, cash_rate=.02)
    shift = np.array([.01, 0])
    span = infer_factor_premia(b, d, y + b @ shift, cash_rate=.02)
    np.testing.assert_allclose(span.premia - result.premia, shift, atol=1e-16)
    assert span.summary["weighted_residual_sse"] == pytest.approx(
        result.summary["weighted_residual_sse"])
    residual = (np.eye(5) - b.to_numpy() @ a) @ np.array([.01, 0, 0, 0, 0])
    res = infer_factor_premia(b, d, y + residual, cash_rate=.02)
    np.testing.assert_allclose(res.premia, result.premia, atol=1e-16)


def test_adviser_disagreement_is_separate_correlated_scenario_without_divisor():
    """IQR conversion and full cross-asset propagation retain their dispersion meaning."""
    b, d, y, _, a = model()
    lo, hi = y - .01, y + .02
    premium = pd.Series(a @ (y - .02), index=b.columns)
    result = compute_gls_adviser_disagreement(b, d, premium, lo, hi,
        correlation=.5, source_provenance="43 adviser survey, annual arithmetic conversion")
    sigma = .03 / (norm.ppf(.75) - norm.ppf(.25))
    covariance = sigma ** 2 * (.5 * np.eye(5) + .5 * np.ones((5, 5)))
    np.testing.assert_allclose(result.covariance, a @ covariance @ a.T)
    assert result.summary["adviser_count_divisor"] is None
    assert "not_confidence_interval" in result.summary["calibration_status"]
    diagonal = compute_gls_adviser_disagreement(b, d, premium, lo, hi, source_provenance="survey")
    supplied = compute_gls_adviser_disagreement(b, d, premium, lo, hi,
                                               correlation=np.eye(5), source_provenance="survey")
    pd.testing.assert_frame_equal(diagonal.covariance, supplied.covariance)
    for kwargs in ({"correlation": -.5}, {"correlation": 2.}, {"correlation": 2 * np.eye(5)},
                   {"source_provenance": ""}):
        with pytest.raises(ValueError):
            compute_gls_adviser_disagreement(b, d, premium, lo, hi,
                                             **dict({"source_provenance": "survey"}, **kwargs))
    with pytest.raises(ValueError, match="must not exceed"):
        compute_gls_adviser_disagreement(b, d, premium, hi, lo, source_provenance="survey")


def test_provider_contrast_same_and_different_maps_uses_cross_covariance():
    """Provider covariance contrasts retain off-diagonal factors and paired dependence."""
    b, d, y, _, a = model()
    premium = pd.Series(a @ (y - .02), index=b.columns)
    c = np.diag(d) * .01
    for rho in (0, .8, 1):
        result = compute_gls_provider_contrast(b, d, premium, b, d, premium / 2,
            covariance_left=c, covariance_right=c, cross_covariance=rho * c,
            left_provider="A", right_provider="B", distribution="normal")
        np.testing.assert_allclose(result.covariance, 2 * (1 - rho) * a @ c @ a.T, atol=1e-18)
        np.testing.assert_allclose(result.factors.estimate, premium / 2)
    other, risk = b.iloc[:4], d.iloc[:4]
    other_map = np.linalg.solve(other.to_numpy().T @ np.diag(1 / risk) @ other.to_numpy(),
                               other.to_numpy().T @ np.diag(1 / risk))
    cross = .1 * np.sqrt(np.outer(d, risk)) * .01
    result = compute_gls_provider_contrast(b, d, premium, other, risk, premium,
        covariance_left=c, covariance_right=np.diag(risk) * .01, cross_covariance=cross,
        left_provider="A", right_provider="B")
    expected = a @ c @ a.T + other_map @ (np.diag(risk) * .01) @ other_map.T
    expected -= a @ cross @ other_map.T + other_map @ cross.T @ a.T
    np.testing.assert_allclose(result.covariance, expected)
    with pytest.raises(ValueError, match="positive semidefinite"):
        compute_gls_provider_contrast(b, d, premium, b, d, premium,
            covariance_left=c, covariance_right=c, cross_covariance=2 * c,
            left_provider="A", right_provider="B")


@pytest.mark.parametrize("kind", ["negative", "asymmetric", "shape", "nan", "labels", "columns",
                                  "complex", "bad_units", "confidence", "distribution", "df",
                                  "reference", "assumptions", "provenance", "method", "premia"])
def test_uncertainty_rejects_malformed_models(kind):
    """Covariances, labels, units and distribution assumptions cannot be silently repaired."""
    b, d, _, _, _ = model()
    c = pd.DataFrame(np.diag(d), index=b.index, columns=b.index)
    kwargs, premium = {}, pd.Series([.03, .01], index=b.columns)
    if kind == "negative":
        c.iloc[0, 0] = -1
    elif kind == "asymmetric":
        c.iloc[0, 1] = 1
    elif kind == "shape":
        c = np.eye(4)
    elif kind == "nan":
        c.iloc[0, 0] = np.nan
    elif kind == "labels":
        c = c.iloc[::-1]
    elif kind == "columns":
        c = c.iloc[:, ::-1]
    elif kind == "complex":
        c = c.astype(complex)
    elif kind == "bad_units":
        kwargs["units"] = "monthly_decimal"
    elif kind == "confidence":
        kwargs["confidence_level"] = 1
    elif kind == "distribution":
        kwargs["distribution"] = "automatic"
    elif kind == "df":
        kwargs["reference_degrees_of_freedom"] = 3
    elif kind == "reference":
        kwargs["independent_reference"] = "yes"
    elif kind == "assumptions":
        kwargs["assumptions"] = "Gaussian"
    elif kind == "provenance":
        kwargs["reference_provenance"] = ""
    elif kind == "method":
        kwargs["method"] = ""
    else:
        premium = premium.iloc[::-1]
    with pytest.raises(ValueError):
        compute_gls_uncertainty(b, d, premium, c, **kwargs)


def test_scenario_deficient_rank_has_no_pseudoinverse_certainty():
    """A deficient design never gets finite coordinate uncertainty from external covariance."""
    b, d, _, _, _ = model()
    b.Rates = b.Equity
    result = compute_gls_uncertainty(b, d, [1, 1], np.diag(d), distribution="normal")
    assert result.factors.isna().all().all()
    assert result.summary["interval_status"] == "unavailable_rank_deficient"


def test_extreme_unit_mismatch_cannot_produce_infinite_report_intervals():
    """Unrepresentable covariance after GLS amplification is rejected explicitly."""
    with np.errstate(over="ignore", invalid="ignore"):
        with pytest.raises(ValueError, match="overflow"):
            compute_gls_uncertainty(np.ones((2, 1)) * 1e-160, np.ones(2), [0.],
                                    np.eye(2), distribution="normal")


def test_paired_bootstrap_matches_lstsq_and_keeps_rank_solver_geometry_failures():
    """One paired model design serves all providers and failed draws retain denominator."""
    b, d, y, _, _ = model()
    forecasts = pd.DataFrame({"A": y, "B": y * 1.2})
    beta = np.repeat(b.to_numpy()[None], 6, axis=0)
    risks = np.repeat(d.to_numpy()[None], 6, axis=0)
    beta[1, :, 0] *= 1.1
    beta[2, :, 1] = beta[2, :, 0]
    risks[3] = -1
    alpha = np.repeat((y / 10).to_numpy()[None], 6, axis=0)
    alpha[1] *= 2
    alpha[5] = np.nan
    kwargs = dict(B_draws=beta, Omega_draws=risks, cash_rate=.02, alpha=y / 10,
        alpha_draws=alpha, alpha_weight=.5, baseline_provenance="common quarters",
        solver_success=np.array([True, True, True, True, False, True]), seeds=[42] * 3 + [43] * 3)
    results = compute_gls_loading_stability(b, d, forecasts,
                                             alpha_policy="paired_admission", **kwargs)
    result = results["A"]
    assert result.summary["n_available"] == 2
    assert result.summary["rank_failure_count"] == 1
    assert result.summary["solver_failure_count"] == 1
    assert result.summary["solver_status_provided"] is True
    assert result.summary["geometry_failure_count"] == 1
    assert result.summary["nonfinite_count"] == 1
    assert result.summary["availability_rate"] == pytest.approx(1 / 3)
    expected = np.linalg.lstsq(beta[1] / np.sqrt(risks[1])[:, None],
        (y - .02 - .5 * alpha[1]) / np.sqrt(risks[1]), rcond=None)[0]
    np.testing.assert_allclose(result.draws.iloc[1], expected)
    np.testing.assert_allclose(result.bands.lower, np.quantile(result.draws.iloc[:2], .025, axis=0))
    contrast = compute_gls_loading_stability_contrast(result, results["B"],
                                                      left_provider="A", right_provider="B")
    np.testing.assert_allclose(contrast.draws.iloc[:2],
                               result.draws.iloc[:2] - results["B"].draws.iloc[:2])
    assert contrast.summary["method"] == "paired_loading_stability_contrast"
    assert contrast.summary["solver_status_provided"] is True
    fixed = compute_gls_loading_stability(b, d, forecasts, alpha_policy="fixed_admission", **kwargs)
    assert fixed["A"].summary["n_available"] == 3
    assert not np.allclose(fixed["A"].draws.iloc[1], result.draws.iloc[1])
    with pytest.raises(ValueError, match="baseline, confidence"):
        compute_gls_loading_stability_contrast(result, fixed["B"],
                                               left_provider="A", right_provider="B")


def test_loading_stability_covariance_draws_single_provider_zero_alpha_and_empty_subset():
    """Diagonal and matrix residual inputs agree; absent successful draws stay unavailable."""
    b, d, y, _, a = model()
    beta = np.repeat(b.to_numpy()[None], 2, axis=0)
    omega = np.repeat(np.diag(d)[None], 2, axis=0)
    result = compute_gls_loading_stability(b, d, y, B_draws=beta, Omega_draws=omega,
        cash_rate=.02, alpha_policy="paired_admission", baseline_provenance="common")["provider"]
    np.testing.assert_allclose(result.bands.baseline, a @ (y - .02))
    assert result.summary["solver_status_provided"] is False
    comparison = compute_gls_loading_stability_contrast(
        result, result, left_provider="A", right_provider="A")
    assert comparison.summary["solver_status_provided"] is False
    empty = summarize_gls_loading_stability(np.full((2, 2), np.nan), [0., 0.], ranks=[1, 1],
        alpha_policy="fixed_admission", baseline_provenance="common")
    assert empty.summary["status"] == "unavailable_no_valid_draws"
    assert empty.summary["solver_status_provided"] is False
    assert empty.bands.lower.isna().all()
    zero = summarize_gls_loading_stability(np.zeros((2, 2)), [0, 0], ranks=[2, 2],
        alpha_policy="fixed_admission", baseline_provenance="common")
    assert zero.bands.lower.eq(0).all()


def test_finite_rank_deficient_and_solver_failed_estimates_do_not_enter_bands():
    """Finite pseudoinverse coordinates do not masquerade as identified bootstrap draws."""
    result = summarize_gls_loading_stability(
        np.array([[.02, .01], [.03, .02], [100., 100.], [-100., -100.]]), [.025, .015],
        ranks=[2, 2, 1, 2], solver_success=[True, True, True, False],
        alpha_policy="paired_admission", baseline_provenance="common")
    assert result.summary["n_available"] == 2
    assert result.summary["rank_failure_count"] == 1
    assert result.summary["solver_failure_count"] == 1
    np.testing.assert_allclose(result.bands['median'], [.025, .015])
    assert result.bands.upper.max() < .03


@pytest.mark.parametrize("name", [
    "GlsResolutionResult", "GlsUncertaintyScenario", "GlsLoadingStabilityResult",
    "compute_gls_resolution", "compute_gls_uncertainty", "compute_gls_reference_uncertainty",
    "compute_gls_adviser_disagreement", "compute_gls_provider_contrast",
    "compute_gls_loading_stability", "compute_gls_loading_stability_contrast",
    "summarize_gls_loading_stability",
])
def test_new_public_exports_are_the_canonical_op_objects(name):
    """Package and MATF exports retain the numerical owner's exact callable objects."""
    import optimalportfolios as op
    from optimalportfolios.matf import gls_diagnostic

    assert getattr(op, name) is getattr(gls_diagnostic, name)
    assert getattr(op.matf, name) is getattr(gls_diagnostic, name)


@pytest.mark.parametrize("key,value", [
    ("premia_draws", []), ("premia_draws", np.ones((3, 2), dtype=complex)),
    ("ranks", [1.5, 2, 2]), ("ranks", [3, 2, 2]), ("ranks", [np.inf, 2, 2]),
    ("ranks", [2, 2]), ("ranks", np.ones(3, dtype=complex)),
    ("solver_success", [1, 1, 1]), ("minimum_singular_values", [-1, 0, 0]),
    ("seeds", [42, 43]), ("seeds", [42, None, 44]), ("confidence_level", 0),
    ("alpha_policy", ""), ("baseline_provenance", ""),
])
def test_stability_validation(key, value):
    """Failures remain explicit and malformed draw metadata cannot alter pairings."""
    kwargs = dict(premia_draws=np.ones((3, 2)), baseline=[.1, .2], ranks=[2, 2, 2],
                  alpha_policy="paired_admission", baseline_provenance="common")
    kwargs[key] = value
    with pytest.raises(ValueError):
        summarize_gls_loading_stability(**kwargs)


@pytest.mark.parametrize("key,value", [
    ("alpha_policy", "wrong"), ("alpha_weight", 2), ("alpha_weight", .5),
    ("B_draws", np.ones((2, 4, 2))), ("Omega_draws", np.ones((2, 4))),
    ("B_draws", np.ones((2, 5, 2), dtype=complex)),
    ("alpha_draws", np.ones((2, 4))), ("alpha_draws", np.ones((2, 5), dtype=complex)),
    ("solver_success", [1, 1]),
])
def test_loading_refit_validation(key, value):
    """Rejected inputs cannot silently substitute alpha, model dimensions or solver state."""
    b, d, y, _, _ = model()
    kwargs = dict(B_draws=np.repeat(b.to_numpy()[None], 2, axis=0),
        Omega_draws=np.repeat(d.to_numpy()[None], 2, axis=0),
        cash_rate=.02, baseline_provenance="common")
    kwargs[key] = value
    with pytest.raises(ValueError):
        compute_gls_loading_stability(b, d, y, **kwargs)
    with pytest.raises(ValueError, match="requires alpha_draws"):
        compute_gls_loading_stability(b, d, y, **dict(kwargs, alpha=y / 10, alpha_weight=.5,
            alpha_policy="paired_admission", alpha_draws=None, B_draws=np.repeat(
                b.to_numpy()[None], 2, axis=0),
            Omega_draws=np.repeat(d.to_numpy()[None], 2, axis=0),
            solver_success=np.ones(2, dtype=bool)))


def test_contrast_labels_and_bootstrap_pairing_rejections():
    """Misordered assets or draws cannot produce apparently precise paired differences."""
    b, d, y, _, _ = model()
    premium = pd.Series([.03, .01], index=b.columns)
    kwargs = dict(covariance_left=np.diag(d), covariance_right=np.diag(d),
        cross_covariance=np.zeros((5, 5)), left_provider="A", right_provider="B")
    with pytest.raises(ValueError, match="factor labels"):
        compute_gls_provider_contrast(b, d, premium, b.iloc[:, ::-1], d, premium, **kwargs)
    with pytest.raises(ValueError, match="dimensions"):
        compute_gls_provider_contrast(b, d, premium, b, d, premium,
                                      **dict(kwargs, cross_covariance=np.zeros((5, 4))))
    cross = pd.DataFrame(np.zeros((5, 5)), index=b.index, columns=b.index[::-1])
    with pytest.raises(ValueError, match="label/order"):
        compute_gls_provider_contrast(b, d, premium, b, d, premium,
                                      **dict(kwargs, cross_covariance=cross))
    basic = summarize_gls_loading_stability(pd.DataFrame([[1, 2], [2, 3]], columns=b.columns),
        premium, ranks=[2, 2], alpha_policy="fixed_admission", baseline_provenance="common")
    for other in (None, replace(basic, draws=basic.draws.iloc[::-1]),
                  replace(basic, draw_status=basic.draw_status.assign(seed=42)),
                  replace(basic, draw_status=basic.draw_status.assign(rank=1))):
        with pytest.raises(ValueError):
            compute_gls_loading_stability_contrast(basic, other,
                                                   left_provider="A", right_provider="B")
    with pytest.raises(ValueError, match="full factor rank"):
        compute_gls_loading_stability(b.assign(Rates=b.Equity), d, y,
            B_draws=np.ones((2, 5, 2)), Omega_draws=np.ones((2, 5)),
            cash_rate=.02, baseline_provenance="common")
