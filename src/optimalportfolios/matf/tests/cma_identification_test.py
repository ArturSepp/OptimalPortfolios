"""Independent restricted fits, contrast covariance and scale invariance of identification."""

from copy import deepcopy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from scipy.stats import t

from optimalportfolios.matf.cma_diagnostics import CmaDiagnosticInputs, CmaDiagnosticSpec
from optimalportfolios.matf.cma_identification import _contrast, compute_cma_identification
from optimalportfolios.matf.cma_uncertainty import compute_conditional_gls_intervals


def _inputs():
    """Use correlated carry loadings, a nuisance factor and heterogeneous residual risk."""
    b = pd.DataFrame(
        [
            [1.0, 0.1, 0.2],
            [0.2, 1.0, 0.4],
            [0.6, 0.2, 0.9],
            [0.4, 1.2, 0.5],
            [1.4, 0.4, 0.3],
            [-0.2, 0.6, 0.8],
            [0.2, 0.4, 1.1],
            [0.8, -0.2, -0.3],
        ],
        index=list("abcdefgh"),
        columns=["Carry G10", "Carry EM", "Equity"],
    )
    cov = np.array([[0.0025, 0.0007, 0.001], [0.0007, 0.0016, 0.0008], [0.001, 0.0008, 0.0225]])
    return CmaDiagnosticInputs(
        b,
        pd.Series([0.01, 0.02, 0.03, 0.015, 0.04, 0.025, 0.01, 0.03], b.index),
        pd.Series([0.04, 0.02, 0.03, 0.017, 0.07, -0.002, 0.01, 0.06], b.index),
        factor_covariance=pd.DataFrame(cov, index=b.columns, columns=b.columns),
    )


def test_contrasts_and_equality_match_independent_restricted_gls():
    """The full covariance matters; a separately fitted equality restriction gives the same F."""
    data = _inputs()
    before = deepcopy(data)
    result = compute_cma_identification(data, precision_target=0.01, equivalence_margin=0.10)
    weights = 1 / np.sqrt(data.residual_variances.to_numpy())
    design = data.betas.to_numpy() * weights[:, None]
    target = data.excess_cma.to_numpy() * weights
    q, r = np.linalg.qr(design, mode='reduced')
    beta = np.linalg.solve(r, q.T @ target)
    residual = target - design @ beta
    df = len(target) - len(beta)
    sse = residual @ residual
    scale = sse / df
    inverse_r = np.linalg.solve(r, np.eye(len(beta)))
    covariance = scale * inverse_r @ inverse_r.T
    vol = np.sqrt(np.diag(data.factor_covariance))
    for label, c in {
        "common": [0.5, 0.5, 0.0],
        "spread": [1.0, -1.0, 0.0],
        "normalized_common": [0.5 / vol[0], 0.5 / vol[1], 0.0],
        "normalized_spread": [1 / vol[0], -1 / vol[1], 0.0],
    }.items():
        c = np.asarray(c)
        estimate = c @ beta
        se = np.sqrt(c @ covariance @ c)
        row = result.contrasts.loc[label]
        np.testing.assert_allclose(
            [row.estimate, row.standard_error],
            [estimate, se],
            atol=1e-13,
        )
        width = t.ppf(.975, df) * se
        np.testing.assert_allclose([row.lower, row.upper],
                                   [estimate-width, estimate+width], atol=1e-13)
    pooled = np.column_stack((data.betas.Equity, data.betas.iloc[:, :2] @ vol[:2]))
    pooled *= weights[:, None]
    q_restricted, r_restricted = np.linalg.qr(pooled, mode='reduced')
    restricted_beta = np.linalg.solve(r_restricted, q_restricted.T @ target)
    restricted_residual = target - pooled @ restricted_beta
    restricted_sse = restricted_residual @ restricted_residual
    assert result.pair["equality_f"] == pytest.approx((restricted_sse - sse) / scale)
    np.testing.assert_allclose(result.factors.estimate, beta)
    for name in ("betas", "factor_covariance"):
        pd.testing.assert_frame_equal(getattr(data, name), getattr(before, name))
    pd.testing.assert_series_equal(data.excess_cma, before.excess_cma)


def test_information_matches_auxiliary_regressions_and_schur_inverse():
    """Check each partial information and the two-factor spectrum using a different derivation."""
    data = _inputs()
    result = compute_cma_identification(data)
    x = data.betas.to_numpy() / np.sqrt(data.residual_variances.to_numpy())[:, None]
    for j, name in enumerate(data.betas):
        other = np.delete(x, j, axis=1)
        q, r = np.linalg.qr(other, mode='reduced')
        residual = x[:, j] - other @ np.linalg.solve(r, q.T @ x[:, j])
        sse = residual @ residual
        assert result.factors.at[name, "conditional_information"] == pytest.approx(sse)
        assert result.factors.at[name, "variance_inflation"] == pytest.approx(
            (x[:, j] @ x[:, j]) / sse
        )
    gram = x.T @ x
    schur = gram[:2, :2] - gram[:2, 2:] @ np.linalg.solve(gram[2:, 2:], gram[2:, :2])
    vol = np.sqrt(np.diag(data.factor_covariance))[:2]
    expected = np.linalg.eigvalsh(vol[:, None] * schur * vol[None, :])
    np.testing.assert_allclose(result.directions.information, expected, atol=1e-14)
    for row in result.directions.itertuples():
        assert row.standard_error**2 == pytest.approx(
            result.summary["scale_squared"] / row.information
        )
    rho = result.pair["partial_loading_cosine"]
    assert result.pair["mutual_variance_inflation"] == pytest.approx(1 / (1 - rho * rho))


def test_factor_units_and_common_residual_scale_do_not_change_normalized_inference():
    """Rescaling a factor's returns cannot manufacture stronger normalized identification."""
    data = _inputs()
    result = compute_cma_identification(data, equivalence_margin=0.10)
    scale = np.array([7.0, 0.4, 2.0])
    changed = replace(
        data,
        betas=data.betas / scale,
        factor_covariance=data.factor_covariance * scale[:, None] * scale[None, :],
    )
    actual = compute_cma_identification(changed, equivalence_margin=0.10)
    pd.testing.assert_frame_equal(
        actual.contrasts.loc[["normalized_common", "normalized_spread"]],
        result.contrasts.loc[["normalized_common", "normalized_spread"]],
    )
    pd.testing.assert_frame_equal(actual.directions, result.directions)
    weighted = compute_cma_identification(
        replace(data, residual_variances=data.residual_variances * 9)
    )
    pd.testing.assert_frame_equal(weighted.contrasts, result.contrasts)
    np.testing.assert_allclose(weighted.directions.information * 9, result.directions.information)


def test_equivalence_uses_ninety_percent_interval_and_never_accepts_equality_by_default():
    """Non-rejection of equality is distinct from proving equivalence within a declared margin."""
    data = _inputs()
    wide = compute_cma_identification(data, precision_target=100.0, equivalence_margin=100.0)
    tight = compute_cma_identification(data, precision_target=1e-9, equivalence_margin=1e-9)
    assert wide.pair["equivalence_status"] == "equivalent"
    assert tight.pair["equivalence_status"] == "not_established"
    row = wide.contrasts.loc["normalized_spread"]
    width = t.ppf(0.95, wide.summary["residual_df"]) * row.standard_error
    assert wide.pair["equivalence_lower"] == pytest.approx(row.estimate - width)
    assert wide.pair["equivalence_confidence"] == pytest.approx(0.90)
    assert wide.factors.precision_status.eq("within_target").all()
    assert tight.factors.precision_status.eq("weak_for_target").all()
    assert compute_cma_identification(data).pair["equivalence_status"] == "not_assessed"


@pytest.mark.parametrize("defect", ["rank", "df", "perfect", "missing", "covariance", "cutoff"])
def test_unavailable_cases_cannot_report_equivalence_or_precision(defect):
    """Unsupported intervals stay unavailable; valid geometry and native contrasts survive."""
    data, spec = _inputs(), CmaDiagnosticSpec()
    if defect == "rank":
        b = data.betas.copy()
        b["Carry EM"] = b["Carry G10"]
        data = replace(data, betas=b)
    elif defect == "df":
        data = replace(
            data,
            betas=data.betas.iloc[:3],
            residual_variances=data.residual_variances.iloc[:3],
            excess_cma=data.excess_cma.iloc[:3],
        )
    elif defect == "perfect":
        data = replace(data, excess_cma=data.betas @ np.array([0.02, 0.01, 0.03]))
    elif defect == "missing":
        data = replace(
            data, betas=data.betas.rename(columns={"Carry EM": "Other"}), factor_covariance=None
        )
    elif defect == "covariance":
        data = replace(data, factor_covariance=None)
    else:
        spec = CmaDiagnosticSpec(rcond=0.9)
    actual = compute_cma_identification(
        data, precision_target=0.01, equivalence_margin=0.1, spec=spec
    )
    assert actual.pair["equivalence_status"] == "unavailable"
    if defect not in ("missing", "covariance"):
        assert actual.contrasts[["lower", "upper"]].isna().all().all()
        assert actual.factors.precision_status.eq("unavailable").all()
    if defect == "missing":
        assert actual.contrasts.empty
    if defect == "covariance":
        assert set(actual.contrasts.index) == {"common", "spread"}


@pytest.mark.parametrize(
    "settings",
    [
        {"precision_target": 0},
        {"equivalence_margin": True},
        {"factor_pair": ("Carry G10", "Carry G10")},
        {"factor_pair": "Carry G10"},
        {"precision_target": [1]},
    ],
)
def test_invalid_settings_fail(settings):
    """An accidental boolean, ambiguous pair or nonpositive tolerance cannot become a policy."""
    with pytest.raises(ValueError):
        compute_cma_identification(_inputs(), **settings)


def test_factor_covariance_alignment_is_exact():
    """The volatility scaling must use the same labelled factor order as exposures."""
    data = _inputs()
    with pytest.raises(ValueError):
        compute_cma_identification(
            replace(data, factor_covariance=data.factor_covariance.iloc[::-1])
        )


def test_invalid_contrast_variance_cannot_report_a_pvalue():
    """Corrupted contrast covariance must fail before producing significance or equivalence."""
    ci = compute_conditional_gls_intervals(_inputs())
    corrupted = replace(ci, covariance=-ci.covariance)
    with pytest.raises(ValueError, match='variance'):
        _contrast(np.array([1., -1., 0.]), corrupted, 'annual_excess_decimal')
