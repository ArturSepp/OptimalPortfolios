"""GLS checked against normal equations, explicit endpoints and deficient designs."""

from itertools import product

import numpy as np
import pandas as pd
import pytest
from scipy.stats import t

from optimalportfolios.matf.gls_diagnostic import (
    compute_gls_geometry, factor_mimicking_portfolios, gls_projector, gls_split,
    infer_factor_premia,
)


def panel():
    """Small identified design with unequal residual variances and nonzero residual."""
    b = pd.DataFrame([[1., .2], [.4, 1.], [-.3, .8], [.9, -.2], [.6, .7]],
                     index=list("abcde"), columns=["Equity", "Rates"])
    d = pd.Series([.01, .02, .03, .015, .025], index=b.index)
    mu = pd.Series([.09, .06, .04, .08, .055], index=b.index)
    alpha = pd.Series([.003, 0, -.001, .004, .002], index=b.index)
    return b, d, mu, alpha


@pytest.mark.parametrize("correlated", [False, True])
@pytest.mark.parametrize("weight", [0., .5, 1.])
def test_estimates_intervals_and_financing_against_normal_equations(correlated, weight):
    """Estimates intervals and financing against normal equations."""
    b, d, mu, alpha = panel()
    omega = np.diag(d) + (.002 * np.ones((5, 5)) if correlated else 0)
    result = infer_factor_premia(b, omega, mu, cash_rate=.03, alpha=alpha, alpha_weight=weight)
    precision = np.linalg.inv(omega)
    gram = b.to_numpy().T @ precision @ b.to_numpy()
    mapping = np.linalg.solve(gram, b.to_numpy().T @ precision)
    target = mu - .03 - weight * alpha
    beta = mapping @ target
    resid = target - b @ beta
    scale = float(resid @ precision @ resid / 3)
    covariance = scale * np.linalg.inv(gram)
    np.testing.assert_allclose(result.premia, beta, rtol=1e-12)
    np.testing.assert_allclose(result.covariance, covariance, rtol=1e-12)
    np.testing.assert_allclose(result.factors.lower, beta-t.ppf(.975, 3)*np.sqrt(
        np.diag(covariance)), rtol=1e-12)
    np.testing.assert_allclose(b.T @ result.mimicking_weights, np.eye(2), atol=1e-14)
    np.testing.assert_allclose(result.assets.fitted + result.assets.residual
                               + result.assets.credited_alpha + result.assets.cash, mu)
    np.testing.assert_allclose(result.mimicking_weights.T @ target, result.premia)
    np.testing.assert_allclose(gls_split(target, b, omega)[0], beta)
    np.testing.assert_allclose(gls_projector(b, omega) @ target, resid, atol=1e-14)
    np.testing.assert_allclose(factor_mimicking_portfolios(b, omega), mapping.T)


def test_boxes_match_exhaustive_signed_endpoints_and_shared_cash():
    """Boxes match exhaustive signed endpoints and shared cash."""
    b, d, mu, alpha = panel()
    lo, hi = mu - .01, mu + .02
    result = infer_factor_premia(b, d, mu, cash_rate=.03, alpha=alpha, alpha_weight=.5,
                                 lower_returns=lo, upper_returns=hi, cash_bounds=(.02, .04))
    endpoints = []
    for choices in product([0, 1], repeat=6):
        forecast = np.where(choices[:5], hi, lo)
        cash = [.02, .04][choices[-1]]
        endpoints.append(np.linalg.lstsq(b.to_numpy()/np.sqrt(d.to_numpy())[:, None],
                         (forecast-cash-.5*alpha)/np.sqrt(d), rcond=None)[0])
    np.testing.assert_allclose(result.sensitivity_bounds.lower, np.min(endpoints, axis=0))
    np.testing.assert_allclose(result.sensitivity_bounds.upper, np.max(endpoints, axis=0))
    assert result.sensitivity_bounds.status.eq("sensitivity_box").all()


def test_rank_cutoff_is_on_whitened_design_not_squared_gram():
    """Rank cutoff is on whitened design not squared gram."""
    b = np.array([[1., 0.], [0., 1e-9], [0., 0.]])
    geometry = compute_gls_geometry(b, np.ones(3))
    assert geometry.rank == 2
    premia, resid = gls_split([.04, 2e-11, .001], b, np.ones(3))
    np.testing.assert_allclose(premia, [.04, .02])
    np.testing.assert_allclose(resid, [0, 0, .001])
    assert compute_gls_geometry(b, np.ones(3), rcond=1e-8).rank == 1


@pytest.mark.parametrize("kind, expected", [
    ("duplicate", "unavailable_rank_deficient"),
    ("zero", "unavailable_rank_deficient"),
    ("square", "unavailable_residual_df"),
    ("exact", "unavailable_zero_residual_scale"),
])
def test_no_false_precision(kind, expected):
    """No false precision."""
    b, d, mu, _ = panel()
    if kind == "duplicate":
        b["Rates"] = b.Equity
    elif kind == "zero":
        b *= 0
    elif kind == "square":
        b, d, mu = b.iloc[:2], d.iloc[:2], mu.iloc[:2]
    else:
        mu = b @ np.array([.04, .01]) + .03
    result = infer_factor_premia(b, d, mu, cash_rate=.03)
    assert result.summary["status"] == expected
    assert result.factors.lower.isna().all()
    if "rank" in expected:
        assert result.factors.estimate.isna().all()
        assert result.mimicking_weights.isna().all().all()
        with pytest.raises(ValueError, match="unattainable"):
            factor_mimicking_portfolios(b, d)
    np.testing.assert_allclose(result.assets.fitted+result.assets.residual+.03, mu)


def test_labels_invalid_risk_and_alpha_cannot_silently_change_the_target():
    """Labels invalid risk and alpha cannot silently change the target."""
    b, d, mu, alpha = panel()
    for kwargs in ({"alpha_weight": 1}, {"alpha_weight": 2}, {"confidence_level": 1},
                   {"confidence_level": True}, {"lower_returns": mu},
                   {"lower_returns": mu+.1, "upper_returns": mu},
                   {"cash_bounds": (.02, .04)}, {"rcond": -1}):
        with pytest.raises(ValueError):
            infer_factor_premia(b, d, mu, cash_rate=.03, **kwargs)
    with pytest.raises(ValueError, match="label/order"):
        infer_factor_premia(b, d, mu.iloc[::-1], cash_rate=.03)
    with pytest.raises(ValueError, match="positive definite"):
        infer_factor_premia(b, d*0, mu, cash_rate=.03)
    with pytest.raises(ValueError, match="label/order"):
        infer_factor_premia(b, d, mu, cash_rate=.03, alpha=alpha.iloc[::-1])


def test_inputs_are_unchanged():
    """Inputs are unchanged."""
    b, d, mu, alpha = panel()
    original = [x.copy(deep=True) for x in (b, d, mu, alpha)]
    infer_factor_premia(b, d, mu, cash_rate=.03, alpha=alpha, alpha_weight=1)
    for left, right in zip((b, d, mu, alpha), original):
        pd.testing.assert_equal(left, right) if hasattr(pd.testing, "assert_equal") else (
            pd.testing.assert_frame_equal(left, right) if isinstance(left, pd.DataFrame)
            else pd.testing.assert_series_equal(left, right))


@pytest.mark.parametrize("kind, message", [
    ("complex_loading", "real"),
    ("empty_loading", "nonempty finite"),
    ("nonfinite_loading", "nonempty finite"),
    ("complex_risk", "real"),
    ("wrong_risk_shape", "symmetric"),
    ("asymmetric_risk", "symmetric"),
    ("three_cash_endpoints", "two ordered"),
    ("reversed_cash_endpoints", "two ordered"),
])
def test_malformed_inputs_are_rejected_before_reporting(kind, message):
    """Reject invalid model geometry and shared-cash boxes instead of inferring premia."""
    b, d, mu, _ = panel()
    options = {}
    if kind == "complex_loading":
        b = b.astype(complex) + .1j
    elif kind == "empty_loading":
        b = b.iloc[:0]
        d, mu = d.iloc[:0], mu.iloc[:0]
    elif kind == "nonfinite_loading":
        b.iloc[0, 0] = np.inf
    elif kind == "complex_risk":
        d = d.astype(complex) + .1j
    elif kind == "wrong_risk_shape":
        d = np.eye(4)
    elif kind == "asymmetric_risk":
        d = np.diag(d)
        d[0, 1] = .001
    else:
        options = dict(lower_returns=mu-.01, upper_returns=mu+.01,
                       cash_bounds=(.02, .03, .04) if kind == "three_cash_endpoints"
                       else (.04, .02))
    with pytest.raises(ValueError, match=message):
        infer_factor_premia(b, d, mu, cash_rate=.03, **options)
