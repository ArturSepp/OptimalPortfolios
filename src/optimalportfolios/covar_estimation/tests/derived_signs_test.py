"""Tests for the LASSO sign-constraint layer reported as ``derived_signs``.

``estimate_lasso_factor_covar_data`` collects ``LassoModel.derived_signs_`` per frequency,
concatenates across frequencies and reindexes onto the target asset universe, then attaches the
result to ``CurrentFactorCovarData``. The whole path was previously uncovered, and its central
invariant is stated only in a comment: the reindex must **not** fill NaN, because NaN means
"no sign requirement on this cell" while 0.0 means "beta forced to zero". A ``fillna(0.0)`` there
turns an unconstrained cell into a hard constraint that was never specified, and every downstream
covariance still assembles cleanly, so nothing else would report it.

The sign layer activates when ``auto_sign_constraints=True`` and/or an explicit
``factors_beta_loading_signs`` frame is supplied; it is absent otherwise.
"""

import numpy as np
import pandas as pd
import pytest

from factorlasso import LassoModel, LassoModelType

from optimalportfolios.covar_estimation.factor_covar_estimator import (
    FactorCovarEstimator,
    _CFCD_SUPPORTS_DERIVED_SIGNS,
)


FACTORS = ["F1", "F2", "F3"]
ASSETS = [f"A{i}" for i in range(8)]


def _inputs() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Return a deterministic monthly factor-price and asset-return panel."""
    rng = np.random.default_rng(72026)
    dates = pd.date_range("2018-01-31", periods=72, freq="ME")
    factor_returns = rng.normal(0.004, 0.035, size=(len(dates), len(FACTORS)))
    loadings = np.array([
        [0.9, 0.1, 0.0], [0.8, 0.2, 0.0], [0.7, 0.1, 0.1],
        [0.0, 0.8, 0.1], [0.1, 0.9, 0.0], [0.0, 0.7, 0.2],
        [0.1, 0.0, 0.8], [0.0, 0.1, 0.9],
    ])
    noise = rng.normal(0.0, 0.012, size=(len(dates), len(loadings)))
    asset_returns = factor_returns @ loadings.T + noise
    factors = pd.DataFrame(
        100.0 * np.exp(np.cumsum(factor_returns, axis=0)), index=dates, columns=FACTORS,
    )
    assets = pd.DataFrame(asset_returns, index=dates, columns=ASSETS)
    return factors, {"ME": assets}


def _estimator(**model_kwargs) -> FactorCovarEstimator:
    """Return the small FCGL estimator shared by these tests."""
    lasso_model = LassoModel(
        model_type=LassoModelType.FACTOR_CLUSTER_GROUP_LASSO,
        reg_lambda=1e-5,
        span=24,
        warmup_period=12,
        n_clusters=3,
        **model_kwargs,
    )
    return FactorCovarEstimator(
        lasso_model=lasso_model,
        rebalancing_freq="ME",
        factor_returns_freq="ME",
        factor_covar_span=24,
    )


def test_no_sign_layer_reports_no_signs() -> None:
    """With the sign layer off the field stays None rather than an all-zero frame.

    Distinguishes "no sign layer ran" from "every cell was constrained to zero", which an
    empty-frame default would conflate.
    """
    factors, returns = _inputs()
    covar_data = _estimator().fit_current_factor_covars(factors, returns)

    assert covar_data.derived_signs is None


def test_auto_sign_constraints_report_a_signs_frame_shaped_like_the_betas() -> None:
    """The reported frame carries one cell per (asset, factor), aligned to ``y_betas``."""
    factors, returns = _inputs()
    covar_data = _estimator(auto_sign_constraints=True).fit_current_factor_covars(
        factors, returns,
    )
    signs = covar_data.derived_signs

    assert signs is not None
    assert list(signs.index) == list(covar_data.y_betas.index)
    assert list(signs.columns) == list(covar_data.y_betas.columns)
    assert set(pd.unique(signs.to_numpy().ravel())) <= {-1.0, 0.0, 1.0}


def test_reported_signs_match_the_betas_that_were_actually_fitted() -> None:
    """Every constrained cell agrees with the fitted beta it constrained.

    This is what makes the frame worth reporting: it is the sign matrix the solver applied, not a
    restatement of the request. A frame collected from the wrong frequency, or stale from a prior
    fit, would still be correctly shaped and would still pass the alignment test above.
    """
    factors, returns = _inputs()
    covar_data = _estimator(auto_sign_constraints=True).fit_current_factor_covars(
        factors, returns,
    )
    signs, betas = covar_data.derived_signs, covar_data.y_betas

    constrained = signs.stack().loc[lambda s: s != 0.0]
    assert not constrained.empty, "fixture produced no constrained cells; the test proves nothing"

    for (asset, factor), sign in constrained.items():
        beta = betas.loc[asset, factor]
        assert np.sign(beta) in (0.0, sign), f"beta {beta} at ({asset}, {factor}) violates {sign}"

    for (asset, factor), sign in signs.stack().loc[lambda s: s == 0.0].items():
        assert betas.loc[asset, factor] == pytest.approx(0.0, abs=1e-8)


def test_asset_without_a_fit_keeps_nan_rather_than_a_forced_zero() -> None:
    """An asset outside the fitted universe stays NaN through the reindex.

    NaN means "unconstrained"; 0.0 means "beta forced to zero". Filling the reindex would impose a
    hard constraint on an asset the model never saw, and the resulting covariance is still
    well-formed, so only this assertion separates the two.
    """
    factors, returns = _inputs()
    covar_data = _estimator(auto_sign_constraints=True).fit_current_factor_covars(
        factors, returns, assets=ASSETS + ["GHOST"],
    )
    signs = covar_data.derived_signs

    assert list(signs.index) == ASSETS + ["GHOST"]
    assert signs.loc["GHOST"].isna().all(), "unfitted asset was filled instead of left NaN"
    assert not signs.loc[ASSETS].isna().any().any(), "fitted assets should carry a sign everywhere"
    assert (signs.loc[ASSETS] == 0.0).any().any(), (
        "fixture produced no genuine zero, so this cannot distinguish NaN from 0.0"
    )


def test_explicit_loading_signs_are_applied_and_reported() -> None:
    """A supplied sign frame activates the layer without ``auto_sign_constraints``."""
    factors, returns = _inputs()
    requested = pd.DataFrame(1.0, index=ASSETS, columns=FACTORS)
    covar_data = _estimator(factors_beta_loading_signs=requested).fit_current_factor_covars(
        factors, returns,
    )

    assert covar_data.derived_signs is not None
    assert covar_data.y_betas.min().min() >= -1e-8, "long-only signs did not bind the fit"


def test_estimator_reports_signs_only_when_factorlasso_supports_the_field() -> None:
    """The compatibility shim matches what the installed factorlasso actually accepts.

    ``_CFCD_SUPPORTS_DERIVED_SIGNS`` gates the kwarg at construction. If the shim and the
    installed dataclass disagree, either the field is silently dropped or construction raises.
    """
    factors, returns = _inputs()
    covar_data = _estimator(auto_sign_constraints=True).fit_current_factor_covars(
        factors, returns,
    )

    if _CFCD_SUPPORTS_DERIVED_SIGNS:
        assert covar_data.derived_signs is not None
    else:
        assert not hasattr(covar_data, "derived_signs")
