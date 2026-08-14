"""Tests for FactorCovarEstimator's shared-interface methods and config plumbing.

``fit_current_covar`` and ``fit_rolling_covars`` are the two methods the optimisation layer
dispatches through — every caller that treats a factor model as a plain ``CovarEstimator`` arrives
here — and neither was covered: the existing suites call the factor-specific
``fit_*_factor_covars`` variants, which return the full decomposition instead of an assembled
matrix. ``copy`` and ``to_dict`` carry the config across a persistence boundary, where ``to_dict``
has to rebuild the nested ``LassoModel`` from its own dict form.
"""

import numpy as np
import pandas as pd
import pytest
import qis

from factorlasso import LassoModel, LassoModelType

from optimalportfolios.covar_estimation.factor_covar_estimator import FactorCovarEstimator


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
    factors = pd.DataFrame(
        100.0 * np.exp(np.cumsum(factor_returns, axis=0)), index=dates, columns=FACTORS,
    )
    assets = pd.DataFrame(factor_returns @ loadings.T + noise, index=dates, columns=ASSETS)
    return factors, {"ME": assets}


def _estimator(**overrides) -> FactorCovarEstimator:
    """Return the small FCGL estimator shared by these tests."""
    defaults = dict(
        lasso_model=LassoModel(
            model_type=LassoModelType.FACTOR_CLUSTER_GROUP_LASSO,
            reg_lambda=1e-5, span=24, warmup_period=12, n_clusters=3,
        ),
        rebalancing_freq="ME",
        factor_returns_freq="ME",
        factor_covar_span=24,
    )
    defaults.update(overrides)
    return FactorCovarEstimator(**defaults)


def test_fit_current_covar_returns_an_assembled_symmetric_matrix() -> None:
    """The shared entry point assembles the decomposition into one covariance matrix.

    ``Σ_y = β Σ_x β' + w·D`` must come back symmetric, positive on the diagonal and square on the
    asset universe — the contract every optimiser downstream assumes without re-checking.
    """
    factors, returns = _inputs()

    covar = _estimator().fit_current_covar(factors, returns, assets=ASSETS)

    assert list(covar.index) == ASSETS
    assert list(covar.columns) == ASSETS
    np.testing.assert_allclose(covar.to_numpy(), covar.to_numpy().T, rtol=1e-12, atol=1e-14)
    assert (np.diag(covar.to_numpy()) > 0.0).all()


def test_residual_var_weight_scales_only_the_idiosyncratic_diagonal() -> None:
    """Dropping the residual weight to zero removes exactly the diagonal residual term.

    With ``w = 0`` the result is the pure factor part ``β Σ_x β'``, so the off-diagonal block is
    unchanged and every diagonal entry falls. A weight silently ignored would leave both equal.
    """
    factors, returns = _inputs()
    estimator = _estimator()

    full = estimator.fit_current_covar(factors, returns, assets=ASSETS)
    factor_only = estimator.fit_current_covar(
        factors, returns, assets=ASSETS, residual_var_weight=0.0,
    )

    assert (np.diag(full.to_numpy()) > np.diag(factor_only.to_numpy())).all()
    off_diagonal = ~np.eye(len(ASSETS), dtype=bool)
    np.testing.assert_allclose(
        full.to_numpy()[off_diagonal], factor_only.to_numpy()[off_diagonal],
        rtol=1e-10, atol=1e-14,
    )


def test_fit_rolling_covars_returns_one_matrix_per_rebalancing_date() -> None:
    """The rolling entry point assembles a matrix at every scheduled date."""
    factors, returns = _inputs()
    period = qis.TimePeriod(factors.index[-8], factors.index[-1])

    covars = _estimator().fit_rolling_covars(factors, returns, period, assets=ASSETS)

    assert covars
    for date, covar in covars.items():
        assert list(covar.index) == ASSETS, f"universe drifted at {date}"
        np.testing.assert_allclose(
            covar.to_numpy(), covar.to_numpy().T, rtol=1e-12, atol=1e-14,
        )


def test_copy_overrides_one_field_and_leaves_the_rest() -> None:
    """``copy`` replaces named fields and carries every other one across unchanged."""
    estimator = _estimator()

    copied = estimator.copy(rebalancing_freq="QE")

    assert copied.rebalancing_freq == "QE"
    assert estimator.rebalancing_freq == "ME", "the original must not be mutated"
    assert copied.factor_covar_span == estimator.factor_covar_span
    assert copied.lasso_model is estimator.lasso_model


def test_to_dict_rebuilds_the_nested_lasso_model() -> None:
    """The nested model comes back as a ``LassoModel``, not the dict ``asdict`` leaves behind.

    ``asdict`` recurses into nested dataclasses, so without the explicit reconstruction the round
    trip yields a dict where a model is expected and the rebuilt estimator fails on first use.
    """
    estimator = _estimator()

    config = estimator.to_dict()

    assert isinstance(config["lasso_model"], LassoModel)
    assert config["rebalancing_freq"] == "ME"
    rebuilt = FactorCovarEstimator(**config)
    assert rebuilt.lasso_model.span == estimator.lasso_model.span
    assert rebuilt.lasso_model.model_type == estimator.lasso_model.model_type


def test_to_dict_tolerates_an_absent_lasso_model() -> None:
    """A config with no model serialises without attempting to rebuild one."""
    config = _estimator(lasso_model=None).to_dict()

    assert config["lasso_model"] is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
