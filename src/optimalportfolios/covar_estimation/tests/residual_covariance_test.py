"""Native alpha and common-grid empirical residual covariance integration."""

import numpy as np
import pandas as pd
import pytest
import qis
import factorlasso as fl

import optimalportfolios.covar_estimation.factor_covar_estimator as module
from optimalportfolios.covar_estimation.factor_covar_estimator import FactorCovarEstimator


requires_prepared = pytest.mark.skipif(
    not hasattr(fl, "estimate_residual_correlation"),
    reason="Empirical integration requires FactorLasso's prepared-correlation API",
)


@pytest.fixture
def panels():
    """Complete synthetic monthly/quarterly log returns with correlated residuals."""
    rng = np.random.default_rng(2092026)
    dates = pd.date_range("2014-01-31", periods=108, freq="ME")
    x = pd.DataFrame(rng.normal(0.002, 0.035, (108, 2)), index=dates, columns=["f", "g"])
    prices = np.exp(x.cumsum())
    m = pd.DataFrame(x.to_numpy() @ np.array([[0.7, 0.2], [0.1, -0.3]])
                     + rng.normal(0.001, 0.02, (108, 2)), index=dates, columns=["a", "b"])
    q = pd.DataFrame({"c": (0.4 * x.f + rng.normal(0., 0.02, 108)).resample("QE").sum()})
    return prices, {"ME": m, "QE": q}


def estimator(**kwargs):
    """Use the native model's per-frequency spans and an explicit empirical opt-in."""
    options = dict(lasso_model=fl.LassoModel(span=36, span_freq_dict={"ME": 36, "QE": 12},
                                           warmup_period=6, reg_lambda=1e-5),
                   factor_returns_freq="ME", rebalancing_freq="ME", residual_type="empirical")
    options.update(kwargs)
    return FactorCovarEstimator(**options)


@requires_prepared
def test_current_prepares_annual_covariance_and_preserves_alpha(panels):
    """Preparation changes no fitted beta, annual-alpha series or orthogonal matrix."""
    prices, returns = panels
    empirical = estimator().fit_current_factor_covars(prices, returns)
    legacy = estimator(residual_type="orthogonal").fit_current_factor_covars(prices, returns)
    pd.testing.assert_frame_equal(empirical.residuals, legacy.residuals)
    pd.testing.assert_series_equal(empirical.estimate_alpha(), legacy.estimate_alpha())
    pd.testing.assert_frame_equal(empirical.get_y_covar(), legacy.get_y_covar())
    assert empirical.residual_correlation.frequency == "QE"
    assert empirical.residual_correlation.span == pytest.approx(12)
    assert empirical.residual_metadata.frequency.to_dict() == {"a": "ME", "b": "ME", "c": "QE"}
    assert empirical.residual_metadata.residual_scale.to_dict() == {"a": 12., "b": 12., "c": 4.}
    assembled = empirical.get_y_covar(residual_type="empirical")
    v = empirical.y_variances.residual_var.to_numpy()
    r = empirical.residual_correlation.correlation.to_numpy()
    d = r * np.sqrt(v[:, None] * v[None, :])
    np.fill_diagonal(d, v)
    expected = empirical.get_y_covar(0.) + d
    np.testing.assert_allclose(assembled, expected, rtol=1e-14)
    np.testing.assert_array_equal(np.diag(assembled), np.diag(legacy.get_y_covar()))
    shared = estimator().fit_current_covar(prices, returns)
    pd.testing.assert_frame_equal(shared, assembled)


@requires_prepared
def test_rolling_holds_residual_estimate_between_quarters(panels):
    """New monthly betas must not refresh a covariance without a new complete quarter."""
    prices, returns = panels
    period = qis.TimePeriod("2021-06-30", "2022-06-30")
    result = estimator().fit_rolling_factor_covars(prices, returns, period)
    first = result[pd.Timestamp("2021-06-30")].residual_correlation
    july = result[pd.Timestamp("2021-07-31")].residual_correlation
    august = result[pd.Timestamp("2021-08-31")].residual_correlation
    assert july is first and august is first
    assert result[pd.Timestamp("2021-09-30")].residual_correlation is not first
    history = result.get_residual_correlations()
    assert all(date.month % 3 == 0 for date in history)
    for date, fitted in result.data.items():
        assert fitted.residual_correlation.estimation_date <= date
        assert fitted.residual_correlation.observation_date <= date
    covars = result.get_y_covars(residual_type="empirical",
                                 dates=pd.to_datetime(["2021-08-15", "2021-10-15"]))
    assert len(covars) == 2
    shared = estimator().fit_rolling_covars(prices, returns, period)
    for date, value in result.get_y_covars(residual_type="empirical").items():
        pd.testing.assert_frame_equal(shared[date], value)


@requires_prepared
def test_empirical_current_fit_truncates_inputs(panels):
    """The empirical opt-in cannot use later beta or residual information."""
    prices, returns = panels
    cutoff = pd.Timestamp("2021-05-31")
    full = estimator().fit_current_factor_covars(prices, returns, estimation_date=cutoff)
    truncated = estimator().fit_current_factor_covars(
        prices.loc[:cutoff], {freq: data.loc[:cutoff] for freq, data in returns.items()},
        estimation_date=cutoff,
    )
    pd.testing.assert_frame_equal(full.get_y_covar(residual_type="empirical"),
                                  truncated.get_y_covar(residual_type="empirical"))
    assert full.residual_correlation.observation_date == pd.Timestamp("2021-03-31")
    assert full.residual_correlation.estimation_date == cutoff
    with pytest.raises(ValueError, match="available"):
        full.residual_correlation.get_corr(pd.Timestamp("2021-03-31"))


@requires_prepared
def test_explicit_grid_and_span(panels):
    """A caller can prepare a common grid explicitly with a span in that grid's units."""
    prices, returns = panels
    result = estimator(residual_covar_freq="QE", residual_covar_span=8).fit_current_factor_covars(
        prices, {"ME": returns["ME"]},
    )
    assert result.residual_correlation.frequency == "QE"
    assert result.residual_correlation.span == 8
    assert result.residual_correlation.schema_version == 2
    assert result.residual_correlation.asset_metadata.annualisation_factor.eq(12).all()


def test_default_is_orthogonal_and_old_dependency_still_works(panels, monkeypatch):
    """The optional feature does not require an unreleased dependency for legacy fits."""
    assert FactorCovarEstimator().residual_type == "orthogonal"
    monkeypatch.setattr(module, "_CFCD_SUPPORTS_RESIDUAL_CORRELATION", False)
    prices, returns = panels
    result = estimator(residual_type="orthogonal").fit_current_factor_covars(prices, returns)
    assert getattr(result, "residual_correlation", None) is None
    assert getattr(result, "residual_metadata", None) is None
    with pytest.raises(ImportError, match="factorlasso"):
        estimator().fit_current_factor_covars(prices, returns)


def test_invalid_selection(panels):
    """Reject unknown residual structures instead of silently selecting a default."""
    with pytest.raises(ValueError, match="residual_type"):
        estimator(residual_type="joint")
    prices, returns = panels
    with pytest.raises(ValueError, match="residual_type"):
        module.estimate_lasso_factor_covar_data(prices, returns, fl.LassoModel(),
                                               residual_type="joint")


@requires_prepared
def test_retention_forwarding_and_refreshing_diagonal(panels):
    """Both estimator interfaces forward rho; held quarterly R uses each month's variance."""
    prices, returns = panels
    model = estimator(residual_corr_weight=.5)
    current = model.fit_current_factor_covars(prices, returns)
    pd.testing.assert_frame_equal(
        model.fit_current_covar(prices, returns),
        current.get_y_covar(residual_type="empirical", residual_corr_weight=.5),
    )
    period = qis.TimePeriod("2021-06-30", "2021-08-31")
    rolling = model.fit_rolling_factor_covars(prices, returns, period)
    covars = rolling.get_residual_covars(residual_type="empirical", residual_corr_weight=.5)
    assert len(rolling.get_residual_correlations()) == 1 and len(covars) == 3
    previous = None
    for date, d in covars.items():
        fitted = rolling[date]
        np.testing.assert_array_equal(np.diag(d), fitted.y_variances.residual_var)
        if previous is not None:
            assert not np.allclose(d, previous)
        previous = d
    shared = model.fit_rolling_covars(prices, returns, period)
    for date, d in rolling.get_y_covars(residual_type="empirical", residual_corr_weight=.5).items():
        pd.testing.assert_frame_equal(shared[date], d)


@pytest.mark.parametrize("weight", [-.1, 1.1, np.nan, np.inf])
def test_invalid_retention(weight):
    """Reject invalid correlation mixtures before fitting."""
    with pytest.raises(ValueError, match="residual_corr_weight"):
        estimator(residual_corr_weight=weight)


def test_orthogonal_retention_cannot_be_ignored():
    """A nondefault empirical-only setting cannot silently select the orthogonal model."""
    with pytest.raises(ValueError, match="residual_corr_weight"):
        estimator(residual_type="orthogonal", residual_corr_weight=.5)


@requires_prepared
def test_empirical_fit_keeps_unfitted_zero_risk_asset_independent(panels):
    """An output-only asset has no residual covariance and cannot block a valid fit."""
    prices, returns = panels
    fitted = estimator().fit_current_factor_covars(prices, returns)
    expanded = estimator().fit_current_factor_covars(
        prices, returns, assets=["a", "b", "c", "future"]
    )
    corr = expanded.residual_correlation.correlation
    assert corr.index.tolist() == ["a", "b", "c", "future"]
    pd.testing.assert_frame_equal(corr.loc[["a", "b", "c"], ["a", "b", "c"]],
                                  fitted.residual_correlation.correlation)
    assert corr.loc["future", "future"] == 1.0
    assert corr.loc["future", ["a", "b", "c"]].eq(0).all()
    covariance = expanded.get_y_covar(residual_type="empirical")
    pd.testing.assert_frame_equal(covariance.loc[["a", "b", "c"], ["a", "b", "c"]],
                                  fitted.get_y_covar(residual_type="empirical"))
    assert covariance.loc["future"].eq(0).all()


@requires_prepared
def test_empirical_fit_rejects_universe_without_positive_residual_risk(panels):
    """A wholly unfitted output universe cannot define empirical correlation."""
    prices, returns = panels
    with pytest.raises(ValueError, match="positive residual risk"):
        estimator().fit_current_factor_covars(prices, returns, assets=["future"])
