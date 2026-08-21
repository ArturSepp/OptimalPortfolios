"""Tests for the factor estimator's input guards and per-cadence span selection.

These are the paths that refuse a configuration rather than estimate from it. Each guard exists
because the alternative is not a crash but a quietly wrong fit: a frequency with no configured
span would otherwise borrow another cadence's, precomputed clusters supplied without their
linkages would produce a decomposition that cannot be rendered, and an external cluster map
missing an asset would shrink the fit universe without saying so.

The two smoother paths that build their own clustering schedule are here too. They anchor the
schedule on the estimation date, which needs an explicit append whenever that date does not fall
on the rebalancing grid — a mid-month valuation being the ordinary case.
"""

import numpy as np
import pandas as pd
import pytest
import qis

from factorlasso import ClusterSmootherType, LassoModel, LassoModelType

from optimalportfolios.covar_estimation.factor_covar_estimator import (
    FactorCovarEstimator,
    _fit_lasso_frequency,
    _model_for_frequency,
    estimate_lasso_factor_covar_data,
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
    factors = pd.DataFrame(
        100.0 * np.exp(np.cumsum(factor_returns, axis=0)), index=dates, columns=FACTORS,
    )
    assets = pd.DataFrame(factor_returns @ loadings.T + noise, index=dates, columns=ASSETS)
    return factors, {"ME": assets}


def _model(**overrides) -> LassoModel:
    """Return the small FCGL model shared by these tests."""
    defaults = dict(
        model_type=LassoModelType.FACTOR_CLUSTER_GROUP_LASSO,
        reg_lambda=1e-5, span=24, warmup_period=12, n_clusters=3,
    )
    defaults.update(overrides)
    return LassoModel(**defaults)


def _estimator(**overrides) -> FactorCovarEstimator:
    """Return the estimator wrapping ``_model`` with monthly cadences."""
    defaults = dict(
        lasso_model=_model(), rebalancing_freq="ME",
        factor_returns_freq="ME", factor_covar_span=24,
    )
    defaults.update(overrides)
    return FactorCovarEstimator(**defaults)


# --- per-cadence span selection ------------------------------------------------------------

def test_per_cadence_span_is_selected_for_the_frequency_being_fitted() -> None:
    """A configured ``span_freq_dict`` replaces the model's default span for that cadence."""
    model = _model(span_freq_dict={"ME": 18, "QE": 6})

    monthly = _model_for_frequency(model, "ME")
    quarterly = _model_for_frequency(model, "QE")

    assert monthly.span == 18
    assert quarterly.span == 6
    assert model.span == 24, "the configured model must not be mutated"


def test_cadence_without_a_configured_span_is_refused() -> None:
    """A frequency absent from ``span_freq_dict`` raises rather than borrowing another span.

    Silently falling back to the model default would fit an annual cadence on a span tuned for a
    monthly one, which still produces a covariance matrix and never announces itself.
    """
    model = _model(span_freq_dict={"ME": 18})

    with pytest.raises(KeyError, match="no span for freq=QE"):
        _model_for_frequency(model, "QE")


def test_missing_cadence_span_is_refused_inside_the_estimation_routine() -> None:
    """The same guard applies on the estimation path, not only in the helper."""
    factors, returns = _inputs()

    with pytest.raises(KeyError, match="no span for freq=ME"):
        estimate_lasso_factor_covar_data(
            risk_factor_prices=factors,
            asset_returns_dict=returns,
            lasso_model=_model(span_freq_dict={"QE": 6}),
            assets=ASSETS,
        )


def test_configured_cadence_span_is_used_by_the_estimation_routine() -> None:
    """A cadence present in ``span_freq_dict`` estimates rather than raising."""
    factors, returns = _inputs()

    covar_data = estimate_lasso_factor_covar_data(
        risk_factor_prices=factors,
        asset_returns_dict=returns,
        lasso_model=_model(span_freq_dict={"ME": 18}),
        assets=ASSETS,
    )

    assert list(covar_data.y_betas.index) == ASSETS


def test_fixed_frequency_helper_matches_direct_lasso_fit() -> None:
    """The extracted cadence fit returns the same components as direct FactorLasso fitting."""
    factors, returns = _inputs()
    asset_returns = returns["ME"]
    actual_model = _model(span_freq_dict={"ME": 18})
    reference_model = _model(span_freq_dict={"ME": 18})

    actual = _fit_lasso_frequency(
        freq="ME",
        asset_returns=asset_returns,
        risk_factor_prices=factors,
        lasso_model=actual_model,
        verbose=False,
    )

    factor_prices = factors.reindex(index=asset_returns.index, method="ffill").ffill()
    factor_returns = qis.to_returns(
        prices=factor_prices,
        is_log_returns=True,
        is_first_zero=False,
        drop_first=False,
        freq=None,
    )
    reference_model.fit(x=factor_returns, y=asset_returns, verbose=False, span=18)
    reference = reference_model.estimation_result_

    pd.testing.assert_frame_equal(actual.betas, reference_model.estimated_betas)
    pd.testing.assert_series_equal(
        actual.ewma_variances,
        pd.Series(reference.ss_total, index=asset_returns.columns),
    )
    pd.testing.assert_series_equal(
        actual.residual_variances,
        pd.Series(reference.ss_res, index=asset_returns.columns),
    )
    pd.testing.assert_series_equal(
        actual.alphas,
        pd.Series(reference.alpha, index=asset_returns.columns),
    )
    pd.testing.assert_series_equal(
        actual.r2,
        pd.Series(reference.r2, index=asset_returns.columns),
    )
    pd.testing.assert_series_equal(actual.clusters, reference_model.clusters)
    np.testing.assert_array_equal(actual.linkage, reference_model.linkage)
    assert actual.cutoff == reference_model.cutoff
    pd.testing.assert_frame_equal(
        actual.residuals,
        asset_returns - factor_returns @ reference_model.estimated_betas.T,
    )
    assert actual.derived_signs is reference_model.derived_signs_ is None
    pd.testing.assert_frame_equal(actual_model.estimated_betas, actual.betas)


# --- precomputed cluster inputs ------------------------------------------------------------

@pytest.mark.parametrize(
    "supplied",
    [
        {"precomputed_clusters"},
        {"precomputed_linkages"},
        {"precomputed_clusters", "precomputed_cutoffs"},
    ],
)
def test_partial_precomputed_cluster_inputs_are_refused(supplied: set) -> None:
    """Clusters, linkages and cutoffs must arrive together or not at all.

    A partial supply yields a decomposition whose dendrogram cannot be rendered from its own
    stored state, which surfaces much later as a reporting failure rather than a config error.
    """
    factors, returns = _inputs()
    placeholder = {
        "precomputed_clusters": {"ME": pd.Series(0, index=ASSETS)},
        "precomputed_linkages": {"ME": np.zeros((len(ASSETS) - 1, 4))},
        "precomputed_cutoffs": {"ME": 0.5},
    }
    kwargs = {name: value for name, value in placeholder.items() if name in supplied}

    with pytest.raises(ValueError, match="must all be provided together"):
        estimate_lasso_factor_covar_data(
            risk_factor_prices=factors, asset_returns_dict=returns,
            lasso_model=_model(), assets=ASSETS, **kwargs,
        )


def test_external_clusters_missing_an_asset_are_refused() -> None:
    """An external cluster map that omits an asset raises and names what is missing.

    Reindexing to the fit universe leaves NaN for the omitted assets. Proceeding would drop them
    from the fit, shrinking the investable universe without any signal that it happened.
    """
    factors, returns = _inputs()
    partial = pd.Series(0, index=ASSETS[:-2])

    with pytest.raises(ValueError, match="missing assignments") as excinfo:
        estimate_lasso_factor_covar_data(
            risk_factor_prices=factors, asset_returns_dict=returns,
            lasso_model=_model(), assets=ASSETS,
            precomputed_clusters={"ME": partial},
            precomputed_linkages={"ME": np.zeros((len(ASSETS) - 1, 4))},
            precomputed_cutoffs={"ME": 0.5},
        )

    assert ASSETS[-1] in str(excinfo.value)


# --- schedules -----------------------------------------------------------------------------

def test_estimation_date_off_the_rebalancing_grid_is_added_to_the_smoothing_schedule() -> None:
    """A valuation date that is not a rebalancing date still gets its own clustering pass.

    The causal smoother builds its schedule from the rebalancing frequency. A mid-month estimation
    date is not on that grid, so without the explicit append the smoother would never produce
    clusters for the date being valued and the lookup for it would fail.
    """
    factors, returns = _inputs()
    mid_month = factors.index[-1] - pd.Timedelta(days=9)
    truncated = {"ME": returns["ME"].loc[:mid_month]}
    estimator = _estimator(
        lasso_model=_model(cluster_smoother_type=ClusterSmootherType.PARTITION_BONUS,
                           smoother_delta=0.05),
    )

    covar_data = estimator.fit_current_factor_covars(
        factors.loc[:mid_month], truncated, assets=ASSETS, estimation_date=mid_month,
    )

    assert covar_data.estimation_date == mid_month
    assert list(covar_data.y_betas.index) == ASSETS


def test_time_period_starting_before_the_warmup_is_refused() -> None:
    """A rolling window that begins before the model can fit raises and says to move the start.

    Fitting on fewer observations than ``warmup_period`` produces betas from an underdetermined
    regression, which are numbers rather than an error, and they propagate into every covariance
    the backtest builds from that date onward.
    """
    factors, returns = _inputs()
    too_early = qis.TimePeriod(factors.index[2], factors.index[10])

    with pytest.raises(ValueError, match="increase start"):
        _estimator().fit_rolling_factor_covars(factors, returns, too_early, assets=ASSETS)


def test_invalid_recluster_frequency_is_refused() -> None:
    """An unparseable recluster frequency is reported as a config error, not a pandas one.

    ``pd.date_range`` raises a bare ``ValueError`` naming only the offset string, which gives no
    hint that the culprit is the recluster/rebalancing pairing on the model.
    """
    factors, returns = _inputs()
    period = qis.TimePeriod(factors.index[-8], factors.index[-1])
    estimator = _estimator(
        lasso_model=_model(cluster_smoother_type=ClusterSmootherType.HOLD,
                           recluster_freq="NOT-A-FREQ"),
    )

    with pytest.raises(ValueError, match="invalid recluster_freq"):
        estimator.fit_rolling_factor_covars(factors, returns, period, assets=ASSETS)
