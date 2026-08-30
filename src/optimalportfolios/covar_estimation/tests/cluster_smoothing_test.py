"""Integration tests for causal cluster smoothing in factor covariance fits."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import qis
import pytest

from factorlasso import (
    ClusterSmootherType,
    LassoModel,
    LassoModelType,
    compute_rolling_smoothed_clusters,
)
from optimalportfolios.covar_estimation.factor_covar_estimator import FactorCovarEstimator


def _inputs() -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Return a deterministic monthly factor-price and asset-return panel."""
    rng = np.random.default_rng(72026)
    dates = pd.date_range("2018-01-31", periods=72, freq="ME")
    factor_returns = rng.normal(0.004, 0.035, size=(len(dates), 3))
    loadings = np.array([
        [0.9, 0.1, 0.0], [0.8, 0.2, 0.0], [0.7, 0.1, 0.1],
        [0.0, 0.8, 0.1], [0.1, 0.9, 0.0], [0.0, 0.7, 0.2],
        [0.1, 0.0, 0.8], [0.0, 0.1, 0.9],
    ])
    noise = rng.normal(0.0, 0.012, size=(len(dates), len(loadings)))
    asset_returns = factor_returns @ loadings.T + noise
    factors = pd.DataFrame(
        100.0 * np.exp(np.cumsum(factor_returns, axis=0)),
        index=dates,
        columns=["F1", "F2", "F3"],
    )
    assets = pd.DataFrame(
        asset_returns,
        index=dates,
        columns=[f"A{i}" for i in range(len(loadings))],
    )
    return factors, {"ME": assets}


def _model(smoother: ClusterSmootherType = ClusterSmootherType.NONE, **kwargs) -> LassoModel:
    """Return the small FCGL model shared by integration tests."""
    return LassoModel(
        model_type=LassoModelType.FACTOR_CLUSTER_GROUP_LASSO,
        reg_lambda=1e-5,
        span=24,
        warmup_period=12,
        n_clusters=3,
        cluster_smoother_type=smoother,
        **kwargs,
    )


def _period(factors: pd.DataFrame) -> qis.TimePeriod:
    """Return a six-observation rolling evaluation period."""
    return qis.TimePeriod(factors.index[-8], factors.index[-1])


def test_none_rolling_path_matches_explicit_current_fits_exactly() -> None:
    """NONE retains the pre-smoothing one-pass rolling implementation bit for bit."""
    factors, returns = _inputs()
    period = _period(factors)
    estimator = FactorCovarEstimator(
        lasso_model=_model(), rebalancing_freq="ME",
        factor_returns_freq="ME", factor_covar_span=24,
    )
    actual = estimator.fit_rolling_factor_covars(factors, returns, period)
    schedule = qis.generate_dates_schedule(
        time_period=period, freq="ME", include_start_date=False, include_end_date=False,
    )
    expected = {}
    for date in schedule:
        expected[date] = estimator.fit_current_factor_covars(
            factors.loc[:date], {"ME": returns["ME"].loc[:date]}, estimation_date=date,
        )

    assert list(actual.data) == list(expected)
    for date, fitted in expected.items():
        pd.testing.assert_frame_equal(actual.data[date].y_betas, fitted.y_betas, check_exact=True)
        pd.testing.assert_series_equal(actual.data[date].clusters, fitted.clusters, check_exact=True)


@pytest.mark.parametrize(
    ("smoother", "kwargs"),
    [
        (ClusterSmootherType.PARTITION_BONUS, {"smoother_delta": 0.05}),
        (ClusterSmootherType.PARTITION_BONUS, {"smoother_delta": 0.20}),
        (ClusterSmootherType.SIMILARITY_EWMA, {"smoother_lambda": 0.50}),
    ],
)
def test_smoothed_rolling_injects_independently_computed_partitions(
        smoother: ClusterSmootherType, kwargs: dict) -> None:
    """The rolling fit consumes exactly the partitions from the causal first pass."""
    factors, returns = _inputs()
    period = _period(factors)
    model = _model(smoother, **kwargs)
    estimator = FactorCovarEstimator(
        lasso_model=model, rebalancing_freq="ME",
        factor_returns_freq="ME", factor_covar_span=24,
    )
    actual = estimator.fit_rolling_factor_covars(factors, returns, period)
    schedule = qis.generate_dates_schedule(
        time_period=period, freq="ME", include_start_date=False, include_end_date=False,
    )
    expected = compute_rolling_smoothed_clusters(returns["ME"], schedule, model)

    for date in schedule:
        labels = actual.data[date].clusters.str.removeprefix("ME:").astype(int)
        actual_partition = labels.to_numpy()
        expected_partition = expected.clusters[date].reindex(labels.index).to_numpy()
        np.testing.assert_array_equal(
            actual_partition[:, None] == actual_partition[None, :],
            expected_partition[:, None] == expected_partition[None, :],
        )


def test_precomputed_clusters_preserve_fcgl_model_type() -> None:
    """External membership must not degrade FCGL to row-grouped GROUP_LASSO."""
    factors, returns = _inputs()
    model = _model()
    date = factors.index[-1]
    rolling = compute_rolling_smoothed_clusters(returns["ME"], [date], model)
    estimator = FactorCovarEstimator(
        lasso_model=model, factor_returns_freq="ME", factor_covar_span=24,
    )
    fitted_types = []
    original_fit = LassoModel.fit

    def capture_fit(self, *args, **kwargs):
        """Record the solver model type before delegating to the real fit."""
        fitted_types.append(self.model_type)
        return original_fit(self, *args, **kwargs)

    with patch.object(LassoModel, "fit", capture_fit):
        estimator.fit_current_factor_covars(
            factors,
            returns,
            estimation_date=date,
            precomputed_clusters={"ME": rolling.clusters[date]},
            precomputed_linkages={"ME": rolling.linkages[date]},
            precomputed_cutoffs={"ME": rolling.cutoffs[date]},
        )

    assert fitted_types == [LassoModelType.FACTOR_CLUSTER_GROUP_LASSO]


@pytest.mark.parametrize(
    ("smoother", "kwargs"),
    [
        (ClusterSmootherType.SIMILARITY_EWMA, {"smoother_lambda": 0.7}),
        (ClusterSmootherType.PARTITION_BONUS, {"smoother_delta": 0.2}),
        (ClusterSmootherType.SIMILARITY_EWMA, {"smoother_lambda": 0.5}),
    ],
)
def test_smoothed_current_fit_reconstructs_the_trailing_partition(
        smoother: ClusterSmootherType, kwargs: dict) -> None:
    """A live fit rebuilds the same causal path as a matching rolling window."""
    factors, returns = _inputs()
    factors = factors.iloc[-20:]
    returns = {'ME': returns['ME'].iloc[-20:]}
    model = _model(smoother, **kwargs)
    estimator = FactorCovarEstimator(
        lasso_model=model,
        rebalancing_freq='ME',
        factor_returns_freq='ME',
        factor_covar_span=24,
    )
    start = returns['ME'].index[model.warmup_period - 1]
    end = returns['ME'].index[-1]
    rolling = estimator.fit_rolling_factor_covars(
        factors,
        returns,
        qis.TimePeriod(start, end + pd.Timedelta(days=1)),
    )
    current = estimator.fit_current_factor_covars(
        factors, returns, estimation_date=end,
    )

    pd.testing.assert_series_equal(
        current.clusters, rolling.data[end].clusters, check_exact=True,
    )


@pytest.mark.skipif(
    not hasattr(LassoModel, 'cluster_correlation_span'),
    reason='requires independent clustering-span support in FactorLasso',
)
def test_frequency_cluster_span_map_reaches_current_and_rolling_smoothers() -> None:
    """Resolve beta and clustering maps independently on both smoothing paths."""
    factors, returns = _inputs()
    model = _model(
        ClusterSmootherType.SIMILARITY_EWMA,
        smoother_lambda=0.5,
        span_freq_dict={'ME': 48},
        cluster_correlation_span_freq_dict={'ME': 24},
    )
    estimator = FactorCovarEstimator(
        lasso_model=model,
        rebalancing_freq='ME',
        factor_returns_freq='ME',
        factor_covar_span=24,
    )

    with patch(
            'optimalportfolios.covar_estimation.factor_covar_estimator.'
            'compute_rolling_smoothed_clusters',
            wraps=compute_rolling_smoothed_clusters,
    ) as smoother:
        estimator.fit_current_factor_covars(
            factors,
            returns,
            estimation_date=factors.index[-1],
        )
        estimator.fit_rolling_factor_covars(
            factors,
            returns,
            _period(factors),
        )

    assert len(smoother.call_args_list) == 2
    selected_models = [
        call.kwargs['lasso_model'] for call in smoother.call_args_list
    ]
    assert all(selected.span == 48 for selected in selected_models)
    assert all(
        selected.cluster_correlation_span == 24
        for selected in selected_models
    )


def test_hold_frequency_must_be_coarser_than_rebalancing() -> None:
    """HOLD rejects an anchor that can recluster as often as the fit schedule."""
    factors, returns = _inputs()
    estimator = FactorCovarEstimator(
        lasso_model=_model(ClusterSmootherType.HOLD, recluster_freq="ME"),
        rebalancing_freq="ME",
        factor_returns_freq="ME",
        factor_covar_span=24,
    )
    try:
        estimator.fit_rolling_factor_covars(factors, returns, _period(factors))
    except ValueError as exc:
        assert "recluster_freq" in str(exc)
        assert "ME" in str(exc)
    else:
        raise AssertionError("HOLD accepted monthly reclustering")
