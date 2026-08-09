"""Minimum-tracking-error solver tests."""
import numpy as np
import pandas as pd
import pytest

from optimalportfolios import (
    Constraints,
    OptimiserConfig,
    rolling_minimise_tracking_error,
    wrapper_minimise_tracking_error,
)


@pytest.mark.parametrize('factorize_covar', [True, False])
def test_minimum_tracking_error_uses_cash_and_hard_bounds(
        factorize_covar: bool,
) -> None:
    """The closest covariance portfolio funds a fixed buy from cash."""
    assets = pd.Index(['Risk A', 'Risk B', 'Cash'])
    covariance = pd.DataFrame(
        np.diag([0.04, 0.09, 1e-6]), index=assets, columns=assets
    )
    model = pd.Series([0.40, 0.40, 0.20], index=assets)
    current = pd.Series([0.20, 0.40, 0.40], index=assets)
    constraints = Constraints(
        min_weights=pd.Series([0.30, 0.40, 0.10], index=assets),
        max_weights=pd.Series([0.30, 0.40, 0.30], index=assets),
        weights_0=current,
    )

    weights, outcome = wrapper_minimise_tracking_error(
        pd_covar=covariance,
        benchmark_weights=model,
        constraints=constraints,
        weights_0=current,
        optimiser_config=OptimiserConfig(factorize_covar=factorize_covar),
        context='minimum tracking error test',
    )

    assert outcome.accepted and outcome.compliant
    assert weights.sum() == pytest.approx(1.0)
    assert weights.to_dict() == pytest.approx(
        {'Risk A': 0.30, 'Risk B': 0.40, 'Cash': 0.30}, abs=1e-7
    )
    assert (outcome.covar_factorization is not None) is factorize_covar


def test_rolling_minimum_tracking_error_supports_time_varying_benchmarks() -> None:
    """Each covariance date uses the latest benchmark without future data."""
    assets = pd.Index(['Risk A', 'Risk B', 'Cash'])
    dates = pd.to_datetime(['2024-01-31', '2024-02-29', '2024-03-31'])
    prices = pd.DataFrame(
        [[100.0, 100.0, 100.0], [105.0, 98.0, 100.0], [103.0, 102.0, 100.0]],
        index=dates,
        columns=assets,
    )
    covariance = pd.DataFrame(
        np.diag([0.04, 0.09, 1e-6]), index=assets, columns=assets
    )
    covar_dict = {date: covariance for date in dates}
    benchmark_dates = dates[[0, 2]]
    benchmarks = pd.DataFrame(
        [[0.60, 0.20, 0.20], [0.20, 0.60, 0.20]],
        index=benchmark_dates,
        columns=assets,
    )

    weights = rolling_minimise_tracking_error(
        prices=prices,
        constraints=Constraints(is_long_only=True),
        benchmark_weights=benchmarks,
        covar_dict=covar_dict,
    )

    pd.testing.assert_index_equal(weights.index, dates)
    pd.testing.assert_index_equal(weights.columns, assets)
    np.testing.assert_allclose(weights.loc[dates[0]], benchmarks.loc[dates[0]], atol=1e-7)
    np.testing.assert_allclose(weights.loc[dates[1]], benchmarks.loc[dates[0]], atol=1e-7)
    np.testing.assert_allclose(weights.loc[dates[2]], benchmarks.loc[dates[2]], atol=1e-7)
