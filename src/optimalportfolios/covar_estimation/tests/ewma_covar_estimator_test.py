"""Tests for the EWMA estimator's configuration branches and schedule guard.

Three paths through ``EwmaCovarEstimator`` had no coverage: the vol-normalised return variant on
both the current and the rolling entry points, the guard that refuses an empty rebalancing
schedule, and the open-ended ``time_period`` whose end is left to the data. Each is selected by
configuration rather than by input shape, so nothing in the default-configured suite reached them.
"""

import numpy as np
import pandas as pd
import pytest
import qis

from optimalportfolios.covar_estimation.covar_estimator import CovarEstimator
from optimalportfolios.covar_estimation.ewma_covar_estimator import (
    EwmaCovarEstimator,
    estimate_current_ewma_covar,
)


TICKERS = ["A", "B", "C"]


def _prices(periods: int = 520) -> pd.DataFrame:
    """Return a deterministic daily price panel with unequal asset volatilities."""
    rng = np.random.default_rng(140826)
    dates = pd.date_range("2021-01-01", periods=periods, freq="B")
    scales = np.array([0.004, 0.011, 0.020])
    returns = rng.normal(0.0002, 1.0, size=(periods, len(TICKERS))) * scales
    return pd.DataFrame(
        100.0 * np.exp(np.cumsum(returns, axis=0)), index=dates, columns=TICKERS,
    )


def test_config_round_trips_through_the_shared_to_dict() -> None:
    """``CovarEstimator.to_dict`` reports every field needed to rebuild the estimator."""
    estimator = EwmaCovarEstimator(returns_freq="W-WED", span=52, rebalancing_freq="QE")

    config = estimator.to_dict()

    assert config["rebalancing_freq"] == "QE"
    assert config["span"] == 52
    assert EwmaCovarEstimator(**config) == estimator


def test_vol_normalised_returns_change_the_current_covariance() -> None:
    """The vol-normalised variant is a different estimator, not a no-op flag.

    Normalising by rolling volatility before the EWMA changes the correlation weighting across
    assets of unequal volatility, so the two variants must not agree on this fixture — if they did,
    the flag would not be reaching the estimator at all.
    """
    prices = _prices()

    plain = estimate_current_ewma_covar(prices=prices, returns_freq="W-WED", span=52)
    vol_normalised = estimate_current_ewma_covar(
        prices=prices, returns_freq="W-WED", span=52, is_apply_vol_normalised_returns=True,
    )

    assert list(plain.columns) == TICKERS
    assert list(vol_normalised.columns) == TICKERS
    assert np.all(np.isfinite(vol_normalised.to_numpy()))
    assert not np.allclose(plain.to_numpy(), vol_normalised.to_numpy())


def test_vol_normalised_returns_apply_on_the_rolling_path_too() -> None:
    """The rolling entry point honours the same flag as the current one."""
    prices = _prices()
    period = qis.TimePeriod(prices.index[260], prices.index[-1])

    covars = EwmaCovarEstimator(
        returns_freq="W-WED", span=52, rebalancing_freq="QE",
        is_apply_vol_normalised_returns=True,
    ).fit_rolling_covars(prices=prices, time_period=period)

    assert covars
    for covar in covars.values():
        assert list(covar.columns) == TICKERS
        assert np.all(np.isfinite(covar.to_numpy()))


def test_open_ended_time_period_runs_to_the_end_of_the_data() -> None:
    """A ``time_period`` with no end estimates through the last available date."""
    prices = _prices()
    open_ended = qis.TimePeriod(prices.index[260], None)

    covars = EwmaCovarEstimator(
        returns_freq="W-WED", span=52, rebalancing_freq="QE",
    ).fit_rolling_covars(prices=prices, time_period=open_ended)

    assert covars
    assert max(covars) <= prices.index[-1]
    assert max(covars) > prices.index[260]


def test_rebalancing_schedule_with_no_dates_is_refused() -> None:
    """A rebalancing frequency coarser than the sample raises instead of returning nothing.

    An empty schedule yields an empty covariance dict, which reads downstream as "no rebalancing
    was due" rather than "the configuration cannot produce one". The error names the sample period
    and the frequency so the mismatch is visible without instrumenting the caller.
    """
    prices = _prices(periods=40)
    period = qis.TimePeriod(prices.index[0], prices.index[-1])

    with pytest.raises(ValueError, match="rebalancing schedule is empty"):
        EwmaCovarEstimator(
            returns_freq="W-WED", span=12, rebalancing_freq="YE",
        ).fit_rolling_covars(prices=prices, time_period=period)


def test_ewma_estimator_is_a_covar_estimator() -> None:
    """The concrete estimator satisfies the shared interface it is dispatched through."""
    assert isinstance(EwmaCovarEstimator(), CovarEstimator)
