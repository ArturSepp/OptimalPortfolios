"""Focused regressions for point-in-time BlackRock AUM eligibility."""
from __future__ import annotations

import numpy as np
import pandas as pd

import papers.cluster_lineage_2026.replication.run_u2_blackrock_aum_filter as run
import papers.cluster_lineage_2026.replication.run_u2_blackrock_etf_grid as funds


def test_rolling_aum_requires_12_observations_without_filling() -> None:
    """Reject exact-threshold funds and incomplete 12-month histories."""
    dates = pd.date_range("2020-01-31", periods=13, freq="ME")
    aum = pd.DataFrame(
        {
            "exact_50": 50.0,
            "above_50": 51.0,
            "one_missing": 100.0,
        },
        index=dates,
    )
    aum.loc[dates[1], "one_missing"] = np.nan
    rolling = run._rolling_aum(aum)
    assert rolling.iloc[:11].isna().all().all()
    assert rolling.loc[dates[11], "exact_50"] == 50.0
    assert rolling.loc[dates[11], "above_50"] == 51.0
    assert np.isnan(rolling.loc[dates[11], "one_missing"])
    assert np.isnan(rolling.loc[dates[12], "one_missing"])


def test_eligibility_uses_strict_greater_than_threshold(monkeypatch) -> None:
    """Combine the return warmup with strict ``> USD 50m`` AUM eligibility."""
    date = pd.DatetimeIndex([pd.Timestamp("2021-01-31")], name="date")
    columns = pd.Index(["exact", "above", "missing"])
    daily = pd.DataFrame(index=date, columns=columns)
    history = pd.DataFrame(True, index=date, columns=columns)
    rolling = pd.DataFrame(
        [[50.0, np.nextafter(50.0, np.inf), np.nan]],
        index=date,
        columns=columns,
    )
    monkeypatch.setattr(funds, "_eligibility_for_dates", lambda _daily, _dates: history)
    observed = run._eligibility_for_dates(daily, date, rolling)
    expected = pd.DataFrame(
        [[False, True, False]], index=date, columns=columns
    )
    pd.testing.assert_frame_equal(observed, expected)


def test_frozen_aum_history_and_independent_rolling_regression() -> None:
    """Pin Bloomberg coverage and independently reproduce one rolling mean."""
    aum = run._read_aum()
    rolling = run._rolling_aum(aum)
    assert aum.shape == (252, 480)
    assert aum.index.min() == pd.Timestamp("2005-08-31")
    assert aum.index.max() == pd.Timestamp("2026-07-31")
    date = pd.Timestamp("2026-06-30")
    expected = float(aum.loc[:date, "AGG"].tail(12).mean())
    assert np.isclose(rolling.loc[date, "AGG"], expected, atol=1e-12)


def test_frozen_combined_eligibility_counts() -> None:
    """Pin the measured AUM-filtered breadth at headline landmarks."""
    daily = funds._read_daily()
    eligibility = run._eligibility_for_dates(daily, funds._dates())
    assert int(eligibility.loc[pd.Timestamp("2009-08-31")].sum()) == 133
    assert int(eligibility.loc[pd.Timestamp("2017-12-31")].sum()) == 242
    assert int(eligibility.loc[pd.Timestamp("2026-06-30")].sum()) == 400


def test_closed_window_prices_never_include_post_window_returns() -> None:
    """Retain one initial mark but stop performance at the declared window end."""
    prices = pd.DataFrame(
        {"A": np.arange(8.0)},
        index=pd.date_range("2017-12-20", periods=8, freq="W-WED"),
    )
    decisions = pd.DatetimeIndex(
        [pd.Timestamp("2017-12-31"), pd.Timestamp("2018-01-31")]
    )
    observed = run._closed_window_prices(
        prices, decisions, pd.Timestamp("2018-01-31")
    )
    assert observed.index.min() == pd.Timestamp("2017-12-27")
    assert observed.index.max() == pd.Timestamp("2018-01-31")
    assert (observed.index <= pd.Timestamp("2018-01-31")).all()
