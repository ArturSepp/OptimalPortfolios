"""Focused regressions for the fixed-model BlackRock AUM sensitivity."""
from __future__ import annotations

import pandas as pd

import papers.cluster_lineage_2026.replication.run_u2_blackrock_aum_filter as aum50
import papers.cluster_lineage_2026.replication.run_u2_blackrock_aum_sensitivity as run
import papers.cluster_lineage_2026.replication.run_u2_blackrock_etf_grid as funds
import papers.cluster_lineage_2026.replication.run_u2_blackrock_sleeve_grid as sleeves


def test_sensitivity_changes_only_the_declared_aum_rule() -> None:
    """Pin the selected model and reasonable monotone cutoff range."""
    assert run.FILTERS == (
        ("history_only", None),
        ("aum_25m", 25.0),
        ("aum_50m", 50.0),
        ("aum_100m", 100.0),
        ("aum_250m", 250.0),
        ("aum_500m", 500.0),
    )
    assert (run.FREQUENCY, run.SPAN, run.Q) == ("W-THU", 156, 0.25)
    assert run.WEIGHT_ID == "E50_F30_R20"
    assert run.CONSTRUCTION == "group_equal"
    assert run.HYBRID_VARIANT == "global_long_cluster_short"
    assert run.SCHEDULE == "every_two_months"
    assert run.COST_BPS == 20.0


def test_cutoff_eligibilities_are_nested_and_reproduce_aum50() -> None:
    """Require larger cutoffs to remove, never add, point-in-time members."""
    daily = funds._read_daily()
    dates = funds._dates()
    rolling = aum50._rolling_aum()
    observed = run._eligibilities(daily, dates, rolling)
    ordered = [filter_id for filter_id, _ in run.FILTERS]
    for lower, higher in zip(ordered, ordered[1:]):
        assert not (observed[higher] & ~observed[lower]).any().any()
    expected_50m = aum50._eligibility_for_dates(daily, dates, rolling)
    pd.testing.assert_frame_equal(observed["aum_50m"], expected_50m)


def test_strictest_cutoff_retains_every_sleeve_at_landmarks() -> None:
    """Ensure USD 500m sensitivity remains a viable three-sleeve portfolio."""
    daily = funds._read_daily()
    dates = funds._dates()
    eligibility = run._eligibilities(
        daily, dates, aum50._rolling_aum()
    )["aum_500m"]
    sleeve_map = sleeves._broad_sleeves(eligibility.columns)
    expected = {
        pd.Timestamp("2009-08-31"): (44, 13, 5),
        pd.Timestamp("2017-12-31"): (113, 29, 10),
        pd.Timestamp("2026-06-30"): (182, 76, 17),
    }
    for date, counts in expected.items():
        observed = tuple(
            int((eligibility.loc[date] & sleeve_map.eq(sleeve)).sum())
            for sleeve in run.search.SLEEVES
        )
        assert observed == counts
