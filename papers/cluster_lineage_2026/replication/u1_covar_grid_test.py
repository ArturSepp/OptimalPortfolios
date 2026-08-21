"""Focused contracts for the U1 covariance frequency/span sensitivity grid."""
import numpy as np
import pandas as pd

from papers.cluster_lineage_2026.replication.run_u1_covar_grid import (
    FREQUENCY_SPANS,
    _cells,
    _native_returns,
    _same_partition,
)


def test_grid_has_owner_requested_28_cells() -> None:
    """Pin cadence-specific span boundaries and total grid size."""
    assert len(_cells()) == 28
    assert FREQUENCY_SPANS["ME"] == (12, 24, 36, 52)
    for frequency in ("B", "W-MON", "W-TUE", "W-WED", "W-THU", "W-FRI"):
        assert FREQUENCY_SPANS[frequency] == (24, 36, 52, 156)


def test_native_return_aggregation_sums_log_returns() -> None:
    """Keep daily observations unchanged and sum native-period log returns."""
    dates = pd.date_range("2026-01-01", periods=40, freq="B")
    daily = pd.DataFrame({"a": np.arange(len(dates), dtype=float)}, index=dates)
    assert _native_returns(daily, "B").equals(daily)
    expected_week = daily.resample("W-MON").sum(min_count=1)
    expected_month = daily.resample("ME").sum(min_count=1)
    assert _native_returns(daily, "W-MON").equals(expected_week)
    assert _native_returns(daily, "ME").equals(expected_month)


def test_partition_comparison_is_label_invariant() -> None:
    """Recognise equivalent partitions while rejecting a changed peer relation."""
    index = pd.Index(["a", "b", "c", "d"])
    left = pd.Series([1, 1, 2, 3], index=index)
    relabelled = pd.Series([9, 9, 4, 7], index=index)
    changed = pd.Series([9, 4, 4, 7], index=index)
    assert _same_partition(left, relabelled)
    assert not _same_partition(left, changed)
