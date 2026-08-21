"""Focused tests for F3 membership-consolidation helpers."""

from __future__ import annotations

import pandas as pd

from papers.cluster_lineage_2026.replication import run_f3_membership as f3


def test_panel_mapping_drops_unassigned_assets() -> None:
    """Membership mappings must exclude missing assignments date by date."""
    frame = pd.DataFrame(
        [["a", None], ["b", "c"]],
        index=pd.to_datetime(["2020-01-31", "2020-02-29"]),
        columns=["x", "y"],
    )
    mapped = f3._panel_mapping(frame)
    assert mapped[pd.Timestamp("2020-01-31")].index.tolist() == ["x"]
    assert mapped[pd.Timestamp("2020-02-29")].index.tolist() == ["x", "y"]


def test_sorted_json_is_deterministic() -> None:
    """Taxonomy mappings must have a stable key order."""
    assert f3._sorted_json({"z": 2.0, "a": 1.0}) == '{"a":1.0,"z":2.0}'


def test_nan_counter_checks_strings_too() -> None:
    """Acceptance must count missing string as well as numerical cells."""
    frame = pd.DataFrame({"number": [1.0], "label": [None]})
    assert f3._numeric_nan_count([frame]) == 1
