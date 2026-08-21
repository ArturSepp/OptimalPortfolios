"""Focused regressions for the owner-selected U2 BlackRock primary rule."""
from __future__ import annotations

import pandas as pd
import pytest

import papers.cluster_lineage_2026.replication.run_u2_blackrock_primary as run
from papers.cluster_lineage_2026.replication.empirical_specs import (
    U2_BLACKROCK_PRIMARY_AUM_SPEC,
)


def test_primary_aum_rule_is_frozen_at_usd_100m() -> None:
    """Freeze the owner-selected point-in-time USD 100m AUM rule."""
    spec = U2_BLACKROCK_PRIMARY_AUM_SPEC
    assert spec.data_field == "FUND_TOTAL_ASSETS"
    assert spec.currency == "USD"
    assert spec.rolling_months == 12
    assert spec.threshold_usd_millions == 100.0
    assert spec.threshold_operator == "strictly_greater_than"
    assert spec.missing_or_incomplete_history == "ineligible"
    assert run.PRIMARY_FILTER_ID == "aum_100m"


def test_primary_row_selection_is_exact() -> None:
    """Select only the USD 100m row and preserve its numerical payload."""
    frame = pd.DataFrame(
        {
            "filter_id": ["aum_50m", "aum_100m", "aum_250m"],
            "threshold_usd_millions": [50.0, 100.0, 250.0],
            "value": [1.0, 2.0, 3.0],
        }
    )
    selected = run._select_primary_rows(frame)
    assert selected.to_dict("records") == [
        {
            "filter_id": "aum_100m",
            "threshold_usd_millions": 100.0,
            "value": 2.0,
        }
    ]


def test_primary_row_selection_rejects_wrong_threshold() -> None:
    """Reject a mislabeled primary row before it reaches canonical outputs."""
    frame = pd.DataFrame(
        {
            "filter_id": ["aum_100m"],
            "threshold_usd_millions": [99.0],
        }
    )
    with pytest.raises(AssertionError, match="wrong AUM threshold"):
        run._select_primary_rows(frame)
