"""Regression tests for the E5b group-equal grouped-ranking construction."""
import numpy as np
import pandas as pd

from papers.cluster_lineage_2026.replication import run_e5b


def test_group_equal_weights_exclude_unavailable_groups() -> None:
    """Available groups receive equal budgets and invalid-score groups leave G."""
    dates = pd.DatetimeIndex(["2026-01-31"])
    columns = pd.Index(["a", "b", "c", "d", "e"])
    ranks = pd.DataFrame([[1.0, 0.49, 1.0, np.nan, np.nan]], index=dates, columns=columns)
    eligibility = pd.DataFrame(True, index=dates, columns=columns)
    groups = pd.DataFrame([["x", "x", "y", "z", "z"]], index=dates, columns=columns)

    weights, counts, validation = run_e5b._group_equal_from_ranks(
        ranks,
        eligibility,
        groups,
        q=0.5,
        universe=run_e5b.e5.UniverseName.FUTURES,
    )

    assert counts.iloc[0] == 2
    assert weights.loc[dates[0], "a"] == 0.5
    assert weights.loc[dates[0], "c"] == 0.5
    assert weights.loc[dates[0], ["b", "d", "e"]].eq(0.0).all()
    assert validation.loc[0, "weight_sum_abs_error"] <= 1e-12
    assert validation.loc[0, "max_group_budget_abs_error"] <= 1e-15


def test_group_equal_splits_selected_group_budget() -> None:
    """Multiple selected assets split their group's exact 1/G budget equally."""
    dates = pd.DatetimeIndex(["2026-01-31"])
    columns = pd.Index(["a", "b", "c"])
    ranks = pd.DataFrame([[1.0, 1.0, 1.0]], index=dates, columns=columns)
    eligibility = pd.DataFrame(True, index=dates, columns=columns)
    groups = pd.DataFrame([["x", "x", "y"]], index=dates, columns=columns)

    weights, counts, validation = run_e5b._group_equal_from_ranks(
        ranks,
        eligibility,
        groups,
        q=0.2,
        universe=run_e5b.e5.UniverseName.FUTURES,
    )

    assert counts.iloc[0] == 2
    assert weights.loc[dates[0], "a"] == 0.25
    assert weights.loc[dates[0], "b"] == 0.25
    assert weights.loc[dates[0], "c"] == 0.5
    assert validation.loc[0, "weight_status"] == "PASS"
    assert validation.loc[0, "group_budget_status"] == "PASS"
