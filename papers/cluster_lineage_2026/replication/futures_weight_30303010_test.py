"""Focused regressions for the futures 30/30/30/10 sleeve experiment."""
from __future__ import annotations

import numpy as np
import pandas as pd

import papers.cluster_lineage_2026.replication.run_futures_sleeve_grid as equal
import papers.cluster_lineage_2026.replication.run_futures_weight_30303010 as run


def test_target_budget_is_frozen_and_complete() -> None:
    """Pin the owner's strategic futures sleeve weights."""
    assert run.TARGET == {
        "Equity": 0.30,
        "Fixed Income": 0.30,
        "Commodities": 0.30,
        "FX": 0.10,
    }
    assert tuple(run.TARGET) == equal.SLEEVES
    assert np.isclose(sum(run.TARGET.values()), 1.0, atol=1e-15)


def test_persisted_acceptance_and_replay_are_green() -> None:
    """Require exact strategic budgets and byte-identical numerical outputs."""
    acceptance = pd.read_csv(run._root() / "acceptance.csv")
    replay = pd.read_csv(run._root() / "determinism.csv")
    assert len(acceptance) == 16
    assert acceptance["status"].eq("PASS").all()
    assert float(acceptance.filter(regex="^max_.*error$").max().max()) <= 1e-12
    assert replay["byte_identical"].all()


def test_allocation_diagnostics_match_each_target() -> None:
    """Require every constrained side to carry its stated sleeve budget."""
    allocation = pd.read_csv(
        run._root() / "allocation_diagnostics.csv", float_precision="round_trip"
    )
    constrained = allocation.loc[~allocation["method"].eq("original_global")]
    for sleeve, target in run.TARGET.items():
        rows = constrained.loc[constrained["sleeve"].eq(sleeve)]
        assert np.isclose(rows["target_budget"], target, atol=1e-15).all()
        assert np.isclose(rows["mean_long_exposure"], target, atol=1e-12).all()
        long_short = rows.loc[rows["strategy"].eq("long_short")]
        assert np.isclose(
            long_short["mean_short_exposure_abs"], target, atol=1e-12
        ).all()


def test_comparison_uses_same_budget_global_and_equal_sleeve_benchmark() -> None:
    """Pin the fair and equal-sleeve comparison channels without an EW yardstick."""
    comparison = pd.read_csv(run._root() / "comparison.csv")
    equal_comparison = pd.read_csv(run._root() / "comparison_vs_equal_sleeves.csv")
    assert len(comparison) == 8
    assert len(equal_comparison) == 16
    assert set(equal_comparison["equal_sleeve_method"]) == {
        "original_global",
        "sleeve_global",
        "sleeve_cluster_baseline",
        "sleeve_cluster_M1_star",
    }
    assert not comparison.columns.str.contains("delta_vs_EW", case=False).any()
    assert not equal_comparison.columns.str.contains("delta_vs_EW", case=False).any()


def test_accepted_global_is_unchanged() -> None:
    """Keep the original q=20% global portfolio byte-for-number equivalent."""
    regression = pd.read_csv(
        run._root() / "global_regression.csv", float_precision="round_trip"
    )
    assert len(regression) == 1
    assert regression["status"].eq("PASS").all()
    assert float(regression["measured_max_abs_error"].max()) <= 1e-12
