"""Focused regressions for the BlackRock broad-sleeve allocation grid."""
from __future__ import annotations

import numpy as np
import pandas as pd

import papers.cluster_lineage_2026.replication.run_u2_blackrock_etf_grid as u2
import papers.cluster_lineage_2026.replication.run_u2_blackrock_sleeve_grid as run


def test_weight_grid_is_frozen_and_contains_owner_proposal() -> None:
    """Pin the eight feasible allocations and the proposed 50/30/20 cell."""
    grid = run._weight_grid()
    assert len(grid) == 8
    assert grid["weight_id"].tolist() == [
        "E40_F30_R30",
        "E40_F40_R20",
        "E50_F20_R30",
        "E50_F30_R20",
        "E50_F40_R10",
        "E60_F20_R20",
        "E60_F30_R10",
        "E70_F20_R10",
    ]
    assert np.allclose(
        grid[["equity_weight", "fixed_income_weight", "rest_weight"]].sum(axis=1),
        1.0,
    )
    assert grid.loc[grid["is_owner_50_30_20"], "weight_id"].item() == "E50_F30_R20"


def test_official_asset_classes_map_completely_to_three_sleeves() -> None:
    """Require all 480 current funds to enter Equity, Fixed Income, or Rest."""
    metadata = pd.read_csv(u2.METADATA_FILE)
    broad = run._broad_sleeves(pd.Index(metadata["ticker"]))
    assert broad.value_counts().to_dict() == {
        "Equity": 288,
        "Fixed Income": 154,
        "Rest": 38,
    }
    assert not broad.isna().any()


def test_persisted_budget_and_determinism_acceptance_is_green() -> None:
    """Require exact portfolio budgets and byte-identical numerical artifacts."""
    acceptance = pd.read_csv(run._root() / "acceptance.csv")
    replay = pd.read_csv(run._root() / "determinism.csv")
    assert len(acceptance) == 150
    assert acceptance["status"].eq("PASS").all()
    assert float(acceptance.filter(regex="^max_.*error$").max().max()) <= 1e-12
    assert len(replay) == 9
    assert replay["byte_identical"].all()


def test_training_only_selections_and_oos_results_are_frozen() -> None:
    """Pin the selected allocations and their untouched evaluation payoffs."""
    selection = pd.read_csv(run._root() / "selection.csv")
    chosen = selection.set_index(["strategy", "selection_type"])
    assert chosen.loc[("long_only", "training_best_absolute"), "weight_id"] == "E70_F20_R10"
    assert chosen.loc[("long_short", "training_best_absolute"), "weight_id"] == "E50_F40_R10"
    assert chosen.loc[("long_only", "training_common_balanced"), "weight_id"] == "E40_F40_R20"
    assert chosen.loc[("long_short", "training_common_balanced"), "weight_id"] == "E40_F40_R20"

    comparison = pd.read_csv(
        run._root() / "comparison.csv", float_precision="round_trip"
    )
    selected = comparison.loc[
        comparison["analysis_window"].eq(run.TEST_WINDOW)
        & comparison["method"].eq("sleeve_cluster_primary")
    ].set_index(["strategy", "weight_id"])
    long_only = selected.loc[("long_only", "E70_F20_R10")]
    assert np.isclose(long_only["net_return_annualized"], 0.0585578092651828, atol=1e-15)
    assert np.isclose(long_only["sharpe_rf0"], 0.480220493415215, atol=1e-15)
    assert np.isclose(
        long_only["delta_vs_sleeve_global_net_return_annualized"],
        0.00304602008407184,
        atol=1e-15,
    )
    assert np.isclose(
        long_only["delta_vs_original_global_net_return_annualized"],
        0.00843290728322943,
        atol=1e-15,
    )
    long_short = selected.loc[("long_short", "E50_F40_R10")]
    assert np.isclose(long_short["net_return_annualized"], -0.0172431426313165, atol=1e-15)
    assert np.isclose(
        long_short["delta_vs_original_global_net_return_annualized"],
        0.00096230708669065,
        atol=1e-15,
    )


def test_me12_edge_does_not_transfer_to_me36() -> None:
    """Record that the long-only result is covariance-specification sensitive."""
    comparison = pd.read_csv(
        run._root() / "comparison.csv", float_precision="round_trip"
    )
    transfer = comparison.loc[
        comparison["analysis_window"].eq(run.TEST_WINDOW)
        & comparison["strategy"].eq("long_only")
        & comparison["method"].eq("sleeve_cluster_transfer_ME36")
        & comparison["weight_id"].eq("E70_F20_R10")
    ].iloc[0]
    assert transfer["delta_vs_sleeve_global_net_return_annualized"] < 0.0
    assert transfer["delta_vs_original_global_net_return_annualized"] < 0.0


def test_ew_all_is_reference_only_and_original_global_is_unchanged() -> None:
    """Exclude EW from ranking legs and retain the accepted global regression."""
    performance = pd.read_csv(run._root() / "performance.csv")
    regression = pd.read_csv(run._root() / "regression.csv")
    assert not performance["method"].str.contains("EW", case=False).any()
    assert performance.loc[
        performance["method"].eq("original_global"), "weight_id"
    ].eq("UNCONSTRAINED").all()
    assert len(regression) == 2
    assert regression["status"].eq("PASS").all()
