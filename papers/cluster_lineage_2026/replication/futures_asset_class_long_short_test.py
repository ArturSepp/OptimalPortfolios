"""Regressions for standalone futures asset-class long-short portfolios."""
from __future__ import annotations

import numpy as np
import pandas as pd

import papers.cluster_lineage_2026.replication.run_futures_asset_class_long_short as run


def test_four_asset_classes_and_grid_are_frozen() -> None:
    """Pin the four standalone books and two selection fractions."""
    assert run.ASSET_CLASSES == (
        "Equity",
        "Fixed Income",
        "Commodities",
        "FX",
    )
    assert run.QUANTILES == (0.20, 0.25)
    assert run.PRIMARY_Q == 0.20


def test_performance_is_separate_by_asset_class() -> None:
    """Require one global and two cluster portfolios per class and q."""
    performance = pd.read_csv(run._root() / "performance.csv")
    comparison = pd.read_csv(run._root() / "comparison.csv")
    assert len(performance) == 24
    assert len(comparison) == 16
    assert performance["strategy"].eq("long_short").all()
    assert set(performance["asset_class"]) == set(run.ASSET_CLASSES)
    assert performance.groupby(["asset_class", "q"]).size().eq(3).all()
    assert comparison.groupby(["asset_class", "q"]).size().eq(2).all()


def test_every_book_is_exactly_plus_one_minus_one() -> None:
    """Require zero net and gross-two exposure without cross-class leakage."""
    acceptance = pd.read_csv(
        run._root() / "acceptance.csv", float_precision="round_trip"
    )
    assert len(acceptance) == 24
    assert acceptance["status"].eq("PASS").all()
    assert float(acceptance["max_net_exposure_abs_error"].max()) <= 1e-12
    assert float(acceptance["max_gross_exposure_abs_error"].max()) <= 1e-12
    assert float(acceptance["max_asset_class_leakage"].max()) <= 1e-12


def test_window_replay_and_combined_weight_reconstruction_are_green() -> None:
    """Pin the corrected horizon, replay hashes, and sleeve aggregation identity."""
    horizon = pd.read_csv(
        run._root() / "horizon_diagnostic.csv", parse_dates=["nav_start", "nav_end"]
    )
    replay = pd.read_csv(run._root() / "determinism.csv")
    reconstruction = pd.read_csv(
        run._root() / "combined_weight_reconstruction.csv",
        float_precision="round_trip",
    )
    assert horizon["nav_start"].ge(run.WINDOW_START).all()
    assert horizon["nav_end"].le(run.WINDOW_END).all()
    assert horizon["pre_window_nav_rows"].eq(0).all()
    assert horizon["post_window_nav_rows"].eq(0).all()
    assert replay["byte_identical"].all()
    assert reconstruction["status"].eq("PASS").all()
    assert np.isclose(
        reconstruction["max_weight_abs_error"], 0.0, atol=1e-15
    ).all()


def test_primary_asset_class_payoffs_and_verdict_are_frozen() -> None:
    """Pin the q=20% class-level drivers and honest no-cluster-win verdict."""
    performance = pd.read_csv(
        run._root() / "performance.csv", float_precision="round_trip"
    )
    primary = performance.loc[performance["q"].eq(run.PRIMARY_Q)].set_index(
        ["asset_class", "method"]
    )
    expected_net = {
        ("Equity", "global"): -0.00900445734617239,
        ("Equity", "cluster_M1_star"): -0.015744139038976,
        ("Fixed Income", "global"): -0.0127199452821727,
        ("Fixed Income", "cluster_M1_star"): -0.0119364377447536,
        ("Commodities", "global"): 0.00593773469215586,
        ("Commodities", "cluster_M1_star"): -0.0167102592976867,
        ("FX", "global"): 0.000251826972131441,
        ("FX", "cluster_M1_star"): -0.0192132165123547,
    }
    for key, expected in expected_net.items():
        assert np.isclose(
            primary.loc[key, "net_return_annualized"], expected, atol=1e-15
        )
    comparison = pd.read_csv(run._root() / "comparison.csv")
    assert not comparison["beats_global_both"].astype(bool).any()
    fixed_income_m1 = comparison.loc[
        comparison["q"].eq(run.PRIMARY_Q)
        & comparison["asset_class"].eq("Fixed Income")
        & comparison["method"].eq("cluster_M1_star")
    ].iloc[0]
    assert fixed_income_m1["delta_vs_global_net_return_annualized"] > 0.0
    assert fixed_income_m1["delta_vs_global_sharpe_rf0"] < 0.0


def test_ew_is_not_a_payoff_leg_or_comparison() -> None:
    """Keep EW-all confined to alpha and beta reference calculations."""
    performance = pd.read_csv(run._root() / "performance.csv")
    comparison = pd.read_csv(run._root() / "comparison.csv")
    assert not performance["method"].str.contains("EW", case=False).any()
    assert not comparison["method"].str.contains("EW", case=False).any()
