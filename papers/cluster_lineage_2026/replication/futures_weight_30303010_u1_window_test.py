"""Regressions for futures 30/30/30/10 portfolios on the U1 headline window."""
from __future__ import annotations

import numpy as np
import pandas as pd

import papers.cluster_lineage_2026.replication.run_futures_weight_30303010_u1_window as run


def test_window_and_target_are_frozen() -> None:
    """Pin the common U1 dates and the owner's futures strategic budget."""
    assert run.WINDOW_START == pd.Timestamp("2009-08-31")
    assert run.WINDOW_END == pd.Timestamp("2026-06-30")
    assert run.TARGET == {
        "Equity": 0.30,
        "Fixed Income": 0.30,
        "Commodities": 0.30,
        "FX": 0.10,
    }
    assert np.isclose(sum(run.TARGET.values()), 1.0, atol=1e-15)


def test_long_only_and_long_short_are_persisted_separately() -> None:
    """Require two unambiguous strategy result tables."""
    long_only = pd.read_csv(run._root() / "performance_long_only.csv")
    long_short = pd.read_csv(run._root() / "performance_long_short.csv")
    assert len(long_only) == 8
    assert len(long_short) == 8
    assert long_only["strategy"].eq("long_only").all()
    assert long_short["strategy"].eq("long_short").all()
    assert set(long_only["method"]) == set(long_short["method"])


def test_nav_horizon_is_bounded_to_u1_window() -> None:
    """Exclude all pre-window cash history from performance statistics."""
    horizon = pd.read_csv(
        run._root() / "horizon_diagnostic.csv", parse_dates=["nav_start", "nav_end"]
    )
    assert len(horizon) == 16
    assert horizon["nav_start"].ge(run.WINDOW_START).all()
    assert horizon["nav_start"].le(run.WINDOW_START + pd.Timedelta(days=7)).all()
    assert horizon["nav_end"].le(run.WINDOW_END).all()
    assert horizon["nav_end"].ge(run.WINDOW_END - pd.Timedelta(days=7)).all()
    assert horizon["pre_window_nav_rows"].eq(0).all()
    assert horizon["post_window_nav_rows"].eq(0).all()
    assert horizon["measurement_years"].between(16.7, 16.9).all()


def test_acceptance_replay_and_global_weight_regression_are_green() -> None:
    """Require exact exposures, deterministic outputs, and unchanged global decisions."""
    acceptance = pd.read_csv(run._root() / "acceptance.csv")
    replay = pd.read_csv(run._root() / "determinism.csv")
    regression = pd.read_csv(run._root() / "global_weight_regression.csv")
    assert len(acceptance) == 16
    assert acceptance["status"].eq("PASS").all()
    assert float(acceptance.filter(regex="^max_.*error$").max().max()) <= 1e-12
    assert replay["byte_identical"].all()
    assert regression["status"].eq("PASS").all()


def test_primary_long_only_and_long_short_results_are_frozen() -> None:
    """Pin the corrected q=20% payoff rows and their separate verdicts."""
    long_only = pd.read_csv(
        run._root() / "performance_long_only.csv", float_precision="round_trip"
    )
    long_short = pd.read_csv(
        run._root() / "performance_long_short.csv", float_precision="round_trip"
    )
    long_only = long_only.loc[long_only["q"].eq(0.20)].set_index("method")
    long_short = long_short.loc[long_short["q"].eq(0.20)].set_index("method")
    assert np.isclose(
        long_only.loc["sleeve_global", "net_return_annualized"],
        0.0686650462319778,
        atol=1e-15,
    )
    assert np.isclose(
        long_only.loc["sleeve_cluster_M1_star", "net_return_annualized"],
        0.0539193962307085,
        atol=1e-15,
    )
    assert np.isclose(
        long_only.loc["sleeve_cluster_M1_star", "sharpe_rf0"],
        0.750481875256298,
        atol=1e-15,
    )
    assert np.isclose(
        long_short.loc["sleeve_global", "net_return_annualized"],
        0.00957157406602338,
        atol=1e-15,
    )
    assert np.isclose(
        long_short.loc["sleeve_cluster_M1_star", "net_return_annualized"],
        -0.0129771678829353,
        atol=1e-15,
    )
    assert long_short.loc["sleeve_cluster_M1_star", "sharpe_rf0"] < 0.0


def test_ew_is_not_a_ranking_or_performance_leg() -> None:
    """Keep EW-all confined to alpha and beta reference calculations."""
    for strategy in ("long_only", "long_short"):
        performance = pd.read_csv(run._root() / f"performance_{strategy}.csv")
        comparison = pd.read_csv(run._root() / f"comparison_{strategy}.csv")
        assert not performance["method"].str.contains("EW", case=False).any()
        assert not comparison["method"].str.contains("EW", case=False).any()
