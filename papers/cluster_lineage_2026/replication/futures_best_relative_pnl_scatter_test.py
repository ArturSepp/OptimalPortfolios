"""Tests for the best-relative futures instrument P&L scatter."""
import pandas as pd
import pytest

import papers.cluster_lineage_2026.replication.run_backtests as e5
import papers.cluster_lineage_2026.replication.run_futures_best_relative_pnl_scatter as scatter
from papers.cluster_lineage_2026.replication.run_futures_best_relative_pnl_scatter import (
    FROZEN_PERFORMANCE,
    _root,
    _instrument_table,
)


EXPECTED_FUTURES_EXCLUSIONS = frozenset(
    {
        "BMR1 Curncy",
        "CUA1 Comdty",
        "IJ1 Comdty",
        "KC1 Comdty",
        "KM1 Index",
        "MES1 Index",
        "QC1 Index",
        "RS1 Comdty",
        "ST1 Index",
        "UXY1 Comdty",
        "WN1 Comdty",
    }
)
EXPECTED_FROZEN_BEST_METHOD_SPEC = {
    "analysis_window": "u1_headline_20090831_20260630",
    "cluster_method": "sleeve_cluster_M1_star",
    "global_benchmark": "sleeve_global",
    "strategy": "long_short",
    "q": 0.25,
    "signal_frequency": "ME",
    "momentum_long_span": 12,
    "momentum_short_span": None,
    "momentum_vol_span": 13,
    "momentum_mean_adj_type": "EWMA",
    "cluster_fallback": 5,
    "implementation_lag_periods": 1,
    "cost_bps_one_way": 10.0,
    "sleeve_budgets_per_side": {
        "Equity": 0.30,
        "Fixed Income": 0.30,
        "Commodities": 0.30,
        "FX": 0.10,
    },
}


def test_owner_updated_futures_exclusions_are_complete() -> None:
    """Freeze every actual source ticker removed from the futures universe."""
    assert e5.FUTURES_INVESTABILITY_EXCLUSIONS == EXPECTED_FUTURES_EXCLUSIONS
    assert e5.FUTURES_INVESTABILITY_EXCLUSION_ALIASES == {
        "MMR1 Curncy": "BMR1 Curncy"
    }


def test_owner_ruling_freezes_futures_universe_and_best_method() -> None:
    """Freeze the owner-ratified low-liquidity screen and selected method."""
    assert e5.FUTURES_ELIGIBLE_UNIVERSE_STATUS == "OWNER_FROZEN_2026-08-15"
    assert e5.FUTURES_INVESTABILITY_EXCLUSION_REASONS == {
        ticker: "low_liquidity_owner_ruling"
        for ticker in EXPECTED_FUTURES_EXCLUSIONS
    }
    assert scatter.BEST_METHOD_STATUS == "OWNER_FROZEN_2026-08-15"
    assert scatter.FROZEN_BEST_METHOD_SPEC == EXPECTED_FROZEN_BEST_METHOD_SPEC


def test_instrument_table_maps_cluster_to_x_and_global_to_y() -> None:
    """The scatter data must preserve the requested cluster-x/global-y orientation."""
    tickers = pd.Index(["A", "B"])
    cluster = pd.Series({"A": 2.0, "B": -1.0})
    global_rank = pd.Series({"A": 1.5, "B": 0.5})
    taxonomy = pd.DataFrame(
        {
            "name": ["Asset_A", "Asset_B"],
            "asset_class": ["Equities", "Bonds"],
        },
        index=tickers,
    )
    sleeves = pd.Series({"A": "Equity", "B": "Fixed Income"})

    table = _instrument_table(
        tickers=tickers,
        cluster_net_pnl=cluster,
        global_net_pnl=global_rank,
        cluster_beginning_nav=100.0,
        global_beginning_nav=100.0,
        taxonomy=taxonomy,
        sleeves=sleeves,
    ).set_index("ticker")

    assert table.loc["A", "cluster_net_pnl_pct_of_start"] == 2.0
    assert table.loc["A", "global_net_pnl_pct_of_start"] == 1.5
    assert table.loc["B", "cluster_minus_global_pnl_pct_of_start"] == -1.5
    assert table.loc["A", "name"] == "Asset A"
    assert table.loc["B", "broad_asset_class"] == "Fixed Income"


def test_updated_performance_and_replay_are_frozen() -> None:
    """Freeze both recomputed payoff rows and require deterministic artifacts."""
    performance = pd.read_csv(
        _root() / "performance.csv", float_precision="round_trip"
    ).set_index("method")
    regression = pd.read_csv(_root() / "performance_regression.csv")
    replay = pd.read_csv(_root() / "determinism.csv")

    assert regression["status"].eq("PASS").all()
    assert replay["byte_identical"].all()
    for method, expected in FROZEN_PERFORMANCE.items():
        for metric, value in expected.items():
            assert performance.loc[method, metric] == pytest.approx(value, abs=1e-12)
