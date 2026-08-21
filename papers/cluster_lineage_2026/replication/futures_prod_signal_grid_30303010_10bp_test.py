"""Regression checks for the futures ROSAA-production signal specification grid."""
from __future__ import annotations

import importlib

import pytest


def _runner():
    """Import the grid runner after the fail-before-pass checkpoint."""
    return importlib.import_module(
        "papers.cluster_lineage_2026.replication."
        "run_futures_prod_signal_grid_30303010_10bp"
    )


def test_signal_grid_design_is_complete() -> None:
    """Freeze every requested signal dimension and the inherited portfolio design."""
    run = _runner()
    assert run.SHORT_SPANS == (None, 1, 2, 3)
    assert run.VOL_SPANS == (13, 26, 52)
    assert run.MEAN_ADJ_TYPES == ("NONE", "EWMA")
    assert run.CLUSTER_FALLBACKS == (5, 7, 10)
    assert run.QUANTILES == (0.20, 0.25)
    assert len(run.SIGNAL_SPECS) == 24
    assert run.COST_BPS == 10.0
    assert run.TARGET == {
        "Equity": 0.30,
        "Fixed Income": 0.30,
        "Commodities": 0.30,
        "FX": 0.10,
    }


def test_grid_outputs_and_fallback_diagnostic_pass() -> None:
    """Require the complete grid, acceptance lines, base regression, and replay."""
    run = _runner()
    performance = run.pd.read_csv(
        run._root() / "performance.csv", float_precision="round_trip"
    )
    comparison = run.pd.read_csv(
        run._root() / "comparison_vs_global.csv", float_precision="round_trip"
    )
    unique = run.pd.read_csv(
        run._root() / "comparison_unique_portfolios.csv",
        float_precision="round_trip",
    )
    signal = run.pd.read_csv(
        run._root() / "signal_diagnostics.csv", float_precision="round_trip"
    )
    acceptance = run.pd.read_csv(
        run._root() / "acceptance.csv", float_precision="round_trip"
    )
    fallback = run.pd.read_csv(
        run._root() / "fallback_invariance.csv", float_precision="round_trip"
    )
    base = run.pd.read_csv(
        run._root() / "base_spec_regression.csv", float_precision="round_trip"
    )
    determinism = run.pd.read_csv(run._root() / "determinism.csv")

    assert len(performance) == 336
    assert len(comparison) == 288
    assert len(unique) == 96
    assert len(signal) == 24
    assert acceptance["status"].eq("PASS").all()
    assert signal["status"].eq("PASS").all()
    assert fallback["status"].eq("PASS").all()
    assert base["status"].eq("PASS").all()
    assert determinism["byte_identical"].all()


def test_grid_leader_is_frozen() -> None:
    """Freeze the selected best cluster-minus-global grid row."""
    run = _runner()
    summary = run.pd.read_csv(
        run._root() / "grid_leaders.csv", float_precision="round_trip"
    ).set_index("leader_type")
    expected = run.FROZEN_LEADER
    row = summary.loc["best_cluster_delta_net_return"]
    assert row["method"] == expected["method"]
    assert row["mean_adj_type"] == expected["mean_adj_type"]
    if expected["short_span_label"] == "None":
        assert run.pd.isna(row["short_span_label"])
    else:
        assert row["short_span_label"] == expected["short_span_label"]
    assert int(row["vol_span"]) == expected["vol_span"]
    assert float(row["q"]) == pytest.approx(expected["q"], abs=1e-15)
    assert row["delta_net_return_annualized"] == pytest.approx(
        expected["delta_net_return_annualized"], abs=1e-14
    )
