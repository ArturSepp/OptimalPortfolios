"""Regression checks for the 10 bp futures cluster long-short experiment."""
from __future__ import annotations

import importlib

import pytest


def _runner():
    """Import the dedicated runner after the fail-before-pass checkpoint."""
    return importlib.import_module(
        "papers.cluster_lineage_2026.replication."
        "run_futures_cluster_30303010_10bp"
    )


def test_cluster_30303010_design_is_frozen() -> None:
    """Freeze the cost, quantiles, strategic budgets, and cluster treatments."""
    run = _runner()
    assert run.COST_BPS == 10.0
    assert run.REFERENCE_COST_BPS == 20.0
    assert run.QUANTILES == (0.20, 0.25)
    assert run.TARGET == {
        "Equity": 0.30,
        "Fixed Income": 0.30,
        "Commodities": 0.30,
        "FX": 0.10,
    }
    assert tuple(run.METHODS) == (
        "sleeve_global",
        "sleeve_cluster_baseline",
        "sleeve_cluster_M1_star",
    )


def test_cluster_30303010_outputs_pass() -> None:
    """Require complete matched output and every construction acceptance line."""
    run = _runner()
    performance = run.pd.read_csv(
        run._root() / "performance.csv", float_precision="round_trip"
    )
    acceptance = run.pd.read_csv(
        run._root() / "acceptance.csv", float_precision="round_trip"
    )
    comparison = run.pd.read_csv(
        run._root() / "comparison.csv", float_precision="round_trip"
    )
    reconstruction = run.pd.read_csv(
        run._root() / "standalone_weight_reconstruction.csv",
        float_precision="round_trip",
    )
    determinism = run.pd.read_csv(run._root() / "determinism.csv")

    assert len(performance) == 6
    assert set(performance["method"]) == set(run.METHODS)
    assert set(performance["q"]) == {0.20, 0.25}
    assert len(comparison) == 4
    assert acceptance["status"].eq("PASS").all()
    assert reconstruction["status"].eq("PASS").all()
    assert determinism["byte_identical"].all()

    measured = performance.set_index(["q", "method"])
    frozen = {
        (0.20, "sleeve_cluster_baseline"): (-0.000911886680589169, 0.000603274096472504),
        (0.20, "sleeve_cluster_M1_star"): (-0.00102101595886606, 0.0040257775941014),
        (0.25, "sleeve_cluster_baseline"): (0.00130067052119687, 0.049196718155582),
        (0.25, "sleeve_cluster_M1_star"): (-0.00178453614277385, -0.0201548576015869),
    }
    for key, (net_return, sharpe) in frozen.items():
        assert measured.loc[key, "net_return_annualized"] == pytest.approx(
            net_return, abs=1e-14
        )
        assert measured.loc[key, "sharpe_rf0"] == pytest.approx(sharpe, abs=1e-14)


def test_global_control_matches_dedicated_run() -> None:
    """Require the in-run global control to reproduce the dedicated 10 bp book."""
    run = _runner()
    regression = run.pd.read_csv(
        run._root() / "global_control_regression.csv",
        float_precision="round_trip",
    )
    assert regression["status"].eq("PASS").all()
    assert regression["max_abs_error"].max() <= run.TOLERANCE
