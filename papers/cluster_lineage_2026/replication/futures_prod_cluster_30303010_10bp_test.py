"""Regression checks for the exact-production futures cluster experiment."""
from __future__ import annotations

import importlib

import pytest


def _runner():
    """Import the dedicated runner after the fail-before-pass checkpoint."""
    return importlib.import_module(
        "papers.cluster_lineage_2026.replication."
        "run_futures_prod_cluster_30303010_10bp"
    )


def test_production_signal_and_portfolio_design_are_frozen() -> None:
    """Freeze the exact ROSAA signal and matched portfolio parameters."""
    run = _runner()
    assert run.SIGNAL_FREQUENCY == "ME"
    assert run.MOMENTUM_LONG_SPAN == 12
    assert run.MOMENTUM_VOL_SPAN == 13
    assert run.MOMENTUM_SHORT_SPAN is None
    assert run.MOMENTUM_MEAN_ADJ_TYPE == "NONE"
    assert run.MIN_CLUSTER_SIZE == 5
    assert run.COST_BPS == 10.0
    assert run.QUANTILES == (0.20, 0.25)
    assert run.TARGET == {
        "Equity": 0.30,
        "Fixed Income": 0.30,
        "Commodities": 0.30,
        "FX": 0.10,
    }


def test_production_outputs_and_signal_preflight_pass() -> None:
    """Require complete output, exact signal provenance, and all acceptance lines."""
    run = _runner()
    performance = run.pd.read_csv(
        run._root() / "performance.csv", float_precision="round_trip"
    )
    acceptance = run.pd.read_csv(
        run._root() / "acceptance.csv", float_precision="round_trip"
    )
    signal = run.pd.read_csv(
        run._root() / "signal_diagnostics.csv", float_precision="round_trip"
    )
    reconstruction = run.pd.read_csv(
        run._root() / "standalone_weight_reconstruction.csv",
        float_precision="round_trip",
    )
    determinism = run.pd.read_csv(run._root() / "determinism.csv")

    assert len(performance) == 6
    assert set(performance["method"]) == set(run.METHODS)
    assert set(performance["q"]) == {0.20, 0.25}
    assert performance["signal_variant"].eq(run.SIGNAL_VARIANT).all()
    assert performance["signal_frequency"].eq("ME").all()
    assert performance["momentum_long_span"].eq(12).all()
    assert performance["momentum_vol_span"].eq(13).all()
    assert performance["momentum_short_span"].isna().all()
    assert performance["momentum_mean_adj_type"].eq("NONE").all()
    assert acceptance["status"].eq("PASS").all()
    assert signal["status"].eq("PASS").all()
    assert reconstruction["status"].eq("PASS").all()
    assert determinism["byte_identical"].all()


def test_production_numerical_regression() -> None:
    """Freeze the measured exact-production annual returns and Sharpes."""
    run = _runner()
    performance = run.pd.read_csv(
        run._root() / "performance.csv", float_precision="round_trip"
    ).set_index(["q", "method"])
    for key, expected in run.FROZEN_NET_RETURN_AND_SHARPE.items():
        net_return, sharpe = expected
        assert performance.loc[key, "net_return_annualized"] == pytest.approx(
            net_return, abs=1e-14
        )
        assert performance.loc[key, "sharpe_rf0"] == pytest.approx(
            sharpe, abs=1e-14
        )
