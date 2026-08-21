"""Regression tests for the U2/U3 fallback-10 signal comparison."""
import numpy as np
import pandas as pd

import papers.cluster_lineage_2026.replication.run_u2_u3_min_cluster10_signal_comparison as run


EXPECTED_NET_DELTAS = {
    ("U2_funds", "classic_12m_ex_1m"): -2.45763920877495e-05,
    ("U2_funds", "rosaa_risk_adjusted_momentum"): -0.00264157365555073,
    ("U3_futures", "classic_12m_ex_1m"): -0.00870821132674982,
    ("U3_futures", "rosaa_risk_adjusted_momentum"): 0.00318623920868477,
}


def test_outputs_are_accepted_and_deterministic() -> None:
    """Require every weight validation and replay hash to pass."""
    acceptance = pd.read_csv(run._root() / "acceptance.csv")
    replay = pd.read_csv(run._root() / "determinism.csv")

    assert acceptance["status"].eq("PASS").all()
    assert replay["byte_identical"].all()


def test_cluster_minus_global_net_deltas_are_frozen() -> None:
    """Freeze the four requested net-return comparisons."""
    comparison = pd.read_csv(
        run._root() / "comparison.csv", float_precision="round_trip"
    ).set_index(["universe", "signal_id"])
    assert set(comparison.index) == set(EXPECTED_NET_DELTAS)
    for key, expected in EXPECTED_NET_DELTAS.items():
        assert np.isclose(
            comparison.loc[key, "delta_net_return_annualized"],
            expected,
            atol=1e-12,
            rtol=0.0,
        )


def test_public_signal_identity_and_no_lookahead() -> None:
    """Require identical raw panels and non-positive sampling lags."""
    diagnostics = pd.read_csv(run._root() / "signal_diagnostics.csv")

    assert diagnostics["min_cluster_size"].eq(10).all()
    assert diagnostics["raw_panel_max_abs_error"].eq(0.0).all()
    assert diagnostics["raw_nan_mask_match"].all()
    assert diagnostics["max_global_lookahead_days"].le(0.0).all()
    assert diagnostics["max_cluster_lookahead_days"].le(0.0).all()


def test_primary_signal_selection_is_frozen() -> None:
    """Use classic momentum for U2 and volatility-adjusted momentum for U3."""
    primary = pd.read_csv(run._root() / "primary_performance.csv")
    selected = {
        universe: frame["signal_id"].unique().tolist()
        for universe, frame in primary.groupby("universe")
    }

    assert selected == {
        universe: [signal]
        for universe, signal in run.PRIMARY_SIGNAL_BY_UNIVERSE.items()
    }
