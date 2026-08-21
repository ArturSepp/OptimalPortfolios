"""Tests for the frozen U1 BICS-sector versus cluster/global comparison."""
import numpy as np
import pandas as pd

import papers.cluster_lineage_2026.replication.run_u1_bics_sector_comparison as run


EXPECTED_SPEC = {
    "universe": "msci_us",
    "analysis_window": "headline_20090831_20260630",
    "strategy": "long_short",
    "q": 0.25,
    "signal_frequency": "ME",
    "momentum_long_span": 12,
    "momentum_short_span": None,
    "momentum_vol_span": 13,
    "momentum_mean_adj_type": "EWMA",
    "cluster_config": "M1_star",
    "cluster_delta": 0.0866,
    "cluster_fallback": 5,
    "sector_column": "bbg_bics_sector",
    "missing_sector_policy": "exclude_from_all_primary_legs",
    "cost_bps_one_way": 10.0,
    "implementation_lag_periods": 1,
}
EXPECTED_PRIMARY_PERFORMANCE = {
    "cluster_M1_star": {
        "net_total_return": -0.365909815899295,
        "net_return_annualized": -0.0225115995727729,
        "volatility_annualized": 0.0635134852393954,
        "sharpe_rf0": -0.326334191336638,
        "one_way_turnover_annualized": 3.0112777898482,
        "cost_drag_bp_per_year": 118.246259675715,
        "gross_return_annualized": -0.0106869736052014,
    },
    "bics_sector": {
        "net_total_return": -0.518119455132675,
        "net_return_annualized": -0.0358303039361876,
        "volatility_annualized": 0.0959805977555937,
        "sharpe_rf0": -0.331165953711517,
        "one_way_turnover_annualized": 3.0815058460601,
        "cost_drag_bp_per_year": 119.634821592612,
        "gross_return_annualized": -0.0238668217769263,
    },
    "global": {
        "net_total_return": -0.563343108539351,
        "net_return_annualized": -0.0405675519922472,
        "volatility_annualized": 0.128129328126028,
        "sharpe_rf0": -0.257192372087837,
        "one_way_turnover_annualized": 3.16090028523388,
        "cost_drag_bp_per_year": 122.189358036834,
        "gross_return_annualized": -0.0283486161885638,
    },
}


def test_owner_selected_u1_comparison_spec_is_frozen() -> None:
    """Require the U1 comparison to inherit the selected futures method."""
    assert run.FROZEN_SPEC == EXPECTED_SPEC


def test_sector_long_short_weights_are_equal_by_sector_and_stock() -> None:
    """Each side must split equally across sectors and selected stocks."""
    date = pd.Timestamp("2020-01-31")
    columns = pd.Index([f"A{i}" for i in range(5)] + [f"B{i}" for i in range(5)])
    scores = pd.DataFrame(
        [np.r_[np.arange(5.0), np.arange(10.0, 15.0)]],
        index=[date],
        columns=columns,
    )
    eligibility = pd.DataFrame(True, index=[date], columns=columns)
    groups = pd.DataFrame(
        [["Sector A"] * 5 + ["Sector B"] * 5], index=[date], columns=columns
    )

    weights, _, _ = run._long_short_weights(scores, eligibility, groups)
    row = weights.loc[date]
    labels = groups.loc[date]
    long_budgets = row.clip(lower=0.0).groupby(labels).sum()
    short_budgets = (-row.clip(upper=0.0)).groupby(labels).sum()

    assert np.allclose(long_budgets.to_numpy(), 0.5, atol=1e-12)
    assert np.allclose(short_budgets.to_numpy(), 0.5, atol=1e-12)
    assert np.allclose(row[row.gt(0.0)].to_numpy(), 0.25, atol=1e-12)
    assert np.allclose((-row[row.lt(0.0)]).to_numpy(), 0.25, atol=1e-12)
    assert abs(row.sum()) <= 1e-12
    assert abs(row.abs().sum() - 2.0) <= 1e-12


def test_primary_performance_acceptance_and_replay_are_frozen() -> None:
    """Freeze payoff rows and require all numerical and replay checks to pass."""
    performance = pd.read_csv(
        run._root() / "performance.csv", float_precision="round_trip"
    ).set_index("leg")
    acceptance = pd.read_csv(run._root() / "acceptance.csv")
    replay = pd.read_csv(run._root() / "determinism.csv")

    assert acceptance["status"].eq("PASS").all()
    assert replay["byte_identical"].all()
    for leg, expected in EXPECTED_PRIMARY_PERFORMANCE.items():
        for metric, value in expected.items():
            assert np.isclose(
                performance.loc[leg, metric], value, atol=1e-12, rtol=0.0
            )
