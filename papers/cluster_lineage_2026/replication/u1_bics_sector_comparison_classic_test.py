"""Tests for the classic 12-minus-1 U1 BICS comparison."""
import numpy as np
import pandas as pd

import papers.cluster_lineage_2026.replication.run_u1_bics_sector_comparison_classic as run


EXPECTED_SPEC = {
    "universe": "msci_us",
    "analysis_window": "headline_20090831_20260630",
    "strategy": "long_short",
    "q": 0.25,
    "signal_variant": "classic_monthly_12m_skip1",
    "signal_frequency": "ME",
    "lookback_months_included": 12,
    "skip_months": 1,
    "volatility_adjustment": False,
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
        "net_total_return": -0.52503000305667,
        "net_return_annualized": -0.0365261198057413,
        "volatility_annualized": 0.0879717362718591,
        "sharpe_rf0": -0.377700234344542,
        "one_way_turnover_annualized": 2.63890725814431,
        "cost_drag_bp_per_year": 102.476682069217,
        "gross_return_annualized": -0.0262784515988196,
    },
    "bics_sector": {
        "net_total_return": -0.515028507967909,
        "net_return_annualized": -0.0355221429911053,
        "volatility_annualized": 0.102086313868583,
        "sharpe_rf0": -0.301866836316297,
        "one_way_turnover_annualized": 2.4734800142188,
        "cost_drag_bp_per_year": 96.1241177540517,
        "gross_return_annualized": -0.0259097312157002,
    },
    "global": {
        "net_total_return": -0.555862069334445,
        "net_return_annualized": -0.0397526261665303,
        "volatility_annualized": 0.127990450886112,
        "sharpe_rf0": -0.250753887884054,
        "one_way_turnover_annualized": 2.50434109019408,
        "cost_drag_bp_per_year": 96.963134082616,
        "gross_return_annualized": -0.0300563127582687,
    },
}


def test_classic_comparison_freezes_the_aligned_u1_specification() -> None:
    """Freeze the classic signal parameters used by the canonical U1 run."""
    assert run.FROZEN_SPEC == EXPECTED_SPEC


def test_classic_score_excludes_the_most_recent_month() -> None:
    """The last score must sum 12 months ending one month before formation."""
    dates = pd.date_range("2019-01-31", periods=14, freq="ME")
    returns = pd.DataFrame({"stock": np.arange(1.0, 15.0)}, index=dates)
    scores = run._classic_scores(returns, dates)

    assert np.isnan(scores.iloc[11, 0])
    assert scores.iloc[12, 0] == np.arange(1.0, 13.0).sum()
    assert scores.iloc[13, 0] == np.arange(2.0, 14.0).sum()


def test_classic_performance_acceptance_and_replay_are_frozen() -> None:
    """Freeze primary payoffs and require every acceptance and replay row."""
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


def test_public_signal_and_canonical_rank_contract_is_recorded() -> None:
    """Require one raw signal and one portfolio rank for every comparison leg."""
    performance = pd.read_csv(run._root() / "performance.csv")
    signal = pd.read_csv(run._root() / "signal_diagnostics.csv").iloc[0]
    membership = pd.read_csv(run._root() / "sector_membership_diagnostics.csv")
    design = pd.read_csv(run._root() / "design.csv").iloc[0]

    assert performance["construction"].eq(
        "canonical_op_global_rank_equal_weight"
    ).all()
    assert signal["classic_raw_global_cluster_max_abs_error"] == 0.0
    assert signal["classic_raw_global_sector_max_abs_error"] == 0.0
    assert bool(signal["classic_raw_global_cluster_nan_mask_match"])
    assert bool(signal["classic_raw_global_sector_nan_mask_match"])
    assert membership["assignments_outside_index"].max() == 0
    assert membership["missing_assignments_for_classified_members"].max() == 0
    assert membership["assigned_sector_members"].equals(
        membership["classified_eligible_index_members"]
    )
    assert design["cluster_role"] == "score standardisation only"
    assert design["sector_role"] == (
        "rolling-cluster score standardisation on point-in-time eligible index members"
    )
