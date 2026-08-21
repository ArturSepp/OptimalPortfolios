"""Tests for the BlackRock ROSAA-versus-classic signal comparison."""
import numpy as np
import pandas as pd

import papers.cluster_lineage_2026.replication.run_u2_blackrock_signal_comparison as run


EXPECTED_SPEC = {
    "universe": "blackrock_us_etfs",
    "analysis_window": "headline_20090831_20260630",
    "strategy": "long_short",
    "q": 0.25,
    "cluster_frequency": "W-THU",
    "cluster_span": 156,
    "cluster_fallback": 5,
    "classification_column": "asset_class",
    "classification_budget": "equal_available_groups",
    "rosaa_signal": "ME_12_none_13_EWMA",
    "classic_signal": "classic_monthly_12m_skip1",
    "cost_bps_one_way": 10.0,
    "implementation_lag_periods": 1,
}
EXPECTED_PERFORMANCE = {
    ("ME_12_none_13_EWMA", "cluster"): {
        "net_return_annualized": -0.0118156807712069,
        "volatility_annualized": 0.0430103955276718,
        "sharpe_rf0": -0.254917559992984,
    },
    ("ME_12_none_13_EWMA", "asset_class"): {
        "net_return_annualized": 0.0104413141385287,
        "volatility_annualized": 0.0714162732741464,
        "sharpe_rf0": 0.181001521743594,
    },
    ("ME_12_none_13_EWMA", "global"): {
        "net_return_annualized": -0.00906712504869445,
        "volatility_annualized": 0.0985231351981832,
        "sharpe_rf0": -0.0430010928699198,
    },
    ("classic_monthly_12m_skip1", "cluster"): {
        "net_return_annualized": -0.0152710808470741,
        "volatility_annualized": 0.0460334909798854,
        "sharpe_rf0": -0.311123078634631,
    },
    ("classic_monthly_12m_skip1", "asset_class"): {
        "net_return_annualized": -0.00130047922042353,
        "volatility_annualized": 0.0773087096595072,
        "sharpe_rf0": 0.0216337029964075,
    },
    ("classic_monthly_12m_skip1", "global"): {
        "net_return_annualized": -0.0121977981471835,
        "volatility_annualized": 0.104670747601593,
        "sharpe_rf0": -0.0648250882308127,
    },
}


def test_blackrock_signal_comparison_spec_is_frozen() -> None:
    """Freeze the transferred comparison and selected fund-cluster cell."""
    assert run.FROZEN_SPEC == EXPECTED_SPEC


def test_classic_score_is_monthly_12_minus_1() -> None:
    """Require exactly 12 included months with the newest month excluded."""
    dates = pd.date_range("2019-01-31", periods=14, freq="ME")
    returns = pd.DataFrame({"fund": np.arange(1.0, 15.0)}, index=dates)
    scores = run._classic_scores(returns, dates)

    assert np.isnan(scores.iloc[11, 0])
    assert scores.iloc[12, 0] == np.arange(1.0, 13.0).sum()
    assert scores.iloc[13, 0] == np.arange(2.0, 14.0).sum()


def test_equal_class_weights_split_each_available_class_equally() -> None:
    """Verify equal class budgets and equal selected-fund weights per side."""
    date = pd.Timestamp("2020-01-31")
    columns = pd.Index([f"A{i}" for i in range(5)] + [f"B{i}" for i in range(5)])
    scores = pd.DataFrame(
        [np.r_[np.arange(5.0), np.arange(10.0, 15.0)]],
        index=[date],
        columns=columns,
    )
    eligibility = pd.DataFrame(True, index=[date], columns=columns)
    groups = pd.DataFrame(
        [["Class A"] * 5 + ["Class B"] * 5], index=[date], columns=columns
    )

    weights, _, _ = run._long_short_weights(scores, eligibility, groups)
    row = weights.loc[date]
    labels = groups.loc[date]

    assert np.allclose(row.clip(lower=0.0).groupby(labels).sum(), 0.5, atol=1e-12)
    assert np.allclose((-row.clip(upper=0.0)).groupby(labels).sum(), 0.5, atol=1e-12)
    assert np.allclose(row[row.gt(0.0)], 0.25, atol=1e-12)
    assert np.allclose(-row[row.lt(0.0)], 0.25, atol=1e-12)


def test_fund_performance_acceptance_and_replay_are_frozen() -> None:
    """Freeze both signals and require every acceptance and replay row to pass."""
    performance = pd.read_csv(
        run._root() / "performance.csv", float_precision="round_trip"
    ).set_index(["signal", "leg"])
    acceptance = pd.read_csv(run._root() / "acceptance.csv")
    replay = pd.read_csv(run._root() / "determinism.csv")

    assert acceptance["status"].eq("PASS").all()
    assert replay["byte_identical"].all()
    for key, expected in EXPECTED_PERFORMANCE.items():
        for metric, value in expected.items():
            assert np.isclose(performance.loc[key, metric], value, atol=1e-12, rtol=0.0)
