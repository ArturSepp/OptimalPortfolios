"""Focused regressions for the futures four-sleeve rank experiment."""
from __future__ import annotations

import numpy as np
import pandas as pd

import papers.cluster_lineage_2026.replication.run_backtests as e5
import papers.cluster_lineage_2026.replication.run_futures_sleeve_grid as run


def test_four_sleeve_mapping_is_complete_and_frozen() -> None:
    """Map every futures contract into the owner's four broad sleeves."""
    data = e5.load_universe(e5.UniverseName.FUTURES)
    sleeves = run._broad_sleeves(data.taxonomy, data.asset_returns["W-WED"].columns)
    assert sleeves.value_counts().to_dict() == {
        "Commodities": 34,
        "Equity": 29,
        "Fixed Income": 21,
        "FX": 11,
    }
    assert not sleeves.isna().any()


def test_experiment_grid_preserves_futures_primary_conventions() -> None:
    """Pin the primary q and the only predeclared cluster configurations."""
    assert run.QUANTILES == (0.20, 0.25)
    assert run.PRIMARY_Q == 0.20
    assert run.CONFIGS == (
        e5.SmootherName.BASELINE,
        e5.SmootherName.M1_STAR,
    )
    assert run.TARGET == {
        "Equity": 0.25,
        "Fixed Income": 0.25,
        "Commodities": 0.25,
        "FX": 0.25,
    }


def test_persisted_acceptance_and_replay_are_green() -> None:
    """Require exact budgets and byte-identical numerical artifacts."""
    acceptance = pd.read_csv(run._root() / "acceptance.csv")
    replay = pd.read_csv(run._root() / "determinism.csv")
    assert len(acceptance) == 16
    assert acceptance["status"].eq("PASS").all()
    assert float(acceptance.filter(regex="^max_.*error$").max().max()) <= 1e-12
    assert len(replay) == 7
    assert replay["byte_identical"].all()


def test_primary_global_regression_and_exposure_trigger_are_frozen() -> None:
    """Pin the accepted global payoff and the material concentration diagnosis."""
    regression = pd.read_csv(
        run._root() / "global_regression.csv", float_precision="round_trip"
    )
    assert len(regression) == 1
    assert regression["status"].eq("PASS").all()
    exposure = pd.read_csv(
        run._root() / "global_exposure_diagnostic.csv", float_precision="round_trip"
    ).set_index("sleeve")
    expected = {
        "Commodities": 0.48651110994501884,
        "Equity": 0.37212175121805757,
        "FX": 0.03999607939408245,
        "Fixed Income": 0.10137105944284112,
    }
    for sleeve, value in expected.items():
        assert np.isclose(exposure.loc[sleeve, "mean_weight"], value, atol=1e-15)
    assert exposure["equal_sleeve_trigger"].all()


def test_primary_payoff_verdicts_are_frozen() -> None:
    """Pin the primary q=20% fair-comparison payoff verdicts."""
    comparison = pd.read_csv(
        run._root() / "comparison.csv", float_precision="round_trip"
    )
    primary = comparison.loc[comparison["q"].eq(run.PRIMARY_Q)].set_index(
        ["strategy", "method"]
    )
    assert set(primary.index.get_level_values("strategy")) == {
        "long_only",
        "long_short",
    }
    assert set(primary.index.get_level_values("method")) == {
        "sleeve_cluster_baseline",
        "sleeve_cluster_M1_star",
    }
    long_only = primary.loc[("long_only", "sleeve_cluster_M1_star")]
    assert np.isclose(long_only["net_return_annualized"], 0.0195287960697501, atol=1e-15)
    assert np.isclose(long_only["sharpe_rf0"], 0.417333484629703, atol=1e-15)
    assert np.isclose(
        long_only["delta_vs_sleeve_global_net_return_annualized"],
        -0.00111311962870619,
        atol=1e-15,
    )
    assert np.isclose(
        long_only["delta_vs_sleeve_global_sharpe_rf0"],
        0.0417226538569331,
        atol=1e-15,
    )
    long_short = primary.loc[("long_short", "sleeve_cluster_M1_star")]
    assert np.isclose(long_short["net_return_annualized"], -0.00477111819238818, atol=1e-15)
    assert long_short["delta_vs_sleeve_global_net_return_annualized"] < 0.0
    assert long_short["delta_vs_sleeve_global_sharpe_rf0"] < 0.0


def test_ew_is_reference_only() -> None:
    """Exclude EW-all from every ranking-leg performance comparison."""
    performance = pd.read_csv(run._root() / "performance.csv")
    comparison = pd.read_csv(run._root() / "comparison.csv")
    assert not performance["method"].str.contains("EW", case=False).any()
    assert not comparison["method"].str.contains("EW", case=False).any()
