"""Focused checks for the BlackRock funds long-short specification search."""

import numpy as np
import pandas as pd

import papers.cluster_lineage_2026.replication.run_u2_blackrock_long_short_search as run


def test_declared_search_space_and_base_are_frozen() -> None:
    """Pin the predeclared marginal grids and the fair 50/30/20 base case."""
    assert len(run.SIGNAL_SPECS) == 25
    assert len(run.COVARIANCE_CELLS) == 28
    assert run.QUANTILES == (0.10, 0.15, 0.20, 0.25, 0.30)
    assert len(run.WEIGHT_GRID) == 8
    assert run.CONSTRUCTIONS == ("group_equal", "sqrt_group_size", "asset_equal")
    assert run.BASE_CANDIDATE.signal_id == "rosaa_short_none_vol_13_mean_EWMA"
    assert (run.BASE_CANDIDATE.frequency, run.BASE_CANDIDATE.span) == ("W-THU", 156)
    assert run.BASE_CANDIDATE.q == 0.25
    assert run.BASE_CANDIDATE.weight_id == "E50_F30_R20"
    assert run.COST_BPS == 20.0


def test_long_short_constructions_preserve_exact_sleeve_budgets() -> None:
    """Require exact +1/-1 exposure and fixed sleeve budgets for every construction."""
    dates = pd.date_range("2025-01-31", periods=2, freq="ME")
    columns = pd.Index([f"a{i}" for i in range(18)])
    scores = pd.DataFrame(
        np.tile(np.arange(18, dtype=float), (2, 1)), index=dates, columns=columns
    )
    eligibility = pd.DataFrame(True, index=dates, columns=columns)
    sleeves = pd.Series(
        ["Equity"] * 6 + ["Fixed Income"] * 6 + ["Rest"] * 6,
        index=columns,
    )
    sleeve_panel = pd.DataFrame(
        np.tile(sleeves.to_numpy(), (2, 1)), index=dates, columns=columns
    )
    labels = pd.Series(
        ["e1"] * 3
        + ["e2"] * 3
        + ["f1"] * 3
        + ["f2"] * 3
        + ["r1"] * 3
        + ["r2"] * 3,
        index=columns,
    )
    groups = pd.DataFrame(
        np.tile(labels.to_numpy(), (2, 1)), index=dates, columns=columns
    )
    target = {"Equity": 0.5, "Fixed Income": 0.3, "Rest": 0.2}

    for construction in run.CONSTRUCTIONS:
        weights, diagnostics = run._long_short_weights(
            scores,
            eligibility,
            sleeve_panel,
            groups,
            target,
            q=0.25,
            construction=construction,
        )
        assert weights.sum(axis=1).abs().max() <= 1e-12
        assert weights.abs().sum(axis=1).sub(2.0).abs().max() <= 1e-12
        for sleeve, budget in target.items():
            sleeve_mask = sleeve_panel.eq(sleeve)
            long_budget = weights.clip(lower=0.0).where(sleeve_mask, 0.0).sum(axis=1)
            short_budget = (-weights.clip(upper=0.0)).where(sleeve_mask, 0.0).sum(axis=1)
            assert long_budget.sub(budget).abs().max() <= 1e-12
            assert short_budget.sub(budget).abs().max() <= 1e-12
        assert diagnostics["max_overlap_assets_removed"] == 0


def test_hybrid_side_substitution_preserves_long_short_exposures() -> None:
    """Require side substitution to retain +1/-1 without changing signs."""
    dates = pd.date_range("2025-01-31", periods=2, freq="ME")
    columns = ["a", "b", "c", "d", "e", "f"]
    cluster = pd.DataFrame(
        [
            [0.3, -0.3, -0.3, 0.4, 0.3, -0.4],
            [0.4, -0.2, -0.4, 0.3, 0.3, -0.4],
        ],
        index=dates,
        columns=columns,
    )
    global_weights = pd.DataFrame(
        [
            [0.3, 0.3, 0.4, -0.3, -0.3, -0.4],
            [0.4, 0.2, 0.4, -0.2, -0.4, -0.4],
        ],
        index=dates,
        columns=columns,
    )
    for variant in run.HYBRID_VARIANTS:
        hybrid = run._hybrid_weights(cluster, global_weights, variant)
        assert hybrid.clip(lower=0.0).sum(axis=1).sub(1.0).abs().max() <= 1e-12
        assert (-hybrid.clip(upper=0.0)).sum(axis=1).sub(1.0).abs().max() <= 1e-12
        assert hybrid.sum(axis=1).abs().max() <= 1e-12
        assert hybrid.abs().sum(axis=1).sub(2.0).abs().max() <= 1e-12


def test_rebalance_schedules_are_calendar_stable_and_start_invested() -> None:
    """Pin monthly, two-month, and calendar-quarter decision subsets."""
    dates = pd.date_range("2025-01-31", periods=12, freq="ME")
    assert run._rebalance_dates(dates, "monthly").equals(dates)
    assert run._rebalance_dates(dates, "every_two_months").equals(dates[::2])
    quarterly = run._rebalance_dates(dates, "quarterly")
    expected = dates[[0, 2, 5, 8, 11]]
    assert quarterly.equals(expected)


def test_executed_search_outputs_pin_the_stability_verdict() -> None:
    """Pin the completed grid, hybrid acceptance, and absence of a stable premium."""
    comparison = pd.read_csv(run._root() / "comparison.csv", float_precision="round_trip")
    acceptance = pd.read_csv(run._root() / "acceptance.csv")
    hybrid = pd.read_csv(
        run._root() / "hybrid_comparison.csv", float_precision="round_trip"
    )
    hybrid_acceptance = pd.read_csv(run._root() / "hybrid_acceptance.csv")
    holding = pd.read_csv(
        run._root() / "holding_period_performance.csv", float_precision="round_trip"
    )
    short3_acceptance = pd.read_csv(run._root() / "short3_acceptance.csv")

    assert acceptance["status"].eq("PASS").all()
    assert hybrid_acceptance["status"].eq("PASS").all()
    assert short3_acceptance["status"].eq("PASS").all()
    assert comparison["candidate_id"].nunique() == 113
    wins = comparison.groupby("analysis_window")["beats_global_net_return"].sum()
    assert int(wins.loc[run.TRAIN_WINDOW]) == 0
    assert int(wins.loc[run.EVALUATION_WINDOW]) == 21
    assert int(wins.loc[run.FULL_WINDOW]) == 0

    base_hybrid = hybrid.loc[
        hybrid["candidate_id"].eq(run.BASE_CANDIDATE.candidate_id)
        & hybrid["hybrid_variant"].eq("global_long_cluster_short")
        & hybrid["analysis_window"].eq(run.FULL_WINDOW)
    ].iloc[0]
    assert np.isclose(
        base_hybrid["delta_net_return_annualized"],
        0.00423184404181947,
        atol=1e-12,
    )
    replay = holding.loc[
        holding["analysis_window"].eq(run.FULL_WINDOW)
        & holding["schedule"].eq("monthly")
        & holding["method"].eq("hybrid_global_long_cluster_short")
    ].iloc[0]
    assert np.isclose(
        replay["net_return_annualized"],
        base_hybrid["hybrid_net_return_annualized"],
        atol=1e-12,
    )
