"""Focused regressions for the BlackRock U2 covariance and payoff grid."""
from __future__ import annotations

import numpy as np
import pandas as pd

import papers.cluster_lineage_2026.replication.run_u2_blackrock_etf_grid as run


def test_u1_operating_spec_is_frozen_before_u2_transfer() -> None:
    """Pin the complete U1-selected operating point consumed by U2."""
    spec = run.SPEC
    assert (spec.covariance_frequency, spec.covariance_span) == ("ME", 36)
    assert spec.quantile == 0.25
    assert (
        spec.signal_frequency,
        spec.momentum_long_span,
        spec.momentum_vol_span,
        spec.momentum_short_span,
    ) == ("ME", 12, 13, None)
    assert spec.momentum_mean_adj_type == "NONE"
    assert spec.momentum_min_cluster_size == 5
    assert spec.performance_frequency == "W-WED"
    assert spec.implementation_lag == 1
    assert spec.cost_bps == 10.0
    assert spec.cluster_construction == "group_equal"
    assert spec.global_construction == "asset_equal"


def test_covariance_grid_is_the_frozen_28_cell_design() -> None:
    """Pin cadence coverage and span restrictions."""
    assert len(run._cells()) == 28
    assert run.FREQUENCY_SPANS["ME"] == (12, 24, 36, 52)
    for frequency in ("B", "W-MON", "W-TUE", "W-WED", "W-THU", "W-FRI"):
        assert run.FREQUENCY_SPANS[frequency] == (24, 36, 52, 156)


def test_blackrock_input_preflight_is_green() -> None:
    """Require source identity, classification completeness, and fixed schedules."""
    output = run.preflight()
    assert output["preflight"]["status"].eq("PASS").all()
    quality = output["data_quality"].iloc[0]
    assert quality["funds"] == 480
    assert quality["asset_classes"] == 7
    assert quality["eligible_headline_start"] == 162
    assert quality["current_vintage_survivor_cohort"]


def test_persisted_construction_and_replay_acceptance() -> None:
    """Require all portfolio checks and numerical artifacts to remain green."""
    acceptance = pd.read_csv(run._root() / "acceptance.csv")
    replay = pd.read_csv(run._root() / "determinism.csv")
    assert len(acceptance) == 116
    assert acceptance["status"].eq("PASS").all()
    assert len(replay) == 16
    assert replay["byte_identical"].all()


def test_headline_global_relative_results_are_frozen() -> None:
    """Pin the exploratory leaders and the honest no-win headline verdict."""
    comparison = pd.read_csv(
        run._root() / "comparison_vs_global.csv", float_precision="round_trip"
    )
    headline = comparison.loc[
        comparison["analysis_window"].eq(run.HEADLINE_WINDOW)
    ]
    expected = {
        "long_only": ("ME", 12, -0.0249115447927719),
        "long_short": ("W-THU", 156, -0.000197031887590349),
    }
    for strategy, (frequency, span, delta) in expected.items():
        panel = headline.loc[headline["strategy"].eq(strategy)]
        assert not panel["beats_global_net_return"].any()
        best = panel.sort_values("delta_net_return_annualized", ascending=False).iloc[0]
        assert (best["frequency"], int(best["span"])) == (frequency, span)
        assert np.isclose(best["delta_net_return_annualized"], delta, atol=1e-15)


def test_ew_all_is_reference_only_and_transfer_cell_is_labelled() -> None:
    """Exclude EW from payoff comparisons and require four ME/36 transfer rows."""
    performance = pd.read_csv(run._root() / "performance.csv")
    comparison = pd.read_csv(run._root() / "comparison_vs_global.csv")
    assert not performance["leg"].str.contains("EW", case=False).any()
    assert comparison["benchmark_leg"].eq("global").all()
    assert int(comparison["is_transferred_u1_cell"].sum()) == 4
