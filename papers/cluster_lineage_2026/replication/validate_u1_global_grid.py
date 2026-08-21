"""Independently validate the U1 original-universe global-benchmark grid."""
from __future__ import annotations

import pandas as pd

from papers.cluster_lineage_2026.replication.run_u1_global_grid import (
    CONFIGS,
    HEADLINE,
    QUANTILES,
    _root,
)
from papers.cluster_lineage_2026.replication.run_e5b import _root as e5b_root


METRICS = (
    "net_return_annualized",
    "sharpe_rf0",
    "one_way_turnover_annualized",
    "cost_drag_bp_per_year",
)


def _check_grid() -> None:
    """Assert complete two-window config-by-q coverage and sole global yardstick."""
    performance = pd.read_csv(_root() / "performance.csv")
    comparison = pd.read_csv(_root() / "comparison_vs_global.csv")
    assert len(performance) == 90
    assert len(comparison) == 80
    assert tuple(sorted(performance["q"].unique(), reverse=True)) == QUANTILES
    assert set(comparison["config"]) == {config.value for config in CONFIGS}
    assert set(performance["analysis_window"]) == {HEADLINE, "full_panel"}
    assert not performance["leg"].str.contains("taxonomy", case=False).any()
    assert not any("taxonomy" in column.lower() for column in comparison.columns)


def _check_construction() -> None:
    """Assert all group-equal weight panels satisfy both numerical tolerances."""
    acceptance = pd.read_csv(_root() / "acceptance.csv")
    assert len(acceptance) == 80
    assert acceptance["status"].eq("PASS").all()
    assert acceptance["weight_sum_error"].max() <= 1e-12
    assert acceptance["group_budget_error"].max() <= 1e-15


def _check_regressions() -> float:
    """Assert reused globals and overlapping cluster rows reproduce the q sweep."""
    grid = pd.read_csv(_root() / "performance.csv", float_precision="round_trip")
    sweep = pd.read_csv(
        e5b_root() / "quantile_sweep" / "msci_us" / "performance.csv",
        float_precision="round_trip",
    )
    indexes = ["analysis_window", "q", "leg"]
    grid = grid.set_index(indexes)
    sweep = sweep.set_index(indexes)
    overlap = sweep.index.intersection(grid.index)
    assert len(overlap) == 30
    errors = []
    for index in overlap:
        for metric in METRICS:
            errors.append(abs(grid.loc[index, metric] - sweep.loc[index, metric]))
    max_error = max(errors)
    assert max_error <= 1e-12
    return max_error


def _check_rankings() -> None:
    """Assert headline winner identities and fidelity-aware conclusions."""
    rankings = pd.read_csv(_root() / "rankings.csv")
    assert len(rankings) == 135
    raw = rankings.loc[
        rankings["analysis_window"].eq(HEADLINE)
        & rankings["scope"].eq("all_configs")
    ].sort_values("rank")
    admissible = rankings.loc[
        rankings["analysis_window"].eq(HEADLINE)
        & rankings["scope"].eq("fidelity_admissible")
    ].sort_values("rank")
    assert raw.iloc[0]["config"] == "M1_delta_0.10"
    assert raw.iloc[0]["q"] == 0.30
    assert admissible.iloc[0]["config"] == "M0_quarterly_hold"
    assert admissible.iloc[0]["q"] == 0.30
    headline = pd.read_csv(_root() / "comparison_vs_global.csv")
    headline = headline.loc[headline["analysis_window"].eq(HEADLINE)]
    admissible_status = {"REFERENCE", "IN_BAND", "IN_BAND_FULL_ONLY"}
    assert not headline.loc[
        headline["fidelity_status"].isin(admissible_status), "beats_global_both"
    ].any()
    raw_beats = headline.loc[headline["beats_global_both"]]
    assert len(raw_beats) == 3
    assert raw_beats["fidelity_status"].eq("REJECTED_FIDELITY").all()


def _check_determinism() -> None:
    """Assert every numerical grid artifact is byte-identical on replay."""
    replay = pd.read_csv(_root() / "determinism.csv")
    assert len(replay) == 6
    assert replay["byte_identical"].all()
    assert replay["first_sha256"].eq(replay["second_sha256"]).all()


def main() -> None:
    """Run all independent grid validations and print measured verdicts."""
    _check_grid()
    _check_construction()
    regression_error = _check_regressions()
    _check_rankings()
    _check_determinism()
    print("U1 global-benchmark grid independent validation: PASS")
    print("grid: 8 configs x 5 q values x 2 windows = 80 cluster rows")
    print("construction: 80/80 PASS")
    print(f"overlap regression max absolute error: {regression_error:.3e}")
    print("headline raw winner: M1_delta_0.10 at q=0.30 (fidelity rejected)")
    print("headline admissible winner: M0_quarterly_hold at q=0.30")
    print("headline admissible configs beating global on return and Sharpe: 0")
    print("determinism: 6/6 numerical artifacts byte-identical")


if __name__ == "__main__":
    main()
