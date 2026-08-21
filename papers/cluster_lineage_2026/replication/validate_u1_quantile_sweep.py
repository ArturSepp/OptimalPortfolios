"""Independently validate the U1 group-equal quantile-sweep artifacts."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from papers.cluster_lineage_2026.replication.run_u1_quantile_sweep import (
    QUANTILES,
    _root,
)


METRICS = (
    "net_return_annualized",
    "sharpe_rf0",
    "one_way_turnover_annualized",
    "cost_drag_bp_per_year",
)


def _check_shapes() -> None:
    """Assert the requested grid, windows, legs, and output row counts."""
    root = _root()
    performance = pd.read_csv(root / "performance.csv")
    comparison = pd.read_csv(root / "comparison.csv")
    diagnostics = pd.read_csv(root / "selection_diagnostics.csv")
    assert len(performance) == 40
    assert len(comparison) == 20
    assert len(diagnostics) == 40
    assert tuple(sorted(performance["q"].unique(), reverse=True)) == QUANTILES
    assert set(performance["analysis_window"]) == {
        "headline_20090831_20260630",
        "full_panel",
    }
    assert set(performance["leg"]) == {
        "global",
        "taxonomy",
        "cluster_baseline",
        "cluster_M1_delta_0.02",
    }
    assert not performance["leg"].eq("EW_all").any()


def _check_construction() -> None:
    """Assert all weight and group-budget acceptance rows pass."""
    acceptance = pd.read_csv(_root() / "acceptance.csv")
    assert len(acceptance) == 40
    assert acceptance["status"].eq("PASS").all()
    assert acceptance["weight_sum_error"].max() <= 1e-12
    assert acceptance["group_budget_error"].max() <= 1e-15


def _check_selection_monotonicity() -> None:
    """Assert selected-asset counts weakly increase with q for every leg and window."""
    diagnostics = pd.read_csv(_root() / "selection_diagnostics.csv")
    for _, panel in diagnostics.groupby(["analysis_window", "leg"]):
        selected = panel.sort_values("q")["mean_selected_assets"]
        assert selected.is_monotonic_increasing


def _check_q20_regression() -> float:
    """Assert q=0.20 exactly reproduces the accepted E5b U1 metric rows."""
    root = _root()
    sweep = pd.read_csv(root / "performance.csv", float_precision="round_trip")
    accepted = pd.read_csv(
        root.parents[1] / "group_equal" / "msci_us" / "performance.csv",
        float_precision="round_trip",
    )
    sweep = sweep.loc[sweep["q"].eq(0.20)].set_index(
        ["analysis_window", "leg"]
    )
    accepted = accepted.set_index(["analysis_window", "leg"])
    errors = []
    for index in sweep.index:
        for metric in METRICS:
            errors.append(abs(sweep.loc[index, metric] - accepted.loc[index, metric]))
    max_error = max(errors)
    assert max_error <= 1e-12
    return max_error


def _check_determinism() -> None:
    """Assert all numerical sweep artifacts are byte-identical on replay."""
    replay = pd.read_csv(_root() / "determinism.csv")
    assert len(replay) == 5
    assert replay["byte_identical"].all()
    assert replay["first_sha256"].eq(replay["second_sha256"]).all()


def _check_report() -> None:
    """Assert all 40 absolute performance rows are quoted to displayed precision."""
    report = (
        Path(__file__).resolve().parents[1]
        / "agents"
        / "2026-08-14_sol_E5b_U1_quantile_sweep_report.md"
    ).read_text(encoding="utf-8")
    performance = pd.read_csv(_root() / "performance.csv")
    window_names = {
        "headline_20090831_20260630": "headline",
        "full_panel": "full",
    }
    report_lines = report.splitlines()
    for row in performance.itertuples(index=False):
        prefix = f"| {window_names[row.analysis_window]} | {row.q:.2f} | {row.leg} |"
        matched = [line for line in report_lines if line.startswith(prefix)]
        assert len(matched) == 1, prefix
        cells = [cell.strip() for cell in matched[0].strip("|").split("|")]
        displayed = [float(value) for value in cells[3:10]]
        expected = [
            row.net_return_annualized,
            row.volatility_annualized,
            row.sharpe_rf0,
            row.alpha_vs_ew_annualized,
            row.beta_vs_ew,
            row.one_way_turnover_annualized,
            row.cost_drag_bp_per_year,
        ]
        tolerances = [5e-7] * 5 + [5.6e-6, 5e-5]
        assert all(
            abs(actual - target) <= tolerance
            for actual, target, tolerance in zip(displayed, expected, tolerances)
        ), matched[0]


def main() -> None:
    """Run all independent sweep checks and print the measured verdicts."""
    _check_shapes()
    _check_construction()
    _check_selection_monotonicity()
    regression_error = _check_q20_regression()
    _check_determinism()
    _check_report()
    print("U1 quantile sweep independent validation: PASS")
    print("grid: 5 q values x 2 windows x 4 ranking legs = 40 rows")
    print("construction: 40/40 rows PASS")
    print("selection monotonicity: 8/8 window-leg series PASS")
    print(f"q=0.20 E5b regression max absolute error: {regression_error:.3e}")
    print("determinism: 5/5 numerical CSV artifacts byte-identical")
    print("report: 40/40 absolute performance rows quoted")


if __name__ == "__main__":
    main()
