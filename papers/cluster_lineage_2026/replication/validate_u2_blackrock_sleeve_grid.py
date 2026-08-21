"""Independently validate the BlackRock broad-sleeve allocation experiment."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
from optimalportfolios.alphas.signals.utils import score_within_clusters

import papers.cluster_lineage_2026.replication.run_backtests as e5
import papers.cluster_lineage_2026.replication.run_u1_covar_grid as u1_grid
import papers.cluster_lineage_2026.replication.run_u2_blackrock_etf_grid as u2
import papers.cluster_lineage_2026.replication.run_u2_blackrock_sleeve_grid as run
from papers.cluster_lineage_2026.replication.run_e5b import _group_equal_from_ranks
from papers.cluster_lineage_2026.replication.run_u1_global_grid import _backtest


TOLERANCE = 5e-12
PERFORMANCE_METRICS = (
    "net_total_return",
    "net_return_annualized",
    "volatility_annualized",
    "sharpe_rf0",
    "alpha_vs_ew_annualized",
    "beta_vs_ew",
    "one_way_turnover_annualized",
    "cost_drag_bp_per_year",
    "gross_return_annualized",
)


def _assert_close(actual: float, expected: float, label: str) -> None:
    """Require one floating result to match its independent reconstruction."""
    error = abs(float(actual) - float(expected))
    if error > TOLERANCE:
        raise AssertionError(f"{label}: error={error:.3e} > {TOLERANCE:.1e}")


def _performance(net, gross, ew_nav: pd.Series) -> dict:
    """Compute frozen payoff metrics without using the sleeve runner wrapper."""
    payload = e5._performance_row(net, gross, ew_nav)
    payload["gross_return_annualized"] = (
        payload["net_return_annualized"]
        + payload["cost_drag_bp_per_year"] / 10000.0
    )
    return payload


def _independent_weighted_side(
    scores: pd.DataFrame,
    eligibility: pd.DataFrame,
    sleeve_panel: pd.DataFrame,
    groups: pd.DataFrame,
    target: Mapping[str, float],
) -> pd.DataFrame:
    """Assemble fixed sleeve budgets directly from within-group rank weights."""
    ranks = e5._rank_panel(scores, groups)
    output = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)
    for sleeve in run.SLEEVES:
        sleeve_eligible = eligibility & sleeve_panel.eq(sleeve)
        weights, available, validation = _group_equal_from_ranks(
            ranks,
            sleeve_eligible,
            groups,
            u2.SPEC.quantile,
            u1_grid.UNIVERSE,
        )
        if available.le(0).any():
            raise AssertionError(f"independent {sleeve} reconstruction is empty")
        errors = validation.filter(like="error")
        if float(errors.to_numpy(dtype=float).max()) > TOLERANCE:
            raise AssertionError(f"independent {sleeve} group budgets fail")
        output = output.add(weights.mul(target[sleeve]), fill_value=0.0)
    return output


def _reconstruct_selected_long_only() -> pd.DataFrame:
    """Recompute both fair-control legs for the selected OOS long-only row."""
    dates = u2._dates()
    window_dates = dates[(dates >= run.TEST_START) & (dates <= run.TEST_END)]
    daily = u2._read_daily()
    eligibility_all = u2._eligibility_for_dates(daily, dates)
    eligibility = eligibility_all.reindex(index=window_dates)
    signal = u2._signal_inputs(daily, dates, eligibility_all)
    prices_all = u2._performance_prices(daily)
    prices = u2._window_prices(prices_all, window_dates)
    ew_nav = u2._ew_reference(
        prices_all, eligibility_all, window_dates, run.TEST_WINDOW
    )
    sleeves = run._broad_sleeves(eligibility.columns)
    sleeve_panel = run._sleeve_panel(window_dates, sleeves)
    selected = run._weight_grid().set_index("weight_id").loc["E70_F20_R10"]
    target = run._target_weights(selected)
    global_scores = signal["global"].reindex(
        index=window_dates, columns=eligibility.columns
    ).where(eligibility)
    clusters, _ = u2._load_partition(*run.PRIMARY_CELLS["long_only"])
    cluster_scores = score_within_clusters(
        raw_signal=signal["raw_source"],
        rolling_clusters=u2._panel_dict(clusters),
        min_cluster_size=u2.SPEC.momentum_min_cluster_size,
    ).reindex(index=window_dates, columns=eligibility.columns).where(eligibility)
    cluster_groups = run._hierarchical_groups(
        clusters.reindex(index=window_dates, columns=eligibility.columns),
        sleeve_panel,
    )
    rows = []
    for method, scores, groups in (
        ("sleeve_global", global_scores, sleeve_panel),
        ("sleeve_cluster_primary", cluster_scores, cluster_groups),
    ):
        weights = _independent_weighted_side(
            scores, eligibility, sleeve_panel, groups, target
        )
        if float(weights.sum(axis=1).sub(1.0).abs().max()) > TOLERANCE:
            raise AssertionError(f"independent {method} weights do not sum to one")
        for sleeve in run.SLEEVES:
            measured = weights.where(sleeve_panel.eq(sleeve), 0.0).sum(axis=1)
            if float(measured.sub(target[sleeve]).abs().max()) > TOLERANCE:
                raise AssertionError(f"independent {method} {sleeve} budget fails")
        net, gross = _backtest(
            prices,
            weights,
            u2.SPEC.cost_bps / 10000.0,
            f"independent_{method}_E70_F20_R10_long_only",
        )
        rows.append({"method": method, **_performance(net, gross, ew_nav)})
    return pd.DataFrame(rows).set_index("method")


def _validate_comparison_arithmetic(
    performance: pd.DataFrame, comparison: pd.DataFrame
) -> None:
    """Recompute every cluster-minus-control delta from persisted payoff rows."""
    sleeve_global = performance.loc[
        performance["method"].eq("sleeve_global")
    ].set_index(["analysis_window", "strategy", "weight_id"])
    original = performance.loc[
        performance["method"].eq("original_global")
    ].set_index(["analysis_window", "strategy"])
    for _, row in comparison.iterrows():
        key = (row["analysis_window"], row["strategy"], row["weight_id"])
        same_budget = sleeve_global.loc[key]
        unconstrained = original.loc[(row["analysis_window"], row["strategy"])]
        for metric in run.COMPARISON_METRICS:
            _assert_close(
                row[f"delta_vs_sleeve_global_{metric}"],
                row[metric] - same_budget[metric],
                f"same-budget delta {key} {row['method']} {metric}",
            )
            _assert_close(
                row[f"delta_vs_original_global_{metric}"],
                row[metric] - unconstrained[metric],
                f"original-global delta {key} {row['method']} {metric}",
            )


def validate() -> None:
    """Run structural, selection, budget, payoff, and replay checks."""
    root = run._root()
    grid = pd.read_csv(root / "weight_grid.csv", float_precision="round_trip")
    performance = pd.read_csv(
        root / "performance.csv", float_precision="round_trip"
    )
    comparison = pd.read_csv(
        root / "comparison.csv", float_precision="round_trip"
    )
    selection = pd.read_csv(root / "selection.csv", float_precision="round_trip")
    selected = pd.read_csv(
        root / "selected_evaluation.csv", float_precision="round_trip"
    )
    allocation = pd.read_csv(
        root / "allocation_diagnostics.csv", float_precision="round_trip"
    )
    acceptance = pd.read_csv(
        root / "acceptance.csv", float_precision="round_trip"
    )
    regression = pd.read_csv(root / "regression.csv", float_precision="round_trip")
    replay = pd.read_csv(root / "determinism.csv")

    if len(grid) != 8 or not np.allclose(
        grid[["equity_weight", "fixed_income_weight", "rest_weight"]].sum(axis=1),
        1.0,
        atol=TOLERANCE,
        rtol=0.0,
    ):
        raise AssertionError("persisted strategic weight grid is incomplete")
    expected_rows = {
        "performance": (len(performance), 150),
        "comparison": (len(comparison), 96),
        "selection": (len(selection), 8),
        "selected_evaluation": (len(selected), 24),
        "allocation_diagnostics": (len(allocation), 450),
        "acceptance": (len(acceptance), 150),
        "regression": (len(regression), 2),
        "determinism": (len(replay), 9),
    }
    failures = {
        name: (measured, expected)
        for name, (measured, expected) in expected_rows.items()
        if measured != expected
    }
    if failures:
        raise AssertionError(f"persisted row counts fail: {failures}")
    if not acceptance["status"].eq("PASS").all():
        raise AssertionError("persisted portfolio acceptance contains a failure")
    if not regression["status"].eq("PASS").all():
        raise AssertionError("accepted original-global regression contains a failure")
    if not replay["byte_identical"].all():
        raise AssertionError("persisted deterministic replay contains a failure")
    if performance["method"].str.contains("EW", case=False).any():
        raise AssertionError("EW-all was incorrectly emitted as a ranking leg")

    grouped = allocation.loc[
        ~allocation["method"].eq("original_global")
    ].copy()
    target = grouped["target_budget"]
    long_only_error = (
        grouped.loc[grouped["strategy"].eq("long_only"), "average_long_exposure"]
        .sub(target)
        .abs()
        .max()
    )
    if float(long_only_error) > TOLERANCE:
        raise AssertionError("long-only top-level sleeve budgets fail")
    short = grouped.loc[grouped["strategy"].eq("long_short")]
    long_error = short["average_long_exposure"].sub(short["target_budget"]).abs().max()
    short_error = (
        short["average_short_exposure_abs"]
        .sub(short["target_budget"])
        .abs()
        .max()
    )
    if float(long_error) > TOLERANCE:
        raise AssertionError("long-short long-side sleeve budgets fail")
    if float(short_error) > TOLERANCE:
        raise AssertionError("long-short short-side sleeve budgets fail")
    if float(short["average_net_exposure"].abs().max()) > TOLERANCE:
        raise AssertionError("long-short sleeve neutrality fails")

    _validate_comparison_arithmetic(performance, comparison)
    training = comparison.loc[
        comparison["analysis_window"].eq(run.TRAIN_WINDOW)
        & comparison["method"].eq("sleeve_cluster_primary")
    ]
    chosen = selection.set_index(["strategy", "selection_type"])
    for strategy, panel in training.groupby("strategy"):
        absolute = panel.sort_values(
            ["net_return_annualized", "sharpe_rf0", "one_way_turnover_annualized"],
            ascending=[False, False, True],
        ).iloc[0]
        if chosen.loc[(strategy, "training_best_absolute"), "weight_id"] != absolute["weight_id"]:
            raise AssertionError(f"{strategy} absolute selection leaks evaluation data")

    reconstructed = _reconstruct_selected_long_only()
    persisted = performance.loc[
        performance["analysis_window"].eq(run.TEST_WINDOW)
        & performance["strategy"].eq("long_only")
        & performance["weight_id"].eq("E70_F20_R10")
        & performance["method"].isin(reconstructed.index)
    ].set_index("method")
    for method in reconstructed.index:
        for metric in PERFORMANCE_METRICS:
            _assert_close(
                reconstructed.loc[method, metric],
                persisted.loc[method, metric],
                f"independent OOS {method} {metric}",
            )
    print(
        "BlackRock broad-sleeve independent validation: PASS "
        "(8 weights, 150 portfolios, 96 comparisons, 2 reconstructed payoffs, "
        "9 hashes)"
    )


if __name__ == "__main__":
    validate()
