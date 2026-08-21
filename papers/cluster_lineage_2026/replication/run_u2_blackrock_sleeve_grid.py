"""Run broad-sleeve BlackRock ETF rank and cluster comparisons.

This follow-up removes the strategic asset-class mismatch discovered in the original U2
grid.  Both the benchmark and cluster legs receive identical fixed budgets across Equity,
Fixed Income, and Rest.  The sleeve-global leg ranks within each broad sleeve.  The
sleeve-cluster leg ranks within correlation clusters and allocates each sleeve budget
equally across its available clusters.  Thus any cluster-minus-global payoff difference
is generated below the common strategic budget, not by weakening only the benchmark.

The candidate grid contains eight plausible allocations in ten-point increments, bounded
to 40-70% Equity, 20-40% Fixed Income, and 10-30% Rest.  It includes the owner's proposed
50/30/20 allocation.  Weight selection uses 2009-08-31 through 2017-12-31 only; evaluation
uses 2018-01-31 through 2026-06-30.  The full headline window is descriptive.  Long-short
books impose the same sleeve budget independently on the long and short sides, so each
sleeve is net neutral before cross-sleeve aggregation.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from optimalportfolios.alphas.signals.utils import score_within_clusters

import papers.cluster_lineage_2026.replication.run_backtests as e5
import papers.cluster_lineage_2026.replication.run_u1_covar_grid as u1_grid
import papers.cluster_lineage_2026.replication.run_u1_me36_long_short as u1_single
import papers.cluster_lineage_2026.replication.run_u2_blackrock_etf_grid as u2
from papers.cluster_lineage_2026.replication.run_e5b import (
    GROUP_BUDGET_TOLERANCE,
    _group_equal_from_ranks,
)
from papers.cluster_lineage_2026.replication.run_u1_global_grid import _backtest


RUNNER = (
    "papers/cluster_lineage_2026/replication/run_u2_blackrock_sleeve_grid.py"
)
SLEEVES = ("Equity", "Fixed Income", "Rest")
FIXED_WEIGHT_ID = "E50_F30_R20"
TRAIN_WINDOW = "selection_20090831_20171231"
TEST_WINDOW = "evaluation_20180131_20260630"
HEADLINE_WINDOW = u2.HEADLINE_WINDOW
TRAIN_START = pd.Timestamp("2009-08-31")
TRAIN_END = pd.Timestamp("2017-12-31")
TEST_START = pd.Timestamp("2018-01-31")
TEST_END = pd.Timestamp("2026-06-30")
PRIMARY_CELLS = {
    "long_only": ("ME", 12),
    "long_short": ("W-THU", 156),
}
TRANSFER_CELL = (u2.SPEC.covariance_frequency, u2.SPEC.covariance_span)
TOLERANCE = 1e-12
COMPARISON_METRICS = u2.COMPARISON_METRICS


def _root() -> Path:
    """Return and create the external broad-sleeve output directory."""
    root = u2._root() / "broad_sleeve_weight_grid"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _weight_id(equity: float, fixed_income: float, rest: float) -> str:
    """Return a stable identifier for one three-sleeve allocation."""
    return (
        f"E{int(round(100 * equity)):02d}_"
        f"F{int(round(100 * fixed_income)):02d}_"
        f"R{int(round(100 * rest)):02d}"
    )


def _weight_grid() -> pd.DataFrame:
    """Return the constrained eight-point strategic sleeve grid."""
    rows = []
    for equity in (0.4, 0.5, 0.6, 0.7):
        for fixed_income in (0.2, 0.3, 0.4):
            rest = round(1.0 - equity - fixed_income, 10)
            if 0.1 - 1e-12 <= rest <= 0.3 + 1e-12:
                rows.append(
                    {
                        "weight_id": _weight_id(equity, fixed_income, rest),
                        "equity_weight": equity,
                        "fixed_income_weight": fixed_income,
                        "rest_weight": rest,
                        "is_owner_50_30_20": (
                            abs(equity - 0.5) <= 1e-12
                            and abs(fixed_income - 0.3) <= 1e-12
                            and abs(rest - 0.2) <= 1e-12
                        ),
                    }
                )
    frame = pd.DataFrame(rows).sort_values(
        ["equity_weight", "fixed_income_weight", "rest_weight"]
    ).reset_index(drop=True)
    if len(frame) != 8 or frame["weight_id"].duplicated().any():
        raise AssertionError("the broad-sleeve weight grid is not the frozen eight cells")
    if int(frame["is_owner_50_30_20"].sum()) != 1:
        raise AssertionError("the owner's 50/30/20 allocation is absent or duplicated")
    return frame


def _target_weights(row: pd.Series) -> Mapping[str, float]:
    """Return one grid row as a broad-sleeve budget mapping."""
    return {
        "Equity": float(row["equity_weight"]),
        "Fixed Income": float(row["fixed_income_weight"]),
        "Rest": float(row["rest_weight"]),
    }


def _broad_sleeves(columns: pd.Index) -> pd.Series:
    """Map official Aladdin asset classes to Equity, Fixed Income, or Rest."""
    metadata = pd.read_csv(u2.METADATA_FILE).set_index("ticker")
    asset_class = metadata["asset_class"].reindex(columns)
    if asset_class.isna().any():
        raise AssertionError("an ETF lacks an official Aladdin asset class")
    broad = asset_class.where(asset_class.isin(SLEEVES[:2]), "Rest")
    broad.name = "broad_sleeve"
    return broad


def _sleeve_panel(index: pd.DatetimeIndex, sleeves: pd.Series) -> pd.DataFrame:
    """Broadcast the static broad-sleeve map over decision dates."""
    return pd.DataFrame(
        np.tile(sleeves.to_numpy(), (len(index), 1)),
        index=index,
        columns=sleeves.index,
    )


def _hierarchical_groups(
    clusters: pd.DataFrame, sleeve_panel: pd.DataFrame
) -> pd.DataFrame:
    """Split each correlation cluster by broad sleeve for nested ranking."""
    cluster_text = clusters.astype("Int64").astype(str)
    groups = sleeve_panel.astype(str) + "|" + cluster_text
    return groups.where(clusters.notna())


def _weighted_side_from_ranks(
    ranks: pd.DataFrame,
    eligibility: pd.DataFrame,
    sleeve_panel: pd.DataFrame,
    subgroups: pd.DataFrame,
    target: Mapping[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Allocate fixed sleeve budgets, equal across available groups within sleeve."""
    combined = pd.DataFrame(0.0, index=ranks.index, columns=ranks.columns)
    rows = []
    for sleeve in SLEEVES:
        sleeve_eligibility = eligibility & sleeve_panel.eq(sleeve)
        weights, group_counts, validation = _group_equal_from_ranks(
            ranks,
            sleeve_eligibility,
            subgroups,
            u2.SPEC.quantile,
            u1_grid.UNIVERSE,
        )
        if group_counts.le(0).any():
            raise AssertionError(f"{sleeve} lacks a selected group on one decision date")
        combined = combined.add(weights.mul(target[sleeve]), fill_value=0.0)
        rows.append(
            pd.DataFrame(
                {
                    "date": ranks.index,
                    "sleeve": sleeve,
                    "target_budget": target[sleeve],
                    "available_groups": group_counts.to_numpy(),
                    "pre_scale_weight_error": validation[
                        "weight_sum_abs_error"
                    ].to_numpy(),
                    "pre_scale_group_budget_error": validation[
                        "max_group_budget_abs_error"
                    ].to_numpy(),
                }
            )
        )
    diagnostics = pd.concat(rows, ignore_index=True)
    return combined, diagnostics


def _sleeve_budget_error(
    weights: pd.DataFrame,
    sleeve_panel: pd.DataFrame,
    target: Mapping[str, float],
    *,
    absolute_short: bool = False,
) -> float:
    """Return the largest top-level sleeve-budget error."""
    measured = -weights.clip(upper=0.0) if absolute_short else weights.clip(lower=0.0)
    errors = []
    for sleeve in SLEEVES:
        exposure = measured.where(sleeve_panel.eq(sleeve), 0.0).sum(axis=1)
        errors.append(float(exposure.sub(target[sleeve]).abs().max()))
    return max(errors)


def _long_only_weights(
    scores: pd.DataFrame,
    eligibility: pd.DataFrame,
    sleeve_panel: pd.DataFrame,
    subgroups: pd.DataFrame,
    target: Mapping[str, float],
) -> tuple[pd.DataFrame, dict]:
    """Return hierarchical sleeve/cluster long-only weights and exact checks."""
    ranks = e5._rank_panel(scores, subgroups)
    weights, diagnostics = _weighted_side_from_ranks(
        ranks, eligibility, sleeve_panel, subgroups, target
    )
    weight_error = float(weights.sum(axis=1).sub(1.0).abs().max())
    sleeve_error = _sleeve_budget_error(weights, sleeve_panel, target)
    group_error = float(diagnostics["pre_scale_group_budget_error"].max())
    return weights, {
        "max_weight_sum_abs_error": weight_error,
        "max_top_level_sleeve_budget_abs_error": sleeve_error,
        "max_within_sleeve_group_budget_abs_error": group_error,
        "max_overlap_assets_removed": 0,
    }


def _renormalize_side_by_sleeve(
    side: pd.DataFrame,
    sleeve_panel: pd.DataFrame,
    target: Mapping[str, float],
) -> pd.DataFrame:
    """Restore exact top-level budgets after removing long/short overlap."""
    output = pd.DataFrame(0.0, index=side.index, columns=side.columns)
    for sleeve in SLEEVES:
        sleeve_side = side.where(sleeve_panel.eq(sleeve), 0.0)
        total = sleeve_side.sum(axis=1)
        if total.le(0.0).any():
            raise AssertionError(f"{sleeve} has an empty signed side after overlap removal")
        output = output.add(
            sleeve_side.div(total, axis=0).mul(target[sleeve]),
            fill_value=0.0,
        )
    return output


def _long_short_weights(
    scores: pd.DataFrame,
    eligibility: pd.DataFrame,
    sleeve_panel: pd.DataFrame,
    subgroups: pd.DataFrame,
    target: Mapping[str, float],
) -> tuple[pd.DataFrame, dict]:
    """Return hierarchical +1/-1 weights with exact sleeve neutrality."""
    long_ranks = e5._rank_panel(scores, subgroups)
    short_ranks = e5._rank_panel(-scores, subgroups)
    long_raw, long_diagnostics = _weighted_side_from_ranks(
        long_ranks, eligibility, sleeve_panel, subgroups, target
    )
    short_raw, short_diagnostics = _weighted_side_from_ranks(
        short_ranks, eligibility, sleeve_panel, subgroups, target
    )
    overlap = long_raw.gt(0.0) & short_raw.gt(0.0)
    long_book = _renormalize_side_by_sleeve(
        long_raw.mask(overlap, 0.0), sleeve_panel, target
    )
    short_book = _renormalize_side_by_sleeve(
        short_raw.mask(overlap, 0.0), sleeve_panel, target
    )
    weights = long_book - short_book
    long_error = float(long_book.sum(axis=1).sub(1.0).abs().max())
    short_error = float(short_book.sum(axis=1).sub(1.0).abs().max())
    net_error = float(weights.sum(axis=1).abs().max())
    gross_error = float(weights.abs().sum(axis=1).sub(2.0).abs().max())
    sleeve_error = max(
        _sleeve_budget_error(weights, sleeve_panel, target),
        _sleeve_budget_error(
            weights, sleeve_panel, target, absolute_short=True
        ),
    )
    group_error = max(
        float(long_diagnostics["pre_scale_group_budget_error"].max()),
        float(short_diagnostics["pre_scale_group_budget_error"].max()),
    )
    return weights, {
        "max_weight_sum_abs_error": net_error,
        "max_top_level_sleeve_budget_abs_error": sleeve_error,
        "max_within_sleeve_group_budget_abs_error": group_error,
        "max_long_exposure_abs_error": long_error,
        "max_short_exposure_abs_error": short_error,
        "max_net_exposure_abs_error": net_error,
        "max_gross_exposure_abs_error": gross_error,
        "max_overlap_assets_removed": int(overlap.sum(axis=1).max()),
    }


def _performance_payload(net, gross, ew_nav: pd.Series) -> dict:
    """Return frozen payoff metrics plus pre-cost annual return."""
    payload = e5._performance_row(net, gross, ew_nav)
    payload["gross_return_annualized"] = (
        payload["net_return_annualized"]
        + payload["cost_drag_bp_per_year"] / 10000.0
    )
    return payload


def _run_leg(
    *,
    window: str,
    strategy: str,
    method: str,
    weight_id: str,
    target: Mapping[str, float] | None,
    frequency: str,
    span: float,
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    diagnostics: Mapping[str, float],
    sleeve_panel: pd.DataFrame,
    ew_nav: pd.Series,
) -> tuple[dict, dict, list[dict]]:
    """Backtest one leg and return performance, acceptance, and sleeve exposures."""
    net, gross = _backtest(
        prices,
        weights,
        u2.SPEC.cost_bps / 10000.0,
        f"{u2.UNIVERSE}_{window}_{strategy}_{method}_{weight_id}",
    )
    performance = {
        "universe": u2.UNIVERSE,
        "analysis_window": window,
        "strategy": strategy,
        "method": method,
        "weight_id": weight_id,
        "equity_weight": np.nan if target is None else target["Equity"],
        "fixed_income_weight": (
            np.nan if target is None else target["Fixed Income"]
        ),
        "rest_weight": np.nan if target is None else target["Rest"],
        "frequency": frequency,
        "span": span,
        **_performance_payload(net, gross, ew_nav),
        "runner": RUNNER,
    }
    maximum_error = max(
        abs(float(value))
        for key, value in diagnostics.items()
        if "error" in key
    )
    acceptance = {
        "analysis_window": window,
        "strategy": strategy,
        "method": method,
        "weight_id": weight_id,
        "frequency": frequency,
        "span": span,
        **diagnostics,
        "error_tolerance": TOLERANCE,
        "group_budget_tolerance": GROUP_BUDGET_TOLERANCE,
        "status": (
            "PASS"
            if maximum_error <= TOLERANCE
            and float(
                diagnostics.get("max_within_sleeve_group_budget_abs_error", 0.0)
            )
            <= GROUP_BUDGET_TOLERANCE
            else "FAIL"
        ),
    }
    exposure_rows = []
    for sleeve in SLEEVES:
        long_exposure = weights.clip(lower=0.0).where(
            sleeve_panel.eq(sleeve), 0.0
        ).sum(axis=1)
        short_exposure = (-weights.clip(upper=0.0)).where(
            sleeve_panel.eq(sleeve), 0.0
        ).sum(axis=1)
        exposure_rows.append(
            {
                "analysis_window": window,
                "strategy": strategy,
                "method": method,
                "weight_id": weight_id,
                "sleeve": sleeve,
                "target_budget": np.nan if target is None else target[sleeve],
                "average_long_exposure": float(long_exposure.mean()),
                "average_short_exposure_abs": float(short_exposure.mean()),
                "average_net_exposure": float(
                    (long_exposure - short_exposure).mean()
                ),
            }
        )
    return performance, acceptance, exposure_rows


def _original_global_weights(
    strategy: str,
    scores: pd.DataFrame,
    eligibility: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:
    """Rebuild the accepted unconstrained global rank for one strategy."""
    groups = pd.DataFrame("global", index=scores.index, columns=scores.columns)
    if strategy == "long_only":
        weights, _, validation = u2._long_only_weights(scores, eligibility, groups)
        return weights, {
            "max_weight_sum_abs_error": float(
                validation["weight_sum_abs_error"].max()
            ),
            "max_top_level_sleeve_budget_abs_error": 0.0,
            "max_within_sleeve_group_budget_abs_error": float(
                validation["max_group_budget_abs_error"].max()
            ),
            "max_overlap_assets_removed": 0,
        }
    weights, exposure, validation = u1_single._leg_weights(
        scores, eligibility, groups
    )
    return weights, {
        "max_weight_sum_abs_error": float(exposure["net_exposure"].abs().max()),
        "max_top_level_sleeve_budget_abs_error": 0.0,
        "max_within_sleeve_group_budget_abs_error": float(
            validation.filter(like="group_budget_error").to_numpy().max()
        ),
        "max_long_exposure_abs_error": float(
            exposure["long_exposure"].sub(1.0).abs().max()
        ),
        "max_short_exposure_abs_error": float(
            exposure["short_exposure_abs"].sub(1.0).abs().max()
        ),
        "max_net_exposure_abs_error": float(exposure["net_exposure"].abs().max()),
        "max_gross_exposure_abs_error": float(
            exposure["gross_exposure"].sub(2.0).abs().max()
        ),
        "max_overlap_assets_removed": int(exposure["overlap_assets_removed"].max()),
    }


def _comparison(performance: pd.DataFrame) -> pd.DataFrame:
    """Compare every cluster leg with identical-budget sleeve and original globals."""
    sleeve_global = performance.loc[
        performance["method"].eq("sleeve_global")
    ].set_index(["analysis_window", "strategy", "weight_id"])
    original = performance.loc[
        performance["method"].eq("original_global")
    ].set_index(["analysis_window", "strategy"])
    rows = []
    clusters = performance.loc[performance["method"].str.startswith("sleeve_cluster")]
    for _, cluster in clusters.iterrows():
        key = (
            cluster["analysis_window"],
            cluster["strategy"],
            cluster["weight_id"],
        )
        sleeve = sleeve_global.loc[key]
        global_row = original.loc[(cluster["analysis_window"], cluster["strategy"])]
        row = cluster.to_dict()
        for metric in COMPARISON_METRICS:
            row[f"sleeve_global_{metric}"] = sleeve[metric]
            row[f"delta_vs_sleeve_global_{metric}"] = cluster[metric] - sleeve[metric]
            row[f"original_global_{metric}"] = global_row[metric]
            row[f"delta_vs_original_global_{metric}"] = (
                cluster[metric] - global_row[metric]
            )
        row["beats_sleeve_global_net_return"] = (
            row["delta_vs_sleeve_global_net_return_annualized"] > 0.0
        )
        row["beats_original_global_net_return"] = (
            row["delta_vs_original_global_net_return_annualized"] > 0.0
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _selection_table(comparison: pd.DataFrame) -> pd.DataFrame:
    """Select absolute, relative, fixed, and common weights on training only."""
    training = comparison.loc[
        comparison["analysis_window"].eq(TRAIN_WINDOW)
        & comparison["method"].eq("sleeve_cluster_primary")
    ].copy()
    rows = []
    for strategy, panel in training.groupby("strategy", sort=False):
        fixed = panel.loc[panel["weight_id"].eq(FIXED_WEIGHT_ID)].iloc[0]
        absolute = panel.sort_values(
            ["net_return_annualized", "sharpe_rf0", "one_way_turnover_annualized"],
            ascending=[False, False, True],
        ).iloc[0]
        relative = panel.sort_values(
            [
                "delta_vs_sleeve_global_net_return_annualized",
                "delta_vs_sleeve_global_sharpe_rf0",
            ],
            ascending=[False, False],
        ).iloc[0]
        for selection_type, selected in (
            ("owner_fixed_50_30_20", fixed),
            ("training_best_absolute", absolute),
            ("training_best_cluster_edge", relative),
        ):
            rows.append(
                {
                    "strategy": strategy,
                    "selection_type": selection_type,
                    "weight_id": selected["weight_id"],
                    "equity_weight": selected["equity_weight"],
                    "fixed_income_weight": selected["fixed_income_weight"],
                    "rest_weight": selected["rest_weight"],
                }
            )

    training["absolute_rank"] = training.groupby("strategy")[
        "net_return_annualized"
    ].rank(method="min", ascending=False)
    training["relative_rank"] = training.groupby("strategy")[
        "delta_vs_sleeve_global_net_return_annualized"
    ].rank(method="min", ascending=False)
    common = (
        training.groupby(
            ["weight_id", "equity_weight", "fixed_income_weight", "rest_weight"],
            as_index=False,
        )
        .agg(
            mean_absolute_rank=("absolute_rank", "mean"),
            mean_relative_rank=("relative_rank", "mean"),
            worst_absolute_rank=("absolute_rank", "max"),
            worst_relative_rank=("relative_rank", "max"),
        )
    )
    common["combined_rank_score"] = (
        common["mean_absolute_rank"] + common["mean_relative_rank"]
    )
    chosen = common.sort_values(
        ["combined_rank_score", "worst_relative_rank", "worst_absolute_rank"]
    ).iloc[0]
    for strategy in training["strategy"].unique():
        rows.append(
            {
                "strategy": strategy,
                "selection_type": "training_common_balanced",
                "weight_id": chosen["weight_id"],
                "equity_weight": chosen["equity_weight"],
                "fixed_income_weight": chosen["fixed_income_weight"],
                "rest_weight": chosen["rest_weight"],
            }
        )
    return pd.DataFrame(rows)


def _selected_evaluation(
    comparison: pd.DataFrame, selection: pd.DataFrame
) -> pd.DataFrame:
    """Return training, evaluation, and headline rows for selected primary allocations."""
    primary = comparison.loc[
        comparison["method"].eq("sleeve_cluster_primary")
    ]
    return selection.merge(
        primary,
        on=[
            "strategy",
            "weight_id",
            "equity_weight",
            "fixed_income_weight",
            "rest_weight",
        ],
        how="left",
        validate="many_to_many",
    ).sort_values(["strategy", "selection_type", "analysis_window"])


def _rank_stability(comparison: pd.DataFrame) -> pd.DataFrame:
    """Measure training-to-evaluation weight ranking stability."""
    primary = comparison.loc[
        comparison["method"].eq("sleeve_cluster_primary")
    ]
    rows = []
    metrics = (
        "net_return_annualized",
        "delta_vs_sleeve_global_net_return_annualized",
        "delta_vs_original_global_net_return_annualized",
    )
    for strategy, panel in primary.groupby("strategy", sort=False):
        train = panel.loc[panel["analysis_window"].eq(TRAIN_WINDOW)].set_index(
            "weight_id"
        )
        test = panel.loc[panel["analysis_window"].eq(TEST_WINDOW)].set_index(
            "weight_id"
        )
        for metric in metrics:
            rows.append(
                {
                    "strategy": strategy,
                    "metric": metric,
                    "training_evaluation_spearman": float(
                        train[metric].corr(test[metric], method="spearman")
                    ),
                }
            )
    return pd.DataFrame(rows)


def _global_regression(performance: pd.DataFrame) -> pd.DataFrame:
    """Require headline original-global rows to reproduce the accepted U2 run."""
    accepted = pd.read_csv(
        u2._root() / "performance.csv", float_precision="round_trip"
    )
    accepted = accepted.loc[
        accepted["analysis_window"].eq(HEADLINE_WINDOW)
        & accepted["leg"].eq("global")
    ].set_index("strategy")
    current = performance.loc[
        performance["analysis_window"].eq(HEADLINE_WINDOW)
        & performance["method"].eq("original_global")
    ].set_index("strategy")
    rows = []
    for strategy in ("long_only", "long_short"):
        errors = {
            metric: abs(
                float(current.loc[strategy, metric])
                - float(accepted.loc[strategy, metric])
            )
            for metric in COMPARISON_METRICS
        }
        maximum = max(errors.values())
        rows.append(
            {
                "check": "accepted original-global payoff regression",
                "strategy": strategy,
                "measured_max_abs_error": maximum,
                "tolerance": TOLERANCE,
                "status": "PASS" if maximum <= TOLERANCE else "FAIL",
            }
        )
    frame = pd.DataFrame(rows)
    if not frame["status"].eq("PASS").all():
        raise AssertionError(frame)
    return frame


def run() -> Mapping[str, pd.DataFrame]:
    """Execute the strategic sleeve grid, selection split, and exact checks."""
    started = time.perf_counter()
    grid = _weight_grid()
    dates = u2._dates()
    windows = {
        TRAIN_WINDOW: dates[(dates >= TRAIN_START) & (dates <= TRAIN_END)],
        TEST_WINDOW: dates[(dates >= TEST_START) & (dates <= TEST_END)],
        HEADLINE_WINDOW: dates[(dates >= TRAIN_START) & (dates <= TEST_END)],
    }
    if len(windows[TRAIN_WINDOW]) + len(windows[TEST_WINDOW]) != len(
        windows[HEADLINE_WINDOW]
    ):
        raise AssertionError("selection and evaluation windows do not partition headline")

    daily = u2._read_daily()
    eligibility_all = u2._eligibility_for_dates(daily, dates)
    signal = u2._signal_inputs(daily, dates, eligibility_all)
    prices_all = u2._performance_prices(daily)
    sleeves = _broad_sleeves(eligibility_all.columns)

    required_cells = sorted(set(PRIMARY_CELLS.values()) | {TRANSFER_CELL})
    cluster_inputs = {}
    for frequency, span in required_cells:
        clusters, _ = u2._load_partition(frequency, span)
        scores = score_within_clusters(
            raw_signal=signal["raw_source"],
            rolling_clusters=u2._panel_dict(clusters),
            min_cluster_size=u2.SPEC.momentum_min_cluster_size,
        )
        cluster_inputs[(frequency, span)] = {
            "clusters": clusters,
            "scores": scores,
        }

    performance_rows = []
    acceptance_rows = []
    exposure_rows = []
    runtime_rows = []
    for window, window_dates in windows.items():
        window_started = time.perf_counter()
        eligibility = eligibility_all.reindex(index=window_dates)
        sleeve_panel = _sleeve_panel(window_dates, sleeves)
        prices = u2._window_prices(prices_all, window_dates)
        ew_nav = u2._ew_reference(
            prices_all, eligibility_all, window_dates, window
        )
        global_scores = signal["global"].reindex(
            index=window_dates, columns=eligibility.columns
        ).where(eligibility)

        for strategy in ("long_only", "long_short"):
            original_weights, diagnostics = _original_global_weights(
                strategy, global_scores, eligibility
            )
            performance, acceptance, exposures = _run_leg(
                window=window,
                strategy=strategy,
                method="original_global",
                weight_id="UNCONSTRAINED",
                target=None,
                frequency="BENCHMARK_INVARIANT",
                span=np.nan,
                prices=prices,
                weights=original_weights,
                diagnostics=diagnostics,
                sleeve_panel=sleeve_panel,
                ew_nav=ew_nav,
            )
            performance_rows.append(performance)
            acceptance_rows.append(acceptance)
            exposure_rows.extend(exposures)

            primary_cell = PRIMARY_CELLS[strategy]
            method_specs = (
                ("sleeve_global", None),
                ("sleeve_cluster_primary", primary_cell),
                ("sleeve_cluster_transfer_ME36", TRANSFER_CELL),
            )
            for _, weight_row in grid.iterrows():
                target = _target_weights(weight_row)
                weight_id = str(weight_row["weight_id"])
                for method, cell in method_specs:
                    if cell is None:
                        scores = global_scores
                        subgroups = sleeve_panel
                        frequency = "BROAD_SLEEVE"
                        span = np.nan
                    else:
                        item = cluster_inputs[cell]
                        clusters = item["clusters"].reindex(
                            index=window_dates, columns=eligibility.columns
                        )
                        scores = item["scores"].reindex(
                            index=window_dates, columns=eligibility.columns
                        ).where(eligibility)
                        subgroups = _hierarchical_groups(clusters, sleeve_panel)
                        frequency, span = cell
                    if strategy == "long_only":
                        weights, diagnostics = _long_only_weights(
                            scores,
                            eligibility,
                            sleeve_panel,
                            subgroups,
                            target,
                        )
                    else:
                        weights, diagnostics = _long_short_weights(
                            scores,
                            eligibility,
                            sleeve_panel,
                            subgroups,
                            target,
                        )
                    performance, acceptance, exposures = _run_leg(
                        window=window,
                        strategy=strategy,
                        method=method,
                        weight_id=weight_id,
                        target=target,
                        frequency=frequency,
                        span=span,
                        prices=prices,
                        weights=weights,
                        diagnostics=diagnostics,
                        sleeve_panel=sleeve_panel,
                        ew_nav=ew_nav,
                    )
                    performance_rows.append(performance)
                    acceptance_rows.append(acceptance)
                    exposure_rows.extend(exposures)
        runtime_rows.append(
            {
                "analysis_window": window,
                "dates": len(window_dates),
                "runtime_seconds": time.perf_counter() - window_started,
            }
        )
        print(f"BlackRock broad-sleeve {window}: complete", flush=True)

    performance = pd.DataFrame(performance_rows).sort_values(
        ["analysis_window", "strategy", "method", "weight_id"]
    ).reset_index(drop=True)
    acceptance = pd.DataFrame(acceptance_rows)
    if not acceptance["status"].eq("PASS").all():
        raise AssertionError(acceptance.loc[~acceptance["status"].eq("PASS")])
    comparison = _comparison(performance)
    selection = _selection_table(comparison)
    selected = _selected_evaluation(comparison, selection)
    rank_stability = _rank_stability(comparison)
    regression = _global_regression(performance)
    runtime = pd.DataFrame(runtime_rows)
    runtime["total_run_seconds"] = time.perf_counter() - started
    output = {
        "weight_grid": grid,
        "performance": performance,
        "comparison": comparison,
        "selection": selection,
        "selected_evaluation": selected,
        "rank_stability": rank_stability,
        "allocation_diagnostics": pd.DataFrame(exposure_rows),
        "acceptance": acceptance,
        "regression": regression,
        "runtime": runtime,
    }
    for name, frame in output.items():
        e5._write(frame, _root() / f"{name}.csv")
    return output


def _hash_outputs() -> dict[str, str]:
    """Hash every numerical artifact except timing and replay records."""
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(_root().glob("*.csv"))
        if path.name not in {"runtime.csv", "determinism.csv"}
    }


def verify_determinism() -> pd.DataFrame:
    """Replay the complete sleeve grid and require byte-identical artifacts."""
    run()
    first = _hash_outputs()
    run()
    second = _hash_outputs()
    names = sorted(set(first) | set(second))
    replay = pd.DataFrame(
        {
            "artifact": names,
            "first_sha256": [first.get(name) for name in names],
            "second_sha256": [second.get(name) for name in names],
            "byte_identical": [first.get(name) == second.get(name) for name in names],
        }
    )
    e5._write(replay, _root() / "determinism.csv")
    if not replay["byte_identical"].all():
        raise AssertionError(replay.loc[~replay["byte_identical"]])
    return replay


def main() -> None:
    """Run the deterministic broad-sleeve allocation experiment."""
    replay = verify_determinism()
    print(
        f"BlackRock U2 broad-sleeve grid: PASS "
        f"({len(replay)}/{len(replay)} deterministic)",
        flush=True,
    )


if __name__ == "__main__":
    main()
