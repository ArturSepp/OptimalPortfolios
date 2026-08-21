"""Two-tier evaluation runner for causal S&P 500 cluster smoothing."""
from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Dict, Mapping, Tuple

import numpy as np
import pandas as pd
import qis as qis
from factorlasso import compute_rolling_smoothed_clusters

from papers.cluster_lineage_2026.replication.methods import (
    PartitionBundle,
    baseline_partitions,
    coassociation_confidence,
    compute_correlation_inputs,
    delta_bonus_ward,
    quarterly_hold,
    similarity_smoothing,
)
from papers.cluster_lineage_2026.replication.sp500_baseline import (
    annualized_churn,
    ari_metrics,
    estimate_rolling,
    extract_partitions,
    get_estimation_dates,
    greedy_membership_panel,
    lineage_metrics,
    load_inputs,
    make_estimator,
    partition_equal,
    residual_diagonality,
    signal_rank_metrics,
)

RUNNER = "rosaa/research/cluster_smoothing/run_sweep.py"
WORKBOOK_NAME = "sp500_cluster_smoothing_sweep_20260811"
DELTA_GRID = (0.0, 0.05, 0.10, 0.20)
SMOOTHING_GRID = (0.3, 0.5, 0.7)
TIER2_CONFIGS = (
    "baseline",
    "M0_quarterly_hold",
    "M1_delta_0.05",
    "M2_lambda_0.7",
)


def _partitions(
    bundles: Mapping[pd.Timestamp, PartitionBundle],
) -> Dict[pd.Timestamp, pd.Series]:
    """Discard linkage metadata for partition-only scoring."""
    return {date: bundle[0] for date, bundle in bundles.items()}


def build_grid() -> Dict[str, Dict[pd.Timestamp, PartitionBundle]]:
    """Build the complete Tier-1 baseline and smoothing grid."""
    correlations = compute_correlation_inputs()
    baseline = baseline_partitions(correlations)
    grid = {"baseline": baseline, "M0_quarterly_hold": quarterly_hold(correlations, baseline)}
    grid.update({f"M1_delta_{delta:.2f}": delta_bonus_ward(correlations, delta)
                 for delta in DELTA_GRID})
    grid.update({f"M2_lambda_{value:.1f}": similarity_smoothing(correlations, value)
                 for value in SMOOTHING_GRID})
    return grid


def score_tier1(
    grid: Mapping[str, Mapping[pd.Timestamp, PartitionBundle]],
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    """Score raw churn, GICS ARI, and signal-rank stability for the full grid."""
    inputs = load_inputs()
    rows = []
    panels = {}
    for config, bundles in grid.items():
        partitions = _partitions(bundles)
        ari, _ = ari_metrics(partitions, inputs["metadata"])
        signal, signal_panel = signal_rank_metrics(
            partitions, inputs["asset_returns_dict"]["W-WED"]
        )
        panels[config] = signal_panel
        method, parameter = _config_fields(config)
        rows.append({
            "config": config,
            "method": method,
            "parameter": parameter,
            "raw_churn": annualized_churn(greedy_membership_panel(partitions)),
            "median_clusters": float(np.median([p.nunique() for p in partitions.values()])),
            **ari,
            **signal,
            "runner": RUNNER,
        })
    result = pd.DataFrame(rows).set_index("config")
    metric_columns = result.select_dtypes(include=[np.number]).columns
    assert not result[metric_columns].isna().any().any()
    assert set(result.index) == {
        "baseline", "M0_quarterly_hold",
        *(f"M1_delta_{value:.2f}" for value in DELTA_GRID),
        *(f"M2_lambda_{value:.1f}" for value in SMOOTHING_GRID),
    }
    return result, panels


def _config_fields(config: str) -> Tuple[str, float]:
    """Split a stable configuration label into method and numeric parameter."""
    if config == "baseline":
        return "baseline", 0.0
    if config == "M0_quarterly_hold":
        return "M0", 1.0
    if config.startswith("M1"):
        return "M1", float(config.rsplit("_", 1)[1])
    return "M2", float(config.rsplit("_", 1)[1])


def choose_shortlist(tier1: pd.DataFrame) -> Tuple[str, str]:
    """Select minimum churn subject to every GICS ARI staying within 0.03."""
    ari_columns = ["ari_sector", "ari_industry_group", "ari_industry"]
    admissible = tier1[ari_columns].sub(tier1.loc["baseline", ari_columns]).abs().le(0.03).all(axis=1)
    m1 = tier1[(tier1.method == "M1") & (tier1.parameter > 0) & admissible]["raw_churn"].idxmin()
    m2 = tier1[(tier1.method == "M2") & admissible]["raw_churn"].idxmin()
    return str(m1), str(m2)


def build_tier2_partitions(
    tier1_grid: Mapping[str, Mapping[pd.Timestamp, PartitionBundle]],
) -> Dict[str, Dict[pd.Timestamp, PartitionBundle]]:
    """Build and verify the four partitions through LassoModel configuration."""
    inputs = load_inputs()
    dates = list(get_estimation_dates())
    output = {}
    for config in TIER2_CONFIGS:
        model = make_estimator(config).lasso_model
        rolling = compute_rolling_smoothed_clusters(
            y=inputs["asset_returns_dict"]["W-WED"],
            estimation_dates=dates,
            lasso_model=model,
        )
        output[config] = {
            date: (
                rolling.clusters[date],
                rolling.linkages[date],
                rolling.cutoffs[date],
            )
            for date in dates
        }
        matches = 0
        mismatches = []
        for date in dates:
            scored = tier1_grid[config][date][0]
            injected = rolling.clusters[date].reindex(scored.index)
            is_equal = partition_equal(injected, scored)
            matches += is_equal
            if not is_equal:
                mismatches.append(str(date.date()))
        assert matches == len(dates), (
            f"{config}: factorlasso partitions match Tier 1 on only "
            f"{matches}/{len(dates)} dates; mismatches={mismatches}"
        )
        print(f"second pass partition identity: {config} {matches}/{len(dates)}")
    return output


def score_tier2(
    tier2_partitions: Mapping[str, Mapping[pd.Timestamp, PartitionBundle]],
) -> pd.DataFrame:
    """Run full FCGL fits and score every S1 metric for the four configurations."""
    inputs = load_inputs()
    rows = []
    for config in TIER2_CONFIGS:
        covar_data = estimate_rolling(
            config=config,
            injections=tier2_partitions[config],
            max_workers=4,
        )
        partitions = extract_partitions(covar_data)
        counts = pd.Series({date: part.nunique() for date, part in partitions.items()})
        lineage, _ = lineage_metrics(covar_data)
        ari, _ = ari_metrics(partitions, inputs["metadata"])
        signal, _ = signal_rank_metrics(
            partitions, inputs["asset_returns_dict"]["W-WED"]
        )
        diagonality, _ = residual_diagonality(covar_data)
        last = covar_data[covar_data.dates[-1]]
        method, parameter = _config_fields(config)
        rows.append({
            "config": config,
            "method": method,
            "parameter": parameter,
            "n_snapshots": float(len(covar_data)),
            "median_clusters": float(counts.median()),
            "min_clusters": float(counts.min()),
            "max_clusters": float(counts.max()),
            "raw_churn": annualized_churn(greedy_membership_panel(partitions)),
            **lineage,
            **ari,
            **signal,
            **diagonality,
            "mean_market_beta_last": float(last.y_betas["Market"].mean()),
            "median_r2_last": float(last.y_variances["r2"].median()),
            "market_variance_last": float(last.x_covar.loc["Market", "Market"]),
            "runner": RUNNER,
        })
    result = pd.DataFrame(rows).set_index("config")
    numeric = result.select_dtypes(include=[np.number])
    assert not numeric.isna().any().any()
    assert set(result.index) == set(TIER2_CONFIGS)
    baseline_lineage = result.loc["baseline", "lineage_churn_panel"]
    assert round(baseline_lineage, 4) == 3.2115
    print(f"second pass baseline lineage churn: {baseline_lineage:.4f} (expected 3.2115)")
    for config, row in result.iterrows():
        print(
            f"second pass churn: {config} raw={row['raw_churn']:.6f}; "
            f"lineage={row['lineage_churn_panel']:.6f}"
        )
    _assert_diagonality_guard(result)
    return result


def _assert_diagonality_guard(tier2: pd.DataFrame) -> None:
    """Require each smoothed residual-diagnostic mean to stay within five percent."""
    columns = [column for column in tier2 if column.startswith("diagonality_")]
    baseline = tier2.loc["baseline", columns].astype(float)
    for config in TIER2_CONFIGS[1:]:
        candidate = tier2.loc[config, columns].astype(float)
        denominator = baseline.abs().where(baseline.abs() > 1e-12, 1.0)
        relative = (candidate - baseline).abs() / denominator
        maximum = float(relative.max())
        assert maximum <= 0.05, (
            f"{config}: residual-diagonality guard failed; max relative change={maximum:.2%}"
        )
        print(f"diagonality guard: {config} max relative change={maximum:.2%} <= 5.00%")


def save_tier1_workbook(
    tier1: pd.DataFrame,
    grid: Mapping[str, Mapping[pd.Timestamp, PartitionBundle]],
    tier2: pd.DataFrame = None,
) -> Path:
    """Write the Tier-1 evidence and, when available, full Tier-2 results."""
    from rosaa import local_path as lp

    best_m1, best_m2 = choose_shortlist(tier1)
    winner = tier1.loc[[best_m1, best_m2]]["raw_churn"].idxmin()
    confidence = coassociation_confidence(_partitions(grid[winner]))
    confidence.insert(0, "runner", RUNNER)
    tables = {
        "tier1_grid": tier1.reset_index(),
        "tier2_status": pd.DataFrame({
            "status": ["COMPLETE" if tier2 is not None else "PENDING"],
            "runner": [RUNNER],
        }),
        "m3_confidence": confidence,
    }
    if tier2 is not None:
        tables["tier2_grid"] = tier2.reset_index()
    qis.save_df_to_excel(
        tables,
        file_name=WORKBOOK_NAME,
        local_path=lp.get_output_path(),
        add_current_date=False,
    )
    path = Path(lp.get_output_path()) / f"{WORKBOOK_NAME}.xlsx"
    print(f"cluster-smoothing workbook written: {path}")
    print(f"shortlist: M1={best_m1}; M2={best_m2}; M3 panel={winner}")
    return path


def run_tier1() -> pd.DataFrame:
    """Run and persist the complete partition-level sweep."""
    grid = build_grid()
    tier1, _ = score_tier1(grid)
    save_tier1_workbook(tier1, grid)
    print(tier1.to_string())
    return tier1


def run_tier2() -> pd.DataFrame:
    """Execute the four-config full-refit sweep through declarative model fields."""
    grid = build_grid()
    tier1, _ = score_tier1(grid)
    tier2_partitions = build_tier2_partitions(grid)
    tier2 = score_tier2(tier2_partitions)
    save_tier1_workbook(tier1, grid, tier2=tier2)
    print(tier2.to_string())
    return tier2


class ResearchWorkflow(Enum):
    """Runnable sweep stages."""

    TIER1 = 1
    TIER2 = 2
    ALL = 3


@qis.timer
def run_research(workflow: ResearchWorkflow = ResearchWorkflow.ALL) -> None:
    """Dispatch the partition screen and full-refit Tier-2 workflow."""
    if workflow in (ResearchWorkflow.TIER1, ResearchWorkflow.ALL):
        run_tier1()
    if workflow in (ResearchWorkflow.TIER2, ResearchWorkflow.ALL):
        run_tier2()


if __name__ == "__main__":
    run_research(ResearchWorkflow.ALL)
