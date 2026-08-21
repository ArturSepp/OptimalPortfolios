"""Grid U1 classic signal scoring over the cluster-size fallback threshold.

The same ``min_cluster_size`` is supplied to the public
``compute_classic_momentum_cluster_alpha`` path for M1-star clusters and for
point-in-time BICS sector groups.  The classic 12-minus-1 raw signal, q=25%
global portfolio rank, matched eligible universe, headline window, one-period
implementation lag, and 10 bp one-way cost remain fixed.
"""
from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

import papers.cluster_lineage_2026.replication.run_backtests as e5
import papers.cluster_lineage_2026.replication.run_u1_bics_sector_comparison_classic as core


MIN_CLUSTER_SIZE_GRID = (5, 10, 15, 20)
RUNNER = (
    "papers/cluster_lineage_2026/replication/"
    "run_u1_classic_min_cluster_size_grid.py"
)


def _root() -> Path:
    """Return the gitignored local grid-output directory."""
    root = (
        Path(__file__).resolve().parents[1]
        / "local_outputs"
        / "e5b"
        / "u1_classic_min_cluster_size_grid_20260816"
    )
    root.mkdir(parents=True, exist_ok=True)
    return root


def _weight_error(
    weights: pd.DataFrame,
    eligibility: pd.DataFrame,
    exposure: pd.DataFrame,
    rank_diagnostics: pd.DataFrame,
) -> Mapping[str, float]:
    """Return canonical eligibility, exposure, selection, and overlap errors."""
    outside = weights.where(~eligibility, 0.0).abs().to_numpy()
    return {
        "weight_outside_eligibility_abs_error": (
            float(outside.max()) if outside.size else 0.0
        ),
        "exposure_abs_error": max(
            float(exposure["long_exposure"].sub(1.0).abs().max()),
            float(exposure["short_exposure_abs"].sub(1.0).abs().max()),
            float(exposure["net_exposure"].abs().max()),
            float(exposure["gross_exposure"].sub(2.0).abs().max()),
        ),
        "selection_count_error": max(
            float(
                rank_diagnostics["long_assets"]
                .sub(rank_diagnostics["expected_assets_per_side"])
                .abs()
                .max()
            ),
            float(
                rank_diagnostics["short_assets"]
                .sub(rank_diagnostics["expected_assets_per_side"])
                .abs()
                .max()
            ),
        ),
        "overlap_assets": float(rank_diagnostics["overlap_assets"].max()),
    }


def _comparison(performance: pd.DataFrame) -> pd.DataFrame:
    """Compare every cluster-size cell with sector and global yardsticks."""
    rows = []
    metrics = (
        "net_return_annualized",
        "gross_return_annualized",
        "volatility_annualized",
        "sharpe_rf0",
        "one_way_turnover_annualized",
        "cost_drag_bp_per_year",
        "net_total_return",
    )
    for threshold, frame in performance.groupby("min_cluster_size", sort=True):
        indexed = frame.set_index("leg")
        cluster = indexed.loc["cluster_M1_star"]
        for benchmark_leg in ("bics_sector", "global"):
            benchmark = indexed.loc[benchmark_leg]
            row = {
                "min_cluster_size": int(threshold),
                "cluster_leg": "cluster_M1_star",
                "benchmark_leg": benchmark_leg,
            }
            for metric in metrics:
                row[f"cluster_{metric}"] = cluster[metric]
                row[f"benchmark_{metric}"] = benchmark[metric]
                row[f"delta_{metric}"] = cluster[metric] - benchmark[metric]
            row["beats_benchmark_net_return"] = (
                row["delta_net_return_annualized"] > 0.0
            )
            row["beats_benchmark_sharpe"] = row["delta_sharpe_rf0"] > 0.0
            rows.append(row)
    return pd.DataFrame(rows)


def run() -> Mapping[str, pd.DataFrame]:
    """Execute and validate the four-cell U1 fallback grid."""
    started = time.perf_counter()
    dates, fixed_eligibility = core._accepted_dates_and_eligibility()
    window_dates = e5._analysis_windows(core.UNIVERSE, dates)[core.WINDOW]
    data = e5.load_universe(core.UNIVERSE)
    columns = fixed_eligibility.columns
    all_eligibility = fixed_eligibility.reindex(index=window_dates).astype(bool)
    bics = data.taxonomy[core.SECTOR_COLUMN].reindex(columns).replace("", np.nan)
    primary_eligibility = all_eligibility & bics.notna().to_numpy()
    sector_membership, membership_diagnostics = core._point_in_time_sector_membership(
        bics, fixed_eligibility
    )
    membership_diagnostics = membership_diagnostics.loc[
        membership_diagnostics["date"].isin(window_dates)
    ].reset_index(drop=True)
    if membership_diagnostics[
        [
            "assignments_outside_index",
            "missing_assignments_for_classified_members",
        ]
    ].to_numpy().max() != 0:
        raise AssertionError("point-in-time BICS membership validation failed")

    daily_returns = core._read_daily(columns)
    monthly_returns = core._native_returns(daily_returns, core.SIGNAL_FREQUENCY)
    cluster_groups = e5._cluster_groups(
        core.UNIVERSE, core.CLUSTER_CONFIG
    ).reindex(columns=columns)
    prices = e5._prices(data).reindex(columns=columns)
    ew_nav = core._ew_navs()[core.WINDOW]
    costs = core.COST_BPS / 10000.0

    performance_rows = []
    acceptance_rows = []
    signal_rows = []
    cached_global_payload = None
    for threshold in MIN_CLUSTER_SIZE_GRID:
        global_all, cluster_all, sector_all, _, signal_diagnostics = (
            core._classic_signal_panels(
                monthly_returns,
                dates,
                fixed_eligibility,
                cluster_groups,
                sector_membership,
                min_cluster_size=threshold,
            )
        )
        signal_rows.append(signal_diagnostics.assign(min_cluster_size=threshold))
        score_panels = {
            "cluster_M1_star": cluster_all,
            "bics_sector": sector_all,
            "global": global_all,
        }
        for leg, full_scores in score_panels.items():
            scores = full_scores.reindex(index=window_dates, columns=columns).where(
                primary_eligibility
            )
            weights, exposure, rank_diagnostics = core._canonical_long_short_weights(
                scores, prices, primary_eligibility
            )
            errors = _weight_error(
                weights, primary_eligibility, exposure, rank_diagnostics
            )
            maximum_error = max(errors.values())
            acceptance_rows.append(
                {
                    "min_cluster_size": threshold,
                    "leg": leg,
                    **errors,
                    "tolerance": core.WEIGHT_TOLERANCE,
                    "status": (
                        "PASS"
                        if maximum_error <= core.WEIGHT_TOLERANCE
                        else "FAIL"
                    ),
                }
            )
            if leg == "global" and cached_global_payload is not None:
                payload = cached_global_payload
            else:
                net, gross = core._backtest(
                    prices,
                    weights,
                    costs,
                    f"u1_classic_min_cluster_{threshold}_{leg}",
                )
                payload = core.grid_ls._performance_payload(net, gross, ew_nav)
                if leg == "global":
                    cached_global_payload = payload
            performance_rows.append(
                {
                    "universe": core.UNIVERSE.value,
                    "analysis_window": core.WINDOW,
                    "leg": leg,
                    "min_cluster_size": threshold,
                    "q": core.Q,
                    "strategy": "long_top_short_bottom",
                    "signal_variant": core.SIGNAL_VARIANT,
                    "construction": "canonical_op_global_rank_equal_weight",
                    **payload,
                    "runner": RUNNER,
                }
            )

    performance = pd.DataFrame(performance_rows)
    acceptance = pd.DataFrame(acceptance_rows)
    signal_diagnostics = pd.concat(signal_rows, ignore_index=True)
    if not acceptance["status"].eq("PASS").all():
        raise AssertionError(acceptance.loc[~acceptance["status"].eq("PASS")])
    if signal_diagnostics[
        [
            "max_global_lookahead_days",
            "max_cluster_lookahead_days",
            "max_sector_lookahead_days",
            "max_raw_lookahead_days",
            "classic_raw_global_cluster_max_abs_error",
            "classic_raw_global_sector_max_abs_error",
        ]
    ].to_numpy().max() != 0.0:
        raise AssertionError("signal identity or no-look-ahead validation failed")

    output = {
        "performance": performance,
        "comparison": _comparison(performance),
        "acceptance": acceptance,
        "signal_diagnostics": signal_diagnostics,
        "sector_membership_diagnostics": membership_diagnostics,
        "design": pd.DataFrame(
            [
                {
                    **core.FROZEN_SPEC,
                    "cluster_fallback": "grid",
                    "min_cluster_size_grid": "|".join(
                        map(str, MIN_CLUSTER_SIZE_GRID)
                    ),
                    "threshold_application": (
                        "same public cluster-alpha fallback for M1-star and "
                        "point-in-time BICS groups"
                    ),
                    "portfolio_rank_rule": (
                        "one global canonical OP top/bottom quantile rank"
                    ),
                    "runner": RUNNER,
                }
            ]
        ),
        "runtime": pd.DataFrame(
            [{"runtime_seconds": time.perf_counter() - started, "runner": RUNNER}]
        ),
    }
    for name, frame in output.items():
        e5._write(frame, _root() / f"{name}.csv")
    return output


def _hash_outputs() -> dict[str, str]:
    """Hash deterministic grid outputs while excluding runtime and replay."""
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(_root().glob("*.csv"))
        if path.name not in {"runtime.csv", "determinism.csv"}
    }


def verify_determinism() -> pd.DataFrame:
    """Replay an existing complete grid once and require byte-identical artifacts."""
    required = {
        "acceptance.csv",
        "comparison.csv",
        "design.csv",
        "performance.csv",
        "sector_membership_diagnostics.csv",
        "signal_diagnostics.csv",
    }
    if not required.issubset({path.name for path in _root().glob("*.csv")}):
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
    """Run, replay, and print the U1 fallback grid."""
    replay = verify_determinism()
    performance = pd.read_csv(
        _root() / "performance.csv", float_precision="round_trip"
    )
    columns = [
        "min_cluster_size",
        "leg",
        "net_return_annualized",
        "volatility_annualized",
        "sharpe_rf0",
        "one_way_turnover_annualized",
    ]
    print(performance[columns].to_string(index=False))
    print(
        f"U1 classic min-cluster-size grid: PASS "
        f"({len(replay)}/{len(replay)} deterministic)",
        flush=True,
    )


if __name__ == "__main__":
    main()
