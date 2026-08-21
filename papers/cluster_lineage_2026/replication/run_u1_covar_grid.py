"""Run the U1 cluster payoff grid over native return cadences and EWMA spans.

The sensitivity changes only the EWMA asset covariance/correlation matrix used to form
the unsmoothed baseline partition.  The point-in-time U1 investable universe, ME decision
schedule, momentum score, selection fractions, group-equal construction, costs, and lag
remain frozen.  Global rank is the sole payoff benchmark; EW-all is consumed only as the
market reference for alpha and beta columns.

The grid contains 28 cells: B and W-MON through W-FRI use spans 24, 36, 52, and 156;
ME uses spans 12, 24, 36, and 52.  Spans are measured in observations at the stated
native cadence.
"""
from __future__ import annotations

import hashlib
import importlib
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from factorlasso import compute_clusters_from_corr_matrix
from factorlasso.cluster_smoothing import _iter_correlation_inputs

import papers.cluster_lineage_2026.replication.run_backtests as e5
from papers.cluster_lineage_2026.replication.recovery_loader import install as install_pyc_finder
from papers.cluster_lineage_2026.replication.run_e5b import (
    GROUP_BUDGET_TOLERANCE,
    WEIGHT_TOLERANCE,
    _group_equal_from_ranks,
    _root as e5b_root,
)
from papers.cluster_lineage_2026.replication.run_u1_global_grid import (
    _accepted_global_rows,
    _backtest,
    _ew_navs,
)
from papers.cluster_lineage_2026.replication.run_u1_quantile_sweep import QUANTILES


install_pyc_finder()
estimate = importlib.import_module("papers.cluster_lineage_2026.replication.estimate")

UNIVERSE = e5.UniverseName.MSCI_US
RUNNER = "papers/cluster_lineage_2026/replication/run_u1_covar_grid.py"
FREQUENCY_SPANS: Mapping[str, tuple[int, ...]] = {
    "B": (24, 36, 52, 156),
    "W-MON": (24, 36, 52, 156),
    "W-TUE": (24, 36, 52, 156),
    "W-WED": (24, 36, 52, 156),
    "W-THU": (24, 36, 52, 156),
    "W-FRI": (24, 36, 52, 156),
    "ME": (12, 24, 36, 52),
}
METRICS = (
    "net_return_annualized",
    "sharpe_rf0",
    "one_way_turnover_annualized",
    "cost_drag_bp_per_year",
)
PARTITION_CACHE_VERSION = 1
DATA_DIR = Path(__file__).resolve().parents[1] / "data"


def _root() -> Path:
    """Return and create the local covariance-grid output directory."""
    root = e5b_root() / "covariance_frequency_span_grid" / UNIVERSE.value
    root.mkdir(parents=True, exist_ok=True)
    return root


def _partition_root() -> Path:
    """Return and create the compact partition-cache directory."""
    root = _root() / "partitions"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _cell_id(frequency: str, span: int) -> str:
    """Return a filesystem-safe stable identifier for one grid cell."""
    return f"{frequency.replace('-', '_')}_span_{span:03d}"


def _cells() -> tuple[tuple[str, int], ...]:
    """Return the frozen 28-cell frequency/span grid."""
    return tuple(
        (frequency, span)
        for frequency, spans in FREQUENCY_SPANS.items()
        for span in spans
    )


def _partition_path(frequency: str, span: int) -> Path:
    """Return the compact cache path for one partition panel."""
    return _partition_root() / f"{_cell_id(frequency, span)}.pkl"


def _read_daily(columns: pd.Index) -> pd.DataFrame:
    """Read the frozen daily U1 excess-log-return panel on accepted columns."""
    daily = pd.read_csv(
        DATA_DIR / "msci_us_log_returns.csv",
        index_col=0,
        parse_dates=True,
        float_precision="round_trip",
    )
    daily.index = pd.DatetimeIndex(daily.index, name="date")
    return daily.reindex(columns=columns)


def _native_returns(daily: pd.DataFrame, frequency: str) -> pd.DataFrame:
    """Aggregate daily log returns to one native covariance cadence."""
    if frequency == "B":
        return daily.copy()
    return daily.resample(frequency).sum(min_count=1)


def _accepted_dates_and_eligibility() -> tuple[pd.DatetimeIndex, pd.DataFrame]:
    """Return the accepted fixed U1 schedule and point-in-time investable mask."""
    dates = e5.load_cached(UNIVERSE, e5.SmootherName.BASELINE).dates
    data = e5.load_universe(UNIVERSE)
    eligibility = e5._investable_eligibility(data, dates)
    return pd.DatetimeIndex(dates), eligibility


def preflight() -> pd.DataFrame:
    """Prove the canonical weekly return panel and fixed universe are unchanged."""
    reconstruction_tolerance = 1e-15
    data = e5.load_universe(UNIVERSE)
    accepted = data.asset_returns["W-WED"]
    daily = _read_daily(accepted.columns)
    rebuilt = _native_returns(daily, "W-WED").reindex_like(accepted)
    delta = rebuilt.subtract(accepted).abs().to_numpy()
    finite = delta[np.isfinite(delta)]
    max_error = float(finite.max()) if finite.size else 0.0
    nan_match = bool(rebuilt.isna().equals(accepted.isna()))
    dates, eligibility = _accepted_dates_and_eligibility()
    same_eligibility = bool(
        eligibility.equals(data.eligibility["W-WED"].reindex(index=dates))
    )
    frame = pd.DataFrame(
        [
            {
                "check": "W-WED return reconstruction",
                "measured": max_error,
                "tolerance": reconstruction_tolerance,
                "status": "PASS"
                if max_error <= reconstruction_tolerance and nan_match
                else "FAIL",
            },
            {
                "check": "fixed accepted eligibility",
                "measured": int(same_eligibility),
                "tolerance": 1,
                "status": "PASS" if same_eligibility else "FAIL",
            },
            {
                "check": "grid cells",
                "measured": len(_cells()),
                "tolerance": 28,
                "status": "PASS" if len(_cells()) == 28 else "FAIL",
            },
        ]
    )
    if not frame["status"].eq("PASS").all():
        raise AssertionError(frame.loc[~frame["status"].eq("PASS")])
    return frame


def _partition_hash(panel: pd.DataFrame) -> str:
    """Hash one membership panel independent of pickle implementation details."""
    values = pd.util.hash_pandas_object(panel, index=True).to_numpy().tobytes()
    columns = "\x1f".join(map(str, panel.columns)).encode("utf-8")
    return hashlib.sha256(values + columns).hexdigest()


def _canonical_labels(labels: pd.Series) -> np.ndarray:
    """Return order-of-first-appearance labels for label-invariant comparison."""
    return pd.factorize(labels, sort=False)[0]


def _same_partition(left: pd.Series, right: pd.Series) -> bool:
    """Return whether two complete assignments encode the same partition."""
    left = left.dropna()
    right = right.dropna()
    if not left.index.equals(right.index):
        return False
    return bool(np.array_equal(_canonical_labels(left), _canonical_labels(right)))


def _model(span: int, frequency: str):
    """Copy the frozen baseline clustering model with only native span changed."""
    base = estimate.make_estimator(UNIVERSE, e5.SmootherName.BASELINE).lasso_model
    return base.copy(
        kwargs={"span": span, "span_freq_dict": {frequency: span}}
    )


def _compute_partition_cell(
    frequency: str, span: int, *, force: bool = False
) -> Mapping[str, object]:
    """Compute or load one point-in-time baseline partition panel."""
    path = _partition_path(frequency, span)
    if path.exists() and not force:
        with path.open("rb") as stream:
            cached = pickle.load(stream)
        expected = {
            "version": PARTITION_CACHE_VERSION,
            "frequency": frequency,
            "span": span,
        }
        if all(cached.get(key) == value for key, value in expected.items()):
            return {
                "frequency": frequency,
                "span": span,
                "cell_id": _cell_id(frequency, span),
                "cache_status": "hit",
                "partition_hash": _partition_hash(cached["panel"]),
                "dates": len(cached["panel"]),
                "runtime_seconds": 0.0,
            }

    started = time.perf_counter()
    dates, eligibility = _accepted_dates_and_eligibility()
    columns = eligibility.columns
    daily = _read_daily(columns)
    returns = _native_returns(daily, frequency)
    model = _model(span, frequency)
    panel = pd.DataFrame(np.nan, index=dates, columns=columns)
    diagnostic_rows: list[dict] = []

    # factorlasso.compute_rolling_smoothed_clusters clusters every listed column before
    # U1's point-in-time restriction.  That public path is therefore unsuitable here;
    # reuse FactorLasso's exact correlation iterator, then apply the accepted asset mask
    # before calling its public compute_clusters_from_corr_matrix implementation.
    iterator = _iter_correlation_inputs(returns, list(dates), model)
    for date, full_corr in iterator:
        assets = eligibility.columns[eligibility.loc[date].fillna(False).astype(bool)]
        corr = full_corr.reindex(index=assets, columns=assets)
        if len(assets) == 0:
            labels = pd.Series(dtype=float)
        elif len(assets) == 1:
            labels = pd.Series(1, index=assets)
        else:
            labels, _, _ = compute_clusters_from_corr_matrix(
                corr,
                cutoff_fraction=model.cutoff_fraction,
                linkage_method=model.linkage_method,
                distance_transform=model.distance_transform,
                n_clusters=model.n_clusters,
            )
        panel.loc[date, labels.index] = labels.to_numpy()
        diagnostic_rows.append(
            {
                "date": date,
                "members": len(labels),
                "clusters": int(labels.nunique()),
            }
        )

    payload = {
        "version": PARTITION_CACHE_VERSION,
        "frequency": frequency,
        "span": span,
        "panel": panel,
        "diagnostics": pd.DataFrame(diagnostic_rows),
    }
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)
    return {
        "frequency": frequency,
        "span": span,
        "cell_id": _cell_id(frequency, span),
        "cache_status": "miss",
        "partition_hash": _partition_hash(panel),
        "dates": len(panel),
        "runtime_seconds": time.perf_counter() - started,
    }


def build_partitions(
    *, max_workers: int = 4, force: bool = False
) -> pd.DataFrame:
    """Build all compact partition caches in parallel and report completion."""
    rows: list[Mapping[str, object]] = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        pending = {
            executor.submit(_compute_partition_cell, frequency, span, force=force): (
                frequency,
                span,
            )
            for frequency, span in _cells()
        }
        for future in as_completed(pending):
            frequency, span = pending[future]
            row = future.result()
            rows.append(row)
            print(
                f"partition {frequency}/{span}: {row['cache_status']} "
                f"({float(row['runtime_seconds']):.1f}s)",
                flush=True,
            )
    return pd.DataFrame(rows).sort_values(["frequency", "span"]).reset_index(drop=True)


def _load_partition(frequency: str, span: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load one verified compact membership cache."""
    with _partition_path(frequency, span).open("rb") as stream:
        cached = pickle.load(stream)
    if (
        cached.get("version") != PARTITION_CACHE_VERSION
        or cached.get("frequency") != frequency
        or cached.get("span") != span
    ):
        raise AssertionError(f"invalid partition cache for {frequency}/{span}")
    return cached["panel"], cached["diagnostics"]


def _partition_diagnostics() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-date and per-cell cluster-count diagnostics."""
    rows = []
    for frequency, span in _cells():
        panel, diagnostics = _load_partition(frequency, span)
        frame = diagnostics.copy()
        frame.insert(0, "cell_id", _cell_id(frequency, span))
        frame.insert(0, "span", span)
        frame.insert(0, "frequency", frequency)
        frame["partition_hash"] = _partition_hash(panel)
        rows.append(frame)
    per_date = pd.concat(rows, ignore_index=True)
    summary = (
        per_date.groupby(["frequency", "span", "cell_id", "partition_hash"], sort=False)
        .agg(
            dates=("date", "size"),
            member_min=("members", "min"),
            member_median=("members", "median"),
            member_max=("members", "max"),
            cluster_mean=("clusters", "mean"),
            cluster_std=("clusters", "std"),
            cluster_min=("clusters", "min"),
            cluster_max=("clusters", "max"),
        )
        .reset_index()
    )
    return per_date, summary


def _partition_regression() -> pd.DataFrame:
    """Compare canonical W-WED/156 partitions with every accepted baseline date."""
    panel, _ = _load_partition("W-WED", 156)
    accepted = e5._cluster_groups(UNIVERSE, e5.SmootherName.BASELINE).reindex_like(panel)
    rows = []
    mismatches = []
    for date in panel.index:
        left = panel.loc[date].dropna()
        right = accepted.loc[date].dropna().reindex(left.index)
        matches = _same_partition(left, right)
        mismatches.extend([] if matches else [date.strftime("%Y-%m-%d")])
        rows.append(
            {
                "date": date,
                "members_new": len(left),
                "members_accepted": int(accepted.loc[date].notna().sum()),
                "partition_match": matches,
            }
        )
    details = pd.DataFrame(rows)
    summary = pd.DataFrame(
        [
            {
                "check": "W-WED span 156 partition regression",
                "measured": float(details["partition_match"].mean()),
                "tolerance": 1.0,
                "mismatched_dates": "|".join(mismatches),
                "status": "PASS" if not mismatches else "FAIL",
            }
        ]
    )
    if mismatches:
        raise AssertionError(summary)
    return summary


def _comparison(performance: pd.DataFrame) -> pd.DataFrame:
    """Add cell-minus-global metrics against matched-q and best global rows."""
    global_rows = performance.loc[performance["leg"].eq("global")]
    cluster_rows = performance.loc[performance["leg"].ne("global")]
    rows = []
    for _, cluster in cluster_rows.iterrows():
        window = cluster["analysis_window"]
        q = cluster["q"]
        same_q = global_rows.loc[
            global_rows["analysis_window"].eq(window) & global_rows["q"].eq(q)
        ].iloc[0]
        best_global = global_rows.loc[
            global_rows["analysis_window"].eq(window)
        ].sort_values("sharpe_rf0", ascending=False).iloc[0]
        row = cluster.to_dict()
        for metric in METRICS:
            row[f"{metric}_delta_vs_same_q_global"] = cluster[metric] - same_q[metric]
            row[f"{metric}_delta_vs_best_global"] = cluster[metric] - best_global[metric]
        row["best_global_q"] = best_global["q"]
        row["beats_same_q_global_both"] = bool(
            row["net_return_annualized_delta_vs_same_q_global"] > 0.0
            and row["sharpe_rf0_delta_vs_same_q_global"] > 0.0
        )
        row["beats_best_global_both"] = bool(
            row["net_return_annualized_delta_vs_best_global"] > 0.0
            and row["sharpe_rf0_delta_vs_best_global"] > 0.0
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _cell_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    """Select each covariance cell's best q by Sharpe in each analysis window."""
    rows = []
    for (window, frequency, span), panel in comparison.groupby(
        ["analysis_window", "frequency", "span"], sort=False
    ):
        best = panel.sort_values(
            ["sharpe_rf0", "net_return_annualized", "one_way_turnover_annualized"],
            ascending=[False, False, True],
        ).iloc[0]
        rows.append(best.to_dict())
    return pd.DataFrame(rows)


def _rankings(comparison: pd.DataFrame) -> pd.DataFrame:
    """Rank every frequency/span/q row within each analysis window."""
    rows = []
    for window, panel in comparison.groupby("analysis_window", sort=False):
        ranked = panel.sort_values(
            ["sharpe_rf0", "net_return_annualized", "one_way_turnover_annualized"],
            ascending=[False, False, True],
        ).reset_index(drop=True)
        ranked.insert(0, "rank", np.arange(1, len(ranked) + 1))
        rows.append(ranked)
    return pd.concat(rows, ignore_index=True)


def _frequency_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    """Select the best span/q combination within every native frequency."""
    rows = []
    for (window, frequency), panel in comparison.groupby(
        ["analysis_window", "frequency"], sort=False
    ):
        best = panel.sort_values(
            ["sharpe_rf0", "net_return_annualized", "one_way_turnover_annualized"],
            ascending=[False, False, True],
        ).iloc[0]
        rows.append(best.to_dict())
    return pd.DataFrame(rows)


def _payoff_regression(performance: pd.DataFrame) -> pd.DataFrame:
    """Match the canonical grid cell to the previously accepted baseline payoff."""
    accepted = pd.read_csv(
        e5b_root() / "global_benchmark_grid" / UNIVERSE.value / "performance.csv",
        float_precision="round_trip",
    )
    accepted = accepted.loc[accepted["leg"].eq("cluster_baseline")].copy()
    current = performance.loc[
        performance["frequency"].eq("W-WED")
        & performance["span"].eq(156)
    ].copy()
    joined = current.merge(
        accepted,
        on=["analysis_window", "q"],
        suffixes=("_new", "_accepted"),
        validate="one_to_one",
    )
    errors = {
        metric: float(
            (joined[f"{metric}_new"] - joined[f"{metric}_accepted"]).abs().max()
        )
        for metric in METRICS
    }
    max_error = max(errors.values())
    frame = pd.DataFrame(
        [
            {
                "check": "W-WED span 156 payoff regression",
                "measured": max_error,
                "tolerance": 1e-12,
                "metric_errors": "|".join(
                    f"{metric}={error:.3e}" for metric, error in errors.items()
                ),
                "status": "PASS" if max_error <= 1e-12 else "FAIL",
            }
        ]
    )
    if max_error > 1e-12:
        raise AssertionError(frame)
    return frame


def run(*, max_workers: int = 4) -> Mapping[str, pd.DataFrame]:
    """Execute the complete cached partition and payoff sensitivity grid."""
    started = time.perf_counter()
    preflight_frame = preflight()
    partition_runtime = build_partitions(max_workers=max_workers)
    partition_regression = _partition_regression()
    per_date_partitions, partition_summary = _partition_diagnostics()

    dates, fixed_eligibility = _accepted_dates_and_eligibility()
    windows = e5._analysis_windows(UNIVERSE, dates)
    data = e5.load_universe(UNIVERSE)
    prices = e5._prices(data)
    costs = e5.get_universe_spec(UNIVERSE).cost_bps / 10000.0
    ew_navs = _ew_navs()
    performance_rows = []
    diagnostic_rows = []
    acceptance_rows = []

    for frequency, span in _cells():
        groups_all, _ = _load_partition(frequency, span)
        cell_id = _cell_id(frequency, span)
        for window, window_dates in windows.items():
            eligibility = fixed_eligibility.reindex(index=window_dates)
            groups = groups_all.reindex(index=window_dates, columns=eligibility.columns)
            scores = e5._raw_momentum_scores(
                data, window_dates, vol_adjusted=False
            ).reindex(columns=eligibility.columns).where(eligibility)
            prices_window = prices.reindex(columns=eligibility.columns)
            ranks = e5._rank_panel(scores, groups)
            for q in QUANTILES:
                weights, counts, validation = _group_equal_from_ranks(
                    ranks, eligibility, groups, q, UNIVERSE
                )
                leg = f"cluster_{cell_id}"
                net, gross = _backtest(
                    prices_window,
                    weights,
                    costs,
                    f"{window}_{cell_id}_q_{q:.2f}",
                )
                performance_rows.append(
                    {
                        "universe": UNIVERSE.value,
                        "analysis_window": window,
                        "frequency": frequency,
                        "span": span,
                        "span_unit": "native_observations",
                        "q": q,
                        "construction": "group_equal",
                        "leg": leg,
                        "cell_id": cell_id,
                        **e5._performance_row(net, gross, ew_navs[window]),
                        "runner": RUNNER,
                    }
                )
                max_weight_error = float(weights.sum(axis=1).sub(1.0).abs().max())
                max_budget_error = float(
                    validation["max_group_budget_abs_error"].max()
                )
                selected = weights.gt(0.0).sum(axis=1)
                diagnostic_rows.append(
                    {
                        "analysis_window": window,
                        "frequency": frequency,
                        "span": span,
                        "q": q,
                        "cell_id": cell_id,
                        "mean_available_groups": float(counts.mean()),
                        "available_group_count_std": float(counts.std()),
                        "mean_selected_assets": float(selected.mean()),
                        "min_selected_assets": int(selected.min()),
                        "max_selected_assets": int(selected.max()),
                        "max_weight_sum_abs_error": max_weight_error,
                        "max_group_budget_abs_error": max_budget_error,
                    }
                )
                status = (
                    "PASS"
                    if max_weight_error <= WEIGHT_TOLERANCE
                    and max_budget_error <= GROUP_BUDGET_TOLERANCE
                    else "FAIL"
                )
                acceptance_rows.append(
                    {
                        "analysis_window": window,
                        "frequency": frequency,
                        "span": span,
                        "q": q,
                        "cell_id": cell_id,
                        "weight_sum_error": max_weight_error,
                        "weight_sum_tolerance": WEIGHT_TOLERANCE,
                        "group_budget_error": max_budget_error,
                        "group_budget_tolerance": GROUP_BUDGET_TOLERANCE,
                        "status": status,
                    }
                )
        print(f"backtest {frequency}/{span}: complete", flush=True)

    cluster_performance = pd.DataFrame(performance_rows)
    globals_frame = _accepted_global_rows().copy()
    globals_frame["frequency"] = "BENCHMARK_INVARIANT"
    globals_frame["span"] = np.nan
    globals_frame["span_unit"] = "not_applicable"
    globals_frame["cell_id"] = "global"
    globals_frame["runner"] = RUNNER
    performance = pd.concat(
        [globals_frame, cluster_performance], ignore_index=True, sort=False
    ).sort_values(
        ["analysis_window", "q", "frequency", "span"],
        ascending=[True, False, True, True],
    )
    comparison = _comparison(performance)
    cell_summary = _cell_summary(comparison)
    rankings = _rankings(comparison)
    frequency_summary = _frequency_summary(comparison)
    acceptance = pd.DataFrame(acceptance_rows)
    if not acceptance["status"].eq("PASS").all():
        raise AssertionError(acceptance.loc[~acceptance["status"].eq("PASS")])
    payoff_regression = _payoff_regression(performance)
    regression = pd.concat(
        [preflight_frame, partition_regression, payoff_regression],
        ignore_index=True,
        sort=False,
    )
    runtime = partition_runtime.copy()
    runtime["total_run_seconds"] = time.perf_counter() - started
    output = {
        "performance": performance.reset_index(drop=True),
        "comparison_vs_global": comparison.reset_index(drop=True),
        "rankings": rankings.reset_index(drop=True),
        "cell_summary": cell_summary.reset_index(drop=True),
        "frequency_summary": frequency_summary.reset_index(drop=True),
        "construction_diagnostics": pd.DataFrame(diagnostic_rows),
        "partition_diagnostics": per_date_partitions,
        "partition_summary": partition_summary,
        "acceptance": acceptance,
        "regression": regression,
        "runtime": runtime,
    }
    for name, frame in output.items():
        e5._write(frame, _root() / f"{name}.csv")
    return output


def _hash_outputs() -> dict[str, str]:
    """Hash every numerical artifact except timing and replay records."""
    excluded = {"runtime.csv", "determinism.csv"}
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(_root().glob("*.csv"))
        if path.name not in excluded
    }


def verify_determinism(*, max_workers: int = 4) -> pd.DataFrame:
    """Replay cached backtests and require byte-identical numerical artifacts."""
    expected = {
        "acceptance.csv",
        "cell_summary.csv",
        "comparison_vs_global.csv",
        "construction_diagnostics.csv",
        "frequency_summary.csv",
        "partition_diagnostics.csv",
        "partition_summary.csv",
        "performance.csv",
        "rankings.csv",
        "regression.csv",
    }
    first = _hash_outputs()
    if set(first) != expected:
        run(max_workers=max_workers)
        first = _hash_outputs()
    run(max_workers=max_workers)
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
    """Run the complete deterministic grid from the command line."""
    replay = verify_determinism(max_workers=4)
    print(
        f"U1 covariance frequency/span grid: PASS "
        f"({len(replay)}/{len(replay)} deterministic)",
        flush=True,
    )


if __name__ == "__main__":
    main()
