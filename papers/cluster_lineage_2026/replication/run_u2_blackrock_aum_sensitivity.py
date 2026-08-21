"""Run AUM-cutoff sensitivity for the fixed selected BlackRock hybrid model.

The model is not reselected: ROSAA production momentum, W-THU/span-156 clusters,
q=25%, Equity/Fixed Income/Rest budgets 50/30/20, group-equal cluster books,
global-long/cluster-short sides, every-two-month rebalancing, and 20 bp one-way costs.
Each AUM cutoff is applied point in time before partition fitting and before both ranking
legs.  The history-only row is the no-AUM-filter control.
"""
from __future__ import annotations

import hashlib
import pickle
import time
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import qis
from factorlasso import compute_clusters_from_corr_matrix
from factorlasso.cluster_smoothing import _iter_correlation_inputs
from optimalportfolios.alphas.signals.momentum import compute_momentum_alpha
from optimalportfolios.alphas.signals.utils import score_within_clusters

import papers.cluster_lineage_2026.replication.run_backtests as e5
import papers.cluster_lineage_2026.replication.run_u1_covar_grid_long_short_prod as prod
import papers.cluster_lineage_2026.replication.run_u2_blackrock_aum_filter as aum50
import papers.cluster_lineage_2026.replication.run_u2_blackrock_etf_grid as funds
import papers.cluster_lineage_2026.replication.run_u2_blackrock_long_short_search as search
import papers.cluster_lineage_2026.replication.run_u2_blackrock_sleeve_grid as sleeves


RUNNER = (
    "papers/cluster_lineage_2026/replication/"
    "run_u2_blackrock_aum_sensitivity.py"
)
FILTERS: tuple[tuple[str, float | None], ...] = (
    ("history_only", None),
    ("aum_25m", 25.0),
    ("aum_50m", 50.0),
    ("aum_100m", 100.0),
    ("aum_250m", 250.0),
    ("aum_500m", 500.0),
)
FREQUENCY = "W-THU"
SPAN = 156
Q = 0.25
WEIGHT_ID = "E50_F30_R20"
CONSTRUCTION = "group_equal"
HYBRID_VARIANT = "global_long_cluster_short"
SCHEDULE = "every_two_months"
COST_BPS = 20.0
PARTITION_CACHE_VERSION = 1
WEIGHT_TOLERANCE = 1e-12


def _root() -> Path:
    """Return the external AUM-sensitivity output directory."""
    root = aum50._root() / "threshold_sensitivity"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _cache_path() -> Path:
    """Return the multi-threshold partition-cache path."""
    return _root() / "partitions.pkl"


def _input_fingerprint() -> str:
    """Return a stable digest of inputs and every frozen sensitivity parameter."""
    payload = "|".join(
        [
            funds._input_fingerprint(),
            funds._sha256(aum50.AUM_FILE),
            repr(FILTERS),
            FREQUENCY,
            str(SPAN),
            str(Q),
            WEIGHT_ID,
            CONSTRUCTION,
            HYBRID_VARIANT,
            SCHEDULE,
            str(COST_BPS),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _eligibilities(
    daily: pd.DataFrame,
    dates: pd.DatetimeIndex,
    rolling_aum: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Return nested point-in-time eligibility panels for every cutoff."""
    history = funds._eligibility_for_dates(daily, dates)
    aum_at_dates = aum50._aum_for_dates(dates, rolling_aum).reindex(
        columns=history.columns
    )
    output = {"history_only": history}
    for filter_id, threshold in FILTERS:
        if threshold is not None:
            output[filter_id] = history & aum_at_dates.gt(threshold)
    return output


def _build_partitions(
    eligibilities: Mapping[str, pd.DataFrame],
    *,
    force: bool = False,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, str]:
    """Fit all cutoff-specific partitions from one correlation-input pass."""
    path = _cache_path()
    fingerprint = _input_fingerprint()
    if path.exists() and not force:
        with path.open("rb") as stream:
            cached = pickle.load(stream)
        if (
            cached.get("version") == PARTITION_CACHE_VERSION
            and cached.get("input_fingerprint") == fingerprint
        ):
            return cached["panels"], cached["diagnostics"], "hit"

    daily = funds._read_daily()
    dates = funds._dates()
    returns = funds._native_returns(daily, FREQUENCY)
    model = funds._model(SPAN, FREQUENCY)
    panels = {
        filter_id: pd.DataFrame(np.nan, index=dates, columns=daily.columns)
        for filter_id, _ in FILTERS
    }
    rows = []
    iterator = _iter_correlation_inputs(returns, list(dates), model)
    for date, full_corr in iterator:
        for filter_id, threshold in FILTERS:
            eligibility = eligibilities[filter_id]
            assets = eligibility.columns[eligibility.loc[date].astype(bool)]
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
            panels[filter_id].loc[date, labels.index] = labels.to_numpy()
            rows.append(
                {
                    "filter_id": filter_id,
                    "threshold_usd_millions": threshold,
                    "date": date,
                    "members": len(labels),
                    "clusters": int(labels.nunique()),
                }
            )
    diagnostics = pd.DataFrame(rows)
    payload = {
        "version": PARTITION_CACHE_VERSION,
        "input_fingerprint": fingerprint,
        "panels": panels,
        "diagnostics": diagnostics,
    }
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)
    return panels, diagnostics, "miss"


def _signal_panels(
    daily: pd.DataFrame,
    dates: pd.DatetimeIndex,
    eligibility: pd.DataFrame,
    monthly_eligibility: pd.DataFrame,
    clusters: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, Mapping[str, float]]:
    """Build the fixed ROSAA global and cluster scores for one cutoff."""
    monthly_returns = funds._native_returns(daily, "ME")
    simple_returns = np.expm1(monthly_returns)
    signal_prices = qis.returns_to_nav(simple_returns)
    benchmark_returns = simple_returns.where(monthly_eligibility).mean(axis=1)
    benchmark = qis.returns_to_nav(benchmark_returns.rename("EW").to_frame())["EW"]
    global_source, raw_source = compute_momentum_alpha(
        prices=signal_prices,
        benchmark_price=benchmark,
        returns_freq="ME",
        group_data=None,
        long_span=12,
        short_span=None,
        vol_span=13,
        mean_adj_type=qis.MeanAdjType.EWMA,
    )
    global_scores, global_timestamps = prod._asof_panel(global_source, dates)
    cluster_source = score_within_clusters(
        raw_signal=raw_source,
        rolling_clusters=funds._panel_dict(clusters),
        min_cluster_size=5,
    )
    cluster_scores, cluster_timestamps = prod._asof_panel(cluster_source, dates)
    global_scores = global_scores.reindex(
        index=dates, columns=eligibility.columns
    ).where(eligibility)
    cluster_scores = cluster_scores.reindex(
        index=dates, columns=eligibility.columns
    ).where(eligibility)
    diagnostics = {
        "max_global_lookahead_days": float(
            global_timestamps.sub(global_timestamps.index).dt.days.max()
        ),
        "max_cluster_lookahead_days": float(
            cluster_timestamps.sub(cluster_timestamps.index).dt.days.max()
        ),
        "global_valid_min": float(global_scores.notna().sum(axis=1).min()),
        "cluster_valid_min": float(cluster_scores.notna().sum(axis=1).min()),
    }
    return global_scores, cluster_scores, diagnostics


def _weights(
    global_scores: pd.DataFrame,
    cluster_scores: pd.DataFrame,
    eligibility: pd.DataFrame,
    clusters: pd.DataFrame,
    sleeve_panel: pd.DataFrame,
) -> tuple[dict[str, pd.DataFrame], Mapping[str, float]]:
    """Build matched global, pure-cluster, and selected hybrid weights."""
    target = search._target_map(WEIGHT_ID)
    hierarchical_groups = sleeves._hierarchical_groups(clusters, sleeve_panel)
    cluster_weights, cluster_exact = search._long_short_weights(
        cluster_scores,
        eligibility,
        sleeve_panel,
        hierarchical_groups,
        target,
        q=Q,
        construction=CONSTRUCTION,
    )
    global_weights, global_exact = search._long_short_weights(
        global_scores,
        eligibility,
        sleeve_panel,
        sleeve_panel,
        target,
        q=Q,
        construction="asset_equal",
    )
    hybrid_weights = search._hybrid_weights(
        cluster_weights,
        global_weights,
        HYBRID_VARIANT,
        sleeve_panel,
        target,
    )
    hybrid_errors = {
        "hybrid_long_error": float(
            hybrid_weights.clip(lower=0.0).sum(axis=1).sub(1.0).abs().max()
        ),
        "hybrid_short_error": float(
            (-hybrid_weights.clip(upper=0.0)).sum(axis=1).sub(1.0).abs().max()
        ),
        "hybrid_net_error": float(hybrid_weights.sum(axis=1).abs().max()),
        "hybrid_gross_error": float(
            hybrid_weights.abs().sum(axis=1).sub(2.0).abs().max()
        ),
    }
    exact = {
        **{f"cluster_{key}": value for key, value in cluster_exact.items()},
        **{f"global_{key}": value for key, value in global_exact.items()},
        **hybrid_errors,
    }
    return {
        "global": global_weights,
        "cluster": cluster_weights,
        "hybrid": hybrid_weights,
    }, exact


def _window(
    prices_all: pd.DataFrame,
    eligibility_all: pd.DataFrame,
    window_name: str,
    window_dates: pd.DatetimeIndex,
) -> Mapping[str, object]:
    """Build one corrected closed performance window and its EW reference."""
    end = search.WINDOWS[window_name][1]
    prices = aum50._closed_window_prices(prices_all, window_dates, end)
    ew_weights = eligibility_all.reindex(index=window_dates).astype(float)
    ew_weights = ew_weights.div(ew_weights.sum(axis=1), axis=0)
    ew_net, _ = funds._backtest(
        prices,
        ew_weights,
        0.0,
        f"aum_sensitivity_{window_name}_EW",
    )
    return {
        "dates": window_dates,
        "prices": prices,
        "ew_nav": ew_net.get_portfolio_nav(),
    }


def _membership_mismatches(left: pd.DataFrame, right: pd.DataFrame) -> int:
    """Count direct membership-cell mismatches with missing values treated equally."""
    return int(left.fillna(-1.0).ne(right.fillna(-1.0)).to_numpy().sum())


def _eligibility_tables(
    eligibilities: Mapping[str, pd.DataFrame],
    sleeve_map: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-date and summarized eligible breadth by broad sleeve."""
    rows = []
    threshold_map = dict(FILTERS)
    for filter_id, eligibility in eligibilities.items():
        for date, row in eligibility.iterrows():
            item = {
                "filter_id": filter_id,
                "threshold_usd_millions": threshold_map[filter_id],
                "date": date,
                "eligible_total": int(row.sum()),
            }
            for sleeve in search.SLEEVES:
                item[f"eligible_{sleeve.lower().replace(' ', '_')}"] = int(
                    (row & sleeve_map.eq(sleeve)).sum()
                )
            rows.append(item)
    per_date = pd.DataFrame(rows)
    value_columns = [column for column in per_date if column.startswith("eligible_")]
    summary = (
        per_date.groupby(
            ["filter_id", "threshold_usd_millions"],
            dropna=False,
            sort=False,
        )[value_columns]
        .agg(["min", "median", "max"])
    )
    summary.columns = [f"{column}_{stat}" for column, stat in summary.columns]
    return per_date, summary.reset_index()


def _comparison(performance: pd.DataFrame) -> pd.DataFrame:
    """Compare pure-cluster and selected-hybrid rows with matched global rows."""
    keys = ["filter_id", "analysis_window"]
    globals_ = performance.loc[performance["method"].eq("global")].set_index(keys)
    rows = []
    for _, row in performance.loc[performance["method"].ne("global")].iterrows():
        reference = globals_.loc[(row["filter_id"], row["analysis_window"])]
        item = row.to_dict()
        for metric in search.COMPARISON_METRICS:
            item[f"global_{metric}"] = reference[metric]
            item[f"delta_{metric}"] = row[metric] - reference[metric]
        item["beats_global_net_return"] = item["delta_net_return_annualized"] > 0.0
        item["beats_global_sharpe"] = item["delta_sharpe_rf0"] > 0.0
        item["beats_global_both"] = (
            item["beats_global_net_return"] and item["beats_global_sharpe"]
        )
        rows.append(item)
    return pd.DataFrame(rows)


def _sensitivity_vs_50(comparison: pd.DataFrame) -> pd.DataFrame:
    """Add each cutoff's metric changes relative to the frozen USD 50m row."""
    keys = ["analysis_window", "method"]
    reference = comparison.loc[comparison["filter_id"].eq("aum_50m")].set_index(keys)
    output = comparison.copy()
    for metric in search.COMPARISON_METRICS:
        output[f"change_vs_50m_{metric}"] = [
            float(row[metric])
            - float(
                reference.loc[
                    (row["analysis_window"], row["method"]), metric
                ]
            )
            for _, row in output.iterrows()
        ]
        output[f"change_vs_50m_delta_{metric}"] = [
            float(row[f"delta_{metric}"])
            - float(reference.loc[(row["analysis_window"], row["method"]), f"delta_{metric}"])
            for _, row in output.iterrows()
        ]
    return output


def run(*, force_partitions: bool = False) -> Mapping[str, pd.DataFrame]:
    """Execute the fixed-model AUM-threshold sensitivity backtest."""
    started = time.perf_counter()
    daily = funds._read_daily()
    dates = funds._dates()
    headline_dates = dates[
        (dates >= funds.HEADLINE_START) & (dates <= funds.HEADLINE_END)
    ]
    rolling_aum = aum50._rolling_aum()
    eligibility_all = _eligibilities(daily, dates, rolling_aum)
    monthly_dates = funds._native_returns(daily, "ME").index
    monthly_eligibility = _eligibilities(daily, monthly_dates, rolling_aum)
    partitions, partition_diagnostics, cache_status = _build_partitions(
        eligibility_all,
        force=force_partitions,
    )
    sleeve_map = sleeves._broad_sleeves(daily.columns)
    sleeve_panel = sleeves._sleeve_panel(headline_dates, sleeve_map)
    prices_all = funds._performance_prices(daily)

    performance_rows = []
    weight_rows = []
    signal_rows = []
    missing_memberships = 0
    for filter_id, threshold in FILTERS:
        eligibility = eligibility_all[filter_id].reindex(index=headline_dates)
        clusters = partitions[filter_id].reindex(index=headline_dates)
        missing_memberships += int((eligibility & clusters.isna()).to_numpy().sum())
        global_scores, cluster_scores, signal_diagnostics = _signal_panels(
            daily,
            headline_dates,
            eligibility,
            monthly_eligibility[filter_id],
            clusters,
        )
        weights, exact = _weights(
            global_scores,
            cluster_scores,
            eligibility,
            clusters,
            sleeve_panel,
        )
        maximum_error = max(
            abs(float(value)) for key, value in exact.items() if "error" in key
        )
        weight_rows.append(
            {
                "filter_id": filter_id,
                "threshold_usd_millions": threshold,
                **exact,
                "maximum_error": maximum_error,
            }
        )
        signal_rows.append(
            {
                "filter_id": filter_id,
                "threshold_usd_millions": threshold,
                **signal_diagnostics,
            }
        )
        for window_name, (start, end) in search.WINDOWS.items():
            window_dates = headline_dates[
                (headline_dates >= start) & (headline_dates <= end)
            ]
            window = _window(
                prices_all,
                eligibility_all[filter_id],
                window_name,
                window_dates,
            )
            for method, method_weights in weights.items():
                payload = search._scheduled_performance_payload(
                    method_weights,
                    window,
                    SCHEDULE,
                    f"aum_sensitivity_{filter_id}_{window_name}_{method}",
                )
                performance_rows.append(
                    {
                        "filter_id": filter_id,
                        "threshold_usd_millions": threshold,
                        "analysis_window": window_name,
                        "method": method,
                        "frequency": FREQUENCY,
                        "span": SPAN,
                        "q": Q,
                        "weight_id": WEIGHT_ID,
                        "construction": CONSTRUCTION,
                        "hybrid_variant": HYBRID_VARIANT,
                        "schedule": SCHEDULE,
                        "cost_bps_one_way": COST_BPS,
                        "rebalance_dates": len(
                            search._rebalance_dates(window_dates, SCHEDULE)
                        ),
                        "runner": RUNNER,
                        **payload,
                    }
                )

    performance = pd.DataFrame(performance_rows)
    comparison = _comparison(performance)
    sensitivity = _sensitivity_vs_50(comparison)
    per_date, eligibility_summary = _eligibility_tables(
        eligibility_all, sleeve_map
    )
    weight_diagnostics = pd.DataFrame(weight_rows)
    signal_diagnostics = pd.DataFrame(signal_rows)

    history_reference = funds._load_partition(FREQUENCY, SPAN)[0]
    aum50_reference = aum50._load_partition(FREQUENCY, SPAN)[0]
    history_mismatches = _membership_mismatches(
        partitions["history_only"], history_reference
    )
    aum50_mismatches = _membership_mismatches(
        partitions["aum_50m"], aum50_reference
    )
    prior = pd.read_csv(
        aum50._root() / "hybrid_recheck_comparison.csv",
        float_precision="round_trip",
    )
    prior = prior.loc[
        prior["candidate_name"].eq("owner_base")
        & prior["hybrid_variant"].eq(HYBRID_VARIANT)
        & prior["schedule"].eq(SCHEDULE)
    ].set_index("analysis_window")
    current = performance.loc[
        performance["filter_id"].eq("aum_50m")
        & performance["method"].isin(("global", "hybrid"))
    ].set_index(["analysis_window", "method"])
    regression_errors = []
    for window_name in search.WINDOWS:
        for method, prior_prefix in (("global", "global"), ("hybrid", "hybrid")):
            regression_errors.extend(
                abs(
                    float(current.loc[(window_name, method), metric])
                    - float(prior.loc[window_name, f"{prior_prefix}_{metric}"])
                )
                for metric in search.COMPARISON_METRICS
            )
    max_regression_error = max(regression_errors)
    max_weight_error = float(weight_diagnostics["maximum_error"].max())
    max_lookahead = float(
        signal_diagnostics[
            ["max_global_lookahead_days", "max_cluster_lookahead_days"]
        ].to_numpy().max()
    )
    acceptance = pd.DataFrame(
        [
            {
                "check": "history-only partition membership mismatches",
                "measured": history_mismatches,
                "tolerance": 0,
            },
            {
                "check": "USD 50m partition membership mismatches",
                "measured": aum50_mismatches,
                "tolerance": 0,
            },
            {
                "check": "eligible memberships missing from partitions",
                "measured": missing_memberships,
                "tolerance": 0,
            },
            {
                "check": "maximum weight/exposure error",
                "measured": max_weight_error,
                "tolerance": WEIGHT_TOLERANCE,
            },
            {
                "check": "maximum signal lookahead days",
                "measured": max_lookahead,
                "tolerance": 0,
            },
            {
                "check": "declared performance rows",
                "measured": len(performance),
                "tolerance": len(FILTERS) * len(search.WINDOWS) * 3,
            },
            {
                "check": "declared comparison rows",
                "measured": len(comparison),
                "tolerance": len(FILTERS) * len(search.WINDOWS) * 2,
            },
            {
                "check": "USD 50m prior-run regression error",
                "measured": max_regression_error,
                "tolerance": WEIGHT_TOLERANCE,
            },
        ]
    )
    equality_checks = {
        "history-only partition membership mismatches",
        "USD 50m partition membership mismatches",
        "eligible memberships missing from partitions",
        "declared performance rows",
        "declared comparison rows",
    }
    acceptance["status"] = [
        "PASS"
        if (
            float(row["measured"]) == float(row["tolerance"])
            if row["check"] in equality_checks
            else float(row["measured"]) <= float(row["tolerance"])
        )
        else "FAIL"
        for _, row in acceptance.iterrows()
    ]
    if not acceptance["status"].eq("PASS").all():
        raise AssertionError(acceptance)

    filter_order = {filter_id: order for order, (filter_id, _) in enumerate(FILTERS)}
    full_summary = comparison.loc[
        comparison["analysis_window"].eq(search.FULL_WINDOW)
    ].copy()
    full_summary["filter_order"] = full_summary["filter_id"].map(filter_order)
    full_summary = full_summary.sort_values(["filter_order", "method"]).drop(
        columns="filter_order"
    )
    runtime = pd.DataFrame(
        [
            {
                "partition_cache_status": cache_status,
                "filters": len(FILTERS),
                "partition_cells": len(FILTERS),
                "performance_rows": len(performance),
                "runtime_seconds": time.perf_counter() - started,
            }
        ]
    )
    output = {
        "specification": pd.DataFrame(
            [
                {
                    "frequency": FREQUENCY,
                    "span": SPAN,
                    "signal": "ROSAA production",
                    "q": Q,
                    "weight_id": WEIGHT_ID,
                    "construction": CONSTRUCTION,
                    "hybrid_variant": HYBRID_VARIANT,
                    "schedule": SCHEDULE,
                    "cost_bps_one_way": COST_BPS,
                }
            ]
        ),
        "eligibility_by_date": per_date,
        "eligibility_summary": eligibility_summary,
        "partition_diagnostics": partition_diagnostics,
        "signal_diagnostics": signal_diagnostics,
        "weight_diagnostics": weight_diagnostics,
        "performance": performance,
        "comparison_vs_global": comparison,
        "sensitivity_vs_50m": sensitivity,
        "full_window_summary": full_summary,
        "acceptance": acceptance,
        "runtime": runtime,
    }
    for name, frame in output.items():
        e5._write(frame, _root() / f"{name}.csv")
    return output


def _hash_outputs() -> dict[str, str]:
    """Hash deterministic numerical artifacts, excluding runtime."""
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(_root().glob("*.csv"))
        if path.name not in {"runtime.csv", "determinism.csv"}
    }


def verify_determinism() -> pd.DataFrame:
    """Replay the cache-first sensitivity and require byte-identical artifacts."""
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
    """Run and replay the fixed-model AUM sensitivity."""
    replay = verify_determinism()
    summary = pd.read_csv(
        _root() / "full_window_summary.csv", float_precision="round_trip"
    )
    columns = [
        "filter_id",
        "threshold_usd_millions",
        "method",
        "net_return_annualized",
        "global_net_return_annualized",
        "delta_net_return_annualized",
        "sharpe_rf0",
        "global_sharpe_rf0",
        "delta_sharpe_rf0",
    ]
    print(summary[columns].to_string(index=False), flush=True)
    print(f"determinism: {int(replay['byte_identical'].sum())}/{len(replay)}", flush=True)


if __name__ == "__main__":
    main()
