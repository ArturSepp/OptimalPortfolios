"""Run the frozen U2 long-only signal-ranking comparison with AUM100 eligibility.

Both ranking legs receive fixed 50/30/20 Equity, Fixed Income, and Rest budgets.
The global control ranks ROSAA risk-adjusted momentum within each broad sleeve.  The
cluster treatment ranks the same signal within point-in-time correlation clusters and
allocates each sleeve budget equally across its available clusters.  Eligibility is the
strict latest-12-completed-month average AUM greater than USD100m rule and is applied
before both ranking and clustering.  The construction is long-only, q=25%, monthly,
one-period lagged, and charged 20 bp one way.
"""

from __future__ import annotations

import hashlib
import pickle
import time
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from factorlasso import compute_clusters_from_corr_matrix
from factorlasso.cluster_smoothing import _iter_correlation_inputs

import papers.cluster_lineage_2026.replication.run_backtests as e5
import papers.cluster_lineage_2026.replication.run_u2_blackrock_aum_filter as aum
import papers.cluster_lineage_2026.replication.run_u2_blackrock_aum_sensitivity as sensitivity
import papers.cluster_lineage_2026.replication.run_u2_blackrock_etf_grid as funds
import papers.cluster_lineage_2026.replication.run_u2_blackrock_long_short_search as search
import papers.cluster_lineage_2026.replication.run_u2_blackrock_sleeve_grid as sleeves


RUNNER = (
    "papers/cluster_lineage_2026/replication/"
    "run_u2_blackrock_aum100_long_only.py"
)
FILTER_ID = "aum_100m"
AUM_THRESHOLD_USD_MILLIONS = 100.0
FREQUENCY = "ME"
SPAN = 12
Q = 0.25
WEIGHT_ID = "E50_F30_R20"
SCHEDULE = "monthly"
COST_BPS = 20.0
TOLERANCE = 1e-12
CACHE_VERSION = 1


def _root() -> Path:
    """Return the isolated external output directory."""
    root = funds._root() / "aum100_long_only_20260816"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _partition_path() -> Path:
    """Return the isolated AUM100 ME/span-12 partition cache path."""
    return _root() / "partition.pkl"


def _fingerprint() -> str:
    """Return a stable input and specification fingerprint."""
    payload = "|".join(
        [
            funds._input_fingerprint(),
            funds._sha256(aum.AUM_FILE),
            FILTER_ID,
            str(AUM_THRESHOLD_USD_MILLIONS),
            FREQUENCY,
            str(SPAN),
            str(Q),
            WEIGHT_ID,
            SCHEDULE,
            str(COST_BPS),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _partition(
    daily: pd.DataFrame,
    dates: pd.DatetimeIndex,
    eligibility: pd.DataFrame,
    *,
    force: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Fit or load the point-in-time AUM100 ME/span-12 partition panel."""
    path = _partition_path()
    fingerprint = _fingerprint()
    if path.exists() and not force:
        with path.open("rb") as stream:
            cached = pickle.load(stream)
        if (
            cached.get("version") == CACHE_VERSION
            and cached.get("fingerprint") == fingerprint
        ):
            return cached["panel"], cached["diagnostics"], "hit"

    returns = funds._native_returns(daily, FREQUENCY)
    model = funds._model(SPAN, FREQUENCY)
    panel = pd.DataFrame(np.nan, index=dates, columns=daily.columns)
    rows = []
    for date, full_corr in _iter_correlation_inputs(returns, list(dates), model):
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
        panel.loc[date, labels.index] = labels.to_numpy()
        rows.append(
            {
                "date": date,
                "eligible_assets": len(assets),
                "partition_assets": len(labels),
                "clusters": int(labels.nunique()),
            }
        )
    diagnostics = pd.DataFrame(rows)
    payload = {
        "version": CACHE_VERSION,
        "fingerprint": fingerprint,
        "panel": panel,
        "diagnostics": diagnostics,
    }
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)
    return panel, diagnostics, "miss"


def _performance(
    prices_all: pd.DataFrame,
    eligibility_all: pd.DataFrame,
    weights: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Backtest both long-only ranking legs on every frozen analysis window."""
    headline_dates = weights["global"].index
    rows = []
    for window_name, (start, end) in search.WINDOWS.items():
        window_dates = headline_dates[
            (headline_dates >= start) & (headline_dates <= end)
        ]
        window = sensitivity._window(
            prices_all,
            eligibility_all,
            window_name,
            window_dates,
        )
        for method, method_weights in weights.items():
            net, gross = funds._backtest(
                window["prices"],
                method_weights.reindex(index=window_dates),
                COST_BPS / 10000.0,
                f"u2_aum100_long_only_{window_name}_{method}",
            )
            rows.append(
                {
                    "analysis_window": window_name,
                    "method": method,
                    "strategy": "long_only",
                    "signal": "ROSAA risk-adjusted momentum",
                    "frequency": FREQUENCY,
                    "span": SPAN,
                    "q": Q,
                    "weight_id": WEIGHT_ID,
                    "schedule": SCHEDULE,
                    "cost_bps_one_way": COST_BPS,
                    "rebalance_dates": len(window_dates),
                    "runner": RUNNER,
                    **sleeves._performance_payload(net, gross, window["ew_nav"]),
                }
            )
    return pd.DataFrame(rows)


def _comparison(performance: pd.DataFrame) -> pd.DataFrame:
    """Compare cluster-contained long-only ranks with matched global sleeve ranks."""
    global_rows = performance.loc[performance["method"].eq("global")].set_index(
        "analysis_window"
    )
    rows = []
    for _, cluster in performance.loc[performance["method"].eq("cluster")].iterrows():
        global_row = global_rows.loc[cluster["analysis_window"]]
        item = cluster.to_dict()
        for metric in search.COMPARISON_METRICS:
            item[f"global_{metric}"] = global_row[metric]
            item[f"delta_{metric}"] = cluster[metric] - global_row[metric]
        rows.append(item)
    return pd.DataFrame(rows)


def run(*, force_partition: bool = False) -> Mapping[str, pd.DataFrame]:
    """Execute and validate the AUM100 long-only U2 comparison."""
    started = time.perf_counter()
    daily = funds._read_daily()
    dates = funds._dates()
    headline_dates = dates[
        (dates >= funds.HEADLINE_START) & (dates <= funds.HEADLINE_END)
    ]
    rolling_aum = aum._rolling_aum()
    eligibility_all = sensitivity._eligibilities(
        daily, dates, rolling_aum
    )[FILTER_ID].astype(bool)
    monthly_dates = funds._native_returns(daily, "ME").index
    monthly_eligibility = sensitivity._eligibilities(
        daily, monthly_dates, rolling_aum
    )[FILTER_ID].astype(bool)
    partition, partition_diagnostics, cache_status = _partition(
        daily,
        dates,
        eligibility_all,
        force=force_partition,
    )
    eligibility = eligibility_all.reindex(index=headline_dates).astype(bool)
    clusters = partition.reindex(index=headline_dates)
    missing_memberships = int((eligibility & clusters.isna()).to_numpy().sum())
    global_scores, cluster_scores, signal_diagnostics = sensitivity._signal_panels(
        daily,
        headline_dates,
        eligibility,
        monthly_eligibility,
        clusters,
    )
    sleeve_map = sleeves._broad_sleeves(eligibility.columns)
    sleeve_panel = sleeves._sleeve_panel(headline_dates, sleeve_map)
    target = search._target_map(WEIGHT_ID)
    hierarchical_groups = sleeves._hierarchical_groups(clusters, sleeve_panel)
    global_weights, global_diagnostics = sleeves._long_only_weights(
        global_scores,
        eligibility,
        sleeve_panel,
        sleeve_panel,
        target,
    )
    cluster_weights, cluster_diagnostics = sleeves._long_only_weights(
        cluster_scores,
        eligibility,
        sleeve_panel,
        hierarchical_groups,
        target,
    )
    weights = {"global": global_weights, "cluster": cluster_weights}
    performance = _performance(
        funds._performance_prices(daily), eligibility_all, weights
    )
    comparison = _comparison(performance)

    aum_at_dates = aum._aum_for_dates(headline_dates, rolling_aum).reindex(
        columns=eligibility.columns
    )
    threshold_violations = int(
        (eligibility & aum_at_dates.le(AUM_THRESHOLD_USD_MILLIONS)).to_numpy().sum()
    )
    diagnostics = pd.DataFrame(
        [
            {"method": "global", **global_diagnostics},
            {"method": "cluster", **cluster_diagnostics},
        ]
    )
    maximum_weight_error = float(
        diagnostics.filter(like="error").abs().to_numpy().max()
    )
    maximum_lookahead = max(
        float(signal_diagnostics["max_global_lookahead_days"]),
        float(signal_diagnostics["max_cluster_lookahead_days"]),
    )
    checks = [
        ("eligible memberships missing from partition", missing_memberships, 0, "eq"),
        ("AUM <= USD100m eligible observations", threshold_violations, 0, "eq"),
        ("maximum weight and sleeve-budget error", maximum_weight_error, TOLERANCE, "le"),
        ("maximum signal lookahead days", maximum_lookahead, 0, "le"),
        ("performance rows", len(performance), len(search.WINDOWS) * 2, "eq"),
        ("comparison rows", len(comparison), len(search.WINDOWS), "eq"),
    ]
    acceptance = pd.DataFrame(
        [
            {
                "check": check,
                "measured": measured,
                "tolerance": tolerance,
                "status": (
                    "PASS"
                    if (measured == tolerance if comparison_type == "eq" else measured <= tolerance)
                    else "FAIL"
                ),
            }
            for check, measured, tolerance, comparison_type in checks
        ]
    )
    if not acceptance["status"].eq("PASS").all():
        raise AssertionError(acceptance)

    output = {
        "specification": pd.DataFrame(
            [
                {
                    "strategy": "long_only",
                    "signal": "ROSAA risk-adjusted momentum",
                    "aum_rule": "latest 12 completed month-end average > USD100m",
                    "frequency": FREQUENCY,
                    "span": SPAN,
                    "q": Q,
                    "weight_id": WEIGHT_ID,
                    "equity_weight": target["Equity"],
                    "fixed_income_weight": target["Fixed Income"],
                    "rest_weight": target["Rest"],
                    "schedule": SCHEDULE,
                    "cost_bps_one_way": COST_BPS,
                }
            ]
        ),
        "performance": performance,
        "comparison_vs_global": comparison,
        "weight_diagnostics": diagnostics,
        "partition_diagnostics": partition_diagnostics,
        "acceptance": acceptance,
        "runtime": pd.DataFrame(
            [
                {
                    "partition_cache_status": cache_status,
                    "runtime_seconds": time.perf_counter() - started,
                }
            ]
        ),
    }
    for name, frame in output.items():
        e5._write(frame, _root() / f"{name}.csv")
    return output


def _hash_outputs() -> dict[str, str]:
    """Hash every deterministic CSV artifact."""
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(_root().glob("*.csv"))
        if path.name not in {"runtime.csv", "determinism.csv"}
    }


def verify_determinism() -> pd.DataFrame:
    """Replay cache-first and require byte-identical numerical outputs."""
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
    """Run, replay, and print the headline long-only comparison."""
    replay = verify_determinism()
    performance = pd.read_csv(
        _root() / "performance.csv", float_precision="round_trip"
    )
    headline = performance.loc[
        performance["analysis_window"].eq(search.FULL_WINDOW)
    ]
    columns = [
        "method",
        "net_return_annualized",
        "volatility_annualized",
        "sharpe_rf0",
        "one_way_turnover_annualized",
        "cost_drag_bp_per_year",
    ]
    print(headline[columns].to_string(index=False), flush=True)
    print(
        f"determinism: {int(replay['byte_identical'].sum())}/{len(replay)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
