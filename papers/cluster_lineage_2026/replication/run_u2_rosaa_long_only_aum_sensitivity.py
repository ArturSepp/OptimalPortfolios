"""Measure AUM-cutoff impact on the U2 ROSAA equal-fund long-only strategy.

Every cutoff is imposed point in time before clustering, signal benchmarking,
and ranking. The model remains ROSAA 12/3 risk-adjusted momentum with a
13-month volatility span, EWMA mean adjustment, minimum cluster size 10, top
quartile selection, equal selected-fund weights, and 20 bp one-way costs.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

import numpy as np
import pandas as pd
import qis

import papers.cluster_lineage_2026.replication.run_backtests as e5
import papers.cluster_lineage_2026.replication.run_u2_all_funds_asset_class_attribution as attribution
import papers.cluster_lineage_2026.replication.run_u2_blackrock_aum_filter as aum
import papers.cluster_lineage_2026.replication.run_u2_blackrock_aum_sensitivity as sensitivity
import papers.cluster_lineage_2026.replication.run_u2_blackrock_etf_grid as funds
import papers.cluster_lineage_2026.replication.run_u2_blackrock_long_short_search as search
import papers.cluster_lineage_2026.replication.run_u2_rosaa_short3_equal_fund_long_only as current
import papers.cluster_lineage_2026.replication.run_u2_u3_min_cluster10_signal_comparison as comparison


RUNNER = (
    "papers/cluster_lineage_2026/replication/"
    "run_u2_rosaa_long_only_aum_sensitivity.py"
)
SIGNAL_ID = "rosaa_risk_adjusted_momentum"
SHORT_SPAN = 3
TOLERANCE = 1e-10
base = current.base


def _root() -> Path:
    """Return the gitignored sensitivity output directory."""
    root = (
        Path(__file__).resolve().parents[1]
        / "local_outputs"
        / "e5b"
        / "u2_rosaa_short3_min10_equal_fund_long_only_aum_sensitivity_20260816"
    )
    root.mkdir(parents=True, exist_ok=True)
    return root


def _filter_root(filter_id: str) -> Path:
    """Return one cutoff's detailed attribution directory."""
    root = _root() / filter_id
    root.mkdir(parents=True, exist_ok=True)
    return root


def _context(
    *,
    filter_id: str,
    daily: pd.DataFrame,
    headline_dates: pd.DatetimeIndex,
    monthly_returns: pd.DataFrame,
    eligibility_all: dict[str, pd.DataFrame],
    monthly_eligibility: dict[str, pd.DataFrame],
    partitions: dict[str, pd.DataFrame],
    performance_prices: pd.DataFrame,
) -> dict[str, object]:
    """Build one cutoff-specific context matching the accepted AUM100 run."""
    eligibility = eligibility_all[filter_id].reindex(index=headline_dates).astype(bool)
    groups = partitions[filter_id].reindex(index=headline_dates)
    missing = int((eligibility & groups.isna()).to_numpy().sum())
    if missing:
        raise AssertionError(f"{filter_id} has {missing} missing memberships")
    simple_returns = np.expm1(monthly_returns)
    signal_prices = qis.returns_to_nav(simple_returns)
    benchmark_returns = simple_returns.where(monthly_eligibility[filter_id]).mean(axis=1)
    benchmark = qis.returns_to_nav(benchmark_returns.rename("EW").to_frame())["EW"]
    window = sensitivity._window(
        performance_prices,
        eligibility_all[filter_id],
        search.FULL_WINDOW,
        headline_dates,
    )
    return {
        "universe": "U2_funds",
        "dates": headline_dates,
        "eligibility": eligibility,
        "groups": groups,
        "signal_prices": signal_prices,
        "benchmark": benchmark,
        "rank_prices": performance_prices.reindex(index=headline_dates, method="ffill"),
        "performance_prices": window["prices"],
        "scheduled_dates": search._rebalance_dates(
            headline_dates, attribution.SCHEDULE
        ),
        "q": attribution.Q,
        "cost_bps": attribution.COST_BPS,
        "ew_nav": window["ew_nav"],
        "window": search.FULL_WINDOW,
        "schedule": attribution.SCHEDULE,
    }


def _comparison(performance: pd.DataFrame) -> pd.DataFrame:
    """Return cluster-minus-global performance deltas by cutoff."""
    metrics = (
        "net_total_return",
        "net_return_annualized",
        "volatility_annualized",
        "sharpe_rf0",
        "one_way_turnover_annualized",
        "cost_drag_bp_per_year",
    )
    rows = []
    for filter_id, frame in performance.groupby("filter_id", sort=False):
        indexed = frame.set_index("method")
        row = {
            "filter_id": filter_id,
            "threshold_usd_millions": frame["threshold_usd_millions"].iloc[0],
        }
        for metric in metrics:
            row[f"global_{metric}"] = indexed.loc["global", metric]
            row[f"cluster_{metric}"] = indexed.loc["cluster", metric]
            row[f"delta_{metric}"] = (
                indexed.loc["cluster", metric] - indexed.loc["global", metric]
            )
        rows.append(row)
    return pd.DataFrame(rows)


def run() -> dict[str, pd.DataFrame]:
    """Run all cached AUM cutoffs and save aggregate and detailed results."""
    started = time.perf_counter()
    daily = funds._read_daily()
    dates = funds._dates()
    headline_dates = dates[
        (dates >= funds.HEADLINE_START) & (dates <= funds.HEADLINE_END)
    ]
    rolling_aum = aum._rolling_aum()
    eligibility_all = sensitivity._eligibilities(daily, dates, rolling_aum)
    monthly_returns = funds._native_returns(daily, "ME")
    monthly_eligibility = sensitivity._eligibilities(
        daily, monthly_returns.index, rolling_aum
    )
    partitions, _, cache_status = sensitivity._build_partitions(eligibility_all)
    if cache_status != "hit":
        raise AssertionError("AUM sensitivity must consume the completed partition cache")
    performance_prices = funds._performance_prices(daily)
    performance_rows = []
    eligibility_rows = []
    signal_rows = []
    reconciliation_rows = []
    threshold_map = dict(sensitivity.FILTERS)
    for filter_id, threshold in sensitivity.FILTERS:
        context = _context(
            filter_id=filter_id,
            daily=daily,
            headline_dates=headline_dates,
            monthly_returns=monthly_returns,
            eligibility_all=eligibility_all,
            monthly_eligibility=monthly_eligibility,
            partitions=partitions,
            performance_prices=performance_prices,
        )
        comparison._u2_context = lambda context=context: context
        base._root = lambda filter_id=filter_id: _filter_root(filter_id)
        base.SIGNAL_ID = SIGNAL_ID
        base.SHORT_SPAN = SHORT_SPAN
        base.BOOK = "equal_fund_single_cross_section_long_only"
        base.RUNNER = RUNNER
        base._equal_fund_weights = current.long_only._equal_fund_weights
        output = base.run()
        frame = output["performance"].copy()
        frame.insert(0, "filter_id", filter_id)
        frame.insert(1, "threshold_usd_millions", threshold)
        performance_rows.append(frame)
        signal = output["signal_diagnostics"].copy()
        signal.insert(0, "filter_id", filter_id)
        signal.insert(1, "threshold_usd_millions", threshold)
        signal_rows.append(signal)
        reconciliation = output["reconciliation"].copy()
        reconciliation.insert(0, "filter_id", filter_id)
        reconciliation.insert(1, "threshold_usd_millions", threshold)
        reconciliation_rows.append(reconciliation)
        eligible = context["eligibility"].sum(axis=1)
        cluster_counts = context["groups"].nunique(axis=1, dropna=True)
        eligibility_rows.append(
            {
                "filter_id": filter_id,
                "threshold_usd_millions": threshold,
                "funds_ever_eligible": int(context["eligibility"].any(axis=0).sum()),
                "eligible_funds_min": int(eligible.min()),
                "eligible_funds_mean": float(eligible.mean()),
                "eligible_funds_max": int(eligible.max()),
                "clusters_mean": float(cluster_counts.mean()),
            }
        )
    performance = pd.concat(performance_rows, ignore_index=True)
    outputs = {
        "performance": performance,
        "comparison": _comparison(performance),
        "eligibility": pd.DataFrame(eligibility_rows),
        "signal_diagnostics": pd.concat(signal_rows, ignore_index=True),
        "reconciliation": pd.concat(reconciliation_rows, ignore_index=True),
        "specification": pd.DataFrame(
            [
                {
                    "signal": SIGNAL_ID,
                    "long_span": 12,
                    "short_span": SHORT_SPAN,
                    "vol_span": 13,
                    "mean_adjustment": "EWMA",
                    "min_cluster_size": comparison.MIN_CLUSTER_SIZE,
                    "q": attribution.Q,
                    "weighting": "equal selected fund",
                    "cost_bps_one_way": attribution.COST_BPS,
                    "filters": ",".join(threshold_map),
                    "partition_cache_status": cache_status,
                    "runner": RUNNER,
                }
            ]
        ),
    }
    if not outputs["reconciliation"]["status"].eq("PASS").all():
        raise AssertionError(outputs["reconciliation"])
    for name, frame in outputs.items():
        e5._write(frame, _root() / f"{name}.csv")
    e5._write(
        pd.DataFrame([{"runtime_seconds": time.perf_counter() - started}]),
        _root() / "runtime.csv",
    )
    return outputs


def _hash_outputs() -> dict[str, str]:
    """Hash deterministic aggregate and cutoff-level artifacts."""
    return {
        str(path.relative_to(_root())): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(_root().rglob("*.csv"))
        if path.name not in {"runtime.csv", "determinism.csv"}
    }


def verify_determinism() -> pd.DataFrame:
    """Replay all cutoffs and require byte-identical artifacts."""
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
    """Run the sensitivity replay and print its comparison table."""
    replay = verify_determinism()
    table = pd.read_csv(_root() / "comparison.csv", float_precision="round_trip")
    print(table.to_string(index=False))
    print(f"U2 ROSAA long-only AUM sensitivity: PASS ({len(replay)}/{len(replay)})")


if __name__ == "__main__":
    main()
