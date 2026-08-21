"""Decompose U3 ROSAA risk-adjusted momentum P&L by strategic asset class.

The runner changes only the signal relative to the accepted U3 classic-momentum
attribution. It retains the owner-screened point-in-time futures universe,
M1-star clusters, minimum cluster size 10, canonical within-sleeve long-short
rank, 30/30/30/10 sleeve budgets, q=0.25, one-period implementation lag, U1
headline window, and 10 bp one-way costs. Instrument net P&L is exact holding
P&L less realised instrument costs and reconciles to the portfolio NAV change.
"""
from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

import papers.cluster_lineage_2026.replication.run_backtests as e5
import papers.cluster_lineage_2026.replication.run_futures_best_relative_pnl_scatter as prior
import papers.cluster_lineage_2026.replication.run_futures_sleeve_grid as u3_equal
import papers.cluster_lineage_2026.replication.run_u2_u3_min_cluster10_signal_comparison as comparison
import papers.cluster_lineage_2026.replication.run_u3_classic_min10_instrument_pnl_scatter as attribution


SIGNAL_ID = "rosaa_risk_adjusted_momentum"
ACCOUNTING_TOLERANCE = 1e-10
PERFORMANCE_TOLERANCE = 1e-12
RUNNER = (
    "papers/cluster_lineage_2026/replication/"
    "run_u3_rosaa_min10_asset_class_pnl.py"
)


def _root() -> Path:
    """Return the gitignored local attribution directory."""
    root = (
        Path(__file__).resolve().parents[1]
        / "local_outputs"
        / "e5b"
        / "u3_rosaa_ra_min10_asset_class_pnl_20260816"
    )
    root.mkdir(parents=True, exist_ok=True)
    return root


def run() -> Mapping[str, pd.DataFrame]:
    """Compute, validate, and save the ROSAA asset-class P&L decomposition."""
    started = time.perf_counter()
    portfolios, weights, diagnostics = attribution._portfolios_and_weights(SIGNAL_ID)
    context = diagnostics["context"]
    eligibility = context["eligibility"]
    cluster_pnl, cluster_diag = prior._net_attribution(portfolios["cluster"])
    global_pnl, global_diag = prior._net_attribution(portfolios["global"])
    data = e5.load_universe(u3_equal.UNIVERSE)
    tickers = eligibility.columns[eligibility.any(axis=0)]
    sleeves = u3_equal._broad_sleeves(data.taxonomy, eligibility.columns)
    instrument = prior._instrument_table(
        tickers=tickers,
        cluster_net_pnl=cluster_pnl.sum(axis=0),
        global_net_pnl=global_pnl.sum(axis=0),
        cluster_beginning_nav=float(cluster_diag["beginning_nav"]),
        global_beginning_nav=float(global_diag["beginning_nav"]),
        taxonomy=data.taxonomy,
        sleeves=sleeves,
        eligibility=eligibility,
        cluster_weights=weights["cluster"],
        global_weights=weights["global"],
    )
    asset_class = (
        instrument.groupby("broad_asset_class", sort=True)
        .agg(
            instruments=("ticker", "size"),
            cluster_net_pnl_pct=("cluster_net_pnl_pct_of_start", "sum"),
            global_net_pnl_pct=("global_net_pnl_pct_of_start", "sum"),
        )
        .reset_index()
    )
    asset_class["cluster_minus_global_pct"] = (
        asset_class["cluster_net_pnl_pct"] - asset_class["global_net_pnl_pct"]
    )

    performance_rows = []
    for method in ("global", "cluster"):
        payload = u3_equal._performance_payload(
            portfolios[method], portfolios[f"{method}_gross"], context["ew_nav"]
        )
        performance_rows.append({"method": method, **payload})
    performance = pd.DataFrame(performance_rows)
    reference = pd.read_csv(
        comparison._root() / "performance.csv", float_precision="round_trip"
    )
    reference = reference.loc[
        reference["universe"].eq("U3_futures")
        & reference["signal_id"].eq(SIGNAL_ID)
    ].set_index("method")
    indexed = performance.set_index("method")
    metrics = (
        "net_return_annualized",
        "volatility_annualized",
        "sharpe_rf0",
        "one_way_turnover_annualized",
    )
    performance_error = max(
        abs(float(indexed.loc[method, metric]) - float(reference.loc[method, metric]))
        for method in ("global", "cluster")
        for metric in metrics
    )
    cluster_total = float(instrument["cluster_net_pnl_pct_of_start"].sum())
    global_total = float(instrument["global_net_pnl_pct_of_start"].sum())
    accounting_error = max(
        float(cluster_diag["max_step_reconciliation_abs_error"]),
        float(global_diag["max_step_reconciliation_abs_error"]),
        float(cluster_diag["cumulative_reconciliation_abs_error"]),
        float(global_diag["cumulative_reconciliation_abs_error"]),
        abs(cluster_total - 100.0 * float(cluster_diag["attributed_net_total_return"])),
        abs(global_total - 100.0 * float(global_diag["attributed_net_total_return"])),
        abs(cluster_total - float(asset_class["cluster_net_pnl_pct"].sum())),
        abs(global_total - float(asset_class["global_net_pnl_pct"].sum())),
    )
    excluded_rows = int(
        instrument["ticker"].isin(e5.FUTURES_INVESTABILITY_EXCLUSIONS).sum()
    )
    finite = bool(
        np.isfinite(
            asset_class[
                [
                    "cluster_net_pnl_pct",
                    "global_net_pnl_pct",
                    "cluster_minus_global_pct",
                ]
            ].to_numpy()
        ).all()
    )
    passed = (
        accounting_error <= ACCOUNTING_TOLERANCE
        and performance_error <= PERFORMANCE_TOLERANCE
        and excluded_rows == 0
        and int(asset_class["instruments"].sum()) == len(instrument)
        and finite
    )
    reconciliation = pd.DataFrame(
        [
            {
                "universe": "U3_futures",
                "signal_id": SIGNAL_ID,
                "analysis_window": context["window"],
                "min_cluster_size": comparison.MIN_CLUSTER_SIZE,
                "q": context["q"],
                "cost_bps_one_way": context["cost_bps"],
                "eligible_instruments": len(instrument),
                "cluster_total_net_pnl_pct": cluster_total,
                "global_total_net_pnl_pct": global_total,
                "cluster_minus_global_pct": cluster_total - global_total,
                "maximum_accounting_error": accounting_error,
                "performance_regression_max_abs_error": performance_error,
                "owner_excluded_rows": excluded_rows,
                "status": "PASS" if passed else "FAIL",
                "runner": RUNNER,
            }
        ]
    )
    if not passed:
        raise AssertionError(reconciliation.to_dict(orient="records")[0])
    outputs = {
        "asset_class_pnl": asset_class,
        "instrument_pnl": instrument,
        "performance": performance,
        "reconciliation": reconciliation,
        "design": pd.DataFrame(
            [
                {
                    "signal": SIGNAL_ID,
                    "cluster_role": "score standardisation only",
                    "min_cluster_size": comparison.MIN_CLUSTER_SIZE,
                    "pnl_unit": "percentage points of beginning NAV",
                    "cost_treatment": "exact realised instrument cost",
                    "runner": RUNNER,
                }
            ]
        ),
    }
    for name, frame in outputs.items():
        e5._write(frame, _root() / f"{name}.csv")
    e5._write(
        pd.DataFrame([{"runtime_seconds": time.perf_counter() - started}]),
        _root() / "runtime.csv",
    )
    return outputs


def _hash_outputs() -> dict[str, str]:
    """Hash deterministic artifacts while excluding runtime and replay."""
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(_root().glob("*.csv"))
        if path.name not in {"runtime.csv", "determinism.csv"}
    }


def verify_determinism() -> pd.DataFrame:
    """Replay the decomposition and require byte-identical artifacts."""
    run()
    first = _hash_outputs()
    run()
    second = _hash_outputs()
    names = sorted(first)
    replay = pd.DataFrame(
        {
            "artifact": names,
            "first_sha256": [first[name] for name in names],
            "second_sha256": [second[name] for name in names],
            "byte_identical": [first[name] == second[name] for name in names],
        }
    )
    e5._write(replay, _root() / "determinism.csv")
    if not replay["byte_identical"].all():
        raise AssertionError(replay.loc[~replay["byte_identical"]])
    return replay


def main() -> None:
    """Run, replay, and print the asset-class decomposition."""
    replay = verify_determinism()
    table = pd.read_csv(_root() / "asset_class_pnl.csv", float_precision="round_trip")
    print(table.to_string(index=False))
    print(f"U3 ROSAA min10 asset-class P&L: PASS ({len(replay)}/{len(replay)})")


if __name__ == "__main__":
    main()
