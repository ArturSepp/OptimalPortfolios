"""Run the U3 ROSAA min-10 design on every source futures contract.

This is a labelled sensitivity only. It disables the owner liquidity screen
while preserving every signal, cluster, rank, sleeve-budget, calendar, lag, and
cost setting of the filtered U3 primary. The production eligibility registry is
restored immediately after the all-source context has been constructed.
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


SIGNAL_ID = "rosaa_risk_adjusted_momentum"
ACCOUNTING_TOLERANCE = 1e-10
RUNNER = (
    "papers/cluster_lineage_2026/replication/"
    "run_u3_rosaa_min10_all_futures_sensitivity.py"
)


def _root() -> Path:
    """Return the gitignored all-futures sensitivity directory."""
    root = (
        Path(__file__).resolve().parents[1]
        / "local_outputs"
        / "e5b"
        / "u3_rosaa_ra_min10_all_futures_sensitivity_20260816"
    )
    root.mkdir(parents=True, exist_ok=True)
    return root


def _all_futures_context() -> tuple[Mapping[str, object], frozenset[str]]:
    """Build an all-source context without mutating the primary screen."""
    owner_exclusions = e5.FUTURES_INVESTABILITY_EXCLUSIONS
    try:
        e5.FUTURES_INVESTABILITY_EXCLUSIONS = frozenset()
        context = comparison._u3_context()
    finally:
        e5.FUTURES_INVESTABILITY_EXCLUSIONS = owner_exclusions
    return context, owner_exclusions


def _portfolios(
    context: Mapping[str, object],
) -> tuple[dict[str, object], dict[str, pd.DataFrame], Mapping[str, object]]:
    """Build the all-futures global and cluster books through accepted APIs."""
    global_scores, cluster_scores, signal_diagnostics = comparison._signal_pair(
        signal_id=SIGNAL_ID,
        prices=context["signal_prices"],
        benchmark=context["benchmark"],
        groups=context["groups"],
        dates=context["dates"],
        eligibility=context["eligibility"],
    )
    portfolios = {}
    weights = {}
    weight_diagnostics = {}
    for method, scores in (("global", global_scores), ("cluster", cluster_scores)):
        method_weights, method_diagnostics = comparison._long_short_weights(
            scores=scores,
            prices=context["rank_prices"],
            eligibility=context["eligibility"],
            sleeve_panel=context["sleeve_panel"],
            sleeves=context["sleeves"],
            target=context["target"],
            q=context["q"],
        )
        net, gross = u3_equal._backtest(
            context["performance_prices"],
            method_weights.reindex(index=context["scheduled_dates"]),
            context["cost_bps"] / 10000.0,
            f"U3_all_futures_{SIGNAL_ID}_{method}_min10",
        )
        portfolios[method] = net
        portfolios[f"{method}_gross"] = gross
        weights[method] = method_weights
        weight_diagnostics[method] = method_diagnostics
    return portfolios, weights, {
        "signal": signal_diagnostics,
        "weights": weight_diagnostics,
    }


def run() -> Mapping[str, pd.DataFrame]:
    """Run and validate the all-source sensitivity and exact attribution."""
    started = time.perf_counter()
    context, owner_exclusions = _all_futures_context()
    portfolios, weights, diagnostics = _portfolios(context)
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
    present_exclusions = instrument["ticker"].isin(owner_exclusions)
    included_exclusions = set(instrument.loc[present_exclusions, "ticker"])
    expected_present = set(eligibility.columns).intersection(owner_exclusions)
    maximum_weight_error = max(
        max(float(value) for value in payload.values())
        for payload in diagnostics["weights"].values()
    )
    passed = (
        accounting_error <= ACCOUNTING_TOLERANCE
        and maximum_weight_error <= comparison.TOLERANCE
        and included_exclusions == expected_present
        and bool(np.isfinite(asset_class.select_dtypes("number").to_numpy()).all())
    )
    reconciliation = pd.DataFrame(
        [
            {
                "universe": "U3_futures_all_source_sensitivity",
                "signal_id": SIGNAL_ID,
                "analysis_window": context["window"],
                "min_cluster_size": comparison.MIN_CLUSTER_SIZE,
                "q": context["q"],
                "cost_bps_one_way": context["cost_bps"],
                "eligible_instruments": len(instrument),
                "owner_exclusions_reincluded": len(included_exclusions),
                "cluster_total_net_pnl_pct": cluster_total,
                "global_total_net_pnl_pct": global_total,
                "cluster_minus_global_pct": cluster_total - global_total,
                "maximum_accounting_error": accounting_error,
                "maximum_weight_error": maximum_weight_error,
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
        "reincluded_owner_exclusions": instrument.loc[present_exclusions].reset_index(
            drop=True
        ),
        "design": pd.DataFrame(
            [
                {
                    "status": "sensitivity_not_primary",
                    "signal": SIGNAL_ID,
                    "eligibility": "all source futures; owner liquidity screen disabled",
                    "primary_u3_universe_unchanged": True,
                    "min_cluster_size": comparison.MIN_CLUSTER_SIZE,
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
    """Replay the sensitivity and require byte-identical artifacts."""
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
    """Run, replay, and print the all-futures sensitivity."""
    replay = verify_determinism()
    performance = pd.read_csv(_root() / "performance.csv", float_precision="round_trip")
    print(performance.to_string(index=False))
    print(f"U3 ROSAA all-futures sensitivity: PASS ({len(replay)}/{len(replay)})")


if __name__ == "__main__":
    main()
