"""Run U3 ROSAA momentum as one equal-contract long-short cross-section.

All eligible futures are ranked together with no strategic asset-class sleeves.
The top q=0.25 contracts receive equal positive weights summing to +1 and the
bottom q=0.25 receive equal negative weights summing to -1. The filtered U3
universe, ROSAA risk-adjusted signal, M1-star cluster standardisation with
minimum cluster size 10, monthly calendar, implementation lag, and 10 bp costs
are unchanged.
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
BOOK = "equal_contract_single_cross_section"
ACCOUNTING_TOLERANCE = 1e-10
RUNNER = (
    "papers/cluster_lineage_2026/replication/"
    "run_u3_rosaa_min10_equal_contract.py"
)


def _root() -> Path:
    """Return the gitignored equal-contract output directory."""
    root = (
        Path(__file__).resolve().parents[1]
        / "local_outputs"
        / "e5b"
        / "u3_rosaa_ra_min10_equal_contract_20260816"
    )
    root.mkdir(parents=True, exist_ok=True)
    return root


def _equal_contract_weights(
    scores: pd.DataFrame,
    context: Mapping[str, object],
) -> tuple[pd.DataFrame, Mapping[str, float]]:
    """Build one canonical equal-contract long-short book across all futures."""
    all_contracts = pd.DataFrame(
        "All",
        index=scores.index,
        columns=scores.columns,
    )
    weights, diagnostics = comparison._long_short_weights(
        scores=scores,
        prices=context["rank_prices"],
        eligibility=context["eligibility"],
        sleeve_panel=all_contracts,
        sleeves=("All",),
        target={"All": 1.0},
        q=context["q"],
    )
    long_weights = weights.clip(lower=0.0)
    short_weights = weights.clip(upper=0.0).abs()
    long_range = long_weights.where(long_weights.gt(0.0)).max(axis=1).subtract(
        long_weights.where(long_weights.gt(0.0)).min(axis=1)
    )
    short_range = short_weights.where(short_weights.gt(0.0)).max(axis=1).subtract(
        short_weights.where(short_weights.gt(0.0)).min(axis=1)
    )
    return weights, {
        **diagnostics,
        "max_long_contract_weight_range": float(long_range.fillna(0.0).max()),
        "max_short_contract_weight_range": float(short_range.fillna(0.0).max()),
        "min_long_contracts": int(long_weights.gt(0.0).sum(axis=1).min()),
        "max_long_contracts": int(long_weights.gt(0.0).sum(axis=1).max()),
        "min_short_contracts": int(short_weights.gt(0.0).sum(axis=1).min()),
        "max_short_contracts": int(short_weights.gt(0.0).sum(axis=1).max()),
    }


def run() -> Mapping[str, pd.DataFrame]:
    """Run, reconcile, and save global and cluster equal-contract books."""
    started = time.perf_counter()
    context = comparison._u3_context()
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
    weight_rows = []
    for method, scores in (("global", global_scores), ("cluster", cluster_scores)):
        method_weights, diagnostics = _equal_contract_weights(scores, context)
        maximum_error = max(
            float(value)
            for key, value in diagnostics.items()
            if "contracts" not in key
        )
        weight_rows.append(
            {
                "method": method,
                **diagnostics,
                "maximum_error": maximum_error,
                "tolerance": comparison.TOLERANCE,
                "status": "PASS" if maximum_error <= comparison.TOLERANCE else "FAIL",
            }
        )
        net, gross = u3_equal._backtest(
            context["performance_prices"],
            method_weights.reindex(index=context["scheduled_dates"]),
            context["cost_bps"] / 10000.0,
            f"U3_{BOOK}_{SIGNAL_ID}_{method}_min10",
        )
        portfolios[method] = net
        portfolios[f"{method}_gross"] = gross
        weights[method] = method_weights
    weight_diagnostics = pd.DataFrame(weight_rows)
    if not weight_diagnostics["status"].eq("PASS").all():
        raise AssertionError(weight_diagnostics)

    performance_rows = []
    pnl_diagnostics = {}
    pnl_by_method = {}
    for method in ("global", "cluster"):
        performance_rows.append(
            {
                "method": method,
                **u3_equal._performance_payload(
                    portfolios[method],
                    portfolios[f"{method}_gross"],
                    context["ew_nav"],
                ),
            }
        )
        pnl_by_method[method], pnl_diagnostics[method] = prior._net_attribution(
            portfolios[method]
        )
    performance = pd.DataFrame(performance_rows)
    eligibility = context["eligibility"]
    data = e5.load_universe(u3_equal.UNIVERSE)
    tickers = eligibility.columns[eligibility.any(axis=0)]
    sleeves = u3_equal._broad_sleeves(data.taxonomy, eligibility.columns)
    instrument = prior._instrument_table(
        tickers=tickers,
        cluster_net_pnl=pnl_by_method["cluster"].sum(axis=0),
        global_net_pnl=pnl_by_method["global"].sum(axis=0),
        cluster_beginning_nav=float(pnl_diagnostics["cluster"]["beginning_nav"]),
        global_beginning_nav=float(pnl_diagnostics["global"]["beginning_nav"]),
        taxonomy=data.taxonomy,
        sleeves=sleeves,
        eligibility=eligibility,
        cluster_weights=weights["cluster"],
        global_weights=weights["global"],
    )
    cluster_total = float(instrument["cluster_net_pnl_pct_of_start"].sum())
    global_total = float(instrument["global_net_pnl_pct_of_start"].sum())
    accounting_error = max(
        float(pnl_diagnostics[method][field])
        for method in ("global", "cluster")
        for field in (
            "max_step_reconciliation_abs_error",
            "cumulative_reconciliation_abs_error",
        )
    )
    accounting_error = max(
        accounting_error,
        abs(
            cluster_total
            - 100.0
            * float(pnl_diagnostics["cluster"]["attributed_net_total_return"])
        ),
        abs(
            global_total
            - 100.0
            * float(pnl_diagnostics["global"]["attributed_net_total_return"])
        ),
    )
    excluded_rows = int(
        instrument["ticker"].isin(e5.FUTURES_INVESTABILITY_EXCLUSIONS).sum()
    )
    passed = accounting_error <= ACCOUNTING_TOLERANCE and excluded_rows == 0
    reconciliation = pd.DataFrame(
        [
            {
                "universe": "U3_futures",
                "construction": BOOK,
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
                "owner_excluded_rows": excluded_rows,
                "status": "PASS" if passed else "FAIL",
                "runner": RUNNER,
            }
        ]
    )
    if not passed:
        raise AssertionError(reconciliation.to_dict(orient="records")[0])
    outputs = {
        "performance": performance,
        "weight_diagnostics": weight_diagnostics,
        "signal_diagnostics": pd.DataFrame([signal_diagnostics]),
        "instrument_pnl": instrument,
        "reconciliation": reconciliation,
        "design": pd.DataFrame(
            [
                {
                    "construction": BOOK,
                    "rank_scope": "all eligible futures together",
                    "long_book": "equal contract weights summing to +1",
                    "short_book": "equal contract weights summing to -1",
                    "asset_class_budgets": "none",
                    "signal": SIGNAL_ID,
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
    """Replay the equal-contract design and require byte-identical outputs."""
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
    """Run, replay, and print the equal-contract performance."""
    replay = verify_determinism()
    performance = pd.read_csv(_root() / "performance.csv", float_precision="round_trip")
    print(performance.to_string(index=False))
    print(f"U3 ROSAA equal-contract analysis: PASS ({len(replay)}/{len(replay)})")


if __name__ == "__main__":
    main()
