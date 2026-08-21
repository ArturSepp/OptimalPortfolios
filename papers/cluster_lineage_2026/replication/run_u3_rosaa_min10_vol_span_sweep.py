"""Sweep U3 per-contract EWMA volatility spans from one month to one year.

Every cell uses the filtered U3 universe, ROSAA risk-adjusted signal, M1-star
cluster standardisation with minimum cluster size 10, one cross-sectional
top/bottom quartile, TrendFollowingSystems inverse-volatility sizing with a 15%
instrument target and cap 5, monthly decisions, one-period lag, and 10 bp costs.
Only the daily EWMA volatility span changes.
"""
from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

import papers.cluster_lineage_2026.replication.run_backtests as e5
import papers.cluster_lineage_2026.replication.run_futures_sleeve_grid as u3_equal
import papers.cluster_lineage_2026.replication.run_u2_u3_min_cluster10_signal_comparison as comparison
import papers.cluster_lineage_2026.replication.run_u3_rosaa_min10_equal_contract as equal_contract
import papers.cluster_lineage_2026.replication.run_u3_rosaa_min10_vol_normalized as vol_normalized


VOL_SPANS = (21, 33, 42, 63, 126, 189, 252)
SPAN_LABELS = {
    21: "1m",
    33: "canonical_33d",
    42: "2m",
    63: "3m",
    126: "6m",
    189: "9m",
    252: "12m",
}
RUNNER = (
    "papers/cluster_lineage_2026/replication/"
    "run_u3_rosaa_min10_vol_span_sweep.py"
)


def _root() -> Path:
    """Return the gitignored volatility-span sweep directory."""
    root = (
        Path(__file__).resolve().parents[1]
        / "local_outputs"
        / "e5b"
        / "u3_rosaa_ra_min10_vol_span_sweep_20260816"
    )
    root.mkdir(parents=True, exist_ok=True)
    return root


def _comparison(performance: pd.DataFrame) -> pd.DataFrame:
    """Return cluster-minus-global metrics for each volatility span."""
    rows = []
    metrics = (
        "net_return_annualized",
        "volatility_annualized",
        "sharpe_rf0",
        "one_way_turnover_annualized",
        "cost_drag_bp_per_year",
        "alpha_vs_ew_annualized",
        "beta_vs_ew",
        "net_total_return",
    )
    for span, frame in performance.groupby("vol_span_days", sort=True):
        indexed = frame.set_index("method")
        row = {
            "vol_span_days": span,
            "vol_span_label": SPAN_LABELS[int(span)],
        }
        for metric in metrics:
            cluster = float(indexed.loc["cluster", metric])
            global_rank = float(indexed.loc["global", metric])
            row[f"cluster_{metric}"] = cluster
            row[f"global_{metric}"] = global_rank
            row[f"delta_{metric}"] = cluster - global_rank
        rows.append(row)
    return pd.DataFrame(rows)


def run() -> Mapping[str, pd.DataFrame]:
    """Execute, validate, and save the volatility-span sweep."""
    started = time.perf_counter()
    context = comparison._u3_context()
    global_scores, cluster_scores, signal_diagnostics = comparison._signal_pair(
        signal_id=vol_normalized.SIGNAL_ID,
        prices=context["signal_prices"],
        benchmark=context["benchmark"],
        groups=context["groups"],
        dates=context["dates"],
        eligibility=context["eligibility"],
    )
    base_weights = {
        method: equal_contract._equal_contract_weights(scores, context)[0]
        for method, scores in (("global", global_scores), ("cluster", cluster_scores))
    }
    performance_rows = []
    acceptance_rows = []
    for span in VOL_SPANS:
        scalers, annual_vols, vol_diagnostics = (
            vol_normalized._point_in_time_vol_scalers(
                context["dates"],
                context["eligibility"].columns,
                vol_span_days=span,
            )
        )
        for method in ("global", "cluster"):
            weights, diagnostics = vol_normalized._vol_normalized_weights(
                base_weights[method],
                scalers,
                annual_vols,
                context["eligibility"],
            )
            passed = (
                diagnostics["max_weight_outside_eligibility"] <= comparison.TOLERANCE
                and diagnostics["max_overlap_assets"] == 0
                and vol_diagnostics["max_vol_source_lookahead_days"] <= 0
                and vol_diagnostics["uncapped_scaler_identity_max_abs_error"]
                <= comparison.TOLERANCE
            )
            acceptance_rows.append(
                {
                    "vol_span_days": span,
                    "vol_span_label": SPAN_LABELS[span],
                    "method": method,
                    **vol_diagnostics,
                    **diagnostics,
                    "status": "PASS" if passed else "FAIL",
                }
            )
            net, gross = u3_equal._backtest(
                context["performance_prices"],
                weights.reindex(index=context["scheduled_dates"]),
                context["cost_bps"] / 10000.0,
                f"U3_volspan_{span}_{method}",
            )
            performance_rows.append(
                {
                    "vol_span_days": span,
                    "vol_span_label": SPAN_LABELS[span],
                    "method": method,
                    **u3_equal._performance_payload(net, gross, context["ew_nav"]),
                }
            )
    performance = pd.DataFrame(performance_rows)
    acceptance = pd.DataFrame(acceptance_rows)
    if not acceptance["status"].eq("PASS").all():
        raise AssertionError(acceptance.loc[~acceptance["status"].eq("PASS")])
    comparison_table = _comparison(performance)
    outputs = {
        "performance": performance,
        "comparison": comparison_table,
        "acceptance": acceptance,
        "signal_diagnostics": pd.DataFrame([signal_diagnostics]),
        "best_cells": pd.DataFrame(
            [
                performance.loc[
                    performance.groupby("method")["sharpe_rf0"].idxmax()
                ].sort_values("method").reset_index(drop=True)
            ][0]
        ),
        "design": pd.DataFrame(
            [
                {
                    "vol_spans_days": "|".join(map(str, VOL_SPANS)),
                    "vol_span_labels": "|".join(SPAN_LABELS[s] for s in VOL_SPANS),
                    "signal": vol_normalized.SIGNAL_ID,
                    "min_cluster_size": comparison.MIN_CLUSTER_SIZE,
                    "instrument_vol_target": vol_normalized.INSTRUMENT_VOL_TARGET,
                    "annualization_factor": vol_normalized.ANNUALIZATION_FACTOR,
                    "instrument_weight_cap": vol_normalized.INSTRUMENT_WEIGHT_CAP,
                    "cost_bps_one_way": context["cost_bps"],
                    "runner": RUNNER,
                }
            ]
        ),
        "runtime": pd.DataFrame(
            [{"runtime_seconds": time.perf_counter() - started, "runner": RUNNER}]
        ),
    }
    for name, frame in outputs.items():
        e5._write(frame, _root() / f"{name}.csv")
    return outputs


def _hash_outputs() -> dict[str, str]:
    """Hash deterministic sweep artifacts while excluding runtime and replay."""
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(_root().glob("*.csv"))
        if path.name not in {"runtime.csv", "determinism.csv"}
    }


def verify_determinism() -> pd.DataFrame:
    """Replay the complete sweep and require byte-identical outputs."""
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
    """Run, replay, and print the span comparison."""
    replay = verify_determinism()
    table = pd.read_csv(_root() / "comparison.csv", float_precision="round_trip")
    columns = [
        "vol_span_label",
        "cluster_net_return_annualized",
        "global_net_return_annualized",
        "cluster_sharpe_rf0",
        "global_sharpe_rf0",
        "delta_sharpe_rf0",
    ]
    print(table[columns].to_string(index=False))
    print(f"U3 volatility-span sweep: PASS ({len(replay)}/{len(replay)})")


if __name__ == "__main__":
    main()
