"""Sweep U3 ROSAA short spans with 13-month volatility-normalized sizing.

The position volatility span is fixed at 282 daily observations, equal to 13
months under the TrendFollowingSystems annualization convention of 260 trading
days per year. The ROSAA monthly signal varies only ``short_span`` over None,
1, 2, and 3 while retaining long span 12, signal-volatility span 13, and EWMA
mean adjustment. All other filtered-U3 construction and cost settings are fixed.
"""
from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

import papers.cluster_lineage_2026.replication.run_backtests as e5
import papers.cluster_lineage_2026.replication.run_futures_sleeve_grid as u3_equal
from papers.cluster_lineage_2026.replication import (
    run_u2_u3_min_cluster10_signal_comparison as comparison,
)
import papers.cluster_lineage_2026.replication.run_u3_rosaa_min10_equal_contract as equal_contract
import papers.cluster_lineage_2026.replication.run_u3_rosaa_min10_vol_normalized as vol_normalized


SHORT_SPANS = (None, 1, 2, 3)
POSITION_VOL_SPAN_DAYS = round(13.0 * vol_normalized.ANNUALIZATION_FACTOR / 12.0)
RUNNER = (
    "papers/cluster_lineage_2026/replication/"
    "run_u3_rosaa_min10_short_span_sweep.py"
)


def _root() -> Path:
    """Return the gitignored short-span sweep directory."""
    root = (
        Path(__file__).resolve().parents[1]
        / "local_outputs"
        / "e5b"
        / "u3_rosaa_ra_min10_short_span_sweep_vol13m_20260816"
    )
    root.mkdir(parents=True, exist_ok=True)
    return root


def _label(short_span: int | None) -> str:
    """Return the stable short-span label."""
    return "None" if short_span is None else str(short_span)


def _comparison(performance: pd.DataFrame) -> pd.DataFrame:
    """Return cluster-minus-global metrics for each short span."""
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
    rows = []
    for label, frame in performance.groupby("short_span_label", sort=False):
        indexed = frame.set_index("method")
        row = {"short_span_label": label}
        for metric in metrics:
            cluster = float(indexed.loc["cluster", metric])
            global_rank = float(indexed.loc["global", metric])
            row[f"cluster_{metric}"] = cluster
            row[f"global_{metric}"] = global_rank
            row[f"delta_{metric}"] = cluster - global_rank
        rows.append(row)
    return pd.DataFrame(rows)


def run() -> Mapping[str, pd.DataFrame]:
    """Execute, validate, and save the ROSAA short-span sweep."""
    started = time.perf_counter()
    context = comparison._u3_context()
    scalers, annual_vols, vol_diagnostics = (
        vol_normalized._point_in_time_vol_scalers(
            context["dates"],
            context["eligibility"].columns,
            vol_span_days=POSITION_VOL_SPAN_DAYS,
        )
    )
    performance_rows = []
    acceptance_rows = []
    signal_rows = []
    stored_weights: dict[tuple[str, str], pd.DataFrame] = {}
    net_navs: dict[str, pd.Series] = {}
    for short_span in SHORT_SPANS:
        label = _label(short_span)
        global_scores, cluster_scores, signal_diagnostics = comparison._signal_pair(
            signal_id=vol_normalized.SIGNAL_ID,
            prices=context["signal_prices"],
            benchmark=context["benchmark"],
            groups=context["groups"],
            dates=context["dates"],
            eligibility=context["eligibility"],
            rosaa_short_span=short_span,
        )
        signal_rows.append({"short_span_label": label, **signal_diagnostics})
        for method, scores in (("global", global_scores), ("cluster", cluster_scores)):
            base_weights = equal_contract._equal_contract_weights(scores, context)[0]
            weights, diagnostics = vol_normalized._vol_normalized_weights(
                base_weights,
                scalers,
                annual_vols,
                context["eligibility"],
            )
            stored_weights[(label, method)] = weights
            passed = (
                diagnostics["max_weight_outside_eligibility"] <= comparison.TOLERANCE
                and diagnostics["max_overlap_assets"] == 0
                and vol_diagnostics["max_vol_source_lookahead_days"] <= 0
                and vol_diagnostics["uncapped_scaler_identity_max_abs_error"]
                <= comparison.TOLERANCE
            )
            acceptance_rows.append(
                {
                    "short_span": short_span,
                    "short_span_label": label,
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
                f"U3_shortspan_{label}_{method}",
            )
            net_navs[f"short_{label}__{method}"] = net.get_portfolio_nav().rename(
                f"short_{label}__{method}"
            )
            performance_rows.append(
                {
                    "short_span": short_span,
                    "short_span_label": label,
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
        "navs": pd.concat(net_navs, axis=1).rename_axis("date").reset_index(),
        "weights": pd.concat(
            [
                frame.reset_index(names="date").assign(
                    short_span_label=label,
                    method=method,
                )
                for (label, method), frame in stored_weights.items()
            ],
            ignore_index=True,
        ),
        "performance": performance,
        "comparison": comparison_table,
        "acceptance": acceptance,
        "signal_diagnostics": pd.DataFrame(signal_rows),
        "best_cells": performance.loc[
            performance.groupby("method")["sharpe_rf0"].idxmax()
        ].sort_values("method").reset_index(drop=True),
        "design": pd.DataFrame(
            [
                {
                    "short_spans": "None|1|2|3",
                    "position_vol_span_months": 13,
                    "position_vol_span_days": POSITION_VOL_SPAN_DAYS,
                    "signal_long_span_months": 12,
                    "signal_vol_span_months": 13,
                    "signal_mean_adjustment": "EWMA",
                    "instrument_vol_target": vol_normalized.INSTRUMENT_VOL_TARGET,
                    "instrument_weight_cap": vol_normalized.INSTRUMENT_WEIGHT_CAP,
                    "min_cluster_size": comparison.MIN_CLUSTER_SIZE,
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
    """Run, replay, and print the short-span comparison."""
    replay = verify_determinism()
    table = pd.read_csv(_root() / "comparison.csv", float_precision="round_trip")
    columns = [
        "short_span_label",
        "cluster_net_return_annualized",
        "global_net_return_annualized",
        "cluster_sharpe_rf0",
        "global_sharpe_rf0",
        "delta_sharpe_rf0",
    ]
    print(table[columns].to_string(index=False))
    print(f"U3 short-span sweep: PASS ({len(replay)}/{len(replay)})")


if __name__ == "__main__":
    main()
