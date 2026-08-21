"""Compare classic and ROSAA short spans for the U2 55/35/10 book.

The grid holds the AUM50 universe, covariance partitions, q=25%, minimum
cluster size 10, long-only 55/35/10 sleeve construction, schedule, and 20 bp
costs fixed. It varies only the public OptimalPortfolios signal definition.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path

import pandas as pd

import papers.cluster_lineage_2026.replication.run_backtests as e5
import papers.cluster_lineage_2026.replication.run_u2_blackrock_etf_grid as funds
import papers.cluster_lineage_2026.replication.run_u2_blackrock_sleeve_grid as sleeves
import papers.cluster_lineage_2026.replication.run_u2_rosaa_long_only_55_35_10_aum50 as spec
import papers.cluster_lineage_2026.replication.run_u2_u3_min_cluster10_signal_comparison as signals


RUNNER = (
    "papers/cluster_lineage_2026/replication/"
    "run_u2_55_35_10_signal_grid.py"
)
SIGNAL_GRID = (
    ("classic_12m_ex_1m", "classic_12m_ex_1m", None),
    ("rosaa_short_none", "rosaa_risk_adjusted_momentum", None),
    ("rosaa_short_1", "rosaa_risk_adjusted_momentum", 1),
    ("rosaa_short_2", "rosaa_risk_adjusted_momentum", 2),
    ("rosaa_short_3", "rosaa_risk_adjusted_momentum", 3),
)
TOLERANCE = 1e-12


def _root() -> Path:
    """Return the gitignored signal-grid output directory."""
    root = (
        Path(__file__).resolve().parents[1]
        / "local_outputs"
        / "e5b"
        / "u2_aum50_E55_F35_R10_classic_rosaa_short_grid_20260816"
    )
    root.mkdir(parents=True, exist_ok=True)
    return root


def _comparison(performance: pd.DataFrame) -> pd.DataFrame:
    """Return matched cluster-minus-global rows for every signal."""
    metrics = (
        "net_total_return",
        "net_return_annualized",
        "volatility_annualized",
        "sharpe_rf0",
        "one_way_turnover_annualized",
        "cost_drag_bp_per_year",
        "gross_return_annualized",
    )
    rows = []
    for signal_variant, frame in performance.groupby("signal_variant", sort=False):
        indexed = frame.set_index("method")
        row = {
            "signal_variant": signal_variant,
            "signal_id": frame["signal_id"].iloc[0],
            "short_span": frame["short_span"].iloc[0],
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
    """Execute the matched signal grid and validate its frozen short-3 cell."""
    started = time.perf_counter()
    context = spec._context()
    performance_rows = []
    signal_rows = []
    weight_rows = []
    stored_weights: dict[tuple[str, str], pd.DataFrame] = {}
    net_navs: dict[str, pd.Series] = {}
    for variant, signal_id, short_span in SIGNAL_GRID:
        global_scores, cluster_scores, signal_diagnostics = signals._signal_pair(
            signal_id=signal_id,
            prices=context["signal_prices"],
            benchmark=context["benchmark"],
            groups=context["groups"],
            dates=context["dates"],
            eligibility=context["eligibility"],
            rosaa_short_span=short_span,
        )
        signal_rows.append({"signal_variant": variant, **signal_diagnostics})
        for method, scores in (("global", global_scores), ("cluster", cluster_scores)):
            weights, diagnostics = spec._weights(scores, context)
            stored_weights[(variant, method)] = weights
            maximum_error = max(
                abs(float(value))
                for key, value in diagnostics.items()
                if "funds" not in key
            )
            weight_rows.append(
                {
                    "signal_variant": variant,
                    "method": method,
                    **diagnostics,
                    "maximum_error": maximum_error,
                    "status": "PASS" if maximum_error <= TOLERANCE else "FAIL",
                }
            )
            net, gross = funds._backtest(
                context["performance_prices"],
                weights.reindex(index=context["scheduled_dates"]),
                context["cost_bps"] / 10000.0,
                f"U2_AUM50_E55_F35_R10_{variant}_{method}",
            )
            net_navs[f"{variant}__{method}"] = net.get_portfolio_nav().rename(
                f"{variant}__{method}"
            )
            performance_rows.append(
                {
                    "signal_variant": variant,
                    "signal_id": signal_id,
                    "short_span": short_span,
                    "method": method,
                    **sleeves._performance_payload(net, gross, context["ew_nav"]),
                }
            )
    performance = pd.DataFrame(performance_rows)
    signal_diagnostics = pd.DataFrame(signal_rows)
    weight_diagnostics = pd.DataFrame(weight_rows)
    comparison = _comparison(performance)

    prior = pd.read_csv(spec._root() / "performance.csv", float_precision="round_trip")
    current = performance.loc[performance["signal_variant"].eq("rosaa_short_3")]
    prior = prior.set_index("method")
    current = current.set_index("method")
    regression_columns = (
        "net_total_return",
        "net_return_annualized",
        "volatility_annualized",
        "sharpe_rf0",
        "one_way_turnover_annualized",
        "cost_drag_bp_per_year",
        "gross_return_annualized",
    )
    regression_error = max(
        abs(float(current.loc[method, metric]) - float(prior.loc[method, metric]))
        for method in ("global", "cluster")
        for metric in regression_columns
    )
    max_lookahead = float(
        signal_diagnostics[
            ["max_global_lookahead_days", "max_cluster_lookahead_days"]
        ].to_numpy().max()
    )
    max_raw_error = float(signal_diagnostics["raw_panel_max_abs_error"].max())
    max_weight_error = float(weight_diagnostics["maximum_error"].max())
    acceptance = pd.DataFrame(
        [
            {
                "check": "performance rows",
                "measured": len(performance),
                "tolerance": 2 * len(SIGNAL_GRID),
                "status": "PASS" if len(performance) == 2 * len(SIGNAL_GRID) else "FAIL",
            },
            {
                "check": "maximum signal lookahead days",
                "measured": max_lookahead,
                "tolerance": 0.0,
                "status": "PASS" if max_lookahead <= 0.0 else "FAIL",
            },
            {
                "check": "maximum raw-panel mismatch",
                "measured": max_raw_error,
                "tolerance": 0.0,
                "status": "PASS" if max_raw_error <= 0.0 else "FAIL",
            },
            {
                "check": "maximum weight and sleeve-budget error",
                "measured": max_weight_error,
                "tolerance": TOLERANCE,
                "status": "PASS" if max_weight_error <= TOLERANCE else "FAIL",
            },
            {
                "check": "ROSAA short-3 prior-run regression error",
                "measured": regression_error,
                "tolerance": TOLERANCE,
                "status": "PASS" if regression_error <= TOLERANCE else "FAIL",
            },
        ]
    )
    if not acceptance["status"].eq("PASS").all():
        raise AssertionError(acceptance)
    outputs = {
        "navs": pd.concat(net_navs, axis=1).rename_axis("date").reset_index(),
        "weights": pd.concat(
            [
                frame.reset_index(names="date").assign(
                    signal_variant=variant,
                    method=method,
                )
                for (variant, method), frame in stored_weights.items()
            ],
            ignore_index=True,
        ),
        "performance": performance,
        "comparison": comparison,
        "signal_diagnostics": signal_diagnostics,
        "weight_diagnostics": weight_diagnostics,
        "acceptance": acceptance,
        "specification": pd.DataFrame(
            [
                {
                    "aum_cutoff_usd_millions": 50,
                    "sleeve_weights": "Equity 0.55; Fixed Income 0.35; Rest 0.10",
                    "strategy": "long_only",
                    "q": context["q"],
                    "min_cluster_size": signals.MIN_CLUSTER_SIZE,
                    "cost_bps_one_way": context["cost_bps"],
                    "schedule": context["schedule"],
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
    """Hash deterministic grid outputs."""
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(_root().glob("*.csv"))
        if path.name not in {"runtime.csv", "determinism.csv"}
    }


def verify_determinism() -> pd.DataFrame:
    """Replay the grid and require byte-identical results."""
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
    """Run the grid replay and print matched comparisons."""
    replay = verify_determinism()
    table = pd.read_csv(_root() / "comparison.csv", float_precision="round_trip")
    print(table.to_string(index=False))
    print(f"U2 55/35/10 signal grid: PASS ({len(replay)}/{len(replay)})")


if __name__ == "__main__":
    main()
