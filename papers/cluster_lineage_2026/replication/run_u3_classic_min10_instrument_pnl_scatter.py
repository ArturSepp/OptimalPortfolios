"""Plot U3 classic-momentum instrument P&L for cluster versus global ranks."""
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


SIGNAL_ID = "classic_12m_ex_1m"
PLOT_FILE = "u3_classic_12m_ex_1m_min10_cluster_vs_global_instrument_pnl.html"
ACCOUNTING_TOLERANCE = 1e-10
PERFORMANCE_TOLERANCE = 1e-12
RUNNER = (
    "papers/cluster_lineage_2026/replication/"
    "run_u3_classic_min10_instrument_pnl_scatter.py"
)


def _root() -> Path:
    """Return the gitignored local attribution directory."""
    root = (
        Path(__file__).resolve().parents[1]
        / "local_outputs"
        / "e5b"
        / "u3_classic_12m_ex_1m_min10_instrument_pnl_20260816"
    )
    root.mkdir(parents=True, exist_ok=True)
    return root


def _portfolios_and_weights(
    signal_id: str = SIGNAL_ID,
) -> tuple[dict, dict[str, pd.DataFrame], Mapping]:
    """Build exact global and cluster portfolios for one accepted signal."""
    context = comparison._u3_context()
    global_scores, cluster_scores, signal_diagnostics = comparison._signal_pair(
        signal_id=signal_id,
        prices=context["signal_prices"],
        benchmark=context["benchmark"],
        groups=context["groups"],
        dates=context["dates"],
        eligibility=context["eligibility"],
    )
    weights = {}
    diagnostics = {}
    portfolios = {}
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
            f"U3_{signal_id}_{method}_min10_attribution",
        )
        weights[method] = method_weights
        diagnostics[method] = method_diagnostics
        portfolios[method] = net
        portfolios[f"{method}_gross"] = gross
    return portfolios, weights, {
        "context": context,
        "signal": signal_diagnostics,
        "weights": diagnostics,
    }


def run() -> Mapping[str, pd.DataFrame]:
    """Compute, validate, save, and plot exact per-instrument U3 net P&L."""
    started = time.perf_counter()
    portfolios, weights, diagnostics = _portfolios_and_weights()
    context = diagnostics["context"]
    eligibility = context["eligibility"]
    cluster_pnl, cluster_diag = prior._net_attribution(portfolios["cluster"])
    global_pnl, global_diag = prior._net_attribution(portfolios["global"])
    data = e5.load_universe(u3_equal.UNIVERSE)
    tickers = eligibility.columns[eligibility.any(axis=0)]
    sleeves = u3_equal._broad_sleeves(data.taxonomy, eligibility.columns)
    table = prior._instrument_table(
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

    cluster_sum_error = abs(
        float(table["cluster_net_pnl_pct_of_start"].sum())
        - 100.0 * float(cluster_diag["attributed_net_total_return"])
    )
    global_sum_error = abs(
        float(table["global_net_pnl_pct_of_start"].sum())
        - 100.0 * float(global_diag["attributed_net_total_return"])
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
    excluded_rows = int(
        table["ticker"].isin(e5.FUTURES_INVESTABILITY_EXCLUSIONS).sum()
    )
    accounting_error = max(
        float(cluster_diag["max_step_reconciliation_abs_error"]),
        float(global_diag["max_step_reconciliation_abs_error"]),
        float(cluster_diag["cumulative_reconciliation_abs_error"]),
        float(global_diag["cumulative_reconciliation_abs_error"]),
        cluster_sum_error,
        global_sum_error,
    )
    status = (
        accounting_error <= ACCOUNTING_TOLERANCE
        and performance_error <= PERFORMANCE_TOLERANCE
        and excluded_rows == 0
    )
    reconciliation = pd.DataFrame(
        [
            {
                "universe": "U3_futures",
                "analysis_window": context["window"],
                "signal_id": SIGNAL_ID,
                "min_cluster_size": comparison.MIN_CLUSTER_SIZE,
                "q": context["q"],
                "cost_bps_one_way": context["cost_bps"],
                "eligible_instruments": len(table),
                "cluster_higher_instruments": int(
                    table["cluster_minus_global_pnl_pct_of_start"].gt(0.0).sum()
                ),
                "global_higher_instruments": int(
                    table["cluster_minus_global_pnl_pct_of_start"].lt(0.0).sum()
                ),
                "equal_contribution_instruments": int(
                    table["cluster_minus_global_pnl_pct_of_start"].eq(0.0).sum()
                ),
                "contribution_correlation": float(
                    table[
                        [
                            "cluster_net_pnl_pct_of_start",
                            "global_net_pnl_pct_of_start",
                        ]
                    ].corr().iloc[0, 1]
                ),
                "cluster_attributed_total_return_pct": 100.0
                * float(cluster_diag["attributed_net_total_return"]),
                "global_attributed_total_return_pct": 100.0
                * float(global_diag["attributed_net_total_return"]),
                "cluster_table_sum_abs_error": cluster_sum_error,
                "global_table_sum_abs_error": global_sum_error,
                "maximum_accounting_error": accounting_error,
                "performance_regression_max_abs_error": performance_error,
                "owner_excluded_rows": excluded_rows,
                "status": "PASS" if status else "FAIL",
                "runner": RUNNER,
            }
        ]
    )
    if not status:
        raise AssertionError(reconciliation.to_dict(orient="records")[0])

    fig = prior._plot(table, reconciliation)
    summary = reconciliation.iloc[0]
    fig.update_layout(
        title={
            "text": (
                "U3 classic 12m-ex-1m: instrument net P&L"
                f"<br><sup>min cluster size 10 · contribution correlation "
                f"{summary['contribution_correlation']:.3f} · cluster higher for "
                f"{int(summary['cluster_higher_instruments'])}/"
                f"{int(summary['eligible_instruments'])} instruments</sup>"
            ),
            "x": 0.05,
            "xanchor": "left",
        }
    )
    fig.update_xaxes(
        title_text="Cluster-standardised net P&L<br>(pp of start NAV)"
    )
    fig.update_yaxes(title_text="Global-rank net P&L<br>(pp of start NAV)")
    outputs = {
        "instrument_pnl": table,
        "performance": performance,
        "reconciliation": reconciliation,
        "design": pd.DataFrame(
            [
                {
                    "x_axis": "cluster-strategy instrument net P&L, pp of start NAV",
                    "y_axis": "global-strategy instrument net P&L, pp of start NAV",
                    "identity_line": "y=x; below line means cluster contribution is higher",
                    "signal": "classic monthly 12m-ex-1m",
                    "min_cluster_size": comparison.MIN_CLUSTER_SIZE,
                    "cluster_role": "score standardisation only",
                    "q": context["q"],
                    "cost_bps_one_way": context["cost_bps"],
                    "sleeve_budgets": "Equity30|FixedIncome30|Commodities30|FX10",
                    "runner": RUNNER,
                }
            ]
        ),
    }
    for name, frame in outputs.items():
        e5._write(frame, _root() / f"{name}.csv")
    fig.write_html(
        _root() / PLOT_FILE,
        include_plotlyjs="cdn",
        full_html=True,
        config={"responsive": True, "displaylogo": False, "scrollZoom": False},
        div_id="u3-classic-min10-instrument-pnl-scatter",
    )
    e5._write(
        pd.DataFrame([{"runtime_seconds": time.perf_counter() - started}]),
        _root() / "runtime.csv",
    )
    return outputs


def _hash_outputs() -> dict[str, str]:
    """Hash numerical and Plotly outputs while excluding runtime and replay."""
    paths = sorted(_root().glob("*.csv")) + [_root() / PLOT_FILE]
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths
        if path.name not in {"runtime.csv", "determinism.csv"}
    }


def verify_determinism() -> pd.DataFrame:
    """Replay a completed attribution once and require byte-identical outputs."""
    required = {
        "design.csv",
        "instrument_pnl.csv",
        "performance.csv",
        "reconciliation.csv",
        PLOT_FILE,
    }
    if not required.issubset({path.name for path in _root().iterdir()}):
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
    """Run, replay, and print attribution reconciliation."""
    replay = verify_determinism()
    print(
        pd.read_csv(
            _root() / "reconciliation.csv", float_precision="round_trip"
        ).to_string(index=False)
    )
    print(f"U3 classic min10 P&L scatter: PASS ({len(replay)}/{len(replay)})")


if __name__ == "__main__":
    main()
