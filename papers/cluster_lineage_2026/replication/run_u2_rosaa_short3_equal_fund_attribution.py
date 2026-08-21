"""Run all eligible U2 funds in one equal-weight ROSAA cross-section.

The complete point-in-time AUM100 BlackRock universe is ranked together with no
Equity/Fixed-Income/Rest construction sleeves. The signal matches the fixed U3
ROSAA specification: monthly long span 12, short span 3, volatility span 13,
EWMA mean adjustment, and cluster minimum size 10. Top and bottom quartiles are
equal weighted within each side. Official asset classes are used only for exact
ex-post P&L attribution. Rebalancing remains every two months with 20 bp costs.
"""
from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

import papers.cluster_lineage_2026.replication.run_backtests as e5
import papers.cluster_lineage_2026.replication.run_u2_all_funds_asset_class_attribution as attribution
import papers.cluster_lineage_2026.replication.run_u2_blackrock_etf_grid as u2_funds
import papers.cluster_lineage_2026.replication.run_u2_blackrock_sleeve_grid as u2_sleeves
import papers.cluster_lineage_2026.replication.run_u2_u3_min_cluster10_signal_comparison as comparison


SIGNAL_ID = "rosaa_risk_adjusted_momentum"
SHORT_SPAN = 3
BOOK = "equal_fund_single_cross_section"
ELIGIBILITY_LABEL = "all BlackRock funds passing point-in-time AUM100"
RANK_SCOPE = "all eligible funds together"
ASSET_CLASS_BUDGETS = "none"
TOLERANCE = 1e-10
RUNNER = (
    "papers/cluster_lineage_2026/replication/"
    "run_u2_rosaa_short3_equal_fund_attribution.py"
)


def _root() -> Path:
    """Return the gitignored equal-fund attribution directory."""
    root = (
        Path(__file__).resolve().parents[1]
        / "local_outputs"
        / "e5b"
        / "u2_rosaa_short3_min10_equal_fund_attribution_20260816"
    )
    root.mkdir(parents=True, exist_ok=True)
    return root


def _equal_fund_weights(
    scores: pd.DataFrame,
    context: Mapping[str, object],
) -> tuple[pd.DataFrame, Mapping[str, float]]:
    """Build one canonical equal-fund long-short book across all eligible funds."""
    all_funds = pd.DataFrame("All", index=scores.index, columns=scores.columns)
    weights, diagnostics = comparison._long_short_weights(
        scores=scores,
        prices=context["rank_prices"],
        eligibility=context["eligibility"],
        sleeve_panel=all_funds,
        sleeves=("All",),
        target={"All": 1.0},
        q=context["q"],
    )
    long_book = weights.clip(lower=0.0)
    short_book = weights.clip(upper=0.0).abs()
    long_range = long_book.where(long_book.gt(0.0)).max(axis=1).subtract(
        long_book.where(long_book.gt(0.0)).min(axis=1)
    )
    short_range = short_book.where(short_book.gt(0.0)).max(axis=1).subtract(
        short_book.where(short_book.gt(0.0)).min(axis=1)
    )
    return weights, {
        **diagnostics,
        "max_long_fund_weight_range": float(long_range.fillna(0.0).max()),
        "max_short_fund_weight_range": float(short_range.fillna(0.0).max()),
        "min_long_funds": int(long_book.gt(0.0).sum(axis=1).min()),
        "max_long_funds": int(long_book.gt(0.0).sum(axis=1).max()),
        "min_short_funds": int(short_book.gt(0.0).sum(axis=1).min()),
        "max_short_funds": int(short_book.gt(0.0).sum(axis=1).max()),
    }


def run() -> Mapping[str, pd.DataFrame]:
    """Run, reconcile, and save the equal-fund asset-class attribution."""
    started = time.perf_counter()
    context = comparison._u2_context()
    global_scores, cluster_scores, signal_diagnostics = comparison._signal_pair(
        signal_id=SIGNAL_ID,
        prices=context["signal_prices"],
        benchmark=context["benchmark"],
        groups=context["groups"],
        dates=context["dates"],
        eligibility=context["eligibility"],
        rosaa_short_span=SHORT_SPAN,
    )
    metadata = attribution.attr._metadata(context["eligibility"].columns)
    observed_classes = set(metadata["asset_class"].dropna().unique())
    if observed_classes != set(attribution.OFFICIAL_CLASSES):
        raise AssertionError(f"official asset classes changed: {observed_classes}")
    portfolios = {}
    weights = {}
    weight_rows = []
    performance_rows = []
    attribution_frames = []
    accounting_errors = []
    for method, scores in (("global", global_scores), ("cluster", cluster_scores)):
        method_weights, diagnostics = _equal_fund_weights(scores, context)
        maximum_error = max(
            float(value)
            for key, value in diagnostics.items()
            if "funds" not in key
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
        net, gross = u2_funds._backtest(
            context["performance_prices"],
            method_weights.reindex(index=context["scheduled_dates"]),
            context["cost_bps"] / 10000.0,
            f"U2_{BOOK}_{SIGNAL_ID}_short{SHORT_SPAN}_{method}",
        )
        portfolios[method] = net
        weights[method] = method_weights
        performance_rows.append(
            {
                "method": method,
                **u2_sleeves._performance_payload(net, gross, context["ew_nav"]),
            }
        )
        frame, diagnostics_pnl = attribution._instrument_attribution(
            net, method, metadata
        )
        attribution_frames.append(frame)
        accounting_errors.extend(
            [
                float(diagnostics_pnl["max_step_reconciliation_abs_error"]),
                float(diagnostics_pnl["cumulative_reconciliation_abs_error"]),
            ]
        )
    performance = pd.DataFrame(performance_rows)
    weight_diagnostics = pd.DataFrame(weight_rows)
    instrument_pnl = pd.concat(attribution_frames, ignore_index=True)
    asset_class_pnl = attribution._asset_class_pnl(instrument_pnl)
    asset_class_delta = attribution._asset_class_delta(asset_class_pnl)
    performance_indexed = performance.set_index("method")
    pnl_errors = []
    for method in ("global", "cluster"):
        attributed = float(
            asset_class_pnl.loc[
                asset_class_pnl["leg"].eq(method), "net_pnl_pct_of_start"
            ].sum()
        )
        portfolio_total = 100.0 * float(
            performance_indexed.loc[method, "net_total_return"]
        )
        pnl_errors.append(abs(attributed - portfolio_total))
    portfolio_gap = 100.0 * (
        float(performance_indexed.loc["cluster", "net_total_return"])
        - float(performance_indexed.loc["global", "net_total_return"])
    )
    attribution_gap = float(asset_class_delta["delta_net_pnl_pct_of_start"].sum())
    maximum_weight_error = float(weight_diagnostics["maximum_error"].max())
    maximum_accounting_error = max(accounting_errors + pnl_errors)
    passed = (
        weight_diagnostics["status"].eq("PASS").all()
        and maximum_accounting_error <= TOLERANCE
        and abs(portfolio_gap - attribution_gap) <= TOLERANCE
        and len(asset_class_pnl) == 2 * len(attribution.OFFICIAL_CLASSES)
    )
    reconciliation = pd.DataFrame(
        [
            {
                "universe": "U2_BlackRock_funds",
                "construction": BOOK,
                "signal_id": SIGNAL_ID,
                "short_span": SHORT_SPAN,
                "analysis_window": context["window"],
                "min_cluster_size": comparison.MIN_CLUSTER_SIZE,
                "q": context["q"],
                "cost_bps_one_way": context["cost_bps"],
                "funds_ever_eligible": int(context["eligibility"].any(axis=0).sum()),
                "maximum_weight_error": maximum_weight_error,
                "maximum_accounting_error": maximum_accounting_error,
                "cluster_global_gap_error": abs(portfolio_gap - attribution_gap),
                "status": "PASS" if passed else "FAIL",
                "runner": RUNNER,
            }
        ]
    )
    if not passed:
        raise AssertionError(reconciliation.to_dict(orient="records")[0])
    outputs = {
        "performance": performance,
        "asset_class_pnl": asset_class_pnl,
        "asset_class_delta": asset_class_delta,
        "instrument_pnl": instrument_pnl,
        "weight_diagnostics": weight_diagnostics,
        "signal_diagnostics": pd.DataFrame([signal_diagnostics]),
        "reconciliation": reconciliation,
        "design": pd.DataFrame(
            [
                {
                    "eligibility": ELIGIBILITY_LABEL,
                    "construction": BOOK,
                    "rank_scope": RANK_SCOPE,
                    "asset_class_budgets": ASSET_CLASS_BUDGETS,
                    "signal": SIGNAL_ID,
                    "long_span": 12,
                    "short_span": SHORT_SPAN,
                    "signal_vol_span": (
                        13 if SIGNAL_ID == "rosaa_risk_adjusted_momentum" else None
                    ),
                    "signal_mean_adjustment": (
                        "EWMA" if SIGNAL_ID == "rosaa_risk_adjusted_momentum" else "none"
                    ),
                    "min_cluster_size": comparison.MIN_CLUSTER_SIZE,
                    "schedule": context["schedule"],
                    "cost_bps_one_way": context["cost_bps"],
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
    """Replay the equal-fund attribution and require identical outputs."""
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
    """Run, replay, and print the asset-class delta."""
    replay = verify_determinism()
    table = pd.read_csv(_root() / "asset_class_delta.csv", float_precision="round_trip")
    print(table.to_string(index=False))
    print(
        f"U2 equal-fund {SIGNAL_ID} attribution: "
        f"PASS ({len(replay)}/{len(replay)})"
    )


if __name__ == "__main__":
    main()
