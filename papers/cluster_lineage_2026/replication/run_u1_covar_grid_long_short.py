"""Run the U1 q=0.25 long-short payoff over the accepted covariance grid.

Only the unsmoothed cluster partition changes across cells.  Every portfolio uses the
accepted point-in-time U1 eligibility mask, 48-week momentum with a four-week skip, ME
decisions, implementation lag one, and 10 bp costs.  Cluster books allocate equal budgets
to every available group separately on the long and short sides.  Global rank is the sole
payoff benchmark; EW-all is used only as the market reference for beta and alpha.

The grid contains 28 cached cells: B and W-MON through W-FRI at native EWMA spans 24,
36, 52, and 156; ME at spans 12, 24, 36, and 52.  No covariance model is refitted.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

import papers.cluster_lineage_2026.replication.run_backtests as e5
import papers.cluster_lineage_2026.replication.run_u1_me36_long_short as single
from papers.cluster_lineage_2026.replication.run_u1_covar_grid import (
    UNIVERSE,
    _accepted_dates_and_eligibility,
    _backtest,
    _cell_id,
    _cells,
    _ew_navs,
    _load_partition,
    _root as covariance_grid_root,
)


Q = 0.25
RUNNER = "papers/cluster_lineage_2026/replication/run_u1_covar_grid_long_short.py"
GROUP_NEUTRALITY_TOLERANCE = 1e-12
REGRESSION_TOLERANCE = 1e-12
COMPARISON_METRICS = (
    "gross_return_annualized",
    "net_return_annualized",
    "volatility_annualized",
    "sharpe_rf0",
    "one_way_turnover_annualized",
    "cost_drag_bp_per_year",
    "net_total_return",
    "alpha_vs_ew_annualized",
    "beta_vs_ew",
)
REGRESSION_METRICS = (
    "net_total_return",
    "net_return_annualized",
    "volatility_annualized",
    "sharpe_rf0",
    "alpha_vs_ew_annualized",
    "beta_vs_ew",
    "one_way_turnover_annualized",
    "cost_drag_bp_per_year",
)


def _root() -> Path:
    """Return and create the local q=0.25 long-short grid directory."""
    root = covariance_grid_root() / "long_short_grid_q_025"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _group_exposure_panel(weights: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    """Return per-date signed exposure remaining after aggregation to groups."""
    rows = []
    for date in weights.index:
        grouped = weights.loc[date].groupby(groups.loc[date], dropna=True).sum()
        rows.append(
            {
                "date": date,
                "group_l1_net_exposure": float(grouped.abs().sum()),
                "largest_abs_group_net_exposure": float(grouped.abs().max()),
            }
        )
    return pd.DataFrame(rows)


def _effective_assets(weights: pd.DataFrame, *, long_side: bool) -> pd.Series:
    """Return the inverse-Herfindahl effective asset count on one signed side."""
    side = weights.clip(lower=0.0) if long_side else weights.clip(upper=0.0)
    return 1.0 / side.pow(2.0).sum(axis=1)


def _risk_diagnostic(
    window: str,
    frequency: str,
    span: int,
    cluster_weights: pd.DataFrame,
    global_weights: pd.DataFrame,
    groups: pd.DataFrame,
) -> dict:
    """Summarise group neutrality and ordinary name diversification for one cell."""
    cluster_exposure = _group_exposure_panel(cluster_weights, groups)
    global_exposure = _group_exposure_panel(global_weights, groups)
    return {
        "analysis_window": window,
        "frequency": frequency,
        "span": span,
        "cell_id": _cell_id(frequency, span),
        "q": Q,
        "cluster_mean_group_l1_net_exposure": float(
            cluster_exposure["group_l1_net_exposure"].mean()
        ),
        "cluster_max_group_l1_net_exposure": float(
            cluster_exposure["group_l1_net_exposure"].max()
        ),
        "global_mean_group_l1_net_exposure": float(
            global_exposure["group_l1_net_exposure"].mean()
        ),
        "global_max_group_l1_net_exposure": float(
            global_exposure["group_l1_net_exposure"].max()
        ),
        "cluster_mean_largest_group_net_exposure": float(
            cluster_exposure["largest_abs_group_net_exposure"].mean()
        ),
        "global_mean_largest_group_net_exposure": float(
            global_exposure["largest_abs_group_net_exposure"].mean()
        ),
        "cluster_mean_effective_long_assets": float(
            _effective_assets(cluster_weights, long_side=True).mean()
        ),
        "cluster_mean_effective_short_assets": float(
            _effective_assets(cluster_weights, long_side=False).mean()
        ),
        "global_mean_effective_long_assets": float(
            _effective_assets(global_weights, long_side=True).mean()
        ),
        "global_mean_effective_short_assets": float(
            _effective_assets(global_weights, long_side=False).mean()
        ),
    }


def _performance_payload(net, gross, ew_nav) -> dict:
    """Return accepted performance metrics plus the explicit pre-cost annual return."""
    payload = e5._performance_row(net, gross, ew_nav)
    payload["gross_return_annualized"] = (
        payload["net_return_annualized"] + payload["cost_drag_bp_per_year"] / 10000.0
    )
    return payload


def _comparison(performance: pd.DataFrame) -> pd.DataFrame:
    """Compare each covariance cell only with the matched global q=0.25 leg."""
    global_rows = performance.loc[performance["leg"].eq("global")].set_index(
        "analysis_window"
    )
    rows = []
    clusters = performance.loc[~performance["leg"].eq("global")]
    for _, cluster in clusters.iterrows():
        global_row = global_rows.loc[cluster["analysis_window"]]
        row = {
            "analysis_window": cluster["analysis_window"],
            "frequency": cluster["frequency"],
            "span": cluster["span"],
            "span_unit": cluster["span_unit"],
            "q": cluster["q"],
            "construction": cluster["construction"],
            "leg": cluster["leg"],
            "cell_id": cluster["cell_id"],
            "benchmark_leg": "global",
        }
        for metric in COMPARISON_METRICS:
            row[f"cluster_{metric}"] = cluster[metric]
            row[f"global_{metric}"] = global_row[metric]
            row[f"delta_{metric}"] = cluster[metric] - global_row[metric]
        row["beats_global_net_return"] = (
            row["delta_net_return_annualized"] > 0.0
        )
        row["lower_volatility_than_global"] = (
            row["delta_volatility_annualized"] < 0.0
        )
        row["mean_variance_dominates_global"] = (
            row["beats_global_net_return"] and row["lower_volatility_than_global"]
        )
        row["beats_global_sharpe"] = row["delta_sharpe_rf0"] > 0.0
        row["beats_global_return_and_sharpe"] = (
            row["beats_global_net_return"] and row["beats_global_sharpe"]
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _rankings(comparison: pd.DataFrame) -> pd.DataFrame:
    """Rank cells by global-relative net return, then volatility and turnover."""
    ranked = comparison.copy()
    ranked["global_relative_return_rank"] = ranked.groupby("analysis_window")[
        "delta_net_return_annualized"
    ].rank(method="first", ascending=False)
    ranked["cluster_net_return_rank"] = ranked.groupby("analysis_window")[
        "cluster_net_return_annualized"
    ].rank(method="first", ascending=False)
    ranked["cluster_volatility_rank"] = ranked.groupby("analysis_window")[
        "cluster_volatility_annualized"
    ].rank(method="first", ascending=True)
    return ranked.sort_values(
        ["analysis_window", "delta_net_return_annualized", "cluster_volatility_annualized"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def _single_run_regression(performance: pd.DataFrame) -> pd.DataFrame:
    """Require the grid's ME/36 and global rows to reproduce the accepted curiosity run."""
    reference_path = single._root() / "performance.csv"
    if not reference_path.exists():
        single.run()
    reference = pd.read_csv(reference_path, float_precision="round_trip")
    rows = []
    for window in performance["analysis_window"].unique():
        for reference_leg, candidate_leg in (
            ("cluster_ME_span_36", "cluster_ME_span_036"),
            ("global", "global"),
        ):
            expected = reference.loc[
                reference["analysis_window"].eq(window)
                & reference["leg"].eq(reference_leg)
            ].iloc[0]
            actual = performance.loc[
                performance["analysis_window"].eq(window)
                & performance["leg"].eq(candidate_leg)
            ].iloc[0]
            errors = {
                metric: abs(float(actual[metric]) - float(expected[metric]))
                for metric in REGRESSION_METRICS
            }
            max_error = max(errors.values())
            rows.append(
                {
                    "analysis_window": window,
                    "reference_leg": reference_leg,
                    "candidate_leg": candidate_leg,
                    "max_metric_abs_error": max_error,
                    "tolerance": REGRESSION_TOLERANCE,
                    "details": "; ".join(
                        f"{metric}={error:.3e}" for metric, error in errors.items()
                    ),
                    "status": (
                        "PASS" if max_error <= REGRESSION_TOLERANCE else "FAIL"
                    ),
                }
            )
    regression = pd.DataFrame(rows)
    if not regression["status"].eq("PASS").all():
        raise AssertionError(regression.loc[~regression["status"].eq("PASS")])
    return regression


def _run_leg(
    window: str,
    frequency: str,
    span: float,
    cell_id: str,
    construction: str,
    leg: str,
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    exposure: pd.DataFrame,
    side_validation: pd.DataFrame,
    groups: pd.DataFrame,
    costs: float,
    ew_nav,
) -> tuple[dict, dict]:
    """Backtest one signed leg and return performance and exact acceptance rows."""
    net, gross = _backtest(prices, weights, costs, f"{window}_{cell_id}_long_short")
    performance = {
        "universe": UNIVERSE.value,
        "analysis_window": window,
        "frequency": frequency,
        "span": span,
        "span_unit": "not_applicable" if leg == "global" else "native_observations",
        "q": Q,
        "construction": construction,
        "strategy": "long_top_short_bottom",
        "target_long_exposure": 1.0,
        "target_short_exposure": -1.0,
        "target_gross_exposure": 2.0,
        "leg": leg,
        "cell_id": cell_id,
        **_performance_payload(net, gross, ew_nav),
        "runner": RUNNER,
    }
    acceptance = single._acceptance(window, leg, exposure, side_validation)
    post_net = _group_exposure_panel(weights, groups)
    neutrality_error = float(post_net["group_l1_net_exposure"].max())
    acceptance.update(
        {
            "frequency": frequency,
            "span": span,
            "cell_id": cell_id,
            "max_post_net_group_l1_exposure": neutrality_error,
            "post_net_group_l1_tolerance": GROUP_NEUTRALITY_TOLERANCE,
        }
    )
    if neutrality_error > GROUP_NEUTRALITY_TOLERANCE:
        acceptance["status"] = "FAIL"
    return performance, acceptance


def run() -> Mapping[str, pd.DataFrame]:
    """Execute all cached covariance cells and the invariant global long-short leg."""
    started = time.perf_counter()
    dates, fixed_eligibility = _accepted_dates_and_eligibility()
    windows = e5._analysis_windows(UNIVERSE, dates)
    data = e5.load_universe(UNIVERSE)
    prices = e5._prices(data)
    costs = e5.get_universe_spec(UNIVERSE).cost_bps / 10000.0
    ew_navs = _ew_navs()
    inputs = {}
    performance_rows = []
    acceptance_rows = []
    risk_rows = []
    runtime_rows = []

    for window, window_dates in windows.items():
        eligibility = fixed_eligibility.reindex(index=window_dates)
        scores = e5._raw_momentum_scores(
            data, window_dates, vol_adjusted=False
        ).reindex(columns=eligibility.columns).where(eligibility)
        prices_window = prices.reindex(columns=eligibility.columns)
        global_groups = pd.DataFrame(
            "global", index=window_dates, columns=eligibility.columns
        )
        global_weights, exposure, side_validation = single._leg_weights(
            scores, eligibility, global_groups
        )
        performance, acceptance = _run_leg(
            window,
            "BENCHMARK_INVARIANT",
            np.nan,
            "global",
            "asset_equal",
            "global",
            prices_window,
            global_weights,
            exposure,
            side_validation,
            global_groups,
            costs,
            ew_navs[window],
        )
        performance_rows.append(performance)
        acceptance_rows.append(acceptance)
        inputs[window] = {
            "eligibility": eligibility,
            "scores": scores,
            "prices": prices_window,
            "global_weights": global_weights,
        }

    for frequency, span in _cells():
        cell_started = time.perf_counter()
        groups_all, _ = _load_partition(frequency, span)
        cell_id = _cell_id(frequency, span)
        for window, window_dates in windows.items():
            item = inputs[window]
            eligibility = item["eligibility"]
            groups = groups_all.reindex(index=window_dates, columns=eligibility.columns)
            weights, exposure, side_validation = single._leg_weights(
                item["scores"], eligibility, groups
            )
            leg = f"cluster_{cell_id}"
            performance, acceptance = _run_leg(
                window,
                frequency,
                span,
                cell_id,
                "group_equal",
                leg,
                item["prices"],
                weights,
                exposure,
                side_validation,
                groups,
                costs,
                ew_navs[window],
            )
            performance_rows.append(performance)
            acceptance_rows.append(acceptance)
            risk_rows.append(
                _risk_diagnostic(
                    window,
                    frequency,
                    span,
                    weights,
                    item["global_weights"],
                    groups,
                )
            )
        runtime_rows.append(
            {
                "frequency": frequency,
                "span": span,
                "cell_id": cell_id,
                "runtime_seconds": time.perf_counter() - cell_started,
            }
        )
        print(f"long-short {frequency}/{span}: complete", flush=True)

    performance = pd.DataFrame(performance_rows).sort_values(
        ["analysis_window", "frequency", "span"], na_position="first"
    ).reset_index(drop=True)
    acceptance = pd.DataFrame(acceptance_rows)
    if not acceptance["status"].eq("PASS").all():
        raise AssertionError(acceptance.loc[~acceptance["status"].eq("PASS")])
    comparison = _comparison(performance)
    rankings = _rankings(comparison)
    regression = _single_run_regression(performance)
    runtime = pd.DataFrame(runtime_rows)
    runtime["total_run_seconds"] = time.perf_counter() - started
    output = {
        "performance": performance,
        "comparison_vs_global": comparison,
        "rankings": rankings,
        "risk_diagnostics": pd.DataFrame(risk_rows),
        "acceptance": acceptance,
        "regression": regression,
        "runtime": runtime,
    }
    for name, frame in output.items():
        e5._write(frame, _root() / f"{name}.csv")
    return output


def _hash_outputs() -> dict[str, str]:
    """Hash numerical artifacts while excluding timing and replay records."""
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(_root().glob("*.csv"))
        if path.name not in {"runtime.csv", "determinism.csv"}
    }


def verify_determinism() -> pd.DataFrame:
    """Run twice from cached partitions and require byte-identical artifacts."""
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
    """Run the complete deterministic q=0.25 long-short grid."""
    replay = verify_determinism()
    print(
        f"U1 q={Q:.2f} covariance long-short grid: PASS "
        f"({len(replay)}/{len(replay)} deterministic)",
        flush=True,
    )


if __name__ == "__main__":
    main()
