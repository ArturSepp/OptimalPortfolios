"""Run the original-universe U1 cluster grid against global momentum rank.

The grid crosses five selection fractions with every estimated E2 cluster config.  It
uses the primary group-equal cluster construction, the original point-in-time eligible
universe, 10 bp costs, and lag 1.  Global rank is the sole payoff yardstick; EW-all is
consumed only as the market reference needed for alpha and beta columns.
"""
from __future__ import annotations

import hashlib
import time
import warnings
from pathlib import Path
from typing import Mapping

import pandas as pd

import qis

import papers.cluster_lineage_2026.replication.run_backtests as e5
from papers.cluster_lineage_2026.replication.run_e5b import (
    GROUP_BUDGET_TOLERANCE,
    WEIGHT_TOLERANCE,
    _group_equal_from_ranks,
    _root as e5b_root,
)
from papers.cluster_lineage_2026.replication.run_u1_quantile_sweep import QUANTILES


UNIVERSE = e5.UniverseName.MSCI_US
CONFIGS = (
    e5.SmootherName.BASELINE,
    e5.SmootherName.M0_QUARTERLY_HOLD,
    e5.SmootherName.M1_DELTA_002,
    e5.SmootherName.M1_DELTA_005,
    e5.SmootherName.M1_DELTA_010,
    e5.SmootherName.M2_LAMBDA_05,
    e5.SmootherName.M2_LAMBDA_07,
    e5.SmootherName.M1_STAR,
)
RUNNER = "papers/cluster_lineage_2026/replication/run_u1_global_grid.py"
HEADLINE = "headline_20090831_20260630"


def _root() -> Path:
    """Return and create the U1 global-grid output directory."""
    root = e5b_root() / "global_benchmark_grid" / UNIVERSE.value
    root.mkdir(parents=True, exist_ok=True)
    return root


def _fidelity_status(window: str, config: e5.SmootherName) -> str:
    """Return the owner-frozen E3b fidelity status for one window/config."""
    if config == e5.SmootherName.BASELINE:
        return "REFERENCE"
    in_band_both = {
        e5.SmootherName.M0_QUARTERLY_HOLD,
        e5.SmootherName.M1_DELTA_002,
        e5.SmootherName.M2_LAMBDA_05,
        e5.SmootherName.M2_LAMBDA_07,
    }
    if config in in_band_both:
        return "IN_BAND"
    if config == e5.SmootherName.M1_DELTA_005 and window == "full_panel":
        return "IN_BAND_FULL_ONLY"
    return "REJECTED_FIDELITY"


def _accepted_global_rows() -> pd.DataFrame:
    """Read deterministic global rows from the completed U1 q sweep."""
    path = e5b_root() / "quantile_sweep" / UNIVERSE.value / "performance.csv"
    frame = pd.read_csv(path, float_precision="round_trip")
    frame = frame.loc[frame["leg"].eq("global")].copy()
    frame["config"] = "global"
    frame["fidelity_status"] = "GLOBAL_BENCHMARK"
    frame["runner"] = RUNNER
    return frame


def _ew_navs() -> dict[str, pd.Series]:
    """Read accepted EW NAVs solely for alpha/beta calculations."""
    path = e5b_root() / "group_equal" / UNIVERSE.value / "navs.csv"
    frame = pd.read_csv(path, parse_dates=["date"], float_precision="round_trip")
    return {
        window: panel.set_index("date")["EW_all"].sort_index()
        for window, panel in frame.groupby("analysis_window", sort=False)
    }


def _backtest(
    prices: pd.DataFrame, weights: pd.DataFrame, costs: float, ticker: str
):
    """Run the frozen net and gross qis paths, preserving cash conventions."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        net = qis.backtest_model_portfolio(
            prices=prices,
            weights=weights,
            rebalancing_freq=None,
            rebalancing_costs=costs,
            weight_implementation_lag=1,
            ticker=ticker,
        )
        gross = qis.backtest_model_portfolio(
            prices=prices,
            weights=weights,
            rebalancing_freq=None,
            rebalancing_costs=None,
            weight_implementation_lag=1,
            ticker=f"{ticker}_gross",
        )
    return net, gross


def _comparison(performance: pd.DataFrame) -> pd.DataFrame:
    """Add cluster-minus-global metrics and payoff verdicts."""
    metrics = (
        "net_return_annualized",
        "sharpe_rf0",
        "one_way_turnover_annualized",
        "cost_drag_bp_per_year",
    )
    rows = []
    for (window, q), panel in performance.groupby(
        ["analysis_window", "q"], sort=False
    ):
        global_row = panel.loc[panel["leg"].eq("global")].iloc[0]
        for _, cluster in panel.loc[panel["leg"].ne("global")].iterrows():
            row = cluster.to_dict()
            for metric in metrics:
                row[f"{metric}_delta_vs_global"] = (
                    cluster[metric] - global_row[metric]
                )
            row["beats_global_net_return"] = (
                row["net_return_annualized_delta_vs_global"] > 0.0
            )
            row["beats_global_sharpe"] = row["sharpe_rf0_delta_vs_global"] > 0.0
            row["beats_global_both"] = (
                row["beats_global_net_return"] and row["beats_global_sharpe"]
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _rankings(comparison: pd.DataFrame) -> pd.DataFrame:
    """Rank all and fidelity-admissible cluster configurations within each window."""
    rows = []
    for window, panel in comparison.groupby("analysis_window", sort=False):
        admissible = panel["fidelity_status"].isin(
            ["REFERENCE", "IN_BAND", "IN_BAND_FULL_ONLY"]
        )
        for scope, subset in (
            ("all_configs", panel),
            ("fidelity_admissible", panel.loc[admissible]),
        ):
            ranked = subset.sort_values(
                ["sharpe_rf0", "net_return_annualized", "one_way_turnover_annualized"],
                ascending=[False, False, True],
            ).reset_index(drop=True)
            ranked.insert(0, "rank", range(1, len(ranked) + 1))
            ranked.insert(0, "scope", scope)
            rows.append(ranked)
    return pd.concat(rows, ignore_index=True)


def _config_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    """Select each config's best q by Sharpe in each analysis window."""
    rows = []
    for (window, config), panel in comparison.groupby(
        ["analysis_window", "config"], sort=False
    ):
        best = panel.sort_values(
            ["sharpe_rf0", "net_return_annualized", "one_way_turnover_annualized"],
            ascending=[False, False, True],
        ).iloc[0]
        rows.append(
            {
                "analysis_window": window,
                "config": config,
                "fidelity_status": best["fidelity_status"],
                "best_q_by_sharpe": best["q"],
                "net_return_annualized": best["net_return_annualized"],
                "sharpe_rf0": best["sharpe_rf0"],
                "one_way_turnover_annualized": best[
                    "one_way_turnover_annualized"
                ],
                "cost_drag_bp_per_year": best["cost_drag_bp_per_year"],
                "net_return_delta_vs_global": best[
                    "net_return_annualized_delta_vs_global"
                ],
                "sharpe_delta_vs_global": best["sharpe_rf0_delta_vs_global"],
                "turnover_delta_vs_global": best[
                    "one_way_turnover_annualized_delta_vs_global"
                ],
                "beats_global_both": best["beats_global_both"],
            }
        )
    return pd.DataFrame(rows)


def run() -> Mapping[str, pd.DataFrame]:
    """Execute the cached original-universe U1 config-by-q grid."""
    started = time.perf_counter()
    all_dates = e5.load_cached(UNIVERSE, e5.SmootherName.BASELINE).dates
    windows = e5._analysis_windows(UNIVERSE, all_dates)
    data = e5.load_universe(UNIVERSE)
    prices = e5._prices(data)
    costs = e5.get_universe_spec(UNIVERSE).cost_bps / 10000.0
    ew_navs = _ew_navs()
    rows = []
    diagnostics = []
    acceptance = []

    for window, dates in windows.items():
        eligibility = e5._investable_eligibility(data, dates)
        prices_window = prices.reindex(columns=eligibility.columns)
        scores = e5._raw_momentum_scores(
            data, dates, vol_adjusted=False
        ).reindex(columns=eligibility.columns).where(eligibility)
        group_panels = {
            config: e5._cluster_groups(UNIVERSE, config).reindex(
                index=dates, columns=eligibility.columns
            )
            for config in CONFIGS
        }
        for q in QUANTILES:
            for config, groups in group_panels.items():
                ranks = e5._rank_panel(scores, groups)
                weights, counts, validation = _group_equal_from_ranks(
                    ranks, eligibility, groups, q, UNIVERSE
                )
                leg = f"cluster_{config.value}"
                net, gross = _backtest(
                    prices_window, weights, costs, f"{window}_q_{q:.2f}_{leg}"
                )
                rows.append(
                    {
                        "universe": UNIVERSE.value,
                        "analysis_window": window,
                        "q": q,
                        "construction": "group_equal",
                        "leg": leg,
                        "config": config.value,
                        "fidelity_status": _fidelity_status(window, config),
                        **e5._performance_row(net, gross, ew_navs[window]),
                        "runner": RUNNER,
                    }
                )
                selected = weights.gt(0.0).sum(axis=1)
                effective = 1.0 / weights.pow(2).sum(axis=1)
                max_weight_error = float(weights.sum(axis=1).sub(1.0).abs().max())
                max_budget_error = float(
                    validation["max_group_budget_abs_error"].max()
                )
                diagnostics.append(
                    {
                        "analysis_window": window,
                        "q": q,
                        "config": config.value,
                        "mean_groups": float(counts.mean()),
                        "group_count_std": float(counts.std()),
                        "mean_selected_assets": float(selected.mean()),
                        "mean_effective_holdings": float(effective.mean()),
                        "max_weight_sum_abs_error": max_weight_error,
                        "max_group_budget_abs_error": max_budget_error,
                    }
                )
                acceptance.append(
                    {
                        "analysis_window": window,
                        "q": q,
                        "config": config.value,
                        "weight_sum_error": max_weight_error,
                        "weight_sum_tolerance": WEIGHT_TOLERANCE,
                        "group_budget_error": max_budget_error,
                        "group_budget_tolerance": GROUP_BUDGET_TOLERANCE,
                        "status": "PASS"
                        if max_weight_error <= WEIGHT_TOLERANCE
                        and max_budget_error <= GROUP_BUDGET_TOLERANCE
                        else "FAIL",
                    }
                )

    cluster_performance = pd.DataFrame(rows)
    global_performance = _accepted_global_rows()
    performance = pd.concat(
        [global_performance, cluster_performance], ignore_index=True, sort=False
    ).sort_values(["analysis_window", "q", "leg"], ascending=[True, False, True])
    comparison = _comparison(performance)
    rankings = _rankings(comparison)
    config_summary = _config_summary(comparison)
    acceptance_frame = pd.DataFrame(acceptance)
    if not acceptance_frame["status"].eq("PASS").all():
        raise AssertionError(
            acceptance_frame.loc[~acceptance_frame["status"].eq("PASS")]
        )
    runtime = pd.DataFrame(
        [
            {
                "universe": UNIVERSE.value,
                "configs": len(CONFIGS),
                "quantiles": len(QUANTILES),
                "windows": len(windows),
                "cluster_backtests": len(cluster_performance),
                "runtime_seconds": time.perf_counter() - started,
                "runner": RUNNER,
            }
        ]
    )
    output = {
        "performance": performance.reset_index(drop=True),
        "comparison_vs_global": comparison.reset_index(drop=True),
        "rankings": rankings.reset_index(drop=True),
        "config_summary": config_summary.reset_index(drop=True),
        "construction_diagnostics": pd.DataFrame(diagnostics),
        "acceptance": acceptance_frame,
        "runtime": runtime,
    }
    for name, frame in output.items():
        e5._write(frame, _root() / f"{name}.csv")
    return output


def _hash_outputs() -> dict[str, str]:
    """Hash every numerical result artifact except timing and replay records."""
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(_root().glob("*.csv"))
        if path.name not in {"runtime.csv", "determinism.csv"}
    }


def verify_determinism() -> pd.DataFrame:
    """Replay the full grid and require byte-identical numerical artifacts."""
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


if __name__ == "__main__":
    result = verify_determinism()
    print(f"U1 global-benchmark grid: PASS ({len(result)}/{len(result)} deterministic)")
