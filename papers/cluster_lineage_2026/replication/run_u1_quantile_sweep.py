"""Run the U1 group-equal momentum backtest over a frozen selection-fraction grid.

The sweep varies only ``q`` over 0.30, 0.25, 0.20, 0.15, and 0.10.  Scores,
partitions, costs, implementation lag, schedules, and group-equal construction are held
fixed.  Global rank and taxonomy rank are the two payoff yardsticks; EW-all is used only
as the alpha/beta market reference.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

import qis

import papers.cluster_lineage_2026.replication.run_backtests as e5
from papers.cluster_lineage_2026.replication.run_e5b import (
    GROUP_BUDGET_TOLERANCE,
    WEIGHT_TOLERANCE,
    _grouped_weights,
    _root as e5b_root,
)


QUANTILES = (0.30, 0.25, 0.20, 0.15, 0.10)
UNIVERSE = e5.UniverseName.MSCI_US
CONFIGS = (e5.SmootherName.BASELINE, e5.SmootherName.M1_DELTA_002)
RUNNER = "papers/cluster_lineage_2026/replication/run_u1_quantile_sweep.py"


def _root() -> Path:
    """Return and create the local quantile-sweep output directory."""
    root = e5b_root() / "quantile_sweep" / UNIVERSE.value
    root.mkdir(parents=True, exist_ok=True)
    return root


def _write(frame: pd.DataFrame, name: str) -> None:
    """Write one deterministic sweep artifact."""
    e5._write(frame, _root() / f"{name}.csv")


def _global_weights(
    scores: pd.DataFrame, eligibility: pd.DataFrame, q: float
) -> pd.DataFrame:
    """Return the unchanged asset-equal global-rank weights."""
    groups = pd.DataFrame("global", index=scores.index, columns=scores.columns)
    ranks = e5._rank_panel(scores, groups)
    return e5._weights_from_ranks(ranks, eligibility, q, UNIVERSE)


def _ew_weights(eligibility: pd.DataFrame) -> pd.DataFrame:
    """Return the unchanged EW-all market-reference weights."""
    return eligibility.astype(float).div(
        eligibility.sum(axis=1).replace(0, np.nan), axis=0
    ).fillna(0.0)


def _backtest(
    prices: pd.DataFrame, weights: pd.DataFrame, costs: float, ticker: str
):
    """Run matched net and gross model portfolios through qis."""
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
    """Add explicit cluster-leg deltas versus the two ranking yardsticks."""
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
        indexed = panel.set_index("leg")
        for leg in ("cluster_baseline", "cluster_M1_delta_0.02"):
            row = indexed.loc[leg].to_dict()
            row.update(
                {
                    "universe": UNIVERSE.value,
                    "analysis_window": window,
                    "q": q,
                    "construction": "group_equal",
                    "leg": leg,
                }
            )
            for yardstick in ("global", "taxonomy"):
                for metric in metrics:
                    row[f"{metric}_delta_vs_{yardstick}"] = (
                        indexed.loc[leg, metric] - indexed.loc[yardstick, metric]
                    )
            rows.append(row)
    return pd.DataFrame(rows)


def run() -> Mapping[str, pd.DataFrame]:
    """Execute the two-window U1 quantile sweep from unchanged E2 caches."""
    started = time.perf_counter()
    dates = e5.load_cached(UNIVERSE, e5.SmootherName.BASELINE).dates
    windows = e5._analysis_windows(UNIVERSE, dates)
    data = e5.load_universe(UNIVERSE)
    prices = e5._prices(data)
    costs = e5.get_universe_spec(UNIVERSE).cost_bps / 10000.0

    performance_rows = []
    diagnostic_rows = []
    acceptance_rows = []
    ew_rows = []
    for window, window_dates in windows.items():
        eligibility = e5._investable_eligibility(data, window_dates)
        prices_window = prices.reindex(columns=eligibility.columns)
        scores = e5._raw_momentum_scores(
            data, window_dates, vol_adjusted=False
        ).reindex(columns=eligibility.columns).where(eligibility)

        ew = _ew_weights(eligibility)
        ew_net, ew_gross = _backtest(prices_window, ew, costs, f"EW_all_{window}")
        benchmark_nav = ew_net.get_portfolio_nav()
        ew_rows.append(
            {
                "universe": UNIVERSE.value,
                "analysis_window": window,
                "role": "alpha_beta_market_reference_only",
                **e5._performance_row(ew_net, ew_gross, benchmark_nav),
            }
        )

        for q in QUANTILES:
            grouped, _, counts, validations = _grouped_weights(
                data, window_dates, scores, eligibility, q, CONFIGS
            )
            weights = {"global": _global_weights(scores, eligibility, q), **grouped}
            for leg, frame in weights.items():
                net, gross = _backtest(
                    prices_window, frame, costs, f"q_{q:.2f}_{window}_{leg}"
                )
                performance_rows.append(
                    {
                        "universe": UNIVERSE.value,
                        "analysis_window": window,
                        "q": q,
                        "construction": (
                            "asset_equal" if leg == "global" else "group_equal"
                        ),
                        "leg": leg,
                        **e5._performance_row(net, gross, benchmark_nav),
                        "runner": RUNNER,
                    }
                )
                selected = frame.gt(0.0).sum(axis=1)
                diagnostic = {
                    "universe": UNIVERSE.value,
                    "analysis_window": window,
                    "q": q,
                    "leg": leg,
                    "mean_selected_assets": float(selected.mean()),
                    "min_selected_assets": int(selected.min()),
                    "max_selected_assets": int(selected.max()),
                    "max_weight_sum_abs_error": float(
                        frame.sum(axis=1).sub(1.0).abs().max()
                    ),
                }
                if leg in counts:
                    diagnostic["mean_available_groups"] = float(counts[leg].mean())
                    diagnostic["available_group_count_std"] = float(counts[leg].std())
                    diagnostic["max_group_budget_abs_error"] = float(
                        validations[leg]["max_group_budget_abs_error"].max()
                    )
                else:
                    diagnostic["mean_available_groups"] = 1.0
                    diagnostic["available_group_count_std"] = 0.0
                    diagnostic["max_group_budget_abs_error"] = 0.0
                diagnostic_rows.append(diagnostic)
                acceptance_rows.append(
                    {
                        "universe": UNIVERSE.value,
                        "analysis_window": window,
                        "q": q,
                        "leg": leg,
                        "weight_sum_error": diagnostic["max_weight_sum_abs_error"],
                        "weight_sum_tolerance": WEIGHT_TOLERANCE,
                        "group_budget_error": diagnostic[
                            "max_group_budget_abs_error"
                        ],
                        "group_budget_tolerance": GROUP_BUDGET_TOLERANCE,
                        "status": "PASS"
                        if diagnostic["max_weight_sum_abs_error"] <= WEIGHT_TOLERANCE
                        and diagnostic["max_group_budget_abs_error"]
                        <= GROUP_BUDGET_TOLERANCE
                        else "FAIL",
                    }
                )

    performance = pd.DataFrame(performance_rows)
    comparison = _comparison(performance)
    diagnostics = pd.DataFrame(diagnostic_rows)
    acceptance = pd.DataFrame(acceptance_rows)
    runtime = pd.DataFrame(
        [
            {
                "universe": UNIVERSE.value,
                "quantiles": "0.30|0.25|0.20|0.15|0.10",
                "windows": 2,
                "portfolio_rows": len(performance),
                "runtime_seconds": time.perf_counter() - started,
                "runner": RUNNER,
            }
        ]
    )
    if not acceptance["status"].eq("PASS").all():
        raise AssertionError(acceptance.loc[~acceptance["status"].eq("PASS")])
    output = {
        "performance": performance,
        "comparison": comparison,
        "selection_diagnostics": diagnostics,
        "acceptance": acceptance,
        "ew_reference": pd.DataFrame(ew_rows),
        "runtime": runtime,
    }
    for name, frame in output.items():
        _write(frame, name)
    return output


def _hash_outputs() -> dict[str, str]:
    """Return hashes of all result CSVs except the replay record."""
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(_root().glob("*.csv"))
        if path.name not in {"determinism.csv", "runtime.csv"}
    }


def verify_determinism() -> pd.DataFrame:
    """Run the sweep twice and assert byte-identical result CSVs."""
    run()
    first = _hash_outputs()
    run()
    second = _hash_outputs()
    names = sorted(set(first) | set(second))
    result = pd.DataFrame(
        {
            "artifact": names,
            "first_sha256": [first.get(name) for name in names],
            "second_sha256": [second.get(name) for name in names],
            "byte_identical": [first.get(name) == second.get(name) for name in names],
        }
    )
    _write(result, "determinism")
    if not result["byte_identical"].all():
        raise AssertionError(result.loc[~result["byte_identical"]])
    return result


if __name__ == "__main__":
    replay = verify_determinism()
    print(f"U1 quantile sweep: PASS ({len(replay)}/{len(replay)} artifacts deterministic)")
