"""Run matched U1 cluster and global dollar-neutral momentum portfolios.

The curiosity test fixes the covariance winner (ME returns, EWMA span 36) and q=0.25.
Both legs target +100% long and -100% short.  The cluster leg ranks within the ME/36
partition and applies group-equal budgets separately to the long and short books; the
global leg applies the same top/bottom rule to one whole-universe group.  Assets selected
on both sides in an unrankable tiny/tied group are removed from both books, after which
each side is renormalised to one.  The implemented signed portfolio therefore has zero
target net exposure and gross exposure two on every rebalance date.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

import papers.cluster_lineage_2026.replication.run_backtests as e5
from papers.cluster_lineage_2026.replication.run_e5b import (
    GROUP_BUDGET_TOLERANCE,
    WEIGHT_TOLERANCE,
    _group_equal_from_ranks,
)
from papers.cluster_lineage_2026.replication.run_u1_covar_grid import (
    UNIVERSE,
    _accepted_dates_and_eligibility,
    _backtest,
    _ew_navs,
    _load_partition,
    _root as covariance_grid_root,
)


FREQUENCY = "ME"
SPAN = 36
Q = 0.25
RUNNER = "papers/cluster_lineage_2026/replication/run_u1_me36_long_short.py"
EXPOSURE_TOLERANCE = 1e-12


def _root() -> Path:
    """Return and create the local long-short curiosity output directory."""
    root = covariance_grid_root() / "long_short_ME_span_36_q_025"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _dollar_neutral_books(
    long_book: pd.DataFrame, short_book: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Remove side overlap and return unit long, unit short, signed weights, diagnostics."""
    long_book = long_book.copy()
    short_book = short_book.copy()
    overlap = long_book.gt(0.0) & short_book.gt(0.0)
    long_book = long_book.mask(overlap, 0.0)
    short_book = short_book.mask(overlap, 0.0)
    long_total = long_book.sum(axis=1)
    short_total = short_book.sum(axis=1)
    if long_total.le(0.0).any() or short_total.le(0.0).any():
        raise AssertionError("a long-short side is empty after overlap removal")
    long_book = long_book.div(long_total, axis=0)
    short_book = short_book.div(short_total, axis=0)
    signed = long_book - short_book
    diagnostics = pd.DataFrame(
        {
            "date": signed.index,
            "overlap_assets_removed": overlap.sum(axis=1).to_numpy(),
            "long_assets": long_book.gt(0.0).sum(axis=1).to_numpy(),
            "short_assets": short_book.gt(0.0).sum(axis=1).to_numpy(),
            "long_exposure": long_book.sum(axis=1).to_numpy(),
            "short_exposure_abs": short_book.sum(axis=1).to_numpy(),
            "net_exposure": signed.sum(axis=1).to_numpy(),
            "gross_exposure": signed.abs().sum(axis=1).to_numpy(),
        }
    )
    return long_book, short_book, signed, diagnostics


def _side_weights(
    scores: pd.DataFrame,
    eligibility: pd.DataFrame,
    groups: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build group-equal top and bottom books using the frozen selection rule."""
    long_ranks = e5._rank_panel(scores, groups)
    short_ranks = e5._rank_panel(-scores, groups)
    long_book, _, long_validation = _group_equal_from_ranks(
        long_ranks, eligibility, groups, Q, UNIVERSE
    )
    short_book, _, short_validation = _group_equal_from_ranks(
        short_ranks, eligibility, groups, Q, UNIVERSE
    )
    return long_book, short_book, long_validation, short_validation


def _leg_weights(
    scores: pd.DataFrame,
    eligibility: pd.DataFrame,
    groups: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return signed weights plus per-date exposure and side-budget diagnostics."""
    long_raw, short_raw, long_validation, short_validation = _side_weights(
        scores, eligibility, groups
    )
    _, _, signed, exposure = _dollar_neutral_books(long_raw, short_raw)
    side_validation = pd.DataFrame(
        {
            "date": signed.index,
            "long_pre_net_weight_error": long_validation[
                "weight_sum_abs_error"
            ].to_numpy(),
            "short_pre_net_weight_error": short_validation[
                "weight_sum_abs_error"
            ].to_numpy(),
            "long_pre_net_group_budget_error": long_validation[
                "max_group_budget_abs_error"
            ].to_numpy(),
            "short_pre_net_group_budget_error": short_validation[
                "max_group_budget_abs_error"
            ].to_numpy(),
        }
    )
    return signed, exposure, side_validation


def _acceptance(
    window: str,
    leg: str,
    exposure: pd.DataFrame,
    side_validation: pd.DataFrame,
) -> dict:
    """Return one exact exposure and pre-net group-budget acceptance row."""
    long_error = float((exposure["long_exposure"] - 1.0).abs().max())
    short_error = float((exposure["short_exposure_abs"] - 1.0).abs().max())
    net_error = float(exposure["net_exposure"].abs().max())
    gross_error = float((exposure["gross_exposure"] - 2.0).abs().max())
    weight_error = float(
        side_validation[
            ["long_pre_net_weight_error", "short_pre_net_weight_error"]
        ].to_numpy().max()
    )
    budget_error = float(
        side_validation[
            [
                "long_pre_net_group_budget_error",
                "short_pre_net_group_budget_error",
            ]
        ].to_numpy().max()
    )
    passed = (
        max(long_error, short_error, net_error, gross_error) <= EXPOSURE_TOLERANCE
        and weight_error <= WEIGHT_TOLERANCE
        and budget_error <= GROUP_BUDGET_TOLERANCE
    )
    return {
        "analysis_window": window,
        "leg": leg,
        "max_long_exposure_error": long_error,
        "max_short_exposure_error": short_error,
        "max_net_exposure_error": net_error,
        "max_gross_exposure_error": gross_error,
        "max_pre_net_weight_error": weight_error,
        "max_pre_net_group_budget_error": budget_error,
        "status": "PASS" if passed else "FAIL",
    }


def run() -> Mapping[str, pd.DataFrame]:
    """Execute the matched cluster and global long-short portfolios."""
    started = time.perf_counter()
    dates, fixed_eligibility = _accepted_dates_and_eligibility()
    windows = e5._analysis_windows(UNIVERSE, dates)
    data = e5.load_universe(UNIVERSE)
    prices = e5._prices(data)
    costs = e5.get_universe_spec(UNIVERSE).cost_bps / 10000.0
    ew_navs = _ew_navs()
    cluster_groups, _ = _load_partition(FREQUENCY, SPAN)
    performance_rows = []
    exposure_rows = []
    side_rows = []
    acceptance_rows = []

    for window, window_dates in windows.items():
        eligibility = fixed_eligibility.reindex(index=window_dates)
        scores = e5._raw_momentum_scores(
            data, window_dates, vol_adjusted=False
        ).reindex(columns=eligibility.columns).where(eligibility)
        groups = cluster_groups.reindex(
            index=window_dates, columns=eligibility.columns
        )
        global_groups = pd.DataFrame(
            "global", index=window_dates, columns=eligibility.columns
        )
        prices_window = prices.reindex(columns=eligibility.columns)
        for leg, leg_groups, construction in (
            ("cluster_ME_span_36", groups, "group_equal"),
            ("global", global_groups, "asset_equal"),
        ):
            weights, exposure, side_validation = _leg_weights(
                scores, eligibility, leg_groups
            )
            net, gross = _backtest(
                prices_window, weights, costs, f"{window}_{leg}_long_short"
            )
            performance_rows.append(
                {
                    "universe": UNIVERSE.value,
                    "analysis_window": window,
                    "frequency": FREQUENCY if leg != "global" else "global",
                    "span": SPAN if leg != "global" else np.nan,
                    "q": Q,
                    "construction": construction,
                    "strategy": "long_top_short_bottom",
                    "target_long_exposure": 1.0,
                    "target_short_exposure": -1.0,
                    "target_gross_exposure": 2.0,
                    "leg": leg,
                    **e5._performance_row(net, gross, ew_navs[window]),
                    "runner": RUNNER,
                }
            )
            exposure.insert(0, "leg", leg)
            exposure.insert(0, "analysis_window", window)
            exposure_rows.append(exposure)
            side_validation.insert(0, "leg", leg)
            side_validation.insert(0, "analysis_window", window)
            side_rows.append(side_validation)
            acceptance_rows.append(_acceptance(window, leg, exposure, side_validation))

    performance = pd.DataFrame(performance_rows)
    comparison_rows = []
    for window, panel in performance.groupby("analysis_window", sort=False):
        indexed = panel.set_index("leg")
        cluster = indexed.loc["cluster_ME_span_36"]
        global_row = indexed.loc["global"]
        row = {
            "analysis_window": window,
            "cluster_leg": "cluster_ME_span_36",
            "benchmark_leg": "global",
        }
        for metric in (
            "net_return_annualized",
            "volatility_annualized",
            "sharpe_rf0",
            "one_way_turnover_annualized",
            "cost_drag_bp_per_year",
            "alpha_vs_ew_annualized",
            "beta_vs_ew",
        ):
            row[f"cluster_{metric}"] = cluster[metric]
            row[f"global_{metric}"] = global_row[metric]
            row[f"delta_{metric}"] = cluster[metric] - global_row[metric]
        comparison_rows.append(row)
    acceptance = pd.DataFrame(acceptance_rows)
    if not acceptance["status"].eq("PASS").all():
        raise AssertionError(acceptance.loc[~acceptance["status"].eq("PASS")])
    output = {
        "performance": performance,
        "comparison": pd.DataFrame(comparison_rows),
        "exposure_diagnostics": pd.concat(exposure_rows, ignore_index=True),
        "side_budget_diagnostics": pd.concat(side_rows, ignore_index=True),
        "acceptance": acceptance,
        "runtime": pd.DataFrame(
            [{"runtime_seconds": time.perf_counter() - started, "runner": RUNNER}]
        ),
    }
    for name, frame in output.items():
        e5._write(frame, _root() / f"{name}.csv")
    return output


def _hash_outputs() -> dict[str, str]:
    """Hash numerical outputs while excluding timing and replay records."""
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(_root().glob("*.csv"))
        if path.name not in {"runtime.csv", "determinism.csv"}
    }


def verify_determinism() -> pd.DataFrame:
    """Run twice and require byte-identical long-short artifacts."""
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
    print(
        f"U1 ME/36 q=0.25 long-short: PASS "
        f"({len(result)}/{len(result)} deterministic)",
        flush=True,
    )
