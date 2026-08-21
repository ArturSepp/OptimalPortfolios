"""Run the U1 covariance long-short grid with classic monthly 12-minus-1 momentum.

The alternative score is the sum of exactly 12 completed monthly log returns after
shifting the return panel by one full month.  Thus the most recent monthly return at every
formation date is excluded, and the portfolio is implemented with the accepted one-period
lag.  The score is raw: it is not ROSAA's benchmark-relative, EWMA-filtered,
volatility-normalised production momentum transform.

Everything else is frozen to the accepted q=0.25 covariance grid: point-in-time U1
membership and eligibility, the 28 cached unsmoothed partitions, group-equal long and
short budgets, gross exposure two, net exposure zero, ME decisions, and 10 bp costs.
Global rank remains the sole payoff benchmark.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from optimalportfolios.alphas import compute_classic_momentum_from_returns

import papers.cluster_lineage_2026.replication.run_backtests as e5
import papers.cluster_lineage_2026.replication.run_u1_covar_grid_long_short as weekly
import papers.cluster_lineage_2026.replication.run_u1_me36_long_short as single
from papers.cluster_lineage_2026.replication.run_u1_covar_grid import (
    UNIVERSE,
    _accepted_dates_and_eligibility,
    _cell_id,
    _cells,
    _ew_navs,
    _load_partition,
    _native_returns,
    _read_daily,
    _root as covariance_grid_root,
)


Q = 0.25
LOOKBACK_MONTHS = 12
SKIP_MONTHS = 1
SIGNAL_VARIANT = "classic_monthly_12m_skip1"
HEADLINE_WINDOW = "headline_20090831_20260630"
AVAILABLE_WINDOW = "monthly_available_20070831_20260731"
RUNNER = (
    "papers/cluster_lineage_2026/replication/"
    "run_u1_covar_grid_long_short_monthly.py"
)
SCORE_REGRESSION_TOLERANCE = 1e-14
WEEKLY_COMPARISON_METRICS = (
    "cluster_gross_return_annualized",
    "cluster_net_return_annualized",
    "delta_net_return_annualized",
    "cluster_volatility_annualized",
    "delta_volatility_annualized",
    "cluster_sharpe_rf0",
    "delta_sharpe_rf0",
    "cluster_one_way_turnover_annualized",
    "cluster_cost_drag_bp_per_year",
)


def _root() -> Path:
    """Return and create the local monthly-signal grid directory."""
    root = covariance_grid_root() / "long_short_grid_q_025_monthly_12m_skip1"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _classic_monthly_scores(
    monthly_log_returns: pd.DataFrame,
    dates: pd.DatetimeIndex,
    *,
    lookback_months: int = LOOKBACK_MONTHS,
    skip_months: int = SKIP_MONTHS,
) -> pd.DataFrame:
    """Delegate the article's frozen return panel to the package signal API."""
    scores = compute_classic_momentum_from_returns(
        returns=monthly_log_returns,
        lookback_periods=lookback_months,
        skip_periods=skip_months,
    )
    return scores.reindex(pd.DatetimeIndex(dates))


def _independent_score_regression(
    monthly_log_returns: pd.DataFrame,
    dates: pd.DatetimeIndex,
    scores: pd.DataFrame,
) -> pd.DataFrame:
    """Reconstruct every score by explicit history slicing and compare masks and values."""
    rows = []
    for date in dates:
        history = monthly_log_returns.loc[:date]
        stop = len(history) - SKIP_MONTHS
        start = stop - LOOKBACK_MONTHS
        if start < 0 or stop <= 0:
            score = pd.Series(np.nan, index=monthly_log_returns.columns)
        else:
            score = history.iloc[start:stop].sum(
                axis=0,
                min_count=LOOKBACK_MONTHS,
            )
        rows.append(score.rename(date))
    reference = pd.DataFrame(rows).reindex_like(scores)
    difference = scores.subtract(reference).abs().to_numpy()
    finite = difference[np.isfinite(difference)]
    max_error = float(finite.max()) if finite.size else 0.0
    nan_mask_match = bool(scores.isna().equals(reference.isna()))
    status = (
        "PASS"
        if max_error <= SCORE_REGRESSION_TOLERANCE and nan_mask_match
        else "FAIL"
    )
    regression = pd.DataFrame(
        [
            {
                "signal_variant": SIGNAL_VARIANT,
                "lookback_months_included": LOOKBACK_MONTHS,
                "skip_months": SKIP_MONTHS,
                "dates": len(dates),
                "assets": scores.shape[1],
                "max_abs_error": max_error,
                "tolerance": SCORE_REGRESSION_TOLERANCE,
                "nan_mask_match": nan_mask_match,
                "status": status,
            }
        ]
    )
    if status != "PASS":
        raise AssertionError(regression)
    return regression


def _run_leg(*args, **kwargs) -> tuple[dict, dict]:
    """Use the accepted long-short engine and relabel the signal and runner."""
    performance, acceptance = weekly._run_leg(*args, **kwargs)
    performance.update(
        {
            "signal_variant": SIGNAL_VARIANT,
            "momentum_frequency": "ME",
            "momentum_lookback_periods": LOOKBACK_MONTHS,
            "momentum_skip_periods": SKIP_MONTHS,
            "momentum_vol_adjusted": False,
            "runner": RUNNER,
        }
    )
    acceptance["signal_variant"] = SIGNAL_VARIANT
    return performance, acceptance


def _weekly_signal_comparison(monthly_comparison: pd.DataFrame) -> pd.DataFrame:
    """Compare signals on the common headline window only."""
    weekly_path = weekly._root() / "comparison_vs_global.csv"
    if not weekly_path.exists():
        weekly.run()
    weekly_comparison = pd.read_csv(weekly_path, float_precision="round_trip")
    monthly_comparison = monthly_comparison.loc[
        monthly_comparison["analysis_window"].eq(HEADLINE_WINDOW)
    ]
    weekly_comparison = weekly_comparison.loc[
        weekly_comparison["analysis_window"].eq(HEADLINE_WINDOW)
    ]
    keys = ["analysis_window", "frequency", "span", "cell_id"]
    monthly_indexed = monthly_comparison.set_index(keys)
    weekly_indexed = weekly_comparison.set_index(keys)
    if not monthly_indexed.index.equals(weekly_indexed.index):
        weekly_indexed = weekly_indexed.reindex(monthly_indexed.index)
    rows = []
    for key, monthly_row in monthly_indexed.iterrows():
        weekly_row = weekly_indexed.loc[key]
        row = dict(zip(keys, key))
        row.update(
            {
                "q": Q,
                "monthly_signal": SIGNAL_VARIANT,
                "weekly_signal": "raw_weekly_48w_skip4",
            }
        )
        for metric in WEEKLY_COMPARISON_METRICS:
            row[f"monthly_{metric}"] = monthly_row[metric]
            row[f"weekly_{metric}"] = weekly_row[metric]
            row[f"monthly_minus_weekly_{metric}"] = (
                monthly_row[metric] - weekly_row[metric]
            )
        rows.append(row)
    return pd.DataFrame(rows)


def run() -> Mapping[str, pd.DataFrame]:
    """Execute all cached covariance cells with monthly 12-minus-1 momentum."""
    started = time.perf_counter()
    dates, fixed_eligibility = _accepted_dates_and_eligibility()
    accepted_windows = e5._analysis_windows(UNIVERSE, dates)
    data = e5.load_universe(UNIVERSE)
    prices = e5._prices(data)
    costs = e5.get_universe_spec(UNIVERSE).cost_bps / 10000.0
    ew_navs = _ew_navs()
    daily_returns = _read_daily(fixed_eligibility.columns)
    monthly_returns = _native_returns(daily_returns, "ME")
    scores_all = _classic_monthly_scores(monthly_returns, dates)
    signal_regression = _independent_score_regression(
        monthly_returns,
        dates,
        scores_all,
    )
    eligible_score_counts = scores_all.where(fixed_eligibility).notna().sum(axis=1)
    available_dates = eligible_score_counts.loc[eligible_score_counts.gt(0.0)].index
    first_available_date = pd.Timestamp(available_dates.min())
    windows = {
        HEADLINE_WINDOW: accepted_windows[HEADLINE_WINDOW],
        AVAILABLE_WINDOW: dates[dates >= first_available_date],
    }
    ew_nav_by_window = {
        HEADLINE_WINDOW: ew_navs[HEADLINE_WINDOW],
        AVAILABLE_WINDOW: ew_navs["full_panel"],
    }
    signal_regression["first_available_date"] = first_available_date
    signal_regression["warmup_empty_dates"] = int(
        eligible_score_counts.loc[eligible_score_counts.eq(0.0)].size
    )
    inputs = {}
    performance_rows = []
    acceptance_rows = []
    risk_rows = []
    score_rows = []
    runtime_rows = []

    for window, window_dates in windows.items():
        eligibility = fixed_eligibility.reindex(index=window_dates)
        scores = scores_all.reindex(
            index=window_dates,
            columns=eligibility.columns,
        ).where(eligibility)
        valid_counts = scores.notna().sum(axis=1)
        score_rows.append(
            {
                "analysis_window": window,
                "signal_variant": SIGNAL_VARIANT,
                "dates": len(window_dates),
                "lookback_months_included": LOOKBACK_MONTHS,
                "skip_months": SKIP_MONTHS,
                "valid_assets_min": int(valid_counts.min()),
                "valid_assets_median": float(valid_counts.median()),
                "valid_assets_max": int(valid_counts.max()),
            }
        )
        prices_window = prices.reindex(columns=eligibility.columns)
        global_groups = pd.DataFrame(
            "global", index=window_dates, columns=eligibility.columns
        )
        global_weights, exposure, side_validation = single._leg_weights(
            scores,
            eligibility,
            global_groups,
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
            ew_nav_by_window[window],
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
                item["scores"],
                eligibility,
                groups,
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
                ew_nav_by_window[window],
            )
            performance_rows.append(performance)
            acceptance_rows.append(acceptance)
            risk = weekly._risk_diagnostic(
                window,
                frequency,
                span,
                weights,
                item["global_weights"],
                groups,
            )
            risk["signal_variant"] = SIGNAL_VARIANT
            risk_rows.append(risk)
        runtime_rows.append(
            {
                "frequency": frequency,
                "span": span,
                "cell_id": cell_id,
                "runtime_seconds": time.perf_counter() - cell_started,
            }
        )
        print(f"monthly long-short {frequency}/{span}: complete", flush=True)

    performance = pd.DataFrame(performance_rows).sort_values(
        ["analysis_window", "frequency", "span"],
        na_position="first",
    ).reset_index(drop=True)
    acceptance = pd.DataFrame(acceptance_rows)
    if not acceptance["status"].eq("PASS").all():
        raise AssertionError(acceptance.loc[~acceptance["status"].eq("PASS")])
    comparison = weekly._comparison(performance)
    comparison["signal_variant"] = SIGNAL_VARIANT
    comparison["momentum_frequency"] = "ME"
    comparison["momentum_lookback_periods"] = LOOKBACK_MONTHS
    comparison["momentum_skip_periods"] = SKIP_MONTHS
    rankings = weekly._rankings(comparison)
    weekly_comparison = _weekly_signal_comparison(comparison)
    runtime = pd.DataFrame(runtime_rows)
    runtime["total_run_seconds"] = time.perf_counter() - started
    output = {
        "performance": performance,
        "comparison_vs_global": comparison,
        "rankings": rankings,
        "comparison_vs_weekly_signal": weekly_comparison,
        "risk_diagnostics": pd.DataFrame(risk_rows),
        "score_diagnostics": pd.DataFrame(score_rows),
        "signal_regression": signal_regression,
        "acceptance": acceptance,
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
    """Run the complete deterministic monthly-signal long-short grid."""
    replay = verify_determinism()
    print(
        f"U1 q={Q:.2f} monthly 12m-skip-1 covariance grid: PASS "
        f"({len(replay)}/{len(replay)} deterministic)",
        flush=True,
    )


if __name__ == "__main__":
    main()
