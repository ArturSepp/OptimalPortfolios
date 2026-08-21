"""Run exact ROSAA-production futures momentum with global and cluster ranks.

The signal is the production primitive used by the U1 and BlackRock U2 grids:
monthly returns, EWMA momentum span 12, EWMA volatility span 13, no short-term
reversal component, ``MeanAdjType.NONE``, a point-in-time eligible EW benchmark,
and the five-name production fallback for cluster scoring.

Every payoff leg is a +1/-1 long-short book with 30% Equity, 30% Fixed Income,
30% Commodities, and 10% FX on each signed side.  The matched methods are global
within sleeve, baseline correlation clusters, and M1-star correlation clusters.
The primary cost is 10 bp per one-way traded notional; an identical 20 bp replay is
cost sensitivity only.  CUA1 Comdty is owner-excluded throughout.
"""
from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import qis
from optimalportfolios.alphas.signals.momentum import compute_momentum_alpha
from optimalportfolios.alphas.signals.utils import score_within_clusters

import papers.cluster_lineage_2026.replication.run_backtests as e5
import papers.cluster_lineage_2026.replication.run_futures_asset_class_long_short as asset
import papers.cluster_lineage_2026.replication.run_futures_cluster_30303010_10bp as legacy
import papers.cluster_lineage_2026.replication.run_futures_sleeve_grid as equal
import papers.cluster_lineage_2026.replication.run_futures_weight_30303010 as construction
import papers.cluster_lineage_2026.replication.run_futures_weight_30303010_u1_window as matched
import papers.cluster_lineage_2026.replication.run_u1_covar_grid_long_short_prod as u1_prod


TARGET = dict(construction.TARGET)
COST_BPS = 10.0
REFERENCE_COST_BPS = 20.0
QUANTILES = tuple(equal.QUANTILES)
METHODS = tuple(legacy.METHODS)
SIGNAL_VARIANT = "rosaa_prod_exact_monthly_12m"
SIGNAL_FREQUENCY = "ME"
MOMENTUM_LONG_SPAN = 12
MOMENTUM_VOL_SPAN = 13
MOMENTUM_SHORT_SPAN = None
MOMENTUM_MEAN_ADJ_TYPE = "NONE"
MIN_CLUSTER_SIZE = 5
TOLERANCE = 1e-12
RECONSTRUCTION_TOLERANCE = 1e-15
COMPARISON_METRICS = tuple(equal.COMPARISON_METRICS)
FROZEN_NET_RETURN_AND_SHARPE = {
    (0.20, "sleeve_global"): (0.00985584795338856, 0.144522571566793),
    (0.20, "sleeve_cluster_baseline"): (
        -0.00401043151523417,
        -0.0838768862697678,
    ),
    (0.20, "sleeve_cluster_M1_star"): (
        -0.00595228289066041,
        -0.121833313564213,
    ),
    (0.25, "sleeve_global"): (0.00476973317858742, 0.0874349360558046),
    (0.25, "sleeve_cluster_baseline"): (
        -0.00282950116182423,
        -0.0589572953247675,
    ),
    (0.25, "sleeve_cluster_M1_star"): (
        -0.000206146181555567,
        0.00303457743455759,
    ),
}
RUNNER = (
    "papers/cluster_lineage_2026/replication/"
    "run_futures_prod_cluster_30303010_10bp.py"
)
DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "futures_log_returns.csv"


def _root() -> Path:
    """Return and create the external exact-production futures output directory."""
    return e5.get_output_path(
        "e5b", "futures_prod_cluster_30_30_30_10_10bp_u1_window", create=True
    )


def _read_daily(columns: pd.Index) -> pd.DataFrame:
    """Read frozen daily futures log returns on the accepted contract columns."""
    daily = pd.read_csv(
        DATA_PATH,
        index_col=0,
        parse_dates=True,
        float_precision="round_trip",
    )
    daily.index = pd.DatetimeIndex(daily.index, name="date")
    if daily.index.has_duplicates or not daily.index.is_monotonic_increasing:
        raise AssertionError("daily futures return index is not unique and sorted")
    return daily.reindex(columns=columns)


def _finite_max(frame: pd.DataFrame) -> float:
    """Return the maximum finite value in a numerical frame, or zero if empty."""
    values = frame.to_numpy()
    finite = values[np.isfinite(values)]
    return float(finite.max()) if finite.size else 0.0


def _signal_inputs(
    data,
    dates: pd.DatetimeIndex,
    eligibility: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    """Build exact monthly production global/raw signals and preflight diagnostics."""
    columns = eligibility.columns
    daily = _read_daily(columns)
    accepted_weekly = data.asset_returns["W-WED"].reindex(columns=columns)
    rebuilt_weekly = daily.resample("W-WED").sum(min_count=1).reindex_like(
        accepted_weekly
    )
    weekly_error = _finite_max(rebuilt_weekly.subtract(accepted_weekly).abs())
    weekly_nan_match = bool(rebuilt_weekly.isna().equals(accepted_weekly.isna()))

    monthly_log_returns = daily.resample(SIGNAL_FREQUENCY).sum(min_count=1)
    weekly_eligibility = e5._investable_eligibility(data, accepted_weekly.index)
    monthly_eligibility = weekly_eligibility.reindex(
        monthly_log_returns.index, method="ffill"
    ).reindex(columns=columns).fillna(False)
    simple_returns = np.expm1(monthly_log_returns)
    signal_prices = qis.returns_to_nav(simple_returns).reindex(columns=columns)
    benchmark_returns = simple_returns.where(monthly_eligibility).mean(
        axis=1, skipna=True
    )
    benchmark = qis.returns_to_nav(
        benchmark_returns.rename("eligible_EW").to_frame()
    )["eligible_EW"]
    global_source, raw_source = compute_momentum_alpha(
        prices=signal_prices,
        benchmark_price=benchmark,
        returns_freq=SIGNAL_FREQUENCY,
        group_data=None,
        long_span=MOMENTUM_LONG_SPAN,
        short_span=MOMENTUM_SHORT_SPAN,
        vol_span=MOMENTUM_VOL_SPAN,
        mean_adj_type=qis.MeanAdjType.NONE,
    )
    global_decision, timestamps = u1_prod._asof_panel(global_source, dates)
    raw_decision, raw_timestamps = u1_prod._asof_panel(raw_source, dates)
    timestamps_match = bool(timestamps.equals(raw_timestamps))
    global_decision = global_decision.reindex(columns=columns).where(eligibility)
    raw_decision = raw_decision.reindex(columns=columns).where(eligibility)
    valid_counts = global_decision.notna().sum(axis=1)
    available = valid_counts.loc[valid_counts.gt(0)]
    if available.empty:
        raise AssertionError("exact production signal has no eligible futures scores")

    monthly_roundtrip = qis.to_returns(
        signal_prices,
        freq=SIGNAL_FREQUENCY,
        is_log_returns=True,
    ).reindex_like(monthly_log_returns)
    monthly_error = _finite_max(
        monthly_roundtrip.subtract(monthly_log_returns).abs()
    )
    global_rebuilt = qis.df_to_cross_sectional_score(df=raw_source)
    global_score_error = _finite_max(global_rebuilt.subtract(global_source).abs())
    lookahead_days = timestamps.sub(timestamps.index).dt.days
    excluded = global_decision.columns.intersection(
        sorted(e5.FUTURES_INVESTABILITY_EXCLUSIONS)
    )
    excluded_valid_scores = int(global_decision.loc[:, excluded].notna().sum().sum())
    passed = (
        weekly_error <= RECONSTRUCTION_TOLERANCE
        and weekly_nan_match
        and monthly_error <= TOLERANCE
        and global_score_error <= TOLERANCE
        and timestamps_match
        and int(lookahead_days.max()) <= 0
        and excluded_valid_scores == 0
    )
    diagnostics = pd.DataFrame(
        [
            {
                "check": "exact_rosaa_production_signal_preflight",
                "signal_variant": SIGNAL_VARIANT,
                "signal_frequency": SIGNAL_FREQUENCY,
                "momentum_long_span": MOMENTUM_LONG_SPAN,
                "momentum_vol_span": MOMENTUM_VOL_SPAN,
                "momentum_short_span": np.nan,
                "momentum_mean_adj_type": MOMENTUM_MEAN_ADJ_TYPE,
                "momentum_benchmark": "point_in_time_eligible_EW",
                "minimum_cluster_size": MIN_CLUSTER_SIZE,
                "daily_to_wwed_max_abs_error": weekly_error,
                "daily_to_wwed_nan_pattern_match": weekly_nan_match,
                "monthly_return_roundtrip_max_abs_error": monthly_error,
                "global_score_reconstruction_max_abs_error": global_score_error,
                "global_raw_timestamp_match": timestamps_match,
                "max_signal_lookahead_days": int(lookahead_days.max()),
                "first_available_decision": available.index.min(),
                "valid_assets_min": int(valid_counts.min()),
                "valid_assets_median": float(valid_counts.median()),
                "valid_assets_max": int(valid_counts.max()),
                "owner_excluded_valid_scores": excluded_valid_scores,
                "reconstruction_tolerance": RECONSTRUCTION_TOLERANCE,
                "general_tolerance": TOLERANCE,
                "status": "PASS" if passed else "FAIL",
            }
        ]
    )
    if not passed:
        raise AssertionError(diagnostics)
    return global_decision, raw_source, timestamps, diagnostics


def _method_scores(
    global_scores: pd.DataFrame,
    raw_source: pd.DataFrame,
    timestamps: pd.Series,
    dates: pd.DatetimeIndex,
    eligibility: pd.DataFrame,
    groups_by_method: Mapping[str, pd.DataFrame],
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Apply the production rolling-cluster score with its five-name fallback."""
    scores_by_method = {"sleeve_global": global_scores}
    rows = []
    for method, groups in groups_by_method.items():
        if method == "sleeve_global":
            continue
        source = score_within_clusters(
            raw_signal=raw_source,
            rolling_clusters=u1_prod._panel_dict(groups),
            min_cluster_size=MIN_CLUSTER_SIZE,
        )
        decision, cluster_timestamps = u1_prod._asof_panel(source, dates)
        timestamp_match = bool(cluster_timestamps.equals(timestamps))
        decision = decision.reindex(columns=eligibility.columns).where(eligibility)
        valid_counts = decision.notna().sum(axis=1)
        scores_by_method[method] = decision
        rows.append(
            {
                "method": method,
                "cluster_score_timestamp_match": timestamp_match,
                "valid_assets_min": int(valid_counts.min()),
                "valid_assets_median": float(valid_counts.median()),
                "valid_assets_max": int(valid_counts.max()),
                "status": "PASS" if timestamp_match else "FAIL",
            }
        )
    diagnostics = pd.DataFrame(rows)
    if tuple(scores_by_method) != METHODS:
        raise AssertionError(f"unexpected score methods: {tuple(scores_by_method)}")
    if not diagnostics["status"].eq("PASS").all():
        raise AssertionError(diagnostics)
    return scores_by_method, diagnostics


def _run_one_cost(
    *,
    method: str,
    q: float,
    cost_bps: float,
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    diagnostics: Mapping[str, float],
    sleeve_panel: pd.DataFrame,
    ew_nav: pd.Series,
) -> tuple[dict, dict, list[dict], dict]:
    """Backtest one fixed exact-production decision panel at one cost rate."""
    records = matched._run_leg(
        strategy="long_short",
        method=method,
        q=q,
        prices=prices,
        weights=weights,
        diagnostics=diagnostics,
        sleeve_panel=sleeve_panel,
        ew_nav=ew_nav,
        costs=cost_bps / 10000.0,
        target=TARGET,
    )
    performance, acceptance, allocation, horizon = records
    performance.update(
        {
            "cost_bps_one_way": cost_bps,
            "signal_variant": SIGNAL_VARIANT,
            "signal_frequency": SIGNAL_FREQUENCY,
            "momentum_long_span": MOMENTUM_LONG_SPAN,
            "momentum_vol_span": MOMENTUM_VOL_SPAN,
            "momentum_short_span": np.nan,
            "momentum_mean_adj_type": MOMENTUM_MEAN_ADJ_TYPE,
            "momentum_benchmark": "point_in_time_eligible_EW",
            "momentum_min_cluster_size": MIN_CLUSTER_SIZE,
            "runner": RUNNER,
        }
    )
    acceptance["cost_bps_one_way"] = cost_bps
    acceptance["signal_variant"] = SIGNAL_VARIANT
    horizon["cost_bps_one_way"] = cost_bps
    return performance, acceptance, allocation, horizon


def _group_count_outputs(
    groups_by_method: Mapping[str, pd.DataFrame],
    scores_by_method: Mapping[str, pd.DataFrame],
    eligibility: pd.DataFrame,
    sleeve_panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return production-score available group counts and summary statistics."""
    rows = []
    for method, groups in groups_by_method.items():
        valid = scores_by_method[method].notna() & eligibility
        for sleeve in equal.SLEEVES:
            available = groups.where(valid & sleeve_panel.eq(sleeve)).nunique(
                axis=1, dropna=True
            )
            rows.extend(
                {
                    "date": date,
                    "method": method,
                    "sleeve": sleeve,
                    "available_groups": int(count),
                }
                for date, count in available.items()
            )
    counts = pd.DataFrame(rows)
    summary = (
        counts.groupby(["method", "sleeve"], sort=True)["available_groups"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
        .rename(columns={"std": "standard_deviation"})
    )
    return counts, summary


def _design(
    dates: pd.DatetimeIndex,
    eligibility: pd.DataFrame,
    scores_by_method: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Return the frozen exact-production experiment specification."""
    global_valid = scores_by_method["sleeve_global"].notna().sum(axis=1)
    return pd.DataFrame(
        [
            {
                "universe": equal.UNIVERSE.value,
                "analysis_window": matched.WINDOW,
                "strategy": "long_short",
                "signal_variant": SIGNAL_VARIANT,
                "signal_frequency": SIGNAL_FREQUENCY,
                "momentum_long_span": MOMENTUM_LONG_SPAN,
                "momentum_vol_span": MOMENTUM_VOL_SPAN,
                "momentum_short_span": np.nan,
                "momentum_mean_adj_type": MOMENTUM_MEAN_ADJ_TYPE,
                "momentum_benchmark": "point_in_time_eligible_EW",
                "momentum_min_cluster_size": MIN_CLUSTER_SIZE,
                "primary_cost_bps_one_way": COST_BPS,
                "reference_cost_bps_one_way": REFERENCE_COST_BPS,
                "q_values": "|".join(f"{q:.2f}" for q in QUANTILES),
                "decision_dates": len(dates),
                "decision_start": dates.min(),
                "decision_end": dates.max(),
                "implementation_lag_observations": 1,
                "construction": "group_equal within sleeve for cluster methods",
                "methods": "|".join(METHODS),
                "equity_budget_per_side": TARGET["Equity"],
                "fixed_income_budget_per_side": TARGET["Fixed Income"],
                "commodities_budget_per_side": TARGET["Commodities"],
                "fx_budget_per_side": TARGET["FX"],
                "eligible_futures_min": int(eligibility.sum(axis=1).min()),
                "eligible_futures_max": int(eligibility.sum(axis=1).max()),
                "valid_production_scores_min": int(global_valid.min()),
                "valid_production_scores_median": float(global_valid.median()),
                "valid_production_scores_max": int(global_valid.max()),
                "owner_exclusions": "|".join(
                    sorted(e5.FUTURES_INVESTABILITY_EXCLUSIONS)
                ),
                "supersedes_signal": "paper_raw_weekly_48w_skip_4w",
                "runner": RUNNER,
            }
        ]
    )


def run() -> Mapping[str, pd.DataFrame]:
    """Execute exact-production global, baseline, and M1-star futures books."""
    started = time.perf_counter()
    construction._validate_target()
    data = e5.load_universe(equal.UNIVERSE)
    dates = matched._window_dates(
        e5.load_cached(equal.UNIVERSE, e5.SmootherName.BASELINE).dates
    )
    eligibility = e5._investable_eligibility(data, dates)
    columns = eligibility.columns
    prices = matched._prices_with_context(e5._prices(data).reindex(columns=columns))
    sleeves = equal._broad_sleeves(data.taxonomy, columns)
    sleeve_panel = equal._sleeve_panel(dates, sleeves)
    groups_by_method = legacy._group_panels(dates, columns, sleeve_panel)
    global_scores, raw_source, timestamps, signal_diagnostics = _signal_inputs(
        data, dates, eligibility
    )
    scores_by_method, cluster_signal_diagnostics = _method_scores(
        global_scores,
        raw_source,
        timestamps,
        dates,
        eligibility,
        groups_by_method,
    )
    accepted_navs = pd.read_csv(
        equal._accepted_root() / "navs.csv",
        parse_dates=["date"],
        float_precision="round_trip",
    ).set_index("date")
    ew_nav = matched._bounded_panel(accepted_navs["EW_all"])
    if not isinstance(ew_nav, pd.Series):
        raise AssertionError("bounded EW reference is not a Series")

    performance_rows = []
    sensitivity_rows = []
    acceptance_rows = []
    allocation_rows = []
    horizon_rows = []
    reconstruction_rows = []
    for q in QUANTILES:
        for method, groups in groups_by_method.items():
            scores = scores_by_method[method]
            weights, diagnostics = construction._build_constrained_weights(
                "long_short",
                scores,
                eligibility,
                sleeve_panel,
                groups,
                q,
            )
            reconstructed = sum(
                asset._standalone_weights(
                    scores,
                    eligibility,
                    sleeve_panel,
                    groups,
                    sleeve,
                    q,
                )[0].mul(TARGET[sleeve])
                for sleeve in equal.SLEEVES
            )
            reconstruction_error = float(
                reconstructed.subtract(weights).abs().to_numpy().max()
            )
            reconstruction_rows.append(
                {
                    "method": method,
                    "q": q,
                    "max_weight_abs_error": reconstruction_error,
                    "tolerance": TOLERANCE,
                    "status": (
                        "PASS" if reconstruction_error <= TOLERANCE else "FAIL"
                    ),
                }
            )
            excluded = weights.columns.intersection(
                sorted(e5.FUTURES_INVESTABILITY_EXCLUSIONS)
            )
            exclusion_error = float(
                weights.loc[:, excluded].abs().to_numpy().max()
            )
            cost_records = {}
            for cost_bps in (COST_BPS, REFERENCE_COST_BPS):
                records = _run_one_cost(
                    method=method,
                    q=q,
                    cost_bps=cost_bps,
                    prices=prices,
                    weights=weights,
                    diagnostics=diagnostics,
                    sleeve_panel=sleeve_panel,
                    ew_nav=ew_nav,
                )
                performance, acceptance, allocation, horizon = records
                acceptance["max_owner_excluded_weight_abs"] = exclusion_error
                acceptance["status"] = (
                    "PASS"
                    if acceptance["status"] == "PASS"
                    and exclusion_error <= TOLERANCE
                    else "FAIL"
                )
                cost_records[cost_bps] = performance
                if cost_bps == COST_BPS:
                    performance_rows.append(performance)
                    acceptance_rows.append(acceptance)
                    allocation_rows.extend(allocation)
                    horizon_rows.append(horizon)

            primary = cost_records[COST_BPS]
            reference = cost_records[REFERENCE_COST_BPS]
            sensitivity_rows.append(
                {
                    "method": method,
                    "q": q,
                    "primary_cost_bps_one_way": COST_BPS,
                    "reference_cost_bps_one_way": REFERENCE_COST_BPS,
                    "gross_return_annualized": primary["gross_return_annualized"],
                    "net_return_annualized_10bp": primary["net_return_annualized"],
                    "net_return_annualized_20bp": reference["net_return_annualized"],
                    "net_return_improvement_10bp_vs_20bp": (
                        primary["net_return_annualized"]
                        - reference["net_return_annualized"]
                    ),
                    "sharpe_rf0_10bp": primary["sharpe_rf0"],
                    "sharpe_rf0_20bp": reference["sharpe_rf0"],
                    "cost_drag_bp_per_year_10bp": primary[
                        "cost_drag_bp_per_year"
                    ],
                    "cost_drag_bp_per_year_20bp": reference[
                        "cost_drag_bp_per_year"
                    ],
                }
            )

    performance = pd.DataFrame(performance_rows).sort_values(
        ["q", "method"]
    ).reset_index(drop=True)
    acceptance = pd.DataFrame(acceptance_rows).sort_values(
        ["q", "method"]
    ).reset_index(drop=True)
    reconstruction = pd.DataFrame(reconstruction_rows).sort_values(
        ["q", "method"]
    ).reset_index(drop=True)
    if not acceptance["status"].eq("PASS").all():
        raise AssertionError(acceptance.loc[~acceptance["status"].eq("PASS")])
    if not reconstruction["status"].eq("PASS").all():
        raise AssertionError(reconstruction.loc[~reconstruction["status"].eq("PASS")])
    group_counts, group_summary = _group_count_outputs(
        groups_by_method, scores_by_method, eligibility, sleeve_panel
    )
    comparison = legacy._comparison(performance)
    comparison.insert(0, "signal_variant", SIGNAL_VARIANT)
    outputs = {
        "design": _design(dates, eligibility, scores_by_method),
        "signal_diagnostics": signal_diagnostics,
        "cluster_signal_diagnostics": cluster_signal_diagnostics,
        "performance": performance,
        "comparison": comparison,
        "cost_sensitivity": pd.DataFrame(sensitivity_rows).sort_values(
            ["q", "method"]
        ),
        "acceptance": acceptance,
        "allocation_diagnostics": pd.DataFrame(allocation_rows),
        "horizon_diagnostic": pd.DataFrame(horizon_rows),
        "standalone_weight_reconstruction": reconstruction,
        "available_group_counts_by_date": group_counts,
        "available_group_count_summary": group_summary,
    }
    for name, frame in outputs.items():
        e5._write(frame, _root() / f"{name}.csv")
    e5._write(
        pd.DataFrame([{"runtime_seconds": time.perf_counter() - started}]),
        _root() / "runtime.csv",
    )
    return outputs


def _hash_outputs() -> dict[str, str]:
    """Hash deterministic outputs while excluding timing and replay records."""
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(_root().glob("*.csv"))
        if path.name not in {"runtime.csv", "determinism.csv"}
    }


def verify_determinism() -> pd.DataFrame:
    """Replay exact production at both costs and require identical outputs."""
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
    """Run, replay, and print the exact-production futures comparison."""
    replay = verify_determinism()
    performance = pd.read_csv(
        _root() / "performance.csv", float_precision="round_trip"
    )
    print(
        performance[
            [
                "q",
                "method",
                "gross_return_annualized",
                "net_return_annualized",
                "volatility_annualized",
                "sharpe_rf0",
                "one_way_turnover_annualized",
                "cost_drag_bp_per_year",
            ]
        ].to_string(index=False)
    )
    print(
        f"Futures exact ROSAA production 30/30/30/10 at 10 bp: PASS "
        f"({len(replay)}/{len(replay)} deterministic)",
        flush=True,
    )


if __name__ == "__main__":
    main()
