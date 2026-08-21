"""Run U1 production-momentum long-short portfolios over the covariance grid.

Two explicitly labelled signal variants are reported.  prod_exact_monthly_12m is the
faithful U1 application of the validated E8b production primitive: monthly returns, EWMA
long span 12, EWMA volatility span 13, no short filter, MeanAdjType.NONE, and the
production five-name cluster fallback.  prod_calendar_scaled_12m keeps those mechanics
but follows each covariance cell's return cadence, using B=252, weekly=52, and ME=12 for
the long filter.  Its volatility spans are B=273, weekly=56, and ME=13, preserving the
production setting's approximately 13-month volatility horizon.

Every cluster cell is compared only with a global rank formed from the same signal
variant and cadence.  Point-in-time U1 membership, q=0.25, group-equal cluster budgets,
gross exposure two, net exposure zero, ME decisions, implementation lag one, 10 bp costs,
and the 28 cached unsmoothed partitions remain frozen.  EW-all is only the beta/alpha
market reference.
"""
from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import qis
from optimalportfolios.alphas.signals.momentum import compute_momentum_alpha
from optimalportfolios.alphas.signals.utils import score_within_clusters

import papers.cluster_lineage_2026.replication.run_backtests as e5
import papers.cluster_lineage_2026.replication.run_u1_covar_grid_long_short as weekly
import papers.cluster_lineage_2026.replication.run_u1_covar_grid_long_short_monthly as classic
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
EXACT_VARIANT = "prod_exact_monthly_12m"
SCALED_VARIANT = "prod_calendar_scaled_12m"
VARIANTS = (EXACT_VARIANT, SCALED_VARIANT)
HEADLINE_WINDOW = "headline_20090831_20260630"
AVAILABLE_WINDOW = "production_common_available"
MIN_CLUSTER_SIZE = 5
RUNNER = (
    "papers/cluster_lineage_2026/replication/"
    "run_u1_covar_grid_long_short_prod.py"
)
LONG_SPANS: Mapping[str, int] = {
    "B": 252,
    "W-MON": 52,
    "W-TUE": 52,
    "W-WED": 52,
    "W-THU": 52,
    "W-FRI": 52,
    "ME": 12,
}
VOL_SPANS: Mapping[str, int] = {
    "B": 273,
    "W-MON": 56,
    "W-TUE": 56,
    "W-WED": 56,
    "W-THU": 56,
    "W-FRI": 56,
    "ME": 13,
}
PERIODS_PER_YEAR: Mapping[str, int] = {
    "B": 252,
    "W-MON": 52,
    "W-TUE": 52,
    "W-WED": 52,
    "W-THU": 52,
    "W-FRI": 52,
    "ME": 12,
}
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
SIGNAL_COMPARISON_METRICS = (
    "cluster_net_return_annualized",
    "delta_net_return_annualized",
    "cluster_volatility_annualized",
    "delta_volatility_annualized",
    "cluster_sharpe_rf0",
    "delta_sharpe_rf0",
    "cluster_one_way_turnover_annualized",
    "cluster_cost_drag_bp_per_year",
)
REGRESSION_TOLERANCE = 1e-12


@dataclass(frozen=True)
class SignalSpec:
    """Define one production-momentum observation cadence and EWMA horizons."""

    variant: str
    frequency: str
    long_span: int
    vol_span: int

    @property
    def cache_key(self) -> tuple[str, int, int]:
        """Return the numerical-input key shared by equivalent variants."""
        return self.frequency, self.long_span, self.vol_span


def _root() -> Path:
    """Return and create the production-momentum grid output directory."""
    root = covariance_grid_root() / "long_short_grid_q_025_prod_12m"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _signal_spec(variant: str, covariance_frequency: str) -> SignalSpec:
    """Return exact-monthly or cadence-scaled production signal settings."""
    if variant == EXACT_VARIANT:
        return SignalSpec(variant, "ME", LONG_SPANS["ME"], VOL_SPANS["ME"])
    if variant == SCALED_VARIANT:
        return SignalSpec(
            variant,
            covariance_frequency,
            LONG_SPANS[covariance_frequency],
            VOL_SPANS[covariance_frequency],
        )
    raise KeyError(f"unknown production signal variant: {variant}")


def _signal_parameter_table() -> pd.DataFrame:
    """Return every covariance-cadence to signal-cadence mapping."""
    rows = []
    for variant in VARIANTS:
        for frequency in LONG_SPANS:
            spec = _signal_spec(variant, frequency)
            periods = PERIODS_PER_YEAR[spec.frequency]
            rows.append(
                {
                    "signal_variant": variant,
                    "covariance_frequency": frequency,
                    "signal_frequency": spec.frequency,
                    "long_span": spec.long_span,
                    "long_horizon_months": 12.0 * spec.long_span / periods,
                    "vol_span": spec.vol_span,
                    "vol_horizon_months": 12.0 * spec.vol_span / periods,
                    "short_span": np.nan,
                    "mean_adj_type": "NONE",
                    "benchmark": "point_in_time_eligible_EW",
                    "min_cluster_size": MIN_CLUSTER_SIZE,
                }
            )
    return pd.DataFrame(rows)


def _period_inputs(
    data,
    daily_log_returns: pd.DataFrame,
    frequency: str,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.DataFrame]:
    """Build cadence-native asset and point-in-time EW benchmark NAVs."""
    log_returns = _native_returns(daily_log_returns, frequency)
    source_eligibility = data.eligibility["W-WED"]
    eligibility = source_eligibility.reindex(log_returns.index, method="ffill")
    eligibility = eligibility.reindex(columns=log_returns.columns).fillna(False)
    investable = data.asset_roles.index[
        data.asset_roles["role"].eq("universe_member")
    ]
    eligibility = eligibility & eligibility.columns.isin(investable)
    simple_returns = np.expm1(log_returns)
    benchmark_returns = simple_returns.where(eligibility).mean(axis=1, skipna=True)
    prices = qis.returns_to_nav(simple_returns).reindex(columns=log_returns.columns)
    benchmark = qis.returns_to_nav(
        benchmark_returns.rename("EW").to_frame()
    )["EW"]
    return prices, benchmark, log_returns, eligibility


def _asof_panel(
    panel: pd.DataFrame,
    dates: pd.DatetimeIndex,
) -> tuple[pd.DataFrame, pd.Series]:
    """Sample each score at the latest source timestamp not after its decision."""
    source = panel.sort_index()
    dates = pd.DatetimeIndex(dates)
    positions = source.index.searchsorted(dates, side="right") - 1
    if np.any(positions < 0):
        raise AssertionError("a decision date precedes the first signal timestamp")
    timestamps = pd.Series(
        source.index.take(positions),
        index=dates,
        name="signal_timestamp",
    )
    sampled = source.iloc[positions].copy()
    sampled.index = dates
    return sampled, timestamps


def _panel_dict(panel: pd.DataFrame) -> dict[pd.Timestamp, pd.Series]:
    """Convert a membership panel to the rolling-cluster API representation."""
    return {
        pd.Timestamp(date): row.dropna()
        for date, row in panel.iterrows()
    }


def _base_signals(
    prices: pd.DataFrame,
    benchmark: pd.Series,
    spec: SignalSpec,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute production global scores and their shared raw momentum panel."""
    return compute_momentum_alpha(
        prices=prices,
        benchmark_price=benchmark,
        returns_freq=spec.frequency,
        group_data=None,
        long_span=spec.long_span,
        short_span=None,
        vol_span=spec.vol_span,
        mean_adj_type=qis.MeanAdjType.NONE,
    )


def _run_leg(
    *args,
    signal_variant: str,
    signal_frequency: str,
    long_span: int,
    vol_span: int,
    **kwargs,
) -> tuple[dict, dict]:
    """Use the accepted long-short engine and attach production provenance."""
    performance, acceptance = weekly._run_leg(*args, **kwargs)
    performance.update(
        {
            "signal_variant": signal_variant,
            "signal_frequency": signal_frequency,
            "momentum_long_span": long_span,
            "momentum_vol_span": vol_span,
            "momentum_short_span": np.nan,
            "momentum_mean_adj_type": "NONE",
            "momentum_benchmark": "point_in_time_eligible_EW",
            "momentum_min_cluster_size": MIN_CLUSTER_SIZE,
            "runner": RUNNER,
        }
    )
    acceptance.update(
        {
            "signal_variant": signal_variant,
            "signal_frequency": signal_frequency,
        }
    )
    return performance, acceptance


def _comparison(performance: pd.DataFrame) -> pd.DataFrame:
    """Compare every cluster only with its same-variant, same-cadence global leg."""
    globals_frame = performance.loc[performance["leg"].str.startswith("global_")]
    clusters = performance.loc[performance["leg"].str.startswith("cluster_")]
    global_index = globals_frame.set_index(
        ["analysis_window", "signal_variant", "signal_frequency"]
    )
    rows = []
    for _, cluster in clusters.iterrows():
        key = (
            cluster["analysis_window"],
            cluster["signal_variant"],
            cluster["signal_frequency"],
        )
        global_row = global_index.loc[key]
        row = {
            "analysis_window": cluster["analysis_window"],
            "signal_variant": cluster["signal_variant"],
            "signal_frequency": cluster["signal_frequency"],
            "frequency": cluster["frequency"],
            "span": cluster["span"],
            "cell_id": cluster["cell_id"],
            "q": Q,
            "cluster_leg": cluster["leg"],
            "benchmark_leg": global_row["leg"],
        }
        for metric in COMPARISON_METRICS:
            row[f"cluster_{metric}"] = cluster[metric]
            row[f"global_{metric}"] = global_row[metric]
            row[f"delta_{metric}"] = cluster[metric] - global_row[metric]
        row["beats_global_net_return"] = row["delta_net_return_annualized"] > 0.0
        row["lower_volatility_than_global"] = (
            row["delta_volatility_annualized"] < 0.0
        )
        row["mean_variance_dominates_global"] = (
            row["beats_global_net_return"]
            and row["lower_volatility_than_global"]
        )
        row["beats_global_sharpe"] = row["delta_sharpe_rf0"] > 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def _other_signal_comparison(production: pd.DataFrame) -> pd.DataFrame:
    """Join headline production results to the accepted classic and weekly grids."""
    classic_path = classic._root() / "comparison_vs_global.csv"
    weekly_path = weekly._root() / "comparison_vs_global.csv"
    if not classic_path.exists():
        classic.run()
    if not weekly_path.exists():
        weekly.run()
    classic_frame = pd.read_csv(classic_path, float_precision="round_trip")
    weekly_frame = pd.read_csv(weekly_path, float_precision="round_trip")
    classic_frame = classic_frame.loc[
        classic_frame["analysis_window"].eq(HEADLINE_WINDOW)
    ].set_index(["frequency", "span"])
    weekly_frame = weekly_frame.loc[
        weekly_frame["analysis_window"].eq(HEADLINE_WINDOW)
    ].set_index(["frequency", "span"])
    production = production.loc[
        production["analysis_window"].eq(HEADLINE_WINDOW)
    ]
    prod_index = production.set_index(["signal_variant", "frequency", "span"])
    rows = []
    for frequency, span in _cells():
        row = {
            "analysis_window": HEADLINE_WINDOW,
            "frequency": frequency,
            "span": span,
            "cell_id": _cell_id(frequency, span),
            "q": Q,
        }
        for label, source in (
            ("prod_exact", prod_index.loc[(EXACT_VARIANT, frequency, span)]),
            ("prod_scaled", prod_index.loc[(SCALED_VARIANT, frequency, span)]),
            ("classic_monthly", classic_frame.loc[(frequency, span)]),
            ("raw_weekly", weekly_frame.loc[(frequency, span)]),
        ):
            for metric in SIGNAL_COMPARISON_METRICS:
                row[f"{label}_{metric}"] = source[metric]
            row[f"{label}_beats_global_net_return"] = bool(
                source["beats_global_net_return"]
            )
            row[f"{label}_mean_variance_dominates_global"] = bool(
                source["mean_variance_dominates_global"]
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _breadth_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    """Summarise the same-signal global win count and leader for four signals."""
    rows = []
    labels = {
        "prod_exact": "ROSAA production exact monthly",
        "prod_scaled": "ROSAA production mechanics calendar-scaled",
        "classic_monthly": "classic monthly 12m skip 1m",
        "raw_weekly": "paper raw weekly 48w skip 4w",
    }
    for prefix, description in labels.items():
        delta = f"{prefix}_delta_net_return_annualized"
        volatility = f"{prefix}_delta_volatility_annualized"
        winner = comparison.sort_values(
            [delta, volatility],
            ascending=[False, True],
        ).iloc[0]
        rows.append(
            {
                "signal": prefix,
                "description": description,
                "cells": len(comparison),
                "return_wins": int(
                    comparison[f"{prefix}_beats_global_net_return"].sum()
                ),
                "mean_variance_wins": int(
                    comparison[
                        f"{prefix}_mean_variance_dominates_global"
                    ].sum()
                ),
                "best_frequency": winner["frequency"],
                "best_span": int(winner["span"]),
                "best_delta_net_return_annualized": winner[delta],
                "best_delta_volatility_annualized": winner[volatility],
                "best_cluster_net_return_annualized": winner[
                    f"{prefix}_cluster_net_return_annualized"
                ],
                "best_cluster_volatility_annualized": winner[
                    f"{prefix}_cluster_volatility_annualized"
                ],
            }
        )
    return pd.DataFrame(rows)


def _me_variant_regression(production: pd.DataFrame) -> pd.DataFrame:
    """Require exact-monthly and scaled variants to coincide in every ME cell."""
    panel = production.loc[
        production["frequency"].eq("ME")
    ].set_index(["analysis_window", "signal_variant", "span"])
    errors = {}
    metrics = (
        "cluster_net_return_annualized",
        "global_net_return_annualized",
        "delta_net_return_annualized",
        "cluster_volatility_annualized",
        "delta_volatility_annualized",
        "cluster_one_way_turnover_annualized",
    )
    for metric in metrics:
        exact = panel.xs(EXACT_VARIANT, level="signal_variant")[metric]
        scaled = panel.xs(SCALED_VARIANT, level="signal_variant")[metric]
        errors[metric] = float(exact.subtract(scaled).abs().max())
    maximum = max(errors.values())
    frame = pd.DataFrame(
        [
            {
                "check": "exact_monthly_equals_scaled_ME",
                "measured": maximum,
                "tolerance": REGRESSION_TOLERANCE,
                "details": "|".join(
                    f"{metric}={value:.3e}" for metric, value in errors.items()
                ),
                "status": "PASS" if maximum <= REGRESSION_TOLERANCE else "FAIL",
            }
        ]
    )
    if maximum > REGRESSION_TOLERANCE:
        raise AssertionError(frame)
    return frame


def run() -> Mapping[str, pd.DataFrame]:
    """Execute both production variants over all cached covariance cells."""
    started = time.perf_counter()
    dates, fixed_eligibility = _accepted_dates_and_eligibility()
    accepted_windows = e5._analysis_windows(UNIVERSE, dates)
    data = e5.load_universe(UNIVERSE)
    performance_prices = e5._prices(data).reindex(
        columns=fixed_eligibility.columns
    )
    costs = e5.get_universe_spec(UNIVERSE).cost_bps / 10000.0
    ew_navs = _ew_navs()
    daily = _read_daily(fixed_eligibility.columns)
    parameter_table = _signal_parameter_table()

    required_specs = {
        _signal_spec(variant, frequency).cache_key:
        _signal_spec(variant, frequency)
        for variant in VARIANTS
        for frequency in LONG_SPANS
    }
    base_cache = {}
    signal_regression_rows = []
    first_available_dates = []
    for key, spec in required_specs.items():
        signal_prices, benchmark, period_returns, _ = _period_inputs(
            data,
            daily,
            spec.frequency,
        )
        global_source, raw_source = _base_signals(
            signal_prices,
            benchmark,
            spec,
        )
        global_decision, timestamps = _asof_panel(global_source, dates)
        raw_decision, raw_timestamps = _asof_panel(raw_source, dates)
        if not timestamps.equals(raw_timestamps):
            raise AssertionError(f"raw/global timestamps differ for {key}")
        eligible_counts = global_decision.where(fixed_eligibility).notna().sum(axis=1)
        available = eligible_counts.loc[eligible_counts.gt(0)]
        if available.empty:
            raise AssertionError(f"no production scores for {key}")
        first_available = pd.Timestamp(available.index.min())
        first_available_dates.append(first_available)
        lookahead_days = (timestamps - timestamps.index).dt.days
        roundtrip = qis.to_returns(
            signal_prices,
            freq=spec.frequency,
            is_log_returns=True,
        ).reindex_like(period_returns)
        difference = roundtrip.subtract(period_returns).abs().to_numpy()
        finite = difference[np.isfinite(difference)]
        roundtrip_error = float(finite.max()) if finite.size else 0.0
        status = (
            "PASS"
            if float(lookahead_days.max()) <= 0.0
            and roundtrip_error <= REGRESSION_TOLERANCE
            else "FAIL"
        )
        signal_regression_rows.append(
            {
                "check": "production_signal_preflight",
                "signal_frequency": spec.frequency,
                "long_span": spec.long_span,
                "long_horizon_months": (
                    12.0 * spec.long_span / PERIODS_PER_YEAR[spec.frequency]
                ),
                "vol_span": spec.vol_span,
                "first_available_date": first_available,
                "max_signal_lookahead_days": float(lookahead_days.max()),
                "return_roundtrip_max_abs_error": roundtrip_error,
                "return_roundtrip_tolerance": REGRESSION_TOLERANCE,
                "status": status,
            }
        )
        if status != "PASS":
            raise AssertionError(signal_regression_rows[-1])
        base_cache[key] = {
            "global": global_decision,
            "raw": raw_decision,
            "timestamps": timestamps,
        }
        print(
            f"production signal {spec.frequency} "
            f"long={spec.long_span} vol={spec.vol_span}: complete",
            flush=True,
        )

    common_first = max(first_available_dates)
    windows = {
        HEADLINE_WINDOW: accepted_windows[HEADLINE_WINDOW],
        AVAILABLE_WINDOW: dates[dates >= common_first],
    }
    ew_nav_by_window = {
        HEADLINE_WINDOW: ew_navs[HEADLINE_WINDOW],
        AVAILABLE_WINDOW: ew_navs["full_panel"],
    }

    performance_rows = []
    acceptance_rows = []
    risk_rows = []
    score_rows = []
    runtime_rows = []
    global_inputs = {}
    unique_global_specs = {}
    for variant in VARIANTS:
        for covariance_frequency in LONG_SPANS:
            spec = _signal_spec(variant, covariance_frequency)
            unique_global_specs[(variant, spec.frequency)] = spec

    for (variant, signal_frequency), spec in unique_global_specs.items():
        base = base_cache[spec.cache_key]
        for window, window_dates in windows.items():
            eligibility = fixed_eligibility.reindex(index=window_dates)
            scores = base["global"].reindex(
                index=window_dates,
                columns=eligibility.columns,
            ).where(eligibility)
            valid_counts = scores.notna().sum(axis=1)
            score_rows.append(
                {
                    "analysis_window": window,
                    "signal_variant": variant,
                    "signal_frequency": signal_frequency,
                    "long_span": spec.long_span,
                    "vol_span": spec.vol_span,
                    "dates": len(window_dates),
                    "valid_assets_min": int(valid_counts.min()),
                    "valid_assets_median": float(valid_counts.median()),
                    "valid_assets_max": int(valid_counts.max()),
                    "common_first_available_date": common_first,
                }
            )
            global_groups = pd.DataFrame(
                "global",
                index=window_dates,
                columns=eligibility.columns,
            )
            weights, exposure, side_validation = single._leg_weights(
                scores,
                eligibility,
                global_groups,
            )
            safe_frequency = signal_frequency.replace("-", "_")
            leg = f"global_{variant}_{safe_frequency}"
            performance, acceptance = _run_leg(
                window,
                "BENCHMARK_INVARIANT",
                np.nan,
                leg,
                "asset_equal",
                leg,
                performance_prices,
                weights,
                exposure,
                side_validation,
                global_groups,
                costs,
                ew_nav_by_window[window],
                signal_variant=variant,
                signal_frequency=signal_frequency,
                long_span=spec.long_span,
                vol_span=spec.vol_span,
            )
            performance_rows.append(performance)
            acceptance_rows.append(acceptance)
            global_inputs[(variant, signal_frequency, window)] = weights

    cluster_score_cache = {}
    for frequency, span in _cells():
        cell_started = time.perf_counter()
        groups_all, _ = _load_partition(frequency, span)
        cell_id = _cell_id(frequency, span)
        for variant in VARIANTS:
            spec = _signal_spec(variant, frequency)
            score_key = (spec.cache_key, cell_id)
            if score_key not in cluster_score_cache:
                raw = base_cache[spec.cache_key]["raw"]
                cluster_score_cache[score_key] = score_within_clusters(
                    raw_signal=raw,
                    rolling_clusters=_panel_dict(groups_all),
                    min_cluster_size=MIN_CLUSTER_SIZE,
                )
            scores_all = cluster_score_cache[score_key]
            for window, window_dates in windows.items():
                eligibility = fixed_eligibility.reindex(index=window_dates)
                groups = groups_all.reindex(
                    index=window_dates,
                    columns=eligibility.columns,
                )
                scores = scores_all.reindex(
                    index=window_dates,
                    columns=eligibility.columns,
                ).where(eligibility)
                weights, exposure, side_validation = single._leg_weights(
                    scores,
                    eligibility,
                    groups,
                )
                leg = f"cluster_{variant}_{cell_id}"
                performance, acceptance = _run_leg(
                    window,
                    frequency,
                    span,
                    cell_id,
                    "group_equal",
                    leg,
                    performance_prices,
                    weights,
                    exposure,
                    side_validation,
                    groups,
                    costs,
                    ew_nav_by_window[window],
                    signal_variant=variant,
                    signal_frequency=spec.frequency,
                    long_span=spec.long_span,
                    vol_span=spec.vol_span,
                )
                performance_rows.append(performance)
                acceptance_rows.append(acceptance)
                risk = weekly._risk_diagnostic(
                    window,
                    frequency,
                    span,
                    weights,
                    global_inputs[(variant, spec.frequency, window)],
                    groups,
                )
                risk.update(
                    {
                        "signal_variant": variant,
                        "signal_frequency": spec.frequency,
                    }
                )
                risk_rows.append(risk)
        runtime_rows.append(
            {
                "frequency": frequency,
                "span": span,
                "cell_id": cell_id,
                "runtime_seconds": time.perf_counter() - cell_started,
            }
        )
        print(f"production long-short {frequency}/{span}: complete", flush=True)

    performance = pd.DataFrame(performance_rows).sort_values(
        [
            "analysis_window",
            "signal_variant",
            "frequency",
            "span",
            "signal_frequency",
        ],
        na_position="first",
    ).reset_index(drop=True)
    acceptance = pd.DataFrame(acceptance_rows)
    if not acceptance["status"].eq("PASS").all():
        raise AssertionError(acceptance.loc[~acceptance["status"].eq("PASS")])
    comparison = _comparison(performance)
    rankings = weekly._rankings(comparison)
    signal_comparison = _other_signal_comparison(comparison)
    breadth = _breadth_summary(signal_comparison)
    me_regression = _me_variant_regression(comparison)
    regression = pd.concat(
        [pd.DataFrame(signal_regression_rows), me_regression],
        ignore_index=True,
        sort=False,
    )
    runtime = pd.DataFrame(runtime_rows)
    runtime["total_run_seconds"] = time.perf_counter() - started
    output = {
        "signal_parameters": parameter_table,
        "performance": performance,
        "comparison_vs_global": comparison,
        "rankings": rankings,
        "comparison_vs_other_signals": signal_comparison,
        "signal_breadth_summary": breadth,
        "risk_diagnostics": pd.DataFrame(risk_rows),
        "score_diagnostics": pd.DataFrame(score_rows),
        "regression": regression,
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
    """Replay a complete persisted run and require byte-identical artifacts."""
    expected = {
        "acceptance.csv",
        "comparison_vs_global.csv",
        "comparison_vs_other_signals.csv",
        "performance.csv",
        "rankings.csv",
        "regression.csv",
        "risk_diagnostics.csv",
        "score_diagnostics.csv",
        "signal_breadth_summary.csv",
        "signal_parameters.csv",
    }
    first = _hash_outputs()
    if set(first) != expected:
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
    """Run the deterministic production-momentum U1 covariance grid."""
    replay = verify_determinism()
    print(
        f"U1 production-momentum covariance grid: PASS "
        f"({len(replay)}/{len(replay)} deterministic)",
        flush=True,
    )


if __name__ == "__main__":
    main()
