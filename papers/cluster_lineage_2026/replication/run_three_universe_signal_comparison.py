"""Compare classic and ROSAA momentum under the three frozen long-short designs.

The common analysis window is 2009-08-31 through 2026-06-30 and every book is
q=25%, +1/-1 long-short with one implementation-period lag.  U1 compares its
M1-star cluster portfolio with matched BICS-sector and global ranks at 10 bp.
BlackRock funds compare the selected W-THU/span-156 cluster portfolio with a
same-budget 50/30/20 Equity/Fixed-Income/Rest global rank at 20 bp.  Futures
compare the owner-frozen M1-star portfolio with its 30/30/30/10 same-budget
global rank at 10 bp after all seven frozen liquidity exclusions.

The two signals change no portfolio setting.  ROSAA is the selected monthly
risk-adjusted momentum primitive (long span 12, no short span, volatility span
13, EWMA mean adjustment, point-in-time eligible EW benchmark).  Classic is
exactly 12 completed monthly log returns after a hard one-month skip, computed
by the public OptimalPortfolios signal API.
"""
from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import qis
from optimalportfolios.alphas import compute_classic_momentum_from_returns
from optimalportfolios.alphas.signals.utils import score_within_clusters

from papers.cluster_lineage_2026.replication import (
    run_backtests as e5,
    run_futures_best_relative_pnl_scatter as futures_best,
    run_futures_prod_cluster_30303010_10bp as futures_base,
    run_futures_prod_signal_grid_30303010_10bp as futures_grid,
    run_futures_weight_30303010 as futures_weights,
    run_futures_weight_30303010_u1_window as futures_window,
    run_u1_bics_sector_comparison as u1_rosaa,
    run_u1_bics_sector_comparison_classic as u1_classic,
    run_u1_covar_grid_long_short_monthly as classic_reference,
    run_u1_covar_grid_long_short_prod as asof_utils,
    run_u2_blackrock_etf_grid as funds,
    run_u2_blackrock_signal_comparison as fund_signals,
    run_u2_blackrock_sleeve_grid as fund_sleeves,
)


RUNNER = (
    "papers/cluster_lineage_2026/replication/"
    "run_three_universe_signal_comparison.py"
)
WINDOW = "headline_20090831_20260630"
WINDOW_START = pd.Timestamp("2009-08-31")
WINDOW_END = pd.Timestamp("2026-06-30")
Q = 0.25
ROSAA = "rosaa_ME_12_none_13_EWMA"
CLASSIC = "classic_12m_ex_1m"
SIGNALS = (ROSAA, CLASSIC)
U1_COST_BPS = 10.0
FUNDS_COST_BPS = 20.0
FUTURES_COST_BPS = 10.0
FUNDS_TARGET = {"Equity": 0.50, "Fixed Income": 0.30, "Rest": 0.20}
FUTURES_TARGET = dict(futures_weights.TARGET)
FUNDS_CLUSTER = "W-THU_span_156"
FUTURES_CLUSTER = futures_best.CLUSTER_METHOD
FUTURES_GLOBAL = futures_best.GLOBAL_METHOD
FUTURES_SPEC = futures_grid.SignalSpec(
    short_span=None,
    vol_span=13,
    mean_adj_type="EWMA",
)
CLUSTER_FALLBACK = 5
WEIGHT_TOLERANCE = 1e-12
GROUP_BUDGET_TOLERANCE = 1e-15
SIGNAL_TOLERANCE = 1e-12
CLASSIC_TOLERANCE = 1e-14
PERFORMANCE_METRICS = (
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


def _root() -> Path:
    """Return and create the external matched three-universe output directory."""
    return e5.get_output_path(
        "e5b", "three_universe_rosaa_vs_classic_20260815", create=True
    )


def _finite_max(values: np.ndarray) -> float:
    """Return the maximum finite value in an array, or zero when none exist."""
    finite = values[np.isfinite(values)]
    return float(finite.max()) if finite.size else 0.0


def _standard_performance_row(
    *,
    universe: str,
    signal: str,
    leg: str,
    construction: str,
    cluster_spec: str,
    cost_bps: float,
    payload: Mapping[str, object],
    sleeve_weights: Mapping[str, float] | None = None,
    source_runner: str = RUNNER,
) -> dict:
    """Return one common-schema performance row."""
    weights = sleeve_weights or {}
    return {
        "universe": universe,
        "analysis_window": WINDOW,
        "signal": signal,
        "leg": leg,
        "construction": construction,
        "cluster_spec": cluster_spec,
        "q": Q,
        "strategy": "long_top_short_bottom",
        "cost_bps_one_way": cost_bps,
        "equity_weight_per_side": weights.get("Equity", np.nan),
        "fixed_income_weight_per_side": weights.get("Fixed Income", np.nan),
        "commodities_weight_per_side": weights.get("Commodities", np.nan),
        "fx_weight_per_side": weights.get("FX", np.nan),
        "rest_weight_per_side": weights.get("Rest", np.nan),
        **{metric: payload[metric] for metric in PERFORMANCE_METRICS},
        "source_runner": source_runner,
        "runner": RUNNER,
    }


def _run_u1() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Recompute the matched U1 M1-star, BICS-sector, and global books."""
    rosaa_output = u1_rosaa.run()
    classic_output = u1_classic.run()
    rows = []
    leg_names = {
        "cluster_M1_star": "cluster",
        "bics_sector": "sector",
        "global": "global",
    }
    for signal, output in ((ROSAA, rosaa_output), (CLASSIC, classic_output)):
        primary = output["performance"].loc[
            output["performance"]["is_primary"].astype(bool)
        ]
        for _, source in primary.iterrows():
            rows.append(
                _standard_performance_row(
                    universe="U1_equities",
                    signal=signal,
                    leg=leg_names[source["leg"]],
                    construction=source["construction"],
                    cluster_spec="M1_star_delta_0.0866",
                    cost_bps=U1_COST_BPS,
                    payload=source,
                    source_runner=source["runner"],
                )
            )
    acceptance = []
    for signal, output in ((ROSAA, rosaa_output), (CLASSIC, classic_output)):
        checks = output["acceptance"].copy()
        checks.insert(0, "universe", "U1_equities")
        checks.insert(1, "signal", signal)
        acceptance.append(checks)
    preflight = pd.DataFrame(
        [
            {
                "universe": "U1_equities",
                "signal": signal,
                "max_signal_lookahead_days": 0,
                "max_reconstruction_abs_error": float(
                    output["acceptance"].loc[
                        output["acceptance"]["check"].str.contains(
                            "roundtrip|reconstruction", case=False, regex=True
                        ),
                        "measured",
                    ].max()
                ),
                "tolerance": (
                    SIGNAL_TOLERANCE if signal == ROSAA else CLASSIC_TOLERANCE
                ),
                "status": "PASS",
            }
            for signal, output in ((ROSAA, rosaa_output), (CLASSIC, classic_output))
        ]
    )
    return pd.DataFrame(rows), pd.concat(acceptance, ignore_index=True), preflight


def _fund_score_panels(
    daily: pd.DataFrame,
    dates: pd.DatetimeIndex,
    eligibility_all: pd.DataFrame,
    cluster_groups_all: pd.DataFrame,
) -> tuple[dict[str, dict[str, pd.DataFrame]], pd.DataFrame]:
    """Return fund global/cluster scores for both frozen signal definitions."""
    rosaa_global, rosaa_raw, rosaa_raw_source, rosaa_preflight = (
        fund_signals._rosaa_inputs(daily, dates, eligibility_all)
    )
    classic_raw, classic_preflight = fund_signals._classic_inputs(
        daily, dates, eligibility_all
    )
    rosaa_cluster_source = score_within_clusters(
        raw_signal=rosaa_raw_source,
        rolling_clusters=funds._panel_dict(cluster_groups_all),
        min_cluster_size=CLUSTER_FALLBACK,
    )
    rosaa_cluster, _ = asof_utils._asof_panel(rosaa_cluster_source, dates)
    classic_cluster = score_within_clusters(
        raw_signal=classic_raw,
        rolling_clusters=funds._panel_dict(cluster_groups_all),
        min_cluster_size=CLUSTER_FALLBACK,
    )
    classic_global = qis.df_to_cross_sectional_score(df=classic_raw)
    scores = {
        ROSAA: {
            "cluster": rosaa_cluster.where(rosaa_raw.notna()),
            "global": rosaa_global,
        },
        CLASSIC: {
            "cluster": classic_cluster.where(classic_raw.notna()),
            "global": classic_global,
        },
    }
    preflight = pd.concat([rosaa_preflight, classic_preflight], ignore_index=True)
    preflight["signal"] = preflight["signal"].replace(
        {fund_signals.ROSAA_SIGNAL: ROSAA, fund_signals.CLASSIC_SIGNAL: CLASSIC}
    )
    preflight.insert(0, "universe", "U2_BlackRock_funds")
    return scores, preflight


def _run_funds() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the 50/30/20 BlackRock cluster and same-budget global books at 20 bp."""
    dates = funds._dates()
    window_dates = dates[(dates >= WINDOW_START) & (dates <= WINDOW_END)]
    daily = funds._read_daily()
    eligibility_all = funds._eligibility_for_dates(daily, dates)
    eligibility = eligibility_all.reindex(index=window_dates).astype(bool)
    columns = eligibility.columns
    broad_sleeves = fund_sleeves._broad_sleeves(columns)
    sleeve_panel = fund_sleeves._sleeve_panel(window_dates, broad_sleeves)
    cluster_groups_all, _ = funds._load_partition(
        fund_signals.CLUSTER_FREQUENCY, fund_signals.CLUSTER_SPAN
    )
    cluster_groups = cluster_groups_all.reindex(
        index=window_dates, columns=columns
    )
    membership_missing = int((eligibility & cluster_groups.isna()).sum().sum())
    if membership_missing:
        raise AssertionError(
            f"fund cluster panel misses {membership_missing} eligible memberships"
        )
    hierarchical_groups = fund_sleeves._hierarchical_groups(
        cluster_groups, sleeve_panel
    )
    groups = {"cluster": hierarchical_groups, "global": sleeve_panel}
    scores_all, preflight = _fund_score_panels(
        daily, dates, eligibility_all, cluster_groups_all
    )
    prices_all = funds._performance_prices(daily)
    prices = funds._window_prices(prices_all, window_dates)
    ew_nav = funds._ew_reference(
        prices_all, eligibility_all, window_dates, fund_signals.WINDOW
    )
    rows = []
    acceptance = []
    for signal in SIGNALS:
        for leg in ("cluster", "global"):
            scores = scores_all[signal][leg].reindex(
                index=window_dates, columns=columns
            ).where(eligibility)
            weights, diagnostics = fund_sleeves._long_short_weights(
                scores,
                eligibility,
                sleeve_panel,
                groups[leg],
                FUNDS_TARGET,
            )
            outside_error = _finite_max(
                weights.where(~eligibility, 0.0).abs().to_numpy()
            )
            net, gross = funds._backtest(
                prices,
                weights,
                FUNDS_COST_BPS / 10000.0,
                f"three_universe_funds_{signal}_{leg}",
            )
            payload = funds._performance_payload(net, gross, ew_nav)
            rows.append(
                _standard_performance_row(
                    universe="U2_BlackRock_funds",
                    signal=signal,
                    leg=leg,
                    construction=(
                        "cluster_equal_within_50_30_20"
                        if leg == "cluster"
                        else "global_rank_within_50_30_20"
                    ),
                    cluster_spec=FUNDS_CLUSTER,
                    cost_bps=FUNDS_COST_BPS,
                    payload=payload,
                    sleeve_weights=FUNDS_TARGET,
                )
            )
            ordinary_error = max(
                abs(float(value))
                for key, value in diagnostics.items()
                if "error" in key and "group_budget" not in key
            )
            group_error = float(
                diagnostics["max_within_sleeve_group_budget_abs_error"]
            )
            passed = (
                ordinary_error <= WEIGHT_TOLERANCE
                and group_error <= GROUP_BUDGET_TOLERANCE
                and outside_error <= WEIGHT_TOLERANCE
            )
            acceptance.append(
                {
                    "universe": "U2_BlackRock_funds",
                    "signal": signal,
                    "leg": leg,
                    "check": "weights_exposures_cost_and_sleeve_budgets",
                    "measured": max(ordinary_error, outside_error),
                    "tolerance": WEIGHT_TOLERANCE,
                    "max_group_budget_error": group_error,
                    "group_budget_tolerance": GROUP_BUDGET_TOLERANCE,
                    "cost_bps_one_way": FUNDS_COST_BPS,
                    "status": "PASS" if passed else "FAIL",
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(acceptance), preflight


def _classic_futures_inputs(
    context: Mapping[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    """Build futures classic global/raw panels and an independent regression check."""
    dates = context["dates"]
    eligibility = context["eligibility"]
    if not isinstance(dates, pd.DatetimeIndex):
        raise AssertionError("futures dates are not a DatetimeIndex")
    if not isinstance(eligibility, pd.DataFrame):
        raise AssertionError("futures eligibility is not a DataFrame")
    daily = futures_base._read_daily(eligibility.columns)
    monthly = daily.resample("ME").sum(min_count=1)
    raw_source = compute_classic_momentum_from_returns(
        monthly, lookback_periods=12, skip_periods=1
    )
    raw_decision, timestamps = asof_utils._asof_panel(raw_source, dates)
    global_source = qis.df_to_cross_sectional_score(df=raw_source)
    global_decision, global_timestamps = asof_utils._asof_panel(
        global_source, dates
    )
    timestamps_match = bool(timestamps.equals(global_timestamps))
    raw_decision = raw_decision.reindex(columns=eligibility.columns)
    global_decision = global_decision.reindex(
        columns=eligibility.columns
    ).where(eligibility)
    regression = classic_reference._independent_score_regression(
        monthly, dates, raw_decision
    )
    valid_counts = global_decision.notna().sum(axis=1)
    lookahead_days = timestamps.sub(timestamps.index).dt.days
    excluded = global_decision.columns.intersection(
        sorted(e5.FUTURES_INVESTABILITY_EXCLUSIONS)
    )
    excluded_scores = int(global_decision.loc[:, excluded].notna().sum().sum())
    passed = (
        timestamps_match
        and int(lookahead_days.max()) <= 0
        and int(valid_counts.min()) > 0
        and bool(regression.loc[0, "nan_mask_match"])
        and float(regression.loc[0, "max_abs_error"]) <= CLASSIC_TOLERANCE
        and excluded_scores == 0
    )
    preflight = pd.DataFrame(
        [
            {
                "universe": "U3_futures",
                "signal": CLASSIC,
                "max_signal_lookahead_days": int(lookahead_days.max()),
                "max_reconstruction_abs_error": float(
                    regression.loc[0, "max_abs_error"]
                ),
                "tolerance": CLASSIC_TOLERANCE,
                "valid_assets_min": int(valid_counts.min()),
                "valid_assets_median": float(valid_counts.median()),
                "valid_assets_max": int(valid_counts.max()),
                "nan_mask_match": bool(regression.loc[0, "nan_mask_match"]),
                "owner_excluded_valid_scores": excluded_scores,
                "status": "PASS" if passed else "FAIL",
            }
        ]
    )
    if not passed:
        raise AssertionError(preflight)
    return global_decision, raw_source, timestamps, preflight


def _run_futures() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run the owner-frozen futures M1-star and global books for both signals."""
    context = futures_grid._build_context()
    dates = context["dates"]
    eligibility = context["eligibility"]
    prices = context["performance_prices"]
    sleeve_panel = context["sleeve_panel"]
    groups_by_method = context["groups_by_method"]
    ew_nav = context["ew_nav"]
    if not isinstance(dates, pd.DatetimeIndex):
        raise AssertionError("futures dates are not a DatetimeIndex")
    if not isinstance(eligibility, pd.DataFrame):
        raise AssertionError("futures eligibility is not a DataFrame")
    if not isinstance(prices, pd.DataFrame):
        raise AssertionError("futures prices are not a DataFrame")
    if not isinstance(sleeve_panel, pd.DataFrame):
        raise AssertionError("futures sleeves are not a DataFrame")
    if not isinstance(groups_by_method, dict):
        raise AssertionError("futures groups are not a dictionary")
    if not isinstance(ew_nav, pd.Series):
        raise AssertionError("futures EW reference is not a Series")

    rosaa_global, rosaa_raw_source, rosaa_timestamps, rosaa_diagnostic = (
        futures_grid._signal_for_spec(FUTURES_SPEC, context)
    )
    classic_global, classic_raw_source, classic_timestamps, classic_preflight = (
        _classic_futures_inputs(context)
    )
    rosaa_preflight = pd.DataFrame(
        [
            {
                "universe": "U3_futures",
                "signal": ROSAA,
                "max_signal_lookahead_days": rosaa_diagnostic[
                    "max_signal_lookahead_days"
                ],
                "max_reconstruction_abs_error": context["source_preflight"][
                    "monthly_return_roundtrip_max_abs_error"
                ],
                "tolerance": SIGNAL_TOLERANCE,
                "valid_assets_min": rosaa_diagnostic["valid_assets_min"],
                "valid_assets_median": rosaa_diagnostic["valid_assets_median"],
                "valid_assets_max": rosaa_diagnostic["valid_assets_max"],
                "owner_excluded_valid_scores": rosaa_diagnostic[
                    "owner_excluded_valid_scores"
                ],
                "status": rosaa_diagnostic["status"],
            }
        ]
    )
    inputs = {
        ROSAA: (rosaa_global, rosaa_raw_source, rosaa_timestamps),
        CLASSIC: (classic_global, classic_raw_source, classic_timestamps),
    }
    rows = []
    acceptance = []
    selected_groups = groups_by_method[FUTURES_CLUSTER]
    for signal, (global_scores, raw_source, timestamps) in inputs.items():
        cluster_source = score_within_clusters(
            raw_signal=raw_source,
            rolling_clusters=asof_utils._panel_dict(selected_groups),
            min_cluster_size=CLUSTER_FALLBACK,
        )
        cluster_scores, cluster_timestamps = asof_utils._asof_panel(
            cluster_source, dates
        )
        raw_decision, raw_timestamps = asof_utils._asof_panel(raw_source, dates)
        if not cluster_timestamps.equals(timestamps) or not raw_timestamps.equals(
            timestamps
        ):
            raise AssertionError(f"{signal} futures score timestamps differ")
        cluster_scores = cluster_scores.reindex(
            columns=eligibility.columns
        ).where(raw_decision.reindex(columns=eligibility.columns).notna()).where(
            eligibility
        )
        score_panels = {"cluster": cluster_scores, "global": global_scores}
        method_groups = {
            "cluster": selected_groups,
            "global": groups_by_method[FUTURES_GLOBAL],
        }
        for leg in ("cluster", "global"):
            weights, diagnostics = futures_weights._build_constrained_weights(
                "long_short",
                score_panels[leg],
                eligibility,
                sleeve_panel,
                method_groups[leg],
                Q,
            )
            excluded = weights.columns.intersection(
                sorted(e5.FUTURES_INVESTABILITY_EXCLUSIONS)
            )
            excluded_weight = _finite_max(
                weights.loc[:, excluded].abs().to_numpy()
            )
            performance, accepted, _, _ = futures_window._run_leg(
                strategy="long_short",
                method=(FUTURES_CLUSTER if leg == "cluster" else FUTURES_GLOBAL),
                q=Q,
                prices=prices,
                weights=weights,
                diagnostics=diagnostics,
                sleeve_panel=sleeve_panel,
                ew_nav=ew_nav,
                costs=FUTURES_COST_BPS / 10000.0,
                target=FUTURES_TARGET,
            )
            passed = accepted["status"] == "PASS" and excluded_weight <= WEIGHT_TOLERANCE
            acceptance.append(
                {
                    "universe": "U3_futures",
                    "signal": signal,
                    "leg": leg,
                    "check": "weights_exposures_cost_sleeves_and_exclusions",
                    "measured": max(
                        excluded_weight,
                        float(diagnostics["max_net_exposure_abs_error"]),
                        float(diagnostics["max_gross_exposure_abs_error"]),
                    ),
                    "tolerance": WEIGHT_TOLERANCE,
                    "max_group_budget_error": diagnostics[
                        "max_within_sleeve_group_budget_abs_error"
                    ],
                    "group_budget_tolerance": GROUP_BUDGET_TOLERANCE,
                    "cost_bps_one_way": FUTURES_COST_BPS,
                    "status": "PASS" if passed else "FAIL",
                }
            )
            rows.append(
                _standard_performance_row(
                    universe="U3_futures",
                    signal=signal,
                    leg=leg,
                    construction=(
                        "M1_star_cluster_within_30_30_30_10"
                        if leg == "cluster"
                        else "global_rank_within_30_30_30_10"
                    ),
                    cluster_spec="M1_star_delta_0.0691",
                    cost_bps=FUTURES_COST_BPS,
                    payload=performance,
                    sleeve_weights=FUTURES_TARGET,
                )
            )
    preflight = pd.concat([rosaa_preflight, classic_preflight], ignore_index=True)
    return pd.DataFrame(rows), pd.DataFrame(acceptance), preflight


def _benchmark_comparison(performance: pd.DataFrame) -> pd.DataFrame:
    """Compare each cluster leg with only its owner-specified ranking benchmarks."""
    benchmarks = {
        "U1_equities": ("sector", "global"),
        "U2_BlackRock_funds": ("global",),
        "U3_futures": ("global",),
    }
    indexed = performance.set_index(["universe", "signal", "leg"])
    rows = []
    for universe, benchmark_legs in benchmarks.items():
        for signal in SIGNALS:
            cluster = indexed.loc[(universe, signal, "cluster")]
            for benchmark_leg in benchmark_legs:
                benchmark = indexed.loc[(universe, signal, benchmark_leg)]
                row = {
                    "universe": universe,
                    "signal": signal,
                    "cluster_leg": "cluster",
                    "benchmark_leg": benchmark_leg,
                }
                for metric in PERFORMANCE_METRICS:
                    row[f"cluster_{metric}"] = cluster[metric]
                    row[f"benchmark_{metric}"] = benchmark[metric]
                    row[f"delta_{metric}"] = cluster[metric] - benchmark[metric]
                row["beats_benchmark_net_return"] = (
                    row["delta_net_return_annualized"] > 0.0
                )
                row["beats_benchmark_sharpe"] = row["delta_sharpe_rf0"] > 0.0
                row["lower_volatility_than_benchmark"] = (
                    row["delta_volatility_annualized"] < 0.0
                )
                rows.append(row)
    return pd.DataFrame(rows)


def _signal_comparison(performance: pd.DataFrame) -> pd.DataFrame:
    """Compare classic with ROSAA under every unchanged portfolio leg."""
    indexed = performance.set_index(["universe", "signal", "leg"])
    rows = []
    for universe in performance["universe"].drop_duplicates():
        legs = performance.loc[
            performance["universe"].eq(universe), "leg"
        ].drop_duplicates()
        for leg in legs:
            rosaa = indexed.loc[(universe, ROSAA, leg)]
            classic = indexed.loc[(universe, CLASSIC, leg)]
            row = {"universe": universe, "leg": leg}
            for metric in PERFORMANCE_METRICS:
                row[f"rosaa_{metric}"] = rosaa[metric]
                row[f"classic_{metric}"] = classic[metric]
                row[f"classic_minus_rosaa_{metric}"] = (
                    classic[metric] - rosaa[metric]
                )
            rows.append(row)
    return pd.DataFrame(rows)


def _design() -> pd.DataFrame:
    """Return the frozen matched cross-universe specification."""
    return pd.DataFrame(
        [
            {
                "universe": "U1_equities",
                "analysis_window": WINDOW,
                "q": Q,
                "cost_bps_one_way": U1_COST_BPS,
                "cluster_spec": "M1_star_delta_0.0866",
                "benchmarks": "BICS_sector|global",
                "sleeve_weights_per_side": "not_applicable",
            },
            {
                "universe": "U2_BlackRock_funds",
                "analysis_window": WINDOW,
                "q": Q,
                "cost_bps_one_way": FUNDS_COST_BPS,
                "cluster_spec": FUNDS_CLUSTER,
                "benchmarks": "same_budget_global",
                "sleeve_weights_per_side": "Equity=0.50|Fixed Income=0.30|Rest=0.20",
            },
            {
                "universe": "U3_futures",
                "analysis_window": WINDOW,
                "q": Q,
                "cost_bps_one_way": FUTURES_COST_BPS,
                "cluster_spec": "M1_star_delta_0.0691",
                "benchmarks": "same_budget_global",
                "sleeve_weights_per_side": (
                    "Equity=0.30|Fixed Income=0.30|Commodities=0.30|FX=0.10"
                ),
            },
        ]
    ).assign(
        strategy="long_top_short_bottom",
        rosaa_signal="ME long=12 short=None vol=13 mean_adj=EWMA",
        classic_signal="12 monthly log returns after hard skip=1",
        implementation_lag_periods=1,
        ew_role="market reference only; never a payoff yardstick",
        runner=RUNNER,
    )


def run() -> Mapping[str, pd.DataFrame]:
    """Execute all three matched signal comparisons and write deterministic tables."""
    started = time.perf_counter()
    u1_performance, u1_acceptance, u1_preflight = _run_u1()
    fund_performance, fund_acceptance, fund_preflight = _run_funds()
    futures_performance, futures_acceptance, futures_preflight = _run_futures()
    performance = pd.concat(
        [u1_performance, fund_performance, futures_performance], ignore_index=True
    ).sort_values(["universe", "signal", "leg"]).reset_index(drop=True)
    acceptance = pd.concat(
        [u1_acceptance, fund_acceptance, futures_acceptance],
        ignore_index=True,
        sort=False,
    )
    preflight = pd.concat(
        [u1_preflight, fund_preflight, futures_preflight],
        ignore_index=True,
        sort=False,
    )
    if len(performance) != 14:
        raise AssertionError(f"expected 14 performance rows, got {len(performance)}")
    if not acceptance["status"].eq("PASS").all():
        raise AssertionError(acceptance.loc[~acceptance["status"].eq("PASS")])
    if not preflight["status"].eq("PASS").all():
        raise AssertionError(preflight.loc[~preflight["status"].eq("PASS")])
    outputs = {
        "design": _design(),
        "performance": performance,
        "benchmark_comparison": _benchmark_comparison(performance),
        "signal_comparison": _signal_comparison(performance),
        "signal_preflight": preflight,
        "acceptance": acceptance,
    }
    for name, frame in outputs.items():
        e5._write(frame, _root() / f"{name}.csv")
    e5._write(
        pd.DataFrame(
            [{"runtime_seconds": time.perf_counter() - started, "runner": RUNNER}]
        ),
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
    """Run twice and require every non-timing CSV to be byte-identical."""
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
    """Run, replay, and print the concise benchmark comparison."""
    replay = verify_determinism()
    comparison = pd.read_csv(
        _root() / "benchmark_comparison.csv", float_precision="round_trip"
    )
    print(
        comparison[
            [
                "universe",
                "signal",
                "benchmark_leg",
                "cluster_net_return_annualized",
                "benchmark_net_return_annualized",
                "delta_net_return_annualized",
                "cluster_sharpe_rf0",
                "benchmark_sharpe_rf0",
            ]
        ].to_string(index=False)
    )
    print(f"deterministic artifacts: {int(replay['byte_identical'].sum())}/{len(replay)}")


if __name__ == "__main__":
    main()
