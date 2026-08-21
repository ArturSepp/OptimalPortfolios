"""Compare U1 M1-star cluster, Bloomberg BICS-sector, and global ranks.

The owner-selected futures method is transferred without a signal search: q=25%,
monthly ROSAA production momentum with a 12-month long span, no short/reversal span,
volatility span 13, EWMA mean adjustment, five-name cluster fallback, +1/-1
long-short exposure, one-period implementation lag, and 10 bp one-way costs.  The
cluster leg uses the cached U1 M1-star partition (U1 delta 0.0866).

The primary comparison is universe-matched.  A stock must be point-in-time eligible,
have a valid score, and have a non-missing ``bbg_bics_sector`` label.  Each available
BICS sector receives equal budget separately on the long and short sides; selected
stocks within a sector split that sector budget equally.  The cluster and global legs
use the identical BICS-classified eligibility mask.  Missing BICS names are enumerated
and full-U1 cluster/global legs are reported separately as a coverage sensitivity.
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
import papers.cluster_lineage_2026.replication.run_u1_covar_grid_long_short as grid_ls
import papers.cluster_lineage_2026.replication.run_u1_covar_grid_long_short_prod as prod
import papers.cluster_lineage_2026.replication.run_u1_me36_long_short as single
from papers.cluster_lineage_2026.replication.run_u1_covar_grid import (
    UNIVERSE,
    _accepted_dates_and_eligibility,
    _backtest,
    _ew_navs,
    _read_daily,
)


WINDOW = "headline_20090831_20260630"
Q = 0.25
COST_BPS = 10.0
SIGNAL_FREQUENCY = "ME"
MOMENTUM_LONG_SPAN = 12
MOMENTUM_SHORT_SPAN = None
MOMENTUM_VOL_SPAN = 13
MOMENTUM_MEAN_ADJ_TYPE = "EWMA"
CLUSTER_CONFIG = e5.SmootherName.M1_STAR
CLUSTER_DELTA = 0.0866
CLUSTER_FALLBACK = 5
SECTOR_COLUMN = "bbg_bics_sector"
MISSING_SECTOR_POLICY = "exclude_from_all_primary_legs"
WEIGHT_TOLERANCE = 1e-12
SIGNAL_TOLERANCE = 1e-12
PRIMARY_LEGS = ("cluster_M1_star", "bics_sector", "global")
RUNNER = (
    "papers/cluster_lineage_2026/replication/"
    "run_u1_bics_sector_comparison.py"
)
FROZEN_SPEC = {
    "universe": UNIVERSE.value,
    "analysis_window": WINDOW,
    "strategy": "long_short",
    "q": Q,
    "signal_frequency": SIGNAL_FREQUENCY,
    "momentum_long_span": MOMENTUM_LONG_SPAN,
    "momentum_short_span": MOMENTUM_SHORT_SPAN,
    "momentum_vol_span": MOMENTUM_VOL_SPAN,
    "momentum_mean_adj_type": MOMENTUM_MEAN_ADJ_TYPE,
    "cluster_config": CLUSTER_CONFIG.value,
    "cluster_delta": CLUSTER_DELTA,
    "cluster_fallback": CLUSTER_FALLBACK,
    "sector_column": SECTOR_COLUMN,
    "missing_sector_policy": MISSING_SECTOR_POLICY,
    "cost_bps_one_way": COST_BPS,
    "implementation_lag_periods": 1,
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


def _root() -> Path:
    """Return and create the external U1 BICS comparison directory."""
    return e5.get_output_path(
        "e5b", "u1_bics_sector_vs_m1_star_owner_20260815", create=True
    )


def _long_short_weights(
    scores: pd.DataFrame,
    eligibility: pd.DataFrame,
    groups: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build accepted group-equal top/bottom weights at the frozen q=25%."""
    return single._leg_weights(scores, eligibility, groups)


def _group_budget_diagnostics(
    leg: str,
    weights: pd.DataFrame,
    groups: pd.DataFrame,
) -> pd.DataFrame:
    """Independently verify equal group and within-group budgets after netting."""
    rows: list[dict] = []
    for date in weights.index:
        labels = groups.loc[date]
        for side_name, side in (
            ("long", weights.loc[date].clip(lower=0.0)),
            ("short", -weights.loc[date].clip(upper=0.0)),
        ):
            selected = side.gt(0.0) & labels.notna()
            budgets = side.loc[selected].groupby(labels.loc[selected], sort=False).sum()
            group_count = int(len(budgets))
            expected = 1.0 / group_count if group_count else np.nan
            budget_error = (
                float((budgets - expected).abs().max()) if group_count else np.inf
            )
            within_errors = []
            for label in budgets.index:
                values = side.loc[selected & labels.eq(label)]
                within_errors.append(float(values.max() - values.min()))
            rows.append(
                {
                    "date": date,
                    "leg": leg,
                    "side": side_name,
                    "available_group_count": group_count,
                    "selected_assets": int(selected.sum()),
                    "side_weight_sum": float(side.sum()),
                    "side_weight_sum_abs_error": abs(float(side.sum()) - 1.0),
                    "max_group_budget_abs_error": budget_error,
                    "max_within_group_weight_range": max(within_errors, default=np.inf),
                    "status": "PASS"
                    if group_count
                    and abs(float(side.sum()) - 1.0) <= WEIGHT_TOLERANCE
                    and budget_error <= WEIGHT_TOLERANCE
                    and max(within_errors, default=np.inf) <= WEIGHT_TOLERANCE
                    else "FAIL",
                }
            )
    return pd.DataFrame(rows)


def _signal_inputs(
    data,
    dates: pd.DatetimeIndex,
    columns: pd.Index,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute the owner-selected production signal and no-look-ahead checks."""
    daily = _read_daily(columns)
    signal_prices, benchmark, monthly_returns, _ = prod._period_inputs(
        data, daily, SIGNAL_FREQUENCY
    )
    global_source, raw_source = compute_momentum_alpha(
        prices=signal_prices,
        benchmark_price=benchmark,
        returns_freq=SIGNAL_FREQUENCY,
        group_data=None,
        long_span=MOMENTUM_LONG_SPAN,
        short_span=MOMENTUM_SHORT_SPAN,
        vol_span=MOMENTUM_VOL_SPAN,
        mean_adj_type=qis.MeanAdjType.EWMA,
    )
    global_decision, timestamps = prod._asof_panel(global_source, dates)
    raw_decision, raw_timestamps = prod._asof_panel(raw_source, dates)
    if not timestamps.equals(raw_timestamps):
        raise AssertionError("raw and global signal timestamps differ")
    lookahead_days = timestamps.sub(timestamps.index).dt.days
    rebuilt = qis.to_returns(
        signal_prices, freq=SIGNAL_FREQUENCY, is_log_returns=True
    ).reindex_like(monthly_returns)
    differences = rebuilt.subtract(monthly_returns).abs().to_numpy()
    finite = differences[np.isfinite(differences)]
    roundtrip_error = float(finite.max()) if finite.size else 0.0
    preflight = pd.DataFrame(
        [
            {
                "max_signal_lookahead_days": int(lookahead_days.max()),
                "return_roundtrip_max_abs_error": roundtrip_error,
                "return_roundtrip_tolerance": SIGNAL_TOLERANCE,
                "first_signal_timestamp": timestamps.min(),
                "last_signal_timestamp": timestamps.max(),
                "status": "PASS"
                if int(lookahead_days.max()) <= 0
                and roundtrip_error <= SIGNAL_TOLERANCE
                else "FAIL",
            }
        ]
    )
    return global_decision, raw_decision, raw_source, preflight


def _missing_bics_table(
    data,
    bics: pd.Series,
    eligibility: pd.DataFrame,
) -> pd.DataFrame:
    """Enumerate every BICS-uncovered name and its headline eligibility history."""
    missing = bics.index[bics.isna()]
    columns = [
        "security",
        "ticker",
        "figi",
        "gics_sector",
        "first_constituent_date",
        "last_constituent_date",
    ]
    frame = data.taxonomy.reindex(missing)[columns].copy()
    frame.insert(0, "asset", frame.index.astype(str))
    eligible_counts = eligibility.reindex(columns=missing, fill_value=False).sum(axis=0)
    first_dates = []
    last_dates = []
    for asset in missing:
        active = eligibility.index[eligibility[asset]]
        first_dates.append(active.min() if len(active) else pd.NaT)
        last_dates.append(active.max() if len(active) else pd.NaT)
    frame["headline_eligible_decisions"] = eligible_counts.reindex(missing).to_numpy()
    frame["first_headline_eligible_date"] = first_dates
    frame["last_headline_eligible_date"] = last_dates
    frame["classification_action"] = "excluded_from_all_primary_legs"
    return frame.reset_index(drop=True)


def _coverage_table(
    eligibility: pd.DataFrame,
    primary_eligibility: pd.DataFrame,
    raw_scores: pd.DataFrame,
    sector_groups: pd.DataFrame,
) -> pd.DataFrame:
    """Report point-in-time BICS coverage and available sector counts."""
    rows = []
    for date in eligibility.index:
        all_mask = eligibility.loc[date]
        primary_mask = primary_eligibility.loc[date]
        scored = primary_mask & raw_scores.loc[date].notna()
        rows.append(
            {
                "date": date,
                "eligible_assets": int(all_mask.sum()),
                "bics_classified_eligible_assets": int(primary_mask.sum()),
                "bics_missing_eligible_assets": int((all_mask & ~primary_mask).sum()),
                "bics_coverage_share": float(primary_mask.sum() / all_mask.sum()),
                "valid_scored_assets": int(scored.sum()),
                "available_bics_sectors": int(
                    sector_groups.loc[date, scored].nunique(dropna=True)
                ),
            }
        )
    return pd.DataFrame(rows)


def _performance_comparison(performance: pd.DataFrame) -> pd.DataFrame:
    """Compare the cluster leg with sector and global ranking yardsticks."""
    primary = performance.loc[performance["is_primary"]].set_index("leg")
    cluster = primary.loc["cluster_M1_star"]
    rows = []
    for benchmark_leg in ("bics_sector", "global"):
        benchmark = primary.loc[benchmark_leg]
        row = {
            "analysis_window": WINDOW,
            "cluster_leg": "cluster_M1_star",
            "benchmark_leg": benchmark_leg,
        }
        for metric in COMPARISON_METRICS:
            row[f"cluster_{metric}"] = cluster[metric]
            row[f"benchmark_{metric}"] = benchmark[metric]
            row[f"delta_{metric}"] = cluster[metric] - benchmark[metric]
        row["beats_benchmark_net_return"] = row["delta_net_return_annualized"] > 0.0
        row["beats_benchmark_sharpe"] = row["delta_sharpe_rf0"] > 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def _acceptance_rows(
    *,
    preflight: pd.DataFrame,
    performance: pd.DataFrame,
    weights: Mapping[str, pd.DataFrame],
    eligibilities: Mapping[str, pd.DataFrame],
    exposures: Mapping[str, pd.DataFrame],
    side_validations: Mapping[str, pd.DataFrame],
    group_diagnostics: pd.DataFrame,
    missing_bics: pd.DataFrame,
    expected_missing_count: int,
    cluster_missing_observations: int,
) -> pd.DataFrame:
    """Assemble measured-versus-tolerance acceptance lines."""
    rows = [
        {
            "check": "signal_no_lookahead_days",
            "leg": "all",
            "measured": float(preflight.loc[0, "max_signal_lookahead_days"]),
            "tolerance": 0.0,
            "status": "PASS"
            if preflight.loc[0, "max_signal_lookahead_days"] <= 0
            else "FAIL",
        },
        {
            "check": "signal_return_roundtrip_abs_error",
            "leg": "all",
            "measured": float(preflight.loc[0, "return_roundtrip_max_abs_error"]),
            "tolerance": SIGNAL_TOLERANCE,
            "status": "PASS"
            if preflight.loc[0, "return_roundtrip_max_abs_error"] <= SIGNAL_TOLERANCE
            else "FAIL",
        },
        {
            "check": "missing_bics_rows_reported",
            "leg": "all",
            "measured": float(len(missing_bics)),
            "tolerance": float(expected_missing_count),
            "status": "PASS" if len(missing_bics) == expected_missing_count else "FAIL",
        },
        {
            "check": "eligible_cluster_membership_missing",
            "leg": "cluster_M1_star",
            "measured": float(cluster_missing_observations),
            "tolerance": 0.0,
            "status": "PASS" if cluster_missing_observations == 0 else "FAIL",
        },
        {
            "check": "performance_rows_complete",
            "leg": "all",
            "measured": float(len(performance)),
            "tolerance": 5.0,
            "status": "PASS" if len(performance) == 5 else "FAIL",
        },
    ]
    for leg, frame in weights.items():
        eligibility = eligibilities[leg]
        outside = frame.where(~eligibility, 0.0).abs().to_numpy()
        outside_max = float(outside.max()) if outside.size else 0.0
        exposure = exposures[leg]
        side = side_validations[leg]
        accepted = single._acceptance(WINDOW, leg, exposure, side)
        diagnostics = group_diagnostics.loc[group_diagnostics["leg"].eq(leg)]
        rows.extend(
            [
                {
                    "check": "weight_outside_eligibility_abs",
                    "leg": leg,
                    "measured": outside_max,
                    "tolerance": WEIGHT_TOLERANCE,
                    "status": "PASS" if outside_max <= WEIGHT_TOLERANCE else "FAIL",
                },
                {
                    "check": "long_short_exposure_abs_error",
                    "leg": leg,
                    "measured": max(
                        accepted["max_long_exposure_error"],
                        accepted["max_short_exposure_error"],
                        accepted["max_net_exposure_error"],
                        accepted["max_gross_exposure_error"],
                    ),
                    "tolerance": WEIGHT_TOLERANCE,
                    "status": accepted["status"],
                },
                {
                    "check": "post_net_equal_group_budget_abs_error",
                    "leg": leg,
                    "measured": float(
                        diagnostics["max_group_budget_abs_error"].max()
                    ),
                    "tolerance": WEIGHT_TOLERANCE,
                    "status": "PASS"
                    if diagnostics["max_group_budget_abs_error"].max()
                    <= WEIGHT_TOLERANCE
                    else "FAIL",
                },
                {
                    "check": "post_net_equal_stock_weight_range",
                    "leg": leg,
                    "measured": float(
                        diagnostics["max_within_group_weight_range"].max()
                    ),
                    "tolerance": WEIGHT_TOLERANCE,
                    "status": "PASS"
                    if diagnostics["max_within_group_weight_range"].max()
                    <= WEIGHT_TOLERANCE
                    else "FAIL",
                },
            ]
        )
    return pd.DataFrame(rows)


def run() -> Mapping[str, pd.DataFrame]:
    """Run the matched U1 cluster-versus-BICS-versus-global comparison."""
    started = time.perf_counter()
    dates, fixed_eligibility = _accepted_dates_and_eligibility()
    window_dates = e5._analysis_windows(UNIVERSE, dates)[WINDOW]
    data = e5.load_universe(UNIVERSE)
    columns = fixed_eligibility.columns
    all_eligibility = fixed_eligibility.reindex(index=window_dates).astype(bool)
    bics = data.taxonomy[SECTOR_COLUMN].reindex(columns).replace("", np.nan)
    classified = bics.notna()
    primary_eligibility = all_eligibility & classified.to_numpy()
    sector_groups = pd.DataFrame(
        np.tile(bics.to_numpy(), (len(window_dates), 1)),
        index=window_dates,
        columns=columns,
    )

    global_all, raw_all, raw_source, preflight = _signal_inputs(data, dates, columns)
    global_scores = global_all.reindex(index=window_dates, columns=columns)
    raw_scores = raw_all.reindex(index=window_dates, columns=columns)
    cluster_groups_all = e5._cluster_groups(UNIVERSE, CLUSTER_CONFIG).reindex(
        columns=columns
    )
    cluster_source = score_within_clusters(
        raw_signal=raw_source,
        rolling_clusters=prod._panel_dict(cluster_groups_all),
        min_cluster_size=CLUSTER_FALLBACK,
    )
    cluster_all, cluster_timestamps = prod._asof_panel(cluster_source, dates)
    cluster_scores = cluster_all.reindex(index=window_dates, columns=columns)
    cluster_scores = cluster_scores.where(raw_scores.notna())
    cluster_groups = cluster_groups_all.reindex(index=window_dates, columns=columns)
    global_groups = pd.DataFrame("global", index=window_dates, columns=columns)

    primary_cluster_missing = int(
        (primary_eligibility & cluster_groups.isna()).sum().sum()
    )
    if primary_cluster_missing:
        raise AssertionError(
            f"M1-star misses {primary_cluster_missing} primary eligible memberships"
        )
    if (cluster_timestamps.sub(cluster_timestamps.index).dt.days > 0).any():
        raise AssertionError("cluster score sampling looks ahead")

    leg_inputs = {
        "cluster_M1_star": (
            cluster_scores.where(primary_eligibility),
            primary_eligibility,
            cluster_groups,
            True,
            "group_equal",
        ),
        "bics_sector": (
            raw_scores.where(primary_eligibility),
            primary_eligibility,
            sector_groups,
            True,
            "sector_equal",
        ),
        "global": (
            global_scores.where(primary_eligibility),
            primary_eligibility,
            global_groups,
            True,
            "asset_equal",
        ),
        "cluster_M1_star_full_u1_robustness": (
            cluster_scores.where(all_eligibility),
            all_eligibility,
            cluster_groups,
            False,
            "group_equal",
        ),
        "global_full_u1_robustness": (
            global_scores.where(all_eligibility),
            all_eligibility,
            global_groups,
            False,
            "asset_equal",
        ),
    }

    weights: dict[str, pd.DataFrame] = {}
    exposures: dict[str, pd.DataFrame] = {}
    side_validations: dict[str, pd.DataFrame] = {}
    eligibilities: dict[str, pd.DataFrame] = {}
    group_diagnostic_rows = []
    performance_rows = []
    prices = e5._prices(data).reindex(columns=columns)
    ew_nav = _ew_navs()[WINDOW]
    costs = COST_BPS / 10000.0
    for leg, (scores, eligibility, groups, is_primary, construction) in leg_inputs.items():
        leg_weights, exposure, side_validation = _long_short_weights(
            scores, eligibility, groups
        )
        weights[leg] = leg_weights
        exposures[leg] = exposure
        side_validations[leg] = side_validation
        eligibilities[leg] = eligibility
        group_diagnostic_rows.append(
            _group_budget_diagnostics(leg, leg_weights, groups)
        )
        net, gross = _backtest(
            prices, leg_weights, costs, f"u1_bics_{leg}_long_short"
        )
        performance_rows.append(
            {
                "universe": UNIVERSE.value,
                "analysis_window": WINDOW,
                "universe_scope": "bics_classified_matched"
                if is_primary
                else "full_u1_robustness",
                "is_primary": is_primary,
                "leg": leg,
                "construction": construction,
                "q": Q,
                "strategy": "long_top_short_bottom",
                **grid_ls._performance_payload(net, gross, ew_nav),
                "runner": RUNNER,
            }
        )

    performance = pd.DataFrame(performance_rows)
    group_diagnostics = pd.concat(group_diagnostic_rows, ignore_index=True)
    missing_bics = _missing_bics_table(data, bics, all_eligibility)
    coverage = _coverage_table(
        all_eligibility, primary_eligibility, raw_scores, sector_groups
    )
    acceptance = _acceptance_rows(
        preflight=preflight,
        performance=performance,
        weights=weights,
        eligibilities=eligibilities,
        exposures=exposures,
        side_validations=side_validations,
        group_diagnostics=group_diagnostics,
        missing_bics=missing_bics,
        expected_missing_count=int(bics.isna().sum()),
        cluster_missing_observations=primary_cluster_missing,
    )
    if not acceptance["status"].eq("PASS").all():
        raise AssertionError(acceptance.loc[~acceptance["status"].eq("PASS")])

    exposure_table = pd.concat(
        [frame.assign(leg=leg) for leg, frame in exposures.items()],
        ignore_index=True,
    )
    side_table = pd.concat(
        [frame.assign(leg=leg) for leg, frame in side_validations.items()],
        ignore_index=True,
    )
    design = pd.DataFrame(
        [
            {
                **FROZEN_SPEC,
                "primary_legs": "|".join(PRIMARY_LEGS),
                "primary_universe": "point_in_time_eligible_and_bics_classified",
                "sector_budget_rule": "1/G per available sector per side",
                "stock_budget_rule": "equal among selected stocks within sector",
                "global_rule": "asset_equal top/bottom across matched universe",
                "cluster_rule": "group_equal top/bottom under U1 M1-star",
                "ew_role": "market reference for beta/alpha only",
                "runner": RUNNER,
            }
        ]
    )
    output = {
        "performance": performance,
        "comparison": _performance_comparison(performance),
        "coverage_per_date": coverage,
        "missing_bics_assets": missing_bics,
        "group_budget_diagnostics": group_diagnostics,
        "exposure_diagnostics": exposure_table,
        "side_budget_diagnostics": side_table,
        "signal_preflight": preflight,
        "acceptance": acceptance,
        "design": design,
        "runtime": pd.DataFrame(
            [{"runtime_seconds": time.perf_counter() - started, "runner": RUNNER}]
        ),
    }
    for name, frame in output.items():
        e5._write(frame, _root() / f"{name}.csv")
    return output


def _hash_outputs() -> dict[str, str]:
    """Hash deterministic outputs while excluding timing and replay records."""
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(_root().glob("*.csv"))
        if path.name not in {"runtime.csv", "determinism.csv"}
    }


def verify_determinism() -> pd.DataFrame:
    """Run twice and require byte-identical U1 comparison artifacts."""
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
    """Execute, replay, and print primary performance and comparisons."""
    replay = verify_determinism()
    performance = pd.read_csv(
        _root() / "performance.csv", float_precision="round_trip"
    )
    comparison = pd.read_csv(
        _root() / "comparison.csv", float_precision="round_trip"
    )
    print(performance.loc[performance["is_primary"]].to_string(index=False))
    print(comparison.to_string(index=False))
    print(
        f"U1 BICS sector comparison: PASS "
        f"({len(replay)}/{len(replay)} deterministic)",
        flush=True,
    )


if __name__ == "__main__":
    main()
