"""Compare BlackRock fund clusters, official asset classes, and global ranks.

The U1 equal-class long-short design is transferred to the 480-fund BlackRock
current-vintage universe.  The fund-specific cluster treatment remains the previously
selected W-THU/span-156 partition.  At each decision, every available official Aladdin
``asset_class`` receives equal budget separately on the long and short sides, and
selected funds split their class budget equally.  The cluster leg gives equal budget
to available statistical clusters; the global leg is asset-equal.

Two signals are compared without changing the portfolio construction: the selected
ROSAA production setting (ME, 12-month long span, no short span, volatility span 13,
EWMA mean adjustment) and classic monthly 12-minus-1 momentum.  Both use q=25%,
+1/-1 exposure, one-period implementation lag, and 10 bp one-way costs.
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
import papers.cluster_lineage_2026.replication.run_u1_bics_sector_comparison as u1
import papers.cluster_lineage_2026.replication.run_u1_covar_grid_long_short_monthly as classic
import papers.cluster_lineage_2026.replication.run_u1_covar_grid_long_short_prod as prod
import papers.cluster_lineage_2026.replication.run_u2_blackrock_etf_grid as funds


UNIVERSE = funds.UNIVERSE
WINDOW = funds.HEADLINE_WINDOW
WINDOW_START = funds.HEADLINE_START
WINDOW_END = funds.HEADLINE_END
Q = 0.25
COST_BPS = 10.0
CLUSTER_FREQUENCY = "W-THU"
CLUSTER_SPAN = 156
CLUSTER_FALLBACK = 5
CLASSIFICATION_COLUMN = "asset_class"
CLASSIFICATION_BUDGET = "equal_available_groups"
ROSAA_SIGNAL = "ME_12_none_13_EWMA"
CLASSIC_SIGNAL = "classic_monthly_12m_skip1"
SIGNALS = (ROSAA_SIGNAL, CLASSIC_SIGNAL)
SIGNAL_FREQUENCY = "ME"
MOMENTUM_LONG_SPAN = 12
MOMENTUM_SHORT_SPAN = None
MOMENTUM_VOL_SPAN = 13
LOOKBACK_MONTHS = 12
SKIP_MONTHS = 1
WEIGHT_TOLERANCE = 1e-12
SIGNAL_TOLERANCE = 1e-12
PRIMARY_LEGS = ("cluster", "asset_class", "global")
RUNNER = (
    "papers/cluster_lineage_2026/replication/"
    "run_u2_blackrock_signal_comparison.py"
)
FROZEN_SPEC = {
    "universe": UNIVERSE,
    "analysis_window": WINDOW,
    "strategy": "long_short",
    "q": Q,
    "cluster_frequency": CLUSTER_FREQUENCY,
    "cluster_span": CLUSTER_SPAN,
    "cluster_fallback": CLUSTER_FALLBACK,
    "classification_column": CLASSIFICATION_COLUMN,
    "classification_budget": CLASSIFICATION_BUDGET,
    "rosaa_signal": ROSAA_SIGNAL,
    "classic_signal": CLASSIC_SIGNAL,
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
    """Return and create the external fund signal-comparison directory."""
    root = funds._root() / "equal_asset_class_rosaa_vs_classic_20260815"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _long_short_weights(
    scores: pd.DataFrame,
    eligibility: pd.DataFrame,
    groups: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the accepted group-equal q=25% top-minus-bottom book."""
    return u1._long_short_weights(scores, eligibility, groups)


def _classic_scores(
    monthly_log_returns: pd.DataFrame,
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Return exactly 12 included monthly returns after a one-month skip."""
    return classic._classic_monthly_scores(
        monthly_log_returns,
        dates,
        lookback_months=LOOKBACK_MONTHS,
        skip_months=SKIP_MONTHS,
    )


def _classification(columns: pd.Index) -> pd.Series:
    """Return the complete official Aladdin asset-class mapping."""
    metadata = pd.read_csv(funds.METADATA_FILE).set_index("ticker")
    labels = metadata[CLASSIFICATION_COLUMN].reindex(columns)
    if labels.isna().any():
        missing = labels.index[labels.isna()].tolist()
        raise AssertionError(f"funds missing {CLASSIFICATION_COLUMN}: {missing}")
    labels.name = CLASSIFICATION_COLUMN
    return labels


def _group_panel(index: pd.DatetimeIndex, labels: pd.Series) -> pd.DataFrame:
    """Broadcast one static fund classification over decision dates."""
    return pd.DataFrame(
        np.tile(labels.to_numpy(), (len(index), 1)),
        index=index,
        columns=labels.index,
    )


def _rosaa_inputs(
    daily: pd.DataFrame,
    dates: pd.DatetimeIndex,
    eligibility: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute exact selected ROSAA scores and independent timing checks."""
    monthly_returns = funds._native_returns(daily, SIGNAL_FREQUENCY)
    monthly_eligibility = funds._eligibility_for_dates(daily, monthly_returns.index)
    simple_returns = np.expm1(monthly_returns)
    signal_prices = qis.returns_to_nav(simple_returns)
    benchmark_returns = simple_returns.where(monthly_eligibility).mean(axis=1)
    benchmark = qis.returns_to_nav(benchmark_returns.rename("EW").to_frame())["EW"]
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
        raise AssertionError("ROSAA raw/global timestamps differ")
    lookahead_days = timestamps.sub(timestamps.index).dt.days
    roundtrip = qis.to_returns(
        signal_prices, freq=SIGNAL_FREQUENCY, is_log_returns=True
    ).reindex_like(monthly_returns)
    differences = roundtrip.subtract(monthly_returns).abs().to_numpy()
    finite = differences[np.isfinite(differences)]
    roundtrip_error = float(finite.max()) if finite.size else 0.0
    valid_counts_full = global_decision.where(eligibility).notna().sum(axis=1)
    valid_counts = valid_counts_full.loc[WINDOW_START:WINDOW_END]
    preflight = pd.DataFrame(
        [
            {
                "signal": ROSAA_SIGNAL,
                "max_signal_lookahead_days": int(lookahead_days.max()),
                "max_reconstruction_abs_error": roundtrip_error,
                "tolerance": SIGNAL_TOLERANCE,
                "valid_assets_min": int(valid_counts.min()),
                "valid_assets_median": float(valid_counts.median()),
                "valid_assets_max": int(valid_counts.max()),
                "warmup_valid_assets_min": int(valid_counts_full.min()),
                "status": "PASS"
                if int(lookahead_days.max()) <= 0
                and roundtrip_error <= SIGNAL_TOLERANCE
                and int(valid_counts.min()) > 0
                else "FAIL",
            }
        ]
    )
    return global_decision, raw_decision, raw_source, preflight


def _classic_inputs(
    daily: pd.DataFrame,
    dates: pd.DatetimeIndex,
    eligibility: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute classic scores and reconstruct them through explicit history slices."""
    monthly_returns = funds._native_returns(daily, SIGNAL_FREQUENCY)
    scores = _classic_scores(monthly_returns, dates)
    regression = classic._independent_score_regression(monthly_returns, dates, scores)
    valid_counts_full = scores.where(eligibility).notna().sum(axis=1)
    valid_counts = valid_counts_full.loc[WINDOW_START:WINDOW_END]
    preflight = pd.DataFrame(
        [
            {
                "signal": CLASSIC_SIGNAL,
                "max_signal_lookahead_days": 0,
                "max_reconstruction_abs_error": float(regression.loc[0, "max_abs_error"]),
                "tolerance": classic.SCORE_REGRESSION_TOLERANCE,
                "valid_assets_min": int(valid_counts.min()),
                "valid_assets_median": float(valid_counts.median()),
                "valid_assets_max": int(valid_counts.max()),
                "warmup_valid_assets_min": int(valid_counts_full.min()),
                "nan_mask_match": bool(regression.loc[0, "nan_mask_match"]),
                "status": "PASS"
                if regression.loc[0, "status"] == "PASS"
                and int(valid_counts.min()) > 0
                else "FAIL",
            }
        ]
    )
    return scores, preflight


def _comparison(performance: pd.DataFrame) -> pd.DataFrame:
    """Compare cluster returns with classification and global ranks by signal."""
    indexed = performance.set_index(["signal", "leg"])
    rows = []
    for signal in SIGNALS:
        cluster = indexed.loc[(signal, "cluster")]
        for benchmark_leg in ("asset_class", "global"):
            benchmark = indexed.loc[(signal, benchmark_leg)]
            row = {
                "analysis_window": WINDOW,
                "signal": signal,
                "cluster_leg": "cluster",
                "benchmark_leg": benchmark_leg,
            }
            for metric in COMPARISON_METRICS:
                row[f"cluster_{metric}"] = cluster[metric]
                row[f"benchmark_{metric}"] = benchmark[metric]
                row[f"delta_{metric}"] = cluster[metric] - benchmark[metric]
            row["beats_benchmark_net_return"] = (
                row["delta_net_return_annualized"] > 0.0
            )
            row["lower_volatility_than_benchmark"] = (
                row["delta_volatility_annualized"] < 0.0
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _signal_comparison(performance: pd.DataFrame) -> pd.DataFrame:
    """Compare classic with ROSAA for each unchanged portfolio leg."""
    indexed = performance.set_index(["signal", "leg"])
    rows = []
    for leg in PRIMARY_LEGS:
        rosaa = indexed.loc[(ROSAA_SIGNAL, leg)]
        classic_row = indexed.loc[(CLASSIC_SIGNAL, leg)]
        row = {"analysis_window": WINDOW, "leg": leg}
        for metric in COMPARISON_METRICS:
            row[f"rosaa_{metric}"] = rosaa[metric]
            row[f"classic_{metric}"] = classic_row[metric]
            row[f"classic_minus_rosaa_{metric}"] = (
                classic_row[metric] - rosaa[metric]
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _coverage(
    eligibility: pd.DataFrame,
    labels: pd.DataFrame,
    scores_by_signal: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Report point-in-time fund and available-class counts for both signals."""
    rows = []
    for signal, scores in scores_by_signal.items():
        for date in eligibility.index:
            valid = eligibility.loc[date] & scores.loc[date].notna()
            rows.append(
                {
                    "date": date,
                    "signal": signal,
                    "eligible_funds": int(eligibility.loc[date].sum()),
                    "valid_scored_funds": int(valid.sum()),
                    "available_asset_classes": int(
                        labels.loc[date, valid].nunique(dropna=True)
                    ),
                }
            )
    return pd.DataFrame(rows)


def _acceptance(
    *,
    preflight: pd.DataFrame,
    performance: pd.DataFrame,
    weights: Mapping[tuple[str, str], pd.DataFrame],
    exposures: Mapping[tuple[str, str], pd.DataFrame],
    side_validations: Mapping[tuple[str, str], pd.DataFrame],
    groups: Mapping[str, pd.DataFrame],
    eligibility: pd.DataFrame,
    classification_count: int,
    cluster_membership_missing: int,
) -> pd.DataFrame:
    """Return measured-versus-tolerance checks for every signal and leg."""
    rows = [
        {
            "check": "official_classifications_present",
            "signal": "all",
            "leg": "all",
            "measured": float(classification_count),
            "tolerance": float(performance.shape[0] * 0 + 480),
            "status": "PASS" if classification_count == 480 else "FAIL",
        },
        {
            "check": "eligible_cluster_membership_missing",
            "signal": "all",
            "leg": "cluster",
            "measured": float(cluster_membership_missing),
            "tolerance": 0.0,
            "status": "PASS" if cluster_membership_missing == 0 else "FAIL",
        },
        {
            "check": "performance_rows_complete",
            "signal": "all",
            "leg": "all",
            "measured": float(len(performance)),
            "tolerance": 6.0,
            "status": "PASS" if len(performance) == 6 else "FAIL",
        },
    ]
    for _, signal_row in preflight.iterrows():
        rows.append(
            {
                "check": "signal_preflight",
                "signal": signal_row["signal"],
                "leg": "all",
                "measured": float(signal_row["max_reconstruction_abs_error"]),
                "tolerance": float(signal_row["tolerance"]),
                "status": signal_row["status"],
            }
        )
    for key, frame in weights.items():
        signal, leg = key
        outside = frame.where(~eligibility, 0.0).abs().to_numpy()
        outside_max = float(outside.max()) if outside.size else 0.0
        accepted = u1.single._acceptance(
            WINDOW, f"{signal}:{leg}", exposures[key], side_validations[key]
        )
        diagnostics = u1._group_budget_diagnostics(leg, frame, groups[leg])
        budget_error = float(diagnostics["max_group_budget_abs_error"].max())
        stock_error = float(diagnostics["max_within_group_weight_range"].max())
        rows.extend(
            [
                {
                    "check": "weight_outside_eligibility_abs",
                    "signal": signal,
                    "leg": leg,
                    "measured": outside_max,
                    "tolerance": WEIGHT_TOLERANCE,
                    "status": "PASS" if outside_max <= WEIGHT_TOLERANCE else "FAIL",
                },
                {
                    "check": "long_short_exposure_abs_error",
                    "signal": signal,
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
                    "check": "equal_group_budget_abs_error",
                    "signal": signal,
                    "leg": leg,
                    "measured": budget_error,
                    "tolerance": WEIGHT_TOLERANCE,
                    "status": "PASS"
                    if budget_error <= WEIGHT_TOLERANCE
                    else "FAIL",
                },
                {
                    "check": "equal_selected_fund_weight_range",
                    "signal": signal,
                    "leg": leg,
                    "measured": stock_error,
                    "tolerance": WEIGHT_TOLERANCE,
                    "status": "PASS" if stock_error <= WEIGHT_TOLERANCE else "FAIL",
                },
            ]
        )
    return pd.DataFrame(rows)


def run() -> Mapping[str, pd.DataFrame]:
    """Run both signals over matched cluster, asset-class, and global books."""
    started = time.perf_counter()
    dates = funds._dates()
    window_dates = dates[(dates >= WINDOW_START) & (dates <= WINDOW_END)]
    daily = funds._read_daily()
    eligibility_all = funds._eligibility_for_dates(daily, dates)
    eligibility = eligibility_all.reindex(index=window_dates).astype(bool)
    columns = eligibility.columns
    asset_class = _classification(columns)
    class_groups = _group_panel(window_dates, asset_class)
    global_groups = pd.DataFrame("global", index=window_dates, columns=columns)
    cluster_groups_all, _ = funds._load_partition(CLUSTER_FREQUENCY, CLUSTER_SPAN)
    cluster_groups = cluster_groups_all.reindex(index=window_dates, columns=columns)
    cluster_membership_missing = int((eligibility & cluster_groups.isna()).sum().sum())
    if cluster_membership_missing:
        raise AssertionError(
            f"selected fund partition misses {cluster_membership_missing} eligible memberships"
        )

    rosaa_global_all, rosaa_raw_all, rosaa_raw_source, rosaa_preflight = _rosaa_inputs(
        daily, dates, eligibility_all
    )
    classic_all, classic_preflight = _classic_inputs(daily, dates, eligibility_all)
    rosaa_cluster_source = score_within_clusters(
        raw_signal=rosaa_raw_source,
        rolling_clusters=funds._panel_dict(cluster_groups_all),
        min_cluster_size=CLUSTER_FALLBACK,
    )
    rosaa_cluster_all, _ = prod._asof_panel(rosaa_cluster_source, dates)
    classic_cluster_all = score_within_clusters(
        raw_signal=classic_all,
        rolling_clusters=funds._panel_dict(cluster_groups_all),
        min_cluster_size=CLUSTER_FALLBACK,
    )

    scores_by_signal = {
        ROSAA_SIGNAL: {
            "cluster": rosaa_cluster_all.reindex(index=window_dates, columns=columns).where(
                rosaa_raw_all.reindex(index=window_dates, columns=columns).notna()
            ),
            "asset_class": rosaa_raw_all.reindex(index=window_dates, columns=columns),
            "global": rosaa_global_all.reindex(index=window_dates, columns=columns),
        },
        CLASSIC_SIGNAL: {
            "cluster": classic_cluster_all.reindex(index=window_dates, columns=columns).where(
                classic_all.reindex(index=window_dates, columns=columns).notna()
            ),
            "asset_class": classic_all.reindex(index=window_dates, columns=columns),
            "global": classic_all.reindex(index=window_dates, columns=columns),
        },
    }
    groups = {
        "cluster": cluster_groups,
        "asset_class": class_groups,
        "global": global_groups,
    }
    prices_all = funds._performance_prices(daily)
    prices = funds._window_prices(prices_all, window_dates)
    ew_nav = funds._ew_reference(prices_all, eligibility_all, window_dates, WINDOW)
    costs = COST_BPS / 10000.0
    weights: dict[tuple[str, str], pd.DataFrame] = {}
    exposures: dict[tuple[str, str], pd.DataFrame] = {}
    side_validations: dict[tuple[str, str], pd.DataFrame] = {}
    group_diagnostic_rows = []
    performance_rows = []
    for signal in SIGNALS:
        for leg in PRIMARY_LEGS:
            scores = scores_by_signal[signal][leg].where(eligibility)
            leg_weights, exposure, side_validation = _long_short_weights(
                scores, eligibility, groups[leg]
            )
            key = (signal, leg)
            weights[key] = leg_weights
            exposures[key] = exposure
            side_validations[key] = side_validation
            diagnostics = u1._group_budget_diagnostics(leg, leg_weights, groups[leg])
            diagnostics.insert(1, "signal", signal)
            group_diagnostic_rows.append(diagnostics)
            net, gross = funds._backtest(
                prices,
                leg_weights,
                costs,
                f"{UNIVERSE}_{signal}_{leg}_long_short",
            )
            performance_rows.append(
                {
                    "universe": UNIVERSE,
                    "analysis_window": WINDOW,
                    "signal": signal,
                    "leg": leg,
                    "construction": {
                        "cluster": "group_equal",
                        "asset_class": "class_equal",
                        "global": "asset_equal",
                    }[leg],
                    "cluster_frequency": CLUSTER_FREQUENCY if leg == "cluster" else np.nan,
                    "cluster_span": CLUSTER_SPAN if leg == "cluster" else np.nan,
                    "q": Q,
                    "strategy": "long_top_short_bottom",
                    **funds._performance_payload(net, gross, ew_nav),
                    "runner": RUNNER,
                }
            )

    performance = pd.DataFrame(performance_rows)
    preflight = pd.concat([rosaa_preflight, classic_preflight], ignore_index=True)
    acceptance = _acceptance(
        preflight=preflight,
        performance=performance,
        weights=weights,
        exposures=exposures,
        side_validations=side_validations,
        groups=groups,
        eligibility=eligibility,
        classification_count=int(asset_class.notna().sum()),
        cluster_membership_missing=cluster_membership_missing,
    )
    if not acceptance["status"].eq("PASS").all():
        raise AssertionError(acceptance.loc[~acceptance["status"].eq("PASS")])

    classification_summary = (
        asset_class.value_counts().rename_axis("asset_class").rename("current_funds")
        .reset_index()
    )
    classification_summary["classification_source"] = (
        "BlackRock Aladdin / iShares product screener"
    )
    coverage_scores = {
        signal: scores_by_signal[signal]["global"] for signal in SIGNALS
    }
    design = pd.DataFrame(
        [
            {
                **FROZEN_SPEC,
                "primary_legs": "|".join(PRIMARY_LEGS),
                "classification_groups": int(asset_class.nunique()),
                "classification_coverage": "480/480 current-vintage funds",
                "equal_class_rule": "1/G per available asset class per side",
                "equal_fund_rule": "equal among selected funds within class",
                "strategic_50_30_20_overlay": False,
                "ew_role": "market reference for beta/alpha only",
                "current_cohort_warning": "not survivorship-free",
                "runner": RUNNER,
            }
        ]
    )
    output = {
        "performance": performance,
        "comparison": _comparison(performance),
        "signal_comparison": _signal_comparison(performance),
        "coverage_per_date": _coverage(
            eligibility, class_groups, coverage_scores
        ),
        "classification_summary": classification_summary,
        "group_budget_diagnostics": pd.concat(
            group_diagnostic_rows, ignore_index=True
        ),
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
    """Run twice and require byte-identical fund comparison artifacts."""
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
    """Execute, replay, and print fund performance and comparisons."""
    replay = verify_determinism()
    performance = pd.read_csv(
        _root() / "performance.csv", float_precision="round_trip"
    )
    comparison = pd.read_csv(
        _root() / "comparison.csv", float_precision="round_trip"
    )
    signal_comparison = pd.read_csv(
        _root() / "signal_comparison.csv", float_precision="round_trip"
    )
    print(performance.to_string(index=False))
    print(comparison.to_string(index=False))
    print(signal_comparison.to_string(index=False))
    print(
        f"BlackRock fund signal comparison: PASS "
        f"({len(replay)}/{len(replay)} deterministic)",
        flush=True,
    )


if __name__ == "__main__":
    main()
