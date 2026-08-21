"""Run the canonical U1 signal-isolation comparison with classic momentum.

All three legs use the public OptimalPortfolios signal and rank APIs.  The global,
BICS-sector, and M1-star variants differ only in how the same 12-minus-1 raw
momentum is cross-sectionally standardised.  Each resulting score panel is ranked
once across the matched investable universe; clusters and sectors are not reused
as portfolio-budget groups.  Long and short books each have unit exposure, use
q=25%, a one-period implementation lag, and 10 bp one-way costs.
"""
from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import qis
from optimalportfolios.alphas import (
    compute_classic_momentum_alpha,
    compute_classic_momentum_cluster_alpha,
    compute_top_quantile_equal_weights,
)

import papers.cluster_lineage_2026.replication.run_backtests as e5
import papers.cluster_lineage_2026.replication.run_u1_bics_sector_comparison as base
import papers.cluster_lineage_2026.replication.run_u1_covar_grid_long_short as grid_ls
import papers.cluster_lineage_2026.replication.run_u1_covar_grid_long_short_monthly as classic
import papers.cluster_lineage_2026.replication.run_u1_covar_grid_long_short_prod as prod
from papers.cluster_lineage_2026.replication.run_u1_covar_grid import (
    UNIVERSE,
    _accepted_dates_and_eligibility,
    _backtest,
    _ew_navs,
    _native_returns,
    _read_daily,
)


WINDOW = base.WINDOW
Q = base.Q
COST_BPS = base.COST_BPS
SIGNAL_VARIANT = "classic_monthly_12m_skip1"
SIGNAL_FREQUENCY = "ME"
LOOKBACK_MONTHS = 12
SKIP_MONTHS = 1
CLUSTER_CONFIG = base.CLUSTER_CONFIG
CLUSTER_DELTA = base.CLUSTER_DELTA
CLUSTER_FALLBACK = base.CLUSTER_FALLBACK
SECTOR_COLUMN = base.SECTOR_COLUMN
MISSING_SECTOR_POLICY = base.MISSING_SECTOR_POLICY
PRIMARY_LEGS = base.PRIMARY_LEGS
WEIGHT_TOLERANCE = base.WEIGHT_TOLERANCE
SIGNAL_TOLERANCE = 1e-12
RUNNER = (
    "papers/cluster_lineage_2026/replication/"
    "run_u1_bics_sector_comparison_classic.py"
)
FROZEN_SPEC = {
    "universe": UNIVERSE.value,
    "analysis_window": WINDOW,
    "strategy": "long_short",
    "q": Q,
    "signal_variant": SIGNAL_VARIANT,
    "signal_frequency": SIGNAL_FREQUENCY,
    "lookback_months_included": LOOKBACK_MONTHS,
    "skip_months": SKIP_MONTHS,
    "volatility_adjustment": False,
    "cluster_config": CLUSTER_CONFIG.value,
    "cluster_delta": CLUSTER_DELTA,
    "cluster_fallback": CLUSTER_FALLBACK,
    "sector_column": SECTOR_COLUMN,
    "missing_sector_policy": MISSING_SECTOR_POLICY,
    "cost_bps_one_way": COST_BPS,
    "implementation_lag_periods": 1,
}
def _root() -> Path:
    """Return the gitignored local directory for the canonical result set."""
    root = (
        Path(__file__).resolve().parents[1]
        / "local_outputs"
        / "e5b"
        / "u1_bics_sector_vs_m1_star_classic_12m_skip1_canonical_20260816"
    )
    root.mkdir(parents=True, exist_ok=True)
    return root


def _classic_scores(
    monthly_log_returns: pd.DataFrame,
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Return exactly 12 completed monthly log returns after a one-month skip."""
    return classic._classic_monthly_scores(
        monthly_log_returns,
        dates,
        lookback_months=LOOKBACK_MONTHS,
        skip_months=SKIP_MONTHS,
    )


def _panel_error(left: pd.DataFrame, right: pd.DataFrame) -> tuple[float, bool]:
    """Return maximum finite absolute error and exact NaN-mask agreement."""
    left, right = left.align(right, join="outer")
    difference = left.subtract(right).abs().to_numpy()
    finite = difference[np.isfinite(difference)]
    error = float(finite.max()) if finite.size else 0.0
    return error, bool(left.isna().equals(right.isna()))


def _point_in_time_sector_membership(
    bics: pd.Series,
    eligibility: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return BICS labels only while each stock is an eligible U1 index member."""
    membership = pd.DataFrame(
        np.tile(bics.reindex(eligibility.columns).to_numpy(), (len(eligibility), 1)),
        index=eligibility.index,
        columns=eligibility.columns,
    ).where(eligibility.fillna(False).astype(bool))
    expected = eligibility.fillna(False).astype(bool) & bics.notna().to_numpy()
    assigned = membership.notna()
    diagnostics = pd.DataFrame(
        {
            "date": eligibility.index,
            "eligible_index_members": eligibility.sum(axis=1).to_numpy(),
            "classified_eligible_index_members": expected.sum(axis=1).to_numpy(),
            "assigned_sector_members": assigned.sum(axis=1).to_numpy(),
            "assignments_outside_index": (
                assigned & ~eligibility.fillna(False).astype(bool)
            ).sum(axis=1).to_numpy(),
            "missing_assignments_for_classified_members": (
                expected & ~assigned
            ).sum(axis=1).to_numpy(),
            "available_sectors": membership.nunique(axis=1, dropna=True).to_numpy(),
        }
    )
    return membership, diagnostics


def _classic_signal_panels(
    monthly_log_returns: pd.DataFrame,
    dates: pd.DatetimeIndex,
    eligibility: pd.DataFrame,
    clusters: pd.DataFrame,
    sector_membership: pd.DataFrame,
    min_cluster_size: int = CLUSTER_FALLBACK,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build all U1 score panels through the same public APIs used by U2."""
    signal_prices = qis.returns_to_nav(np.expm1(monthly_log_returns))
    global_source, global_raw = compute_classic_momentum_alpha(
        prices=signal_prices,
        returns_freq=SIGNAL_FREQUENCY,
        group_data=None,
        lookback_periods=LOOKBACK_MONTHS,
        skip_periods=SKIP_MONTHS,
    )
    cluster_source, cluster_raw = compute_classic_momentum_cluster_alpha(
        prices=signal_prices,
        rolling_clusters=prod._panel_dict(clusters),
        returns_freq=SIGNAL_FREQUENCY,
        lookback_periods=LOOKBACK_MONTHS,
        skip_periods=SKIP_MONTHS,
        min_cluster_size=min_cluster_size,
    )
    sector_source, sector_raw = compute_classic_momentum_cluster_alpha(
        prices=signal_prices,
        rolling_clusters=prod._panel_dict(sector_membership),
        returns_freq=SIGNAL_FREQUENCY,
        lookback_periods=LOOKBACK_MONTHS,
        skip_periods=SKIP_MONTHS,
        min_cluster_size=min_cluster_size,
    )

    cluster_raw_error, cluster_raw_nan_match = _panel_error(
        global_raw, cluster_raw
    )
    sector_raw_error, sector_raw_nan_match = _panel_error(
        global_raw, sector_raw
    )
    if cluster_raw_error > 0.0 or not cluster_raw_nan_match:
        raise AssertionError("global and cluster classic raw panels differ")
    if sector_raw_error > 0.0 or not sector_raw_nan_match:
        raise AssertionError("global and BICS classic raw panels differ")

    global_scores, global_timestamps = prod._asof_panel(global_source, dates)
    cluster_scores, cluster_timestamps = prod._asof_panel(cluster_source, dates)
    sector_scores, sector_timestamps = prod._asof_panel(sector_source, dates)
    raw_scores, raw_timestamps = prod._asof_panel(global_raw, dates)
    panels = (global_scores, cluster_scores, sector_scores, raw_scores)
    panels = tuple(
        panel.reindex(index=dates, columns=eligibility.columns).where(eligibility)
        for panel in panels
    )
    direct_raw = _classic_scores(monthly_log_returns, dates).where(eligibility)
    direct_error, _ = _panel_error(panels[-1], direct_raw)
    diagnostics = pd.DataFrame(
        [
            {
                "signal_variant": SIGNAL_VARIANT,
                "max_global_lookahead_days": float(
                    global_timestamps.sub(global_timestamps.index).dt.days.max()
                ),
                "max_cluster_lookahead_days": float(
                    cluster_timestamps.sub(cluster_timestamps.index).dt.days.max()
                ),
                "max_sector_lookahead_days": float(
                    sector_timestamps.sub(sector_timestamps.index).dt.days.max()
                ),
                "max_raw_lookahead_days": float(
                    raw_timestamps.sub(raw_timestamps.index).dt.days.max()
                ),
                "classic_raw_global_cluster_max_abs_error": cluster_raw_error,
                "classic_raw_global_cluster_nan_mask_match": cluster_raw_nan_match,
                "classic_raw_global_sector_max_abs_error": sector_raw_error,
                "classic_raw_global_sector_nan_mask_match": sector_raw_nan_match,
                "public_vs_direct_raw_max_abs_error": direct_error,
            }
        ]
    )
    return (*panels, diagnostics)


def _canonical_long_short_weights(
    scores: pd.DataFrame,
    prices: pd.DataFrame,
    eligibility: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Rank once across the eligible universe using the canonical OP function."""
    prices = prices.reindex(columns=scores.columns).reindex(
        index=scores.index, method="ffill"
    )
    eligibility = eligibility.reindex_like(scores).fillna(False).astype(bool)
    available_scores = scores.where(eligibility)
    available_prices = prices.where(eligibility)
    long_book = compute_top_quantile_equal_weights(
        alpha_scores=available_scores,
        prices=available_prices,
        quantile=Q,
    )
    short_book = compute_top_quantile_equal_weights(
        alpha_scores=-available_scores,
        prices=available_prices,
        quantile=Q,
    )
    overlap = long_book.gt(0.0) & short_book.gt(0.0)
    if overlap.to_numpy().any():
        raise AssertionError("canonical U1 top and bottom quantiles overlap")
    weights = long_book - short_book
    valid = available_scores.notna() & available_prices.notna()
    expected = np.ceil(Q * valid.sum(axis=1)).astype(int)
    exposure = pd.DataFrame(
        {
            "date": scores.index,
            "long_assets": long_book.gt(0.0).sum(axis=1).to_numpy(),
            "short_assets": short_book.gt(0.0).sum(axis=1).to_numpy(),
            "long_exposure": long_book.sum(axis=1).to_numpy(),
            "short_exposure_abs": short_book.sum(axis=1).to_numpy(),
            "net_exposure": weights.sum(axis=1).to_numpy(),
            "gross_exposure": weights.abs().sum(axis=1).to_numpy(),
        }
    )
    rank_diagnostics = pd.DataFrame(
        {
            "date": scores.index,
            "valid_assets": valid.sum(axis=1).to_numpy(),
            "expected_assets_per_side": expected.to_numpy(),
            "long_assets": long_book.gt(0.0).sum(axis=1).to_numpy(),
            "short_assets": short_book.gt(0.0).sum(axis=1).to_numpy(),
            "long_weight_sum_abs_error": long_book.sum(axis=1).sub(1.0).abs().to_numpy(),
            "short_weight_sum_abs_error": short_book.sum(axis=1).sub(1.0).abs().to_numpy(),
            "overlap_assets": overlap.sum(axis=1).to_numpy(),
        }
    )
    return weights, exposure, rank_diagnostics


def _acceptance_rows(
    *,
    signal_diagnostics: pd.DataFrame,
    performance: pd.DataFrame,
    weights: Mapping[str, pd.DataFrame],
    eligibilities: Mapping[str, pd.DataFrame],
    exposures: Mapping[str, pd.DataFrame],
    rank_diagnostics: pd.DataFrame,
    sector_membership_diagnostics: pd.DataFrame,
    missing_bics: pd.DataFrame,
    expected_missing_count: int,
    cluster_missing_observations: int,
) -> pd.DataFrame:
    """Assemble canonical signal and rank measured-versus-tolerance checks."""
    diagnostics = signal_diagnostics.iloc[0]
    rows = [
        {
            "check": "signal_no_lookahead_days",
            "leg": "all",
            "measured": max(
                float(diagnostics[column])
                for column in (
                    "max_global_lookahead_days",
                    "max_cluster_lookahead_days",
                    "max_sector_lookahead_days",
                    "max_raw_lookahead_days",
                )
            ),
            "tolerance": 0.0,
        },
        {
            "check": "classic_raw_global_cluster_abs_error",
            "leg": "all",
            "measured": float(
                diagnostics["classic_raw_global_cluster_max_abs_error"]
            ),
            "tolerance": 0.0,
        },
        {
            "check": "classic_raw_global_sector_abs_error",
            "leg": "all",
            "measured": float(
                diagnostics["classic_raw_global_sector_max_abs_error"]
            ),
            "tolerance": 0.0,
        },
        {
            "check": "public_vs_direct_raw_abs_error",
            "leg": "all",
            "measured": float(diagnostics["public_vs_direct_raw_max_abs_error"]),
            "tolerance": SIGNAL_TOLERANCE,
        },
        {
            "check": "missing_bics_rows_reported",
            "leg": "all",
            "measured": float(len(missing_bics)),
            "tolerance": float(expected_missing_count),
        },
        {
            "check": "eligible_cluster_membership_missing",
            "leg": "cluster_M1_star",
            "measured": float(cluster_missing_observations),
            "tolerance": 0.0,
        },
        {
            "check": "performance_rows_complete",
            "leg": "all",
            "measured": float(len(performance)),
            "tolerance": 5.0,
        },
        {
            "check": "sector_assignments_outside_index",
            "leg": "bics_sector",
            "measured": float(
                sector_membership_diagnostics[
                    "assignments_outside_index"
                ].max()
            ),
            "tolerance": 0.0,
        },
        {
            "check": "sector_assignments_missing_for_classified_members",
            "leg": "bics_sector",
            "measured": float(
                sector_membership_diagnostics[
                    "missing_assignments_for_classified_members"
                ].max()
            ),
            "tolerance": 0.0,
        },
    ]
    exact_checks = {
        "missing_bics_rows_reported",
        "performance_rows_complete",
    }
    for row in rows:
        if row["check"] in exact_checks:
            row["status"] = (
                "PASS" if row["measured"] == row["tolerance"] else "FAIL"
            )
        else:
            row["status"] = (
                "PASS" if row["measured"] <= row["tolerance"] else "FAIL"
            )

    for leg, frame in weights.items():
        eligibility = eligibilities[leg]
        outside = frame.where(~eligibility, 0.0).abs().to_numpy()
        outside_max = float(outside.max()) if outside.size else 0.0
        exposure = exposures[leg]
        ranked = rank_diagnostics.loc[rank_diagnostics["leg"].eq(leg)]
        exposure_error = max(
            float(exposure["long_exposure"].sub(1.0).abs().max()),
            float(exposure["short_exposure_abs"].sub(1.0).abs().max()),
            float(exposure["net_exposure"].abs().max()),
            float(exposure["gross_exposure"].sub(2.0).abs().max()),
        )
        selection_error = max(
            float(
                ranked["long_assets"]
                .sub(ranked["expected_assets_per_side"])
                .abs()
                .max()
            ),
            float(
                ranked["short_assets"]
                .sub(ranked["expected_assets_per_side"])
                .abs()
                .max()
            ),
        )
        for check, measured, tolerance in (
            ("weight_outside_eligibility_abs", outside_max, WEIGHT_TOLERANCE),
            ("long_short_exposure_abs_error", exposure_error, WEIGHT_TOLERANCE),
            ("canonical_rank_selection_count_error", selection_error, 0.0),
            ("canonical_rank_overlap_assets", float(ranked["overlap_assets"].max()), 0.0),
        ):
            rows.append(
                {
                    "check": check,
                    "leg": leg,
                    "measured": measured,
                    "tolerance": tolerance,
                    "status": "PASS" if measured <= tolerance else "FAIL",
                }
            )
    return pd.DataFrame(rows)


def run() -> Mapping[str, pd.DataFrame]:
    """Run the classic-signal cluster, BICS-sector, and global comparison."""
    started = time.perf_counter()
    dates, fixed_eligibility = _accepted_dates_and_eligibility()
    window_dates = e5._analysis_windows(UNIVERSE, dates)[WINDOW]
    data = e5.load_universe(UNIVERSE)
    columns = fixed_eligibility.columns
    all_eligibility = fixed_eligibility.reindex(index=window_dates).astype(bool)
    bics = data.taxonomy[SECTOR_COLUMN].reindex(columns).replace("", np.nan)
    classified = bics.notna()
    primary_eligibility = all_eligibility & classified.to_numpy()
    sector_membership_all, sector_membership_diagnostics_all = (
        _point_in_time_sector_membership(bics, fixed_eligibility)
    )
    sector_groups = sector_membership_all.reindex(index=window_dates, columns=columns)
    sector_membership_diagnostics = sector_membership_diagnostics_all.loc[
        sector_membership_diagnostics_all["date"].isin(window_dates)
    ].reset_index(drop=True)

    daily_returns = _read_daily(columns)
    monthly_returns = _native_returns(daily_returns, SIGNAL_FREQUENCY)
    direct_scores = _classic_scores(monthly_returns, dates)
    signal_regression = classic._independent_score_regression(
        monthly_returns, dates, direct_scores
    )
    cluster_groups_all = e5._cluster_groups(UNIVERSE, CLUSTER_CONFIG).reindex(
        columns=columns
    )
    global_all, cluster_all, sector_all, raw_all, signal_diagnostics = (
        _classic_signal_panels(
            monthly_returns,
            dates,
            fixed_eligibility,
            cluster_groups_all,
            sector_membership_all,
        )
    )
    global_scores = global_all.reindex(index=window_dates, columns=columns)
    cluster_scores = cluster_all.reindex(index=window_dates, columns=columns)
    sector_scores = sector_all.reindex(index=window_dates, columns=columns)
    raw_scores = raw_all.reindex(index=window_dates, columns=columns)
    cluster_groups = cluster_groups_all.reindex(index=window_dates, columns=columns)
    primary_cluster_missing = int(
        (primary_eligibility & cluster_groups.isna()).sum().sum()
    )
    if primary_cluster_missing:
        raise AssertionError(
            f"M1-star misses {primary_cluster_missing} primary eligible memberships"
        )

    leg_inputs = {
        "cluster_M1_star": (
            cluster_scores.where(primary_eligibility),
            primary_eligibility,
            True,
            "canonical_op_global_rank_equal_weight",
        ),
        "bics_sector": (
            sector_scores.where(primary_eligibility),
            primary_eligibility,
            True,
            "canonical_op_global_rank_equal_weight",
        ),
        "global": (
            global_scores.where(primary_eligibility),
            primary_eligibility,
            True,
            "canonical_op_global_rank_equal_weight",
        ),
        "cluster_M1_star_full_u1_robustness": (
            cluster_scores.where(all_eligibility),
            all_eligibility,
            False,
            "canonical_op_global_rank_equal_weight",
        ),
        "global_full_u1_robustness": (
            global_scores.where(all_eligibility),
            all_eligibility,
            False,
            "canonical_op_global_rank_equal_weight",
        ),
    }

    weights: dict[str, pd.DataFrame] = {}
    exposures: dict[str, pd.DataFrame] = {}
    eligibilities: dict[str, pd.DataFrame] = {}
    rank_diagnostic_rows = []
    performance_rows = []
    net_navs: dict[str, pd.Series] = {}
    prices = e5._prices(data).reindex(columns=columns)
    ew_nav = _ew_navs()[WINDOW]
    costs = COST_BPS / 10000.0
    for leg, (scores, eligibility, is_primary, construction) in leg_inputs.items():
        leg_weights, exposure, rank_diagnostics = _canonical_long_short_weights(
            scores, prices, eligibility
        )
        weights[leg] = leg_weights
        exposures[leg] = exposure
        eligibilities[leg] = eligibility
        rank_diagnostic_rows.append(rank_diagnostics.assign(leg=leg))
        net, gross = _backtest(
            prices, leg_weights, costs, f"u1_bics_classic_{leg}_long_short"
        )
        net_navs[leg] = net.get_portfolio_nav().rename(leg)
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
                "signal_variant": SIGNAL_VARIANT,
                **grid_ls._performance_payload(net, gross, ew_nav),
                "runner": RUNNER,
            }
        )

    performance = pd.DataFrame(performance_rows)
    rank_diagnostics = pd.concat(rank_diagnostic_rows, ignore_index=True)
    missing_bics = base._missing_bics_table(data, bics, all_eligibility)
    coverage = base._coverage_table(
        all_eligibility, primary_eligibility, raw_scores, sector_groups
    )
    acceptance = _acceptance_rows(
        signal_diagnostics=signal_diagnostics,
        performance=performance,
        weights=weights,
        eligibilities=eligibilities,
        exposures=exposures,
        rank_diagnostics=rank_diagnostics,
        sector_membership_diagnostics=sector_membership_diagnostics,
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
    design = pd.DataFrame(
        [
            {
                **FROZEN_SPEC,
                "primary_legs": "|".join(PRIMARY_LEGS),
                "primary_universe": "point_in_time_eligible_and_bics_classified",
                "score_construction": (
                    "public OP classic global/rolling-cluster APIs"
                ),
                "portfolio_rank_rule": (
                    "one global canonical OP top/bottom quantile rank per score panel"
                ),
                "cluster_role": "score standardisation only",
                "sector_role": (
                    "rolling-cluster score standardisation on point-in-time "
                    "eligible index members"
                ),
                "asset_weight_rule": "equal within each selected side",
                "ew_role": "market reference for beta/alpha only",
                "runner": RUNNER,
            }
        ]
    )
    output = {
        "navs": pd.concat(net_navs, axis=1).rename_axis("date").reset_index(),
        "weights": pd.concat(
            [
                frame.reset_index(names="date").assign(leg=leg)
                for leg, frame in weights.items()
            ],
            ignore_index=True,
        ),
        "performance": performance,
        "comparison": base._performance_comparison(performance),
        "coverage_per_date": coverage,
        "missing_bics_assets": missing_bics,
        "sector_membership_diagnostics": sector_membership_diagnostics,
        "rank_diagnostics": rank_diagnostics,
        "exposure_diagnostics": exposure_table,
        "signal_regression": signal_regression,
        "signal_diagnostics": signal_diagnostics,
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
    """Run twice and require byte-identical classic-signal artifacts."""
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
    """Execute, replay, and print canonical classic U1 results."""
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
        f"U1 classic BICS sector comparison: PASS "
        f"({len(replay)}/{len(replay)} deterministic)",
        flush=True,
    )


if __name__ == "__main__":
    main()
