"""Run matched futures cluster long-short books at 30/30/30/10 and 10 bp.

Each signed side allocates 30% to Equity, 30% to Fixed Income, 30% to Commodities,
and 10% to FX.  Within a sleeve, baseline or M1-star correlation clusters receive
equal budgets across the groups that have at least one eligible asset with a valid
paper raw-weekly momentum score; selected contracts then share their group budget equally.

The global-within-sleeve book is rebuilt as the matched payoff control.  All methods
use q=20% primary and q=25% robustness, the corrected U1 calendar window, one W-WED
implementation lag, and 10 bp per one-way traded notional.  A matched 20 bp replay is
retained only as cost sensitivity.  CUA1 Comdty is owner-excluded throughout.
"""
from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

import papers.cluster_lineage_2026.replication.run_backtests as e5
import papers.cluster_lineage_2026.replication.run_futures_asset_class_long_short as asset
import papers.cluster_lineage_2026.replication.run_futures_global_30303010_10bp as global10
import papers.cluster_lineage_2026.replication.run_futures_sleeve_grid as equal
import papers.cluster_lineage_2026.replication.run_futures_weight_30303010 as construction
import papers.cluster_lineage_2026.replication.run_futures_weight_30303010_u1_window as matched


TARGET = dict(construction.TARGET)
COST_BPS = 10.0
REFERENCE_COST_BPS = 20.0
QUANTILES = tuple(equal.QUANTILES)
METHODS = (
    "sleeve_global",
    "sleeve_cluster_baseline",
    "sleeve_cluster_M1_star",
)
RUNNER = (
    "papers/cluster_lineage_2026/replication/"
    "run_futures_cluster_30303010_10bp.py"
)
TOLERANCE = 1e-12
COMPARISON_METRICS = tuple(equal.COMPARISON_METRICS)


def _root() -> Path:
    """Return and create the external 10 bp cluster output directory."""
    return e5.get_output_path(
        "e5b", "futures_cluster_30_30_30_10_10bp_u1_window", create=True
    )


def _group_panels(
    dates: pd.DatetimeIndex,
    columns: pd.Index,
    sleeve_panel: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Return the global control and both cached hierarchical cluster panels."""
    panels = {"sleeve_global": sleeve_panel}
    for config in equal.CONFIGS:
        clusters = e5._cluster_groups(equal.UNIVERSE, config).reindex(
            index=dates, columns=columns
        )
        panels[f"sleeve_cluster_{config.value}"] = equal._hierarchical_groups(
            clusters, sleeve_panel
        )
    if tuple(panels) != METHODS:
        raise AssertionError(f"unexpected method order: {tuple(panels)}")
    return panels


def _design(
    dates: pd.DatetimeIndex,
    eligibility: pd.DataFrame,
    sleeves: pd.Series,
) -> pd.DataFrame:
    """Return the frozen signal, cost, grouping, and availability specification."""
    class_counts = {
        sleeve: eligibility.loc[:, sleeves.eq(sleeve)].sum(axis=1)
        for sleeve in equal.SLEEVES
    }
    return pd.DataFrame(
        [
            {
                "universe": equal.UNIVERSE.value,
                "analysis_window": matched.WINDOW,
                "strategy": "long_short",
                "primary_cost_bps_one_way": COST_BPS,
                "reference_cost_bps_one_way": REFERENCE_COST_BPS,
                "q_values": "|".join(f"{q:.2f}" for q in QUANTILES),
                "decision_dates": len(dates),
                "decision_start": dates.min(),
                "decision_end": dates.max(),
                "signal": "48-week log-return sum excluding latest 4 weeks",
                "signal_sum_min_count": 1,
                "full_lookback_required": False,
                "implementation_lag_observations": 1,
                "construction": "group_equal within sleeve for cluster methods",
                "methods": "|".join(METHODS),
                "equity_budget_per_side": TARGET["Equity"],
                "fixed_income_budget_per_side": TARGET["Fixed Income"],
                "commodities_budget_per_side": TARGET["Commodities"],
                "fx_budget_per_side": TARGET["FX"],
                "eligible_futures_min": int(eligibility.sum(axis=1).min()),
                "eligible_futures_max": int(eligibility.sum(axis=1).max()),
                **{
                    f"{sleeve.lower().replace(' ', '_')}_eligible_min": int(
                        counts.min()
                    )
                    for sleeve, counts in class_counts.items()
                },
                **{
                    f"{sleeve.lower().replace(' ', '_')}_eligible_max": int(
                        counts.max()
                    )
                    for sleeve, counts in class_counts.items()
                },
                "owner_exclusions": "|".join(
                    sorted(e5.FUTURES_INVESTABILITY_EXCLUSIONS)
                ),
                "runner": RUNNER,
            }
        ]
    )


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
    """Backtest one fixed decision panel at one proportional cost rate."""
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
    performance["cost_bps_one_way"] = cost_bps
    performance["runner"] = RUNNER
    acceptance["cost_bps_one_way"] = cost_bps
    horizon["cost_bps_one_way"] = cost_bps
    return performance, acceptance, allocation, horizon


def _comparison(performance: pd.DataFrame) -> pd.DataFrame:
    """Return explicit cluster-minus-global payoff differences at each q."""
    global_rows = performance.loc[
        performance["method"].eq("sleeve_global")
    ].set_index("q")
    rows = []
    clusters = performance.loc[
        performance["method"].str.startswith("sleeve_cluster_")
    ]
    for _, cluster in clusters.iterrows():
        reference = global_rows.loc[cluster["q"]]
        row = {
            "strategy": "long_short",
            "q": cluster["q"],
            "cluster_method": cluster["method"],
            "benchmark_method": "sleeve_global",
        }
        for metric in COMPARISON_METRICS:
            row[f"cluster_{metric}"] = cluster[metric]
            row[f"global_{metric}"] = reference[metric]
            row[f"delta_vs_global_{metric}"] = cluster[metric] - reference[metric]
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["q", "cluster_method"]).reset_index(drop=True)


def _group_count_outputs(
    groups_by_method: Mapping[str, pd.DataFrame],
    scores: pd.DataFrame,
    eligibility: pd.DataFrame,
    sleeve_panel: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return date-level available-group counts and their method/sleeve summaries."""
    rows = []
    valid_scores = scores.notna() & eligibility
    for method, groups in groups_by_method.items():
        for sleeve in equal.SLEEVES:
            available = groups.where(valid_scores & sleeve_panel.eq(sleeve)).nunique(
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


def _global_control_regression(performance: pd.DataFrame) -> pd.DataFrame:
    """Compare the rebuilt global rows with the dedicated 10 bp global output."""
    reference = pd.read_csv(
        global10._root() / "performance.csv", float_precision="round_trip"
    ).set_index("q")
    current = performance.loc[
        performance["method"].eq("sleeve_global")
    ].set_index("q")
    rows = []
    for q in QUANTILES:
        errors = [abs(float(current.loc[q, metric] - reference.loc[q, metric]))
                  for metric in COMPARISON_METRICS]
        maximum = max(errors)
        rows.append(
            {
                "q": q,
                "compared_metrics": len(COMPARISON_METRICS),
                "max_abs_error": maximum,
                "tolerance": TOLERANCE,
                "status": "PASS" if maximum <= TOLERANCE else "FAIL",
            }
        )
    return pd.DataFrame(rows)


def run() -> Mapping[str, pd.DataFrame]:
    """Execute the global, baseline-cluster, and M1-star-cluster matched books."""
    started = time.perf_counter()
    construction._validate_target()
    data = e5.load_universe(equal.UNIVERSE)
    dates = matched._window_dates(
        e5.load_cached(equal.UNIVERSE, e5.SmootherName.BASELINE).dates
    )
    eligibility = e5._investable_eligibility(data, dates)
    columns = eligibility.columns
    scores = e5._raw_momentum_scores(data, dates, vol_adjusted=False)
    scores = scores.reindex(columns=columns).where(eligibility)
    prices = matched._prices_with_context(e5._prices(data).reindex(columns=columns))
    sleeves = equal._broad_sleeves(data.taxonomy, columns)
    sleeve_panel = equal._sleeve_panel(dates, sleeves)
    groups_by_method = _group_panels(dates, columns, sleeve_panel)
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
    regression = _global_control_regression(performance)
    if not regression["status"].eq("PASS").all():
        raise AssertionError(regression)
    group_counts, group_summary = _group_count_outputs(
        groups_by_method, scores, eligibility, sleeve_panel
    )
    outputs = {
        "design": _design(dates, eligibility, sleeves),
        "performance": performance,
        "comparison": _comparison(performance),
        "cost_sensitivity": pd.DataFrame(sensitivity_rows).sort_values(
            ["q", "method"]
        ),
        "acceptance": acceptance,
        "allocation_diagnostics": pd.DataFrame(allocation_rows),
        "horizon_diagnostic": pd.DataFrame(horizon_rows),
        "standalone_weight_reconstruction": reconstruction,
        "available_group_counts_by_date": group_counts,
        "available_group_count_summary": group_summary,
        "global_control_regression": regression,
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
    """Replay all methods and costs and require byte-identical numerical output."""
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
    """Run, replay, and print the matched cluster-versus-global results."""
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
        f"Futures cluster 30/30/30/10 at 10 bp: PASS "
        f"({len(replay)}/{len(replay)} deterministic)",
        flush=True,
    )


if __name__ == "__main__":
    main()
