"""Explore the exact ROSAA futures momentum signal specification grid.

The grid varies the monthly production primitive over short/reversal spans
``None, 1, 2, 3``, volatility spans ``13, 26, 52``, and mean adjustments
``NONE, EWMA``.  Baseline and M1-star cluster legs additionally enumerate the
production small-cluster fallback ``5, 7, 10``.  Every treatment is compared
with the same-signal global-within-sleeve rank.

Portfolio mechanics remain frozen: +1/-1 long-short exposure, 30% Equity,
30% Fixed Income, 30% Commodities, and 10% FX on each side, q=20% primary and
q=25% robustness, 10 bp one-way costs, one W-WED implementation lag, and the
corrected U1 calendar window.  CUA1 Comdty is owner-excluded.
"""
from __future__ import annotations

import hashlib
import os
import time
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import qis
from optimalportfolios.alphas.signals.momentum import compute_momentum_alpha
from optimalportfolios.alphas.signals.utils import score_within_clusters

import papers.cluster_lineage_2026.replication.run_backtests as e5
import papers.cluster_lineage_2026.replication.run_futures_asset_class_long_short as asset
import papers.cluster_lineage_2026.replication.run_futures_cluster_30303010_10bp as legacy
import papers.cluster_lineage_2026.replication.run_futures_prod_cluster_30303010_10bp as base
import papers.cluster_lineage_2026.replication.run_futures_sleeve_grid as equal
import papers.cluster_lineage_2026.replication.run_futures_weight_30303010 as construction
import papers.cluster_lineage_2026.replication.run_futures_weight_30303010_u1_window as matched
import papers.cluster_lineage_2026.replication.run_u1_covar_grid_long_short_prod as u1_prod


SHORT_SPANS = (None, 1, 2, 3)
VOL_SPANS = (13, 26, 52)
MEAN_ADJ_TYPES = ("NONE", "EWMA")
CLUSTER_FALLBACKS = (5, 7, 10)
QUANTILES = tuple(equal.QUANTILES)
TARGET = dict(construction.TARGET)
METHODS = tuple(legacy.METHODS)
COST_BPS = 10.0
SIGNAL_FREQUENCY = "ME"
MOMENTUM_LONG_SPAN = 12
TOLERANCE = 1e-12
SOURCE_RECONSTRUCTION_TOLERANCE = 1e-15
MAX_WORKERS = int(os.environ.get("CLUSTER_LINEAGE_WORKERS", "4"))
RUNNER = (
    "papers/cluster_lineage_2026/replication/"
    "run_futures_prod_signal_grid_30303010_10bp.py"
)
COMPARISON_METRICS = tuple(equal.COMPARISON_METRICS)
FROZEN_LEADER = {
    "method": "sleeve_cluster_M1_star",
    "q": 0.25,
    "short_span_label": "None",
    "vol_span": 13,
    "mean_adj_type": "EWMA",
    "delta_net_return_annualized": -0.00224242264761521,
}
_WORKER_CONTEXT: dict[str, object] | None = None


@dataclass(frozen=True)
class SignalSpec:
    """Identify one requested monthly ROSAA momentum configuration."""

    short_span: int | None
    vol_span: int
    mean_adj_type: str

    @property
    def short_span_label(self) -> str:
        """Return a CSV-stable label for the optional reversal span."""
        return "None" if self.short_span is None else str(self.short_span)

    @property
    def signal_id(self) -> str:
        """Return a deterministic identifier for this signal specification."""
        short = "none" if self.short_span is None else str(self.short_span)
        return f"short_{short}__vol_{self.vol_span}__mean_{self.mean_adj_type}"


SIGNAL_SPECS = tuple(
    SignalSpec(short_span, vol_span, mean_adj_type)
    for short_span in SHORT_SPANS
    for vol_span in VOL_SPANS
    for mean_adj_type in MEAN_ADJ_TYPES
)


def _root() -> Path:
    """Return and create the external production-signal-grid output directory."""
    return e5.get_output_path(
        "e5b", "futures_prod_signal_grid_30_30_30_10_10bp_u1_window", create=True
    )


def _finite_max(frame: pd.DataFrame) -> float:
    """Return the largest finite value in a numerical frame, or zero if empty."""
    values = frame.to_numpy()
    finite = values[np.isfinite(values)]
    return float(finite.max()) if finite.size else 0.0


def _build_context() -> dict[str, object]:
    """Load invariant futures, benchmark, grouping, and payoff inputs."""
    data = e5.load_universe(equal.UNIVERSE)
    dates = matched._window_dates(
        e5.load_cached(equal.UNIVERSE, e5.SmootherName.BASELINE).dates
    )
    eligibility = e5._investable_eligibility(data, dates)
    columns = eligibility.columns
    daily = base._read_daily(columns)
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
    roundtrip = qis.to_returns(
        signal_prices,
        freq=SIGNAL_FREQUENCY,
        is_log_returns=True,
    ).reindex_like(monthly_log_returns)
    monthly_error = _finite_max(roundtrip.subtract(monthly_log_returns).abs())

    performance_prices = matched._prices_with_context(
        e5._prices(data).reindex(columns=columns)
    )
    sleeves = equal._broad_sleeves(data.taxonomy, columns)
    sleeve_panel = equal._sleeve_panel(dates, sleeves)
    groups_by_method = legacy._group_panels(dates, columns, sleeve_panel)
    accepted_navs = pd.read_csv(
        equal._accepted_root() / "navs.csv",
        parse_dates=["date"],
        float_precision="round_trip",
    ).set_index("date")
    ew_nav = matched._bounded_panel(accepted_navs["EW_all"])
    if not isinstance(ew_nav, pd.Series):
        raise AssertionError("bounded EW reference is not a Series")
    excluded = eligibility.columns.intersection(
        sorted(e5.FUTURES_INVESTABILITY_EXCLUSIONS)
    )
    excluded_eligible = int(eligibility.loc[:, excluded].sum().sum())
    source_passed = (
        weekly_error <= SOURCE_RECONSTRUCTION_TOLERANCE
        and weekly_nan_match
        and monthly_error <= TOLERANCE
        and excluded_eligible == 0
    )
    source_preflight = {
        "check": "common_signal_source_preflight",
        "daily_to_wwed_max_abs_error": weekly_error,
        "daily_to_wwed_nan_pattern_match": weekly_nan_match,
        "monthly_return_roundtrip_max_abs_error": monthly_error,
        "owner_excluded_eligible_observations": excluded_eligible,
        "source_reconstruction_tolerance": SOURCE_RECONSTRUCTION_TOLERANCE,
        "general_tolerance": TOLERANCE,
        "status": "PASS" if source_passed else "FAIL",
    }
    if not source_passed:
        raise AssertionError(source_preflight)
    return {
        "data": data,
        "dates": dates,
        "eligibility": eligibility,
        "signal_prices": signal_prices,
        "benchmark": benchmark,
        "performance_prices": performance_prices,
        "sleeve_panel": sleeve_panel,
        "groups_by_method": groups_by_method,
        "ew_nav": ew_nav,
        "source_preflight": source_preflight,
    }


def _initialize_worker() -> None:
    """Load immutable experiment inputs once inside each process."""
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = _build_context()


def _context() -> dict[str, object]:
    """Return the initialized worker context or build a local fallback."""
    global _WORKER_CONTEXT
    if _WORKER_CONTEXT is None:
        _WORKER_CONTEXT = _build_context()
    return _WORKER_CONTEXT


def _mean_adj_enum(label: str) -> qis.MeanAdjType:
    """Resolve the requested CSV label to the qis production enum."""
    if label == "NONE":
        return qis.MeanAdjType.NONE
    if label == "EWMA":
        return qis.MeanAdjType.EWMA
    raise KeyError(f"unknown mean adjustment: {label}")


def _signal_for_spec(
    spec: SignalSpec,
    context: Mapping[str, object],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, dict]:
    """Compute and validate one global/raw monthly production signal pair."""
    signal_prices = context["signal_prices"]
    benchmark = context["benchmark"]
    dates = context["dates"]
    eligibility = context["eligibility"]
    if not isinstance(signal_prices, pd.DataFrame):
        raise AssertionError("signal prices are not a DataFrame")
    if not isinstance(benchmark, pd.Series):
        raise AssertionError("signal benchmark is not a Series")
    if not isinstance(dates, pd.DatetimeIndex):
        raise AssertionError("decision dates are not a DatetimeIndex")
    if not isinstance(eligibility, pd.DataFrame):
        raise AssertionError("eligibility is not a DataFrame")
    global_source, raw_source = compute_momentum_alpha(
        prices=signal_prices,
        benchmark_price=benchmark,
        returns_freq=SIGNAL_FREQUENCY,
        group_data=None,
        long_span=MOMENTUM_LONG_SPAN,
        short_span=spec.short_span,
        vol_span=spec.vol_span,
        mean_adj_type=_mean_adj_enum(spec.mean_adj_type),
    )
    global_decision, timestamps = u1_prod._asof_panel(global_source, dates)
    raw_decision, raw_timestamps = u1_prod._asof_panel(raw_source, dates)
    timestamps_match = bool(timestamps.equals(raw_timestamps))
    global_decision = global_decision.reindex(
        columns=eligibility.columns
    ).where(eligibility)
    raw_decision = raw_decision.reindex(columns=eligibility.columns).where(
        eligibility
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
        and excluded_scores == 0
    )
    diagnostic = {
        "signal_id": spec.signal_id,
        "short_span": np.nan if spec.short_span is None else spec.short_span,
        "short_span_label": spec.short_span_label,
        "vol_span": spec.vol_span,
        "mean_adj_type": spec.mean_adj_type,
        "global_raw_timestamp_match": timestamps_match,
        "max_signal_lookahead_days": int(lookahead_days.max()),
        "valid_assets_min": int(valid_counts.min()),
        "valid_assets_median": float(valid_counts.median()),
        "valid_assets_max": int(valid_counts.max()),
        "owner_excluded_valid_scores": excluded_scores,
        "status": "PASS" if passed else "FAIL",
    }
    if not passed:
        raise AssertionError(diagnostic)
    return global_decision, raw_source, timestamps, diagnostic


def _attach_spec(row: dict, spec: SignalSpec, fallback: float) -> dict:
    """Attach the complete signal-grid provenance to one output row."""
    row.update(
        {
            "signal_id": spec.signal_id,
            "signal_frequency": SIGNAL_FREQUENCY,
            "momentum_long_span": MOMENTUM_LONG_SPAN,
            "short_span": np.nan if spec.short_span is None else spec.short_span,
            "short_span_label": spec.short_span_label,
            "vol_span": spec.vol_span,
            "mean_adj_type": spec.mean_adj_type,
            "cluster_fallback": fallback,
            "cost_bps_one_way": COST_BPS,
            "runner": RUNNER,
        }
    )
    return row


def _backtest_weights(
    *,
    spec: SignalSpec,
    method: str,
    fallback: float,
    q: float,
    weights: pd.DataFrame,
    diagnostics: Mapping[str, float],
    context: Mapping[str, object],
) -> tuple[dict, dict]:
    """Backtest one unique weight panel and return performance and acceptance."""
    engine_method = f"{method}__{spec.signal_id}__fallback_{fallback}"
    records = matched._run_leg(
        strategy="long_short",
        method=engine_method,
        q=q,
        prices=context["performance_prices"],
        weights=weights,
        diagnostics=diagnostics,
        sleeve_panel=context["sleeve_panel"],
        ew_nav=context["ew_nav"],
        costs=COST_BPS / 10000.0,
        target=TARGET,
    )
    performance, acceptance, _, _ = records
    performance["method"] = method
    acceptance["method"] = method
    _attach_spec(performance, spec, fallback)
    _attach_spec(acceptance, spec, fallback)
    excluded = weights.columns.intersection(
        sorted(e5.FUTURES_INVESTABILITY_EXCLUSIONS)
    )
    exclusion_error = float(weights.loc[:, excluded].abs().to_numpy().max())
    acceptance["max_owner_excluded_weight_abs"] = exclusion_error
    acceptance["status"] = (
        "PASS"
        if acceptance["status"] == "PASS" and exclusion_error <= TOLERANCE
        else "FAIL"
    )
    return performance, acceptance


def _clone_acceptance(
    base_acceptance: Mapping[str, object],
    diagnostics: Mapping[str, float],
    spec: SignalSpec,
    fallback: int,
) -> dict:
    """Clone acceptance for an exactly invariant fallback-weight panel."""
    row = dict(base_acceptance)
    row.update(diagnostics)
    _attach_spec(row, spec, float(fallback))
    return row


def _reconstruction_row(
    *,
    spec: SignalSpec,
    method: str,
    fallback: float,
    q: float,
    scores: pd.DataFrame,
    groups: pd.DataFrame,
    weights: pd.DataFrame,
    context: Mapping[str, object],
) -> dict:
    """Independently reconstruct one combined book from four standalone sleeves."""
    reconstructed = sum(
        asset._standalone_weights(
            scores,
            context["eligibility"],
            context["sleeve_panel"],
            groups,
            sleeve,
            q,
        )[0].mul(TARGET[sleeve])
        for sleeve in equal.SLEEVES
    )
    error = float(reconstructed.subtract(weights).abs().to_numpy().max())
    row = {
        "method": method,
        "q": q,
        "max_weight_abs_error": error,
        "tolerance": TOLERANCE,
        "status": "PASS" if error <= TOLERANCE else "FAIL",
    }
    return _attach_spec(row, spec, fallback)


def _run_spec(spec: SignalSpec) -> dict[str, list[dict] | dict]:
    """Run one signal specification over both q values and all cluster fallbacks."""
    started = time.perf_counter()
    context = _context()
    eligibility = context["eligibility"]
    sleeve_panel = context["sleeve_panel"]
    groups_by_method = context["groups_by_method"]
    if not isinstance(eligibility, pd.DataFrame):
        raise AssertionError("worker eligibility is not a DataFrame")
    if not isinstance(sleeve_panel, pd.DataFrame):
        raise AssertionError("worker sleeve panel is not a DataFrame")
    if not isinstance(groups_by_method, dict):
        raise AssertionError("worker group panels are not a dictionary")
    global_scores, raw_source, timestamps, signal_diagnostic = _signal_for_spec(
        spec, context
    )

    performance_rows = []
    acceptance_rows = []
    fallback_rows = []
    reconstruction_rows = []
    global_groups = groups_by_method["sleeve_global"]
    for q in QUANTILES:
        weights, diagnostics = construction._build_constrained_weights(
            "long_short",
            global_scores,
            eligibility,
            sleeve_panel,
            global_groups,
            q,
        )
        performance, acceptance = _backtest_weights(
            spec=spec,
            method="sleeve_global",
            fallback=np.nan,
            q=q,
            weights=weights,
            diagnostics=diagnostics,
            context=context,
        )
        performance_rows.append(performance)
        acceptance_rows.append(acceptance)
        reconstruction_rows.append(
            _reconstruction_row(
                spec=spec,
                method="sleeve_global",
                fallback=np.nan,
                q=q,
                scores=global_scores,
                groups=global_groups,
                weights=weights,
                context=context,
            )
        )

    for method in METHODS[1:]:
        groups = groups_by_method[method]
        scores_by_fallback = {}
        cluster_timestamp_rows = []
        for fallback in CLUSTER_FALLBACKS:
            source = score_within_clusters(
                raw_signal=raw_source,
                rolling_clusters=u1_prod._panel_dict(groups),
                min_cluster_size=fallback,
            )
            decision, cluster_timestamps = u1_prod._asof_panel(
                source, context["dates"]
            )
            timestamp_match = bool(cluster_timestamps.equals(timestamps))
            decision = decision.reindex(columns=eligibility.columns).where(eligibility)
            scores_by_fallback[fallback] = decision
            cluster_timestamp_rows.append(timestamp_match)
        if not all(cluster_timestamp_rows):
            raise AssertionError(f"cluster signal timestamps differ for {spec.signal_id}")

        for q in QUANTILES:
            weights_by_fallback = {}
            diagnostics_by_fallback = {}
            for fallback in CLUSTER_FALLBACKS:
                weights, diagnostics = construction._build_constrained_weights(
                    "long_short",
                    scores_by_fallback[fallback],
                    eligibility,
                    sleeve_panel,
                    groups,
                    q,
                )
                weights_by_fallback[fallback] = weights
                diagnostics_by_fallback[fallback] = diagnostics
                reconstruction_rows.append(
                    _reconstruction_row(
                        spec=spec,
                        method=method,
                        fallback=float(fallback),
                        q=q,
                        scores=scores_by_fallback[fallback],
                        groups=groups,
                        weights=weights,
                        context=context,
                    )
                )

            base_fallback = CLUSTER_FALLBACKS[0]
            base_weights = weights_by_fallback[base_fallback]
            base_performance, base_acceptance = _backtest_weights(
                spec=spec,
                method=method,
                fallback=float(base_fallback),
                q=q,
                weights=base_weights,
                diagnostics=diagnostics_by_fallback[base_fallback],
                context=context,
            )
            performance_rows.append(base_performance)
            acceptance_rows.append(base_acceptance)
            base_scores = scores_by_fallback[base_fallback]
            for fallback in CLUSTER_FALLBACKS[1:]:
                weight_error = float(
                    weights_by_fallback[fallback]
                    .subtract(base_weights)
                    .abs()
                    .to_numpy()
                    .max()
                )
                score_error = _finite_max(
                    scores_by_fallback[fallback].subtract(base_scores).abs()
                )
                invariant = weight_error <= TOLERANCE
                fallback_rows.append(
                    {
                        "signal_id": spec.signal_id,
                        "short_span": (
                            np.nan if spec.short_span is None else spec.short_span
                        ),
                        "short_span_label": spec.short_span_label,
                        "vol_span": spec.vol_span,
                        "mean_adj_type": spec.mean_adj_type,
                        "method": method,
                        "q": q,
                        "base_fallback": base_fallback,
                        "compared_fallback": fallback,
                        "max_score_abs_difference": score_error,
                        "max_weight_abs_difference": weight_error,
                        "tolerance": TOLERANCE,
                        "status": "PASS" if invariant else "FAIL",
                    }
                )
                if invariant:
                    performance = dict(base_performance)
                    performance["cluster_fallback"] = float(fallback)
                    acceptance = _clone_acceptance(
                        base_acceptance,
                        diagnostics_by_fallback[fallback],
                        spec,
                        fallback,
                    )
                else:
                    performance, acceptance = _backtest_weights(
                        spec=spec,
                        method=method,
                        fallback=float(fallback),
                        q=q,
                        weights=weights_by_fallback[fallback],
                        diagnostics=diagnostics_by_fallback[fallback],
                        context=context,
                    )
                performance_rows.append(performance)
                acceptance_rows.append(acceptance)

    return {
        "performance": performance_rows,
        "acceptance": acceptance_rows,
        "fallback": fallback_rows,
        "reconstruction": reconstruction_rows,
        "signal": signal_diagnostic,
        "runtime": {
            "signal_id": spec.signal_id,
            "runtime_seconds": time.perf_counter() - started,
        },
    }


def _comparison(performance: pd.DataFrame) -> pd.DataFrame:
    """Compare every cluster row with its same-signal and same-q global control."""
    global_rows = performance.loc[
        performance["method"].eq("sleeve_global")
    ].set_index(["signal_id", "q"])
    rows = []
    clusters = performance.loc[
        performance["method"].str.startswith("sleeve_cluster_")
    ]
    for _, cluster in clusters.iterrows():
        reference = global_rows.loc[(cluster["signal_id"], cluster["q"])]
        row = {
            "signal_id": cluster["signal_id"],
            "short_span": cluster["short_span"],
            "short_span_label": cluster["short_span_label"],
            "vol_span": cluster["vol_span"],
            "mean_adj_type": cluster["mean_adj_type"],
            "cluster_fallback": cluster["cluster_fallback"],
            "q": cluster["q"],
            "method": cluster["method"],
            "benchmark_method": "sleeve_global",
        }
        for metric in COMPARISON_METRICS:
            row[f"cluster_{metric}"] = cluster[metric]
            row[f"global_{metric}"] = reference[metric]
            row[f"delta_{metric}"] = cluster[metric] - reference[metric]
        row["beats_global_net_return"] = row[
            "delta_net_return_annualized"
        ] > 0.0
        row["beats_global_sharpe"] = row["delta_sharpe_rf0"] > 0.0
        row["lower_volatility_than_global"] = row[
            "delta_volatility_annualized"
        ] < 0.0
        row["mean_variance_dominates_global"] = (
            row["beats_global_net_return"]
            and row["lower_volatility_than_global"]
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["q", "method", "signal_id", "cluster_fallback"]
    ).reset_index(drop=True)


def _grid_summary(unique: pd.DataFrame) -> pd.DataFrame:
    """Summarize global-win breadth and the best row per q and cluster method."""
    rows = []
    for (q, method), panel in unique.groupby(["q", "method"], sort=True):
        leader = panel.sort_values(
            ["delta_net_return_annualized", "delta_sharpe_rf0"],
            ascending=[False, False],
        ).iloc[0]
        rows.append(
            {
                "q": q,
                "method": method,
                "unique_signal_specs": len(panel),
                "net_return_wins": int(panel["beats_global_net_return"].sum()),
                "sharpe_wins": int(panel["beats_global_sharpe"].sum()),
                "mean_variance_wins": int(
                    panel["mean_variance_dominates_global"].sum()
                ),
                "best_signal_id": leader["signal_id"],
                "best_short_span_label": leader["short_span_label"],
                "best_vol_span": int(leader["vol_span"]),
                "best_mean_adj_type": leader["mean_adj_type"],
                "best_cluster_net_return_annualized": leader[
                    "cluster_net_return_annualized"
                ],
                "best_global_net_return_annualized": leader[
                    "global_net_return_annualized"
                ],
                "best_delta_net_return_annualized": leader[
                    "delta_net_return_annualized"
                ],
                "best_cluster_sharpe_rf0": leader["cluster_sharpe_rf0"],
                "best_global_sharpe_rf0": leader["global_sharpe_rf0"],
                "best_delta_sharpe_rf0": leader["delta_sharpe_rf0"],
            }
        )
    return pd.DataFrame(rows)


def _grid_leaders(
    unique: pd.DataFrame, performance: pd.DataFrame
) -> pd.DataFrame:
    """Return the overall cluster-delta, cluster-level, and global leaders."""
    best_delta = unique.sort_values(
        ["delta_net_return_annualized", "delta_sharpe_rf0"],
        ascending=[False, False],
    ).iloc[0]
    best_cluster = unique.sort_values(
        ["cluster_net_return_annualized", "cluster_sharpe_rf0"],
        ascending=[False, False],
    ).iloc[0]
    globals_frame = performance.loc[
        performance["method"].eq("sleeve_global")
    ].sort_values(
        ["net_return_annualized", "sharpe_rf0"],
        ascending=[False, False],
    )
    best_global = globals_frame.iloc[0]

    def cluster_row(label: str, row: pd.Series) -> dict:
        """Format one cluster leader row."""
        return {
            "leader_type": label,
            "method": row["method"],
            "q": row["q"],
            "signal_id": row["signal_id"],
            "short_span_label": row["short_span_label"],
            "vol_span": row["vol_span"],
            "mean_adj_type": row["mean_adj_type"],
            "cluster_fallback": row["cluster_fallback"],
            "cluster_net_return_annualized": row[
                "cluster_net_return_annualized"
            ],
            "global_net_return_annualized": row[
                "global_net_return_annualized"
            ],
            "delta_net_return_annualized": row[
                "delta_net_return_annualized"
            ],
            "cluster_sharpe_rf0": row["cluster_sharpe_rf0"],
            "global_sharpe_rf0": row["global_sharpe_rf0"],
            "delta_sharpe_rf0": row["delta_sharpe_rf0"],
        }

    global_row = {
        "leader_type": "best_global_net_return",
        "method": best_global["method"],
        "q": best_global["q"],
        "signal_id": best_global["signal_id"],
        "short_span_label": best_global["short_span_label"],
        "vol_span": best_global["vol_span"],
        "mean_adj_type": best_global["mean_adj_type"],
        "cluster_fallback": np.nan,
        "cluster_net_return_annualized": np.nan,
        "global_net_return_annualized": best_global["net_return_annualized"],
        "delta_net_return_annualized": np.nan,
        "cluster_sharpe_rf0": np.nan,
        "global_sharpe_rf0": best_global["sharpe_rf0"],
        "delta_sharpe_rf0": np.nan,
    }
    return pd.DataFrame(
        [
            cluster_row("best_cluster_delta_net_return", best_delta),
            cluster_row("best_cluster_absolute_net_return", best_cluster),
            global_row,
        ]
    )


def _base_spec_regression(performance: pd.DataFrame) -> pd.DataFrame:
    """Match the grid's base cell to the completed exact-production run."""
    current = performance.loc[
        performance["signal_id"].eq("short_none__vol_13__mean_NONE")
        & (
            performance["method"].eq("sleeve_global")
            | performance["cluster_fallback"].eq(5.0)
        )
    ].set_index(["q", "method"])
    reference = pd.read_csv(
        base._root() / "performance.csv", float_precision="round_trip"
    ).set_index(["q", "method"])
    rows = []
    for key in reference.index:
        errors = [
            abs(float(current.loc[key, metric] - reference.loc[key, metric]))
            for metric in COMPARISON_METRICS
        ]
        maximum = max(errors)
        rows.append(
            {
                "q": key[0],
                "method": key[1],
                "compared_metrics": len(COMPARISON_METRICS),
                "max_abs_error": maximum,
                "tolerance": TOLERANCE,
                "status": "PASS" if maximum <= TOLERANCE else "FAIL",
            }
        )
    return pd.DataFrame(rows).sort_values(["q", "method"]).reset_index(drop=True)


def _design() -> pd.DataFrame:
    """Return the complete requested grid and inherited portfolio specification."""
    return pd.DataFrame(
        [
            {
                "signal_frequency": SIGNAL_FREQUENCY,
                "momentum_long_span": MOMENTUM_LONG_SPAN,
                "short_spans": "None|1|2|3",
                "vol_spans": "13|26|52",
                "mean_adj_types": "NONE|EWMA",
                "cluster_fallbacks": "5|7|10",
                "signal_specs": len(SIGNAL_SPECS),
                "q_values": "0.20|0.25",
                "methods": "|".join(METHODS),
                "performance_rows": 336,
                "cost_bps_one_way": COST_BPS,
                "workers": MAX_WORKERS,
                "equity_budget_per_side": TARGET["Equity"],
                "fixed_income_budget_per_side": TARGET["Fixed Income"],
                "commodities_budget_per_side": TARGET["Commodities"],
                "fx_budget_per_side": TARGET["FX"],
                "analysis_window": matched.WINDOW,
                "owner_exclusions": "|".join(
                    sorted(e5.FUTURES_INVESTABILITY_EXCLUSIONS)
                ),
                "runner": RUNNER,
            }
        ]
    )


def run() -> Mapping[str, pd.DataFrame]:
    """Execute the 24-signal grid with four worker processes."""
    started = time.perf_counter()
    construction._validate_target()
    parent_context = _build_context()
    results = []
    with ProcessPoolExecutor(
        max_workers=MAX_WORKERS,
        initializer=_initialize_worker,
    ) as executor:
        futures = {executor.submit(_run_spec, spec): spec for spec in SIGNAL_SPECS}
        for future in as_completed(futures):
            spec = futures[future]
            results.append(future.result())
            print(f"production signal grid {spec.signal_id}: complete", flush=True)

    performance = pd.DataFrame(
        row for result in results for row in result["performance"]
    ).sort_values(
        ["q", "method", "signal_id", "cluster_fallback"],
        na_position="first",
    ).reset_index(drop=True)
    acceptance = pd.DataFrame(
        row for result in results for row in result["acceptance"]
    ).sort_values(
        ["q", "method", "signal_id", "cluster_fallback"],
        na_position="first",
    ).reset_index(drop=True)
    fallback = pd.DataFrame(
        row for result in results for row in result["fallback"]
    ).sort_values(
        ["q", "method", "signal_id", "compared_fallback"]
    ).reset_index(drop=True)
    reconstruction = pd.DataFrame(
        row for result in results for row in result["reconstruction"]
    ).sort_values(
        ["q", "method", "signal_id", "cluster_fallback"],
        na_position="first",
    ).reset_index(drop=True)
    signal_diagnostics = pd.DataFrame(
        result["signal"] for result in results
    ).sort_values("signal_id").reset_index(drop=True)
    runtime = pd.DataFrame(result["runtime"] for result in results).sort_values(
        "signal_id"
    )
    runtime["total_run_seconds"] = time.perf_counter() - started

    if len(performance) != 336:
        raise AssertionError(f"expected 336 performance rows, got {len(performance)}")
    if not acceptance["status"].eq("PASS").all():
        raise AssertionError(acceptance.loc[~acceptance["status"].eq("PASS")])
    if not fallback["status"].eq("PASS").all():
        raise AssertionError(fallback.loc[~fallback["status"].eq("PASS")])
    if not reconstruction["status"].eq("PASS").all():
        raise AssertionError(reconstruction.loc[~reconstruction["status"].eq("PASS")])
    if not signal_diagnostics["status"].eq("PASS").all():
        raise AssertionError(
            signal_diagnostics.loc[~signal_diagnostics["status"].eq("PASS")]
        )
    comparison = _comparison(performance)
    unique = comparison.loc[comparison["cluster_fallback"].eq(5.0)].reset_index(
        drop=True
    )
    summary = _grid_summary(unique)
    leaders = _grid_leaders(unique, performance)
    base_regression = _base_spec_regression(performance)
    if not base_regression["status"].eq("PASS").all():
        raise AssertionError(base_regression)
    source_preflight = pd.DataFrame([parent_context["source_preflight"]])
    outputs = {
        "design": _design(),
        "source_preflight": source_preflight,
        "signal_diagnostics": signal_diagnostics,
        "performance": performance,
        "comparison_vs_global": comparison,
        "comparison_unique_portfolios": unique,
        "grid_summary": summary,
        "grid_leaders": leaders,
        "fallback_invariance": fallback,
        "acceptance": acceptance,
        "standalone_weight_reconstruction": reconstruction,
        "base_spec_regression": base_regression,
    }
    for name, frame in outputs.items():
        e5._write(frame, _root() / f"{name}.csv")
    e5._write(runtime, _root() / "runtime.csv")
    return outputs


def _hash_outputs() -> dict[str, str]:
    """Hash deterministic outputs while excluding timing and replay records."""
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(_root().glob("*.csv"))
        if path.name not in {"runtime.csv", "determinism.csv"}
    }


def verify_determinism() -> pd.DataFrame:
    """Replay the complete parallel grid and require identical numerical output."""
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
    """Run, replay, and print the production-signal grid leaders."""
    replay = verify_determinism()
    leaders = pd.read_csv(
        _root() / "grid_leaders.csv", float_precision="round_trip"
    )
    print(leaders.to_string(index=False))
    print(
        f"Futures ROSAA production signal grid: PASS "
        f"({len(replay)}/{len(replay)} deterministic)",
        flush=True,
    )


if __name__ == "__main__":
    main()
