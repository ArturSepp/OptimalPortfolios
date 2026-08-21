"""Re-run the BlackRock fund search with a point-in-time USD 50m AUM filter.

Eligibility requires both the frozen 12-week return-history warmup and a strictly
greater-than USD 50 million trailing average of the latest 12 completed monthly
``FUND_TOTAL_ASSETS`` observations.  Missing AUM and incomplete 12-month histories are
ineligible.  The filter is applied before each partition is fitted and before both the
cluster and matched-global ranks are formed.  The accepted price-history-only artifacts
and caches are left untouched.
"""
from __future__ import annotations

import hashlib
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from factorlasso import compute_clusters_from_corr_matrix
from factorlasso.cluster_smoothing import _iter_correlation_inputs

import papers.cluster_lineage_2026.replication.run_backtests as e5
import papers.cluster_lineage_2026.replication.run_u2_blackrock_etf_grid as funds
import papers.cluster_lineage_2026.replication.run_u2_blackrock_long_short_search as search
import papers.cluster_lineage_2026.replication.run_u2_blackrock_sleeve_grid as sleeves


RUNNER = (
    "papers/cluster_lineage_2026/replication/"
    "run_u2_blackrock_aum_filter.py"
)
AUM_FILE = (
    funds.DATA_DIR / "blackrock_etf_aum_usd_millions_monthly.csv"
)
AUM_AUDIT_FILE = funds.DATA_DIR / "blackrock_etf_aum_audit.csv"
AUM_THRESHOLD_USD_MILLIONS = 50.0
AUM_ROLLING_MONTHS = 12
PARTITION_CACHE_VERSION = 1


def _root() -> Path:
    """Return the isolated AUM-filter experiment directory."""
    root = funds._root() / "aum50_filter_20260816"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _partition_root() -> Path:
    """Return the isolated AUM-filter partition-cache directory."""
    root = _root() / "partitions"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _read_aum() -> pd.DataFrame:
    """Read and validate completed-month Bloomberg AUM in USD millions."""
    aum = funds._read_frame(AUM_FILE)
    expected = funds._read_daily().columns
    if not aum.columns.equals(expected):
        raise AssertionError("AUM and return ticker order differ")
    if not aum.index.is_monotonic_increasing or aum.index.has_duplicates:
        raise AssertionError("AUM dates must be unique and increasing")
    if not aum.index.equals(aum.index.to_period("M").to_timestamp("M")):
        raise AssertionError("AUM panel is not labelled at completed month ends")
    if aum.index.max() > funds.FULL_END:
        raise AssertionError("AUM panel contains a month after the frozen study end")
    values = aum.to_numpy(dtype=float)
    if np.any(values[np.isfinite(values)] <= 0.0):
        raise AssertionError("AUM must be positive where observed")
    audit = pd.read_csv(AUM_AUDIT_FILE, index_col=0)
    if not audit.index.equals(pd.Index(expected, name="ticker")):
        raise AssertionError("AUM audit and return ticker order differ")
    if not audit["aum_currency"].eq("USD").all():
        raise AssertionError("all included AUM histories must be USD")
    return aum


def _rolling_aum(aum: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return the unfilled 12-completed-month arithmetic AUM average."""
    if aum is None:
        aum = _read_aum()
    return aum.rolling(
        window=AUM_ROLLING_MONTHS,
        min_periods=AUM_ROLLING_MONTHS,
    ).mean()


def _aum_for_dates(
    dates: pd.DatetimeIndex,
    rolling_aum: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Map each decision date to the latest already-completed rolling AUM value."""
    if rolling_aum is None:
        rolling_aum = _rolling_aum()
    dates = pd.DatetimeIndex(dates)
    result = rolling_aum.reindex(dates, method="ffill")
    result.index = dates.rename("date")
    return result


def _eligibility_for_dates(
    daily: pd.DataFrame,
    dates: pd.DatetimeIndex,
    rolling_aum: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Combine return warmup with strict trailing-average AUM eligibility."""
    history = funds._eligibility_for_dates(daily, dates)
    point_in_time_aum = _aum_for_dates(dates, rolling_aum).reindex(
        columns=history.columns
    )
    aum_eligible = point_in_time_aum.gt(AUM_THRESHOLD_USD_MILLIONS)
    return history & aum_eligible


def _input_fingerprint() -> str:
    """Return the joint source digest for AUM-filtered partitions."""
    payload = (
        f"returns={funds._input_fingerprint()}|"
        f"aum={funds._sha256(AUM_FILE)}|"
        f"threshold={AUM_THRESHOLD_USD_MILLIONS:.15g}|"
        f"months={AUM_ROLLING_MONTHS}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _partition_path(frequency: str, span: int) -> Path:
    """Return one AUM-filtered compact partition-cache path."""
    return _partition_root() / f"{funds._cell_id(frequency, span)}.pkl"


def _compute_partition_cell(
    frequency: str,
    span: int,
    *,
    force: bool = False,
) -> Mapping[str, object]:
    """Compute or load one partition fitted only on AUM-eligible funds."""
    path = _partition_path(frequency, span)
    fingerprint = _input_fingerprint()
    if path.exists() and not force:
        with path.open("rb") as stream:
            cached = pickle.load(stream)
        expected = {
            "version": PARTITION_CACHE_VERSION,
            "input_fingerprint": fingerprint,
            "frequency": frequency,
            "span": span,
        }
        if all(cached.get(key) == value for key, value in expected.items()):
            return {
                "frequency": frequency,
                "span": span,
                "cell_id": funds._cell_id(frequency, span),
                "cache_status": "hit",
                "partition_hash": funds._partition_hash(cached["panel"]),
                "dates": len(cached["panel"]),
                "runtime_seconds": 0.0,
            }

    started = time.perf_counter()
    daily = funds._read_daily()
    dates = funds._dates()
    eligibility = _eligibility_for_dates(daily, dates)
    returns = funds._native_returns(daily, frequency)
    model = funds._model(span, frequency)
    panel = pd.DataFrame(np.nan, index=dates, columns=daily.columns)
    diagnostic_rows = []
    iterator = _iter_correlation_inputs(returns, list(dates), model)
    for date, full_corr in iterator:
        assets = eligibility.columns[eligibility.loc[date].astype(bool)]
        corr = full_corr.reindex(index=assets, columns=assets)
        if len(assets) == 0:
            labels = pd.Series(dtype=float)
        elif len(assets) == 1:
            labels = pd.Series(1, index=assets)
        else:
            labels, _, _ = compute_clusters_from_corr_matrix(
                corr,
                cutoff_fraction=model.cutoff_fraction,
                linkage_method=model.linkage_method,
                distance_transform=model.distance_transform,
                n_clusters=model.n_clusters,
            )
        panel.loc[date, labels.index] = labels.to_numpy()
        diagnostic_rows.append(
            {
                "date": date,
                "members": len(labels),
                "clusters": int(labels.nunique()),
            }
        )

    payload = {
        "version": PARTITION_CACHE_VERSION,
        "input_fingerprint": fingerprint,
        "frequency": frequency,
        "span": span,
        "panel": panel,
        "diagnostics": pd.DataFrame(diagnostic_rows),
    }
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)
    return {
        "frequency": frequency,
        "span": span,
        "cell_id": funds._cell_id(frequency, span),
        "cache_status": "miss",
        "partition_hash": funds._partition_hash(panel),
        "dates": len(panel),
        "runtime_seconds": time.perf_counter() - started,
    }


def build_partitions(
    *,
    max_workers: int = 4,
    force: bool = False,
) -> pd.DataFrame:
    """Build all 28 AUM-filtered partition cells in parallel."""
    rows = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        pending = {
            executor.submit(
                _compute_partition_cell, frequency, span, force=force
            ): (frequency, span)
            for frequency, span in funds._cells()
        }
        for future in as_completed(pending):
            row = future.result()
            rows.append(row)
            print(
                f"AUM50 partition {row['frequency']}/{row['span']}: "
                f"{row['cache_status']} ({float(row['runtime_seconds']):.1f}s)",
                flush=True,
            )
    return pd.DataFrame(rows).sort_values(["frequency", "span"]).reset_index(drop=True)


def _load_partition(
    frequency: str,
    span: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and verify one AUM-filtered partition cache."""
    with _partition_path(frequency, span).open("rb") as stream:
        cached = pickle.load(stream)
    expected = {
        "version": PARTITION_CACHE_VERSION,
        "input_fingerprint": _input_fingerprint(),
        "frequency": frequency,
        "span": span,
    }
    if not all(cached.get(key) == value for key, value in expected.items()):
        raise AssertionError(f"invalid AUM50 partition cache for {frequency}/{span}")
    return cached["panel"], cached["diagnostics"]


def _context() -> dict[str, object]:
    """Build AUM-filtered search inputs and matched benchmark panels."""
    dates = funds._dates()
    headline_dates = dates[
        (dates >= funds.HEADLINE_START) & (dates <= funds.HEADLINE_END)
    ]
    daily = funds._read_daily()
    rolling_aum = _rolling_aum()
    eligibility_all = _eligibility_for_dates(daily, dates, rolling_aum)
    eligibility = eligibility_all.reindex(index=headline_dates).astype(bool)
    monthly_returns = funds._native_returns(daily, search.SIGNAL_FREQUENCY)
    monthly_eligibility = _eligibility_for_dates(
        daily, monthly_returns.index, rolling_aum
    )
    signal_panels, signal_diagnostics = search._signal_panels(
        daily,
        dates,
        eligibility_all,
        monthly_eligibility=monthly_eligibility,
    )
    broad_sleeves = sleeves._broad_sleeves(eligibility.columns)
    sleeve_panel = sleeves._sleeve_panel(headline_dates, broad_sleeves)
    prices_all = funds._performance_prices(daily)
    windows = {}
    for name, (start, end) in search.WINDOWS.items():
        window_dates = headline_dates[
            (headline_dates >= start) & (headline_dates <= end)
        ]
        windows[name] = {
            "dates": window_dates,
            "prices": _closed_window_prices(prices_all, window_dates, end),
        }
        ew_weights = eligibility_all.reindex(index=window_dates).astype(float)
        ew_weights = ew_weights.div(ew_weights.sum(axis=1), axis=0)
        ew_net, _ = funds._backtest(
            windows[name]["prices"],
            ew_weights,
            0.0,
            f"{funds.UNIVERSE}_{name}_AUM50_EW",
        )
        windows[name]["ew_nav"] = ew_net.get_portfolio_nav()
    return {
        "dates": headline_dates,
        "eligibility": eligibility,
        "signals": signal_panels,
        "signal_diagnostics": signal_diagnostics,
        "sleeves": broad_sleeves,
        "sleeve_panel": sleeve_panel,
        "windows": windows,
        "partition_loader": _load_partition,
        "runner": RUNNER,
        "rolling_aum": rolling_aum,
        "eligibility_all": eligibility_all,
    }


def _closed_window_prices(
    prices: pd.DataFrame,
    window_dates: pd.DatetimeIndex,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    """Crop prices to the declared window, retaining one pre-decision mark."""
    start = prices.index.searchsorted(window_dates.min(), side="right") - 1
    if start < 0:
        raise AssertionError("the first decision precedes the first performance price")
    end = prices.index.searchsorted(pd.Timestamp(end_date), side="right")
    result = prices.iloc[start:end]
    if result.empty or result.index.max() > pd.Timestamp(end_date):
        raise AssertionError("performance prices escape the declared window")
    return result


def _unfiltered_closed_context() -> dict[str, object]:
    """Build a price-history-only control with the same corrected window closure."""
    dates = funds._dates()
    headline_dates = dates[
        (dates >= funds.HEADLINE_START) & (dates <= funds.HEADLINE_END)
    ]
    daily = funds._read_daily()
    eligibility_all = funds._eligibility_for_dates(daily, dates)
    eligibility = eligibility_all.reindex(index=headline_dates).astype(bool)
    monthly_returns = funds._native_returns(daily, search.SIGNAL_FREQUENCY)
    monthly_eligibility = funds._eligibility_for_dates(
        daily, monthly_returns.index
    )
    signal_panels, signal_diagnostics = search._signal_panels(
        daily,
        dates,
        eligibility_all,
        monthly_eligibility=monthly_eligibility,
    )
    broad_sleeves = sleeves._broad_sleeves(eligibility.columns)
    sleeve_panel = sleeves._sleeve_panel(headline_dates, broad_sleeves)
    prices_all = funds._performance_prices(daily)
    windows = {}
    for name, (start, end) in search.WINDOWS.items():
        window_dates = headline_dates[
            (headline_dates >= start) & (headline_dates <= end)
        ]
        window_prices = _closed_window_prices(prices_all, window_dates, end)
        ew_weights = eligibility_all.reindex(index=window_dates).astype(float)
        ew_weights = ew_weights.div(ew_weights.sum(axis=1), axis=0)
        ew_net, _ = funds._backtest(
            window_prices,
            ew_weights,
            0.0,
            f"{funds.UNIVERSE}_{name}_UNFILTERED_CLOSED_EW",
        )
        windows[name] = {
            "dates": window_dates,
            "prices": window_prices,
            "ew_nav": ew_net.get_portfolio_nav(),
        }
    return {
        "dates": headline_dates,
        "eligibility": eligibility,
        "signals": signal_panels,
        "signal_diagnostics": signal_diagnostics,
        "sleeves": broad_sleeves,
        "sleeve_panel": sleeve_panel,
        "windows": windows,
        "partition_loader": funds._load_partition,
        "runner": f"{RUNNER}:matched_unfiltered_closed_control",
    }


def _eligibility_outputs(context: Mapping[str, object]) -> Mapping[str, pd.DataFrame]:
    """Return auditable per-date and per-fund AUM eligibility diagnostics."""
    dates = funds._dates()
    daily = funds._read_daily()
    rolling_aum = context["rolling_aum"]
    eligibility = context["eligibility_all"]
    if not isinstance(rolling_aum, pd.DataFrame):
        raise AssertionError("rolling AUM is not a DataFrame")
    if not isinstance(eligibility, pd.DataFrame):
        raise AssertionError("eligibility is not a DataFrame")
    history = funds._eligibility_for_dates(daily, dates)
    aum_at_dates = _aum_for_dates(dates, rolling_aum)
    aum_eligible = aum_at_dates.gt(AUM_THRESHOLD_USD_MILLIONS)
    per_date = pd.DataFrame(
        {
            "history_eligible": history.sum(axis=1),
            "aum_eligible": aum_eligible.sum(axis=1),
            "combined_eligible": eligibility.sum(axis=1),
            "removed_by_aum": (history & ~aum_eligible).sum(axis=1),
            "retained_share": eligibility.sum(axis=1).div(
                history.sum(axis=1).replace(0, np.nan)
            ),
        }
    )
    first = eligibility.apply(
        lambda column: column.index[column].min() if column.any() else pd.NaT
    )
    per_fund = pd.DataFrame(
        {
            "first_combined_eligible_date": first,
            "combined_eligible_dates": eligibility.sum(axis=0),
            "latest_12m_average_aum_usd_millions": aum_at_dates.iloc[-1],
            "ever_aum_eligible": aum_eligible.any(axis=0),
            "ever_combined_eligible": eligibility.any(axis=0),
        }
    )
    per_fund.index.name = "ticker"
    return {
        "aum_rolling_12m_usd_millions": rolling_aum.reset_index(),
        "aum_at_decision_dates": aum_at_dates.reset_index(),
        "eligibility_panel": eligibility.astype(int).reset_index(),
        "eligibility_by_date": per_date.reset_index(),
        "eligibility_by_fund": per_fund.reset_index(),
    }


def _acceptance(
    context: Mapping[str, object],
    diagnostics: pd.DataFrame,
    partition_runtime: pd.DataFrame,
) -> pd.DataFrame:
    """Validate AUM units/timing, partitions, signals, and exact portfolio weights."""
    rolling = context["rolling_aum"]
    eligibility = context["eligibility_all"]
    if not isinstance(rolling, pd.DataFrame) or not isinstance(eligibility, pd.DataFrame):
        raise AssertionError("invalid AUM context")
    aum_at_dates = _aum_for_dates(funds._dates(), rolling)
    eligible_aum = aum_at_dates.where(
        aum_at_dates.gt(AUM_THRESHOLD_USD_MILLIONS)
    ).stack()
    audit = pd.read_csv(AUM_AUDIT_FILE, index_col=0)
    signal_diagnostics = context["signal_diagnostics"]
    if not isinstance(signal_diagnostics, pd.DataFrame):
        raise AssertionError("invalid signal diagnostics")
    window_escape_days = max(
        (
            item["prices"].index.max() - pd.Timestamp(search.WINDOWS[name][1])
        ).days
        for name, item in context["windows"].items()
    )
    max_weight_error = float(diagnostics["maximum_exact_weight_error"].max())
    rows = [
        {
            "check": "Bloomberg AUM currencies equal USD",
            "measured": int(audit["aum_currency"].eq("USD").sum()),
            "tolerance": len(audit),
        },
        {
            "check": "funds with Bloomberg AUM history",
            "measured": int(audit["monthly_observations"].gt(0).sum()),
            "tolerance": len(audit),
        },
        {
            "check": "completed monthly AUM end date",
            "measured": str(_read_aum().index.max().date()),
            "tolerance": str(funds.FULL_END.date()),
        },
        {
            "check": "minimum AUM admitted by strict threshold",
            "measured": float(eligible_aum.min()),
            "tolerance": f"> {AUM_THRESHOLD_USD_MILLIONS}",
        },
        {
            "check": "partition cells",
            "measured": len(partition_runtime),
            "tolerance": len(funds._cells()),
        },
        {
            "check": "partition dates per cell",
            "measured": int(partition_runtime["dates"].min()),
            "tolerance": len(funds._dates()),
        },
        {
            "check": "eligible memberships missing from partitions",
            "measured": int(diagnostics["eligible_memberships_missing"].sum()),
            "tolerance": 0,
        },
        {
            "check": "maximum portfolio weight/exposure error",
            "measured": max_weight_error,
            "tolerance": search.WEIGHT_TOLERANCE,
        },
        {
            "check": "signal timing/reconstruction rows green",
            "measured": int(signal_diagnostics["status"].eq("PASS").sum()),
            "tolerance": len(signal_diagnostics),
        },
        {
            "check": "maximum performance-window end escape days",
            "measured": window_escape_days,
            "tolerance": 0,
        },
    ]
    result = pd.DataFrame(rows)
    numeric_checks = {
        "Bloomberg AUM currencies equal USD": True,
        "funds with Bloomberg AUM history": True,
        "partition cells": True,
        "partition dates per cell": True,
        "eligible memberships missing from partitions": True,
        "signal timing/reconstruction rows green": True,
    }
    result["status"] = "PASS"
    for index, row in result.iterrows():
        if row["check"] in numeric_checks:
            passed = float(row["measured"]) == float(row["tolerance"])
        elif row["check"] == "completed monthly AUM end date":
            passed = row["measured"] == row["tolerance"]
        elif row["check"] == "minimum AUM admitted by strict threshold":
            passed = float(row["measured"]) > AUM_THRESHOLD_USD_MILLIONS
        elif row["check"] == "maximum performance-window end escape days":
            passed = float(row["measured"]) <= 0.0
        else:
            passed = float(row["measured"]) <= float(row["tolerance"])
        result.loc[index, "status"] = "PASS" if passed else "FAIL"
    if not result["status"].eq("PASS").all():
        raise AssertionError(result.loc[~result["status"].eq("PASS")])
    return result


def _impact_vs_unfiltered(comparison: pd.DataFrame) -> pd.DataFrame:
    """Compare common full-window rows with the price-history-only legacy run."""
    old = pd.read_csv(
        search._root() / "comparison.csv", float_precision="round_trip"
    )
    comparison = comparison.loc[
        comparison["analysis_window"].eq(search.FULL_WINDOW)
    ]
    old = old.loc[old["analysis_window"].eq(search.FULL_WINDOW)]
    keys = ["candidate_id", "analysis_window"]
    metrics = [
        "cluster_net_return_annualized",
        "global_net_return_annualized",
        "delta_net_return_annualized",
        "cluster_sharpe_rf0",
        "global_sharpe_rf0",
        "delta_sharpe_rf0",
        "cluster_one_way_turnover_annualized",
        "global_one_way_turnover_annualized",
    ]
    merged = comparison[keys + metrics].merge(
        old[keys + metrics],
        on=keys,
        how="inner",
        suffixes=("_aum50", "_unfiltered"),
        validate="one_to_one",
    )
    for metric in metrics:
        merged[f"change_{metric}"] = (
            merged[f"{metric}_aum50"] - merged[f"{metric}_unfiltered"]
        )
    return merged


def run(*, max_workers: int = 4) -> Mapping[str, pd.DataFrame]:
    """Fit filtered partitions and execute the complete staged specification search."""
    started = time.perf_counter()
    partition_runtime = build_partitions(max_workers=max_workers)
    context = _context()
    marginal, marginal_tags = search._marginal_candidates()
    candidate_map = {candidate.candidate_id: candidate for candidate in marginal}
    score_cache: dict[tuple[str, str, int], pd.DataFrame] = {}
    partition_cache: dict[tuple[str, int], pd.DataFrame] = {}
    cluster_weight_cache: dict[str, pd.DataFrame] = {}
    global_weight_cache: dict[tuple[str, float, str], pd.DataFrame] = {}
    global_performance_cache: dict[
        tuple[str, float, str, str], dict[str, float]
    ] = {}

    phase_one_perf, phase_one_comparison, phase_one_diag = search._run_candidates(
        marginal,
        context,
        score_cache,
        partition_cache,
        cluster_weight_cache,
        global_weight_cache,
        global_performance_cache,
    )
    interaction, marginal_selection = search._top_interaction_candidates(
        phase_one_comparison, marginal_tags, candidate_map
    )
    candidate_map.update(
        {candidate.candidate_id: candidate for candidate in interaction}
    )
    phase_two_perf, phase_two_comparison, phase_two_diag = search._run_candidates(
        interaction,
        context,
        score_cache,
        partition_cache,
        cluster_weight_cache,
        global_weight_cache,
        global_performance_cache,
    )
    performance = pd.concat(
        [phase_one_perf, phase_two_perf], ignore_index=True
    )
    comparison = pd.concat(
        [phase_one_comparison, phase_two_comparison], ignore_index=True
    ).drop_duplicates(["candidate_id", "analysis_window"])
    diagnostics = pd.concat(
        [phase_one_diag, phase_two_diag], ignore_index=True
    ).drop_duplicates("candidate_id")
    selection = search._selection_table(comparison)
    drivers = search._driver_table(comparison, diagnostics, selection)
    costs = search._cost_sensitivity(
        selection,
        context,
        cluster_weight_cache,
        global_weight_cache,
        candidate_map,
    )
    components = search._component_attribution(
        selection,
        context,
        cluster_weight_cache,
        global_weight_cache,
        candidate_map,
    )
    acceptance = _acceptance(context, diagnostics, partition_runtime)
    grid_tags = pd.DataFrame(
        [
            {"candidate_id": candidate_id, "marginal_dimension": tag}
            for candidate_id, tags in marginal_tags.items()
            for tag in sorted(tags)
        ]
    )
    runtime = pd.DataFrame(
        [
            {
                "marginal_candidates": len(marginal),
                "interaction_candidates": len(interaction),
                "unique_candidates": int(comparison["candidate_id"].nunique()),
                "qis_performance_rows": len(performance),
                "runtime_seconds": time.perf_counter() - started,
            }
        ]
    )
    output = {
        **_eligibility_outputs(context),
        "signal_diagnostics": context["signal_diagnostics"],
        "partition_runtime": partition_runtime,
        "marginal_grid_tags": grid_tags,
        "marginal_finalist_selection": marginal_selection,
        "performance": performance,
        "comparison": comparison,
        "impact_vs_unfiltered": _impact_vs_unfiltered(comparison),
        "selection": selection,
        "weight_diagnostics": diagnostics,
        "driver_decomposition": drivers,
        "cost_sensitivity": costs,
        "component_attribution": components,
        "acceptance": acceptance,
        "runtime": runtime,
    }
    for name, frame in output.items():
        if not isinstance(frame, pd.DataFrame):
            raise AssertionError(f"output {name} is not a DataFrame")
        e5._write(frame, _root() / f"{name}.csv")
    return output


def run_matched_unfiltered_control() -> Mapping[str, pd.DataFrame]:
    """Reprice selected specifications with AUM as the only changed input."""
    filtered = pd.read_csv(
        _root() / "comparison.csv", float_precision="round_trip"
    )
    selection = pd.read_csv(
        _root() / "selection.csv", float_precision="round_trip"
    )
    selected_ids = list(selection["candidate_id"].drop_duplicates())
    if search.BASE_CANDIDATE.candidate_id not in selected_ids:
        selected_ids.append(search.BASE_CANDIDATE.candidate_id)
    candidates = []
    for candidate_id in selected_ids:
        row = filtered.loc[filtered["candidate_id"].eq(candidate_id)].iloc[0]
        candidates.append(search._candidate_from_row(row))

    context = _unfiltered_closed_context()
    performance, comparison, diagnostics = search._run_candidates(
        candidates,
        context,
        {},
        {},
        {},
        {},
        {},
    )
    keys = ["candidate_id", "analysis_window"]
    prefixes = ("cluster", "global", "delta")
    metrics = [
        f"{prefix}_{metric}"
        for prefix in prefixes
        for metric in search.COMPARISON_METRICS
    ]
    effects = filtered.loc[
        filtered["candidate_id"].isin(selected_ids), keys + metrics
    ].merge(
        comparison[keys + metrics],
        on=keys,
        suffixes=("_aum50", "_unfiltered"),
        validate="one_to_one",
    )
    for metric in metrics:
        effects[f"effect_{metric}"] = (
            effects[f"{metric}_aum50"] - effects[f"{metric}_unfiltered"]
        )
    acceptance = pd.DataFrame(
        [
            {
                "check": "matched control candidate rows",
                "measured": len(comparison),
                "tolerance": len(candidates) * len(search.WINDOWS),
            },
            {
                "check": "maximum matched-control weight error",
                "measured": float(
                    diagnostics["maximum_exact_weight_error"].max()
                ),
                "tolerance": search.WEIGHT_TOLERANCE,
            },
        ]
    )
    acceptance["status"] = [
        "PASS" if len(comparison) == len(candidates) * len(search.WINDOWS) else "FAIL",
        "PASS"
        if float(diagnostics["maximum_exact_weight_error"].max())
        <= search.WEIGHT_TOLERANCE
        else "FAIL",
    ]
    if not acceptance["status"].eq("PASS").all():
        raise AssertionError(acceptance)
    output = {
        "matched_unfiltered_performance": performance,
        "matched_unfiltered_comparison": comparison,
        "matched_aum_effect": effects,
        "matched_unfiltered_weight_diagnostics": diagnostics,
        "matched_unfiltered_acceptance": acceptance,
    }
    for name, frame in output.items():
        e5._write(frame, _root() / f"{name}.csv")
    return output


def run_holding_period_followup() -> Mapping[str, pd.DataFrame]:
    """Test slower rebalancing on the sole positive-training-gross specification."""
    context = _context()
    base = replace(
        search.BASE_CANDIDATE,
        signal_id="classic_12m_skip1",
        frequency="W-WED",
        span=156,
        q=0.15,
        weight_id="E40_F40_R20",
        stage="aum50_turnover_followup",
    )
    candidates = {
        construction: replace(base, construction=construction)
        for construction in search.CONSTRUCTIONS
    }
    score_cache: dict[tuple[str, str, int], pd.DataFrame] = {}
    partition_cache: dict[tuple[str, int], pd.DataFrame] = {}
    global_cache: dict[tuple[str, float, str], pd.DataFrame] = {}
    cluster_weights = {}
    global_weights = None
    for construction, candidate in candidates.items():
        cluster, control = search._build_weight_pair(
            candidate,
            context,
            score_cache,
            partition_cache,
            global_cache,
        )
        cluster_weights[construction] = cluster
        global_weights = control
    if global_weights is None:
        raise AssertionError("matched global weights were not built")

    rows = []
    for window_name, window in context["windows"].items():
        for schedule in search.HOLDING_SCHEDULES:
            global_payload = search._scheduled_performance_payload(
                global_weights,
                window,
                schedule,
                f"aum50_holding_{window_name}_{schedule}_global",
            )
            rows.append(
                {
                    "analysis_window": window_name,
                    "schedule": schedule,
                    "construction": "global",
                    "candidate_id": "global",
                    "rebalance_dates": len(
                        search._rebalance_dates(window["dates"], schedule)
                    ),
                    **global_payload,
                }
            )
            for construction, candidate in candidates.items():
                payload = search._scheduled_performance_payload(
                    cluster_weights[construction],
                    window,
                    schedule,
                    f"aum50_holding_{window_name}_{schedule}_{construction}",
                )
                rows.append(
                    {
                        "analysis_window": window_name,
                        "schedule": schedule,
                        "construction": construction,
                        "candidate_id": candidate.candidate_id,
                        "rebalance_dates": len(
                            search._rebalance_dates(window["dates"], schedule)
                        ),
                        **payload,
                    }
                )
    performance = pd.DataFrame(rows)
    global_rows = performance.loc[
        performance["construction"].eq("global")
    ].set_index(["analysis_window", "schedule"])
    comparison_rows = []
    for _, row in performance.loc[
        performance["construction"].ne("global")
    ].iterrows():
        reference = global_rows.loc[(row["analysis_window"], row["schedule"])]
        item = row.to_dict()
        for metric in search.COMPARISON_METRICS:
            item[f"global_{metric}"] = reference[metric]
            item[f"delta_{metric}"] = row[metric] - reference[metric]
        item["beats_global_net_return"] = (
            item["delta_net_return_annualized"] > 0.0
        )
        item["beats_global_sharpe"] = item["delta_sharpe_rf0"] > 0.0
        item["beats_global_both"] = (
            item["beats_global_net_return"] and item["beats_global_sharpe"]
        )
        comparison_rows.append(item)
    comparison = pd.DataFrame(comparison_rows)
    training = comparison.loc[
        comparison["analysis_window"].eq(search.TRAIN_WINDOW)
    ]
    selected = training.sort_values(
        ["delta_net_return_annualized", "delta_sharpe_rf0"],
        ascending=[False, False],
    ).iloc[0]
    selected_rows = comparison.loc[
        comparison["construction"].eq(selected["construction"])
        & comparison["schedule"].eq(selected["schedule"])
    ].copy()
    selected_rows.insert(0, "selection_rule", "max_training_net_delta")

    main = pd.read_csv(
        _root() / "comparison.csv", float_precision="round_trip"
    ).set_index(["candidate_id", "analysis_window"])
    monthly = comparison.loc[comparison["schedule"].eq("monthly")]
    regression_errors = []
    for _, row in monthly.iterrows():
        accepted = main.loc[(row["candidate_id"], row["analysis_window"])]
        regression_errors.extend(
            abs(float(row[f"delta_{metric}"]) - float(accepted[f"delta_{metric}"]))
            for metric in search.COMPARISON_METRICS
        )
    max_regression_error = max(regression_errors)
    acceptance = pd.DataFrame(
        [
            {
                "check": "monthly schedule regression",
                "measured": max_regression_error,
                "tolerance": search.WEIGHT_TOLERANCE,
                "status": "PASS"
                if max_regression_error <= search.WEIGHT_TOLERANCE
                else "FAIL",
            },
            {
                "check": "declared schedule comparison rows",
                "measured": len(comparison),
                "tolerance": (
                    len(search.WINDOWS)
                    * len(search.HOLDING_SCHEDULES)
                    * len(candidates)
                ),
                "status": "PASS"
                if len(comparison)
                == len(search.WINDOWS)
                * len(search.HOLDING_SCHEDULES)
                * len(candidates)
                else "FAIL",
            },
        ]
    )
    if not acceptance["status"].eq("PASS").all():
        raise AssertionError(acceptance)
    output = {
        "holding_period_performance": performance,
        "holding_period_comparison": comparison,
        "holding_period_selection": selected_rows,
        "holding_period_acceptance": acceptance,
    }
    for name, frame in output.items():
        e5._write(frame, _root() / f"{name}.csv")
    return output


def run_hybrid_recheck() -> Mapping[str, pd.DataFrame]:
    """Recheck the previously reported side-specific hybrid on closed AUM50 windows."""
    context = _context()
    candidates = {
        "owner_base": search.BASE_CANDIDATE,
        "classic_training_gross_leader": replace(
            search.BASE_CANDIDATE,
            signal_id="classic_12m_skip1",
            frequency="W-WED",
            span=156,
            q=0.15,
            weight_id="E40_F40_R20",
            construction="group_equal",
            stage="aum50_hybrid_recheck",
        ),
    }
    score_cache: dict[tuple[str, str, int], pd.DataFrame] = {}
    partition_cache: dict[tuple[str, int], pd.DataFrame] = {}
    global_cache: dict[tuple[str, float, str], pd.DataFrame] = {}
    weight_pairs = {
        name: search._build_weight_pair(
            candidate,
            context,
            score_cache,
            partition_cache,
            global_cache,
        )
        for name, candidate in candidates.items()
    }
    rows = []
    exposure_rows = []
    for candidate_name, candidate in candidates.items():
        cluster_weights, global_weights = weight_pairs[candidate_name]
        target = search._target_map(candidate.weight_id)
        hybrids = {
            variant: search._hybrid_weights(
                cluster_weights,
                global_weights,
                variant,
                context["sleeve_panel"],
                target,
            )
            for variant in search.HYBRID_VARIANTS
        }
        for variant, weights in hybrids.items():
            errors = {
                "long": float(
                    weights.clip(lower=0.0).sum(axis=1).sub(1.0).abs().max()
                ),
                "short": float(
                    (-weights.clip(upper=0.0)).sum(axis=1).sub(1.0).abs().max()
                ),
                "net": float(weights.sum(axis=1).abs().max()),
                "gross": float(weights.abs().sum(axis=1).sub(2.0).abs().max()),
            }
            exposure_rows.append(
                {
                    "candidate_name": candidate_name,
                    "hybrid_variant": variant,
                    **{f"max_{key}_error": value for key, value in errors.items()},
                    "maximum_error": max(errors.values()),
                }
            )
            for window_name, window in context["windows"].items():
                for schedule in search.HOLDING_SCHEDULES:
                    global_payload = search._scheduled_performance_payload(
                        global_weights,
                        window,
                        schedule,
                        f"aum50_hybrid_{candidate_name}_{window_name}_{schedule}_global",
                    )
                    hybrid_payload = search._scheduled_performance_payload(
                        weights,
                        window,
                        schedule,
                        f"aum50_hybrid_{candidate_name}_{variant}_{window_name}_{schedule}",
                    )
                    row = {
                        "candidate_name": candidate_name,
                        "candidate_id": candidate.candidate_id,
                        "hybrid_variant": variant,
                        "analysis_window": window_name,
                        "schedule": schedule,
                    }
                    for metric in search.COMPARISON_METRICS:
                        row[f"hybrid_{metric}"] = hybrid_payload[metric]
                        row[f"global_{metric}"] = global_payload[metric]
                        row[f"delta_{metric}"] = (
                            hybrid_payload[metric] - global_payload[metric]
                        )
                    row["beats_global_net_return"] = (
                        row["delta_net_return_annualized"] > 0.0
                    )
                    row["beats_global_sharpe"] = row["delta_sharpe_rf0"] > 0.0
                    row["beats_global_both"] = (
                        row["beats_global_net_return"]
                        and row["beats_global_sharpe"]
                    )
                    rows.append(row)
    comparison = pd.DataFrame(rows)
    exposure = pd.DataFrame(exposure_rows)
    acceptance = pd.DataFrame(
        [
            {
                "check": "hybrid exposure errors",
                "measured": float(exposure["maximum_error"].max()),
                "tolerance": search.WEIGHT_TOLERANCE,
            },
            {
                "check": "hybrid comparison rows",
                "measured": len(comparison),
                "tolerance": (
                    len(candidates)
                    * len(search.HYBRID_VARIANTS)
                    * len(search.WINDOWS)
                    * len(search.HOLDING_SCHEDULES)
                ),
            },
        ]
    )
    acceptance["status"] = [
        "PASS"
        if float(exposure["maximum_error"].max()) <= search.WEIGHT_TOLERANCE
        else "FAIL",
        "PASS"
        if len(comparison)
        == len(candidates)
        * len(search.HYBRID_VARIANTS)
        * len(search.WINDOWS)
        * len(search.HOLDING_SCHEDULES)
        else "FAIL",
    ]
    if not acceptance["status"].eq("PASS").all():
        raise AssertionError(acceptance)
    output = {
        "hybrid_recheck_comparison": comparison,
        "hybrid_recheck_exposure": exposure,
        "hybrid_recheck_acceptance": acceptance,
    }
    for name, frame in output.items():
        e5._write(frame, _root() / f"{name}.csv")
    return output


def main() -> None:
    """Run the complete AUM-filtered BlackRock specification search."""
    output = run(max_workers=4)
    print(output["selection"].to_string(index=False), flush=True)
    print(output["acceptance"].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
