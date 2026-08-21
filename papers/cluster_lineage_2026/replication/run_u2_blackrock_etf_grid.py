"""Run the BlackRock U2 long-only and long-short production-momentum grid.

The experiment transfers the frozen U1 operating specification before inspecting U2
payoffs: exact monthly ROSAA production momentum (long span 12, volatility span 13,
no reversal filter, MeanAdjType.NONE), q=0.25, group-equal cluster books, an asset-equal
global rank, ME decisions, one W-WED implementation period, and 10 bp costs.  The
exploratory treatment grid changes only the unsmoothed covariance/correlation partition:
B and W-MON through W-FRI at spans 24, 36, 52, and 156, plus ME at spans 12, 24, 36,
and 52.  The ex-ante transferred U1 cell is ME/span 36.

The universe is the 2026-08-15 official U.S. iShares ETF screener vintage.  Eligibility
is point-in-time with respect to each surviving fund's observed Bloomberg price history:
a fund enters after 12 valid W-WED return observations.  The source vintage cannot recover
funds liquidated before retrieval, so this is explicitly a current-cohort historical study,
not a survivorship-free historical product census.  EW-all is used only as the market
reference for alpha and beta; global rank is the sole payoff benchmark.
"""
from __future__ import annotations

import hashlib
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import qis
from factorlasso import compute_clusters_from_corr_matrix
from factorlasso.cluster_smoothing import _iter_correlation_inputs
from optimalportfolios.alphas.signals.momentum import compute_momentum_alpha
from optimalportfolios.alphas.signals.utils import score_within_clusters

import papers.cluster_lineage_2026.replication.run_backtests as e5
import papers.cluster_lineage_2026.replication.run_u1_covar_grid as u1_grid
import papers.cluster_lineage_2026.replication.run_u1_covar_grid_long_short as u1_ls
import papers.cluster_lineage_2026.replication.run_u1_covar_grid_long_short_prod as u1_prod
import papers.cluster_lineage_2026.replication.run_u1_me36_long_short as u1_single
from papers.cluster_lineage_2026.replication.empirical_specs import U1_OPTIMAL_SPEC
from papers.cluster_lineage_2026.replication.run_e5b import (
    GROUP_BUDGET_TOLERANCE,
    WEIGHT_TOLERANCE,
    _group_equal_from_ranks,
)
from papers.cluster_lineage_2026.replication.run_u1_global_grid import _backtest


SPEC = U1_OPTIMAL_SPEC
UNIVERSE = "blackrock_us_etfs"
RUNNER = (
    "papers/cluster_lineage_2026/replication/run_u2_blackrock_etf_grid.py"
)
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
ETF_LIST_FILE = DATA_DIR / "blackrock_us_etfs.csv"
METADATA_FILE = DATA_DIR / "blackrock_etf_metadata.csv"
PRICES_FILE = DATA_DIR / "blackrock_etf_adjusted_prices.csv"
RETURNS_FILE = DATA_DIR / "blackrock_etf_excess_log_returns.csv"
INPUT_FILES = (ETF_LIST_FILE, METADATA_FILE, PRICES_FILE, RETURNS_FILE)
FREQUENCY_SPANS: Mapping[str, tuple[int, ...]] = u1_grid.FREQUENCY_SPANS
FULL_START = pd.Timestamp("2006-08-31")
FULL_END = pd.Timestamp("2026-07-31")
HEADLINE_START = pd.Timestamp("2009-08-31")
HEADLINE_END = pd.Timestamp("2026-06-30")
HEADLINE_WINDOW = "headline_20090831_20260630"
AVAILABLE_WINDOW = "production_available"
PARTITION_CACHE_VERSION = 1
REGRESSION_TOLERANCE = 1e-12
EXPOSURE_TOLERANCE = 1e-12
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
    """Return and create the external BlackRock covariance-grid directory."""
    root = u1_grid._root().parent / UNIVERSE
    root.mkdir(parents=True, exist_ok=True)
    return root


def _partition_root() -> Path:
    """Return and create the compact partition-cache directory."""
    root = _root() / "partitions"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _cell_id(frequency: str, span: int) -> str:
    """Return a filesystem-safe stable grid-cell identifier."""
    return f"{frequency.replace('-', '_')}_span_{span:03d}"


def _cells() -> tuple[tuple[str, int], ...]:
    """Return the frozen 28-cell covariance grid."""
    return tuple(
        (frequency, span)
        for frequency, spans in FREQUENCY_SPANS.items()
        for span in spans
    )


@lru_cache(maxsize=None)
def _sha256(path: Path) -> str:
    """Return one source file's SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _input_fingerprint() -> str:
    """Return a stable joint digest of every frozen BlackRock input."""
    payload = "|".join(f"{path.name}:{_sha256(path)}" for path in INPUT_FILES)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _read_frame(path: Path) -> pd.DataFrame:
    """Read one wide dated research panel with round-trip floating precision."""
    frame = pd.read_csv(
        path,
        index_col=0,
        parse_dates=True,
        float_precision="round_trip",
    )
    frame.index = pd.DatetimeIndex(frame.index, name="date")
    return frame


def _read_daily() -> pd.DataFrame:
    """Read the frozen daily excess-log-return panel through the final full month."""
    return _read_frame(RETURNS_FILE).loc[:FULL_END]


def _read_prices() -> pd.DataFrame:
    """Read adjusted prices through the final full month for source checks."""
    return _read_frame(PRICES_FILE).loc[:FULL_END]


def _native_returns(daily: pd.DataFrame, frequency: str) -> pd.DataFrame:
    """Aggregate daily log returns to one covariance or signal cadence."""
    if frequency == "B":
        return daily.copy()
    return daily.resample(frequency).sum(min_count=1)


def _dates() -> pd.DatetimeIndex:
    """Return the fixed 240-date ME estimation and decision schedule."""
    return pd.date_range(FULL_START, FULL_END, freq="ME", name="date")


def _eligibility_for_dates(
    daily: pd.DataFrame, dates: pd.DatetimeIndex
) -> pd.DataFrame:
    """Return actual-history eligibility after the frozen 12-week warmup."""
    weekly = _native_returns(daily, SPEC.membership_warmup_frequency)
    counts = weekly.notna().cumsum()
    eligibility = counts.reindex(pd.DatetimeIndex(dates), method="ffill").fillna(0)
    eligibility = eligibility.ge(SPEC.membership_warmup_observations)
    eligibility.index.name = "date"
    return eligibility


def _partition_hash(panel: pd.DataFrame) -> str:
    """Hash one membership panel independently of pickle serialization."""
    values = pd.util.hash_pandas_object(panel, index=True).to_numpy().tobytes()
    columns = "\x1f".join(map(str, panel.columns)).encode("utf-8")
    return hashlib.sha256(values + columns).hexdigest()


def _partition_path(frequency: str, span: int) -> Path:
    """Return one compact partition-cache path."""
    return _partition_root() / f"{_cell_id(frequency, span)}.pkl"


def _model(span: int, frequency: str):
    """Copy the validated U1 clustering model and change only cadence and span."""
    model = u1_grid._model(span, frequency)
    observed = (
        model.cutoff_fraction,
        str(model.linkage_method),
        str(model.distance_transform.value),
        str(model.dependence_measure.value),
        bool(model.demean),
    )
    expected = (
        SPEC.covariance_cutoff_fraction,
        SPEC.covariance_linkage_method,
        SPEC.covariance_distance_transform,
        SPEC.covariance_dependence_measure,
        SPEC.covariance_demean,
    )
    if observed != expected:
        raise AssertionError(f"clustering model differs from frozen U1 spec: {observed}")
    return model


def preflight() -> Mapping[str, pd.DataFrame]:
    """Audit inputs, current-vintage limitations, schedule, and frozen specification."""
    etf_list = pd.read_csv(ETF_LIST_FILE)
    metadata = pd.read_csv(
        METADATA_FILE,
        parse_dates=[
            "blackrock_inception_date",
            "price_history_start",
            "price_history_end",
        ],
    )
    prices = _read_prices()
    daily = _read_daily()
    dates = _dates()
    eligibility = _eligibility_for_dates(daily, dates)
    classification_columns = [
        "asset_class",
        "sub_asset_class",
        "market_type",
        "region",
        "country",
        "investment_style",
    ]
    expected_return_mask = prices.notna() & prices.shift().notna()
    availability_mismatches = int(
        expected_return_mask.ne(daily.notna()).to_numpy().sum()
    )
    checks = pd.DataFrame(
        [
            {
                "check": "official list/metadata/price/return ticker identity",
                "measured": int(
                    etf_list["ticker"].tolist()
                    == metadata["ticker"].tolist()
                    == prices.columns.tolist()
                    == daily.columns.tolist()
                ),
                "tolerance": 1,
            },
            {
                "check": "price-to-return availability mask mismatches",
                "measured": availability_mismatches,
                "tolerance": 0,
            },
            {
                "check": "missing Aladdin classification cells",
                "measured": int(metadata[classification_columns].isna().sum().sum()),
                "tolerance": 0,
            },
            {
                "check": "duplicate ETF tickers",
                "measured": int(metadata["ticker"].duplicated().sum()),
                "tolerance": 0,
            },
            {
                "check": "decision schedule dates",
                "measured": len(dates),
                "tolerance": 240,
            },
            {
                "check": "headline dates",
                "measured": int(((dates >= HEADLINE_START) & (dates <= HEADLINE_END)).sum()),
                "tolerance": 203,
            },
            {
                "check": "covariance grid cells",
                "measured": len(_cells()),
                "tolerance": 28,
            },
            {
                "check": "U1 transferred ex-ante cell",
                "measured": f"{SPEC.covariance_frequency}/{SPEC.covariance_span}",
                "tolerance": "ME/36",
            },
        ]
    )
    checks["status"] = np.where(
        checks["measured"].astype(str).eq(checks["tolerance"].astype(str)),
        "PASS",
        "FAIL",
    )
    if not checks["status"].eq("PASS").all():
        raise AssertionError(checks.loc[~checks["status"].eq("PASS")])

    input_files = pd.DataFrame(
        [
            {
                "file": path.name,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for path in INPUT_FILES
        ]
    )
    eligible_counts = eligibility.sum(axis=1)
    data_quality = pd.DataFrame(
        [
            {
                "universe": UNIVERSE,
                "source_vintage": str(metadata["retrieval_date"].iloc[0]),
                "current_vintage_survivor_cohort": True,
                "funds": len(metadata),
                "daily_rows": len(daily),
                "daily_start": daily.index.min(),
                "daily_end": daily.index.max(),
                "asset_classes": metadata["asset_class"].nunique(),
                "sub_asset_classes": metadata["sub_asset_class"].nunique(),
                "eligible_first": int(eligible_counts.iloc[0]),
                "eligible_headline_start": int(eligible_counts.loc[HEADLINE_START]),
                "eligible_median": float(eligible_counts.median()),
                "eligible_last": int(eligible_counts.iloc[-1]),
                "inception_to_history_lag_gt_365_days": int(
                    metadata["price_history_start_lag_days"].gt(365).sum()
                ),
            }
        ]
    )
    anomalies = metadata.loc[
        metadata["price_history_start_lag_days"].gt(31),
        [
            "ticker",
            "name",
            "asset_class",
            "blackrock_inception_date",
            "price_history_start",
            "price_history_start_lag_days",
        ],
    ].sort_values("price_history_start_lag_days", ascending=False)
    anomalies = anomalies.reset_index(drop=True)
    return {
        "specification": SPEC.to_frame(name="U1_OPTIMAL_SPEC_20260815"),
        "preflight": checks,
        "input_files": input_files,
        "data_quality": data_quality,
        "coverage_anomalies": anomalies,
    }


def _compute_partition_cell(
    frequency: str, span: int, *, force: bool = False
) -> Mapping[str, object]:
    """Compute or load one point-in-time BlackRock partition panel."""
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
                "cell_id": _cell_id(frequency, span),
                "cache_status": "hit",
                "partition_hash": _partition_hash(cached["panel"]),
                "dates": len(cached["panel"]),
                "runtime_seconds": 0.0,
            }

    started = time.perf_counter()
    daily = _read_daily()
    dates = _dates()
    eligibility = _eligibility_for_dates(daily, dates)
    returns = _native_returns(daily, frequency)
    model = _model(span, frequency)
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
        "cell_id": _cell_id(frequency, span),
        "cache_status": "miss",
        "partition_hash": _partition_hash(panel),
        "dates": len(panel),
        "runtime_seconds": time.perf_counter() - started,
    }


def build_partitions(
    *, max_workers: int = 4, force: bool = False
) -> pd.DataFrame:
    """Build the 28 compact partition caches with four workers by default."""
    rows = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        pending = {
            executor.submit(_compute_partition_cell, frequency, span, force=force): (
                frequency,
                span,
            )
            for frequency, span in _cells()
        }
        for future in as_completed(pending):
            frequency, span = pending[future]
            row = future.result()
            rows.append(row)
            print(
                f"BlackRock partition {frequency}/{span}: {row['cache_status']} "
                f"({float(row['runtime_seconds']):.1f}s)",
                flush=True,
            )
    return pd.DataFrame(rows).sort_values(["frequency", "span"]).reset_index(drop=True)


def _load_partition(frequency: str, span: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load one verified compact membership cache."""
    with _partition_path(frequency, span).open("rb") as stream:
        cached = pickle.load(stream)
    if (
        cached.get("version") != PARTITION_CACHE_VERSION
        or cached.get("input_fingerprint") != _input_fingerprint()
        or cached.get("frequency") != frequency
        or cached.get("span") != span
    ):
        raise AssertionError(f"invalid partition cache for {frequency}/{span}")
    return cached["panel"], cached["diagnostics"]


def _partition_diagnostics() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-date and per-cell partition diagnostics."""
    rows = []
    for frequency, span in _cells():
        panel, diagnostics = _load_partition(frequency, span)
        frame = diagnostics.copy()
        frame.insert(0, "cell_id", _cell_id(frequency, span))
        frame.insert(0, "span", span)
        frame.insert(0, "frequency", frequency)
        frame["partition_hash"] = _partition_hash(panel)
        rows.append(frame)
    per_date = pd.concat(rows, ignore_index=True)
    summary = (
        per_date.groupby(
            ["frequency", "span", "cell_id", "partition_hash"], sort=False
        )
        .agg(
            dates=("date", "size"),
            member_min=("members", "min"),
            member_median=("members", "median"),
            member_max=("members", "max"),
            cluster_mean=("clusters", "mean"),
            cluster_std=("clusters", "std"),
            cluster_min=("clusters", "min"),
            cluster_max=("clusters", "max"),
        )
        .reset_index()
    )
    return per_date, summary


def _panel_dict(panel: pd.DataFrame) -> dict[pd.Timestamp, pd.Series]:
    """Convert a membership panel to the production cluster-score representation."""
    return {pd.Timestamp(date): row.dropna() for date, row in panel.iterrows()}


def _signal_inputs(
    daily: pd.DataFrame, dates: pd.DatetimeIndex, eligibility: pd.DataFrame
) -> Mapping[str, object]:
    """Build and validate the exact monthly ROSAA production signal."""
    period_returns = _native_returns(daily, SPEC.signal_frequency)
    period_eligibility = _eligibility_for_dates(daily, period_returns.index)
    simple_returns = np.expm1(period_returns)
    signal_prices = qis.returns_to_nav(simple_returns)
    benchmark_returns = simple_returns.where(period_eligibility).mean(axis=1)
    benchmark = qis.returns_to_nav(benchmark_returns.rename("EW").to_frame())["EW"]
    global_source, raw_source = compute_momentum_alpha(
        prices=signal_prices,
        benchmark_price=benchmark,
        returns_freq=SPEC.signal_frequency,
        group_data=None,
        long_span=SPEC.momentum_long_span,
        short_span=SPEC.momentum_short_span,
        vol_span=SPEC.momentum_vol_span,
        mean_adj_type=qis.MeanAdjType.NONE,
    )
    global_decision, global_timestamps = u1_prod._asof_panel(global_source, dates)
    raw_decision, raw_timestamps = u1_prod._asof_panel(raw_source, dates)
    if not global_timestamps.equals(raw_timestamps):
        raise AssertionError("global and raw production signal timestamps differ")
    lookahead_days = global_timestamps.sub(global_timestamps.index).dt.days
    valid_counts = global_decision.where(eligibility).notna().sum(axis=1)
    available = valid_counts.loc[valid_counts.gt(0)]
    if available.empty:
        raise AssertionError("the production signal has no eligible BlackRock observations")
    first_available = pd.Timestamp(available.index.min())
    roundtrip = qis.to_returns(
        signal_prices,
        freq=SPEC.signal_frequency,
        is_log_returns=True,
    ).reindex_like(period_returns)
    differences = roundtrip.subtract(period_returns).abs().to_numpy()
    finite = differences[np.isfinite(differences)]
    roundtrip_error = float(finite.max()) if finite.size else 0.0
    diagnostics = pd.DataFrame(
        [
            {
                "signal": SPEC.signal_name,
                "signal_frequency": SPEC.signal_frequency,
                "long_span": SPEC.momentum_long_span,
                "vol_span": SPEC.momentum_vol_span,
                "short_span": SPEC.momentum_short_span,
                "mean_adj_type": SPEC.momentum_mean_adj_type,
                "first_available_date": first_available,
                "valid_assets_first": int(available.iloc[0]),
                "valid_assets_headline_start": int(valid_counts.loc[HEADLINE_START]),
                "valid_assets_median": float(valid_counts.loc[available.index].median()),
                "valid_assets_last": int(valid_counts.iloc[-1]),
                "max_signal_lookahead_days": int(lookahead_days.max()),
                "return_roundtrip_max_abs_error": roundtrip_error,
                "return_roundtrip_tolerance": REGRESSION_TOLERANCE,
                "status": (
                    "PASS"
                    if int(lookahead_days.max()) <= 0
                    and roundtrip_error <= REGRESSION_TOLERANCE
                    else "FAIL"
                ),
            }
        ]
    )
    if not diagnostics["status"].eq("PASS").all():
        raise AssertionError(diagnostics)
    return {
        "global": global_decision,
        "raw_source": raw_source,
        "raw_decision": raw_decision,
        "first_available": first_available,
        "diagnostics": diagnostics,
    }


def _performance_prices(daily: pd.DataFrame) -> pd.DataFrame:
    """Return W-WED excess-return NAVs matching the frozen U1 payoff cadence."""
    weekly = _native_returns(daily, SPEC.performance_frequency)
    return qis.returns_to_nav(np.expm1(weekly))


def _window_prices(prices: pd.DataFrame, window_dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Crop prices while retaining the last mark not after the first decision."""
    start = prices.index.searchsorted(window_dates.min(), side="right") - 1
    if start < 0:
        raise AssertionError("the first decision precedes the first performance price")
    return prices.iloc[start:].loc[:FULL_END]


def _ew_reference(
    prices: pd.DataFrame,
    eligibility: pd.DataFrame,
    window_dates: pd.DatetimeIndex,
    window: str,
) -> pd.Series:
    """Backtest the no-cost EW-all market reference on the same window and lag."""
    eligible = eligibility.reindex(index=window_dates).astype(float)
    weights = eligible.div(eligible.sum(axis=1), axis=0)
    window_prices = _window_prices(prices, window_dates)
    net, _ = _backtest(window_prices, weights, 0.0, f"{UNIVERSE}_{window}_EW")
    return net.get_portfolio_nav()


def _performance_payload(net, gross, ew_nav: pd.Series) -> dict:
    """Return frozen performance metrics plus explicit gross annual return."""
    payload = e5._performance_row(net, gross, ew_nav)
    payload["gross_return_annualized"] = (
        payload["net_return_annualized"]
        + payload["cost_drag_bp_per_year"] / 10000.0
    )
    return payload


def _allocation_rows(
    *,
    window: str,
    strategy: str,
    frequency: str,
    span: float,
    cell_id: str,
    leg: str,
    weights: pd.DataFrame,
    asset_class: pd.Series,
) -> list[dict]:
    """Summarise decision-date gross and net exposure by Aladdin asset class."""
    mapping = asset_class.reindex(weights.columns)
    if mapping.isna().any():
        raise AssertionError("an ETF weight column lacks an Aladdin asset class")
    long_exposure = weights.clip(lower=0.0).T.groupby(mapping).sum().T
    short_exposure = (-weights.clip(upper=0.0)).T.groupby(mapping).sum().T
    net_exposure = weights.T.groupby(mapping).sum().T
    rows = []
    for classification in sorted(mapping.unique()):
        rows.append(
            {
                "analysis_window": window,
                "strategy": strategy,
                "frequency": frequency,
                "span": span,
                "cell_id": cell_id,
                "leg": leg,
                "asset_class": classification,
                "funds_in_current_vintage": int(mapping.eq(classification).sum()),
                "average_long_exposure": float(long_exposure[classification].mean()),
                "average_short_exposure_abs": float(
                    short_exposure[classification].mean()
                ),
                "average_net_exposure": float(net_exposure[classification].mean()),
                "max_abs_net_exposure": float(
                    net_exposure[classification].abs().max()
                ),
            }
        )
    return rows


def _long_only_weights(
    scores: pd.DataFrame,
    eligibility: pd.DataFrame,
    groups: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Build the frozen q=0.25 group-equal top-rank long-only book."""
    ranks = e5._rank_panel(scores, groups)
    return _group_equal_from_ranks(
        ranks,
        eligibility,
        groups,
        SPEC.quantile,
        u1_grid.UNIVERSE,
    )


def _run_long_only_leg(
    *,
    window: str,
    frequency: str,
    span: float,
    cell_id: str,
    leg: str,
    construction: str,
    prices: pd.DataFrame,
    scores: pd.DataFrame,
    eligibility: pd.DataFrame,
    groups: pd.DataFrame,
    ew_nav: pd.Series,
) -> tuple[dict, dict, dict, pd.DataFrame]:
    """Backtest one long-only leg and return performance and exact checks."""
    weights, group_counts, validation = _long_only_weights(
        scores, eligibility, groups
    )
    net, gross = _backtest(
        prices,
        weights,
        SPEC.cost_bps / 10000.0,
        f"{UNIVERSE}_{window}_{leg}_long_only",
    )
    performance = {
        "universe": UNIVERSE,
        "analysis_window": window,
        "strategy": "long_only",
        "frequency": frequency,
        "span": span,
        "span_unit": "not_applicable" if leg == "global" else "native_observations",
        "q": SPEC.quantile,
        "construction": construction,
        "leg": leg,
        "cell_id": cell_id,
        **_performance_payload(net, gross, ew_nav),
        "runner": RUNNER,
    }
    weight_error = float(validation["weight_sum_abs_error"].max())
    budget_error = float(validation["max_group_budget_abs_error"].max())
    acceptance = {
        "analysis_window": window,
        "strategy": "long_only",
        "leg": leg,
        "frequency": frequency,
        "span": span,
        "cell_id": cell_id,
        "max_weight_sum_abs_error": weight_error,
        "weight_sum_tolerance": WEIGHT_TOLERANCE,
        "max_group_budget_abs_error": budget_error,
        "group_budget_tolerance": GROUP_BUDGET_TOLERANCE,
        "status": (
            "PASS"
            if weight_error <= WEIGHT_TOLERANCE
            and budget_error <= GROUP_BUDGET_TOLERANCE
            else "FAIL"
        ),
    }
    selected = weights.gt(0.0).sum(axis=1)
    diagnostics = {
        "analysis_window": window,
        "strategy": "long_only",
        "frequency": frequency,
        "span": span,
        "cell_id": cell_id,
        "leg": leg,
        "available_groups_mean": float(group_counts.mean()),
        "available_groups_std": float(group_counts.std()),
        "selected_assets_mean": float(selected.mean()),
        "selected_assets_min": int(selected.min()),
        "selected_assets_max": int(selected.max()),
    }
    return performance, acceptance, diagnostics, weights


def _run_long_short_leg(
    *,
    window: str,
    frequency: str,
    span: float,
    cell_id: str,
    leg: str,
    construction: str,
    prices: pd.DataFrame,
    scores: pd.DataFrame,
    eligibility: pd.DataFrame,
    groups: pd.DataFrame,
    ew_nav: pd.Series,
) -> tuple[dict, dict, pd.DataFrame]:
    """Backtest one +100/-100 leg and enforce side and group neutrality."""
    weights, exposure, side_validation = u1_single._leg_weights(
        scores, eligibility, groups
    )
    net, gross = _backtest(
        prices,
        weights,
        SPEC.cost_bps / 10000.0,
        f"{UNIVERSE}_{window}_{leg}_long_short",
    )
    performance = {
        "universe": UNIVERSE,
        "analysis_window": window,
        "strategy": "long_short",
        "frequency": frequency,
        "span": span,
        "span_unit": "not_applicable" if leg == "global" else "native_observations",
        "q": SPEC.quantile,
        "construction": construction,
        "target_long_exposure": SPEC.long_short_long_exposure,
        "target_short_exposure": SPEC.long_short_short_exposure,
        "target_gross_exposure": 2.0,
        "leg": leg,
        "cell_id": cell_id,
        **_performance_payload(net, gross, ew_nav),
        "runner": RUNNER,
    }
    acceptance = u1_single._acceptance(window, leg, exposure, side_validation)
    post_net = u1_ls._group_exposure_panel(weights, groups)
    neutrality_error = float(post_net["group_l1_net_exposure"].max())
    acceptance.update(
        {
            "strategy": "long_short",
            "frequency": frequency,
            "span": span,
            "cell_id": cell_id,
            "max_post_net_group_l1_exposure": neutrality_error,
            "post_net_group_l1_tolerance": EXPOSURE_TOLERANCE,
        }
    )
    if neutrality_error > EXPOSURE_TOLERANCE:
        acceptance["status"] = "FAIL"
    return performance, acceptance, weights


def _comparison(performance: pd.DataFrame) -> pd.DataFrame:
    """Compare every cluster cell only with the matched-strategy global rank."""
    globals_frame = performance.loc[performance["leg"].eq("global")].set_index(
        ["analysis_window", "strategy"]
    )
    rows = []
    for _, cluster in performance.loc[~performance["leg"].eq("global")].iterrows():
        key = (cluster["analysis_window"], cluster["strategy"])
        global_row = globals_frame.loc[key]
        row = {
            "analysis_window": cluster["analysis_window"],
            "strategy": cluster["strategy"],
            "frequency": cluster["frequency"],
            "span": cluster["span"],
            "cell_id": cluster["cell_id"],
            "q": cluster["q"],
            "cluster_leg": cluster["leg"],
            "benchmark_leg": "global",
        }
        for metric in COMPARISON_METRICS:
            row[f"cluster_{metric}"] = cluster[metric]
            row[f"global_{metric}"] = global_row[metric]
            row[f"delta_{metric}"] = cluster[metric] - global_row[metric]
        row["beats_global_net_return"] = row["delta_net_return_annualized"] > 0.0
        row["beats_global_sharpe"] = row["delta_sharpe_rf0"] > 0.0
        row["lower_volatility_than_global"] = (
            row["delta_volatility_annualized"] < 0.0
        )
        row["beats_global_return_and_sharpe"] = (
            row["beats_global_net_return"] and row["beats_global_sharpe"]
        )
        row["mean_variance_dominates_global"] = (
            row["beats_global_net_return"]
            and row["lower_volatility_than_global"]
        )
        row["is_transferred_u1_cell"] = (
            cluster["frequency"] == SPEC.covariance_frequency
            and int(cluster["span"]) == SPEC.covariance_span
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _win_summary(comparison: pd.DataFrame) -> pd.DataFrame:
    """Summarise global-relative breadth and leaders for each strategy and window."""
    rows = []
    for (window, strategy), panel in comparison.groupby(
        ["analysis_window", "strategy"], sort=False
    ):
        best = panel.sort_values(
            ["delta_net_return_annualized", "delta_sharpe_rf0"],
            ascending=[False, False],
        ).iloc[0]
        best_absolute = panel.sort_values(
            ["cluster_net_return_annualized", "cluster_sharpe_rf0"],
            ascending=[False, False],
        ).iloc[0]
        transferred = panel.loc[panel["is_transferred_u1_cell"]].iloc[0]
        rows.append(
            {
                "analysis_window": window,
                "strategy": strategy,
                "cells": len(panel),
                "net_return_wins": int(panel["beats_global_net_return"].sum()),
                "sharpe_wins": int(panel["beats_global_sharpe"].sum()),
                "return_and_sharpe_wins": int(
                    panel["beats_global_return_and_sharpe"].sum()
                ),
                "mean_variance_wins": int(
                    panel["mean_variance_dominates_global"].sum()
                ),
                "best_relative_frequency": best["frequency"],
                "best_relative_span": int(best["span"]),
                "best_relative_delta_net_return_annualized": best[
                    "delta_net_return_annualized"
                ],
                "best_relative_delta_sharpe_rf0": best["delta_sharpe_rf0"],
                "best_absolute_frequency": best_absolute["frequency"],
                "best_absolute_span": int(best_absolute["span"]),
                "best_absolute_cluster_net_return_annualized": best_absolute[
                    "cluster_net_return_annualized"
                ],
                "transferred_u1_delta_net_return_annualized": transferred[
                    "delta_net_return_annualized"
                ],
                "transferred_u1_delta_sharpe_rf0": transferred["delta_sharpe_rf0"],
                "transferred_u1_beats_global_net_return": transferred[
                    "beats_global_net_return"
                ],
            }
        )
    return pd.DataFrame(rows)


def run(*, max_workers: int = 4) -> Mapping[str, pd.DataFrame]:
    """Execute cached partitions and both production-momentum payoff grids."""
    started = time.perf_counter()
    output = dict(preflight())
    partition_runtime = build_partitions(max_workers=max_workers)
    per_date_partitions, partition_summary = _partition_diagnostics()
    daily = _read_daily()
    dates = _dates()
    eligibility = _eligibility_for_dates(daily, dates)
    signal = _signal_inputs(daily, dates, eligibility)
    metadata = pd.read_csv(METADATA_FILE).set_index("ticker")
    asset_class = metadata["asset_class"]
    windows = {
        HEADLINE_WINDOW: dates[(dates >= HEADLINE_START) & (dates <= HEADLINE_END)],
        AVAILABLE_WINDOW: dates[dates >= signal["first_available"]],
    }
    prices = _performance_prices(daily)
    ew_navs = {
        window: _ew_reference(prices, eligibility, window_dates, window)
        for window, window_dates in windows.items()
    }

    performance_rows = []
    acceptance_rows = []
    construction_rows = []
    allocation_rows = []
    risk_rows = []
    inputs = {}
    for window, window_dates in windows.items():
        eligible = eligibility.reindex(index=window_dates)
        scores = signal["global"].reindex(
            index=window_dates, columns=eligible.columns
        ).where(eligible)
        groups = pd.DataFrame("global", index=window_dates, columns=eligible.columns)
        window_prices = _window_prices(prices, window_dates)
        long_performance, long_acceptance, long_diagnostic, long_weights = (
            _run_long_only_leg(
                window=window,
                frequency="BENCHMARK_INVARIANT",
                span=np.nan,
                cell_id="global",
                leg="global",
                construction=SPEC.global_construction,
                prices=window_prices,
                scores=scores,
                eligibility=eligible,
                groups=groups,
                ew_nav=ew_navs[window],
            )
        )
        short_performance, short_acceptance, short_weights = _run_long_short_leg(
            window=window,
            frequency="BENCHMARK_INVARIANT",
            span=np.nan,
            cell_id="global",
            leg="global",
            construction=SPEC.global_construction,
            prices=window_prices,
            scores=scores,
            eligibility=eligible,
            groups=groups,
            ew_nav=ew_navs[window],
        )
        performance_rows.extend([long_performance, short_performance])
        acceptance_rows.extend([long_acceptance, short_acceptance])
        construction_rows.append(long_diagnostic)
        allocation_rows.extend(
            _allocation_rows(
                window=window,
                strategy="long_only",
                frequency="BENCHMARK_INVARIANT",
                span=np.nan,
                cell_id="global",
                leg="global",
                weights=long_weights,
                asset_class=asset_class,
            )
        )
        allocation_rows.extend(
            _allocation_rows(
                window=window,
                strategy="long_short",
                frequency="BENCHMARK_INVARIANT",
                span=np.nan,
                cell_id="global",
                leg="global",
                weights=short_weights,
                asset_class=asset_class,
            )
        )
        inputs[window] = {
            "dates": window_dates,
            "eligibility": eligible,
            "prices": window_prices,
            "global_long_weights": long_weights,
            "global_short_weights": short_weights,
        }

    for frequency, span in _cells():
        cell_started = time.perf_counter()
        groups_all, _ = _load_partition(frequency, span)
        cell_id = _cell_id(frequency, span)
        cluster_scores_all = score_within_clusters(
            raw_signal=signal["raw_source"],
            rolling_clusters=_panel_dict(groups_all),
            min_cluster_size=SPEC.momentum_min_cluster_size,
        )
        for window, item in inputs.items():
            window_dates = item["dates"]
            eligible = item["eligibility"]
            groups = groups_all.reindex(
                index=window_dates, columns=eligible.columns
            )
            scores = cluster_scores_all.reindex(
                index=window_dates, columns=eligible.columns
            ).where(eligible)
            leg = f"cluster_{cell_id}"
            long_performance, long_acceptance, long_diagnostic, long_weights = (
                _run_long_only_leg(
                    window=window,
                    frequency=frequency,
                    span=span,
                    cell_id=cell_id,
                    leg=leg,
                    construction=SPEC.cluster_construction,
                    prices=item["prices"],
                    scores=scores,
                    eligibility=eligible,
                    groups=groups,
                    ew_nav=ew_navs[window],
                )
            )
            short_performance, short_acceptance, short_weights = _run_long_short_leg(
                window=window,
                frequency=frequency,
                span=span,
                cell_id=cell_id,
                leg=leg,
                construction=SPEC.cluster_construction,
                prices=item["prices"],
                scores=scores,
                eligibility=eligible,
                groups=groups,
                ew_nav=ew_navs[window],
            )
            performance_rows.extend([long_performance, short_performance])
            acceptance_rows.extend([long_acceptance, short_acceptance])
            construction_rows.append(long_diagnostic)
            allocation_rows.extend(
                _allocation_rows(
                    window=window,
                    strategy="long_only",
                    frequency=frequency,
                    span=span,
                    cell_id=cell_id,
                    leg=leg,
                    weights=long_weights,
                    asset_class=asset_class,
                )
            )
            allocation_rows.extend(
                _allocation_rows(
                    window=window,
                    strategy="long_short",
                    frequency=frequency,
                    span=span,
                    cell_id=cell_id,
                    leg=leg,
                    weights=short_weights,
                    asset_class=asset_class,
                )
            )
            risk = u1_ls._risk_diagnostic(
                window,
                frequency,
                span,
                short_weights,
                item["global_short_weights"],
                groups,
            )
            risk["strategy"] = "long_short"
            risk_rows.append(risk)
        partition_runtime.loc[
            partition_runtime["cell_id"].eq(cell_id), "payoff_runtime_seconds"
        ] = time.perf_counter() - cell_started
        print(f"BlackRock payoffs {frequency}/{span}: complete", flush=True)

    performance = pd.DataFrame(performance_rows).sort_values(
        ["analysis_window", "strategy", "frequency", "span"], na_position="first"
    ).reset_index(drop=True)
    acceptance = pd.DataFrame(acceptance_rows)
    if not acceptance["status"].eq("PASS").all():
        raise AssertionError(acceptance.loc[~acceptance["status"].eq("PASS")])
    comparison = _comparison(performance)
    win_summary = _win_summary(comparison)
    transferred = comparison.loc[comparison["is_transferred_u1_cell"]].copy()
    partition_runtime["total_run_seconds"] = time.perf_counter() - started
    output.update(
        {
            "signal_diagnostics": signal["diagnostics"],
            "partition_diagnostics": per_date_partitions,
            "partition_summary": partition_summary,
            "performance": performance,
            "comparison_vs_global": comparison,
            "win_summary": win_summary,
            "u1_transfer_cell": transferred,
            "construction_diagnostics": pd.DataFrame(construction_rows),
            "allocation_diagnostics": pd.DataFrame(allocation_rows),
            "risk_diagnostics": pd.DataFrame(risk_rows),
            "acceptance": acceptance,
            "runtime": partition_runtime,
        }
    )
    for name, frame in output.items():
        e5._write(frame, _root() / f"{name}.csv")
    return output


def _hash_outputs() -> dict[str, str]:
    """Hash every numerical artifact except timing and replay records."""
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(_root().glob("*.csv"))
        if path.name not in {"runtime.csv", "determinism.csv"}
    }


def verify_determinism(*, max_workers: int = 4) -> pd.DataFrame:
    """Replay cached payoffs and require byte-identical research artifacts."""
    run(max_workers=max_workers)
    first = _hash_outputs()
    run(max_workers=max_workers)
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
    """Run the complete deterministic BlackRock U2 grid."""
    replay = verify_determinism(max_workers=4)
    print(
        f"BlackRock U2 long-only/long-short grid: PASS "
        f"({len(replay)}/{len(replay)} deterministic)",
        flush=True,
    )


if __name__ == "__main__":
    main()
