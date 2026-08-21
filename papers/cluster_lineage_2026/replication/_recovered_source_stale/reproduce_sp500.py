"""Reproduce S&P 500 baseline and M1 cluster-lineage stability with FactorLasso only."""

from __future__ import annotations

import os
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from factorlasso import (
    ClusterSmootherType,
    CurrentFactorCovarData,
    DependenceMeasure,
    DistanceTransform,
    LassoModel,
    LassoModelType,
    RollingFactorCovarData,
    VarianceColumns,
    analyze_cluster_lineage,
    compute_ewm,
    compute_ewm_covar,
    compute_rolling_smoothed_clusters,
)
from factorlasso.ewm_utils import NanBackfill

PRICE_FILE = "sp500_adjusted_close_2005_to_current.csv"
CONFIGS = ("baseline", "M1_delta_0.05")
DATES = pd.date_range("2021-08-01", "2026-08-01", freq="ME")
ANNUALISATION = 52.0
OP_PIPELINE_CHURN = {"baseline": 3.21146937738715, "M1_delta_0.05": 0.557430}

_WORKER_RETURNS: Optional[pd.DataFrame] = None
_WORKER_FACTORS: Optional[pd.DataFrame] = None
_WORKER_CONFIG: Optional[str] = None
_WORKER_CLUSTERS: Optional[Dict[pd.Timestamp, pd.Series]] = None
_WORKER_LINKAGES: Optional[Dict[pd.Timestamp, np.ndarray]] = None
_WORKER_CUTOFFS: Optional[Dict[pd.Timestamp, float]] = None


def _data_dir() -> Path:
    """Return the configured folder containing the S&P 500 price CSV."""
    default = Path.home() / "OneDrive" / "analytics" / "outputs" / "factorlasso_returns"
    return Path(os.environ.get("FACTORLASSO_SP500_DATA_DIR", default))


def _output_dir() -> Path:
    """Return and create the external cache/output folder."""
    default = Path.home() / "OneDrive" / "analytics" / "outputs" / "factorlasso_cluster_lineage"
    output = Path(os.environ.get("FACTORLASSO_LINEAGE_OUTPUT_DIR", default))
    output.mkdir(parents=True, exist_ok=True)
    return output


def fetch_prices_with_yfinance(tickers: list[str], start: str, output: Path) -> None:
    """Optionally fetch adjusted closes for external reproduction users.

    This papers-only convenience is intentionally guarded and is never called by the official
    reproduction. Uncomment a call in a private copy if the fixed input CSV is unavailable.
    """
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover - optional external-user path
        raise ImportError("install yfinance separately to use this papers-only helper") from exc
    prices = yf.download(tickers, start=start, auto_adjust=True, progress=False)["Close"]
    prices.to_csv(output)


# Example for external users only (the fixed study never downloads data):
# fetch_prices_with_yfinance(["AAPL", "MSFT"], "2005-01-01", Path("prices.csv"))


def load_inputs() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load fixed prices and construct weekly asset and equal-weight market log returns."""
    prices = pd.read_csv(_data_dir() / PRICE_FILE, index_col=0, parse_dates=True).sort_index()
    prices = prices.loc["2017-01-01":].asfreq("B", method="ffill").resample("W-WED").last()
    asset_returns = np.log(prices / prices.shift(1))
    market_simple = prices.pct_change(fill_method=None).mean(axis=1, skipna=True)
    factor_returns = np.log1p(market_simple).rename("Market").to_frame()
    factor_returns.iloc[0] = np.nan
    return asset_returns, factor_returns


def make_model(config: str) -> LassoModel:
    """Return the fixed FCGL model for the baseline or M1 configuration."""
    smoother = {
        "baseline": {"cluster_smoother_type": ClusterSmootherType.NONE},
        "M1_delta_0.05": {
            "cluster_smoother_type": ClusterSmootherType.PARTITION_BONUS,
            "smoother_delta": 0.05,
        },
    }
    if config not in smoother:
        raise ValueError(f"unknown configuration {config!r}")
    return LassoModel(
        model_type=LassoModelType.FACTOR_CLUSTER_GROUP_LASSO,
        reg_lambda=1e-5,
        span=156,
        span_freq_dict={"W-WED": 156},
        cutoff_fraction=0.6,
        linkage_method="ward",
        distance_transform=DistanceTransform.ONE_MINUS_RHO,
        dependence_measure=DependenceMeasure.PEARSON,
        group_penalty="normalized",
        l1_weight=0.0,
        demean=True,
        solver="CLARABEL",
        warmup_period=12,
        nonneg=False,
        auto_sign_constraints=True,
        auto_sign_threshold_t=1.0,
        auto_sign_adaptive_weights=True,
        auto_sign_adaptive_gamma=1.0,
        auto_sign_adaptive_floor=0.5,
        unilasso_loo=True,
        unilasso_non_negative=True,
        **smoother[config],
    )


def _factor_covar(factor_returns: pd.DataFrame, date: pd.Timestamp) -> pd.DataFrame:
    """Compute annualised W-WED span-52 demeaned factor covariance through one date."""
    raw = factor_returns.loc[:date].dropna()
    demeaned = raw - compute_ewm(raw, span=52)
    demeaned = demeaned.iloc[1:]
    covariance = compute_ewm_covar(
        demeaned.to_numpy(),
        span=52,
        nan_backfill=NanBackfill.ZERO_FILL,
    )
    return pd.DataFrame(
        ANNUALISATION * covariance,
        index=factor_returns.columns,
        columns=factor_returns.columns,
    )


def _fit_snapshot(
    date: pd.Timestamp,
    asset_returns: pd.DataFrame,
    factor_returns: pd.DataFrame,
    config: str,
    clusters: Optional[pd.Series] = None,
    linkage: Optional[np.ndarray] = None,
    cutoff: Optional[float] = None,
) -> CurrentFactorCovarData:
    """Fit and assemble one expanding-window covariance snapshot from primitives."""
    date = pd.Timestamp(date)
    y = asset_returns.loc[:date]
    x = factor_returns.reindex(y.index)
    model = make_model(config)
    model.fit(
        x=x,
        y=y,
        span=156,
        external_clusters=clusters,
        external_linkage=linkage,
        external_cutoff=cutoff,
    )
    result = model.estimation_result_
    betas = model.estimated_betas
    residuals = y - x @ betas.T
    cluster_series = model.clusters.astype(str).radd("W-WED:")
    y_variances = pd.DataFrame(
        {
            VarianceColumns.EWMA_VARIANCE.value: ANNUALISATION * result.ss_total,
            VarianceColumns.RESIDUAL_VARS.value: ANNUALISATION * result.ss_res,
            VarianceColumns.INSAMPLE_ALPHA.value: ANNUALISATION * result.alpha,
            VarianceColumns.R2.value: np.clip(result.r2, 0.0, None),
        },
        index=y.columns,
    )
    return CurrentFactorCovarData(
        x_covar=_factor_covar(factor_returns, date),
        y_betas=betas,
        y_variances=y_variances,
        estimation_date=date,
        residuals=ANNUALISATION * residuals,
        clusters=cluster_series,
    )


def _initialize_worker(
    config: str,
    clusters: Optional[Dict[pd.Timestamp, pd.Series]],
    linkages: Optional[Dict[pd.Timestamp, np.ndarray]],
    cutoffs: Optional[Dict[pd.Timestamp, float]],
) -> None:
    """Load fixed inputs once and install one configuration's smoothing path in a worker."""
    global _WORKER_RETURNS, _WORKER_FACTORS, _WORKER_CONFIG
    global _WORKER_CLUSTERS, _WORKER_LINKAGES, _WORKER_CUTOFFS
    _WORKER_RETURNS, _WORKER_FACTORS = load_inputs()
    _WORKER_CONFIG = config
    _WORKER_CLUSTERS = clusters
    _WORKER_LINKAGES = linkages
    _WORKER_CUTOFFS = cutoffs


def _fit_and_cache(date: pd.Timestamp, path: Path) -> str:
    """Fit one worker snapshot and pickle it to the external cache."""
    assert _WORKER_RETURNS is not None
    assert _WORKER_FACTORS is not None
    assert _WORKER_CONFIG is not None
    clusters = None if _WORKER_CLUSTERS is None else _WORKER_CLUSTERS[date]
    linkage = None if _WORKER_LINKAGES is None else _WORKER_LINKAGES[date]
    cutoff = None if _WORKER_CUTOFFS is None else _WORKER_CUTOFFS[date]
    snapshot = _fit_snapshot(
        date,
        _WORKER_RETURNS,
        _WORKER_FACTORS,
        _WORKER_CONFIG,
        clusters,
        linkage,
        cutoff,
    )
    with path.open("wb") as file:
        pickle.dump(snapshot, file)
    return path.name


def _smoothing_path(
    asset_returns: pd.DataFrame,
) -> Tuple[
    Dict[pd.Timestamp, pd.Series],
    Dict[pd.Timestamp, np.ndarray],
    Dict[pd.Timestamp, float],
]:
    """Compute the causal M1 partition, linkage, and cutoff path once."""
    rolling = compute_rolling_smoothed_clusters(
        y=asset_returns,
        estimation_dates=list(DATES),
        lasso_model=make_model("M1_delta_0.05"),
    )
    return rolling.clusters, rolling.linkages, rolling.cutoffs


def estimate_rolling(config: str, max_workers: int) -> RollingFactorCovarData:
    """Fit missing external-cache snapshots and return one rolling covariance panel."""
    asset_returns, _ = load_inputs()
    clusters = linkages = cutoffs = None
    if config == "M1_delta_0.05":
        clusters, linkages, cutoffs = _smoothing_path(asset_returns)
    cache = _output_dir() / config
    cache.mkdir(parents=True, exist_ok=True)
    jobs = [
        (pd.Timestamp(date), cache / f"{pd.Timestamp(date):%Y%m%d}.pkl")
        for date in DATES
        if not (cache / f"{pd.Timestamp(date):%Y%m%d}.pkl").exists()
    ]
    if jobs:
        with ProcessPoolExecutor(
            max_workers=max_workers,
            initializer=_initialize_worker,
            initargs=(config, clusters, linkages, cutoffs),
        ) as executor:
            futures = [executor.submit(_fit_and_cache, date, path) for date, path in jobs]
            for count, future in enumerate(as_completed(futures), start=1):
                print(f"{config}: cached {count}/{len(futures)} {future.result()}", flush=True)
    data = {}
    for date in DATES:
        with (cache / f"{pd.Timestamp(date):%Y%m%d}.pkl").open("rb") as file:
            data[pd.Timestamp(date)] = pickle.load(file)
    return RollingFactorCovarData(data=data)


def annualized_churn(panel: pd.DataFrame) -> float:
    """Return per-asset id changes per year over consecutive valid observations."""
    left, right = panel.iloc[:-1].copy(), panel.iloc[1:].copy()
    left.index = right.index
    valid = left.notna() & right.notna()
    changes = ((left != right) & valid).to_numpy().sum()
    pairs = valid.to_numpy().sum()
    years = (panel.index[-1] - panel.index[0]).days / 365.25
    return float(changes * (len(panel) - 1) / (pairs * years))


def greedy_raw_membership_panel(covar_data: RollingFactorCovarData) -> pd.DataFrame:
    """Greedily normalize raw clusters by maximum consecutive member overlap."""
    rows = {}
    next_id = 0
    prior_members = {}
    for date in covar_data.dates:
        current = covar_data[date].clusters.dropna()
        groups = {label: set(current.index[current == label]) for label in pd.unique(current)}
        assigned = {}
        candidates = []
        for label, members in groups.items():
            for track, old_members in prior_members.items():
                overlap = len(members & old_members)
                if overlap:
                    candidates.append((-overlap, str(label), track, label))
        used_tracks = set()
        for _, _, track, label in sorted(candidates):
            if label not in assigned and track not in used_tracks:
                assigned[label] = track
                used_tracks.add(track)
        for label in sorted(groups, key=str):
            if label not in assigned:
                assigned[label] = f"R{next_id:05d}"
                next_id += 1
        rows[date] = {asset: assigned[label] for asset, label in current.items()}
        prior_members = {assigned[label]: members for label, members in groups.items()}
    return pd.DataFrame.from_dict(rows, orient="index").sort_index(axis=1)


def stability_table(config: str, covar_data: RollingFactorCovarData) -> pd.DataFrame:
    """Run canonical lineage analysis and return one stability summary row."""
    report = analyze_cluster_lineage(covar_data)
    lineage_churn = annualized_churn(report.to_membership_panel())
    row = {
        "config": config,
        "snapshots": len(covar_data.dates),
        "median_clusters": float(
            np.median([covar_data[date].clusters.nunique() for date in covar_data.dates])
        ),
        "raw_churn": annualized_churn(greedy_raw_membership_panel(covar_data)),
        "lineage_churn": lineage_churn,
        "derived_tracks": report.relabel["derived_id"].nunique(),
        "median_beta_stability": float(report.classification["beta_stability"].median()),
        "op_pipeline_lineage_churn": OP_PIPELINE_CHURN[config],
        "lineage_churn_delta": lineage_churn - OP_PIPELINE_CHURN[config],
    }
    table = pd.DataFrame([row]).set_index("config")
    table.to_csv(_output_dir() / f"{config}_stability.csv")
    return table


def main() -> None:
    """Run both fits, write stability tables, and enforce reproduction tolerances."""
    workers = int(os.environ.get("FACTORLASSO_LINEAGE_WORKERS", "2"))
    tables = {
        config: stability_table(config, estimate_rolling(config, workers))
        for config in CONFIGS
    }
    combined = pd.concat(tables.values())
    combined.to_csv(_output_dir() / "sp500_cluster_lineage_stability.csv")
    baseline = float(combined.loc["baseline", "lineage_churn"])
    m1 = float(combined.loc["M1_delta_0.05", "lineage_churn"])
    reduction = 1.0 - m1 / baseline
    print(combined.to_string())
    print(f"M1 lineage-churn reduction: {reduction:.2%}")
    print(
        "Assembly difference: direct FactorLasso market-return, EWMA, annualisation, and "
        "CurrentFactorCovarData construction replace the consumer estimator wrappers."
    )
    assert abs(baseline / OP_PIPELINE_CHURN["baseline"] - 1.0) <= 0.02
    assert reduction >= 0.75
    assert not any(
        name == "optimalportfolios" or name.startswith("optimalportfolios.")
        for name in sys.modules
    )


if __name__ == "__main__":
    main()
