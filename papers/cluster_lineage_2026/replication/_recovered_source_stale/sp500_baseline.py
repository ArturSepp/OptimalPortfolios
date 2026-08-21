"""S&P 500 baseline harness and shared metrics for the cluster-smoothing study.

All estimation is expanding-window and point-in-time.  Cached snapshots live under the
configured ROSAA output directory, never in the repository.
"""
from __future__ import annotations

import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

import numpy as np
import pandas as pd
import qis as qis
from factorlasso import (
    ClusterSmootherType,
    DependenceMeasure,
    DistanceTransform,
    LassoModel,
    LassoModelType,
    RollingFactorCovarData,
    diagnose_residuals,
)
from optimalportfolios.covar_estimation import risk_labelling as rl
from optimalportfolios.covar_estimation.factor_covar_estimator import FactorCovarEstimator

RUNNER = "rosaa/research/cluster_smoothing/sp500_baseline.py"
RETURNS_FILE = "sp500_adjusted_close_2005_to_current.csv"
CONSTITUENTS_FILE = "sp500_current_constituents.csv"
CACHE_DIR = "cluster_smoothing/sp500_baseline"
TIME_PERIOD = qis.TimePeriod("01Aug2021", "01Aug2026")
COHORT_RETENTION_CUT = 0.60
GICS_COLUMNS = ("gics_sector", "gics_industry_group", "gics_industry")


def get_estimation_dates() -> pd.DatetimeIndex:
    """Return the fixed 60-date month-end evaluation schedule."""
    return qis.generate_dates_schedule(
        time_period=TIME_PERIOD,
        freq="ME",
        include_start_date=False,
        include_end_date=False,
    )


def _smoother_kwargs(config: str) -> Dict[str, Any]:
    """Map stable research labels to declarative LassoModel smoothing fields."""
    configs = {
        "baseline": {"cluster_smoother_type": ClusterSmootherType.NONE},
        "M0_quarterly_hold": {
            "cluster_smoother_type": ClusterSmootherType.HOLD,
            "recluster_freq": "QE",
        },
        "M1_delta_0.05": {
            "cluster_smoother_type": ClusterSmootherType.PARTITION_BONUS,
            "smoother_delta": 0.05,
        },
        "M2_lambda_0.7": {
            "cluster_smoother_type": ClusterSmootherType.SIMILARITY_EWMA,
            "smoother_lambda": 0.7,
        },
    }
    if config not in configs:
        raise ValueError(f"unknown Tier-2 smoothing config {config!r}")
    return configs[config]


def make_estimator(config: str = "baseline") -> FactorCovarEstimator:
    """Build one roadmap estimator using only declarative LassoModel fields."""
    lasso_model = LassoModel(
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
        **_smoother_kwargs(config),
    )
    return FactorCovarEstimator(
        rebalancing_freq="ME",
        lasso_model=lasso_model,
        factor_returns_freq="W-WED",
        factor_covar_span=52,
        is_apply_vol_normalised_returns=False,
        demean=True,
    )


def load_inputs() -> Dict[str, Any]:
    """Load and transform the fixed static S&P 500 inputs."""
    from rosaa import local_path as lp

    root = Path(lp.get_output_path()) / "factorlasso_returns"
    prices = pd.read_csv(root / RETURNS_FILE, index_col=0, parse_dates=True).sort_index()
    prices = prices.loc["2017-01-01":].asfreq("B", method="ffill").resample("W-WED").last()
    asset_returns = np.log(prices / prices.shift(1))
    simple_returns = prices.pct_change(fill_method=None)
    market_returns = simple_returns.mean(axis=1, skipna=True).rename("Market")
    market_prices = qis.returns_to_nav(market_returns.to_frame())
    market_prices.columns = ["Market"]

    constituents = pd.read_csv(root / CONSTITUENTS_FILE)
    if "yahoo_symbol" in constituents.columns:
        constituents = constituents.set_index("yahoo_symbol")
    elif constituents.index.name != "yahoo_symbol":
        raise ValueError("constituent metadata must contain yahoo_symbol")
    constituents = constituents.reindex(prices.columns)
    return {
        "prices": prices,
        "asset_returns_dict": {"W-WED": asset_returns},
        "risk_factor_prices": market_prices,
        "metadata": constituents,
    }


def _cache_root(config: str = "baseline") -> Path:
    """Return and create the external snapshot-cache directory for a configuration."""
    from rosaa import local_path as lp

    path = Path(lp.get_output_path()) / CACHE_DIR / config
    path.mkdir(parents=True, exist_ok=True)
    return path


def _snapshot_path(date: pd.Timestamp, config: str = "baseline") -> Path:
    """Return the deterministic cache path for one date and configuration."""
    return _cache_root(config) / f"{pd.Timestamp(date):%Y%m%d}.pkl"


def _fit_one(
    date: pd.Timestamp,
    config: str = "baseline",
    injection: Optional[Tuple[pd.Series, np.ndarray, float]] = None,
) -> str:
    """Fit and cache one expanding-window snapshot; return its path."""
    date = pd.Timestamp(date)
    path = _snapshot_path(date, config)
    if path.exists():
        return str(path)
    inputs = load_inputs()
    estimator = make_estimator(config)
    kwargs: Dict[str, Any] = {}
    if injection is not None:
        clusters, linkage, cutoff = injection
        kwargs = {
            "precomputed_clusters": {"W-WED": clusters},
            "precomputed_linkages": {"W-WED": linkage},
            "precomputed_cutoffs": {"W-WED": cutoff},
        }
    snapshot = estimator.fit_current_factor_covars(
        risk_factor_prices=inputs["risk_factor_prices"].loc[:date],
        asset_returns_dict={
            freq: returns.loc[:date]
            for freq, returns in inputs["asset_returns_dict"].items()
        },
        estimation_date=date,
        **kwargs,
    )
    with path.open("wb") as file:
        pickle.dump(snapshot, file)
    return str(path)


def estimate_rolling(
    config: str = "baseline",
    injections: Optional[Mapping[pd.Timestamp, Tuple[pd.Series, np.ndarray, float]]] = None,
    max_workers: int = 2,
) -> RollingFactorCovarData:
    """Fit missing dates in parallel and return all cached snapshots."""
    dates = get_estimation_dates()
    jobs = []
    for date in dates:
        injection = None if injections is None else injections[pd.Timestamp(date)]
        if not _snapshot_path(date, config).exists():
            jobs.append((pd.Timestamp(date), config, injection))
    if jobs:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = [executor.submit(_fit_one, *job) for job in jobs]
            for count, future in enumerate(as_completed(futures), start=1):
                print(f"cached {count}/{len(futures)}: {Path(future.result()).name}")
    data = {}
    for date in dates:
        with _snapshot_path(date, config).open("rb") as file:
            data[pd.Timestamp(date)] = pickle.load(file)
    return RollingFactorCovarData(data=data)


def extract_partitions(covar_data: RollingFactorCovarData) -> Dict[pd.Timestamp, pd.Series]:
    """Extract within-date partitions from fitted snapshots."""
    partitions = {}
    for date in covar_data.dates:
        clusters = covar_data[date].clusters.copy()
        clusters = clusters.astype(str).str.replace(r"^W-WED:", "", regex=True)
        partitions[pd.Timestamp(date)] = clusters
    return partitions


def partition_equal(left: pd.Series, right: pd.Series) -> bool:
    """Return whether two label vectors induce exactly the same partition."""
    common = left.dropna().index.intersection(right.dropna().index)
    if len(common) != left.notna().sum() or len(common) != right.notna().sum():
        return False
    a = left.loc[common].to_numpy()
    b = right.loc[common].to_numpy()
    return bool(np.array_equal(a[:, None] == a[None, :], b[:, None] == b[None, :]))


def greedy_membership_panel(
    partitions: Mapping[pd.Timestamp, pd.Series],
) -> pd.DataFrame:
    """Greedily map consecutive raw clusters by maximum member overlap."""
    dates = sorted(partitions)
    rows: Dict[pd.Timestamp, Dict[str, str]] = {}
    next_id = 0
    prior_members: Dict[str, set] = {}
    for date in dates:
        current = partitions[date].dropna()
        groups = {label: set(current.index[current == label]) for label in pd.unique(current)}
        assigned: Dict[Any, str] = {}
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


def annualized_churn(panel: pd.DataFrame) -> float:
    """Compute derived-id changes per asset-year over consecutive valid pairs."""
    left, right = panel.iloc[:-1], panel.iloc[1:]
    left.index = right.index
    valid = left.notna() & right.notna()
    changes = ((left != right) & valid).to_numpy().sum()
    pairs = valid.to_numpy().sum()
    years = (panel.index[-1] - panel.index[0]).days / 365.25
    return float(changes * (len(panel) - 1) / (pairs * years))


def adjusted_rand_index(labels_a: pd.Series, labels_b: pd.Series) -> float:
    """Compute adjusted Rand index without adding a scikit-learn dependency."""
    frame = pd.concat([labels_a, labels_b], axis=1).dropna()
    contingency = pd.crosstab(frame.iloc[:, 0], frame.iloc[:, 1]).to_numpy(dtype=float)
    choose2 = lambda values: np.sum(values * (values - 1.0) / 2.0)
    n = contingency.sum()
    if n < 2:
        return np.nan
    observed = choose2(contingency)
    rows = choose2(contingency.sum(axis=1))
    cols = choose2(contingency.sum(axis=0))
    total = n * (n - 1.0) / 2.0
    expected = rows * cols / total
    maximum = 0.5 * (rows + cols)
    return float((observed - expected) / (maximum - expected)) if maximum != expected else 1.0


def ari_metrics(
    partitions: Mapping[pd.Timestamp, pd.Series], metadata: pd.DataFrame,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Return median and per-date ARI against the three GICS levels."""
    per_date = pd.DataFrame(index=pd.DatetimeIndex(sorted(partitions)))
    for column in GICS_COLUMNS:
        per_date[f"ari_{column.removeprefix('gics_')}"] = [
            adjusted_rand_index(partitions[date], metadata[column]) for date in per_date.index
        ]
    medians = {column: float(per_date[column].median()) for column in per_date}
    return medians, per_date


def signal_rank_metrics(
    partitions: Mapping[pd.Timestamp, pd.Series], weekly_returns: pd.DataFrame,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Measure month-to-month cluster-relative momentum-rank stability."""
    dates = sorted(partitions)
    ranks: Dict[pd.Timestamp, pd.Series] = {}
    prior_partition_ranks: Dict[pd.Timestamp, pd.Series] = {}
    for i, date in enumerate(dates):
        window = weekly_returns.loc[date - pd.Timedelta(weeks=52):
                                    date - pd.Timedelta(weeks=4)]
        score = window.sum(min_count=1)
        current = partitions[date].reindex(score.index)
        ranks[date] = score.groupby(current).rank(pct=True)
        if i:
            prior = partitions[dates[i - 1]].reindex(score.index)
            prior_partition_ranks[date] = score.groupby(prior).rank(pct=True)
    rows = []
    for i in range(1, len(dates)):
        date, prior_date = dates[i], dates[i - 1]
        current_pair = pd.concat([ranks[prior_date], ranks[date]], axis=1).dropna()
        held_pair = pd.concat([ranks[prior_date], prior_partition_ranks[date]], axis=1).dropna()
        rows.append({
            "date": date,
            "rank_spearman": current_pair.iloc[:, 0].corr(current_pair.iloc[:, 1], method="spearman"),
            "rank_mad": (current_pair.iloc[:, 0] - current_pair.iloc[:, 1]).abs().mean(),
            "held_rank_spearman": held_pair.iloc[:, 0].corr(held_pair.iloc[:, 1], method="spearman"),
            "held_rank_mad": (held_pair.iloc[:, 0] - held_pair.iloc[:, 1]).abs().mean(),
        })
    per_date = pd.DataFrame(rows).set_index("date")
    metrics = {column: float(per_date[column].mean()) for column in per_date}
    metrics["reassignment_rank_mad_gap"] = metrics["rank_mad"] - metrics["held_rank_mad"]
    return metrics, per_date


def _lineage_pair_metrics(report: rl.RiskClusterReport) -> Dict[str, float]:
    """Compute sweep-style lineage churn, matcher churn, and link overlap."""
    partitions = {}
    cluster_members = {}
    for date, frame in report.relabel.groupby("date"):
        raw_to_id = frame.set_index("raw_label")["derived_id"].to_dict()
        members = {raw: set(report.tracks[did].members[pd.Timestamp(date)])
                   for raw, did in raw_to_id.items()}
        partitions[pd.Timestamp(date)] = {
            asset: (raw, raw_to_id[raw]) for raw, assets in members.items() for asset in assets
        }
        cluster_members[pd.Timestamp(date)] = members
    dates = sorted(partitions)
    changes = matcher_changes = pairs = 0
    overlaps = []
    for d0, d1 in zip(dates[:-1], dates[1:]):
        for asset, (c1, did1) in partitions[d1].items():
            if asset not in partitions[d0]:
                continue
            pairs += 1
            c0, did0 = partitions[d0][asset]
            if did0 != did1:
                changes += 1
                cohort = cluster_members[d0][c0]
                if len(cohort & cluster_members[d1][c1]) / len(cohort) >= COHORT_RETENTION_CUT:
                    matcher_changes += 1
        prior_by_id = {did: raw for raw, did in
                       report.relabel.loc[report.relabel.date == d0,
                                          ["raw_label", "derived_id"]].itertuples(index=False)}
        for raw, did in report.relabel.loc[report.relabel.date == d1,
                                           ["raw_label", "derived_id"]].itertuples(index=False):
            if did in prior_by_id:
                old = cluster_members[d0][prior_by_id[did]]
                new = cluster_members[d1][raw]
                overlaps.append(len(old & new) / min(len(old), len(new)))
    years = (dates[-1] - dates[0]).days / 365.25
    scale = (len(dates) - 1) / (pairs * years)
    return {
        "lineage_churn_pair_count": float(changes * scale),
        "matcher_attributable_churn": float(matcher_changes * scale),
        "mean_link_overlap": float(np.mean(overlaps)),
    }


def lineage_metrics(covar_data: RollingFactorCovarData) -> Tuple[Dict[str, float], Any]:
    """Run the default lineage matcher and return required summary metrics."""
    report = rl.analyze_risk_clusters(covar_data)
    panel_churn = annualized_churn(report.to_membership_panel())
    metrics = _lineage_pair_metrics(report)
    metrics.update({
        "lineage_churn_panel": panel_churn,
        "n_derived_tracks": float(len(report.tracks)),
        "median_beta_stability": float(report.classification["beta_stability"].median()),
    })
    assert round(metrics["lineage_churn_pair_count"], 4) == round(panel_churn, 4)
    return metrics, report


def residual_diagonality(covar_data: RollingFactorCovarData) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Summarise factorlasso residual-diagonality diagnostics across dates."""
    rows = []
    for date in covar_data.dates:
        snapshot = covar_data[date]
        residuals = snapshot.residuals.loc[:, snapshot.residuals.notna().sum() >= 30]
        fitted = float((snapshot.y_betas.loc[residuals.columns].abs() > 1e-12)
                       .sum(axis=1).mean())
        diagnostics = diagnose_residuals(residuals, n_fitted_per_asset=fitted)
        rows.append({"date": date, **diagnostics.to_dict()})
    panel = pd.DataFrame(rows).set_index("date")
    numeric = panel.select_dtypes(include=[np.number]).mean().to_dict()
    return {f"diagonality_{key}": float(value) for key, value in numeric.items()}, panel


def evaluate_baseline(covar_data: RollingFactorCovarData) -> Dict[str, Any]:
    """Evaluate the full metric suite and assert the fixed baseline tolerances."""
    inputs = load_inputs()
    partitions = extract_partitions(covar_data)
    counts = pd.Series({date: part.nunique() for date, part in partitions.items()})
    raw_churn = annualized_churn(greedy_membership_panel(partitions))
    ari, ari_panel = ari_metrics(partitions, inputs["metadata"])
    signal, signal_panel = signal_rank_metrics(
        partitions, inputs["asset_returns_dict"]["W-WED"]
    )
    lineage, report = lineage_metrics(covar_data)
    diagonality, diagonal_panel = residual_diagonality(covar_data)
    last = covar_data[covar_data.dates[-1]]
    headline = {
        "n_snapshots": len(covar_data),
        "median_clusters": float(counts.median()),
        "min_clusters": float(counts.min()),
        "max_clusters": float(counts.max()),
        "raw_churn": raw_churn,
        **lineage,
        **ari,
        **signal,
        **diagonality,
        "mean_market_beta_last": float(last.y_betas["Market"].mean()),
        "median_r2_last": float(last.y_variances["r2"].median()),
        "market_variance_last": float(last.x_covar.loc["Market", "Market"]),
    }
    assert headline["n_snapshots"] == 60
    assert abs(headline["median_clusters"] - 72) <= 3
    assert abs(headline["lineage_churn_panel"] - 3.21) <= 0.05
    assert abs(headline["n_derived_tracks"] - 216) <= 10
    assert abs(headline["matcher_attributable_churn"] - 0.49) <= 0.02
    assert abs(headline["mean_link_overlap"] - 0.938) <= 0.005
    assert abs(headline["ari_sector"] - 0.20) <= 0.02
    assert abs(headline["ari_industry_group"] - 0.30) <= 0.02
    assert abs(headline["ari_industry"] - 0.33) <= 0.02
    assert abs(headline["mean_market_beta_last"] - 0.99) <= 0.02
    assert abs(headline["median_r2_last"] - 0.17) <= 0.02
    assert abs(headline["market_variance_last"] - 0.0093) <= 0.0005
    return {
        "headline": pd.Series(headline, name="value").to_frame(),
        "ari_per_date": ari_panel,
        "signal_per_date": signal_panel,
        "diagonality_per_date": diagonal_panel,
        "membership_panel": report.to_membership_panel(),
    }


class ResearchWorkflow(Enum):
    """Runnable baseline workflow stages."""

    ESTIMATE = 1
    EVALUATE = 2
    ALL = 3


@qis.timer
def run_research(workflow: ResearchWorkflow = ResearchWorkflow.ALL) -> None:
    """Run baseline estimation and/or evaluation."""
    if workflow in (ResearchWorkflow.ESTIMATE, ResearchWorkflow.ALL):
        estimate_rolling()
    if workflow in (ResearchWorkflow.EVALUATE, ResearchWorkflow.ALL):
        tables = evaluate_baseline(estimate_rolling())
        print(tables["headline"].to_string())
        print("second pass: pair-count and membership-panel churn agree to 4 decimals")


if __name__ == "__main__":
    run_research(ResearchWorkflow.ALL)
