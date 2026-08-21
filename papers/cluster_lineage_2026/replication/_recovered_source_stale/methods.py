"""Causal deterministic partition generators for temporal cluster smoothing."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, Mapping, Tuple

import numpy as np
import pandas as pd
from factorlasso import (
    DependenceMeasure,
    DistanceTransform,
    compute_clusters_from_corr_matrix,
    compute_dependence_matrix,
    get_x_y_np,
)
from scipy.cluster import hierarchy as sch
from scipy.spatial.distance import squareform

from papers.cluster_lineage_2026.replication.sp500_baseline import (
    extract_partitions,
    get_estimation_dates,
    load_inputs,
    partition_equal,
)

PartitionBundle = Tuple[pd.Series, np.ndarray, float]


def compute_correlation_inputs() -> Dict[pd.Timestamp, pd.DataFrame]:
    """Replicate exactly the demeaned EWMA correlation clustered in the fit."""
    from rosaa import local_path as lp

    cache = Path(lp.get_output_path()) / "cluster_smoothing" / "correlation_inputs_v1.pkl"
    if cache.exists():
        with cache.open("rb") as file:
            return pickle.load(file)
    inputs = load_inputs()
    y_full = inputs["asset_returns_dict"]["W-WED"]
    prices = inputs["risk_factor_prices"]
    output = {}
    for date in get_estimation_dates():
        y = y_full.loc[:date]
        x_prices = prices.reindex(index=y.index, method="ffill").ffill()
        x = np.log(x_prices / x_prices.shift(1))
        _, y_np, valid_mask = get_x_y_np(x=x, y=y, span=156, demean=True)
        y_for_corr = np.where(valid_mask > 0, y_np, np.nan)
        corr = compute_dependence_matrix(
            a=y_for_corr,
            dependence_measure=DependenceMeasure.PEARSON,
            span=156,
        )
        frame = pd.DataFrame(corr, index=y.columns, columns=y.columns)
        frame.attrs["eligible"] = y.columns[y.notna().sum() >= 12].tolist()
        output[pd.Timestamp(date)] = frame
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("wb") as file:
        pickle.dump(output, file)
    return output


def baseline_partitions(
    correlations: Mapping[pd.Timestamp, pd.DataFrame],
) -> Dict[pd.Timestamp, PartitionBundle]:
    """Cluster every date with the unmodified factorlasso entry point."""
    output = {}
    for date, corr in correlations.items():
        clusters, linkage, cutoff = compute_clusters_from_corr_matrix(
            corr,
            cutoff_fraction=0.6,
            linkage_method="ward",
            distance_transform=DistanceTransform.ONE_MINUS_RHO,
        )
        output[date] = clusters.loc[corr.attrs["eligible"]], linkage, cutoff
    return output


def quarterly_hold(
    correlations: Mapping[pd.Timestamp, pd.DataFrame],
    baseline: Mapping[pd.Timestamp, PartitionBundle],
) -> Dict[pd.Timestamp, PartitionBundle]:
    """Recluster at quarter ends and causally hold memberships between recuts."""
    output: Dict[pd.Timestamp, PartitionBundle] = {}
    held: PartitionBundle | None = None
    for date in sorted(correlations):
        if held is None or pd.Timestamp(date).month in (3, 6, 9, 12):
            held = baseline[date]
        clusters, linkage, cutoff = held
        current_assets = pd.Index(correlations[date].attrs["eligible"])
        assigned = clusters.reindex(current_assets).copy()
        for asset in assigned[assigned.isna()].index:
            means = {
                label: correlations[date].loc[asset, members].mean()
                for label, members in clusters.groupby(clusters).groups.items()
            }
            assigned.loc[asset] = max(means, key=means.get)
        output[date] = (assigned, linkage, cutoff)
        held = output[date]
    return output


def delta_bonus_ward(
    correlations: Mapping[pd.Timestamp, pd.DataFrame], delta: float,
) -> Dict[pd.Timestamp, PartitionBundle]:
    """Apply a causal distance bonus to pairs co-clustered at the prior date."""
    if delta < 0:
        raise ValueError("delta must be non-negative")
    output: Dict[pd.Timestamp, PartitionBundle] = {}
    prior: pd.Series | None = None
    for date in sorted(correlations):
        corr = correlations[date]
        if prior is None or delta == 0:
            bundle = compute_clusters_from_corr_matrix(
                corr, cutoff_fraction=0.6, linkage_method="ward",
                distance_transform=DistanceTransform.ONE_MINUS_RHO,
            )
        else:
            distance = np.clip(1.0 - corr.fillna(0.0).to_numpy(), 0.0, 2.0)
            np.fill_diagonal(distance, 0.0)
            labels = prior.reindex(corr.index)
            same = labels.notna().to_numpy()[:, None] & labels.notna().to_numpy()[None, :]
            same &= labels.to_numpy()[:, None] == labels.to_numpy()[None, :]
            distance[same] = np.maximum(distance[same] - delta, 0.0)
            condensed = squareform(distance, checks=False)
            linkage = sch.linkage(condensed, method="ward")
            cutoff = float(0.6 * condensed.max())
            clusters = pd.Series(sch.fcluster(linkage, cutoff, criterion="distance"),
                                 index=corr.index)
            bundle = clusters, linkage, cutoff
        eligible = correlations[date].attrs["eligible"]
        output[date] = (bundle[0].loc[eligible], bundle[1], bundle[2])
        prior = output[date][0]
    return output


def similarity_smoothing(
    correlations: Mapping[pd.Timestamp, pd.DataFrame], smoothing: float,
) -> Dict[pd.Timestamp, PartitionBundle]:
    """Causally smooth correlation matrices before unchanged Ward clustering."""
    if not 0.0 <= smoothing < 1.0:
        raise ValueError("smoothing must lie in [0, 1)")
    output: Dict[pd.Timestamp, PartitionBundle] = {}
    state: pd.DataFrame | None = None
    for date in sorted(correlations):
        corr = correlations[date]
        if state is None:
            state = corr.copy()
        else:
            prior = state.reindex(index=corr.index, columns=corr.columns)
            state = (1.0 - smoothing) * corr + smoothing * prior.fillna(corr)
            values = state.to_numpy(copy=True)
            np.fill_diagonal(values, 1.0)
            state = pd.DataFrame(values, index=corr.index, columns=corr.columns)
        clusters, linkage, cutoff = compute_clusters_from_corr_matrix(
            state, cutoff_fraction=0.6, linkage_method="ward",
            distance_transform=DistanceTransform.ONE_MINUS_RHO,
        )
        output[date] = clusters.loc[corr.attrs["eligible"]], linkage, cutoff
    return output


def coassociation_confidence(
    partitions: Mapping[pd.Timestamp, pd.Series], window: int = 6,
) -> pd.DataFrame:
    """Return trailing co-cluster confidence for every current asset assignment."""
    dates = sorted(partitions)
    rows = {}
    for i, date in enumerate(dates):
        current = partitions[date].dropna()
        history = dates[max(0, i - window + 1):i + 1]
        values = {}
        for asset, label in current.items():
            peers = current.index[(current == label) & (current.index != asset)]
            if len(peers) == 0:
                values[asset] = 1.0
                continue
            observations = []
            for prior_date in history:
                prior = partitions[prior_date]
                for peer in peers:
                    if asset in prior.index and peer in prior.index:
                        if pd.notna(prior[asset]) and pd.notna(prior[peer]):
                            observations.append(prior[asset] == prior[peer])
            values[asset] = float(np.mean(observations)) if observations else np.nan
        rows[date] = values
    return pd.DataFrame.from_dict(rows, orient="index").sort_index(axis=1)


def run_checks() -> None:
    """Run exact degenerate-case and quarter-end assertions."""
    from papers.cluster_lineage_2026.replication.sp500_baseline import estimate_rolling

    correlations = compute_correlation_inputs()
    baseline = baseline_partitions(correlations)
    fitted = extract_partitions(estimate_rolling())
    for date in baseline:
        assert partition_equal(baseline[date][0], fitted[date])
    delta_zero = delta_bonus_ward(correlations, 0.0)
    smooth_zero = similarity_smoothing(correlations, 0.0)
    held = quarterly_hold(correlations, baseline)
    for date in baseline:
        assert partition_equal(delta_zero[date][0], baseline[date][0])
        assert partition_equal(smooth_zero[date][0], baseline[date][0])
        if pd.Timestamp(date).month in (3, 6, 9, 12):
            assert partition_equal(held[date][0], baseline[date][0])
    print("all assertions pass: exact baseline replication, degenerate grids, and QE control")
    print("causality: dates are processed in ascending order and only current/prior state is read")


if __name__ == "__main__":
    run_checks()
