"""
Shared utilities for cluster-based alpha signal scoring.

Provides score_within_clusters() used by all cluster signal variants
(momentum_cluster, low_beta_cluster, residual_momentum_cluster).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import qis as qis
from typing import Dict


def score_within_clusters(
        raw_signal: pd.DataFrame,
        rolling_clusters: Dict[pd.Timestamp, pd.Series],
        min_cluster_size: int = 3,
) -> pd.DataFrame:
    """Apply cross-sectional scoring within time-varying clusters.

    For each row (date) in raw_signal, looks up the cluster assignment
    at the most recent estimation date and scores within each cluster
    independently.

    Clusters with fewer than ``min_cluster_size`` members are scored
    using the global (full cross-section) mean and standard deviation
    rather than within-cluster statistics. This avoids noisy z-scores
    from 2-3 observations while still giving these assets a meaningful
    score relative to the full universe.

    Dates before the first cluster estimation receive score 0.0.

    Args:
        raw_signal: Raw signal values (T × N).
        rolling_clusters: Dict mapping dates to cluster assignments
            (pd.Series with ticker index and cluster_id values).
        min_cluster_size: Minimum cluster size for within-cluster scoring.
            Clusters with size <= min_cluster_size are scored using
            global statistics. Default 3.

    Returns:
        Cross-sectional scores (T × N) scored within time-varying clusters.
    """
    if not rolling_clusters:
        # no clusters available: fall back to global scoring
        return qis.df_to_cross_sectional_score(df=raw_signal)

    cluster_dates = sorted(rolling_clusters.keys())
    all_cols = raw_signal.columns.tolist()
    scores = []

    for date in raw_signal.index:
        row_values = raw_signal.loc[date, :]

        # find most recent cluster assignment
        try:
            cluster_date = qis.find_upto_date_from_datetime_index(
                index=cluster_dates, date=date)
        except Exception:
            # date is before first cluster estimation
            scores.append(pd.Series(0.0, index=all_cols, name=date))
            continue

        clusters_t = rolling_clusters[cluster_date]

        # defensive: drop NaN cluster assignments, intersect with signal columns
        clusters_t = clusters_t.dropna()
        valid_cols = [c for c in clusters_t.index if c in all_cols]
        clusters_t = clusters_t.loc[valid_cols]

        if len(clusters_t) < 2 or clusters_t.nunique() < 2:
            # degenerate: all one cluster or too few assets → global scoring
            scored_row = _global_zscore(row_values, valid_cols)
            scores.append(scored_row.reindex(all_cols).fillna(0.0).rename(date))
            continue

        # compute global mean/std for fallback on small clusters
        global_values = row_values[valid_cols].dropna()
        if len(global_values) >= 2:
            global_mean = global_values.mean()
            global_std = global_values.std()
        else:
            global_mean = 0.0
            global_std = 1.0

        # score within each cluster
        scored_row = pd.Series(0.0, index=all_cols, name=date)

        for cluster_id, tickers in clusters_t.groupby(clusters_t).groups.items():
            cols = [c for c in tickers if c in row_values.index]
            if not cols:
                continue

            if len(cols) <= min_cluster_size:
                # small cluster: normalize using global statistics
                if global_std > 0:
                    scored_row[cols] = (row_values[cols] - global_mean) / global_std
                # else: leave as 0.0
            else:
                # large cluster: within-cluster z-score
                cluster_vals = row_values[cols].dropna()
                if len(cluster_vals) >= 2:
                    cluster_mean = cluster_vals.mean()
                    cluster_std = cluster_vals.std()
                    if cluster_std > 0:
                        scored_row[cols] = (row_values[cols] - cluster_mean) / cluster_std

        scores.append(scored_row)

    return pd.DataFrame(scores)


def _global_zscore(row_values: pd.Series, cols: list) -> pd.Series:
    """Z-score a subset of columns using their own mean/std."""
    if len(cols) < 2:
        return pd.Series(0.0, index=cols)
    vals = row_values[cols].dropna()
    if len(vals) < 2 or vals.std() == 0:
        return pd.Series(0.0, index=cols)
    return (row_values[cols] - vals.mean()) / vals.std()
