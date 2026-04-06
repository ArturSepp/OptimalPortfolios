from __future__ import annotations

import pandas as pd
import qis as qis
from typing import Dict


def score_within_clusters(
        raw_signal: pd.DataFrame,
        rolling_clusters: Dict[pd.Timestamp, pd.Series],
) -> pd.DataFrame:
    """Apply cross-sectional scoring within time-varying clusters.

    For each row (date) in raw_signal, looks up the cluster assignment
    at the most recent estimation date and scores within each cluster
    independently.

    Assets in singleton clusters (only 1 member) receive score 0.0.
    Dates before the first cluster estimation receive score 0.0.

    Args:
        raw_signal: EWMA-smoothed residual momentum (T × N).
        rolling_clusters: Dict mapping dates to cluster assignments.

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
        row = raw_signal.loc[[date], :]

        # find most recent cluster assignment
        try:
            cluster_date = qis.find_upto_date_from_datetime_index(
                index=cluster_dates, date=date)
        except Exception:
            # date is before first cluster estimation
            scores.append(pd.DataFrame(0.0, index=[date], columns=all_cols))
            continue

        clusters_t = rolling_clusters[cluster_date]

        # defensive checks
        clusters_t = clusters_t.dropna()
        valid_cols = [c for c in clusters_t.index if c in all_cols]
        clusters_t = clusters_t.loc[valid_cols]

        if len(clusters_t) < 2 or clusters_t.nunique() < 2:
            # degenerate: all one cluster or too few assets → global scoring
            if len(valid_cols) >= 2:
                scored_row = qis.df_to_cross_sectional_score(df=row[valid_cols])
            else:
                scored_row = pd.DataFrame(0.0, index=[date], columns=valid_cols)
            scores.append(scored_row.reindex(columns=all_cols).fillna(0.0))
            continue

        # score within each cluster
        group_scores = []
        for cluster_id, tickers in clusters_t.groupby(clusters_t).groups.items():
            cols = [c for c in tickers if c in row.columns]
            if len(cols) >= 2:
                group_scores.append(
                    qis.df_to_cross_sectional_score(df=row[cols]))
            # singleton clusters: score 0 (omitted → filled with 0 below)

        if group_scores:
            scored_row = pd.concat(group_scores, axis=1).reindex(columns=all_cols).fillna(0.0)
        else:
            scored_row = pd.DataFrame(0.0, index=[date], columns=all_cols)
        scores.append(scored_row)

    return pd.concat(scores)
