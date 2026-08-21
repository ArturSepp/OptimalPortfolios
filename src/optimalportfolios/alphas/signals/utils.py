"""
Shared utilities for cluster-based alpha signal scoring.

Provides the two pieces of cluster plumbing that every cluster-aware
signal needs:

    * :func:`extract_rolling_clusters` — extract time-varying cluster
      assignments from a ``RollingFactorCovarData``.
    * :func:`score_within_clusters` — apply cross-sectional scoring
      within those time-varying clusters.
    * :func:`align_rolling_clusters` — relabel those assignments so one
      cluster keeps one id through time, which is what makes them
      readable as a series rather than only as a cross-section.

Used by :mod:`momentum_cluster`, :mod:`low_beta_cluster`, and
:mod:`residual_momentum_cluster`.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import qis as qis
from scipy.optimize import linear_sum_assignment
from typing import TYPE_CHECKING, Dict, List, Mapping, Optional, Tuple, Union

if TYPE_CHECKING:
    from factorlasso import StabilityPoolingType


def resolve_span(span: Optional[Union[int, Mapping[str, int]]],
                 freq: str,
                 name: str = 'span',
                 ) -> Optional[int]:
    """resolve an EWMA span for one asset reporting cadence.

    A span is a number of PERIODS, so one scalar means different calendar time
    on each cadence: ``long_span=12`` is a year of monthly returns and three
    years of quarterly ones. Every signal estimates one frequency bucket at a
    time, so it can take a per-cadence mapping and give each bucket the same
    calendar horizon instead.

    Called at each site that has a scalar cadence in scope and is about to call
    a ``_compute_raw_*_single_freq``: inside the mixed-frequency bucket loop,
    and on the single-frequency branch of each entry point. Those raw functions
    therefore take a plain ``int`` and never see a mapping.

    A scalar remains valid and is applied unchanged at every cadence, so
    existing callers are bit-identical.

    Args:
        span: A scalar number of periods, a mapping of cadence code (``'ME'``,
            ``'QE'``, ...) to periods, or None where the span is optional.
        freq: The reporting cadence being estimated.
        name: Parameter name, quoted back in the error.

    Returns:
        The number of periods for ``freq``, or None when ``span`` is None.

    Raises:
        ValueError: If a mapping does not cover ``freq``, or the resolved value
            is not a positive integer. Falling back to another cadence's entry
            would silently reintroduce the scalar's behaviour.
    """
    if span is None:
        return None
    if isinstance(span, Mapping) and str(freq) not in span:
        raise ValueError(f"{name} covers {sorted(span)} but an asset reports at "
                         f"{str(freq)!r}; add the cadence rather than letting it inherit "
                         f"another cadence's horizon")
    if isinstance(span, Mapping):
        span = span[str(freq)]
    if isinstance(span, bool) or not isinstance(span, (int, np.integer)):
        raise ValueError(f"{name} must be an int number of periods, or a per-cadence "
                         f"mapping of them, got {span!r}")
    if span <= 0:
        raise ValueError(f"{name} must be > 0, got {span!r}")
    return int(span)


def extract_rolling_clusters(
        rolling_covar_data,
        assets: Optional[List[str]] = None,
) -> Dict[pd.Timestamp, pd.Series]:
    """Extract time-varying cluster assignments from RollingFactorCovarData.

    ``CurrentFactorCovarData.clusters`` is persisted as a single flat
    ``pd.Series`` keyed by asset ticker (with freq-prefixed cluster IDs
    such as ``'QE:4'`` as values), already merged across frequencies by
    ``FactorCovarEstimator``. This function just filters it to the
    requested asset universe and returns a dict keyed by estimation date.

    Args:
        rolling_covar_data: RollingFactorCovarData from FactorCovarEstimator.
        assets: Asset tickers to include. If None, includes all.

    Returns:
        Dict mapping estimation dates to pd.Series (ticker → cluster_id).
        Dates where clusters are None or empty are skipped.
    """
    rolling_clusters: Dict[pd.Timestamp, pd.Series] = {}
    for date, current_data in rolling_covar_data.data.items():
        clusters = current_data.clusters
        if clusters is None or len(clusters) == 0:
            continue
        # drop duplicate index entries (defensive: a ticker should only
        # carry one freq-tagged cluster label, but guard against merges
        # that might have kept both an ME and a QE label)
        clusters = clusters[~clusters.index.duplicated(keep='last')]
        if assets is not None:
            clusters = clusters.reindex(assets).dropna()
        if len(clusters) > 0:
            rolling_clusters[date] = clusters
    return rolling_clusters


def score_within_clusters(
        raw_signal: pd.DataFrame,
        rolling_clusters: Dict[pd.Timestamp, pd.Series],
        min_cluster_size: int = 3,
        stability_pooling_type: Optional['StabilityPoolingType'] = None,
        stability_weights: Optional[pd.DataFrame] = None,
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
        stability_pooling_type: Optional FactorLasso stability-pooling mode.
            ``None`` or ``NONE`` preserves the existing implementation exactly.
        stability_weights: Causal dates-by-assets co-cluster weights. Required
            only when stability pooling is enabled.

    Returns:
        Cross-sectional scores (T × N) scored within time-varying clusters.
    """
    if stability_pooling_type is not None:
        from factorlasso import StabilityPoolingType, score_with_stability_pooled_clusters

        stability_pooling_type = StabilityPoolingType(stability_pooling_type)
        if stability_pooling_type != StabilityPoolingType.NONE:
            return score_with_stability_pooled_clusters(
                raw_signal=raw_signal,
                rolling_clusters=rolling_clusters,
                stability_weights=stability_weights,
                min_cluster_size=min_cluster_size,
                pooling_type=stability_pooling_type,
            )

    if not rolling_clusters:
        # no clusters available: fall back to global scoring
        return qis.df_to_cross_sectional_score(df=raw_signal)

    cluster_dates = sorted(rolling_clusters.keys())
    first_cluster_date = cluster_dates[0]
    all_cols = raw_signal.columns.tolist()
    scores = []

    for date in raw_signal.index:
        row_values = raw_signal.loc[date, :]

        # rows before the first cluster estimation have no assignment yet: score
        # 0.0 and skip the lookup (calling find_upto on them returns None and
        # emits a warning per row, which floods backtest logs).
        if date < first_cluster_date:
            scores.append(pd.Series(0.0, index=all_cols, name=date))
            continue

        # find most recent cluster assignment
        try:
            cluster_date = qis.find_upto_date_from_datetime_index(
                index=cluster_dates, date=date)
        except Exception:
            cluster_date = None

        if cluster_date is None:
            # backstop: find_upto_date_from_datetime_index raises in some qis
            # builds and returns None in others; the date < first_cluster_date
            # guard above already covers this, but keep the None check too.
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
                # Not covered, and unreachable as written: `clusters_t` was already filtered to
                # `all_cols` above, and `row_values.index` is exactly `all_cols`, so every group
                # produced by the groupby has at least one member that survives this filter.
                continue  # pragma: no cover

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

# Cluster labels carry the estimation frequency as a prefix, e.g. 'QE:4'. A
# response belongs to exactly one frequency bucket and the estimator partitions
# each bucket separately, so alignment runs per prefix and the prefix is
# re-emitted unchanged.
CLUSTER_LABEL_SEPARATOR: str = ':'


def _split_cluster_label(label: str) -> Tuple[str, str]:
    """the frequency prefix and the raw id of a cluster label.

    Args:
        label: A cluster label, ``'<freq>:<id>'`` or a bare id.

    Returns:
        ``(prefix, raw_id)``; the prefix is the empty string for a bare id.
    """
    text = str(label)
    if CLUSTER_LABEL_SEPARATOR in text:
        prefix, raw = text.split(CLUSTER_LABEL_SEPARATOR, 1)
        return prefix, raw
    return '', text


def _align_one_partition(current: pd.Series,
                         previous: Optional[pd.Series],
                         next_id: int,
                         ) -> Tuple[pd.Series, int]:
    """relabel one date's partition to agree with the previous date's.

    Maximum-overlap assignment: the contingency matrix of current raw labels
    against previous aligned ids is solved with ``linear_sum_assignment``, so
    each current cluster inherits the id of the previous cluster it shares most
    members with, and no two clusters can claim the same id. A current cluster
    that matches nothing takes a fresh id.

    Args:
        current: Ticker to raw label, one frequency bucket, one date.
        previous: Ticker to aligned id at the previous date, same bucket.
            None for the first date.
        next_id: The next unused aligned id.

    Returns:
        ``(aligned, next_id)`` — ticker to aligned id, and the next unused id.
    """
    raw_labels = list(dict.fromkeys(current.tolist()))
    if previous is None or previous.empty:
        mapping = {label: next_id + n for n, label in enumerate(raw_labels)}
        return current.map(mapping), next_id + len(raw_labels)

    shared = current.index.intersection(previous.index)
    previous_ids = sorted(set(previous.loc[shared].tolist())) if len(shared) > 0 else []
    mapping: Dict[str, int] = {}
    if previous_ids:
        overlap = np.zeros((len(raw_labels), len(previous_ids)), dtype=float)
        for ticker in shared:
            row = raw_labels.index(current.loc[ticker])
            column = previous_ids.index(previous.loc[ticker])
            overlap[row, column] += 1.0
        rows, columns = linear_sum_assignment(-overlap)
        for row, column in zip(rows, columns):
            if overlap[row, column] > 0.0:
                mapping[raw_labels[row]] = previous_ids[column]
    for label in raw_labels:
        if label not in mapping:
            mapping[label] = next_id
            next_id += 1
    return current.map(mapping), next_id


def align_rolling_clusters(
        rolling_clusters: Dict[pd.Timestamp, pd.Series],
) -> Tuple[Dict[pd.Timestamp, pd.Series], pd.Series]:
    """give one cluster one id through time, so the labels can be read as a series.

    ``compute_clusters_from_corr_matrix`` returns ``scipy.cluster.hierarchy.
    fcluster`` output, whose numbering follows the dendrogram traversal and is
    re-derived independently at every estimation date. Cluster ``'QE:4'`` at one
    date and ``'QE:4'`` at the next are therefore unrelated, and a time series
    of the raw labels shows migrations that never happened. This walks the dates
    forward and relabels each partition to the one before it by maximum overlap,
    so a stable group keeps a stable id and a genuine migration shows as one.

    The output labels keep the ``'<freq>:<id>'`` shape, alignment running within
    each frequency prefix because the estimator partitions each frequency bucket
    separately.

    Args:
        rolling_clusters: Estimation date to (ticker -> raw label), from
            :func:`extract_rolling_clusters`.

    Returns:
        ``(aligned, n_reassigned)``. ``aligned`` mirrors the input with stable
        labels. ``n_reassigned`` is indexed by date and counts the tickers whose
        aligned id differs from their own previous one, among those present at
        both dates — zero when only the numbering moved, positive when
        membership actually changed. The first date is 0 by construction.

    Raises:
        ValueError: If any date's assignment is not a ``pd.Series``.
    """
    aligned: Dict[pd.Timestamp, pd.Series] = {}
    reassigned: Dict[pd.Timestamp, int] = {}
    previous_by_prefix: Dict[str, pd.Series] = {}
    next_id_by_prefix: Dict[str, int] = {}

    for date in sorted(rolling_clusters):
        clusters = rolling_clusters[date]
        if not isinstance(clusters, pd.Series):
            raise ValueError(f"cluster assignment at {date} must be a pd.Series, "
                             f"got {type(clusters).__name__}")
        split = {ticker: _split_cluster_label(label) for ticker, label in clusters.items()}
        prefixes = sorted({prefix for prefix, _ in split.values()})
        date_labels: Dict[str, str] = {}
        n_changed = 0
        for prefix in prefixes:
            tickers = [t for t, (p, _) in split.items() if p == prefix]
            bucket = pd.Series({t: split[t][1] for t in tickers})
            previous = previous_by_prefix.get(prefix)
            bucket_aligned, next_id = _align_one_partition(
                current=bucket,
                previous=previous,
                next_id=next_id_by_prefix.get(prefix, 1))
            next_id_by_prefix[prefix] = next_id
            if previous is not None and not previous.empty:
                shared = bucket_aligned.index.intersection(previous.index)
                n_changed += int((bucket_aligned.loc[shared] != previous.loc[shared]).sum())
            previous_by_prefix[prefix] = bucket_aligned
            for ticker, cluster_id in bucket_aligned.items():
                date_labels[ticker] = (f"{prefix}{CLUSTER_LABEL_SEPARATOR}{cluster_id}"
                                       if prefix else str(cluster_id))
        aligned[date] = pd.Series(date_labels).reindex(clusters.index)
        reassigned[date] = n_changed

    return aligned, pd.Series(reassigned, name='n_reassigned').sort_index()
