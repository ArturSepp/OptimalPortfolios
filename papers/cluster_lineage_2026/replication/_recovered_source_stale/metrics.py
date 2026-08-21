"""Frozen empirical metrics for the cluster-lineage paper.

Raw and lineage churn are annualised asset-transition counts. Consecutive partitions are
compared on their common clustered assets by adjusted Rand index (ARI) and
``VI(A, B) = H(A | B) + H(B | A)`` using natural logarithms. Shape diagnostics report the
cluster-count distribution, singleton and largest-cluster asset shares, and
``-sum_j p_j log(p_j)`` size entropy.

The assignment margin is the average distance to the nearest other cluster minus average
distance to the asset's own cluster. It is an average-linkage proxy for the dendrogram cut
margin, not the Ward merge cost. Its noise scale is
``sigma_d = sqrt(2 * (1 - lambda**k)) * (1 - rho_hat**2) / sqrt(span)`` and the Gaussian
switch probability is ``Phi(-(margin + delta) / (sqrt(2) * sigma_d))``.

Fidelity requires median same-date ARI to baseline to be reported, absolute median taxonomy
ARI changes no greater than 0.03 at every supplied level, and median cluster count within
15 percent of baseline. Covariance invariance uses the common asset block and reports
``||Sigma-Sigma0||_F/||Sigma0||_F``, maximum absolute entry change divided by the maximum
absolute baseline entry, and absolute equal-weight ex-ante volatility change.

Turnover is decomposed with counterfactual weights built from current scores under the prior
partition: reassignment is ``0.5 sum|w_t-w_tilde_t|`` and signal is
``0.5 sum|w_tilde_t-w_drift,t-1|``. This is a bound-oriented decomposition rather than an
identity; the signed residual to observed total turnover is always reported.

Lineage and label metrics are offline full-panel diagnostics and must never feed a score or
portfolio weight. All functions are deterministic and contain no random state.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
from factorlasso import RiskClusterReport, RollingFactorCovarData, analyze_cluster_lineage
from factorlasso import diagnose_residuals
from scipy.special import ndtr

COHORT_RETENTION_CUT = 0.60
GICS_COLUMNS = ("gics_sector", "gics_industry_group", "gics_industry")


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
    """Return derived-id changes per asset-year over consecutive valid pairs."""
    if len(panel.index) < 2:
        return 0.0
    left, right = panel.iloc[:-1].copy(), panel.iloc[1:].copy()
    left.index = right.index
    valid = left.notna() & right.notna()
    changes = ((left != right) & valid).to_numpy().sum()
    pairs = valid.to_numpy().sum()
    years = (panel.index[-1] - panel.index[0]).days / 365.25
    if pairs == 0 or years <= 0:
        return 0.0
    return float(changes * (len(panel) - 1) / (pairs * years))


def adjusted_rand_index(labels_a: pd.Series, labels_b: pd.Series) -> float:
    """Return adjusted Rand index on assets with two non-null labels."""
    frame = pd.concat([labels_a, labels_b], axis=1).dropna()
    contingency = pd.crosstab(frame.iloc[:, 0], frame.iloc[:, 1]).to_numpy(dtype=float)
    n = contingency.sum()
    if n < 2:
        return np.nan
    observed = np.sum(contingency * (contingency - 1.0) / 2.0)
    row_counts = contingency.sum(axis=1)
    column_counts = contingency.sum(axis=0)
    rows = np.sum(row_counts * (row_counts - 1.0) / 2.0)
    cols = np.sum(column_counts * (column_counts - 1.0) / 2.0)
    total = n * (n - 1.0) / 2.0
    expected = rows * cols / total
    maximum = 0.5 * (rows + cols)
    return float((observed - expected) / (maximum - expected)) if maximum != expected else 1.0


def variation_of_information(labels_a: pd.Series, labels_b: pd.Series) -> float:
    """Return ``H(A|B)+H(B|A)`` in nats on common non-null assets."""
    frame = pd.concat([labels_a, labels_b], axis=1).dropna()
    if frame.empty:
        return np.nan
    counts = pd.crosstab(frame.iloc[:, 0], frame.iloc[:, 1]).to_numpy(dtype=float)
    joint = counts / counts.sum()
    pa = joint.sum(axis=1, keepdims=True)
    pb = joint.sum(axis=0, keepdims=True)
    nonzero = joint > 0
    mutual_information = float(
        np.sum(joint[nonzero] * np.log((joint / (pa @ pb))[nonzero]))
    )
    entropy_a = float(-np.sum(pa[pa > 0] * np.log(pa[pa > 0])))
    entropy_b = float(-np.sum(pb[pb > 0] * np.log(pb[pb > 0])))
    return max(entropy_a + entropy_b - 2.0 * mutual_information, 0.0)


def consecutive_partition_metrics(
    partitions: Mapping[pd.Timestamp, pd.Series],
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Return per-transition ARI/VI and their median and interquartile range."""
    dates = sorted(partitions)
    rows = [
        {
            "date": right,
            "ari": adjusted_rand_index(partitions[left], partitions[right]),
            "vi": variation_of_information(partitions[left], partitions[right]),
        }
        for left, right in zip(dates[:-1], dates[1:])
    ]
    panel = pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame(columns=["ari", "vi"])
    summary: Dict[str, float] = {}
    for column in ("ari", "vi"):
        values = panel[column]
        summary[f"consecutive_{column}_median"] = float(values.median())
        summary[f"consecutive_{column}_iqr"] = float(values.quantile(0.75) - values.quantile(0.25))
    return summary, panel


def ari_metrics(
    partitions: Mapping[pd.Timestamp, pd.Series],
    metadata: pd.DataFrame,
    taxonomy_columns: Sequence[str] = GICS_COLUMNS,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Return median and per-date ARI against each supplied taxonomy level."""
    per_date = pd.DataFrame(index=pd.DatetimeIndex(sorted(partitions)))
    for column in taxonomy_columns:
        key = f"ari_{column.removeprefix('gics_').replace(' ', '_').lower()}"
        per_date[key] = [
            adjusted_rand_index(partitions[date], metadata[column]) for date in per_date.index
        ]
    medians = {column: float(per_date[column].median()) for column in per_date}
    return medians, per_date


def size_shape_metrics(
    partitions: Mapping[pd.Timestamp, pd.Series],
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Return cluster-count, size, singleton, largest-share, and entropy diagnostics."""
    rows = []
    for date in sorted(partitions):
        labels = partitions[date].dropna()
        sizes = labels.value_counts()
        shares = sizes / sizes.sum()
        rows.append({
            "date": date,
            "cluster_count": float(len(sizes)),
            "median_cluster_size": float(sizes.median()),
            "singleton_share": float(sizes.loc[sizes == 1].sum() / sizes.sum()),
            "largest_cluster_share": float(shares.max()),
            "largest_cluster_gt_025": float(shares.max() > 0.25),
            "size_entropy": float(-(shares * np.log(shares)).sum()),
        })
    panel = pd.DataFrame(rows).set_index("date")
    summary = {
        "cluster_count_median": float(panel["cluster_count"].median()),
        "cluster_count_min": float(panel["cluster_count"].min()),
        "cluster_count_max": float(panel["cluster_count"].max()),
        "median_cluster_size": float(panel["median_cluster_size"].median()),
        "singleton_share": float(panel["singleton_share"].mean()),
        "largest_cluster_share": float(panel["largest_cluster_share"].mean()),
        "largest_cluster_gt_025_share": float(panel["largest_cluster_gt_025"].mean()),
        "size_entropy": float(panel["size_entropy"].mean()),
    }
    return summary, panel


def coassociation_metrics(confidence_panel: pd.DataFrame) -> Dict[str, float]:
    """Return the pooled cross-sectional median confidence and share below 0.5."""
    values = confidence_panel.stack(future_stack=True).dropna()
    return {
        "coassociation_median": float(values.median()),
        "coassociation_share_lt_05": float((values < 0.5).mean()),
    }


def assignment_margins(
    partitions: Mapping[pd.Timestamp, pd.Series],
    distance_matrices: Mapping[pd.Timestamp, pd.DataFrame],
    *,
    span: int,
    decay: float,
    step_k: float = 1.0,
    delta: float = 0.0,
) -> pd.DataFrame:
    """Return average-distance cut-margin proxies, noise scales, and switch probabilities.

    For each asset, the margin is its mean distance to the nearest other cluster minus its
    mean distance to its own peers. Singleton own-cluster distance is zero. ``rho_hat`` is
    one minus mean own-cluster distance. The Gaussian prediction is
    ``Phi(-(m+delta)/(sqrt(2)*sigma_d))`` with the module-level noise-scale formula.
    """
    if span <= 0 or not 0.0 <= decay <= 1.0 or step_k < 0:
        raise ValueError("span, decay, and step_k must define a non-negative EWMA scale")
    rows = []
    for date in sorted(partitions):
        labels = partitions[date].dropna()
        distance = distance_matrices[date].reindex(index=labels.index, columns=labels.index)
        for asset, own_label in labels.items():
            own_assets = labels.index[(labels == own_label) & (labels.index != asset)]
            own_distance = float(distance.loc[asset, own_assets].mean()) if len(own_assets) else 0.0
            other_means = [
                float(distance.loc[asset, members].mean())
                for label, members in labels.groupby(labels).groups.items()
                if label != own_label
            ]
            nearest_other = min(other_means) if other_means else np.nan
            margin = nearest_other - own_distance
            rho_hat = float(np.clip(1.0 - own_distance, -1.0, 1.0))
            sigma = (
                np.sqrt(2.0 * (1.0 - decay**step_k))
                * (1.0 - rho_hat**2)
                / np.sqrt(span)
            )
            if np.isnan(margin):
                probability = np.nan
            elif sigma <= 0:
                probability = float(margin + delta < 0)
            else:
                probability = float(ndtr(-(margin + delta) / (np.sqrt(2.0) * sigma)))
            rows.append({
                "date": date,
                "asset": asset,
                "margin": margin,
                "rho_hat": rho_hat,
                "sigma_d": sigma,
                "predicted_churn_probability": probability,
            })
    return pd.DataFrame(rows).set_index(["date", "asset"])


def predicted_realized_churn(
    margin_panel: pd.DataFrame,
    membership_panel: pd.DataFrame,
) -> pd.DataFrame:
    """Compare summed Gaussian switch probabilities with realised id changes by date."""
    predicted = margin_panel["predicted_churn_probability"].groupby(level="date").sum(min_count=1)
    left, right = membership_panel.shift(1), membership_panel
    valid = left.notna() & right.notna()
    realised = ((left != right) & valid).sum(axis=1).astype(float)
    return pd.concat(
        [predicted.rename("predicted_changes"), realised.rename("realised_changes")], axis=1
    ).dropna()


def membership_flow_decomposition(
    membership_panel: pd.DataFrame,
    inclusion_panel: pd.DataFrame,
    warmup_panel: pd.DataFrame,
) -> pd.DataFrame:
    """Classify U1 membership flows as index, warmup, or clusterer events.

    Index entry/exit is tested first from the point-in-time inclusion mask. A newly clustered
    continuing member is a warmup entry. Only a changed non-null derived id for an asset
    clustered at both dates is a clusterer reassignment and enters the churn numerator.
    """
    dates = membership_panel.index
    inclusion = inclusion_panel.reindex(index=dates, columns=membership_panel.columns).fillna(False)
    warmup = warmup_panel.reindex(index=dates, columns=membership_panel.columns).fillna(False)
    rows = []
    for i in range(1, len(dates)):
        date, prior_date = dates[i], dates[i - 1]
        for asset in membership_panel.columns:
            old_in = bool(inclusion.loc[prior_date, asset])
            new_in = bool(inclusion.loc[date, asset])
            old_id = membership_panel.loc[prior_date, asset]
            new_id = membership_panel.loc[date, asset]
            event = None
            if not old_in and new_in:
                event = "index_entry"
            elif old_in and not new_in:
                event = "index_exit"
            elif pd.isna(old_id) and pd.notna(new_id) and bool(warmup.loc[date, asset]):
                event = "warmup_entry"
            elif pd.notna(old_id) and pd.notna(new_id) and old_id != new_id:
                event = "clusterer_reassignment"
            if event is not None:
                rows.append({"date": date, "asset": asset, "event": event})
    events = pd.DataFrame(rows)
    if events.empty:
        return pd.DataFrame(columns=[
            "index_entry", "index_exit", "warmup_entry", "clusterer_reassignment"
        ])
    return pd.crosstab(events["date"], events["event"]).reindex(
        columns=["index_entry", "index_exit", "warmup_entry", "clusterer_reassignment"],
        fill_value=0,
    )


def fidelity_band(
    candidate: Mapping[pd.Timestamp, pd.Series],
    baseline: Mapping[pd.Timestamp, pd.Series],
    metadata: pd.DataFrame,
    taxonomy_columns: Sequence[str],
    *,
    taxonomy_tolerance: float = 0.03,
    cluster_count_tolerance: float = 0.15,
) -> Dict[str, Any]:
    """Return fidelity metrics and PASS/REJECTED under the frozen paper thresholds."""
    dates = sorted(set(candidate).intersection(baseline))
    same_date_ari = pd.Series([
        adjusted_rand_index(candidate[date], baseline[date]) for date in dates
    ])
    candidate_ari, _ = ari_metrics(candidate, metadata, taxonomy_columns)
    baseline_ari, _ = ari_metrics(baseline, metadata, taxonomy_columns)
    deltas = {key: candidate_ari[key] - baseline_ari[key] for key in candidate_ari}
    candidate_count = float(np.median([candidate[date].dropna().nunique() for date in dates]))
    baseline_count = float(np.median([baseline[date].dropna().nunique() for date in dates]))
    count_change = candidate_count / baseline_count - 1.0
    passed = all(abs(value) <= taxonomy_tolerance for value in deltas.values())
    passed &= abs(count_change) <= cluster_count_tolerance
    return {
        "baseline_partition_ari_median": float(same_date_ari.median()),
        **{f"delta_{key}": float(value) for key, value in deltas.items()},
        "cluster_count_relative_change": count_change,
        "fidelity_status": "PASS" if passed else "REJECTED",
    }


def covariance_invariance_metrics(
    candidate: RollingFactorCovarData,
    baseline: RollingFactorCovarData,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Return common-block covariance and equal-weight ex-ante-vol changes by date."""
    rows = []
    dates = sorted(set(candidate.dates).intersection(baseline.dates))
    for date in dates:
        candidate_snapshot, baseline_snapshot = candidate[date], baseline[date]
        assets = baseline_snapshot.y_betas.index.intersection(candidate_snapshot.y_betas.index)
        base = baseline_snapshot.get_y_covar(assets=assets).to_numpy()
        fitted = candidate_snapshot.get_y_covar(assets=assets).to_numpy()
        difference = fitted - base
        denominator = np.linalg.norm(base, ord="fro")
        max_denominator = np.max(np.abs(base))
        weights = np.full(len(assets), 1.0 / len(assets))
        base_vol = np.sqrt(max(float(weights @ base @ weights), 0.0))
        fitted_vol = np.sqrt(max(float(weights @ fitted @ weights), 0.0))
        rows.append({
            "date": date,
            "covar_relative_frobenius": float(np.linalg.norm(difference, ord="fro") / denominator),
            "covar_max_relative_entry": float(np.max(np.abs(difference)) / max_denominator),
            "ew_ex_ante_vol_abs_change": abs(fitted_vol - base_vol),
        })
    panel = pd.DataFrame(rows).set_index("date")
    return {column: float(panel[column].mean()) for column in panel}, panel


def signal_rank_metrics(
    partitions: Mapping[pd.Timestamp, pd.Series],
    weekly_returns: pd.DataFrame,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Measure month-to-month cluster-relative 48-week/4-week-skip rank stability."""
    dates = sorted(partitions)
    ranks: Dict[pd.Timestamp, pd.Series] = {}
    prior_partition_ranks: Dict[pd.Timestamp, pd.Series] = {}
    for i, date in enumerate(dates):
        window = weekly_returns.loc[
            date - pd.Timedelta(weeks=52):date - pd.Timedelta(weeks=4)
        ]
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
            "rank_spearman": current_pair.iloc[:, 0].corr(
                current_pair.iloc[:, 1], method="spearman"
            ),
            "rank_mad": (current_pair.iloc[:, 0] - current_pair.iloc[:, 1]).abs().mean(),
            "held_rank_spearman": held_pair.iloc[:, 0].corr(
                held_pair.iloc[:, 1], method="spearman"
            ),
            "held_rank_mad": (held_pair.iloc[:, 0] - held_pair.iloc[:, 1]).abs().mean(),
        })
    per_date = pd.DataFrame(rows).set_index("date")
    metrics = {column: float(per_date[column].mean()) for column in per_date}
    metrics["reassignment_rank_mad_gap"] = metrics["rank_mad"] - metrics["held_rank_mad"]
    return metrics, per_date


def turnover_decomposition(
    weights: pd.DataFrame,
    prior_partition_weights: pd.DataFrame,
    drifted_prior_weights: pd.DataFrame,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Return reassignment, signal, total, and residual one-way turnover.

    ``prior_partition_weights`` contains current-score weights under the prior partition.
    The two components form a triangle-inequality bound, not an additive identity, so
    ``residual = total - reassignment - signal`` is retained rather than suppressed.
    """
    index = weights.index.intersection(prior_partition_weights.index).intersection(
        drifted_prior_weights.index
    )
    columns = weights.columns.union(prior_partition_weights.columns).union(
        drifted_prior_weights.columns
    )
    current = weights.reindex(index=index, columns=columns, fill_value=0.0)
    held = prior_partition_weights.reindex(index=index, columns=columns, fill_value=0.0)
    drifted = drifted_prior_weights.reindex(index=index, columns=columns, fill_value=0.0)
    panel = pd.DataFrame({
        "reassignment_turnover": 0.5 * (current - held).abs().sum(axis=1),
        "signal_turnover": 0.5 * (held - drifted).abs().sum(axis=1),
        "total_turnover": 0.5 * (current - drifted).abs().sum(axis=1),
    })
    panel["turnover_residual"] = (
        panel["total_turnover"]
        - panel["reassignment_turnover"]
        - panel["signal_turnover"]
    )
    return {column: float(panel[column].mean()) for column in panel}, panel


def _lineage_pair_metrics(report: RiskClusterReport) -> Dict[str, float]:
    """Return pair-count lineage churn, matcher churn, and continuation overlap."""
    partitions = {}
    cluster_members = {}
    for date, frame in report.relabel.groupby("date"):
        raw_to_id = frame.set_index("raw_label")["derived_id"].to_dict()
        members = {
            raw: set(report.tracks[did].members[pd.Timestamp(date)])
            for raw, did in raw_to_id.items()
        }
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
        prior_by_id = {
            did: raw
            for raw, did in report.relabel.loc[
                report.relabel.date == d0, ["raw_label", "derived_id"]
            ].itertuples(index=False)
        }
        for raw, did in report.relabel.loc[
            report.relabel.date == d1, ["raw_label", "derived_id"]
        ].itertuples(index=False):
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


def lineage_metrics(
    covar_data: RollingFactorCovarData,
) -> Tuple[Dict[str, float], RiskClusterReport]:
    """Run canonical default lineage matching and return churn and track statistics."""
    report = analyze_cluster_lineage(covar_data)
    panel_churn = annualized_churn(report.to_membership_panel())
    metrics = _lineage_pair_metrics(report)
    lifetimes = report.classification["lifetime"]
    assets = report.to_membership_panel().notna().any(axis=0).sum()
    metrics.update({
        "lineage_churn_panel": panel_churn,
        "n_derived_tracks": float(len(report.tracks)),
        "tracks_per_asset": float(len(report.tracks) / assets),
        "median_track_life": float(lifetimes.median()),
        "median_beta_stability": float(report.classification["beta_stability"].median()),
    })
    assert round(metrics["lineage_churn_pair_count"], 4) == round(panel_churn, 4)
    return metrics, report


def residual_diagonality(
    covar_data: RollingFactorCovarData,
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """Return means of FactorLasso residual-diagonality diagnostics across dates."""
    rows = []
    for date in covar_data.dates:
        snapshot = covar_data[date]
        residuals = snapshot.residuals.loc[:, snapshot.residuals.notna().sum() >= 30]
        fitted = float((snapshot.y_betas.loc[residuals.columns].abs() > 1e-12).sum(axis=1).mean())
        diagnostics = diagnose_residuals(residuals, n_fitted_per_asset=fitted)
        rows.append({"date": date, **diagnostics.to_dict()})
    panel = pd.DataFrame(rows).set_index("date")
    numeric = panel.select_dtypes(include=[np.number]).mean().to_dict()
    return {f"diagonality_{key}": float(value) for key, value in numeric.items()}, panel


def track_taxonomy_purity(report: RiskClusterReport, taxonomy: pd.Series) -> pd.Series:
    """Return member-observation-weighted modal taxonomy purity for every track."""
    purity = {}
    for track_id, track in report.tracks.items():
        observations = pd.Series([
            taxonomy.get(asset) for members in track.members.values() for asset in members
        ]).dropna()
        purity[track_id] = (
            float(observations.value_counts(normalize=True).iloc[0])
            if len(observations)
            else np.nan
        )
    return pd.Series(purity, name="modal_taxonomy_purity")


def primary_factor_variance_share(report: RiskClusterReport) -> pd.Series:
    """Return each track's largest positive factor contribution share at its mean beta."""
    sigma = report.factor_covar.to_numpy()
    factors = report.factor_covar.index
    shares = {}
    for track_id, track in report.tracks.items():
        beta = track.betas.mean().reindex(factors).fillna(0.0).to_numpy()
        contribution = np.clip(beta * (sigma @ beta), 0.0, None)
        shares[track_id] = (
            float(contribution.max() / contribution.sum()) if contribution.sum() else 0.0
        )
    return pd.Series(shares, name="primary_factor_variance_share")


def interpretability_metrics(
    report: RiskClusterReport,
    metadata: pd.DataFrame,
    taxonomy_columns: Sequence[str],
    label_panel: pd.DataFrame,
    *,
    core_coverage: float = 0.70,
) -> Dict[str, float]:
    """Return the frozen taxonomy, purity, persistence, label, and factor metrics."""
    partitions = {
        pd.Timestamp(date): row.dropna() for date, row in report.to_membership_panel().iterrows()
    }
    taxonomy_ari, _ = ari_metrics(partitions, metadata, taxonomy_columns)
    purity = pd.concat([
        track_taxonomy_purity(report, metadata[column]).rename(column)
        for column in taxonomy_columns
    ], axis=1)
    modal_label_share = []
    for track_id, track in report.tracks.items():
        values = []
        for date, members in track.members.items():
            labels = label_panel.reindex(index=[date], columns=members).iloc[0].dropna()
            values.extend(labels.tolist())
        counts = pd.Series(values).value_counts(normalize=True)
        modal_label_share.append(float(counts.iloc[0]) if len(counts) else np.nan)
    labels = label_panel.stack(future_stack=True).dropna().astype(str)
    primary = primary_factor_variance_share(report)
    return {
        **taxonomy_ari,
        "track_modal_taxonomy_purity": float(purity.mean(axis=1).mean()),
        "core_track_count": float((report.classification["coverage"] >= core_coverage).sum()),
        "label_string_churn": annualized_churn(label_panel),
        "modal_label_life_share": float(np.nanmean(modal_label_share)),
        "distinct_label_count": float(labels.nunique()),
        "median_cluster_count": float(np.median([part.nunique() for part in partitions.values()])),
        "primary_factor_variance_share": float(primary.mean()),
        "idio_label_share": float(labels.str.contains("Idio", case=False).mean()),
    }


def deterministic_metric_bytes(metrics: Mapping[str, Any]) -> bytes:
    """Serialise scalar metric output into stable sorted UTF-8 JSON bytes."""
    clean = {
        key: (float(value) if isinstance(value, (np.floating, np.integer)) else value)
        for key, value in metrics.items()
    }
    return json.dumps(clean, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
