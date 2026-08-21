"""Run the fixed raw-versus-de-PC1 cluster experiment for U1, U2, and U3.

The clustering transform is supplied by the local FactorLasso checkout.  At every
date this runner forms the causal raw dependence matrix, restricts it to the exact
point-in-time investable asset set, removes the dominant common mode when requested,
and only then applies the frozen temporal smoother.  It never writes into the E2/E3/E5
cache trees: every snapshot is isolated below ``$CLUSTER_LINEAGE_OUTPUT_DIR/depc1``.
"""
from __future__ import annotations

import hashlib
import importlib
import json
import os
import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from factorlasso import (
    ClusterCorrelationTransform,
    CurrentFactorCovarData,
    RollingFactorCovarData,
    VarianceColumns,
    analyze_cluster_lineage,
    compute_clusters_from_corr_matrix,
    compute_rolling_smoothed_clusters,
    remove_first_principal_component,
)
from factorlasso.cluster_smoothing import _iter_correlation_inputs

import papers.cluster_lineage_2026.replication.run_backtests as e5
import papers.cluster_lineage_2026.replication.run_u1_covar_grid as u1_grid
import papers.cluster_lineage_2026.replication.run_u2_blackrock_aum_filter as u2_aum
import papers.cluster_lineage_2026.replication.run_u2_blackrock_aum_sensitivity as u2_sensitivity
import papers.cluster_lineage_2026.replication.run_u2_blackrock_etf_grid as u2_funds
from papers.cluster_lineage_2026.replication.recovery_loader import install as install_pyc_finder
from papers.cluster_lineage_2026.replication import (
    run_futures_prod_signal_grid_30303010_10bp as futures_grid,
)


install_pyc_finder()
estimate = importlib.import_module("papers.cluster_lineage_2026.replication.estimate")

RUNNER = "papers/cluster_lineage_2026/replication/run_depc1_cluster_comparison.py"
CACHE_VERSION = 1
TRANSFORMS = ("raw", "depc1")
UNIVERSES = ("blackrock_funds", "futures", "msci_us")
TOLERANCE = 1e-12
FACTORLASSO_ROOT = Path(
    os.environ.get(
        "FACTORLASSO_DEV_ROOT",
        r"C:\Users\artur\OneDrive\analytics\my_github\FactorLasso",
    )
)


@dataclass(frozen=True)
class UniverseInputs:
    """Hold one frozen universe's cluster-discovery inputs."""

    universe: str
    returns: pd.DataFrame
    dates: pd.DatetimeIndex
    eligibility: pd.DataFrame
    model: object
    taxonomy: Mapping[str, pd.Series]
    frozen_panel: pd.DataFrame
    config_id: str
    input_paths: tuple[Path, ...]


def _output_root() -> Path:
    """Return the isolated de-PC1 output root."""
    base = Path(
        os.environ.get(
            "CLUSTER_LINEAGE_OUTPUT_DIR",
            r"C:\Users\artur\OneDrive\analytics\outputs\cluster_lineage_2026",
        )
    )
    root = base / "depc1"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _universe_root(universe: str) -> Path:
    """Return one universe's report directory."""
    root = _output_root() / universe
    root.mkdir(parents=True, exist_ok=True)
    return root


def _cache_dir(inputs: UniverseInputs, transform: str) -> Path:
    """Return the required per-date cache directory."""
    root = _universe_root(inputs.universe) / transform / inputs.config_id
    root.mkdir(parents=True, exist_ok=True)
    return root


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _frame_hash(frame: pd.DataFrame) -> str:
    """Hash a labelled frame independently of pickle serialization."""
    values = pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()
    columns = "\x1f".join(map(str, frame.columns)).encode("utf-8")
    return hashlib.sha256(values + columns).hexdigest()


def _factorlasso_manifest() -> dict[str, str]:
    """Return source hashes proving the local development implementation."""
    files = (
        FACTORLASSO_ROOT / "factorlasso" / "cluster_utils.py",
        FACTORLASSO_ROOT / "factorlasso" / "cluster_smoothing.py",
        FACTORLASSO_ROOT / "factorlasso" / "lasso_estimator.py",
    )
    return {str(path): _sha256(path) for path in files}


def _model_payload(model: object) -> dict[str, object]:
    """Return only clustering fields needed in a cache fingerprint."""
    names = (
        "span",
        "span_freq_dict",
        "cutoff_fraction",
        "linkage_method",
        "distance_transform",
        "dependence_measure",
        "n_clusters",
        "cluster_smoother_type",
        "smoother_delta",
        "smoother_lambda",
        "recluster_freq",
        "warmup_period",
        "demean",
    )
    return {name: str(getattr(model, name)) for name in names}


def _fingerprint(inputs: UniverseInputs, transform: str) -> str:
    """Return a stable source/data/specification fingerprint."""
    payload = {
        "version": CACHE_VERSION,
        "universe": inputs.universe,
        "transform": transform,
        "config_id": inputs.config_id,
        "returns": _frame_hash(inputs.returns),
        "eligibility": _frame_hash(inputs.eligibility.astype(bool)),
        "dates": [str(date) for date in inputs.dates],
        "factorlasso": _factorlasso_manifest(),
        "runner_sha256": _sha256(Path(__file__)),
        "model": _model_payload(inputs.model),
        "inputs": {
            str(path): _sha256(path) for path in inputs.input_paths if path.exists()
        },
    }
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _same_partition(left: pd.Series, right: pd.Series) -> bool:
    """Return whether complete assignments induce the same equivalence classes."""
    left = left.dropna().sort_index()
    right = right.dropna().sort_index()
    if not left.index.equals(right.index):
        return False
    if left.empty:
        return True
    left_values = left.to_numpy()
    right_values = right.to_numpy()
    return bool(
        np.array_equal(
            left_values[:, None] == left_values[None, :],
            right_values[:, None] == right_values[None, :],
        )
    )


def _adjusted_rand(left: pd.Series, right: pd.Series) -> float:
    """Return adjusted Rand on the common non-missing asset set."""
    frame = pd.concat([left, right], axis=1).dropna()
    if len(frame) < 2:
        return np.nan
    table = pd.crosstab(frame.iloc[:, 0], frame.iloc[:, 1]).to_numpy(dtype=float)

    def choose_two(values):
        """Return the elementwise number of unordered pairs."""
        return values * (values - 1.0) / 2.0

    observed = float(choose_two(table).sum())
    rows = float(choose_two(table.sum(axis=1)).sum())
    columns = float(choose_two(table.sum(axis=0)).sum())
    total = len(frame) * (len(frame) - 1.0) / 2.0
    expected = rows * columns / total
    maximum = 0.5 * (rows + columns)
    return 1.0 if maximum == expected else (observed - expected) / (maximum - expected)


def _pairwise_rand(left: pd.Series, right: pd.Series) -> float:
    """Return the unadjusted pairwise Rand agreement."""
    frame = pd.concat([left, right], axis=1).dropna()
    if len(frame) < 2:
        return np.nan
    first = frame.iloc[:, 0].to_numpy()
    second = frame.iloc[:, 1].to_numpy()
    upper = np.triu_indices(len(frame), k=1)
    same_first = (first[:, None] == first[None, :])[upper]
    same_second = (second[:, None] == second[None, :])[upper]
    return float(np.mean(same_first == same_second))


def _offdiagonal(values: np.ndarray) -> np.ndarray:
    """Return finite upper-triangle off-diagonal values."""
    output = values[np.triu_indices(len(values), k=1)]
    return output[np.isfinite(output)]


def _u2_inputs() -> UniverseInputs:
    """Load the owner-frozen BlackRock AUM100 operating point."""
    daily = u2_funds._read_daily()
    dates = u2_funds._dates()
    rolling_aum = u2_aum._rolling_aum()
    eligibilities = u2_sensitivity._eligibilities(daily, dates, rolling_aum)
    eligibility = eligibilities["aum_100m"].astype(bool)
    returns = u2_funds._native_returns(daily, "W-THU")
    model = u2_funds._model(156, "W-THU")
    cached_panels, _, _ = u2_sensitivity._build_partitions(eligibilities)
    metadata = pd.read_csv(u2_funds.METADATA_FILE).set_index("ticker")
    taxonomy = {
        "asset_class": metadata["asset_class"].reindex(daily.columns),
        "sub_asset_class": metadata["sub_asset_class"].reindex(daily.columns),
    }
    return UniverseInputs(
        universe="blackrock_funds",
        returns=returns,
        dates=dates,
        eligibility=eligibility,
        model=model,
        taxonomy=taxonomy,
        frozen_panel=cached_panels["aum_100m"].reindex(index=dates, columns=daily.columns),
        config_id="W_THU_span_156_aum100",
        input_paths=tuple(u2_funds.INPUT_FILES) + (u2_aum.AUM_FILE,),
    )


def _u3_inputs() -> UniverseInputs:
    """Load the owner-frozen futures M1-star operating point."""
    data = e5.load_universe(e5.UniverseName.FUTURES)
    dates = pd.DatetimeIndex(
        e5.load_cached(e5.UniverseName.FUTURES, e5.SmootherName.BASELINE).dates
    )
    eligibility = e5._investable_eligibility(data, dates).astype(bool)
    returns = data.asset_returns["W-WED"].reindex(columns=eligibility.columns)
    model = estimate.make_estimator(
        e5.UniverseName.FUTURES, e5.SmootherName.M1_STAR
    ).lasso_model
    frozen = e5._cluster_groups(
        e5.UniverseName.FUTURES, e5.SmootherName.M1_STAR
    ).reindex(index=dates, columns=eligibility.columns)
    frozen = frozen.where(eligibility)
    taxonomy = {
        "asset_class": data.taxonomy["asset_class"].reindex(eligibility.columns)
    }
    return UniverseInputs(
        universe="futures",
        returns=returns,
        dates=pd.DatetimeIndex(dates),
        eligibility=eligibility,
        model=model,
        taxonomy=taxonomy,
        frozen_panel=frozen,
        config_id="W_WED_span_156_M1_star_delta_0.0691",
        input_paths=(futures_grid.base.DATA_PATH,),
    )


def _u1_inputs() -> UniverseInputs:
    """Load the owner-frozen U1 ME/span36 operating point."""
    full_dates, full_eligibility = u1_grid._accepted_dates_and_eligibility()
    dates = full_dates[
        (full_dates >= pd.Timestamp("2009-08-31"))
        & (full_dates <= pd.Timestamp("2026-06-30"))
    ]
    eligibility = full_eligibility.reindex(index=dates)
    daily = u1_grid._read_daily(full_eligibility.columns)
    returns = u1_grid._native_returns(daily, "ME")
    # U1's accepted external mask already applies its 12-observation W-WED
    # warmup. Do not apply the generic model's count again at the ME covariance
    # cadence, which would silently turn that into a 12-month warmup.
    model = u1_grid._model(36, "ME").copy(kwargs={"warmup_period": None})
    frozen, _ = u1_grid._load_partition("ME", 36)
    data = e5.load_universe(e5.UniverseName.MSCI_US)
    taxonomy = {
        "bbg_bics_sector": data.taxonomy["bbg_bics_sector"].reindex(
            eligibility.columns
        )
    }
    return UniverseInputs(
        universe="msci_us",
        returns=returns,
        dates=pd.DatetimeIndex(dates),
        eligibility=eligibility.astype(bool),
        model=model,
        taxonomy=taxonomy,
        frozen_panel=frozen.reindex(index=dates, columns=eligibility.columns),
        config_id="ME_span_036",
        input_paths=(u1_grid.DATA_DIR / "msci_us_log_returns.csv",),
    )


def load_inputs(universe: str) -> UniverseInputs:
    """Load one named frozen universe in the roadmap's execution order."""
    loaders = {
        "blackrock_funds": _u2_inputs,
        "futures": _u3_inputs,
        "msci_us": _u1_inputs,
    }
    if universe not in loaders:
        raise KeyError(f"unknown universe {universe!r}; expected {tuple(loaders)}")
    return loaders[universe]()


def _cache_path(inputs: UniverseInputs, transform: str, date: pd.Timestamp) -> Path:
    """Return one required YYYYMMDD pickle path."""
    return _cache_dir(inputs, transform) / f"{pd.Timestamp(date):%Y%m%d}.pkl"


def _load_cached_panel(
    inputs: UniverseInputs,
    transform: str,
) -> tuple[pd.DataFrame, list[dict[str, object]]] | None:
    """Load a complete valid per-date cache, or return None."""
    fingerprint = _fingerprint(inputs, transform)
    panel = pd.DataFrame(
        np.nan,
        index=inputs.dates,
        columns=inputs.eligibility.columns,
        dtype=object,
    )
    diagnostics = []
    for date in inputs.dates:
        path = _cache_path(inputs, transform, date)
        if not path.exists():
            return None
        with path.open("rb") as stream:
            payload = pickle.load(stream)
        if (
            payload.get("version") != CACHE_VERSION
            or payload.get("fingerprint") != fingerprint
            or pd.Timestamp(payload.get("date")) != pd.Timestamp(date)
        ):
            return None
        labels = payload["clusters"]
        panel.loc[date, labels.index] = labels.to_numpy()
        diagnostics.append(payload["diagnostics"])
    return panel, diagnostics


def _write_cache(
    inputs: UniverseInputs,
    transform: str,
    rolling,
    diagnostics: Mapping[pd.Timestamp, dict[str, object]],
) -> pd.DataFrame:
    """Write one pickle per date and return the assembled membership panel."""
    fingerprint = _fingerprint(inputs, transform)
    panel = pd.DataFrame(
        np.nan,
        index=inputs.dates,
        columns=inputs.eligibility.columns,
        dtype=object,
    )
    for date in inputs.dates:
        labels = rolling.clusters.get(date, pd.Series(dtype=float)).dropna()
        panel.loc[date, labels.index] = labels.to_numpy()
        diagnostics[pd.Timestamp(date)][
            "injected_partition_matches_fitted"
        ] = _injected_partition_matches(
            inputs.model,
            labels,
            rolling.linkages.get(date, np.empty((0, 4))),
            rolling.cutoffs.get(date, 0.0),
        )
        payload = {
            "version": CACHE_VERSION,
            "fingerprint": fingerprint,
            "date": pd.Timestamp(date),
            "clusters": labels,
            "linkage": rolling.linkages.get(date, np.empty((0, 4))),
            "cutoff": rolling.cutoffs.get(date, 0.0),
            "diagnostics": diagnostics[pd.Timestamp(date)],
        }
        path = _cache_path(inputs, transform, date)
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
        temporary.replace(path)
    return panel


def _empty_rolling(dates: pd.DatetimeIndex):
    """Return empty dictionaries matching RollingClusterData's interface."""
    from factorlasso import RollingClusterData

    return RollingClusterData(
        clusters={date: pd.Series(dtype=float) for date in dates},
        linkages={date: np.empty((0, 4)) for date in dates},
        cutoffs={date: 0.0 for date in dates},
        co_association=pd.DataFrame(),
    )


def _injected_partition_matches(
    model,
    clusters: pd.Series,
    linkage: np.ndarray,
    cutoff: float,
) -> bool:
    """Validate the external-cluster preparation path without solving a regression."""
    clusters = clusters.dropna()
    if len(clusters) < 2:
        return True
    dates = pd.date_range("2000-01-03", periods=4, freq="B")
    factor = pd.DataFrame(
        {"injection_check_factor": [-0.02, 0.01, 0.03, -0.01]}, index=dates
    )
    loadings = np.linspace(0.5, 1.5, len(clusters))
    responses = pd.DataFrame(
        factor.to_numpy() @ loadings[None, :],
        index=dates,
        columns=clusters.index,
    )
    responses_np = responses.to_numpy()
    prepared = model.copy()._prepare_fit(
        x=factor,
        y=responses,
        x_np=factor.to_numpy(),
        y_np=responses_np,
        valid_mask=np.ones_like(responses_np),
        eff_span=model.span,
        external_clusters=clusters,
        external_linkage=linkage,
        external_cutoff=cutoff,
    )
    return _same_partition(prepared.asset_clusters, clusters)


def _rolling_for_transform(inputs: UniverseInputs, transform: str):
    """Compute rolling clusters only on dates with a non-empty eligible set."""
    active_dates = inputs.dates[inputs.eligibility.sum(axis=1).gt(0)]
    if len(active_dates) == 0:
        return _empty_rolling(inputs.dates)
    mode = (
        ClusterCorrelationTransform.NONE
        if transform == "raw"
        else ClusterCorrelationTransform.REMOVE_PC1
    )
    model = inputs.model.copy(
        kwargs={"cluster_correlation_transform": mode}
    )
    computed = compute_rolling_smoothed_clusters(
        inputs.returns,
        list(active_dates),
        model,
        eligibility=inputs.eligibility.loc[active_dates],
    )
    if inputs.universe == "futures" and transform == "raw":
        exact_matches = {
            pd.Timestamp(date): _same_partition(
                computed.clusters[pd.Timestamp(date)],
                inputs.frozen_panel.loc[date],
            )
            for date in active_dates
        }
        computed.co_association.attrs["exact_refit_frozen_matches"] = exact_matches
        # The current owner-frozen U3 strategy reuses the accepted M1-star
        # partitions and applies the seven later liquidity exclusions only in
        # eligibility/weights. Preserve that raw arm exactly; the exact-universe
        # refit share above is reported as a diagnostic, not substituted silently.
        for date in active_dates:
            computed.clusters[pd.Timestamp(date)] = inputs.frozen_panel.loc[
                date
            ].dropna()
    empty = _empty_rolling(inputs.dates[~inputs.dates.isin(active_dates)])
    computed.clusters.update(empty.clusters)
    computed.linkages.update(empty.linkages)
    computed.cutoffs.update(empty.cutoffs)
    return computed


def _correlation_diagnostics(
    inputs: UniverseInputs,
    raw_panel: pd.DataFrame,
    depc1_panel: pd.DataFrame,
) -> tuple[dict[pd.Timestamp, dict[str, object]], dict[pd.Timestamp, dict[str, object]]]:
    """Compute PC1, geometry, taxonomy, and matched-count diagnostics."""
    raw_rows: dict[pd.Timestamp, dict[str, object]] = {}
    depc_rows: dict[pd.Timestamp, dict[str, object]] = {}
    active_dates = inputs.dates[inputs.eligibility.sum(axis=1).gt(0)]
    iterator = _iter_correlation_inputs(
        inputs.returns, list(active_dates), inputs.model
    )
    for date, full_corr in iterator:
        assets = inputs.eligibility.columns[inputs.eligibility.loc[date]]
        corr = full_corr.reindex(index=assets, columns=assets)
        result = remove_first_principal_component(corr)
        eigenvalues, eigenvectors = np.linalg.eigh(corr.fillna(0.0).to_numpy())
        loading = eigenvectors[:, -1]
        raw_values = _offdiagonal(corr.fillna(0.0).to_numpy())
        residual_values = _offdiagonal(result.correlation.to_numpy())
        raw_labels = raw_panel.loc[date].dropna()
        depc_labels = depc1_panel.loc[date].dropna()
        count = max(int(raw_labels.nunique()), 1)
        raw_matched = compute_clusters_from_corr_matrix(corr, n_clusters=count)[0]
        depc_matched = compute_clusters_from_corr_matrix(
            result.correlation, n_clusters=count
        )[0]
        base = {
            "date": pd.Timestamp(date),
            "eligible_assets": len(assets),
            "removed_eigenvalue": result.removed_eigenvalue,
            "pc1_variance_share": result.removed_variance_share,
            "top_eigengap": result.eigengap,
            "dominant_component_unique": result.dominant_component_unique,
            "pc1_loading_concentration_sum_fourth": float(np.sum(loading**4)),
            "minimum_residual_variance": result.minimum_residual_variance,
            "isolated_residual_assets": "|".join(map(str, result.isolated_assets)),
            "isolated_residual_asset_count": len(result.isolated_assets),
            "missing_offdiagonal_pairs": result.missing_offdiagonal_pairs,
            "raw_offdiagonal_mean": float(np.mean(raw_values)) if raw_values.size else np.nan,
            "raw_offdiagonal_median": float(np.median(raw_values)) if raw_values.size else np.nan,
            "depc1_offdiagonal_mean": (
                float(np.mean(residual_values)) if residual_values.size else np.nan
            ),
            "depc1_offdiagonal_median": (
                float(np.median(residual_values)) if residual_values.size else np.nan
            ),
            "fixed_cut_ari_raw_vs_depc1": _adjusted_rand(raw_labels, depc_labels),
            "fixed_cut_pairwise_rand": _pairwise_rand(raw_labels, depc_labels),
            "matched_count_ari_raw_vs_depc1": _adjusted_rand(
                raw_matched, depc_matched
            ),
            "raw_cluster_count": int(raw_labels.nunique()),
            "depc1_cluster_count": int(depc_labels.nunique()),
        }
        for name, taxonomy in inputs.taxonomy.items():
            base[f"raw_taxonomy_ari_{name}"] = _adjusted_rand(raw_labels, taxonomy)
            base[f"depc1_taxonomy_ari_{name}"] = _adjusted_rand(
                depc_labels, taxonomy
            )
        raw_sizes = raw_labels.value_counts()
        depc_sizes = depc_labels.value_counts()
        raw_rows[pd.Timestamp(date)] = {
            **base,
            "transform": "raw",
            "cluster_count": int(raw_labels.nunique()),
            "singleton_share": (
                float(raw_sizes.eq(1).mean()) if len(raw_sizes) else np.nan
            ),
            "median_cluster_size": (
                float(raw_sizes.median()) if len(raw_sizes) else np.nan
            ),
        }
        depc_rows[pd.Timestamp(date)] = {
            **base,
            "transform": "depc1",
            "cluster_count": int(depc_labels.nunique()),
            "singleton_share": (
                float(depc_sizes.eq(1).mean()) if len(depc_sizes) else np.nan
            ),
            "median_cluster_size": (
                float(depc_sizes.median()) if len(depc_sizes) else np.nan
            ),
        }
    for date in inputs.dates[~inputs.dates.isin(active_dates)]:
        empty = {
            "date": pd.Timestamp(date),
            "eligible_assets": 0,
            "cluster_count": 0,
            "raw_cluster_count": 0,
            "depc1_cluster_count": 0,
        }
        raw_rows[pd.Timestamp(date)] = {**empty, "transform": "raw"}
        depc_rows[pd.Timestamp(date)] = {**empty, "transform": "depc1"}
    return raw_rows, depc_rows


def _lineage_data(panel: pd.DataFrame) -> RollingFactorCovarData:
    """Build neutral factor snapshots so canonical lineage reads memberships only."""
    rolling = RollingFactorCovarData()
    factor = pd.DataFrame([[1.0]], index=["Common"], columns=["Common"])
    for date, row in panel.iterrows():
        clusters = row.dropna()
        if clusters.empty:
            continue
        betas = pd.DataFrame(0.0, index=clusters.index, columns=["Common"])
        variances = pd.DataFrame(
            {VarianceColumns.RESIDUAL_VARS.value: 1.0}, index=clusters.index
        )
        rolling.add(
            pd.Timestamp(date),
            CurrentFactorCovarData(
                x_covar=factor,
                y_betas=betas,
                y_variances=variances,
                estimation_date=pd.Timestamp(date),
                clusters=clusters,
            ),
        )
    return rolling


def _annualized_membership_churn(panel: pd.DataFrame) -> float:
    """Return annualized derived-ID changes over jointly observed assets."""
    if len(panel) < 2:
        return 0.0
    left = panel.iloc[:-1].copy()
    right = panel.iloc[1:].copy()
    left.index = right.index
    valid = left.notna() & right.notna()
    pairs = int(valid.to_numpy().sum())
    years = (panel.index[-1] - panel.index[0]).days / 365.25
    if pairs == 0 or years <= 0:
        return 0.0
    changes = int(((left != right) & valid).to_numpy().sum())
    return float(changes * (len(panel) - 1) / (pairs * years))


def _lineage_metrics(panel: pd.DataFrame, transform: str) -> dict[str, object]:
    """Return canonical lineage counts, churn, and E0b tracks-per-asset."""
    report = analyze_cluster_lineage(_lineage_data(panel))
    membership = report.to_membership_panel()
    distinct = membership.nunique(axis=0, dropna=True)
    distinct = distinct[distinct.gt(0)]
    events = report.lineage["event"].value_counts() if not report.lineage.empty else pd.Series()
    median_clusters = float(panel.nunique(axis=1, dropna=True).replace(0, np.nan).median())
    return {
        "transform": transform,
        "n_derived_tracks": len(report.tracks),
        "tracks_per_asset": float(distinct.mean()) if len(distinct) else np.nan,
        "track_to_asset_ratio": (
            float(len(report.tracks) / len(distinct)) if len(distinct) else np.nan
        ),
        "lineage_churn_annualized": _annualized_membership_churn(membership),
        "fragmentation_tracks_per_median_cluster": (
            float(len(report.tracks) / median_clusters)
            if median_clusters > 0
            else np.nan
        ),
        "births": int(events.get("birth", 0)),
        "deaths": int(events.get("death", 0)),
        "splits": int(events.get("split", 0)),
        "merges": int(events.get("merge", 0)),
    }


def _partition_hash(panel: pd.DataFrame) -> str:
    """Return a stable membership-panel hash."""
    return _frame_hash(panel)


def _source_manifest(inputs: UniverseInputs) -> pd.DataFrame:
    """Return local-source and input provenance rows."""
    rows = [
        {"kind": "factorlasso_source", "path": path, "sha256": digest}
        for path, digest in _factorlasso_manifest().items()
    ]
    rows.extend(
        {
            "kind": "input",
            "path": str(path),
            "sha256": _sha256(path),
        }
        for path in inputs.input_paths
        if path.exists()
    )
    rows.append(
        {
            "kind": "runner",
            "path": RUNNER,
            "sha256": _sha256(Path(__file__)),
        }
    )
    return pd.DataFrame(rows)


def _write(frame: pd.DataFrame, path: Path) -> None:
    """Write a deterministic research CSV."""
    frame.to_csv(path, index=False, float_format="%.17g", lineterminator="\n")


def run_universe(universe: str, *, force: bool = False) -> Mapping[str, pd.DataFrame]:
    """Run one universe cache-first and emit every D4 acceptance table."""
    started = time.perf_counter()
    inputs = load_inputs(universe)
    cached = {
        transform: None if force else _load_cached_panel(inputs, transform)
        for transform in TRANSFORMS
    }
    cache_status = {
        transform: "hit" if cached[transform] is not None else "miss"
        for transform in TRANSFORMS
    }
    if all(item is not None for item in cached.values()):
        raw_panel = cached["raw"][0]
        depc1_panel = cached["depc1"][0]
        raw_diagnostics = {
            pd.Timestamp(row["date"]): row for row in cached["raw"][1]
        }
        depc_diagnostics = {
            pd.Timestamp(row["date"]): row for row in cached["depc1"][1]
        }
    else:
        raw_rolling = _rolling_for_transform(inputs, "raw")
        depc_rolling = _rolling_for_transform(inputs, "depc1")
        raw_panel = pd.DataFrame(
            {date: labels for date, labels in raw_rolling.clusters.items()}
        ).T.reindex(index=inputs.dates, columns=inputs.eligibility.columns)
        depc1_panel = pd.DataFrame(
            {date: labels for date, labels in depc_rolling.clusters.items()}
        ).T.reindex(index=inputs.dates, columns=inputs.eligibility.columns)
        raw_diagnostics, depc_diagnostics = _correlation_diagnostics(
            inputs, raw_panel, depc1_panel
        )
        exact_matches = raw_rolling.co_association.attrs.get(
            "exact_refit_frozen_matches", {}
        )
        for date, matches in exact_matches.items():
            raw_diagnostics[pd.Timestamp(date)][
                "raw_exact_eligibility_refit_matches_frozen"
            ] = bool(matches)
        raw_panel = _write_cache(
            inputs, "raw", raw_rolling, raw_diagnostics
        )
        depc1_panel = _write_cache(
            inputs, "depc1", depc_rolling, depc_diagnostics
        )

    per_date = pd.DataFrame(
        [raw_diagnostics[date] for date in inputs.dates]
    )
    depc_per_date = pd.DataFrame(
        [depc_diagnostics[date] for date in inputs.dates]
    )
    pc1_prefixes = (
        "date",
        "eligible",
        "removed",
        "pc1_",
        "top_",
        "dominant_",
        "minimum_",
        "isolated_",
        "missing_",
        "raw_off",
        "depc1_off",
    )
    pc1_columns = [
        column
        for column in per_date
        if column.startswith(pc1_prefixes)
    ]
    pc1_diagnostics = per_date[pc1_columns].copy()
    comparison_prefixes = (
        "date",
        "eligible",
        "fixed_",
        "matched_",
        "raw_cluster",
        "depc1_cluster",
        "raw_taxonomy",
        "depc1_taxonomy",
        "raw_exact_eligibility",
    )
    comparison_columns = [
        column
        for column in per_date
        if column.startswith(comparison_prefixes)
    ]
    partition_comparison = per_date[comparison_columns].copy()
    metric_panel = pd.concat([per_date, depc_per_date], ignore_index=True)
    numeric = metric_panel.select_dtypes(include=[np.number]).columns
    cluster_summary = (
        metric_panel.groupby("transform", sort=False)[list(numeric)]
        .median(numeric_only=True)
        .reset_index()
    )
    lineage = pd.DataFrame(
        [
            _lineage_metrics(raw_panel, "raw"),
            _lineage_metrics(depc1_panel, "depc1"),
        ]
    )

    eligibility_match = raw_panel.notna().eq(inputs.eligibility).to_numpy().all()
    depc_eligibility_match = depc1_panel.notna().eq(inputs.eligibility).to_numpy().all()
    reference_matches = []
    depc_matches = []
    for date in inputs.dates:
        reference_matches.append(
            _same_partition(raw_panel.loc[date], inputs.frozen_panel.loc[date])
        )
        depc_matches.append(
            bool(
                depc_diagnostics[pd.Timestamp(date)].get(
                    "injected_partition_matches_fitted", False
                )
            )
        )
    active = int(inputs.eligibility.sum(axis=1).gt(0).sum())
    acceptance = pd.DataFrame(
        [
            {
                "check": "raw and de-PC1 schedule identity",
                "measured": int(raw_panel.index.equals(depc1_panel.index)),
                "tolerance": 1,
            },
            {
                "check": "raw exact eligibility membership",
                "measured": int(eligibility_match),
                "tolerance": 1,
            },
            {
                "check": "de-PC1 exact eligibility membership",
                "measured": int(depc_eligibility_match),
                "tolerance": 1,
            },
            {
                "check": "raw frozen partition match share",
                "measured": float(np.mean(reference_matches)),
                "tolerance": 1.0,
            },
            {
                "check": "de-PC1 injected/fitted partition match share",
                "measured": float(np.mean(depc_matches)),
                "tolerance": 1.0,
            },
            {
                "check": "active snapshot count",
                "measured": active,
                "tolerance": active,
            },
        ]
    )
    acceptance["status"] = np.where(
        acceptance["measured"].astype(float).eq(
            acceptance["tolerance"].astype(float)
        ),
        "PASS",
        "FAIL",
    )
    runtime = pd.DataFrame(
        [
            {
                "universe": universe,
                "raw_cache_status": cache_status["raw"],
                "depc1_cache_status": cache_status["depc1"],
                "snapshots": len(inputs.dates),
                "active_snapshots": active,
                "raw_cache_bytes": sum(
                    path.stat().st_size
                    for path in _cache_dir(inputs, "raw").glob("*.pkl")
                ),
                "depc1_cache_bytes": sum(
                    path.stat().st_size
                    for path in _cache_dir(inputs, "depc1").glob("*.pkl")
                ),
                "runtime_seconds": time.perf_counter() - started,
            }
        ]
    )
    manifest = _source_manifest(inputs)
    panels = pd.DataFrame(
        [
            {
                "transform": "raw",
                "partition_hash": _partition_hash(raw_panel),
                "dates": len(raw_panel),
                "assets": len(raw_panel.columns),
            },
            {
                "transform": "depc1",
                "partition_hash": _partition_hash(depc1_panel),
                "dates": len(depc1_panel),
                "assets": len(depc1_panel.columns),
            },
        ]
    )
    output = {
        "pc1_diagnostics": pc1_diagnostics,
        "partition_comparison": partition_comparison,
        "cluster_metric_summary": cluster_summary,
        "lineage_comparison": lineage,
        "partition_manifest": panels,
        "acceptance": acceptance,
        "runtime": runtime,
        "source_manifest": manifest,
    }
    root = _universe_root(universe)
    for name, frame in output.items():
        _write(frame, root / f"{name}.csv")
    return {**output, "raw_panel": raw_panel, "depc1_panel": depc1_panel}


def _artifact_hashes(universe: str) -> dict[str, str]:
    """Hash deterministic D4 CSVs, excluding runtime and replay output."""
    return {
        path.name: _sha256(path)
        for path in sorted(_universe_root(universe).glob("*.csv"))
        if path.name not in {"runtime.csv", "determinism.csv"}
    }


def verify_determinism(universe: str) -> pd.DataFrame:
    """Require two cache-first emissions to be byte-identical."""
    run_universe(universe)
    first = _artifact_hashes(universe)
    run_universe(universe)
    second = _artifact_hashes(universe)
    names = sorted(set(first) | set(second))
    frame = pd.DataFrame(
        [
            {
                "artifact": name,
                "first_sha256": first.get(name),
                "second_sha256": second.get(name),
                "byte_identical": first.get(name) == second.get(name),
            }
            for name in names
        ]
    )
    _write(frame, _universe_root(universe) / "determinism.csv")
    if not frame["byte_identical"].all():
        raise AssertionError(frame.loc[~frame["byte_identical"]])
    return frame


def run_all(*, verify: bool = True) -> Mapping[str, Mapping[str, pd.DataFrame]]:
    """Run U2, U3, then U1 as frozen in the roadmap."""
    output = {}
    for universe in UNIVERSES:
        print(f"de-PC1 cluster comparison: {universe}", flush=True)
        output[universe] = run_universe(universe)
        if verify:
            verify_determinism(universe)
    return output


def main() -> None:
    """Execute the complete isolated partition experiment."""
    run_all(verify=True)
    for universe in UNIVERSES:
        acceptance = pd.read_csv(_universe_root(universe) / "acceptance.csv")
        print(f"\n{universe}\n{acceptance.to_string(index=False)}", flush=True)


if __name__ == "__main__":
    main()
