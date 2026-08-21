"""Rebuild the owner-authorised U1 ME/36 adopted smoothing cache.

This runner is intentionally limited to the frozen application cell: monthly returns,
EWMA span 36, point-in-time index eligibility, Ward distance clustering, and partition-bonus
delta 0.0866.  It performs no parameter search and leaves the existing unsmoothed partition
cache untouched.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from factorlasso import ClusterSmootherType, compute_rolling_smoothed_clusters

import papers.cluster_lineage_2026.replication.run_u1_covar_grid as u1_grid


DELTA = 0.0866
FREQUENCY = "ME"
SPAN = 36
START = pd.Timestamp("2009-08-31")
END = pd.Timestamp("2026-06-30")
CACHE_VERSION = 1
TOLERANCE = 1e-12


def _output_root() -> Path:
    """Return the configured cluster-lineage output root."""

    value = os.environ.get("CLUSTER_LINEAGE_OUTPUT_DIR")
    if not value:
        raise RuntimeError("CLUSTER_LINEAGE_OUTPUT_DIR must be set")
    return Path(value).resolve()


def _cache_dir() -> Path:
    """Return the isolated adopted-cell cache directory."""

    return _output_root() / "msci_us/ME_span_036_M1_star_delta_0.0866"


def _baseline_path() -> Path:
    """Return the frozen unsmoothed ME/36 partition-panel path."""

    return (
        _output_root()
        / "e5b/covariance_frequency_span_grid/msci_us/partitions/ME_span_036.pkl"
    )


def _sha256(path: Path) -> str:
    """Return one file's SHA-256 digest."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _frame_hash(frame: pd.DataFrame) -> str:
    """Return a labelled pandas-frame digest."""

    values = pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()
    columns = "\x1f".join(map(str, frame.columns)).encode("utf-8")
    return hashlib.sha256(values + columns).hexdigest()


def _same_partition(left: pd.Series, right: pd.Series) -> bool:
    """Return whether two assignments induce the same equivalence classes."""

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


def _inputs() -> tuple[pd.DataFrame, pd.DatetimeIndex, pd.DataFrame, pd.DataFrame, object]:
    """Load the exact frozen U1 ME/36 inputs without fitting anything."""

    all_dates, all_eligibility = u1_grid._accepted_dates_and_eligibility()
    dates = all_dates[(all_dates >= START) & (all_dates <= END)]
    eligibility = all_eligibility.reindex(index=dates).astype(bool)
    daily = u1_grid._read_daily(all_eligibility.columns)
    returns = u1_grid._native_returns(daily, FREQUENCY)
    baseline, _ = u1_grid._load_partition(FREQUENCY, SPAN)
    baseline = baseline.reindex(index=dates, columns=eligibility.columns)
    model = u1_grid._model(SPAN, FREQUENCY).copy(
        kwargs={
            "warmup_period": None,
            "cluster_smoother_type": ClusterSmootherType.PARTITION_BONUS,
            "smoother_delta": DELTA,
        }
    )
    return returns, pd.DatetimeIndex(dates), eligibility, baseline, model


def _fingerprint(
    returns: pd.DataFrame,
    dates: pd.DatetimeIndex,
    eligibility: pd.DataFrame,
    baseline: pd.DataFrame,
    model: object,
) -> str:
    """Return the frozen data-and-specification fingerprint."""

    payload = {
        "version": CACHE_VERSION,
        "frequency": FREQUENCY,
        "span": SPAN,
        "delta": DELTA,
        "dates": [str(date) for date in dates],
        "returns": _frame_hash(returns),
        "eligibility": _frame_hash(eligibility),
        "baseline": _frame_hash(baseline),
        "model": repr(model),
        "baseline_file_sha256": _sha256(_baseline_path()),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _cache_path(date: pd.Timestamp) -> Path:
    """Return one per-date pickle path."""

    return _cache_dir() / f"{date:%Y%m%d}.pkl"


def _load_cached(
    dates: pd.DatetimeIndex,
    fingerprint: str,
    columns: pd.Index,
) -> pd.DataFrame | None:
    """Return the complete valid cached membership panel, if present."""

    panel = pd.DataFrame(np.nan, index=dates, columns=columns, dtype=object)
    for date in dates:
        path = _cache_path(date)
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
        labels = payload["clusters"].dropna()
        panel.loc[date, labels.index] = labels.to_numpy()
    return panel


def _injected_matches(model: object, payload: dict[str, object]) -> bool:
    """Check the estimator's external-partition preparation path."""

    clusters = payload["clusters"].dropna()
    if len(clusters) < 2:
        return True
    dates = pd.date_range("2000-01-03", periods=4, freq="B")
    factor = pd.DataFrame(
        {"injection_check_factor": [-0.02, 0.01, 0.03, -0.01]},
        index=dates,
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
        external_linkage=payload["linkage"],
        external_cutoff=payload["cutoff"],
    )
    return _same_partition(prepared.asset_clusters, clusters)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write one deterministic audit CSV."""

    frame.to_csv(path, index=False, float_format="%.17g", lineterminator="\n")


def run() -> dict[str, pd.DataFrame]:
    """Build or validate the complete owner-authorised adopted-cell cache."""

    returns, dates, eligibility, baseline, model = _inputs()
    fingerprint = _fingerprint(returns, dates, eligibility, baseline, model)
    cache_dir = _cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    panel = _load_cached(dates, fingerprint, eligibility.columns)
    cache_status = "hit"
    if panel is None:
        cache_status = "miss_rebuilt"
        rolling = compute_rolling_smoothed_clusters(
            returns,
            list(dates),
            model,
            eligibility=eligibility,
        )
        panel = pd.DataFrame(
            {
                date: rolling.clusters.get(date, pd.Series(dtype=float))
                for date in dates
            }
        ).T.reindex(index=dates, columns=eligibility.columns)
        for date in dates:
            labels = panel.loc[date].dropna()
            payload = {
                "version": CACHE_VERSION,
                "fingerprint": fingerprint,
                "date": pd.Timestamp(date),
                "clusters": labels,
                "linkage": rolling.linkages.get(date, np.empty((0, 4))),
                "cutoff": rolling.cutoffs.get(date, 0.0),
            }
            payload["injected_partition_matches"] = _injected_matches(model, payload)
            path = _cache_path(date)
            temporary = path.with_suffix(".tmp")
            with temporary.open("wb") as stream:
                pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
            temporary.replace(path)

    baseline_asset_sets = baseline.notna().eq(eligibility)
    smoothed_asset_sets = panel.notna().eq(eligibility)
    injected = []
    for date in dates:
        with _cache_path(date).open("rb") as stream:
            injected.append(bool(pickle.load(stream)["injected_partition_matches"]))
    acceptance = pd.DataFrame(
        [
            {
                "check": "snapshot_count",
                "measured": len(panel),
                "tolerance": len(dates),
                "status": "PASS" if len(panel) == len(dates) else "FAIL",
            },
            {
                "check": "baseline_asset_set_match_share",
                "measured": float(baseline_asset_sets.all(axis=1).mean()),
                "tolerance": 1.0,
                "status": "PASS" if baseline_asset_sets.to_numpy().all() else "FAIL",
            },
            {
                "check": "smoothed_asset_set_match_share",
                "measured": float(smoothed_asset_sets.all(axis=1).mean()),
                "tolerance": 1.0,
                "status": "PASS" if smoothed_asset_sets.to_numpy().all() else "FAIL",
            },
            {
                "check": "injected_partition_match_share",
                "measured": float(np.mean(injected)),
                "tolerance": 1.0,
                "status": "PASS" if all(injected) else "FAIL",
            },
            {
                "check": "smoother_delta",
                "measured": float(model.smoother_delta),
                "tolerance": DELTA,
                "status": (
                    "PASS" if abs(float(model.smoother_delta) - DELTA) <= TOLERANCE else "FAIL"
                ),
            },
        ]
    )
    if not acceptance["status"].eq("PASS").all():
        raise AssertionError(acceptance.loc[~acceptance["status"].eq("PASS")])
    manifest = pd.DataFrame(
        [
            {
                "cache_version": CACHE_VERSION,
                "fingerprint": fingerprint,
                "frequency": FREQUENCY,
                "span": SPAN,
                "smoother": ClusterSmootherType.PARTITION_BONUS.name,
                "delta": DELTA,
                "dates": len(dates),
                "assets": len(eligibility.columns),
                "start": dates.min(),
                "end": dates.max(),
                "baseline_path": str(_baseline_path()),
            }
        ]
    )
    _write_csv(acceptance, cache_dir / "acceptance.csv")
    _write_csv(manifest, cache_dir / "manifest.csv")
    print(f"cache_status={cache_status} dates={len(dates)} fingerprint={fingerprint}")
    return {"acceptance": acceptance, "manifest": manifest, "panel": panel}


def _artifact_hashes() -> dict[str, str]:
    """Hash every deterministic cache artifact."""

    return {
        path.name: _sha256(path)
        for path in sorted(_cache_dir().glob("*"))
        if path.is_file() and path.name != "determinism.csv"
    }


def verify_determinism() -> pd.DataFrame:
    """Require a cache-first replay to leave every artifact byte-identical."""

    run()
    first = _artifact_hashes()
    run()
    second = _artifact_hashes()
    names = sorted(set(first) | set(second))
    replay = pd.DataFrame(
        {
            "artifact": names,
            "first_sha256": [first.get(name) for name in names],
            "second_sha256": [second.get(name) for name in names],
            "byte_identical": [first.get(name) == second.get(name) for name in names],
        }
    )
    if not replay["byte_identical"].all():
        raise AssertionError(replay.loc[~replay["byte_identical"]])
    _write_csv(replay, _cache_dir() / "determinism.csv")
    return replay


def main() -> None:
    """Build and deterministically replay the adopted-cell cache."""

    replay = verify_determinism()
    print(f"U1 ME/36 delta 0.0866 cache: PASS ({len(replay)}/{len(replay)})")


if __name__ == "__main__":
    main()
