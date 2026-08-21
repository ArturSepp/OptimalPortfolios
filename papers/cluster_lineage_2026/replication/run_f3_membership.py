"""Consolidate frozen membership, lineage, and interpretability evidence for F3.

The runner reads the corrected E3b stability tables, the accepted E4 lineage reports, and
the two adopted application-cell partition caches.  It never fits a covariance estimator or
clusterer.  Taxonomy deltas and levels are encoded as sorted JSON objects so the consolidated
tables remain rectangular without inventing values for taxonomies that do not apply.
"""

from __future__ import annotations

import hashlib
import json
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

import papers.cluster_lineage_2026.replication.run_backtests as e5


PANELS = {
    "equity_panel": "msci_us",
    "futures_panel": "futures",
    "fund_panel": "mac",
}
ADOPTED_INTERPRETABILITY = {
    "equity_panel": ("msci_us", "M1_delta_0.02"),
    "futures_panel": ("futures", "M1_star"),
    "fund_panel": ("mac", "M1_delta_0.05"),
}
TAXONOMIES = {
    "equity_panel": ("gics_sector", "gics_industry_group", "gics_industry"),
    "futures_panel": ("asset_class", "ac_geography"),
}
HEADLINE_START = pd.Timestamp("2009-08-31")
HEADLINE_END = pd.Timestamp("2026-06-30")
TOLERANCE = 1e-12


def _output_root() -> Path:
    """Return the configured external output root."""
    value = os.environ.get("CLUSTER_LINEAGE_OUTPUT_DIR")
    if not value:
        raise RuntimeError("CLUSTER_LINEAGE_OUTPUT_DIR must be set")
    return Path(value).resolve()


def _root() -> Path:
    """Return the isolated F3 output directory."""
    root = _output_root() / "finalisation" / "f3"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _read(path: Path, **kwargs: object) -> pd.DataFrame:
    """Read one frozen CSV with round-trip float parsing."""
    return pd.read_csv(path, float_precision="round_trip", **kwargs)


def _write(frame: pd.DataFrame, path: Path) -> None:
    """Write one deterministic high-precision CSV."""
    frame.to_csv(path, index=False, float_format="%.17g", lineterminator="\n")


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sorted_json(values: dict[str, float]) -> str:
    """Serialise a numerical mapping deterministically without NaNs."""
    return json.dumps(
        {key: float(values[key]) for key in sorted(values)},
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def _adjusted_rand_index(labels_a: pd.Series, labels_b: pd.Series) -> float:
    """Return the frozen metrics-registry ARI on common non-null assets."""
    frame = pd.concat([labels_a, labels_b], axis=1).dropna()
    contingency = pd.crosstab(frame.iloc[:, 0], frame.iloc[:, 1]).to_numpy(dtype=float)
    n = contingency.sum()
    if n < 2:
        return np.nan
    observed = np.sum(contingency * (contingency - 1.0) / 2.0)
    row_counts = contingency.sum(axis=1)
    column_counts = contingency.sum(axis=0)
    rows = np.sum(row_counts * (row_counts - 1.0) / 2.0)
    columns = np.sum(column_counts * (column_counts - 1.0) / 2.0)
    total = n * (n - 1.0) / 2.0
    expected = rows * columns / total
    maximum = 0.5 * (rows + columns)
    return float((observed - expected) / (maximum - expected)) if maximum != expected else 1.0


def _taxonomy_ari(
    partitions: dict[pd.Timestamp, pd.Series],
    metadata: pd.DataFrame,
    taxonomy_columns: tuple[str, ...],
) -> dict[str, float]:
    """Return median taxonomy ARI using the frozen registry convention."""
    output = {}
    for column in taxonomy_columns:
        key = column.removeprefix("gics_").replace(" ", "_").lower()
        values = [
            _adjusted_rand_index(partitions[date], metadata[column])
            for date in sorted(partitions)
        ]
        output[key] = float(pd.Series(values).median())
    return output


def _fidelity_band(
    candidate: dict[pd.Timestamp, pd.Series],
    baseline: dict[pd.Timestamp, pd.Series],
    metadata: pd.DataFrame,
    taxonomy_columns: tuple[str, ...],
) -> dict[str, object]:
    """Return the frozen 0.03-taxonomy/0.15-count fidelity verdict."""
    dates = sorted(set(candidate).intersection(baseline))
    same_date = pd.Series(
        [_adjusted_rand_index(candidate[date], baseline[date]) for date in dates]
    )
    candidate_ari = _taxonomy_ari(candidate, metadata, taxonomy_columns)
    baseline_ari = _taxonomy_ari(baseline, metadata, taxonomy_columns)
    deltas = {key: candidate_ari[key] - baseline_ari[key] for key in candidate_ari}
    candidate_count = float(
        np.median([candidate[date].dropna().nunique() for date in dates])
    )
    baseline_count = float(
        np.median([baseline[date].dropna().nunique() for date in dates])
    )
    count_change = candidate_count / baseline_count - 1.0
    passed = all(abs(value) <= 0.03 for value in deltas.values())
    passed &= abs(count_change) <= 0.15
    return {
        "baseline_partition_ari_median": float(same_date.median()),
        **{f"delta_ari_{key}": float(value) for key, value in deltas.items()},
        "cluster_count_relative_change": count_change,
        "fidelity_status": "PASS" if passed else "REJECTED",
    }


def _f0_sources() -> pd.DataFrame:
    """Return the exact F0 registry rows consumed by this stage."""
    inventory = _read(_output_root() / "finalisation" / "f0" / "cache_inventory.csv")
    ids = {
        "e3b_stability_evidence__equity_panel",
        "e3b_stability_evidence__futures_panel",
        "e3b_stability_evidence__fund_panel",
        "e4_interpretability",
        "adopted_equity_me36_baseline",
        "adopted_equity_me36_delta_0.0866",
        "adopted_futures_wwed156_baseline",
        "adopted_futures_wwed156_delta_0.0691",
    }
    selected = inventory.loc[inventory["input_id"].isin(ids)].copy()
    if len(selected) != len(ids) or not selected["status"].eq("PASS").all():
        raise AssertionError("F3 inputs are not uniquely resolved and green in F0")
    return selected.loc[
        :, ["input_id", "path", "manifest_sha256", "status"]
    ].sort_values("input_id").reset_index(drop=True)


def _churn_fidelity() -> pd.DataFrame:
    """Consolidate corrected E3b churn and fidelity rows."""
    rows: list[dict[str, object]] = []
    for panel, source in PANELS.items():
        table = _read(_output_root() / "stability" / source / "metric_suite.csv")
        delta_columns = sorted(
            column for column in table if column.startswith("delta_ari_")
        )
        for _, record in table.iterrows():
            deltas = {
                column.removeprefix("delta_ari_"): float(record[column])
                for column in delta_columns
            }
            rows.append(
                {
                    "panel": panel,
                    "analysis_window": record["analysis_window"],
                    "config": record["config"],
                    "raw_churn": float(record["raw_churn"]),
                    "lineage_churn": float(record["lineage_churn_panel"]),
                    "median_same_date_ari": float(
                        record["baseline_partition_ari_median"]
                    ),
                    "taxonomy_delta_ari_by_level": _sorted_json(deltas),
                    "maximum_absolute_taxonomy_delta_ari": max(map(abs, deltas.values())),
                    "cluster_count_relative_change": float(
                        record["cluster_count_relative_change"]
                    ),
                    "fidelity_status": record["fidelity_status"],
                    "source_path": str(
                        _output_root() / "stability" / source / "metric_suite.csv"
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(
        ["panel", "analysis_window", "config"]
    ).reset_index(drop=True)


def _panel_mapping(frame: pd.DataFrame) -> dict[pd.Timestamp, pd.Series]:
    """Convert one membership panel to a date-to-partition mapping."""
    return {
        pd.Timestamp(date): row.dropna()
        for date, row in frame.sort_index().iterrows()
    }


def _read_snapshot_partitions(path: Path) -> dict[pd.Timestamp, pd.Series]:
    """Read per-date factor-covariance or adopted-cache snapshots as partitions."""
    output: dict[pd.Timestamp, pd.Series] = {}
    for file in sorted(path.glob("????????.pkl")):
        with file.open("rb") as stream:
            payload = pickle.load(stream)
        date = pd.Timestamp(file.stem)
        labels = payload["clusters"] if isinstance(payload, dict) else payload.clusters
        output[date] = labels.dropna()
    return output


def _fidelity_row(
    *,
    application_cell: str,
    panel: str,
    baseline: dict[pd.Timestamp, pd.Series],
    candidate: dict[pd.Timestamp, pd.Series],
    metadata: pd.DataFrame,
    taxonomy_columns: tuple[str, ...],
    baseline_path: Path,
    candidate_path: Path,
    baseline_fingerprint: str,
    candidate_fingerprint: str,
) -> dict[str, object]:
    """Score one adopted cell against its own cached baseline."""
    scored = _fidelity_band(candidate, baseline, metadata, taxonomy_columns)
    deltas = {
        key.removeprefix("delta_ari_"): float(value)
        for key, value in scored.items()
        if key.startswith("delta_ari_")
    }
    return {
        "application_cell": application_cell,
        "panel": panel,
        "dates": len(set(baseline).intersection(candidate)),
        "median_same_date_ari": float(scored["baseline_partition_ari_median"]),
        "taxonomy_delta_ari_by_level": _sorted_json(deltas),
        "maximum_absolute_taxonomy_delta_ari": max(map(abs, deltas.values())),
        "cluster_count_relative_change": float(scored["cluster_count_relative_change"]),
        "fidelity_status": scored["fidelity_status"],
        "baseline_cache_path": str(baseline_path),
        "baseline_cache_fingerprint": baseline_fingerprint,
        "candidate_cache_path": str(candidate_path),
        "candidate_cache_fingerprint": candidate_fingerprint,
    }


def _adopted_verdicts(sources: pd.DataFrame) -> pd.DataFrame:
    """Re-score the two adopted application cells from cached partitions."""
    by_id = sources.set_index("input_id")
    u1_base_path = Path(by_id.loc["adopted_equity_me36_baseline", "path"])
    with u1_base_path.open("rb") as stream:
        u1_base_panel = pickle.load(stream)["panel"]
    u1_base_panel = u1_base_panel.loc[
        u1_base_panel.index.to_series().between(HEADLINE_START, HEADLINE_END).to_numpy()
    ]
    u1_candidate_path = Path(
        by_id.loc["adopted_equity_me36_delta_0.0866", "path"]
    )
    u1_candidate = _read_snapshot_partitions(u1_candidate_path)
    u1_candidate_internal = _read(u1_candidate_path / "manifest.csv").iloc[0][
        "fingerprint"
    ]
    u1_data = e5.load_universe(e5.UniverseName.MSCI_US)

    u3_base_path = Path(by_id.loc["adopted_futures_wwed156_baseline", "path"])
    u3_candidate_path = Path(
        by_id.loc["adopted_futures_wwed156_delta_0.0691", "path"]
    )
    u3_data = e5.load_universe(e5.UniverseName.FUTURES)
    rows = [
        _fidelity_row(
            application_cell="U1_ME_span36_delta_0.0866",
            panel="equity_panel",
            baseline=_panel_mapping(u1_base_panel),
            candidate=u1_candidate,
            metadata=u1_data.taxonomy,
            taxonomy_columns=TAXONOMIES["equity_panel"],
            baseline_path=u1_base_path,
            candidate_path=u1_candidate_path,
            baseline_fingerprint=by_id.loc[
                "adopted_equity_me36_baseline", "manifest_sha256"
            ],
            candidate_fingerprint=(
                f"f0:{by_id.loc['adopted_equity_me36_delta_0.0866', 'manifest_sha256']}"
                f";internal:{u1_candidate_internal}"
            ),
        ),
        _fidelity_row(
            application_cell="U3_W-WED_span156_delta_0.0691",
            panel="futures_panel",
            baseline=_read_snapshot_partitions(u3_base_path),
            candidate=_read_snapshot_partitions(u3_candidate_path),
            metadata=u3_data.taxonomy,
            taxonomy_columns=TAXONOMIES["futures_panel"],
            baseline_path=u3_base_path,
            candidate_path=u3_candidate_path,
            baseline_fingerprint=by_id.loc[
                "adopted_futures_wwed156_baseline", "manifest_sha256"
            ],
            candidate_fingerprint=by_id.loc[
                "adopted_futures_wwed156_delta_0.0691", "manifest_sha256"
            ],
        ),
    ]
    return pd.DataFrame(rows)


def _interpretability() -> pd.DataFrame:
    """Consolidate the accepted E4 metrics for baseline and adopted configurations."""
    root = _output_root() / "interpretability"
    rows: list[dict[str, object]] = []
    for panel, (universe, adopted) in ADOPTED_INTERPRETABILITY.items():
        table_path = (
            root / "msci_us" / "metric_set_12.csv"
            if universe == "msci_us"
            else root / "metric_set_12.csv"
        )
        table = _read(table_path)
        for config in ("baseline", adopted):
            record = table.loc[
                table["universe"].eq(universe) & table["config"].eq(config)
            ].iloc[0]
            ari = {
                column.removeprefix("ari_"): float(record[column])
                for column in table
                if column.startswith("ari_") and pd.notna(record[column])
            }
            peak_level = max(ari, key=ari.get)
            rows.append(
                {
                    "panel": panel,
                    "config": config,
                    "taxonomy_ari_by_level": _sorted_json(ari),
                    "taxonomy_ari_peak_level": peak_level,
                    "taxonomy_ari_peak": ari[peak_level],
                    "track_modal_taxonomy_purity": float(
                        record["track_modal_taxonomy_purity"]
                    ),
                    "modal_label_life_share": float(record["modal_label_life_share"]),
                    "label_churn_per_asset_year": float(record["label_string_churn"]),
                    "source_path": str(table_path),
                }
            )
    return pd.DataFrame(rows)


def _case_studies() -> pd.DataFrame:
    """Return three accepted, well-covered lineage-track summaries per panel."""
    root = _output_root() / "interpretability"
    tables = [
        (
            "equity_panel",
            root / "msci_us" / "case_study_tracks.csv",
            root / "msci_us" / "case_study_membership.csv",
        ),
        (
            "futures_panel",
            root / "case_study_tracks.csv",
            root / "case_study_membership.csv",
        ),
        (
            "fund_panel",
            root / "case_study_tracks.csv",
            root / "case_study_membership.csv",
        ),
    ]
    rows: list[dict[str, object]] = []
    for panel, track_path, member_path in tables:
        universe = PANELS[panel]
        tracks = _read(track_path)
        memberships = _read(member_path, parse_dates=["date"])
        tracks = tracks.loc[tracks["universe"].eq(universe)].copy()
        for _, track in tracks.iterrows():
            selected = memberships.loc[
                memberships["universe"].eq(universe)
                & memberships["config"].eq(track["config"])
                & memberships["derived_id"].eq(track["derived_id"])
            ]
            all_members: set[str] = set()
            for value in selected["members"].astype(str):
                all_members.update(value.split("|"))
            counts = selected["member_count"].astype(float)
            summary = (
                f"{len(selected)} dates; members/date "
                f"{int(counts.min())}/{float(counts.median()):g}/{int(counts.max())} "
                f"min/median/max; {len(all_members)} distinct assets"
            )
            rows.append(
                {
                    "panel": panel,
                    "config": track["config"],
                    "derived_id": track["derived_id"],
                    "start": track["start"],
                    "end": track["end"],
                    "coverage": float(track["coverage"]),
                    "lifetime_dates": int(track["lifetime"]),
                    "membership_path_summary": summary,
                    "modal_label": track["label"],
                    "track_source_path": str(track_path),
                    "membership_source_path": str(member_path),
                }
            )
    return pd.DataFrame(rows).sort_values(["panel", "config", "derived_id"]).reset_index(
        drop=True
    )


def _numeric_nan_count(frames: list[pd.DataFrame]) -> int:
    """Count missing values across every deliverable cell."""
    return int(sum(frame.isna().sum().sum() for frame in frames))


def _artifacts() -> dict[str, pd.DataFrame]:
    """Build all F3 tabular artifacts in memory."""
    sources = _f0_sources()
    return {
        "churn_fidelity.csv": _churn_fidelity(),
        "adopted_cell_verdicts.csv": _adopted_verdicts(sources),
        "interpretability.csv": _interpretability(),
        "case_study_tracks.csv": _case_studies(),
        "source_manifest.csv": sources,
    }


def _write_artifacts(artifacts: dict[str, pd.DataFrame]) -> None:
    """Write all F3 deliverables."""
    for name, frame in artifacts.items():
        _write(frame, _root() / name)


def _hash_artifacts(names: list[str]) -> dict[str, str]:
    """Hash named F3 outputs."""
    return {name: _sha256(_root() / name) for name in names}


def run() -> dict[str, pd.DataFrame]:
    """Execute F3, assert acceptance, and prove deterministic replay."""
    artifacts = _artifacts()
    churn = artifacts["churn_fidelity.csv"]
    adopted = artifacts["adopted_cell_verdicts.csv"]
    cases = artifacts["case_study_tracks.csv"]
    futures_frozen = churn.loc[
        churn["panel"].eq("futures_panel")
        & churn["analysis_window"].eq("full_panel")
        & churn["config"].eq("M1_star")
    ].iloc[0]
    futures_adopted = adopted.loc[adopted["panel"].eq("futures_panel")].iloc[0]
    overlap_error = max(
        abs(
            float(futures_adopted["median_same_date_ari"])
            - float(futures_frozen["median_same_date_ari"])
        ),
        abs(
            float(futures_adopted["cluster_count_relative_change"])
            - float(futures_frozen["cluster_count_relative_change"])
        ),
        abs(
            float(futures_adopted["maximum_absolute_taxonomy_delta_ari"])
            - float(futures_frozen["maximum_absolute_taxonomy_delta_ari"])
        ),
    )
    count_by_panel = cases.groupby("panel").size()
    acceptance = pd.DataFrame(
        [
            {
                "check": "F0 sources resolved once",
                "measured": len(artifacts["source_manifest.csv"]),
                "tolerance": 8,
                "status": "PASS" if len(artifacts["source_manifest.csv"]) == 8 else "FAIL",
            },
            {
                "check": "churn-fidelity rows",
                "measured": len(churn),
                "tolerance": 32,
                "status": "PASS" if len(churn) == 32 else "FAIL",
            },
            {
                "check": "adopted-cell rows",
                "measured": len(adopted),
                "tolerance": 2,
                "status": "PASS" if len(adopted) == 2 else "FAIL",
            },
            {
                "check": "published-overlap regression error",
                "measured": overlap_error,
                "tolerance": TOLERANCE,
                "status": "PASS" if overlap_error <= TOLERANCE else "FAIL",
            },
            {
                "check": "interpretability rows",
                "measured": len(artifacts["interpretability.csv"]),
                "tolerance": 6,
                "status": (
                    "PASS" if len(artifacts["interpretability.csv"]) == 6 else "FAIL"
                ),
            },
            {
                "check": "case-study tracks per panel",
                "measured": int(count_by_panel.min()),
                "tolerance": 3,
                "status": "PASS" if count_by_panel.eq(3).all() else "FAIL",
            },
            {
                "check": "minimum case-study coverage",
                "measured": float(cases["coverage"].min()),
                "tolerance": 0.7,
                "status": "PASS" if cases["coverage"].min() >= 0.7 else "FAIL",
            },
            {
                "check": "NaNs across deliverables",
                "measured": _numeric_nan_count(list(artifacts.values())),
                "tolerance": 0,
                "status": (
                    "PASS"
                    if _numeric_nan_count(list(artifacts.values())) == 0
                    else "FAIL"
                ),
            },
        ]
    )
    if not acceptance["status"].eq("PASS").all():
        raise AssertionError(acceptance.loc[~acceptance["status"].eq("PASS")])
    _write_artifacts(artifacts)
    _write(acceptance, _root() / "acceptance.csv")
    names = sorted([*artifacts, "acceptance.csv"])
    first = _hash_artifacts(names)
    replay = _artifacts()
    _write_artifacts(replay)
    _write(acceptance, _root() / "acceptance.csv")
    second = _hash_artifacts(names)
    determinism = pd.DataFrame(
        [
            {
                "artifact": name,
                "first_sha256": first[name],
                "second_sha256": second[name],
                "byte_identical": first[name] == second[name],
            }
            for name in names
        ]
    )
    if not determinism["byte_identical"].all():
        raise AssertionError("F3 deterministic replay failed")
    _write(determinism, _root() / "determinism.csv")
    print(f"f3_root={_root()} rows={len(churn)} adopted={len(adopted)}")
    return {**artifacts, "acceptance.csv": acceptance, "determinism.csv": determinism}


if __name__ == "__main__":
    run()
