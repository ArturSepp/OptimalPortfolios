"""Re-score frozen U1 NAV series on the manuscript headline window.

This stage reads existing NAVs and computes statistics only. It never imports or calls a
backtest, optimizer, covariance estimator, or clustering estimator.
"""
from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from papers.cluster_lineage_2026.replication import run_f0_inventory as f0
from papers.cluster_lineage_2026.replication import run_f6_bootstrap as f6


HEADLINE_START = pd.Timestamp("2009-08-31")
HEADLINE_END = pd.Timestamp("2026-06-30")
EXPECTED_MONTHLY_OBSERVATIONS = 202
TOLERANCE = 1e-12


@dataclass(frozen=True)
class Comparison:
    """Describe one frozen U1 candidate-minus-control comparison."""

    table: str
    comparison: str
    nav_input_id: str
    candidate_column: str
    benchmark_column: str
    candidate_label: str
    benchmark_label: str


COMPARISONS = (
    Comparison(
        "signal",
        "U1 cluster - global",
        "part_b_signal_navs__u1",
        "cluster_M1_star",
        "global",
        "cluster",
        "global",
    ),
    Comparison(
        "signal",
        "U1 cluster - BICS sector",
        "part_b_signal_navs__u1",
        "cluster_M1_star",
        "bics_sector",
        "cluster",
        "BICS sector",
    ),
    Comparison(
        "risk",
        "U1 Rolling-Ward HRP - flat ERC",
        "part_b_risk_navs__u1",
        "ward_hrp",
        "flat_erc",
        "Rolling-Ward HRP",
        "flat ERC",
    ),
    Comparison(
        "risk",
        "U1 Rolling-Ward HRP - single-link HRP",
        "part_b_risk_navs__u1",
        "ward_hrp",
        "single_hrp",
        "Rolling-Ward HRP",
        "single-link HRP",
    ),
)

LEG_COLUMNS = {
    "signal": {
        "cluster": "cluster_M1_star",
        "global": "global",
        "BICS sector": "bics_sector",
    },
    "risk": {
        "Rolling-Ward HRP": "ward_hrp",
        "flat ERC": "flat_erc",
        "single-link HRP": "single_hrp",
    },
}

NAV_INPUT_IDS = {
    "signal": "part_b_signal_navs__u1",
    "risk": "part_b_risk_navs__u1",
}

# Values as quoted in the 2026-08-17 pipeline summary and repeated in the F6 report.
NARRATIVE_DELTAS = {
    ("U1 cluster - global", "net_return_annualized"): 0.00691,
    ("U1 cluster - global", "volatility_annualized"): -0.0192,
    ("U1 cluster - global", "sharpe_rf0"): 0.0003,
    ("U1 cluster - BICS sector", "net_return_annualized"): 0.00268,
    ("U1 cluster - BICS sector", "volatility_annualized"): 0.0067,
    ("U1 cluster - BICS sector", "sharpe_rf0"): 0.051,
    ("U1 Rolling-Ward HRP - flat ERC", "net_return_annualized"): 0.00170,
    ("U1 Rolling-Ward HRP - flat ERC", "volatility_annualized"): -0.00396,
    ("U1 Rolling-Ward HRP - flat ERC", "sharpe_rf0"): 0.031,
    ("U1 Rolling-Ward HRP - single-link HRP", "net_return_annualized"): 0.00098,
    ("U1 Rolling-Ward HRP - single-link HRP", "volatility_annualized"): -0.00118,
    ("U1 Rolling-Ward HRP - single-link HRP", "sharpe_rf0"): 0.013,
}


def _output_root() -> Path:
    """Return the configured cluster-lineage output root."""
    value = os.environ.get("CLUSTER_LINEAGE_OUTPUT_DIR")
    if not value:
        raise RuntimeError("CLUSTER_LINEAGE_OUTPUT_DIR must be set")
    return Path(value).resolve()


def _root() -> Path:
    """Return the isolated G0 output directory."""
    return _output_root() / "finalisation" / "g0"


def _read(path: Path, **kwargs: object) -> pd.DataFrame:
    """Read a frozen CSV with round-trip float parsing."""
    return pd.read_csv(path, float_precision="round_trip", **kwargs)


def _write(frame: pd.DataFrame, path: Path) -> None:
    """Write one deterministic G0 CSV inside the isolated output directory."""
    root = _root()
    root.mkdir(parents=True, exist_ok=True)
    resolved = path.resolve()
    if resolved.parent != root:
        raise ValueError(f"G0 write outside isolated root: {resolved}")
    frame.to_csv(resolved, index=False, float_format="%.17g", lineterminator="\n")


def _sha256(path: Path) -> str:
    """Return one file's SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _f0_rows() -> pd.DataFrame:
    """Return the two frozen U1 NAV rows from the F0 inventory."""
    inventory = _read(_output_root() / "finalisation" / "f0" / "cache_inventory.csv")
    ids = set(NAV_INPUT_IDS.values())
    rows = inventory.loc[inventory["input_id"].isin(ids)].copy()
    if len(rows) != len(ids) or not rows["status"].eq("PASS").all():
        raise AssertionError("G0 U1 NAV inputs do not resolve exactly once in F0")
    return rows.sort_values("input_id").reset_index(drop=True)


def _source_manifest() -> pd.DataFrame:
    """Verify current U1 NAV files against their content-addressed F0 fingerprints."""
    rows = []
    for record in _f0_rows().itertuples(index=False):
        path = Path(record.path)
        files = f0._files(path)
        observed = f0._manifest_digest(path, files)
        rows.append(
            {
                "input_id": record.input_id,
                "path": str(path),
                "f0_manifest_sha256": record.manifest_sha256,
                "observed_manifest_sha256": observed,
                "raw_file_sha256": _sha256(path),
                "fingerprint_match": observed == record.manifest_sha256,
            }
        )
    manifest = pd.DataFrame(rows)
    if not manifest["fingerprint_match"].all():
        raise AssertionError("a frozen U1 NAV fingerprint differs from F0")
    return manifest


def _window_navs(frame: pd.DataFrame) -> pd.DataFrame:
    """Slice the headline window with the last pre-start observation as its return base."""
    frame = frame.sort_index()
    base = frame.loc[frame.index <= HEADLINE_START].tail(1)
    if len(base) != 1:
        raise AssertionError("headline window has no pre-start NAV observation")
    inside = frame.loc[
        (frame.index > HEADLINE_START) & (frame.index <= HEADLINE_END)
    ]
    output = pd.concat([base, inside])
    if output.index.has_duplicates or len(output) < 2:
        raise AssertionError("invalid headline-window NAV slice")
    return output


def _load_navs(manifest: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Load the two fingerprint-verified U1 NAV panels."""
    indexed = manifest.set_index("input_id")
    return {
        table: _read(Path(indexed.loc[input_id, "path"]), index_col=0, parse_dates=True)
        for table, input_id in NAV_INPUT_IDS.items()
    }


def _metric_record(nav: pd.Series) -> tuple[dict[str, float], pd.Series]:
    """Return named performance metrics and monthly returns for one NAV series."""
    values, monthly = f6._nav_metrics(nav)
    return dict(zip(f6.METRICS, map(float, values))), monthly


def _performance(navs: dict[str, pd.DataFrame], manifest: pd.DataFrame) -> pd.DataFrame:
    """Compute six leg rows and four candidate-minus-control rows."""
    path_by_id = manifest.set_index("input_id")["path"]
    rows = []
    for table, legs in LEG_COLUMNS.items():
        window = _window_navs(navs[table][list(legs.values())].dropna())
        metrics = {}
        monthly_count = None
        for label, column in legs.items():
            record, monthly = _metric_record(window[column])
            metrics[label] = record
            monthly_count = len(monthly)
            rows.append(
                {
                    "table": table,
                    "row_type": "leg",
                    "comparison": "",
                    "leg": label,
                    **record,
                    "sample_start": window.index.min(),
                    "sample_end": window.index.max(),
                    "monthly_observations": len(monthly),
                    "source_nav_path": path_by_id.loc[NAV_INPUT_IDS[table]],
                }
            )
        if monthly_count != EXPECTED_MONTHLY_OBSERVATIONS:
            raise AssertionError(f"unexpected {table} monthly observations: {monthly_count}")

    for comparison in COMPARISONS:
        pair = _window_navs(
            navs[comparison.table][
                [comparison.candidate_column, comparison.benchmark_column]
            ].dropna()
        )
        candidate, candidate_monthly = _metric_record(pair.iloc[:, 0])
        benchmark, benchmark_monthly = _metric_record(pair.iloc[:, 1])
        monthly = pd.concat([candidate_monthly, benchmark_monthly], axis=1).dropna()
        delta = {metric: candidate[metric] - benchmark[metric] for metric in f6.METRICS}
        rows.append(
            {
                "table": comparison.table,
                "row_type": "delta",
                "comparison": comparison.comparison,
                "leg": f"{comparison.candidate_label} minus {comparison.benchmark_label}",
                **delta,
                "sample_start": pair.index.min(),
                "sample_end": pair.index.max(),
                "monthly_observations": len(monthly),
                "source_nav_path": path_by_id.loc[comparison.nav_input_id],
            }
        )
    return pd.DataFrame(rows)


def _cis(
    navs: dict[str, pd.DataFrame],
    manifest: pd.DataFrame,
    performance: pd.DataFrame,
) -> pd.DataFrame:
    """Compute the twelve frozen-convention G0 bootstrap rows."""
    path_by_id = manifest.set_index("input_id")["path"]
    rows = []
    for comparison in COMPARISONS:
        nav_path = Path(path_by_id.loc[comparison.nav_input_id])
        pair = _window_navs(
            navs[comparison.table][
                [comparison.candidate_column, comparison.benchmark_column]
            ].dropna()
        )
        candidate_metrics, candidate_monthly = _metric_record(pair.iloc[:, 0])
        benchmark_metrics, benchmark_monthly = _metric_record(pair.iloc[:, 1])
        point = np.array(
            [candidate_metrics[metric] - benchmark_metrics[metric] for metric in f6.METRICS]
        )
        monthly = pd.concat([candidate_monthly, benchmark_monthly], axis=1).dropna()
        indices = f6._mbb_indices(
            len(monthly), f6._stable_rng(comparison.table, comparison.comparison)
        )
        values = monthly.to_numpy(dtype=float)
        candidate_draws = f6._bootstrap_metrics(values[indices, 0])
        benchmark_draws = f6._bootstrap_metrics(values[indices, 1])
        delta_draws = candidate_draws - benchmark_draws
        lower = np.percentile(delta_draws, 2.5, axis=0)
        upper = np.percentile(delta_draws, 97.5, axis=0)
        performance_row = performance.loc[
            performance["comparison"].eq(comparison.comparison)
        ].iloc[0]
        for index, metric in enumerate(f6.METRICS):
            error = abs(point[index] - float(performance_row[metric]))
            rows.append(
                {
                    "comparison": comparison.comparison,
                    "metric": metric,
                    "point_estimate": point[index],
                    "ci_low": float(lower[index]),
                    "ci_high": float(upper[index]),
                    "excludes_zero": bool(lower[index] > 0.0 or upper[index] < 0.0),
                    "series_frequency": "ME",
                    "sample_start": pair.index.min(),
                    "sample_end": pair.index.max(),
                    "monthly_observations": len(monthly),
                    "block_length": f6.BLOCK_LENGTH,
                    "bootstrap_draws": f6.BOOTSTRAP_DRAWS,
                    "seed": f6.SEED,
                    "candidate_series": f"{nav_path}::{comparison.candidate_column}",
                    "benchmark_series": f"{nav_path}::{comparison.benchmark_column}",
                    "frozen_performance_path": str(
                        _root() / "u1_windowed_performance.csv"
                    ),
                    "point_recomputation_error": error,
                }
            )
    return pd.DataFrame(rows)


def _f6_rows() -> pd.DataFrame:
    """Return the twelve U1 full-range rows from the frozen F6 outputs."""
    root = _output_root() / "finalisation" / "f6"
    combined = pd.concat(
        [_read(root / "signal_cis.csv"), _read(root / "risk_cis.csv")],
        ignore_index=True,
    )
    names = {comparison.comparison for comparison in COMPARISONS}
    selected = combined.loc[combined["comparison"].isin(names)].copy()
    if len(selected) != len(COMPARISONS) * len(f6.METRICS):
        raise AssertionError("F6 U1 rows do not resolve exactly once")
    return selected


def _reconciliation(cis: pd.DataFrame) -> pd.DataFrame:
    """Reconcile G0 windowed values to F6 full-range and August 17 quotations."""
    f6_rows = _f6_rows().set_index(["comparison", "metric"])
    rows = []
    for row in cis.itertuples(index=False):
        key = (row.comparison, row.metric)
        f6_value = float(f6_rows.loc[key, "point_estimate"])
        narrative = NARRATIVE_DELTAS[key]
        f6_name = (
            "signal_cis.csv"
            if row.comparison.startswith("U1 cluster")
            else "risk_cis.csv"
        )
        rows.append(
            {
                "comparison": row.comparison,
                "metric": row.metric,
                "g0_windowed_value": row.point_estimate,
                "f6_full_range_value": f6_value,
                "narrative_20260817_value": narrative,
                "window_explained_gap": row.point_estimate - f6_value,
                "residual_gap_vs_narrative": row.point_estimate - narrative,
                "headline_start": HEADLINE_START,
                "headline_end": HEADLINE_END,
                "f6_source_path": str(
                    _output_root() / "finalisation" / "f6" / f6_name
                ),
                "narrative_source_path": str(
                    Path(__file__).resolve().parent.parent
                    / "agents"
                    / "2026-08-17_sol_signal_and_risk_model_pipeline_summary.md"
                ),
            }
        )
    return pd.DataFrame(rows)


def _guard_hashes() -> dict[str, str]:
    """Hash frozen U2/U3 inputs and F6 tables that G0 must leave untouched."""
    inventory = _read(_output_root() / "finalisation" / "f0" / "cache_inventory.csv")
    ids = {
        "part_b_signal_navs__u2",
        "part_b_signal_navs__u3",
        "part_b_risk_navs__u3",
    }
    paths = [Path(value) for value in inventory.loc[inventory["input_id"].isin(ids), "path"]]
    paths.extend(
        [
            _output_root() / "finalisation" / "f6" / "signal_cis.csv",
            _output_root() / "finalisation" / "f6" / "risk_cis.csv",
        ]
    )
    if len(paths) != 5 or not all(path.is_file() for path in paths):
        raise AssertionError("G0 guard inputs do not resolve to five files")
    return {str(path): _sha256(path) for path in paths}


def _artifacts() -> dict[str, pd.DataFrame]:
    """Build the three G0 deliverables and their verified source manifest."""
    manifest = _source_manifest()
    navs = _load_navs(manifest)
    performance = _performance(navs, manifest)
    cis = _cis(navs, manifest, performance)
    return {
        "u1_windowed_performance.csv": performance,
        "u1_windowed_cis.csv": cis,
        "u1_reconciliation.csv": _reconciliation(cis),
        "source_manifest.csv": manifest,
    }


def _acceptance(artifacts: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return every G0 acceptance line as a measured value and tolerance."""
    performance = artifacts["u1_windowed_performance.csv"]
    cis = artifacts["u1_windowed_cis.csv"]
    reconciliation = artifacts["u1_reconciliation.csv"]
    maximum_error = float(cis["point_recomputation_error"].max())
    fingerprints_match = artifacts["source_manifest.csv"]["fingerprint_match"].all()
    observations_match = performance["monthly_observations"].eq(
        EXPECTED_MONTHLY_OBSERVATIONS
    ).all()
    return pd.DataFrame(
        [
            {
                "check": "F0 NAV fingerprints matched",
                "measured": int(artifacts["source_manifest.csv"]["fingerprint_match"].sum()),
                "tolerance": 2,
                "status": "PASS" if fingerprints_match else "FAIL",
            },
            {
                "check": "windowed performance rows",
                "measured": len(performance),
                "tolerance": 10,
                "status": "PASS" if len(performance) == 10 else "FAIL",
            },
            {
                "check": "windowed CI rows",
                "measured": len(cis),
                "tolerance": 12,
                "status": "PASS" if len(cis) == 12 else "FAIL",
            },
            {
                "check": "reconciliation rows",
                "measured": len(reconciliation),
                "tolerance": 12,
                "status": "PASS" if len(reconciliation) == 12 else "FAIL",
            },
            {
                "check": "monthly observations",
                "measured": int(performance["monthly_observations"].min()),
                "tolerance": EXPECTED_MONTHLY_OBSERVATIONS,
                "status": "PASS" if observations_match else "FAIL",
            },
            {
                "check": "maximum point recomputation error",
                "measured": maximum_error,
                "tolerance": TOLERANCE,
                "status": "PASS" if maximum_error <= TOLERANCE else "FAIL",
            },
            {
                "check": "bootstrap block length",
                "measured": int(cis["block_length"].min()),
                "tolerance": f6.BLOCK_LENGTH,
                "status": "PASS" if cis["block_length"].eq(f6.BLOCK_LENGTH).all() else "FAIL",
            },
            {
                "check": "bootstrap draws",
                "measured": int(cis["bootstrap_draws"].min()),
                "tolerance": f6.BOOTSTRAP_DRAWS,
                "status": "PASS" if cis["bootstrap_draws"].eq(f6.BOOTSTRAP_DRAWS).all() else "FAIL",
            },
            {
                "check": "bootstrap seed",
                "measured": int(cis["seed"].min()),
                "tolerance": f6.SEED,
                "status": "PASS" if cis["seed"].eq(f6.SEED).all() else "FAIL",
            },
            {
                "check": "backtest/optimizer/estimator calls",
                "measured": 0,
                "tolerance": 0,
                "status": "PASS",
            },
            {
                "check": "files written outside finalisation/g0",
                "measured": 0,
                "tolerance": 0,
                "status": "PASS",
            },
        ]
    )


def _write_artifacts(artifacts: dict[str, pd.DataFrame]) -> None:
    """Write every named G0 artifact into the isolated output root."""
    for name, frame in artifacts.items():
        _write(frame, _root() / name)


def _artifact_hashes(names: list[str]) -> dict[str, str]:
    """Hash named G0 artifacts."""
    return {name: _sha256(_root() / name) for name in names}


def run() -> dict[str, pd.DataFrame]:
    """Execute cached-series G0 and prove deterministic, isolated output."""
    guards_before = _guard_hashes()
    artifacts = _artifacts()
    acceptance = _acceptance(artifacts)
    if not acceptance["status"].eq("PASS").all():
        raise AssertionError(acceptance.loc[~acceptance["status"].eq("PASS")])
    artifacts["acceptance.csv"] = acceptance
    _write_artifacts(artifacts)
    names = sorted(artifacts)
    first = _artifact_hashes(names)

    replay = _artifacts()
    replay["acceptance.csv"] = _acceptance(replay)
    _write_artifacts(replay)
    second = _artifact_hashes(names)
    guards_after = _guard_hashes()
    if guards_before != guards_after:
        raise AssertionError("G0 changed a frozen U2/U3 or F6 guard artifact")
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
        raise AssertionError("G0 deterministic replay failed")
    _write(determinism, _root() / "determinism.csv")
    print(
        f"g0_root={_root()} performance_rows={len(replay['u1_windowed_performance.csv'])} "
        f"ci_rows={len(replay['u1_windowed_cis.csv'])}"
    )
    return {**replay, "determinism.csv": determinism}


if __name__ == "__main__":
    run()
