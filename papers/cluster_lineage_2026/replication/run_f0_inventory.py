"""Freeze the provenance of every input required by finalisation stages F1--F6.

The inventory is deliberately read-only with respect to all source caches.  It hashes a
stable manifest for each input and writes only to ``finalisation/f0`` below the configured
cluster-lineage output root.  Missing inputs remain explicit rows so the roadmap's hard
escalation rule cannot be bypassed by a partial inventory.
"""

from __future__ import annotations

import csv
import hashlib
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


CONFIGS = (
    "baseline",
    "M0_quarterly_hold",
    "M1_delta_0.02",
    "M1_delta_0.05",
    "M1_delta_0.10",
    "M1_star",
    "M2_lambda_0.5",
    "M2_lambda_0.7",
)


@dataclass(frozen=True)
class RequiredInput:
    """Describe one path that later finalisation stages must consume."""

    input_id: str
    stages: str
    kind: str
    path: Path
    note: str = ""


def _repo_root() -> Path:
    """Return the OptimalPortfolios checkout root."""

    return Path(__file__).resolve().parents[3]


def _output_root() -> Path:
    """Return the configured cluster-lineage output root."""

    value = os.environ.get("CLUSTER_LINEAGE_OUTPUT_DIR")
    if not value:
        raise RuntimeError("CLUSTER_LINEAGE_OUTPUT_DIR must be set")
    return Path(value).resolve()


def required_inputs(output_root: Path, repo_root: Path) -> list[RequiredInput]:
    """Build the exhaustive F1--F6 input registry."""

    rows: list[RequiredInput] = []
    panels = {
        "equity_panel": "msci_us",
        "futures_panel": "futures",
        "fund_panel": "mac",
    }
    for panel, cache_name in panels.items():
        for config in CONFIGS:
            rows.append(
                RequiredInput(
                    f"e2e3b_cache__{panel}__{config}",
                    "F1|F2|F3|F5",
                    "directory",
                    output_root / cache_name / config,
                    "E3b-corrected cache is mandatory for equity smoothed configurations",
                )
            )

    for panel, cache_name in panels.items():
        rows.extend(
            (
                RequiredInput(
                    f"e3b_stability_evidence__{panel}",
                    "F1|F2|F3|F5",
                    "directory",
                    output_root / "stability" / cache_name,
                ),
                RequiredInput(
                    f"e3b_workbook__{panel}",
                    "F1|F2|F3|F5",
                    "file",
                    output_root
                    / "stability"
                    / cache_name
                    / f"{cache_name}_stability_20260813.xlsx",
                ),
            )
        )

    rows.extend(
        (
            RequiredInput(
                "e1_data_quality",
                "F1|F8",
                "directory",
                output_root / "data_quality",
            ),
            RequiredInput(
                "e2_rho_bar",
                "F1",
                "file",
                output_root / "e2_baseline_rho_bar.csv",
            ),
            RequiredInput(
                "e2_runtime_cache",
                "F0",
                "file",
                output_root / "e2_runtime_cache.csv",
            ),
            RequiredInput(
                "e3b_corrected_frequency_scaling",
                "F2",
                "file",
                output_root / "stability" / "corrected_frequency_scaling.csv",
            ),
            RequiredInput(
                "e4_interpretability",
                "F3|F5",
                "directory",
                output_root / "interpretability",
            ),
            RequiredInput(
                "e6_inference",
                "F1|F2|F3|F5",
                "directory",
                output_root / "inference",
            ),
        )
    )

    for construction, prefix in (("asset_equal", "backtests"), ("group_equal", "e5b/group_equal")):
        for panel in panels.values():
            rows.append(
                RequiredInput(
                    f"e5_turnover__{construction}__{panel}",
                    "F5",
                    "file",
                    output_root / Path(prefix) / panel / "turnover_decomposition_per_date.csv",
                )
            )

    risk_dirs = {
        "u1": output_root / "risk_allocation" / "u1_hierarchical_20260816",
        "u2": output_root / "risk_allocation" / "u2_hierarchical_20260816",
        "u3": output_root / "risk_allocation" / "u3_hierarchical_20260816",
    }
    for universe, path in risk_dirs.items():
        rows.append(
            RequiredInput(
                f"part_b_risk_output__{universe}",
                "F6",
                "directory",
                path,
            )
        )

    signal_dirs = {
        "u1": repo_root
        / "papers/cluster_lineage_2026/local_outputs/e5b"
        / "u1_bics_sector_vs_m1_star_classic_12m_skip1_canonical_20260816",
        "u2": repo_root
        / "papers/cluster_lineage_2026/local_outputs/e5b"
        / "u2_aum50_E55_F35_R10_classic_rosaa_short_grid_20260816",
        "u3": repo_root
        / "papers/cluster_lineage_2026/local_outputs/e5b"
        / "u3_rosaa_ra_min10_short_span_sweep_vol13m_20260816",
    }
    for universe, path in signal_dirs.items():
        rows.append(
            RequiredInput(
                f"part_b_signal_grid__{universe}",
                "F6",
                "directory",
                path,
            )
        )
        rows.extend(
            (
                RequiredInput(
                    f"part_b_signal_navs__{universe}",
                    "F6",
                    "file",
                    path / "navs.csv",
                    "Frozen per-period NAV series for all final signal legs",
                ),
                RequiredInput(
                    f"part_b_signal_weights__{universe}",
                    "F6",
                    "file",
                    path / "weights.csv",
                    "Frozen decision-weight series for all final signal legs",
                ),
            )
        )

    rows.extend(
        (
            RequiredInput(
                "part_b_risk_navs__u1",
                "F6",
                "file",
                risk_dirs["u1"] / "navs.csv",
                "Frozen per-period NAVs for Ward HRP, flat ERC, and single-link HRP",
            ),
            RequiredInput(
                "part_b_risk_weights__u1_flat_erc",
                "F6",
                "file",
                risk_dirs["u1"] / "weights_flat_erc.csv",
            ),
            RequiredInput(
                "part_b_risk_weights__u1_single_hrp",
                "F6",
                "file",
                risk_dirs["u1"] / "weights_single_hrp.csv",
            ),
            RequiredInput(
                "part_b_risk_weights__u1_ward_hrp",
                "F6",
                "file",
                risk_dirs["u1"] / "weights_ward_hrp.csv",
            ),
            RequiredInput(
                "part_b_risk_navs__u3",
                "F6",
                "file",
                risk_dirs["u3"] / "navs.csv",
                "Frozen per-period NAVs for equal-cluster RB and flat ERC",
            ),
            RequiredInput(
                "part_b_risk_weights__u3_flat_erc",
                "F6",
                "file",
                risk_dirs["u3"] / "weights_flat_erc.csv",
            ),
            RequiredInput(
                "part_b_risk_weights__u3_equal_cluster",
                "F6",
                "file",
                risk_dirs["u3"] / "weights_cluster_rb_alpha_0.csv",
            ),
        )
    )

    rows.extend(
        (
            RequiredInput(
                "adopted_equity_me36_baseline",
                "F1|F3",
                "file",
                output_root
                / "e5b/covariance_frequency_span_grid/msci_us/partitions/ME_span_036.pkl",
            ),
            RequiredInput(
                "adopted_equity_me36_delta_0.0866",
                "F3",
                "directory",
                output_root / "msci_us/ME_span_036_M1_star_delta_0.0866",
                "Required cached smoothed adopted-cell partition; fitting is forbidden",
            ),
            RequiredInput(
                "adopted_futures_wwed156_baseline",
                "F1|F3",
                "directory",
                output_root / "futures/baseline",
            ),
            RequiredInput(
                "adopted_futures_wwed156_delta_0.0691",
                "F1|F3",
                "directory",
                output_root / "futures/M1_star",
            ),
        )
    )
    return rows


def _sha256(path: Path) -> str:
    """Hash one file without loading it into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _files(path: Path) -> list[Path]:
    """Return the stable file list represented by one input path."""

    if path.is_file():
        return [path]
    if path.is_dir():
        return sorted(item for item in path.rglob("*") if item.is_file())
    return []


def _timestamp(value: float) -> str:
    """Format one modification time in UTC."""

    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()


def _manifest_digest(path: Path, files: Iterable[Path]) -> str:
    """Hash a content-addressed, path-stable manifest for one input."""

    digest = hashlib.sha256()
    for file in files:
        relative = file.name if path.is_file() else file.relative_to(path).as_posix()
        stat = file.stat()
        line = f"{relative}\0{stat.st_size}\0{_sha256(file)}\n"
        digest.update(line.encode("utf-8"))
    return digest.hexdigest()


def inventory(output_root: Path, repo_root: Path) -> list[dict[str, object]]:
    """Measure and fingerprint the complete required-input registry."""

    output: list[dict[str, object]] = []
    for item in required_inputs(output_root, repo_root):
        path = item.path.resolve()
        exists = path.exists()
        files = _files(path) if exists else []
        mtimes = [file.stat().st_mtime for file in files]
        actual_kind = "file" if path.is_file() else "directory" if path.is_dir() else "missing"
        kind_matches = actual_kind == item.kind
        status = "PASS" if exists and kind_matches and files else "MISSING"
        if exists and not kind_matches:
            status = "WRONG_KIND"
        elif exists and kind_matches and not files:
            status = "EMPTY"
        output.append(
            {
                "input_id": item.input_id,
                "stages": item.stages,
                "expected_kind": item.kind,
                "path": str(path),
                "resolution_count": int(exists),
                "status": status,
                "file_count": len(files),
                "total_bytes": sum(file.stat().st_size for file in files),
                "min_mtime_utc": _timestamp(min(mtimes)) if mtimes else "",
                "max_mtime_utc": _timestamp(max(mtimes)) if mtimes else "",
                "manifest_sha256": _manifest_digest(path, files) if files else "",
                "note": item.note,
            }
        )
    return output


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    """Write the deterministic inventory table."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    """Build F0, print its acceptance summary, and fail closed on missing inputs."""

    output_root = _output_root()
    rows = inventory(output_root, _repo_root())
    destination = output_root / "finalisation/f0/cache_inventory.csv"
    _write_csv(destination, rows)
    failures = [row for row in rows if row["status"] != "PASS"]
    print(f"inventory_path={destination}")
    print(f"inputs={len(rows)} passed={len(rows) - len(failures)} failed={len(failures)}")
    for row in failures:
        print(f"{row['status']}: {row['input_id']} -> {row['path']}")
    if failures:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
