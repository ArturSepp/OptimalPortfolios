"""Stage E0 regression and determinism runner for the frozen S&P 500 panel."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from papers.cluster_lineage_2026.replication import metrics
from papers.cluster_lineage_2026.replication.sp500_baseline import (
    estimate_rolling,
    extract_partitions,
    load_inputs,
)

RUNNER = "papers/cluster_lineage_2026/replication/validate_e0.py"
EXPECTED = {
    "lineage_churn_panel": (3.2115, 0.0001),
    "n_derived_tracks": (216.0, 0.0),
    "matcher_attributable_churn": (0.486, 0.005),
    "ari_sector": (0.202967, 0.005),
    "ari_industry_group": (0.297012, 0.005),
    "ari_industry": (0.331935, 0.005),
}


def _metric_suite() -> Dict[str, Any]:
    """Evaluate deterministic E0 metrics on the cached frozen baseline panel."""
    covar_data = estimate_rolling("baseline")
    inputs = load_inputs()
    partitions = extract_partitions(covar_data)
    lineage, _ = metrics.lineage_metrics(covar_data)
    ari, _ = metrics.ari_metrics(partitions, inputs["metadata"])
    consecutive, _ = metrics.consecutive_partition_metrics(partitions)
    shape, _ = metrics.size_shape_metrics(partitions)
    return {
        "n_snapshots": len(covar_data),
        "raw_churn": metrics.annualized_churn(metrics.greedy_membership_panel(partitions)),
        **lineage,
        **ari,
        **consecutive,
        **shape,
    }


def run_validation() -> Dict[str, Any]:
    """Assert frozen tolerances and byte-identical repeated metric serialisation."""
    first = _metric_suite()
    second = _metric_suite()
    first_bytes = metrics.deterministic_metric_bytes(first)
    second_bytes = metrics.deterministic_metric_bytes(second)
    assert first_bytes == second_bytes
    checks = {}
    for name, (expected, tolerance) in EXPECTED.items():
        actual = float(first[name])
        error = abs(actual - expected)
        assert error <= tolerance, f"{name}: {actual} outside {expected} +/- {tolerance}"
        checks[name] = {
            "actual": actual,
            "expected": expected,
            "tolerance": tolerance,
            "absolute_error": error,
            "status": "PASS",
        }
    assert first["n_snapshots"] == 60
    result = {
        "runner": RUNNER,
        "cache_directory": str(
            Path("cluster_smoothing") / "sp500_baseline" / "baseline"
        ),
        "deterministic_bytes": len(first_bytes),
        "determinism_status": "PASS",
        "checks": checks,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    run_validation()
