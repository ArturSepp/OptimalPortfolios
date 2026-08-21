"""Materialize the owner-selected U2 BlackRock primary specification.

The primary fund model is the fixed selected hybrid from the AUM sensitivity run with
a strictly greater-than USD 100 million point-in-time AUM cutoff.  This module does not
search or refit a different model: it runs the verified sensitivity harness cache-first,
selects its exact ``aum_100m`` rows, and writes a canonical primary artifact set.  The
earlier USD 50 million experiment remains frozen and auditable.
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Mapping

import pandas as pd

import papers.cluster_lineage_2026.replication.run_backtests as e5
import papers.cluster_lineage_2026.replication.run_u2_blackrock_aum_filter as aum_history
import papers.cluster_lineage_2026.replication.run_u2_blackrock_aum_sensitivity as sensitivity
import papers.cluster_lineage_2026.replication.run_u2_blackrock_etf_grid as funds
from papers.cluster_lineage_2026.replication.empirical_specs import (
    U2_BLACKROCK_PRIMARY_AUM_SPEC,
)


RUNNER = "papers/cluster_lineage_2026/replication/run_u2_blackrock_primary.py"
PRIMARY_THRESHOLD = U2_BLACKROCK_PRIMARY_AUM_SPEC.threshold_usd_millions
PRIMARY_FILTER_ID = f"aum_{int(PRIMARY_THRESHOLD)}m"
DECISION_DATE = "2026-08-16"
WEIGHT_TOLERANCE = 1e-12


def _root() -> Path:
    """Return the external canonical U2-primary output directory."""
    root = funds._root() / "aum100_primary_20260816"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _select_primary_rows(frame: pd.DataFrame) -> pd.DataFrame:
    """Select and validate the exact primary-cutoff rows from one source table."""
    if "filter_id" not in frame:
        raise KeyError("source table has no filter_id")
    selected = frame.loc[frame["filter_id"].eq(PRIMARY_FILTER_ID)].copy()
    if selected.empty:
        raise AssertionError(f"source table has no {PRIMARY_FILTER_ID} rows")
    if "threshold_usd_millions" in selected:
        thresholds = pd.to_numeric(
            selected["threshold_usd_millions"], errors="raise"
        )
        if not thresholds.eq(PRIMARY_THRESHOLD).all():
            raise AssertionError("primary rows carry the wrong AUM threshold")
    return selected.reset_index(drop=True)


def _source_manifest() -> pd.DataFrame:
    """Return hashes for the sensitivity cache and persisted source artifacts."""
    paths = [sensitivity._cache_path()]
    paths.extend(
        path
        for path in sorted(sensitivity._root().glob("*.csv"))
        if path.name != "runtime.csv"
    )
    return pd.DataFrame(
        [
            {
                "artifact": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in paths
        ]
    )


def _acceptance(selected: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Return measured primary-promotion checks and require every one to pass."""
    performance = selected["performance"]
    comparison = selected["comparison_vs_global"]
    signal = selected["signal_diagnostics"]
    weights = selected["weight_diagnostics"]
    full_hybrid = comparison.loc[
        comparison["analysis_window"].eq(sensitivity.search.FULL_WINDOW)
        & comparison["method"].eq("hybrid")
    ]
    if len(full_hybrid) != 1:
        raise AssertionError("primary full-window hybrid row is not unique")
    row = full_hybrid.iloc[0]
    source_acceptance = pd.read_csv(
        sensitivity._root() / "acceptance.csv", float_precision="round_trip"
    )
    source_determinism = pd.read_csv(
        sensitivity._root() / "determinism.csv", float_precision="round_trip"
    )
    checks = [
        {
            "check": "primary threshold USD millions",
            "measured": PRIMARY_THRESHOLD,
            "tolerance": 100.0,
            "status": "PASS" if PRIMARY_THRESHOLD == 100.0 else "FAIL",
        },
        {
            "check": "selected performance rows",
            "measured": len(performance),
            "tolerance": 9,
            "status": "PASS" if len(performance) == 9 else "FAIL",
        },
        {
            "check": "selected comparison rows",
            "measured": len(comparison),
            "tolerance": 6,
            "status": "PASS" if len(comparison) == 6 else "FAIL",
        },
        {
            "check": "source sensitivity acceptance passes",
            "measured": int(source_acceptance["status"].eq("PASS").sum()),
            "tolerance": len(source_acceptance),
            "status": (
                "PASS" if source_acceptance["status"].eq("PASS").all() else "FAIL"
            ),
        },
        {
            "check": "source deterministic artifacts",
            "measured": int(source_determinism["byte_identical"].astype(bool).sum()),
            "tolerance": len(source_determinism),
            "status": (
                "PASS"
                if source_determinism["byte_identical"].astype(bool).all()
                else "FAIL"
            ),
        },
        {
            "check": "maximum signal lookahead days",
            "measured": float(
                signal[
                    ["max_global_lookahead_days", "max_cluster_lookahead_days"]
                ].to_numpy().max()
            ),
            "tolerance": 0.0,
            "status": (
                "PASS"
                if float(
                    signal[
                        ["max_global_lookahead_days", "max_cluster_lookahead_days"]
                    ].to_numpy().max()
                )
                <= 0.0
                else "FAIL"
            ),
        },
        {
            "check": "maximum weight/exposure error",
            "measured": float(weights["maximum_error"].max()),
            "tolerance": WEIGHT_TOLERANCE,
            "status": (
                "PASS"
                if float(weights["maximum_error"].max()) <= WEIGHT_TOLERANCE
                else "FAIL"
            ),
        },
        {
            "check": "full-window hybrid net-return delta versus global",
            "measured": float(row["delta_net_return_annualized"]),
            "tolerance": "> 0",
            "status": (
                "PASS" if float(row["delta_net_return_annualized"]) > 0.0 else "FAIL"
            ),
        },
        {
            "check": "full-window hybrid Sharpe delta versus global",
            "measured": float(row["delta_sharpe_rf0"]),
            "tolerance": "> 0",
            "status": "PASS" if float(row["delta_sharpe_rf0"]) > 0.0 else "FAIL",
        },
    ]
    acceptance = pd.DataFrame(checks)
    if not acceptance["status"].eq("PASS").all():
        raise AssertionError(acceptance.loc[~acceptance["status"].eq("PASS")])
    return acceptance


def _materialize(source: Mapping[str, pd.DataFrame]) -> Mapping[str, pd.DataFrame]:
    """Write the canonical primary tables from one verified sensitivity result."""
    started = time.perf_counter()
    selected = {
        name: _select_primary_rows(source[name])
        for name in (
            "eligibility_by_date",
            "eligibility_summary",
            "partition_diagnostics",
            "signal_diagnostics",
            "weight_diagnostics",
            "performance",
            "comparison_vs_global",
            "full_window_summary",
        )
    }
    specification = source["specification"].copy()
    specification["primary_aum_threshold_usd_millions"] = PRIMARY_THRESHOLD
    specification["primary_filter_id"] = PRIMARY_FILTER_ID
    specification["decision_date"] = DECISION_DATE
    specification["selection_provenance"] = "owner_selected_after_aum_sensitivity"
    output = {
        "specification": specification,
        "aum_eligibility_specification": U2_BLACKROCK_PRIMARY_AUM_SPEC.to_frame(
            name="U2_BLACKROCK_PRIMARY_AUM_SPEC_20260816"
        ),
        "selection_record": pd.DataFrame(
            [
                {
                    "decision_date": DECISION_DATE,
                    "primary_filter_id": PRIMARY_FILTER_ID,
                    "threshold_usd_millions": PRIMARY_THRESHOLD,
                    "status": "owner_selected",
                    "selection_timing": "after_reported_threshold_sensitivity",
                    "supersedes_primary_cutoff_usd_millions": (
                        aum_history.AUM_THRESHOLD_USD_MILLIONS
                    ),
                    "runner": RUNNER,
                }
            ]
        ),
        **selected,
        "source_manifest": _source_manifest(),
    }
    output["acceptance"] = _acceptance(selected)
    output["runtime"] = pd.DataFrame(
        [{"source_cache_status": "hit", "runtime_seconds": time.perf_counter() - started}]
    )
    for name, frame in output.items():
        e5._write(frame, _root() / f"{name}.csv")
    return output


def _hash_outputs() -> dict[str, str]:
    """Hash deterministic primary artifacts, excluding runtime and replay output."""
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(_root().glob("*.csv"))
        if path.name not in {"runtime.csv", "determinism.csv"}
    }


def run() -> Mapping[str, pd.DataFrame]:
    """Run the source harness cache-first and materialize the USD 100m primary rows."""
    return _materialize(sensitivity.run())


def verify_determinism() -> pd.DataFrame:
    """Materialize twice from one verified source result and compare output hashes."""
    source = sensitivity.run()
    _materialize(source)
    first = _hash_outputs()
    _materialize(source)
    second = _hash_outputs()
    names = sorted(set(first) | set(second))
    replay = pd.DataFrame(
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
    if not replay["byte_identical"].all():
        raise AssertionError(replay.loc[~replay["byte_identical"]])
    e5._write(replay, _root() / "determinism.csv")
    return replay


def main() -> None:
    """Execute and print the primary-promotion acceptance summary."""
    replay = verify_determinism()
    acceptance = pd.read_csv(_root() / "acceptance.csv")
    print(acceptance.to_string(index=False), flush=True)
    print(f"determinism: {int(replay['byte_identical'].sum())}/{len(replay)}", flush=True)


if __name__ == "__main__":
    main()
