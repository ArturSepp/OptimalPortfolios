"""Validate the deterministic lineage matcher against the former NetworkX backend.

This local roadmap diagnostic loads the cached S&P 500 and mac_apac covariance panels, runs
the same production edge construction with each matching backend, checks total matched weight
and relabel output, and prints comparable wall-clock timings. It writes no repository output.
"""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Any, Callable, Dict, List, Tuple

import pandas as pd

from optimalportfolios.covar_estimation import risk_labelling as rl
from optimalportfolios.covar_estimation._lineage_matching import solve_max_weight_matching


@dataclass(frozen=True)
class ValidationResult:
    """One panel's matching identity, objective, runtime, and churn comparison."""

    panel: str
    dates: int
    reference_seconds: float
    solver_seconds: float
    reference_weight: float
    solver_weight: float
    relabel_identical: bool
    first_different_date: Any
    matching_difference_edges: int
    lineage_churn: float


def _networkx_solver(
        n_left: int,
        n_right: int,
        edges: List[Tuple[int, int, float]],
) -> Dict[int, int]:
    """Reproduce the removed NetworkX min-cost-flow matching backend exactly."""
    import networkx as nx

    graph = nx.DiGraph()
    graph.add_node("source", demand=-n_left)
    graph.add_node("sink", demand=n_left)
    graph.add_edge("source", "sink", capacity=n_left, weight=0)
    for left in range(n_left):
        graph.add_edge("source", ("left", left), capacity=1, weight=0)
    for right in range(n_right):
        graph.add_edge(("right", right), "sink", capacity=1, weight=0)
    for left, right, weight in edges:
        graph.add_edge(
            ("left", left), ("right", right), capacity=1, weight=-int(weight)
        )
    flow = nx.min_cost_flow(graph)
    return {
        left: right
        for left, right, _ in edges
        if flow.get(("left", left), {}).get(("right", right), 0) == 1
    }


def _build_snapshots(covar_data) -> Tuple[Dict, Dict]:
    """Build production fingerprints and clipped factor covariances once per panel."""
    snapshots, x_covars = {}, {}
    for date in covar_data.dates:
        snapshot = covar_data[date]
        snapshots[date], _ = rl._snapshot_fingerprints(snapshot, weighting="equal")
        x_covars[date] = rl._psd_clip(snapshot.x_covar.to_numpy())
    return snapshots, x_covars


def _run_backend(
        snapshots: Dict,
        x_covars: Dict,
        solver: Callable[[int, int, List[Tuple[int, int, float]]], Dict[int, int]],
) -> Tuple[pd.DataFrame, pd.DataFrame, float, float, set]:
    """Run production matching with one injected solver and capture objective and time."""
    objectives, selections = [], []

    def tracked_solver(n_left, n_right, edges):
        """Capture the selected unperturbed objective from an injected solver."""
        matching = solver(n_left, n_right, edges)
        weights = {(left, right): weight for left, right, weight in edges}
        objectives.append(sum(weights[(left, right)] for left, right in matching.items()))
        selections.append({(left, right, weights[(left, right)])
                           for left, right in matching.items()})
        return matching

    original = rl.solve_max_weight_matching
    started = perf_counter()
    try:
        rl.solve_max_weight_matching = tracked_solver
        relabel, lineage = rl._match_panel_mcf(
            snapshots,
            x_covars,
            overlap_metric="overlap",
            combine="gated",
            overlap_band=(0.15, 0.60),
            spread_vol_cut=0.015,
            w_overlap=0.6,
            bridge_window=6,
            bridge_decay=0.5,
        )
    finally:
        rl.solve_max_weight_matching = original
    return relabel, lineage, objectives[0], perf_counter() - started, selections[0]


def _canonical_relabel(relabel: pd.DataFrame) -> pd.DataFrame:
    """Return a stable row ordering for byte-level relabel comparisons."""
    return relabel.assign(_raw=relabel["raw_label"].map(str)).sort_values(
        ["date", "_raw"]
    ).drop(columns="_raw").reset_index(drop=True)


def _first_difference(left: pd.DataFrame, right: pd.DataFrame) -> Any:
    """Return the first date whose raw-cluster derived ids differ, or ``None``."""
    comparable = left.merge(
        right,
        on=["date", "raw_label"],
        how="outer",
        suffixes=("_reference", "_solver"),
        indicator=True,
    )
    different = comparable[
        (comparable["_merge"] != "both")
        | (comparable["derived_id_reference"] != comparable["derived_id_solver"])
    ]
    return None if different.empty else different["date"].min()


def validate_panel(panel: str, covar_data) -> ValidationResult:
    """Validate one cached covariance panel and return its comparison metrics."""
    from papers.cluster_lineage_2026.replication.sp500_baseline import annualized_churn

    snapshots, x_covars = _build_snapshots(covar_data)
    reference_result = _run_backend(snapshots, x_covars, _networkx_solver)
    solved_result = _run_backend(snapshots, x_covars, solve_max_weight_matching)
    reference, reference_lineage, reference_weight, reference_seconds, reference_edges = (
        reference_result
    )
    solved, solved_lineage, solver_weight, solver_seconds, solved_edges = solved_result
    reference = _canonical_relabel(reference)
    solved = _canonical_relabel(solved)
    identical = reference.equals(solved) and reference_lineage.equals(solved_lineage)
    first_different_date = None if identical else _first_difference(reference, solved)
    if abs(reference_weight - solver_weight) > 1e-9:
        raise AssertionError(
            f"{panel}: objective mismatch {reference_weight} != {solver_weight}"
        )
    difference_edges = reference_edges ^ solved_edges
    if difference_edges:
        nodes = [(date, label) for date in sorted(snapshots) for label in snapshots[date]]
        rows = []
        for left, right, weight in sorted(difference_edges):
            source, target = nodes[left], nodes[right]
            rows.append({
                "backend": "reference" if (left, right, weight) in reference_edges else "solver",
                "source_date": source[0],
                "source_label": source[1],
                "target_date": target[0],
                "target_label": target[1],
                "weight": weight,
            })
        print(pd.DataFrame(rows).to_string(index=False))
    report = rl.analyze_risk_clusters(covar_data)
    return ValidationResult(
        panel=panel,
        dates=len(covar_data.dates),
        reference_seconds=reference_seconds,
        solver_seconds=solver_seconds,
        reference_weight=reference_weight,
        solver_weight=solver_weight,
        relabel_identical=identical,
        first_different_date=first_different_date,
        matching_difference_edges=len(difference_edges),
        lineage_churn=annualized_churn(report.to_membership_panel()),
    )


def run_validation() -> pd.DataFrame:
    """Load both roadmap panels, execute all checks, and print the result table."""
    from rosaa.research.analysis.risk_label_config_sweep import (
        estimate_rolling_covar_data,
        load_covar_data,
    )
    from papers.cluster_lineage_2026.replication.sp500_baseline import estimate_rolling

    try:
        mac_apac = load_covar_data()
    except FileNotFoundError:
        mac_apac = estimate_rolling_covar_data()
    results = [
        validate_panel("sp500", estimate_rolling("baseline")),
        validate_panel("mac_apac", mac_apac),
    ]
    table = pd.DataFrame([result.__dict__ for result in results]).set_index("panel")
    print(table.to_string())
    return table


if __name__ == "__main__":
    run_validation()
