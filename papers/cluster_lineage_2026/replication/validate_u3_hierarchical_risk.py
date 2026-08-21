"""Independently validate persisted U3 hierarchical-risk artifacts.

The validator reads CSV and pickle outputs rather than invoking the experiment.  It
re-solves flat and equal-cluster risk budgets on three dates, independently reconstructs
Ward-HRP recursive bisection, and reconciles the frozen long-short source table.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from optimalportfolios import Constraints
from optimalportfolios.optimization.general.risk_budgeting import wrapper_risk_budgeting
from scipy.cluster.hierarchy import leaves_list

import papers.cluster_lineage_2026.replication.run_backtests as e5
import papers.cluster_lineage_2026.replication.run_depc1_cluster_comparison as d4
import papers.cluster_lineage_2026.replication.run_futures_best_relative_pnl_scatter as best
import papers.cluster_lineage_2026.replication.run_u3_hierarchical_risk as run


TOLERANCE = 5e-10
SOLVER_TOLERANCE = 2e-8


def _base() -> Path:
    """Return the external cluster-lineage output root."""
    return Path(
        os.environ.get(
            "CLUSTER_LINEAGE_OUTPUT_DIR",
            r"C:\Users\artur\OneDrive\analytics\outputs\cluster_lineage_2026",
        )
    )


def _root() -> Path:
    """Return the U3 hierarchical-risk artifact directory."""
    return _base() / "risk_allocation" / "u3_hierarchical_20260816"


def _read(name: str) -> pd.DataFrame:
    """Read one persisted CSV table."""
    return pd.read_csv(_root() / f"{name}.csv", float_precision="round_trip")


def _load_cache(date: pd.Timestamp) -> dict[str, object]:
    """Read one per-date allocation cache."""
    path = _root() / "allocation_cache" / f"{date:%Y%m%d}.pkl"
    with path.open("rb") as stream:
        return pickle.load(stream)


def _inputs() -> d4.UniverseInputs:
    """Load frozen U3 inputs without calling the experiment's input wrapper."""
    source = d4._u3_inputs()
    dates = source.dates[
        (source.dates >= pd.Timestamp("2009-08-31")) & (source.dates <= pd.Timestamp("2026-06-30"))
    ]
    return d4.UniverseInputs(
        universe=source.universe,
        returns=source.returns,
        dates=dates,
        eligibility=source.eligibility.reindex(index=dates),
        model=source.model,
        taxonomy=source.taxonomy,
        frozen_panel=source.frozen_panel.reindex(index=dates),
        config_id=source.config_id,
        input_paths=source.input_paths,
    )


def _direct_equal_cluster_budget(clusters: pd.Series) -> pd.Series:
    """Construct equal-cluster/equal-within-cluster budgets independently."""
    sizes = clusters.value_counts(sort=False).astype(float)
    group_budget = pd.Series(1.0 / len(sizes), index=sizes.index)
    return clusters.map(group_budget / sizes)


def _cluster_variance(covar: pd.DataFrame, assets: list[str]) -> float:
    """Return inverse-variance portfolio variance for one HRP branch."""
    branch = covar.loc[assets, assets]
    weights = 1.0 / np.diag(branch.to_numpy(dtype=float))
    weights /= weights.sum()
    return float(weights @ branch.to_numpy(dtype=float) @ weights)


def _direct_hrp(covar: pd.DataFrame, linkage: np.ndarray) -> pd.Series:
    """Reconstruct canonical HRP recursive bisection independently."""
    ordered = list(covar.index[leaves_list(linkage)])
    weights = pd.Series(1.0, index=covar.index)
    branches = [ordered]
    while branches:
        next_branches = []
        for branch in branches:
            if len(branch) <= 1:
                continue
            midpoint = len(branch) // 2
            left = branch[:midpoint]
            right = branch[midpoint:]
            left_variance = _cluster_variance(covar, left)
            right_variance = _cluster_variance(covar, right)
            left_fraction = 1.0 - left_variance / (left_variance + right_variance)
            weights.loc[left] *= left_fraction
            weights.loc[right] *= 1.0 - left_fraction
            next_branches.extend([left, right])
        branches = next_branches
    return weights / weights.sum()


def _acceptance_row(check: str, measured: float, tolerance: float, passed: bool) -> dict:
    """Format one independent acceptance row."""
    return {
        "check": check,
        "measured": measured,
        "tolerance": tolerance,
        "status": "PASS" if passed else "FAIL",
    }


def validate() -> pd.DataFrame:
    """Run independent persisted-artifact checks and return their table."""
    performance = _read("performance")
    paper = _read("paper_comparison")
    risk = _read("risk_per_date")
    contributions = _read("cluster_risk_contributions")
    diagnostics = _read("allocation_diagnostics")
    replay = _read("determinism")
    asset_classes = _read("allocation_asset_class_summary")
    signal_performance = _read("signal_performance")
    inputs = _inputs()
    dates = pd.DatetimeIndex(pd.to_datetime(diagnostics["date"]))
    eligibility = inputs.eligibility.reindex(index=dates).astype(bool)

    maximum_weight_sum_error = 0.0
    minimum_weight = np.inf
    maximum_outside = 0.0
    for method in run.METHODS:
        panel = _read(f"weights_{method}").set_index("date")
        panel.index = pd.to_datetime(panel.index)
        panel = panel.astype(float).reindex(index=dates)
        maximum_weight_sum_error = max(
            maximum_weight_sum_error,
            float(panel.sum(axis=1).subtract(1.0).abs().max()),
        )
        minimum_weight = min(minimum_weight, float(panel.min().min()))
        maximum_outside = max(
            maximum_outside,
            float(panel.where(~eligibility, 0.0).abs().to_numpy().max()),
        )

    sample_dates = dates[[0, len(dates) // 2, -1]]
    flat_errors = []
    cluster_errors = []
    hrp_errors = []
    for date in sample_dates:
        payload = _load_cache(date)
        clusters = inputs.frozen_panel.loc[date].dropna()
        snapshot_path = _base() / "futures" / "M1_star" / f"{date:%Y%m%d}.pkl"
        with snapshot_path.open("rb") as stream:
            snapshot = pickle.load(stream)
        covar = (
            snapshot.get_y_covar().reindex(index=clusters.index, columns=clusters.index)
            * run.ANNUALIZATION
        )
        direct_flat = wrapper_risk_budgeting(
            pd_covar=covar,
            constraints=Constraints(is_long_only=True),
            risk_budget=None,
        )
        direct_cluster = wrapper_risk_budgeting(
            pd_covar=covar,
            constraints=Constraints(is_long_only=True),
            risk_budget=_direct_equal_cluster_budget(clusters),
        )
        ward_path = (
            _base()
            / "depc1"
            / "futures"
            / "raw"
            / "W_WED_span_156_M1_star_delta_0.0691"
            / f"{date:%Y%m%d}.pkl"
        )
        with ward_path.open("rb") as stream:
            ward = pickle.load(stream)
        direct_hrp = _direct_hrp(covar, np.asarray(ward["linkage"], dtype=float))
        flat_errors.append(float(direct_flat.subtract(payload["methods"]["flat_erc"]).abs().max()))
        cluster_errors.append(
            float(direct_cluster.subtract(payload["methods"]["cluster_rb_alpha_0"]).abs().max())
        )
        hrp_errors.append(float(direct_hrp.subtract(payload["methods"]["ward_hrp"]).abs().max()))

    contribution_error = float(
        contributions.groupby(["date", "method"], sort=False)["risk_share"]
        .sum()
        .subtract(1.0)
        .abs()
        .max()
    )
    source = pd.read_csv(best._root() / "performance.csv", float_precision="round_trip").set_index(
        "method"
    )
    signal = signal_performance.set_index("method")
    metrics = [
        "net_return_annualized",
        "volatility_annualized",
        "sharpe_rf0",
        "one_way_turnover_annualized",
    ]
    signal_error = float(
        signal.loc[source.index, metrics].subtract(source[metrics]).abs().to_numpy().max()
    )
    class_sums = asset_classes.groupby("method")["mean_capital_weight"].sum()
    class_sum_error = float(class_sums.subtract(1.0).abs().max())
    excluded_columns = eligibility.columns.intersection(run.OWNER_EXCLUSIONS)
    excluded_observations = int(eligibility.loc[:, excluded_columns].sum().sum())

    checks = [
        (
            "performance methods",
            performance["method"].nunique(),
            len(run.METHODS),
            set(performance["method"]) == set(run.METHODS),
        ),
        ("allocation dates", len(dates), 203, len(dates) == 203),
        ("risk rows", len(risk), 203 * len(run.METHODS), len(risk) == 203 * len(run.METHODS)),
        (
            "maximum persisted weight-sum error",
            maximum_weight_sum_error,
            TOLERANCE,
            maximum_weight_sum_error <= TOLERANCE,
        ),
        ("minimum persisted long-only weight", minimum_weight, 0.0, minimum_weight >= 0.0),
        (
            "maximum weight outside eligibility",
            maximum_outside,
            TOLERANCE,
            maximum_outside <= TOLERANCE,
        ),
        (
            "sampled flat ERC independent solver error",
            max(flat_errors),
            SOLVER_TOLERANCE,
            max(flat_errors) <= SOLVER_TOLERANCE,
        ),
        (
            "sampled equal-cluster independent solver error",
            max(cluster_errors),
            SOLVER_TOLERANCE,
            max(cluster_errors) <= SOLVER_TOLERANCE,
        ),
        (
            "sampled Ward-HRP independent recursion error",
            max(hrp_errors),
            TOLERANCE,
            max(hrp_errors) <= TOLERANCE,
        ),
        (
            "Euler cluster-risk contribution sum error",
            contribution_error,
            TOLERANCE,
            contribution_error <= TOLERANCE,
        ),
        (
            "mean asset-class capital sum error",
            class_sum_error,
            TOLERANCE,
            class_sum_error <= TOLERANCE,
        ),
        (
            "owner-excluded eligible observations",
            excluded_observations,
            0,
            excluded_observations == 0,
        ),
        ("frozen long-short source-table error", signal_error, 1e-12, signal_error <= 1e-12),
        ("paper comparison rows", len(paper), 5, len(paper) == 5),
        (
            "Ward-HERC paper rows",
            paper["method_id"].eq("ward_herc").sum(),
            0,
            not paper["method_id"].eq("ward_herc").any(),
        ),
        (
            "EW-all paper yardstick rows",
            paper["comparison_benchmark"].str.contains("EW", case=False).sum(),
            0,
            not paper["comparison_benchmark"].str.contains("EW", case=False).any(),
        ),
        (
            "deterministic CSV artifacts",
            replay["byte_identical"].sum(),
            len(replay),
            replay["byte_identical"].all(),
        ),
    ]
    result = pd.DataFrame(
        [
            _acceptance_row(check, measured, tolerance, passed)
            for check, measured, tolerance, passed in checks
        ]
    )
    e5._write(result, _root() / "independent_validation.csv")
    if not result["status"].eq("PASS").all():
        raise AssertionError(result.loc[~result["status"].eq("PASS")])
    return result


def main() -> None:
    """Run independent validation and print a concise verdict."""
    result = validate()
    print(f"U3 hierarchical risk independent validation: PASS ({len(result)}/{len(result)})")


if __name__ == "__main__":
    main()
