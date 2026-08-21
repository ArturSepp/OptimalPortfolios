"""Independently validate the persisted U1 hierarchical-risk experiment.

The validator reads the external CSV and per-date pickle artifacts rather than calling
the experiment runner.  It independently reconstructs the production flat-ERC and
equal-cluster risk-budgeting solutions on three dates, checks saved portfolio panels,
and reconciles the persisted Euler cluster-risk contributions.
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from optimalportfolios import Constraints
from optimalportfolios.optimization.general.risk_budgeting import wrapper_risk_budgeting

import papers.cluster_lineage_2026.replication.run_depc1_cluster_comparison as d4


METHODS = (
    "flat_erc",
    "cluster_rb_alpha_0_5",
    "cluster_rb_alpha_0",
    "ward_hrp",
    "ward_herc",
    "single_hrp",
)
TOLERANCE = 1e-10
SOLVER_TOLERANCE = 2e-8


def _base() -> Path:
    """Return the external root holding persisted experiment artifacts."""
    return Path(
        os.environ.get(
            "CLUSTER_LINEAGE_OUTPUT_DIR",
            r"C:\Users\artur\OneDrive\analytics\outputs\cluster_lineage_2026",
        )
    )


def _root() -> Path:
    """Return the U1 hierarchical-risk artifact directory."""
    return _base() / "risk_allocation" / "u1_hierarchical_20260816"


def _read(name: str) -> pd.DataFrame:
    """Read one persisted CSV table."""
    return pd.read_csv(_root() / f"{name}.csv")


def _load_cache(date: pd.Timestamp) -> dict[str, object]:
    """Read one per-date allocation cache."""
    path = _root() / "allocation_cache" / f"{date:%Y%m%d}.pkl"
    with path.open("rb") as stream:
        return pickle.load(stream)


def _load_covar(date: pd.Timestamp) -> pd.DataFrame:
    """Read one frozen accepted HCGL covariance matrix."""
    path = _base() / "msci_us" / "baseline" / f"{date:%Y%m%d}.pkl"
    with path.open("rb") as stream:
        snapshot = pickle.load(stream)
    return snapshot.get_y_covar()


def _direct_equal_cluster_budget(clusters: pd.Series) -> pd.Series:
    """Construct equal-cluster, equal-within-cluster budgets independently."""
    sizes = clusters.value_counts(sort=False).astype(float)
    group_budget = pd.Series(1.0 / len(sizes), index=sizes.index)
    return clusters.map(group_budget / sizes)


def validate() -> pd.DataFrame:
    """Run independent persisted-artifact checks and return the acceptance table."""
    performance = _read("performance")
    risk = _read("risk_per_date")
    contributions = _read("cluster_risk_contributions")
    replay = _read("determinism")
    comparison = _read("comparison_vs_flat_erc")
    paper = _read("paper_comparison")
    diagnostics = _read("allocation_diagnostics")
    dates = pd.DatetimeIndex(pd.to_datetime(diagnostics["date"]))
    inputs = d4._u1_inputs()

    maximum_weight_sum_error = 0.0
    minimum_weight = np.inf
    maximum_outside_eligibility = 0.0
    for method in METHODS:
        panel = _read(f"weights_{method}").set_index("date")
        panel.index = pd.to_datetime(panel.index)
        panel = panel.astype(float).reindex(index=inputs.dates)
        maximum_weight_sum_error = max(
            maximum_weight_sum_error,
            float(panel.sum(axis=1).subtract(1.0).abs().max()),
        )
        minimum_weight = min(minimum_weight, float(panel.min().min()))
        outside = panel.where(~inputs.eligibility, 0.0).abs().to_numpy().max()
        maximum_outside_eligibility = max(
            maximum_outside_eligibility, float(outside)
        )

    sample_dates = dates[[0, len(dates) // 2, -1]]
    maximum_flat_solver_error = 0.0
    maximum_equal_cluster_solver_error = 0.0
    for date in sample_dates:
        payload = _load_cache(date)
        saved_flat = payload["methods"]["flat_erc"]
        saved_equal_cluster = payload["methods"]["cluster_rb_alpha_0"]
        covar = _load_covar(date).reindex(
            index=saved_flat.index, columns=saved_flat.index
        )
        ward_path = (
            _base()
            / "depc1"
            / "msci_us"
            / "raw"
            / "ME_span_036"
            / f"{date:%Y%m%d}.pkl"
        )
        with ward_path.open("rb") as stream:
            memberships = pickle.load(stream)["clusters"].reindex(saved_flat.index)
        direct_flat = wrapper_risk_budgeting(
            pd_covar=covar,
            constraints=Constraints(is_long_only=True),
            risk_budget=None,
        )
        direct_budget = _direct_equal_cluster_budget(memberships)
        direct_equal_cluster = wrapper_risk_budgeting(
            pd_covar=covar,
            constraints=Constraints(is_long_only=True),
            risk_budget=direct_budget,
        )
        maximum_flat_solver_error = max(
            maximum_flat_solver_error,
            float(direct_flat.subtract(saved_flat).abs().max()),
        )
        maximum_equal_cluster_solver_error = max(
            maximum_equal_cluster_solver_error,
            float(direct_equal_cluster.subtract(saved_equal_cluster).abs().max()),
        )

    contribution_sums = contributions.groupby(["date", "method"], sort=False)[
        "risk_share"
    ].sum()
    maximum_contribution_error = float(contribution_sums.subtract(1.0).abs().max())
    paper_metric_columns = [
        "net_return_annualized",
        "volatility_annualized",
        "sharpe_rf0",
        "one_way_turnover_annualized",
    ]
    source_signal = pd.read_csv(
        _base() / "depc1" / "msci_us" / "performance.csv",
        float_precision="round_trip",
    ).set_index("leg")
    paper_signal = paper.loc[paper["panel"].eq("long_short_momentum")].set_index(
        "method_id"
    )
    signal_table_error = float(
        paper_signal.loc[["global", "cluster_raw"], paper_metric_columns]
        .subtract(source_signal.loc[["global", "cluster_raw"], paper_metric_columns])
        .abs()
        .to_numpy()
        .max()
    )
    checks = [
        (
            "performance methods",
            float(performance["method"].nunique()),
            float(len(METHODS)),
            set(performance["method"]) == set(METHODS),
        ),
        (
            "allocation dates",
            float(len(dates)),
            203.0,
            len(dates) == 203,
        ),
        (
            "risk rows",
            float(len(risk)),
            float(203 * len(METHODS)),
            len(risk) == 203 * len(METHODS),
        ),
        (
            "maximum persisted weight-sum error",
            maximum_weight_sum_error,
            TOLERANCE,
            maximum_weight_sum_error <= TOLERANCE,
        ),
        (
            "minimum persisted long-only weight",
            minimum_weight,
            0.0,
            minimum_weight >= 0.0,
        ),
        (
            "maximum weight outside eligibility",
            maximum_outside_eligibility,
            TOLERANCE,
            maximum_outside_eligibility <= TOLERANCE,
        ),
        (
            "sampled flat ERC independent solver error",
            maximum_flat_solver_error,
            SOLVER_TOLERANCE,
            maximum_flat_solver_error <= SOLVER_TOLERANCE,
        ),
        (
            "sampled equal-cluster RB independent solver error",
            maximum_equal_cluster_solver_error,
            SOLVER_TOLERANCE,
            maximum_equal_cluster_solver_error <= SOLVER_TOLERANCE,
        ),
        (
            "persisted Euler cluster-risk reconciliation error",
            maximum_contribution_error,
            TOLERANCE,
            maximum_contribution_error <= TOLERANCE,
        ),
        (
            "deterministic CSV replay share",
            float(replay["byte_identical"].mean()),
            1.0,
            bool(replay["byte_identical"].all()),
        ),
        (
            "EW-all performance comparisons",
            float(
                comparison.astype(str)
                .apply(lambda column: column.str.contains("ew", case=False).any())
                .sum()
            ),
            0.0,
            not comparison.astype(str)
            .apply(lambda column: column.str.contains("ew", case=False).any())
            .any(),
        ),
        (
            "paper comparison rows",
            float(len(paper)),
            5.0,
            len(paper) == 5,
        ),
        (
            "Ward-HERC paper rows",
            float(paper["method_id"].eq("ward_herc").sum()),
            0.0,
            not paper["method_id"].eq("ward_herc").any(),
        ),
        (
            "paper long-short source-table error",
            signal_table_error,
            TOLERANCE,
            signal_table_error <= TOLERANCE,
        ),
    ]
    result = pd.DataFrame(
        [
            {
                "check": check,
                "measured": measured,
                "tolerance": tolerance,
                "status": "PASS" if passed else "FAIL",
            }
            for check, measured, tolerance, passed in checks
        ]
    )
    result.to_csv(_root() / "independent_validation.csv", index=False)
    if not result["status"].eq("PASS").all():
        raise AssertionError(result.loc[~result["status"].eq("PASS")])
    return result


def main() -> None:
    """Run and print the independent acceptance table."""
    result = validate()
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
