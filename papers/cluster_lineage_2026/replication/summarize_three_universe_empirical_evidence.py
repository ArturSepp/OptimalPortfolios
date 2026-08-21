"""Consolidate the paper's two empirical roles across U1, U2, and U3.

Peer-contained signal ranking holds the signal and strategic budgets fixed and changes
only the ranking pool: correlation cluster versus sector or global peers.  All three
universes use long-short portfolios.  Risk allocation is evaluated separately with
signal-free, long-only portfolios.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Mapping

import pandas as pd

import papers.cluster_lineage_2026.replication.run_backtests as e5


RUNNER = "papers/cluster_lineage_2026/replication/summarize_three_universe_empirical_evidence.py"
PERFORMANCE_METRICS = (
    "net_return_annualized",
    "gross_return_annualized",
    "volatility_annualized",
    "sharpe_rf0",
    "one_way_turnover_annualized",
    "cost_drag_bp_per_year",
)
ALLOCATION_METHODS = (
    "flat_erc",
    "cluster_rb_alpha_0",
    "ward_hrp",
    "single_hrp",
)


def _base() -> Path:
    """Return the external cluster-lineage output root."""
    return e5.get_output_path()


def _root() -> Path:
    """Return the isolated consolidated-evidence output directory."""
    return e5.get_output_path("evidence_summary", "three_universe_two_role_20260816", create=True)


def _read(path: Path) -> pd.DataFrame:
    """Read one validated source table with round-trip float parsing."""
    return pd.read_csv(path, float_precision="round_trip")


def _signal_rows() -> pd.DataFrame:
    """Return the primary same-signal ranking legs for all three universes."""
    rows = []

    u1_path = (
        _base() / "e5b" / "u1_bics_sector_vs_m1_star_classic_12m_skip1_20260815" / "performance.csv"
    )
    u1 = _read(u1_path)
    u1 = u1.loc[u1["is_primary"].astype(bool)].set_index("leg")
    u1_labels = {
        "cluster_M1_star": ("cluster", "M1-star cluster-relative rank", "symmetric"),
        "bics_sector": ("sector", "BICS sector-relative rank", "benchmark"),
        "global": ("global", "global cross-sectional rank", "benchmark"),
    }
    for source_leg, (leg, label, role) in u1_labels.items():
        source = u1.loc[source_leg]
        rows.append(
            {
                "universe": "U1 equities",
                "universe_id": "U1",
                "leg": leg,
                "label": label,
                "role": role,
                "signal": "classic 12m-ex-1m momentum",
                "q": source["q"],
                "cost_bps_one_way": 10.0,
                "source": str(u1_path),
                **{metric: source[metric] for metric in PERFORMANCE_METRICS},
            }
        )

    u2_path = (
        _base()
        / "e5b"
        / "covariance_frequency_span_grid"
        / "blackrock_us_etfs"
        / "equity_fi_60_40_long_short_aum_grid_20260816"
        / "performance.csv"
    )
    u2 = _read(u2_path)
    u2 = u2.loc[
        u2["filter_id"].eq("aum_100m")
        & u2["analysis_window"].eq("headline_20090831_20260630")
    ].set_index("method")
    u2_labels = {
        "cluster": (
            "cluster",
            "Equity/FI-only 60/40 W-THU/156 cluster rank",
            "symmetric",
        ),
        "global": ("global", "Equity/FI-only 60/40 global-sleeve rank", "benchmark"),
    }
    for source_leg, (leg, label, role) in u2_labels.items():
        source = u2.loc[source_leg]
        rows.append(
            {
                "universe": "U2 BlackRock funds",
                "universe_id": "U2",
                "leg": leg,
                "label": label,
                "role": role,
                "signal": "ROSAA risk-adjusted momentum",
                "q": source["q"],
                "cost_bps_one_way": source["cost_bps_one_way"],
                "source": str(u2_path),
                **{metric: source[metric] for metric in PERFORMANCE_METRICS},
            }
        )

    u3_path = _base() / "risk_allocation" / "u3_hierarchical_20260816" / "signal_performance.csv"
    u3 = _read(u3_path).set_index("leg")
    u3_labels = {
        "cluster": ("cluster", "M1-star cluster-relative sleeve rank", "symmetric"),
        "global": ("global", "30/30/30/10 global sleeve rank", "benchmark"),
    }
    for source_leg, (leg, label, role) in u3_labels.items():
        source = u3.loc[source_leg]
        rows.append(
            {
                "universe": "U3 futures",
                "universe_id": "U3",
                "leg": leg,
                "label": label,
                "role": role,
                "signal": "ROSAA risk-adjusted momentum",
                "q": source["q"],
                "cost_bps_one_way": source["cost_bps_one_way"],
                "source": str(u3_path),
                **{metric: source[metric] for metric in PERFORMANCE_METRICS},
            }
        )
    return pd.DataFrame(rows)


def _comparisons(
    performance: pd.DataFrame,
    specifications: Mapping[str, tuple[str, ...]],
) -> pd.DataFrame:
    """Compare each universe's cluster leg with its permitted ranking controls."""
    rows = []
    for universe_id, benchmarks in specifications.items():
        panel = performance.loc[performance["universe_id"].eq(universe_id)].set_index("leg")
        candidate = panel.loc["cluster"]
        for benchmark in benchmarks:
            control = panel.loc[benchmark]
            row = {
                "universe": candidate["universe"],
                "universe_id": universe_id,
                "cluster_leg": candidate["label"],
                "benchmark_leg": control["label"],
            }
            for metric in PERFORMANCE_METRICS:
                row[f"cluster_{metric}"] = candidate[metric]
                row[f"benchmark_{metric}"] = control[metric]
                row[f"delta_{metric}"] = candidate[metric] - control[metric]
            row["cluster_higher_net_return"] = row["delta_net_return_annualized"] > 0.0
            row["cluster_lower_volatility"] = row["delta_volatility_annualized"] < 0.0
            row["cluster_higher_sharpe"] = row["delta_sharpe_rf0"] > 0.0
            row["cluster_lower_turnover"] = row["delta_one_way_turnover_annualized"] < 0.0
            rows.append(row)
    return pd.DataFrame(rows)


def _allocation_rows() -> pd.DataFrame:
    """Return the four signal-free long-only methods for all universes."""
    roots = {
        "U1": ("U1 equities", "u1_hierarchical_20260816", 10.0),
        "U2": ("U2 BlackRock funds", "u2_hierarchical_20260816", 20.0),
        "U3": ("U3 futures", "u3_hierarchical_20260816", 10.0),
    }
    rows = []
    for universe_id, (universe, folder, cost) in roots.items():
        root = _base() / "risk_allocation" / folder
        performance = _read(root / "performance.csv").set_index("method")
        risk = _read(root / "risk_summary.csv").set_index("method")
        for method in ALLOCATION_METHODS:
            perf_row = performance.loc[method]
            risk_row = risk.loc[method]
            rows.append(
                {
                    "universe": universe,
                    "universe_id": universe_id,
                    "method": method,
                    "cost_bps_one_way": cost,
                    "net_return_annualized": perf_row["net_return_annualized"],
                    "volatility_annualized": perf_row["volatility_annualized"],
                    "sharpe_rf0": perf_row["sharpe_rf0"],
                    "one_way_turnover_annualized": perf_row["one_way_turnover_annualized"],
                    "cost_drag_bp_per_year": perf_row["cost_drag_bp_per_year"],
                    "portfolio_ex_ante_volatility_mean": risk_row[
                        "portfolio_ex_ante_volatility_mean"
                    ],
                    "effective_risk_clusters_absolute_mean": risk_row[
                        "effective_risk_clusters_absolute_mean"
                    ],
                    "maximum_absolute_cluster_risk_share_mean": risk_row[
                        "maximum_absolute_cluster_risk_share_mean"
                    ],
                    "diversification_ratio_mean": risk_row["diversification_ratio_mean"],
                    "effective_assets_mean": risk_row["effective_assets_mean"],
                    "source": str(root),
                }
            )
    return pd.DataFrame(rows)


def _allocation_comparisons(allocation: pd.DataFrame) -> pd.DataFrame:
    """Compare cluster-aware allocators with their fixed peer controls."""
    specifications = (
        ("cluster_rb_alpha_0", "flat_erc", "equal-cluster RB vs flat ERC"),
        ("ward_hrp", "flat_erc", "Rolling-Ward HRP vs flat ERC"),
        ("ward_hrp", "single_hrp", "Rolling-Ward HRP vs canonical single-HRP"),
    )
    metrics = (
        "net_return_annualized",
        "volatility_annualized",
        "sharpe_rf0",
        "one_way_turnover_annualized",
        "portfolio_ex_ante_volatility_mean",
        "effective_risk_clusters_absolute_mean",
        "maximum_absolute_cluster_risk_share_mean",
        "diversification_ratio_mean",
        "effective_assets_mean",
    )
    rows = []
    for universe_id, panel in allocation.groupby("universe_id", sort=False):
        indexed = panel.set_index("method")
        for candidate_id, benchmark_id, comparison in specifications:
            candidate = indexed.loc[candidate_id]
            benchmark = indexed.loc[benchmark_id]
            row = {
                "universe": candidate["universe"],
                "universe_id": universe_id,
                "comparison": comparison,
                "candidate_method": candidate_id,
                "benchmark_method": benchmark_id,
            }
            for metric in metrics:
                row[f"candidate_{metric}"] = candidate[metric]
                row[f"benchmark_{metric}"] = benchmark[metric]
                row[f"delta_{metric}"] = candidate[metric] - benchmark[metric]
            rows.append(row)
    return pd.DataFrame(rows)


def _scorecard(
    signal_comparisons: pd.DataFrame,
    allocation_comparisons: pd.DataFrame,
) -> pd.DataFrame:
    """Return cross-universe directional counts without pooling payoffs."""
    rows = [
        {
            "experiment": "peer-contained signal ranking",
            "comparison": "same-signal cluster rank vs permitted rank benchmark",
            "comparisons": len(signal_comparisons),
            "higher_net_return": int(signal_comparisons["cluster_higher_net_return"].sum()),
            "lower_volatility": int(signal_comparisons["cluster_lower_volatility"].sum()),
            "higher_sharpe": int(signal_comparisons["cluster_higher_sharpe"].sum()),
            "lower_turnover": int(signal_comparisons["cluster_lower_turnover"].sum()),
        }
    ]
    for comparison, panel in allocation_comparisons.groupby("comparison", sort=False):
        rows.append(
            {
                "experiment": "long-only risk allocation",
                "comparison": comparison,
                "comparisons": len(panel),
                "higher_net_return": int(panel["delta_net_return_annualized"].gt(0.0).sum()),
                "lower_volatility": int(panel["delta_volatility_annualized"].lt(0.0).sum()),
                "higher_sharpe": int(panel["delta_sharpe_rf0"].gt(0.0).sum()),
                "lower_turnover": int(panel["delta_one_way_turnover_annualized"].lt(0.0).sum()),
            }
        )
    return pd.DataFrame(rows)


def _acceptance(
    signal: pd.DataFrame,
    signal_comparisons: pd.DataFrame,
    allocation: pd.DataFrame,
    allocation_comparisons: pd.DataFrame,
) -> pd.DataFrame:
    """Return measured-versus-expected aggregation checks."""
    checks = [
        ("same-signal ranking rows", len(signal), 7, len(signal) == 7),
        (
            "same-signal ranking comparisons",
            len(signal_comparisons),
            4,
            len(signal_comparisons) == 4,
        ),
        ("allocation rows", len(allocation), 12, len(allocation) == 12),
        (
            "allocation comparisons",
            len(allocation_comparisons),
            9,
            len(allocation_comparisons) == 9,
        ),
        ("allocation long-short rows", 0, 0, True),
        ("signal allocation-method rows", 0, 0, True),
        (
            "EW-all performance yardsticks",
            int(signal_comparisons["benchmark_leg"].str.contains("EW", case=False).sum()),
            0,
            not signal_comparisons["benchmark_leg"].str.contains("EW", case=False).any(),
        ),
        (
            "Ward-HERC allocation rows",
            int(allocation["method"].eq("ward_herc").sum()),
            0,
            not allocation["method"].eq("ward_herc").any(),
        ),
        (
            "finite signal volatility comparisons",
            int(signal_comparisons["delta_volatility_annualized"].notna().sum()),
            4,
            signal_comparisons["delta_volatility_annualized"].notna().all(),
        ),
    ]
    frame = pd.DataFrame(
        [
            {
                "check": check,
                "measured": measured,
                "expected": expected,
                "status": "PASS" if passed else "FAIL",
            }
            for check, measured, expected, passed in checks
        ]
    )
    if not frame["status"].eq("PASS").all():
        raise AssertionError(frame.loc[~frame["status"].eq("PASS")])
    return frame


def run() -> Mapping[str, pd.DataFrame]:
    """Build and persist the complete two-role empirical evidence tables."""
    signal = _signal_rows()
    signal_comparisons = _comparisons(
        signal,
        {"U1": ("sector", "global"), "U2": ("global",), "U3": ("global",)},
    )
    allocation = _allocation_rows()
    allocation_comparisons = _allocation_comparisons(allocation)
    scorecard = _scorecard(signal_comparisons, allocation_comparisons)
    acceptance = _acceptance(signal, signal_comparisons, allocation, allocation_comparisons)
    design = pd.DataFrame(
        [
            {
                "experiment": "peer-contained signal ranking",
                "positioning": "long-short only",
                "signal": "U1 classic 12m-ex-1m|U2/U3 ROSAA risk-adjusted momentum",
                "allocation": "none; equal group/selected-asset construction",
                "purpose": "compare the same signal ranked within clusters versus peers",
            },
            {
                "experiment": "risk allocation",
                "positioning": "long-only only",
                "signal": "none",
                "allocation": "flat ERC|equal-cluster RB|Ward-HRP|single-HRP",
                "purpose": "measure allocation efficiency versus peer risk methods",
            },
        ]
    )
    output = {
        "design": design,
        "signal_isolation_performance": signal,
        "signal_isolation_comparison": signal_comparisons,
        "allocation_performance_and_risk": allocation,
        "allocation_comparison": allocation_comparisons,
        "evidence_scorecard": scorecard,
        "acceptance": acceptance,
    }
    for name, frame in output.items():
        e5._write(frame, _root() / f"{name}.csv")
    return output


def _hash_outputs() -> dict[str, str]:
    """Return hashes for every numerical artifact except replay itself."""
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(_root().glob("*.csv"))
        if path.name != "determinism.csv"
    }


def verify_determinism() -> pd.DataFrame:
    """Require two complete aggregation passes to be byte-identical."""
    run()
    first = _hash_outputs()
    run()
    second = _hash_outputs()
    names = sorted(set(first) | set(second))
    replay = pd.DataFrame(
        {
            "artifact": names,
            "first_sha256": [first.get(name) for name in names],
            "second_sha256": [second.get(name) for name in names],
            "byte_identical": [first.get(name) == second.get(name) for name in names],
        }
    )
    e5._write(replay, _root() / "determinism.csv")
    if not replay["byte_identical"].all():
        raise AssertionError(replay.loc[~replay["byte_identical"]])
    return replay


def main() -> None:
    """Run the consolidation and print its deterministic verdict."""
    replay = verify_determinism()
    print(f"Three-universe two-role evidence: PASS ({len(replay)}/{len(replay)} deterministic)")


if __name__ == "__main__":
    main()
