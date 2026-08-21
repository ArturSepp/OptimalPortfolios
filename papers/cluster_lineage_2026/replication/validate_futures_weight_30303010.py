"""Independently validate the futures 30/30/30/10 sleeve experiment."""
from __future__ import annotations

import pandas as pd

import papers.cluster_lineage_2026.replication.run_backtests as e5
import papers.cluster_lineage_2026.replication.run_futures_sleeve_grid as equal
import papers.cluster_lineage_2026.replication.run_futures_weight_30303010 as run
from papers.cluster_lineage_2026.replication.run_e5b import _group_equal_from_ranks
from papers.cluster_lineage_2026.replication.run_u1_global_grid import _backtest


TOLERANCE = 5e-12


def _assert_close(actual: float, expected: float, label: str) -> None:
    """Require one floating result to match its independent reconstruction."""
    error = abs(float(actual) - float(expected))
    if error > TOLERANCE:
        raise AssertionError(f"{label}: error={error:.3e} > {TOLERANCE:.1e}")


def _independent_long_only_weights(
    scores: pd.DataFrame,
    eligibility: pd.DataFrame,
    sleeve_panel: pd.DataFrame,
    groups: pd.DataFrame,
) -> pd.DataFrame:
    """Assemble the target budgets directly from accepted rank primitives."""
    ranks = e5._rank_panel(scores, groups)
    output = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)
    for sleeve in equal.SLEEVES:
        sleeve_eligibility = eligibility & sleeve_panel.eq(sleeve)
        weights, available, validation = _group_equal_from_ranks(
            ranks,
            sleeve_eligibility,
            groups,
            equal.PRIMARY_Q,
            equal.UNIVERSE,
        )
        if available.le(0).any():
            raise AssertionError(f"independent {sleeve} reconstruction is empty")
        errors = validation.filter(like="error")
        if float(errors.to_numpy(dtype=float).max()) > TOLERANCE:
            raise AssertionError(f"independent {sleeve} group allocation fails")
        output = output.add(weights.mul(run.TARGET[sleeve]), fill_value=0.0)
    return output


def _reconstruct_primary_long_only() -> pd.DataFrame:
    """Recompute target-budget global and M1-star primary payoff rows."""
    data = e5.load_universe(equal.UNIVERSE)
    dates = e5.load_cached(equal.UNIVERSE, e5.SmootherName.BASELINE).dates
    eligibility = e5._investable_eligibility(data, dates)
    scores = e5._raw_momentum_scores(
        data, dates, vol_adjusted=False
    ).reindex(columns=eligibility.columns).where(eligibility)
    prices = e5._prices(data).reindex(columns=eligibility.columns)
    sleeves = equal._broad_sleeves(data.taxonomy, eligibility.columns)
    sleeve_panel = equal._sleeve_panel(dates, sleeves)
    clusters = e5._cluster_groups(
        equal.UNIVERSE, e5.SmootherName.M1_STAR
    ).reindex(index=dates, columns=eligibility.columns)
    hierarchical = equal._hierarchical_groups(clusters, sleeve_panel)
    ew_nav = pd.read_csv(
        equal._accepted_root() / "navs.csv",
        parse_dates=["date"],
        float_precision="round_trip",
    ).set_index("date")["EW_all"]
    costs = e5.get_universe_spec(equal.UNIVERSE).cost_bps / 10000.0
    rows = []
    for method, groups in (
        ("sleeve_global", sleeve_panel),
        ("sleeve_cluster_M1_star", hierarchical),
    ):
        weights = _independent_long_only_weights(
            scores, eligibility, sleeve_panel, groups
        )
        if float(weights.sum(axis=1).sub(1.0).abs().max()) > TOLERANCE:
            raise AssertionError(f"independent {method} weights do not sum to one")
        for sleeve in equal.SLEEVES:
            measured = weights.where(sleeve_panel.eq(sleeve), 0.0).sum(axis=1)
            if float(measured.sub(run.TARGET[sleeve]).abs().max()) > TOLERANCE:
                raise AssertionError(f"independent {method} {sleeve} budget fails")
        net, gross = _backtest(
            prices,
            weights,
            costs,
            f"independent_futures_30303010_{method}_q_020",
        )
        payload = e5._performance_row(net, gross, ew_nav)
        payload["gross_return_annualized"] = (
            payload["net_return_annualized"]
            + payload["cost_drag_bp_per_year"] / 10000.0
        )
        rows.append({"method": method, **payload})
    return pd.DataFrame(rows).set_index("method")


def _validate_cluster_comparison(
    performance: pd.DataFrame, comparison: pd.DataFrame
) -> None:
    """Recompute every cluster-minus-global comparison delta."""
    fair = performance.loc[performance["method"].eq("sleeve_global")].set_index(
        ["strategy", "q"]
    )
    original = performance.loc[
        performance["method"].eq("original_global")
    ].set_index(["strategy", "q"])
    for _, row in comparison.iterrows():
        key = (row["strategy"], row["q"])
        for metric in equal.COMPARISON_METRICS:
            _assert_close(
                row[f"delta_vs_sleeve_global_{metric}"],
                row[metric] - fair.loc[key, metric],
                f"same-budget delta {key} {row['method']} {metric}",
            )
            _assert_close(
                row[f"delta_vs_original_global_{metric}"],
                row[metric] - original.loc[key, metric],
                f"original delta {key} {row['method']} {metric}",
            )


def _validate_equal_sleeve_comparison(comparison: pd.DataFrame) -> None:
    """Recompute every scenario-minus-equal-sleeve delta from source outputs."""
    benchmark = pd.read_csv(
        equal._root() / "performance.csv", float_precision="round_trip"
    ).set_index(["strategy", "q", "method"])
    for _, row in comparison.iterrows():
        key = (row["strategy"], row["q"], row["method"])
        for metric in equal.COMPARISON_METRICS:
            _assert_close(
                row[f"equal_sleeve_{metric}"],
                benchmark.loc[key, metric],
                f"equal-sleeve source {key} {metric}",
            )
            _assert_close(
                row[f"delta_vs_equal_sleeves_{metric}"],
                row[metric] - benchmark.loc[key, metric],
                f"equal-sleeve delta {key} {metric}",
            )


def validate() -> None:
    """Run structural, budget, arithmetic, replay, and independent payoff checks."""
    root = run._root()
    design = pd.read_csv(root / "design.csv")
    performance = pd.read_csv(
        root / "performance.csv", float_precision="round_trip"
    )
    comparison = pd.read_csv(
        root / "comparison.csv", float_precision="round_trip"
    )
    equal_comparison = pd.read_csv(
        root / "comparison_vs_equal_sleeves.csv", float_precision="round_trip"
    )
    allocation = pd.read_csv(
        root / "allocation_diagnostics.csv", float_precision="round_trip"
    )
    acceptance = pd.read_csv(
        root / "acceptance.csv", float_precision="round_trip"
    )
    regression = pd.read_csv(
        root / "global_regression.csv", float_precision="round_trip"
    )
    replay = pd.read_csv(root / "determinism.csv")
    expected = {
        "design": (len(design), 1),
        "performance": (len(performance), 16),
        "comparison": (len(comparison), 8),
        "equal_comparison": (len(equal_comparison), 16),
        "allocation": (len(allocation), 64),
        "acceptance": (len(acceptance), 16),
        "regression": (len(regression), 1),
        "replay": (len(replay), 7),
    }
    failures = {
        name: (actual, target)
        for name, (actual, target) in expected.items()
        if actual != target
    }
    if failures:
        raise AssertionError(f"persisted row counts fail: {failures}")
    if not acceptance["status"].eq("PASS").all():
        raise AssertionError("persisted construction acceptance contains a failure")
    if not regression["status"].eq("PASS").all():
        raise AssertionError("accepted global regression contains a failure")
    if not replay["byte_identical"].all():
        raise AssertionError("persisted deterministic replay contains a failure")
    if performance["method"].str.contains("EW", case=False).any():
        raise AssertionError("EW-all was incorrectly emitted as a ranking leg")

    constrained = allocation.loc[~allocation["method"].eq("original_global")]
    for sleeve, target in run.TARGET.items():
        rows = constrained.loc[constrained["sleeve"].eq(sleeve)]
        if float(rows["target_budget"].sub(target).abs().max()) > TOLERANCE:
            raise AssertionError(f"persisted {sleeve} target label fails")
        if float(rows["mean_long_exposure"].sub(target).abs().max()) > TOLERANCE:
            raise AssertionError(f"persisted {sleeve} long budget fails")
        short = rows.loc[rows["strategy"].eq("long_short")]
        if float(short["mean_short_exposure_abs"].sub(target).abs().max()) > TOLERANCE:
            raise AssertionError(f"persisted {sleeve} short budget fails")

    _validate_cluster_comparison(performance, comparison)
    _validate_equal_sleeve_comparison(equal_comparison)
    reconstructed = _reconstruct_primary_long_only()
    persisted = performance.loc[
        performance["strategy"].eq("long_only")
        & performance["q"].eq(equal.PRIMARY_Q)
        & performance["method"].isin(reconstructed.index)
    ].set_index("method")
    for method in reconstructed.index:
        for metric in equal.COMPARISON_METRICS:
            _assert_close(
                reconstructed.loc[method, metric],
                persisted.loc[method, metric],
                f"independent primary payoff {method} {metric}",
            )
    print(
        "Futures 30/30/30/10 independent validation: PASS "
        "(95 contracts, 16 portfolios, 8 fair comparisons, 16 budget comparisons, "
        "2 reconstructed payoffs, 7 hashes)"
    )


if __name__ == "__main__":
    validate()
