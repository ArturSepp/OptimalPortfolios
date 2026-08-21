"""Independently validate the futures four-sleeve rank experiment."""
from __future__ import annotations

import pandas as pd

import papers.cluster_lineage_2026.replication.run_backtests as e5
import papers.cluster_lineage_2026.replication.run_futures_sleeve_grid as run
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
    """Assemble four exact sleeve budgets directly from accepted rank primitives."""
    ranks = e5._rank_panel(scores, groups)
    output = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)
    for sleeve in run.SLEEVES:
        sleeve_eligibility = eligibility & sleeve_panel.eq(sleeve)
        weights, available, validation = _group_equal_from_ranks(
            ranks,
            sleeve_eligibility,
            groups,
            run.PRIMARY_Q,
            run.UNIVERSE,
        )
        if available.le(0).any():
            raise AssertionError(f"independent {sleeve} reconstruction is empty")
        errors = validation.filter(like="error")
        if float(errors.to_numpy(dtype=float).max()) > TOLERANCE:
            raise AssertionError(f"independent {sleeve} group allocation fails")
        output = output.add(weights.mul(run.TARGET[sleeve]), fill_value=0.0)
    return output


def _reconstruct_primary_long_only() -> pd.DataFrame:
    """Recompute fair global and M1-star primary long-only payoff rows."""
    data = e5.load_universe(run.UNIVERSE)
    dates = e5.load_cached(run.UNIVERSE, e5.SmootherName.BASELINE).dates
    eligibility = e5._investable_eligibility(data, dates)
    scores = e5._raw_momentum_scores(
        data, dates, vol_adjusted=False
    ).reindex(columns=eligibility.columns).where(eligibility)
    prices = e5._prices(data).reindex(columns=eligibility.columns)
    sleeves = run._broad_sleeves(data.taxonomy, eligibility.columns)
    sleeve_panel = run._sleeve_panel(dates, sleeves)
    clusters = e5._cluster_groups(
        run.UNIVERSE, e5.SmootherName.M1_STAR
    ).reindex(index=dates, columns=eligibility.columns)
    hierarchical = run._hierarchical_groups(clusters, sleeve_panel)
    ew_nav = pd.read_csv(
        run._accepted_root() / "navs.csv",
        parse_dates=["date"],
        float_precision="round_trip",
    ).set_index("date")["EW_all"]
    costs = e5.get_universe_spec(run.UNIVERSE).cost_bps / 10000.0
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
        for sleeve in run.SLEEVES:
            measured = weights.where(sleeve_panel.eq(sleeve), 0.0).sum(axis=1)
            if float(measured.sub(run.TARGET[sleeve]).abs().max()) > TOLERANCE:
                raise AssertionError(f"independent {method} {sleeve} budget fails")
        net, gross = _backtest(
            prices,
            weights,
            costs,
            f"independent_futures_{method}_q_020",
        )
        payload = e5._performance_row(net, gross, ew_nav)
        payload["gross_return_annualized"] = (
            payload["net_return_annualized"]
            + payload["cost_drag_bp_per_year"] / 10000.0
        )
        rows.append({"method": method, **payload})
    return pd.DataFrame(rows).set_index("method")


def _validate_comparison_arithmetic(
    performance: pd.DataFrame, comparison: pd.DataFrame
) -> None:
    """Recompute every persisted cluster-minus-global comparison delta."""
    fair = performance.loc[performance["method"].eq("sleeve_global")].set_index(
        ["strategy", "q"]
    )
    original = performance.loc[
        performance["method"].eq("original_global")
    ].set_index(["strategy", "q"])
    for _, row in comparison.iterrows():
        key = (row["strategy"], row["q"])
        for metric in run.COMPARISON_METRICS:
            _assert_close(
                row[f"delta_vs_sleeve_global_{metric}"],
                row[metric] - fair.loc[key, metric],
                f"fair delta {key} {row['method']} {metric}",
            )
            _assert_close(
                row[f"delta_vs_original_global_{metric}"],
                row[metric] - original.loc[key, metric],
                f"original delta {key} {row['method']} {metric}",
            )


def validate() -> None:
    """Run structural, budget, regression, arithmetic, and payoff checks."""
    root = run._root()
    design = pd.read_csv(root / "design.csv")
    exposure = pd.read_csv(
        root / "global_exposure_diagnostic.csv", float_precision="round_trip"
    )
    performance = pd.read_csv(
        root / "performance.csv", float_precision="round_trip"
    )
    comparison = pd.read_csv(
        root / "comparison.csv", float_precision="round_trip"
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
        "exposure": (len(exposure), 4),
        "performance": (len(performance), 16),
        "comparison": (len(comparison), 8),
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
    if not exposure["equal_sleeve_trigger"].all():
        raise AssertionError("persisted global concentration did not trigger follow-up")
    if not acceptance["status"].eq("PASS").all():
        raise AssertionError("persisted construction acceptance contains a failure")
    if not regression["status"].eq("PASS").all():
        raise AssertionError("accepted global regression contains a failure")
    if not replay["byte_identical"].all():
        raise AssertionError("persisted deterministic replay contains a failure")
    if performance["method"].str.contains("EW", case=False).any():
        raise AssertionError("EW-all was incorrectly emitted as a ranking leg")

    equal = allocation.loc[~allocation["method"].eq("original_global")]
    long_only = equal.loc[equal["strategy"].eq("long_only")]
    if float(long_only["mean_long_exposure"].sub(0.25).abs().max()) > TOLERANCE:
        raise AssertionError("long-only sleeve budgets fail")
    if float(long_only["mean_short_exposure_abs"].abs().max()) > TOLERANCE:
        raise AssertionError("long-only sleeve diagnostics contain shorts")
    long_short = equal.loc[equal["strategy"].eq("long_short")]
    for column in ("mean_long_exposure", "mean_short_exposure_abs"):
        if float(long_short[column].sub(0.25).abs().max()) > TOLERANCE:
            raise AssertionError(f"long-short sleeve budget fails for {column}")
    if float(long_short["mean_net_exposure"].abs().max()) > TOLERANCE:
        raise AssertionError("long-short sleeve neutrality fails")

    _validate_comparison_arithmetic(performance, comparison)
    reconstructed = _reconstruct_primary_long_only()
    persisted = performance.loc[
        performance["strategy"].eq("long_only")
        & performance["q"].eq(run.PRIMARY_Q)
        & performance["method"].isin(reconstructed.index)
    ].set_index("method")
    for method in reconstructed.index:
        for metric in run.COMPARISON_METRICS:
            _assert_close(
                reconstructed.loc[method, metric],
                persisted.loc[method, metric],
                f"independent primary payoff {method} {metric}",
            )
    print(
        "Futures four-sleeve independent validation: PASS "
        "(95 contracts, 16 portfolios, 8 comparisons, 2 reconstructed payoffs, "
        "7 hashes)"
    )


if __name__ == "__main__":
    validate()
