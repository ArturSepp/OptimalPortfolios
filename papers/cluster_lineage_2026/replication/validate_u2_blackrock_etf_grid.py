"""Independently validate the persisted BlackRock U2 covariance-grid outputs."""
from __future__ import annotations

import pickle

import pandas as pd
from optimalportfolios.alphas.signals.utils import score_within_clusters

import papers.cluster_lineage_2026.replication.run_backtests as e5
import papers.cluster_lineage_2026.replication.run_u1_covar_grid as u1_grid
import papers.cluster_lineage_2026.replication.run_u1_me36_long_short as u1_single
import papers.cluster_lineage_2026.replication.run_u2_blackrock_etf_grid as run
from papers.cluster_lineage_2026.replication.run_e5b import _group_equal_from_ranks
from papers.cluster_lineage_2026.replication.run_u1_global_grid import _backtest


TOLERANCE = 5e-12
PERFORMANCE_METRICS = (
    "net_total_return",
    "net_return_annualized",
    "volatility_annualized",
    "sharpe_rf0",
    "alpha_vs_ew_annualized",
    "beta_vs_ew",
    "one_way_turnover_annualized",
    "cost_drag_bp_per_year",
    "gross_return_annualized",
)


def _assert_close(actual: float, expected: float, label: str) -> None:
    """Require one persisted floating result to match its reconstruction."""
    error = abs(float(actual) - float(expected))
    if error > TOLERANCE:
        raise AssertionError(f"{label}: error={error:.3e} > {TOLERANCE:.1e}")


def _performance(net, gross, ew_nav: pd.Series) -> dict:
    """Compute the frozen metrics without calling the grid's wrapper."""
    payload = e5._performance_row(net, gross, ew_nav)
    payload["gross_return_annualized"] = (
        payload["net_return_annualized"]
        + payload["cost_drag_bp_per_year"] / 10000.0
    )
    return payload


def _long_only_weights(
    scores: pd.DataFrame,
    eligibility: pd.DataFrame,
    groups: pd.DataFrame,
) -> pd.DataFrame:
    """Reconstruct group-equal long-only weights from the stack primitives."""
    ranks = e5._rank_panel(scores, groups)
    weights, _, validation = _group_equal_from_ranks(
        ranks,
        eligibility,
        groups,
        run.SPEC.quantile,
        u1_grid.UNIVERSE,
    )
    if float(validation["weight_sum_abs_error"].max()) > TOLERANCE:
        raise AssertionError("independent long-only weights do not sum to one")
    return weights


def _reconstruct_transfer_rows() -> pd.DataFrame:
    """Recompute the U1-transfer ME/36 global and cluster headline rows."""
    daily = run._read_daily()
    dates = run._dates()
    eligibility = run._eligibility_for_dates(daily, dates)
    signal = run._signal_inputs(daily, dates, eligibility)
    window_dates = dates[
        (dates >= run.HEADLINE_START) & (dates <= run.HEADLINE_END)
    ]
    eligible = eligibility.reindex(index=window_dates)
    prices_all = run._performance_prices(daily)
    prices = run._window_prices(prices_all, window_dates)
    ew_nav = run._ew_reference(
        prices_all,
        eligibility,
        window_dates,
        run.HEADLINE_WINDOW,
    )
    cluster_groups, _ = run._load_partition(
        run.SPEC.covariance_frequency,
        run.SPEC.covariance_span,
    )
    cluster_groups = cluster_groups.reindex(
        index=window_dates, columns=eligible.columns
    )
    global_groups = pd.DataFrame(
        "global", index=window_dates, columns=eligible.columns
    )
    global_scores = signal["global"].reindex(
        index=window_dates, columns=eligible.columns
    ).where(eligible)
    cluster_source = score_within_clusters(
        raw_signal=signal["raw_source"],
        rolling_clusters=run._panel_dict(
            run._load_partition(
                run.SPEC.covariance_frequency,
                run.SPEC.covariance_span,
            )[0]
        ),
        min_cluster_size=run.SPEC.momentum_min_cluster_size,
    )
    cluster_scores = cluster_source.reindex(
        index=window_dates, columns=eligible.columns
    ).where(eligible)

    rows = []
    for leg, scores, groups in (
        ("global", global_scores, global_groups),
        (
            f"cluster_{run._cell_id(run.SPEC.covariance_frequency, run.SPEC.covariance_span)}",
            cluster_scores,
            cluster_groups,
        ),
    ):
        long_weights = _long_only_weights(scores, eligible, groups)
        long_net, long_gross = _backtest(
            prices,
            long_weights,
            run.SPEC.cost_bps / 10000.0,
            f"independent_{leg}_long_only",
        )
        rows.append(
            {
                "strategy": "long_only",
                "leg": leg,
                **_performance(long_net, long_gross, ew_nav),
            }
        )
        short_weights, exposure, validation = u1_single._leg_weights(
            scores, eligible, groups
        )
        if float(exposure["net_exposure"].abs().max()) > TOLERANCE:
            raise AssertionError("independent long-short weights are not neutral")
        if float(validation.filter(like="error").to_numpy().max()) > TOLERANCE:
            raise AssertionError("independent long-short group budgets fail")
        short_net, short_gross = _backtest(
            prices,
            short_weights,
            run.SPEC.cost_bps / 10000.0,
            f"independent_{leg}_long_short",
        )
        rows.append(
            {
                "strategy": "long_short",
                "leg": leg,
                **_performance(short_net, short_gross, ew_nav),
            }
        )
    return pd.DataFrame(rows)


def validate() -> None:
    """Run structural, cache, comparison, and independent payoff checks."""
    root = run._root()
    preflight = pd.read_csv(root / "preflight.csv")
    acceptance = pd.read_csv(root / "acceptance.csv", float_precision="round_trip")
    replay = pd.read_csv(root / "determinism.csv")
    performance = pd.read_csv(
        root / "performance.csv", float_precision="round_trip"
    )
    allocation = pd.read_csv(
        root / "allocation_diagnostics.csv", float_precision="round_trip"
    )
    comparison = pd.read_csv(
        root / "comparison_vs_global.csv", float_precision="round_trip"
    )
    partition_summary = pd.read_csv(
        root / "partition_summary.csv", float_precision="round_trip"
    )
    if not preflight["status"].eq("PASS").all():
        raise AssertionError("persisted preflight contains a failure")
    if len(acceptance) != 116 or not acceptance["status"].eq("PASS").all():
        raise AssertionError("persisted construction acceptance is incomplete")
    if len(replay) != 16 or not replay["byte_identical"].all():
        raise AssertionError("persisted deterministic replay is incomplete")
    if len(performance) != 116 or len(comparison) != 112:
        raise AssertionError("persisted payoff row counts differ from the frozen grid")
    if set(performance["leg"].loc[performance["leg"].eq("global")]) != {"global"}:
        raise AssertionError("global payoff benchmark is missing")
    if performance["leg"].str.contains("EW", case=False).any():
        raise AssertionError("EW-all was incorrectly emitted as a payoff leg")
    if len(allocation) != 812:
        raise AssertionError("asset-class allocation diagnostic is incomplete")
    grouped_allocation = allocation.groupby(
        ["analysis_window", "strategy", "cell_id", "leg"], sort=False
    ).agg(
        long=("average_long_exposure", "sum"),
        short=("average_short_exposure_abs", "sum"),
        net=("average_net_exposure", "sum"),
        current_funds=("funds_in_current_vintage", "sum"),
    )
    if not grouped_allocation["current_funds"].eq(480).all():
        raise AssertionError("asset-class diagnostics do not span all current funds")
    long_only = grouped_allocation.xs("long_only", level="strategy")
    long_short = grouped_allocation.xs("long_short", level="strategy")
    if float(long_only["long"].sub(1.0).abs().max()) > TOLERANCE:
        raise AssertionError("long-only asset-class exposures do not sum to one")
    if float(long_only["short"].abs().max()) > TOLERANCE:
        raise AssertionError("long-only diagnostics contain short exposure")
    if float(long_short["long"].sub(1.0).abs().max()) > TOLERANCE:
        raise AssertionError("long-short long exposures do not sum to one")
    if float(long_short["short"].sub(1.0).abs().max()) > TOLERANCE:
        raise AssertionError("long-short short exposures do not sum to one")
    if float(long_short["net"].abs().max()) > TOLERANCE:
        raise AssertionError("long-short asset-class net exposures do not sum to zero")

    if len(partition_summary) != 28:
        raise AssertionError("partition summary does not contain 28 cells")
    for _, row in partition_summary.iterrows():
        path = run._partition_path(str(row["frequency"]), int(row["span"]))
        with path.open("rb") as stream:
            payload = pickle.load(stream)
        if payload["input_fingerprint"] != run._input_fingerprint():
            raise AssertionError(f"stale input fingerprint in {path.name}")
        if run._partition_hash(payload["panel"]) != row["partition_hash"]:
            raise AssertionError(f"partition hash mismatch in {path.name}")
        if len(payload["panel"]) != 240:
            raise AssertionError(f"partition date count mismatch in {path.name}")

    globals_frame = performance.loc[performance["leg"].eq("global")].set_index(
        ["analysis_window", "strategy"]
    )
    for _, row in comparison.iterrows():
        cluster = performance.loc[
            performance["analysis_window"].eq(row["analysis_window"])
            & performance["strategy"].eq(row["strategy"])
            & performance["cell_id"].eq(row["cell_id"])
            & ~performance["leg"].eq("global")
        ].iloc[0]
        global_row = globals_frame.loc[(row["analysis_window"], row["strategy"])]
        for metric in run.COMPARISON_METRICS:
            _assert_close(
                row[f"delta_{metric}"],
                cluster[metric] - global_row[metric],
                f"comparison {row['strategy']} {row['cell_id']} {metric}",
            )

    reconstructed = _reconstruct_transfer_rows().set_index(["strategy", "leg"])
    persisted = performance.loc[
        performance["analysis_window"].eq(run.HEADLINE_WINDOW)
        & (
            performance["leg"].eq("global")
            | performance["cell_id"].eq(
                run._cell_id(
                    run.SPEC.covariance_frequency,
                    run.SPEC.covariance_span,
                )
            )
        )
    ].set_index(["strategy", "leg"])
    if not reconstructed.index.equals(persisted.index):
        persisted = persisted.reindex(reconstructed.index)
    for key in reconstructed.index:
        for metric in PERFORMANCE_METRICS:
            _assert_close(
                reconstructed.loc[key, metric],
                persisted.loc[key, metric],
                f"independent transfer payoff {key} {metric}",
            )
    print(
        "BlackRock U2 independent validation: PASS "
        "(28 caches, 116 constructions, 4 transfer payoffs, 16 hashes)"
    )


if __name__ == "__main__":
    validate()
