"""Independently validate U1-window futures long-only and long-short results."""
from __future__ import annotations

import pandas as pd

import papers.cluster_lineage_2026.replication.run_backtests as e5
import papers.cluster_lineage_2026.replication.run_futures_sleeve_grid as equal
import papers.cluster_lineage_2026.replication.run_futures_weight_30303010_u1_window as run
from papers.cluster_lineage_2026.replication.run_e5b import _group_equal_from_ranks
from papers.cluster_lineage_2026.replication.run_u1_global_grid import _backtest


TOLERANCE = 5e-12


def _assert_close(actual: float, expected: float, label: str) -> None:
    """Require one floating result to match its independent reconstruction."""
    error = abs(float(actual) - float(expected))
    if error > TOLERANCE:
        raise AssertionError(f"{label}: error={error:.3e} > {TOLERANCE:.1e}")


class _IndependentWindowView:
    """Crop a qis result without using the production runner's view class."""

    def __init__(self, portfolio) -> None:
        """Store the underlying portfolio result."""
        self._portfolio = portfolio

    @staticmethod
    def _crop(panel):
        """Keep observations inside the common calendar window."""
        return panel.loc[
            (panel.index >= run.WINDOW_START) & (panel.index <= run.WINDOW_END)
        ]

    def get_portfolio_nav(self):
        """Return independently cropped NAV observations."""
        return self._crop(self._portfolio.get_portfolio_nav())

    def get_turnover(self, *args, **kwargs):
        """Return independently cropped turnover observations."""
        return self._crop(self._portfolio.get_turnover(*args, **kwargs))


def _independent_side(
    source: pd.DataFrame,
    eligibility: pd.DataFrame,
    sleeve_panel: pd.DataFrame,
    groups: pd.DataFrame,
) -> pd.DataFrame:
    """Build one signed side directly from accepted group-rank primitives."""
    ranks = e5._rank_panel(source, groups)
    output = pd.DataFrame(0.0, index=source.index, columns=source.columns)
    for sleeve in equal.SLEEVES:
        eligible = eligibility & sleeve_panel.eq(sleeve)
        weights, available, validation = _group_equal_from_ranks(
            ranks,
            eligible,
            groups,
            equal.PRIMARY_Q,
            equal.UNIVERSE,
        )
        if available.le(0).any():
            raise AssertionError(f"independent {sleeve} side is empty")
        errors = validation.filter(like="error")
        if float(errors.to_numpy(dtype=float).max()) > TOLERANCE:
            raise AssertionError(f"independent {sleeve} group allocation fails")
        output = output.add(weights.mul(run.TARGET[sleeve]), fill_value=0.0)
    return output


def _renormalize_independently(
    side: pd.DataFrame, sleeve_panel: pd.DataFrame
) -> pd.DataFrame:
    """Restore each strategic sleeve budget after overlap removal."""
    output = pd.DataFrame(0.0, index=side.index, columns=side.columns)
    for sleeve in equal.SLEEVES:
        component = side.where(sleeve_panel.eq(sleeve), 0.0)
        total = component.sum(axis=1)
        if total.le(0.0).any():
            raise AssertionError(f"independent {sleeve} side is empty after overlap")
        output = output.add(
            component.div(total, axis=0).mul(run.TARGET[sleeve]), fill_value=0.0
        )
    return output


def _independent_weights(
    strategy: str,
    scores: pd.DataFrame,
    eligibility: pd.DataFrame,
    sleeve_panel: pd.DataFrame,
    groups: pd.DataFrame,
) -> pd.DataFrame:
    """Build one independent long-only or disjoint +1/-1 portfolio."""
    long_book = _independent_side(scores, eligibility, sleeve_panel, groups)
    if strategy == "long_only":
        return long_book
    short_book = _independent_side(-scores, eligibility, sleeve_panel, groups)
    overlap = long_book.gt(0.0) & short_book.gt(0.0)
    long_book = _renormalize_independently(
        long_book.mask(overlap, 0.0), sleeve_panel
    )
    short_book = _renormalize_independently(
        short_book.mask(overlap, 0.0), sleeve_panel
    )
    return long_book - short_book


def _reconstruct_primary_payoffs() -> pd.DataFrame:
    """Recompute two methods under both strategy constructions."""
    data = e5.load_universe(equal.UNIVERSE)
    all_dates = e5.load_cached(equal.UNIVERSE, e5.SmootherName.BASELINE).dates
    dates = all_dates[
        (all_dates >= run.WINDOW_START) & (all_dates <= run.WINDOW_END)
    ]
    if len(dates) != 203:
        raise AssertionError(f"independent decision count is {len(dates)}, not 203")
    eligibility = e5._investable_eligibility(data, dates)
    scores = e5._raw_momentum_scores(
        data, dates, vol_adjusted=False
    ).reindex(columns=eligibility.columns).where(eligibility)
    full_prices = e5._prices(data).reindex(columns=eligibility.columns)
    prior = full_prices.loc[full_prices.index <= run.WINDOW_START].tail(1)
    prices = pd.concat(
        [
            prior,
            full_prices.loc[
                (full_prices.index > run.WINDOW_START)
                & (full_prices.index <= run.WINDOW_END)
            ],
        ]
    ).sort_index()
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
    ew_nav = ew_nav.loc[
        (ew_nav.index >= run.WINDOW_START) & (ew_nav.index <= run.WINDOW_END)
    ]
    costs = e5.get_universe_spec(equal.UNIVERSE).cost_bps / 10000.0
    rows = []
    for strategy in ("long_only", "long_short"):
        for method, groups in (
            ("sleeve_global", sleeve_panel),
            ("sleeve_cluster_M1_star", hierarchical),
        ):
            weights = _independent_weights(
                strategy, scores, eligibility, sleeve_panel, groups
            )
            target_sum = 1.0 if strategy == "long_only" else 0.0
            if float(weights.sum(axis=1).sub(target_sum).abs().max()) > TOLERANCE:
                raise AssertionError(f"independent {strategy} {method} net exposure fails")
            net, gross = _backtest(
                prices,
                weights,
                costs,
                f"independent_u1_window_{strategy}_{method}",
            )
            net_view = _IndependentWindowView(net)
            gross_view = _IndependentWindowView(gross)
            payload = e5._performance_row(net_view, gross_view, ew_nav)
            payload["gross_return_annualized"] = (
                payload["net_return_annualized"]
                + payload["cost_drag_bp_per_year"] / 10000.0
            )
            rows.append(
                {"strategy": strategy, "method": method, **payload}
            )
    return pd.DataFrame(rows).set_index(["strategy", "method"])


def _validate_comparison(
    performance: pd.DataFrame, comparison: pd.DataFrame
) -> None:
    """Recompute cluster-minus-global arithmetic for one strategy table."""
    fair = performance.loc[performance["method"].eq("sleeve_global")].set_index("q")
    original = performance.loc[
        performance["method"].eq("original_global")
    ].set_index("q")
    for _, row in comparison.iterrows():
        q = row["q"]
        for metric in equal.COMPARISON_METRICS:
            _assert_close(
                row[f"delta_vs_sleeve_global_{metric}"],
                row[metric] - fair.loc[q, metric],
                f"same-budget delta {row['strategy']} {q} {row['method']} {metric}",
            )
            _assert_close(
                row[f"delta_vs_original_global_{metric}"],
                row[metric] - original.loc[q, metric],
                f"original delta {row['strategy']} {q} {row['method']} {metric}",
            )


def validate() -> None:
    """Run horizon, construction, arithmetic, replay, and payoff checks."""
    root = run._root()
    design = pd.read_csv(root / "design.csv")
    long_only = pd.read_csv(
        root / "performance_long_only.csv", float_precision="round_trip"
    )
    long_short = pd.read_csv(
        root / "performance_long_short.csv", float_precision="round_trip"
    )
    comparison_long_only = pd.read_csv(
        root / "comparison_long_only.csv", float_precision="round_trip"
    )
    comparison_long_short = pd.read_csv(
        root / "comparison_long_short.csv", float_precision="round_trip"
    )
    allocation = pd.read_csv(
        root / "allocation_diagnostics.csv", float_precision="round_trip"
    )
    acceptance = pd.read_csv(
        root / "acceptance.csv", float_precision="round_trip"
    )
    horizon = pd.read_csv(
        root / "horizon_diagnostic.csv",
        parse_dates=["nav_start", "nav_end"],
        float_precision="round_trip",
    )
    regression = pd.read_csv(
        root / "global_weight_regression.csv", float_precision="round_trip"
    )
    legacy = pd.read_csv(
        root / "legacy_horizon_diagnostic.csv", float_precision="round_trip"
    )
    u1_reference = pd.read_csv(
        root / "u1_reference_horizon_diagnostic.csv",
        parse_dates=[
            "u1_artifact_nav_start",
            "u1_artifact_first_active_nav",
            "u1_artifact_nav_end",
        ],
        float_precision="round_trip",
    )
    replay = pd.read_csv(root / "determinism.csv")
    expected = {
        "design": (len(design), 1),
        "long_only": (len(long_only), 8),
        "long_short": (len(long_short), 8),
        "comparison_long_only": (len(comparison_long_only), 4),
        "comparison_long_short": (len(comparison_long_short), 4),
        "allocation": (len(allocation), 64),
        "acceptance": (len(acceptance), 16),
        "horizon": (len(horizon), 16),
        "regression": (len(regression), 1),
        "legacy": (len(legacy), 1),
        "u1_reference": (len(u1_reference), 1),
        "replay": (len(replay), 11),
    }
    failures = {
        name: (actual, target)
        for name, (actual, target) in expected.items()
        if actual != target
    }
    if failures:
        raise AssertionError(f"persisted row counts fail: {failures}")
    if not acceptance["status"].eq("PASS").all():
        raise AssertionError("construction acceptance contains a failure")
    if not regression["status"].eq("PASS").all():
        raise AssertionError("global decision regression contains a failure")
    if not replay["byte_identical"].all():
        raise AssertionError("deterministic replay contains a failure")
    if not legacy["status"].eq("SUPERSEDED_PRE_STRATEGY_CASH_HISTORY").all():
        raise AssertionError("legacy horizon was not marked superseded")
    expected_u1_status = "REMEASURE_U1_BEFORE_CROSS_UNIVERSE_PAYOFF_COMPARISON"
    if not u1_reference["status"].eq(expected_u1_status).all():
        raise AssertionError("accepted U1 horizon diagnostic status fails")
    if not u1_reference["u1_artifact_nav_start"].lt(run.WINDOW_START).all():
        raise AssertionError("accepted U1 artifact did not expose its pre-window NAV")
    if not horizon["nav_start"].between(
        run.WINDOW_START, run.WINDOW_START + pd.Timedelta(days=7)
    ).all():
        raise AssertionError("NAV starts outside the U1 window boundary")
    if not horizon["nav_end"].between(
        run.WINDOW_END - pd.Timedelta(days=7), run.WINDOW_END
    ).all():
        raise AssertionError("NAV ends outside the U1 window boundary")
    if not horizon["pre_window_nav_rows"].eq(0).all():
        raise AssertionError("pre-window NAV rows remain")
    if not horizon["post_window_nav_rows"].eq(0).all():
        raise AssertionError("post-window NAV rows remain")

    constrained = allocation.loc[~allocation["method"].eq("original_global")]
    for sleeve, target in run.TARGET.items():
        rows = constrained.loc[constrained["sleeve"].eq(sleeve)]
        if float(rows["mean_long_exposure"].sub(target).abs().max()) > TOLERANCE:
            raise AssertionError(f"persisted {sleeve} long budget fails")
        short = rows.loc[rows["strategy"].eq("long_short")]
        if float(short["mean_short_exposure_abs"].sub(target).abs().max()) > TOLERANCE:
            raise AssertionError(f"persisted {sleeve} short budget fails")

    _validate_comparison(long_only, comparison_long_only)
    _validate_comparison(long_short, comparison_long_short)
    reconstructed = _reconstruct_primary_payoffs()
    persisted = pd.concat([long_only, long_short]).loc[
        lambda frame: frame["q"].eq(equal.PRIMARY_Q)
        & frame["method"].isin(reconstructed.index.get_level_values("method"))
    ].set_index(["strategy", "method"])
    for key in reconstructed.index:
        for metric in equal.COMPARISON_METRICS:
            _assert_close(
                reconstructed.loc[key, metric],
                persisted.loc[key, metric],
                f"independent primary payoff {key} {metric}",
            )
    print(
        "Futures 30/30/30/10 U1-window independent validation: PASS "
        "(203 decisions, 16 portfolios, 8 comparisons, 4 reconstructed payoffs, "
        "11 hashes)"
    )


if __name__ == "__main__":
    validate()
