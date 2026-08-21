"""Independently validate standalone futures asset-class long-short portfolios."""
from __future__ import annotations

import pandas as pd

import papers.cluster_lineage_2026.replication.run_backtests as e5
import papers.cluster_lineage_2026.replication.run_futures_asset_class_long_short as run
import papers.cluster_lineage_2026.replication.run_futures_sleeve_grid as equal
from papers.cluster_lineage_2026.replication.run_e5b import _group_equal_from_ranks
from papers.cluster_lineage_2026.replication.run_u1_global_grid import _backtest


TOLERANCE = 5e-12


def _assert_close(actual: float, expected: float, label: str) -> None:
    """Require one floating result to match its independent reconstruction."""
    error = abs(float(actual) - float(expected))
    if error > TOLERANCE:
        raise AssertionError(f"{label}: error={error:.3e} > {TOLERANCE:.1e}")


class _IndependentWindowView:
    """Expose only calendar-bounded NAV and turnover from a qis result."""

    def __init__(self, portfolio) -> None:
        """Store the underlying portfolio result."""
        self._portfolio = portfolio

    @staticmethod
    def _crop(panel):
        """Keep observations inside the U1 calendar interval."""
        return panel.loc[
            (panel.index >= run.WINDOW_START) & (panel.index <= run.WINDOW_END)
        ]

    def get_portfolio_nav(self):
        """Return independently bounded NAV observations."""
        return self._crop(self._portfolio.get_portfolio_nav())

    def get_turnover(self, *args, **kwargs):
        """Return independently bounded turnover observations."""
        return self._crop(self._portfolio.get_turnover(*args, **kwargs))


def _side(
    source: pd.DataFrame,
    eligibility: pd.DataFrame,
    sleeve_panel: pd.DataFrame,
    groups: pd.DataFrame,
    asset_class: str,
) -> pd.DataFrame:
    """Build one independent unit side inside an asset class."""
    ranks = e5._rank_panel(source, groups)
    weights, available, validation = _group_equal_from_ranks(
        ranks,
        eligibility & sleeve_panel.eq(asset_class),
        groups,
        run.PRIMARY_Q,
        equal.UNIVERSE,
    )
    if available.le(0).any():
        raise AssertionError(f"independent {asset_class} side is empty")
    if float(validation.filter(like="error").to_numpy(dtype=float).max()) > TOLERANCE:
        raise AssertionError(f"independent {asset_class} side allocation fails")
    return weights


def _weights(
    scores: pd.DataFrame,
    eligibility: pd.DataFrame,
    sleeve_panel: pd.DataFrame,
    groups: pd.DataFrame,
    asset_class: str,
) -> pd.DataFrame:
    """Build one independent disjoint +1/-1 asset-class book."""
    long_book = _side(scores, eligibility, sleeve_panel, groups, asset_class)
    short_book = _side(-scores, eligibility, sleeve_panel, groups, asset_class)
    overlap = long_book.gt(0.0) & short_book.gt(0.0)
    long_book = long_book.mask(overlap, 0.0)
    short_book = short_book.mask(overlap, 0.0)
    long_book = long_book.div(long_book.sum(axis=1), axis=0)
    short_book = short_book.div(short_book.sum(axis=1), axis=0)
    weights = long_book - short_book
    if float(weights.sum(axis=1).abs().max()) > TOLERANCE:
        raise AssertionError(f"independent {asset_class} net exposure fails")
    if float(weights.abs().sum(axis=1).sub(2.0).abs().max()) > TOLERANCE:
        raise AssertionError(f"independent {asset_class} gross exposure fails")
    if float(weights.where(~sleeve_panel.eq(asset_class), 0.0).abs().to_numpy().max()) > 0.0:
        raise AssertionError(f"independent {asset_class} weights leak across classes")
    return weights


def _reconstruct_primary_payoffs() -> pd.DataFrame:
    """Recompute all twelve primary asset-class payoff rows independently."""
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
    baseline = equal._hierarchical_groups(
        e5._cluster_groups(equal.UNIVERSE, e5.SmootherName.BASELINE).reindex(
            index=dates, columns=eligibility.columns
        ),
        sleeve_panel,
    )
    m1_star = equal._hierarchical_groups(
        e5._cluster_groups(equal.UNIVERSE, e5.SmootherName.M1_STAR).reindex(
            index=dates, columns=eligibility.columns
        ),
        sleeve_panel,
    )
    groups_by_method = {
        "global": sleeve_panel,
        "cluster_baseline": baseline,
        "cluster_M1_star": m1_star,
    }
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
    for asset_class in run.ASSET_CLASSES:
        for method, groups in groups_by_method.items():
            weights = _weights(
                scores, eligibility, sleeve_panel, groups, asset_class
            )
            net, gross = _backtest(
                prices,
                weights,
                costs,
                f"independent_{asset_class}_{method}_long_short",
            )
            net_view = _IndependentWindowView(net)
            gross_view = _IndependentWindowView(gross)
            payload = e5._performance_row(net_view, gross_view, ew_nav)
            payload["gross_return_annualized"] = (
                payload["net_return_annualized"]
                + payload["cost_drag_bp_per_year"] / 10000.0
            )
            rows.append(
                {"asset_class": asset_class, "method": method, **payload}
            )
    return pd.DataFrame(rows).set_index(["asset_class", "method"])


def _validate_comparison(
    performance: pd.DataFrame, comparison: pd.DataFrame
) -> None:
    """Recompute every cluster-minus-global asset-class delta."""
    global_rows = performance.loc[performance["method"].eq("global")].set_index(
        ["asset_class", "q"]
    )
    for _, row in comparison.iterrows():
        key = (row["asset_class"], row["q"])
        for metric in equal.COMPARISON_METRICS:
            _assert_close(
                row[f"global_{metric}"],
                global_rows.loc[key, metric],
                f"global source {key} {row['method']} {metric}",
            )
            _assert_close(
                row[f"delta_vs_global_{metric}"],
                row[metric] - global_rows.loc[key, metric],
                f"global delta {key} {row['method']} {metric}",
            )


def validate() -> None:
    """Run structural, horizon, arithmetic, replay, and payoff checks."""
    root = run._root()
    design = pd.read_csv(root / "design.csv")
    performance = pd.read_csv(
        root / "performance.csv", float_precision="round_trip"
    )
    comparison = pd.read_csv(
        root / "comparison.csv", float_precision="round_trip"
    )
    construction = pd.read_csv(
        root / "construction_diagnostics.csv", float_precision="round_trip"
    )
    acceptance = pd.read_csv(
        root / "acceptance.csv", float_precision="round_trip"
    )
    horizon = pd.read_csv(
        root / "horizon_diagnostic.csv",
        parse_dates=["nav_start", "nav_end"],
        float_precision="round_trip",
    )
    reconstruction = pd.read_csv(
        root / "combined_weight_reconstruction.csv", float_precision="round_trip"
    )
    replay = pd.read_csv(root / "determinism.csv")
    expected = {
        "design": (len(design), 4),
        "performance": (len(performance), 24),
        "comparison": (len(comparison), 16),
        "construction": (len(construction), 24),
        "acceptance": (len(acceptance), 24),
        "horizon": (len(horizon), 24),
        "reconstruction": (len(reconstruction), 6),
        "replay": (len(replay), 7),
    }
    failures = {
        name: (actual, target)
        for name, (actual, target) in expected.items()
        if actual != target
    }
    if failures:
        raise AssertionError(f"persisted row counts fail: {failures}")
    expected_contracts = {
        "Equity": 29,
        "Fixed Income": 21,
        "Commodities": 33,
        "FX": 11,
    }
    measured_contracts = design.set_index("asset_class")[
        "contracts_ever_eligible"
    ].to_dict()
    if measured_contracts != expected_contracts:
        raise AssertionError(f"asset-class counts fail: {measured_contracts}")
    if not acceptance["status"].eq("PASS").all():
        raise AssertionError("construction acceptance contains a failure")
    if not reconstruction["status"].eq("PASS").all():
        raise AssertionError("combined-book weight reconstruction contains a failure")
    if not replay["byte_identical"].all():
        raise AssertionError("deterministic replay contains a failure")
    if performance["method"].str.contains("EW", case=False).any():
        raise AssertionError("EW-all was emitted as a payoff leg")
    if not horizon["nav_start"].between(
        run.WINDOW_START, run.WINDOW_START + pd.Timedelta(days=7)
    ).all():
        raise AssertionError("NAV starts outside the common window")
    if not horizon["nav_end"].between(
        run.WINDOW_END - pd.Timedelta(days=7), run.WINDOW_END
    ).all():
        raise AssertionError("NAV ends outside the common window")
    if not horizon["pre_window_nav_rows"].eq(0).all():
        raise AssertionError("pre-window NAV observations remain")
    if not horizon["post_window_nav_rows"].eq(0).all():
        raise AssertionError("post-window NAV observations remain")

    _validate_comparison(performance, comparison)
    reconstructed = _reconstruct_primary_payoffs()
    persisted = performance.loc[performance["q"].eq(run.PRIMARY_Q)].set_index(
        ["asset_class", "method"]
    )
    for key in reconstructed.index:
        for metric in equal.COMPARISON_METRICS:
            _assert_close(
                reconstructed.loc[key, metric],
                persisted.loc[key, metric],
                f"independent primary payoff {key} {metric}",
            )
    print(
        "Futures asset-class long-short independent validation: PASS "
        "(4 classes, 24 portfolios, 16 comparisons, 12 reconstructed payoffs, "
        "6 combined-weight identities, 7 hashes)"
    )


if __name__ == "__main__":
    validate()
