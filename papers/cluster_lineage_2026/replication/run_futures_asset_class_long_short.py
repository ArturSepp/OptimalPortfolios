"""Run standalone futures long-short books for four broad asset classes.

Equity, Fixed Income, Commodities, and FX are tested independently.  Within each class,
the global control ranks all eligible contracts together; baseline and M1-star treatments
rank within the accepted correlation clusters.  Every portfolio is a self-contained
+1/-1 book, so performance differences are not contaminated by strategic sleeve weights.

The corrected U1 calendar window, production 48-week-minus-4-week signal, monthly
decisions, one-observation implementation lag, 20 bp costs, q=0.20 primary, and q=0.25
robustness are frozen.  One real pre-window W-WED mark is retained only for alignment;
all reported NAV and turnover statistics are cropped to the stated calendar window.
EW-all is reference-only for beta and alpha and is never a payoff comparator.
"""
from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

import papers.cluster_lineage_2026.replication.run_backtests as e5
import papers.cluster_lineage_2026.replication.run_futures_sleeve_grid as equal
import papers.cluster_lineage_2026.replication.run_futures_weight_30303010_u1_window as matched
from papers.cluster_lineage_2026.replication.run_e5b import _group_equal_from_ranks


ASSET_CLASSES = equal.SLEEVES
PRIMARY_Q = equal.PRIMARY_Q
QUANTILES = equal.QUANTILES
WINDOW = matched.WINDOW
WINDOW_START = matched.WINDOW_START
WINDOW_END = matched.WINDOW_END
METHODS = ("global", "cluster_baseline", "cluster_M1_star")
RUNNER = (
    "papers/cluster_lineage_2026/replication/"
    "run_futures_asset_class_long_short.py"
)
RECONSTRUCTION_TOLERANCE = 1e-12


def _root() -> Path:
    """Return and create the external standalone asset-class output directory."""
    return e5.get_output_path(
        "e5b", "futures_asset_class_long_short_u1_window", create=True
    )


def _one_side(
    source: pd.DataFrame,
    eligibility: pd.DataFrame,
    sleeve_panel: pd.DataFrame,
    groups: pd.DataFrame,
    asset_class: str,
    q: float,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Build one unit-exposure side inside one broad asset class."""
    ranks = e5._rank_panel(source, groups)
    class_eligibility = eligibility & sleeve_panel.eq(asset_class)
    weights, available, validation = _group_equal_from_ranks(
        ranks,
        class_eligibility,
        groups,
        q,
        equal.UNIVERSE,
    )
    if available.le(0).any():
        raise AssertionError(f"{asset_class} has no available group on a decision date")
    return weights, available, validation


def _standalone_weights(
    scores: pd.DataFrame,
    eligibility: pd.DataFrame,
    sleeve_panel: pd.DataFrame,
    groups: pd.DataFrame,
    asset_class: str,
    q: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Construct one disjoint +1/-1 broad-asset-class portfolio."""
    long_raw, long_groups, long_validation = _one_side(
        scores, eligibility, sleeve_panel, groups, asset_class, q
    )
    short_raw, short_groups, short_validation = _one_side(
        -scores, eligibility, sleeve_panel, groups, asset_class, q
    )
    overlap = long_raw.gt(0.0) & short_raw.gt(0.0)
    long_book = long_raw.mask(overlap, 0.0)
    short_book = short_raw.mask(overlap, 0.0)
    long_total = long_book.sum(axis=1)
    short_total = short_book.sum(axis=1)
    if long_total.le(0.0).any() or short_total.le(0.0).any():
        raise AssertionError(f"{asset_class} has an empty side after overlap removal")
    long_book = long_book.div(long_total, axis=0)
    short_book = short_book.div(short_total, axis=0)
    weights = long_book - short_book
    outside = ~sleeve_panel.eq(asset_class)
    diagnostics = {
        "max_pre_scale_weight_sum_abs_error": max(
            float(long_validation["weight_sum_abs_error"].max()),
            float(short_validation["weight_sum_abs_error"].max()),
        ),
        "max_group_budget_abs_error": max(
            float(long_validation["max_group_budget_abs_error"].max()),
            float(short_validation["max_group_budget_abs_error"].max()),
        ),
        "minimum_available_groups": int(min(long_groups.min(), short_groups.min())),
        "mean_available_groups": float(
            pd.concat([long_groups, short_groups], axis=1).mean().mean()
        ),
        "mean_long_assets": float(long_book.gt(0.0).sum(axis=1).mean()),
        "mean_short_assets": float(short_book.gt(0.0).sum(axis=1).mean()),
        "max_long_exposure_abs_error": float(
            long_book.sum(axis=1).sub(1.0).abs().max()
        ),
        "max_short_exposure_abs_error": float(
            short_book.sum(axis=1).sub(1.0).abs().max()
        ),
        "max_net_exposure_abs_error": float(weights.sum(axis=1).abs().max()),
        "max_gross_exposure_abs_error": float(
            weights.abs().sum(axis=1).sub(2.0).abs().max()
        ),
        "max_asset_class_leakage": float(
            weights.where(outside, 0.0).abs().sum(axis=1).max()
        ),
        "max_overlap_assets_removed": int(overlap.sum(axis=1).max()),
    }
    return weights, diagnostics


def _run_portfolio(
    *,
    asset_class: str,
    method: str,
    q: float,
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    diagnostics: Mapping[str, float],
    ew_nav: pd.Series,
    costs: float,
) -> tuple[dict, dict, dict, dict]:
    """Backtest one standalone book and return performance and audit records."""
    ticker_class = asset_class.lower().replace(" ", "_")
    net, gross = equal._backtest(
        prices,
        weights,
        costs,
        f"futures_{ticker_class}_{method}_long_short_q_{q:.2f}",
    )
    net_view = matched._WindowedPortfolio(net)
    gross_view = matched._WindowedPortfolio(gross)
    nav = net_view.get_portfolio_nav().dropna()
    nav_start = pd.Timestamp(nav.index.min())
    nav_end = pd.Timestamp(nav.index.max())
    measurement_years = (nav_end - nav_start).days / 365.25
    performance = {
        "universe": equal.UNIVERSE.value,
        "analysis_window": WINDOW,
        "strategy": "long_short",
        "asset_class": asset_class,
        "method": method,
        "q": q,
        **equal._performance_payload(net_view, gross_view, ew_nav),
        "nav_start": nav_start,
        "nav_end": nav_end,
        "measurement_years": measurement_years,
        "runner": RUNNER,
    }
    ordinary_errors = [
        abs(float(value))
        for key, value in diagnostics.items()
        if key.startswith("max_")
        and key.endswith(("error", "leakage"))
        and "group_budget" not in key
    ]
    group_error = float(diagnostics["max_group_budget_abs_error"])
    passed = (
        max(ordinary_errors) <= equal.EXPOSURE_TOLERANCE
        and group_error <= equal.GROUP_BUDGET_TOLERANCE
    )
    acceptance = {
        "analysis_window": WINDOW,
        "strategy": "long_short",
        "asset_class": asset_class,
        "method": method,
        "q": q,
        **diagnostics,
        "exposure_tolerance": equal.EXPOSURE_TOLERANCE,
        "group_budget_tolerance": equal.GROUP_BUDGET_TOLERANCE,
        "status": "PASS" if passed else "FAIL",
    }
    construction = {
        "analysis_window": WINDOW,
        "asset_class": asset_class,
        "method": method,
        "q": q,
        **diagnostics,
    }
    horizon = {
        "analysis_window": WINDOW,
        "asset_class": asset_class,
        "method": method,
        "q": q,
        "nav_start": nav_start,
        "nav_end": nav_end,
        "measurement_years": measurement_years,
        "nav_rows": len(nav),
        "pre_window_nav_rows": int((nav.index < WINDOW_START).sum()),
        "post_window_nav_rows": int((nav.index > WINDOW_END).sum()),
    }
    return performance, acceptance, construction, horizon


def _comparison(performance: pd.DataFrame) -> pd.DataFrame:
    """Compare every cluster book with its own within-class global rank."""
    global_rows = performance.loc[performance["method"].eq("global")].set_index(
        ["asset_class", "q"]
    )
    rows = []
    for _, cluster in performance.loc[
        performance["method"].str.startswith("cluster_")
    ].iterrows():
        key = (cluster["asset_class"], cluster["q"])
        benchmark = global_rows.loc[key]
        row = cluster.to_dict()
        for metric in equal.COMPARISON_METRICS:
            row[f"global_{metric}"] = benchmark[metric]
            row[f"delta_vs_global_{metric}"] = cluster[metric] - benchmark[metric]
        row["beats_global_net_return"] = (
            row["delta_vs_global_net_return_annualized"] > 0.0
        )
        row["beats_global_sharpe"] = row["delta_vs_global_sharpe_rf0"] > 0.0
        row["beats_global_both"] = (
            row["beats_global_net_return"] and row["beats_global_sharpe"]
        )
        row["mean_variance_dominates_global"] = (
            row["delta_vs_global_net_return_annualized"] >= 0.0
            and row["delta_vs_global_volatility_annualized"] <= 0.0
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["asset_class", "q", "method"])


def _combined_weight_reconstruction(
    standalone: Mapping[tuple[str, float, str], pd.DataFrame],
    scores: pd.DataFrame,
    eligibility: pd.DataFrame,
    sleeve_panel: pd.DataFrame,
    groups_by_method: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Recombine standalone books and match the accepted strategic-book weights."""
    rows = []
    for q in QUANTILES:
        for method, groups in groups_by_method.items():
            combined = sum(
                standalone[(asset_class, q, method)].mul(
                    matched.TARGET[asset_class]
                )
                for asset_class in ASSET_CLASSES
            )
            reference, _ = matched.full._build_constrained_weights(
                "long_short",
                scores,
                eligibility,
                sleeve_panel,
                groups,
                q,
            )
            error = float((combined - reference).abs().to_numpy().max())
            rows.append(
                {
                    "analysis_window": WINDOW,
                    "q": q,
                    "method": method,
                    "max_weight_abs_error": error,
                    "tolerance": RECONSTRUCTION_TOLERANCE,
                    "status": "PASS"
                    if error <= RECONSTRUCTION_TOLERANCE
                    else "FAIL",
                }
            )
    return pd.DataFrame(rows)


def _design(
    dates: pd.DatetimeIndex,
    prices: pd.DataFrame,
    sleeves: pd.Series,
    eligibility: pd.DataFrame,
) -> pd.DataFrame:
    """Return one frozen design row per class with dynamic eligible counts."""
    spec = e5.get_universe_spec(equal.UNIVERSE)
    rows = []
    for asset_class in ASSET_CLASSES:
        class_mask = pd.DataFrame(
            [sleeves.eq(asset_class).to_numpy()] * len(eligibility),
            index=eligibility.index,
            columns=eligibility.columns,
        )
        class_counts = eligibility.where(class_mask, False).sum(axis=1)
        ever_eligible = eligibility.where(class_mask, False).any(axis=0)
        rows.append(
            {
                "universe": equal.UNIVERSE.value,
                "analysis_window": WINDOW,
                "asset_class": asset_class,
                "contracts_ever_eligible": int(ever_eligible.sum()),
                "eligible_contracts_min": int(class_counts.min()),
                "eligible_contracts_median": float(class_counts.median()),
                "eligible_contracts_max": int(class_counts.max()),
                "decision_dates": len(dates),
                "decision_start": dates.min(),
                "decision_end": dates.max(),
                "alignment_price_start": prices.index.min(),
                "performance_calendar_start": WINDOW_START,
                "performance_price_end": prices.index.max(),
                "signal": "48-week log-return sum excluding latest 4 weeks",
                "primary_q": PRIMARY_Q,
                "robustness_q": QUANTILES[1],
                "cost_bps": spec.cost_bps,
                "implementation_lag": 1,
                "methods": "global|cluster_baseline|cluster_M1_star",
                "returns_convention": equal.data_convention(spec),
                "runner": RUNNER,
            }
        )
    return pd.DataFrame(rows)


def run() -> Mapping[str, pd.DataFrame]:
    """Execute all standalone asset-class long-short portfolios once."""
    started = time.perf_counter()
    data = e5.load_universe(equal.UNIVERSE)
    dates = matched._window_dates(
        e5.load_cached(equal.UNIVERSE, e5.SmootherName.BASELINE).dates
    )
    eligibility = e5._investable_eligibility(data, dates)
    columns = eligibility.columns
    scores = e5._raw_momentum_scores(
        data, dates, vol_adjusted=False
    ).reindex(columns=columns).where(eligibility)
    prices = matched._prices_with_context(e5._prices(data).reindex(columns=columns))
    sleeves = equal._broad_sleeves(data.taxonomy, columns)
    sleeve_panel = equal._sleeve_panel(dates, sleeves)
    cluster_groups = {
        config: equal._hierarchical_groups(
            e5._cluster_groups(equal.UNIVERSE, config).reindex(
                index=dates, columns=columns
            ),
            sleeve_panel,
        )
        for config in equal.CONFIGS
    }
    groups_by_method = {
        "global": sleeve_panel,
        "cluster_baseline": cluster_groups[e5.SmootherName.BASELINE],
        "cluster_M1_star": cluster_groups[e5.SmootherName.M1_STAR],
    }
    accepted_navs = pd.read_csv(
        equal._accepted_root() / "navs.csv",
        parse_dates=["date"],
        float_precision="round_trip",
    ).set_index("date")
    ew_nav = matched._bounded_panel(accepted_navs["EW_all"])
    if not isinstance(ew_nav, pd.Series):
        raise AssertionError("bounded EW reference is not a Series")
    costs = e5.get_universe_spec(equal.UNIVERSE).cost_bps / 10000.0

    performance_rows = []
    acceptance_rows = []
    construction_rows = []
    horizon_rows = []
    standalone = {}
    for asset_class in ASSET_CLASSES:
        for q in QUANTILES:
            for method, groups in groups_by_method.items():
                weights, diagnostics = _standalone_weights(
                    scores,
                    eligibility,
                    sleeve_panel,
                    groups,
                    asset_class,
                    q,
                )
                standalone[(asset_class, q, method)] = weights
                records = _run_portfolio(
                    asset_class=asset_class,
                    method=method,
                    q=q,
                    prices=prices,
                    weights=weights,
                    diagnostics=diagnostics,
                    ew_nav=ew_nav,
                    costs=costs,
                )
                performance, acceptance, construction, horizon = records
                performance_rows.append(performance)
                acceptance_rows.append(acceptance)
                construction_rows.append(construction)
                horizon_rows.append(horizon)

    performance = pd.DataFrame(performance_rows).sort_values(
        ["asset_class", "q", "method"]
    ).reset_index(drop=True)
    acceptance = pd.DataFrame(acceptance_rows).sort_values(
        ["asset_class", "q", "method"]
    ).reset_index(drop=True)
    if not acceptance["status"].eq("PASS").all():
        raise AssertionError(acceptance.loc[~acceptance["status"].eq("PASS")])
    reconstruction = _combined_weight_reconstruction(
        standalone, scores, eligibility, sleeve_panel, groups_by_method
    )
    if not reconstruction["status"].eq("PASS").all():
        raise AssertionError(reconstruction.loc[~reconstruction["status"].eq("PASS")])
    outputs = {
        "design": _design(dates, prices, sleeves, eligibility),
        "performance": performance,
        "comparison": _comparison(performance),
        "construction_diagnostics": pd.DataFrame(construction_rows),
        "acceptance": acceptance,
        "horizon_diagnostic": pd.DataFrame(horizon_rows),
        "combined_weight_reconstruction": reconstruction,
    }
    for name, frame in outputs.items():
        e5._write(frame, _root() / f"{name}.csv")
    e5._write(
        pd.DataFrame(
            [
                {
                    "portfolios": len(performance),
                    "runtime_seconds": time.perf_counter() - started,
                }
            ]
        ),
        _root() / "runtime.csv",
    )
    return outputs


def _hash_outputs() -> dict[str, str]:
    """Hash numerical outputs while excluding timing and replay records."""
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(_root().glob("*.csv"))
        if path.name not in {"runtime.csv", "determinism.csv"}
    }


def verify_determinism() -> pd.DataFrame:
    """Replay all standalone books and require identical numerical artifacts."""
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
    """Run and replay the standalone futures asset-class experiment."""
    replay = verify_determinism()
    print(
        "Futures asset-class long-short: PASS "
        f"({len(replay)}/{len(replay)} deterministic)",
        flush=True,
    )


if __name__ == "__main__":
    main()
