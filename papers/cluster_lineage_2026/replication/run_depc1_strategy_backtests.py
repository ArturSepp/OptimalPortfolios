"""Backtest the frozen U1, U2, and U3 strategies with raw and de-PC1 clusters.

Only cluster discovery changes between the two arms. Momentum inputs, global and
classification comparators, point-in-time eligibility, quantiles, sleeve budgets,
holding schedules, implementation lag, and costs remain owner-frozen. Portfolio
accounting is delegated to the existing qis-backed paper engines.
"""
from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from pathlib import Path

import numpy as np
import pandas as pd
from optimalportfolios.alphas.signals.utils import score_within_clusters

import papers.cluster_lineage_2026.replication.run_backtests as e5
import papers.cluster_lineage_2026.replication.run_depc1_cluster_comparison as d4
import papers.cluster_lineage_2026.replication.empirical_specs as empirical_specs
import papers.cluster_lineage_2026.replication.run_futures_best_relative_pnl_scatter as futures_best
import papers.cluster_lineage_2026.replication.run_futures_sleeve_grid as futures_equal
import papers.cluster_lineage_2026.replication.run_futures_weight_30303010 as futures_construction
import papers.cluster_lineage_2026.replication.run_u1_bics_sector_comparison as u1_bics
import papers.cluster_lineage_2026.replication.run_u1_covar_grid_long_short as u1_long_short
import papers.cluster_lineage_2026.replication.run_u1_covar_grid_long_short_prod as u1_prod
import papers.cluster_lineage_2026.replication.run_u2_blackrock_aum_filter as u2_aum
import papers.cluster_lineage_2026.replication.run_u2_blackrock_aum_sensitivity as u2_sensitivity
import papers.cluster_lineage_2026.replication.run_u2_blackrock_etf_grid as u2_funds
import papers.cluster_lineage_2026.replication.run_u2_blackrock_long_short_search as u2_search
import papers.cluster_lineage_2026.replication.run_u2_blackrock_sleeve_grid as u2_sleeves
from papers.cluster_lineage_2026.replication import (
    run_futures_prod_signal_grid_30303010_10bp as futures_grid,
)
from papers.cluster_lineage_2026.replication import (
    run_futures_weight_30303010_u1_window as futures_window,
)


RUNNER = "papers/cluster_lineage_2026/replication/run_depc1_strategy_backtests.py"
UNIVERSES = d4.UNIVERSES
TOLERANCE = 1e-12
ACCOUNTING_TOLERANCE = 1e-10
PERFORMANCE_METRICS = (
    "gross_return_annualized",
    "net_return_annualized",
    "volatility_annualized",
    "sharpe_rf0",
    "one_way_turnover_annualized",
    "cost_drag_bp_per_year",
    "net_total_return",
    "alpha_vs_ew_annualized",
    "beta_vs_ew",
)


def _finite_max(values) -> float:
    """Return the maximum finite absolute value, or zero for an empty input."""
    array = np.asarray(values, dtype=float)
    finite = np.abs(array[np.isfinite(array)])
    return float(finite.max()) if finite.size else 0.0


def _partition_panels(universe: str) -> tuple[d4.UniverseInputs, pd.DataFrame, pd.DataFrame]:
    """Load complete fingerprinted raw and de-PC1 partition caches."""
    inputs = d4.load_inputs(universe)
    loaded = {
        transform: d4._load_cached_panel(inputs, transform)
        for transform in d4.TRANSFORMS
    }
    if any(value is None for value in loaded.values()):
        rebuilt = d4.run_universe(universe)
        return inputs, rebuilt["raw_panel"], rebuilt["depc1_panel"]
    return inputs, loaded["raw"][0], loaded["depc1"][0]


def _exposure_error(weights: pd.DataFrame) -> float:
    """Return the largest error from the frozen +1/-1 exposure identities."""
    long_side = weights.clip(lower=0.0).sum(axis=1)
    short_side = -weights.clip(upper=0.0).sum(axis=1)
    return max(
        float(long_side.sub(1.0).abs().max()),
        float(short_side.sub(1.0).abs().max()),
        float(weights.sum(axis=1).abs().max()),
        float(weights.abs().sum(axis=1).sub(2.0).abs().max()),
    )


def _outside_eligibility_error(
    weights: Mapping[str, pd.DataFrame], eligibility: pd.DataFrame
) -> float:
    """Return the largest absolute weight assigned outside eligibility."""
    errors = [
        _finite_max(frame.where(~eligibility, 0.0).to_numpy())
        for frame in weights.values()
    ]
    return max(errors, default=0.0)


def _portfolio_difference(
    left,
    right,
    bound: Callable[[pd.DataFrame | pd.Series], pd.DataFrame | pd.Series] | None = None,
) -> float:
    """Compare NAV, turnover, and realised costs from two comparator replays."""
    panels = (
        (left.get_portfolio_nav(), right.get_portfolio_nav()),
        (left.get_turnover(), right.get_turnover()),
        (left.realized_costs, right.realized_costs),
    )
    errors = []
    for first, second in panels:
        if bound is not None:
            first = bound(first)
            second = bound(second)
        first, second = first.align(second, join="outer")
        errors.append(_finite_max(first.subtract(second).to_numpy()))
    return max(errors)


def _instrument_attribution(
    portfolio,
    leg: str,
    universe: str,
    bound: Callable[[pd.DataFrame | pd.Series], pd.DataFrame | pd.Series] | None = None,
) -> tuple[pd.DataFrame, Mapping[str, object]]:
    """Return exact currency P&L under QIS's missing-price valuation convention."""
    nav = portfolio.get_portfolio_nav()
    if bound is not None:
        nav = bound(nav)
    nav = nav.dropna()
    start = pd.Timestamp(nav.index.min())
    end = pd.Timestamp(nav.index.max())
    # PortfolioData.get_instruments_pnl is rejected here because it stores return-on-NAV
    # contributions rather than the currency identity, while get_costs defaults to a
    # 260-observation rolling sum. QIS's backtester values a missing price as no instrument
    # value through np.nansum; zero-filling prices reproduces that exact drop/reappearance.
    selected = (portfolio.prices.index > start) & (portfolio.prices.index <= end)
    valued_prices = portfolio.prices.fillna(0.0)
    prior_units = portfolio.units.shift(1)
    gross_currency = prior_units.multiply(valued_prices.diff()).loc[selected].fillna(0.0)
    costs = portfolio.realized_costs.reindex_like(gross_currency).fillna(0.0)
    net_currency = gross_currency.subtract(costs)
    nav_changes = nav.diff().dropna().reindex(net_currency.index)
    if nav_changes.isna().any():
        raise AssertionError("instrument attribution does not align with portfolio NAV")
    step_error = net_currency.sum(axis=1).subtract(nav_changes)
    cumulative_error = abs(
        float(net_currency.to_numpy().sum()) - float(nav.iloc[-1] - nav.iloc[0])
    )
    diagnostics = {
        "nav_start": start,
        "nav_end": end,
        "beginning_nav": float(nav.iloc[0]),
        "ending_nav": float(nav.iloc[-1]),
        "portfolio_net_total_return": float(nav.iloc[-1] / nav.iloc[0] - 1.0),
        "attributed_net_total_return": float(
            net_currency.to_numpy().sum() / nav.iloc[0]
        ),
        "max_step_reconciliation_abs_error": float(step_error.abs().max()),
        "cumulative_reconciliation_abs_error": cumulative_error,
    }
    long_gross = gross_currency.where(prior_units.gt(0.0), 0.0).sum(axis=0)
    short_gross = gross_currency.where(prior_units.lt(0.0), 0.0).sum(axis=0)
    cost_total = costs.sum(axis=0)
    net_total = net_currency.sum(axis=0)
    beginning_nav = float(diagnostics["beginning_nav"])
    frame = pd.DataFrame(
        {
            "universe": universe,
            "leg": leg,
            "asset": net_currency.columns,
            "long_gross_pnl_currency": long_gross.reindex(net_currency.columns).to_numpy(),
            "short_gross_pnl_currency": short_gross.reindex(net_currency.columns).to_numpy(),
            "transaction_cost_currency": cost_total.reindex(net_currency.columns).to_numpy(),
            "net_pnl_currency": net_total.reindex(net_currency.columns).to_numpy(),
            "net_pnl_pct_of_start": 100.0
            * net_total.reindex(net_currency.columns).to_numpy()
            / beginning_nav,
        }
    )
    return frame, diagnostics


def _nav_table(
    portfolios: Mapping[str, object],
    bound: Callable[[pd.DataFrame | pd.Series], pd.DataFrame | pd.Series] | None = None,
) -> pd.DataFrame:
    """Return one aligned cumulative net-NAV table."""
    series = {}
    for leg, portfolio in portfolios.items():
        nav = portfolio.get_portfolio_nav()
        series[leg] = bound(nav) if bound is not None else nav
    frame = pd.concat(series, axis=1)
    frame.index.name = "date"
    return frame.reset_index()


def _comparison(
    performance: pd.DataFrame,
    contrasts: tuple[tuple[str, str], ...],
) -> pd.DataFrame:
    """Build only the roadmap-authorised ranking-leg contrasts."""
    rows = []
    for window, window_frame in performance.groupby("analysis_window", sort=False):
        indexed = window_frame.set_index("leg")
        for treatment, benchmark in contrasts:
            if treatment not in indexed.index or benchmark not in indexed.index:
                continue
            row = {
                "analysis_window": window,
                "treatment_leg": treatment,
                "benchmark_leg": benchmark,
            }
            for metric in PERFORMANCE_METRICS:
                row[f"treatment_{metric}"] = indexed.at[treatment, metric]
                row[f"benchmark_{metric}"] = indexed.at[benchmark, metric]
                row[f"delta_{metric}"] = (
                    indexed.at[treatment, metric] - indexed.at[benchmark, metric]
                )
            rows.append(row)
    return pd.DataFrame(rows)


def _acceptance_row(
    check: str,
    measured: float,
    tolerance: float,
    *,
    exact: bool = False,
) -> dict[str, object]:
    """Return one measured-versus-tolerance strategy acceptance row."""
    passed = measured == tolerance if exact else measured <= tolerance
    return {
        "stage": "D5",
        "check": check,
        "measured": measured,
        "tolerance": tolerance,
        "status": "PASS" if passed else "FAIL",
    }


def _turnover_and_costs(
    performance: pd.DataFrame, instrument_pnl: pd.DataFrame
) -> pd.DataFrame:
    """Return the frozen turnover/cost metrics and exact realised costs."""
    columns = [
        "universe",
        "analysis_window",
        "leg",
        "cost_bps_one_way",
        "one_way_turnover_annualized",
        "cost_drag_bp_per_year",
    ]
    output = performance[columns].copy()
    primary = performance["is_primary_window"]
    primary_window = performance.loc[primary, "analysis_window"].iloc[0]
    realised = instrument_pnl.groupby("leg")["transaction_cost_currency"].sum()
    mask = output["analysis_window"].eq(primary_window)
    output.loc[mask, "realised_cost_currency"] = output.loc[mask, "leg"].map(realised)
    return output


def _run_u2() -> Mapping[str, pd.DataFrame]:
    """Run BlackRock AUM100 raw/de-PC1 hybrid and pure-cluster books."""
    started = time.perf_counter()
    inputs, raw_panel, depc1_panel = _partition_panels("blackrock_funds")
    daily = u2_funds._read_daily()
    dates = inputs.dates[
        (inputs.dates >= u2_funds.HEADLINE_START)
        & (inputs.dates <= u2_funds.HEADLINE_END)
    ]
    rolling_aum = u2_aum._rolling_aum()
    eligibility_all = u2_sensitivity._eligibilities(
        daily, inputs.dates, rolling_aum
    )["aum_100m"]
    eligibility = eligibility_all.reindex(index=dates).astype(bool)
    monthly_dates = u2_funds._native_returns(daily, "ME").index
    monthly_eligibility = u2_sensitivity._eligibilities(
        daily, monthly_dates, rolling_aum
    )["aum_100m"]
    sleeve_map = u2_sleeves._broad_sleeves(daily.columns)
    sleeve_panel = u2_sleeves._sleeve_panel(dates, sleeve_map)

    score_sets = {}
    signal_diagnostics = {}
    weights_by_arm = {}
    weight_diagnostics = {}
    for arm, clusters in (("raw", raw_panel), ("depc1", depc1_panel)):
        clusters = clusters.reindex(index=dates, columns=eligibility.columns)
        global_scores, cluster_scores, diagnostics = u2_sensitivity._signal_panels(
            daily,
            dates,
            eligibility,
            monthly_eligibility,
            clusters,
        )
        score_sets[arm] = (global_scores, cluster_scores)
        signal_diagnostics[arm] = diagnostics
        weights, exact = u2_sensitivity._weights(
            global_scores, cluster_scores, eligibility, clusters, sleeve_panel
        )
        weights_by_arm[arm] = weights
        weight_diagnostics[arm] = exact

    global_score_error = _finite_max(
        score_sets["raw"][0].subtract(score_sets["depc1"][0]).to_numpy()
    )
    global_weight_error = _finite_max(
        weights_by_arm["raw"]["global"]
        .subtract(weights_by_arm["depc1"]["global"])
        .to_numpy()
    )
    leg_weights = {
        "cluster_raw": weights_by_arm["raw"]["hybrid"],
        "cluster_depc1": weights_by_arm["depc1"]["hybrid"],
        "pure_cluster_raw": weights_by_arm["raw"]["cluster"],
        "pure_cluster_depc1": weights_by_arm["depc1"]["cluster"],
        "global": weights_by_arm["raw"]["global"],
    }
    prices_all = u2_funds._performance_prices(daily)
    performance_rows = []
    primary_portfolios = {}
    global_portfolio_error = 0.0
    for window_name, (window_start, window_end) in u2_search.WINDOWS.items():
        window_dates = dates[(dates >= window_start) & (dates <= window_end)]
        window = u2_sensitivity._window(
            prices_all, eligibility_all, window_name, window_dates
        )
        scheduled_dates = u2_search._rebalance_dates(
            window_dates, u2_sensitivity.SCHEDULE
        )
        for leg, weights in leg_weights.items():
            net, gross = u2_funds._backtest(
                window["prices"],
                weights.reindex(index=scheduled_dates),
                u2_sensitivity.COST_BPS / 10000.0,
                f"depc1_u2_{window_name}_{leg}",
            )
            payload = u2_sleeves._performance_payload(net, gross, window["ew_nav"])
            performance_rows.append(
                {
                    "universe": "blackrock_funds",
                    "analysis_window": window_name,
                    "is_primary_window": window_name == u2_search.FULL_WINDOW,
                    "leg": leg,
                    "q": u2_sensitivity.Q,
                    "cost_bps_one_way": u2_sensitivity.COST_BPS,
                    **payload,
                }
            )
            if window_name == u2_search.FULL_WINDOW:
                primary_portfolios[leg] = net
        duplicate_global, _ = u2_funds._backtest(
            window["prices"],
            weights_by_arm["depc1"]["global"].reindex(index=scheduled_dates),
            u2_sensitivity.COST_BPS / 10000.0,
            f"depc1_u2_{window_name}_global_duplicate",
        )
        global_portfolio_error = max(
            global_portfolio_error,
            _portfolio_difference(
                primary_portfolios["global"]
                if window_name == u2_search.FULL_WINDOW
                else u2_funds._backtest(
                    window["prices"],
                    weights_by_arm["raw"]["global"].reindex(index=scheduled_dates),
                    u2_sensitivity.COST_BPS / 10000.0,
                    f"depc1_u2_{window_name}_global_reference",
                )[0],
                duplicate_global,
            ),
        )

    performance = pd.DataFrame(performance_rows)
    attribution_frames = []
    attribution_errors = []
    for leg, portfolio in primary_portfolios.items():
        frame, diagnostics = _instrument_attribution(
            portfolio, leg, "blackrock_funds"
        )
        attribution_frames.append(frame)
        attribution_errors.extend(
            [
                float(diagnostics["max_step_reconciliation_abs_error"]),
                float(diagnostics["cumulative_reconciliation_abs_error"]),
            ]
        )
    instrument_pnl = pd.concat(attribution_frames, ignore_index=True)
    aum_at_dates = u2_aum._aum_for_dates(dates, rolling_aum).reindex(
        columns=eligibility.columns
    )
    aum_violations = int((eligibility & ~aum_at_dates.gt(100.0)).sum().sum())
    maximum_weight_error = max(
        _exposure_error(weights) for weights in leg_weights.values()
    )
    helper_weight_error = max(
        abs(float(value))
        for diagnostics in weight_diagnostics.values()
        for key, value in diagnostics.items()
        if "error" in key
    )
    maximum_lookahead = max(
        float(diagnostics[key])
        for diagnostics in signal_diagnostics.values()
        for key in ("max_global_lookahead_days", "max_cluster_lookahead_days")
    )
    acceptance = pd.DataFrame(
        [
            _acceptance_row("maximum signal lookahead days", maximum_lookahead, 0.0),
            _acceptance_row(
                "maximum weight and sleeve exposure error",
                max(maximum_weight_error, helper_weight_error),
                TOLERANCE,
            ),
            _acceptance_row(
                "weight outside point-in-time eligibility",
                _outside_eligibility_error(leg_weights, eligibility),
                TOLERANCE,
            ),
            _acceptance_row("AUM <= USD100m eligible observations", aum_violations, 0, exact=True),
            _acceptance_row("global score arm difference", global_score_error, TOLERANCE),
            _acceptance_row("global weight arm difference", global_weight_error, TOLERANCE),
            _acceptance_row("global portfolio arm difference", global_portfolio_error, TOLERANCE),
            _acceptance_row(
                "instrument P&L reconciliation error",
                max(attribution_errors),
                ACCOUNTING_TOLERANCE,
            ),
            _acceptance_row(
                "one-way transaction cost bps",
                u2_sensitivity.COST_BPS,
                20.0,
                exact=True,
            ),
        ]
    )
    return {
        "performance": performance,
        "performance_comparison": _comparison(
            performance,
            (
                ("cluster_depc1", "cluster_raw"),
                ("cluster_depc1", "global"),
                ("cluster_raw", "global"),
            ),
        ),
        "instrument_pnl": instrument_pnl,
        "navs": _nav_table(primary_portfolios),
        "turnover_and_costs": _turnover_and_costs(performance, instrument_pnl),
        "strategy_acceptance": acceptance,
        "design": pd.DataFrame(
            [
                {
                    "universe": "blackrock_funds",
                    "cluster_cell": "W-THU span 156",
                    "signal": "ROSAA production risk-adjusted momentum",
                    "q": 0.25,
                    "primary_leg": "global long / group-equal cluster short",
                    "sleeve_budgets_per_side": "Equity 50%|Fixed Income 30%|Rest 20%",
                    "schedule": "every_two_months",
                    "aum_rule": "12 completed month-end average AUM > USD 100m",
                    "cost_bps_one_way": 20.0,
                    "global_role": "ranking-performance comparator",
                    "ew_role": "market reference for beta/alpha only",
                }
            ]
        ),
        "strategy_runtime": pd.DataFrame(
            [{"stage": "D5", "runtime_seconds": time.perf_counter() - started}]
        ),
    }


def _cluster_scores(
    raw_source: pd.DataFrame,
    groups: pd.DataFrame,
    dates: pd.DatetimeIndex,
    eligibility: pd.DataFrame,
    raw_decision: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build point-in-time within-cluster scores at the frozen fallback."""
    source = score_within_clusters(
        raw_signal=raw_source,
        rolling_clusters=u1_prod._panel_dict(groups),
        min_cluster_size=5,
    )
    scores, timestamps = u1_prod._asof_panel(source, dates)
    scores = scores.reindex(index=dates, columns=eligibility.columns).where(eligibility)
    if raw_decision is not None:
        scores = scores.where(raw_decision.notna())
    return scores, timestamps


def _run_u3() -> Mapping[str, pd.DataFrame]:
    """Run futures raw/de-PC1 clusters against the frozen sleeve-global rank."""
    started = time.perf_counter()
    _, raw_panel, depc1_panel = _partition_panels("futures")
    context = futures_grid._build_context()
    dates = context["dates"]
    eligibility = context["eligibility"].astype(bool)
    sleeve_panel = context["sleeve_panel"]
    global_scores, raw_source, timestamps, signal_diagnostic = (
        futures_grid._signal_for_spec(futures_best.SPEC, context)
    )
    global_groups = sleeve_panel
    global_weights_by_arm = {}
    global_diagnostics = {}
    for arm in ("raw", "depc1"):
        global_weights_by_arm[arm], global_diagnostics[arm] = (
            futures_construction._build_constrained_weights(
                "long_short",
                global_scores,
                eligibility,
                sleeve_panel,
                global_groups,
                futures_best.Q,
            )
        )
    cluster_weights = {}
    cluster_diagnostics = {}
    cluster_timestamps = {}
    raw_groups = None
    for arm, panel in (("raw", raw_panel), ("depc1", depc1_panel)):
        panel = panel.reindex(index=dates, columns=eligibility.columns)
        groups = futures_equal._hierarchical_groups(panel, sleeve_panel)
        if arm == "raw":
            raw_groups = groups
        scores, cluster_timestamps[arm] = _cluster_scores(
            raw_source, groups, dates, eligibility
        )
        cluster_weights[arm], cluster_diagnostics[arm] = (
            futures_construction._build_constrained_weights(
                "long_short",
                scores,
                eligibility,
                sleeve_panel,
                groups,
                futures_best.Q,
            )
        )
    if raw_groups is None:
        raise AssertionError("raw futures groups were not constructed")
    accepted_groups = context["groups_by_method"][
        futures_best.CLUSTER_METHOD
    ].where(eligibility)
    frozen_partition_match = float(
        np.mean(
            [
                d4._same_partition(raw_groups.loc[date], accepted_groups.loc[date])
                for date in dates
            ]
        )
    )
    leg_weights = {
        "cluster_raw": cluster_weights["raw"],
        "cluster_depc1": cluster_weights["depc1"],
        "global": global_weights_by_arm["raw"],
    }
    net_portfolios = {}
    gross_portfolios = {}
    for leg, weights in leg_weights.items():
        net, gross = futures_equal._backtest(
            context["performance_prices"],
            weights,
            futures_best.COST_BPS / 10000.0,
            f"depc1_futures_{leg}",
        )
        net_portfolios[leg] = net
        gross_portfolios[leg] = gross
    duplicate_global, _ = futures_equal._backtest(
        context["performance_prices"],
        global_weights_by_arm["depc1"],
        futures_best.COST_BPS / 10000.0,
        "depc1_futures_global_duplicate",
    )
    performance_rows = []
    for leg in leg_weights:
        payload = futures_equal._performance_payload(
            futures_window._WindowedPortfolio(net_portfolios[leg]),
            futures_window._WindowedPortfolio(gross_portfolios[leg]),
            context["ew_nav"],
        )
        performance_rows.append(
            {
                "universe": "futures",
                "analysis_window": futures_window.WINDOW,
                "is_primary_window": True,
                "leg": leg,
                "q": futures_best.Q,
                "cost_bps_one_way": futures_best.COST_BPS,
                **payload,
            }
        )
    performance = pd.DataFrame(performance_rows)
    attribution_frames = []
    attribution_errors = []
    for leg, portfolio in net_portfolios.items():
        frame, diagnostics = _instrument_attribution(
            portfolio, leg, "futures", futures_window._bounded_panel
        )
        attribution_frames.append(frame)
        attribution_errors.extend(
            [
                float(diagnostics["max_step_reconciliation_abs_error"]),
                float(diagnostics["cumulative_reconciliation_abs_error"]),
            ]
        )
    instrument_pnl = pd.concat(attribution_frames, ignore_index=True)
    timestamp_lookahead = max(
        float(series.sub(series.index).dt.days.max())
        for series in cluster_timestamps.values()
    )
    excluded = eligibility.columns.intersection(e5.FUTURES_INVESTABILITY_EXCLUSIONS)
    excluded_eligibility = int(eligibility.reindex(columns=excluded).sum().sum())
    excluded_weight = max(
        _finite_max(frame.reindex(columns=excluded).fillna(0.0).to_numpy())
        for frame in leg_weights.values()
    )
    diagnostic_error = max(
        abs(float(value))
        for diagnostics in (*global_diagnostics.values(), *cluster_diagnostics.values())
        for key, value in diagnostics.items()
        if key.startswith("max_") and key.endswith("error")
    )
    global_error = max(
        _finite_max(
            global_weights_by_arm["raw"]
            .subtract(global_weights_by_arm["depc1"])
            .to_numpy()
        ),
        _portfolio_difference(
            net_portfolios["global"], duplicate_global, futures_window._bounded_panel
        ),
    )
    acceptance = pd.DataFrame(
        [
            _acceptance_row(
                "maximum signal lookahead days",
                max(
                    float(signal_diagnostic["max_signal_lookahead_days"]),
                    timestamp_lookahead,
                ),
                0.0,
            ),
            _acceptance_row(
                "maximum weight and sleeve exposure error",
                max(diagnostic_error, *(_exposure_error(w) for w in leg_weights.values())),
                TOLERANCE,
            ),
            _acceptance_row(
                "weight outside point-in-time eligibility",
                _outside_eligibility_error(leg_weights, eligibility),
                TOLERANCE,
            ),
            _acceptance_row(
                "raw partition match to owner-frozen M1-star",
                frozen_partition_match,
                1.0,
                exact=True,
            ),
            _acceptance_row(
                "owner-excluded eligible observations",
                excluded_eligibility,
                0,
                exact=True,
            ),
            _acceptance_row("maximum owner-excluded weight", excluded_weight, TOLERANCE),
            _acceptance_row(
                "owner exclusion set size",
                len(e5.FUTURES_INVESTABILITY_EXCLUSIONS),
                7,
                exact=True,
            ),
            _acceptance_row("global comparator arm difference", global_error, TOLERANCE),
            _acceptance_row(
                "instrument P&L reconciliation error",
                max(attribution_errors),
                ACCOUNTING_TOLERANCE,
            ),
            _acceptance_row(
                "one-way transaction cost bps",
                futures_best.COST_BPS,
                10.0,
                exact=True,
            ),
        ]
    )
    return {
        "performance": performance,
        "performance_comparison": _comparison(
            performance,
            (
                ("cluster_depc1", "cluster_raw"),
                ("cluster_depc1", "global"),
                ("cluster_raw", "global"),
            ),
        ),
        "instrument_pnl": instrument_pnl,
        "navs": _nav_table(net_portfolios, futures_window._bounded_panel),
        "turnover_and_costs": _turnover_and_costs(performance, instrument_pnl),
        "strategy_acceptance": acceptance,
        "design": pd.DataFrame(
            [
                {
                    "universe": "futures",
                    "cluster_cell": "W-WED span 156 M1-star delta 0.0691",
                    "signal": futures_best.SPEC.signal_id,
                    "q": futures_best.Q,
                    "sleeve_budgets_per_side": "Equity 30%|Fixed Income 30%|Commodities 30%|FX 10%",
                    "cost_bps_one_way": futures_best.COST_BPS,
                    "owner_exclusions": "|".join(sorted(e5.FUTURES_INVESTABILITY_EXCLUSIONS)),
                    "global_role": "ranking-performance comparator",
                    "ew_role": "market reference for beta/alpha only",
                }
            ]
        ),
        "strategy_runtime": pd.DataFrame(
            [{"stage": "D5", "runtime_seconds": time.perf_counter() - started}]
        ),
    }


def _u1_signal_inputs(
    data,
    dates: pd.DatetimeIndex,
    columns: pd.Index,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Build the exact frozen U1 production signal with no mean adjustment."""
    frozen = empirical_specs.U1_OPTIMAL_SPEC
    if (
        frozen.signal_frequency != "ME"
        or frozen.momentum_long_span != 12
        or frozen.momentum_short_span is not None
        or frozen.momentum_vol_span != 13
        or frozen.momentum_mean_adj_type != "NONE"
    ):
        raise AssertionError("U1_OPTIMAL_SPEC no longer matches the de-PC1 roadmap")
    daily = u1_bics._read_daily(columns)
    signal_prices, benchmark, _, _ = u1_prod._period_inputs(
        data, daily, frozen.signal_frequency
    )
    spec = u1_prod._signal_spec(u1_prod.EXACT_VARIANT, frozen.covariance_frequency)
    global_source, raw_source = u1_prod._base_signals(signal_prices, benchmark, spec)
    global_decision, timestamps = u1_prod._asof_panel(global_source, dates)
    raw_decision, raw_timestamps = u1_prod._asof_panel(raw_source, dates)
    if not timestamps.equals(raw_timestamps):
        raise AssertionError("U1 raw and global signal timestamps differ")
    preflight = pd.DataFrame(
        [
            {
                "max_signal_lookahead_days": int(
                    timestamps.sub(timestamps.index).dt.days.max()
                ),
                "signal_frequency": spec.frequency,
                "momentum_long_span": spec.long_span,
                "momentum_vol_span": spec.vol_span,
                "momentum_mean_adj_type": frozen.momentum_mean_adj_type,
            }
        ]
    )
    return global_decision, raw_decision, raw_source, preflight


def _run_u1() -> Mapping[str, pd.DataFrame]:
    """Run U1 ME/span36 raw/de-PC1 clusters against global and BICS ranks."""
    started = time.perf_counter()
    inputs, raw_panel, depc1_panel = _partition_panels("msci_us")
    dates = inputs.dates
    eligibility = inputs.eligibility.astype(bool)
    columns = eligibility.columns
    data = e5.load_universe(e5.UniverseName.MSCI_US)
    bics = inputs.taxonomy["bbg_bics_sector"].replace("", np.nan)
    primary_eligibility = eligibility & bics.notna().to_numpy()
    sector_groups = pd.DataFrame(
        np.tile(bics.to_numpy(), (len(dates), 1)),
        index=dates,
        columns=columns,
    )
    global_scores, raw_decision, raw_source, preflight = _u1_signal_inputs(
        data, dates, columns
    )
    global_scores = global_scores.reindex(index=dates, columns=columns)
    raw_decision = raw_decision.reindex(index=dates, columns=columns)
    cluster_scores = {}
    cluster_timestamps = {}
    panels = {"raw": raw_panel, "depc1": depc1_panel}
    for arm, panel in panels.items():
        groups = panel.reindex(index=dates, columns=columns)
        cluster_scores[arm], cluster_timestamps[arm] = _cluster_scores(
            raw_source,
            groups,
            dates,
            primary_eligibility,
            raw_decision,
        )
    global_groups = pd.DataFrame("global", index=dates, columns=columns)
    weights = {}
    weights["cluster_raw"] = u1_bics._long_short_weights(
        cluster_scores["raw"], primary_eligibility, panels["raw"].reindex(index=dates)
    )[0]
    weights["cluster_depc1"] = u1_bics._long_short_weights(
        cluster_scores["depc1"],
        primary_eligibility,
        panels["depc1"].reindex(index=dates),
    )[0]
    weights["sector"] = u1_bics._long_short_weights(
        raw_decision.where(primary_eligibility), primary_eligibility, sector_groups
    )[0]
    global_raw = u1_bics._long_short_weights(
        global_scores.where(primary_eligibility),
        primary_eligibility,
        global_groups,
    )[0]
    global_depc1 = u1_bics._long_short_weights(
        global_scores.where(primary_eligibility),
        primary_eligibility,
        global_groups,
    )[0]
    weights["global"] = global_raw
    prices = e5._prices(data).reindex(columns=columns)
    ew_nav = u1_bics._ew_navs()[u1_bics.WINDOW]
    net_portfolios = {}
    gross_portfolios = {}
    for leg, leg_weights in weights.items():
        net, gross = u1_bics._backtest(
            prices,
            leg_weights,
            u1_bics.COST_BPS / 10000.0,
            f"depc1_u1_{leg}",
        )
        net_portfolios[leg] = net
        gross_portfolios[leg] = gross
    duplicate_global, _ = u1_bics._backtest(
        prices,
        global_depc1,
        u1_bics.COST_BPS / 10000.0,
        "depc1_u1_global_duplicate",
    )
    performance_rows = []
    for leg in weights:
        performance_rows.append(
            {
                "universe": "msci_us",
                "analysis_window": u1_bics.WINDOW,
                "is_primary_window": True,
                "leg": leg,
                "q": u1_bics.Q,
                "cost_bps_one_way": u1_bics.COST_BPS,
                **u1_long_short._performance_payload(
                    net_portfolios[leg], gross_portfolios[leg], ew_nav
                ),
            }
        )
    performance = pd.DataFrame(performance_rows)
    attribution_frames = []
    attribution_errors = []
    for leg, portfolio in net_portfolios.items():
        frame, diagnostics = _instrument_attribution(portfolio, leg, "msci_us")
        attribution_frames.append(frame)
        attribution_errors.extend(
            [
                float(diagnostics["max_step_reconciliation_abs_error"]),
                float(diagnostics["cumulative_reconciliation_abs_error"]),
            ]
        )
    instrument_pnl = pd.concat(attribution_frames, ignore_index=True)
    membership_missing = max(
        int((primary_eligibility & panel.reindex(index=dates).isna()).sum().sum())
        for panel in panels.values()
    )
    maximum_lookahead = max(
        float(preflight.at[0, "max_signal_lookahead_days"]),
        *(
            float(series.sub(series.index).dt.days.max())
            for series in cluster_timestamps.values()
        ),
    )
    global_error = max(
        _finite_max(global_raw.subtract(global_depc1).to_numpy()),
        _portfolio_difference(net_portfolios["global"], duplicate_global),
    )
    group_errors = []
    group_map = {
        "cluster_raw": panels["raw"].reindex(index=dates),
        "cluster_depc1": panels["depc1"].reindex(index=dates),
        "sector": sector_groups,
        "global": global_groups,
    }
    for leg, frame in weights.items():
        diagnostics = u1_bics._group_budget_diagnostics(leg, frame, group_map[leg])
        group_errors.extend(
            diagnostics[
                [
                    "side_weight_sum_abs_error",
                    "max_group_budget_abs_error",
                    "max_within_group_weight_range",
                ]
            ].to_numpy().ravel()
        )
    acceptance = pd.DataFrame(
        [
            _acceptance_row("maximum signal lookahead days", maximum_lookahead, 0.0),
            _acceptance_row(
                "maximum weight and group exposure error",
                max(
                    max(_exposure_error(frame) for frame in weights.values()),
                    _finite_max(group_errors),
                ),
                TOLERANCE,
            ),
            _acceptance_row(
                "weight outside matched BICS eligibility",
                _outside_eligibility_error(weights, primary_eligibility),
                TOLERANCE,
            ),
            _acceptance_row(
                "eligible cluster memberships missing",
                membership_missing,
                0,
                exact=True,
            ),
            _acceptance_row("global comparator arm difference", global_error, TOLERANCE),
            _acceptance_row(
                "instrument P&L reconciliation error",
                max(attribution_errors),
                ACCOUNTING_TOLERANCE,
            ),
            _acceptance_row(
                "one-way transaction cost bps", u1_bics.COST_BPS, 10.0, exact=True
            ),
        ]
    )
    return {
        "performance": performance,
        "performance_comparison": _comparison(
            performance,
            (
                ("cluster_depc1", "cluster_raw"),
                ("cluster_depc1", "global"),
                ("cluster_raw", "global"),
                ("cluster_depc1", "sector"),
                ("cluster_raw", "sector"),
            ),
        ),
        "instrument_pnl": instrument_pnl,
        "navs": _nav_table(net_portfolios),
        "turnover_and_costs": _turnover_and_costs(performance, instrument_pnl),
        "strategy_acceptance": acceptance,
        "design": pd.DataFrame(
            [
                {
                    "universe": "msci_us",
                    "analysis_window": u1_bics.WINDOW,
                    "cluster_cell": "ME span 36",
                    "signal": "U1_OPTIMAL_SPEC ROSAA production ME 12/none/13 NONE",
                    "q": u1_bics.Q,
                    "construction": "group_equal",
                    "sector_comparator": "Bloomberg BICS equal-sector rank",
                    "cost_bps_one_way": u1_bics.COST_BPS,
                    "global_role": "ranking-performance comparator",
                    "ew_role": "market reference for beta/alpha only",
                    "warmup_robustness_status": (
                        "not emitted: pre-headline pairwise correlation is "
                        "materially indefinite"
                    ),
                }
            ]
        ),
        "strategy_runtime": pd.DataFrame(
            [{"stage": "D5", "runtime_seconds": time.perf_counter() - started}]
        ),
    }


def _source_manifest(universe: str) -> pd.DataFrame:
    """Append the strategy runner hash to the D4 source manifest."""
    path = d4._universe_root(universe) / "source_manifest.csv"
    manifest = pd.read_csv(path, float_precision="round_trip")
    manifest = manifest.loc[~manifest["kind"].eq("strategy_runner")]
    runner_path = Path(__file__)
    addition = pd.DataFrame(
        [
            {
                "kind": "strategy_runner",
                "path": RUNNER,
                "sha256": d4._sha256(runner_path),
            }
        ]
    )
    return pd.concat([manifest, addition], ignore_index=True)


def _emit(universe: str, result: Mapping[str, pd.DataFrame]) -> Mapping[str, pd.DataFrame]:
    """Write D5 artifacts while preserving the D4 evidence in combined tables."""
    root = d4._universe_root(universe)
    d4_acceptance = pd.read_csv(root / "acceptance.csv", float_precision="round_trip")
    if "stage" in d4_acceptance:
        d4_acceptance = d4_acceptance.loc[d4_acceptance["stage"].eq("D4")]
    else:
        d4_acceptance.insert(0, "stage", "D4")
    acceptance = pd.concat(
        [d4_acceptance, result["strategy_acceptance"]], ignore_index=True
    )
    if not acceptance["status"].eq("PASS").all():
        raise AssertionError(acceptance.loc[~acceptance["status"].eq("PASS")])
    d4_runtime = pd.read_csv(root / "runtime.csv", float_precision="round_trip")
    if "stage" in d4_runtime:
        d4_runtime = d4_runtime.loc[d4_runtime["stage"].eq("D4")]
    else:
        d4_runtime.insert(0, "stage", "D4")
    runtime = pd.concat([d4_runtime, result["strategy_runtime"]], ignore_index=True)
    output = {
        "performance": result["performance"],
        "performance_comparison": result["performance_comparison"],
        "turnover_and_costs": result["turnover_and_costs"],
        "instrument_pnl": result["instrument_pnl"],
        "navs": result["navs"],
        "strategy_design": result["design"],
        "acceptance": acceptance,
        "runtime": runtime,
        "source_manifest": _source_manifest(universe),
    }
    for name, frame in output.items():
        d4._write(frame, root / f"{name}.csv")
    return output


def run_universe(universe: str) -> Mapping[str, pd.DataFrame]:
    """Run and emit one universe's fixed strategy comparison."""
    runners = {
        "blackrock_funds": _run_u2,
        "futures": _run_u3,
        "msci_us": _run_u1,
    }
    if universe not in runners:
        raise KeyError(f"unknown universe {universe!r}; expected {tuple(runners)}")
    return _emit(universe, runners[universe]())


def _artifact_hashes(universe: str) -> dict[str, str]:
    """Hash all deterministic D4-D5 CSVs except timing and replay records."""
    return {
        path.name: d4._sha256(path)
        for path in sorted(d4._universe_root(universe).glob("*.csv"))
        if path.name not in {"runtime.csv", "determinism.csv"}
    }


def verify_determinism(universe: str) -> pd.DataFrame:
    """Require two complete cache-first strategy emissions to be byte-identical."""
    run_universe(universe)
    first = _artifact_hashes(universe)
    run_universe(universe)
    second = _artifact_hashes(universe)
    names = sorted(set(first) | set(second))
    replay = pd.DataFrame(
        {
            "artifact": names,
            "first_sha256": [first.get(name) for name in names],
            "second_sha256": [second.get(name) for name in names],
            "byte_identical": [first.get(name) == second.get(name) for name in names],
        }
    )
    d4._write(replay, d4._universe_root(universe) / "determinism.csv")
    if not replay["byte_identical"].all():
        raise AssertionError(replay.loc[~replay["byte_identical"]])
    return replay


def run_all(*, verify: bool = True) -> Mapping[str, Mapping[str, pd.DataFrame]]:
    """Run U2, U3, then U1 in the roadmap's frozen order."""
    output = {}
    for universe in UNIVERSES:
        print(f"de-PC1 strategy comparison: {universe}", flush=True)
        output[universe] = run_universe(universe)
        if verify:
            verify_determinism(universe)
    return output


def main() -> None:
    """Execute and deterministically replay all three strategy comparisons."""
    run_all(verify=True)
    for universe in UNIVERSES:
        performance = pd.read_csv(
            d4._universe_root(universe) / "performance.csv",
            float_precision="round_trip",
        )
        columns = ["analysis_window", "leg", "net_return_annualized", "sharpe_rf0"]
        print(f"\n{universe}\n{performance[columns].to_string(index=False)}", flush=True)


if __name__ == "__main__":
    main()
