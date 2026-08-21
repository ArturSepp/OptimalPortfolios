"""Attribute the corrected all-fund U2 long-short comparison by official asset class.

The primary point-in-time USD100m AUM rule is applied to the complete BlackRock fund
universe before clustering and ranking. Equity, Fixed Income, and Rest receive 50/30/20
of each long and short side. Both the global and cluster-score legs use the canonical
OptimalPortfolios top-quantile equal-weight function; clusters affect score
standardisation only. Official asset classes are used only for ex-post attribution.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
import qis
from optimalportfolios.alphas import (
    compute_classic_momentum_alpha,
    compute_classic_momentum_cluster_alpha,
    compute_top_quantile_equal_weights,
)

import papers.cluster_lineage_2026.replication.run_backtests as e5
import papers.cluster_lineage_2026.replication.run_depc1_strategy_backtests as accounting
import papers.cluster_lineage_2026.replication.run_u2_blackrock_aum_filter as aum
import papers.cluster_lineage_2026.replication.run_u2_blackrock_aum_sensitivity as sensitivity
import papers.cluster_lineage_2026.replication.run_u2_blackrock_etf_grid as funds
import papers.cluster_lineage_2026.replication.run_u2_blackrock_long_short_search as search
import papers.cluster_lineage_2026.replication.run_u2_blackrock_sleeve_grid as sleeves
import papers.cluster_lineage_2026.replication.run_u2_equity_fi_fund_pnl_attribution as attr
import papers.cluster_lineage_2026.replication.run_u1_covar_grid_long_short_prod as prod


RUNNER = (
    "papers/cluster_lineage_2026/replication/"
    "run_u2_all_funds_asset_class_attribution.py"
)
FILTER_ID = "aum_100m"
AUM_CUTOFF_USD_MILLIONS = 100.0
WEIGHT_ID = "E50_F30_R20"
TARGET = search._target_map(WEIGHT_ID)
Q = 0.25
SCHEDULE = "every_two_months"
COST_BPS = 20.0
TOLERANCE = 1e-10
WEIGHT_TOLERANCE = 1e-12
ROSAA_SIGNAL = "rosaa_risk_adjusted_momentum"
CLASSIC_SIGNAL = "classic_12m_ex_1m"
SIGNAL_LABELS = {
    ROSAA_SIGNAL: "ROSAA production risk-adjusted momentum",
    CLASSIC_SIGNAL: "classic 12m-ex-1m momentum",
}
OFFICIAL_CLASSES = (
    "Equity",
    "Fixed Income",
    "Multi Asset",
    "Digital Assets",
    "Commodity",
    "Real Estate",
    "Cash",
)


def _root(signal: str = ROSAA_SIGNAL) -> Path:
    """Return the isolated all-fund attribution directory for one signal."""
    names = {
        ROSAA_SIGNAL: "all_funds_aum100_asset_class_attribution_20260816",
        CLASSIC_SIGNAL: (
            "all_funds_aum100_asset_class_attribution_classic_12m_ex_1m_20260816"
        ),
    }
    if signal not in names:
        raise KeyError(signal)
    root = funds._root() / names[signal]
    root.mkdir(parents=True, exist_ok=True)
    return root


def _classic_signal_panels(
    daily: pd.DataFrame,
    dates: pd.DatetimeIndex,
    eligibility: pd.DataFrame,
    clusters: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, Mapping[str, float]]:
    """Build classic global and cluster scores through the public OP signal API."""
    monthly_returns = funds._native_returns(daily, "ME")
    signal_prices = qis.returns_to_nav(np.expm1(monthly_returns))
    global_source, global_raw = compute_classic_momentum_alpha(
        prices=signal_prices,
        returns_freq="ME",
        group_data=None,
        lookback_periods=12,
        skip_periods=1,
    )
    cluster_source, cluster_raw = compute_classic_momentum_cluster_alpha(
        prices=signal_prices,
        rolling_clusters=funds._panel_dict(clusters),
        returns_freq="ME",
        lookback_periods=12,
        skip_periods=1,
        min_cluster_size=5,
    )
    raw_difference = (global_raw - cluster_raw).abs().to_numpy()
    finite_difference = raw_difference[np.isfinite(raw_difference)]
    raw_error = float(finite_difference.max()) if finite_difference.size else 0.0
    raw_nan_match = bool(global_raw.isna().equals(cluster_raw.isna()))
    global_scores, global_timestamps = prod._asof_panel(global_source, dates)
    cluster_scores, cluster_timestamps = prod._asof_panel(cluster_source, dates)
    global_scores = global_scores.reindex(
        index=dates, columns=eligibility.columns
    ).where(eligibility)
    cluster_scores = cluster_scores.reindex(
        index=dates, columns=eligibility.columns
    ).where(eligibility)
    diagnostics = {
        "max_global_lookahead_days": float(
            global_timestamps.sub(global_timestamps.index).dt.days.max()
        ),
        "max_cluster_lookahead_days": float(
            cluster_timestamps.sub(cluster_timestamps.index).dt.days.max()
        ),
        "global_valid_min": float(global_scores.notna().sum(axis=1).min()),
        "cluster_valid_min": float(cluster_scores.notna().sum(axis=1).min()),
        "classic_raw_panel_max_abs_error": raw_error,
        "classic_raw_nan_mask_match": raw_nan_match,
    }
    if raw_error > 0.0 or not raw_nan_match:
        raise AssertionError(f"classic raw signal panels differ: {diagnostics}")
    return global_scores, cluster_scores, diagnostics


def _ranked_side(
    scores: pd.DataFrame,
    prices: pd.DataFrame,
    eligibility: pd.DataFrame,
    sleeve_panel: pd.DataFrame,
) -> pd.DataFrame:
    """Apply the canonical OP rank rule within each fixed strategic sleeve."""
    prices = prices.reindex(index=scores.index, columns=scores.columns)
    eligibility = eligibility.reindex_like(scores).fillna(False).astype(bool)
    sleeve_panel = sleeve_panel.reindex_like(scores)
    output = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)
    for sleeve in sleeves.SLEEVES:
        available = eligibility & sleeve_panel.eq(sleeve)
        weights = compute_top_quantile_equal_weights(
            alpha_scores=scores.where(available),
            prices=prices.where(available),
            quantile=Q,
        )
        if weights.sum(axis=1).le(0.0).any():
            raise AssertionError(f"{sleeve} has no canonical rank selection")
        output = output.add(weights.mul(TARGET[sleeve]), fill_value=0.0)
    return output


def _long_short_weights(
    scores: pd.DataFrame,
    prices: pd.DataFrame,
    eligibility: pd.DataFrame,
    sleeve_panel: pd.DataFrame,
) -> tuple[pd.DataFrame, Mapping[str, float]]:
    """Apply the canonical long-only rank rule to scores and negative scores."""
    long_book = _ranked_side(scores, prices, eligibility, sleeve_panel)
    short_book = _ranked_side(-scores, prices, eligibility, sleeve_panel)
    overlap = long_book.gt(0.0) & short_book.gt(0.0)
    if overlap.to_numpy().any():
        raise AssertionError("canonical top and bottom quantiles overlap")
    weights = long_book - short_book
    errors = {
        "long_exposure_error": float(long_book.sum(axis=1).sub(1.0).abs().max()),
        "short_exposure_error": float(short_book.sum(axis=1).sub(1.0).abs().max()),
        "net_exposure_error": float(weights.sum(axis=1).abs().max()),
        "gross_exposure_error": float(
            weights.abs().sum(axis=1).sub(2.0).abs().max()
        ),
        "overlap_assets": int(overlap.to_numpy().sum()),
        "maximum_abs_target_weight": float(weights.abs().to_numpy().max()),
    }
    for sleeve in sleeves.SLEEVES:
        mask = sleeve_panel.eq(sleeve)
        errors[f"{sleeve}_long_budget_error"] = float(
            long_book.where(mask, 0.0).sum(axis=1).sub(TARGET[sleeve]).abs().max()
        )
        errors[f"{sleeve}_short_budget_error"] = float(
            short_book.where(mask, 0.0).sum(axis=1).sub(TARGET[sleeve]).abs().max()
        )
    return weights, errors


def _instrument_attribution(
    portfolio,
    method: str,
    metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, Mapping[str, object]]:
    """Return exact per-fund P&L as a percentage of beginning portfolio NAV."""
    frame, diagnostics = accounting._instrument_attribution(
        portfolio, method, "U2_All_BlackRock_Funds"
    )
    frame["name"] = frame["asset"].map(metadata["name"])
    frame["asset_class"] = frame["asset"].map(metadata["asset_class"])
    frame["sub_asset_class"] = frame["asset"].map(metadata["sub_asset_class"])
    beginning_nav = float(diagnostics["beginning_nav"])
    for source, target in (
        ("long_gross_pnl_currency", "long_gross_pnl_pct_of_start"),
        ("short_gross_pnl_currency", "short_gross_pnl_pct_of_start"),
        ("transaction_cost_currency", "transaction_cost_pct_of_start"),
        ("net_pnl_currency", "net_pnl_pct_of_start"),
    ):
        frame[target] = 100.0 * frame[source] / beginning_nav
    return frame, diagnostics


def _asset_class_pnl(attribution: pd.DataFrame) -> pd.DataFrame:
    """Aggregate exact instrument P&L into the seven official asset classes."""
    pnl_columns = [
        "long_gross_pnl_pct_of_start",
        "short_gross_pnl_pct_of_start",
        "transaction_cost_pct_of_start",
        "net_pnl_pct_of_start",
    ]
    grouped = (
        attribution.groupby(["leg", "asset_class"], observed=True)[pnl_columns]
        .sum()
        .reindex(
            pd.MultiIndex.from_product(
                [["global", "cluster"], OFFICIAL_CLASSES],
                names=["leg", "asset_class"],
            ),
            fill_value=0.0,
        )
        .reset_index()
    )
    grouped["component_reconciliation_error"] = (
        grouped["net_pnl_pct_of_start"]
        - grouped["long_gross_pnl_pct_of_start"]
        - grouped["short_gross_pnl_pct_of_start"]
        + grouped["transaction_cost_pct_of_start"]
    )
    totals = grouped.groupby("leg")["net_pnl_pct_of_start"].transform("sum")
    grouped["share_of_leg_net_pnl"] = grouped["net_pnl_pct_of_start"] / totals
    return grouped


def _asset_class_delta(asset_class_pnl: pd.DataFrame) -> pd.DataFrame:
    """Return cluster-minus-global P&L contributions by official asset class."""
    global_pnl = asset_class_pnl.loc[
        asset_class_pnl["leg"].eq("global")
    ].set_index("asset_class")
    cluster_pnl = asset_class_pnl.loc[
        asset_class_pnl["leg"].eq("cluster")
    ].set_index("asset_class")
    rows = []
    for asset_class in OFFICIAL_CLASSES:
        control = global_pnl.loc[asset_class]
        treatment = cluster_pnl.loc[asset_class]
        long_delta = (
            treatment["long_gross_pnl_pct_of_start"]
            - control["long_gross_pnl_pct_of_start"]
        )
        short_delta = (
            treatment["short_gross_pnl_pct_of_start"]
            - control["short_gross_pnl_pct_of_start"]
        )
        cost_effect = -(
            treatment["transaction_cost_pct_of_start"]
            - control["transaction_cost_pct_of_start"]
        )
        net_delta = treatment["net_pnl_pct_of_start"] - control["net_pnl_pct_of_start"]
        rows.append(
            {
                "asset_class": asset_class,
                "global_net_pnl_pct_of_start": control["net_pnl_pct_of_start"],
                "cluster_net_pnl_pct_of_start": treatment["net_pnl_pct_of_start"],
                "delta_long_gross_pnl_pct_of_start": long_delta,
                "delta_short_gross_pnl_pct_of_start": short_delta,
                "delta_cost_effect_pct_of_start": cost_effect,
                "delta_net_pnl_pct_of_start": net_delta,
                "component_reconciliation_error": (
                    net_delta - long_delta - short_delta - cost_effect
                ),
            }
        )
    frame = pd.DataFrame(rows)
    total_delta = frame["delta_net_pnl_pct_of_start"].sum()
    frame["share_of_total_cluster_gap"] = (
        frame["delta_net_pnl_pct_of_start"] / total_delta
    )
    return frame.sort_values("delta_net_pnl_pct_of_start").reset_index(drop=True)


def _asset_class_weights(
    all_weights: Mapping[str, pd.DataFrame],
    scheduled_dates: pd.DatetimeIndex,
    asset_class: pd.Series,
) -> pd.DataFrame:
    """Summarise implemented long and short exposure by official asset class."""
    rows = []
    for method, weights in all_weights.items():
        selected = weights.reindex(index=scheduled_dates).fillna(0.0)
        for label in OFFICIAL_CLASSES:
            panel = selected.loc[:, asset_class.eq(label)]
            long_exposure = panel.clip(lower=0.0).sum(axis=1)
            short_exposure = -panel.clip(upper=0.0).sum(axis=1)
            rows.append(
                {
                    "method": method,
                    "asset_class": label,
                    "fund_count": int(asset_class.eq(label).sum()),
                    "mean_long_exposure": long_exposure.mean(),
                    "mean_short_exposure": short_exposure.mean(),
                    "minimum_long_exposure": long_exposure.min(),
                    "maximum_long_exposure": long_exposure.max(),
                    "minimum_short_exposure": short_exposure.min(),
                    "maximum_short_exposure": short_exposure.max(),
                    "funds_ever_long": int(panel.gt(0.0).any(axis=0).sum()),
                    "funds_ever_short": int(panel.lt(0.0).any(axis=0).sum()),
                }
            )
    return pd.DataFrame(rows)


def _asset_class_eligibility(
    eligibility: pd.DataFrame,
    asset_class: pd.Series,
) -> pd.DataFrame:
    """Summarise point-in-time AUM100 eligible breadth by official asset class."""
    rows = []
    for label in OFFICIAL_CLASSES:
        panel = eligibility.loc[:, asset_class.eq(label)]
        counts = panel.sum(axis=1)
        rows.append(
            {
                "asset_class": label,
                "fund_columns": int(asset_class.eq(label).sum()),
                "funds_ever_eligible": int(panel.any(axis=0).sum()),
                "eligible_at_start": int(counts.iloc[0]),
                "eligible_median": float(counts.median()),
                "eligible_at_end": int(counts.iloc[-1]),
                "eligible_minimum": int(counts.min()),
                "eligible_maximum": int(counts.max()),
            }
        )
    return pd.DataFrame(rows)


def run(signal: str = ROSAA_SIGNAL) -> Mapping[str, pd.DataFrame]:
    """Run and validate one all-fund AUM100 asset-class attribution."""
    if signal not in SIGNAL_LABELS:
        raise KeyError(signal)
    started = time.perf_counter()
    daily = funds._read_daily()
    metadata = attr._metadata(daily.columns)
    observed_classes = tuple(metadata["asset_class"].value_counts().index)
    if set(observed_classes) != set(OFFICIAL_CLASSES):
        raise AssertionError(f"official asset classes changed: {observed_classes}")
    dates = funds._dates()
    headline_dates = dates[
        (dates >= funds.HEADLINE_START) & (dates <= funds.HEADLINE_END)
    ]
    rolling_aum = aum._rolling_aum()
    eligibility_all = sensitivity._eligibilities(daily, dates, rolling_aum)
    monthly_dates = funds._native_returns(daily, "ME").index
    monthly_eligibility = sensitivity._eligibilities(
        daily, monthly_dates, rolling_aum
    )
    partitions, partition_diagnostics, cache_status = sensitivity._build_partitions(
        eligibility_all
    )
    if cache_status != "hit":
        raise AssertionError("all-fund attribution must consume the completed partition cache")

    eligibility = eligibility_all[FILTER_ID].reindex(index=headline_dates).astype(bool)
    clusters = partitions[FILTER_ID].reindex(index=headline_dates)
    missing_memberships = int((eligibility & clusters.isna()).to_numpy().sum())
    if signal == ROSAA_SIGNAL:
        global_scores, cluster_scores, signal_diagnostics = (
            sensitivity._signal_panels(
                daily,
                headline_dates,
                eligibility,
                monthly_eligibility[FILTER_ID],
                clusters,
            )
        )
    else:
        global_scores, cluster_scores, signal_diagnostics = _classic_signal_panels(
            daily, headline_dates, eligibility, clusters
        )
    broad_sleeves = sleeves._broad_sleeves(daily.columns)
    sleeve_panel = sleeves._sleeve_panel(headline_dates, broad_sleeves)
    prices_all = funds._performance_prices(daily)
    rank_prices = prices_all.reindex(index=headline_dates, method="ffill")
    scheduled_dates = search._rebalance_dates(headline_dates, SCHEDULE)

    all_weights = {}
    weight_diagnostics = []
    for method, scores in (("global", global_scores), ("cluster", cluster_scores)):
        weights, diagnostics = _long_short_weights(
            scores, rank_prices, eligibility, sleeve_panel
        )
        all_weights[method] = weights
        outside = float(weights.where(~eligibility, 0.0).abs().to_numpy().max())
        weight_diagnostics.append(
            {"method": method, **diagnostics, "weight_outside_eligibility": outside}
        )

    filtered_window = sensitivity._window(
        prices_all, eligibility_all[FILTER_ID], search.FULL_WINDOW, headline_dates
    )
    performance_rows = []
    attribution_frames = []
    accounting_errors = []
    for method in ("global", "cluster"):
        net, gross = funds._backtest(
            filtered_window["prices"],
            all_weights[method].reindex(index=scheduled_dates),
            COST_BPS / 10000.0,
            f"u2_all_funds_aum100_{method}",
        )
        performance_rows.append(
            {
                "method": method,
                "analysis_window": search.FULL_WINDOW,
                "signal": SIGNAL_LABELS[signal],
                "signal_id": signal,
                "ranking": "canonical OP top-quantile equal-weight by broad sleeve",
                "cluster_role": "score standardisation only",
                "q": Q,
                "weight_id": WEIGHT_ID,
                "schedule": SCHEDULE,
                "cost_bps_one_way": COST_BPS,
                **sleeves._performance_payload(net, gross, filtered_window["ew_nav"]),
            }
        )
        frame, diagnostics = _instrument_attribution(net, method, metadata)
        attribution_frames.append(frame)
        accounting_errors.extend(
            [
                float(diagnostics["max_step_reconciliation_abs_error"]),
                float(diagnostics["cumulative_reconciliation_abs_error"]),
            ]
        )

    performance = pd.DataFrame(performance_rows)
    attribution = pd.concat(attribution_frames, ignore_index=True)
    asset_class_pnl = _asset_class_pnl(attribution)
    asset_class_delta = _asset_class_delta(asset_class_pnl)
    asset_class_weights = _asset_class_weights(
        all_weights, scheduled_dates, metadata["asset_class"]
    )
    asset_class_eligibility = _asset_class_eligibility(
        eligibility, metadata["asset_class"]
    )
    weight_diagnostics = pd.DataFrame(weight_diagnostics)

    pnl_reconciliation = []
    performance_by_method = performance.set_index("method")
    for method in ("global", "cluster"):
        attributed = asset_class_pnl.loc[
            asset_class_pnl["leg"].eq(method), "net_pnl_pct_of_start"
        ].sum()
        portfolio = 100.0 * float(
            performance_by_method.loc[method, "net_total_return"]
        )
        pnl_reconciliation.append(abs(attributed - portfolio))
    portfolio_delta = 100.0 * (
        float(performance_by_method.loc["cluster", "net_total_return"])
        - float(performance_by_method.loc["global", "net_total_return"])
    )
    attribution_delta = asset_class_delta["delta_net_pnl_pct_of_start"].sum()
    error_columns = [column for column in weight_diagnostics if column.endswith("_error")]
    maximum_weight_error = float(
        weight_diagnostics[error_columns].abs().to_numpy().max()
    )
    checks = [
        ("partition cache status", cache_status, "hit", "eq"),
        ("eligible memberships missing from partitions", missing_memberships, 0, "eq"),
        (
            "maximum signal lookahead days",
            max(
                float(signal_diagnostics["max_global_lookahead_days"]),
                float(signal_diagnostics["max_cluster_lookahead_days"]),
            ),
            0.0,
            "le",
        ),
        ("maximum exposure and sleeve-budget error", maximum_weight_error, WEIGHT_TOLERANCE, "le"),
        (
            "maximum weight outside eligibility",
            float(weight_diagnostics["weight_outside_eligibility"].max()),
            WEIGHT_TOLERANCE,
            "le",
        ),
        ("maximum instrument accounting error", max(accounting_errors), TOLERANCE, "le"),
        (
            "maximum asset-class component error",
            float(asset_class_pnl["component_reconciliation_error"].abs().max()),
            TOLERANCE,
            "le",
        ),
        (
            "maximum asset-class delta component error",
            float(asset_class_delta["component_reconciliation_error"].abs().max()),
            TOLERANCE,
            "le",
        ),
        ("maximum portfolio PnL attribution error", max(pnl_reconciliation), TOLERANCE, "le"),
        (
            "cluster-global delta attribution error",
            abs(attribution_delta - portfolio_delta),
            TOLERANCE,
            "le",
        ),
        ("official asset-class rows", len(asset_class_pnl), 14, "eq"),
        (
            "official classes with an eligible fund",
            int(asset_class_eligibility["funds_ever_eligible"].gt(0).sum()),
            len(OFFICIAL_CLASSES),
            "eq",
        ),
    ]
    if signal == CLASSIC_SIGNAL:
        checks.extend(
            [
                (
                    "classic global-cluster raw panel error",
                    float(signal_diagnostics["classic_raw_panel_max_abs_error"]),
                    0.0,
                    "le",
                ),
                (
                    "classic global-cluster raw NaN mask match",
                    bool(signal_diagnostics["classic_raw_nan_mask_match"]),
                    True,
                    "eq",
                ),
            ]
        )
    acceptance = pd.DataFrame(
        [
            {
                "check": check,
                "measured": measured,
                "tolerance": tolerance,
                "status": "PASS"
                if (measured == tolerance if comparison == "eq" else measured <= tolerance)
                else "FAIL",
            }
            for check, measured, tolerance, comparison in checks
        ]
    )
    if not acceptance["status"].eq("PASS").all():
        raise AssertionError(acceptance.loc[~acceptance["status"].eq("PASS")])

    output = {
        "specification": pd.DataFrame(
            [
                {
                    "filter_id": FILTER_ID,
                    "signal": SIGNAL_LABELS[signal],
                    "signal_id": signal,
                    "aum_cutoff_usd_millions": AUM_CUTOFF_USD_MILLIONS,
                    "included_official_classes": "|".join(OFFICIAL_CLASSES),
                    "equity_budget_per_side": TARGET["Equity"],
                    "fixed_income_budget_per_side": TARGET["Fixed Income"],
                    "rest_budget_per_side": TARGET["Rest"],
                    "q": Q,
                    "schedule": SCHEDULE,
                    "cost_bps_one_way": COST_BPS,
                    "runner": RUNNER,
                }
            ]
        ),
        "performance": performance,
        "asset_class_pnl": asset_class_pnl,
        "asset_class_delta_vs_global": asset_class_delta,
        "asset_class_weight_summary": asset_class_weights,
        "asset_class_eligibility": asset_class_eligibility,
        "instrument_pnl": attribution,
        "weight_diagnostics": weight_diagnostics,
        "partition_diagnostics": partition_diagnostics.loc[
            partition_diagnostics["filter_id"].eq(FILTER_ID)
        ].reset_index(drop=True),
        "acceptance": acceptance,
        "runtime": pd.DataFrame(
            [
                {
                    "signal_id": signal,
                    "partition_cache_status": cache_status,
                    "runtime_seconds": time.perf_counter() - started,
                }
            ]
        ),
    }
    for name, frame in output.items():
        e5._write(frame, _root(signal) / f"{name}.csv")
    return output


def _hash_outputs(signal: str = ROSAA_SIGNAL) -> dict[str, str]:
    """Hash deterministic output artifacts for one signal."""
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(_root(signal).glob("*.csv"))
        if path.name not in {"runtime.csv", "determinism.csv"}
    }


def verify_determinism(signal: str = ROSAA_SIGNAL) -> pd.DataFrame:
    """Replay one cache-first attribution and require byte-identical outputs."""
    run(signal)
    first = _hash_outputs(signal)
    run(signal)
    second = _hash_outputs(signal)
    names = sorted(set(first) | set(second))
    replay = pd.DataFrame(
        {
            "artifact": names,
            "first_sha256": [first.get(name) for name in names],
            "second_sha256": [second.get(name) for name in names],
            "byte_identical": [first.get(name) == second.get(name) for name in names],
        }
    )
    e5._write(replay, _root(signal) / "determinism.csv")
    if not replay["byte_identical"].all():
        raise AssertionError(replay.loc[~replay["byte_identical"]])
    return replay


def main() -> None:
    """Run and print the default ROSAA attribution."""
    replay = verify_determinism(ROSAA_SIGNAL)
    performance = pd.read_csv(
        _root(ROSAA_SIGNAL) / "performance.csv", float_precision="round_trip"
    )
    attribution = pd.read_csv(
        _root(ROSAA_SIGNAL) / "asset_class_delta_vs_global.csv",
        float_precision="round_trip",
    )
    print(performance.to_string(index=False), flush=True)
    print(attribution.to_string(index=False), flush=True)
    print(
        f"determinism: {int(replay['byte_identical'].sum())}/{len(replay)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
