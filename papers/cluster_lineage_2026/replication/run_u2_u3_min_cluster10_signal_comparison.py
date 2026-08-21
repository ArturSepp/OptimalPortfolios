"""Compare classic and ROSAA cluster signals for U2 and U3 at fallback 10.

Both signal definitions use the public OptimalPortfolios global and rolling-cluster
APIs.  Clusters affect score standardisation only; global and cluster score panels
then use the same canonical top/bottom quantile rank within each universe's fixed
strategic sleeves.  U2 retains AUM100, 50/30/20 budgets, every-two-month decisions,
and 20 bp costs.  U3 retains the owner-frozen futures exclusions, 30/30/30/10
budgets, monthly decisions on the U1 window, and 10 bp costs.
"""
from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
import qis
from optimalportfolios.alphas import (
    compute_classic_momentum_alpha,
    compute_classic_momentum_cluster_alpha,
    compute_momentum_alpha,
    compute_momentum_cluster_alpha,
    compute_top_quantile_equal_weights,
)

import papers.cluster_lineage_2026.replication.run_backtests as e5
import papers.cluster_lineage_2026.replication.run_futures_best_relative_pnl_scatter as u3_best
import papers.cluster_lineage_2026.replication.run_futures_prod_signal_grid_30303010_10bp as u3_grid
import papers.cluster_lineage_2026.replication.run_futures_sleeve_grid as u3_equal
import papers.cluster_lineage_2026.replication.run_futures_weight_30303010 as u3_weights
import papers.cluster_lineage_2026.replication.run_u1_covar_grid_long_short_prod as sampler
import papers.cluster_lineage_2026.replication.run_u2_all_funds_asset_class_attribution as u2
import papers.cluster_lineage_2026.replication.run_u2_blackrock_aum_filter as u2_aum
import papers.cluster_lineage_2026.replication.run_u2_blackrock_aum_sensitivity as u2_sensitivity
import papers.cluster_lineage_2026.replication.run_u2_blackrock_etf_grid as u2_funds
import papers.cluster_lineage_2026.replication.run_u2_blackrock_long_short_search as u2_search
import papers.cluster_lineage_2026.replication.run_u2_blackrock_sleeve_grid as u2_sleeves


MIN_CLUSTER_SIZE = 10
SIGNALS = ("classic_12m_ex_1m", "rosaa_risk_adjusted_momentum")
PRIMARY_SIGNAL_BY_UNIVERSE = {
    "U2_funds": "classic_12m_ex_1m",
    "U3_futures": "rosaa_risk_adjusted_momentum",
}
RUNNER = (
    "papers/cluster_lineage_2026/replication/"
    "run_u2_u3_min_cluster10_signal_comparison.py"
)
TOLERANCE = 1e-12


def _root() -> Path:
    """Return the gitignored local comparison directory."""
    root = (
        Path(__file__).resolve().parents[1]
        / "local_outputs"
        / "e5b"
        / "u2_u3_min_cluster10_classic_vs_rosaa_20260816"
    )
    root.mkdir(parents=True, exist_ok=True)
    return root


def _panel_error(left: pd.DataFrame, right: pd.DataFrame) -> tuple[float, bool]:
    """Return maximum finite absolute error and NaN-mask agreement."""
    left, right = left.align(right, join="outer")
    difference = left.subtract(right).abs().to_numpy()
    finite = difference[np.isfinite(difference)]
    return (
        float(finite.max()) if finite.size else 0.0,
        bool(left.isna().equals(right.isna())),
    )


def _signal_pair(
    *,
    signal_id: str,
    prices: pd.DataFrame,
    benchmark: pd.Series,
    groups: pd.DataFrame,
    dates: pd.DatetimeIndex,
    eligibility: pd.DataFrame,
    rosaa_short_span: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, Mapping[str, object]]:
    """Compute matched global and cluster scores through public signal APIs."""
    rolling_groups = sampler._panel_dict(groups)
    if signal_id == "classic_12m_ex_1m":
        global_source, global_raw = compute_classic_momentum_alpha(
            prices=prices,
            returns_freq="ME",
            group_data=None,
            lookback_periods=12,
            skip_periods=1,
        )
        cluster_source, cluster_raw = compute_classic_momentum_cluster_alpha(
            prices=prices,
            rolling_clusters=rolling_groups,
            returns_freq="ME",
            lookback_periods=12,
            skip_periods=1,
            min_cluster_size=MIN_CLUSTER_SIZE,
        )
    elif signal_id == "rosaa_risk_adjusted_momentum":
        global_source, global_raw = compute_momentum_alpha(
            prices=prices,
            benchmark_price=benchmark,
            returns_freq="ME",
            group_data=None,
            long_span=12,
            short_span=rosaa_short_span,
            vol_span=13,
            mean_adj_type=qis.MeanAdjType.EWMA,
        )
        cluster_source, cluster_raw = compute_momentum_cluster_alpha(
            prices=prices,
            benchmark_price=benchmark,
            rolling_clusters=rolling_groups,
            returns_freq="ME",
            long_span=12,
            short_span=rosaa_short_span,
            vol_span=13,
            mean_adj_type=qis.MeanAdjType.EWMA,
            min_cluster_size=MIN_CLUSTER_SIZE,
        )
    else:
        raise KeyError(signal_id)

    raw_error, raw_nan_match = _panel_error(global_raw, cluster_raw)
    if raw_error > 0.0 or not raw_nan_match:
        raise AssertionError(f"{signal_id} global and cluster raw panels differ")
    global_scores, global_timestamps = sampler._asof_panel(global_source, dates)
    cluster_scores, cluster_timestamps = sampler._asof_panel(cluster_source, dates)
    global_scores = global_scores.reindex(
        index=dates, columns=eligibility.columns
    ).where(eligibility)
    cluster_scores = cluster_scores.reindex(
        index=dates, columns=eligibility.columns
    ).where(eligibility)
    diagnostics = {
        "signal_id": signal_id,
        "min_cluster_size": MIN_CLUSTER_SIZE,
        "rosaa_short_span": rosaa_short_span,
        "raw_panel_max_abs_error": raw_error,
        "raw_nan_mask_match": raw_nan_match,
        "max_global_lookahead_days": float(
            global_timestamps.sub(global_timestamps.index).dt.days.max()
        ),
        "max_cluster_lookahead_days": float(
            cluster_timestamps.sub(cluster_timestamps.index).dt.days.max()
        ),
        "global_valid_min": int(global_scores.notna().sum(axis=1).min()),
        "cluster_valid_min": int(cluster_scores.notna().sum(axis=1).min()),
    }
    if max(
        diagnostics["max_global_lookahead_days"],
        diagnostics["max_cluster_lookahead_days"],
    ) > 0.0:
        raise AssertionError(f"{signal_id} sampled with look-ahead")
    return global_scores, cluster_scores, diagnostics


def _long_short_weights(
    *,
    scores: pd.DataFrame,
    prices: pd.DataFrame,
    eligibility: pd.DataFrame,
    sleeve_panel: pd.DataFrame,
    sleeves: tuple[str, ...],
    target: Mapping[str, float],
    q: float,
) -> tuple[pd.DataFrame, Mapping[str, float]]:
    """Apply the canonical OP rank separately within each strategic sleeve."""
    prices = prices.reindex(columns=scores.columns).reindex(
        index=scores.index, method="ffill"
    )
    eligibility = eligibility.reindex_like(scores).fillna(False).astype(bool)
    sleeve_panel = sleeve_panel.reindex_like(scores)

    def ranked_side(side_scores: pd.DataFrame) -> pd.DataFrame:
        """Return one target-budgeted canonical rank side."""
        output = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)
        for sleeve in sleeves:
            available = eligibility & sleeve_panel.eq(sleeve)
            weights = compute_top_quantile_equal_weights(
                alpha_scores=side_scores.where(available),
                prices=prices.where(available),
                quantile=q,
            )
            if weights.sum(axis=1).le(0.0).any():
                raise AssertionError(f"{sleeve} has an empty canonical selection")
            output = output.add(weights.mul(target[sleeve]), fill_value=0.0)
        return output

    long_book = ranked_side(scores)
    short_book = ranked_side(-scores)
    overlap = long_book.gt(0.0) & short_book.gt(0.0)
    if overlap.to_numpy().any():
        raise AssertionError("canonical long and short selections overlap")
    weights = long_book - short_book
    errors = {
        "long_exposure_abs_error": float(
            long_book.sum(axis=1).sub(1.0).abs().max()
        ),
        "short_exposure_abs_error": float(
            short_book.sum(axis=1).sub(1.0).abs().max()
        ),
        "net_exposure_abs_error": float(weights.sum(axis=1).abs().max()),
        "gross_exposure_abs_error": float(
            weights.abs().sum(axis=1).sub(2.0).abs().max()
        ),
        "weight_outside_eligibility_abs_error": float(
            weights.where(~eligibility, 0.0).abs().to_numpy().max()
        ),
        "overlap_assets": int(overlap.to_numpy().sum()),
    }
    for sleeve in sleeves:
        mask = sleeve_panel.eq(sleeve)
        errors[f"{sleeve}_long_budget_abs_error"] = float(
            long_book.where(mask, 0.0).sum(axis=1).sub(target[sleeve]).abs().max()
        )
        errors[f"{sleeve}_short_budget_abs_error"] = float(
            short_book.where(mask, 0.0).sum(axis=1).sub(target[sleeve]).abs().max()
        )
    return weights, errors


def _u2_context() -> Mapping[str, object]:
    """Load the frozen AUM100 U2 inputs without refitting partitions."""
    daily = u2_funds._read_daily()
    dates = u2_funds._dates()
    headline_dates = dates[
        (dates >= u2_funds.HEADLINE_START) & (dates <= u2_funds.HEADLINE_END)
    ]
    rolling_aum = u2_aum._rolling_aum()
    eligibility_all = u2_sensitivity._eligibilities(daily, dates, rolling_aum)
    monthly_returns = u2_funds._native_returns(daily, "ME")
    monthly_eligibility = u2_sensitivity._eligibilities(
        daily, monthly_returns.index, rolling_aum
    )[u2.FILTER_ID]
    partitions, _, cache_status = u2_sensitivity._build_partitions(eligibility_all)
    if cache_status != "hit":
        raise AssertionError("U2 comparison must consume the completed partition cache")
    eligibility = eligibility_all[u2.FILTER_ID].reindex(index=headline_dates).astype(bool)
    groups = partitions[u2.FILTER_ID].reindex(index=headline_dates)
    missing = int((eligibility & groups.isna()).to_numpy().sum())
    if missing:
        raise AssertionError(f"U2 has {missing} missing eligible cluster memberships")
    simple_returns = np.expm1(monthly_returns)
    signal_prices = qis.returns_to_nav(simple_returns)
    benchmark_returns = simple_returns.where(monthly_eligibility).mean(axis=1)
    benchmark = qis.returns_to_nav(benchmark_returns.rename("EW").to_frame())["EW"]
    sleeve_map = u2_sleeves._broad_sleeves(daily.columns)
    sleeve_panel = u2_sleeves._sleeve_panel(headline_dates, sleeve_map)
    performance_prices = u2_funds._performance_prices(daily)
    rank_prices = performance_prices.reindex(index=headline_dates, method="ffill")
    scheduled_dates = u2_search._rebalance_dates(headline_dates, u2.SCHEDULE)
    window = u2_sensitivity._window(
        performance_prices,
        eligibility_all[u2.FILTER_ID],
        u2_search.FULL_WINDOW,
        headline_dates,
    )
    return {
        "universe": "U2_funds",
        "dates": headline_dates,
        "eligibility": eligibility,
        "groups": groups,
        "signal_prices": signal_prices,
        "benchmark": benchmark,
        "rank_prices": rank_prices,
        "performance_prices": window["prices"],
        "scheduled_dates": scheduled_dates,
        "sleeve_panel": sleeve_panel,
        "sleeves": tuple(u2_sleeves.SLEEVES),
        "target": dict(u2.TARGET),
        "q": u2.Q,
        "cost_bps": u2.COST_BPS,
        "ew_nav": window["ew_nav"],
        "window": u2_search.FULL_WINDOW,
        "schedule": u2.SCHEDULE,
    }


def _u3_context() -> Mapping[str, object]:
    """Load the frozen owner-selected U3 futures inputs and M1-star groups."""
    context = u3_grid._build_context()
    groups = context["groups_by_method"][u3_best.CLUSTER_METHOD]
    eligibility = context["eligibility"]
    missing = int((eligibility & groups.isna()).to_numpy().sum())
    if missing:
        raise AssertionError(f"U3 has {missing} missing eligible cluster memberships")
    return {
        "universe": "U3_futures",
        "dates": context["dates"],
        "eligibility": eligibility,
        "groups": groups,
        "signal_prices": context["signal_prices"],
        "benchmark": context["benchmark"],
        "rank_prices": context["performance_prices"],
        "performance_prices": context["performance_prices"],
        "scheduled_dates": context["dates"],
        "sleeve_panel": context["sleeve_panel"],
        "sleeves": tuple(u3_equal.SLEEVES),
        "target": dict(u3_weights.TARGET),
        "q": u3_best.Q,
        "cost_bps": u3_best.COST_BPS,
        "ew_nav": context["ew_nav"],
        "window": u3_best.matched.WINDOW,
        "schedule": "ME",
    }


def _backtest(
    context: Mapping[str, object],
    weights: pd.DataFrame,
    ticker: str,
) -> Mapping[str, float]:
    """Run one universe through its accepted qis-backed backtest wrapper."""
    scheduled = weights.reindex(index=context["scheduled_dates"])
    if context["universe"] == "U2_funds":
        net, gross = u2_funds._backtest(
            context["performance_prices"],
            scheduled,
            context["cost_bps"] / 10000.0,
            ticker,
        )
        return u2_sleeves._performance_payload(net, gross, context["ew_nav"])
    net, gross = u3_equal._backtest(
        context["performance_prices"],
        scheduled,
        context["cost_bps"] / 10000.0,
        ticker,
    )
    return u3_equal._performance_payload(net, gross, context["ew_nav"])


def _comparison(performance: pd.DataFrame) -> pd.DataFrame:
    """Return cluster-minus-global deltas for every universe and signal."""
    rows = []
    metrics = (
        "gross_return_annualized",
        "net_return_annualized",
        "volatility_annualized",
        "sharpe_rf0",
        "one_way_turnover_annualized",
        "cost_drag_bp_per_year",
        "net_total_return",
    )
    for (universe, signal_id), frame in performance.groupby(
        ["universe", "signal_id"], sort=True
    ):
        indexed = frame.set_index("method")
        cluster = indexed.loc["cluster"]
        global_rank = indexed.loc["global"]
        row = {"universe": universe, "signal_id": signal_id}
        for metric in metrics:
            row[f"cluster_{metric}"] = cluster[metric]
            row[f"global_{metric}"] = global_rank[metric]
            row[f"delta_{metric}"] = cluster[metric] - global_rank[metric]
        row["cluster_beats_global_net_return"] = (
            row["delta_net_return_annualized"] > 0.0
        )
        row["cluster_beats_global_sharpe"] = row["delta_sharpe_rf0"] > 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def run() -> Mapping[str, pd.DataFrame]:
    """Execute and validate the U2/U3 two-signal comparison."""
    started = time.perf_counter()
    contexts = (_u2_context(), _u3_context())
    performance_rows = []
    signal_rows = []
    acceptance_rows = []
    for context in contexts:
        for signal_id in SIGNALS:
            global_scores, cluster_scores, signal_diagnostics = _signal_pair(
                signal_id=signal_id,
                prices=context["signal_prices"],
                benchmark=context["benchmark"],
                groups=context["groups"],
                dates=context["dates"],
                eligibility=context["eligibility"],
            )
            signal_rows.append(
                {"universe": context["universe"], **signal_diagnostics}
            )
            for method, scores in (
                ("global", global_scores),
                ("cluster", cluster_scores),
            ):
                weights, errors = _long_short_weights(
                    scores=scores,
                    prices=context["rank_prices"],
                    eligibility=context["eligibility"],
                    sleeve_panel=context["sleeve_panel"],
                    sleeves=context["sleeves"],
                    target=context["target"],
                    q=context["q"],
                )
                maximum_error = max(float(value) for value in errors.values())
                acceptance_rows.append(
                    {
                        "universe": context["universe"],
                        "signal_id": signal_id,
                        "method": method,
                        **errors,
                        "maximum_error": maximum_error,
                        "tolerance": TOLERANCE,
                        "status": "PASS" if maximum_error <= TOLERANCE else "FAIL",
                    }
                )
                payload = _backtest(
                    context,
                    weights,
                    f"{context['universe']}_{signal_id}_{method}_min10",
                )
                performance_rows.append(
                    {
                        "universe": context["universe"],
                        "analysis_window": context["window"],
                        "signal_id": signal_id,
                        "method": method,
                        "min_cluster_size": MIN_CLUSTER_SIZE,
                        "q": context["q"],
                        "schedule": context["schedule"],
                        "cost_bps_one_way": context["cost_bps"],
                        "construction": "canonical_OP_rank_within_strategic_sleeve",
                        **payload,
                        "runner": RUNNER,
                    }
                )

    performance = pd.DataFrame(performance_rows)
    signal_diagnostics = pd.DataFrame(signal_rows)
    acceptance = pd.DataFrame(acceptance_rows)
    if not acceptance["status"].eq("PASS").all():
        raise AssertionError(acceptance.loc[~acceptance["status"].eq("PASS")])
    output = {
        "performance": performance,
        "primary_performance": performance.loc[
            performance.apply(
                lambda row: row["signal_id"]
                == PRIMARY_SIGNAL_BY_UNIVERSE[row["universe"]],
                axis=1,
            )
        ].reset_index(drop=True),
        "comparison": _comparison(performance),
        "signal_diagnostics": signal_diagnostics,
        "acceptance": acceptance,
        "design": pd.DataFrame(
            [
                {
                    "universes": "U2_funds|U3_futures",
                    "signals": "|".join(SIGNALS),
                    "min_cluster_size": MIN_CLUSTER_SIZE,
                    "cluster_role": "score standardisation only",
                    "rank_rule": "canonical OP top/bottom rank within strategic sleeve",
                    "u2_filter": "12m average AUM strictly above USD100m",
                    "u2_sleeve_budgets": "Equity50|FixedIncome30|Rest20",
                    "u2_cost_bps_one_way": u2.COST_BPS,
                    "u3_sleeve_budgets": "Equity30|FixedIncome30|Commodities30|FX10",
                    "u3_cost_bps_one_way": u3_best.COST_BPS,
                    "classic_spec": "ME 12m-ex-1m raw momentum",
                    "rosaa_spec": "ME long12 shortNone vol13 meanEWMA eligible-EW benchmark",
                    "primary_signals": "U2=classic_12m_ex_1m|U3=rosaa_risk_adjusted_momentum",
                    "implementation_lag_periods": 1,
                    "runner": RUNNER,
                }
            ]
        ),
        "runtime": pd.DataFrame(
            [{"runtime_seconds": time.perf_counter() - started, "runner": RUNNER}]
        ),
    }
    for name, frame in output.items():
        e5._write(frame, _root() / f"{name}.csv")
    return output


def _hash_outputs() -> dict[str, str]:
    """Hash deterministic comparison artifacts excluding runtime and replay."""
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(_root().glob("*.csv"))
        if path.name not in {"runtime.csv", "determinism.csv"}
    }


def verify_determinism() -> pd.DataFrame:
    """Replay a complete result once and require byte-identical artifacts."""
    required = {
        "acceptance.csv",
        "comparison.csv",
        "design.csv",
        "performance.csv",
        "primary_performance.csv",
        "signal_diagnostics.csv",
    }
    if not required.issubset({path.name for path in _root().glob("*.csv")}):
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
    """Execute, replay, and print U2/U3 signal comparisons."""
    replay = verify_determinism()
    performance = pd.read_csv(
        _root() / "performance.csv", float_precision="round_trip"
    )
    columns = [
        "universe",
        "signal_id",
        "method",
        "net_return_annualized",
        "volatility_annualized",
        "sharpe_rf0",
        "one_way_turnover_annualized",
    ]
    print(performance[columns].to_string(index=False))
    print(
        f"U2/U3 min-cluster-10 signal comparison: PASS "
        f"({len(replay)}/{len(replay)} deterministic)",
        flush=True,
    )


if __name__ == "__main__":
    main()
