"""Run U2 Equity/Fixed-Income 60/40 long-short across an AUM-filter grid.

Commodity, Multi Asset, Digital Assets, Real Estate, and Cash funds are removed before
signal benchmarking, covariance estimation, cluster discovery, ranking, and backtesting.
Both cluster and global books carry 60% Equity and 40% Fixed Income gross exposure on
each side. The fixed model uses ROSAA production risk-adjusted momentum, W-THU/span-156
clusters, q=25%, every-two-month rebalancing, lag 1, and 20 bp one-way costs. The two
methods use the same canonical OptimalPortfolios rank-and-equal-weight construction; only
the score panel differs. The point-in-time AUM cutoff is none, USD50m, or USD100m.
"""

from __future__ import annotations

import hashlib
import pickle
import time
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from factorlasso import compute_clusters_from_corr_matrix
from factorlasso.cluster_smoothing import _iter_correlation_inputs
from optimalportfolios.alphas import compute_top_quantile_equal_weights

import papers.cluster_lineage_2026.replication.run_backtests as e5
import papers.cluster_lineage_2026.replication.run_u2_blackrock_aum_filter as aum
import papers.cluster_lineage_2026.replication.run_u2_blackrock_aum_sensitivity as sensitivity
import papers.cluster_lineage_2026.replication.run_u2_blackrock_etf_grid as funds
import papers.cluster_lineage_2026.replication.run_u2_blackrock_long_short_search as search
import papers.cluster_lineage_2026.replication.run_u2_blackrock_sleeve_grid as sleeves


RUNNER = (
    "papers/cluster_lineage_2026/replication/"
    "run_u2_equity_fi_long_short_aum_grid.py"
)
FILTERS: tuple[tuple[str, float], ...] = (
    ("history_only", 0.0),
    ("aum_50m", 50.0),
    ("aum_100m", 100.0),
)
INCLUDED_CLASSES = ("Equity", "Fixed Income")
EXCLUDED_CLASSES = (
    "Cash",
    "Commodity",
    "Digital Assets",
    "Multi Asset",
    "Real Estate",
)
TARGET = {"Equity": 0.60, "Fixed Income": 0.40, "Rest": 0.0}
FREQUENCY = "W-THU"
SPAN = 156
Q = 0.25
SCHEDULE = "every_two_months"
COST_BPS = 20.0
CACHE_VERSION = 1
TOLERANCE = 1e-12


def _root() -> Path:
    """Return the isolated external output directory."""
    root = funds._root() / "equity_fi_60_40_long_short_aum_grid_20260816"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _filter_label(filter_id: str) -> str:
    """Return a reader-facing AUM-rule label."""
    return {
        "history_only": "no AUM cutoff",
        "aum_50m": "12m average AUM > USD50m",
        "aum_100m": "12m average AUM > USD100m",
    }[filter_id]


def _metadata(columns: pd.Index) -> pd.Series:
    """Return the official Aladdin asset class for every fund."""
    metadata = pd.read_csv(funds.METADATA_FILE).set_index("ticker")
    asset_class = metadata["asset_class"].reindex(columns)
    if asset_class.isna().any():
        raise AssertionError("official asset-class mapping is incomplete")
    observed = set(asset_class.unique())
    declared = set(INCLUDED_CLASSES) | set(EXCLUDED_CLASSES)
    if observed != declared:
        raise AssertionError(f"asset-class set changed: {sorted(observed)}")
    return asset_class


def _restrict_eligibility(
    eligibility: Mapping[str, pd.DataFrame],
    asset_class: pd.Series,
) -> dict[str, pd.DataFrame]:
    """Remove every fund outside Equity and Fixed Income before estimation."""
    included = asset_class.isin(INCLUDED_CLASSES)
    return {
        filter_id: panel & included.reindex(panel.columns).fillna(False)
        for filter_id, panel in eligibility.items()
        if filter_id in dict(FILTERS)
    }


def _fingerprint() -> str:
    """Return a stable digest for inputs and all frozen parameters."""
    payload = "|".join(
        [
            funds._input_fingerprint(),
            funds._sha256(aum.AUM_FILE),
            repr(FILTERS),
            repr(INCLUDED_CLASSES),
            repr(EXCLUDED_CLASSES),
            repr(TARGET),
            FREQUENCY,
            str(SPAN),
            str(Q),
            SCHEDULE,
            str(COST_BPS),
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _partition_path() -> Path:
    """Return the isolated multi-filter partition-cache path."""
    return _root() / "partitions.pkl"


def _partitions(
    daily: pd.DataFrame,
    dates: pd.DatetimeIndex,
    eligibility: Mapping[str, pd.DataFrame],
    *,
    force: bool = False,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, str]:
    """Fit or load partitions after the official-class exclusions."""
    path = _partition_path()
    fingerprint = _fingerprint()
    if path.exists() and not force:
        with path.open("rb") as stream:
            cached = pickle.load(stream)
        if (
            cached.get("version") == CACHE_VERSION
            and cached.get("fingerprint") == fingerprint
        ):
            return cached["panels"], cached["diagnostics"], "hit"

    returns = funds._native_returns(daily, FREQUENCY)
    model = funds._model(SPAN, FREQUENCY)
    panels = {
        filter_id: pd.DataFrame(np.nan, index=dates, columns=daily.columns)
        for filter_id, _ in FILTERS
    }
    rows = []
    for date, full_corr in _iter_correlation_inputs(returns, list(dates), model):
        for filter_id, cutoff in FILTERS:
            assets = eligibility[filter_id].columns[
                eligibility[filter_id].loc[date].astype(bool)
            ]
            corr = full_corr.reindex(index=assets, columns=assets)
            if len(assets) == 0:
                labels = pd.Series(dtype=float)
            elif len(assets) == 1:
                labels = pd.Series(1, index=assets)
            else:
                labels, _, _ = compute_clusters_from_corr_matrix(
                    corr,
                    cutoff_fraction=model.cutoff_fraction,
                    linkage_method=model.linkage_method,
                    distance_transform=model.distance_transform,
                    n_clusters=model.n_clusters,
                )
            panels[filter_id].loc[date, labels.index] = labels.to_numpy()
            rows.append(
                {
                    "filter_id": filter_id,
                    "aum_cutoff_usd_millions": cutoff,
                    "date": date,
                    "eligible_assets": len(assets),
                    "partition_assets": len(labels),
                    "clusters": int(labels.nunique()),
                }
            )
    diagnostics = pd.DataFrame(rows)
    payload = {
        "version": CACHE_VERSION,
        "fingerprint": fingerprint,
        "panels": panels,
        "diagnostics": diagnostics,
    }
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)
    return panels, diagnostics, "miss"


def _rosaa_ranked_side(
    scores: pd.DataFrame,
    prices: pd.DataFrame,
    eligibility: pd.DataFrame,
    sleeve_panel: pd.DataFrame,
) -> pd.DataFrame:
    """Build one side with the canonical OP top-quantile equal-weight rule."""
    prices = prices.reindex(index=scores.index, columns=scores.columns)
    eligibility = eligibility.reindex_like(scores).fillna(False).astype(bool)
    sleeve_panel = sleeve_panel.reindex_like(scores)
    output = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)
    for sleeve in INCLUDED_CLASSES:
        available = eligibility & sleeve_panel.eq(sleeve)
        sleeve_weights = compute_top_quantile_equal_weights(
            alpha_scores=scores.where(available),
            prices=prices.where(available),
            quantile=Q,
        )
        if sleeve_weights.sum(axis=1).le(0.0).any():
            raise AssertionError(f"{sleeve} has no canonical rank selection")
        output = output.add(sleeve_weights.mul(TARGET[sleeve]), fill_value=0.0)
    return output


def _long_short_weights(
    scores: pd.DataFrame,
    prices: pd.DataFrame,
    eligibility: pd.DataFrame,
    sleeve_panel: pd.DataFrame,
) -> tuple[pd.DataFrame, Mapping[str, float]]:
    """Build +1/-1 by applying the canonical OP long-only rank rule twice."""
    long_book = _rosaa_ranked_side(scores, prices, eligibility, sleeve_panel)
    short_book = _rosaa_ranked_side(-scores, prices, eligibility, sleeve_panel)
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
    }
    for sleeve in INCLUDED_CLASSES:
        mask = sleeve_panel.eq(sleeve)
        errors[f"{sleeve}_long_budget_error"] = float(
            long_book.where(mask, 0.0).sum(axis=1).sub(TARGET[sleeve]).abs().max()
        )
        errors[f"{sleeve}_short_budget_error"] = float(
            short_book.where(mask, 0.0).sum(axis=1).sub(TARGET[sleeve]).abs().max()
        )
    errors["overlap_assets_removed"] = int(overlap.sum(axis=1).max())
    errors["maximum_abs_target_weight"] = float(weights.abs().to_numpy().max())
    return weights, errors


def _independent_rank_reference_error(
    scores: pd.DataFrame,
    prices: pd.DataFrame,
    eligibility: pd.DataFrame,
    sleeve_panel: pd.DataFrame,
    weights: pd.DataFrame,
) -> float:
    """Check canonical weights against an independent pandas rank reference."""
    prices = prices.reindex(index=scores.index, columns=scores.columns)
    eligibility = eligibility.reindex_like(scores).fillna(False).astype(bool)
    sleeve_panel = sleeve_panel.reindex_like(scores)
    reference = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)
    for sleeve in INCLUDED_CLASSES:
        available = (
            eligibility
            & sleeve_panel.eq(sleeve)
            & scores.notna()
            & prices.notna()
        )
        available_count = available.sum(axis=1)
        selected_count = np.ceil(Q * available_count).astype(int)
        for direction, sign in ((scores, 1.0), (-scores, -1.0)):
            ranks = direction.where(available).rank(
                axis=1, ascending=False, method="first"
            )
            selected = ranks.le(selected_count, axis=0) & available
            side = selected.astype(float).div(selected.sum(axis=1), axis=0)
            reference = reference.add(
                side.mul(sign * TARGET[sleeve]), fill_value=0.0
            )
    return float((weights - reference).abs().to_numpy().max())


def _performance(
    prices_all: pd.DataFrame,
    eligibility_all: Mapping[str, pd.DataFrame],
    weights: Mapping[str, Mapping[str, pd.DataFrame]],
) -> pd.DataFrame:
    """Backtest cluster and global books for every filter and fixed window."""
    rows = []
    headline_dates = next(iter(weights.values()))["global"].index
    for filter_id, cutoff in FILTERS:
        for window_name, (start, end) in search.WINDOWS.items():
            window_dates = headline_dates[
                (headline_dates >= start) & (headline_dates <= end)
            ]
            window = sensitivity._window(
                prices_all,
                eligibility_all[filter_id],
                window_name,
                window_dates,
            )
            scheduled_dates = search._rebalance_dates(window_dates, SCHEDULE)
            for method in ("global", "cluster"):
                net, gross = funds._backtest(
                    window["prices"],
                    weights[filter_id][method].reindex(index=scheduled_dates),
                    COST_BPS / 10000.0,
                    f"u2_eqfi_{filter_id}_{window_name}_{method}",
                )
                rows.append(
                    {
                        "filter_id": filter_id,
                        "aum_cutoff_usd_millions": cutoff,
                        "aum_filter": _filter_label(filter_id),
                        "analysis_window": window_name,
                        "method": method,
                        "strategy": "long_short",
                        "equity_budget_per_side": TARGET["Equity"],
                        "fixed_income_budget_per_side": TARGET["Fixed Income"],
                        "frequency": FREQUENCY,
                        "span": SPAN,
                        "q": Q,
                        "schedule": SCHEDULE,
                        "cost_bps_one_way": COST_BPS,
                        "rebalance_dates": len(scheduled_dates),
                        "runner": RUNNER,
                        **sleeves._performance_payload(net, gross, window["ew_nav"]),
                    }
                )
    return pd.DataFrame(rows)


def _comparison(performance: pd.DataFrame) -> pd.DataFrame:
    """Return cluster-minus-global metrics by filter and window."""
    keys = ["filter_id", "analysis_window"]
    global_rows = performance.loc[performance["method"].eq("global")].set_index(keys)
    rows = []
    for _, cluster in performance.loc[performance["method"].eq("cluster")].iterrows():
        global_row = global_rows.loc[(cluster["filter_id"], cluster["analysis_window"])]
        item = cluster.to_dict()
        for metric in search.COMPARISON_METRICS:
            item[f"global_{metric}"] = global_row[metric]
            item[f"delta_{metric}"] = cluster[metric] - global_row[metric]
        rows.append(item)
    return pd.DataFrame(rows)


def _filter_sensitivity(
    performance: pd.DataFrame,
    comparison: pd.DataFrame,
) -> pd.DataFrame:
    """Return one headline row per AUM cutoff for direct display."""
    headline = performance.loc[
        performance["analysis_window"].eq(search.FULL_WINDOW)
    ].set_index(["filter_id", "method"])
    deltas = comparison.loc[
        comparison["analysis_window"].eq(search.FULL_WINDOW)
    ].set_index("filter_id")
    rows = []
    for filter_id, cutoff in FILTERS:
        global_row = headline.loc[(filter_id, "global")]
        cluster_row = headline.loc[(filter_id, "cluster")]
        delta = deltas.loc[filter_id]
        rows.append(
            {
                "filter_id": filter_id,
                "aum_cutoff_usd_millions": cutoff,
                "aum_filter": _filter_label(filter_id),
                "global_net_return_annualized": global_row["net_return_annualized"],
                "cluster_net_return_annualized": cluster_row["net_return_annualized"],
                "delta_net_return_annualized": delta["delta_net_return_annualized"],
                "global_volatility_annualized": global_row["volatility_annualized"],
                "cluster_volatility_annualized": cluster_row["volatility_annualized"],
                "delta_volatility_annualized": delta["delta_volatility_annualized"],
                "global_sharpe_rf0": global_row["sharpe_rf0"],
                "cluster_sharpe_rf0": cluster_row["sharpe_rf0"],
                "delta_sharpe_rf0": delta["delta_sharpe_rf0"],
                "global_one_way_turnover_annualized": global_row[
                    "one_way_turnover_annualized"
                ],
                "cluster_one_way_turnover_annualized": cluster_row[
                    "one_way_turnover_annualized"
                ],
                "delta_one_way_turnover_annualized": delta[
                    "delta_one_way_turnover_annualized"
                ],
            }
        )
    return pd.DataFrame(rows)


def _eligibility_summary(
    eligibility: Mapping[str, pd.DataFrame],
    dates: pd.DatetimeIndex,
    asset_class: pd.Series,
) -> pd.DataFrame:
    """Return eligible breadth by filter and retained sleeve."""
    rows = []
    for filter_id, cutoff in FILTERS:
        panel = eligibility[filter_id].reindex(index=dates).astype(bool)
        for sleeve in INCLUDED_CLASSES:
            counts = panel.loc[:, asset_class.eq(sleeve)].sum(axis=1)
            rows.append(
                {
                    "filter_id": filter_id,
                    "aum_cutoff_usd_millions": cutoff,
                    "aum_filter": _filter_label(filter_id),
                    "asset_class": sleeve,
                    "eligible_start": int(counts.iloc[0]),
                    "eligible_median": float(counts.median()),
                    "eligible_end": int(counts.iloc[-1]),
                }
            )
        total = panel.sum(axis=1)
        rows.append(
            {
                "filter_id": filter_id,
                "aum_cutoff_usd_millions": cutoff,
                "aum_filter": _filter_label(filter_id),
                "asset_class": "Total",
                "eligible_start": int(total.iloc[0]),
                "eligible_median": float(total.median()),
                "eligible_end": int(total.iloc[-1]),
            }
        )
    return pd.DataFrame(rows)


def run() -> Mapping[str, pd.DataFrame]:
    """Execute the Equity/Fixed-Income AUM grid and exact checks."""
    started = time.perf_counter()
    daily = funds._read_daily()
    asset_class = _metadata(daily.columns)
    dates = funds._dates()
    headline_dates = dates[
        (dates >= funds.HEADLINE_START) & (dates <= funds.HEADLINE_END)
    ]
    rolling_aum = aum._rolling_aum()
    eligibility_all = _restrict_eligibility(
        sensitivity._eligibilities(daily, dates, rolling_aum), asset_class
    )
    monthly_dates = funds._native_returns(daily, "ME").index
    monthly_eligibility = _restrict_eligibility(
        sensitivity._eligibilities(daily, monthly_dates, rolling_aum), asset_class
    )
    partitions, partition_diagnostics, cache_status = _partitions(
        daily, dates, eligibility_all
    )
    sleeve_panel = sleeves._sleeve_panel(headline_dates, asset_class)
    rank_prices = funds._performance_prices(daily).reindex(
        index=headline_dates, method="ffill"
    )
    weights = {}
    diagnostics_rows = []
    lookahead_errors = []
    missing_memberships = 0
    for filter_id, cutoff in FILTERS:
        eligibility = eligibility_all[filter_id].reindex(index=headline_dates).astype(bool)
        clusters = partitions[filter_id].reindex(index=headline_dates)
        missing_memberships += int((eligibility & clusters.isna()).to_numpy().sum())
        global_scores, cluster_scores, signal_diagnostics = sensitivity._signal_panels(
            daily,
            headline_dates,
            eligibility,
            monthly_eligibility[filter_id],
            clusters,
        )
        lookahead_errors.extend(
            [
                float(signal_diagnostics["max_global_lookahead_days"]),
                float(signal_diagnostics["max_cluster_lookahead_days"]),
            ]
        )
        global_weights, global_diagnostics = _long_short_weights(
            global_scores,
            rank_prices,
            eligibility,
            sleeve_panel,
        )
        cluster_weights, cluster_diagnostics = _long_short_weights(
            cluster_scores,
            rank_prices,
            eligibility,
            sleeve_panel,
        )
        global_diagnostics["independent_rank_reference_error"] = (
            _independent_rank_reference_error(
                global_scores,
                rank_prices,
                eligibility,
                sleeve_panel,
                global_weights,
            )
        )
        cluster_diagnostics["independent_rank_reference_error"] = (
            _independent_rank_reference_error(
                cluster_scores,
                rank_prices,
                eligibility,
                sleeve_panel,
                cluster_weights,
            )
        )
        weights[filter_id] = {
            "global": global_weights,
            "cluster": cluster_weights,
        }
        for method, diagnostic in (
            ("global", global_diagnostics),
            ("cluster", cluster_diagnostics),
        ):
            outside_eligibility = float(
                np.nanmax(
                    np.abs(
                        weights[filter_id][method]
                        .where(~eligibility, 0.0)
                        .to_numpy()
                    )
                )
            )
            diagnostics_rows.append(
                {
                    "filter_id": filter_id,
                    "aum_cutoff_usd_millions": cutoff,
                    "method": method,
                    **diagnostic,
                    "maximum_weight_outside_eligibility": outside_eligibility,
                }
            )
            if not np.isfinite(outside_eligibility):
                raise AssertionError("outside-eligibility diagnostic is not finite")

    performance = _performance(
        funds._performance_prices(daily), eligibility_all, weights
    )
    comparison = _comparison(performance)
    sensitivity_table = _filter_sensitivity(performance, comparison)
    diagnostics = pd.DataFrame(diagnostics_rows)
    error_columns = [column for column in diagnostics if column.endswith("_error")]
    maximum_weight_error = float(diagnostics[error_columns].abs().to_numpy().max())
    excluded_mask = asset_class.isin(EXCLUDED_CLASSES)
    excluded_eligible = sum(
        int(panel.loc[:, excluded_mask].to_numpy().sum())
        for panel in eligibility_all.values()
    )
    excluded_weight = max(
        float(frame.loc[:, excluded_mask].abs().to_numpy().max())
        for filter_weights in weights.values()
        for frame in filter_weights.values()
    )
    partition_count_error = int(
        partition_diagnostics["eligible_assets"]
        .sub(partition_diagnostics["partition_assets"])
        .abs()
        .max()
    )
    acceptance = pd.DataFrame(
        [
            {
                "check": "declared included official classes",
                "measured": "|".join(INCLUDED_CLASSES),
                "tolerance": "Equity|Fixed Income",
                "status": "PASS",
            },
            {
                "check": "excluded-class eligible observations",
                "measured": excluded_eligible,
                "tolerance": 0,
                "status": "PASS" if excluded_eligible == 0 else "FAIL",
            },
            {
                "check": "maximum excluded-class weight",
                "measured": excluded_weight,
                "tolerance": TOLERANCE,
                "status": "PASS" if excluded_weight <= TOLERANCE else "FAIL",
            },
            {
                "check": "partition eligible-member count error",
                "measured": partition_count_error,
                "tolerance": 0,
                "status": "PASS" if partition_count_error == 0 else "FAIL",
            },
            {
                "check": "eligible memberships missing from partitions",
                "measured": missing_memberships,
                "tolerance": 0,
                "status": "PASS" if missing_memberships == 0 else "FAIL",
            },
            {
                "check": "maximum weight, exposure, and sleeve-budget error",
                "measured": maximum_weight_error,
                "tolerance": TOLERANCE,
                "status": "PASS" if maximum_weight_error <= TOLERANCE else "FAIL",
            },
            {
                "check": "maximum signal lookahead days",
                "measured": max(lookahead_errors),
                "tolerance": 0,
                "status": "PASS" if max(lookahead_errors) <= 0 else "FAIL",
            },
            {
                "check": "performance rows",
                "measured": len(performance),
                "tolerance": len(FILTERS) * len(search.WINDOWS) * 2,
                "status": "PASS"
                if len(performance) == len(FILTERS) * len(search.WINDOWS) * 2
                else "FAIL",
            },
            {
                "check": "headline filter-sensitivity rows",
                "measured": len(sensitivity_table),
                "tolerance": len(FILTERS),
                "status": "PASS" if len(sensitivity_table) == len(FILTERS) else "FAIL",
            },
        ]
    )
    if not acceptance["status"].eq("PASS").all():
        raise AssertionError(acceptance.loc[~acceptance["status"].eq("PASS")])

    output = {
        "specification": pd.DataFrame(
            [
                {
                    "strategy": "long_short",
                    "included_official_classes": "|".join(INCLUDED_CLASSES),
                    "excluded_official_classes": "|".join(EXCLUDED_CLASSES),
                    "equity_budget_per_side": TARGET["Equity"],
                    "fixed_income_budget_per_side": TARGET["Fixed Income"],
                    "signal": "ROSAA production risk-adjusted momentum",
                    "ranking": "optimalportfolios top-quantile equal-weight by broad sleeve",
                    "cluster_role": "score standardisation only",
                    "frequency": FREQUENCY,
                    "span": SPAN,
                    "q": Q,
                    "schedule": SCHEDULE,
                    "cost_bps_one_way": COST_BPS,
                    "runner": RUNNER,
                }
            ]
        ),
        "eligibility_summary": _eligibility_summary(
            eligibility_all, headline_dates, asset_class
        ),
        "partition_diagnostics": partition_diagnostics,
        "weight_diagnostics": diagnostics,
        "performance": performance,
        "comparison_vs_global": comparison,
        "filter_sensitivity": sensitivity_table,
        "acceptance": acceptance,
        "runtime": pd.DataFrame(
            [
                {
                    "partition_cache_status": cache_status,
                    "runtime_seconds": time.perf_counter() - started,
                }
            ]
        ),
    }
    for name, frame in output.items():
        e5._write(frame, _root() / f"{name}.csv")
    return output


def _hash_outputs() -> dict[str, str]:
    """Hash every deterministic CSV artifact."""
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(_root().glob("*.csv"))
        if path.name not in {"runtime.csv", "determinism.csv"}
    }


def verify_determinism() -> pd.DataFrame:
    """Replay the cache-first grid and require byte-identical outputs."""
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
    """Run, replay, and print performance as a function of AUM cutoff."""
    replay = verify_determinism()
    sensitivity_table = pd.read_csv(
        _root() / "filter_sensitivity.csv", float_precision="round_trip"
    )
    print(sensitivity_table.to_string(index=False), flush=True)
    print(
        f"determinism: {int(replay['byte_identical'].sum())}/{len(replay)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
