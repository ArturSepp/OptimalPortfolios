"""Rank fund-level cluster-minus-global P&L for the U2 60/40 AUM grid.

The analysis consumes the class-restricted partitions and exact portfolio construction
from ``run_u2_equity_fi_long_short_aum_grid``. For each AUM cutoff and fund, it decomposes
cluster-minus-global P&L into long gross selection, short gross selection, and realized
transaction-cost effects under QIS's exact currency-NAV accounting convention.
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Mapping

import pandas as pd

import papers.cluster_lineage_2026.replication.run_backtests as e5
import papers.cluster_lineage_2026.replication.run_depc1_strategy_backtests as accounting
import papers.cluster_lineage_2026.replication.run_u2_blackrock_aum_filter as aum
import papers.cluster_lineage_2026.replication.run_u2_blackrock_aum_sensitivity as sensitivity
import papers.cluster_lineage_2026.replication.run_u2_blackrock_etf_grid as funds
import papers.cluster_lineage_2026.replication.run_u2_blackrock_long_short_search as search
import papers.cluster_lineage_2026.replication.run_u2_blackrock_sleeve_grid as sleeves
import papers.cluster_lineage_2026.replication.run_u2_equity_fi_long_short_aum_grid as grid


RUNNER = (
    "papers/cluster_lineage_2026/replication/"
    "run_u2_equity_fi_fund_pnl_attribution.py"
)
TOLERANCE = 1e-10


def _root() -> Path:
    """Return the isolated fund-attribution output directory."""
    root = grid._root() / "fund_pnl_attribution"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _metadata(columns: pd.Index) -> pd.DataFrame:
    """Return reader-facing fund metadata aligned to the return panel."""
    metadata = pd.read_csv(funds.METADATA_FILE).set_index("ticker").reindex(columns)
    required = ["name", "asset_class"]
    if metadata[required].isna().any().any():
        missing = metadata.index[metadata[required].isna().any(axis=1)].tolist()
        raise AssertionError(f"fund metadata is incomplete: {missing}")
    optional = ["sub_asset_class", "region", "investment_style"]
    metadata[optional] = metadata[optional].fillna("Unclassified")
    return metadata


def _weight_statistics(
    weights: pd.DataFrame,
    dates: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Return fund-level decision-weight diagnostics for one portfolio."""
    selected = weights.reindex(index=dates).fillna(0.0)
    return pd.DataFrame(
        {
            "asset": selected.columns,
            "average_signed_weight": selected.mean(axis=0).to_numpy(),
            "average_long_weight": selected.clip(lower=0.0).mean(axis=0).to_numpy(),
            "average_short_weight_abs": (
                -selected.clip(upper=0.0).mean(axis=0)
            ).to_numpy(),
            "long_selection_dates": selected.gt(0.0).sum(axis=0).to_numpy(),
            "short_selection_dates": selected.lt(0.0).sum(axis=0).to_numpy(),
        }
    )


def _attribution(
    portfolio,
    *,
    filter_id: str,
    cutoff: float,
    method: str,
    metadata: pd.DataFrame,
) -> tuple[pd.DataFrame, Mapping[str, object]]:
    """Return exact per-fund P&L normalized to beginning portfolio NAV."""
    frame, diagnostics = accounting._instrument_attribution(
        portfolio,
        method,
        "U2_Equity_Fixed_Income",
    )
    frame.insert(0, "filter_id", filter_id)
    frame.insert(1, "aum_cutoff_usd_millions", cutoff)
    frame["name"] = frame["asset"].map(metadata["name"])
    frame["asset_class"] = frame["asset"].map(metadata["asset_class"])
    frame["sub_asset_class"] = frame["asset"].map(metadata["sub_asset_class"])
    frame["region"] = frame["asset"].map(metadata["region"])
    frame["investment_style"] = frame["asset"].map(metadata["investment_style"])
    beginning_nav = float(diagnostics["beginning_nav"])
    columns = {
        "long_gross_pnl_currency": "long_gross_pnl_pct_of_start",
        "short_gross_pnl_currency": "short_gross_pnl_pct_of_start",
        "transaction_cost_currency": "transaction_cost_pct_of_start",
        "net_pnl_currency": "net_pnl_pct_of_start",
    }
    for source, target in columns.items():
        frame[target] = 100.0 * frame[source] / beginning_nav
    return frame, diagnostics


def _fund_deltas(
    attribution: pd.DataFrame,
    weight_statistics: pd.DataFrame,
) -> pd.DataFrame:
    """Build exact cluster-minus-global contribution rows and negative ranks."""
    metadata_columns = [
        "asset",
        "name",
        "asset_class",
        "sub_asset_class",
        "region",
        "investment_style",
    ]
    pnl_columns = [
        "long_gross_pnl_pct_of_start",
        "short_gross_pnl_pct_of_start",
        "transaction_cost_pct_of_start",
        "net_pnl_pct_of_start",
    ]
    rows = []
    for filter_id, cutoff in grid.FILTERS:
        panel = attribution.loc[attribution["filter_id"].eq(filter_id)]
        global_pnl = panel.loc[panel["leg"].eq("global")].set_index("asset")
        cluster_pnl = panel.loc[panel["leg"].eq("cluster")].set_index("asset")
        global_weights = weight_statistics.loc[
            weight_statistics["filter_id"].eq(filter_id)
            & weight_statistics["method"].eq("global")
        ].set_index("asset")
        cluster_weights = weight_statistics.loc[
            weight_statistics["filter_id"].eq(filter_id)
            & weight_statistics["method"].eq("cluster")
        ].set_index("asset")
        for asset in cluster_pnl.index:
            cluster = cluster_pnl.loc[asset]
            control = global_pnl.loc[asset]
            long_delta = (
                cluster["long_gross_pnl_pct_of_start"]
                - control["long_gross_pnl_pct_of_start"]
            )
            short_delta = (
                cluster["short_gross_pnl_pct_of_start"]
                - control["short_gross_pnl_pct_of_start"]
            )
            cost_effect = -(
                cluster["transaction_cost_pct_of_start"]
                - control["transaction_cost_pct_of_start"]
            )
            net_delta = (
                cluster["net_pnl_pct_of_start"]
                - control["net_pnl_pct_of_start"]
            )
            row = {
                "filter_id": filter_id,
                "aum_cutoff_usd_millions": cutoff,
                **{column: cluster[column] for column in metadata_columns[1:]},
                "asset": asset,
                "cluster_long_gross_pnl_pct_of_start": cluster[pnl_columns[0]],
                "global_long_gross_pnl_pct_of_start": control[pnl_columns[0]],
                "delta_long_gross_pnl_pct_of_start": long_delta,
                "cluster_short_gross_pnl_pct_of_start": cluster[pnl_columns[1]],
                "global_short_gross_pnl_pct_of_start": control[pnl_columns[1]],
                "delta_short_gross_pnl_pct_of_start": short_delta,
                "cluster_transaction_cost_pct_of_start": cluster[pnl_columns[2]],
                "global_transaction_cost_pct_of_start": control[pnl_columns[2]],
                "delta_cost_effect_pct_of_start": cost_effect,
                "cluster_net_pnl_pct_of_start": cluster[pnl_columns[3]],
                "global_net_pnl_pct_of_start": control[pnl_columns[3]],
                "delta_net_pnl_pct_of_start": net_delta,
                "component_reconciliation_error": net_delta
                - long_delta
                - short_delta
                - cost_effect,
            }
            for method, weights in (
                ("cluster", cluster_weights),
                ("global", global_weights),
            ):
                for column in weights.columns:
                    row[f"{method}_{column}"] = weights.loc[asset, column]
            rows.append(row)
    frame = pd.DataFrame(rows)
    frame = frame.loc[frame["asset_class"].isin(grid.INCLUDED_CLASSES)].copy()
    frame["negative_contribution_rank"] = frame.groupby("filter_id")[
        "delta_net_pnl_pct_of_start"
    ].rank(method="first", ascending=True).astype(int)
    totals = frame.groupby("filter_id")["delta_net_pnl_pct_of_start"].transform("sum")
    frame["share_of_total_cluster_gap"] = frame["delta_net_pnl_pct_of_start"] / totals
    return frame.sort_values(
        ["aum_cutoff_usd_millions", "delta_net_pnl_pct_of_start", "asset"]
    ).reset_index(drop=True)


def _persistent_table(fund_deltas: pd.DataFrame) -> pd.DataFrame:
    """Return funds ranked by mean contribution and number of negative filters."""
    index_columns = [
        "asset",
        "name",
        "asset_class",
        "sub_asset_class",
        "region",
        "investment_style",
    ]
    pivot = fund_deltas.pivot(
        index=index_columns,
        columns="filter_id",
        values="delta_net_pnl_pct_of_start",
    ).reset_index()
    filter_columns = [filter_id for filter_id, _ in grid.FILTERS]
    pivot["negative_filter_count"] = pivot[filter_columns].lt(0.0).sum(axis=1)
    pivot["mean_delta_net_pnl_pct_of_start"] = pivot[filter_columns].mean(axis=1)
    pivot["minimum_delta_net_pnl_pct_of_start"] = pivot[filter_columns].min(axis=1)
    pivot["persistent_negative"] = pivot["negative_filter_count"].eq(len(grid.FILTERS))
    return pivot.sort_values(
        ["mean_delta_net_pnl_pct_of_start", "asset"],
        ascending=[True, True],
    ).reset_index(drop=True)


def run() -> Mapping[str, pd.DataFrame]:
    """Execute exact fund attribution for all three AUM cutoffs."""
    started = time.perf_counter()
    daily = funds._read_daily()
    metadata = _metadata(daily.columns)
    asset_class = metadata["asset_class"]
    dates = funds._dates()
    headline_dates = dates[
        (dates >= funds.HEADLINE_START) & (dates <= funds.HEADLINE_END)
    ]
    rolling_aum = aum._rolling_aum()
    eligibility_all = grid._restrict_eligibility(
        sensitivity._eligibilities(daily, dates, rolling_aum), asset_class
    )
    monthly_dates = funds._native_returns(daily, "ME").index
    monthly_eligibility = grid._restrict_eligibility(
        sensitivity._eligibilities(daily, monthly_dates, rolling_aum), asset_class
    )
    partitions, _, cache_status = grid._partitions(daily, dates, eligibility_all)
    if cache_status != "hit":
        raise AssertionError("fund attribution must consume the completed partition cache")
    sleeve_panel = sleeves._sleeve_panel(headline_dates, asset_class)
    prices_all = funds._performance_prices(daily)
    rank_prices = prices_all.reindex(index=headline_dates, method="ffill")
    scheduled_dates = search._rebalance_dates(headline_dates, grid.SCHEDULE)

    attribution_frames = []
    weight_rows = []
    portfolio_rows = []
    accounting_errors = []
    excluded_weight_errors = []
    for filter_id, cutoff in grid.FILTERS:
        eligibility = eligibility_all[filter_id].reindex(index=headline_dates).astype(bool)
        clusters = partitions[filter_id].reindex(index=headline_dates)
        global_scores, cluster_scores, _ = sensitivity._signal_panels(
            daily,
            headline_dates,
            eligibility,
            monthly_eligibility[filter_id],
            clusters,
        )
        global_weights, _ = grid._long_short_weights(
            global_scores,
            rank_prices,
            eligibility,
            sleeve_panel,
        )
        cluster_weights, _ = grid._long_short_weights(
            cluster_scores,
            rank_prices,
            eligibility,
            sleeve_panel,
        )
        for method, weights in (
            ("global", global_weights),
            ("cluster", cluster_weights),
        ):
            filtered_window = sensitivity._window(
                prices_all,
                eligibility_all[filter_id],
                search.FULL_WINDOW,
                headline_dates,
            )
            net, _ = funds._backtest(
                filtered_window["prices"],
                weights.reindex(index=scheduled_dates),
                grid.COST_BPS / 10000.0,
                f"u2_eqfi_fund_attr_{filter_id}_{method}",
            )
            frame, diagnostics = _attribution(
                net,
                filter_id=filter_id,
                cutoff=cutoff,
                method=method,
                metadata=metadata,
            )
            attribution_frames.append(frame)
            accounting_errors.extend(
                [
                    float(diagnostics["max_step_reconciliation_abs_error"]),
                    float(diagnostics["cumulative_reconciliation_abs_error"]),
                ]
            )
            portfolio_rows.append(
                {
                    "filter_id": filter_id,
                    "aum_cutoff_usd_millions": cutoff,
                    "method": method,
                    **diagnostics,
                }
            )
            stats = _weight_statistics(weights, scheduled_dates)
            stats.insert(0, "filter_id", filter_id)
            stats.insert(1, "aum_cutoff_usd_millions", cutoff)
            stats.insert(2, "method", method)
            weight_rows.append(stats)
            excluded = asset_class.isin(grid.EXCLUDED_CLASSES)
            excluded_weight_errors.append(
                float(weights.loc[:, excluded].abs().to_numpy().max())
            )

    attribution = pd.concat(attribution_frames, ignore_index=True)
    weight_statistics = pd.concat(weight_rows, ignore_index=True)
    fund_deltas = _fund_deltas(attribution, weight_statistics)
    persistent = _persistent_table(fund_deltas)
    portfolio_summary = pd.DataFrame(portfolio_rows)
    reconciliation_errors = []
    performance = pd.read_csv(
        grid._root() / "performance.csv", float_precision="round_trip"
    )
    primary = performance.loc[
        performance["analysis_window"].eq(search.FULL_WINDOW)
    ].set_index(["filter_id", "method"])
    for filter_id, _ in grid.FILTERS:
        contribution_delta = fund_deltas.loc[
            fund_deltas["filter_id"].eq(filter_id),
            "delta_net_pnl_pct_of_start",
        ].sum()
        portfolio_delta = 100.0 * (
            float(primary.loc[(filter_id, "cluster"), "net_total_return"])
            - float(primary.loc[(filter_id, "global"), "net_total_return"])
        )
        reconciliation_errors.append(abs(contribution_delta - portfolio_delta))

    included_count = int(asset_class.isin(grid.INCLUDED_CLASSES).sum())
    maximum_component_error = float(
        fund_deltas["component_reconciliation_error"].abs().max()
    )
    acceptance = pd.DataFrame(
        [
            {
                "check": "partition cache status",
                "measured": cache_status,
                "tolerance": "hit",
                "status": "PASS" if cache_status == "hit" else "FAIL",
            },
            {
                "check": "maximum excluded-class weight",
                "measured": max(excluded_weight_errors),
                "tolerance": grid.TOLERANCE,
                "status": "PASS"
                if max(excluded_weight_errors) <= grid.TOLERANCE
                else "FAIL",
            },
            {
                "check": "instrument P&L accounting error",
                "measured": max(accounting_errors),
                "tolerance": TOLERANCE,
                "status": "PASS" if max(accounting_errors) <= TOLERANCE else "FAIL",
            },
            {
                "check": "fund component reconciliation error",
                "measured": maximum_component_error,
                "tolerance": TOLERANCE,
                "status": "PASS" if maximum_component_error <= TOLERANCE else "FAIL",
            },
            {
                "check": "fund total-to-portfolio delta error",
                "measured": max(reconciliation_errors),
                "tolerance": TOLERANCE,
                "status": "PASS"
                if max(reconciliation_errors) <= TOLERANCE
                else "FAIL",
            },
            {
                "check": "fund-delta rows",
                "measured": len(fund_deltas),
                "tolerance": included_count * len(grid.FILTERS),
                "status": "PASS"
                if len(fund_deltas) == included_count * len(grid.FILTERS)
                else "FAIL",
            },
            {
                "check": "persistent fund rows",
                "measured": len(persistent),
                "tolerance": included_count,
                "status": "PASS" if len(persistent) == included_count else "FAIL",
            },
        ]
    )
    if not acceptance["status"].eq("PASS").all():
        raise AssertionError(acceptance.loc[~acceptance["status"].eq("PASS")])

    output = {
        "portfolio_summary": portfolio_summary,
        "weight_statistics": weight_statistics.loc[
            weight_statistics["asset"].isin(metadata.index[asset_class.isin(grid.INCLUDED_CLASSES)])
        ].reset_index(drop=True),
        "instrument_pnl": attribution.loc[
            attribution["asset_class"].isin(grid.INCLUDED_CLASSES)
        ].reset_index(drop=True),
        "fund_delta_vs_global": fund_deltas,
        "persistent_negative_contributors": persistent,
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
    """Hash deterministic CSV artifacts."""
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(_root().glob("*.csv"))
        if path.name not in {"runtime.csv", "determinism.csv"}
    }


def verify_determinism() -> pd.DataFrame:
    """Replay the complete attribution and require byte-identical outputs."""
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
    """Run, replay, and print the ten worst contributors per AUM cutoff."""
    replay = verify_determinism()
    table = pd.read_csv(
        _root() / "fund_delta_vs_global.csv", float_precision="round_trip"
    )
    columns = [
        "filter_id",
        "negative_contribution_rank",
        "asset",
        "name",
        "asset_class",
        "delta_long_gross_pnl_pct_of_start",
        "delta_short_gross_pnl_pct_of_start",
        "delta_cost_effect_pct_of_start",
        "delta_net_pnl_pct_of_start",
    ]
    print(
        table.loc[table["negative_contribution_rank"].le(10), columns].to_string(
            index=False
        ),
        flush=True,
    )
    print(
        f"determinism: {int(replay['byte_identical'].sum())}/{len(replay)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
