"""Trace EPOL through the primary U2 fund clusters, ranks, and positions.

The primary experiment applies a point-in-time USD 100m AUM filter and ranks the ROSAA
risk-adjusted momentum score once across the eligible Equity sleeve. Cluster membership
affects only production score standardisation; it does not define a ranking or capital-budget
group. Cluster integers are date-local, so this diagnostic records the actual peer set rather
than treating the raw integer as a persistent identity. Target weights are decision-date
weights and are implemented with the frozen one-period lag.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

import papers.cluster_lineage_2026.replication.run_backtests as e5
import papers.cluster_lineage_2026.replication.run_u2_blackrock_aum_filter as aum
import papers.cluster_lineage_2026.replication.run_u2_blackrock_aum_sensitivity as sensitivity
import papers.cluster_lineage_2026.replication.run_u2_blackrock_etf_grid as funds
import papers.cluster_lineage_2026.replication.run_u2_blackrock_long_short_search as search
import papers.cluster_lineage_2026.replication.run_u2_blackrock_sleeve_grid as sleeves
import papers.cluster_lineage_2026.replication.run_u2_equity_fi_long_short_aum_grid as grid


ASSET = "EPOL"
FILTER_ID = "aum_100m"
TOLERANCE = 1e-12


def _root() -> Path:
    """Return the isolated EPOL diagnostic directory."""
    root = Path(__file__).resolve().parents[1] / "data" / "diagnostics" / "epol_history"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _position(weight: float) -> str:
    """Return the signed position label for one target weight."""
    if weight > TOLERANCE:
        return "long"
    if weight < -TOLERANCE:
        return "short"
    return "flat"


def _ordinal(values: pd.Series, asset: str, *, ascending: bool) -> float:
    """Return the asset's one-based ordinal rank among finite values."""
    finite = values.dropna()
    if asset not in finite.index:
        return np.nan
    return float(finite.rank(method="average", ascending=ascending).loc[asset])


def _metadata() -> pd.DataFrame:
    """Return reader-facing fund metadata indexed by ticker."""
    metadata = pd.read_csv(funds.METADATA_FILE).set_index("ticker")
    required = ["name", "asset_class", "sub_asset_class", "region", "country"]
    if metadata.loc[ASSET, required].isna().any():
        raise AssertionError(f"{ASSET} metadata is incomplete")
    return metadata


def _summary(history: pd.DataFrame, cache_status: str) -> pd.DataFrame:
    """Return one compact overview of EPOL eligibility and portfolio treatment."""
    eligible = history.loc[history["eligible"]]
    cluster_long = eligible.loc[eligible["cluster_position"].eq("long")]
    cluster_short = eligible.loc[eligible["cluster_position"].eq("short")]
    global_long = eligible.loc[eligible["global_position"].eq("long")]
    global_short = eligible.loc[eligible["global_position"].eq("short")]
    return pd.DataFrame(
        [
            {
                "asset": ASSET,
                "filter_id": FILTER_ID,
                "partition_cache_status": cache_status,
                "decision_dates": len(history),
                "return_history_eligible_dates": int(history["history_eligible"].sum()),
                "aum_over_100m_dates": int(history["aum_over_100m"].sum()),
                "strategy_eligible_dates": len(eligible),
                "first_strategy_eligible": eligible["date"].min(),
                "last_strategy_eligible": eligible["date"].max(),
                "distinct_correlation_peer_sets": int(
                    eligible["correlation_cluster_members"].nunique()
                ),
                "correlation_cluster_size_mean": eligible[
                    "correlation_cluster_size"
                ].mean(),
                "correlation_cluster_size_median": eligible[
                    "correlation_cluster_size"
                ].median(),
                "correlation_cluster_size_min": eligible[
                    "correlation_cluster_size"
                ].min(),
                "correlation_cluster_size_max": eligible[
                    "correlation_cluster_size"
                ].max(),
                "cluster_long_dates": len(cluster_long),
                "cluster_short_dates": len(cluster_short),
                "cluster_flat_dates": int(eligible["cluster_position"].eq("flat").sum()),
                "cluster_overlap_removed_dates": int(
                    eligible["cluster_selected_both_before_overlap"].sum()
                ),
                "cluster_average_weight_when_long": cluster_long[
                    "cluster_target_weight"
                ].mean(),
                "cluster_average_weight_when_short_abs": -cluster_short[
                    "cluster_target_weight"
                ].mean(),
                "global_long_dates": len(global_long),
                "global_short_dates": len(global_short),
                "global_flat_dates": int(eligible["global_position"].eq("flat").sum()),
                "global_average_weight_when_long": global_long[
                    "global_target_weight"
                ].mean(),
                "global_average_weight_when_short_abs": -global_short[
                    "global_target_weight"
                ].mean(),
            }
        ]
    )


def _peer_table(
    history: pd.DataFrame,
    metadata: pd.DataFrame,
    peer_counters: Mapping[str, Counter],
) -> pd.DataFrame:
    """Return EPOL peer frequencies overall and by EPOL position."""
    eligible_dates = int(history["eligible"].sum())
    rows = []
    for peer, count in peer_counters["all"].most_common():
        item = metadata.loc[peer]
        rows.append(
            {
                "peer": peer,
                "name": item["name"],
                "asset_class": item["asset_class"],
                "sub_asset_class": item["sub_asset_class"],
                "region": item["region"],
                "country": item["country"],
                "co_cluster_decision_dates": count,
                "share_of_epol_eligible_dates": count / eligible_dates,
                "co_cluster_when_epol_long": peer_counters["long"][peer],
                "co_cluster_when_epol_short": peer_counters["short"][peer],
                "co_cluster_when_epol_flat": peer_counters["flat"][peer],
            }
        )
    return pd.DataFrame(rows)


def run() -> Mapping[str, pd.DataFrame]:
    """Execute the exact cached primary-specification EPOL diagnostic."""
    daily = funds._read_daily()
    metadata = _metadata()
    asset_class = metadata["asset_class"]
    all_dates = funds._dates()
    headline_dates = all_dates[
        (all_dates >= funds.HEADLINE_START) & (all_dates <= funds.HEADLINE_END)
    ]
    decision_dates = search._rebalance_dates(headline_dates, grid.SCHEDULE)
    rolling_aum = aum._rolling_aum()
    eligibility_all = grid._restrict_eligibility(
        sensitivity._eligibilities(daily, all_dates, rolling_aum), asset_class
    )
    monthly_dates = funds._native_returns(daily, "ME").index
    monthly_eligibility = grid._restrict_eligibility(
        sensitivity._eligibilities(daily, monthly_dates, rolling_aum), asset_class
    )
    partitions, _, cache_status = grid._partitions(daily, all_dates, eligibility_all)
    if cache_status != "hit":
        raise AssertionError("EPOL diagnostic must consume the completed partition cache")

    eligibility = eligibility_all[FILTER_ID].reindex(index=headline_dates).astype(bool)
    history_eligibility = funds._eligibility_for_dates(daily, headline_dates)
    point_in_time_aum = aum._aum_for_dates(headline_dates, rolling_aum)
    clusters = partitions[FILTER_ID].reindex(index=headline_dates)
    sleeve_panel = sleeves._sleeve_panel(headline_dates, asset_class)
    rank_prices = funds._performance_prices(daily).reindex(
        index=headline_dates, method="ffill"
    )
    global_scores, cluster_scores, _ = sensitivity._signal_panels(
        daily,
        headline_dates,
        eligibility,
        monthly_eligibility[FILTER_ID],
        clusters,
    )
    cluster_long_percentile = e5._rank_panel(cluster_scores, sleeve_panel)
    cluster_short_percentile = e5._rank_panel(-cluster_scores, sleeve_panel)
    global_long_percentile = e5._rank_panel(global_scores, sleeve_panel)
    global_short_percentile = e5._rank_panel(-global_scores, sleeve_panel)
    cluster_weights, _ = grid._long_short_weights(
        cluster_scores,
        rank_prices,
        eligibility,
        sleeve_panel,
    )
    global_weights, _ = grid._long_short_weights(
        global_scores,
        rank_prices,
        eligibility,
        sleeve_panel,
    )

    peers = {key: Counter() for key in ("all", "long", "short", "flat")}
    rows = []
    for date in decision_dates:
        is_eligible = bool(eligibility.at[date, ASSET])
        label = clusters.at[date, ASSET]
        raw_members = pd.Index([])
        if is_eligible and pd.notna(label):
            raw_members = clusters.columns[clusters.loc[date].eq(label)]

        cluster_weight = float(cluster_weights.at[date, ASSET])
        global_weight = float(global_weights.at[date, ASSET])
        cluster_position = _position(cluster_weight)
        global_position = _position(global_weight)
        strategy_peers = raw_members.drop(ASSET, errors="ignore")
        if is_eligible:
            peers["all"].update(strategy_peers)
            peers[cluster_position].update(strategy_peers)

        equity_members = eligibility.columns[
            eligibility.loc[date] & sleeve_panel.loc[date].eq("Equity")
        ]
        cluster_values = cluster_scores.loc[date, equity_members]
        global_values = global_scores.loc[date, equity_members]
        cluster_long_rule = cluster_weight > TOLERANCE
        cluster_short_rule = cluster_weight < -TOLERANCE
        global_long_rule = global_weight > TOLERANCE
        global_short_rule = global_weight < -TOLERANCE

        rows.append(
            {
                "date": date,
                "history_eligible": bool(history_eligibility.at[date, ASSET]),
                "rolling_12m_aum_usd_millions": point_in_time_aum.at[date, ASSET],
                "aum_over_100m": bool(point_in_time_aum.at[date, ASSET] > 100.0)
                if pd.notna(point_in_time_aum.at[date, ASSET])
                else False,
                "eligible": is_eligible,
                "correlation_cluster_id_date_local": label,
                "correlation_cluster_size": len(raw_members),
                "ranking_universe_size": len(equity_members),
                "correlation_cluster_members": "|".join(raw_members),
                "ranking_universe": "Equity",
                "cluster_score": cluster_scores.at[date, ASSET],
                "cluster_long_percentile": cluster_long_percentile.at[date, ASSET],
                "cluster_short_percentile": cluster_short_percentile.at[date, ASSET],
                "cluster_rank_high_1_is_best": _ordinal(
                    cluster_values, ASSET, ascending=False
                ),
                "cluster_rank_low_1_is_worst": _ordinal(
                    cluster_values, ASSET, ascending=True
                ),
                "cluster_selected_long_rule": cluster_long_rule,
                "cluster_selected_short_rule": cluster_short_rule,
                "cluster_selected_both_before_overlap": (
                    cluster_long_rule and cluster_short_rule
                ),
                "cluster_position": cluster_position,
                "cluster_target_weight": cluster_weight,
                "global_score": global_scores.at[date, ASSET],
                "global_equity_group_size": len(equity_members),
                "global_long_percentile": global_long_percentile.at[date, ASSET],
                "global_short_percentile": global_short_percentile.at[date, ASSET],
                "global_rank_high_1_is_best": _ordinal(
                    global_values, ASSET, ascending=False
                ),
                "global_rank_low_1_is_worst": _ordinal(
                    global_values, ASSET, ascending=True
                ),
                "global_selected_long_rule": global_long_rule,
                "global_selected_short_rule": global_short_rule,
                "global_position": global_position,
                "global_target_weight": global_weight,
                "implementation": "next_W-WED_mark_lag_1",
            }
        )

    history = pd.DataFrame(rows)
    peer_table = _peer_table(history, metadata, peers)
    summary = _summary(history, cache_status)
    delta_path = grid._root() / "fund_pnl_attribution" / "fund_delta_vs_global.csv"
    pnl = pd.read_csv(delta_path, float_precision="round_trip")
    pnl = pnl.loc[
        pnl["filter_id"].eq(FILTER_ID) & pnl["asset"].eq(ASSET)
    ].reset_index(drop=True)
    outside_weight = float(
        history.loc[
            ~history["eligible"], ["cluster_target_weight", "global_target_weight"]
        ]
        .abs()
        .to_numpy()
        .max()
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
                "check": "decision rows",
                "measured": len(history),
                "tolerance": len(decision_dates),
                "status": "PASS" if len(history) == len(decision_dates) else "FAIL",
            },
            {
                "check": "weight outside eligibility",
                "measured": outside_weight,
                "tolerance": TOLERANCE,
                "status": "PASS" if outside_weight <= TOLERANCE else "FAIL",
            },
            {
                "check": "fund PnL attribution rows",
                "measured": len(pnl),
                "tolerance": 1,
                "status": "PASS" if len(pnl) == 1 else "FAIL",
            },
        ]
    )
    if not acceptance["status"].eq("PASS").all():
        raise AssertionError(acceptance.loc[~acceptance["status"].eq("PASS")])

    output = {
        "epol_summary": summary,
        "epol_decision_history": history,
        "epol_correlation_cluster_peers": peer_table,
        "epol_pnl_attribution": pnl,
        "acceptance": acceptance,
    }
    for name, frame in output.items():
        e5._write(frame, _root() / f"{name}.csv")
    return output


def main() -> None:
    """Run the diagnostic and print its compact summary and leading peers."""
    output = run()
    print(output["epol_summary"].to_string(index=False), flush=True)
    print(
        output["epol_correlation_cluster_peers"].head(15).to_string(index=False),
        flush=True,
    )
    print(output["epol_pnl_attribution"].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
