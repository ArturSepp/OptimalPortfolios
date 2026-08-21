"""Diagnose the U1 sector-rank performance gap with matched-universe controls."""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import qis

import papers.cluster_lineage_2026.replication.run_backtests as e5
from papers.cluster_lineage_2026.replication.run_e5b import (
    _group_equal_from_ranks,
    _root as e5b_root,
)


UNIVERSE = e5.UniverseName.MSCI_US
Q = 0.20
WINDOW = "headline_20090831_20260630"


def _root() -> Path:
    """Return the local sector-gap diagnostic directory."""
    root = e5b_root() / "diagnostics" / "u1_sector_gap"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _global_weights(
    scores: pd.DataFrame, eligibility: pd.DataFrame
) -> pd.DataFrame:
    """Return asset-equal global-rank weights on the supplied eligible universe."""
    groups = pd.DataFrame("global", index=scores.index, columns=scores.columns)
    ranks = e5._rank_panel(scores.where(eligibility), groups)
    return e5._weights_from_ranks(ranks, eligibility, Q, UNIVERSE)


def _group_weights(
    scores: pd.DataFrame,
    eligibility: pd.DataFrame,
    groups: pd.DataFrame,
) -> pd.DataFrame:
    """Return group-equal weights after ranking on the supplied eligible universe."""
    ranks = e5._rank_panel(scores.where(eligibility), groups)
    weights, _, _ = _group_equal_from_ranks(
        ranks, eligibility, groups, Q, UNIVERSE
    )
    return weights


def _backtest(prices: pd.DataFrame, weights: pd.DataFrame, costs: float, ticker: str):
    """Run the accepted net and gross qis paths while silencing known cash warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        net = qis.backtest_model_portfolio(
            prices=prices,
            weights=weights,
            rebalancing_freq=None,
            rebalancing_costs=costs,
            weight_implementation_lag=1,
            ticker=ticker,
        )
        gross = qis.backtest_model_portfolio(
            prices=prices,
            weights=weights,
            rebalancing_freq=None,
            rebalancing_costs=None,
            weight_implementation_lag=1,
            ticker=f"{ticker}_gross",
        )
    return net, gross


def run() -> dict[str, pd.DataFrame]:
    """Run original, classified-only, and explicit-unclassified U1 controls."""
    data = e5.load_universe(UNIVERSE)
    dates = e5._analysis_windows(
        UNIVERSE, e5.load_cached(UNIVERSE, e5.SmootherName.BASELINE).dates
    )[WINDOW]
    eligibility = e5._investable_eligibility(data, dates)
    prices = e5._prices(data).reindex(columns=eligibility.columns)
    scores = e5._raw_momentum_scores(
        data, dates, vol_adjusted=False
    ).reindex(columns=eligibility.columns).where(eligibility)
    sector = e5._taxonomy_groups(data, eligibility.columns)
    sector_groups = pd.DataFrame(
        np.tile(sector.to_numpy(), (len(dates), 1)),
        index=dates,
        columns=eligibility.columns,
    )
    classified_mask = pd.DataFrame(
        np.tile(sector.notna().to_numpy(), (len(dates), 1)),
        index=dates,
        columns=eligibility.columns,
    )
    classified_eligibility = eligibility & classified_mask
    unclassified_groups = sector_groups.fillna("Unclassified")
    baseline_groups = e5._cluster_groups(
        UNIVERSE, e5.SmootherName.BASELINE
    ).reindex(index=dates, columns=eligibility.columns)
    smooth_groups = e5._cluster_groups(
        UNIVERSE, e5.SmootherName.M1_DELTA_002
    ).reindex(index=dates, columns=eligibility.columns)

    weights = {
        "global_all": _global_weights(scores, eligibility),
        "global_classified_only": _global_weights(
            scores, classified_eligibility
        ),
        "sector_drop_unclassified": _group_weights(
            scores, classified_eligibility, sector_groups
        ),
        "sector_explicit_unclassified": _group_weights(
            scores, eligibility, unclassified_groups
        ),
        "cluster_baseline_all": _group_weights(
            scores, eligibility, baseline_groups
        ),
        "cluster_baseline_classified_only": _group_weights(
            scores, classified_eligibility, baseline_groups
        ),
        "cluster_M1_002_all": _group_weights(
            scores, eligibility, smooth_groups
        ),
        "cluster_M1_002_classified_only": _group_weights(
            scores, classified_eligibility, smooth_groups
        ),
    }
    ew = eligibility.astype(float).div(
        eligibility.sum(axis=1).replace(0, np.nan), axis=0
    ).fillna(0.0)
    costs = e5.get_universe_spec(UNIVERSE).cost_bps / 10000.0
    ew_net, ew_gross = _backtest(prices, ew, costs, "EW_all_reference")
    benchmark_nav = ew_net.get_portfolio_nav()

    performance_rows = []
    missing_assets = sector.index[sector.isna()]
    weight_rows = []
    for leg, frame in weights.items():
        net, gross = _backtest(prices, frame, costs, leg)
        performance_rows.append(
            {
                "leg": leg,
                **e5._performance_row(net, gross, benchmark_nav),
            }
        )
        missing_weight = frame.reindex(columns=missing_assets, fill_value=0.0).sum(axis=1)
        weight_rows.append(
            {
                "leg": leg,
                "mean_target_weight_missing_sector": float(missing_weight.mean()),
                "median_target_weight_missing_sector": float(missing_weight.median()),
                "max_target_weight_missing_sector": float(missing_weight.max()),
                "first_target_weight_missing_sector": float(missing_weight.iloc[0]),
                "last_target_weight_missing_sector": float(missing_weight.iloc[-1]),
            }
        )

    eligible_count = eligibility.sum(axis=1)
    classified_count = classified_eligibility.sum(axis=1)
    coverage = pd.DataFrame(
        {
            "eligible_assets": eligible_count,
            "classified_assets": classified_count,
            "unclassified_assets": eligible_count - classified_count,
            "unclassified_share": 1.0 - classified_count / eligible_count,
        }
    ).rename_axis("date").reset_index()
    metadata = pd.DataFrame(
        [
            {
                "metadata_assets": len(sector),
                "missing_sector_assets": int(sector.isna().sum()),
                "missing_sector_share": float(sector.isna().mean()),
                "missing_ended_before_2026_share": float(
                    pd.to_datetime(
                        data.taxonomy.loc[sector.isna(), "last_constituent_date"]
                    ).lt("2026-01-01").mean()
                ),
                "classified_ended_before_2026_share": float(
                    pd.to_datetime(
                        data.taxonomy.loc[sector.notna(), "last_constituent_date"]
                    ).lt("2026-01-01").mean()
                ),
                "taxonomy_time_structure": "one static label per security",
            }
        ]
    )
    output = {
        "performance_controls": pd.DataFrame(performance_rows),
        "target_weight_controls": pd.DataFrame(weight_rows),
        "coverage_per_date": coverage,
        "metadata_diagnostic": metadata,
        "ew_reference": pd.DataFrame(
            [{"leg": "EW_all", **e5._performance_row(ew_net, ew_gross, benchmark_nav)}]
        ),
    }
    for name, frame in output.items():
        e5._write(frame, _root() / f"{name}.csv")
    return output


if __name__ == "__main__":
    result = run()
    print(result["performance_controls"].to_string(index=False))
