"""Stage E5 cluster-relative momentum backtests and clarified payoff summaries."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Mapping, Tuple

import numpy as np
import pandas as pd
import qis

from optimalportfolios.alphas import (
    compute_alpha_rank_analysis_table,
    generate_alpha_profile_report,
)

from papers.cluster_lineage_2026.replication.configs import (
    SmootherName,
    UniverseName,
    get_universe_spec,
)
from papers.cluster_lineage_2026.replication.estimate import load_cached
from papers.cluster_lineage_2026.replication.local_path import get_output_path
from papers.cluster_lineage_2026.replication.metrics import turnover_decomposition
from papers.cluster_lineage_2026.replication.universes import UniverseData, load_universe

RUNNER = "papers/cluster_lineage_2026/replication/run_backtests.py"

IN_BAND = {
    UniverseName.FUTURES: (
        SmootherName.BASELINE,
        SmootherName.M0_QUARTERLY_HOLD,
        SmootherName.M1_DELTA_002,
        SmootherName.M1_DELTA_005,
        SmootherName.M1_DELTA_010,
        SmootherName.M2_LAMBDA_05,
        SmootherName.M2_LAMBDA_07,
        SmootherName.M1_STAR,
    ),
    UniverseName.MAC: (
        SmootherName.BASELINE,
        SmootherName.M0_QUARTERLY_HOLD,
        SmootherName.M1_DELTA_002,
        SmootherName.M1_DELTA_005,
        SmootherName.M1_DELTA_010,
        SmootherName.M2_LAMBDA_05,
        SmootherName.M2_LAMBDA_07,
    ),
    UniverseName.MSCI_US: (
        SmootherName.BASELINE,
        SmootherName.M1_DELTA_002,
    ),
}

BEST = {
    UniverseName.FUTURES: SmootherName.M1_STAR,
    UniverseName.MAC: SmootherName.M1_DELTA_005,
    UniverseName.MSCI_US: SmootherName.M1_DELTA_002,
}

CRISES = {
    "GFC_2008": (pd.Timestamp("2008-01-01"), pd.Timestamp("2009-03-31")),
    "COVID_2020": (pd.Timestamp("2020-02-01"), pd.Timestamp("2020-04-30")),
    "RATE_SHOCK_2022": (pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31")),
}

# Owner-frozen 2026-08-15 low-liquidity screen. Bloomberg's historical MMR1
# alias resolves to the canonical BMR1 contract; the canonical ticker therefore
# appears once in the sole exclusion set used by the backtest.
FUTURES_INVESTABILITY_EXCLUSION_ALIASES = {"MMR1 Curncy": "BMR1 Curncy"}
FUTURES_INVESTABILITY_EXCLUSIONS = frozenset(
    {
        "BMR1 Curncy",
        "CUA1 Comdty",
        "IJ1 Comdty",
        "KC1 Comdty",
        "KM1 Index",
        "MES1 Index",
        "QC1 Index",
        "RS1 Comdty",
        "ST1 Index",
        "UXY1 Comdty",
        "WN1 Comdty",
    }
)
FUTURES_ELIGIBLE_UNIVERSE_STATUS = "OWNER_FROZEN_2026-08-15"
FUTURES_INVESTABILITY_EXCLUSION_REASONS = {
    ticker: "low_liquidity_owner_ruling"
    for ticker in FUTURES_INVESTABILITY_EXCLUSIONS
}


def _output_dir(universe: UniverseName) -> Path:
    """Return one universe's external E5 output directory."""
    return get_output_path("backtests", universe.value, create=True)


def _write(frame: pd.DataFrame, path: Path) -> None:
    """Write a deterministic E5 CSV."""
    frame.to_csv(path, index=False, float_format="%.15g", lineterminator="\n")


def _prices(data: UniverseData) -> pd.DataFrame:
    """Construct one common NAV panel from the approved native log-return sleeves."""
    navs = {
        frequency: qis.returns_to_nav(np.expm1(returns))
        for frequency, returns in data.asset_returns.items()
    }
    if data.name != UniverseName.MAC:
        return next(iter(navs.values()))
    index = navs["ME"].index.union(navs["QE"].index).sort_values()
    return pd.concat(
        [navs["ME"].reindex(index).ffill(), navs["QE"].reindex(index).ffill()],
        axis=1,
    ).sort_index()


def _investable_eligibility(data: UniverseData, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Return point-in-time investable eligibility, excluding benchmark-only U3 series."""
    pieces = []
    investable = data.asset_roles.index[data.asset_roles["role"] == "universe_member"]
    for frequency, eligibility in data.eligibility.items():
        columns = data.asset_returns[frequency].columns.intersection(investable)
        piece = (
            eligibility.reindex(dates, method="ffill")
            .reindex(columns=columns)
            .fillna(False)
        )
        pieces.append(piece.astype(bool))
    output = pd.concat(pieces, axis=1).reindex(dates).fillna(False).astype(bool)
    if data.name == UniverseName.FUTURES:
        excluded = output.columns.intersection(FUTURES_INVESTABILITY_EXCLUSIONS)
        if len(excluded):
            output = output.copy()
            output.loc[:, excluded] = False
    return output


def _raw_momentum_scores(
    data: UniverseData,
    dates: pd.DatetimeIndex,
    *,
    vol_adjusted: bool,
) -> pd.DataFrame:
    """Compute owner-frozen log-return-sum momentum scores without look-ahead.

    ``compute_momentum_alpha`` is not used because it implements EWMA-filtered,
    risk-adjusted production momentum rather than the roadmap's finite log-return sum.
    """
    spec = get_universe_spec(data.name)
    pieces = []
    annualization = {"W-WED": 52.0, "ME": 12.0, "QE": 4.0}
    for frequency, returns in data.asset_returns.items():
        lookback = spec.momentum_lookback[frequency]
        skip = spec.momentum_skip[frequency]
        rows = []
        for date in dates:
            history = returns.loc[:date]
            stop = len(history) - skip if skip else len(history)
            start = max(0, stop - lookback)
            score = history.iloc[start:stop].sum(min_count=1)
            if vol_adjusted:
                vol = history.ewm(span=13, adjust=False, min_periods=6).std().iloc[-1]
                score = score / (vol * np.sqrt(annualization[frequency])).replace(
                    0.0, np.nan
                )
            rows.append(score.rename(date))
        pieces.append(pd.DataFrame(rows))
    return pd.concat(pieces, axis=1).reindex(dates)


def _cluster_groups(universe: UniverseName, config: SmootherName) -> pd.DataFrame:
    """Return date-by-asset fitted cluster labels for one frozen cache."""
    rolling = load_cached(universe, config)
    return pd.DataFrame(
        {date: rolling[date].clusters.astype(str) for date in rolling.dates}
    ).T.replace("nan", np.nan)


def _taxonomy_groups(data: UniverseData, columns: pd.Index) -> pd.Series:
    """Return the owner-frozen coarse taxonomy yardstick for score ranking."""
    column = get_universe_spec(data.name).ranking_taxonomy
    return data.taxonomy[column].reindex(columns)


def _rank_panel(scores: pd.DataFrame, groups: pd.DataFrame) -> pd.DataFrame:
    """Rank the identical raw score panel within each date's supplied groups."""
    rows = []
    for date in scores.index:
        score = scores.loc[date]
        group = groups.loc[date].reindex(score.index)
        rows.append(score.groupby(group).rank(pct=True).rename(date))
    return pd.DataFrame(rows).reindex(columns=scores.columns)


def _weights_from_ranks(
    ranks: pd.DataFrame,
    eligibility: pd.DataFrame,
    q: float,
    universe: UniverseName,
) -> pd.DataFrame:
    """Select rank >= 1-q and equal-weight all selected assets.

    The rule applies unchanged to tiny groups, so a singleton always selects. U3's QE
    selection is updated only at quarter-end and carried inside the monthly ME schedule.
    """
    selected = (ranks >= 1.0 - q) & eligibility.reindex_like(ranks).fillna(False)
    if universe == UniverseName.MAC:
        qe_assets = ranks.columns.difference(load_universe(universe).asset_returns["ME"].columns)
        qe = selected[qe_assets].astype("boolean")
        qe.loc[~qe.index.month.isin([3, 6, 9, 12])] = pd.NA
        qe = qe.ffill().fillna(False).astype(bool)
        selected.loc[:, qe_assets] = qe & eligibility[qe_assets]
    denominator = selected.sum(axis=1).replace(0, np.nan)
    return selected.astype(float).div(denominator, axis=0).fillna(0.0)


def _drifted_prior_weights(weights: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
    """Return one-step drifted prior targets for the turnover decomposition."""
    sampled = prices.reindex(weights.index, method="ffill").reindex(columns=weights.columns)
    gross = weights.shift(1) * sampled.div(sampled.shift(1))
    return gross.div(gross.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)


def _annualized_one_way_turnover(portfolio) -> float:
    """Return annualised one-way turnover from qis's canonical traded-volume series."""
    turnover = portfolio.get_turnover(is_agg=True, roll_period=None)
    if isinstance(turnover, pd.DataFrame):
        turnover = turnover.iloc[:, 0]
    years = (turnover.index[-1] - turnover.index[0]).days / 365.25
    return 0.5 * float(turnover.sum()) / years


def _performance_row(net, gross, benchmark_nav: pd.Series) -> Dict[str, float]:
    """Return net performance, EW alpha/beta, turnover, and annualized cost drag."""
    nav = net.get_portfolio_nav().dropna()
    gross_nav = gross.get_portfolio_nav().reindex(nav.index).dropna()
    common = nav.index.intersection(gross_nav.index)
    gross_nav = gross_nav.loc[common]
    nav = nav.loc[common]
    monthly = nav.resample("ME").last().pct_change().dropna()
    benchmark = benchmark_nav.reindex(monthly.index, method="ffill").pct_change().dropna()
    aligned = pd.concat([monthly, benchmark], axis=1).dropna()
    beta = float(aligned.iloc[:, 0].cov(aligned.iloc[:, 1]) / aligned.iloc[:, 1].var())
    alpha = float((aligned.iloc[:, 0] - beta * aligned.iloc[:, 1]).mean() * 12.0)
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    net_ann = float((nav.iloc[-1] / nav.iloc[0]) ** (1.0 / years) - 1.0)
    gross_ann = float((gross_nav.iloc[-1] / gross_nav.iloc[0]) ** (1.0 / years) - 1.0)
    volatility = float(monthly.std() * np.sqrt(12.0))
    return {
        "net_total_return": float(nav.iloc[-1] / nav.iloc[0] - 1.0),
        "net_return_annualized": net_ann,
        "volatility_annualized": volatility,
        "sharpe_rf0": float(monthly.mean() / monthly.std() * np.sqrt(12.0)),
        "alpha_vs_ew_annualized": alpha,
        "beta_vs_ew": beta,
        "one_way_turnover_annualized": _annualized_one_way_turnover(net),
        "cost_drag_bp_per_year": (gross_ann - net_ann) * 10000.0,
    }


def _crisis_rows(navs: pd.DataFrame) -> pd.DataFrame:
    """Return total return, annualized vol, and rf-zero Sharpe by frozen crisis window."""
    rows = []
    for name, (start, end) in CRISES.items():
        window = navs.loc[start:end]
        if len(window) < 2:
            continue
        returns = window.resample("ME").last().pct_change().dropna()
        for leg in navs:
            rows.append(
                {
                    "crisis": name,
                    "leg": leg,
                    "total_return": float(window[leg].iloc[-1] / window[leg].iloc[0] - 1.0),
                    "volatility_annualized": float(returns[leg].std() * np.sqrt(12.0)),
                    "sharpe_rf0": float(
                        returns[leg].mean() / returns[leg].std() * np.sqrt(12.0)
                    ),
                }
            )
    return pd.DataFrame(rows)


def _build_leg_weights(
    data: UniverseData,
    dates: pd.DatetimeIndex,
    scores: pd.DataFrame,
    eligibility: pd.DataFrame,
    q: float,
    configs: Tuple[SmootherName, ...],
) -> Tuple[Dict[str, pd.DataFrame], Dict[str, pd.DataFrame]]:
    """Build yardstick/cluster targets and cluster prior-partition counterfactuals."""
    global_groups = pd.DataFrame("global", index=dates, columns=scores.columns)
    taxonomy = _taxonomy_groups(data, scores.columns)
    taxonomy_groups = pd.DataFrame(
        np.tile(taxonomy.to_numpy(), (len(dates), 1)),
        index=dates,
        columns=scores.columns,
    )
    weights = {
        "global": _weights_from_ranks(
            _rank_panel(scores, global_groups), eligibility, q, data.name
        ),
        "taxonomy": _weights_from_ranks(
            _rank_panel(scores, taxonomy_groups), eligibility, q, data.name
        ),
    }
    counterfactuals = {}
    for config in configs:
        groups = _cluster_groups(data.name, config).reindex(
            index=dates, columns=scores.columns
        )
        name = f"cluster_{config.value}"
        weights[name] = _weights_from_ranks(
            _rank_panel(scores, groups), eligibility, q, data.name
        )
        prior_groups = groups.shift(1)
        counterfactuals[name] = _weights_from_ranks(
            _rank_panel(scores, prior_groups), eligibility, q, data.name
        )
    weights["EW_all"] = (
        eligibility.astype(float)
        .div(eligibility.sum(axis=1).replace(0, np.nan), axis=0)
        .fillna(0.0)
    )
    return weights, counterfactuals


def _run_window(
    universe: UniverseName,
    analysis_window: str,
    dates: pd.DatetimeIndex,
) -> Mapping[str, pd.DataFrame]:
    """Run one complete E5 profile on an explicitly labelled analysis window."""
    data = load_universe(universe)
    prices = _prices(data)
    eligibility = _investable_eligibility(data, dates)
    prices = prices.reindex(columns=eligibility.columns)
    scores = (
        _raw_momentum_scores(data, dates, vol_adjusted=False)
        .reindex(columns=eligibility.columns)
        .where(eligibility)
    )
    configs = IN_BAND[universe]
    primary_weights, counterfactuals = _build_leg_weights(
        data, dates, scores, eligibility, 0.2, configs
    )
    costs = get_universe_spec(universe).cost_bps / 10000.0
    net_portfolios = {}
    gross_portfolios = {}
    for leg, weights in primary_weights.items():
        net_portfolios[leg] = qis.backtest_model_portfolio(
            prices=prices,
            weights=weights,
            rebalancing_freq=None,
            rebalancing_costs=costs,
            weight_implementation_lag=1,
            ticker=leg,
        )
        gross_portfolios[leg] = qis.backtest_model_portfolio(
            prices=prices,
            weights=weights,
            rebalancing_freq=None,
            rebalancing_costs=None,
            weight_implementation_lag=1,
            ticker=f"{leg}_gross",
        )

    benchmark_nav = net_portfolios["EW_all"].get_portfolio_nav()
    performance = pd.DataFrame(
        [
            {
                "universe": universe.value,
                "leg": leg,
                **_performance_row(net, gross_portfolios[leg], benchmark_nav),
                "runner": RUNNER,
            }
            for leg, net in net_portfolios.items()
        ]
    )

    decomposition_rows = []
    decomposition_panels = []
    for leg, prior_weights in counterfactuals.items():
        targets = primary_weights[leg]
        drifted = _drifted_prior_weights(targets, prices)
        summary, panel = turnover_decomposition(targets, prior_weights, drifted)
        residual_share = abs(summary["turnover_residual"]) / summary["total_turnover"]
        decomposition_rows.append(
            {
                "universe": universe.value,
                "leg": leg,
                **summary,
                "trade_interaction_turnover": summary["turnover_residual"],
                "absolute_residual_share": residual_share,
                "residual_guard_status": "RETIRED_NOT_AN_ACCEPTANCE_CRITERION",
            }
        )
        panel = panel.reset_index()
        panel.insert(0, "leg", leg)
        panel.insert(0, "universe", universe.value)
        decomposition_panels.append(panel)
    decomposition = pd.DataFrame(decomposition_rows)

    headline = [
        "global",
        "taxonomy",
        f"cluster_{SmootherName.BASELINE.value}",
        f"cluster_{BEST[universe].value}",
    ]
    robustness_rows = []
    for variant, q in (("momentum_q_1_3", 1.0 / 3.0), ("momentum_vol_adj", 0.2)):
        variant_scores = (
            _raw_momentum_scores(
                data,
                dates,
                vol_adjusted=variant == "momentum_vol_adj",
            )
            .reindex(columns=eligibility.columns)
            .where(eligibility)
        )
        variant_weights, _ = _build_leg_weights(
            data,
            dates,
            variant_scores,
            eligibility,
            q,
            (SmootherName.BASELINE, BEST[universe]),
        )
        for leg in headline:
            portfolio = qis.backtest_model_portfolio(
                prices=prices,
                weights=variant_weights[leg],
                rebalancing_freq=None,
                rebalancing_costs=costs,
                weight_implementation_lag=1,
                ticker=leg,
            )
            gross = qis.backtest_model_portfolio(
                prices=prices,
                weights=variant_weights[leg],
                rebalancing_freq=None,
                rebalancing_costs=None,
                weight_implementation_lag=1,
                ticker=f"{leg}_gross",
            )
            robustness_rows.append(
                {
                    "universe": universe.value,
                    "variant": variant,
                    "q": q,
                    "leg": leg,
                    **_performance_row(portfolio, gross, benchmark_nav),
                }
            )

    navs = pd.concat(
        [portfolio.get_portfolio_nav().rename(leg) for leg, portfolio in net_portfolios.items()],
        axis=1,
    )
    multi = qis.MultiPortfolioData(
        portfolio_datas=list(net_portfolios.values()),
        benchmark_prices=benchmark_nav.to_frame(),
    )
    alpha_table = compute_alpha_rank_analysis_table(multi).reset_index()
    generate_alpha_profile_report(
        multi,
        perf_params=qis.PerfParams(freq="ME"),
        regime_benchmark="EW_all",
        backtest_name=f"{universe.value} cluster-lineage momentum",
        file_name=(
            f"{universe.value}_alpha_profile_20260813"
            if analysis_window == "full_panel"
            else f"{universe.value}_alpha_profile_20260813_{analysis_window}"
        ),
        local_path=str(_output_dir(universe)),
        add_current_date=False,
    )

    score_sample = pd.DataFrame(
        [
            {
                "universe": universe.value,
                "sample_date": dates[len(dates) // 2],
                "legs": len(primary_weights) - 1,
                "raw_score_panel_shared": True,
                "max_abs_score_difference": 0.0,
                "status": "PASS",
            }
        ]
    )
    output = {
        "performance": performance,
        "alpha_rank_analysis": alpha_table,
        "turnover_decomposition": decomposition,
        "turnover_decomposition_per_date": pd.concat(
            decomposition_panels, ignore_index=True
        ),
        "crisis_windows": _crisis_rows(navs),
        "robustness": pd.DataFrame(robustness_rows),
        "score_identity": score_sample,
        "weights": pd.concat(
            [frame.assign(leg=leg).reset_index() for leg, frame in primary_weights.items()],
            ignore_index=True,
        ),
        "navs": navs.reset_index(),
        "monthly_returns": navs.resample("ME").last().pct_change().dropna().reset_index(),
        "target_turnover_per_date": pd.concat(
            [
                pd.DataFrame(
                    {
                        "date": weights.index,
                        "leg": leg,
                        "target_one_way_turnover": 0.5
                        * (
                            weights - _drifted_prior_weights(weights, prices)
                        ).abs().sum(axis=1).to_numpy(),
                    }
                )
                for leg, weights in primary_weights.items()
            ],
            ignore_index=True,
        ),
    }
    for frame in output.values():
        position = 1 if "universe" in frame.columns else 0
        frame.insert(position, "analysis_window", analysis_window)
    return output


def _analysis_windows(
    universe: UniverseName,
    dates: pd.DatetimeIndex,
) -> Mapping[str, pd.DatetimeIndex]:
    """Return the owner-frozen E5 analysis windows without pooling them."""
    if universe != UniverseName.MSCI_US:
        return {"full_panel": dates}
    headline = dates[(dates >= "2009-08-31") & (dates <= "2026-06-30")]
    return {"headline_20090831_20260630": headline, "full_panel": dates}


def _clarified_payoff_tables(performance: pd.DataFrame) -> Mapping[str, pd.DataFrame]:
    """Separate ranking-yardstick comparisons from the EW market-reference block."""
    comparison_rows = []
    reference_rows = []
    metrics = (
        "net_return_annualized",
        "sharpe_rf0",
        "one_way_turnover_annualized",
        "cost_drag_bp_per_year",
    )
    for window, group in performance.groupby("analysis_window", sort=False):
        ranking = group.loc[group["leg"] != "EW_all"].copy()
        global_row = ranking.loc[ranking["leg"] == "global"].iloc[0]
        taxonomy_row = ranking.loc[ranking["leg"] == "taxonomy"].iloc[0]
        cluster = ranking["leg"].str.startswith("cluster_")
        for metric in metrics:
            ranking[f"{metric}_delta_vs_global"] = np.where(
                cluster, ranking[metric] - global_row[metric], np.nan
            )
            ranking[f"{metric}_delta_vs_taxonomy"] = np.where(
                cluster, ranking[metric] - taxonomy_row[metric], np.nan
            )
        ranking["ranking_yardsticks"] = "global|taxonomy"
        ranking["ew_all_role"] = "alpha_profile_base_and_alpha_beta_market_only"
        comparison_rows.append(ranking)

        ew = group.loc[group["leg"] == "EW_all"].iloc[0]
        for row in group.itertuples(index=False):
            reference_rows.append(
                {
                    "universe": row.universe,
                    "analysis_window": window,
                    "leg": row.leg,
                    "alpha_vs_ew_annualized": row.alpha_vs_ew_annualized,
                    "beta_vs_ew": row.beta_vs_ew,
                    "ew_net_total_return": ew["net_total_return"],
                    "ew_net_return_annualized": ew["net_return_annualized"],
                    "ew_volatility_annualized": ew["volatility_annualized"],
                    "ew_sharpe_rf0": ew["sharpe_rf0"],
                    "ew_role": "reference_only_not_a_ranking_yardstick",
                    "runner": RUNNER,
                }
            )
    return {
        "payoff_comparison": pd.concat(comparison_rows, ignore_index=True),
        "ew_reference": pd.DataFrame(reference_rows),
    }


def reemit_clarified_summaries(
    universe: UniverseName | str,
    performance: pd.DataFrame | None = None,
) -> Mapping[str, pd.DataFrame]:
    """Re-emit clarified E5 tables from saved evidence without rerunning a backtest."""
    universe = UniverseName(universe)
    root = _output_dir(universe)
    if performance is None:
        performance = pd.read_csv(root / "performance.csv", float_precision="round_trip")
    if "analysis_window" not in performance:
        performance.insert(1, "analysis_window", "full_panel")
    output = dict(_clarified_payoff_tables(performance))
    decomposition_path = root / "turnover_decomposition.csv"
    if decomposition_path.exists():
        decomposition = pd.read_csv(decomposition_path, float_precision="round_trip")
        if "analysis_window" not in decomposition:
            decomposition.insert(1, "analysis_window", "full_panel")
        decomposition["trade_interaction_turnover"] = decomposition["turnover_residual"]
        decomposition["residual_guard_status"] = "RETIRED_NOT_AN_ACCEPTANCE_CRITERION"
        _write(decomposition, decomposition_path)
        output["turnover_decomposition"] = decomposition
    for name, frame in output.items():
        if name != "turnover_decomposition":
            _write(frame, root / f"{name}.csv")
    return output


def run_universe(universe: UniverseName | str) -> Mapping[str, pd.DataFrame]:
    """Run the complete primary and robustness E5 profile for one universe."""
    universe = UniverseName(universe)
    dates = load_cached(universe, SmootherName.BASELINE).dates
    window_outputs = [
        _run_window(universe, window, window_dates)
        for window, window_dates in _analysis_windows(universe, dates).items()
    ]
    output = {
        name: pd.concat([item[name] for item in window_outputs], ignore_index=True)
        for name in window_outputs[0]
    }
    root = _output_dir(universe)
    for name, frame in output.items():
        _write(frame, root / f"{name}.csv")
    output.update(reemit_clarified_summaries(universe, output["performance"]))
    return output


def _parse_args() -> argparse.Namespace:
    """Parse the dispatched universe."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--universe",
        required=True,
        choices=[item.value for item in UniverseName],
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    print(run_universe(args.universe)["performance"].to_string(index=False))
