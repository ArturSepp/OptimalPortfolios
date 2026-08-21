"""Rank the global Commodities futures book by exact net P&L contribution.

The portfolio is the accepted standalone global Commodities long-short book: q=0.20,
production 48-week-minus-4-week momentum, monthly decisions, one-observation lag, and
20 bp transaction costs over the corrected U1 calendar window.  Attribution is done in
currency P&L, not by summing unlinked daily return contributions.  For every instrument
and date it is prior units times the price change less that instrument's realised cost.
Consequently, instrument contributions sum exactly to the portfolio NAV change; dividing
by beginning NAV expresses each contribution as percentage points of total net return.

The generic ``PortfolioData.get_instruments_pnl(is_net=True)`` accessor is deliberately
not used because its cost accessor applies the module's default 260-observation rolling
aggregation.  That convention is suitable for rolling cost reports, not a daily P&L
identity.  The stored qis units, prices, costs, and NAV provide the exact accounting path.
"""
from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

import papers.cluster_lineage_2026.replication.run_backtests as e5
import papers.cluster_lineage_2026.replication.run_futures_asset_class_long_short as asset
import papers.cluster_lineage_2026.replication.run_futures_sleeve_grid as equal
import papers.cluster_lineage_2026.replication.run_futures_weight_30303010_u1_window as matched


ASSET_CLASS = "Commodities"
METHOD = "global"
Q = equal.PRIMARY_Q
ACCOUNTING_TOLERANCE = 1e-10
RUNNER = (
    "papers/cluster_lineage_2026/replication/"
    "run_futures_commodity_pnl_attribution.py"
)


def _root() -> Path:
    """Return and create the external commodity-attribution output directory."""
    root = asset._root() / "commodity_global_q020_attribution"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _net_currency_pnl(
    portfolio,
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> tuple[pd.DataFrame, dict[str, float | int | pd.Timestamp]]:
    """Return exact instrument net currency P&L and NAV reconciliation diagnostics."""
    nav = portfolio.nav.loc[(portfolio.nav.index >= start) & (portfolio.nav.index <= end)]
    nav = nav.dropna()
    if nav.empty or nav.index.min() != start or nav.index.max() != end:
        raise AssertionError("NAV does not span the requested attribution interval")

    prior_units = portfolio.units.shift(1)
    price_changes = portfolio.prices.diff()
    gross_currency = prior_units.multiply(price_changes)
    costs = portfolio.realized_costs.reindex_like(gross_currency).fillna(0.0)
    selected = (gross_currency.index > start) & (gross_currency.index <= end)
    active_missing = (
        prior_units.loc[selected].ne(0.0) & price_changes.loc[selected].isna()
    )
    if active_missing.to_numpy().any():
        locations = active_missing.stack().loc[lambda values: values].index.tolist()[:5]
        raise AssertionError(f"active futures positions have missing price changes: {locations}")
    gross_currency = gross_currency.loc[selected].fillna(0.0)
    costs = costs.loc[selected]
    net_currency = gross_currency.subtract(costs)

    nav_changes = nav.diff().dropna().reindex(net_currency.index)
    if nav_changes.isna().any():
        raise AssertionError("P&L dates do not align with the bounded NAV")
    step_error = net_currency.sum(axis=1).subtract(nav_changes)
    cumulative_error = float(
        abs(net_currency.to_numpy().sum() - (nav.iloc[-1] - nav.iloc[0]))
    )
    diagnostics: dict[str, float | int | pd.Timestamp] = {
        "nav_start": nav.index.min(),
        "nav_end": nav.index.max(),
        "pnl_rows": len(net_currency),
        "beginning_nav": float(nav.iloc[0]),
        "ending_nav": float(nav.iloc[-1]),
        "portfolio_net_pnl_currency": float(nav.iloc[-1] - nav.iloc[0]),
        "attributed_net_pnl_currency": float(net_currency.to_numpy().sum()),
        "portfolio_net_total_return": float(nav.iloc[-1] / nav.iloc[0] - 1.0),
        "attributed_net_total_return": float(
            net_currency.to_numpy().sum() / nav.iloc[0]
        ),
        "max_step_reconciliation_abs_error": float(step_error.abs().max()),
        "cumulative_reconciliation_abs_error": cumulative_error,
    }
    return net_currency, diagnostics


def _rank_contributions(
    *,
    net_currency: pd.DataFrame,
    costs: pd.DataFrame,
    beginning_nav: float,
    taxonomy: pd.DataFrame,
    sleeves: pd.Series,
    eligible_tickers: pd.Index,
    weights: pd.DataFrame,
) -> pd.DataFrame:
    """Create the descending commodity-contract contribution ranking."""
    tickers = sleeves.index[
        sleeves.eq(ASSET_CLASS) & sleeves.index.isin(eligible_tickers)
    ]
    net_contribution = net_currency.reindex(columns=tickers).sum(axis=0)
    cost_contribution = costs.reindex(
        index=net_currency.index, columns=tickers, fill_value=0.0
    ).sum(axis=0)
    gross_contribution = net_contribution.add(cost_contribution)
    metadata = taxonomy.reindex(tickers)
    names = metadata["name"].where(metadata["name"].notna(), tickers.to_series())

    ranking = pd.DataFrame(
        {
            "ticker": tickers,
            "name": names.astype(str).str.replace("_", " ", regex=False).to_numpy(),
            "commodity_subclass": metadata["asset_class"].to_numpy(),
            "exchange": metadata["exchange"].to_numpy(),
            "exchange_symbol": metadata["exch_symbol"].to_numpy(),
            "gross_pnl_currency": gross_contribution.to_numpy(),
            "transaction_cost_currency": cost_contribution.to_numpy(),
            "net_pnl_currency": net_contribution.to_numpy(),
            "gross_pnl_pct_of_start": 100.0
            * gross_contribution.to_numpy()
            / beginning_nav,
            "cost_pct_of_start": 100.0
            * cost_contribution.to_numpy()
            / beginning_nav,
            "net_pnl_pct_of_start": 100.0
            * net_contribution.to_numpy()
            / beginning_nav,
            "long_decision_count": weights.reindex(columns=tickers).gt(0.0).sum().to_numpy(),
            "short_decision_count": weights.reindex(columns=tickers).lt(0.0).sum().to_numpy(),
        }
    )
    total = float(net_contribution.sum())
    ranking["share_of_net_pnl_pct"] = 100.0 * ranking["net_pnl_currency"] / total
    ranking = ranking.sort_values(
        ["net_pnl_currency", "ticker"], ascending=[False, True]
    ).reset_index(drop=True)
    ranking.insert(0, "rank", range(1, len(ranking) + 1))
    return ranking


def _availability_diagnostics(
    *,
    data,
    dates: pd.DatetimeIndex,
    eligibility: pd.DataFrame,
    raw_scores: pd.DataFrame,
    sleeves: pd.Series,
    weights: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Report the changing eligible cross-section and per-contract entry dates."""
    valid = eligibility & raw_scores.notna()
    spec = e5.get_universe_spec(equal.UNIVERSE)
    frequency = spec.asset_frequencies[0]
    lookback = spec.momentum_lookback[frequency]
    skip = spec.momentum_skip[frequency]
    returns = data.asset_returns[frequency].reindex(columns=eligibility.columns)
    observation_rows = []
    for date in dates:
        history = returns.loc[:date]
        stop = len(history) - skip if skip else len(history)
        start = max(0, stop - lookback)
        observation_rows.append(history.iloc[start:stop].notna().sum().rename(date))
    momentum_observations = pd.DataFrame(observation_rows).reindex_like(eligibility)
    full_history = eligibility & momentum_observations.ge(lookback)
    commodity_columns = sleeves.index[sleeves.eq(ASSET_CLASS)]
    commodity_eligibility = eligibility.reindex(columns=commodity_columns)
    commodity_valid = valid.reindex(columns=commodity_columns)
    commodity_full_history = full_history.reindex(columns=commodity_columns)
    per_date = pd.DataFrame(
        {
            "date": dates,
            "eligible_futures": eligibility.sum(axis=1).to_numpy(),
            "eligible_futures_with_valid_score": valid.sum(axis=1).to_numpy(),
            "eligible_commodities": commodity_eligibility.sum(axis=1).to_numpy(),
            "eligible_commodities_with_valid_score": commodity_valid.sum(axis=1).to_numpy(),
            "eligible_futures_with_full_lookback": full_history.sum(axis=1).to_numpy(),
            "eligible_commodities_with_full_lookback": commodity_full_history.sum(
                axis=1
            ).to_numpy(),
            "selected_long_commodities": weights.gt(0.0).sum(axis=1).to_numpy(),
            "selected_short_commodities": weights.lt(0.0).sum(axis=1).to_numpy(),
        }
    )

    prices = e5._prices(data).reindex(columns=eligibility.columns)
    rows = []
    for ticker in eligibility.columns:
        eligible_dates = dates[eligibility[ticker].to_numpy()]
        valid_dates = dates[valid[ticker].to_numpy()]
        eligible_observations = momentum_observations.loc[eligible_dates, ticker]
        full_dates = eligible_observations.index[eligible_observations.ge(lookback)]
        rows.append(
            {
                "ticker": ticker,
                "name": str(data.taxonomy.at[ticker, "name"]).replace("_", " "),
                "broad_asset_class": sleeves[ticker],
                "source_price_first": prices[ticker].first_valid_index(),
                "eligible_first": eligible_dates.min() if len(eligible_dates) else pd.NaT,
                "eligible_last": eligible_dates.max() if len(eligible_dates) else pd.NaT,
                "eligible_decision_count": len(eligible_dates),
                "valid_score_first": valid_dates.min() if len(valid_dates) else pd.NaT,
                "valid_score_decision_count": len(valid_dates),
                "momentum_observations_at_entry": int(eligible_observations.iloc[0])
                if len(eligible_observations)
                else 0,
                "partial_lookback_decision_count": int(
                    eligible_observations.lt(lookback).sum()
                ),
                "first_full_lookback_decision": full_dates.min()
                if len(full_dates)
                else pd.NaT,
                "owner_excluded": ticker in e5.FUTURES_INVESTABILITY_EXCLUSIONS,
            }
        )
    history = pd.DataFrame(rows).sort_values(
        ["owner_excluded", "eligible_first", "ticker"], na_position="last"
    )
    return per_date, history


def run() -> Mapping[str, pd.DataFrame]:
    """Reconstruct the global Commodities book and write its exact P&L ranking."""
    started = time.perf_counter()
    data = e5.load_universe(equal.UNIVERSE)
    dates = matched._window_dates(
        e5.load_cached(equal.UNIVERSE, e5.SmootherName.BASELINE).dates
    )
    eligibility = e5._investable_eligibility(data, dates)
    columns = eligibility.columns
    raw_scores = e5._raw_momentum_scores(data, dates, vol_adjusted=False)
    raw_scores = raw_scores.reindex(columns=columns)
    scores = raw_scores.where(eligibility)
    prices = matched._prices_with_context(e5._prices(data).reindex(columns=columns))
    sleeves = equal._broad_sleeves(data.taxonomy, columns)
    sleeve_panel = equal._sleeve_panel(dates, sleeves)
    weights, weight_diagnostics = asset._standalone_weights(
        scores,
        eligibility,
        sleeve_panel,
        sleeve_panel,
        ASSET_CLASS,
        Q,
    )
    costs_rate = e5.get_universe_spec(equal.UNIVERSE).cost_bps / 10000.0
    net, _ = equal._backtest(
        prices,
        weights,
        costs_rate,
        "futures_commodities_global_long_short_q_0.20_attribution",
    )
    bounded_nav = matched._bounded_panel(net.get_portfolio_nav()).dropna()
    start = bounded_nav.index.min()
    end = bounded_nav.index.max()
    net_currency, diagnostics = _net_currency_pnl(net, start, end)
    raw_costs = net.realized_costs.reindex_like(net_currency).fillna(0.0)
    ranking = _rank_contributions(
        net_currency=net_currency,
        costs=raw_costs,
        beginning_nav=float(diagnostics["beginning_nav"]),
        taxonomy=data.taxonomy,
        sleeves=sleeves,
        eligible_tickers=eligibility.columns[eligibility.any(axis=0)],
        weights=weights,
    )

    eligible_once = eligibility.any(axis=0)
    commodity_tickers = sleeves.index[sleeves.eq(ASSET_CLASS) & eligible_once]
    outside_tickers = sleeves.index[~sleeves.index.isin(commodity_tickers)]
    outside_pnl = float(
        net_currency.reindex(columns=outside_tickers).abs().to_numpy().max()
    )
    ranked_total = float(ranking["net_pnl_currency"].sum())
    attribution_total = float(diagnostics["attributed_net_pnl_currency"])
    ranking_error = abs(ranked_total - attribution_total)
    exclusion_error = float(
        weights.reindex(columns=e5.FUTURES_INVESTABILITY_EXCLUSIONS)
        .fillna(0.0)
        .abs()
        .to_numpy()
        .max()
    )
    availability, history = _availability_diagnostics(
        data=data,
        dates=dates,
        eligibility=eligibility,
        raw_scores=raw_scores,
        sleeves=sleeves,
        weights=weights,
    )
    valid_count_error = int(
        (
            availability["eligible_futures"]
            - availability["eligible_futures_with_valid_score"]
        )
        .abs()
        .max()
    )
    partial_history_max = int(
        (
            availability["eligible_futures"]
            - availability["eligible_futures_with_full_lookback"]
        ).max()
    )
    passed = (
        len(ranking) == len(commodity_tickers)
        and float(diagnostics["max_step_reconciliation_abs_error"])
        <= ACCOUNTING_TOLERANCE
        and float(diagnostics["cumulative_reconciliation_abs_error"])
        <= ACCOUNTING_TOLERANCE
        and outside_pnl <= ACCOUNTING_TOLERANCE
        and ranking_error <= ACCOUNTING_TOLERANCE
        and exclusion_error <= ACCOUNTING_TOLERANCE
        and valid_count_error == 0
    )
    acceptance = pd.DataFrame(
        [
            {
                "universe": equal.UNIVERSE.value,
                "analysis_window": matched.WINDOW,
                "asset_class": ASSET_CLASS,
                "method": METHOD,
                "q": Q,
                "contracts_ranked": len(ranking),
                "decision_dates": len(dates),
                **diagnostics,
                "max_outside_commodity_abs_pnl_currency": outside_pnl,
                "ranking_total_reconciliation_abs_error": ranking_error,
                "max_owner_excluded_weight_abs": exclusion_error,
                "max_eligible_minus_valid_score_count_abs": valid_count_error,
                "max_eligible_partial_lookback_count": partial_history_max,
                "eligible_futures_min": int(availability["eligible_futures"].min()),
                "eligible_futures_max": int(availability["eligible_futures"].max()),
                "eligible_commodities_min": int(
                    availability["eligible_commodities"].min()
                ),
                "eligible_commodities_max": int(
                    availability["eligible_commodities"].max()
                ),
                "max_weight_exposure_abs_error": max(
                    abs(float(value))
                    for key, value in weight_diagnostics.items()
                    if key.startswith("max_")
                    and key.endswith(("error", "leakage"))
                    and "group_budget" not in key
                ),
                "accounting_tolerance": ACCOUNTING_TOLERANCE,
                "status": "PASS" if passed else "FAIL",
                "runner": RUNNER,
            }
        ]
    )
    if not passed:
        raise AssertionError(acceptance.to_dict(orient="records")[0])

    design = pd.DataFrame(
        [
            {
                "universe": equal.UNIVERSE.value,
                "analysis_window": matched.WINDOW,
                "strategy": "standalone_commodities_long_short",
                "method": METHOD,
                "q": Q,
                "signal": "48-week log-return sum excluding latest 4 weeks",
                "signal_sum_min_count": 1,
                "full_lookback_required": False,
                "decision_frequency": "ME",
                "implementation_lag": 1,
                "cost_bps": 10000.0 * costs_rate,
                "attribution": "exact currency holding P&L less realised instrument costs",
                "normalization": "percentage points of beginning net NAV",
                "rank_direction": "descending net P&L contribution",
                "owner_exclusions": "|".join(
                    sorted(e5.FUTURES_INVESTABILITY_EXCLUSIONS)
                ),
                "runner": RUNNER,
            }
        ]
    )
    outputs = {
        "ranking": ranking,
        "acceptance": acceptance,
        "design": design,
        "availability_by_date": availability,
        "contract_history": history,
    }
    for name, frame in outputs.items():
        e5._write(frame, _root() / f"{name}.csv")
    e5._write(
        pd.DataFrame([{"runtime_seconds": time.perf_counter() - started}]),
        _root() / "runtime.csv",
    )
    return outputs


def _hash_outputs() -> dict[str, str]:
    """Hash deterministic numerical outputs, excluding timing and replay files."""
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(_root().glob("*.csv"))
        if path.name not in {"runtime.csv", "determinism.csv"}
    }


def verify_determinism() -> pd.DataFrame:
    """Replay the attribution and require byte-identical numerical outputs."""
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
    """Run, replay, and print the full commodity-contract net-P&L ranking."""
    replay = verify_determinism()
    ranking = pd.read_csv(
        _root() / "ranking.csv", float_precision="round_trip"
    )
    print(
        ranking[["rank", "ticker", "name", "net_pnl_pct_of_start"]].to_string(
            index=False
        )
    )
    print(
        f"Commodity attribution: PASS ({len(replay)}/{len(replay)} deterministic)",
        flush=True,
    )


if __name__ == "__main__":
    main()
