"""Run futures 30/30/30/10 portfolios on the exact U1 headline window.

Decision dates are restricted to 2009-08-31 through 2026-06-30, inclusive.  The
tradable W-WED price path and EW reference are bounded to the same calendar interval,
so pre-strategy cash history cannot dilute annual return, volatility, or Sharpe.  This
corrects the horizon defect in the earlier exploratory full-price-path futures output.

Long-only and +1/-1 long-short results are persisted in separate tables.  Both use the
owner's 30% Equity, 30% Fixed Income, 30% Commodities, and 10% FX target on every
constrained side.  Signal, q values, monthly decisions, implementation lag, costs, and
the accepted baseline and M1-star cluster caches are otherwise unchanged.  EW-all is
reference-only for beta and alpha columns and is never a payoff yardstick.
"""
from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

import papers.cluster_lineage_2026.replication.run_backtests as e5
import papers.cluster_lineage_2026.replication.run_futures_sleeve_grid as equal
import papers.cluster_lineage_2026.replication.run_futures_weight_30303010 as full


WINDOW = "u1_headline_20090831_20260630"
WINDOW_START = pd.Timestamp("2009-08-31")
WINDOW_END = pd.Timestamp("2026-06-30")
TARGET = dict(full.TARGET)
RUNNER = (
    "papers/cluster_lineage_2026/replication/"
    "run_futures_weight_30303010_u1_window.py"
)


def _root() -> Path:
    """Return and create the external U1-window futures output directory."""
    return e5.get_output_path(
        "e5b", "futures_weight_30_30_30_10_u1_window", create=True
    )


def _window_dates(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """Restrict monthly decisions to the owner-frozen U1 headline dates."""
    selected = dates[(dates >= WINDOW_START) & (dates <= WINDOW_END)]
    if len(selected) != 203:
        raise AssertionError(f"expected 203 U1-window decisions, measured {len(selected)}")
    if selected.min() != WINDOW_START or selected.max() != WINDOW_END:
        raise AssertionError(
            f"decision bounds differ from U1: {selected.min()}..{selected.max()}"
        )
    return selected


def _bounded_panel(panel: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Remove every observation outside the common calendar performance window."""
    bounded = panel.loc[(panel.index >= WINDOW_START) & (panel.index <= WINDOW_END)]
    if bounded.empty:
        raise AssertionError("bounded performance panel is empty")
    return bounded


def _prices_with_context(prices: pd.DataFrame) -> pd.DataFrame:
    """Keep one real pre-window mark for alignment, then only in-window prices."""
    prior = prices.loc[prices.index <= WINDOW_START].tail(1)
    bounded = prices.loc[(prices.index > WINDOW_START) & (prices.index <= WINDOW_END)]
    if len(prior) != 1 or bounded.empty:
        raise AssertionError("futures price context cannot bracket the U1 window")
    return pd.concat([prior, bounded]).sort_index()


class _WindowedPortfolio:
    """Expose only in-window NAV and turnover from an otherwise standard qis backtest."""

    def __init__(self, portfolio) -> None:
        """Store the underlying qis portfolio result."""
        self._portfolio = portfolio

    def get_portfolio_nav(self) -> pd.Series:
        """Return NAV observations inside the common calendar window."""
        return _bounded_panel(self._portfolio.get_portfolio_nav())

    def get_turnover(self, *args, **kwargs):
        """Return turnover observations inside the common calendar window."""
        return _bounded_panel(self._portfolio.get_turnover(*args, **kwargs))


def _design(
    dates: pd.DatetimeIndex, prices: pd.DataFrame, sleeves: pd.Series
) -> pd.DataFrame:
    """Return the machine-readable matched-window experiment design."""
    spec = e5.get_universe_spec(equal.UNIVERSE)
    return pd.DataFrame(
        [
            {
                "universe": equal.UNIVERSE.value,
                "analysis_window": WINDOW,
                "contracts": len(sleeves),
                "decision_dates": len(dates),
                "decision_start": dates.min(),
                "decision_end": dates.max(),
                "alignment_price_start": prices.index.min(),
                "performance_calendar_start": WINDOW_START,
                "performance_price_end": prices.index.max(),
                "signal": "48-week log-return sum excluding latest 4 weeks",
                "primary_q": equal.PRIMARY_Q,
                "robustness_q": equal.QUANTILES[1],
                "cost_bps": spec.cost_bps,
                "implementation_lag": 1,
                "equity_target": TARGET["Equity"],
                "fixed_income_target": TARGET["Fixed Income"],
                "commodities_target": TARGET["Commodities"],
                "fx_target": TARGET["FX"],
                "configs": "baseline|M1_star",
                "returns_convention": equal.data_convention(spec),
                "runner": RUNNER,
            }
        ]
    )


def _run_leg(
    *,
    strategy: str,
    method: str,
    q: float,
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    diagnostics: Mapping[str, float],
    sleeve_panel: pd.DataFrame,
    ew_nav: pd.Series,
    costs: float,
    target: Mapping[str, float] | None,
) -> tuple[dict, dict, list[dict], dict]:
    """Backtest one bounded-window portfolio and return all audit records."""
    net, gross = equal._backtest(
        prices,
        weights,
        costs,
        f"futures_u1_window_{strategy}_{method}_q_{q:.2f}",
    )
    windowed_net = _WindowedPortfolio(net)
    windowed_gross = _WindowedPortfolio(gross)
    nav = windowed_net.get_portfolio_nav().dropna()
    nav_start = pd.Timestamp(nav.index.min())
    nav_end = pd.Timestamp(nav.index.max())
    measurement_years = (nav_end - nav_start).days / 365.25
    performance = {
        "universe": equal.UNIVERSE.value,
        "analysis_window": WINDOW,
        "strategy": strategy,
        "method": method,
        "q": q,
        **equal._performance_payload(windowed_net, windowed_gross, ew_nav),
        "nav_start": nav_start,
        "nav_end": nav_end,
        "measurement_years": measurement_years,
        "runner": RUNNER,
    }
    ordinary_errors = [
        abs(float(value))
        for key, value in diagnostics.items()
        if key.startswith("max_")
        and key.endswith("error")
        and "group_budget" not in key
    ]
    group_error = float(
        diagnostics["max_within_sleeve_group_budget_abs_error"]
    )
    passed = (
        max(ordinary_errors) <= equal.EXPOSURE_TOLERANCE
        and group_error <= equal.GROUP_BUDGET_TOLERANCE
    )
    acceptance = {
        "analysis_window": WINDOW,
        "strategy": strategy,
        "method": method,
        "q": q,
        **diagnostics,
        "exposure_tolerance": equal.EXPOSURE_TOLERANCE,
        "weight_tolerance": equal.WEIGHT_TOLERANCE,
        "group_budget_tolerance": equal.GROUP_BUDGET_TOLERANCE,
        "status": "PASS" if passed else "FAIL",
    }
    allocation = equal._allocation_rows(
        strategy, method, q, weights, sleeve_panel, target
    )
    for row in allocation:
        row["analysis_window"] = WINDOW
    horizon = {
        "analysis_window": WINDOW,
        "strategy": strategy,
        "method": method,
        "q": q,
        "nav_start": nav_start,
        "nav_end": nav_end,
        "measurement_years": measurement_years,
        "nav_rows": len(nav),
        "pre_window_nav_rows": int((nav.index < WINDOW_START).sum()),
        "post_window_nav_rows": int((nav.index > WINDOW_END).sum()),
    }
    return performance, acceptance, allocation, horizon


def _global_weight_regression(weights: pd.DataFrame) -> pd.DataFrame:
    """Match primary global decisions to the accepted panel on common dates."""
    accepted = pd.read_csv(
        equal._accepted_root() / "weights.csv",
        parse_dates=["index"],
        float_precision="round_trip",
    )
    accepted = accepted.loc[accepted["leg"].eq("global")].set_index("index")
    accepted = accepted.drop(columns="leg").reindex(
        index=weights.index, columns=weights.columns
    )
    error = float((weights - accepted).abs().to_numpy().max())
    return pd.DataFrame(
        [
            {
                "check": "accepted q=0.20 global decisions restricted to U1 window",
                "measured_max_abs_error": error,
                "tolerance": 1e-12,
                "status": "PASS" if error <= 1e-12 else "FAIL",
            }
        ]
    )


def _legacy_horizon_diagnostic() -> pd.DataFrame:
    """Document why the earlier full-price-path performance rows are superseded."""
    performance = pd.read_csv(
        full._root() / "performance.csv", float_precision="round_trip"
    )
    implied = np.log1p(performance["net_total_return"]) / np.log1p(
        performance["net_return_annualized"]
    )
    design = pd.read_csv(full._root() / "design.csv")
    return pd.DataFrame(
        [
            {
                "legacy_decision_start": design.loc[0, "decision_start"],
                "legacy_decision_end": design.loc[0, "decision_end"],
                "legacy_implied_measurement_years_min": float(implied.min()),
                "legacy_implied_measurement_years_max": float(implied.max()),
                "status": "SUPERSEDED_PRE_STRATEGY_CASH_HISTORY",
                "replacement_window": WINDOW,
            }
        ]
    )


def _u1_reference_horizon_diagnostic() -> pd.DataFrame:
    """Audit the accepted U1 headline artifact before any cross-universe comparison."""
    root = e5.get_output_path("e5b", "group_equal") / e5.UniverseName.MSCI_US.value
    navs = pd.read_csv(root / "navs.csv", parse_dates=["date"])
    navs = navs.loc[
        navs["analysis_window"].eq("headline_20090831_20260630")
    ].set_index("date")
    global_nav = navs["global"].dropna()
    active = global_nav.loc[global_nav.sub(global_nav.iloc[0]).abs().gt(1e-12)]
    performance = pd.read_csv(
        root / "performance.csv", float_precision="round_trip"
    )
    performance = performance.loc[
        performance["analysis_window"].eq("headline_20090831_20260630")
    ]
    implied = np.log1p(performance["net_total_return"]) / np.log1p(
        performance["net_return_annualized"]
    )
    return pd.DataFrame(
        [
            {
                "u1_artifact_nav_start": global_nav.index.min(),
                "u1_artifact_first_active_nav": active.index.min(),
                "u1_artifact_nav_end": global_nav.index.max(),
                "u1_implied_measurement_years_min": float(implied.min()),
                "u1_implied_measurement_years_max": float(implied.max()),
                "stated_u1_window_start": WINDOW_START,
                "stated_u1_window_end": WINDOW_END,
                "status": "REMEASURE_U1_BEFORE_CROSS_UNIVERSE_PAYOFF_COMPARISON",
            }
        ]
    )


def run() -> Mapping[str, pd.DataFrame]:
    """Execute the U1-matched long-only and long-short futures experiment once."""
    started = time.perf_counter()
    full._validate_target()
    data = e5.load_universe(equal.UNIVERSE)
    dates = _window_dates(
        e5.load_cached(equal.UNIVERSE, e5.SmootherName.BASELINE).dates
    )
    eligibility = e5._investable_eligibility(data, dates)
    columns = eligibility.columns
    scores = e5._raw_momentum_scores(
        data, dates, vol_adjusted=False
    ).reindex(columns=columns).where(eligibility)
    prices = _prices_with_context(e5._prices(data).reindex(columns=columns))
    sleeves = equal._broad_sleeves(data.taxonomy, columns)
    sleeve_panel = equal._sleeve_panel(dates, sleeves)
    accepted_navs = pd.read_csv(
        equal._accepted_root() / "navs.csv",
        parse_dates=["date"],
        float_precision="round_trip",
    ).set_index("date")
    ew_nav = _bounded_panel(accepted_navs["EW_all"])
    if not isinstance(ew_nav, pd.Series):
        raise AssertionError("bounded futures EW reference is not a Series")
    costs = e5.get_universe_spec(equal.UNIVERSE).cost_bps / 10000.0
    cluster_groups = {
        config: equal._hierarchical_groups(
            e5._cluster_groups(equal.UNIVERSE, config).reindex(
                index=dates, columns=columns
            ),
            sleeve_panel,
        )
        for config in equal.CONFIGS
    }
    constrained_groups = {
        "sleeve_global": sleeve_panel,
        **{
            f"sleeve_cluster_{config.value}": groups
            for config, groups in cluster_groups.items()
        },
    }

    performance_rows = []
    acceptance_rows = []
    allocation_rows = []
    horizon_rows = []
    primary_global_weights = None
    for q in equal.QUANTILES:
        for strategy in ("long_only", "long_short"):
            original_weights, diagnostics = equal._original_global_weights(
                strategy, scores, eligibility, q
            )
            if strategy == "long_only" and q == equal.PRIMARY_Q:
                primary_global_weights = original_weights
            records = _run_leg(
                strategy=strategy,
                method="original_global",
                q=q,
                prices=prices,
                weights=original_weights,
                diagnostics=diagnostics,
                sleeve_panel=sleeve_panel,
                ew_nav=ew_nav,
                costs=costs,
                target=None,
            )
            performance, acceptance, allocation, horizon = records
            performance_rows.append(performance)
            acceptance_rows.append(acceptance)
            allocation_rows.extend(allocation)
            horizon_rows.append(horizon)

            for method, groups in constrained_groups.items():
                weights, diagnostics = full._build_constrained_weights(
                    strategy,
                    scores,
                    eligibility,
                    sleeve_panel,
                    groups,
                    q,
                )
                records = _run_leg(
                    strategy=strategy,
                    method=method,
                    q=q,
                    prices=prices,
                    weights=weights,
                    diagnostics=diagnostics,
                    sleeve_panel=sleeve_panel,
                    ew_nav=ew_nav,
                    costs=costs,
                    target=TARGET,
                )
                performance, acceptance, allocation, horizon = records
                performance_rows.append(performance)
                acceptance_rows.append(acceptance)
                allocation_rows.extend(allocation)
                horizon_rows.append(horizon)

    if primary_global_weights is None:
        raise AssertionError("primary accepted global weights were not constructed")
    performance = pd.DataFrame(performance_rows).sort_values(
        ["strategy", "q", "method"]
    ).reset_index(drop=True)
    comparison = equal._comparison(performance).reset_index(drop=True)
    acceptance = pd.DataFrame(acceptance_rows).sort_values(
        ["strategy", "q", "method"]
    ).reset_index(drop=True)
    if not acceptance["status"].eq("PASS").all():
        raise AssertionError(acceptance.loc[~acceptance["status"].eq("PASS")])
    regression = _global_weight_regression(primary_global_weights)
    if not regression["status"].eq("PASS").all():
        raise AssertionError(regression)
    outputs = {
        "design": _design(dates, prices, sleeves),
        "performance_long_only": performance.loc[
            performance["strategy"].eq("long_only")
        ].reset_index(drop=True),
        "performance_long_short": performance.loc[
            performance["strategy"].eq("long_short")
        ].reset_index(drop=True),
        "comparison_long_only": comparison.loc[
            comparison["strategy"].eq("long_only")
        ].reset_index(drop=True),
        "comparison_long_short": comparison.loc[
            comparison["strategy"].eq("long_short")
        ].reset_index(drop=True),
        "allocation_diagnostics": pd.DataFrame(allocation_rows),
        "acceptance": acceptance,
        "horizon_diagnostic": pd.DataFrame(horizon_rows),
        "global_weight_regression": regression,
        "legacy_horizon_diagnostic": _legacy_horizon_diagnostic(),
        "u1_reference_horizon_diagnostic": _u1_reference_horizon_diagnostic(),
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
    """Replay both strategies and require byte-identical numerical outputs."""
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
    """Run and replay the U1-window futures experiment."""
    replay = verify_determinism()
    print(
        "Futures 30/30/30/10 U1-window: PASS "
        f"({len(replay)}/{len(replay)} deterministic)",
        flush=True,
    )


if __name__ == "__main__":
    main()
