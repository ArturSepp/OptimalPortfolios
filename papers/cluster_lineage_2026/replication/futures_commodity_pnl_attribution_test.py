"""Tests for exact net-P&L attribution of the futures Commodities global book."""
from types import SimpleNamespace

import pandas as pd
import pytest

import papers.cluster_lineage_2026.replication.run_backtests as e5
from papers.cluster_lineage_2026.replication.run_futures_commodity_pnl_attribution import (
    _net_currency_pnl,
)


def test_net_currency_pnl_reconciles_costs_and_nav() -> None:
    """Instrument holding P&L less costs must exactly explain each NAV change."""
    index = pd.date_range("2026-01-07", periods=3, freq="W-WED")
    prices = pd.DataFrame(
        {"A": [10.0, 11.0, 12.0], "B": [20.0, 18.0, 21.0]}, index=index
    )
    units = pd.DataFrame(
        {"A": [2.0, 2.0, 1.0], "B": [-1.0, -1.0, -1.0]}, index=index
    )
    costs = pd.DataFrame(
        {"A": [0.0, 0.2, 0.1], "B": [0.0, 0.3, 0.0]}, index=index
    )
    # Step P&L is [+2.0, +2.0] - [0.2, 0.3] = +3.5, then
    # [+2.0, -3.0] - [0.1, 0.0] = -1.1.
    nav = pd.Series([100.0, 103.5, 102.4], index=index, name="portfolio")
    portfolio = SimpleNamespace(
        prices=prices, units=units, realized_costs=costs, nav=nav
    )

    pnl, diagnostics = _net_currency_pnl(portfolio, index[0], index[-1])

    assert pnl.loc[index[1], "A"] == pytest.approx(1.8)
    assert pnl.loc[index[1], "B"] == pytest.approx(1.7)
    assert pnl.loc[index[2], "A"] == pytest.approx(1.9)
    assert pnl.loc[index[2], "B"] == pytest.approx(-3.0)
    assert diagnostics["max_step_reconciliation_abs_error"] <= 1e-12
    assert diagnostics["cumulative_reconciliation_abs_error"] <= 1e-12


def test_owner_excluded_contracts_are_never_investable_in_futures_backtests() -> None:
    """Every owner-excluded source contract must stay ineligible on every date."""
    data = e5.load_universe(e5.UniverseName.FUTURES)
    dates = data.eligibility["W-WED"].index

    eligibility = e5._investable_eligibility(data, dates)

    excluded = sorted(e5.FUTURES_INVESTABILITY_EXCLUSIONS)
    assert set(excluded).issubset(eligibility.columns)
    assert not eligibility.loc[:, excluded].any(axis=None)
