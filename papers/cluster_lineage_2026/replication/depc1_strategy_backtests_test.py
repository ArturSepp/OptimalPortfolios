"""Focused tests for the fixed de-PC1 strategy comparison harness."""

from types import SimpleNamespace

import pandas as pd

import papers.cluster_lineage_2026.replication.run_depc1_strategy_backtests as run


def test_performance_comparison_never_uses_ew_as_a_yardstick():
    """Only explicitly supplied ranking-leg contrasts may enter the table."""
    rows = []
    for leg, value in (("cluster_raw", 1.0), ("cluster_depc1", 2.0), ("global", 0.5)):
        row = {
            "analysis_window": "window",
            "leg": leg,
            **{metric: value for metric in run.PERFORMANCE_METRICS},
        }
        rows.append(row)
    output = run._comparison(
        pd.DataFrame(rows),
        (("cluster_depc1", "cluster_raw"), ("cluster_depc1", "global")),
    )
    assert set(output["benchmark_leg"]) == {"cluster_raw", "global"}
    assert not output.astype(str).apply(lambda column: column.str.contains("EW").any()).any()


def test_long_short_exposure_error_is_exact_for_balanced_book():
    """The generic gate must recognize exact +1/-1 exposure."""
    weights = pd.DataFrame([[0.6, 0.4, -0.25, -0.75]])
    assert run._exposure_error(weights) == 0.0


def test_acceptance_row_supports_bound_and_identity_checks():
    """Measured/tolerance reporting must distinguish inequalities and identities."""
    assert run._acceptance_row("bound", 1e-13, 1e-12)["status"] == "PASS"
    assert run._acceptance_row("identity", 10.0, 10.0, exact=True)["status"] == "PASS"
    assert run._acceptance_row("identity", 9.0, 10.0, exact=True)["status"] == "FAIL"


def test_costs_and_primary_parameters_are_frozen():
    """The runner must retain the owner-frozen q and one-way costs."""
    assert run.u1_bics.Q == run.u2_sensitivity.Q == run.futures_best.Q == 0.25
    assert run.u1_bics.COST_BPS == run.futures_best.COST_BPS == 10.0
    assert run.u2_sensitivity.COST_BPS == 20.0
    assert run.empirical_specs.U1_OPTIMAL_SPEC.momentum_mean_adj_type == "NONE"


def test_instrument_attribution_matches_qis_missing_price_valuation():
    """A disappearing valued leg must contribute its full drop from NAV."""
    dates = pd.DatetimeIndex(["2020-01-01", "2020-01-02"])
    prices = pd.DataFrame({"A": [100.0, float("nan")]}, index=dates)
    portfolio = SimpleNamespace(
        prices=prices,
        units=pd.DataFrame({"A": [1.0, 1.0]}, index=dates),
        realized_costs=pd.DataFrame(0.0, index=dates, columns=["A"]),
        get_portfolio_nav=lambda: pd.Series([100.0, 0.0], index=dates),
    )
    table, diagnostics = run._instrument_attribution(portfolio, "leg", "test")
    assert table.at[0, "net_pnl_currency"] == -100.0
    assert diagnostics["cumulative_reconciliation_abs_error"] == 0.0
