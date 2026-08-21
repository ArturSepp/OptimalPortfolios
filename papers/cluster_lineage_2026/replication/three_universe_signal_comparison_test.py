"""Regression checks for the matched three-universe momentum comparison."""
import pandas as pd
import pytest

import papers.cluster_lineage_2026.replication.run_three_universe_signal_comparison as runner


@pytest.fixture(scope="module")
def outputs() -> dict[str, pd.DataFrame]:
    """Execute the cache-first comparison once for all assertions."""
    return dict(runner.run())


def test_frozen_costs_weights_and_signals() -> None:
    """The owner-specified costs, sleeve budgets, q, and signal pair stay fixed."""
    assert runner.Q == 0.25
    assert runner.SIGNALS == (runner.ROSAA, runner.CLASSIC)
    assert runner.U1_COST_BPS == 10.0
    assert runner.FUNDS_COST_BPS == 20.0
    assert runner.FUTURES_COST_BPS == 10.0
    assert runner.FUNDS_TARGET == {
        "Equity": 0.50,
        "Fixed Income": 0.30,
        "Rest": 0.20,
    }
    assert runner.FUTURES_TARGET == {
        "Equity": 0.30,
        "Fixed Income": 0.30,
        "Commodities": 0.30,
        "FX": 0.10,
    }


def test_performance_table_is_complete_and_matched(
    outputs: dict[str, pd.DataFrame],
) -> None:
    """Every requested signal/leg appears once on the common headline window."""
    performance = outputs["performance"]
    assert len(performance) == 14
    assert not performance.duplicated(["universe", "signal", "leg"]).any()
    assert set(performance["analysis_window"]) == {runner.WINDOW}
    assert set(performance["q"]) == {runner.Q}
    expected_legs = {
        "U1_equities": {"cluster", "sector", "global"},
        "U2_BlackRock_funds": {"cluster", "global"},
        "U3_futures": {"cluster", "global"},
    }
    for universe, legs in expected_legs.items():
        current = performance.loc[performance["universe"].eq(universe)]
        assert set(current["signal"]) == set(runner.SIGNALS)
        assert set(current["leg"]) == legs


def test_only_owner_specified_payoff_benchmarks_are_compared(
    outputs: dict[str, pd.DataFrame],
) -> None:
    """U1 uses sector/global; funds and futures use their same-budget global ranks."""
    comparison = outputs["benchmark_comparison"]
    assert len(comparison) == 8
    assert set(
        comparison.loc[
            comparison["universe"].eq("U1_equities"), "benchmark_leg"
        ]
    ) == {"sector", "global"}
    assert set(
        comparison.loc[
            ~comparison["universe"].eq("U1_equities"), "benchmark_leg"
        ]
    ) == {"global"}
    assert not comparison.astype(str).apply(
        lambda column: column.str.contains("EW_all", case=False).any()
    ).any()


def test_costs_and_every_acceptance_line_pass(
    outputs: dict[str, pd.DataFrame],
) -> None:
    """Costs match the instruction and all signal/weight checks are green."""
    performance = outputs["performance"]
    expected_costs = {
        "U1_equities": 10.0,
        "U2_BlackRock_funds": 20.0,
        "U3_futures": 10.0,
    }
    measured = performance.groupby("universe")["cost_bps_one_way"].unique()
    for universe, cost in expected_costs.items():
        assert measured[universe].tolist() == [cost]
    assert outputs["acceptance"]["status"].eq("PASS").all()
    assert outputs["signal_preflight"]["status"].eq("PASS").all()


def test_signal_comparison_has_one_row_per_unchanged_leg(
    outputs: dict[str, pd.DataFrame],
) -> None:
    """Classic-minus-ROSAA deltas cover all seven unchanged portfolio legs."""
    comparison = outputs["signal_comparison"]
    assert len(comparison) == 7
    assert not comparison.duplicated(["universe", "leg"]).any()


def test_futures_rosaa_reproduces_the_owner_frozen_selected_result(
    outputs: dict[str, pd.DataFrame],
) -> None:
    """The unified runner must reproduce the selected futures cell exactly."""
    performance = outputs["performance"].set_index(
        ["universe", "signal", "leg"]
    )
    method_by_leg = {
        "cluster": runner.FUTURES_CLUSTER,
        "global": runner.FUTURES_GLOBAL,
    }
    for leg, method in method_by_leg.items():
        measured = performance.loc[("U3_futures", runner.ROSAA, leg)]
        frozen = runner.futures_best.FROZEN_PERFORMANCE[method]
        for metric, expected in frozen.items():
            assert measured[metric] == pytest.approx(expected, abs=1e-12)
