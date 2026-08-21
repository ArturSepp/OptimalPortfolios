"""Focused tests for the F8 frozen-artifact manuscript exhibit builder."""
from __future__ import annotations

import ast
from pathlib import Path

import pandas as pd

from papers.cluster_lineage_2026.replication import build_final_exhibits as f8


def _read(name: str) -> pd.DataFrame:
    """Read one emitted F8 evidence table."""
    return pd.read_csv(f8._root() / name, float_precision="round_trip")


def test_exhibit_budget_and_index_are_exact() -> None:
    """The index must contain six figures, eight body tables, and two appendix tables."""
    index = _read("exhibit_index.csv")
    assert len(index) == 16
    assert index["exhibit_id"].is_unique
    assert index["category"].value_counts().to_dict() == {
        "figure": 6,
        "existing_body_table": 4,
        "new_body_table": 4,
        "appendix_table": 2,
    }
    assert set(index.loc[index["category"].eq("figure"), "exhibit_id"]) == set(f8.FIGURE_IDS)


def test_all_payloads_are_byte_identical_on_replay() -> None:
    """Every non-acceptance F8 payload must reproduce byte for byte."""
    determinism = _read("determinism.csv")
    assert len(determinism) == 37
    assert determinism["byte_identical"].all()
    assert not determinism["artifact"].str.contains("acceptance|determinism").any()


def test_visible_number_reconciliations_are_tight() -> None:
    """F5 endpoints and F6 averages must agree with their manuscript tables."""
    checks = _read("precision_reconciliation.csv")
    assert len(checks) == 27
    assert checks["absolute_error"].max() <= 1e-12
    assert set(checks["check"]) == {
        "F5 NAV endpoint vs tab:signal",
        "F6 mean vs tab:concentration",
    }


def test_selection_tables_label_only_frozen_operating_points() -> None:
    """Each of the four disclosed grids must identify exactly one selected row."""
    table_te = _read("tables/table_TE_selection_grids.csv")
    table_tf = _read("tables/table_TF_selection_grids.csv")
    combined = pd.concat([table_te, table_tf], ignore_index=True)
    selected = combined.loc[combined["selection_role"].eq("selected_operating_point")]
    assert selected.groupby("grid").size().to_dict() == {
        "U1 covariance frequency/span": 1,
        "U1 minimum cluster size": 1,
        "U2 eligibility": 1,
        "U3 short span": 1,
    }
    nonselected = combined.loc[~combined.index.isin(selected.index), "selection_role"]
    assert nonselected.eq("selection_record_not_independent_confirmation").all()


def test_u1_performance_exhibits_trace_to_g0() -> None:
    """Every exhibit carrying corrected U1 performance or CIs must name G0 as source."""
    index = _read("exhibit_index.csv").set_index("exhibit_id")
    for exhibit_id in ("F5", "tab:signal", "tab:risk"):
        assert "g0" in index.loc[exhibit_id, "source_artifact_path"].lower()
    signal = _read("tables/table_existing_signal.csv")
    assert signal.loc[signal["universe"].eq("U1"), "monthly_observations"].eq(202).all()


def test_builder_has_no_backtest_or_estimator_entry_point() -> None:
    """F8 must remain a cached-artifact renderer rather than an empirical runner."""
    source = Path(f8.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imported.update(
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    )
    assert not any("run_backtests" in name for name in imported)
    called = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert called.isdisjoint(
        {"backtest_model_portfolio", "compute_rolling_smoothed_clusters", "fit", "solve"}
    )
