"""Focused tests for the G0 cached-series U1 headline-window re-score."""
from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pandas as pd

from papers.cluster_lineage_2026.replication import run_g0_u1_window_rescore as g0


def test_window_uses_last_pre_start_nav_as_return_base() -> None:
    """The August 26 observation must seed the first headline-window monthly return."""
    index = pd.to_datetime(
        ["2009-08-19", "2009-08-26", "2009-09-02", "2026-06-24", "2026-07-01"]
    )
    frame = pd.DataFrame({"nav": np.arange(5.0)}, index=index)
    window = g0._window_navs(frame)
    assert window.index[0] == pd.Timestamp("2009-08-26")
    assert window.index[-1] == pd.Timestamp("2026-06-24")
    assert window.index.is_monotonic_increasing


def test_metric_record_matches_independent_formula() -> None:
    """G0 leg statistics must match an independently written monthly formula."""
    index = pd.date_range("2010-01-31", periods=25, freq="ME")
    returns = np.linspace(-0.02, 0.03, 24)
    nav = pd.Series(100.0 * np.r_[1.0, np.cumprod(1.0 + returns)], index=index)
    observed, monthly = g0._metric_record(nav)
    years = (index[-1] - index[0]).days / 365.25
    expected = {
        "net_return_annualized": (nav.iloc[-1] / nav.iloc[0]) ** (1.0 / years) - 1.0,
        "volatility_annualized": monthly.std() * np.sqrt(12.0),
        "sharpe_rf0": monthly.mean() / monthly.std() * np.sqrt(12.0),
    }
    for metric, value in expected.items():
        assert np.isclose(observed[metric], value, rtol=0.0, atol=1e-15)


def test_narrative_registry_is_complete_and_narrow() -> None:
    """The reconciliation registry must contain exactly four comparisons by three metrics."""
    expected = {
        (comparison.comparison, metric)
        for comparison in g0.COMPARISONS
        for metric in g0.f6.METRICS
    }
    assert set(g0.NARRATIVE_DELTAS) == expected
    assert len(expected) == 12


def test_runner_has_no_backtest_or_estimator_entry_point() -> None:
    """The G0 source must remain a statistic-only loader and scorer."""
    source = Path(g0.__file__).read_text(encoding="utf-8")
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
    assert "qis" not in imported
    forbidden_calls = {
        "backtest_model_portfolio",
        "load_cached",
        "compute_rolling_smoothed_clusters",
        "fit",
    }
    called = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
    }
    assert called.isdisjoint(forbidden_calls)


def test_frozen_nav_fingerprints_match_f0() -> None:
    """Both U1 NAV files must retain their F0 content-addressed fingerprints."""
    manifest = g0._source_manifest()
    assert len(manifest) == 2
    assert manifest["fingerprint_match"].all()


def test_g0_artifact_shapes_and_window_convention() -> None:
    """G0 must produce only the four-comparison, 202-month frozen deliverables."""
    artifacts = g0._artifacts()
    performance = artifacts["u1_windowed_performance.csv"]
    cis = artifacts["u1_windowed_cis.csv"]
    reconciliation = artifacts["u1_reconciliation.csv"]
    assert len(performance) == 10
    assert len(cis) == 12
    assert len(reconciliation) == 12
    assert performance["monthly_observations"].eq(202).all()
    assert cis["monthly_observations"].eq(202).all()
    assert pd.to_datetime(performance["sample_start"]).eq(pd.Timestamp("2009-08-26")).all()
    assert pd.to_datetime(performance["sample_end"]).eq(pd.Timestamp("2026-06-24")).all()
    assert float(cis["point_recomputation_error"].max()) <= g0.TOLERANCE
