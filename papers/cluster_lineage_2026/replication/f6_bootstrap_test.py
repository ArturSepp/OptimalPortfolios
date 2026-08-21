"""Focused tests for the F6 joint moving-block bootstrap."""

from __future__ import annotations

import numpy as np
import pandas as pd

from papers.cluster_lineage_2026.replication import run_f6_bootstrap as f6


def test_moving_blocks_are_circular_and_contiguous() -> None:
    """Every six-position block must advance by one modulo sample length."""
    indices = f6._mbb_indices(11, np.random.default_rng(7))
    assert indices.shape == (f6.BOOTSTRAP_DRAWS, 11)
    first_block = indices[:, : f6.BLOCK_LENGTH]
    assert np.all(np.diff(first_block, axis=1) % 11 == 1)


def test_identical_legs_have_zero_bootstrap_deltas() -> None:
    """Joint resampling of identical legs must yield exactly zero for all metrics."""
    values = np.linspace(-0.02, 0.03, 24)
    indices = f6._mbb_indices(len(values), np.random.default_rng(11))
    candidate = f6._bootstrap_metrics(values[indices])
    benchmark = f6._bootstrap_metrics(values[indices])
    assert np.array_equal(candidate - benchmark, np.zeros_like(candidate))


def test_nav_metrics_use_month_end_simple_returns() -> None:
    """The point estimator must follow the frozen month-end NAV convention."""
    index = pd.date_range("2020-01-31", periods=13, freq="ME")
    nav = pd.Series(100.0 * 1.01 ** np.arange(13), index=index)
    measured, monthly = f6._nav_metrics(nav)
    assert len(monthly) == 12
    years = (index[-1] - index[0]).days / 365.25
    assert np.isclose(measured[0], (nav.iloc[-1] / nav.iloc[0]) ** (1.0 / years) - 1.0)
