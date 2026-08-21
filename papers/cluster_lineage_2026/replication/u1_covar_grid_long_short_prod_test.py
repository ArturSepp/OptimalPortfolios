"""Focused horizon and causality tests for the U1 production-momentum grid."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from papers.cluster_lineage_2026.replication.run_u1_covar_grid_long_short_prod import (
    EXACT_VARIANT,
    LONG_SPANS,
    MIN_CLUSTER_SIZE,
    SCALED_VARIANT,
    SignalSpec,
    _asof_panel,
    _base_signals,
    _panel_dict,
    _signal_parameter_table,
    _signal_spec,
)


def test_calendar_scaled_long_spans_are_exactly_twelve_months() -> None:
    """Require every cadence-scaled long filter to represent twelve months."""
    parameters = _signal_parameter_table()
    scaled = parameters.loc[parameters["signal_variant"].eq(SCALED_VARIANT)]
    assert set(scaled["covariance_frequency"]) == set(LONG_SPANS)
    assert scaled["long_horizon_months"].eq(12.0).all()
    assert scaled["min_cluster_size"].eq(MIN_CLUSTER_SIZE).all()
    assert scaled["short_span"].isna().all()
    assert scaled["mean_adj_type"].eq("NONE").all()


def test_exact_monthly_control_is_invariant_to_covariance_cadence() -> None:
    """Require the faithful production control to stay at monthly 12/13."""
    specs = [_signal_spec(EXACT_VARIANT, frequency) for frequency in LONG_SPANS]
    assert {(spec.frequency, spec.long_span, spec.vol_span) for spec in specs} == {
        ("ME", 12, 13)
    }
    assert _signal_spec(EXACT_VARIANT, "ME").cache_key == _signal_spec(
        SCALED_VARIANT, "ME"
    ).cache_key
    with pytest.raises(KeyError):
        _signal_spec("unknown", "ME")


def test_asof_sampling_never_uses_a_future_signal_timestamp() -> None:
    """Sample irregular signal dates and prove every chosen row is causal."""
    source_dates = pd.to_datetime(["2020-01-03", "2020-01-10", "2020-01-17"])
    decisions = pd.to_datetime(["2020-01-03", "2020-01-08", "2020-01-16"])
    panel = pd.DataFrame({"asset": [1.0, 2.0, 3.0]}, index=source_dates)
    sampled, timestamps = _asof_panel(panel, decisions)
    assert sampled["asset"].tolist() == [1.0, 1.0, 2.0]
    assert timestamps.tolist() == [
        pd.Timestamp("2020-01-03"),
        pd.Timestamp("2020-01-03"),
        pd.Timestamp("2020-01-10"),
    ]
    assert (timestamps <= timestamps.index).all()


def test_production_score_is_invariant_to_future_price_changes() -> None:
    """Perturb all future NAVs and preserve the already-formed production score."""
    dates = pd.date_range("2018-01-31", periods=18, freq="ME")
    returns = pd.DataFrame(
        {
            "a": np.linspace(0.003, 0.024, len(dates)),
            "b": np.linspace(0.020, -0.004, len(dates)),
            "c": 0.006 + 0.004 * np.sin(np.arange(len(dates))),
        },
        index=dates,
    )
    prices = np.exp(returns.cumsum())
    benchmark = np.exp(returns.mean(axis=1).cumsum()).rename("EW")
    spec = SignalSpec("test", "ME", 4, 5)
    score, raw = _base_signals(prices, benchmark, spec)
    formation = dates[10]
    changed_prices = prices.copy()
    changed_benchmark = benchmark.copy()
    changed_prices.loc[dates[11]:] *= np.array([1.7, 0.6, 1.3])
    changed_benchmark.loc[dates[11]:] *= 1.4
    changed_score, changed_raw = _base_signals(
        changed_prices,
        changed_benchmark,
        spec,
    )
    pd.testing.assert_series_equal(
        score.loc[formation],
        changed_score.loc[formation],
    )
    pd.testing.assert_series_equal(
        raw.loc[formation],
        changed_raw.loc[formation],
    )


def test_panel_dict_drops_unassigned_assets() -> None:
    """Pass only assigned assets to the rolling production-scoring API."""
    panel = pd.DataFrame(
        [[1.0, np.nan], [2.0, 3.0]],
        index=pd.to_datetime(["2020-01-31", "2020-02-29"]),
        columns=["a", "b"],
    )
    converted = _panel_dict(panel)
    assert converted[pd.Timestamp("2020-01-31")].index.tolist() == ["a"]
    assert converted[pd.Timestamp("2020-02-29")].index.tolist() == ["a", "b"]
