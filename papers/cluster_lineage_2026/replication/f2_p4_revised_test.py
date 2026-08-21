"""Focused unit tests for the revised P4 scoring helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from papers.cluster_lineage_2026.replication import run_f2_p4_revised as f2


def test_probability_sum_is_averaged_by_date() -> None:
    """Panel length must not change a constant per-date crossing count."""
    frame = pd.DataFrame(
        {
            "date": ["2020-01-31", "2020-01-31", "2020-02-29", "2020-02-29"],
            "margin": [0.0, 0.0, 0.0, 0.0],
        }
    )
    measured = f2._mean_probability_sum(frame, 0.0, np.ones(4))
    assert measured == 1.0


def test_positive_delta_reduces_probability_sum() -> None:
    """A positive partition bonus must reduce the Gaussian crossing prediction."""
    frame = pd.DataFrame(
        {"date": ["2020-01-31"] * 3, "margin": [-0.01, 0.0, 0.01]}
    )
    sigma = np.full(3, 0.05)
    assert f2._mean_probability_sum(frame, 0.05, sigma) < f2._mean_probability_sum(
        frame, 0.0, sigma
    )
