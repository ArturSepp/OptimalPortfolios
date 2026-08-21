"""Focused tests for the fixed F4 synthetic design and scoring helpers."""

from __future__ import annotations

import numpy as np
import pandas as pd

from papers.cluster_lineage_2026.replication import run_f4_simulation as f4


def test_garch_kappa_matches_closed_form_cells() -> None:
    """The three fixed GARCH cells must reproduce their analytical kappas."""
    assert f4.garch_kappa(0.0, 0.0) == 0.0
    assert np.isclose(f4.garch_kappa(0.05, 0.90), 0.005 / 0.0925)
    assert np.isclose(f4.garch_kappa(0.10, 0.85), 0.02 / 0.0775)


def test_population_correlation_has_requested_blocks() -> None:
    """The synthetic correlation must encode exact within/between levels and be PSD."""
    correlation = f4.population_correlation(50, 5, 0.20)
    assert correlation[0, 1] == 0.40
    assert correlation[0, 10] == 0.20
    assert np.linalg.eigvalsh(correlation).min() > 0.0


def test_fractional_step_pattern_is_exact_over_each_quarter() -> None:
    """The 13/3 schedule must use 4, 4, 5 observations without rounding drift."""
    pattern = f4._step_pattern(13.0 / 3.0)
    assert pattern[:3].tolist() == [4, 4, 5]
    assert pattern.sum() == 8 * 13


def test_overlap_alignment_is_label_invariant() -> None:
    """A pure label permutation must produce zero realised asset flips."""
    previous = np.array([0, 0, 1, 1, 2, 2])
    current = np.array([8, 8, 4, 4, 9, 9])
    aligned, flips = f4._align_to_previous(previous, current)
    assert flips == 0
    assert np.array_equal(aligned, previous)


def test_delta_grid_is_ordered() -> None:
    """The four frozen hysteresis arms must be weakly increasing."""
    deltas = f4._delta_grid(36, 13.0 / 3.0, 0.0, 0.40)
    values = [deltas[label] for label in f4.DELTA_LABELS]
    assert np.all(np.diff(values) > 0.0)


def test_monotonicity_uses_numeric_delta_not_calibration_label_order() -> None:
    """Innovation and level labels may reverse, without creating a false violation."""
    frame = pd.DataFrame(
        {
            "cell_index": [0] * 4,
            "dimension": [50] * 4,
            "groups": [5] * 4,
            "separation": [0.20] * 4,
            "alpha": [0.0] * 4,
            "beta": [0.0] * 4,
            "span": [36] * 4,
            "step": [13.0] * 4,
            "method": ["flat"] * 4,
            "delta_label": ["zero", "innovation", "level", "double_level"],
            "delta": [0.0, 0.20, 0.10, 0.30],
            "predicted_flip_probability": [0.40, 0.20, 0.30, 0.10],
            "realised_flip_probability": [0.41, 0.21, 0.31, 0.11],
        }
    )
    _, violations, table = f4._flat_acceptance(frame)
    assert violations == 0
    assert table.empty
    assert table.columns.tolist()[-1] == "realised_by_delta"


def test_cell_constant_is_fitted_on_zero_and_held_across_delta_arms() -> None:
    """A cell must use one baseline multiplier rather than fitting every arm."""
    frame = pd.DataFrame(
        {
            "cell_index": [0, 0],
            "dimension": [50, 50],
            "groups": [5, 5],
            "separation": [0.20, 0.20],
            "alpha": [0.0, 0.0],
            "beta": [0.0, 0.0],
            "span": [36, 36],
            "step": [13.0, 13.0],
            "method": ["flat", "flat"],
            "delta_label": ["zero", "level"],
            "predicted_flip_probability": [0.20, 0.10],
            "realised_flip_probability": [0.30, 0.12],
        }
    )
    measured = f4._apply_cell_constants(frame)
    np.testing.assert_allclose(measured["fitted_proportionality_constant"], 1.5)
    np.testing.assert_allclose(measured["predicted_flip_probability"], [0.30, 0.15])
    np.testing.assert_allclose(
        measured["expected_total_churn_per_transition"], [15.0, 7.5]
    )
