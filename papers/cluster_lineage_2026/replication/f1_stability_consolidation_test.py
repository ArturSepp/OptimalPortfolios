"""Focused unit tests for manuscript-finalisation stage F1."""

from __future__ import annotations

import numpy as np

from papers.cluster_lineage_2026.replication import run_f1_stability_consolidation as f1


def test_noise_floor_formula_reproduces_owner_frozen_equity_values() -> None:
    """The published formula must round to both frozen U1 calibration markers."""
    level, innovation = f1._formula(156, 52.0 / 12.0, 2.124418, 0.622741298852166)
    assert round(level, 4) == 0.0866
    assert round(innovation, 4) == 0.0285


def test_moving_block_indices_are_circular_and_contiguous() -> None:
    """Every sampled block must advance one index modulo the sample length."""
    indices = f1._mbb_indices(17, np.random.default_rng(7))
    assert indices.shape == (f1.BOOTSTRAP_DRAWS, 17)
    for start in range(0, 17, f1.BLOCK_LENGTH):
        block = indices[:, start : min(start + f1.BLOCK_LENGTH, 17)]
        if block.shape[1] > 1:
            assert np.all(np.diff(block, axis=1) % 17 == 1)


def test_menger_curvature_marks_the_interior_bend() -> None:
    """A right-angle polyline has its unique curvature maximum at the bend."""
    points = np.array([[0.0, 0.0], [0.5, 0.0], [0.5, 0.5], [0.5, 1.0]])
    curvature = f1._menger_curvature(points)
    assert int(np.argmax(curvature)) == 1
    assert curvature[0] == curvature[-1] == 0.0
