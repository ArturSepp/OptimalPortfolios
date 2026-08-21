"""Focused tests for the F5 scorecard bootstrap and serialization helpers."""

from __future__ import annotations

import json

import numpy as np

from papers.cluster_lineage_2026.replication import run_f5_scorecard as f5


def test_moving_blocks_wrap_circularly_and_have_frozen_shape() -> None:
    """Every draw must contain n valid circular block indices."""
    indices = f5._mbb_indices(11, np.random.default_rng(7))
    assert indices.shape == (f5.BOOTSTRAP_DRAWS, 11)
    assert indices.min() == 0
    assert indices.max() == 10


def test_stable_rng_is_keyed_not_call_order_dependent() -> None:
    """Equal semantic keys must reproduce equal bootstrap streams."""
    first = f5._stable_rng("P7", "equity").integers(0, 100, size=20)
    f5._stable_rng("unrelated").integers(0, 100, size=20)
    second = f5._stable_rng("P7", "equity").integers(0, 100, size=20)
    np.testing.assert_array_equal(first, second)


def test_json_cells_are_stable_and_finite() -> None:
    """Scorecard JSON must sort keys and reject non-finite values."""
    assert f5._json({"b": 2, "a": 1}) == '{"a":1,"b":2}'
    try:
        f5._json({"bad": float("nan")})
    except ValueError:
        pass
    else:
        raise AssertionError("NaN must not enter a scorecard cell")
    assert json.loads(f5._json({"value": 1.5})) == {"value": 1.5}
