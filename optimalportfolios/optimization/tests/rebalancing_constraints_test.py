"""Tests for current/model implementation-corridor constraint construction."""

import pandas as pd
import pytest

from optimalportfolios.optimization.constraints import (
    compute_eligible_rebalancing_bounds,
)


def test_compute_eligible_rebalancing_bounds_projects_limits_into_corridor() -> None:
    """Workbook limits narrow the current/model corridor and never expand it."""
    assets = pd.Index(['SPDR Bank', 'Below Min', 'Above Max', 'Absent'])
    eligible_min, eligible_max, rebalancing = compute_eligible_rebalancing_bounds(
        current_weights=pd.Series([0.0, 0.0744, 0.05058, 0.0], index=assets),
        model_weights=pd.Series([0.002378, 0.075, 0.05, 0.0], index=assets),
        current_min_weights=pd.Series([0.0, 0.075, 0.0, 0.0], index=assets),
        current_max_weights=pd.Series([0.02, 0.10, 0.05, 0.05], index=assets),
    )

    pd.testing.assert_series_equal(
        eligible_min,
        pd.Series([0.0, 0.075, 0.05, 0.0], index=assets),
    )
    pd.testing.assert_series_equal(
        eligible_max,
        pd.Series([0.002378, 0.075, 0.05, 0.0], index=assets),
    )
    pd.testing.assert_series_equal(
        rebalancing,
        pd.Series([1, 1, 1, 0], index=assets),
    )


def test_compute_eligible_rebalancing_bounds_rejects_inverted_limits() -> None:
    """An invalid candidate minimum/maximum pair is rejected before solving."""
    assets = pd.Index(['Asset'])

    with pytest.raises(
        ValueError,
        match='current_min_weights exceeds current_max_weights',
    ):
        compute_eligible_rebalancing_bounds(
            current_weights=pd.Series([0.0], index=assets),
            model_weights=pd.Series([0.01], index=assets),
            current_min_weights=pd.Series([0.02], index=assets),
            current_max_weights=pd.Series([0.01], index=assets),
        )
