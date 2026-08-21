"""Focused tests for U1 covariance-grid long-short diagnostics."""
import pandas as pd

from papers.cluster_lineage_2026.replication.run_u1_covar_grid_long_short import (
    _group_exposure_panel,
)


def test_group_exposure_panel_distinguishes_neutral_and_cross_group_books() -> None:
    """Show that dollar neutrality alone does not imply cluster neutrality."""
    dates = pd.DatetimeIndex(["2026-01-31"])
    columns = list("abcd")
    groups = pd.DataFrame(
        [["g1", "g1", "g2", "g2"]], index=dates, columns=columns
    )
    cluster_neutral = pd.DataFrame(
        [[0.5, -0.5, 0.5, -0.5]], index=dates, columns=columns
    )
    globally_neutral = pd.DataFrame(
        [[0.5, 0.5, -0.5, -0.5]], index=dates, columns=columns
    )

    cluster_result = _group_exposure_panel(cluster_neutral, groups).iloc[0]
    global_result = _group_exposure_panel(globally_neutral, groups).iloc[0]

    assert cluster_result["group_l1_net_exposure"] == 0.0
    assert cluster_result["largest_abs_group_net_exposure"] == 0.0
    assert global_result["group_l1_net_exposure"] == 2.0
    assert global_result["largest_abs_group_net_exposure"] == 1.0
