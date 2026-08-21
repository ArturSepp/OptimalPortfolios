"""Focused no-look-ahead tests for classic monthly 12-minus-1 momentum."""
import numpy as np
import pandas as pd

from papers.cluster_lineage_2026.replication.run_u1_covar_grid_long_short_monthly import (
    _classic_monthly_scores,
)


def test_classic_monthly_score_includes_twelve_returns_and_skips_latest() -> None:
    """Use months 1..12 at month 13 and exclude the month-13 return."""
    dates = pd.date_range("2020-01-31", periods=14, freq="ME")
    returns = pd.DataFrame(
        {"asset": np.arange(1.0, 15.0)},
        index=dates,
    )

    scores = _classic_monthly_scores(returns, dates)

    assert np.isnan(scores.loc[dates[11], "asset"])
    assert scores.loc[dates[12], "asset"] == sum(range(1, 13))
    assert scores.loc[dates[13], "asset"] == sum(range(2, 14))


def test_classic_monthly_score_is_invariant_to_skipped_and_future_returns() -> None:
    """Perturb the excluded month and future data without changing formation score."""
    dates = pd.date_range("2020-01-31", periods=15, freq="ME")
    returns = pd.DataFrame(0.01, index=dates, columns=["asset"])
    formation_date = dates[12]
    base = _classic_monthly_scores(returns, dates).loc[formation_date, "asset"]

    perturbed = returns.copy()
    perturbed.loc[formation_date, "asset"] = 100.0
    perturbed.loc[dates[13]:, "asset"] = -100.0
    changed = _classic_monthly_scores(perturbed, dates).loc[formation_date, "asset"]

    assert base == changed
