"""Classic fixed-window momentum and its standard and cluster score paths.

Classic momentum is not an EWMA parameterisation.  Its raw signal is the sum
of a fixed number of completed log returns after a hard skip.  The tests anchor
that definition on an explicit price-ratio calculation, then check that fixed
groups and rolling clusters alter only the cross-sectional scoring layer.
"""
import numpy as np
import pandas as pd
import pytest
import qis

import optimalportfolios as op
from optimalportfolios.alphas.signals.classic_momentum import (
    compute_classic_momentum_from_returns,
    compute_classic_momentum_alpha,
    compute_classic_momentum_cluster_alpha,
)
from optimalportfolios.alphas.signals.utils import score_within_clusters


def prices() -> pd.DataFrame:
    """Return a deterministic monthly panel with distinct cross-sectional trends."""
    dates = pd.date_range('2018-01-31', periods=30, freq='ME')
    time = np.arange(len(dates), dtype=float)
    log_returns = pd.DataFrame(
        {
            'a': 0.005 + 0.0002 * time,
            'b': 0.010 - 0.0001 * time,
            'c': -0.002 + 0.0003 * time,
            'd': 0.003 + 0.0020 * np.sin(time),
            'e': -0.001 + 0.0015 * np.cos(time),
            'f': 0.007 - 0.0010 * np.sin(time),
        },
        index=dates,
    )
    return 100.0 * np.exp(log_returns.cumsum())


def explicit_raw(
        navs: pd.DataFrame,
        lookback_periods: int = 12,
        skip_periods: int = 1,
) -> pd.DataFrame:
    """Compute the reference signal directly from adjacent log price ratios."""
    log_returns = np.log(navs).diff()
    return log_returns.shift(skip_periods).rolling(
        lookback_periods, min_periods=lookback_periods
    ).sum()


def test_raw_signal_is_exact_fixed_window_momentum_with_a_hard_skip() -> None:
    """The package result must equal the independent 12-return, one-skip sum."""
    navs = prices()
    _, raw = compute_classic_momentum_alpha(
        prices=navs, returns_freq='ME', lookback_periods=12, skip_periods=1
    )
    expected = explicit_raw(navs)

    pd.testing.assert_frame_equal(raw, expected, atol=1e-14, rtol=0.0)


def test_return_panel_entry_point_preserves_missing_history_exactly() -> None:
    """The direct return API retains the source panel's exact finite-window mask."""
    navs = prices()
    returns = np.log(navs).diff()
    returns.iloc[17, 0] = np.nan
    raw = compute_classic_momentum_from_returns(returns)
    expected = returns.shift(1).rolling(12, min_periods=12).sum()

    pd.testing.assert_frame_equal(raw, expected)


def test_the_skipped_return_cannot_move_the_formation_date_signal() -> None:
    """Changing only the latest price leaves the latest skipped-month signal fixed."""
    navs = prices()
    _, base = compute_classic_momentum_alpha(prices=navs)
    perturbed = navs.copy()
    perturbed.iloc[-1, :] *= np.arange(2.0, 8.0)
    _, changed = compute_classic_momentum_alpha(prices=perturbed)

    pd.testing.assert_series_equal(base.iloc[-1], changed.iloc[-1])


def test_the_last_included_return_does_move_the_signal() -> None:
    """Changing the prior price changes the latest score, proving the window is live."""
    navs = prices()
    _, base = compute_classic_momentum_alpha(prices=navs)
    perturbed = navs.copy()
    perturbed.iloc[-2, 0] *= 1.5
    _, changed = compute_classic_momentum_alpha(prices=perturbed)

    assert changed.iloc[-1, 0] != base.iloc[-1, 0]


def test_fixed_groups_change_scores_but_not_raw_momentum() -> None:
    """Fixed-group and global constructors share one raw signal exactly."""
    navs = prices()
    groups = pd.Series(['x', 'x', 'x', 'y', 'y', 'y'], index=navs.columns)
    global_score, global_raw = compute_classic_momentum_alpha(prices=navs)
    group_score, group_raw = compute_classic_momentum_alpha(
        prices=navs, group_data=groups
    )

    pd.testing.assert_frame_equal(group_raw, global_raw)
    assert not group_score.equals(global_score)


def test_cluster_constructor_changes_only_the_scoring_layer() -> None:
    """The cluster entry point must reuse raw classic momentum without alteration."""
    navs = prices()
    assignment = pd.Series(['x', 'x', 'x', 'y', 'y', 'y'], index=navs.columns)
    rolling_clusters = {navs.index[0]: assignment}
    _, standard_raw = compute_classic_momentum_alpha(prices=navs)
    cluster_score, cluster_raw = compute_classic_momentum_cluster_alpha(
        prices=navs,
        rolling_clusters=rolling_clusters,
        min_cluster_size=2,
    )
    expected_score = score_within_clusters(
        raw_signal=standard_raw,
        rolling_clusters=rolling_clusters,
        min_cluster_size=2,
    )

    pd.testing.assert_frame_equal(cluster_raw, standard_raw)
    pd.testing.assert_frame_equal(cluster_score, expected_score)


def test_mixed_cadence_mappings_cover_and_order_every_asset() -> None:
    """Per-cadence lookback and skip mappings merge back without column drift."""
    navs = prices()
    frequencies = pd.Series(
        ['ME', 'ME', 'ME', 'QE', 'QE', 'QE'], index=navs.columns
    )
    score, raw = compute_classic_momentum_alpha(
        prices=navs,
        returns_freq=frequencies,
        lookback_periods={'ME': 12, 'QE': 4},
        skip_periods={'ME': 1, 'QE': 1},
    )

    assert list(score.columns) == list(navs.columns)
    assert list(raw.columns) == list(navs.columns)
    assert score.notna().any().all()
    assert raw.notna().any().all()


@pytest.mark.parametrize(
    'kwargs',
    [
        {'lookback_periods': 0},
        {'lookback_periods': True},
        {'skip_periods': -1},
        {'skip_periods': True},
        {'returns_freq': 'QE', 'lookback_periods': {'ME': 12}},
        {'returns_freq': 'QE', 'skip_periods': {'ME': 1}},
    ],
)
def test_invalid_or_uncovered_period_settings_raise(kwargs: dict) -> None:
    """Invalid periods and cadence gaps fail instead of changing the horizon."""
    with pytest.raises(ValueError):
        compute_classic_momentum_alpha(prices=prices(), **kwargs)


def test_public_exports_are_the_signal_module_objects() -> None:
    """Both constructors are reachable through the package's public surface."""
    assert op.compute_classic_momentum_alpha is compute_classic_momentum_alpha
    assert op.compute_classic_momentum_cluster_alpha is compute_classic_momentum_cluster_alpha
    assert op.compute_classic_momentum_from_returns is compute_classic_momentum_from_returns


def test_global_score_uses_the_canonical_qis_cross_sectional_transform() -> None:
    """The new raw calculation still uses the stack's existing score transform."""
    score, raw = compute_classic_momentum_alpha(prices=prices())
    expected = qis.df_to_cross_sectional_score(df=raw)

    pd.testing.assert_frame_equal(score, expected)
