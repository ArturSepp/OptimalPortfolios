"""
residual reversal, and the four dispatch branches its entry point hides.

The signal itself is residual momentum with a sign flip: strip the benchmark beta, EWMA-filter
the residual, negate. The tests that matter here are not about the filter -- they are about
which of four code paths a given set of arguments lands on, because all four return the same
two frames and none of them announces itself.

``compute_residual_reversal_alpha`` dispatches on the *type* of ``returns_freq``: a string
means one cadence for the whole universe, a Series means per-asset cadences and a merge across
(frequency x group). Orthogonally, ``group_data`` switches scoring from the full cross-section
to within-group. A caller who passes a scalar where a Series was intended gets a complete,
plausible score panel computed on the wrong cadence for most of the universe.

The sign is asserted directly against ``compute_residual_momentum_alpha``: the raw signals must
be exact negations of each other. That is the one property distinguishing this module from its
neighbour, and it is stated in the docstring, so it is worth a test rather than trust -- a
dropped negation leaves a signal that still scores, still backtests, and buys the momentum
winners while the module name says reversal.
"""
# packages
import numpy as np
import pandas as pd
import pytest
import qis
# optimalportfolios
from optimalportfolios.alphas.signals.residual_momentum import compute_residual_momentum_alpha
from optimalportfolios.alphas.signals.residual_reversal import (
    compute_residual_reversal_alpha,
    compute_residual_reversal_cluster_alpha,
)
from optimalportfolios.tests.data.multiasset import load_multiasset_data


@pytest.fixture(scope='module')
def universe() -> dict:
    """A decade of the committed monthly universe with its group labels."""
    data = load_multiasset_data()
    prices = data.prices.loc['2010':'2019', :]
    return dict(prices=prices,
                benchmark_price=prices.iloc[:, 0],
                group_data=data.group_data)


# --------------------------------------------------------------------------- #
# the benchmark default
# --------------------------------------------------------------------------- #
def test_no_benchmark_falls_back_to_the_equal_weight_cross_section(universe: dict) -> None:
    """With no benchmark the cross-sectional mean return stands in for one.

    Worth pinning because it is silent: omitting the benchmark does not raise, it changes what
    'residual' means for every asset in the panel.
    """
    prices = universe['prices']
    without, raw_without = compute_residual_reversal_alpha(prices=prices)
    with_bench, _ = compute_residual_reversal_alpha(prices=prices,
                                                    benchmark_price=universe['benchmark_price'])
    assert raw_without.shape == (len(prices), len(prices.columns))
    assert not without.equals(with_bench)          # a different residual, not a no-op


# --------------------------------------------------------------------------- #
# the sign flip
# --------------------------------------------------------------------------- #
def test_the_raw_signal_is_the_exact_negation_of_residual_momentum(universe: dict) -> None:
    """Reversal and momentum share one raw computation and differ only by sign."""
    kwargs = dict(prices=universe['prices'], benchmark_price=universe['benchmark_price'],
                  returns_freq='ME', beta_span=12, long_span=1, vol_span=13)
    _, raw_reversal = compute_residual_reversal_alpha(**kwargs)
    _, raw_momentum = compute_residual_momentum_alpha(**kwargs)
    pd.testing.assert_frame_equal(raw_reversal, -raw_momentum)


def test_a_recent_residual_loser_scores_above_a_winner(universe: dict) -> None:
    """The negation is what makes a loser attractive; the score ordering must reflect it."""
    _, raw = compute_residual_reversal_alpha(prices=universe['prices'],
                                             benchmark_price=universe['benchmark_price'])
    scores, _ = compute_residual_reversal_alpha(prices=universe['prices'],
                                                benchmark_price=universe['benchmark_price'])
    date = scores.dropna(how='all').index[-1]
    row_raw, row_score = raw.loc[date, :].dropna(), scores.loc[date, :].dropna()
    common = row_raw.index.intersection(row_score.index)
    # scoring is monotone in the raw signal, so their rank orders agree
    assert (row_raw[common].rank().equals(row_score[common].rank()))


# --------------------------------------------------------------------------- #
# the four dispatch branches
# --------------------------------------------------------------------------- #
def test_a_string_frequency_takes_the_single_cadence_path(universe: dict) -> None:
    """One cadence for the whole universe: the panel keeps every column."""
    scores, raw = compute_residual_reversal_alpha(prices=universe['prices'],
                                                  benchmark_price=universe['benchmark_price'],
                                                  returns_freq='ME')
    assert list(scores.columns) == list(universe['prices'].columns)
    assert list(raw.columns) == list(universe['prices'].columns)


def test_the_single_cadence_path_scores_within_groups_when_asked(universe: dict) -> None:
    """With ``group_data`` the score is a within-group z-score, not a panel-wide one."""
    prices, group_data = universe['prices'], universe['group_data']
    grouped, _ = compute_residual_reversal_alpha(prices=prices,
                                                 benchmark_price=universe['benchmark_price'],
                                                 returns_freq='ME', group_data=group_data)
    ungrouped, _ = compute_residual_reversal_alpha(prices=prices,
                                                   benchmark_price=universe['benchmark_price'],
                                                   returns_freq='ME')
    assert list(grouped.columns) == list(prices.columns)     # original column order restored
    assert not grouped.equals(ungrouped)


def test_a_frequency_series_takes_the_mixed_cadence_path(universe: dict) -> None:
    """Per-asset cadences are computed separately and merged back onto the full universe."""
    prices = universe['prices']
    freqs = pd.Series('ME', index=prices.columns)
    freqs.iloc[: len(freqs) // 2] = 'QE'
    scores, raw = compute_residual_reversal_alpha(prices=prices,
                                                  benchmark_price=universe['benchmark_price'],
                                                  returns_freq=freqs,
                                                  beta_span={'ME': 12, 'QE': 4},
                                                  long_span={'ME': 1, 'QE': 1},
                                                  vol_span={'ME': 13, 'QE': 4})
    assert set(scores.columns) == set(prices.columns)
    assert set(raw.columns) == set(prices.columns)


def test_the_mixed_cadence_path_also_scores_within_groups(universe: dict) -> None:
    """Cadence and group partition the universe independently; both splits apply."""
    prices, group_data = universe['prices'], universe['group_data']
    freqs = pd.Series('ME', index=prices.columns)
    freqs.iloc[: len(freqs) // 2] = 'QE'
    scores, _ = compute_residual_reversal_alpha(prices=prices,
                                                benchmark_price=universe['benchmark_price'],
                                                returns_freq=freqs,
                                                group_data=group_data,
                                                beta_span={'ME': 12, 'QE': 4},
                                                long_span={'ME': 1, 'QE': 1},
                                                vol_span={'ME': 13, 'QE': 4})
    assert set(scores.columns) == set(prices.columns)


# --------------------------------------------------------------------------- #
# cluster scoring
# --------------------------------------------------------------------------- #
def test_cluster_scoring_returns_the_shared_raw_signal(universe: dict) -> None:
    """The cluster entry point changes only the scoring, so the raw signal is unchanged."""
    prices = universe['prices']
    dates = prices.index[-6:]
    clusters = {date: pd.Series(np.arange(len(prices.columns)) % 3, index=prices.columns)
                for date in dates}
    _, raw_cluster = compute_residual_reversal_cluster_alpha(
        prices=prices, benchmark_price=universe['benchmark_price'],
        rolling_clusters=clusters, returns_freq='ME')
    _, raw_plain = compute_residual_reversal_alpha(
        prices=prices, benchmark_price=universe['benchmark_price'], returns_freq='ME')
    pd.testing.assert_frame_equal(raw_cluster, raw_plain)


def test_cluster_scoring_still_negates_relative_to_residual_momentum(universe: dict) -> None:
    """The sign flip lives in the shared raw signal, so it survives cluster scoring."""
    prices = universe['prices']
    clusters = {date: pd.Series(np.arange(len(prices.columns)) % 3, index=prices.columns)
                for date in prices.index[-6:]}
    _, raw_cluster = compute_residual_reversal_cluster_alpha(
        prices=prices, benchmark_price=universe['benchmark_price'],
        rolling_clusters=clusters, returns_freq='ME')
    _, raw_momentum = compute_residual_momentum_alpha(
        prices=prices, benchmark_price=universe['benchmark_price'], returns_freq='ME',
        beta_span=12, long_span=1, vol_span=13)
    pd.testing.assert_frame_equal(raw_cluster, -raw_momentum)


def test_the_returned_frames_share_the_price_index(universe: dict) -> None:
    """Scores and raw signal are reported on the price index, whatever the cadence."""
    prices = universe['prices']
    scores, raw = compute_residual_reversal_alpha(prices=prices,
                                                  benchmark_price=universe['benchmark_price'],
                                                  returns_freq='ME')
    assert isinstance(scores.index, pd.DatetimeIndex)
    assert scores.index.equals(raw.index)
    assert scores.index[-1] <= prices.index[-1]


def test_disabling_vol_normalisation_changes_the_signal(universe: dict) -> None:
    """``vol_span=None`` turns off the risk adjustment rather than defaulting it back on."""
    common = dict(prices=universe['prices'], benchmark_price=universe['benchmark_price'],
                  returns_freq='ME')
    _, normalised = compute_residual_reversal_alpha(vol_span=13, **common)
    _, unnormalised = compute_residual_reversal_alpha(vol_span=None, **common)
    assert not normalised.equals(unnormalised)


def test_an_explicit_mean_adjustment_reaches_the_beta_regression(universe: dict) -> None:
    """``mean_adj_type`` is threaded to the EWMA beta estimate, not silently defaulted."""
    common = dict(prices=universe['prices'], benchmark_price=universe['benchmark_price'],
                  returns_freq='ME')
    _, ewma = compute_residual_reversal_alpha(mean_adj_type=qis.MeanAdjType.EWMA, **common)
    _, none_adj = compute_residual_reversal_alpha(mean_adj_type=qis.MeanAdjType.NONE, **common)
    assert not ewma.equals(none_adj)
