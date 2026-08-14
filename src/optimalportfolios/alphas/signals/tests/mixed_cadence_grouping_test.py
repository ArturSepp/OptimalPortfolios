"""
the mixed-cadence-with-groups branch, across all four signal modules.

Every signal here dispatches on the *type* of ``returns_freq``: a string means one cadence for
the whole universe, a Series means per-asset cadences computed separately and merged. Crossed
with that, ``group_data`` switches scoring from the full cross-section to within-group. The
combination -- a Series cadence *and* group labels -- is the one path where the universe is
partitioned twice, once by cadence and once by group, and each cell is scored on its own.

That double split is where an alignment bug hides. Each cell is scored independently and the
results are concatenated, so a cell that silently drops or duplicates a ticker still produces a
full-looking panel; the merge just fills the gap from another cell or keeps the first copy. The
tests below therefore assert on the *column set* of the merged result and on the group labels
actually used, rather than only on the panel's shape.

Written as one parametrised suite because the four modules implement the identical branch with
different inner signal functions -- and because a fifth signal added later should be able to
join the list rather than acquire its own copy of these cases.
"""
# packages
import pandas as pd
import pytest
import qis
# optimalportfolios
from optimalportfolios.alphas.signals.carry import (
    compute_ra_carry_alpha,
    compute_ra_carry_alphas,
)
from optimalportfolios.alphas.signals.low_beta import compute_low_beta_alpha
from optimalportfolios.alphas.signals.momentum import compute_momentum_alpha
from optimalportfolios.alphas.signals.residual_momentum import compute_residual_momentum_alpha
from optimalportfolios.tests.data.multiasset import load_multiasset_data


@pytest.fixture(scope='module')
def universe() -> dict:
    """A decade of the committed monthly universe, its groups, and a carry panel."""
    data = load_multiasset_data()
    prices = data.prices.loc['2010':'2019', :]
    carry = pd.DataFrame(0.02, index=prices.index, columns=prices.columns)
    carry.iloc[:, :5] = 0.06                    # a stable cross-sectional carry spread
    return dict(prices=prices,
                benchmark_price=prices.iloc[:, 0],
                group_data=data.group_data,
                carry=carry)


def split_cadences(columns: pd.Index) -> pd.Series:
    """Assign half the universe to quarterly and half to monthly cadence."""
    freqs = pd.Series('ME', index=columns)
    freqs.iloc[: len(freqs) // 2] = 'QE'
    return freqs


SPANS = {'ME': 12, 'QE': 4}
SHORT_SPANS = {'ME': 13, 'QE': 4}


def call_signal(name: str, universe: dict, returns_freq, group_data) -> tuple:
    """Invoke one signal by name with cadence-appropriate spans."""
    common = dict(prices=universe['prices'], returns_freq=returns_freq, group_data=group_data)
    if name == 'carry':
        return compute_ra_carry_alpha(carry=universe['carry'], vol_span=SHORT_SPANS, **common)
    if name == 'low_beta':
        return compute_low_beta_alpha(benchmark_price=universe['benchmark_price'],
                                      beta_span=SPANS, **common)
    if name == 'momentum':
        return compute_momentum_alpha(benchmark_price=universe['benchmark_price'],
                                      long_span=SPANS, vol_span=SHORT_SPANS, **common)
    if name == 'residual_momentum':
        return compute_residual_momentum_alpha(benchmark_price=universe['benchmark_price'],
                                               beta_span=SPANS, long_span=SPANS,
                                               vol_span=SHORT_SPANS, **common)
    raise AssertionError(f"unknown signal {name!r}")


SIGNALS = ['carry', 'low_beta', 'momentum', 'residual_momentum']


# --------------------------------------------------------------------------- #
# the mixed-cadence path, grouped and ungrouped
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize('signal', SIGNALS)
def test_mixed_cadences_with_groups_cover_the_whole_universe(signal: str,
                                                             universe: dict) -> None:
    """The double split by cadence and group must merge back to every ticker, exactly once."""
    prices = universe['prices']
    scores, raw = call_signal(signal, universe,
                              returns_freq=split_cadences(prices.columns),
                              group_data=universe['group_data'])
    assert set(scores.columns) == set(prices.columns)
    assert set(raw.columns) == set(prices.columns)
    assert not scores.columns.has_duplicates
    assert not raw.columns.has_duplicates


@pytest.mark.parametrize('signal', SIGNALS)
def test_mixed_cadences_without_groups_cover_the_whole_universe(signal: str,
                                                               universe: dict) -> None:
    """The cadence split alone must also merge back to the full column set."""
    prices = universe['prices']
    scores, _ = call_signal(signal, universe,
                            returns_freq=split_cadences(prices.columns),
                            group_data=None)
    assert set(scores.columns) == set(prices.columns)


@pytest.mark.parametrize('signal', SIGNALS)
def test_grouping_changes_the_score_under_mixed_cadences(signal: str,
                                                         universe: dict) -> None:
    """Within-group scoring is not a no-op: the same raw signal scores differently.

    If ``group_data`` were dropped on the way into the cadence loop, these would be identical
    while every shape assertion above still passed.
    """
    prices = universe['prices']
    freqs = split_cadences(prices.columns)
    grouped, _ = call_signal(signal, universe, returns_freq=freqs,
                             group_data=universe['group_data'])
    ungrouped, _ = call_signal(signal, universe, returns_freq=freqs, group_data=None)
    common = grouped.columns.intersection(ungrouped.columns)
    assert not grouped[common].equals(ungrouped[common])


@pytest.mark.parametrize('signal', SIGNALS)
def test_the_raw_signal_differs_between_cadences(signal: str, universe: dict) -> None:
    """A quarterly asset must not be computed on the monthly cadence.

    Comparing the same asset under an all-monthly panel against a mixed panel that assigns it
    to quarterly is what shows the per-asset cadence actually reached the inner computation.
    """
    prices = universe['prices']
    freqs = split_cadences(prices.columns)
    # not column 0: that is the benchmark itself, whose excess return over itself is zero, so
    # the vol-normalised signal is NaN on every date and every cadence compares equal
    quarterly_ticker = freqs.index[1]
    assert freqs[quarterly_ticker] == 'QE'

    _, mixed_raw = call_signal(signal, universe, returns_freq=freqs, group_data=None)
    _, monthly_raw = call_signal(signal, universe, returns_freq='ME', group_data=None)
    mixed_col = mixed_raw[quarterly_ticker].dropna()
    monthly_col = monthly_raw[quarterly_ticker].dropna()
    assert not mixed_col.empty                           # a vacuous comparison proves nothing
    assert not mixed_col.equals(monthly_col)


@pytest.mark.parametrize('signal', SIGNALS)
def test_a_single_cadence_series_still_takes_the_mixed_path(signal: str,
                                                            universe: dict) -> None:
    """A Series naming one cadence for every asset routes through the merge, not the fast path.

    It should agree with the equivalent scalar call: the merge over a single cell must be the
    identity, which is the cheapest available check that the concatenation preserves order.
    """
    prices = universe['prices']
    uniform = pd.Series('ME', index=prices.columns)
    via_series, _ = call_signal(signal, universe, returns_freq=uniform, group_data=None)
    via_scalar, _ = call_signal(signal, universe, returns_freq='ME', group_data=None)
    assert set(via_series.columns) == set(via_scalar.columns)
    pd.testing.assert_frame_equal(via_series[via_scalar.columns], via_scalar)


# --------------------------------------------------------------------------- #
# the legacy carry entry point
# --------------------------------------------------------------------------- #
def test_the_legacy_carry_entry_point_returns_only_the_score(universe: dict) -> None:
    """``compute_ra_carry_alphas`` is retained for callers of the original signature.

    It returns the score alone rather than the (score, raw) tuple, and forces ungrouped
    scoring regardless of any group structure in the universe.
    """
    score = compute_ra_carry_alphas(prices=universe['prices'], carry=universe['carry'],
                                    returns_freq='ME', vol_span=13)
    assert isinstance(score, pd.DataFrame)
    expected, _ = compute_ra_carry_alpha(prices=universe['prices'], carry=universe['carry'],
                                         returns_freq='ME', group_data=None, vol_span=13)
    pd.testing.assert_frame_equal(score, expected)


def test_the_legacy_carry_entry_point_accepts_a_mean_adjustment(universe: dict) -> None:
    """``mean_adj_type`` is threaded through to the vol normalisation, not defaulted away."""
    common = dict(prices=universe['prices'], carry=universe['carry'], returns_freq='ME',
                  vol_span=13)
    ewma = compute_ra_carry_alphas(mean_adj_type=qis.MeanAdjType.EWMA, **common)
    none_adj = compute_ra_carry_alphas(mean_adj_type=qis.MeanAdjType.NONE, **common)
    assert not ewma.equals(none_adj)
