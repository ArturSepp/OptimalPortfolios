"""
regression tests for ``round_weights_to_pct``: fully invested weights sum to exactly 100.

The largest-remainder step floors every percentage to ``decimals`` places and hands the shortfall
from 100 to the largest remainders, one unit of ``10**-decimals`` each. The bump count used to be
``int(shortfall * 10**decimals)``; floating-point error put that product just below an integer,
28.999999999999996 for a shortfall of 0.29, and ``int`` dropped a bump. At two decimals that
happened whenever 29, 57, 58 or 113 to 116 bumps were due, so only allocations of thirty or more
assets were exposed.

Each result is checked against an exact reference built with ``fractions.Fraction`` from the same
float weights: every percentage is its exact floor or one unit above it, the units add to exactly
``100 * 10**decimals``, and no asset left at its floor has a larger exact remainder than one that
was bumped.
"""
# packages
from fractions import Fraction
import math

import numpy as np
import pandas as pd
import pytest

# optimalportfolios
from optimalportfolios.utils.portfolio_funcs import round_weights_to_pct

N_DRAWS = 200
REMAINDER_TOL = Fraction(1, 10**9)   # float remainders tie-break within this many units


def _assert_largest_remainder(weights: pd.Series, decimals: int) -> None:
    """Assert the fully invested contract of ``round_weights_to_pct`` against exact arithmetic."""
    pct = round_weights_to_pct(weights, decimals=decimals)
    scale = 10**decimals
    units = np.round(pct.to_numpy() * scale)
    assert np.abs(pct.to_numpy() * scale - units).max() < 1e-6   # values sit on the grid
    units = [int(u) for u in units]
    assert sum(units) == 100 * scale
    assert round(float(pct.sum()), decimals) == 100.0

    exact = [Fraction(float(w)) * 100 * scale for w in weights]
    floors = [math.floor(x) for x in exact]
    bumps = [u - f for u, f in zip(units, floors)]
    assert set(bumps) <= {0, 1}   # within one unit of the exact percentage
    remainders = [x - f for x, f in zip(exact, floors)]
    bumped = [r for r, b in zip(remainders, bumps) if b == 1]
    kept = [r for r, b in zip(remainders, bumps) if b == 0]
    if bumped and kept:
        assert min(bumped) >= max(kept) - REMAINDER_TOL


def test_thirty_assets_whose_floors_fall_29_hundredths_short() -> None:
    """
    the reported failure: 29 bumps were due and 28 were made, so the total was 99.99.

    Each 0.033399 floors to 3.33% with a remainder of 0.0099, larger than the complement's 0.0029,
    so all 29 of them take a bump and the complement stays at its floor.
    """
    weights = pd.Series([0.033399] * 29 + [1 - 0.033399 * 29])
    pct = round_weights_to_pct(weights)
    assert pct.tolist() == [3.34] * 29 + [3.14]
    assert round(float(pct.sum()), 2) == 100.0
    _assert_largest_remainder(weights, decimals=2)


@pytest.mark.parametrize('decimals, expected', [(0, [34.0, 33.0, 33.0]),
                                                (1, [33.4, 33.3, 33.3]),
                                                (2, [33.34, 33.33, 33.33])])
def test_equal_thirds_at_each_precision(decimals: int, expected: list) -> None:
    """naive rounding of three thirds loses one unit; the first of the tied remainders takes it"""
    weights = pd.Series(1.0 / 3.0, index=['a', 'b', 'c'])
    naive = (100.0 * weights).round(decimals)
    assert round(float(naive.sum()), decimals) == round(100.0 - 10**-decimals, decimals)
    pct = round_weights_to_pct(weights, decimals=decimals)
    assert pct.tolist() == expected
    assert list(pct.index) == ['a', 'b', 'c']


@pytest.mark.parametrize('decimals', [0, 1, 2])
def test_random_fully_invested_weights_sum_to_100(decimals: int) -> None:
    """
    random long-only weights summing to one, from 2 to 150 assets, round to exactly 100.

    Two families: continuous Dirichlet draws, and the same draws quoted to six decimals, as a
    weight file or an optimiser report would carry them. Quoted weights sit on the percentage
    grid, where the float product can land a hair below it and floor one unit low; the
    largest-remainder step must give that unit back.
    """
    rng = np.random.default_rng(20260927 + decimals)
    for _ in range(N_DRAWS):
        n = int(rng.integers(2, 151))
        weights = rng.dirichlet(np.ones(n))
        quoted = rng.multinomial(10**6, weights) / 10**6
        _assert_largest_remainder(pd.Series(weights), decimals=decimals)
        _assert_largest_remainder(pd.Series(quoted), decimals=decimals)


def test_weights_summing_above_one_are_left_at_their_floors() -> None:
    """
    floors above 100 leave a negative shortfall, which bumps nothing rather than cutting a value.

    Only weights summing above one get there. The result reports their total, 121.10, instead of
    forcing 100.
    """
    pct = round_weights_to_pct(pd.Series([0.605555, 0.605555]))
    assert pct.tolist() == [60.55, 60.55]


def test_missing_weight_stays_missing() -> None:
    """a NaN weight is reported as NaN, and the others still add to exactly 100"""
    pct = round_weights_to_pct(pd.Series([1 / 3, np.nan, 1 / 3, 1 / 3]))
    assert np.isnan(pct.iloc[1])
    assert pct.dropna().tolist() == [33.34, 33.33, 33.33]
    assert round(float(pct.sum()), 2) == 100.0
