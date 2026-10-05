"""Subset validation and nonadditive bundle-removal numerical contracts."""
from dataclasses import replace

import numpy as np
import pytest

from optimalportfolios.matf.tests.universe_diagnostics_test import example
from optimalportfolios.matf.universe_diagnostics import (
    bundle_removal_diagnostics, compare_universe_dispersion, removal_diagnostics,
    sample_universe_dispersion,
)


def test_single_asset_universe_cannot_supply_leave_one_out_diagnostics():
    """Removing its sole asset leaves no valid universe for a GLS comparison."""
    m = example()
    one = replace(m, betas=m.betas.iloc[:1], residual_variances=m.residual_variances.iloc[:1],
                  alpha=m.alpha.iloc[:1])
    with pytest.raises(ValueError, match='at least two assets'):
        removal_diagnostics(one)


def test_joint_removal_reconciles_to_independent_centered_sums_of_squares():
    """Two equal negative alphas have nonadditive J/K removal losses of four thirds."""
    m = example()
    before = m.alpha.copy()
    result = bundle_removal_diagnostics(m, {'negative_pair': ['a', 'b']}).iloc[0]
    assert result.bundle == 'negative_pair' and result.removed_N == result.remaining_N == 2
    assert result.rank_after == 1
    for name in ('J', 'K'):
        assert result[name + '_loss'] == pytest.approx(4.)
        assert result[name + '_sum_single_losses'] == pytest.approx(8 / 3)
        assert result[name + '_nonadditivity'] == pytest.approx(4 / 3)
    assert result.A_loss == pytest.approx(2.)
    assert result.A_sum_single_losses == pytest.approx(2.)
    assert result.A_nonadditivity == pytest.approx(0.)
    np.testing.assert_array_equal(m.alpha, before)


@pytest.mark.parametrize('members', [[], ['a', 'a'], ['unknown'], ['a', 'b', 'c', 'd']])
def test_bundle_removal_requires_a_nonempty_remaining_known_universe(members):
    """Ambiguous, unknown or complete removals cannot create a fabricated subset result."""
    with pytest.raises(ValueError):
        bundle_removal_diagnostics(example(), {'invalid': members})


@pytest.mark.parametrize('name', ['', 1])
def test_comparison_scenarios_require_nonempty_string_names(name):
    """Scenario identities must remain suitable for labelled baseline comparisons."""
    m = example()
    with pytest.raises(ValueError, match='nonempty strings'):
        compare_universe_dispersion(m, m.alpha.index, {name: m.alpha.index[:3]})


@pytest.mark.parametrize('repetitions', [True, 0, -1, 1.5])
def test_subset_repetitions_require_a_positive_integer(repetitions):
    """Invalid draw counts do not silently create an empty or truncated sampling study."""
    with pytest.raises(ValueError, match='positive integer'):
        sample_universe_dispersion(example(), [3], repetitions=repetitions)
