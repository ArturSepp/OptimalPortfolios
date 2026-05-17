"""Tests for the frozen-overshoot group-bound relaxation in
``Constraints.update_with_valid_tickers``.

Reproduces the production failure mode that surfaced when drift was enabled
on the MAC fund: drifted weights_0 pushes frozen Alternatives assets above
group_max_allocation, ``__post_init__`` Check 2 raises ``Infeasible
constraints``.

The fix relaxes group_max_allocation upward (or group_min_allocation
downward) for the offending group, emits a warning, and lets the
optimisation proceed.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from optimalportfolios.optimization.constraints import (
    Constraints,
    GroupLowerUpperConstraints,
)


# -----------------------------------------------------------------------------
# fixtures
# -----------------------------------------------------------------------------

def _simple_alts_setup():
    """Five assets: 3 in 'Alts' (1 frozen, 2 tradable), 2 in 'Equity'."""
    tickers = ['PE', 'HF', 'REIT', 'EQ_US', 'EQ_EU']
    loadings = pd.DataFrame(
        {
            'Alts':   [1.0, 1.0, 1.0, 0.0, 0.0],
            'Equity': [0.0, 0.0, 0.0, 1.0, 1.0],
        },
        index=tickers,
    )
    gluc = GroupLowerUpperConstraints(
        group_loadings=loadings,
        group_min_allocation=pd.Series({'Alts': 0.30, 'Equity': 0.30}),
        group_max_allocation=pd.Series({'Alts': 0.60, 'Equity': 0.70}),
    )
    return tickers, gluc


# -----------------------------------------------------------------------------
# tests
# -----------------------------------------------------------------------------

def test_frozen_overshoot_raises_group_max():
    """Drifted frozen PE pushes Alts above 0.60 → group_max relaxed, no error."""
    tickers, gluc = _simple_alts_setup()
    constraints = Constraints(
        is_long_only=True,
        min_weights=pd.Series(0.0, index=tickers),
        max_weights=pd.Series(1.0, index=tickers),
        group_lower_upper_constraints=gluc,
    )
    # PE drifted above its prior target; HF and REIT also drift slightly.
    # Frozen sum within Alts = 0.20 + 0.21 + 0.20 = 0.61 (overshoot vs 0.60).
    drifted_w0 = pd.Series({
        'PE':    0.20,
        'HF':    0.21,   # drifted up
        'REIT':  0.20,
        'EQ_US': 0.20,
        'EQ_EU': 0.19,
    })
    # all 3 Alts assets frozen; equity legs rebalance.
    rebal = pd.Series({'PE': 0, 'HF': 0, 'REIT': 0, 'EQ_US': 1, 'EQ_EU': 1})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        new_constraints = constraints.update_with_valid_tickers(
            valid_tickers=tickers,
            weights_0=drifted_w0,
            rebalancing_indicators=rebal,
        )

    # the relaxation message must surface
    relax_msgs = [w for w in caught
                  if 'frozen-min overshoot' in str(w.message)]
    assert len(relax_msgs) == 1, \
        f"expected one relaxation warning, got {[str(w.message) for w in caught]}"
    assert "group 'Alts'" in str(relax_msgs[0].message)

    # group_max_allocation for Alts was raised; Equity untouched.
    # Frozen-min-sum within Alts = 0.20 + 0.21 + 0.20 = 0.61, so new cap ≈ 0.61 + tol.
    new_gmax = new_constraints.group_lower_upper_constraints.group_max_allocation
    assert 0.60 < new_gmax['Alts'] < 0.62
    assert new_gmax['Equity'] == pytest.approx(0.70)


def test_frozen_undershoot_lowers_group_min():
    """Drifted frozen assets undershoot Equity floor → group_min lowered."""
    tickers, gluc = _simple_alts_setup()
    constraints = Constraints(
        is_long_only=True,
        min_weights=pd.Series(0.0, index=tickers),
        max_weights=pd.Series(0.50, index=tickers),  # cap each at 50%
        group_lower_upper_constraints=gluc,
    )
    # EQ_US frozen low; EQ_EU frozen low. Their max sum < group_min (0.30).
    drifted_w0 = pd.Series({
        'PE':    0.20,
        'HF':    0.20,
        'REIT':  0.20,
        'EQ_US': 0.10,   # frozen
        'EQ_EU': 0.05,   # frozen, undershoot
    })
    rebal = pd.Series({'PE': 1, 'HF': 1, 'REIT': 1, 'EQ_US': 0, 'EQ_EU': 0})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        new_constraints = constraints.update_with_valid_tickers(
            valid_tickers=tickers,
            weights_0=drifted_w0,
            rebalancing_indicators=rebal,
        )

    undershoot_msgs = [w for w in caught
                       if 'frozen-max undershoot' in str(w.message)]
    assert len(undershoot_msgs) == 1
    assert "group 'Equity'" in str(undershoot_msgs[0].message)

    # group_min_allocation for Equity was lowered;
    # Equity max-sum from frozen assets = 0.10 + 0.05 = 0.15 (frozen max == frozen min == w0)
    new_gmin = new_constraints.group_lower_upper_constraints.group_min_allocation
    assert 0.14 < new_gmin['Equity'] < 0.16
    assert new_gmin['Alts'] == pytest.approx(0.30)


def test_no_overshoot_no_relaxation():
    """Feasible drifted weights → no warning, no group-bound changes."""
    tickers, gluc = _simple_alts_setup()
    constraints = Constraints(
        is_long_only=True,
        min_weights=pd.Series(0.0, index=tickers),
        max_weights=pd.Series(1.0, index=tickers),
        group_lower_upper_constraints=gluc,
    )
    drifted_w0 = pd.Series({
        'PE': 0.20, 'HF': 0.19, 'REIT': 0.20,    # Alts = 0.59 < 0.60 cap
        'EQ_US': 0.21, 'EQ_EU': 0.20,             # Equity = 0.41
    })
    rebal = pd.Series({'PE': 0, 'HF': 0, 'REIT': 1, 'EQ_US': 1, 'EQ_EU': 1})

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        new_constraints = constraints.update_with_valid_tickers(
            valid_tickers=tickers,
            weights_0=drifted_w0,
            rebalancing_indicators=rebal,
        )

    relax_msgs = [w for w in caught
                  if 'overshoot' in str(w.message) or 'undershoot' in str(w.message)]
    assert relax_msgs == []

    # bounds unchanged
    new_gmax = new_constraints.group_lower_upper_constraints.group_max_allocation
    assert new_gmax['Alts'] == pytest.approx(0.60)
    assert new_gmax['Equity'] == pytest.approx(0.70)


def test_no_freeze_no_relaxation():
    """No rebalancing_indicators → no freeze, no relaxation regardless of w0."""
    tickers, gluc = _simple_alts_setup()
    constraints = Constraints(
        is_long_only=True,
        min_weights=pd.Series(0.0, index=tickers),
        max_weights=pd.Series(1.0, index=tickers),
        group_lower_upper_constraints=gluc,
    )
    # Same overshooting w0 as the first test
    drifted_w0 = pd.Series({
        'PE': 0.20, 'HF': 0.21, 'REIT': 0.20,
        'EQ_US': 0.20, 'EQ_EU': 0.19,
    })

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        new_constraints = constraints.update_with_valid_tickers(
            valid_tickers=tickers,
            weights_0=drifted_w0,
            # rebalancing_indicators=None → nothing is frozen
        )

    relax_msgs = [w for w in caught
                  if 'overshoot' in str(w.message) or 'undershoot' in str(w.message)]
    assert relax_msgs == []
    # min_weights stay at zero (not pinned), so group_min_sum = 0, no overshoot
    new_gmax = new_constraints.group_lower_upper_constraints.group_max_allocation
    assert new_gmax['Alts'] == pytest.approx(0.60)


def test_pre_patch_failure_mode_now_passes():
    """Direct reproduction of the MAC error from the user report.

    Mimics: frozen Alts at 0.6048, group_max_allocation 0.6000. Before the
    fix this raised ``ValueError: Infeasible constraints detected``. After
    the fix it returns a constraints object with relaxed group_max.
    """
    tickers = ['PE', 'HF', 'OtherAlt', 'Equity']
    loadings = pd.DataFrame(
        {'Alts': [1.0, 1.0, 1.0, 0.0], 'Equity': [0.0, 0.0, 0.0, 1.0]},
        index=tickers,
    )
    gluc = GroupLowerUpperConstraints(
        group_loadings=loadings,
        group_min_allocation=pd.Series({'Alts': 0.30, 'Equity': 0.20}),
        group_max_allocation=pd.Series({'Alts': 0.6000, 'Equity': 0.80}),
    )
    constraints = Constraints(
        is_long_only=True,
        min_weights=pd.Series(0.0, index=tickers),
        max_weights=pd.Series(1.0, index=tickers),
        group_lower_upper_constraints=gluc,
    )
    drifted_w0 = pd.Series({
        'PE':       0.2016,
        'HF':       0.2016,
        'OtherAlt': 0.2016,   # sum within Alts = 0.6048
        'Equity':   0.3952,
    })
    rebal = pd.Series({'PE': 0, 'HF': 0, 'OtherAlt': 0, 'Equity': 1})

    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        # this must not raise
        new_constraints = constraints.update_with_valid_tickers(
            valid_tickers=tickers,
            weights_0=drifted_w0,
            rebalancing_indicators=rebal,
        )
    new_gmax = new_constraints.group_lower_upper_constraints.group_max_allocation
    assert new_gmax['Alts'] > 0.6048
