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

import logging
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

def test_frozen_overshoot_raises_group_max(caplog):
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

    with caplog.at_level(logging.INFO,
                         logger="optimalportfolios.optimization.constraints"):
        new_constraints = constraints.update_with_valid_tickers(
            valid_tickers=tickers,
            weights_0=drifted_w0,
            rebalancing_indicators=rebal,
        )

    # the relaxation message must surface (now on the logger, not warnings)
    assert "frozen-min overshoot" in caplog.text
    assert "group 'Alts'" in caplog.text

    # group_max_allocation for Alts was raised; Equity untouched.
    # Frozen-min-sum within Alts = 0.20 + 0.21 + 0.20 = 0.61, so new cap ≈ 0.61 + tol.
    new_gmax = new_constraints.group_lower_upper_constraints.group_max_allocation
    assert 0.60 < new_gmax['Alts'] < 0.62
    assert new_gmax['Equity'] == pytest.approx(0.70)


def test_frozen_undershoot_lowers_group_min(caplog):
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

    with caplog.at_level(logging.INFO,
                         logger="optimalportfolios.optimization.constraints"):
        new_constraints = constraints.update_with_valid_tickers(
            valid_tickers=tickers,
            weights_0=drifted_w0,
            rebalancing_indicators=rebal,
        )

    assert "frozen-max undershoot" in caplog.text
    assert "group 'Equity'" in caplog.text

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


def test_relaxation_message_includes_context_date(caplog):
    """When the caller passes context (the rebalance date), the relaxation
    message is prefixed with [date] — mirroring the solver-rejection messages."""
    tickers, gluc = _simple_alts_setup()
    constraints = Constraints(
        is_long_only=True,
        min_weights=pd.Series(0.0, index=tickers),
        max_weights=pd.Series(1.0, index=tickers),
        group_lower_upper_constraints=gluc,
    )
    drifted_w0 = pd.Series({'PE': 0.20, 'HF': 0.21, 'REIT': 0.20,
                            'EQ_US': 0.20, 'EQ_EU': 0.19})
    rebal = pd.Series({'PE': 0, 'HF': 0, 'REIT': 0, 'EQ_US': 1, 'EQ_EU': 1})

    def _relax_msgs():
        return [r.getMessage() for r in caplog.records
                if 'overshoot' in r.getMessage() or 'undershoot' in r.getMessage()]

    # with context -> [date] prefix
    with caplog.at_level(logging.INFO,
                         logger="optimalportfolios.optimization.constraints"):
        constraints.update_with_valid_tickers(
            valid_tickers=tickers, weights_0=drifted_w0,
            rebalancing_indicators=rebal, context="2024-01-31")
    relax = _relax_msgs()
    assert relax, "expected a relaxation log record"
    assert relax[0].startswith("[2024-01-31] ")

    caplog.clear()
    # without context -> unchanged (no prefix); backward compatible
    with caplog.at_level(logging.INFO,
                         logger="optimalportfolios.optimization.constraints"):
        constraints.update_with_valid_tickers(
            valid_tickers=tickers, weights_0=drifted_w0,
            rebalancing_indicators=rebal)
    relax = _relax_msgs()
    assert relax and relax[0].startswith("Constraints.")


def test_relaxation_magnitude_bound_escalates(caplog):
    """A relaxation larger than max_relaxation_tol escalates the log to ERROR and
    the RelaxationRecord marks breached_tol; a generous tolerance does not."""
    from optimalportfolios.optimization.constraints import RelaxationRecord
    tickers, gluc = _simple_alts_setup()
    constraints = Constraints(
        is_long_only=True,
        min_weights=pd.Series(0.0, index=tickers),
        max_weights=pd.Series(1.0, index=tickers),
        group_lower_upper_constraints=gluc,
    )
    # Alts frozen sum 0.61 vs cap 0.60 -> relaxation magnitude ~0.0101.
    drifted_w0 = pd.Series({'PE': 0.20, 'HF': 0.21, 'REIT': 0.20,
                            'EQ_US': 0.20, 'EQ_EU': 0.19})
    rebal = pd.Series({'PE': 0, 'HF': 0, 'REIT': 0, 'EQ_US': 1, 'EQ_EU': 1})

    def _records():
        return [r for r in (getattr(rec, "relaxation", None) for rec in caplog.records)
                if isinstance(r, RelaxationRecord)]

    with caplog.at_level(logging.INFO,
                         logger="optimalportfolios.optimization.constraints"):
        constraints.update_with_valid_tickers(
            valid_tickers=tickers, weights_0=drifted_w0,
            rebalancing_indicators=rebal, max_relaxation_tol=0.005)   # 0.0101 > 0.005
    recs = _records()
    assert recs and recs[0].breached_tol is True
    assert any(r.levelno == logging.ERROR for r in caplog.records)

    caplog.clear()
    with caplog.at_level(logging.INFO,
                         logger="optimalportfolios.optimization.constraints"):
        constraints.update_with_valid_tickers(
            valid_tickers=tickers, weights_0=drifted_w0,
            rebalancing_indicators=rebal, max_relaxation_tol=0.05)     # 0.0101 < 0.05
    recs = _records()
    assert recs and recs[0].breached_tol is False
    assert not any(r.levelno == logging.ERROR for r in caplog.records)
