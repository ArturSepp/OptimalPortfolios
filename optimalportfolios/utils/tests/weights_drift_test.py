"""Tests for ``apply_drift_to_weights_0`` covering production and edge cases.

Coverage:
    - Long-only fully invested: drift matches gross/sum_gross identity
    - Long-short: NAV-growth divisor differs from sum(gross), correct formula
    - Variable exposure: drift respects implicit cash sleeve
    - Toggle off → passthrough
    - None / zero weights_0 → passthrough
    - Missing prices / prev_date → passthrough
    - NaN prices at anchor date → ffill recovers from earlier valid price
    - Asset never priced → treated as flat (drift multiplier 1.0)
    - Zero or negative price at anchor → asset treated as flat
    - NAV collapse → passthrough
    - Index misalignment (assets in weights_0 not in prices) → flat
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from optimalportfolios.utils.weights_drift import apply_drift_to_weights_0


# -----------------------------------------------------------------------------
# fixtures
# -----------------------------------------------------------------------------

def _build_prices(values, dates, columns):
    return pd.DataFrame(values, index=pd.DatetimeIndex(dates), columns=columns)


@pytest.fixture
def prices_clean():
    dates = ['2025-01-31', '2025-02-28', '2025-03-31', '2025-04-30']
    values = [
        [100.0, 100.0, 100.0],
        [105.0, 102.0, 99.0],
        [110.0, 103.0, 95.0],
        [108.0, 104.0, 97.0],
    ]
    return _build_prices(values, dates, ['A', 'B', 'C'])


# -----------------------------------------------------------------------------
# core formula
# -----------------------------------------------------------------------------

def test_long_only_fully_invested_matches_gross_renorm(prices_clean):
    """For sum(w)=1, NAV-growth norm equals gross/sum(gross)."""
    w0 = pd.Series({'A': 0.5, 'B': 0.3, 'C': 0.2})
    out = apply_drift_to_weights_0(
        weights_0=w0, prices=prices_clean,
        prev_date=pd.Timestamp('2025-01-31'),
        date=pd.Timestamp('2025-02-28'),
    )
    # gross = w * p1/p0
    gross = pd.Series({'A': 0.5 * 1.05, 'B': 0.3 * 1.02, 'C': 0.2 * 0.99})
    expected = gross / gross.sum()
    np.testing.assert_allclose(out.values, expected.values, rtol=1e-12)
    # drift preserves sum-to-1 for long-only fully-invested
    assert abs(out.sum() - 1.0) < 1e-12


def test_long_short_nav_growth_divisor():
    """For long-short with sum(w)=0, NAV-growth ≠ sum(gross); formula handles it."""
    dates = ['2025-01-31', '2025-02-28']
    prices = _build_prices(
        [[100.0, 100.0], [110.0, 95.0]],
        dates, ['A', 'B'])
    w0 = pd.Series({'A': +1.0, 'B': -1.0})  # sum = 0
    out = apply_drift_to_weights_0(
        weights_0=w0, prices=prices,
        prev_date=pd.Timestamp('2025-01-31'),
        date=pd.Timestamp('2025-02-28'),
    )
    # NAV growth = 1 + sum w*r = 1 + 1.0*0.10 + (-1.0)*(-0.05) = 1.15
    expected_A = 1.0 * 1.10 / 1.15
    expected_B = -1.0 * 0.95 / 1.15
    np.testing.assert_allclose(out['A'], expected_A, rtol=1e-12)
    np.testing.assert_allclose(out['B'], expected_B, rtol=1e-12)
    # sum no longer zero — longs gained vs shorts as fraction of new NAV
    assert out.sum() > 0


def test_variable_exposure_cash_sleeve():
    """Implicit cash modelled by sum(w) < 1; formula stays well-defined."""
    dates = ['2025-01-31', '2025-02-28']
    prices = _build_prices(
        [[100.0, 100.0], [110.0, 105.0]],
        dates, ['A', 'B'])
    w0 = pd.Series({'A': 0.6, 'B': 0.35})  # sum=0.95, 5% implicit cash
    out = apply_drift_to_weights_0(
        weights_0=w0, prices=prices,
        prev_date=pd.Timestamp('2025-01-31'),
        date=pd.Timestamp('2025-02-28'),
    )
    # NAV growth on explicit assets only: 1 + 0.6*0.10 + 0.35*0.05 = 1.0775
    nav = 1.0 + 0.6*0.10 + 0.35*0.05
    np.testing.assert_allclose(out['A'], 0.6 * 1.10 / nav, rtol=1e-12)
    np.testing.assert_allclose(out['B'], 0.35 * 1.05 / nav, rtol=1e-12)
    # explicit sum slightly higher than 0.95 since both risky assets gained
    assert 0.95 < out.sum() < 1.0


# -----------------------------------------------------------------------------
# passthrough gates
# -----------------------------------------------------------------------------

def test_toggle_off_returns_input_unchanged(prices_clean):
    w0 = pd.Series({'A': 0.5, 'B': 0.3, 'C': 0.2})
    out = apply_drift_to_weights_0(
        weights_0=w0, prices=prices_clean,
        prev_date=pd.Timestamp('2025-01-31'),
        date=pd.Timestamp('2025-02-28'),
        use_drifted_weights_0=False,
    )
    pd.testing.assert_series_equal(out, w0)


def test_none_weights_passthrough(prices_clean):
    out = apply_drift_to_weights_0(
        weights_0=None, prices=prices_clean,
        prev_date=pd.Timestamp('2025-01-31'),
        date=pd.Timestamp('2025-02-28'),
    )
    assert out is None


def test_zero_weights_passthrough(prices_clean):
    w0 = pd.Series({'A': 0.0, 'B': 0.0, 'C': 0.0})
    out = apply_drift_to_weights_0(
        weights_0=w0, prices=prices_clean,
        prev_date=pd.Timestamp('2025-01-31'),
        date=pd.Timestamp('2025-02-28'),
    )
    pd.testing.assert_series_equal(out, w0)


def test_none_prices_passthrough():
    w0 = pd.Series({'A': 0.5, 'B': 0.5})
    out = apply_drift_to_weights_0(
        weights_0=w0, prices=None,
        prev_date=pd.Timestamp('2025-01-31'),
        date=pd.Timestamp('2025-02-28'),
    )
    pd.testing.assert_series_equal(out, w0)


def test_none_prev_date_passthrough(prices_clean):
    w0 = pd.Series({'A': 0.5, 'B': 0.3, 'C': 0.2})
    out = apply_drift_to_weights_0(
        weights_0=w0, prices=prices_clean,
        prev_date=None,
        date=pd.Timestamp('2025-02-28'),
    )
    pd.testing.assert_series_equal(out, w0)


# -----------------------------------------------------------------------------
# NaN price handling
# -----------------------------------------------------------------------------

def test_nan_price_at_prev_date_ffilled_from_earlier_valid(prices_clean):
    """If asset has NaN at prev_date, ffill picks up the most recent valid."""
    prices = prices_clean.copy()
    # blank A and B at prev_date — should ffill from 2025-01-31 (100, 100)
    prices.loc['2025-02-28', ['A', 'B']] = np.nan
    w0 = pd.Series({'A': 0.5, 'B': 0.3, 'C': 0.2})
    out = apply_drift_to_weights_0(
        weights_0=w0, prices=prices,
        prev_date=pd.Timestamp('2025-02-28'),
        date=pd.Timestamp('2025-03-31'),
    )
    # A: ffilled p_prev=100, p_curr=110 → ratio 1.10
    # B: ffilled p_prev=100, p_curr=103 → ratio 1.03
    # C: p_prev=99,           p_curr=95  → ratio 0.95960...
    ratios = pd.Series({'A': 1.10, 'B': 1.03, 'C': 95.0 / 99.0})
    nav = 1.0 + (w0 * (ratios - 1.0)).sum()
    expected = w0 * ratios / nav
    np.testing.assert_allclose(out.values, expected.values, rtol=1e-12)


def test_asset_never_priced_treated_as_flat():
    """Asset with no price history → multiplier 1.0, contributes nothing to NAV move."""
    dates = ['2025-01-31', '2025-02-28']
    prices = _build_prices(
        [[100.0, np.nan], [110.0, np.nan]],
        dates, ['A', 'B'])
    w0 = pd.Series({'A': 0.5, 'B': 0.5})
    out = apply_drift_to_weights_0(
        weights_0=w0, prices=prices,
        prev_date=pd.Timestamp('2025-01-31'),
        date=pd.Timestamp('2025-02-28'),
    )
    # B treated as flat → multiplier 1.0; NAV growth = 1 + 0.5*0.10 + 0.5*0 = 1.05
    np.testing.assert_allclose(out['A'], 0.5 * 1.10 / 1.05, rtol=1e-12)
    np.testing.assert_allclose(out['B'], 0.5 * 1.00 / 1.05, rtol=1e-12)


def test_zero_price_at_prev_treated_as_flat():
    """p_prev = 0 → divide-by-zero blocked; asset treated as flat."""
    dates = ['2025-01-31', '2025-02-28']
    prices = _build_prices(
        [[0.0, 100.0], [50.0, 110.0]],
        dates, ['A', 'B'])
    w0 = pd.Series({'A': 0.5, 'B': 0.5})
    out = apply_drift_to_weights_0(
        weights_0=w0, prices=prices,
        prev_date=pd.Timestamp('2025-01-31'),
        date=pd.Timestamp('2025-02-28'),
    )
    # A flat (ratio 1.0); B normal (1.10). NAV = 1 + 0.5*0 + 0.5*0.10 = 1.05
    np.testing.assert_allclose(out['A'], 0.5 / 1.05, rtol=1e-12)
    np.testing.assert_allclose(out['B'], 0.5 * 1.10 / 1.05, rtol=1e-12)


def test_negative_price_treated_as_flat():
    """p_prev < 0 (pathological) → asset treated as flat, not sign-flipped."""
    dates = ['2025-01-31', '2025-02-28']
    prices = _build_prices(
        [[-10.0, 100.0], [50.0, 110.0]],
        dates, ['A', 'B'])
    w0 = pd.Series({'A': 0.5, 'B': 0.5})
    out = apply_drift_to_weights_0(
        weights_0=w0, prices=prices,
        prev_date=pd.Timestamp('2025-01-31'),
        date=pd.Timestamp('2025-02-28'),
    )
    np.testing.assert_allclose(out['A'], 0.5 / 1.05, rtol=1e-12)
    np.testing.assert_allclose(out['B'], 0.5 * 1.10 / 1.05, rtol=1e-12)


# -----------------------------------------------------------------------------
# universe / index alignment
# -----------------------------------------------------------------------------

def test_weights_assets_missing_from_prices_treated_as_flat():
    """Asset in weights_0 but not in prices.columns → flat (ratio = 1)."""
    dates = ['2025-01-31', '2025-02-28']
    prices = _build_prices(
        [[100.0], [110.0]], dates, ['A'])
    w0 = pd.Series({'A': 0.5, 'CASH_NOT_IN_PRICES': 0.5})
    out = apply_drift_to_weights_0(
        weights_0=w0, prices=prices,
        prev_date=pd.Timestamp('2025-01-31'),
        date=pd.Timestamp('2025-02-28'),
    )
    # NAV growth: 1 + 0.5*0.10 + 0.5*0 = 1.05
    np.testing.assert_allclose(out['A'], 0.5 * 1.10 / 1.05, rtol=1e-12)
    np.testing.assert_allclose(out['CASH_NOT_IN_PRICES'], 0.5 / 1.05, rtol=1e-12)
    # output index matches input
    assert list(out.index) == list(w0.index)


def test_prices_have_extra_assets_not_in_weights():
    """Assets in prices but not in weights_0 → silently ignored."""
    dates = ['2025-01-31', '2025-02-28']
    prices = _build_prices(
        [[100.0, 100.0, 100.0], [110.0, 90.0, 200.0]],
        dates, ['A', 'B', 'NOT_HELD'])
    w0 = pd.Series({'A': 0.6, 'B': 0.4})
    out = apply_drift_to_weights_0(
        weights_0=w0, prices=prices,
        prev_date=pd.Timestamp('2025-01-31'),
        date=pd.Timestamp('2025-02-28'),
    )
    # only A and B contribute. NAV = 1 + 0.6*0.10 + 0.4*(-0.10) = 1.02
    np.testing.assert_allclose(out['A'], 0.6 * 1.10 / 1.02, rtol=1e-12)
    np.testing.assert_allclose(out['B'], 0.4 * 0.90 / 1.02, rtol=1e-12)
    assert list(out.index) == ['A', 'B']


# -----------------------------------------------------------------------------
# date / NAV pathologies
# -----------------------------------------------------------------------------

def test_dates_before_price_index_passthrough():
    """prev_date / date before all price data → empty prefix → passthrough."""
    dates = ['2025-02-28', '2025-03-31']
    prices = _build_prices(
        [[100.0, 100.0], [110.0, 105.0]], dates, ['A', 'B'])
    w0 = pd.Series({'A': 0.5, 'B': 0.5})
    # prev_date before any price → loc[:prev_date] is empty
    out = apply_drift_to_weights_0(
        weights_0=w0, prices=prices,
        prev_date=pd.Timestamp('2024-01-01'),
        date=pd.Timestamp('2025-02-28'),
    )
    pd.testing.assert_series_equal(out, w0)


def test_dates_use_ffill_semantics_when_not_in_index():
    """prev_date and date need not be exact index keys; use ``loc[:date]``."""
    dates = ['2025-01-15', '2025-02-15', '2025-03-15']
    prices = _build_prices(
        [[100.0, 100.0], [110.0, 95.0], [115.0, 90.0]], dates, ['A', 'B'])
    w0 = pd.Series({'A': 0.6, 'B': 0.4})
    # neither '2025-01-31' nor '2025-02-28' is in the index; ffill picks
    # 2025-01-15 and 2025-02-15 respectively
    out = apply_drift_to_weights_0(
        weights_0=w0, prices=prices,
        prev_date=pd.Timestamp('2025-01-31'),
        date=pd.Timestamp('2025-02-28'),
    )
    # ratios: A 110/100=1.10, B 95/100=0.95
    nav = 1.0 + 0.6 * 0.10 + 0.4 * (-0.05)
    np.testing.assert_allclose(out['A'], 0.6 * 1.10 / nav, rtol=1e-12)
    np.testing.assert_allclose(out['B'], 0.4 * 0.95 / nav, rtol=1e-12)


def test_nav_collapse_passthrough():
    """If 1 + sum(w*r) ≤ eps, fall back to weights_0 unchanged."""
    dates = ['2025-01-31', '2025-02-28']
    # huge long-short blowup: long loses 95%, short gains 95% (negative for shorts)
    prices = _build_prices(
        [[100.0, 100.0], [5.0, 195.0]], dates, ['A', 'B'])
    # gross-short: w = [+1, -1] means long A, short B. A: -95%, B: +95%
    # NAV growth = 1 + 1.0*(-0.95) + (-1.0)*(0.95) = 1 - 0.95 - 0.95 = -0.9 (negative)
    w0 = pd.Series({'A': +1.0, 'B': -1.0})
    out = apply_drift_to_weights_0(
        weights_0=w0, prices=prices,
        prev_date=pd.Timestamp('2025-01-31'),
        date=pd.Timestamp('2025-02-28'),
    )
    pd.testing.assert_series_equal(out, w0)


# -----------------------------------------------------------------------------
# stability properties
# -----------------------------------------------------------------------------

def test_flat_returns_returns_weights_unchanged(prices_clean):
    """If all prices are flat over the period, drifted weights == original."""
    dates = ['2025-01-31', '2025-02-28']
    prices = _build_prices(
        [[100.0, 100.0, 100.0], [100.0, 100.0, 100.0]], dates, ['A', 'B', 'C'])
    w0 = pd.Series({'A': 0.5, 'B': 0.3, 'C': 0.2})
    out = apply_drift_to_weights_0(
        weights_0=w0, prices=prices,
        prev_date=pd.Timestamp('2025-01-31'),
        date=pd.Timestamp('2025-02-28'),
    )
    np.testing.assert_allclose(out.values, w0.values, rtol=1e-12)


def test_idempotent_when_drift_immediately_followed_by_identical_call():
    """Second drift call with the same period parameters but on already-drifted
    weights is a non-identity operation (drift is path-dependent on weights),
    but stays bounded and finite — sanity check for numerical stability."""
    dates = ['2025-01-31', '2025-02-28']
    prices = _build_prices(
        [[100.0, 100.0], [110.0, 95.0]], dates, ['A', 'B'])
    w0 = pd.Series({'A': 0.5, 'B': 0.5})
    out1 = apply_drift_to_weights_0(
        weights_0=w0, prices=prices,
        prev_date=pd.Timestamp('2025-01-31'),
        date=pd.Timestamp('2025-02-28'),
    )
    out2 = apply_drift_to_weights_0(
        weights_0=out1, prices=prices,
        prev_date=pd.Timestamp('2025-01-31'),
        date=pd.Timestamp('2025-02-28'),
    )
    assert np.all(np.isfinite(out2.values))
    assert (out2.abs() < 10).all()
