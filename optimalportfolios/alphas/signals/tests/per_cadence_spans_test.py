"""Per-cadence EWMA spans in the mixed-frequency signal paths.

Every signal estimates one reporting-frequency bucket at a time and, before
6.8.0, handed the same scalar span to each bucket -- so ``long_span=12`` was a
year of monthly returns and three years of quarterly ones, from one number,
with nothing in the signature saying the unit changed between buckets.

The two properties that matter are asserted here rather than described:

1. a scalar span is bit-identical to the pre-6.8.0 behaviour, so no existing
   caller moves;
2. a mapping that keeps the ``'ME'`` entry at its scalar value leaves every
   monthly column bit-identical and moves only the quarterly ones.

Property 2 is what makes a production delta attributable: the change is
confined to quarterly-reporting instruments by construction.

Data is generated, never downloaded; the seed is fixed at 20260802.
"""
# packages
import numpy as np
import pandas as pd
import pytest
import qis as qis
from typing import Tuple
# project
from optimalportfolios.alphas.signals.carry import (compute_ra_carry_alpha,
                                                   compute_ra_carry_cluster_alpha)
from optimalportfolios.alphas.signals.low_beta import (compute_low_beta_alpha,
                                                      compute_low_beta_cluster_alpha)
from optimalportfolios.alphas.signals.momentum import (compute_momentum_alpha,
                                                      compute_momentum_cluster_alpha)
from optimalportfolios.alphas.signals.residual_momentum import (
    compute_residual_momentum_alpha, compute_residual_momentum_cluster_alpha)
from optimalportfolios.alphas.signals.residual_reversal import (
    compute_residual_reversal_alpha, compute_residual_reversal_cluster_alpha)
from optimalportfolios.alphas.signals.utils import resolve_span

SEED: int = 20260802
MONTHLY = ['m_fund_1', 'm_fund_2', 'm_fund_3']
QUARTERLY = ['q_fund_1', 'q_fund_2']


def _panel() -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """a mixed-cadence price panel, its benchmark, its freq map and a carry panel."""
    rng = np.random.default_rng(SEED)
    dates = pd.date_range('2008-01-31', '2026-06-30', freq='ME')
    n = len(dates)

    def nav(vol: float) -> np.ndarray:
        return 100.0 * np.exp(np.cumsum(rng.normal(0.005, vol, size=n)))

    prices = pd.DataFrame({t: nav(0.03 + 0.005 * i)
                           for i, t in enumerate(MONTHLY + QUARTERLY)}, index=dates)
    benchmark = pd.Series(nav(0.04), index=dates, name='benchmark')
    returns_freq = pd.Series({**{t: 'ME' for t in MONTHLY},
                              **{t: 'QE' for t in QUARTERLY}})
    carry = pd.DataFrame(rng.uniform(0.01, 0.06, size=(n, len(prices.columns))),
                         index=dates, columns=prices.columns)
    return prices, benchmark, returns_freq, carry


# each entry: label, callable(**span_kwargs) -> (score, raw), the span under test
def _signal_cases():
    prices, benchmark, returns_freq, carry = _panel()
    common = dict(prices=prices, returns_freq=returns_freq)
    return [
        ('momentum', lambda **k: compute_momentum_alpha(
            benchmark_price=benchmark, short_span=None, vol_span=13, **common, **k), 'long_span', 12),
        ('momentum_vol', lambda **k: compute_momentum_alpha(
            benchmark_price=benchmark, short_span=None, long_span=12, **common, **k), 'vol_span', 13),
        ('low_beta', lambda **k: compute_low_beta_alpha(
            benchmark_price=benchmark, **common, **k), 'beta_span', 12),
        ('residual_momentum', lambda **k: compute_residual_momentum_alpha(
            benchmark_price=benchmark, long_span=12, short_span=None, vol_span=13,
            **common, **k), 'beta_span', 12),
        ('residual_reversal', lambda **k: compute_residual_reversal_alpha(
            benchmark_price=benchmark, long_span=1, short_span=None, vol_span=13,
            **common, **k), 'beta_span', 12),
        ('carry', lambda **k: compute_ra_carry_alpha(
            carry=carry, **common, **k), 'vol_span', 13),
    ]


CASES = _signal_cases()
IDS = [c[0] for c in CASES]


# -- the resolver -------------------------------------------------------------

def test_scalar_resolves_to_itself_at_every_cadence() -> None:
    for freq in ('ME', 'QE', 'W-WED'):
        assert resolve_span(12, freq=freq, name='long_span') == 12


def test_mapping_resolves_per_cadence() -> None:
    table = {'ME': 12, 'QE': 4}
    assert resolve_span(table, freq='ME', name='long_span') == 12
    assert resolve_span(table, freq='QE', name='long_span') == 4


def test_none_stays_none_so_an_optional_span_can_still_be_disabled() -> None:
    assert resolve_span(None, freq='QE', name='vol_span') is None


def test_an_uncovered_cadence_raises_rather_than_inheriting() -> None:
    with pytest.raises(ValueError, match='but an asset reports at'):
        resolve_span({'ME': 12}, freq='QE', name='long_span')


@pytest.mark.parametrize('bad', [0, -3, True, 2.5])
def test_a_span_that_is_not_a_positive_int_raises(bad) -> None:
    with pytest.raises(ValueError):
        resolve_span(bad, freq='ME', name='long_span')


# -- the two properties that matter -------------------------------------------

@pytest.mark.parametrize('label,run,span_name,scalar', CASES, ids=IDS)
def test_a_scalar_span_is_bit_identical_to_a_flat_mapping(label, run, span_name, scalar) -> None:
    """no existing caller moves: {'ME': s, 'QE': s} must equal the scalar s."""
    scalar_score, scalar_raw = run(**{span_name: scalar})
    flat_score, flat_raw = run(**{span_name: {'ME': scalar, 'QE': scalar}})
    pd.testing.assert_frame_equal(scalar_score, flat_score)
    pd.testing.assert_frame_equal(scalar_raw, flat_raw)


@pytest.mark.parametrize('label,run,span_name,scalar', CASES, ids=IDS)
def test_a_per_cadence_span_moves_only_the_quarterly_columns(label, run, span_name, scalar) -> None:
    """anchoring the ME entry on the scalar confines the change to quarterly lines.

    The RAW signal is computed per bucket, so a monthly column cannot see the
    quarterly entry. (The SCORE is cross-sectional and therefore moves for
    everyone -- that is the point of a cross-sectional score, and it is checked
    separately below.)
    """
    _, base_raw = run(**{span_name: scalar})
    _, moved_raw = run(**{span_name: {'ME': scalar, 'QE': 4}})
    for ticker in MONTHLY:
        pd.testing.assert_series_equal(base_raw[ticker], moved_raw[ticker],
                                       obj=f"{label}: monthly column {ticker}")
    moved = [t for t in QUARTERLY if not base_raw[t].equals(moved_raw[t])]
    assert moved, f"{label}: the quarterly columns did not move at all"


@pytest.mark.parametrize('label,run,span_name,scalar', CASES, ids=IDS)
def test_the_cross_sectional_score_does_move(label, run, span_name, scalar) -> None:
    """the score ranks the whole cross-section, so a quarterly change reaches it."""
    base_score, _ = run(**{span_name: scalar})
    moved_score, _ = run(**{span_name: {'ME': scalar, 'QE': 4}})
    assert not base_score.equals(moved_score)


@pytest.mark.parametrize('label,run,span_name,scalar', CASES, ids=IDS)
def test_an_uncovered_cadence_raises_through_the_public_entry_point(
        label, run, span_name, scalar) -> None:
    """the guard is reachable from the caller, not only from the resolver."""
    with pytest.raises(ValueError, match='but an asset reports at'):
        run(**{span_name: {'ME': scalar}})


def test_the_warmup_follows_the_span_without_being_passed() -> None:
    """momentum masks ``warmup_period=long_span``; a per-cadence span moves it too.

    Shortening the quarterly span must let the quarterly column start earlier.
    This is the reason no separate per-cadence warmup table is needed on the
    signal path: warmup is derived from the span it warms up.
    """
    prices, benchmark, returns_freq, _ = _panel()
    kwargs = dict(prices=prices, benchmark_price=benchmark, returns_freq=returns_freq,
                  short_span=None, vol_span=13)
    _, long_warmup = compute_momentum_alpha(long_span={'ME': 12, 'QE': 12}, **kwargs)
    _, short_warmup = compute_momentum_alpha(long_span={'ME': 12, 'QE': 4}, **kwargs)
    for ticker in QUARTERLY:
        assert short_warmup[ticker].first_valid_index() <= long_warmup[ticker].first_valid_index()
    assert any(short_warmup[t].first_valid_index() < long_warmup[t].first_valid_index()
               for t in QUARTERLY)


# -- the other two call sites: single-frequency and cluster entry points -------
#
# resolve_span is called wherever a scalar cadence is in scope and a
# _compute_raw_*_single_freq is about to run: inside the mixed-frequency bucket
# loop, and on the single-frequency branch of each entry point. The tests above
# only reach the first of those, because they all pass a Series.

def test_the_single_frequency_entry_point_resolves_a_mapping() -> None:
    """returns_freq as a STRING is a different branch and must resolve too."""
    prices, benchmark, _, _ = _panel()
    kwargs = dict(prices=prices, benchmark_price=benchmark, returns_freq='QE',
                  short_span=None, vol_span=13)
    mapped_score, mapped_raw = compute_momentum_alpha(long_span={'ME': 12, 'QE': 4}, **kwargs)
    scalar_score, scalar_raw = compute_momentum_alpha(long_span=4, **kwargs)
    pd.testing.assert_frame_equal(mapped_raw, scalar_raw)
    pd.testing.assert_frame_equal(mapped_score, scalar_score)


def test_the_single_frequency_entry_point_raises_on_an_uncovered_cadence() -> None:
    prices, benchmark, _, _ = _panel()
    with pytest.raises(ValueError, match='but an asset reports at'):
        compute_momentum_alpha(prices=prices, benchmark_price=benchmark, returns_freq='QE',
                               long_span={'ME': 12}, short_span=None, vol_span=13)


# each cluster entry point, with the cadence passed both ways. These reach
# ``_compute_raw_*_mixed_freq`` -- the ONLY caller of it -- which the tests above
# never touch, because compute_*_alpha dispatches to _compute_*_alpha_mixed_freq
# instead. Without them, a bucket loop can stop resolving and nothing fails.
def _cluster_cases():
    prices, benchmark, returns_freq, carry = _panel()
    return [
        ('momentum', lambda freq, **k: compute_momentum_cluster_alpha(
            prices=prices, benchmark_price=benchmark, returns_freq=freq,
            short_span=None, vol_span=13, **k), 'long_span', 12),
        ('low_beta', lambda freq, **k: compute_low_beta_cluster_alpha(
            prices=prices, benchmark_price=benchmark, returns_freq=freq, **k), 'beta_span', 12),
        ('residual_momentum', lambda freq, **k: compute_residual_momentum_cluster_alpha(
            prices=prices, benchmark_price=benchmark, returns_freq=freq,
            long_span=12, short_span=None, vol_span=13, **k), 'beta_span', 12),
        ('residual_reversal', lambda freq, **k: compute_residual_reversal_cluster_alpha(
            prices=prices, benchmark_price=benchmark, returns_freq=freq,
            long_span=1, short_span=None, vol_span=13, **k), 'beta_span', 12),
        ('carry', lambda freq, **k: compute_ra_carry_cluster_alpha(
            prices=prices, carry=carry, returns_freq=freq, **k), 'vol_span', 13),
    ]


CLUSTER_CASES = _cluster_cases()
CLUSTER_IDS = [c[0] for c in CLUSTER_CASES]
MIXED_FREQ = _panel()[2]


@pytest.mark.parametrize('label,run,span_name,scalar', CLUSTER_CASES, ids=CLUSTER_IDS)
def test_the_cluster_mixed_branch_resolves_per_bucket(label, run, span_name, scalar) -> None:
    """the raw bucket loop must not carry one bucket's horizon into the next."""
    _, base = run(MIXED_FREQ, **{span_name: scalar})
    _, moved = run(MIXED_FREQ, **{span_name: {'ME': scalar, 'QE': 4}})
    for ticker in MONTHLY:
        pd.testing.assert_series_equal(base[ticker], moved[ticker],
                                       obj=f"{label}: monthly column {ticker}")
    assert any(not base[t].equals(moved[t]) for t in QUARTERLY), (
        f"{label}: the quarterly columns did not move -- the bucket loop is not "
        f"resolving per cadence")


@pytest.mark.parametrize('label,run,span_name,scalar', CLUSTER_CASES, ids=CLUSTER_IDS)
def test_the_cluster_mixed_branch_raises_on_an_uncovered_cadence(
        label, run, span_name, scalar) -> None:
    with pytest.raises(ValueError, match='but an asset reports at'):
        run(MIXED_FREQ, **{span_name: {'ME': scalar}})


@pytest.mark.parametrize('label,run,span_name,scalar', CLUSTER_CASES, ids=CLUSTER_IDS)
def test_the_cluster_single_branch_resolves_a_mapping(label, run, span_name, scalar) -> None:
    """returns_freq as a string takes the other branch of the same dispatch."""
    _, mapped = run('QE', **{span_name: {'ME': scalar, 'QE': 4}})
    _, direct = run('QE', **{span_name: 4})
    pd.testing.assert_frame_equal(mapped, direct)
