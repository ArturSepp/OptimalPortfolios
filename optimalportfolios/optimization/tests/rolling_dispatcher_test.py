"""
the contract every rolling dispatcher keeps, whatever it optimises.

The nine ``rolling_*`` entry points are how a backtest reaches this package. They differ in
objective and in what they need — means, a risk budget, a benchmark — but they agree on a shape:
prices and a dict of covariances in, a weight panel out, indexed by the rebalancing dates the
covariance dict carries. This module asserts that shape and, more importantly, two properties that
are invisible when they break.

**No look-ahead.** A weight at *t* must not depend on anything after *t*. Asserted as a
differential: run the dispatcher on the full panel, run it again on the panel truncated at *t*
with the covariance dict restricted the same way, and require the weights up to *t* to be
identical. Nothing about a look-ahead is visible in a single run — the numbers are plausible, the
backtest is optimistic, and the error only appears out of sample. This is the test the roadmap
calls the most valuable one in the suite, and it is the reason the estimation frequency and the
rebalancing frequency are separate arguments rather than one.

**The investable universe comes from ``inclusion_indicators``, not from NaN prices.** With a
delisted instrument in the panel and no inclusion indicators, the covariance estimator zero-fills
the missing returns, the instrument looks riskless, and a minimum-variance optimiser allocates to
it — measured here at 47.5% each for two dead instruments, 95% of the portfolio. Passing
inclusion indicators removes them cleanly: zero weight on an instrument with no price, at every
one of the 77 rebalancing dates. Both behaviours are pinned, because the first is a hazard a
caller must know about and the second is the contract that avoids it.

The panel is the committed `multiasset` fixture, masked by ``optimalportfolios.tests.data_masks``
where a defect is needed. No network, no vendor data.
"""
# packages
from typing import Callable, Dict

import numpy as np
import pandas as pd
import pytest
import qis

# optimalportfolios
import optimalportfolios as op
from optimalportfolios import Constraints
from optimalportfolios.tests.data.multiasset import load_multiasset_data
from optimalportfolios.tests.data_masks import (mask_delistings, mask_late_listings,
                                                instruments_alive_at)

RETURNS_FREQ = 'ME'
REBALANCING_FREQ = 'QE'
SPAN = 24
WARMUP = 60          # observations before the first covariance is taken


def _panel() -> pd.DataFrame:
    """the committed offline fixture: 292 monthly observations, 19 instruments, no NaN."""
    return load_multiasset_data().prices


def _covars(prices: pd.DataFrame, end: pd.Timestamp = None) -> Dict[pd.Timestamp, pd.DataFrame]:
    """
    rolling covariances over ``prices``, optionally stopping at ``end``.

    Args:
        prices: the price panel
        end: last date to keep. None uses the whole panel

    Returns:
        rebalancing date to covariance matrix
    """
    full = _panel()
    time_period = qis.TimePeriod(full.index[WARMUP], end if end is not None else full.index[-1])
    return op.estimate_rolling_ewma_covar(prices=prices, time_period=time_period,
                                          returns_freq=RETURNS_FREQ,
                                          rebalancing_freq=REBALANCING_FREQ, span=SPAN)


def _means(prices: pd.DataFrame) -> pd.DataFrame:
    """a constant expected-return panel, one row per price date; value is not the point here."""
    values = np.linspace(0.02, 0.08, len(prices.columns))
    return pd.DataFrame(np.tile(values, (len(prices), 1)),
                        index=prices.index, columns=prices.columns)


# Each entry: a name, and a callable taking (prices, constraints, covar_dict, **extra) so the
# contract can be asserted once per dispatcher rather than once per objective.
def _dispatchers() -> Dict[str, Callable]:
    """the rolling entry points reachable with the fixture alone."""
    return {
        'rolling_quadratic_optimisation':
            lambda prices, constraints, covar_dict, **kw: op.rolling_quadratic_optimisation(
                prices=prices, constraints=constraints, covar_dict=covar_dict, **kw),
        'rolling_maximise_diversification':
            lambda prices, constraints, covar_dict, **kw: op.rolling_maximise_diversification(
                prices=prices, constraints=constraints, covar_dict=covar_dict, **kw),
        'rolling_risk_budgeting':
            lambda prices, constraints, covar_dict, **kw: op.rolling_risk_budgeting(
                prices=prices, constraints=constraints, covar_dict=covar_dict,
                risk_budget=pd.Series(1.0 / len(prices.columns), index=prices.columns), **kw),
        'rolling_maximize_portfolio_sharpe':
            lambda prices, constraints, covar_dict, **kw: op.rolling_maximize_portfolio_sharpe(
                prices=prices, constraints=constraints, covar_dict=covar_dict,
                expected_returns=_means(prices), **kw),
        'rolling_minimise_tracking_error':
            lambda prices, constraints, covar_dict, **kw: op.rolling_minimise_tracking_error(
                prices=prices, constraints=constraints, covar_dict=covar_dict,
                benchmark_weights=pd.Series(
                    [1.0] + [0.0] * (len(prices.columns) - 1), index=prices.columns
                ), **kw),
    }


DISPATCHER_NAMES = sorted(_dispatchers())


def _run(name: str, prices: pd.DataFrame, covar_dict: Dict[pd.Timestamp, pd.DataFrame],
         constraints: Constraints = None, **kwargs) -> pd.DataFrame:
    """call one dispatcher by name."""
    constraints = constraints if constraints is not None else Constraints(is_long_only=True)
    return _dispatchers()[name](prices, constraints, covar_dict, **kwargs)


# ───────────────────────────────────────────────────────────────────────────────
# Shape
# ───────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize('name', DISPATCHER_NAMES)
def test_output_is_indexed_by_the_rebalancing_schedule(name: str) -> None:
    """one row per covariance date, one column per instrument, in panel order."""
    prices = _panel()
    covar_dict = _covars(prices)
    weights = _run(name, prices, covar_dict)
    assert list(weights.index) == sorted(covar_dict), (
        'the weight panel is not indexed by the covariance dates it was given')
    assert list(weights.columns) == list(prices.columns)


@pytest.mark.parametrize('name', DISPATCHER_NAMES)
def test_weights_are_finite_and_fully_invested(name: str) -> None:
    """no NaN anywhere, every row sums to one, and long-only means non-negative."""
    prices = _panel()
    weights = _run(name, prices, _covars(prices))
    assert not weights.isna().any().any(), 'the weight panel contains NaN'
    np.testing.assert_allclose(weights.sum(axis=1).to_numpy(), 1.0, atol=1e-6)
    assert weights.to_numpy().min() >= -1e-6, 'long-only constraint produced a negative weight'


@pytest.mark.parametrize('name', DISPATCHER_NAMES)
def test_constraints_reach_the_solver(name: str) -> None:
    """
    a weight cap holds at every rebalancing date, not just the first.

    A constraint built once and passed into a loop can be consumed, mutated or dropped after the
    first iteration. The result still looks like a weight panel.

    The cap is also checked to bind. A cap the optimiser would have respected anyway makes the
    assertion vacuous: it passes whether or not the constraint reached the solver, which is
    exactly the thing being tested. This guard was added after raising the cap to 0.999 left the
    test green.
    """
    prices = _panel()
    covar_dict = _covars(prices)
    cap = 0.20

    unconstrained = _run(name, prices, covar_dict)
    assert unconstrained.to_numpy().max() > cap, (
        f'{name} already respects a {cap} cap without being asked, so this test would pass '
        f'whether or not the constraint reached the solver. Lower the cap')

    constraints = Constraints(is_long_only=True,
                              max_weights=pd.Series(cap, index=prices.columns))
    weights = _run(name, prices, covar_dict, constraints=constraints)
    worst = weights.to_numpy().max()
    assert worst <= cap + 1e-6, f'weight cap {cap} breached, worst weight {worst:.6f}'


# ───────────────────────────────────────────────────────────────────────────────
# No look-ahead
# ───────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize('name', DISPATCHER_NAMES)
def test_weights_do_not_depend_on_the_future(name: str) -> None:
    """
    the differential check: truncating the panel must not move any earlier weight.

    Run on the full sample and on the sample cut at *t*; every weight dated at or before *t* must
    be identical. A full-sample statistic anywhere in the path — an in-sample mean, a covariance
    estimated over everything, a normalisation by a total — moves them.
    """
    prices = _panel()
    covar_full = _covars(prices)
    cut = sorted(covar_full)[40]

    weights_full = _run(name, prices, covar_full)
    weights_cut = _run(name, prices.loc[:cut], {k: v for k, v in covar_full.items() if k <= cut})

    common = weights_full.index.intersection(weights_cut.index)
    assert len(common) > 20, (
        f'only {len(common)} dates in common; the truncation is not testing much')
    difference = (weights_full.loc[common] - weights_cut.loc[common]).abs().to_numpy().max()
    assert difference < 1e-10, (
        f'weights up to {cut.date()} moved by {difference:.3e} when data after {cut.date()} was '
        f'removed, so the dispatcher is using information it would not have had')


def test_rolling_covariance_does_not_depend_on_the_future() -> None:
    """
    the same differential one layer down, where a look-ahead is likelier to hide.

    The dispatcher receives covariances rather than computing them, so a leak in the estimator
    would not be caught by the test above.
    """
    prices = _panel()
    covar_full = _covars(prices)
    cut = sorted(covar_full)[40]
    covar_cut = _covars(prices.loc[:cut], end=cut)

    common = sorted(set(covar_full) & set(covar_cut))
    assert len(common) > 20
    difference = max(np.abs(covar_full[k].to_numpy() - covar_cut[k].to_numpy()).max()
                     for k in common)
    assert difference < 1e-12, (
        f'covariance matrices dated at or before {cut.date()} moved by {difference:.3e} when '
        f'later data was removed')


# ───────────────────────────────────────────────────────────────────────────────
# The investable universe
# ───────────────────────────────────────────────────────────────────────────────


def _masked_panel() -> pd.DataFrame:
    """the fixture with three late listings and two delistings."""
    return mask_delistings(mask_late_listings(_panel()))


def test_inclusion_indicators_keep_weight_off_dead_instruments() -> None:
    """
    the contract: an instrument with no price at a rebalancing date receives no weight.

    Checked at every date rather than the last, because a delisting part-way through is the case
    a single end-of-sample check misses.
    """
    prices = _masked_panel()
    assert prices.isna().sum().sum() > 0, 'the mask produced no NaN; the test would prove nothing'
    inclusion = prices.notna().astype(float)
    weights = _run('rolling_quadratic_optimisation', prices, _covars(prices),
                   inclusion_indicators=inclusion)

    violations = []
    for date in weights.index:
        alive = set(instruments_alive_at(prices, date))
        for ticker in prices.columns:
            if ticker not in alive and abs(float(weights.loc[date, ticker])) > 1e-8:
                violations.append((date.date(), ticker, float(weights.loc[date, ticker])))
    assert not violations, f'weight given to instruments with no price: {violations[:5]}'
    np.testing.assert_allclose(weights.sum(axis=1).to_numpy(), 1.0, atol=1e-6)


def test_without_inclusion_indicators_the_universe_is_not_inferred_from_prices() -> None:
    """
    pins today's contract, and the hazard that comes with it.

    Without inclusion indicators the covariance estimator zero-fills a delisted instrument's
    returns. Zero returns look like zero variance, so a minimum-variance optimiser buys it: two
    dead instruments took 47.5% each when this was written.

    This asserts the package does *not* infer the universe from NaN prices. If that ever changes
    it is an improvement, but a deliberate one — it moves every backtest that relied on the old
    behaviour, so it needs a CHANGELOG entry and this test updated in the same commit, not a
    silent pass.
    """
    prices = _masked_panel()
    weights = _run('rolling_quadratic_optimisation', prices, _covars(prices))
    last = weights.index[-1]
    alive = set(instruments_alive_at(prices, last))
    dead = [t for t in prices.columns if t not in alive]
    assert dead, 'the delisting mask left nothing dead at the last rebalancing date'
    dead_weight = float(weights.loc[last, dead].sum())
    assert dead_weight > 0.01, (
        f'instruments with no price took only {dead_weight:.4f} of the portfolio. If the universe '
        f'is now inferred from prices this is better behaviour - update this test and the '
        f'CHANGELOG together')


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
