"""
the per-signal profiler wrappers.

These are deliberately thin -- each computes one signal panel and hands it to the core rank
backtester -- so the only thing that can go wrong in them is the wiring, and the wiring is
exactly what the tests below check: that each wrapper reaches its own signal function, labels
its leg with its own name, and passes the caller's quantile and schedule through rather than
silently using the default. A wrapper that called the wrong signal function would still return
a perfectly well-formed MultiPortfolioData, which is why the leg name is asserted explicitly.
"""
# packages
import pandas as pd
import pytest
import qis
# optimalportfolios
from optimalportfolios.alphas.profile.signal_profilers import (
    ProfileSignal,
    profile_alpha_signals,
    profile_carry,
    profile_classic_momentum,
    profile_low_beta,
    profile_momentum,
    profile_residual_momentum,
)
from optimalportfolios.tests.data.multiasset import load_multiasset_data


@pytest.fixture(scope='module')
def universe() -> tuple:
    """A decade of the committed monthly universe, with the first column as benchmark."""
    data = load_multiasset_data()
    prices = data.prices.loc['2010':'2019', :]
    benchmark_price = prices.iloc[:, 0]
    return prices, benchmark_price, data.group_data


def leg_tickers(multi: qis.MultiPortfolioData) -> list:
    """Leg names of a profiled MultiPortfolioData, benchmark included."""
    return [portfolio.ticker for portfolio in multi.portfolio_datas]


def test_profile_momentum_labels_its_leg_and_appends_the_benchmark(universe: tuple) -> None:
    """The momentum wrapper names its leg 'momentum' and leaves the benchmark last."""
    prices, benchmark_price, _ = universe
    multi = profile_momentum(prices=prices, benchmark_price=benchmark_price)
    assert leg_tickers(multi) == ['momentum', 'Equal Weight']


def test_profile_classic_momentum_labels_its_leg(universe: tuple) -> None:
    """The classic wrapper reaches the fixed-window signal and labels its leg."""
    prices, _, _ = universe
    multi = profile_classic_momentum(prices=prices)
    assert leg_tickers(multi) == ['classic_momentum', 'Equal Weight']


def test_profile_low_beta_labels_its_leg(universe: tuple) -> None:
    """The low-beta wrapper reaches the low-beta signal and names the leg for it."""
    prices, benchmark_price, _ = universe
    multi = profile_low_beta(prices=prices, benchmark_price=benchmark_price)
    assert leg_tickers(multi) == ['low_beta', 'Equal Weight']


def test_profile_residual_momentum_labels_its_leg(universe: tuple) -> None:
    """The residual-momentum wrapper is wired to its own signal, not to plain momentum."""
    prices, benchmark_price, _ = universe
    multi = profile_residual_momentum(prices=prices, benchmark_price=benchmark_price)
    assert leg_tickers(multi) == ['residual_momentum', 'Equal Weight']


def test_profile_carry_takes_a_yield_panel_rather_than_deriving_one(universe: tuple) -> None:
    """Carry is the one profiler needing an exogenous panel; prices only normalise its vol."""
    prices, _, group_data = universe
    carry = pd.DataFrame(0.02, index=prices.index, columns=prices.columns)
    carry.iloc[:, :5] = 0.06                       # a stable cross-sectional carry spread
    multi = profile_carry(prices=prices, carry=carry, group_data=group_data)
    assert leg_tickers(multi) == ['carry', 'Equal Weight']


def test_the_quantile_reaches_the_selection_rule(universe: tuple) -> None:
    """A wider quantile must hold more names, so the wrapper cannot be dropping the argument."""
    prices, benchmark_price, _ = universe
    narrow = profile_momentum(prices=prices, benchmark_price=benchmark_price, quantile=1.0 / 6.0)
    wide = profile_momentum(prices=prices, benchmark_price=benchmark_price, quantile=2.0 / 3.0)
    n_narrow = narrow.portfolio_datas[0].get_weights().gt(0.0).sum(axis=1).max()
    n_wide = wide.portfolio_datas[0].get_weights().gt(0.0).sum(axis=1).max()
    assert n_wide > n_narrow


def test_the_rebalancing_frequency_reaches_the_backtester(universe: tuple) -> None:
    """Monthly rebalancing must trade more than quarterly.

    Not a row count: ``get_weights`` reports realised, drifting weights on the price index, so
    both schedules return the same number of rows and the rarer schedule actually shows *more*
    distinct ones. Turnover is what the schedule genuinely moves.
    """
    prices, benchmark_price, _ = universe
    monthly = profile_momentum(prices=prices, benchmark_price=benchmark_price,
                               rebalancing_freq='ME')
    quarterly = profile_momentum(prices=prices, benchmark_price=benchmark_price,
                                 rebalancing_freq='QE')
    assert (monthly.portfolio_datas[0].get_turnover(is_agg=True, roll_period=None).sum().sum()
            > quarterly.portfolio_datas[0].get_turnover(is_agg=True, roll_period=None).sum().sum())


def test_profile_alpha_signals_makes_one_leg_per_named_panel(universe: tuple) -> None:
    """The signal-agnostic entry point keys its legs off the dict it is given."""
    prices, _, _ = universe
    multi = profile_alpha_signals(prices=prices,
                                  alpha_scores={'fast': prices.pct_change(3).shift(1),
                                                'slow': prices.pct_change(12).shift(1)})
    assert leg_tickers(multi) == ['fast', 'slow', 'Equal Weight']


def test_profile_alpha_signals_rejects_an_empty_dict(universe: tuple) -> None:
    """Zero signals would silently return a benchmark-only run, so it raises instead."""
    prices, _, _ = universe
    with pytest.raises(ValueError, match='at least one named signal panel'):
        profile_alpha_signals(prices=prices, alpha_scores={})


def test_profile_signal_values_are_stable_strings() -> None:
    """The enum is a str enum used as a label; its values are part of the interface."""
    assert [signal.value for signal in ProfileSignal] == [
        'momentum', 'classic_momentum', 'low_beta', 'residual_momentum', 'carry']
