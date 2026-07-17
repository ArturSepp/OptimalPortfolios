"""
per-signal alpha profilers.

Each function here takes the parameters for one signal, computes its alpha score panel via the
canonical signal function, and delegates to the core ``backtest_alpha_rank_portfolio``. They are thin
by design: the value is in ``profile.core``; these just wire a named signal into it. To profile a
signal the core doesn't have a wrapper for, compute the panel yourself and call the core directly.

    profile_momentum          -- risk-adjusted momentum
    profile_low_beta          -- low-beta (beta-to-benchmark)
    profile_residual_momentum -- residual (benchmark-neutral) momentum
    profile_carry             -- risk-adjusted carry (needs a yield panel)
    profile_alpha_signals     -- profile several named signals jointly in one MultiPortfolioData
"""
# packages
import pandas as pd
import qis as qis
from enum import Enum
from typing import Dict, List, Optional, Union

# optimalportfolios
from optimalportfolios.alphas.signals.momentum import compute_momentum_alpha
from optimalportfolios.alphas.signals.low_beta import compute_low_beta_alpha
from optimalportfolios.alphas.signals.residual_momentum import compute_residual_momentum_alpha
from optimalportfolios.alphas.signals.carry import compute_ra_carry_alpha
from optimalportfolios.alphas.profile.core import backtest_alpha_rank_portfolio


class ProfileSignal(str, Enum):
    """named signals the joint profiler can build from prices (+ optional carry)."""
    MOMENTUM = 'momentum'
    LOW_BETA = 'low_beta'
    RESIDUAL_MOMENTUM = 'residual_momentum'
    CARRY = 'carry'


def profile_momentum(prices: pd.DataFrame,
                     benchmark_price: pd.Series,
                     returns_freq: str = 'ME',
                     long_span: Optional[int] = 12,
                     short_span: Optional[int] = None,
                     vol_span: Optional[int] = 13,
                     quantile: float = 1.0 / 3.0,
                     rebalancing_freq: str = 'QE',
                     time_period: qis.TimePeriod = None,
                     rebalancing_costs: Optional[pd.Series] = None,
                     ) -> qis.MultiPortfolioData:
    """profile risk-adjusted momentum: compute the score, then rank-backtest the top quantile."""
    scores, _ = compute_momentum_alpha(
        prices=prices, benchmark_price=benchmark_price, returns_freq=returns_freq,
        long_span=long_span, short_span=short_span, vol_span=vol_span)
    return backtest_alpha_rank_portfolio(
        prices=prices, alpha_scores=scores, quantile=quantile,
        rebalancing_freq=rebalancing_freq, time_period=time_period,
        rebalancing_costs=rebalancing_costs, strategy_ticker='momentum')


def profile_low_beta(prices: pd.DataFrame,
                    benchmark_price: pd.Series,
                    returns_freq: str = 'ME',
                    beta_span: Optional[int] = 12,
                    quantile: float = 1.0 / 3.0,
                    rebalancing_freq: str = 'QE',
                    time_period: qis.TimePeriod = None,
                    rebalancing_costs: Optional[pd.Series] = None,
                    ) -> qis.MultiPortfolioData:
    """profile low-beta: compute the score, then rank-backtest the top quantile."""
    scores, _ = compute_low_beta_alpha(
        prices=prices, benchmark_price=benchmark_price, returns_freq=returns_freq,
        beta_span=beta_span)
    return backtest_alpha_rank_portfolio(
        prices=prices, alpha_scores=scores, quantile=quantile,
        rebalancing_freq=rebalancing_freq, time_period=time_period,
        rebalancing_costs=rebalancing_costs, strategy_ticker='low_beta')


def profile_residual_momentum(prices: pd.DataFrame,
                              benchmark_price: pd.Series,
                              returns_freq: str = 'ME',
                              beta_span: Optional[int] = 12,
                              long_span: Optional[int] = 12,
                              short_span: Optional[int] = None,
                              vol_span: Optional[int] = 13,
                              quantile: float = 1.0 / 3.0,
                              rebalancing_freq: str = 'QE',
                              time_period: qis.TimePeriod = None,
                              rebalancing_costs: Optional[pd.Series] = None,
                              ) -> qis.MultiPortfolioData:
    """profile residual (benchmark-neutral) momentum: compute the score, then rank-backtest."""
    scores, _ = compute_residual_momentum_alpha(
        prices=prices, benchmark_price=benchmark_price, returns_freq=returns_freq,
        beta_span=beta_span, long_span=long_span, short_span=short_span, vol_span=vol_span)
    return backtest_alpha_rank_portfolio(
        prices=prices, alpha_scores=scores, quantile=quantile,
        rebalancing_freq=rebalancing_freq, time_period=time_period,
        rebalancing_costs=rebalancing_costs, strategy_ticker='residual_momentum')


def profile_carry(prices: pd.DataFrame,
                 carry: pd.DataFrame,
                 returns_freq: str = 'ME',
                 vol_span: Optional[int] = 13,
                 group_data: Optional[pd.Series] = None,
                 quantile: float = 1.0 / 3.0,
                 rebalancing_freq: str = 'QE',
                 time_period: qis.TimePeriod = None,
                 rebalancing_costs: Optional[pd.Series] = None,
                 ) -> qis.MultiPortfolioData:
    """profile risk-adjusted carry: compute the score from the yield panel, then rank-backtest.

    Unlike the other profilers, carry needs the ``carry`` (yield) panel as input; prices are used only
    for the volatility normalisation.
    """
    scores, _ = compute_ra_carry_alpha(
        prices=prices, carry=carry, returns_freq=returns_freq,
        vol_span=vol_span, group_data=group_data)
    return backtest_alpha_rank_portfolio(
        prices=prices, alpha_scores=scores, quantile=quantile,
        rebalancing_freq=rebalancing_freq, time_period=time_period,
        rebalancing_costs=rebalancing_costs, strategy_ticker='carry')


def profile_alpha_signals(prices: pd.DataFrame,
                         alpha_scores: Dict[str, pd.DataFrame],
                         quantile: float = 1.0 / 3.0,
                         rebalancing_freq: str = 'QE',
                         time_period: qis.TimePeriod = None,
                         rebalancing_costs: Optional[pd.Series] = None,
                         ) -> qis.MultiPortfolioData:
    """profile a dict of externally-computed alpha signals against one equal-weight benchmark.

    Signal-agnostic: it does not compute any signal. It takes a dict of named score panels -- built
    anywhere (the per-signal profilers, the rosaa covariance/managers/cluster pipeline, or by hand) --
    and hands them to the core rank profiler as legs of one ``MultiPortfolioData``. For each signal, at
    each date the instruments with a finite score (and a valid price) are ranked and the top quantile
    is held equal-weighted; instruments whose score is NaN that date are excluded from the ranking. The
    equal-weight benchmark is all instruments, equally invested.

    Args:
        prices: T x N price panel.
        alpha_scores: {signal name: T x N score panel}. Higher score = better. NaN excludes that
            instrument from the ranking on that date. Not computed here.
        quantile: top fraction held, in (0, 1]. Default 1/3.
        rebalancing_freq: rebalance schedule. Default 'QE'.
        time_period: optional reporting window.
        rebalancing_costs: optional per-asset cost in bp.

    Returns:
        qis.MultiPortfolioData: one leg per signal in the dict, plus the equal-weight benchmark last.

    Raises:
        ValueError: if alpha_scores is empty.
    """
    if not alpha_scores:
        raise ValueError("alpha_scores is empty; pass at least one named signal panel")
    return backtest_alpha_rank_portfolio(
        prices=prices, alpha_scores=alpha_scores, quantile=quantile,
        rebalancing_freq=rebalancing_freq, time_period=time_period,
        rebalancing_costs=rebalancing_costs)