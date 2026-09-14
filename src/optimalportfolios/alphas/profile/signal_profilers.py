"""Signal-specific adapters for the rank-based alpha profiler.

The five single-signal adapters call their canonical alphas.signals
constructors and pass the score panel to backtest_alpha_rank_portfolio().
The raw second constructor result is not returned by these adapters.

profile_alpha_signals() instead accepts a nonempty dictionary of precomputed
named score panels. It does not build signals from ProfileSignal members.

All adapters return QIS MultiPortfolioData with an equal-weight benchmark
last. Costs are fractional rates; time_period selects target-weight rows and
does not end the price history. The source checkout's
docs/alphas_module_readme.md covers the signal and timing limitations.
"""
# packages
import pandas as pd
import qis as qis
from enum import Enum
from typing import Dict, Optional

# optimalportfolios
from optimalportfolios.alphas.signals.momentum import compute_momentum_alpha
from optimalportfolios.alphas.signals.classic_momentum import compute_classic_momentum_alpha
from optimalportfolios.alphas.signals.low_beta import compute_low_beta_alpha
from optimalportfolios.alphas.signals.residual_momentum import compute_residual_momentum_alpha
from optimalportfolios.alphas.signals.carry import compute_ra_carry_alpha
from optimalportfolios.alphas.profile.core import backtest_alpha_rank_portfolio


class ProfileSignal(str, Enum):
    """String labels for the signal families represented by the adapters.

    These labels do not dispatch signal construction. profile_alpha_signals()
    accepts a dictionary of already computed panels with caller-chosen labels.

    Attributes:
        MOMENTUM: EWMA risk-adjusted momentum label.
        CLASSIC_MOMENTUM: Fixed-window momentum label.
        LOW_BETA: Low-beta score label.
        RESIDUAL_MOMENTUM: Benchmark-residual momentum label.
        CARRY: Risk-adjusted carry label.
    """
    MOMENTUM = 'momentum'
    CLASSIC_MOMENTUM = 'classic_momentum'
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
    """Compute momentum scores and backtest their top-quantile selection.

    Delegates to compute_momentum_alpha() with its default mean adjustment, then
    passes the cross-sectional score to the profile core. The benchmark input
    defines signal excess returns; the comparison portfolio remains equal weight.
    The short EWMA leg is a filter component, not a fixed skipped-return interval.

    Args:
        prices: Date-by-ticker prices for signal construction and QIS backtesting.
        benchmark_price: Benchmark price Series for log-return subtraction. Passing
            None uses each asset's own returns in the signal constructor.
        returns_freq: Log-return sampling cadence, default 'ME'.
        long_span: Long EWMA filter span in return observations, default 12.
        short_span: Optional short EWMA span in return observations; None omits it.
        vol_span: EWMA volatility span in return observations; None disables
            volatility normalization.
        quantile: Fraction of score-eligible assets to select, in (0, 1].
        rebalancing_freq: Calendar frequency for target weights, such as 'QE' or 'ME'.
        time_period: Optional window selecting target-weight rows before rebalance
            sampling. Signal estimation uses the supplied history; prices are not
            truncated, so NAV can continue beyond the end date.
        rebalancing_costs: Per-ticker proportional costs in fractional units, passed unchanged
            to QIS for every leg; 0.001 means 10 basis points. None means no costs.

    Returns:
        QIS MultiPortfolioData with the 'momentum' leg followed by 'Equal Weight'.
        Scores and raw momentum are available from the signal constructor separately.
    """
    scores, _ = compute_momentum_alpha(
        prices=prices, benchmark_price=benchmark_price, returns_freq=returns_freq,
        long_span=long_span, short_span=short_span, vol_span=vol_span)
    return backtest_alpha_rank_portfolio(
        prices=prices, alpha_scores=scores, quantile=quantile,
        rebalancing_freq=rebalancing_freq, time_period=time_period,
        rebalancing_costs=rebalancing_costs, strategy_ticker='momentum')


def profile_classic_momentum(prices: pd.DataFrame,
                             returns_freq: str = 'ME',
                             lookback_periods: int = 12,
                             skip_periods: int = 1,
                             group_data: Optional[pd.Series] = None,
                             quantile: float = 1.0 / 3.0,
                             rebalancing_freq: str = 'QE',
                             time_period: qis.TimePeriod = None,
                             rebalancing_costs: Optional[pd.Series] = None,
                             ) -> qis.MultiPortfolioData:
    """Compute fixed-window momentum scores and backtest their top quantile.

    Delegates to compute_classic_momentum_alpha(): sum exactly lookback_periods
    sampled log returns after excluding skip_periods recent observations, then
    score cross-sectionally. This signal uses neither benchmark subtraction nor
    volatility scaling. Incomplete windows remain missing.

    Args:
        prices: Date-by-ticker prices for signal construction and QIS backtesting.
        returns_freq: Log-return sampling cadence, default 'ME'.
        lookback_periods: Number of included return observations, default 12.
        skip_periods: Number of most recent observations excluded, default 1.
        group_data: Optional ticker-to-group labels for within-group scoring.
            Portfolio selection still ranks scores across the whole universe.
        quantile: Fraction of score-eligible assets to select, in (0, 1].
        rebalancing_freq: Calendar frequency for target weights, such as 'QE' or 'ME'.
        time_period: Optional window selecting target-weight rows before rebalance
            sampling. Signal estimation uses the supplied history; prices are not
            truncated, so NAV can continue beyond the end date.
        rebalancing_costs: Per-ticker proportional costs in fractional units, passed unchanged
            to QIS for every leg; 0.001 means 10 basis points. None means no costs.

    Returns:
        QIS MultiPortfolioData with 'classic_momentum' followed by 'Equal Weight'.
        The score and raw fixed-window sum are not returned.
    """
    scores, _ = compute_classic_momentum_alpha(
        prices=prices, returns_freq=returns_freq, group_data=group_data,
        lookback_periods=lookback_periods, skip_periods=skip_periods)
    return backtest_alpha_rank_portfolio(
        prices=prices, alpha_scores=scores, quantile=quantile,
        rebalancing_freq=rebalancing_freq, time_period=time_period,
        rebalancing_costs=rebalancing_costs, strategy_ticker='classic_momentum')


def profile_low_beta(prices: pd.DataFrame,
                    benchmark_price: pd.Series,
                    returns_freq: str = 'ME',
                    beta_span: Optional[int] = 12,
                    quantile: float = 1.0 / 3.0,
                    rebalancing_freq: str = 'QE',
                    time_period: qis.TimePeriod = None,
                    rebalancing_costs: Optional[pd.Series] = None,
                    ) -> qis.MultiPortfolioData:
    """Compute low-beta scores and backtest their top-quantile selection.

    Delegates to compute_low_beta_alpha(). Lower estimated benchmark beta gives
    a higher score; the resulting basket is long-only and equal-weighted, without
    a beta-neutrality constraint.

    This adapter retains the constructor's default EWMA mean adjustment and QIS
    mean initialization. In the verified environment, full-sample initialization
    lets later observations affect earlier scores. The default path is not a
    verified point-in-time signal. To control mean adjustment, call the signal
    constructor explicitly and pass its score panel to the profile core.

    Args:
        prices: Date-by-ticker prices for signal construction and QIS backtesting.
        benchmark_price: Benchmark price Series for beta estimation. Passing None
            uses the equal-weight mean of asset log returns in the signal constructor.
        returns_freq: Log-return sampling cadence, default 'ME'.
        beta_span: EWMA regression span and warm-up in return observations, default 12.
        quantile: Fraction of score-eligible assets to select, in (0, 1].
        rebalancing_freq: Calendar frequency for target weights, such as 'QE' or 'ME'.
        time_period: Optional window selecting target-weight rows before rebalance
            sampling. Signal estimation uses the supplied history; prices are not
            truncated, so NAV can continue beyond the end date.
        rebalancing_costs: Per-ticker proportional costs in fractional units, passed unchanged
            to QIS for every leg; 0.001 means 10 basis points. None means no costs.

    Returns:
        QIS MultiPortfolioData with 'low_beta' followed by 'Equal Weight'.
        Fitted betas and scores are not returned.
    """
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
    """Compute benchmark-residual momentum scores and backtest their top quantile.

    The signal constructor subtracts benchmark log returns multiplied by the
    preceding fitted beta, then applies the risk-adjusted EWMA momentum filter
    and cross-sectional scoring. This residual signal does not impose portfolio
    beta neutrality.

    The adapter retains the beta constructor's default EWMA mean adjustment and
    QIS mean initialization. Full-sample initialization can affect earlier scores
    when later observations change; lagging fitted betas does not remove that
    limitation. Construct scores explicitly when controlling mean adjustment.

    Args:
        prices: Date-by-ticker prices for signal construction and QIS backtesting.
        benchmark_price: Benchmark price Series for beta estimation and residual
            returns. Passing None uses the equal-weight mean of asset log returns.
        returns_freq: Log-return sampling cadence, default 'ME'.
        beta_span: EWMA regression span and warm-up in return observations, default 12.
        long_span: Long residual-momentum EWMA span in return observations, default 12.
        short_span: Optional short EWMA filter span; None omits this component.
            It does not specify a fixed skipped-return interval.
        vol_span: EWMA residual-volatility span in return observations; None disables
            volatility normalization.
        quantile: Fraction of score-eligible assets to select, in (0, 1].
        rebalancing_freq: Calendar frequency for target weights, such as 'QE' or 'ME'.
        time_period: Optional window selecting target-weight rows before rebalance
            sampling. Signal estimation uses the supplied history; prices are not
            truncated, so NAV can continue beyond the end date.
        rebalancing_costs: Per-ticker proportional costs in fractional units, passed unchanged
            to QIS for every leg; 0.001 means 10 basis points. None means no costs.

    Returns:
        QIS MultiPortfolioData with 'residual_momentum' followed by 'Equal Weight'.
        The score and raw filtered residual signal are not returned.
    """
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
    """Compute risk-adjusted carry scores and backtest their top quantile.

    The constructor divides supplied yields by annualized EWMA log-return
    volatility and scores the result cross-sectionally. Prices supply both that
    volatility estimate and the subsequent QIS backtest.

    The carry panel is used only to construct scores here. This adapter does not
    pass instruments_carry to the backtester or add separate carry cash flows.
    The simulated returns follow the supplied price series and transaction costs.

    Args:
        prices: Date-by-ticker prices for volatility estimation and backtesting.
        carry: Date-by-ticker annual fractional yields with the price tickers.
            The constructor forward-fills to volatility dates; supply values known
            on their recorded dates. For example, 0.02 represents a 2% annual yield.
        returns_freq: Log-return cadence used for volatility estimation, default 'ME'.
        vol_span: EWMA volatility span in return observations, default 13; forwarded
            unchanged to the signal constructor and QIS volatility estimator.
        group_data: Optional ticker-to-group labels for within-group scoring.
            Portfolio selection still ranks scores across the whole universe.
        quantile: Fraction of score-eligible assets to select, in (0, 1].
        rebalancing_freq: Calendar frequency for target weights, such as 'QE' or 'ME'.
        time_period: Optional window selecting target-weight rows before rebalance
            sampling. Signal estimation uses the supplied history; prices are not
            truncated, so NAV can continue beyond the end date.
        rebalancing_costs: Per-ticker proportional costs in fractional units, passed unchanged
            to QIS for every leg; 0.001 means 10 basis points. None means no costs.

    Returns:
        QIS MultiPortfolioData with 'carry' followed by 'Equal Weight'.
        The score and raw yield-to-volatility ratio are not returned.
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
    """Backtest a nonempty dictionary of named, precomputed score panels.

    Construct the panels through alphas.signals or supply externally computed
    scores, then pass them as alpha_scores. This function delegates to the core;
    it accepts no signal-enum dispatch or signal-construction parameters.

    The core ranks non-missing score/price pairs and breaks ties in price-column
    order. It does not validate score finiteness or price positivity. The
    equal-weight benchmark uses the price universe without requiring scores.

    Args:
        prices: Date-by-ticker prices for eligibility and QIS simulation.
        alpha_scores: Nonempty dictionary mapping strategy names to score DataFrames.
            Higher scores rank first; NaN excludes that asset on that date. Input
            order determines strategy order, and score columns align to price columns.
        quantile: Fraction of score-eligible assets to hold, in (0, 1].
        rebalancing_freq: Calendar frequency for target rows, default 'QE'.
        time_period: Optional target-row window before rebalance sampling. Prices
            are not truncated, so NAV can continue beyond the end date.
        rebalancing_costs: Per-ticker proportional costs in fractional units, passed unchanged
            to QIS for every leg; 0.001 means 10 basis points. None means no costs.

    Returns:
        QIS MultiPortfolioData with one leg per named panel in dictionary order,
        followed by 'Equal Weight'. Signal panels are not recomputed or returned.

    Raises:
        ValueError: If alpha_scores is empty or quantile is outside (0, 1].
    """
    if not alpha_scores:
        raise ValueError("alpha_scores is empty; pass at least one named signal panel")
    return backtest_alpha_rank_portfolio(
        prices=prices, alpha_scores=alpha_scores, quantile=quantile,
        rebalancing_freq=rebalancing_freq, time_period=time_period,
        rebalancing_costs=rebalancing_costs)
