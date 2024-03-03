"""
linking engine to different optimisation routines
"""
# packages
import pandas as pd
from typing import Dict, Optional

# qis portfolio
import qis
from qis.portfolio.portfolio_data import PortfolioData

from optimalportfolios.optimization.config import PortfolioObjective
from optimalportfolios.optimization.rolling.max_mixture_carra import (compute_rolling_weights_mixture_carra)
from optimalportfolios.optimization.rolling.max_utility_sharpe import compute_rolling_max_utility_sharpe_weights
from optimalportfolios.optimization.rolling.risk_based import compute_rolling_ewma_risk_based_weights


def compute_rolling_optimal_weights(prices: pd.DataFrame,
                                    portfolio_objective: PortfolioObjective = PortfolioObjective.MAX_DIVERSIFICATION,
                                    min_weights: Dict[str, float] = None,  # asset with min weights
                                    max_weights: Dict[str, float] = None,
                                    fixed_weights: Dict[str, float] = None,
                                    is_long_only: bool = True,
                                    returns_freq: Optional[str] = 'W-WED',  # returns freq
                                    rebalancing_freq: str = 'QE',  # portfolio rebalancing
                                    span: int = 52,  # ewma span for covariance matrix estimation
                                    roll_window: int = 20,  # linked to returns at rebalancing_freq
                                    carra: float = 0.5,  # carra parameters
                                    **kwargs
                                    ) -> pd.DataFrame:
    """
    wrapper function that links implemented optimisation rolling optimisation methods
    prices contains price data with portfolio universe
    """

    if portfolio_objective in [PortfolioObjective.EQUAL_RISK_CONTRIBUTION,
                               PortfolioObjective.MAX_DIVERSIFICATION,
                               PortfolioObjective.RISK_PARITY_ALT,
                               PortfolioObjective.MIN_VARIANCE]:
        # these portfolios are using rolling covariance matrix
        weights = compute_rolling_ewma_risk_based_weights(prices=prices,
                                                          portfolio_objective=portfolio_objective,
                                                          min_weights=min_weights,
                                                          max_weights=max_weights,
                                                          fixed_weights=fixed_weights,
                                                          is_long_only=is_long_only,
                                                          rebalancing_freq=rebalancing_freq,
                                                          returns_freq=returns_freq,
                                                          roll_window=roll_window,
                                                          span=span,
                                                          **kwargs)

    elif portfolio_objective in [PortfolioObjective.MAXIMUM_SHARPE_RATIO,
                                 PortfolioObjective.QUADRATIC_UTILITY]:
        weights = compute_rolling_max_utility_sharpe_weights(prices=prices,
                                                             portfolio_objective=portfolio_objective,
                                                             min_weights=min_weights,
                                                             max_weights=max_weights,
                                                             fixed_weights=fixed_weights,
                                                             rebalancing_freq=rebalancing_freq,
                                                             returns_freq=returns_freq,
                                                             span=span,
                                                             roll_window=roll_window,
                                                             carra=carra,
                                                             **kwargs)

    elif portfolio_objective in [PortfolioObjective.MAX_MIXTURE_CARA]:
        weights = compute_rolling_weights_mixture_carra(prices=prices,
                                                        carra=carra,  # carra parameters
                                                        min_weights=min_weights,
                                                        max_weights=max_weights,
                                                        fixed_weights=fixed_weights,
                                                        rebalancing_freq=rebalancing_freq,
                                                        returns_freq=returns_freq,
                                                        roll_window=roll_window,
                                                        **kwargs)

    else:
        raise NotImplementedError(f"{portfolio_objective}")

    return weights


def backtest_rolling_optimal_portfolio(prices: pd.DataFrame,
                                       portfolio_objective: PortfolioObjective = PortfolioObjective.MAX_DIVERSIFICATION,
                                       min_weights: Dict[str, float] = None,  # asset with min weights
                                       max_weights: Dict[str, float] = None,
                                       fixed_weights: Dict[str, float] = None,
                                       is_long_only: bool = True,
                                       returns_freq: Optional[str] = 'W-WED',  # returns freq
                                       rebalancing_freq: str = 'QE',  # portfolio rebalancing
                                       span: int = 52,  # ewma span for covariance matrix estimation
                                       carra: float = 0.5,  # carra parameters
                                       time_period: qis.TimePeriod = None,  # portfolio
                                       ticker: str = None,
                                       rebalancing_costs: float = 0.0010,  # 10 bp
                                       **kwargs
                                       ) -> PortfolioData:
    """
    compute rolling portfolio weights and return portfolio data
    """
    weights = compute_rolling_optimal_weights(prices=prices,
                                              portfolio_objective=portfolio_objective,
                                              min_weights=min_weights,
                                              max_weights=max_weights,
                                              fixed_weights=fixed_weights,
                                              is_long_only=is_long_only,
                                              returns_freq=returns_freq,
                                              rebalancing_freq=rebalancing_freq,
                                              span=span,
                                              carra=carra,
                                              **kwargs)

    if time_period is not None:
        weights = time_period.locate(weights)

    # make sure price exists for the first weight date: can happen when the first weight date falls on weekend
    prices_ = qis.truncate_prior_to_start(df=prices, start=weights.index[0])
    portfolio_out = qis.backtest_model_portfolio(prices=prices_,
                                                 weights=weights,
                                                 rebalancing_costs=rebalancing_costs,
                                                 is_rebalanced_at_first_date=True,
                                                 ticker=ticker,
                                                 is_output_portfolio_data=True)
    return portfolio_out
