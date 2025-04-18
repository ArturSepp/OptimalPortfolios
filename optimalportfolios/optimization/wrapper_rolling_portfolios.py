"""
linking engine to different optimisation routines
"""
# packages
import pandas as pd
import qis as qis
from typing import Optional, Dict
# optimalportfolios
import optimalportfolios as opt
from optimalportfolios.covar_estimation.covar_estimator import CovarEstimator
from optimalportfolios.optimization.constraints import Constraints
from optimalportfolios.config import PortfolioObjective


def compute_rolling_optimal_weights(prices: pd.DataFrame,
                                    constraints0: Constraints,
                                    time_period: qis.TimePeriod,
                                    portfolio_objective: PortfolioObjective = PortfolioObjective.MAX_DIVERSIFICATION,
                                    covar_dict: Dict[pd.Timestamp, pd.DataFrame] = None,  # can be precomputed
                                    risk_budget: pd.Series = None,
                                    returns_freq: Optional[str] = 'W-WED',  # returns freq
                                    rebalancing_freq: str = 'QE',  # portfolio rebalancing
                                    span: int = 52,  # ewma span for covariance matrix estimation
                                    roll_window: int = 20,  # linked to returns at rebalancing_freq
                                    carra: float = 0.5,  # carra parameters
                                    n_mixures: int = 3
                                    ) -> pd.DataFrame:
    """
    wrapper function that links implemented optimisation solvers optimisation methods
    for portfolio_objective in config.PortfolioObjective
    covar_dict: Dict[timestamp, covar matrix] can be precomputed
    portolio is rebalances at covar_dict.keys()
    """
    covar_estimator = CovarEstimator(returns_freqs=returns_freq, rebalancing_freq=rebalancing_freq, span=span)
    if portfolio_objective == PortfolioObjective.EQUAL_RISK_CONTRIBUTION:
        weights = opt.rolling_risk_budgeting(prices=prices,
                                             constraints0=constraints0,
                                             time_period=time_period,
                                             covar_dict=covar_dict,
                                             risk_budget=risk_budget,
                                             covar_estimator=covar_estimator)

    elif portfolio_objective == PortfolioObjective.MAX_DIVERSIFICATION:
        weights = opt.rolling_maximise_diversification(prices=prices,
                                                       constraints0=constraints0,
                                                       time_period=time_period,
                                                       covar_dict=covar_dict,
                                                       covar_estimator=covar_estimator)

    elif portfolio_objective in [PortfolioObjective.MIN_VARIANCE, PortfolioObjective.QUADRATIC_UTILITY]:
        weights = opt.rolling_quadratic_optimisation(prices=prices,
                                                     constraints0=constraints0,
                                                     portfolio_objective=portfolio_objective,
                                                     time_period=time_period,
                                                     covar_dict=covar_dict,
                                                     covar_estimator=covar_estimator,
                                                     carra=carra)

    elif portfolio_objective == PortfolioObjective.MAXIMUM_SHARPE_RATIO:
        weights = opt.rolling_maximize_portfolio_sharpe(prices=prices,
                                                        constraints0=constraints0,
                                                        time_period=time_period,
                                                        returns_freq=returns_freq,
                                                        rebalancing_freq=rebalancing_freq,
                                                        span=span,
                                                        roll_window=roll_window)

    elif portfolio_objective == PortfolioObjective.MAX_CARA_MIXTURE:
        weights = opt.rolling_maximize_cara_mixture(prices=prices,
                                                    constraints0=constraints0,
                                                    time_period=time_period,
                                                    returns_freq=returns_freq,
                                                    rebalancing_freq=rebalancing_freq,
                                                    carra=carra,
                                                    n_components=n_mixures,
                                                    roll_window=roll_window)

    else:
        raise NotImplementedError(f"{portfolio_objective}")

    return weights


def backtest_rolling_optimal_portfolio(prices: pd.DataFrame,
                                       constraints0: Constraints,
                                       time_period: qis.TimePeriod,  # for computing weights
                                       covar_dict: Dict[pd.Timestamp, pd.DataFrame] = None,  # can be precomputed
                                       perf_time_period: qis.TimePeriod = None,  # for computing performance
                                       portfolio_objective: PortfolioObjective = PortfolioObjective.MAX_DIVERSIFICATION,
                                       returns_freq: Optional[str] = 'W-WED',  # returns freq
                                       rebalancing_freq: str = 'QE',  # portfolio rebalancing
                                       span: int = 52,  # ewma span for covariance matrix estimation
                                       roll_window: int = 6*52,  # linked to returns at rebalancing_freq: 6y of weekly returns
                                       carra: float = 0.5,  # carra parameter
                                       n_mixures: int = 3,  # for mixture carra utility
                                       ticker: str = None,
                                       rebalancing_costs: float = 0.0010,  # 10 bp
                                       weight_implementation_lag: Optional[int] = None  # = 1 for daily data
                                       ) -> qis.PortfolioData:
    """
    compute solvers portfolio weights and return portfolio data
    weight_implementation_lag: Optional[int] = None  # = 1 for daily data otherwise skip
    covar_dict: Dict[timestamp, covar matrix] can be precomputed
    portolio is rebalances at covar_dict.keys()
    """
    weights = compute_rolling_optimal_weights(prices=prices,
                                              time_period=time_period,
                                              constraints0=constraints0,
                                              covar_dict=covar_dict,
                                              portfolio_objective=portfolio_objective,
                                              returns_freq=returns_freq,
                                              rebalancing_freq=rebalancing_freq,
                                              span=span,
                                              carra=carra,
                                              roll_window=roll_window,
                                              n_mixures=n_mixures)

    # make sure price exists for the first weight date: can happen when the first weight date falls on weekend
    if perf_time_period is not None:
        weights = perf_time_period.locate(weights)
    prices_ = qis.truncate_prior_to_start(df=prices, start=weights.index[0])
    portfolio_out = qis.backtest_model_portfolio(prices=prices_,
                                                 weights=weights,
                                                 rebalancing_costs=rebalancing_costs,
                                                 weight_implementation_lag=weight_implementation_lag,
                                                 ticker=ticker)
    return portfolio_out
