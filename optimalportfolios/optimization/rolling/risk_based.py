"""
    risk-based methods using EWMA covariance matrix
    implementation of [PortfolioObjective.EQUAL_RISK_CONTRIBUTION,
                       PortfolioObjective.MAX_DIVERSIFICATION,
                       PortfolioObjective.RISK_PARITY_ALT,
                       PortfolioObjective.MIN_VAR]
"""

# packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Optional
from enum import Enum

# qis
import qis
from qis.portfolio import backtester as bp
from qis import TimePeriod, PortfolioData

# optimisers
import optimalportfolios.optimization.solvers.nonlinear as ops
import optimalportfolios.optimization.solvers.quadratic as qup
from optimalportfolios.optimization.config import PortfolioObjective, set_min_max_weights, set_to_zero_not_investable_weights


def compute_rolling_ewma_risk_based_weights(prices: pd.DataFrame,
                                            portfolio_objective: PortfolioObjective = PortfolioObjective.MIN_VARIANCE,
                                            min_weights: Dict[str, float] = None,
                                            max_weights: Dict[str, float] = None,
                                            fixed_weights: Dict[str, float] = None,
                                            is_long_only: bool = True,
                                            target_vol: float = None,
                                            returns_freq: Optional[str] = 'W-WED',
                                            rebalancing_freq: str = 'Q',
                                            span: int = 52,  # ewma span in periods of returns_freq
                                            is_regularize: bool = False,
                                            is_log_returns: bool = True,
                                            budget: np.ndarray = None,
                                            **kwargs
                                            ) -> pd.DataFrame:
    """
    compute time series of ewma matrix and solve for optimal weights at rebalancing_freq
    fixed_weights are fixed weights for principal portfolios
    implementation of [PortfolioObjective.EQUAL_RISK_CONTRIBUTION,
                       PortfolioObjective.MAX_DIVERSIFICATION,
                       PortfolioObjective.RISK_PARITY_ALT,
                       PortfolioObjective.MIN_VAR,
                       PortfolioObjective.QUADRATIC_UTILITY]
    asset becomes investable when its price time series is not zero for covariance estimation
    """
    returns = qis.to_returns(prices=prices,
                             is_log_returns=is_log_returns,
                             freq=returns_freq,
                             ffill_nans=True,
                             include_end_date=False)

    # drift adjusted returns
    returns_np = returns.to_numpy()
    x = returns_np - qis.compute_ewm(returns_np, span=span)

    # fill nans using zeros
    covar_tensor_txy = qis.compute_ewm_covar_tensor(a=x, span=span, nan_backfill=qis.NanBackfill.ZERO_FILL)

    rebalancing_schedule = qis.generate_rebalancing_indicators(df=returns, freq=rebalancing_freq)

    # set weights
    min_weights0, max_weights0 = set_min_max_weights(assets=list(prices.columns),
                                                     min_weights=min_weights,
                                                     max_weights=max_weights,
                                                     fixed_weights=fixed_weights,
                                                     is_long_only=is_long_only)

    an_factor = qis.infer_an_from_data(data=returns)
    weights = {}
    for idx, (date, value) in enumerate(rebalancing_schedule.items()):
        if value:
            covar = an_factor*covar_tensor_txy[idx]
            if is_regularize:
                covar = qis.matrix_regularization(covar=covar)

            # if diag(covar) == 0 , the asset has missing returns: set its weights to zero
            min_weights1, max_weights1 = set_to_zero_not_investable_weights(min_weights=min_weights0,
                                                                            max_weights=max_weights0,
                                                                            covar=covar)

            if portfolio_objective == PortfolioObjective.EQUAL_RISK_CONTRIBUTION:
                if target_vol is None:
                    weights[date] = ops.solve_equal_risk_contribution(covar=covar,
                                                                      budget=budget,
                                                                      min_weights=min_weights1.to_numpy(),
                                                                      max_weights=max_weights1.to_numpy())
                else:
                    weights[date] = ops.solve_risk_parity_constr_vol(covar=covar,
                                                                     target_vol=target_vol)

            elif portfolio_objective == PortfolioObjective.MAX_DIVERSIFICATION:
                weights[date] = ops.solve_max_diversification(covar=covar,
                                                              min_weights=min_weights1.to_numpy(),
                                                              max_weights=max_weights1.to_numpy(),
                                                              is_long_only=is_long_only)

            elif portfolio_objective == PortfolioObjective.RISK_PARITY_ALT:
                weights[date] = ops.solve_risk_parity_alt(covar=covar)

            elif portfolio_objective == PortfolioObjective.MIN_VARIANCE:
                weights[date] = qup.maximize_portfolio_objective_qp(portfolio_objective=portfolio_objective,
                                                                    covar=covar,
                                                                    is_long_only=is_long_only,
                                                                    min_weights=min_weights1.to_numpy(),
                                                                    max_weights=max_weights1.to_numpy())
            else:
                raise NotImplementedError(f"{portfolio_objective}")

    weights = pd.DataFrame.from_dict(weights, orient='index', columns=returns.columns)
    return weights


def backtest_rolling_ewma_risk_based_portfolio(prices: pd.DataFrame,
                                               min_weights: np.ndarray = None,
                                               max_weights: np.ndarray = None,
                                               time_period: TimePeriod = None,
                                               portfolio_objective: PortfolioObjective = PortfolioObjective.EQUAL_RISK_CONTRIBUTION,
                                               span: int = 52,
                                               returns_freq: str = 'W-WED',
                                               rebalancing_freq: str = 'Q',
                                               is_log_returns: bool = True,
                                               budget: np.ndarray = None,
                                               ticker: str = None,
                                               rebalancing_costs: float = 0.0010,  # 10 bp
                                               ) -> PortfolioData:
    """
    wrapper to get portfolio data using compute_rolling_optimal_weights_ewm_covar
    covariance estimation starts from the period of price data
    time_period limits the weights of the portfolio
    """
    weights = compute_rolling_ewma_risk_based_weights(prices=prices,
                                                      portfolio_objective=portfolio_objective,
                                                      min_weights=min_weights,
                                                      max_weights=max_weights,
                                                      rebalancing_freq=rebalancing_freq,
                                                      returns_freq=returns_freq,
                                                      span=span,
                                                      is_log_returns=is_log_returns,
                                                      budget=budget)
    if time_period is not None:
        weights = time_period.locate(weights)

    # make sure price exists for the first weight date: can happen when the first weight date falls on weekend
    prices_ = qis.truncate_prior_to_start(df=prices, start=weights.index[0])
    portfolio_out = bp.backtest_model_portfolio(prices=prices_,
                                                weights=weights,
                                                rebalancing_costs=rebalancing_costs,
                                                is_rebalanced_at_first_date=True,
                                                ticker=ticker,
                                                is_output_portfolio_data=True)
    return portfolio_out


class UnitTests(Enum):
    MIN_VAR = 1
    MIN_VAR_OVERLAY = 2


def run_unit_test(unit_test: UnitTests):

    from optimalportfolios.test_data import load_test_data
    prices = load_test_data()
    prices = prices.loc['2000':, :]  # need 5 years for max sharpe and max carra methods

    kwargs = dict(add_mean_levels=True,
                  is_yaxis_limit_01=True,
                  baseline='zero',
                  bbox_to_anchor=(0.4, 1.1),
                  legend_stats=qis.LegendStats.AVG_STD_LAST,
                  ncol=len(prices.columns)//3,
                  var_format='{:.0%}')

    if unit_test == UnitTests.MIN_VAR:
        weights = compute_rolling_ewma_risk_based_weights(prices=prices,
                                                          portfolio_objective=PortfolioObjective.MIN_VARIANCE)
        qis.plot_stack(df=weights, **kwargs)

    elif unit_test == UnitTests.MIN_VAR_OVERLAY:
        fixed_weights = {'SPY': 0.6}
        weights = compute_rolling_ewma_risk_based_weights(prices=prices, fixed_weights=fixed_weights)
        qis.plot_stack(df=weights, **kwargs)

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.MIN_VAR

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
