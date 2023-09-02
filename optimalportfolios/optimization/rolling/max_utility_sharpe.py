"""
    implementation of [PortfolioObjective.QUADRATIC_UTILITY,
                        PortfolioObjective.MAXIMUM_SHARPE_RATIO]

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import qis
from qis import PortfolioData, TimePeriod
from typing import Tuple, List, Dict, Optional
from enum import Enum

from optimalportfolios.optimization.config import PortfolioObjective, set_min_max_weights, set_to_zero_not_investable_weights
from optimalportfolios.optimization.solvers.quadratic import max_portfolio_sharpe_qp, maximize_portfolio_objective_qp


def compute_rolling_max_utility_sharpe_weights(prices: pd.DataFrame,
                                               portfolio_objective: PortfolioObjective = PortfolioObjective.MAXIMUM_SHARPE_RATIO,
                                               min_weights: Dict[str, float] = None,
                                               max_weights: Dict[str, float] = None,
                                               fixed_weights: Dict[str, float] = None,
                                               is_long_only: bool = True,
                                               returns_freq: Optional[str] = 'W-WED',
                                               rebalancing_freq: str = 'Q',
                                               roll_window: int = 20,  # defined on number of periods in rebalancing_freq
                                               span: int = 52,
                                               carra: float = 0.5,
                                               is_log_returns: bool = True,
                                               is_print_log: bool = False
                                               ) -> pd.DataFrame:
    """
    implementation of [PortfolioObjective.QUADRATIC_UTILITY,
                        PortfolioObjective.MAXIMUM_SHARPE_RATIO]
    """
    means, covars = estimate_rolling_means_covar(prices=prices,
                                                 returns_freq=returns_freq,
                                                 rebalancing_freq=rebalancing_freq,
                                                 roll_window=roll_window,
                                                 annualize=True,
                                                 span=span,
                                                 is_log_returns=is_log_returns)

    # set weights
    min_weights0, max_weights0 = set_min_max_weights(assets=list(prices.columns),
                                                     min_weights=min_weights,
                                                     max_weights=max_weights,
                                                     fixed_weights=fixed_weights,
                                                     is_long_only=is_long_only)

    weights = {}
    for index, covar in zip(means.index, covars):
        min_weights1, max_weights1 = set_to_zero_not_investable_weights(min_weights=min_weights0,
                                                                        max_weights=max_weights0,
                                                                        covar=covar)

        if portfolio_objective == PortfolioObjective.MAXIMUM_SHARPE_RATIO:
            weights[index] = max_portfolio_sharpe_qp(means=means.loc[index, :].to_numpy(),
                                                     covar=covar,
                                                     min_weights=min_weights1.to_numpy(),
                                                     max_weights=max_weights1.to_numpy(),
                                                     is_print_log=is_print_log)

        elif portfolio_objective == PortfolioObjective.QUADRATIC_UTILITY:
            weights[index] = maximize_portfolio_objective_qp(portfolio_objective=PortfolioObjective.QUADRATIC_UTILITY,
                                                             means=means.loc[index, :].to_numpy(),
                                                             covar=covar,
                                                             min_weights=min_weights1.to_numpy(),
                                                             max_weights=max_weights1.to_numpy(),
                                                             carra=carra)
        else:
            raise NotImplementedError(f"{portfolio_objective}")

    weights = pd.DataFrame.from_dict(weights, orient='index', columns=prices.columns)

    return weights


def backtest_rolling_max_utility_sharpe_portfolios(prices: pd.DataFrame,
                                                   portfolio_objective: PortfolioObjective = PortfolioObjective.MAXIMUM_SHARPE_RATIO,
                                                   min_weights: Dict[str, float] = None,
                                                   max_weights: Dict[str, float] = None,
                                                   fixed_weights: Dict[str, float] = None,
                                                   returns_freq: Optional[str] = 'W-WED',
                                                   rebalancing_freq: str = 'Q',
                                                   roll_window: int = 20,
                                                   span: int = 52,
                                                   carra: float = 0.5,
                                                   time_period: TimePeriod = None,
                                                   is_log_returns: bool = True,
                                                   ticker: str = None,
                                                   rebalancing_costs: float = 0.0010  # 10 bp
                                                   ) -> PortfolioData:
    """
    portfolio for compute_rolling_max_utility_sharpe_weights
    """
    weights = compute_rolling_max_utility_sharpe_weights(prices=prices,
                                                         portfolio_objective=portfolio_objective,
                                                         returns_freq=returns_freq,
                                                         rebalancing_freq=rebalancing_freq,
                                                         roll_window=roll_window,
                                                         span=span,
                                                         carra=carra,
                                                         is_log_returns=is_log_returns,
                                                         min_weights=min_weights,
                                                         max_weights=max_weights,
                                                         fixed_weights=fixed_weights)
    if time_period is not None:
        weights = time_period.locate(weights)

    # make sure price exists for the first weight date: can happen when the first weight date falls on weekend
    prices_ = qis.truncate_prior_to_start(df=prices, start=weights.index[0])
    portfolio_out = qis.backtest_model_portfolio(prices=prices_,
                                                 weights=weights,
                                                 is_rebalanced_at_first_date=True,
                                                 ticker=ticker,
                                                 is_output_portfolio_data=True,
                                                 rebalancing_costs=rebalancing_costs)
    return portfolio_out


def estimate_rolling_means_covar(prices: pd.DataFrame,
                                 returns_freq: str = 'W-WED',
                                 rebalancing_freq: str = 'Q',
                                 roll_window: int = 20,  # defined on number of periods in rebalancing_freq
                                 span: int = 52,
                                 is_log_returns: bool = True,
                                 annualize: bool = True,
                                 is_regularize: bool = True,
                                 is_ewm_covar: bool = True
                                 ) -> Tuple[pd.DataFrame, List[np.ndarray]]:

    """
    inputs for rolling portfolios
    """
    rets = qis.to_returns(prices=prices, is_log_returns=is_log_returns, drop_first=True, freq=returns_freq)

    dates_schedule = qis.generate_dates_schedule(time_period=qis.get_time_period(df=rets),
                                                 freq=rebalancing_freq,
                                                 include_start_date=True,
                                                 include_end_date=False)

    if annualize:
        _, scaler = qis.get_period_days(freq=returns_freq)
    else:
        scaler = 1.0
    means = {}
    covars = []
    covar0 = np.zeros((len(prices.columns), len(prices.columns)))
    for idx, end in enumerate(dates_schedule[1:]):
        if idx >= roll_window-1:
            period = TimePeriod(dates_schedule[idx - roll_window + 1], end)
            # period.print()
            rets_ = period.locate(rets).to_numpy()
            means[end] = scaler*np.nanmean(rets_, axis=0)
            if is_ewm_covar:
                covar = qis.compute_ewm_covar(a=rets_, span=span, covar0=covar0)
                covar0 = covar
            else:
                covar = qis.compute_masked_covar_corr(returns=rets_, bias=True)

            if is_regularize:
                covar = qis.matrix_regularization(covar=covar, cut=1e-5)

            covars.append(scaler * covar)
    means = pd.DataFrame.from_dict(means, orient="index")

    return means, covars


class UnitTests(Enum):
    ROLLING_MEANS_COVAR = 1
    ROLLING_PORTFOLIOS = 2


def run_unit_test(unit_test: UnitTests):

    from optimalportfolios.test_data import load_test_data
    prices = load_test_data()
    prices = prices.loc['2000':, :]  # need 5 years for max sharpe and max carra methods

    if unit_test == UnitTests.ROLLING_MEANS_COVAR:
        # prices = prices[['SPY', 'TLT']].dropna()

        means, covars = estimate_rolling_means_covar(prices=prices, rebalancing_freq='Q', roll_window=20)
        #  = estimate_rolling_data(prices=prices, rebalancing_freq='M', roll_window=60)

        vols = {}
        covs = {}
        for index, covar in zip(means.index, covars):
            vols[index] = pd.Series(np.sqrt(np.diag(covar)))
            covs[index] = pd.Series(np.extract(1 - np.eye(2), covar))
        vols = pd.DataFrame.from_dict(vols, orient='index')
        covs = pd.DataFrame.from_dict(covs, orient='index')
        print(vols)
        print(covs)

        with sns.axes_style("darkgrid"):
            fig, axs = plt.subplots(3, 1, figsize=(7, 12))
            qis.plot_time_series(df=means,
                                 var_format='{:.0%}',
                                 trend_line=qis.TrendLine.AVERAGE,
                                 legend_stats=qis.LegendStats.FIRST_AVG_LAST,
                                 ax=axs[0])
            qis.plot_time_series(df=vols,
                                 var_format='{:.0%}',
                                 trend_line=qis.TrendLine.AVERAGE,
                                 legend_stats=qis.LegendStats.FIRST_AVG_LAST,
                                 ax=axs[1])
            qis.plot_time_series(df=covs,
                                 var_format='{:.0%}',
                                 trend_line=qis.TrendLine.AVERAGE,
                                 legend_stats=qis.LegendStats.FIRST_AVG_LAST,
                                 ax=axs[2])

    elif unit_test == UnitTests.ROLLING_PORTFOLIOS:

        kwargs = dict(add_mean_levels=True,
                      is_yaxis_limit_01=True,
                      bbox_to_anchor=(0.4, 1.1),
                      legend_stats=qis.LegendStats.AVG_STD_LAST,
                      ncol=len(prices.columns) // 3,
                      var_format='{:.0%}')

        port_data = backtest_rolling_max_utility_sharpe_portfolios(prices=prices,
                                                                   portfolio_objective=PortfolioObjective.MAXIMUM_SHARPE_RATIO,
                                                                   carra=0.0,
                                                                   rebalancing_freq='Q',
                                                                   roll_window=20,
                                                                   returns_freq='W-WED')
        with sns.axes_style("darkgrid"):
            fig, ax = plt.subplots(1, 1, figsize=(7, 12))
            port_data.plot_weights(ax=ax,
                                   **kwargs)
    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.ROLLING_PORTFOLIOS

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
