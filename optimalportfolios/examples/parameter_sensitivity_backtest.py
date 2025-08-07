"""
backtest parameter sensitivity of one method
"""
# imports
import pandas as pd
import matplotlib.pyplot as plt
from typing import List
from enum import Enum
import qis as qis

# package
from optimalportfolios import (PortfolioObjective, backtest_rolling_optimal_portfolio,
                               Constraints, GroupLowerUpperConstraints)
from optimalportfolios.examples.universe import fetch_benchmark_universe_data


def run_max_diversification_sensitivity_to_span(prices: pd.DataFrame,
                                                benchmark_prices: pd.DataFrame,
                                                group_data: pd.Series,
                                                time_period: qis.TimePeriod,  # weight computations
                                                perf_time_period: qis.TimePeriod,  # for reporting
                                                constraints0: Constraints
                                                ) -> List[ plt.Figure]:
    """
    test maximum diversification optimiser to span parameter
    span is number period for ewm filter
    span = 20 for daily data implies last 20 (trading) days contribute 50% of weight for covariance estimation
    we test sensitivity from fast (small span) to slow (large span)
    """
    # use daily returns
    returns_freq = 'W-WED'  # returns freq
    # span defined on number periods using returns_freq
    # for weekly returns assume 5 weeeks per month
    spans = {'1m': 5, '3m': 13, '6m': 26, '1y': 52, '2y': 104}

    # now create a list of portfolios
    portfolio_datas = []
    for ticker, span in spans.items():
        portfolio_data = backtest_rolling_optimal_portfolio(prices=prices,
                                                            constraints0=constraints0,
                                                            time_period=time_period,
                                                            portfolio_objective=PortfolioObjective.MAX_DIVERSIFICATION,
                                                            rebalancing_freq='QE',  # portfolio rebalancing
                                                            returns_freq=returns_freq,
                                                            span=span,
                                                            ticker=f"span-{ticker}",  # portfolio id
                                                            rebalancing_costs=0.0010,  # 10bp for rebalancin
                                                            weight_implementation_lag=1
                                                            )
        portfolio_data.set_group_data(group_data=group_data)
        portfolio_datas.append(portfolio_data)

    # run cross portfolio report
    multi_portfolio_data = qis.MultiPortfolioData(portfolio_datas=portfolio_datas, benchmark_prices=benchmark_prices)
    figs = qis.generate_multi_portfolio_factsheet(multi_portfolio_data=multi_portfolio_data,
                                                  time_period=perf_time_period,
                                                  add_strategy_factsheets=False,
                                                  **qis.fetch_default_report_kwargs(time_period=time_period))
    return figs


class LocalTests(Enum):
    MAX_DIVERSIFICATION_SPAN = 1


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    import optimalportfolios.local_path as local_path

    prices, benchmark_prices, ac_loadings, benchmark_weights, group_data, ac_benchmark_prices = fetch_benchmark_universe_data()

    # add costraints that each asset class is 10% <= sum ac weights <= 30% (benchamrk is 20% each)
    group_min_allocation = pd.Series(0.0, index=ac_loadings.columns)
    group_max_allocation = pd.Series(0.3, index=ac_loadings.columns)
    group_lower_upper_constraints = GroupLowerUpperConstraints(group_loadings=ac_loadings,
                                                               group_min_allocation=group_min_allocation,
                                                               group_max_allocation=group_max_allocation)
    constraints0 = Constraints(is_long_only=True,
                               min_weights=pd.Series(0.0, index=prices.columns),
                               max_weights=pd.Series(0.2, index=prices.columns),
                               group_lower_upper_constraints=group_lower_upper_constraints)

    if local_test == LocalTests.MAX_DIVERSIFICATION_SPAN:

        time_period = qis.TimePeriod(start='31Dec1998', end=prices.index[-1])  # backtest start for weights computation
        perf_time_period = qis.TimePeriod(start='31Dec2004', end=prices.index[-1])  # backtest reporting
        figs = run_max_diversification_sensitivity_to_span(prices=prices,
                                                           benchmark_prices=benchmark_prices,
                                                           constraints0=constraints0,
                                                           group_data=group_data,
                                                           time_period=time_period,
                                                           perf_time_period=perf_time_period)

        # save png and pdf
        qis.save_fig(fig=figs[0], file_name=f"max_diversification_span", local_path=f"figures/")
        qis.save_figs_to_pdf(figs=figs,
                             file_name=f"max_diversification_span",
                             orientation='landscape',
                             local_path=local_path.get_output_path())
    plt.show()


if __name__ == '__main__':

    run_local_test(local_test=LocalTests.MAX_DIVERSIFICATION_SPAN)
