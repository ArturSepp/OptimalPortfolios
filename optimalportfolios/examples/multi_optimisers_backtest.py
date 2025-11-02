"""
backtest several optimisers
"""
# imports
import pandas as pd
import matplotlib.pyplot as plt
from typing import List
from enum import Enum
import qis as qis

# package
from optimalportfolios import Constraints, backtest_rolling_optimal_portfolio, PortfolioObjective
from optimalportfolios.examples.universe import fetch_benchmark_universe_data


def run_multi_optimisers_backtest(prices: pd.DataFrame,
                                  benchmark_prices: pd.DataFrame,
                                  group_data: pd.Series,
                                  time_period: qis.TimePeriod,  # for weights
                                  perf_time_period: qis.TimePeriod  # for reporting
                                  ) -> List[plt.Figure]:
    """
    backtest multi optimisers
    test maximum diversification optimiser to span parameter
    span is number period for ewm filter
    span = 20 for daily data implies last 20 (trading) days contribute 50% of weight for covariance estimation
    we test sensitivity from fast (small span) to slow (large span)
    """
    portfolio_objectives = {'EqualRisk': PortfolioObjective.EQUAL_RISK_CONTRIBUTION,
                            'MinVariance': PortfolioObjective.MIN_VARIANCE,
                            'MaxDiversification': PortfolioObjective.MAX_DIVERSIFICATION,
                            'MaxSharpe': PortfolioObjective.MAXIMUM_SHARPE_RATIO,
                            'MaxCarraMixture': PortfolioObjective.MAX_CARA_MIXTURE}

    # set global constaints for portfolios
    constraints = Constraints(is_long_only=True,
                               min_weights=pd.Series(0.0, index=prices.columns),
                               max_weights=pd.Series(0.5, index=prices.columns))

    # now create a list of portfolios
    portfolio_datas = []
    for ticker, portfolio_objective in portfolio_objectives.items():
        print(ticker)
        portfolio_data = backtest_rolling_optimal_portfolio(prices=prices,
                                                            portfolio_objective=portfolio_objective,
                                                            constraints=constraints,
                                                            time_period=time_period,
                                                            perf_time_period=perf_time_period,
                                                            returns_freq='W-WED',  # covar matrix estimation on weekly returns
                                                            rebalancing_freq='QE',  # portfolio rebalancing
                                                            span=52,  # ewma span for covariance matrix estimation: span = 1y of weekly returns
                                                            roll_window=5*52,  # linked to returns at rebalancing_freq: 5y of data for max sharpe and mixture carra
                                                            carra=0.5,  # carra parameter
                                                            n_mixures=3,  # for mixture carra utility
                                                            rebalancing_costs=0.0010,  # 10bp for rebalancin
                                                            weight_implementation_lag=1,  # weights are implemnted next day after comuting
                                                            ticker=f"{ticker}"  # portfolio id
                                                            )
        portfolio_data.set_group_data(group_data=group_data)
        portfolio_datas.append(portfolio_data)

    # run cross portfolio report
    multi_portfolio_data = qis.MultiPortfolioData(portfolio_datas=portfolio_datas, benchmark_prices=benchmark_prices)
    figs = qis.generate_multi_portfolio_factsheet(multi_portfolio_data=multi_portfolio_data,
                                                  time_period=time_period,
                                                  add_strategy_factsheets=False,
                                                  **qis.fetch_default_report_kwargs(time_period=time_period))
    return figs


class LocalTests(Enum):
    MULTI_OPTIMISERS_BACKTEST = 1


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    import optimalportfolios.local_path as local_path

    if local_test == LocalTests.MULTI_OPTIMISERS_BACKTEST:
        prices, benchmark_prices, ac_loadings, benchmark_weights, group_data, ac_benchmark_prices = fetch_benchmark_universe_data()
        time_period = qis.TimePeriod(start='31Dec1998', end=prices.index[-1])  # backtest start: need 6y of data for rolling Sharpe and max mixure portfolios
        perf_time_period = qis.TimePeriod(start='31Dec2004', end=prices.index[-1])  # backtest reporting
        figs = run_multi_optimisers_backtest(prices=prices,
                                             benchmark_prices=benchmark_prices,
                                             group_data=group_data,
                                             time_period=time_period,
                                             perf_time_period=perf_time_period)

        # save png and pdf
        qis.save_fig(fig=figs[0], file_name=f"multi_optimisers_backtest", local_path=f"figures/")
        qis.save_figs_to_pdf(figs=figs,
                             file_name=f"multi_optimisers_backtest",
                             orientation='landscape',
                             local_path=local_path.get_output_path())
    plt.show()


if __name__ == '__main__':

    run_local_test(local_test=LocalTests.MULTI_OPTIMISERS_BACKTEST)
