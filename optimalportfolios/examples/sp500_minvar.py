"""
run Minimum Variance portfolio optimiser for S&P 500 universe
The goal is to backtest the sensetivity of squeezing of the covariance matrix using SSRN paper
Squeezing Financial Noise: A Novel Approach to Covariance Matrix Estimation
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4986939
"""

# packages
import pandas as pd
import qis as qis
import yfinance as yf
from typing import List
from enum import Enum

# optimalportfolios
from optimalportfolios import PortfolioObjective, Constraints, rolling_quadratic_optimisation, CovarEstimator


def run_cross_backtest(prices: pd.DataFrame,
                       inclusion_indicators: pd.DataFrame,
                       group_data: pd.Series,
                       time_period: qis.TimePeriod,
                       squeeze_factors: List[float] = (0.0, 0.125, 0.250, 0.375, 0.5, 0.7, 0.9)
                       ) -> List[qis.PortfolioData]:
    """Runs cross-validation backtest for minimum variance portfolios with different covariance shrinkage factors.

    Tests sensitivity of portfolio performance to covariance matrix shrinkage by running
    backtests across multiple squeeze factors. Uses long-only constraints with 5% max weight.

    Args:
        prices: Asset price DataFrame.
        inclusion_indicators: S&P 500 inclusion indicators DataFrame.
        group_data: Sector group data Series.
        time_period: Backtest time period.
        squeeze_factors: List of shrinkage factors for covariance estimation. Defaults to (0.0, 0.125, 0.250, 0.375, 0.5, 0.7, 0.9).

    Returns:
        List of PortfolioData objects, one for each squeeze factor.
    """
    constraints0 = Constraints(is_long_only=True,
                               min_weights=pd.Series(0.0, index=prices.columns),
                               max_weights=pd.Series(0.05, index=prices.columns))

    portfolio_datas = []
    for squeeze_factor in squeeze_factors:
        covar_estimator = CovarEstimator(squeeze_factor=squeeze_factor, returns_freqs='W-WED', rebalancing_freq='QE')
        weights = rolling_quadratic_optimisation(prices=prices,
                                                 constraints0=constraints0,
                                                 portfolio_objective=PortfolioObjective.MIN_VARIANCE,
                                                 time_period=time_period,
                                                 inclusion_indicators=inclusion_indicators,
                                                 covar_estimator=covar_estimator)
        portfolio_data = qis.backtest_model_portfolio(prices=time_period.locate(prices),
                                                      weights=weights,
                                                      ticker=f"squeeze={squeeze_factor: 0.3f}",
                                                      funding_rate=None,
                                                      weight_implementation_lag=1,
                                                      rebalancing_costs=0.0030)
        portfolio_data.set_group_data(group_data=group_data)
        portfolio_datas.append(portfolio_data)
    return portfolio_datas


class LocalTests(Enum):
    CROSS_BACKTEST = 1


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """
    import quant_strats.local_path as lp
    from optimalportfolios.examples.sp500_universe import load_sp500_universe_yahoo

    if local_test == LocalTests.CROSS_BACKTEST:

        # time_period = qis.TimePeriod('31Dec2010', '31Jan2024', tz='UTC')
        time_period = qis.TimePeriod('31Dec2010', '31Jan2024')
        # define squeeze_factors
        squeeze_factors = [0.0, 0.25, 0.5]
        # squeeze_factors = [0.0, 0.125, 0.250, 0.375, 0.5, 0.7, 0.9]

        prices, inclusion_indicators, group_data = load_sp500_universe_yahoo()

        portfolio_datas = run_cross_backtest(prices=prices,
                                             inclusion_indicators=inclusion_indicators,
                                             group_data=group_data,
                                             time_period=time_period,
                                             squeeze_factors=squeeze_factors)

        # run cross portfolio report
        benchmark_prices = yf.download('SPY', start="2003-12-31", end=None, ignore_tz=True, auto_adjust=True)['Close'].asfreq('B').ffill()
        multi_portfolio_data = qis.MultiPortfolioData(portfolio_datas=portfolio_datas,
                                                      benchmark_prices=benchmark_prices)

        figs = qis.generate_multi_portfolio_factsheet(multi_portfolio_data=multi_portfolio_data,
                                                      time_period=time_period,
                                                      add_benchmarks_to_navs=True,
                                                      add_strategy_factsheets=False,
                                                      **qis.fetch_default_report_kwargs(time_period=time_period))

        # save report to pdf and png
        qis.save_figs_to_pdf(figs=figs,
                             file_name=f"sp500_squeeze_portfolio_factsheet",
                             orientation='landscape',
                             local_path=lp.get_output_path())


if __name__ == '__main__':

    run_local_test(local_test=LocalTests.CROSS_BACKTEST)
