"""
run Minimum Variance portfolio optimiser for S&P 500 universe
S&P 500 universe compositions are obtained from https://github.com/fja05680/sp500
prices are fetched from yfinance
note that some of the companies ever included in the S&P500 are de-listed and yfinance does not have data on them
I run backtest from 31Dec2010 which should be less sensitive to de-listing bias
By optimisation, I account for the index inclusions using dataframe with inclusion_indicators

The goal is to backtest the sensetivity of squeezing of the covariance matrix using SSRN paper
Squeezing Financial Noise: A Novel Approach to Covariance Matrix Estimation
https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4986939
"""

# packages
import pandas as pd
import qis as qis
import yfinance as yf
from typing import Tuple, List
from enum import Enum

# optimalportfolios
from optimalportfolios import (PortfolioObjective, Constraints, rolling_quadratic_optimisation, CovarEstimator)
from optimalportfolios.local_path import get_resource_path

# path to save universe data
LOCAL_PATH = f"{get_resource_path()}//sp500//"
# download from source: https://github.com/fja05680/sp500
SP500_FILE = "S&P 500 Historical Components & Changes(08-17-2024).csv"


def create_sp500_universe():
    """
    use SP500_FILE file to fetch the list of universe
    load price and industry data using yfinance
    """
    def create_inclusion_indicators(universe: pd.DataFrame) -> pd.DataFrame:
        inclusion_indicators = {}
        for date in universe.index:
            tickers = universe.loc[date, :].apply(lambda x: sorted(x.split(','))).to_list()[0]
            inclusion_indicators[date] = pd.Series(1.0, index=tickers)
        inclusion_indicators = pd.DataFrame.from_dict(inclusion_indicators, orient='index').sort_index()
        return inclusion_indicators

    def fetch_universe_prices(tickers: List[str]) -> pd.DataFrame:
        prices = yf.download(tickers=tickers, start=None, end=None, ignore_tz=True)['Adj Close']
        return prices[tickers]

    def fetch_universe_industry(tickers: List[str]) -> pd.Series:
        group_data = {}
        for ticker in tickers:
            this = yf.Ticker(ticker).info
            if 'sector' in this:
                group_data[ticker] = this['sector']
            else:
                group_data[ticker] = 'unclassified'
        return pd.Series(group_data)

    universe = pd.read_csv(f"{LOCAL_PATH}{SP500_FILE}", index_col='date')
    inclusion_indicators = create_inclusion_indicators(universe)
    prices = fetch_universe_prices(tickers=inclusion_indicators.columns.to_list())
    # remove all nans
    prices = prices.dropna(axis=1, how='all').asfreq('B', method='ffill')
    group_data = fetch_universe_industry(tickers=prices.columns.to_list())
    inclusion_indicators = inclusion_indicators[prices.columns]
    qis.save_df_to_csv(df=prices, file_name='sp500_prices', local_path=LOCAL_PATH)
    qis.save_df_to_csv(df=inclusion_indicators, file_name='sp500_inclusions', local_path=LOCAL_PATH)
    qis.save_df_to_csv(df=group_data.to_frame(), file_name='sp500_groups', local_path=LOCAL_PATH)


def load_sp500_universe() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    prices = qis.load_df_from_csv(file_name='sp500_prices', local_path=LOCAL_PATH)
    inclusion_indicators = qis.load_df_from_csv(file_name='sp500_inclusions', local_path=LOCAL_PATH)
    inclusion_indicators.index = inclusion_indicators.index.tz_localize(tz=prices.index.tz)  # align tz info
    group_data = qis.load_df_from_csv(file_name='sp500_groups', parse_dates=False, local_path=LOCAL_PATH).iloc[:, 0]
    return prices, inclusion_indicators, group_data


def run_cross_backtest(time_period: qis.TimePeriod,
                       squeeze_factors: List[float] = (0.0, 0.125, 0.250, 0.375, 0.5, 0.7, 0.9)
                       ):
    # run cross-backtest for sensetivity to
    prices, inclusion_indicators, group_data = load_sp500_universe()

    constraints0 = Constraints(is_long_only=True,
                               min_weights=pd.Series(0.0, index=prices.columns),
                               max_weights=pd.Series(0.05, index=prices.columns))

    portfolio_datas = []
    for squeeze_factor in squeeze_factors:
        covar_estimator = CovarEstimator(squeeze_factor=squeeze_factor, returns_freq='W-WED', rebalancing_freq='QE')
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


class UnitTests(Enum):
    CREATE_UNIVERSE_DATA = 1
    CROSS_BACKTEST = 2


def run_unit_test(unit_test: UnitTests):

    if unit_test == UnitTests.CREATE_UNIVERSE_DATA:
        create_sp500_universe()

    elif unit_test == UnitTests.CROSS_BACKTEST:

        time_period = qis.TimePeriod('31Dec2010', '31Jan2024', tz='UTC')
        # define squeeze_factors
        squeeze_factors = [0.0, 0.25, 0.5]
        # squeeze_factors = [0.0, 0.125, 0.250, 0.375, 0.5, 0.7, 0.9]

        portfolio_datas = run_cross_backtest(time_period=time_period,
                                             squeeze_factors=squeeze_factors)

        # run cross portfolio report
        benchmark_prices = yf.download('SPY', start=None, end=None, ignore_tz=True)['Adj Close'].asfreq('B').ffill()
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
                             local_path="C://Users//Artur//OneDrive//analytics//outputs")


if __name__ == '__main__':

    unit_test = UnitTests.CROSS_BACKTEST

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)

