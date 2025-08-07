"""
implement test data for optimisations
use update and save data for speed-up of test cases
"""

# imports
import pandas as pd
import yfinance as yf
import qis
import optimalportfolios.local_path as local_path
from enum import Enum

FILE_NAME = 'test_prices'

UNIVERSE_DATA = dict(SPY='Equities',
                     QQQ='Equities',
                     EEM='Equities',
                     TLT='Bonds',
                     IEF='Bonds',
                     LQD='Credit',
                     HYG='HighYield',
                     GLD='Gold')


def update_test_prices() -> pd.DataFrame:
    tickers = list(UNIVERSE_DATA.keys())
    prices = yf.download(tickers=tickers, start="2003-12-31", end=None, ignore_tz=True, auto_adjust=True)
    prices = prices['Close']
    prices = prices.asfreq('B', method='ffill')  # rescale to business days
    prices = prices[tickers]  # align order
    qis.save_df_to_csv(df=prices, file_name=FILE_NAME, local_path=local_path.get_resource_path())
    return prices


def load_test_data() -> pd.DataFrame:
    prices = qis.load_df_from_csv(file_name=FILE_NAME, local_path=local_path.get_resource_path())
    return prices


class LocalTests(Enum):
    UPDATE_TEST_PRICES = 1
    LOAD_TEST_PRICES = 2


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    if local_test == LocalTests.UPDATE_TEST_PRICES:
        prices = update_test_prices()
        print(prices)

    elif local_test == LocalTests.LOAD_TEST_PRICES:
        prices = load_test_data()
        print(prices)


if __name__ == '__main__':

    run_local_test(local_test=LocalTests.UPDATE_TEST_PRICES)
