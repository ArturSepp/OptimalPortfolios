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
                     SHY='Bonds',
                     LQD='Credit',
                     HYG='HighYield',
                     GLD='Gold')


def update_test_prices() -> pd.DataFrame:
    tickers = list(UNIVERSE_DATA.keys())
    prices = yf.download(tickers=tickers,
                         start=None, end=None,
                         ignore_tz=True)['Adj Close']
    prices = prices[tickers]  # align order
    qis.save_df_to_csv(df=prices, file_name=FILE_NAME, local_path=local_path.get_resource_path())
    return prices


def load_test_data() -> pd.DataFrame:
    prices = qis.load_df_from_csv(file_name=FILE_NAME, local_path=local_path.get_resource_path())
    return prices


class UnitTests(Enum):
    UPDATE_TEST_PRICES = 1
    LOAD_TEST_PRICES = 2


def run_unit_test(unit_test: UnitTests):

    if unit_test == UnitTests.UPDATE_TEST_PRICES:
        prices = update_test_prices()
        print(prices)

    elif unit_test == UnitTests.LOAD_TEST_PRICES:
        prices = load_test_data()
        print(prices)


if __name__ == '__main__':

    unit_test = UnitTests.UPDATE_TEST_PRICES

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
