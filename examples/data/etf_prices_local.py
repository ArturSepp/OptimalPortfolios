"""the eight-ETF cross-asset universe the local diagnostics run on.

Both halves of this module need the author's machine: ``update_test_prices`` downloads from
Yahoo, and ``load_test_data`` reads the saved panel out of ``RESOURCE_PATH``. Neither is
reachable in CI, and nothing here asserts anything.

Hence the ``_local`` suffix. The file was called ``test_data.py`` and so matched pytest's
``test_*.py`` pattern: it collected no tests but was still *imported* at collection time,
which is the mechanism behind the two CI failures a module-level ``yfinance`` import has
already caused here. The ``yfinance`` import is function-local below, but the naming is what
made that a rule to remember rather than a property of the layout.
"""

# imports
import pandas as pd
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
    """Download the test universe prices and save them as the committed fixture."""
    try:
        import yfinance as yf
    except ImportError as e:
        raise ImportError("update_test_prices needs yfinance: "
                          "pip install optimalportfolios[data]") from e
    tickers = list(UNIVERSE_DATA.keys())
    prices = yf.download(tickers=tickers, start="2003-12-31", end=None, ignore_tz=True, auto_adjust=True)
    prices = prices['Close']
    prices = prices.asfreq('B', method='ffill')  # rescale to business days
    prices = prices[tickers]  # align order
    qis.save_df_to_csv(df=prices, file_name=FILE_NAME, local_path=local_path.get_resource_path())
    return prices


def load_test_data() -> pd.DataFrame:
    """Load the committed test universe prices."""
    prices = qis.load_df_from_csv(file_name=FILE_NAME, local_path=local_path.get_resource_path())
    return prices


class LocalTests(Enum):
    """Local diagnostic scenarios ``run_local_test`` can run."""
    UPDATE_TEST_PRICES = 1
    LOAD_TEST_PRICES = 2


def run_local_test(local_test: LocalTests):
    """Run local tests for product_development and debugging purposes.

    These are integration tests that download real universe and generate reports.
    Use for quick verification during product_development.
    """

    if local_test == LocalTests.UPDATE_TEST_PRICES:
        prices = update_test_prices()
        print(prices)

    elif local_test == LocalTests.LOAD_TEST_PRICES:
        prices = load_test_data()
        print(prices)


if __name__ == '__main__':

    run_local_test(local_test=LocalTests.UPDATE_TEST_PRICES)
