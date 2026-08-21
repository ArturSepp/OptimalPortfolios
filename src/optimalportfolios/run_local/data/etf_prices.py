"""Shared eight-ETF price data for source-adjacent development runners."""

# imports
import pandas as pd
import qis
import optimalportfolios.local_path as local_path

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
