"""
create composition of S&P 500 universe compositions using members in https://github.com/fja05680/sp500
note that some of the companies ever included in the S&P500 are de-listed and yfinance does not have data on them
"""

# packages
import pandas as pd
import qis as qis
import yfinance as yf
from typing import Tuple, List
from enum import Enum

from optimalportfolios.local_path import get_resource_path

# path to save universe data
LOCAL_PATH = f"{get_resource_path()}//sp500//"
# download from source: https://github.com/fja05680/sp500
SP500_FILE = "S&P 500 Historical Components & Changes(07-12-2025).csv"


def create_inclusion_indicators() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create binary inclusion indicator DataFrames for S&P 500 constituents over time.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: Regular and Bloomberg-formatted ticker
            inclusion indicators with dates as index and 1.0 for included stocks.
    """
    universe = pd.read_csv(f"{LOCAL_PATH}{SP500_FILE}", index_col='date')
    inclusion_indicators = {}
    inclusion_indicators_bbg = {}
    for date in universe.index:
        tickers = universe.loc[date, :].apply(lambda x: sorted(x.split(','))).to_list()[0]
        bbg_tickers = [f"{x} US Equity" for x in tickers]
        inclusion_indicators[date] = pd.Series(1.0, index=tickers)
        inclusion_indicators_bbg[date] = pd.Series(1.0, index=bbg_tickers)
    inclusion_indicators = pd.DataFrame.from_dict(inclusion_indicators, orient='index').sort_index()
    inclusion_indicators_bbg = pd.DataFrame.from_dict(inclusion_indicators_bbg, orient='index').sort_index()
    return inclusion_indicators, inclusion_indicators_bbg


def create_sp500_universe_with_yahoo(local_path: str = LOCAL_PATH) -> None:
    """
    Fetch S&P 500 price and sector data from Yahoo Finance and save to CSV files.

    Downloads historical prices and industry classifications for S&P 500 constituents,
    then saves the cleaned data as 'sp500_prices_yahoo.csv', 'sp500_inclusions_yahoo.csv',
    and 'sp500_groups_yahoo.csv' files.
    """
    inclusion_indicators, inclusion_indicators_bbg = create_inclusion_indicators()
    def fetch_universe_prices(tickers: List[str]) -> pd.DataFrame:
        prices = yf.download(tickers=tickers, start="2003-12-31", end=None, ignore_tz=True, auto_adjust=True)['Close']
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

    prices = fetch_universe_prices(tickers=inclusion_indicators.columns.to_list())
    # remove all nans
    prices = prices.dropna(axis=1, how='all').asfreq('B', method='ffill')
    group_data = fetch_universe_industry(tickers=prices.columns.to_list())
    inclusion_indicators = inclusion_indicators[prices.columns]
    qis.save_df_to_csv(df=prices, file_name='sp500_prices_yahoo', local_path=local_path)
    qis.save_df_to_csv(df=inclusion_indicators, file_name='sp500_inclusions_yahoo', local_path=local_path)
    qis.save_df_to_csv(df=group_data.to_frame(), file_name='sp500_groups_yahoo', local_path=local_path)


def load_sp500_universe_yahoo(local_path: str = LOCAL_PATH) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """Loads S&P 500 universe data from Yahoo Finance CSV files.

    Loads prices, inclusion indicators, and sector group data. Aligns timezone
    information between prices and inclusion indicators.

    Args:
        local_path: Path to directory containing CSV files. Defaults to LOCAL_PATH.

    Returns:
        Tuple of (prices DataFrame, inclusion indicators DataFrame, group data Series).
    """
    prices = qis.load_df_from_csv(file_name='sp500_prices_yahoo', local_path=local_path)
    inclusion_indicators = qis.load_df_from_csv(file_name='sp500_inclusions_yahoo', local_path=local_path)
    inclusion_indicators.index = inclusion_indicators.index.tz_localize(tz=prices.index.tz)  # align tz info
    group_data = qis.load_df_from_csv(file_name='sp500_groups_yahoo', parse_dates=False, local_path=local_path).iloc[:, 0]
    return prices, inclusion_indicators, group_data


def create_sp500_universe_with_bloomberg(start_date: pd.Timestamp = pd.Timestamp('31Dec1995'),
                                         local_path: str = LOCAL_PATH
                                         ) -> None:
    """Creates S&P 500 universe data using Bloomberg API.

    Fetches prices, market cap, sector data, and inclusion indicators for S&P 500
    constituents. Filters out delisted stocks without sectors and saves results to CSV.

    Args:
        start_date: Start date for time series data. Defaults to Dec 31, 1995.
        local_path: Path to directory for saving CSV files. Defaults to LOCAL_PATH.
    """
    from bbg_fetch import fetch_field_timeseries_per_tickers, fetch_fundamentals
    inclusion_indicators, inclusion_indicators_bbg = create_inclusion_indicators()

    tickers = inclusion_indicators_bbg.columns.to_list()
    print(tickers)

    # first get industries
    group_datas = fetch_fundamentals(tickers=tickers, fields=['gics_sector_name'])
    # drop stocks without sectors: for delisted stocks their tickers can become funds or etfs
    clean_group_datas = group_datas.dropna()
    print(f"original n = {len(group_datas.index)}, new n = {len(clean_group_datas.index)}")

    tickers = clean_group_datas.index.to_list()
    prices = fetch_field_timeseries_per_tickers(tickers=tickers, start_date=start_date, freq='B')
    # remove all nans
    prices = prices.dropna(axis=1, how='all')

    market_cap = fetch_field_timeseries_per_tickers(tickers=tickers, start_date=start_date, freq='B', field='CUR_MKT_CAP')
    market_cap = market_cap.reindex(columns=prices.columns)

    group_datas = clean_group_datas.reindex(index=prices.columns)

    inclusion_indicators_bbg = inclusion_indicators_bbg.reindex(columns=prices.columns)
    qis.save_df_to_csv(df=prices, file_name='sp500_prices_bloomberg', local_path=local_path)
    qis.save_df_to_csv(df=market_cap, file_name='sp500_market_cap_bloomberg', local_path=local_path)
    qis.save_df_to_csv(df=inclusion_indicators_bbg, file_name='sp500_inclusions_bloomberg', local_path=local_path)
    qis.save_df_to_csv(df=group_datas, file_name='sp500_groups_bloomberg', local_path=local_path)


def load_sp500_universe_bloomberg(local_path: str = LOCAL_PATH
                                  ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    """Loads S&P 500 universe data from Bloomberg CSV files.

    Loads prices, market cap, inclusion indicators, and sector group data
    previously saved by create_sp500_universe_with_bloomberg().

    Args:
        local_path: Path to directory containing CSV files. Defaults to LOCAL_PATH.

    Returns:
        Tuple of (prices DataFrame, market cap DataFrame, inclusion indicators DataFrame, group data Series).
    """
    prices = qis.load_df_from_csv(file_name='sp500_prices_bloomberg', local_path=local_path)
    market_cap = qis.load_df_from_csv(file_name='sp500_market_cap_bloomberg', local_path=local_path)
    inclusion_indicators = qis.load_df_from_csv(file_name='sp500_inclusions_bloomberg', local_path=local_path)
    group_data = qis.load_df_from_csv(file_name='sp500_groups_bloomberg', parse_dates=False, local_path=local_path).iloc[:, 0]
    return prices, market_cap, inclusion_indicators, group_data


class LocalTests(Enum):
    CREATE_UNIVERSE_DATA_WITH_YAHOO = 1
    CREATE_UNIVERSE_DATA_WITH_BLOOMBERG = 2
    LOAD = 3


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    if local_test == LocalTests.CREATE_UNIVERSE_DATA_WITH_YAHOO:
        create_sp500_universe_with_yahoo()

    elif local_test == LocalTests.CREATE_UNIVERSE_DATA_WITH_BLOOMBERG:
        create_sp500_universe_with_bloomberg()

    elif local_test == LocalTests.LOAD:
        prices, market_cap, inclusion_indicators, group_data = load_sp500_universe_bloomberg()
        print(prices)
        print(market_cap)
        print(inclusion_indicators)
        print(group_data)


if __name__ == '__main__':

    run_local_test(local_test=LocalTests.CREATE_UNIVERSE_DATA_WITH_BLOOMBERG)
