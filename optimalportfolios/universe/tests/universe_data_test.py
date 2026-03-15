
import pandas as pd
import qis as qis
from enum import Enum
import yfinance as yf

import optimalportfolios.local_path as lp
from optimalportfolios.universe import UniverseData, MetadataField


def fetch_universe_data(start_date: str = "2003-12-31") -> UniverseData:
    """
    fetch a universe of etfs
    define custom universe with asset class grouping
    5 asset groups with 3 etfs in each
    """
    universe_data = dict(SPY='Equities',
                         QQQ='Equities',
                         EEM='Equities',
                         TLT='Bonds',
                         IEF='Bonds',
                         TIP='Bonds',
                         IGSB='IG',
                         LQD='IG',
                         EMB='IG',
                         HYG='HighYield',
                         SHYG='HighYield',
                         FALN='HighYield',
                         GLD='Commodts',
                         GSG='Commodts',
                         COMT='Commodts')
    group_data = pd.Series(universe_data, name=MetadataField.ASSET_CLASS)  # for portfolio reporting
    names = pd.Series([f"{ticker}_{ac}" for ticker, ac in group_data.to_dict().items()], index=group_data.index, name=MetadataField.NAME)
    group_loadings_level1 = qis.set_group_loadings(group_data=group_data)
    tickers = list(universe_data.keys())
    prices = yf.download(tickers=tickers, start=start_date, end=None, ignore_tz=True, auto_adjust=True)['Close'][tickers]
    prices = prices.asfreq('B').ffill()

    metadata = pd.concat([names,
                          group_data,
                          pd.Series('USD', index=group_data.index, name=MetadataField.CURRENCY)
                          ], axis=1)

    universe_data = UniverseData(prices=prices, metadata=metadata, group_loadings_level1=group_loadings_level1)

    return universe_data


def fetch_risk_factor_prices(start_date: str = "2003-12-31") -> pd.DataFrame:
    tickers = {'SPY': 'Equity', 'TLT': 'Rate'}
    risk_factor_prices = yf.download(tickers=list(tickers.keys()), start=start_date, end=None, ignore_tz=True, auto_adjust=True)['Close']
    risk_factor_prices = risk_factor_prices.rename(tickers)
    return risk_factor_prices



class LocalTests(Enum):
    CREATE_UNIVERSE_DATA = 1
    LOAD_UNIVERSE_DATA = 2


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real universe and generate reports.
    Use for quick verification during development.
    """

    if local_test == LocalTests.CREATE_UNIVERSE_DATA:
        universe_data = fetch_universe_data()
        print(universe_data)
        universe_data.save(file_name='universe_test', local_path=lp.get_output_path())

    elif local_test == LocalTests.LOAD_UNIVERSE_DATA:
        universe_data = UniverseData.load(file_name='universe_test', local_path=lp.get_output_path())
        print(universe_data)


if __name__ == '__main__':

    run_local_test(local_test=LocalTests.CREATE_UNIVERSE_DATA)
