"""
generate and load prices for portfolio allocation problem
update_prices_with_bloomberg() must used with  Bloomberg open
update_prices_with_yf() uses yfinance + some data uploaded manually
NB: CmcScraper stopped working so now only option is to use bloomberg
"""
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from typing import List, Literal, Union, Optional
from enum import Enum
# from cryptocmd import CmcScraper  its stopped working
import qis

# add the local path to data files
LOCAL_PATH = 'C://Users//artur//OneDrive//analytics//my_github//OptimalPortfolios//optimalportfolios//examples//crypto_allocation//data//'

# data sources
BTC_PRICES_FROM_2010 = 'BTC_from_2010'  # csv data with static BTC from2010 upto 31Jul2022
HFRXGL_PRICE = 'HFRX_historical_HFRXGL'  # global HF from https://www.hfr.com/indices/hfrx-global-hedge-fund-index - download daily data, remove top and bottom rows with descriptive data
CTA_PRICE = 'CTA_Historical'  # SG CTA Index from https://wholesale.banking.societegenerale.com/en/prime-services-indices/
MACRO_PRICE = 'Macro_Trading_Index_Historical'  # Macro SCTas from https://wholesale.banking.societegenerale.com/fileadmin/indices_feeds/Macro_Trading_Index_Historical.xls

PRICE_DATA_FILE = 'crypto_allocation_prices'
PRICE_DATA_FILE_UPDATED = 'crypto_allocation_prices_updated'


class Assets(Enum):
    # with name and bloomberg ticker
    BTC = 'BTC'
    ETH = 'ETH'
    BAL = '60/40'
    HF = 'HFs'
    PE = 'PE'
    RE = 'RealEstate'
    MACRO = 'Macro'
    CTA = 'SG CTA'
    GLD = 'Gold'
    COMS = 'Commodities'


def update_prices_with_yf() -> pd.DataFrame:
    """
    generate price data using yfinance
    """
    btc = create_btc_price()
    eth = create_eth_price(btc_price=qis.load_df_from_csv(file_name=BTC_PRICES_FROM_2010, local_path=LOCAL_PATH).iloc[:, 0])
    bal = create_balanced_price()

    # REET starts from Jul 08, 2014 - use IYR for backfill
    iyr = yf.download('IYR', start="2003-12-31", end=None, ignore_tz=True, auto_adjust=True)['Close'].rename(Assets.RE.value)
    reet = yf.download('REET', start="2003-12-31", end=None, ignore_tz=True, auto_adjust=True)['Close'].rename(Assets.RE.value)
    re = qis.bfill_timeseries(df_newer=reet, df_older=iyr, is_prices=True)

    # COMT starts from 2014-10-16, use GSG for backfill
    coms0 = yf.download('GSG', start="2003-12-31", end=None, ignore_tz=True, auto_adjust=True)['Close'].rename(Assets.COMS.value)
    coms1 = yf.download('COMT', start="2003-12-31", end=None, ignore_tz=True, auto_adjust=True)['Close'].rename(Assets.COMS.value)
    coms = qis.bfill_timeseries(df_newer=coms1, df_older=coms0, is_prices=True)

    # use local copies
    hf = qis.load_df_from_csv(file_name=HFRXGL_PRICE, local_path=LOCAL_PATH)['Index Value'].rename(Assets.HF.value).sort_index()
    cta = qis.load_df_from_excel(file_name=CTA_PRICE, local_path=LOCAL_PATH).iloc[:, 0].rename(Assets.CTA.value)
    macro = qis.load_df_from_excel(file_name=MACRO_PRICE, local_path=LOCAL_PATH).iloc[:, 0].rename(Assets.MACRO.value)

    # etfs
    gld = yf.download('GLD', start="2003-12-31", end=None, ignore_tz=True, auto_adjust=True)['Close'].rename(Assets.GLD.value)
    pe = yf.download('PSP', start="2003-12-31", end=None, ignore_tz=True, auto_adjust=True)['Close'].rename(Assets.PE.value)

    prices = pd.concat([bal, btc, eth, hf, pe, re, macro, cta, coms, gld], axis=1)
    # to business day frequency
    prices = prices.asfreq('B', method='ffill').ffill()
    qis.save_df_to_csv(prices, file_name=PRICE_DATA_FILE, local_path=LOCAL_PATH)

    return prices


def update_prices_with_bloomberg() -> pd.DataFrame:
    """
    generate price data using yfinance
    """
    from bbg_fetch import fetch_field_timeseries_per_tickers

    btc = fetch_field_timeseries_per_tickers(tickers={'XBTUSD Curncy': Assets.BTC.value}).iloc[:, 0]
    eth_bbg = fetch_field_timeseries_per_tickers(tickers={'XETUSD Curncy': Assets.ETH.value}).iloc[:, 0]
    # bfill eth with bal
    eth = qis.bfill_timeseries(df_newer=eth_bbg, df_older=btc, freq='D', is_prices=True)
    bal = create_balanced_price()

    # REET starts from Jul 08, 2014 - use IYR for backfill
    iyr = yf.download('IYR', start="2003-12-31", end=None, ignore_tz=True, auto_adjust=True)['Close'].rename(Assets.RE.value)
    reet = yf.download('REET', start="2003-12-31", end=None, ignore_tz=True, auto_adjust=True)['Close'].rename(Assets.RE.value)
    re = qis.bfill_timeseries(df_newer=reet, df_older=iyr, is_prices=True)

    # COMT starts from 2014-10-16, use GSG for backfill
    coms0 = yf.download('GSG', start="2003-12-31", end=None, ignore_tz=True, auto_adjust=True)['Close'].rename(Assets.COMS.value)
    coms1 = yf.download('COMT', start="2003-12-31", end=None, ignore_tz=True, auto_adjust=True)['Close'].rename(Assets.COMS.value)
    coms = qis.bfill_timeseries(df_newer=coms1, df_older=coms0, is_prices=True)

    # use bbg
    hf = fetch_field_timeseries_per_tickers(tickers={'HFRXGL Index': Assets.HF.value}).iloc[:, 0]
    cta = fetch_field_timeseries_per_tickers(tickers={'NEIXCTA Index': Assets.CTA.value}).iloc[:, 0]
    macro = fetch_field_timeseries_per_tickers(tickers={'HFRIMDT Index': Assets.MACRO.value}).iloc[:, 0]

    # etfs
    gld = yf.download('GLD', start="2003-12-31", end=None, ignore_tz=True, auto_adjust=True)['Close'].rename(Assets.GLD.value)
    pe = yf.download('PSP', start="2003-12-31", end=None, ignore_tz=True, auto_adjust=True)['Close'].rename(Assets.PE.value)

    prices = pd.concat([bal, btc, eth, hf, pe, re, macro, cta, coms, gld], axis=1).sort_index()
    # to business day frequency
    prices = prices.asfreq('B', method='ffill').ffill()
    qis.save_df_to_csv(prices, file_name=PRICE_DATA_FILE_UPDATED, local_path=LOCAL_PATH)

    return prices


def create_btc_price() -> pd.Series:
    """
    backfill BTC_PRICES_FROM_2010
    """
    btc_bbg = qis.load_df_from_csv(file_name=BTC_PRICES_FROM_2010, local_path=LOCAL_PATH).iloc[:, 0].rename(Assets.BTC.value)
    new_bbg = fetch_cmc_price(ticker='BTC').rename(Assets.BTC.value)
    btc_price = qis.bfill_timeseries(df_newer=new_bbg, df_older=btc_bbg, freq='D', is_prices=True)
    return btc_price


def create_eth_price(btc_price: pd.Series) -> pd.Series:
    btc_price = btc_price.rename(Assets.ETH.value)
    mc_price = fetch_cmc_price(ticker='ETH').rename(Assets.ETH.value)
    eth_price = qis.bfill_timeseries(df_newer=mc_price, df_older=btc_price, freq='D', is_prices=True)
    return eth_price


def create_balanced_price() -> pd.Series:
    spy = yf.download('SPY', start=None, end=None, ignore_tz=True)['Close'].rename('SPY')
    ief = yf.download('IEF', start=None, end=None, ignore_tz=True)['Close'].rename('IEF')
    prices = pd.concat([spy, ief], axis=1).dropna()
    balanced = qis.backtest_model_portfolio(prices=prices,
                                            weights=[0.6, 0.4],
                                            rebalancing_freq='QE',
                                            is_rebalanced_at_first_date=True,
                                            rebalancing_costs=0.005,
                                            ticker=Assets.BAL.value)
    return balanced.nav


def fetch_cmc_price(ticker: str = 'ETH') -> pd.Series:
    scraper = CmcScraper(coin_code=ticker, all_time=True, order_ascending=True)
    data = scraper.get_dataframe()
    data['Date'] = pd.to_datetime(data['Date'])
    data = data.set_index('Date', drop=True)
    data.index = data.index.normalize()
    return data['Close']


def load_prices(assets: List[Assets] = None,
                crypto_asset: Optional[Union[Literal['BTC', 'ETH'], str]] = 'BTC',
                is_updated: bool = False
                ) -> pd.DataFrame:
    if is_updated:
        prices = qis.load_df_from_csv(file_name=PRICE_DATA_FILE_UPDATED, local_path=LOCAL_PATH)
    else:
        prices = qis.load_df_from_csv(file_name=PRICE_DATA_FILE, local_path=LOCAL_PATH)
    if assets is not None:
        prices = prices[[x.value for x in assets]]
    elif crypto_asset is not None:
        if crypto_asset == 'BTC':
            prices = prices.copy().drop(Assets.ETH.value, axis=1)
        else:
            prices = prices.copy().drop(Assets.BTC.value, axis=1)
    return prices


def load_risk_free_rate() -> pd.Series:
    return yf.download('^IRX', start="2003-12-31", end=None, ignore_tz=True, auto_adjust=True)['Close'].dropna() / 100.0


class UnitTests(Enum):
    UPDATE_PRICES_WITH_YF = 1
    CREATE_ETH = 2
    CHECK_PRICES = 3
    CREATE_BALANCED_PRICE = 4
    CHECK_REAL_ESTATE = 5
    UPDATE_PRICES_WITH_BLOOMBERG = 6


def run_unit_test(unit_test: UnitTests):

    if unit_test == UnitTests.UPDATE_PRICES_WITH_YF:
        update_prices_with_yf()
        prices = load_prices()
        print(prices)

    elif unit_test == UnitTests.CREATE_ETH:
        # btc_price = qis.load_df_from_csv(file_name=BTC_PRICES_FROM_2010, local_path=LOCAL_PATH).iloc[:, 0]
        # eth_price = create_eth_price(btc_price=btc_price)
        # print(eth_price)
        from bbg_fetch import fetch_field_timeseries_per_tickers
        bbg_price = fetch_field_timeseries_per_tickers(tickers={'XETUSD Curncy': f"{Assets.ETH.value} - BBG"}).iloc[:, 0]
        mc_price = load_prices(assets=[Assets.ETH], is_updated=False).reindex(index=bbg_price.index)
        prices = pd.concat([bbg_price, mc_price], axis=1).dropna()
        qis.plot_ra_perf_table(prices=prices)
        qis.plot_prices_with_dd(prices=prices, start_to_one=False)

    elif unit_test == UnitTests.CHECK_PRICES:
        prices = load_prices(crypto_asset=None, is_updated=True)
        qis.plot_ra_perf_table(prices=prices)
        qis.plot_prices_with_dd(prices=prices)

    elif unit_test == UnitTests.CREATE_BALANCED_PRICE:
        price = create_balanced_price()
        print(price)
        bal = yf.download('AOR', start="2003-12-31", end=None, ignore_tz=True, auto_adjust=True)['Close'].rename('AOR')
        prices = pd.concat([price, bal], axis=1).dropna()
        qis.plot_prices_with_dd(prices=prices)

    elif unit_test == UnitTests.CHECK_REAL_ESTATE:
        assets = ['IYR', 'REZ', 'REET']
        prices = []
        for asset in assets:
            prices.append(yf.download(asset, start="2003-12-31", end=None, ignore_tz=True, auto_adjust=True)['Close'].rename(asset))
        prices = pd.concat(prices, axis=1).dropna()
        qis.plot_prices_with_dd(prices=prices)

    elif unit_test == UnitTests.UPDATE_PRICES_WITH_BLOOMBERG:
        prices = update_prices_with_bloomberg()
        qis.plot_prices_with_dd(prices=prices)

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.CREATE_ETH

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
