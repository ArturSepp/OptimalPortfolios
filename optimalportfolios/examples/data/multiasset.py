"""
canonical offline multi-asset universe for examples and tests.

Loads the committed CSV fixture ``multiasset_returns.csv`` (monthly total
returns, decimal) — no network access, unlike the yfinance-based loaders in
this package. The fixture carries two metadata header rows (Asset Class, Sub
Asset Class) so the group structure travels with the data and feeds
``group_data`` / ``GroupLowerUpperConstraints`` directly.

Source: public index data [TODO: add source and attribution].

Universe: 19 instruments across Fixed Income, Equity, Alternatives and
Liquidity (incl. Cash), monthly from 1999-12 to 2026-06. Three series have a
short start-of-history gap; ``load_multiasset_data`` returns the fully
populated sub-panel by default so downstream covariance estimation sees no
NaNs.
"""
# packages
import pandas as pd
from dataclasses import dataclass
from pathlib import Path

FILE_NAME = 'multiasset_returns.csv'
FIRST_METADATA_ROW = 'Asset Class'
SECOND_METADATA_ROW = 'Sub Asset Class'


@dataclass(frozen=True)
class MultiAssetData:
    """immutable snapshot of the multi-asset universe.

    Attributes:
        returns: monthly total returns (decimal), index of month-end dates,
            columns of instrument names.
        prices: total-return price panel normalised to 100 at the first row,
            built from ``returns`` by (1 + r).cumprod().
        group_data: instrument -> Asset Class label (Series).
        sub_group_data: instrument -> Sub Asset Class label (Series).
    """
    returns: pd.DataFrame
    prices: pd.DataFrame
    group_data: pd.Series
    sub_group_data: pd.Series

    def __post_init__(self):
        if not self.returns.columns.equals(self.prices.columns):
            raise ValueError("returns and prices columns are misaligned")
        if not self.group_data.index.equals(self.returns.columns):
            raise ValueError("group_data index does not match the instrument columns")
        if not self.sub_group_data.index.equals(self.returns.columns):
            raise ValueError("sub_group_data index does not match the instrument columns")


def _data_path() -> Path:
    return Path(__file__).parent / FILE_NAME


def load_multiasset_data(drop_incomplete_history: bool = True,  # trim leading NaN rows
                         ) -> MultiAssetData:
    """load the committed multi-asset universe fixture.

    Parameters
    ----------
    drop_incomplete_history : bool
        When True (default) the panel is trimmed to the first fully populated
        month so covariance estimation sees no NaNs; when False the full ragged
        history since 1999-12 is returned with leading NaNs intact.

    Returns
    -------
    MultiAssetData
        returns, prices, group_data, sub_group_data.
    """
    path = _data_path()
    # metadata: two header rows below the instrument-name row (skip the leading
    # provenance comment line)
    meta = pd.read_csv(path, skiprows=1, nrows=2, index_col=0)
    group_data = meta.loc[FIRST_METADATA_ROW]
    group_data.name = None
    group_data.index.name = None
    sub_group_data = meta.loc[SECOND_METADATA_ROW]
    sub_group_data.name = None
    sub_group_data.index.name = None

    returns = pd.read_csv(path, skiprows=[0, 2, 3], index_col=0, parse_dates=True)
    returns = returns.dropna(how='all')
    if drop_incomplete_history:
        # keep the contiguous fully populated span: some series start late and
        # the most recent month can lag for a slow-reporting series
        complete = returns.dropna().index
        returns = returns.loc[complete[0]:complete[-1]]

    prices = 100.0 * (1.0 + returns.fillna(0.0)).cumprod()
    prices = prices.reindex(columns=returns.columns)

    return MultiAssetData(returns=returns,
                          prices=prices,
                          group_data=group_data,
                          sub_group_data=sub_group_data)


def load_multiasset_prices(drop_incomplete_history: bool = True,
                           ) -> pd.DataFrame:
    """convenience: the total-return price panel only."""
    return load_multiasset_data(drop_incomplete_history=drop_incomplete_history).prices
