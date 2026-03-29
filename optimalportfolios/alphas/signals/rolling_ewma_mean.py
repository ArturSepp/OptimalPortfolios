import pandas as pd
import qis as qis


def estimate_rolling_ewma_means(prices: pd.DataFrame,
                                rebalancing_dates: list,
                                returns_freq: str = 'W-WED',
                                span: int = 52,
                                annualize: bool = True,
                                ) -> pd.DataFrame:
    """
    Compute expanding EWMA means at each rebalancing date.

    For each date in rebalancing_dates, computes the EWMA mean of returns
    using all data up to that date (expanding window).

    Args:
        prices: Asset price panel. Index=dates, columns=tickers.
        rebalancing_dates: List of dates at which to estimate means.
        returns_freq: Frequency for return computation (e.g., 'W-WED', 'ME').
        span: EWMA span in periods at returns_freq.
        annualize: If True, multiply means by annualisation factor.

    Returns:
        DataFrame of estimated means. Index=rebalancing_dates, columns=tickers.
    """
    returns = qis.to_returns(prices=prices, is_log_returns=True, drop_first=True, freq=returns_freq)

    # compute full EWMA mean series, then slice at rebalancing dates
    ewma_means = qis.compute_ewm(returns.to_numpy(), span=span)
    ewma_means = pd.DataFrame(ewma_means, index=returns.index, columns=returns.columns)

    if annualize:
        an_factor = qis.infer_annualisation_factor_from_df(data=returns)
        ewma_means = an_factor * ewma_means

    # select rebalancing dates that exist in returns index
    valid_dates = ewma_means.index.intersection(pd.DatetimeIndex(rebalancing_dates))
    means = ewma_means.loc[valid_dates]

    # for rebalancing dates not exactly on returns index, use ffill lookup
    if len(valid_dates) < len(rebalancing_dates):
        means = ewma_means.reindex(index=pd.DatetimeIndex(rebalancing_dates), method='ffill')

    return means
