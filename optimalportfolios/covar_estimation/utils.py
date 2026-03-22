from __future__ import annotations

from typing import Optional

import pandas as pd
import qis as qis


def compute_returns_from_prices(prices: pd.DataFrame,
                                returns_freq: Optional[str] = 'ME',
                                demean: bool = True,
                                drop_first: bool = True,
                                is_first_zero: bool = False,
                                is_log_returns: bool = True,
                                span: Optional[int] = 52
                                ) -> pd.DataFrame:
    """
    compute returns for covar matrix estimation
    """
    returns = qis.to_returns(prices=prices,
                             is_log_returns=is_log_returns,
                             is_first_zero=is_first_zero,
                             drop_first=drop_first,
                             freq=returns_freq)
    if demean:
        returns = returns - qis.compute_ewm(returns, span=span)
        # returns.iloc[0, :] will be zero so shift the period
        if drop_first:
            returns = returns.iloc[1:, :]
    return returns
