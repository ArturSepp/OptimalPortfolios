"""Run a fully offline constrained rolling portfolio backtest.

The example uses the monthly multi-asset fixture shipped in the wheel. It
needs only a plain ``pip install optimalportfolios`` and writes no files.
"""

from time import perf_counter

import pandas as pd
import qis

from optimalportfolios import (
    Constraints,
    EwmaCovarEstimator,
    PortfolioObjective,
    compute_rolling_optimal_weights,
)
from optimalportfolios.tests.data.multiasset import load_multiasset_data


ASSETS = [
    "Global Bonds",
    "US Treasuries",
    "Global HY Bonds",
    "MSCI World USD",
    "Hedge Funds",
    "Commodities EX-Precious",
]
PRICE_START = "2010-01-31"
FIRST_REBALANCE = "2015-03-31"
LAST_REBALANCE = "2022-09-30"
PRICE_END = "2022-12-31"


def main() -> None:
    """Estimate rolling weights, backtest them, and print compact evidence."""
    started = perf_counter()
    prices = load_multiasset_data().prices.loc[PRICE_START:PRICE_END, ASSETS]

    # Each quarterly covariance uses information available at that date. The
    # earlier price history provides a warm-up for the 24-month EWMA estimator.
    estimation_period = qis.TimePeriod(FIRST_REBALANCE, LAST_REBALANCE)
    estimator = EwmaCovarEstimator(returns_freq="ME", span=24)
    covar_dict = estimator.fit_rolling_covars(
        prices=prices,
        time_period=estimation_period,
        rebalancing_freq="QE",
    )

    constraints = Constraints(
        is_long_only=True,
        max_weights=pd.Series(0.35, index=prices.columns),
    )
    weights = compute_rolling_optimal_weights(
        prices=prices,
        constraints=constraints,
        covar_dict=covar_dict,
        portfolio_objective=PortfolioObjective.MIN_VARIANCE,
    )

    # A weight decided at month-end t starts trading at the next observation.
    portfolio = qis.backtest_model_portfolio(
        prices=prices.loc[weights.index[0]:],
        weights=weights,
        rebalancing_costs=0.001,
        weight_implementation_lag=1,
        ticker="MinVar",
    )
    nav = portfolio.get_portfolio_nav()

    print(f"Price history: {prices.index[0]:%Y-%m-%d} to {prices.index[-1]:%Y-%m-%d}")
    print(f"Rolling weights: {weights.shape[0]} dates x {weights.shape[1]} assets")
    print(f"Last rebalance: {weights.index[-1]:%Y-%m-%d}")
    print(weights.iloc[-1].round(4).to_string())
    print(f"Final NAV after 10 bp transaction costs: {float(nav.iloc[-1]):.4f}")
    print(f"Runtime: {perf_counter() - started:.2f} seconds")


if __name__ == "__main__":
    main()
