"""
strategic multi-asset allocation on the offline multi-asset universe.

Runs annually-rebalanced rolling optimisation across objectives on the
committed ``multiasset_returns.csv`` fixture (19 instruments, monthly, no
network). Group constraints are built from the fixture's own Asset Class
metadata, so this doubles as a worked example of driving
``GroupLowerUpperConstraints`` from ``group_data``.
"""
# packages
import pandas as pd
import matplotlib.pyplot as plt
import qis as qis
from enum import Enum

# optimalportfolios
from optimalportfolios import (Constraints, GroupLowerUpperConstraints, EwmaCovarEstimator,
                               PortfolioObjective,
                               compute_rolling_optimal_weights,
                               backtest_rolling_optimal_portfolio)
from optimalportfolios.examples.data.multiasset import load_multiasset_data

# annual SAA cadence: 3y EWMA on monthly returns, rebalanced year-end
RETURNS_FREQ = 'ME'
EWMA_SPAN = 36
REBALANCING_FREQ = 'YE'
BACKTEST_START = '31Dec2005'


def _saa_covar_dict(prices: pd.DataFrame):
    estimator = EwmaCovarEstimator(returns_freq=RETURNS_FREQ, span=EWMA_SPAN,
                                   rebalancing_freq=REBALANCING_FREQ)
    time_period = qis.TimePeriod(start=BACKTEST_START, end=prices.index[-1])
    return estimator.fit_rolling_covars(prices=prices, time_period=time_period)


def _group_constraints(group_data: pd.Series) -> GroupLowerUpperConstraints:
    """asset-class band: equities <= 60%, fixed income >= 20%, alternatives <= 40%."""
    group_loadings = qis.set_group_loadings(group_data=group_data)
    group_min = pd.Series(0.0, index=group_loadings.columns)
    group_max = pd.Series(1.0, index=group_loadings.columns)
    group_min['Fixed Income'] = 0.20
    group_max['Equity'] = 0.60
    group_max['Alternatives'] = 0.40
    return GroupLowerUpperConstraints(group_loadings=group_loadings,
                                      group_min_allocation=group_min,
                                      group_max_allocation=group_max)


class LocalTests(Enum):
    ERC_VS_MIN_VARIANCE = 1
    GROUP_CONSTRAINED_ERC = 2
    OBJECTIVE_SWEEP = 3


def run_local_test(local_test: LocalTests):
    data = load_multiasset_data()
    prices = data.prices
    tickers = prices.columns.to_list()
    covar_dict = _saa_covar_dict(prices)

    if local_test == LocalTests.ERC_VS_MIN_VARIANCE:
        constraints = Constraints(is_long_only=True,
                                  max_weights=pd.Series(0.25, index=tickers))
        navs = {}
        for label, objective in [('ERC', PortfolioObjective.EQUAL_RISK_CONTRIBUTION),
                                 ('MinVar', PortfolioObjective.MIN_VARIANCE)]:
            portfolio = backtest_rolling_optimal_portfolio(
                prices=prices, constraints=constraints, covar_dict=covar_dict,
                portfolio_objective=objective, ticker=label)
            navs[label] = portfolio.get_portfolio_nav()
        navs = pd.DataFrame(navs)
        print(navs.tail())
        qis.plot_prices(prices=navs)
        plt.show()

    elif local_test == LocalTests.GROUP_CONSTRAINED_ERC:
        constraints = Constraints(is_long_only=True,
                                  group_lower_upper_constraints=_group_constraints(data.group_data))
        weights = compute_rolling_optimal_weights(
            prices=prices, constraints=constraints, covar_dict=covar_dict,
            portfolio_objective=PortfolioObjective.EQUAL_RISK_CONTRIBUTION,
            risk_budget=pd.Series(1.0 / len(tickers), index=tickers))
        group_weights = weights.T.groupby(data.group_data).sum().T
        print("asset-class weights by rebalance:")
        print(group_weights.round(3))

    elif local_test == LocalTests.OBJECTIVE_SWEEP:
        constraints = Constraints(is_long_only=True,
                                  max_weights=pd.Series(0.25, index=tickers))
        last_date = list(covar_dict.keys())[-1]
        for objective in [PortfolioObjective.MIN_VARIANCE,
                          PortfolioObjective.MAX_DIVERSIFICATION,
                          PortfolioObjective.EQUAL_RISK_CONTRIBUTION]:
            weights = compute_rolling_optimal_weights(
                prices=prices, constraints=constraints, covar_dict=covar_dict,
                portfolio_objective=objective,
                risk_budget=(pd.Series(1.0 / len(tickers), index=tickers)
                             if objective == PortfolioObjective.EQUAL_RISK_CONTRIBUTION else None))
            w_last = weights.loc[last_date]
            print(f"\n{objective.name} weights at {last_date.date()} (top 6):")
            print(w_last.sort_values(ascending=False).head(6).round(4).to_string())


if __name__ == '__main__':
    run_local_test(local_test=LocalTests.OBJECTIVE_SWEEP)
