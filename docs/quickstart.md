---
icon: material/rocket-launch-outline
---

# Quickstart

This example uses the committed monthly multi-asset fixture, so it runs offline. It
estimates rolling EWMA covariance matrices, constructs a long-only
maximum-diversification portfolio, and backtests it with 10 basis points of rebalancing
cost.

```python
import qis

from optimalportfolios import (
    Constraints,
    EwmaCovarEstimator,
    PortfolioObjective,
    compute_rolling_optimal_weights,
)
from optimalportfolios.examples.data.multiasset import load_multiasset_data

prices = load_multiasset_data().prices.iloc[-120:, :4]
time_period = qis.TimePeriod(prices.index[0], prices.index[-1])

estimator = EwmaCovarEstimator(
    returns_freq="ME", span=24, rebalancing_freq="QE"
)
covar_dict = estimator.fit_rolling_covars(
    prices=prices, time_period=time_period
)
weights = compute_rolling_optimal_weights(
    prices=prices,
    portfolio_objective=PortfolioObjective.MAX_DIVERSIFICATION,
    constraints=Constraints(is_long_only=True),
    time_period=time_period,
    covar_dict=covar_dict,
)

portfolio = qis.backtest_model_portfolio(
    prices=prices.loc[weights.index[0]:],
    weights=weights,
    rebalancing_costs=0.001,
    ticker="MaxDiv",
)
print(portfolio.nav.tail())
```

!!! note "Conventions"

    Weights decided at a rebalancing date are applied to the following holding period.
    Returns in this example are simple monthly returns, and covariance matrices are
    annualised from that frequency.

## What to change first

- **The objective.** Swap `PortfolioObjective.MAX_DIVERSIFICATION` for any other member
  of [`PortfolioObjective`][optimalportfolios.config.PortfolioObjective] — the rest of the
  call is unchanged.
- **The constraints.** [`Constraints`][optimalportfolios.optimization.Constraints] carries
  long-only, leverage, group bounds, turnover and tracking-error limits in one object
  shared by every solver.
- **The covariance estimator.** Replace
  [`EwmaCovarEstimator`][optimalportfolios.covar_estimation.EwmaCovarEstimator] with
  [`FactorCovarEstimator`][optimalportfolios.covar_estimation.FactorCovarEstimator] to use the HCGL
  sparse factor model instead of a sample EWMA matrix.
