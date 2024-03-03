"""
    implementation of [PortfolioObjective.MAX_MIXTURE_CARA]

compute rolling time series of weight for optimal portfolios
using mixure carra see
 Sepp, Artur, Optimal Allocation to Cryptocurrencies in Diversified Portfolios
 Available at SSRN: https://ssrn.com/abstract=4217841
"""
# packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Union, Dict
from enum import Enum
import qis
from qis import TimePeriod, PortfolioData

# portfolio
from optimalportfolios.optimization.config import set_min_max_weights, set_to_zero_not_investable_weights
import optimalportfolios.utils.gaussian_mixture as gm
import optimalportfolios.optimization.solvers.nonlinear as ops


def compute_rolling_weights_mixture_carra(prices: pd.DataFrame,
                                          min_weights: Dict[str, float] = None,
                                          max_weights: Dict[str, float] = None,
                                          fixed_weights: Dict[str, float] = None,
                                          is_long_only: bool = True,
                                          rebalancing_freq: str = 'QE',
                                          roll_window: int = 20,  # number of periods in mixure estimation
                                          returns_freq: str = 'W-WED',  # frequency for returns computing mixure distr
                                          is_log_returns: bool = True,
                                          carra: float = 0.5,  # carra parameters
                                          n_components: int = 3
                                          ) -> pd.DataFrame:
    """
    solve rolling mixture Carra portfolios
    estimation is applied for the whole period of prices
    """
    rets = qis.to_returns(prices=prices, is_log_returns=is_log_returns, drop_first=True, freq=returns_freq)
    # nb mixture cannot handle nans
    # simple solution is to use zero returns for nans and set max weights = 0 for assets with zero vol
    rets = rets.fillna(value=0.0)
    dates_schedule = qis.generate_dates_schedule(time_period=qis.get_time_period(df=rets),
                                                 freq=rebalancing_freq,
                                                 include_start_date=True,
                                                 include_end_date=False)
    _, scaler = qis.get_period_days(freq=returns_freq)

    # set weights
    min_weights0, max_weights0 = set_min_max_weights(assets=list(prices.columns),
                                                     min_weights=min_weights,
                                                     max_weights=max_weights,
                                                     fixed_weights=fixed_weights,
                                                     is_long_only=is_long_only)

    weights = {}
    for idx, end in enumerate(dates_schedule[1:]):
        if idx >= roll_window-1:
            period = qis.TimePeriod(dates_schedule[idx - roll_window+1], end)
            # period.print()
            rets_ = period.locate(rets).to_numpy()

            # nb mixture cannot handle nans
            # simple solution is to use zero returns for nans and set max weights = 0 for assets with zero vol
            min_weights1, max_weights1 = set_to_zero_not_investable_weights(min_weights=min_weights0,
                                                                            max_weights=max_weights0,
                                                                            covar=np.diag(np.var(rets_, axis=0)))

            params = gm.fit_gaussian_mixture(x=rets_, n_components=n_components, scaler=scaler)
            # print(params)
            weights[end] = ops.solve_cara_mixture(means=params.means,
                                                  covars=params.covars,
                                                  probs=params.probs,
                                                  carra=carra,
                                                  min_weights=min_weights1.to_numpy(),
                                                  max_weights=max_weights1.to_numpy())

    weights = pd.DataFrame.from_dict(weights, orient='index', columns=prices.columns)

    return weights


def backtest_rolling_mixure_portfolio(prices: pd.DataFrame,
                                      time_period: TimePeriod = None,
                                      rebalancing_freq: str = 'QE',
                                      roll_window: int = 20,  # number of periods in mixure estimation
                                      returns_freq: str = 'W-WED',  # frequency for returns computing mixure distr
                                      is_log_returns: bool = True,
                                      carra: float = 0.5,  # carra parameters
                                      n_components: int = 3,
                                      min_weights: np.ndarray = None,
                                      max_weights: np.ndarray = None,
                                      ticker: str = None,
                                      rebalancing_costs: float = 0.0010  # 10 bp
                                      ) -> PortfolioData:
    """
    wrapper to get portfolio data with mixure weights
    """
    weights = compute_rolling_weights_mixture_carra(prices=prices,
                                                    rebalancing_freq=rebalancing_freq,
                                                    roll_window=roll_window,
                                                    returns_freq=returns_freq,
                                                    is_log_returns=is_log_returns,
                                                    carra=carra,
                                                    n_components=n_components,
                                                    min_weights=min_weights,
                                                    max_weights=max_weights)
    if time_period is not None:
        weights = time_period.locate(weights)

    portfolio_out = qis.backtest_model_portfolio(prices=prices,
                                                 weights=weights,
                                                 is_rebalanced_at_first_date=True,
                                                 ticker=ticker,
                                                 is_output_portfolio_data=True,
                                                 rebalancing_costs=rebalancing_costs)

    return portfolio_out


def estimate_rolling_mixture(prices: Union[pd.Series, pd.DataFrame],
                             returns_freq: str = 'W-WED',
                             rebalancing_freq: str = 'QE',
                             roll_window: int = 20,
                             n_components: int = 3,
                             is_log_returns: bool = True,
                             annualize: bool = True
                             ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    if isinstance(prices, pd.Series):
        prices = prices.to_frame()
    elif isinstance(prices, pd.DataFrame) and len(prices.columns) > 1:
        raise ValueError(f"supported only 1-d price time series")

    rets = qis.to_returns(prices=prices, is_log_returns=is_log_returns, drop_first=True, freq=returns_freq)

    dates_schedule = qis.generate_dates_schedule(time_period=qis.get_time_period(df=rets),
                                                 freq=rebalancing_freq,
                                                 include_start_date=True,
                                                 include_end_date=True)
    if annualize:
        _, scaler = qis.get_period_days(freq=returns_freq)
    else:
        scaler = 1.0

    means, sigmas, probs = [], [], []
    for idx, end in enumerate(dates_schedule[1:]):
        if idx >= roll_window-1:
            period = qis.TimePeriod(dates_schedule[idx - roll_window+1], end)
            # period.print()
            rets_ = period.locate(rets).to_numpy()
            params = gm.fit_gaussian_mixture(x=rets_, n_components=n_components, scaler=scaler)
            mean = np.stack(params.means, axis=0).T[0]
            std = np.sqrt(np.array([params.covars[0][0], params.covars[1][0]]))
            prob = params.probs
            ranks = mean.argsort().argsort()
            means.append(pd.DataFrame(mean[ranks].reshape(1, -1), index=[end]))
            sigmas.append(pd.DataFrame(std[ranks].reshape(1, -1), index=[end]))
            probs.append(pd.DataFrame(prob[ranks].reshape(1, -1), index=[end]))

    means = pd.concat(means)
    sigmas = pd.concat(sigmas)
    probs = pd.concat(probs)

    return means, sigmas, probs


class UnitTests(Enum):
    ROLLING_MIXTURES = 1
    MIXTURE_PORTFOLIOS = 2


def run_unit_test(unit_test: UnitTests):

    # data
    from optimalportfolios.test_data import load_test_data
    prices = load_test_data()
    prices = prices.loc['2000':, :]  # have at least 3 assets

    if unit_test == UnitTests.ROLLING_MIXTURES:
        prices = prices['SPY'].dropna()
        means, sigmas, probs = estimate_rolling_mixture(prices=prices)
        print(means)

    elif unit_test == UnitTests.MIXTURE_PORTFOLIOS:
        #prices = prices.dropna()
        weights = compute_rolling_weights_mixture_carra(prices=prices,
                                                        rebalancing_freq='QE',
                                                        n_components=3,
                                                        roll_window=20,
                                                        carra=0.5)
        with sns.axes_style("darkgrid"):
            fig, ax = plt.subplots(1, 1, figsize=(7, 12))
            qis.plot_time_series(df=weights,
                                 var_format='{:.0%}',
                                 legend_stats=qis.LegendStats.FIRST_AVG_LAST,
                                 ax=ax)
    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.MIXTURE_PORTFOLIOS

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
