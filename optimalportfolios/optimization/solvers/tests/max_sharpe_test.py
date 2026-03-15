"""
Tests for maximum Sharpe ratio portfolio optimisation.
"""
import numpy as np
import pandas as pd
import qis as qis
from enum import Enum

from optimalportfolios.optimization.solvers.max_sharpe import estimate_rolling_ewma_means, wrapper_maximize_portfolio_sharpe
from optimalportfolios.covar_estimation.ewma_covar_estimator import EwmaCovarEstimator
from optimalportfolios.optimization.constraints import Constraints


class LocalTests(Enum):
    ROLLING_MEANS = 1
    MAXIMIZE_SHARPE = 2


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    import seaborn as sns
    import matplotlib.pyplot as plt
    from optimalportfolios.test_data import load_test_data
    prices = load_test_data()
    prices = prices.loc['2000':, :]

    time_period = qis.TimePeriod(start='31Dec2004', end=prices.index[-1])

    if local_test == LocalTests.ROLLING_MEANS:
        # compute rolling covariances to get rebalancing dates
        ewma_estimator = EwmaCovarEstimator(returns_freq='W-WED', span=52, rebalancing_freq='QE')
        covar_dict = ewma_estimator.fit_rolling_covars(prices=prices, time_period=time_period)
        rebalancing_dates = list(covar_dict.keys())

        # compute expanding EWMA means at rebalancing dates
        means = estimate_rolling_ewma_means(prices=prices,
                                            rebalancing_dates=rebalancing_dates,
                                            returns_freq='W-WED',
                                            span=52,
                                            annualize=True)
        print(means)

        # extract rolling vols from covar_dict for plotting
        vols = {}
        for date, covar in covar_dict.items():
            vols[date] = pd.Series(np.sqrt(np.diag(covar.values)), index=covar.columns)
        vols = pd.DataFrame.from_dict(vols, orient='index')

        with sns.axes_style("darkgrid"):
            fig, axs = plt.subplots(2, 1, figsize=(7, 8))
            qis.plot_time_series(df=means,
                                 var_format='{:.0%}',
                                 trend_line=qis.TrendLine.AVERAGE,
                                 legend_stats=qis.LegendStats.FIRST_AVG_LAST,
                                 title='Expanding EWMA Means (annualised)',
                                 ax=axs[0])
            qis.plot_time_series(df=vols,
                                 var_format='{:.0%}',
                                 trend_line=qis.TrendLine.AVERAGE,
                                 legend_stats=qis.LegendStats.FIRST_AVG_LAST,
                                 title='EWMA Vols (annualised)',
                                 ax=axs[1])

    elif local_test == LocalTests.MAXIMIZE_SHARPE:
        # compute covariance at last date
        ewma_estimator = EwmaCovarEstimator(returns_freq='W-WED', span=52)
        pd_covar = ewma_estimator.fit_current_covar(prices=prices)

        # compute means at last date
        means = estimate_rolling_ewma_means(prices=prices,
                                            rebalancing_dates=[prices.index[-1]],
                                            returns_freq='W-WED',
                                            span=52,
                                            annualize=True)
        means = means.iloc[0]  # Series for the single date

        # solve max Sharpe
        constraints = Constraints(is_long_only=True,
                                  max_weights=pd.Series(0.5, index=prices.columns))

        weights = wrapper_maximize_portfolio_sharpe(pd_covar=pd_covar,
                                                    means=means,
                                                    constraints=constraints,
                                                    solver='ECOS_BB')

        print(f"\nEstimated means:\n{means.to_string(float_format='{:.2%}'.format)}")
        print(f"\nOptimal weights (max Sharpe):\n{weights.to_string(float_format='{:.2%}'.format)}")
        print(f"\nExpected return: {weights @ means:.2%}")
        port_vol = np.sqrt(weights.values @ pd_covar.values @ weights.values)
        print(f"Portfolio vol:   {port_vol:.2%}")
        print(f"Sharpe ratio:    {(weights @ means) / port_vol:.2f}")

    plt.show()


if __name__ == '__main__':
    run_local_test(local_test=LocalTests.MAXIMIZE_SHARPE)
