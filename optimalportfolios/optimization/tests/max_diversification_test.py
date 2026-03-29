"""
Tests for maximum diversification portfolio optimisation.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from enum import Enum

from optimalportfolios.optimization.constraints import Constraints
from optimalportfolios.optimization.general.max_diversification import (
    wrapper_maximise_diversification,
    rolling_maximise_diversification
)
from optimalportfolios.utils.portfolio_funcs import calculate_diversification_ratio
from optimalportfolios.covar_estimation.ewma_covar_estimator import EwmaCovarEstimator


class LocalTests(Enum):
    MAX_DIVERSIFICATION_SIMPLE = 1
    MAX_DIVERSIFICATION_WITH_BOUNDS = 2
    MAX_DIVERSIFICATION_ROLLING = 3


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes."""

    if local_test == LocalTests.MAX_DIVERSIFICATION_SIMPLE:
        # two-asset case: negative correlation should yield high DR
        covar = np.array([[0.2**2, -0.0075],
                          [-0.0075, 0.1**2]])
        pd_covar = pd.DataFrame(covar, index=['A', 'B'], columns=['A', 'B'])

        constraints = Constraints(is_long_only=True)

        weights = wrapper_maximise_diversification(pd_covar=pd_covar,
                                                   constraints=constraints)

        vols = np.sqrt(np.diag(covar))
        port_vol = np.sqrt(weights.values @ covar @ weights.values)
        dr = calculate_diversification_ratio(w=weights.values, covar=covar)

        print(f"\n── Simple two-asset case ──")
        print(f"Asset vols:           {vols}")
        print(f"Weights:              {weights.to_string(float_format='{:.4f}'.format)}")
        print(f"Portfolio vol:        {port_vol:.4%}")
        print(f"Diversification ratio: {dr:.4f}")
        print(f"Sum of weights:       {weights.sum():.4f}")

    elif local_test == LocalTests.MAX_DIVERSIFICATION_WITH_BOUNDS:
        # four-asset case with max weight constraint
        n = 4
        vols = np.array([0.20, 0.15, 0.10, 0.25])
        corr = np.array([[1.0, 0.3, 0.1, 0.5],
                          [0.3, 1.0, 0.2, 0.4],
                          [0.1, 0.2, 1.0, 0.1],
                          [0.5, 0.4, 0.1, 1.0]])
        covar = np.outer(vols, vols) * corr
        tickers = ['Equity', 'Bonds', 'Gold', 'HighYield']
        pd_covar = pd.DataFrame(covar, index=tickers, columns=tickers)

        constraints = Constraints(is_long_only=True,
                                  max_weights=pd.Series(0.4, index=tickers))

        weights = wrapper_maximise_diversification(pd_covar=pd_covar,
                                                   constraints=constraints)

        port_vol = np.sqrt(weights.values @ covar @ weights.values)
        dr = calculate_diversification_ratio(w=weights.values, covar=covar)

        print(f"\n── Four-asset case with 40% cap ──")
        print(f"Asset vols:           {dict(zip(tickers, [f'{v:.1%}' for v in vols]))}")
        print(f"Weights:\n{weights.to_string(float_format='{:.4f}'.format)}")
        print(f"Portfolio vol:        {port_vol:.4%}")
        print(f"Diversification ratio: {dr:.4f}")
        print(f"Sum of weights:       {weights.sum():.4f}")

        # compare with equal weight
        ew = np.ones(n) / n
        dr_ew = calculate_diversification_ratio(w=ew, covar=covar)
        print(f"\nEqual-weight DR:      {dr_ew:.4f}")
        print(f"MDP improvement:      {dr / dr_ew - 1:.2%}")

    elif local_test == LocalTests.MAX_DIVERSIFICATION_ROLLING:
        import qis as qis
        from optimalportfolios.test_data import load_test_data
        prices = load_test_data()
        prices = prices.loc['2000':, :]

        time_period = qis.TimePeriod(start='31Dec2004', end=prices.index[-1])

        # compute rolling covariances
        ewma_estimator = EwmaCovarEstimator(returns_freq='W-WED', span=52, rebalancing_freq='QE')
        covar_dict = ewma_estimator.fit_rolling_covars(prices=prices, time_period=time_period)

        # solve rolling max diversification
        constraints = Constraints(is_long_only=True,
                                  max_weights=pd.Series(0.5, index=prices.columns))

        rolling_weights = rolling_maximise_diversification(prices=prices,
                                                           constraints=constraints,
                                                           covar_dict=covar_dict)

        # compute rolling diversification ratios
        dr_series = {}
        for date, pd_covar in covar_dict.items():
            if date in rolling_weights.index:
                w = rolling_weights.loc[date].values
                dr_series[date] = calculate_diversification_ratio(w=w, covar=pd_covar.values)
        dr_series = pd.Series(dr_series, name='Diversification Ratio')

        print(f"\n── Rolling max diversification ──")
        print(f"Mean DR:    {dr_series.mean():.3f}")
        print(f"Min DR:     {dr_series.min():.3f}")
        print(f"Max DR:     {dr_series.max():.3f}")

        fig, axs = plt.subplots(2, 1, figsize=(12, 8), tight_layout=True)
        qis.plot_time_series(df=rolling_weights,
                             var_format='{:.0%}',
                             legend_stats=qis.LegendStats.FIRST_AVG_LAST,
                             title='Max Diversification Weights',
                             ax=axs[0])
        qis.plot_time_series(df=dr_series.to_frame(),
                             var_format='{:.2f}',
                             legend_stats=qis.LegendStats.FIRST_AVG_LAST,
                             title='Diversification Ratio',
                             ax=axs[1])

    plt.show()


if __name__ == '__main__':

    run_local_test(local_test=LocalTests.MAX_DIVERSIFICATION_WITH_BOUNDS)
