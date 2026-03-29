"""
Tests for maximum Sharpe ratio portfolio optimisation.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from enum import Enum

from optimalportfolios.optimization.general.max_sharpe import (
    wrapper_maximize_portfolio_sharpe,
)
from optimalportfolios.optimization.constraints import Constraints


class LocalTests(Enum):
    MAXIMIZE_SHARPE = 1
    SHARPE_WITH_BOUNDS = 2
    WRAPPER_WITH_NANS = 3
    SHARPE_FRONTIER = 4


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes."""

    tickers = ['Equity', 'Bonds', 'Gold', 'HighYield']
    vols = np.array([0.20, 0.05, 0.15, 0.10])
    corr = np.array([[1.0, -0.1, 0.1, 0.6],
                      [-0.1, 1.0, 0.05, 0.2],
                      [0.1, 0.05, 1.0, 0.1],
                      [0.6, 0.2, 0.1, 1.0]])
    covar = np.outer(vols, vols) * corr
    pd_covar = pd.DataFrame(covar, index=tickers, columns=tickers)

    # expected returns (CMAs)
    means = pd.Series({'Equity': 0.06, 'Bonds': 0.02, 'Gold': 0.01, 'HighYield': 0.045})

    if local_test == LocalTests.MAXIMIZE_SHARPE:
        # basic max Sharpe with long-only constraint
        constraints = Constraints(is_long_only=True)

        weights = wrapper_maximize_portfolio_sharpe(pd_covar=pd_covar,
                                                    means=means,
                                                    constraints=constraints)

        port_ret = weights @ means
        port_vol = np.sqrt(weights.values @ covar @ weights.values)
        sharpe = port_ret / port_vol

        print(f"── Max Sharpe (long-only, no caps) ──")
        print(f"Expected returns:\n{means.to_string(float_format='{:.2%}'.format)}")
        print(f"\nOptimal weights:\n{weights.to_string(float_format='{:.4f}'.format)}")
        print(f"\nPortfolio return: {port_ret:.4%}")
        print(f"Portfolio vol:    {port_vol:.4%}")
        print(f"Sharpe ratio:     {sharpe:.2f}")
        print(f"Sum of weights:   {weights.sum():.4f}")

    elif local_test == LocalTests.SHARPE_WITH_BOUNDS:
        # max Sharpe with weight caps — compare uncapped vs capped
        constraints_uncapped = Constraints(is_long_only=True)
        constraints_capped = Constraints(is_long_only=True,
                                          max_weights=pd.Series(0.4, index=tickers))

        w_uncapped = wrapper_maximize_portfolio_sharpe(pd_covar=pd_covar,
                                                       means=means,
                                                       constraints=constraints_uncapped)

        w_capped = wrapper_maximize_portfolio_sharpe(pd_covar=pd_covar,
                                                     means=means,
                                                     constraints=constraints_capped)

        for label, w in [('Uncapped', w_uncapped), ('Max 40%', w_capped)]:
            port_ret = w @ means
            port_vol = np.sqrt(w.values @ covar @ w.values)
            sharpe = port_ret / port_vol
            print(f"\n── {label} ──")
            print(f"Weights:\n{w.to_string(float_format='{:.4f}'.format)}")
            print(f"Return: {port_ret:.4%}  Vol: {port_vol:.4%}  Sharpe: {sharpe:.2f}")

    elif local_test == LocalTests.WRAPPER_WITH_NANS:
        # test NaN handling
        constraints = Constraints(is_long_only=True,
                                  max_weights=pd.Series(0.5, index=tickers))

        # (a) normal case
        w_normal = wrapper_maximize_portfolio_sharpe(pd_covar=pd_covar,
                                                     means=means,
                                                     constraints=constraints)
        port_vol = np.sqrt(w_normal.values @ covar @ w_normal.values)
        sharpe = (w_normal @ means) / port_vol

        print(f"── Wrapper: normal case ──")
        print(f"Weights:\n{w_normal.to_string(float_format='{:.4f}'.format)}")
        print(f"Sharpe: {sharpe:.2f}  Sum: {w_normal.sum():.4f}\n")

        # (b) Gold has NaN covariance
        pd_covar_nan = pd_covar.copy()
        pd_covar_nan.loc['Gold', :] = np.nan
        pd_covar_nan.loc[:, 'Gold'] = np.nan

        w_nan = wrapper_maximize_portfolio_sharpe(pd_covar=pd_covar_nan,
                                                   means=means,
                                                   constraints=constraints)

        print(f"── Wrapper: Gold NaN covariance ──")
        print(f"Weights:\n{w_nan.to_string(float_format='{:.4f}'.format)}")
        print(f"Gold weight: {w_nan['Gold']:.6f} (should be 0)")
        print(f"Sum: {w_nan.sum():.4f}\n")

        # (c) with warm-start from previous weights
        weights_0 = pd.Series({'Equity': 0.3, 'Bonds': 0.4, 'Gold': 0.1, 'HighYield': 0.2})
        w_warm = wrapper_maximize_portfolio_sharpe(pd_covar=pd_covar,
                                                    means=means,
                                                    constraints=constraints,
                                                    weights_0=weights_0)

        print(f"── Wrapper: with warm-start ──")
        print(f"Weights:\n{w_warm.to_string(float_format='{:.4f}'.format)}")
        sharpe_warm = (w_warm @ means) / np.sqrt(w_warm.values @ covar @ w_warm.values)
        print(f"Sharpe: {sharpe_warm:.2f}  Sum: {w_warm.sum():.4f}")

    elif local_test == LocalTests.SHARPE_FRONTIER:
        # vary max weight cap and trace Sharpe ratio vs diversification
        caps = np.arange(0.25, 1.01, 0.05)
        results = []
        for cap in caps:
            constraints = Constraints(is_long_only=True,
                                      max_weights=pd.Series(cap, index=tickers))
            w = wrapper_maximize_portfolio_sharpe(pd_covar=pd_covar,
                                                  means=means,
                                                  constraints=constraints)
            port_ret = w @ means
            port_vol = np.sqrt(w.values @ covar @ w.values)
            sharpe = port_ret / port_vol if port_vol > 1e-8 else 0.0
            results.append({
                'max_weight': cap,
                'return': port_ret,
                'vol': port_vol,
                'sharpe': sharpe,
                **{t: w[t] for t in tickers},
            })

        df = pd.DataFrame(results).set_index('max_weight')
        print(df.to_string(float_format='{:.4f}'.format))

        fig, axs = plt.subplots(2, 1, figsize=(10, 8), tight_layout=True)

        # Sharpe vs max weight cap
        axs[0].plot(df.index, df['sharpe'], 'o-')
        axs[0].set_xlabel('Max Weight Cap')
        axs[0].set_ylabel('Sharpe Ratio')
        axs[0].set_title('Sharpe Ratio vs Weight Cap Constraint')

        # weight allocation vs cap
        df[tickers].plot.area(ax=axs[1], stacked=True)
        axs[1].set_xlabel('Max Weight Cap')
        axs[1].set_ylabel('Weight')
        axs[1].set_title('Weight Allocation vs Weight Cap')
        axs[1].legend(loc='upper left')

    plt.show()


if __name__ == '__main__':
    run_local_test(local_test=LocalTests.MAXIMIZE_SHARPE)
