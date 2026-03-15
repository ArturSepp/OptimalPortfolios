"""
Tests for alpha-maximising portfolio optimisation with target return constraint.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from enum import Enum

from optimalportfolios.optimization.constraints import Constraints
from optimalportfolios.optimization.solvers.target_return import  wrapper_maximise_alpha_with_target_return


class LocalTests(Enum):
    ALPHA_TARGET_RETURN = 1
    ALPHA_FRONTIER = 2


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

    # alpha signal: CMA-derived excess return views
    alphas = pd.Series({'Equity': 0.04, 'Bonds': 0.005, 'Gold': 0.02, 'HighYield': 0.03})

    # yields: carry / income component
    yields = pd.Series({'Equity': 0.02, 'Bonds': 0.035, 'Gold': 0.00, 'HighYield': 0.055})

    if local_test == LocalTests.ALPHA_TARGET_RETURN:
        # solve at three different return targets to see how
        # the constraint shifts the portfolio from alpha-optimal
        # toward yield-heavy assets
        targets = [0.01, 0.025, 0.04]

        constraints = Constraints(is_long_only=True,
                                  max_weights=pd.Series(0.5, index=tickers))

        print(f"── Inputs ──")
        print(f"Alphas:\n{alphas.to_string(float_format='{:.3%}'.format)}")
        print(f"\nYields:\n{yields.to_string(float_format='{:.3%}'.format)}")
        print(f"\nVols:   {dict(zip(tickers, [f'{v:.1%}' for v in vols]))}")

        for target in targets:
            weights = wrapper_maximise_alpha_with_target_return(
                pd_covar=pd_covar,
                alphas=alphas,
                yields=yields,
                target_return=target,
                constraints=constraints,
            )

            port_alpha = weights @ alphas
            port_yield = weights @ yields
            port_vol = np.sqrt(weights.values @ covar @ weights.values)

            print(f"\n── Target return: {target:.2%} ──")
            print(f"Weights:\n{weights.to_string(float_format='{:.4f}'.format)}")
            print(f"Portfolio alpha:  {port_alpha:.4%}")
            print(f"Portfolio yield:  {port_yield:.4%}  (target: {target:.4%})")
            print(f"Portfolio vol:    {port_vol:.4%}")
            print(f"Sum:              {weights.sum():.4f}")

    elif local_test == LocalTests.ALPHA_FRONTIER:
        # trace out the efficient frontier: for each target return level,
        # solve for max alpha and record the resulting portfolio metrics
        constraints = Constraints(is_long_only=True,
                                  max_weights=pd.Series(0.5, index=tickers))

        targets = np.arange(0.005, 0.055, 0.005)
        results = []
        for target in targets:
            weights = wrapper_maximise_alpha_with_target_return(
                pd_covar=pd_covar,
                alphas=alphas,
                yields=yields,
                target_return=target,
                constraints=constraints,
            )
            port_alpha = weights @ alphas
            port_yield = weights @ yields
            port_vol = np.sqrt(weights.values @ covar @ weights.values)
            results.append({
                'target_return': target,
                'alpha': port_alpha,
                'yield': port_yield,
                'vol': port_vol,
                **{t: weights[t] for t in tickers},
            })

        df = pd.DataFrame(results).set_index('target_return')
        print(df.to_string(float_format='{:.4f}'.format))

        fig, axs = plt.subplots(2, 1, figsize=(10, 8), tight_layout=True)

        # alpha vs target return
        axs[0].plot(df.index, df['alpha'], 'o-', label='Portfolio alpha')
        axs[0].set_xlabel('Target return')
        axs[0].set_ylabel('Alpha')
        axs[0].set_title('Alpha vs Target Return Constraint')
        ax_vol = axs[0].twinx()
        ax_vol.plot(df.index, df['vol'], 's--', color='red', label='Portfolio vol')
        ax_vol.set_ylabel('Vol')
        axs[0].legend(loc='upper left')
        ax_vol.legend(loc='upper right')

        # weight allocation vs target return
        df[tickers].plot.area(ax=axs[1], stacked=True)
        axs[1].set_xlabel('Target return')
        axs[1].set_ylabel('Weight')
        axs[1].set_title('Weight Allocation vs Target Return')
        axs[1].legend(loc='upper left')

    plt.show()


if __name__ == '__main__':
    run_local_test(local_test=LocalTests.ALPHA_TARGET_RETURN)
