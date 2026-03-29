"""
Tests for minimum-variance portfolio optimisation with target return constraint.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from enum import Enum

from optimalportfolios.optimization.constraints import Constraints, ConstraintEnforcementType
from optimalportfolios.optimization.saa.min_variance_target_return import (
    wrapper_min_variance_target_return,
)


def print_portfolio(label: str,
                    weights: np.ndarray,
                    covar: np.ndarray,
                    expected_returns: np.ndarray,
                    tickers: list,
                    benchmark: np.ndarray = None) -> None:
    """Print portfolio diagnostics."""
    port_ret = expected_returns @ weights
    port_vol = np.sqrt(weights @ covar @ weights)
    sharpe = port_ret / port_vol if port_vol > 1e-8 else np.nan

    print(f"\n{'─' * 55}")
    print(f"  {label}")
    print(f"{'─' * 55}")
    for i, t in enumerate(tickers):
        line = f"  {t:12s}  w={weights[i]:7.4f}"
        if benchmark is not None:
            line += f"  bench={benchmark[i]:5.2f}  active={weights[i] - benchmark[i]:+7.4f}"
        print(line)
    print(f"  Portfolio return:  {port_ret:.4%}")
    print(f"  Portfolio vol:     {port_vol:.4%}")
    print(f"  Sharpe ratio:      {sharpe:.2f}")
    if benchmark is not None:
        active = weights - benchmark
        te = np.sqrt(active @ covar @ active)
        print(f"  Tracking error:    {te:.4%}")
    print(f"  Weight sum:        {np.sum(weights):.4f}")


class LocalTests(Enum):
    MIN_VAR_TARGET_RETURN = 1
    MIN_VAR_WITH_BENCHMARK = 2
    HARD_VS_UTILITY = 3
    RETURN_FRONTIER = 4
    WRAPPER_WITH_NANS = 5


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
    expected_returns = pd.Series({'Equity': 0.06, 'Bonds': 0.02, 'Gold': 0.01, 'HighYield': 0.045})

    if local_test == LocalTests.MIN_VAR_TARGET_RETURN:
        # solve at different return targets: as target increases, portfolio
        # shifts from low-vol assets toward higher-return (higher-vol) assets
        targets = [0.02, 0.03, 0.04, 0.05]

        constraints = Constraints(is_long_only=True,
                                  max_weights=pd.Series(0.5, index=tickers))

        print(f"── Inputs ──")
        print(f"Expected returns:\n{expected_returns.to_string(float_format='{:.3%}'.format)}")
        print(f"Vols:   {dict(zip(tickers, [f'{v:.1%}' for v in vols]))}")

        for target in targets:
            weights = wrapper_min_variance_target_return(
                pd_covar=pd_covar,
                expected_returns=expected_returns,
                target_return=target,
                constraints=constraints,
            )

            print_portfolio(
                f"Target return: {target:.2%}",
                weights.values, covar, expected_returns.values, tickers
            )

    elif local_test == LocalTests.MIN_VAR_WITH_BENCHMARK:
        # minimise TE variance subject to return floor
        benchmark = np.array([0.40, 0.30, 0.10, 0.20])
        targets = [0.02, 0.035, 0.05]

        constraints = Constraints(is_long_only=True,
                                  max_weights=pd.Series(0.6, index=tickers))

        print(f"── Min TE variance with return floor ──")
        print(f"Benchmark: {dict(zip(tickers, [f'{b:.0%}' for b in benchmark]))}")

        for target in targets:
            weights = wrapper_min_variance_target_return(
                pd_covar=pd_covar,
                expected_returns=expected_returns,
                target_return=target,
                benchmark_weights=pd.Series(benchmark, index=tickers),
                constraints=constraints,
            )

            print_portfolio(
                f"Target return: {target:.2%}",
                weights.values, covar, expected_returns.values, tickers,
                benchmark=benchmark
            )

    elif local_test == LocalTests.HARD_VS_UTILITY:
        # compare hard constraints vs utility (turnover penalty) formulation
        benchmark = np.array([0.40, 0.30, 0.10, 0.20])
        target = 0.035
        weights_0 = pd.Series(benchmark, index=tickers)

        # hard constraints
        constraints_hard = Constraints(
            is_long_only=True,
            max_weights=pd.Series(0.6, index=tickers),
        )
        w_hard = wrapper_min_variance_target_return(
            pd_covar=pd_covar,
            expected_returns=expected_returns,
            target_return=target,
            benchmark_weights=pd.Series(benchmark, index=tickers),
            constraints=constraints_hard,
            weights_0=weights_0,
        )

        # utility with turnover penalty
        constraints_util = Constraints(
            is_long_only=True,
            max_weights=pd.Series(0.6, index=tickers),
            constraint_enforcement_type=ConstraintEnforcementType.UTILITY_CONSTRAINTS,
            turnover_utility_weight=5.0,
        )
        w_util = wrapper_min_variance_target_return(
            pd_covar=pd_covar,
            expected_returns=expected_returns,
            target_return=target,
            benchmark_weights=pd.Series(benchmark, index=tickers),
            constraints=constraints_util,
            weights_0=weights_0,
        )

        print(f"── Hard vs Utility at target return = {target:.2%} ──")
        print_portfolio("Hard constraints", w_hard.values, covar,
                        expected_returns.values, tickers, benchmark)
        print_portfolio("Utility (λ_TO=5.0)", w_util.values, covar,
                        expected_returns.values, tickers, benchmark)

        print(f"\n  Weight difference (hard - utility):")
        for i, t in enumerate(tickers):
            print(f"    {t:12s}  Δw = {w_hard.values[i] - w_util.values[i]:+.4f}")

    elif local_test == LocalTests.RETURN_FRONTIER:
        # trace portfolio vol vs target return (the risk-return tradeoff)
        constraints = Constraints(is_long_only=True,
                                  max_weights=pd.Series(0.5, index=tickers))

        targets = np.arange(0.015, 0.060, 0.005)
        results = []
        for target in targets:
            weights = wrapper_min_variance_target_return(
                pd_covar=pd_covar,
                expected_returns=expected_returns,
                target_return=target,
                constraints=constraints,
            )
            port_ret = weights @ expected_returns
            port_vol = np.sqrt(weights.values @ covar @ weights.values)
            results.append({
                'target_return': target,
                'achieved_return': port_ret,
                'vol': port_vol,
                **{t: weights[t] for t in tickers},
            })

        df = pd.DataFrame(results).set_index('target_return')
        print(df.to_string(float_format='{:.4f}'.format))

        fig, axs = plt.subplots(2, 1, figsize=(10, 8), tight_layout=True)

        # efficient frontier
        axs[0].plot(df['vol'], df['achieved_return'], 'o-')
        axs[0].set_xlabel('Portfolio Vol')
        axs[0].set_ylabel('Achieved Return')
        axs[0].set_title('Min-Variance Efficient Frontier with Return Floor')

        # weight allocation vs target return
        df[tickers].plot.area(ax=axs[1], stacked=True)
        axs[1].set_xlabel('Target Return')
        axs[1].set_ylabel('Weight')
        axs[1].set_title('Weight Allocation vs Target Return')
        axs[1].legend(loc='upper left')

    elif local_test == LocalTests.WRAPPER_WITH_NANS:
        # test NaN handling and edge cases
        constraints = Constraints(is_long_only=True,
                                  max_weights=pd.Series(0.5, index=tickers))

        # (a) normal case
        w_normal = wrapper_min_variance_target_return(
            pd_covar=pd_covar,
            expected_returns=expected_returns,
            target_return=0.03,
            constraints=constraints,
        )
        print(f"── Wrapper: normal case ──")
        print(f"Weights:\n{w_normal.to_string(float_format='{:.4f}'.format)}")
        print(f"Return: {w_normal @ expected_returns:.4%}")
        print(f"Sum: {w_normal.sum():.4f}\n")

        # (b) Gold has NaN covariance
        pd_covar_nan = pd_covar.copy()
        pd_covar_nan.loc['Gold', :] = np.nan
        pd_covar_nan.loc[:, 'Gold'] = np.nan

        w_nan = wrapper_min_variance_target_return(
            pd_covar=pd_covar_nan,
            expected_returns=expected_returns,
            target_return=0.03,
            constraints=constraints,
        )
        print(f"── Wrapper: Gold NaN covariance ──")
        print(f"Weights:\n{w_nan.to_string(float_format='{:.4f}'.format)}")
        print(f"Gold weight: {w_nan['Gold']:.6f} (should be 0)")
        print(f"Sum: {w_nan.sum():.4f}\n")

        # (c) infeasible target return (above max asset return) — should clamp
        w_infeasible = wrapper_min_variance_target_return(
            pd_covar=pd_covar,
            expected_returns=expected_returns,
            target_return=0.10,  # above max expected return of 6%
            constraints=constraints,
        )
        print(f"── Wrapper: infeasible target (10% > max 6%) ──")
        print(f"Weights:\n{w_infeasible.to_string(float_format='{:.4f}'.format)}")
        print(f"Return: {w_infeasible @ expected_returns:.4%}")
        print(f"Sum: {w_infeasible.sum():.4f}\n")

    plt.show()


if __name__ == '__main__':
    run_local_test(local_test=LocalTests.MIN_VAR_TARGET_RETURN)
