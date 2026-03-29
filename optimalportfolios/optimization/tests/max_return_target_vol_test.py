"""
Tests for return-maximising portfolio optimisation with volatility constraint.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from enum import Enum

from optimalportfolios.optimization.constraints import Constraints, ConstraintEnforcementType
from optimalportfolios.optimization.saa.max_return_target_vol import (
    wrapper_max_return_target_vol,
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
        active_ret = expected_returns @ active
        ir = active_ret / te if te > 1e-8 else np.nan
        print(f"  Tracking error:    {te:.4%}")
        print(f"  Active return:     {active_ret:.4%}")
        print(f"  Information ratio: {ir:.2f}")
    print(f"  Weight sum:        {np.sum(weights):.4f}")


class LocalTests(Enum):
    MAX_RETURN_ABS_VOL = 1
    MAX_RETURN_TE_BUDGET = 2
    HARD_VS_UTILITY = 3
    VOL_FRONTIER = 4
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

    if local_test == LocalTests.MAX_RETURN_ABS_VOL:
        # maximise return subject to absolute vol budget (no benchmark)
        # as vol budget increases, portfolio tilts toward higher-return assets
        vol_budgets = [0.05, 0.08, 0.10, 0.15]

        constraints = Constraints(is_long_only=True,
                                  max_weights=pd.Series(0.5, index=tickers))

        print(f"── Inputs ──")
        print(f"Expected returns:\n{expected_returns.to_string(float_format='{:.3%}'.format)}")
        print(f"Vols:   {dict(zip(tickers, [f'{v:.1%}' for v in vols]))}")

        for vol_budget in vol_budgets:
            weights = wrapper_max_return_target_vol(
                pd_covar=pd_covar,
                expected_returns=expected_returns,
                target_vol=vol_budget,
                constraints=constraints,
            )

            print_portfolio(
                f"Vol budget: {vol_budget:.1%}",
                weights.values, covar, expected_returns.values, tickers
            )

    elif local_test == LocalTests.MAX_RETURN_TE_BUDGET:
        # maximise active return subject to tracking error budget
        benchmark = np.array([0.40, 0.30, 0.10, 0.20])
        te_budgets = [0.01, 0.03, 0.05, 0.10]

        constraints = Constraints(is_long_only=True,
                                  max_weights=pd.Series(0.6, index=tickers))

        print(f"── Max active return with TE budget ──")
        print(f"Benchmark: {dict(zip(tickers, [f'{b:.0%}' for b in benchmark]))}")

        for te_budget in te_budgets:
            weights = wrapper_max_return_target_vol(
                pd_covar=pd_covar,
                expected_returns=expected_returns,
                target_vol=te_budget,
                benchmark_weights=pd.Series(benchmark, index=tickers),
                constraints=constraints,
            )

            print_portfolio(
                f"TE budget: {te_budget:.1%}",
                weights.values, covar, expected_returns.values, tickers,
                benchmark=benchmark
            )

    elif local_test == LocalTests.HARD_VS_UTILITY:
        # compare hard vol constraint vs utility penalty at matched vol
        vol_budget = 0.08

        # hard constraint (absolute vol)
        constraints_hard = Constraints(
            is_long_only=True,
            max_weights=pd.Series(0.5, index=tickers),
        )
        w_hard = wrapper_max_return_target_vol(
            pd_covar=pd_covar,
            expected_returns=expected_returns,
            target_vol=vol_budget,
            constraints=constraints_hard,
        )
        vol_hard = np.sqrt(w_hard.values @ covar @ w_hard.values)

        # utility: sweep lambda to match vol
        best_w_util = None
        best_vol_diff = np.inf
        best_lam = None
        for lam in np.arange(0.5, 50.0, 0.5):
            constraints_util = Constraints(
                is_long_only=True,
                max_weights=pd.Series(0.5, index=tickers),
                constraint_enforcement_type=ConstraintEnforcementType.UTILITY_CONSTRAINTS,
                tre_utility_weight=lam,
            )
            w_util = wrapper_max_return_target_vol(
                pd_covar=pd_covar,
                expected_returns=expected_returns,
                target_vol=vol_budget,
                constraints=constraints_util,
            )
            vol_util = np.sqrt(w_util.values @ covar @ w_util.values)
            if abs(vol_util - vol_hard) < best_vol_diff:
                best_vol_diff = abs(vol_util - vol_hard)
                best_w_util = w_util
                best_lam = lam

        print(f"── Hard constraint vs Utility at vol ≈ {vol_budget:.1%} ──")
        print_portfolio("Hard constraint", w_hard.values, covar,
                        expected_returns.values, tickers)
        print_portfolio(f"Utility (λ_vol={best_lam:.1f}, matched vol)",
                        best_w_util.values, covar, expected_returns.values, tickers)

        print(f"\n  Weight difference (hard - utility):")
        for i, t in enumerate(tickers):
            print(f"    {t:12s}  Δw = {w_hard.values[i] - best_w_util.values[i]:+.4f}")

    elif local_test == LocalTests.VOL_FRONTIER:
        # trace return vs vol budget (efficient frontier via vol budgeting)
        constraints = Constraints(is_long_only=True,
                                  max_weights=pd.Series(0.5, index=tickers))

        vol_budgets = np.arange(0.03, 0.16, 0.01)
        results = []
        for vol_budget in vol_budgets:
            weights = wrapper_max_return_target_vol(
                pd_covar=pd_covar,
                expected_returns=expected_returns,
                target_vol=vol_budget,
                constraints=constraints,
            )
            port_ret = weights @ expected_returns
            port_vol = np.sqrt(weights.values @ covar @ weights.values)
            results.append({
                'vol_budget': vol_budget,
                'vol_realised': port_vol,
                'return': port_ret,
                **{t: weights[t] for t in tickers},
            })

        df = pd.DataFrame(results).set_index('vol_budget')
        print(df.to_string(float_format='{:.4f}'.format))

        fig, axs = plt.subplots(2, 1, figsize=(10, 8), tight_layout=True)

        # return vs realised vol
        axs[0].plot(df['vol_realised'], df['return'], 'o-')
        axs[0].set_xlabel('Realised Vol')
        axs[0].set_ylabel('Expected Return')
        axs[0].set_title('Max Return Efficient Frontier (Vol Budget)')

        # weight allocation vs vol budget
        df[tickers].plot.area(ax=axs[1], stacked=True)
        axs[1].set_xlabel('Vol Budget')
        axs[1].set_ylabel('Weight')
        axs[1].set_title('Weight Allocation vs Vol Budget')
        axs[1].legend(loc='upper left')

    elif local_test == LocalTests.WRAPPER_WITH_NANS:
        # test NaN handling and edge cases
        constraints = Constraints(is_long_only=True,
                                  max_weights=pd.Series(0.5, index=tickers))

        # (a) normal case — absolute vol
        w_normal = wrapper_max_return_target_vol(
            pd_covar=pd_covar,
            expected_returns=expected_returns,
            target_vol=0.08,
            constraints=constraints,
        )
        port_vol = np.sqrt(w_normal.values @ covar @ w_normal.values)
        print(f"── Wrapper: normal case (vol budget = 8%) ──")
        print(f"Weights:\n{w_normal.to_string(float_format='{:.4f}'.format)}")
        print(f"Return:  {w_normal @ expected_returns:.4%}")
        print(f"Vol:     {port_vol:.4%}  (budget: 8.00%)")
        print(f"Sum:     {w_normal.sum():.4f}\n")

        # (b) Gold has NaN covariance
        pd_covar_nan = pd_covar.copy()
        pd_covar_nan.loc['Gold', :] = np.nan
        pd_covar_nan.loc[:, 'Gold'] = np.nan

        w_nan = wrapper_max_return_target_vol(
            pd_covar=pd_covar_nan,
            expected_returns=expected_returns,
            target_vol=0.08,
            constraints=constraints,
        )
        print(f"── Wrapper: Gold NaN covariance ──")
        print(f"Weights:\n{w_nan.to_string(float_format='{:.4f}'.format)}")
        print(f"Gold weight: {w_nan['Gold']:.6f} (should be 0)")
        print(f"Sum:         {w_nan.sum():.4f}\n")

        # (c) with benchmark (TE mode)
        benchmark = pd.Series({'Equity': 0.40, 'Bonds': 0.30, 'Gold': 0.10, 'HighYield': 0.20})
        w_te = wrapper_max_return_target_vol(
            pd_covar=pd_covar,
            expected_returns=expected_returns,
            target_vol=0.03,
            benchmark_weights=benchmark,
            constraints=constraints,
        )
        active = w_te.values - benchmark.values
        te = np.sqrt(active @ covar @ active)
        print(f"── Wrapper: TE mode (budget = 3%) ──")
        print(f"Weights:\n{w_te.to_string(float_format='{:.4f}'.format)}")
        print(f"TE:     {te:.4%}  (budget: 3.00%)")
        print(f"Active return: {expected_returns.values @ active:.4%}")
        print(f"Sum:    {w_te.sum():.4f}\n")

        # (d) very tight vol budget — should produce conservative allocation
        w_tight = wrapper_max_return_target_vol(
            pd_covar=pd_covar,
            expected_returns=expected_returns,
            target_vol=0.03,
            constraints=constraints,
        )
        port_vol_tight = np.sqrt(w_tight.values @ covar @ w_tight.values)
        print(f"── Wrapper: tight vol budget (3%) ──")
        print(f"Weights:\n{w_tight.to_string(float_format='{:.4f}'.format)}")
        print(f"Vol:     {port_vol_tight:.4%}  (budget: 3.00%)")
        print(f"Return:  {w_tight @ expected_returns:.4%}")
        print(f"Sum:     {w_tight.sum():.4f}\n")

    plt.show()


if __name__ == '__main__':
    run_local_test(local_test=LocalTests.MAX_RETURN_ABS_VOL)
