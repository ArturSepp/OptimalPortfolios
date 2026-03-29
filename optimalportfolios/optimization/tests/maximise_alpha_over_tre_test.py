"""
Tests for alpha-maximising portfolio optimisation with tracking error constraints.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from enum import Enum

from optimalportfolios.optimization.constraints import Constraints, ConstraintEnforcementType
from optimalportfolios.optimization.taa.maximise_alpha_over_tre import (
    wrapper_maximise_alpha_over_tre,
    cvx_maximise_alpha_over_tre,
    cvx_maximise_tre_utility,
)


def print_portfolio(label: str,
                    weights: np.ndarray,
                    covar: np.ndarray,
                    alphas: np.ndarray,
                    benchmark: np.ndarray,
                    tickers: list) -> None:
    """Print portfolio diagnostics relative to benchmark."""
    active = weights - benchmark
    port_vol = np.sqrt(weights @ covar @ weights)
    te = np.sqrt(active @ covar @ active)
    active_alpha = alphas @ active
    ir = active_alpha / te if te > 1e-8 else np.nan

    print(f"\n{'─' * 55}")
    print(f"  {label}")
    print(f"{'─' * 55}")
    for i, t in enumerate(tickers):
        print(f"  {t:12s}  w={weights[i]:7.4f}  bench={benchmark[i]:5.2f}  active={active[i]:+7.4f}")
    print(f"  Portfolio vol:    {port_vol:.4%}")
    print(f"  Tracking error:   {te:.4%}")
    print(f"  Active alpha:     {active_alpha:.4%}")
    print(f"  Information ratio: {ir:.2f}")
    print(f"  Weight sum:       {np.sum(weights):.4f}")


class LocalTests(Enum):
    HARD_CONSTRAINTS = 1
    UTILITY_PENALTIES = 2
    HARD_VS_UTILITY = 3
    TE_FRONTIER = 4
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

    # alpha signal: active return views
    alphas = np.array([0.04, 0.005, 0.02, 0.03])

    # SAA benchmark: 40/30/10/20
    benchmark = np.array([0.40, 0.30, 0.10, 0.20])

    if local_test == LocalTests.HARD_CONSTRAINTS:
        # solve at different TE budgets with hard constraints
        # as TE budget increases, the portfolio tilts further from benchmark
        te_budgets = [0.01, 0.03, 0.05, 0.10]

        print(f"── Inputs ──")
        print(f"Alphas:    {dict(zip(tickers, [f'{a:.2%}' for a in alphas]))}")
        print(f"Benchmark: {dict(zip(tickers, [f'{b:.0%}' for b in benchmark]))}")
        print(f"Vols:      {dict(zip(tickers, [f'{v:.0%}' for v in vols]))}")

        for te_budget in te_budgets:
            constraints = Constraints(
                is_long_only=True,
                max_weights=pd.Series(0.6, index=tickers),
                benchmark_weights=pd.Series(benchmark, index=tickers),
                tracking_err_vol_constraint=te_budget,
            )

            w = cvx_maximise_alpha_over_tre(
                covar=covar,
                alphas=alphas,
                constraints=constraints,
            )

            print_portfolio(
                f"TE budget = {te_budget:.1%}",
                w, covar, alphas, benchmark, tickers
            )

    elif local_test == LocalTests.UTILITY_PENALTIES:
        # solve with utility-based penalties at different lambda_TE
        # higher lambda_TE = more risk aversion = closer to benchmark
        lambda_tes = [0.5, 2.0, 10.0, 50.0]

        print(f"── Utility penalty formulation ──")
        print(f"Alphas:    {dict(zip(tickers, [f'{a:.2%}' for a in alphas]))}")
        print(f"Benchmark: {dict(zip(tickers, [f'{b:.0%}' for b in benchmark]))}")

        for lam in lambda_tes:
            constraints = Constraints(
                is_long_only=True,
                max_weights=pd.Series(0.6, index=tickers),
                benchmark_weights=pd.Series(benchmark, index=tickers),
                constraint_enforcement_type=ConstraintEnforcementType.UTILITY_CONSTRAINTS,
                tre_penalty_weight=lam,
            )

            w = cvx_maximise_tre_utility(
                covar=covar,
                alphas=alphas,
                constraints=constraints,
            )

            print_portfolio(
                f"λ_TE = {lam:.1f}",
                w, covar, alphas, benchmark, tickers
            )

    elif local_test == LocalTests.HARD_VS_UTILITY:
        # compare hard constraint vs utility formulation at matched TE levels
        # first solve with hard TE = 3%, then find utility lambda that matches
        te_budget = 0.03

        # hard constraint
        constraints_hard = Constraints(
            is_long_only=True,
            max_weights=pd.Series(0.6, index=tickers),
            benchmark_weights=pd.Series(benchmark, index=tickers),
            tracking_err_vol_constraint=te_budget,
        )
        w_hard = cvx_maximise_alpha_over_tre(
            covar=covar, alphas=alphas, constraints=constraints_hard
        )
        active_hard = w_hard - benchmark
        te_hard = np.sqrt(active_hard @ covar @ active_hard)

        # utility: sweep lambda to match TE
        best_w_util = None
        best_te_diff = np.inf
        best_lam = None
        for lam in np.arange(0.5, 100.0, 0.5):
            constraints_util = Constraints(
                is_long_only=True,
                max_weights=pd.Series(0.6, index=tickers),
                benchmark_weights=pd.Series(benchmark, index=tickers),
                constraint_enforcement_type=ConstraintEnforcementType.UTILITY_CONSTRAINTS,
                tre_penalty_weight=lam,
            )
            w_util = cvx_maximise_tre_utility(
                covar=covar, alphas=alphas, constraints=constraints_util
            )
            active_util = w_util - benchmark
            te_util = np.sqrt(active_util @ covar @ active_util)
            if abs(te_util - te_hard) < best_te_diff:
                best_te_diff = abs(te_util - te_hard)
                best_w_util = w_util
                best_lam = lam

        print(f"── Hard constraint vs Utility at TE ≈ {te_budget:.1%} ──")
        print_portfolio("Hard constraint", w_hard, covar, alphas, benchmark, tickers)
        print_portfolio(f"Utility (λ={best_lam:.1f}, matched TE)", best_w_util, covar, alphas, benchmark, tickers)

        # weight difference
        diff = w_hard - best_w_util
        print(f"\n  Weight difference (hard - utility):")
        for i, t in enumerate(tickers):
            print(f"    {t:12s}  Δw = {diff[i]:+.4f}")

    elif local_test == LocalTests.TE_FRONTIER:
        # trace the active efficient frontier: alpha vs TE
        te_budgets = np.arange(0.005, 0.12, 0.005)
        results = []

        for te_budget in te_budgets:
            constraints = Constraints(
                is_long_only=True,
                max_weights=pd.Series(0.6, index=tickers),
                benchmark_weights=pd.Series(benchmark, index=tickers),
                tracking_err_vol_constraint=te_budget,
            )
            w = cvx_maximise_alpha_over_tre(
                covar=covar, alphas=alphas, constraints=constraints
            )
            active = w - benchmark
            te_realised = np.sqrt(active @ covar @ active)
            active_alpha = alphas @ active
            ir = active_alpha / te_realised if te_realised > 1e-8 else 0.0

            results.append({
                'te_budget': te_budget,
                'te_realised': te_realised,
                'active_alpha': active_alpha,
                'ir': ir,
                **{t: w[i] for i, t in enumerate(tickers)},
            })

        df = pd.DataFrame(results).set_index('te_budget')
        print(df.to_string(float_format='{:.4f}'.format))

        fig, axs = plt.subplots(3, 1, figsize=(10, 12), tight_layout=True)

        # active alpha vs TE (the active efficient frontier)
        axs[0].plot(df['te_realised'], df['active_alpha'], 'o-')
        axs[0].set_xlabel('Tracking Error')
        axs[0].set_ylabel('Active Alpha')
        axs[0].set_title('Active Efficient Frontier: Alpha vs Tracking Error')
        axs[0].axhline(0, color='grey', linestyle='--', linewidth=0.5)

        # information ratio vs TE budget
        axs[1].plot(df.index, df['ir'], 's-', color='green')
        axs[1].set_xlabel('TE Budget')
        axs[1].set_ylabel('Information Ratio')
        axs[1].set_title('Information Ratio vs TE Budget')

        # weight allocation vs TE budget
        df[tickers].plot.area(ax=axs[2], stacked=True)
        axs[2].axhline(1.0, color='black', linestyle='--', linewidth=0.5)
        axs[2].set_xlabel('TE Budget')
        axs[2].set_ylabel('Weight')
        axs[2].set_title('Weight Allocation vs TE Budget')
        axs[2].legend(loc='upper left')

        # annotate benchmark weights
        for i, t in enumerate(tickers):
            axs[2].axhline(sum(benchmark[:i+1]), color='grey', linestyle=':', linewidth=0.3)

    elif local_test == LocalTests.WRAPPER_WITH_NANS:
        # test the wrapper with NaN handling, zero-alpha, and detailed output
        benchmark_s = pd.Series(benchmark, index=tickers)
        alphas_s = pd.Series(alphas, index=tickers)

        constraints = Constraints(
            is_long_only=True,
            max_weights=pd.Series(0.5, index=tickers),
            tracking_err_vol_constraint=0.03,
        )

        # (a) normal case
        w_normal = wrapper_maximise_alpha_over_tre(
            pd_covar=pd_covar,
            alphas=alphas_s,
            benchmark_weights=benchmark_s,
            constraints=constraints,
        )
        active = w_normal.values - benchmark
        te = np.sqrt(active @ covar @ active)
        print(f"── Wrapper: normal case ──")
        print(f"Weights:\n{w_normal.to_string(float_format='{:.4f}'.format)}")
        print(f"TE: {te:.4%}\n")

        # (b) Gold has NaN covariance
        pd_covar_nan = pd_covar.copy()
        pd_covar_nan.loc['Gold', :] = np.nan
        pd_covar_nan.loc[:, 'Gold'] = np.nan

        w_nan = wrapper_maximise_alpha_over_tre(
            pd_covar=pd_covar_nan,
            alphas=alphas_s,
            benchmark_weights=benchmark_s,
            constraints=constraints,
        )
        print(f"── Wrapper: Gold NaN covariance ──")
        print(f"Weights:\n{w_nan.to_string(float_format='{:.4f}'.format)}")
        print(f"Gold weight: {w_nan['Gold']:.6f} (should be 0)\n")

        # (c) no alpha signal (pure benchmark tracking)
        w_tracking = wrapper_maximise_alpha_over_tre(
            pd_covar=pd_covar,
            alphas=None,
            benchmark_weights=benchmark_s,
            constraints=constraints,
        )
        active_tracking = w_tracking.values - benchmark
        te_tracking = np.sqrt(active_tracking @ covar @ active_tracking)
        print(f"── Wrapper: no alpha (pure tracking) ──")
        print(f"Weights:\n{w_tracking.to_string(float_format='{:.4f}'.format)}")
        print(f"Benchmark:\n{benchmark_s.to_string(float_format='{:.4f}'.format)}")
        print(f"TE: {te_tracking:.4%} (should be ≈ 0)\n")

        # (d) detailed output with risk contributions
        df_detail = wrapper_maximise_alpha_over_tre(
            pd_covar=pd_covar,
            alphas=alphas_s,
            benchmark_weights=benchmark_s,
            constraints=constraints,
            detailed_output=True,
        )
        print(f"── Wrapper: detailed output ──")
        print(df_detail.to_string(float_format='{:.4f}'.format))

    plt.show()


if __name__ == '__main__':
    run_local_test(local_test=LocalTests.TE_FRONTIER)
