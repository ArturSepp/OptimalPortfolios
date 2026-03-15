"""
Tests for risk budgeting portfolio optimisation.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from enum import Enum

from optimalportfolios.optimization.constraints import Constraints
from optimalportfolios.optimization.solvers.risk_budgeting import (
    opt_risk_budgeting,
    opt_risk_budgeting_scipy,
    wrapper_risk_budgeting,
    rolling_risk_budgeting,
    compute_portfolio_risk_contributions,
    compute_portfolio_variance,
)
from pyrb import ConstrainedRiskBudgeting


class LocalTests(Enum):
    RISK_PARITY_COMPARE = 1
    RISK_BUDGETING_WITH_BOUNDS = 2
    WRAPPER_RISK_BUDGETING = 3
    ROLLING_RISK_BUDGETING = 4


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes."""

    if local_test == LocalTests.RISK_PARITY_COMPARE:
        # three-asset case with negative correlations: compare scipy vs pyrb vs pyrb-with-bounds
        risk_budget = np.array([0.45, 0.45, 0.1])
        covar = np.array([[0.2 ** 2,       0.5*0.15*0.2, -0.01],
                          [0.5*0.15*0.2,  0.15 ** 2,     -0.005],
                          [-0.01,         -0.005,         0.1**2]])

        vol = np.sqrt(np.diag(covar))
        corr = covar / np.outer(vol, vol)
        print(f"── Input ──")
        print(f"Vols:   {vol}")
        print(f"Corr:\n{np.array2string(corr, precision=3)}")
        print(f"Budget: {risk_budget}")

        constraints = Constraints(is_long_only=True)

        # 1. scipy SLSQP solver
        w_scipy = opt_risk_budgeting_scipy(covar=covar,
                                           constraints=constraints,
                                           risk_budget=risk_budget)
        rc_scipy = compute_portfolio_risk_contributions(w_scipy, covar)
        vol_scipy = np.sqrt(compute_portfolio_variance(w_scipy, covar))

        # 2. pyrb solver (unconstrained beyond long-only)
        w_pyrb = opt_risk_budgeting(covar=covar,
                                    constraints=constraints,
                                    risk_budget=risk_budget)
        rc_pyrb = compute_portfolio_risk_contributions(w_pyrb, covar)
        vol_pyrb = np.sqrt(compute_portfolio_variance(w_pyrb, covar))

        # 3. pyrb solver with explicit box bounds (max 40% per asset)
        constraints_bounded = Constraints(is_long_only=True,
                                          max_weights=pd.Series(0.4, index=['A', 'B', 'C']))
        w_pyrb_bounded = opt_risk_budgeting(covar=covar,
                                            constraints=constraints_bounded,
                                            risk_budget=risk_budget)
        rc_pyrb_bounded = compute_portfolio_risk_contributions(w_pyrb_bounded, covar)
        vol_pyrb_bounded = np.sqrt(compute_portfolio_variance(w_pyrb_bounded, covar))

        # print comparison
        print(f"\n── Scipy SLSQP ──")
        print(f"Weights:    {np.array2string(w_scipy, precision=4)}")
        print(f"RC (norm):  {np.array2string(rc_scipy / np.nansum(rc_scipy), precision=4)}")
        print(f"Port vol:   {vol_scipy:.4%}")

        print(f"\n── pyrb (unconstrained) ──")
        print(f"Weights:    {np.array2string(w_pyrb, precision=4)}")
        print(f"RC (norm):  {np.array2string(rc_pyrb / np.nansum(rc_pyrb), precision=4)}")
        print(f"Port vol:   {vol_pyrb:.4%}")

        print(f"\n── pyrb (max 40% per asset) ──")
        print(f"Weights:    {np.array2string(w_pyrb_bounded, precision=4)}")
        print(f"RC (norm):  {np.array2string(rc_pyrb_bounded / np.nansum(rc_pyrb_bounded), precision=4)}")
        print(f"Port vol:   {vol_pyrb_bounded:.4%}")

        # budget tracking error: |realised RC - target budget|
        print(f"\n── Budget tracking ──")
        print(f"Scipy MAE:        {np.mean(np.abs(rc_scipy / np.nansum(rc_scipy) - risk_budget)):.6f}")
        print(f"pyrb MAE:         {np.mean(np.abs(rc_pyrb / np.nansum(rc_pyrb) - risk_budget)):.6f}")
        print(f"pyrb bounded MAE: {np.mean(np.abs(rc_pyrb_bounded / np.nansum(rc_pyrb_bounded) - risk_budget)):.6f}")

    elif local_test == LocalTests.RISK_BUDGETING_WITH_BOUNDS:
        # four-asset case: equal risk vs tilted budget, with and without weight caps
        n = 4
        vols = np.array([0.20, 0.15, 0.10, 0.25])
        corr = np.array([[1.0, 0.3, 0.1, 0.5],
                          [0.3, 1.0, 0.2, 0.4],
                          [0.1, 0.2, 1.0, 0.1],
                          [0.5, 0.4, 0.1, 1.0]])
        covar = np.outer(vols, vols) * corr
        tickers = ['Equity', 'Bonds', 'Gold', 'HighYield']

        budgets = {
            'Equal (1/N)': np.ones(n) / n,
            'Tilted (50/20/20/10)': np.array([0.50, 0.20, 0.20, 0.10]),
        }
        constraint_sets = {
            'Unconstrained': Constraints(is_long_only=True),
            'Max 40%': Constraints(is_long_only=True,
                                   max_weights=pd.Series(0.4, index=tickers)),
        }

        print(f"── Four-asset risk budgeting ──")
        print(f"Vols:  {dict(zip(tickers, [f'{v:.1%}' for v in vols]))}")
        print(f"Corr:\n{np.array2string(corr, precision=2)}\n")

        for budget_name, budget in budgets.items():
            for constr_name, constraints in constraint_sets.items():
                w = opt_risk_budgeting(covar=covar,
                                       constraints=constraints,
                                       risk_budget=budget)
                rc = compute_portfolio_risk_contributions(w, covar)
                rc_norm = rc / np.nansum(rc)
                port_vol = np.sqrt(compute_portfolio_variance(w, covar))

                print(f"Budget: {budget_name} | Constraints: {constr_name}")
                for i, t in enumerate(tickers):
                    print(f"  {t:12s}  w={w[i]:.4f}  RC={rc_norm[i]:.4f}  target={budget[i]:.4f}")
                print(f"  Portfolio vol: {port_vol:.4%}  |  Budget MAE: {np.mean(np.abs(rc_norm - budget)):.6f}")
                print()

    elif local_test == LocalTests.WRAPPER_RISK_BUDGETING:
        # test wrapper with NaN handling and zero-budget exclusion
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
                                  max_weights=pd.Series(0.5, index=tickers))

        # (a) all assets active
        budget_all = pd.Series({'Equity': 0.4, 'Bonds': 0.3, 'Gold': 0.2, 'HighYield': 0.1})
        w_all = wrapper_risk_budgeting(pd_covar=pd_covar,
                                       constraints=constraints,
                                       risk_budget=budget_all)
        rc_all = compute_portfolio_risk_contributions(w_all.values, covar)

        print(f"── Wrapper: all assets active ──")
        print(f"Weights:\n{w_all.to_string(float_format='{:.4f}'.format)}")
        print(f"RC (norm): {np.array2string(rc_all / np.nansum(rc_all), precision=4)}")
        print(f"Sum: {w_all.sum():.4f}\n")

        # (b) exclude HighYield via zero budget
        budget_no_hy = pd.Series({'Equity': 0.4, 'Bonds': 0.3, 'Gold': 0.3, 'HighYield': 0.0})
        w_no_hy = wrapper_risk_budgeting(pd_covar=pd_covar,
                                          constraints=constraints,
                                          risk_budget=budget_no_hy)

        print(f"── Wrapper: HighYield excluded (budget=0) ──")
        print(f"Weights:\n{w_no_hy.to_string(float_format='{:.4f}'.format)}")
        print(f"HighYield weight: {w_no_hy['HighYield']:.6f} (should be 0)")
        print(f"Sum: {w_no_hy.sum():.4f}\n")

        # (c) NaN in covariance (simulate missing data for Gold)
        pd_covar_nan = pd_covar.copy()
        pd_covar_nan.loc['Gold', :] = np.nan
        pd_covar_nan.loc[:, 'Gold'] = np.nan
        budget_with_nan = pd.Series({'Equity': 0.4, 'Bonds': 0.3, 'Gold': 0.2, 'HighYield': 0.1})
        w_nan = wrapper_risk_budgeting(pd_covar=pd_covar_nan,
                                        constraints=constraints,
                                        risk_budget=budget_with_nan)

        print(f"── Wrapper: Gold has NaN covariance ──")
        print(f"Weights:\n{w_nan.to_string(float_format='{:.4f}'.format)}")
        print(f"Gold weight: {w_nan['Gold']:.6f} (should be 0)")
        print(f"Sum: {w_nan.sum():.4f}\n")

        # (d) detailed output with risk contribution diagnostics
        df_detail = wrapper_risk_budgeting(pd_covar=pd_covar,
                                           constraints=constraints,
                                           risk_budget=budget_all,
                                           detailed_output=True)
        print(f"── Wrapper: detailed output ──")
        print(df_detail.to_string(float_format='{:.4f}'.format))

    elif local_test == LocalTests.ROLLING_RISK_BUDGETING:
        import qis as qis
        from optimalportfolios.test_data import load_test_data
        from optimalportfolios.covar_estimation.ewma_covar_estimator import EwmaCovarEstimator

        prices = load_test_data()
        prices = prices.loc['2000':, :]
        tickers = prices.columns.to_list()

        time_period = qis.TimePeriod(start='31Dec2004', end=prices.index[-1])

        # compute rolling covariances
        ewma_estimator = EwmaCovarEstimator(returns_freq='W-WED', span=52, rebalancing_freq='QE')
        covar_dict = ewma_estimator.fit_rolling_covars(prices=prices, time_period=time_period)

        constraints = Constraints(is_long_only=True,
                                  max_weights=pd.Series(0.5, index=tickers))

        # (a) equal risk budget
        equal_budget = pd.Series({t: 1.0 / len(tickers) for t in tickers})
        w_equal = rolling_risk_budgeting(prices=prices,
                                          constraints=constraints,
                                          risk_budget=equal_budget,
                                          covar_dict=covar_dict)

        # (b) tilted risk budget (60% equity-like, 40% bond-like — approximate)
        tilted_budget = pd.Series({t: 0.6 / max(1, sum(1 for _ in tickers[:len(tickers)//2]))
                                   if i < len(tickers)//2
                                   else 0.4 / max(1, sum(1 for _ in tickers[len(tickers)//2:]))
                                   for i, t in enumerate(tickers)})
        # normalise
        tilted_budget = tilted_budget / tilted_budget.sum()
        w_tilted = rolling_risk_budgeting(prices=prices,
                                           constraints=constraints,
                                           risk_budget=tilted_budget,
                                           covar_dict=covar_dict)

        # compute rolling risk contributions for equal budget
        rc_series = {}
        for date, pd_covar in covar_dict.items():
            if date in w_equal.index:
                w = w_equal.loc[date].reindex(index=pd_covar.columns).fillna(0.0).values
                rc = compute_portfolio_risk_contributions(w, pd_covar.values)
                rc_norm = rc / np.nansum(rc) if np.nansum(rc) > 0 else rc
                rc_series[date] = pd.Series(rc_norm, index=pd_covar.columns)
        rc_df = pd.DataFrame.from_dict(rc_series, orient='index')

        # backtest both
        equal_portfolio = qis.backtest_model_portfolio(prices=prices,
                                                        weights=w_equal,
                                                        ticker='Equal Risk Budget',
                                                        weight_implementation_lag=1,
                                                        rebalancing_costs=0.0010)
        tilted_portfolio = qis.backtest_model_portfolio(prices=prices,
                                                         weights=w_tilted,
                                                         ticker='Tilted Risk Budget',
                                                         weight_implementation_lag=1,
                                                         rebalancing_costs=0.0010)

        print(f"\n── Rolling risk budgeting ──")
        print(f"Equal budget:  {equal_budget.to_dict()}")
        print(f"Tilted budget: {tilted_budget.to_dict()}")

        fig, axs = plt.subplots(3, 1, figsize=(12, 12), tight_layout=True)

        # weights
        qis.plot_time_series(df=w_equal,
                             var_format='{:.0%}',
                             legend_stats=qis.LegendStats.FIRST_AVG_LAST,
                             title='Equal Risk Budget — Weights',
                             ax=axs[0])

        # risk contributions (should track 1/N)
        qis.plot_time_series(df=rc_df,
                             var_format='{:.0%}',
                             legend_stats=qis.LegendStats.FIRST_AVG_LAST,
                             title='Equal Risk Budget — Realised Risk Contributions',
                             ax=axs[1])

        # performance comparison
        navs = pd.concat([equal_portfolio.nav, tilted_portfolio.nav], axis=1)
        qis.plot_time_series(df=navs,
                             var_format='{:.0f}',
                             legend_stats=qis.LegendStats.FIRST_AVG_LAST,
                             title='Risk Budget Portfolios — NAV',
                             ax=axs[2])

    plt.show()


if __name__ == '__main__':
    run_local_test(local_test=LocalTests.RISK_PARITY_COMPARE)