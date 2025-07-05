"""
illustration of solving for risk budgets using input weights
create 60/40 portfolio with static weights
find equivalent risk budget portfolio with weights matching in average weights of 60/40 portfolio
show weights/risk contributions for both
"""

import pandas as pd
import qis as qis
import matplotlib.pyplot as plt
import yfinance as yf
from typing import Dict, List
from enum import Enum
from optimalportfolios import estimate_rolling_ewma_covar, rolling_risk_budgeting, Constraints
from optimalportfolios.optimization.solvers.risk_budgeting import solve_for_risk_budgets_from_given_weights


def plot_static_risk_budgets_vs_weights(prices: pd.DataFrame,
                                        risk_budgets_weights: pd.DataFrame,
                                        given_static_weights: pd.Series,
                                        covar_dict: Dict[pd.Timestamp, pd.DataFrame],
                                        time_period: qis.TimePeriod = None,
                                        figsize=(16, 10),
                                        add_titles: bool = True,
                                        var_format: str = '{:.1%}',
                                        strategy_ticker: str = 'Risk-budgeted portfolio',
                                        benchmark_ticker: str = 'Static portfolio'
                                        ) -> List[plt.Figure]:
    # create static_weights on same
    static_weights = pd.DataFrame.from_dict({date: given_static_weights for date in risk_budgets_weights.index}, orient='index')

    static_portfolio = qis.backtest_model_portfolio(prices=prices, weights=static_weights,
                                                    ticker=benchmark_ticker)

    risk_budget_portfolio = qis.backtest_model_portfolio(prices=prices, weights=risk_budgets_weights,
                                                         ticker=strategy_ticker)

    multi_portfolio_data = qis.MultiPortfolioData(portfolio_datas=[risk_budget_portfolio, static_portfolio],
                                                  benchmark_prices=prices.iloc[:, 0],
                                                  covar_dict=covar_dict)

    report_kwargs = qis.fetch_default_report_kwargs(reporting_frequency=qis.ReportingFrequency.MONTHLY,
                                                    add_rates_data=False)

    figs = qis.generate_strategy_benchmark_factsheet_plt(multi_portfolio_data=multi_portfolio_data,
                                                         time_period=time_period,
                                                          strategy_idx=0,
                                                          benchmark_idx=1,
                                                          add_benchmarks_to_navs=False,
                                                          add_exposures_comp=False,
                                                          add_strategy_factsheet=False,
                                                          **report_kwargs)

    # plot strategy and benchmark weights by ac
    kwargs = qis.update_kwargs(report_kwargs, dict(strategy_ticker=f"(B) {strategy_ticker}",
                                            benchmark_ticker=f"(A) {benchmark_ticker}"))
    fig, axs = plt.subplots(1, 2, figsize=figsize, tight_layout=True)
    if add_titles:
        qis.set_suptitle(fig, title=f"Time series of weights by asset classes")
    figs.append(fig)
    qis.plot_exposures_strategy_vs_benchmark_stack(strategy_exposures=risk_budgets_weights,
                                               benchmark_exposures=static_weights,
                                               axs=axs,
                                               var_format=var_format,
                                               **kwargs)

    # risk contributions
    rc_kwargs = dict(covar_dict=multi_portfolio_data.covar_dict, normalise=True)
    strategy_risk_contributions_ac = risk_budget_portfolio.compute_risk_contributions_implied_by_covar(**rc_kwargs)
    benchmark_risk_contributions_ac = static_portfolio.compute_risk_contributions_implied_by_covar(**rc_kwargs)
    fig, axs = plt.subplots(1, 2, figsize=figsize, tight_layout=True)
    if add_titles:
        qis.set_suptitle(fig, title=f"Time Series of risk contributions by asset classes")
    figs.append(fig)
    qis.plot_exposures_strategy_vs_benchmark_stack(strategy_exposures=strategy_risk_contributions_ac,
                                                   benchmark_exposures=benchmark_risk_contributions_ac,
                                                   var_format=var_format,
                                                   axs=axs,
                                                   **kwargs)

    # portfolio vol
    strategy_ex_anti_vol = risk_budget_portfolio.compute_ex_anti_portfolio_vol_implied_by_covar(covar_dict=covar_dict)
    benchmark_ex_anti_vol = static_portfolio.compute_ex_anti_portfolio_vol_implied_by_covar(covar_dict=covar_dict)
    ex_anti_vols = pd.concat([strategy_ex_anti_vol, benchmark_ex_anti_vol], axis=1)
    fig, ax = plt.subplots(1, 1, figsize=figsize, tight_layout=True)
    figs.append(fig)
    qis.plot_time_series(df=ex_anti_vols, var_format='{:.2%}',
                         title='Ex-anti volatilities',
                         ax=ax, **report_kwargs)
    return figs


class UnitTests(Enum):
    SOLVE_FOR_RISK_BUDGETS = 1
    ILLUSTRATE_WEIGHTS = 2


@qis.timer
def run_unit_test(unit_test: UnitTests):

    from optimalportfolios import local_path as lp

    is_60_40 = False

    if is_60_40:
        given_static_weights = {'SPY': 0.6, 'IEF': 0.4}
    else:
        given_static_weights = {'SPY': 0.55, 'IEF': 0.35, 'GLD': 0.1}
    given_static_weights = pd.Series(given_static_weights)

    prices = yf.download(tickers=given_static_weights.index.to_list(), start=None, end=None, ignore_tz=True)['Close']
    prices = prices[given_static_weights.index].dropna()
    print(prices)

    time_period = qis.TimePeriod('31Dec2004', '27Jun2025')
    rebalancing_freq = 'QE'

    # compute covar matrix using 1y span
    covar_dict = estimate_rolling_ewma_covar(prices=prices,
                                             time_period=time_period,
                                             rebalancing_freq=rebalancing_freq,
                                             returns_freq='W-WED',
                                             span=52)

    if unit_test == UnitTests.SOLVE_FOR_RISK_BUDGETS:
        risk_budgets = solve_for_risk_budgets_from_given_weights(prices=prices,
                                                                 given_weights=given_static_weights,
                                                                 time_period=time_period,
                                                                 covar_dict=covar_dict)
        print(risk_budgets)

    elif unit_test == UnitTests.ILLUSTRATE_WEIGHTS:
        risk_budgets = solve_for_risk_budgets_from_given_weights(prices=prices,
                                                                 given_weights=given_static_weights,
                                                                 time_period=time_period,
                                                                 covar_dict=covar_dict)
        risk_budgets_weights = rolling_risk_budgeting(prices=prices,
                                                      time_period=time_period,
                                                      covar_dict=covar_dict,
                                                      risk_budget=risk_budgets,
                                                      constraints0=Constraints(is_long_only=True))

        figs = plot_static_risk_budgets_vs_weights(prices=prices,
                                                   risk_budgets_weights=risk_budgets_weights,
                                                   given_static_weights=given_static_weights,
                                                   covar_dict=covar_dict,
                                                   time_period=time_period)
        qis.save_figs_to_pdf(figs, file_name='risk_budget_portfolio', local_path=lp.get_output_path())


if __name__ == '__main__':

    unit_test = UnitTests.ILLUSTRATE_WEIGHTS

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
