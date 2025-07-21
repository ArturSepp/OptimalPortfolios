"""
example of minimization of tracking error
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import qis as qis
from enum import Enum

from optimalportfolios import (Constraints, GroupLowerUpperConstraints,
                               compute_tre_turnover_stats,
                               rolling_maximize_portfolio_sharpe,
                               wrapper_maximize_portfolio_sharpe)

from optimalportfolios.examples.universe import fetch_benchmark_universe_data


class UnitTests(Enum):
    ONE_STEP_OPTIMISATION = 1
    ROLLING_OPTIMISATION = 2


def run_unit_test(unit_test: UnitTests):

    import optimalportfolios.local_path as lp

    prices, benchmark_prices, ac_loadings, benchmark_weights, group_data, ac_benchmark_prices = fetch_benchmark_universe_data()

    # add costraints that each asset class is 10% <= sum ac weights <= 30% (benchamrk is 20% each)
    group_min_allocation = pd.Series(0.05, index=ac_loadings.columns)
    group_max_allocation = pd.Series(0.25, index=ac_loadings.columns)
    group_lower_upper_constraints = GroupLowerUpperConstraints(group_loadings=ac_loadings,
                                                               group_min_allocation=group_min_allocation,
                                                               group_max_allocation=group_max_allocation)
    constraints0 = Constraints(is_long_only=True,
                               group_lower_upper_constraints=group_lower_upper_constraints,
                               min_weights=pd.Series(0.0, index=prices.columns),
                               max_weights=pd.Series(1.0, index=prices.columns),
                               weights_0=benchmark_weights)

    if unit_test == UnitTests.ONE_STEP_OPTIMISATION:
        # optimise using last available data as inputs
        returns = qis.to_returns(prices, freq='W-WED', is_log_returns=True)
        pd_covar = pd.DataFrame(52.0 * qis.compute_masked_covar_corr(data=returns, is_covar=True),
                                index=prices.columns, columns=prices.columns)
        print(f"pd_covar=\n{pd_covar}")

        weights = wrapper_maximize_portfolio_sharpe(pd_covar=pd_covar,
                                                    means=52.0*returns.mean(0),
                                                    constraints0=constraints0,
                                                    weights_0=benchmark_weights)

        df_weight = pd.concat([benchmark_weights.rename('benchmark'), weights.rename('portfolio')], axis=1)
        print(f"weights=\n{df_weight}")
        qis.plot_bars(df=df_weight)

        te_vol, turnover, alpha, port_vol, benchmark_vol = compute_tre_turnover_stats(covar=pd_covar.to_numpy(),
                                                                                      benchmark_weights=benchmark_weights,
                                                                                      weights=weights,
                                                                                      weights_0=benchmark_weights)
        print(f"port_vol={port_vol:0.4f}, benchmark_vol={benchmark_vol:0.4f}, te_vol={te_vol:0.4f}, "
              f"turnover={turnover:0.4f}, alpha={alpha:0.4f}")

        plt.show()

    elif unit_test == UnitTests.ROLLING_OPTIMISATION:
        # optimise using last available data as inputs
        time_period = qis.TimePeriod('31Jan2007', '17Apr2025')
        rebalancing_costs = 0.0003

        weights = rolling_maximize_portfolio_sharpe(prices=prices,
                                                    constraints0=constraints0,
                                                    time_period=time_period)

        print(weights)

        portfolio_dict = {'Optimal Portfolio': weights,
                          'EqualWeight Portfolio': qis.df_to_equal_weight_allocation(prices, index=weights.index)}
        portfolio_datas = []
        for ticker, weights in portfolio_dict.items():
            portfolio_data = qis.backtest_model_portfolio(prices=prices,
                                                          weights=weights,
                                                          rebalancing_costs=rebalancing_costs,
                                                          weight_implementation_lag=1,
                                                          ticker=ticker)
            portfolio_data.set_group_data(group_data=group_data)
            portfolio_datas.append(portfolio_data)
        multi_portfolio_data = qis.MultiPortfolioData(portfolio_datas, benchmark_prices=benchmark_prices)
        kwargs = qis.fetch_default_report_kwargs(time_period=time_period, add_rates_data=True)
        figs = qis.generate_strategy_benchmark_factsheet_plt(multi_portfolio_data=multi_portfolio_data,
                                                             time_period=time_period,
                                                             add_strategy_factsheet=True,
                                                             add_grouped_exposures=False,
                                                             add_grouped_cum_pnl=False,
                                                             **kwargs)
        qis.save_figs_to_pdf(figs=figs,
                             file_name=f"max sharpe portfolio", orientation='landscape',
                             local_path=lp.get_output_path())


if __name__ == '__main__':

    unit_test = UnitTests.ROLLING_OPTIMISATION

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
