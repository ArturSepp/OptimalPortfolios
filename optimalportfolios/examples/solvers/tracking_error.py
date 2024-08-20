"""
example of minimization of tracking error
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import qis as qis
from enum import Enum

from optimalportfolios import (Constraints, GroupLowerUpperConstraints,
                               compute_te_turnover,
                               rolling_maximise_alpha_over_tre,
                               wrapper_maximise_alpha_over_tre)

from optimalportfolios.examples.universe import fetch_benchmark_universe_data


def run_etf_tracking_portfolio(prices: pd.DataFrame,
                               benchmark_weights: pd.Series,
                               ac_loadings: pd.DataFrame,
                               time_period: qis.TimePeriod
                               ) -> pd.DataFrame:
    """
    run the optimal portfolio
    """
    momentum = qis.compute_ewm_long_short_filtered_ra_returns(returns=qis.to_returns(prices, freq='W-WED'), vol_span=13,
                                                              long_span=13, short_span=None, weight_lag=0)
    # momentum = qis.map_signal_to_weight(signals=momentum, loc=0.0, slope_right=0.5, slope_left=0.5, tail_level=3.0)
    alphas = qis.df_to_cross_sectional_score(df=momentum)

    # add costraints that each asset class is 10% <= sum ac weights <= 30% (benchamrk is 20% each)
    group_min_allocation = pd.Series(0.1, index=ac_loadings.columns)
    group_max_allocation = pd.Series(0.4, index=ac_loadings.columns)
    group_lower_upper_constraints = GroupLowerUpperConstraints(group_loadings=ac_loadings,
                                                               group_min_allocation=group_min_allocation,
                                                               group_max_allocation=group_max_allocation)

    constraints0 = Constraints(is_long_only=True,
                               min_weights=0.0 * benchmark_weights,
                               max_weights=3.0 * benchmark_weights,
                               tracking_err_vol_constraint=0.05,  # annualised vol difference
                               turnover_constraint=1.00,  # max turover at rebalancing
                               weights_0=benchmark_weights,
                               group_lower_upper_constraints=group_lower_upper_constraints)

    weights = rolling_maximise_alpha_over_tre(prices=prices,
                                              alphas=alphas,
                                              benchmark_weights=benchmark_weights,
                                              constraints0=constraints0,
                                              time_period=time_period)
    return weights


class UnitTests(Enum):
    ONE_STEP_OPTIMISATION = 1
    TRACKING_ERROR_GRID = 2
    ROLLING_OPTIMISATION = 3


def run_unit_test(unit_test: UnitTests):

    import optimalportfolios.local_path as lp

    prices, benchmark_prices, ac_loadings, benchmark_weights, group_data = fetch_benchmark_universe_data()

    if unit_test == UnitTests.ONE_STEP_OPTIMISATION:
        # optimise using last available data as inputs
        returns = qis.to_returns(prices, freq='W-WED', is_log_returns=True)
        pd_covar = pd.DataFrame(52.0 * qis.compute_masked_covar_corr(data=returns, is_covar=True),
                                index=prices.columns, columns=prices.columns)
        momenum_1y = prices.divide(prices.shift(260)) - 1.0
        alphas = qis.df_to_cross_sectional_score(df=momenum_1y.iloc[-1, :])
        print(f"pd_covar=\n{pd_covar}")
        print(f"alphas=\n{alphas}")

        # add costraints that each asset class is 10% <= sum ac weights <= 30% (benchamrk is 20% each)
        group_min_allocation = pd.Series(0.1, index=ac_loadings.columns)
        group_max_allocation = pd.Series(0.3, index=ac_loadings.columns)
        group_lower_upper_constraints = GroupLowerUpperConstraints(group_loadings=ac_loadings,
                                                                   group_min_allocation=group_min_allocation,
                                                                   group_max_allocation=group_max_allocation)

        constraints0 = Constraints(is_long_only=True,
                                   min_weights=0.0*benchmark_weights,
                                   max_weights=2.0*benchmark_weights,
                                   tracking_err_vol_constraint=0.06,  # annualised vol difference
                                   turnover_constraint=0.75,  # max turover at rebalancing
                                   weights_0=benchmark_weights,
                                   group_lower_upper_constraints=group_lower_upper_constraints)

        weights = wrapper_maximise_alpha_over_tre(pd_covar=pd_covar,
                                                  alphas=alphas,
                                                  benchmark_weights=benchmark_weights,
                                                  constraints0=constraints0,
                                                  weights_0=benchmark_weights)

        df_weight = pd.concat([benchmark_weights.rename('benchmark'),
                               weights.rename('portfolio'),
                               alphas.rename('alpha')],
                              axis=1)
        print(f"df_weight=\n{df_weight}")
        qis.plot_bars(df=df_weight.iloc[:, :2])

        te_vol, turnover, alpha, port_vol, benchmark_vol = compute_te_turnover(covar=pd_covar.to_numpy(),
                                                                               benchmark_weights=benchmark_weights,
                                                                               weights=weights,
                                                                               weights_0=benchmark_weights,
                                                                               alphas=alphas)
        print(f"port_vol={port_vol:0.4f}, benchmark_vol={benchmark_vol:0.4f}, te_vol={te_vol:0.4f}, "
              f"turnover={turnover:0.4f}, alpha={alpha:0.4f}")

        plt.show()

    elif unit_test == UnitTests.TRACKING_ERROR_GRID:

        # optimise using last available data as inputs
        returns = qis.to_returns(prices, freq='W-WED', is_log_returns=True)
        pd_covar = pd.DataFrame(52.0 * qis.compute_masked_covar_corr(data=returns, is_covar=True),
                                index=prices.columns, columns=prices.columns)
        momenum_1y = prices.divide(prices.shift(260)) - 1.0
        alphas = qis.df_to_cross_sectional_score(df=momenum_1y.iloc[-1, :])
        print(f"pd_covar=\n{pd_covar}")
        print(f"alphas=\n{alphas}")

        tracking_err_vol_constraints = [0.01, 0.02, 0.03, 0.05, 0.1]
        turnover_constraints = [0.1, 0.25, 0.5, 1.0, 10.0]

        weights_grid = {}
        port_vols, te_vols, turnovers, port_alphas = {}, {}, {}, {}
        for tracking_err_vol_constraint in tracking_err_vol_constraints:
            port_vols_, te_vols_, turnovers_, port_alphas_ = {}, {}, {}, {}
            for turnover_constraint in turnover_constraints:
                port_name = f"te_vol<{tracking_err_vol_constraint:0.2f}, turnover<{turnover_constraint:0.2f}"

                constraints0 = Constraints(is_long_only=True,
                                           min_weights=0.0 * benchmark_weights,
                                           max_weights=2.0 * benchmark_weights,
                                           tracking_err_vol_constraint=tracking_err_vol_constraint,  # annualised vol difference
                                           turnover_constraint=turnover_constraint,  # max turover at rebalancing
                                           weights_0=benchmark_weights)

                weights = wrapper_maximise_alpha_over_tre(pd_covar=pd_covar,
                                                          alphas=alphas,
                                                          benchmark_weights=benchmark_weights,
                                                          constraints0=constraints0,
                                                          weights_0=benchmark_weights)

                weights = pd.Series(weights, index=prices.columns)
                weights_grid[port_name] = weights
                te_vol, turnover, port_alpha, port_vol, benchmark_vol = compute_te_turnover(covar=pd_covar.to_numpy(),
                                                                                            benchmark_weights=benchmark_weights,
                                                                                            weights=weights,
                                                                                            weights_0=benchmark_weights,
                                                                                            alphas=alphas)
                port_name_ = f"turnover<{turnover_constraint:0.2f}"
                port_vols_[port_name_] = port_vol
                te_vols_[port_name_] = te_vol
                turnovers_[port_name_] = turnover
                port_alphas_[port_name_] = port_alpha

            port_name_index = f"te_vol<{tracking_err_vol_constraint:0.2f}"
            port_vols[port_name_index] = pd.Series(port_vols_)
            te_vols[port_name_index] = pd.Series(te_vols_)
            turnovers[port_name_index] = pd.Series(turnovers_)
            port_alphas[port_name_index] = pd.Series(port_alphas_)
        port_vols = pd.DataFrame.from_dict(port_vols)
        te_vols = pd.DataFrame.from_dict(te_vols)
        turnovers = pd.DataFrame.from_dict(turnovers)
        port_alphas = pd.DataFrame.from_dict(port_alphas)

        print(f"port_vols=\n{port_vols}")
        print(f"te_vols=\n{te_vols}")
        print(f"turnovers=\n{turnovers}")
        print(f"port_alphas=\n{port_alphas}")

    elif unit_test == UnitTests.ROLLING_OPTIMISATION:
        # optimise using last available data as inputs
        time_period = qis.TimePeriod('31Jan2007', '19Jul2024')
        rebalancing_costs = 0.0003

        weights = run_etf_tracking_portfolio(prices=prices,
                                             benchmark_weights=benchmark_weights,
                                             ac_loadings=ac_loadings,
                                             time_period=time_period)
        print(weights)

        portfolio_data = qis.backtest_model_portfolio(prices=prices,
                                                      weights=weights,
                                                      rebalancing_costs=rebalancing_costs,
                                                      ticker=f"Optimal Portfolio")
        portfolio_data.set_group_data(group_data=group_data)
        # benchmark portfolio
        equal_weights = qis.df_to_equal_weight_allocation(weights)  # same allocatin with equal weights
        benchmark_portfolio_data = qis.backtest_model_portfolio(prices=prices,
                                                                weights=equal_weights,
                                                                rebalancing_costs=rebalancing_costs,
                                                                ticker=f"Benchmark Portfolio")
        benchmark_portfolio_data.set_group_data(group_data=group_data)

        kwargs = qis.fetch_default_report_kwargs(time_period=time_period, is_daily=True, add_rates_data=True)
        multi_portfolio_data = qis.MultiPortfolioData([portfolio_data, benchmark_portfolio_data],
                                                      benchmark_prices=benchmark_prices)
        figs = qis.generate_strategy_benchmark_factsheet_plt(multi_portfolio_data=multi_portfolio_data,
                                                             time_period=time_period,
                                                             add_strategy_factsheet=True,
                                                             add_grouped_exposures=False,
                                                             add_grouped_cum_pnl=False,
                                                             **kwargs)
        qis.save_figs_to_pdf(figs=figs,
                             file_name=f"tracking error portfolio", orientation='landscape',
                             local_path=lp.get_output_path())


if __name__ == '__main__':

    unit_test = UnitTests.ROLLING_OPTIMISATION

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
