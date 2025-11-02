"""
backtest multiple covariance estimators given parameter backtest several optimisers
"""
# imports
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Optional
from enum import Enum
import qis as qis

# package
from optimalportfolios import (Constraints, PortfolioObjective,
                               backtest_rolling_optimal_portfolio,
                               estimate_rolling_ewma_covar,
                               LassoModelType, LassoModel,
                               estimate_rolling_lasso_covar)
from optimalportfolios.examples.universe import fetch_benchmark_universe_data

SUPPORTED_SOLVERS = [PortfolioObjective.EQUAL_RISK_CONTRIBUTION,
                     PortfolioObjective.MIN_VARIANCE,
                     PortfolioObjective.MAX_DIVERSIFICATION]


def run_multi_covar_estimators_backtest(prices: pd.DataFrame,
                                        benchmark_prices: pd.DataFrame,
                                        ac_benchmark_prices: pd.DataFrame,
                                        group_data: pd.Series,
                                        time_period: qis.TimePeriod,  # for weights
                                        perf_time_period: qis.TimePeriod,  # for reporting
                                        returns_freq: str = 'W-WED',  # covar matrix estimation on weekly returns
                                        rebalancing_freq: str = 'QE',  # portfolio rebalancing
                                        span: int = 52,  # ewma span for covariance matrix estimation: span = 1y of weekly returns
                                        portfolio_objective: PortfolioObjective = PortfolioObjective.MAX_DIVERSIFICATION,
                                        squeeze_factor: Optional[float] = None
                                        ) -> List[plt.Figure]:
    """
    backtest multi covar estimation
    test maximum diversification optimiser to span parameter
    portfolios are rebalanced at rebalancing_freq
    """
    if portfolio_objective not in SUPPORTED_SOLVERS:
        raise NotImplementedError(f"not supported {portfolio_objective}")

    # 1. EWMA covar
    ewma_covars = estimate_rolling_ewma_covar(prices=prices, time_period=time_period,
                                              rebalancing_freq=rebalancing_freq,
                                              returns_freq=returns_freq,
                                              span=span,
                                              is_apply_vol_normalised_returns=False,
                                              squeeze_factor=squeeze_factor)
    # 2. ewma covar with vol norm returns
    ewma_covars_vol_norm = estimate_rolling_ewma_covar(prices=prices, time_period=time_period,
                                                       rebalancing_freq=rebalancing_freq,
                                                       returns_freq=returns_freq,
                                                       span=span,
                                                       is_apply_vol_normalised_returns=True,
                                                       squeeze_factor=squeeze_factor)
    # lasso params
    lasso_kwargs = dict(risk_factor_prices=ac_benchmark_prices,
                        prices=prices,
                        time_period=time_period,
                        returns_freq=returns_freq,
                        rebalancing_freq=rebalancing_freq,
                        span=span,
                        squeeze_factor=squeeze_factor,
                        residual_var_weight=1.0)
    # 3. Group Lasso model using ac_benchmarks from universe
    lasso_model = LassoModel(model_type=LassoModelType.LASSO,
                             group_data=group_data, reg_lambda=1e-6, span=span,
                             warmup_period=span, solver='ECOS_BB')
    lasso_covar_data = estimate_rolling_lasso_covar(lasso_model=lasso_model,
                                                    is_apply_vol_normalised_returns=False,
                                                    **lasso_kwargs)
    lasso_covars = lasso_covar_data.y_covars
    lasso_covar_data_norm = estimate_rolling_lasso_covar(lasso_model=lasso_model,
                                                     is_apply_vol_normalised_returns=True,
                                                     **lasso_kwargs)
    lasso_covars_norm = lasso_covar_data_norm.y_covars


    # 4. Group Lasso model using ac_benchmarks from universe
    group_lasso_model = LassoModel(model_type=LassoModelType.GROUP_LASSO,
                             group_data=group_data, reg_lambda=1e-6, span=span, solver='ECOS_BB')
    group_lasso_covars = estimate_rolling_lasso_covar(lasso_model=group_lasso_model,
                                                      is_apply_vol_normalised_returns=False,
                                                      **lasso_kwargs)
    group_lasso_covars = group_lasso_covars.y_covars
    group_lasso_covars_norm = estimate_rolling_lasso_covar(lasso_model=group_lasso_model,
                                                           is_apply_vol_normalised_returns=True,
                                                           **lasso_kwargs)
    group_lasso_covars_norm = group_lasso_covars_norm.y_covars
    # create dict of estimated covars
    covars_dict = {'EWMA': ewma_covars, 'EWMA vol norm': ewma_covars_vol_norm,
                   'Lasso': lasso_covars, 'Lasso VolNorn': lasso_covars_norm,
                   'Group Lasso': group_lasso_covars, 'Group Lasso VolNorn': group_lasso_covars_norm}

    # set global constaints for portfolios
    constraints = Constraints(is_long_only=True,
                               min_weights=pd.Series(0.0, index=prices.columns),
                               max_weights=pd.Series(0.5, index=prices.columns))

    # now create a list of portfolios
    portfolio_datas = []
    for key, covar_dict in covars_dict.items():
        portfolio_data = backtest_rolling_optimal_portfolio(prices=prices,
                                                            portfolio_objective=portfolio_objective,
                                                            constraints=constraints,
                                                            time_period=time_period,
                                                            perf_time_period=perf_time_period,
                                                            covar_dict=covar_dict,
                                                            rebalancing_costs=0.0010,  # 10bp for rebalancin
                                                            weight_implementation_lag=1,  # weights are implemnted next day after comuting
                                                            ticker=f"{key}"  # portfolio id
                                                            )
        portfolio_data.set_group_data(group_data=group_data)
        portfolio_datas.append(portfolio_data)

    # run cross portfolio report
    multi_portfolio_data = qis.MultiPortfolioData(portfolio_datas=portfolio_datas, benchmark_prices=benchmark_prices)
    figs = qis.generate_multi_portfolio_factsheet(multi_portfolio_data=multi_portfolio_data,
                                                  time_period=time_period,
                                                  add_strategy_factsheets=False,
                                                  **qis.fetch_default_report_kwargs(time_period=time_period))
    return figs


class LocalTests(Enum):
    MULTI_COVAR_ESTIMATORS_BACKTEST = 1


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """

    import optimalportfolios.local_path as local_path

    if local_test == LocalTests.MULTI_COVAR_ESTIMATORS_BACKTEST:
        # portfolio_objective = PortfolioObjective.MAX_DIVERSIFICATION
        # portfolio_objective = PortfolioObjective.EQUAL_RISK_CONTRIBUTION
        portfolio_objective = PortfolioObjective.MIN_VARIANCE

        # params = dict(returns_freq='W-WED', rebalancing_freq='QE', span=52)
        params = dict(returns_freq='ME', rebalancing_freq='ME', span=12, squeeze_factor=0.01)

        prices, benchmark_prices, ac_loadings, benchmark_weights, group_data, ac_benchmark_prices = fetch_benchmark_universe_data()
        time_period = qis.TimePeriod(start='31Dec1998', end=prices.index[-1])  # backtest start: need 6y of data for rolling Sharpe and max mixure portfolios
        perf_time_period = qis.TimePeriod(start='31Dec2004', end=prices.index[-1])  # backtest reporting
        figs = run_multi_covar_estimators_backtest(prices=prices,
                                                   benchmark_prices=benchmark_prices,
                                                   ac_benchmark_prices=ac_benchmark_prices,
                                                   group_data=group_data,
                                                   time_period=time_period,
                                                   perf_time_period=perf_time_period,
                                                   portfolio_objective=portfolio_objective,
                                                   **params)

        # save png and pdf
        # qis.save_fig(fig=figs[0], file_name=f"{portfolio_objective.value}_multi_covar_estimator_backtest", local_path=f"figures/")
        qis.save_figs_to_pdf(figs=figs,
                             file_name=f"{portfolio_objective.value} multi_covar_estimator_backtest",
                             orientation='landscape',
                             local_path=local_path.get_output_path())
    plt.show()


if __name__ == '__main__':

    run_local_test(local_test=LocalTests.MULTI_COVAR_ESTIMATORS_BACKTEST)
