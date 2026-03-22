"""
Backtest multiple covariance estimators given parameter backtest several optimisers.

Compares EWMA vs LASSO-based factor covariance estimation using:
- EWMA (plain and vol-normalised)
- LASSO factor model (plain and vol-normalised)
- Group LASSO factor model (plain and vol-normalised)

Uses FactorCovarEstimator with qis.compute_asset_returns_dict() for the

Reference:
    Sepp A., Ossa I., and Kastenholz M. (2026),
    "Robust Optimization of Strategic and Tactical Asset Allocation for Multi-Asset Portfolios",
    The Journal of Portfolio Management, 52(4), 86-120.
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
                               EwmaCovarEstimator,
                               LassoModelType, LassoModel,
                               FactorCovarEstimator)
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
                                        returns_freq: str = 'ME',  # covar matrix estimation on weekly returns
                                        rebalancing_freq: str = 'QE',  # portfolio rebalancing
                                        span: int = 52,  # ewma span for covariance matrix estimation
                                        portfolio_objective: PortfolioObjective = PortfolioObjective.MAX_DIVERSIFICATION,
                                        ) -> List[plt.Figure]:
    """
    Backtest multi covar estimation.
    Test maximum diversification optimiser to span parameter.
    Portfolios are rebalanced at rebalancing_freq.
    """
    if portfolio_objective not in SUPPORTED_SOLVERS:
        raise NotImplementedError(f"not supported {portfolio_objective}")

    # 1. EWMA covar
    ewma_covar_estimator = EwmaCovarEstimator(rebalancing_freq=rebalancing_freq,
                                              returns_freq=returns_freq,
                                              span=span,
                                              is_apply_vol_normalised_returns=False)
    ewma_covars = ewma_covar_estimator.fit_rolling_covars(prices=prices, time_period=time_period)

    # 2. EWMA covar with vol norm returns
    ewma_covar_estimator_norm = EwmaCovarEstimator(rebalancing_freq=rebalancing_freq,
                                              returns_freq=returns_freq,
                                              span=span,
                                              is_apply_vol_normalised_returns=True)
    ewma_covars_vol_norm = ewma_covar_estimator_norm.fit_rolling_covars(prices=prices, time_period=time_period)

    # precompute asset returns dict for lasso-based estimators
    asset_returns_dict = qis.compute_asset_returns_dict(
        prices=prices, is_log_returns=True, returns_freqs=returns_freq,
    )
    # 3. LASSO factor model
    lasso_model = LassoModel(model_type=LassoModelType.LASSO,
                             group_data=group_data, reg_lambda=1e-6, span=span,
                             warmup_period=span, solver='CLARABEL')

    lasso_estimator = FactorCovarEstimator(lasso_model=lasso_model,
                                           factor_returns_freq=returns_freq,
                                           rebalancing_freq=rebalancing_freq)

    lasso_covar_data = lasso_estimator.fit_rolling_factor_covars(
        risk_factor_prices=ac_benchmark_prices,
        asset_returns_dict=asset_returns_dict,
        time_period=time_period
    )
    lasso_covars = lasso_covar_data.get_y_covars()

    # 4. LASSO with vol-normalised returns
    asset_returns_dict_vol_norm = qis.compute_asset_returns_dict(
        prices=prices, is_log_returns=True, returns_freqs=returns_freq,
    )

    lasso_covar_data_norm = lasso_estimator.fit_rolling_factor_covars(
        risk_factor_prices=ac_benchmark_prices,
        asset_returns_dict=asset_returns_dict_vol_norm,
        time_period=time_period,
    )
    lasso_covars_norm = lasso_covar_data_norm.get_y_covars()

    # 5. Group LASSO factor model
    group_lasso_model = LassoModel(model_type=LassoModelType.GROUP_LASSO,
                                   group_data=group_data, reg_lambda=1e-6,
                                   span=span, solver='CLARABEL')

    group_lasso_estimator = FactorCovarEstimator(lasso_model=group_lasso_model,
                                                  factor_returns_freq=returns_freq,
                                                  rebalancing_freq=rebalancing_freq)

    group_lasso_covar_data = group_lasso_estimator.fit_rolling_factor_covars(
        risk_factor_prices=ac_benchmark_prices,
        asset_returns_dict=asset_returns_dict,
        time_period=time_period,
    )
    group_lasso_covars = group_lasso_covar_data.get_y_covars()

    # 6. Group LASSO with vol-normalised returns
    group_lasso_covar_data_norm = group_lasso_estimator.fit_rolling_factor_covars(
        risk_factor_prices=ac_benchmark_prices,
        asset_returns_dict=asset_returns_dict_vol_norm,
        time_period=time_period,
    )
    group_lasso_covars_norm = group_lasso_covar_data_norm.get_y_covars()

    # create dict of estimated covars
    covars_dict = {'EWMA': ewma_covars, 'EWMA vol norm': ewma_covars_vol_norm,
                   'Lasso': lasso_covars, 'Lasso VolNorm': lasso_covars_norm,
                   'Group Lasso': group_lasso_covars, 'Group Lasso VolNorm': group_lasso_covars_norm}

    # set global constraints for portfolios
    constraints = Constraints(is_long_only=True,
                               min_weights=pd.Series(0.0, index=prices.columns),
                               max_weights=pd.Series(0.5, index=prices.columns))

    # backtest each covar estimator
    portfolio_datas = []
    for key, covar_dict in covars_dict.items():
        portfolio_data = backtest_rolling_optimal_portfolio(prices=prices,
                                                            portfolio_objective=portfolio_objective,
                                                            constraints=constraints,
                                                            perf_time_period=perf_time_period,
                                                            covar_dict=covar_dict,
                                                            rebalancing_costs=0.0010,
                                                            weight_implementation_lag=1,
                                                            ticker=f"{key}")
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
    """Run local tests for development and debugging purposes."""

    if local_test == LocalTests.MULTI_COVAR_ESTIMATORS_BACKTEST:
        portfolio_objective = PortfolioObjective.MIN_VARIANCE
        params = dict(returns_freq='ME', rebalancing_freq='QE', span=36)

        prices, benchmark_prices, ac_loadings, benchmark_weights, group_data, ac_benchmark_prices = fetch_benchmark_universe_data()
        time_period = qis.TimePeriod(start='31Dec2006', end='15Mar2026')
        perf_time_period = qis.TimePeriod(start='31Dec2015', end='15Mar2026')
        figs = run_multi_covar_estimators_backtest(prices=prices,
                                                   benchmark_prices=benchmark_prices,
                                                   ac_benchmark_prices=ac_benchmark_prices,
                                                   group_data=group_data,
                                                   time_period=time_period,
                                                   perf_time_period=perf_time_period,
                                                   portfolio_objective=portfolio_objective,
                                                   **params)

        qis.save_figs_to_pdf(figs=figs,
                             file_name=f"{portfolio_objective.value} multi_covar_estimator_backtest",
                             orientation='landscape',
                             local_path=f"figures/")
    plt.show()


if __name__ == '__main__':

    run_local_test(local_test=LocalTests.MULTI_COVAR_ESTIMATORS_BACKTEST)
