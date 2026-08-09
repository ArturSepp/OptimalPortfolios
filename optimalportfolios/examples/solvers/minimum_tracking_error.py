"""Minimum-tracking-error portfolio example with one-step and rolling variants."""
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import qis as qis

from optimalportfolios import (
    Constraints,
    EwmaCovarEstimator,
    GroupLowerUpperConstraints,
    compute_tre_turnover_stats,
    rolling_minimise_tracking_error,
    wrapper_minimise_tracking_error,
)
from optimalportfolios.examples.data.universe import fetch_benchmark_universe_data


class LocalTests(Enum):
    """Diagnostic modes supported by ``run_local_test``."""

    ONE_STEP_OPTIMISATION = 1
    ROLLING_OPTIMISATION = 2


def _build_constraints(
        benchmark_weights: pd.Series,
        ac_loadings: pd.DataFrame,
) -> Constraints:
    """Create a feasible set that forces a small deviation from the benchmark."""
    group_lower_upper_constraints = GroupLowerUpperConstraints(
        group_loadings=ac_loadings,
        group_min_allocation=pd.Series(0.1, index=ac_loadings.columns),
        group_max_allocation=pd.Series(0.3, index=ac_loadings.columns),
    )
    max_weights = pd.Series(0.20, index=benchmark_weights.index)
    max_weights.loc[benchmark_weights.index[0]] = 0.03
    return Constraints(
        is_long_only=True,
        min_weights=pd.Series(0.0, index=benchmark_weights.index),
        max_weights=max_weights,
        turnover_constraint=0.25,
        group_lower_upper_constraints=group_lower_upper_constraints,
    )


def run_local_test(local_test: LocalTests) -> None:
    """Run a minimum-tracking-error diagnostic or rolling backtest."""
    import optimalportfolios.local_path as lp

    (
        prices,
        benchmark_prices,
        ac_loadings,
        benchmark_weights,
        group_data,
        _ac_benchmark_prices,
    ) = fetch_benchmark_universe_data()
    constraints = _build_constraints(
        benchmark_weights=benchmark_weights,
        ac_loadings=ac_loadings,
    )

    if local_test == LocalTests.ONE_STEP_OPTIMISATION:
        returns = qis.to_returns(prices, freq='W-WED', is_log_returns=True)
        pd_covar = pd.DataFrame(
            52.0 * qis.compute_masked_covar_corr(data=returns, is_covar=True),
            index=prices.columns,
            columns=prices.columns,
        )
        weights, outcome = wrapper_minimise_tracking_error(
            pd_covar=pd_covar,
            benchmark_weights=benchmark_weights,
            constraints=constraints,
            weights_0=benchmark_weights,
        )

        weight_comparison = pd.concat(
            [benchmark_weights.rename('Benchmark'), weights.rename('Minimum TE')],
            axis=1,
        )
        print(f"solver_status={outcome.status}\nweights=\n{weight_comparison}")
        qis.plot_bars(df=weight_comparison)

        te_vol, turnover, _alpha, port_vol, benchmark_vol = compute_tre_turnover_stats(
            covar=pd_covar.to_numpy(),
            benchmark_weights=benchmark_weights,
            weights=weights,
            weights_0=benchmark_weights,
        )
        print(
            f"portfolio_vol={port_vol:0.4f}, benchmark_vol={benchmark_vol:0.4f}, "
            f"tracking_error={te_vol:0.4f}, turnover={turnover:0.4f}"
        )
        plt.show()

    elif local_test == LocalTests.ROLLING_OPTIMISATION:
        time_period = qis.TimePeriod('31Jan2007', '17Apr2025')
        rebalancing_costs = 0.0003
        covar_dict = EwmaCovarEstimator().fit_rolling_covars(
            prices=prices,
            time_period=time_period,
        )
        weights = rolling_minimise_tracking_error(
            prices=prices,
            constraints=constraints,
            benchmark_weights=benchmark_weights,
            covar_dict=covar_dict,
        )
        benchmark_weight_path = pd.DataFrame(
            np.tile(benchmark_weights.to_numpy(), (len(weights.index), 1)),
            index=weights.index,
            columns=benchmark_weights.index,
        )

        portfolio_datas = []
        for ticker, portfolio_weights in {
            'Minimum Tracking Error': weights,
            'Benchmark Portfolio': benchmark_weight_path,
        }.items():
            portfolio_data = qis.backtest_model_portfolio(
                prices=prices,
                weights=portfolio_weights,
                rebalancing_costs=rebalancing_costs,
                weight_implementation_lag=1,
                ticker=ticker,
            )
            portfolio_data.set_group_data(group_data=group_data)
            portfolio_datas.append(portfolio_data)

        multi_portfolio_data = qis.MultiPortfolioData(
            portfolio_datas,
            benchmark_prices=benchmark_prices,
        )
        kwargs = qis.fetch_default_report_kwargs(
            time_period=time_period,
            add_rates_data=True,
        )
        figs = qis.generate_strategy_benchmark_factsheet_plt(
            multi_portfolio_data=multi_portfolio_data,
            time_period=time_period,
            add_strategy_factsheet=True,
            add_grouped_exposures=False,
            add_grouped_cum_pnl=False,
            **kwargs,
        )
        qis.save_figs_to_pdf(
            figs=figs,
            file_name='minimum tracking error portfolio',
            orientation='landscape',
            local_path=lp.get_output_path(),
        )


if __name__ == '__main__':
    run_local_test(local_test=LocalTests.ROLLING_OPTIMISATION)
