"""
minimal example of using the backtester
"""
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import qis as qis

# package
from optimalportfolios import compute_rolling_optimal_weights, PortfolioObjective, Constraints, EwmaCovarEstimator
from optimalportfolios.examples.data.universe import fetch_minimal_universe_data
import optimalportfolios.local_path as lp


# Resolve the figures directory relative to this file, not the current working
# directory, so the script can be launched from anywhere.
FIGURES_PATH = str(Path(__file__).resolve().parent.parent / 'figures') + '/'
Path(FIGURES_PATH).mkdir(parents=True, exist_ok=True)


# 1. fetch universe (8 ETFs, 6 asset-class groups)
prices, benchmark_prices, group_data = fetch_minimal_universe_data()
print(prices)
time_period = qis.TimePeriod('31Dec2004', '15Mar2026')   # period for computing weights backtest

# 3.a. define optimisation setup
portfolio_objective = PortfolioObjective.MAX_DIVERSIFICATION  # define portfolio objective
returns_freq = 'W-WED'  # use weekly returns
rebalancing_freq = 'QE'  # weights rebalancing frequency: rebalancing is quarterly on WED
span = 52  # span of number of returns_freq-returns for covariance estimation = 12y
constraints = Constraints(is_long_only=True,
                           min_weights=pd.Series(0.0, index=prices.columns),
                           max_weights=pd.Series(0.5, index=prices.columns))

# 3.b. compute taa portfolio weights rebalanced every quarter
ewma_estimator = EwmaCovarEstimator(returns_freq=returns_freq, span=span, rebalancing_freq=rebalancing_freq)
covar_dict = ewma_estimator.fit_rolling_covars(prices=prices, time_period=time_period)
weights = compute_rolling_optimal_weights(prices=prices,
                                          portfolio_objective=portfolio_objective,
                                          constraints=constraints,
                                          time_period=time_period,
                                          rebalancing_freq=rebalancing_freq,
                                          covar_dict=covar_dict)

# 4. given portfolio weights, construct the performance of the portfolio
funding_rate = None  # on positive / negative cash balances
rebalancing_costs = 0.0010  # rebalancing costs per volume = 10bp
weight_implementation_lag = 1  # portfolio is implemented next day after weights are computed
portfolio_data = qis.backtest_model_portfolio(prices=prices.loc[weights.index[0]:, :],
                                              weights=weights,
                                              ticker='MaxDiversification',
                                              funding_rate=funding_rate,
                                              weight_implementation_lag=weight_implementation_lag,
                                              rebalancing_costs=rebalancing_costs)

# 5. using portfolio_data run the report with strategy factsheet
# for group-based report set_group_data
portfolio_data.set_group_data(group_data=group_data, group_order=list(group_data.unique()))
# set time period for portfolio report
figs = qis.generate_strategy_factsheet(portfolio_data=portfolio_data,
                                       benchmark_prices=benchmark_prices,
                                       add_current_position_var_risk_sheet=True,
                                       time_period=time_period,
                                       **qis.fetch_default_report_kwargs(time_period=time_period))
# save report to pdf and png
qis.save_figs_to_pdf(figs=figs,
                     file_name=f"{portfolio_data.nav.name}_portfolio_factsheet",
                     orientation='landscape',
                     local_path=lp.get_output_path())
qis.save_fig(fig=figs[0], file_name=f"example_portfolio_factsheet1", local_path=FIGURES_PATH)
qis.save_fig(fig=figs[1], file_name=f"example_portfolio_factsheet2", local_path=FIGURES_PATH)


# 6. can create customised report using portfolio_data custom report
def run_customised_reporting(portfolio_data) -> plt.Figure:
    with sns.axes_style("darkgrid"):
        fig, axs = plt.subplots(3, 1, figsize=(12, 12), tight_layout=True)
    perf_params = qis.PerfParams(freq='W-WED', freq_reg='ME')
    kwargs = dict(x_date_freq='YE', framealpha=0.8, perf_params=perf_params)
    portfolio_data.plot_nav(ax=axs[0], **kwargs)
    portfolio_data.plot_weights(ncol=len(prices.columns)//3,
                                legend_stats=qis.LegendStats.AVG_LAST,
                                title='Portfolio weights',
                                freq='QE',
                                ax=axs[1],
                                **kwargs)
    portfolio_data.plot_returns_scatter(benchmark_price=benchmark_prices.iloc[:, 0],
                                        ax=axs[2],
                                        **kwargs)
    return fig


# run customised report
fig = run_customised_reporting(portfolio_data)
# save png
qis.save_fig(fig=fig, file_name=f"example_customised_report", local_path=FIGURES_PATH)

plt.show()
