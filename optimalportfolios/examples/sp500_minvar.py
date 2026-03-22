"""
Run Minimum Variance portfolio optimiser for S&P 500 universe.
Cross-backtest sensitivity of EWMA span for factor covariance estimation.

Spans tested correspond to half-life periods in weekly returns:
    26  = 6 months
    52  = 1 year
    104 = 2 years
    208 = 4 years
"""

import pandas as pd
import qis as qis
import yfinance as yf
from typing import List
from enum import Enum

from optimalportfolios import (
    PortfolioObjective,
    Constraints,
    rolling_quadratic_optimisation,
    EwmaCovarEstimator,
)


def run_cross_backtest(prices: pd.DataFrame,
                       inclusion_indicators: pd.DataFrame,
                       group_data: pd.Series,
                       time_period: qis.TimePeriod,
                       spans: List[int] = (26, 52, 104, 208),
                       ) -> List[qis.PortfolioData]:
    """
    Cross-backtest minimum variance portfolios across different EWMA spans.

    Args:
        prices: Asset price DataFrame.
        inclusion_indicators: S&P 500 inclusion indicators DataFrame.
        group_data: Sector group universe Series.
        time_period: Backtest time period.
        spans: List of EWMA spans for factor covariance estimation.
            26=6m, 52=1y, 104=2y, 208=4y half-life in weekly returns.

    Returns:
        List of PortfolioData objects, one per span.
    """
    constraints = Constraints(is_long_only=True,
                              min_weights=pd.Series(0.0, index=prices.columns),
                              max_weights=pd.Series(0.05, index=prices.columns))

    portfolio_datas = []
    for span in spans:
        covar_estimator = EwmaCovarEstimator(
            span=span,
            returns_freq='W-WED',
            rebalancing_freq='QE',
        )
        covar_dict = covar_estimator.fit_rolling_covars(prices=prices, time_period=time_period)
        weights = rolling_quadratic_optimisation(
            covar_dict=covar_dict,
            prices=prices,
            constraints=constraints,
            portfolio_objective=PortfolioObjective.MIN_VARIANCE,
            inclusion_indicators=inclusion_indicators,
        )
        portfolio_data = qis.backtest_model_portfolio(
            prices=time_period.locate(prices),
            weights=weights,
            ticker=f"span={span}",
            funding_rate=None,
            weight_implementation_lag=1,
            rebalancing_costs=0.0030,
        )
        portfolio_data.set_group_data(group_data=group_data)
        portfolio_datas.append(portfolio_data)
    return portfolio_datas


class LocalTests(Enum):
    CROSS_BACKTEST = 1


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes."""

    import quant_strats.local_path as lp
    from optimalportfolios.examples.sp500_universe import load_sp500_universe_yahoo

    if local_test == LocalTests.CROSS_BACKTEST:

        time_period = qis.TimePeriod('31Dec2010', '31Jan2024')
        spans = [26, 52, 104, 208]

        prices, inclusion_indicators, group_data = load_sp500_universe_yahoo()

        portfolio_datas = run_cross_backtest(
            prices=prices,
            inclusion_indicators=inclusion_indicators,
            group_data=group_data,
            time_period=time_period,
            spans=spans,
        )

        benchmark_prices = yf.download(
            'SPY', start="2003-12-31", end=None, ignore_tz=True, auto_adjust=True
        )['Close'].asfreq('B').ffill()

        multi_portfolio_data = qis.MultiPortfolioData(
            portfolio_datas=portfolio_datas,
            benchmark_prices=benchmark_prices,
        )

        figs = qis.generate_multi_portfolio_factsheet(
            multi_portfolio_data=multi_portfolio_data,
            time_period=time_period,
            add_benchmarks_to_navs=True,
            add_strategy_factsheets=False,
            **qis.fetch_default_report_kwargs(time_period=time_period),
        )

        qis.save_figs_to_pdf(
            figs=figs,
            file_name=f"sp500_span_sensitivity_factsheet",
            orientation='landscape',
            local_path=lp.get_output_path(),
        )


if __name__ == '__main__':
    run_local_test(local_test=LocalTests.CROSS_BACKTEST)