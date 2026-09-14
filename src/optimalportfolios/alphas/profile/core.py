"""Rank-based alpha selection, QIS backtesting and reporting.

This module accepts precomputed score panels and forms equal-weight
top-quantile targets. Signal construction belongs in alphas.signals;
the adapters in signal_profilers construct scores and call this core.

QIS owns holdings simulation, performance statistics and factsheets. It holds
units between rebalances, so realized weights drift with prices. The core
returns alpha strategies followed by one equal-weight benchmark.

The source checkout's docs/alphas_module_readme.md documents methodology and
limitations; src/optimalportfolios/alphas/README.md gives an offline workflow.
"""
import numpy as np
import pandas as pd
import qis as qis
from typing import Dict, List, Optional, Union


def compute_top_quantile_equal_weights(alpha_scores: pd.DataFrame,
                                       prices: pd.DataFrame,
                                       quantile: float = 1.0 / 3.0,
                                       ) -> pd.DataFrame:
    """Build equal-weight targets for the highest-scoring eligible assets.

    On each score date, select ceil(quantile * n_available) assets and divide
    equally among them. Higher scores rank first; ties follow prices.columns
    order. Eligibility uses non-missing scores and prices, not a positivity or
    finiteness check. Validate those properties before calling.

    A quantile of 1.0 selects all score-eligible assets. This can differ from the
    equal-weight benchmark, which does not require an alpha score. This function
    forms targets only; it does not optimize, rebalance or simulate holdings.

    Args:
        alpha_scores: Date-by-ticker scores. Columns are reindexed to prices.columns:
            extra score columns are dropped and missing ones become NaN.
        prices: Date-by-ticker prices, reindexed to score dates for the eligibility
            mask without forward-filling. Column order also resolves score ties.
        quantile: Fraction of eligible assets to hold, in (0, 1]. The basket size
            rounds upward, so a nonempty eligible universe selects at least one asset.

    Returns:
        Target-weight DataFrame on the score index and price columns. Nonempty
        baskets sum to one; excluded assets and rows with no eligible asset are zero.

    Raises:
        ValueError: If quantile is outside (0, 1].
    """
    if not 0.0 < quantile <= 1.0:
        raise ValueError(f"quantile must lie in (0, 1], got {quantile!r}")
    if list(alpha_scores.columns) != list(prices.columns):
        alpha_scores = alpha_scores.reindex(columns=prices.columns)

    # an asset is selectable on a date only if it has a finite score AND a finite price there
    available = alpha_scores.notna() & prices.reindex(index=alpha_scores.index).notna()
    ranks = alpha_scores.where(available).rank(axis=1, ascending=False, method='first')
    n_available = available.sum(axis=1)
    n_hold = np.ceil(quantile * n_available).astype(int).clip(lower=0)

    hold_mask = ranks.le(n_hold, axis=0) & available            # top n_hold by score, per row
    weights = hold_mask.astype(float)
    row_sums = weights.sum(axis=1)
    weights = weights.divide(row_sums.where(row_sums > 0.0), axis=0).fillna(0.0)
    return weights


def backtest_alpha_rank_portfolio(prices: pd.DataFrame,
                                  alpha_scores: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
                                  quantile: float = 1.0 / 3.0,
                                  rebalancing_freq: str = 'QE',
                                  time_period: qis.TimePeriod = None,
                                  rebalancing_costs: Optional[pd.Series] = None,
                                  instruments_carry: Optional[pd.DataFrame] = None,
                                  strategy_ticker: str = 'Top-quantile',
                                  benchmark_ticker: str = 'Equal Weight',
                                  ) -> qis.MultiPortfolioData:
    """Backtest named top-quantile targets against one equal-weight benchmark.

    Each score panel produces a strategy through compute_top_quantile_equal_weights().
    A separate QIS equal-weight allocation uses non-missing prices independently
    of the scores. Target rows are optionally filtered by time_period and sampled
    with asfreq(rebalancing_freq, method='ffill') before each QIS backtest.

    The full price panel is passed through. The target-window end is therefore
    not a simulation end date: existing holdings can continue through later prices.
    Choose the price sample explicitly. QIS holds units between rebalances, and
    the wrapper leaves its weight-implementation lag at the default of zero.
    Supplied scores must be available at formation time.

    Args:
        prices: Date-by-ticker price panel for target eligibility and QIS simulation.
            Validate the calendar and price data before profiling.
        alpha_scores: One score DataFrame or a dictionary mapping strategy labels
            to score DataFrames. Higher scores rank first; non-missing score/price
            pairs are eligible. Dictionary iteration order determines strategy order.
        quantile: Fraction of score-eligible assets selected, in (0, 1].
        rebalancing_freq: Calendar frequency used to sample target weights, such
            as 'QE' or 'ME'. Dated target rows determine the QIS trading schedule.
        time_period: Optional window selecting target rows before frequency sampling.
            It does not truncate prices or define a reporting window.
        rebalancing_costs: Per-ticker proportional costs in fractional units, passed unchanged
            to QIS for every leg; 0.001 means 10 basis points. None means no costs.
        instruments_carry: Optional date-by-ticker annual fractional carry rates,
            passed to QIS for accrual on holdings. These affect simulated NAV,
            not just report labels. None adds no separate carry cash flows.
        strategy_ticker: Strategy label for a single score panel; dictionary keys
            take precedence when alpha_scores is a dictionary.
        benchmark_ticker: Label for the equal-weight reference.

    Returns:
        QIS MultiPortfolioData containing the strategy legs in input order and the
        equal-weight benchmark last. benchmark_prices contains that final leg's NAV.
        An empty dictionary produces only the benchmark; profile_alpha_signals()
        instead requires at least one named panel.
    """
    # normalise to a label -> score-panel dict so single-panel and multi-panel share one code path
    if isinstance(alpha_scores, pd.DataFrame):
        scores_by_label = {strategy_ticker: alpha_scores}
    else:
        scores_by_label = dict(alpha_scores)

    benchmark_weights = qis.df_to_equal_weight_allocation(df=prices)
    if time_period is not None:
        benchmark_weights = time_period.locate(benchmark_weights)
    benchmark_weights = benchmark_weights.asfreq(rebalancing_freq, method='ffill')

    portfolio_datas = []
    for label, scores in scores_by_label.items():
        strategy_weights = compute_top_quantile_equal_weights(
            alpha_scores=scores, prices=prices, quantile=quantile)
        if time_period is not None:
            strategy_weights = time_period.locate(strategy_weights)
        strategy_weights = strategy_weights.asfreq(rebalancing_freq, method='ffill')
        portfolio_datas.append(qis.backtest_model_portfolio(
            prices=prices,
            weights=strategy_weights,
            rebalancing_freq=rebalancing_freq,
            rebalancing_costs=rebalancing_costs,
            instruments_carry=instruments_carry,
            ticker=label))

    # the equal-weight-all benchmark, appended last
    portfolio_datas.append(qis.backtest_model_portfolio(
        prices=prices,
        weights=benchmark_weights,
        rebalancing_freq=rebalancing_freq,
        rebalancing_costs=rebalancing_costs,
        instruments_carry=instruments_carry,
        ticker=benchmark_ticker))

    return qis.MultiPortfolioData(portfolio_datas=portfolio_datas,
                                  benchmark_prices=portfolio_datas[-1].get_portfolio_nav().to_frame())


def compute_alpha_rank_analysis_table(multi_portfolio_data: qis.MultiPortfolioData,
                                      time_period: qis.TimePeriod = None,
                                      perf_params: qis.PerfParams = None,
                                      ) -> pd.DataFrame:
    """Tabulate QIS performance and annualized turnover for each portfolio leg.

    Performance uses each leg's full supplied NAV history. Only turnover is
    filtered by time_period, so specifying that argument can mix measurement
    windows. Supply consistently selected histories when comparing columns.

    Turnover comes from each PortfolioData.get_turnover() with aggregation and
    no rolling sum. For the rank profiler's standard QIS backtests this is
    two-sided traded turnover, including buys and sells. Its sum is divided by
    elapsed calendar days between the first and last turnover dates / 365.25.
    A nonempty series spanning zero days produces NaN; an empty selected series
    is not handled and raises IndexError.

    Args:
        multi_portfolio_data: Profiled portfolio legs, normally including the final
            equal-weight benchmark. Each leg supplies its NAV and turnover convention.
        time_period: Optional turnover-only measurement window. It does not filter
            NAVs before performance statistics are computed.
        perf_params: QIS performance configuration. None uses PerfParams(freq='ME').
            The output always selects the zero-rate 'Sharpe (rf=0)' column.

    Returns:
        DataFrame indexed by leg ticker with 'Return p.a.', 'Vol', 'Sharpe',
        'Max DD' and 'Turnover p.a.' columns. Return, volatility and drawdown are
        fractional values, Sharpe is dimensionless, and turnover is a fraction
        per year under the leg's turnover convention.

    Raises:
        IndexError: If a leg has no turnover observations in the selected window.
    """
    if perf_params is None:
        perf_params = qis.PerfParams(freq='ME')

    navs = pd.concat([portfolio_data.get_portfolio_nav()
                      for portfolio_data in multi_portfolio_data.portfolio_datas],
                     axis=1, sort=True)
    tickers = [portfolio_data.ticker for portfolio_data in multi_portfolio_data.portfolio_datas]
    navs.columns = tickers
    perf = qis.compute_ra_perf_table(prices=navs, perf_params=perf_params)

    rows = []
    for portfolio_data, ticker in zip(multi_portfolio_data.portfolio_datas, tickers):
        turnover = portfolio_data.get_turnover(is_agg=True, roll_period=None, time_period=time_period)
        turnover = turnover.iloc[:, 0] if isinstance(turnover, pd.DataFrame) else turnover
        years = (turnover.index[-1] - turnover.index[0]).days / 365.25
        annualised_turnover = float(turnover.sum()) / years if years > 0.0 else float('nan')
        rows.append({
            'Ticker': ticker,
            'Return p.a.': perf.loc[ticker, 'P.a. return'],
            'Vol': perf.loc[ticker, 'Vol'],
            'Sharpe': perf.loc[ticker, 'Sharpe (rf=0)'],
            'Max DD': perf.loc[ticker, 'Max DD'],
            'Turnover p.a.': annualised_turnover,
        })
    return pd.DataFrame(rows).set_index('Ticker')


def generate_alpha_profile_report(multi_portfolio_data: qis.MultiPortfolioData,
                                  time_period: qis.TimePeriod = None,
                                  perf_params: qis.PerfParams = None,
                                  regime_benchmark: Optional[str] = None,
                                  group_data: Optional[pd.Series] = None,
                                  backtest_name: str = 'Alpha Signal Profile',
                                  file_name: str = 'alpha_profile_report',
                                  local_path: Optional[str] = None,
                                  add_current_date: bool = True,
                                  ) -> List:
    """Render QIS factsheet figures and save them as a multipage PDF.

    Reporting is separate from signal construction and backtesting. The reporting
    window is forwarded to the QIS factsheet; it does not rerun the strategies.
    Use an explicit output directory for reproducible batch generation.

    Args:
        multi_portfolio_data: Profiled legs, normally returned by
            backtest_alpha_rank_portfolio(), with the equal-weight reference last.
        time_period: Reporting window forwarded to QIS; None leaves it unspecified.
        perf_params: Performance configuration; None uses PerfParams(freq='ME').
        regime_benchmark: Benchmark series name for regime classification; None
            uses the final portfolio leg's ticker.
        group_data: Ticker-to-group labels forwarded to the QIS factsheet.
        backtest_name: Title shown on the report.
        file_name: Output stem; QIS appends the date when requested and then '.pdf'.
        local_path: Output directory. Explicit None is passed to QIS and writes
            relative to the current working directory.
        add_current_date: Append the generation date to the file stem.

    Returns:
        List of matplotlib figures also saved to the PDF. The returned value
        contains figures, not the PDF path. The caller owns their lifecycle.
    """
    if perf_params is None:
        perf_params = qis.PerfParams(freq='ME')
    if regime_benchmark is None:
        regime_benchmark = multi_portfolio_data.portfolio_datas[-1].ticker

    kwargs = qis.fetch_default_report_kwargs(time_period=time_period)
    kwargs.pop('perf_params', None)  # passed explicitly below
    figs = qis.generate_multi_portfolio_factsheet(
        multi_portfolio_data=multi_portfolio_data,
        time_period=time_period,
        perf_params=perf_params,
        regime_benchmark=regime_benchmark,
        backtest_name=backtest_name,
        group_data=group_data,
        **kwargs)
    qis.save_figs_to_pdf(figs=figs,
                         file_name=file_name,
                         orientation='landscape',
                         local_path=local_path,
                         add_current_date=add_current_date)
    return figs
