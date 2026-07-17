"""
core alpha profiler: rank-based long-only backtest for any pre-computed alpha score panel.

This is THE backtester/profiler for alphas in this package. It takes an alpha score panel (or a dict
of named panels) and evaluates it by holding the top-quantile of assets, equal-weighted, against an
equal-weight-all benchmark. It computes NO signals itself -- signal construction lives in the signal
modules, and the per-signal profilers in signal_profilers compute a panel and call in here.

    backtest_alpha_rank_portfolio  -- backtest one or several alpha panels (the core entry point)
    compute_top_quantile_equal_weights -- the rank-and-select weighting rule
    compute_alpha_rank_analysis_table  -- performance + annualised turnover per leg
    generate_alpha_profile_report      -- multi-strategy factsheet PDF from the profiled legs
"""
import numpy as np
import pandas as pd
import qis as qis
from typing import Dict, List, Optional, Union


def compute_top_quantile_equal_weights(alpha_scores: pd.DataFrame,
                                       prices: pd.DataFrame,
                                       quantile: float = 1.0 / 3.0,
                                       ) -> pd.DataFrame:
    """long-only equal weights on the top-quantile assets by alpha score, per date.

    At each date, rank the assets that have BOTH a finite alpha score and a valid price (available
    universe), keep the best ceil(quantile * n_available), and equal-weight them; all other assets
    get zero. This is a pure rank-and-select rule -- no covariance, no optimisation -- so it isolates
    the selection power of the alpha, independent of any sizing model.

    Args:
        alpha_scores: T x N panel of alpha scores (higher = better). Any alpha; not computed here.
        prices: T x N price panel, used only to mask assets to those actually trading on each date.
        quantile: top fraction to hold, in (0, 1]. 1/3 keeps the best third. 1.0 keeps all
            available assets (i.e. reduces to equal-weight-all).

    Returns:
        pd.DataFrame: T x N long-only weights summing to 1 across the held basket on each row
        (0 on rows with no available asset).

    Raises:
        ValueError: if quantile is not in (0, 1] or the panels do not share columns.
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
    """backtest one or several long-only top-quantile alpha baskets against equal-weight-all.

    Generic: pass any pre-computed alpha score panel (momentum, carry, low-beta, a blend, ...), or a
    dict mapping a label to each such panel to compare several alphas in one run. Each strategy holds
    the top quantile of assets by that alpha's score, equal-weighted, rebalanced on
    rebalancing_freq; a single equal-weight-all benchmark uses the same schedule. Every leg goes
    through qis.backtest_model_portfolio and the set is returned as a MultiPortfolioData for
    the standard strategy-vs-benchmark factsheet.

    This does NOT compute alphas and takes no signal enum -- alpha construction lives in the signal
    modules; this only evaluates the ranking they produce.

    Args:
        prices: T x N price panel.
        alpha_scores: either a single T x N score panel (higher = better), or a dict
            {label: score_panel} to backtest several alphas jointly. Labels name the legs.
        quantile: top fraction held, in (0, 1]. Default 1/3.
        rebalancing_freq: rebalance schedule (e.g. 'QE', 'ME').
        time_period: optional window to restrict the backtest.
        rebalancing_costs: optional per-asset cost, in bp, passed to the backtester.
        instruments_carry: optional carry panel for carry-aware reporting in the factsheet.
        strategy_ticker: leg name when alpha_scores is a single panel (ignored for a dict, whose
            keys are the leg names).
        benchmark_ticker: name for the equal-weight benchmark leg.

    Returns:
        qis.MultiPortfolioData: [*alpha legs, equal-weight benchmark]. The benchmark is always
        last, so portfolio_datas[-1] is the equal-weight reference.
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
    """performance + annualised turnover, one row per leg of an alpha-rank backtest.

    Pulls the standard risk-adjusted performance columns from qis.compute_ra_perf_table and
    appends the annualised (two-sided, buys+sells) turnover for each leg -- the metric the perf table
    does not carry, and the one that matters for a rank strategy where the basket churns as the
    ranking moves.

    Args:
        multi_portfolio_data: the output of backtest_alpha_rank_portfolio (legs + benchmark).
        time_period: window over which to measure both performance and turnover.
        perf_params: performance parameters; defaults to monthly.

    Returns:
        pd.DataFrame: indexed by leg ticker, with return / vol / Sharpe / max-dd / annualised turnover.
    """
    if perf_params is None:
        perf_params = qis.PerfParams(freq='ME')

    navs = pd.concat([portfolio_data.get_portfolio_nav()
                      for portfolio_data in multi_portfolio_data.portfolio_datas], axis=1)
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
    """generate the multi-strategy factsheet for profiled alpha legs and save it to PDF.

    Takes the MultiPortfolioData produced by ``backtest_alpha_rank_portfolio`` (every signal leg plus
    the equal-weight benchmark) and renders the multi-portfolio factsheet, then writes it to
    ``{local_path}/{file_name}.pdf``. Kept separate from the backtester so the backtest stays a pure
    data producer; call this when a report is wanted.

    Parameters
    ----------
    multi_portfolio_data : qis.MultiPortfolioData
        the profiled legs, as returned by backtest_alpha_rank_portfolio.
    time_period : qis.TimePeriod, optional
        reporting window; defaults to the full sample.
    perf_params : qis.PerfParams, optional
        performance parameters; defaults to monthly-frequency stats.
    regime_benchmark : str, optional
        ticker of the leg to use as the regime-classification benchmark; defaults to the last leg
        (the equal-weight benchmark appended by the profiler).
    group_data : pd.Series, optional
        ticker -> group labels for grouped exposures.
    backtest_name : str
        title shown on the factsheet.
    file_name : str
        output file stem (``.pdf`` appended).
    local_path : str, optional
        output directory; if None the qis default output path is used.
    add_current_date : bool
        append the run date to the file name.

    Returns
    -------
    List[matplotlib.figure.Figure]
        the generated figures (also written to disk).
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
