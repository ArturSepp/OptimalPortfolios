"""QIS performance, allocation, trading and tracking-error reports for the public example."""
from copy import copy
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import qis

from examples.oss_portfolio_pipeline.backtest import PortfolioBacktest
from examples.oss_portfolio_pipeline.data import external_directory
from examples.oss_portfolio_pipeline.pipeline import PipelineInputs


NET_NAMES = ('Active (net)', 'Benchmark (net)')
LINE_COLORS = ['#126782', '#c76d29', '#7faebc', '#dfb58f']
GROUP_COLORS = ['#126782', '#639fab', '#c76d29', '#d5b56c', '#8c969b', '#b34b5d']
PERF_COLUMNS = (qis.PerfStat.TOTAL_RETURN, qis.PerfStat.PA_RETURN, qis.PerfStat.VOL,
                qis.PerfStat.SHARPE_RF0, qis.PerfStat.MAX_DD)


def performance_params() -> qis.PerfParams:
    """Use monthly simple-return risk, native daily drawdowns and zero-rate compound Sharpe."""
    return qis.PerfParams(freq='ME', freq_drawdown='D', freq_skewness='ME',
                          return_type=qis.ReturnTypes.RELATIVE, rates_data=None)


def report_portfolios(inputs: PipelineInputs, result: PortfolioBacktest) -> list[qis.PortfolioData]:
    """Attach display metadata to shallow copies without modifying the backtest objects."""
    portfolios = []
    for name, source in zip(result.navs.columns, (
            result.executed.portfolio, result.benchmark,
            result.gross_portfolio, result.gross_benchmark), strict=True):
        portfolio = copy(source)
        portfolio.set_ticker(name)
        portfolio.set_group_data(inputs.universe.economic_group.copy())
        portfolios.append(portfolio)
    return portfolios


def calendar_month_labels(data: pd.Series | pd.DataFrame) -> pd.Series | pd.DataFrame:
    """Label each observed ME decision by its calendar month without shifting information."""
    labelled = data.copy()
    labelled.index = labelled.index.to_period('M').to_timestamp('M')
    if not labelled.index.is_unique:
        raise ValueError('Tracking-error reporting requires at most one observation per month')
    return labelled


def prepare_report_tables(inputs: PipelineInputs, result: PortfolioBacktest) -> dict:
    """Obtain QIS analytics and retain numerical tables underlying every report panel.

    Performance uses native NAV endpoints; monthly trading tables stop at the last completed
    model month. Turnover excludes initial funding at any execution lag, while costs include
    it. The QIS realised-risk warm-up remains missing, never zero or backfilled.
    """
    portfolios = report_portfolios(inputs, result)
    tables = {'performance': qis.compute_ra_perf_table(result.navs, performance_params())}
    completed_month = inputs.benchmark_targets.index[-1].to_period('M').to_timestamp('M')
    daily_turnover, monthly_costs, rolling_turnover = {}, {}, {}
    for name, portfolio in zip(NET_NAMES, portfolios[:2], strict=True):
        key = 'active' if name == NET_NAMES[0] else 'benchmark'
        allocation = portfolio.get_weights(is_grouped=True, freq=None)
        # QIS reports instrument weights. Complete the NAV accounting identity with
        # currency cash; BIL is already an ETF position in the Cash sleeve.
        allocation['Funding cash'] = 1. - portfolio.weights.sum(axis=1)
        tables[f'allocation_{key}'] = allocation
        turnover = portfolio.get_turnover(
            is_agg=True, roll_period=None,
            turnover_computation_type=qis.TurnoverComputationType.EXECUTED_NOTIONAL_NAV)
        funded = portfolio.units.abs().sum(axis=1).gt(0.)
        if not funded.any():
            raise ValueError('Cannot report trading without an invested portfolio')
        first_execution = funded[funded].index[0]
        turnover.loc[:first_execution] = np.nan
        daily_turnover[name] = turnover
        # QIS's monthly rolling sum follows its canonical executed-notional convention.
        # Exclude the funding month so lagged opening trades cannot enter rolling totals.
        funding_month = first_execution.to_period('M').to_timestamp('M')
        rolling = portfolio.get_turnover(
            is_agg=True, freq='ME', roll_period=12,
            turnover_computation_type=qis.TurnoverComputationType.EXECUTED_NOTIONAL_NAV)
        rolling.loc[rolling.index < funding_month + pd.offsets.MonthEnd(12)] = np.nan
        rolling_turnover[name] = rolling
        monthly_costs[name] = portfolio.get_costs(is_agg=True, freq='ME', roll_period=None)
    tables['turnover_daily'] = pd.DataFrame(daily_turnover)
    tables['turnover_monthly'] = tables['turnover_daily'].resample('ME').sum(min_count=1)
    tables['turnover_12m'] = pd.DataFrame(rolling_turnover)
    tables['costs_monthly'] = pd.DataFrame(monthly_costs)
    for key in ('turnover_monthly', 'turnover_12m', 'costs_monthly'):
        tables[key] = tables[key].loc[:completed_month]

    snapshot = pd.DataFrame({
        'Active target': result.executed.targets.iloc[-1],
        'Fixed target': inputs.benchmark_weights,
        'Active held': portfolios[0].weights.iloc[-1],
        'Benchmark held': portfolios[1].weights.iloc[-1],
    })
    snapshot.loc['Funding cash'] = [0., 0.,
                                    tables['allocation_active']['Funding cash'].iloc[-1],
                                    tables['allocation_benchmark']['Funding cash'].iloc[-1]]
    tables['allocation_snapshot'] = snapshot
    target_te = result.risk_model.compute_tre_history(
        inputs.benchmark_weights, result.executed.targets)
    held_te = result.risk_model.compute_tre_history(
        portfolios[1].weights, portfolios[0].weights)
    tables['tracking_error'] = pd.concat({
        'Target vs fixed benchmark': calendar_month_labels(target_te),
        'Held vs held benchmark': calendar_month_labels(held_te),
        'Realised net (36m EWMA)': calendar_month_labels(result.realised_tracking_error),
    }, axis=1)
    tables['tracking_error']['Target TE limit'] = result.settings.tracking_error
    tables['tracking_error_summary'] = result.te_ir.copy()
    return tables


def report_page(title: str, subtitle: str, footer: str, page_number: int):
    """Create an A4 page with a common header and reserved space for convention notes."""
    fig = plt.figure(figsize=(8.3, 11.7), facecolor='white')
    fig.text(.07, .965, 'OSS PORTFOLIO PIPELINE  /  QIS', fontsize=9, color='#126782')
    fig.text(.07, .932, title, fontsize=19, weight='bold', color='#17313b')
    fig.text(.07, .906, subtitle, fontsize=9, color='#52646c')
    fig.text(.07, .072, footer, fontsize=8, color='#52646c', va='top', linespacing=1.5)
    fig.text(.93, .025, str(page_number), fontsize=8, color='#52646c', ha='right')
    return fig


def draw_table(data: pd.DataFrame, ax, percent: bool = True, **kwargs) -> None:
    """Render missing values explicitly and let QIS handle table layout."""
    display = data.copy()
    if percent:
        display = qis.df_to_str(display, var_format='{:.2%}')
    display = display.replace({'nan%': 'n/a', 'nan': 'n/a'})
    qis.plot_df_table(display, ax=ax, fontsize=8, header_color='#126782',
                      left_aligned_first_col=True, **kwargs)


def create_report_figures(inputs: PipelineInputs, result: PortfolioBacktest,
                          tables: dict | None = None) -> dict[str, plt.Figure]:
    """Compose five factsheet pages exclusively from public QIS charts and analytics."""
    tables = tables if tables is not None else prepare_report_tables(inputs, result)
    portfolios = report_portfolios(inputs, result)
    net = qis.MultiPortfolioData(portfolio_datas=portfolios[:2])
    start, end = result.navs.index[[0, -1]]
    period = f'{start:%d %b %Y} - {end:%d %b %Y}  |  USD  |  36 ETFs  |  ME decisions'
    common = dict(fontsize=8, x_date_freq='YS', date_format='%Y', linewidth=1.2, x_rotation=0)
    figures = {}

    fig = report_page(
        'Performance factsheet', period,
        'QIS: monthly simple-return volatility; daily drawdowns; compound Sharpe, rf=0.\n'
        'Total and p.a. returns use native endpoints; Sharpe uses completed ME boundaries.\n'
        'Opening purchase costs reduce inception NAV; returns from that NAV omit the debit.', 1)
    figures['performance'] = fig
    grid = fig.add_gridspec(3, 1, left=.09, right=.94, bottom=.17, top=.87,
                           height_ratios=[3, 2, 1.25], hspace=.50)
    qis.plot_time_series(result.navs, ax=fig.add_subplot(grid[0]), colors=LINE_COLORS,
                         linestyles=['-', '-', '--', '--'], legend_stats=qis.LegendStats.NONE,
                         title='NAV from 100 initial capital | net and same-target gross replays',
                         var_format='{:,.0f}', **common)
    net.plot_drawdowns(ax=fig.add_subplot(grid[1]), colors=LINE_COLORS[:2],
                       title='Net portfolio drawdowns', **common)
    perf = tables['performance'][[stat.to_str() for stat in PERF_COLUMNS]].copy()
    for stat in PERF_COLUMNS:
        column = stat.to_str()
        fmt = '{:.2f}' if stat == qis.PerfStat.SHARPE_RF0 else '{:.1%}'
        perf[column] = perf[column].map(fmt.format)
    draw_table(perf, fig.add_subplot(grid[2]), percent=False, col_widths=[2.2, 1, 1, 1, 1.2, 1])
    fig.text(.09, .115, f'Both portfolios: {result.settings.transaction_cost * 10000:g} bp per '
             f'traded notional; execution lag {result.settings.implementation_lag} observations.',
             fontsize=8, color='#52646c')

    fig = report_page(
        'Allocation through time', period,
        'QIS held-unit allocations, marked daily; holdings drift between monthly trades.\n'
        'Cash sleeve = BIL. Funding cash is the separate currency balance after costs.\n'
        'ETF weights can exceed 100% by the small negative funding cash balance.', 2)
    figures['allocations'] = fig
    grid = fig.add_gridspec(3, 1, left=.09, right=.94, bottom=.15, top=.85,
                           height_ratios=[2, 2, 1.2], hspace=.65)
    for idx, key in enumerate(('active', 'benchmark')):
        stack = tables[f'allocation_{key}'].drop(columns='Funding cash')
        # QIS plot_stack maps DatetimeIndex to formatted strings, collapsing daily dates
        # when labels repeat. Numeric Matplotlib dates preserve every observed coordinate.
        stack.index = mdates.date2num(stack.index.to_pydatetime())
        ax = fig.add_subplot(grid[idx])
        qis.plot_stack(stack, ax=ax, colors=GROUP_COLORS[:5],
                       title=f'{NET_NAMES[idx]} | economic sleeves', ncols=3,
                       legend_loc='upper center', skip_y_axis=False, **common)
        ax.set_ylim(0., 1.2)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax.set_xlim(stack.index[0], stack.index[-1])
    latest = pd.concat({name: tables[f'allocation_{key}'].iloc[-1]
                        for name, key in zip(NET_NAMES, ('active', 'benchmark'), strict=True)},
                       axis=1)
    draw_table(latest, fig.add_subplot(grid[2]), col_widths=[2.5, 1.5, 1.5])

    target_date = result.executed.targets.index[-1]
    fig = report_page(
        'ETF allocation detail',
        f'Targets: {target_date:%d %b %Y}  |  Actual holdings: {end:%d %b %Y}',
        'Active target: latest accepted OP decision. Fixed target: constant benchmark policy.\n'
        'Held weights include subsequent market drift and cost debits; they are not targets.\n'
        'All 36 ETFs are retained, including zero allocations.', 3)
    figures['etf_allocations'] = fig
    snapshot = tables['allocation_snapshot'].copy()
    snapshot.index = [f'{ticker} | {inputs.universe.loc[ticker, "name"]}'
                      if ticker in inputs.universe.index else ticker for ticker in snapshot.index]
    ax = fig.add_axes((.06, .13, .89, .73))
    draw_table(snapshot, ax, col_widths=[4.1, 1.05, 1.05, 1.05, 1.25])

    monthly_end = tables['turnover_monthly'].index[-1]
    fig = report_page(
        'Turnover and trading costs',
        f'Completed monthly totals through {monthly_end:%d %b %Y}',
        'Turnover: QIS bought + sold notional / same-day NAV; initial funding excluded.\n'
        '12m turnover sums 12 completed monthly totals. It is two-way, not one-way turnover.\n'
        'Cost drag: sum of daily costs / NAV, including funding; not a compounded loss.', 4)
    figures['turnover'] = fig
    grid = fig.add_gridspec(3, 1, left=.09, right=.94, bottom=.16, top=.86, hspace=.50)
    for position, (key, title) in enumerate((
            ('turnover_monthly', 'Monthly two-way turnover | excluding initial funding'),
            ('turnover_12m', 'Trailing 12-month two-way turnover'),
            ('costs_monthly', 'Monthly trading cost drag | including initial funding'))):
        qis.plot_time_series(tables[key], ax=fig.add_subplot(grid[position]),
                             title=title, colors=LINE_COLORS[:2],
                             var_format='{:.2%}' if key == 'costs_monthly' else '{:.1%}',
                             legend_stats=qis.LegendStats.NONE, **common)

    fig = report_page(
        'Ex-ante and realised tracking error', period,
        'Ex-ante: annual log-return covariance. Realised: completed ME simple-return differences.\n'
        'Forecasts are contemporaneous at each ME; realised EWMA looks backward, not forward.\n'
        'The target risk cap does not constrain realised TE or later drifted holdings.', 5)
    figures['tracking_error'] = fig
    grid = fig.add_gridspec(2, 1, left=.09, right=.94, bottom=.39, top=.86,
                           height_ratios=[3.5, 1], hspace=.55)
    qis.plot_time_series(tables['tracking_error'], ax=fig.add_subplot(grid[0]),
                         colors=['#126782', '#639fab', '#c76d29', '#7c8286'],
                         linestyles=['-', '--', '-', ':'], var_format='{:.1%}',
                         legend_stats=qis.LegendStats.NONE,
                         title='Annualised TE | QIS RiskModel and 36-month EWMA', **common)
    summary = tables['tracking_error_summary'].copy()
    summary['TE'] = summary['TE'].map('{:.2%}'.format)
    summary['IR'] = summary['IR'].map('{:.2f}'.format)
    summary.index = ['Net active vs net benchmark', 'Gross active vs gross benchmark']
    draw_table(summary, fig.add_subplot(grid[1]), percent=False,
               col_widths=[3, 1, 1], title='Whole-sample completed monthly simple-return TE / IR')
    first_realised = tables['tracking_error']['Realised net (36m EWMA)'].first_valid_index()
    warmup = (f'First realised EWMA estimate: {first_realised:%d %b %Y}.'
              if first_realised is not None else 'Realised EWMA: insufficient history.')
    fig.text(.09, .31,
             'Target vs fixed benchmark: decision weights against constant strategic weights.\n'
             'Held vs held benchmark: actual active and benchmark weights at each model close.\n'
             'Weekend decision dates share the corresponding calendar ME label.\n'
             f'Warm-up remains missing. {warmup}\n\n'
             'Sources: Yahoo adjusted prices; FM factors; FL estimation; OP optimisation.\n'
             'QIS supplies accounting, performance, risk, turnover and all report charts.\n'
             'Illustrative study; present-day ETF selection retains survivorship bias.',
             fontsize=8, color='#52646c', va='top', linespacing=1.7)
    return figures


def save_reports(inputs: PipelineInputs, result: PortfolioBacktest,
                 output_dir: Path | str) -> Path:
    """Write a five-page PDF, per-page PNGs and source tables with a linked SHA-256 manifest."""
    output = external_directory(output_dir)
    backtest_manifest = output / 'backtest_manifest.json'
    if not backtest_manifest.is_file():
        raise ValueError('Save the portfolio backtest before the QIS reports')
    destination = external_directory(output / 'reports')
    destination.mkdir(parents=True, exist_ok=True)
    tables = prepare_report_tables(inputs, result)
    hashes = {}
    for name, table in tables.items():
        path = destination / f'{name}.csv'
        path.write_text(table.to_csv(float_format='%.17g'), encoding='utf-8', newline='')
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    # Scope styling to this export; do not change interactive users' global configuration.
    with plt.rc_context({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42}):
        figures = create_report_figures(inputs, result, tables)
        try:
            pdf = Path(qis.save_figs_to_pdf(
                figures, file_name='portfolio_report', local_path=str(destination),
                orientation='portrait', add_current_date=False))
            hashes[pdf.name] = hashlib.sha256(pdf.read_bytes()).hexdigest()
            for name, fig in figures.items():
                path = destination / f'{name}.png'
                fig.savefig(path, dpi=160, facecolor='white')
                hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        finally:
            for fig in figures.values():
                plt.close(fig)
    manifest = {
        'built_at_utc': datetime.now(timezone.utc).isoformat(),
        'backtest_manifest_sha256': hashlib.sha256(backtest_manifest.read_bytes()).hexdigest(),
        'file_sha256': hashes, 'pages': list(figures),
        'nav_start': str(result.navs.index[0].date()),
        'nav_end': str(result.navs.index[-1].date()),
        'latest_target_date': str(result.executed.targets.index[-1].date()),
        'completed_trading_month': str(tables['turnover_monthly'].index[-1].date()),
        'performance': 'native-endpoint returns; ME simple-return volatility; daily drawdowns',
        'sharpe': 'QIS SHARPE_RF0: completed-ME compound p.a. return / volatility; rf=0',
        'allocations': 'daily QIS instrument weights plus currency-cash accounting residual',
        'turnover': 'QIS executed bought + sold notional / same-day NAV; initial funding excluded',
        'costs': 'monthly sum of QIS daily costs / NAV; includes initial funding',
        'ex_ante_risk': 'QIS RiskModel; target vs fixed and actual holdings vs actual benchmark',
        'realised_risk': 'QIS completed-ME simple-return differences; 36m EWMA; warm-up retained',
        'date_alignment': 'calendar ME labels, same-month estimates; no forecasting shift',
        'versions': {name: version(name) for name in ('qis', 'matplotlib', 'pandas')},
    }
    (destination / 'report_manifest.json').write_text(
        json.dumps(manifest, indent=2, allow_nan=False), encoding='utf-8')
    return pdf
