"""Native QIS factsheets and a shared external report-bundle writer."""
from copy import copy
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import qis

from examples.oss_portfolio_pipeline.backtest import PortfolioBacktest
from examples.oss_portfolio_pipeline.data import external_directory
from examples.oss_portfolio_pipeline.pipeline import PipelineInputs
from examples.oss_portfolio_pipeline.reporting import performance_params, report_portfolios


def strategy_name(run: PortfolioBacktest) -> str:
    """Give comparable net strategies compact names for native factsheet legends."""
    return f'{run.settings.momentum_span}m / TE {run.settings.tracking_error:.1%}'


def comparison_runs(full: PortfolioBacktest,
                    alternatives: list[PortfolioBacktest] | None = None) -> dict:
    """Collect distinct horizon/TE configurations on a common benchmark and history."""
    runs = {strategy_name(full): full}
    for run in alternatives or []:
        if not run.navs.index.equals(full.navs.index):
            raise ValueError('Comparison strategies must share the same NAV observations')
        if not run.benchmark.nav.equals(full.benchmark.nav):
            raise ValueError('Comparison strategies must share the same net benchmark')
        if run.settings == full.settings:
            continue
        name = strategy_name(run)
        if name in runs:
            raise ValueError(f'Ambiguous comparison configuration: {name}')
        runs[name] = run
    return runs


def native_factsheet_figures(inputs: PipelineInputs, full: PortfolioBacktest,
                             runs: dict | None = None) -> dict[str, list]:
    """Generate the three native QIS report types, including their regime panels.

    Every comparison uses net portfolios. The fixed net benchmark also classifies regimes.
    QIS owns all panels and their labelled grids; optional position/VaR pages are omitted.
    """
    runs = runs if runs is not None else comparison_runs(full)
    active, benchmark = report_portfolios(inputs, full)[:2]
    active.set_ticker(strategy_name(full))
    benchmark.set_ticker('Fixed benchmark')
    benchmark_prices = benchmark.nav.to_frame()
    multi_portfolios = []
    for name, run in runs.items():
        portfolio = copy(run.executed.portfolio)
        portfolio.set_ticker(name)
        portfolio.set_group_data(inputs.universe.economic_group.copy())
        multi_portfolios.append(portfolio)
    multi_portfolios.append(benchmark)
    period = qis.TimePeriod(full.navs.index[0], full.navs.index[-1])
    kwargs = qis.fetch_default_report_kwargs(
        time_period=period, reporting_frequency=qis.ReportingFrequency.MONTHLY,
        add_rates_data=False,
        turnover_computation_type=qis.TurnoverComputationType.EXECUTED_NOTIONAL_NAV,
        override=dict(perf_params=performance_params(), freq_drawdown='D', fontsize=6,
                      figsize=(11.7, 16.5), legend_ncols=1,
                      recent_ra_perf_table_start_date=max(
                          full.navs.index[0], full.navs.index[-1] - pd.DateOffset(years=3)),
                      regime_classifier=qis.BenchmarkReturnsQuantilesRegime(freq='ME')))
    pair = qis.MultiPortfolioData([active, benchmark], benchmark_prices=benchmark_prices)
    multi = qis.MultiPortfolioData(multi_portfolios, benchmark_prices=benchmark_prices)
    return {
        'strategy_factsheet': qis.generate_strategy_factsheet(
            active, benchmark_prices, time_period=period, is_grouped=True,
            factsheet_name='OSS active strategy | net of trading costs', **kwargs),
        'strategy_benchmark_factsheet': qis.generate_strategy_benchmark_factsheet_plt(
            pair, time_period=period, is_grouped=True, add_brinson_attribution=False,
            backtest_name='OSS active vs fixed benchmark | both net', **kwargs),
        'multi_strategy_factsheet': qis.generate_multi_portfolio_factsheet(
            multi, time_period=period, regime_benchmark='Fixed benchmark',
            backtest_name='OSS strategy comparison | all net | same benchmark', **kwargs),
    }


def write_report_bundle(output_dir: Path | str, folder: str, figures: dict,
                        tables: dict, metadata: dict) -> dict[str, Path]:
    """Save QIS-rendered pages and numerical tables with hashes in a fresh external folder."""
    output = external_directory(output_dir)
    sources = {name: output / name for name in ('backtest_manifest.json',
                                               'covariance_manifest.json')}
    if not all(path.is_file() for path in sources.values()):
        raise ValueError('Save the baseline backtest and covariance before these reports')
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
    destination = output / folder / stamp
    destination.mkdir(parents=True, exist_ok=False)
    hashes, pdfs = {}, {}
    for name, table in tables.items():
        path = destination / f'{name}.csv'
        path.write_text(table.to_csv(float_format='%.17g'), encoding='utf-8', newline='')
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    for name, pages in figures.items():
        path = Path(qis.save_figs_to_pdf(pages, name, local_path=str(destination),
                                        orientation='portrait', add_current_date=False))
        pdfs[name] = path
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        for i, fig in enumerate(pages, 1):
            path = destination / f'{name}_{i:02d}.png'
            fig.savefig(path, dpi=140, facecolor='white')
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {
        'built_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_sha256': {name: hashlib.sha256(path.read_bytes()).hexdigest()
                          for name, path in sources.items()},
        'file_sha256': hashes, 'pages': {name: len(pages) for name, pages in figures.items()},
        'versions': {name: version(name) for name in ('qis', 'factormodels', 'factorlasso',
                                                    'optimalportfolios')},
        **metadata,
    }
    (destination / 'manifest.json').write_text(
        json.dumps(manifest, indent=2, allow_nan=False), encoding='utf-8')
    return pdfs


def save_native_factsheets(inputs: PipelineInputs, full: PortfolioBacktest,
                           output_dir: Path | str,
                           alternatives: list[PortfolioBacktest] | None = None) -> dict:
    """Export strategy, strategy-benchmark and multi-strategy QIS factsheets."""
    runs = comparison_runs(full, alternatives)
    figures = {}
    existing_figures = set(plt.get_fignums())
    with plt.rc_context({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42}):
        try:
            figures = native_factsheet_figures(inputs, full, runs)
            navs = pd.concat({name: run.executed.portfolio.nav for name, run in runs.items()},
                             axis=1)
            navs['Fixed benchmark'] = full.benchmark.nav
            tables = {'net_navs': navs,
                      'performance': qis.compute_ra_perf_table(navs, performance_params())}
            return write_report_bundle(output_dir, 'factsheets', figures, tables, {
                'strategies': {name: asdict(run.settings) for name, run in runs.items()},
                'performance': 'ME simple-return risk; compound rf=0 Sharpe; daily drawdowns',
                'regimes': 'descriptive full-sample ME benchmark quantiles; 16% and 84%',
                'turnover': 'QIS executed-notional / NAV; native factsheets include funding',
                'brinson': 'separate gross/net report; no duplicate factsheet Brinson page',
            })
        finally:
            for number in set(plt.get_fignums()) - existing_figures:
                plt.close(number)
