"""Compare fixed EWMA spans on the shared offline portfolio-report teaching baseline.

OptimalPortfolios owns the rolling construction; qis supplies the frozen synthetic data,
holdings simulation, statistics and plotting. The five spans are fixed before the comparison.
This is a sensitivity exhibit, not parameter selection or historical performance evidence.
"""

import argparse
from copy import deepcopy
from itertools import combinations
from pathlib import Path

from tools.docs_analytics import portfolio_reports as reports
from tools.docs_analytics.validate import check_result


NAME = 'span_sensitivity'
SPANS = (5, 13, 26, 52, 104)
COLORS = ['#cc8537', '#40856b', '#9d688f', '#255b90', '#777777']
NET_PORTFOLIO = 'Max diversification (net)'


def configuration() -> dict:
    """Freeze the legacy weekly span grid while sharing the portfolio-report conventions."""
    config = reports.configuration()
    config['parameters'].pop('span')
    config['parameters']['spans'] = list(SPANS)
    config['conventions']['estimation_frequency'] = (
        'Weekly Wednesday log returns; each span controls trailing demeaning and covariance '
        'smoothing. Span is an EWMA parameter, not a half-life or a finite window.')
    config['conventions']['sharpe_convention'] = (
        'qis SharpeConvention.PA: compounded annual return / weekly annualized volatility; '
        'zero risk-free rate. Every displayed portfolio is net of trading costs.')
    config['tables'].append('cost_summary')
    config['checks'] = [
        'same_inputs', 'same_decision_schedule', 'same_report_sample', 'all_span_checks',
        'declared_span_configuration', 'complete_solver_grid', 'distinct_span_covariances',
        'distinct_span_weights', 'finite_outputs',
        'cost_summary_identity',
    ]
    return config


def build_analytics(config: dict) -> dict:
    """Run the five declared spans through the existing checked portfolio-report calculation.

    Args:
        config: Exact fixed span-sensitivity configuration from the registry.

    Returns:
        Shared inputs, per-span results, comparison tables and actual solver diagnostics.

    Raises:
        ValueError: If configuration, diagnostics, common inputs or numerical checks are invalid.
    """
    import numpy as np
    import pandas as pd

    if config != configuration():
        raise ValueError(
            'Unsupported span-sensitivity configuration; update the baseline explicitly')
    base = reports.configuration()
    cases = {f'{span} weeks': reports.build_analytics(base, span=span)
             for span in config['parameters']['spans']}
    first = next(iter(cases.values()))['tables']
    tables = {name: first[name].copy()
              for name in ('prices', 'benchmark_prices', 'decision_schedule')}
    for name in ('target_weights', 'realized_weights', 'units', 'last_covariance',
                 'risk_contributions', 'trading'):
        tables[name] = pd.concat(
            {label: case['tables'][name] for label, case in cases.items()}, names=['span'])
    for name in ('navs', 'drawdowns'):
        tables[name] = pd.concat(
            {label: case['tables'][name][NET_PORTFOLIO] for label, case in cases.items()}, axis=1)
    tables['performance'] = pd.DataFrame.from_dict({
        label: case['tables']['performance'].loc[NET_PORTFOLIO]
        for label, case in cases.items()}, orient='index').rename_axis('span')
    tables['cost_summary'] = tables['trading'].groupby(level='span', sort=False).sum().rename(
        columns={'gross_turnover_fraction': 'summed_gross_turnover',
                 'cost_fraction_of_same_day_nav': 'summed_cost_fraction'})
    diagnostics = {'solves': [], 'cases': {}}
    for span, (label, case) in zip(config['parameters']['spans'], cases.items()):
        diagnostics['cases'][label] = {
            'configuration': case['configuration'], 'checks': case['checks']}
        for record in case['diagnostics']['solves']:
            diagnostics['solves'].append({
                **deepcopy(record), 'span': span, 'decision_date': record['context'],
                'context': f"span={span}; decision={record['context']}"})
    pairs = list(combinations(cases.values(), 2))
    checks = {
        'same_inputs': all(case['tables'][name].equals(first[name])
                           for case in cases.values() for name in ('prices', 'benchmark_prices')),
        'same_decision_schedule': all(
            case['tables']['decision_schedule'].equals(first['decision_schedule'])
            for case in cases.values()),
        'same_report_sample': all(
            case['tables']['navs'].index.equals(first['navs'].index) for case in cases.values()),
        'all_span_checks': all(
            set(case['checks']) == set(base['checks'])
            and all(value is True for value in case['checks'].values()) for case in cases.values()),
        'declared_span_configuration': all(
            case['configuration'] == {**base, 'parameters': {**base['parameters'], 'span': span}}
            for span, case in zip(config['parameters']['spans'], cases.values())),
        'complete_solver_grid': (
            len(diagnostics['solves']) == sum(
                len(case['tables']['target_weights']) for case in cases.values())
            and {(item['span'], item['decision_date']) for item in diagnostics['solves']} == {
                (span, date.strftime('%Y-%m-%d'))
                for span, case in zip(config['parameters']['spans'], cases.values())
                for date in case['tables']['target_weights'].index}),
        'distinct_span_covariances': all(
            not np.allclose(a['tables']['last_covariance'], b['tables']['last_covariance'])
            for a, b in pairs),
        'distinct_span_weights': all(
            not np.allclose(a['tables']['target_weights'], b['tables']['target_weights'])
            for a, b in pairs),
        'finite_outputs': all(
            np.isfinite(table.to_numpy()).all() for name, table in tables.items()
            if name != 'decision_schedule'),
        'cost_summary_identity': all(np.isclose(
            tables['cost_summary'].loc[label, 'summed_cost_fraction'],
            (case['portfolio'].realized_costs.sum(axis=1) / case['portfolio'].nav).sum(),
            rtol=1e-12, atol=1e-14) for label, case in cases.items()),
    }
    result = {'configuration': deepcopy(config),
              'checks': {name: bool(value) for name, value in checks.items()},
              'diagnostics': diagnostics}
    check_result(result, config)
    return {**result, 'tables': tables, 'cases': cases}


def _figure(tables):
    """Show net growth and full-period trading costs with one consistent span palette."""
    import matplotlib.pyplot as plt
    import qis

    navs = tables['navs']
    figure, axes = plt.subplots(2, 1, figsize=(11, 7.6))
    figure.subplots_adjust(top=0.78, bottom=0.18, left=0.11, right=0.97, hspace=0.64)
    figure.suptitle('Covariance span sensitivity | synthetic example', fontsize=18, y=0.97)
    period = f'{navs.index[0]:%d %b %Y} - {navs.index[-1]:%d %b %Y}'
    figure.text(0.5, 0.915, period + '\nSame assets, constraints, costs and implementation dates',
                ha='center', va='top', fontsize=12)
    common = {'fontsize': 12, 'legend_stats': qis.LegendStats.NONE}
    qis.plot_time_series(
        navs, ax=axes[0], title='Net growth of 100', ylabel='Index level', colors=COLORS,
        var_format='{:,.0f}', date_format='%Y', x_rotation=0, linewidth=1.3,
        legend_loc='upper center', bbox_to_anchor=(0.5, -0.16), ncols=3, **common)
    costs = tables['cost_summary']['summed_cost_fraction'] * 10000
    qis.plot_bars(
        costs, ax=axes[1], title='Trading costs over the common report period',
        ylabel='Sum of daily cost / NAV (bp)', colors=COLORS, stacked=False, is_sns=False,
        x_rotation=0, yvar_format='{:,.0f}', add_bar_values=True, legend_loc=None,
        xlabel='EWMA span (weekly observations)', **common)
    figure.text(0.11, 0.035,
                'OptimalPortfolios construction | qis backtesting and analytics\n'
                '10 bp on traded notional; costs include entry. Spans are not half-lives.\n'
                'Cost sums are descriptive, not compounded performance drag or a selection rule.',
                fontsize=10)
    return figure


def produce(spec: dict) -> dict:
    """Return the registered span figure, shared inputs, results and solver evidence.

    Args:
        spec: The span_sensitivity producer entry from the analytics registry.

    Returns:
        Executor-compatible figure, tables, configuration, checks and diagnostics.
    """
    analytical = build_analytics(spec['configuration'])
    figure = _figure(analytical['tables'])
    return {key: analytical[key] for key in
            ('configuration', 'checks', 'diagnostics', 'tables')} | {
                'figures': {'max_diversification_span.PNG': figure}}


def main(argv: list[str] | None = None) -> int:
    """Generate the non-publishable span family through the shared offline preview writer."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        print(reports.generate_preview(args.output_root, family_name=NAME))
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(2, f'{error}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
