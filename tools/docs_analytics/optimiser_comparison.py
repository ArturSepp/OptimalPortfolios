"""Compare three covariance-only objectives on the fixed offline documentation baseline.

This deliberately narrower exhibit does not reproduce the legacy six-objective live-data
report. Mean-dependent and mixture objectives require different estimation contracts.
OptimalPortfolios constructs targets; qis owns the data, holdings, statistics and plots.
"""

import argparse
from copy import deepcopy
from itertools import combinations
from pathlib import Path

from tools.docs_analytics import portfolio_reports as reports
from tools.docs_analytics.validate import check_result


NAME = 'optimiser_comparison'
OBJECTIVES = {
    'MIN_VARIANCE': 'Minimum variance',
    'MAX_DIVERSIFICATION': 'Max diversification',
    'EQUAL_RISK_CONTRIBUTION': 'Equal risk budgets',
}
COLORS = ['#40856b', '#255b90', '#9d688f']


def configuration() -> dict:
    """Freeze the three objectives, native backends and common estimation/execution inputs."""
    config = reports.configuration()
    config['parameters']['objectives'] = list(OBJECTIVES)
    config['parameters']['risk_budget'] = 'Equal positive budgets (1/6); only risk budgeting'
    config['parameters']['factorize_covar'] = False
    config['conventions']['sharpe_convention'] = (
        'qis SharpeConvention.PA: compounded annual return / weekly annualized volatility; '
        'zero risk-free rate. Every displayed portfolio is net of trading costs.')
    config['solver']['backends'] = {}
    for objective in OBJECTIVES:
        case = reports.case_configuration(reports.configuration(), objective)
        config['solver']['backends'].update(case['solver']['backends'])
        config['dependencies'].update(case['dependencies'])
    config['tables'] += ['cost_summary', 'risk_budgets']
    config['input_tables'].append('risk_budgets')
    config['checks'] = [
        'same_inputs', 'same_covariances', 'same_decision_schedule', 'same_report_sample',
        'all_case_checks', 'declared_case_configuration', 'complete_solver_grid',
        'distinct_objective_weights', 'finite_outputs', 'cost_summary_identity',
        'positive_definite_covariances', 'risk_budget_floor_unused',
    ]
    return config


def build_analytics(config: dict) -> dict:
    """Run each declared objective through the shared checked portfolio-report calculation.

    Args:
        config: Exact fixed optimizer-comparison configuration from the registry.

    Returns:
        Shared inputs, per-objective tables, numerical checks and actual solver diagnostics.

    Raises:
        ValueError: If inputs, configurations, solver evidence or comparison checks disagree.
    """
    import numpy as np
    import pandas as pd

    if config != configuration():
        raise ValueError(
            'Unsupported optimizer-comparison configuration; update the baseline explicitly')
    base = reports.configuration()
    cases = {label: reports.build_analytics(base, objective=objective)
             for objective, label in OBJECTIVES.items()}
    first_case = next(iter(cases.values()))
    first = first_case['tables']
    tables = {name: first[name].copy()
              for name in ('prices', 'benchmark_prices', 'decision_schedule')}
    tables['risk_budgets'] = pd.DataFrame(
        {'target_fraction': 1 / len(first['prices'].columns)},
        index=first['prices'].columns).rename_axis('asset')
    for name in ('target_weights', 'realized_weights', 'units',
                 'risk_contributions', 'trading'):
        tables[name] = pd.concat(
            {label: case['tables'][name] for label, case in cases.items()}, names=['objective'])
    tables['last_covariance'] = first['last_covariance'].copy()
    for name in ('navs', 'drawdowns'):
        tables[name] = pd.concat(
            {label: case['tables'][name][label + ' (net)'] for label, case in cases.items()},
            axis=1)
    tables['performance'] = pd.DataFrame.from_dict({
        label: case['tables']['performance'].loc[label + ' (net)']
        for label, case in cases.items()}, orient='index').rename_axis('objective')
    tables['cost_summary'] = tables['trading'].groupby(level='objective', sort=False).sum().rename(
        columns={'gross_turnover_fraction': 'summed_gross_turnover',
                 'cost_fraction_of_same_day_nav': 'summed_cost_fraction'})
    diagnostics = {'solves': [], 'cases': {}}
    for objective, label in OBJECTIVES.items():
        case = cases[label]
        diagnostics['cases'][label] = {
            'configuration': case['configuration'], 'checks': case['checks']}
        for record in case['diagnostics']['solves']:
            diagnostics['solves'].append({
                **deepcopy(record), 'objective': objective, 'decision_date': record['context'],
                'context': f"objective={objective}; decision={record['context']}"})
    pairs = list(combinations(cases.values(), 2))
    checks = {
        'same_inputs': all(case['tables'][name].equals(first[name])
                           for case in cases.values() for name in ('prices', 'benchmark_prices')),
        'same_covariances': all(
            list(case['covars']) == list(first_case['covars'])
            and all(matrix.equals(first_case['covars'][date])
                    for date, matrix in case['covars'].items()) for case in cases.values()),
        'same_decision_schedule': all(
            case['tables']['decision_schedule'].equals(first['decision_schedule'])
            for case in cases.values()),
        'same_report_sample': all(
            case['tables']['navs'].index.equals(first['navs'].index) for case in cases.values()),
        'all_case_checks': all(
            set(case['checks']) == set(base['checks'])
            and all(value is True for value in case['checks'].values()) for case in cases.values()),
        'declared_case_configuration': all(
            cases[label]['configuration'] == reports.case_configuration(base, objective)
            for objective, label in OBJECTIVES.items()),
        'complete_solver_grid': (
            len(diagnostics['solves']) == len(OBJECTIVES) * len(first['target_weights'])
            and {(item['objective'], item['decision_date']) for item in diagnostics['solves']} == {
                (objective, date.strftime('%Y-%m-%d'))
                for objective in OBJECTIVES for date in first['target_weights'].index}),
        'distinct_objective_weights': all(
            not np.allclose(a['tables']['target_weights'], b['tables']['target_weights'])
            for a, b in pairs),
        'finite_outputs': all(
            np.isfinite(table.to_numpy()).all() for name, table in tables.items()
            if name != 'decision_schedule'),
        'cost_summary_identity': all(np.isclose(
            tables['cost_summary'].loc[label, 'summed_cost_fraction'],
            (case['portfolio'].realized_costs.sum(axis=1) / case['portfolio'].nav).sum(),
            rtol=1e-12, atol=1e-14) for label, case in cases.items()),
        'positive_definite_covariances': all(
            (np.linalg.eigvalsh(matrix) > 0).all() for matrix in first_case['covars'].values()),
        'risk_budget_floor_unused': all(
            (np.diag(matrix) >= 1e-6).all() for matrix in first_case['covars'].values()),
    }
    result = {'configuration': deepcopy(config),
              'checks': {name: bool(value) for name, value in checks.items()},
              'diagnostics': diagnostics}
    check_result(result, config)
    return {**result, 'tables': tables, 'cases': cases}


def _figure(tables):
    """Show net growth and full-period costs with a common objective palette and no ranking."""
    import matplotlib.pyplot as plt
    import qis

    navs = tables['navs']
    figure, axes = plt.subplots(2, 1, figsize=(11, 7.6))
    figure.subplots_adjust(top=0.78, bottom=0.18, left=0.11, right=0.97, hspace=0.64)
    figure.suptitle('Covariance-only objectives | synthetic example', fontsize=18, y=0.97)
    period = f'{navs.index[0]:%d %b %Y} - {navs.index[-1]:%d %b %Y}'
    figure.text(0.5, 0.915, period + '\nSame covariance, assets, constraints and trade dates',
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
        x_rotation=0, yvar_format='{:,.0f}', add_bar_values=True, legend_loc=None, **common)
    figure.text(0.11, 0.035,
                'OptimalPortfolios construction | qis backtesting and analytics\n'
                '35% asset cap; equal risk budgets need not yield equal risk contributions.\n'
                '10 bp costs include entry. Cost sums are not compounded performance drag.',
                fontsize=10)
    return figure


def produce(spec: dict) -> dict:
    """Return the registered comparison preview, shared inputs, tables and solver evidence.

    Args:
        spec: The optimiser_comparison producer entry from the analytics registry.

    Returns:
        Executor-compatible figure, tables, configuration, checks and diagnostics.
    """
    analytical = build_analytics(spec['configuration'])
    return {key: analytical[key] for key in
            ('configuration', 'checks', 'diagnostics', 'tables')} | {
                'figures': {'multi_optimisers_backtest.PNG': _figure(analytical['tables'])}}


def main(argv: list[str] | None = None) -> int:
    """Generate this non-publishable family through the common offline preview writer."""
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
