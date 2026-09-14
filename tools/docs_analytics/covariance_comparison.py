"""Compare six covariance estimators on a fixed, known-factor synthetic teaching panel.

OptimalPortfolios assembles covariances and minimum-variance targets, factorlasso fits
regressions, and qis converts returns, backtests holdings and produces statistics/plots.
The existing simulator is unchanged. Known truth is used only for descriptive error checks.
"""

import argparse
from contextlib import contextmanager
from copy import deepcopy
from itertools import combinations
from pathlib import Path
from unittest.mock import patch

from tools.docs_analytics import portfolio_reports as reports
from tools.docs_analytics.validate import check_result


NAME = 'covariance_comparison'
CASES = {
    'EWMA': {'model': 'EWMA', 'vol_normalized': False},
    'EWMA vol norm': {'model': 'EWMA', 'vol_normalized': True},
    'Lasso': {'model': 'LASSO', 'vol_normalized': False, 'reg_lambda': 1e-6},
    'Lasso factor vol norm': {'model': 'LASSO', 'vol_normalized': True, 'reg_lambda': 1e-6},
    'Group Lasso': {'model': 'GROUP_LASSO', 'vol_normalized': False, 'reg_lambda': 1e-5},
    'Group Lasso factor vol norm': {
        'model': 'GROUP_LASSO', 'vol_normalized': True, 'reg_lambda': 1e-5},
}
COLORS = ['#255b90', '#528dc0', '#40856b', '#91b797', '#9d688f', '#c397b8']
SHORT_LABELS = ['EWMA', 'EWMA\nvol norm', 'Lasso', 'Lasso\nfactor VN',
                'Group\nLasso', 'Group Lasso\nfactor VN']


def configuration() -> dict:
    """Freeze the simulator, six estimator cases, shared clock and minimum-variance policy."""
    config = reports.configuration()
    config['fixture'] = {
        'factory': ('examples.covar_estimation.simulate_factor_returns.'
                    'simulate_factor_model_returns'),
        'source_files': ['examples/covar_estimation/simulate_factor_returns.py'],
    }
    config['parameters'] = {
        'simulation': {
            'n_factors': 4, 'n_assets': 8, 'n_periods': 783, 'factor_vol': 0.15,
            'idio_vol_range': [0.01, 0.15], 'beta_range': [-1.5, 1.5],
            'factor_corr': None, 'dt': 1 / 260, 'seed': 42,
        },
        'report_start': '2023-12-01', 'decision_end': '2025-10-01',
        'returns_freq': 'W-WED', 'rebalancing_freq': 'QE', 'span': 52, 'warmup': 52,
        'demean': True, 'max_weight': 0.35, 'initial_nav': 100.0, 'cost_rate': 0.001,
        'objective': 'MIN_VARIANCE', 'factorize_covar': False,
        'group_assignments': [0, 0, 1, 1, 2, 2, 3, 3],
        'group_penalty': 'normalized', 'l1_weight': 0.0,
        'cases': deepcopy(CASES),
    }
    config['conventions'].update({
        'data_kind': 'Known-factor Gaussian simulation, not market or replication evidence.',
        'sample_start': '2023-01-02', 'sample_end': '2025-12-31', 'seed': 42,
        'universe': [f'Asset_{i}' for i in range(1, 9)],
        'missing_data': 'Complete simulated panel; no imputation or asset filtering.',
        'observation_frequency': 'Business-day Gaussian increments interpreted as log returns.',
        'estimation_frequency': (
            'Weekly Wednesday log returns. EWMA span 52; factor regressions use the same '
            'weekly data and span. Vol normalization affects asset/factor covariance only.'),
        'warmup': 'History from 2023-01-02; at least 52 weekly observations before each fit.',
        'rebalance_frequency': (
            'Shared EWMA quarterly schedule mapped to observed Wednesdays. All estimators '
            'fit current covariances on these exact truncated dates.'),
        'sharpe_convention': (
            'qis SharpeConvention.PA with zero risk-free rate; all portfolios are net. '
            'Compounded annual return / weekly annualized volatility.'),
    })
    config['tables'] = [
        'prices', 'factor_prices', 'simulated_asset_returns', 'simulated_factor_returns',
        'simulated_residual_returns', 'true_betas', 'true_factor_covariance',
        'true_asset_covariance', 'true_idio_volatility', 'group_assignments',
        'target_weights', 'realized_weights', 'units', 'navs', 'drawdowns', 'performance',
        'covariances', 'estimated_betas', 'estimated_factor_covariances',
        'estimated_residual_variances', 'covariance_errors', 'trading', 'decision_schedule',
    ]
    config['input_tables'] = config['tables'][:10]
    config['checks'] = [
        'finite_inputs_outputs', 'same_decision_schedule', 'same_report_sample',
        'complete_allocation_grid', 'complete_factor_fit_grid', 'positive_definite_covariances',
        'distinct_covariances', 'distinct_weights', 'implementation_schedule',
        'realized_cost_identity', 'opening_trade_identity', 'error_summary_identity',
        'generative_identity', 'warmup_satisfied',
    ]
    config['solver']['backends'] = {
        'CLARABEL': {'options': 'CVXPY/CLARABEL defaults; no overrides or fallback chain'}}
    config['dependencies'].update({'cvxpy': 'cvxpy', 'clarabel': 'clarabel'})
    return config


def _fixture(config):
    """Use the existing simulator while restoring its legacy global NumPy random-state effect."""
    import numpy as np
    import pandas as pd
    import qis
    from examples.covar_estimation.simulate_factor_returns import simulate_factor_model_returns

    params = config['parameters']
    state = np.random.get_state()
    try:
        simulated = simulate_factor_model_returns(**params['simulation'])
    finally:
        np.random.set_state(state)
    prices, factors = [
        qis.returns_to_nav(simulated[name], is_log_returns=True, init_period=1, init_value=100.0)
        for name in ('asset_returns', 'factor_returns')]
    groups = pd.Series(params['group_assignments'], index=prices.columns, name='group')
    annual = 1 / params['simulation']['dt']
    tables = {
        'prices': prices, 'factor_prices': factors,
        'simulated_asset_returns': simulated['asset_returns'].copy(),
        'simulated_factor_returns': simulated['factor_returns'].copy(),
        'simulated_residual_returns': simulated['residual_returns'].copy(),
        'true_betas': simulated['betas'].copy(),
        'true_factor_covariance': annual * simulated['factor_covar'],
        'true_asset_covariance': annual * simulated['theoretical_asset_covar'],
        'true_idio_volatility': simulated['idio_vol'].rename('daily_volatility').to_frame(),
        'group_assignments': groups.to_frame(),
    }
    for name, table in tables.items():
        table.index.name = ('factor' if name in {'true_betas', 'true_factor_covariance'}
                            else 'asset' if name in {
                                'true_asset_covariance', 'true_idio_volatility',
                                'group_assignments'}
                            else 'date')
    return tables


@contextmanager
def _capture_factor_solve(context):
    """Observe native CVXPY results in this single-threaded producer and always restore solve."""
    import cvxpy as cp
    import numpy as np

    original, records = cp.Problem.solve, []

    def observe(problem, *args, **kwargs):
        """Forward the unchanged solve and retain actual status, options and constraint evidence."""
        result = original(problem, *args, **kwargs)
        residuals = []
        for index, constraint in enumerate(problem.constraints):
            violation = float(np.max(np.abs(constraint.violation())))
            residuals.append({
                'name': f'factor_constraint_{index}', 'hard': True,
                'violation': violation, 'tolerance': 1e-6, 'passed': violation <= 1e-6})
        accepted = problem.status in {'optimal', 'optimal_inaccurate'}
        records.append({
            'solver': problem.solver_stats.solver_name, 'context': context,
            'status': problem.status, 'raw_status': problem.status, 'accepted': accepted,
            'compliant': all(item['passed'] for item in residuals),
            'fallback_source': None, 'constraint_residuals': residuals,
            'constraints_not_applicable': (
                'Free regression coefficients; L1/group terms are penalties.'),
            'covariance_stabilization': {
                'factorized': False,
                'not_applicable': (
                    'Regression on observed returns; no covariance input to factorize.')},
            'options': deepcopy(kwargs), 'objective_value': float(problem.value),
        })
        if kwargs != {'verbose': False, 'solver': 'CLARABEL'}:
            raise ValueError('Unexpected factor-solver options or fallback attempt')
        return result

    with patch.object(cp.Problem, 'solve', observe):
        yield records


def _estimate_case(label, case, tables, dates, config):
    """Fit public current estimators on causal dates and preserve factor components."""
    import numpy as np
    import pandas as pd
    import qis
    import optimalportfolios as op
    import factorlasso as fl

    params = config['parameters']
    prices, factors = tables['prices'], tables['factor_prices']
    covars, components, records = {}, {}, []
    if case['model'] == 'EWMA':
        estimator = op.EwmaCovarEstimator(
            returns_freq=params['returns_freq'], span=params['span'], demean=True,
            is_apply_vol_normalised_returns=case['vol_normalized'])
        covars = {date: estimator.fit_current_covar(prices.loc[:date]) for date in dates}
    else:
        model = fl.LassoModel(
            model_type=getattr(fl.LassoModelType, case['model']),
            group_data=tables['group_assignments']['group'], reg_lambda=case['reg_lambda'],
            span=params['span'], warmup_period=params['warmup'], demean=True, nonneg=False,
            group_penalty=params['group_penalty'], l1_weight=params['l1_weight'],
            solver='CLARABEL', solver_fallbacks=None, auto_sign_constraints=False,
            auto_sign_adaptive_weights=False)
        estimator = op.FactorCovarEstimator(
            lasso_model=model, factor_returns_freq=params['returns_freq'],
            factor_covar_span=params['span'], demean=True,
            is_apply_vol_normalised_returns=case['vol_normalized'])
        asset_returns = qis.compute_asset_returns_dict(
            prices=prices, is_log_returns=True, returns_freqs=params['returns_freq'])
        for date in dates:
            inputs = {freq: panel.loc[:date] for freq, panel in asset_returns.items()}
            if any(len(panel) < params['warmup'] for panel in inputs.values()):
                raise ValueError('Insufficient factor-estimation warmup')
            context = f'estimator={label}; fit={date:%Y-%m-%d}'
            with _capture_factor_solve(context) as observed:
                data = estimator.fit_current_factor_covars(
                    risk_factor_prices=factors.loc[:date], asset_returns_dict=inputs,
                    assets=prices.columns, estimation_date=date)
            if len(observed) != 1:
                raise ValueError('Missing, duplicate or fallback factor solve')
            if not np.isfinite(model.estimation_result_.estimated_beta).all():
                raise ValueError('Factor estimation returned non-finite coefficients')
            records.extend({**record, 'stage': 'factor_fit', 'estimator': label,
                            'decision_date': date.strftime('%Y-%m-%d')} for record in observed)
            covars[date] = data.get_y_covar()
            components[date] = {
                'estimated_betas': data.y_betas.copy(),
                'estimated_factor_covariances': data.x_covar.copy(),
                'estimated_residual_variances': data.y_variances[['residual_var']].copy(),
            }
    component_tables = {}
    if components:
        for name in next(iter(components.values())):
            for component in components.values():
                component[name].index.name = (
                    'factor' if name == 'estimated_factor_covariances' else 'asset')
            component_tables[name] = pd.concat(
                {date: data[name] for date, data in components.items()}, names=['date'])
    return {'covars': covars, 'components': component_tables, 'solves': records}


def _portfolio(label, covars, prices, config):
    """Use production minimum-variance construction and qis holdings/statistics for one case."""
    import pandas as pd
    import qis
    import optimalportfolios as op

    params = config['parameters']
    constraints = op.Constraints(
        is_long_only=True, min_exposure=1.0, max_exposure=1.0,
        max_weights=pd.Series(params['max_weight'], index=prices.columns))
    solve_config = reports.case_configuration(reports.configuration(), params['objective'])
    weights, records = reports._solve(prices, covars, constraints, solve_config)
    diagnostics = reports._diagnostics(weights, covars, constraints, records)
    portfolio = qis.backtest_model_portfolio(
        prices.loc[weights.index[0]:], weights, initial_nav=params['initial_nav'],
        rebalancing_costs=params['cost_rate'],
        weight_implementation_lag=config['conventions']['implementation_lag'],
        funding_rate=None, management_fee=None, instruments_carry=None,
        is_rebalanced_at_first_date=False, ticker=label)
    nav = portfolio.nav.copy()
    perf = qis.compute_ra_perf_table(nav.to_frame(), perf_params=qis.PerfParams(
        freq=params['returns_freq'], return_type=qis.ReturnTypes.LOG,
        sharpe_convention=qis.SharpeConvention.PA))
    perf = perf[[stat.value.name for stat in (
        qis.PerfStat.PA_RETURN, qis.PerfStat.VOL, qis.PerfStat.SHARPE_RF0)]]
    turnover = portfolio.get_turnover(is_agg=True, roll_period=1)
    turnover.iloc[0] = 0.0
    trading = pd.concat([
        turnover.rename('gross_turnover_fraction'),
        portfolio.get_costs(is_agg=True, roll_period=1).rename('cost_fraction_of_same_day_nav')],
        axis=1)
    tables = {
        'target_weights': weights, 'realized_weights': portfolio.weights.copy(),
        'units': portfolio.units.copy(), 'navs': nav.to_frame(),
        'drawdowns': qis.compute_rolling_drawdowns(nav).to_frame(),
        'performance': perf, 'trading': trading,
    }
    for name, table in tables.items():
        table.index.name = 'estimator' if name == 'performance' else 'date'
    for record in diagnostics['solves']:
        record.update({
            'stage': 'allocation', 'estimator': label, 'decision_date': record['context'],
            'context': f"estimator={label}; allocation={record['context']}"})
    return {'portfolio': portfolio, 'tables': tables, 'solves': diagnostics['solves']}


def build_analytics(config: dict) -> dict:
    """Compute six estimator cases and require complete numerical and native solver evidence.

    Args:
        config: Exact fixed configuration stored in the analytics registry.

    Returns:
        Input/result tables, checks, factor/allocation diagnostics and review objects.

    Raises:
        ValueError: If configuration, estimates, solver records or numerical checks fail.
    """
    import numpy as np
    import pandas as pd
    import qis

    if config != configuration():
        raise ValueError('Unsupported covariance-comparison configuration')
    tables = _fixture(config)
    params, prices = config['parameters'], tables['prices']
    dates = list(reports._estimator(config).fit_rolling_covars(
        prices, qis.TimePeriod(params['report_start'], params['decision_end'])))
    if not dates or not pd.Index(dates).isin(prices.index).all():
        raise ValueError('Comparison needs shared observed decision dates')
    estimates = {label: _estimate_case(label, case, tables, dates, config)
                 for label, case in config['parameters']['cases'].items()}
    portfolios = {label: _portfolio(label, item['covars'], prices, config)
                  for label, item in estimates.items()}
    for name in ('target_weights', 'realized_weights', 'units', 'trading'):
        tables[name] = pd.concat(
            {label: item['tables'][name] for label, item in portfolios.items()},
            names=['estimator'])
    for name in ('navs', 'drawdowns'):
        tables[name] = pd.concat([item['tables'][name] for item in portfolios.values()], axis=1)
    tables['performance'] = pd.concat([item['tables']['performance']
                                      for item in portfolios.values()])
    covariances = {}
    for label, item in estimates.items():
        for date, matrix in item['covars'].items():
            matrix.index.name = 'asset'
        covariances[label] = pd.concat(item['covars'], names=['date'])
    tables['covariances'] = pd.concat(covariances, names=['estimator'])
    for name in ('estimated_betas', 'estimated_factor_covariances', 'estimated_residual_variances'):
        tables[name] = pd.concat(
            {label: item['components'][name] for label, item in estimates.items()
             if item['components']}, names=['estimator'])
    truth = tables['true_asset_covariance']
    error_rows = {(label, date): np.linalg.norm(matrix - truth) / np.linalg.norm(truth)
                  for label, item in estimates.items() for date, matrix in item['covars'].items()}
    tables['covariance_errors'] = pd.Series(error_rows, name='relative_frobenius_error').to_frame()
    tables['covariance_errors'].index.names = ['estimator', 'date']
    implementation = prices.index[prices.index.searchsorted(dates)
                                   + config['conventions']['implementation_lag']]
    tables['decision_schedule'] = pd.DataFrame(
        {'implementation_date': implementation}, index=pd.DatetimeIndex(dates, name='date'))
    solves = [record for item in estimates.values() for record in item['solves']]
    solves += [record for item in portfolios.values() for record in item['solves']]
    expected_dates = {date.strftime('%Y-%m-%d') for date in dates}
    pairs = list(combinations(estimates.values(), 2))
    portfolio_pairs = list(combinations(portfolios.values(), 2))
    first_nav = next(iter(portfolios.values()))['portfolio'].nav
    weekly = qis.compute_asset_returns_dict(
        prices=prices, is_log_returns=True, returns_freqs=params['returns_freq'])
    checks = {
        'finite_inputs_outputs': all(np.isfinite(table.to_numpy()).all()
                                     for name, table in tables.items()
                                     if name != 'decision_schedule'),
        'same_decision_schedule': all(
            list(item['covars']) == dates for item in estimates.values())
            and all(item['tables']['target_weights'].index.equals(pd.DatetimeIndex(dates))
                    for item in portfolios.values()),
        'same_report_sample': all(
            item['portfolio'].nav.index.equals(first_nav.index) for item in portfolios.values()),
        'complete_allocation_grid': (
            len([item for item in solves if item['stage'] == 'allocation'])
            == len(CASES) * len(dates)
            and {(item['estimator'], item['decision_date']) for item in solves
                 if item['stage'] == 'allocation'} == {
                     (label, date) for label in CASES for date in expected_dates}),
        'complete_factor_fit_grid': (
            len([item for item in solves if item['stage'] == 'factor_fit']) == 4 * len(dates)
            and {(item['estimator'], item['decision_date']) for item in solves
                 if item['stage'] == 'factor_fit'} == {
                     (label, date) for label, case in CASES.items() if case['model'] != 'EWMA'
                     for date in expected_dates}),
        'positive_definite_covariances': all(
            (np.linalg.eigvalsh(matrix) > 0).all() for item in estimates.values()
            for matrix in item['covars'].values()),
        'distinct_covariances': all(
            not np.allclose(a['covars'][dates[-1]], b['covars'][dates[-1]]) for a, b in pairs),
        'distinct_weights': all(
            not np.allclose(a['tables']['target_weights'], b['tables']['target_weights'])
            for a, b in portfolio_pairs),
        'implementation_schedule': all(
            item['portfolio'].is_rebalancing[item['portfolio'].is_rebalancing].index.equals(
                implementation) for item in portfolios.values()),
        'realized_cost_identity': all(np.allclose(
            item['portfolio'].realized_costs,
            item['portfolio'].units.diff().fillna(0).abs() * item['portfolio'].prices
            * params['cost_rate'], rtol=1e-11, atol=1e-12) for item in portfolios.values()),
        'opening_trade_identity': all(np.isclose(
            item['portfolio'].nav.iloc[1], params['initial_nav'] * (1 - params['cost_rate']),
            atol=1e-10, rtol=0) for item in portfolios.values()),
        'error_summary_identity': all(np.isclose(
            tables['covariance_errors'].loc[(label, dates[-1]), 'relative_frobenius_error'],
            np.sqrt(np.square((item['covars'][dates[-1]] - truth).to_numpy()).sum()
                    / np.square(truth.to_numpy()).sum()), rtol=1e-12, atol=1e-14)
            for label, item in estimates.items()),
        'generative_identity': np.allclose(
            tables['simulated_asset_returns'],
            tables['simulated_factor_returns'] @ tables['true_betas']
            + tables['simulated_residual_returns'], rtol=1e-12, atol=1e-14),
        'warmup_satisfied': all(
            len(panel.loc[:date]) >= params['warmup']
            for panel in weekly.values() for date in dates),
    }
    result = {
        'configuration': deepcopy(config),
        'checks': {name: bool(value) for name, value in checks.items()},
        'diagnostics': {'solves': solves, 'case_configurations': deepcopy(CASES)},
    }
    check_result(result, config)
    return {**result, 'tables': tables, 'estimates': estimates, 'portfolios': portfolios}


def _figure(tables):
    """Show comparable net NAVs and last-date covariance errors against the simulated truth."""
    import matplotlib.pyplot as plt
    import pandas as pd
    import qis

    navs = tables['navs']
    figure, axes = plt.subplots(2, 1, figsize=(11, 8.2))
    figure.subplots_adjust(top=0.80, bottom=0.19, left=0.11, right=0.97, hspace=0.88)
    figure.suptitle('Covariance estimators | synthetic factor model', fontsize=18, y=0.97)
    period = f'{navs.index[0]:%d %b %Y} - {navs.index[-1]:%d %b %Y}'
    figure.text(
        0.5, 0.925, period + '\nSame minimum-variance objective, constraints and trade dates',
        ha='center', va='top', fontsize=12)
    common = {'fontsize': 11, 'legend_stats': qis.LegendStats.NONE}
    qis.plot_time_series(
        navs, ax=axes[0], title='Net growth of 100', ylabel='Index level', colors=COLORS,
        var_format='{:,.0f}', date_format='%b %Y', x_date_freq='2QE', x_rotation=0,
        linewidth=1.3, linestyles=['-', '--', '-', '--', '-', '--'],
        legend_loc='upper center', bbox_to_anchor=(0.5, -0.20), ncols=3, **common)
    date = tables['decision_schedule'].index[-1]
    errors = tables['covariance_errors'].xs(date, level='date').iloc[:, 0].reindex(CASES)
    errors.index = pd.Index(SHORT_LABELS)
    qis.plot_bars(
        errors, ax=axes[1], title=f'Covariance error at the last decision ({date:%d %b %Y})',
        ylabel='Relative Frobenius error', colors=COLORS, stacked=False, is_sns=False,
        x_rotation=0, yvar_format='{:.0%}', add_bar_values=True, legend_loc=None, **common)
    figure.text(0.11, 0.035,
                'OptimalPortfolios construction | factorlasso fits | '
                'qis backtesting and analytics\n'
                'VN = volatility normalization; factor variants normalize factor covariance only.\n'
                '35% asset cap; 10 bp costs include entry. One simulated path is not a ranking.',
                fontsize=10)
    return figure


def produce(spec: dict) -> dict:
    """Return the registered figure, reproducible tables and validated native solve evidence.

    Args:
        spec: The covariance_comparison entry from the analytics registry.

    Returns:
        Executor-compatible figure, tables, configuration, checks and diagnostics.
    """
    result = build_analytics(spec['configuration'])
    return {key: result[key] for key in ('configuration', 'checks', 'diagnostics', 'tables')} | {
        'figures': {'MinVariance_multi_covar_estimator_backtest.PNG': _figure(result['tables'])}}


def main(argv: list[str] | None = None) -> int:
    """Generate the covariance family through the shared offline, C-local preview writer."""
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
