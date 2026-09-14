"""Offline synthetic portfolio-report previews for the documentation analytics registry.

OptimalPortfolios constructs quarterly maximum-diversification targets; qis supplies the
frozen synthetic data, holdings backtest, statistics and plots. This teaching baseline does
not reproduce the legacy live-data reports. Run this module for a C-local family preview;
only the complete-bundle runner and publisher can replace the six registered README assets.
"""

import argparse
from copy import deepcopy
from dataclasses import asdict
from datetime import datetime, timezone
import inspect
import logging
from pathlib import Path
import uuid

from tools.docs_analytics.registry import ROOT, load_registry
from tools.docs_analytics.validate import (
    check_result, dependencies_for, environment_identity, environment_snapshot, expected_files,
    file_record, input_files, offline, output_boundary, producer_contract, rendering_record,
    source_fingerprint, write_json,
)


NAME = 'portfolio_reports'
LABELS = ['US equity', 'Europe equity', 'Treasuries', 'IG bonds', 'Gold', 'Commodities']
COLORS = ['#255b90', '#528dc0', '#40856b', '#91b797', '#d7a334', '#9d688f']


def configuration() -> dict:
    """Declare the supported teaching baseline; registry edits must match executable choices."""
    return {
        'fixture': {'factory': 'qis.datasets.generate_synthetic_universe', 'source_files': []},
        'parameters': {
            'apply_quirks': False, 'report_start': '2015-01-01', 'decision_end': '2025-10-01',
            'returns_freq': 'W-WED', 'rebalancing_freq': 'QE', 'span': 52, 'demean': True,
            'max_weight': 0.35, 'initial_nav': 100.0, 'cost_rate': 0.001,
        },
        'conventions': {
            'data_kind': 'Synthetic teaching example, not market or paper-replication evidence.',
            'sample_start': '2010-01-04', 'sample_end': '2025-12-31', 'seed': 20260725,
            'universe': ['SEQ_US', 'SEQ_EU', 'SBD_TSY', 'SBD_IG', 'SCM_GLD', 'SCM_BCOM'],
            'missing_data': 'Clean fixture mode (apply_quirks=False); no filling or filtering.',
            'returns': 'log', 'observation_frequency': 'Business-day prices (B).',
            'estimation_frequency': 'Weekly Wednesday log returns; trailing EWMA demeaning.',
            'annualization': 52,
            'warmup': 'History from 2010-01-04; decisions requested from 2015-01-01.',
            'rebalance_frequency': (
                'Quarter-end mapped to the first weekly Wednesday observation on or after '
                'the boundary; save actual decision and implementation dates.'),
            'implementation_lag': 1,
            'transaction_costs': (
                '10 bp on gross traded notional, including entry; no funding or management fee.'),
            'sharpe_convention': (
                'qis SharpeConvention.PA: compounded annual return / weekly annualized volatility; '
                'zero risk-free rate. Benchmark is the fixture daily-rebalanced gross 60/40.'),
        },
        'tables': [
            'prices', 'benchmark_prices', 'target_weights', 'realized_weights', 'units',
            'navs', 'drawdowns', 'performance', 'last_covariance', 'risk_contributions',
            'trading', 'decision_schedule',
        ],
        'input_tables': ['prices', 'benchmark_prices'],
        'checks': [
            'finite_inputs_outputs', 'weight_constraints', 'accepted_solves',
            'point_in_time_covariance', 'implementation_schedule', 'realized_cost_identity',
            'opening_trade_identity',
        ],
        'rendering': {'dpi': 150, 'font_family': 'DejaVu Sans'},
        'solver': {'required': True, 'backends': {'SLSQP': {'ftol': 1e-8, 'maxiter': 500}}},
        'dependencies': {
            'optimalportfolios': 'optimalportfolios', 'qis': 'qis', 'scipy': 'scipy',
            'factorlasso': 'factorlasso', 'numba': 'numba', 'seaborn': 'seaborn',
        },
    }


class _SolverCapture(logging.Handler):
    """Collect actual structured solver records, including accepted DEBUG-level results."""

    def __init__(self):
        """Start a fresh per-run record list without changing application handlers."""
        super().__init__(logging.DEBUG)
        self.records = []

    def emit(self, record):
        """Retain only diagnostics emitted by OptimalPortfolios' solver validator."""
        diagnostic = getattr(record, 'solver_diag', None)
        if diagnostic is not None:
            self.records.append(diagnostic)


def _estimator(config):
    """Construct the existing causal EWMA estimator with every relevant choice explicit."""
    import optimalportfolios as op

    params = config['parameters']
    return op.EwmaCovarEstimator(
        rebalancing_freq=params['rebalancing_freq'], returns_freq=params['returns_freq'],
        span=params['span'], demean=params['demean'], is_apply_vol_normalised_returns=False)


def case_configuration(config: dict, objective: str) -> dict:
    """Declare an explicit covariance-only comparison case without changing the baseline."""
    choices = {
        'MAX_DIVERSIFICATION': ('SLSQP', {'ftol': 1e-8, 'maxiter': 500}),
        'MIN_VARIANCE': ('CLARABEL', {'options': 'CVXPY/CLARABEL defaults; no overrides'}),
        'EQUAL_RISK_CONTRIBUTION': (
            'risk_budgeting', {'options': 'Internal ADMM-CCD defaults; no overrides'}),
    }
    if objective not in choices:
        raise ValueError('Unsupported covariance-only objective')
    result = deepcopy(config)
    result['parameters']['objective'] = objective
    backend, options = choices[objective]
    result['solver']['backends'] = {backend: options}
    if objective == 'MIN_VARIANCE':
        result['dependencies'].update({'cvxpy': 'cvxpy', 'clarabel': 'clarabel'})
    return result


def _solve(prices, covars, constraints, config):
    """Use the production rolling dispatcher and restore logging state even after a failure."""
    import optimalportfolios as op

    # The rolling wrapper forwards these production defaults; refuse a silent upstream change.
    objective = config['parameters'].get('objective', 'MAX_DIVERSIFICATION')
    backend = next(iter(config['solver']['backends']))
    if objective == 'MAX_DIVERSIFICATION':
        signature = inspect.signature(op.opt_maximise_diversification)
        options = config['solver']['backends']['SLSQP']
        if any(signature.parameters[name].default != value for name, value in options.items()):
            raise ValueError('SLSQP defaults changed; review the recorded solver configuration')
    logger = logging.getLogger('optimalportfolios.optimization.solver_diagnostics')
    capture, previous_level = _SolverCapture(), logger.level
    logger.addHandler(capture)
    logger.setLevel(logging.DEBUG)
    try:
        weights = op.compute_rolling_optimal_weights(
            prices=prices, constraints=constraints, covar_dict=covars,
            portfolio_objective=getattr(op.PortfolioObjective, objective),
            optimiser_config=op.OptimiserConfig(
                solver=backend, verbose=False, apply_total_to_good_ratio=False,
                use_drifted_weights_0=True, factorize_covar=False))
    finally:
        logger.removeHandler(capture)
        logger.setLevel(previous_level)
        capture.close()
    return weights, capture.records


def _diagnostics(weights, covars, constraints, records):
    """Join each observed solver result to independently evaluated hard-constraint residuals."""
    import optimalportfolios as op

    if len(records) != len(weights):
        raise ValueError('Missing or duplicate solver diagnostics')
    solves = []
    for (date, row), record in zip(weights.iterrows(), records):
        if record.context != date.strftime('%Y-%m-%d'):
            raise ValueError('Solver diagnostic date does not match the decision')
        residuals = [asdict(item) for item in op.evaluate_constraint_residuals(
            row.to_numpy(), constraints, covar=covars[date].to_numpy())]
        native_success = {
            'SLSQP': record.status == '0',
            'CLARABEL': record.status in {'optimal', 'optimal_inaccurate'},
            'risk_budgeting': record.status is None and record.outcome == 'accepted',
        }.get(record.solver, False)
        explanation = {
            'SLSQP': 'SLSQP consumes the raw EWMA covariance; no factorization.',
            'CLARABEL': 'Explicit factorization disabled; CVXPY uses the raw EWMA covariance.',
            'risk_budgeting': (
                'ADMM-CCD does not use explicit factorization; the risk-budget wrapper '
                'floors variances below 1e-6. The comparison checks that none are below it.'),
        }.get(record.solver, 'Unknown backend; solver validation must reject this record.')
        solves.append({
            'solver': record.solver, 'context': record.context,
            'status': 'success' if record.accepted and native_success else 'rejected',
            'raw_status': record.status, 'accepted': record.accepted,
            'compliant': all(item['passed'] for item in residuals if item['hard']),
            'fallback_source': record.fallback_source, 'reason': record.reason,
            'constraint_residuals': residuals,
            'covariance_stabilization': {
                'factorized': False,
                'not_applicable': explanation,
            },
        })
    return {'solves': solves}


def build_analytics(
        config: dict, *, span: int | None = None, objective: str | None = None) -> dict:
    """Compute fixed offline inputs, production results and numerical checks before plotting.

    Args:
        config: Exact configuration returned by configuration() and stored in the registry.
        span: Optional explicit EWMA span for the sensitivity producer. It changes both
            trailing demeaning and covariance smoothing and is recorded in the returned config.
        objective: Optional covariance-only objective for the optimizer comparison; its native
            solver is recorded explicitly. This cannot be combined with a span override.

    Returns:
        Tables, named checks, actual solver diagnostics and the qis portfolio used for review.

    Raises:
        ValueError: If configuration, inputs, solver evidence or a numerical check is invalid.
    """
    import numpy as np
    import pandas as pd
    import qis
    from qis.datasets import generate_synthetic_universe
    import optimalportfolios as op

    if config != configuration():
        raise ValueError(
            'Unsupported portfolio-report configuration; update the baseline explicitly')
    if span is not None and (type(span) is not int or span < 2):
        raise ValueError('The sensitivity span must be an integer of at least two observations')
    if objective is not None and span is not None:
        raise ValueError('Vary one comparison dimension at a time')
    config = deepcopy(config) if objective is None else case_configuration(config, objective)
    if span is not None:
        config['parameters']['span'] = span
    labels = {
        'MAX_DIVERSIFICATION': 'Max diversification',
        'MIN_VARIANCE': 'Minimum variance',
        'EQUAL_RISK_CONTRIBUTION': 'Equal risk budgets',
    }
    ticker = labels[config['parameters'].get('objective', 'MAX_DIVERSIFICATION')] + ' (net)'
    params, conventions = config['parameters'], config['conventions']
    fixture = generate_synthetic_universe(
        start=conventions['sample_start'], end=conventions['sample_end'],
        seed=conventions['seed'], apply_quirks=params['apply_quirks'])
    prices = fixture.prices[conventions['universe']].copy()
    benchmark = fixture.benchmark_prices.copy()
    if not np.isfinite(prices.to_numpy()).all() or not (prices > 0).all().all():
        raise ValueError('The clean teaching fixture requires finite positive prices')
    estimator = _estimator(config)
    covars = estimator.fit_rolling_covars(
        prices, qis.TimePeriod(params['report_start'], params['decision_end']))
    constraints = op.Constraints(
        is_long_only=True, min_exposure=1.0, max_exposure=1.0,
        max_weights=pd.Series(params['max_weight'], index=prices.columns))
    weights, records = _solve(prices, covars, constraints, config)
    diagnostics = _diagnostics(weights, covars, constraints, records)
    portfolio = qis.backtest_model_portfolio(
        prices.loc[weights.index[0]:], weights, initial_nav=params['initial_nav'],
        rebalancing_costs=params['cost_rate'],
        weight_implementation_lag=conventions['implementation_lag'],
        funding_rate=None, management_fee=None, instruments_carry=None,
        is_rebalanced_at_first_date=False, ticker=ticker)
    navs = pd.concat([portfolio.nav, benchmark.reindex(portfolio.nav.index)], axis=1)
    navs.columns = [ticker, 'Synthetic 60/40 (gross)']
    navs = params['initial_nav'] * navs.div(navs.iloc[0])
    drawdowns = qis.compute_rolling_drawdowns(navs)
    perf = qis.compute_ra_perf_table(navs, perf_params=qis.PerfParams(
        freq='W-WED', return_type=qis.ReturnTypes.LOG, sharpe_convention=qis.SharpeConvention.PA))
    perf = perf[[stat.value.name for stat in
                 (qis.PerfStat.PA_RETURN, qis.PerfStat.VOL, qis.PerfStat.SHARPE_RF0)]]
    last_date = weights.index[-1]
    risk = qis.compute_portfolio_risk_contributions(weights.loc[last_date], covars[last_date])
    turnover = portfolio.get_turnover(is_agg=True, roll_period=1)
    costs = portfolio.get_costs(is_agg=True, roll_period=1)
    # The initial cash-only observation has no preceding units; its turnover is defined as zero.
    turnover.iloc[0] = 0.0
    trading = pd.concat([turnover.rename('gross_turnover_fraction'),
                         costs.rename('cost_fraction_of_same_day_nav')], axis=1)
    traded_dates = portfolio.prices.index[
        portfolio.prices.index.searchsorted(weights.index) + conventions['implementation_lag']]
    schedule = pd.DataFrame({'implementation_date': traded_dates}, index=weights.index)
    tables = {
        'prices': prices, 'benchmark_prices': benchmark, 'target_weights': weights,
        'realized_weights': portfolio.weights.copy(), 'units': portfolio.units.copy(),
        'navs': navs, 'drawdowns': drawdowns, 'performance': perf,
        'last_covariance': covars[last_date],
        'risk_contributions': risk.rename('annual_volatility_contribution').to_frame(),
        'trading': trading, 'decision_schedule': schedule,
    }
    for name, table in tables.items():
        table.index.name = 'asset' if name in {'last_covariance', 'risk_contributions'} else (
            'strategy' if name == 'performance' else 'date')
    unit_changes = portfolio.units.diff().fillna(portfolio.units.iloc[0])
    cost_reference = unit_changes.abs() * portfolio.prices * params['cost_rate']
    sampled_dates = [weights.index[0], weights.index[len(weights) // 2], weights.index[-1]]
    point_in_time = all(np.allclose(
        covars[date], estimator.fit_current_covar(prices.loc[:date]), rtol=1e-12, atol=1e-14)
        for date in sampled_dates)
    opening = portfolio.nav.iloc[1]
    checks = {
        'finite_inputs_outputs': all(np.isfinite(table.to_numpy()).all()
                                     for name, table in tables.items()
                                     if name != 'decision_schedule'),
        'weight_constraints': all(item['compliant'] for item in diagnostics['solves']),
        'accepted_solves': all(item['accepted'] and item['fallback_source'] is None
                              for item in diagnostics['solves']),
        'point_in_time_covariance': point_in_time,
        'implementation_schedule': portfolio.is_rebalancing[
            portfolio.is_rebalancing].index.equals(traded_dates),
        'realized_cost_identity': np.allclose(
            portfolio.realized_costs, cost_reference, rtol=1e-11, atol=1e-12),
        'opening_trade_identity': np.isclose(
            opening, params['initial_nav'] * (1 - params['cost_rate']), atol=1e-10, rtol=0),
    }
    checks = {name: bool(value) for name, value in checks.items()}
    result = {'configuration': deepcopy(config), 'checks': checks, 'diagnostics': diagnostics}
    check_result(result, config)
    return {**result, 'tables': tables, 'portfolio': portfolio, 'covars': covars}


def _figures(tables, config):
    """Draw three readable two-panel previews with qis and common colors, dates and units."""
    import matplotlib.pyplot as plt
    import pandas as pd
    import qis

    navs, weights = tables['navs'], tables['target_weights']
    period = f'{navs.index[0]:%d %b %Y} - {navs.index[-1]:%d %b %Y}'
    figures = {}
    common = {'fontsize': 12, 'legend_stats': qis.LegendStats.NONE}
    figure, axes = plt.subplots(2, 1, figsize=(11, 7.6))
    figure.subplots_adjust(top=0.79, bottom=0.12, left=0.10, right=0.97, hspace=0.55)
    figure.suptitle('Portfolio performance | synthetic example', fontsize=18, y=0.97)
    figure.text(0.5, 0.915, period + '\n10 bp trading costs | one business-day lag',
                ha='center', va='top', fontsize=12)
    qis.plot_time_series(
        navs, ax=axes[0], title='Growth of 100', ylabel='Index level', var_format='{:,.0f}',
        colors=[COLORS[0], '#777777'], legend_loc='upper left',
        date_format='%Y', x_rotation=0, **common)
    qis.plot_time_series(
        tables['drawdowns'], ax=axes[1], title='Drawdown from the running peak',
        ylabel='Drawdown', var_format='{:.0%}', colors=[COLORS[0], '#777777'],
        legend_loc=None, date_format='%Y', x_rotation=0, **common)
    figure.text(0.10, 0.035, 'OptimalPortfolios construction | qis analytics\n'
                '60/40 is the fixture gross benchmark; results are illustrative.', fontsize=10)
    figures['example_portfolio_factsheet1.PNG'] = figure

    figure, axes = plt.subplots(2, 1, figsize=(11, 7.6))
    figure.subplots_adjust(top=0.79, bottom=0.15, left=0.10, right=0.97, hspace=0.75)
    last_date = weights.index[-1]
    figure.suptitle('Allocation and risk | synthetic example', fontsize=18, y=0.97)
    figure.text(0.5, 0.915, f'Target decision: {last_date:%d %b %Y}\n'
                'Weekly log returns | EWMA span 52 | covariance annualized by 52',
                ha='center', va='top', fontsize=12)
    allocation = weights.iloc[-1].copy()
    allocation.index = pd.Index(LABELS)
    risk = tables['risk_contributions'].iloc[:, 0].copy()
    risk.index = pd.Index(LABELS)
    for axis, series, title, ylabel in [
            (axes[0], allocation, 'Target weights (35% cap per asset)', 'Portfolio weight'),
            (axes[1], risk, 'Contribution to annualized portfolio volatility',
             'Volatility contribution')]:
        qis.plot_bars(
            series, ax=axis, title=title, ylabel=ylabel, stacked=False,
            colors=COLORS, is_sns=False, x_rotation=0, yvar_format='{:.1%}',
            add_bar_values=True, legend_loc=None, **common)
    figure.text(0.10, 0.035, 'OptimalPortfolios targets | qis volatility contributions\n'
                'Target weights are decided weights; holdings drift between trades.', fontsize=10)
    figures['example_portfolio_factsheet2.PNG'] = figure

    figure, axes = plt.subplots(2, 1, figsize=(11, 7.6))
    figure.subplots_adjust(top=0.79, bottom=0.12, left=0.10, right=0.97, hspace=0.68)
    figure.suptitle('Allocation through time | synthetic example', fontsize=18, y=0.97)
    figure.text(0.5, 0.915, period + '\nQuarterly targets and realized trading costs',
                ha='center', va='top', fontsize=12)
    # Extend only the display of the last target to the end; do not add a decision or trade.
    display = weights.reindex(navs.index, method='ffill')
    display.columns = LABELS
    # Numeric display positions keep dates unique; year-only categorical labels collapse quarters.
    display.index = pd.RangeIndex(len(display))
    qis.plot_stack(
        display, ax=axes[0], title='Decided target weights', step='post', colors=COLORS,
        ncols=3, legend_loc='upper center', bbox_to_anchor=(0.5, -0.15),
        is_yaxis_limit_01=True, skip_y_axis=False, x_rotation=0, x_date_freq=None,
        ylabel='Portfolio weight', **common)
    years = pd.date_range(navs.index[0], navs.index[-1], freq='YS')
    axes[0].set_xticks(navs.index.searchsorted(years), years.strftime('%Y'))
    costs = tables['trading']['cost_fraction_of_same_day_nav'].resample('QE').sum() * 10000
    qis.plot_time_series(
        costs.rename('Trading costs'), ax=axes[1], title='Trading costs by calendar quarter',
        ylabel='Sum of daily cost / NAV (bp)', var_format='{:.1f}',
        legend_loc=None, colors=[COLORS[0]], date_format='%Y', x_rotation=0, **common)
    figure.text(0.10, 0.035, 'OptimalPortfolios construction | qis holdings and costs\n'
                'Costs include entry; quarterly sums are not compounded performance drag.',
                fontsize=10)
    figures['example_customised_report.PNG'] = figure
    return figures


def produce(spec: dict) -> dict:
    """Return the three registered figures, input/result tables and validated solver evidence.

    Args:
        spec: The portfolio_reports producer entry from the analytics registry.

    Returns:
        Executor-compatible figures, tables, configuration, checks and diagnostics.
    """
    analytical = build_analytics(spec['configuration'])
    figures = _figures(analytical['tables'], spec['configuration'])
    return {key: analytical[key] for key in
            ('configuration', 'checks', 'diagnostics', 'tables')} | {'figures': figures}


def generate_preview(output: Path, root: Path = ROOT, *, family_name: str = NAME) -> Path:
    """Write a checked C-local family preview without creating a publishable bundle.

    Args:
        output: New directory below AGENT_LOCAL_ROOT, outside the source tree and OneDrive.
        root: Source export supplying the registry and producer implementation.
        family_name: Implemented registered family to render; defaults to portfolio_reports.

    Returns:
        Preview directory with registered images, CSV snapshots and family_manifest.json.

    Raises:
        ValueError: If configuration, outputs, source identity or numerical checks fail.
    """
    from tools.docs_analytics.run import _produce, _save_outputs

    registry = load_registry(root)
    if family_name not in registry['producers']:
        raise ValueError(f'Unknown preview family: {family_name}')
    spec = registry['producers'][family_name]
    if spec['status'] != 'implemented':
        raise ValueError(f'Preview family is pending: {family_name}')
    config = producer_contract(spec, root)
    family = {**registry, 'producers': {family_name: spec},
              'assets': [asset for asset in registry['assets'] if asset['producer'] == family_name]}
    dependencies = dependencies_for(family)
    output = output_boundary(output, root)
    source, inputs = source_fingerprint(root), input_files(family, root)
    staging = output.with_name(f'.{output.name}-building-{uuid.uuid4().hex}')
    output_boundary(staging, root)
    staging.mkdir(parents=True)
    (staging / 'images').mkdir()
    try:
        with offline():
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt

            environment = environment_snapshot(dependencies)
            imported = Path(environment['libraries']['optimalportfolios']['module_path'])
            if imported != (root / 'src/optimalportfolios/__init__.py').resolve():
                raise ValueError('Import optimalportfolios from the selected source export')
            rendering = rendering_record(config['rendering'])
            old_figures = set(plt.get_fignums())
            try:
                with matplotlib.rc_context({
                        'font.family': config['rendering']['font_family'],
                        'figure.dpi': config['rendering']['dpi']}):
                    result = _produce(family_name, spec, root)
                    result = _save_outputs(family_name, result, family, staging)
            finally:
                for number in set(plt.get_fignums()) - old_figures:
                    plt.close(number)
            outputs = {name: file_record(staging / name)
                       for name in sorted(expected_files(family))}
            if source_fingerprint(root) != source or load_registry(root) != registry:
                raise ValueError('Source changed during preview generation')
            if input_files(family, root) != inputs:
                raise ValueError('Fixture inputs changed during preview generation')
            if environment_identity(environment_snapshot(dependencies)) != environment_identity(
                    environment):
                raise ValueError('Imported dependency source changed during preview generation')
            write_json(staging / 'family_manifest.json', {
                'schema_version': 1, 'kind': 'documentation_analytics_family_preview',
                'status': 'complete', 'family': family_name, 'publication_ready': False,
                'review_status': 'pending',
                'generated_at_utc': datetime.now(timezone.utc).isoformat(),
                'registry': registry, 'source': source, 'input_files': inputs,
                'input_tables': {f'tables/{family_name}/{label}.csv':
                                 outputs[f'tables/{family_name}/{label}.csv']
                                 for label in config['input_tables']},
                'environment': environment, 'rendering': {family_name: rendering},
                'producers': {family_name: result}, 'outputs': outputs,
            })
        output_boundary(output, root)
        staging.rename(output)
    except BaseException as error:
        manifest = staging / 'family_manifest.json'
        if manifest.exists():
            manifest.unlink()
        write_json(staging / 'FAILED.json', {
            'status': 'failed', 'error': f'{type(error).__name__}: {error}'})
        raise
    return output


def main(argv: list[str] | None = None) -> int:
    """Generate this family's offline previews; publication remains a complete-bundle action."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        print(generate_preview(args.output_root))
    except (ValueError, RuntimeError, OSError) as error:
        parser.exit(2, f'{error}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
