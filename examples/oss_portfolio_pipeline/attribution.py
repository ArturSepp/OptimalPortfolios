"""QIS layer attribution and optional complete two-feature experiment on the public pipeline."""
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import optimalportfolios as op
import qis
from qis.portfolio.attribution.model_layer import (
    ALPHA_AN_CI_HIGH_COLUMN, ALPHA_AN_CI_LOW_COLUMN,
)

from examples.oss_portfolio_pipeline.backtest import (
    PortfolioBacktest, execution_outcome, portfolio_constraints, run_backtest,
)
from examples.oss_portfolio_pipeline.config import PipelineConfig
from examples.oss_portfolio_pipeline.covariance import RollingCovariances
from examples.oss_portfolio_pipeline.data import external_directory
from examples.oss_portfolio_pipeline.pipeline import PipelineInputs
from examples.oss_portfolio_pipeline.reporting import draw_table, report_page


MOMENTUM_FEATURE = 'Longer momentum span'
TE_FEATURE = 'Raise TE budget by 50%'
LAYER_FIELDS = ('benchmark_nav', 'risk_layer_nav', 'signal_layer_nav',
                'full_model_nav', 'full_model_net_nav')
BRIDGE_COLUMNS = ('Systematic Return', 'Risk Layer Alpha', 'Signal Layer Alpha',
                  'Integration Alpha', 'Trading Cost Drag')


@dataclass
class LayerRun:
    """Completed implementable layer portfolios and QIS descriptive attribution."""

    full: PortfolioBacktest
    risk_executed: op.ExecutedRollingResult
    risk_gross: qis.PortfolioData
    signal: qis.PortfolioData
    navs: qis.ModelLayerNavs
    attribution: qis.ModelLayerAlphaBetaAttribution
    risk_diagnostics: pd.DataFrame
    risk_audit: pd.DataFrame
    identity_errors: pd.Series


@dataclass
class AttributionStudy:
    """Baseline layers, optional complete feature coalitions, and QIS feature effects."""

    scenarios: dict[frozenset[str], LayerRun]
    features: qis.ModelFeatureAlphaBetaAttribution | None = None


def build_layer_run(inputs: PipelineInputs, risk: RollingCovariances,
                    full: PortfolioBacktest) -> LayerRun:
    """Backtest minimum-variance and top-quartile signal layers on the full model's dates.

    The risk layer uses OP's minimum-variance objective with the same excess covariance,
    hard constraints, costs and execution lag as the full model, then replays its targets
    gross. A zero-alpha max-alpha solve has no unique risk-only objective and is not used.
    The signal layer holds the highest-scoring quarter of the 35 non-BIL ETFs equally,
    without the full model's TE, sleeve or turnover constraints. QIS owns both backtests.
    """
    settings = full.settings
    prices = full.executed.portfolio.prices
    decisions = full.executed.targets.index
    policy = portfolio_constraints(inputs, settings)
    covariances = risk.covariances
    diagnostics, audits = {}, {}

    def decide(history: pd.DataFrame, state: op.ExecutedState):
        """Use only the dated risk estimate and actual pretrade holdings in the risk-only solve."""
        if history.index[-1] != state.date:
            raise ValueError('Risk-only decision received future history')
        previous = None if state.date == decisions[0] else state.weights
        outcome = op.cvx_quadratic_optimisation(
            portfolio_objective=op.PortfolioObjective.MIN_VARIANCE,
            covar=covariances[state.date].to_numpy(), constraints=policy.copy(weights_0=previous),
            solver=settings.solver, context=f'risk layer {state.date.date()}',
            factorize_covar=True)
        if not outcome.accepted or not outcome.compliant:
            raise RuntimeError(f'Risk layer {state.date.date()}: {outcome.status}; '
                               'no fallback portfolio accepted')
        outcome = execution_outcome(outcome)
        target = pd.Series(outcome.weights, index=inputs.universe.index)
        diagnostics[state.date] = {
            'solver': outcome.solver, 'status': outcome.status,
            'accepted': outcome.accepted, 'compliant': outcome.compliant,
            'ex_ante_tracking_error': full.risk_model.compute_tre_at_date(
                inputs.benchmark_weights, target, state.date),
        }
        audits[state.date] = outcome.residuals_frame()
        return target, outcome

    risk_executed = op.backtest_with_qis_state(
        prices, decisions, decide, implementation_lag=settings.implementation_lag,
        transaction_costs=settings.transaction_cost, initial_nav=100.)
    common = dict(prices=prices, rebalancing_costs=0., initial_nav=100.,
                  weight_implementation_lag=settings.implementation_lag)
    risk_gross = qis.backtest_model_portfolio(
        **common, weights=risk_executed.targets, ticker='Risk layer (gross)')
    scores = full.alpha_scores.loc[decisions].drop(columns='BIL')
    if not np.isfinite(scores).all().all():
        raise ValueError('Signal-only scores must be finite before ranking')
    signal_targets = op.compute_top_quantile_equal_weights(
        alpha_scores=scores, prices=prices.loc[:, scores.columns], quantile=.25)
    signal_targets = signal_targets.reindex(columns=prices.columns, fill_value=0.)
    signal = qis.backtest_model_portfolio(
        **common, weights=signal_targets, ticker='Signal layer (gross)')
    navs = qis.ModelLayerNavs(
        benchmark_nav=full.gross_benchmark.nav,
        risk_layer_nav=risk_gross.nav, signal_layer_nav=signal.nav,
        full_model_nav=full.gross_portfolio.nav, full_model_net_nav=full.executed.portfolio.nav)
    attribution = qis.compute_model_layer_alpha_beta_attribution(
        **{field: getattr(navs, field) for field in LAYER_FIELDS},
        freq='ME', hac_lags=3, confidence_level=.95)
    components = attribution.component_returns
    errors = pd.Series({
        'Gross return bridge': (components[list(BRIDGE_COLUMNS[:-1])].sum(axis=1)
                                - components['Full Model Return']).abs().max(),
        'Net return bridge': (components[list(BRIDGE_COLUMNS)].sum(axis=1)
                              - components['Full Model Net Return']).abs().max(),
    }, name='Maximum absolute log-return error')
    if errors.max() > 1e-10:
        raise RuntimeError('QIS model-layer return bridge failed to reconcile')
    return LayerRun(full, risk_executed, risk_gross, signal, navs, attribution,
                    pd.DataFrame.from_dict(diagnostics, orient='index'),
                    pd.concat(audits, names=['decision_date', 'constraint_row']), errors)


def run_attribution(inputs: PipelineInputs, risk: RollingCovariances, config: PipelineConfig,
                    full: PortfolioBacktest, include_features: bool = False) -> AttributionStudy:
    """Attribute baseline layers and optionally rerun all four horizon/TE-budget coalitions.

    Feature values are deterministic sensitivities, not parameters selected on performance:
    Twice the momentum span and 1.5 times its TE cap. Data, covariance fits,
    benchmark, all other constraints, transaction costs and execution lag remain identical.
    """
    monthly = qis.to_returns(full.navs, freq='ME', is_log_returns=True).dropna(how='any')
    if len(monthly) < 3:
        raise ValueError('Layer attribution needs three completed monthly returns; '
                         'extend --end or use --no-attribution')
    scenarios = {frozenset(): build_layer_run(inputs, risk, full)}
    if not include_features:
        return AttributionStudy(scenarios)
    for coalition in (frozenset({MOMENTUM_FEATURE}), frozenset({TE_FEATURE}),
                      frozenset({MOMENTUM_FEATURE, TE_FEATURE})):
        settings = replace(
            full.settings,
            momentum_span=(2 * full.settings.momentum_span if MOMENTUM_FEATURE in coalition
                           else full.settings.momentum_span),
            tracking_error=(1.5 * full.settings.tracking_error if TE_FEATURE in coalition
                            else full.settings.tracking_error))
        scenario = run_backtest(inputs, risk, config, settings)
        scenarios[coalition] = build_layer_run(inputs, risk, scenario)
    # These are completed counterfactual NAV bundles, not fitted coefficient tables.
    # QIS enforces complete coalitions, identical benchmarks and exact path reconstruction.
    features = qis.compute_model_feature_alpha_beta_attribution(
        {key: run.navs for key, run in scenarios.items()},
        freq='ME', hac_lags=3, confidence_level=.95)
    return AttributionStudy(scenarios, features)


def layer_nav_table(navs: qis.ModelLayerNavs) -> pd.DataFrame:
    """Expose the five actual input paths with readable layer names."""
    return pd.concat({
        'Benchmark (gross)': navs.benchmark_nav, 'Risk layer (gross)': navs.risk_layer_nav,
        'Signal layer (gross)': navs.signal_layer_nav, 'Full model (gross)': navs.full_model_nav,
        'Full model (net)': navs.full_model_net_nav,
    }, axis=1)


def attribution_figures(study: AttributionStudy) -> dict[str, plt.Figure]:
    """Render QIS layer paths, regression intervals, exact bridges and optional feature effects."""
    base = study.scenarios[frozenset()]
    attribution = base.attribution
    first, last = attribution.periodic_returns.index[[0, -1]]
    subtitle = f'Completed ME log returns: {first:%b %Y} - {last:%b %Y} | OLS, HAC(3), 95% CI'
    common = dict(fontsize=8, x_date_freq='YS', date_format='%Y', x_rotation=0,
                  legend_stats=qis.LegendStats.NONE)
    figures = {}
    fig = report_page(
        'Attribution by model layer', subtitle,
        'Risk: constrained minimum excess variance. Signal: top-quarter score equal weights.\n'
        'Full: original constrained max-alpha policy. Gross paths replay the net-policy targets.\n'
        'Full-sample descriptive regressions; these coefficients are not portfolio inputs.', 1)
    figures['model_layers'] = fig
    grid = fig.add_gridspec(2, 1, left=.09, right=.94, bottom=.16, top=.86,
                           height_ratios=[3, 1.5], hspace=.40)
    qis.plot_time_series(layer_nav_table(base.navs), ax=fig.add_subplot(grid[0]),
                         title='Standalone layers and integrated portfolio | USD NAV',
                         colors=['#687684', '#5B9A91', '#D99D1E', '#126782', '#17313b'], **common)
    regression = attribution.regression_table
    table = pd.DataFrame({
        'Alpha p.a.': regression[qis.PerfStat.ALPHA_AN.to_str()].map('{:.2%}'.format),
        '95% CI low': regression[ALPHA_AN_CI_LOW_COLUMN].map('{:.2%}'.format),
        '95% CI high': regression[ALPHA_AN_CI_HIGH_COLUMN].map('{:.2%}'.format),
        'Beta': regression[qis.PerfStat.BETA.to_str()].map('{:.2f}'.format),
    })
    draw_table(table, fig.add_subplot(grid[1]), percent=False,
               col_widths=[2, 1.2, 1.2, 1.2, .8],
               title='Benchmark-OLS alpha with Bartlett-HAC intervals')

    fig = report_page(
        'From layers to net performance', subtitle,
        'Systematic + risk alpha + signal alpha + integration alpha + cost drag = net log return.\n'
        'Integration is the reconciling residual; it is not a separately investable portfolio.\n'
        'Annualised means and cumulative log contributions are additive; neither is a CAGR.', 2)
    figures['layer_bridge'] = fig
    grid = fig.add_gridspec(2, 1, left=.09, right=.94, bottom=.18, top=.86, hspace=.55)
    annual = attribution.annualised_components.reindex(
        [*BRIDGE_COLUMNS, 'Full Model Net Return'])
    annual.index = ['Systematic', 'Risk alpha', 'Signal alpha', 'Integration', 'Costs', 'Full net']
    qis.plot_bars(annual, ax=fig.add_subplot(grid[0]), stacked=False, add_bar_values=True,
                   x_rotation=0, yvar_format='{:.1%}', xvar_format='{:.1%}', fontsize=8,
                   legend_loc=None, title='Annualised mean log-return contributions')
    cumulative = attribution.component_returns[list(BRIDGE_COLUMNS[1:])].cumsum()
    cumulative['Total model alpha (net)'] = cumulative.sum(axis=1)
    qis.plot_time_series(cumulative, ax=fig.add_subplot(grid[1]), var_format='{:.0%}',
                         title='Cumulative alpha and cost contributions | log-return points',
                         **common)

    if study.features is not None:
        fig = report_page(
            'Feature effects across layers', subtitle,
            'Four combinations share data, covariance estimates, benchmark and cost assumptions.\n'
            'Shapley effects allocate interactions equally across feature orderings.\n'
            'Intervals use effect paths directly. This is descriptive sensitivity analysis.', 3)
        figures['feature_attribution'] = fig
        grid = fig.add_gridspec(2, 1, left=.08, right=.95, bottom=.28, top=.85,
                               height_ratios=[1, 1.25], hspace=.55)
        summary = study.features.summary
        effects = summary.loc['Shapley']
        metrics = ['Risk Layer Alpha', 'Signal Layer Alpha', 'Integration Alpha',
                   'Full Model Alpha', 'Annualised Full Model Net Return']
        labels = ['Risk alpha', 'Signal alpha', 'Integration', 'Full alpha', 'Net return']
        qis.plot_bars(effects[metrics].rename(columns=dict(zip(metrics, labels))).T,
                       stacked=False, ax=fig.add_subplot(grid[0]), x_rotation=0,
                       fontsize=8, yvar_format='{:.1%}', legend_loc='upper left',
                       title='Shapley annualised effects | log-return units')
        estimates = pd.DataFrame(index=pd.Index(labels, name='Effect'))
        for feature, row in effects.iterrows():
            estimates[feature] = [
                f'{row[metric]:.2%} [{row[metric + " CI Low"]:.2%}, '
                f'{row[metric + " CI High"]:.2%}]' for metric in metrics]
        draw_table(estimates, fig.add_subplot(grid[1]), percent=False,
                   col_widths=[1.6, 3, 3], title='Estimate [95% Bartlett-HAC interval]')
        fig.text(.09, .20,
                 f'Baseline momentum span: {base.full.settings.momentum_span} months; '
                 f'feature: {2 * base.full.settings.momentum_span} months.\n'
                 f'Baseline TE cap: {base.full.settings.tracking_error:.1%}; '
                 f'feature: {1.5 * base.full.settings.tracking_error:.1%}.\n'
                 'Factorial effects, joint effects and identity checks are in the CSVs.',
                 fontsize=8, color='#52646c', va='top', linespacing=1.6)
    return figures


def save_attribution(study: AttributionStudy, output_dir: Path | str,
                     render: bool = True) -> Path:
    """Save layer runs, feature effects and a separate PDF linked to the baseline backtest."""
    output = external_directory(output_dir)
    backtest_manifest = output / 'backtest_manifest.json'
    if not backtest_manifest.is_file():
        raise ValueError('Save the baseline portfolio backtest before attribution')
    kind = 'features' if study.features is not None else 'layers'
    destination = external_directory(output / 'attribution' / kind)
    destination.mkdir(parents=True, exist_ok=True)
    tables, scenarios = {}, {}
    for index, (coalition, run) in enumerate(study.scenarios.items()):
        name = f'scenario_{index}'
        scenarios[name] = {'features': sorted(coalition), 'settings': asdict(run.full.settings)}
        for suffix, table in {
            'layer_navs': layer_nav_table(run.navs),
            'periodic_returns': run.attribution.periodic_returns,
            'regressions': run.attribution.regression_table,
            'component_returns': run.attribution.component_returns,
            'annualised_components': run.attribution.annualised_components,
            'identity_errors': run.identity_errors,
            'full_targets': run.full.executed.targets,
            'risk_targets': run.risk_executed.targets,
            'signal_targets': run.signal.input_weights,
            'alpha_scores': run.full.alpha_scores,
            'full_diagnostics': run.full.diagnostics,
            'full_constraint_audit': run.full.constraint_audit,
            'risk_diagnostics': run.risk_diagnostics,
            'risk_constraint_audit': run.risk_audit,
        }.items():
            tables[f'{name}_{suffix}'] = table
    if study.features is not None:
        features = study.features
        tables['feature_summary'] = features.summary
        tables['feature_identity_errors'] = features.identity_errors
        tables['shapley_effect_navs'] = pd.concat(
            {name: layer_nav_table(navs) for name, navs in features.shapley_feature_navs.items()},
            names=['feature', 'date'])
        tables['factorial_effect_navs'] = pd.concat(
            {' + '.join(sorted(key)): layer_nav_table(navs)
             for key, navs in features.factorial_effect_navs.items()}, names=['coalition', 'date'])
        tables['joint_effect_navs'] = layer_nav_table(features.joint_effect_navs)
    hashes = {}
    for name, table in tables.items():
        path = destination / f'{name}.csv'
        path.write_text(table.to_csv(float_format='%.17g'), encoding='utf-8', newline='')
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    pages = []
    if render:
        with plt.rc_context({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42}):
            figures = attribution_figures(study)
            try:
                pdf = Path(qis.save_figs_to_pdf(figures, file_name='model_attribution',
                           local_path=str(destination), orientation='portrait',
                           add_current_date=False))
                hashes[pdf.name] = hashlib.sha256(pdf.read_bytes()).hexdigest()
                for name, fig in figures.items():
                    path = destination / f'{name}.png'
                    fig.savefig(path, dpi=160, facecolor='white')
                    hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
                pages = list(figures)
            finally:
                for fig in figures.values():
                    plt.close(fig)
    manifest = {
        'built_at_utc': datetime.now(timezone.utc).isoformat(),
        'backtest_manifest_sha256': hashlib.sha256(backtest_manifest.read_bytes()).hexdigest(),
        'scenarios': scenarios, 'pages': pages, 'file_sha256': hashes,
        'regression': {'frequency': 'ME', 'return_type': 'log', 'hac_lags': 3,
                       'confidence_level': .95, 'type': 'full-sample descriptive OLS'},
        'risk_layer': 'OP minimum excess variance with full-model constraints and net execution',
        'signal_layer': 'OP equal-weight top 25% of 35 non-BIL ranks; gross; unconstrained',
        'benchmark': 'identical fixed-weight gross benchmark in every coalition',
        'costs': 'gross replays net-policy targets; cost drag is full net minus gross log return',
        'opening_cost': 'embedded in inception NAV; outside measured attribution returns',
        'features': 'full factorial experiment; QIS Harsanyi / exact Shapley effects on NAV ratios',
        'versions': {name: version(name) for name in ('qis', 'optimalportfolios', 'matplotlib')},
    }
    (destination / 'attribution_manifest.json').write_text(
        json.dumps(manifest, indent=2, allow_nan=False), encoding='utf-8')
    return destination
