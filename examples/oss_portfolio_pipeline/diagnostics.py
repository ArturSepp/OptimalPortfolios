"""QIS predictive-signal, allocation-breadth and realised Brinson diagnostics."""
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import pandas as pd
import qis

from examples.oss_portfolio_pipeline.backtest import PortfolioBacktest
from examples.oss_portfolio_pipeline.covariance import RollingCovariances
from examples.oss_portfolio_pipeline.factsheets import comparison_runs, write_report_bundle
from examples.oss_portfolio_pipeline.pipeline import PipelineInputs
from examples.oss_portfolio_pipeline.reporting import (
    draw_table, report_page, report_portfolios,
)


GROUP_LABELS = {'Government bonds': 'Govt bonds', 'Credit and EM debt': 'Credit / EM'}


@dataclass
class InvestmentDiagnostics:
    """QIS numerical results and their complete, exportable input/audit tables."""

    signals: dict[str, qis.SignalDiagnosticsResult]
    ic_series: pd.DataFrame
    ic_summary: pd.DataFrame
    breadth: dict[str, qis.PortfolioBreadthResult]
    brinson: dict[str, tuple]
    tables: dict[str, pd.DataFrame | pd.Series]
    metadata: dict


def prepare_investment_diagnostics(inputs: PipelineInputs, risk: RollingCovariances,
                                   full: PortfolioBacktest,
                                   alternatives: list[PortfolioBacktest] | None = None
                                   ) -> InvestmentDiagnostics:
    """Evaluate existing decisions with QIS; do not refit signals or covariance models.

    Forward signal pairs use completed ME excess log returns and a one-period QIS lag.
    Integer horizons are non-overlapping and anchored at the first post-inception month.
    Breadth uses exact production covariances and excludes BIL from every count.
    Brinson uses daily beginning-of-period holdings and Frongello linking before aggregation.
    """
    runs = comparison_runs(full, alternatives)
    start, end = full.executed.targets.index[[0, -1]]
    first_month, last_month = pd.DatetimeIndex([start, end]).to_period('M').to_timestamp('M')
    assets = inputs.universe.index.drop('BIL')
    returns = risk.asset_excess_returns.loc[:, assets]
    returns = returns.loc[(returns.index > first_month) & (returns.index <= last_month)]
    if returns.empty:
        raise ValueError('Signal diagnostics need at least one completed post-inception month')
    groups = inputs.universe.economic_group.loc[assets]
    signals, summaries, monthly_ic, tables, spans = {}, {}, {}, {}, {}
    tables['signal_forward_monthly_excess_log_returns'] = returns
    for run in runs.values():
        span = run.settings.momentum_span
        if span in spans:
            prior = spans[span]
            if not (prior.raw_alpha.equals(run.raw_alpha)
                    and prior.alpha_scores.equals(run.alpha_scores)):
                raise ValueError('Same momentum span has different signal histories')
            continue
        spans[span] = run
        for kind, scores in (('raw', run.raw_alpha), ('scored', run.alpha_scores)):
            name = f'{span}m {kind}'
            scores = scores.loc[start:end, assets]
            label = f'signal_{span}m_{kind}'
            tables[f'{label}_observed_scores'] = scores
            result = qis.estimate_signal_diagnostics(
                {'ME': returns}, scores, group_data=groups, horizons=(1, 3, 6),
                is_log_returns=True, fit_intercept=False, is_vol_normalised=True)
            signals[name] = result
            summaries[name] = qis.estimate_ic_ir(result, method='spearman', periods_per_year=12)
            series = qis.compute_ic_timeseries(result, method='spearman')
            monthly_ic[name] = series['1'].IC
            tables[f'{label}_pooled'] = result.pooled_universe
            tables[f'{label}_groups'] = result.per_group
            observed = pd.Series(scores.index, index=scores.index.to_period('M').to_timestamp('M'))
            for horizon, pairs in result.pairs.items():
                audit = pairs.copy()
                audit['signal_observation_date'] = (
                    audit.date - pd.offsets.MonthEnd(1)).map(observed)
                audit['forward_window_end'] = audit.date + pd.offsets.MonthEnd(int(horizon)-1)
                if (audit.signal_observation_date.isna().any()
                        or audit.forward_window_end.gt(last_month).any()
                        or audit.signal_observation_date.ge(audit.date).any()):
                    raise ValueError('Signal pairs contain an unknown or future observation')
                tables[f'{label}_pairs_{horizon}m'] = audit
                tables[f'{label}_ic_{horizon}m'] = series[horizon]
    ic_summary = pd.concat(summaries, names=['signal', 'horizon'])
    ic_series = pd.DataFrame(monthly_ic)
    tables.update(signal_rank_ic_summary=ic_summary, signal_monthly_rank_ic=ic_series)

    targets = {'Active': full.executed.targets,
               'Benchmark': pd.DataFrame(
                   [inputs.benchmark_weights] * len(full.executed.targets),
                   index=full.executed.targets.index)}
    breadth = {}
    for name, weights in targets.items():
        result = qis.compute_portfolio_breadth(
            risk.asset_excess_returns, weights, covar_dict=risk.covariances,
            cash_columns=['BIL'], position_threshold=1e-4)
        breadth[name] = result
        for field in ('metrics', 'availability', 'absolute_weight_shares',
                      'absolute_risk_contribution_shares', 'covariance_dates'):
            tables[f'breadth_{name.lower()}_{field}'] = getattr(result, field)
        tables[f'breadth_{name.lower()}targets'] = weights

    portfolios = report_portfolios(inputs, full)
    brinson, bridges = {}, {}
    for name, pair, is_net in (('Net', portfolios[:2], True),
                               ('Gross', portfolios[2:], False)):
        multi = qis.MultiPortfolioData(pair)
        result = multi.compute_brinson_attribution(
            group_data=inputs.universe.economic_group, freq=None, is_net=is_net,
            is_linked=True, is_exclude_interaction_term=False)
        if is_net:
            # Costs on near-empty sleeves make separate selection/interaction ill-conditioned.
            # QIS's canonical merge retains net active P&L; keep unmerged values as an audit.
            for field, table in zip(('totals', 'active', 'allocation', 'selection', 'interaction'),
                                    result, strict=True):
                tables[f'brinson_net_unmerged_{field}'] = table
            result = multi.compute_brinson_attribution(
                group_data=inputs.universe.economic_group, freq=None, is_net=True,
                is_linked=True, is_exclude_interaction_term=True)
        brinson[name] = result
        for field, table in zip(('totals', 'active', 'allocation', 'selection', 'interaction'),
                                result, strict=True):
            tables[f'brinson_{name.lower()}_{field}'] = table
        a, b = pair
        direct = a.nav / a.nav.iloc[0] - b.nav / b.nav.iloc[0]
        linked = sum(frame['Total Sum'].cumsum() for frame in result[2:])
        bridge = pd.DataFrame({'Linked effects': linked,
                              'Compounded active return': direct.reindex(linked.index)})
        bridge['Reconciliation error'] = bridge.iloc[:, 0] - bridge.iloc[:, 1]
        if bridge['Reconciliation error'].abs().max() > 1e-9:
            raise ValueError('Brinson attribution does not reconcile to portfolio NAVs')
        tables[f'brinson_{name.lower()}_reconciliation'] = bridge
        bridges[name] = bridge
    tables['brinson_cost_bridge'] = pd.DataFrame({
        'Net active': bridges['Net']['Linked effects'],
        'Gross active': bridges['Gross']['Linked effects'],
    })
    tables['brinson_cost_bridge']['Differential trading-cost effect'] = (
        tables['brinson_cost_bridge']['Net active'] - tables['brinson_cost_bridge']['Gross active'])
    metadata = {
        'first_decision': str(start.date()), 'last_decision': str(end.date()),
        'nav_end': str(full.navs.index[-1].date()), 'signal_variants': list(signals),
        'signal_contract': 'Completed ME BIL-relative log returns; one-period lag; '
                           'non-overlapping 1/3/6-month windows from first post-inception ME',
        'signal_inference': 'Pooled slopes use classical independent-pair SE; IC t is descriptive, '
                            'not HAC or a multiple-testing adjustment. No model selection.',
        'breadth_contract': 'Exact full FM covariances as-of actual target dates; BIL excluded; '
                            'strict 1bp absolute position threshold; absolute Euler risk shares',
        'brinson_contract': 'BHB, daily prior holdings, Frongello linked; '
                            'gross separate interaction; net selection includes interaction '
                            'to avoid cost-only sleeve instability; '
                            'unmerged net audit retained; first NAV is the baseline',
    }
    return InvestmentDiagnostics(signals, ic_series, ic_summary, breadth, brinson, tables, metadata)


def predictive_figure(result: qis.SignalDiagnosticsResult, name: str):
    """Use native QIS panels, marking horizons without twelve per-asset observations."""
    title = f'{name} | QIS forward-return diagnostics | horizons in months'
    betas = qis.compute_per_asset_betas(result, min_obs_per_asset=12)
    available = set(betas.loc[betas.group.notna(), 'horizon'])
    if set(result.horizon_labels) <= available:
        return qis.plot_signal_diagnostics(result, figsize=(15, 10), num_buckets=5, title=title)
    # QIS's composite currently raises when its per-asset beta frame is empty.
    # Native component plots preserve the minimum sample instead of weakening it.
    fig, axes = plt.subplots(2, len(result.horizon_labels), figsize=(15, 10), squeeze=False)
    fig.suptitle(title, fontsize=14)
    for i, horizon in enumerate(result.horizon_labels):
        qis.plot_signal_diagnostics_boxplot(result, horizon, num_buckets=5, ax=axes[0, i])
        if horizon in available:
            qis.plot_signal_diagnostics_group_boxplot(result, horizon, ax=axes[1, i])
        else:
            axes[1, i].set_title(f'{horizon}m | per-asset slopes', fontsize=11)
            axes[1, i].text(.5, .5, 'Insufficient history:\nrequires 12 observations per asset',
                            ha='center', va='center', transform=axes[1, i].transAxes)
            axes[1, i].set_axis_off()
    return fig


def signal_figures(study: InvestmentDiagnostics) -> list:
    """Compare score variants, then show QIS predictive boxplots for each scored horizon."""
    fig = report_page(
        'Predictive signal diagnostics', 'Public momentum | forward BIL-relative log returns',
        'QIS pairs lag each score by one ME period; 3m/6m windows do not overlap.\n'
        'IC is cross-sectional Spearman correlation; IC IR is annualised by sqrt(12 / horizon).\n'
        'Descriptive inference; pooled t-statistics are not HAC or selection-adjusted.', 1)
    grid = fig.add_gridspec(3, 1, left=.09, right=.94, bottom=.17, top=.86,
                           height_ratios=[2, 2, 1.7], hspace=.60)
    table = study.ic_summary[['n_dates', 'mean_IC', 'IC_IR_an', 'hit_rate']].copy()
    table.index = [f'{name} | {horizon}m' for name, horizon in table.index]
    for column in table:
        fmt = '{:.0f}' if column == 'n_dates' else '{:.2f}'
        table[column] = table[column].map(fmt.format)
    table.columns = ['Dates', 'Mean rank IC', 'IC IR p.a.', 'Hit rate']
    draw_table(table, fig.add_subplot(grid[0]), percent=False,
               col_widths=[2.5, .8, 1.2, 1.2, .9])
    qis.plot_time_series(study.ic_series, ax=fig.add_subplot(grid[1]),
                         title='One-month rank IC | all completed observations',
                         fontsize=8, x_date_freq='YS', date_format='%Y', x_rotation=0,
                         var_format='{:.1f}', legend_stats=qis.LegendStats.NONE)
    groups = pd.DataFrame({name: result.per_group.xs('1').IC_spearman
                           for name, result in study.signals.items()})
    ax = fig.add_subplot(grid[2])
    box = ax.get_position()
    ax.set_position([.22, box.y0, .72, box.height])
    qis.plot_heatmap(groups.rename(index=GROUP_LABELS), ax=ax, fontsize=8, x_rotation=0,
                     title='One-month pooled rank IC within each asset class', var_format='{:.2f}')
    figures = [fig]
    for name, result in study.signals.items():
        if name.endswith('scored'):
            fig = predictive_figure(result, name)
            fig.subplots_adjust(bottom=.17, top=.88, hspace=.62)
            for ax in fig.axes[len(result.horizon_labels):]:
                ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
                labels = [GROUP_LABELS.get(t.get_text(), t.get_text())
                          for t in ax.get_xticklabels()]
                ax.set_xticks(ax.get_xticks(), labels, rotation=20, ha='right', fontsize=9)
            fig.text(.03, .025, 'Pooled t-statistics assume independent pairs; use the IC '
                     'time series to assess stability. No performance-based parameter selection.',
                     fontsize=9)
            figures.append(fig)
    return figures


def breadth_figures(study: InvestmentDiagnostics) -> list:
    """Show target-date breadth versus the benchmark and the latest concentration curves."""
    figures = []
    for page, title in enumerate(('Breadth through time', 'Current breadth and concentration'), 1):
        fig = report_page(
            title, 'Active vs fixed benchmark | supplied FM/FL full covariance',
            'Target allocations; BIL excluded from all counts. Material absolute weight >1bp.\n'
            'Capital breadth: absolute weights. Risk breadth: absolute Euler contributions.\n'
            'Concentration measures; not forecast independence or additive risk shares.', page)
        if page == 1:
            grid = fig.add_gridspec(3, 1, left=.09, right=.94, bottom=.17, top=.86, hspace=.52)
            for i, metric in enumerate(('Invested assets', 'Effective capital assets',
                                         'Effective risk contributors')):
                data = pd.DataFrame({name: result.metrics[metric]
                                     for name, result in study.breadth.items()})
                qis.plot_time_series(data, ax=fig.add_subplot(grid[i]), title=metric,
                                     fontsize=8, x_date_freq='YS', date_format='%Y', x_rotation=0,
                                     var_format='{:.1f}', legend_stats=qis.LegendStats.NONE)
        else:
            grid = fig.add_gridspec(2, 2, left=.11, right=.95, bottom=.17, top=.83,
                                   height_ratios=[1.1, 1], hspace=.6, wspace=.5)
            axes = [fig.add_subplot(grid[0, 0]), fig.add_subplot(grid[0, 1])]
            qis.plot_portfolio_breadth_current_comparison(
                study.breadth, fontsize=6, detailed_mode=False, axs=axes)
            for ax, title in zip(axes, ('Breadth counts', 'Allocation efficiency'), strict=True):
                ax.set_title(title, fontsize=9)
            for i, (name, result) in enumerate(study.breadth.items()):
                ax = fig.add_subplot(grid[1, i])
                qis.plot_portfolio_breadth_concentration(
                    result, detailed_mode=False, fontsize=6, ax=ax)
                ax.set_title(f'{name} | {result.metrics.index[-1]:%d %b %Y}', fontsize=9)
        figures.append(fig)
    return figures


def brinson_figures(study: InvestmentDiagnostics) -> list:
    """Render explicit allocation, selection and interaction effects and a cost bridge."""
    figures = []
    for page, (basis, result) in enumerate(study.brinson.items(), 1):
        fig = report_page(
            f'Brinson attribution | {basis.lower()}',
            'Economic sleeves | active vs fixed benchmark',
            'BHB; daily prior holdings; Frongello linking to compounded active returns.\n'
            'Net selection includes interaction: near-empty sleeves make the cost split unstable.\n'
            'Gross keeps interaction separate. CSVs retain the unmerged net audit and cost bridge.',
            page)
        grid = fig.add_gridspec(3, 1, left=.09, right=.94, bottom=.17, top=.86,
                               height_ratios=[1.2, 2, 1.6], hspace=.60)
        totals, active, allocation, selection, interaction = result
        display = totals.rename(index=GROUP_LABELS).copy()
        display.columns = ['Active\navg weight', 'Benchmark\navg weight', 'Active\nreturn',
                           'Benchmark\nreturn', *display.columns[4:]]
        if basis == 'Net':
            display = display.rename(columns={'Instrument\nSelection': 'Selection +\ninteraction'})
        qis.plot_brinson_totals_table(display, ax=fig.add_subplot(grid[0]),
                                      var_format='{:.1%}', fontsize=7, first_row_height=.20)
        effects = pd.DataFrame({'Allocation': allocation['Total Sum'],
                                'Selection': selection['Total Sum'],
                                'Interaction': interaction['Total Sum']}).cumsum()
        if basis == 'Net':
            effects = effects.drop(columns='Interaction').rename(
                columns={'Selection': 'Selection + interaction'})
        effects['Total active'] = active.sum(axis=1).cumsum()
        qis.plot_time_series(effects, ax=fig.add_subplot(grid[1]),
                             title='Cumulative linked contributions', var_format='{:.0%}',
                             fontsize=8, x_date_freq='YS', date_format='%Y', x_rotation=0,
                             legend_stats=qis.LegendStats.NONE)
        if basis == 'Net':
            data = study.tables['brinson_cost_bridge']
            title = 'Net / same-target gross active return and differential cost effect'
        else:
            data = (allocation + selection + interaction).drop(columns='Total Sum').cumsum()
            title = 'Cumulative active contribution by economic sleeve'
        qis.plot_time_series(data, ax=fig.add_subplot(grid[2]), title=title, var_format='{:.0%}',
                             fontsize=8, x_date_freq='YS', date_format='%Y', x_rotation=0,
                             legend_stats=qis.LegendStats.NONE)
        figures.append(fig)
    return figures


def save_investment_reports(study: InvestmentDiagnostics, output_dir: Path | str) -> dict:
    """Export three QIS diagnostic reports with all input, result and reconciliation tables."""
    figures = {}
    existing_figures = set(plt.get_fignums())
    with plt.rc_context({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42}):
        try:
            figures['signal_diagnostics'] = signal_figures(study)
            figures['portfolio_breadth'] = breadth_figures(study)
            figures['brinson_attribution'] = brinson_figures(study)
            return write_report_bundle(output_dir, 'investment_diagnostics', figures,
                                        study.tables, study.metadata)
        finally:
            for number in set(plt.get_fignums()) - existing_figures:
                plt.close(number)
