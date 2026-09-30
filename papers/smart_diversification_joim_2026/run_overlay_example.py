"""Offline 60/40 overlay example using synthetic funds, OP optimization and qis reports.

Requires optimalportfolios >= 7.10.0 and qis >= 5.33.0. Select a Locals mode in the
run_local call at the bottom; edit the example settings at the start of run_local.
Run this file in the IDE, or from the repository root:
    python -m papers.smart_diversification_joim_2026.run_overlay_example

All observations are simulated monthly simple excess returns. Full-sample estimates
and illustrations are descriptive, not an out-of-sample investment track record.
No private replication module, manuscript, provider data or network is required.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import qis
import qis.regimes as rg
# qis re-exports neither column-name constant; the README records the tested qis version.
from qis.regimes.partition import REGIME_COLUMN
from qis.portfolio.attribution.portfolio_breadth import EFFECTIVE_CAPITAL_COUNT
import optimalportfolios as op
from optimalportfolios.local_path import get_output_path

CORE = '60/40 core'
Q = (0.0, 0.16, 0.84, 1.0)
AF = 12.0
DEFAULT_THETA_GRID = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)
INPUT_COLUMNS = ('sharpe', 'ann_vol', 'bear_sharpe', 'rho', 'beta_bear', 'beta_normal',
                 'beta_bull', 'beta_total', 'idio_vol', 'convexity_premium', 'cp_star',
                 'bear_return_pa')
OVERLAY_GROUPS = {'Trend bucket': ('Trend',), 'L/S bucket': ('Equity L/S', 'Market neutral'),
                  'Long-vol bucket': ('Tail hedge',)}


class Locals(Enum):
    """Independent estimation, single-allocation and coverage-frontier workflows."""
    ESTIMATE_INPUTS = 1
    RUN_SINGLE_OPTIMISATION = 2
    RUN_COVERAGE_FLOOR_FRONTIER = 3


@dataclass(frozen=True)
class OverlayInputs:
    """Full-precision monthly estimation snapshot consumed directly by the optimiser.

    statistics and betas supply the seven fund inputs in the paper's estimation
    inventory; regime_moments supply the shared benchmark inputs. covariance is
    annualised; expected returns and Bear contributions are annual decimal returns.
    sampled_returns preserves the single classification used by every calculation.
    Constructor inputs are copied. Frozen bindings do not make pandas cells immutable:
    treat the owned frames as read-only and create a new snapshot to change estimates.
    """
    statistics: pd.DataFrame
    betas: pd.DataFrame
    covariance: pd.DataFrame
    regime_moments: pd.DataFrame
    sampled_returns: pd.DataFrame

    def __post_init__(self) -> None:
        """Copy caller-owned frames and reject misaligned optimiser coefficients."""
        for name in ('statistics', 'betas', 'covariance', 'regime_moments', 'sampled_returns'):
            object.__setattr__(self, name, getattr(self, name).copy(deep=True))
        names = self.statistics.index
        if CORE not in names or len(names) < 2 or not names.is_unique:
            raise ValueError('Inputs require the core and uniquely named overlay funds.')
        if not all(names.equals(index) for index in (
                self.betas.index, self.covariance.index, self.covariance.columns,
                self.sampled_returns.drop(columns=REGIME_COLUMN).columns)):
            raise ValueError('Input statistics, betas, covariance and sample must be aligned.')
        if (not np.isfinite(self.input_sheet.to_numpy()).all()
                or not np.isfinite(self.covariance.to_numpy()).all()):
            raise ValueError('Optimiser inputs must be finite.')

    @property
    def input_sheet(self) -> pd.DataFrame:
        """Return the numerical input sheet, with no display rounding or percent scaling."""
        return self.statistics.join(self.betas).loc[:, list(INPUT_COLUMNS)]

    @property
    def expected_returns(self) -> pd.Series:
        """Infer annual decimal expected excess returns from Sharpe times volatility."""
        return (self.statistics['sharpe'] * self.statistics['ann_vol']).rename('expected_return')

    @property
    def bear_contributions(self) -> pd.Series:
        """Return annual decimal Bear contributions used in the coverage-floor row."""
        return self.statistics['bear_return_pa'].copy()

    @property
    def returns(self) -> pd.DataFrame:
        """Return the complete monthly excess-return panel used in this snapshot."""
        return self.sampled_returns.drop(columns=REGIME_COLUMN).copy()


@dataclass(frozen=True)
class OverlayResult:
    """Case columns, raw capital weights, realised statistics and a Table 4-style view.

    table uses percent of overlay budget for allocation/bucket rows, percentages for
    volatility, Bear contribution and realised coverage, and unscaled Sharpe/effective N.
    coverage_floors is in fractions; NaN denotes a reference or an unconstrained case.
    """
    weights: pd.DataFrame
    statistics: pd.DataFrame
    table: pd.DataFrame
    coverage_floors: pd.Series


@dataclass(frozen=True)
class ExampleParams:
    """Illustrative annual decimal-return assumptions and a monthly sample size."""
    months: int = 360
    seed: int = 17
    equity_weight: float = 0.60
    equity_excess_return: float = 0.065
    equity_vol: float = 0.18
    bond_excess_return: float = 0.02
    bond_vol: float = 0.07
    equity_bond_corr: float = -0.15
    overlay_budget: float = 1.0
    coverage: float = 0.40


def create_example_data(params: ExampleParams) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simulate a 60/40 core and four stylized funds; also return their annual assumptions."""
    if params.months < 120:
        raise ValueError('Use at least 120 months to estimate three regime regressions.')
    if not 0.0 < params.equity_weight < 1.0 or not -1.0 < params.equity_bond_corr < 1.0:
        raise ValueError('Core weight must lie in (0, 1) and correlation in (-1, 1).')
    if min(params.equity_vol, params.bond_vol, params.overlay_budget) <= 0.0:
        raise ValueError('Volatilities and overlay budget must be positive.')
    if not 0.0 <= params.coverage <= 1.0:
        raise ValueError('Coverage must be between zero and one.')
    rng = np.random.default_rng(params.seed)
    shocks = rng.standard_normal((params.months, 6))
    equity = params.equity_excess_return / AF + params.equity_vol / np.sqrt(AF) * shocks[:, 0]
    bond_shock = (params.equity_bond_corr * shocks[:, 0]
                  + np.sqrt(1.0 - params.equity_bond_corr ** 2) * shocks[:, 1])
    bonds = params.bond_excess_return / AF + params.bond_vol / np.sqrt(AF) * bond_shock
    # Constant-mix monthly core returns, on the same excess-return capital base.
    core = params.equity_weight * equity + (1.0 - params.equity_weight) * bonds
    z = (core - core.mean()) / core.std(ddof=1)
    # Stylized contemporaneous payoff shapes, not trading signals or actual funds.
    shapes = np.column_stack([
        0.65 * np.abs(z) + 0.65 * shocks[:, 2],
        0.70 * z + 0.70 * shocks[:, 3],
        0.08 * z + shocks[:, 4],
        -0.75 * z + 0.40 * np.maximum(-z, 0.0) + 0.25 * shocks[:, 5],
    ])
    assumptions = pd.DataFrame({
        'annual_excess_return': [0.045, 0.055, 0.035, -0.04],
        'annual_volatility': [0.10, 0.12, 0.07, 0.12],
    }, index=['Trend', 'Equity L/S', 'Market neutral', 'Tail hedge'])
    shapes = (shapes - shapes.mean(axis=0)) / shapes.std(axis=0, ddof=1)
    funds = (shapes * assumptions['annual_volatility'].to_numpy() / np.sqrt(AF)
             + assumptions['annual_excess_return'].to_numpy() / AF)
    dates = pd.date_range('1996-01-31', periods=params.months, freq='ME')
    panel = pd.DataFrame(funds, index=dates, columns=assumptions.index)
    panel.insert(0, CORE, core)
    return panel, assumptions


def estimate_inputs(panel: pd.DataFrame) -> OverlayInputs:
    """Estimate the paper's inventory once through qis, retaining full precision."""
    if (not isinstance(panel.index, pd.DatetimeIndex) or not panel.index.is_unique
            or not panel.index.is_monotonic_increasing):
        raise ValueError('Monthly returns require a unique, increasing DatetimeIndex.')
    if not panel.columns.is_unique or CORE not in panel or len(panel.columns) < 2:
        raise ValueError('Supply the core and uniquely named overlay funds.')
    if not np.isfinite(panel.to_numpy(dtype=float)).all():
        raise ValueError('Choose a complete finite common sample before estimating inputs.')
    panel = panel.loc[:, [CORE] + list(panel.columns.drop(CORE))]
    sampled = rg.create_sampled_returns_with_regime_id(panel, benchmark=CORE, q=Q)
    betas = rg.compute_regime_betas(sampled, benchmark=CORE, af=AF)
    covariance = rg.compute_regime_mixture_covar_from_sample(
        sampled, benchmark=CORE, af=AF, betas=betas)
    statistics = rg.compute_regime_premium_table(sampled, benchmark=CORE, af=AF, q=Q)
    betas.loc[CORE, ['beta_bear', 'beta_normal', 'beta_bull', 'beta_total']] = 1.0
    betas.loc[CORE, 'idio_vol'] = 0.0
    moments = rg.compute_sample_regime_moments(sampled, benchmark=CORE)
    return OverlayInputs(statistics, betas.reindex(panel.columns), covariance, moments, sampled)


def solve_overlay(inputs: OverlayInputs, budget: float = 1.0,
                  coverage: float | None = 0.40) -> pd.Series:
    """Maximize the stacked Sharpe ratio with a fixed core and optional Bear-loss coverage."""
    if not np.isfinite(budget) or budget <= 0.0 or (
            coverage is not None and not 0.0 <= coverage <= 1.0):
        raise ValueError('Budget must be positive and coverage, if supplied, in [0, 1].')
    means = inputs.expected_returns
    names = means.index
    minimum = pd.Series(0.0, index=names)
    maximum = pd.Series(budget, index=names)
    minimum[CORE] = maximum[CORE] = 1.0
    exposure = 1.0 + budget
    coefficients = inputs.bear_contributions
    floor = None
    if coverage is not None:
        if coefficients[CORE] >= 0.0:
            raise ValueError('Coverage requires a negative benchmark Bear contribution.')
        floor = (1.0 - coverage) * coefficients[CORE]
        best_possible = coefficients[CORE] + budget * coefficients.drop(CORE).max()
        if best_possible < floor - 1e-10:
            raise ValueError('Requested coverage is infeasible for these funds and budget.')
    linear = None if floor is None else op.LinearConstraints(
        loadings=coefficients.to_frame('bear_coverage'),
        lower=pd.Series({'bear_coverage': floor}),
    )
    constraints = op.Constraints(
        is_long_only=True, min_weights=minimum, max_weights=maximum,
        min_exposure=exposure, max_exposure=exposure,
        linear_constraints=linear,
    )
    outcome = op.cvx_maximize_portfolio_sharpe(
        covar=inputs.covariance.loc[names, names].to_numpy(), means=means.to_numpy(),
        constraints=constraints,
    )
    if not outcome.accepted:
        raise RuntimeError(f'OP rejected the allocation: {outcome.status}; {outcome.reason}')
    weights = pd.Series(outcome.weights, index=names)
    np.testing.assert_allclose(weights[CORE], 1.0, atol=1e-6)
    np.testing.assert_allclose(weights.drop(CORE).sum(), budget, atol=1e-6)
    if (weights < -1e-6).any() or (weights > maximum + 1e-6).any():
        raise AssertionError('OP weights violate an asset bound.')
    if floor is not None and coefficients @ weights < floor - 1e-6:
        raise AssertionError('OP weights violate the requested Bear contribution floor.')
    return weights


def _coverage_label(theta: float) -> str:
    """Label a fractional requested coverage without rounding it to integer percent."""
    return f'theta={100.0 * theta:.12g}%'


def _summarise_allocations(inputs: OverlayInputs, weights: pd.DataFrame,
                           floors: pd.Series, budget: float) -> OverlayResult:
    """Arrange allocation and native qis statistics in the paper's Table 4 orientation."""
    weights = weights.copy()
    benchmark = pd.Series(0.0, index=weights.index)
    benchmark[CORE] = 1.0
    weights.insert(0, 'Benchmark', benchmark)
    # Static constant-mix sample returns, not a sequential trading backtest.
    sampled = inputs.returns @ weights
    sampled[REGIME_COLUMN] = inputs.sampled_returns[REGIME_COLUMN]
    statistics = rg.compute_regime_premium_table(sampled, benchmark='Benchmark', af=AF, q=Q)
    statistics['bear_loss_coverage'] = (
        1.0 - statistics['bear_return_pa'] / inputs.bear_contributions[CORE])
    funds = weights.index.drop(CORE)
    table = 100.0 * weights.loc[funds] / budget
    for label, members in OVERLAY_GROUPS.items():
        present = funds.intersection(members)
        if len(present):
            table.loc[label] = 100.0 * weights.loc[present].sum() / budget
    table.loc['Realized SR_Total'] = statistics['sharpe']
    table.loc['Realized SR_Bear'] = statistics['bear_sharpe']
    # Use qis's inverse-Herfindahl capital breadth on each static end-of-sample snapshot.
    date = inputs.returns.index[-1]
    covariance = {date: inputs.covariance.loc[funds, funds]}
    effective_n = {'Benchmark': np.nan}
    for name in weights.columns.drop('Benchmark'):
        targets = weights.loc[funds, [name]].T.clip(lower=0.0)
        targets.index = pd.DatetimeIndex([date])
        breadth = qis.compute_portfolio_breadth(
            inputs.returns[funds], targets, covar_dict=covariance, position_threshold=0.0)
        effective_n[name] = breadth.metrics.loc[date, EFFECTIVE_CAPITAL_COUNT]
    table.loc['Effective N'] = pd.Series(effective_n)
    for label, column in [('Volatility sigma_P', 'ann_vol'),
                           ('Bear-regime contribution p.a.', 'bear_return_pa'),
                           ('Realized coverage theta', 'bear_loss_coverage')]:
        table.loc[label] = 100.0 * statistics[column]
    return OverlayResult(weights, statistics, table,
                         floors.reindex(weights.columns).rename('requested_coverage'))


def run_single_optimisation(inputs: OverlayInputs, coverage: float | None = 0.40,
                            budget: float = 1.0) -> OverlayResult:
    """Solve exactly one case from a supplied estimation snapshot, with a core reference."""
    label = 'No floor' if coverage is None else _coverage_label(coverage)
    weights = solve_overlay(inputs, budget=budget, coverage=coverage).to_frame(label)
    return _summarise_allocations(inputs, weights, pd.Series({label: coverage}), budget)


def run_coverage_floor_frontier(inputs: OverlayInputs,
                                theta_grid: tuple[float, ...] | list[float] = DEFAULT_THETA_GRID,
                                budget: float = 1.0) -> OverlayResult:
    """Solve a theta grid with benchmark, equal-weight and no-floor reference columns.

    Every grid value is an enforced floor, including theta=0. No floor is a separate
    unconstrained reference. An infeasible case raises instead of returning a fallback.
    All cases reuse exactly the same full-precision input snapshot.
    """
    grid = np.asarray(theta_grid, dtype=float)
    if (grid.ndim != 1 or not len(grid) or not np.isfinite(grid).all()
            or (grid < 0.0).any() or (grid > 1.0).any() or len(np.unique(grid)) != len(grid)):
        raise ValueError('The theta grid must contain unique finite fractions in [0, 1].')
    labels = [_coverage_label(theta) for theta in grid]
    if len(set(labels)) != len(labels):
        raise ValueError('The theta grid contains indistinguishably close display labels.')
    no_floor = solve_overlay(inputs, budget=budget, coverage=None)
    equal = pd.Series(budget / (len(no_floor) - 1), index=no_floor.index)
    equal[CORE] = 1.0
    allocations = {'Equal weight': equal, 'No floor': no_floor}
    for label, theta in zip(labels, grid):
        try:
            allocations[label] = solve_overlay(inputs, budget=budget, coverage=float(theta))
        except (ValueError, RuntimeError) as error:
            raise type(error)(f'{label}: {error}') from error
    return _summarise_allocations(inputs, pd.DataFrame(allocations),
                                  pd.Series(dict(zip(labels, grid))), budget)


def create_illustrations(panel: pd.DataFrame, weights: pd.DataFrame,
                         output: Path, params: ExampleParams) -> None:
    """Draw native qis regime, allocation and smart-diversification exhibits."""
    funds = panel.drop(columns=CORE)
    sleeves = funds @ (weights.drop(index=CORE) / params.overlay_budget)
    overlays = pd.concat([funds, sleeves], axis=1)
    returns = pd.concat([panel[CORE], overlays], axis=1)
    # Supply a NAV anchor before the first observation so no simulated month is lost.
    anchor = returns.index[0] - pd.offsets.MonthEnd(1)
    returns.loc[anchor] = 0.0
    navs = qis.returns_to_nav(returns.sort_index(), init_period=None, init_value=100.0,
                              is_log_returns=False)
    report = qis.SmartDiversificationReport(
        principal_nav=navs[CORE], overlay_navs=navs.drop(columns=CORE),
        regime_classifier=qis.BenchmarkReturnsQuantilesRegime(freq='ME', q=np.array(Q)),
        perf_params=qis.PerfParams(freq='ME', freq_vol='ME', freq_reg='ME',
                                   sharpe_convention=qis.SharpeConvention.ARITHMETIC),
        benchmark_description=CORE,
    )
    fig, ax = plt.subplots(figsize=(11, 6), layout='constrained')
    report.plot_smart_diversification_curve(
        rebalancing_freq='ME', principal_weight=1.0, is_principal_weight_fixed=True,
        constraints={name: params.overlay_budget for name in overlays},
        x_var=qis.PerfStat.BEAR_SHARPE, y_var=qis.PerfStat.SHARPE_ARITH,
        title='Synthetic overlays on a 60/40 core (in-sample)',
        xlabel='Bear-regime arithmetic Sharpe contribution',
        ylabel='Arithmetic excess Sharpe ratio', ax=ax, legend_loc='upper right',
        shifts={'No floor': True, weights.columns[-1]: False,
                'Market neutral': True, 'Equity L/S': False},
        x_limits=(-0.95, 0.60), y_limits=(-0.20, 0.70),
    )
    fig.savefig(output / 'smart_diversification.png', dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(11, 6), layout='constrained')
    report.plot_conditional_sharpes(
        title='Synthetic standalone funds and optimized sleeves', ax=ax,
    )
    fig.savefig(output / 'regime_contributions.png', dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(9, 5), layout='constrained')
    qis.plot_bars(df=weights.drop(index=CORE).T, stacked=True, yvar_format='{:.0%}',
                  x_rotation=0, ylabel='Overlay exposure / capital',
                  title='OP allocations; 60/40 core held at 100%', ax=ax)
    fig.savefig(output / 'overlay_weights.png', dpi=160)
    plt.close(fig)


def _export_inputs(inputs: OverlayInputs, output: Path) -> None:
    """Export the full-precision estimation inventory and a separate display-only sheet."""
    output.mkdir(parents=True, exist_ok=True)
    display = inputs.input_sheet
    percent_columns = ['ann_vol', 'idio_vol', 'bear_return_pa']
    display[percent_columns] *= 100.0
    display = display.round({name: 1 if name in percent_columns else 2 for name in display})
    for filename, frame in {
        'monthly_excess_returns': inputs.returns, 'regime_statistics': inputs.statistics,
        'regime_betas': inputs.betas, 'regime_moments': inputs.regime_moments,
        'regime_covariance': inputs.covariance, 'allocator_input_sheet': inputs.input_sheet,
        'allocator_input_sheet_display': display, 'expected_returns': inputs.expected_returns,
    }.items():
        frame.to_csv(output / f'{filename}.csv')


def create_frontier_illustrations(result: OverlayResult, output: Path) -> None:
    """Plot the requested frontier and its allocations through native qis plot functions."""
    floors = result.coverage_floors.dropna().sort_values()
    metrics = result.table.loc[['Realized SR_Total', 'Realized SR_Bear', 'Effective N'],
                               floors.index].T
    metrics.index = pd.Index(floors.to_numpy(), name='Coverage floor theta')
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), layout='constrained')
    qis.plot_line(metrics[['Realized SR_Total', 'Realized SR_Bear']], ax=axes[0],
                  xlabel='Requested coverage floor', xvar_format='{:.0%}',
                  ylabel='Arithmetic excess Sharpe contribution',
                  legend_labels=['Total Sharpe', 'Bear contribution'], legend_loc='center left',
                  title='Synthetic coverage-floor frontier (in-sample)')
    qis.plot_line(metrics[['Effective N']], ax=axes[1], xvar_format='{:.0%}',
                  xlabel='Requested coverage floor', ylabel='Effective number of overlays')
    fig.savefig(output / 'coverage_floor_frontier.png', dpi=160)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(10, 5), layout='constrained')
    qis.plot_bars(result.weights.loc[result.weights.index.drop(CORE), floors.index].T,
                  stacked=True, yvar_format='{:.0%}', x_rotation=0,
                  legend_loc='upper left', bbox_to_anchor=(1.01, 1.0),
                  ylabel='Overlay exposure / capital', title='Allocations by coverage floor', ax=ax)
    fig.savefig(output / 'coverage_floor_weights.png', dpi=160)
    plt.close(fig)


def create_allocation_frontier_illustration(inputs: OverlayInputs, result: OverlayResult,
                                           output: Path, params: ExampleParams) -> None:
    """Draw the Figure 4 view through qis, using realised core-plus-overlay statistics."""
    panel = inputs.returns
    funds = panel.columns.drop(CORE)
    # Static monthly constant-mix stacks on the same capital base as the optimiser.
    stacks = panel[funds].mul(params.overlay_budget).add(panel[CORE], axis=0)
    sampled = pd.concat([panel[CORE].rename('Benchmark'), stacks], axis=1)
    sampled[REGIME_COLUMN] = inputs.sampled_returns[REGIME_COLUMN]
    stack_stats = rg.compute_regime_premium_table(sampled, benchmark='Benchmark', af=AF, q=Q)
    styles = {
        'Equal weight': dict(marker='P', color='#999999', s=150),
        'No floor': dict(marker='*', color='#009E73'),
        _coverage_label(params.coverage): dict(marker='D', color='#E69F00'),
    }
    highlights = {name: style for name, style in styles.items() if name in result.statistics.index}
    references = result.statistics.loc[['Benchmark', *highlights]]
    points = pd.concat([stack_stats.loc[funds], references])
    groups = pd.Series('Allocations', index=points.index)
    for group, members in OVERLAY_GROUPS.items():
        groups.loc[funds.intersection(members)] = group
    groups.loc['Benchmark'] = 'Benchmark'
    floors = result.coverage_floors.dropna()
    frontier = result.statistics.loc[floors.index].copy()
    frontier.insert(0, 'theta', floors)
    points.to_csv(output / 'overlay_allocation_points.csv')
    frontier.to_csv(output / 'overlay_allocation_frontier.csv')
    fig, ax = plt.subplots(figsize=(10, 6), layout='constrained')
    qis.plot_overlay_allocation_frontier(
        portfolio_stats=points, frontier_stats=frontier if len(frontier) > 1 else None,
        groups=groups, benchmark='Benchmark', highlights=highlights,
        group_styles={'Benchmark': dict(label=CORE)},
        label_offsets={'Market neutral': (6, -12)},
        title=f'Synthetic core + {params.overlay_budget:.0%} overlays (in-sample)',
        xlabel='Bear-regime arithmetic Sharpe contribution',
        ylabel='Arithmetic excess Sharpe ratio', ax=ax)
    ax.margins(x=0.1)
    fig.savefig(output / 'overlay_allocation_frontier.png', dpi=160)
    plt.close(fig)


def run_local(local: Locals) -> OverlayInputs | OverlayResult:
    """Run the selected example; edit the settings below for local exploration."""
    params = ExampleParams(coverage=0.40, overlay_budget=1.0)
    theta_grid = DEFAULT_THETA_GRID

    if not isinstance(local, Locals):
        raise ValueError('local must be a Locals enum member.')
    output = Path(get_output_path()) / 'joim-overlay-example' / local.name.lower()
    panel, assumptions = create_example_data(params)
    inputs = estimate_inputs(panel)
    _export_inputs(inputs, output)
    assumptions.to_csv(output / 'overlay_assumptions.csv')
    print(f'Generated outputs: {output}')

    if local == Locals.ESTIMATE_INPUTS:
        print('Allocator input sheet (decimal annual units):')
        print(inputs.input_sheet.to_string(float_format='{:.6f}'.format))
        print('\nBenchmark regime moments (periodic units):\n' + inputs.regime_moments.to_string())
        return inputs

    elif local == Locals.RUN_SINGLE_OPTIMISATION:
        result = run_single_optimisation(inputs, coverage=params.coverage,
                                         budget=params.overlay_budget)
        create_illustrations(panel, result.weights.drop(columns='Benchmark'), output, params)
        prefix = 'single_optimisation'

    elif local == Locals.RUN_COVERAGE_FLOOR_FRONTIER:
        result = run_coverage_floor_frontier(inputs, theta_grid, budget=params.overlay_budget)
        create_frontier_illustrations(result, output)
        prefix = 'coverage_floor_frontier'

    create_allocation_frontier_illustration(inputs, result, output, params)
    print('Table 4 layout (allocations as % of overlay budget; return-unit rows as %):')
    print(result.table.round(3).to_string(na_rep='--'))
    result.table.to_csv(output / f'{prefix}.csv')
    result.weights.to_csv(output / f'{prefix}_weights.csv')
    result.statistics.to_csv(output / f'{prefix}_statistics.csv')
    result.coverage_floors.to_csv(output / 'coverage_floors.csv')
    return result


if __name__ == '__main__':
    run_local(local=Locals.RUN_COVERAGE_FLOOR_FRONTIER)
