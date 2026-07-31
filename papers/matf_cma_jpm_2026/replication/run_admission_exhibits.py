"""
Admission-dial, scenario, and governed-dial exhibits (roadmap Stages J4c, J4e, J4f).

Everything that sweeps the admitted-alpha channel and re-solves. The dial is
the scalar s of the main text: the admitted channel enters the excess CMA as
s * w_i * alpha_i, so s = 0 is the pure market-return book and s = 1 is the
production policy. The factor-implied component never scales, which is the
whole point of the exhibit family.

  admission_dial.PNG        fig:admission_dial — alternatives sleeve weights
      and the claimed ex-ante excess Sharpe ratio across s, on the Balanced
      with Alternatives mandate under the box and tracking-error guardrails.
  admission_dial_nobox.PNG  fig:admission_nobox — the same sweep with the
      guardrails removed (long-only, fully invested, at the benchmark
      volatility), which is what the admission does when nothing contains it.
  scenario_admission.PNG    fig:scenario_admission — the base-optimized book at
      each s repriced under the 2022 stress and 2023 upside factor shocks.
      Admitted alpha does not shock, so the whole band shifts up with s while
      its width stays put.
  dial_sweeps.PNG           fig:dial_sweeps (main text, new) — per-sleeve
      admission sweeps for Gold, Insurance-Linked and Private Equity: the
      weight response and the claim response side by side, one sleeve moving
      at a time with the others held at the production policy.
  sleeve_tornado.PNG        fig:sleeve_tornado (Appendix E) — the one-at-a-time
      ranking of how far each sleeve's admission moves the claimed Sharpe.
  governed_dial.PNG         fig:governed_dial (Appendix E, new) — the admissible
      range of each sleeve's admission weight under BOTH governance
      constraints: the Cap 3 portfolio budget at the kappa grid and the B13
      one-sigma benchmark stress floor.

The B13 floor (owner decision O-J5, one-sigma form). For every mandate m the
stress-repriced expected TOTAL return of the mandate book must satisfy

    stress_total_m >= mu_bm,m - n * sigma_bm,m,      n = FLOOR_SIGMA_MULTIPLE

with mu_bm,m the benchmark expected total return under the base CMAs and
sigma_bm,m the benchmark volatility from the risk model. The rule is
anchor-invariant: r_f appears on both sides and cancels, so it is implemented
in excess space and r_f is added only for display. No floor value is hand-set
anywhere; n is one constant in one place.

The floor table reports headroom under BOTH admission policies: the PRODUCTION
policy w_paper (PE recut to 0.5, the vector every other R3 exhibit uses) and the
PRE-RECUT WORKBOOK policy w_workbook (PE at 1.0). Owner ruling O-J11 settles the
naming — the manuscript convention wins, so w_paper is "production" and
w_workbook is never labelled production anywhere (B6 as reworded in roadmap v1.4).

Units: decimals per annum internally, percent in the exhibits and the printed
tables. Sharpe ratios are EXCESS Sharpe ratios and are dimensionless.
Main entry point: run_local_test(local_test).

Does not belong here: the mandate frontier family (run_mandate_exhibits.py),
the consistency pair (run_consistency_exhibits.py), and the Cap 3 projection
mathematics (governed_cma_projection.py).
"""
# packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple
# qis / project
import exhibit_style as es
from local_path import load_cma_data
from governed_cma_projection import (SNAPSHOT,
                                     KAPPA_GRID,
                                     load_paper_inputs,
                                     compute_sharpe_accounting)
from run_optimisation import (MANDATE, BAND, TE_CONSTRAINT, build_moments, get_benchmark,
                             solve_mandate, solve_max_return_at_vol, report_portfolio)

_cma_data = load_cma_data()

DIAL_GRID = np.linspace(0.0, 1.5, 31)        # admission scale s, production policy at s = 1
PRODUCTION_SCALE = 1.0
FLOOR_SIGMA_MULTIPLE = 1.0                   # n of the B13 one-sigma rule; one constant, one place
SWEEP_SLEEVES: Dict[str, str] = {'BCOMGCTR Index': 'Gold',
                                 'EHFI804 Index': 'Insurance-Linked',
                                 'MP503001 Index': 'Private Equity'}
TORNADO_SLEEVES = ['MP503001 Index', 'MP503008 Index', 'MP503009 Index',
                   'EHFI804 Index', 'HFRIFWI Index', 'BCOMGCTR Index']
SLEEVE_GRID = np.linspace(0.0, 1.0, 21)      # per-sleeve admission weight grid
ADMISSION_POLICIES = ('w_paper', 'w_workbook')

# R2 printed anchors, reported beside the regenerated values (never asserted)
R2_DIAL_SHARPE: Dict[float, float] = {0.0: 0.30, 1.0: 0.35, 1.5: 0.39}
R2_STRESS_SIGN_FLIP: Tuple[float, float] = (-0.003, 0.003)     # excess return at s = 0 and s = 1
R2_SCENARIO_BAND_WIDTH = 0.052


# --------------------------------------------------------------------------
# the dial sweep
# --------------------------------------------------------------------------

def sweep_admission_dial(inputs,
                         mandate: str = MANDATE,
                         grid: np.ndarray = DIAL_GRID,
                         apply_guardrails: bool = True,     # False: long-only at the benchmark vol
                         ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """re-solve the mandate at every admission scale s; return the weight path and the statistics."""
    benchmark = get_benchmark(inputs=inputs, mandate=mandate)
    covar, _, rf_rate = build_moments(inputs=inputs)
    benchmark_vol = float(np.sqrt(benchmark @ covar.values @ benchmark))

    weights, stats = {}, {}
    for scale in grid:
        _, mu_x, _ = build_moments(inputs=inputs, admission_scale=float(scale))
        if apply_guardrails:
            book = solve_mandate(covar=covar, cmas=mu_x, benchmark_weights=benchmark)
        else:
            book = solve_max_return_at_vol(covar=covar, cmas=mu_x, vol_target=benchmark_vol)
            if book is None:
                raise ValueError(f"unconstrained solve infeasible at s = {scale!r}")
        weights[float(scale)] = book
        stats[float(scale)] = report_portfolio(
            weights=book, covar=covar, cmas=mu_x, rf_rate=rf_rate, inputs=inputs,
            benchmark_weights=benchmark if apply_guardrails else None)
    weight_path = pd.DataFrame(weights).T
    weight_path.index.name = 'admission_scale'
    statistics = pd.DataFrame(stats).T
    statistics.index.name = 'admission_scale'
    statistics.attrs['benchmark_vol'] = benchmark_vol
    return weight_path, statistics


def plot_admission_dial(weight_path: pd.DataFrame,
                        statistics: pd.DataFrame,
                        inputs,
                        title: str,
                        figsize: Tuple[float, float] = (12.6, 4.8),
                        ) -> plt.Figure:
    """the frozen two-panel grammar: alternatives weight paths left, claimed Sharpe right."""
    alternatives = inputs.assets.index[inputs.assets['asset_class'] == 'Alternatives']
    sleeves = inputs.assets.loc[alternatives, 'sleeve']
    colors = [es.BLUE, es.LIGHT_BLUE, '#2ca02c', es.ORANGE, es.DARK_RED, '#bcbd22']

    fig, axs = plt.subplots(1, 2, figsize=figsize)

    ax = axs[0]
    for color, ticker in zip(colors, alternatives):
        ax.plot(weight_path.index, 1e2 * weight_path[ticker], color=color, lw=1.8,
                label=sleeves[ticker], zorder=3)
    ax.axvline(PRODUCTION_SCALE, color='0.15', lw=1.2, ls='--', zorder=2)
    ax.annotate('production policy', xy=(PRODUCTION_SCALE, 1.0),
                xytext=(4, -12), textcoords='offset points',
                xycoords=('data', 'axes fraction'),
                va='top', ha='left', fontsize=8.4, color='0.25')
    ax.set_xlabel('Admission scale $s$ on the historical-alpha channel', fontsize=9.8)
    ax.set_ylabel('Optimal sleeve weight (%)', fontsize=9.8)
    ax.set_title('Alternatives weights across the admission dial', fontsize=10.5, loc='left')
    es.style_axis(ax=ax, grid_axis='both', fontsize=9.0)
    ax.legend(fontsize=8.4, loc='upper left', frameon=False)

    ax = axs[1]
    ax.plot(statistics.index, statistics['excess_sharpe'], color=es.BLUE, lw=2.2, zorder=3)
    ax.axvline(PRODUCTION_SCALE, color='0.15', lw=1.2, ls='--', zorder=2)
    for scale, reference in R2_DIAL_SHARPE.items():
        if scale in statistics.index:
            value = float(statistics.loc[scale, 'excess_sharpe'])
            ax.annotate(f"{value:.2f}", xy=(scale, value), xytext=(4, -12),
                        textcoords='offset points', fontsize=8.8, color='0.2')
    ax.set_xlabel('Admission scale $s$', fontsize=9.8)
    ax.set_ylabel('Claimed ex-ante excess Sharpe ratio', fontsize=9.8)
    ax.set_title('Claimed Sharpe rises with the dial the committee sets',
                 fontsize=10.5, loc='left')
    es.style_axis(ax=ax, grid_axis='both', fontsize=9.0)

    fig.suptitle(title, fontsize=12.0, x=0.008, ha='left')
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    return fig


# --------------------------------------------------------------------------
# scenario repricing and the B13 floor
# --------------------------------------------------------------------------

def reprice_under_scenarios(weights: pd.Series, inputs) -> pd.Series:
    """base, stress and upside EXCESS returns of a fixed book; evaluation-only, no re-solve."""
    assets = inputs.assets
    base = assets['factor_excess_cma'] + assets['w_paper'] * assets['alpha']
    out = {'base': float(base @ weights)}
    for scenario in ('stress', 'upside'):
        bump = inputs.factor_premia_scenarios[scenario]
        repriced = base + inputs.betas.values @ bump.values
        out[scenario] = float(repriced @ weights)
    return pd.Series(out)


def sweep_scenario_admission(inputs,
                             mandate: str = MANDATE,
                             grid: np.ndarray = DIAL_GRID,
                             ) -> pd.DataFrame:
    """the base-optimized book at each s, repriced under both scenarios (evaluation-only)."""
    benchmark = get_benchmark(inputs=inputs, mandate=mandate)
    covar, _, rf_rate = build_moments(inputs=inputs)
    assets = inputs.assets
    rows = {}
    for scale in grid:
        _, mu_x, _ = build_moments(inputs=inputs, admission_scale=float(scale))
        book = solve_mandate(covar=covar, cmas=mu_x, benchmark_weights=benchmark)
        # the admitted channel carries into every scenario unchanged, scaled by s
        admitted = float((scale * assets['w_paper'] * assets['alpha']) @ book)
        factor_implied = float(assets['factor_excess_cma'] @ book)
        row = {'base': factor_implied + admitted, 'admitted_carry': admitted}
        for scenario in ('stress', 'upside'):
            bump = inputs.factor_premia_scenarios[scenario]
            row[scenario] = factor_implied + float((inputs.betas.values @ bump.values) @ book) + admitted
        rows[float(scale)] = row
    table = pd.DataFrame(rows).T
    table.index.name = 'admission_scale'
    table['band_width'] = table['upside'] - table['stress']
    table.attrs['rf_rate'] = rf_rate
    return table


def build_scenario_floor_table(inputs,
                               sigma_multiple: float = FLOOR_SIGMA_MULTIPLE,
                               ) -> pd.DataFrame:
    """the B13 one-sigma floor for all eight mandates, under both admission policies.

    Implemented in EXCESS space, which is where the rule is anchor-invariant:
    stress_excess >= mu_bm_excess - n * sigma_bm. The reference cash rate is
    added to both sides only for display, so the headroom column is identical
    in excess and total space.
    """
    if sigma_multiple <= 0.0:
        raise ValueError(f"floor sigma multiple must be positive, got {sigma_multiple!r}")
    covar, _, rf_rate = build_moments(inputs=inputs)
    assets = inputs.assets
    rows = {}
    for mandate in _cma_data.MANDATES:
        benchmark = get_benchmark(inputs=inputs, mandate=mandate)
        base_bm = assets['factor_excess_cma'] + assets['w_paper'] * assets['alpha']
        mu_bm_excess = float(base_bm @ benchmark)
        sigma_bm = float(np.sqrt(benchmark @ covar.values @ benchmark))
        floor_excess = mu_bm_excess - sigma_multiple * sigma_bm
        row = {'mu_bm_total': mu_bm_excess + rf_rate,
               'sigma_bm': sigma_bm,
               'floor_total': floor_excess + rf_rate}
        for policy in ADMISSION_POLICIES:
            _, mu_x, _ = build_moments(inputs=inputs, admission_scale=1.0,
                                       admission_weights=assets[policy])
            book = solve_mandate(covar=covar, cmas=mu_x, benchmark_weights=benchmark)
            admitted = assets[policy] * assets['alpha']
            base = assets['factor_excess_cma'] + admitted
            stress = base + inputs.betas.values @ inputs.factor_premia_scenarios['stress'].values
            stress_excess = float(stress @ book)
            row[f"stress_total_{policy}"] = stress_excess + rf_rate
            row[f"headroom_{policy}"] = stress_excess - floor_excess
            row[f"binds_{policy}"] = 'BINDS' if stress_excess < floor_excess else 'slack'
        rows[mandate] = row
    table = pd.DataFrame(rows).T
    table.attrs['sigma_multiple'] = sigma_multiple
    table.attrs['rf_rate'] = rf_rate
    return table


def plot_scenario_admission(table: pd.DataFrame,
                            figsize: Tuple[float, float] = (10.4, 5.6),
                            ) -> plt.Figure:
    """the frozen grammar: three lines with the band shaded, and the sign-flip annotation."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.fill_between(table.index, 1e2 * table['stress'], 1e2 * table['upside'],
                    color=es.BLUE, alpha=0.10, zorder=1)
    for column, color, label in (('upside', es.GREEN, '2023 upside'),
                                 ('base', es.BLUE, 'Base'),
                                 ('stress', es.DARK_RED, '2022 stress')):
        ax.plot(table.index, 1e2 * table[column], color=color, lw=2.2, label=label, zorder=3)
    ax.axhline(0.0, color='0.4', lw=0.8, zorder=2)
    ax.axvline(PRODUCTION_SCALE, color='0.15', lw=1.2, ls='--', zorder=2)
    ax.annotate('production\npolicy', xy=(PRODUCTION_SCALE + 0.03, 1e2 * table['stress'].min()),
                va='bottom', ha='left', fontsize=8.6, color='0.25')

    first, last = table.index[0], table.index[-1]
    improvement = 1e2 * (table.loc[last, 'stress'] - table.loc[first, 'stress'])
    carry = 1e2 * (table.loc[last, 'admitted_carry'] - table.loc[first, 'admitted_carry'])
    ax.annotate(f"stress case improves {improvement:.1f}% from $s=0$ to $s={last:.1f}$;\n"
                f"admitted-alpha carry accounts for {carry:.1f}% of it",
                xy=(last, 1e2 * table.loc[last, 'stress']),
                xytext=(-0.72, 1e2 * table['stress'].min() + 0.45),
                textcoords='data', fontsize=8.8, color='0.2',
                arrowprops=dict(arrowstyle='->', color='0.35', lw=0.9))

    ax.set_xlabel('Admission scale $s$ on the historical-alpha channel', fontsize=10.0)
    ax.set_ylabel('Balanced portfolio excess return (%)', fontsize=10.0)
    es.style_axis(ax=ax, grid_axis='both', fontsize=9.0)
    ax.legend(fontsize=9.0, loc='upper left', frameon=False)
    ax.set_title('Admitted alpha does not shock: rising admission flatters the stress case '
                 'one-for-one', fontsize=11.5, loc='left')
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# per-sleeve sweeps: dial_sweeps, sleeve_tornado, governed_dial
# --------------------------------------------------------------------------

def sweep_single_sleeve(inputs,
                        ticker: str,
                        mandate: str = MANDATE,
                        grid: np.ndarray = SLEEVE_GRID,
                        ) -> pd.DataFrame:
    """one sleeve's admission weight swept over the grid, all other sleeves at production."""
    assets = inputs.assets
    if ticker not in assets.index:
        raise ValueError(f"unknown ticker, got {ticker!r}")
    benchmark = get_benchmark(inputs=inputs, mandate=mandate)
    covar, _, rf_rate = build_moments(inputs=inputs)
    rows = {}
    for weight in grid:
        policy = assets['w_paper'].copy()
        policy[ticker] = float(weight)
        _, mu_x, _ = build_moments(inputs=inputs, admission_weights=policy)
        book = solve_mandate(covar=covar, cmas=mu_x, benchmark_weights=benchmark)
        stats = report_portfolio(weights=book, covar=covar, cmas=mu_x, rf_rate=rf_rate,
                                 inputs=inputs, benchmark_weights=benchmark)
        admitted = policy * assets['alpha']
        rows[float(weight)] = {'sleeve_weight': float(book[ticker]),
                               'excess_sharpe': float(stats['excess_sharpe']),
                               'total_return': float(stats['total_return']),
                               'raw_claim': float((admitted / assets['resid_vol']).pow(2).sum())}
    table = pd.DataFrame(rows).T
    table.index.name = 'admission_weight'
    return table


def plot_dial_sweeps(sweeps: Dict[str, pd.DataFrame],
                     inputs,
                     figsize: Tuple[float, float] = (12.8, 4.6),
                     weight_decimals: int = 2,                 # printed precision, percentage points
                     ) -> plt.Figure:
    """three sleeves, weight response and claim response on one panel pair each.

    Two presentation disciplines (roadmap J8a). Optimal weights are rounded to
    `weight_decimals` percentage points before plotting, because a sleeve sitting
    on its box cap returns solver jitter of order 1e-6 pp that matplotlib would
    otherwise render on an offset axis as a dramatic zigzag: Private Equity is
    flat at its 22.5% cap across the whole sweep and must look flat. The
    right-hand claimed-Sharpe axes share one limit across the three panels,
    because the exhibit's point IS the comparison between sleeves, and per-panel
    autoscaling makes the Insurance-Linked span of 0.007 look as large as Gold's
    0.050.
    """
    axes = np.atleast_1d(plt.subplots(1, len(sweeps), figsize=figsize, sharex=True)[1])
    fig = axes[0].figure
    sharpe_low = min(float(t['excess_sharpe'].min()) for t in sweeps.values())
    sharpe_high = max(float(t['excess_sharpe'].max()) for t in sweeps.values())
    pad = 0.06 * (sharpe_high - sharpe_low)
    shared_limits = (sharpe_low - pad, sharpe_high + pad)

    for ax, (ticker, table) in zip(axes, sweeps.items()):
        sleeve = SWEEP_SLEEVES[ticker]
        production = float(inputs.assets.loc[ticker, 'w_paper'])
        weights = (1e2 * table['sleeve_weight']).round(weight_decimals)
        ax.plot(table.index, weights, color=es.BLUE, lw=2.2, zorder=3)
        ax.axvline(production, color='0.15', lw=1.1, ls='--', zorder=2)
        ax.set_xlabel(f"{sleeve} admission weight $w_i$", fontsize=9.6)
        ax.set_ylabel('Optimal sleeve weight (%)', fontsize=9.6, color=es.BLUE)
        ax.tick_params(axis='y', labelcolor=es.BLUE)
        if float(weights.max() - weights.min()) < 10.0 ** -weight_decimals:
            centre = float(weights.iloc[0])              # flat at a box cap: show it as flat
            ax.set_ylim(centre - 1.0, centre + 1.0)
        es.style_axis(ax=ax, grid_axis='both', fontsize=8.8)

        twin = ax.twinx()
        twin.plot(table.index, table['excess_sharpe'], color=es.ORANGE, lw=2.2, zorder=3)
        twin.set_ylim(*shared_limits)                    # one scale across the three panels
        twin.set_ylabel('Claimed excess Sharpe', fontsize=9.6, color=es.ORANGE)
        twin.tick_params(axis='y', labelcolor=es.ORANGE, labelsize=8.8)
        twin.spines['top'].set_visible(False)
        span = float(table['excess_sharpe'].max() - table['excess_sharpe'].min())
        ax.set_title(f"{sleeve}: Sharpe span {span:.3f}", fontsize=10.2, loc='left')

    axes[0].plot([], [], color=es.BLUE, lw=2.2, label='Optimal sleeve weight (%, left axis)')
    axes[0].plot([], [], color=es.ORANGE, lw=2.2, label='Claimed excess Sharpe (right axis)')
    axes[0].legend(fontsize=8.4, loc='center left', frameon=False)
    fig.suptitle('Turning one sleeve\'s dial moves the weight until a guardrail binds, '
                 'and the claim all the way',
                 fontsize=12.0, x=0.008, ha='left')
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.90))
    return fig


def build_sleeve_tornado(inputs, mandate: str = MANDATE) -> pd.DataFrame:
    """one-at-a-time ranking: the claimed-Sharpe move from w_i = 0 to w_i = 1 per sleeve."""
    assets = inputs.assets
    covar, mu_production, rf_rate = build_moments(inputs=inputs)
    benchmark = get_benchmark(inputs=inputs, mandate=mandate)
    production_book = solve_mandate(covar=covar, cmas=mu_production, benchmark_weights=benchmark)
    production_sharpe = float(report_portfolio(weights=production_book, covar=covar,
                                               cmas=mu_production, rf_rate=rf_rate,
                                               inputs=inputs)['excess_sharpe'])
    rows = {}
    for ticker in TORNADO_SLEEVES:
        sharpes = {}
        for setting, weight in (('at_zero', 0.0), ('at_full', 1.0)):
            policy = assets['w_paper'].copy()
            policy[ticker] = weight
            _, mu_x, _ = build_moments(inputs=inputs, admission_weights=policy)
            book = solve_mandate(covar=covar, cmas=mu_x, benchmark_weights=benchmark)
            sharpes[setting] = float(report_portfolio(weights=book, covar=covar, cmas=mu_x,
                                                      rf_rate=rf_rate,
                                                      inputs=inputs)['excess_sharpe'])
            sharpes[f"weight_{setting}"] = float(book[ticker])
        rows[ticker] = {'sleeve': assets.loc[ticker, 'sleeve'],
                        'w_production': float(assets.loc[ticker, 'w_paper']),
                        **sharpes,
                        'span': sharpes['at_full'] - sharpes['at_zero'],
                        'releases_box': abs(sharpes['weight_at_zero']
                                            - sharpes['weight_at_full']) > 1e-6}
    table = pd.DataFrame(rows).T
    table.attrs['production_sharpe'] = production_sharpe
    return table.sort_values('span', key=lambda s: s.abs())


def plot_sleeve_tornado(table: pd.DataFrame,
                        figsize: Tuple[float, float] = (9.6, 4.4),
                        ) -> plt.Figure:
    """horizontal spans from the production Sharpe, widest at the bottom."""
    production = table.attrs['production_sharpe']
    fig, ax = plt.subplots(figsize=figsize)
    positions = np.arange(len(table))
    for position, (_, row) in zip(positions, table.iterrows()):
        low, high = sorted((float(row['at_zero']), float(row['at_full'])))
        ax.barh(position, high - low, left=low, height=0.58, color=es.BLUE, zorder=3)
        ax.annotate(f"{low:.2f}", xy=(low - 0.002, position), va='center', ha='right',
                    fontsize=8.4, color='0.25')
        ax.annotate(f"{high:.2f}", xy=(high + 0.002, position), va='center', ha='left',
                    fontsize=8.4, color='0.25')
    ax.axvline(production, color=es.ORANGE, lw=1.6, ls='--', zorder=4,
               label=f"production policy ({production:.2f})")
    ax.set_yticks(positions)
    ax.set_yticklabels(list(table['sleeve']), fontsize=9.2)
    ax.set_xlabel('Claimed ex-ante excess Sharpe ratio, sleeve admission from $w_i=0$ to $w_i=1$',
                  fontsize=9.8)
    es.style_axis(ax=ax, grid_axis='x', fontsize=9.0)
    ax.legend(fontsize=8.8, loc='lower right', frameon=False)
    ax.set_title('One dial at a time: which admission moves the claim most',
                 fontsize=11.2, loc='left')
    fig.tight_layout()
    return fig


def build_governed_dial(inputs,
                        mandate: str = MANDATE,
                        grid: np.ndarray = SLEEVE_GRID,
                        sigma_multiple: float = FLOOR_SIGMA_MULTIPLE,
                        ) -> pd.DataFrame:
    """per-sleeve admissible admission range under Cap 3 at each kappa and under the B13 floor.

    Cap 3 is a PORTFOLIO constraint, so the admissible range of one sleeve
    depends on where the others sit. Two baselines matter, and both are
    reported:

    Production baseline. Hold the other sleeves at the production policy
    (w_paper). That policy claims 1.398 against a budget of 0.238 at kappa = 1, and
    the largest single sleeve carries only 0.68 of that, so the remaining
    sleeves alone exceed the budget: NO w_i is admissible for any sleeve at any
    kappa on the grid. That is the Cap 3 audit result of Section 4.3 restated
    per sleeve, not a defect, and it is why the exhibit uses the second
    baseline.

    Governed baseline. Hold the other sleeves at their Cap 3-projected weights
    theta(kappa) * w_j, the uniform scaling of governed_cma_projection. The
    remaining budget then admits a closed-form range for sleeve i,

        w_i,max = min(1, (sigma_eps,i / |alpha_i|)
                         * sqrt(max(0, kappa * SR2_MATFCMA - claim_others(kappa))))

    which is what the committee can still turn once the portfolio budget is
    being respected elsewhere.

    The floor leg needs the re-solve and stays a sweep: for each w_i on the
    grid, re-solve the mandate and test the stress-repriced total against
    mu_bm - n * sigma_bm.
    """
    assets = inputs.assets
    covar, _, rf_rate = build_moments(inputs=inputs)
    benchmark = get_benchmark(inputs=inputs, mandate=mandate)
    attainable = float(compute_sharpe_accounting(inputs=inputs)['attainable'])
    base_bm = assets['factor_excess_cma'] + assets['w_paper'] * assets['alpha']
    floor_excess = float(base_bm @ benchmark) - sigma_multiple * float(
        np.sqrt(benchmark @ covar.values @ benchmark))

    production_claim = (assets['w_paper'] * assets['alpha'] / assets['resid_vol']) ** 2
    thetas = {}
    for kappa in KAPPA_GRID:
        raw = float(production_claim.sum())
        thetas[kappa] = min(1.0, float(np.sqrt(kappa * attainable / raw))) if raw > 0.0 else 1.0

    rows = {}
    for ticker in TORNADO_SLEEVES:
        alpha = float(assets.loc[ticker, 'alpha'])
        resid_vol = float(assets.loc[ticker, 'resid_vol'])
        row = {'sleeve': assets.loc[ticker, 'sleeve'],
               'w_production': float(assets.loc[ticker, 'w_paper'])}
        for kappa in KAPPA_GRID:
            budget = kappa * attainable
            others_production = float(production_claim.drop(ticker).sum())
            others_governed = thetas[kappa] ** 2 * others_production
            for label, others in (('prod', others_production), ('gov', others_governed)):
                remaining = budget - others
                if remaining <= 0.0 or abs(alpha) < 1e-12:
                    admissible = 0.0 if remaining <= 0.0 else 1.0
                else:
                    admissible = min(1.0, resid_vol / abs(alpha) * float(np.sqrt(remaining)))
                row[f"w_max_kappa_{kappa:.2f}_{label}"] = admissible
        # floor leg: needs the mandate re-solve at every grid point
        stress_totals = {}
        for weight in grid:
            policy = assets['w_paper'].copy()
            policy[ticker] = float(weight)
            _, mu_x, _ = build_moments(inputs=inputs, admission_weights=policy)
            book = solve_mandate(covar=covar, cmas=mu_x, benchmark_weights=benchmark)
            admitted = policy * assets['alpha']
            stress = (assets['factor_excess_cma'] + admitted
                      + inputs.betas.values @ inputs.factor_premia_scenarios['stress'].values)
            stress_totals[float(weight)] = float(stress @ book)
        stress_series = pd.Series(stress_totals)
        floor_admissible = stress_series.index[stress_series >= floor_excess]
        row['w_max_floor'] = float(floor_admissible.max()) if len(floor_admissible) else np.nan
        row['floor_binds'] = bool(stress_series.min() < floor_excess)
        row['stress_total_min'] = float(stress_series.min()) + rf_rate
        rows[ticker] = row

    table = pd.DataFrame(rows).T
    table.attrs['attainable'] = attainable
    table.attrs['floor_total'] = floor_excess + rf_rate
    table.attrs['sigma_multiple'] = sigma_multiple
    table.attrs['thetas'] = thetas
    table.attrs['production_claim'] = float(production_claim.sum())
    return table


def plot_governed_dial(table: pd.DataFrame,
                       figsize: Tuple[float, float] = (10.4, 4.8),
                       ) -> plt.Figure:
    """admissible admission ranges per sleeve: nested Cap 3 bars with the floor annotated."""
    fig, ax = plt.subplots(figsize=figsize)
    positions = np.arange(len(table))[::-1]
    shades = {1.00: es.LIGHT_BLUE, 0.50: es.BLUE, 0.25: '#0d3f66'}
    heights = {1.00: 0.62, 0.50: 0.42, 0.25: 0.22}
    for kappa in sorted(KAPPA_GRID, reverse=True):
        column = f"w_max_kappa_{kappa:.2f}_gov"
        ax.barh(positions, table[column].astype(float), height=heights[kappa],
                color=shades[kappa], zorder=3, label=fr"Cap 3 at $\kappa = {kappa:.2f}$")
    for position, (_, row) in zip(positions, table.iterrows()):
        ax.plot([row['w_production']], [position], marker='D', ms=6.0, color=es.ORANGE,
                zorder=5)
        floor_limit = row['w_max_floor']
        if not row['floor_binds']:
            continue
        ax.plot([floor_limit, floor_limit], [position - 0.36, position + 0.36],
                color=es.DARK_RED, lw=2.0, zorder=6)

    ax.plot([], [], marker='D', ms=6.0, ls='none', color=es.ORANGE,
            label='production admission $w_i$')
    ax.set_yticks(positions)
    ax.set_yticklabels(list(table['sleeve']), fontsize=9.2)
    ax.set_xlim(0.0, 1.05)
    ax.set_xlabel('Admissible sleeve admission weight $w_i$', fontsize=9.8)
    es.style_axis(ax=ax, grid_axis='x', fontsize=9.0)
    ax.legend(fontsize=8.6, loc='lower right', frameon=False)
    # J8d: the frozen Appendix E sibling (sleeve_tornado.PNG) carries a takeaway title and
    # NO in-figure note, so the title stays and the two lines of caption-grade note move to
    # the tex fragment, which already carries the floor level and the finding (B5).
    ax.set_title('The governed dial: Cap 3 sets the admissible admission range',
                 fontsize=11.2, loc='left')
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def run_admission_exhibits(snapshot: str = SNAPSHOT,
                           save_outputs: bool = True,
                           ) -> Dict[str, pd.DataFrame]:
    """build and write the six Stage J4c/J4e/J4f exhibits and the B13 floor table."""
    inputs = load_paper_inputs(snapshot=snapshot)
    print('=' * 78)
    print(f"Stage J4c/J4e/J4f — admission, scenario, and governed-dial exhibits, cut {snapshot}")
    print('=' * 78)

    boxed_weights, boxed_stats = sweep_admission_dial(inputs=inputs, apply_guardrails=True)
    print('\n--- fig:admission_dial, boxed sweep (selected s) ---')
    print(boxed_stats.loc[[0.0, 0.5, 1.0, 1.5]].round(4).to_string())
    print('\nclaimed excess Sharpe, R2 printed vs regenerated:')
    for scale, reference in R2_DIAL_SHARPE.items():
        print(f"  s = {scale:.1f}   R2 {reference:.2f}   "
              f"new {float(boxed_stats.loc[scale, 'excess_sharpe']):.2f}")

    nobox_weights, nobox_stats = sweep_admission_dial(inputs=inputs, apply_guardrails=False)
    print(f"\n--- fig:admission_nobox, unconstrained at the benchmark vol "
          f"{boxed_stats.attrs['benchmark_vol']:.2%} (selected s) ---")
    print(nobox_stats.loc[[0.0, 0.5, 1.0, 1.5]].round(4).to_string())

    scenario = sweep_scenario_admission(inputs=inputs)
    print('\n--- fig:scenario_admission (selected s, excess returns) ---')
    print(scenario.loc[[0.0, 0.5, 1.0, 1.5]].round(4).to_string())
    print(f"\nstress sign-flip pair: R2 {R2_STRESS_SIGN_FLIP[0]:+.1%} at s=0 and "
          f"{R2_STRESS_SIGN_FLIP[1]:+.1%} at s=1; new "
          f"{float(scenario.loc[0.0, 'stress']):+.1%} and "
          f"{float(scenario.loc[1.0, 'stress']):+.1%}")
    print(f"band width: R2 {R2_SCENARIO_BAND_WIDTH:.1%}, new "
          f"{float(scenario['band_width'].mean()):.1%} "
          f"(range {float(scenario['band_width'].min()):.2%} to "
          f"{float(scenario['band_width'].max()):.2%})")

    floors = build_scenario_floor_table(inputs=inputs)
    print(f"\n--- Section 5.3 B13 one-sigma floor table, n = "
          f"{floors.attrs['sigma_multiple']:.1f}, both admission policies ---")
    print(floors.round(4).to_string())

    sweeps = {t: sweep_single_sleeve(inputs=inputs, ticker=t) for t in SWEEP_SLEEVES}
    print('\n--- fig:dial_sweeps, per-sleeve Sharpe span ---')
    for ticker, table in sweeps.items():
        print(f"  {SWEEP_SLEEVES[ticker]:<18s} weight "
              f"{1e2 * float(table['sleeve_weight'].iloc[0]):.2f}% -> "
              f"{1e2 * float(table['sleeve_weight'].iloc[-1]):.2f}%, Sharpe "
              f"{float(table['excess_sharpe'].iloc[0]):.3f} -> "
              f"{float(table['excess_sharpe'].iloc[-1]):.3f}")

    tornado = build_sleeve_tornado(inputs=inputs)
    print(f"\n--- fig:sleeve_tornado, production Sharpe "
          f"{tornado.attrs['production_sharpe']:.3f} ---")
    print(tornado.round(4).to_string())

    governed = build_governed_dial(inputs=inputs)
    print(f"\n--- fig:governed_dial, SR2_MATFCMA {governed.attrs['attainable']:.3f}, "
          f"production claim {governed.attrs['production_claim']:.3f}, "
          f"floor {governed.attrs['floor_total']:.2%} total ---")
    print(f"Cap 3 projection scalars theta(kappa): "
          f"{ {k: round(v, 3) for k, v in governed.attrs['thetas'].items()} }")
    print(governed.round(3).to_string())
    print("\n'prod' columns hold the other sleeves at the production policy w_paper; those alone "
          "exhaust the Cap 3 budget,\nso no admission is admissible for any sleeve "
          "(the Section 4.3 audit result restated per sleeve).")
    print("'gov' columns hold the other sleeves at their Cap 3-projected weights and are "
          "what the exhibit plots.")
    print(f"\nfloor binds anywhere across the sweeps: {bool(governed['floor_binds'].any())} "
          f"-> {'the floor is the binding constraint somewhere' if governed['floor_binds'].any() else 'Cap 3 binds first everywhere (the expected finding at n = 1)'}")

    if save_outputs:
        es.save_figure(plot_admission_dial(
            weight_path=boxed_weights, statistics=boxed_stats, inputs=inputs,
            title='The alpha admission dial moves the SAA; it belongs to the committee, '
                  'not the estimator'), 'admission_dial.PNG')
        es.save_figure(plot_admission_dial(
            weight_path=nobox_weights, statistics=nobox_stats, inputs=inputs,
            title='Without the mandate guardrails the same admission rebuilds the book'),
            'admission_dial_nobox.PNG')
        es.save_figure(plot_scenario_admission(table=scenario), 'scenario_admission.PNG')
        es.save_figure(plot_dial_sweeps(sweeps=sweeps, inputs=inputs), 'dial_sweeps.PNG')
        es.save_figure(plot_sleeve_tornado(table=tornado), 'sleeve_tornado.PNG')
        es.save_figure(plot_governed_dial(table=governed), 'governed_dial.PNG')
        write_floor_table_tex(floors=floors)
        write_governed_dial_tex(governed=governed)
    return {'boxed_stats': boxed_stats, 'nobox_stats': nobox_stats, 'scenario': scenario,
            'floors': floors, 'tornado': tornado, 'governed': governed}


def write_floor_table_tex(floors: pd.DataFrame,
                          file_name: str = 'exhibit_scenario_floors.tex',
                          ) -> Path:
    """the Section 5.3 floor table, folded into prose per owner decision O-J3."""
    n = floors.attrs['sigma_multiple']
    lines = [
        '% ===== Section 5.3 scenario floors, B13 one-sigma rule =====',
        '% Source: replication/run_admission_exhibits.py on cma_data snapshot 2026q2.',
        f"% Rule: stress-repriced expected TOTAL return >= mu_bm - {n:.1f} * sigma_bm per mandate,",
        '%   with mu_bm the benchmark expected total return under the base CMAs and sigma_bm',
        '%   the benchmark volatility from the risk model. Anchor-invariant: r_f cancels on',
        '%   both sides, so the rule is computed in excess space and r_f added for display.',
        '%   No floor value is hand-set; n = FLOOR_SIGMA_MULTIPLE is one constant in one place.',
        '% Headroom is reported under BOTH admission policies: the PRODUCTION policy',
        '%   w_paper (PE recut to 0.5) and the PRE-RECUT WORKBOOK policy w_workbook',
        '%   (PE at 1.0). Naming per owner ruling O-J11 / B6: w_paper is the production',
        '%   policy, w_workbook is never labelled production.',
        '',
        r"			\toprule",
        r"			\textbf{Mandate} & $\mu_{bm}$ & $\sigma_{bm}$ & \textbf{Floor} & "
        r"\multicolumn{2}{c}{\textbf{Stress total}} & \multicolumn{2}{c}{\textbf{Headroom}} \\",
        r"			& (\%) & (\%) & (\%) & production & pre-recut & production & pre-recut \\",
        r"			\midrule",
    ]
    for mandate, row in floors.iterrows():
        lines.append(
            f"\t\t\t{es.tex_escape(mandate):<20s} & {1e2 * row['mu_bm_total']:.2f} & "
            f"{1e2 * row['sigma_bm']:.2f} & {1e2 * row['floor_total']:.2f} & "
            f"{1e2 * row['stress_total_w_paper']:.2f} & {1e2 * row['stress_total_w_workbook']:.2f} & "
            f"{1e2 * row['headroom_w_paper']:+.2f} & {1e2 * row['headroom_w_workbook']:+.2f} \\\\")
    lines.append(r"			\bottomrule")
    lines.append('%')
    binding = [m for m, r in floors.iterrows()
               if 'BINDS' in (r['binds_w_paper'], r['binds_w_workbook'])]
    lines.append(f"% Finding: the floor is {'BINDING on ' + str(binding) if binding else 'slack on all eight mandates'} "
                 f"at n = {n:.1f}. Headroom is identical in excess and total space.")
    return es.write_fragment(lines=lines, file_name=file_name)


def write_governed_dial_tex(governed: pd.DataFrame,
                            file_name: str = 'exhibit_governed_dial.tex',
                            ) -> Path:
    """drop-in tex for the new Appendix E governed-dial figure, in the exhibit_cap3 pattern."""
    floor_binds = bool(governed['floor_binds'].any())
    finding = ('the one-sigma benchmark stress floor binds before Cap~3 on at least one sleeve'
               if floor_binds else
               'the one-sigma benchmark stress floor never binds across the sweep, so Cap~3 is '
               'the operative constraint on every sleeve')
    lines = [
        '% ===== EXHIBIT: the governed dial (Appendix E per owner decision O-J3) =====',
        '% Source: replication/run_admission_exhibits.py on cma_data snapshot 2026q2.',
        '',
        r"\begin{figure}[H]",
        r"	\captionof{figure}{The Governed Dial: Cap 3 Sets the Admissible Admission Range}"
        r"\label{fig:governed_dial}\vspace*{-0.5\baselineskip}",
        r"	\begin{center}",
        r"		\includegraphics[width=0.90\linewidth]{figures/governed_dial.PNG}",
        r"	\end{center}",
        r"	\vspace*{-1.0\baselineskip}",
        r"	{\footnotesize Notes: 2026-Q2 production cut, 18 assets, USD, Balanced with "
        r"Alternatives mandate. For each admitted sleeve we sweep its admission weight "
        r"$w_i$ over $[0,1]$ with the other sleeves held at the production policy, re-solve "
        r"the mandate, and record the largest $w_i$ that satisfies each governance "
        r"constraint. Nested bars are the Cap~3 portfolio budget "
        r"$\boldsymbol\alpha_{adm}^{\intercal}\boldsymbol D^{-1}\boldsymbol\alpha_{adm} \le "
        r"\kappa \cdot SR^2_{MATFCMA}$ at the three budgets, with the OTHER sleeves held at "
        r"their Cap~3-projected weights $\theta(\kappa) w_j$, so a shorter bar is a tighter "
        r"budget. Held instead at the production policy, the other sleeves alone exhaust the "
        r"budget at every $\kappa$ shown and no admission is admissible: the sleeve-level "
        r"restatement of the Cap~3 audit result. Orange diamonds mark the production "
        r"admission. The benchmark stress floor "
        f"of Section~5.3, $\\text{{stress total}} \\ge \\mu_{{bm}} - {governed.attrs['sigma_multiple']:.1f}"
        r"\sigma_{bm}$, evaluates to "
        f"{governed.attrs['floor_total']:.2%}".replace('%', r'\%')
        + r" of expected total return. "
        + finding + r".}",
        r"\end{figure}",
    ]
    return es.write_fragment(lines=lines, file_name=file_name)


class LocalTests(str, Enum):
    ALL_EXHIBITS = 'all_exhibits'
    FLOOR_TABLE_ONLY = 'floor_table_only'
    DIAL_ONLY = 'dial_only'


def run_local_test(local_test: LocalTests) -> None:
    """Run local tests for development and debugging purposes."""
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 300)

    if local_test == LocalTests.ALL_EXHIBITS:
        run_admission_exhibits()

    elif local_test == LocalTests.FLOOR_TABLE_ONLY:
        inputs = load_paper_inputs()
        print(build_scenario_floor_table(inputs=inputs).round(4).to_string())

    elif local_test == LocalTests.DIAL_ONLY:
        inputs = load_paper_inputs()
        _, stats = sweep_admission_dial(inputs=inputs)
        print(stats.round(4).to_string())

    else:
        raise NotImplementedError(f"{local_test}")


if __name__ == '__main__':
    run_local_test(local_test=LocalTests.ALL_EXHIBITS)
