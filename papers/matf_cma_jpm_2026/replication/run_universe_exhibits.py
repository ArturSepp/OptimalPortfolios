"""
Universe exhibits from assets + betas + factor premia (roadmap Stage J3).

Five exhibits, all functions of the frozen config with no optimizer and no
randomness. The correctness gate for the two decomposition exhibits is the
add-on identity

    factor_excess_cma = beta @ lambda + equity_regional_addon      (to 1e-13 bp)

so every stack is built from beta @ lambda PLUS the add-on PLUS the admitted
alpha, never from factor_excess_cma plus the add-on, which would double-count
the regional blend (roadmap B9 / Stage J0b):

  cma_snapshot.png            tb:cma_snapshot — per-asset table of the
      significant betas (blank where the sign constraint zeroed the loading,
      |beta| < 1e-6), the base, stress and upside TOTAL CMAs (excess plus the
      reference cash rate), R^2, alpha, and total volatility.
  factor_attribution.png      tb:factor_attribution — per-asset stacked bars of
      beta_j lambda_j by factor, plus the regional blend add-on and the
      admitted alpha, with a tick at the published excess CMA.
  construction_waterfall.PNG  fig:construction_waterfall — the same
      decomposition grouped into the four DECLARED CHANNELS (factor-implied,
      regional blend, admitted alpha, gross add-on = 0 on this universe),
      keeping the frozen exhibit's horizontal grammar.
  benchmark_table.png         tab:benchmark_table — the two-level benchmark
      construction, D8-correct by construction because it renders
      cma_data.benchmarks rather than the R2 exhibit build's input.
  sr2_decomposition.PNG       fig:sr2_decomposition — the four headline squared
      Sharpe bars (frictionless ceiling, attainable systematic content, raw
      admitted claim, GLS-projected claim) beside the per-carrier split of the
      claim into the part that survives a solo projection and the part that is
      factor premium in disguise.

Units: snapshot quantities are decimal per annum. CMAs in cma_snapshot are
TOTAL (excess + rf_rate) and print in percent; the attribution and waterfall
are EXCESS and print in percent; the squared Sharpe panels are dimensionless.
As-of date is 30 June 2026 (defect D3: the R2 caption says 31 March 2026).
Main entry point: run_local_test(local_test).

Does not belong here: anything requiring a solve (Stage J4) or the factor
return history (Stage J2).
"""
# packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple
# qis / project
import qis as qis
import exhibit_style as es
from local_path import load_cma_data
from governed_cma_projection import (SNAPSHOT,
                                     load_paper_inputs,
                                     compute_sharpe_accounting,
                                     compute_gls_decomposition,
                                     compute_solo_premium_like_shares)

_cma_data = load_cma_data()

AS_OF_DATE = '30 June 2026'      # defect D3: the R2 caption says 31 March 2026
# Blanking threshold for the printed beta cells (roadmap J8b). The table prints two
# decimals, so anything below half a printed unit rounds to 0.00 and would render as
# a spurious "0.00" or "-0.00" against a caption that promises blank = zero exposure.
# Blanking at 0.005 makes the rendered table match the caption exactly and removes
# every signed zero. Sign-constraint zeros are exact zeros and fall inside it.
BETA_BLANK_TOL = 5e-3
IDENTITY_TOL = 1e-10             # decimal p.a.

# Factor colors for the attribution stack. Nine distinguishable hues that
# EXCLUDE orange and light blue, because those two carry fixed meanings in the
# frozen construction waterfall: light blue = regional blend, orange = admitted
# alpha. Equity leads on the manuscript primary so the largest segment matches
# the rest of the exhibit set.
FACTOR_COLORS: Dict[str, str] = {
    'Equity': '#1f77b4',
    'Rates': '#2ca02c',
    'Credit': '#8c564b',
    'Carry': '#9467bd',
    'Inflation': '#17becf',
    'Commodities': '#bcbd22',
    'Private Equity': '#e377c2',
    'Rates Vol': '#7f7f7f',
    'Fx': '#c7c7c7',
}
ADDON_COLOR = es.LIGHT_BLUE       # regional premium blend, as in the frozen waterfall
ADMITTED_COLOR = es.ORANGE        # admitted historical alpha, as in the frozen waterfall

# Compact column headers for tb:cma_snapshot: nine betas plus three scenario CMAs in
# one printable width. Spelled out in the exhibit caption.
SNAPSHOT_HEADERS: Dict[str, str] = {
    'Equity': 'Eq', 'Rates': 'Rt', 'Credit': 'Cr', 'Carry': 'Cy', 'Inflation': 'Inf',
    'Commodities': 'Cmd', 'Private Equity': 'PE', 'Rates Vol': 'RVol', 'Fx': 'FX',
}
CLASS_ABBREVIATIONS: Dict[str, str] = {'Bonds': 'Bonds', 'Equities': 'Equities',
                                       'Alternatives': 'Alts'}


def build_channels(inputs) -> pd.DataFrame:
    """the four declared channels per asset, asserted to sum to the published excess CMA.

    Columns: factor_implied (beta @ lambda), blend (equity_regional_addon),
    admitted (w_paper * alpha), gross_addon (zero on this universe), and
    published_excess. The assert is the Guardrail-2 gate for this stage.
    """
    assets = inputs.assets
    factor_implied = pd.Series(inputs.betas.values @ inputs.factor_premia.values,
                               index=assets.index, name='factor_implied')
    blend = assets['equity_regional_addon'].rename('blend')
    admitted = (assets['w_paper'] * assets['alpha'] + 0.0).rename('admitted')
    gross_addon = pd.Series(0.0, index=assets.index, name='gross_addon')
    published = (assets['factor_excess_cma'] + admitted).rename('published_excess')

    channels = pd.concat([factor_implied, blend, admitted, gross_addon, published], axis=1)
    stacked = channels[['factor_implied', 'blend', 'admitted', 'gross_addon']].sum(axis=1)
    gap = float((stacked - published).abs().max())
    if gap > IDENTITY_TOL:
        raise ValueError(f"declared channels do not sum to the published excess CMA, "
                         f"got max gap {gap!r} (add-on double-count?)")
    channels.attrs['reconciliation_gap_bp'] = 1e4 * gap
    return channels


# --------------------------------------------------------------------------
# tb:cma_snapshot
# --------------------------------------------------------------------------

def build_cma_snapshot(inputs) -> pd.DataFrame:
    """per-asset betas, total CMAs under the three scenarios, R^2, alpha, and total vol."""
    assets = inputs.assets
    rf_rate = float(assets['rf_rate'].iloc[0])
    channels = build_channels(inputs=inputs)
    base_excess = channels['published_excess']

    table = pd.DataFrame({'Ticker': [t.replace(' Index', '') for t in assets.index],
                          'Class': [CLASS_ABBREVIATIONS[c] for c in assets['asset_class']]},
                         index=assets.index)
    for factor in inputs.betas.columns:
        loading = inputs.betas[factor]
        # + 0.0 normalises the signed zero of exactly-negative-zero loadings
        table[SNAPSHOT_HEADERS[factor]] = [('' if abs(v) < BETA_BLANK_TOL else f"{v + 0.0:.2f}")
                                           for v in loading]
    for label, scenario in (('Base', None), ('Stress', 'stress'), ('Upside', 'upside')):
        excess = base_excess if scenario is None else \
            base_excess + inputs.betas.values @ inputs.factor_premia_scenarios[scenario].values
        table[label] = [f"{1e2 * (v + rf_rate):.1f}%" for v in excess]
    table['R2'] = [f"{1e2 * v:.0f}%" for v in assets['r2']]
    table['Alpha'] = [f"{1e2 * v:.1f}%" for v in assets['alpha']]
    table['Vol'] = [f"{1e2 * v:.1f}%" for v in assets['total_vol']]
    table.index = assets['sleeve']
    table.index.name = None
    table.attrs['rf_rate'] = rf_rate
    return table


def plot_cma_snapshot(table: pd.DataFrame) -> plt.Figure:
    """render the universe snapshot table with the asset-class blocks separated."""
    boundaries = [i for i in range(1, len(table)) if table['Class'].iloc[i] != table['Class'].iloc[i - 1]]
    widths = [1.60, 0.82, 0.62] + [0.44] * 9 + [0.52, 0.54, 0.54] + [0.42, 0.50, 0.44]
    return es.table_figure(df=table, col_widths=widths, row_height=0.40, fontsize=7.4,
                           special_columns_colors=[(0, '#c6e2f0')],
                           rows_edge_lines=boundaries, rows_edge_color=es.BLUE)


# --------------------------------------------------------------------------
# tb:factor_attribution
# --------------------------------------------------------------------------

def build_factor_attribution(inputs) -> pd.DataFrame:
    """per-asset beta_j lambda_j by factor, plus the blend add-on and the admitted alpha."""
    assets = inputs.assets
    contributions = pd.DataFrame(inputs.betas.values * inputs.factor_premia.values,
                                 index=assets.index, columns=inputs.betas.columns)
    channels = build_channels(inputs=inputs)
    # the per-factor contributions must reproduce the factor-implied channel exactly
    gap = float((contributions.sum(axis=1) - channels['factor_implied']).abs().max())
    if gap > IDENTITY_TOL:
        raise ValueError(f"per-factor contributions do not sum to beta @ lambda, got {gap!r}")
    attribution = contributions.copy()
    attribution['Regional blend'] = channels['blend']
    attribution['Admitted alpha'] = channels['admitted']
    attribution.attrs['published_excess'] = channels['published_excess']
    return attribution


def plot_stacked_attribution(attribution: pd.DataFrame,
                             inputs,
                             figsize: Tuple[float, float] = (11.0, 5.6),
                             ) -> plt.Figure:
    """horizontal stacked bars per asset, positive and negative segments accumulated separately."""
    sleeves = list(es.sleeve_labels(inputs.assets))
    published = attribution.attrs['published_excess']
    positions = np.arange(len(sleeves))[::-1]        # snapshot order top to bottom
    colors = {**{f: FACTOR_COLORS[f] for f in inputs.betas.columns},
              'Regional blend': ADDON_COLOR, 'Admitted alpha': ADMITTED_COLOR}
    # drop channels that contribute nothing anywhere rather than showing a legend entry for
    # an invisible segment; FX carries a zero premium by the equilibrium argument
    live = [c for c in attribution.columns if float(attribution[c].abs().max()) > 1e-12]
    dropped = [es.factor_label(c) for c in attribution.columns if c not in live]
    if dropped:
        print(f"attribution: channels with zero contribution on every asset, "
              f"omitted from the stack and the legend: {dropped}")

    fig, ax = plt.subplots(figsize=figsize)
    left_positive = np.zeros(len(sleeves))
    left_negative = np.zeros(len(sleeves))
    for column in live:
        values = 1e2 * attribution[column].values
        base = np.where(values >= 0.0, left_positive, left_negative)
        label = es.factor_label(column)
        ax.barh(positions, values, left=base, color=colors[column], height=0.66,
                label=label, zorder=3, linewidth=0.0)
        left_positive = left_positive + np.where(values >= 0.0, values, 0.0)
        left_negative = left_negative + np.where(values < 0.0, values, 0.0)

    for position, total in zip(positions, 1e2 * published.values):
        ax.plot([total, total], [position - 0.34, position + 0.34], color='0.1', lw=1.6, zorder=4)
        ax.annotate(f"{total:.2f}", xy=(total + 0.08, position), va='center', ha='left',
                    fontsize=7.6, color='0.2', zorder=4)

    ax.set_yticks(positions)
    ax.set_yticklabels(sleeves, fontsize=8.5)
    ax.set_xlim(1.10 * float(np.min(left_negative)), 1.16 * float(1e2 * published.max()))
    ax.set_xlabel('Base excess CMA attribution by factor (% p.a.)', fontsize=9.5)
    ax.axvline(0.0, color='0.3', lw=0.8, zorder=2)
    for boundary in es.class_boundaries(inputs.assets):
        ax.axhline(len(sleeves) - boundary - 0.5, color='0.75', lw=0.8, ls='--', zorder=1)
    es.style_axis(ax=ax, grid_axis='x', fontsize=8.5)
    ax.legend(fontsize=7.8, ncols=2, loc='upper right', frameon=False)
    ax.set_title('Every basis point of every excess CMA carries a factor name',
                 fontsize=11.0, loc='left')
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# fig:construction_waterfall
# --------------------------------------------------------------------------

def plot_construction_waterfall(channels: pd.DataFrame,
                                inputs,
                                figsize: Tuple[float, float] = (10.4, 6.2),
                                ) -> plt.Figure:
    """the frozen main-text grammar: channels stacked left to right, black tick at the total."""
    sleeves = list(es.sleeve_labels(inputs.assets))
    positions = np.arange(len(sleeves))[::-1]
    segments = [('factor_implied', r'Factor-implied  $\hat{\beta}^{\top}\lambda$', es.BLUE),
                ('blend', 'Regional premium blend', ADDON_COLOR),
                ('admitted', r'Admitted historical alpha  $w_i \alpha_i$', ADMITTED_COLOR)]

    fig, ax = plt.subplots(figsize=figsize)
    left_positive = np.zeros(len(sleeves))
    left_negative = np.zeros(len(sleeves))
    for column, label, color in segments:
        values = 1e2 * channels[column].values
        base = np.where(values >= 0.0, left_positive, left_negative)
        ax.barh(positions, values, left=base, color=color, height=0.66, label=label,
                zorder=3, linewidth=0.0)
        left_positive = left_positive + np.where(values >= 0.0, values, 0.0)
        left_negative = left_negative + np.where(values < 0.0, values, 0.0)

    totals = 1e2 * channels['published_excess'].values
    for position, total in zip(positions, totals):
        ax.plot([total, total], [position - 0.34, position + 0.34], color='0.1', lw=2.0, zorder=4)
        ax.annotate(f"{total:.2f}", xy=(total + 0.10, position), va='center', ha='left',
                    fontsize=8.4, color='0.15', zorder=4)

    ax.set_yticks(positions)
    ax.set_yticklabels(sleeves, fontsize=9.0)
    ax.set_xlim(min(0.0, 1.15 * float(np.min(left_negative))), 1.14 * float(totals.max()))
    ax.set_xlabel('Base excess CMA decomposition (% p.a.)', fontsize=10.0)
    ax.axvline(0.0, color='0.3', lw=0.8, zorder=2)
    for boundary in es.class_boundaries(inputs.assets):
        ax.axhline(len(sleeves) - boundary - 0.5, color='0.75', lw=0.9, ls='--', zorder=1)
    es.style_axis(ax=ax, grid_axis='x', fontsize=9.0)
    ax.legend(fontsize=8.6, loc='upper right', frameon=False)
    ax.set_title('Every excess CMA decomposes into declared channels; nothing is unattributed',
                 fontsize=11.5, loc='left')
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.13)
    ax.annotate('Black tick = total base excess CMA. Channels zero on this universe: gross add-on.',
                xy=(0.0, -0.115), xycoords='axes fraction', fontsize=8.0, color='0.35')
    return fig


# --------------------------------------------------------------------------
# tab:benchmark_table
# --------------------------------------------------------------------------

def build_benchmark_table() -> pd.DataFrame:
    """the two-level benchmark construction: class allocations, decomposition weights, and optima."""
    benchmarks = _cma_data.get_all_benchmarks()
    universe = _cma_data.get_universe()
    asset_classes = pd.Series(_cma_data.ASSET_CLASSES)
    # CLASS_ALLOCATIONS and DECOMPOSITION_WEIGHTS are module-level in cma_data.benchmarks
    # and not re-exported by the package __init__; reach them through the submodule
    # rather than editing cma_data (Guardrail 3: the shared layer is read-only here).
    class_allocations = _cma_data.benchmarks.CLASS_ALLOCATIONS
    decomposition = pd.Series(_cma_data.benchmarks.DECOMPOSITION_WEIGHTS)

    # two-line mandate headers so eight columns fit the printable width
    headers = {m: m.replace(' w/o Alts', '\nw/o Alts').replace(' with Alts', '\nwith Alts')
               for m in _cma_data.MANDATES}

    class_rows = {}
    for asset_class in ('Bonds', 'Equities', 'Alternatives'):
        row = {'Asset': '', 'Class': CLASS_ABBREVIATIONS[asset_class], 'd_i': ''}
        for mandate in _cma_data.MANDATES:
            row[headers[mandate]] = f"{1e2 * class_allocations[mandate][asset_class]:.0f}%"
        class_rows[f"[{CLASS_ABBREVIATIONS[asset_class]}]"] = row

    asset_rows = {}
    for ticker, sleeve in universe.items():
        row = {'Asset': sleeve, 'Class': CLASS_ABBREVIATIONS[asset_classes[ticker]],
               'd_i': f"{1e2 * decomposition[ticker]:.2f}%"}
        for mandate in _cma_data.MANDATES:
            row[headers[mandate]] = f"{1e2 * benchmarks.loc[ticker, mandate]:.2f}%"
        asset_rows[sleeve] = row

    table = pd.DataFrame.from_dict({**class_rows, **asset_rows}, orient='index')
    table = table.drop(columns=['Asset'])
    table.attrs['n_class_rows'] = len(class_rows)
    table.attrs['column_sums'] = benchmarks.sum(axis=0)
    return table


def plot_benchmark_table(table: pd.DataFrame) -> plt.Figure:
    """render the benchmark construction, class-allocation block separated from the instruments."""
    boundaries = [table.attrs['n_class_rows']]
    classes = list(table['Class'])
    boundaries += [i for i in range(table.attrs['n_class_rows'] + 1, len(classes))
                   if classes[i] != classes[i - 1]]
    widths = [1.75, 0.62, 0.66] + [0.86] * len(_cma_data.MANDATES)
    return es.table_figure(df=table, col_widths=widths, row_height=0.42, fontsize=7.6,
                           first_row_height=0.55,
                           special_columns_colors=[(0, '#c6e2f0')],
                           rows_edge_lines=boundaries, rows_edge_color=es.BLUE)


# --------------------------------------------------------------------------
# fig:sr2_decomposition
# --------------------------------------------------------------------------

def build_sr2_decomposition(inputs) -> Tuple[pd.Series, pd.DataFrame]:
    """the four headline squared Sharpe values and the per-carrier claim split."""
    assets = inputs.assets
    accounting = compute_sharpe_accounting(inputs=inputs)
    admitted = assets['w_paper'] * assets['alpha']
    raw_claim = float((admitted / assets['resid_vol']).pow(2).sum())
    _, deviation, _ = compute_gls_decomposition(mu_excess=admitted, inputs=inputs)
    gls_claim = float(deviation @ (deviation / assets['resid_vol'] ** 2))

    headline = pd.Series({'Frictionless\nfactor ceiling': accounting['ceiling'],
                          'Attainable systematic\n(MATF identity)': accounting['attainable'],
                          'Claimed admitted\nalpha (raw)': raw_claim,
                          'Honest idiosyncratic\n(GLS-projected)': gls_claim})

    shares = compute_solo_premium_like_shares(inputs=inputs)
    carriers = assets.index[assets['w_paper'] > 0.0]
    per_carrier = pd.DataFrame({'sleeve': assets.loc[carriers, 'sleeve'],
                                'claimed': (admitted[carriers] / assets.loc[carriers, 'resid_vol']) ** 2,
                                'premium_like_share': shares[carriers]})
    per_carrier['survives'] = per_carrier['claimed'] * (1.0 - per_carrier['premium_like_share'])
    gap = float(abs(per_carrier['claimed'].sum() - raw_claim))
    if gap > IDENTITY_TOL:
        raise ValueError(f"per-carrier claims do not sum to the raw claim, got {gap!r}")
    return headline, per_carrier


def plot_sr2_decomposition(headline: pd.Series,
                           per_carrier: pd.DataFrame,
                           figsize: Tuple[float, float] = (12.6, 5.2),
                           ) -> plt.Figure:
    """the frozen two-panel grammar: headline bars left, per-carrier claim split right."""
    fig, axs = plt.subplots(1, 2, figsize=figsize)

    ax = axs[0]
    colors = [es.BLUE, es.LIGHT_BLUE, es.ORANGE, es.DARK_RED]
    ax.bar(np.arange(len(headline)), headline.values, color=colors, width=0.62, zorder=3)
    for i, value in enumerate(headline.values):
        ax.annotate(f"{value:.2f}", xy=(i, value + 0.02 * headline.max()),
                    ha='center', va='bottom', fontsize=10.0, color='0.15')
    ax.set_xticks(np.arange(len(headline)))
    ax.set_xticklabels(list(headline.index), fontsize=8.4)
    ax.set_ylabel(r'Squared Sharpe  $SR^2$', fontsize=10.0)
    ax.set_ylim(0.0, 1.16 * float(headline.max()))
    ax.set_title('The claim against the opportunity set', fontsize=10.5, loc='left')
    es.style_axis(ax=ax, grid_axis='y', fontsize=9.0)

    ax = axs[1]
    order = per_carrier.index[::-1]
    positions = np.arange(len(order))
    ax.barh(positions + 0.19, per_carrier.loc[order, 'survives'], height=0.36,
            color=es.BLUE, label='Survives solo projection', zorder=3)
    ax.barh(positions - 0.19, per_carrier.loc[order, 'claimed'], height=0.36,
            color=es.ORANGE, label='Claimed (raw admitted)', zorder=3)
    for position, ticker in zip(positions, order):
        claimed = float(per_carrier.loc[ticker, 'claimed'])
        share = float(per_carrier.loc[ticker, 'premium_like_share'])
        ax.annotate(f"{share:.0%} premium-like", xy=(claimed + 0.012, position - 0.19),
                    va='center', ha='left', fontsize=8.6, color='0.2')
    ax.set_yticks(positions)
    ax.set_yticklabels(list(per_carrier.loc[order, 'sleeve']), fontsize=9.0)
    ax.set_xlim(0.0, 1.30 * float(per_carrier['claimed'].max()))
    ax.set_xlabel(r'Per-carrier $SR^2$ contribution (diagonal $D$)', fontsize=10.0)
    ax.set_title('Which alphas survive the projection', fontsize=10.5, loc='left')
    es.style_axis(ax=ax, grid_axis='x', fontsize=9.0)
    ax.legend(fontsize=8.8, loc='lower right', frameon=False)

    fig.suptitle('Most of what the book calls alpha is disguised premium '
                 'or a claim larger than the factor set',
                 fontsize=12.0, x=0.008, ha='left')
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    return fig


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def run_universe_exhibits(snapshot: str = SNAPSHOT,
                          save_outputs: bool = True,
                          ) -> Dict[str, pd.DataFrame]:
    """build and write the five Stage J3 exhibits, asserting the channel identities."""
    inputs = load_paper_inputs(snapshot=snapshot)
    print('=' * 78)
    print(f"Stage J3 — universe exhibits, cut {snapshot}, as of {AS_OF_DATE}")
    print('=' * 78)

    channels = build_channels(inputs=inputs)
    print(f"\ndeclared-channel reconciliation gap: "
          f"{channels.attrs['reconciliation_gap_bp']:.2e} bp (Guardrail-2 gate)")
    print('\n--- declared channels, percent p.a. ---')
    print((1e2 * channels).round(3).to_string())

    snapshot_table = build_cma_snapshot(inputs=inputs)
    print(f"\n--- tb:cma_snapshot (total CMAs, rf = {snapshot_table.attrs['rf_rate']:.2%}) ---")
    print(snapshot_table.to_string())

    attribution = build_factor_attribution(inputs=inputs)
    print('\n--- tb:factor_attribution components, percent p.a. ---')
    print((1e2 * attribution).round(3).to_string())

    benchmark_table = build_benchmark_table()
    print('\n--- tab:benchmark_table (D8-correct by construction) ---')
    print(benchmark_table.to_string())
    print(f"\nmandate weight sums: "
          f"{benchmark_table.attrs['column_sums'].round(10).unique()}")

    headline, per_carrier = build_sr2_decomposition(inputs=inputs)
    print('\n--- fig:sr2_decomposition headline ---')
    print(headline.round(3).to_string())
    print('\nper-carrier claim split:')
    print(per_carrier.round(4).to_string())

    if save_outputs:
        es.save_figure(plot_cma_snapshot(table=snapshot_table), 'cma_snapshot.png')
        es.save_figure(plot_stacked_attribution(attribution=attribution, inputs=inputs),
                       'factor_attribution.png')
        es.save_figure(plot_construction_waterfall(channels=channels, inputs=inputs),
                       'construction_waterfall.PNG')
        es.save_figure(plot_benchmark_table(table=benchmark_table), 'benchmark_table.png')
        es.save_figure(plot_sr2_decomposition(headline=headline, per_carrier=per_carrier),
                       'sr2_decomposition.PNG')
        write_universe_notes(per_carrier=per_carrier, channels=channels)
    return {'channels': channels, 'cma_snapshot': snapshot_table,
            'attribution': attribution, 'benchmark_table': benchmark_table,
            'sr2_headline': headline.to_frame('value'), 'sr2_per_carrier': per_carrier}


def write_universe_notes(per_carrier: pd.DataFrame,
                         channels: pd.DataFrame,
                         file_name: str = 'exhibit_universe_notes.tex',
                         ) -> Path:
    """caption-change notes for the Stage J3 exhibits, including defect D3 and the ILS upgrade."""
    ils = per_carrier.loc['EHFI804 Index']
    lines = [
        '% ===== Caption notes for the Stage J3 universe exhibits =====',
        '% Source: replication/run_universe_exhibits.py on cma_data snapshot 2026q2.',
        '%',
        f"% DEFECT D3, caption change required on tb:cma_snapshot: the as-of date is",
        f"%   {AS_OF_DATE}, not 31 March 2026. Every illustration in Appendix C restates",
        '%   as of the same date.',
        '%',
        '% tb:cma_snapshot — CMAs are TOTAL returns (excess + the reference cash rate).',
        f"%   A blank beta cell denotes zero exposure at the printed precision:",
        f"%   |beta| < {BETA_BLANK_TOL:g}, which covers both the exact sign-constraint zeros and",
        '%   any loading that would round to 0.00. No cell prints 0.00 or -0.00, so the',
        '%   rendered table matches the caption (roadmap J8b).',
        '%',
        '% tb:factor_attribution and fig:construction_waterfall — the stacks are built from',
        '%   beta @ lambda PLUS the regional add-on PLUS the admitted alpha. The add-on is',
        '%   INSIDE factor_excess_cma, so a stack built from factor_excess_cma plus the',
        '%   add-on would double-count it. Reconciliation to the published excess CMA:',
        f"%   {channels.attrs['reconciliation_gap_bp']:.1e} bp.",
        '%',
        '% tab:benchmark_table — rendered from cma_data/benchmarks.py, so the Asia ex-Japan',
        '%   4.52% / EM ex-Asia 0.88% pair is D8-correct by construction. The R2 exhibit',
        '%   build printed the transposed pair.',
        '%',
        '% fig:sr2_decomposition — the Insurance-Linked bar carries a premium-like share of',
        f"%   {ils['premium_like_share']:.0%} on a claimed {ils['claimed']:.2f} of squared Sharpe, of which",
        f"%   {ils['survives']:.2f} survives the solo projection. Section 4.2 gains that share",
        '%   (content map 4.2): ILS is the limit case, R^2 = 0.17 yet most of its admitted',
        '%   alpha is premium in disguise.',
    ]
    return es.write_fragment(lines=lines, file_name=file_name)


class LocalTests(str, Enum):
    ALL_EXHIBITS = 'all_exhibits'
    CHANNELS_ONLY = 'channels_only'


def run_local_test(local_test: LocalTests) -> None:
    """Run local tests for development and debugging purposes."""
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 300)

    if local_test == LocalTests.ALL_EXHIBITS:
        run_universe_exhibits()

    elif local_test == LocalTests.CHANNELS_ONLY:
        inputs = load_paper_inputs()
        print((1e2 * build_channels(inputs=inputs)).round(4).to_string())

    else:
        raise NotImplementedError(f"{local_test}")


if __name__ == '__main__':
    run_local_test(local_test=LocalTests.ALL_EXHIBITS)
