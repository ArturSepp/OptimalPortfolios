"""
Provider family: Decision One exhibits on a common risk model (roadmap Stage J4g).

Two phases, because the provider inputs arrive separately.

PHASE 1, live now — the Consensus provider. cma_data.consensus maps the Horizon
Actuarial 2025 survey averages (printed 10-year ARITHMETIC nominal total returns
for a USD investor, Exhibit 17) onto the paper universe. Every sleeve carries a
source flag: PUBLISHED where the survey line matches the sleeve in kind,
CONVERTED where a proxy mapping applies judgment, HELD_AT_MATF where the survey
has no line and the sleeve is held at the MATF value with no cross-provider tilt.
This script produces:
  the Consensus column of tab:provider_saa      mandate solve on the Consensus
      excess vector (total minus r_f), held-at-MATF sleeves at the MATF value
      per the R2 footnote convention.
  the Consensus row of tab:provider_decomposition (NEW, a table per owner
      decision O-J3) implied lambda_gls against the MATF calibration, the
      orthogonal claim s^2_h, and the implied tangency Sharpe
      sqrt(s^2_K + s^2_h).

Mapping caveats, enforced here. One developed-ex-US survey line spans four
equity sleeves and one emerging line spans both EM sleeves, so PER-SLEEVE
statements are undefined for the converted lines: the decomposition aggregates
those groups before printing anything. Held-at-MATF sleeves are excluded from
every decomposition, because they would contribute zero deviation by
construction and dilute the test.

PHASE 2, GATED on providers.csv — the anonymised A-D provider vectors. The
loader below reads the fixed schema (provider, sleeve, total_cma_arith, source,
survey_category, vintage, note) and raises a clear message when the file is
absent. Gated by owner item O-J7b: the exhibit-zip transfer plus a per-provider
confirmation that each vector traces to a publicly published edition. Nothing
is fabricated. When the file lands, the A-D columns of tab:provider_saa, the A-D
rows of the decomposition table, and both frontier PNGs
(provider_frontier_with_alts.PNG, provider_frontier_wo_alts.PNG) build from the
same functions.

Units: provider CMAs arrive as arithmetic TOTAL returns, decimal per annum; the
solve runs on excess returns (total minus r_f) per the adopted convention, and
the reporting layer adds r_f back.
Main entry point: run_local_test(local_test).

Does not belong here: the GLS mathematics (governed_cma_projection and
consensus_decomposition) and the MATF-only mandate exhibits
(run_mandate_exhibits.py).
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
from local_path import load_cma_data, get_cma_data_path
from governed_cma_projection import (SNAPSHOT,
                                     load_paper_inputs,
                                     compute_sharpe_accounting,
                                     compute_factor_information_matrix)
from consensus_decomposition import decompose_on_subset
from run_optimisation import (MANDATE, build_moments, get_benchmark, solve_mandate,
                             report_portfolio)

_cma_data = load_cma_data()

PROVIDERS_FILE = 'providers.csv'
PROVIDERS_SCHEMA = ['provider', 'sleeve', 'total_cma_arith', 'source', 'survey_category',
                    'vintage', 'note']
MATF_LABEL = 'MATF'
WITH_ALTS_FAMILY = ['Income with Alts', 'Low with Alts', 'Balanced with Alts', 'Growth with Alts']
WO_ALTS_FAMILY = ['Income w/o Alts', 'Low w/o Alts', 'Balanced w/o Alts', 'Growth w/o Alts']

# converted survey lines that span several sleeves: per-sleeve statements are undefined,
# so the decomposition aggregates each group before printing
CONVERTED_GROUPS: Dict[str, List[str]] = {
    'Developed ex-US equity (one survey line, four sleeves)':
        ['MSDEXKSN Index', 'NDDLUK Index', 'NDDLSZ Index', 'NDDLJN Index'],
    'Emerging equity (one survey line, two sleeves)':
        ['M1APJ Index', 'M1EFZ Index'],
}
R2_CONSENSUS_SR2_ALPHA = 0.091          # the parity reference on the 17-sleeve subset


def load_provider_vectors(snapshots_path: Optional[Path] = None) -> pd.DataFrame:
    """the anonymised A-D provider vectors; raises with a clear gate message when absent."""
    root = get_cma_data_path() / 'snapshots' if snapshots_path is None else Path(snapshots_path)
    file_path = root / SNAPSHOT / PROVIDERS_FILE
    if not file_path.exists():
        raise ValueError(
            f"providers.csv absent at {str(file_path)!r}. Stage J4g phase 2 is GATED on owner "
            f"item O-J7b: the frozen exhibit-zip transfer plus a per-provider confirmation that "
            f"each vector traces to a publicly published edition (title and date recorded in "
            f"the untracked name map). Provider data is never fabricated here. Expected schema: "
            f"{PROVIDERS_SCHEMA}.")
    providers = pd.read_csv(file_path)
    missing = [c for c in PROVIDERS_SCHEMA if c not in providers.columns]
    if missing:
        raise ValueError(f"providers.csv schema mismatch, missing {missing!r}")
    return providers


def build_provider_excess_vector(inputs,
                                 provider_totals: pd.Series,       # arithmetic TOTAL, decimal p.a.
                                 held_at_matf: List[str],
                                 ) -> pd.Series:
    """one provider's excess CMA vector, held-at-MATF sleeves carried at the MATF value."""
    assets = inputs.assets
    rf_rate = float(assets['rf_rate'].iloc[0])
    matf_excess = assets['factor_excess_cma'] + assets['w_paper'] * assets['alpha']
    excess = (provider_totals.reindex(assets.index) - rf_rate)
    excess.loc[held_at_matf] = matf_excess.loc[held_at_matf]
    if excess.isna().any():
        raise ValueError(f"provider vector incomplete for "
                         f"{list(excess[excess.isna()].index)!r}; either publish a value or "
                         f"list the sleeve as held at MATF")
    return excess.rename('provider_excess')


def build_consensus_column(inputs, mandate: str = MANDATE) -> Dict[str, pd.Series]:
    """the Consensus column of tab:provider_saa beside the MATF column, same design."""
    consensus = _cma_data.build_consensus_provider()
    held = list(consensus.index[consensus['source'] == 'held_at_matf'])
    covar, mu_matf, rf_rate = build_moments(inputs=inputs)
    mu_consensus = build_provider_excess_vector(inputs=inputs,
                                                provider_totals=consensus['total_cma_arith'],
                                                held_at_matf=held)
    benchmark = get_benchmark(inputs=inputs, mandate=mandate)

    books, stats = {'Benchmark': benchmark}, {}
    for label, cmas in ((MATF_LABEL, mu_matf), (_cma_data.CONSENSUS_LABEL, mu_consensus)):
        book = solve_mandate(covar=covar, cmas=cmas, benchmark_weights=benchmark)
        books[label] = book
        stats[label] = report_portfolio(weights=book, covar=covar, cmas=cmas, rf_rate=rf_rate,
                                        inputs=inputs, benchmark_weights=benchmark)
    return {'weights': pd.DataFrame(books), 'stats': pd.DataFrame(stats),
            'excess_vectors': pd.DataFrame({MATF_LABEL: mu_matf,
                                            _cma_data.CONSENSUS_LABEL: mu_consensus}),
            'held_at_matf': pd.Series(held, name='held_at_matf')}


def build_provider_decomposition(inputs,
                                 excess_vectors: pd.DataFrame,
                                 held_at_matf: List[str],
                                 ) -> pd.DataFrame:
    """one row per provider: implied lambda_gls, orthogonal claim, implied tangency Sharpe.

    Held-at-MATF sleeves are excluded, so every provider is decomposed on the
    same published-plus-converted subset and the rows are like-for-like. The
    implied tangency Sharpe is sqrt(s2_K + s2_h) with s2_K the attainable
    systematic content of the calibration and s2_h the orthogonal claim.
    """
    tickers = [t for t in inputs.assets.index if t not in held_at_matf]
    betas = inputs.betas.loc[tickers]
    resid_vol = inputs.assets.loc[tickers, 'resid_vol']
    attainable = float(compute_sharpe_accounting(inputs=inputs)['attainable'])

    rows, deviations = {}, {}
    for provider in excess_vectors.columns:
        decomposition = decompose_on_subset(mu_excess=excess_vectors.loc[tickers, provider],
                                            betas=betas, resid_vol=resid_vol)
        lam_gls = decomposition.attrs['lambda_gls']
        s2_h = decomposition.attrs['sr2_alpha']
        rows[provider] = {**{f"lambda_{es.factor_label(f)}_bp": 1e4 * lam_gls[f]
                             for f in betas.columns},
                          's2_h': s2_h,
                          's2_K': attainable,
                          'implied_tangency_sharpe': float(np.sqrt(attainable + s2_h))}
        deviations[provider] = decomposition['unattributed']
    table = pd.DataFrame(rows).T
    table.attrs['n_sleeves'] = len(tickers)
    table.attrs['excluded'] = list(held_at_matf)
    table.attrs['deviations'] = pd.DataFrame(deviations)
    return table


def build_unattributed_by_group(inputs,
                               deviations: pd.DataFrame,
                               provider: str,
                               ) -> pd.DataFrame:
    """unattributed return per reportable unit: single sleeves, plus aggregates for converted groups.

    The converted survey lines span several sleeves, so a per-sleeve number
    there would be an artefact of the mapping. Those sleeves are aggregated by
    residual-variance weights (the metric the claim is measured in) before
    printing.
    """
    assets = inputs.assets
    grouped_tickers = {t for group in CONVERTED_GROUPS.values() for t in group}
    rows = {}
    for ticker in deviations.index:
        if ticker in grouped_tickers:
            continue
        rows[assets.loc[ticker, 'sleeve']] = {
            'unattributed_bp': 1e4 * float(deviations.loc[ticker, provider]),
            'unattributed_ir': float(deviations.loc[ticker, provider]
                                     / assets.loc[ticker, 'resid_vol']),
            'reportable': 'sleeve'}
    for label, group in CONVERTED_GROUPS.items():
        present = [t for t in group if t in deviations.index]
        if not present:
            continue
        weights = 1.0 / assets.loc[present, 'resid_vol'] ** 2
        weights = weights / weights.sum()
        aggregate = float((deviations.loc[present, provider] * weights).sum())
        rows[label] = {'unattributed_bp': 1e4 * aggregate,
                       'unattributed_ir': float(np.sqrt(
                           ((deviations.loc[present, provider]
                             / assets.loc[present, 'resid_vol']) ** 2).sum())),
                       'reportable': 'aggregate only'}
    return pd.DataFrame(rows).T.sort_values('unattributed_bp')


def plot_provider_frontier(stats_by_mandate: Dict[str, pd.DataFrame],
                           scenario_band: Optional[pd.DataFrame] = None,
                           title: str = '',
                           figsize: Tuple[float, float] = (10.2, 6.0),
                           ) -> plt.Figure:
    """providers on the mandate frontier, with the MATF scenario band shaded where supplied."""
    fig, ax = plt.subplots(figsize=figsize)
    providers = list(next(iter(stats_by_mandate.values())).columns)
    colors = [es.BLUE, es.ORANGE, es.GREEN, es.DARK_RED, es.LIGHT_BLUE, es.GRAY]
    if scenario_band is not None:
        ax.fill_between(1e2 * scenario_band['vol'],
                        1e2 * scenario_band['stress_total'],
                        1e2 * scenario_band['upside_total'],
                        color=es.BLUE, alpha=0.10, zorder=1,
                        label='MATF 2022-stress to 2023-upside band')
    for color, provider in zip(colors, providers):
        vols = [1e2 * float(stats_by_mandate[m].loc['vol', provider])
                for m in stats_by_mandate]
        totals = [1e2 * float(stats_by_mandate[m].loc['total_return', provider])
                  for m in stats_by_mandate]
        ax.plot(vols, totals, color=color, lw=2.0, marker='o', ms=6.0, zorder=4, label=provider)
    ax.set_xlabel('Book volatility (% p.a.)', fontsize=10.0)
    ax.set_ylabel('Expected total return (% p.a.)', fontsize=10.0)
    es.style_axis(ax=ax, grid_axis='both', fontsize=9.0)
    ax.legend(fontsize=8.8, loc='lower right', frameon=False)
    ax.set_title(title, fontsize=11.5, loc='left')
    fig.tight_layout()
    return fig


def build_scenario_band(inputs, mandates: List[str]) -> pd.DataFrame:
    """the MATF base-optimized books repriced under both scenarios, one row per mandate."""
    assets = inputs.assets
    covar, mu_matf, rf_rate = build_moments(inputs=inputs)
    rows = {}
    for mandate in mandates:
        benchmark = get_benchmark(inputs=inputs, mandate=mandate)
        book = solve_mandate(covar=covar, cmas=mu_matf, benchmark_weights=benchmark)
        row = {'vol': float(np.sqrt(book @ covar.values @ book))}
        for scenario in ('stress', 'upside'):
            bump = inputs.factor_premia_scenarios[scenario]
            repriced = mu_matf + inputs.betas.values @ bump.values
            row[f"{scenario}_total"] = float(repriced @ book) + rf_rate
        rows[mandate] = row
    return pd.DataFrame(rows).T


def run_provider_exhibits(snapshot: str = SNAPSHOT,
                          save_outputs: bool = True,
                          ) -> Dict[str, pd.DataFrame]:
    """Phase 1: the Consensus column and decomposition row. Phase 2 reports its gate."""
    inputs = load_paper_inputs(snapshot=snapshot)
    print('=' * 78)
    print(f"Stage J4g phase 1 — Consensus provider, cut {snapshot}")
    print('=' * 78)

    consensus = build_consensus_column(inputs=inputs)
    held = list(consensus['held_at_matf'])
    print(f"\nheld at MATF (no survey line): "
          f"{[inputs.assets.loc[t, 'sleeve'] for t in held]}")
    print('\n--- tab:provider_saa Panel A, optimal weights (%) ---')
    panel_a = 1e2 * consensus['weights']
    panel_a.index = inputs.assets['sleeve']
    print(panel_a.round(1).to_string())
    print('\n--- tab:provider_saa Panel B, portfolio statistics ---')
    print(consensus['stats'].round(4).to_string())

    decomposition = build_provider_decomposition(inputs=inputs,
                                                excess_vectors=consensus['excess_vectors'],
                                                held_at_matf=held)
    print(f"\n--- tab:provider_decomposition (NEW), "
          f"{decomposition.attrs['n_sleeves']}-sleeve subset ---")
    print(decomposition.round(4).to_string())
    consensus_s2h = float(decomposition.loc[_cma_data.CONSENSUS_LABEL, 's2_h'])
    print(f"\nparity: Consensus orthogonal claim s2_h = {consensus_s2h:.3f} "
          f"(reference {R2_CONSENSUS_SR2_ALPHA:.3f})")
    if round(consensus_s2h, 3) != R2_CONSENSUS_SR2_ALPHA:
        raise ValueError(f"Consensus SR2_alpha off parity, got {consensus_s2h!r}")

    grouped = build_unattributed_by_group(inputs=inputs,
                                         deviations=decomposition.attrs['deviations'],
                                         provider=_cma_data.CONSENSUS_LABEL)
    print('\n--- Consensus unattributed return by reportable unit ---')
    print(grouped.round(1).to_string())

    print('\n--- Stage J4g phase 2 gate ---')
    try:
        providers = load_provider_vectors()
        print(f"providers.csv present: {providers['provider'].nunique()} providers, "
              f"{len(providers)} rows")
    except ValueError as error:
        print(f"BLOCKED: {error}")
        providers = None

    if save_outputs:
        write_provider_decomposition_tex(decomposition=decomposition, grouped=grouped,
                                        inputs=inputs)
        write_provider_saa_tex(weights=consensus['weights'], stats=consensus['stats'],
                              inputs=inputs)
        if providers is not None:
            band = build_scenario_band(inputs=inputs, mandates=WITH_ALTS_FAMILY)
            print(f"phase 2 frontiers would build here on {len(band)} mandates")
    return {'weights': consensus['weights'], 'stats': consensus['stats'],
            'decomposition': decomposition, 'grouped': grouped}


def write_provider_saa_tex(weights: pd.DataFrame,
                           stats: pd.DataFrame,
                           inputs,
                           file_name: str = 'exhibit_provider_saa.tex',
                           ) -> Path:
    """the Benchmark, MATF and Consensus columns of tab:provider_saa; A-D columns gated."""
    sleeves = inputs.assets['sleeve']
    bonds = inputs.assets.index[inputs.assets['asset_class'] == 'Bonds']
    lines = [
        '% ===== tab:provider_saa — Benchmark, MATF and Consensus columns =====',
        '% Source: replication/run_provider_exhibits.py on cma_data snapshot 2026q2.',
        '% Consensus = Horizon Actuarial 2025 survey averages via cma_data.consensus',
        '%   (printed 10Y ARITHMETIC nominal total returns, USD investor, Exhibit 17).',
        '% Insurance-Linked has no survey line and is held at the MATF value, so it carries',
        '%   no cross-provider tilt (R2 footnote convention).',
        '% The A-D provider columns are GATED on providers.csv (owner item O-J7b).',
        '%',
        '% PARTIAL BODY. Three value columns only (Bench, MATF, Consensus), so the tabular',
        '% preamble narrows from the R2 "l rrrrrr" to "l rrr" and the Panel spans from 7 to 4',
        '% until the A-D columns land. Re-run this script after providers.csv arrives and the',
        '% spans widen with the column set.',
        '',
        r"			\begin{tabular}{l rrr}",
        r"				\toprule",
        r"				& \textbf{Bench} & \textbf{MATF} & \textbf{Consensus} \\",
        r"				\midrule",
        r"				\multicolumn{4}{l}{\textit{Panel A: optimal weights (\%)}} \\",
    ]
    bond_row = 1e2 * weights.loc[bonds].sum()
    lines.append(f"\t\t\t\tFixed income (5 sleeves)  & {bond_row['Benchmark']:.1f} & "
                 f"{bond_row[MATF_LABEL]:.1f} & {bond_row[_cma_data.CONSENSUS_LABEL]:.1f} \\\\")
    for ticker in inputs.assets.index:
        if ticker in bonds:
            continue
        row = 1e2 * weights.loc[ticker]
        lines.append(f"\t\t\t\t{es.tex_escape(sleeves[ticker]):<25s} & "
                     f"{row['Benchmark']:.1f} & {row[MATF_LABEL]:.1f} & "
                     f"{row[_cma_data.CONSENSUS_LABEL]:.1f} \\\\")
    lines.append(r"				\midrule")
    lines.append(r"				\multicolumn{4}{l}{\textit{Panel B: portfolio statistics}} \\")
    for label, key, scale, decimals in (('Expected total return (\\%)', 'total_return', 1e2, 2),
                                        ('Volatility (\\%)', 'vol', 1e2, 1),
                                        ('Excess Sharpe ratio', 'excess_sharpe', 1.0, 2),
                                        ('Tracking error (\\%)', 'tracking_error', 1e2, 2)):
        lines.append(f"\t\t\t\t{label:<27s} & & "
                     f"{scale * float(stats.loc[key, MATF_LABEL]):.{decimals}f} & "
                     f"{scale * float(stats.loc[key, _cma_data.CONSENSUS_LABEL]):.{decimals}f} \\\\")
    lines.append(r"				\bottomrule")
    lines.append(r"			\end{tabular}")
    return es.write_fragment(lines=lines, file_name=file_name)


def write_provider_decomposition_tex(decomposition: pd.DataFrame,
                                     grouped: pd.DataFrame,
                                     inputs,
                                     file_name: str = 'exhibit_provider_decomposition.tex',
                                     ) -> Path:
    """the new tab:provider_decomposition (a table per owner decision O-J3)."""
    consensus_label = _cma_data.CONSENSUS_LABEL
    largest = grouped.loc[grouped['unattributed_bp'].abs().sort_values(ascending=False).index[:3]]
    lines = [
        '% ===== EXHIBIT: tab:provider_decomposition (NEW, table per owner decision O-J3) =====',
        '% Source: replication/run_provider_exhibits.py on cma_data snapshot 2026q2.',
        f"% Subset: {decomposition.attrs['n_sleeves']} published-plus-converted sleeves;",
        f"%   held-at-MATF sleeves excluded ({decomposition.attrs['excluded']}) because they",
        '%   contribute zero deviation by construction.',
        '% A-D rows are GATED on providers.csv (owner item O-J7b).',
        '',
        r"\begin{table}[H]",
        r"	\captionof{table}{Every Provider Vector Decomposes Into Implied Premia Plus an "
        r"Unattributed Return}\label{tab:provider_decomposition}\vspace*{-0.5\baselineskip}",
        r"	\begin{center}",
        r"		\footnotesize",
        r"		\begin{tabular}{l rrr rr r}",
        r"			\toprule",
        r"			\textbf{Provider} & $\lambda_{Eq}$ & $\lambda_{Rates}$ & $\lambda_{Credit}$ "
        r"& $s^2_h$ & $s^2_K$ & \textbf{Implied tangency} \\",
        r"			& (bp) & (bp) & (bp) & & & $\sqrt{s^2_K + s^2_h}$ \\",
        r"			\midrule",
    ]
    for provider, row in decomposition.iterrows():
        lines.append(f"\t\t\t{es.tex_escape(str(provider)):<12s} & "
                     f"{row['lambda_Equity_bp']:.0f} & {row['lambda_Rates_bp']:.0f} & "
                     f"{row['lambda_Credit_bp']:.0f} & {row['s2_h']:.3f} & "
                     f"{row['s2_K']:.3f} & {row['implied_tangency_sharpe']:.2f} \\\\")
    lines.append(r"			\bottomrule")
    lines.append(r"		\end{tabular}")
    lines.append(r"	\end{center}")
    lines.append(r"	\vspace*{-0.5\baselineskip}")
    lines.append(
        r"	{\footnotesize Notes: 2026-Q2 production cut, USD, "
        f"{decomposition.attrs['n_sleeves']}"
        r" sleeves. Each provider's excess CMA vector is regressed on the common loading matrix "
        r"in the $\boldsymbol D^{-1}$ metric, giving the implied factor premia "
        r"$\boldsymbol\lambda_{gls}$ and an orthogonal deviation whose squared Sharpe ratio "
        r"$s^2_h$ is the claimed idiosyncratic content. $s^2_K$ is the attainable systematic "
        r"content of the MATF calibration, common to every row because the risk model is held "
        r"fixed. Sleeves without a provider line are held at the MATF value and excluded, since "
        r"they contribute zero deviation by construction. Machinery in "
        r"\citep{SeppKastenholzFAJ2026}.}")
    lines.append(r"\end{table}")
    lines.append('%')
    lines.append('% Largest unattributed units for the 2.2 prose (aggregate-only where the survey')
    lines.append('% line spans several sleeves):')
    for unit, row in largest.iterrows():
        lines.append(f"%   {unit}: {row['unattributed_bp']:+.0f} bp ({row['reportable']})")
    return es.write_fragment(lines=lines, file_name=file_name)


class LocalTests(str, Enum):
    ALL_EXHIBITS = 'all_exhibits'
    DECOMPOSITION_ONLY = 'decomposition_only'


def run_local_test(local_test: LocalTests) -> None:
    """Run local tests for development and debugging purposes."""
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 300)

    if local_test == LocalTests.ALL_EXHIBITS:
        run_provider_exhibits()

    elif local_test == LocalTests.DECOMPOSITION_ONLY:
        inputs = load_paper_inputs()
        consensus = build_consensus_column(inputs=inputs)
        print(build_provider_decomposition(
            inputs=inputs, excess_vectors=consensus['excess_vectors'],
            held_at_matf=list(consensus['held_at_matf'])).round(4).to_string())

    else:
        raise NotImplementedError(f"{local_test}")


if __name__ == '__main__':
    run_local_test(local_test=LocalTests.ALL_EXHIBITS)
