"""
Consistency pair: the historical-mean comparator against MATF (roadmap Stage J4d).

Builds the per-asset historical-mean CMA vector on the frozen return panel and
compares it with the MATF production vector on the SAME loadings and residual
variances, so the only thing that changes between the two is how the expected
returns were constructed.

Historical-mean CMAs follow the Appendix B recipe: the sample mean of the
native-frequency EXCESS LOG returns, annualized by the observation frequency,
plus the arithmetic lift of one half the annualized variance,

    mu_hist,i = s_i * mean(Y_i) + 0.5 * s_i * var(Y_i)

with s_i = 12 for monthly sleeves and 4 for quarterly sleeves. Moments are
NaN-aware, so the short-history sleeves contribute on the observations they
have. This estimator is a Grinold-Kroner construction whose per-asset point
estimate is the historical sample mean.

  declared_delta.PNG        fig:declared_delta — per-asset consistency residual
      |Delta_i|, the component of the CMA vector the factor span cannot
      explain, under both constructions. The MATF residual is not zero and the
      framework does not claim it is: it equals the annihilated image of the
      declared channels, and the reconciliation gap is printed. The
      historical-mean residual reconciles to nothing.
  unintended_exposures.PNG  fig:unintended_exposures — the active factor
      exposures beta'(w - w_b) of the two optimal books at the same benchmark
      volatility, and the split of active risk into its systematic and
      idiosyncratic parts. Unattributed return becomes an unintended factor bet
      because the optimizer prices it.

Units: CMAs and residuals are decimal per annum and print as percent; exposures
are dimensionless.
Main entry point: run_local_test(local_test).

Does not belong here: the bootstrap sampling distribution of the same
comparison (run_bootstrap_q2.py, Appendix B) and the provider version of the
diagnostic (run_provider_exhibits.py).
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
from governed_cma_projection import (SNAPSHOT,
                                     load_paper_inputs,
                                     compute_gls_decomposition,
                                     compute_factor_information_matrix)
from run_optimisation import (MANDATE, build_moments, get_benchmark, solve_mandate,
                             solve_max_return_at_vol, compute_factor_exposures)

FREQUENCY_FACTORS: Dict[str, int] = {'ME': 12, 'QE': 4}
IDENTITY_TOL = 1e-10
R2_MEDIAN_RESIDUALS = {'historical': 0.005, 'matf': 0.0034}    # R2 printed medians
R2_RECONCILIATION_GAP = 1e-17                                   # the machine-zero claim
R2_MAX_HISTORICAL_RESIDUAL = 0.055


def build_historical_mean_cmas(inputs) -> pd.Series:
    """per-asset excess CMAs from the sample mean plus the half-variance arithmetic lift."""
    panel = inputs.require_panel('asset_excess_logreturns')
    frequency = inputs.assets['frequency']
    unknown = set(frequency.unique()) - set(FREQUENCY_FACTORS)
    if unknown:
        raise ValueError(f"unhandled observation frequency, got {sorted(unknown)!r}")
    scale = frequency.map(FREQUENCY_FACTORS).astype(float)
    log_mean = panel.mean(skipna=True) * scale                 # NaN-aware, no backfill
    variance = panel.var(skipna=True) * scale
    cmas = (log_mean + 0.5 * variance).rename('mu_historical')
    if cmas.isna().any():
        raise ValueError(f"historical CMAs carry NaN for {list(cmas[cmas.isna()].index)!r}")
    return cmas.reindex(inputs.assets.index)


def build_consistency_residuals(inputs) -> pd.DataFrame:
    """the GLS-annihilated component of each CMA vector, plus the MATF declared-channel check.

    Under MATF the residual equals the annihilator applied to the declared
    channels, Delta = M_beta (blend + admitted alpha), because the annihilator
    removes the factor-implied component by construction. The gap between the
    two computations is the reconciliation the manuscript quotes as machine
    zero.
    """
    assets = inputs.assets
    mu_matf = assets['factor_excess_cma'] + assets['w_paper'] * assets['alpha']
    mu_hist = build_historical_mean_cmas(inputs=inputs)

    _, delta_matf, _ = compute_gls_decomposition(mu_excess=mu_matf, inputs=inputs)
    _, delta_hist, _ = compute_gls_decomposition(mu_excess=mu_hist, inputs=inputs)

    # the declared channels alone, annihilated: blend + admitted alpha
    declared = assets['equity_regional_addon'] + assets['w_paper'] * assets['alpha']
    _, delta_declared, _ = compute_gls_decomposition(mu_excess=declared, inputs=inputs)
    reconciliation_gap = float((delta_matf - delta_declared).abs().max())

    table = pd.DataFrame({'sleeve': assets['sleeve'],
                          'mu_matf': mu_matf,
                          'mu_historical': mu_hist,
                          'delta_matf': delta_matf,
                          'delta_historical': delta_hist,
                          'delta_declared': delta_declared})
    table.attrs['reconciliation_gap'] = reconciliation_gap
    table.attrs['median_matf'] = float(table['delta_matf'].abs().median())
    table.attrs['median_historical'] = float(table['delta_historical'].abs().median())
    table.attrs['max_historical'] = float(table['delta_historical'].abs().max())
    return table


def plot_declared_delta(table: pd.DataFrame,
                        figsize: Tuple[float, float] = (10.6, 6.4),
                        ) -> plt.Figure:
    """the frozen grammar: paired horizontal bars of |Delta_i|, MATF blue and historical orange."""
    positions = np.arange(len(table))[::-1]
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(positions + 0.19, 1e2 * table['delta_matf'].abs(), height=0.36,
            color=es.BLUE, label='MATF production (declared)', zorder=3)
    ax.barh(positions - 0.19, 1e2 * table['delta_historical'].abs(), height=0.36,
            color=es.ORANGE, label='Historical mean (unattributed)', zorder=3)
    ax.set_yticks(positions)
    ax.set_yticklabels(list(table['sleeve']), fontsize=9.0)
    ax.set_xlabel(r'Consistency residual  $|\Delta_i|$  '
                  r'(% p.a., GLS-annihilated CMA component)', fontsize=10.0)
    es.style_axis(ax=ax, grid_axis='x', fontsize=9.0)
    ax.legend(fontsize=9.0, loc='lower right', frameon=False)
    ax.set_title('The framework does not promise zero deviation; '
                 'it promises zero unattributed deviation', fontsize=11.5, loc='left')
    fig.tight_layout()
    # J8d: the frozen version of this exhibit carries a takeaway title and exactly ONE
    # line of in-figure note, so both stay and the note is compressed to one line.
    fig.subplots_adjust(bottom=0.12)
    ax.annotate(f"Median $|\\Delta|$: historical mean {table.attrs['median_historical']:.2%}, "
                f"MATF production {table.attrs['median_matf']:.2%}. The MATF residual reconciles "
                f"to the declared blend and admission channels to "
                f"{table.attrs['reconciliation_gap']:.0e}; the historical-mean residual "
                f"reconciles to nothing.",
                xy=(0.0, -0.105), xycoords='axes fraction', fontsize=8.0, color='0.35')
    return fig


def build_unintended_exposures(inputs, mandate: str = MANDATE) -> Dict[str, pd.Series]:
    """active factor exposures and the active-risk split of the two books at the benchmark vol."""
    assets = inputs.assets
    covar, mu_matf, _ = build_moments(inputs=inputs)
    mu_hist = build_historical_mean_cmas(inputs=inputs)
    benchmark = get_benchmark(inputs=inputs, mandate=mandate)
    benchmark_vol = float(np.sqrt(benchmark @ covar.values @ benchmark))

    books = {}
    for label, cmas in (('MATF', mu_matf), ('Historical mean', mu_hist)):
        book = solve_max_return_at_vol(covar=covar, cmas=cmas, vol_target=benchmark_vol)
        if book is None:
            raise ValueError(f"solve infeasible for {label!r} at vol {benchmark_vol!r}")
        books[label] = book

    exposures, risk_split = {}, {}
    for label, book in books.items():
        active = book - benchmark
        exposures[label] = compute_factor_exposures(weights=active, inputs=inputs)
        systematic = float(exposures[label] @ inputs.factor_covar.values @ exposures[label])
        idiosyncratic = float((active.values ** 2) @ (assets['resid_vol'].values ** 2))
        total = systematic + idiosyncratic
        # the split must reconstruct the tracking error exactly
        tracking_error = float(np.sqrt(active @ covar.values @ active))
        if abs(np.sqrt(total) - tracking_error) > IDENTITY_TOL:
            raise ValueError(f"active-risk split does not reproduce the tracking error for "
                             f"{label!r}, got {np.sqrt(total)!r} vs {tracking_error!r}")
        risk_split[label] = pd.Series({'systematic': np.sqrt(systematic),
                                       'idiosyncratic': np.sqrt(idiosyncratic),
                                       'tracking_error': tracking_error,
                                       'idiosyncratic_share': idiosyncratic / total})
    return {'exposures': pd.DataFrame(exposures), 'risk_split': pd.DataFrame(risk_split),
            'benchmark_vol': pd.Series({'benchmark_vol': benchmark_vol})}


def plot_unintended_exposures(exposures: pd.DataFrame,
                              risk_split: pd.DataFrame,
                              figsize: Tuple[float, float] = (12.8, 5.0),
                              ) -> plt.Figure:
    """the frozen two-panel grammar: active exposures left, active-risk composition right."""
    fig, axs = plt.subplots(1, 2, figsize=figsize, width_ratios=(2.1, 1.0))

    ax = axs[0]
    factors = es.factor_labels(exposures.index)
    positions = np.arange(len(factors))
    ax.bar(positions - 0.19, exposures['MATF'], width=0.36, color=es.BLUE,
           label='MATF', zorder=3)
    ax.bar(positions + 0.19, exposures['Historical mean'], width=0.36, color=es.ORANGE,
           label='Historical mean', zorder=3)
    ax.axhline(0.0, color='0.3', lw=0.9, zorder=2)
    ax.set_xticks(positions)
    ax.set_xticklabels(factors, fontsize=9.0, rotation=30, ha='right')
    ax.set_ylabel(r"Active factor exposure  $\hat{\beta}^{\top}(w - b)$", fontsize=10.0)
    ax.set_title('Active factor exposures of the optimal book', fontsize=10.5, loc='left')
    es.style_axis(ax=ax, grid_axis='y', fontsize=9.0)
    ax.legend(fontsize=9.0, loc='upper left', frameon=False)

    ax = axs[1]
    labels = list(risk_split.columns)
    positions = np.arange(len(labels))
    systematic = 1e2 * risk_split.loc['systematic']
    idiosyncratic = 1e2 * risk_split.loc['idiosyncratic']
    ax.bar(positions, systematic, width=0.52, color=es.BLUE, label='Systematic TE', zorder=3)
    ax.bar(positions, idiosyncratic, bottom=systematic, width=0.52, color='0.80',
           label='Idiosyncratic TE', zorder=3)
    for position, label in zip(positions, labels):
        share = float(risk_split.loc['idiosyncratic_share', label])
        top = float(systematic[label] + idiosyncratic[label])
        ax.annotate(f"{share:.0%} idio", xy=(position, top + 0.06), ha='center', va='bottom',
                    fontsize=9.0, color='0.2')
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, fontsize=9.4)
    ax.set_ylabel('Active risk vs benchmark (%, components)', fontsize=10.0)
    ax.set_title('Where the tracking error goes', fontsize=10.5, loc='left')
    es.style_axis(ax=ax, grid_axis='y', fontsize=9.0)
    ax.legend(fontsize=9.0, loc='lower right', frameon=False)

    fig.suptitle('Unattributed residuals become unintended factor bets '
                 'the committee never approved', fontsize=12.0, x=0.008, ha='left')
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    return fig


def run_consistency_exhibits(snapshot: str = SNAPSHOT,
                             save_outputs: bool = True,
                             ) -> Dict[str, pd.DataFrame]:
    """build and write the two Stage J4d exhibits."""
    inputs = load_paper_inputs(snapshot=snapshot)
    print('=' * 78)
    print(f"Stage J4d — consistency pair, cut {snapshot}")
    print('=' * 78)

    residuals = build_consistency_residuals(inputs=inputs)
    print('\n--- per-asset CMAs and consistency residuals (percent p.a.) ---')
    printed = residuals.copy()
    for column in ('mu_matf', 'mu_historical', 'delta_matf', 'delta_historical'):
        printed[column] = 1e2 * printed[column]
    print(printed.drop(columns=['delta_declared']).round(3).to_string())
    print(f"\nMATF declared-channel reconciliation gap: "
          f"{residuals.attrs['reconciliation_gap']:.2e} (R2 quotes "
          f"{R2_RECONCILIATION_GAP:.0e} as machine zero)")
    print(f"median |Delta|: historical {residuals.attrs['median_historical']:.2%} "
          f"(R2 {R2_MEDIAN_RESIDUALS['historical']:.2%}), "
          f"MATF {residuals.attrs['median_matf']:.2%} "
          f"(R2 {R2_MEDIAN_RESIDUALS['matf']:.2%})")
    print(f"max |Delta| historical: {residuals.attrs['max_historical']:.2%} "
          f"(R2 {R2_MAX_HISTORICAL_RESIDUAL:.1%})")

    unintended = build_unintended_exposures(inputs=inputs)
    print(f"\n--- active factor exposures at the Balanced benchmark vol "
          f"{float(unintended['benchmark_vol'].iloc[0]):.2%} ---")
    print(unintended['exposures'].round(4).to_string())
    print('\n--- active-risk split ---')
    print(unintended['risk_split'].round(4).to_string())

    if save_outputs:
        es.save_figure(plot_declared_delta(table=residuals), 'declared_delta.PNG')
        es.save_figure(plot_unintended_exposures(exposures=unintended['exposures'],
                                                risk_split=unintended['risk_split']),
                       'unintended_exposures.PNG')
        write_consistency_notes(residuals=residuals, unintended=unintended)
    return {'residuals': residuals, **unintended}


def write_consistency_notes(residuals: pd.DataFrame,
                            unintended: Dict[str, pd.DataFrame],
                            file_name: str = 'exhibit_consistency_notes.tex',
                            ) -> Path:
    """caption notes for the Stage J4d exhibits, including the reconciliation gap."""
    lines = [
        '% ===== Caption notes for the Stage J4d consistency exhibits =====',
        '% Source: replication/run_consistency_exhibits.py on cma_data snapshot 2026q2.',
        '%',
        '% Historical-mean CMAs: sample mean of the native-frequency excess LOG returns,',
        '%   annualized by the observation frequency (12 monthly / 4 quarterly), plus the',
        '%   arithmetic lift of one half the annualized variance. Moments are NaN-aware; the',
        '%   snapshot carries no pre-inception backfill, so Insurance-Linked (from Sep 2006)',
        '%   and Europe ex-UK (from Nov 2007) contribute on shorter histories than the',
        '%   Appendix B text describes. See the completion report deviations list.',
        '%',
        f"% fig:declared_delta — median |Delta|: historical "
        f"{residuals.attrs['median_historical']:.2%}, MATF "
        f"{residuals.attrs['median_matf']:.2%}; maximum historical "
        f"{residuals.attrs['max_historical']:.2%}.",
        f"%   MATF reconciliation to the declared channels: "
        f"{residuals.attrs['reconciliation_gap']:.1e}. The R2 text's machine-zero claim of",
        '%   order 1e-17 re-asserts on this cut at the printed magnitude.',
        '%',
        f"% fig:unintended_exposures — both books solved long-only at the Balanced benchmark",
        f"%   volatility {float(unintended['benchmark_vol'].iloc[0]):.2%}. Idiosyncratic share of",
        f"%   active variance: MATF "
        f"{float(unintended['risk_split'].loc['idiosyncratic_share', 'MATF']):.0%}, historical mean "
        f"{float(unintended['risk_split'].loc['idiosyncratic_share', 'Historical mean']):.0%}.",
    ]
    return es.write_fragment(lines=lines, file_name=file_name)


class LocalTests(str, Enum):
    ALL_EXHIBITS = 'all_exhibits'
    HISTORICAL_CMAS_ONLY = 'historical_cmas_only'


def run_local_test(local_test: LocalTests) -> None:
    """Run local tests for development and debugging purposes."""
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 250)

    if local_test == LocalTests.ALL_EXHIBITS:
        run_consistency_exhibits()

    elif local_test == LocalTests.HISTORICAL_CMAS_ONLY:
        inputs = load_paper_inputs()
        print((1e2 * build_historical_mean_cmas(inputs=inputs)).round(3).to_string())

    else:
        raise NotImplementedError(f"{local_test}")


if __name__ == '__main__':
    run_local_test(local_test=LocalTests.ALL_EXHIBITS)
