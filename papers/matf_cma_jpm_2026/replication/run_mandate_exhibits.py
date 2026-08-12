"""
Mandate frontier exhibits for the eight benchmark mandates (roadmap Stage J4b).

  efficient_frontier.PNG  fig:illust_frontier — the two mandate families
      (without alternatives, with alternatives) in expected TOTAL return
      against volatility, each as the benchmark line and the TE-constrained
      optimum line, with every mandate labelled. The vertical gap between the
      two lines of a family is what the optimizer buys inside the guardrails;
      the horizontal gap between families is what the alternatives sleeve buys.
  factor_exposures.png    tab:factor_exposures — per mandate, the book factor
      exposures beta' w and the percentage risk contributions of each factor
      plus the residual.

Both regenerate on the D8-corrected benchmark of cma_data.benchmarks, so the
Asia ex-Japan and EM ex-Asia rows move against the R2 panels beyond the premia
effect.

Units: returns and volatilities are decimal per annum, printed as percent;
exposures are dimensionless betas; risk contributions are shares summing to
one.
Main entry point: run_local_test(local_test).

Does not belong here: the admission dial and the scenario repricing
(run_admission_exhibits.py) and the provider frontiers
(run_provider_exhibits.py), which reuse the same solve through
run_optimisation.
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
from governed_cma_projection import SNAPSHOT, load_paper_inputs
from run_optimisation import (MANDATE, build_moments, get_benchmark, solve_all_mandates,
                             compute_factor_exposures, compute_factor_risk_contributions)

_cma_data = load_cma_data()

FAMILIES: Dict[str, List[str]] = {
    'w/o Alternatives': ['Income w/o Alts', 'Low w/o Alts', 'Balanced w/o Alts', 'Growth w/o Alts'],
    'with Alternatives': ['Income with Alts', 'Low with Alts', 'Balanced with Alts',
                          'Growth with Alts'],
}
FAMILY_COLORS = {'w/o Alternatives': es.GRAY, 'with Alternatives': es.BLUE}
SHORT_LABELS = {'Income w/o Alts': 'Income', 'Low w/o Alts': 'Low',
                'Balanced w/o Alts': 'Balanced', 'Growth w/o Alts': 'Growth',
                'Income with Alts': 'Income', 'Low with Alts': 'Low',
                'Balanced with Alts': 'Balanced', 'Growth with Alts': 'Growth'}

# Compact factor headers so the exposure block and the risk block fit one printable
# width. Spelled out in the exhibit caption.
FACTOR_HEADERS: Dict[str, str] = {
    'Equity': 'Eq', 'Rates': 'Rt', 'Credit': 'Cr', 'Credit EM': 'CrEM',
    'Carry G10': 'CyG10', 'Carry EM': 'CyEM', 'Inflation': 'Inf',
    'Commodities': 'Cmd', 'Private Equity': 'PE', 'Rates Vol': 'RVol', 'Fx': 'FX',
}


def plot_mandate_frontier(stats: pd.DataFrame,
                          figsize: Tuple[float, float] = (10.2, 6.2),
                          ) -> plt.Figure:
    """the two mandate families, optimum against benchmark, in total-return space."""
    fig, ax = plt.subplots(figsize=figsize)
    for family, mandates in FAMILIES.items():
        color = FAMILY_COLORS[family]
        block = stats.loc[mandates]
        ax.plot(1e2 * block['benchmark_vol'], 1e2 * block['benchmark_total_return'],
                color=color, lw=1.4, ls=':', marker='o', ms=5.0, mfc='w', zorder=3,
                label=f"{family} — benchmark")
        ax.plot(1e2 * block['vol'], 1e2 * block['total_return'],
                color=color, lw=2.0, marker='o', ms=6.0, zorder=4,
                label=f"{family} — TE-constrained optimum")
        for mandate in mandates:
            ax.annotate(SHORT_LABELS[mandate],
                        xy=(1e2 * block.loc[mandate, 'vol'], 1e2 * block.loc[mandate, 'total_return']),
                        xytext=(4, 5), textcoords='offset points',
                        fontsize=8.6, color=color, zorder=5)

    ax.set_xlabel('Book volatility (% p.a.)', fontsize=10.0)
    ax.set_ylabel('Expected total return (% p.a.)', fontsize=10.0)
    es.style_axis(ax=ax, grid_axis='both', fontsize=9.0)
    ax.legend(fontsize=8.8, loc='lower right', frameon=False)
    ax.set_title('The alternatives sleeve moves the family; the guardrails bound '
                 'what the optimizer adds inside it', fontsize=11.5, loc='left')
    fig.tight_layout()
    return fig


def build_factor_exposures_table(weights: pd.DataFrame, inputs) -> pd.DataFrame:
    """per-mandate factor exposures and percentage risk contributions, canonical factor order."""
    order = list(inputs.betas.columns)
    rows = {}
    for mandate in weights.columns:
        exposures = compute_factor_exposures(weights=weights[mandate], inputs=inputs)
        risk = compute_factor_risk_contributions(weights=weights[mandate], inputs=inputs)
        row = {}
        for factor in order:
            row[FACTOR_HEADERS[factor]] = f"{exposures[factor]:.2f}"
        for factor in order:
            row[f"{FACTOR_HEADERS[factor]}\nrisk"] = f"{1e2 * risk[factor]:.0f}%"
        row['Resid\nrisk'] = f"{1e2 * risk['Residual']:.0f}%"
        rows[mandate] = row
        # risk shares must sum to one by construction
        total = float(risk.sum())
        if abs(total - 1.0) > 1e-10:
            raise ValueError(f"risk shares do not sum to one for {mandate!r}, got {total!r}")
    return pd.DataFrame.from_dict(rows, orient='index')


def plot_factor_exposures_table(table: pd.DataFrame, inputs) -> plt.Figure:
    """render the exposures block and the risk-share block side by side, one row per mandate."""
    n_factors = len(inputs.betas.columns)
    widths = [1.62] + [0.46] * n_factors + [0.50] * (n_factors + 1)
    # qis draws the edge at the left border of column (id - 1); the index occupies column 0,
    # so the exposures / risk-share separator sits at n_factors + 2
    return es.table_figure(df=table, col_widths=widths, row_height=0.44, fontsize=7.4,
                           special_columns_colors=[(0, '#c6e2f0')],
                           columns_edge_lines=[(n_factors + 2, es.BLUE)],
                           first_row_height=0.45)


def run_mandate_exhibits(snapshot: str = SNAPSHOT,
                         save_outputs: bool = True,
                         ) -> Dict[str, pd.DataFrame]:
    """build and write the two Stage J4b exhibits."""
    inputs = load_paper_inputs(snapshot=snapshot)
    print('=' * 78)
    print(f"Stage J4b — mandate frontier family, cut {snapshot}")
    print('=' * 78)

    weights, stats = solve_all_mandates(inputs=inputs)
    print('\n--- mandate statistics (decimal p.a.) ---')
    print(stats.round(4).to_string())

    exposures_table = build_factor_exposures_table(weights=weights, inputs=inputs)
    print('\n--- tab:factor_exposures ---')
    print(exposures_table.to_string())

    balanced = stats.loc[MANDATE]
    print(f"\n{MANDATE}: benchmark total {balanced['benchmark_total_return']:.2%} -> "
          f"optimum total {balanced['total_return']:.2%} at TE "
          f"{balanced['tracking_error']:.2%}, benchmark vol {balanced['benchmark_vol']:.2%}")

    if save_outputs:
        es.save_figure(plot_mandate_frontier(stats=stats), 'efficient_frontier.PNG')
        es.save_figure(plot_factor_exposures_table(table=exposures_table, inputs=inputs),
                       'factor_exposures.png')
        write_mandate_notes(stats=stats)
    return {'weights': weights, 'stats': stats, 'exposures': exposures_table}


def write_mandate_notes(stats: pd.DataFrame,
                        file_name: str = 'exhibit_mandate_notes.tex',
                        ) -> Path:
    """caption notes for the two restored mandate exhibits."""
    balanced = stats.loc[MANDATE]
    lines = [
        '% ===== Caption notes for the Stage J4b mandate exhibits =====',
        '% Source: replication/run_mandate_exhibits.py on cma_data snapshot 2026q2.',
        '%',
        '% fig:illust_frontier — expected TOTAL returns (excess CMA plus the reference cash',
        '%   rate). Dotted lines are the mandate benchmarks, solid lines the TE-constrained',
        '%   optima under a +-50% box and a 1.5% tracking-error cap.',
        '%',
        '% tab:factor_exposures — the left block is beta\' w, the right block the percentage',
        '%   contribution of each factor to book variance plus the residual, summing to 100%.',
        '%',
        '% BOTH exhibits regenerate on the D8-corrected benchmark of cma_data/benchmarks.py.',
        '%   The R2 panels solved against a benchmark whose Asia ex-Japan and EM ex-Asia',
        '%   weights were transposed, so those two rows move for a reason separate from the',
        '%   July premia config.',
        '%',
        f"% {MANDATE}: benchmark vol {balanced['benchmark_vol']:.2%} (R2 quoted 9.3%),",
        f"%   benchmark total {balanced['benchmark_total_return']:.2%}, optimum total",
        f"%   {balanced['total_return']:.2%}, realized tracking error {balanced['tracking_error']:.2%}.",
    ]
    return es.write_fragment(lines=lines, file_name=file_name)


class LocalTests(str, Enum):
    ALL_EXHIBITS = 'all_exhibits'


def run_local_test(local_test: LocalTests) -> None:
    """Run local tests for development and debugging purposes."""
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 300)

    if local_test == LocalTests.ALL_EXHIBITS:
        run_mandate_exhibits()
    else:
        raise NotImplementedError(f"{local_test}")


if __name__ == '__main__':
    run_local_test(local_test=LocalTests.ALL_EXHIBITS)
