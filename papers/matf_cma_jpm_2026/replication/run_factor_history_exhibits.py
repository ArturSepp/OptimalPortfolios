"""
Factor-history exhibits from the frozen factor NAV panel (roadmap Stage J2).

Restores the three Appendix C exhibits that read the realized factor record.
The NAVs are EXCESS-return NAVs (base 100), so every statistic below is an
excess-return statistic and Sharpe ratios carry rf = 0 by construction:

  risk_factors_perf.PNG    tb:risk_factors_perf — per factor: annualized
      return, volatility, Sharpe (rf = 0), maximum drawdown, skewness, and
      vs-Equity regression statistics (annualized alpha, beta, R^2, p-value
      of alpha), Dec 2004 - Jun 2026. The empirical-Sharpe column of
      tab:sharpe_cal is written out here and joined by run_snapshot_tables.
  risk_factors_corr.PNG    tb:risk_factors_corr — EWMA correlation matrix with
      annualized EWMA volatilities on the diagonal, on WEEKLY W-WED returns
      with span 260. That is the production covar spec of the manifest
      (covar_estimation_spec.factor_returns_freq = W-WED,
      factor_covar_span = 260), NOT the monthly span 36 the R2 caption
      claims: defect D7. Both specs print to the console for the owner's
      diff; only the production-spec PNG is written.
  risk_factors_annual.PNG  tb:risk_factors_annual — calendar-year excess
      returns 2005 to 2026 YTD as a heatmap table.

Units: returns, volatilities, alphas and drawdowns are decimal per annum and
print as percent; Sharpe, beta, R^2, skewness and p-values are dimensionless.
Main entry point: run_local_test(local_test).

Does not belong here: the factor premia (a config quantity, Stage J1) and any
asset-level statistic (Stage J3).
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
from qis import TimePeriod, PerfParams, PerfStat
import exhibit_style as es
from governed_cma_projection import SNAPSHOT, load_paper_inputs
from run_snapshot_tables import build_factor_returns

HISTORY_START = '2004-12-31'      # last business day of 2004, so calendar 2005 is complete
HISTORY_END = '2026-06-30'
BENCHMARK_FACTOR = 'Equity'       # the regression benchmark of tb:risk_factors_perf

# production covariance spec, from MANIFEST prod_config_snapshot (defect D7)
CORR_FREQ = 'W-WED'
CORR_SPAN = 260                   # weeks, ~5 years
CORR_ANNUALISATION = 52.0
# the spec the R2 caption claims, printed for the owner's diff only
LEGACY_CORR_FREQ = 'ME'
LEGACY_CORR_SPAN = 36
LEGACY_ANNUALISATION = 12.0

PERF_COLUMNS: List[PerfStat] = [PerfStat.PA_RETURN, PerfStat.VOL, PerfStat.SHARPE_RF0,
                                PerfStat.MAX_DD, PerfStat.SKEWNESS, PerfStat.ALPHA_AN,
                                PerfStat.BETA, PerfStat.R2, PerfStat.ALPHA_PVALUE]

# R2 printed values of tab:sharpe_cal's empirical-SR column, for the reported diff
R2_EMPIRICAL_SR: Dict[str, float] = {
    # Custom-factor values computed from this script's 2005-2026-Q2 output on 2026-08-12.
    'Credit EM': 0.175499, 'Carry G10': 0.164557, 'Carry EM': 0.514247,
    'Inflation': 0.05, 'Commodities': -0.07, 'Private Equity': 0.55, 'Rates Vol': 0.46,
}
R2_CREDIT_EQUITY_CORR = 0.9       # the value the R2 text quotes
EMPIRICAL_SR_FILE = 'factor_empirical_sr.csv'


def load_factor_navs(snapshot: str = SNAPSHOT,
                     time_period: Optional[TimePeriod] = None,
                     ) -> pd.DataFrame:
    """the daily excess-return factor NAVs on a business-day grid, sliced to the exhibit window."""
    inputs = load_paper_inputs(snapshot=snapshot)
    navs = inputs.require_panel('factor_navs').asfreq('B', method='ffill')
    period = TimePeriod(HISTORY_START, HISTORY_END) if time_period is None else time_period
    navs = period.locate(navs)
    if navs.empty:
        raise ValueError(f"no factor NAVs inside the exhibit window, got {period!r}")
    return navs


# --------------------------------------------------------------------------
# tb:risk_factors_perf
# --------------------------------------------------------------------------

def build_factor_performance(navs: pd.DataFrame) -> pd.DataFrame:
    """per-factor excess-return performance and vs-Equity regression statistics."""
    if BENCHMARK_FACTOR not in navs.columns:
        raise ValueError(f"benchmark factor absent, got {list(navs.columns)!r}")
    perf_params = PerfParams(freq='ME')
    table = qis.compute_ra_perf_table_with_benchmark(prices=navs,
                                                     benchmark=BENCHMARK_FACTOR,
                                                     perf_params=perf_params)
    return table[[stat.to_str() for stat in PERF_COLUMNS]].reindex(navs.columns)


def format_factor_performance(table: pd.DataFrame) -> pd.DataFrame:
    """the printed exhibit frame: percent, ratio, and p-value formatting per column."""
    formatted = pd.DataFrame(index=[es.factor_label(f) for f in table.index])
    formatted['P.a. return'] = [f"{1e2 * v:.2f}%" for v in table[PerfStat.PA_RETURN.to_str()]]
    formatted['Vol'] = [f"{1e2 * v:.2f}%" for v in table[PerfStat.VOL.to_str()]]
    formatted['Sharpe (rf=0)'] = [f"{v:.2f}" for v in table[PerfStat.SHARPE_RF0.to_str()]]
    formatted['Max DD'] = [f"{1e2 * v:.0f}%" for v in table[PerfStat.MAX_DD.to_str()]]
    formatted['Skewness'] = [f"{v:.2f}" for v in table[PerfStat.SKEWNESS.to_str()]]
    formatted['An Alpha'] = [f"{1e2 * v:.2f}%" for v in table[PerfStat.ALPHA_AN.to_str()]]
    formatted['Beta'] = [f"{v:.2f}" for v in table[PerfStat.BETA.to_str()]]
    formatted['R2'] = [f"{1e2 * v:.0f}%" for v in table[PerfStat.R2.to_str()]]
    formatted['p-Alpha'] = [f"{v:.2f}" for v in table[PerfStat.ALPHA_PVALUE.to_str()]]
    return formatted


def write_empirical_sr(table: pd.DataFrame,
                       local_path: Optional[Path] = None,     # default: replication/figures
                       ) -> Path:
    """the empirical Sharpe column consumed by tab:sharpe_cal in Stage J1."""
    folder = es.FRAGMENTS_PATH if local_path is None else Path(local_path)
    folder.mkdir(parents=True, exist_ok=True)
    file_path = folder / EMPIRICAL_SR_FILE
    table[PerfStat.SHARPE_RF0.to_str()].rename('empirical_sr').to_csv(file_path)
    print(f"empirical SR column written: {file_path}")
    return file_path


# --------------------------------------------------------------------------
# tb:risk_factors_corr
# --------------------------------------------------------------------------

def build_correlation_matrix(navs: pd.DataFrame,
                             freq: str = CORR_FREQ,
                             span: int = CORR_SPAN,
                             annualisation: float = CORR_ANNUALISATION,
                             ) -> pd.DataFrame:
    """EWMA correlation matrix with annualized EWMA volatilities on the diagonal.

    Correlations and volatilities come from one EWMA covariance on returns at
    `freq`, so the matrix is internally consistent: off-diagonal entries are
    correlations, diagonal entries are annualized volatilities.
    """
    if span <= 1:
        raise ValueError(f"span must exceed one period, got {span!r}")
    returns = qis.to_returns(prices=navs, freq=freq, drop_first=True, is_log_returns=False)
    covar = qis.compute_ewm_covar(a=returns.to_numpy(), span=span)
    vols = np.sqrt(np.diag(covar))
    corr = covar / np.outer(vols, vols)
    np.fill_diagonal(corr, vols * np.sqrt(annualisation))
    labels = es.factor_labels(navs.columns)
    matrix = pd.DataFrame(corr, index=labels, columns=labels)
    matrix.attrs['freq'] = freq
    matrix.attrs['span'] = span
    return matrix


def plot_correlation_matrix(matrix: pd.DataFrame,
                            figsize: Tuple[float, float] = (9.2, 6.2),
                            fontsize: int = 9,
                            ) -> plt.Figure:
    """lower-triangular correlation heatmap with annualized vols on the diagonal.

    Keeps the frozen exhibit's grammar: PiYG diverging map centred at zero,
    upper triangle blank, rotated column headers, diagonal cells drawn neutral
    and outlined so the volatilities read as a different quantity from the
    correlations around them.
    """
    vols = pd.Series(np.diag(matrix.values), index=matrix.index)
    values = matrix.values.copy()
    np.fill_diagonal(values, 0.0)                 # neutral in the diverging map, text overlaid below
    lower = pd.DataFrame(values, index=matrix.index, columns=matrix.columns)
    lower = lower.mask(np.triu(np.ones(lower.shape, dtype=bool), k=1))
    annotations = np.array(
        [['' if j > i else (f"{1e2 * vols.iloc[i]:.1f}%" if i == j else f"{lower.iloc[i, j]:.2f}")
          for j in range(len(lower))] for i in range(len(lower))], dtype=object)

    fig, ax = plt.subplots(figsize=figsize)
    qis.plot_heatmap(df=lower, cmap='PiYG', var_format=None, annot=annotations,
                     fontsize=fontsize, vmin=-1.0, vmax=1.0, ax=ax)
    for i in range(len(lower)):                   # outline the volatility cells
        ax.add_patch(plt.Rectangle((i, i), 1.0, 1.0, fill=False,
                                   edgecolor=es.BLUE, lw=1.4, zorder=5))
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# tb:risk_factors_annual
# --------------------------------------------------------------------------

def build_annual_returns(navs: pd.DataFrame,
                         add_total_column: bool = True,
                         ) -> pd.DataFrame:
    """calendar-year excess returns per factor, plus the window total, factors down."""
    year_end = navs.resample('YE').last()
    annual = year_end.pct_change().dropna(how='all')
    annual.index = [str(date.year) for date in annual.index]
    annual.columns = es.factor_labels(annual.columns)
    table = annual.T                      # factors down, years across, as in the frozen exhibit
    if add_total_column:
        table['Total'] = navs.iloc[-1].values / navs.iloc[0].values - 1.0
    return table


def plot_annual_returns(annual: pd.DataFrame,
                        figsize: Tuple[float, float] = (16.0, 4.0),
                        fontsize: int = 8,
                        ) -> plt.Figure:
    """calendar-year heatmap in the frozen exhibit's grammar (RdYlGn, rotated year headers)."""
    fig, ax = plt.subplots(figsize=figsize)
    qis.plot_heatmap(df=annual, cmap='RdYlGn', var_format='{:.0%}', fontsize=fontsize,
                     date_format=None, vmin=-0.45, vmax=0.45, ax=ax)
    total_column = list(annual.columns).index('Total') if 'Total' in annual.columns else None
    if total_column is not None:          # separate the window total from the calendar years
        ax.axvline(total_column, color=es.BLUE, lw=1.4, zorder=5)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def run_factor_history_exhibits(snapshot: str = SNAPSHOT,
                                save_outputs: bool = True,
                                ) -> Dict[str, pd.DataFrame]:
    """build and write the three Stage J2 exhibits, with the D7 spec diff printed."""
    navs = load_factor_navs(snapshot=snapshot)
    print('=' * 78)
    print(f"Stage J2 — factor-history exhibits, cut {snapshot}, "
          f"{navs.index[0].date()} to {navs.index[-1].date()}")
    print('=' * 78)

    performance = build_factor_performance(navs=navs)
    print('\n--- tb:risk_factors_perf ---')
    print(performance.round(4).to_string())
    empirical_sr = performance[PerfStat.SHARPE_RF0.to_str()]
    diff = pd.DataFrame({'r2_printed': pd.Series(R2_EMPIRICAL_SR),
                         'regenerated': empirical_sr.reindex(R2_EMPIRICAL_SR)})
    diff['delta'] = diff['regenerated'] - diff['r2_printed']
    print('\nempirical SR for tab:sharpe_cal, R2 printed vs regenerated:')
    print(diff.round(2).to_string())

    production_corr = build_correlation_matrix(navs=navs)
    legacy_corr = build_correlation_matrix(navs=navs, freq=LEGACY_CORR_FREQ,
                                           span=LEGACY_CORR_SPAN,
                                           annualisation=LEGACY_ANNUALISATION)
    print(f"\n--- tb:risk_factors_corr, PRODUCTION spec "
          f"({CORR_FREQ}, span {CORR_SPAN}) — this is what is written ---")
    print(production_corr.round(2).to_string())
    print(f"\n--- tb:risk_factors_corr, LEGACY spec claimed by the R2 caption "
          f"({LEGACY_CORR_FREQ}, span {LEGACY_CORR_SPAN}) — console only, defect D7 ---")
    print(legacy_corr.round(2).to_string())
    print("\nmax |production - legacy| off-diagonal correlation: "
          f"{float((production_corr - legacy_corr).abs().where(~np.eye(len(production_corr), dtype=bool)).max().max()):.2f}")
    credit_equity = float(production_corr.loc['Credit', 'Equity'])
    print(f"Credit-Equity correlation: production spec {credit_equity:.2f} "
          f"vs the {R2_CREDIT_EQUITY_CORR:.1f} the R2 text quotes "
          f"(legacy spec {float(legacy_corr.loc['Credit', 'Equity']):.2f})")

    annual = build_annual_returns(navs=navs)
    print('\n--- tb:risk_factors_annual ---')
    print((1e2 * annual).round(1).to_string())
    # acceptance gate: the 2022 column must reproduce the Stage J1 annual stress column
    scenario_table = build_factor_returns(inputs=load_paper_inputs(snapshot=snapshot))
    scenario_2022 = scenario_table['annual_stress_pct'].rename(index=es.factor_label)
    gap_2022 = float((1e2 * annual['2022'] - scenario_2022).abs().max())
    print(f"\nacceptance: 2022 column vs tab:factor_returns annual stress column, "
          f"max |delta| = {gap_2022:.2e} pp")
    if gap_2022 > 1e-8:
        raise ValueError(f"2022 annual returns disagree with the Stage J1 table, "
                         f"got max gap {gap_2022!r} pp")

    if save_outputs:
        performance_figure = es.table_figure(df=format_factor_performance(table=performance),
                                            column_width=1.05,
                                            first_column_width=1.65,
                                            special_columns_colors=[(0, '#c6e2f0')])
        es.save_figure(performance_figure, 'risk_factors_perf.PNG')

        es.save_figure(plot_correlation_matrix(matrix=production_corr), 'risk_factors_corr.PNG')

        es.save_figure(plot_annual_returns(annual=annual), 'risk_factors_annual.PNG')

        write_empirical_sr(table=performance)
        write_factor_history_notes(production_corr=production_corr, legacy_corr=legacy_corr,
                                   navs=navs)
    return {'performance': performance, 'correlation': production_corr,
            'correlation_legacy': legacy_corr, 'annual': annual}


def write_factor_history_notes(production_corr: pd.DataFrame,
                               legacy_corr: pd.DataFrame,
                               navs: pd.DataFrame,
                               file_name: str = 'exhibit_factor_history_notes.tex',
                               ) -> Path:
    """caption-change notes for the three restored exhibits, including the D7 fix."""
    credit_equity = float(production_corr.loc['Credit', 'Equity'])
    lines = [
        '% ===== Caption notes for the three restored factor-history exhibits =====',
        '% Source: replication/run_factor_history_exhibits.py on cma_data snapshot 2026q2_custom.',
        f"% Window: {navs.index[0].date()} to {navs.index[-1].date()} (calendar 2005 onward).",
        '%',
        '% tb:risk_factors_perf — the NAVs are excess-return NAVs, so the Sharpe column',
        '%   carries rf = 0 by construction and the regression is against the Equity factor.',
        '%',
        '% tb:risk_factors_corr — DEFECT D7, caption change required. The exhibit is EWMA on',
        f"%   WEEKLY {CORR_FREQ} returns with span {CORR_SPAN}, the production covariance spec of the",
        '%   manifest, NOT "monthly, span 36" as the R2 caption states. Replace:',
        '%     old: exponentially weighted correlations of monthly factor returns, span 36',
        f"%     new: exponentially weighted correlations of weekly ({CORR_FREQ}) factor returns,",
        f"%          span {CORR_SPAN} weeks, with annualized volatilities on the diagonal",
        f"%   Credit-Equity reads {credit_equity:.2f} under the production spec against the 0.9 the",
        f"%   R2 text quotes ({float(legacy_corr.loc['Credit', 'Equity']):.2f} under the legacy spec), so the",
        '%   sentence quoting 0.9 moves with the caption.',
        '%',
        '% tb:risk_factors_annual — final column is 2026 year-to-date through 30 June 2026.',
    ]
    return es.write_fragment(lines=lines, file_name=file_name)


class LocalTests(str, Enum):
    ALL_EXHIBITS = 'all_exhibits'
    CORRELATION_SPEC_DIFF = 'correlation_spec_diff'


def run_local_test(local_test: LocalTests) -> None:
    """Run local tests for development and debugging purposes."""
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 300)

    if local_test == LocalTests.ALL_EXHIBITS:
        run_factor_history_exhibits()

    elif local_test == LocalTests.CORRELATION_SPEC_DIFF:
        navs = load_factor_navs()
        print(build_correlation_matrix(navs=navs).round(3).to_string())
        print(build_correlation_matrix(navs=navs, freq=LEGACY_CORR_FREQ,
                                      span=LEGACY_CORR_SPAN,
                                      annualisation=LEGACY_ANNUALISATION).round(3).to_string())

    else:
        raise NotImplementedError(f"{local_test}")


if __name__ == '__main__':
    run_local_test(local_test=LocalTests.ALL_EXHIBITS)
