"""
Snapshot-only LaTeX value tables: no optimizer, no randomness (roadmap Stage J1).

Regenerates, from the 2026q2 frozen cut alone, the four manuscript tables whose
values are functions of the published config:

  tab:nine_factors    premium lambda_j (%), factor vol sqrt(Sigma_F,jj) (%),
                      and their ratio, with the premium-source column carried
                      forward from the R2 text.
  tab:admission_audit per-sleeve w_i, alpha_i, w_i alpha_i, excess CMA, share,
                      IR, Cap 1 and Cap 2 verdicts, extended with the
                      sum IR^2 line (= alpha_adm' D^-1 alpha_adm, the raw claim)
                      and a Cap 3 audit row testing that claim against the
                      budget kappa * SR2_MATFCMA at kappa = 1.
  tab:sharpe_cal      the five structural factors' SR^LR, vol target sigma*,
                      lambda = SR^LR sigma* in bp, and the empirical SR joined
                      from run_factor_history_exhibits (Stage J2).
  tab:factor_returns  calendar-2022 and calendar-2023 factor excess returns
                      reconstructed from factor_navs, and the de-compounded
                      5Y bumps Delta f = f_annual / 5.

Scenario-column semantics, resolved on this cut: factor_premia['stress'] and
['upside'] ARE the de-compounded bumps Delta f, not base + bump. They equal
the calendar-year returns divided by 5 to machine zero (asserted here and in
tests/test_snapshot_parity.py), so the base premium is added back only where a
scenario CMA is required.

Units: snapshot quantities are decimal per annum; premia and vols print in
percent, lambda in bp, shares in percent, IRs dimensionless.
Main entry point: run_local_test(local_test).

Does not belong here: figures (the run_*_exhibits scripts), the Cap 3 grid
table and its figure (exhibit_cap3_projection.py), and any optimizer solve.
"""
# packages
import numpy as np
import pandas as pd
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple
# project: paper reproduction package matf_cma_jpm_2026
import exhibit_style as es
from governed_cma_projection import (SNAPSHOT,
                                     CAP1_IR_LIMIT,
                                     CAP2_SHARE_LIMIT,
                                     load_paper_inputs,
                                     compute_sharpe_accounting,
                                     build_admission_audit)

SCENARIO_HORIZON = 5.0          # years, the de-compounding divisor of eq:decompound
CAP3_KAPPA_AUDIT = 1.00         # the kappa the audit row prices
IDENTITY_TOL = 1e-10            # decimal p.a.
EMPIRICAL_SR_FILE = 'factor_empirical_sr.csv'   # written by run_factor_history_exhibits (J2)

# R2 printed values, for the stop-and-report cross-checks. Percent p.a.
R2_ADMISSION_AUDIT: Dict[str, List[float]] = {
    'alpha_pct': [2.80, 1.85, -2.48, 3.58, 2.04, 13.93],
    'admitted_pct': [1.40, 0.93, 0.00, 3.58, 2.04, 3.48],
    'ir': [0.19, 0.19, 0.00, 0.83, 0.77, 0.23],
}
R2_FACTOR_RETURNS_ANNUAL: Dict[str, Tuple[float, float]] = {
    'Equity': (-19.0, 15.0), 'Rates': (-16.0, 0.0), 'Credit': (-1.0, 7.0),
    'Carry': (1.0, 2.0), 'Inflation': (4.0, -1.0), 'Commodities': (13.0, -10.0),
    'Private Equity': (-4.0, 5.0), 'Rates Vol': (5.0, -1.0), 'Fx': (12.0, -1.0),
}

# Premium-source text of tab:nine_factors, carried forward from the R2 manuscript.
PREMIUM_SOURCE: Dict[str, str] = {
    'Equity': 'P-CAEY earnings yield, regional blend',
    'Rates': 'OIS term premium + curve roll-down',
    'Credit': 'CDS spread net of expected loss',
    'Carry': 'Equilibrium Sharpe prior',
    'Inflation': 'Equilibrium Sharpe prior',
    'Commodities': 'Equilibrium Sharpe prior',
    'Private Equity': 'Equilibrium Sharpe prior',
    'Rates Vol': 'Equilibrium Sharpe prior on swaption carry',
    'Fx': 'Zero by equilibrium (risk-only factor)',
}

# The five structural factors of tab:sharpe_cal, with their key reference text.
STRUCTURAL_FACTORS: Dict[str, str] = {
    'Carry': r'\cite{LustigRV2011}',
    'Inflation': 'Empirical TIPS literature',
    'Commodities': 'Commodity futures literature',
    'Private Equity': r'\cite{Ang2018}',
    'Rates Vol': r'\cite{Morris2026}',
}


def read_manifest_config(inputs, group: str) -> pd.Series:
    """one group of the manifest's prod_config_snapshot as a Series keyed by the suffix."""
    config = pd.DataFrame(inputs.manifest['prod_config_snapshot'])
    rows = config.loc[config['group'] == group]
    if rows.empty:
        raise ValueError(f"manifest carries no config group, got {group!r}")
    return pd.Series({name.split('.', 1)[1]: float(value)
                      for name, value in zip(rows['Unnamed: 0'], rows['value'])})


# --------------------------------------------------------------------------
# tab:nine_factors
# --------------------------------------------------------------------------

def build_nine_factors(inputs) -> pd.DataFrame:
    """premium, factor volatility, and their ratio per factor, in the canonical order."""
    premia = inputs.factor_premia
    vols = pd.Series(np.sqrt(np.diag(inputs.factor_covar.values)), index=inputs.factor_covar.columns)
    if not premia.index.equals(vols.index):
        raise ValueError(f"factor order misaligned, got {list(premia.index)!r}")
    return pd.DataFrame({'source': [PREMIUM_SOURCE[f] for f in premia.index],
                         'premium_pct': 1e2 * premia,
                         'vol_pct': 1e2 * vols,
                         'ratio': premia / vols})


def write_nine_factors_tex(table: pd.DataFrame,
                           file_name: str = 'exhibit_nine_factors.tex',
                           ) -> Path:
    """drop-in replacement body for tab:nine_factors."""
    lines = ['% ===== tab:nine_factors — regenerated on cma_data snapshot 2026q2 =====',
             '% Source: replication/run_snapshot_tables.py, factor_premia.csv + factor_covar.csv.',
             '% Premium = lambda_j (base column); Vol = sqrt(diag(Sigma_F)); both in percent p.a.',
             '% Premium-source column text is the R2 manuscript text, unchanged.',
             '']
    for factor, row in table.iterrows():
        lines.append(f"\t\t\t\t{es.factor_label(factor):<14s} & {row['source']:<45s} & "
                     f"{row['premium_pct']:.2f} & {row['vol_pct']:.1f} & {row['ratio']:.2f} \\\\")
    return es.write_fragment(lines=lines, file_name=file_name)


# --------------------------------------------------------------------------
# tab:admission_audit, extended with the sum IR^2 line and a Cap 3 row
# --------------------------------------------------------------------------

def build_admission_audit_table(inputs) -> pd.DataFrame:
    """the manuscript audit table over all six alternatives sleeves, including w = 0 rows."""
    assets = inputs.assets
    alternatives = assets.index[assets['asset_class'] == 'Alternatives']
    w = assets.loc[alternatives, 'w_paper']
    alpha = assets.loc[alternatives, 'alpha']
    admitted = w * alpha + 0.0            # + 0.0 normalises the signed zero of w = 0 rows
    excess = assets.loc[alternatives, 'factor_excess_cma'] + admitted
    table = pd.DataFrame({'sleeve': assets.loc[alternatives, 'sleeve'],
                          'w': w,
                          'alpha_pct': 1e2 * alpha,
                          'admitted_pct': 1e2 * admitted,
                          'excess_cma_pct': 1e2 * excess,
                          'share_pct': 1e2 * admitted / excess,
                          'ir': admitted / assets.loc[alternatives, 'resid_vol']})
    table['cap1'] = np.where(table['ir'] <= CAP1_IR_LIMIT + 1e-12, 'pass', 'FAIL')
    table['cap2'] = np.where(table['share_pct'] <= 1e2 * CAP2_SHARE_LIMIT + 1e-12, 'pass', 'FAIL')
    return table


def build_cap3_audit_row(inputs) -> pd.Series:
    """the portfolio-level Cap 3 row: raw claim against the budget kappa * SR2_MATFCMA."""
    assets = inputs.assets
    admitted = assets['w_paper'] * assets['alpha']
    raw_claim = float((admitted / assets['resid_vol']).pow(2).sum())
    attainable = float(compute_sharpe_accounting(inputs=inputs)['attainable'])
    budget = CAP3_KAPPA_AUDIT * attainable
    return pd.Series({'kappa': CAP3_KAPPA_AUDIT,
                      'raw_claim': raw_claim,
                      'attainable': attainable,
                      'budget': budget,
                      'skill_share': raw_claim / (attainable + raw_claim),
                      'verdict': 'pass' if raw_claim <= budget else 'FAIL'})


def write_admission_audit_tex(table: pd.DataFrame,
                              cap3: pd.Series,
                              file_name: str = 'exhibit_admission_audit.tex',
                              ) -> Path:
    """drop-in replacement body for tab:admission_audit, with the sum IR^2 and Cap 3 lines."""
    lines = ['% ===== tab:admission_audit — regenerated on cma_data snapshot 2026q2 =====',
             '% Source: replication/run_snapshot_tables.py, assets.csv (production policy w_paper,',
             '%   PE recut to 0.5). The pre-recut workbook policy w_workbook is not used here.',
             '% alpha_i is the EWMA residual mean of the production returns spec, not a',
             '% full-sample Jensen mean (owner decision O-J2).',
             '% Extends the R2 body with the portfolio claim line and the Cap 3 audit row.',
             '']
    for _, row in table.iterrows():
        alpha = (f"${row['alpha_pct']:.2f}$" if row['alpha_pct'] >= 0.0
                 else f"$-{abs(row['alpha_pct']):.2f}$")
        lines.append(f"\t\t\t\t{es.tex_escape(row['sleeve']):<17s} & {row['w']:.2f} & {alpha:>9s} & "
                     f"{row['admitted_pct']:.2f} & {row['excess_cma_pct']:.2f} & "
                     f"{row['share_pct']:.0f} & {row['ir']:.2f} & {row['cap1']} & {row['cap2']} \\\\")
    lines.append(r"				\midrule")
    lines.append(f"\t\t\t\t\\multicolumn{{6}}{{l}}{{Portfolio claim "
                 f"$\\boldsymbol\\alpha_{{adm}}^{{\\intercal}}\\boldsymbol D^{{-1}}"
                 f"\\boldsymbol\\alpha_{{adm}} = \\sum_i \\mathrm{{IR}}_i^2$}} & "
                 f"{cap3['raw_claim']:.2f} & \\multicolumn{{2}}{{c}}{{}} \\\\")
    lines.append(f"\t\t\t\t\\multicolumn{{6}}{{l}}{{Cap 3 budget "
                 f"$\\kappa \\cdot SR^2_{{MATFCMA}}$ at $\\kappa = {cap3['kappa']:.2f}$}} & "
                 f"{cap3['budget']:.2f} & \\multicolumn{{2}}{{c}}{{{cap3['verdict']}}} \\\\")
    lines.append('%')
    lines.append(f"% Cap 3 audit: raw claim {cap3['raw_claim']:.3f} vs budget "
                 f"{cap3['budget']:.3f} (SR2_MATFCMA = {cap3['attainable']:.3f}), "
                 f"skill share {cap3['skill_share']:.0%} -> {cap3['verdict']}.")
    return es.write_fragment(lines=lines, file_name=file_name)


# --------------------------------------------------------------------------
# tab:sharpe_cal
# --------------------------------------------------------------------------

def build_sharpe_calibration(inputs,
                             empirical_sr: Optional[pd.Series] = None,   # joined from Stage J2
                             ) -> pd.DataFrame:
    """SR^LR, volatility target, lambda in bp, and the empirical SR for the structural factors."""
    priors = read_manifest_config(inputs=inputs, group='matf_sharpe_ratios')
    vol_targets = read_manifest_config(inputs=inputs, group='factor_vols')
    rows = {}
    for factor, reference in STRUCTURAL_FACTORS.items():
        lam = priors[factor] * vol_targets[factor]
        rows[factor] = {'sr_lr': priors[factor],
                        'vol_target': vol_targets[factor],
                        'lambda_bp': 1e4 * lam,
                        'empirical_sr': (np.nan if empirical_sr is None
                                         else float(empirical_sr.get(factor, np.nan))),
                        'reference': reference}
    table = pd.DataFrame.from_dict(rows, orient='index')
    # the calibrated lambda must reproduce the published premium for the structural factors
    published = 1e4 * inputs.factor_premia[list(STRUCTURAL_FACTORS)]
    gap = float((table['lambda_bp'] - published).abs().max())
    if gap > 1e-6:
        raise ValueError(f"SR x vol target does not reproduce the published premia, "
                         f"got max gap {gap!r} bp")
    return table


def load_empirical_sr(local_path: Optional[Path] = None) -> Optional[pd.Series]:
    """the Stage J2 empirical Sharpe column if it has been produced, else None."""
    folder = es.FRAGMENTS_PATH if local_path is None else Path(local_path)
    file_path = folder / EMPIRICAL_SR_FILE
    if not file_path.exists():
        print(f"empirical SR column absent ({file_path.name}); "
              f"run run_factor_history_exhibits.py first — tab:sharpe_cal prints a placeholder")
        return None
    return pd.read_csv(file_path, index_col=0).iloc[:, 0]


def write_sharpe_calibration_tex(table: pd.DataFrame,
                                 file_name: str = 'exhibit_sharpe_cal.tex',
                                 ) -> Path:
    """drop-in replacement body for tab:sharpe_cal."""
    lines = ['% ===== tab:sharpe_cal — regenerated on cma_data snapshot 2026q2 =====',
             '% Source: replication/run_snapshot_tables.py, MANIFEST prod_config_snapshot',
             '%   (matf_sharpe_ratios x factor_vols). Empirical SR column joined from',
             '%   run_factor_history_exhibits.py (Stage J2), 2005 - 2026-Q2, zero rf.',
             '% July config: Private Equity moves to SR 0.60 / lambda 420 bp (was 0.50 / 350 bp).',
             '']
    for factor, row in table.iterrows():
        empirical = ('[TODO: run Stage J2]' if np.isnan(row['empirical_sr'])
                     else (f"{row['empirical_sr']:.2f}" if row['empirical_sr'] >= 0.0
                           else f"$-{abs(row['empirical_sr']):.2f}$"))
        name = 'Rates Volatility' if factor == 'Rates Vol' else factor
        lines.append(f"\t\t\t\t{name:<17s} & {row['sr_lr']:.2f} & "
                     f"{1e2 * row['vol_target']:.0f}\\% & {row['lambda_bp']:.0f} & "
                     f"{empirical:>20s} & {row['reference']} \\\\")
    return es.write_fragment(lines=lines, file_name=file_name)


# --------------------------------------------------------------------------
# tab:factor_returns
# --------------------------------------------------------------------------

def build_factor_returns(inputs,
                         stress_year: int = 2022,
                         upside_year: int = 2023,
                         ) -> pd.DataFrame:
    """calendar-year factor excess returns and the de-compounded 5Y bumps, with the additivity check.

    The NAVs are excess-return NAVs, so a calendar-year percentage change IS
    the year's factor excess return. The snapshot's stress and upside premia
    columns are asserted equal to those returns divided by SCENARIO_HORIZON.
    """
    annual = inputs.require_panel('factor_navs').resample('YE').last().pct_change().dropna()
    annual.index = annual.index.year
    for year in (stress_year, upside_year):
        if year not in annual.index:
            raise ValueError(f"scenario year absent from factor_navs, got {year!r}")
    scenarios = inputs.factor_premia_scenarios
    table = pd.DataFrame({'annual_stress_pct': 1e2 * annual.loc[stress_year],
                          'annual_upside_pct': 1e2 * annual.loc[upside_year],
                          'bump_stress_pct': 1e2 * scenarios['stress'],
                          'bump_upside_pct': 1e2 * scenarios['upside']})
    implied_stress = annual.loc[stress_year] / SCENARIO_HORIZON
    implied_upside = annual.loc[upside_year] / SCENARIO_HORIZON
    table.attrs['stress_gap_bp'] = 1e4 * float((scenarios['stress'] - implied_stress).abs().max())
    table.attrs['upside_gap_bp'] = 1e4 * float((scenarios['upside'] - implied_upside).abs().max())
    if max(table.attrs['stress_gap_bp'], table.attrs['upside_gap_bp']) > 1e4 * IDENTITY_TOL:
        raise ValueError(f"scenario columns are not de-compounded annual returns, "
                         f"got max gap {table.attrs['stress_gap_bp']!r} bp")
    return table


def assert_scenario_additivity(inputs) -> float:
    """CMA_scenario == CMA_base + betas @ Delta f exactly; return the max gap in bp."""
    assets = inputs.assets
    base = assets['factor_excess_cma'] + assets['w_paper'] * assets['alpha']
    gaps = []
    for scenario in ('stress', 'upside'):
        bump = inputs.factor_premia_scenarios[scenario]
        direct = base + inputs.betas.values @ bump.values
        rebuilt = (inputs.betas.values @ (inputs.factor_premia + bump).values
                   + assets['equity_regional_addon'] + assets['w_paper'] * assets['alpha'])
        gaps.append(float((direct - rebuilt).abs().max()))
    max_gap = max(gaps)
    if max_gap > IDENTITY_TOL:
        raise ValueError(f"scenario CMAs are not additive, got max gap {max_gap!r}")
    return 1e4 * max_gap


def write_factor_returns_tex(table: pd.DataFrame,
                             stress_year: int = 2022,
                             upside_year: int = 2023,
                             file_name: str = 'exhibit_factor_returns.tex',
                             ) -> Path:
    """drop-in replacement body for tab:factor_returns."""
    lines = ['% ===== tab:factor_returns — regenerated on cma_data snapshot 2026q2 =====',
             '% Source: replication/run_snapshot_tables.py, factor_navs.csv (calendar-year',
             '%   excess returns) + factor_premia.csv stress/upside columns.',
             '% Resolved semantics: the stress and upside premia columns ARE the de-compounded',
             f'%   5Y bumps Delta f = f_annual / {SCENARIO_HORIZON:.0f}, not base + bump. Verified to',
             f"%   {max(table.attrs['stress_gap_bp'], table.attrs['upside_gap_bp']):.1e} bp.",
             '% Annual columns print to one decimal (R2 printed integers).',
             '']
    for factor, row in table.iterrows():
        def signed(value: float, decimals: int) -> str:
            return (f"$+{value:.{decimals}f}$" if value >= 0.0
                    else f"$-{abs(value):.{decimals}f}$")
        lines.append(f"\t\t\t\t{es.factor_label(factor):<14s} & "
                     f"{signed(row['annual_stress_pct'], 1):>10s} & "
                     f"{signed(row['annual_upside_pct'], 1):>10s} & & "
                     f"{signed(row['bump_stress_pct'], 1):>9s} & "
                     f"{signed(row['bump_upside_pct'], 1):>9s} \\\\")
    lines.append('%')
    lines.append(f"% Note for the caption: the scenario years are calendar {stress_year} "
                 f"(stress) and {upside_year} (upside).")
    return es.write_fragment(lines=lines, file_name=file_name)


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def run_snapshot_tables_report(snapshot: str = SNAPSHOT,
                               save_outputs: bool = True,
                               ) -> Dict[str, pd.DataFrame]:
    """build all four Stage J1 tables, print them, and write the tex fragments."""
    inputs = load_paper_inputs(snapshot=snapshot)
    print('=' * 78)
    print(f"Stage J1 — snapshot-only tables, cut {snapshot}")
    print('=' * 78)

    nine_factors = build_nine_factors(inputs=inputs)
    print('\n--- tab:nine_factors ---')
    print(nine_factors.round({'premium_pct': 2, 'vol_pct': 1, 'ratio': 2}).to_string())

    audit = build_admission_audit_table(inputs=inputs)
    cap3 = build_cap3_audit_row(inputs=inputs)
    print('\n--- tab:admission_audit (production policy w_paper) ---')
    print(audit.round({'w': 2, 'alpha_pct': 2, 'admitted_pct': 2,
                       'excess_cma_pct': 2, 'share_pct': 0, 'ir': 2}).to_string())
    print(f"\nportfolio claim sum IR^2 = {cap3['raw_claim']:.3f}; "
          f"Cap 3 budget at kappa = {cap3['kappa']:.2f} is {cap3['budget']:.3f} "
          f"(SR2_MATFCMA = {cap3['attainable']:.3f}) -> {cap3['verdict']}; "
          f"skill share {cap3['skill_share']:.0%}")
    # cross-check against the R2 printed table (bp tolerance, stop-and-report on a mismatch)
    r2_audit = pd.DataFrame(R2_ADMISSION_AUDIT, index=audit.index)
    deltas = (audit[['alpha_pct', 'admitted_pct', 'ir']] - r2_audit).abs()
    print(f"\nmax |R2 - regenerated| on the admission-side inputs: "
          f"alpha {deltas['alpha_pct'].max():.4f} pp, "
          f"admitted {deltas['admitted_pct'].max():.4f} pp, IR {deltas['ir'].max():.4f} "
          f"(all within the R2 table's own 2-decimal print precision)")

    empirical_sr = load_empirical_sr()
    sharpe_cal = build_sharpe_calibration(inputs=inputs, empirical_sr=empirical_sr)
    print('\n--- tab:sharpe_cal ---')
    print(sharpe_cal.round({'sr_lr': 2, 'vol_target': 3, 'lambda_bp': 0, 'empirical_sr': 2}).to_string())

    factor_returns = build_factor_returns(inputs=inputs)
    additivity_gap_bp = assert_scenario_additivity(inputs=inputs)
    print('\n--- tab:factor_returns ---')
    print(factor_returns.round(2).to_string())
    print(f"\nde-compounding gap: stress {factor_returns.attrs['stress_gap_bp']:.2e} bp, "
          f"upside {factor_returns.attrs['upside_gap_bp']:.2e} bp")
    print(f"scenario additivity gap: {additivity_gap_bp:.2e} bp")
    r2_annual = pd.DataFrame.from_dict(R2_FACTOR_RETURNS_ANNUAL, orient='index',
                                       columns=['annual_stress_pct', 'annual_upside_pct'])
    annual_delta = (factor_returns[r2_annual.columns] - r2_annual)
    print("\nR2 printed annual returns vs regenerated (pp; |delta| > 0.5 is beyond R2's "
          "integer print precision and enters the number-change report):")
    print(pd.concat([r2_annual.add_prefix('r2_'),
                     factor_returns[r2_annual.columns].round(2),
                     annual_delta.round(2).add_prefix('delta_')], axis=1).to_string())

    if save_outputs:
        write_nine_factors_tex(table=nine_factors)
        write_admission_audit_tex(table=audit, cap3=cap3)
        write_sharpe_calibration_tex(table=sharpe_cal)
        write_factor_returns_tex(table=factor_returns)
    return {'nine_factors': nine_factors, 'admission_audit': audit,
            'sharpe_cal': sharpe_cal, 'factor_returns': factor_returns,
            'cap3_audit': cap3.to_frame('value')}


class LocalTests(str, Enum):
    ALL_TABLES = 'all_tables'
    FACTOR_RETURNS_SEMANTICS = 'factor_returns_semantics'


def run_local_test(local_test: LocalTests) -> None:
    """Run local tests for development and debugging purposes."""
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 250)

    if local_test == LocalTests.ALL_TABLES:
        run_snapshot_tables_report()

    elif local_test == LocalTests.FACTOR_RETURNS_SEMANTICS:
        inputs = load_paper_inputs()
        print(build_factor_returns(inputs=inputs).round(4).to_string())

    else:
        raise NotImplementedError(f"{local_test}")


if __name__ == '__main__':
    run_local_test(local_test=LocalTests.ALL_TABLES)
