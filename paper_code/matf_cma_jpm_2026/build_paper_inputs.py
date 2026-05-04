"""
LGT-side parser: produces ``paper_inputs.xlsx`` from the production CMA pipeline.

This script is run **once at LGT inside the `rosaa` environment** to extract
everything the paper figures and optimisation need into a single self-contained
Excel artefact. Replicators downstream consume only ``paper_inputs.xlsx``
through the ``PaperInputs`` container in ``paper_inputs.py`` — they do not
import ``rosaa``.

Inputs
------
- A production CMA pipeline xlsx, e.g.
  ``global_saa_universe_data_cmas_usd_2026q1_jpm.xlsx``
- A daily factor NAV CSV, e.g. ``futures_risk_factors.csv``
- A universe-snapshot xlsx with a ``universe weight`` tab giving benchmark
  mandate weights for the 8 mandates (Income/Low/Balanced/Growth × {with, w/o}
  Alts).
- A ``rosaa`` install for the regional equity / regional rates CMA tables
  that are computed inside ``CmaDataReport`` and not present in the
  production xlsx.

Output
------
- ``paper_inputs.xlsx`` with the following sheets:

    cma_metadata          17 rows × per-asset metadata + CMAs + betas + diagnostics
    factor_excess_cma     9 × 1 (factor → excess CMA)
    equity_excess_cmas    Series indexed by region (Global, US, Europe, ...)
    rates_total_cmas      regions × {Real, Inflation Premium, Term Premium, Default}
    factor_attribution    17 × 12 (raw factor contributions + addons + alpha + rf_rate)
    x_covar               9 × 9 factor covariance (annualised)
    y_betas               17 × 9 asset factor loadings
    y_variances           17 × {ewma_var, residual_var, r2, ...}
    y_covar               17 × 17 asset covariance assembled as β·Σ_F·βᵀ + diag(D)
    benchmark_weights     17 × 8 mandate weights
    factors_prices        daily NAVs × 9 factors (price index, NAV_0 = 100)

Reference currency
------------------
USD-only by design — the public paper exhibits are USD throughout.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from universe import load_paper_assets_short


# ─────────────────────────────────────────────────────────────────────
# Pure-pandas extractors (no rosaa dependency)
# ─────────────────────────────────────────────────────────────────────
def _read_production_xlsx(production_xlsx: Path,
                          tickers: list[str]
                          ) -> dict[str, pd.DataFrame]:
    """Read the per-asset and factor-level sheets from the production xlsx,
    sliced to the 17 paper tickers where applicable.
    """
    out = {}

    meta_full = pd.read_excel(production_xlsx, sheet_name='cma_metadata')\
        .set_index('Unnamed: 0')
    meta_full = meta_full[~meta_full.index.duplicated(keep='first')]
    out['cma_metadata'] = meta_full.loc[tickers].copy()

    fa_full = pd.read_excel(production_xlsx, sheet_name='factor_attribution')\
        .set_index('Unnamed: 0')
    fa_full = fa_full[~fa_full.index.duplicated(keep='first')]
    fa_cols_keep = ['Equity', 'Rates', 'Credit', 'Carry', 'Inflation',
                    'Commodities', 'Private Equity', 'Rates Vol', 'Fx',
                    'equity_regional_addon', 'rates_regional_addon',
                    'alpha', 'rf_rate']
    out['factor_attribution'] = fa_full.loc[tickers, fa_cols_keep].copy()

    out['factor_cmas'] = pd.read_excel(production_xlsx, sheet_name='factor_cmas')\
        .set_index('Unnamed: 0')

    out['x_covar'] = pd.read_excel(production_xlsx, sheet_name='x_covar')\
        .set_index('Unnamed: 0')

    yb = pd.read_excel(production_xlsx, sheet_name='y_betas')\
        .set_index('Unnamed: 0')
    yb = yb[~yb.index.duplicated(keep='first')]
    out['y_betas'] = yb.loc[tickers].copy()

    yv = pd.read_excel(production_xlsx, sheet_name='y_variances')\
        .set_index('Unnamed: 0')
    yv = yv[~yv.index.duplicated(keep='first')]
    out['y_variances'] = yv.loc[tickers].copy()

    return out


def _build_factor_excess_cma(factor_cmas: pd.DataFrame) -> pd.Series:
    """Single-column Series of base-case excess CMA per factor."""
    return factor_cmas['base'].rename('factor_excess_cma')


def _assemble_y_covar(y_betas: pd.DataFrame,
                      x_covar: pd.DataFrame,
                      y_variances: pd.DataFrame) -> pd.DataFrame:
    """Asset covariance via β · Σ_F · βᵀ + diag(D).

    Σ_F is indexed by factor name; β rows by ticker, columns by factor.
    D is the per-asset residual variance (annualised) from y_variances.
    """
    factors = list(x_covar.index)
    if list(x_covar.columns) != factors:
        raise ValueError("x_covar must be square and identically indexed.")
    if list(y_betas.columns) != factors:
        raise ValueError(
            f"y_betas columns {list(y_betas.columns)} ≠ x_covar index {factors}.")

    beta_arr = y_betas.to_numpy(dtype=float)
    sigma_F  = x_covar.to_numpy(dtype=float)
    D        = y_variances['residual_var'].to_numpy(dtype=float)

    sys_cov = beta_arr @ sigma_F @ beta_arr.T
    y_cov   = sys_cov + np.diag(D)
    y_cov   = 0.5 * (y_cov + y_cov.T)  # symmetrise

    return pd.DataFrame(y_cov, index=y_betas.index, columns=y_betas.index)


def _read_benchmark_weights(universe_xlsx: Path,
                            tickers: list[str]) -> pd.DataFrame:
    """Read the 17 × 8 benchmark weights matrix from the 'universe weight'
    tab of the universe-snapshot xlsx.
    """
    df = pd.read_excel(universe_xlsx, sheet_name='universe weight')
    asset_rows = df[df['Ticker'].notna() & df['Asset'].notna()].copy()
    asset_rows = asset_rows.set_index('Ticker')

    # Reorder to match the canonical paper-asset order
    missing = [t for t in tickers if t not in asset_rows.index]
    if missing:
        raise ValueError(f"benchmark weights missing tickers: {missing}")
    asset_rows = asset_rows.loc[tickers]

    weight_cols = ['Income\nw/o Alts', 'Low\nw/o Alts',
                   'Balanced\nw/o Alts', 'Growth\nw/o Alts',
                   'Income\nwith Alts', 'Low\nwith Alts',
                   'Balanced\nwith Alts', 'Growth\nwith Alts']
    weights = asset_rows[weight_cols].copy()
    weights.columns = [c.replace('\n', ' ') for c in weight_cols]
    return weights


def _read_factor_navs(factor_navs_csv: Path,
                      factor_names: list[str]) -> pd.DataFrame:
    """Read daily factor NAV history. Case-insensitive Fx/FX matching."""
    fnav = pd.read_csv(factor_navs_csv, index_col=0, parse_dates=True)
    fnav_lower = {c.lower(): c for c in fnav.columns}
    rename_map = {}
    missing = []
    for fn in factor_names:
        if fn in fnav.columns:
            continue
        if fn.lower() in fnav_lower:
            rename_map[fnav_lower[fn.lower()]] = fn
        else:
            missing.append(fn)
    if missing:
        raise ValueError(f"Factor NAV file missing columns: {missing}")
    if rename_map:
        fnav = fnav.rename(columns=rename_map)
    return fnav[factor_names]


# ─────────────────────────────────────────────────────────────────────
# rosaa-dependent extractors (regional equity/rates CMAs)
# ─────────────────────────────────────────────────────────────────────
def _build_regional_cmas_from_rosaa(period: str = '2026q1_jpm',
                                    reference_ccy: str = 'USD',
                                    date: Optional[pd.Timestamp] = None
                                    ) -> tuple[pd.Series, pd.DataFrame]:
    """Pull the regional equity-excess and rates-total CMA tables from
    ``rosaa.CmaDataReport``.

    These are computed inside ``CmaDataReport`` and not present in the
    production xlsx, so the parser must run inside the rosaa environment
    to extract them.

    Returns
    -------
    equity_excess_cmas : pd.Series
        Indexed by region (Global, US, Europe, Japan, UK, Switzerland, EM, ...).
    rates_total_cmas : pd.DataFrame
        Indexed by region; columns stack into the total (e.g. Real,
        Inflation Premium, Term Premium, Default).
    """
    try:
        from rosaa import CmaDataReport, local_path as lp
    except ImportError as exc:
        raise ImportError(
            "build_paper_inputs.py requires rosaa to extract regional CMAs. "
            "Run inside the rosaa environment."
        ) from exc

    local_path = lp.get_resource_path()
    cma_data_report = CmaDataReport.load(local_path, period=period)
    if date is None:
        date = pd.Timestamp('31Mar2026')

    eq = cma_data_report.get_equity_excess_cmas(date=date)
    eq = eq.rename({'Equity': 'Global'}) if 'Equity' in eq.index else eq

    rt = cma_data_report.get_rates_total_cmas(date=date)
    if 'Rates' in rt.index:
        rt = rt.rename({'Rates': 'Global'}, axis=0)
    # Drop any region the paper does not display
    for drop_region in ['EM', 'NewZealand']:
        if drop_region in rt.index:
            rt = rt.drop(index=drop_region)

    return eq, rt


# ─────────────────────────────────────────────────────────────────────
# Top-level entry point
# ─────────────────────────────────────────────────────────────────────
def build_paper_inputs(production_xlsx: Path,
                       factor_navs_csv: Path,
                       universe_xlsx: Path,
                       output_xlsx: Path,
                       period: str = '2026q1_jpm',
                       date: Optional[pd.Timestamp] = None,
                       use_rosaa: bool = True,
                       verbose: bool = True) -> None:
    """Build ``paper_inputs.xlsx`` from production data sources.

    Parameters
    ----------
    production_xlsx : Path
        Path to the production CMA pipeline xlsx (USD reference).
    factor_navs_csv : Path
        Daily factor NAV CSV.
    universe_xlsx : Path
        Path to the universe-snapshot xlsx with the 'universe weight' sheet.
    output_xlsx : Path
        Where to write paper_inputs.xlsx.
    period : str
        Production CMA period tag for ``CmaDataReport.load``. Only used if
        ``use_rosaa = True``.
    date : pd.Timestamp, optional
        Snapshot date for regional CMA extraction. Defaults to 2026-03-31.
    use_rosaa : bool, default True
        If True, extract regional equity and rates CMA tables from
        ``rosaa.CmaDataReport``. If False, skip those two sheets — the
        equity_factor_cmas and rates_factor_cmas figures will not be
        reproducible from the resulting paper_inputs.xlsx.
    verbose : bool
        Print progress messages.
    """
    assets = load_paper_assets_short()
    tickers = list(assets.index)
    if verbose:
        print(f"[parser] paper universe: {len(tickers)} assets")

    if verbose:
        print(f"[parser] reading production xlsx: {production_xlsx}")
    sheets = _read_production_xlsx(production_xlsx, tickers)

    if verbose:
        print(f"[parser] assembling y_covar = β·Σ_F·βᵀ + diag(D)")
    y_covar = _assemble_y_covar(
        sheets['y_betas'], sheets['x_covar'], sheets['y_variances'])

    if verbose:
        print(f"[parser] reading benchmark mandate weights from: {universe_xlsx}")
    benchmark_weights = _read_benchmark_weights(universe_xlsx, tickers)
    col_sum_max_dev = float(np.abs(benchmark_weights.sum(axis=0) - 1.0).max())
    if col_sum_max_dev > 1e-3:
        raise ValueError(
            f"benchmark mandate columns do not sum to 1.0 (max dev = {col_sum_max_dev:.4f})")
    if verbose:
        print(f"[parser]   8 mandates, max deviation from sum=1: {col_sum_max_dev:.6f}")

    factor_names = list(sheets['x_covar'].index)
    if verbose:
        print(f"[parser] reading factor NAVs ({len(factor_names)} factors): {factor_navs_csv}")
    factors_prices = _read_factor_navs(factor_navs_csv, factor_names)

    factor_excess_cma = _build_factor_excess_cma(sheets['factor_cmas'])

    if use_rosaa:
        if verbose:
            print(f"[parser] pulling regional CMAs from rosaa (period={period})")
        equity_excess_cmas, rates_total_cmas = _build_regional_cmas_from_rosaa(
            period=period, date=date)
    else:
        if verbose:
            print(f"[parser] skipping regional CMA extraction (use_rosaa=False)")
        equity_excess_cmas = pd.Series(dtype=float, name='equity_excess_cma')
        rates_total_cmas = pd.DataFrame()

    # Write all sheets
    if verbose:
        print(f"[parser] writing: {output_xlsx}")
    output_xlsx = Path(output_xlsx)
    output_xlsx.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(output_xlsx, engine='openpyxl') as wr:
        sheets['cma_metadata'].to_excel(wr, sheet_name='cma_metadata')
        factor_excess_cma.to_excel(wr, sheet_name='factor_excess_cma')
        if not equity_excess_cmas.empty:
            equity_excess_cmas.to_excel(wr, sheet_name='equity_excess_cmas')
        if not rates_total_cmas.empty:
            rates_total_cmas.to_excel(wr, sheet_name='rates_total_cmas')
        sheets['factor_attribution'].to_excel(wr, sheet_name='factor_attribution')
        sheets['x_covar'].to_excel(wr, sheet_name='x_covar')
        sheets['y_betas'].to_excel(wr, sheet_name='y_betas')
        sheets['y_variances'].to_excel(wr, sheet_name='y_variances')
        y_covar.to_excel(wr, sheet_name='y_covar')
        benchmark_weights.to_excel(wr, sheet_name='benchmark_weights')
        factors_prices.to_excel(wr, sheet_name='factors_prices')

    if verbose:
        print(f"[parser] done.")


def _cli() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--production-xlsx", required=True, type=Path)
    p.add_argument("--factor-navs-csv", required=True, type=Path)
    p.add_argument("--universe-xlsx", required=True, type=Path)
    p.add_argument("--output-xlsx", default=Path("data") / "paper_inputs.xlsx",
                   type=Path)
    p.add_argument("--period", default='2026q1_jpm')
    p.add_argument("--date", default='2026-03-31')
    p.add_argument("--no-rosaa", action='store_true',
                   help="Skip regional CMA extraction (paper_inputs.xlsx will "
                        "lack the equity_excess_cmas and rates_total_cmas sheets).")
    args = p.parse_args()

    build_paper_inputs(
        production_xlsx=args.production_xlsx,
        factor_navs_csv=args.factor_navs_csv,
        universe_xlsx=args.universe_xlsx,
        output_xlsx=args.output_xlsx,
        period=args.period,
        date=pd.Timestamp(args.date),
        use_rosaa=not args.no_rosaa,
    )


if __name__ == "__main__":
    _cli()
