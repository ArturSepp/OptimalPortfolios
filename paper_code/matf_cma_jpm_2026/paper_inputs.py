"""
``PaperInputs`` — container loaded from ``paper_inputs.xlsx``.

Mirrors the small subset of ``MatfCmaData`` and ``CmaDataReport`` calls used
by the paper figures and optimisation, so that downstream code reads a single
self-contained Excel artefact instead of a live ``rosaa`` install.

Reference currency is USD throughout — the public paper exhibits are USD only.

Usage
-----
    from pathlib import Path
    from paper_inputs import PaperInputs

    inputs = PaperInputs.load(Path("data") / "paper_inputs.xlsx")
    inputs.get_factor_cmas()         # 9-vector of factor excess CMAs
    inputs.get_universe_snapshot()   # per-asset metadata + CMAs + betas
    inputs.factors_prices            # daily factor NAV panel
    inputs.benchmark_weights         # 17 × 8 mandate weights
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


# Factor name list, used to tighten DataFrame column-orderings on load.
_FACTOR_NAMES = ['Equity', 'Rates', 'Credit', 'Carry', 'Inflation',
                 'Commodities', 'Private Equity', 'Rates Vol', 'Fx']


@dataclass(frozen=True)
class PaperInputs:
    """Self-contained paper-input artefact.

    Attributes
    ----------
    cma_metadata : pd.DataFrame
        17 rows × per-asset metadata. Columns include name, asset_class,
        base_total_cma, stress_total_cma, upside_total_cma, base_excess_cma,
        the 9 factor betas, equity_regional_addon, rates_regional_addon,
        alpha, rf_rate, r2, stat_alpha, total_vol, sys_vol, resid_vol,
        Rebalancing (ME/QE), and the regional_cma_equity / _rates labels.
    factor_excess_cma : pd.Series
        9-vector indexed by factor name → base-case excess CMA.
    equity_excess_cmas : pd.Series
        Indexed by region (Global, US, Europe, Japan, UK, Switzerland, EM, ...).
        Empty if the parser ran with `use_rosaa=False`.
    rates_total_cmas : pd.DataFrame
        Indexed by region; columns stack into a regional rates total.
        Empty if the parser ran with `use_rosaa=False`.
    factor_attribution : pd.DataFrame
        17 × 13: 9 factor contributions + equity_regional_addon +
        rates_regional_addon + alpha + rf_rate (raw, pre-merge).
    x_covar : pd.DataFrame
        9 × 9 factor covariance matrix (annualised).
    y_betas : pd.DataFrame
        17 × 9 asset factor loadings.
    y_variances : pd.DataFrame
        17 × {ewma_var, residual_var, insample_alpha, r2, cluster}.
    y_covar : pd.DataFrame
        17 × 17 asset covariance assembled as β·Σ_F·βᵀ + diag(D).
    benchmark_weights : pd.DataFrame
        17 × 8 mandate weights (Income/Low/Balanced/Growth × {without, with} Alts).
    factors_prices : pd.DataFrame
        Daily factor NAVs × 9 factors. NAV_0 = 100.
    """

    cma_metadata:       pd.DataFrame
    factor_excess_cma:  pd.Series
    equity_excess_cmas: pd.Series
    rates_total_cmas:   pd.DataFrame
    factor_attribution: pd.DataFrame
    x_covar:            pd.DataFrame
    y_betas:            pd.DataFrame
    y_variances:        pd.DataFrame
    y_covar:            pd.DataFrame
    benchmark_weights:  pd.DataFrame
    factors_prices:     pd.DataFrame

    # ─────────────────────────────────────────────────────────────────
    # Loader
    # ─────────────────────────────────────────────────────────────────
    @classmethod
    def load(cls, paper_inputs_xlsx: str | Path) -> "PaperInputs":
        """Load all sheets of ``paper_inputs.xlsx`` into a single container.

        Optional sheets (``equity_excess_cmas``, ``rates_total_cmas``) are
        loaded as empty objects if absent, so the container always exposes
        every attribute even when the parser ran with ``use_rosaa=False``.
        """
        path = Path(paper_inputs_xlsx)
        if not path.exists():
            raise FileNotFoundError(
                f"paper_inputs.xlsx not found at {path}. "
                f"Run build_paper_inputs.py first.")

        def _read(sheet, **kwargs):
            return pd.read_excel(path, sheet_name=sheet, **kwargs)

        cma_metadata = _read('cma_metadata').set_index('Unnamed: 0')
        cma_metadata.index.name = 'ticker'

        factor_excess_cma = _read('factor_excess_cma').set_index('Unnamed: 0')\
            ['factor_excess_cma'].rename('factor_excess_cma')
        factor_excess_cma.index.name = 'factor'
        # Re-order to canonical factor sequence
        factor_excess_cma = factor_excess_cma.reindex(_FACTOR_NAMES)

        # Optional sheets — return empty if absent
        try:
            eq = _read('equity_excess_cmas').set_index('Unnamed: 0')
            equity_excess_cmas = eq.iloc[:, 0].rename('equity_excess_cma')
            equity_excess_cmas.index.name = 'region'
        except (ValueError, KeyError):
            equity_excess_cmas = pd.Series(dtype=float, name='equity_excess_cma')

        try:
            rates_total_cmas = _read('rates_total_cmas').set_index('Unnamed: 0')
            rates_total_cmas.index.name = 'region'
        except (ValueError, KeyError):
            rates_total_cmas = pd.DataFrame()

        factor_attribution = _read('factor_attribution').set_index('Unnamed: 0')
        factor_attribution.index.name = 'ticker'

        x_covar = _read('x_covar').set_index('Unnamed: 0')
        x_covar.index.name = 'factor'
        x_covar = x_covar.reindex(index=_FACTOR_NAMES, columns=_FACTOR_NAMES)

        y_betas = _read('y_betas').set_index('Unnamed: 0')
        y_betas.index.name = 'ticker'
        y_betas = y_betas.reindex(columns=_FACTOR_NAMES)

        y_variances = _read('y_variances').set_index('Unnamed: 0')
        y_variances.index.name = 'ticker'

        y_covar = _read('y_covar').set_index('Unnamed: 0')
        y_covar.index.name = 'ticker'

        benchmark_weights = _read('benchmark_weights').set_index('Ticker')

        factors_prices = _read('factors_prices').set_index('date')
        factors_prices.index = pd.to_datetime(factors_prices.index)
        factors_prices = factors_prices.reindex(columns=_FACTOR_NAMES)

        return cls(
            cma_metadata=cma_metadata,
            factor_excess_cma=factor_excess_cma,
            equity_excess_cmas=equity_excess_cmas,
            rates_total_cmas=rates_total_cmas,
            factor_attribution=factor_attribution,
            x_covar=x_covar,
            y_betas=y_betas,
            y_variances=y_variances,
            y_covar=y_covar,
            benchmark_weights=benchmark_weights,
            factors_prices=factors_prices,
        )

    # ─────────────────────────────────────────────────────────────────
    # Methods that mirror MatfCmaData / CmaDataReport call signatures
    # ─────────────────────────────────────────────────────────────────
    # The `date` argument is a no-op: paper_inputs.xlsx contains a single
    # snapshot at the publication's reference date. It exists for call-site
    # compatibility with the original rosaa code.

    def get_factor_cmas(self, date: Optional[pd.Timestamp] = None) -> pd.Series:
        """9-vector of base-case factor excess CMAs."""
        return self.factor_excess_cma.copy()

    def get_equity_excess_cmas(self,
                               date: Optional[pd.Timestamp] = None
                               ) -> pd.Series:
        """Series of regional equity excess CMAs (rows = regions)."""
        if self.equity_excess_cmas.empty:
            raise ValueError(
                "equity_excess_cmas is empty — parser must run with use_rosaa=True.")
        return self.equity_excess_cmas.copy()

    def get_rates_total_cmas(self,
                             date: Optional[pd.Timestamp] = None
                             ) -> pd.DataFrame:
        """DataFrame of regional rates total CMA components (rows = regions,
        columns = stackable components: e.g. Real, Inflation Premium,
        Term Premium, Default).
        """
        if self.rates_total_cmas.empty:
            raise ValueError(
                "rates_total_cmas is empty — parser must run with use_rosaa=True.")
        return self.rates_total_cmas.copy()

    def get_factor_attribution(self) -> pd.DataFrame:
        """17 × 13 raw factor attribution table.

        Use ``compose_displayed_attribution`` to fold regional addons into
        the corresponding factor columns and produce the display version.
        """
        return self.factor_attribution.copy()

    def get_universe_snapshot(self,
                              assets: Optional[pd.Series] = None,
                              ) -> pd.DataFrame:
        """Per-asset metadata + CMAs + betas + diagnostics for the paper
        universe, with display names as the row index.

        Parameters
        ----------
        assets : pd.Series, optional
            Ticker → display name mapping. Defaults to `load_paper_assets_short()`.
        """
        if assets is None:
            from universe import load_paper_assets_short
            assets = load_paper_assets_short()

        df = self.cma_metadata.loc[list(assets.index), :].copy()

        df['Ticker'] = [t.split(' ')[0] for t in df.index]
        keep_cols = ['Ticker', 'asset_class',
                     'base_total_cma', 'stress_total_cma', 'upside_total_cma',
                     'Equity', 'Rates', 'Credit', 'Carry', 'Inflation',
                     'Commodities', 'Private Equity', 'Rates Vol', 'Fx',
                     'r2', 'stat_alpha', 'total_vol', 'sys_vol', 'resid_vol']
        rename_map = {'asset_class': 'Asset Class',
                      'base_total_cma': 'BaseCMA',
                      'stress_total_cma': 'StressCMA',
                      'upside_total_cma': 'UpsideCMA',
                      'r2': 'R2',
                      'stat_alpha': 'Alpha',
                      'total_vol': 'Vol',
                      'sys_vol': 'SystVol',
                      'resid_vol': 'ResidVol'}
        df = df[keep_cols].rename(rename_map, axis=1)
        df = df.rename(assets.to_dict(), axis=0)

        # Mask near-zero betas (display convention from the original figures)
        factor_cols = ['Equity', 'Rates', 'Credit', 'Carry', 'Inflation',
                       'Commodities', 'Private Equity', 'Rates Vol', 'Fx']
        df[factor_cols] = df[factor_cols].where(
            np.abs(df[factor_cols]) > 1e-2, other=np.nan)

        return df

    def compose_displayed_attribution(self,
                                      assets: Optional[pd.Series] = None,
                                      ) -> pd.DataFrame:
        """Display-ready factor attribution: drop Fx (λ_Fx = 0), fold
        equity_regional_addon into Equity, fold rates_regional_addon into
        Rates, append Alpha column.
        """
        if assets is None:
            from universe import load_paper_assets_short
            assets = load_paper_assets_short()

        df = self.factor_attribution.loc[list(assets.index), :].copy()

        matf_cols = ['Equity', 'Rates', 'Credit', 'Carry', 'Inflation',
                     'Commodities', 'Private Equity', 'Rates Vol', 'Fx']
        df_attrib = df[matf_cols].copy().drop('Fx', axis=1)
        df_attrib['Equity'] = df_attrib['Equity'] + df['equity_regional_addon']
        df_attrib['Rates']  = df_attrib['Rates']  + df['rates_regional_addon']
        df_attrib['Alpha']  = df['alpha']

        return df_attrib.rename(assets.to_dict(), axis=0)


__all__ = ['PaperInputs']
