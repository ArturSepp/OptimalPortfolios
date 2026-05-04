"""
Real-data bootstrap driver for the MATF-CMA frontier comparison exhibit.

Reproduces Exhibits 1 (frontier fan) and 2 (alpha-beta consistency violation)
of Sepp, Hansen, Kastenholz (2026), "Capital Market Assumptions Using
Multi-Asset Tradable Factors: The MATF-CMA Framework", Journal of Portfolio
Management, forthcoming.

Pipeline summary
----------------
  1. Load production CMA pipeline outputs (xlsx) and factor NAVs (csv).

  2. Resample three paired panels with a stationary block bootstrap
     (mean L = 12 months, minimum L_min = 3 months for QE-safe sampling):
       - asset returns Y    (T x 15, mixed ME/QE frequency)
       - factor returns X   (T x 9,  monthly log returns from production NAVs)
       - residuals  ε        (T x 15, native-frequency residuals)

  3. Two estimators per draw, both producing long-only efficient frontiers:
       - Raw asset-level: NaN-aware sample mean and pairwise sample covariance
                          on Y; mathematically equivalent to a per-asset
                          Grinold-Kroner construction whose μ is the sample mean.
       - MATF factor-structured: equal-weighted Σ_F on X, equal-weighted D
                          on ε, and λ drawn from a structural Sharpe-ratio
                          prior SR ~ N(m_SR, Σ_SR), giving
                              μ = r_f + β λ
                              Σ = β Σ_F βᵀ + diag(D)

  4. Per-draw alpha-beta consistency residual
         Δ_b = (I − β(βᵀβ)⁻¹βᵀ)(μ_GK_b − r_f)
     where μ_GK_b is the per-draw asset sample mean (Grinold-Kroner proxy).
     For MATF, Δ ≡ 0 by construction.

References
----------
Politis, D.N., Romano, J.P. (1994). The Stationary Bootstrap. JASA 89(428):1303-1313.
Chopra, V.K., Ziemba, W.T. (1993). The Effect of Errors in Means, Variances,
    and Covariances on Optimal Portfolio Choice. JPM 19(2):6-11.
DeMiguel, V., Garlappi, L., Uppal, R. (2009). Optimal Versus Naive Diversification.
    RFS 22(5):1915-1953.
Jagannathan, R., Ma, T. (2003). Risk Reduction in Large Portfolios.
    JoF 58(4):1651-1683.
Michaud, R. (1989). The Markowitz Optimization Enigma. FAJ 45(1):31-42.
Michaud, R. (1998). Efficient Asset Management. HBS Press.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from bootstrap_frontier_analytics import (
    stationary_block_indices,
    solve_long_only_frontier,
    min_variance_vol,
)


# ═══════════════════════════════════════════════════════════════════
# 1. Universe and benchmark mandates — paper §5 illustrative universe
#    Universe matches `universe weight` tab of universe_snapshot.xlsx
#    (snapshot universe_20260503_1152), 17 assets, USD investor, 2026-Q1.
# ═══════════════════════════════════════════════════════════════════
PAPER_TICKERS: List[Tuple[str, str]] = [
    # Fixed income (7)
    ("LGTRTRUH Index",         "Global Government"),
    ("LGCPTRUH Index",         "Global IG Bonds"),
    ("H23059US Index",         "Global HY Bonds"),
    ("EMUSTRUU Index",         "EM HC Bonds"),
    ("LGT_OTHERFI Index",      "Other Fixed Income"),
    ("LF94TRUH Index",         "Global Inflation-Linked"),
    ("H24641US Index",         "Global Convertibles"),
    # Equities (5)
    ("NDDUUS Index",           "MSCI US"),
    ("MSDEE15N Index",         "MSCI Europe"),
    ("NDDLJN Index",           "MSCI Japan"),
    ("M1APJ Index",            "MSCI Asia Ex-Japan"),
    ("M1EFZ Index",            "MSCI EM ex-Asia"),
    # Alternatives (5)
    ("MP503001 Index",         "Private Equity"),
    ("MP503008 Index",         "Private Credit"),
    ("LGT_ILS Index",          "Insurance-Linked"),
    ("HFRXGL Index",           "Hedge Funds"),
    ("LGT_REAL_ASSETS Index",  "Real Assets"),
]

# Eight policy benchmark mandates: 4 risk levels × {without alts, with alts}.
# Order: Income, Low, Balanced, Growth, then with-alts versions.
# Source: 'universe weight' tab of universe_snapshot.xlsx (May 2026 update).
# Rows align with PAPER_TICKERS above.
BENCHMARK_WEIGHTS = np.array([
    # Income w/o, Low w/o, Bal w/o, Growth w/o,  Income w/, Low w/, Bal w/, Growth w/
    [0.4770, 0.3339, 0.1908, 0.0000,  0.4293, 0.2671, 0.1336, 0.0000],  # Global Government
    [0.2056, 0.1439, 0.0822, 0.0000,  0.1850, 0.1151, 0.0576, 0.0000],  # Global IG Bonds
    [0.0266, 0.0186, 0.0106, 0.0000,  0.0239, 0.0149, 0.0074, 0.0000],  # Global HY Bonds
    [0.1848, 0.1294, 0.0739, 0.0000,  0.1663, 0.1035, 0.0517, 0.0000],  # EM HC Bonds
    [0.0360, 0.0252, 0.0144, 0.0000,  0.0324, 0.0202, 0.0101, 0.0000],  # Other Fixed Income
    [0.0500, 0.0350, 0.0200, 0.0000,  0.0450, 0.0280, 0.0140, 0.0000],  # Global Inflation-Linked
    [0.0200, 0.0140, 0.0080, 0.0000,  0.0180, 0.0112, 0.0056, 0.0000],  # Global Convertibles
    [0.0000, 0.2048, 0.4096, 0.6827,  0.0000, 0.1638, 0.2867, 0.4096],  # MSCI US
    [0.0000, 0.0420, 0.0841, 0.1401,  0.0000, 0.0336, 0.0588, 0.0841],  # MSCI Europe
    [0.0000, 0.0146, 0.0292, 0.0486,  0.0000, 0.0117, 0.0204, 0.0292],  # MSCI Japan
    [0.0000, 0.0323, 0.0646, 0.1076,  0.0000, 0.0258, 0.0452, 0.0646],  # MSCI Asia Ex-Japan
    [0.0000, 0.0063, 0.0126, 0.0210,  0.0000, 0.0050, 0.0088, 0.0126],  # MSCI EM ex-Asia
    [0.0000, 0.0000, 0.0000, 0.0000,  0.0500, 0.1000, 0.1500, 0.2000],  # Private Equity
    [0.0000, 0.0000, 0.0000, 0.0000,  0.0100, 0.0200, 0.0300, 0.0400],  # Private Credit
    [0.0000, 0.0000, 0.0000, 0.0000,  0.0100, 0.0200, 0.0300, 0.0400],  # Insurance-Linked
    [0.0000, 0.0000, 0.0000, 0.0000,  0.0100, 0.0200, 0.0300, 0.0400],  # Hedge Funds
    [0.0000, 0.0000, 0.0000, 0.0000,  0.0200, 0.0400, 0.0600, 0.0800],  # Real Assets
])
BENCHMARK_LABELS = [
    "Income w/o Alts", "Low w/o Alts", "Balanced w/o Alts", "Growth w/o Alts",
    "Income w/Alts",   "Low w/Alts",   "Balanced w/Alts",   "Growth w/Alts",
]


# ═══════════════════════════════════════════════════════════════════
# 2. Structural Sharpe-ratio prior for λ
# ═══════════════════════════════════════════════════════════════════
# Order matches y_betas columns in the production xlsx:
#   Equity, Rates, Credit, Carry, Inflation, Commodities, Private Equity,
#   Rates Vol, FX
#
# SR_MEANS matches the production MATF_SHARPE_RATIOS vector used in the
# MATF-CMA pipeline:
#   - Equity, Rates, Carry, Long Rates Vol = 0.30 (conservative structural value)
#   - Credit = 0.25
#   - Inflation, Commodities = 0.10
#   - Private Equity = 0.50 (illiquidity premium)
#   - FX = 0.00 (no long-run drift; structural)
SR_MEANS = np.array([0.30, 0.30, 0.25, 0.30, 0.10, 0.10, 0.50, 0.30, 0.00])

# 95% range ≈ ±0.20 around per-factor mean → std = 0.10.
# Tighter than the empirical sample-mean noise scale σ/√T_eff ≈ 0.20,
# reflecting that structural equilibrium SR priors carry information beyond
# the 26-year effective sample.
SR_STD = 0.10

# Cross-factor correlations of the SR prior are derived from the production
# factor covariance Σ_F (sheet `x_covar` in the production xlsx), which is the
# same EWMA-based estimator (36-month span) used elsewhere in the paper —
# notably Exhibit~\ref{tb:risk_factors_corr}.
#
# The SR-prior covariance is then:
#     Σ_SR = SR_STD² · ρ_F
# where ρ_F is derived from Σ_F at data-load time and stored in RealData.

def _correlation_from_covariance(cov: np.ndarray) -> np.ndarray:
    """Convert covariance matrix to correlation matrix."""
    sigma = np.sqrt(np.diag(cov))
    sigma = np.where(sigma > 0, sigma, 1.0)  # defensive guard
    return cov / np.outer(sigma, sigma)


def sample_lambda_from_sr(rng: np.random.Generator,
                          sigma_F_diag: np.ndarray,
                          Sigma_SR: np.ndarray) -> np.ndarray:
    """Draw λ_b = SR_b ⊙ σ_F,b from the structural prior.

    Sigma_SR is the SR-prior covariance: SR_STD² · ρ_F, where ρ_F is the
    production factor correlation matrix (derived from the EWMA Σ_F).
    """
    SR_b = rng.multivariate_normal(SR_MEANS, Sigma_SR)
    return SR_b * sigma_F_diag


def lambda_baseline_from_sr(Sigma_F: np.ndarray) -> np.ndarray:
    """Baseline λ at SR_means (no random shock); used for the dashed
    point-estimate frontier."""
    return SR_MEANS * np.sqrt(np.diag(Sigma_F))


# ═══════════════════════════════════════════════════════════════════
# 3. Data container
# ═══════════════════════════════════════════════════════════════════
@dataclass
class RealData:
    # Frontier universe (15 assets)
    frontier_tickers: List[str]
    frontier_names:   List[str]
    frontier_freq:    List[str]               # 'ME' / 'QE'
    fpy_per_asset:    np.ndarray              # frequency-per-year, (15,)
    beta_frontier:    np.ndarray              # (15, M) loadings on M factors
    rf_rate:          float
    lambda_:          np.ndarray              # (M,) baseline λ

    # Paired bootstrap panels
    factor_returns:   np.ndarray              # (T, M)  monthly log returns
    residuals_native: np.ndarray              # (T, 15) native-freq residuals

    # Pipeline prior — used as fallback for any unobserved factor blocks
    Sigma_F_prior:    np.ndarray              # (M, M)
    factor_names:     List[str]
    bootstrapped_factors: np.ndarray          # indices of factors to resample

    # SR-prior covariance: SR_STD² · ρ_F, where ρ_F is derived from Σ_F_prior.
    # Loaded once at data-load time so the bootstrap reuses the production
    # factor correlation matrix consistently across draws.
    Sigma_SR:         np.ndarray              # (M, M)

    # Asset panel (centered) for the raw-bootstrap comparator
    asset_returns:    np.ndarray              # (T, 15)
    asset_index:      pd.DatetimeIndex

    bmk_weights:      np.ndarray
    bmk_labels:       List[str]
    freq_per_year:    int = 12

    @property
    def N(self) -> int: return len(self.frontier_tickers)
    @property
    def M(self) -> int: return self.beta_frontier.shape[1]
    @property
    def T(self) -> int: return self.asset_returns.shape[0]


# ═══════════════════════════════════════════════════════════════════
# 4. Loader
# ═══════════════════════════════════════════════════════════════════
# Late-start asset backfill rules
# ────────────────────────────────────────────────────────────────────
# Two assets in the 17-asset universe begin after the bootstrap window
# start (April 2001) and require backfill via proxy series:
#
#   1. Global Convertibles (H24641US)  starts Jan 2009.
#      Backfill rule: 1.3 × H23059US (Global HY) at native monthly freq.
#      Justification: vol-scaling on the post-2009 sample where both
#      observed (Convertibles ann vol 10.1% / HY ann vol 7.9% ≈ 1.28).
#      Reproduces the actual 2008 Bloomberg US Convertibles drawdown
#      of ≈ −36% within historical bounds (1.3×HY 2008 = −34%).
#
#   2. Insurance-Linked (LGT_ILS)  starts Dec 2002.
#      Backfill rule: HFRXGL resampled to quarterly compounded returns,
#      1× scaling, applied at QE-grid dates only (ME months stay NaN
#      by convention). Apr 2001 – Sep 2002 window only (6 QE prints).
#      The 2008-stress concern does not apply: HFRXGL during
#      Apr 2001 – Sep 2002 had max DD only −1.7%, comparable to ILS.
BACKFILL_RULES = [
    # (target_ticker, proxy_ticker, scale, end_date) — end_date inclusive
    ("H24641US Index",  "H23059US Index", 1.30, "2008-12-31"),
    ("LGT_ILS Index",   "HFRXGL Index",   1.00, "2002-09-30"),
]
# Bootstrap window starts at the first valid index of H23059US (April 2001).
BOOTSTRAP_WINDOW_START = "2001-04-01"


# ─────────────────────────────────────────────────────────────────────
# AR(1) unsmoothing for appraisal-based assets (PE / PC / ILS)
# ─────────────────────────────────────────────────────────────────────
# Mirrors production semantics from
# ``rosaa.universe.unsmoothing.copy_universe_data_with_unsmoothed_prices``
# called inside ``estimate_global_saa_cma``. The same AR(1) regression
# is applied to PE-flagged columns of the price panel before returns are
# computed for the bootstrap, so the bootstrap operates on the same
# unsmoothed series the production CMA pipeline sees.
#
# Asset selection: prefers an explicit `Unsmoothing` column on
# `cma_metadata`. Falls back to `Sub Asset Class` membership in
# {Private Equity, Private Debt, Diversified, Insurance-Linked} only
# when the explicit column is absent. The set above lists the
# *production* sub-asset-class labels (verbatim from
# `rosaa.universe.unsmoothing.PE_SUB_ASSET_CLASSES`), which must match
# the upstream string exactly. The paper renames the corresponding
# asset (`MP503008 Index`) to "Private Credit" for display purposes,
# but the sub-asset-class label remains "Private Debt" in the
# production xlsx.
#
# Production parameters (from rosaa.universe.unsmoothing module):
#   span = 20, max_value_for_beta = 0.5, warmup_period = 8,
#   mean_adj_type = qis.MeanAdjType.EWMA, is_log_returns = True
PE_SUB_ASSET_CLASSES = {
    'Private Equity', 'Private Debt', 'Diversified', 'Insurance-Linked',
}
UNSMOOTH_SPAN              = 20
UNSMOOTH_MAX_BETA          = 0.5
UNSMOOTH_WARMUP            = 8
UNSMOOTH_IS_LOG_RETURNS    = True


def _resolve_unsmoothing_assets(meta: pd.DataFrame,
                                ar_columns: List[str]) -> List[str]:
    """Resolve assets flagged for AR(1) unsmoothing.

    Mirrors ``rosaa.universe.unsmoothing.get_unsmoothing_mask``:
        1. If `Unsmoothing` column exists, treat as bool mask.
        2. Else, fall back to `Sub Asset Class` ∈ PE_SUB_ASSET_CLASSES.
    """
    if 'Unsmoothing' in meta.columns:
        flagged = meta.index[meta['Unsmoothing'].astype(bool)]
    elif 'Sub Asset Class' in meta.columns:
        flagged = meta.index[meta['Sub Asset Class'].isin(PE_SUB_ASSET_CLASSES)]
    else:
        return []
    return [t for t in flagged if t in ar_columns]


def _apply_unsmoothing(ar: pd.DataFrame,
                       res_native: pd.DataFrame,
                       meta: pd.DataFrame,
                       verbose: bool = True
                       ) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply AR(1) unsmoothing to PE-flagged columns of both the
    asset-returns panel and the native-frequency residuals panel.

    NOTE: Currently UNUSED. The production xlsx's `asset_returns` and
    `residuals` sheets are already written with unsmoothing applied
    upstream by `rosaa.market_data.cmas.estimate_global_saa_cma` (which
    calls `copy_universe_data_with_unsmoothed_prices` before
    `estimate_asset_universe_cma_data` populates the sheets). Calling
    this helper on those sheets would double-correct.

    Kept for documentation and as a fallback for any future code path
    that reads the *raw* (smoothed) `universe_data` excel directly
    rather than the post-CMA-pipeline xlsx.

    Production order: prices → unsmooth → returns. We invert: returns →
    cumulative price → unsmooth → returns. The cumulative-price step is
    deterministic so the round-trip preserves the AR(1) regression's
    output exactly.

    Frequency for the regression comes from `meta['Rebalancing']`
    (per-asset Series), matching production's `freq=freq` argument
    routed through `_resolve_freq` in unsmoothing.py.
    """
    pe_assets = _resolve_unsmoothing_assets(meta, list(ar.columns))
    if not pe_assets:
        if verbose:
            print(f"[unsmooth] no assets flagged — skipping")
        return ar, res_native

    if verbose:
        flagged_str = ", ".join(t.split(' ')[0] for t in pe_assets)
        print(f"[unsmooth] applying AR(1) to {len(pe_assets)} assets: "
              f"{flagged_str}")

    try:
        from optimalportfolios import compute_ar1_unsmoothed_prices
        import qis
    except ImportError as exc:
        raise ImportError(
            "AR(1) unsmoothing requires `optimalportfolios` and `qis`. "
            "Install them or set apply_unsmoothing=False at load time."
        ) from exc

    freq_per_asset = meta.loc[pe_assets, 'Rebalancing']

    def _unsmooth_panel(panel: pd.DataFrame, label: str) -> pd.DataFrame:
        """Convert returns→cumulative price→unsmooth→returns for the
        flagged assets, leave the rest untouched."""
        result = panel.copy()
        # Build cumulative-price proxies starting at 100; NaN-aware via
        # pandas cumsum semantics (NaN-aware cumulative product on
        # 1+returns would propagate NaNs forward incorrectly for QE
        # assets). Use cumulative log-returns then exponentiate.
        log_panel = result[pe_assets].copy()
        # Anchor each asset at its first valid observation
        cum = log_panel.fillna(0.0).cumsum()
        prices = 100.0 * np.exp(cum)
        # Restore NaN positions so the regression sees the QE pattern
        # exactly as in the input panel.
        prices = prices.where(~log_panel.isna(), other=np.nan)
        # Forward-fill QE prices to ME grid for the regression's
        # rebalancing logic; production does this internally via
        # `compute_ar1_unsmoothed_prices` when the freq argument is QE.
        prices = prices.ffill()

        un_prices, _, _, _ = compute_ar1_unsmoothed_prices(
            prices=prices,
            freq=freq_per_asset,
            span=UNSMOOTH_SPAN,
            max_value_for_beta=UNSMOOTH_MAX_BETA,
            warmup_period=UNSMOOTH_WARMUP,
            mean_adj_type=qis.MeanAdjType.EWMA,
            is_log_returns=UNSMOOTH_IS_LOG_RETURNS,
        )
        un_prices = un_prices.reindex(index=panel.index).ffill()
        # Convert unsmoothed prices back to log-returns
        un_log = np.log(un_prices / un_prices.shift(1))
        # Restore the original NaN pattern (QE on non-quarter months)
        un_log = un_log.where(~log_panel.isna(), other=np.nan)
        result[pe_assets] = un_log

        if verbose:
            for t in pe_assets:
                old_vol = log_panel[t].std() * np.sqrt(
                    12 if meta.loc[t, 'Rebalancing'] == 'ME' else 4) * 100
                new_vol = un_log[t].std() * np.sqrt(
                    12 if meta.loc[t, 'Rebalancing'] == 'ME' else 4) * 100
                print(f"[unsmooth]   {label:<10s} {t.split(' ')[0]:<20s}: "
                      f"vol {old_vol:5.2f}% → {new_vol:5.2f}%")
        return result

    ar_un = _unsmooth_panel(ar, "returns")
    res_un = _unsmooth_panel(res_native, "residuals")
    return ar_un, res_un


def _apply_backfill(ar: pd.DataFrame, res_native: pd.DataFrame, meta: pd.DataFrame,
                    verbose: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Apply BACKFILL_RULES to both asset_returns and native-freq residuals.

    For ME-frequency proxies feeding QE-frequency targets, the proxy series
    is first compounded to QE returns; the target is then filled at QE-grid
    dates only, leaving ME-grid months NaN per the QE convention.
    """
    ar  = ar.copy()
    res = res_native.copy()
    window_start = pd.Timestamp(BOOTSTRAP_WINDOW_START)

    for target, proxy, scale, end_date in BACKFILL_RULES:
        if target not in ar.columns or proxy not in ar.columns:
            if verbose:
                print(f"[backfill] skip {target}: ticker not in panel")
            continue
        target_freq = meta.loc[target, 'Rebalancing']
        proxy_freq  = meta.loc[proxy,  'Rebalancing']
        end_ts = pd.Timestamp(end_date)

        # Mask: rows in [start, end_date] where target is NaN
        mask = (ar.index <= end_ts) & ar[target].isna()

        # Build the proxy series in the target's native frequency
        if target_freq == proxy_freq:
            proxy_series = ar[proxy] * scale
        elif target_freq == 'QE' and proxy_freq == 'ME':
            # Compound monthly proxy returns to QE; assign at QE month-ends
            proxy_qe = (1.0 + ar[proxy]).resample('QE').prod() - 1.0
            proxy_series = pd.Series(np.nan, index=ar.index)
            for qe_dt, qe_val in proxy_qe.items():
                # Align QE date to nearest panel index (panel uses month-end)
                if qe_dt in ar.index:
                    proxy_series.loc[qe_dt] = qe_val * scale
        else:
            raise NotImplementedError(
                f"Backfill {proxy_freq} → {target_freq} not supported")

        # Apply to asset_returns
        ar.loc[mask, target] = proxy_series.loc[mask]
        # Count only in-window fills (the bootstrap-relevant population)
        in_window_filled_ar = (mask & proxy_series.notna() &
                               (ar.index >= window_start)).sum()

        # Apply to residuals: same rule, same scaling, same QE-alignment.
        if target_freq == proxy_freq:
            proxy_res = res[proxy] * scale
        else:  # QE target, ME proxy
            proxy_res_qe = (1.0 + res[proxy]).resample('QE').prod() - 1.0
            proxy_res = pd.Series(np.nan, index=res.index)
            for qe_dt, qe_val in proxy_res_qe.items():
                if qe_dt in res.index:
                    proxy_res.loc[qe_dt] = qe_val * scale

        mask_res = (res.index <= end_ts) & res[target].isna()
        res.loc[mask_res, target] = proxy_res.loc[mask_res]
        in_window_filled_res = (mask_res & proxy_res.notna() &
                                (res.index >= window_start)).sum()

        if verbose:
            print(f"[backfill] {target} ← {scale}× {proxy}  "
                  f"asset_returns: {in_window_filled_ar} obs, residuals: {in_window_filled_res} obs, "
                  f"through {end_date}")

    return ar, res


def load_real_data(xlsx_path: str,
                   factor_navs_path: str,
                   verbose: bool = True) -> RealData:
    """Load production CMA pipeline data and factor NAVs into RealData.

    Parameters
    ----------
    xlsx_path : str
        Path to the production CMA pipeline xlsx for the USD investor at the
        publication's reference quarter (2026-Q1). Sheets required:
            asset_returns, residuals, y_betas, x_covar, y_variances,
            cma_metadata, factor_cmas
    factor_navs_path : str
        Path to the daily factor NAV CSV (column per factor, in the same
        order as y_betas columns). Each NAV is a price index with NAV_0 = 100.
    """
    if verbose:
        print(f"[load] {xlsx_path}")
        print(f"[load] factor NAVs: {factor_navs_path}")

    ar    = pd.read_excel(xlsx_path, sheet_name='asset_returns').set_index('Unnamed: 0')
    res   = pd.read_excel(xlsx_path, sheet_name='residuals').set_index('Unnamed: 0')
    betas = pd.read_excel(xlsx_path, sheet_name='y_betas').set_index('Unnamed: 0')
    xcov  = pd.read_excel(xlsx_path, sheet_name='x_covar').set_index('Unnamed: 0')
    meta  = pd.read_excel(xlsx_path, sheet_name='cma_metadata').set_index('Unnamed: 0')

    meta = meta[~meta.index.duplicated(keep='first')]
    ar.index  = pd.to_datetime(ar.index)
    res.index = pd.to_datetime(res.index)
    factor_names = list(betas.columns)
    M = len(factor_names)

    # Annualised → native-frequency residuals
    me_all = [t for t in ar.columns if t in meta.index and meta.loc[t, 'Rebalancing'] == 'ME']
    qe_all = [t for t in ar.columns if t in meta.index and meta.loc[t, 'Rebalancing'] == 'QE']
    res_native = res.copy().astype(float)
    res_native[me_all] = res[me_all] / 12.0
    res_native[qe_all] = res[qe_all] / 4.0

    # Backfill late-start assets (H24641US, LGT_ILS) per BACKFILL_RULES.
    # Both panels (asset_returns, residuals_native) get the same treatment.
    ar, res_native = _apply_backfill(ar, res_native, meta, verbose=verbose)

    # NOTE: AR(1) Geltner-style unsmoothing for PE / PC / ILS has already been
    # applied upstream by `rosaa.market_data.cmas.estimate_global_saa_cma`,
    # which calls `copy_universe_data_with_unsmoothed_prices` on the universe
    # before `estimate_asset_universe_cma_data` writes the asset_returns and
    # residuals sheets. The series stored in the xlsx are therefore the
    # unsmoothed ones; applying unsmoothing here would double-correct.
    # Empirical confirmation: lag-1 autocorrelation of MP503001 (PE) = +0.10,
    # MP503008 (PC) = +0.07, LGT_ILS = +0.02, all well below the +0.35 to +0.50
    # range typical of raw appraisal-based series.

    # Trim panels to the bootstrap window. Start = first valid index of HY
    # (Apr 2001). After the trim and backfill, all 17 columns are either fully
    # observed (ME assets) or QE-pattern by design (PE / PC; ILS is now
    # fully QE-observed within the window thanks to the backfill).
    window_start = pd.Timestamp(BOOTSTRAP_WINDOW_START)
    ar         = ar[ar.index >= window_start]
    res_native = res_native[res_native.index >= window_start]
    if verbose:
        print(f"[load] bootstrap window: {ar.index[0].date()} → "
              f"{ar.index[-1].date()} ({len(ar)} months)")

    # Frontier subset
    front_tickers = [t for t, _ in PAPER_TICKERS]
    front_names   = [n for _, n in PAPER_TICKERS]
    front_freq    = meta.loc[front_tickers, 'Rebalancing'].tolist()
    fpy_asset     = np.array([12 if f == 'ME' else 4 for f in front_freq])

    beta_front = betas.loc[front_tickers].to_numpy(dtype=float)
    rf_rate    = float(meta.loc[front_tickers, 'rf_rate'].iloc[0])

    # Factor returns: load directly from production NAV file
    fnav = pd.read_csv(factor_navs_path, index_col=0, parse_dates=True)
    # Case-insensitive column match: handle 'FX' vs 'Fx' between data sources.
    fnav_lower_to_orig = {c.lower(): c for c in fnav.columns}
    rename_map = {}
    missing = []
    for fn in factor_names:
        if fn in fnav.columns:
            continue
        if fn.lower() in fnav_lower_to_orig:
            rename_map[fnav_lower_to_orig[fn.lower()]] = fn
        else:
            missing.append(fn)
    if missing:
        raise ValueError(f"Factor NAV file missing columns: {missing}")
    if rename_map:
        fnav = fnav.rename(columns=rename_map)
    fnav = fnav[factor_names]

    # Re-index to asset_returns calendar (forward-fill non-trading days)
    fnav_aligned = fnav.reindex(ar.index, method='ffill')
    if fnav_aligned.isna().any().any():
        n_nan = fnav_aligned.isna().sum().sum()
        raise ValueError(f"Factor NAV alignment produced {n_nan} NaNs — "
                         f"factor history doesn't cover asset_returns range")

    # Monthly log returns; first row dropped because log return at t=0
    # requires NAV at t=-1.
    factor_returns_df = np.log(fnav_aligned / fnav_aligned.shift(1))

    factor_returns     = factor_returns_df.iloc[1:].to_numpy(dtype=float)
    asset_returns_full = ar.iloc[1:][front_tickers].to_numpy(dtype=float)
    residuals_full     = res_native.iloc[1:][front_tickers].to_numpy(dtype=float)
    asset_index_full   = ar.index[1:]
    T_aligned          = factor_returns.shape[0]

    if verbose:
        print(f"[load] paired panel: {asset_index_full[0].date()} → "
              f"{asset_index_full[-1].date()} ({T_aligned} months)")

    # Sanity check: full-sample factor cov from CSV vs pipeline EWMA
    full_cov_csv = np.cov(factor_returns, rowvar=False, ddof=1) * 12
    if verbose:
        print(f"[load] factor vols (CSV full-sample) vs pipeline EWMA:")
        for j, fn in enumerate(factor_names):
            rec   = np.sqrt(full_cov_csv[j, j]) * 100
            prior = np.sqrt(xcov.iloc[j, j]) * 100
            print(f"       {fn:<16s} CSV={rec:5.2f}%  pipeline={prior:5.2f}%")

    # λ baseline at SR prior mean
    sigma_F_diag_full = np.sqrt(np.diag(full_cov_csv))
    lambda_baseline = SR_MEANS * sigma_F_diag_full
    mu_matf_baseline = rf_rate + beta_front @ lambda_baseline

    # Per-asset μ recentering: shift each series so its sample mean matches
    # the MATF baseline. Variances/covariances/serial correlation/NaN
    # positions are all preserved.
    mu_raw_full = np.array([
        np.nanmean(asset_returns_full[:, i]) * fpy_asset[i]
        for i in range(len(front_tickers))
    ])
    delta_native = (mu_matf_baseline - mu_raw_full) / fpy_asset
    asset_returns_centered = asset_returns_full.copy()
    for i in range(len(front_tickers)):
        mask = ~np.isnan(asset_returns_centered[:, i])
        asset_returns_centered[mask, i] += delta_native[i]

    if verbose:
        print(f"[load] per-asset recentering: shift range "
              f"[{delta_native.min()*100:+.3f}%, "
              f"{delta_native.max()*100:+.3f}%] per period")
        rmse = np.sqrt(np.mean(
            (beta_front @ lambda_baseline -
             meta.loc[front_tickers, 'base_excess_cma'].to_numpy()) ** 2)) * 100
        print(f"[load] β·λ vs base_excess_cma RMSE (15 frontier): {rmse:.2f}%")

    Sigma_F_prior_arr = xcov.loc[factor_names, factor_names].to_numpy(dtype=float)
    rho_F = _correlation_from_covariance(Sigma_F_prior_arr)
    Sigma_SR = (SR_STD ** 2) * rho_F
    if verbose:
        print(f"[load] SR-prior cross-factor correlation matrix taken from x_covar")
        print(f"[load]   ρ(Equity, Credit)      = {rho_F[0, 2]:+.2f}")
        print(f"[load]   ρ(Equity, Rates)       = {rho_F[0, 1]:+.2f}")
        print(f"[load]   ρ(Equity, Rates Vol)   = {rho_F[0, 7]:+.2f}")

    return RealData(
        frontier_tickers     = front_tickers,
        frontier_names       = front_names,
        frontier_freq        = front_freq,
        fpy_per_asset        = fpy_asset,
        beta_frontier        = beta_front,
        rf_rate              = rf_rate,
        lambda_              = lambda_baseline,
        factor_returns       = factor_returns,
        residuals_native     = residuals_full,
        Sigma_F_prior        = Sigma_F_prior_arr,
        factor_names         = factor_names,
        bootstrapped_factors = np.arange(M),
        Sigma_SR             = Sigma_SR,
        asset_returns        = asset_returns_centered,
        asset_index          = asset_index_full,
        bmk_weights          = BENCHMARK_WEIGHTS,
        bmk_labels           = BENCHMARK_LABELS,
    )


# ═══════════════════════════════════════════════════════════════════
# 5. Per-draw moment estimators
# ═══════════════════════════════════════════════════════════════════
def factor_cov_eq_weighted(factor_sample: np.ndarray,
                           prior_Sigma_F: np.ndarray,
                           bootstrapped_idx: np.ndarray,
                           fpy: int = 12) -> np.ndarray:
    """Σ_F: equal-weighted sample cov on the bootstrapped sub-block,
    pipeline prior on any unobserved factor block."""
    Sigma_F = prior_Sigma_F.copy()
    sub = factor_sample[:, bootstrapped_idx]
    valid = ~np.isnan(sub).any(axis=1)
    if valid.sum() < 24:
        return Sigma_F
    cov_sub = np.cov(sub[valid], rowvar=False, ddof=1) * fpy
    for ii, i in enumerate(bootstrapped_idx):
        for jj, j in enumerate(bootstrapped_idx):
            Sigma_F[i, j] = cov_sub[ii, jj]
    Sigma_F = 0.5 * (Sigma_F + Sigma_F.T)
    w, V = np.linalg.eigh(Sigma_F)
    w = np.clip(w, 1e-10, None)
    return V @ np.diag(w) @ V.T


def residual_var_eq_weighted(residual_sample: np.ndarray,
                             fpy_per_asset: np.ndarray) -> np.ndarray:
    """Per-asset annualised residual variance from the resampled ε panel.
    NaN-safe; falls back to 1e-8 if a column has no observations in the draw."""
    df = pd.DataFrame(residual_sample)
    var_native = df.var().to_numpy()
    var_native = np.where(np.isnan(var_native), 1e-8, var_native)
    return var_native * fpy_per_asset


def asset_moments_raw(asset_sample: np.ndarray,
                      freq: List[str],
                      fpy: int = 12) -> Tuple[np.ndarray, np.ndarray]:
    """NaN-aware annualised (μ̂, Σ̂) for the raw asset-level estimator.

    For pairwise covariance with mixed monthly/quarterly frequencies, scale
    each series by √(fpy_asset) before computing the standard pairwise-
    complete covariance, giving directly annualised cross-terms.
    """
    N = asset_sample.shape[1]
    df = pd.DataFrame(asset_sample)
    fpy_asset = np.array([fpy if f == 'ME' else 4 for f in freq])
    mu = np.array([df[i].dropna().mean() * fpy_asset[i] for i in range(N)])
    scaled = df.multiply(np.sqrt(fpy_asset), axis=1)
    Sigma = scaled.cov().to_numpy()
    Sigma = 0.5 * (Sigma + Sigma.T)
    w, V = np.linalg.eigh(Sigma)
    w = np.clip(w, 1e-10, None)
    return mu, V @ np.diag(w) @ V.T


# ═══════════════════════════════════════════════════════════════════
# 6. Bootstrap driver
# ═══════════════════════════════════════════════════════════════════
@dataclass
class RealResult:
    data: RealData
    vol_grid: np.ndarray
    raw_returns: np.ndarray            # (B, G)
    factor_returns: np.ndarray         # (B, G)
    baseline_raw: np.ndarray           # (G,)
    baseline_factor: np.ndarray        # (G,)
    n_boot: int
    mean_block: float
    min_block: int
    Sigma_F_baseline: np.ndarray
    D_baseline: np.ndarray
    lambda_baseline: np.ndarray
    delta_norm_gk: np.ndarray = None       # (B,) ‖Δ‖₂ per draw under GK
    delta_norm_matf: np.ndarray = None     # (B,) ‖Δ‖₂ per draw under MATF (≡ 0)
    delta_per_asset_gk: np.ndarray = None  # (B, N) per-asset Δ_i under GK


def run_real_bootstrap(data: RealData,
                       n_boot: int = 500,
                       mean_block: float = 12.0,
                       min_block: int = 3,
                       n_vol_points: int = 24,
                       seed: int = 42,
                       verbose: bool = True) -> RealResult:
    """Run the bootstrap exhibit on real production data.

    Parameters
    ----------
    data : RealData
        Output of ``load_real_data``.
    n_boot : int
        Number of bootstrap draws (paper uses 500).
    mean_block : float
        Mean stationary block length in months (paper uses 12).
    min_block : int
        Minimum block length; 3 ensures at least one quarter per block for
        QE-frequency assets (Private Equity, Private Credit, Insurance-Linked).
    n_vol_points : int
        Number of vol-target grid points spanning the long-only feasible
        range (paper uses 24).
    seed : int
        RNG seed (paper uses 42 for reproducibility).
    """
    rng = np.random.default_rng(seed)
    fpy = data.freq_per_year

    # Equal-weighted full-sample baselines (consistent with bootstrap)
    Sigma_F_baseline = factor_cov_eq_weighted(
        data.factor_returns, data.Sigma_F_prior, data.bootstrapped_factors, fpy=fpy)
    D_baseline = residual_var_eq_weighted(data.residuals_native, data.fpy_per_asset)
    lambda_baseline = lambda_baseline_from_sr(Sigma_F_baseline)

    mu_full = data.rf_rate + data.beta_frontier @ lambda_baseline
    Sigma_full = (data.beta_frontier @ Sigma_F_baseline @ data.beta_frontier.T
                  + np.diag(D_baseline))
    Sigma_full = 0.5 * (Sigma_full + Sigma_full.T)

    mu_raw_full, Sigma_raw_full = asset_moments_raw(
        data.asset_returns, data.frontier_freq, fpy=fpy)

    # Vol grid: covers all 8 benchmark mandates, capped below the long-only
    # max-μ corner singularity.
    vol_min = max(min_variance_vol(Sigma_raw_full),
                  min_variance_vol(Sigma_full)) * 1.02
    vol_max_target = 0.15
    vol_max_feasible = min(
        np.sqrt(Sigma_raw_full[np.argmax(mu_raw_full), np.argmax(mu_raw_full)]),
        np.sqrt(Sigma_full[np.argmax(mu_full), np.argmax(mu_full)]),
    ) * 0.99
    vol_max = min(vol_max_target, vol_max_feasible)
    vol_grid = np.linspace(vol_min, vol_max, n_vol_points)

    if verbose:
        print(f"[boot] vol grid ∈ [{vol_min:.3f}, {vol_max:.3f}], {n_vol_points} pts")
        print("[boot] baseline frontiers (equal-weighted full-sample)...")

    baseline_raw    = solve_long_only_frontier(mu_raw_full, Sigma_raw_full, vol_grid)
    baseline_factor = solve_long_only_frontier(mu_full,     Sigma_full,     vol_grid)

    # Bootstrap loop
    raw_out = np.full((n_boot, n_vol_points), np.nan)
    fac_out = np.full((n_boot, n_vol_points), np.nan)
    T = data.T

    if verbose:
        print(f"[boot] {n_boot} draws | block: mean={mean_block:g}m, "
              f"min={min_block}m (QE-safe) | paired (factor, residual, asset)")
        print(f"[boot] λ via SR prior: SR ~ N(SR_means, Σ_SR), λ_b = SR_b · σ_F,b")
        print(f"[boot] SR_means = {SR_MEANS}")
        print(f"[boot] SR_std   = {SR_STD:.3f} (95% range ~[-0.20, +0.20] around mean)")

    delta_norm_gk      = np.full(n_boot, np.nan)
    delta_norm_matf    = np.zeros(n_boot)              # ≡ 0 by construction
    delta_per_asset_gk = np.full((n_boot, data.N), np.nan)

    # OLS projection matrix: P = β (β'β)⁻¹ β'
    beta_pinv = np.linalg.pinv(data.beta_frontier.T @ data.beta_frontier) @ data.beta_frontier.T
    proj_matrix = data.beta_frontier @ beta_pinv

    for b in range(n_boot):
        idx = stationary_block_indices(T, mean_block, rng, min_block=min_block)

        # Raw / GK panel — μ_GK = sample mean per asset (math equivalent of GK)
        asset_sample = data.asset_returns[idx]
        mu_r, S_r = asset_moments_raw(asset_sample, data.frontier_freq, fpy=fpy)
        raw_out[b] = solve_long_only_frontier(mu_r, S_r, vol_grid)

        excess_mu_gk = mu_r - data.rf_rate
        delta_b = excess_mu_gk - proj_matrix @ excess_mu_gk
        delta_per_asset_gk[b] = delta_b
        delta_norm_gk[b] = np.linalg.norm(delta_b)

        # MATF factor panel — paired resampling
        factor_sample   = data.factor_returns[idx]
        residual_sample = data.residuals_native[idx]

        Sigma_F_b = factor_cov_eq_weighted(
            factor_sample, data.Sigma_F_prior, data.bootstrapped_factors, fpy=fpy)
        D_b = residual_var_eq_weighted(residual_sample, data.fpy_per_asset)

        sigma_F_diag_b = np.sqrt(np.diag(Sigma_F_b))
        lambda_b = sample_lambda_from_sr(rng, sigma_F_diag_b, data.Sigma_SR)

        mu_f = data.rf_rate + data.beta_frontier @ lambda_b
        S_f  = (data.beta_frontier @ Sigma_F_b @ data.beta_frontier.T
                + np.diag(D_b))
        S_f  = 0.5 * (S_f + S_f.T)
        fac_out[b] = solve_long_only_frontier(mu_f, S_f, vol_grid)

        if verbose and (b + 1) % max(1, n_boot // 10) == 0:
            print(f"  {b+1:4d}/{n_boot}")

    return RealResult(
        data=data, vol_grid=vol_grid,
        raw_returns=raw_out, factor_returns=fac_out,
        baseline_raw=baseline_raw, baseline_factor=baseline_factor,
        n_boot=n_boot, mean_block=mean_block, min_block=min_block,
        Sigma_F_baseline=Sigma_F_baseline, D_baseline=D_baseline,
        lambda_baseline=lambda_baseline,
        delta_norm_gk=delta_norm_gk,
        delta_norm_matf=delta_norm_matf,
        delta_per_asset_gk=delta_per_asset_gk,
    )


# ═══════════════════════════════════════════════════════════════════
# 7. Plots and dispersion tables
# ═══════════════════════════════════════════════════════════════════
def plot_frontier_fan(r: RealResult,
                      savepath: str,
                      figsize: Tuple[float, float] = (14, 6)) -> plt.Figure:
    """Two-panel bootstrap frontier exhibit (Exhibit 1 in the paper)."""
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True, sharex=True)
    vol_pct = r.vol_grid * 100

    # Benchmark dots use the same baselines as the point-estimate frontier
    mu_r, S_r = asset_moments_raw(r.data.asset_returns, r.data.frontier_freq,
                                  fpy=r.data.freq_per_year)
    mu_f = r.data.rf_rate + r.data.beta_frontier @ r.lambda_baseline
    S_f = (r.data.beta_frontier @ r.Sigma_F_baseline @ r.data.beta_frontier.T
           + np.diag(r.D_baseline))

    panels = [
        (axes[0], r.raw_returns,    r.baseline_raw,    mu_r, S_r,
         "A. Raw asset-level bootstrap",          "#d62728"),
        (axes[1], r.factor_returns, r.baseline_factor, mu_f, S_f,
         "B. MATF factor-structured bootstrap",  "#2ca02c"),
    ]
    for ax, samples, baseline, mu_b, S_b, name, col in panels:
        ax.plot(vol_pct, samples.T * 100, color=col, alpha=0.03, linewidth=0.7)
        p5  = np.nanpercentile(samples, 5,  axis=0) * 100
        p95 = np.nanpercentile(samples, 95, axis=0) * 100
        p50 = np.nanpercentile(samples, 50, axis=0) * 100
        ax.fill_between(vol_pct, p5, p95, color=col, alpha=0.25, label="5–95% band")
        ax.plot(vol_pct, p50, color=col, linewidth=2.0, label="Median")
        ax.plot(vol_pct, baseline * 100, color="black", linewidth=1.6,
                linestyle="--", label="Point estimate")

        W = r.data.bmk_weights
        bmk_ret = (mu_b @ W) * 100
        bmk_vol = np.sqrt(np.einsum("ni,nm,mi->i", W, S_b, W)) * 100
        ax.scatter(bmk_vol, bmk_ret, marker="D", s=55, facecolors="white",
                   edgecolors="black", linewidths=1.2, zorder=5, label="Benchmarks")
        ax.set_title(name, fontsize=11, loc="left")
        ax.set_xlabel("Volatility (% ann.)")
        ax.grid(True, alpha=0.3)
        ax.legend(loc="lower right", fontsize=9, framealpha=0.9)

    axes[0].set_ylabel("Expected return (% ann.)")
    fig.tight_layout()
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    return fig


def plot_consistency_violation(r: RealResult,
                               savepath: str,
                               figsize: Tuple[float, float] = (14, 5)) -> plt.Figure:
    """Two-panel alpha-beta consistency violation diagnostic (Exhibit 2).

    Panel A: histogram of ‖Δ_b‖₂ across bootstrap draws — GK distribution
    versus MATF (≡ 0 by construction).

    Panel B: per-asset boxplots of |Δ_i| under GK, sorted by median violation.
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Panel A — distribution of ‖Δ‖ across draws
    ax = axes[0]
    norm_gk_pct = r.delta_norm_gk * 100
    bins = np.linspace(0, np.nanpercentile(norm_gk_pct, 99) * 1.05, 40)
    ax.hist(norm_gk_pct, bins=bins, alpha=0.55, color="#d62728", density=True,
            label=f"Sample-mean proxy: median={np.nanmedian(norm_gk_pct):.2f}%, "
                  f"95%={np.nanpercentile(norm_gk_pct,95):.2f}%")
    ax.axvline(0, color="#2ca02c", linewidth=2.5,
               label="MATF: Δ ≡ 0 (by construction)")
    ax.set_xlabel(r"$\|\Delta_b\|_2$ — total consistency violation per draw (%)")
    ax.set_ylabel("Density")
    ax.set_title("A. Distribution of consistency violation across bootstrap draws",
                 fontsize=11, loc="left")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=9)

    # Panel B — per-asset |Δ_i| boxplot
    ax = axes[1]
    abs_per_asset = np.abs(r.delta_per_asset_gk) * 100
    medians = np.nanmedian(abs_per_asset, axis=0)
    order = np.argsort(-medians)
    asset_names_sorted = [r.data.frontier_names[i] for i in order]
    box_data = [abs_per_asset[:, i][np.isfinite(abs_per_asset[:, i])]
                for i in order]
    bp = ax.boxplot(box_data, vert=False, labels=asset_names_sorted,
                    patch_artist=True, widths=0.6, showfliers=False)
    for patch in bp['boxes']:
        patch.set_facecolor("#d62728")
        patch.set_alpha(0.55)
    for median_line in bp['medians']:
        median_line.set_color("black")
    ax.axvline(0, color="#2ca02c", linewidth=2.0, alpha=0.7,
               label="MATF: 0 by construction")
    ax.set_xlabel(r"$|\Delta_i|$ per asset under sample-mean proxy (%)")
    ax.set_title("B. Per-asset consistency violation under sample-mean proxy",
                 fontsize=11, loc="left")
    ax.grid(True, alpha=0.3, axis="x")
    ax.legend(loc="lower right", fontsize=9)
    ax.invert_yaxis()

    fig.tight_layout()
    fig.savefig(savepath, dpi=150, bbox_inches="tight")
    return fig


def dispersion_table(r: RealResult,
                     vol_points: Tuple[float, ...] = (0.04, 0.05, 0.06, 0.07, 0.08)
                     ) -> pd.DataFrame:
    """90% bandwidth comparison of raw vs MATF at selected vol targets."""
    rows = []
    seen: set = set()
    for v in vol_points:
        k = int(np.argmin(np.abs(r.vol_grid - v)))
        if k in seen:
            continue
        seen.add(k)
        rr = r.raw_returns[:, k]
        fr = r.factor_returns[:, k]
        rr = rr[np.isfinite(rr)]
        fr = fr[np.isfinite(fr)]
        if len(rr) == 0 or len(fr) == 0:
            continue
        wr = np.percentile(rr, 95) - np.percentile(rr, 5)
        wf = np.percentile(fr, 95) - np.percentile(fr, 5)
        rows.append({
            "Target vol":   f"{r.vol_grid[k]*100:.1f}%",
            "Raw 5%":       f"{np.percentile(rr, 5)*100:.2f}%",
            "Raw 95%":      f"{np.percentile(rr, 95)*100:.2f}%",
            "Raw 90% w":    f"{wr*100:.2f}%",
            "Factor 5%":    f"{np.percentile(fr, 5)*100:.2f}%",
            "Factor 95%":   f"{np.percentile(fr, 95)*100:.2f}%",
            "Factor 90% w": f"{wf*100:.2f}%",
            "Reduction":    f"{wr / max(wf, 1e-8):.0f}×",
        })
    return pd.DataFrame(rows)


def consistency_violation_summary(r: RealResult) -> pd.DataFrame:
    """Per-asset summary of |Δ_i| under GK across bootstrap draws."""
    abs_per_asset = np.abs(r.delta_per_asset_gk) * 100
    rows = []
    for i, n in enumerate(r.data.frontier_names):
        col = abs_per_asset[:, i]
        col = col[np.isfinite(col)]
        if len(col) == 0:
            continue
        rows.append({
            "Asset":      n,
            "Median |Δ|": f"{np.median(col):.2f}%",
            "5%":         f"{np.percentile(col, 5):.2f}%",
            "95%":        f"{np.percentile(col, 95):.2f}%",
        })
    df = pd.DataFrame(rows)
    return df.sort_values(
        "Median |Δ|",
        key=lambda s: s.str.rstrip('%').astype(float),
        ascending=False,
    ).reset_index(drop=True)


def reconciliation_check(r: RealResult) -> pd.DataFrame:
    """Bootstrap median vs point-estimate frontier (Michaud lift diagnostic).

    Under λ resampling, the median sits above the point estimate by 30–70 bp
    because of Jensen's inequality: with a long-only constraint, the optimiser
    exploits positive λ shocks more than it suffers from negative ones.
    The lift is a structural feature of resampled efficient frontiers, not a
    bias in the bootstrap implementation.
    """
    p50 = np.nanpercentile(r.factor_returns, 50, axis=0)
    gap = (p50 - r.baseline_factor) * 100
    return pd.DataFrame({
        "Vol target":       [f"{v*100:.1f}%" for v in r.vol_grid],
        "Point estimate":   [f"{v*100:.2f}%" for v in r.baseline_factor],
        "Bootstrap median": [f"{v*100:.2f}%" for v in p50],
        "Michaud lift":     [f"{g:+5.2f}%" for g in gap],
    })


# ═══════════════════════════════════════════════════════════════════
# 8. Entry point
# ═══════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    HERE = os.path.dirname(os.path.abspath(__file__))
    XLSX = os.path.join(HERE, "data", "global_saa_universe_data_cmas_usd_2026q1_jpm.xlsx")
    CSV  = os.path.join(HERE, "data", "futures_risk_factors.csv")
    FIG  = os.path.join(HERE, "figures")
    os.makedirs(FIG, exist_ok=True)

    data   = load_real_data(XLSX, CSV)
    result = run_real_bootstrap(
        data, n_boot=500, mean_block=12.0, min_block=3,
        n_vol_points=24, seed=42, verbose=True,
    )

    out_frontier = os.path.join(FIG, "bootstrap_frontier.png")
    plot_frontier_fan(result, out_frontier)
    print(f"\n[plot] {out_frontier}")

    out_consistency = os.path.join(FIG, "consistency_violation.png")
    plot_consistency_violation(result, out_consistency)
    print(f"[plot] {out_consistency}")

    print("\nMichaud lift (median > point estimate is correct under λ noise):")
    print(reconciliation_check(result).to_string(index=False))

    print("\nλ baseline (SR × σ_F^{full-sample}):")
    for fn, lam, sr in zip(data.factor_names, result.lambda_baseline, SR_MEANS):
        print(f"  {fn:<16s} SR={sr:5.2f}  λ={lam*100:5.2f}%")

    mu_raw_centered, _ = asset_moments_raw(
        data.asset_returns, data.frontier_freq, fpy=12)
    mu_matf = data.rf_rate + data.beta_frontier @ result.lambda_baseline
    print(f"\nPer-asset alignment (recentered raw vs MATF baseline):")
    print(f"{'Asset':<22s} {'μ_raw':>8s} {'μ_matf':>8s} {'err':>6s}")
    for i, n in enumerate(data.frontier_names):
        e = (mu_raw_centered[i] - mu_matf[i]) * 100
        print(f"  {n:<20s} {mu_raw_centered[i]*100:>7.2f}% "
              f"{mu_matf[i]*100:>7.2f}% {e:>+5.3f}%")

    print(f"\nFrontier point-estimate gap (raw - matf):")
    print(f"{'vol':>6s} {'gap':>8s}")
    for k in [0, 6, 12, 18, 23]:
        v = result.vol_grid[k]
        g = (result.baseline_raw[k] - result.baseline_factor[k]) * 100
        print(f"{v*100:>5.1f}% {g:>+7.2f}%")

    print("\nDispersion (90% bands):")
    print(dispersion_table(result).to_string(index=False))

    print(f"\n=== Consistency violation: GK vs MATF ===")
    print(f"GK   median ‖Δ‖₂: {np.nanmedian(result.delta_norm_gk)*100:>6.2f}%")
    print(f"GK   95%    ‖Δ‖₂: {np.nanpercentile(result.delta_norm_gk,95)*100:>6.2f}%")
    print(f"MATF        ‖Δ‖₂:   0.00% (by construction)")
    print(f"\nPer-asset |Δ| under GK (sorted):")
    print(consistency_violation_summary(result).to_string(index=False))
