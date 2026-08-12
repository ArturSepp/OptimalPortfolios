"""
Bootstrap of the long-only frontier: raw asset-level against MATF factor-structured (Stage J5).

Reimplements Appendix B on the frozen 2026q2_custom panels, because the original
exhibit-build script did not arrive with the package. Two estimators of the same
frontier on the same 300-month window, so the comparison isolates SAMPLING
BANDWIDTH rather than level:

1) Raw asset-level. Each draw takes NaN-aware sample moments of the resampled
   asset panel. The panel holds native-frequency excess LOG returns, so a
   per-asset mean carries the arithmetic lift of one half the variance,
   mu_i = s_i mean(Y_i) + 0.5 s_i var(Y_i), and mixed-frequency covariances
   scale each series by sqrt(s_i) before the pairwise-complete covariance.
   Mathematically a Grinold-Kroner construction whose per-asset point estimate
   is the historical sample mean.

2) MATF factor-structured. Each draw re-estimates Sigma_F from the resampled
   factor panel and D from the resampled residual panel, draws a factor Sharpe
   vector from the structural prior, and rebuilds

       mu = beta (SR .* sigma_F),      Sigma = beta Sigma_F beta' + diag(D)

   on the SAME loading matrix as the rest of the framework.

Resampling is the stationary block bootstrap of Politis and Romano (1994) with
mean block L = 12 months and MINIMUM block L_min = 3 months, block starts
aligned to multiples of L_min under wrap-around so every block contains a full
quarter and each quarterly column receives an observation. One index vector per
draw is applied to the asset, factor and residual panels in PAIRED fashion, so
any joint dependence between factor stress and residual heteroskedasticity
survives.

Prior (roadmap B10, owner decision O-J1). The Sharpe mean vector resolves each
factor through the manifest's family_sharpe_ratio rows: (0.40, 0.25, 0.40,
0.40, 0.25, 0.25, 0.15, 0.15, 0.60, 0.25, 0.00) in the canonical factor
order, with sigma_SR = 0.10 and Sigma_SR = sigma_SR^2 rho_F.

Recentering (owner decision, 2026-07-30). The raw panel is shifted by a
constant per-asset offset to the MATF baseline at the prior mean before the raw
bootstrap runs, per the private methodology note. The shift preserves
variances, covariances, serial correlation and NaN positions; only the sample
mean moves. Without it the two point-estimate frontiers diverge at low
volatility and the bandwidth comparison mixes level disagreement into a
dispersion statement. Appendix B prose does not currently state the step, which
is an owner prose item.

Deviation from Appendix B, logged. Appendix B backfills two short-history
sleeves with pre-inception proxies (Insurance-Linked from the hedge-fund index
through Jun 2006, Europe ex-UK from the broad Europe index through Oct 2007).
Neither proxy series is in the frozen snapshot, so this implementation runs
NaN-aware on the un-backfilled panel: Insurance-Linked contributes from Sep 2006
and Europe ex-UK from Nov 2007. Owner decision 2026-07-30; the Appendix B
sentence needs the correction.

Outputs: bootstrap_frontier.PNG, figures/bootstrap_headline_q2.json,
figures/sr_sensitivity_q2.csv, and the Appendix B standard-error triple.

Units: decimal per annum. Frontier returns plot as TOTAL returns (excess plus
the reference cash rate); every optimization runs in excess space.
Main entry point: run_local_test(local_test).

Does not belong here: the mandate exhibits (run_optimisation.py and the
run_*_exhibits scripts) and the per-asset consistency comparison
(run_consistency_exhibits.py), whose point-estimate version of this comparison
carries the declared-channel reconciliation.
"""
# packages
import json
import warnings
import numpy as np
import pandas as pd
import cvxpy as cvx
import matplotlib.pyplot as plt
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple
# qis / project
import exhibit_style as es
from local_path import load_cma_data
from governed_cma_projection import SNAPSHOT, load_paper_inputs
from run_optimisation import build_moments, get_benchmark

_cma_data = load_cma_data()

# --- bootstrap configuration (roadmap B8: any other randomness is a defect) ---
N_DRAWS = 500
SEED = 42
MEAN_BLOCK = 12                  # months
MIN_BLOCK = 3                    # months, keeps each block quarter-complete
N_VOL_TARGETS = 24
SIGMA_SR = 0.10
SIGMA_SR_GRID = (0.050, 0.075, 0.100, 0.125, 0.150, 0.200)
BANDWIDTH_QUANTILES = (0.05, 0.95)
FREQUENCY_FACTORS: Dict[str, int] = {'ME': 12, 'QE': 4}
PSD_EIGENVALUE_FLOOR = 1e-10
HEADLINE_MANDATE = 'Balanced with Alts'      # its benchmark vol is the R2 9.3% target

# R2 anchors, reported beside the regenerated values
R2_REDUCTION_RATIO = 2.01
# The R2 Grinold-Kroner value of 3.64% does not reproduce from the components Appendix B
# prints with rho = 0.5, which give 3.77%. Owner ruling O-J9 (2026-07-30): ADOPT 3.77% as
# computed from the printed components. The [UNTRACED] marker is resolved by adoption.
R2_SE_TRIPLE = {'sample_mean': 0.0324, 'grinold_kroner': 0.0364, 'matf': 0.0153}
GK_ADOPTED_NOTE = ('O-J9: adopted at the value the printed components imply; '
                   'the R2 3.64% is superseded')
R2_SE_SPEC = {'us_equity_vol': 0.162, 'beta_equity': 0.99, 'sigma_f_equity': 0.154,
              't_eff': 25.0}
# Grinold-Kroner component standard errors and the shared pairwise correlation (Appendix B)
GK_COMPONENT_SES = (0.001, 0.015, 0.010, 0.020)
GK_CORRELATION = 0.5


# --------------------------------------------------------------------------
# panels
# --------------------------------------------------------------------------

def build_bootstrap_panels(inputs) -> Dict[str, pd.DataFrame]:
    """the three paired panels on one monthly grid: asset returns, factor returns, residuals.

    Factor returns enter at each asset's NATIVE frequency, so the residuals are
    the native-frequency residuals of the loading regression. Quarterly assets
    carry NaN on non-quarter-end months in every panel, and the MIN_BLOCK rule
    keeps the quarter phase intact under resampling.
    """
    assets = inputs.assets
    returns = inputs.require_panel('asset_excess_logreturns').copy()
    grid = returns.index
    navs = inputs.require_panel('factor_navs').asfreq('B', method='ffill')

    monthly = np.log(navs.resample('ME').last()).diff().reindex(grid)
    quarterly = np.log(navs.resample('QE').last()).diff().reindex(grid)

    residuals = pd.DataFrame(index=grid, columns=returns.columns, dtype=float)
    for ticker in returns.columns:
        frequency = assets.loc[ticker, 'frequency']
        if frequency not in FREQUENCY_FACTORS:
            raise ValueError(f"unhandled frequency for {ticker!r}, got {frequency!r}")
        factor_panel = monthly if frequency == 'ME' else quarterly
        fitted = factor_panel.values @ inputs.betas.loc[ticker].values
        residuals[ticker] = returns[ticker].values - fitted

    # drop the first grid row: the factor differences are undefined there
    keep = grid[1:]
    panels = {'returns': returns.loc[keep],
              'factors_monthly': monthly.loc[keep],
              'residuals': residuals.loc[keep]}
    if len(keep) != 300:
        print(f"bootstrap window is {len(keep)} months, not the 300 Appendix B states")
    return panels


def recenter_returns(returns: pd.DataFrame,
                     inputs,
                     prior_mean: pd.Series,
                     ) -> Tuple[pd.DataFrame, pd.Series]:
    """shift each asset's series so its sample mean matches the MATF baseline at the prior mean.

    Y_centered[t, i] = Y[t, i] + (mu_matf_i - mu_raw_i) / s_i, in EXCESS space so
    the reference cash rate never enters. Variances, covariances, serial
    correlation and NaN positions are unchanged; only the sample mean moves.
    """
    assets = inputs.assets
    scale = assets['frequency'].map(FREQUENCY_FACTORS).astype(float)
    factor_vols = pd.Series(np.sqrt(np.diag(inputs.factor_covar.values)),
                            index=inputs.factor_covar.columns)
    mu_matf = pd.Series(inputs.betas.values @ (prior_mean * factor_vols).values,
                        index=assets.index)
    mu_raw = returns.mean(skipna=True) * scale + 0.5 * returns.var(skipna=True) * scale
    offsets = (mu_matf - mu_raw) / scale
    centered = returns.add(offsets, axis=1)
    return centered, offsets


# --------------------------------------------------------------------------
# stationary block bootstrap
# --------------------------------------------------------------------------

def stationary_block_indices(n_obs: int,
                             mean_block: int = MEAN_BLOCK,
                             min_block: int = MIN_BLOCK,
                             generator: Optional[np.random.Generator] = None,
                             ) -> np.ndarray:
    """one wrap-around index vector of length n_obs from geometric blocks with a minimum length.

    Block lengths are max(min_block, Geom(1 / mean_block)); block starts are
    drawn on the multiples of min_block so the quarter phase of the quarterly
    columns survives resampling.
    """
    if min_block < 1 or mean_block < min_block:
        raise ValueError(f"need 1 <= min_block <= mean_block, got {min_block!r} and {mean_block!r}")
    rng = np.random.default_rng() if generator is None else generator
    n_starts = n_obs // min_block
    indices: List[int] = []
    while len(indices) < n_obs:
        length = max(min_block, int(rng.geometric(p=1.0 / mean_block)))
        start = min_block * int(rng.integers(0, n_starts))
        indices.extend(((start + np.arange(length)) % n_obs).tolist())
    return np.asarray(indices[:n_obs], dtype=int)


def project_to_psd(matrix: np.ndarray, floor: float = PSD_EIGENVALUE_FLOOR) -> np.ndarray:
    """symmetrize and clip eigenvalues at a floor, so every draw yields a usable covariance."""
    symmetric = 0.5 * (matrix + matrix.T)
    values, vectors = np.linalg.eigh(symmetric)
    return vectors @ np.diag(np.clip(values, floor, None)) @ vectors.T


def annualized_sample_moments(returns: np.ndarray,
                              scale: np.ndarray,
                              ) -> Tuple[np.ndarray, np.ndarray]:
    """NaN-aware annualized arithmetic mean and pairwise-complete covariance of a log panel."""
    with np.errstate(invalid='ignore'):
        log_mean = np.nanmean(returns, axis=0) * scale
        variance = np.nanvar(returns, axis=0, ddof=1) * scale
    mu = log_mean + 0.5 * variance
    scaled = returns * np.sqrt(scale)[None, :]
    covar = pd.DataFrame(scaled).cov(min_periods=12).values      # pairwise-complete
    return mu, project_to_psd(np.nan_to_num(covar, nan=0.0))


# --------------------------------------------------------------------------
# frontier solver, parametrized once and re-solved per draw
# --------------------------------------------------------------------------

class FrontierSolver:
    """max mu'w s.t. w >= 0, 1'w = 1, ||L'w|| <= vol, compiled once through cvxpy Parameters.

    Parametrizing the objective vector, the covariance Cholesky factor and the
    volatility cap keeps the problem DPP-compliant, so 500 draws times 24 vol
    targets re-solve without recompiling.
    """

    def __init__(self, n_assets: int):
        if n_assets < 2:
            raise ValueError(f"need at least two assets, got {n_assets!r}")
        self.weights = cvx.Variable(n_assets, nonneg=True)
        self.mu = cvx.Parameter(n_assets)
        self.chol = cvx.Parameter((n_assets, n_assets))
        self.vol = cvx.Parameter(nonneg=True)
        self.problem = cvx.Problem(
            cvx.Maximize(self.mu @ self.weights),
            [cvx.sum(self.weights) == 1.0,
             cvx.norm(self.chol.T @ self.weights) <= self.vol])
        self.n_solves = 0
        self.n_inaccurate = 0        # solved but not to OPTIMAL status
        self.n_failed = 0            # returned no solution and dropped to NaN

    def solve(self, mu: np.ndarray, chol: np.ndarray, vol: float) -> Optional[np.ndarray]:
        """one frontier point; None when the solver does not reach an optimal status."""
        self.mu.value = mu
        self.chol.value = chol
        self.vol.value = float(vol)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')      # inaccurate solves are counted, not printed
                self.problem.solve(solver=cvx.CLARABEL)
        except cvx.error.SolverError:
            self.n_failed += 1
            return None
        self.n_solves += 1
        if self.problem.status != cvx.OPTIMAL:
            self.n_inaccurate += 1
        if self.weights.value is None:
            self.n_failed += 1
            return None
        return np.asarray(self.weights.value)

    def frontier_returns(self, mu: np.ndarray, covar: np.ndarray,
                         vol_targets: np.ndarray) -> np.ndarray:
        """the frontier over a vol grid; infeasible or failed points come back NaN.

        The parameter is the LOWER Cholesky factor L with L L' = Sigma, so the
        constraint ||L' w|| <= vol is exactly w' Sigma w <= vol^2.
        """
        lower = np.linalg.cholesky(project_to_psd(covar))
        out = np.full(len(vol_targets), np.nan)
        for i, vol in enumerate(vol_targets):
            weights = self.solve(mu=mu, chol=lower, vol=float(vol))
            if weights is not None:
                out[i] = float(mu @ weights)
        return out

    def minimum_variance_vol(self, covar: np.ndarray) -> float:
        n = covar.shape[0]
        w = cvx.Variable(n, nonneg=True)
        problem = cvx.Problem(cvx.Minimize(cvx.quad_form(w, cvx.psd_wrap(project_to_psd(covar)))),
                              [cvx.sum(w) == 1.0])
        problem.solve(solver=cvx.CLARABEL)
        if w.value is None:
            raise ValueError("minimum-variance solve failed")
        return float(np.sqrt(np.asarray(w.value) @ covar @ np.asarray(w.value)))


def build_vol_grid(solver: FrontierSolver,
                   moments: Dict[str, Tuple[np.ndarray, np.ndarray]],
                   n_points: int = N_VOL_TARGETS,
                   pin_targets: Tuple[float, ...] = (),      # exact targets to include
                   ) -> np.ndarray:
    """the Appendix B grid: 1.02 x the tighter minimum-variance vol to 0.92 x the best asset's vol.

    pin_targets are inserted exactly, so a headline reduction ratio is read off
    the grid rather than interpolated or snapped to a neighbour.
    """
    lower = 1.02 * min(solver.minimum_variance_vol(covar=covar) for _, covar in moments.values())
    upper = 0.92 * min(float(np.sqrt(covar[int(np.argmax(mu)), int(np.argmax(mu))]))
                       for mu, covar in moments.values())
    if upper <= lower:
        raise ValueError(f"empty vol grid, got lower {lower!r} and upper {upper!r}")
    grid = np.linspace(lower, upper, n_points)
    inside = [t for t in pin_targets if lower <= t <= upper]
    dropped = [t for t in pin_targets if t not in inside]
    if dropped:
        print(f"vol grid: pinned targets outside [{lower:.2%}, {upper:.2%}] and dropped: "
              f"{[f'{t:.2%}' for t in dropped]}")
    return np.unique(np.concatenate([grid, np.asarray(inside)])) if inside else grid


# --------------------------------------------------------------------------
# the bootstrap
# --------------------------------------------------------------------------

def get_prior_mean(inputs) -> pd.Series:
    """the July production Sharpe prior, read from the manifest rather than hard-coded (B10)."""
    config = pd.DataFrame(inputs.manifest['prod_config_snapshot'])
    rows = config.loc[config['group'] == 'matf_family_sharpe_ratios']
    family_prior = pd.Series({name.split('.', 1)[1]: float(value)
                              for name, value in zip(rows['Unnamed: 0'], rows['value'])})
    factor_family = {'Equity': 'Equity', 'Rates': 'Rates', 'Credit': 'Credit',
                     'Credit EM': 'Credit', 'Carry G10': 'Carry', 'Carry EM': 'Carry',
                     'Inflation': 'Inflation', 'Commodities': 'Commodities',
                     'Private Equity': 'Private Equity', 'Rates Vol': 'Rates Vol', 'Fx': 'Fx'}
    factors = list(inputs.factor_covar.columns)
    missing = [factor for factor in factors if factor not in factor_family]
    if missing:
        raise ValueError(f"factor-to-family prior mapping missing, got {missing!r}")
    prior = pd.Series({factor: family_prior[factor_family[factor]] for factor in factors})
    expected = pd.Series([0.40, 0.25, 0.40, 0.40, 0.25, 0.25, 0.15, 0.15,
                          0.60, 0.25, 0.00], index=factors)
    if float((prior - expected).abs().max()) > 1e-12:
        raise ValueError(f"manifest prior is not the July B10 vector, got {prior.to_dict()!r}")
    return prior


def run_bootstrap(inputs,
                  sigma_sr: float = SIGMA_SR,
                  n_draws: int = N_DRAWS,
                  seed: int = SEED,
                  vol_targets: Optional[np.ndarray] = None,
                  recenter: bool = True,
                  pin_targets: Tuple[float, ...] = (),
                  ) -> Dict[str, object]:
    """B draws of both frontiers on one shared index vector per draw."""
    if n_draws < 2:
        raise ValueError(f"need at least two draws, got {n_draws!r}")
    assets = inputs.assets
    prior_mean = get_prior_mean(inputs=inputs)
    panels = build_bootstrap_panels(inputs=inputs)
    returns = panels['returns']
    if recenter:
        returns, offsets = recenter_returns(returns=returns, inputs=inputs, prior_mean=prior_mean)
    else:
        offsets = pd.Series(0.0, index=assets.index)

    scale = assets['frequency'].map(FREQUENCY_FACTORS).astype(float).values
    returns_array = returns.values
    factors_array = panels['factors_monthly'].values
    residuals_array = panels['residuals'].values
    betas = inputs.betas.values
    correlation = inputs.factor_covar.values / np.outer(
        np.sqrt(np.diag(inputs.factor_covar.values)), np.sqrt(np.diag(inputs.factor_covar.values)))
    prior_covar = project_to_psd(sigma_sr ** 2 * correlation)
    prior_chol = np.linalg.cholesky(prior_covar)
    n_obs = returns_array.shape[0]

    solver = FrontierSolver(n_assets=returns_array.shape[1])
    if vol_targets is None:
        point_raw = annualized_sample_moments(returns=returns_array, scale=scale)
        point_matf = point_estimate_matf(inputs=inputs, prior_mean=prior_mean,
                                         factors=factors_array, residuals=residuals_array,
                                         scale=scale)
        vol_targets = build_vol_grid(solver=solver,
                                    moments={'raw': point_raw, 'matf': point_matf},
                                    pin_targets=pin_targets)

    # full-sample point estimates on the same grid, for the dashed lines the caption promises
    point_raw_moments = annualized_sample_moments(returns=returns_array, scale=scale)
    point_matf_moments = point_estimate_matf(inputs=inputs, prior_mean=prior_mean,
                                             factors=factors_array, residuals=residuals_array,
                                             scale=scale)
    point_estimates = {
        'raw': solver.frontier_returns(mu=point_raw_moments[0], covar=point_raw_moments[1],
                                       vol_targets=vol_targets),
        'matf': solver.frontier_returns(mu=point_matf_moments[0], covar=point_matf_moments[1],
                                        vol_targets=vol_targets)}

    # J8e: the consistency residual of each raw-estimator draw, against the SAME loadings and
    # residual variances the point-estimate diagnostic uses (run_consistency_exhibits)
    annihilator = build_annihilator(inputs=inputs)
    residual_norms = np.full(n_draws, np.nan)
    residual_abs = np.full((n_draws, returns_array.shape[1]), np.nan)

    rng = np.random.default_rng(seed)
    raw_paths = np.full((n_draws, len(vol_targets)), np.nan)
    matf_paths = np.full((n_draws, len(vol_targets)), np.nan)
    for draw in range(n_draws):
        indices = stationary_block_indices(n_obs=n_obs, generator=rng)
        mu_raw, covar_raw = annualized_sample_moments(returns=returns_array[indices], scale=scale)
        raw_paths[draw] = solver.frontier_returns(mu=mu_raw, covar=covar_raw,
                                                  vol_targets=vol_targets)
        deviation = annihilator @ mu_raw
        residual_abs[draw] = np.abs(deviation)
        residual_norms[draw] = float(np.linalg.norm(deviation))

        factor_covar = project_to_psd(np.cov(factors_array[indices], rowvar=False) * 12.0)
        factor_vols = np.sqrt(np.diag(factor_covar))
        with np.errstate(invalid='ignore'):
            residual_var = np.nanvar(residuals_array[indices], axis=0, ddof=1) * scale
        sharpe = prior_mean.values + prior_chol @ rng.standard_normal(len(prior_mean))
        mu_matf = betas @ (sharpe * factor_vols)
        covar_matf = betas @ factor_covar @ betas.T + np.diag(np.nan_to_num(residual_var))
        matf_paths[draw] = solver.frontier_returns(mu=mu_matf, covar=project_to_psd(covar_matf),
                                                   vol_targets=vol_targets)

    return {'vol_targets': vol_targets, 'raw_paths': raw_paths, 'matf_paths': matf_paths,
            'point_estimates': point_estimates,
            'residual_abs': pd.DataFrame(residual_abs, columns=assets.index),
            'residual_norms': pd.Series(residual_norms, name='violation_norm'),
            'offsets': offsets, 'prior_mean': prior_mean, 'sigma_sr': sigma_sr,
            'n_obs': n_obs,
            'solver_diagnostics': {'solves': solver.n_solves,
                                   'inaccurate': solver.n_inaccurate,
                                   'failed': solver.n_failed,
                                   'nan_raw': int(np.isnan(raw_paths).sum()),
                                   'nan_matf': int(np.isnan(matf_paths).sum())}}


def build_annihilator(inputs) -> np.ndarray:
    """the GLS annihilator M = I - beta (beta' D^-1 beta)^-1 beta' D^-1.

    M mu is the component of an excess-CMA vector the factor span cannot
    explain: exactly the consistency residual Delta of Appendix A, on the same
    loadings and residual variances as the point-estimate diagnostic in
    run_consistency_exhibits. Built once and reused across draws, because the
    loadings and D are held fixed by design — the comparison is about the
    expected-return vector, not the risk model.
    """
    betas = inputs.betas.values
    d_inv = 1.0 / inputs.assets['resid_vol'].values ** 2
    weighted = betas * d_inv[:, None]
    beta_f = weighted.T @ betas
    return np.eye(len(d_inv)) - betas @ np.linalg.solve(beta_f, weighted.T)


def summarize_residual_distribution(result: Dict[str, object]) -> Dict[str, object]:
    """the J8e triple: per-asset residual medians, the portfolio violation norm, and the range."""
    residual_abs = result['residual_abs']
    norms = result['residual_norms']
    per_asset = pd.DataFrame({'median_bp': 1e4 * residual_abs.median(axis=0),
                              'p05_bp': 1e4 * residual_abs.quantile(0.05, axis=0),
                              'p95_bp': 1e4 * residual_abs.quantile(0.95, axis=0)})
    return {'per_asset': per_asset,
            'median_norm': float(norms.median()),
            'p95_norm': float(norms.quantile(0.95)),
            'median_range_bp': (float(per_asset['median_bp'].min()),
                                float(per_asset['median_bp'].max()))}


def point_estimate_matf(inputs,
                        prior_mean: pd.Series,
                        factors: np.ndarray,
                        residuals: np.ndarray,
                        scale: np.ndarray,
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """the MATF point estimate at the prior mean on the full sample."""
    factor_covar = project_to_psd(np.cov(factors, rowvar=False) * 12.0)
    factor_vols = np.sqrt(np.diag(factor_covar))
    with np.errstate(invalid='ignore'):
        residual_var = np.nanvar(residuals, axis=0, ddof=1) * scale
    betas = inputs.betas.values
    mu = betas @ (prior_mean.values * factor_vols)
    covar = betas @ factor_covar @ betas.T + np.diag(np.nan_to_num(residual_var))
    return mu, project_to_psd(covar)


def summarize_bandwidth(result: Dict[str, object],
                        quantiles: Tuple[float, float] = BANDWIDTH_QUANTILES,
                        ) -> pd.DataFrame:
    """per-vol-target 90% bandwidth of each estimator and the reduction ratio."""
    low, high = quantiles
    table = pd.DataFrame(index=pd.Index(result['vol_targets'], name='vol_target'))
    for label, paths in (('raw', result['raw_paths']), ('matf', result['matf_paths'])):
        table[f"{label}_p{int(1e2 * low):02d}"] = np.nanquantile(paths, low, axis=0)
        table[f"{label}_p{int(1e2 * high):02d}"] = np.nanquantile(paths, high, axis=0)
        table[f"{label}_median"] = np.nanmedian(paths, axis=0)
        table[f"{label}_width"] = (table[f"{label}_p{int(1e2 * high):02d}"]
                                   - table[f"{label}_p{int(1e2 * low):02d}"])
    table['reduction'] = table['raw_width'] / table['matf_width']
    return table


def plot_bootstrap_frontier(bandwidth: pd.DataFrame,
                            benchmark_points: pd.DataFrame,
                            point_estimates: Dict[str, np.ndarray],
                            rf_rate: float,
                            n_draws: int = N_DRAWS,
                            figsize: Tuple[float, float] = (12.4, 5.4),
                            ) -> plt.Figure:
    """the frozen two-panel grammar: raw left, MATF right, band, median, point estimate, mandates.

    Follows the frozen exhibit exactly (roadmap J8c and J8d). The frozen version
    is the ONE exhibit in the build that carries no in-figure takeaway title and
    labels its panels tersely as 'A. …' / 'B. …', so no suptitle is drawn here.
    The dashed black full-sample point-estimate line is the line the manuscript
    caption already promises ("dashed lines the full-sample point estimates").
    """
    for key in ('raw', 'matf'):
        if key not in point_estimates:
            raise ValueError(f"point estimate missing for {key!r}, got {list(point_estimates)!r}")
    fig, axs = plt.subplots(1, 2, figsize=figsize, sharey=True)
    vols = 1e2 * bandwidth.index.values
    for ax, label, color, title in (
            (axs[0], 'raw', es.ORANGE, 'A. Raw asset-level bootstrap'),
            (axs[1], 'matf', es.BLUE, 'B. MATF factor-structured bootstrap')):
        ax.fill_between(vols, 1e2 * (bandwidth[f"{label}_p05"] + rf_rate),
                        1e2 * (bandwidth[f"{label}_p95"] + rf_rate),
                        color=color, alpha=0.20, zorder=2,
                        label=f"5–95% band, {n_draws} draws")
        ax.plot(vols, 1e2 * (bandwidth[f"{label}_median"] + rf_rate), color=color, lw=2.2,
                zorder=4, label='Median')
        ax.plot(vols, 1e2 * (np.asarray(point_estimates[label]) + rf_rate), color='0.15',
                lw=1.6, ls='--', zorder=5, label='Point estimate')
        ax.scatter(1e2 * benchmark_points['vol'], 1e2 * benchmark_points['total_return'],
                   marker='D', s=38, facecolor='w', edgecolor='0.2', lw=1.1, zorder=6,
                   label='Benchmarks')
        ax.set_xlabel('Volatility (% ann.)', fontsize=10.0)
        ax.set_title(title, fontsize=10.4, loc='left')
        es.style_axis(ax=ax, grid_axis='both', fontsize=9.0)
        ax.legend(fontsize=8.6, loc='lower right', frameon=False)
    axs[0].set_ylabel('Expected return (% ann.)', fontsize=10.0)
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------
# Appendix B standard-error triple
# --------------------------------------------------------------------------

def build_se_triple(inputs,
                    sigma_sr: float = SIGMA_SR,
                    t_eff: float = R2_SE_SPEC['t_eff'],
                    ) -> pd.DataFrame:
    """the tab:se_comparison triple for US equity, recomputed on snapshot inputs."""
    ticker = 'NDDUUS Index'
    asset_vol = float(inputs.assets.loc[ticker, 'total_vol'])
    beta_equity = float(inputs.betas.loc[ticker, 'Equity'])
    factor_vol = float(np.sqrt(inputs.factor_covar.loc['Equity', 'Equity']))

    sample_mean = asset_vol / np.sqrt(t_eff)
    variance = sum(se ** 2 for se in GK_COMPONENT_SES)
    variance += 2.0 * GK_CORRELATION * sum(
        GK_COMPONENT_SES[i] * GK_COMPONENT_SES[j]
        for i in range(len(GK_COMPONENT_SES)) for j in range(i + 1, len(GK_COMPONENT_SES)))
    grinold_kroner = float(np.sqrt(variance))
    matf = abs(beta_equity) * sigma_sr * factor_vol

    table = pd.DataFrame({
        'estimator': ['Sample mean', 'Grinold-Kroner', 'MATF-CMA'],
        'se': [sample_mean, grinold_kroner, matf],
        'r2_printed': [R2_SE_TRIPLE['sample_mean'], R2_SE_TRIPLE['grinold_kroner'],
                       R2_SE_TRIPLE['matf']],
        'note': ['', GK_ADOPTED_NOTE, '']}).set_index('estimator')
    table['delta'] = table['se'] - table['r2_printed']
    table.attrs['spec'] = {'us_equity_vol': asset_vol, 'beta_equity': beta_equity,
                           'sigma_f_equity': factor_vol, 'sigma_sr': sigma_sr, 't_eff': t_eff}
    table.attrs['ratio_phrase'] = ' : '.join(f"{1e2 * v:.1f}" for v in table['se'])
    return table


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def run_bootstrap_report(snapshot: str = SNAPSHOT,
                         n_draws: int = N_DRAWS,
                         save_outputs: bool = True,
                         ) -> Dict[str, object]:
    """the full Stage J5 run: headline bandwidth, sigma_SR sensitivity, and the SE triple."""
    inputs = load_paper_inputs(snapshot=snapshot)
    covar, mu_x, rf_rate = build_moments(inputs=inputs)
    print('=' * 78)
    print(f"Stage J5 — bootstrap, cut {snapshot}, B = {n_draws}, seed {SEED}, "
          f"block mean {MEAN_BLOCK}m / min {MIN_BLOCK}m")
    print('=' * 78)

    prior_mean = get_prior_mean(inputs=inputs)
    print(f"\nSR prior mean (B10, asserted against the manifest): {prior_mean.to_dict()}")

    benchmark_points = {}
    for mandate in _cma_data.MANDATES:
        benchmark = get_benchmark(inputs=inputs, mandate=mandate)
        benchmark_points[mandate] = {
            'vol': float(np.sqrt(benchmark @ covar.values @ benchmark)),
            'total_return': float(mu_x @ benchmark) + rf_rate}
    benchmark_points = pd.DataFrame(benchmark_points).T
    headline_vol = float(benchmark_points.loc[HEADLINE_MANDATE, 'vol'])

    result = run_bootstrap(inputs=inputs, n_draws=n_draws, pin_targets=(headline_vol,))
    print(f"\nrecentering offsets applied to the raw panel (bp per period):")
    print((1e4 * result['offsets']).round(1).to_string())
    print(f"\nsolver diagnostics: {result['solver_diagnostics']}")

    bandwidth = summarize_bandwidth(result=result)
    print('\n--- 90% frontier bandwidth by volatility target (percent p.a.) ---')
    printed = (1e2 * bandwidth[['raw_width', 'matf_width']]).round(2)
    printed['reduction'] = bandwidth['reduction'].round(2)
    print(printed.to_string())

    nearest = int(np.argmin(np.abs(bandwidth.index.values - headline_vol)))
    headline = bandwidth.iloc[nearest]
    print(f"\nheadline reduction ratio at the {HEADLINE_MANDATE} benchmark volatility "
          f"{headline_vol:.2%} (grid point {bandwidth.index[nearest]:.4%}, pinned exactly): "
          f"{headline['reduction']:.2f}x against the R2 {R2_REDUCTION_RATIO:.2f}x")
    print(f"mean reduction across the grid: {bandwidth['reduction'].mean():.2f}x "
          f"(range {bandwidth['reduction'].min():.2f} to {bandwidth['reduction'].max():.2f})")
    print("NOTE: Appendix B labels the 9.3% target the 'Balanced w/o-Alts' volatility. On the "
          f"corrected benchmark, Balanced w/o Alts is "
          f"{float(benchmark_points.loc['Balanced w/o Alts', 'vol']):.2%} and Balanced WITH Alts "
          f"is {headline_vol:.2%}, so 9.3% is the with-Alts figure. Owner prose item.")

    sensitivity = {}
    for sigma in SIGMA_SR_GRID:
        run = run_bootstrap(inputs=inputs, sigma_sr=sigma, n_draws=n_draws,
                            vol_targets=result['vol_targets'])
        table = summarize_bandwidth(result=run)
        ratio = float(table.iloc[nearest]['reduction'])
        sensitivity[sigma] = {'reduction_ratio': ratio, 't_eff_equivalent': ratio ** 2}
        print(f"  sigma_SR = {sigma:.3f}   reduction {ratio:.2f}x   "
              f"T_eff equivalent {ratio ** 2:.1f}x")
    sensitivity = pd.DataFrame(sensitivity).T
    sensitivity.index.name = 'sigma_sr'

    se_triple = build_se_triple(inputs=inputs)
    print('\n--- tab:se_comparison, US equity standard errors ---')
    print(se_triple.round(4).to_string())
    print(f"spec: {se_triple.attrs['spec']}")
    print(f"ratio phrase for the NOTE line: {se_triple.attrs['ratio_phrase']} "
          f"(R2 printed 3.2 : 3.6 : 1.5)")

    # J8e: the per-draw consistency-residual distribution the R2 prose quotes
    residuals = summarize_residual_distribution(result=result)
    low_bp, high_bp = residuals['median_range_bp']
    print('\n--- J8e: consistency residual of the raw estimator across draws ---')
    printed = residuals['per_asset'].copy()
    printed.index = inputs.assets['sleeve']
    print(printed.round(1).to_string())
    print(f"\nper-asset median |Delta_i| range: {low_bp:.0f} to {high_bp:.0f} bp "
          f"(R2 prose quotes 30 to 115 bp)")
    print(f"median portfolio violation norm ||Delta||_2: {residuals['median_norm']:.2%} "
          f"(R2 prose quotes 3.4%); 95th percentile {residuals['p95_norm']:.2%}")

    if save_outputs:
        es.save_figure(plot_bootstrap_frontier(bandwidth=bandwidth,
                                              benchmark_points=benchmark_points,
                                              point_estimates=result['point_estimates'],
                                              rf_rate=rf_rate, n_draws=n_draws),
                       'bootstrap_frontier.PNG')
        write_bootstrap_artifacts(bandwidth=bandwidth, sensitivity=sensitivity,
                                 se_triple=se_triple, headline=headline,
                                 headline_vol=headline_vol, result=result,
                                 residuals=residuals)
    return {'bandwidth': bandwidth, 'sensitivity': sensitivity, 'se_triple': se_triple,
            'benchmark_points': benchmark_points, 'residuals': residuals}


def write_bootstrap_artifacts(bandwidth: pd.DataFrame,
                              sensitivity: pd.DataFrame,
                              se_triple: pd.DataFrame,
                              headline: pd.Series,
                              headline_vol: float,
                              result: Dict[str, object],
                              residuals: Dict[str, object],
                              ) -> None:
    """the named data artifacts the manuscript quotes, plus the tab:sr_sensitivity fragment."""
    es.FIGURES_PATH.mkdir(parents=True, exist_ok=True)
    headline_file = es.FIGURES_PATH / 'bootstrap_headline_q2.json'
    low_bp, high_bp = residuals['median_range_bp']
    payload = {'snapshot': SNAPSHOT,
               'n_draws': N_DRAWS,
               'seed': SEED,
               'mean_block_months': MEAN_BLOCK,
               'min_block_months': MIN_BLOCK,
               'n_months': int(result['n_obs']),
               'sigma_sr': float(result['sigma_sr']),
               'sr_prior_mean': {k: float(v) for k, v in result['prior_mean'].items()},
               'recentered': True,
               'backfill_applied': False,
               'headline_mandate': HEADLINE_MANDATE,
               'headline_vol_target': float(headline_vol),
               'raw_width_90': float(headline['raw_width']),
               'matf_width_90': float(headline['matf_width']),
               'reduction_ratio': float(headline['reduction']),
               'mean_reduction_ratio': float(bandwidth['reduction'].mean()),
               # J8e: the per-draw consistency-residual distribution of the raw estimator
               'residual_median_bp_low': low_bp,
               'residual_median_bp_high': high_bp,
               'residual_norm_median': residuals['median_norm'],
               'residual_norm_p95': residuals['p95_norm']}
    headline_file.write_text(json.dumps(payload, indent=2), encoding='utf-8')
    print(f"artifact written: {headline_file}")

    residual_file = es.FIGURES_PATH / 'bootstrap_residuals_q2.csv'
    residuals['per_asset'].to_csv(residual_file)
    print(f"artifact written: {residual_file}")

    sensitivity_file = es.FIGURES_PATH / 'sr_sensitivity_q2.csv'
    sensitivity.to_csv(sensitivity_file)
    print(f"artifact written: {sensitivity_file}")

    regimes = {0.050: 'Aggressive prior', 0.075: 'Moderate--aggressive',
               0.100: 'Paper baseline', 0.125: 'Moderate--loose', 0.150: 'Loose',
               0.200: 'Prior equals data noise scale'}
    lines = ['% ===== tab:sr_sensitivity — regenerated on cma_data snapshot 2026q2_custom =====',
             '% Source: replication/run_bootstrap_q2.py, B = 500 per row, seed 42.',
             f"% Reduction ratio of the 90% frontier bandwidth at the {HEADLINE_MANDATE}",
             f"%   benchmark volatility {headline_vol:.2%}. Only sigma_SR varies across rows.",
             '% Prior mean is the July MATF_SHARPE_RATIOS vector (owner decision O-J1 / B10).',
             '']
    for sigma, row in sensitivity.iterrows():
        bold = sigma == SIGMA_SR
        ratio = f"{row['reduction_ratio']:.2f}$\\times$"
        t_eff = f"{row['t_eff_equivalent']:.1f}$\\times$"
        regime = regimes.get(sigma, '')
        if bold:
            lines.append(f"\t\t\t\t\\textbf{{{sigma:.3f}}} & \\textbf{{{ratio}}} & "
                         f"\\textbf{{{t_eff}}} & \\textbf{{{regime}}} \\\\")
        else:
            lines.append(f"\t\t\t\t{sigma:.3f} & {ratio} & {t_eff} & {regime} \\\\")
    lines.append('%')
    lines.append('% tab:se_comparison, recomputed on snapshot inputs. Grinold-Kroner is ADOPTED')
    lines.append('%   at 3.77% per owner ruling O-J9: the R2 3.64% does not reproduce from the')
    lines.append('%   components Appendix B prints (0.1%, 1.5%, 1.0%, 2.0%) with rho = 0.5.')
    for estimator, row in se_triple.iterrows():
        suffix = f"  [{row['note']}]" if row['note'] else ''
        lines.append(f"%   {estimator}: {row['se']:.2%} "
                     f"(R2 printed {row['r2_printed']:.2%}){suffix}")
    spec = se_triple.attrs['spec']
    lines.append(f"%   spec: sigma_i = {spec['us_equity_vol']:.1%}, T_eff = {spec['t_eff']:.0f}, "
                 f"beta_Eq = {spec['beta_equity']:.2f}, sigma_F,Eq = {spec['sigma_f_equity']:.1%}, "
                 f"sigma_SR = {spec['sigma_sr']:.2f}")
    lines.append(f"%   NOTE-line ratio phrase: {se_triple.attrs['ratio_phrase']} "
                 f"(R2 printed 3.2 : 3.6 : 1.5)")
    lines.append('%')
    lines.append(r"% --- tab:se_comparison body, US equity SE column ---")
    formulas = {'Sample mean': r"$\sigma_i / \sqrt{T_{\mathrm{eff}}}$",
                'Grinold-Kroner': r"Equation~\eqref{eq:var_gk}",
                'MATF-CMA': r"Equation~\eqref{eq:se_matf_full}"}
    inputs_col = {'Sample mean': r"$T_{\mathrm{eff}}$ asset returns",
                  'Grinold-Kroner': r"$K_{GK} = 4$ forecasts per asset",
                  'MATF-CMA': r"$M = 11$ universe-wide factor SRs"}
    sample_col = {'Sample mean': r"$T/L$ scales noise",
                  'Grinold-Kroner': r"fixed; no $T$ scaling",
                  'MATF-CMA': r"cross-asset pooling via $\hat{\boldsymbol\beta}$"}
    for estimator, row in se_triple.iterrows():
        lines.append(f"\t\t\t\t{estimator:<16s} & {inputs_col[estimator]} & "
                     f"{sample_col[estimator]} & {formulas[estimator]} & "
                     f"${row['se']:.2%}$ \\\\".replace('%', r'\%'))
    es.write_fragment(lines=lines, file_name='exhibit_sr_sensitivity.tex')


class LocalTests(str, Enum):
    FULL_BOOTSTRAP = 'full_bootstrap'
    QUICK_BOOTSTRAP = 'quick_bootstrap'
    SE_TRIPLE_ONLY = 'se_triple_only'


def run_local_test(local_test: LocalTests) -> None:
    """Run local tests for development and debugging purposes."""
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 250)

    if local_test == LocalTests.FULL_BOOTSTRAP:
        run_bootstrap_report()

    elif local_test == LocalTests.QUICK_BOOTSTRAP:
        run_bootstrap_report(n_draws=40, save_outputs=False)

    elif local_test == LocalTests.SE_TRIPLE_ONLY:
        inputs = load_paper_inputs()
        table = build_se_triple(inputs=inputs)
        print(table.round(4).to_string())
        print(table.attrs['spec'])

    else:
        raise NotImplementedError(f"{local_test}")


if __name__ == '__main__':
    run_local_test(local_test=LocalTests.FULL_BOOTSTRAP)
