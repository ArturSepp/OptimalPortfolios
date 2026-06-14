"""
identity_analytics.py — closed-form analytics for the MATF Identity.

Companion code to:
    Sepp, A., and M.A. Kastenholz (2026), "Achievable Sharpe and Universe
    Selection: A Closed-Form Factor Decomposition", working paper, under
    review at the Financial Analysts Journal.

Core objects (notation of the paper, Appendix A):
    lam   : (M,)   factor excess risk premia
    Sig_F : (M,M)  factor covariance, annualized
    beta  : (N,M)  loadings (production HCGL)
    D     : (N,)   residual variances, annualized
    alpha : (N,)   admitted idiosyncratic alpha (excess units)

All Sharpe quantities are excess and annualized; the algebra is on SR^2.
"""
from __future__ import annotations
import numpy as np


# ── building blocks ──────────────────────────────────────────────────
def factor_information_matrix(beta: np.ndarray, D: np.ndarray) -> np.ndarray:
    """beta_F = beta' D^-1 beta  (M x M)."""
    return beta.T @ (beta / D[:, None])


def frictionless_ceiling(lam: np.ndarray, Sig_F: np.ndarray) -> float:
    """lam' Sig_F^-1 lam — squared Sharpe with direct, costless factor access."""
    return float(lam @ np.linalg.solve(Sig_F, lam))


def matf_identity(lam: np.ndarray, Sig_F: np.ndarray,
                  beta: np.ndarray, D: np.ndarray) -> float:
    """Systematic squared Sharpe lam'(Sig_F + beta_F^-1)^-1 lam.

    Computed in the rank-safe Woodbury form (paper Appendix D):
        ceiling - (Sig_F^-1 lam)' (Sig_F^-1 + beta_F)^-1 (Sig_F^-1 lam),
    valid for rank-deficient beta_F (N < M or zero loading columns).
    """
    bF = factor_information_matrix(beta, D)
    SinvL = np.linalg.solve(Sig_F, lam)
    return float(lam @ SinvL
                 - SinvL @ np.linalg.solve(np.linalg.inv(Sig_F) + bF, SinvL))


def fpir(lam, Sig_F, beta, D) -> float:
    """Factor premium identification ratio = identity / ceiling, in [0, 1]."""
    return matf_identity(lam, Sig_F, beta, D) / frictionless_ceiling(lam, Sig_F)


def per_factor_contributions(lam, Sig_F, beta, D):
    """Additive per-factor split (paper Eq. per_factor).

    Returns (contrib_identity, contrib_ceiling): lam_k * u_k with
    u = (Sig_F + beta_F^-1)^-1 lam   (identity side)
    u = Sig_F^-1 lam                 (ceiling side).
    Each vector sums to the respective SR^2.
    """
    bF = factor_information_matrix(beta, D)
    u_id = bF @ np.linalg.solve(np.eye(len(lam)) + Sig_F @ bF, lam)
    u_ce = np.linalg.solve(Sig_F, lam)
    return lam * u_id, lam * u_ce


def identification_deficit(lam, beta, D):
    """Per-factor SE_k / lam_k with SE_k = sqrt([beta_F^-1]_kk).

    The sampling standard error of the GLS cross-sectional premium estimate
    relative to the premium. np.inf where lam_k = 0 (risk-only factors)."""
    bFi = np.linalg.inv(factor_information_matrix(beta, D))
    se = np.sqrt(np.diag(bFi))
    out = np.full_like(lam, np.inf, dtype=float)
    nz = np.abs(lam) > 0
    out[nz] = se[nz] / np.abs(lam[nz])
    return out


# ── weights (paper Eq. 7) ────────────────────────────────────────────
def factor_mimicking_portfolios(beta, D) -> np.ndarray:
    """P_F = D^-1 beta beta_F^-1   (N x M)."""
    bF = factor_information_matrix(beta, D)
    return (beta / D[:, None]) @ np.linalg.inv(bF)


def eq7_weight_blocks(lam, Sig_F, beta, D, alpha):
    """Systematic and idiosyncratic blocks of the tangency weights.

    w_sys  = P_F (Sig_F + beta_F^-1)^-1 lam
    w_idio = D^-1 alpha
    Exact reconstruction of Sigma^-1(mu - rf 1) requires GLS-orthogonal
    alpha (beta' D^-1 alpha = 0); see gls_orthogonal_alpha().
    """
    bF = factor_information_matrix(beta, D)
    P_F = factor_mimicking_portfolios(beta, D)
    w_sys = P_F @ np.linalg.solve(Sig_F + np.linalg.inv(bF), lam)
    return w_sys, alpha / D


# ── alpha handling ───────────────────────────────────────────────────
def gls_orthogonal_alpha(alpha_raw, beta, D) -> np.ndarray:
    """Project raw admitted alpha onto col(beta)^perp in the D^-1 metric."""
    bF = factor_information_matrix(beta, D)
    return alpha_raw - beta @ np.linalg.solve(bF, beta.T @ (alpha_raw / D))


def idiosyncratic_sr2(alpha, D) -> float:
    """alpha' D^-1 alpha."""
    return float(alpha @ (alpha / D))


def consistency_residual(alpha, beta, D) -> float:
    """max_k |[beta' D^-1 alpha]_k| — zero iff the (lam, alpha) pair is GLS-consistent."""
    return float(np.max(np.abs(beta.T @ (alpha / D))))


# ── factor-Sharpe coordinates (paper Eq. sr_coords) ──────────────────
def identity_sr_coordinates(SR_F, rho_F, beta, D, sigma_F):
    """SR_F' (rho_F + btilde_F^-1)^-1 SR_F with btilde_F the information
    matrix on unit-volatility factors. Volatility-free given (SR_F, rho_F)
    and the signal-to-noise structure inside btilde_F."""
    btF = np.outer(sigma_F, sigma_F) * factor_information_matrix(beta, D)
    return float(SR_F @ np.linalg.solve(rho_F + np.linalg.inv(btF), SR_F))


# ── universe selection ───────────────────────────────────────────────
def candidate_fpir_gain(lam, Sig_F, beta, D,
                        beta_new: np.ndarray, resid_var_new: float) -> float:
    """FPIR gain from adding one instrument: rank-one update
    beta_F -> beta_F + beta_new beta_new' / resid_var_new, computable
    before the sleeve is held (paper, 'Pricing a Candidate Sleeve')."""
    beta_aug = np.vstack([beta, beta_new[None, :]])
    D_aug = np.append(D, resid_var_new)
    return fpir(lam, Sig_F, beta_aug, D_aug) - fpir(lam, Sig_F, beta, D)


# ── constrained portfolios ───────────────────────────────────────────
def longonly_max_sharpe(mu_excess, Sigma):
    """Long-only tangency via min y'Sigma y s.t. mu'y = 1, y >= 0 (CVXPY/CLARABEL)."""
    import cvxpy as cp
    N = len(mu_excess)
    y = cp.Variable(N)
    prob = cp.Problem(cp.Minimize(cp.quad_form(y, cp.psd_wrap(Sigma))),
                      [mu_excess @ y == 1, y >= 0])
    prob.solve(solver=cp.CLARABEL)
    w = y.value / y.value.sum()
    sr = float(mu_excess @ w / np.sqrt(w @ Sigma @ w))
    return w, sr


def te_overlay(mu_excess, Sigma, sigma_te: float):
    """TE-budgeted active overlay dw = (sigma_te / SR) Sigma^-1 mu_excess.
    IR equals the tangency Sharpe at every budget (paper Eq. ir_bound)."""
    sr = np.sqrt(float(mu_excess @ np.linalg.solve(Sigma, mu_excess)))
    return (sigma_te / sr) * np.linalg.solve(Sigma, mu_excess), sr
