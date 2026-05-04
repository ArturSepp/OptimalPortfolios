"""
Bootstrap frontier analytics — primitives for the MATF-CMA paper bootstrap exhibit.

Three functions:
    stationary_block_indices  : Politis-Romano (1994) sampler with optional
                                minimum-block-length floor for QE-safe sampling.
    solve_long_only_frontier  : CVXPY/CLARABEL solver for the QCQP
                                max μᵀw  s.t.  wᵀΣw ≤ v_k², 1ᵀw = 1, w ≥ 0
                                across a vol-target grid.
    min_variance_vol          : the global minimum-variance volatility implied
                                by Σ on the long-only fully-invested simplex.

These are kept as standalone primitives so the bootstrap driver can be read
independently of the rest of optimalportfolios.

References
----------
Politis, D.N., Romano, J.P. (1994). The Stationary Bootstrap. JASA 89(428):1303-1313.
"""
from __future__ import annotations

import numpy as np
import cvxpy as cp


# ──────────────────────────────────────────────────────────────────────
# Stationary block bootstrap (Politis-Romano 1994) with min-block floor
# ──────────────────────────────────────────────────────────────────────
def stationary_block_indices(T: int,
                             mean_block: float,
                             rng: np.random.Generator,
                             min_block: int = 1) -> np.ndarray:
    """Return ``T`` resampled indices into ``[0, T)``.

    Block lengths are drawn from ``Geom(1/mean_block)`` and floored at
    ``min_block``. Blocks wrap around the index space (circular sampling).

    Parameters
    ----------
    T : int
        Length of the time series being resampled.
    mean_block : float
        Mean block length. Per Politis-Romano, this is the parameter ``L``
        such that block lengths follow ``Geom(1/L)`` with mean ``L``.
    rng : np.random.Generator
        Numpy random generator (use ``np.random.default_rng(seed)``).
    min_block : int, default 1
        Minimum block length. With ``min_block = 3`` and a 12-month mean,
        every block contains at least one full quarter, so quarterly-
        frequency assets receive at least one non-NaN observation per block.
        This is required for the mixed-frequency 15-asset universe of the
        MATF-CMA paper.

    Returns
    -------
    idx : np.ndarray of shape (T,) and dtype int64
        Indices into the original series; can be applied to multiple paired
        panels (asset returns, factor returns, residuals) for joint resampling.
    """
    p = 1.0 / float(mean_block)
    idx = np.empty(T, dtype=np.int64)
    filled = 0
    while filled < T:
        start = int(rng.integers(0, T))
        L = max(min_block, int(rng.geometric(p)))
        end = min(filled + L, T)
        span = end - filled
        idx[filled:end] = (start + np.arange(span)) % T   # circular
        filled = end
    return idx


# ──────────────────────────────────────────────────────────────────────
# Long-only frontier solver
# ──────────────────────────────────────────────────────────────────────
def _psd_wrap(Sigma: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Symmetrise + add small ridge to make Σ numerically PSD for CVXPY."""
    S = 0.5 * (Sigma + Sigma.T)
    return S + eps * np.eye(S.shape[0])


def min_variance_vol(Sigma: np.ndarray, eps_psd: float = 1e-10) -> float:
    """Volatility of the global minimum-variance long-only portfolio.

    Solves min wᵀΣw  s.t. 1ᵀw = 1, w ≥ 0 and returns sqrt of the optimum.
    """
    Sigma_psd = _psd_wrap(Sigma, eps=eps_psd)
    L = np.linalg.cholesky(Sigma_psd)
    N = Sigma.shape[0]
    w = cp.Variable(N, nonneg=True)
    prob = cp.Problem(cp.Minimize(cp.norm(L.T @ w, 2)),
                      [cp.sum(w) == 1])
    try:
        prob.solve(solver=cp.CLARABEL)
    except cp.SolverError:
        return float("nan")
    if prob.status not in ("optimal", "optimal_inaccurate") or w.value is None:
        return float("nan")
    var_opt = max(float(w.value @ Sigma @ w.value), 0.0)
    return float(np.sqrt(var_opt))


def solve_long_only_frontier(mu: np.ndarray,
                             Sigma: np.ndarray,
                             vol_grid: np.ndarray,
                             eps_psd: float = 1e-10) -> np.ndarray:
    """Long-only fully-invested mean-variance frontier on a vol-target grid.

    For each ``v_k`` in ``vol_grid``, solves
        max μᵀw  s.t.  wᵀΣw ≤ v_k²,  1ᵀw = 1,  w ≥ 0.
    Returns the optimal expected returns; infeasible solves return NaN
    (typically when ``v_k`` is below the global minimum-variance volatility).

    Parameters
    ----------
    mu : np.ndarray (N,)
        Expected returns at the working frequency (annualised in our pipeline).
    Sigma : np.ndarray (N, N)
        Covariance matrix at the same frequency. Symmetrised internally;
        a ``eps_psd * I`` ridge is added before solving.
    vol_grid : np.ndarray (G,)
        Vol-target grid; same units as ``sqrt(diag(Sigma))``.
    eps_psd : float
        Ridge added to ``Sigma`` for numerical PSD-ness inside CLARABEL.

    Returns
    -------
    returns : np.ndarray (G,)
        Optimal expected returns at each vol target. NaN for infeasible.
    """
    Sigma_psd = _psd_wrap(Sigma, eps=eps_psd)
    G = len(vol_grid)
    out = np.full(G, np.nan)

    # Use Cholesky factor so the vol budget becomes a second-order cone:
    #     ||L'w||_2 ≤ v_k    ⇔    wᵀΣw ≤ v_k²
    # This is DPP-compliant and warm-starts efficiently across the vol grid.
    L = np.linalg.cholesky(Sigma_psd)

    N = mu.shape[0]
    w = cp.Variable(N, nonneg=True)
    v_param = cp.Parameter(nonneg=True)
    prob = cp.Problem(
        cp.Maximize(mu @ w),
        [cp.norm(L.T @ w, 2) <= v_param,
         cp.sum(w) == 1],
    )
    for k, v in enumerate(vol_grid):
        v_param.value = float(v)
        try:
            prob.solve(solver=cp.CLARABEL, warm_start=True)
            if prob.status in ("optimal", "optimal_inaccurate") and w.value is not None:
                out[k] = float(mu @ w.value)
        except cp.SolverError:
            pass
    return out
