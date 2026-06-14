"""
run_bootstrap.py — two-configuration bootstrap of the MATF Identity
(paper Section "Sampling Uncertainty", Figure 10 and Table 2).

Mirrors the two return panels of the MATF-CMA bootstrap (matf_cma_jpm_2026/
run_bootstrap.py): identical block sampler, draw count (500), and seed (42).

    Config A (ex ante) : lambda_b = SR_b (.) sigma_F_b, SR_b ~ N(SR_MEANS, Sigma_SR)
    Config B (ex post) : lambda_b = annualized mean of the resampled factor path

Per draw (shared): Politis-Romano stationary blocks (mean 12m, min 3m QE-safe),
equal-weighted Sig_F_b on resampled monthly factor log returns, native-frequency
equal-weighted D_b on resampled residuals, beta fixed at production HCGL.
"""
from __future__ import annotations
import os
import numpy as np

from identity_analytics import factor_information_matrix
from paper_inputs import load_production_inputs, build_bootstrap_panels, fpy_per_asset

FIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")


# ── canonical sampler ─────────────────────────────────────────────────
# Lifted VERBATIM from matf_cma_jpm_2026/bootstrap_frontier_analytics.py
# (Politis-Romano 1994 with min-block floor). Destined for the qis package;
# keep the signature identical so both papers' replication code can later
# `from qis import stationary_block_indices` unchanged.
def stationary_block_indices(T: int,
                             mean_block: float,
                             rng: np.random.Generator,
                             min_block: int = 1) -> np.ndarray:
    p = 1.0 / float(mean_block)
    idx = np.empty(T, dtype=np.int64)
    filled = 0
    while filled < T:
        start = int(rng.integers(0, T))
        L = max(min_block, int(rng.geometric(p)))
        end = min(filled + L, T)
        span = end - filled
        idx[filled:end] = (start + np.arange(span)) % T
        filled = end
    return idx


# ── structural SR prior (lifted from matf_cma_jpm_2026/run_bootstrap.py) ──
SR_MEANS = np.array([0.30, 0.30, 0.25, 0.30, 0.10, 0.10, 0.50, 0.30, 0.00])
SR_STD = 0.10


def correlation_from_covariance(cov: np.ndarray) -> np.ndarray:
    s = np.sqrt(np.diag(cov))
    s = np.where(s > 0, s, 1.0)
    return cov / np.outer(s, s)


def identity_woodbury(lam, Sig_F, bF):
    """(identity, ceiling) via the rank-safe Woodbury form."""
    SinvL = np.linalg.solve(Sig_F, lam)
    ceil = float(lam @ SinvL)
    corr = float(SinvL @ np.linalg.solve(np.linalg.inv(Sig_F) + bF, SinvL))
    return ceil - corr, ceil


def run(n_boot: int = 500, mean_block: float = 12.0, min_block: int = 3,
        seed: int = 42, verbose: bool = True):
    inputs = load_production_inputs()
    F_df, E_df = build_bootstrap_panels(inputs)
    F, E = F_df.values, E_df.values
    fpy = fpy_per_asset()
    beta, Sig_F_prod = inputs['beta'], inputs['Sig_F']
    T, M = F.shape
    rng = np.random.default_rng(seed)
    Sigma_SR = SR_STD ** 2 * correlation_from_covariance(Sig_F_prod)

    # full-sample equal-weighted baselines (consistent with bootstrap)
    Sig_F_full = 12.0 * np.cov(F, rowvar=False)
    D_full = np.array([np.nanvar(E[:, j], ddof=1) for j in range(E.shape[1])]) * fpy
    bF_full = factor_information_matrix(beta, D_full)
    lam_A0 = SR_MEANS * np.sqrt(np.diag(Sig_F_full))
    lam_B0 = 12.0 * F.mean(axis=0)
    idA0, _ = identity_woodbury(lam_A0, Sig_F_full, bF_full)
    idB0, _ = identity_woodbury(lam_B0, Sig_F_full, bF_full)

    idA = np.empty(n_boot)
    idB = np.empty(n_boot)
    for b in range(n_boot):
        ix = stationary_block_indices(T, mean_block, rng, min_block=min_block)
        Fb, Eb = F[ix], E[ix]
        Sig_Fb = 12.0 * np.cov(Fb, rowvar=False)
        Db = np.array([np.nanvar(Eb[:, j], ddof=1) for j in range(Eb.shape[1])]) * fpy
        bFb = factor_information_matrix(beta, Db)
        lamA = rng.multivariate_normal(SR_MEANS, Sigma_SR) * np.sqrt(np.diag(Sig_Fb))
        idA[b], _ = identity_woodbury(lamA, Sig_Fb, bFb)
        lamB = 12.0 * Fb.mean(axis=0)
        idB[b], _ = identity_woodbury(lamB, Sig_Fb, bFb)
        if verbose and (b + 1) % max(1, n_boot // 5) == 0:
            print(f"  [boot] {b + 1}/{n_boot}")
    return dict(idA=idA, idB=idB, idA0=idA0, idB0=idB0,
                n_boot=n_boot, seed=seed, T=T)


def report(res: dict, prod_sr2: float = 0.261306):
    prod = np.sqrt(prod_sr2)
    rows = []
    for name, sr, b0 in [('A (ex ante)', np.sqrt(res['idA']), np.sqrt(res['idA0'])),
                         ('B (ex post)', np.sqrt(res['idB']), np.sqrt(res['idB0']))]:
        rows.append((name, b0, sr.mean(), sr.std(ddof=1),
                     np.percentile(sr, 5), np.percentile(sr, 95),
                     (sr > 0.40).mean(), (sr > prod).mean()))
    hdr = f"{'config':<14}{'baseline':>9}{'mean':>7}{'sd':>7}{'p5':>7}{'p95':>7}{'P>0.40':>8}{'P>prod':>8}"
    print(hdr)
    for r in rows:
        print(f"{r[0]:<14}" + "".join(f"{x:>7.3f}" for x in r[1:6])
              + f"{r[6]:>8.1%}{r[7]:>8.1%}")
    return rows


def plot(res: dict, prod_sr2: float = 0.261306,
         outfile: str = os.path.join(FIG, "exhibit_bootstrap.png")):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family': 'serif', 'font.size': 10})
    NAVY, STEEL, ORANGE, GREY = '#1f3864', '#2E6DB4', '#E8943A', '#888888'
    srA, srB = np.sqrt(res['idA']), np.sqrt(res['idB'])
    baseA, baseB = np.sqrt(res['idA0']), np.sqrt(res['idB0'])
    prod = np.sqrt(prod_sr2)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.0), sharey=True)
    panels = [(axes[0], srA, baseA, STEEL, 'Ex-ante: $\\lambda_b$ from SR prior'),
              (axes[1], srB, baseB, ORANGE, 'Ex-post: realized $\\lambda_b$')]
    for ax, sr, base, col, lab in panels:
        ax.hist(sr, bins=34, color=col, alpha=0.85, density=True, zorder=2)
        ax.axvline(prod, color='black', ls='--', lw=1.6, zorder=4,
                   label=f'Production point = {prod:.2f}')
        ax.axvline(base, color=NAVY, ls='-', lw=1.6, zorder=4,
                   label=f'Full-sample baseline = {base:.2f}')
        ax.axvline(sr.mean(), color=col, ls='-', lw=1.4, alpha=0.7, zorder=3,
                   label=f'Bootstrap mean = {sr.mean():.2f}')
        for q in (5, 95):
            ax.axvline(np.percentile(sr, q), color=GREY, ls=':', lw=1.1, zorder=3)
        ax.set_xlabel('Achievable Sharpe ratio (annualized)')
        ax.legend(fontsize=7.6, loc='upper right', framealpha=0.95)
        ax.grid(axis='y', alpha=0.2)
        ax.set_xlim(0.30, 1.15)
        ax.text(0.03, 0.96, lab, transform=ax.transAxes, fontsize=9,
                va='top', style='italic')
    axes[0].set_ylabel('Density')
    fig.tight_layout()
    fig.savefig(outfile, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f"[boot] figure written to {outfile}")


if __name__ == '__main__':
    res = run()
    report(res)
    plot(res)
