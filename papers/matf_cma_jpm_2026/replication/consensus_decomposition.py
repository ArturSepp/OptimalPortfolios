"""
Decision One diagnostic on the Consensus provider: GLS decomposition of the Horizon 2025 vector.

Runs the paper's Equation (3) test on the Consensus column (Horizon Actuarial
2025 survey averages mapped to the paper universe by cma_data.consensus):
mu_e = total arithmetic CMA - r_f decomposed into lambda_gls, the residual
unattributed return a, and its squared Sharpe ratio a' D^-1 a, on the 2026q2
frozen cut. Held-at-MATF sleeves (Insurance-Linked) are EXCLUDED from the
decomposition: they would contribute zero deviation by construction and
dilute the test. The MATF vector is decomposed on the same 17-sleeve subset
for a like-for-like comparison.

Main entry point: run_consensus_report(). Units: decimal returns, annualized.

The MATF comparison vector is mu_x = factor_excess_cma + w_paper * alpha.
equity_regional_addon is NOT added on top: the identity
factor_excess_cma = beta @ lambda + equity_regional_addon holds on the
snapshot to 1e-13 bp, so adding it again double-counts the regional blend
(2-311 bp on the seven equity sleeves) and contaminates the MATF-side
gap_bp column, its lambda_gls column, and its same-subset SR2_alpha. The
Consensus-side numbers never depended on it. Corrected 2026-07-30, roadmap
Stage J0b; the identity is asserted in tests/test_snapshot_parity.py.

Does not belong here: the A-D provider vectors (pending providers.csv) and
the mandate optimizations (Decision One exhibit scripts).
"""
# packages
import numpy as np
import pandas as pd
from enum import Enum
# shared paper-data layer, imported by file location (no sys.path mutation)
from local_path import load_cma_data
from governed_cma_projection import load_paper_inputs, SNAPSHOT

_cma_data = load_cma_data()


def decompose_on_subset(mu_excess: pd.Series,
                        betas: pd.DataFrame,
                        resid_vol: pd.Series,
                        ) -> pd.DataFrame:
    """GLS decomposition of an excess-CMA vector on an arbitrary asset subset.

    Same mathematics as governed_cma_projection.compute_gls_decomposition,
    restricted to the rows of mu_excess (Shanken second pass with D^-1 weights).
    Returns per-sleeve residuals; lambda_gls and SR2 in DataFrame.attrs.
    """
    if not mu_excess.index.equals(betas.index):
        raise ValueError(f"index misaligned, got {list(mu_excess.index)!r}")
    b = betas.values
    d_inv = 1.0 / resid_vol.values ** 2
    beta_f = (b * d_inv[:, None]).T @ b
    lam_gls = np.linalg.solve(beta_f, (b * d_inv[:, None]).T @ mu_excess.values)
    a = mu_excess.values - b @ lam_gls
    out = pd.DataFrame({'mu_excess': mu_excess,
                        'factor_span': b @ lam_gls,
                        'unattributed': a,
                        'unattributed_ir': a / resid_vol.values})
    out.attrs['lambda_gls'] = pd.Series(lam_gls, index=betas.columns)
    out.attrs['sr2_alpha'] = float(a @ (d_inv * a))
    return out


def run_consensus_report(snapshot: str = SNAPSHOT) -> None:
    inputs = load_paper_inputs(snapshot=snapshot)
    consensus = _cma_data.build_consensus_provider()
    rf = float(inputs.assets['rf_rate'].iloc[0])

    # published + converted subset (drop held-at-MATF sleeves)
    mask = consensus['source'] != 'held_at_matf'
    tickers = consensus.index[mask]
    mu_cons = (consensus.loc[tickers, 'total_cma_arith'] - rf).rename('consensus_excess')

    # MATF published excess CMAs on the same subset, like-for-like
    assets = inputs.assets.loc[tickers]
    mu_matf = (assets['factor_excess_cma'] + assets['w_paper'] * assets['alpha']).rename('matf_excess')

    betas = inputs.betas.loc[tickers]
    resid_vol = assets['resid_vol']
    cons = decompose_on_subset(mu_excess=mu_cons, betas=betas, resid_vol=resid_vol)
    matf = decompose_on_subset(mu_excess=mu_matf, betas=betas, resid_vol=resid_vol)

    table = pd.DataFrame({'sleeve': assets['sleeve'],
                          'consensus_total': consensus.loc[tickers, 'total_cma_arith'],
                          'consensus_excess': mu_cons,
                          'matf_excess': mu_matf,
                          'gap_bp': 1e4 * (mu_cons - mu_matf),
                          'unattributed_bp': 1e4 * cons['unattributed'],
                          'unattributed_ir': cons['unattributed_ir'],
                          'source': consensus.loc[tickers, 'source']})
    pd.set_option('display.width', 200)
    print(f"snapshot {snapshot}, r_f = {rf:.4%}, subset = {len(tickers)} sleeves "
          f"(held at MATF: {list(consensus.index[~mask])})")
    print(table.round(4).to_string())
    print("\nlambda_gls (consensus vs matf):")
    print(pd.DataFrame({'consensus': cons.attrs['lambda_gls'],
                        'matf': matf.attrs['lambda_gls'],
                        'production': inputs.factor_premia}).round(4).to_string())
    print(f"\nSR2_alpha consensus = {cons.attrs['sr2_alpha']:.3f}  "
          f"| SR2_alpha matf (same subset) = {matf.attrs['sr2_alpha']:.3f}")


class LocalTests(str, Enum):
    CONSENSUS_REPORT = 'consensus_report'


def run_local_test(local_test: LocalTests) -> None:
    if local_test == LocalTests.CONSENSUS_REPORT:
        run_consensus_report()
    else:
        raise NotImplementedError(f"{local_test}")


if __name__ == '__main__':
    run_local_test(local_test=LocalTests.CONSENSUS_REPORT)
