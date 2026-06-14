"""
tests/test_reproducibility.py — headline numbers of the FAJ paper.

Run from the package root:  python -m pytest tests/ -v
Asserted against the published values (17-asset universe, USD, 31 Mar 2026,
production workbook global_saa_universe_data_cmas_usd_newbeta_2026q1.xlsx).
"""
import os
import sys
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

import identity_analytics as ia
from paper_inputs import load_production_inputs


@pytest.fixture(scope="module")
def inputs():
    return load_production_inputs()


def test_headline_levels(inputs):
    """Table 1: ceiling 0.586, identity 0.247 (42.1% of ceiling)."""
    ceil = ia.frictionless_ceiling(inputs['lam'], inputs['Sig_F'])
    ident = ia.matf_identity(inputs['lam'], inputs['Sig_F'],
                             inputs['beta'], inputs['D'])
    assert abs(ceil - 0.586336) < 1e-4
    assert abs(ident - 0.246987) < 1e-4
    assert abs(ident / ceil - 0.4213) < 1e-3


def test_idiosyncratic_alpha(inputs):
    """Figure 3 (waterfall): GLS-orthogonal SR2_alpha = 0.120, total SR 0.606."""
    a_gls = ia.gls_orthogonal_alpha(inputs['alpha'], inputs['beta'], inputs['D'])
    sr2a = ia.idiosyncratic_sr2(a_gls, inputs['D'])
    assert abs(sr2a - 0.1202) < 1e-3
    assert ia.consistency_residual(a_gls, inputs['beta'], inputs['D']) < 1e-8
    total = 0.246987 + sr2a
    assert abs(np.sqrt(total) - 0.606) < 2e-3


def test_factor_sharpe_coordinates(inputs):
    """Eq. (sr_coords): volatility-free form equals the identity exactly."""
    lam, Sig_F = inputs['lam'], inputs['Sig_F']
    sig = np.sqrt(np.diag(Sig_F))
    rho = Sig_F / np.outer(sig, sig)
    SR_F = np.where(sig > 0, lam / sig, 0.0)
    lhs = ia.identity_sr_coordinates(SR_F, rho, inputs['beta'], inputs['D'], sig)
    rhs = ia.matf_identity(lam, Sig_F, inputs['beta'], inputs['D'])
    assert abs(lhs - rhs) < 1e-10


def test_unit_invariance(inputs):
    """Identity invariant to factor unit rescaling F -> cF."""
    lam, Sig_F, beta, D = (inputs['lam'], inputs['Sig_F'],
                           inputs['beta'], inputs['D'])
    c = np.array([2.0, 0.5, 3.0, 1.0, 1.0, 0.25, 1.0, 5.0, 1.0])
    id1 = ia.matf_identity(lam, Sig_F, beta, D)
    id2 = ia.matf_identity(c * lam, np.outer(c, c) * Sig_F, beta / c, D)
    assert abs(id1 - id2) < 1e-10


def test_candidate_credit_instrument(inputs):
    """FPIR section: clean Credit instrument (4% resid vol) lifts FPIR 0.42 -> 0.46."""
    bnew = np.zeros(9)
    bnew[2] = 1.0  # Credit
    gain = ia.candidate_fpir_gain(inputs['lam'], inputs['Sig_F'],
                                  inputs['beta'], inputs['D'], bnew, 0.04 ** 2)
    assert abs(gain - 0.038) < 4e-3


def test_bootstrap_sampler_deterministic():
    """Politis-Romano sampler reproduces under seed 42 (qis-port guard)."""
    from run_bootstrap import stationary_block_indices
    rng = np.random.default_rng(42)
    ix1 = stationary_block_indices(300, 12.0, rng, min_block=3)
    rng = np.random.default_rng(42)
    ix2 = stationary_block_indices(300, 12.0, rng, min_block=3)
    assert np.array_equal(ix1, ix2)
    assert ix1.shape == (300,) and ix1.min() >= 0 and ix1.max() < 300


@pytest.mark.slow
def test_bootstrap_headline(inputs):
    """Table 2 (window Apr 2001-Mar 2026, T=300, matching MATF-CMA):
    Config A mean 0.509, P(SR>0.40)=92%; Config B mean 0.720.
    Full 500-draw run (~1 min); deselect with -m 'not slow' for quick CI."""
    from run_bootstrap import run
    res = run(n_boot=500, seed=42, verbose=False)
    srA, srB = np.sqrt(res['idA']), np.sqrt(res['idB'])
    assert abs(np.sqrt(res['idA0']) - 0.477) < 5e-3
    assert abs(srA.mean() - 0.509) < 5e-3
    assert abs((srA > 0.40).mean() - 0.92) < 0.01
    assert abs(srB.mean() - 0.720) < 5e-3


def test_longonly_tangency(inputs):
    """Table 1: long-only tangency SR 0.421 (requires cvxpy/CLARABEL)."""
    cvxpy = pytest.importorskip("cvxpy")
    a = np.load(os.path.join(ROOT, "data", "allfig17.npz"), allow_pickle=True)
    _, sr = ia.longonly_max_sharpe(a['mu'], a['Sigma'])
    assert abs(sr - 0.421) < 3e-3
