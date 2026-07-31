"""
Parity harness for the 2026q2 frozen cut: the numbers the manuscript rests on.

Asserts three families on snapshot 2026q2, so any drift in the shared data
layer or in the replication mathematics fails loudly before an exhibit is
built:

  reference values (3 decimals)
      ceiling      lambda' Sigma_F^-1 lambda                       = 0.614
      attainable   lambda' (Sigma_F + beta_F^-1)^-1 lambda         = 0.238
      FPIR         attainable / ceiling                            = 0.387
      raw claim    sum_i (w_i alpha_i / sigma_eps,i)^2             = 1.398
      GLS claim    a' D^-1 a of the admitted-alpha vector          = 0.625
      full vector  SR2_alpha(mu) of the published excess CMAs      = 0.456
      solo premium-like shares  PE .84 / PC .77 / ILS .61 / HF .29 / Gold .29
      Cap 3 theta  at kappa = 1.00 / 0.50 / 0.25                   = .41/.29/.21
      Consensus    SR2_alpha on the 17-sleeve published+converted subset = 0.091

  identities (tolerance 1e-10, decimal p.a.)
      factor_excess_cma == betas @ factor_premia + equity_regional_addon
      scenario premia   == de-compounded calendar-year factor returns / 5
      audit-table IRs square-sum to the raw claim
      waterfall components sum to the published excess CMA per asset

  loader and benchmark contracts
      a tampered snapshot file raises out of verify_manifest
      Balanced with Alts sums to 1 with class sums 0.28 / 0.42 / 0.30
      Asia ex-Japan 4.52% and EM ex-Asia 0.88% (the D8-correct pair)

Units: every asserted quantity is decimal per annum except the dimensionless
squared Sharpe ratios, shares, and theta. Main entry point: pytest.

Does not belong here: exhibit rendering, optimizer solves (they carry solver
tolerances of their own and are checked in the exhibit scripts), and any
write into cma_data/snapshots (snapshots are immutable).
"""
# packages
import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
# project: paper reproduction package matf_cma_jpm_2026
from local_path import load_cma_data, get_cma_data_path
from governed_cma_projection import (SNAPSHOT,
                                     KAPPA_GRID,
                                     load_paper_inputs,
                                     compute_sharpe_accounting,
                                     compute_gls_decomposition,
                                     compute_solo_premium_like_shares,
                                     build_admission_audit,
                                     project_onto_governed_set)
from consensus_decomposition import decompose_on_subset

_cma_data = load_cma_data()

IDENTITY_TOL = 1e-10        # decimal p.a.; 1e-10 = 1e-6 bp
SCENARIO_HORIZON = 5.0      # years, the de-compounding divisor of eq:decompound


@pytest.fixture(scope='module')
def inputs():
    """the pinned frozen cut, manifest-verified on load."""
    return load_paper_inputs(snapshot=SNAPSHOT)


# --------------------------------------------------------------------------
# reference values (Guardrail 1)
# --------------------------------------------------------------------------

def test_sharpe_accounting_reference_values(inputs):
    accounting = compute_sharpe_accounting(inputs=inputs)
    assert round(accounting['ceiling'], 3) == 0.614
    assert round(accounting['attainable'], 3) == 0.238
    assert round(accounting['fpir'], 3) == 0.387


def test_admitted_claim_reference_values(inputs):
    assets = inputs.assets
    admitted = assets['w_paper'] * assets['alpha']
    raw = float((admitted / assets['resid_vol']).pow(2).sum())
    _, a_adm, _ = compute_gls_decomposition(mu_excess=admitted, inputs=inputs)
    gls = float(a_adm @ (a_adm / assets['resid_vol'] ** 2))
    assert round(raw, 3) == 1.398
    assert round(gls, 3) == 0.625


def test_full_vector_consistency_measure(inputs):
    assets = inputs.assets
    mu_excess = assets['factor_excess_cma'] + assets['w_paper'] * assets['alpha']
    _, _, sr2_full = compute_gls_decomposition(mu_excess=mu_excess, inputs=inputs)
    assert round(sr2_full, 3) == 0.456


def test_solo_premium_like_shares(inputs):
    shares = compute_solo_premium_like_shares(inputs=inputs)
    expected = {'MP503001 Index': 0.84,      # Private Equity
                'MP503008 Index': 0.77,      # Private Credit
                'EHFI804 Index': 0.61,       # Insurance-Linked
                'HFRIFWI Index': 0.29,       # Hedge Funds
                'BCOMGCTR Index': 0.29}      # Gold
    for ticker, share in expected.items():
        assert round(float(shares[ticker]), 2) == share, ticker


def test_cap3_projection_grid(inputs):
    expected_theta = {1.00: 0.41, 0.50: 0.29, 0.25: 0.21}
    expected_skill_share = {1.00: 0.50, 0.50: 0.33, 0.25: 0.20}
    assert set(KAPPA_GRID) == set(expected_theta)
    for kappa in KAPPA_GRID:
        proj = project_onto_governed_set(inputs=inputs, kappa=kappa)
        assert round(proj.attrs['theta'], 2) == expected_theta[kappa], kappa
        assert round(proj.attrs['rho_after'], 2) == expected_skill_share[kappa], kappa
        assert round(proj.attrs['rho_before'], 2) == 0.85


def test_consensus_claimed_sr2_alpha(inputs):
    """0.091 on the 17-sleeve published+converted subset (held-at-MATF excluded)."""
    consensus = _cma_data.build_consensus_provider()
    tickers = consensus.index[consensus['source'] != 'held_at_matf']
    assert len(tickers) == 17
    rf = float(inputs.assets['rf_rate'].iloc[0])
    mu_cons = (consensus.loc[tickers, 'total_cma_arith'] - rf).rename('consensus_excess')
    decomposition = decompose_on_subset(mu_excess=mu_cons,
                                        betas=inputs.betas.loc[tickers],
                                        resid_vol=inputs.assets.loc[tickers, 'resid_vol'])
    assert round(decomposition.attrs['sr2_alpha'], 3) == 0.091


# --------------------------------------------------------------------------
# identities (Guardrail 2)
# --------------------------------------------------------------------------

def test_addon_is_inside_factor_excess_cma(inputs):
    """factor_excess_cma == betas @ lambda + equity_regional_addon.

    The identity behind the Stage J0b correction: adding the add-on on top of
    factor_excess_cma double-counts the regional blend.
    """
    assets = inputs.assets
    implied = inputs.betas.values @ inputs.factor_premia.values + assets['equity_regional_addon']
    assert float((implied - assets['factor_excess_cma']).abs().max()) < IDENTITY_TOL


def test_scenario_premia_are_decompounded_annual_returns(inputs):
    """stress and upside columns are calendar-2022 / 2023 factor returns divided by 5.

    Needs factor_navs.csv, which is not redistributed publicly (licensed factor
    histories). Skips rather than fails on a public checkout; every other
    identity in this file runs on the committed config files alone.
    """
    if not inputs.has_panel('factor_navs'):
        pytest.skip("factor_navs.csv not in this checkout (not redistributed publicly)")
    annual = inputs.factor_navs.resample('YE').last().pct_change().dropna()
    annual.index = annual.index.year
    scenarios = inputs.factor_premia_scenarios
    stress_gap = (scenarios['stress'] - annual.loc[2022] / SCENARIO_HORIZON).abs().max()
    upside_gap = (scenarios['upside'] - annual.loc[2023] / SCENARIO_HORIZON).abs().max()
    assert float(stress_gap) < IDENTITY_TOL
    assert float(upside_gap) < IDENTITY_TOL


def test_scenario_cmas_are_additive(inputs):
    """scenario excess CMA == base + betas @ bump, exactly."""
    assets = inputs.assets
    base = assets['factor_excess_cma'] + assets['w_paper'] * assets['alpha']
    for scenario in ('stress', 'upside'):
        bump = inputs.factor_premia_scenarios[scenario]
        direct = base + inputs.betas.values @ bump.values
        rebuilt = (inputs.betas.values @ (inputs.factor_premia + bump).values
                   + assets['equity_regional_addon'] + assets['w_paper'] * assets['alpha'])
        assert float((direct - rebuilt).abs().max()) < IDENTITY_TOL, scenario


def test_audit_irs_square_sum_to_raw_claim(inputs):
    """the printed per-sleeve IRs reconstruct the portfolio claim exactly, not approximately."""
    assets = inputs.assets
    raw = float((assets['w_paper'] * assets['alpha'] / assets['resid_vol']).pow(2).sum())
    audit = build_admission_audit(inputs=inputs)
    assert float(abs(audit['ir'].pow(2).sum() - raw)) < IDENTITY_TOL
    assert round(raw, 3) == 1.398


def test_waterfall_components_sum_to_published_excess_cma(inputs):
    """factor-implied + blend add-on + admitted alpha + gross add-on == published excess CMA."""
    assets = inputs.assets
    factor_implied = pd.Series(inputs.betas.values @ inputs.factor_premia.values, index=assets.index)
    blend = assets['equity_regional_addon']
    admitted = assets['w_paper'] * assets['alpha']
    gross_addon = pd.Series(0.0, index=assets.index)     # zero on this universe
    published = assets['factor_excess_cma'] + admitted
    stacked = factor_implied + blend + admitted + gross_addon
    assert float((stacked - published).abs().max()) < IDENTITY_TOL


# --------------------------------------------------------------------------
# loader and benchmark contracts
# --------------------------------------------------------------------------

def test_manifest_verification_rejects_a_tampered_file(tmp_path):
    """copy the snapshot, perturb one byte of one csv, expect the loader to raise."""
    source = get_cma_data_path() / 'snapshots' / SNAPSHOT
    target = tmp_path / SNAPSHOT
    target.mkdir(parents=True)
    for file_path in source.glob('*'):
        (target / file_path.name).write_bytes(file_path.read_bytes())
    tampered = target / 'assets.csv'
    tampered.write_text(tampered.read_text(encoding='utf-8') + '\n', encoding='utf-8')
    with pytest.raises(ValueError, match='hash mismatch'):
        _cma_data.load_snapshot(tag=SNAPSHOT, snapshots_path=tmp_path)


def test_snapshot_manifest_pins_the_july_sharpe_priors(inputs):
    """B10: the bootstrap prior mean vector equals the manifest matf_sharpe_ratios rows."""
    config = pd.DataFrame(inputs.manifest['prod_config_snapshot'])
    rows = config.loc[config['group'] == 'matf_sharpe_ratios']
    manifest_priors = pd.Series(
        {name.split('.', 1)[1]: float(value)
         for name, value in zip(rows['Unnamed: 0'], rows['value'])})
    expected = pd.Series({'Equity': 0.40, 'Rates': 0.25, 'Credit': 0.40, 'Carry': 0.25,
                          'Inflation': 0.15, 'Commodities': 0.15, 'Private Equity': 0.60,
                          'Rates Vol': 0.25, 'Fx': 0.00})
    pd.testing.assert_series_equal(manifest_priors[list(expected.index)], expected,
                                   check_names=False, atol=1e-12)


def test_benchmark_weights_are_d8_correct():
    weights = _cma_data.get_benchmark_weights(mandate='Balanced with Alts')
    assert abs(float(weights.sum()) - 1.0) < IDENTITY_TOL
    class_sums = weights.groupby(pd.Series(_cma_data.ASSET_CLASSES)).sum()
    assert abs(float(class_sums['Bonds']) - 0.28) < IDENTITY_TOL
    assert abs(float(class_sums['Equities']) - 0.42) < IDENTITY_TOL
    assert abs(float(class_sums['Alternatives']) - 0.30) < IDENTITY_TOL
    # D8: the R2 exhibit build transposed this pair
    assert round(1e2 * float(weights['M1APJ Index']), 2) == 4.52       # Asia ex-Japan
    assert round(1e2 * float(weights['M1EFZ Index']), 2) == 0.88       # EM ex-Asia


def test_all_eight_benchmarks_are_fully_invested():
    benchmarks = _cma_data.get_all_benchmarks()
    assert list(benchmarks.columns) == _cma_data.MANDATES
    assert float((benchmarks.sum(axis=0) - 1.0).abs().max()) < IDENTITY_TOL


def test_benchmarks_reindex_onto_the_snapshot_asset_order(inputs):
    """every mandate benchmark aligns to the snapshot universe with no NaN (B11)."""
    benchmarks = _cma_data.get_all_benchmarks().reindex(inputs.assets.index)
    assert not benchmarks.isna().any().any()
