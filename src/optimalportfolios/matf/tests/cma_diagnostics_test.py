"""Independent forecast accounting, applicability and input contract checks."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from optimalportfolios.matf.cma_diagnostics import (
    AdmissionPolicy, CmaDiagnosticInputs, CmaDiagnosticSpec, compute_cma_diagnostics,
)


@pytest.fixture
def inputs():
    """A known full-rank factor model with two distinct forecast adjustments."""
    ids, factors = pd.Index(['a', 'b', 'c', 'd']), pd.Index(['equity', 'rates'])
    b = pd.DataFrame([[1., 0.], [0., 1.], [.5, .6], [1., -.2]], index=ids, columns=factors)
    d = pd.Series([.01, .02, .03, .04], index=ids)
    f = pd.DataFrame([[.04, .005], [.005, .01]], index=factors, columns=factors)
    lam = pd.Series([.05, .02], index=factors)
    channels = pd.DataFrame({'regional': [.002, 0., -.001, .001],
                             'admitted_alpha': [.01, .015, -.01, .003]}, index=ids)
    m = b@lam+channels.sum(axis=1)
    return CmaDiagnosticInputs(b, d, m, f, lam, channels,
                               scenarios=pd.DataFrame({'stress': m-.02, 'upside': m+.01}),
                               asset_metadata=pd.DataFrame({'name': ids}, index=ids),
                               provenance={'reference_ccy': 'USD', 'as_of': '2026-09-30'})


def test_asset_space_and_independent_whitened_qr(inputs):
    """Asset covariance solves and QR reproduce the returned accounting identities."""
    result = compute_cma_diagnostics(inputs, policy=AdmissionPolicy())
    b, d = inputs.betas.to_numpy(), inputs.residual_variances.to_numpy()
    f = inputs.factor_covariance
    covariance = b@f.to_numpy()@b.T+np.diag(d)
    q, _ = np.linalg.qr(b/np.sqrt(d)[:, None])
    residual = np.sqrt(d)*((np.eye(4)-q@q.T)@(inputs.excess_cma/np.sqrt(d)))
    np.testing.assert_allclose(result.assets.residual, residual, atol=1e-15)
    direct = inputs.excess_cma@np.linalg.solve(covariance, inputs.excess_cma)
    assert result.summary['total_sharpe2'] == pytest.approx(direct)
    assert result.summary['total_sharpe2'] == pytest.approx(
        result.summary['implied_systematic_sharpe2']+result.summary['residual_sharpe2'])
    assert result.validation.status.eq('pass').all()
    assert result.channels.residual_attribution.sum() == pytest.approx(
        result.summary['residual_sharpe2'])
    assert result.assets.residual_share.sum() == pytest.approx(1.)
    assert result.summary['admission_score'] > result.summary['admission_residual_score']
    assert result.summary['policy_name'] == 'paper_illustrative'
    assert result.factors.status.eq('identified').all()
    assert len(result.scenarios.columns) == 2
    np.testing.assert_allclose(result.assets.excess_cma, inputs.excess_cma)


def test_owned_inputs_and_metadata_collision(inputs):
    """Input construction owns its pandas values and rejects display-field collisions."""
    original = inputs.excess_cma.copy()
    copied = replace(inputs)
    copied.excess_cma.iloc[0] = 100.
    pd.testing.assert_series_equal(inputs.excess_cma, original)
    with pytest.raises(ValueError, match='shadows'):
        compute_cma_diagnostics(replace(inputs, asset_metadata=pd.DataFrame(
            {'residual': 1.}, index=inputs.betas.index)))


def test_subsets_are_reprojected_and_saturated_status(inputs):
    """A two-asset saturated projection differs from slicing the full residual."""
    full = compute_cma_diagnostics(inputs)
    ids = inputs.betas.index[:2]
    subset = CmaDiagnosticInputs(inputs.betas.loc[ids], inputs.residual_variances.loc[ids],
                                 inputs.excess_cma.loc[ids], inputs.factor_covariance,
                                 inputs.factor_premia)
    result = compute_cma_diagnostics(subset)
    assert result.summary['identification_status'] == 'saturated'
    assert result.summary['residual_dimension'] == 0
    assert result.summary['top1_residual_share'] is None
    assert np.linalg.norm(full.assets.loc[ids, 'residual']) > .001
    assert np.linalg.norm(result.assets.residual) < 1e-14
    one = CmaDiagnosticInputs(inputs.betas.iloc[:1], inputs.residual_variances.iloc[:1],
                              inputs.excess_cma.iloc[:1])
    result = compute_cma_diagnostics(one)
    assert result.factors.status.eq('minimum_norm').all()
    assert result.factors.identification_scale.isna().all()


def test_rank_deficient_and_deliberate_truncation(inputs):
    """Truncation is recorded and cannot claim exact original-model accounting."""
    b = inputs.betas.copy()
    b.iloc[:, 1] = b.iloc[:, 0]
    plain = CmaDiagnosticInputs(b, inputs.residual_variances, inputs.excess_cma)
    result = compute_cma_diagnostics(plain)
    assert result.summary['rank'] == 1
    assert result.summary['condition_number'] is None
    assert result.summary['identification_status'] == 'rank_deficient'
    truncated = compute_cma_diagnostics(inputs, spec=CmaDiagnosticSpec(rcond=.9))
    assert truncated.summary['rank'] == 1
    assert truncated.summary['accounting_status'] == 'fail'
    assert truncated.validation.set_index('check').loc['sharpe_accounting', 'status'] == 'fail'


def test_optional_moments_zero_premia_and_singular_factor_covariance(inputs):
    """Absent inputs and undefined inverse metrics remain explicitly unavailable."""
    plain = CmaDiagnosticInputs(inputs.betas, inputs.residual_variances, inputs.excess_cma)
    result = compute_cma_diagnostics(plain, policy=AdmissionPolicy())
    assert result.summary['total_sharpe2'] is None
    assert result.summary['availability']['admission'] == 'admitted_alpha_not_supplied'
    assert result.policy_checks.status.eq('unavailable').all()
    result = compute_cma_diagnostics(replace(plain, factor_covariance=inputs.factor_covariance))
    assert result.summary['availability']['factor_access'] == 'factor_premia_not_supplied'
    zero = replace(inputs, factor_premia=inputs.factor_premia*0.,
                   excess_cma=inputs.channels.sum(axis=1))
    result = compute_cma_diagnostics(zero, policy=AdmissionPolicy())
    assert result.summary['fpir'] is None
    assert result.summary['admission_budget_ratio'] is None
    assert result.summary['availability']['fpir'] == 'zero_factor_ceiling'
    f = inputs.factor_covariance.copy()
    f.iloc[:, :] = [[.01, .01], [.01, .01]]
    result = compute_cma_diagnostics(replace(inputs, factor_covariance=f))
    assert result.summary['factor_ceiling2'] is None
    assert result.summary['total_sharpe2'] is not None
    assert result.summary['availability']['factor_access'] == (
        'factor_covariance_not_positive_definite')


def test_policy_positive_domain_missing_admission_and_zero_admission(inputs):
    """A nonpositive denominator never becomes a spurious Cap 2 pass."""
    means = inputs.excess_cma.copy()
    means.iloc[0:2] = [0., -.01]
    plain = replace(inputs, channels=None, excess_cma=means,
                    admitted_alpha=inputs.channels.admitted_alpha)
    result = compute_cma_diagnostics(plain, policy=AdmissionPolicy(max_positive_ir=.001,
                                                                 max_admission_share=.01,
                                                                 kappa=0.))
    table = result.policy_checks.set_index(['asset', 'check'])
    assert table.loc[('a', 'cap2_admission_share'), 'status'] == 'not_applicable'
    assert table.loc[('b', 'cap2_admission_share'), 'status'] == 'not_applicable'
    assert table.loc[('c', 'cap1_positive_ir'), 'status'] == 'not_applicable'
    assert table.loc[('d', 'cap2_admission_share'), 'status'] == 'breach'
    assert table.loc[('__universe__', 'cap3_admission_budget'), 'status'] == 'breach'
    result = compute_cma_diagnostics(replace(plain, admitted_alpha=means*0.),
                                     policy=AdmissionPolicy())
    assert result.summary['admission_in_span_share'] is None
    assert result.policy_checks.iloc[-1].status == 'pass'


def test_production_covariance_is_compared_without_substitution(inputs):
    """A different production covariance qualifies accounting but does not replace D."""
    b = inputs.betas
    covariance = b@inputs.factor_covariance@b.T+np.diag(inputs.residual_variances)
    expected = compute_cma_diagnostics(inputs).summary['total_sharpe2']
    result = compute_cma_diagnostics(replace(inputs, production_covariance=covariance*1.2))
    assert result.summary['total_sharpe2'] == expected
    assert result.validation.set_index('check').loc[
        'production_covariance_reconciliation', 'status'] == 'fail'


@pytest.mark.parametrize('field,value,match', [
    ('excess_cma', np.ones(4), 'labelled'),
    ('betas', np.ones((4, 2)), 'labelled'),
    ('excess_cma', pd.Series([1., 2., 3., 4.], index=['b', 'a', 'c', 'd']), 'exact'),
    ('excess_cma', pd.Series([1., 2., np.nan, 4.], index=list('abcd')), 'finite'),
    ('excess_cma', pd.Series([1j, 2., 3., 4.], index=list('abcd')), 'finite'),
    ('residual_variances', pd.Series([0., .1, .2, .3], index=list('abcd')), 'positive'),
    ('residual_variances', pd.Series(['.1', '.2', '.3', '.4'], index=list('abcd')), 'finite'),
    ('channels', pd.DataFrame(np.ones((4, 2)), index=list('abcd'), columns=['x', 'x']), 'unique'),
    ('channels', pd.DataFrame({'wrong': 1.}, index=list('abcd')), 'reconcile'),
    ('admitted_alpha', pd.Series(1., index=list('abcd')), 'differs'),
])
def test_invalid_inputs_fail_before_reporting(inputs, field, value, match):
    """Invalid inputs cannot be repaired or silently reordered by the audit."""
    with pytest.raises(ValueError, match=match):
        compute_cma_diagnostics(replace(inputs, **{field: value}))


def test_invalid_covariance_axes_and_definiteness(inputs):
    """Covariance axes, symmetry and PSD are explicit contracts."""
    f = inputs.factor_covariance
    for bad, match in [(f.rename(columns={'rates': 'other'}), 'columns'),
                       (f+pd.DataFrame([[0., .01], [0., 0.]], index=f.index, columns=f.columns),
                        'symmetric'), (-f, 'semidefinite')]:
        with pytest.raises(ValueError, match=match):
            compute_cma_diagnostics(replace(inputs, factor_covariance=bad))
    ids = inputs.betas.index
    bad = pd.DataFrame(np.eye(4), index=ids, columns=list('abce'))
    with pytest.raises(ValueError, match='columns'):
        compute_cma_diagnostics(replace(inputs, production_covariance=bad))
    with pytest.raises(ValueError, match='require calibrated'):
        compute_cma_diagnostics(replace(inputs, factor_premia=None))


@pytest.mark.parametrize('kwargs', [{'atol': -1.}, {'rtol': float('inf')},
                                    {'rcond': -1.}, {'rcond': 1.}, {'rcond': float('nan')}])
def test_bad_numerical_spec(kwargs):
    """A numerical contract cannot accept negative or nonfinite tolerances."""
    with pytest.raises(ValueError):
        CmaDiagnosticSpec(**kwargs)


@pytest.mark.parametrize('kwargs', [{'name': ''}, {'max_positive_ir': -1.},
                                    {'max_admission_share': float('nan')}, {'kappa': -1.}])
def test_bad_policy(kwargs):
    """Reject malformed policy profiles independently of forecast data."""
    with pytest.raises(ValueError):
        AdmissionPolicy(**kwargs)


def test_api_type_guards(inputs):
    """Wrong API-level types fail with actionable messages."""
    with pytest.raises(TypeError, match='inputs'):
        compute_cma_diagnostics(None)
    with pytest.raises(TypeError, match='spec'):
        compute_cma_diagnostics(inputs, spec={})
    with pytest.raises(TypeError, match='policy'):
        compute_cma_diagnostics(inputs, policy={})
