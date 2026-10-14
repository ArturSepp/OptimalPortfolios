"""Independent GLS reference, equivariance and unavailable-inference contracts."""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from scipy.stats import t

from optimalportfolios.matf.cma_diagnostics import CmaDiagnosticInputs, CmaDiagnosticSpec
from optimalportfolios.matf.cma_uncertainty import compute_conditional_gls_intervals


def _inputs():
    """Use a heteroskedastic two-factor cross section with a nonzero pricing residual."""
    b = pd.DataFrame([[1., .1], [.2, 1.], [.6, .2], [.4, 1.2], [1.4, .4], [-.2, .6]],
                     index=list('abcdef'), columns=['Equity', 'Rates'])
    return CmaDiagnosticInputs(b, pd.Series([.01, .02, .03, .015, .04, .025], index=b.index),
                               pd.Series([.04, .02, .03, .017, .07, -.002], index=b.index))


def test_intervals_match_independent_weighted_qr():
    """Check the SVD estimator against a weighted QR fit and triangular covariance solve."""
    inputs = _inputs()
    before = inputs.excess_cma.copy()
    actual = compute_conditional_gls_intervals(inputs, confidence_level=.9)
    weights = 1 / np.sqrt(inputs.residual_variances.to_numpy())
    design = inputs.betas.to_numpy() * weights[:, None]
    target = inputs.excess_cma.to_numpy() * weights
    q, r = np.linalg.qr(design, mode='reduced')
    beta = np.linalg.solve(r, q.T @ target)
    residual = target - design @ beta
    df = len(target) - len(beta)
    scale = residual @ residual / df
    inverse_r = np.linalg.solve(r, np.eye(len(beta)))
    covariance = scale * inverse_r @ inverse_r.T
    se = np.sqrt(np.diag(covariance))
    width = t.ppf(.95, df) * se
    np.testing.assert_allclose(actual.factors.estimate, beta, atol=1e-14)
    np.testing.assert_allclose(actual.covariance, covariance, atol=1e-14)
    np.testing.assert_allclose(actual.factors.standard_error, se, atol=1e-14)
    np.testing.assert_allclose(actual.factors[['lower', 'upper']],
                               np.column_stack((beta-width, beta+width)), atol=1e-14)
    assert actual.summary['residual_df'] == df == 4
    assert actual.summary['scale_squared'] == pytest.approx(scale)
    assert actual.summary['scope'] == 'pointwise'
    pd.testing.assert_series_equal(inputs.excess_cma, before)


def test_covariance_matches_projector_sandwich_and_scale_invariance():
    """The covariance is A (tau^2 D) A.T and is invariant to rescaling weighting D."""
    inputs = _inputs()
    result = compute_conditional_gls_intervals(inputs)
    b, d = inputs.betas.to_numpy(), inputs.residual_variances.to_numpy()
    a = np.linalg.solve(b.T@(b/d[:, None]), b.T/d)
    expected = a @ np.diag(d*result.summary['scale_squared']) @ a.T
    np.testing.assert_allclose(result.covariance, expected, atol=1e-14)
    scaled = compute_conditional_gls_intervals(replace(
        inputs, residual_variances=19*inputs.residual_variances))
    np.testing.assert_allclose(scaled.covariance, result.covariance, atol=1e-14)
    np.testing.assert_allclose(scaled.factors[['lower', 'upper']],
                               result.factors[['lower', 'upper']])
    moved = compute_conditional_gls_intervals(replace(
        inputs, excess_cma=inputs.excess_cma+inputs.betas@np.array([.007, -.003])))
    np.testing.assert_allclose(moved.covariance, result.covariance, atol=1e-14)
    np.testing.assert_allclose(moved.factors.estimate-result.factors.estimate, [.007, -.003])


@pytest.mark.parametrize('case,expected', [
    ('rank', 'unavailable_rank_deficient'), ('saturated', 'unavailable_residual_df'),
    ('perfect', 'unavailable_zero_residual_scale'), ('cutoff', 'unavailable_rank_deficient')])
def test_unavailable_intervals_do_not_claim_certainty(case, expected):
    """A pseudoinverse or mechanically zero residual does not create finite confidence bounds."""
    inputs, spec = _inputs(), CmaDiagnosticSpec()
    if case == 'rank':
        b = inputs.betas.copy()
        b['Rates'] = b.Equity
        inputs = replace(inputs, betas=b)
    elif case == 'saturated':
        inputs = replace(inputs, betas=inputs.betas.iloc[:2],
                         residual_variances=inputs.residual_variances.iloc[:2],
                         excess_cma=inputs.excess_cma.iloc[:2])
    elif case == 'perfect':
        inputs = replace(inputs, excess_cma=inputs.betas@np.array([.03, .01]))
    else:
        spec = CmaDiagnosticSpec(rcond=.9)
    result = compute_conditional_gls_intervals(inputs, spec=spec)
    assert result.summary['status'] == expected
    assert result.factors[['standard_error', 'lower', 'upper']].isna().all().all()
    assert result.covariance.isna().all().all()


@pytest.mark.parametrize('confidence', [0., 1., float('nan'), -.5, True, [.95]])
def test_invalid_confidence_is_rejected(confidence):
    """Reject invalid confidence levels rather than silently reinterpret percentages."""
    with pytest.raises(ValueError, match='confidence_level'):
        compute_conditional_gls_intervals(_inputs(), confidence_level=confidence)


def test_asset_alignment_is_exact():
    """Covariance weighting and forecasts must use the same labelled rows."""
    inputs = _inputs()
    with pytest.raises(ValueError, match='exact unique index'):
        compute_conditional_gls_intervals(replace(inputs, excess_cma=inputs.excess_cma.iloc[::-1]))


def test_invalid_input_and_spec_types_are_rejected():
    """Public inference requires labelled inputs and the declared numerical policy type."""
    with pytest.raises(TypeError, match='inputs'):
        compute_conditional_gls_intervals(None)
    with pytest.raises(TypeError, match='spec'):
        compute_conditional_gls_intervals(_inputs(), spec={})
    with pytest.raises(ValueError, match='betas'):
        compute_conditional_gls_intervals(replace(_inputs(), betas=np.eye(2)))


def test_interval_overflow_is_explicitly_unavailable():
    """Finite but incompatible units cannot escape as infinite confidence bounds."""
    data = _inputs()
    extreme = replace(data, betas=data.betas*1e-150, excess_cma=data.excess_cma*1e50)
    with np.errstate(over='ignore', invalid='ignore'):
        with pytest.raises(ValueError, match='overflow'):
            compute_conditional_gls_intervals(extreme)
