"""Independent null-space references and invalid-input contracts for universe diagnostics."""
import numpy as np
import pandas as pd
import pytest
from scipy.linalg import null_space

from optimalportfolios.matf import compute_residual_opportunity


def fixture():
    """Return a nontrivial heteroskedastic, correlated five-asset example."""
    b = np.array([[1., 0.], [1., .2], [.3, 1.], [0., 1.], [.2, -.1]])
    d = np.array([.01, .03, .02, .04, .05])
    a = np.array([.01, -.02, .03, .02, .04])
    v = .0001 * (np.eye(5) + .2*np.ones((5, 5)))
    return b, d, a, v


def test_nullspace_capacity_floor_and_spectral_breadth():
    """Compare with a SciPy null-space basis, independently of OP's GLS projector."""
    b, d, a, v = fixture()
    controls = np.ones((5, 1))
    z = null_space((np.column_stack([b, controls]) / np.sqrt(d)[:, None]).T)
    r = .7*np.eye(5) + .3*np.ones((5, 5))
    result = compute_residual_opportunity(b, d, a, controls=controls,
        alpha_covariance=v, residual_correlation=r)
    coordinates = z.T @ (a/np.sqrt(d))
    expected_floor = np.trace(z.T @ (v/np.sqrt(np.outer(d, d))) @ z)
    spectrum = np.linalg.eigvalsh(z.T @ r @ z)
    assert result.summary['capacity'] == pytest.approx(coordinates@coordinates)
    assert result.summary['noise_floor'] == pytest.approx(expected_floor)
    assert result.summary['excess_capacity'] == pytest.approx(
        coordinates@coordinates-expected_floor)
    assert result.summary['breadth'] == pytest.approx(spectrum.sum()**2/(spectrum@spectrum))
    np.testing.assert_allclose(result.neutral_projector, z@z.T, atol=2e-14)
    np.testing.assert_allclose(b.T @ (result.orthogonal_alpha/d), 0., atol=2e-14)


def test_rank_units_labels_and_empty_residual_risk():
    """Controls may be redundant; unit changes preserve scores and null risk has zero breadth."""
    b, d, a, v = fixture()
    base = compute_residual_opportunity(b, d, a, alpha_covariance=v)
    duplicate = compute_residual_opportunity(np.column_stack([b, b[:, 0]]), d, a)
    assert duplicate.summary['capacity'] == pytest.approx(base.summary['capacity'])
    assert base.summary['breadth'] == pytest.approx(3.)
    scale = np.arange(1., 6.)
    scaled = compute_residual_opportunity(b*scale[:, None], d*scale**2, a*scale,
        alpha_covariance=v*np.outer(scale, scale))
    for key in ['capacity', 'noise_floor', 'breadth']:
        assert scaled.summary[key] == pytest.approx(base.summary[key])
    full = compute_residual_opportunity(np.eye(5), d, a)
    assert full.summary['capacity'] == full.summary['breadth'] == 0.
    zero = compute_residual_opportunity(np.ones((3, 1)), np.ones(3), np.ones(3),
        residual_correlation=np.ones((3, 3)))
    assert zero.summary['breadth'] == 0.
    ids = list('abcde')
    labelled = compute_residual_opportunity(pd.DataFrame(b, index=ids), pd.Series(d, index=ids),
        pd.Series(a, index=ids), controls=pd.DataFrame({'funding': np.ones(5)}, index=ids),
        alpha_covariance=pd.DataFrame(v, index=ids, columns=ids))
    assert labelled.summary['neutral_dimension'] == 2
    compute_residual_opportunity(b, d, a, alpha_covariance=pd.DataFrame(v))


def test_explicit_one_pass_repair():
    """An analytically diagonalizable indefinite correlation requires opt-in repair."""
    r = np.array([[1., .9, .9], [.9, 1., -.9], [.9, -.9, 1.]])
    with pytest.raises(ValueError, match='explicit repair'):
        compute_residual_opportunity(np.zeros((3, 1)), np.ones(3), np.ones(3),
                                     residual_correlation=r)
    result = compute_residual_opportunity(np.zeros((3, 1)), np.ones(3), np.ones(3),
        residual_correlation=r, repair_correlation=True)
    expected = np.array([[1., .5, .5], [.5, 1., -.5], [.5, -.5, 1.]])
    np.testing.assert_allclose(result.residual_correlation, expected, atol=2e-15)
    assert result.summary['sum_clipped'] == pytest.approx(.8)
    assert result.summary['n_clipped'] == 1
    assert result.summary['breadth'] == pytest.approx(2.)
    assert result.summary['breadth_raw'] != pytest.approx(2.)


@pytest.mark.parametrize('kwargs, message', [
    ({'repair_correlation': 1}, 'boolean'),
    ({'controls': np.ones((4, 1))}, 'one row'),
    ({'alpha_covariance': np.eye(4)}, 'symmetric asset'),
    ({'alpha_covariance': -np.eye(5)}, 'positive semidefinite'),
    ({'residual_correlation': 2*np.eye(5)}, 'unit diagonal'),
    ({'residual_correlation': np.ones((5, 5)) + np.triu(np.ones((5, 5)), 1)}, 'symmetric asset'),
    ({'residual_correlation': np.eye(5) + 2*(np.ones((5, 5))-np.eye(5))}, 'unit diagonal'),
])
def test_invalid_matrices(kwargs, message):
    """Reject malformed inputs instead of silently repairing or reordering them."""
    b, d, a, _ = fixture()
    with pytest.raises(ValueError, match=message):
        compute_residual_opportunity(b, d, a, **kwargs)


def test_reordered_covariance_rejected():
    """A correctly shaped matrix with incorrect labels cannot change the asset contract."""
    b, d, a, v = fixture()
    ids = list('abcde')
    with pytest.raises(ValueError, match='label/order'):
        compute_residual_opportunity(pd.DataFrame(b, index=ids), d, a,
            alpha_covariance=pd.DataFrame(v, index=ids, columns=ids[::-1]))
