"""Independent rank-boundary and whitening contracts for the shared GLS geometry."""

import numpy as np
import pytest

from optimalportfolios.matf import sharpe_accounting as sa


def test_cutoff_boundary_uses_one_projection_for_means_and_holdings():
    """A direction below eps*max(shape) cannot survive only in the projector."""
    beta = np.zeros((10, 2))
    beta[0, 0], beta[1, 1] = 1., 1.5e-15
    mean = np.eye(10)[1]
    residual = np.ones(10)
    projector = sa.gls_projector(beta, residual)
    premia, orthogonal = sa.gls_split(mean, beta, residual)
    np.testing.assert_allclose(projector @ mean, orthogonal, atol=1e-14)
    np.testing.assert_allclose(orthogonal, mean, atol=1e-14)
    np.testing.assert_allclose(premia, 0., atol=1e-14)
    factor_book, neutral_book, _ = sa.holdings_decomposition(mean, beta, residual)
    np.testing.assert_allclose(neutral_book, projector.T @ mean, atol=1e-14)
    np.testing.assert_allclose(factor_book, 0., atol=1e-14)
    with pytest.raises(ValueError, match='unattainable'):
        sa.factor_mimicking_portfolios(beta, residual)
    with pytest.raises(ValueError, match='unattainable'):
        sa.fixed_exposure_fpir([0., 1.], np.eye(2), beta, residual)


@pytest.mark.parametrize('small, rank', [(0., 1), (1.5e-15, 1), (3.e-15, 2)])
def test_shared_rank_metadata_and_explicit_cutoff(small, rank):
    """The declared threshold decides retention once, including a zero column."""
    beta = np.zeros((10, 2))
    beta[0, 0], beta[1, 1] = 1., small
    geometry = sa._gls_geometry(beta, np.ones(10))
    assert geometry.relative_cutoff == np.finfo(float).eps * 10
    assert geometry.cutoff == geometry.relative_cutoff
    assert geometry.rank == rank
    np.testing.assert_allclose(geometry.singular_values, [1., small])
    forced = sa._gls_geometry(beta, np.ones(10), rcond=.1)
    assert forced.rank == 1
    assert forced.cutoff == .1
    np.testing.assert_allclose(forced.white_pinv @ beta, np.diag([1., 0.]))


@pytest.mark.parametrize('rcond', [-1., 1., np.nan, np.inf, [.1], 1j, 'invalid'])
def test_invalid_relative_cutoff(rcond):
    """A custom threshold is an explicit finite scalar relative tolerance."""
    with pytest.raises(ValueError, match='rcond'):
        sa._gls_geometry(np.ones((2, 1)), np.ones(2), rcond=rcond)


@pytest.mark.parametrize('beta', [np.zeros((3, 2)), np.array([[1., 2., 0.],
                                                           [0., 0., 1.]])])
def test_zero_and_wide_loading_matrices(beta):
    """Zero and saturated wide spans have exact identity and zero residual operators."""
    size = len(beta)
    residual = np.arange(1., size + 1)
    mean = np.arange(1., size + 1)
    geometry = sa._gls_geometry(beta, residual)
    expected = np.eye(size) if not beta.any() else np.zeros((size, size))
    np.testing.assert_allclose(geometry.residual_projector, expected, atol=1e-14)
    assert geometry.rank == (0 if not beta.any() else size)
    premia, orthogonal = sa.gls_split(mean, beta, residual)
    np.testing.assert_allclose(beta @ premia + orthogonal, mean, atol=1e-14)


def test_correlated_whitening_matches_independent_qr_and_permutations():
    """QR and relabelled axes verify metric symmetry, projection and decomposition."""
    beta = np.array([[1., .2], [.3, 1.], [.7, -.2], [-.3, .8]])
    chol = np.array([[.2, 0., 0., 0.], [.01, .3, 0., 0.],
                     [.02, -.01, .2, 0.], [.01, .03, .02, .3]])
    residual = chol @ chol.T
    q, _ = np.linalg.qr(np.linalg.solve(chol, beta))
    expected = chol @ (np.eye(4) - q @ q.T) @ np.linalg.solve(chol, np.eye(4))
    geometry = sa._gls_geometry(beta, residual)
    p = geometry.residual_projector
    precision = np.linalg.solve(residual, np.eye(4))
    np.testing.assert_allclose(p, expected, atol=1e-13)
    np.testing.assert_allclose(p @ p, p, atol=1e-13)
    np.testing.assert_allclose(p.T @ precision, precision @ p, atol=1e-13)
    np.testing.assert_allclose(p @ beta, 0., atol=1e-13)
    mean = np.array([.01, -.02, .04, .01])
    premia, h = sa.gls_split(mean, beta, residual)
    np.testing.assert_allclose(beta @ premia + h, mean, atol=1e-13)
    np.testing.assert_allclose(beta.T @ precision @ h, 0., atol=1e-13)
    order = np.array([2, 0, 3, 1])
    permuted = sa.gls_projector(beta[order, ::-1], residual[np.ix_(order, order)])
    np.testing.assert_allclose(permuted, p[np.ix_(order, order)], atol=1e-13)
    covariance = beta @ np.array([[.04, .01], [.01, .03]]) @ beta.T + residual
    total = mean @ np.linalg.solve(covariance, mean)
    fitted = beta @ premia
    assert total == pytest.approx(fitted @ np.linalg.solve(covariance, fitted)
                                  + h @ precision @ h, abs=1e-13)
