"""
input guards and container validation for the covariance stabiliser.

``covar_factorization_test.py`` covers the solver-facing behaviour: that the factor reproduces
the stabilized covariance, that factorized and legacy solves agree, and that each solver
factorizes exactly once. This file covers the other half -- what happens when the input is not
a usable covariance at all.

That matters because every one of these guards sits between a caller mistake and a silently
wrong optimisation. A non-square array is the wrong slice of a panel; a NaN propagates into
every eigenvalue and out into the weights; a negative ``eigenvalue_floor`` leaves the
stabilized covariance indefinite, which is the exact condition the module exists to remove.
Each returns a stated error rather than a LinAlgError from three frames down.

The tolerance tests pin the line between numerical residue and a materially indefinite risk
model -- and, in particular, that the threshold is *relative* to the largest eigenvalue, so
an annualised covariance and a daily one do not get the same absolute cutoff.
"""
# packages
import numpy as np
import pytest
# optimalportfolios
from optimalportfolios.optimization.covar_factorization import (
    DEFAULT_EIGENVALUE_FLOOR,
    CovarianceFactorization,
    factorize_covariance,
)


def psd_covar(n: int = 4, seed: int = 20260812) -> np.ndarray:
    """A well-conditioned positive-definite covariance."""
    rng = np.random.default_rng(seed)
    root = rng.normal(size=(n, n))
    return root @ root.T + np.eye(n)


def with_eigenvalues(values: list) -> np.ndarray:
    """A symmetric matrix with exactly the given spectrum."""
    rng = np.random.default_rng(11)
    vectors, _ = np.linalg.qr(rng.normal(size=(len(values), len(values))))
    return vectors @ np.diag(values) @ vectors.T


# --------------------------------------------------------------------------- #
# behaviour on inputs that are usable but awkward
# --------------------------------------------------------------------------- #
def test_a_well_conditioned_input_is_returned_essentially_unchanged() -> None:
    """Nothing is floored when every eigenvalue already clears the floor."""
    covar = psd_covar()
    result = factorize_covariance(covar)
    np.testing.assert_allclose(result.covar, covar, atol=1e-10)
    assert result.n_eigenvalues_floored == 0
    assert result.max_eigenvalue_adjustment == pytest.approx(0.0, abs=1e-12)


def test_an_asymmetric_input_is_symmetrized_rather_than_refused() -> None:
    """The input is symmetrized before eigh, so a lopsided covariance still factors."""
    covar = psd_covar()
    covar[0, 1] += 1e-6                      # break symmetry by a hair
    result = factorize_covariance(covar)
    np.testing.assert_allclose(result.covar, result.covar.T, atol=1e-12)


def test_the_negative_tolerance_is_relative_to_the_largest_eigenvalue() -> None:
    """The same -1e-9 eigenvalue is a defect at unit scale and residue at scale 1e3.

    The guard multiplies the tolerance by max(1, max|eigenvalue|), so what counts as residue
    depends on the size of the *other* eigenvalues. Note this is not invariant to rescaling
    the whole matrix: that moves the eigenvalue and the tolerance together.
    """
    with pytest.raises(ValueError, match='materially indefinite'):
        factorize_covariance(with_eigenvalues([-1e-9, 1.0, 1.5, 2.0]))
    assert factorize_covariance(
        with_eigenvalues([-1e-9, 1e3, 1.5e3, 2e3])).n_eigenvalues_floored == 1


def test_a_rank_deficient_input_gives_an_infinite_raw_condition_number() -> None:
    """A degenerate covariance is reported as infinitely conditioned, not divided through.

    ``raw_condition_number`` is only a ratio when every eigenvalue is strictly positive;
    otherwise it is inf, which is what the production diagnostics key off.
    """
    result = factorize_covariance(np.zeros((3, 3)))
    assert result.raw_condition_number == float('inf')
    assert result.n_eigenvalues_floored == 3
    assert result.stabilized_min_eigenvalue == pytest.approx(DEFAULT_EIGENVALUE_FLOOR)
    assert np.isfinite(result.stabilized_condition_number)


# --------------------------------------------------------------------------- #
# input guards
# --------------------------------------------------------------------------- #
def test_a_non_square_input_raises() -> None:
    """The commonest caller error: a panel slice rather than a covariance."""
    with pytest.raises(ValueError, match='must be a square matrix'):
        factorize_covariance(np.ones((3, 4)))


def test_a_one_dimensional_input_raises() -> None:
    """A vector is not a covariance, and eigh would not say so clearly."""
    with pytest.raises(ValueError, match='must be a square matrix'):
        factorize_covariance(np.ones(4))


def test_an_empty_universe_raises() -> None:
    """Zero assets is a caller error, not an empty optimisation."""
    with pytest.raises(ValueError, match='at least one asset'):
        factorize_covariance(np.zeros((0, 0)))


def test_a_non_finite_entry_raises() -> None:
    """A NaN would propagate into every eigenvalue and out into the weights."""
    covar = psd_covar()
    covar[1, 1] = np.nan
    with pytest.raises(ValueError, match='only finite values'):
        factorize_covariance(covar)


def test_a_negative_eigenvalue_floor_raises() -> None:
    """A negative floor would leave the stabilized covariance indefinite."""
    with pytest.raises(ValueError, match='eigenvalue_floor must be non-negative'):
        factorize_covariance(psd_covar(), eigenvalue_floor=-1e-8)


def test_a_negative_tolerance_raises() -> None:
    """A negative tolerance inverts the residue test."""
    with pytest.raises(ValueError, match='negative_eigenvalue_tolerance must be non-negative'):
        factorize_covariance(psd_covar(), negative_eigenvalue_tolerance=-1e-8)


def test_an_eigendecomposition_failure_is_reported_as_a_value_error(monkeypatch) -> None:
    """LAPACK non-convergence is surfaced in this module's own vocabulary."""
    def fail(_matrix):
        """Stand in for a non-converging LAPACK call."""
        raise np.linalg.LinAlgError('eigenvalues did not converge')

    monkeypatch.setattr(np.linalg, 'eigh', fail)
    with pytest.raises(ValueError, match='eigendecomposition did not converge'):
        factorize_covariance(psd_covar())


def test_the_reconstruction_check_cannot_fire(monkeypatch) -> None:
    """The post-decomposition check is unreachable, and the reason is worth recording.

    ``factor`` is ``V sqrt(L)`` and ``stabilized_covar`` is ``V L V.T`` -- both built from the
    same ``V`` -- so ``factor @ factor.T`` equals the stabilized covariance identically, not
    approximately, whatever ``eigh`` returns. Feeding back deliberately non-orthogonal
    eigenvectors still reconstructs, so the raise inside ``factorize_covariance`` is defensive
    code with no reachable input. Stated here rather than contriving a monkeypatch to reach it.
    """
    real_eigh = np.linalg.eigh

    def skewed(matrix):
        """Return real eigenvalues with a deliberately non-orthogonal eigenvector matrix."""
        eigenvalues, eigenvectors = real_eigh(matrix)
        return eigenvalues, eigenvectors + 0.5

    monkeypatch.setattr(np.linalg, 'eigh', skewed)
    result = factorize_covariance(psd_covar())
    np.testing.assert_allclose(result.factor @ result.factor.T, result.covar, atol=1e-12)


# --------------------------------------------------------------------------- #
# the container's own validation
# --------------------------------------------------------------------------- #
def test_the_container_rejects_a_non_square_covariance() -> None:
    """Constructed directly, the container re-checks what the factory guarantees."""
    with pytest.raises(ValueError, match='covar must be square'):
        CovarianceFactorization(covar=np.ones((2, 3)), factor=np.ones((2, 3)))


def test_the_container_rejects_a_factor_with_the_wrong_row_count() -> None:
    """One row per asset: a transposed factor is the likely mistake and is caught."""
    with pytest.raises(ValueError, match='one row per asset'):
        CovarianceFactorization(covar=np.eye(3), factor=np.ones((2, 3)))


def test_the_container_rejects_non_finite_inputs() -> None:
    """A NaN in either input fails before the reconstruction check divides by a NaN norm."""
    with pytest.raises(ValueError, match='inputs must be finite'):
        CovarianceFactorization(covar=np.eye(2) * np.nan, factor=np.eye(2))


def test_the_container_rejects_a_factor_that_misses_the_covariance() -> None:
    """The reconstruction tolerance is the container's core invariant."""
    with pytest.raises(ValueError, match='does not reconstruct covar'):
        CovarianceFactorization(covar=np.eye(3), factor=2.0 * np.eye(3))


def test_the_container_accepts_a_matching_pair() -> None:
    """A genuine (covar, B) pair passes and is stored as float arrays."""
    result = CovarianceFactorization(covar=np.eye(3), factor=np.eye(3))
    assert result.covar.dtype == float
    assert result.factor.dtype == float
