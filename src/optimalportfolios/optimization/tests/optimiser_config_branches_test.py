"""Tests for optimiser branches selected by configuration rather than by data.

Each path here is chosen by an ``OptimiserConfig`` flag, a ``Constraints`` flag, or a degenerate
solve, so a suite that only ever runs the default configuration never reaches them. They are the
last uncovered branches in the optimisation layer that a test can reach.
"""

import numpy as np
import pandas as pd
import pytest

import optimalportfolios.optimization.covar_factorization as covar_factorization
from optimalportfolios.optimization.config import OptimiserConfig
from optimalportfolios.optimization.constraints import Constraints
from optimalportfolios.optimization.covar_factorization import factorize_covariance
from optimalportfolios.optimization.general.max_diversification import (
    wrapper_maximise_diversification,
)
from optimalportfolios.optimization.taa.maximise_alpha_with_target_yield import (
    wrapper_maximise_alpha_with_target_return,
)


TICKERS = pd.Index(["A", "B", "C"])


def _covar() -> pd.DataFrame:
    """Return a well-conditioned three-asset covariance matrix."""
    correlation = np.array([[1.0, 0.3, 0.1], [0.3, 1.0, 0.2], [0.1, 0.2, 1.0]])
    vols = np.array([0.10, 0.15, 0.22])
    return pd.DataFrame(
        correlation * np.outer(vols, vols), index=TICKERS, columns=TICKERS,
    )


def test_diversification_without_the_total_to_good_ratio_still_solves() -> None:
    """Disabling the ratio leaves the filtered universe unscaled and still fully invested.

    ``apply_total_to_good_ratio`` rescales the constraint set when NaN filtering shrinks the
    universe. With no asset filtered out the two configurations should agree, which is what shows
    the flag selects a code path rather than changing the answer on clean input.
    """
    covar = _covar()
    constraints = Constraints(is_long_only=True)

    scaled = wrapper_maximise_diversification(
        pd_covar=covar, constraints=constraints,
        optimiser_config=OptimiserConfig(apply_total_to_good_ratio=True),
    )
    unscaled = wrapper_maximise_diversification(
        pd_covar=covar, constraints=constraints,
        optimiser_config=OptimiserConfig(apply_total_to_good_ratio=False),
    )

    assert list(unscaled.index) == list(TICKERS)
    assert unscaled.sum() == pytest.approx(1.0, abs=1e-6)
    assert (unscaled >= -1e-9).all()
    pd.testing.assert_series_equal(scaled, unscaled, atol=1e-6)


def test_long_short_target_return_solve_permits_negative_weights() -> None:
    """Dropping ``is_long_only`` releases the non-negativity on the solver variable.

    The declaration is made once when the CVXPY variable is built, so a long-short problem that
    still returned non-negative weights everywhere would indicate the flag never reached it.
    """
    covar = _covar()
    alphas = pd.Series([0.05, -0.04, 0.01], index=TICKERS)
    yields = pd.Series([0.03, 0.02, 0.04], index=TICKERS)

    weights, outcome = wrapper_maximise_alpha_with_target_return(
        pd_covar=covar, alphas=alphas, yields=yields, target_return=0.03,
        constraints=Constraints(is_long_only=False, min_weights=pd.Series(-1.0, index=TICKERS),
                                max_weights=pd.Series(1.0, index=TICKERS)),
    )

    assert list(weights.index) == list(TICKERS)
    assert outcome is not None
    assert np.all(np.isfinite(weights.to_numpy()))
    assert weights.min() < 0.0, "a long-short problem produced no short leg"


def test_reconstruction_guard_rejects_a_factor_that_does_not_rebuild_the_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The factorisation refuses to return a factor whose product misses the covariance.

    ``factor @ factor.T`` is exact to floating point on a symmetric eigendecomposition, so the
    guard cannot be tripped by any covariance matrix that reaches it — the tolerance is lowered
    here instead. What is under test is the guard's behaviour, not the arithmetic: without it a
    silently wrong square root propagates into every solve that uses the factorised form.
    """
    monkeypatch.setattr(covar_factorization, "DEFAULT_RECONSTRUCTION_RTOL", -1.0)

    with pytest.raises(ValueError, match="covariance factor reconstruction failed"):
        factorize_covariance(_covar().to_numpy())


def test_factorisation_succeeds_at_the_shipped_tolerance() -> None:
    """The same matrix factorises cleanly at the real tolerance, so the guard is not load-bearing.

    Pairs with the test above: together they show the rejection came from the lowered tolerance
    rather than from a genuinely defective factorisation.
    """
    covar = _covar().to_numpy()

    factorization = factorize_covariance(covar)

    np.testing.assert_allclose(
        factorization.factor @ factorization.factor.T, factorization.covar,
        rtol=1e-10, atol=1e-14,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
