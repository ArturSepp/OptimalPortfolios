"""Independent asset solves, QR projections and fixed-book overlay oracles."""

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import minimize

from optimalportfolios.matf import sharpe_accounting as sa


@pytest.fixture
def model():
    """Return a fixed full-rank model with off-diagonal residual covariance."""
    lam = np.array([0.05, -0.02])
    factor = np.array([[0.04, 0.006], [0.006, 0.02]])
    beta = np.array([[1., 0.2], [0.3, 1.], [0.7, -0.2]])
    residual = np.array([[0.03, 0.004, -0.002],
                         [0.004, 0.02, 0.001], [-0.002, 0.001, 0.025]])
    return lam, factor, beta, residual


def test_achievable_matches_asset_solve_and_diagonal_contract(model):
    """Asset-space tangency independently agrees with the rank-safe identity."""
    lam, factor, beta, residual = model
    for omega in (residual, np.diag(residual)):
        covariance = beta @ factor @ beta.T + (
            np.diag(omega) if omega.ndim == 1 else omega)
        mean = beta @ lam
        expected = mean @ np.linalg.solve(covariance, mean)
        actual = sa.achievable_sharpe2(lam, factor, beta, omega)
        assert actual == pytest.approx(expected, abs=1e-12)
        assert sa.fpir(lam, factor, beta, omega) == pytest.approx(
            expected / sa.factor_ceiling(lam, factor))
        split = sa.tangency_access_split(lam, factor, beta, omega)
        assert sum(x.sum() for x in split) == pytest.approx(expected)


def test_gls_and_holdings_against_qr(model):
    """QR validates the projector independently of normal equations/pseudoinverses."""
    lam, factor, beta, residual = model
    chol = np.linalg.cholesky(residual)
    q, _ = np.linalg.qr(np.linalg.solve(chol, beta))
    expected = chol @ (np.eye(3) - q @ q.T) @ np.linalg.inv(chol)
    projector = sa.gls_projector(beta, residual)
    np.testing.assert_allclose(projector, expected, atol=1e-12)
    mean = beta @ lam + np.array([0.02, -0.01, 0.04])
    premia, h = sa.gls_split(mean, beta, residual)
    np.testing.assert_allclose(beta @ premia + h, mean)
    np.testing.assert_allclose(beta.T @ np.linalg.solve(residual, h), 0, atol=1e-12)
    weights = np.array([0.2, 0.3, 0.5])
    wf, wh, exposure = sa.holdings_decomposition(weights, beta, residual)
    np.testing.assert_allclose(wf + wh, weights)
    np.testing.assert_allclose(beta.T @ wh, 0, atol=1e-12)
    assert wf @ residual @ wh == pytest.approx(0, abs=1e-12)
    covariance = beta @ factor @ beta.T + residual
    assert weights @ covariance @ weights == pytest.approx(
        exposure @ factor @ exposure + wf @ residual @ wf + wh @ residual @ wh)
    assert weights @ mean == pytest.approx(exposure @ premia + wh @ h)
    np.testing.assert_allclose(wh, projector.T @ weights, atol=1e-12)


def test_rank_deficiency_and_unattainable_targets(model):
    """Deficient rank preserves GLS/access but cannot manufacture unit portfolios."""
    lam, factor, beta, residual = model
    beta[:, 1] = 2 * beta[:, 0]
    mean = beta @ lam
    covariance = beta @ factor @ beta.T + residual
    assert sa.achievable_sharpe2(lam, factor, beta, residual) == pytest.approx(
        mean @ np.linalg.solve(covariance, mean))
    p = sa.gls_projector(beta, residual)
    np.testing.assert_allclose(p @ p, p, atol=1e-12)
    with pytest.raises(ValueError, match="unattainable"):
        sa.factor_mimicking_portfolios(beta, residual)
    with pytest.raises(ValueError, match="unattainable"):
        sa.fixed_exposure_fpir([1, 0], factor, beta, residual)
    ratio = sa.fixed_exposure_fpir([1, 2], factor, beta, residual)
    assert 0 < ratio < 1


def test_mimicking_hurdles_and_correlation(model):
    """Drop-one ceilings and directly measured residual risk verify factor diagnostics."""
    lam, factor, beta, residual = model
    mimicking = sa.factor_mimicking_portfolios(beta, residual)
    np.testing.assert_allclose(beta.T @ mimicking, np.eye(2), atol=1e-12)
    for j in range(2):
        e = np.eye(2)[j]
        expected = factor[j, j] / (factor[j, j] + mimicking[:, j] @ residual @ mimicking[:, j])
        assert sa.fixed_exposure_fpir(e, factor, beta, residual) == pytest.approx(expected)
    hurdle, variance, gain = sa.hurdle_premia(lam, factor)
    for j in range(2):
        other = 1 - j
        assert hurdle[j] == pytest.approx(factor[j, other] * lam[other] / factor[other, other])
        assert variance[j] == pytest.approx(
            factor[j, j] - factor[j, other]**2 / factor[other, other])
        assert gain[j] == pytest.approx(
            sa.factor_ceiling(lam, factor) - lam[other]**2 / factor[other, other])
    standalone, correlation = sa.ceiling_premium_correlation_split(lam, factor)
    assert standalone.sum() + correlation.sum() == pytest.approx(sa.factor_ceiling(lam, factor))


@pytest.mark.parametrize("selected", [[], [0], [1], [0, 1], [True, False], None])
def test_partial_overlay_against_augmented_solve(model, selected):
    """Direct augmented tangency keeps residual alpha in partial-overlay capacity."""
    lam, factor, beta, residual = model
    alpha = np.array([0.02, 0.01, -0.03])
    mean = beta @ lam + alpha
    covariance = beta @ factor @ beta.T + residual
    idx = np.arange(2) if selected is None else np.asarray(selected)
    if idx.dtype == bool:
        idx = np.flatnonzero(idx)
    idx = idx.astype(int)
    cross = beta @ factor[:, idx]
    augmented = np.block([[covariance, cross], [cross.T, factor[np.ix_(idx, idx)]]])
    augmented_mean = np.r_[mean, lam[idx]]
    expected = augmented_mean @ np.linalg.solve(augmented, augmented_mean)
    gain, capacity = sa.partial_overlay_value(lam, factor, beta, residual, alpha, selected)
    assert capacity == pytest.approx(expected, abs=1e-12)
    assert gain == pytest.approx(expected - mean @ np.linalg.solve(covariance, mean))
    if len(idx) == 2:
        np.testing.assert_allclose(sa.overlay_value(lam, factor, beta, residual, alpha),
                                   [gain, capacity], atol=1e-12)


@pytest.mark.parametrize("selected", [[0], [1], [0, 1]])
def test_fixed_book_completion_against_optimization(model, selected):
    """A numerical optimizer varies only overlays and holds asset holdings fixed."""
    lam, factor, beta, residual = model
    weights = np.array([0.2, 0.3, 0.5])
    mean = beta @ lam + np.array([0.06, 0.05, 0.04])
    result = sa.completion_sharpe2(weights, mean, lam, factor, beta, residual, selected)
    exposure = beta.T @ weights
    c = weights @ (mean - beta @ lam)
    v = weights @ residual @ weights

    def objective(overlay):
        """Return negative signed Sharpe while preserving the original asset book."""
        total = exposure.copy()
        total[selected] += overlay
        return -(c + total @ lam) / np.sqrt(v + total @ factor @ total)

    direct = minimize(objective, np.zeros(len(selected)), method="BFGS", tol=1e-11)
    assert result.status is sa.CompletionStatus.FINITE
    assert result.signed_sharpe == pytest.approx(-direct.fun, abs=1e-8)
    np.testing.assert_allclose(result.overlays[selected], direct.x, atol=1e-5)
    assert objective(result.overlays[selected]) == pytest.approx(-result.signed_sharpe)


@pytest.mark.parametrize("alpha", [0.0, -0.03])
def test_nonpositive_mean_supremum_is_not_full_hedge(model, alpha):
    """Positive factor leverage approaches the bound without any finite optimizer."""
    lam, factor, beta, residual = model
    weights = np.array([0.2, 0.3, 0.5])
    result = sa.completion_sharpe2(weights, beta @ lam + alpha, lam, factor, beta, residual)
    assert result.status is sa.CompletionStatus.SUPREMUM
    assert result.overlays is None and result.exposures is None
    assert result.hedge_only_sharpe <= 0
    direction = np.linalg.solve(factor, lam)
    v = weights @ residual @ weights
    values = [(alpha + t * lam @ direction) / np.sqrt(v + t*t * direction @ factor @ direction)
              for t in (10, 100, 10000)]
    assert values[0] < values[1] < values[2] < result.signed_sharpe
    assert values[-1] == pytest.approx(result.signed_sharpe, abs=2e-5)


def test_completion_boundary_cases(model):
    """Empty sets, zero premia and zero books have explicit signed/degenerate semantics."""
    lam, factor, beta, residual = model
    weights = np.array([0.2, 0.3, 0.5])
    mean = beta @ lam + 0.04
    fixed = sa.completion_sharpe2(weights, mean, lam, factor, beta, residual, [])
    assert fixed.status is sa.CompletionStatus.FIXED_BOOK
    assert fixed.sharpe2 == pytest.approx(fixed.signed_sharpe**2)
    negative = sa.completion_sharpe2(weights, -mean, lam, factor, beta, residual, [])
    assert negative.signed_sharpe < 0 and negative.sharpe2 is None
    for c in (0, -0.02, 0.02):
        result = sa.completion_sharpe2(weights, np.full(3, c), np.zeros(2), factor, beta, residual)
        if c > 0:
            assert result.status is sa.CompletionStatus.FINITE
            np.testing.assert_allclose(result.exposures, 0)
        else:
            assert result.status is sa.CompletionStatus.NO_POSITIVE_MEAN
            assert result.sharpe2 is None
    zero = np.zeros(3)
    result = sa.completion_sharpe2(zero, mean, lam, factor, beta, residual)
    assert result.status is sa.CompletionStatus.FINITE
    degenerate = sa.completion_sharpe2(zero, mean, np.zeros(2), factor, beta, residual)
    assert degenerate.status is sa.CompletionStatus.ZERO_VARIANCE
    assert degenerate.signed_sharpe is None
    fixed_zero = sa.completion_sharpe2(zero, mean, lam, factor, beta, residual, [])
    assert fixed_zero.signed_sharpe is None


def test_ir_bound_against_zero_investment_basis(model):
    """A separately constructed active subspace verifies the benchmark-relative bound."""
    lam, factor, beta, residual = model
    mean = beta @ lam + np.array([0.02, 0.01, -0.03])
    covariance = beta @ factor @ beta.T + residual
    basis = np.array([[1, 0], [0, 1], [-1, -1]])
    expected = (basis.T @ mean) @ np.linalg.solve(basis.T @ covariance @ basis, basis.T @ mean)
    assert sa.ir_bound(mean, covariance) == pytest.approx(expected)


def test_labelled_inputs_reject_reordering(model):
    """Pandas labels must agree exactly, including both covariance axes."""
    lam, factor, beta, residual = model
    factors = pd.Index(['Equity', 'Rates'])
    assets = pd.Index(['A', 'B', 'C'])
    p = pd.Series(lam, index=factors)
    f = pd.DataFrame(factor, index=factors, columns=factors)
    b = pd.DataFrame(beta, index=assets, columns=factors)
    o = pd.DataFrame(residual, index=assets, columns=assets)
    assert sa.achievable_sharpe2(p, f, b, o) == pytest.approx(sa.achievable_sharpe2(*model))
    sa.gls_split(pd.Series([0.01, 0.02, 0.03], index=assets), b, o)
    sa.fixed_exposure_fpir(pd.Series([1., 0.], index=factors), f, b, o)
    for wrong in (p.iloc[::-1], pd.Series(lam, index=['Equity', 'Equity'])):
        with pytest.raises(ValueError, match='label/order'):
            sa.achievable_sharpe2(wrong, f, b, o)
    with pytest.raises(ValueError, match='label/order'):
        sa.gls_split(pd.Series([1, 2, 3], index=assets[::-1]), b, o)
    with pytest.raises(ValueError, match='label/order'):
        sa.achievable_sharpe2(p, f, b, o.iloc[:, ::-1])


@pytest.mark.parametrize("which,value", [
    (0, [np.nan, 1]), (0, []), (0, [[1, 2]]), (0, [1j, 2]),
    (0, [1, 2, 3]), (1, [[1, 2], [0, 1]]), (1, np.zeros((2, 2))),
    (1, np.eye(3)), (2, np.ones((3, 3))), (3, [-1, 1, 1]),
    (3, [1j, 1, 1]), (3, [1, 2]), (3, [1, np.inf, 1]),
])
def test_invalid_model_inputs(model, which, value):
    """Malformed, singular and nonfinite covariance inputs fail without repair."""
    inputs = list(model)
    inputs[which] = value
    with pytest.raises(ValueError):
        sa.achievable_sharpe2(*inputs)


@pytest.mark.parametrize("selected", [[[0]], [0., 1.], [-1], [2], [0, 0], [True]])
def test_invalid_hedge_sets(model, selected):
    """Ambiguous hedge selections are rejected before optimization."""
    with pytest.raises(ValueError, match='hedgeable'):
        sa.partial_overlay_value(*model, np.zeros(3), selected)


def test_undefined_ratios_and_vector_dimensions(model):
    """Zero denominators and mismatched targets fail explicitly."""
    lam, factor, beta, residual = model
    with pytest.raises(ValueError, match='zero factor premia'):
        sa.fpir(np.zeros(2), factor, beta, residual)
    with pytest.raises(ValueError, match='zero exposure'):
        sa.fixed_exposure_fpir(np.zeros(2), factor, beta, residual)
    with pytest.raises(ValueError, match='e factor dimension'):
        sa.fixed_exposure_fpir([1, 2, 3], factor, beta, residual)
    with pytest.raises(ValueError, match='m asset dimension'):
        sa.gls_split([1, 2], beta, residual)


def test_underflow_does_not_disguise_zero_risk_arbitrage():
    """An extreme valid covariance that underflows book risk is diagnosed explicitly."""
    with pytest.raises(ValueError, match='zero conditional variance'):
        sa.completion_sharpe2([1e-100], [1e100], [0.01], [[1.]], [[1.]], [[1e-200]])


def test_negative_conditional_variance_is_rejected(monkeypatch):
    """A corrupted Schur solve cannot turn negative conditional variance into a result."""
    original_solve = np.linalg.solve

    def unstable_solve(matrix, rhs):
        """Inject an erroneous cross-covariance solve, leaving all other solves intact."""
        if matrix.shape == (1, 1) and np.shape(rhs) == (1, 1) and rhs[0, 0] == 0.5:
            return np.array([[3.]])
        return original_solve(matrix, rhs)

    monkeypatch.setattr(np.linalg, 'solve', unstable_solve)
    with pytest.raises(ValueError, match='negative conditional variance'):
        sa.completion_sharpe2([1.], [0.1], [0.01, 0.02],
                              [[1., 0.5], [0.5, 1.]], [[0., 1.]], [[1e-200]], [0])
