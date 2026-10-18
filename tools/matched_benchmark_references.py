"""Independent B0 benchmark references, outside the production allocation path.

QIS owns the reused cash-ledger oracle; FactorLasso owns the covariance
container used for assembly. The small matrix references here are benchmark assertions,
not alternative portfolio statistics, estimators or execution implementations.
"""
from importlib.metadata import version
from pathlib import Path
import re

import numpy as np
import pandas as pd


def package_version(module):
    """Record a source export's version or the installed distribution's version."""
    metadata = Path(module.__file__).resolve().parents[2] / 'pyproject.toml'
    if metadata.is_file():
        return re.search(r'(?m)^version\s*=\s*["\']([^"\']+)',
                         metadata.read_text(encoding='utf-8'))[1]
    return version(module.__name__)


def optimizer_oracle(covariance, means, gamma):
    """Solve the full-investment equality KKT equations independently of CVXPY.

    Bounds are checked separately; this oracle is valid only for an interior solution.
    A positive-definite covariance establishes uniqueness for these cases.
    """
    n = len(means)
    matrix = np.block([[gamma * covariance, np.ones((n, 1))],
                       [np.ones((1, n)), np.zeros((1, 1))]])
    return np.linalg.solve(matrix, np.r_[means, 1.0])[:n]


def verify_optimizer(weights, covariance, means, gamma, cap, atol=2e-5):
    """Check feasibility, objective optimality and unique-solution weight agreement."""
    reference = optimizer_oracle(covariance, means, gamma)
    if np.linalg.eigvalsh(covariance).min() <= 0:
        raise AssertionError('Benchmark covariance must be strictly positive definite')
    if not (reference.min() > 0 and reference.max() < cap):
        raise AssertionError('KKT oracle is outside the declared inactive bounds')
    assert abs(weights.sum() - 1) < 2e-7
    assert weights.min() >= -2e-7 and weights.max() <= cap + 2e-7
    objective = 0.5 * gamma * weights @ covariance @ weights - means @ weights
    optimum = 0.5 * gamma * reference @ covariance @ reference - means @ reference
    assert abs(objective - optimum) < 2e-8
    np.testing.assert_allclose(weights, reference, atol=atol, rtol=0)
    return {'objective': float(objective), 'oracle_objective': float(optimum),
            'objective_gap': float(objective - optimum),
            'budget_residual': float(weights.sum() - 1),
            'max_weight_error': float(np.max(np.abs(weights - reference))),
            'unique': True}


def ledger_reference(prices, targets, lag, cost):
    """Delegate independent wealth/trade arithmetic to the existing QIS test oracle."""
    from qis.portfolio.tests import backtester_missing_prices_test as ledger

    positions = prices.index.get_indexer(targets.index) + lag
    if (positions >= len(prices)).any():
        raise ValueError('A target lies beyond the benchmark observation window')
    flags = np.zeros(len(prices), dtype=bool)
    flags[positions] = True
    zeros = np.zeros(len(prices))
    nav, units, expenses, cash, marks = ledger._ledger_reference(
        prices.to_numpy(), targets.to_numpy(), flags, zeros, zeros,
        np.zeros(prices.shape), np.full(prices.shape, cost), None)
    return {'nav': nav, 'units': units, 'costs': expenses, 'cash': cash, 'marks': marks,
            'source': str(Path(ledger.__file__).resolve())}


def factorlasso_covariance(model, factors):
    """Assemble per-observation covariance through FactorLasso's public snapshot."""
    import factorlasso as fl

    residual_variances = pd.DataFrame(
        {fl.VarianceColumns.RESIDUAL_VARS.value: model.estimation_result_.ss_res},
        index=model.coef_.index,
    )
    assembled = fl.CurrentFactorCovarData(
        x_covar=factors.cov(ddof=0),
        y_betas=model.coef_,
        y_variances=residual_variances,
        estimation_date=factors.index[-1],
    )
    return assembled.get_y_covar().to_numpy()


def factor_oracle(factors, responses, residual_ddof):
    """Use an augmented least-squares design and raw residual sums as an OLS reference."""
    x, y = np.asarray(factors), np.asarray(responses)
    design = np.c_[np.ones(len(x)), x]
    solution = np.linalg.lstsq(design, y, rcond=None)[0]
    residuals = y - design @ solution
    centered = x - x.mean(axis=0)
    factor_covariance = centered.T @ centered / len(x)
    beta = solution[1:].T
    covariance = (beta @ factor_covariance @ beta.T
                  + np.diag(np.sum(residuals * residuals, axis=0) / (len(x) - residual_ddof)))
    return beta, covariance


def forecast_scores(covariance, heldout):
    """Score per-period forecasts against future observations, without annualisation."""
    values = np.asarray(heldout)
    centered = values - values.mean(axis=0)
    realized = centered.T @ centered / len(values)
    equal_weight = np.full(values.shape[1], 1 / values.shape[1])
    forecast_variance = float(equal_weight @ covariance @ equal_weight)
    portfolio_returns = values.mean(axis=1)
    realized_variance = float(np.mean((portfolio_returns - portfolio_returns.mean()) ** 2))
    squared_loss = float(np.sum((covariance - realized) ** 2) / np.sum(realized ** 2))
    # Independent matrix-norm and projected-return calculations protect the score formulas.
    np.testing.assert_allclose(squared_loss,
                               (np.linalg.norm(covariance - realized)
                                / np.linalg.norm(realized)) ** 2, rtol=1e-12)
    np.testing.assert_allclose(realized_variance, equal_weight @ realized @ equal_weight)
    sign_f, log_f = np.linalg.slogdet(covariance)
    sign_r, log_r = np.linalg.slogdet(realized)
    if sign_f <= 0 or sign_r <= 0:
        raise AssertionError('Stein loss requires positive-definite forecast and realized moments')
    stein = float(np.trace(np.linalg.solve(covariance, realized))
                  - (log_r - log_f) - len(covariance))
    inverse_cholesky = np.linalg.inv(np.linalg.cholesky(covariance))
    eigenvalues = np.linalg.eigvalsh(inverse_cholesky @ realized @ inverse_cholesky.T)
    np.testing.assert_allclose(stein, np.sum(eigenvalues - np.log(eigenvalues) - 1),
                               atol=1e-10, rtol=1e-10)
    return {'stein_covariance_loss': stein, 'relative_squared_frobenius_loss': squared_loss,
            'forecast_equal_weight_variance': forecast_variance,
            'realized_equal_weight_variance': realized_variance,
            'realized_over_forecast_variance': realized_variance / forecast_variance}
