"""
Portfolio optimisation using maximum diversification objective.

Maximises the diversification ratio:

    DR(w) = w'σ / sqrt(w'Σw)

where σ is the vector of asset volatilities (sqrt of diagonal of Σ) and
w'Σw is the portfolio variance. The diversification ratio measures the
ratio of the weighted-average asset volatility to the portfolio volatility,
quantifying the benefit of diversification. DR = 1 implies no diversification
(perfect correlation or single asset); DR > 1 implies diversification gains.

The maximum diversification portfolio (MDP) maximises this ratio subject to
portfolio constraints. It can be interpreted as the tangency portfolio on
the efficient frontier when Sharpe ratios are proportional to volatilities.

Uses scipy SLSQP for the non-convex ratio objective.

References:
    Choueifaty Y. and Coignard Y. (2008),
    "Toward Maximum Diversification",
    The Journal of Portfolio Management, 35(1), 40-51.

    Sepp A. (2023),
    "Optimal Allocation to Cryptocurrencies in Diversified Portfolios",
    Risk Magazine, pp. 1-6, October 2023.
    Available at https://ssrn.com/abstract=4217841
"""
# packages
import warnings
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from typing import List, Dict

# optimalportfolios
from optimalportfolios.utils.portfolio_funcs import calculate_diversification_ratio
from optimalportfolios.utils.filter_nans import filter_covar_and_vectors_for_nans
from optimalportfolios.optimization.constraints import Constraints


def rolling_maximise_diversification(prices: pd.DataFrame,
                                     constraints: Constraints,
                                     covar_dict: Dict[pd.Timestamp, pd.DataFrame]
                                     ) -> pd.DataFrame:
    """
    Compute rolling maximum diversification portfolios.

    At each rebalancing date (defined by the keys of ``covar_dict``),
    solves the maximum diversification problem using the pre-computed
    covariance matrix. Previous-period weights are passed as warm-start
    to stabilise turnover.

    The covariance matrices are produced externally by any CovarEstimator
    (EwmaCovarEstimator, FactorCovarEstimator, etc.), decoupling the
    estimation step from the optimisation step.

    Args:
        prices: Asset price panel. Index=dates, columns=tickers. Used only
            for column alignment of the output weights DataFrame.
        constraints: Portfolio constraints (long-only, weight bounds, group
            exposures, etc.).
        covar_dict: Pre-computed covariance matrices keyed by rebalancing date.
            Typically produced by ``estimator.fit_rolling_covars()``.

    Returns:
        DataFrame of portfolio weights. Index=rebalancing dates,
        columns=tickers aligned to ``prices.columns``.
    """
    weights = {}
    weights_0 = None
    for date, pd_covar in covar_dict.items():
        weights_ = wrapper_maximise_diversification(pd_covar=pd_covar,
                                                    constraints=constraints,
                                                    weights_0=weights_0)
        weights_0 = weights_
        weights[date] = weights_

    weights = pd.DataFrame.from_dict(weights, orient='index')
    weights = weights.reindex(columns=prices.columns.to_list())
    return weights


def wrapper_maximise_diversification(pd_covar: pd.DataFrame,
                                     constraints: Constraints,
                                     weights_0: pd.Series = None
                                     ) -> pd.Series:
    """
    Single-date maximum diversification with NaN/zero-variance filtering.

    Removes assets with NaN or zero diagonal entries in the covariance matrix,
    solves the reduced problem, and maps weights back to the full asset universe
    (excluded assets receive zero weight).

    Args:
        pd_covar: Covariance matrix (N x N) as DataFrame.
        constraints: Portfolio constraints.
        weights_0: Previous-period weights for warm-start / fallback.

    Returns:
        Portfolio weights as pd.Series aligned to pd_covar.index.
    """
    vectors = None
    clean_covar, good_vectors = filter_covar_and_vectors_for_nans(pd_covar=pd_covar, vectors=vectors)

    constraints1 = constraints.update_with_valid_tickers(valid_tickers=clean_covar.columns.to_list(),
                                                         total_to_good_ratio=len(pd_covar.columns) / len(clean_covar.columns),
                                                         weights_0=weights_0)

    weights = opt_maximise_diversification(covar=clean_covar.to_numpy(),
                                           constraints=constraints1)
    weights = pd.Series(weights, index=clean_covar.columns)
    weights = weights.reindex(index=pd_covar.columns).fillna(0.0)
    return weights


def opt_maximise_diversification(covar: np.ndarray,
                                 constraints: Constraints,
                                 verbose: bool = False,
                                 ftol: float = 1e-8,
                                 maxiter: int = 500
                                 ) -> np.ndarray:
    """
    Maximise the diversification ratio via scipy SLSQP.

    Minimises (note sign flip):

        f(w) = -DR(w) = -w'σ / sqrt(w'Σw)

    Starting from equal weights. The objective is non-convex due to the
    ratio form, so the solution depends on the starting point; equal weights
    is a natural initialisation for diversified portfolios.

    Args:
        covar: Covariance matrix (N x N) as numpy array.
        constraints: Portfolio constraints (bounds, exposures, turnover).
        verbose: If True, print SLSQP solver diagnostics.
        ftol: Function tolerance for convergence.
        maxiter: Maximum number of SLSQP iterations.

    Returns:
        Optimal weights (N,). Falls back to weights_0 or zeros if the
        solver fails. Long-only constraint violations are clipped to zero
        when ``constraints.is_long_only`` is True.
    """
    n = covar.shape[0]
    x0 = np.ones(n) / n

    constraints_, bounds = constraints.set_scipy_constraints(covar=covar)
    res = minimize(max_diversification_objective, x0, args=[covar], method='SLSQP',
                   constraints=constraints_,
                   bounds=bounds,
                   options={'disp': verbose, 'ftol': ftol, 'maxiter': maxiter})

    optimal_weights = res.x

    if res.success == False or optimal_weights is None:
        if constraints.weights_0 is not None:
            optimal_weights = constraints.weights_0
            mes = f"using weights_0"
        else:
            optimal_weights = np.zeros(n)
            mes = f"using zero weights"
        warnings.warn(f"opt_maximise_diversification(): problem is not solved, {mes}")

    else:
        # clip numerical noise for long-only portfolios
        if constraints.is_long_only:
            optimal_weights = np.where(optimal_weights > 0.0, optimal_weights, 0.0)

    return optimal_weights


def max_diversification_objective(w: np.ndarray, pars: List[np.ndarray]) -> float:
    """
    Negative diversification ratio (for minimisation).

    f(w) = -w'σ / sqrt(w'Σw)

    Args:
        w: Portfolio weights (N,).
        pars: [covar] — covariance matrix (N x N).

    Returns:
        Negative diversification ratio (scalar).
    """
    covar = pars[0]
    return -calculate_diversification_ratio(w=w, covar=covar)
