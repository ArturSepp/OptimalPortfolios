"""
Portfolio optimisation using maximum diversification objective.

Maximises the diversification ratio:

    DR(w) = w'σ / sqrt(w'Σw)

where σ is the vector of asset volatilities (sqrt of diagonal of Σ) and
w'Σw is the portfolio variance.

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
from optimalportfolios.optimization.config import OptimiserConfig


def rolling_maximise_diversification(prices: pd.DataFrame,
                                     constraints: Constraints,
                                     covar_dict: Dict[pd.Timestamp, pd.DataFrame],
                                     optimiser_config: OptimiserConfig = OptimiserConfig(apply_total_to_good_ratio=True)
                                     ) -> pd.DataFrame:
    """
    Compute rolling maximum diversification portfolios.

    Args:
        prices: Asset price panel. Used for column alignment.
        constraints: Portfolio constraints.
        covar_dict: Pre-computed covariance matrices keyed by rebalancing date.
        optimiser_config: Solver configuration.

    Returns:
        DataFrame of portfolio weights.
    """
    weights = {}
    weights_0 = None
    for date, pd_covar in covar_dict.items():
        weights_ = wrapper_maximise_diversification(pd_covar=pd_covar,
                                                    constraints=constraints,
                                                    weights_0=weights_0,
                                                    optimiser_config=optimiser_config)
        weights_0 = weights_
        weights[date] = weights_

    weights = pd.DataFrame.from_dict(weights, orient='index')
    weights = weights.reindex(columns=prices.columns).fillna(0.0)
    return weights


def wrapper_maximise_diversification(pd_covar: pd.DataFrame,
                                     constraints: Constraints,
                                     weights_0: pd.Series = None,
                                     optimiser_config: OptimiserConfig = OptimiserConfig(apply_total_to_good_ratio=True)
                                     ) -> pd.Series:
    """
    Single-date maximum diversification with NaN/zero-variance filtering.

    Args:
        pd_covar: Covariance matrix (N x N) as DataFrame.
        constraints: Portfolio constraints.
        weights_0: Previous-period weights for warm-start / fallback.
        optimiser_config: Solver configuration.

    Returns:
        Portfolio weights as pd.Series aligned to pd_covar.index.
    """
    vectors = None
    clean_covar, good_vectors = filter_covar_and_vectors_for_nans(pd_covar=pd_covar, vectors=vectors)

    if optimiser_config.apply_total_to_good_ratio:
        total_to_good_ratio = len(pd_covar.columns) / len(clean_covar.columns)
    else:
        total_to_good_ratio = None

    constraints1 = constraints.update_with_valid_tickers(valid_tickers=clean_covar.columns.to_list(),
                                                         total_to_good_ratio=total_to_good_ratio,
                                                         weights_0=weights_0)

    weights = opt_maximise_diversification(covar=clean_covar.to_numpy(),
                                           constraints=constraints1,
                                           verbose=optimiser_config.verbose)
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

    Args:
        covar: Covariance matrix (N x N).
        constraints: Portfolio constraints.
        verbose: If True, print SLSQP solver diagnostics.
        ftol: Function tolerance for convergence.
        maxiter: Maximum number of SLSQP iterations.

    Returns:
        Optimal weights (N,). Falls back to weights_0 or zeros on failure.
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
        if constraints.is_long_only:
            optimal_weights = np.where(optimal_weights > 0.0, optimal_weights, 0.0)

    return optimal_weights


def max_diversification_objective(w: np.ndarray, pars: List[np.ndarray]) -> float:
    """Negative diversification ratio (for minimisation)."""
    covar = pars[0]
    return -calculate_diversification_ratio(w=w, covar=covar)
