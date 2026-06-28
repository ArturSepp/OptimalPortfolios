"""
Portfolio optimisation using CARA (Constant Absolute Risk Aversion) utility.

Implements expected utility maximisation under Gaussian and Gaussian mixture
return distributions. The CARA utility U(r) = -exp(-γr) yields closed-form
certainty equivalents under Gaussian assumptions:

    CE = μ'w - (γ/2) w'Σw               (single Gaussian)
    CE = -Σ_k p_k exp(-γμ_k'w + γ²/2 w'Σ_k w)   (K-component mixture)

The mixture formulation captures fat tails and regime-dependent correlations
by fitting a Gaussian Mixture Model to rolling return windows, then optimising
the expected CARA utility across mixture components via scipy SLSQP.

Reference:
    Sepp A. (2023),
    "Optimal Allocation to Cryptocurrencies in Diversified Portfolios",
    Risk Magazine, pp. 1-6, October 2023.
    Available at https://ssrn.com/abstract=4217841
"""

import warnings
import logging
import numpy as np
import pandas as pd
import qis as qis
from scipy.optimize import minimize
from typing import List

from optimalportfolios.utils.gaussian_mixture import fit_gaussian_mixture
from optimalportfolios.utils.portfolio_funcs import (compute_portfolio_variance, compute_portfolio_risk_contributions)
from optimalportfolios.optimization.constraints import (Constraints, total_weight_constraint, long_only_constraint)
from optimalportfolios.optimization.solver_diagnostics import validate_scipy_solution
from optimalportfolios.optimization.config import OptimiserConfig
from optimalportfolios.utils.weights_drift import apply_drift_to_weights_0

logger = logging.getLogger(__name__)


def rolling_maximize_cara_mixture(prices: pd.DataFrame,
                                  constraints: Constraints,
                                  time_period: qis.TimePeriod,
                                  rebalancing_freq: str = 'QE',
                                  roll_window: int = 52*6,
                                  returns_freq: str = 'W-WED',
                                  carra: float = 0.5,
                                  n_components: int = 3,
                                  optimiser_config: OptimiserConfig = OptimiserConfig(apply_total_to_good_ratio=True)
                                  ) -> pd.DataFrame:
    """
    Compute rolling CARA-optimal portfolios under a Gaussian mixture model.

    Args:
        prices: Asset price panel.
        constraints: Portfolio constraints.
        time_period: Reporting period for output weights.
        rebalancing_freq: Rebalancing frequency (e.g., 'QE', 'ME').
        roll_window: Number of return observations for GMM estimation.
        returns_freq: Frequency for return computation.
        carra: CARA risk aversion parameter γ.
        n_components: Number of Gaussian mixture components K.
        optimiser_config: Solver configuration.

    Returns:
        DataFrame of portfolio weights.
    """
    returns = qis.to_returns(prices=prices, is_log_returns=True, drop_first=True, freq=returns_freq)
    rebalancing_schedule = qis.generate_rebalancing_indicators(df=returns, freq=rebalancing_freq)

    scaler = qis.get_annualization_factor(freq=returns_freq)

    tickers = prices.columns.to_list()
    weights = {}
    weights_0 = None
    prev_date = None
    for idx, (date, value) in enumerate(rebalancing_schedule.items()):
        if idx >= roll_window-1 and value:
            period = qis.TimePeriod(rebalancing_schedule.index[idx - roll_window+1], date)
            rets_ = period.locate(returns).dropna(axis=1, how='any')
            params = fit_gaussian_mixture(x=rets_.to_numpy(), n_components=n_components, an_factor=scaler)
            # drift weights_0 from the last actual rebalance date (not from
            # every schedule tick) to ``date`` before passing to the wrapper.
            weights_0 = apply_drift_to_weights_0(
                weights_0=weights_0, prices=prices,
                prev_date=prev_date, date=date,
                use_drifted_weights_0=optimiser_config.use_drifted_weights_0,
            )
            constraints1 = constraints.update_with_valid_tickers(context=str(pd.Timestamp(date).date()), valid_tickers=rets_.columns.to_list(),
                                                                 total_to_good_ratio=len(tickers)/len(rets_.columns),
                                                                 weights_0=weights_0)

            weights_ = wrapper_maximize_cara_mixture(means=params.means,
                                                     covars=params.covars,
                                                     probs=params.probs,
                                                     constraints=constraints1,
                                                     tickers=rets_.columns.to_list(),
                                                     carra=carra,
                                                     optimiser_config=optimiser_config,
                                                     context=str(pd.Timestamp(date).date()))
            weights_ = weights_.reindex(index=tickers).fillna(0.0)
            weights_0 = weights_
            prev_date = date
            weights[date] = weights_
    weights = pd.DataFrame.from_dict(weights, orient='index')
    weights = weights.reindex(columns=tickers).fillna(0.0)
    if time_period is not None:
        weights = time_period.locate(weights)

    return weights


def wrapper_maximize_cara_mixture(means: List[np.ndarray],
                                  covars: List[np.ndarray],
                                  probs: np.ndarray,
                                  constraints: Constraints,
                                  tickers: List[str],
                                  carra: float = 0.5,
                                  optimiser_config: OptimiserConfig = OptimiserConfig(apply_total_to_good_ratio=True),
                                  context: str = ''
                                  ) -> pd.Series:
    """
    Solve CARA mixture optimisation and return labelled weights.

    Args:
        means: List of K mean vectors, each (N,).
        covars: List of K covariance matrices, each (N x N).
        probs: Mixture probabilities (K,), summing to 1.
        constraints: Portfolio constraints.
        tickers: Asset ticker labels.
        carra: CARA risk aversion parameter γ.
        optimiser_config: Solver configuration.

    Returns:
        Portfolio weights as pd.Series with index=tickers.
    """
    weights = opt_maximize_cara_mixture(means=means,
                                        covars=covars,
                                        probs=probs,
                                        constraints=constraints,
                                        carra=carra,
                                        verbose=optimiser_config.verbose,
                                        context=context)
    weights = pd.Series(weights, index=tickers)
    return weights


def opt_maximize_cara_mixture(means: List[np.ndarray],
                              covars: List[np.ndarray],
                              probs: np.ndarray,
                              constraints: Constraints,
                              carra: float = 0.5,
                              verbose: bool = False,
                              context: str = ''
                              ) -> np.ndarray:
    """
    Maximise expected CARA utility under a Gaussian mixture model via SLSQP.

    Minimises (note sign flip for scipy):

        f(w) = Σ_k p_k exp(-γ μ_k'w + (γ²/2) w'Σ_k w)

    Args:
        means: List of K mean vectors, each (N,). Annualised.
        covars: List of K covariance matrices, each (N x N). Annualised.
        probs: Mixture probabilities (K,), summing to 1.
        constraints: Portfolio constraints.
        carra: CARA risk aversion parameter γ.
        verbose: If True, print SLSQP solver diagnostics.

    Returns:
        Optimal weights (N,). Falls back to weights_0 or equal-weight
        if the solver fails.
    """
    n = covars[0].shape[0]
    if constraints.weights_0 is not None:
        x0 = np.array(constraints.weights_0.to_numpy(), dtype=float)
    else:
        x0 = np.ones(n) / n

    constraints_, bounds = constraints.set_scipy_constraints(covar=covars[0])
    res = minimize(carra_objective_mixture, x0, args=[means, covars, probs, carra], method='SLSQP',
                   constraints=constraints_,
                   bounds=bounds,
                   options={'disp': verbose, 'ftol': 1e-8})
    optimal_weights, _is_valid = validate_scipy_solution(
        res.x, res, constraints, n, solver='SLSQP', context=context)

    return optimal_weights


def opt_maximize_cara(means: np.ndarray,
                      covar: np.ndarray,
                      carra: float = 0.5,
                      min_weights: np.ndarray = None,
                      max_weights: np.ndarray = None,
                      disp: bool = False,
                      is_exp: bool = False,
                      is_print_log: bool = False
                      ) -> np.ndarray:
    """
    Maximise CARA utility under a single Gaussian distribution via SLSQP.

    Supports two objective formulations controlled by ``is_exp``:

        Quadratic (is_exp=False):
            max  μ'w - (γ/2) w'Σw

        Exponential (is_exp=True):
            min  exp(-γ μ'w + (γ²/2) w'Σw)

    Args:
        means: Expected returns (N,). Annualised.
        covar: Covariance matrix (N x N). Annualised.
        carra: CARA risk aversion parameter γ.
        min_weights: Lower bounds per asset (N,).
        max_weights: Upper bounds per asset (N,).
        disp: If True, print SLSQP solver diagnostics.
        is_exp: If True, use the exponential objective; otherwise quadratic.
        is_print_log: If True, print portfolio diagnostics after solving.

    Returns:
        Optimal weights (N,).
    """
    n = covar.shape[0]
    x0 = np.ones(n) / n
    cons = [{'type': 'ineq', 'fun': long_only_constraint},
            {'type': 'eq', 'fun': total_weight_constraint}]
    if min_weights is not None:
        cons.append({'type': 'ineq', 'fun': lambda x: x - min_weights})
    if max_weights is not None:
        cons.append({'type': 'ineq', 'fun': lambda x: max_weights - x})

    if is_exp:
        func = carra_objective_exp
    else:
        func = carra_objective
    res = minimize(func, x0, args=[means, covar, carra], method='SLSQP', constraints=cons,
                   options={'disp': disp, 'ftol': 1e-12})
    w_rb = res.x

    if (not res.success) or w_rb is None or not np.all(np.isfinite(w_rb)):
        logger.warning("opt_maximize_cara: SLSQP did not converge (status=%s: %s); "
                       "returning initial guess", res.status, res.message)
        w_rb = x0

    if is_print_log:
        print(f'return_p = {w_rb@means}, '
              f'sigma_p = {np.sqrt(compute_portfolio_variance(w_rb, covar))}, weights: {w_rb}, '
              f'risk contrib.s: {compute_portfolio_risk_contributions(w_rb, covar).T} '
              f'sum of weights: {sum(w_rb)}')
    return w_rb


def carra_objective(w: np.ndarray, pars: List[np.ndarray]) -> float:
    """Quadratic CARA objective (single Gaussian, negated for minimisation)."""
    means, covar, carra = pars[0], pars[1], pars[2]
    v = means.T @ w - 0.5*carra*w.T @ covar @ w
    return -v


def carra_objective_exp(w: np.ndarray, pars: List[np.ndarray]) -> float:
    """Exponential CARA objective (single Gaussian)."""
    means, covar, carra = pars[0], pars[1], pars[2]
    v = np.exp(-carra*means.T @ w + 0.5*carra*carra*w.T @ covar @ w)
    return v


def carra_objective_mixture(w: np.ndarray, pars: List[np.ndarray]) -> float:
    """Expected CARA disutility under a K-component Gaussian mixture."""
    means, covars, probs, carra = pars[0], pars[1], pars[2], pars[3]
    v = 0.0
    for idx, prob in enumerate(probs):
        v = v + prob*np.exp(-carra*means[idx].T @ w + 0.5*carra*carra*w.T @ covars[idx] @ w)
    return v
