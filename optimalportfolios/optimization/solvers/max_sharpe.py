"""
Implementation of maximum Sharpe ratio portfolios.
"""
import numpy as np
import pandas as pd
import cvxpy as cvx
import qis as qis
from typing import Dict

from optimalportfolios.utils.filter_nans import filter_covar_and_vectors_for_nans
from optimalportfolios.optimization.constraints import Constraints


def rolling_maximize_portfolio_sharpe(prices: pd.DataFrame,
                                      constraints: Constraints,
                                      covar_dict: Dict[pd.Timestamp, pd.DataFrame],
                                      returns_freq: str = 'W-WED',
                                      span: int = 52,
                                      solver: str = 'ECOS_BB',
                                      ) -> pd.DataFrame:
    """
    Maximise portfolio Sharpe ratio at each rebalancing date.

    Uses pre-computed covariance matrices from covar_dict and expanding
    EWMA means estimated from prices at each rebalancing date.

    Args:
        prices: Asset price panel. Index=dates, columns=tickers.
        constraints: Portfolio constraints.
        covar_dict: Pre-computed covariance matrices keyed by rebalancing date.
        returns_freq: Frequency for return computation for mean estimation.
        span: EWMA span for mean estimation.
        solver: CVXPY solver name.

    Returns:
        Rolling portfolio weights. Index=rebalancing dates, columns=tickers.
    """
    rebalancing_dates = list(covar_dict.keys())
    means = estimate_rolling_ewma_means(prices=prices,
                                        rebalancing_dates=rebalancing_dates,
                                        returns_freq=returns_freq,
                                        span=span,
                                        annualize=True)

    tickers = prices.columns.to_list()
    weights = {}
    weights_0 = None
    for date, pd_covar in covar_dict.items():
        weights_ = wrapper_maximize_portfolio_sharpe(pd_covar=pd_covar,
                                                     means=means.loc[date, :],
                                                     constraints=constraints,
                                                     weights_0=weights_0,
                                                     solver=solver)

        weights_0 = weights_  # update for next rebalancing
        weights[date] = weights_

    weights = pd.DataFrame.from_dict(weights, orient='index')
    weights = weights.reindex(columns=tickers).fillna(0.0)
    return weights


def wrapper_maximize_portfolio_sharpe(pd_covar: pd.DataFrame,
                                      means: pd.Series,
                                      constraints: Constraints,
                                      weights_0: pd.Series = None,
                                      solver: str = 'ECOS_BB'
                                      ) -> pd.Series:
    """
    Create wrapper accounting for nans or zeros in covar matrix.
    Assets in columns/rows of covar must correspond to means.index.
    """
    vectors = dict(means=means)
    clean_covar, good_vectors = filter_covar_and_vectors_for_nans(pd_covar=pd_covar, vectors=vectors)

    constraints1 = constraints.update_with_valid_tickers(valid_tickers=clean_covar.columns.to_list(),
                                                         total_to_good_ratio=len(pd_covar.columns) / len(clean_covar.columns),
                                                         weights_0=weights_0)

    weights = cvx_maximize_portfolio_sharpe(covar=clean_covar.to_numpy(),
                                            means=good_vectors['means'].to_numpy(),
                                            constraints=constraints1,
                                            solver=solver)
    weights[np.isinf(weights)] = 0.0
    weights = pd.Series(weights, index=clean_covar.index)
    weights = weights.reindex(index=pd_covar.index).fillna(0.0)

    return weights


def cvx_maximize_portfolio_sharpe(covar: np.ndarray,
                                  means: np.ndarray,
                                  constraints: Constraints,
                                  verbose: bool = False,
                                  solver: str = 'ECOS_BB'
                                  ) -> np.ndarray:
    """
    max means^t*w / sqrt(w^t @ covar @ w)
    subject to
     1. weight_min <= w <= weight_max
    """
    n = covar.shape[0]
    z = cvx.Variable(n+1)
    w = z[:n]
    k = z[n]
    objective = cvx.Minimize(cvx.quad_form(w, covar))

    constraints_ = constraints.set_cvx_all_constraints(w=w, covar=covar, exposure_scaler=k)
    constraints_ += [means.T @ w == constraints.max_exposure]

    problem = cvx.Problem(objective, constraints_)
    problem.solve(verbose=verbose, solver=solver)

    optimal_weights = z.value

    if optimal_weights is not None:
        optimal_weights = optimal_weights[:n] / optimal_weights[n]
    else:
        print(f"not solved")
        if constraints.weights_0 is not None:
            optimal_weights = constraints.weights_0.to_numpy()
            print(f"using weights_0")
        else:
            optimal_weights = np.zeros(n)
            print(f"using zero weights")

    return optimal_weights


def estimate_rolling_ewma_means(prices: pd.DataFrame,
                                rebalancing_dates: list,
                                returns_freq: str = 'W-WED',
                                span: int = 52,
                                annualize: bool = True,
                                ) -> pd.DataFrame:
    """
    Compute expanding EWMA means at each rebalancing date.

    For each date in rebalancing_dates, computes the EWMA mean of returns
    using all data up to that date (expanding window).

    Args:
        prices: Asset price panel. Index=dates, columns=tickers.
        rebalancing_dates: List of dates at which to estimate means.
        returns_freq: Frequency for return computation (e.g., 'W-WED', 'ME').
        span: EWMA span in periods at returns_freq.
        annualize: If True, multiply means by annualisation factor.

    Returns:
        DataFrame of estimated means. Index=rebalancing_dates, columns=tickers.
    """
    returns = qis.to_returns(prices=prices, is_log_returns=True, drop_first=True, freq=returns_freq)

    # compute full EWMA mean series, then slice at rebalancing dates
    ewma_means = qis.compute_ewm(returns.to_numpy(), span=span)
    ewma_means = pd.DataFrame(ewma_means, index=returns.index, columns=returns.columns)

    if annualize:
        an_factor = qis.infer_annualisation_factor_from_df(data=returns)
        ewma_means = an_factor * ewma_means

    # select rebalancing dates that exist in returns index
    valid_dates = ewma_means.index.intersection(pd.DatetimeIndex(rebalancing_dates))
    means = ewma_means.loc[valid_dates]

    # for rebalancing dates not exactly on returns index, use ffill lookup
    if len(valid_dates) < len(rebalancing_dates):
        means = ewma_means.reindex(index=pd.DatetimeIndex(rebalancing_dates), method='ffill')

    return means