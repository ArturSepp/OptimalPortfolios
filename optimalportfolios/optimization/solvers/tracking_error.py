"""
run optimisation for given
 1. benchmark weights = w_bench
 2. given current weights = w_0
 3. vector of alphas
 4. covariance matrix = Sigma

max alpha@w
such that
w @ Sigma @ w.t <= tracking_err_vol_constraint
sum(abs(w-w_0)) <= turnover_constraint
sum(w) = 1 # exposure constraint
w >= 0  # long only constraint
"""

# packages
import numpy as np
import pandas as pd
import cvxpy as cvx
import matplotlib.pyplot as plt
import qis as qis
from enum import Enum
from typing import Tuple, Optional

# optimalportfolios
from optimalportfolios.utils.filter_nans import filter_covar_and_vectors_for_nans


def withnans_maximize_alpha_over_tracking_error(covar: pd.DataFrame,
                                                alphas: pd.Series,
                                                benchmark_weights: pd.Series,
                                                min_weights: pd.Series,
                                                max_weights: pd.Series,
                                                weights_0: pd.Series,  # for turnover constraints
                                                is_long_only: bool = True,
                                                max_leverage: float = None,  # for long short portfolios
                                                tracking_err_vol_constraint: float = 0.05,  # annualised sqrt tracking error
                                                turnover_constraint: Optional[float] = 0.5,
                                                solver: str = 'ECOS'
                                                ) -> pd.Series:
    """
    create wrapper accounting for nans in covar matrix
    assets in columns/rows of covar must correspond to alphas.index
    """
    # filter out assets with zero variance or nans
    vectors = dict(alphas=alphas, benchmark_weights=benchmark_weights,
                   min_weights=min_weights, max_weights=max_weights,
                   weights_0=weights_0)
    covar_pd, good_vectors = filter_covar_and_vectors_for_nans(covar=covar, vectors=vectors)

    weights = maximize_alpha_over_tracking_error(covar=covar_pd.to_numpy(),
                                                 alphas=good_vectors['alphas'].to_numpy(),
                                                 benchmark_weights=good_vectors['benchmark_weights'].to_numpy(),
                                                 min_weights=good_vectors['min_weights'].to_numpy(),
                                                 max_weights=good_vectors['max_weights'].to_numpy(),
                                                 weights_0=good_vectors['weights_0'].to_numpy(),
                                                 is_long_only=is_long_only,
                                                 max_leverage=max_leverage,
                                                 tracking_err_vol_constraint=tracking_err_vol_constraint,
                                                 turnover_constraint=turnover_constraint,
                                                 solver=solver)
    weights = pd.Series(weights, index=covar_pd.index)
    weights = weights.reindex(index=covar.index).fillna(0.0)  # align with tickers

    return weights


def maximize_alpha_over_tracking_error(covar: np.ndarray,
                                       alphas: np.ndarray = None,
                                       benchmark_weights: np.ndarray = None,
                                       min_weights: np.ndarray = None,
                                       max_weights: np.ndarray = None,
                                       is_long_only: bool = True,
                                       max_leverage: float = None,  # for long short portfolios
                                       tracking_err_vol_constraint: float = 0.05,  # annualised sqrt tracking error
                                       turnover_constraint: Optional[float] = 0.5,
                                       weights_0: np.ndarray = None,
                                       solver: str = 'ECOS'
                                       ) -> np.ndarray:
    """
    max alpha@w
    such that
    w @ Sigma @ w.t <= tracking_err_vol_constraint
    sum(abs(w-w_0)) <= turnover_constraint
    sum(w) = 1 # exposure constraint
    w >= 0  # long only constraint

    subject to linear constraints
         1. weight_min <= w <= weight_max
         2. sum(w) = 1
         3. exposure_budget_eq[0]^t*w = exposure_budget_eq[1]
    """

    n = covar.shape[0]
    w = cvx.Variable(n)
    tracking_error_var = cvx.quad_form(w-benchmark_weights, covar)

    objective_fun = alphas.T @ (w - benchmark_weights)

    objective = cvx.Maximize(objective_fun)

    # add constraints
    constraints = []
    # gross_notional = 1:
    constraints = constraints + [cvx.sum(w) == 1]

    # tracking error constraint
    constraints += [tracking_error_var <= tracking_err_vol_constraint ** 2]  # variance constraint

    # turnover_constraint:
    if turnover_constraint is not None:
        if weights_0 is None:
            raise ValueError(f"weights_0 must be given")
        constraints += [cvx.norm(w-weights_0, 1) <= turnover_constraint]

    if is_long_only:
        constraints = constraints + [w >= 0.0]
    if min_weights is not None:
        constraints = constraints + [w >= min_weights]
    if max_weights is not None:
        constraints = constraints + [w <= max_weights]
    #if exposure_budget_eq is not None:
    #    constraints = constraints + [exposure_budget_eq[0] @ w == exposure_budget_eq[1]]
    if max_leverage is not None:
        constraints = constraints + [cvx.norm(w, 1) <= max_leverage]

    problem = cvx.Problem(objective, constraints)
    problem.solve(verbose=False, solver=solver)

    optimal_weights = w.value
    if optimal_weights is None:
        raise ValueError(f"not solved")

    return optimal_weights


def compute_te_turnover(covar: np.ndarray,
                        benchmark_weights: pd.Series,
                        weights: pd.Series,
                        weights_0: pd.Series,
                        alpha: pd.Series
                        ) -> Tuple[float, float, float, float, float]:
    weight_diff = weights.subtract(benchmark_weights)
    benchmark_vol = np.sqrt(benchmark_weights @ covar @ benchmark_weights.T)
    port_vol = np.sqrt(weights @ covar @ weights.T)
    te_vol = np.sqrt(weight_diff @ covar @ weight_diff.T)
    turnover = np.nansum(np.abs(weights.subtract(weights_0)))
    port_alpha = alpha @ weights
    return te_vol, turnover, port_alpha, port_vol, benchmark_vol


def fetch_benchmark_universe_data() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    define custom universe with asset class grouping
    """
    # for data
    import yfinance as yf

    universe_data = dict(SPY='Equities',
                         QQQ='Equities',
                         EEM='Equities',
                         TLT='Bonds',
                         IEF='Bonds',
                         LQD='Credit',
                         HYG='HighYield',
                         GLD='Gold')
    benchmark_weights = dict(SPY=0.1,
                             QQQ=0.1,
                             EEM=0.1,
                             TLT=0.2,
                             IEF=0.2,
                             LQD=0.1,
                             HYG=0.1,
                             GLD=0.1)

    tickers = list(universe_data.keys())
    group_data = pd.Series(universe_data)  # for portfolio reporting
    benchmark_weights = pd.Series(benchmark_weights)
    prices = yf.download(tickers=tickers, start=None, end=None, ignore_tz=True)['Adj Close'][tickers]
    prices = prices.asfreq('B', method='ffill').dropna()
    benchmark_prices = prices[['SPY', 'TLT']]
    return prices, benchmark_prices, benchmark_weights, group_data


class UnitTests(Enum):
    TRACKING_ERROR = 1
    TRACKING_ERROR_GRID = 2


def run_unit_test(unit_test: UnitTests):

    prices, benchmark_prices, benchmark_weights, group_data = fetch_benchmark_universe_data()

    # 1y momentum
    momentum = prices.divide(prices.shift(260)) - 1.0
    alphas = np.divide(np.subtract(momentum, np.nanmean(momentum, keepdims=True, axis=1)),
                       np.nanstd(momentum, keepdims=True, axis=1), where=np.isfinite(momentum.to_numpy()))
    alpha = alphas.iloc[-1, :]
    print(f"alpha=\n{alpha}")

    covar = 52.0 * qis.compute_masked_covar_corr(data=qis.to_returns(prices, freq='ME'), is_covar=True)
    print(f"covar=\n{covar}")

    min_weights = np.zeros(len(prices.columns))
    max_weights = np.ones(len(prices.columns))

    if unit_test == UnitTests.TRACKING_ERROR:

        weights = maximize_alpha_over_tracking_error(covar=covar.to_numpy(),
                                                     alphas=alpha.to_numpy(),
                                                     benchmark_weights=benchmark_weights.to_numpy(),
                                                     min_weights=min_weights,
                                                     max_weights=max_weights,
                                                     tracking_err_vol_constraint=0.06,
                                                     turnover_constraint=0.75,
                                                     weights_0=benchmark_weights.to_numpy())

        weights = pd.Series(weights, index=prices.columns)
        df_weight = pd.concat([benchmark_weights.rename('benchmark'),
                               weights.rename('portfolio'),
                               alpha.rename('alpha')],
                              axis=1)
        print(f"df_weight=\n{df_weight}")

        te_vol, turnover, alpha, port_vol, benchmark_vol = compute_te_turnover(covar=covar,
                                                                               benchmark_weights=benchmark_weights,
                                                                               weights=weights,
                                                                               weights_0=benchmark_weights,
                                                                               alpha=alpha)
        print(f"port_vol={port_vol:0.4f}, benchmark_vol={benchmark_vol:0.4f}, te_vol={te_vol:0.4f}, "
              f"turnover={turnover:0.4f}, alpha={alpha:0.4f}")

    elif unit_test == UnitTests.TRACKING_ERROR_GRID:
        tracking_err_vol_constraints = [0.01, 0.02, 0.03, 0.05, 0.1]
        turnover_constraints = [0.1, 0.25, 0.5, 1.0, 10.0]

        weights_grid = {}
        port_vols, te_vols, turnovers, alphas = {}, {}, {}, {}
        for tracking_err_vol_constraint in tracking_err_vol_constraints:
            port_vols_, te_vols_, turnovers_, alphas_ = {}, {}, {}, {}
            for turnover_constraint in turnover_constraints:
                port_name = f"te_vol<{tracking_err_vol_constraint:0.2f}, turnover<{turnover_constraint:0.2f}"
                weights = maximize_alpha_over_tracking_error(covar=covar.to_numpy(),
                                                             alphas=alpha.to_numpy(),
                                                             benchmark_weights=benchmark_weights.to_numpy(),
                                                             min_weights=min_weights,
                                                             max_weights=max_weights,
                                                             tracking_err_vol_constraint=tracking_err_vol_constraint,
                                                             turnover_constraint=turnover_constraint,
                                                             weights_0=benchmark_weights.to_numpy())
                weights = pd.Series(weights, index=prices.columns)
                weights_grid[port_name] = weights
                te_vol, turnover, port_alpha, port_vol, benchmark_vol = compute_te_turnover(covar=covar,
                                                                   benchmark_weights=benchmark_weights,
                                                                   weights=weights,
                                                                   weights_0=benchmark_weights,
                                                                   alpha=alpha)
                port_name_ = f"turnover<{turnover_constraint:0.2f}"
                port_vols_[port_name_] = port_vol
                te_vols_[port_name_] = te_vol
                turnovers_[port_name_] = turnover
                alphas_[port_name_] = port_alpha

            port_name_index = f"te_vol<{tracking_err_vol_constraint:0.2f}"
            port_vols[port_name_index] = pd.Series(port_vols_)
            te_vols[port_name_index] = pd.Series(te_vols_)
            turnovers[port_name_index] = pd.Series(turnovers_)
            alphas[port_name_index] = pd.Series(alphas_)
        port_vols = pd.DataFrame.from_dict(port_vols)
        te_vols = pd.DataFrame.from_dict(te_vols)
        turnovers = pd.DataFrame.from_dict(turnovers)
        alphas = pd.DataFrame.from_dict(alphas)

        print(f"port_vols=\n{port_vols}")
        print(f"te_vols=\n{te_vols}")
        print(f"turnovers=\n{turnovers}")
        print(f"alphas=\n{alphas}")


    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.TRACKING_ERROR

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
