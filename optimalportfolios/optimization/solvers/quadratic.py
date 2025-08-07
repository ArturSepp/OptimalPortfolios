"""
portfolio optimization using quadratic objective functions
"""
# packages
import numpy as np
import pandas as pd
import cvxpy as cvx
import qis as qis
from numba import jit
from enum import Enum
from typing import Tuple, Optional, Dict

# optimalportfolios
from optimalportfolios.config import PortfolioObjective
from optimalportfolios.optimization.constraints import Constraints
from optimalportfolios.utils.filter_nans import filter_covar_and_vectors_for_nans
from optimalportfolios.covar_estimation.covar_estimator import CovarEstimator


def rolling_quadratic_optimisation(prices: pd.DataFrame,
                                   constraints0: Constraints,
                                   time_period: qis.TimePeriod,  # when we start building portfolios
                                   covar_dict: Dict[pd.Timestamp, pd.DataFrame] = None,  # can be precomputed
                                   inclusion_indicators: Optional[pd.DataFrame] = None,  # if asset is included into optimisation
                                   portfolio_objective: PortfolioObjective = PortfolioObjective.MIN_VARIANCE,
                                   covar_estimator: CovarEstimator = CovarEstimator(),  # default estimator
                                   carra: float = 1.0
                                   ) -> pd.DataFrame:
    """
    compute quadratic optimisation for portfolio_objective in [PortfolioObjective.MIN_VARIANCE,
                                                                PortfolioObjective.QUADRATIC_UTILITY]
    covar_dict: Dict[timestamp, covar matrix] can be precomputed
    portolio is rebalances at covar_dict.keys()
    """
    if covar_dict is None:  # use default ewm covar with covar_estimator
        covar_dict = covar_estimator.fit_rolling_covars(prices=prices, time_period=time_period).y_covars

    # generate rebalancing dates on the returns index
    rebalancing_schedule = list(covar_dict.keys())
    tickers = prices.columns.to_list()

    if inclusion_indicators is not None:  # reindex at rebalancing
        inclusion_indicators1 = inclusion_indicators.reindex(columns=tickers)
        inclusion_indicators1 = inclusion_indicators1.reindex(index=rebalancing_schedule, method='ffill')
    else:
        inclusion_indicators1 = pd.DataFrame(1.0, index=rebalancing_schedule, columns=tickers)

    weights = {}
    weights_0 = None
    for date, pd_covar in covar_dict.items():
        weights_ = wrapper_quadratic_optimisation(pd_covar=pd_covar,
                                                  constraints0=constraints0,
                                                  weights_0=weights_0,
                                                  portfolio_objective=portfolio_objective,
                                                  carra=carra,
                                                  inclusion_indicators=inclusion_indicators1.loc[date, :])
        weights_0 = weights_  # update for next rebalancing
        weights[date] = weights_

    weights = pd.DataFrame.from_dict(weights, orient='index')
    weights = weights.reindex(columns=tickers)
    return weights


def wrapper_quadratic_optimisation(pd_covar: pd.DataFrame,
                                   constraints0: Constraints,
                                   inclusion_indicators: pd.Series = None,
                                   portfolio_objective: PortfolioObjective = PortfolioObjective.MIN_VARIANCE,
                                   weights_0: pd.Series = None,
                                   carra: float = 1.0,
                                   solver: str = 'ECOS_BB'
                                   ) -> pd.Series:
    """
    create wrapper accounting for nans or zeros in covar matrix
    assets in columns/rows of covar must correspond to alphas.index
    """
    # filter out assets with zero variance or nans
    clean_covar, good_vectors = filter_covar_and_vectors_for_nans(pd_covar=pd_covar,
                                                                  inclusion_indicators=inclusion_indicators)

    constraints = constraints0.update_with_valid_tickers(valid_tickers=clean_covar.columns.to_list(),
                                                         total_to_good_ratio=len(pd_covar.columns) / len(clean_covar.columns),
                                                         weights_0=weights_0)

    weights = cvx_quadratic_optimisation(portfolio_objective=portfolio_objective,
                                         covar=clean_covar.to_numpy(),
                                         constraints=constraints,
                                         carra=carra,
                                         solver=solver)
    weights = pd.Series(weights, index=clean_covar.index)
    weights = weights.reindex(index=pd_covar.index).fillna(0.0)  # align with tickers
    return weights


def cvx_quadratic_optimisation(portfolio_objective: PortfolioObjective,
                               covar: np.ndarray,
                               constraints: Constraints,
                               means: np.ndarray = None,
                               verbose: bool = False,
                               solver: str = 'ECOS_BB',
                               carra: float = 1.0
                               ) -> np.ndarray:
    """
    cvx solution for max objective
    subject to linear constraints
         1. weight_min <= w <= weight_max
         2. sum(w) = 1
         3. exposure_budget_eq[0]^t*w = exposure_budget_eq[1]
    """
    # set up problem
    covar = cvx.psd_wrap(covar)  # ensure covar is semi-definite
    n = covar.shape[0]
    if constraints.is_long_only:
        nonneg = True
    else:
        nonneg = False
    w = cvx.Variable(n, nonneg=nonneg)

    portfolio_var = cvx.quad_form(w, covar)

    if portfolio_objective == PortfolioObjective.MIN_VARIANCE:
        objective_fun = -portfolio_var

    elif portfolio_objective == PortfolioObjective.QUADRATIC_UTILITY:
        if means is None:
            raise ValueError(f"means must be given")
        objective_fun = means.T @ w - 0.5 * carra * portfolio_var

    else:
        raise ValueError(f"unknown portfolio_objective")

    # set solver
    objective = cvx.Maximize(objective_fun)
    constraints_ = constraints.set_cvx_constraints(w=w, covar=covar)
    problem = cvx.Problem(objective, constraints_)
    problem.solve(verbose=verbose, solver=solver)

    optimal_weights = w.value
    if optimal_weights is None:
        # raise ValueError(f"not solved")
        print(f"not solved")
        if constraints.weights_0 is not None:
            optimal_weights = constraints.weights_0.to_numpy()
            print(f"using weights_0")
        else:
            optimal_weights = np.zeros(n)
            print(f"using zeroweights")

    return optimal_weights


def max_qp_portfolio_vol_target(portfolio_objective: PortfolioObjective,
                                covar: np.ndarray,
                                constraints: Constraints,
                                means: np.ndarray = None,
                                vol_target: float = 0.12
                                ) -> np.ndarray:
    """
    implement vol target
    """
    max_iter = 20
    sol_tol = 10e-6
    def f(lambda_n: float) -> float:
        w_n = cvx_quadratic_optimisation(portfolio_objective=portfolio_objective,
                                         covar=covar,
                                         means=means,
                                         constraints=constraints,
                                         carra=lambda_n)

        print('lambda_n='+str(lambda_n))
        print_portfolio_outputs(optimal_weights=w_n,
                                covar=covar,
                                means=means)
        target = w_n.T @ covar @ w_n - vol_target**2
        return target

    # find initials
    cov_inv = np.linalg.inv(covar)
    e = np.ones(covar.shape[0])

    if means is not None:
        a = np.sqrt(e.T@cov_inv@e/(2*vol_target**2))
        b = np.sqrt(means.T@cov_inv@means/(2*vol_target**2))
    else:
        a = np.sqrt(e.T@cov_inv@e/(2*vol_target ** 2))
        b = 100
    f_a = f(a)
    f_b = f(b)

    print((f"initial: {[f_a, f_b]}"))
    if np.sign(f_a) == np.sign(f_b):
        raise ValueError(f"the same signs: {[f_a, f_b]}")

    lambda_n = 0.5 * (a + b)
    for it in range(max_iter):
        lambda_n = 0.5 * (a + b) #new midpoint
        f_n = f(lambda_n)

        if (np.abs(f_n) <= sol_tol) or (np.abs((b-a)/2.0) < sol_tol):
            break
        if np.sign(f_n) == np.sign(f_a):
            a = lambda_n
            f_a = f_n
        else:
            b = lambda_n
        print('it='+str(it))

    w_n = cvx_quadratic_optimisation(portfolio_objective=portfolio_objective,
                                     covar=covar,
                                     means=means,
                                     constraints=constraints,
                                     carra=lambda_n)
    print_portfolio_outputs(optimal_weights=w_n,
                            covar=covar,
                            means=means)
    return w_n


@jit(nopython=True)
def solve_analytic_log_opt(covar: np.ndarray,
                           means: np.ndarray,
                           exposure_budget_eq: Tuple[np.ndarray, float] = None,
                           gamma: float = 1.0
                           ) -> np.ndarray:

    """
    analytic solution for max{means^t*w - 0.5*gamma*w^t*covar*w}
    subject to exposure_budget_eq[0]^t*w = exposure_budget_eq[1]
    """
    sigma_i = np.linalg.inv(covar)

    if exposure_budget_eq is not None:

        # get constraints
        a = exposure_budget_eq[0]
        # if len(a) != covar.shape[0]:
            # raise ValueError(f"dimensions of exposure constraint {a} not matichng covar dimensions")
        a0 = exposure_budget_eq[1]
        # if not isinstance(a0, float):
            # raise ValueError(f"a0 = {a0} must be single float")

        a_sigma_a = a.T @ sigma_i @ a
        a_sigma_mu = a.T @ sigma_i @ means
        l_lambda = (-gamma*a0+a_sigma_mu) / a_sigma_a
        optimal_weights = (1.0/gamma) * sigma_i @ (means - l_lambda * a)

    else:
        optimal_weights = (1.0/gamma) * sigma_i @ means

    return optimal_weights


def print_portfolio_outputs(optimal_weights: np.ndarray,
                            covar: np.ndarray,
                            means: np.ndarray) -> None:

    mean = means.T @ optimal_weights
    vol = np.sqrt(optimal_weights.T @ covar @ optimal_weights)
    sharpe = mean / vol
    inst_sharpes = means / np.sqrt(np.diag(covar))
    sharpe_weighted = inst_sharpes.T @ (optimal_weights / np.sum(optimal_weights))

    line_str = (f"expected={mean: 0.2%}, "
                f"vol={vol: 0.2%}, "
                f"Sharpe={sharpe: 0.2f}, "
                f"weighted Sharpe={sharpe_weighted: 0.2f}, "
                f"inst Sharpes={np.array2string(inst_sharpes, precision=2)}, "
                f"weights={np.array2string(optimal_weights, precision=2)}")

    print(line_str)


class LocalTests(Enum):
    MIN_VAR = 1
    MAX_UTILITY = 2
    EFFICIENT_FRONTIER = 3
    MAX_UTILITY_VOL_TARGET = 4
    SHARPE = 5
    REGIME_SHARPE = 6


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real data and generate reports.
    Use for quick verification during development.
    """
    
    import seaborn as sns
    import matplotlib.pyplot as plt
    from optimalportfolios.optimization.solvers.max_sharpe import cvx_maximize_portfolio_sharpe

    means = np.array([-0.01, 0.05])  # sharpe = [-.1, 0.5]
    covar = np.array([[0.2**2, -0.0075],
                      [-0.0075, 0.1**2]])
    constraints = Constraints()

    if local_test == LocalTests.MIN_VAR:

        weight_min = np.array([0.0, 0.0])
        weight_max = np.array([10.0, 10.0])

        optimal_weights = cvx_quadratic_optimisation(portfolio_objective=PortfolioObjective.MIN_VARIANCE,
                                                     covar=covar,
                                                     means=means,
                                                     constraints=constraints)

        print_portfolio_outputs(optimal_weights=optimal_weights,
                                covar=covar,
                                means=means)

    elif local_test == LocalTests.MAX_UTILITY:

        gamma = 50*np.trace(covar)
        optimal_weights = cvx_quadratic_optimisation(portfolio_objective=PortfolioObjective.QUADRATIC_UTILITY,
                                                     covar=covar,
                                                     means=means,
                                                     constraints=constraints,
                                                     carra=gamma)

        print_portfolio_outputs(optimal_weights=optimal_weights,
                                covar=covar,
                                means=means)

    elif local_test == LocalTests.EFFICIENT_FRONTIER:

        portfolio_mus = []
        portfolio_vols = []
        portfolio_sharpes = []
        w_lambdas = []
        lang_lambdas = np.arange(0.5, 100.0, 1.0)
        exposure_budget_eq = (np.ones_like(means), 1.0)

        for lang_lambda in lang_lambdas:
            is_analytic = False
            if is_analytic:
                w_lambda = solve_analytic_log_opt(covar=covar,
                                                  means=means,
                                                  exposure_budget_eq=exposure_budget_eq,
                                                  gamma=lang_lambda)
            else:
                w_lambda = cvx_quadratic_optimisation(portfolio_objective=PortfolioObjective.QUADRATIC_UTILITY,
                                                      covar=covar,
                                                      means=means,
                                                      constraints=constraints,
                                                      carra=lang_lambda)

            portfolio_vol = np.sqrt(w_lambda.T@covar@w_lambda)
            portfolio_sharpe = means.T @ w_lambda / portfolio_vol
            portfolio_mus.append(means.T @ w_lambda)
            w_lambdas.append(w_lambda)
            portfolio_vols.append(portfolio_vol)
            portfolio_sharpes.append(portfolio_sharpe)

        portfolio_return = pd.Series(portfolio_mus, index=lang_lambdas).rename('mean')
        portfolio_vol = pd.Series(portfolio_vols, index=lang_lambdas).rename('vol')
        portfolio_sharpe = pd.Series(portfolio_sharpes, index=lang_lambdas).rename('Sharpe')
        w_lambdas = pd.DataFrame(w_lambdas, index=lang_lambdas)
        protfolio_data = pd.concat([portfolio_return, portfolio_vol, portfolio_sharpe, w_lambdas], axis=1)
        print(protfolio_data)
        fig, axs = plt.subplots(2, 1, figsize=(15, 12))
        sns.lineplot(x='vol', y='mean', data=protfolio_data, ax=axs[0])
        sns.lineplot(data=protfolio_data[['mean', 'vol']], ax=axs[1])

    elif local_test == LocalTests.MAX_UTILITY_VOL_TARGET:
        optimal_weights = max_qp_portfolio_vol_target(portfolio_objective=PortfolioObjective.QUADRATIC_UTILITY,
                                                      covar=covar,
                                                      means=means,
                                                      vol_target=0.08)

        print_portfolio_outputs(optimal_weights=optimal_weights,
                                covar=covar,
                                means=means)

    elif local_test == LocalTests.SHARPE:

        portfolio_mus = []
        portfolio_vols = []
        portfolio_sharpes = []
        exposure_budget_eq = (np.ones_like(means), 1.0)

        lang_lambdas = np.arange(1.0, 20.0, 1.0)
        for lang_lambda in lang_lambdas:
            is_analytic = True
            if is_analytic:
                w_lambda = solve_analytic_log_opt(covar=covar,
                                                  means=means,
                                                  exposure_budget_eq=exposure_budget_eq,
                                                  gamma=lang_lambda)
            else:
                w_lambda = cvx_quadratic_optimisation(portfolio_objective=PortfolioObjective.QUADRATIC_UTILITY,
                                                      covar=covar,
                                                      means=means,
                                                      constraints=constraints,
                                                      carra=lang_lambda)

            print(f"portfolio with lambda = {lang_lambda}")
            print_portfolio_outputs(optimal_weights=w_lambda,
                                    covar=covar,
                                    means=means)

            portfolio_vol = np.sqrt(w_lambda.T@covar@w_lambda)
            portfolio_sharpe = means.T @ w_lambda / portfolio_vol
            portfolio_mus.append(means.T @ w_lambda)
            portfolio_vols.append(portfolio_vol)
            portfolio_sharpes.append(portfolio_sharpe)

        portfolio_return = pd.Series(portfolio_mus, index=lang_lambdas)
        portfolio_vol = pd.Series(portfolio_vols, index=lang_lambdas)
        portfolio_sharpe = pd.Series(portfolio_sharpes, index=lang_lambdas)
        protfolio_data = pd.concat([portfolio_return, portfolio_vol, portfolio_sharpe], axis=1)
        print(protfolio_data)

        opt_sharpe_w = cvx_maximize_portfolio_sharpe(covar=covar,
                                                     means=means,
                                                     constraints=constraints)

        print(f"exact solution")
        print_portfolio_outputs(optimal_weights=opt_sharpe_w,
                                covar=covar,
                                means=means)

    elif local_test == LocalTests.REGIME_SHARPE:

        # case of two assets:
        # inputs:
        g = 3

        # individual
        sharpes = np.array((0.4, 0.3))
        betas_port = np.array((1.0, 0.8, 1.0))
        betas_cta = np.array((-1.0, 0.25, 0.25))
        idio_vols = np.array((0.01, 0.1))

        # factor
        p_regimes = np.array((0.16, 0.68, 0.16))
        factor_vol = 0.15

        betas_matrix = np.stack((betas_port, betas_cta))
        print(betas_matrix)

        n = betas_matrix.shape[0]
        covar = np.zeros((n, n))
        for g_ in range(g):
            b = betas_matrix[:, g_]
            covar_regime = np.outer(b, b)
            print(f"covar_regime=\n{covar_regime}")
            covar += covar_regime*p_regimes[g_]

        covar = (factor_vol**2) * covar + np.diag(idio_vols**2)
        print(f"t_covar_regime=\n{covar}")

        implied_vols = np.sqrt(np.diag(covar))
        print(f"implied_vols=\n{implied_vols}")

        means = sharpes * implied_vols
        print(f"implied_means=\n{means}")

        # invest 100% in first asset
        exposure_budget_eq = (np.array([1.0, 0.0]), 1.0)
        optimal_weights = cvx_maximize_portfolio_sharpe(covar=covar,
                                                        means=means,
                                                        constraints=constraints)

        print_portfolio_outputs(optimal_weights=optimal_weights,
                                covar=covar,
                                means=means)

    plt.show()


if __name__ == '__main__':

    run_local_test(local_test=LocalTests.SHARPE)
