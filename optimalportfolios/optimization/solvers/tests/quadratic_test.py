
# packages
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from enum import Enum

# optimalportfolios
from optimalportfolios.config import PortfolioObjective
from optimalportfolios.optimization.constraints import Constraints
from optimalportfolios.optimization.solvers.max_sharpe import cvx_maximize_portfolio_sharpe
from optimalportfolios.optimization.solvers.quadratic import (cvx_quadratic_optimisation,
                                                              print_portfolio_outputs,
                                                              solve_analytic_log_opt,
                                                              max_qp_portfolio_vol_target)



class LocalTests(Enum):
    MIN_VAR = 1
    MAX_UTILITY = 2
    EFFICIENT_FRONTIER = 3
    MAX_UTILITY_VOL_TARGET = 4
    SHARPE = 5
    REGIME_SHARPE = 6


def run_local_test(local_test: LocalTests):
    """Run local tests for development and debugging purposes.

    These are integration tests that download real universe and generate reports.
    Use for quick verification during development.
    """


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

        gamma = 5.0*np.trace(covar)
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

            portfolio_vol = np.sqrt(w_lambda. T @covar @w_lambda)
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

            portfolio_vol = np.sqrt(w_lambda. T @covar @w_lambda)
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
            covar += covar_regime *p_regimes[g_]

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
