"""
example from
https://www.cvxpy.org/examples/machine_learning/lasso_regression.html
"""

import cvxpy as cvx
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import qis as qis
from typing import Optional
from enum import Enum


def solve_lasso(x: np.ndarray,
                y: np.ndarray,
                reg_lambda: float = 1e-8,
                span: Optional[int] = None,  # for weight
                verbose: bool = False,
                solver: str = 'ECOS_BB',
                nonneg: bool = False
                ) -> np.ndarray:
    """
    solve lasso for 1-dim y
    """
    assert y.ndim == 1

    if x.ndim == 1:
        n = 1
    else:
        n = x.shape[1]

    is_nan_cond = np.logical_or(np.isnan(y), np.isnan(x).any(axis=1))
    if np.any(is_nan_cond):
        x = x[is_nan_cond == False, :]
        y = y[is_nan_cond == False]
    t = x.shape[0]
    if t < 5:  # too little observations
        print(f"small number of non nans in lasso t={t}")
        return np.full(n, np.nan)

    # compute weights
    if span is not None:
        weights = qis.compute_expanding_power(n=t, power_lambda=np.sqrt(1.0 - 2.0 / (span+1.0)), reverse_columns=True)
    else:
        weights = np.ones(t)

    beta = cvx.Variable(n, nonneg=nonneg)
    objective_fun = (1.0/t)*cvx.norm2(cvx.multiply(weights, x @ beta - y))**2 + reg_lambda * cvx.norm1(beta)
    objective = cvx.Minimize(objective_fun)
    problem = cvx.Problem(objective)
    problem.solve(verbose=verbose, solver=solver)
    # problem.solve()

    estimated_beta = beta.value
    if estimated_beta is None:
        # raise ValueError(f"not solved")
        print(f"not solved")
        return np.full(n, np.nan)
    else:
        return estimated_beta


def estimate_lasso_betas(x: pd.DataFrame,
                         y: pd.DataFrame,
                         reg_lambda: float = 1e-8,
                         span: Optional[int] = None,
                         de_mean: bool = True,
                         nonneg: bool = False
                         ) -> pd.DataFrame:
    """
    for each y run univaretae lasso
    """
    x_np = x.to_numpy()
    y_np = y.to_numpy()
    assert x_np.shape[0] == y_np.shape[0]  # equal time obs
    betas = np.zeros((x_np.shape[1], y_np.shape[1]))

    if de_mean:
        if span is None:
            x_np = x_np - np.nanmean(x_np, axis=0)
            y_np = y_np - np.nanmean(y_np, axis=0)
        else:
            x_np = x_np - qis.compute_ewm(x_np, span=span)
            y_np = y_np - qis.compute_ewm(y_np, span=span)

    for idx, column in enumerate(y.columns):
        betas[:, idx] = solve_lasso(x=x_np,
                                    y=y_np[:, idx],
                                    reg_lambda=reg_lambda,
                                    span=span,
                                    nonneg=nonneg)
    betas = pd.DataFrame(betas, index=x.columns, columns=y.columns)
    return betas


def estimate_lasso_covar(x: pd.DataFrame,
                         y: pd.DataFrame,
                         covar: np.ndarray,
                         reg_lambda: float = 1e-8,
                         span: Optional[int] = None,
                         de_mean: bool = True,
                         nonneg: bool = False
                         ) -> np.ndarray:
    """
    covar = benchmarks covar N*N
    betas = benachmark * asset: N*M
    betas covar = betas.T @ covar @ betas: M*M
    """
    betas = estimate_lasso_betas(x=x, y=y, reg_lambda=reg_lambda, span=span, de_mean=de_mean, nonneg=nonneg)
    betas_np = betas.to_numpy()
    betas_covar = np.transpose(betas_np) @ covar @ betas_np
    return betas_covar


def compute_r2(x, y, beta, span: Optional[int] = None):
    t = x.shape[0]
    if span is not None:
        weights = qis.compute_expanding_power(n=t, power_lambda=np.sqrt(1.0 - 2.0 / (span+1.0)), reverse_columns=True)
    else:
        weights = np.ones(t)
    ss_res = np.linalg.norm(weights * (x @ beta - y), 2)**2
    ss_total = np.linalg.norm(weights * (y - np.nanmean(y)), 2)**2
    return 1.0 - ss_res / ss_total


def generate_data(m=100, n=20, sigma=5, density=0.2):
    "Generates data matrix X and observations Y."
    np.random.seed(1)
    beta_star = np.random.randn(n)
    idxs = np.random.choice(range(n), int((1-density)*n), replace=False)
    for idx in idxs:
        beta_star[idx] = 0
    X = np.random.randn(m,n)
    Y = X.dot(beta_star) + np.random.normal(0, sigma, size=m)
    return X, Y, beta_star


def plot_train_test_errors(train_errors, test_errors, lambd_values):
    plt.plot(lambd_values, train_errors, label="Train error")
    plt.plot(lambd_values, test_errors, label="Test error")
    plt.xscale("log")
    plt.legend(loc="upper left")
    plt.xlabel(r"$\lambda$", fontsize=16)
    plt.title("Mean Squared Error (MSE)")


class UnitTests(Enum):
    CHECK1 = 1


def run_unit_test(unit_test: UnitTests):

    if unit_test == UnitTests.CHECK1:
        m = 100
        n = 20
        sigma = 5

        X, Y, _ = generate_data(m, n, sigma)
        X_train = X[:50, :]
        Y_train = Y[:50]
        X_test = X[50:, :]
        Y_test = Y[50:]

        lambd_values = np.logspace(-2, 3, 50)
        train_errors = []
        test_errors = []
        beta_values = []
        for v in lambd_values:
            beta = solve_lasso(x=X_train, y=Y_train, reg_lambda=v)
            train_errors.append(compute_r2(X_train, Y_train, beta))
            test_errors.append(compute_r2(X_test, Y_test, beta))
            beta_values.append(beta)
        print(train_errors)
        plot_train_test_errors(train_errors, test_errors, lambd_values)

    plt.show()


if __name__ == '__main__':

    unit_test = UnitTests.CHECK1

    is_run_all_tests = False
    if is_run_all_tests:
        for unit_test in UnitTests:
            run_unit_test(unit_test=unit_test)
    else:
        run_unit_test(unit_test=unit_test)
