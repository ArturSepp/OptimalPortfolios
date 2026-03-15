"""
Portfolio optimisation using quadratic objective functions.

Implements minimum variance and quadratic utility (mean-variance) portfolio
optimisation via CVXPY, with support for rolling rebalancing, NaN-aware
covariance filtering, and vol-targeting via bisection.

Supported objectives:
    - MIN_VARIANCE: min w' Σ w  s.t. constraints
    - QUADRATIC_UTILITY: max μ'w - (γ/2) w' Σ w  s.t. constraints

The rolling wrapper accepts pre-computed covariance matrices (from any
CovarEstimator) and rebalances at each date in the covar dict.
"""
# packages
import numpy as np
import pandas as pd
import cvxpy as cvx
from numba import jit
from typing import Tuple, Optional, Dict

# optimalportfolios
from optimalportfolios.config import PortfolioObjective
from optimalportfolios.optimization.constraints import Constraints
from optimalportfolios.utils.filter_nans import filter_covar_and_vectors_for_nans


def rolling_quadratic_optimisation(prices: pd.DataFrame,
                                   constraints: Constraints,
                                   covar_dict: Dict[pd.Timestamp, pd.DataFrame],
                                   inclusion_indicators: Optional[pd.DataFrame] = None,
                                   portfolio_objective: PortfolioObjective = PortfolioObjective.MIN_VARIANCE,
                                   carra: float = 1.0
                                   ) -> pd.DataFrame:
    """
    Compute rolling quadratic portfolio optimisation at each rebalancing date.

    At each date in ``covar_dict``, solves the quadratic programme defined by
    ``portfolio_objective`` using the pre-computed covariance matrix for that date.
    Previous-period weights are passed as warm-start (``weights_0``) to stabilise
    turnover when the solver falls back to a feasible starting point.

    The covariance matrices are produced externally by any CovarEstimator
    (EwmaCovarEstimator, FactorCovarEstimator, etc.), decoupling the estimation
    step from the optimisation step.

    Args:
        prices: Asset price panel. Index=dates, columns=tickers. Used only for
            column alignment of the output weights DataFrame.
        constraints: Portfolio constraints (long-only, weight bounds, group
            exposures, turnover limits, etc.).
        covar_dict: Pre-computed covariance matrices keyed by rebalancing date.
            Typically produced by ``estimator.fit_rolling_covars()``.
            The dict keys define the rebalancing schedule.
        inclusion_indicators: Optional binary DataFrame (index=dates, columns=tickers)
            indicating whether each asset is eligible for inclusion at each date.
            Values of 0 exclude the asset from the optimisation at that date.
            If None, all assets are included at all dates. Reindexed to rebalancing
            dates via forward-fill.
        portfolio_objective: Optimisation objective. One of:
            - ``PortfolioObjective.MIN_VARIANCE``: minimise portfolio variance.
            - ``PortfolioObjective.QUADRATIC_UTILITY``: maximise μ'w - (γ/2) w'Σw.
        carra: Constant absolute risk aversion coefficient γ for QUADRATIC_UTILITY.
            Higher values produce more conservative (lower-variance) portfolios.
            Ignored for MIN_VARIANCE.

    Returns:
        DataFrame of portfolio weights. Index=rebalancing dates from covar_dict,
        columns=tickers aligned to ``prices.columns``. Missing assets filled with 0.
    """
    rebalancing_schedule = list(covar_dict.keys())
    tickers = prices.columns.to_list()

    # align inclusion indicators to rebalancing dates via forward-fill
    if inclusion_indicators is not None:
        inclusion_indicators1 = inclusion_indicators.reindex(columns=tickers)
        inclusion_indicators1 = inclusion_indicators1.reindex(index=rebalancing_schedule, method='ffill')
    else:
        inclusion_indicators1 = pd.DataFrame(1.0, index=rebalancing_schedule, columns=tickers)

    weights = {}
    weights_0 = None
    for date, pd_covar in covar_dict.items():
        weights_ = wrapper_quadratic_optimisation(
            pd_covar=pd_covar,
            constraints=constraints,
            weights_0=weights_0,
            portfolio_objective=portfolio_objective,
            carra=carra,
            inclusion_indicators=inclusion_indicators1.loc[date, :]
        )
        weights_0 = weights_  # warm-start next period with current solution
        weights[date] = weights_

    weights = pd.DataFrame.from_dict(weights, orient='index')
    weights = weights.reindex(columns=tickers)
    return weights


def wrapper_quadratic_optimisation(pd_covar: pd.DataFrame,
                                   constraints: Constraints,
                                   inclusion_indicators: pd.Series = None,
                                   portfolio_objective: PortfolioObjective = PortfolioObjective.MIN_VARIANCE,
                                   weights_0: pd.Series = None,
                                   carra: float = 1.0,
                                   solver: str = 'ECOS_BB'
                                   ) -> pd.Series:
    """
    Single-date quadratic optimisation with NaN/zero-variance filtering.

    Removes assets with NaN or zero diagonal entries in the covariance matrix,
    solves the reduced problem, and maps weights back to the full asset universe
    (excluded assets receive zero weight).

    Args:
        pd_covar: Covariance matrix (N x N) as DataFrame.
        constraints: Portfolio constraints.
        inclusion_indicators: Binary series indicating asset eligibility.
            Assets with value 0 are excluded from optimisation.
        portfolio_objective: MIN_VARIANCE or QUADRATIC_UTILITY.
        weights_0: Previous-period weights for warm-start / fallback.
        carra: Risk aversion coefficient for QUADRATIC_UTILITY.
        solver: CVXPY solver name.

    Returns:
        Portfolio weights as pd.Series aligned to pd_covar.index.
    """
    # filter out assets with zero variance, NaN entries, or excluded by indicators
    clean_covar, good_vectors = filter_covar_and_vectors_for_nans(
        pd_covar=pd_covar,
        inclusion_indicators=inclusion_indicators
    )

    # rescale constraints (e.g., group exposure budgets) to account for reduced universe
    constraints1 = constraints.update_with_valid_tickers(
        valid_tickers=clean_covar.columns.to_list(),
        total_to_good_ratio=len(pd_covar.columns) / len(clean_covar.columns),
        weights_0=weights_0
    )

    weights = cvx_quadratic_optimisation(
        portfolio_objective=portfolio_objective,
        covar=clean_covar.to_numpy(),
        constraints=constraints1,
        carra=carra,
        solver=solver
    )
    weights[np.isinf(weights)] = 0.0
    weights = pd.Series(weights, index=clean_covar.index)
    weights = weights.reindex(index=pd_covar.index).fillna(0.0)
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
    Solve quadratic portfolio optimisation via CVXPY.

    For MIN_VARIANCE:
        max  -w' Σ w
        s.t. constraints

    For QUADRATIC_UTILITY:
        max  μ'w - (γ/2) w' Σ w
        s.t. constraints

    The covariance matrix is wrapped with ``cvx.psd_wrap`` to handle
    near-singular matrices that may not pass CVXPY's PSD check.

    Args:
        portfolio_objective: MIN_VARIANCE or QUADRATIC_UTILITY.
        covar: Covariance matrix (N x N) as numpy array.
        constraints: Portfolio constraints (bounds, exposures, turnover).
        means: Expected returns vector (N,). Required for QUADRATIC_UTILITY.
        verbose: If True, print CVXPY solver diagnostics.
        solver: CVXPY solver name.
        carra: Risk aversion coefficient γ. Higher values penalise variance more.

    Returns:
        Optimal weights (N,) as numpy array. Falls back to weights_0 or zeros
        if the solver fails.
    """
    covar = cvx.psd_wrap(covar)
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
            raise ValueError(f"means must be given for QUADRATIC_UTILITY objective")
        objective_fun = means.T @ w - 0.5 * carra * portfolio_var

    else:
        raise ValueError(f"unsupported portfolio_objective: {portfolio_objective}")

    objective = cvx.Maximize(objective_fun)
    constraints_ = constraints.set_cvx_all_constraints(w=w, covar=covar)
    problem = cvx.Problem(objective, constraints_)
    problem.solve(verbose=verbose, solver=solver)

    optimal_weights = w.value
    if optimal_weights is None:
        print(f"not solved")
        if constraints.weights_0 is not None:
            optimal_weights = constraints.weights_0.to_numpy()
            print(f"using weights_0")
        else:
            optimal_weights = np.zeros(n)
            print(f"using zero weights")

    return optimal_weights


def max_qp_portfolio_vol_target(portfolio_objective: PortfolioObjective,
                                covar: np.ndarray,
                                constraints: Constraints,
                                means: np.ndarray = None,
                                vol_target: float = 0.12
                                ) -> np.ndarray:
    """
    Solve quadratic optimisation with a portfolio volatility target via bisection.

    Finds the risk aversion parameter γ such that the optimal portfolio
    achieves exactly ``vol_target`` by bisecting over γ in
    ``cvx_quadratic_optimisation``. The bisection brackets are initialised
    analytically from the unconstrained solution.

    This is useful for constructing efficient frontier points at a specified
    risk level, or for vol-targeting in mean-variance allocation.

    Args:
        portfolio_objective: MIN_VARIANCE or QUADRATIC_UTILITY.
        covar: Covariance matrix (N x N).
        constraints: Portfolio constraints.
        means: Expected returns (N,). Required for QUADRATIC_UTILITY.
        vol_target: Target portfolio volatility (annualised).

    Returns:
        Optimal weights (N,) achieving the target volatility.

    Raises:
        ValueError: If bisection brackets have the same sign (no root exists).
    """
    max_iter = 20
    sol_tol = 10e-6

    def f(lambda_n: float) -> float:
        w_n = cvx_quadratic_optimisation(
            portfolio_objective=portfolio_objective,
            covar=covar,
            means=means,
            constraints=constraints,
            carra=lambda_n
        )
        print('lambda_n='+str(lambda_n))
        print_portfolio_outputs(optimal_weights=w_n, covar=covar, means=means)
        target = w_n.T @ covar @ w_n - vol_target**2
        return target

    # initialise bisection brackets from unconstrained analytics
    cov_inv = np.linalg.inv(covar)
    e = np.ones(covar.shape[0])

    if means is not None:
        a = np.sqrt(e.T @ cov_inv @ e / (2 * vol_target**2))
        b = np.sqrt(means.T @ cov_inv @ means / (2 * vol_target**2))
    else:
        a = np.sqrt(e.T @ cov_inv @ e / (2 * vol_target**2))
        b = 100

    f_a = f(a)
    f_b = f(b)

    print(f"initial: {[f_a, f_b]}")
    if np.sign(f_a) == np.sign(f_b):
        raise ValueError(f"the same signs: {[f_a, f_b]}")

    # bisection loop
    lambda_n = 0.5 * (a + b)
    for it in range(max_iter):
        lambda_n = 0.5 * (a + b)
        f_n = f(lambda_n)

        if (np.abs(f_n) <= sol_tol) or (np.abs((b - a) / 2.0) < sol_tol):
            break
        if np.sign(f_n) == np.sign(f_a):
            a = lambda_n
            f_a = f_n
        else:
            b = lambda_n
        print('it=' + str(it))

    w_n = cvx_quadratic_optimisation(
        portfolio_objective=portfolio_objective,
        covar=covar,
        means=means,
        constraints=constraints,
        carra=lambda_n
    )
    print_portfolio_outputs(optimal_weights=w_n, covar=covar, means=means)
    return w_n


@jit(nopython=True)
def solve_analytic_log_opt(covar: np.ndarray,
                           means: np.ndarray,
                           exposure_budget_eq: Tuple[np.ndarray, float] = None,
                           gamma: float = 1.0
                           ) -> np.ndarray:
    """
    Analytic solution for the unconstrained quadratic utility problem.

    Solves: max μ'w - (γ/2) w'Σw  subject to a'w = a₀

    The closed-form solution is:
        w* = (1/γ) Σ⁻¹ (μ - λa)
    where λ = (-γa₀ + a'Σ⁻¹μ) / (a'Σ⁻¹a) is the Lagrange multiplier.

    Without the equality constraint:
        w* = (1/γ) Σ⁻¹ μ

    Compiled with numba for use in tight loops (e.g., efficient frontier
    computation across many γ values).

    Args:
        covar: Covariance matrix (N x N).
        means: Expected returns (N,).
        exposure_budget_eq: Tuple (a, a₀) defining the equality constraint a'w = a₀.
            Typically a = ones(N), a₀ = 1.0 for full investment.
        gamma: Risk aversion coefficient.

    Returns:
        Optimal weights (N,).
    """
    sigma_i = np.linalg.inv(covar)

    if exposure_budget_eq is not None:
        a = exposure_budget_eq[0]
        a0 = exposure_budget_eq[1]
        a_sigma_a = a.T @ sigma_i @ a
        a_sigma_mu = a.T @ sigma_i @ means
        l_lambda = (-gamma * a0 + a_sigma_mu) / a_sigma_a
        optimal_weights = (1.0 / gamma) * sigma_i @ (means - l_lambda * a)
    else:
        optimal_weights = (1.0 / gamma) * sigma_i @ means

    return optimal_weights


def print_portfolio_outputs(optimal_weights: np.ndarray,
                            covar: np.ndarray,
                            means: np.ndarray) -> None:
    """
    Print portfolio diagnostics: expected return, vol, Sharpe, and weights.

    Args:
        optimal_weights: Portfolio weights (N,).
        covar: Covariance matrix (N x N).
        means: Expected returns (N,).
    """
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
