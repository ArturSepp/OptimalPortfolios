"""
Alpha-maximising portfolio optimisation with tracking error constraints.

Solves the tactical asset allocation (TAA) problem relative to a benchmark:

    max_w  α'(w - w_b)

    s.t.   (w - w_b)' Σ (w - w_b) <= TE²_max   (tracking error budget)
           ||w - w_0||_1 <= TO_max               (turnover limit, optional)
           1'w = 1                                (full investment)
           w >= 0                                 (long-only, optional)
           w_min <= w <= w_max                    (weight bounds)

where α is the vector of alpha signals (active return views), w_b is the
benchmark (SAA) weight vector, TE_max is the tracking error budget, and
w_0 is the previous-period portfolio for turnover control.

Two formulations are supported:

1. **Hard constraints** (``ConstraintEnforcementType.HARD_CONSTRAINTS``):
   The tracking error and turnover limits are enforced as explicit CVXPY
   constraints. The objective is purely linear (maximise active alpha).
   This is appropriate when the risk budget is strict (e.g., mandated TE
   limit from an investment committee).

2. **Utility penalties** (``ConstraintEnforcementType.UTILITY_CONSTRAINTS``):
   The tracking error and turnover are penalised in the objective:

       max_w  α'(w - w_b) - λ_TE (w - w_b)'Σ(w - w_b) - λ_TO ||w - w_0||_1

   This is smoother and avoids infeasibility when hard constraints conflict,
   but requires calibrating the penalty weights λ_TE and λ_TO.

The separation between SAA (benchmark_weights) and TAA (active tilts) is
central to the ROSAA framework: the SAA provides the strategic anchor,
and the TAA overlay tilts within a risk budget toward alpha opportunities.

Reference:
    Sepp A., Ossa I., and Kastenholz M. (2026),
    "Robust Optimization of Strategic and Tactical Asset Allocation
    for Multi-Asset Portfolios",
    The Journal of Portfolio Management, 52(4), 86-120.
    Available at https://www.pm-research.com/content/iijpormgmt/52/4/86
"""
import numpy as np
import pandas as pd
import cvxpy as cvx
from typing import Optional, Union, Dict

# optimalportfolios
from optimalportfolios.optimization.constraints import Constraints, ConstraintEnforcementType
from optimalportfolios.utils.filter_nans import filter_covar_and_vectors_for_nans
from optimalportfolios.utils.portfolio_funcs import compute_portfolio_risk_contribution_outputs
from optimalportfolios.optimization.config import OptimiserConfig


def rolling_maximise_alpha_over_tre(prices: pd.DataFrame,
                                    alphas: Optional[pd.DataFrame],
                                    constraints: Constraints,
                                    benchmark_weights: Union[pd.Series, pd.DataFrame],
                                    covar_dict: Dict[pd.Timestamp, pd.DataFrame],
                                    rebalancing_indicators: pd.DataFrame = None,
                                    optimiser_config: OptimiserConfig = OptimiserConfig()
                                    ) -> pd.DataFrame:
    """
    Compute rolling alpha-maximising portfolios with tracking error control.

    At each rebalancing date (defined by the keys of ``covar_dict``), solves:

        max_w  α_t'(w - w_b,t)   s.t.  TE(w, w_b,t) <= TE_max,  constraints

    where α_t is the alpha signal and w_b,t is the benchmark at date t.

    The benchmark can be static (pd.Series, constant across dates) or
    time-varying (pd.DataFrame, e.g., from a rolling SAA). Both are
    forward-filled to the rebalancing schedule.

    Alphas of None produce a pure tracking-error-minimising portfolio
    (minimum TE to benchmark), which is equivalent to the SAA itself
    when the feasible set contains the benchmark.

    When the solver returns all-zero weights (infeasible problem), the
    warm-start is reset to None for the next period to avoid propagating
    a degenerate starting point.

    Args:
        prices: Asset price panel. Index=dates, columns=tickers. Used only
            for column alignment of the output weights DataFrame.
        alphas: Alpha signals per asset. Index=dates, columns=tickers.
            Forward-filled to rebalancing dates. None for pure benchmark tracking.
        constraints: Portfolio constraints including tracking error budget,
            turnover limits, weight bounds, and group exposures.
        benchmark_weights: SAA benchmark weights. Either a pd.Series (static
            benchmark, same weights at every date) or pd.DataFrame (time-varying
            benchmark, index=dates, columns=tickers). Forward-filled to
            rebalancing dates.
        covar_dict: Pre-computed covariance matrices keyed by rebalancing date.
            Typically produced by ``estimator.fit_rolling_covars()``.
        rebalancing_indicators: Optional binary DataFrame (index=dates,
            columns=tickers) indicating whether each asset is eligible for
            rebalancing. Assets with value 0 are frozen at previous weights.
        optimiser_config: Solver configuration (solver name, constraint scaling).

    Returns:
        DataFrame of portfolio weights. Index=rebalancing dates,
        columns=tickers aligned to ``prices.columns``.
    """
    rebalancing_dates = list(covar_dict.keys())

    # align alphas to rebalancing schedule; None alphas = pure tracking
    if alphas is not None:
        alphas = alphas.reindex(index=rebalancing_dates, method='ffill').fillna(0.0)

    # align benchmark weights: static Series -> constant DataFrame
    if isinstance(benchmark_weights, pd.DataFrame):
        benchmark_weights = benchmark_weights.reindex(index=rebalancing_dates, method='ffill').fillna(0.0)
    else:
        benchmark_weights = benchmark_weights.to_frame(
            name=rebalancing_dates[0]).T.reindex(index=rebalancing_dates, method='ffill').fillna(0.0)

    # align rebalancing indicators (default: all assets rebalanced)
    if rebalancing_indicators is not None:
        rebalancing_indicators = rebalancing_indicators.reindex(index=rebalancing_dates).fillna(0.0)

    weights = {}
    weights_0 = None  # no warm-start for first period (relaxes turnover constraint)

    for date, pd_covar in covar_dict.items():
        rebalancing_indicators_t = (
            rebalancing_indicators.loc[date, :] if rebalancing_indicators is not None else None
        )
        alphas_t = alphas.loc[date, :] if alphas is not None else None

        weights_ = wrapper_maximise_alpha_over_tre(
            pd_covar=pd_covar,
            alphas=alphas_t,
            benchmark_weights=benchmark_weights.loc[date, :],
            constraints=constraints,
            rebalancing_indicators=rebalancing_indicators_t,
            weights_0=weights_0,
            optimiser_config=optimiser_config
        )

        # reset warm-start on infeasible solutions to avoid propagation
        if np.all(np.equal(weights_, 0.0)):
            weights_0 = None
        else:
            weights_0 = weights_
        weights[date] = weights_

    weights = pd.DataFrame.from_dict(weights, orient='index')
    weights = weights.reindex(columns=prices.columns.to_list())
    return weights


def wrapper_maximise_alpha_over_tre(pd_covar: pd.DataFrame,
                                    alphas: Optional[pd.Series],
                                    benchmark_weights: pd.Series,
                                    constraints: Constraints,
                                    weights_0: pd.Series = None,
                                    rebalancing_indicators: pd.Series = None,
                                    detailed_output: bool = False,
                                    optimiser_config: OptimiserConfig = OptimiserConfig(),
                                    verbose: bool = False
                                    ) -> Union[pd.Series, pd.DataFrame]:
    """
    Single-date alpha-over-TE optimisation with NaN filtering and routing.

    Performs three steps:

    1. **Filter**: remove assets with NaN or zero variance from the covariance
       matrix and alpha vector.
    2. **Route**: select the solver based on ``constraint_enforcement_type``:
       - HARD_CONSTRAINTS → ``cvx_maximise_alpha_over_tre`` (explicit TE constraint)
       - UTILITY_CONSTRAINTS → ``cvx_maximise_tre_utility`` (TE and turnover penalties)
    3. **Map**: align weights back to the full asset universe (excluded assets
       receive zero weight).

    The constraint update injects the benchmark weights into the Constraints
    object, enabling the TE constraint (w - w_b)'Σ(w - w_b) <= TE² and
    the turnover constraint ||w - w_0||_1 <= TO.

    Args:
        pd_covar: Covariance matrix (N x N) as DataFrame.
        alphas: Alpha signal per asset. Index=tickers. None for pure tracking.
        benchmark_weights: SAA benchmark weights for tracking error computation.
        constraints: Portfolio constraints including TE budget and turnover limits.
        weights_0: Previous-period weights for warm-start and turnover control.
        rebalancing_indicators: Binary series. Assets with value 0 are frozen.
        detailed_output: If True, return DataFrame with risk contribution diagnostics.
        optimiser_config: Solver configuration.
        verbose: If True, print CVXPY solver diagnostics.

    Returns:
        Portfolio weights as pd.Series (or DataFrame if detailed_output=True),
        aligned to pd_covar.index.
    """
    # filter covariance and alpha vectors for NaN/zero-variance assets
    if alphas is None:
        vectors = None
    else:
        vectors = dict(alphas=alphas)
    clean_covar, good_vectors = filter_covar_and_vectors_for_nans(pd_covar=pd_covar, vectors=vectors)

    # rescale constraints for reduced universe (e.g., group budgets)
    if optimiser_config.apply_total_to_good_ratio:
        total_to_good_ratio = len(pd_covar.columns) / len(clean_covar.columns)
    else:
        total_to_good_ratio = None

    # inject benchmark weights and rebalancing indicators into constraints
    valid_tickers = clean_covar.columns.to_list()
    constraints1 = constraints.update_with_valid_tickers(
        valid_tickers=valid_tickers,
        total_to_good_ratio=total_to_good_ratio,
        weights_0=weights_0,
        benchmark_weights=benchmark_weights,
        rebalancing_indicators=rebalancing_indicators
    )

    alphas_np = good_vectors['alphas'].to_numpy() if alphas is not None else None

    # route to appropriate solver based on constraint enforcement type
    if constraints.constraint_enforcement_type == ConstraintEnforcementType.UTILITY_CONSTRAINTS:
        weights = cvx_maximise_tre_utility(
            covar=clean_covar.to_numpy(),
            alphas=alphas_np,
            constraints=constraints1,
            solver=optimiser_config.solver,
            verbose=verbose
        )
    else:
        weights = cvx_maximise_alpha_over_tre(
            covar=clean_covar.to_numpy(),
            alphas=alphas_np,
            constraints=constraints1,
            solver=optimiser_config.solver,
            verbose=verbose
        )

    weights[np.isinf(weights)] = 0.0
    weights = pd.Series(weights, index=valid_tickers)
    weights = weights.reindex(index=pd_covar.index).fillna(0.0)

    if detailed_output:
        out = compute_portfolio_risk_contribution_outputs(weights=weights, clean_covar=clean_covar)
    else:
        out = weights
    return out


def cvx_maximise_alpha_over_tre(covar: np.ndarray,
                                alphas: np.ndarray,
                                constraints: Constraints,
                                solver: str = 'CLARABEL',
                                verbose: bool = False
                                ) -> np.ndarray:
    """
    Maximise active alpha subject to a hard tracking error constraint.

    Solves:

        max_w  α'(w - w_b)

        s.t.   (w - w_b)' Σ (w - w_b) <= TE²_max
               ||w - w_0||_1 <= TO_max   (if turnover constraint active)
               1'w = 1
               w >= 0   (if long-only)
               w_min <= w <= w_max

    The objective is linear in w (maximise active alpha exposure relative to
    benchmark). The tracking error constraint is a second-order cone constraint
    (SOCP), making the overall problem convex.

    The covariance matrix is wrapped with ``cvx.psd_wrap`` to handle
    near-singular matrices from factor models with low residual variance.

    Args:
        covar: Covariance matrix (N x N) as numpy array.
        alphas: Alpha signal vector (N,). The objective maximises α'(w - w_b).
        constraints: Portfolio constraints with benchmark_weights, TE budget,
            and turnover limits already injected.
        solver: CVXPY solver name.
        verbose: If True, print CVXPY solver diagnostics.

    Returns:
        Optimal weights (N,). Falls back to weights_0 or zeros if the solver
        fails (e.g., infeasible TE budget for the given alpha signal).
    """
    n = covar.shape[0]
    nonneg = constraints.is_long_only
    w = cvx.Variable(n, nonneg=nonneg)
    covar = cvx.psd_wrap(covar)

    # linear objective: maximise active alpha = α'(w - w_b)
    benchmark_weights = constraints.benchmark_weights.to_numpy()
    objective_fun = alphas.T @ (w - benchmark_weights)
    objective = cvx.Maximize(objective_fun)

    # constraints include TE budget, turnover, weight bounds, full investment
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


def cvx_maximise_tre_utility(covar: np.ndarray,
                             constraints: Constraints,
                             alphas: Optional[np.ndarray] = None,
                             solver: str = 'CLARABEL',
                             verbose: bool = False
                             ) -> np.ndarray:
    """
    Maximise utility with tracking error and turnover penalties.

    Solves the soft-constraint formulation:

        max_w  α'(w - w_b) - λ_TE (w - w_b)'Σ(w - w_b) - λ_TO ||w - w_0||_1

        s.t.   1'w = 1
               w >= 0   (if long-only)
               w_min <= w <= w_max

    The penalty weights λ_TE (risk aversion on tracking error) and λ_TO
    (turnover aversion) are encoded in the Constraints object and applied
    via ``set_cvx_utility_objective_constraints``.

    This formulation is always feasible (no hard TE or turnover constraints
    to violate) and produces smoother weight transitions, but requires
    calibrating the penalty weights. It is preferred when:
        - Hard TE constraints frequently bind or cause infeasibility
        - Gradual convergence toward the benchmark is acceptable
        - Turnover costs are material and should be traded off continuously

    Args:
        covar: Covariance matrix (N x N) as numpy array.
        constraints: Portfolio constraints with benchmark_weights, penalty
            weights, and weight bounds.
        alphas: Alpha signal vector (N,). None for pure benchmark tracking
            with turnover minimisation.
        solver: CVXPY solver name.
        verbose: If True, print CVXPY solver diagnostics.

    Returns:
        Optimal weights (N,). Falls back to weights_0 or zeros if the solver
        fails.
    """
    n = covar.shape[0]
    nonneg = constraints.is_long_only
    w = cvx.Variable(n, nonneg=nonneg)
    covar = cvx.psd_wrap(covar)

    # build utility objective with TE and turnover penalties
    constraints1 = constraints.copy()
    objective_fun, constraints_ = constraints1.set_cvx_utility_objective_constraints(
        w=w,
        alphas=alphas,
        covar=covar
    )

    problem = cvx.Problem(cvx.Maximize(objective_fun), constraints_)
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
