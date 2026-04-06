"""
Dispatcher linking covariance estimators to portfolio optimisation solvers.

Provides a unified interface for computing rolling optimal portfolio weights
across all supported optimisation objectives. The dispatcher receives
pre-computed covariance matrices (from any CovarEstimator) and routes to
the appropriate solver based on ``PortfolioObjective``.

Supported objectives and their solvers:

    EQUAL_RISK_CONTRIBUTION
        Constrained risk budgeting via pyrb (ADMM).
        → ``rolling_risk_budgeting``

    MAX_DIVERSIFICATION
        Maximise the diversification ratio DR = w'σ / √(w'Σw) via SLSQP.
        → ``rolling_maximise_diversification``

    MIN_VARIANCE
        Minimise portfolio variance w'Σw via CVXPY (convex QP).
        → ``rolling_quadratic_optimisation``

    QUADRATIC_UTILITY
        Maximise μ'w - (γ/2)w'Σw via CVXPY (convex QP).
        → ``rolling_quadratic_optimisation``

    MAXIMUM_SHARPE_RATIO
        Maximise μ'w / √(w'Σw) via CVXPY using the Charnes-Cooper
        transformation (SOCP).
        → ``rolling_maximize_portfolio_sharpe``

    MAX_CARA_MIXTURE
        Maximise expected CARA utility under a K-component Gaussian mixture
        model via SLSQP.
        → ``rolling_maximize_cara_mixture``

References:
    Sepp A., Ossa I., and Kastenholz M. (2026),
    "Robust Optimization of Strategic and Tactical Asset Allocation
    for Multi-Asset Portfolios",
    The Journal of Portfolio Management, 52(4), 86-120.
    Available at https://www.pm-research.com/content/iijpormgmt/52/4/86

    Sepp A. (2023),
    "Optimal Allocation to Cryptocurrencies in Diversified Portfolios",
    Risk Magazine, pp. 1-6, October 2023.
    Available at https://ssrn.com/abstract=4217841
"""
# packages
import pandas as pd
import qis as qis
from typing import Optional, Dict
import optimalportfolios as opt
from optimalportfolios.optimization.constraints import Constraints
from optimalportfolios.optimization.config import OptimiserConfig
from optimalportfolios.config import PortfolioObjective


def compute_rolling_optimal_weights(prices: pd.DataFrame,
                                    constraints: Constraints,
                                    covar_dict: Dict[pd.Timestamp, pd.DataFrame],
                                    portfolio_objective: PortfolioObjective = PortfolioObjective.MAX_DIVERSIFICATION,
                                    time_period: qis.TimePeriod = None,
                                    risk_budget: pd.Series = None,
                                    returns_freq: Optional[str] = 'W-WED',
                                    rebalancing_freq: str = 'QE',
                                    span: int = 52,
                                    roll_window: int = 20,
                                    carra: float = 0.5,
                                    n_mixures: int = 3,
                                    optimiser_config: OptimiserConfig = OptimiserConfig(apply_total_to_good_ratio=True)
                                    ) -> pd.DataFrame:
    """
    Compute rolling optimal portfolio weights for any supported objective.

    Routes to the appropriate solver based on ``portfolio_objective``, passing
    the pre-computed covariance matrices and objective-specific parameters.

    Args:
        prices: Asset price panel. Index=dates, columns=tickers.
        constraints: Portfolio constraints.
        covar_dict: Pre-computed covariance matrices keyed by rebalancing date.
        portfolio_objective: Optimisation objective.
        time_period: Reporting period (MAX_CARA_MIXTURE only).
        risk_budget: Target risk budgets (EQUAL_RISK_CONTRIBUTION only).
        returns_freq: Return frequency for mean-dependent objectives.
        rebalancing_freq: Rebalancing frequency (MAX_CARA_MIXTURE only).
        span: EWMA span for mean estimation (MAXIMUM_SHARPE_RATIO).
        roll_window: Rolling window for mixture estimation (MAX_CARA_MIXTURE).
        carra: CARA risk aversion parameter γ.
        n_mixures: Number of mixture components K.
        optimiser_config: Solver configuration passed through to all solvers.

    Returns:
        DataFrame of portfolio weights.
    """
    if portfolio_objective == PortfolioObjective.EQUAL_RISK_CONTRIBUTION:
        weights = opt.rolling_risk_budgeting(prices=prices,
                                             constraints=constraints,
                                             covar_dict=covar_dict,
                                             risk_budget=risk_budget,
                                             optimiser_config=optimiser_config)

    elif portfolio_objective == PortfolioObjective.MAX_DIVERSIFICATION:
        weights = opt.rolling_maximise_diversification(prices=prices,
                                                       constraints=constraints,
                                                       covar_dict=covar_dict,
                                                       optimiser_config=optimiser_config)

    elif portfolio_objective in [PortfolioObjective.MIN_VARIANCE, PortfolioObjective.QUADRATIC_UTILITY]:
        weights = opt.rolling_quadratic_optimisation(prices=prices,
                                                     constraints=constraints,
                                                     portfolio_objective=portfolio_objective,
                                                     covar_dict=covar_dict,
                                                     carra=carra,
                                                     optimiser_config=optimiser_config)

    elif portfolio_objective == PortfolioObjective.MAXIMUM_SHARPE_RATIO:
        expected_returns = opt.estimate_rolling_ewma_means(prices=prices,
                                                rebalancing_dates=list(covar_dict.keys()),
                                                returns_freq=returns_freq,
                                                span=span, annualize=True)
        weights = opt.rolling_maximize_portfolio_sharpe(prices=prices,
                                                        expected_returns=expected_returns,
                                                        constraints=constraints,
                                                        covar_dict=covar_dict,
                                                        optimiser_config=optimiser_config)

    elif portfolio_objective == PortfolioObjective.MAX_CARA_MIXTURE:
        weights = opt.rolling_maximize_cara_mixture(prices=prices,
                                                    constraints=constraints,
                                                    time_period=time_period,
                                                    returns_freq=returns_freq,
                                                    rebalancing_freq=rebalancing_freq,
                                                    carra=carra,
                                                    n_components=n_mixures,
                                                    roll_window=roll_window,
                                                    optimiser_config=optimiser_config)

    else:
        raise NotImplementedError(f"{portfolio_objective}")

    return weights


def backtest_rolling_optimal_portfolio(prices: pd.DataFrame,
                                       constraints: Constraints,
                                       covar_dict: Dict[pd.Timestamp, pd.DataFrame],
                                       perf_time_period: qis.TimePeriod = None,
                                       portfolio_objective: PortfolioObjective = PortfolioObjective.MAX_DIVERSIFICATION,
                                       risk_budget: pd.Series = None,
                                       returns_freq: Optional[str] = 'W-WED',
                                       rebalancing_freq: str = 'QE',
                                       span: int = 52,
                                       roll_window: int = 6*52,
                                       carra: float = 0.5,
                                       n_mixures: int = 3,
                                       ticker: str = None,
                                       rebalancing_costs: float = 0.0010,
                                       weight_implementation_lag: Optional[int] = None,
                                       optimiser_config: OptimiserConfig = OptimiserConfig(apply_total_to_good_ratio=True)
                                       ) -> qis.PortfolioData:
    """
    Compute optimal weights and run a backtest in one call.

    Args:
        prices: Asset price panel. Index=dates, columns=tickers.
        constraints: Portfolio constraints.
        covar_dict: Pre-computed covariance matrices keyed by rebalancing date.
        perf_time_period: Reporting period for output weights.
        portfolio_objective: Optimisation objective.
        risk_budget: Target risk budgets (EQUAL_RISK_CONTRIBUTION only).
        returns_freq: Return frequency for mean-dependent objectives.
        rebalancing_freq: Rebalancing frequency (MAX_CARA_MIXTURE only).
        span: EWMA span for mean estimation.
        roll_window: Rolling window for mixture estimation.
        carra: CARA risk aversion parameter γ.
        n_mixures: Number of mixture components K.
        ticker: Portfolio identifier string for report.
        rebalancing_costs: Proportional transaction cost per unit traded.
        weight_implementation_lag: Number of periods to delay weight
            implementation. None for no lag.
        optimiser_config: Solver configuration passed through to all solvers.

    Returns:
        qis.PortfolioData with NAV, returns, weights history, and turnover.
    """
    weights = compute_rolling_optimal_weights(prices=prices,
                                              constraints=constraints,
                                              covar_dict=covar_dict,
                                              portfolio_objective=portfolio_objective,
                                              risk_budget=risk_budget,
                                              returns_freq=returns_freq,
                                              rebalancing_freq=rebalancing_freq,
                                              span=span,
                                              carra=carra,
                                              roll_window=roll_window,
                                              n_mixures=n_mixures,
                                              optimiser_config=optimiser_config)

    if perf_time_period is not None:
        weights = perf_time_period.locate(weights)

    prices_ = qis.truncate_prior_to_start(df=prices, start=weights.index[0])

    portfolio_out = qis.backtest_model_portfolio(prices=prices_,
                                                 weights=weights,
                                                 rebalancing_costs=rebalancing_costs,
                                                 weight_implementation_lag=weight_implementation_lag,
                                                 ticker=ticker)
    return portfolio_out
