"""
Dispatcher linking covariance estimators to portfolio optimisation solvers.

Provides a unified interface for computing rolling optimal portfolio weights
across all supported optimisation objectives. The dispatcher receives
pre-computed covariance matrices (from any CovarEstimator) and routes to
the appropriate solver based on ``PortfolioObjective``.

Supported objectives and their solvers:

    EQUAL_RISK_CONTRIBUTION
        Constrained risk budgeting via pyrb (ADMM). Each asset's risk
        contribution matches its prescribed budget.
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
        Maximise μ'w / √(w'Σw) via CVXPY using the Cornuejols-Tütüncü
        rescaling trick (SOCP).
        → ``rolling_maximize_portfolio_sharpe``

    MAX_CARA_MIXTURE
        Maximise expected CARA utility under a K-component Gaussian mixture
        model via SLSQP. Captures regime-dependent correlations and fat tails.
        → ``rolling_maximize_cara_mixture``

The covariance matrices are always pre-computed and passed as
``covar_dict: Dict[Timestamp, DataFrame]``, decoupling estimation from
optimisation. This allows mixing any CovarEstimator (EWMA, Factor LASSO,
shrinkage, etc.) with any portfolio objective.

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
                                    n_mixures: int = 3
                                    ) -> pd.DataFrame:
    """
    Compute rolling optimal portfolio weights for any supported objective.

    Routes to the appropriate solver based on ``portfolio_objective``, passing
    the pre-computed covariance matrices and objective-specific parameters.
    The rebalancing schedule is defined by the keys of ``covar_dict``.

    This is the main entry point for portfolio construction: pair any
    CovarEstimator output with any optimisation objective.

    Args:
        prices: Asset price panel. Index=dates, columns=tickers. Used for
            column alignment and, for some objectives (MAX_CARA_MIXTURE,
            MAXIMUM_SHARPE_RATIO), for computing returns within the solver.
        constraints: Portfolio constraints (long-only, weight bounds, group
            exposures, turnover limits). Passed through to the solver.
        covar_dict: Pre-computed covariance matrices keyed by rebalancing date.
            Produced by ``EwmaCovarEstimator.fit_rolling_covars()`` or
            ``FactorCovarEstimator.fit_rolling_covars()``.
        portfolio_objective: Optimisation objective. Determines which solver
            is called. See module docstring for the full mapping.
        time_period: Reporting period for output weights (MAX_CARA_MIXTURE only).
            Weights outside this period are trimmed.
        risk_budget: Target risk budgets per asset (EQUAL_RISK_CONTRIBUTION only).
            Index=tickers, values=budgets. Assets with budget 0 are excluded.
        returns_freq: Return frequency for objectives that estimate means
            internally (MAXIMUM_SHARPE_RATIO, MAX_CARA_MIXTURE).
        rebalancing_freq: Rebalancing frequency (MAX_CARA_MIXTURE only;
            other objectives derive the schedule from covar_dict keys).
        span: EWMA span for mean estimation (MAXIMUM_SHARPE_RATIO) or
            covariance in mixture fitting (MAX_CARA_MIXTURE).
        roll_window: Rolling window length in return periods for mixture
            estimation (MAX_CARA_MIXTURE). Default 20 for quarterly,
            6*52=312 for weekly returns (6 years).
        carra: CARA risk aversion parameter γ (QUADRATIC_UTILITY,
            MAX_CARA_MIXTURE). Higher values → more conservative portfolios.
        n_mixures: Number of Gaussian mixture components K (MAX_CARA_MIXTURE).

    Returns:
        DataFrame of portfolio weights. Index=rebalancing dates,
        columns=tickers aligned to ``prices.columns``.

    Raises:
        NotImplementedError: If ``portfolio_objective`` is not supported.
    """
    if portfolio_objective == PortfolioObjective.EQUAL_RISK_CONTRIBUTION:
        weights = opt.rolling_risk_budgeting(prices=prices,
                                             constraints=constraints,
                                             covar_dict=covar_dict,
                                             risk_budget=risk_budget)

    elif portfolio_objective == PortfolioObjective.MAX_DIVERSIFICATION:
        weights = opt.rolling_maximise_diversification(prices=prices,
                                                       constraints=constraints,
                                                       covar_dict=covar_dict)

    elif portfolio_objective in [PortfolioObjective.MIN_VARIANCE, PortfolioObjective.QUADRATIC_UTILITY]:
        weights = opt.rolling_quadratic_optimisation(prices=prices,
                                                     constraints=constraints,
                                                     portfolio_objective=portfolio_objective,
                                                     covar_dict=covar_dict,
                                                     carra=carra)

    elif portfolio_objective == PortfolioObjective.MAXIMUM_SHARPE_RATIO:
        weights = opt.rolling_maximize_portfolio_sharpe(prices=prices,
                                                        constraints=constraints,
                                                        returns_freq=returns_freq,
                                                        covar_dict=covar_dict,
                                                        span=span)

    elif portfolio_objective == PortfolioObjective.MAX_CARA_MIXTURE:
        weights = opt.rolling_maximize_cara_mixture(prices=prices,
                                                    constraints=constraints,
                                                    time_period=time_period,
                                                    returns_freq=returns_freq,
                                                    rebalancing_freq=rebalancing_freq,
                                                    carra=carra,
                                                    n_components=n_mixures,
                                                    roll_window=roll_window)

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
                                       weight_implementation_lag: Optional[int] = None
                                       ) -> qis.PortfolioData:
    """
    Compute optimal weights and run a backtest in one call.

    Combines ``compute_rolling_optimal_weights`` with ``qis.backtest_model_portfolio``
    to produce a complete portfolio backtest. This is the end-to-end entry point
    for strategy evaluation: from pre-computed covariance matrices to
    PortfolioData with NAV, returns, turnover, and performance attribution.

    The backtest applies rebalancing costs as a proportional charge on
    traded volume at each rebalancing date, and optionally implements
    weights with a one-day lag (standard for daily price data to avoid
    look-ahead bias).

    Args:
        prices: Asset price panel. Index=dates, columns=tickers.
        constraints: Portfolio constraints.
        covar_dict: Pre-computed covariance matrices keyed by rebalancing date.
        perf_time_period: Reporting period. If provided, weights are trimmed
            to this period before backtesting. Useful when the covariance
            estimation period starts earlier than the desired reporting window.
        portfolio_objective: Optimisation objective (see ``compute_rolling_optimal_weights``).
        returns_freq: Return frequency for mean-dependent objectives.
        rebalancing_freq: Rebalancing frequency (MAX_CARA_MIXTURE only).
        span: EWMA span for mean estimation.
        roll_window: Rolling window for mixture estimation (MAX_CARA_MIXTURE).
        carra: CARA risk aversion parameter γ.
        n_mixures: Number of mixture components K.
        ticker: Portfolio identifier string for reporting (e.g., 'Min Var SAA').
        rebalancing_costs: Proportional transaction cost per unit of traded
            volume. Default 0.0010 (10 basis points).
        weight_implementation_lag: Number of periods to delay weight
            implementation. Use 1 for daily prices to avoid look-ahead bias.
            None for no lag (weights applied on computation date).

    Returns:
        qis.PortfolioData with NAV, returns, weights history, turnover,
        and attribution. Ready for ``qis.generate_strategy_benchmark_factsheet_plt``.
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
                                              n_mixures=n_mixures)

    # trim weights to performance reporting period
    if perf_time_period is not None:
        weights = perf_time_period.locate(weights)

    # ensure prices start at or before the first weight date
    prices_ = qis.truncate_prior_to_start(df=prices, start=weights.index[0])

    portfolio_out = qis.backtest_model_portfolio(prices=prices_,
                                                 weights=weights,
                                                 rebalancing_costs=rebalancing_costs,
                                                 weight_implementation_lag=weight_implementation_lag,
                                                 ticker=ticker)
    return portfolio_out
