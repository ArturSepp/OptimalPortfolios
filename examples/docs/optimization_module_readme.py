"""Canonical script of docs/optimization_module_readme.md.

The page's nine Python blocks are excerpts of ``main`` and run here in the same order; every
number and property the page states is asserted after them against a reference computed a
different way: the closed-form optima of the diagonal fixture (inverse variance, inverse
volatility, the half-gamma utility, the tangency portfolio, the two-equality return floor and the
tracking-error ellipsoid), the stated mixture utility evaluated by hand, group and deviation
exposures from the published loadings, an explicit EWMA recursion, the signatures of every public
entry point and recording stand-ins for the dispatcher's routes. The script runs offline after
``pip install optimalportfolios``, with outgoing connections refused, and needs no data file or
random seed:

    python -m examples.docs.optimization_module_readme
"""
from contextlib import contextmanager
from dataclasses import FrozenInstanceError, fields
import functools
import importlib
import inspect
import socket

import numpy as np
import pandas as pd

TICKERS = ['Equity A', 'Equity B', 'Bond A', 'Bond B', 'Gold']
ANNUAL_VOLS = [0.20, 0.25, 0.10, 0.15, 0.30]
# The page's allocation table: minimum-variance weights rounded to six decimals.
DISPLAYED_WEIGHTS = [0.127191, 0.081402, 0.508762, 0.226116, 0.056529]
# The page's OptimiserConfig table, in its row order.
DOCUMENTED_CONFIG = {
    'solver': 'CLARABEL', 'verbose': False, 'apply_total_to_good_ratio': False,
    'use_drifted_weights_0': True, 'diagnose_infeasibility': True, 'validate_inputs': True,
    'max_constraint_relaxation': None, 'factorize_covar': True,
}
# The page's dispatch table: enum member to rolling target.
ROUTES = {
    'EQUAL_RISK_CONTRIBUTION': 'rolling_risk_budgeting',
    'MAX_DIVERSIFICATION': 'rolling_maximise_diversification',
    'MIN_VARIANCE': 'rolling_quadratic_optimisation',
    'QUADRATIC_UTILITY': 'rolling_quadratic_optimisation',
    'MAXIMUM_SHARPE_RATIO': 'rolling_maximize_portfolio_sharpe',
    'MAX_CARA_MIXTURE': 'rolling_maximize_cara_mixture',
}
# The page's two lists of entry points by their default apply_total_to_good_ratio.
RATIO_DEFAULT_TRUE = [
    'compute_rolling_optimal_weights', 'backtest_rolling_optimal_portfolio',
    'rolling_quadratic_optimisation', 'wrapper_quadratic_optimisation',
    'rolling_maximize_portfolio_sharpe', 'wrapper_maximize_portfolio_sharpe',
    'rolling_maximise_diversification', 'wrapper_maximise_diversification',
    'rolling_maximize_cara_mixture', 'wrapper_maximize_cara_mixture',
    'rolling_risk_budgeting', 'wrapper_risk_budgeting',
    'rolling_maximise_alpha_with_target_return', 'wrapper_maximise_alpha_with_target_return',
]
RATIO_DEFAULT_FALSE = [
    'rolling_minimise_tracking_error', 'wrapper_minimise_tracking_error',
    'rolling_min_variance_target_return', 'wrapper_min_variance_target_return',
    'rolling_max_return_target_vol', 'wrapper_max_return_target_vol',
    'rolling_maximise_alpha_over_tre', 'wrapper_maximise_alpha_over_tre',
]


@contextmanager
def replaced(target, **attributes):
    """Replace attributes of a module or class inside the block and restore them afterwards."""
    missing = object()
    saved = {name: vars(target).get(name, missing) for name in attributes}
    for name, value in attributes.items():
        setattr(target, name, value)
    try:
        yield
    finally:
        for name, value in saved.items():
            if value is missing:
                delattr(target, name)  # the attribute was inherited, not the target's own
            else:
                setattr(target, name, value)


def refuse_network(function):
    """Run ``function`` with outgoing socket connections refused, as the page promises."""
    @functools.wraps(function)
    def guarded(*args, **kwargs):
        """Call ``function`` while every socket connection attempt raises."""
        def refuse(*_, **__):
            """Refuse one connection attempt."""
            raise AssertionError('The optimization page must run offline.')

        with replaced(socket, create_connection=refuse), replaced(socket.socket, connect=refuse):
            return function(*args, **kwargs)
    return guarded


def equality_minimum_variance(variances, rows, targets) -> np.ndarray:
    """Minimise w' diag(v) w subject to rows' w = targets in closed form.

    The first-order conditions give w = D^-1 A (A' D^-1 A)^-1 b; the caller checks that the
    solution is interior, so the long-only bounds and unit caps do not bind.
    """
    inverse = 1.0 / np.asarray(variances, dtype=float)
    matrix = np.column_stack(rows)
    return inverse * (matrix @ np.linalg.solve(matrix.T @ (inverse[:, None] * matrix), targets))


def diagonal_tracking_error(weights, benchmark, vols) -> float:
    """Return sqrt(sum(((w - b) * vol)^2)), tracking error under the diagonal covariance."""
    active = np.asarray(weights, dtype=float) - np.asarray(benchmark, dtype=float)
    return float(np.linalg.norm(active * np.asarray(vols, dtype=float)))


def mixture_loss(weights, means, variances) -> float:
    """Negative expected CARA utility (gamma 5) of the page's two-component diagonal mixture.

    Component one has the stated means and covariance with probability 0.75; component two has
    half the means and 1.5 times the covariance with probability 0.25. For a Gaussian component,
    E[exp(-gamma R)] = exp(-gamma m + gamma^2 s^2 / 2).
    """
    mean = float(np.dot(means, weights))
    variance = float(np.sum(np.asarray(variances) * np.asarray(weights) ** 2))
    return (0.75 * np.exp(-5.0 * mean + 12.5 * variance)
            + 0.25 * np.exp(-2.5 * mean + 12.5 * 1.5 * variance))


def ewma_log_means(prices: pd.DataFrame, span: int, periods_per_year: int) -> pd.DataFrame:
    """Annualised EWMA of log returns by the explicit recursion, seeded at the first return."""
    returns = np.log(prices).diff().iloc[1:]
    decay = 1.0 - 2.0 / (span + 1.0)
    means = returns.copy()
    for row in range(1, len(means)):
        means.iloc[row] = decay * means.iloc[row - 1] + (1.0 - decay) * returns.iloc[row]
    return periods_per_year * means


def config_default_of(function):
    """Return the default ``optimiser_config`` of ``function``."""
    return inspect.signature(function).parameters['optimiser_config'].default


def defaults(function) -> dict:
    """Return the default value of every parameter of ``function``."""
    return {name: parameter.default
            for name, parameter in inspect.signature(function).parameters.items()}


def check_dispatch_route(objective: str, prices, constraints, covar_dict, config,
                         forecasts) -> None:
    """Route ``objective`` through the dispatcher with every solver recorded instead of run.

    Checks the page's dispatch table and parameter scope: exactly one documented target is
    called; the five covariance routes receive ``covar_dict`` and neither ``time_period`` nor
    ``rebalancing_freq``; CARA receives no covariance, the schedule arguments and ``n_mixures``
    as ``n_components``; and only quadratic utility and maximum Sharpe estimate annualised means
    on the covariance keys.
    """
    import optimalportfolios as opt
    import qis

    module = importlib.import_module('optimalportfolios.optimization.wrapper_rolling_portfolios')
    calls, mean_calls = [], []
    sentinel = pd.DataFrame([[1.0]], index=[pd.Timestamp('2024-12-31')], columns=['sentinel'])

    def record(**kwargs):
        """Record the routed solver call."""
        calls.append(kwargs)
        return sentinel

    def record_means(**kwargs):
        """Record the mean-estimation call and return fixed forecasts."""
        mean_calls.append(kwargs)
        return forecasts

    def wrong_route(**kwargs):
        """Fail on any solver other than the documented one."""
        raise AssertionError(f'{objective} took an undocumented route')

    stubs = {name: wrong_route for name in set(ROUTES.values())}
    stubs[ROUTES[objective]] = record
    period = qis.TimePeriod('2024-01-01', '2024-12-31')
    with replaced(module, estimate_rolling_ewma_means=record_means, **stubs):
        result = module.compute_rolling_optimal_weights(
            prices, constraints, covar_dict,
            portfolio_objective=getattr(opt.PortfolioObjective, objective), time_period=period,
            returns_freq='ME', rebalancing_freq='YE', span=9, roll_window=17, n_mixures=2,
            optimiser_config=config)
    assert result is sentinel and len(calls) == 1
    call = calls[0]
    assert call['optimiser_config'] is config
    if objective == 'MAX_CARA_MIXTURE':
        assert 'covar_dict' not in call
        assert call['time_period'] is period and call['rebalancing_freq'] == 'YE'
        assert call['n_components'] == 2 and call['roll_window'] == 17
        assert call['returns_freq'] == 'ME' and call['carra'] == 0.5
    else:
        assert call['covar_dict'] is covar_dict
        assert 'time_period' not in call and 'rebalancing_freq' not in call
    if objective in ('MIN_VARIANCE', 'QUADRATIC_UTILITY'):
        assert call['carra'] == 0.5
    if objective in ('QUADRATIC_UTILITY', 'MAXIMUM_SHARPE_RATIO'):
        assert call['expected_returns'] is forecasts and len(mean_calls) == 1
        means_call = mean_calls[0]
        assert means_call['annualize'] is True and means_call['returns_freq'] == 'ME'
        assert means_call['span'] == 9 and means_call['rebalancing_dates'] == list(covar_dict)
    else:
        assert mean_calls == []


def sharpe_backend(covar: np.ndarray, means: np.ndarray, constraints) -> list:
    """Return the backend names ``cvx_maximize_portfolio_sharpe`` calls for ``constraints``."""
    module = importlib.import_module('optimalportfolios.optimization.general.max_sharpe')
    used = []
    sentinel = object()

    def transformed(**kwargs):
        """Record the Charnes-Cooper CVXPY route."""
        used.append('CVXPY')
        return sentinel

    def ratio(**kwargs):
        """Record the direct SLSQP ratio route."""
        used.append('SLSQP')
        return sentinel

    with replaced(module, _cvx_maximize_sharpe_charnes_cooper=transformed,
                  _scipy_maximize_sharpe=ratio):
        assert module.cvx_maximize_portfolio_sharpe(covar, means, constraints) is sentinel
    return used


def assert_raises(error: type, function, *args, **kwargs) -> None:
    """Fail unless ``function(*args, **kwargs)`` raises ``error``."""
    try:
        function(*args, **kwargs)
    except error:
        return
    raise AssertionError(f'expected {error.__name__}')


@refuse_network
def main() -> None:
    """Run the page's blocks in order and assert every number and property it states."""
    import numpy as np
    import pandas as pd
    import cvxpy as cvx
    from dataclasses import asdict, replace
    import qis
    import optimalportfolios as opt
    from optimalportfolios.optimization.constraints import (
        GroupLowerUpperConstraints, BenchmarkDeviationConstraints,
    )

    tickers = ["Equity A", "Equity B", "Bond A", "Bond B", "Gold"]
    annual_vols = pd.Series([0.20, 0.25, 0.10, 0.15, 0.30], index=tickers)
    pd_covar = pd.DataFrame(np.diag(annual_vols ** 2), index=tickers, columns=tickers)
    expected_returns = pd.Series([0.07, 0.08, 0.03, 0.04, 0.05], index=tickers)
    benchmark = pd.Series(0.20, index=tickers)
    alphas = pd.Series([-0.3, 0.6, 0.1, -0.2, 0.4], index=tickers)
    constraints = opt.Constraints(
        is_long_only=True, min_exposure=1.0, max_exposure=1.0,
        min_weights=pd.Series(0.0, index=tickers),
        max_weights=pd.Series(1.0, index=tickers),
    )
    config = opt.OptimiserConfig(apply_total_to_good_ratio=False)
    weights, outcome = opt.wrapper_quadratic_optimisation(
        pd_covar, constraints, optimiser_config=config, context="guide: minimum variance",
    )
    assert outcome.accepted and outcome.compliant
    allocation = pd.DataFrame({"Annual volatility": annual_vols, "Weight": weights})
    print(allocation.round(6))

    from optimalportfolios.optimization.solver_diagnostics import OptimizationOutcome

    def check_outcome(result) -> None:
        """An accepted, compliant five-asset outcome with a covariance factorization."""
        assert isinstance(result, OptimizationOutcome)
        assert result.accepted and result.compliant and result.fallback_source is None
        assert result.covar_factorization is not None and result.weights.shape == (5,)

    def check_weights(series) -> None:
        """Complete, finite, long-only, fully invested labelled weights."""
        assert isinstance(series, pd.Series) and list(series.index) == TICKERS
        assert np.isfinite(series).all() and (series >= -1e-7).all()
        assert abs(series.sum() - 1.0) < 1e-6

    # The fixture: five assets, the page's volatilities, a diagonal annual covariance.
    assert tickers == TICKERS and annual_vols.tolist() == ANNUAL_VOLS
    np.testing.assert_array_equal(np.diag(pd_covar), np.square(ANNUAL_VOLS))
    assert np.count_nonzero(pd_covar.to_numpy() - np.diag(np.diag(pd_covar))) == 0
    check_outcome(outcome)
    check_weights(weights)
    # Fully invested minimum variance on a diagonal covariance weights by inverse variance, not
    # inverse volatility; the page's table is that closed form rounded to six decimals.
    variances = np.square(ANNUAL_VOLS)
    inverse_variance = equality_minimum_variance(variances, [np.ones(5)], [1.0])
    np.testing.assert_allclose(inverse_variance, (1 / variances) / np.sum(1 / variances),
                               rtol=1e-14)
    np.testing.assert_allclose(weights, inverse_variance, rtol=0.0, atol=2e-6)
    assert np.round(inverse_variance, 6).tolist() == DISPLAYED_WEIGHTS
    assert allocation.index.tolist() == TICKERS
    np.testing.assert_allclose(allocation["Weight"].round(6), DISPLAYED_WEIGHTS, rtol=0.0,
                               atol=1e-6)
    assert not np.allclose(weights, (1 / annual_vols) / (1 / annual_vols).sum(), atol=1e-2)

    default_config = opt.OptimiserConfig()
    configuration = asdict(default_config)
    legacy_drift_config = replace(default_config, use_drifted_weights_0=False)
    assert default_config.use_drifted_weights_0
    assert not legacy_drift_config.use_drifted_weights_0

    # Eight fields with the page's defaults, in the table's order; the dataclass is frozen.
    assert configuration == DOCUMENTED_CONFIG
    assert [field.name for field in fields(default_config)] == list(DOCUMENTED_CONFIG)
    assert len(configuration) == 8 and default_config.__dataclass_params__.frozen
    assert_raises(FrozenInstanceError, setattr, default_config, "solver", "SCS")
    assert asdict(legacy_drift_config) == {**DOCUMENTED_CONFIG, "use_drifted_weights_0": False}

    # Dispatcher and entry-point defaults stated in the dispatch flow.
    dispatch = defaults(opt.compute_rolling_optimal_weights)
    adapter = defaults(opt.backtest_rolling_optimal_portfolio)
    assert dispatch["portfolio_objective"] == opt.PortfolioObjective.MAX_DIVERSIFICATION
    assert adapter["portfolio_objective"] == opt.PortfolioObjective.MAX_DIVERSIFICATION
    assert dispatch["returns_freq"] == "W-WED" and dispatch["span"] == 52
    assert dispatch["carra"] == 0.5 and dispatch["roll_window"] == 20
    assert dispatch["n_mixures"] == 3 and "n_components" not in dispatch
    assert defaults(opt.wrapper_quadratic_optimisation)["carra"] == 1.0
    assert defaults(opt.rolling_quadratic_optimisation)["carra"] == 1.0
    assert adapter["roll_window"] == 312 == 6 * 52
    assert defaults(opt.rolling_maximize_cara_mixture)["roll_window"] == 312
    assert adapter["rebalancing_costs"] == 0.0010
    assert adapter["weight_implementation_lag"] is None
    # apply_total_to_good_ratio: the page's two lists, complete over every public entry point
    # that takes an optimiser_config.
    public = {}
    for name in dir(opt):
        member = getattr(opt, name)
        if inspect.isfunction(member) and "optimiser_config" in defaults(member):
            public[name] = config_default_of(member).apply_total_to_good_ratio
    assert sorted(public) == sorted(RATIO_DEFAULT_TRUE + RATIO_DEFAULT_FALSE)
    assert all(public[name] is True for name in RATIO_DEFAULT_TRUE)
    assert all(public[name] is False for name in RATIO_DEFAULT_FALSE)
    assert all(config_default_of(getattr(opt, name)) == opt.OptimiserConfig()
               for name in RATIO_DEFAULT_FALSE)
    # Only that field differs: every True default is the dataclass with the ratio switched on.
    assert all(config_default_of(getattr(opt, name))
               == opt.OptimiserConfig(apply_total_to_good_ratio=True)
               for name in RATIO_DEFAULT_TRUE)

    # Pitfall. With a zero-variance sixth asset excluded, no configuration means the wrapper's
    # default, which rescales the five remaining 0.25 caps to 0.30 = 0.25 * 6 / 5, while
    # OptimiserConfig() keeps them. Outcomes describe the filtered universe; the Series does not.
    names = TICKERS + ["Unavailable"]
    padded = pd_covar.reindex(index=names, columns=names, fill_value=0.0)
    capped = replace(constraints, min_weights=pd.Series(0.0, index=names),
                     max_weights=pd.Series(0.25, index=names))
    full, explicit = opt.wrapper_quadratic_optimisation(padded, capped, optimiser_config=config)
    _, dataclass_default = opt.wrapper_quadratic_optimisation(
        padded, capped, optimiser_config=opt.OptimiserConfig())
    _, wrapper_default = opt.wrapper_quadratic_optimisation(padded, capped)
    assert full.shape == (6,) and full["Unavailable"] == 0.0
    assert explicit.weights.shape == (5,)
    np.testing.assert_allclose(full.loc[TICKERS], explicit.weights, rtol=0.0, atol=1e-15)
    for kept in (explicit, dataclass_default):
        assert kept.constraints.max_weights.index.tolist() == TICKERS
        assert kept.constraints.max_weights.tolist() == [0.25] * 5
    np.testing.assert_allclose(wrapper_default.constraints.max_weights, [0.30] * 5, rtol=0.0,
                               atol=1e-15)
    assert abs(0.25 * len(names) / len(TICKERS) - 0.30) < 1e-15
    assert wrapper_default.weights[TICKERS.index("Bond A")] > 0.25 + 0.04

    utility_weights, utility_outcome = opt.wrapper_quadratic_optimisation(
        pd_covar, constraints, portfolio_objective=opt.PortfolioObjective.QUADRATIC_UTILITY,
        means=expected_returns, carra=5.0, optimiser_config=config,
    )
    sharpe_weights, sharpe_outcome = opt.wrapper_maximize_portfolio_sharpe(
        pd_covar, expected_returns, constraints, optimiser_config=config,
    )
    tracking_weights, tracking_outcome = opt.wrapper_minimise_tracking_error(
        pd_covar, benchmark, constraints, optimiser_config=config,
    )
    risk_weights = opt.wrapper_risk_budgeting(
        pd_covar, constraints, risk_budget=pd.Series(0.2, index=tickers),
        optimiser_config=config,
    )
    diversification_weights = opt.wrapper_maximise_diversification(
        pd_covar, constraints, optimiser_config=config,
    )
    mixture_weights = opt.wrapper_maximize_cara_mixture(
        means=[expected_returns.to_numpy(), 0.5 * expected_returns.to_numpy()],
        covars=[pd_covar.to_numpy(), 1.5 * pd_covar.to_numpy()],
        probs=np.array([0.75, 0.25]), constraints=constraints, tickers=tickers,
        carra=5.0, optimiser_config=config,
    )
    raw_outcome = opt.cvx_quadratic_optimisation(
        opt.PortfolioObjective.MIN_VARIANCE, pd_covar.to_numpy(), constraints,
        solver=config.solver, factorize_covar=config.factorize_covar,
    )

    # Three CVXPY outcome tuples; risk budgeting, diversification and CARA return Series; the
    # numerical layer returns an outcome holding an array.
    for result in (utility_outcome, sharpe_outcome, tracking_outcome, raw_outcome):
        check_outcome(result)
    for series in (utility_weights, sharpe_weights, tracking_weights, risk_weights,
                   diversification_weights, mixture_weights):
        check_weights(series)
    assert isinstance(raw_outcome.weights, np.ndarray)
    np.testing.assert_allclose(raw_outcome.weights, inverse_variance, rtol=0.0, atol=2e-6)
    detailed = opt.wrapper_risk_budgeting(pd_covar, constraints,
                                          risk_budget=pd.Series(0.2, index=tickers),
                                          optimiser_config=config, detailed_output=True)
    assert isinstance(detailed, pd.DataFrame)
    # Quadratic utility carries the one-half factor: w = D^-1 (mu - lambda 1) / gamma, with
    # lambda set by full investment, is interior here.
    inverse = 1 / variances
    means = expected_returns.to_numpy()
    multiplier = (np.sum(inverse * means) - 5.0) / np.sum(inverse)
    half_gamma = inverse * (means - multiplier) / 5.0
    assert half_gamma.min() > 0 and half_gamma.max() < 1 and abs(half_gamma.sum() - 1) < 1e-15
    np.testing.assert_allclose(utility_weights, half_gamma, rtol=0.0, atol=3e-6)
    # Without the one-half factor the optimum would be the gamma-10 solution, which differs.
    no_half = (np.sum(inverse * means) - 10.0) / np.sum(inverse)
    assert not np.allclose(utility_weights, inverse * (means - no_half) / 10.0, atol=1e-2)
    # Maximum Sharpe uses the supplied means: the tangency portfolio is proportional to mu / v.
    tangency = means / variances
    np.testing.assert_allclose(sharpe_weights, tangency / tangency.sum(), rtol=0.0, atol=3e-6)
    # A feasible benchmark has zero active risk and is the positive-definite optimum.
    np.testing.assert_allclose(tracking_weights, benchmark, rtol=0.0, atol=2e-6)
    # Equal risk budgets and maximum diversification both give inverse volatility here.
    inverse_volatility = (1 / np.array(ANNUAL_VOLS)) / np.sum(1 / np.array(ANNUAL_VOLS))
    for series in (risk_weights, diversification_weights):
        np.testing.assert_allclose(series, inverse_volatility, rtol=0.0, atol=3e-5)
    contributions = risk_weights.to_numpy() ** 2 * variances
    np.testing.assert_allclose(contributions / contributions.sum(), 0.2, rtol=0.0, atol=1e-4)
    # The CARA weights beat the benchmark and every single-asset portfolio under the stated
    # two-component exponential utility, evaluated by hand.
    fitted = mixture_loss(mixture_weights, means, variances)
    for candidate in [benchmark.to_numpy(), *np.eye(5)]:
        assert fitted <= mixture_loss(candidate, means, variances) + 1e-8
    # Insight: the objective alone moves the allocation. Minimum variance holds 50.9% in Bond A
    # and 5.7% in Gold; inverse volatility holds 34.5% and 11.5%.
    bond_a, gold = TICKERS.index("Bond A"), TICKERS.index("Gold")
    assert round(100 * weights.iloc[bond_a], 1) == 50.9
    assert round(100 * weights.iloc[gold], 1) == 5.7
    for series in (risk_weights, diversification_weights):
        assert round(100 * series.iloc[bond_a], 1) == 34.5
        assert round(100 * series.iloc[gold], 1) == 11.5
    assert abs(inverse_volatility[bond_a] - 10 / 29) < 1e-15
    # Solver reference: fixed net exposure takes the transformed CVXPY problem, variable net
    # exposure SLSQP; the Sharpe wrapper keeps its outcome tuple on either path.
    assert sharpe_backend(pd_covar.to_numpy(), means, constraints) == ["CVXPY"]
    variable = replace(constraints, min_exposure=0.5)
    assert sharpe_backend(pd_covar.to_numpy(), means, variable) == ["SLSQP"]
    variable_weights, variable_outcome = opt.wrapper_maximize_portfolio_sharpe(
        pd_covar, expected_returns, variable, optimiser_config=config)
    assert isinstance(variable_weights, pd.Series)
    assert isinstance(variable_outcome, OptimizationOutcome)

    return_weights, return_outcome = opt.wrapper_min_variance_target_return(
        pd_covar, expected_returns, target_return=0.055,
        constraints=constraints, optimiser_config=config,
    )
    vol_weights, vol_outcome = opt.wrapper_max_return_target_vol(
        pd_covar, expected_returns, target_vol=0.12,
        constraints=constraints, optimiser_config=config,
    )
    tactical_constraints = replace(
        constraints, benchmark_weights=benchmark, tracking_err_vol_constraint=0.03,
    )
    tactical_weights, tactical_outcome = opt.wrapper_maximise_alpha_over_tre(
        pd_covar, alphas, benchmark, tactical_constraints, optimiser_config=config,
    )
    yield_weights, yield_outcome = opt.wrapper_maximise_alpha_with_target_return(
        pd_covar, alphas, yields=expected_returns, target_return=0.05,
        constraints=tactical_constraints, benchmark_weights=benchmark,
        optimiser_config=config,
    )
    utility_constraints = replace(
        tactical_constraints,
        constraint_enforcement_type=opt.ConstraintEnforcementType.UTILITY_CONSTRAINTS,
        tre_utility_weight=5.0,
    )
    soft_weights, soft_outcome = opt.wrapper_maximise_alpha_over_tre(
        pd_covar, alphas, benchmark, utility_constraints, optimiser_config=config,
    )

    # All five calls return complete weights and an accepted, compliant outcome.
    for result in (return_outcome, vol_outcome, tactical_outcome, yield_outcome, soft_outcome):
        check_outcome(result)
    for series in (return_weights, vol_weights, tactical_weights, yield_weights, soft_weights):
        check_weights(series)
    # The 0.055 return floor binds; with full investment it fixes the interior solution.
    floor = equality_minimum_variance(variances, [np.ones(5), means], [1.0, 0.055])
    assert floor.min() > 0 and floor.max() < 1
    np.testing.assert_allclose(return_weights, floor, rtol=0.0, atol=3e-5)
    assert abs(means @ return_weights - 0.055) < 1e-6
    # The 0.12 budget is annual volatility because the covariance is annual, and it binds.
    assert abs(diagonal_tracking_error(vol_weights, np.zeros(5), ANNUAL_VOLS) - 0.12) < 2e-6
    # Hard alpha/TE maximises active alpha on the 3% ellipsoid: the active direction is
    # D^-1 (alpha - lambda 1), zero-sum, scaled to 0.03 tracking error.
    alpha = alphas.to_numpy()
    direction = inverse * (alpha - np.sum(inverse * alpha) / np.sum(inverse))
    scale = 0.03 / diagonal_tracking_error(direction, np.zeros(5), ANNUAL_VOLS)
    ellipsoid = benchmark.to_numpy() + scale * direction
    assert ellipsoid.min() > 0 and ellipsoid.max() < 1 and abs(ellipsoid.sum() - 1) < 1e-15
    np.testing.assert_allclose(tactical_weights, ellipsoid, rtol=0.0, atol=3e-5)
    # The utility form does not retain the 3% cap, while its hard rows stay compliant.
    assert diagonal_tracking_error(soft_weights, benchmark, ANNUAL_VOLS) > 0.03 + 1e-3
    # The yield example keeps its 0.05 floor on the supplied yields and the hard 3% budget.
    assert yield_weights @ expected_returns >= 0.05 - 1e-6
    assert diagonal_tracking_error(yield_weights, benchmark, ANNUAL_VOLS) <= 0.03 + 2e-6

    dates = pd.date_range("2020-12-31", "2025-01-31", freq="ME")
    step = np.arange(len(dates), dtype=float)
    monthly_changes = (
        expected_returns.to_numpy()[None, :] / 12
        + 0.008 * np.sin(step[:, None] * np.arange(1, 6)[None, :])
    )
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(monthly_changes, axis=0)), index=dates, columns=tickers,
    )
    decision_dates = pd.date_range("2023-03-31", "2024-12-31", freq="QE")
    covar_dict = {date: pd_covar.copy() for date in decision_dates}
    rolling_weights = opt.compute_rolling_optimal_weights(
        prices, constraints, covar_dict,
        portfolio_objective=opt.PortfolioObjective.MIN_VARIANCE, optimiser_config=config,
    )
    return_forecasts = pd.DataFrame(
        np.broadcast_to(expected_returns, (len(decision_dates), len(tickers))),
        index=decision_dates, columns=tickers,
    )
    rolling_utility = opt.rolling_quadratic_optimisation(
        prices, constraints, covar_dict,
        portfolio_objective=opt.PortfolioObjective.QUADRATIC_UTILITY,
        expected_returns=return_forecasts, carra=5.0, optimiser_config=config,
    )
    portfolio = opt.backtest_rolling_optimal_portfolio(
        prices, constraints, covar_dict,
        portfolio_objective=opt.PortfolioObjective.MIN_VARIANCE,
        optimiser_config=config, rebalancing_costs=0.0,
        weight_implementation_lag=1, ticker="Synthetic minimum variance",
    )

    # Monthly levels through January 2025: eight quarterly decisions, each with a later price.
    assert dates[-1] == pd.Timestamp("2025-01-31") and len(decision_dates) == 8
    assert (decision_dates < dates[-1]).all() and set(decision_dates) <= set(dates)
    # Both tables: the supplied schedule, eight rows by five assets, each row the single-date
    # solution; the dispatcher returns a weight table, not a PortfolioOptimisationResult.
    for table, single in ((rolling_weights, weights), (rolling_utility, utility_weights)):
        assert isinstance(table, pd.DataFrame) and table.shape == (8, 5)
        assert list(table.columns) == TICKERS
        pd.testing.assert_index_equal(table.index, decision_dates, check_names=False)
        np.testing.assert_allclose(table, np.broadcast_to(single, table.shape), rtol=0.0,
                                   atol=3e-6)
    assert not isinstance(rolling_weights, opt.PortfolioOptimisationResult)
    # The adapter returns qis.PortfolioData and executes each decision one observation later.
    assert isinstance(portfolio, qis.PortfolioData)
    rebalancing = portfolio.is_rebalancing
    pd.testing.assert_index_equal(rebalancing[rebalancing].index,
                                  decision_dates + pd.offsets.MonthEnd(1), check_names=False)
    # Later prices change no earlier dispatch result; with estimated means they do change later
    # quadratic-utility rows, so the comparison can detect look-ahead.
    cutoff = pd.Timestamp("2023-12-31")
    changed = prices.copy()
    later = changed.index > cutoff
    changed.loc[later, "Equity A"] *= np.linspace(1.1, 2.0, later.sum())
    for objective in (opt.PortfolioObjective.MIN_VARIANCE,
                      opt.PortfolioObjective.QUADRATIC_UTILITY):
        arguments = dict(constraints=constraints, covar_dict=covar_dict,
                         portfolio_objective=objective, returns_freq="ME", span=12,
                         optimiser_config=config)
        before = opt.compute_rolling_optimal_weights(prices, **arguments)
        after = opt.compute_rolling_optimal_weights(changed, **arguments)
        np.testing.assert_allclose(before.loc[:cutoff], after.loc[:cutoff], rtol=0.0, atol=2e-7)
        if objective == opt.PortfolioObjective.QUADRATIC_UTILITY:
            assert not np.allclose(before.loc[cutoff:].iloc[1:], after.loc[cutoff:].iloc[1:],
                                   atol=1e-4)
            # The dispatcher's means are annualised EWMA log-return means on the covariance
            # keys, and it solves with carra=0.5.
            estimated = ewma_log_means(prices, span=12, periods_per_year=12).loc[decision_dates]
            direct = opt.rolling_quadratic_optimisation(
                prices, constraints, covar_dict, portfolio_objective=objective,
                expected_returns=estimated, carra=0.5, optimiser_config=config)
            np.testing.assert_allclose(before, direct, rtol=0.0, atol=1e-7)
    # Every enum member takes its documented route with its documented parameter scope; there
    # is no minimum-tracking-error member, and an unsupported objective raises.
    assert {member.name for member in opt.PortfolioObjective} == set(ROUTES)
    for objective in ROUTES:
        check_dispatch_route(objective, prices, constraints, covar_dict, config,
                             return_forecasts)
    assert_raises(NotImplementedError, opt.compute_rolling_optimal_weights, prices, constraints,
                  covar_dict, portfolio_objective="MINIMUM_TRACKING_ERROR")

    covar = pd_covar.to_numpy()
    w = cvx.Variable(len(tickers))
    constraints.set_cvx_all_constraints(w, covar)  # → list of cvxpy constraints
    constraints.set_scipy_constraints(covar)       # → (list of dicts, bounds) for scipy
    constraints.set_pyrb_constraints(covar)        # → (bounds, C, d) for the risk-budgeting solver

    cvx_rows = constraints.set_cvx_all_constraints(w, covar)
    scipy_rows, scipy_bounds = constraints.set_scipy_constraints(covar)
    pyrb_bounds, pyrb_c, pyrb_d = constraints.set_pyrb_constraints(covar)

    # CVXPY: a list of expressions; SciPy: constraint dictionaries and bounds; risk budgeting:
    # bounds and the group matrices. None of them is a solved portfolio.
    assert isinstance(cvx_rows, list) and cvx_rows
    assert all(isinstance(row, cvx.constraints.constraint.Constraint) for row in cvx_rows)
    assert isinstance(scipy_rows, list) and scipy_rows
    assert all(isinstance(row, dict) and "fun" in row for row in scipy_rows)
    assert len(scipy_bounds) == 5 and pyrb_bounds is not None
    assert w.value is None

    gluc = GroupLowerUpperConstraints(
        group_loadings=pd.DataFrame({
            "Equities":  [1, 1, 0, 0, 0],
            "Bonds":     [0, 0, 1, 1, 0],
            "Gold":      [0, 0, 0, 0, 1],
        }, index=tickers, dtype=float),
        group_min_allocation=pd.Series({"Equities": 0.30, "Bonds": 0.20, "Gold": 0.05}),
        group_max_allocation=pd.Series({"Equities": 0.60, "Bonds": 0.50, "Gold": 0.20}),
    )

    bdc = BenchmarkDeviationConstraints(
        factor_loading_mat=pd.DataFrame({
            "Tech":    [1, 1, 0, 0, 0],
            "Finance": [0, 0, 1, 1, 0],
            "Energy":  [0, 0, 0, 0, 1],
        }, index=tickers, dtype=float),
        factor_max_deviation=pd.Series({"Tech": 0.05, "Finance": 0.05, "Energy": 0.03}),
    )

    group_constraints = replace(
        constraints, group_lower_upper_constraints=gluc,
        sector_deviation_constraints=bdc, benchmark_weights=benchmark,
    )
    group_weights, group_outcome = opt.wrapper_quadratic_optimisation(
        pd_covar, group_constraints, optimiser_config=config,
    )
    assert group_outcome.accepted and group_outcome.compliant

    # Both sets hold, from the published loadings: absolute group allocations and active
    # deviations from the benchmark.
    check_outcome(group_outcome)
    check_weights(group_weights)
    loadings = np.array([[1, 1, 0, 0, 0], [0, 0, 1, 1, 0], [0, 0, 0, 0, 1]], dtype=float)
    held = loadings @ group_weights.to_numpy()
    assert (held >= np.array([0.30, 0.20, 0.05]) - 1e-6).all()
    assert (held <= np.array([0.60, 0.50, 0.20]) + 1e-6).all()
    deviations = loadings @ (group_weights - benchmark).to_numpy()
    assert (np.abs(deviations) <= np.array([0.05, 0.05, 0.03]) + 1e-6).all()
    # Neither set is slack: bonds stop at 0.45 = 0.40 + 0.05, inside their 0.50 group cap, and
    # Gold at 0.17 = 0.20 - 0.03, where unconstrained minimum variance holds 0.735 and 0.057.
    np.testing.assert_allclose(deviations[1:], [0.05, -0.03], rtol=0.0, atol=1e-6)
    assert held[1] < 0.50 - 0.04 and inverse_variance[2:4].sum() > 0.73
    print("optimization_module_readme: all page statements verified.")


if __name__ == '__main__':
    main()
