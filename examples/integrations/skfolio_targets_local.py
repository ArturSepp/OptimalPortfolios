"""Optional offline skfolio targets evaluated by QIS, with an explicit execution contract.

Run this repository-only example after installing the pinned optional packages in the guide.
The established *_local.py convention excludes optional/manual examples from core discovery.
No skfolio import occurs until fitting is requested; prices and labelled targets remain neutral.
"""
from importlib.metadata import version
import json

import numpy as np
import pandas as pd
import qis


def _ordered_dates(index, name, *, timezone):
    """Require an explicit ordered, unique, nonmissing calendar without timezone conversion."""
    if (not isinstance(index, pd.DatetimeIndex) or index.empty or index.hasnans
            or not index.is_unique or not index.is_monotonic_increasing
            or index.tz != timezone):
        raise ValueError(f'{name} needs ordered unique timestamps in the price timezone')


def evaluate_targets(prices, targets, *, information_dates, execution_dates,
                     return_units, moment_units, excluded_assets=(), transaction_costs=0.0):
    """Evaluate labelled external targets using the canonical QIS holdings/cash engine.

    Args:
        prices: Positive total-return marks on a sorted DatetimeIndex; NaNs freeze trades.
        targets: Long-only fully invested target rows indexed by decision timestamp.
            Columns label the fitted asset universe; no positional conversion is performed.
        information_dates: Datetime Series indexed exactly like targets, recording the last
            observation used for each decision. It must strictly precede execution.
        execution_dates: Datetime Series with the same index, naming exact price observations
            when targets are implemented. Decisions may be earlier than the evaluation panel.
        return_units: Required declaration, exactly 'simple_fraction'. Percent/log inputs fail.
        moment_units: Required declaration, exactly 'per_observation'; no moment rescaling.
        excluded_assets: Explicit price columns omitted from the fitted universe, held at zero.
            All other price columns must be present, including zero pre-entry targets.
        transaction_costs: Nonnegative proportional charge on each actual QIS trade, including
            entry; 0.001 means ten basis points. No native skfolio fee is translated.

    Returns:
        QIS PortfolioData with NAV, units, executed weights and realised cash costs. Prices
        before the first execution are estimation history, not a zero-return backtest sleeve.

    Raises:
        ValueError: A unit, universe, date, eligibility, mark or target contract is violated.
    """
    if return_units != 'simple_fraction' or moment_units != 'per_observation':
        raise ValueError('Require simple_fraction returns and per_observation moments')
    _ordered_dates(prices.index, 'prices', timezone=getattr(prices.index, 'tz', None))
    _ordered_dates(targets.index, 'decisions', timezone=prices.index.tz)
    if (len(prices) < 2 or prices.columns.empty or not prices.columns.is_unique
            or not targets.columns.is_unique or targets.columns.empty):
        raise ValueError('Require nonempty unique asset labels and at least two price rows')
    for name, series in [('information', information_dates), ('execution', execution_dates)]:
        if (not isinstance(series, pd.Series) or not series.index.equals(targets.index)
                or not pd.api.types.is_datetime64_any_dtype(series.dtype)):
            raise ValueError(f'{name} dates must be a datetime Series indexed by decisions')
    execution = pd.DatetimeIndex(execution_dates)
    information = pd.DatetimeIndex(information_dates)
    _ordered_dates(execution, 'executions', timezone=prices.index.tz)
    if (information.hasnans or information.tz != prices.index.tz
            or (information > targets.index).any() or (information >= execution).any()
            or (targets.index > execution).any()):
        raise ValueError('Require information <= decision <= execution; information < execution')
    if (not execution.isin(prices.index).all() or execution[-1] >= prices.index[-1]):
        raise ValueError('Executions need exact price dates with a subsequent observation')
    excluded = pd.Index(excluded_assets)
    expected = prices.columns.difference(excluded, sort=False)
    if (not excluded.is_unique or not excluded.isin(prices.columns).all()
            or set(targets.columns) != set(expected)):
        raise ValueError('Target assets must match prices minus explicit exclusions')
    weights = targets.to_numpy(dtype=float)
    marks = prices.to_numpy(dtype=float)
    if (not np.isfinite(weights).all() or (weights < 0).any()
            or not np.allclose(weights.sum(axis=1), 1.0, rtol=0, atol=1e-8)):
        raise ValueError('Targets must be finite, long-only and sum to one without renormalising')
    if np.isinf(marks).any() or (marks[np.isfinite(marks)] <= 0).any():
        raise ValueError('Quoted marks must be finite and strictly positive; NaNs mean unquoted')
    if not np.isfinite(transaction_costs) or transaction_costs < 0:
        raise ValueError('Trade cost must be finite and nonnegative')
    schedule = targets.reindex(columns=prices.columns, fill_value=0.0).copy()
    schedule.index = execution
    for asset in schedule:
        first_quote = prices[asset].first_valid_index()
        active_dates = execution[schedule[asset].to_numpy() != 0]
        if len(active_dates) and (first_quote is None or (active_dates < first_quote).any()):
            raise ValueError(f'Nonzero target before the first quote for {asset}')
    # Execution dates are already resolved. QIS owns drift, cash, trade-date fees and freezes.
    return qis.backtest_model_portfolio(
        prices=prices.loc[execution[0]:], weights=schedule, rebalancing_freq=None,
        is_rebalanced_at_first_date=True, weight_implementation_lag=0,
        rebalancing_costs=transaction_costs, initial_nav=100.0, ticker='Skfolio targets')


def run_example():
    """Fit two chronological allocations, evaluate gross/net and return printable evidence."""
    try:
        from skfolio import RiskMeasure
        from skfolio.optimization import MeanRisk, ObjectiveFunction
    except ModuleNotFoundError as error:
        if error.name != 'skfolio':
            raise
        raise ModuleNotFoundError(
            'Optional example requires skfolio==1.9.3; see docs/skfolio_interoperability.md'
        ) from error
    from optimalportfolios.tests.data.multiasset import load_multiasset_data

    assets = ['Global Bonds', 'US Treasuries', 'Global HY Bonds', 'MSCI World USD',
              'Hedge Funds', 'Commodities EX-Precious']
    prices = load_multiasset_data().prices.loc['2010-01-31':'2022-12-31', assets]
    decisions = pd.DatetimeIndex(['2018-12-31', '2020-12-31'])
    rows, dates = [], []
    for decision in decisions:
        training = qis.to_returns(prices=prices.loc[:decision], is_log_returns=False,
                                  ffill_nans=False, drop_first=True)
        if not np.isfinite(training.to_numpy()).all():
            raise ValueError('Fitting needs complete training returns; choose eligible assets')
        model = MeanRisk(risk_measure=RiskMeasure.VARIANCE,
                         objective_function=ObjectiveFunction.MINIMIZE_RISK,
                         min_weights=0.0, max_weights=0.6, solver='CLARABEL')
        model.fit(training)
        # Labels are from the actual fit, never the possibly permuted evaluation panel.
        rows.append(pd.Series(model.weights_, index=model.feature_names_in_, name=decision))
        dates.append(prices.index[prices.index > decision][0])
    targets = pd.DataFrame(rows)
    information = pd.Series(decisions, index=decisions)
    execution = pd.Series(pd.DatetimeIndex(dates), index=decisions)
    contract = dict(information_dates=information, execution_dates=execution,
                    return_units='simple_fraction', moment_units='per_observation')
    gross = evaluate_targets(prices, targets, **contract)
    net = evaluate_targets(prices, targets, transaction_costs=0.001, **contract)
    return {'versions': {name: version(name) for name in ['qis', 'optimalportfolios', 'skfolio']},
            'assets': len(assets), 'decisions': decisions.astype(str).tolist(),
            'information_cutoffs': information.astype(str).tolist(),
            'executions': execution.astype(str).tolist(),
            'observations': len(gross.nav), 'gross_final_nav': float(gross.nav.iloc[-1]),
            'net_final_nav': float(net.nav.iloc[-1]),
            'realised_trade_cost': float(net.realized_costs.to_numpy().sum()),
            'return_units': 'simple_fraction', 'moment_units': 'per_observation',
            'trade_cost_rate': 0.001, 'targets': targets.to_dict(orient='split') | {
                'index': decisions.astype(str).tolist()}}


def run_executed_state_example():
    """Run the optional holdings-feedback path with the existing single-date solver.

    Returns:
        ExecutedRollingResult for QIS reporting and independent ledger verification.
    """
    import numpy as np
    import qis
    import optimalportfolios as opt
    from examples.docs.rolling_backtests import ledger

    dates = pd.bdate_range('2024-01-02', periods=9)
    prices = pd.DataFrame({'A': [100., 110., 121., 121., 133.1, 140., 145., 150., 152.],
                           'B': [100., 90., 90., 99., 99., 101., 102., 103., 104.]}, index=dates)

    def decide(history, state):
        """Use actual risky weights, including cost funding, as the turnover baseline."""
        covariance = pd.DataFrame([[.04, 0.], [0., .01]],
                                  index=history.columns, columns=history.columns)
        return opt.wrapper_quadratic_optimisation(
            pd_covar=covariance, constraints=opt.Constraints(turnover_constraint=.2),
            weights_0=state.weights if state.units.abs().sum() else None,
            context=str(state.date))

    result = opt.backtest_with_qis_state(
        prices, dates[[0, 3, 6]], decide, implementation_lag=1, transaction_costs=.001)
    assert isinstance(result.portfolio, qis.PortfolioData)
    units, nav, cash, costs = ledger(prices, result.targets, lag=1, rate=.001)
    np.testing.assert_allclose(result.portfolio.units, units, atol=1e-12, rtol=0)
    np.testing.assert_allclose(result.portfolio.nav, nav, atol=1e-10, rtol=0)
    np.testing.assert_allclose(result.portfolio.realized_costs, costs, atol=1e-13, rtol=0)
    for state in result.states:
        pos = prices.index.get_loc(state.date)
        np.testing.assert_allclose([state.nav, state.cash],
                                   [nav.iloc[pos], cash.iloc[pos]], atol=1e-10)
    assert result.states[1].cash < 0
    return result



def run_chronological_selection_example():
    """Select covariance/alpha settings on research folds and evaluate one forward portfolio.

    Returns:
        PortfolioSelectionResult with independent fold scores and a separate outer L2 path.
    """
    import numpy as np
    import qis
    import optimalportfolios as opt
    from examples.docs.rolling_backtests import ledger

    dates = pd.date_range('2022-01-05', periods=80, freq='W-WED')
    t = np.arange(len(dates))
    prices = pd.DataFrame({'A': 100 * np.exp(.002 * t + .04 * np.sin(t / 3)),
                           'B': 100 * np.exp(.001 * t + .03 * np.cos(t / 4))}, index=dates)
    grid = {f'span{span}_alpha{alpha}': {'covariance_span': span, 'alpha_strength': alpha}
            for span in (8, 16) for alpha in (0., .5)}

    def fit(training, configuration):
        """Create covariance estimates and the mean prior using only this training panel."""
        covariance = opt.estimate_current_ewma_covar(
            training, returns_freq='W-WED', span=configuration['covariance_span'],
            demean=True, apply_an_factor=False)
        # Fractional log-return moments per weekly observation, without annualisation.
        means = (qis.to_returns(training, is_log_returns=True).dropna().mean()
                 * configuration['alpha_strength'])

        def decide(history, state):
            """Use the frozen moments and actual executed weights for constrained turnover."""
            return opt.wrapper_quadratic_optimisation(
                pd_covar=covariance, means=means, carra=10.,
                portfolio_objective=opt.PortfolioObjective.QUADRATIC_UTILITY,
                constraints=opt.Constraints(turnover_constraint=.3),
                weights_0=state.weights if state.units.abs().sum() else None,
                context=str(state.date))
        return decide

    folds = (opt.ChronologicalFold(dates[25], dates[[26, 30]], dates[35]),
             opt.ChronologicalFold(dates[41], dates[[42, 46]], dates[51]))
    outer = opt.ChronologicalFold(dates[61], dates[[62, 66, 70]], dates[79])
    result = opt.select_portfolio_model(
        prices, grid, fit, folds, outer, label_horizon=0, embargo=1,
        implementation_lag=1, transaction_costs=.001)
    evaluation = prices.loc[dates[62]:dates[79]]
    units, nav, _, costs = ledger(evaluation, result.outer.targets, lag=1, rate=.001)
    np.testing.assert_allclose(result.outer.portfolio.units, units, atol=1e-12, rtol=0)
    np.testing.assert_allclose(result.outer.portfolio.nav, nav, atol=1e-10, rtol=0)
    np.testing.assert_allclose(result.outer.portfolio.realized_costs, costs, atol=1e-13, rtol=0)
    assert len(result.outer.targets) == 3
    assert result.outer_score == float(qis.compute_total_return(pd.Series(
        [100., nav.iloc[-1]], index=nav.index[[0, -1]])))
    return result

if __name__ == '__main__':
    print(json.dumps(run_example(), indent=2, sort_keys=True))
