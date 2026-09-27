"""Canonical script of docs/implied_risk_budgets.md.

The page shows excerpts of ``main``; every number and property it states is asserted here
against a reference computed a different way: the risk shares recomputed by qis, the forward
risk-budgeting solve, the first-order conditions of a held asset, an explicit EWMA loop and its
closed-form row weights. The script runs offline after ``pip install optimalportfolios`` and
needs no data file or random seed:

    python -m examples.docs.implied_risk_budgets

``exhibit`` draws the page's figure; ``tools/docs_analytics/teaching.py`` calls it with the
constants below and records their values.
"""
import inspect
import warnings

import numpy as np
import pandas as pd
import qis

import optimalportfolios as op
from optimalportfolios.optimization.risk_allocation import risk_budgeting as rb_module

ASSETS = ['Equity', 'Bonds', 'Gold']
VOLS = [0.16, 0.07, 0.15]  # annual volatilities
EQUITY_GOLD = 0.1  # correlations that stay fixed
BONDS_GOLD = 0.2
TARGET_WEIGHTS = [0.40, 0.45, 0.15]
STOCK_BOND = 0.2  # stock-bond correlation of the one-date examples
HEDGE_STOCK_BOND = -0.7  # a stock-bond correlation at which Bonds hedge the target portfolio
STOCK_BOND_PATH = [-0.7, 0.4]  # first and last stock-bond correlation of the rolling example
REBALANCES = 12
FIRST_REBALANCE = '2023-03-31'
# The four-sleeve example of examples/solvers/inverse_risk_budget_bonds.py.
BOUNDARY_ASSETS = ['Equity', 'Other', 'Bond A', 'Bond B']
BOUNDARY_TARGET = [0.65, 0.198925, 0.1253, 0.025775]
BOUNDARY_VOLS = [0.20, 0.12, 0.06, 0.07]
BOUNDARY_EQUITY_BOND_B = [-0.3, 0.2, 0.2]  # equity/Bond B correlation at each of three dates
TOLERANCE = 1e-3  # the fit's tolerance on the largest average-weight gap


def covariance(stock_bond: float) -> pd.DataFrame:
    """Return the annual covariance of the three assets for one stock-bond correlation."""
    corr = np.array([[1.0, stock_bond, EQUITY_GOLD],
                     [stock_bond, 1.0, BONDS_GOLD],
                     [EQUITY_GOLD, BONDS_GOLD, 1.0]])
    return pd.DataFrame(np.outer(VOLS, VOLS) * corr, index=ASSETS, columns=ASSETS)


def implied_budgets(weights: pd.Series, covar: pd.DataFrame) -> pd.Series:
    """Return the risk shares w_i (Sigma w)_i / (w' Sigma w) of the weights."""
    w = weights.to_numpy(dtype=float)
    sigma = covar.reindex(index=weights.index, columns=weights.index).to_numpy(dtype=float)
    return pd.Series(w * (sigma @ w) / (w @ sigma @ w), index=weights.index)


def rebalance_dates() -> pd.DatetimeIndex:
    """Return the quarter-end rebalance dates of the examples."""
    return pd.date_range(FIRST_REBALANCE, periods=REBALANCES, freq='QE')


def rolling_covariances(dates: pd.DatetimeIndex) -> dict:
    """Return covariances whose stock-bond correlation drifts linearly along STOCK_BOND_PATH."""
    path = np.linspace(STOCK_BOND_PATH[0], STOCK_BOND_PATH[1], len(dates))
    return {date: covariance(rho) for date, rho in zip(dates, path)}


def held_constraints(target: pd.Series, held: list) -> op.Constraints:
    """Long-only constraints whose equal bounds hold the named assets at their target weights."""
    minimum = pd.Series(0.0, index=target.index)
    maximum = pd.Series(1.0, index=target.index)
    minimum[held] = target[held]
    maximum[held] = target[held]
    return op.Constraints(is_long_only=True, min_weights=minimum, max_weights=maximum)


def held_budgets(target: pd.Series, covar: pd.DataFrame, held: list) -> pd.Series:
    """First-order budgets b_i = r_i + r_H w_i / (1 - w_H) of the free assets; zero if held."""
    shares = implied_budgets(target, covar)
    is_held = target.index.isin(held)
    budgets = shares + shares[is_held].sum() * target / (1.0 - target[is_held].sum())
    return budgets.where(~is_held, 0.0)


def explicit_ewma(path: pd.DataFrame, span: float) -> pd.Series:
    """Run m_t = lambda m_(t-1) + (1 - lambda) w_t from m_0 = w_0 and return the last m_t."""
    decay = 1.0 - 2.0 / (span + 1.0)
    state = path.iloc[0].to_numpy(dtype=float)
    for row in path.iloc[1:].to_numpy(dtype=float):
        state = decay * state + (1.0 - decay) * row
    return pd.Series(state, index=path.columns)


def boundary_inputs() -> tuple:
    """Prices, target weights and three covariances of the four-sleeve boundary example."""
    dates = pd.DatetimeIndex(['2020-03-31', '2023-03-31', '2026-03-31'])
    covars = {}
    for date, equity_bond_b in zip(dates, BOUNDARY_EQUITY_BOND_B):
        corr = np.array([[1.0, 0.4, -0.7, equity_bond_b],
                         [0.4, 1.0, -0.2, 0.0],
                         [-0.7, -0.2, 1.0, 0.0],
                         [equity_bond_b, 0.0, 0.0, 1.0]])
        covars[date] = pd.DataFrame(np.outer(BOUNDARY_VOLS, BOUNDARY_VOLS) * corr,
                                    index=BOUNDARY_ASSETS, columns=BOUNDARY_ASSETS)
    prices = pd.DataFrame(1.0, index=dates, columns=BOUNDARY_ASSETS)
    return prices, pd.Series(BOUNDARY_TARGET, index=BOUNDARY_ASSETS), covars


def assert_raises(error: type, function, **arguments) -> None:
    """Fail unless ``function(**arguments)`` raises ``error``."""
    try:
        function(**arguments)
    except error:
        return
    raise AssertionError(f'expected {error.__name__}')


def caught_warnings(function, **arguments) -> tuple:
    """Call ``function`` and return its result with the messages of the warnings it issued."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        result = function(**arguments)
    return result, [str(item.message) for item in caught]


def main() -> None:
    """Run the page's blocks in order and assert every number and property it states."""
    covar = covariance(STOCK_BOND)
    target = pd.Series(TARGET_WEIGHTS, index=ASSETS)
    budgets = implied_budgets(target, covar)
    weights = op.wrapper_risk_budgeting(pd_covar=covar,
                                        constraints=op.Constraints(is_long_only=True),
                                        risk_budget=budgets)
    print(pd.concat([target.rename('target weight'), budgets.rename('implied budget'),
                     weights.rename('forward weight')], axis=1).round(4))

    # The implied budgets are the Euler risk shares: they sum to one and equal the shares that
    # qis computes; 40% of capital in Equity carries 66.6% of the risk.
    assert abs(budgets.sum() - 1.0) < 1e-14 and (budgets > 0.0).all()
    np.testing.assert_allclose(budgets, qis.compute_portfolio_risk_contribution_ratios(
        weights=target, covar=covar), atol=1e-14)
    assert budgets.round(3).tolist() == [0.666, 0.220, 0.114]
    # They do not depend on the scale of the covariance.
    np.testing.assert_allclose(implied_budgets(target, 12.0 * covar), budgets, atol=1e-15)
    # Round trip: the forward risk-budgeting solve returns the target weights.
    assert np.abs(weights - target).max() < 1e-8

    start = pd.Series([0.50, 0.30, 0.20], index=ASSETS)
    forward = op.wrapper_risk_budgeting(pd_covar=covar,
                                        constraints=op.Constraints(is_long_only=True),
                                        risk_budget=start)
    print(implied_budgets(forward, covar).round(6).tolist())  # [0.5, 0.3, 0.2]

    # The other direction: budgets to weights to budgets.
    assert np.abs(implied_budgets(forward, covar) - start).max() < 1e-8
    assert implied_budgets(forward, covar).round(6).tolist() == [0.5, 0.3, 0.2]

    dates = rebalance_dates()
    prices = pd.DataFrame(100.0, index=dates, columns=ASSETS)
    fitted = op.solve_for_risk_budgets_from_given_weights(
        prices=prices, given_weights=target, covar_dict={date: covar for date in dates})
    print(fitted.round(4).tolist())  # the implied budgets

    # With one covariance on every date, the fit returns the closed-form implied budgets: its
    # seed is their average, and the first forward solve already meets the tolerance.
    assert np.abs(fitted - budgets).max() < 1e-8
    signature = inspect.signature(op.solve_for_risk_budgets_from_given_weights).parameters
    assert signature['min_risk_budget'].default == 1e-4
    assert signature['max_risk_budget'].default == 0.99
    assert signature['ewma_span'].default is None
    assert signature['fixed_weight_assets'].default is None
    assert rb_module._INVERSE_MEAN_WEIGHT_TOL == 1e-4
    assert rb_module._INVERSE_MAX_WEIGHT_TOL == TOLERANCE
    assert rb_module._NEGATIVE_RC_SHARE_WARNING == 0.5
    diagnostics = rb_module._target_risk_contributions(target, {date: covar for date in dates},
                                                       None)
    np.testing.assert_allclose(diagnostics['average_rc'], budgets, atol=1e-14)

    hedge = covariance(HEDGE_STOCK_BOND)
    print((hedge @ target).round(5).tolist())  # the marginal risk of Bonds is negative
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        held = op.solve_for_risk_budgets_from_given_weights(
            prices=prices, given_weights=target, covar_dict={date: hedge for date in dates})
    print(held.round(2).tolist())  # [0.79, 0.0, 0.21]
    print(str(caught[0].message)[:88])

    # Bonds hedge the target: (Sigma w)_Bonds < 0, so its implied budget is negative and no
    # non-negative budget reproduces the target. The hold rule gives it a zero budget and
    # holds its 45%; the free budgets are the first-order budgets of the held asset.
    marginal = hedge.to_numpy() @ target.to_numpy()
    assert marginal[1] < 0.0 < min(marginal[0], marginal[2]) and round(marginal[1], 5) == -0.00062
    shares = implied_budgets(target, hedge)
    assert shares['Bonds'] < 0.0 and round(shares['Bonds'], 3) == -0.083
    assert len(caught) == 1 and 'pinned assets with non-positive average marginal' in str(
        caught[0].message)
    assert 'Bonds: central weight fixed at 0.4500' in str(caught[0].message)
    assert held['Bonds'] == 0.0 and abs(held.sum() - 1.0) < 1e-12
    assert held.round(2).tolist() == [0.79, 0.0, 0.21]
    reference = held_budgets(target, hedge, ['Bonds'])
    assert np.abs(held - reference).max() < TOLERANCE
    assert round(shares['Equity'], 3) == 0.846
    assert reference.round(3).tolist() == [0.786, 0.0, 0.214]
    np.testing.assert_allclose(reference[['Equity', 'Gold']],
                               shares[['Equity', 'Gold']] + shares['Bonds'] * target[
                                   ['Equity', 'Gold']] / (1.0 - target['Bonds']), atol=1e-15)
    # The first-order budgets reproduce the target through the forward solve with the pin.
    exact = op.wrapper_risk_budgeting(pd_covar=hedge,
                                      constraints=held_constraints(target, ['Bonds']),
                                      risk_budget=reference)
    assert np.abs(exact - target).max() < 1e-8

    lower = pd.Series({'Equity': 0.0, 'Bonds': target['Bonds'], 'Gold': 0.0})
    upper = pd.Series({'Equity': 1.0, 'Bonds': target['Bonds'], 'Gold': 1.0})
    pin = op.Constraints(is_long_only=True, min_weights=lower, max_weights=upper)
    reused = op.wrapper_risk_budgeting(pd_covar=hedge, constraints=pin, risk_budget=held)
    dropped = op.wrapper_risk_budgeting(pd_covar=hedge,
                                        constraints=op.Constraints(is_long_only=True),
                                        risk_budget=held)
    print(pd.concat([reused.rename('with the pin'), dropped.rename('without it')],
                    axis=1).round(3))

    # Pitfall: a zero reported budget does not mean a zero weight. With the pin the budgets
    # reproduce the target; without it the wrapper excludes the zero-budget asset.
    assert np.abs(reused - target).max() < TOLERANCE
    assert dropped['Bonds'] == 0.0 and abs(dropped.sum() - 1.0) < 1e-9
    assert dropped.round(2).tolist() == [0.66, 0.0, 0.34]

    gold_held = op.solve_for_risk_budgets_from_given_weights(
        prices=prices, given_weights=target, covar_dict={date: covar for date in dates},
        fixed_weight_assets=['Gold'])
    print(gold_held.round(2).tolist())  # [0.72, 0.28, 0.0]

    # An explicit pin: Gold carries risk (positive implied budget) but is held at 15% with a
    # zero budget; the other two budgets follow the same first-order rule.
    assert budgets['Gold'] > 0.0 and gold_held['Gold'] == 0.0
    assert gold_held.round(2).tolist() == [0.72, 0.28, 0.0]
    assert np.abs(gold_held - held_budgets(target, covar, ['Gold'])).max() < TOLERANCE
    kept = op.wrapper_risk_budgeting(pd_covar=covar,
                                     constraints=held_constraints(target, ['Gold']),
                                     risk_budget=gold_held)
    assert np.abs(kept - target).max() < TOLERANCE
    # Invalid pins raise before any fit; a pin that leaves one free asset gives it the budget.
    solve = op.solve_for_risk_budgets_from_given_weights
    inputs = dict(prices=prices, given_weights=target, covar_dict={date: covar for date in dates})
    assert_raises(TypeError, solve, fixed_weight_assets='Gold', **inputs)
    assert_raises(ValueError, solve, fixed_weight_assets=['Cash'], **inputs)
    assert_raises(ValueError, solve, fixed_weight_assets=['Gold', 'Gold'], **inputs)
    assert_raises(ValueError, solve, fixed_weight_assets=ASSETS, **inputs)
    zero_gold = dict(inputs, given_weights=pd.Series([0.5, 0.5, 0.0], index=ASSETS))
    assert_raises(ValueError, solve, fixed_weight_assets=['Gold'], **zero_gold)
    one_free = solve(fixed_weight_assets=['Bonds', 'Gold'], **inputs)
    assert one_free.tolist() == [1.0, 0.0, 0.0]
    assert_raises(ValueError, solve, prices=prices, covar_dict=inputs['covar_dict'],
                  given_weights=pd.Series([0.5, 0.3, 0.3], index=ASSETS))
    # A missing variance makes an averaged share non-finite, which raises.
    missing = covar.copy()
    missing.loc['Gold', 'Gold'] = np.nan
    assert_raises(ValueError, solve, prices=prices, given_weights=target,
                  covar_dict={date: missing for date in dates})

    covar_dict = rolling_covariances(dates)
    implied_path = pd.DataFrame({date: implied_budgets(target, c)
                                 for date, c in covar_dict.items()}).T
    fitted_path = op.solve_for_risk_budgets_from_given_weights(
        prices=prices, given_weights=target, covar_dict=covar_dict)
    path = op.rolling_risk_budgeting(prices=prices,
                                     constraints=op.Constraints(is_long_only=True),
                                     risk_budget=fitted_path, covar_dict=covar_dict)
    average = op.average_rolling_weights(path)
    print(pd.concat([target.rename('target weight'), fitted_path.rename('fitted budget'),
                     average.rename('average weight')], axis=1).round(4))

    # The implied budgets of Bonds are negative on the first two dates and positive after.
    negative = implied_path['Bonds'] < 0.0
    assert negative.tolist() == [True, True] + [False] * (REBALANCES - 2)
    assert np.allclose(implied_path.sum(axis=1), 1.0, atol=1e-14)
    # The figure: Bonds from -8% to 25%, Equity from 85% to 65%, Gold from 24% to 10%, and
    # fitted budgets of about 70.6%, 14.5% and 14.9%.
    ends = implied_path.iloc[[0, -1]].round(2)
    assert ends.to_numpy().tolist() == [[0.85, -0.08, 0.24], [0.65, 0.25, 0.10]]
    assert np.abs(fitted_path - [0.706, 0.145, 0.149]).max() < 0.002
    # At every date where all contributions are positive, that date's implied budgets
    # reproduce the target; on the two hedging dates the wrapper drops Bonds.
    for date, c in covar_dict.items():
        one_date = op.wrapper_risk_budgeting(pd_covar=c,
                                             constraints=op.Constraints(is_long_only=True),
                                             risk_budget=implied_path.loc[date])
        if negative[date]:
            assert one_date['Bonds'] == 0.0
        else:
            assert np.abs(one_date - target).max() < 1e-8
    # A date-by-asset panel of the implied budgets does the same in one rolling call.
    panel = op.rolling_risk_budgeting(prices=prices,
                                      constraints=op.Constraints(is_long_only=True),
                                      risk_budget=implied_path, covar_dict=covar_dict)
    assert np.abs(panel.loc[~negative] - target).to_numpy().max() < 1e-8
    # One static budget vector reproduces the target on average, within the tolerance; the
    # average is the simple mean, and Bonds' forward weight moves from 59% to 32%.
    assert np.abs(average - target).max() < TOLERANCE
    pd.testing.assert_series_equal(average, path.mean(axis=0))
    assert round(path['Bonds'].iloc[0], 2) == 0.59 and round(path['Bonds'].iloc[-1], 2) == 0.32
    # Proposition 1, only if: with positive budgets every forward weight has a positive
    # marginal risk, also on the dates where Bonds hedge the target.
    assert (fitted_path > 0.0).all()
    for date, c in covar_dict.items():
        assert (c.to_numpy() @ path.loc[date].to_numpy() > 0.0).all()
    # Insight: the seed, the averaged implied budgets, is not the answer. Its forward path
    # misses the target by about two points; the fitted Bonds budget is about 14.5%, two
    # points above the 12.5% average of its implied budgets.
    seed = implied_path.mean(axis=0)
    diagnostics = rb_module._target_risk_contributions(target, covar_dict, None)
    np.testing.assert_allclose(diagnostics['average_rc'], seed, atol=1e-14)
    seed_path = op.rolling_risk_budgeting(prices=prices,
                                          constraints=op.Constraints(is_long_only=True),
                                          risk_budget=seed, covar_dict=covar_dict)
    seed_gap = np.abs(seed_path.mean(axis=0) - target).max()
    assert 0.015 < seed_gap < 0.025 and round(seed_gap, 2) == 0.02
    assert abs(fitted_path['Bonds'] - 0.145) < 0.002 and round(seed['Bonds'], 3) == 0.125
    # Bonds hedge on 2 of 12 dates, below half, so no warning; on half the dates it warns.
    alternating = {date: covariance(HEDGE_STOCK_BOND if k % 2 == 0 else STOCK_BOND_PATH[1])
                   for k, date in enumerate(dates)}
    _, messages = caught_warnings(solve, prices=prices, given_weights=target,
                                  covar_dict=alternating)
    assert len(messages) == 1 and 'Bonds has a negative marginal risk contribution' in messages[0]
    _, messages = caught_warnings(solve, prices=prices, given_weights=target,
                                  covar_dict=covar_dict)
    assert messages == []

    from optimalportfolios.optimization.risk_allocation.risk_budgeting import INVERSE_EWMA_SPAN
    recent = op.average_rolling_weights(path, ewma_span=INVERSE_EWMA_SPAN)
    print(INVERSE_EWMA_SPAN, recent.round(4).tolist())

    # The EWMA average equals the explicit recursion seeded at the first row, and the
    # closed-form row weights (1 - lambda) lambda^(T-1-t), with lambda^(T-1) on the first row.
    assert INVERSE_EWMA_SPAN == 12 and not hasattr(op, 'INVERSE_EWMA_SPAN')
    np.testing.assert_allclose(recent, explicit_ewma(path, INVERSE_EWMA_SPAN), atol=1e-14)
    decay = 1.0 - 2.0 / (INVERSE_EWMA_SPAN + 1.0)
    row_weights = (1.0 - decay) * decay ** np.arange(REBALANCES)[::-1]
    row_weights[0] = decay ** (REBALANCES - 1)
    assert abs(row_weights.sum() - 1.0) < 1e-14
    np.testing.assert_allclose(recent, row_weights @ path.to_numpy(), atol=1e-14)
    # On this 12-date path the first rebalance carries 15.9%, more than the latest (15.4%).
    assert round(row_weights[0], 3) == 0.159 and round(row_weights[-1], 3) == 0.154
    assert row_weights[0] > row_weights[-1] and np.isclose(row_weights[-1], 2.0 / 13.0)
    # A NaN row holds the state, as if the row were dropped; an all-NaN path gives NaN; a
    # span that is not finite and positive raises.
    gapped = path.copy()
    gapped.iloc[5] = np.nan
    np.testing.assert_allclose(op.average_rolling_weights(gapped, ewma_span=4.0),
                               explicit_ewma(path.drop(index=path.index[5]), 4.0), atol=1e-14)
    assert op.average_rolling_weights(path * np.nan, ewma_span=4.0).isna().all()
    for span in (0.0, -1.0, np.inf):
        assert_raises(ValueError, op.average_rolling_weights, weights=path, ewma_span=span)

    recent_fit = op.solve_for_risk_budgets_from_given_weights(
        prices=prices, given_weights=target, covar_dict=covar_dict,
        ewma_span=INVERSE_EWMA_SPAN)
    recent_path = op.rolling_risk_budgeting(prices=prices,
                                            constraints=op.Constraints(is_long_only=True),
                                            risk_budget=recent_fit, covar_dict=covar_dict)
    print(op.average_rolling_weights(recent_path, ewma_span=INVERSE_EWMA_SPAN).round(4).tolist())

    # The span-12 fit matches the EWMA average, not the simple mean of its own path.
    recent_average = op.average_rolling_weights(recent_path, ewma_span=INVERSE_EWMA_SPAN)
    assert np.abs(recent_average - target).max() < TOLERANCE
    assert np.abs(recent_path.mean(axis=0) - target).max() > 5 * TOLERANCE
    assert np.abs(recent_fit - fitted_path).max() > 0.01

    boundary_prices, boundary_target, boundary_covars = boundary_inputs()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        probed = op.solve_for_risk_budgets_from_given_weights(
            prices=boundary_prices, given_weights=boundary_target, covar_dict=boundary_covars)
    print(probed.round(4).tolist())
    print([str(item.message)[:72] for item in caught])

    # Bond A hedges on average and is held by the rule; Bond B carries risk on average, yet at
    # the smallest positive budget its average weight stays far above target, so the probe
    # holds it too after a complete refit meets both tolerances.
    assert (100 * boundary_target).round(1).tolist() == [65.0, 19.9, 12.5, 2.6]
    shares_by_date = pd.DataFrame({date: implied_budgets(boundary_target, c)
                                   for date, c in boundary_covars.items()}).T
    assert shares_by_date['Bond A'].mean() < 0.0 < shares_by_date['Bond B'].mean()
    assert (shares_by_date['Bond B'] < 0.0).sum() == 1
    assert probed['Bond A'] == 0.0 and probed['Bond B'] == 0.0
    assert abs(probed.sum() - 1.0) < 1e-12 and (probed[['Equity', 'Other']] > 0.0).all()
    assert len(caught) == 2
    assert 'pinned assets with non-positive average marginal' in str(caught[0].message)
    assert 'Bond A' in str(caught[0].message) and 'Bond B' not in str(caught[0].message)
    assert 'at the budget boundary: Bond B' in str(caught[1].message)
    floor = pd.Series([0.91 - 1e-4, 0.09, 0.0, 1e-4], index=BOUNDARY_ASSETS)
    at_floor = op.rolling_risk_budgeting(prices=boundary_prices,
                                         constraints=held_constraints(boundary_target,
                                                                      ['Bond A']),
                                         risk_budget=floor, covar_dict=boundary_covars)
    assert at_floor.mean(axis=0)['Bond B'] > boundary_target['Bond B'] + 0.10
    both_held = op.rolling_risk_budgeting(prices=boundary_prices,
                                          constraints=held_constraints(boundary_target,
                                                                       ['Bond A', 'Bond B']),
                                          risk_budget=probed, covar_dict=boundary_covars)
    assert np.abs(both_held.mean(axis=0) - boundary_target).max() < TOLERANCE

    # Special cases: a one-asset panel gets the whole budget; a one-asset target, too.
    single = solve(prices=prices[['Equity']], given_weights=pd.Series({'Equity': 1.0}),
                   covar_dict={date: covar.loc[['Equity'], ['Equity']] for date in dates})
    assert single.tolist() == [1.0]
    assert_raises(ValueError, solve, prices=prices[['Equity']],
                  given_weights=pd.Series({'Equity': 1.0}), fixed_weight_assets=['Equity'],
                  covar_dict={date: covar.loc[['Equity'], ['Equity']] for date in dates})
    assert_raises(ValueError, solve, ewma_span=0.0, **inputs)
    # A gap of 1e-3 is about 4% of a 2.6% target weight.
    assert round(TOLERANCE / 0.026, 2) == 0.04
    only_bonds = solve(prices=prices, given_weights=pd.Series([0.0, 1.0, 0.0], index=ASSETS),
                       covar_dict={date: covar for date in dates})
    assert only_bonds.tolist() == [0.0, 1.0, 0.0]
    # Diagonal covariance: the undamped update b_i (w_i / w_i(b))^2 lands on the target in one
    # step, because there w_i is proportional to sqrt(b_i) / sigma_i.
    diagonal = pd.DataFrame(np.diag(np.square(VOLS)), index=ASSETS, columns=ASSETS)
    equal = pd.Series(1.0 / 3.0, index=ASSETS)
    first = op.wrapper_risk_budgeting(pd_covar=diagonal,
                                      constraints=op.Constraints(is_long_only=True),
                                      risk_budget=equal)
    updated = equal * np.square(target / first)
    second = op.wrapper_risk_budgeting(pd_covar=diagonal,
                                       constraints=op.Constraints(is_long_only=True),
                                       risk_budget=updated / updated.sum())
    assert np.abs(first - target).max() > 0.05 and np.abs(second - target).max() < 1e-8
    print('implied_risk_budgets: all page statements verified.')


def exhibit(path) -> dict:
    """Draw the page's figure: implied budgets, forward weights and their average through time.

    Args:
        path: PNG file to write.

    Returns:
        The plotted table and the checks the figure illustrates.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    dates = rebalance_dates()
    covar_dict = rolling_covariances(dates)
    target = pd.Series(TARGET_WEIGHTS, index=ASSETS)
    prices = pd.DataFrame(100.0, index=dates, columns=ASSETS)
    implied = pd.DataFrame({date: implied_budgets(target, c) for date, c in covar_dict.items()}).T
    round_trip = pd.DataFrame({date: op.wrapper_risk_budgeting(
        pd_covar=c, constraints=op.Constraints(is_long_only=True), risk_budget=implied.loc[date])
        for date, c in covar_dict.items()}).T
    fitted = op.solve_for_risk_budgets_from_given_weights(prices=prices, given_weights=target,
                                                          covar_dict=covar_dict)
    forward = op.rolling_risk_budgeting(prices=prices,
                                        constraints=op.Constraints(is_long_only=True),
                                        risk_budget=fitted, covar_dict=covar_dict)
    running = pd.DataFrame({date: op.average_rolling_weights(forward.loc[:date])
                            for date in dates}).T
    admissible = (implied > 0.0).all(axis=1)
    fitted_rows = pd.DataFrame([fitted.to_numpy()] * len(dates), index=dates, columns=ASSETS)
    parts = {'implied budget': implied, 'round-trip weight': round_trip,
             'fitted budget': fitted_rows, 'forward weight': forward,
             'running average weight': running}
    table = pd.concat([part.add_prefix(f'{name}: ') for name, part in parts.items()], axis=1)
    table['all contributions positive'] = admissible

    ink, muted, grid, surface = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
    colours = dict(zip(ASSETS, ('#2a78d6', '#eb6834', '#1baf7a')))
    plt.rcParams.update({'font.size': 11, 'axes.edgecolor': grid, 'axes.labelcolor': muted,
                         'xtick.color': muted, 'ytick.color': muted})
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.0, 4.4), facecolor=surface)
    # Shade the hedging dates, from half a quarter before the first to midway to the next.
    hedging = dates[~admissible.to_numpy()]
    half_quarter = pd.Timedelta(days=45)
    left.axvspan(hedging[0] - half_quarter, hedging[-1] + half_quarter, color=grid, linewidth=0)
    left.text(dates[0] - half_quarter + pd.Timedelta(days=10), 0.47,
              'Bonds hedge:\nno admissible\nbudget', color=ink, fontsize=9, va='center')
    for asset in ASSETS:
        left.plot(dates, implied[asset], color=colours[asset], marker='o', markersize=3.5,
                  linewidth=1.8)
        left.hlines(fitted[asset], dates[0], dates[-1], color=colours[asset], linestyle='--',
                    linewidth=1.2)
        right.plot(dates, forward[asset], color=colours[asset], linewidth=1.0, alpha=0.55)
        right.plot(dates, running[asset], color=colours[asset], linewidth=2.4)
        right.hlines(target[asset], dates[0], dates[-1], color=colours[asset], linestyle=':',
                     linewidth=1.4)
    left.axhline(0.0, color=muted, linewidth=0.8)
    end = dates[-1] + pd.Timedelta(days=25)
    for asset, y in zip(ASSETS, (implied['Equity'].iloc[-1], implied['Bonds'].iloc[-1] + 0.03,
                                 implied['Gold'].iloc[-1] - 0.04)):
        left.text(end, y, asset, color=ink, va='center', fontsize=10)
    for asset in ASSETS:
        right.text(end, target[asset], asset, color=ink, va='center', fontsize=10)
    left.legend(handles=[plt.Line2D([], [], color=muted, marker='o', markersize=3.5),
                         plt.Line2D([], [], color=muted, linestyle='--')],
                labels=['implied at each date', 'fitted static budget'], frameon=False,
                loc='center right', bbox_to_anchor=(1.0, 0.43), fontsize=9, labelcolor=ink)
    right.legend(handles=[plt.Line2D([], [], color=muted, linewidth=1.0, alpha=0.55),
                          plt.Line2D([], [], color=muted, linewidth=2.4),
                          plt.Line2D([], [], color=muted, linestyle=':')],
                 labels=['forward weight', 'running average', 'target weight'], frameon=False,
                 loc='upper right', fontsize=9, labelcolor=ink)
    left.set_title('Implied risk budgets of the target', loc='left', color=ink)
    right.set_title('Weights under the fitted budgets', loc='left', color=ink)
    left.set_ylim(-0.12, 1.0)
    right.set_ylim(0.0, 0.78)
    for axis in (left, right):
        axis.set_facecolor(surface)
        axis.set_xlim(dates[0] - half_quarter, dates[-1] + pd.Timedelta(days=170))
        axis.xaxis.set_major_locator(mdates.YearLocator())
        axis.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        axis.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        axis.grid(axis='y', color=grid, linewidth=0.8)
        axis.set_axisbelow(True)
        for side in ('top', 'right'):
            axis.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=surface)
    plt.close(fig)

    reproduced = (round_trip.loc[admissible] - target).abs().max(axis=1) < 1e-8
    checks = {
        'implied_budgets_sum_to_one': bool(np.allclose(implied.sum(axis=1), 1.0, atol=1e-14)),
        'round_trip_where_all_contributions_positive': bool(reproduced.all()),
        'hedging_dates_have_no_admissible_budget': bool(
            (~admissible).sum() == 2 and (round_trip.loc[~admissible, 'Bonds'] == 0.0).all()),
        'average_weight_matches_target': bool(
            np.abs(running.iloc[-1] - target).max() < TOLERANCE),
        'average_is_the_simple_mean': bool(np.allclose(running.iloc[-1], forward.mean(axis=0),
                                                       atol=1e-14)),
    }
    return {'table': table, 'checks': checks}


if __name__ == '__main__':
    main()
