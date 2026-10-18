"""Executed-state rolling contracts against QIS's independent scalar cash ledger."""
import numpy as np
import pandas as pd
import pytest

from optimalportfolios.optimization.executed_state import backtest_with_qis_state
from optimalportfolios.optimization.general.quadratic import wrapper_quadratic_optimisation
from optimalportfolios.optimization.constraints import Constraints
from optimalportfolios.utils.weights_drift import apply_drift_to_weights_0

_oracle = pytest.importorskip(
    'qis.portfolio.tests.backtester_missing_prices_test',
    reason='Executed-state workflow requires the QIS 5.37+ accounting contract',
)
if not hasattr(_oracle, '_ledger_reference'):
    pytest.skip('QIS 5.37+ ledger oracle is unavailable', allow_module_level=True)
_ledger_reference = _oracle._ledger_reference


def panel():
    """Return a fixed observed two-security universe with no estimation randomness."""
    return pd.DataFrame({'A': [100., 110., 121., 121., 133.1, 140., 145., 150., 152.],
                         'B': [100., 90., 90., 99., 99., 101., 102., 103., 104.]},
                        index=pd.bdate_range('2024-01-01', periods=9))


def assert_ledger(result, prices, rate):
    """Compare units, cash, NAV, costs and turnover using independently evolved wealth."""
    flags = np.zeros(len(prices), dtype=bool)
    accepted = result.events[result.events.status == 'scheduled']
    flags[prices.index.get_indexer(pd.DatetimeIndex(accepted.execution))] = True
    targets = result.targets.to_numpy()
    zeros = np.zeros(len(prices))
    nav, units, costs, cash, marks = _ledger_reference(
        prices.to_numpy(), targets, flags, zeros, zeros,
        np.zeros(prices.shape), np.full(prices.shape, rate), None)
    p = result.portfolio
    np.testing.assert_allclose(p.nav, nav, atol=1e-11, rtol=0)
    np.testing.assert_allclose(p.units, units, atol=1e-13, rtol=0)
    np.testing.assert_allclose(p.realized_costs, costs, atol=1e-13, rtol=0)
    np.testing.assert_allclose(nav - (units * marks).sum(axis=1), cash, atol=1e-11, rtol=0)
    for state in result.states:
        pos = prices.index.get_loc(state.date)
        # At zero lag a current decision trades after the snapshot. Replay only earlier orders.
        prior = accepted[accepted.index < state.date]
        prior_flags = flags.copy()
        prior_flags[:] = False
        prior_flags[prices.index.get_indexer(pd.DatetimeIndex(prior.execution))] = True
        pn, pu, _, pc, _ = _ledger_reference(
            prices.to_numpy()[:pos+1], result.targets.loc[prior.index].to_numpy(),
            prior_flags[:pos+1], zeros[:pos+1], zeros[:pos+1],
            np.zeros((pos+1, prices.shape[1])), np.full((pos+1, prices.shape[1]), rate), None)
        np.testing.assert_allclose(state.units, pu[-1], atol=1e-13, rtol=0)
        np.testing.assert_allclose([state.nav, state.cash], [pn[-1], pc[-1]], atol=1e-11)
    expected = np.full(prices.shape, np.nan)
    expected[1:] = abs(np.diff(units, axis=0)) * prices.to_numpy()[1:] / nav[1:, None]
    actual = p.get_turnover(roll_period=None, add_total=False)
    np.testing.assert_allclose(actual, expected, atol=1e-13, rtol=0, equal_nan=True)


@pytest.mark.parametrize('lag,rate,frozen,failed', [
    (0, 0., False, False), (1, 0., False, False), (1, .01, False, False),
    (1, .01, True, False), (1, .01, True, True), (0, .01, True, True)])
def test_reconciles_actual_cash_units_costs_turnover(lag, rate, frozen, failed):
    """Exercise timing, cost funding, freeze/eligibility changes and a real infeasible solve."""
    prices = panel()
    decisions = prices.index[[0, 3, 6]]
    if frozen:
        prices.loc[prices.index[3:5], 'B'] = np.nan
    eligibility = pd.DataFrame([[1, 1], [1, 0]], index=decisions[:2], columns=prices.columns)
    outcomes = []

    def decide(history, state):
        """Exclude quoted B or preserve its frozen state; fail the final solve when requested."""
        assert history.index[-1] == state.date
        assert set(history.columns) == set(state.units.index)
        if failed and state.date == decisions[-1]:
            weights, outcome = wrapper_quadratic_optimisation(
                pd_covar=pd.DataFrame(np.eye(2), index=prices.columns, columns=prices.columns),
                constraints=Constraints(max_weights=pd.Series(.1, index=prices.columns)),
                weights_0=state.weights)
            assert not outcome.accepted
            outcomes.append(outcome)
            return weights, outcome
        if state.date == decisions[0]:
            return pd.Series({'B': .5, 'A': .5})  # Deliberately permuted labels.
        return pd.Series({'A': .25, 'B': state.weights.B if not state.quoted.B else 0.})

    result = backtest_with_qis_state(prices, decisions, decide, implementation_lag=lag,
                                   transaction_costs=rate, eligibility=eligibility)
    assert_ledger(result, prices, rate)
    if rate:
        assert result.states[1].cash < 0  # Entry funding remains an actual cash debit.
        assert not np.isclose(result.states[1].weights.sum(), 1.)
    if frozen:
        assert result.states[1].units.B > 0 and not result.states[1].eligible.B
        assert not result.states[1].quoted.B
    if failed:
        assert outcomes and result.events.iloc[-1].status == 'held'
        assert pd.isna(result.events.iloc[-1].execution)
        assert not result.portfolio.is_rebalancing.iloc[6+lag]
    else:
        assert result.portfolio.units.iloc[6+lag].B == 0


def test_execution_date_state_differs_from_legacy_decision_drift():
    """A delayed entry must not earn the return before its actual execution."""
    dates = pd.bdate_range('2024-01-02', periods=5)
    prices = pd.DataFrame({'A': [100., 110., 121., 133.1, 146.41], 'B': 100.}, index=dates)

    def decide(history, state):
        """Take an equal target once, then preserve actual holdings on failed/no-order policy."""
        return pd.Series(.5, index=prices.columns) if state.date == dates[0] else None

    result = backtest_with_qis_state(prices, dates[[0, 2]], decide)
    actual = result.states[1].weights
    legacy = apply_drift_to_weights_0(pd.Series(.5, index=prices.columns), prices,
                                      dates[0], dates[2])
    assert actual.A == pytest.approx(55/105)
    assert legacy.A == pytest.approx(60.5/110.5)
    assert abs(actual.A - legacy.A) > .02
    np.testing.assert_allclose(result.portfolio.nav, [100., 100., 105., 110.5, 116.55])


def test_observed_prefix_isolation_and_future_shock():
    """A policy can mutate its copies, but cannot change the audit or observe later prices."""
    prices = panel()
    decisions = prices.index[[0, 3, 6]]

    def decide(history, state):
        """Size using observed history and mutate detached inputs after reading them."""
        a = .4 if history.iloc[-1].A < 200 else .6
        state.units[:] = 999
        state.weights[:] = 999
        history[:] = 999
        return pd.Series({'A': a, 'B': 1-a})

    result = backtest_with_qis_state(prices, decisions, decide)
    shocked = prices.copy()
    shocked.loc[shocked.index > decisions[1], 'A'] *= 3
    other = backtest_with_qis_state(shocked, decisions, decide)
    pd.testing.assert_frame_equal(result.targets.iloc[:2], other.targets.iloc[:2])
    for x, y in zip(result.states[:2], other.states[:2]):
        pd.testing.assert_series_equal(x.units, y.units)
        assert (x.units != 999).all() and (x.weights != 999).all()
    assert result.targets.iloc[-1].A != other.targets.iloc[-1].A
    pd.testing.assert_frame_equal(prices, panel())


def test_all_skipped_stays_in_cash():
    """No accepted decisions require no invented holdings or failing empty QIS schedule."""
    prices = panel()
    result = backtest_with_qis_state(prices, prices.index[[0, 3]], lambda h, s: None)
    assert result.targets.empty
    np.testing.assert_allclose(result.portfolio.nav, 100.)
    np.testing.assert_allclose(result.portfolio.units, 0.)
    assert all(s.cash == 100 for s in result.states)


@pytest.mark.parametrize('bad', ['labels', 'duplicate', 'negative', 'leveraged', 'nan',
                                 'positional', 'tuple', 'ineligible', 'unquoted'])
def test_rejects_ambiguous_or_impossible_targets(bad):
    """Policy errors fail explicitly rather than renormalising or changing eligibility."""
    prices = panel()
    if bad == 'unquoted':
        prices.iloc[0, 1] = np.nan
    target = pd.Series({'A': .5, 'B': .5})
    if bad == 'labels':
        target.index = ['A', 'C']
    elif bad == 'duplicate':
        target.index = ['A', 'A']
    elif bad == 'negative':
        target.iloc[0] = -.5
    elif bad == 'leveraged':
        target[:] = .8
    elif bad == 'nan':
        target.iloc[0] = np.nan
    elif bad == 'positional':
        target = np.array([.5, .5])
    elif bad == 'tuple':
        target = (target, 'optimal')
    eligibility = pd.DataFrame(True, index=prices.index, columns=prices.columns)
    if bad == 'ineligible':
        eligibility.B = False
    with pytest.raises((TypeError, ValueError)):
        backtest_with_qis_state(prices, prices.index[[0]], lambda h, s: target,
                               eligibility=eligibility)


@pytest.mark.parametrize('bad', ['lag_bool', 'lag_negative', 'lag_fraction', 'overlap',
                                 'offgrid', 'unordered', 'last_execution', 'price_zero',
                                 'price_inf', 'duplicate_assets', 'cost', 'nav',
                                 'eligibility_labels', 'eligibility_unknown',
                                 'eligibility_nonbinary'])
def test_validates_observation_and_policy_contract(bad):
    """Reject data that would silently change the economic ordering or price interpretation."""
    prices = panel()
    dates = prices.index[[0, 3]]
    kwargs = {}
    if bad.startswith('lag_'):
        kwargs['implementation_lag'] = {'lag_bool': True, 'lag_negative': -1,
                                        'lag_fraction': .5}[bad]
    elif bad == 'overlap':
        dates = prices.index[[0, 1]]
    elif bad == 'offgrid':
        dates = dates + pd.Timedelta(hours=1)
    elif bad == 'unordered':
        dates = dates[::-1]
    elif bad == 'last_execution':
        dates = prices.index[[-2]]
    elif bad.startswith('price_'):
        prices.iloc[0, 0] = 0 if bad == 'price_zero' else np.inf
    elif bad == 'duplicate_assets':
        prices.columns = ['A', 'A']
    elif bad == 'cost':
        kwargs['transaction_costs'] = -.1
    elif bad == 'nav':
        kwargs['initial_nav'] = 0
    else:
        eligibility = pd.DataFrame(1., index=prices.index, columns=prices.columns)
        if bad == 'eligibility_labels':
            eligibility.columns = ['A', 'C']
        elif bad == 'eligibility_unknown':
            eligibility = eligibility.iloc[1:]
        else:
            eligibility.iloc[0, 0] = .5
        kwargs['eligibility'] = eligibility
    with pytest.raises(ValueError):
        backtest_with_qis_state(prices, dates, lambda h, s: pd.Series(.5, index=h.columns),
                               **kwargs)


def test_noncompliant_outcome_holds_even_if_accepted():
    """The interface respects the canonical structured compliance audit as well as status."""
    from optimalportfolios.optimization.solver_diagnostics import OptimizationOutcome
    from optimalportfolios.optimization.constraints.analytics import ConstraintResidual

    prices = panel()
    residual = ConstraintResidual('budget', 'test', 2., 1., 1., 1., 1e-6, True, False)
    outcome = OptimizationOutcome(np.array([.5, .5]), True, 'test', 'optimal', '',
                                  constraint_residuals=(residual,))
    result = backtest_with_qis_state(prices, prices.index[[0]],
                                   lambda h, s: (pd.Series(.5, index=h.columns), outcome))
    assert result.targets.empty
    assert result.events.iloc[0].status == 'held'


@pytest.mark.parametrize('resume', [False, True])
def test_quote_availability_changes_during_lag(resume):
    """Future quote changes remain QIS execution policy and cannot leak into decisions."""
    prices = panel()
    if resume:
        prices.loc[prices.index[3], 'B'] = np.nan
    else:
        prices.loc[prices.index[4], 'B'] = np.nan

    def decide(history, state):
        """Set a labelled target using only quote availability at the decision close."""
        b = state.weights.B if not state.quoted.B else .5
        return pd.Series({'A': .25, 'B': b})

    result = backtest_with_qis_state(prices, prices.index[[0, 3]], decide,
                                   transaction_costs=.001)
    assert result.states[1].quoted.B == (not resume)
    assert_ledger(result, prices, .001)
    if not resume:
        assert result.portfolio.units.iloc[4].B == result.portfolio.units.iloc[3].B


def test_valid_fallback_weights_are_not_executed():
    """A failed solve must skip even a finite feasible fallback target."""
    from optimalportfolios.optimization.solver_diagnostics import OptimizationOutcome

    prices = panel()
    outcome = OptimizationOutcome(np.array([.5, .5]), False, 'test', 'infeasible', '',
                                  reason='controlled failed solve', fallback_source='weights_0')
    result = backtest_with_qis_state(prices, prices.index[[0]],
                                   lambda h, s: (pd.Series(.5, index=h.columns), outcome))
    assert result.targets.empty
    np.testing.assert_allclose(result.portfolio.units, 0.)
    assert result.events.iloc[0].reason == 'controlled failed solve'


def test_cash_scale_and_exhausted_nav_validation():
    """Initial currency scaling is linear; a bankrupt actual state cannot feed another solve."""
    prices = panel()
    def policy(history, state):
        """Return the same fully invested target independently of initial currency scale."""
        return pd.Series(.5, index=history.columns)

    a = backtest_with_qis_state(prices, prices.index[[0, 3]], policy)
    b = backtest_with_qis_state(prices, prices.index[[0, 3]], policy, initial_nav=200)
    np.testing.assert_allclose(b.portfolio.nav, 2*a.portfolio.nav)
    np.testing.assert_allclose(b.portfolio.units, 2*a.portfolio.units)
    with pytest.raises(ValueError, match='positive finite NAV'):
        backtest_with_qis_state(prices, prices.index[[0, 3]], policy, transaction_costs=2.)


def test_price_frame_type_is_explicit():
    """A vector is not an implicitly aligned price panel."""
    with pytest.raises(ValueError, match='DataFrame'):
        backtest_with_qis_state(None, panel().index[[0]], lambda h, s: None)


def test_late_entry_then_quoted_eligibility_exit():
    """A fixed asset superset supports pre-listing cash and a later explicit liquidation."""
    prices = panel()
    prices.loc[prices.index[:3], 'B'] = np.nan
    dates = prices.index[[0, 3, 6]]
    eligibility = pd.DataFrame([[1, 0], [1, 1], [1, 0]],
                               index=dates, columns=prices.columns)

    def decide(history, state):
        """Enter only after the asset is observed and eligible; preserve cash otherwise."""
        return pd.Series({'A': .5, 'B': .5 if state.eligible.B else 0.})

    result = backtest_with_qis_state(prices, dates, decide, eligibility=eligibility,
                                   transaction_costs=.001)
    assert_ledger(result, prices, .001)
    assert result.states[0].weights.B == 0 and not result.states[0].quoted.B
    assert result.portfolio.units.iloc[1].B == 0
    assert result.portfolio.units.iloc[4].B > 0
    assert result.portfolio.units.iloc[7].B == 0


@pytest.mark.parametrize('installed', ['5.33.1', '5.36.9', 'unknown'])
def test_old_or_unknown_engine_gets_actionable_opt_in_error(monkeypatch, installed):
    """The optional workflow must not silently use pre-freeze accounting at the core floor."""
    import importlib

    module = importlib.import_module('optimalportfolios.optimization.executed_state')
    monkeypatch.setattr(module, '_distribution_version', lambda name: installed)
    with pytest.raises(RuntimeError, match=r'qis>=5\.37'):
        backtest_with_qis_state(panel(), panel().index[[0]], lambda history, state: None)
