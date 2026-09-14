"""Execute the turnover article and verify target budgets, traded units and cash costs."""

from dataclasses import replace
from fractions import Fraction
import re
import runpy

import numpy as np
import pandas as pd
import pytest

LEGACY_ANCHORS = {
    'turnover-and-transaction-costs', 'target-turnover-in-optimisation',
    'realised-turnover-and-costs-in-the-backtest', 'interpretation-and-failure-modes', 'see-also',
}


@pytest.fixture(scope='module')
def article(root):
    """Load the canonical article after checkout-only checks have resolved the root."""
    return (root / 'docs/turnover_and_transaction_costs.md').read_text(encoding='utf-8')


@pytest.fixture(scope='module')
def examples(article):
    """Execute exactly four sequential blocks, including the two preserved original examples."""
    matches = re.findall(r'^```python([^\n]*)\n(.*?)^```', article, re.M | re.S)
    assert len(matches) == 4 and all(not options.strip() for options, _ in matches)
    state = {'__name__': '__turnover_article__'}
    for index, (_, code) in enumerate(matches, start=1):
        exec(compile(code, f'turnover_and_transaction_costs.md (block {index})', 'exec'), state)
    return state


@pytest.fixture(scope='module')
def ledger():
    """Build an exact two-trade currency reference without calling a backtest or cost helper."""
    first_units = [Fraction(60, 102), Fraction(40, 99)]
    first_cost = Fraction(1, 10)
    pre_nav = sum(units * 101 for units in first_units) - first_cost
    new_units = pre_nav / (2 * 101)
    second_trade = sum(abs(new_units - units) * 101 for units in first_units)
    second_cost = second_trade / 1000
    return {
        'first_units': np.array(first_units, dtype=float),
        'new_units': float(new_units), 'second_trade': float(second_trade),
        'costs': np.array([0, first_cost, second_cost], dtype=float),
        'nav': np.array([100, 100 - first_cost, pre_nav - second_cost], dtype=float),
    }


def test_article_structure_links_and_legacy_fragments(article, root):
    """Preserve the methodology contract, ordinary links and all five original sections."""
    checker = runpy.run_path(str(root / 'tools/check_docs.py'))
    assert not checker['check_document'](article, methodology=True)
    assert not checker['check_local_links'](
        article, root / 'docs/turnover_and_transaction_costs.md', root)
    visible, _, _ = checker['prose_lines'](article)
    anchors = set(re.findall(r'<a id="([^"]+)"></a>', article))
    for _, line in visible:
        match = re.match(r'^#{1,6} (.+)', line)
        if match:
            anchors.add(re.sub(r'[^\w -]', '', match[1]).lower().replace(' ', '-'))
    assert LEGACY_ANCHORS <= anchors
    assert not (root / 'docs/turnover_and_transaction_costs.rst').exists()


def test_first_solve_respects_stored_baseline_and_has_optimality_reference(examples):
    """The convex one-dimensional objective decreases down to the active 55% lower bound."""
    actual, outcome = examples['optimal_weights'], examples['outcome']
    assert outcome.accepted and outcome.compliant and outcome.fallback_source is None
    assert outcome.covar_factorization.n_eigenvalues_floored == 0
    # w_A + w_B = 1; 2*abs(w_A - .60) <= .10 implies w_A >= .55.
    # Derivative of .04*x**2 + .01*(1-x)**2 is positive throughout [.55, .65].
    assert 0.10 * 0.55 - 0.02 > 0
    np.testing.assert_allclose(actual, [0.55, 0.45], atol=2e-7, rtol=0)
    assert np.abs(actual - examples['current']).sum() == pytest.approx(0.10, abs=4e-7)


def test_displayed_target_table_matches_solve(article, examples):
    """Catch a stale allocation or a one-sided change in the quoted full-L1 example."""
    rows = re.findall(r'^\| ([AB]) \| ([0-9.]+) \| ([0-9.]+) \| ([0-9.]+) \|$', article, re.M)
    assert [row[0] for row in rows] == ['A', 'B']
    values = np.array([[float(value) for value in row[1:]] for row in rows])
    expected = np.column_stack([
        examples['current'], examples['optimal_weights'],
        np.abs(examples['optimal_weights'] - examples['current'])])
    np.testing.assert_allclose(values, expected, atol=2e-7, rtol=0)
    assert 'full L1 change is **0.10**' in article


def test_weighted_budget_uses_all_multipliers(examples):
    """With coefficients [2,1], the same budget permits a transfer of only .10/3."""
    spec = replace(examples['constraints'], turnover_costs=pd.Series({'A': 2.0, 'B': 1.0}))
    weights, outcome = examples['opt'].wrapper_quadratic_optimisation(
        pd_covar=examples['covar'], constraints=spec)
    assert outcome.accepted and outcome.compliant
    np.testing.assert_allclose(weights, [0.60 - 0.10 / 3, 0.40 + 0.10 / 3], atol=3e-7)
    assert (spec.turnover_costs * (weights - examples['current']).abs()).sum() <= 0.100001


def test_absent_baseline_is_not_a_zero_starting_portfolio(examples):
    """Absent holdings skip the row; explicit zeros make a .10 fully-invested budget infeasible."""
    opt = examples['opt']
    spec = replace(examples['constraints'], weights_0=None)
    weights, outcome = opt.wrapper_quadratic_optimisation(
        pd_covar=examples['covar'], constraints=spec)
    assert outcome.accepted and outcome.compliant
    np.testing.assert_allclose(weights, [0.20, 0.80], atol=2e-6)
    _, rejected = opt.wrapper_quadratic_optimisation(
        pd_covar=examples['covar'], constraints=spec, weights_0=examples['current'] * 0)
    assert not rejected.accepted


def test_wrapper_argument_overrides_stored_baseline(examples):
    """An explicit 50/50 baseline permits a 45/55 optimum, not the stored 55/45 result."""
    current = pd.Series({'A': 0.50, 'B': 0.50})
    weights, outcome = examples['opt'].wrapper_quadratic_optimisation(
        pd_covar=examples['covar'], constraints=examples['constraints'], weights_0=current)
    assert outcome.accepted and outcome.compliant
    np.testing.assert_allclose(weights, [0.45, 0.55], atol=3e-7)


def test_two_executed_trades_match_exact_currency_ledger(examples, ledger):
    """Reconstruct entry and the second trade from rational prices, units and cash."""
    portfolio = examples['portfolio']
    np.testing.assert_allclose(portfolio.units.iloc[0], 0, atol=0)
    np.testing.assert_allclose(portfolio.units.iloc[1], ledger['first_units'], atol=1e-12)
    np.testing.assert_allclose(portfolio.units.iloc[2], ledger['new_units'], atol=1e-12)
    np.testing.assert_allclose(portfolio.nav, ledger['nav'], atol=1e-12)
    np.testing.assert_allclose(portfolio.realized_costs.sum(axis=1), ledger['costs'], atol=1e-12)
    assert portfolio.is_rebalancing.tolist() == [False, True, True]


def test_displayed_trade_table_matches_ledger(article, ledger):
    """Quoted turnover amounts and cost-debited NAV must agree at their displayed precision."""
    rows = re.findall(
        r'^\| (2024-01-0[34]) \| ([0-9.]+) \| ([0-9.]+) \| ([0-9.]+) \|$', article, re.M)
    assert [row[0] for row in rows] == ['2024-01-03', '2024-01-04']
    actual = np.array([[float(value) for value in row[1:]] for row in rows])
    expected = np.column_stack([
        [100, ledger['second_trade']], ledger['costs'][1:], ledger['nav'][1:]])
    np.testing.assert_allclose(actual, expected, atol=0.5e-6, rtol=0)


def test_reported_turnover_and_cost_denominators_are_explicit(article, examples, ledger):
    """Use post-cost NAV, distinguish target dates, and compare raw versus normalised charges."""
    turnover = examples['executed_turnover']
    assert np.isnan(turnover.iloc[0])
    expected = np.array([100, ledger['second_trade']]) / ledger['nav'][1:]
    np.testing.assert_allclose(turnover.iloc[1:], expected, atol=1e-12)
    target = examples['target_turnover']
    assert np.isnan(target.iloc[0]) and np.isnan(target.iloc[-1])
    assert target.loc['2024-01-03'] == pytest.approx(0.20)
    np.testing.assert_allclose(
        examples['cost_fractions'], ledger['costs'] / ledger['nav'], atol=1e-12)
    np.testing.assert_allclose(
        examples['portfolio'].get_costs(
            is_agg=True, roll_period=None, is_unit_based_traded_volume=False),
        ledger['costs'], atol=1e-12)
    assert '**1.001001** and **0.185849**' in article


def test_opening_on_first_row_charges_cost_despite_missing_turnover(examples):
    """The cash ledger assumes no pre-entry units; the turnover differencing statistic does not."""
    prices = examples['prices']
    portfolio = examples['qis'].backtest_model_portfolio(
        prices, examples['targets'].iloc[[0]], rebalancing_costs=0.0010,
        weight_implementation_lag=0)
    assert portfolio.realized_costs.iloc[0].sum() == pytest.approx(0.10)
    assert portfolio.nav.iloc[0] == pytest.approx(99.90)
    assert np.isnan(portfolio.get_turnover(is_agg=True, roll_period=None).iloc[0])


def test_gross_exposure_and_nav_turnover_differ_after_costs(examples):
    """At entry gross notional is 100 while post-cost NAV is 99.90."""
    qis, portfolio = examples['qis'], examples['portfolio']
    with pytest.warns(RuntimeWarning, match='gross exposure is zero'):
        gross = portfolio.get_turnover(
            is_agg=True, roll_period=None,
            turnover_computation_type=qis.TurnoverComputationType.EXECUTED_NOTIONAL_GROSS)
    assert gross.iloc[1] == pytest.approx(1.0)
    assert examples['executed_turnover'].iloc[1] > gross.iloc[1]
    # The legacy selector emits its own warning and retains the zero-gross warning.
    with pytest.warns((DeprecationWarning, RuntimeWarning)) as warnings:
        legacy = portfolio.get_turnover(
            is_agg=True, roll_period=None, is_unit_based_traded_volume=True)
    assert any(isinstance(item.message, DeprecationWarning) for item in warnings)
    pd.testing.assert_series_equal(gross, legacy)


def test_reporting_windows_are_observation_sums(examples):
    """Defaults are rolling 260 observations; resampling and rolling do not annualise rates."""
    portfolio = examples['portfolio']
    assert portfolio.get_turnover(is_agg=True).isna().all()
    observed = examples['executed_turnover']
    rolled = portfolio.get_turnover(is_agg=True, roll_period=2)
    np.testing.assert_allclose(rolled, observed.rolling(2).sum(), equal_nan=True)
    monthly = portfolio.get_turnover(is_agg=True, roll_period=None, freq='ME')
    assert monthly.iloc[0] == pytest.approx(observed.sum())
    costs = portfolio.get_costs(is_agg=True, roll_period=None, freq='ME')
    assert costs.iloc[0] == pytest.approx(examples['cost_fractions'].sum())


def test_rate_schedule_is_read_at_execution_and_is_causal(examples):
    """Lagged entry must use the trade-date rates; a later schedule row cannot alter past costs."""
    prices, targets = examples['prices'], examples['targets']
    schedule = pd.DataFrame(
        {'A': [0.0, 0.001, 0.003], 'B': [0.0, 0.002, 0.001]}, index=prices.index)
    first = examples['qis'].backtest_model_portfolio(
        prices, targets, rebalancing_costs=schedule, weight_implementation_lag=1)
    assert first.realized_costs.iloc[1].sum() == pytest.approx(0.14)
    expected = first.units.diff().abs().mul(prices).mul(schedule)
    np.testing.assert_allclose(first.realized_costs.iloc[1:], expected.iloc[1:], atol=1e-12)
    future = schedule.copy()
    future.loc[pd.Timestamp('2024-01-05')] = [1.0, 1.0]
    second = examples['qis'].backtest_model_portfolio(
        prices, targets, rebalancing_costs=future, weight_implementation_lag=1)
    pd.testing.assert_frame_equal(first.realized_costs, second.realized_costs)
    pd.testing.assert_series_equal(first.nav, second.nav)


def test_missing_schedule_rates_follow_documented_zero_policy(examples):
    """A late first cost row and an explicit missing cell become zero, not future-filled rates."""
    prices = examples['prices']
    schedule = pd.DataFrame(
        {'A': [0.001], 'B': [np.nan]}, index=prices.index[[-1]])
    portfolio = examples['qis'].backtest_model_portfolio(
        prices, examples['targets'], rebalancing_costs=schedule, weight_implementation_lag=1)
    np.testing.assert_allclose(portfolio.realized_costs.iloc[1], 0)
    assert portfolio.realized_costs.iloc[2, 0] > 0
    assert portfolio.realized_costs.iloc[2, 1] == 0


@pytest.mark.parametrize('kind', ['date_series', 'missing_column'])
def test_ambiguous_or_incomplete_cost_inputs_raise(examples, kind):
    """Cost-shape errors must be reported rather than silently aligned to another meaning."""
    prices = examples['prices']
    costs = (pd.Series(0.001, index=prices.index) if kind == 'date_series'
             else pd.DataFrame({'A': 0.001}, index=prices.index))
    match = 'date-indexed' if kind == 'date_series' else 'missing price columns'
    with pytest.raises(ValueError, match=match):
        examples['qis'].backtest_model_portfolio(
            prices, examples['targets'], rebalancing_costs=costs, weight_implementation_lag=1)
