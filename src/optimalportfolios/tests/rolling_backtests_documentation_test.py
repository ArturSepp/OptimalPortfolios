"""Execute the rolling article and verify timing, drift, prices and documented results."""

import importlib
import re
import runpy
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

LEGACY_ANCHORS = {
    'rolling-portfolio-backtests', 'inputs-and-conventions', 'minimal-offline-example',
    'point-in-time-and-drift-rules', 'failure-modes-and-missing-data', 'see-also',
}


@pytest.fixture(scope='module')
def article(root):
    """Read the actual migrated source; the root fixture skips installed-wheel runs."""
    return (root / 'docs/rolling_backtests.md').read_text(encoding='utf-8')


@pytest.fixture(scope='module')
def blocks(article):
    """Extract exactly the two canonical article examples."""
    matches = re.findall(r'^```python([^\n]*)\n(.*?)^```', article, re.M | re.S)
    assert len(matches) == 2 and all(not options.strip() for options, _ in matches)
    return [code for _, code in matches]


@pytest.fixture(scope='module')
def examples(blocks):
    """Run both article blocks and observe unchanged single-date solver outcomes."""
    module = importlib.import_module('optimalportfolios.optimization.general.quadratic')
    original = module.wrapper_quadratic_optimisation
    outcomes = []

    def observe(*args, **kwargs):
        """Forward the real call and retain its returned diagnostic record."""
        weights, outcome = original(*args, **kwargs)
        outcomes.append(outcome)
        return weights, outcome

    state = {'__name__': '__rolling_backtests_article__'}
    with patch.object(module, 'wrapper_quadratic_optimisation', observe):
        for index, code in enumerate(blocks, start=1):
            exec(compile(code, f'rolling_backtests.md (block {index})', 'exec'), state)
    state['outcomes'] = outcomes
    return state


def test_article_meets_source_and_link_standard(article, root):
    """Enforce the complete methodology structure and resolve every ordinary local link."""
    checker = runpy.run_path(str(root / 'tools/check_docs.py'))
    assert not checker['check_document'](article, methodology=True)
    assert not checker['check_local_links'](article, root / 'docs/rolling_backtests.md', root)


def test_legacy_fragments_and_incoming_links_survive(article, root):
    """Keep the six original public fragments and repair incoming source links."""
    checker = runpy.run_path(str(root / 'tools/check_docs.py'))
    visible, _, _ = checker['prose_lines'](article)
    anchors = set(re.findall(r'<a id="([^"]+)"></a>', article))
    for _, line in visible:
        heading = re.match(r'^#{1,6} (.+)', line)
        if heading:
            anchors.add(re.sub(r'[^\w -]', '', heading[1]).lower().replace(' ', '-'))
    assert LEGACY_ANCHORS <= anchors
    assert not (root / 'docs/rolling_backtests.rst').exists()
    for name in ('risk_budgeting.md', 'minimum_tracking_error.md'):
        source = (root / 'docs' / name).read_text(encoding='utf-8')
        assert '(rolling_backtests.rst)' not in source
        assert source.count('(rolling_backtests.md)') == 2


def test_real_monthly_solves_are_accepted_without_fallback(examples):
    """Running the construction and convenience wrapper executes eight real valid solves."""
    outcomes = examples['outcomes']
    assert len(outcomes) == 8
    assert all(outcome.accepted and outcome.compliant and outcome.fallback_source is None
               for outcome in outcomes)
    assert all(outcome.covar_factorization.n_eigenvalues_floored == 0 for outcome in outcomes)
    weights = examples['weights']
    np.testing.assert_allclose(weights.sum(axis=1), 1, atol=1e-8)
    assert (weights >= -1e-8).all().all() and (weights <= 0.80 + 1e-8).all().all()


def test_toy_holds_units_and_earns_returns_after_execution(examples):
    """Reconstruct all four observations using initial cash and the two executed trades."""
    portfolio = examples['toy_portfolio']
    expected_units = np.array([
        [0, 0], [50 / 110, 50 / 100], [50 / 110, 50 / 100],
        [55.25 / 133.1, 55.25 / 100],
    ])
    np.testing.assert_allclose(portfolio.units, expected_units, atol=1e-12, rtol=0)
    np.testing.assert_allclose(portfolio.nav, [100, 100, 105, 110.5], atol=1e-12)
    assert portfolio.is_rebalancing.tolist() == [False, True, False, True]
    pd.testing.assert_frame_equal(portfolio.prices, examples['toy_prices'])
    np.testing.assert_allclose(portfolio.realized_costs, 0, atol=0)


def test_displayed_toy_nav_matches_execution(article, examples):
    """Detect stale dates, entry returns or quoted NAV values in the rendered table."""
    rows = re.findall(r'^\| (2024-01-\d{2}) \| [^|\n]+ \| ([0-9.]+) \|$', article, re.M)
    assert len(rows) == 4
    assert [row[0] for row in rows] == list(examples['toy_dates'].strftime('%Y-%m-%d'))
    np.testing.assert_allclose(
        [float(row[1]) for row in rows], examples['toy_portfolio'].nav, atol=1e-10)


def test_drift_baseline_differs_from_executed_holdings(article, examples):
    """Use different, explicit purchase-price anchors to verify the documented distinction."""
    baseline = examples['decision_baseline']
    holdings = examples['toy_portfolio'].weights.loc['2024-01-04']
    np.testing.assert_allclose(baseline, [60.5 / 110.5, 50 / 110.5], atol=1e-12)
    np.testing.assert_allclose(holdings, [55 / 105, 50 / 105], atol=1e-12)
    assert abs(baseline.iloc[0] - holdings.iloc[0]) > 0.02
    for label, expected in [('Decision-date drift baseline', baseline),
                            ('Realised holdings with lag one', holdings)]:
        row = re.search(rf'^\| {label} \| ([0-9.]+) \| ([0-9.]+) \|$', article, re.M)
        assert row
        np.testing.assert_allclose(
            [float(row[1]), float(row[2])], expected, atol=0.5e-6, rtol=0)


@pytest.mark.parametrize('prior', [[0.6, 0.4], [0.3, 0.2], [0.8, -0.2]])
def test_drift_nav_growth_retains_cash_and_short_exposure(examples, prior):
    """Independent currency positions plus residual cash verify the drift denominator."""
    prices = examples['toy_prices']
    weights = pd.Series(prior, index=prices.columns)
    positions = 100 * weights / prices.iloc[0]
    cash = 100 * (1 - weights.sum())
    values = positions * prices.iloc[2]
    reference = values / (cash + values.sum())
    actual = examples['opt'].apply_drift_to_weights_0(
        weights, prices, prices.index[0], prices.index[2])
    np.testing.assert_allclose(actual, reference, atol=1e-12)


def test_monthly_grid_and_displayed_entry_cost(article, examples):
    """A one-observation lag on this monthly panel must not become one business day."""
    portfolio = examples['portfolio']
    rows = re.findall(r'^\| (202[01]-\d{2}-\d{2}) \| (2021-\d{2}-\d{2}) \|$', article, re.M)
    assert len(rows) == 4
    assert list(examples['weights'].index.strftime('%Y-%m-%d')) == [row[0] for row in rows]
    actual = portfolio.is_rebalancing[portfolio.is_rebalancing].index
    assert list(actual.strftime('%Y-%m-%d')) == [row[1] for row in rows]
    assert portfolio.nav.index[0] == pd.Timestamp('2020-12-31')
    assert portfolio.nav.iloc[0] == pytest.approx(100)
    assert portfolio.nav.loc['2021-01-31'] == pytest.approx(99.97, abs=1e-8)
    assert 'NAV **99.97**' in article
    pd.testing.assert_frame_equal(
        portfolio.prices, examples['prices'].loc[portfolio.prices.index])


def test_monthly_costs_and_first_holding_interval_have_independent_reference(examples):
    """Units times execution prices reproduce costs and NAV before the second trade."""
    portfolio = examples['portfolio']
    cost = portfolio.units.diff().fillna(0).abs() * portfolio.prices * 0.0003
    np.testing.assert_allclose(portfolio.realized_costs, cost, atol=1e-12)
    assert cost.loc['2021-01-31'].sum() == pytest.approx(0.03, abs=1e-8)
    units = 100 * examples['weights'].iloc[0] / examples['prices'].loc['2021-01-31']
    interval = examples['prices'].loc['2021-01-31':'2021-03-31']
    reference = interval.dot(units) - 0.03
    np.testing.assert_allclose(portfolio.nav.loc[interval.index], reference, atol=1e-8)


def test_future_prices_do_not_change_earlier_estimates_or_targets(examples):
    """A later shock cannot affect the first three dated covariances or their targets."""
    prices = examples['prices'].copy()
    mask = prices.index > pd.Timestamp('2021-06-30')
    prices.loc[mask, 'Equity'] *= np.linspace(1.5, 4.0, mask.sum())
    opt, qis = examples['opt'], examples['qis']
    covars = examples['estimator'].fit_rolling_covars(
        prices, qis.TimePeriod('31Dec2020', '30Sep2021'))
    for date, expected in examples['covar_dict'].items():
        if date <= pd.Timestamp('2021-06-30'):
            pd.testing.assert_frame_equal(covars[date], expected, atol=1e-12, rtol=0)
    weights = opt.compute_rolling_optimal_weights(
        prices=prices, constraints=examples['constraints'], covar_dict=covars,
        portfolio_objective=opt.PortfolioObjective.MIN_VARIANCE)
    np.testing.assert_allclose(
        weights.iloc[:3], examples['weights'].iloc[:3], atol=1e-10, rtol=0)


def test_reporting_filter_restarts_at_first_retained_target(examples):
    """Filtering targets is observably different from cropping an already invested NAV."""
    opt, qis = examples['opt'], examples['qis']
    filtered = opt.backtest_rolling_optimal_portfolio(
        prices=examples['prices'], constraints=examples['constraints'],
        covar_dict=examples['covar_dict'],
        perf_time_period=qis.TimePeriod('31Mar2021', '30Sep2021'),
        portfolio_objective=opt.PortfolioObjective.MIN_VARIANCE,
        rebalancing_costs=0.0003, weight_implementation_lag=1)
    assert filtered.nav.index[0] == pd.Timestamp('2021-03-31')
    assert filtered.nav.iloc[0] == pytest.approx(100)
    np.testing.assert_allclose(filtered.units.iloc[0], 0)
    assert examples['portfolio'].nav.loc['2021-03-31'] > 100.4
    assert (examples['portfolio'].units.loc['2021-03-31'] > 0).all()
    assert filtered.is_rebalancing[filtered.is_rebalancing].index[0] == pd.Timestamp('2021-04-30')


def test_off_grid_decision_maps_forward_before_lag(examples):
    """A Saturday decision maps to Monday, then lag one trades on Tuesday."""
    dates = pd.to_datetime(['2024-01-05', '2024-01-08', '2024-01-09'])
    prices = pd.DataFrame({'A': [100., 110., 121.], 'B': [100., 100., 100.]}, index=dates)
    weights = pd.DataFrame(
        [[0.5, 0.5]], index=pd.to_datetime(['2024-01-06']), columns=prices.columns)
    result = examples['qis'].backtest_model_portfolio(
        prices=prices, weights=weights, weight_implementation_lag=1, rebalancing_costs=0)
    assert result.is_rebalancing.tolist() == [False, False, True]
    np.testing.assert_allclose(result.nav, 100)


def test_unexecutable_tail_warns_and_duplicate_execution_dates_raise(examples):
    """Exercise the article's distinct end-of-panel and colliding-target qualifications."""
    prices = examples['toy_prices']
    targets = examples['toy_targets'].copy()
    targets.loc[prices.index[-1]] = [0.5, 0.5]
    qis = examples['qis']
    with pytest.warns(UserWarning, match='trade past the end'):
        result = qis.backtest_model_portfolio(
            prices=prices, weights=targets, weight_implementation_lag=1)
    assert result.is_rebalancing.tolist() == [False, True, False, True]
    with pytest.raises(ValueError, match='no weight date is traded'):
        qis.backtest_model_portfolio(prices, targets.iloc[[-1]], weight_implementation_lag=1)
    sparse_prices = prices.iloc[[0, 3]]
    with pytest.raises(ValueError, match='resolve to'):
        qis.backtest_model_portfolio(sparse_prices, targets.iloc[[1, 2]])
