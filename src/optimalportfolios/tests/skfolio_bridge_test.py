"""L1 label/date/universe contracts and B0 ledger reconciliation, without a second engine."""
import builtins
import importlib
import importlib.util

import numpy as np
import pandas as pd
import pytest


@pytest.fixture(scope='module')
def bridge(root):
    """Load the repository-only optional example; installed wheels skip via root."""
    return importlib.import_module('examples.integrations.skfolio_targets_local')


@pytest.fixture(scope='module')
def benchmark(root):
    """Reuse B0's owning ledger oracle, input hashes and native skfolio comparisons."""
    return importlib.import_module('tools.matched_benchmarks')


@pytest.fixture
def panel():
    """Repeat B0's tiny input; each case proves its input hash equals the B0 fixture."""
    dates = pd.bdate_range('2024-01-01', periods=5)
    prices = pd.DataFrame({'A': [100., 110., 121., 121., 133.1],
                           'B': [100., 90., 90., 99., 99.]}, index=dates)
    weights = pd.DataFrame([[.5, .5], [.25, .75]], columns=prices.columns, index=dates[[0, 2]])
    return prices, weights


def arguments(weights, executions=None):
    """State targets as decisions just before the given exact execution quotes."""
    executions = weights.index if executions is None else executions
    targets = weights.copy()
    targets.index = weights.index - pd.Timedelta(nanoseconds=1)
    return dict(targets=targets, information_dates=pd.Series(targets.index, index=targets.index),
                execution_dates=pd.Series(executions, index=targets.index),
                return_units='simple_fraction', moment_units='per_observation')


@pytest.mark.parametrize('case', ['zero_cost_rebalance', 'buy_hold_drift', 'entry_trade_cost',
                                 'rebalance_trade_cost', 'implementation_lag_one',
                                 'unquoted_position_freeze'])
def test_bridge_reconciles_all_b0_ledgers(bridge, benchmark, panel, case):
    """All six B0 targets preserve NAV, units, trade cost and exact execution timing."""
    prices, weights = panel
    cost = .01 if 'trade_cost' in case else 0.
    lag = 1 if case == 'implementation_lag_one' else 0
    if case in ['buy_hold_drift', 'entry_trade_cost']:
        weights = weights.iloc[:1]
    if case == 'unquoted_position_freeze':
        prices.iloc[2, 1] = np.nan
    with benchmark.offline():
        records = benchmark.accounting_cases(include_skfolio=False)
        record = next(r for r in records if r['case'] == case)
        assert benchmark.array_hash(prices, weights) == record['input_hash']
        executions = prices.index[prices.index.get_indexer(weights.index) + lag]
        portfolio = bridge.evaluate_targets(prices, transaction_costs=cost,
                                            **arguments(weights, executions))
        reference = importlib.import_module('tools.matched_benchmark_references')
        ledger = reference.ledger_reference(prices, weights, lag, cost)
    offset = prices.index.get_loc(executions[0])
    np.testing.assert_allclose(portfolio.nav, ledger['nav'][offset:], atol=1e-11, rtol=0)
    np.testing.assert_allclose(portfolio.units, ledger['units'][offset:], atol=1e-12, rtol=0)
    np.testing.assert_allclose(portfolio.realized_costs, ledger['costs'][offset:], atol=1e-12)


def test_column_permutation_is_label_invariant(bridge, panel):
    """A reversed evaluation panel must not swap the meaning of asymmetric target weights."""
    prices, weights = panel
    first = bridge.evaluate_targets(prices, **arguments(weights))
    second = bridge.evaluate_targets(prices[['B', 'A']], **arguments(weights))
    np.testing.assert_allclose(first.nav, second.nav, rtol=0, atol=1e-12)
    np.testing.assert_allclose(first.units, second.units[['A', 'B']], rtol=0, atol=1e-12)


@pytest.mark.parametrize('key,value', [('return_units', 'percent'), ('return_units', 'log'),
                                     ('moment_units', 'annualized')])
def test_incompatible_units_fail(bridge, panel, key, value):
    """No implicit unit conversion can change the interpretation of a supplied target."""
    prices, weights = panel
    args = arguments(weights)
    args[key] = value
    with pytest.raises(ValueError, match='simple_fraction'):
        bridge.evaluate_targets(prices, **args)


@pytest.mark.parametrize('defect', ['cutoff_at_fill', 'future_information', 'decision_after_fill',
                                  'nonquote_fill', 'duplicate_fill', 'late_fill',
                                  'misindexed_series', 'naive_aware', 'unsorted_prices'])
def test_date_contract_rejects_unimplementable_targets(bridge, panel, defect):
    """Cutoffs, clocks and exact quote dates are validated before calling the engine."""
    prices, weights = panel
    args = arguments(weights)
    if defect == 'cutoff_at_fill':
        args['information_dates'] = args['execution_dates'].copy()
    elif defect == 'future_information':
        args['information_dates'] += pd.Timedelta(days=2)
    elif defect == 'decision_after_fill':
        args['execution_dates'] -= pd.Timedelta(days=1)
    elif defect == 'nonquote_fill':
        args['execution_dates'] += pd.Timedelta(hours=1)
    elif defect == 'duplicate_fill':
        args['execution_dates'].iloc[1] = args['execution_dates'].iloc[0]
    elif defect == 'late_fill':
        args['execution_dates'].iloc[1] = prices.index[-1]
    elif defect == 'misindexed_series':
        args['information_dates'] = args['information_dates'].iloc[::-1]
    elif defect == 'naive_aware':
        args['execution_dates'] = args['execution_dates'].dt.tz_localize('UTC')
    else:
        prices = prices.iloc[::-1]
    with pytest.raises(ValueError):
        bridge.evaluate_targets(prices, **args)


def test_late_entrant_zero_until_first_quote(bridge, panel):
    """A pre-entry zero weight remains zero; the asset can enter on its first priced date."""
    prices, weights = panel
    prices.loc[prices.index[:2], 'B'] = np.nan
    weights.iloc[0] = [1., 0.]
    result = bridge.evaluate_targets(prices, **arguments(weights))
    np.testing.assert_allclose(result.units['B'].iloc[:2], 0.)
    assert result.units['B'].iloc[2] > 0
    bad = weights.copy()
    bad.iloc[0] = [.5, .5]
    with pytest.raises(ValueError, match='before the first quote'):
        bridge.evaluate_targets(prices, **arguments(bad))


def test_exclusions_are_explicit_and_do_not_renormalise(bridge, panel):
    """Absent assets require explicit exclusion; unknown or nonzero excluded targets fail."""
    prices, weights = panel
    only_a = weights[['A']].copy()
    only_a['A'] = 1.
    with pytest.raises(ValueError, match='explicit exclusions'):
        bridge.evaluate_targets(prices, **arguments(only_a))
    result = bridge.evaluate_targets(prices, excluded_assets=('B',), **arguments(only_a))
    np.testing.assert_allclose(result.units['B'], 0.)
    for excluded in [('unknown',), ('B', 'B'), ('A',)]:
        with pytest.raises(ValueError, match='explicit exclusions'):
            bridge.evaluate_targets(prices, excluded_assets=excluded, **arguments(only_a))
    with pytest.raises(ValueError, match='explicit exclusions'):
        bridge.evaluate_targets(prices, excluded_assets=('B',), **arguments(weights))


@pytest.mark.parametrize('defect', ['nan_weight', 'negative_weight', 'underinvested',
                                  'duplicate_asset', 'infinite_price', 'zero_price',
                                  'negative_cost', 'nan_cost'])
def test_invalid_economic_inputs_fail(bridge, panel, defect):
    """Bad marks, nonfinite targets and accidental scaling cannot reach the QIS kernel."""
    prices, weights = panel
    cost = 0.
    if defect == 'nan_weight':
        weights.iloc[0, 0] = np.nan
    elif defect == 'negative_weight':
        weights.iloc[0] = [-.5, 1.5]
    elif defect == 'underinvested':
        weights *= .01
    elif defect == 'duplicate_asset':
        weights.columns = ['A', 'A']
    elif defect == 'infinite_price':
        prices.iloc[1, 0] = np.inf
    elif defect == 'zero_price':
        prices.iloc[1, 0] = 0.
    else:
        cost = -.001 if defect == 'negative_cost' else np.nan
    with pytest.raises(ValueError):
        bridge.evaluate_targets(prices, transaction_costs=cost, **arguments(weights))


def test_missing_optional_dependency_keeps_bridge_usable(bridge, panel, monkeypatch):
    """Block skfolio imports, exercise neutral evaluation and require an actionable fit error."""
    original = builtins.__import__
    def without_skfolio(name, *args, **kwargs):
        """Simulate a core install even when optional integration tools are on sys.path."""
        if name == 'skfolio' or name.startswith('skfolio.'):
            raise ModuleNotFoundError("No module named 'skfolio'", name='skfolio')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', without_skfolio)
    prices, weights = panel
    assert np.isfinite(bridge.evaluate_targets(prices, **arguments(weights)).nav).all()
    with pytest.raises(ModuleNotFoundError, match='Optional example requires'):
        bridge.run_example()


def test_optional_bridge_preserves_native_b0_differences(bridge, benchmark, panel):
    """Reconcile gross returns first; retain native cost/freeze mismatches without translating."""
    if importlib.util.find_spec('skfolio') is None:
        pytest.skip('L1 optional integration requires pinned skfolio')
    with benchmark.offline():
        records = benchmark.accounting_cases(include_skfolio=True)
    for record in records:
        prices, weights = panel[0].copy(), panel[1].copy()
        if record['case'] in ['buy_hold_drift', 'entry_trade_cost']:
            weights = weights.iloc[:1]
        if record['case'] == 'unquoted_position_freeze':
            prices.iloc[2, 1] = np.nan
        executions = prices.index[prices.index.get_indexer(weights.index) + record['lag']]
        evaluated = bridge.evaluate_targets(prices, transaction_costs=record['cost_rate'],
                                             **arguments(weights, executions))
        offset = prices.index.get_loc(executions[0])
        q = np.asarray(evaluated.nav)
        native = np.asarray(record['skfolio_nav'])[offset:]
        np.testing.assert_allclose(q, np.asarray(record['qis_nav'])[offset:], atol=1e-11)
        if record['expected_match']:
            np.testing.assert_allclose(np.diff(q)/q[:-1], np.diff(native)/native[:-1], atol=1e-12)
        else:
            assert np.max(np.abs(q - native)) > 1e-4
    assert sum(r['expected_match'] for r in records) == 3


def test_optional_pinned_example_runs_offline(bridge, benchmark):
    """Run real chronological native fitting through both canonical gross and net accounting."""
    if importlib.util.find_spec('skfolio') is None:
        pytest.skip('L1 optional integration requires pinned skfolio')
    with benchmark.offline():
        result = bridge.run_example()
    assert result['assets'] == 6 and result['observations'] > 24
    assert result['net_final_nav'] < result['gross_final_nav']
    assert result['realised_trade_cost'] > 0
    assert all(a < b for a, b in zip(result['information_cutoffs'], result['executions']))


def test_optional_future_prices_do_not_change_targets(bridge, benchmark, monkeypatch):
    """Perturb only prices after the final cutoff; fitted weights cannot see those rows."""
    if importlib.util.find_spec('skfolio') is None:
        pytest.skip('L1 optional integration requires pinned skfolio')
    from types import SimpleNamespace
    from optimalportfolios.tests.data import multiasset
    original = multiasset.load_multiasset_data
    def changed_future():
        """Return a modified copy, preserving the original packaged fixture unchanged."""
        prices = original().prices.copy()
        future = prices.index > pd.Timestamp('2020-12-31')
        prices.loc[future] *= np.linspace(1., 3., int(future.sum()))[:, None]
        return SimpleNamespace(prices=prices)
    with benchmark.offline():
        baseline = bridge.run_example()
        monkeypatch.setattr(multiasset, 'load_multiasset_data', changed_future)
        changed = bridge.run_example()
    assert baseline['targets'] == changed['targets']
    assert not np.isclose(baseline['gross_final_nav'], changed['gross_final_nav'])
