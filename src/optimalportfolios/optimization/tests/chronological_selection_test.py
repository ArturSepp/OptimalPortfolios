"""Chronological selection guards: isolation, timing and independent cash accounting."""
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import qis

from optimalportfolios.optimization.chronological_selection import (
    ChronologicalFold, select_portfolio_model,
)
from optimalportfolios.optimization.tests.executed_state_test import assert_ledger


def fixture():
    """Return deterministic price data, disjoint folds and two bounded settings."""
    t = np.arange(40)
    dates = pd.bdate_range('2024-01-02', periods=len(t))
    prices = pd.DataFrame({'A': 100 * np.exp(.004 * t + .01 * np.sin(t)),
                           'B': 100 * np.exp(-.002 * t + .008 * np.cos(t))}, index=dates)
    folds = (ChronologicalFold(dates[9], dates[[10, 13]], dates[17]),
             ChronologicalFold(dates[19], dates[[20, 23]], dates[27]))
    outer = ChronologicalFold(dates[31], dates[[32, 35]], dates[39])
    grid = {'growth': {'covariance': 8, 'alpha': .75},
            'defensive': {'covariance': 16, 'alpha': .25}}
    return prices, folds, outer, grid


def execute(prices=None, fit=None, **kwargs):
    """Run the fixed nested search while allowing one input to be replaced."""
    p, folds, outer, grid = fixture()
    if fit is None:
        def fit(training, configuration):
            """Build an isolated fixed-target policy for the candidate."""
            def decide(history, state):
                """Use a labelled, fixed long-only target."""
                return pd.Series([configuration['alpha'], 1 - configuration['alpha']],
                                 index=history.columns)
            return decide
    args = dict(prices=p if prices is None else prices, candidates=grid, fit=fit,
                inner_folds=folds, outer_fold=outer, label_horizon=1, embargo=1,
                transaction_costs=.001)
    args.update(kwargs)
    return select_portfolio_model(**args)


@pytest.mark.parametrize('lag', [0, 1])
def test_scores_reconcile_with_qis_and_independent_ledger(lag):
    """Selection and held-out scores include entry fees and canonical unit drift."""
    p, folds, outer, grid = fixture()
    results = []
    import optimalportfolios.optimization.chronological_selection as module
    original = module.backtest_with_qis_state

    def capture(*args, **kwargs):
        """Record every independently simulated fold without replacing accounting."""
        result = original(*args, **kwargs)
        assert_ledger(result, args[0], .001)
        results.append(result)
        return result

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(module, 'backtest_with_qis_state', capture)
        result = execute(implementation_lag=lag)
    assert len(results) == len(grid) * len(folds) + 1
    for row, portfolio in zip(result.fold_scores.itertuples(), results):
        expected = portfolio.portfolio.nav.iloc[-1] / 100 - 1
        assert row.net_return == pytest.approx(expected, abs=1e-13)
        assert row.net_return == pytest.approx(float(qis.compute_total_return(
            pd.Series([100., portfolio.portfolio.nav.iloc[-1]],
                      index=portfolio.portfolio.nav.index[[0, -1]]))))
    assert result.outer_score == pytest.approx(result.outer.portfolio.nav.iloc[-1] / 100 - 1)
    assert result.outer_score == pytest.approx(float(qis.compute_total_return(pd.Series(
        [100., result.outer.portfolio.nav.iloc[-1]],
        index=result.outer.portfolio.nav.index[[0, -1]]))))
    expected = result.fold_scores.groupby('candidate', sort=False).net_return.mean()
    pd.testing.assert_series_equal(result.scores.mean_net_return, expected,
                                   check_names=False)
    assert result.selected == expected.idxmax()
    assert result.outer.portfolio.nav.index.equals(p.loc[outer.decision_dates[0]:].index)
    if lag == 0:
        assert result.outer.portfolio.nav.iloc[0] < 100
        assert result.outer_score < float(qis.compute_total_return(result.outer.portfolio.nav))


def test_cutoffs_purging_label_bounds_and_factory_isolation():
    """Factories see only past training prices/configuration, with no tuning artifacts."""
    p, _, _, grid = fixture()
    original = p.copy(deep=True)
    calls = []

    def fit(training, configuration):
        """Record a fitted transform and prior, then intentionally mutate local inputs."""
        assert set(configuration) == {'covariance', 'alpha'}
        calls.append((training.copy(), deepcopy(configuration), training.mean(), training.std()))
        training.iloc[:] = -123
        configuration['alpha'] = 0
        configuration['covariance'] = -1

        def decide(history, state):
            """Hold cash so this test isolates factory inputs."""
            assert (history > 0).all().all()
            return None
        return decide

    result = execute(prices=p, fit=fit, candidates=grid)
    assert len(calls) == 5
    for call, pos in zip(calls, [7, 17, 7, 17, 29]):
        pd.testing.assert_frame_equal(call[0], original.iloc[:pos + 1])
        pd.testing.assert_series_equal(call[2], original.iloc[:pos + 1].mean())
        pd.testing.assert_series_equal(call[3], original.iloc[:pos + 1].std())
    pd.testing.assert_frame_equal(p, original)
    assert grid == fixture()[3]
    assert result.selected == 'growth'
    assert result.scores['rank'].tolist() == [1, 2]
    assert result.provenance.train_end.tolist() == p.index[[7, 17, 29]].tolist()
    assert result.provenance.label_end.tolist() == p.index[[8, 18, 30]].tolist()
    assert result.provenance.purged_rows.tolist() == [2, 2, 2]
    assert result.provenance.role.tolist() == ['independent_inner'] * 2 + ['sequential_outer']
    assert (result.provenance.label_end < result.provenance.evaluation_start).all()


def test_future_shocks_cannot_change_selection_or_fitted_transforms():
    """An extreme outer shock changes only outer performance, never training or tuning."""
    prices, _, _, _ = fixture()
    calls = []

    def fit(training, configuration):
        """Fit a centering/scaling transform and prior wholly within this training fold."""
        calls.append((training.copy(), training.mean(), training.std(), deepcopy(configuration)))
        target = configuration['alpha']

        def decide(history, state):
            """Use the frozen candidate estimate without fitting on evaluation prices."""
            return pd.Series([target, 1 - target], index=history.columns)
        return decide

    baseline = execute(fit=fit)
    first = calls.copy()
    calls.clear()
    prices.iloc[32:, 0] *= np.exp(np.arange(8) * .2)
    shocked = execute(prices=prices, fit=fit)
    assert baseline.selected == shocked.selected
    pd.testing.assert_frame_equal(baseline.scores, shocked.scores)
    pd.testing.assert_frame_equal(baseline.fold_scores, shocked.fold_scores)
    for a, b in zip(first, calls):
        pd.testing.assert_frame_equal(a[0], b[0])
        pd.testing.assert_series_equal(a[1], b[1])
        pd.testing.assert_series_equal(a[2], b[2])
        assert a[3] == b[3]
    assert baseline.outer_score != shocked.outer_score


def test_reset_and_sequential_state_and_repeatability():
    """Independent factories start in cash; later decisions receive actual held state."""
    starts, later = [], []

    def fit(training, configuration):
        """Create a fresh policy with no cross-fold holdings or fitted-state sharing."""
        count = 0

        def decide(history, state):
            """Record reset/propagation and use cash-state feedback in the target."""
            nonlocal count
            assert history.index[-1] == state.date
            if count == 0:
                starts.append((state.cash, state.nav, state.units.sum()))
            else:
                later.append(state)
                assert state.units.abs().sum() > 0
                assert state.cash != 100
                assert state.nav != 100
            count += 1
            weight = configuration['alpha'] if count == 1 else .5
            return pd.Series([weight, 1 - weight], index=history.columns)
        return decide

    result = execute(fit=fit)
    repeated = execute(fit=fit)
    assert starts == [(100., 100., 0.)] * 10
    assert len(later) == 10
    assert result.selected == repeated.selected
    pd.testing.assert_frame_equal(result.scores, repeated.scores)
    pd.testing.assert_frame_equal(result.outer.targets, repeated.outer.targets)
    pd.testing.assert_series_equal(result.outer.portfolio.nav, repeated.outer.portfolio.nav)


def test_no_gap_when_labels_are_contemporaneous():
    """With no label overlap or embargo the explicit requested cutoff is respected."""
    result = execute(label_horizon=0, embargo=0)
    p = fixture()[0]
    assert result.provenance.train_end.tolist() == p.index[[9, 19, 31]].tolist()
    assert result.provenance.purged_rows.tolist() == [0, 0, 0]


@pytest.mark.parametrize('candidates', [{}, {str(i): {} for i in range(33)}, [],
                                         {'bad': 1}, {'': {}}, {1: {}}])
def test_invalid_grids(candidates):
    """Reject ambiguous or unbounded searches before fitting."""
    with pytest.raises(ValueError, match='configuration'):
        execute(candidates=candidates)


@pytest.mark.parametrize('key,value', [('label_horizon', -1), ('label_horizon', True),
                                     ('embargo', 1.5), ('embargo', -1),
                                     ('label_horizon', 40)])
def test_invalid_gaps(key, value):
    """Gaps must count observations and preserve a nonempty training history."""
    with pytest.raises(ValueError):
        execute(**{key: value})


@pytest.mark.parametrize('kind', ['not_frame', 'not_datetime', 'empty', 'nat',
                                 'duplicate', 'unordered'])
def test_invalid_prices(kind):
    """Reject ambiguous observation calendars."""
    p = fixture()[0]
    if kind == 'not_frame':
        p = p.A
    elif kind == 'not_datetime':
        p.index = range(len(p))
    elif kind == 'empty':
        p = p.iloc[:0]
    elif kind == 'nat':
        p.index = pd.DatetimeIndex([pd.NaT, *p.index[1:]])
    elif kind == 'duplicate':
        p.index = pd.DatetimeIndex([p.index[1], *p.index[1:]])
    else:
        p = p.iloc[::-1]
    with pytest.raises(ValueError, match='prices need'):
        execute(prices=p)


@pytest.mark.parametrize('kind', ['wrong_type', 'not_datetime', 'empty', 'nat', 'duplicate',
                                 'unordered', 'timezone', 'unobserved', 'train_future',
                                 'no_tail', 'bad_boundary'])
def test_invalid_folds(kind):
    """Fold boundaries must be observed, causal and executable on one calendar."""
    p, folds, outer, _ = fixture()
    f = folds[0]
    dates = f.decision_dates
    if kind == 'wrong_type':
        f = object()
    elif kind == 'train_future':
        f = replace(f, train_end=p.index[11])
    elif kind == 'no_tail':
        f = replace(f, evaluation_end=dates[-1])
    elif kind == 'bad_boundary':
        f = replace(f, train_end=p.index[9] + pd.Timedelta(hours=1))
    else:
        choices = {'not_datetime': [dates[0]], 'empty': dates[:0],
                   'nat': pd.DatetimeIndex([pd.NaT]), 'duplicate': dates[[0, 0]],
                   'unordered': dates[::-1], 'timezone': dates.tz_localize('UTC'),
                   'unobserved': dates + pd.Timedelta(hours=1)}
        f = replace(f, decision_dates=choices[kind])
    with pytest.raises((ValueError, TypeError)):
        execute(inner_folds=(f,), outer_fold=outer)


@pytest.mark.parametrize('kind', ['empty', 'many', 'overlap', 'reverse', 'outer_training'])
def test_invalid_inner_windows(kind):
    """Research windows stay disjoint, chronological and inside allowed outer history."""
    p, folds, outer, _ = fixture()
    choices = {'empty': (), 'many': folds * 6, 'overlap': (folds[0], folds[0]),
               'reverse': folds[::-1], 'outer_training': (replace(
                   folds[1], evaluation_end=p.index[30]),)}
    with pytest.raises(ValueError):
        execute(inner_folds=choices[kind], outer_fold=outer)


def test_factory_must_return_policy_and_errors_propagate():
    """Do not silently drop a failed candidate or change the selection universe."""
    with pytest.raises(TypeError, match='callable'):
        execute(fit=lambda training, config: None)

    def fail(training, config):
        """Represent an estimator fit failure that must reach the caller."""
        raise RuntimeError('bad fit')

    with pytest.raises(RuntimeError, match='bad fit'):
        execute(fit=fail)


def test_nonfinite_scores_fail_closed(monkeypatch):
    """A missing score cannot become an implicitly discarded fold."""
    monkeypatch.setattr(qis, 'compute_total_return', lambda prices: np.nan)
    with pytest.raises(ValueError, match='finite'):
        execute()


def test_canonical_covariance_alpha_example(root):
    """Execute the offline example with real covariance estimation and constrained solves."""
    from examples.integrations.skfolio_targets_local import run_chronological_selection_example

    result = run_chronological_selection_example()
    assert len(result.scores) == 4
    assert len(result.fold_scores) == 8
    assert result.outer_score == pytest.approx(result.outer.portfolio.nav.iloc[-1] / 100 - 1)
