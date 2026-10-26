"""Offline contracts for public alpha, cash-preserving OP solves and QIS execution."""
from dataclasses import replace
import hashlib
import json

import numpy as np
import pandas as pd
import pytest
import optimalportfolios as op
import qis

from examples.oss_portfolio_pipeline import covariance_validation_local as covariance_checks
from examples.oss_portfolio_pipeline.backtest import (
    BacktestConfig, build_alpha_scores, execution_outcome, portfolio_constraints,
    run_backtest, save_backtest,
)
from examples.oss_portfolio_pipeline.covariance import estimate_covariances, save_covariances
from examples.oss_portfolio_pipeline.pipeline import build_inputs, save_inputs


@pytest.fixture(scope='module')
def prepared():
    """Extend synthetic history beyond leap-year February ME for an executable weekend fit."""
    inputs, config = covariance_checks.prepared.__wrapped__()
    extra = pd.DataFrame(
        [inputs.source_prices.iloc[-1] * 1.002, inputs.source_prices.iloc[-1] * 1.003],
        index=pd.to_datetime(['2020-03-02', '2020-03-03']))
    config = replace(config, end='2020-03-04')
    inputs = build_inputs(pd.concat([inputs.source_prices, extra]), config)
    risk = estimate_covariances(inputs, config, solver='CLARABEL')
    return inputs, risk, config


@pytest.fixture(scope='module')
def completed(prepared):
    """Force a positive cash sleeve so dropping BIL cannot masquerade as a valid allocation."""
    inputs, risk, config = prepared
    return run_backtest(inputs, risk, config, BacktestConfig(solver='CLARABEL', group_band=.02))


def test_risk_budget_and_holdings_constraints(prepared, completed):
    """Independently check TE quadratic forms, sleeve bounds, cash and actual-holding turnover."""
    inputs, risk, _ = prepared
    targets = completed.executed.targets
    assert targets.index.tolist() == list(pd.to_datetime(
        ['2019-12-31', '2020-01-31', '2020-02-28']))
    assert targets.columns.equals(inputs.universe.index)
    np.testing.assert_allclose(targets.sum(axis=1), 1., atol=1e-8)
    assert targets.min().min() >= 0.
    assert targets.max().max() <= .30 + 1e-7
    assert targets.BIL.ge(.03 - 1e-7).all()
    assert completed.diagnostics.accepted.all() and completed.diagnostics.compliant.all()
    assert completed.constraint_audit.passed.all()
    policy = portfolio_constraints(inputs, completed.settings)
    groups = policy.group_lower_upper_constraints
    allocation = targets @ groups.group_loadings
    assert (allocation >= groups.group_min_allocation - 1e-7).all().all()
    assert (allocation <= groups.group_max_allocation + 1e-7).all().all()
    for date in targets.index:
        active = (targets.loc[date] - inputs.benchmark_weights).to_numpy()
        independent_te = np.sqrt(active @ risk.covariances[date].to_numpy() @ active)
        assert independent_te <= completed.settings.tracking_error + 1e-6
        assert completed.diagnostics.loc[date, 'ex_ante_tracking_error'] == pytest.approx(
            independent_te, abs=1e-12)
    for state in completed.executed.states[1:]:
        turnover = (targets.loc[state.date] - state.weights).abs().sum()
        assert turnover <= completed.settings.turnover_limit + 1e-6


def test_units_costs_and_benchmark_reference(prepared, completed):
    """Recover opening units, held-unit NAV, actual trade costs and the fixed benchmark."""
    inputs, _, config = prepared
    prices = inputs.prices.loc[config.backtest_start:]
    first, following = prices.index[:2]
    target = completed.executed.targets.loc[first]
    units = 100 * target / prices.loc[first]
    cash = 100 * (1 - target.sum())
    opening_cost = 100 * target.abs().sum() * completed.settings.transaction_cost
    np.testing.assert_allclose(completed.executed.portfolio.units.loc[first], units, atol=1e-10)
    assert completed.navs.loc[first, 'Active (net)'] == pytest.approx(100 - opening_cost)
    assert completed.navs.loc[following, 'Active (net)'] == pytest.approx(
        units @ prices.loc[following] + cash - opening_cost, abs=1e-10)
    portfolio = completed.executed.portfolio
    traded_units = portfolio.units.diff().fillna(portfolio.units.iloc[0])
    expected_cost = traded_units.abs() * prices * completed.settings.transaction_cost
    np.testing.assert_allclose(portfolio.realized_costs, expected_cost, atol=1e-12)
    benchmark_units = 100 * inputs.benchmark_weights / prices.loc[first]
    assert completed.navs.loc[following, 'Benchmark (gross)'] == pytest.approx(
        benchmark_units @ prices.loc[following], abs=1e-10)
    np.testing.assert_allclose(completed.gross_benchmark.nav, inputs.benchmark.nav, atol=1e-10)


def test_alpha_and_portfolio_prefix_invariance(prepared, completed):
    """Changing future observations and refitting prefixes cannot alter prior signals or targets."""
    inputs, risk, config = prepared
    cutoff = pd.Timestamp('2019-12-31')
    changed_prices = inputs.prices.copy()
    changed_prices.loc[changed_prices.index > cutoff, 'SPY'] *= 1.5
    _, changed_scores = build_alpha_scores(changed_prices, risk, completed.settings)
    pd.testing.assert_series_equal(changed_scores.loc[cutoff], completed.alpha_scores.loc[cutoff])
    short_config = replace(config, end='2020-02-01')
    short_inputs = build_inputs(inputs.source_prices, short_config)
    short_risk = estimate_covariances(short_inputs, short_config, solver='CLARABEL')
    short = run_backtest(short_inputs, short_risk, short_config, completed.settings)
    pd.testing.assert_series_equal(short.executed.targets.loc[cutoff],
                                    completed.executed.targets.loc[cutoff])
    np.testing.assert_allclose(short.navs.iloc[:-1],
                               completed.navs.loc[short.navs.index[:-1]], atol=1e-10)
    # The Jan 31 terminal decision is omitted, so no pointless closing trade costs appear.
    assert short.executed.targets.index.tolist() == [cutoff]
    terminal = short.navs.index[-1]
    for label, portfolio in [('Active (net)', completed.executed.portfolio),
                              ('Benchmark (net)', completed.benchmark)]:
        assert short.navs.loc[terminal, label] == pytest.approx(
            completed.navs.loc[terminal, label] + portfolio.realized_costs.loc[terminal].sum())


def test_24_month_signal_uses_complete_causal_prehistory(prepared):
    """Check 24-month readiness, a direct price-ratio reference, and future-price invariance."""
    inputs, risk, _ = prepared
    settings = BacktestConfig(solver='CLARABEL', momentum_span=24)
    raw, scores = build_alpha_scores(inputs.prices, risk, settings)
    cutoff = pd.Timestamp('2019-12-31')
    assert raw.index[0] == cutoff and np.isfinite(raw).all().all()
    monthly = inputs.prices.groupby(inputs.prices.index.to_period('M')).last()
    monthly.index = monthly.index.to_timestamp('M')
    monthly = monthly.loc[:risk.schedule.index[-1]]
    relative = monthly.div(monthly.BIL, axis=0).drop(columns='BIL')
    _, expected = op.compute_momentum_alpha(
        relative, returns_freq='ME', long_span=24, short_span=None, vol_span=13,
        mean_adj_type=qis.MeanAdjType.NONE)
    np.testing.assert_allclose(raw.drop(columns='BIL'), expected.loc[risk.schedule.index],
                               atol=1e-10)
    changed = inputs.prices.copy()
    changed.loc[changed.index > cutoff, 'SPY'] *= 2.
    _, changed_scores = build_alpha_scores(changed, risk, settings)
    pd.testing.assert_series_equal(scores.loc[cutoff], changed_scores.loc[cutoff])
    with pytest.raises(ValueError, match='Momentum warm-up is incomplete'):
        build_alpha_scores(inputs.prices.loc['2017-12-01':], risk, settings)


def test_rejected_optimisation_cannot_be_reported_as_success(prepared, monkeypatch):
    """A finite fallback weight vector is still an explicit failed optimisation."""
    import examples.oss_portfolio_pipeline.backtest as backtest
    inputs, risk, config = prepared

    def rejected(**kwargs):
        """Return an apparently harmless benchmark with a rejected solver outcome."""
        return op.OptimizationOutcome(
            weights=inputs.benchmark_weights.to_numpy(), accepted=False, solver='CLARABEL',
            status='infeasible', context=kwargs['context'], reason='deliberate test failure')

    monkeypatch.setattr(backtest.op, 'cvx_maximise_alpha_over_tre', rejected)
    with pytest.raises(RuntimeError, match='optimisation failed.*infeasible'):
        run_backtest(inputs, risk, config, BacktestConfig(solver='CLARABEL'))


def test_cash_solver_analytic_reference():
    """The two-asset OP solution with zero-risk cash has a closed-form TE-limited weight."""
    names = pd.Index(['risky', 'BIL'])
    covariance = np.diag([.04, 0.])
    policy = op.Constraints(
        benchmark_weights=pd.Series([.5, .5], index=names),
        min_weights=pd.Series(0., index=names), max_weights=pd.Series(1., index=names),
        tracking_err_vol_constraint=.03)
    result = op.cvx_maximise_alpha_over_tre(covariance, np.array([1., 0.]), policy,
                                           solver='CLARABEL')
    assert result.accepted and result.compliant
    np.testing.assert_allclose(result.weights, [.5 + .03/.2, .5 - .03/.2], atol=1e-6)
    corrected = execution_outcome(replace(result, weights=result.weights * (1 + 2e-8)))
    assert corrected.weights.sum() == pytest.approx(1., abs=1e-14)
    assert corrected.compliant
    with pytest.raises(RuntimeError, match='bounded round-off'):
        execution_outcome(replace(result, weights=result.weights * 1.001))


def test_positive_execution_lag(prepared):
    """The next-session option changes execution prices without changing the first decision."""
    inputs, risk, config = prepared
    settings = BacktestConfig(solver='CLARABEL', implementation_lag=1, transaction_cost=0.)
    result = run_backtest(inputs, risk, config, settings)
    assert result.executed.events.iloc[0].execution == pd.Timestamp('2020-01-01')
    assert result.executed.portfolio.units.iloc[0].eq(0.).all()
    assert result.benchmark.units.iloc[0].eq(0.).all()
    entry = result.executed.events.iloc[0].execution
    units = 100 * result.executed.targets.iloc[0] / inputs.prices.loc[entry]
    np.testing.assert_allclose(result.executed.portfolio.units.loc[entry], units, atol=1e-10)


def test_saved_backtest_links_risk_bundle(prepared, completed, tmp_path):
    """Hash every saved portfolio table and retain the exact risk-bundle identity."""
    inputs, risk, config = prepared
    save_inputs(inputs, config, tmp_path)
    save_covariances(risk, tmp_path)
    save_backtest(completed, tmp_path)
    manifest = json.loads((tmp_path / 'backtest_manifest.json').read_text())
    assert manifest['covariance_manifest_sha256'] == hashlib.sha256(
        (tmp_path / 'covariance_manifest.json').read_bytes()).hexdigest()
    for name, digest in manifest['file_sha256'].items():
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == digest
    assert manifest['settings']['transaction_cost'] == .0005
    assert manifest['versions']['optimalportfolios'] == op.__version__


@pytest.mark.parametrize('kwargs', [
    {'tracking_error': 0.}, {'max_weight': 2.}, {'transaction_cost': -.01},
    {'implementation_lag': True}, {'momentum_span': 1}, {'solver': 'missing'},
])
def test_invalid_policy_settings(kwargs):
    """Reject invalid assumptions before downloading or optimising a portfolio."""
    with pytest.raises(ValueError):
        BacktestConfig(**kwargs)
