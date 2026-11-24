"""Offline contracts for implementable layer portfolios and QIS factorial attribution."""
from dataclasses import asdict, replace
import hashlib
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import qis

from examples.oss_portfolio_pipeline.tests import covariance_test as covariance_checks
from examples.oss_portfolio_pipeline.attribution import (
    BRIDGE_COLUMNS, LAYER_FIELDS, MOMENTUM_FEATURE, TE_FEATURE,
    run_attribution, save_attribution,
)
from examples.oss_portfolio_pipeline.backtest import BacktestConfig, run_backtest
from examples.oss_portfolio_pipeline.covariance import estimate_covariances
from examples.oss_portfolio_pipeline.pipeline import build_inputs

pytestmark = pytest.mark.factor_integration


@pytest.fixture(scope='module')
def prepared():
    """Extend the seeded inputs to four full months, enough for QIS HAC regressions."""
    inputs, config = covariance_checks.prepared.__wrapped__()
    dates = pd.bdate_range('2020-03-02', '2020-05-04')
    rng = np.random.default_rng(734)
    returns = rng.normal(.0002, .004, (len(dates), inputs.source_prices.shape[1]))
    extra = pd.DataFrame(np.exp(returns.cumsum(axis=0)), index=dates,
                         columns=inputs.source_prices.columns) * inputs.source_prices.iloc[-1]
    extra['BIL'] = inputs.source_prices.BIL.iloc[-1] * np.exp(.00008 * np.arange(1, len(dates) + 1))
    config = replace(config, end='2020-05-05')
    inputs = build_inputs(pd.concat([inputs.source_prices, extra]), config)
    risk = estimate_covariances(inputs, config, solver='CLARABEL')
    full = run_backtest(inputs, risk, config, BacktestConfig(solver='CLARABEL'))
    return inputs, risk, config, full


@pytest.fixture(scope='module')
def study(prepared):
    """Run all four real OP configurations on the shared synthetic FM/FL risk model."""
    inputs, risk, config, full = prepared
    return run_attribution(inputs, risk, config, full, include_features=True)


def test_risk_and_signal_portfolios(prepared, study):
    """Verify risk budgets and minimisation, and independently rank the nine signal holdings."""
    inputs, risk, _, full = prepared
    base = study.scenarios[frozenset()]
    assert base.risk_audit.passed.all()
    assert base.risk_diagnostics.accepted.all() and base.risk_diagnostics.compliant.all()
    for date, target in base.risk_executed.targets.iterrows():
        active = (target - inputs.benchmark_weights).to_numpy()
        variance = active @ risk.covariances[date].to_numpy() @ active
        assert np.sqrt(variance) <= full.settings.tracking_error + 1e-6
        scores = full.alpha_scores.loc[date].drop('BIL')
        selected = scores.sort_values(ascending=False, kind='stable').iloc[:9].index
        expected = pd.Series(0., index=inputs.universe.index)
        expected.loc[selected] = 1 / 9
        pd.testing.assert_series_equal(base.signal.input_weights.loc[date], expected,
                                       check_names=False)
    date = full.executed.targets.index[0]
    covariance = risk.covariances[date].to_numpy()
    target = base.risk_executed.targets.loc[date].to_numpy()
    benchmark = inputs.benchmark_weights.to_numpy()
    assert target @ covariance @ target <= benchmark @ covariance @ benchmark + 1e-9


def test_qis_bridge_and_independent_ols(study):
    """Recover the net log returns and full-model regression from a separate least-squares fit."""
    for run in study.scenarios.values():
        attribution = run.attribution
        components = attribution.component_returns
        np.testing.assert_allclose(components[list(BRIDGE_COLUMNS)].sum(axis=1),
                                   attribution.periodic_returns['Full Model Net'], atol=1e-12)
        x = attribution.periodic_returns['Benchmark'].to_numpy()
        y = attribution.periodic_returns['Full Model'].to_numpy()
        alpha, beta = np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), y, rcond=None)[0]
        regression = attribution.regression_table
        assert regression.loc['Full Model', qis.PerfStat.ALPHA_AN.to_str()] == pytest.approx(
            12 * alpha)
        assert regression.loc['Full Model', qis.PerfStat.BETA.to_str()] == pytest.approx(beta)
        assert run.identity_errors.max() < 1e-12
        np.testing.assert_allclose(components['Trading Cost Drag'],
                                   attribution.periodic_returns['Full Model Net'] - y, atol=1e-12)


def test_complete_coalitions_and_controlled_inputs(prepared, study):
    """Change only declared settings and preserve independent layer controls."""
    _, _, _, full = prepared
    baseline, momentum, te, joint = (
        frozenset(), frozenset({MOMENTUM_FEATURE}), frozenset({TE_FEATURE}),
        frozenset({MOMENTUM_FEATURE, TE_FEATURE}))
    assert set(study.scenarios) == {baseline, momentum, te, joint}
    for key, run in study.scenarios.items():
        expected = asdict(full.settings)
        if MOMENTUM_FEATURE in key:
            expected['momentum_span'] *= 2
        if TE_FEATURE in key:
            expected['tracking_error'] *= 1.5
        assert asdict(run.full.settings) == expected
        pd.testing.assert_series_equal(run.navs.benchmark_nav, full.gross_benchmark.nav)
        assert run.full.executed.targets.index.equals(full.executed.targets.index)
    pd.testing.assert_frame_equal(study.scenarios[baseline].risk_executed.targets,
                                  study.scenarios[momentum].risk_executed.targets)
    pd.testing.assert_frame_equal(study.scenarios[baseline].signal.input_weights,
                                  study.scenarios[te].signal.input_weights)
    assert study.scenarios[baseline].full is full


def test_shapley_paths_against_two_orderings(study):
    """Check QIS effects with the explicit average of the two feature-insertion orders."""
    scenarios = study.features.scenario_layer_navs
    for feature, other in ((MOMENTUM_FEATURE, TE_FEATURE), (TE_FEATURE, MOMENTUM_FEATURE)):
        for field in LAYER_FIELDS[1:]:
            baseline = np.log(getattr(scenarios[frozenset()], field)).diff()
            single = np.log(getattr(scenarios[frozenset({feature})], field)).diff()
            other_path = np.log(getattr(scenarios[frozenset({other})], field)).diff()
            both = np.log(getattr(scenarios[frozenset({feature, other})], field)).diff()
            expected = .5 * ((single - baseline) + (both - other_path))
            actual = np.log(getattr(study.features.shapley_feature_navs[feature], field)).diff()
            np.testing.assert_allclose(actual.iloc[1:], expected.iloc[1:], atol=1e-12)
    assert study.features.identity_errors.max() < 1e-10
    assert len(study.features.factorial_effect_navs) == 3


def test_qis_requires_complete_features(study):
    """A partial experiment is never silently attributed as a full two-feature study."""
    incomplete = {key: run.navs for key, run in study.scenarios.items() if len(key) < 2}
    with pytest.raises(ValueError, match='complete factorial'):
        qis.compute_model_feature_alpha_beta_attribution(incomplete, freq='ME')


def test_short_sample_rejected_before_layer_solves(prepared):
    """A short public run can omit attribution rather than infer alpha from too few returns."""
    inputs, risk, config, full = prepared
    short = replace(full, navs=full.navs.loc[:'2020-02-28'])
    with pytest.raises(ValueError, match='three completed monthly returns'):
        run_attribution(inputs, risk, config, short)


def test_attribution_pdf_and_provenance(study, tmp_path):
    """Persist all actual layer paths and identities, with a rendered three-page report."""
    source = tmp_path / 'backtest_manifest.json'
    source.write_text('{"test_bundle": true}', encoding='utf-8')
    before = plt.get_fignums()
    destination = save_attribution(study, tmp_path)
    manifest = json.loads((destination / 'attribution_manifest.json').read_text())
    assert len(manifest['scenarios']) == 4 and len(manifest['pages']) == 3
    assert manifest['backtest_manifest_sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
    for name, checksum in manifest['file_sha256'].items():
        assert hashlib.sha256((destination / name).read_bytes()).hexdigest() == checksum
    assert (destination / 'model_attribution.pdf').read_bytes().startswith(b'%PDF-')
    assert plt.get_fignums() == before


def test_baseline_only_tables(prepared, tmp_path):
    """The default layer report needs one run; feature experiments remain explicit opt-in."""
    inputs, risk, config, full = prepared
    baseline = run_attribution(inputs, risk, config, full)
    assert set(baseline.scenarios) == {frozenset()} and baseline.features is None
    (tmp_path / 'backtest_manifest.json').write_text('{}')
    destination = save_attribution(baseline, tmp_path, render=False)
    manifest = json.loads((destination / 'attribution_manifest.json').read_text())
    assert manifest['pages'] == []
    assert not (destination / 'model_attribution.pdf').exists()
