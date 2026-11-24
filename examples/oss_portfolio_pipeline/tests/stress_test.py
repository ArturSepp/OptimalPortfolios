"""Offline contracts for dated stress books, full covariance and QIS scenario valuation."""
from dataclasses import replace
import hashlib
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from examples.oss_portfolio_pipeline.tests import backtest_test as backtest_checks
from examples.oss_portfolio_pipeline.stress import build_stress_study, save_stress_report

pytestmark = pytest.mark.factor_integration


@pytest.fixture(scope='module')
def prepared():
    """Reuse the real three-fit synthetic FM/FL model and a cash-positive OP portfolio."""
    values = backtest_checks.prepared.__wrapped__()
    full = backtest_checks.completed.__wrapped__(values)
    inputs, risk, _ = values
    return inputs, risk, full


@pytest.fixture(scope='module')
def study(prepared):
    """Build the weekend-month-end stress book with all its independent/conditional paths."""
    return build_stress_study(*prepared)


def test_exact_dated_model_and_risk(prepared, study):
    """Preserve production covariance and reconcile risk to independent full quadratic forms."""
    inputs, risk, full = prepared
    date = pd.Timestamp('2020-02-28')
    assert study.model_date == pd.Timestamp('2020-02-29')
    assert study.portfolio.risk_date == study.portfolio.valuation_date == date
    weights = full.executed.targets.loc[date]
    covariance = risk.covariances[date]
    pd.testing.assert_frame_equal(study.portfolio.risk_model.covar[date], covariance)
    np.testing.assert_allclose(study.result.response_exposures / 1e6, weights, atol=1e-12)
    expected = np.sqrt(weights.to_numpy() @ covariance.to_numpy() @ weights.to_numpy())
    assert study.result.risk.annual_total_vol == pytest.approx(expected, abs=1e-12)
    active = (weights - inputs.benchmark_weights).to_numpy()
    assert study.portfolio.risk_model.compute_tre_at_date(
        inputs.benchmark_weights, weights, date) == pytest.approx(
            np.sqrt(active @ covariance.to_numpy() @ active), abs=1e-12)
    snapshot = risk.model.snapshots[study.model_date]
    pd.testing.assert_frame_equal(
        study.result.report_diagnostics['Production full residual covariance'],
        snapshot.residual_covariance)
    assert 'diag(residual R)' in ' '.join(study.report_config.notes)


def test_scenario_pnl_and_zero_reconciliation(prepared, study):
    """Recover funded ETF P&L independently and verify conditional anchors exactly."""
    _, _, full = prepared
    date = study.portfolio.risk_date
    model = study.portfolio.risk_model
    weights = full.executed.targets.loc[date]
    for mode in ('independent', 'conditional'):
        shocks = study.result.valuations[mode].factor_log_shocks
        responses = shocks.to_numpy() @ model.factor_loadings[date].to_numpy().T
        expected = np.expm1(responses) @ weights.to_numpy()
        np.testing.assert_allclose(study.result.summaries[mode].portfolio_return, expected,
                                   atol=1e-12)
        anchors = study.scenarios.expanded_anchors(model, date)
        np.testing.assert_allclose(shocks.where(anchors.notna()), anchors, equal_nan=True)
    zero = pd.DataFrame(0., index=['zero'], columns=model.factor_loadings[date].columns)
    valuation = study.portfolio.evaluate(zero)
    assert valuation.portfolio_pnl.iloc[0] == pytest.approx(0., abs=1e-10)
    assert valuation.portfolio_mtm.iloc[0] == pytest.approx(1e6, abs=1e-7)
    audit = study.result.report_diagnostics['OP target snapshot']
    np.testing.assert_allclose(audit.adjusted_price_units * audit.adjusted_price,
                               audit.target_mtm_usd, atol=1e-10)


def test_last_target_wins_over_later_available_model(prepared):
    """A later fitted covariance is never substituted for the last selected target date."""
    inputs, risk, full = prepared
    executed = replace(full.executed, targets=full.executed.targets.iloc[:-1])
    earlier = replace(full, executed=executed)
    result = build_stress_study(inputs, risk, earlier)
    assert result.portfolio.risk_date == pd.Timestamp('2020-01-31')
    assert result.model_date == pd.Timestamp('2020-01-31')
    assert result.history.index[-1] == pd.Timestamp('2020-01-31')


def test_history_is_causal_and_preserves_weekend_month(prepared, study):
    """Changing all post-snapshot prices cannot alter scenarios or completed monthly history."""
    inputs, risk, full = prepared
    date = study.portfolio.risk_date
    prices, factors = inputs.prices.copy(), inputs.factor_navs.copy()
    prices.loc[prices.index > date] *= 3.
    factors.loc[factors.index > date] *= 5.
    changed = build_stress_study(replace(inputs, prices=prices, factor_navs=factors), risk, full)
    pd.testing.assert_frame_equal(changed.history, study.history)
    pd.testing.assert_frame_equal(changed.result.summaries['conditional'],
                                  study.result.summaries['conditional'])
    assert study.history.index[-1] == date
    assert study.history.index.to_period('M').is_unique
    assert study.result.historical_coverage.status.eq('eligible').all()
    sessions = inputs.factor_navs.groupby(inputs.factor_navs.index.to_period('M')).tail(1)
    sessions = sessions.loc[:date]
    expected = np.log(sessions.to_numpy()[1:] / sessions.to_numpy()[:-1])
    np.testing.assert_allclose(study.history, expected, atol=1e-12)


def test_notional_scales_dollars_only(prepared, study):
    """A second illustrative capital base changes dollar P&L but neither returns nor risk."""
    larger = build_stress_study(*prepared, notional=2e6)
    np.testing.assert_allclose(larger.result.summaries['requested'].portfolio_pnl,
                               2 * study.result.summaries['requested'].portfolio_pnl, atol=1e-8)
    np.testing.assert_allclose(larger.result.risk, study.result.risk, atol=1e-12)


@pytest.mark.parametrize('notional', [0., -1., np.inf, True])
def test_invalid_notional(prepared, notional):
    """Reject invalid capital bases before invoking QIS analytics."""
    with pytest.raises(ValueError, match='notional'):
        build_stress_study(*prepared, notional=notional)


def test_saved_stress_report_provenance(study, tmp_path):
    """Render the native QIS report and retain complete numerical tables and input lineage."""
    for name in ('backtest_manifest.json', 'covariance_manifest.json'):
        (tmp_path / name).write_text('{}')
    before = plt.get_fignums()
    artifacts = save_stress_report(study, tmp_path)
    report = json.loads(artifacts.manifest_path.read_text())
    assert report['page_count'] >= 13
    assert artifacts.pdf_path.read_bytes().startswith(b'%PDF-')
    assert artifacts.workbook_path is None
    assert 'Production full residual covariance' in artifacts.table_paths
    for name, checksum in report['hashes'].items():
        digest = hashlib.sha256((artifacts.pdf_path.parent / name).read_bytes()).hexdigest()
        assert digest == checksum
    provenance = json.loads((artifacts.pdf_path.parent / 'pipeline_manifest.json').read_text())
    assert provenance['decision_date'] == '2020-02-28'
    assert provenance['model_date'] == '2020-02-29'
    for name, checksum in provenance['source_sha256'].items():
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == checksum
    assert plt.get_fignums() == before


def test_source_bundle_required(study, tmp_path):
    """A report must be linked to saved portfolio and covariance provenance."""
    with pytest.raises(ValueError, match='Save the covariance'):
        save_stress_report(study, tmp_path)
