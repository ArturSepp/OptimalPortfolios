"""L4 integration contracts using existing FactorLasso/MATF and QIS calculations."""
import importlib

import numpy as np
import pandas as pd
import pytest

pytestmark = pytest.mark.factor_integration


@pytest.fixture(scope='module')
def workflow(root):
    """Load the repository-only canonical workflow; installed-wheel runs skip via root."""
    return importlib.import_module('examples.docs.factor_diagnostics_workflow')


@pytest.fixture(scope='module')
def result(workflow):
    """Fit the deterministic three-period workflow once for the integration contracts."""
    return workflow.run_workflow()


@pytest.fixture(scope='module')
def coverage(workflow, result):
    """Run the prespecified full-panel Gaussian AR(1) coverage experiment."""
    return workflow.coverage_experiment(result)


def test_independent_projection_risk_and_accounting_references(workflow, result):
    """Weighted QR and matrix/wealth identities verify production outputs independently."""
    metrics = workflow.assert_references(result)
    assert metrics['max_accounting_gap'] < 1e-12
    assert metrics['max_projection_error'] < 1e-12


def test_joint_alpha_covariance_is_not_residual_return_risk(result):
    """Estimation covariance and idiosyncratic return variance have distinct estimands."""
    m, u = result['opportunity'], result['uncertainty']
    assert not np.allclose(np.diag(u.covariance), m.residual_variances)
    assert not u.method['beta_selection_uncertainty']
    assert u.method['scope'] == 'fixed_model_initialized_EWMA_mean'
    assert result['intervals'].scope.eq('pointwise').all()
    assert result['intervals'].status.eq('approximate_conditional').all()
    assert 'fixed-book conditional alpha' in result['fixed_interval']['scope']
    off_diagonal = u.covariance.to_numpy() - np.diag(np.diag(u.covariance))
    assert np.linalg.norm(off_diagonal) > 1e-8


def test_units_are_scaled_once_and_covariance_twice(workflow, result):
    """Annual alpha scales by 12 and its estimation covariance by 144."""
    raw = workflow.fl.estimate_alpha_uncertainty(
        result['residuals'], workflow.ALPHA_SPAN,
        calendar=np.arange(len(result['residuals'])), bandwidth=workflow.BANDWIDTH, scale=1.)
    np.testing.assert_allclose(result['uncertainty'].estimates, 12 * raw.estimates, atol=1e-14)
    np.testing.assert_allclose(result['uncertainty'].covariance, 144 * raw.covariance, atol=1e-14)


def test_future_observations_do_not_change_training_outputs(workflow, result):
    """A held-out shock changes evaluation but never betas, alpha, risk or intervals."""
    x, y = workflow.synthetic_returns()
    y.loc[y.index > result['asof'], 'Asset 1'] += .02
    shocked = workflow.run_workflow(x, y)
    for name in ['betas', 'alpha', 'intervals', 'bootstrap_intervals']:
        pd.testing.assert_frame_equal(result[name], shocked[name])
    pd.testing.assert_series_equal(result['risk'], shocked['risk'])
    pd.testing.assert_frame_equal(result['uncertainty'].covariance,
                                  shocked['uncertainty'].covariance)
    np.testing.assert_array_equal(result['bootstrap']['errors'], shocked['bootstrap']['errors'])
    assert not np.allclose(result['portfolio'].portfolio.nav, shocked['portfolio'].portfolio.nav)


def test_provenance_and_execution_state_are_chronological(result):
    """There are disjoint beta, alpha/risk and untouched evaluation blocks."""
    p = result['provenance']
    assert p.observations.tolist() == [72, 120, 48]
    assert p.loc[0, 'last'] < p.loc[1, 'first']
    assert p.loc[1, 'last'] == result['asof'] < p.loc[2, 'first']
    assert result['opportunity'].risk_model.dates.tolist() == [result['asof']]
    for path in [result['portfolio'], result['reference']]:
        assert path.states[0].cash == path.states[0].nav == 100
        assert path.states[0].units.sum() == 0
        assert path.states[1].units.abs().sum() > 0
        assert path.events.execution.iloc[0] == p.loc[2, 'first']
        assert path.events.status.eq('scheduled').all()
    assert result['attribution'].index[0] == p.loc[2, 'first']


def test_held_out_residual_diagnostics_are_reported_without_forcing_pass(result):
    """An assumption check is retained rather than used to repair or relabel residuals."""
    diagnostics = result['diagnostics']
    assert diagnostics.n_obs.tolist() == [120, 48]
    assert diagnostics.passes.isin([True, False]).all()
    assert (diagnostics.mean_abs_offdiag > 0).all()


def test_bootstrap_keeps_joint_targets_and_unvalidated_scope(result):
    """Shared serial calendar multipliers preserve the alpha decomposition in every draw."""
    b = result['bootstrap']
    assert b['method'] == 'residual_dependent_wild_bootstrap'
    assert b['bandwidth'] == 6 and b['draws'] == 400
    assert b['status'] == 'bootstrap_approximation_unvalidated'
    assert result['bootstrap_intervals'].status.eq(b['status']).all()
    assert result['bootstrap_intervals'].scope.eq('pointwise').all()
    np.testing.assert_allclose(b['errors'][:, 0], b['errors'][:, 1] + b['errors'][:, 2], atol=1e-13)
    np.testing.assert_allclose(b['covariance'], result['target_covariance'], atol=1e-14)


def test_known_shape_intervals_have_prespecified_conditional_coverage(coverage):
    """The complete Gaussian panel experiment tests coverage, not only array shapes."""
    assert len(coverage) == 6
    assert set(coverage.phi) == {-.3, .5}
    assert coverage.draws.eq(3000).all()
    assert ((coverage.known_shape_coverage - .95).abs() <= coverage.binomial_tolerance).all()
    assert coverage.scope.eq('pointwise_fixed_model_true_Gaussian_AR1_shape').all()
    positive = coverage[coverage.phi == .5]
    assert (positive.known_shape_critical > positive.normal_critical).all()
    assert (positive.normal_HAC_coverage < positive.known_shape_coverage).all()


def test_report_uses_canonical_qis_panels(workflow, result):
    """Render a report without saving generated state beside source files."""
    import matplotlib.pyplot as plt

    fig = workflow.render_report(result)
    assert len(fig.axes) == 3
    assert 'conditional' in fig.axes[0].get_title()
    assert 'Untouched' in fig.axes[1].get_title()
    assert 'accounting' in fig.axes[2].get_title()
    plt.close(fig)
