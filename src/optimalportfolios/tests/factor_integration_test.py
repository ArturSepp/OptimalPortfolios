"""Composed FM-to-OP handoff checks; imports remain safe in OP-only installations."""
import numpy as np
import pandas as pd
import pytest
import qis

pytestmark = pytest.mark.factor_integration


def test_factorlasso_cv_stays_inside_each_factory():
    """Reuse FactorLasso's expanding estimator CV on only the purged training returns."""
    from factorlasso import LassoModel, LassoModelCV

    from optimalportfolios.optimization.tests.chronological_selection_test import execute, fixture

    cutoffs = []

    def fit(training, configuration):
        """Fit the existing estimator CV, without portfolio-level leakage or new CV logic."""
        returns = qis.to_returns(training, is_log_returns=False).dropna()
        cv = LassoModelCV(lambdas=(.0001, .001), n_splits=1,
                          base_model=LassoModel(warmup_period=1))
        cv.fit(x=returns[['A']], y=returns[['B']])
        assert cv.best_lambda_ in (.0001, .001)
        cutoffs.append(training.index[-1])

        def decide(history, state):
            """Retain cash; this contract tests nested estimator ownership."""
            return None
        return decide

    result = execute(fit=fit, candidates={'cv': {'covariance': 'factorlasso', 'alpha': 0}})
    assert cutoffs == fixture()[0].index[[7, 17, 29]].tolist()
    assert result.outer_score == 0


@pytest.fixture(scope='module')
def fitted_snapshot():
    """Fit mixed native coordinates and empirical residual risk on a deterministic panel."""
    from factorlasso import LassoModel, LassoModelType
    from qis.perfstats.config import ReturnTypes
    from factormodels import FactorCovarEstimator

    rng = np.random.default_rng(5102026)
    dates = pd.date_range('2020-01-31', periods=72, freq='ME')
    factor_returns = rng.normal(size=(len(dates), 2)) * [0.03, 0.2]
    prices = pd.DataFrame({'growth': np.exp(factor_returns[:, 0].cumsum()),
                           'yield_change': 100. + factor_returns[:, 1].cumsum()}, index=dates)
    loadings = np.array([[0.8, -0.03], [0.1, -0.08], [0.5, -0.01], [0., 0.]])
    common = rng.normal(0., 0.01, (len(dates), 1))
    returns = pd.DataFrame(factor_returns @ loadings.T + common * [1., .5, .8, .005]
                           + rng.normal(size=(len(dates), 4)) * [.006, .004, .005, .0001],
                           index=dates, columns=['A', 'B', 'C', 'Cash'])
    estimator = FactorCovarEstimator(
        lasso_model=LassoModel(model_type=LassoModelType.LASSO, span=24,
                               warmup_period=6, reg_lambda=1e-6, solver='CLARABEL'),
        factor_returns_freq='ME', factor_covar_span=24, rebalancing_freq='ME',
        factor_return_types={'growth': ReturnTypes.LOG, 'yield_change': ReturnTypes.DIFFERENCE},
        residual_type='empirical', residual_covar_freq='ME', residual_covar_span=24,
        residual_corr_weight=.6, residual_var_weight=.7)
    return estimator.fit_current(prices, {'ME': returns}, cutoff=dates[-1])


def _manual_risk_inputs(snapshot, diagonal_reference=False):
    """Independently assemble full selected residual risk and a plain QIS interchange view."""
    data, date = snapshot, snapshot.cutoff
    b, f = data.loadings, data.factor_covariance
    variances = data.fit_statistics['residual_var'].to_numpy()
    corr = data.residual_correlation.to_numpy()
    residual = .7 * (.6 * corr + .4 * np.eye(len(variances))) * np.sqrt(
        np.outer(variances, variances))
    residual = pd.DataFrame(residual, index=b.index, columns=b.index)
    covariance = b @ f @ b.T + residual
    kwargs = {}
    if diagonal_reference:
        kwargs = {'factor_covar': {date: f},
                  'residual_vars': {date: pd.Series(np.diag(residual), index=b.index)}}
    risk = qis.RiskModel(covar={date: covariance}, factor_loadings={date: b},
                         factor_return_types=snapshot.factor_return_types, **kwargs)
    return risk, residual


def _assert_report_equal(left, right):
    """Compare every nested report section, including prepared diagnostic and risk tables."""
    assert left.keys() == right.keys()
    for name in left:
        if isinstance(left[name], dict):
            _assert_report_equal(left[name], right[name])
        else:
            pd.testing.assert_frame_equal(left[name], right[name], check_exact=False,
                                           atol=1e-13, rtol=1e-11)


@pytest.mark.parametrize('diagonal_reference', [False, True])
def test_fitted_tf_context_matches_manual_risk_and_complete_report(
        fitted_snapshot, diagonal_reference):
    """A real FM fit and independent supplied matrices produce identical OP reporting data."""
    from qis.perfstats.config import ReturnTypes
    from factormodels import prepare_portfolio_context, prepare_snapshot_diagnostics
    from optimalportfolios import PortfolioOptimisationResult

    snapshot = fitted_snapshot
    risk, residual = _manual_risk_inputs(snapshot, diagonal_reference)
    b, f = snapshot.loadings, snapshot.factor_covariance
    ids, date = b.index, snapshot.cutoff
    volatility = pd.DataFrame({
        'total_vol': np.sqrt(np.diag(risk.covar[date])),
        'sys_vol': np.sqrt(np.diag(b @ f @ b.T)),
        'resid_vol': np.sqrt(np.diag(residual)),
    }, index=ids)
    prepared = prepare_snapshot_diagnostics(snapshot, alpha_span=120)
    diagnostics = prepared.table
    diagnostics.loc[:, volatility.columns] = volatility
    weights = pd.Series([.4, .2, .3, .1], index=ids, name='candidate')
    benchmark = pd.Series([.3, .3, .25, .15], index=ids, name='benchmark')
    common = dict(weights=weights, benchmark_weights=benchmark, optimisation_date=date,
                  current_weights=benchmark, expected_return=pd.Series(.04, index=ids),
                  group_attributions={'class': pd.Series(['risk'] * 3 + ['cash'], index=ids)})
    adapted = PortfolioOptimisationResult(**common, **prepare_portfolio_context(
        snapshot, diagnostics=prepared, diagonal_reference=diagonal_reference))
    manual = PortfolioOptimisationResult(**common, risk_model=risk, risk_model_date=date,
        decomposition_mode='diagonal_reference' if diagonal_reference else 'exact',
        asset_diagnostics=diagnostics, asset_volatilities=volatility)
    _assert_report_equal(adapted.report(), manual.report())
    pd.testing.assert_frame_equal(adapted.get_asset_betas_table(), manual.get_asset_betas_table(),
                                   check_exact=False, atol=1e-13, rtol=1e-11)
    assert adapted.risk_model.factor_return_types['yield_change'] is ReturnTypes.DIFFERENCE
    active = weights - benchmark
    assert adapted.compute_tracking_error() ** 2 == pytest.approx(
        active @ risk.covar[date] @ active)
    assert adapted.compute_portfolio_variance() == pytest.approx(
        weights @ risk.covar[date] @ weights)
    diagonal = b @ f @ b.T + np.diag(np.diag(residual))
    assert not np.isclose(weights @ diagonal @ weights, weights @ risk.covar[date] @ weights)


@pytest.mark.parametrize('mode', ['none', 'cluster_variance', 'asset_variance'])
def test_prepared_cluster_scores_match_supplied_cluster_mapping(mode):
    """Upstream preparation and unit-stability pooling retain OP's generic score formula."""
    from factormodels import prepare_cluster_scores
    from optimalportfolios.alphas.signals.utils import score_within_clusters

    dates = pd.date_range('2024-01-31', periods=3, freq='ME')
    names = [f't{i}' for i in range(8)]
    raw = pd.DataFrame([np.arange(8.) + i for i in range(3)], index=dates, columns=names)
    clusters = {dates[0]: pd.Series(['ME:1'] * 5 + ['QE:2'] * 3, index=names),
                dates[2]: pd.Series(['ME:1'] * 4 + ['QE:2'] * 4, index=names)}
    expected = score_within_clusters(raw, clusters)
    actual = prepare_cluster_scores(raw, clusters, pooling_type=mode,
                                    stability_weights=pd.DataFrame(1., index=dates, columns=names))
    pd.testing.assert_frame_equal(actual, expected, check_exact=True)
    assert actual.to_numpy().tobytes() == expected.to_numpy().tobytes()
    if mode == 'none':
        pd.testing.assert_frame_equal(
            prepare_cluster_scores(raw, {}), score_within_clusters(raw, {}), check_exact=True)


@pytest.mark.parametrize('objective', ['upper', 'orthogonal_upper'])
def test_prepared_opportunity_matches_manual_arrays_and_constraint_audit(objective):
    """Statistical preparation upstream preserves constrained solutions and hard feasibility."""
    from factorlasso import ResidualCorrelationData
    from factormodels import FactorOpportunityModel, snapshot_from_moments
    from optimalportfolios.optimization.opportunity import OpportunityModel, solve_opportunity

    ids, date = pd.Index(['a', 'b', 'c', 'd']), pd.Timestamp('2026-06-30')
    b = pd.DataFrame([.5, 1., 1., 1.5], index=ids, columns=['factor'])
    f = pd.DataFrame([[.09]], index=b.columns, columns=b.columns)
    alpha = pd.Series([-.02, .01, .03, .015], index=ids)
    correlation = pd.DataFrame(.4 * np.ones((4, 4)) + .6 * np.eye(4), index=ids, columns=ids)
    prepared = ResidualCorrelationData(correlation,
        pd.DataFrame([[0.] * 4], index=[date], columns=ids), pd.DataFrame(index=ids),
        'ME', 24., date, date)
    upstream = FactorOpportunityModel.from_snapshot(snapshot_from_moments(
        b, f, pd.Series(.04, index=ids),
        date=date, residual_correlation=prepared, residual_type='empirical',
        residual_corr_weight=.5), alpha)
    residual = .04 * (.5 * correlation + .5 * np.eye(4))
    design = np.column_stack([np.ones(4), b])
    projection = np.eye(4) - design @ np.linalg.pinv(design)
    adapted = OpportunityModel(**upstream.optimisation_inputs())
    manual = OpportunityModel(covariance=b @ f @ b.T + residual, alpha=alpha,
                              factor_loadings=b, factor_covariance=f, residual_covariance=residual,
                              orthogonal_alpha=pd.Series(projection @ alpha, index=ids), date=date)
    reference = pd.Series(.25, index=ids)
    options = dict(total_te=.03, factor_te=.005, objective=objective)
    left, right = solve_opportunity(adapted, reference, **options), solve_opportunity(
        manual, reference, **options)
    assert left.accepted and right.accepted
    assert left.status == right.status
    np.testing.assert_allclose(left.weights, right.weights, atol=1e-8)
    assert left.weights.sum() == pytest.approx(1., abs=1e-8)
    assert left.weights.min() >= -1e-8
    delta = left.weights - reference
    assert delta @ manual.covariance @ delta <= .03 ** 2 + 1e-9
    assert (b.T @ delta) @ f @ (b.T @ delta) <= .005 ** 2 + 1e-9
    assert left.metrics['residual_te'] ** 2 == pytest.approx(delta @ residual @ delta)


def test_tf_covariance_preserves_execution_selection_and_feasibility(fitted_snapshot):
    """Prepared upstream and manual risk give identical audited execution holdings."""
    from dataclasses import replace
    from factormodels import prepare_portfolio_context
    from optimalportfolios.execution import (
        ExecutionProblem, ExecutionRules, RankedConfig, audit_execution, wrapper_ranked_execution,
    )

    snapshot = fitted_snapshot
    risk, _ = _manual_risk_inputs(snapshot)
    prepared = prepare_portfolio_context(snapshot)['risk_model'].covar[snapshot.cutoff]
    current = np.array([.30, .25, .25, .20])
    target = np.array([.40, .15, .30, .15])
    lower, upper = np.minimum(current, target), np.maximum(current, target)
    lower[-1], upper[-1] = 0., 1.
    manual = ExecutionProblem(
        names=('A', 'B', 'C', 'Cash'), cash=3, current=current, base=current,
        target=target, lower=lower, upper=upper, full_lower=np.zeros(4),
        full_upper=np.ones(4), groups=np.zeros((0, 4)), group_lower=np.zeros(0),
        group_upper=np.zeros(0), mandatory=np.zeros(4, bool), categories=('',) * 4,
        eligible=np.ones(4, bool), frozen=np.zeros(4, bool),
        rules=ExecutionRules(ticket_cap=2), covariance=risk.covar[snapshot.cutoff].to_numpy())
    adapted = replace(manual, covariance=prepared.loc[list(manual.names), list(manual.names)])
    config = RankedConfig(qp_solver='CLARABEL')
    left, right = wrapper_ranked_execution(adapted, config), wrapper_ranked_execution(manual, config)
    assert audit_execution(adapted, left.weights)['passed']
    assert audit_execution(manual, right.weights)['passed']
    assert left.status == right.status
    np.testing.assert_allclose(left.weights, right.weights, atol=1e-8, rtol=0.)
    np.testing.assert_array_equal(abs(left.weights-current) > 1e-8,
                                  abs(right.weights-current) > 1e-8)
    assert left.weights.sum() == pytest.approx(1., abs=1e-8)
