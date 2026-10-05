"""Conditional alpha uncertainty propagated through canonical OP GLS accounting.

FactorLasso owns covariance estimation, Gaussian scenarios and quadratic-region
calibration. Here B, D and F are fixed; no risk-model or beta CI is inferred.
Scenario percentiles describe sensitivity, not posterior or calibrated coverage.
"""
import numpy as np
import pandas as pd
from factorlasso import (
    gaussian_quadratic_summary, sample_gaussian_estimates, linear_confidence_intervals,
    quadratic_confidence_summary,
)

from optimalportfolios.matf import sharpe_accounting as sa
from optimalportfolios.matf.opportunity import alpha_dispersion


def _covariance(model, covariance):
    """Require exact labelled covariance axes before downstream FactorLasso validation."""
    ids = model.betas.index
    if (not isinstance(covariance, pd.DataFrame) or not covariance.index.equals(ids)
            or not covariance.columns.equals(ids)):
        raise ValueError('alpha covariance must match the exact model asset axis')
    return covariance.to_numpy(dtype=float)


def _projectors(model):
    """Use the canonical rank-safe projector for every dispersion definition."""
    ids = model.betas.index
    one = pd.DataFrame({'funding': np.ones(len(ids))}, index=ids)
    joint = pd.concat([one, model.betas], axis=1, sort=False)
    return dict(A=np.eye(len(ids)), J=sa.gls_projector(one, model.residual_variances),
                S_h2=sa.gls_projector(model.betas, model.residual_variances),
                K=sa.gls_projector(joint, model.residual_variances))


def _metric(projector, variances):
    """Construct the quadratic corresponding to an existing residual projection."""
    value = projector.T @ (projector / np.asarray(variances)[:, None])
    return (value+value.T)/2


def _interval_table(mean, covariance, index, confidence, supplied=None, simultaneous=False):
    """Consume calibrated endpoints or delegate default linear inference to FactorLasso."""
    if supplied is None:
        return pd.DataFrame(linear_confidence_intervals(
            mean, covariance, confidence=confidence, simultaneous=simultaneous), index=index)
    required = {'estimate', 'standard_error', 'lower', 'upper', 'confidence',
                'interval_method', 'scope', 'status'}
    if (not isinstance(supplied, pd.DataFrame) or not supplied.index.equals(index)
            or not required.issubset(supplied.columns)):
        raise ValueError('supplied intervals require exact target labels and inference metadata')
    if (not np.allclose(supplied.estimate, mean, rtol=1e-10, atol=1e-13)
            or not np.allclose(supplied.standard_error**2, np.diag(covariance),
                               rtol=1e-9, atol=1e-14)
            or not np.allclose(supplied.confidence, confidence, rtol=0, atol=1e-14)
            or np.any(supplied.standard_error < 0)):
        raise ValueError('supplied intervals do not describe the current estimates/covariance')
    available = supplied.lower.notna() & supplied.upper.notna()
    if (not supplied.lower.notna().equals(supplied.upper.notna())
            or not np.isfinite(supplied.loc[available, ['lower', 'upper']]).all().all()
            or (supplied.loc[available, 'lower'] > supplied.loc[available, 'upper']).any()):
        raise ValueError('interval endpoints must be ordered finite pairs or unavailable pairs')
    if simultaneous and not supplied.scope.eq('simultaneous').all():
        raise ValueError('simultaneous reporting requires family-calibrated intervals')
    return supplied.copy()


def alpha_uncertainty_metrics(model, covariance, *, confidence=.95, draws=1000, seed=0,
                              scenarios=None, intervals=None, quadratic_method='spectral',
                              bootstrap_errors=None):
    """Return conditional metric bounds, contributor intervals and joint scenario stability.

    Bounds use conservative Gaussian regions with plug-in HAC covariance. Their
    finite-sample coverage for fitted residuals is not asserted. Raw point
    estimates reconcile to alpha_dispersion. Signed noise corrections target the
    initialized EWMA mean conditional on fixed B/D, not a drifting endpoint.
    """
    v = _covariance(model, covariance)
    intervals = {} if intervals is None else intervals
    if (not isinstance(intervals, dict)
            or set(intervals)-{'alpha', 'projected', 'projected_simultaneous'}):
        raise ValueError('intervals must map the named alpha/projected target families')
    a, d = model.alpha.to_numpy(), model.residual_variances.to_numpy()
    generated = sample_gaussian_estimates(a, v, draws=draws if scenarios is None else 1, seed=seed)
    scenarios = generated if scenarios is None else np.asarray(scenarios, dtype=float)
    if scenarios.shape != (draws, len(a)) or not np.isfinite(scenarios).all():
        raise ValueError('supplied scenarios must be finite with shape (draws, assets)')
    projectors = _projectors(model)
    points, detail = alpha_dispersion(model)
    summary, values = [], {}
    for name, p in projectors.items():
        matrix = _metric(p, d)
        row = quadratic_confidence_summary(a, v, matrix, confidence=confidence,
                                             method=quadratic_method, errors=bootstrap_errors)
        if not np.isclose(row['observed'], points[name], rtol=1e-10, atol=1e-12):
            raise ValueError('canonical dispersion reconciliation failed')
        row['metric'] = name
        row['per_asset'] = row['observed']/len(a)
        row['noise_per_asset'] = row['noise']/len(a)
        row['adjusted_per_asset'] = row['noise_adjusted']/len(a)
        for key in ['observed', 'lower', 'upper']:
            row[key+'_rms'] = np.sqrt(max(0., row[key])/len(a))
        summary.append(row)
        projected_draws = scenarios @ p.T
        values[name] = np.sum(projected_draws**2/d, axis=1)
    p = projectors['K']
    h = p @ a
    vh = p @ v @ p.T
    vh = (vh+vh.T)/2
    raw_intervals = _interval_table(a, v, model.alpha.index, confidence, intervals.get('alpha'))
    projected_intervals = _interval_table(h, vh, model.alpha.index, confidence,
                                          intervals.get('projected'))
    simultaneous_intervals = _interval_table(h, vh, model.alpha.index, confidence,
                                             intervals.get('projected_simultaneous'), True)
    se = projected_intervals.standard_error.to_numpy()
    low, high = projected_intervals.lower.to_numpy(), projected_intervals.upper.to_numpy()
    crosses_zero = (low <= 0) & (high >= 0)
    contributions = (scenarios @ p.T)**2/d
    totals = contributions.sum(axis=1)
    shares = np.divide(contributions, totals[:, None], out=np.full_like(contributions, np.nan),
                       where=totals[:, None] > 0)
    order = np.argsort(-contributions, axis=1, kind='stable')
    ranks = np.argsort(order, axis=1, kind='stable')+1
    detail['alpha_standard_error'] = raw_intervals.standard_error
    detail['alpha_lower'] = raw_intervals.lower
    detail['alpha_upper'] = raw_intervals.upper
    detail['projected_standard_error'] = se
    detail['projected_lower'], detail['projected_upper'] = low, high
    detail['K_noise'] = np.diag(vh)/d
    detail['K_noise_adjusted'] = detail.K_contribution-detail.K_noise
    detail['K_contribution_lower'] = np.where(crosses_zero, 0., np.minimum(low**2, high**2))/d
    detail['K_contribution_upper'] = np.maximum(low**2, high**2)/d
    detail['K_share'] = detail.K_contribution/points['K'] if points['K'] > 1e-24 else np.nan
    detail['rank'] = detail.K_contribution.rank(ascending=False, method='first').astype(int)
    detail['top10_scenario_frequency'] = (ranks <= min(10, len(a))).mean(axis=0)
    detail['rank_scenario_p05'], detail['rank_scenario_p95'] = np.quantile(
        ranks, [.05, .95], axis=0)
    detail['share_scenario_p05'], detail['share_scenario_p95'] = np.quantile(
        shares, [.05, .95], axis=0)
    detail['projected_simultaneous_lower'] = simultaneous_intervals.lower
    detail['projected_simultaneous_upper'] = simultaneous_intervals.upper
    for prefix, table in [('alpha', raw_intervals), ('projected', projected_intervals),
                           ('projected_simultaneous', simultaneous_intervals)]:
        for column in ['interval_method', 'scope', 'status', 'confidence']:
            detail[prefix+'_'+column] = table[column]
    values['A_plus'] = np.sum(np.maximum(scenarios, 0.)**2/d, axis=1)
    values['alpha_mean'] = scenarios.mean(axis=1)
    values['alpha_median'] = np.median(scenarios, axis=1)
    values['positive_count'] = (scenarios > 0).sum(axis=1)
    sorted_shares = np.sort(shares, axis=1)[:, ::-1]
    values['top5_share'] = sorted_shares[:, :5].sum(axis=1)
    values['top10_share'] = sorted_shares[:, :10].sum(axis=1)
    values['effective_contributors'] = 1/np.sum(shares**2, axis=1)
    return dict(summary=pd.DataFrame(summary).set_index('metric'), contributors=detail,
                projected_covariance=pd.DataFrame(vh, index=model.alpha.index,
                                                  columns=model.alpha.index),
                alpha_scenarios=scenarios, scenario_metrics=pd.DataFrame(values),
                scope='conditional on fitted B, D, F; '
                      'plug-in covariance; scenarios are sensitivity')


def removal_uncertainty(model, covariance, alpha_scenarios, *, bundles=None, confidence=.95,
                        quadratic_method='spectral', bootstrap_errors=None):
    """Evaluate paired leave-one-out or named-bundle K loss with subset GLS recomputation."""
    v = _covariance(model, covariance)
    ids = model.alpha.index
    a = model.alpha.to_numpy()
    scenarios = np.asarray(alpha_scenarios, dtype=float)
    if scenarios.ndim != 2 or scenarios.shape[1] != len(a) or not np.isfinite(scenarios).all():
        raise ValueError('scenario columns must match the complete asset axis')
    full = _metric(_projectors(model)['K'], model.residual_variances)
    groups = {name: [name] for name in ids} if bundles is None else bundles
    rows = []
    for name, removed in groups.items():
        remove = pd.Index(removed)
        if remove.empty or remove.has_duplicates or not remove.isin(ids).all():
            raise ValueError('removal members must be unique known assets')
        mask = ~ids.isin(remove)
        if not mask.any():
            raise ValueError('cannot remove the whole universe')
        b, d = model.betas.loc[mask], model.residual_variances.loc[mask]
        c = pd.DataFrame(np.column_stack([np.ones(len(b)), b]), index=b.index)
        p = sa.gls_projector(c, d)
        change = full.copy()
        change[np.ix_(mask, mask)] -= _metric(p, d)
        row = quadratic_confidence_summary(a, v, change, confidence=confidence,
                                             method=quadratic_method, errors=bootstrap_errors)
        sampled = np.einsum('ij,ij->i', scenarios @ change, scenarios)
        row.update(asset=name, removed_N=len(remove), scenario_p05=float(np.quantile(sampled, .05)),
                   scenario_p95=float(np.quantile(sampled, .95)))
        rows.append(row)
    return pd.DataFrame(rows)


def fixed_portfolio_alpha_interval(model, covariance, weights, reference, *,
                                   projected=False, confidence=.95, interval=None):
    """Approximate pointwise interval for a fixed book's active alpha, not return risk."""
    v = _covariance(model, covariance)
    for name, book in [('weights', weights), ('reference', reference)]:
        sa._vector(book, model.betas, name)
        if book.min() < 0 or not np.isclose(book.sum(), 1., atol=1e-8):
            raise ValueError('books must be long-only and unit funded')
    loading = (weights-reference).to_numpy()
    if projected:
        loading = _projectors(model)['K'].T @ loading
    result = gaussian_quadratic_summary(model.alpha, v, np.outer(loading, loading),
                                        confidence=confidence)
    estimate = float(loading @ model.alpha)
    standard_error = np.sqrt(max(0., result['noise']))
    table = _interval_table(np.array([estimate]), np.array([[standard_error**2]]),
                             pd.Index(['active_alpha']), confidence, interval)
    return dict(estimate=estimate, standard_error=float(standard_error),
                lower=float(table.lower.iloc[0]), upper=float(table.upper.iloc[0]),
                interval_method=table.interval_method.iloc[0], status=table.status.iloc[0],
                scope='fixed-book conditional alpha; '+table.scope.iloc[0])
