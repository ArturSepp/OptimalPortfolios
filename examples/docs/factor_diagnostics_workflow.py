"""Offline FactorLasso -> MATF alpha uncertainty -> QIS risk and realised attribution.

Monthly simple fractional returns, one currency, zero cash/funding rate. Betas fit on
an early block, then freeze; residual means/risk fit on a later training block. Annual
moments use arithmetic scale 12, never compounded annual returns. Conditional intervals
exclude beta selection/estimation, risk-model uncertainty and future-return prediction.
No file writes or network access. ``render_report`` returns a QIS-based figure to its caller.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import factorlasso as fl
import qis

from optimalportfolios.optimization.executed_state import backtest_with_qis_state
from factormodels.premia.sharpe_accounting import gls_projector, gls_split
from factormodels.analytics import (
    FactorOpportunityModel, portfolio_opportunity_metrics, fixed_portfolio_alpha_interval,
)

ANNUALIZATION = 12.
ALPHA_SPAN = 36
BANDWIDTH = 6.


def synthetic_returns(seed=17):
    """Generate fixed-seed monthly factors and stationary serial/cross-correlated residuals."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range('2004-02-29', periods=240, freq='ME')
    betas = np.array([[.8, .1], [1.2, -.2], [.3, .9], [-.2, .7], [.6, .5], [1., .2]])
    factors = rng.normal([.003, .001], [.025, .015], (len(dates), 2))
    scales = np.linspace(.009, .014, 6)
    spatial = np.outer(scales, scales) * (.2 + .8 * np.eye(6))
    innovations = rng.standard_normal((len(dates), 6)) @ np.linalg.cholesky(spatial).T
    errors = innovations.copy()
    for i in range(1, len(dates)):
        errors[i] = .35 * errors[i - 1] + np.sqrt(1 - .35**2) * innovations[i]
    returns = factors @ betas.T + errors + [.001, -.0005, .0007, 0., .0012, -.0003]
    return (pd.DataFrame(factors, index=dates, columns=['Equity', 'Rates']),
            pd.DataFrame(returns, index=dates, columns=[f'Asset {i + 1}' for i in range(6)]))


def run_workflow(factor_returns=None, asset_returns=None):
    """Connect existing estimators and diagnostics without changing their calculations.

    Args:
        factor_returns: Optional complete monthly simple-return frame of the fixed example.
        asset_returns: Matching monthly simple-return frame. Both frames must be supplied
            together; no transforms or estimates see rows after the training cutoff.

    Returns:
        Dictionary of fitted snapshots, interval/risk tables, QIS portfolios, attribution
        and provenance. Bootstrap bounds remain explicitly unvalidated sensitivity output.
    """
    if factor_returns is None and asset_returns is None:
        factor_returns, asset_returns = synthetic_returns()
    assert factor_returns.index.equals(asset_returns.index) and len(asset_returns) == 240
    x, y = factor_returns.copy(), asset_returns.copy()
    assert np.isfinite(x).all().all() and np.isfinite(y).all().all() and (y > -1).all().all()
    beta_end, asof = y.index[71], y.index[191]
    model = fl.LassoModel(model_type=fl.LassoModelType.LASSO, reg_lambda=1e-6,
                          span=None, demean=True, solver='CLARABEL')
    model.fit(x=x.loc[:beta_end], y=y.loc[:beta_end])
    b = model.coef_.copy()
    xc, yc = x.iloc[72:192], y.iloc[72:192]
    residuals = yc - xc @ b.T  # Raw residual means, not demeaned solver alpha/intercept.
    calendar = np.arange(len(residuals), dtype=float)
    uncertainty = fl.estimate_alpha_uncertainty(
        residuals, ALPHA_SPAN, calendar=calendar, bandwidth=BANDWIDTH,
        scale=ANNUALIZATION)
    # Reuse FactorLasso moment recursion and covariance container; no covariance fit here.
    f = pd.DataFrame(ANNUALIZATION * fl.compute_ewm_covar(
        (xc - xc.mean()).to_numpy(), span=60), index=xc.columns, columns=xc.columns)
    residual_moment = fl.compute_ewm_covar((residuals - residuals.mean()).to_numpy(), span=60)
    d = pd.Series(ANNUALIZATION * np.diag(residual_moment), index=y.columns)
    snapshot = fl.CurrentFactorCovarData(x_covar=f, y_betas=b,
        y_variances=pd.DataFrame({'residual_var': d}), residuals=residuals,
        estimation_date=asof)
    opportunity = FactorOpportunityModel(b, d, f, uncertainty.estimates, date=asof)
    weights = pd.Series([.30, .15, .10, .10, .20, .15], index=y.columns)
    benchmark = pd.Series(1 / len(weights), index=y.columns)
    active = weights - benchmark
    metrics, exposures, holdings = portfolio_opportunity_metrics(opportunity, weights, benchmark)
    coordinates, orthogonal = gls_split(uncertainty.estimates, b, d)
    spanned = b @ coordinates
    p = gls_projector(b, d)
    contrasts = np.stack([active, active @ (np.eye(len(b)) - p), active @ p])
    targets = contrasts @ uncertainty.estimates.to_numpy()
    covariance = contrasts @ uncertainty.covariance.to_numpy() @ contrasts.T
    names = pd.Index(['Active alpha', 'Factor-spanned alpha', 'Factor-orthogonal alpha'])
    intervals = pd.DataFrame(fl.linear_confidence_intervals(targets, covariance), index=names)
    fixed_interval = fixed_portfolio_alpha_interval(
        opportunity, uncertainty.covariance, weights, benchmark)
    bootstrap = fl.bootstrap_weighted_means(
        residuals, uncertainty.weights, calendar=calendar, bandwidth=BANDWIDTH,
        scale=ANNUALIZATION, contrasts=contrasts, draws=400, seed=29)
    bootstrap_intervals = pd.DataFrame(fl.linear_confidence_intervals(
        targets, covariance, method='bootstrap_t', errors=bootstrap['errors'],
        replicate_standard_errors=bootstrap['standard_errors']), index=names)
    alpha_table = pd.DataFrame({'alpha': uncertainty.estimates, 'factor_spanned': spanned,
                               'factor_orthogonal': orthogonal,
                               'estimation_se': uncertainty.diagnostics.standard_error})
    # Only held-out prices reach the execution loop; estimation ends at the opening close.
    prices = 100 * (1 + y.loc[asof:]).cumprod()
    prices = prices.div(prices.iloc[0]) * 100
    decisions = prices.index[[0, 12, 24, 36]]
    portfolio = backtest_with_qis_state(prices, decisions, lambda history, state: weights,
                                       implementation_lag=1, transaction_costs=.001)
    reference = backtest_with_qis_state(prices, decisions, lambda history, state: benchmark,
                                       implementation_lag=1, transaction_costs=.001)
    future = y.index[192:]
    pw, bw = portfolio.portfolio, reference.portfolio
    lagged_active = (pw.weights - bw.weights).shift(1).loc[future]
    forward_residuals = y.loc[future] - x.loc[future] @ b.T
    factor_component = ((lagged_active @ b) * x.loc[future]).sum(axis=1)
    residual_component = (lagged_active * forward_residuals).sum(axis=1)
    gross = (pw.get_instruments_pnl() - bw.get_instruments_pnl()).sum(axis=1).loc[future]
    net = (pw.get_instruments_pnl(is_net=True)
           - bw.get_instruments_pnl(is_net=True)).sum(axis=1).loc[future]
    actual = (qis.to_returns(pw.nav, is_log_returns=False)
              - qis.to_returns(bw.nav, is_log_returns=False)).loc[future]
    attribution = pd.DataFrame({'factor': factor_component, 'residual': residual_component,
                                'trade_cost_effect': net - gross, 'net_NAV_active': actual})
    attribution['unexplained_accounting'] = (actual - attribution.factor
        - attribution.residual - attribution.trade_cost_effect)
    te, ir = qis.compute_te_ir_errors(actual.to_frame('Implemented active'))
    ewma_te = qis.compute_ewma_realised_tracking_error(
        pw.nav, bw.nav, ewma_span=12, freq='ME', is_log_returns=False)
    risk = pd.Series({k: metrics[k] for k in ['total_te', 'factor_te', 'residual_te',
                                              'portfolio_vol']})
    diagnostics = pd.DataFrame({
        'training_residuals': fl.diagnose_residuals(residuals).to_dict(),
        'held_out_residuals': fl.diagnose_residuals(forward_residuals).to_dict()}).T
    provenance = pd.DataFrame([
        dict(stage='beta fit', first=x.index[0], last=beta_end, observations=72),
        dict(stage='conditional alpha/risk', first=xc.index[0], last=asof, observations=len(xc)),
        dict(stage='forward evaluation', first=future[0], last=future[-1],
             observations=len(future))])
    return dict(betas=b, snapshot=snapshot, opportunity=opportunity, residuals=residuals,
                alpha=alpha_table, uncertainty=uncertainty, contrasts=contrasts,
                target_covariance=covariance, intervals=intervals,
                bootstrap_intervals=bootstrap_intervals, bootstrap=bootstrap,
                fixed_interval=fixed_interval, risk=risk, exposures=exposures, holdings=holdings,
                diagnostics=diagnostics, attribution=attribution, realised_te=te,
                realised_ir=ir, ewma_realised_te=ewma_te, portfolio=portfolio, reference=reference,
                provenance=provenance, asof=asof)


def assert_references(result):
    """Check identities with independent whitened QR, matrix and return references."""
    b = result['betas'].to_numpy()
    m = result['opportunity']
    d, f, a = m.residual_variances.to_numpy(), m.factor_covariance.to_numpy(), m.alpha.to_numpy()
    q, _ = np.linalg.qr(b / np.sqrt(d)[:, None], mode='reduced')
    # An independent QR calculation only for verification, not a second production projector.
    p = np.eye(len(b)) - np.sqrt(d)[:, None] * (q @ q.T) / np.sqrt(d)[None, :]
    h = p @ a
    np.testing.assert_allclose(result['alpha'].factor_orthogonal, h, atol=1e-12, rtol=0)
    np.testing.assert_allclose(result['alpha'].factor_spanned + h, a, atol=1e-12, rtol=0)
    np.testing.assert_allclose(b.T @ (h / d), 0., atol=1e-11, rtol=0)
    active = result['holdings'].active_weight.to_numpy()
    c = np.stack([active, active @ (np.eye(len(b)) - p), active @ p])
    expected_v = c @ result['uncertainty'].covariance @ c.T
    np.testing.assert_allclose(result['target_covariance'], expected_v, atol=1e-14, rtol=0)
    np.testing.assert_allclose(result['intervals'].standard_error**2,
                               np.diag(expected_v), atol=1e-14, rtol=0)
    assert np.isclose(result['fixed_interval']['standard_error'],
                      result['intervals'].standard_error.iloc[0], atol=1e-14)
    np.testing.assert_allclose(result['intervals'].estimate.iloc[0],
                               result['intervals'].estimate.iloc[1:].sum(), atol=1e-13)
    sigma = b @ f @ b.T + np.diag(d)
    np.testing.assert_allclose(result['snapshot'].get_y_covar(), sigma, atol=1e-14, rtol=0)
    risk = result['risk']
    np.testing.assert_allclose(risk.total_te**2, active @ sigma @ active, atol=1e-14)
    np.testing.assert_allclose(risk.factor_te**2, (b.T @ active) @ f @ (b.T @ active), atol=1e-14)
    np.testing.assert_allclose(risk.residual_te**2, (active**2) @ d, atol=1e-14)
    attr = result['attribution']
    assert attr.unexplained_accounting.abs().max() < 1e-12
    np.testing.assert_allclose(attr[['factor', 'residual', 'trade_cost_effect']].sum(axis=1),
                               attr.net_NAV_active, atol=1e-12, rtol=0)
    # Independent sample standard deviation verifies the canonical QIS scalar, not a new API.
    assert np.isclose(result['realised_te'].iloc[0],
                      np.std(attr.net_NAV_active, ddof=1) * np.sqrt(ANNUALIZATION), atol=1e-13)
    boot = result['bootstrap']
    np.testing.assert_allclose(boot['errors'][:, 0], boot['errors'][:, 1:].sum(axis=1), atol=1e-13)
    np.testing.assert_allclose(boot['covariance'], expected_v, atol=1e-14, rtol=0)
    implied = expected_v.to_numpy() * (len(result['residuals']) - 1) / len(result['residuals'])
    empirical = np.cov(boot['errors'], rowvar=False, ddof=1)
    relative = np.linalg.norm(empirical - implied) / np.linalg.norm(implied)
    assert relative < .30, relative
    return dict(max_accounting_gap=float(attr.unexplained_accounting.abs().max()),
                bootstrap_covariance_relative_error=float(relative),
                max_projection_error=float(np.max(abs(result['alpha'].factor_orthogonal - h))))


def coverage_experiment(result, draws=3000, seed=43):
    """Measure conditional interval coverage on full stationary Gaussian AR(1) panels.

    B, D, contrasts and estimator weights are fixed, not estimated in the Monte Carlo.
    Cross-asset correlation is .25, temporal phi is -.3 or .5, monthly residual volatility
    is .01 and there are 96 observations. Known-shape calibration uses the true AR(1)
    covariance shape. Approximate normal/HAC coverage is measured without a nominal claim.
    Every target/phi known-shape coverage must be within five binomial standard errors
    of .95. This is conditional synthetic evidence, not a fitted-model coverage theorem.
    """
    rng = np.random.default_rng(seed)
    t, n = 96, len(result['betas'])
    weights = fl.compute_ewm(np.eye(t), span=24)[-1]
    geometry = fl.compute_weighted_mean_hac_geometry(
        weights, calendar=np.arange(t), bandwidth=BANDWIDTH, scale=ANNUALIZATION)
    contrasts = result['contrasts']
    spatial = .0001 * (.25 + .75 * np.eye(n))
    root = np.linalg.cholesky(spatial)
    nominal = .95
    tolerance = 5 * np.sqrt(nominal * (1 - nominal) / draws)
    normal_critical = fl.linear_confidence_intervals([0.], [[1.]])['critical_value'][0]
    rows = []
    for phi in (-.3, .5):
        innovations = rng.standard_normal((draws, t, n)) @ root.T
        errors = innovations.copy()
        for i in range(1, t):
            errors[:, i] = phi * errors[:, i - 1] + np.sqrt(1 - phi**2) * innovations[:, i]
        target_paths = (errors @ contrasts.T).transpose(0, 2, 1).reshape(-1, t)
        estimates, standard_errors = geometry.statistics(target_paths)
        estimates = estimates.reshape(draws, len(contrasts))
        standard_errors = standard_errors.reshape(estimates.shape)
        shape = phi ** np.abs(np.arange(t)[:, None] - np.arange(t)[None, :])
        critical = float(fl.gaussian_critical_value(geometry, shape, alpha=1 - nominal))
        coverage = (abs(estimates) <= critical * standard_errors).mean(axis=0)
        approximate = (abs(estimates) <= normal_critical * standard_errors).mean(axis=0)
        for j, name in enumerate(result['intervals'].index):
            rows.append(dict(phi=phi, target=name, draws=draws, nominal=nominal,
                known_shape_coverage=float(coverage[j]), normal_HAC_coverage=float(approximate[j]),
                binomial_tolerance=float(tolerance), known_shape_critical=critical,
                normal_critical=float(normal_critical),
                scope='pointwise_fixed_model_true_Gaussian_AR1_shape'))
    table = pd.DataFrame(rows)
    assert ((table.known_shape_coverage - nominal).abs() <= tolerance).all(), table
    return table


def render_report(result):
    """Return a labelled three-panel QIS report; the caller owns C-local figure output."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 1, figsize=(12, 12), constrained_layout=True)
    table = result['intervals'][['estimate', 'standard_error', 'lower', 'upper']].copy()
    table.columns = ['Estimate', 'Estimation SE', '95% lower', '95% upper']
    qis.plot_df_table(table, ax=axes[0], var_format='{:.2%}', fontsize=10,
                      title='Annual residual-mean contrasts: conditional normal/HAC intervals')
    navs = pd.concat([result['portfolio'].portfolio.nav.rename('Portfolio'),
                      result['reference'].portfolio.nav.rename('Benchmark')], axis=1)
    qis.plot_prices(navs, start_to_one=False, ax=axes[1], x_date_freq='YE', perf_stats_labels=None,
                 title='Untouched forward implementation: cash start, lag 1, trade fees 10 bp')
    attr = result['attribution'].sum().to_frame('Arithmetic sum')
    qis.plot_df_table(attr, ax=axes[2], var_format='{:.3%}', fontsize=10,
                      title='Monthly active contributions: factor / residual / costs / accounting')
    fig.suptitle('Fixed fitted B/D/F; estimation intervals exclude selection and forecast risk',
                  fontsize=12)
    return fig


def main():
    """Execute the guide, independent checks and prespecified Monte Carlo coverage gate."""
    result = run_workflow()
    checks = assert_references(result)
    coverage = coverage_experiment(result)
    print(result['intervals'][['estimate', 'standard_error', 'lower', 'upper', 'status']])
    print(result['risk'])
    print(coverage.to_string(index=False))
    print(checks)
    return result, coverage


if __name__ == '__main__':
    main()
