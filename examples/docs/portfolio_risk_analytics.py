"""Canonical script of docs/portfolio_risk_analytics.md.

The page shows excerpts of this file; every number and property it states is asserted here
against a reference computed a different way: explicit matrix products, central finite
differences, least-squares regressions on simulated returns with a fixed seed, the joint
covariance of a factor model, and the optimality conditions of the risk-based portfolios. The
script runs offline after ``pip install optimalportfolios`` and needs no data file:

    python -m examples.docs.portfolio_risk_analytics

``exhibit`` draws the page's figure; ``tools/docs_analytics/teaching.py`` calls it with the
constants below and records their values.
"""
import numpy as np
import pandas as pd
import qis
from factorlasso import CurrentFactorCovarData, RollingFactorCovarData

import optimalportfolios as op

TICKERS = ['Govt', 'Credit', 'US eq', 'EM eq', 'Gold']
FACTORS = ['Rates', 'Equity', 'Commodities']
FACTOR_VOLS = [0.06, 0.16, 0.20]  # annual
FACTOR_CORR = [[1.0, -0.2, 0.0],
               [-0.2, 1.0, 0.3],
               [0.0, 0.3, 1.0]]
LOADINGS = [[1.0, 0.0, 0.0],
            [0.8, 0.3, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.2, 0.3],
            [0.4, 0.1, 0.6]]
RESIDUAL_VOLS = [0.01, 0.03, 0.05, 0.10, 0.10]  # annual
PORTFOLIO = [0.30, 0.15, 0.30, 0.15, 0.10]
BENCHMARK = {'Govt': 0.40, 'US eq': 0.60}
# The second quarter's factor correlation: rates and equities now move together.
STRESSED_FACTOR_CORR = [[1.0, 0.3, 0.0],
                        [0.3, 1.0, 0.3],
                        [0.0, 0.3, 1.0]]
COVAR_DATES = ['2024-03-29', '2024-06-28']
INDEX_LOADINGS = [0.4, 0.6, 0.0]  # factor loadings of an external benchmark index
INDEX_RESIDUAL_VOL = 0.02
N_DRAWS = 10_000  # weekly returns simulated for the regression check
SEED = 11
HELD = 1e-4  # a weight above this is a held asset


def factor_snapshot(factor_corr) -> CurrentFactorCovarData:
    """Return the factor-model snapshot of the page with the given factor correlation."""
    factor_covar = pd.DataFrame(np.outer(FACTOR_VOLS, FACTOR_VOLS) * np.array(factor_corr),
                                index=FACTORS, columns=FACTORS)
    return CurrentFactorCovarData(
        x_covar=factor_covar,
        y_betas=pd.DataFrame(LOADINGS, index=TICKERS, columns=FACTORS),
        y_variances=pd.DataFrame({'residual_var': np.square(RESIDUAL_VOLS)}, index=TICKERS))


def model_covariance(factor_corr) -> np.ndarray:
    """Assemble B F B' + D from the constants with NumPy alone, without factorlasso."""
    vols = np.array(FACTOR_VOLS)
    factor_covar = np.outer(vols, vols) * np.array(factor_corr)
    loadings = np.array(LOADINGS)
    return loadings @ factor_covar @ loadings.T + np.diag(np.square(RESIDUAL_VOLS))


def euler_shares(weights, covar) -> np.ndarray:
    """Euler risk shares w_i (Sigma w)_i / (w' Sigma w), computed without the package or qis."""
    w = np.asarray(weights, dtype=float)
    sigma = np.asarray(covar, dtype=float)
    return w * (sigma @ w) / (w @ sigma @ w)


def vol_gradient(covar: np.ndarray, weights: np.ndarray, step: float = 1e-6) -> np.ndarray:
    """Central finite differences of the portfolio volatility with respect to each weight."""
    def vol(w):
        """Portfolio volatility sqrt(w' Sigma w)."""
        return np.sqrt(w @ covar @ w)

    bumps = np.eye(len(weights)) * step
    return np.array([(vol(weights + bump) - vol(weights - bump)) / (2.0 * step)
                     for bump in bumps])


def simulated_returns(covar: pd.DataFrame, n_draws: int, seed: int) -> pd.DataFrame:
    """Draw weekly Gaussian returns whose annual covariance is ``covar``."""
    rng = np.random.default_rng(seed)
    draws = rng.multivariate_normal(np.zeros(len(covar)), covar.to_numpy() / 52.0, size=n_draws)
    return pd.DataFrame(draws, columns=covar.columns)


def ols_slopes(returns: pd.DataFrame, regressor: pd.Series) -> tuple:
    """Least-squares slope of each column on ``regressor`` with an intercept, and its error."""
    design = np.column_stack([np.ones(len(regressor)), regressor.to_numpy()])
    coefficients, *_ = np.linalg.lstsq(design, returns.to_numpy(), rcond=None)
    residuals = returns.to_numpy() - design @ coefficients
    variance = (residuals ** 2).sum(axis=0) / (len(regressor) - 2)
    inverse = np.linalg.inv(design.T @ design)
    return (pd.Series(coefficients[1], index=returns.columns),
            pd.Series(np.sqrt(variance * inverse[1, 1]), index=returns.columns))


def risk_based_portfolios(covar: pd.DataFrame) -> dict:
    """Solve equal weight, minimum variance, equal risk and maximum diversification."""
    long_only = op.Constraints(is_long_only=True)
    minimum_variance, _ = op.wrapper_quadratic_optimisation(pd_covar=covar, constraints=long_only)
    return {
        'Equal weight': pd.Series(1.0 / len(covar), index=covar.index),
        'Minimum variance': minimum_variance,
        'Equal risk contribution': op.wrapper_risk_budgeting(pd_covar=covar,
                                                             constraints=long_only),
        'Maximum diversification': op.wrapper_maximise_diversification(pd_covar=covar,
                                                                       constraints=long_only),
    }


def assert_raises(error: type, function, **arguments) -> None:
    """Fail unless ``function(**arguments)`` raises ``error``."""
    try:
        function(**arguments)
    except error:
        return
    raise AssertionError(f'expected {error.__name__}')


def main() -> None:
    """Run the worked example of the page and assert every number and property it states."""
    snapshot = factor_snapshot(FACTOR_CORR)
    covar = snapshot.get_y_covar()
    weights = pd.Series(PORTFOLIO, index=TICKERS)
    variance = op.compute_portfolio_variance(w=weights.to_numpy(), covar=covar.to_numpy())
    vol = op.compute_portfolio_vol(covar=covar, weights=weights)
    table = op.compute_portfolio_risk_contribution_outputs(weights=weights, clean_covar=covar)
    print(table.round(4))
    assert np.isclose(table['risk contribution'].sum(), vol, rtol=1e-14, atol=0.0)

    # The factorlasso covariance equals B F B' + D assembled with NumPy; the asset volatilities
    # quoted on the page run from 6.1% (Govt) to 24.0% (EM eq).
    sigma, w = covar.to_numpy(), weights.to_numpy()
    np.testing.assert_allclose(sigma, model_covariance(FACTOR_CORR), rtol=0.0, atol=1e-16)
    asset_vols = np.sqrt(np.diag(sigma))
    assert np.round(asset_vols, 3).tolist() == [0.061, 0.068, 0.168, 0.240, 0.162]
    # Variance and volatility equal the explicit double sum and its square root.
    double_sum = sum(w[i] * w[j] * sigma[i, j] for i in range(5) for j in range(5))
    assert np.isclose(variance, double_sum, rtol=1e-14) and np.isclose(vol, np.sqrt(double_sum))
    assert round(vol, 4) == 0.0961
    # The contributions are w_i (Sigma w)_i / sigma(w); the shares sum to one; the budget column
    # defaults to zeros; the columns keep their source names.
    contributions = np.array([w[i] * sigma[i] @ w / vol for i in range(5)])
    np.testing.assert_allclose(table['risk contribution'], contributions, rtol=1e-13, atol=0.0)
    np.testing.assert_allclose(table['asset_rc_ratio'], contributions / vol, rtol=1e-13)
    assert np.isclose(table['asset_rc_ratio'].sum(), 1.0, rtol=1e-14)
    assert table['Risk Budget'].eq(0.0).all()
    assert table.columns.tolist() == ['weights', 'risk contribution', 'Risk Budget',
                                      'asset_rc_ratio']
    # The table's quoted rows: US eq 30% of capital and 47.3% of risk, EM eq 15% and 33.9%,
    # Govt 30% and 2.1%.
    shares = table['asset_rc_ratio']
    assert round(shares['US eq'], 3) == 0.473 and round(shares['EM eq'], 3) == 0.339
    assert round(shares['Govt'], 3) == 0.021
    # The page's table of this output: contributions in percent and risk shares.
    assert np.round(100.0 * table['risk contribution'], 2).tolist() == \
        [0.21, 0.75, 4.55, 3.26, 0.86]
    assert np.round(shares, 3).tolist() == [0.021, 0.078, 0.473, 0.339, 0.089]
    # Euler: volatility is homogeneous of degree one, and its gradient Sigma w / sigma(w)
    # matches central finite differences.
    assert np.isclose(op.compute_portfolio_vol(covar=sigma, weights=2.5 * w), 2.5 * vol,
                      rtol=1e-14)
    np.testing.assert_allclose(vol_gradient(sigma, w), sigma @ w / vol, rtol=0.0, atol=1e-9)
    assert np.isclose(w @ vol_gradient(sigma, w), vol, rtol=1e-8)
    # A contribution is not the weight times the asset's own volatility.
    assert (w * asset_vols).sum() > vol * 1.3

    beta_to_portfolio = op.compute_benchmark_beta_loadings_from_covar(
        covar=covar, benchmark_weights=weights, asset_tickers=TICKERS)
    assert np.allclose(table['asset_rc_ratio'], weights * beta_to_portfolio, rtol=1e-12)

    # Insight: the risk share is the weight times the beta to the portfolio itself, so an asset
    # takes more than its capital share of risk exactly when that beta exceeds one.
    np.testing.assert_allclose(beta_to_portfolio, sigma @ w / (w @ sigma @ w), rtol=1e-13)
    assert ((shares > weights) == (beta_to_portfolio > 1.0)).all()
    assert np.round(beta_to_portfolio, 2).tolist() == [0.07, 0.52, 1.58, 2.26, 0.89]
    assert np.isclose(beta_to_portfolio @ weights, 1.0, rtol=1e-14)

    long_only = op.Constraints(is_long_only=True)
    minimum_variance, _ = op.wrapper_quadratic_optimisation(pd_covar=covar, constraints=long_only)
    portfolios = {
        'Equal weight': pd.Series(1.0 / len(TICKERS), index=TICKERS),
        'Minimum variance': minimum_variance,
        'Equal risk contribution': op.wrapper_risk_budgeting(pd_covar=covar,
                                                             constraints=long_only),
        'Maximum diversification': op.wrapper_maximise_diversification(pd_covar=covar,
                                                                       constraints=long_only),
    }
    risk_shares = {name: op.compute_portfolio_risk_contribution_outputs(
        weights=portfolio, clean_covar=covar)['asset_rc_ratio']
        for name, portfolio in portfolios.items()}

    # The exhibit's helper solves the same portfolios.
    for name, portfolio in risk_based_portfolios(covar).items():
        np.testing.assert_allclose(portfolio, portfolios[name], atol=1e-12)
    # Every portfolio is fully invested and long-only, and its risk shares match the explicit
    # Euler formula and sum to one.
    for name, portfolio in portfolios.items():
        assert np.isclose(portfolio.sum(), 1.0, atol=1e-8) and (portfolio > -1e-9).all()
        np.testing.assert_allclose(risk_shares[name], euler_shares(portfolio, sigma), atol=1e-12)
        assert np.isclose(risk_shares[name].sum(), 1.0, atol=1e-12)
    # Equal weight: EM eq carries 42% of the risk and Govt 1%.
    ew = risk_shares['Equal weight']
    assert round(ew['EM eq'], 2) == 0.42 and round(ew['Govt'], 2) == 0.01
    # Equal risk contribution: shares of one fifth, and weights inversely proportional to each
    # asset's beta to the portfolio.
    erc = portfolios['Equal risk contribution']
    np.testing.assert_allclose(risk_shares['Equal risk contribution'], 0.2, atol=1e-6)
    erc_beta = sigma @ erc.to_numpy() / (erc.to_numpy() @ sigma @ erc.to_numpy())
    np.testing.assert_allclose(erc.to_numpy() * erc_beta, 0.2, atol=1e-6)
    assert round(erc['Govt'], 2) == 0.45 and round(erc['EM eq'], 2) == 0.08
    # Minimum variance: held assets have beta one to the portfolio and excluded assets at
    # least one, so risk shares equal capital weights; Credit and EM eq are not held.
    mv = portfolios['Minimum variance'].to_numpy()
    mv_beta = sigma @ mv / (mv @ sigma @ mv)
    held = mv > HELD
    np.testing.assert_allclose(mv_beta[held], 1.0, atol=1e-5)
    assert (mv_beta[~held] > 1.0).all()
    assert [TICKERS[i] for i in np.flatnonzero(~held)] == ['Credit', 'EM eq']
    np.testing.assert_allclose(risk_shares['Minimum variance'], mv, atol=1e-5)
    assert np.round(mv, 2).tolist() == [0.83, 0.0, 0.15, 0.0, 0.02]
    # Maximum diversification: the risk share of each asset is its share of weighted
    # volatility w_i sigma_i / sigma'w.
    mdp = portfolios['Maximum diversification'].to_numpy()
    np.testing.assert_allclose(risk_shares['Maximum diversification'],
                               mdp * asset_vols / (mdp @ asset_vols), atol=1e-4)
    assert round(mdp[0], 2) == 0.67
    assert round(risk_shares['Maximum diversification']['Govt'], 2) == 0.41

    benchmark = pd.Series(BENCHMARK)
    beta_loadings = op.compute_benchmark_beta_loadings_from_covar(
        covar=covar, benchmark_weights=benchmark, asset_tickers=TICKERS)
    portfolio_beta = float(beta_loadings @ weights)
    print(beta_loadings.round(3).to_dict(), round(portfolio_beta, 3))

    # The loadings are the explicit slice Sigma[:, C] b / (b' Sigma[C, C] b); the benchmark's
    # own beta is one; the page quotes the loadings to two decimals and the beta 0.92.
    constituents = [TICKERS.index(name) for name in benchmark.index]
    b = benchmark.to_numpy()
    explicit = sigma[:, constituents] @ b / (b @ sigma[np.ix_(constituents, constituents)] @ b)
    np.testing.assert_allclose(beta_loadings, explicit, rtol=1e-13, atol=0.0)
    assert np.isclose(beta_loadings[benchmark.index] @ benchmark, 1.0, rtol=1e-14)
    assert np.round(beta_loadings, 2).tolist() == [0.03, 0.47, 1.64, 1.97, 0.51]
    assert round(portfolio_beta, 2) == 0.92
    # The benchmark volatility is 9.9%; Govt is 40% of the benchmark but has almost no beta.
    full_benchmark = benchmark.reindex(TICKERS, fill_value=0.0).to_numpy()
    assert round(np.sqrt(full_benchmark @ sigma @ full_benchmark), 3) == 0.099
    # Pitfall: benchmark weights in percent scale every loading by 1/100; only the
    # benchmark's own beta stays one.
    in_percent = op.compute_benchmark_beta_loadings_from_covar(
        covar=covar, benchmark_weights=100.0 * benchmark, asset_tickers=TICKERS)
    np.testing.assert_allclose(in_percent, beta_loadings / 100.0, rtol=1e-13)
    assert round(float(in_percent @ weights), 4) == 0.0092
    assert np.isclose(in_percent[benchmark.index] @ (100.0 * benchmark), 1.0, rtol=1e-14)
    # A constituent missing from the covariance raises KeyError; a benchmark without variance
    # raises ValueError.
    assert_raises(KeyError, op.compute_benchmark_beta_loadings_from_covar, covar=covar,
                  benchmark_weights=pd.Series({'Govt': 0.4, 'Cash': 0.6}), asset_tickers=TICKERS)
    assert_raises(ValueError, op.compute_benchmark_beta_loadings_from_covar, covar=covar,
                  benchmark_weights=0.0 * benchmark, asset_tickers=TICKERS)

    returns = simulated_returns(covar, n_draws=N_DRAWS, seed=SEED)
    benchmark_returns = returns[benchmark.index] @ benchmark
    slopes, errors = ols_slopes(returns, benchmark_returns)
    assert (np.abs(slopes - beta_loadings) < 4.0 * errors).all()

    # The regression is an independent estimate: its slopes agree with the loadings within
    # four standard errors; the standard errors run from 0.004 to 0.015 and the largest gap is
    # below 0.01. The portfolio's return regressed on the benchmark return gives the portfolio
    # beta, and its slope is the weighted sum of the asset slopes.
    assert round(errors.min(), 3) == 0.004 and round(errors.max(), 3) == 0.015
    assert np.abs(slopes - beta_loadings).max() < 0.01
    # The page's table of slopes and standard errors.
    assert np.round(slopes, 2).tolist() == [0.03, 0.47, 1.65, 1.97, 0.50]
    assert np.round(errors, 3).tolist() == [0.006, 0.005, 0.004, 0.014, 0.015]
    portfolio_slope, portfolio_error = ols_slopes((returns @ weights).to_frame('p'),
                                                  benchmark_returns)
    assert abs(portfolio_slope['p'] - portfolio_beta) < 4.0 * portfolio_error['p']
    assert np.isclose(portfolio_slope['p'], slopes @ weights, rtol=1e-10)

    factor_covar = snapshot.x_covar
    index_loadings = pd.Series(INDEX_LOADINGS, index=FACTORS)
    index_beta = op.compute_benchmark_beta_loadings(
        asset_betas=snapshot.y_betas, benchmark_betas=index_loadings,
        factor_covar=factor_covar, benchmark_idio_var=INDEX_RESIDUAL_VOL ** 2)

    # The factor-model loadings equal the joint-covariance loadings when the index is added to
    # the model as a sixth asset whose residual is independent of the others.
    joint_loadings = np.vstack([np.array(LOADINGS), INDEX_LOADINGS])
    joint = joint_loadings @ factor_covar.to_numpy() @ joint_loadings.T \
        + np.diag(np.square([*RESIDUAL_VOLS, INDEX_RESIDUAL_VOL]))
    labels = [*TICKERS, 'Index']
    np.testing.assert_allclose(index_beta, op.compute_benchmark_beta_loadings_from_covar(
        covar=pd.DataFrame(joint, index=labels, columns=labels),
        benchmark_weights=pd.Series({'Index': 1.0}), asset_tickers=TICKERS), rtol=1e-12)
    # For a benchmark that holds the assets, the factor variant omits the residual covariance
    # D w_bm of each constituent with the benchmark, so it understates their loadings.
    residual_var = np.square(RESIDUAL_VOLS)
    constituent_beta = op.compute_benchmark_beta_loadings(
        asset_betas=snapshot.y_betas, benchmark_betas=snapshot.y_betas.T @ full_benchmark,
        factor_covar=factor_covar, benchmark_idio_var=float(residual_var @ full_benchmark ** 2))
    gap = beta_loadings - constituent_beta
    benchmark_variance = full_benchmark @ sigma @ full_benchmark
    np.testing.assert_allclose(gap, residual_var * full_benchmark / benchmark_variance,
                               atol=1e-14)
    assert (gap[benchmark.index] > 0.0).all()
    assert gap.drop(benchmark.index).abs().max() < 1e-14
    assert round(gap['US eq'], 3) == 0.153
    assert_raises(ValueError, op.compute_benchmark_beta_loadings, asset_betas=snapshot.y_betas,
                  benchmark_betas=pd.Series(0.0, index=FACTORS), factor_covar=factor_covar)
    # A factor of factor_covar missing from a set of loadings counts as a zero loading.
    np.testing.assert_allclose(op.compute_benchmark_beta_loadings(
        asset_betas=snapshot.y_betas, benchmark_betas=index_loadings.drop('Commodities'),
        factor_covar=factor_covar, benchmark_idio_var=INDEX_RESIDUAL_VOL ** 2), index_beta,
        rtol=1e-15)

    dates = pd.to_datetime(COVAR_DATES)
    stressed = factor_snapshot(STRESSED_FACTOR_CORR).get_y_covar()
    covar_dict = {dates[0]: covar, dates[1]: stressed}
    loadings_ts = op.compute_benchmark_beta_loadings_ts(
        covar_dict=covar_dict, benchmark_weights=benchmark, asset_tickers=TICKERS)
    monthly = pd.DataFrame([PORTFOLIO] * 6, columns=TICKERS,
                           index=pd.date_range('2024-02-29', periods=6, freq='ME'))
    ex_ante_beta = op.compute_ex_ante_beta_ts(weights=monthly, beta_loadings=loadings_ts)
    print(ex_ante_beta.round(3))

    # Each row equals the single-date loadings of its covariance; each weight date uses the
    # loadings of the last covariance date on or before it; before the first covariance date
    # the beta is reported as 0.0, not NaN.
    np.testing.assert_allclose(loadings_ts.loc[dates[0]], beta_loadings, rtol=1e-14)
    second = model_covariance(STRESSED_FACTOR_CORR)
    second_b = second[:, constituents] @ b / (b @ second[np.ix_(constituents, constituents)] @ b)
    np.testing.assert_allclose(loadings_ts.loc[dates[1]], second_b, rtol=1e-12)
    for date, row in monthly.iterrows():
        known = [d for d in dates if d <= date]
        expected = 0.0 if not known else loadings_ts.loc[known[-1]] @ row
        assert np.isclose(ex_ante_beta[date], expected, rtol=1e-14, atol=0.0)
    assert ex_ante_beta.iloc[0] == 0.0 and ex_ante_beta.name == 'ex_ante_beta'
    assert np.round(ex_ante_beta, 3).tolist() == [0.0, 0.92, 0.92, 0.92, 0.939, 0.939]
    assert round(ex_ante_beta['2024-05-31'], 2) == 0.92
    assert round(ex_ante_beta['2024-06-30'], 2) == 0.94
    assert round(loadings_ts.loc[dates[1], 'Govt'], 2) == 0.27
    # A weight column without a loading, or a non-finite loading, raises ValueError; a missing
    # weight counts as zero; the loadings table is in date order whatever the dictionary order.
    assert_raises(ValueError, op.compute_ex_ante_beta_ts,
                  weights=monthly.assign(Cash=0.0), beta_loadings=loadings_ts)
    assert_raises(ValueError, op.compute_ex_ante_beta_ts, weights=monthly,
                  beta_loadings=loadings_ts.replace(loadings_ts.iloc[0, 0], np.nan))
    gappy = monthly.copy()
    gappy.iloc[2, 1] = np.nan
    zeroed = monthly.copy()
    zeroed.iloc[2, 1] = 0.0
    pd.testing.assert_series_equal(op.compute_ex_ante_beta_ts(weights=gappy,
                                                              beta_loadings=loadings_ts),
                                   op.compute_ex_ante_beta_ts(weights=zeroed,
                                                              beta_loadings=loadings_ts))
    reversed_dict = {dates[1]: stressed, dates[0]: covar}
    pd.testing.assert_frame_equal(op.compute_benchmark_beta_loadings_ts(
        covar_dict=reversed_dict, benchmark_weights=benchmark, asset_tickers=TICKERS),
        loadings_ts)

    date = dates[0]
    risk_model = op.build_risk_model({date: snapshot})
    tracking_error = risk_model.compute_tre_at_date(
        benchmark_weights=benchmark, portfolio_weights=weights, date=date)
    exposures = risk_model.compute_exposures_at_date(portfolio_weights=weights, date=date)
    split = risk_model.compute_tre_decomposition_at_date(
        benchmark_weights=benchmark, portfolio_weights=weights, date=date)
    marginal = risk_model.compute_marginal_tre_at_date(
        benchmark_weights=benchmark, portfolio_weights=weights, date=date)
    assert isinstance(risk_model, qis.RiskModel)

    # Tracking error is sqrt(d' Sigma d) with d = w - w_bm, 3.19% here; exposures are B'w;
    # the factor and residual parts add in squares; the marginal contributions add up to the
    # tracking error, and each splits into its systematic and residual parts.
    d = w - full_benchmark
    assert np.isclose(tracking_error, np.sqrt(d @ sigma @ d), rtol=1e-13)
    assert round(tracking_error, 4) == 0.0319
    np.testing.assert_allclose(exposures, np.array(LOADINGS).T @ w, rtol=1e-14)
    assert np.round(exposures, 3).tolist() == [0.46, 0.535, 0.105]
    assert np.isclose(split['factor_te'] ** 2 + split['residual_te'] ** 2, tracking_error ** 2,
                      rtol=1e-12)
    assert np.isclose(split['tracking_error'], tracking_error, rtol=1e-12)
    assert np.isclose(marginal['mcte'].sum(), tracking_error, rtol=1e-12)
    np.testing.assert_allclose(marginal['mcte'], d * (sigma @ d) / tracking_error, atol=1e-15)
    np.testing.assert_allclose(marginal['mcte_systematic'] + marginal['mcte_residual'],
                               marginal['mcte'], atol=1e-15)
    # The page's numbers: factor 2.11% and residual 2.39%; the Govt underweight is a hedge with
    # a negative contribution, and US eq has the largest contribution, 1.43%.
    assert round(split['factor_te'], 4) == 0.0211 and round(split['residual_te'], 4) == 0.0239
    assert marginal.loc['Govt', 'mcte'] < 0.0 and d[0] < 0.0
    assert marginal['mcte'].idxmax() == 'US eq'
    assert round(marginal.loc['US eq', 'mcte'], 4) == 0.0143
    # The risk model's benchmark beta and loadings are the package helper's.
    assert np.isclose(risk_model.compute_benchmark_beta_at_date(
        benchmark_weights=benchmark, portfolio_weights=weights, date=date), portfolio_beta,
        rtol=1e-12)
    np.testing.assert_allclose(risk_model.compute_benchmark_beta_loadings_at_date(
        benchmark_weights=benchmark, date=date), beta_loadings, rtol=1e-12)
    # A plain {date: covar} dictionary gives a covariance-only model: the same tracking error,
    # and no factor exposures.
    covariance_only = op.build_risk_model({date: covar})
    assert isinstance(covariance_only, qis.RiskModel) and covariance_only.factor_loadings is None
    assert np.isclose(covariance_only.compute_tre_at_date(
        benchmark_weights=benchmark, portfolio_weights=weights, date=date), tracking_error,
        rtol=1e-14)
    assert_raises(ValueError, covariance_only.compute_exposures_at_date,
                  portfolio_weights=weights, date=date)
    # The dated history on the covariance grid matches the ex-ante beta series.
    history = op.build_risk_model(covar_dict).compute_benchmark_beta_history(
        benchmark_weights=benchmark, portfolio_weights=monthly)
    np.testing.assert_allclose(history, [portfolio_beta, loadings_ts.loc[dates[1]] @ w],
                               rtol=1e-12)
    # The history gives zero weights, and so zero beta, before the first weight date.
    late = op.build_risk_model(covar_dict).compute_benchmark_beta_history(
        benchmark_weights=benchmark, portfolio_weights=monthly.loc['2024-04-30':])
    assert late.iloc[0] == 0.0 and np.isclose(late.iloc[1], history.iloc[1], rtol=1e-14)
    # The risk model uses exact covariance dates only and applies no annualisation;
    # build_risk_model rejects other inputs.
    assert_raises(KeyError, risk_model.compute_tre_at_date, benchmark_weights=benchmark,
                  portfolio_weights=weights, date=pd.Timestamp('2024-03-31'))
    weekly_model = op.build_risk_model({date: covar / 52.0})
    assert np.isclose(weekly_model.compute_tre_at_date(
        benchmark_weights=benchmark, portfolio_weights=weights, date=date),
        tracking_error / np.sqrt(52.0), rtol=1e-13)
    assert_raises(ValueError, op.build_risk_model, covar_data=[covar])
    # A RollingFactorCovarData container gives the same model as its dated snapshots.
    rolling = RollingFactorCovarData(data={dates[0]: snapshot,
                                           dates[1]: factor_snapshot(STRESSED_FACTOR_CORR)})
    rolling_model = op.build_risk_model(rolling)
    snapshots_model = op.build_risk_model(dict(rolling.data))
    pd.testing.assert_series_equal(
        rolling_model.compute_tre_history(benchmark_weights=benchmark, portfolio_weights=weights),
        snapshots_model.compute_tre_history(benchmark_weights=benchmark,
                                            portfolio_weights=weights))
    np.testing.assert_allclose(rolling_model.compute_exposures_history(portfolio_weights=weights),
                               [np.array(LOADINGS).T @ w] * 2, rtol=1e-14)
    # wrapper_risk_budgeting returns the risk table with detailed_output=True.
    detailed = op.wrapper_risk_budgeting(pd_covar=covar, constraints=long_only,
                                         detailed_output=True)
    assert detailed.columns.tolist() == table.columns.tolist()
    np.testing.assert_allclose(detailed['asset_rc_ratio'], 0.2, atol=1e-6)

    # Limitations. compute_portfolio_vol matches weights to the covariance by position, not by
    # label; the risk table drops weights outside the covariance and needs every asset in it.
    reversed_weights = weights.iloc[::-1]
    assert not np.isclose(op.compute_portfolio_vol(covar=covar, weights=reversed_weights), vol)
    assert np.isclose(op.compute_portfolio_vol(covar=covar,
                                               weights=reversed_weights.reindex(covar.index)), vol)
    with_cash = pd.concat([weights, pd.Series({'Cash': 0.2})])
    pd.testing.assert_frame_equal(op.compute_portfolio_risk_contribution_outputs(
        weights=with_cash, clean_covar=covar), table)
    assert_raises(KeyError, op.compute_portfolio_risk_contribution_outputs,
                  weights=weights.drop('Gold'), clean_covar=covar)
    # A portfolio without risk has zero contributions and undefined (NaN) shares.
    empty = op.compute_portfolio_risk_contribution_outputs(weights=0.0 * weights,
                                                           clean_covar=covar)
    assert empty['risk contribution'].eq(0.0).all() and empty['asset_rc_ratio'].isna().all()
    # A hedge has a negative contribution although its weight is positive.
    hedge = euler_shares([0.8, 0.2], [[0.04, -0.012], [-0.012, 0.01]])
    assert hedge[1] < 0.0 and hedge[0] > 1.0 and np.isclose(hedge.sum(), 1.0)
    np.testing.assert_allclose(op.compute_portfolio_risk_contribution_outputs(
        weights=pd.Series([0.8, 0.2], index=['A', 'B']),
        clean_covar=pd.DataFrame([[0.04, -0.012], [-0.012, 0.01]], index=['A', 'B'],
                                 columns=['A', 'B']))['asset_rc_ratio'], hedge, rtol=1e-13)
    print('portfolio_risk_analytics: all page statements verified.')


def exhibit(path) -> dict:
    """Draw the page's figure: capital and risk shares of four portfolios on one covariance.

    Args:
        path: PNG file to write.

    Returns:
        The plotted table and the checks the figure illustrates.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    covar = factor_snapshot(FACTOR_CORR).get_y_covar()
    sigma = covar.to_numpy()
    asset_vols = np.sqrt(np.diag(sigma))
    portfolios = risk_based_portfolios(covar)
    shares = {name: op.compute_portfolio_risk_contribution_outputs(
        weights=w, clean_covar=covar)['asset_rc_ratio'] for name, w in portfolios.items()}
    table = pd.DataFrame(index=TICKERS)
    for name, w in portfolios.items():
        table[f'{name}: weight'] = w.to_numpy()
        table[f'{name}: risk share'] = shares[name].to_numpy()

    ink, muted, grid, surface = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
    colours = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
    plt.rcParams.update({'font.size': 11, 'axes.edgecolor': grid, 'axes.labelcolor': muted,
                         'xtick.color': muted, 'ytick.color': muted})
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.0, 4.4), facecolor=surface)
    names = list(portfolios)
    labels = ['Equal\nweight', 'Minimum\nvariance', 'Equal risk\ncontribution',
              'Maximum\ndiversification']
    x = np.arange(len(names))
    panels = ((left, {name: portfolios[name].to_numpy() for name in names}, 'Capital weights'),
              (right, {name: shares[name].to_numpy() for name in names}, 'Shares of risk'))
    for axis, values, title in panels:
        bottom = np.zeros(len(names))
        for k, (ticker, colour) in enumerate(zip(TICKERS, colours)):
            heights = np.array([values[name][k] for name in names])
            axis.bar(x, heights, 0.62, bottom=bottom, color=colour, edgecolor=surface,
                     linewidth=0.8, label=ticker)
            for position, low, height in zip(x, bottom, heights):
                if height >= 0.07:
                    axis.text(position, low + height / 2, f'{height:.0%}', ha='center',
                              va='center', color=ink, fontsize=9)
            bottom += heights
        axis.set_title(title, loc='left', color=ink)
        axis.set_facecolor(surface)
        axis.set_xticks(x, labels, fontsize=10)
        axis.set_ylim(0.0, 1.0)
        axis.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
        axis.grid(axis='y', color=grid, linewidth=0.8)
        axis.set_axisbelow(True)
        for side in ('top', 'right'):
            axis.spines[side].set_visible(False)
    handles, legend_labels = left.get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='lower center', ncol=len(TICKERS), frameon=False,
               fontsize=10, labelcolor=ink)
    fig.tight_layout(rect=(0.0, 0.07, 1.0, 1.0))
    fig.savefig(path, dpi=150, facecolor=surface)
    plt.close(fig)

    erc = shares['Equal risk contribution'].to_numpy()
    mv = portfolios['Minimum variance'].to_numpy()
    mdp = portfolios['Maximum diversification'].to_numpy()
    checks = {
        'capital_weights_sum_to_one': bool(all(np.isclose(w.sum(), 1.0, atol=1e-8)
                                               for w in portfolios.values())),
        'risk_shares_sum_to_one': bool(all(np.isclose(s.sum(), 1.0, atol=1e-12)
                                           for s in shares.values())),
        'erc_risk_shares_equal': bool(np.allclose(erc, 1.0 / len(TICKERS), atol=1e-6)),
        'min_variance_risk_shares_equal_weights': bool(np.allclose(
            shares['Minimum variance'], mv, atol=1e-5)),
        'max_diversification_risk_shares_equal_volatility_shares': bool(np.allclose(
            shares['Maximum diversification'], mdp * asset_vols / (mdp @ asset_vols),
            atol=1e-4)),
        'package_shares_match_euler_formula': bool(all(
            np.allclose(shares[name], euler_shares(w, sigma), atol=1e-12)
            for name, w in portfolios.items())),
    }
    return {'table': table, 'checks': checks}


if __name__ == '__main__':
    main()
