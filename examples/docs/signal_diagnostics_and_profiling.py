"""Canonical script of docs/signal_diagnostics_and_profiling.md.

The page's Python blocks are excerpts of this file and run here in the same order; every number
and property the page states is asserted after them against a reference computed a different
way: the rank correlation of a bivariate normal pair in closed form, truncated-normal means of
the quantile baskets, per-date rank correlations and basket returns recomputed from the panels
with pandas, the qis estimator called directly, and the Cauchy-Schwarz solution of a
tracking-error cap. The synthetic universe is drawn from a fixed seed with the Cholesky rule, so
it is the same on every platform. The script runs offline after ``pip install optimalportfolios``
and needs no data file:

    python -m examples.docs.signal_diagnostics_and_profiling

``exhibit`` draws the page's figure; ``tools/docs_analytics/teaching.py`` calls it with the
constants below and records their values.
"""
import inspect
import logging
import math
from statistics import NormalDist

import numpy as np
import pandas as pd
import qis

import optimalportfolios as op
import optimalportfolios.alphas as alphas
import optimalportfolios.alphas.signal_diagnostics as diagnostics_module
import optimalportfolios.alphas.signals as signals

SEED = 11
NOISE_SEED = 12
N_ASSETS = 500
MONTHS = 240
FIRST_DATE = '2005-12-31'
RHO = 0.1  # correlation of each score with the shock of the next month's return
MU = 0.005  # monthly log drift of every asset
VOL = 0.05  # monthly volatility of the asset-specific shock
MARKET_VOL = 0.04  # monthly volatility of the common market factor
QUANTILES = 5
ROLLING_MONTHS = 12


def synthetic_panel() -> tuple:
    """Monthly log returns, prices and scores of the page's synthetic universe."""
    rng = np.random.default_rng(SEED)
    dates = pd.date_range(FIRST_DATE, periods=MONTHS + 1, freq='ME')
    tickers = [f'A{i:03d}' for i in range(N_ASSETS)]
    # Each (shock, score) pair is standard normal with correlation RHO: the Cholesky rule.
    pairs = rng.standard_normal(((MONTHS + 1) * N_ASSETS, 2)) @ np.linalg.cholesky(
        np.array([[1.0, RHO], [RHO, 1.0]])).T
    shocks, signal = (pairs[:, j].reshape(MONTHS + 1, N_ASSETS) for j in (0, 1))
    market = MARKET_VOL * rng.standard_normal((MONTHS, 1))
    # The score dated t is paired with the shock of the return over the following month;
    # the score at the last date belongs to a month beyond the sample.
    log_returns = pd.DataFrame(MU + market + VOL * shocks[:-1], index=dates[1:],
                               columns=tickers)
    prices = 100.0 * np.exp(pd.concat([pd.DataFrame(0.0, index=dates[:1], columns=tickers),
                                       log_returns]).cumsum())
    scores = pd.DataFrame(signal, index=dates, columns=tickers)
    return log_returns, prices, scores


def gaussian_rank_ic(rho: float) -> float:
    """Spearman correlation (6/pi) arcsin(rho/2) of a bivariate normal pair of correlation rho."""
    return 6.0 / math.pi * math.asin(rho / 2.0)


def bucket_means(count: int) -> np.ndarray:
    """Mean of a standard normal in each of ``count`` equal-probability buckets, top first."""
    normal = NormalDist()
    edges = [math.inf] + [normal.inv_cdf(1.0 - k / count) for k in range(1, count)] + [-math.inf]
    density = [0.0 if math.isinf(edge) else normal.pdf(edge) for edge in edges]
    return np.array([count * (density[k + 1] - density[k]) for k in range(count)])


def spearman_by_date(signal: pd.DataFrame, returns: pd.DataFrame) -> pd.Series:
    """Per-date Spearman correlation of two aligned panels, from pandas ranks."""
    return signal.rank(axis=1).corrwith(returns.rank(axis=1), axis=1)


def top_members(scores: pd.DataFrame, count: int) -> pd.DataFrame:
    """Boolean panel of the ``count`` highest scores on each date, from an explicit sort."""
    order = np.argsort(-scores.to_numpy(), axis=1, kind='stable')[:, :count]
    members = np.zeros(scores.shape, dtype=bool)
    np.put_along_axis(members, order, True, axis=1)
    return pd.DataFrame(members, index=scores.index, columns=scores.columns)


def basket_simple_returns(prices: pd.DataFrame, members: pd.DataFrame) -> pd.Series:
    """Equal-weight simple return over each period of the assets selected at its start."""
    growth = prices / prices.shift(1) - 1.0
    held = members.shift(1).astype('boolean').fillna(False).astype(bool)
    return growth.where(held).mean(axis=1).iloc[1:]


def assert_raises(error: type, function, *args, **kwargs) -> None:
    """Fail unless ``function(*args, **kwargs)`` raises ``error``."""
    try:
        function(*args, **kwargs)
    except error:
        return
    raise AssertionError(f'expected {error.__name__}')


def defaults(function) -> dict:
    """Return the default value of every parameter of ``function`` that has one."""
    return {name: parameter.default
            for name, parameter in inspect.signature(function).parameters.items()
            if parameter.default is not inspect.Parameter.empty}


def main() -> None:
    """Run the page's blocks in order and assert every number and property it states."""
    log_returns, prices, scores = synthetic_panel()

    # 500 assets on 241 month ends, 2005-12-31 to 2025-12-31: 240 monthly returns and a score
    # at every month end. Across all 120,000 pairs the score is correlated RHO with the shock,
    # which the cross-sectional demeaning below recovers without knowing the market factor.
    assert prices.shape == (MONTHS + 1, N_ASSETS) and scores.shape == prices.shape
    assert (prices.index[0], prices.index[-1]) == (pd.Timestamp('2005-12-31'),
                                                   pd.Timestamp('2025-12-31'))
    np.testing.assert_allclose(np.log(prices / prices.shift(1)).iloc[1:], log_returns,
                               rtol=0, atol=1e-12)
    demeaned = log_returns.sub(log_returns.mean(axis=1), axis=0)
    lagged = scores.shift(1).loc[log_returns.index]
    pooled = np.corrcoef(lagged.to_numpy().ravel(), demeaned.to_numpy().ravel())[0, 1]
    assert abs(pooled - RHO) < 0.01

    diagnostics = alphas.run_signal_diagnostics(
        asset_returns_dict={'ME': log_returns}, signal=scores, horizons=(1, 3))
    ic_summary = qis.estimate_ic_ir(diagnostics)
    print(diagnostics.pooled_universe[['n', 'beta', 'IC_pearson', 'IC_spearman']].round(4))
    print(ic_summary[['n_dates', 'mean_IC', 'std_IC', 't_stat', 'hit_rate']].round(4))

    # The wrapper adds nothing to the numbers: qis called directly gives the same result.
    direct = qis.estimate_signal_diagnostics(asset_returns_dict={'ME': log_returns},
                                             signal=scores, horizons=(1, 3))
    pd.testing.assert_frame_equal(diagnostics.pooled_universe, direct.pooled_universe)
    assert diagnostics.horizon_labels == ['1', '3'] and not diagnostics.fit_intercept
    # Timing: the pair dated t holds the score of t-1 and the log return over (t-1, t].
    pairs = diagnostics.pairs['1']
    assert len(pairs) == MONTHS * N_ASSETS
    np.testing.assert_array_equal(pairs['z'], lagged.to_numpy().ravel())
    np.testing.assert_array_equal(pairs['r'], log_returns.to_numpy().ravel())
    np.testing.assert_array_equal(pairs['date'], np.repeat(log_returns.index, N_ASSETS))
    # Three months: every third date, the sum of the next three log returns, no overlap.
    three = diagnostics.pairs['3']
    starts = log_returns.index[::3]
    assert three['date'].nunique() == 80 and (three['date'].unique() == starts).all()
    summed = log_returns.rolling(3).sum().shift(-2).loc[starts]
    np.testing.assert_allclose(three['r'], summed.to_numpy().ravel(), rtol=0, atol=1e-14)
    # The per-date rank IC is the Spearman correlation of the lagged score with the return;
    # the market factor and the per-date normalisation of qis leave every rank unchanged.
    ic = qis.compute_ic_timeseries(diagnostics)['1']['IC']
    np.testing.assert_allclose(ic, spearman_by_date(lagged, log_returns), rtol=0, atol=1e-12)
    np.testing.assert_allclose(ic, spearman_by_date(lagged, demeaned), rtol=0, atol=1e-12)
    # Mean rank IC 0.0966 against (6/pi) arcsin(RHO/2) = 0.0955, less than half a standard
    # error s(IC)/sqrt(T) = 0.0029 away; its t-statistic mean/s(IC) sqrt(T) exceeds 30.
    row = ic_summary.loc['1']
    standard_error = ic.std(ddof=1) / math.sqrt(len(ic))
    population = gaussian_rank_ic(RHO)
    assert round(population, 4) == 0.0955 and round(row['mean_IC'], 4) == 0.0966
    assert abs(row['mean_IC'] - ic.mean()) < 1e-15 and row['n_dates'] == MONTHS
    assert round(standard_error, 4) == 0.0029
    assert abs(row['mean_IC'] - population) < 0.5 * standard_error
    assert abs(row['t_stat'] - ic.mean() / ic.std(ddof=1) * math.sqrt(len(ic))) < 1e-9
    assert row['t_stat'] > 30
    # The monthly IC has s(IC) = 0.0457, close to 1/sqrt(n - 1) = 0.0448 for a constant IC; it
    # is negative in 2 of the 240 months (hit rate 99%), and its 12-month mean stays between
    # 0.06 and 0.14.
    assert round(row['std_IC'], 4) == 0.0457 and round(1 / math.sqrt(N_ASSETS - 1), 4) == 0.0448
    assert abs(row['std_IC'] * math.sqrt(N_ASSETS - 1) - 1.0) < 0.05
    assert (ic < 0).sum() == 2 and abs(row['hit_rate'] - 238 / 240) < 1e-12
    assert round(row['hit_rate'], 2) == 0.99
    rolling = ic.rolling(ROLLING_MONTHS).mean().dropna()
    assert 0.06 < rolling.min() and rolling.max() < 0.14
    # At three months the one-month score is diluted: 0.051 against (6/pi) arcsin(RHO/(2 sqrt 3))
    # = 0.055, within two standard errors.
    ic_three = qis.compute_ic_timeseries(diagnostics)['3']['IC']
    three_error = ic_three.std(ddof=1) / math.sqrt(len(ic_three))
    three_population = gaussian_rank_ic(RHO / math.sqrt(3.0))
    assert round(ic_summary.loc['3', 'mean_IC'], 3) == 0.051
    assert round(three_population, 3) == 0.055
    assert abs(ic_summary.loc['3', 'mean_IC'] - three_population) < 2 * three_error
    # The pooled Pearson IC and slope are both 0.102, near RHO: the score has unit dispersion
    # and qis scales the returns to unit cross-sectional dispersion.
    pooled_row = diagnostics.pooled_universe.loc['1']
    assert round(pooled_row['IC_pearson'], 3) == 0.102 and round(pooled_row['beta'], 3) == 0.102
    assert abs(pooled_row['beta'] - RHO) < 2 * pooled_row['se']

    window = qis.TimePeriod(prices.index[0], prices.index[-2])
    top = alphas.backtest_alpha_rank_portfolio(
        prices=prices, alpha_scores={'Top quintile': scores}, quantile=0.2,
        rebalancing_freq='ME', time_period=window)
    print(alphas.compute_alpha_rank_analysis_table(top).round(3))

    # Two legs, the strategy and the equal-weight benchmark last, without costs; the last
    # target is formed at the penultimate month end and held into the last month.
    legs = top.portfolio_datas
    assert [leg.ticker for leg in legs] == ['Top quintile', 'Equal Weight']
    pd.testing.assert_series_equal(top.benchmark_prices.iloc[:, 0], legs[-1].get_portfolio_nav(),
                                   check_names=False)
    assert all(np.allclose(leg.realized_costs, 0.0) for leg in legs)
    assert legs[0].get_portfolio_nav().index[-1] == prices.index[-1]
    rebalancing = legs[0].is_rebalancing
    assert rebalancing[rebalancing].index[-1] == prices.index[-2] and rebalancing.sum() == MONTHS
    # Held as units and rebalanced monthly, the leg earns the equal-weight simple return of the
    # 100 highest scores at the previous month end, and the benchmark that of all 500.
    top_nav = legs[0].get_portfolio_nav()
    np.testing.assert_allclose(top_nav.pct_change().iloc[1:],
                               basket_simple_returns(prices, top_members(scores, 100)),
                               rtol=0, atol=1e-12)
    np.testing.assert_allclose(legs[1].get_portfolio_nav().pct_change().iloc[1:],
                               (prices / prices.shift(1) - 1.0).mean(axis=1).iloc[1:],
                               rtol=0, atol=1e-12)
    # The analysis table: its columns, about 19% a year for the top quintile against 9% for the
    # benchmark, and turnover near 12 x 2 x (1 - 0.2) = 19.2 per year, since independent
    # monthly scores renew the basket.
    table = alphas.compute_alpha_rank_analysis_table(top)
    assert list(table.columns) == ['Return p.a.', 'Vol', 'Sharpe', 'Max DD', 'Turnover p.a.']
    assert list(table.index) == ['Top quintile', 'Equal Weight']
    assert (round(table.loc['Top quintile', 'Return p.a.'], 2),
            round(table.loc['Equal Weight', 'Return p.a.'], 2)) == (0.19, 0.09)
    # Turnover is the value of the units traded over NAV, summed and divided by the years
    # between the first and the last date.
    traded = (legs[0].units.diff().abs() * prices).sum(axis=1) / top_nav
    years = (prices.index[-1] - prices.index[0]).days / 365.25
    assert abs(table.loc['Top quintile', 'Turnover p.a.'] - traded.sum() / years) < 1e-9
    assert abs(table.loc['Top quintile', 'Turnover p.a.'] - 12 * 2 * (1 - 0.2)) < 0.5
    assert round(table.loc['Top quintile', 'Turnover p.a.']) == 19
    # The ranking rule: ceil(quantile x eligible) assets, ties by column order, and a mask of
    # non-missing values only, so an infinite score with a negative price is still selected.
    date = prices.index[-1]
    eight = prices.columns[:8]
    ties = pd.DataFrame([[3.0, 3.0, 2.0, 1.0, 0.0, -1.0, -2.0, -3.0]], index=[date],
                        columns=eight)
    few = prices.loc[[date], eight]
    top_eighth = alphas.compute_top_quantile_equal_weights(ties, few, quantile=0.125)
    np.testing.assert_array_equal(top_eighth.iloc[0], [1, 0, 0, 0, 0, 0, 0, 0])
    np.testing.assert_array_equal(alphas.compute_top_quantile_equal_weights(
        ties, few, quantile=0.3).iloc[0], [1 / 3] * 3 + [0] * 5)
    infinite = ties.copy()
    infinite.iloc[0, -1] = np.inf
    negative = few.copy()
    negative.iloc[0, -1] = -1.0
    assert alphas.compute_top_quantile_equal_weights(infinite, negative,
                                                     quantile=0.125).iloc[0, -1] == 1.0
    # A missing score excludes the asset; quantile=1.0 then holds the seven scored assets,
    # while the benchmark would hold all eight priced ones; a quantile outside (0, 1] raises.
    partial = ties.copy()
    partial.iloc[0, 0] = np.nan
    everything = alphas.compute_top_quantile_equal_weights(partial, few, quantile=1.0)
    np.testing.assert_array_equal(everything.iloc[0], [0.0] + [1 / 7] * 7)
    unscored = scores.iloc[:3, :8].copy()
    unscored.iloc[:, 0] = np.nan
    all_scored = alphas.backtest_alpha_rank_portfolio(prices=prices.iloc[:3, :8],
                                                      alpha_scores=unscored, quantile=1.0,
                                                      rebalancing_freq='ME')
    first_weights = [leg.weights.iloc[0] for leg in all_scored.portfolio_datas]
    np.testing.assert_allclose(first_weights[0], [0.0] + [1 / 7] * 7, rtol=0, atol=1e-12)
    np.testing.assert_allclose(first_weights[1], [1 / 8] * 8, rtol=0, atol=1e-12)
    for quantile in (0.0, 1.5):
        assert_raises(ValueError, alphas.compute_top_quantile_equal_weights, ties, few,
                      quantile=quantile)

    tops = [alphas.compute_top_quantile_equal_weights(scores, prices, quantile=k / QUANTILES) > 0
            for k in range(1, QUANTILES + 1)]
    panels = {'Q1': scores.where(tops[0])}
    panels.update({f'Q{k + 1}': scores.where(tops[k] & ~tops[k - 1])
                   for k in range(1, QUANTILES)})
    quintiles = alphas.backtest_alpha_rank_portfolio(
        prices=prices, alpha_scores=panels, quantile=1.0, rebalancing_freq='ME',
        time_period=window)
    navs = pd.concat([leg.get_portfolio_nav() for leg in quintiles.portfolio_datas], axis=1)
    print(navs.iloc[-1].round(1))

    # Five disjoint baskets of 100 assets on every date, covering the universe; Q1 is the top
    # quintile leg above, and the equal-weight benchmark comes last.
    members = {name: panel.notna() for name, panel in panels.items()}
    assert all((member.sum(axis=1) == N_ASSETS // QUANTILES).all() for member in members.values())
    assert (sum(member.astype(int) for member in members.values()) == 1).all().all()
    assert list(navs.columns) == [*panels, 'Equal Weight']
    np.testing.assert_allclose(navs['Q1'], top_nav, rtol=0, atol=1e-10)
    # Monotone ordering with a wide margin: mean monthly returns and final values fall from Q1
    # to Q5, and each adjacent gap exceeds five standard errors of the monthly difference.
    monthly = navs[list(panels)].pct_change().iloc[1:]
    gaps = -monthly.diff(axis=1).iloc[:, 1:]
    assert (gaps.mean() > 5 * gaps.std(ddof=1) / math.sqrt(MONTHS)).all()
    assert (navs[list(panels)].iloc[-1].diff().iloc[1:] < 0).all()
    assert round(navs['Q1'].iloc[-1], -1) == 3060 and round(navs['Q5'].iloc[-1]) == 103
    # Proposition 2 on log returns: the Q1-Q5 spread averages 1.41% a month against
    # 2 RHO VOL K phi(Phi^-1(0.8)) = 1.40%, and each basket's excess over the cross-sectional
    # mean is within two standard errors of RHO VOL times its truncated-normal mean.
    baskets = pd.DataFrame({name: log_returns.where(member.shift(1).loc[log_returns.index]
                                                    .astype(bool)).mean(axis=1)
                            for name, member in members.items()})
    excess = baskets.sub(log_returns.mean(axis=1), axis=0)
    expected = RHO * VOL * bucket_means(QUANTILES)
    # For K = 5 the bucket means are 1.400, 0.532, 0, -0.532, -1.400, and the spread is
    # 2 K phi(c_1) = 2.80 times RHO VOL.
    means = bucket_means(QUANTILES)
    assert [round(value, 3) for value in means[[0, 1, 3, 4]]] == [1.4, 0.532, -0.532, -1.4]
    assert abs(means[2]) < 1e-12 and (np.diff(means) < 0).all()
    assert round(2 * QUANTILES * NormalDist().pdf(NormalDist().inv_cdf(0.8)), 2) == 2.80
    assert (np.abs(excess.mean() - expected) < 2 * excess.std(ddof=1) / math.sqrt(MONTHS)).all()
    spread = baskets['Q1'] - baskets['Q5']
    closed_form = 2 * RHO * VOL * QUANTILES * NormalDist().pdf(NormalDist().inv_cdf(0.8))
    assert round(100 * closed_form, 2) == 1.40 and round(100 * spread.mean(), 2) == 1.41
    assert abs(spread.mean() - closed_form) < 2 * spread.std(ddof=1) / math.sqrt(MONTHS)

    quarterly = alphas.profile_classic_momentum(prices=prices, quantile=0.2)
    realised = quarterly.portfolio_datas[0].weights
    print(realised.loc['2024-12-31':'2025-03-31'].max(axis=1).round(5))

    # The adapter computes the classic momentum score and hands it to the core: the same NAV.
    assert [leg.ticker for leg in quarterly.portfolio_datas] == [
        alphas.ProfileSignal.CLASSIC_MOMENTUM.value, 'Equal Weight']
    momentum_score, _ = signals.compute_classic_momentum_alpha(prices=prices, returns_freq='ME')
    routed = alphas.backtest_alpha_rank_portfolio(prices=prices, alpha_scores=momentum_score,
                                                  quantile=0.2, strategy_ticker='classic_momentum')
    pd.testing.assert_series_equal(quarterly.portfolio_datas[0].get_portfolio_nav(),
                                   routed.portfolio_datas[0].get_portfolio_nav())
    # Quarter ends are the only trade dates; at each the realised weights are the equal targets
    # 1/100, and between them the units are held, so the weights drift with prices.
    held = quarterly.portfolio_datas[0]
    trades = held.is_rebalancing[held.is_rebalancing].index
    assert (trades == prices.index[prices.index.is_quarter_end]).all()
    invested = [date for date in trades if realised.loc[date].sum() > 0.5]
    assert np.allclose(realised.loc[invested].max(axis=1), 0.01, rtol=0, atol=1e-12)
    between = realised.loc[~realised.index.isin(trades) & (realised.index > invested[0])]
    assert (between.max(axis=1) > 0.0101).all()
    # The printed quarter: 1% at 2024-12-31, above 1.1% and 1.2% at the next two month ends,
    # and 1% again after the trade at 2025-03-31.
    shown = realised.loc['2024-12-31':'2025-03-31'].max(axis=1)
    assert abs(shown.iloc[0] - 0.01) < 1e-12 and abs(shown.iloc[-1] - 0.01) < 1e-12
    assert 0.011 < shown.iloc[1] < 0.012 < shown.iloc[2] < 0.013
    # So the NAV grows over each quarter by the mean price ratio of the selected assets.
    nav = held.get_portfolio_nav()
    for start, end in zip(invested[:-1], invested[1:]):
        chosen = realised.loc[start] > 0
        ratio = (prices.loc[end, chosen] / prices.loc[start, chosen]).mean()
        assert abs(nav[end] / nav[start] - ratio) < 1e-12
    # The first 13 month ends have no momentum score: the leg holds cash until the first
    # quarter end with one, 2007-03-31.
    assert momentum_score.iloc[:13].isna().all().all() and momentum_score.iloc[13].notna().all()
    assert invested[0] == pd.Timestamp('2007-03-31') and (nav.loc[:'2007-03-31'] == 100.0).all()

    noise = pd.DataFrame(np.random.default_rng(NOISE_SEED).standard_normal(scores.shape),
                         index=scores.index, columns=scores.columns)
    data = alphas.AlphasData(alpha_scores=(scores + noise) / np.sqrt(2.0),
                             momentum_score=scores, beta_score=noise)
    components = alphas.run_signal_diagnostics_per_component(
        asset_returns_dict={'ME': log_returns}, alphas_data=data, horizons=(1,))
    comparison = alphas.compare_signal_diagnostics(components, horizon='1')
    print(comparison[['n', 'beta', 'IC_pearson', 'IC_spearman']].round(4))

    # The populated score fields, in the module's fixed order; unpopulated fields are skipped.
    order = ['alpha_scores', 'momentum_score', 'beta_score']
    assert list(alphas.signal_diagnostics_panel(data)) == order == list(components)
    assert list(alphas.signal_diagnostics_panel(data, components=['beta_score'])) == ['beta_score']
    assert list(comparison.index) == order and comparison.index.name == 'signal'
    # Each component equals the single-panel call on that field; the default field is
    # alpha_scores; an unknown field raises AttributeError and an empty one ValueError.
    returns_dict = {'ME': log_returns}
    for name in order:
        single = alphas.run_signal_diagnostics(returns_dict, data, horizons=(1,),
                                               signal_attribute=name)
        pd.testing.assert_frame_equal(single.pooled_universe, components[name].pooled_universe)
    default_field = alphas.run_signal_diagnostics(returns_dict, data, horizons=(1,))
    pd.testing.assert_frame_equal(default_field.pooled_universe,
                                  components['alpha_scores'].pooled_universe)
    assert_raises(AttributeError, alphas.run_signal_diagnostics, returns_dict, data,
                  signal_attribute='momentum_scores')
    assert_raises(ValueError, alphas.run_signal_diagnostics, returns_dict, data,
                  signal_attribute='managers_scores')
    # The blend carries the Pearson IC RHO/sqrt(2): its rank IC 0.069 is within two standard
    # errors of (6/pi) arcsin(RHO/(2 sqrt 2)) = 0.068; the noise component's is within two of 0.
    references = {'alpha_scores': gaussian_rank_ic(RHO / math.sqrt(2.0)),
                  'momentum_score': gaussian_rank_ic(RHO), 'beta_score': 0.0}
    for name, reference in references.items():
        series = qis.compute_ic_timeseries(components[name])['1']['IC']
        assert abs(series.mean() - reference) < 2 * series.std(ddof=1) / math.sqrt(len(series))
    assert round(references['alpha_scores'], 3) == 0.068
    assert round(comparison.loc['alpha_scores', 'IC_spearman'], 3) == 0.069
    assert abs(comparison.loc['beta_score', 'IC_spearman']) < 0.005
    pd.testing.assert_series_equal(comparison.loc['momentum_score'], pooled_row,
                                   check_names=False, check_dtype=False)
    assert round(pooled_row['IC_spearman'], 4) == 0.0968
    # With every score field populated, the panel follows the module's fixed order.
    score_fields = ['alpha_scores', 'momentum_score', 'momentum_cluster_score', 'beta_score',
                    'beta_cluster_score', 'residual_momentum_score',
                    'residual_momentum_cluster_score', 'managers_scores']
    full = alphas.AlphasData(**{field: scores for field in reversed(score_fields)})
    assert list(alphas.signal_diagnostics_panel(full)) == score_fields
    # The module's two IC-ratio helpers are not exported by the alpha layer.
    for helper in ('compare_signal_ic_ir', 'build_signal_diagnostics_table'):
        assert callable(getattr(diagnostics_module, helper)) and not hasattr(alphas, helper)
    # Without a horizon the rows keep a (signal, horizon) index; no results give an empty frame.
    assert alphas.compare_signal_diagnostics(components).index.names == ['signal', 'horizon']
    assert alphas.compare_signal_diagnostics({}).empty
    # Pitfall: horizon labels are strings; an integer horizon matches none, logs a warning per
    # signal and returns an empty frame.
    records = []
    logger = logging.getLogger(diagnostics_module.__name__)
    handler = logging.Handler()
    handler.emit = records.append
    propagate, logger.propagate = logger.propagate, False
    logger.addHandler(handler)
    try:
        assert alphas.compare_signal_diagnostics(components, horizon=1).empty
    finally:
        logger.removeHandler(handler)
        logger.propagate = propagate
    assert len(records) == len(order)
    assert all(record.levelno == logging.WARNING for record in records)

    cubed = alphas.run_signal_diagnostics(
        asset_returns_dict={'ME': log_returns}, signal=scores ** 3, horizons=(1,))
    print(pd.concat({'score': diagnostics.pooled_universe.loc['1'],
                     'cubed score': cubed.pooled_universe.loc['1']}, axis=1)
          .loc[['beta', 'IC_pearson', 'IC_spearman']].round(4))

    # Insight: the cube keeps every rank, so every monthly rank IC and every quintile basket
    # is unchanged; the Pearson IC falls to 3 RHO/sqrt(15) = 0.077 (0.081 here) and the slope
    # to 3 RHO/15 = 0.02, within two standard errors.
    cube_row = cubed.pooled_universe.loc['1']
    np.testing.assert_array_equal(qis.compute_ic_timeseries(cubed)['1']['IC'], ic)
    assert cube_row['IC_spearman'] == pooled_row['IC_spearman']
    pd.testing.assert_frame_equal(
        alphas.compute_top_quantile_equal_weights(scores ** 3, prices, quantile=0.2),
        alphas.compute_top_quantile_equal_weights(scores, prices, quantile=0.2))
    assert abs(cube_row['IC_pearson'] - 3 * RHO / math.sqrt(15.0)) < 0.01
    assert round(3 * RHO / math.sqrt(15.0), 3) == 0.077 and round(3 / math.sqrt(15.0), 3) == 0.775
    assert abs(cube_row['beta'] - 3 * RHO / 15.0) < 2 * cube_row['se']
    assert (round(pooled_row['IC_pearson'], 2), round(cube_row['IC_pearson'], 2)) == (0.10, 0.08)
    assert (round(pooled_row['beta'], 2), round(cube_row['beta'], 2)) == (0.10, 0.02)

    probe = pd.Series([1.5, 1.0, 0.5, 0.2, -0.2, -0.5, -1.0, -1.5], index=list('ABCDEFGH'))
    covar = pd.DataFrame(np.diag(np.full(8, 0.2 ** 2)), index=probe.index, columns=probe.index)
    benchmark = pd.Series(1.0 / 8, index=probe.index)
    te_cap = op.Constraints(is_long_only=True, benchmark_weights=benchmark,
                            tracking_err_vol_constraint=0.02)
    views = {'score': probe, '10 x score': 10.0 * probe, 'cubed score': probe ** 3}
    active = pd.DataFrame({name: op.wrapper_maximise_alpha_over_tre(
        covar, alpha, benchmark, te_cap)[0] - benchmark for name, alpha in views.items()})
    print(active.round(4))

    # With a diagonal covariance and no binding bound, the active weights of the hard cap are
    # proportional to the demeaned alpha (Cauchy-Schwarz) and use the whole 2% budget: the same
    # for the score and ten times the score, different for the cube. The top-quantile targets
    # are the same two assets for all three. The default solver is CLARABEL.
    assert op.OptimiserConfig().solver == 'CLARABEL'
    assert te_cap.constraint_enforcement_type == op.ConstraintEnforcementType.FORCED_CONSTRAINTS
    flat = pd.DataFrame(100.0, index=[date], columns=probe.index)
    for name, alpha in views.items():
        demeaned_alpha = alpha - alpha.mean()
        closed = 0.02 / 0.2 * demeaned_alpha / np.linalg.norm(demeaned_alpha)
        assert np.abs(active[name] - closed).max() < 1e-5
        assert abs(0.2 * np.linalg.norm(active[name]) - 0.02) < 1e-6
        assert (benchmark + active[name]).min() > 0.05  # no bound binds
        targets = alphas.compute_top_quantile_equal_weights(alpha.to_frame(date).T, flat,
                                                            quantile=0.25)
        np.testing.assert_array_equal(targets.iloc[0], [0.5, 0.5, 0, 0, 0, 0, 0, 0])
    assert np.abs(active['score'] - active['10 x score']).max() < 1e-5
    assert np.abs(active['score'] - active['cubed score']).max() > 0.01
    # The two extreme assets carry 47% of the absolute active weight under the score and 75%
    # under the cube: 3/6.4 and 6.75/9.016.
    extremes = active.abs().iloc[[0, -1]].sum() / active.abs().sum()
    assert (round(extremes['score'], 2), round(extremes['cubed score'], 2)) == (0.47, 0.75)
    assert abs(extremes['score'] - 3.0 / 6.4) < 1e-4 and abs(extremes['cubed score']
                                                             - 6.75 / 9.016) < 1e-4

    # Implementation: the defaults the page states, and what the wrapper does not expose.
    assert defaults(alphas.run_signal_diagnostics) == {
        'group_data': None, 'horizons': (1, 2, 3, 6), 'signal_attribute': 'alpha_scores',
        'group_order': None, 'is_log_returns': True}
    qis_defaults = defaults(qis.estimate_signal_diagnostics)
    assert qis_defaults['horizons'] == (1, 3, 6) and qis_defaults['fit_intercept'] is False
    assert (qis_defaults['is_vol_normalised'], qis_defaults['min_obs_per_date'],
            qis_defaults['min_obs_per_group']) == (True, 5, 10)
    assert not {'fit_intercept', 'is_vol_normalised', 'min_obs_per_date',
                'min_obs_per_group'} & set(inspect.signature(alphas.run_signal_diagnostics)
                                           .parameters)
    assert defaults(qis.to_returns)['is_log_returns'] is False
    assert defaults(alphas.run_signal_diagnostics_per_component)['horizons'] == (1, 2, 3, 6)
    assert defaults(alphas.compare_signal_diagnostics) == {'horizon': None}
    core = defaults(alphas.backtest_alpha_rank_portfolio)
    assert core == {'quantile': 1.0 / 3.0, 'rebalancing_freq': 'QE', 'time_period': None,
                    'rebalancing_costs': None, 'instruments_carry': None,
                    'strategy_ticker': 'Top-quantile', 'benchmark_ticker': 'Equal Weight'}
    assert defaults(alphas.compute_top_quantile_equal_weights) == {'quantile': 1.0 / 3.0}
    assert defaults(alphas.compute_alpha_rank_analysis_table) == {'time_period': None,
                                                                  'perf_params': None}
    report = defaults(alphas.generate_alpha_profile_report)
    assert (report['regime_benchmark'], report['file_name'], report['local_path'],
            report['add_current_date']) == (None, 'alpha_profile_report', None, True)
    adapters = {alphas.profile_momentum: 'MOMENTUM',
                alphas.profile_classic_momentum: 'CLASSIC_MOMENTUM',
                alphas.profile_low_beta: 'LOW_BETA',
                alphas.profile_residual_momentum: 'RESIDUAL_MOMENTUM',
                alphas.profile_carry: 'CARRY'}
    assert [member.name for member in alphas.ProfileSignal] == list(adapters.values())
    # Every adapter defaults to monthly returns, a top third and quarter-end rebalancing, and
    # labels its leg with the value of its ProfileSignal member.
    few_prices = prices.iloc[:, :12]
    extra = {'benchmark_price': few_prices.mean(axis=1),
             'carry': pd.DataFrame(0.02, index=few_prices.index, columns=few_prices.columns)}
    for adapter, member in adapters.items():
        settings = defaults(adapter)
        assert (settings['returns_freq'], settings['quantile'], settings['rebalancing_freq'],
                settings['rebalancing_costs']) == ('ME', 1.0 / 3.0, 'QE', None)
        needed = {key: value for key, value in extra.items()
                  if key in inspect.signature(adapter).parameters}
        profile = adapter(prices=few_prices, **needed)
        assert [leg.ticker for leg in profile.portfolio_datas] == [
            alphas.ProfileSignal[member].value, 'Equal Weight']
    # profile_alpha_signals delegates a nonempty dictionary to the core; the core accepts an
    # empty one and returns the benchmark only.
    joint = alphas.profile_alpha_signals(prices=prices, alpha_scores={'Top quintile': scores},
                                         quantile=0.2, rebalancing_freq='ME', time_period=window)
    pd.testing.assert_series_equal(joint.portfolio_datas[0].get_portfolio_nav(), top_nav)
    assert_raises(ValueError, alphas.profile_alpha_signals, prices=prices, alpha_scores={})
    only = alphas.backtest_alpha_rank_portfolio(prices=few_prices, alpha_scores={})
    assert [leg.ticker for leg in only.portfolio_datas] == ['Equal Weight']
    # Limitations: a constant IC of 0.02 on 100 assets over 120 months has a t-statistic of
    # about 0.02 sqrt(99 x 120) = 2.2.
    assert round(0.02 * math.sqrt(99 * 120), 1) == 2.2
    print('signal_diagnostics_and_profiling: all page statements verified.')


def exhibit(path) -> dict:
    """Draw the page's figure: quintile NAVs and the rank IC with its rolling mean.

    Args:
        path: PNG file to write.

    Returns:
        The plotted table and the checks the figure illustrates.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    log_returns, prices, scores = synthetic_panel()
    diagnostics = alphas.run_signal_diagnostics(asset_returns_dict={'ME': log_returns},
                                                signal=scores, horizons=(1,))
    ic = qis.compute_ic_timeseries(diagnostics)['1']['IC']
    rolling = ic.rolling(ROLLING_MONTHS).mean()
    tops = [alphas.compute_top_quantile_equal_weights(scores, prices, quantile=k / QUANTILES) > 0
            for k in range(1, QUANTILES + 1)]
    panels = {'Q1': scores.where(tops[0])}
    panels.update({f'Q{k + 1}': scores.where(tops[k] & ~tops[k - 1])
                   for k in range(1, QUANTILES)})
    quintiles = alphas.backtest_alpha_rank_portfolio(
        prices=prices, alpha_scores=panels, quantile=1.0, rebalancing_freq='ME',
        time_period=qis.TimePeriod(prices.index[0], prices.index[-2]))
    navs = pd.concat([leg.get_portfolio_nav() for leg in quintiles.portfolio_datas[:-1]], axis=1)
    population = gaussian_rank_ic(RHO)
    table = navs.add_prefix('NAV ').join(pd.DataFrame({'rank IC': ic,
                                                       'rolling rank IC': rolling}))

    ink, muted, grid, surface = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
    colours = dict(zip(panels, ('#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4')))
    labels = {'Q1': 'Q1, top', 'Q2': 'Q2', 'Q3': 'Q3', 'Q4': 'Q4', 'Q5': 'Q5, bottom'}
    plt.rcParams.update({'font.size': 11, 'axes.edgecolor': grid, 'axes.labelcolor': muted,
                         'xtick.color': muted, 'ytick.color': muted})
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.0, 4.4), facecolor=surface)
    for name in panels:
        left.plot(navs.index, navs[name], color=colours[name], linewidth=1.6)
    left.set_yscale('log')
    left.yaxis.set_major_locator(matplotlib.ticker.FixedLocator([30, 100, 300, 1000, 3000]))
    left.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f'{v:,.0f}'))
    left.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    end = navs.index[-1] + pd.Timedelta(days=80)
    for name in panels:
        left.text(end, navs[name].iloc[-1], labels[name], color=ink, va='center', fontsize=10)
    left.set_title('Quintile portfolios sorted by the score', loc='left', color=ink)
    left.set_ylabel('NAV, log scale (start = 100)')
    left.set_xlim(navs.index[0], navs.index[-1] + pd.Timedelta(days=1500))
    right.bar(ic.index, ic, width=25, color='#b9b8b3', label='monthly rank IC')
    right.plot(rolling.index, rolling, color=muted, linewidth=2.0,
               label=f'{ROLLING_MONTHS}-month mean')
    right.axhline(population, color=ink, linestyle='--', linewidth=1.1,
                  label=f'population value {population:.4f}')
    right.axhline(0.0, color=muted, linewidth=0.8)
    right.set_title('Rank IC with next-month returns', loc='left', color=ink)
    right.set_ylabel('Spearman correlation')
    right.set_ylim(-0.06, 0.31)
    right.set_xlim(ic.index[0] - pd.Timedelta(days=60), ic.index[-1] + pd.Timedelta(days=60))
    right.legend(frameon=False, loc='upper left', fontsize=9, labelcolor=ink, ncol=2)
    years = [pd.Timestamp(f'{year}-01-01') for year in (2010, 2015, 2020, 2025)]
    for axis in (left, right):
        axis.set_facecolor(surface)
        axis.set_xticks(years)
        axis.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        axis.grid(axis='y', color=grid, linewidth=0.8)
        axis.set_axisbelow(True)
        for side in ('top', 'right'):
            axis.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=surface)
    plt.close(fig)

    monthly = navs.pct_change().iloc[1:]
    standard_error = ic.std(ddof=1) / math.sqrt(len(ic))
    checks = {
        'quintile_means_fall_from_top_to_bottom': bool((monthly.mean().diff().iloc[1:] < 0).all()),
        'final_navs_fall_from_top_to_bottom': bool((navs.iloc[-1].diff().iloc[1:] < 0).all()),
        'mean_rank_ic_within_one_standard_error': bool(
            abs(ic.mean() - population) < standard_error),
        'rank_ic_is_the_per_date_spearman_correlation': bool(np.allclose(
            ic, spearman_by_date(scores.shift(1).loc[log_returns.index], log_returns),
            rtol=0, atol=1e-12)),
        'rolling_ic_stays_positive': bool((rolling.dropna() > 0).all()),
    }
    return {'table': table, 'checks': checks}


if __name__ == '__main__':
    main()
