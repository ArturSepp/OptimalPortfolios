"""
the rank-and-select rule behind the alpha profiler, and the backtest built on it.

``compute_top_quantile_equal_weights`` is the whole selection model: no covariance, no
optimiser, just "hold the best ceil(q*n) of what is actually available today, equally". That
makes every one of its answers checkable by hand, and the cases below do exactly that -- a
score panel whose ranking is stated in the test, not recorded from a run. The parts worth
pinning are the ones that quietly produce a plausible-but-wrong weight: an asset with a score
but no price must not be held, ``ceil`` must round the basket up rather than down, and a date
with nothing available must give a flat zero row rather than a NaN one that the backtester
would silently carry forward.

The backtest and table layers are checked for the contracts their callers rely on -- that the
equal-weight benchmark is *last*, since ``portfolio_datas[-1]`` is how every caller finds it,
and that a rank strategy's turnover exceeds a buy-and-hold-weights benchmark's, which is the
one number ``compute_ra_perf_table`` does not carry.
"""
# packages
import numpy as np
import pandas as pd
import pytest
import qis
# optimalportfolios
from optimalportfolios.alphas.profile.core import (
    backtest_alpha_rank_portfolio,
    compute_alpha_rank_analysis_table,
    compute_top_quantile_equal_weights,
    generate_alpha_profile_report,
)
from optimalportfolios.tests.data.multiasset import load_multiasset_data

TICKERS = ['a', 'b', 'c', 'd', 'e', 'f']
DATES = pd.DatetimeIndex(['2024-01-31', '2024-02-29', '2024-03-31'])


def make_scores(values: list) -> pd.DataFrame:
    """A score panel over the six test tickers, one row per test date."""
    return pd.DataFrame(values, index=DATES, columns=TICKERS, dtype=float)


def make_prices(values: list = None) -> pd.DataFrame:
    """A price panel matching the score panel; all assets tradable unless overridden."""
    if values is None:
        values = [[100.0] * len(TICKERS)] * len(DATES)
    return pd.DataFrame(values, index=DATES, columns=TICKERS, dtype=float)


# --------------------------------------------------------------------------- #
# compute_top_quantile_equal_weights
# --------------------------------------------------------------------------- #
def test_top_quantile_holds_the_best_third_equally_weighted() -> None:
    """Six assets at q=1/3 hold the two highest scores at 0.5 each and nothing else."""
    scores = make_scores([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]] * 3)
    weights = compute_top_quantile_equal_weights(alpha_scores=scores, prices=make_prices())
    expected = pd.Series([0.0, 0.0, 0.0, 0.0, 0.5, 0.5], index=TICKERS)
    for date in DATES:
        pd.testing.assert_series_equal(weights.loc[date, :], expected, check_names=False)


def test_top_quantile_rounds_the_basket_up_not_down() -> None:
    """ceil(1/3 * 5) is 2, so five available assets hold two -- never one."""
    scores = make_scores([[1.0, 2.0, 3.0, 4.0, 5.0, np.nan]] * 3)
    weights = compute_top_quantile_equal_weights(alpha_scores=scores, prices=make_prices())
    assert (weights > 0.0).sum(axis=1).eq(2).all()


def test_a_score_without_a_price_is_not_investable() -> None:
    """The top-scoring asset is excluded on the date its price is missing."""
    scores = make_scores([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]] * 3)
    prices = make_prices()
    prices.loc[DATES[1], 'f'] = np.nan             # best score, but not trading that date
    weights = compute_top_quantile_equal_weights(alpha_scores=scores, prices=prices)
    assert weights.loc[DATES[1], 'f'] == 0.0
    assert weights.loc[DATES[0], 'f'] == 0.5       # unaffected on the other dates
    assert weights.loc[DATES[1], :].sum() == pytest.approx(1.0)


def test_a_date_with_nothing_available_is_flat_rather_than_nan() -> None:
    """A fully NaN score row gives zero weights, not NaNs the backtester would carry."""
    scores = make_scores([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]] * 3)
    scores.loc[DATES[0], :] = np.nan
    weights = compute_top_quantile_equal_weights(alpha_scores=scores, prices=make_prices())
    assert weights.loc[DATES[0], :].eq(0.0).all()
    assert not weights.isna().any().any()


def test_quantile_of_one_reduces_to_equal_weight_all() -> None:
    """q=1 keeps every available asset, which is the benchmark rule."""
    scores = make_scores([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]] * 3)
    weights = compute_top_quantile_equal_weights(alpha_scores=scores, prices=make_prices(),
                                                 quantile=1.0)
    assert weights.eq(1.0 / len(TICKERS)).all().all()


def test_scores_are_reindexed_onto_the_price_columns() -> None:
    """A score panel in a different column order is aligned to prices, not zipped positionally."""
    scores = make_scores([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]] * 3)
    weights = compute_top_quantile_equal_weights(alpha_scores=scores[TICKERS[::-1]],
                                                 prices=make_prices())
    assert list(weights.columns) == TICKERS
    assert weights.loc[DATES[0], ['e', 'f']].eq(0.5).all()


def test_a_score_column_missing_entirely_is_simply_never_held() -> None:
    """Reindexing a short score panel introduces NaNs, which exclude rather than raise."""
    scores = make_scores([[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]] * 3).drop(columns=['f'])
    weights = compute_top_quantile_equal_weights(alpha_scores=scores, prices=make_prices())
    assert weights['f'].eq(0.0).all()
    assert weights.loc[DATES[0], ['d', 'e']].eq(0.5).all()


@pytest.mark.parametrize('quantile', [0.0, -0.1, 1.5])
def test_a_quantile_outside_the_unit_interval_raises(quantile: float) -> None:
    """The selection fraction must lie in (0, 1]; anything else is a caller error."""
    with pytest.raises(ValueError, match='quantile must lie'):
        compute_top_quantile_equal_weights(alpha_scores=make_scores([[1.0] * 6] * 3),
                                           prices=make_prices(), quantile=quantile)


# --------------------------------------------------------------------------- #
# backtest_alpha_rank_portfolio and the analysis table
# --------------------------------------------------------------------------- #
@pytest.fixture(scope='module')
def multiasset() -> tuple:
    """The committed offline universe, trimmed to a decade of monthly prices."""
    data = load_multiasset_data()
    prices = data.prices.loc['2010':'2019', :]
    return prices, data.group_data


@pytest.fixture(scope='module')
def momentum_scores(multiasset: tuple) -> pd.DataFrame:
    """A trailing 12-month return panel: a real ranking, computed without look-ahead."""
    prices, _ = multiasset
    return prices.pct_change(12).shift(1)


def test_the_equal_weight_benchmark_is_the_last_leg(multiasset: tuple,
                                                    momentum_scores: pd.DataFrame) -> None:
    """Callers find the benchmark at portfolio_datas[-1]; that position is the contract."""
    prices, _ = multiasset
    multi = backtest_alpha_rank_portfolio(prices=prices, alpha_scores=momentum_scores,
                                          strategy_ticker='mom', benchmark_ticker='EW')
    tickers = [portfolio.ticker for portfolio in multi.portfolio_datas]
    assert tickers == ['mom', 'EW']


def test_a_dict_of_panels_becomes_one_leg_per_key(multiasset: tuple,
                                                  momentum_scores: pd.DataFrame) -> None:
    """Dict keys name the legs and the benchmark is still appended last."""
    prices, _ = multiasset
    multi = backtest_alpha_rank_portfolio(
        prices=prices,
        alpha_scores={'fast': prices.pct_change(3).shift(1), 'slow': momentum_scores})
    tickers = [portfolio.ticker for portfolio in multi.portfolio_datas]
    assert tickers == ['fast', 'slow', 'Equal Weight']


def test_the_time_period_delays_the_first_trade_without_ending_the_backtest(
        multiasset: tuple, momentum_scores: pd.DataFrame) -> None:
    """``time_period`` trims the *weights*, not the price panel.

    Both legs therefore sit flat at par until the window opens, and then keep running to the
    end of the prices -- the window's end date does not stop them. Worth stating: it reads
    like a reporting window and is not one, so a caller expecting the NAV to end in 2017 gets
    two extra years of live backtest instead.
    """
    prices, _ = multiasset
    multi = backtest_alpha_rank_portfolio(prices=prices, alpha_scores=momentum_scores,
                                          time_period=qis.TimePeriod('2015-01-01', '2017-12-31'))
    for portfolio in multi.portfolio_datas:
        nav = portfolio.get_portfolio_nav()
        assert nav.loc[:'2014-12-31'].nunique() == 1               # flat before the window
        assert nav.loc['2015-06-30':].nunique() > 1                # invested inside it
        assert nav.index[-1] == prices.index[-1]                   # and not stopped at the end


def test_the_analysis_table_carries_a_row_per_leg_and_a_turnover_column(
        multiasset: tuple, momentum_scores: pd.DataFrame) -> None:
    """The table adds the one metric the qis perf table omits: annualised turnover."""
    prices, _ = multiasset
    multi = backtest_alpha_rank_portfolio(prices=prices, alpha_scores=momentum_scores,
                                          strategy_ticker='mom', benchmark_ticker='EW')
    table = compute_alpha_rank_analysis_table(multi_portfolio_data=multi)
    assert list(table.index) == ['mom', 'EW']
    assert list(table.columns) == ['Return p.a.', 'Vol', 'Sharpe', 'Max DD', 'Turnover p.a.']
    assert table['Turnover p.a.'].gt(0.0).all()


def test_the_rank_basket_churns_more_than_the_equal_weight_benchmark(
        multiasset: tuple, momentum_scores: pd.DataFrame) -> None:
    """A basket that follows a moving ranking must trade more than fixed equal weights."""
    prices, _ = multiasset
    multi = backtest_alpha_rank_portfolio(prices=prices, alpha_scores=momentum_scores,
                                          strategy_ticker='mom', benchmark_ticker='EW')
    table = compute_alpha_rank_analysis_table(multi_portfolio_data=multi)
    assert table.loc['mom', 'Turnover p.a.'] > table.loc['EW', 'Turnover p.a.']


def test_the_analysis_table_accepts_explicit_perf_params(multiasset: tuple,
                                                         momentum_scores: pd.DataFrame) -> None:
    """Passing PerfParams bypasses the monthly default without changing the table shape."""
    prices, _ = multiasset
    multi = backtest_alpha_rank_portfolio(prices=prices, alpha_scores=momentum_scores)
    table = compute_alpha_rank_analysis_table(multi_portfolio_data=multi,
                                              perf_params=qis.PerfParams(freq='QE'))
    assert len(table.index) == 2


def test_the_profile_report_writes_a_pdf(multiasset: tuple, momentum_scores: pd.DataFrame,
                                         tmp_path) -> None:
    """The factsheet renders and lands on disk under the requested stem."""
    prices, group_data = multiasset
    multi = backtest_alpha_rank_portfolio(prices=prices, alpha_scores=momentum_scores)
    figs = generate_alpha_profile_report(multi_portfolio_data=multi,
                                         group_data=group_data,
                                         file_name='profile_test',
                                         local_path=f"{tmp_path}/",
                                         add_current_date=False)
    assert len(figs) > 0
    assert (tmp_path / 'profile_test.pdf').exists()
