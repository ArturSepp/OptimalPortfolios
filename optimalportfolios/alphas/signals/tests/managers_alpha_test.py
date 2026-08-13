"""
the managers-alpha signal: excess return after stripping factor exposure.

The signal is ``excess_t = r_t - beta_asof(t-1) @ f_t``, and the whole file turns on the
``asof(t-1)``. A beta estimated *at or after* ``t`` would use information the manager could
not have had, which is the look-ahead AGENTS.md names as the source of numerically wrong code
that runs clean — the backtest would simply look better.

So the cases are built to make that visible rather than plausible. The fixture states the
betas, states the factor returns, and lets the test recompute the residual by hand: with a
known beta and a known factor return, the expected excess return is arithmetic. One case
changes the betas partway through the sample and asserts the residual before the change is
computed on the *old* beta, which is exactly what a look-ahead bug would get wrong.

The remaining cases cover the two grid problems the docstring records as having bitten:
resolving the beta as-of rather than by exact timestamp membership (which silently produced
an empty frame whenever the betas schedule and the return grid did not coincide), and the
per-cadence path where assets report at different frequencies.
"""
# packages
from typing import Dict
import numpy as np
import pandas as pd
import qis
# optimalportfolios
from optimalportfolios.alphas.signals.managers_alpha import (
    _estimate_rolling_regression_alphas, compute_managers_alpha)

SEED = 20260810
ASSETS = ['fund_a', 'fund_b', 'fund_c']
FACTORS = ['Equity', 'Rates']


def make_prices(n_months: int = 48, seed: int = SEED) -> pd.DataFrame:
    """A seeded monthly price panel for the managed funds."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range('2020-01-31', periods=n_months, freq='ME')
    returns = rng.normal(0.004, 0.030, size=(n_months, len(ASSETS)))
    return pd.DataFrame(100.0 * np.exp(np.cumsum(returns, axis=0)), index=dates,
                        columns=ASSETS)


def make_factor_prices(n_months: int = 48, seed: int = SEED + 1) -> pd.DataFrame:
    """A seeded monthly factor price panel on the same calendar."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range('2020-01-31', periods=n_months, freq='ME')
    returns = rng.normal(0.003, 0.025, size=(n_months, len(FACTORS)))
    return pd.DataFrame(100.0 * np.exp(np.cumsum(returns, axis=0)), index=dates,
                        columns=FACTORS)


def make_betas(dates, values=(0.8, 0.2)) -> Dict[pd.Timestamp, pd.DataFrame]:
    """The same stated loadings at every estimation date."""
    frame = pd.DataFrame([list(values)] * len(ASSETS), index=ASSETS, columns=FACTORS)
    return {pd.Timestamp(date): frame for date in dates}


def factor_returns_on(prices: pd.DataFrame, factor_prices: pd.DataFrame) -> pd.DataFrame:
    """Factor returns over the asset-return periods, exactly as the signal computes them."""
    asset_returns = qis.to_returns(prices=prices, is_log_returns=True, drop_first=True,
                                   freq='ME')
    aligned = factor_prices.reindex(index=asset_returns.index, method='ffill').ffill()
    return qis.to_returns(prices=aligned, is_log_returns=True, is_first_zero=False,
                          drop_first=False, freq=None)


# --------------------------------------------------------------------------- #
# the residual itself
# --------------------------------------------------------------------------- #
def test_excess_return_is_the_return_less_the_factor_exposure() -> None:
    """with stated betas the residual is arithmetic, so it is checked by hand"""
    prices, factor_prices = make_prices(), make_factor_prices()
    asset_returns = qis.to_returns(prices=prices, is_log_returns=True, drop_first=True,
                                   freq='ME')
    betas = make_betas(asset_returns.index, values=(0.8, 0.2))
    excess = _estimate_rolling_regression_alphas(
        prices=prices, risk_factor_prices=factor_prices, estimated_betas=betas,
        rebalancing_freq='ME', annualise=False)

    factor_ret = factor_returns_on(prices, factor_prices)
    date = excess.index[3]
    expected = (asset_returns.loc[date]
                - betas[asset_returns.index[0]] @ factor_ret.loc[date])
    np.testing.assert_allclose(excess.loc[date].to_numpy(), expected.to_numpy(), atol=1e-12)


def test_a_zero_beta_leaves_the_return_untouched() -> None:
    """with no factor exposure the excess return is the raw return"""
    prices, factor_prices = make_prices(), make_factor_prices()
    asset_returns = qis.to_returns(prices=prices, is_log_returns=True, drop_first=True,
                                   freq='ME')
    excess = _estimate_rolling_regression_alphas(
        prices=prices, risk_factor_prices=factor_prices,
        estimated_betas=make_betas(asset_returns.index, values=(0.0, 0.0)),
        rebalancing_freq='ME', annualise=False)
    common = excess.index
    np.testing.assert_allclose(excess.to_numpy(),
                               asset_returns.loc[common].to_numpy(), atol=1e-12)


def test_the_beta_used_is_the_one_estimated_before_the_return() -> None:
    """a beta change is applied only from the following period — no look-ahead

    The betas switch at the midpoint. A period ending on or before the switch must still be
    residualised on the old loadings; using the new ones would be exactly the look-ahead the
    module's docstring guards against.
    """
    prices, factor_prices = make_prices(), make_factor_prices()
    asset_returns = qis.to_returns(prices=prices, is_log_returns=True, drop_first=True,
                                   freq='ME')
    dates = list(asset_returns.index)
    switch = len(dates) // 2
    old = pd.DataFrame([[0.0, 0.0]] * len(ASSETS), index=ASSETS, columns=FACTORS)
    new = pd.DataFrame([[1.5, 1.5]] * len(ASSETS), index=ASSETS, columns=FACTORS)
    betas = {date: (old if index < switch else new) for index, date in enumerate(dates)}

    excess = _estimate_rolling_regression_alphas(
        prices=prices, risk_factor_prices=factor_prices, estimated_betas=betas,
        rebalancing_freq='ME', annualise=False)

    # a period whose *prior* date still carries the old (zero) beta is the raw return
    early = dates[switch - 1]
    np.testing.assert_allclose(excess.loc[early].to_numpy(),
                               asset_returns.loc[early].to_numpy(), atol=1e-12)
    # once the lagged beta is the new one, the residual is no longer the raw return
    late = dates[switch + 2]
    assert not np.allclose(excess.loc[late].to_numpy(),
                           asset_returns.loc[late].to_numpy())


def test_periods_before_the_first_beta_are_skipped_not_fabricated() -> None:
    """with no estimate yet the period is dropped rather than residualised on nothing"""
    prices, factor_prices = make_prices(), make_factor_prices()
    asset_returns = qis.to_returns(prices=prices, is_log_returns=True, drop_first=True,
                                   freq='ME')
    late_dates = asset_returns.index[20:]
    excess = _estimate_rolling_regression_alphas(
        prices=prices, risk_factor_prices=factor_prices,
        estimated_betas=make_betas(late_dates), rebalancing_freq='ME', annualise=False)
    assert excess.index.min() > asset_returns.index[20]


def test_a_missing_asset_observation_does_not_drop_the_other_managers() -> None:
    """a gap stays local to its manager instead of deleting the whole cross-section.

    A missing price makes both the return ending in the gap and the one starting from it NaN for
    ``fund_a``. Those two cells stay NaN, while ``fund_b`` retains its valid residuals on both
    dates. Dropping the dates would silently shorten every manager's alpha history to the latest
    common starting date in its cadence bucket.
    """
    prices, factor_prices = make_prices(), make_factor_prices()
    asset_returns = qis.to_returns(prices=prices, is_log_returns=True, drop_first=True,
                                   freq='ME')
    gapped = prices.copy()
    gapped.iloc[10, 0] = np.nan  # one month-end observation of fund_a never reported

    excess = _estimate_rolling_regression_alphas(
        prices=gapped, risk_factor_prices=factor_prices,
        estimated_betas=make_betas(asset_returns.index), rebalancing_freq='ME',
        annualise=False)

    gap_date = prices.index[10]
    following = asset_returns.index[list(asset_returns.index).index(gap_date) + 1]
    assert set(excess.index) == set(asset_returns.index[1:])
    assert excess.loc[[gap_date, following], 'fund_a'].isna().all()
    assert excess.loc[[gap_date, following], 'fund_b'].notna().all()
    assert excess.drop(columns='fund_a').notna().all().all()


def test_the_beta_is_resolved_as_of_rather_than_by_exact_timestamp() -> None:
    """an offset betas grid still produces a signal instead of an empty frame

    Exact-membership matching silently returned nothing whenever the betas schedule and the
    resampled return grid did not share timestamps. The as-of lookup is what makes the signal
    survive that, so a deliberately offset grid must still yield rows.
    """
    prices, factor_prices = make_prices(), make_factor_prices()
    asset_returns = qis.to_returns(prices=prices, is_log_returns=True, drop_first=True,
                                   freq='ME')
    # betas keyed a few days off the month-end return grid
    offset_dates = [date - pd.Timedelta(days=3) for date in asset_returns.index]
    excess = _estimate_rolling_regression_alphas(
        prices=prices, risk_factor_prices=factor_prices,
        estimated_betas=make_betas(offset_dates), rebalancing_freq='ME', annualise=False)
    assert not excess.empty, "an offset betas grid produced no signal at all"
    assert len(excess) > len(asset_returns) // 2


def test_annualising_scales_the_excess_returns() -> None:
    """the annualisation factor follows from the frequency and is applied once"""
    prices, factor_prices = make_prices(), make_factor_prices()
    asset_returns = qis.to_returns(prices=prices, is_log_returns=True, drop_first=True,
                                   freq='ME')
    betas = make_betas(asset_returns.index)
    kwargs = dict(prices=prices, risk_factor_prices=factor_prices, estimated_betas=betas,
                  rebalancing_freq='ME')
    raw = _estimate_rolling_regression_alphas(**kwargs, annualise=False)
    annual = _estimate_rolling_regression_alphas(**kwargs, annualise=True)
    np.testing.assert_allclose(annual.to_numpy(), 12.0 * raw.to_numpy(), rtol=1e-9)


def test_a_per_asset_reporting_cadence_is_handled_bucket_by_bucket() -> None:
    """assets reporting at different frequencies are residualised on their own grid"""
    prices, factor_prices = make_prices(n_months=72), make_factor_prices(n_months=72)
    asset_returns = qis.to_returns(prices=prices, is_log_returns=True, drop_first=True,
                                   freq='ME')
    cadence = pd.Series(['ME', 'ME', 'QE'], index=ASSETS)
    excess = _estimate_rolling_regression_alphas(
        prices=prices, risk_factor_prices=factor_prices,
        estimated_betas=make_betas(asset_returns.index), rebalancing_freq=cadence,
        annualise=False)
    # every asset keeps a column even though the buckets ran on different grids
    assert list(excess.columns) == ASSETS
    assert excess['fund_a'].notna().any()
    assert excess['fund_c'].notna().any()


# --------------------------------------------------------------------------- #
# the score
# --------------------------------------------------------------------------- #
def test_managers_alpha_returns_a_score_and_its_raw_alpha() -> None:
    """the score is the smoothed alpha normalised cross-sectionally"""
    prices, factor_prices = make_prices(), make_factor_prices()
    asset_returns = qis.to_returns(prices=prices, is_log_returns=True, drop_first=True,
                                   freq='ME')
    score, raw = compute_managers_alpha(
        prices=prices, risk_factor_prices=factor_prices,
        estimated_betas=make_betas(asset_returns.index), returns_freq='ME', alpha_span=6)
    assert list(score.columns) == ASSETS
    assert score.shape == raw.shape
    # the score is the raw alpha divided by its own cross-sectional dispersion, so the
    # ranking within a date is unchanged
    date = score.dropna().index[-1]
    assert list(score.loc[date].rank()) == list(raw.loc[date].rank())


def test_a_longer_smoothing_span_produces_a_calmer_alpha() -> None:
    """alpha_span is an EWMA span, so raising it must reduce period-to-period movement"""
    prices, factor_prices = make_prices(), make_factor_prices()
    asset_returns = qis.to_returns(prices=prices, is_log_returns=True, drop_first=True,
                                   freq='ME')
    betas = make_betas(asset_returns.index)
    _, fast = compute_managers_alpha(prices=prices, risk_factor_prices=factor_prices,
                                     estimated_betas=betas, returns_freq='ME', alpha_span=2)
    _, slow = compute_managers_alpha(prices=prices, risk_factor_prices=factor_prices,
                                     estimated_betas=betas, returns_freq='ME', alpha_span=24)
    assert slow.diff().abs().mean().mean() < fast.diff().abs().mean().mean()
