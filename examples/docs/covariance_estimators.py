"""Canonical script of docs/covariance_estimators.md.

The page's four Python blocks are excerpts of ``main`` and run here in the same order; every
number and property the page states is asserted after them against a reference computed a
different way: an explicit EWMA weighted sum, exact fractions, prefix-only refits and
perturbations of later prices. The factor example of the page moved to
docs/factor_covariance_hcgl.md and its script. The script runs offline after
``pip install optimalportfolios`` and needs no data file or random seed:

    python -m examples.docs.covariance_estimators
"""
from dataclasses import replace
from fractions import Fraction
from importlib.metadata import version
import re

import numpy as np
import pandas as pd

# The page's table of the three-observation example, in annual units.
SMALL_TABLE = [[0.006750, -0.002100], [-0.002100, 0.001500]]
EWMA_KEYS = ['2022-04-06', '2022-07-06', '2022-10-05']


def installed_version(package: str) -> tuple:
    """Return the first three numeric parts of an installed distribution's version."""
    return tuple(int(part) for part in re.findall(r'\d+', version(package))[:3])


def weighted_reference(prices: pd.DataFrame, span: int, annualization: float,
                       demean: bool) -> np.ndarray:
    """Final EWMA covariance as a finite weighted sum, without the qis return or EWMA helpers."""
    returns = pd.DataFrame(np.diff(np.log(prices.to_numpy()), axis=0))
    if demean:
        returns = (returns - returns.ewm(span=span, adjust=False).mean()).iloc[1:]
    decay = 1 - 2 / (span + 1)
    weights = (1 - decay) * decay ** np.arange(len(returns) - 1, -1, -1)
    return annualization * np.einsum('t,ti,tj->ij', weights, returns, returns)


def perturbed_after(prices: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Scale the Equity prices after ``cutoff`` by a factor rising from 1.1 to 3.0."""
    changed = prices.copy()
    later = changed.index > cutoff
    changed.loc[later, 'Equity'] *= np.linspace(1.1, 3.0, later.sum())
    return changed


def main() -> None:
    """Run the page's blocks in order and assert every number and property it states."""
    import numpy as np
    import pandas as pd

    dates = pd.date_range("2021-01-06", periods=160, freq="W-WED")
    steps = np.arange(len(dates), dtype=float)
    prices = pd.DataFrame(
        {"Equity": 100.0 * np.exp(0.002 * steps + 0.05 * np.sin(steps / 5)),
         "Bonds": 100.0 * np.exp(0.0007 * steps + 0.02 * np.cos(steps / 7))},
        index=dates,
    )

    # 160 weekly Wednesdays of positive prices for two assets.
    assert len(prices) == 160 and list(prices.columns) == ["Equity", "Bonds"]
    assert (prices > 0).all().all() and (prices.index.dayofweek == 2).all()

    import optimalportfolios as opt

    estimator = opt.EwmaCovarEstimator(
        returns_freq="W-WED",
        span=52,
        rebalancing_freq="QE",
        demean=True,
    )
    current_covar = estimator.fit_current_covar(prices=prices)

    # Class defaults: weekly Wednesday returns, span 52, quarter-end dates, demeaning on and the
    # ordinary kernel.
    default = opt.EwmaCovarEstimator()
    assert (default.returns_freq, default.span, default.rebalancing_freq) == ("W-WED", 52, "QE")
    assert default.demean is True and default.is_apply_vol_normalised_returns is False
    # Span 52 means decay 51/53 and a half-life of about 18 observations, not 52.
    decay = 1 - 2 / (52 + 1)
    assert round(np.log(0.5) / np.log(decay)) == 18
    # The current fit is the zero-seeded weighted sum of demeaned weekly log returns, times 52,
    # over the complete supplied panel.
    expected = weighted_reference(prices, span=52, annualization=52, demean=True)
    np.testing.assert_allclose(current_covar, expected, rtol=1e-10, atol=1e-14)
    assert current_covar.shape == (2, 2) and current_covar.index.equals(current_covar.columns)
    assert current_covar.columns.tolist() == ["Equity", "Bonds"]
    # No hard lookback: changing the first of 160 prices still moves the current matrix.
    earlier = prices.copy()
    earlier.iloc[0] *= 1.1
    assert np.abs(estimator.fit_current_covar(prices=earlier) - current_covar).max().max() > 1e-6
    # rebalancing_freq only selects rolling output dates; the current fit ignores it.
    pd.testing.assert_frame_equal(
        replace(estimator, rebalancing_freq="ME").fit_current_covar(prices=prices), current_covar)
    # estimate_current_ewma_covar is the function behind the current fit; without the
    # annualization factor it returns weekly units, one 52nd of the annual matrix.
    pd.testing.assert_frame_equal(
        opt.estimate_current_ewma_covar(prices, returns_freq="W-WED", span=52), current_covar)
    weekly = opt.estimate_current_ewma_covar(prices, returns_freq="W-WED", span=52,
                                             apply_an_factor=False)
    np.testing.assert_allclose(52 * weekly, current_covar, rtol=1e-12, atol=0)

    small_returns = np.array([[0.01, 0.02], [-0.02, 0.01], [0.03, -0.01]])
    small_prices = pd.DataFrame(
        100.0 * np.exp(np.vstack([np.zeros(2), np.cumsum(small_returns, axis=0)])),
        index=pd.date_range("2023-12-31", periods=4, freq="ME"),
        columns=["A", "B"],
    )
    small_estimator = opt.EwmaCovarEstimator(returns_freq="ME", span=3, demean=False)
    small_covar = small_estimator.fit_current_covar(prices=small_prices)

    # Span 3 gives decay 1/2: zero-seeded updates weight the three returns 1/8, 1/4 and 1/2 in
    # chronological order, and the weights sum to 7/8.
    observations = [
        [Fraction(1, 100), Fraction(2, 100)],
        [Fraction(-2, 100), Fraction(1, 100)],
        [Fraction(3, 100), Fraction(-1, 100)],
    ]
    half_decay = 1 - Fraction(2, 3 + 1)
    weights = [Fraction(1, 8), Fraction(1, 4), Fraction(1, 2)]
    assert weights == [(1 - half_decay) * half_decay ** age for age in (2, 1, 0)]
    assert sum(weights) == Fraction(7, 8)
    np.testing.assert_allclose(small_returns, np.array(observations, dtype=float),
                               rtol=0, atol=1e-15)
    # Twelve times the weighted second moments, in exact arithmetic, and the page's table.
    exact = [[float(12 * sum(weight * row[i] * row[j]
                             for weight, row in zip(weights, observations)))
              for j in range(2)] for i in range(2)]
    np.testing.assert_allclose(small_covar, exact, rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(small_covar, SMALL_TABLE, rtol=0, atol=0.5e-6)
    assert small_covar.columns.tolist() == ["A", "B"]

    import qis

    ewma_period = qis.TimePeriod(dates[60], dates[100])
    rolling_covars = estimator.fit_rolling_covars(prices=prices, time_period=ewma_period)

    # EWMA keys lie on the weekly grid, each the first Wednesday after a quarter end, so the
    # calendar quarter end itself is not a key.
    assert list(rolling_covars) == list(pd.to_datetime(EWMA_KEYS))
    for key in rolling_covars:
        assert key.dayofweek == 2 and (key - (key - pd.offsets.QuarterEnd())).days < 7
    assert rolling_covars.get(pd.Timestamp("2022-03-31")) is None
    # Point in time with either kernel: each rolling EWMA matrix equals a current fit on the
    # prices through its date, and rescaled later prices change none of them.
    assert installed_version("qis") >= (5, 31, 0)
    later_prices = perturbed_after(prices, max(rolling_covars))
    normalised = replace(estimator, is_apply_vol_normalised_returns=True)
    normalised_covars = normalised.fit_rolling_covars(prices=prices, time_period=ewma_period)
    for kernel, covars in ((estimator, rolling_covars), (normalised, normalised_covars)):
        changed = kernel.fit_rolling_covars(prices=later_prices, time_period=ewma_period)
        assert list(changed) == list(covars)
        for date, covar in covars.items():
            pd.testing.assert_frame_equal(covar, changed[date], check_exact=False,
                                          rtol=1e-10, atol=1e-14)
            np.testing.assert_allclose(covar, kernel.fit_current_covar(prices.loc[:date]),
                                       rtol=1e-10, atol=1e-14)
    # The normalized kernel is a different estimate, and qis seeds each of its volatilities
    # with the column's first squared return rather than a full-array statistic.
    assert not np.allclose(normalised_covars[max(normalised_covars)],
                           rolling_covars[max(rolling_covars)], rtol=1e-6, atol=0)
    log_returns = np.log(prices).diff().iloc[1:].to_numpy()
    _, _, ewm_vols = qis.compute_ewm_covar_tensor_vol_norm_returns(a=log_returns, span=52)
    np.testing.assert_allclose(ewm_vols[0], np.abs(log_returns[0]), rtol=1e-12, atol=0)

    # The legacy estimate_rolling_ewma_covar is the qis function itself.
    assert opt.estimate_rolling_ewma_covar is qis.estimate_rolling_ewma_covar
    print("covariance_estimators: all page statements verified.")


if __name__ == '__main__':
    main()
