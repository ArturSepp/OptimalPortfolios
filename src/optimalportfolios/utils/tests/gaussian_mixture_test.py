"""Regression tests for the rolling two-component Gaussian mixture contract."""

import numpy as np
import pandas as pd
import pytest

from optimalportfolios.utils.gaussian_mixture import estimate_rolling_mixture


def _seeded_two_regime_prices(n_returns: int = 420) -> pd.Series:
    """Create weekly prices from seeded log returns with two distinct regimes."""
    rng = np.random.default_rng(20260809)
    is_low_mean = rng.random(n_returns) < 0.35
    log_returns = np.where(
        is_low_mean,
        rng.normal(-0.012, 0.003, n_returns),
        rng.normal(0.006, 0.002, n_returns),
    )
    index = pd.date_range('2012-01-04', periods=n_returns + 1, freq='W-WED')
    return pd.Series(100.0 * np.exp(np.r_[0.0, log_returns].cumsum()),
                     index=index, name='asset')


def test_estimate_rolling_mixture_defaults_succeed() -> None:
    """The public defaults fit and return exactly two finite components."""
    means, vols, probs = estimate_rolling_mixture(prices=_seeded_two_regime_prices())

    assert not means.empty
    assert means.shape == vols.shape == probs.shape
    assert means.shape[1] == 2
    assert np.isfinite(means.to_numpy()).all()
    assert np.isfinite(vols.to_numpy()).all()
    assert np.isfinite(probs.to_numpy()).all()


def test_estimate_rolling_mixture_rejects_other_component_counts() -> None:
    """A non-two-component request raises the explicit contract error."""
    with pytest.raises(ValueError, match=r'supports n_components=2, got 3'):
        estimate_rolling_mixture(prices=_seeded_two_regime_prices(2), n_components=3)


def test_estimate_rolling_mixture_orders_components_by_mean() -> None:
    """Every seeded rolling estimate returns component means in ascending order."""
    means, _, _ = estimate_rolling_mixture(
        prices=_seeded_two_regime_prices(), roll_window=8, annualize=False
    )

    assert (means.iloc[:, 0] < means.iloc[:, 1]).all()
