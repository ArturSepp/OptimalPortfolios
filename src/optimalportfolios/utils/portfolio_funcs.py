"""
examples of
"""
from __future__ import division

import numpy as np
import pandas as pd
import qis
from typing import Tuple, Union, Optional


def compute_portfolio_variance(w: np.ndarray, covar: np.ndarray) -> float:
    """Return the portfolio variance ``w' Sigma w``."""
    return w.T @ covar @ w


def compute_portfolio_vol(covar: Union[np.ndarray, pd.DataFrame],
                          weights: Union[np.ndarray, pd.Series]
                          ):
    """Return the portfolio volatility, accepting numpy or pandas inputs."""
    if isinstance(covar, pd.DataFrame):
        covar = covar.to_numpy()
    if isinstance(weights, pd.Series):
        weights = weights.to_numpy()
    return np.sqrt(compute_portfolio_variance(w=weights, covar=covar))


def compute_tre_turnover_stats(covar: np.ndarray,
                               benchmark_weights: pd.Series,
                               weights: pd.Series,
                               weights_0: pd.Series,
                               alphas: pd.Series = None
                               ) -> Tuple[float, float, float, float, float]:
    """Summarise one solution against its benchmark and its prior weights.

    Args:
        covar: Covariance aligned with the weight indices.
        benchmark_weights: Benchmark weights.
        weights: Solution weights.
        weights_0: Prior weights, used for turnover.
        alphas: Optional alphas, used for the portfolio alpha.

    Returns:
        ``(te_vol, turnover, port_alpha, port_vol, benchmark_vol)``, with the alpha
        reported as 0.0 when no alphas are given.
    """
    weight_diff = weights.subtract(benchmark_weights)
    benchmark_vol = np.sqrt(benchmark_weights @ covar @ benchmark_weights.T)
    port_vol = np.sqrt(weights @ covar @ weights.T)
    te_vol = np.sqrt(weight_diff @ covar @ weight_diff.T)
    turnover = np.nansum(np.abs(weights.subtract(weights_0)))
    if alphas is not None:
        port_alpha = alphas @ weights
    else:
        port_alpha = 0.0
    return te_vol, turnover, port_alpha, port_vol, benchmark_vol


def calculate_diversification_ratio(w: np.ndarray, covar: np.ndarray) -> float:
    """Return the weighted average asset vol over the portfolio vol.

    The ratio is 1.0 when there is nothing to diversify and rises as correlation falls. With
    uncorrelated unit variances, weights of 0.6 and 0.8 give a portfolio vol of 1.0 against a
    weighted average asset vol of 1.4:

    >>> import numpy as np
    >>> float(calculate_diversification_ratio(np.array([0.6, 0.8]), np.eye(2)))
    1.4
    """
    avg_weighted_vol = np.sqrt(np.diag(covar)) @ w.T
    portfolio_vol = np.sqrt(compute_portfolio_variance(w, covar))
    diversification_ratio = avg_weighted_vol/portfolio_vol
    return diversification_ratio


def compute_portfolio_risk_contribution_outputs(weights: pd.Series,
                                                clean_covar: pd.DataFrame,
                                                risk_budget: Optional[pd.Series] = None
                                                ) -> pd.DataFrame:
    """Tabulate weights, risk contributions and risk budgets per asset.

    Weights are aligned to the covariance columns; ``risk_budget`` defaults to
    zeros when it is not supplied.
    """
    weights = weights.loc[clean_covar.columns]
    asset_rc = qis.compute_portfolio_risk_contributions(w=weights, covar=clean_covar)
    asset_rc_ratio = asset_rc / np.nansum(asset_rc)
    if risk_budget is None:
        risk_budget = pd.Series(0.0, index=clean_covar.columns)
    df = pd.concat([pd.Series(weights, index=clean_covar.columns, name='weights'),
                    asset_rc.rename('risk contribution'),
                    risk_budget.rename('Risk Budget'),
                    asset_rc_ratio.rename('asset_rc_ratio')
                    ], axis=1, sort=False)
    return df


def round_weights_to_pct(weights: pd.Series, decimals: int = 2) -> pd.Series:
    """
    Map portfolio weights from [0,1] to percentage [0,100] with rounding
    that preserves the sum to exactly 100.0 using largest remainder method.

    Each percentage is floored to ``decimals`` places, and the shortfall from 100 is handed to
    the largest remainders, one unit of ``10**-decimals`` each. Weights that sum to one
    therefore give percentages that sum to exactly 100 at ``decimals`` places, each within one
    unit of its exact value. Floors exceed 100 only when the weights sum above one; nothing is
    then bumped.

    Naive rounding of an equal three-asset allocation gives 33.33 each, a total of 99.99; the
    largest remainder takes the bump, so the reported allocation adds to exactly 100:

    >>> import pandas as pd
    >>> weights = pd.Series(1.0 / 3.0, index=['a', 'b', 'c'])
    >>> naive = (100.0 * weights).round(2)
    >>> naive.tolist(), round(float(naive.sum()), 2)
    ([33.33, 33.33, 33.33], 99.99)
    >>> pct = round_weights_to_pct(weights)
    >>> pct.tolist(), round(float(pct.sum()), 2)
    ([33.34, 33.33, 33.33], 100.0)
    """
    scale = 10**decimals
    scaled = weights * 100.0
    units = np.floor(scaled * scale)
    floored = units / scale
    remainders = scaled - floored
    # count the bumps on the integer-valued units, which sum exactly: the former
    # int(shortfall * 10**decimals) truncated 28.999999999999996 to 28 and lost a bump
    n_bumps = max(int(100 * scale - units.sum()), 0)
    # bump the largest remainders
    bump_idx = remainders.nlargest(n_bumps).index
    floored.loc[bump_idx] += 10**(-decimals)
    return floored.round(decimals)
