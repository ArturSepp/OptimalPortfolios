"""
examples of
"""
from __future__ import division

import numpy as np
import pandas as pd
from typing import Tuple, Union, Optional


def compute_portfolio_variance(w: np.ndarray, covar: np.ndarray) -> float:
    """Return the portfolio variance ``w' Sigma w``."""
    return w.T @ covar @ w


def compute_portfolio_risk_contributions(w: np.ndarray, covar: np.ndarray) -> np.ndarray:
    """Return per-asset risk contributions, which sum to the portfolio vol."""
    portfolio_vol = np.sqrt(w.T @ covar @ w)
    marginal_risk_contribution = covar @ w.T
    rc = np.multiply(marginal_risk_contribution, w) / portfolio_vol
    return rc


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
    asset_rc = compute_portfolio_risk_contributions(weights.to_numpy(), clean_covar.to_numpy())
    asset_rc_ratio = asset_rc / np.nansum(asset_rc)
    if risk_budget is None:
        risk_budget = pd.Series(0.0, index=clean_covar.columns)
    df = pd.concat([pd.Series(weights, index=clean_covar.columns, name='weights'),
                    pd.Series(asset_rc, index=clean_covar.columns, name='risk contribution'),
                    risk_budget.rename('Risk Budget'),
                    pd.Series(asset_rc_ratio, index=clean_covar.columns, name='asset_rc_ratio')
                    ], axis=1, sort=False)
    return df


def round_weights_to_pct(weights: pd.Series, decimals: int = 2) -> pd.Series:
    """
    Map portfolio weights from [0,1] to percentage [0,100] with rounding
    that preserves the sum to exactly 100.0 using largest remainder method.

    Naive rounding of three near-equal weights gives 99.99; the largest remainder takes the
    bump, so the reported allocation always adds to exactly 100:

    >>> import pandas as pd
    >>> pct = round_weights_to_pct(pd.Series([0.3333, 0.3333, 0.3334], index=['a', 'b', 'c']))
    >>> pct.tolist()
    [33.33, 33.33, 33.34]
    >>> float(pct.sum())
    100.0
    """
    scaled = weights * 100.0
    floored = np.floor(scaled * 10**decimals) / 10**decimals
    remainders = scaled - floored
    shortfall = round(100.0 - floored.sum(), decimals)
    n_bumps = int(shortfall * 10**decimals)
    # bump the largest remainders
    bump_idx = remainders.nlargest(n_bumps).index
    floored.loc[bump_idx] += 10**(-decimals)
    return floored.round(decimals)


def compute_risk_contributions(weights: pd.Series, covar: pd.DataFrame) -> pd.Series:
    """Compute per-asset risk contribution as fraction of portfolio variance.

    RC_i = w_i * (Sigma @ w)_i / (w' Sigma w)

    The covariance matrix's index defines the asset universe. Weights are
    reindexed to this universe (0-fill for missing assets, extra assets
    in weights are ignored). This handles the case where model_backtest
    weights are on the joint universe but covar is TAA-only.

    Args:
        weights: Asset weights (may be on a superset of covar's assets).
        covar: Covariance matrix, indexed and columned by asset names.

    Returns:
        Series of risk contributions (fraction of portfolio variance,
        sums to 1.0), indexed by covar's assets.

    Equal weights on equal, uncorrelated variances split the risk evenly:

    >>> import pandas as pd
    >>> covar = pd.DataFrame([[0.04, 0.0], [0.0, 0.04]], index=['a', 'b'], columns=['a', 'b'])
    >>> compute_risk_contributions(pd.Series([0.5, 0.5], index=['a', 'b']), covar).tolist()
    [0.5, 0.5]

    An asset outside the covariance universe is dropped rather than raising, which is what lets
    a joint-universe weight vector be scored against a TAA-only covariance:

    >>> compute_risk_contributions(pd.Series([0.5, 0.5, 9.0], index=['a', 'b', 'z']),
    ...                            covar).tolist()
    [0.5, 0.5]

    A zero portfolio has no variance to attribute, so the guard returns zeros:

    >>> compute_risk_contributions(pd.Series([0.0, 0.0], index=['a', 'b']), covar).tolist()
    [0.0, 0.0]
    """
    assets = covar.index
    w = weights.reindex(assets).fillna(0.0).values
    cov = covar.values
    port_var = float(w @ cov @ w)
    if port_var <= 0:
        return pd.Series(0.0, index=assets)
    mctr = cov @ w  # marginal contribution to risk
    rc = w * mctr   # risk contribution (sums to port_var)
    rc_ratio = rc / port_var  # fraction of variance (sums to 1.0)
    return pd.Series(rc_ratio, index=assets)


def compute_group_risk_contributions(weights: pd.Series,
                                     covar: pd.DataFrame,
                                     groups: pd.Series
                                     ) -> pd.Series:
    """Aggregate normalized Euler risk contributions over supplied groups.

    Group labels can represent statistical clusters, sectors, asset classes, or any
    other complete partition of the covariance universe. Contributions retain their
    sign and reconcile to the asset-level total returned by
    :func:`compute_risk_contributions`.

    Args:
        weights: Asset weights, which may cover a superset of the covariance assets.
        covar: Labelled covariance matrix defining the risk universe.
        groups: One group label for every covariance asset.

    Returns:
        Normalized group risk contributions in first-seen group order.

    Raises:
        TypeError: If ``groups`` is not a Series.
        ValueError: If covariance or group labels cannot define a complete partition.
    """
    if not isinstance(groups, pd.Series):
        raise TypeError("groups must be a pandas Series")
    if covar.empty or covar.shape[0] != covar.shape[1]:
        raise ValueError("covar must be non-empty and square")
    if not covar.index.equals(covar.columns) or not covar.index.is_unique:
        raise ValueError("covar index and columns must be identical unique asset labels")
    if not groups.index.is_unique:
        raise ValueError("group asset labels must be unique")

    aligned_groups = groups.reindex(covar.index)
    if aligned_groups.isna().any():
        missing = aligned_groups.index[aligned_groups.isna()].tolist()
        raise ValueError(
            f"groups must classify every covariance asset; missing {missing[:5]}"
        )
    contributions = compute_risk_contributions(weights=weights, covar=covar)
    grouped = contributions.groupby(aligned_groups, sort=False).sum()
    grouped.name = "risk_contribution"
    return grouped
