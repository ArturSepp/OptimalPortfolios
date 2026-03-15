"""
when we roll optimisation in time, we need to filter our universe with nans
add some utils to deal to provide solution
"""
import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional


def filter_covar_and_vectors(covar: np.ndarray,
                             tickers: pd.Index,
                             vectors: Dict[str, pd.Series] = None
                             ) -> Tuple[pd.DataFrame, Optional[Dict[str, pd.Series]]]:
    """
    filter out assets with zero variance or nans
    filter corresponding vectors (can be means, win_max_weights, etc
    """
    covar_pd = pd.DataFrame(covar, index=tickers, columns=tickers)
    variances = np.diag(covar)
    is_good_asset = np.where(np.logical_and(np.greater(variances, 0.0), np.isnan(variances) == False))
    good_tickers = tickers[is_good_asset]
    covar_pd = covar_pd.loc[good_tickers, good_tickers]
    if vectors is not None:
        good_vectors = {key: vector[good_tickers] for key, vector in vectors.items()}
    else:
        good_vectors = None
    return covar_pd, good_vectors


def filter_covar_and_vectors_for_nans(pd_covar: pd.DataFrame,
                                      vectors: Dict[str, pd.Series] = None,
                                      inclusion_indicators: pd.Series = None,
                                      variance_floor: float = (0.001) ** 2
                                      ) -> Tuple[pd.DataFrame, Optional[Dict[str, pd.Series]]]:
    """Filter out assets with NaN variance and clamp near-zero variances to a floor.

    Assets with NaN variance are removed. Assets with near-zero but valid variance
    (e.g., cash instruments) have their diagonal clamped to variance_floor to ensure
    numerical stability in the optimizer without excluding them from allocation.

    variance_floor default of 0.001² = 1e-6 corresponds to ~10bps annualized vol,
    which is a reasonable lower bound for any tradeable instrument. You can pass a different value
    if your universe includes instruments with genuinely lower vol

    Args:
        pd_covar: Covariance matrix as DataFrame (must be square with matching index/columns).
        vectors: Optional dict of named Series (e.g., alphas, returns) to filter in parallel.
        inclusion_indicators: Optional binary Series (1=include, 0=exclude) for asset filtering.
        variance_floor: Minimum diagonal variance for included assets. Assets below this
            threshold are clamped (not removed). Default corresponds to ~10bps annualized vol.

    Returns:
        Tuple of (filtered covariance DataFrame, filtered vectors dict or None).
    """
    assert pd_covar.index.equals(pd_covar.columns), "pd_covar index and columns must match"

    covar_np = pd_covar.to_numpy().copy()
    variances = np.diag(covar_np)

    # identify assets with valid (non-NaN) variance
    is_good_asset = ~np.isnan(variances)

    # apply inclusion indicators if provided
    if inclusion_indicators is not None:
        is_included = inclusion_indicators.reindex(index=pd_covar.columns, fill_value=1.0).to_numpy()
        is_good_asset = np.where(np.isclose(is_included, 1.0), is_good_asset, False)

    good_tickers = pd_covar.index[is_good_asset]

    # subset covariance to good assets
    covar_np = covar_np[np.ix_(is_good_asset, is_good_asset)]

    # clamp near-zero variances to floor (keeps cash/low-vol assets in the optimization)
    diag = np.diag(covar_np)
    below_floor = diag < variance_floor
    if below_floor.any():
        # Increasing diagonal entries preserves positive semi-definiteness.
        np.fill_diagonal(covar_np, np.maximum(diag, variance_floor))

    pd_covar = pd.DataFrame(covar_np, index=good_tickers, columns=good_tickers)

    # filter vectors to match good tickers
    if vectors is not None:
        good_vectors = {}
        for key, vector in vectors.items():
            if vector is not None:
                if isinstance(vector, pd.Series):
                    good_vectors[key] = vector.reindex(index=good_tickers, fill_value=0.0)
                else:
                    raise TypeError(f"vector must be pd.Series not type={type(vector)}")
    else:
        good_vectors = None

    return pd_covar, good_vectors
