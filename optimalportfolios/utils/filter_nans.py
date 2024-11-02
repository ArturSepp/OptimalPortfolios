"""
when we roll optimisation in time, we need to filter our data with nans
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
                                      inclusion_indicators: pd.Series = None
                                      ) -> Tuple[pd.DataFrame, Optional[Dict[str, pd.Series]]]:
    """
    filter out assets with zero variance or nans
    filter corresponding vectors (can be means, win_max_weights, etc
    inclusion_indicators are ones if asset is included for the allocation
    """
    variances = np.diag(pd_covar.to_numpy())
    is_good_asset = np.logical_and(np.greater(variances, 0.0), np.isnan(variances) == False)
    if inclusion_indicators is not None:
        is_included = inclusion_indicators.loc[pd_covar.columns].to_numpy()
        is_good_asset = np.where(np.isclose(is_included, 1.0), is_good_asset, False)

    good_tickers = pd_covar.index[is_good_asset]
    pd_covar = pd_covar.loc[good_tickers, good_tickers]
    if vectors is not None:
        good_vectors = {}
        for key, vector in vectors.items():
            if vector is not None:
                good_vectors[key] = vector[good_tickers].fillna(0.0)
    else:
        good_vectors = None
    return pd_covar, good_vectors
