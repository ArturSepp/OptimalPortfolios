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
    eliminate covar with nans
    filter corresponding vectors (can be means, win_max_weights, etc
    """
    covar_pd = pd.DataFrame(covar, index=tickers, columns=tickers)
    is_good_asset = np.where(np.greater(np.diag(covar), 0.0))
    good_tickers = tickers[is_good_asset]
    covar_pd = covar_pd.loc[good_tickers, good_tickers]
    if vectors is not None:
        good_vectors = {key: vector[good_tickers] for key, vector in vectors.items()}
    else:
        good_vectors = None
    return covar_pd, good_vectors
