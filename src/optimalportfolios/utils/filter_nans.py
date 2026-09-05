"""
when we roll optimisation in time, we need to filter our universe with nans
add some utils to deal to provide solution
"""
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional


def filter_covar_and_vectors_for_nans(pd_covar: pd.DataFrame,
                                      vectors: Dict[str, pd.Series] = None,
                                      inclusion_indicators: pd.Series = None,
                                      variance_floor: Optional[float] = None,
                                      drop_non_finite_vectors: bool = False,
                                      ) -> Tuple[pd.DataFrame, Optional[Dict[str, pd.Series]]]:
    """Filter assets with non-positive or NaN variance and align companion vectors.

    Zero, negative, and NaN diagonal entries remove their assets before any optional flooring.
    When ``variance_floor`` is supplied, smaller positive diagonals among the remaining assets are
    raised to that floor. When ``drop_non_finite_vectors`` is true, an asset with a non-finite
    aligned solver vector is removed as well.

    Args:
        pd_covar: Covariance matrix as DataFrame (must be square with matching index/columns).
        vectors: Optional dict of named Series (e.g., alphas, returns) to filter in parallel.
        inclusion_indicators: Optional binary Series (1=include, 0=exclude) for asset filtering.
        variance_floor: Optional minimum diagonal variance for the remaining assets. Positive
            variances below this value are raised to the floor. ``None`` leaves them unchanged.
        drop_non_finite_vectors: If true, require every supplied vector to contain a finite numeric
            value for an asset. Use for objective vectors such as means and alphas; leave false for
            inputs whose caller validates or fills missing values under a different contract.

    Returns:
        Tuple of (filtered covariance DataFrame, filtered vectors dict or None).

    A zero or NaN variance drops the asset, and every supplied vector loses the same entry:

    >>> import numpy as np
    >>> import pandas as pd
    >>> assets = ['a', 'b', 'c']
    >>> covar = pd.DataFrame([[0.04, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, np.nan]],
    ...                      index=assets, columns=assets)
    >>> good, vectors = filter_covar_and_vectors_for_nans(
    ...     covar, vectors={'alphas': pd.Series([1.0, 2.0, 3.0], index=assets)})
    >>> good.columns.tolist()
    ['a']
    >>> vectors['alphas'].tolist()
    [1.0]

    A small positive variance remains unchanged by default and is raised only when a floor is
    supplied:

    >>> covar = pd.DataFrame([[0.04, 0.0, 0.0], [0.0, 1e-12, 0.0], [0.0, 0.0, 0.09]],
    ...                      index=assets, columns=assets)
    >>> good, _ = filter_covar_and_vectors_for_nans(covar, variance_floor=1e-6)
    >>> np.diag(good.to_numpy()).tolist()
    [0.04, 1e-06, 0.09]
    """
    assert pd_covar.index.equals(pd_covar.columns), "pd_covar index and columns must match"

    covar_np = pd_covar.to_numpy().copy()
    variances = np.diag(covar_np)

    # remove non-positive and NaN-variance assets before applying any optional floor
    is_good_asset = np.logical_and(np.greater(variances, 0.0), ~np.isnan(variances))

    if drop_non_finite_vectors and vectors is not None:
        for key, vector in vectors.items():
            if vector is None:
                continue
            if not isinstance(vector, pd.Series):
                raise TypeError(f"vector must be pd.Series not type={type(vector)}")
            aligned = vector.reindex(index=pd_covar.columns)
            try:
                is_good_asset &= np.isfinite(aligned.to_numpy(dtype=float))
            except (TypeError, ValueError) as exc:
                raise TypeError(f"vector {key!r} must contain numeric values") from exc

    # apply inclusion indicators if provided
    if inclusion_indicators is not None:
        is_included = inclusion_indicators.reindex(
            index=pd_covar.columns, fill_value=1.0).to_numpy()
        is_good_asset = np.where(np.isclose(is_included, 1.0), is_good_asset, False)

    good_tickers = pd_covar.index[is_good_asset]

    # subset covariance to good assets
    covar_np = covar_np[np.ix_(is_good_asset, is_good_asset)]

    if variance_floor is not None:
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
