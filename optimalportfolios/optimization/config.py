import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
from enum import Enum


class PortfolioObjective(Enum):
    MIN_VAR = 1  # min w^t @ covar @ w
    QUADRATIC_UTILITY = 2  # max means^t*w- 0.5*gamma*w^t*covar*w
    EQUAL_RISK_CONTRIBUTION = 3  # implementation in risk_parity
    RISK_PARITY_ALT = 4  # alternative implementation of risk_parity
    MAX_DIVERSIFICATION = 5
    MAXIMUM_SHARPE_RATIO = 6
    MAX_MIXTURE_CARA = 7


def set_min_max_weights(assets: List[str],
                        min_weights: Dict[str, float] = None,
                        max_weights: Dict[str, float] = None,
                        fixed_weights: Dict[str, float] = None,
                        is_long_only: bool = True,
                        abs_weight_bound: float = 1.0
                        ) -> Tuple[pd.Series, pd.Series]:
    """
    set min and max weights for portfolio allocator
    """
    if min_weights is not None:  # map given weights and fill na with 0.0
        min_weights1 = pd.Series(assets, index=assets).map(min_weights).fillna(0.0)
        if is_long_only:
            min_weights1 = min_weights1.fillna(0.0)
        else:
            min_weights1 = min_weights1.fillna(-abs_weight_bound)
    else:
        if is_long_only:
            min_weights1 = pd.Series(0.0, index=assets)
        else:
            min_weights1 = pd.Series(-abs_weight_bound, index=assets)

    if max_weights is not None:  # map given weights and fill na with 0.0
        max_weights1 = pd.Series(assets, index=assets).map(max_weights)
        max_weights1 = max_weights1.fillna(abs_weight_bound)
    else:
        max_weights1 = pd.Series(abs_weight_bound, index=assets)

    if fixed_weights is not None:  # replace max and min weight with fixed
        for asset, weight in fixed_weights.items():
            if asset in assets:
                min_weights1[asset], max_weights1[asset] = weight, weight
            else:
                raise ValueError(f"{asset} is not in {assets}")

    return min_weights1, max_weights1


def set_to_zero_not_investable_weights(min_weights: pd.Series,
                                       max_weights: pd.Series,
                                       covar: np.ndarray
                                       ) -> Tuple[pd.Series, pd.Series]:
    """
    if main diagonal of covar matrix is zero the max/min weight is set to zero
    """
    min_weights, max_weights = min_weights.copy(), max_weights.copy()
    is_not_investable = np.where(np.isclose(np.diag(covar), 0.0), True, False)
    if np.any(is_not_investable):
        min_weights[is_not_investable] = 0.0
        max_weights[is_not_investable] = 0.0
    return min_weights, max_weights

