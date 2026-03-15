"""
Abstract base class for covariance matrix estimators.

Defines the shared interface and configuration for all covariance estimators
in the framework. Concrete implementations (EwmaCovarEstimator,
FactorCovarEstimator) provide estimator-specific parameters and logic.

The shared output contract is:
    - fit_rolling_covars(**kwargs) -> Dict[pd.Timestamp, pd.DataFrame]

Each subclass defines its own input signature (EWMA takes prices;
factor model takes risk_factor_prices + asset_returns_dict), but all
return the same type for downstream portfolio optimisation.
"""
from __future__ import annotations

import pandas as pd
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from typing import Dict, Any


@dataclass
class CovarEstimator(ABC):
    """
    Abstract base for covariance matrix estimators.

    Subclasses must implement:
        - ``fit_current_covar``: Single-date covariance estimation.
        - ``fit_rolling_covars``: Rolling covariance estimation over a rebalancing schedule.

    Input signatures are estimator-specific (different subclasses require
    different data). The output contract is shared:
        - fit_rolling_covars() -> Dict[pd.Timestamp, pd.DataFrame]

    Args:
        rebalancing_freq: Pandas frequency string for rolling estimation schedule
            (e.g., 'QE' for quarter-end, 'ME' for month-end, 'YE' for year-end).
    """
    rebalancing_freq: str = 'QE'

    def to_dict(self) -> Dict[str, Any]:
        """Serialise estimator configuration to dictionary."""
        return asdict(self)

    @abstractmethod
    def fit_current_covar(self, **kwargs) -> pd.DataFrame:
        """
        Estimate covariance matrix at a single (latest) date.

        Returns:
            Annualised covariance matrix (N x N) as pd.DataFrame.
        """
        ...

    @abstractmethod
    def fit_rolling_covars(self, **kwargs) -> Dict[pd.Timestamp, pd.DataFrame]:
        """
        Estimate rolling covariance matrices at each rebalancing date.

        Returns:
            Dict mapping rebalancing dates to annualised covariance matrices.
        """
        ...
