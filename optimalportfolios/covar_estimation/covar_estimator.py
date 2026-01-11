"""
some utilities for estimation of covariance matrices
"""
from __future__ import annotations
import pandas as pd
import qis as qis
from typing import Union, Optional, Dict, Any
from dataclasses import dataclass, asdict

# project
from optimalportfolios.covar_estimation.config import CovarEstimatorType
from optimalportfolios.lasso.lasso_model_estimator import LassoModel
from optimalportfolios.covar_estimation.rolling_covar import EstimatedRollingCovarData, wrapper_estimate_rolling_covar
from optimalportfolios.covar_estimation.current_covar import EstimatedCurrentCovarData, wrapper_estimate_current_covar


@dataclass
class CovarEstimator:
    """
    specifies estimator specific parameters
    CovarEstimator supports:
    fit_rolling_covars()
    fit_covars()
    """
    covar_estimator_type: CovarEstimatorType = CovarEstimatorType.EWMA
    lasso_model: LassoModel = None  # for mandatory lasso estimator
    factor_returns_freq: str = 'W-WED' # for lasso estimator
    rebalancing_freq: str = 'QE'  # sampling frequency for computing covariance matrix at rebalancing dates
    returns_freqs: Union[str, pd.Series] = 'ME'  # frequency of returns for beta estimation
    span: int = 52  # span for ewma estimate
    is_apply_vol_normalised_returns: bool = False  # for ewma
    demean: bool = True  # adjust for mean
    squeeze_factor: Optional[float] = None  # squeezing factor for ewma covars
    residual_var_weight: float = 1.0  # for lasso covars
    span_freq_dict: Optional[Dict[str, int]] = None  # spans for different freqs
    num_lags_newey_west_dict: Optional[Dict[str, int]] = None

    def to_dict(self) -> Dict[str, Any]:
        this = asdict(self)
        if self.lasso_model is not None:  # need to make it dataclass
            this['lasso_model'] = LassoModel(**this['lasso_model'])
        return this

    def fit_rolling_covars(self,
                           prices: pd.DataFrame,
                           time_period: qis.TimePeriod,
                           risk_factor_prices: pd.DataFrame = None,
                           factors_beta_loading_signs: pd.DataFrame = None,
                           ) -> EstimatedRollingCovarData:
        """
        fit rolling covars at rebalancing_freq
        time_period is for what period we need
        """
        rolling_covar_data = wrapper_estimate_rolling_covar(prices=prices,
                                                            risk_factor_prices=risk_factor_prices,
                                                            time_period=time_period,
                                                            returns_freq=self.factor_returns_freq,
                                                            factors_beta_loading_signs=factors_beta_loading_signs,
                                                            **self.to_dict())
        return rolling_covar_data

    def fit_current_covars(self,
                           prices: pd.DataFrame,
                           risk_factor_prices: pd.DataFrame = None,
                           ) -> EstimatedCurrentCovarData:
        """
        fit rolling covars at rebalancing_freq
        time_period is for what period we need
        """
        rolling_covar_data = wrapper_estimate_current_covar(prices=prices,
                                                            risk_factor_prices=risk_factor_prices,
                                                            **self.to_dict())
        return rolling_covar_data
