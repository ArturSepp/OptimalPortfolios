"""Adapters from factorlasso covariance containers to :class:`qis.RiskModel`."""

from typing import Dict, Union

import pandas as pd
import qis
from factorlasso import CurrentFactorCovarData, RollingFactorCovarData, VarianceColumns


def build_risk_model(
        covar_data: Union[
            RollingFactorCovarData,
            Dict[pd.Timestamp, CurrentFactorCovarData],
            Dict[pd.Timestamp, pd.DataFrame],
        ],
) -> qis.RiskModel:
    """Build the canonical qis risk model from covariance estimation output.

    Args:
        covar_data: Rolling factor covariance data, dated current factor
            covariance snapshots, or dated asset covariance matrices.

    Returns:
        Risk model containing the supplied covariance view and, when available,
        its factor loadings, factor covariances, and residual variances.

    Raises:
        ValueError: If ``covar_data`` is not one of the supported container types.
    """
    if isinstance(covar_data, RollingFactorCovarData):
        residual_var_panel = covar_data.get_residual_vars()
        return qis.RiskModel(
            covar=covar_data.get_y_covars(residual_var_weight=1.0),
            factor_loadings=covar_data.get_y_betas(),
            factor_covar=covar_data.get_x_covars(),
            residual_vars={date: residual_var_panel.loc[date] for date in covar_data.dates},
        )

    if isinstance(covar_data, dict):
        if all(isinstance(value, CurrentFactorCovarData) for value in covar_data.values()):
            return qis.RiskModel(
                covar={
                    date: value.get_y_covar(residual_var_weight=1.0)
                    for date, value in covar_data.items()
                },
                factor_loadings={date: value.y_betas for date, value in covar_data.items()},
                factor_covar={date: value.x_covar for date, value in covar_data.items()},
                residual_vars={
                    date: value.y_variances[VarianceColumns.RESIDUAL_VARS.value]
                    for date, value in covar_data.items()
                },
            )
        if all(isinstance(value, pd.DataFrame) for value in covar_data.values()):
            return qis.RiskModel(covar=covar_data)

    raise ValueError(f"unsupported covar_data type: {type(covar_data).__name__}")
