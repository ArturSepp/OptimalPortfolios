"""
LASSO-based factor covariance matrix estimator.

Concrete implementation of CovarEstimator using sparse factor model
estimation (LASSO / Group LASSO / HCGL) via CVXPY.

Provides both the shared interface (fit_current_covar, fit_rolling_covars)
and factor-model-specific methods (fit_current_factor_covars,
fit_rolling_factor_covars) that expose the full decomposition.

Reference:
    Sepp A., Ossa I., and Kastenholz M. (2026),
    "Robust Optimization of Strategic and Tactical Asset Allocation for Multi-Asset Portfolios",
    The Journal of Portfolio Management, 52(4), 86-120.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import qis as qis
from typing import Union, Optional, Dict, Any, List
from dataclasses import dataclass, asdict, fields

from optimalportfolios.covar_estimation.covar_estimator import CovarEstimator
from optimalportfolios.covar_estimation.ewma_covar_estimator import estimate_current_ewma_covar
from factorlasso import LassoModel, CurrentFactorCovarData, RollingFactorCovarData, VarianceColumns


@dataclass
class FactorCovarEstimator(CovarEstimator):
    """
    Factor model covariance estimator using LASSO-based sparse regression.

    Estimates Σ_y = β Σ_x β' + D where β is estimated via LASSO/Group LASSO
    and Σ_x is the factor covariance matrix estimated via EWMA.

    Provides two levels of API:
        - **Shared interface** (from CovarEstimator):
            ``fit_current_covar()`` and ``fit_rolling_covars()`` returning
            plain covariance matrices (pd.DataFrame / Dict[Timestamp, DataFrame]).
        - **Factor-model-specific**:
            ``fit_current_factor_covars()`` and ``fit_rolling_factor_covars()``
            returning CurrentFactorCovarData / RollingFactorCovarData with
            full decomposition (betas, residuals, clusters, R², etc.).

    Args:
        lasso_model: Configured LassoModel instance (model_type, reg_lambda, etc.).
        factor_returns_freq: Frequency for computing factor returns (e.g., 'W-WED').
        factor_covar_span: EWMA span for factor covariance estimation.
        is_apply_vol_normalised_returns: If True, normalise returns by rolling vol.
        demean: If True, subtract rolling mean before covariance estimation.

    Example:
        >>> estimator = FactorCovarEstimator(
        ...     lasso_model=LassoModel(model_type=LassoModelType.GROUP_LASSO_CLUSTERS,
        ...                            reg_lambda=1e-5, span=36),
        ...     factor_returns_freq='ME',
        ...     rebalancing_freq='QE',
        ... )
        >>> # Shared interface — plain covar dict
        >>> covar_dict = estimator.fit_rolling_covars(
        ...     risk_factor_prices=factors, asset_returns_dict=returns_dict,
        ...     time_period=time_period)
        >>> # Factor-specific — full decomposition
        >>> rolling_data = estimator.fit_rolling_factor_covars(
        ...     risk_factor_prices=factors, asset_returns_dict=returns_dict,
        ...     time_period=time_period)
        >>> r2_panel = rolling_data.get_r2()
    """
    lasso_model: Optional[LassoModel] = None
    factor_returns_freq: str = 'W-WED'
    factor_covar_span: int = 52
    is_apply_vol_normalised_returns: bool = False
    demean: bool = True

    def copy(self, **overrides) -> FactorCovarEstimator:
        """Create a copy, optionally overriding specific fields.

        Args:
            **overrides: Field names and new values to replace.

        Returns:
            New ProductConfig instance.
        """
        self_dict = {f.name: getattr(self, f.name) for f in fields(self)}
        self_dict.update(overrides)
        return FactorCovarEstimator(**self_dict)

    def to_dict(self) -> Dict[str, Any]:
        """
        Serialise estimator config to dictionary.

        Handles nested ``LassoModel`` dataclass by reconstructing it from its dict
        representation, ensuring round-trip compatibility.
        """
        this = asdict(self)
        if self.lasso_model is not None:
            this['lasso_model'] = LassoModel(**this['lasso_model'])
        return this

    # ── Shared interface (CovarEstimator) ────────────────────────────────

    def fit_current_covar(self,
                          risk_factor_prices: pd.DataFrame,
                          asset_returns_dict: Dict[str, pd.DataFrame],
                          assets: Union[List[str], pd.Index] = None,
                          x_covar: Optional[pd.DataFrame] = None,
                          estimation_date: Optional[pd.Timestamp] = None,
                          residual_var_weight: float = 1.0,
                          ) -> pd.DataFrame:
        """
        Estimate annualised asset covariance matrix at a single date.

        Fits the factor model via ``fit_current_factor_covars`` and returns
        Σ_y = β Σ_x β' + w·D as a plain DataFrame.

        Args:
            risk_factor_prices: Factor price panel.
            asset_returns_dict: Asset returns at multiple frequencies.
            assets: Asset universe to estimate.
            x_covar: Pre-computed factor covariance. If None, estimated internally.
            estimation_date: Reference date for metadata.
            residual_var_weight: Weight on diagonal residual variances.

        Returns:
            Annualised covariance matrix (N x N) as pd.DataFrame.
        """
        factor_data = self.fit_current_factor_covars(
            risk_factor_prices=risk_factor_prices,
            asset_returns_dict=asset_returns_dict,
            assets=assets,
            x_covar=x_covar,
            estimation_date=estimation_date,
        )
        return factor_data.get_y_covar(residual_var_weight=residual_var_weight, assets=assets)

    def fit_rolling_covars(self,
                           risk_factor_prices: pd.DataFrame,
                           asset_returns_dict: Dict[str, pd.DataFrame],
                           time_period: qis.TimePeriod,
                           assets: Union[List[str], pd.Index] = None,
                           rebalancing_freq: Optional[str] = None,
                           residual_var_weight: float = 1.0,
                           ) -> Dict[pd.Timestamp, pd.DataFrame]:
        """
        Estimate rolling covariance matrices at each rebalancing date.

        Calls ``fit_rolling_factor_covars`` and extracts plain covariance dicts.

        Args:
            risk_factor_prices: Factor price panel.
            asset_returns_dict: Asset returns at multiple frequencies.
            time_period: Estimation period.
            assets: Asset universe to estimate.
            rebalancing_freq: Override rebalancing frequency.
            residual_var_weight: Weight on diagonal residual variances in
                Σ_y = β Σ_x β' + w·D. Default 1.0 (full idiosyncratic risk).

        Returns:
            Dict mapping rebalancing dates to annualised asset covariance matrices.
        """
        rolling_data = self.fit_rolling_factor_covars(
            risk_factor_prices=risk_factor_prices,
            asset_returns_dict=asset_returns_dict,
            time_period=time_period,
            assets=assets,
            rebalancing_freq=rebalancing_freq,
        )
        return rolling_data.get_y_covars(residual_var_weight=residual_var_weight,
                                         assets=assets)

    # ── Factor-model-specific API ────────────────────────────────────────

    def fit_current_factor_covars(
            self,
            risk_factor_prices: pd.DataFrame,
            asset_returns_dict: Dict[str, pd.DataFrame],
            assets: Union[List[str], pd.Index] = None,
            x_covar: Optional[pd.DataFrame] = None,
            estimation_date: Optional[pd.Timestamp] = None,
    ) -> CurrentFactorCovarData:
        """
        Fit factor covariance model at a single estimation date.

        Estimates the decomposition Σ_y = β Σ_x β' + D using LASSO-based
        factor selection, where β is sparse factor loadings, Σ_x is factor
        covariance, and D is diagonal idiosyncratic variance.

        Args:
            risk_factor_prices: Factor price panel. Index=dates, columns=factor names.
            asset_returns_dict: Asset returns at multiple frequencies.
                Keys are frequency strings (e.g., 'W-WED'), values are return DataFrames.
            assets: Asset universe to estimate. Must be subset of return columns.
            x_covar: Pre-computed factor covariance matrix. If None, estimated from
                ``risk_factor_prices``.
            estimation_date: Reference date for the estimation.

        Returns:
            Factor covariance decomposition at the estimation date.
        """
        factor_covar_data = estimate_lasso_factor_covar_data(
            risk_factor_prices=risk_factor_prices,
            asset_returns_dict=asset_returns_dict,
            assets=assets,
            lasso_model=self.lasso_model,
            x_covar=x_covar,
            factor_returns_freq=self.factor_returns_freq,
            factor_covar_span=self.factor_covar_span,
            is_apply_vol_normalised_returns=self.is_apply_vol_normalised_returns,
            estimation_date=estimation_date
        )
        return factor_covar_data

    def fit_rolling_factor_covars(
            self,
            risk_factor_prices: pd.DataFrame,
            asset_returns_dict: Dict[str, pd.DataFrame],
            time_period: qis.TimePeriod,
            assets: Union[List[str], pd.Index] = None,
            rebalancing_freq: Optional[str] = None,
    ) -> RollingFactorCovarData:
        """
        Fit factor covariance model at each date in a rebalancing schedule.

        For each rebalancing date, truncates all input data to ``[:estimation_date]``
        and calls ``fit_current_factor_covars`` with expanding-window estimation.

        Args:
            risk_factor_prices: Factor price panel. Index=dates, columns=factor names.
            asset_returns_dict: Asset returns at multiple frequencies.
            assets: Asset universe to estimate at each rebalancing date.
            time_period: Period over which to generate the rebalancing schedule.
            rebalancing_freq: Pandas frequency string for rebalancing dates.

        Returns:
            RollingFactorCovarData with full decomposition at each date.
        """
        rebalancing_schedule = qis.generate_dates_schedule(
            time_period=time_period,
            freq=rebalancing_freq or self.rebalancing_freq,
            include_start_date=False,
            include_end_date=False
        )
        covar_datas: Dict[pd.Timestamp, CurrentFactorCovarData] = {}
        for estimation_date in rebalancing_schedule:
            # Expanding window: use all data up to estimation date
            asset_returns_dict_upto_date = {}
            for freq, returns in asset_returns_dict.items():
                return_t = returns.loc[:estimation_date]
                if len(return_t.index) < self.lasso_model.warmup_period:
                    raise ValueError(
                        f"too early time_period.start={time_period.start} "
                        f"for return {returns.index}: increase start"
                    )
                asset_returns_dict_upto_date[freq] = return_t

            covar_datas[estimation_date] = self.fit_current_factor_covars(
                risk_factor_prices=risk_factor_prices.loc[:estimation_date],
                asset_returns_dict=asset_returns_dict_upto_date,
                assets=assets,
                estimation_date=estimation_date
            )

        return RollingFactorCovarData(data=covar_datas)


def estimate_lasso_factor_covar_data(risk_factor_prices: pd.DataFrame,
                                     asset_returns_dict: Dict[str, pd.DataFrame],
                                     lasso_model: LassoModel,
                                     assets: Union[List[str], pd.Index] = None,
                                     x_covar: pd.DataFrame = None,
                                     factor_returns_freq: str = 'W-WED',
                                     factor_covar_span: int = 52,
                                     is_apply_vol_normalised_returns: bool = False,
                                     estimation_date: pd.Timestamp = None,
                                     verbose: bool = False
                                     ) -> CurrentFactorCovarData:
    """
    Compute factor covariance data at last valuation date.

    Uses LASSO/Group LASSO to estimate sparse factor loadings per asset,
    then assembles the factor covariance matrix, betas, idiosyncratic variances,
    and in-sample diagnostics into CurrentFactorCovarData.

    Convention:
        estimated_betas from LassoModel is (N x M) with index=assets, columns=factors.
        y_betas in CurrentFactorCovarData follows the same (N x M) convention.

    Args:
        risk_factor_prices: Factor price series. Index=dates, columns=factor names.
        asset_returns_dict: Dict[freq_str, DataFrame] of asset returns at different
            rebalancing frequencies (e.g., {'ME': monthly_returns, 'QE': quarterly_returns}).
        assets: Ordered list/index of asset names for output alignment.
        lasso_model: Configured LassoModel instance (model_type, reg_lambda, etc.).
        x_covar: Pre-computed factor covariance matrix (M x M). If None, estimated from
            risk_factor_prices using EWMA.
        factor_returns_freq: Frequency for factor return computation (default 'W-WED').
        factor_covar_span: EWMA span for factor covariance estimation.
        is_apply_vol_normalised_returns: If True, use vol-normalised returns for
            factor covariance estimation.
        estimation_date: Override estimation date. Defaults to last date in first
            frequency's returns.
        verbose: If True, print solver diagnostics.

    Returns:
        CurrentFactorCovarData with factor covariance, betas (N x M), variances,
        clusters, and residuals.
    """
    # 1. compute x-factors ewm covar at rebalancing freq
    if x_covar is None:
        x_covar = estimate_current_ewma_covar(prices=risk_factor_prices,
                                              returns_freq=factor_returns_freq,
                                              demean=True,
                                              span=factor_covar_span,
                                              is_apply_vol_normalised_returns=is_apply_vol_normalised_returns,
                                              apply_an_factor=False)
        factor_scale_an = qis.get_annualisation_conversion_factor(from_freq=factor_returns_freq, to_freq='YE')
        x_covar *= factor_scale_an

    # 2. estimate betas and diagnostics per frequency
    betas_freqs: Dict[str, pd.DataFrame] = {}       # each is (N_freq x M)
    ewma_vars_freqs: Dict[str, pd.Series] = {}
    residual_vars_freqs: Dict[str, pd.Series] = {}
    alphas_freqs: Dict[str, pd.Series] = {}
    r2_freqs: Dict[str, pd.Series] = {}
    clusters: Dict[str, pd.Series] = {}
    linkages: Dict[str, np.ndarray] = {}
    cutoffs: Dict[str, float] = {}
    residuals_freqs: Dict[str, pd.DataFrame] = {}

    for freq in asset_returns_dict.keys():
        y = asset_returns_dict[freq]
        x_prices = risk_factor_prices.reindex(index=y.index, method='ffill').ffill()
        x = qis.to_returns(prices=x_prices, is_log_returns=True, is_first_zero=False, drop_first=False, freq=None)

        if lasso_model.span_freq_dict is not None:
            if freq in lasso_model.span_freq_dict.keys():
                span_f = lasso_model.span_freq_dict[freq]
            else:
                raise KeyError(f"no span for freq={freq} in lasso_model.span_freq_dict")
        else:
            span_f = lasso_model.span

        # fit() returns self; estimated_betas is (N x M) DataFrame
        lasso_model.fit(x=x, y=y, verbose=verbose, span=span_f)

        # estimated_betas: index=assets, columns=factors (N x M)
        betas_freqs[freq] = lasso_model.estimated_betas

        # Diagnostics from LassoEstimationResult stored on model
        result = lasso_model.estimation_result_
        ewma_vars_freqs[freq] = pd.Series(result.ss_total, index=y.columns)
        residual_vars_freqs[freq] = pd.Series(result.ss_res, index=y.columns)
        alphas_freqs[freq] = pd.Series(result.alpha, index=y.columns)
        r2_freqs[freq] = pd.Series(result.r2, index=y.columns)

        clusters[freq] = lasso_model.clusters
        linkages[freq] = lasso_model.linkage
        cutoffs[freq] = lasso_model.cutoff

        # in-sample residuals: y - x @ beta' where beta is (N x M)
        # x is (T x M), beta.T is (M x N), result is (T x N)
        residuals_freqs[freq] = y - x @ betas_freqs[freq].T

    # 3. annualise and merge across frequencies
    asset_last_betas = []
    last_ewma_vars = []
    last_residual_vars = []
    last_alphas = []
    last_r2 = []
    residuals = []
    for freq in asset_returns_dict.keys():
        asset_last_betas.append(betas_freqs[freq])  # (N_freq x M)
        idio_var_scaler = qis.get_annualisation_conversion_factor(from_freq=freq, to_freq='YE')
        last_ewma_vars.append(idio_var_scaler * ewma_vars_freqs[freq])
        last_residual_vars.append(idio_var_scaler * residual_vars_freqs[freq])
        last_alphas.append(idio_var_scaler * alphas_freqs[freq])
        last_r2.append(r2_freqs[freq])
        residuals.append(idio_var_scaler * residuals_freqs[freq])

    # align to target asset universe
    # betas: concat along axis=0 (rows=assets), reindex rows to target assets, columns to factors
    asset_last_betas = pd.concat(asset_last_betas, axis=0).reindex(columns=x_covar.index).fillna(0.0)
    last_ewma_vars = pd.concat(last_ewma_vars, axis=0).fillna(0.0)
    last_residual_vars = pd.concat(last_residual_vars, axis=0).fillna(0.0)
    last_alphas = pd.concat(last_alphas, axis=0).fillna(0.0)
    last_r2 = pd.concat(last_r2, axis=0).fillna(0.0).clip(0.0, None)
    residuals = pd.concat(residuals, axis=1).fillna(0.0)
    if assets is not None:
        asset_last_betas = asset_last_betas.reindex(index=assets).fillna(0.0)
        last_ewma_vars = last_ewma_vars.reindex(index=assets).fillna(0.0)
        last_residual_vars = last_residual_vars.reindex(index=assets).fillna(0.0)
        last_alphas = last_alphas.reindex(index=assets).fillna(0.0)
        last_r2 = last_r2.reindex(index=assets).fillna(0.0)
        residuals = residuals.reindex(columns=assets).fillna(0.0)

    y_variances = pd.concat([last_ewma_vars.rename(VarianceColumns.EWMA_VARIANCE.value),
                             last_residual_vars.rename(VarianceColumns.RESIDUAL_VARS.value),
                             last_alphas.rename(VarianceColumns.INSAMPLE_ALPHA.value),
                             last_r2.rename(VarianceColumns.R2.value)],
                            axis=1)

    estimation_date = estimation_date or asset_returns_dict[list(asset_returns_dict.keys())[0]].index[-1]
    covar_data = CurrentFactorCovarData(x_covar=x_covar,
                                        y_betas=asset_last_betas,
                                        y_variances=y_variances,
                                        clusters=clusters,
                                        linkages=linkages,
                                        cutoffs=cutoffs,
                                        residuals=residuals,
                                        estimation_date=estimation_date)
    return covar_data
