"""
implementation of the current covar estimator for which we use the most recent data to get
point in time estimate
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import qis as qis
from dataclasses import dataclass, asdict
from typing import Optional, Union, Dict, Any

# project
from optimalportfolios.covar_estimation.config import CovarEstimatorType
from optimalportfolios.lasso.lasso_model_estimator import LassoModel
from optimalportfolios.covar_estimation.utils import squeeze_covariance_matrix, compute_returns_from_prices


@dataclass
class EstimatedCurrentCovarData:
    """
    outputs from lasso estimation of current covariances
    """
    x_covar: pd.DataFrame
    y_covar: pd.DataFrame
    asset_last_betas: Optional[pd.DataFrame] = None
    last_residual_vars: Optional[pd.Series] = None
    last_total_vars: Optional[pd.Series] = None
    last_r2: Optional[pd.Series] = None
    clusters: Optional[Union[Dict[str, pd.Series], pd.Series]] = None
    linkages: Optional[Union[Dict[str, np.ndarray], np.ndarray]] = None
    cutoffs: Optional[Union[Dict[str, float], float]] = None
    nw_ratios: Optional[pd.Series] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def wrapper_estimate_current_covar(prices: pd.DataFrame,
                                   covar_estimator_type: CovarEstimatorType = CovarEstimatorType.EWMA,
                                   risk_factor_prices: pd.DataFrame = None,  # for lasso covars
                                   **kwargs
                                   ) -> EstimatedCurrentCovarData:
    """
    wrap several methods for covariance estimation
    """
    if covar_estimator_type == CovarEstimatorType.EWMA:
        covars = estimate_current_ewma_covar(prices=prices,
                                             **kwargs)
        covar_data = EstimatedCurrentCovarData(x_covar=covars, y_covar=covars)

    elif covar_estimator_type == CovarEstimatorType.LASSO:
        if risk_factor_prices is None:
            raise ValueError(f"risk_factor_prices must be passed for Lasso estimator")
        covar_data = wrapper_estimate_current_lasso_covar(risk_factors_prices=risk_factor_prices,
                                                          prices=prices,
                                                          **kwargs)

    else:
        raise NotImplementedError(f"covar_estimator_specs.covar_estimator={covar_estimator_type}")

    return covar_data


def estimate_current_ewma_covar(prices: pd.DataFrame,
                                returns_freq: str = 'W-WED',
                                span: int = 52,
                                is_apply_vol_normalised_returns: bool = False,
                                demean: bool = True,
                                squeeze_factor: Optional[float] = None,
                                apply_an_factor: bool = True,
                                **kwargs
                                ) -> pd.DataFrame:
    """
    compute ewma covar matrix: supporting for nans in prices
    output is dict[estimation timestamp, pd.Dataframe(estimated_covar)
    """
    returns = compute_returns_from_prices(prices=prices, returns_freq=returns_freq, demean=demean, span=span)
    x = returns.to_numpy()
    if is_apply_vol_normalised_returns:
        covar_tensor_txy, _, _ = qis.compute_ewm_covar_tensor_vol_norm_returns(a=x, span=span, nan_backfill=qis.NanBackfill.ZERO_FILL)
    else:
        covar_tensor_txy = qis.compute_ewm_covar_tensor(a=x, span=span, nan_backfill=qis.NanBackfill.ZERO_FILL)

    covar_t = covar_tensor_txy[-1]
    if squeeze_factor is not None:
        covar_t = squeeze_covariance_matrix(covar=covar_t, squeeze_factor=squeeze_factor)
    if apply_an_factor:
        an_factor = qis.infer_an_from_data(data=returns)
    else:
        an_factor = 1.0
    current_covar = pd.DataFrame(an_factor*covar_t, columns=returns.columns, index=returns.columns)
    return current_covar


def wrapper_estimate_current_lasso_covar(risk_factors_prices: pd.DataFrame,
                                         prices: pd.DataFrame,
                                         lasso_model: LassoModel,
                                         returns_freqs: Union[str, pd.Series] = 'ME',
                                         factor_returns_freq: str = 'W-WED',
                                         span_freq_dict: Optional[Dict[str, int]] = None,  # spans for different freqs
                                         var_scaler_freq_dict: Optional[Dict[str, float]] = None,  # var scaler for different freqs
                                         squeeze_factor: Optional[float] = None,
                                         is_apply_vol_normalised_returns: bool = False,
                                         residual_var_weight: float = 1.0,
                                         is_adjust_for_newey_west: bool = False,
                                         num_lags_newey_west: Dict[str, int] = {'ME': 0, 'QE': 2},
                                         **kwargs
                                         ) -> EstimatedCurrentCovarData:
    """
    wrapper for lasso covar estimation using either fixed rebalancing frequency or rolling rebalancing frequency
    """
    if isinstance(returns_freqs, str):
        covar_data = estimate_lasso_covar(risk_factor_prices=risk_factors_prices,
                                          prices=prices,
                                          lasso_model=lasso_model,
                                          returns_freq=factor_returns_freq,
                                          is_apply_vol_normalised_returns=is_apply_vol_normalised_returns,
                                          squeeze_factor=squeeze_factor,
                                          residual_var_weight=residual_var_weight,
                                          is_adjust_for_newey_west=is_adjust_for_newey_west,
                                          num_lags_newey_west=num_lags_newey_west)
    else:
        covar_data = estimate_lasso_covar_different_freq(risk_factor_prices=risk_factors_prices,
                                                         prices=prices,
                                                         lasso_model=lasso_model,
                                                         factor_returns_freq=factor_returns_freq,
                                                         returns_freqs=returns_freqs,
                                                         span_freq_dict=span_freq_dict,
                                                         var_scaler_freq_dict=var_scaler_freq_dict,
                                                         is_apply_vol_normalised_returns=is_apply_vol_normalised_returns,
                                                         squeeze_factor=squeeze_factor,
                                                         residual_var_weight=residual_var_weight,
                                                         is_adjust_for_newey_west=is_adjust_for_newey_west,
                                                         num_lags_newey_west=num_lags_newey_west)
    return covar_data


def estimate_lasso_covar(risk_factor_prices: pd.DataFrame,
                         prices: pd.DataFrame,
                         lasso_model: LassoModel,
                         x_covar: Optional[np.ndarray] = None,
                         factor_returns_freq: str = 'W-WED',
                         returns_freq: str = 'ME',
                         squeeze_factor: Optional[float] = None,
                         is_apply_vol_normalised_returns: bool = False,
                         residual_var_weight: float = 1.0,
                         is_adjust_for_newey_west: bool = True,
                         num_lags_newey_west: Dict[str, int] = {'ME': 0, 'QE': 2},
                         verbose: bool = False
                         ) -> EstimatedCurrentCovarData:
    """
    covar = benchmarks covar N*N
    betas = benachmark * asset: N*M
    betas covar = betas.T @ covar @ betas: M*M
    """
    # 1. compute x-factors ewm covar at rebalancing freq
    if x_covar is None:
        x_covar = estimate_current_ewma_covar(prices=risk_factor_prices,
                                              returns_freq=factor_returns_freq,
                                              demean=True,
                                              span=lasso_model.span,
                                              is_apply_vol_normalised_returns=is_apply_vol_normalised_returns,
                                              squeeze_factor=squeeze_factor,
                                              apply_an_factor=False)

    # 2. estimate betas at returns_freq
    y = compute_returns_from_prices(prices=prices, returns_freq=returns_freq, demean=False)
    x = compute_returns_from_prices(prices=risk_factor_prices, returns_freq=returns_freq, demean=False)

    lasso_model.fit(x=x, y=y, verbose=verbose,
                    is_adjust_for_newey_west=is_adjust_for_newey_west,
                    num_lags=num_lags_newey_west[returns_freq])
    asset_last_betas, last_total_vars, last_residual_vars, last_r2 = lasso_model.compute_residual_alpha_r2()

    # make sure aligned
    asset_last_betas = asset_last_betas.reindex(index=x.columns).reindex(columns=y.columns).fillna(0.0)
    betas_np = asset_last_betas.to_numpy()
    betas_covar = np.transpose(betas_np) @ x_covar @ betas_np
    if not np.isclose(residual_var_weight, 0.0):
        betas_covar += residual_var_weight * np.diag(last_residual_vars.to_numpy())

    if is_adjust_for_newey_west:
        ewm_nw, nw_ratios_pd = qis.compute_ewm_newey_west_vol(data=y,
                                                              span=lasso_model.span,
                                                              num_lags=num_lags_newey_west[returns_freq],
                                                              mean_adj_type=qis.MeanAdjType.EWMA)
        last_nw_ratios = nw_ratios_pd.iloc[-1, :].reindex(index=prices.columns).fillna(1.0) # series
        adj = np.sqrt(last_nw_ratios.to_numpy())
        norm = np.outer(adj, adj)
        betas_covar = norm * betas_covar
    else:
        last_nw_ratios = None

    _, an_factor = qis.get_period_days(returns_freq)  # an factor is frequency of x returns
    y_covar = pd.DataFrame(an_factor*betas_covar, index=y.columns, columns=y.columns)

    covar_data = EstimatedCurrentCovarData(x_covar=x_covar,
                                           y_covar=y_covar,
                                           asset_last_betas=asset_last_betas,
                                           last_residual_vars=last_residual_vars,
                                           last_total_vars=last_total_vars,
                                           last_r2=last_r2,
                                           clusters=lasso_model.clusters,
                                           linkages=lasso_model.linkage,
                                           cutoffs=lasso_model.cutoff,
                                           nw_ratios=last_nw_ratios)

    return covar_data


def estimate_lasso_covar_different_freq(risk_factor_prices: pd.DataFrame,
                                        prices: pd.DataFrame,
                                        lasso_model: LassoModel,
                                        returns_freqs: pd.Series,
                                        x_covar: Dict[pd.Timestamp, pd.DataFrame] = None,
                                        factor_returns_freq: str = 'W-WED',
                                        rebalancing_freq: str = 'ME',  # for x returns
                                        is_apply_vol_normalised_returns: bool = False,
                                        span_freq_dict: Optional[Dict[str, int]] = None,  # spans for different freqs
                                        var_scaler_freq_dict: Optional[Dict[str, float]] = None,
                                        residual_var_weight: float = 1.0,
                                        squeeze_factor: Optional[float] = None,
                                        verbose: bool = False,
                                        is_adjust_for_newey_west: bool = True,
                                        num_lags_newey_west: Dict[str, int] = {'ME': 0, 'QE': 2}
                                        ) -> EstimatedCurrentCovarData:
    """
    compute covar matrix at last valuation date
    use benchmarks to compute the benchmark covar matrix
    use lasso to estimate betas
    compute covar matrix
    rebalancing_freq is rebalancing for x returns
    rebalancing_freqs is rebalancing for y  returns
    span_freq_dict: for example {'ME': 12, 'QE': 4} defines spans specific to frequency
    """
    # 1. compute x-factors ewm covar at rebalancing freq
    if x_covar is None:
        x_covar = estimate_current_ewma_covar(prices=risk_factor_prices,
                                              returns_freq=factor_returns_freq,
                                              demean=True,
                                              span=lasso_model.span,
                                              is_apply_vol_normalised_returns=is_apply_vol_normalised_returns,
                                              squeeze_factor=squeeze_factor,
                                              apply_an_factor=False)

    # 2. estimate betas of y-returns at different samples
    returns_freqs = returns_freqs[prices.columns]
    group_freqs = qis.get_group_dict(group_data=returns_freqs)
    betas_freqs: Dict[str, pd.DataFrame] = {}
    total_vars_freqs: Dict[str, pd.Series] = {}
    residual_vars_freqs: Dict[str, pd.Series] = {}
    r2_freqs: Dict[str, Dict[pd.Timestamp, pd.Series]] = {}
    clusters: Dict[str, pd.Series] = {}
    linkages: Dict[str, np.ndarray] = {}
    cutoffs: Dict[str, float] = {}
    nw_ratios: Dict[str, pd.Series] = {}

    # estimate by frequencies
    for freq, asset_tickers in group_freqs.items():
        y = compute_returns_from_prices(prices=prices[asset_tickers], returns_freq=freq, demean=False)
        x = compute_returns_from_prices(prices=risk_factor_prices, returns_freq=freq, demean=False)

        if span_freq_dict is not None:
            if freq in span_freq_dict.keys():
                span_f = span_freq_dict[freq]
            else:
                raise KeyError(f"no span for freq={freq}")
        else:
            span_f = lasso_model.span

        lasso_model.fit(x=x, y=y, verbose=verbose,
                        is_adjust_for_newey_west=is_adjust_for_newey_west,
                        num_lags=num_lags_newey_west[freq])

        betas_freqs[freq], total_vars_freqs[freq], residual_vars_freqs[freq], r2_freqs[freq] =\
            lasso_model.compute_residual_alpha_r2(span=span_f)
        clusters[freq] = lasso_model.clusters
        linkages[freq] = lasso_model.linkage
        cutoffs[freq] = lasso_model.cutoff

        if is_adjust_for_newey_west:
            ewm_nw, nw_ratios_pd = qis.compute_ewm_newey_west_vol(data=y,
                                                                  span=span_f,
                                                                  num_lags=num_lags_newey_west[freq],
                                                                  mean_adj_type=qis.MeanAdjType.EWMA)
            nw_ratios[freq] = nw_ratios_pd.iloc[-1, :]

    # 3. compute y_covars at x_covars frequency
    _, an_factor = qis.get_period_days(rebalancing_freq)  # an factor is frequency of x returns

    # generate aligned betas
    asset_last_betas = []
    last_total_vars = []
    last_residual_vars = []
    last_r2 = []
    last_nw_ratios = []
    for freq in group_freqs.keys():
        asset_last_betas.append(betas_freqs[freq])
        if var_scaler_freq_dict is not None:
            scaler = var_scaler_freq_dict[freq]
        else:
            scaler = 1.0
        last_total_vars.append(scaler * total_vars_freqs[freq])
        last_residual_vars.append(scaler * residual_vars_freqs[freq])
        last_r2.append(r2_freqs[freq])

        if is_adjust_for_newey_west:
            last_nw_ratios.append(nw_ratios[freq])

    # merge and align
    # asset_last_betas: dataframe with columns = assets
    asset_last_betas = pd.concat(asset_last_betas, axis=1).reindex(index=x_covar.index).reindex(columns=prices.columns).fillna(0.0)
    last_total_vars = pd.concat(last_total_vars, axis=0).reindex(index=prices.columns).fillna(0.0)  # series
    last_residual_vars = pd.concat(last_residual_vars, axis=0).reindex(index=prices.columns).fillna(0.0)  # series
    last_r2 = pd.concat(last_r2, axis=0).reindex(index=prices.columns).fillna(0.0)  # series

    # compute covar
    betas_np = asset_last_betas.to_numpy()
    betas_covar = np.transpose(betas_np) @ x_covar.to_numpy() @ betas_np
    if not np.isclose(residual_var_weight, 0.0):
        betas_covar += residual_var_weight * np.diag(last_residual_vars.to_numpy())

    if is_adjust_for_newey_west:
        last_nw_ratios = pd.concat(last_nw_ratios).reindex(index=prices.columns).fillna(1.0) # series
        adj = np.sqrt(last_nw_ratios.to_numpy())
        norm = np.outer(adj, adj)
        betas_covar = norm * betas_covar
    else:
        last_nw_ratios = None

    betas_covar *= an_factor
    y_covar = pd.DataFrame(betas_covar, index=prices.columns, columns=prices.columns)

    covar_data = EstimatedCurrentCovarData(x_covar=x_covar,
                                           y_covar=y_covar,
                                           asset_last_betas=asset_last_betas,
                                           last_residual_vars=last_residual_vars,
                                           last_total_vars=last_total_vars,
                                           last_r2=last_r2,
                                           clusters=clusters,
                                           linkages=linkages,
                                           cutoffs=cutoffs,
                                           nw_ratios=last_nw_ratios)

    return covar_data
