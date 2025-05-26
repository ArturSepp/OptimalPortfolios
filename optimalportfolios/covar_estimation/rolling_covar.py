"""
implementation of rolling covar estimators
rolling covar is estimated as a Dict[timestamp, covar]
at the following rebalancing frequency
rebalancing_schedule = qis.generate_rebalancing_indicators(df=returns, freq=rebalancing_freq)

"""

from __future__ import annotations

import numpy as np
import pandas as pd
import qis as qis
from dataclasses import dataclass, asdict
from typing import Dict, Optional, Union

# project
from optimalportfolios.covar_estimation.config import CovarEstimatorType
from optimalportfolios.lasso.lasso_model_estimator import LassoModel
from optimalportfolios.covar_estimation.utils import squeeze_covariance_matrix, compute_returns_from_prices


@dataclass
class EstimatedRollingCovarData:
    """
    outputs from lasso estimation of rolling covariances
    """
    x_covars: Dict[pd.Timestamp, pd.DataFrame]
    y_covars: Optional[Dict[pd.Timestamp, pd.DataFrame]] = None
    asset_last_betas_t: Optional[Dict[pd.Timestamp, pd.DataFrame]] = None
    last_residual_vars_t: Optional[Dict[pd.Timestamp, pd.DataFrame]] = None
    total_vars_pd: Optional[pd.DataFrame] = None
    residual_vars_pd: Optional[pd.DataFrame] = None
    r2_pd: Optional[pd.DataFrame] = None
    last_nw_ratios_pd: Optional[pd.DataFrame] = None

    def to_dict(self):
        return asdict(self)


def wrapper_estimate_rolling_covar(prices: pd.DataFrame,
                                   time_period: qis.TimePeriod,  # starting time of sampling estimator
                                   covar_estimator_type: CovarEstimatorType = CovarEstimatorType.EWMA,
                                   risk_factor_prices: pd.DataFrame = None,  # for lasso covars
                                   returns_freq: str = 'ME',
                                   **kwargs
                                   ) -> EstimatedRollingCovarData:
    """
    wrap several methods for covariance estimation
    """
    if covar_estimator_type == CovarEstimatorType.EWMA:
        covars = estimate_rolling_ewma_covar(prices=prices,
                                             time_period=time_period,
                                             returns_freq=returns_freq,
                                             **kwargs)
        covar_data = EstimatedRollingCovarData(x_covars=covars, y_covars=covars)

    elif covar_estimator_type == CovarEstimatorType.LASSO:
        if risk_factor_prices is None:
            raise ValueError(f"risk_factor_prices must be passed for Lasso estimator")
        covar_data = wrapper_estimate_rolling_lasso_covar(risk_factors_prices=risk_factor_prices,
                                                          prices=prices,
                                                          time_period=time_period,  # when we start building portfolios
                                                          **kwargs)

    else:
        raise NotImplementedError(f"covar_estimator_specs.covar_estimator={covar_estimator_type}")

    return covar_data


def estimate_rolling_ewma_covar(prices: pd.DataFrame,
                                time_period: qis.TimePeriod,  # when we start building portfolios
                                returns_freq: str = 'W-WED',
                                rebalancing_freq: str = 'QE',
                                span: int = 52,
                                is_apply_vol_normalised_returns: bool = False,
                                demean: bool = True,
                                squeeze_factor: Optional[float] = None,
                                apply_an_factor: bool = True,
                                **kwargs
                                ) -> Dict[pd.Timestamp, pd.DataFrame]:
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

    # create rebalancing schedule
    rebalancing_schedule = qis.generate_rebalancing_indicators(df=returns, freq=rebalancing_freq)

    tickers = prices.columns.to_list()
    covars = {}
    if apply_an_factor:
        an_factor = qis.infer_an_from_data(data=returns)
    else:
        an_factor = 1.0
    start_date = time_period.start.tz_localize(tz=returns.index.tz)  # make sure tz is alined with rebalancing_schedule
    for idx, (date, value) in enumerate(rebalancing_schedule.items()):
        if value and date >= start_date:
            covar_t = pd.DataFrame(covar_tensor_txy[idx], index=tickers, columns=tickers)
            if squeeze_factor is not None:
                covar_t = squeeze_covariance_matrix(covar=covar_t, squeeze_factor=squeeze_factor)
            covars[date] = an_factor*covar_t
    return covars


def wrapper_estimate_rolling_lasso_covar(risk_factors_prices: pd.DataFrame,
                                         prices: pd.DataFrame,
                                         returns_freqs: Union[str, pd.Series],
                                         time_period: qis.TimePeriod,  # when we start building portfolios
                                         lasso_model: LassoModel,
                                         rebalancing_freq: str = 'ME',
                                         factor_returns_freq: str = 'ME',
                                         span_freq_dict: Optional[Dict[str, int]] = None,  # spans for different freqs
                                         var_scaler_freq_dict: Optional[Dict[str, float]] = None,  # var scaler for different freqs
                                         squeeze_factor: Optional[float] = None,
                                         is_apply_vol_normalised_returns: bool = False,
                                         residual_var_weight: float = 1.0,
                                         is_adjust_for_newey_west: bool = False,
                                         num_lags_newey_west: Dict[str, int] = {'ME': 0, 'QE': 2},
                                         **kwargs
                                         ) -> EstimatedRollingCovarData:
    """
    wrapper for lasso rolling covar estimation using either fixed rebalancing frequency or rolling rebalancing frequency
    returns_freqs can str of pd.Series for sampling of prices
    rebalancing_freq is when we compute asset covar
    factor_returns_freq is frequency for computing factors covar matrix
    """
    if isinstance(returns_freqs, str):
        covar_data = estimate_rolling_lasso_covar(risk_factor_prices=risk_factors_prices,
                                                  prices=prices,
                                                  time_period=time_period,
                                                  lasso_model=lasso_model,
                                                  factor_returns_freq=factor_returns_freq,
                                                  rebalancing_freq=rebalancing_freq,
                                                  returns_freq=returns_freqs,
                                                  is_apply_vol_normalised_returns=is_apply_vol_normalised_returns,
                                                  squeeze_factor=squeeze_factor,
                                                  residual_var_weight=residual_var_weight,
                                                  is_adjust_for_newey_west=is_adjust_for_newey_west,
                                                  num_lags_newey_west=num_lags_newey_west)
    else:
        covar_data = estimate_rolling_lasso_covar_different_freq(risk_factor_prices=risk_factors_prices,
                                                                 prices=prices,
                                                                 time_period=time_period,
                                                                 lasso_model=lasso_model,
                                                                 factor_returns_freq=factor_returns_freq,
                                                                 rebalancing_freq=rebalancing_freq,
                                                                 returns_freqs=returns_freqs,
                                                                 span_freq_dict=span_freq_dict,
                                                                 var_scaler_freq_dict=var_scaler_freq_dict,
                                                                 is_apply_vol_normalised_returns=is_apply_vol_normalised_returns,
                                                                 squeeze_factor=squeeze_factor,
                                                                 residual_var_weight=residual_var_weight,
                                                                 is_adjust_for_newey_west=is_adjust_for_newey_west,
                                                                 num_lags_newey_west=num_lags_newey_west)
    return covar_data


def estimate_rolling_lasso_covar(risk_factor_prices: pd.DataFrame,
                                 prices: pd.DataFrame,
                                 time_period: qis.TimePeriod,  # when we start building portfolios
                                 lasso_model: LassoModel,
                                 returns_freq: str = 'ME',
                                 factor_returns_freq: str = 'W-WED',
                                 rebalancing_freq: str = 'QE',
                                 span: int = 52,  # 1y of weekly returns
                                 is_apply_vol_normalised_returns: bool = False,
                                 squeeze_factor: Optional[float] = None,
                                 residual_var_weight: float = 1.0,
                                 is_adjust_for_newey_west: bool = False,
                                 num_lags_newey_west: Optional[Dict[str, int]] = {'ME': 0, 'QE': 2}
                                 ) -> EstimatedRollingCovarData:
    """
    use benchmarks to compute the benchmark covar matrix
    use lasso to estimate betas and to compute covar matrix
    returns_freq is frequency of return for estimation of betas
    factor_returns_freq is frequency for estimation of factors covar matrix
    rebalancing_freq is when covar matrix and betas are computed for output
    """
    # 1. compute x-factors ewm covar at rebalancing freq
    x_covars = estimate_rolling_ewma_covar(prices=risk_factor_prices,
                                           time_period=time_period,
                                           returns_freq=factor_returns_freq,
                                           rebalancing_freq=rebalancing_freq,
                                           span=span,
                                           demean=lasso_model.demean,
                                           is_apply_vol_normalised_returns=is_apply_vol_normalised_returns,
                                           squeeze_factor=squeeze_factor,
                                           apply_an_factor=False)

    # 2. estimate betas of y-returns at different samples, no adjustment for mean - this will be done in fit()
    y = compute_returns_from_prices(prices=prices, returns_freq=returns_freq, demean=False, span=None)
    x = compute_returns_from_prices(prices=risk_factor_prices, returns_freq=returns_freq, demean=False, span=None)

    # todo: can reduce the number of evaluations if returns_freq << rebalancing_freq
    if num_lags_newey_west is not None and factor_returns_freq in num_lags_newey_west:
        num_lags = num_lags_newey_west[factor_returns_freq]
    else:
        num_lags = None
    betas, total_vars, residual_vars, r2_t = lasso_model.estimate_rolling_betas(x=x, y=y,
                                                                                is_adjust_for_newey_west=is_adjust_for_newey_west,
                                                                                num_lags=num_lags)

    if is_adjust_for_newey_west:
        ewm_nw, nw_ratios = qis.compute_ewm_newey_west_vol(data=y, span=span, num_lags=num_lags,
                                                           mean_adj_type=qis.MeanAdjType.EWMA)
        # nw_ratios = qis.compute_ewm(data=nw_ratios.clip(lower=1.0), span=span)

    # 3. compute y_covars at x_covars frequency
    _, an_factor = qis.get_period_days(freq=factor_returns_freq)  # annualisation factor is provided based on returns freq
    y_covars = {}
    asset_last_betas_t = {}
    last_residual_vars_t = {}
    last_nw_ratios_t = {}
    for date, x_covar in x_covars.items():
        # find last update date from
        last_update_date = qis.find_upto_date_from_datetime_index(index=list(betas.keys()), date=date)
        # align
        asset_last_betas = betas[last_update_date].loc[x_covar.index, prices.columns]
        last_residual_vars = residual_vars[last_update_date][prices.columns]

        # compute covar
        betas_np = asset_last_betas.to_numpy()
        betas_covar = np.transpose(betas_np) @ x_covar.to_numpy() @ betas_np
        if not np.isclose(residual_var_weight, 0.0):
            betas_covar += residual_var_weight * np.diag(last_residual_vars.to_numpy())

        if is_adjust_for_newey_west:
            last_nw_ratios = nw_ratios.loc[last_update_date, :].replace({0: np.nan}).fillna(1.0)
            adj = np.sqrt(last_nw_ratios.to_numpy())
            norm = np.outer(adj, adj)
            betas_covar = norm * betas_covar
            last_nw_ratios_t[date] = last_nw_ratios

        y_covars[date] = pd.DataFrame(an_factor*betas_covar, index=prices.columns, columns=prices.columns)

        asset_last_betas_t[date] = asset_last_betas
        last_residual_vars_t[date] = last_residual_vars

    if is_adjust_for_newey_west:
        last_nw_ratios_pd = pd.DataFrame.from_dict(last_nw_ratios_t, orient='index')
    else:
        last_nw_ratios_pd = None

    covar_data = EstimatedRollingCovarData(x_covars=x_covars,
                                           y_covars=y_covars,
                                           asset_last_betas_t=asset_last_betas_t,
                                           last_residual_vars_t=last_residual_vars_t,
                                           total_vars_pd=pd.DataFrame.from_dict(total_vars, orient='index'),
                                           residual_vars_pd=pd.DataFrame.from_dict(residual_vars, orient='index'),
                                           r2_pd=pd.DataFrame.from_dict(r2_t, orient='index'),
                                           last_nw_ratios_pd=last_nw_ratios_pd)
    return covar_data


def estimate_rolling_lasso_covar_different_freq(risk_factor_prices: pd.DataFrame,
                                                prices: pd.DataFrame,
                                                returns_freqs: pd.Series,
                                                time_period: qis.TimePeriod,  # when we start building portfolios
                                                lasso_model: LassoModel,
                                                rebalancing_freq: str = 'ME',  # for x returns
                                                factor_returns_freq: str = 'W-WED',
                                                is_apply_vol_normalised_returns: bool = False,
                                                span_freq_dict: Optional[Dict[str, int]] = None,  # spans for different freqs
                                                var_scaler_freq_dict: Optional[Dict[str, float]] = None,
                                                squeeze_factor: Optional[float] = None,
                                                residual_var_weight: float = 1.0,
                                                is_adjust_for_newey_west: bool = False,
                                                num_lags_newey_west: Dict[str, int] = {'ME': 0, 'QE': 2}
                                                ) -> EstimatedRollingCovarData:
    """
    use benchmarks to compute the benchmark covar matrix
    use lasso to estimate betas
    compute covar matrix
    returns_freq is frequency of return for estimation of betas
    factor_returns_freq is frequency for estimation of factors covar matrix
    rebalancing_freq is when covar matrix and betas are computed for output

    var_scaler_freq_dict: var scaler for different freq. If factor_returns_freq is ME and some frequencies ar QE
    we neet var_scaler_freq_dict['QE'] = 1/3 to extrapolate vara and covars to ME frequency
    rebalancing_freq must be the smallest of returns_freqs
    """
    # 1. compute x-factors ewm covar at rebalancing freq
    x_covars = estimate_rolling_ewma_covar(prices=risk_factor_prices,
                                           time_period=time_period,
                                           returns_freq=factor_returns_freq,
                                           rebalancing_freq=rebalancing_freq,
                                           span=lasso_model.span,
                                           demean=lasso_model.demean,
                                           is_apply_vol_normalised_returns=is_apply_vol_normalised_returns,
                                           squeeze_factor=squeeze_factor,
                                           apply_an_factor=False)

    # 2. estimate betas of y-returns at different samples
    returns_freqs = returns_freqs[prices.columns]
    group_freqs = qis.get_group_dict(group_data=returns_freqs)
    betas_freqs: Dict[str, Dict[pd.Timestamp, pd.DataFrame]] = {}
    total_vars_freqs: Dict[str, Dict[pd.Timestamp, pd.Series]] = {}
    residual_vars_freqs: Dict[str, Dict[pd.Timestamp, pd.Series]] = {}
    r2_freqs: Dict[str, Dict[pd.Timestamp, pd.Series]] = {}
    nw_ratios: Dict[str, pd.DataFrame] = {}

    for freq, asset_tickers in group_freqs.items():
        y = qis.to_returns(prices=prices[asset_tickers], is_log_returns=True, drop_first=True, freq=freq)
        x = qis.to_returns(prices=risk_factor_prices, is_log_returns=True, drop_first=True, freq=freq)
        if span_freq_dict is not None:
            if freq in span_freq_dict.keys():
                span_f = span_freq_dict[freq]
            else:
                raise KeyError(f"no span for freq={freq}")
        else:
            span_f = lasso_model.span
        betas_freqs[freq], total_vars_freqs[freq], residual_vars_freqs[freq], r2_freqs[freq] \
            = lasso_model.estimate_rolling_betas(x=x, y=y, span=span_f,
                                                 is_adjust_for_newey_west=is_adjust_for_newey_west,
                                                 num_lags=num_lags_newey_west[freq])

        if is_adjust_for_newey_west:
            ewm_nw, nw_ratios_ = qis.compute_ewm_newey_west_vol(data=y, span=span_f, num_lags=num_lags_newey_west[freq],
                                                                mean_adj_type=qis.MeanAdjType.EWMA)
            # nw_ratios_ = qis.compute_ewm(data=nw_ratios_.clip(lower=1.0), span=span_f)
            nw_ratios[freq] = nw_ratios_

    # 3. compute y_covars at x_covars frequency
    _, an_factor = qis.get_period_days(rebalancing_freq)  # an factor is frequency of rebalancing_freq
    y_covars = {}
    asset_last_betas_t = {}
    last_total_vars_t = {}
    last_residual_vars_t = {}
    last_r2_vars_t = {}
    last_nw_ratios_t = {}
    for idx, (date, x_covar) in enumerate(x_covars.items()):
        # generate aligned betas
        # residual vars are
        asset_last_betas = []
        last_total_vars = []
        last_residual_vars = []
        last_r2 = []
        last_nw_ratios = []
        for freq in group_freqs.keys():
            last_update_date = qis.find_upto_date_from_datetime_index(index=list(betas_freqs[freq].keys()), date=date)
            if last_update_date is None:  # wait until all last dates are valid
                continue
            asset_last_betas.append(betas_freqs[freq][last_update_date])
            if var_scaler_freq_dict is not None:
                scaler = var_scaler_freq_dict[freq]
            else:
                scaler = 1.0
            last_total_vars.append(scaler * total_vars_freqs[freq][last_update_date])
            last_residual_vars.append(scaler * residual_vars_freqs[freq][last_update_date])
            last_r2.append(r2_freqs[freq][last_update_date])
            if is_adjust_for_newey_west:
                last_nw_ratios.append(nw_ratios[freq].loc[last_update_date, :])

        asset_last_betas = pd.concat(asset_last_betas, axis=1)  # pandas with colums = assets
        last_total_vars = pd.concat(last_total_vars, axis=0)  # series
        last_residual_vars = pd.concat(last_residual_vars, axis=0)  # series
        last_r2 = pd.concat(last_r2, axis=0)  # series

        # align
        asset_last_betas = asset_last_betas.reindex(index=x_covar.index).reindex(columns=prices.columns).fillna(0.0)
        last_total_vars = last_total_vars.reindex(index=prices.columns).fillna(0.0)
        last_residual_vars = last_residual_vars.reindex(index=prices.columns).fillna(0.0)
        last_r2 = last_r2.reindex(index=prices.columns).fillna(0.0)

        # compute covar
        betas_np = asset_last_betas.to_numpy()
        betas_covar = np.transpose(betas_np) @ x_covar.to_numpy() @ betas_np
        if not np.isclose(residual_var_weight, 0.0):
            betas_covar += residual_var_weight * np.diag(last_residual_vars.to_numpy())
        betas_covar *= an_factor

        # adjust
        if is_adjust_for_newey_west:
            last_nw_ratios = pd.concat(last_nw_ratios, axis=0)  # series
            last_nw_ratios = last_nw_ratios.reindex(index=prices.columns).fillna(1.0)
            adj = np.sqrt(last_nw_ratios.to_numpy())
            norm = np.outer(adj, adj)
            betas_covar = norm * betas_covar
            last_nw_ratios_t[date] = last_nw_ratios

        y_covars[date] = pd.DataFrame(betas_covar, index=prices.columns, columns=prices.columns)

        # store outs
        asset_last_betas_t[date] = asset_last_betas
        last_total_vars_t[date] = last_total_vars
        last_residual_vars_t[date] = last_residual_vars
        last_r2_vars_t[date] = last_r2

    if is_adjust_for_newey_west:
        last_nw_ratios_pd = pd.DataFrame.from_dict(last_nw_ratios_t, orient='index')
    else:
        last_nw_ratios_pd = None

    covar_data = EstimatedRollingCovarData(x_covars=x_covars,
                                           y_covars=y_covars,
                                           asset_last_betas_t=asset_last_betas_t,
                                           last_residual_vars_t=last_residual_vars_t,
                                           total_vars_pd=pd.DataFrame.from_dict(last_total_vars_t, orient='index'),
                                           residual_vars_pd=pd.DataFrame.from_dict(last_residual_vars_t, orient='index'),
                                           r2_pd=pd.DataFrame.from_dict(last_r2_vars_t, orient='index'),
                                           last_nw_ratios_pd=last_nw_ratios_pd)

    return covar_data
