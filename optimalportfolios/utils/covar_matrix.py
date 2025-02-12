"""
some utilities for estimation of covariance matrices
"""
import pandas as pd
import numpy as np
import qis as qis
from typing import Union, Optional, Dict
from dataclasses import dataclass
from enum import Enum
from optimalportfolios.utils.lasso import LassoModel


class CovarEstimatorType(Enum):
    EWMA = 1
    LASSO = 2


@dataclass
class CovarEstimator:
    """
    summarise estimator specs here
    """
    covar_estimator_type: CovarEstimatorType = CovarEstimatorType.EWMA
    returns_freq: str = 'W-WED'  # sampling frequency of returns
    rebalancing_freq: Union[str, pd.Series] = 'QE'  # rebalancing frequency for rolling estimators
    span: int = 52  # span for ewma estimate
    is_apply_vol_normalised_returns: bool = False  # for ewma
    demean: bool = True  # adjust for mean
    squeeze_factor: Optional[float] = None  # squeezing factor
    lasso_model: LassoModel = None  # for lasso estimator
    residual_var_weight: float = 1.0  # for lasso covars
    risk_factor_prices: pd.DataFrame = None  # for lasso covars

    def fit_rolling_covars(self,
                           prices: pd.DataFrame,
                           time_period: qis.TimePeriod
                           ) -> Dict[pd.Timestamp, pd.DataFrame]:
        """
        fit rolling covars at rebalancing_freq
        """
        covars = wrapper_estimate_rolling_covar(prices=prices,
                                                time_period=time_period,
                                                risk_factor_prices=self.risk_factor_prices,
                                                covar_estimator=self)
        return covars


def wrapper_estimate_rolling_covar(prices: pd.DataFrame,
                                   time_period: qis.TimePeriod,  # starting time of sampling estimator
                                   risk_factor_prices: pd.DataFrame = None,  # for lasso covars
                                   covar_estimator: CovarEstimator = CovarEstimator()
                                   ) -> Dict[pd.Timestamp, pd.DataFrame]:
    """
    wrap several methods for covariance estimation
    """
    if covar_estimator.covar_estimator_type == CovarEstimatorType.EWMA:
        if not isinstance(covar_estimator.rebalancing_freq, str):
            raise ValueError(f"covar_estimator_specs.rebalancing_freq must be str for EWMA stimator")
        covars = estimate_rolling_ewma_covar(prices=prices,
                                             time_period=time_period,
                                             returns_freq=covar_estimator.returns_freq,
                                             span=covar_estimator.span,
                                             is_apply_vol_normalised_returns=covar_estimator.is_apply_vol_normalised_returns,
                                             demean=covar_estimator.demean,
                                             squeeze_factor=covar_estimator.squeeze_factor)

    elif covar_estimator.covar_estimator_type == CovarEstimatorType.LASSO:
        if risk_factor_prices is None:
            raise ValueError(f"risk_factor_prices must be passed for Lasso estimator")
        if covar_estimator.lasso_model is None:
            raise ValueError(f"lasso_model must be passed for Lasso estimator")
        covars = wrapper_estimate_rolling_lasso_covar(risk_factors_prices=risk_factor_prices,
                                                      prices=prices,
                                                      time_period=time_period,  # when we start building portfolios
                                                      lasso_model=covar_estimator.lasso_model,
                                                      returns_freq=covar_estimator.returns_freq,
                                                      rebalancing_freq=covar_estimator.rebalancing_freq,
                                                      span=covar_estimator.span,  # 1y of weekly returns
                                                      is_apply_vol_normalised_returns=covar_estimator.is_apply_vol_normalised_returns,
                                                      squeeze_factor=covar_estimator.squeeze_factor,
                                                      residual_var_weight=covar_estimator.residual_var_weight)

    else:
        raise NotImplementedError(f"covar_estimator_specs.covar_estimator={covar_estimator.covar_estimator_type}")

    return covars


def estimate_rolling_ewma_covar(prices: pd.DataFrame,
                                time_period: qis.TimePeriod,  # when we start building portfolios
                                returns_freq: str = 'W-WED',
                                rebalancing_freq: str = 'QE',
                                span: int = 52,
                                is_apply_vol_normalised_returns: bool = False,
                                demean: bool = True,
                                squeeze_factor: Optional[float] = None,
                                apply_an_factor: bool = True
                                ) -> Dict[pd.Timestamp, pd.DataFrame]:
    """
    compute ewma covar matrix: supporting for nans in prices
    output is dict[estimation timestamp, pd.Dataframe(estimated_covar)
    """
    returns = qis.to_returns(prices=prices, is_log_returns=True, drop_first=True, freq=returns_freq)
    returns_np = returns.to_numpy()
    if demean:
        x = returns_np - qis.compute_ewm(returns_np, span=span)
    else:
        x = returns_np

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


def estimate_rolling_lasso_covar(risk_factor_prices: pd.DataFrame,
                                 prices: pd.DataFrame,
                                 time_period: qis.TimePeriod,  # when we start building portfolios
                                 lasso_model: LassoModel,
                                 returns_freq: str = 'W-WED',
                                 rebalancing_freq: str = 'QE',
                                 span: int = 52,  # 1y of weekly returns
                                 is_apply_vol_normalised_returns: bool = False,
                                 squeeze_factor: Optional[float] = None,
                                 residual_var_weight: float = 1.0
                                 ) -> Dict[pd.Timestamp, pd.DataFrame]:
    """
    use benchmarks to compute the benchmark covar matrix
    use lasso to estimate betas
    compute covar matrix
    """
    # 1. compute x-factors ewm covar at rebalancing freq
    x_covars = estimate_rolling_ewma_covar(prices=risk_factor_prices,
                                           time_period=time_period,
                                           returns_freq=returns_freq,
                                           rebalancing_freq=rebalancing_freq,
                                           span=span,
                                           demean=lasso_model.demean,
                                           is_apply_vol_normalised_returns=is_apply_vol_normalised_returns,
                                           squeeze_factor=squeeze_factor,
                                           apply_an_factor=False)

    # 2. estimate betas of y-returns at different samples
    y = qis.to_returns(prices=prices, is_log_returns=True, drop_first=True, freq=rebalancing_freq)
    x = qis.to_returns(prices=risk_factor_prices, is_log_returns=True, drop_first=True, freq=rebalancing_freq)
    betas, residual_vars, r2_t = lasso_model.estimate_rolling_betas(x=x, y=y)

    # 3. compute y_covars at x_covars frequency
    an_factor = qis.infer_an_from_data(data=x)
    y_covars = {}
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
        y_covars[date] = pd.DataFrame(an_factor*betas_covar, index=prices.columns, columns=prices.columns)

    return y_covars


def estimate_rolling_lasso_covar_different_freq(risk_factor_prices: pd.DataFrame,
                                                prices: pd.DataFrame,
                                                rebalancing_freqs: pd.Series,
                                                time_period: qis.TimePeriod,  # when we start building portfolios
                                                lasso_model: LassoModel,
                                                returns_freq: str = 'W-WED',
                                                rebalancing_freq: str = 'ME',  # for x returns
                                                is_apply_vol_normalised_returns: bool = False,
                                                span: int = 52,  # 1y of weekly returns
                                                span_freq_dict: Optional[Dict[str, int]] = None,  # spans for different freqs
                                                var_scaler_freq_dict: Optional[Dict[str, float]] = None,  # var scaler for different freqs
                                                squeeze_factor: Optional[float] = None,
                                                residual_var_weight: float = 1.0
                                                ) -> Dict[pd.Timestamp, pd.DataFrame]:
    """
    use benchmarks to compute the benchmark covar matrix
    use lasso to estimate betas
    compute covar matrix
    rebalancing_freq is rebalancing for x returns
    rebalancing_freqs is rebalancing for y  returns
    """
    # 1. compute x-factors ewm covar at rebalancing freq
    x_covars = estimate_rolling_ewma_covar(prices=risk_factor_prices,
                                           time_period=time_period,
                                           returns_freq=returns_freq,
                                           rebalancing_freq=rebalancing_freq,
                                           span=span,
                                           demean=lasso_model.demean,
                                           is_apply_vol_normalised_returns=is_apply_vol_normalised_returns,
                                           squeeze_factor=squeeze_factor,
                                           apply_an_factor=False)

    # 2. estimate betas of y-returns at different samples
    rebalancing_freqs = rebalancing_freqs[prices.columns]
    group_freqs = qis.get_group_dict(group_data=rebalancing_freqs)
    betas_freqs: Dict[str, Dict[pd.Timestamp, pd.DataFrame]] = {}
    residual_vars_freqs: Dict[str, Dict[pd.Timestamp, pd.Series]] = {}
    for freq, asset_tickers in group_freqs.items():
        y = qis.to_returns(prices=prices[asset_tickers], is_log_returns=True, drop_first=True, freq=freq)
        x = qis.to_returns(prices=risk_factor_prices, is_log_returns=True, drop_first=True, freq=freq)
        if span_freq_dict is not None:
            span_f = span_freq_dict[freq]
        else:
            span_f = span
        betas_freqs[freq], residual_vars_freqs[freq], r2_t = lasso_model.estimate_rolling_betas(x=x, y=y, span=span_f)

    # 3. compute y_covars at x_covars frequency
    _, an_factor = qis.get_period_days(rebalancing_freq)  # an factor is frequency of x returns
    y_covars = {}
    for idx, (date, x_covar) in enumerate(x_covars.items()):
        # generate aligned betas
        # residual vars are
        asset_last_betas = []
        last_residual_vars = []
        for freq in group_freqs.keys():
            last_update_date = qis.find_upto_date_from_datetime_index(index=list(betas_freqs[freq].keys()), date=date)
            if last_update_date is None:  # wait until all last dates are valie
                continue
            asset_last_betas.append(betas_freqs[freq][last_update_date])
            if var_scaler_freq_dict is not None:
                scaler = var_scaler_freq_dict[freq]
            else:
                scaler = 1.0
            last_residual_vars.append(scaler*residual_vars_freqs[freq][last_update_date])
        asset_last_betas = pd.concat(asset_last_betas, axis=1)  # pandas with colums = assets
        last_residual_vars = pd.concat(last_residual_vars, axis=0)  # series

        # align
        asset_last_betas = asset_last_betas.reindex(index=x_covar.index).reindex(columns=prices.columns).fillna(0.0)
        last_residual_vars = last_residual_vars.reindex(index=prices.columns).fillna(0.0)

        # compute covar
        betas_np = asset_last_betas.to_numpy()
        betas_covar = np.transpose(betas_np) @ x_covar.to_numpy() @ betas_np
        if not np.isclose(residual_var_weight, 0.0):
            betas_covar += residual_var_weight * np.diag(last_residual_vars.to_numpy())
        betas_covar *= an_factor
        y_covars[date] = pd.DataFrame(betas_covar, index=prices.columns, columns=prices.columns)
    return y_covars


def wrapper_estimate_rolling_lasso_covar(risk_factors_prices: pd.DataFrame,
                                         prices: pd.DataFrame,
                                         rebalancing_freq: Union[str, pd.Series],
                                         time_period: qis.TimePeriod,  # when we start building portfolios
                                         lasso_model: LassoModel,
                                         returns_freq: str = 'W-WED',
                                         span: int = 52,  # 1y of weekly returns
                                         span_freq_dict: Optional[Dict[str, int]] = None,  # spans for different freqs
                                         var_scaler_freq_dict: Optional[Dict[str, float]] = None,  # var scaler for different freqs
                                         squeeze_factor: Optional[float] = None,
                                         is_apply_vol_normalised_returns: bool = False,
                                         residual_var_weight: float = 1.0
                                         ) -> Dict[pd.Timestamp, pd.DataFrame]:
    """
    wrapper for lasso covar estimation using either fixed rebalancing frequency or rolling rebalancing frequency
    """
    if isinstance(rebalancing_freq, str):
        covar_dict = estimate_rolling_lasso_covar(risk_factor_prices=risk_factors_prices,
                                                  prices=prices,
                                                  time_period=time_period,
                                                  lasso_model=lasso_model,
                                                  returns_freq=returns_freq,
                                                  rebalancing_freq=rebalancing_freq,
                                                  span=span,
                                                  is_apply_vol_normalised_returns=is_apply_vol_normalised_returns,
                                                  squeeze_factor=squeeze_factor,
                                                  residual_var_weight=residual_var_weight)
    else:
        covar_dict = estimate_rolling_lasso_covar_different_freq(risk_factor_prices=risk_factors_prices,
                                                                 prices=prices,
                                                                 time_period=time_period,
                                                                 lasso_model=lasso_model,
                                                                 returns_freq=returns_freq,
                                                                 rebalancing_freqs=rebalancing_freq,
                                                                 span=span,
                                                                 span_freq_dict=span_freq_dict,
                                                                 var_scaler_freq_dict=var_scaler_freq_dict,
                                                                 is_apply_vol_normalised_returns=is_apply_vol_normalised_returns,
                                                                 squeeze_factor=squeeze_factor,
                                                                 residual_var_weight=residual_var_weight)
    return covar_dict


def estimate_lasso_covar(x: pd.DataFrame,
                         y: pd.DataFrame,
                         lasso_model: LassoModel,
                         covar_x: Optional[np.ndarray] = None,
                         span: Optional[int] = None,
                         squeeze_factor: Optional[float] = None,
                         is_apply_vol_normalised_returns: bool = False,
                         verbose: bool = False,
                         residual_var_weight: float = 1.0
                         ) -> pd.DataFrame:
    """
    covar = benchmarks covar N*N
    betas = benachmark * asset: N*M
    betas covar = betas.T @ covar @ betas: M*M
    """
    # 1. compute covar and take the last value
    if covar_x is None:
        if is_apply_vol_normalised_returns:
            covar_tensor_txy, _, _ = qis.compute_ewm_covar_tensor_vol_norm_returns(a=x.to_numpy(), span=span,
                                                                                   nan_backfill=qis.NanBackfill.ZERO_FILL)
        else:
            covar_tensor_txy = qis.compute_ewm_covar_tensor(a=x.to_numpy(), span=span, nan_backfill=qis.NanBackfill.ZERO_FILL)
        covar_x = covar_tensor_txy[-1]

        if squeeze_factor is not None:
            covar_x = squeeze_covariance_matrix(covar=covar_x, squeeze_factor=squeeze_factor)

    # 2. estimate betas
    betas, residual_vars, r2_t = lasso_model.fit(x=x, y=y, verbose=verbose).get_betas_residual_var_r2()

    # make sure aligned
    betas = betas.reindex(index=x.columns).reindex(columns=y.columns).fillna(0.0)
    betas_np = betas.to_numpy()
    betas_covar = np.transpose(betas_np) @ covar_x @ betas_np
    if not np.isclose(residual_var_weight, 0.0):
        betas_covar += residual_var_weight * np.diag(residual_vars.to_numpy())
    betas_covar = pd.DataFrame(betas_covar, index=y.columns, columns=y.columns)
    return betas_covar


def squeeze_covariance_matrix(covar: Union[np.ndarray, pd.DataFrame],
                              squeeze_factor: Optional[float] = 0.05,
                              is_preserve_variance: bool = True
                              ) -> Union[np.ndarray, pd.DataFrame]:
    """
    Adjusts the covariance matrix by applying a squeezing factor to eigenvalues.
    Smaller eigenvalues are reduced to mitigate noise.
    for the methodology see SSRN paper
    Squeezing Financial Noise: A Novel Approach to Covariance Matrix Estimation
    https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4986939
    """
    if squeeze_factor is None or np.isclose(squeeze_factor, 0.0):
        return covar

    # need to create pd.Dataframe for keeping track of good indices
    if isinstance(covar, pd.DataFrame):
        cov_matrix_pd = covar.copy()
    else:
        cov_matrix_pd = pd.DataFrame(covar)

    # filter out nans and zero variances
    variances = np.diag(cov_matrix_pd.to_numpy())
    is_good_asset = np.where(np.logical_and(np.greater(variances, 0.0), np.isnan(variances) == False))
    good_tickers = cov_matrix_pd.columns[is_good_asset]
    clean_covar_pd = cov_matrix_pd.loc[good_tickers, good_tickers]
    clean_covar_np = clean_covar_pd.to_numpy()

    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eigh(clean_covar_np)

    # Squeeze smaller eigenvalues (simple threshold-based squeezing)
    squeezed_eigenvalues = np.array([np.maximum(eigenvalue, squeeze_factor * np.max(eigenvalues))
                                     for eigenvalue in eigenvalues])

    # Reconstruct squeezed covariance matrix
    squeezed_cov_matrix = eigenvectors @ np.diag(squeezed_eigenvalues) @ eigenvectors.T

    if is_preserve_variance:
        # adjustment should be applied to off-dioagonal elements too otherwise we may end up with noncosistent matrix
        original_variance = np.diag(clean_covar_np)
        squeezed_variance = np.diag(squeezed_cov_matrix)
        adjustment_ratio = np.sqrt(original_variance / squeezed_variance)
        norm = np.outer(adjustment_ratio, adjustment_ratio)
        squeezed_cov_matrix = norm*squeezed_cov_matrix

    # now extend back
    squeezed_cov_matrix_pd = pd.DataFrame(squeezed_cov_matrix, index=good_tickers, columns=good_tickers)
    # reindex for all tickers and fill nans with zeros
    all_tickers = cov_matrix_pd.columns
    squeezed_cov_matrix = squeezed_cov_matrix_pd.reindex(index=all_tickers).reindex(columns=all_tickers).fillna(0.0)

    if isinstance(covar, np.ndarray):  # match return to original type
        squeezed_cov_matrix = squeezed_cov_matrix.to_numpy()

    return squeezed_cov_matrix


