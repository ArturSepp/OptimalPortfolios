"""Estimate monthly FM/FL risk with the ROSAA TAA production configuration."""
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

import cvxpy as cp
import numpy as np
import pandas as pd
import qis
import factormodels as fm
from factorlasso import ClusterSmootherType, LassoModel, LassoModelType
from factorlasso.cluster import DependenceMeasure, DistanceTransform

from examples.oss_portfolio_pipeline.config import PipelineConfig
from examples.oss_portfolio_pipeline.data import external_directory, validate_prices
from examples.oss_portfolio_pipeline.factors import MacroFactor, month_end_dates
from examples.oss_portfolio_pipeline.pipeline import PipelineInputs


def production_estimator(solver: str = 'MOSEK') -> fm.FactorCovarEstimator:
    """Pin ROSAA TAA settings; public factors use automatic highest-R2 OLS selection.

    No proprietary sign table, expert factor selection or ROSAA import is required.
    The solver override is explicit and never used as an automatic fallback.
    """
    if solver not in ('MOSEK', 'CLARABEL'):
        raise ValueError('solver must be MOSEK or CLARABEL')
    model = LassoModel(
        model_type=LassoModelType.FACTOR_CLUSTER_GROUP_LASSO,
        span=36, span_freq_dict={'ME': 36, 'WOM-2FRI': 36, 'QE': 12},
        reg_lambda=1e-5, l1_weight=0.0, group_penalty='normalized',
        loss_normalization='weight_sum', demean=True, warmup_period=12,
        solver=solver, solver_fallbacks=None,
        cutoff_fraction=0.60, linkage_method='ward',
        distance_transform=DistanceTransform.ONE_MINUS_RHO,
        dependence_measure=DependenceMeasure.PEARSON,
        gerber_threshold=0.5, n_clusters=None,
        cluster_smoother_type=ClusterSmootherType.NONE,
        smoother_delta=0.05, smoother_lambda=0.7, recluster_freq=None,
        cluster_correlation_span=None, cluster_correlation_span_freq_dict=None,
        auto_sign_constraints=True, auto_sign_adaptive_weights=True,
        auto_sign_threshold_t=1.0, auto_sign_adaptive_floor=0.5,
        auto_sign_use_fit_span=True, auto_sign_variance='date',
        auto_sign_excluded_factors=None,
        apply_ols_prior=True, prior_selection_type='highest_r2',
        expert_prior_bound_n_std=1.0, expert_prior_hac_lags=3,
        expert_prior_hac_lags_freq_dict={'ME': 3, 'WOM-2FRI': 3, 'QE': 1},
        expert_prior_bound_centre='prior', named_set_max_vif=None,
    )
    return fm.FactorCovarEstimator(
        lasso_model=model, factor_returns_freq='W-WED', factor_covar_span=52,
        rebalancing_freq='ME', is_apply_vol_normalised_returns=False,
        include_factors_in_clustering=False, residual_type='empirical',
        residual_covar_freq=None, residual_covar_span=None,
        residual_corr_weight=1.0, residual_var_weight=1.0,
        factor_return_types={factor.value: qis.ReturnTypes.LOG for factor in MacroFactor},
    )


@dataclass
class RollingCovariances:
    """FM calendar-ME snapshots and their observed-session mapping for OP consumption."""

    model: fm.RollingFactorModel
    asset_excess_returns: pd.DataFrame
    schedule: pd.DataFrame
    settings: dict

    @property
    def covariances(self) -> dict[pd.Timestamp, pd.DataFrame]:
        """Return selected annual asset covariances keyed by the actual ME trading session.

        Every calendar snapshot uses prices available at that session's close. These
        matrices can determine holdings for subsequent returns, never that day's return.
        """
        return {self.schedule.loc[date, 'observation_date']: covar
                for date, covar in self.model.covariances.items()}


def _json_value(value):
    """Convert public estimator parameters and provenance into portable JSON values."""
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.generic):
        return value.item()
    return value


def estimate_covariances(inputs: PipelineInputs, config: PipelineConfig,
                         solver: str = 'MOSEK') -> RollingCovariances:
    """Fit expanding histories at completed calendar MEs, beginning 31 December 2019.

    QIS converts BIL-relative adjusted prices to monthly excess log returns. The
    initial calendar-ME level is retained as a NaN return anchor for FM alignment.
    FM estimates factor covariance and orchestrates FL regression/residual fitting.
    """
    estimator = production_estimator(solver)
    if solver not in cp.installed_solvers():
        raise RuntimeError(f'{solver} is unavailable; install it or explicitly select CLARABEL')
    names = tuple(inputs.universe.index)
    prices = validate_prices(inputs.prices, names)
    factors = validate_prices(inputs.factor_navs, tuple(f.value for f in MacroFactor))
    if not factors.index.isin(prices.index).all():
        raise ValueError('Every factor date must have an asset observation')
    end = pd.Timestamp(config.end) - pd.Timedelta(days=1)
    dates = pd.date_range(factors.index[0], end, freq='ME')
    sessions = month_end_dates(prices.loc[factors.index[0]:].index, config.end)
    if len(dates) != len(sessions) or len(dates) < 14:
        raise ValueError('Need a common monthly panel and at least 12 return observations')
    if (sessions > factors.index[-1]).any():
        raise ValueError('Factor history does not cover the completed monthly observations')
    # The calendar labels describe completed periods; values come strictly from the
    # last observed session. FL empirical residuals require this exact calendar grid.
    monthly = prices.loc[sessions].copy()
    monthly.index = dates
    relative_prices = monthly.div(monthly['BIL'], axis=0)
    returns = qis.to_returns(relative_prices, is_log_returns=True, ffill_nans=False)
    updates = dates[dates >= pd.Timestamp(config.backtest_start)]
    if updates.empty or updates[0] != pd.Timestamp(config.backtest_start):
        raise ValueError('Backtest start must be a completed calendar month-end')
    if returns.loc[:updates[0]].iloc[1:].dropna().shape[0] < 12:
        raise ValueError('First covariance update needs at least 12 complete monthly returns')
    # Cash relative to itself has exactly zero risk. Exclude its identically-zero
    # response from pooled priors; FM's explicit output universe restores a zero row.
    model = estimator.fit_rolling(
        risk_factor_prices=factors, asset_returns_dict={'ME': returns.drop(columns='BIL')},
        time_period=qis.TimePeriod(updates[0], updates[-1]), assets=list(names))
    if not pd.DatetimeIndex(model.snapshots).equals(updates):
        raise RuntimeError('FM returned an unexpected covariance schedule')
    observations = pd.Series(sessions, index=dates).loc[updates]
    schedule = observations.rename('observation_date').to_frame()
    schedule.index.name = 'model_date'
    # Guard the calendar-to-session mapping: only those observed closes may feed a fit.
    for date, snapshot in model.snapshots.items():
        if snapshot.provenance['factor_observation_date'] > schedule.loc[date, 'observation_date']:
            raise ValueError('A factor observation occurs after the intended trading session')
        if not snapshot.asset_covariance.index.equals(inputs.universe.index):
            raise RuntimeError('FM changed the asset ordering')
    settings = {name: value for name, value in vars(estimator).items() if name != 'lasso_model'}
    settings['lasso_model'] = estimator.lasso_model.get_params()
    return RollingCovariances(model, returns, schedule, _json_value(settings))


def save_covariances(result: RollingCovariances, output_dir: Path | str) -> Path:
    """Write hashed model tables and provenance beside the previously saved input bundle."""
    output = external_directory(output_dir)
    input_manifest = output / 'manifest.json'
    if not input_manifest.is_file():
        raise ValueError('Save the pipeline inputs before saving covariance estimates')
    snapshots = result.model.snapshots
    tables = {'asset_excess_monthly_log_returns.csv': result.asset_excess_returns,
              'covariance_schedule.csv': result.schedule}
    fields = ('factor_covariance', 'loadings', 'residual_covariance', 'asset_covariance',
              'fit_statistics', 'clusters', 'derived_signs')
    for field in fields:
        tables[f'{field}.csv'] = pd.concat(
            {date: getattr(snapshot, field) for date, snapshot in snapshots.items()},
            names=['model_date', 'factor' if field == 'factor_covariance' else 'asset'])
    hashes = {}
    for name, table in tables.items():
        content = table.to_csv(float_format='%.17g').encode('utf-8')
        (output / name).write_bytes(content)
        hashes[name] = hashlib.sha256(content).hexdigest()
    manifest = {
        'built_at_utc': datetime.now(timezone.utc).isoformat(),
        'input_manifest_sha256': hashlib.sha256(input_manifest.read_bytes()).hexdigest(),
        'file_sha256': hashes, 'settings': result.settings,
        'configuration_reference': 'ROSAA TAA ProdCovarEstimationSpec, 2026-10-26',
        'prior_selection': 'automatic highest_r2; no proprietary expert selections or signs',
        'asset_returns': 'monthly log changes of adjusted ETF price / adjusted BIL price',
        'cash_numeraire': 'BIL excluded from regression; FM restores zero risk in output universe',
        'factor_returns': 'log changes of financed excess NAVs; no second cash debit',
        'covariance_units': 'annual log-return squared; factors W-WED x52, residuals ME x12',
        'schedule': 'calendar ME snapshots; observation_date maps to actual session close',
        'application': 'weights decided at observation_date apply to subsequent returns',
        'provenance': {str(date.date()): _json_value(snapshot.provenance)
                       for date, snapshot in snapshots.items()},
        'versions': {name: version(name) for name in
                     ('factormodels', 'factorlasso', 'qis', 'cvxpy', 'pandas', 'numpy')},
    }
    (output / 'covariance_manifest.json').write_text(
        json.dumps(manifest, indent=2, allow_nan=False), encoding='utf-8')
    return output
