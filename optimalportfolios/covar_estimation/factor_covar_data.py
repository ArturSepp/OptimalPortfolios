from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
import qis as qis
from enum import Enum
from dataclasses import dataclass, asdict, field
from typing import Optional, Union, Dict, Any, List
from optimalportfolios.covar_estimation.utils import compute_returns_from_prices


class VarianceColumns(str, Enum):
    # provided by estimation
    EWMA_VARIANCE = 'ewma_var'  # ewma-based total var of y
    RESIDUAL_VARS = 'residual_var'  # residual var of y-beta*x
    INSAMPLE_ALPHA = 'insample_alpha' # ewma alpha as mean of residuals, annualised - regression outcome
    R2 = 'r2'  # R-squared per asset: 1 - residual_var/total_var
    # derived
    ALPHA = 'stat_alpha' # statistical alpha of annualised residuals - computed with ewma span of residuals
    TOTAL_VOL = 'total_vol'
    SYST_VOL = 'sys_vol'
    RESID_VOL = 'resid_vol'

    # Alpha estimation approaches:
    # 1) INSAMPLE_ALPHA: In-sample alpha (compute_residual_variance_r2): EWMA-weighted mean of residuals
    #    over the full sample — a summary statistic of average model misfit.
    # 2) ALPHA: Rolling alpha (estimate_alpha): last value of ewm(residuals, span=alpha_span),
    #    a causal filter reflecting recent residual behavior (~2yr half-life for span=36).
    #    Preferred for forward-looking portfolio construction as it adapts to regime changes.
    # When residual mean is non-stationary, the two estimates can diverge materially.


@dataclass(frozen=True)
class CurrentFactorCovarData:
    """
    Factor model covariance decomposition: Sigma_y = beta Sigma_x beta' + D

    Convention (paper):
        beta is (N x M): index=assets, columns=factors
        Sigma_x is (M x M): factor covariance
        Sigma_y is (N x N): asset covariance
        D is (N x N): diagonal residual variances

    where N is number of assets and M is number of factors.
    """

    # Core factor model components
    x_covar: pd.DataFrame       # Factor covariance Sigma_x (M x M): index=factors, columns=factors
    y_betas: pd.DataFrame       # Factor loadings beta (N x M): index=assets, columns=factors
    y_variances: pd.DataFrame   # variances of y: index=assets

    # Metadata
    estimation_date: Optional[pd.Timestamp] = None

    # Optional time series
    residuals: Optional[pd.DataFrame] = None  # eps_t = y_t - x_t @ beta', annualised by a_f

    # Clustering outputs (for HCGL)
    clusters: Optional[Union[Dict[str, pd.Series], pd.Series]] = None
    linkages: Optional[Union[Dict[str, np.ndarray], np.ndarray]] = None
    cutoffs: Optional[Union[Dict[str, float], float]] = None

    def save(self, file_name: str, local_path: str = None) -> None:
        """save to excel with minimum data"""
        data = dict(x_covar=self.x_covar,
                    y_betas=self.y_betas,
                    y_variances=self.y_variances)
        if self.residuals is not None:
            data.update(dict(residuals=self.residuals))
        qis.save_df_to_excel(data=data, file_name=file_name, local_path=local_path)

    @classmethod
    def load(cls, file_name: str, local_path: str = None) -> CurrentFactorCovarData:
        data = qis.load_df_dict_from_excel(file_name=file_name, local_path=local_path)
        return CurrentFactorCovarData(**data)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    def filter_on_tickers(
            self,
            assets: Union[List[str], pd.Index, Dict[str, str]]
    ) -> CurrentFactorCovarData:
        """
        Subset covariance universe to selected assets.

        Args:
            assets: Asset tickers to retain (or dict for renaming).

        Returns:
            New instance with subset. Factor covariance unchanged, residuals dropped.
        """
        if isinstance(assets, dict):
            # y_betas: (N x M), select and rename rows (assets)
            y_betas = self.y_betas.loc[list(assets.keys()), :].rename(index=assets)
            y_variances = self.y_variances.loc[list(assets.keys())].rename(index=assets)
        else:
            y_betas = self.y_betas.loc[assets, :]
            y_variances = self.y_variances.loc[assets]

        if self.residuals is not None:
            if isinstance(assets, dict):
                residuals = self.residuals.loc[:, list(assets.keys())].rename(columns=assets)
            else:
                residuals = self.residuals[assets]
        else:
            residuals = None

        return CurrentFactorCovarData(
            x_covar=self.x_covar,
            y_betas=y_betas,
            y_variances=y_variances,
            residuals=residuals,
            estimation_date=self.estimation_date
        )

    def get_model_vols(self, assets: List[str] = None) -> pd.DataFrame:
        if assets is None:
            assets = self.y_betas.index.tolist()
        # y_betas: (N x M), select rows for assets
        betas_np = self.y_betas.loc[assets, :].values  # (N x M)
        # Sigma_y_systematic = beta @ Sigma_x @ beta'  => diag gives per-asset systematic var
        systematic_var = np.diag(betas_np @ self.x_covar.values @ betas_np.T)
        residual_vars = self.y_variances.loc[assets, VarianceColumns.RESIDUAL_VARS.value].values
        total_vars = systematic_var + residual_vars
        total_vols = pd.Series(np.sqrt(total_vars), index=assets, name=VarianceColumns.TOTAL_VOL.value)
        systematic_vol = pd.Series(np.sqrt(systematic_var), index=assets, name=VarianceColumns.SYST_VOL.value)
        residual_vol = pd.Series(np.sqrt(residual_vars), index=assets, name=VarianceColumns.RESID_VOL.value)
        df = pd.concat([total_vols, systematic_vol, residual_vol], axis=1)
        return df

    def get_snapshot(self,
                     assets: List[str] = None,
                     alpha_span: int = 120
                     ) -> pd.DataFrame:
        """
        Summary table: betas, R², volatilities, alpha per asset.
        """
        assets = assets or self.y_betas.index.tolist()

        # y_betas is (N x M) with index=assets, columns=factors — already in display format
        df = self.y_betas.loc[assets, :].copy()
        vols = self.get_model_vols(assets=assets)

        if self.residuals is not None:
            alphas = self.estimate_alpha(alpha_span=alpha_span).loc[assets].rename(VarianceColumns.ALPHA.value)
        else:
            warnings.warn(f"no residuals data, using statistical alpha")
            alphas = self.y_variances.loc[assets, VarianceColumns.ALPHA.value]

        regression_part = pd.concat([self.y_variances[VarianceColumns.R2.value],
                                     alphas,
                                     self.y_variances[VarianceColumns.INSAMPLE_ALPHA.value]],
                                    axis=1)

        df = pd.concat([df, regression_part, vols], axis=1)

        return df

    def estimate_alpha(self, alpha_span: int = 120) -> pd.Series:
        """
        Estimate alpha from residuals: alpha = EWM(eps_t, span).
        """
        if self.residuals is None:
            raise ValueError("Residuals required for alpha estimation")

        if alpha_span is not None:
            alphas = qis.compute_ewm(self.residuals, span=alpha_span)
            alpha = alphas.iloc[-1, :]
        else:
            alpha = self.residuals.iloc[-1, :]

        return alpha.rename(VarianceColumns.ALPHA.value)

    @property
    def y_covar(self,
                residual_var_weight: float = 1.0,
                assets: Optional[Union[List[str], pd.Index]] = None
                ) -> pd.DataFrame:
        return self.get_y_covar(residual_var_weight=residual_var_weight, assets=assets)

    def get_y_covar(
            self,
            residual_var_weight: float = 1.0,
            assets: Optional[Union[List[str], pd.Index]] = None
    ) -> pd.DataFrame:
        """
        Compute asset covariance with adjustable residual variance weight.

        Sigma_y(w) = beta Sigma_x beta' + w * D

        where beta is (N x M), Sigma_x is (M x M), and w is the
        residual_var_weight controlling the contribution of idiosyncratic risk.

        Args:
            residual_var_weight: Scaling weight for diagonal residual variances.
            assets: Optional subset of assets. None uses all assets.

        Returns:
            Asset covariance matrix (N x N).
        """
        betas = self.y_betas if assets is None else self.y_betas.loc[assets, :]
        residual_vars = self.y_variances[VarianceColumns.RESIDUAL_VARS.value]
        resid = residual_vars if assets is None else residual_vars.loc[assets]
        assets_out = betas.index

        betas_np = betas.values  # (N x M)
        # Sigma_y = beta @ Sigma_x @ beta'
        y_covar = betas_np @ self.x_covar.to_numpy() @ betas_np.T

        if not np.isclose(residual_var_weight, 0.0):
            y_covar += residual_var_weight * np.diag(resid.to_numpy())

        return pd.DataFrame(y_covar, index=assets_out, columns=assets_out)


@dataclass
class RollingFactorCovarData:
    """
    Container for rolling time series of CurrentFactorCovarData.

    Stores Dict[pd.Timestamp, CurrentFactorCovarData] and provides
    convenient accessors for extracting time series of components.
    """

    data: Dict[pd.Timestamp, CurrentFactorCovarData] = field(default_factory=dict)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return pd.DatetimeIndex(sorted(self.data.keys()))

    @property
    def n_observations(self) -> int:
        return len(self.data)

    def __getitem__(self, date: pd.Timestamp) -> CurrentFactorCovarData:
        return self.data[date]

    def __len__(self) -> int:
        return len(self.data)

    def __iter__(self):
        return iter(sorted(self.data.keys()))

    def add(self, date: pd.Timestamp, estimation: CurrentFactorCovarData) -> None:
        self.data[date] = estimation

    def get_latest(self) -> CurrentFactorCovarData:
        date = max(self.data.keys())
        return self.data[date]

    # --- Matrix time series accessors ---

    def get_x_covars(self) -> Dict[pd.Timestamp, pd.DataFrame]:
        """Factor covariance matrices over time."""
        return {date: est.x_covar for date, est in sorted(self.data.items())}

    @property
    def y_covars(self, residual_var_weight: float = 1.0,
                     assets: Optional[Union[List[str], pd.Index]] = None
                 ) -> Dict[pd.Timestamp, pd.DataFrame]:
        return self.get_y_covars(residual_var_weight=residual_var_weight,
                                 assets=assets)

    def get_y_covars(self,
                     residual_var_weight: float = 1.0,
                     assets: Optional[Union[List[str], pd.Index]] = None
                     ) -> Dict[pd.Timestamp, pd.DataFrame]:
        """Asset covariance matrices over time."""
        return {date: est.get_y_covar(residual_var_weight=residual_var_weight, assets=assets)
                for date, est in sorted(self.data.items())}

    def get_y_betas(self) -> Dict[pd.Timestamp, pd.DataFrame]:
        """Factor loadings over time. Each DataFrame is (N x M)."""
        return {date: est.y_betas for date, est in sorted(self.data.items())}

    # --- Panel DataFrame accessors ---

    def get_residual_vars(self) -> pd.DataFrame:
        """Residual variances: index=dates, columns=assets."""
        return pd.DataFrame(
            {date: est.y_variances[VarianceColumns.RESIDUAL_VARS.value] for date, est in sorted(self.data.items())}
        ).T

    def get_ewma_vars(self) -> pd.DataFrame:
        """ewma variances: index=dates, columns=assets."""
        return pd.DataFrame(
            {date: est.y_variances[VarianceColumns.EWMA_VARIANCE.value] for date, est in sorted(self.data.items())}
        ).T

    def get_r2(self) -> pd.DataFrame:
        """R-squared: index=dates, columns=assets."""
        return pd.DataFrame(
            {date: est.y_variances[VarianceColumns.R2.value] for date, est in sorted(self.data.items())}
        ).T

    def get_total_vols(self) -> pd.DataFrame:
        """Total vols (sqrt of total_vars): index=dates, columns=assets."""
        return np.sqrt(self.get_total_vars())  # type: ignore

    def get_residual_vols(self) -> pd.DataFrame:
        """Residual vols: index=dates, columns=assets."""
        return np.sqrt(self.get_residual_vars())  # type: ignore

    def get_alphas(self, alpha_span: int = 120) -> pd.DataFrame:
        """Alphas: index=dates, columns=assets."""
        records = {}
        for date, est in sorted(self.data.items()):
            if est.residuals is not None:
                records[date] = est.estimate_alpha(alpha_span=alpha_span)
            else:
                 records[date] = est.y_variances[VarianceColumns.INSAMPLE_ALPHA.value]
        return pd.DataFrame(records).T if records else pd.DataFrame()

    def get_factor_var(self, factor: str) -> pd.Series:
        """Variance of a single factor over time."""
        return pd.Series(
            {date: est.x_covar.loc[factor, factor]
             for date, est in sorted(self.data.items())},
            name=factor
        )

    def get_beta(self, factor: str) -> pd.DataFrame:
        """
        Single factor loadings over time: index=dates, columns=assets.

        y_betas is (N x M) with index=assets, columns=factors, so we
        select the factor column and transpose into a time series.
        """
        return pd.DataFrame(
            {date: est.y_betas[factor] for date, est in sorted(self.data.items())}
        ).T

    def filter_on_tickers(self, tickers: Union[List[str], pd.Index]) -> RollingFactorCovarData:
        """Subset all estimations to selected assets."""
        return RollingFactorCovarData(
            data={date: est.filter_on_tickers(tickers)
                  for date, est in self.data.items()}
        )

    def get_snapshot(self, alpha_span: int = 120) -> Dict[pd.Timestamp, pd.DataFrame]:
        dfs = {date: est.get_snapshot(alpha_span=alpha_span) for date, est in self.data.items()}
        return dfs

    def get_linear_factor_model(
            self,
            x_factors: pd.DataFrame,
            y_assets: pd.DataFrame,
            to_returns: bool = True,
            demean: bool = True,
            span: Optional[int] = 36
    ) -> qis.LinearModel:
        """
        Construct a LinearModel from rolling factor covariance estimations.

        Extracts time-varying factor loadings from the rolling estimations and
        pairs them with factor/asset return series and covariance universe to build
        a qis.LinearModel for performance attribution and risk decomposition.

        The factor loadings are restructured from per-date beta matrices
        beta_t (N x M) into per-factor time series DataFrames (T x N), where
        T is the number of estimation dates, M is factors, and N is assets.

        Args:
            x_factors: Factor prices or returns. Index=dates, columns=factor names.
            y_assets: Asset prices or returns. Index=dates, columns=asset tickers.
            to_returns: If True, convert prices to returns at estimation dates.
            demean: If True, demean returns using EWMA.
            span: EWMA span for demeaning.

        Returns:
            LinearModel with time-varying loadings, factor covariances, and residual variances.
        """
        if self.n_observations == 0:
            raise ValueError("Rolling universe is empty, cannot construct LinearModel")

        estimation_dates = list(self.dates)

        # Extract per-factor loading time series: {factor: DataFrame(T x N)}
        # y_betas is (N x M) with index=assets, columns=factors
        first_estimation = self.data[estimation_dates[0]]
        factors = first_estimation.y_betas.columns  # factor names from columns

        factor_loadings: Dict[str, pd.DataFrame] = {}
        for factor in factors:
            factor_betas = {
                date: est.y_betas[factor]  # Series with index=assets
                for date, est in sorted(self.data.items())
            }
            factor_loadings[factor] = pd.DataFrame.from_dict(factor_betas, orient='index')

        # Align inputs to estimation dates
        if to_returns:
            x_aligned = compute_returns_from_prices(
                prices=x_factors.reindex(index=estimation_dates).ffill(),
                returns_freq=None, demean=demean, span=span,
                drop_first=False, is_first_zero=True
            )
            y_aligned = compute_returns_from_prices(
                prices=y_assets.reindex(index=estimation_dates).ffill(),
                returns_freq=None, demean=demean, span=span,
                drop_first=False, is_first_zero=True
            )
        else:
            x_aligned = x_factors.reindex(index=estimation_dates)
            y_aligned = y_assets.reindex(index=estimation_dates)

        linear_model = qis.LinearModel(
            x=x_aligned,
            y=y_aligned,
            loadings=factor_loadings,
            x_covars=self.get_x_covars(),
            residual_vars=self.get_residual_vars()
        )
        return linear_model
