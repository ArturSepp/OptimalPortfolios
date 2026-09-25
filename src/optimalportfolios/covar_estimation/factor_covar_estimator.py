"""Integrate FactorLasso regression with QIS factor covariance estimation.

FactorLasso owns sparse fitting, clustering and decomposition containers.
This module aligns factor prices to asset-return cadences, annualizes and
merges fitted components, and provides current and rolling estimator APIs.
QIS owns factor-return construction, EWMA covariance and date utilities.

The shared methods return asset covariance matrices. Factor-specific methods
return CurrentFactorCovarData or RollingFactorCovarData. Covariance combines
the factor component with diagonal (default) or prepared empirical residual risk;
betas are dimensionless. Empirical residual covariance uses complete common log-return
periods and annual units, with native frequency/span/scale metadata supplied here.
Supplied factor covariance must already have compatible annual units.

An orthogonal current fit without factor references does not truncate inputs from
an estimation_date label. Empirical current fits truncate inputs at that cutoff.
The rolling wrapper slices each input through the scheduled date. See
docs/covariance_estimators.md for cutoff, normalization and demeaning qualifications.

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
from dataclasses import dataclass, asdict, fields, replace

from optimalportfolios.covar_estimation.covar_estimator import CovarEstimator
from optimalportfolios.covar_estimation.ewma_covar_estimator import estimate_current_ewma_covar
from factorlasso import (
    ClusterSmootherType,
    CurrentFactorCovarData,
    LassoModel,
    LassoModelType,
    RollingFactorCovarData,
    VarianceColumns,
    compute_rolling_smoothed_clusters,
)


# Backward compatibility shim — older factorlasso releases predate the
# ``derived_signs`` field on CurrentFactorCovarData. Detect support once
# at import time and skip the kwarg at construction sites when absent.
# This lets the LASSO sign-constraint matrix flow into Excel output on
# upgraded factorlasso without forcing a hard version pin from this side.
_CFCD_SUPPORTS_DERIVED_SIGNS = (
    'derived_signs' in {f.name for f in fields(CurrentFactorCovarData)}
)
_CFCD_SUPPORTS_RESIDUAL_CORRELATION = (
    'residual_correlation' in {f.name for f in fields(CurrentFactorCovarData)}
)


def _model_for_frequency(lasso_model: LassoModel, freq: str) -> LassoModel:
    """Resolve regression and clustering spans for one return cadence.

    Args:
        lasso_model: FactorLasso configuration, optionally carrying cadence maps.
        freq: Asset-return frequency code used to look up each configured map.

    Returns:
        A model copy with span overrides when maps are present; otherwise the
        supplied model itself. Spans count observations at the requested cadence.

    Raises:
        KeyError: If a configured regression or clustering span map lacks freq.
    """
    overrides = {}
    if lasso_model.span_freq_dict is not None:
        if freq not in lasso_model.span_freq_dict:
            raise KeyError(f"no span for freq={freq} in lasso_model.span_freq_dict")
        overrides['span'] = lasso_model.span_freq_dict[freq]
    cluster_span_map = getattr(
        lasso_model, 'cluster_correlation_span_freq_dict', None
    )
    if cluster_span_map is not None:
        if freq not in cluster_span_map:
            raise KeyError(
                f"no cluster correlation span for freq={freq} in "
                "lasso_model.cluster_correlation_span_freq_dict"
            )
        overrides['cluster_correlation_span'] = cluster_span_map[freq]
    return lasso_model.copy(kwargs=overrides) if overrides else lasso_model


def _validate_recluster_frequency(recluster_freq: str, rebalancing_freq: str) -> None:
    """Check that the reclustering frequency generates fewer calendar anchors.

    The check compares counts over a fixed 2000-2029 calendar interval. It checks
    relative cadence, not whether every recluster date is a covariance date.

    Args:
        recluster_freq: Pandas frequency string for reclustering anchors.
        rebalancing_freq: Pandas frequency string for covariance output dates.

    Raises:
        ValueError: If either frequency is invalid or reclustering generates at
            least as many anchors as covariance rebalancing.
    """
    sample_start = pd.Timestamp('2000-01-01')
    sample_end = pd.Timestamp('2029-12-31')
    try:
        recluster_count = len(pd.date_range(sample_start, sample_end, freq=recluster_freq))
        rebalancing_count = len(pd.date_range(sample_start, sample_end, freq=rebalancing_freq))
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"invalid recluster_freq={recluster_freq!r} or "
            f"rebalancing_freq={rebalancing_freq!r}"
        ) from exc
    if recluster_count >= rebalancing_count:
        raise ValueError(
            f"recluster_freq={recluster_freq!r} must be coarser than "
            f"rebalancing_freq={rebalancing_freq!r}"
        )


# FactorLasso's get_linkage_array extracts a whole tree but cannot restrict its leaves.
# This reporting adapter retains the discovered merge heights; it never reclusters assets.
def _restrict_linkage(linkage: np.ndarray, labels: pd.Index,
                      assets: pd.Index) -> np.ndarray:
    """Remove reference leaves from a discovered clustering tree.

    The adapter retains discovered merge heights and the requested asset order;
    it does not recluster the retained assets.

    Args:
        linkage: SciPy-style linkage array for the complete discovered tree.
        labels: Leaf labels in the order used by linkage.
        assets: Ordered subset of leaf labels to retain.

    Returns:
        Four-column linkage array over retained assets; zero or one retained
        leaf gives no merge rows.

    Raises:
        ValueError: If any requested asset label is absent from labels.
    """
    positions = labels.get_indexer(assets)
    if np.any(positions < 0):
        raise ValueError('clustering linkage is missing fitted asset labels')
    nodes = {int(old): (new, 1) for new, old in enumerate(positions)}
    rows = []
    for step, (left, right, height, _) in enumerate(linkage):
        lhs, rhs = nodes.get(int(left)), nodes.get(int(right))
        parent = len(labels) + step
        if lhs is not None and rhs is not None:
            count = lhs[1] + rhs[1]
            nodes[parent] = (len(assets) + len(rows), count)
            rows.append((lhs[0], rhs[0], height, count))
        elif lhs is not None or rhs is not None:
            nodes[parent] = lhs if lhs is not None else rhs
    return np.asarray(rows, dtype=float).reshape(-1, 4)


def _factor_cluster_path(risk_factor_prices: pd.DataFrame, returns: pd.DataFrame,
                         schedule: List[pd.Timestamp], model: LassoModel):
    """Build factor-assisted cluster paths and retain asset-only reporting trees.

    Factor prices are forward-filled to the return index and converted to log
    returns. FactorLasso discovers clusters on assets plus renamed factor
    references. References are then removed from assignments and linkages.
    Missing asset assignments receive distinct groups; warm-up-eligible asset
    leaves determine each reported tree.

    Args:
        risk_factor_prices: Date-by-factor prices for clustering references.
        returns: One cadence's date-by-asset log returns.
        schedule: Estimation dates passed to FactorLasso's cluster-path builder.
        model: Configuration with spans resolved for this return cadence.

    Returns:
        FactorLasso cluster-path container with asset assignments and restricted
        linkages. Its cutoff records remain those of the discovered reference tree.

    Raises:
        ValueError: If asset/factor columns are not unique, asset names collide
            with reserved factor-anchor labels, or retained linkage labels are missing.
    """
    if not returns.columns.is_unique or not risk_factor_prices.columns.is_unique:
        raise ValueError('factor clustering requires unique asset and factor columns')
    prices = risk_factor_prices.reindex(index=returns.index, method='ffill').ffill()
    factors = qis.to_returns(
        prices, is_log_returns=True, is_first_zero=False, drop_first=False, freq=None,
    )
    factors.columns = pd.Index([f'__factor_anchor__:{i}' for i in range(len(factors.columns))])
    if not returns.columns.intersection(factors.columns).empty:
        raise ValueError('asset names collide with reserved __factor_anchor__: labels')
    panel = pd.concat([returns, factors], axis=1, sort=True)
    path = compute_rolling_smoothed_clusters(y=panel, estimation_dates=schedule, lasso_model=model)
    for date in schedule:
        discovered = path.clusters[date]
        assignments = discovered.reindex(returns.columns).copy()
        # A common-mode transform may omit not-yet-eligible assets from discovery. Give each
        # missing response its own solver group; FactorLasso still zeroes/drops it at warmup.
        next_group = int(discovered.max()) + 1
        for asset in assignments.index[assignments.isna()]:
            assignments.loc[asset] = next_group
            next_group += 1
        active = returns.columns
        if model.warmup_period is not None:
            active = active[returns.loc[:date].notna().sum() >= model.warmup_period]
        path.linkages[date] = _restrict_linkage(path.linkages[date], discovered.index, active)
        path.clusters[date] = assignments.astype(int)
    return path


@dataclass(frozen=True)
class _FrequencyFitResult:
    """Unannualized components from one asset-return cadence.

    Attributes:
        betas: Asset-by-factor loadings from the fitted model.
        ewma_variances: Per-asset total variation reported by the fit.
        residual_variances: Per-asset residual variation reported by the fit.
        alphas: Per-asset fitted intercepts.
        r2: Dimensionless fit R-squared diagnostics.
        clusters: Optional per-asset fitted cluster assignments.
        linkage: Optional SciPy-style tree for this cadence.
        cutoff: Optional dendrogram cut distance.
        residuals: Supplied asset returns minus fitted factor returns times betas;
            the fitted intercept is not subtracted.
        derived_signs: Optional asset-by-factor sign requirements; NaN is unconstrained.
    """

    betas: pd.DataFrame
    ewma_variances: pd.Series
    residual_variances: pd.Series
    alphas: pd.Series
    r2: pd.Series
    clusters: Optional[pd.Series]
    linkage: Optional[np.ndarray]
    cutoff: Optional[float]
    residuals: pd.DataFrame
    derived_signs: Optional[pd.DataFrame]


def _fit_lasso_frequency(
        *,
        freq: str,
        asset_returns: pd.DataFrame,
        risk_factor_prices: pd.DataFrame,
        lasso_model: LassoModel,
        verbose: bool,
        precomputed_clusters: Optional[Dict[str, pd.Series]] = None,
        precomputed_linkages: Optional[Dict[str, np.ndarray]] = None,
        precomputed_cutoffs: Optional[Dict[str, float]] = None,
        reg_lambda: Optional[float] = None,
) -> _FrequencyFitResult:
    """Fit one return cadence and collect unannualized model components.

    Factor prices are aligned with historical forward-filling to asset dates,
    then converted to log returns. Cadence maps override regression/clustering
    spans for the fit. The original LassoModel receives the fitted state; a
    combined multi-cadence result must be read from the returned containers.

    Args:
        freq: Frequency code for this asset-return bucket.
        asset_returns: Date-by-asset log returns at the bucket's observation cadence.
        risk_factor_prices: Date-by-factor prices covering the observation history.
        lasso_model: Configured model to fit in place.
        verbose: Whether to print solver diagnostics.
        precomputed_clusters: Optional cadence-to-assignment map. A matching key
            supplies external memberships without replacing the configured model type.
        precomputed_linkages: Corresponding linkage map, required when memberships
            are supplied for this cadence.
        precomputed_cutoffs: Corresponding cut-distance map, required when memberships
            are supplied for this cadence.
        reg_lambda: Optional fixed penalty for this cadence. The model's scalar
            configuration is restored after fitting, including on solver failure.

    Returns:
        Unannualized fit components. Residual time series exclude only the fitted
        factor contribution, not the regression intercept.

    Raises:
        ValueError: If supplied cluster assignments are missing for any fitted asset.
        KeyError: If required cadence spans, linkages or cutoffs are absent.
    """
    factor_prices = risk_factor_prices.reindex(index=asset_returns.index, method='ffill').ffill()
    factor_returns = qis.to_returns(
        prices=factor_prices,
        is_log_returns=True,
        is_first_zero=False,
        drop_first=False,
        freq=None,
    )

    frequency_model = _model_for_frequency(lasso_model=lasso_model, freq=freq)
    span = frequency_model.span
    cluster_correlation_span = getattr(
        frequency_model, 'cluster_correlation_span', None
    )

    use_precomputed = precomputed_clusters is not None and freq in precomputed_clusters
    external_clusters = None
    if use_precomputed:
        external_clusters = precomputed_clusters[freq].reindex(asset_returns.columns)
        if external_clusters.isna().any():
            missing = external_clusters[external_clusters.isna()].index.tolist()
            raise ValueError(
                f"precomputed_clusters[{freq!r}] is missing assignments "
                f"for {len(missing)} assets: {missing}"
            )

    # Preserve the public function's fitted-state contract: callers receive their original
    # LassoModel back with the final cadence's fit attached. A fresh model copy would be a separate
    # numerical and behavioural change, even though clustering schedules use that pattern safely.
    fit_model = lasso_model
    cluster_span_kwargs = {}
    if hasattr(fit_model, 'cluster_correlation_span'):
        cluster_span_kwargs['cluster_correlation_span'] = cluster_correlation_span
    original_lambda = fit_model.reg_lambda
    if reg_lambda is not None:
        fit_model.reg_lambda = reg_lambda
    try:
        fit_model.fit(
            x=factor_returns,
            y=asset_returns,
            verbose=verbose,
            span=span,
            external_clusters=external_clusters,
            external_linkage=precomputed_linkages[freq] if use_precomputed else None,
            external_cutoff=precomputed_cutoffs[freq] if use_precomputed else None,
            **cluster_span_kwargs,
        )
    finally:
        fit_model.reg_lambda = original_lambda

    estimation_result = fit_model.estimation_result_
    linkage = precomputed_linkages[freq] if use_precomputed else fit_model.linkage
    cutoff = precomputed_cutoffs[freq] if use_precomputed else fit_model.cutoff
    return _FrequencyFitResult(
        betas=fit_model.estimated_betas,
        ewma_variances=pd.Series(estimation_result.ss_total, index=asset_returns.columns),
        residual_variances=pd.Series(estimation_result.ss_res, index=asset_returns.columns),
        alphas=pd.Series(estimation_result.alpha, index=asset_returns.columns),
        r2=pd.Series(estimation_result.r2, index=asset_returns.columns),
        clusters=fit_model.clusters,
        linkage=linkage,
        cutoff=cutoff,
        residuals=asset_returns - factor_returns @ fit_model.estimated_betas.T,
        derived_signs=fit_model.derived_signs_,
    )


@dataclass
class FactorCovarEstimator(CovarEstimator):
    """Configure sparse factor fitting and annual covariance assembly.

    QIS estimates factor covariance; FactorLasso estimates loadings and residual
    variation. Plain covariance methods assemble the factor component plus a
    weighted residual diagonal. Factor-specific methods retain diagnostics,
    clusters and residual time series in FactorLasso containers.

    An ordinary current fit uses supplied histories even when estimation_date
    is earlier. Rolling methods slice inputs through each output date. Optional
    factor references also impose a current-fit cutoff; smoothing alone does
    not make the final current regression truncate its inputs.

    Attributes:
        rebalancing_freq: Inherited calendar frequency for rolling estimation,
            default 'QE'; separate from return sampling and regression spans.
        lasso_model: FactorLasso fit configuration. Required for fitting; None permits
            construction only when factor references are disabled. Fits update this
            model's state, leaving the final cadence's fit attached.
        factor_returns_freq: Factor-covariance return cadence, default 'W-WED'.
            Regression factor returns instead follow each asset-return bucket.
        factor_covar_span: EWMA span in factor-covariance return observations,
            default 52; not a half-life or hard lookback.
        is_apply_vol_normalised_returns: Select the normalized-return QIS kernel for
            internally estimated factor covariance. No identity shrinkage is applied.
        demean: Stored configuration field currently not forwarded to the internal
            factor-covariance helper, which always uses demean=True. Regression
            demeaning is controlled separately by lasso_model.demean.
        include_factors_in_clustering: Add factor returns as clustering references
            for HCGL/FCGL. References are excluded from response fitting, pooled signs,
            group sizes, residuals and asset covariance. Explicit partitions take
            precedence; reported trees retain the induced asset merge heights.
        factor_clustering_freqs: Optional nonempty sequence of asset-return cadences
            receiving references when enabled. None includes every cadence.
        residual_type: 'orthogonal' (default diagonal) or 'empirical' (prepared
            common-period correlation scaled by current MATF residual standard deviations).
        residual_covar_freq: Common residual grid, default lowest native frequency.
            Only complete nested log-return periods can be summed; no extrapolation.
        residual_covar_span: EWMA span in common observations; None uses the lowest
            bucket's beta span, converting decay if an explicitly coarser grid is chosen.
        residual_corr_weight: Empirical correlation retention in [0, 1], default 1.
            Separate from residual_var_weight, which scales the entire residual risk block.

    Example:
        Illustrative calls: supply factors, returns_dict and time_period first.
        EwmaCovarEstimator provides runnable examples of the shared interface.

        >>> from factorlasso import LassoModel, LassoModelType  # doctest: +SKIP
        >>> estimator = FactorCovarEstimator(  # doctest: +SKIP
        ...     lasso_model=LassoModel(model_type=LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO,
        ...                            reg_lambda=1e-5, span=36),
        ...     factor_returns_freq='ME',
        ...     rebalancing_freq='QE',
        ... )
        >>> # Shared interface — plain covar dict
        >>> covar_dict = estimator.fit_rolling_covars(  # doctest: +SKIP
        ...     risk_factor_prices=factors, asset_returns_dict=returns_dict,
        ...     time_period=time_period)
        >>> # Factor-specific — full decomposition
        >>> rolling_data = estimator.fit_rolling_factor_covars(  # doctest: +SKIP
        ...     risk_factor_prices=factors, asset_returns_dict=returns_dict,
        ...     time_period=time_period)
        >>> r2_panel = rolling_data.get_r2()  # doctest: +SKIP
    """
    lasso_model: Optional[LassoModel] = None
    factor_returns_freq: str = 'W-WED'
    factor_covar_span: int = 52
    is_apply_vol_normalised_returns: bool = False
    demean: bool = True
    include_factors_in_clustering: bool = False
    factor_clustering_freqs: Optional[List[str]] = None
    residual_type: str = "orthogonal"
    residual_covar_freq: Optional[str] = None
    residual_covar_span: Optional[float] = None
    residual_corr_weight: float = 1.0
    # Fixed penalties by native response cadence; None preserves the scalar model setting.
    reg_lambda_freq_dict: Optional[Dict[str, float]] = None

    def __post_init__(self) -> None:
        """Validate factor-reference configuration at construction.

        This validates the opt-in reference fields, not complete fit readiness.

        Raises:
            TypeError: If include_factors_in_clustering is not a bool.
            ValueError: If factor_clustering_freqs is not a nonempty list/tuple of
                nonempty strings, or enabled references lack an HCGL/FCGL model.
        """
        if self.residual_type not in ('orthogonal', 'empirical'):
            raise ValueError("residual_type must be 'orthogonal' or 'empirical'")
        if not np.isfinite(self.residual_corr_weight) or not 0 <= self.residual_corr_weight <= 1:
            raise ValueError("residual_corr_weight must be finite and in [0, 1]")
        if self.residual_type == 'orthogonal' and self.residual_corr_weight != 1.:
            raise ValueError("residual_corr_weight applies only to empirical residuals")
        if not isinstance(self.include_factors_in_clustering, bool):
            raise TypeError('include_factors_in_clustering must be a bool')
        if self.factor_clustering_freqs is not None and (
                not isinstance(self.factor_clustering_freqs, (list, tuple))
                or not self.factor_clustering_freqs
                or not all(
                    isinstance(freq, str) and freq for freq in self.factor_clustering_freqs
                )):
            raise ValueError(
                'factor_clustering_freqs must be a non-empty sequence of frequency names'
            )
        if self.include_factors_in_clustering and (
                self.lasso_model is None or self.lasso_model.model_type not in (
                    LassoModelType.HIERARCHICAL_CLUSTER_GROUP_LASSO,
                    LassoModelType.FACTOR_CLUSTER_GROUP_LASSO,
                )):
            raise ValueError('factor clustering references require HCGL or FCGL')

    def _use_factor_references(self, freq: str) -> bool:
        """Check whether factor references are enabled for one response cadence.

        Args:
            freq: Asset-return cadence code.

        Returns:
            True when references are enabled and the cadence is selected, or when
            the enabled configuration has no cadence restriction.
        """
        return self.include_factors_in_clustering and (
            self.factor_clustering_freqs is None or freq in self.factor_clustering_freqs
        )

    def copy(self, **overrides) -> FactorCovarEstimator:
        """Create a replacement estimator with supplied field overrides.

        The copy is shallow: unchanged nested objects, including LassoModel, remain
        shared with the original estimator.

        Args:
            **overrides: Dataclass field names and replacement values.

        Returns:
            New FactorCovarEstimator with constructor validation applied.
        """
        self_dict = {f.name: getattr(self, f.name) for f in fields(self)}
        self_dict.update(overrides)
        return FactorCovarEstimator(**self_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Return configuration fields with a fresh, unfitted nested model.

        The dataclass conversion copies field values, then replaces a configured
        LassoModel with one rebuilt from its public constructor parameters.
        Fitted-state fields do not cross that model-configuration boundary.

        Returns:
            Dictionary containing estimator fields. Its lasso_model value is a
            LassoModel instance or None, not a JSON configuration mapping.
        """
        this = asdict(self)
        if self.lasso_model is not None:
            this['lasso_model'] = LassoModel(**self.lasso_model.get_params())
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
        """Fit a factor model and return its annual asset covariance matrix.

        Delegates to fit_current_factor_covars(), then asks FactorLasso to assemble
        the factor component plus residual_var_weight times the selected annual
        residual covariance. Both choices assume zero factor-residual covariance.

        Args:
            risk_factor_prices: Ordered date-by-factor total-return prices.
            asset_returns_dict: Frequency-code-to-log-return-panel mapping. Each asset
                belongs to exactly one cadence bucket.
            assets: Optional ordered output universe. Missing fitted asset rows are
                zero-filled by the factor-data helper.
            x_covar: Optional already annualized factor covariance with consistent
                factor labels and axis order. It is used without further scaling.
            estimation_date: Result label and optional clustering endpoint. In the
                ordinary current path it does not truncate inputs; slice histories
                explicitly. Enabled factor references or empirical residuals truncate histories.
            residual_var_weight: Multiplier on the selected annual residual covariance, default
                1.0. Values numerically close to zero omit that term.

        Returns:
            Square annual asset covariance DataFrame in fractional log-return-squared
            units when the supplied returns and factor covariance obey that convention.
        """
        factor_data = self.fit_current_factor_covars(
            risk_factor_prices=risk_factor_prices,
            asset_returns_dict=asset_returns_dict,
            assets=assets,
            x_covar=x_covar,
            estimation_date=estimation_date,
        )
        options = ({'residual_type': self.residual_type,
                    'residual_corr_weight': self.residual_corr_weight}
                   if self.residual_type == 'empirical' else {})
        return factor_data.get_y_covar(residual_var_weight=residual_var_weight,
                                       assets=assets, **options)

    def fit_rolling_covars(self,
                           risk_factor_prices: pd.DataFrame,
                           asset_returns_dict: Dict[str, pd.DataFrame],
                           time_period: qis.TimePeriod,
                           assets: Union[List[str], pd.Index] = None,
                           rebalancing_freq: Optional[str] = None,
                           residual_var_weight: float = 1.0,
                           ) -> Dict[pd.Timestamp, pd.DataFrame]:
        """Return annual asset covariance matrices from scheduled factor fits.

        Delegates to fit_rolling_factor_covars(), which slices every input through
        each scheduled date, then extracts plain matrices through FactorLasso.

        Args:
            risk_factor_prices: Ordered date-by-factor total-return prices.
            asset_returns_dict: Frequency-code-to-log-return-panel mapping; buckets
                partition the asset universe.
            time_period: Period for the calendar estimation schedule, not a lower
                bound on the histories used by each fit.
            assets: Optional ordered output universe for every matrix.
            rebalancing_freq: Calendar output-frequency override; None uses the
                estimator's inherited rebalancing_freq.
            residual_var_weight: Multiplier on the selected annual residual covariance in each
                assembled matrix. Values numerically close to zero omit that term.

        Returns:
            Dictionary from calendar estimation dates to annual asset covariance
            DataFrames. Keys need not match the direct EWMA estimator's return grid.
        """
        rolling_data = self.fit_rolling_factor_covars(
            risk_factor_prices=risk_factor_prices,
            asset_returns_dict=asset_returns_dict,
            time_period=time_period,
            assets=assets,
            rebalancing_freq=rebalancing_freq,
        )
        options = ({'residual_type': self.residual_type,
                    'residual_corr_weight': self.residual_corr_weight}
                   if self.residual_type == 'empirical' else {})
        return rolling_data.get_y_covars(residual_var_weight=residual_var_weight,
                                         assets=assets, **options)

    # ── Factor-model-specific API ────────────────────────────────────────

    def fit_current_factor_covars(
            self,
            risk_factor_prices: pd.DataFrame,
            asset_returns_dict: Dict[str, pd.DataFrame],
            assets: Union[List[str], pd.Index] = None,
            x_covar: Optional[pd.DataFrame] = None,
            estimation_date: Optional[pd.Timestamp] = None,
            precomputed_clusters: Optional[Dict[str, pd.Series]] = None,
            precomputed_linkages: Optional[Dict[str, np.ndarray]] = None,
            precomputed_cutoffs: Optional[Dict[str, float]] = None,
    ) -> CurrentFactorCovarData:
        """Fit a current factor decomposition and retain its diagnostics.

        With factor references disabled and orthogonal residuals, estimation_date
        labels the result and
        sets any rebuilt smoother's final date; it does not truncate the final
        regression inputs or the internally estimated factor covariance. Slice
        factor prices and every return bucket before historical current fits.

        With factor references or empirical residuals enabled, truncate histories through
        estimation_date, defaulting to the latest last-return date across buckets.
        A supplied x_covar is still used unchanged and must respect that cutoff.

        Without explicit partitions, active smoothing rebuilds the clustering path
        from the configured warm-up position through the endpoint. Factor references
        are added only for selected cadences. The original LassoModel retains the
        last cadence's fitted state.

        Args:
            risk_factor_prices: Ordered date-by-factor prices for covariance and fits.
            asset_returns_dict: Frequency-code-to-log-return-panel mapping, with each
                asset in exactly one cadence bucket.
            assets: Optional ordered output universe. The helper zero-fills missing
                fitted statistics instead of requiring a strict subset of fitted assets.
            x_covar: Optional annual factor covariance, already aligned in factor units
                and axis order. None estimates it from the supplied factor-price history.
            estimation_date: Result date and clustering endpoint, with the cutoff
                qualification above. Without a rebuilt path, None defaults in the
                helper to the first bucket's last date.
            precomputed_clusters: Optional cadence-to-membership map. Explicit
                memberships override automatic partition construction for supplied
                cadences and retain the configured model's penalty semantics.
            precomputed_linkages: Matching cadence-to-SciPy-linkage map for supplied
                memberships; used to preserve reference dendrograms.
            precomputed_cutoffs: Matching cadence-to-cut-distance map. Supply all three
                precomputed maps together; omitted cadence keys fit their own clusters.

        Returns:
            FactorLasso CurrentFactorCovarData with annual covariance/variance
            components, asset-by-factor betas, diagnostics and clustering metadata.
            Its residual panel is annual-scaled by cadence and excludes factor
            contributions without subtracting the fitted intercept.
        """
        if self.include_factors_in_clustering or self.residual_type == 'empirical':
            estimation_date = estimation_date or max(
                returns.index[-1] for returns in asset_returns_dict.values()
            )
            risk_factor_prices = risk_factor_prices.loc[:estimation_date]
            asset_returns_dict = {
                freq: returns.loc[:estimation_date] for freq, returns in asset_returns_dict.items()
            }
        smoother_type = ClusterSmootherType(self.lasso_model.cluster_smoother_type)
        if (smoother_type != ClusterSmootherType.NONE or self.include_factors_in_clustering
                ) and precomputed_clusters is None:
            estimation_date = estimation_date or max(
                returns.index[-1] for returns in asset_returns_dict.values()
            )
            precomputed_clusters = {}
            precomputed_linkages = {}
            precomputed_cutoffs = {}
            for freq, returns in asset_returns_dict.items():
                if (smoother_type == ClusterSmootherType.NONE
                        and not self._use_factor_references(freq)):
                    continue
                fit_model = _model_for_frequency(self.lasso_model, freq)
                start_position = min((fit_model.warmup_period or 1) - 1, len(returns.index) - 1)
                start_date = returns.index[start_position]
                schedule = qis.generate_dates_schedule(
                    time_period=qis.TimePeriod(start_date, estimation_date),
                    freq=self.rebalancing_freq,
                    include_start_date=False,
                    include_end_date=True,
                )
                final_date = pd.Timestamp(estimation_date)
                if smoother_type == ClusterSmootherType.NONE:
                    schedule = [final_date]
                if final_date not in schedule:
                    # Not covered, and currently unreachable: the schedule above is generated with
                    # include_end_date=True and estimation_date as the end, so qis always returns
                    # it whether or not the date falls on the rebalancing grid. Kept as a guard in
                    # case that flag or its semantics change -- the lookups below index the
                    # smoother output by final_date and would raise if it were ever absent.
                    schedule = sorted([*schedule, final_date])  # pragma: no cover
                rolling_clusters = (
                    _factor_cluster_path(risk_factor_prices, returns, schedule, fit_model)
                    if self._use_factor_references(freq) else compute_rolling_smoothed_clusters(
                        y=returns, estimation_dates=schedule, lasso_model=fit_model,
                    )
                )
                precomputed_clusters[freq] = rolling_clusters.clusters[final_date]
                precomputed_linkages[freq] = rolling_clusters.linkages[final_date]
                precomputed_cutoffs[freq] = rolling_clusters.cutoffs[final_date]

        residual_options = ({
            'residual_type': self.residual_type,
            'residual_covar_freq': self.residual_covar_freq,
            'residual_covar_span': self.residual_covar_span,
        } if self.residual_type == 'empirical' else {})
        if self.reg_lambda_freq_dict is not None:
            residual_options['reg_lambda_freq_dict'] = self.reg_lambda_freq_dict
        factor_covar_data = estimate_lasso_factor_covar_data(
            risk_factor_prices=risk_factor_prices,
            asset_returns_dict=asset_returns_dict,
            assets=assets,
            lasso_model=self.lasso_model,
            x_covar=x_covar,
            factor_returns_freq=self.factor_returns_freq,
            factor_covar_span=self.factor_covar_span,
            is_apply_vol_normalised_returns=self.is_apply_vol_normalised_returns,
            estimation_date=estimation_date,
            precomputed_clusters=precomputed_clusters,
            precomputed_linkages=precomputed_linkages,
            precomputed_cutoffs=precomputed_cutoffs,
            **residual_options,
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
        """Fit factor decompositions using expanding histories at calendar dates.

        The QIS schedule uses the effective rebalancing frequency without adding
        off-grid period endpoints. Before each fit, factor prices and every
        asset-return bucket are sliced through that date; history before the period
        start remains available.

        An active smoother or enabled factor references first builds cluster paths
        over the same schedule through FactorLasso. Those per-date partitions are
        injected into the configured model fits. Supplied observation dates must
        reflect when data was available; the schedule cannot establish that.

        Args:
            risk_factor_prices: Ordered date-by-factor total-return prices.
            asset_returns_dict: Frequency-code-to-log-return-panel mapping, with
                nonoverlapping asset buckets.
            time_period: Calendar period for output dates, separate from fit histories.
            assets: Optional ordered output universe at every date.
            rebalancing_freq: Output-frequency override; None uses self.rebalancing_freq.

        Returns:
            FactorLasso RollingFactorCovarData mapping each scheduled date to its
            decomposition. The estimator's model retains the final cadence/date fit.

        Raises:
            ValueError: If a bucket has fewer rows than the configured warmup_period
                at a fit date, or a configured reclustering cadence is not coarser
                than the effective output cadence. The row-count check is not a
                per-asset completeness or eligibility test.
        """
        effective_rebalancing_freq = rebalancing_freq or self.rebalancing_freq
        rebalancing_schedule = qis.generate_dates_schedule(
            time_period=time_period,
            freq=effective_rebalancing_freq,
            include_start_date=False,
            include_end_date=False
        )
        smoother_type = ClusterSmootherType(self.lasso_model.cluster_smoother_type)
        rolling_clusters_by_freq = None
        if smoother_type != ClusterSmootherType.NONE or self.include_factors_in_clustering:
            if self.lasso_model.recluster_freq is not None:
                _validate_recluster_frequency(
                    recluster_freq=str(self.lasso_model.recluster_freq),
                    rebalancing_freq=effective_rebalancing_freq,
                )
            rolling_clusters_by_freq = {
                freq: (
                    _factor_cluster_path(
                        risk_factor_prices, returns, rebalancing_schedule,
                        _model_for_frequency(self.lasso_model, freq),
                    ) if self._use_factor_references(freq) else compute_rolling_smoothed_clusters(
                        y=returns, estimation_dates=rebalancing_schedule,
                        lasso_model=_model_for_frequency(self.lasso_model, freq),
                    )
                )
                for freq, returns in asset_returns_dict.items()
                if smoother_type != ClusterSmootherType.NONE or self._use_factor_references(freq)
            }

        covar_datas: Dict[pd.Timestamp, CurrentFactorCovarData] = {}
        previous_residual = None
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

            cluster_kwargs = {}
            if rolling_clusters_by_freq is not None:
                cluster_kwargs = {
                    'precomputed_clusters': {
                        freq: data.clusters[estimation_date]
                        for freq, data in rolling_clusters_by_freq.items()
                    },
                    'precomputed_linkages': {
                        freq: data.linkages[estimation_date]
                        for freq, data in rolling_clusters_by_freq.items()
                    },
                    'precomputed_cutoffs': {
                        freq: data.cutoffs[estimation_date]
                        for freq, data in rolling_clusters_by_freq.items()
                    },
                }

            covar_datas[estimation_date] = self.fit_current_factor_covars(
                risk_factor_prices=risk_factor_prices.loc[:estimation_date],
                asset_returns_dict=asset_returns_dict_upto_date,
                assets=assets,
                estimation_date=estimation_date,
                **cluster_kwargs,
            )
            if self.residual_type == 'empirical':
                current = covar_datas[estimation_date]
                prepared = current.residual_correlation
                if (previous_residual is not None
                        and prepared.observation_date == previous_residual.observation_date
                        and prepared.frequency == previous_residual.frequency
                        and prepared.span == previous_residual.span
                        and prepared.correlation.index.equals(previous_residual.correlation.index)
                        and prepared.asset_metadata.equals(previous_residual.asset_metadata)):
                    prepared = previous_residual
                    covar_datas[estimation_date] = replace(current, residual_correlation=prepared)
                previous_residual = prepared

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
                                     verbose: bool = False,
                                     precomputed_clusters: Optional[Dict[str, pd.Series]] = None,
                                     precomputed_linkages: Optional[Dict[str, np.ndarray]] = None,
                                     precomputed_cutoffs: Optional[Dict[str, float]] = None,
                                     *,
                                     residual_type: str = 'orthogonal',
                                     residual_covar_freq: Optional[str] = None,
                                     residual_covar_span: Optional[float] = None,
                                     reg_lambda_freq_dict: Optional[Dict[str, float]] = None,
                                     ) -> CurrentFactorCovarData:
    """Assemble current factor covariance data from supplied return histories.

    For each asset-return cadence, align factor prices by historical
    forward-filling, form log returns, and fit the supplied FactorLasso model.
    The original model is updated in place and retains the final bucket's fit.
    In the default orthogonal mode, estimation_date is metadata only; truncate
    inputs explicitly for a historical fit. Empirical mode truncates at that date
    before fitting and preparing residual covariance.

    Internal factor covariance always uses demean=True and is annualized from
    factor_returns_freq. Regression mean adjustment belongs to LassoModel.
    Asset variances, intercepts and stored residual series are multiplied by
    their bucket's annualization factor; betas and R-squared are not.
    Stored residuals are asset returns minus the fitted factor contribution,
    without intercept subtraction. Annual scaling does not aggregate those
    observations into realized annual returns.

    Cluster IDs and linkage merge-step labels are frequency-prefixed; cutoffs
    are indexed by cadence. FactorLasso owns the resulting data container and
    the methods that assemble asset covariance from its components.

    Args:
        risk_factor_prices: Ordered date-by-factor total-return prices.
        asset_returns_dict: Nonempty mapping from return-frequency codes to
            date-by-asset log-return panels. Codes describe observation cadence,
            not rebalancing. Each asset belongs to exactly one bucket.
        lasso_model: Configured FactorLasso model, fitted in place for each bucket.
            Cadence maps control regression and clustering spans where configured.
        reg_lambda_freq_dict: Optional fixed penalty per response cadence. Every
            fitted cadence must be present. None retains lasso_model.reg_lambda.
            Values are not rescaled as estimation histories grow.
        assets: Optional ordered asset universe for output alignment. Missing
            fitted rows receive zero betas and zero variance/alpha/R-squared
            statistics; cluster/sign entries remain NaN.
        x_covar: Optional square annual factor covariance used without rescaling.
            Both axes must use consistent factor labels and order.
        factor_returns_freq: Return cadence for internally estimated factor
            covariance, default 'W-WED'; regression factors follow each bucket.
        factor_covar_span: EWMA span in factor-covariance return observations,
            default 52; not a half-life or hard lookback.
        is_apply_vol_normalised_returns: Use the normalized-return QIS kernel for
            internal factor covariance. Ignored when x_covar is supplied.
        estimation_date: In orthogonal mode, a metadata label defaulting to the first
            bucket's last date. Empirical mode uses this as an input cutoff, defaulting
            to the latest last-return date across buckets.
        verbose: Whether to print solver diagnostics.
        precomputed_clusters: Optional cadence-to-membership map. Matching keys
            supply external partitions while retaining the configured model and
            penalty semantics; missing keys use the model's own clustering.
        precomputed_linkages: Matching linkage arrays for supplied memberships.
            Must be provided together with both other precomputed maps.
        precomputed_cutoffs: Matching dendrogram cut distances for supplied
            memberships; all three maps must be supplied together or all be None.
        residual_type: 'orthogonal' (default) or 'empirical'. Empirical attaches a
            prepared dimensionless correlation, with no factor-residual cross term.
        residual_covar_freq: Common residual grid; None selects the lowest native
            frequency. Raw log residuals are summed only over complete nested periods.
        residual_covar_span: EWMA span in common periods; None derives it from the
            lowest-frequency bucket's beta configuration. Retrieval needs no span/scale.

    Returns:
        FactorLasso CurrentFactorCovarData. Betas use asset rows and factor
        columns; variance/alpha statistics are annualized and R-squared is filled
        then clipped below at zero. Residuals keep structural NaNs across cadence
        grids; with an explicit assets universe, entirely missing residual columns
        become zero. Optional derived-sign NaNs retain their unconstrained meaning.

    Raises:
        ValueError: If only some precomputed maps are supplied, or a supplied
            membership panel lacks an assignment for a fitted asset.
        KeyError: If a required cadence span, linkage or cutoff entry is missing.
    """
    if reg_lambda_freq_dict is not None:
        if any(not np.isfinite(value) or value < 0
               for value in reg_lambda_freq_dict.values()):
            raise ValueError('cadence penalties must be finite and nonnegative')
        missing = set(asset_returns_dict) - set(reg_lambda_freq_dict)
        if missing:
            raise KeyError(f'no reg_lambda for cadence(s): {sorted(missing)}')
    if residual_type not in ('orthogonal', 'empirical'):
        raise ValueError("residual_type must be 'orthogonal' or 'empirical'")
    if residual_type == 'empirical':
        if not _CFCD_SUPPORTS_RESIDUAL_CORRELATION:
            raise ImportError(
                "Upgrade factorlasso to a version supporting prepared residual correlation"
            )
        estimation_date = estimation_date or max(
            data.index[-1] for data in asset_returns_dict.values()
        )
        risk_factor_prices = risk_factor_prices.loc[:estimation_date]
        asset_returns_dict = {
            freq: data.loc[:estimation_date] for freq, data in asset_returns_dict.items()
        }
    # Validate precomputed cluster inputs: either all three or none.
    # Partial supply would produce inconsistent CurrentFactorCovarData
    # (e.g. clusters without linkage/cutoff for dendrogram rendering).
    _pc_provided = [p is not None for p in
                    (precomputed_clusters, precomputed_linkages, precomputed_cutoffs)]
    if any(_pc_provided) and not all(_pc_provided):
        raise ValueError(
            "precomputed_clusters, precomputed_linkages, and precomputed_cutoffs "
            "must all be provided together, or all be None. Received: "
            f"clusters={'yes' if _pc_provided[0] else 'no'}, "
            f"linkages={'yes' if _pc_provided[1] else 'no'}, "
            f"cutoffs={'yes' if _pc_provided[2] else 'no'}."
        )

    # 1. compute x-factors ewm covar at rebalancing freq
    if x_covar is None:
        x_covar = estimate_current_ewma_covar(prices=risk_factor_prices,
                                              returns_freq=factor_returns_freq,
                                              demean=True,
                                              span=factor_covar_span,
                                              is_apply_vol_normalised_returns=is_apply_vol_normalised_returns,
                                              apply_an_factor=False)
        factor_scale_an = qis.get_annualisation_conversion_factor(
            from_freq=factor_returns_freq, to_freq='YE'
        )
        x_covar *= factor_scale_an

    # 2. estimate betas and diagnostics per frequency
    frequency_results = {
        freq: _fit_lasso_frequency(
            freq=freq,
            asset_returns=asset_returns,
            risk_factor_prices=risk_factor_prices,
            lasso_model=lasso_model,
            verbose=verbose,
            precomputed_clusters=precomputed_clusters,
            precomputed_linkages=precomputed_linkages,
            precomputed_cutoffs=precomputed_cutoffs,
            **({'reg_lambda': reg_lambda_freq_dict[freq]}
               if reg_lambda_freq_dict is not None else {}),
        )
        for freq, asset_returns in asset_returns_dict.items()
    }

    # 3. annualise and merge across frequencies
    asset_last_betas = []
    last_ewma_vars = []
    last_residual_vars = []
    last_alphas = []
    last_r2 = []
    residuals = []
    residual_metadata = []
    derived_signs_list: List[pd.DataFrame] = []
    for freq in asset_returns_dict.keys():
        result = frequency_results[freq]
        asset_last_betas.append(result.betas)  # (N_freq x M)
        idio_var_scaler = qis.get_annualisation_conversion_factor(from_freq=freq, to_freq='YE')
        last_ewma_vars.append(idio_var_scaler * result.ewma_variances)
        last_residual_vars.append(idio_var_scaler * result.residual_variances)
        last_alphas.append(idio_var_scaler * result.alphas)
        last_r2.append(result.r2)
        residuals.append(idio_var_scaler * result.residuals)
        if _CFCD_SUPPORTS_RESIDUAL_CORRELATION:
            residual_metadata.append(pd.DataFrame({
                'frequency': freq, 'beta_span': _model_for_frequency(lasso_model, freq).span,
                'annualisation_factor': idio_var_scaler, 'residual_scale': idio_var_scaler,
            }, index=result.betas.index))
        # derived_signs: only freqs that actually got a sign layer contribute.
        # Unlike betas (always emitted by every freq), signs may be absent
        # entirely if auto_sign_constraints=False and no explicit
        # factors_beta_loading_signs was passed.
        sub = result.derived_signs
        if sub is not None:
            derived_signs_list.append(sub)

    # align to target asset universe
    # betas: concat along axis=0 (rows=assets), reindex rows to target assets, columns to factors
    asset_last_betas = pd.concat(asset_last_betas, axis=0).reindex(
        columns=x_covar.index
    ).fillna(0.0)
    last_ewma_vars = pd.concat(last_ewma_vars, axis=0).fillna(0.0)
    last_residual_vars = pd.concat(last_residual_vars, axis=0).fillna(0.0)
    last_alphas = pd.concat(last_alphas, axis=0).fillna(0.0)
    last_r2 = pd.concat(last_r2, axis=0).fillna(0.0).clip(0.0, None)
    # derived_signs: same concat+reindex pattern as betas but DO NOT
    # fillna — NaN here means "unconstrained" (no sign requirement on
    # this asset/factor cell), which is semantically distinct from 0
    # ("forced zero"). Filling with 0 would silently introduce a hard
    # constraint that was never specified. Reindex columns to the full
    # factor universe so factor-naming stays consistent with y_betas.
    if derived_signs_list:
        derived_signs = pd.concat(derived_signs_list, axis=0).reindex(columns=x_covar.index)
    else:
        derived_signs = None

    # Flatten per-freq clustering outputs into persistable pandas objects.
    # Each asset appears in exactly one frequency bucket (freqs partition
    # the universe), so concat along axis=0 is safe for clusters and for
    # linkages — no overlap, no conflict. Per-freq identifiers are prefixed
    # with the freq code so the merged objects can be split back per freq
    # downstream (e.g. factor_covar.get_linkage_array).
    cluster_series_list = []
    linkage_frames = []
    cutoff_values: Dict[str, float] = {}
    for freq in asset_returns_dict.keys():
        result = frequency_results[freq]
        s = result.clusters
        if s is not None:
            cluster_series_list.append(s.astype(str).radd(f"{freq}:"))

        L = result.linkage
        if L is not None:
            linkage_frames.append(pd.DataFrame(
                L,
                columns=['left', 'right', 'distance', 'n_samples'],
                index=pd.Index(
                    [f"{freq}:step_{i}" for i in range(L.shape[0])],
                    name='merge_step',
                ),
            ))

        c = result.cutoff
        if c is not None:
            cutoff_values[freq] = float(c)

    clusters_flat: Optional[pd.Series]
    if cluster_series_list:
        clusters_flat = pd.concat(cluster_series_list, axis=0)
        clusters_flat.name = VarianceColumns.CLUSTER.value
    else:
        clusters_flat = None

    linkages_flat: Optional[pd.DataFrame] = (
        pd.concat(linkage_frames, axis=0) if linkage_frames else None
    )

    cutoffs_flat: Optional[pd.Series] = (
        pd.Series(cutoff_values, name='cluster_cutoff') if cutoff_values else None
    )

    # Preserve structural NaN from frequency mismatch.
    # When ME and QE residual frames are concat'd on axis=1, QE columns
    # naturally get NaN on non-quarter-end months. That NaN is meaningful
    # ("no observation"), not zero ("zero return"). Downstream pandas
    # EWMA in CurrentFactorCovarData.estimate_alpha handles NaN correctly
    # by carrying the previous value forward.
    residuals = pd.concat(residuals, axis=1, sort=True)
    if assets is not None:
        asset_last_betas = asset_last_betas.reindex(index=assets).fillna(0.0)
        last_ewma_vars = last_ewma_vars.reindex(index=assets).fillna(0.0)
        last_residual_vars = last_residual_vars.reindex(index=assets).fillna(0.0)
        last_alphas = last_alphas.reindex(index=assets).fillna(0.0)
        last_r2 = last_r2.reindex(index=assets).fillna(0.0)
        if clusters_flat is not None:
            # Reindex cluster assignment to the target asset universe.
            # Missing assets (no fit) get NaN — distinct from any valid cluster ID.
            clusters_flat = clusters_flat.reindex(index=assets)
        if derived_signs is not None:
            # NO fillna — leave NaN for assets that didn't get a fit;
            # downstream consumers distinguish "unconstrained" from
            # "no fit" by the absence vs presence of an asset row anyway,
            # but keeping NaN here preserves the original semantics.
            derived_signs = derived_signs.reindex(index=assets)
        # Reindex preserves existing NaN from frequency mismatch.
        # Only fill columns that are *entirely* missing (assets with no
        # fitted history at all) with zeros so downstream consumers
        # don't see surprise all-NaN columns.
        residuals = residuals.reindex(columns=assets)
        _missing_cols = residuals.columns[residuals.isna().all(axis=0)]
        if len(_missing_cols) > 0:
            residuals.loc[:, _missing_cols] = 0.0

    y_variances = pd.concat([last_ewma_vars.rename(VarianceColumns.EWMA_VARIANCE.value),
                             last_residual_vars.rename(VarianceColumns.RESIDUAL_VARS.value),
                             last_alphas.rename(VarianceColumns.INSAMPLE_ALPHA.value),
                             last_r2.rename(VarianceColumns.R2.value)],
                            axis=1, sort=False)

    estimation_date = estimation_date or asset_returns_dict[
        list(asset_returns_dict.keys())[0]
    ].index[-1]
    cfcd_kwargs: Dict[str, Any] = dict(
        x_covar=x_covar,
        y_betas=asset_last_betas,
        y_variances=y_variances,
        clusters=clusters_flat,
        linkages=linkages_flat,
        cutoffs=cutoffs_flat,
        residuals=residuals,
        estimation_date=estimation_date,
    )
    # Only attach derived_signs when factorlasso supports the field — see
    # _CFCD_SUPPORTS_DERIVED_SIGNS at module top.
    if _CFCD_SUPPORTS_DERIVED_SIGNS and derived_signs is not None:
        cfcd_kwargs['derived_signs'] = derived_signs
    if _CFCD_SUPPORTS_RESIDUAL_CORRELATION:
        metadata = pd.concat(residual_metadata).reindex(asset_last_betas.index)
        cfcd_kwargs['residual_metadata'] = metadata
        if residual_type == 'empirical':
            from factorlasso import estimate_residual_correlation

            annualisation = (qis.get_annualisation_conversion_factor(residual_covar_freq, 'YE')
                             if residual_covar_freq is not None else None)
            # A reindexed output universe can contain assets with no fitted history. Their
            # zero marginal residual variance makes correlation undefined but contributes no
            # covariance; estimate the supported block and keep those assets independent.
            positive_risk = last_residual_vars.index[last_residual_vars.gt(0)]
            if positive_risk.empty:
                raise ValueError('Empirical residual correlation needs positive residual risk')
            has_zero_risk = len(positive_risk) != len(residuals.columns)
            fitted_correlation = estimate_residual_correlation(
                residuals=residuals[positive_risk] if has_zero_risk else residuals,
                metadata=metadata.loc[positive_risk] if has_zero_risk else metadata,
                estimation_date=estimation_date,
                frequency=residual_covar_freq, span=residual_covar_span,
                periods_per_year=annualisation,
            )
            if has_zero_risk:
                names = residuals.columns
                correlation = pd.DataFrame(np.eye(len(names)), index=names, columns=names)
                correlation.loc[positive_risk, positive_risk] = fitted_correlation.correlation
                fitted_correlation = replace(
                    fitted_correlation, correlation=correlation,
                    residual_returns=fitted_correlation.residual_returns.reindex(
                        columns=names, fill_value=0.0),
                    asset_metadata=metadata,
                )
            cfcd_kwargs['residual_correlation'] = fitted_correlation
    covar_data = CurrentFactorCovarData(**cfcd_kwargs)
    return covar_data
