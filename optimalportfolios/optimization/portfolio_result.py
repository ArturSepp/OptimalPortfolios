"""
Portfolio optimisation result container with risk attribution.

optimalportfolios/optimization/portfolio_result.py

Convention:
    y_betas is (N x M) with index=assets, columns=factors.
    Factor exposure: f = w @ beta  where w is (N,), beta is (N x M), f is (M,).
    Systematic covariance: Sigma_sys = beta @ Sigma_x @ beta'  (N x N).

Supports N portfolios: weights and benchmark_weights can be pd.Series (single)
or pd.DataFrame (multiple portfolios as columns). current_weights is optional
and follows the same convention.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Dict, Union, List, Tuple

from optimalportfolios.utils.portfolio_funcs import round_weights_to_pct
from optimalportfolios.covar_estimation.factor_covar_data import CurrentFactorCovarData, VarianceColumns
from optimalportfolios.optimization.constraints import Constraints


def _to_dataframe(w: Union[pd.Series, pd.DataFrame], default_name: str = 'portfolio') -> pd.DataFrame:
    """Normalize Series or DataFrame to DataFrame with named columns."""
    if isinstance(w, pd.Series):
        return w.to_frame(w.name or default_name)
    return w


@dataclass
class PortfolioOptimisationResult:
    """
    Portfolio optimisation output with factor model context for risk attribution.

    Supports N portfolios via DataFrame inputs. When weights is a DataFrame,
    each column is a separate portfolio. benchmark_weights is matched:
      - Series: same benchmark for all portfolios
      - DataFrame: per-portfolio benchmarks (columns must match weights columns)

    current_weights is optional (needed only for turnover analysis):
      - Series: same current weights for all portfolios
      - DataFrame: per-portfolio current weights (columns must match weights columns)

    Risk decomposition:
        sigma^2_p = w' Sigma_y w = w' beta Sigma_x beta' w + w' D w
                  = systematic risk  + idiosyncratic risk

    Active risk (tracking error):
        TE^2 = delta' Sigma_y delta  where delta = w - w_bench

    Turnover:
        turnover = sum |w_new - w_current| / 2
    """

    # Core portfolio weights: Series (single) or DataFrame (N portfolios as columns)
    weights: Union[pd.Series, pd.DataFrame]
    # Benchmark: Series (shared) or DataFrame (per-portfolio, columns match weights)
    benchmark_weights: Union[pd.Series, pd.DataFrame]
    # Factor model universe (has-a)
    covar_data: CurrentFactorCovarData
    # For group attributions
    group_attributions: Dict[str, pd.Series]

    # Optional: current weights for turnover analysis
    current_weights: Optional[Union[pd.Series, pd.DataFrame]] = None

    # Constraints object for audit
    constraints: Constraints = None
    # Metadata
    reference_ccy: str = 'USD'
    optimisation_date: Optional[pd.Timestamp] = field(default_factory=pd.Timestamp.now)
    metadata: pd.DataFrame = None
    expected_return: pd.Series = None
    ac_bounds: pd.DataFrame = None
    portfolio_id: str = None

    def __post_init__(self):
        """Normalize inputs to DataFrames and validate alignment."""
        # Normalize weights to DataFrame
        self._weights_df = _to_dataframe(self.weights, 'portfolio')

        # Normalize benchmark: broadcast Series to match all portfolio columns
        if isinstance(self.benchmark_weights, pd.Series):
            bench_name = self.benchmark_weights.name or 'benchmark'
            self._benchmark_df = pd.DataFrame(
                {name: self.benchmark_weights.rename(bench_name)
                 for name in self._weights_df.columns}
            )
            self._benchmark_df.columns = self._weights_df.columns
            self._shared_benchmark = True
            self._benchmark_name = bench_name
        else:
            self._benchmark_df = self.benchmark_weights.copy()
            self._shared_benchmark = False
            self._benchmark_name = None
            # Validate portfolio names match
            if set(self._weights_df.columns) != set(self._benchmark_df.columns):
                raise ValueError(
                    f"Portfolio names mismatch between weights {list(self._weights_df.columns)} "
                    f"and benchmark_weights {list(self._benchmark_df.columns)}"
                )
            self._benchmark_df = self._benchmark_df[self._weights_df.columns]

        # Normalize current_weights if provided
        if self.current_weights is not None:
            if isinstance(self.current_weights, pd.Series):
                self._current_df = pd.DataFrame(
                    {name: self.current_weights for name in self._weights_df.columns}
                )
                self._current_df.columns = self._weights_df.columns
            else:
                if set(self._weights_df.columns) != set(self.current_weights.columns):
                    raise ValueError(
                        f"Portfolio names mismatch between weights {list(self._weights_df.columns)} "
                        f"and current_weights {list(self.current_weights.columns)}"
                    )
                self._current_df = self.current_weights[self._weights_df.columns].copy()
        else:
            self._current_df = None

        self._validate_group_attributions()

    def _validate_group_attributions(self) -> None:
        """Check that group attributions cover all tickers."""
        msg = "weights and group index mismatch."
        for group, ds in self.group_attributions.items():
            missing_ds = set(self.tickers) - set(ds.index)
            if missing_ds:
                msg += f" Missing in {group}: {missing_ds}"
                raise ValueError(msg)

    # === Properties ===

    @property
    def portfolio_names(self) -> List[str]:
        """Names of all portfolios."""
        return list(self._weights_df.columns)

    @property
    def n_portfolios(self) -> int:
        """Number of portfolios."""
        return len(self._weights_df.columns)

    @property
    def tickers(self) -> pd.Index:
        """Asset universe."""
        return self._weights_df.index

    @property
    def n_assets(self) -> int:
        """Number of assets."""
        return len(self.tickers)

    @property
    def has_current_weights(self) -> bool:
        """Whether current weights are available for turnover analysis."""
        return self._current_df is not None

    def get_weights(self, name: str) -> pd.Series:
        """Get weights for a specific portfolio."""
        return self._weights_df[name]

    def get_benchmark(self, name: str) -> pd.Series:
        """Get benchmark weights for a specific portfolio."""
        return self._benchmark_df[name]

    def get_current(self, name: str) -> pd.Series:
        """Get current weights for a specific portfolio."""
        if self._current_df is None:
            raise ValueError("current_weights not provided.")
        return self._current_df[name]

    def get_active_weights(self, name: str) -> pd.Series:
        """Active weights: delta = w - w_bench for a specific portfolio."""
        return self.get_weights(name) - self.get_benchmark(name)

    def get_trade_weights(self, name: str) -> pd.Series:
        """Trade weights: dw = w - w_current for a specific portfolio."""
        return self.get_weights(name) - self.get_current(name)

    # Convenience for single-portfolio backward compat
    @property
    def active_weights_df(self) -> pd.DataFrame:
        """Active weights for all portfolios."""
        return self._weights_df - self._benchmark_df

    @property
    def current_active_weights_df(self) -> Optional[pd.DataFrame]:
        """Current active weights for all portfolios."""
        if self._current_df is None:
            return None
        return self._current_df - self._benchmark_df

    @property
    def trade_weights_df(self) -> Optional[pd.DataFrame]:
        """Trade weights for all portfolios."""
        if self._current_df is None:
            return None
        return self._weights_df - self._current_df

    def get_assets_metadata(self, return_name: Optional[str] = 'CMA') -> pd.DataFrame:
        asset_table = self.metadata
        if self.expected_return is not None and return_name is not None:
            asset_table = pd.concat([asset_table, self.expected_return.rename(return_name)], axis=1)
        return asset_table

    def get_asset_betas_table(self, return_name: Optional[str] = 'CMA') -> pd.DataFrame:
        asset_risk_table = self.covar_data.get_snapshot()
        if self.expected_return is not None and return_name is not None:
            asset_risk_table = pd.concat([self.expected_return.rename(return_name), asset_risk_table], axis=1)
        return asset_risk_table

    def get_combined_asset_weight_table(self, weights_to_pct: bool = False) -> pd.DataFrame:
        weights_df = self.compute_weight_summary(weights_to_pct=weights_to_pct)
        metadata = self.get_assets_metadata()
        weights = pd.concat([metadata, weights_df], axis=1)
        return weights

    # === Risk metrics (single weight vector) ===

    def compute_portfolio_variance(self, weights: pd.Series = None) -> float:
        """Portfolio variance: sigma^2_p = w' Sigma_y w."""
        w = self._weights_df.iloc[:, 0] if weights is None else weights
        w = w.loc[self.tickers].values
        cov = self.covar_data.y_covar.loc[self.tickers, self.tickers].values
        return float(w @ cov @ w)

    def compute_portfolio_vol(self, weights: pd.Series = None) -> float:
        """Portfolio volatility: sigma_p = sqrt(w' Sigma_y w)."""
        return np.sqrt(self.compute_portfolio_variance(weights))

    def compute_tracking_error(self, name: str = None) -> float:
        """Tracking error: TE = sqrt(delta' Sigma_y delta) for a portfolio."""
        name = name or self.portfolio_names[0]
        return np.sqrt(self.compute_portfolio_variance(self.get_active_weights(name)))

    # === Turnover analysis ===

    def compute_turnover(self, name: str = None, one_way: bool = True) -> float:
        """Portfolio turnover for a specific portfolio."""
        name = name or self.portfolio_names[0]
        trades = self.get_trade_weights(name)
        total_trades = float(np.abs(trades).sum())
        return total_trades / 2 if one_way else total_trades

    def compute_turnover_analysis(self, name: str = None) -> pd.Series:
        """Detailed turnover metrics for a specific portfolio."""
        name = name or self.portfolio_names[0]
        trades = self.get_trade_weights(name)
        buys = trades[trades > 0].sum()
        sells = trades[trades < 0].abs().sum()
        n_trades = (trades.abs() > 1e-6).sum()
        avg_trade = trades.abs().mean()
        max_trade = trades.abs().max()

        return pd.Series({
            'turnover': (buys + sells) / 2,
            'buys': buys,
            'sells': sells,
            'n_trades': n_trades,
            'avg_trade_size': avg_trade,
            'max_trade_size': max_trade
        }, name='turnover_analysis')

    # === Factor risk attribution ===

    def _compute_risk_decomposition(self, weights: pd.Series) -> Dict[str, float]:
        """
        Decompose variance for given weights into factor and residual components.

        sigma^2_p = w' beta Sigma_x beta' w + w' D w

        where beta is (N x M), w is (N,).
        Factor exposure: f = w @ beta  -> (M,)
        Factor variance: f' Sigma_x f  -> scalar
        """
        w = weights.loc[self.tickers].values  # (N,)
        betas = self.covar_data.y_betas.loc[self.tickers, :].values  # (N x M)

        factor_exp = w @ betas  # (N,) @ (N,M) = (M,)
        factor_var = float(factor_exp @ self.covar_data.x_covar.values @ factor_exp)

        residual_vars = self.covar_data.y_variances.loc[self.tickers, VarianceColumns.RESIDUAL_VARS.value].values
        residual_var = float((w ** 2) @ residual_vars)

        total_var = factor_var + residual_var
        total_vol = np.sqrt(total_var)

        out = {}
        if self.expected_return is not None:
            exp_ret = float(np.nansum(self.expected_return.loc[self.tickers].values * w))
            out['exp_return'] = exp_ret
        out['total_vol'] = total_vol
        out['factor_vol'] = np.sqrt(factor_var)
        out['residual_vol'] = np.sqrt(residual_var)
        out['factor_pct'] = factor_var / total_var if total_var > 0 else 0.0
        out['residual_pct'] = residual_var / total_var if total_var > 0 else 0.0
        if self.expected_return is not None and total_vol > 0:
            out['sharpe_ratio'] = out['exp_return'] / total_vol
        return out

    def _compute_tracking_error_metrics(self, active_weights: pd.Series) -> Dict[str, float]:
        """
        Compute tracking error decomposition: TE, factor_te, residual_te.

        Factor exposure of active weights: f_delta = delta @ beta  -> (M,)
        """
        delta = active_weights.loc[self.tickers].values  # (N,)
        betas = self.covar_data.y_betas.loc[self.tickers, :].values  # (N x M)

        active_factor_exp = delta @ betas  # (N,) @ (N,M) = (M,)
        factor_te_var = float(active_factor_exp @ self.covar_data.x_covar.values @ active_factor_exp)

        residual_vars = self.covar_data.y_variances.loc[self.tickers, VarianceColumns.RESIDUAL_VARS.value].values
        residual_te_var = float((delta ** 2) @ residual_vars)

        total_te_var = factor_te_var + residual_te_var

        return {
            'tracking_error': np.sqrt(total_te_var),
            'factor_te': np.sqrt(factor_te_var),
            'residual_te': np.sqrt(residual_te_var),
        }

    def compute_returns_risk_snapshot(self) -> pd.DataFrame:
        """
        Decompose portfolio variance for all portfolios, benchmarks, and current.
        Portfolio-level rows also include tracking error metrics (TE, factor_te, residual_te).

        Returns:
            DataFrame: rows = metric names, columns = all weight vectors.
        """
        result = {}

        # All portfolios — include TE metrics
        for name in self.portfolio_names:
            risk = self._compute_risk_decomposition(self.get_weights(name))
            te = self._compute_tracking_error_metrics(self.get_active_weights(name))
            risk.update(te)
            result[name] = risk

        # Benchmarks — no TE
        if self._shared_benchmark:
            result[self._benchmark_name] = self._compute_risk_decomposition(
                self._benchmark_df.iloc[:, 0]
            )
        else:
            for name in self.portfolio_names:
                result[f'{name}_benchmark'] = self._compute_risk_decomposition(
                    self.get_benchmark(name)
                )

        # Current weights — no TE
        if self.has_current_weights:
            seen = set()
            for name in self.portfolio_names:
                current_w = self.get_current(name)
                current_key = tuple(current_w.values.round(10))
                if current_key not in seen:
                    label = 'current' if len(seen) == 0 and isinstance(self.current_weights, pd.Series) \
                        else f'{name}_current'
                    result[label] = self._compute_risk_decomposition(current_w)
                    seen.add(current_key)

        return pd.DataFrame(result)

    def _compute_active_risk_attribution(self, active_weights: pd.Series) -> Dict[str, float]:
        """
        Decompose tracking error for a given active weight vector (full detail).

        Factor exposure: f_delta = delta @ beta  -> (M,)
        """
        delta = active_weights.loc[self.tickers].values  # (N,)
        betas = self.covar_data.y_betas.loc[self.tickers, :].values  # (N x M)

        active_factor_exp = delta @ betas  # (N,) @ (N,M) = (M,)
        factor_te_var = float(active_factor_exp @ self.covar_data.x_covar.values @ active_factor_exp)

        residual_vars = self.covar_data.y_variances.loc[self.tickers, VarianceColumns.RESIDUAL_VARS.value].values
        residual_te_var = float((delta ** 2) @ residual_vars)

        total_te_var = factor_te_var + residual_te_var

        return {
            'tracking_error': np.sqrt(total_te_var),
            'factor_te': np.sqrt(factor_te_var),
            'residual_te': np.sqrt(residual_te_var),
            'factor_pct': factor_te_var / total_te_var if total_te_var > 0 else 0.0,
            'residual_pct': residual_te_var / total_te_var if total_te_var > 0 else 0.0
        }

    def compute_active_risk_attribution(self) -> pd.DataFrame:
        """
        Active risk attribution for all portfolios.

        Returns:
            DataFrame: rows = metric names, columns = portfolio names with '_active' suffix.
        """
        result = {}
        for name in self.portfolio_names:
            result[f'{name}_active'] = self._compute_active_risk_attribution(
                self.get_active_weights(name)
            )
        return pd.DataFrame(result)

    def _compute_factor_exposures_for_weights(self, weights: pd.Series) -> pd.Series:
        """
        Factor exposures: f = w @ beta for arbitrary weight vector.

        w is (N,), beta is (N x M), result is (M,) indexed by factor names.
        """
        w = weights.loc[self.tickers].values  # (N,)
        betas = self.covar_data.y_betas.loc[self.tickers, :]  # (N x M)
        exposures = w @ betas.values  # (N,) @ (N,M) = (M,)
        return pd.Series(exposures, index=betas.columns)

    def _compute_factor_risk_contribution(self, weights: pd.Series) -> pd.Series:
        """
        Marginal contribution of each factor to portfolio variance for given weights.

        factor_exp = w @ beta  -> (M,)
        factor_contrib_k = (Sigma_x @ factor_exp)_k * factor_exp_k

        Returns:
            Risk contribution per factor (sums to 1.0 within factor variance).
        """
        w = weights.loc[self.tickers].values  # (N,)
        betas = self.covar_data.y_betas.loc[self.tickers, :].values  # (N x M)
        x_covar = self.covar_data.x_covar.values  # (M x M)

        factor_exp = w @ betas  # (N,) @ (N,M) = (M,)
        factor_var = float(factor_exp @ x_covar @ factor_exp)

        factor_contrib = (x_covar @ factor_exp) * factor_exp

        return pd.Series(
            factor_contrib / factor_var if factor_var > 0 else factor_contrib,
            index=self.covar_data.x_covar.index,
        )

    # === Attribution by grouping (asset class, currency) ===

    def _compute_group_attribution(
        self,
        groups: pd.Series,
        weights: pd.Series
    ) -> pd.DataFrame:
        """
        Compute weight and risk attribution by grouping for a single weight vector.

        Returns:
            DataFrame with weight, risk_contrib, risk_pct per group.
        """
        cov = self.covar_data.y_covar.loc[self.tickers, self.tickers]
        w = weights.loc[self.tickers]

        port_var = float(w.values @ cov.values @ w.values)
        port_vol = np.sqrt(port_var)

        mctr = cov @ w / port_vol if port_vol > 0 else cov @ w
        risk_contrib = w * mctr

        result = pd.DataFrame({
            'weight': w.groupby(groups).sum(),
            'risk_contrib': risk_contrib.groupby(groups).sum(),
        })
        result['risk_pct'] = result['risk_contrib'] / result['risk_contrib'].sum()

        return result.sort_values('risk_contrib', ascending=False)

    def _get_all_labelled_weight_vectors(self) -> List[tuple]:
        """
        Build ordered list of (label, weights_series) for all weight vectors
        used in group/factor attribution.

        Order: portfolios, current (if any), benchmarks, active, current_active (if any).
        """
        vectors = []

        # Portfolios
        for name in self.portfolio_names:
            vectors.append((name, self.get_weights(name)))

        # Current weights
        if self.has_current_weights:
            if isinstance(self.current_weights, pd.Series):
                vectors.append(('current', self._current_df.iloc[:, 0]))
            else:
                for name in self.portfolio_names:
                    vectors.append((f'{name}_current', self.get_current(name)))

        # Benchmarks
        if self._shared_benchmark:
            vectors.append((self._benchmark_name, self._benchmark_df.iloc[:, 0]))
        else:
            for name in self.portfolio_names:
                vectors.append((f'{name}_benchmark', self.get_benchmark(name)))

        # Active weights
        for name in self.portfolio_names:
            vectors.append((f'{name}_active', self.get_active_weights(name)))

        # Current active weights
        if self.has_current_weights:
            if isinstance(self.current_weights, pd.Series):
                vectors.append(('current_active', self._current_df.iloc[:, 0] - self._benchmark_df.iloc[:, 0]))
            else:
                for name in self.portfolio_names:
                    vectors.append((f'{name}_current_active',
                                    self.get_current(name) - self.get_benchmark(name)))

        return vectors

    def compute_group_attribution(self) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Compute weight and risk attribution by group for all weight vectors.

        Returns:
            Dict[group_name -> Dict[metric -> DataFrame]] where:
                metric in {'weight', 'rc', 'rc_pct', 'bounds' (if available)}
                Each DataFrame: rows = portfolio labels, columns = group labels
                    (sorted by first portfolio's risk contrib desc)
        """
        vectors = self._get_all_labelled_weight_vectors()
        labels = [label for label, _ in vectors]

        result = {}
        for group_name, groups in self.group_attributions.items():
            attribs = {}
            for label, w in vectors:
                attribs[label] = self._compute_group_attribution(groups, w)

            # Use first portfolio for column ordering
            first_label = vectors[0][0]
            col_order = attribs[first_label].index

            group_result = {}

            # Bounds if available
            if self.ac_bounds is not None:
                bounds_rows = {}
                for col in self.ac_bounds.columns:
                    bound_series = self.ac_bounds[col].reindex(col_order)
                    if bound_series.notna().any():
                        bounds_rows[col] = bound_series
                if bounds_rows:
                    group_result['bounds'] = pd.DataFrame(bounds_rows).T[col_order]

            # Separate DataFrame per metric
            for metric, src_col in [('weight', 'weight'), ('rc', 'risk_contrib'), ('rc_pct', 'risk_pct')]:
                metric_rows = {}
                for label in labels:
                    metric_rows[label] = attribs[label][src_col]
                group_result[metric] = pd.DataFrame(metric_rows).T[col_order]

            result[group_name] = group_result
        return result

    def compute_group_allocation(self, group: str, name: str = None, weights_to_pct: bool = False) -> pd.Series:
        name = name or self.portfolio_names[0]
        group_data = self.group_attributions[group]
        allocation = self.get_weights(name).groupby(group_data).sum()
        if weights_to_pct:
            allocation = round_weights_to_pct(allocation)
        return allocation

    def compute_weight_summary(self, name: str = None, weights_to_pct: bool = False) -> pd.DataFrame:
        """Weight summary for a specific portfolio (or first if not specified)."""
        name = name or self.portfolio_names[0]
        w = self.get_weights(name)
        bench = self.get_benchmark(name)
        active = w - bench

        weight_components = [
            w.rename('new'),
            bench.rename('benchmark'),
            active.rename('active'),
        ]
        if self.has_current_weights:
            current = self.get_current(name)
            trade = w - current
            weight_components.extend([
                current.rename('current'),
                trade.rename('trade')
            ])

        weights_df = pd.concat(weight_components, axis=1)

        if weights_to_pct:
            level_cols = [c for c in weights_df.columns if c not in ('active', 'trade')]
            weights_df[level_cols] = weights_df[level_cols].apply(round_weights_to_pct)
            weights_df['active'] = weights_df['new'] - weights_df['benchmark']
            if 'trade' in weights_df.columns:
                weights_df['trade'] = weights_df['new'] - weights_df['current']
        return weights_df

    def compute_all_weights_summary(self, weights_to_pct: bool = False) -> Dict[str, pd.DataFrame]:
        """
        Weight summary across all portfolios, returned as separate tables
        for portfolio weights, benchmark weights, and active weights.

        Each table: rows = tickers, columns = portfolio names.

        Returns:
            Dict with keys 'portfolio', 'benchmark', 'active',
            and optionally 'current', 'trade' if current_weights provided.
        """
        portfolio_weights = {}
        benchmark_weights = {}
        active_weights = {}
        current_w = {}
        trade_w = {}

        for name in self.portfolio_names:
            w = self.get_weights(name)
            bench = self.get_benchmark(name)
            portfolio_weights[name] = w
            benchmark_weights[name] = bench
            active_weights[name] = w - bench
            if self.has_current_weights:
                cur = self.get_current(name)
                current_w[name] = cur
                trade_w[name] = w - cur

        def _build_df(data: Dict[str, pd.Series]) -> pd.DataFrame:
            df = pd.DataFrame(data)  # rows = tickers, cols = portfolio names
            if weights_to_pct:
                df = df.apply(round_weights_to_pct, axis=0)
            return df

        result = {
            'portfolio': _build_df(portfolio_weights),
            'benchmark': _build_df(benchmark_weights),
            'active': _build_df(active_weights),
        }
        if self.has_current_weights:
            result['current'] = _build_df(current_w)
            result['trade'] = _build_df(trade_w)

        return result

    def compute_factor_exposures_summary(self) -> Dict[str, pd.DataFrame]:
        """
        Factor exposures and risk contributions split by weight type
        (portfolio, benchmark, active), each as a plain-index DataFrame.

        Returns:
            Dict with keys:
                'exposure_portfolio': DataFrame (rows=portfolio names, cols=factors, float)
                'exposure_benchmark': DataFrame (rows=benchmark labels, cols=factors, float)
                'exposure_active': DataFrame (rows=active labels, cols=factors, float)
                'risk_pct_portfolio': DataFrame (rows=portfolio names, cols=factors, pct)
                'risk_pct_benchmark': DataFrame (rows=benchmark labels, cols=factors, pct)
                'risk_pct_active': DataFrame (rows=active labels, cols=factors, pct)
        """
        # Portfolio exposures
        portfolio_exp = {}
        portfolio_risk = {}
        for name in self.portfolio_names:
            w = self.get_weights(name)
            portfolio_exp[name] = self._compute_factor_exposures_for_weights(w)
            portfolio_risk[name] = self._compute_factor_risk_contribution(w)

        # Benchmark exposures — use original mandate names
        benchmark_exp = {}
        benchmark_risk = {}
        if self._shared_benchmark:
            bench_w = self._benchmark_df.iloc[:, 0]
            benchmark_exp[self._benchmark_name] = self._compute_factor_exposures_for_weights(bench_w)
            benchmark_risk[self._benchmark_name] = self._compute_factor_risk_contribution(bench_w)
        else:
            for name in self.portfolio_names:
                bench_w = self.get_benchmark(name)
                benchmark_exp[name] = self._compute_factor_exposures_for_weights(bench_w)
                benchmark_risk[name] = self._compute_factor_risk_contribution(bench_w)

        # Active exposures — use original mandate names
        active_exp = {}
        active_risk = {}
        for name in self.portfolio_names:
            active_w = self.get_active_weights(name)
            active_exp[name] = self._compute_factor_exposures_for_weights(active_w)
            active_risk[name] = self._compute_factor_risk_contribution(active_w)

        return {
            'exposure_portfolio': pd.DataFrame(portfolio_exp).T,
            'exposure_benchmark': pd.DataFrame(benchmark_exp).T,
            'exposure_active': pd.DataFrame(active_exp).T,
            'risk_pct_portfolio': pd.DataFrame(portfolio_risk).T,
            'risk_pct_benchmark': pd.DataFrame(benchmark_risk).T,
            'risk_pct_active': pd.DataFrame(active_risk).T,
        }

    def compute_risk_summary(self) -> Dict[str, pd.DataFrame]:
        """
        Risk summary split into portfolio and benchmark tables.

        Portfolio table includes tracking_error, factor_te, residual_te.
        Benchmark table uses original mandate names (no _benchmark suffix).

        Returns:
            Dict with keys:
                'portfolio': DataFrame (rows=portfolio names, cols=risk metrics + TE)
                'benchmark': DataFrame (rows=mandate names, cols=risk metrics)
        """
        # Portfolio risk with TE
        portfolio_result = {}
        for name in self.portfolio_names:
            risk = self._compute_risk_decomposition(self.get_weights(name))
            te = self._compute_tracking_error_metrics(self.get_active_weights(name))
            risk.update(te)
            portfolio_result[name] = risk

        # Benchmark risk — use original mandate names
        benchmark_result = {}
        if self._shared_benchmark:
            benchmark_result[self._benchmark_name] = self._compute_risk_decomposition(
                self._benchmark_df.iloc[:, 0]
            )
        else:
            for name in self.portfolio_names:
                benchmark_result[name] = self._compute_risk_decomposition(
                    self.get_benchmark(name)
                )

        return {
            'portfolio': pd.DataFrame(portfolio_result).T,
            'benchmark': pd.DataFrame(benchmark_result).T,
        }

    # === Reporting ===

    def report(self, name: str = None, add_asset_details: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Generate comprehensive risk attribution report for a specific portfolio.

        Args:
            name: Portfolio name. Defaults to first portfolio.
            add_asset_details: Whether to prepend metadata to weights table.

        Returns:
            Dict with report sections.
        """
        name = name or self.portfolio_names[0]

        weights_df = self.compute_weight_summary(name=name)
        if add_asset_details:
            weights_df = pd.concat([self.metadata, weights_df], axis=1)

        risk_summary = self.compute_risk_summary()
        factor_exp_dict = self.compute_factor_exposures_summary()

        vol_port = self.covar_data.get_model_vols()

        # Asset-level snapshot
        w = self.get_weights(name)
        active_w = self.get_active_weights(name)

        if self.expected_return is not None:
            asset_snapshot = self.expected_return.to_frame('CMA')
        else:
            asset_snapshot = pd.DataFrame(index=self.tickers)

        asset_snapshot = asset_snapshot.loc[self.tickers]
        asset_snapshot['weight'] = w
        asset_snapshot['active_weight'] = active_w
        if self.has_current_weights:
            asset_snapshot['current_weight'] = self.get_current(name)
            asset_snapshot['trade'] = self.get_trade_weights(name)

        asset_snapshot = pd.concat([asset_snapshot, self.metadata, vol_port], axis=1)

        result = {
            'weights': weights_df,
            'risk_summary': risk_summary,
            'asset_snapshot': asset_snapshot
        }
        result.update(factor_exp_dict)

        # Turnover analysis
        if self.has_current_weights:
            turnover_analysis = self.compute_turnover_analysis(name=name)
            result['turnover'] = turnover_analysis.to_frame('value')

        group_attribs = self.compute_group_attribution()
        result.update(group_attribs)

        return result

    def to_weights_df(self, name: str = None) -> pd.DataFrame:
        """Simple weight comparison DataFrame."""
        name = name or self.portfolio_names[0]
        w = self.get_weights(name)
        bench = self.get_benchmark(name)
        cols = [
            w.rename('new'),
            bench.rename('benchmark'),
            (w - bench).rename('active'),
        ]
        if self.has_current_weights:
            current = self.get_current(name)
            cols.extend([
                current.rename('current'),
                (w - current).rename('trade')
            ])
        return pd.concat(cols, axis=1)

    def summary(self, name: str = None) -> pd.Series:
        """One-line summary metrics for a specific portfolio."""
        name = name or self.portfolio_names[0]
        w = self.get_weights(name)
        bench = self.get_benchmark(name)
        active = w - bench

        metrics = {
            'n_assets': self.n_assets,
            'total_vol': self.compute_portfolio_vol(w),
            'tracking_error': self.compute_portfolio_vol(active),
            'benchmark_vol': self.compute_portfolio_vol(bench),
            'max_active_weight': active.abs().max(),
            'sum_active_long': active[active > 0].sum(),
            'sum_active_short': active[active < 0].sum(),
        }

        if self.has_current_weights:
            metrics['turnover'] = self.compute_turnover(name=name)
            trade = self.get_trade_weights(name)
            metrics['n_trades'] = (trade.abs() > 1e-6).sum()

        return pd.Series(metrics, name=f'{name}_summary')

    def compute_efficient_frontier_data(
        self,
        profiles: Optional[Dict[str, List[str]]] = None
    ) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """
        Compute expected return and volatility for portfolio and benchmark weights,
        grouped by profile (e.g. 'with Alts', 'without Alts').

        Args:
            profiles: Dict mapping profile name to list of portfolio names.
                      If None, returns a single profile 'all' containing all portfolios.
                      Example: {'w/o Alts': ['Income w/o Alts', 'Low w/o Alts', ...],
                                'with Alts': ['Income with Alts', 'Low with Alts', ...]}

        Returns:
            Tuple of:
                dfs: concatenated DataFrame with columns [mandate, exp_return, total_vol, hue]
                result: Dict with keys '{profile} - portfolio' and '{profile} - benchmark',
                    each mapping to DataFrame with index=portfolio names,
                    columns=['exp_return', 'total_vol']
        """
        if profiles is None:
            profiles = {'all': self.portfolio_names}

        result = {}
        for profile_name, names in profiles.items():
            portfolio_rows = {}
            benchmark_rows = {}
            for name in names:
                w_risk = self._compute_risk_decomposition(self.get_weights(name))
                b_risk = self._compute_risk_decomposition(self.get_benchmark(name))
                portfolio_rows[name] = {
                    'exp_return': w_risk.get('exp_return', np.nan),
                    'total_vol': w_risk['total_vol'],
                }
                benchmark_rows[name] = {
                    'exp_return': b_risk.get('exp_return', np.nan),
                    'total_vol': b_risk['total_vol'],
                }
            result[f'{profile_name} - portfolio'] = pd.DataFrame(portfolio_rows).T
            result[f'{profile_name} - benchmark'] = pd.DataFrame(benchmark_rows).T

        dfs = []
        for key, df in result.items():
            df = df.reset_index(drop=False, names=['mandate'])
            df['hue'] = key
            dfs.append(df)
        dfs = pd.concat(dfs, axis=0, ignore_index=True)

        return dfs, result

    def __repr__(self) -> str:
        name = self.portfolio_names[0]
        w = self.get_weights(name)
        vol = self.compute_portfolio_vol(w)
        te = self.compute_tracking_error(name)
        turnover_str = ""
        if self.has_current_weights:
            turnover = self.compute_turnover(name=name)
            turnover_str = f", turnover={turnover:.2%}"
        return (
            f"PortfolioOptimisationResult("
            f"n_portfolios={self.n_portfolios}, "
            f"n_assets={self.n_assets}, "
            f"vol={vol:.2%}, "
            f"TE={te:.2%}{turnover_str}, "
            f"date={self.optimisation_date})"
        )