"""Benchmark-relative portfolio constraints and beta-loading helpers.

This internal module defines benchmark-relative sector, style, and beta
constraints. It translates those specifications into linear CVXPY constraints
without estimating covariance matrices or constructing optimiser objectives.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import List, Optional

import cvxpy as cvx
import numpy as np
import pandas as pd
from cvxpy.constraints.nonpos import Inequality

from optimalportfolios.utils.benchmark_beta import (
    compute_benchmark_beta_loadings as compute_benchmark_beta_loadings,
    compute_benchmark_beta_loadings_from_covar as compute_benchmark_beta_loadings_from_covar,
)


@dataclass(frozen=True)
class BenchmarkDeviationConstraints:
    """Constrain benchmark-relative sector, industry, or style exposures.

    For every loading column ``g``, the constraint is
    ``|L_g.T @ (w - benchmark_weights)| <= factor_max_deviation[g]``.

    Attributes:
        factor_loading_mat: Asset-by-group loading matrix. Sector loadings are normally binary
            membership indicators; style loadings are normally continuous factor exposures.
        factor_max_deviation: Maximum absolute active exposure for each loading column. Sector
            limits are normally in portfolio-weight units; style-limit units follow the scaling
            of the supplied style loadings.
    """
    factor_loading_mat: pd.DataFrame
    factor_max_deviation: pd.Series

    def __post_init__(self):
        """Validate that at least one constraint type is specified and aligned."""
        if self.factor_max_deviation is not None:
            this = self.factor_max_deviation.index.isin(self.factor_loading_mat.columns)
            if not this.all():
                missing = self.factor_max_deviation.index[~this]
                warnings.warn(f"factor_max_deviation entries not in factor_loading_mat.columns: {missing.tolist()}")
        else:
            raise ValueError("factor_max_deviation must be given")

    def copy(self) -> BenchmarkDeviationConstraints:
        """Return a deep copy of the deviation loadings and bounds."""
        return BenchmarkDeviationConstraints(
            factor_loading_mat=self.factor_loading_mat.copy(),
            factor_max_deviation=self.factor_max_deviation.copy(),
        )

    def update(self, valid_tickers: List[str]) -> BenchmarkDeviationConstraints:
        """Filter benchmark-deviation loadings to valid tickers.

        Args:
            valid_tickers: Asset labels retained by the solver wrapper.

        Returns:
            A new aligned benchmark-deviation constraint.
        """
        new_self = BenchmarkDeviationConstraints(
            factor_loading_mat=self.factor_loading_mat.loc[valid_tickers, :],
            factor_max_deviation=self.factor_max_deviation
        )
        return new_self

    def set_cvx_constraints(
            self,
            w: cvx.Variable,
            benchmark_weights: pd.Series,
    ) -> List[Inequality]:
        """Build the absolute active-deviation inequalities for each factor.

        Groups whose loadings are all zero are skipped, so an unloaded group cannot
        make the problem infeasible.

        Args:
            w: Weight variable.
            benchmark_weights: Benchmark weights defining the active positions.

        Returns:
            One ``|c' (w - b)| <= max_deviation`` inequality per loaded factor.
        """
        constraints = []
        for group in self.factor_max_deviation.index:
            group_loading = self.factor_loading_mat[group]
            if np.any(np.isclose(group_loading, 0.0) == False):  # exclude groups with zero loading
                # Align indices
                group_loading = group_loading.loc[benchmark_weights.index]
                active_deviation = cvx.sum(cvx.multiply(group_loading.to_numpy(), w - benchmark_weights.to_numpy()))
                constraints += [cvx.abs(active_deviation) <= self.factor_max_deviation.loc[group]]
        return constraints

    def print(self):
        """Print constraint details."""
        print(f"factor_loading_mat:\n{self.factor_loading_mat}")
        print(f"factor_max_deviation:\n{self.factor_max_deviation}")


@dataclass(frozen=True)
class BenchmarkBetaConstraint:
    """Range constraint on ex-ante portfolio beta to a (static) benchmark.

    Given per-asset ``beta_loadings`` c with beta(w) = c'w (see
    ``compute_benchmark_beta_loadings``), creates linear constraints:

        beta_min <= c' @ w <= beta_max

    Follows the ``weights_0`` convention for per-rebalance state: the
    (beta_min, beta_max) spec is static, while ``beta_loadings`` depend on
    the rolling covariance and are injected per rebalancing date via
    ``with_loadings`` before ``set_cvx_all_constraints`` is called.

    Attributes:
        beta_min: Lower bound on ex-ante beta (None = unbounded below).
        beta_max: Upper bound on ex-ante beta (None = unbounded above).
        beta_loadings: Per-asset loadings c (indexed by asset). None until
            injected for the current rebalancing date.
    """
    beta_min: Optional[float] = None
    beta_max: Optional[float] = None
    beta_loadings: Optional[pd.Series] = None

    def __post_init__(self):
        """Validate that a bound is given and that the range is not inverted."""
        if self.beta_min is None and self.beta_max is None:
            raise ValueError("at least one of beta_min / beta_max must be given")
        if (self.beta_min is not None and self.beta_max is not None
                and self.beta_min > self.beta_max):
            raise ValueError(f"beta_min={self.beta_min} > beta_max={self.beta_max}")

    def copy(self) -> BenchmarkBetaConstraint:
        """Return a copy of the bounds and of the loadings, when present."""
        return BenchmarkBetaConstraint(
            beta_min=self.beta_min,
            beta_max=self.beta_max,
            beta_loadings=self.beta_loadings.copy() if self.beta_loadings is not None else None,
        )

    def with_loadings(self, beta_loadings: pd.Series) -> BenchmarkBetaConstraint:
        """Create an instance carrying this rebalance's beta loadings.

        Args:
            beta_loadings: Ticker-indexed linear benchmark-beta loadings.

        Returns:
            A new constraint with the same bounds and supplied loadings.
        """
        return BenchmarkBetaConstraint(
            beta_min=self.beta_min, beta_max=self.beta_max,
            beta_loadings=beta_loadings)

    def update(self, valid_tickers: List[str]) -> BenchmarkBetaConstraint:
        """Filter loadings to valid tickers (dropped names carry zero weight).

        Args:
            valid_tickers: Asset labels retained by the solver wrapper.

        Returns:
            A new aligned constraint, or this instance if loadings are absent.
        """
        if self.beta_loadings is None:
            return self
        return BenchmarkBetaConstraint(
            beta_min=self.beta_min, beta_max=self.beta_max,
            beta_loadings=self.beta_loadings.reindex(valid_tickers).fillna(0.0))

    def set_cvx_beta_constraints(self, w: cvx.Variable) -> List[Inequality]:
        """Two linear inequalities beta_min <= c'w <= beta_max."""
        if self.beta_loadings is None:
            raise ValueError(
                "beta_loadings not set — inject per-rebalance loadings via "
                "with_loadings() before building cvx constraints")
        c = self.beta_loadings.to_numpy()
        constraints = []
        if self.beta_min is not None:
            constraints += [c @ w >= self.beta_min]
        if self.beta_max is not None:
            constraints += [c @ w <= self.beta_max]
        return constraints

    def print(self):
        """Print the beta range and the current loadings."""
        print(f"beta range: [{self.beta_min}, {self.beta_max}]")
        print(f"beta_loadings:\n{self.beta_loadings}")
