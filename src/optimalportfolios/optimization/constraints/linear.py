"""Named signed linear policies, independent of a solver or financial interpretation."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterator, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class LinearConstraints:
    """Named rows ``lower_j <= loadings[j] @ weights <= upper_j``.

    Loadings may be signed. Bounds carry each row's own units; no rescaling or
    annualisation is inferred. Missing bound labels, None and NaN are unbounded;
    -inf on the lower side and +inf on the upper side are also unbounded.
    Input pandas objects are copied on construction, but remain mutable: treat
    them as policy inputs and use copy() to obtain independent replacements.

    Attributes:
        loadings: Finite coefficients, assets in rows and uniquely named policies
            in columns. Every asset in the solver universe must be supplied.
        lower: Optional lower bounds indexed by policy names.
        upper: Optional upper bounds indexed by policy names.
    """

    loadings: pd.DataFrame
    lower: Optional[pd.Series] = None
    upper: Optional[pd.Series] = None

    def __post_init__(self) -> None:
        """Validate labels and finite coefficients and copy caller-owned inputs."""
        if not isinstance(self.loadings, pd.DataFrame) or self.loadings.empty:
            raise ValueError('loadings must be a nonempty DataFrame')
        if not self.loadings.index.is_unique or not self.loadings.columns.is_unique:
            raise ValueError('loadings asset and row labels must be unique')
        if self.loadings.index.hasnans or self.loadings.columns.hasnans:
            raise ValueError('loadings labels must not be missing')
        loadings = self.loadings.astype(float).copy()
        if not np.isfinite(loadings.to_numpy()).all():
            raise ValueError('loadings must contain only finite coefficients')
        object.__setattr__(self, 'loadings', loadings)
        for side in ('lower', 'upper'):
            bound = getattr(self, side)
            if bound is None:
                continue
            if not isinstance(bound, pd.Series) or not bound.index.is_unique:
                raise ValueError(f'{side} must be a Series with unique row labels')
            unknown = bound.index.difference(loadings.columns)
            if len(unknown):
                raise ValueError(f'{side} contains unknown row labels: {unknown.tolist()}')
            bound = bound.reindex(loadings.columns).astype(float).copy()
            invalid = np.isposinf(bound) if side == 'lower' else np.isneginf(bound)
            if invalid.any():
                raise ValueError(f'{side} has an impossible infinite bound')
            object.__setattr__(self, side, bound)
        if self.lower is not None and self.upper is not None:
            if (self.lower > self.upper).any():
                raise ValueError('linear lower bounds must not exceed upper bounds')

    def copy(self, **overrides) -> LinearConstraints:
        """Return independently copied policy data with optional field replacements."""
        return replace(self, **overrides)

    def iter_bounds(self) -> Iterator[Tuple[object, pd.Series, Optional[float], Optional[float]]]:
        """Yield bounded rows in column order, with None for an unbounded side."""
        for name in self.loadings.columns:
            lower = None if self.lower is None else self.lower[name]
            upper = None if self.upper is None else self.upper[name]
            lower = float(lower) if lower is not None and np.isfinite(lower) else None
            upper = float(upper) if upper is not None and np.isfinite(upper) else None
            if lower is not None or upper is not None:
                yield name, self.loadings[name], lower, upper

    def update(self, valid_tickers: Sequence[str]) -> LinearConstraints:
        """Reorder the universe, rejecting unknown assets or dropped nonzero loadings.

        Dropping an asset is allowed only when its coefficient is exactly zero
        in every bounded row. A caller intentionally changing the investment
        universe must explicitly rebuild the policy for that universe.
        """
        index = pd.Index(valid_tickers)
        if not index.is_unique:
            raise ValueError('valid_tickers must be unique')
        unknown = index.difference(self.loadings.index)
        if len(unknown):
            raise ValueError(f'Missing linear loadings for assets: {unknown.tolist()}')
        dropped = self.loadings.index.difference(index)
        for name, loading, _, _ in self.iter_bounds():
            if loading.loc[dropped].ne(0.0).any():
                raise ValueError(f"Cannot drop loaded assets from linear row '{name}'")
        return self.copy(loadings=self.loadings.loc[index])
