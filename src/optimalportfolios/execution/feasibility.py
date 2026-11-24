"""Core interval diagnostics for model-execution group-bound feasibility.

The helpers in this module diagnose executable linear ranges only. They never
widen product bounds or modify a portfolio. Target construction and rescue use
the same arithmetic so a reported bridge has one stable meaning throughout the
execution pipeline.
"""

import numpy as np
import pandas as pd


BRIDGE_COLUMNS = (
    "e_min",
    "e_max",
    "group_min",
    "group_max",
    "bridge_min",
    "bridge_max",
)


def compute_group_bound_bridges(
    lower_bounds: pd.Series,
    upper_bounds: pd.Series,
    group_loadings: pd.DataFrame,
    group_min: pd.Series,
    group_max: pd.Series,
) -> pd.DataFrame:
    """Compute interval feasibility of every group against executable bounds.

    Args:
        lower_bounds: Executable lower weight per instrument.
        upper_bounds: Executable upper weight per instrument.
        group_loadings: Instrument-by-group linear exposure matrix.
        group_min: Hard minimum exposure per group. Missing or NaN means no
            minimum constraint.
        group_max: Hard maximum exposure per group. Missing or NaN means no
            maximum constraint.

    Returns:
        Group-indexed table with executable exposure range, hard bounds, and
        positive minimum/maximum bridges. Values are unrounded portfolio
        weights.
    """
    if not lower_bounds.index.is_unique or not upper_bounds.index.is_unique:
        raise ValueError("executable bound indices must be unique")
    if not group_loadings.index.is_unique or not group_loadings.columns.is_unique:
        raise ValueError("group loading axes must be unique")
    index = lower_bounds.index
    upper = upper_bounds.reindex(index)
    loadings = group_loadings.reindex(index=index)
    if upper.isna().any() or loadings.isna().any().any():
        raise ValueError("executable bounds and aligned group loadings must be complete")
    lower = lower_bounds.astype(float)
    upper = upper.astype(float)
    loadings = loadings.astype(float)
    if (
        not np.isfinite(lower.to_numpy()).all()
        or not np.isfinite(upper.to_numpy()).all()
        or not np.isfinite(loadings.to_numpy()).all()
    ):
        raise ValueError("executable bounds and group loadings must be finite")
    invalid = lower > upper
    if invalid.any():
        raise ValueError(
            f"executable lower bound exceeds upper bound for {invalid.index[invalid].tolist()}"
        )

    positive = loadings.ge(0.0)
    lower_matrix = pd.DataFrame(
        np.where(positive, lower.to_numpy()[:, None], upper.to_numpy()[:, None]),
        index=index,
        columns=loadings.columns,
    )
    upper_matrix = pd.DataFrame(
        np.where(positive, upper.to_numpy()[:, None], lower.to_numpy()[:, None]),
        index=index,
        columns=loadings.columns,
    )
    e_min = (loadings * lower_matrix).sum(axis=0)
    e_max = (loadings * upper_matrix).sum(axis=0)
    minimum = group_min.reindex(loadings.columns).astype(float)
    maximum = group_max.reindex(loadings.columns).astype(float)
    bridge_min = (minimum - e_max).clip(lower=0.0).fillna(0.0)
    bridge_max = (e_min - maximum).clip(lower=0.0).fillna(0.0)
    out = pd.DataFrame(
        {
            "e_min": e_min,
            "e_max": e_max,
            "group_min": minimum,
            "group_max": maximum,
            "bridge_min": bridge_min,
            "bridge_max": bridge_max,
        }
    )
    out.index.name = "group"
    return out.loc[:, BRIDGE_COLUMNS]
