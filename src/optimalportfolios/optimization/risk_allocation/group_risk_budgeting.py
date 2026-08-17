"""Translate point-in-time group memberships into asset-level risk budgets."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _compute_group_risk_budget_row(
        groups: pd.Series,
        *,
        group_size_exponent: float,
        ) -> pd.Series:
    """Return one asset budget vector from one group-membership vector."""
    if not groups.index.is_unique:
        raise ValueError("group asset labels must be unique")
    valid = groups.dropna()
    if valid.empty:
        raise ValueError("at least one valid group membership is required")

    sizes = valid.value_counts(sort=False).astype(float)
    group_budgets = sizes.pow(group_size_exponent)
    denominator = float(group_budgets.sum())
    if not np.isfinite(denominator) or denominator <= 0.0:
        raise ValueError("group-size budgets must have a positive finite sum")
    group_budgets /= denominator

    budgets = pd.Series(0.0, index=groups.index, name="risk_budget")
    budgets.loc[valid.index] = valid.map(group_budgets / sizes).astype(float)
    return budgets


def compute_group_risk_budgets(
        groups: pd.Series | pd.DataFrame,
        *,
        group_size_exponent: float = 0.0,
        ) -> pd.Series | pd.DataFrame:
    """Convert group labels into asset risk budgets.

    For a group ``g`` containing ``n_g`` classified assets, its aggregate budget is
    ``n_g**alpha / sum_h(n_h**alpha)`` and each member receives an equal share of that
    budget. Unclassified assets receive zero. The useful endpoints are ``alpha=0`` for
    equal group budgets and ``alpha=1`` for equal asset budgets; ``alpha=0.5`` supplies
    the square-root-size compromise.

    A Series is interpreted as one asset-to-group mapping. A DataFrame is interpreted
    point in time, with observations on rows and assets on columns; every row is computed
    independently, so future group membership cannot affect an earlier budget.

    Args:
        groups: Current group labels or a row-wise membership panel.
        group_size_exponent: Exponent controlling shrinkage toward equal asset budgets.

    Returns:
        Asset risk budgets with the same shape and labels as ``groups``.

    Raises:
        TypeError: If ``groups`` is not a Series or DataFrame.
        ValueError: If labels are duplicated, the exponent is not finite, or an
            observation has no valid group membership.
    """
    exponent = float(group_size_exponent)
    if not np.isfinite(exponent):
        raise ValueError("group_size_exponent must be finite")

    if isinstance(groups, pd.Series):
        return _compute_group_risk_budget_row(
            groups, group_size_exponent=exponent,
        )
    if not isinstance(groups, pd.DataFrame):
        raise TypeError("groups must be a pandas Series or DataFrame")
    if groups.empty:
        raise ValueError("groups must contain at least one observation and one asset")
    if not groups.index.is_unique:
        raise ValueError("group observation labels must be unique")
    if not groups.columns.is_unique:
        raise ValueError("group asset labels must be unique")

    rows = []
    for observation, row in groups.iterrows():
        try:
            budget = _compute_group_risk_budget_row(
                row, group_size_exponent=exponent,
            )
        except ValueError as exc:
            raise ValueError(f"invalid groups at observation {observation!r}: {exc}") from exc
        budget.name = observation
        rows.append(budget)
    budgets = pd.DataFrame(rows, index=groups.index, columns=groups.columns, dtype=float)
    budgets.index.name = groups.index.name
    budgets.columns.name = groups.columns.name
    return budgets
