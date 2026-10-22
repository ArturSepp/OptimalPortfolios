"""Explicit final-state adapter, without lifecycle reconstruction or hidden repair."""

import numpy as np

from optimalportfolios.execution import schema as s
from .types import ExecutionProblem


def from_resolved_problem(
    problem, rules, *, prepared, full_lower, full_upper, provenance=None, preparation_seconds=0.0
):
    """Adapt a saved v1 state and its separately qualified common preparation.

    ``prepared`` must provide names, cash, current, base, lower, upper, mandatory,
    categories, groups and group bounds. The full domain already retains every
    declared repair pin and mandatory interval. Nothing is inferred from a solve.
    Extra convex mandates beyond long-only funding and group bands are rejected.
    """
    table = problem.target
    names = tuple(table.index)
    if tuple(prepared["names"]) != names:
        raise ValueError("Prepared state labels differ from resolved state")
    current = table[s.CURRENT_WEIGHT].to_numpy(dtype=float)
    if not np.array_equal(current, np.asarray(prepared["current"])):
        raise ValueError("Common preparation changed original current holdings")
    policy = problem.constraints
    if not getattr(policy, "is_long_only", True):
        raise ValueError("Adapter supports long-only execution")
    for field in (
        "tracking_err_vol_constraint",
        "group_tracking_error_constraint",
        "turnover_constraint",
        "group_turnover_constraint",
        "max_target_portfolio_vol_an",
        "target_return",
        "benchmark_beta_constraint",
        "sector_deviation_constraints",
        "style_deviation_constraints",
        "linear_constraints",
    ):
        if getattr(policy, field, None) is not None:
            raise ValueError(f"Unsupported extra hard rule: {field}")
    groups = getattr(policy, "group_lower_upper_constraints", None)
    if groups is not None:
        labels = groups.group_loadings.columns
        originals = (
            groups.group_loadings.loc[list(names), labels].to_numpy().T,
            groups.group_min_allocation.loc[labels].to_numpy(),
            groups.group_max_allocation.loc[labels].to_numpy(),
        )
        for key, original in zip(("groups", "group_lower", "group_upper"), originals):
            if not np.array_equal(np.asarray(prepared[key]), original):
                raise ValueError("Common preparation changed an original group rule")
    full_lower, full_upper = np.asarray(full_lower), np.asarray(full_upper)
    fixed = full_lower == full_upper
    cash = int(prepared["cash"])
    fixed[cash] = False
    # Eligibility is the final declared domain, including mandatory exits and pins.
    # The caller has already applied calendars and original hard restrictions.
    args = {
        key: prepared[key]
        for key in (
            "base",
            "lower",
            "upper",
            "mandatory",
            "categories",
            "groups",
            "group_lower",
            "group_upper",
        )
    }
    return ExecutionProblem(
        names=names,
        cash=cash,
        current=current,
        target=table[s.RAW_MODEL_WEIGHT].to_numpy(dtype=float),
        full_lower=full_lower,
        full_upper=full_upper,
        eligible=~fixed,
        frozen=fixed,
        rules=rules,
        covariance=problem.covariance.loc[list(names), list(names)].to_numpy(),
        investment=prepared.get("investment", 1.0),
        provenance=provenance or {},
        preparation_seconds=preparation_seconds,
        **args,
    )
