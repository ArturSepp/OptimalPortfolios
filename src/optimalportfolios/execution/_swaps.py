"""Deterministic funded-risk ordering for optional one-for-one support edits."""
import numpy as np
import pandas as pd
from dataclasses import replace

from optimalportfolios.execution import schema as s
from optimalportfolios.execution.improvement import _eligible_domain, _risk_gains
from optimalportfolios.execution.solver import _build_selected_execution_bound_series
from optimalportfolios.optimization.constraints import LinearConstraints


def _support(table):
    """Identify selected coordinates without treating tiny executed trades as deletions."""
    return tuple(table.index[table[s.SELECTED_TRADE].astype(bool)])


def _swap_candidates(problem, result, candidate_limit, canonical_ties):
    """Rank pairs by funded removal plus best incoming interval gain, without solving.

Reuse OP's existing optimization kernel, not a new risk reporting calculation.
This heuristic ignores joint resizing/group interactions; every admitted pair
still requires the complete strict projection and original return audit.
"""
    table, index = result.trade_table, problem.target.index
    eligible = _eligible_domain(table)
    selected = table[s.SELECTED_TRADE].astype(bool)
    outgoing = index[selected & eligible]
    incoming = index[~selected & eligible]
    if not len(outgoing) or not len(incoming):
        return []
    opened = table.copy(deep=True)
    opened[s.SELECTED_TRADE] = selected | eligible
    lower, upper = _build_selected_execution_bound_series(problem.constraints, opened, False)
    weights = result.weights.reindex(index).to_numpy()
    active = weights-problem.target[s.RAW_MODEL_WEIGHT].to_numpy()
    displacement = weights-problem.target[s.BASE_WEIGHT].to_numpy()
    cash = int(np.flatnonzero(table[s.SETTLEMENT_CASH].to_numpy())[0])
    covariance = problem.covariance.to_numpy()
    removal_gains, _ = _risk_gains(covariance, active, cash, -displacement)
    lo, hi = lower.to_numpy()-weights, upper.to_numpy()-weights
    positions = {name: i for i, name in enumerate(index)}
    def tie(name):
        """Keep exact score ties deterministic without permuting numerical inputs."""
        return str(name) if canonical_ties else positions[name]
    pairs = []
    for removed in outgoing:
        i = positions[removed]
        after = active.copy()
        after[i] -= displacement[i]
        after[cash] += displacement[i]
        gains, _ = _risk_gains(covariance, after, cash, np.zeros(len(index)), lo, hi)
        ranked = sorted(incoming, key=lambda name: (-gains[positions[name]], tie(name)))
        for added in ranked[:candidate_limit]:
            pairs.append((float(removal_gains[i]+gains[positions[added]]), removed, added))
    return sorted(pairs, key=lambda pair: (-pair[0], tie(pair[1]), tie(pair[2])))


def _exact_swap(parent, candidate, removed, added):
    """Require exactly the prescribed optional edit, preserving every protected row."""
    if not parent.index.equals(candidate.index):
        return False
    eligible = _eligible_domain(parent)
    if (removed not in parent.index or added not in parent.index or removed == added
            or not eligible.at[removed] or not eligible.at[added]
            or not parent.at[removed, s.SELECTED_TRADE] or parent.at[added, s.SELECTED_TRADE]):
        return False
    expected = parent[s.SELECTED_TRADE].copy()
    expected.at[removed], expected.at[added] = False, True
    return candidate[s.SELECTED_TRADE].equals(expected)


def _turnover_capped_problem(problem, table, reference, controls):
    """Append a conservative original-turnover row, preserving every existing linear policy.

Absolute change is linear on a directional interval. If an interval straddles
original current holdings, its secant upper envelope is conservative. Fixed
mandatory changes are constants. Settlement cash never enters the row. Reserve
1e-4 bp (one registered-zero weight tolerance) inside a positive original cap;
the independent return audit remains unchanged and authoritative.
"""
    if controls.max_turnover_increase_bp is None:
        return problem
    lower, upper = _build_selected_execution_bound_series(problem.constraints, table, False)
    current = problem.target[s.CURRENT_WEIGHT]
    width = upper-lower
    slope = pd.Series(0., index=table.index)
    moving = width.gt(0.)
    slope.loc[moving] = ((upper-current).abs()-(lower-current).abs()).loc[moving]/width.loc[moving]
    intercept = (lower-current).abs()-slope*lower
    cash = table[s.SETTLEMENT_CASH].astype(bool)
    slope.loc[cash], intercept.loc[cash] = 0., 0.
    cap = max(0., reference['gross_turnover_bp']+controls.max_turnover_increase_bp-1e-4)
    block = problem.constraints.linear_constraints
    name = '__execution_swap_turnover'
    while block is not None and name in block.loadings.columns:
        name += '_'
    loadings = (1e4*slope).to_frame(name)
    bound = pd.Series({name: cap-1e4*float(intercept.sum())})
    if block is not None:
        loadings = pd.concat([block.loadings.reindex(table.index), loadings], axis=1)
        lower_bounds = block.lower
        upper_bounds = pd.concat([block.upper, bound]) if block.upper is not None else bound
    else:
        lower_bounds, upper_bounds = None, bound
    policy = LinearConstraints(loadings, lower=lower_bounds, upper=upper_bounds)
    return replace(problem, constraints=problem.constraints.copy(linear_constraints=policy))
