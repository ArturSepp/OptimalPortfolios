"""Compile actual endpoint states and independently audit original-space delivery."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .types import ExecutionProblem, ExecutionResult


@dataclass(frozen=True, order=True)
class TradeState:
    """One closed endpoint interval and its instrument ticket classification."""

    lower: float
    upper: float
    ticket: bool


def domain_bounds(problem, formulation="CORRIDOR"):
    """Select an explicitly supplied domain, without deriving or widening corridors."""
    if formulation == "CORRIDOR":
        return problem.lower, problem.upper
    if formulation == "FULL":
        return problem.full_lower, problem.full_upper
    raise ValueError("Formulation must be FULL or CORRIDOR")


def floor_intervals(lower, upper, floor):
    """Intersect a closed long-only interval with zero or a valid held position."""
    if lower > upper:
        return []
    if floor == 0:
        return [(lower, upper)]
    result = [(0.0, 0.0)] if lower <= 0 <= upper else []
    if max(lower, floor) <= upper:
        result.append((max(lower, floor), upper))
    return result


def compile_states(
    problem: ExecutionProblem, formulation: str = "CORRIDOR"
) -> tuple[tuple[TradeState, ...], ...]:
    """Return mutually exclusive endpoint intervals, counting instruments once."""
    lower, upper = domain_bounds(problem, formulation)
    floor = np.broadcast_to(problem.rules.holding_floor, problem.current.shape)
    exempt = np.broadcast_to(problem.rules.exemption_mask, problem.current.shape)
    states = []
    for i, current in enumerate(problem.current):
        lo, hi = lower[i], upper[i]
        if i == problem.cash:
            states.append((TradeState(lo, hi, False),))
            continue
        branches = []
        base = problem.base[i]
        if not problem.mandatory[i] and lo <= base <= hi and (base == 0 or base >= floor[i]):
            branches.append(TradeState(base, base, False))
        if exempt[i]:
            intervals = [(lo, hi)]
        else:
            delta = problem.rules.minimum_ticket
            intervals = [(lo, min(hi, current - delta)), (max(lo, current + delta), hi)]
        for left, right in intervals:
            branches.extend(
                TradeState(a, b, True) for a, b in floor_intervals(left, right, floor[i])
            )
        states.append(tuple(sorted(set(branches))))
    return tuple(states)


def audit_execution(
    problem: ExecutionProblem, weights: np.ndarray, formulation: str = "CORRIDOR"
) -> dict:
    """Check original bounds, fractional groups, funding, floors and c-relative tickets.

    This audit intentionally does not consult compiled states or native statuses.
    """
    w = np.asarray(weights, dtype=float)
    if w.shape != problem.current.shape or not np.isfinite(w).all():
        return dict(passed=False, violations=["invalid_weights"])
    lo, hi = domain_bounds(problem, formulation)
    rules = problem.rules
    noncash = np.arange(len(w)) != problem.cash
    exempt = np.broadcast_to(rules.exemption_mask, w.shape)
    floor = np.broadcast_to(rules.holding_floor, w.shape)
    delta = np.abs(w - problem.current)
    traded = noncash & (delta > rules.zero_tolerance)
    optional = traded & ~problem.mandatory
    residuals = dict(
        bounds=max(0.0, float(np.max(lo - w)), float(np.max(w - hi))),
        groups=max(
            0.0,
            float(np.max(problem.group_lower - problem.groups @ w, initial=0)),
            float(np.max(problem.groups @ w - problem.group_upper, initial=0)),
        ),
        funding=abs(float(w.sum() - problem.investment)),
    )
    violations = [key for key, val in residuals.items() if val > rules.audit_tolerance]
    if np.any(noncash & (w > rules.zero_tolerance) & (w < floor - rules.audit_tolerance)):
        violations.append("holding_floor")
    if np.any(traded & ~exempt & (delta < rules.minimum_ticket - rules.audit_tolerance)):
        violations.append("minimum_ticket")
    if rules.ticket_cap is not None and np.sum(traded) > rules.ticket_cap:
        violations.append("total_ticket_cap")
    if (
        rules.discretionary_allowance is not None
        and np.sum(optional) > rules.discretionary_allowance
    ):
        violations.append("discretionary_allowance")
    return dict(
        passed=not violations,
        violations=violations,
        residuals=residuals,
        tickets=int(traded.sum()),
        mandatory_tickets=int((traded & problem.mandatory).sum()),
        discretionary_tickets=int(optional.sum()),
        subminimum_exempt_tickets=int((traded & exempt & (delta < rules.minimum_ticket)).sum()),
        gross_noncash_turnover=float(delta[noncash].sum()),
        category_tickets={
            category: int(sum(traded[i] for i, c in enumerate(problem.categories) if c == category))
            for category in sorted(set(problem.categories))
            if category
        },
    )


# QIS RiskModel supplies reporting analytics, but cannot define the solver's
# conditional quadratic objective or the independent original-space return gate.
def variance(problem, weights):
    """Evaluate the optimisation objective in unchanged annual covariance units."""
    active = np.asarray(weights) - problem.target
    return float(active @ problem.matvec(active))


def result(problem, weights, status, diagnostics, formulation="CORRIDOR"):
    """Construct delivery only through the common independent audit."""
    audit = audit_execution(problem, weights, formulation) if weights is not None else None
    if audit is not None and not audit["passed"]:
        raise ValueError("Refusing an unaudited execution result")
    value = None if weights is None else variance(problem, weights)
    states = compile_states(problem, formulation)
    candidates = sum(
        any(s.ticket for s in row)
        for i, row in enumerate(states)
        if i != problem.cash and not problem.mandatory[i]
    )
    details = dict(
        diagnostics,
        audit=audit,
        variance=value,
        annual_te_bp=None if value is None else 1e4 * np.sqrt(max(0.0, value)),
        M=problem.mandatory_count,
        K=problem.rules.ticket_cap,
        b=problem.rules.discretionary_allowance,
        candidate_instruments=candidates,
        cap_convention=(
            "mandatory_adjusted"
            if problem.rules.discretionary_allowance is not None
            else "total_or_unlimited"
        ),
        exemption_hash=problem.exemption_hash,
        preparation_seconds=problem.preparation_seconds,
        turnover_convention="gross noncash absolute weights minus original current",
        cap_utilisation=(
            None
            if audit is None or not problem.rules.ticket_cap
            else audit["tickets"] / problem.rules.ticket_cap
        ),
    )
    return ExecutionResult(weights, status, details)
