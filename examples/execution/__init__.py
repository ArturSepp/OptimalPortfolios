"""Synthetic supplied inputs and audited reporting for the execution examples.

Run the sibling modules from the repository root. No private data, downloads,
calibration, compulsory-repair inference or file writes are performed here.
"""

import numpy as np
import pandas as pd

from optimalportfolios.execution import ExecutionProblem, ExecutionRules, audit_execution


def portfolio_problem(ticket_cap=3, cost=0.0):
    """Build a feasible seven-name CORRIDOR example with overlapping group rows.

    Covariance is a supplied diagonal annual covariance, not an estimated risk
    model. The cost coefficient is annual variance per unit traded NAV, not bps.
    Current holdings already satisfy the hard domain, so base equals current.
    """
    current = np.array([.22, .16, .12, .18, .14, .13, .05])
    target = np.array([.16, .22, .10, .14, .18, .15, .05])
    full_lower = np.array([0., 0., 0., 0., 0., 0., .049])
    full_upper = np.array([.30, .30, .30, .30, .30, .20, .051])
    # Prepared CORRIDOR bounds are explicit inputs; the wrapper does not infer them.
    lower, upper = np.minimum(current, target), np.maximum(current, target)
    lower[-1], upper[-1] = full_lower[-1], full_upper[-1]
    problem = ExecutionProblem(
        names=("Equity_A", "Equity_B", "Equity_C", "Bond_A", "Bond_B", "Diversifier", "Cash"),
        cash=6, current=current, base=current, target=target,
        lower=lower, upper=upper, full_lower=full_lower, full_upper=full_upper,
        groups=np.array([[1, 1, 1, 0, 0, 0, 0], [0, 0, 0, 1, 1, 0, 0],
                         [1, 1, 1, 0, 0, 1, 0]], dtype=float),
        group_lower=np.array([.42, .28, .55]),
        group_upper=np.array([.55, .40, .70]),
        mandatory=np.zeros(7, dtype=bool), categories=("",) * 7,
        eligible=np.ones(7, dtype=bool), frozen=np.zeros(7, dtype=bool),
        rules=ExecutionRules(minimum_ticket=.02, holding_floor=.01, ticket_cap=ticket_cap),
        covariance=np.diag(np.square([.20, .18, .16, .06, .08, .12, 0.])),
        trade_cost=np.array([cost] * 6 + [0.]),
        provenance={"source": "deterministic synthetic teaching inputs"},
    )
    assert audit_execution(problem, problem.base)["passed"]
    return problem


def pair_problem(ticket_cap):
    """Supply a two-name switch for which tight cash rules out one-sided trades."""
    current = np.array([.50, .45, .05])
    lower, upper = np.array([.45, .45, .049]), np.array([.50, .50, .051])
    return ExecutionProblem(
        names=("Seller", "Buyer", "Cash"), cash=2, current=current, base=current,
        target=np.array([.45, .50, .05]), lower=lower, upper=upper,
        full_lower=np.array([0., 0., .049]), full_upper=np.array([.80, .80, .051]),
        groups=np.array([[1., 1., 0.]]),
        group_lower=np.array([.949]), group_upper=np.array([.951]),
        mandatory=np.zeros(3, dtype=bool), categories=("",) * 3,
        eligible=np.ones(3, dtype=bool), frozen=np.zeros(3, dtype=bool),
        rules=ExecutionRules(minimum_ticket=.02, holding_floor=.01, ticket_cap=ticket_cap),
        covariance=np.diag([.04, .04, 0.]),
    )


def delivery_row(method, problem, solved, seconds):
    """Independently audit delivery and read risk and objective from OP diagnostics."""
    audit = audit_execution(problem, solved.weights)
    if not audit["passed"]:
        raise RuntimeError(f"{method} delivered no valid portfolio: {audit}")
    info = solved.diagnostics
    return dict(
        method=method, status=solved.status, audited=True,
        objective=info["scored_objective"], annual_te_bp=info["annual_te_bp"],
        cost_term=info["trade_cost_value"], tickets=audit["tickets"],
        turnover_pct=100 * audit["gross_noncash_turnover"], elapsed_s=seconds,
    )


def trade_table(problem, solved):
    """Return NAV percentages, including cash, without rounding solver inputs."""
    return pd.DataFrame({
        "current_pct": 100 * problem.current, "target_pct": 100 * problem.target,
        "execution_pct": 100 * solved.weights,
        "trade_pct": 100 * (solved.weights - problem.current),
    }, index=problem.names)


def native_evidence(method, solved):
    """Expose native failure and candidate evidence without certifying optimality.

    An audited prepared incumbent is a valid portfolio but does not prove that
    a native solve ran. Kernel's bounds apply only to its restricted subproblems.
    """
    info = solved.diagnostics
    if method == "Exact":
        records = [info]
        stages = "full corridor domain"
    else:
        subproblems = info.get("subproblems", [])
        records = [stage.get("native", stage) for stage in subproblems]
        stages = f"{info.get('completed_buckets', 0)}/{info.get('total_buckets', 0)} buckets"
    statuses = [str(record.get("native_status", "error")) for record in records]
    errors = [str(record["error"]) for record in records if record.get("error")]
    witness = any(
        event.get("audit", {}).get("passed", False)
        and event.get("source", "").startswith("native")
        for event in info.get("incumbent_events", [])
    )
    bad = {"solver_error", "solver_unavailable", "not_run", "preparation_deadline", "error"}
    usable = (bool(records) and witness and not errors and not bad.intersection(statuses)
              and not any(record.get("callback_errors") for record in records)
              and (info.get("native_bound_check") or {}).get("consistent_with_delivery")
              is not False)
    return dict(native_candidate=witness, native_status=";".join(statuses),
                native_usable=usable, stages=stages, errors=";".join(errors))
