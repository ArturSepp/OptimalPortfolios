"""Test tiny enumeration, independent KKT solutions and return-gate counterexamples."""

from dataclasses import replace
from itertools import combinations, product

import cvxpy as cp
import numpy as np
import pytest

from optimalportfolios.execution.v2 import (
    ExecutionProblem,
    ExecutionRules,
    RankedConfig,
    audit_execution,
    compile_states,
    wrapper_ranked_execution,
    wrapper_exact_execution,
    compute_perspective_lower_bound,
)
from optimalportfolios.execution.v2.domain import variance
from optimalportfolios.execution.v2.model import build_model, solve_qp
from optimalportfolios.execution.v2.ranked import ranked_selection, state_gain


def example(cap=2, coupled=True):
    """Create a complete 20%-NAV test block, independent of private ETF data."""
    rng = np.random.default_rng(1103)
    b = rng.normal(size=(5, 2)) * 0.12
    sigma = b @ b.T + np.diag([0.02, 0.015, 0.01, 0.025, 0.004])
    c = np.array([0.04, 0.03, 0.06, 0.03, 0.04])
    m = np.array([0.06, 0.015, 0.04, 0.05, 0.035])
    lo, hi = np.minimum(c, m), np.maximum(c, m)
    lo[-1], hi[-1] = 0.0, 0.1
    return ExecutionProblem(
        names=("a", "b", "c", "d", "cash"),
        cash=4,
        current=c,
        base=c,
        target=m,
        lower=lo,
        upper=hi,
        full_lower=np.zeros(5),
        full_upper=np.full(5, 0.1),
        groups=np.array([[1.0, 1.0, 0.0, 0.0, 0.0]]) if coupled else np.zeros((0, 5)),
        group_lower=np.array([0.05]) if coupled else np.zeros(0),
        group_upper=np.array([0.09]) if coupled else np.zeros(0),
        mandatory=np.zeros(5, bool),
        categories=("",) * 5,
        eligible=np.ones(5, bool),
        frozen=np.zeros(5, bool),
        rules=ExecutionRules(ticket_cap=cap, holding_floor=0.005),
        covariance=sigma,
        investment=0.2,
    )


def independent_kkt(p, fixed):
    """Solve a tiny QP by enumerating active sets without using the model compiler."""
    names = [i for i in fixed if fixed[i][0] != fixed[i][1]]
    r = p.base.copy()
    for i, (lo, hi) in fixed.items():
        if lo == hi:
            r[i] = lo
    r[p.cash] = p.investment - r[: p.cash].sum()
    u = np.zeros((len(r), len(names)))
    u[names, np.arange(len(names))] = 1
    u[p.cash] = -1
    lower, upper = p.lower.copy(), p.upper.copy()
    for i, interval in fixed.items():
        lower[i], upper[i] = interval
    a = np.vstack([u, -u, p.groups @ u, -p.groups @ u])
    rhs = np.r_[upper - r, r - lower, p.group_upper - p.groups @ r, p.groups @ r - p.group_lower]
    cov = u.T @ p.covariance @ u
    grad = u.T @ p.covariance @ (r - p.target)
    best = np.inf
    for size in range(len(names) + 1):
        for idx in combinations(range(len(rhs)), size):
            aa = a[list(idx)]
            kkt = np.block([[2 * cov, aa.T], [aa, np.zeros((size, size))]])
            try:
                solution = np.linalg.solve(kkt, np.r_[-2 * grad, rhs[list(idx)]])
            except np.linalg.LinAlgError:
                continue
            x = solution[: len(names)]
            if np.max(a @ x - rhs) < 1e-8 and np.min(solution[len(names) :], initial=0) >= -1e-8:
                active = r + u @ x - p.target
                best = min(best, float(active @ p.covariance @ active))
    return best


@pytest.mark.parametrize("cap", [0, 1, 2, 4, None])
def test_enumerated_fixed_qps_and_bounds(cap):
    """Compare compiled QPs to independent active sets and both relaxations."""
    p = example(cap)
    states = compile_states(p)
    best = np.inf
    for choice in product(*[range(len(row)) for row in states[:-1]]):
        if cap is not None and sum(states[i][j].ticket for i, j in enumerate(choice)) > cap:
            continue
        fixed = dict(enumerate(choice))
        model = build_model(p, fixed_states=fixed)
        w = solve_qp(model)
        reference = independent_kkt(
            p, {i: (states[i][j].lower, states[i][j].upper) for i, j in fixed.items()}
        )
        if w is not None:
            assert audit_execution(p, w)["passed"]
            assert abs(variance(p, w) - reference) < 2e-9
            assert abs(model.optimisation.value / 1e4 - variance(p, w)) < 2e-9
            best = min(best, variance(p, w))
    ranked = wrapper_ranked_execution(p)
    assert ranked.weights is not None
    assert ranked.diagnostics["variance"] <= variance(p, p.base) + 1e-12
    bounds = compute_perspective_lower_bound(p)
    for bound in bounds.values():
        assert bound["lower_bound"] <= best + 1e-10
        assert bound["primal"] <= best + 1e-9
        assert "certificate_error" not in bound
    assert bounds["plain"]["primal"] <= bounds["perspective"]["primal"] + 1e-9


def test_floors_exits_and_minimum_ticket():
    """A target below a floor cannot authorise an out-of-corridor zero exit."""
    p = example()
    c, m = p.current.copy(), p.target.copy()
    c[0], c[-1], m[0] = 0.008, 0.072, 0.003
    lo, hi = p.lower.copy(), p.upper.copy()
    lo[0], hi[0] = 0.003, 0.008
    p = replace(p, current=c, base=c, target=m, lower=lo, upper=hi)
    assert all(s.lower >= 0.005 for s in compile_states(p)[0])
    lo[0] = 0.0
    p = replace(p, lower=lo)
    assert any(s.lower == s.upper == 0 and s.ticket for s in compile_states(p)[0])
    w = p.base.copy()
    w[1] -= 0.001
    w[-1] += 0.001
    assert "minimum_ticket" in audit_execution(p, w)["violations"]


def test_immutable_inputs_and_rejected_approximations():
    """Inputs cannot mutate and an approximate diagonal residual is rejected."""
    p = example()
    with pytest.raises(ValueError):
        p.current.flags.writeable = True
    with pytest.raises(ValueError, match="exact representation"):
        replace(
            p,
            factor_loadings=np.ones((5, 1)),
            factor_covariance=np.ones((1, 1)),
            residual_variance=np.ones(5),
        )
    with pytest.raises(ValueError, match="exemption"):
        replace(p, rules=ExecutionRules(exemption_mask=True))
    with pytest.raises(ValueError):
        replace(p, covariance=-np.eye(5))


def test_mandatory_subminimum_counts_and_floors():
    """A 10bp required move is exempt in size and still consumes one total ticket."""
    p = example()
    q, lo, hi = p.base.copy(), p.lower.copy(), p.upper.copy()
    q[0] += 0.001
    q[-1] -= 0.001
    lo[0], hi[0] = q[0], 0.06
    fl, fu = p.full_lower.copy(), p.full_upper.copy()
    fl[0], fu[0] = lo[0], hi[0]
    mask = np.array([True, False, False, False, False])
    p = replace(
        p,
        base=q,
        lower=lo,
        upper=hi,
        full_lower=fl,
        full_upper=fu,
        mandatory=mask,
        categories=("product_min_buy", "", "", "", ""),
        rules=ExecutionRules(ticket_cap=1, exemption_mask=mask, discretionary_allowance=0),
    )
    assert audit_execution(p, q)["tickets"] == 1
    assert wrapper_ranked_execution(p).weights is not None
    zero = replace(p, rules=ExecutionRules(ticket_cap=0, exemption_mask=mask))
    assert not audit_execution(zero, q)["passed"]
    assert wrapper_ranked_execution(zero).weights is None


def test_rank_updates_scalar_scores_and_structured_equivalence():
    """Scalar clipping and dense recomputation independently verify incremental selection."""
    p = example()
    selected, working, trace, _ = ranked_selection(p, compile_states(p), 4)
    assert selected and trace
    previous = p.base.copy()
    for event in trace:
        i = p.names.index(event["instrument"])
        old = variance(p, previous)
        previous[i] += event["step"]
        previous[p.cash] -= event["step"]
        assert abs(old - variance(p, previous) - event["gain"]) < 1e-15
    assert np.array_equal(previous, working)
    assert state_gain(1.0, 0.0, -0.01, 0.02) == (0.02, -0.01)
    eig, vec = np.linalg.eigh(p.covariance)
    structured = replace(
        p,
        covariance=None,
        factor_loadings=vec,
        factor_covariance=np.diag(eig),
        residual_variance=np.zeros(5),
    )
    assert np.allclose(structured.matvec(p.current), p.matvec(p.current), atol=1e-15)
    assert ranked_selection(structured, compile_states(p), 4)[0] == selected


@pytest.mark.parametrize("perspective", [False, True])
def test_native_start_receipt_and_full_containment(perspective):
    """Exercise the real optional solver path when MOSEK is available and licensed."""
    p = example(2)
    if "MOSEK" not in cp.installed_solvers():
        pytest.skip("Optional MOSEK backend unavailable")
    result = wrapper_exact_execution(
        p, "CORRIDOR", perspective=perspective, time_limit_s=20, threads=1
    )
    if "license" in result.diagnostics.get("error", "").lower():
        pytest.skip("Optional MOSEK licence unavailable")
    assert result.diagnostics.get("error") is None
    assert result.diagnostics["initial_start"]["supplied"] is True
    assert result.diagnostics["initial_start"]["received"] is True
    assert audit_execution(p, result.weights, "FULL")["passed"]
    full = wrapper_exact_execution(p, "FULL", perspective=perspective, time_limit_s=20, threads=1)
    assert full.diagnostics.get("error") is None
    assert full.diagnostics["variance"] <= result.diagnostics["variance"] + 1e-8


def test_zero_objective_unavailable_solver_and_preserved_incumbent():
    """No backend and a zero covariance both retain a valid prepared baseline."""
    p = replace(example(), covariance=np.zeros((5, 5)))
    ranked = wrapper_ranked_execution(p, RankedConfig(compute_bound=True))
    assert ranked.diagnostics["variance"] == 0
    unavailable = wrapper_exact_execution(p, "CORRIDOR", solver="UNAVAILABLE")
    assert unavailable.diagnostics["native_status"] == "solver_unavailable"
    assert np.array_equal(unavailable.weights, p.base)


def test_no_ticket_keeps_frozen_base_endpoint():
    """An optional sub-ZERO q-c residual is retained without snapping inputs."""
    p = example(0)
    q = p.base.copy()
    q[0] += 5e-9
    q[-1] -= 5e-9
    p = replace(p, base=q)
    state = [s for s in compile_states(p)[0] if not s.ticket][0]
    assert state.lower == q[0]
    out = wrapper_ranked_execution(p)
    assert out.weights[0] == q[0]
