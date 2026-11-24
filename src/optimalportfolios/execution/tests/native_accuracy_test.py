"""Qualify the public synthetic native-MIQP counterexample with an independent QP."""

import cvxpy as cp
import numpy as np
import pytest

from optimalportfolios.execution import ExecutionProblem, ExecutionRules
from optimalportfolios.execution import (
    audit_execution, compile_states, wrapper_exact_execution, wrapper_ranked_execution,
)
from optimalportfolios.execution.domain import variance
from optimalportfolios.execution.model import build_model


def public_counterexample(optional=24):
    """Reconstruct public joint_007/015 without importing paper or production modules.

    This frozen test fixture uses the public generator's seed and parameters;
    it contains no observations or private holdings. The old 1e-4 MIO setting
    missed the six-ticket witness even though its native model admits it.
    """
    n, cash = optional+3, optional+2
    rng = np.random.default_rng(np.random.SeedSequence([111001, optional]))
    relative = np.exp(rng.normal(0., .45, n-1))
    current = np.r_[.006+(.94-.006*(n-1))*relative/relative.sum(), .06]
    base = current.copy()
    base[cash] += base[:2].sum()
    base[:2] = 0.
    desired = current[2:cash]*np.exp(rng.normal(0., 1., optional))
    target = np.r_[0., 0., .98*desired/desired.sum(), .02]
    sector = np.zeros((3, n))
    sector[np.arange(n-1) % 3, np.arange(n-1)] = 1.
    baskets = np.c_[rng.uniform(.05, .95, (3, n-1)), np.zeros(3)]
    groups = np.vstack([sector, baskets])
    # Known synthetic PSD fixture, not an estimator duplicating an upstream API.
    loadings = np.c_[rng.uniform(.04, .12, n), .10*sector.T,
                     rng.normal(0., .04, (n, 2))]
    loadings[cash] = 0.
    covariance = loadings@loadings.T+np.diag(np.r_[rng.uniform(.04, .09, n-1)**2, .001**2])
    lower, upper = np.zeros(n), np.full(n, .35)
    upper[:2] = 0.
    lower[cash], upper[cash] = .005, .5
    lo, hi = np.minimum(base, target), np.minimum(np.maximum(base, target), upper)
    lo[cash], hi[cash] = lower[cash], upper[cash]
    exposure = groups@base
    return ExecutionProblem(names=tuple([f"Asset{i:03}" for i in range(n-1)]+["Cash"]),
        cash=cash, current=current, base=base, target=target, lower=lo, upper=hi,
        full_lower=lower, full_upper=upper, groups=groups,
        group_lower=np.maximum(0., exposure-.01), group_upper=exposure+.01,
        mandatory=np.arange(n) < 2, categories=("model_exit", "model_exit")+("",)*(n-2),
        eligible=np.ones(n, bool), frozen=np.zeros(n, bool), covariance=covariance,
        rules=ExecutionRules(minimum_ticket=.005, holding_floor=.0025,
            ticket_cap=8, discretionary_allowance=6))


# QIS RiskModel has no constrained CVXPY reference compiler; this test must
# reconstruct full-weight sizing independently of OP's funded-coordinate graph.
def independent_witness(problem):
    """Size the known feasible six-ticket support in full weights, without OP's graph.

    The reference enforces literal group limits, a subset of the audited domain.
    QIS RiskModel has no constrained CVXPY fixed-support reference compiler.
    """
    selected = {4, 7, 8, 12, 14, 24}
    lo, hi = problem.lower.copy(), problem.upper.copy()
    for i in range(problem.cash):
        if i not in selected:
            lo[i] = hi[i] = problem.base[i]
        elif problem.target[i] > problem.current[i]:
            lo[i] = max(lo[i], problem.current[i]+problem.rules.minimum_ticket)
        else:
            hi[i] = min(hi[i], problem.current[i]-problem.rules.minimum_ticket)
    weights_bp = cp.Variable(len(lo))
    rows = [cp.sum(weights_bp) == 1e4, weights_bp >= 1e4*lo, weights_bp <= 1e4*hi,
            problem.groups@weights_bp >= 1e4*problem.group_lower,
            problem.groups@weights_bp <= 1e4*problem.group_upper]
    objective = cp.quad_form(weights_bp-1e4*problem.target,
                             cp.psd_wrap(problem.covariance))/1e4
    reference = cp.Problem(cp.Minimize(objective), rows)
    reference.solve(solver="CLARABEL", tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10)
    assert reference.status == cp.OPTIMAL
    weights = np.asarray(weights_bp.value)/1e4
    assert audit_execution(problem, weights)["passed"]
    assert reference.value/1e4 == pytest.approx(variance(problem, weights), abs=1e-12)
    return weights


@pytest.mark.skipif(
    "MOSEK" not in cp.installed_solvers(), reason="Optional native backend unavailable"
)
def test_independent_native_solve_cannot_exclude_a_known_feasible_witness():
    """Recover the previously missed support without supplying its weights as a start."""
    problem = public_counterexample()
    witness = independent_witness(problem)
    solved = wrapper_exact_execution(problem, "CORRIDOR", time_limit_s=10, threads=1)
    if "license" in solved.diagnostics.get("error", "").lower():
        pytest.skip("Optional native licence unavailable")
    assert audit_execution(problem, solved.weights)["passed"]
    assert variance(problem, solved.weights) <= variance(problem, witness)+1e-12
    assert solved.diagnostics["native_bound"] <= variance(problem, witness)+1e-12
    assert solved.diagnostics["native_bound_check"]["consistent_with_delivery"] is True


@pytest.mark.skipif(
    "MOSEK" not in cp.installed_solvers(), reason="Optional native backend unavailable"
)
def test_balanced_integer_cone_does_not_exclude_audited_witness():
    """Tight MIO feasibility alone still misses this support with the old cone scaling."""
    problem = public_counterexample(72)
    witness = wrapper_ranked_execution(problem).weights
    assert audit_execution(problem, witness)["passed"]
    solved = wrapper_exact_execution(problem, "CORRIDOR", time_limit_s=10, threads=1)
    if "license" in solved.diagnostics.get("error", "").lower():
        pytest.skip("Optional native licence unavailable")
    assert audit_execution(problem, solved.weights)["passed"]
    assert variance(problem, solved.weights) <= variance(problem, witness)+1e-12
    assert solved.diagnostics["native_bound"] <= variance(problem, witness)+1e-12


@pytest.mark.skipif(
    "MOSEK" not in cp.installed_solvers(), reason="Optional native backend unavailable"
)
@pytest.mark.parametrize("formulation", ["CORRIDOR", "FULL"])
@pytest.mark.parametrize("perspective", [False, True])
def test_native_gap_uses_original_variance(formulation, perspective):
    """Check native units against funded weights for both integer formulations."""
    problem = public_counterexample()
    solved = wrapper_exact_execution(
        problem, formulation, perspective=perspective, time_limit_s=10, threads=1)
    if "license" in solved.diagnostics.get("error", "").lower():
        pytest.skip("Optional native licence unavailable")
    diagnostics = solved.diagnostics
    assert audit_execution(problem, solved.weights, formulation)["passed"]
    assert diagnostics["native_objective_origin"] == "original_variance"
    assert diagnostics["native_incumbent"] == pytest.approx(
        variance(problem, solved.weights), rel=0., abs=1e-12)
    assert diagnostics["native_bound"] == pytest.approx(
        diagnostics["native_bound_canonical"] + diagnostics["native_objective_offset"],
        rel=0., abs=1e-15)
    assert diagnostics["native_bound_canonical"] < 0 < diagnostics["native_incumbent"]
    # MOSEK's relative-gap denominator floor is 1e-10 in scaled objective units.
    expected = abs(diagnostics["native_incumbent"] - diagnostics["native_bound"]) / max(
        1e-14, abs(diagnostics["native_incumbent"]))
    assert diagnostics["native_gap_relative"] == pytest.approx(expected, rel=0., abs=1e-12)
    assert diagnostics["native_gap_relative"] <= diagnostics["gap_rel"]


@pytest.mark.parametrize("kind", ["integer", "continuous", "fixed"])
@pytest.mark.parametrize("perspective", [False, True])
def test_integer_cone_scaling_preserves_objective_and_other_graphs(monkeypatch, kind, perspective):
    """Balance only integer cones; verify the same quadratic in original funded units."""
    problem = public_counterexample()
    quadratic, observed = cp.quad_form, []

    def capture(vector, matrix):
        """Retain the conic quadratic input for an independent scaling check."""
        observed.append(vector)
        return quadratic(vector, matrix)

    monkeypatch.setattr(cp, "quad_form", capture)
    fixed = {i: min(1, len(row)-1) for i, row in enumerate(compile_states(problem)) if row}
    model = build_model(problem, perspective=perspective, continuous=kind == "continuous",
                        fixed_states=fixed if kind == "fixed" else None)
    x = np.linspace(1., 50., len(model.indices))
    model.displacement_bp.value = x
    scale = 100. if kind == "integer" else 1.
    np.testing.assert_array_equal(observed[0].value, x/scale)
    # QIS RiskModel does not expose CVXPY objective-coordinate identities.
    expected = x @ (model.funded_covariance-np.diag(model.split)) @ x / 1e4
    actual = quadratic(observed[0], cp.psd_wrap(
        model.funded_covariance-np.diag(model.split))).value
    assert actual/(1. if kind == "integer" else 1e4) == pytest.approx(expected, abs=1e-12)
