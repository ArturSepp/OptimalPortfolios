"""Independent capacity and lifecycle checks for strict baseline qualification."""
from dataclasses import replace
from types import SimpleNamespace

import cvxpy as cvx

import numpy as np
import pandas as pd
import pytest

from optimalportfolios.execution import schema as s
from optimalportfolios.execution._audited_cutoff_repair import _repair_audited_cutoff_target
from optimalportfolios.execution._constraint_scope import _original_group_bounds_scope
from optimalportfolios.execution.solver import (
    _positive_bridges, _snap_group_bounds_within_rounding_tolerance,
)

from optimalportfolios.optimization.constraints.analytics import evaluate_constraint_residuals
from optimalportfolios.execution.tests.cutoff_repair_test import _floor_problem as problem


def test_scope_preserves_original_floor_and_restores_legacy_behavior_after_error():
    """A four-millionth percent gap must not lower an explicitly fixed SAA floor."""
    constraints = problem().constraints
    bridges = pd.DataFrame(dict(bridge_min=[4e-7], bridge_max=[0.],
        e_max=[.05-4e-7], e_min=[0.]), index=['Alternatives'])
    before = constraints.group_lower_upper_constraints.group_min_allocation.copy()
    assert _snap_group_bounds_within_rounding_tolerance(constraints, bridges) is not constraints
    with pytest.raises(RuntimeError):
        with _original_group_bounds_scope():
            assert _snap_group_bounds_within_rounding_tolerance(constraints, bridges) is constraints
            raise RuntimeError('scope unwinds')
    assert _snap_group_bounds_within_rounding_tolerance(constraints, bridges) is not constraints
    pd.testing.assert_series_equal(
        constraints.group_lower_upper_constraints.group_min_allocation, before)


def test_linear_precheck_uses_common_audit_without_changing_ticket_zero_threshold():
    """A sub-audit bridge can reach sizing; an above-audit bridge remains rejected."""
    bridges = pd.DataFrame({'bridge_min': [9e-8, 1.2e-7], 'bridge_max': [0., 0.]},
                           index=['within_audit', 'outside_audit'])
    assert list(_positive_bridges(bridges).index) == ['within_audit', 'outside_audit']
    with _original_group_bounds_scope():
        assert list(_positive_bridges(bridges).index) == ['outside_audit']
        assert s.WEIGHT_ZERO_TOLERANCE == 1e-8
    assert list(_positive_bridges(bridges).index) == ['within_audit', 'outside_audit']


@pytest.mark.parametrize('gap', [4e-7, 8e-7, .003])
def test_support_consistent_endpoint_attains_independent_widening(gap):
    """Fractional AOR equations require exactly a Gold top-up of the known gap."""
    p = problem()
    table = p.target.copy()
    table.loc['Excluded', s.RULE4_TRADE] = True
    table.loc['Gold', [s.CURRENT_WEIGHT, s.BASE_WEIGHT, s.RAW_MODEL_WEIGHT,
        s.EFFECTIVE_MODEL_WEIGHT]] = .05-gap
    table.loc['Equity', [s.CURRENT_WEIGHT, s.BASE_WEIGHT]] = .5+gap
    table.loc['Excluded', s.RAW_MODEL_WEIGHT] = gap
    table.loc['Cash', s.EFFECTIVE_MODEL_WEIGHT] = .05+gap
    desired = table[s.EFFECTIVE_MODEL_WEIGHT]-table[s.BASE_WEIGHT]
    table[s.DESIRED_REWEIGHT] = desired
    table[s.MATERIAL_TRADE] = desired.abs().ge(.0025)
    table[s.TRADE_CANDIDATE] = (
        table[s.RULE4_TRADE] & table[s.MATERIAL_TRADE] & desired.abs().gt(1e-8))
    p = replace(p, target=table, optimiser_config=replace(p.optimiser_config, solver='CLARABEL'))
    before = p.target.copy(deep=True)
    result = _repair_audited_cutoff_target(p, ValueError('native audit rejected'))
    expected = pd.Series([.05, 0., .5, .3, .1, .05], index=table.index)
    np.testing.assert_allclose(result.result.weights, expected, atol=1e-7, rtol=0.)
    assert result.summary['minimum_widening'] == pytest.approx(gap, abs=1e-10)
    assert result.summary['actual_widening'] <= gap+1e-8+1e-9
    assert all(r.passed for r in evaluate_constraint_residuals(
        result.result.weights.to_numpy(), result.revised_problem.constraints,
        tolerance=1e-7) if r.hard)
    assert result.result.weights['Excluded'] == pytest.approx(0., abs=1e-8)
    assert result.result.weights['AOR'] == pytest.approx(.1, abs=1e-8)
    pd.testing.assert_frame_equal(p.target, before)


def test_infeasible_original_cap_is_not_relaxed():
    """A sub-5% Gold cap and an excluded alternative cannot meet a 5% floor."""
    p = problem()
    caps = p.constraints.max_weights.copy()
    caps.loc['Gold'] = .049
    p = replace(p, constraints=p.constraints.copy(max_weights=caps))
    with pytest.raises(ValueError, match='[Ii]nfeasible'):
        _repair_audited_cutoff_target(p, ValueError('native failed'))


def _funding_problem():
    """An independent 8e-8 cash deficit can be concentrated in one of ten ETFs."""
    p = problem()
    names = pd.Index([f'Asset{i}' for i in range(10)]+['Cash'])
    current = pd.Series([.089999992]*10+[.10000008], index=names)
    effective = pd.Series([.089999987]*10+[.10000013], index=names)
    cash = pd.Series(names == 'Cash', index=names)
    desired = effective-current
    table = pd.DataFrame({s.CURRENT_WEIGHT: current, s.BASE_WEIGHT: current,
        s.RAW_MODEL_WEIGHT: effective, s.EFFECTIVE_MODEL_WEIGHT: effective,
        s.DESIRED_REWEIGHT: desired, s.SETTLEMENT_CASH: cash,
        s.MANDATORY_TRADE: False, s.MANDATORY_FUNDING: 0., s.RULE4_TRADE: ~cash,
        s.REBALANCE_CADENCE_ELIGIBLE: True, s.MATERIAL_TRADE_THRESHOLD: .0025,
        s.MATERIAL_TRADE: False, s.TRADE_CANDIDATE: False,
        s.POLICY_MIN_WEIGHT: 0., s.POLICY_MAX_WEIGHT: .1,
        s.LOW_WEIGHT_EXCLUDED: False, s.PINNED_INSTRUCTION: False})
    groups = pd.DataFrame({'Risky': (~cash).astype(float), 'Liquidity': cash.astype(float)})
    bounds = type(p.constraints.group_lower_upper_constraints)(groups,
        group_min_allocation=pd.Series({'Risky': .8, 'Liquidity': 0.}),
        group_max_allocation=pd.Series({'Risky': .95, 'Liquidity': .1}))
    constraints = p.constraints.copy(min_weights=pd.Series(0., index=names),
        max_weights=pd.Series(.1, index=names), group_lower_upper_constraints=bounds)
    covariance = pd.DataFrame(np.diag([.04]*10+[.0001]), index=names, columns=names)
    p = replace(p, target=table, covariance=covariance, constraints=constraints,
        alphas=pd.Series(0., index=names), asset_classes=pd.Series('All', index=names),
        optimiser_config=replace(p.optimiser_config, solver='CLARABEL'))
    return p


def test_microscopic_funding_is_concentrated_without_freezing_all_buying_capacity():
    """An independent 8e-8 funding deficit requires a real admitted correction."""
    p = _funding_problem()
    result = _repair_audited_cutoff_target(p, ValueError('discarded microscopic buys'))
    assert result.summary['minimum_widening'] == pytest.approx(8e-8, abs=1e-11)
    assert result.summary['lp_witness_admissions'] >= 1
    assert result.endpoint['Cash'] <= .1+1e-10
    assert all(r.passed for r in evaluate_constraint_residuals(
        result.result.weights.to_numpy(), result.revised_problem.constraints,
        tolerance=1e-7) if r.hard)


def test_endpoint_fallback_is_audited_and_labelled_when_native_sizing_fails(monkeypatch):
    """A qualified endpoint supplies a valid baseline without disguising solver failure."""
    from optimalportfolios.execution import _audited_cutoff_repair as repair
    from optimalportfolios.execution.solver import ExecutionSolverInfeasibility

    def reject(*args):
        """Simulate an unavailable native projection without producing a portfolio."""
        raise ExecutionSolverInfeasibility('native projection deliberately unavailable')

    monkeypatch.setattr(repair, 'solve_ranked_execution', reject)
    result = _repair_audited_cutoff_target(problem(), ValueError('native rejected'))
    assert result.summary['feasible_endpoint_baseline']
    assert result.result.accepted and result.result.compliant
    assert result.result.weights['Excluded'] == 0.
    assert result.result.weights['AOR'] == .1
    assert result.result.weights['Gold'] >= .05-1e-7
    assert not result.result.trade_table['relaxed_corridor_solve'].any()


@pytest.mark.parametrize('tolerance', [-1., np.nan, np.inf, .001])
def test_numerical_allowance_cannot_become_a_material_corridor_relaxation(tolerance):
    """Only the declared small lexicographic allowance can enter the endpoint solve."""
    with pytest.raises(ValueError, match='widening_atol'):
        _repair_audited_cutoff_target(problem(), ValueError('native'), tolerance)


@pytest.mark.parametrize('defect', ['missing', 'negative', 'nonfinite'])
def test_endpoint_rejects_invalid_material_thresholds(defect):
    """Invalid policy inputs must fail before any auxiliary solver is invoked."""
    p = problem()
    if defect == 'missing':
        p.target.drop(columns=s.MATERIAL_TRADE_THRESHOLD, inplace=True)
    else:
        p.target.loc['Gold', s.MATERIAL_TRADE_THRESHOLD] = -.1 if defect == 'negative' else np.nan
    with pytest.raises(ValueError, match='threshold'):
        _repair_audited_cutoff_target(p, ValueError('native'))


def test_closed_domain_cannot_reopen_an_excluded_entry():
    """Failure to find a repair coordinate never changes lifecycle exclusions."""
    p = problem()
    p.target[s.RULE4_TRADE] = False
    p.target[s.TRADE_CANDIDATE] = False
    with pytest.raises(ValueError, match='no eligible'):
        _repair_audited_cutoff_target(p, ValueError('native'))


def test_tolerance_feasible_pinned_entry_is_labelled_without_claiming_an_optimum():
    """An exact pin 9e-8 below a floor passes the unchanged 1e-7 delivery audit."""
    p = problem()
    t = p.target.copy()
    current = pd.Series([0., 0., .5, .3, .1, .1], index=t.index)
    base = pd.Series([.05-9e-8, 0., .5, .3, .1, .05+9e-8], index=t.index)
    t[s.CURRENT_WEIGHT], t[s.BASE_WEIGHT] = current, base
    t[s.EFFECTIVE_MODEL_WEIGHT] = base
    t[s.DESIRED_REWEIGHT] = 0.
    t[s.MANDATORY_TRADE] = t.index == 'Gold'
    t[s.MANDATORY_CATEGORY] = ''
    t.loc['Gold', s.MANDATORY_CATEGORY] = 'model_entry'
    t[s.MANDATORY_FUNDING] = base-current
    t.loc['Gold', s.RULE4_TRADE] = False
    t[s.TRADE_CANDIDATE] = False
    t[s.MATERIAL_TRADE] = False
    p = replace(p, target=t)
    result = _repair_audited_cutoff_target(p, ValueError('exact pinned domain infeasible'))
    assert result.summary['provided_endpoint']
    assert result.summary['minimum_widening'] is None
    assert result.result.outcome.status == 'audited_feasible_endpoint'
    assert result.result.weights['Gold'] == base['Gold']
    assert (result.revised_problem.constraints.group_lower_upper_constraints
            .group_min_allocation['Alternatives'] == .05)
    assert result.result.accepted and result.result.compliant


@pytest.mark.parametrize('defect', ['nonfinite', 'group_breach'])
def test_provided_endpoint_cannot_bypass_raw_weight_and_hard_group_audits(defect):
    """A feasibility-only label never authorizes an invalid supplied vector."""
    from optimalportfolios.execution._audited_cutoff_repair import _provided_endpoint_baseline

    p = problem()
    p.target.loc['Gold', s.EFFECTIVE_MODEL_WEIGHT] = .05
    if defect == 'nonfinite':
        # Individually finite corrupted inputs can still overflow aggregate funding.
        p.target.loc[['Gold', 'Equity'], s.CURRENT_WEIGHT] = 1e308
    if defect == 'group_breach':
        p.target.loc['Equity', s.EFFECTIVE_MODEL_WEIGHT] = .5
        p.target.loc['Bonds', s.EFFECTIVE_MODEL_WEIGHT] = .31
    with pytest.raises(ValueError, match='nonfinite weights|original hard audit failed'):
        _provided_endpoint_baseline(p, ValueError('exact solve rejected'))


def test_failed_exact_lp_does_not_authorize_an_invalid_provided_endpoint():
    """An exact pinned-floor conflict must remain failed when the supplied point fails."""
    p = problem()
    p.target.loc['Gold', [s.BASE_WEIGHT, s.EFFECTIVE_MODEL_WEIGHT]] = .05-9e-8
    p.target.loc['Gold', s.MANDATORY_TRADE] = True
    p.target[s.MANDATORY_CATEGORY] = ''
    p.target.loc['Gold', s.MANDATORY_CATEGORY] = 'model_entry'
    p.target.loc['Gold', s.RULE4_TRADE] = False
    p.target.loc['Equity', s.EFFECTIVE_MODEL_WEIGHT] = .6
    with pytest.raises(ValueError, match='audited_minimum_widening: infeasible'):
        _repair_audited_cutoff_target(p, ValueError('exact native solve rejected'))


@pytest.mark.parametrize('linear,qp,failures,fallback,expected', [
    (True, False, ('SCIPY',), None, 'CLARABEL'),
    (False, True, ('CLARABEL',), 'MOSEK', 'MOSEK'),
    (False, True, ('CLARABEL', 'MOSEK'), 'MOSEK', 'both_endpoint_solvers_failed'),
    (False, True, ('CLARABEL',), None, 'solver_error'),
    (True, True, ('SCIPY',), None, 'solver_error'),
])
def test_backend_failures_are_bounded_and_never_treated_as_solver_success(
        linear, qp, failures, fallback, expected):
    """An unavailable backend can only use its declared retry or report failure."""
    from optimalportfolios.execution._audited_cutoff_repair import _endpoint_solve

    calls = []

    def solve(solver, **kwargs):
        """Record backend attempts and fail only the parametrized unavailable solvers."""
        calls.append(solver)
        if solver in failures:
            raise cvx.error.SolverError('deliberately unavailable backend')

    program = SimpleNamespace(solve=solve, is_qp=lambda: qp, status=cvx.OPTIMAL)
    if expected in ('CLARABEL', 'MOSEK'):
        assert _endpoint_solve(program, 'test_backend', linear, fallback) == expected
    else:
        with pytest.raises(ValueError, match=expected):
            _endpoint_solve(program, 'test_backend', linear, fallback)
    assert len(calls) <= 2


@pytest.mark.parametrize('defect', ['native_rejected', 'hard_breach', 'excess_widening'])
def test_bad_native_outputs_use_the_qualified_endpoint_instead(monkeypatch, defect):
    """A native acceptance flag cannot override bounds or the recorded widening budget."""
    from optimalportfolios.execution import _audited_cutoff_repair as repair

    def corrupt(revised):
        """Supply a rejected or invalid native result to exercise endpoint fallback."""
        weights = revised.target[s.EFFECTIVE_MODEL_WEIGHT].copy()
        if defect == 'hard_breach':
            weights.loc['Cash'] += .01
        elif defect == 'excess_widening':
            weights.loc['Gold'], weights.loc['Cash'] = .06, .04
        return SimpleNamespace(accepted=defect != 'native_rejected', compliant=True,
            weights=weights, outcome=SimpleNamespace(reason='deliberate invalid native output'))

    monkeypatch.setattr(repair, 'solve_ranked_execution', corrupt)
    result = _repair_audited_cutoff_target(problem(), ValueError('native rejected'))
    assert result.summary['feasible_endpoint_baseline']
    assert result.result.weights['Gold'] == pytest.approx(.05, abs=1e-7)
    assert result.result.accepted and result.result.compliant


def test_funding_failure_is_rejected_even_when_budget_and_group_audits_pass(monkeypatch):
    """Two individually small residuals cannot combine into an unfunded delivery."""
    from optimalportfolios.execution import _audited_cutoff_repair as repair

    p = problem()
    p.target.loc['Cash', [s.CURRENT_WEIGHT, s.BASE_WEIGHT]] -= 8e-8

    def unfunded(revised):
        """Introduce a cash residual that violates the unchanged funding identity."""
        weights = revised.target[s.EFFECTIVE_MODEL_WEIGHT].copy()
        weights.loc['Cash'] += 5e-8
        return SimpleNamespace(accepted=True, compliant=True, weights=weights)

    monkeypatch.setattr(repair, 'solve_ranked_execution', unfunded)
    with pytest.raises(ValueError, match='cash funding identity'):
        _repair_audited_cutoff_target(p, ValueError('native rejected'))


def test_admission_stall_stops_at_the_declared_refinement_limit(monkeypatch):
    """A numerically stalled backend cannot cause an unbounded sequence of retries."""
    from optimalportfolios.execution import _audited_cutoff_repair as repair

    p = _funding_problem()
    original = repair._endpoint_solve
    calls = []

    def stalled(program, stage, **kwargs):
        """Reset the endpoint to its base to simulate a stalled admission refinement."""
        solver = original(program, stage, **kwargs)
        if stage == 'audited_admission_endpoint':
            calls.append(stage)
            for variable in program.variables():
                if variable.name() == 'audited_endpoint_bp':
                    variable.value = 1e4*p.target[s.BASE_WEIGHT].to_numpy()
        return solver

    monkeypatch.setattr(repair, '_endpoint_solve', stalled)
    with pytest.raises(ValueError, match='refinement limit'):
        _repair_audited_cutoff_target(p, ValueError('backend stalls on microscopic activation'))
    assert len(calls) == 9
