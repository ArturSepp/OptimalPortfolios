"""Independent feasible-endpoint examples for the opt-in cutoff repair."""
from dataclasses import replace
from types import SimpleNamespace

import cvxpy as cvx

import numpy as np
import pandas as pd
import pytest

from optimalportfolios import Constraints, GroupLowerUpperConstraints, OptimiserConfig
from optimalportfolios.execution import ExecutionRankingConfig, ResolvedExecutionProblem
from optimalportfolios.execution import schema as s
from optimalportfolios.execution import _cutoff_repair as repair
from optimalportfolios.execution._cutoff_repair import _repair_cutoff_target


def _floor_problem(gap=.003):
    """A blocked small gold entry requires a known top-up in an existing ETF."""
    names = pd.Index(['Gold', 'Excluded', 'Equity', 'Bonds', 'AOR', 'Cash'])
    current = pd.Series([.05-gap, 0., .5+gap, .3, .1, .05], index=names)
    raw = pd.Series([.05-gap, gap, .5, .3, .1, .05], index=names)
    effective = raw.copy()
    effective.loc['Excluded'] = 0.
    effective.loc['Cash'] += gap
    cash = pd.Series(names == 'Cash', index=names)
    rule4 = pd.Series([True, False, True, True, False, False], index=names)
    desired = effective-current
    table = pd.DataFrame({s.CURRENT_WEIGHT: current, s.BASE_WEIGHT: current,
        s.RAW_MODEL_WEIGHT: raw, s.EFFECTIVE_MODEL_WEIGHT: effective,
        s.DESIRED_REWEIGHT: desired, s.SETTLEMENT_CASH: cash,
        s.MANDATORY_TRADE: False, s.MANDATORY_FUNDING: 0., s.RULE4_TRADE: rule4,
        s.REBALANCE_CADENCE_ELIGIBLE: True, s.MATERIAL_TRADE_THRESHOLD: .0025,
        s.MATERIAL_TRADE: desired.abs().ge(.0025),
        s.TRADE_CANDIDATE: rule4 & desired.abs().ge(.0025),
        s.POLICY_MIN_WEIGHT: 0., s.POLICY_MAX_WEIGHT: 1.,
        s.LOW_WEIGHT_EXCLUDED: names == 'Excluded', s.PINNED_INSTRUCTION: names == 'AOR'})
    table.loc['AOR', [s.POLICY_MIN_WEIGHT, s.POLICY_MAX_WEIGHT]] = .1
    loadings = pd.DataFrame({'Alternatives': [1., 1., 0., 0., 0., 0.],
        'Equity': [0., 0., 1., 0., .6, 0.],
        'Fixed Income': [0., 0., 0., 1., .4, 0.]}, index=names)
    constraints = Constraints(min_weights=pd.Series(0., index=names),
        max_weights=pd.Series([.1, .1, .7, .5, .1, .1], index=names),
        group_lower_upper_constraints=GroupLowerUpperConstraints(loadings,
            group_min_allocation=pd.Series(
                {'Alternatives': .05, 'Equity': .56, 'Fixed Income': .34}),
            group_max_allocation=pd.Series(
                {'Alternatives': .15, 'Equity': .56, 'Fixed Income': .34})))
    covariance = pd.DataFrame(np.diag([.04, .04, .09, .01, .04, .0001]), index=names, columns=names)
    return ResolvedExecutionProblem(target=table, covariance=covariance,
        alphas=pd.Series(0., index=names), asset_classes=pd.Series('All', index=names),
        constraints=constraints, ranking_config=ExecutionRankingConfig({'All': 1.}, max_trades=1),
        optimiser_config=OptimiserConfig(solver='CLARABEL'), expand_for_feasibility=True)


@pytest.mark.parametrize('gap', [.003, .001])
def test_minimum_widening_recovers_floor_with_fractional_groups(gap):
    """Group equations give an independent exact portfolio and widening bound."""
    problem = _floor_problem(gap)
    before = problem.target.copy(deep=True)
    result = _repair_cutoff_target(problem)
    # Fixed AOR=10% implies Equity=50%, Bonds=30%. Gold must be >=5%;
    # its old endpoint is 5%-gap, hence widening >=gap, attained below.
    expected = pd.Series([.05, 0., .5, .3, .1, .05], index=before.index)
    np.testing.assert_allclose(result.result.weights, expected, atol=2e-6, rtol=0.)
    assert result.summary['minimum_widening'] == pytest.approx(gap, abs=2e-7)
    assert result.summary['actual_widening'] == pytest.approx(gap, abs=2e-7)
    assert result.result.accepted and result.result.compliant
    assert result.result.weights['Excluded'] == pytest.approx(0., abs=1e-8)
    assert result.result.weights['AOR'] == pytest.approx(.1, abs=1e-9)
    pd.testing.assert_frame_equal(problem.target, before, check_exact=True)
    pd.testing.assert_series_equal(result.revised_problem.target[s.RAW_MODEL_WEIGHT],
                                   before[s.RAW_MODEL_WEIGHT], check_exact=True)
    if gap < .0025:
        assert not result.result.trade_table.loc['Gold', s.TRADE_CANDIDATE]
        assert result.result.trade_table.loc['Gold', s.FEASIBILITY_RESCUE_TRADE]


def test_repair_preserves_entry_exclusion_even_with_inconsistent_rule4_flag():
    """Descriptive eligibility must never override an explicit entry exclusion."""
    problem = _floor_problem()
    problem.target.loc['Excluded', s.RULE4_TRADE] = True
    result = _repair_cutoff_target(problem)
    assert result.result.weights['Excluded'] == pytest.approx(0., abs=1e-8)
    assert not result.audit.loc['Excluded', 'repair_eligible']


@pytest.mark.parametrize('restriction', ['cadence', 'hard_cap', 'desk_pin'])
def test_unreachable_floor_is_not_repaired_by_breaking_hard_restrictions(restriction):
    """Gold below 5% and a blocked entry cannot jointly satisfy the 5% floor."""
    problem = _floor_problem()
    if restriction == 'cadence':
        problem.target.loc['Gold', s.REBALANCE_CADENCE_ELIGIBLE] = False
    elif restriction == 'desk_pin':
        problem.target.loc['Gold', [s.POLICY_MIN_WEIGHT, s.POLICY_MAX_WEIGHT]] = .047
    else:
        maximum = problem.constraints.max_weights.copy()
        maximum.loc['Gold'] = .049
        problem = replace(problem, constraints=problem.constraints.copy(max_weights=maximum))
    with pytest.raises(ValueError, match='Infeasible|infeasible'):
        _repair_cutoff_target(problem)


def test_repair_rejects_nonfinite_or_material_lexicographic_tolerance():
    """The widening tolerance is numerical, never an implicit allocation allowance."""
    for tolerance in [-1., np.nan, np.inf, .001]:
        with pytest.raises(ValueError, match='widening_atol'):
            _repair_cutoff_target(_floor_problem(), tolerance)

@pytest.mark.parametrize('defect', ['missing', 'negative', 'nonfinite'])
def test_repair_requires_available_valid_material_thresholds(defect):
    """Invalid resolved thresholds cannot enter a corridor-expansion calculation."""
    problem = _floor_problem()
    if defect == 'missing':
        problem.target.drop(columns=s.MATERIAL_TRADE_THRESHOLD, inplace=True)
    else:
        problem.target.loc['Gold', s.MATERIAL_TRADE_THRESHOLD] = (
            -.01 if defect == 'negative' else np.nan)
    before = problem.target.copy(deep=True)
    with pytest.raises(ValueError, match='material_trade_threshold'):
        _repair_cutoff_target(problem)
    pd.testing.assert_frame_equal(problem.target, before, check_exact=True)


def test_repair_without_eligible_coordinates_reports_its_domain_failure():
    """A closed repair domain never authorizes reopening an excluded instrument."""
    problem = _floor_problem()
    problem.target[s.RULE4_TRADE] = False
    problem.target[s.TRADE_CANDIDATE] = False
    with pytest.raises(repair._CutoffRepairFailure) as failed:
        _repair_cutoff_target(problem)
    assert failed.value.stage == 'domain'
    assert failed.value.status == 'no eligible repair coordinates'


@pytest.mark.parametrize('failure', ['solver_error', 'optimal_inaccurate'])
def test_solver_failure_preserves_the_unsuccessful_stage(failure):
    """Neither an exception nor an inaccurate solve supplies a repair endpoint."""
    options = {}
    def solve(**kwargs):
        """Capture cold solver options and simulate the selected failed solve."""
        options.update(kwargs)
        if failure == 'solver_error':
            raise cvx.error.SolverError('synthetic numerical failure')
    program = SimpleNamespace(solve=solve, status=cvx.OPTIMAL_INACCURATE)
    with pytest.raises(repair._CutoffRepairFailure) as failed:
        repair._solve_stage(program, _floor_problem(), 'endpoint_risk')
    assert failed.value.stage == 'endpoint_risk' and failed.value.status == failure
    assert options['warm_start'] is False and options['max_iter'] == 1000
    if failure == 'solver_error':
        assert isinstance(failed.value.__cause__, cvx.error.SolverError)


def test_rejected_endpoint_audit_stops_before_ranked_projection(monkeypatch):
    """An optimal status cannot bypass an independent endpoint compliance audit."""
    def rejected(**kwargs):
        """Return an explicit independent-audit rejection."""
        return SimpleNamespace(accepted=False, compliant=False, reason='endpoint rejected')
    monkeypatch.setattr(repair, 'validate_solution', rejected)
    with pytest.raises(repair._CutoffRepairFailure) as failed:
        _repair_cutoff_target(_floor_problem())
    assert (failed.value.stage, failed.value.status) == ('endpoint_audit', 'endpoint rejected')


def test_endpoint_hard_compliance_does_not_override_widening_budget(monkeypatch):
    """Extra gold funded from cash satisfies the mandate but exceeds minimum widening."""
    original = repair._solve_stage
    def excess_endpoint(program, problem, stage):
        """Simulate a solver violating only the auxiliary expansion constraint."""
        original(program, problem, stage)
        if stage == 'endpoint_risk':
            variable = program.variables()[0]
            weights = variable.value.copy()
            weights[problem.target.index.get_loc('Gold')] += .002
            weights[problem.target.index.get_loc('Cash')] -= .002
            variable.value = weights
    monkeypatch.setattr(repair, '_solve_stage', excess_endpoint)
    with pytest.raises(repair._CutoffRepairFailure) as failed:
        _repair_cutoff_target(_floor_problem())
    assert failed.value.stage == 'endpoint_audit'
    assert failed.value.status == 'widening budget exceeded'


def test_rejected_ranked_projection_has_no_repair_result(monkeypatch):
    """A feasible endpoint alone cannot authorize the final execution holdings."""
    def rejected(problem):
        """Return a typed-stage failure instead of an accepted execution."""
        return SimpleNamespace(accepted=False, compliant=False,
                               outcome=SimpleNamespace(reason='projection rejected'))
    monkeypatch.setattr(repair, 'solve_ranked_execution', rejected)
    with pytest.raises(repair._CutoffRepairFailure) as failed:
        _repair_cutoff_target(_floor_problem())
    assert (failed.value.stage, failed.value.status) == ('ranked_projection', 'projection rejected')


@pytest.mark.parametrize('defect', ['hard_bounds', 'widening_budget'])
def test_final_execution_is_reaudited_against_original_limits(monkeypatch, defect):
    """Nominal acceptance cannot conceal a floor violation or excess aggregate expansion."""
    original = repair.solve_ranked_execution
    def changed_holdings(problem):
        """Keep an accepted solver report but corrupt its returned holdings."""
        result = original(problem)
        weights = result.weights.copy(deep=True)
        amount = -.002 if defect == 'hard_bounds' else .002
        weights.loc['Gold'] += amount
        weights.loc['Cash'] -= amount
        return replace(result, weights=weights)
    monkeypatch.setattr(repair, 'solve_ranked_execution', changed_holdings)
    with pytest.raises(repair._CutoffRepairFailure) as failed:
        _repair_cutoff_target(_floor_problem())
    assert failed.value.stage == 'final_audit'
    if defect == 'widening_budget':
        assert failed.value.status == 'widening budget exceeded'
    else:
        assert failed.value.status
