"""Independent feasible-endpoint examples for the opt-in cutoff repair."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from optimalportfolios import Constraints, GroupLowerUpperConstraints, OptimiserConfig
from optimalportfolios.execution import ExecutionRankingConfig, ResolvedExecutionProblem
from optimalportfolios.execution import schema as s
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
