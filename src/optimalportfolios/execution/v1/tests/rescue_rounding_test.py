"""Rescue and direct sizing must apply the same group-rounding contract."""
from contextlib import nullcontext

import numpy as np
import pandas as pd
import pytest

from optimalportfolios import Constraints, GroupLowerUpperConstraints, OptimiserConfig
from optimalportfolios.execution.v1 import schema as s
from optimalportfolios.execution.v1._constraint_scope import _original_group_bounds_scope
from optimalportfolios.execution.v1.solver import (
    ExecutionSolverInfeasibility,
    GROUP_BOUND_ROUNDING_TOLERANCE,
    solve_feasible_execution_portfolio,
    solve_selected_execution_portfolio,
)


def _problem(direction, residual, *, cash_bound=False, target_bands=False):
    """Require a rescue that leaves a known signed interval gap at its target."""
    assets = pd.Index(['Pinned', 'Rescue', 'Cash'])
    if direction == 'min':
        current = pd.Series([.4, .2, .4], index=assets)
        base = pd.Series([.3, .2, .5], index=assets)
        model = pd.Series([.3, .3-residual, .4+residual], index=assets)
    else:
        current = pd.Series([.3, .3, .4], index=assets)
        base = pd.Series([.4, .3, .3], index=assets)
        model = pd.Series([.4, .2+residual, .4-residual], index=assets)
    table = pd.DataFrame({
        s.CURRENT_WEIGHT: current, s.BASE_WEIGHT: base,
        s.RAW_MODEL_WEIGHT: model, s.EFFECTIVE_MODEL_WEIGHT: model,
        s.DESIRED_REWEIGHT: (model-base).where([False, True, False], 0.),
        s.MANDATORY_FUNDING: base-current,
        s.MANDATORY_TRADE: [True, False, True],
        s.RULE4_TRADE: [False, True, False],
        s.POLICY_MIN_WEIGHT: 0., s.POLICY_MAX_WEIGHT: 1.,
        s.REBALANCE_CADENCE_ELIGIBLE: True,
        s.SETTLEMENT_CASH: [False, False, True], s.SELECTED_TRADE: False,
        s.TRADE_SCORE: [0., 1., 0.], s.SCORE_PER_TURNOVER: [0., 1., 0.],
    }, index=assets)
    if target_bands:
        table[s.TARGET_BAND_LOWER] = pd.concat([base, model], axis=1).min(axis=1)
        table[s.TARGET_BAND_UPPER] = pd.concat([base, model], axis=1).max(axis=1)
        table[s.MATERIAL_TRADE_THRESHOLD] = 0.
        table[s.MODEL_ENTRY] = False
    lower, upper = pd.Series(0., index=assets), pd.Series(1., index=assets)
    if cash_bound:
        (upper if direction == 'min' else lower).loc['Cash'] = .4
    group = None if cash_bound else GroupLowerUpperConstraints(
        group_loadings=pd.DataFrame({'Risk': [1., 1., 0.]}, index=assets),
        group_min_allocation=pd.Series({'Risk': .6 if direction == 'min' else 0.}),
        group_max_allocation=pd.Series({'Risk': .6 if direction == 'max' else 1.}),
    )
    return dict(
        trade_table=table,
        base_constraints=Constraints(
            min_weights=lower, max_weights=upper, group_lower_upper_constraints=group),
        covariance=pd.DataFrame(np.eye(3)*.04, index=assets, columns=assets),
        optimiser_config=OptimiserConfig(solver='CLARABEL'),
    )


@pytest.mark.parametrize('direction', ['min', 'max'])
@pytest.mark.parametrize('residual', [1e-7, 2.4e-5])
def test_rescue_reaches_same_rounded_solution_as_direct_sizing(direction, residual):
    """A residual left after admission must reach the existing audited sizing solve."""
    problem = _problem(direction, residual)
    original = problem['base_constraints'].group_lower_upper_constraints
    before_min = original.group_min_allocation.copy()
    before_max = original.group_max_allocation.copy()
    selected = problem['trade_table'].copy()
    selected.loc['Rescue', s.SELECTED_TRADE] = True
    direct = solve_selected_execution_portfolio(**{**problem, 'trade_table': selected})
    assert direct.accepted and direct.compliant

    rescued = solve_feasible_execution_portfolio(**problem, expand_for_feasibility=True)
    assert rescued.accepted and rescued.compliant
    pd.testing.assert_series_equal(rescued.weights, direct.weights)
    assert rescued.trade_table[s.FEASIBILITY_RESCUE_TRADE].sum() == 1
    assert rescued.trade_table.loc['Rescue', s.FEASIBILITY_RESCUE_TRADE]
    assert rescued.weights['Pinned'] == pytest.approx(
        selected.loc['Pinned', s.BASE_WEIGHT], abs=1e-8)
    assert rescued.weights.sum() == pytest.approx(1., abs=1e-7)
    assert not problem['trade_table'][s.SELECTED_TRADE].any()
    pd.testing.assert_series_equal(original.group_min_allocation, before_min)
    pd.testing.assert_series_equal(original.group_max_allocation, before_max)
    effective = rescued.outcome.constraints.group_lower_upper_constraints
    bounds = (effective.group_min_allocation if direction == 'min'
              else effective.group_max_allocation)
    expected = .6-residual if direction == 'min' else .6+residual
    assert bounds['Risk'] == pytest.approx(expected, abs=1e-14)


@pytest.mark.parametrize('direction', ['min', 'max'])
@pytest.mark.parametrize('restriction', ['material', 'cash', 'strict', 'target_bands'])
def test_rescue_preserves_material_cash_and_strict_group_failures(direction, restriction):
    """Residual handling must not weaken material bounds or contracts forbidding rounding."""
    residual = GROUP_BOUND_ROUNDING_TOLERANCE+1e-7 if restriction == 'material' else 5e-7
    problem = _problem(direction, residual, cash_bound=restriction == 'cash',
                       target_bands=restriction == 'target_bands')
    scope = _original_group_bounds_scope() if restriction == 'strict' else nullcontext()
    with scope, pytest.raises(ExecutionSolverInfeasibility, match='Infeasible execution intervals'):
        solve_feasible_execution_portfolio(**problem, expand_for_feasibility=True)


def test_rescue_cannot_trade_an_ineligible_instrument():
    """A material deficit stays infeasible if the only possible rescue is frozen."""
    problem = _problem('min', 1e-7)
    problem['trade_table'].loc['Rescue', s.REBALANCE_CADENCE_ELIGIBLE] = False
    with pytest.raises(ExecutionSolverInfeasibility, match="Group 'Risk'"):
        solve_feasible_execution_portfolio(**problem, expand_for_feasibility=True)
