"""Execution boundary failures and signed funding-repair eligibility."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from optimalportfolios.execution import schema as s
from optimalportfolios.execution import solver
from optimalportfolios.execution.tests.solver_test import _base_constraints, _trade_table
from optimalportfolios.optimization.constraints import Constraints, BenchmarkBetaConstraint


@pytest.mark.parametrize('case,message', [
    ('target_band', 'target-band'), ('missing', 'missing columns'),
    ('duplicates', 'unique'), ('cash', 'exactly one'),
    ('cash_selected', 'Rule 4'), ('nonfinite', 'finite'),
    ('hard_intersection', 'do not intersect'),
    ('selected_intersection', 'selected execution corridors'),
    ('base_excluded', 'post-mandatory base'), ('frozen', 'fixed execution base'),
])
def test_invalid_corridor_inputs_are_not_relaxed(case, message) -> None:
    """Invalid inputs never silently widen a selected interval or a frozen holding."""
    table = _trade_table([0.4, 0.2, 0.4], [0.3, 0.3, 0.4], [True, False, False])
    constraints = _base_constraints()
    if case == 'target_band':
        table[s.TARGET_BAND_LOWER] = 0.0
        table[s.TARGET_BAND_UPPER] = 1.0
    elif case == 'missing':
        table = table.drop(columns=s.BASE_WEIGHT)
    elif case == 'duplicates':
        table = pd.concat([table, table.iloc[:1]], axis=0)
    elif case == 'cash':
        table[s.SETTLEMENT_CASH] = False
    elif case == 'cash_selected':
        table.loc['Cash', s.SELECTED_TRADE] = True
    elif case == 'nonfinite':
        table.loc['A', s.BASE_WEIGHT] = np.nan
    elif case == 'hard_intersection':
        table.loc['A', s.POLICY_MIN_WEIGHT] = 2.0
    elif case == 'selected_intersection':
        table.loc['A', s.POLICY_MIN_WEIGHT] = 0.5
    elif case == 'base_excluded':
        table.loc['A', s.POLICY_MAX_WEIGHT] = 0.35
    elif case == 'frozen':
        table.loc['B', s.POLICY_MAX_WEIGHT] = 0.1
    expected = NotImplementedError if case == 'target_band' else ValueError
    with pytest.raises(expected, match=message):
        solver.build_selected_execution_constraints(constraints, table)


def test_empty_bridges_and_unresolved_beta() -> None:
    """An empty bridge set is valid; unresolved risk constraints are not executable."""
    assert solver._positive_bridges(pd.DataFrame()).empty
    table = _trade_table([0.4, 0.2, 0.4], [0.3, 0.3, 0.4], [True, True, False])
    constraints = replace(_base_constraints(),
                          benchmark_beta_constraint=BenchmarkBetaConstraint(beta_min=0.1))
    with pytest.raises(ValueError, match='resolved benchmark beta'):
        solver.build_selected_execution_constraints(constraints, table)


@pytest.mark.parametrize('message,exception', [
    ('Infeasible constraints detected: injected incompatible mandate',
     solver.ExecutionSolverInfeasibility),
    ('unexpected constraint validation failure', ValueError),
])
def test_constraint_compilation_preserves_failure_category(monkeypatch, message, exception) -> None:
    """Only explicit infeasibility is classified as a structural solver failure."""
    table = _trade_table([0.4, 0.2, 0.4], [0.3, 0.3, 0.4], [True, True, False])
    constraints = _base_constraints()
    original = ValueError(message)

    def fail_compilation(self, **kwargs):
        """Simulate constraint validation independently of numerical optimization."""
        raise original

    monkeypatch.setattr(Constraints, 'copy', fail_compilation)
    with pytest.raises(exception, match=message) as failure:
        solver.build_selected_execution_constraints(constraints, table)
    if exception is solver.ExecutionSolverInfeasibility:
        assert failure.value.__cause__ is original
        pd.testing.assert_frame_equal(failure.value.trade_table, table)
    else:
        assert failure.value is original


@pytest.mark.parametrize('entrypoint', [
    solver.solve_selected_execution_portfolio, solver.solve_feasible_execution_portfolio,
])
def test_target_band_policy_is_explicitly_unsupported(entrypoint) -> None:
    """A different objective cannot silently fall through to legacy ranked projection."""
    table = _trade_table([0.4, 0.2, 0.4], [0.3, 0.3, 0.4], [True, True, False])
    covariance = pd.DataFrame(np.eye(3), index=table.index, columns=table.index)
    with pytest.raises(NotImplementedError, match='group-split'):
        entrypoint(table, _base_constraints(), covariance,
                   group_split_asset_classes=pd.Series('Risk', index=table.index))


@pytest.mark.parametrize('breach,expected', [
    ('cash_min', ['A']), ('cash_max', ['B']), ('group_min', ['B']),
])
def test_signed_rescue_obeys_cash_and_group_funding_direction(breach, expected) -> None:
    """A sale funds a cash deficit; a purchase reduces excess cash or a group shortfall."""
    table = _trade_table([0.4, 0.2, 0.4], [0.3, 0.3, 0.4], [False, False, False])
    constraints = _base_constraints()
    if breach == 'cash_min':
        table.loc['Cash', s.POLICY_MIN_WEIGHT] = 0.5
    elif breach == 'cash_max':
        table.loc['Cash', s.POLICY_MAX_WEIGHT] = 0.3
    else:
        constraints = replace(constraints, group_lower_upper_constraints=replace(
            constraints.group_lower_upper_constraints,
            group_min_allocation=pd.Series({'Risk Assets': 0.7})))
    eligible = pd.Series([True, True, False], index=table.index)
    actual = solver._get_sign_directed_rescue_candidates(table, constraints, eligible)
    assert table.index[actual].tolist() == expected
    table[s.SETTLEMENT_CASH] = False
    with pytest.raises(ValueError, match='exactly one'):
        solver._get_sign_directed_rescue_candidates(table, constraints, eligible)
