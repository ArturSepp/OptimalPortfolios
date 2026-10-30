"""Preserved lifecycle and instrument-bound failure contracts after V1 retirement."""

import numpy as np
import pandas as pd
import pytest
from optimalportfolios import Constraints, GroupLowerUpperConstraints
from optimalportfolios.execution import schema as s
from optimalportfolios.execution import instrument_bounds as bounds
from optimalportfolios.execution.lifecycle import MandatoryTrades, build_execution_target
from optimalportfolios.execution.tests.lifecycle_validation_test import target, config

ASSETS = pd.Index(["A", "B", "Cash"])


def _base_constraints(group_max: float = 0.8) -> Constraints:
    """Construct hard asset/group bounds with inherited model utilities."""
    group = GroupLowerUpperConstraints(
        group_loadings=pd.DataFrame({"Risk Assets": [1.0, 1.0, 0.0]}, index=ASSETS),
        group_min_allocation=pd.Series({"Risk Assets": 0.0}),
        group_max_allocation=pd.Series({"Risk Assets": group_max}),
    )
    return Constraints(
        min_weights=pd.Series(0.0, index=ASSETS),
        max_weights=pd.Series(1.0, index=ASSETS),
        tracking_err_vol_constraint=0.001,
        turnover_constraint=0.001,
        tre_utility_weight=25.0,
        turnover_utility_weight=10.0,
        group_lower_upper_constraints=group,
    )


def _trade_table(
    current: list[float],
    model: list[float],
    selected: list[bool],
    base: list[float] | None = None,
    mandatory: list[bool] | None = None,
) -> pd.DataFrame:
    """Construct a resolved, self-financing three-asset legacy trade table."""
    current_s = pd.Series(current, index=ASSETS)
    model_s = pd.Series(model, index=ASSETS)
    base_s = current_s.copy() if base is None else pd.Series(base, index=ASSETS)
    mandatory_s = pd.Series([False, False, False] if mandatory is None else mandatory, index=ASSETS)
    rule4 = ~mandatory_s & pd.Series([True, True, False], index=ASSETS)
    desired = (model_s - base_s).where(rule4, 0.0)
    return pd.DataFrame(
        {
            s.CURRENT_WEIGHT: current_s,
            s.RAW_MODEL_WEIGHT: model_s,
            s.EFFECTIVE_MODEL_WEIGHT: model_s,
            s.BASE_WEIGHT: base_s,
            s.DESIRED_REWEIGHT: desired,
            s.MANDATORY_FUNDING: (base_s - current_s).where(
                ~pd.Series([False, False, True], index=ASSETS), -(base_s - current_s).iloc[:2].sum()
            ),
            s.MANDATORY_TRADE: mandatory_s,
            s.RULE4_TRADE: rule4,
            s.POLICY_MIN_WEIGHT: 0.0,
            s.POLICY_MAX_WEIGHT: 1.0,
            s.REBALANCE_CADENCE_ELIGIBLE: True,
            s.SETTLEMENT_CASH: [False, False, True],
            s.SELECTED_TRADE: selected,
            s.TRADE_SCORE: [3.0, 2.0, 0.0],
            s.SCORE_PER_TURNOVER: [3.0, 2.0, 0.0],
        },
        index=ASSETS,
    )


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
        bounds._build_selected_execution_bound_series(constraints, table, False)


def test_paired_instructions_and_funding_refusal():
    """Explicit ranges preserve both endpoints and require a real funding leg."""
    table = target(mandatory_trades=MandatoryTrades(hard_buy_min={"A": 0.3}, hard_max={"A": 0.5}))
    assert (
        table.loc["A", "instruction_min_weight"] == 0.3
        and table.loc["A", "instruction_max_weight"] == 0.5
    )
    with pytest.raises(ValueError, match="settlement-cash"):
        build_execution_target(
            pd.Series([0.4, 0.6], index=["A", "B"]),
            pd.Series([0.0, 1.0], index=["A", "B"]),
            config=config(),
        )
    with pytest.raises(RuntimeError, match="effective execution target"):
        build_execution_target(
            pd.Series([0.4, 0.6], index=["A", "B"]),
            pd.Series([0.3, 0.3], index=["A", "B"]),
            config=config(),
        )
    with pytest.raises(RuntimeError, match="post-mandatory base"):
        build_execution_target(
            pd.Series([1e16, 1.0, -1e16], index=["A", "B", "Cash"]),
            pd.Series([0.0, 0.5, 0.5], index=["A", "B", "Cash"]),
            config=config(),
        )


@pytest.mark.parametrize('relaxed', [False, True])
def test_selected_bounds_preserve_intersections_and_optional_relaxation(relaxed):
    """Only the declared optional corridor flag changes the interval endpoints."""
    table = _trade_table([0.4, 0.2, 0.4], [0.3, 0.3, 0.4], [True, True, False])
    minimum, maximum = bounds._build_selected_execution_bound_series(
        _base_constraints(), table, relaxed)
    np.testing.assert_allclose(minimum, [0, 0, 0] if relaxed else [0.3, 0.2, 0])
    np.testing.assert_allclose(maximum, [1, 1, 1] if relaxed else [0.4, 0.3, 1])


def test_target_band_intervals_preserve_tickets_and_compulsory_directions():
    """A selected ticket has a minimum size; compulsory ranges retain their direction."""
    table = _trade_table([0.4, 0.2, 0.4], [0.3, 0.3, 0.4], [True, True, False])
    table[s.TARGET_BAND_LOWER] = [0.28, 0.28, 0.0]
    table[s.TARGET_BAND_UPPER] = [0.32, 0.32, 1.0]
    table[s.MODEL_ENTRY] = False
    table[s.MATERIAL_TRADE_THRESHOLD] = 0.02
    minimum, maximum = bounds._build_selected_execution_bound_series(
        _base_constraints(), table, False)
    np.testing.assert_allclose(minimum, [0.28, 0.28, 0.0])
    np.testing.assert_allclose(maximum, [0.32, 0.32, 1.0])
    with pytest.raises(ValueError, match='never relaxes target bands'):
        bounds._build_selected_execution_bound_series(_base_constraints(), table, True)
    table[s.MANDATORY_TRADE] = [True, True, False]
    table[s.MANDATORY_CATEGORY] = ['instruction_max_sell', 'instruction_min_buy', '']
    minimum, maximum = bounds._build_selected_execution_bound_series(
        _base_constraints(), table, False)
    np.testing.assert_allclose(minimum, [0.3, 0.2, 0.0])
    np.testing.assert_allclose(maximum, [0.4, 0.3, 1.0])


def test_structural_failure_retains_original_evidence():
    """A structural rejection copies the diagnostic bridge, retaining the attempted table."""
    table = _trade_table([0.4, 0.2, 0.4], [0.3, 0.3, 0.4], [True, True, False])
    bridge = pd.DataFrame({'bridge_min': [0.01]})
    error = bounds.ExecutionSolverInfeasibility('structural', bridge, table)
    bridge.iloc[0, 0] = 1
    assert error.bridges.iloc[0, 0] == 0.01
    assert error.trade_table is table
    assert bounds.ExecutionSolverInfeasibility('empty').bridges.empty


def test_group_minimum_preserves_compulsory_sell_down_capacity():
    """A small model allocation cannot force an exit that destroys group capacity."""
    current = pd.Series([0.06, 0.84, 0.10], index=['A', 'B', 'Cash'])
    model = pd.Series([0.004, 0.896, 0.10], index=current.index)
    table = build_execution_target(
        current, model, previous_model_weights=model, config=config(),
        settlement_cash_asset='Cash',
        group_loadings=pd.DataFrame({'Required': [1., 0., 0.]}, index=current.index),
        group_min=pd.Series({'Required': 0.05}), group_max=pd.Series({'Required': 0.2}))
    assert table.loc['A', s.CUTOFF_SELL_DOWN]
    assert not table.loc['A', s.CUTOFF_EXIT]
    assert table.loc['A', s.BASE_WEIGHT] == 0.06


@pytest.mark.parametrize('case,message', [
    ('direction', 'funded-trade variance'), ('post_trade', 'incremental active variance')])
def test_vector_score_rejects_indefinite_risk(case, message):
    """The vector kernel must reject the same invalid portfolios as the scalar reference."""
    from optimalportfolios.execution.tests.diagnostic_contract_test import make_problem, score
    from optimalportfolios.execution.diagnostics import _incremental_funded_tre

    problem = make_problem()
    problem.covariance.iloc[:, :] = 0.0
    block = np.array([[1., 2.], [2., 1. if case == 'direction' else 3.]])
    problem.covariance.loc[['A', 'Cash'], ['A', 'Cash']] = block
    active = np.zeros(4)
    if case == 'post_trade':
        active[0] = 1.0
    problem.target[s.RAW_MODEL_WEIGHT] = problem.target[s.BASE_WEIGHT] - active
    problem.target[s.DESIRED_REWEIGHT] = [2., 0., 0., 0.]
    with pytest.raises(ValueError, match=message):
        _incremental_funded_tre(problem.covariance.to_numpy(), active, 0, 3, 2.)
    with pytest.raises(ValueError, match=message):
        score(problem)
