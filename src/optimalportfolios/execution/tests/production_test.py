"""Functional contracts for the V2 production bridge and strict fallback gate."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from optimalportfolios import Constraints, GroupLowerUpperConstraints
from optimalportfolios.execution.schema import (
    BASE_WEIGHT, CURRENT_WEIGHT, DESIRED_REWEIGHT, EFFECTIVE_MODEL_WEIGHT, MANDATORY_FUNDING, MANDATORY_TRADE, POLICY_MAX_WEIGHT, POLICY_MIN_WEIGHT, RAW_MODEL_WEIGHT, REBALANCE_CADENCE_ELIGIBLE, RULE4_TRADE, SCORE_PER_TURNOVER, SELECTED_TRADE, SETTLEMENT_CASH, TRADE_SCORE
)
ASSETS = pd.Index(["A", "B", "Cash"])

def _base_constraints(group_max: float = 0.8) -> Constraints:
    """Return full product constraints for the three-asset reference."""
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
    """Build a resolved trade table independently of target preparation."""
    current_s = pd.Series(current, index=ASSETS)
    model_s = pd.Series(model, index=ASSETS)
    base_s = current_s.copy() if base is None else pd.Series(base, index=ASSETS)
    mandatory_s = pd.Series([False, False, False] if mandatory is None else mandatory, index=ASSETS)
    rule4 = ~mandatory_s & pd.Series([True, True, False], index=ASSETS)
    desired = (model_s - base_s).where(rule4, 0.0)
    return pd.DataFrame(
        {
            CURRENT_WEIGHT: current_s,
            RAW_MODEL_WEIGHT: model_s,
            EFFECTIVE_MODEL_WEIGHT: model_s,
            BASE_WEIGHT: base_s,
            DESIRED_REWEIGHT: desired,
            MANDATORY_FUNDING: (base_s - current_s).where(
                ~pd.Series([False, False, True], index=ASSETS), -(base_s - current_s).iloc[:2].sum()
            ),
            MANDATORY_TRADE: mandatory_s,
            RULE4_TRADE: rule4,
            POLICY_MIN_WEIGHT: 0.0,
            POLICY_MAX_WEIGHT: 1.0,
            REBALANCE_CADENCE_ELIGIBLE: True,
            SETTLEMENT_CASH: [False, False, True],
            SELECTED_TRADE: selected,
            TRADE_SCORE: [3.0, 2.0, 0.0],
            SCORE_PER_TURNOVER: [3.0, 2.0, 0.0],
        },
        index=ASSETS,
    )

from optimalportfolios.execution import schema as s


def inputs():
    """Build a three-asset funded example independently of the V2 compiler."""
    from optimalportfolios.execution.production import RankedExecutionPolicyConfig
    config = RankedExecutionPolicyConfig(
        tre_weight_by_asset_class={'Risk': 1., 'Cash': 1.},
        small_trade_cutoff=.005, material_trade_threshold=.025,
        max_trades=1, add_mandatory_trades_to_max_trades=True,
        preparation_solver='SCIPY',
    )
    table = _trade_table([.4, .2, .4], [.3, .3, .4], [True, False, False])
    table[s.MANDATORY_CATEGORY] = ''
    table[s.EFFECTIVE_SMALL_TRADE_CUTOFF] = [.005, .005, 0.]
    table[s.MATERIAL_TRADE_THRESHOLD] = config.material_trade_threshold
    covariance = pd.DataFrame(np.eye(3)*.04, index=ASSETS, columns=ASSETS)
    return table, _base_constraints(), covariance, config


def test_production_v2_has_hard_actual_tickets_and_original_groups():
    """A one-ticket portfolio is independently verified, including unchanged hard bands."""
    from optimalportfolios.execution.production import solve_ranked_execution_portfolio
    table, constraints, covariance, config = inputs()
    solved = solve_ranked_execution_portfolio(table, constraints, covariance, config)
    assert solved.accepted and solved.compliant
    assert solved.trade_table.attrs['execution']['audit']['passed']
    delta = (solved.weights-table[s.CURRENT_WEIGHT]).abs().iloc[:2]
    assert sum(delta > 1e-8) <= 1
    assert (delta[delta > 1e-8] >= .025-1e-7).all()
    assert abs(solved.weights.sum()-1.) <= 1e-7
    assert (solved.outcome.constraints.group_lower_upper_constraints
            is constraints.group_lower_upper_constraints)
    assert solved.outcome.constraints.tracking_err_vol_constraint is None
    assert config.execution_method == 'ranked_execution'


def test_unavailable_cadence_is_algebraically_pinned():
    """A high-priority instrument cannot be opened by the V2 compiler off calendar."""
    from optimalportfolios.execution.production import resolve_production_problem
    table, constraints, covariance, config = inputs()
    table.loc['A', s.REBALANCE_CADENCE_ELIGIBLE] = False
    problem, _ = resolve_production_problem(table, constraints, covariance, config)
    assert problem.lower[0] == problem.upper[0] == .4
    assert problem.full_lower[0] == problem.full_upper[0] == .4


def test_unsupported_hard_rule_fails_with_dated_trade_audit():
    """A volatility constraint cannot disappear while converting a fund mandate."""
    from optimalportfolios.execution.production import solve_ranked_execution_portfolio
    table, constraints, covariance, config = inputs()
    with pytest.raises(ValueError, match='max_target_portfolio_vol_an') as caught:
        solve_ranked_execution_portfolio(
            table, replace(constraints, max_target_portfolio_vol_an=.2), covariance, config)
    pd.testing.assert_frame_equal(caught.value.trade_table, table)




