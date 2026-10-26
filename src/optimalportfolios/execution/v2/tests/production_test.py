"""Functional contracts for the V2 production bridge and strict fallback gate."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from optimalportfolios.execution.v1.tests.production_solver_test import (
    ASSETS, _base_constraints, _trade_table,
)
from optimalportfolios.execution.v1 import schema as s


def inputs():
    """Build a three-asset funded example independently of the V2 compiler."""
    from optimalportfolios.execution.v2.production import V2ExecutionPolicyConfig
    config = V2ExecutionPolicyConfig(
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
    from optimalportfolios.execution.v2.production import solve_v2_execution_portfolio
    table, constraints, covariance, config = inputs()
    solved = solve_v2_execution_portfolio(table, constraints, covariance, config)
    assert solved.accepted and solved.compliant
    assert solved.trade_table.attrs['v2']['audit']['passed']
    delta = (solved.weights-table[s.CURRENT_WEIGHT]).abs().iloc[:2]
    assert sum(delta > 1e-8) <= 1
    assert (delta[delta > 1e-8] >= .025-1e-7).all()
    assert abs(solved.weights.sum()-1.) <= 1e-7
    assert (solved.outcome.constraints.group_lower_upper_constraints
            is constraints.group_lower_upper_constraints)
    assert solved.outcome.constraints.tracking_err_vol_constraint is None
    assert config.execution_method == 'ranked_v2'


def test_unavailable_cadence_is_algebraically_pinned():
    """A high-priority instrument cannot be opened by the V2 compiler off calendar."""
    from optimalportfolios.execution.v2.production import resolve_production_problem
    table, constraints, covariance, config = inputs()
    table.loc['A', s.REBALANCE_CADENCE_ELIGIBLE] = False
    problem, _ = resolve_production_problem(table, constraints, covariance, config)
    assert problem.lower[0] == problem.upper[0] == .4
    assert problem.full_lower[0] == problem.full_upper[0] == .4


def test_unsupported_hard_rule_fails_with_dated_trade_audit():
    """A volatility constraint cannot disappear while converting a fund mandate."""
    from optimalportfolios.execution.v2.production import solve_v2_execution_portfolio
    table, constraints, covariance, config = inputs()
    with pytest.raises(ValueError, match='max_target_portfolio_vol_an') as caught:
        solve_v2_execution_portfolio(
            table, replace(constraints, max_target_portfolio_vol_an=.2), covariance, config)
    pd.testing.assert_frame_equal(caught.value.trade_table, table)


def test_fallback_is_called_only_on_abstention_and_must_pass_v2_audit(monkeypatch):
    """A V1 output is not accepted merely because its own policy approved it."""
    from optimalportfolios.execution.v2 import production
    from optimalportfolios.execution.v2.types import ExecutionResult
    table, constraints, covariance, config = inputs()
    monkeypatch.setattr(production, 'wrapper_ranked_execution',
                        lambda *a, **k: ExecutionResult(None, 'bounded_abstention', {}))
    calls = []

    def bad_fallback():
        """Return a plausible V1 portfolio violating the new hard ticket cap."""
        calls.append(True)
        return pd.Series([.3, .3, .4], index=ASSETS)

    with pytest.raises(RuntimeError, match='fallback_rejected') as caught:
        production.solve_v2_execution_portfolio(
            table, constraints, covariance, config, fallback=bad_fallback)
    assert calls == [True]
    assert 'total_ticket_cap' in caught.value.diagnostics['fallback_audit']['violations']


def test_valid_fallback_is_explicitly_attributed(monkeypatch):
    """A compatible V1 endpoint retains fallback identity in both result and table."""
    from optimalportfolios.execution.v2 import production
    from optimalportfolios.execution.v2.types import ExecutionResult
    table, constraints, covariance, config = inputs()
    monkeypatch.setattr(production, 'wrapper_ranked_execution',
                        lambda *a, **k: ExecutionResult(None, 'bounded_abstention', {}))
    solved = production.solve_v2_execution_portfolio(
        table, constraints, covariance, config,
        fallback=lambda: pd.Series([.35, .2, .45], index=ASSETS))
    assert solved.accepted and solved.compliant
    assert solved.outcome.fallback_source == 'v1_audited_fallback'
    assert solved.trade_table['execution_engine'].eq('v1_audited_fallback').all()
