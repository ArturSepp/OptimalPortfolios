"""Fail-closed configuration, lifecycle and fallback contracts for V2 production."""
from dataclasses import replace

import numpy as np
import pytest

from optimalportfolios.execution.v1 import schema as s
from optimalportfolios.execution.v1.policy import ExactExecutionConfig
from optimalportfolios.execution.v2 import production
from optimalportfolios.execution.v2.types import ExecutionResult
from optimalportfolios.execution.v2.tests.production_test import inputs


@pytest.mark.parametrize('change, message', [
    ({'exact_execution': ExactExecutionConfig(), 'expand_for_feasibility': False},
     'operational MIQP'),
    ({'minimum_trade_score': .01}, 'own state search'),
    ({'expand_for_feasibility': False}, 'common feasibility'),
    ({'preparation_time_limit_s': float('nan')}, 'finite and positive'),
    ({'allow_v1_fallback': 1}, 'boolean'),
])
def test_incompatible_policy_combinations_fail_before_execution(change, message):
    """A method selector must not silently discard another policy's controls."""
    _, _, _, config = inputs()
    with pytest.raises(ValueError, match=message):
        replace(config, **change)


def test_short_or_variable_investment_is_not_dropped():
    """Unsupported investment semantics stop translation instead of becoming long-only."""
    table, constraints, covariance, config = inputs()
    with pytest.raises(ValueError, match='long-only fixed investment'):
        production.resolve_production_problem(
            table, replace(constraints, min_exposure=.8), covariance, config)


def test_group_free_instructions_preserve_direction_and_cash():
    """Fixed buy and sell instructions retain categories without synthetic group rows."""
    table, constraints, covariance, config = inputs()
    table[s.BASE_WEIGHT] = [.45, .15, .4]
    table[s.MANDATORY_TRADE] = [True, True, False]
    table[s.MANDATORY_CATEGORY] = ['fixed_instruction', 'fixed_instruction', '']
    table[s.RULE4_TRADE] = False
    table[s.SELECTED_TRADE] = False
    p, _ = production.resolve_production_problem(
        table, replace(constraints, group_lower_upper_constraints=None), covariance, config)
    assert p.categories == ('instruction_min_buy', 'instruction_max_sell', '')
    assert p.groups.shape == (0, 3)
    np.testing.assert_array_equal(p.base, [.45, .15, .4])


def test_failed_common_preparation_cannot_deliver_a_fallback(monkeypatch):
    """There is no common feasible domain against which to qualify a V1 fallback."""
    table, constraints, covariance, config = inputs()
    monkeypatch.setattr(production, 'prepare_common_execution',
                        lambda *a, **k: (None, {'status': 'preparation_failed'}))

    def forbidden():
        """A fallback cannot bypass failed preparation."""
        pytest.fail('Fallback bypassed failed preparation')

    with pytest.raises(RuntimeError, match='common preparation failed'):
        production.solve_v2_execution_portfolio(
            table, constraints, covariance, config, fallback=forbidden)


@pytest.mark.parametrize('accepted', [False, True])
def test_structured_fallback_keeps_both_acceptance_gates(monkeypatch, accepted):
    """A structured fallback needs its original acceptance and the V2 audit."""
    table, constraints, covariance, config = inputs()
    original = production.solve_v2_execution_portfolio(table, constraints, covariance, config)
    candidate = replace(original, outcome=replace(original.outcome, accepted=accepted))
    monkeypatch.setattr(production, 'wrapper_ranked_execution',
                        lambda *a, **k: ExecutionResult(None, 'bounded_abstention', {}))
    if accepted:
        result = production.solve_v2_execution_portfolio(
            table, constraints, covariance, config, fallback=lambda: candidate)
        assert result.accepted and result.compliant
        assert result.outcome.fallback_source == 'v1_audited_fallback'
    else:
        with pytest.raises(RuntimeError, match='not accepted by its own policy'):
            production.solve_v2_execution_portfolio(
                table, constraints, covariance, config, fallback=lambda: candidate)


def test_abstention_without_fallback_is_explicit(monkeypatch):
    """A missing audited incumbent cannot become the display model or current book."""
    table, constraints, covariance, config = inputs()
    monkeypatch.setattr(production, 'wrapper_ranked_execution',
                        lambda *a, **k: ExecutionResult(None, 'bounded_abstention', {}))
    with pytest.raises(RuntimeError, match='without a qualified fallback'):
        production.solve_v2_execution_portfolio(table, constraints, covariance, config)
