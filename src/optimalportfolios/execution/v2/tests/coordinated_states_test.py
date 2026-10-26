"""Ensure coordinated supports contain changes to distinct instruments."""

import numpy as np

from optimalportfolios.execution.v2 import ExecutionProblem, ExecutionRules, compile_states


def test_partial_sale_and_exit_are_alternatives_not_a_pair():
    """Enumerate partial/exit alternatives once each, never paired with themselves."""
    from optimalportfolios.execution.v2.ranked import _coordinated_supports

    current = np.array([0.05, 0.05, 0.10])
    problem = ExecutionProblem(
        names=("buy", "sell", "cash"), cash=2, current=current, base=current,
        target=np.array([0.07, 0.0, 0.13]),
        lower=np.array([0.05, 0.0, 0.0]), upper=np.array([0.07, 0.05, 0.2]),
        full_lower=np.zeros(3), full_upper=np.full(3, 0.2),
        groups=np.zeros((0, 3)), group_lower=np.zeros(0), group_upper=np.zeros(0),
        mandatory=np.zeros(3, bool), categories=("",) * 3,
        eligible=np.ones(3, bool), frozen=np.zeros(3, bool), investment=0.2,
        rules=ExecutionRules(ticket_cap=2, minimum_ticket=0.01, holding_floor=0.04),
        covariance=np.diag([1., 1., 0.]),
    )
    states = compile_states(problem)
    buy, = [j for j, state in enumerate(states[0]) if state.ticket]
    partial, exit_state = [j for j, state in enumerate(states[1]) if state.ticket]
    expected = {
        ((0, buy),), ((1, partial),), ((1, exit_state),),
        ((0, buy), (1, partial)), ((0, buy), (1, exit_state)),
    }
    proposals = list(_coordinated_supports(problem, states, {}, problem.base, 2, set()))
    supports = [key for _, _, key in proposals]
    assert len(supports) == len(expected)
    assert set(supports) == expected
