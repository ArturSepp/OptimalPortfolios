"""Regressions for bounded search restarted from an improved sized incumbent."""

import numpy as np
import pytest

from optimalportfolios.execution import (
    ExecutionProblem, ExecutionRules, RankedConfig, audit_execution, compile_states,
    wrapper_ranked_execution,
)
from optimalportfolios.execution.ranked_execution import ranked_selection


def three_trade_problem():
    """Three correlated trades have an exactly feasible zero-risk joint target."""
    current = np.array([.05, .05, .08, .12])
    target = np.array([.07, .07, .04, .12])
    covariance = np.zeros((4, 4))
    covariance[:3, :3] = .99
    np.fill_diagonal(covariance[:3, :3], 1.)
    return ExecutionProblem(
        names=("buy_a", "buy_b", "sell", "cash"), cash=3,
        current=current, base=current, target=target,
        lower=np.minimum(current, target), upper=np.maximum(current, target),
        full_lower=np.zeros(4), full_upper=np.full(4, .3),
        groups=np.zeros((0, 4)), group_lower=np.zeros(0), group_upper=np.zeros(0),
        mandatory=np.zeros(4, bool), categories=("",) * 4,
        eligible=np.ones(4, bool), frozen=np.zeros(4, bool), investment=.3,
        rules=ExecutionRules(ticket_cap=3, minimum_ticket=.01, holding_floor=.005),
        covariance=covariance,
    )


def test_revisit_improved_pair_to_admit_a_third_trade():
    """A new anchor must expose the third trade that the original two-edit search misses."""
    problem = three_trade_problem()
    selected, _, _, _ = ranked_selection(problem, compile_states(problem), 3)
    assert selected == {}
    result = wrapper_ranked_execution(problem)
    assert audit_execution(problem, result.weights)["passed"]
    # Independent optimum: the target is feasible and attains the PSD lower bound zero.
    assert result.diagnostics["variance"] < 1e-10
    np.testing.assert_allclose(result.weights, problem.target, atol=1e-5, rtol=0)
    passes = result.diagnostics["coordinated_passes"]
    assert 1 < len(passes) <= 3
    assert sum(row["trials"] for row in passes) <= 24


def test_one_anchor_retains_the_previous_search_result():
    """An explicit one-pass control retains the pair-only historical result."""
    result = wrapper_ranked_execution(three_trade_problem(), RankedConfig(improvement_passes=1))
    assert result.diagnostics["variance"] == pytest.approx(8e-6, abs=1e-10)
    assert len(result.diagnostics["coordinated_passes"]) == 1


@pytest.mark.parametrize("limit", [None, -1, 1.5])
def test_restart_count_must_be_a_finite_nonnegative_integer(limit):
    """A restart setting cannot turn the bounded heuristic into an unbounded search."""
    with pytest.raises(ValueError):
        RankedConfig(improvement_passes=limit)
