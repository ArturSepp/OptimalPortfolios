"""Independent separable and integer-perspective objective identities."""

from dataclasses import replace

import numpy as np

from optimalportfolios.execution import compile_states, wrapper_ranked_execution
from optimalportfolios.execution.domain import variance
from optimalportfolios.execution.model import build_model, solve_qp
from optimalportfolios.execution.tests.formulation_test import example


def test_separable_exactness_and_integer_perspective_identity():
    """With diagonal funded covariance the two largest independent gains are exact."""
    p = replace(example(2, coupled=False), covariance=np.diag([0.01, 0.02, 0.03, 0.04, 0.0]))
    gains = np.diag(p.covariance) * (p.target - p.current) ** 2
    expected = p.current.copy()
    chosen = np.argsort(gains[:-1])[-2:]
    expected[chosen] = p.target[chosen]
    expected[-1] = p.investment - expected[:-1].sum()
    actual = wrapper_ranked_execution(p)
    assert abs(variance(p, actual.weights) - variance(p, expected)) < 1e-12
    states = compile_states(p)
    selected = {
        i: next(j for j, state in enumerate(row) if state.ticket == (i in chosen))
        for i, row in enumerate(states[:-1])
    }
    model = build_model(p, perspective=True, fixed_states=selected)
    weights = solve_qp(model)
    assert abs(model.optimisation.value / 1e4 - variance(p, weights)) < 1e-10
    assert abs(variance(p, weights) - variance(p, expected)) < 1e-10


def test_optional_states_obey_independent_trade_disjunction():
    """Every active interval lies wholly outside the current-relative no-ticket band."""
    p = example()
    for i, row in enumerate(compile_states(p)):
        if i == p.cash:
            continue
        for state in row:
            if state.ticket:
                assert (
                    state.upper <= p.current[i] - p.rules.minimum_ticket
                    or state.lower >= p.current[i] + p.rules.minimum_ticket
                )
