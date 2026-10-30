"""Independent checks for isolated numerical mechanism controls."""
import numpy as np
import pytest

from papers.ranked_execution_2026.replication import study_numerics as numerics
from papers.ranked_execution_2026.replication.study_inputs import corridor_counterexample
from optimalportfolios.execution.v2 import model


def test_objective_translation_changes_only_relative_stopping_decision():
    """A known bound pair must not become converged through a removed constant."""
    example=numerics.origin_example()
    assert example['absolute_gap']>example['absolute_target']
    assert not example['original_stops'] and example['translated_stops']
    assert example['original_relative_gap']==pytest.approx(.1)


def test_equivalent_cone_graphs_equal_independent_scalar_quadratic():
    """Check full objective, including its origin, at several funded portfolios."""
    problem=corridor_counterexample()
    for balanced in (False,True):
        graph=numerics.cone_model(problem,balanced)
        for first in (.2,.25,.3):
            weights=np.array([first,.2,.8-first])
            graph.displacement_bp.value=1e4*(weights[graph.indices]-graph.reference[graph.indices])
            x=first-.3
            expected=.01*(x*x-.18*x+.01)
            assert graph.optimisation.objective.value/1e4==pytest.approx(expected,abs=1e-15)


def test_constant_row_control_restores_original_builder_and_genuine_breach():
    """The research toggle cannot persist or erase an inconsistent constant row."""
    original=model._interval_rows
    problem=numerics.pinned_group('upper',1.01e-7)
    for reduce_rows in (False,True):
        graph,_,_=numerics.fixed_row_model(problem,reduce_rows)
        assert model._interval_rows is original
        weights=model.solve_qp(graph)
        assert weights is None and graph.optimisation.status=='infeasible'
