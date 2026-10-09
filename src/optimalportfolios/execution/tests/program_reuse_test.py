"""Parameterized program identity, fresh native solves and complete-path parity."""
from copy import deepcopy
from dataclasses import replace

import cvxpy as cvx
import numpy as np
import pandas as pd
import pytest

from optimalportfolios.execution import branches as b
from optimalportfolios.execution.tests.improvement_test import _problem
from optimalportfolios.execution.tests.projection_reuse_test import _inputs
from optimalportfolios.optimization._covariance_cache import (
    _CovarianceCache, _covariance_reuse_scope,
)
from optimalportfolios.optimization._tracking_error_program import (
    _TrackingErrorProgramCache, _program_reuse_scope,
)
from optimalportfolios.optimization.general import minimum_tracking_error as mt


def test_changed_boxes_reuse_graph_and_match_fresh_native_problem_data(monkeypatch):
    """The existing compiler must produce exactly the same native matrices and bounds."""
    inputs, preparations, calls = _inputs(), _CovarianceCache(), []
    programs = _TrackingErrorProgramCache(preparations)
    original = mt._build_tracking_error_problem
    def counted(*args):
        """Count actual CVXPY construction independently of program telemetry."""
        result = original(*args)
        calls.append(result)
        return result
    monkeypatch.setattr(mt, '_build_tracking_error_problem', counted)
    with _covariance_reuse_scope(preparations), _program_reuse_scope(programs):
        mt.wrapper_minimise_tracking_error(**inputs)
        changed = dict(inputs, constraints=inputs['constraints'].copy(
            min_weights=pd.Series(0., index=inputs['pd_covar'].index),
            max_weights=pd.Series(.8, index=inputs['pd_covar'].index)))
        actual = mt.wrapper_minimise_tracking_error(**changed)
        assert len(calls)==1 and programs.hits==1
        reused_data = calls[0][1].get_problem_data('CLARABEL')[0]
    expected = mt.wrapper_minimise_tracking_error(**changed)
    fresh_data = calls[-1][1].get_problem_data('CLARABEL')[0]
    for name in ('A','P'):
        np.testing.assert_array_equal(reused_data[name].toarray(),fresh_data[name].toarray())
    for name in ('b','c'):
        np.testing.assert_array_equal(reused_data[name],fresh_data[name])
    pd.testing.assert_series_equal(actual[0],expected[0],check_exact=True)
    pd.testing.assert_frame_equal(actual[1].residuals_frame(),expected[1].residuals_frame(),check_exact=True)


@pytest.mark.parametrize('change',['covariance','benchmark','holdings','exposure','order','box_presence'])
def test_changed_fixed_inputs_require_a_new_program(change):
    """Bounds are the only variable numerical fields of a reusable program."""
    inputs, preparations = _inputs(), _CovarianceCache()
    programs = _TrackingErrorProgramCache(preparations)
    inputs['constraints']=inputs['constraints'].copy(min_weights=pd.Series(0.,index=inputs['pd_covar'].index),
        max_weights=pd.Series(1.,index=inputs['pd_covar'].index))
    changed=deepcopy(inputs)
    if change=='covariance':
        changed['pd_covar'].iloc[0,0]+=.01
    elif change=='benchmark':
        changed['benchmark_weights'].iloc[0]+=.01
    elif change=='holdings':
        changed['weights_0'].iloc[0]+=.01
    elif change=='exposure':
        changed['constraints']=changed['constraints'].copy(max_exposure=1.1)
    elif change=='order':
        changed['pd_covar']=changed['pd_covar'].iloc[::-1,::-1]
    else:
        changed['constraints']=changed['constraints'].copy(min_weights=None)
    with _covariance_reuse_scope(preparations), _program_reuse_scope(programs):
        mt.wrapper_minimise_tracking_error(**inputs)
        mt.wrapper_minimise_tracking_error(**changed)
    assert programs.builds==2 and programs.hits==0


def test_program_solves_disable_native_warm_start_and_detach_reported_weights(monkeypatch):
    """Every numerical request uses a new native solve rather than an earlier solution."""
    inputs, preparations, options, native = _inputs(), _CovarianceCache(), [], []
    programs = _TrackingErrorProgramCache(preparations)
    original = cvx.Problem.solve
    def counted(problem,*args,**kwargs):
        """Inspect actual solve options at the public CVXPY boundary."""
        options.append(kwargs.copy())
        return original(problem,*args,**kwargs)
    monkeypatch.setattr(cvx.Problem,'solve',counted)
    from cvxpy.reductions.solvers.conic_solvers.clarabel_conif import CLARABEL
    original_native=CLARABEL.solve_via_data
    def native_counted(self,data,warm_start,verbose,solver_opts,solver_cache=None):
        """Check the native adapter's actual flag after all public solve wrappers."""
        native.append(warm_start)
        return original_native(self,data,warm_start,verbose,solver_opts,solver_cache)
    monkeypatch.setattr(CLARABEL,'solve_via_data',native_counted)
    with _covariance_reuse_scope(preparations), _program_reuse_scope(programs):
        first=mt.wrapper_minimise_tracking_error(**inputs)
        expected=first[1].weights.copy()
        second=mt.wrapper_minimise_tracking_error(**dict(inputs,context='new audit context'))
    assert len(options)==2 and all(o['warm_start'] is False for o in options)
    assert native==[False,False]
    np.testing.assert_array_equal(first[1].weights,expected)
    assert second[1].context=='new audit context' and programs.hits==1


def test_consumer_constraint_mutation_cannot_poison_cached_constants():
    """The template's fixed policy data belong to its own deep copy."""
    inputs, preparations = _inputs(), _CovarianceCache()
    programs = _TrackingErrorProgramCache(preparations)
    with _covariance_reuse_scope(preparations), _program_reuse_scope(programs):
        first=mt.wrapper_minimise_tracking_error(**inputs)
        original=first[0].copy()
        first[1].constraints.benchmark_weights.iloc[0]=-10.
        second=mt.wrapper_minimise_tracking_error(**inputs)
    pd.testing.assert_series_equal(second[0],original,check_exact=True)


def test_infeasible_update_cannot_return_a_previous_portfolio():
    """A valid graph with impossible new bounds retains native failure attribution."""
    inputs, preparations = _inputs(), _CovarianceCache()
    programs = _TrackingErrorProgramCache(preparations)
    inputs['constraints']=inputs['constraints'].copy(min_weights=pd.Series(0.,index=inputs['pd_covar'].index),
        max_weights=pd.Series(1.,index=inputs['pd_covar'].index))
    with _covariance_reuse_scope(preparations), _program_reuse_scope(programs):
        mt.wrapper_minimise_tracking_error(**inputs)
        failed=mt.wrapper_minimise_tracking_error(**dict(inputs,constraints=inputs['constraints'].copy(
            max_weights=pd.Series(.1,index=inputs['pd_covar'].index))))
    assert not failed[1].accepted and failed[1].status=='infeasible'
    assert not programs.entries


def test_complete_real_search_preserves_every_checkpoint_and_logical_decision():
    """Program reuse must exactly replicate the preceding projection/preparation mode."""
    problem=_problem(max_trades=3)
    controls=b.ExecutionBranchConfig(reuse_projection_solves=True,allow_support_plateaus=True,
        preserve_guarded_path=True,max_sized_ticket_increase=0,max_turnover_increase_bp=0)
    before=b.solve_branched_execution(problem,controls)
    after=b.solve_branched_execution(problem,replace(controls,reuse_projection_programs=True))
    assert list(before.checkpoints)==list(after.checkpoints)
    for name,state in before.checkpoints.items():
        other=after.checkpoints[name]
        pd.testing.assert_series_equal(state.weights,other.weights,check_exact=True)
        pd.testing.assert_frame_equal(state.trade_table,other.trade_table,check_exact=True)
        pd.testing.assert_frame_equal(state.outcome.residuals_frame(),other.outcome.residuals_frame(),check_exact=True)
    pd.testing.assert_frame_equal(before.attempts.drop(columns='seconds'),after.attempts.drop(columns='seconds'),check_exact=True)
    assert after.summary['search_program_builds']==1
    assert (after.summary['search_program_cache_hits']
            == after.summary['search_numerical_projection_calls']-1)


@pytest.mark.parametrize('value',[None,1,'yes'])
def test_program_control_requires_boolean(value):
    """Reject implicit truthiness for the opt-in model-reuse policy."""
    with pytest.raises(ValueError,match='reuse_projection_programs must be boolean'):
        b.ExecutionBranchConfig(reuse_projection_programs=value)


def test_program_control_requires_projection_reuse():
    """Keep model ownership and lifetime attached to a single bounded projection cache."""
    with pytest.raises(ValueError,match='requires reuse_projection_solves'):
        b.ExecutionBranchConfig(reuse_projection_programs=True)


@pytest.mark.parametrize('unsupported',['solver','geometry','bounds','metadata'])
def test_unsupported_requests_use_fresh_compiler(unsupported):
    """Optional graph reuse cannot make unsupported inputs depend on parameterization."""
    inputs, preparations = _inputs(), _CovarianceCache()
    programs = _TrackingErrorProgramCache(preparations)
    covariance=inputs['pd_covar'].to_numpy()
    from optimalportfolios.optimization.covar_factorization import factorize_covariance
    factor=preparations.factorize(factorize_covariance,covariance,inputs['pd_covar'].index)
    specification=inputs['constraints'].copy(benchmark_weights=inputs['benchmark_weights'])
    solver='CLARABEL'
    if unsupported=='solver':
        solver='SCS'
    elif unsupported=='geometry':
        factor=factorize_covariance(covariance)
    elif unsupported=='bounds':
        specification=specification.copy(max_weights=specification.max_weights.iloc[::-1])
    else:
        specification.benchmark_weights.attrs['unserializable']=lambda:None
    programs.acquire(mt._build_tracking_error_problem,specification,factor.covar,factor,solver)
    assert programs.builds==1 and programs.bypasses==1 and not programs.entries


def test_unused_parameters_are_detected_and_fresh_problem_is_used():
    """A delegated compiler that drops parameter bounds cannot create a stale graph."""
    from optimalportfolios.optimization.covar_factorization import factorize_covariance
    inputs, preparations = _inputs(), _CovarianceCache()
    programs=_TrackingErrorProgramCache(preparations)
    factor=preparations.factorize(factorize_covariance,inputs['pd_covar'].to_numpy(),inputs['pd_covar'].index)
    specification=inputs['constraints'].copy(benchmark_weights=inputs['benchmark_weights'])
    def broken_proxy_builder(specification,covariance,factor):
        """Simulate the original missing exposure-delegation defect without editing code."""
        return mt._build_tracking_error_problem(
            getattr(specification,'specification',specification),covariance,factor)
    (_,problem),key=programs.acquire(broken_proxy_builder,specification,factor.covar,factor,'CLARABEL')
    assert key is None and not problem.parameters() and programs.builds==2
    assert programs.bypasses==1 and not programs.entries


@pytest.mark.parametrize('error',[cvx.error.SolverError,RuntimeError])
def test_solver_errors_evict_graph_and_never_return_stale_primal_values(monkeypatch,error):
    """A solve error after a successful call cannot masquerade as its earlier solution."""
    inputs,preparations=_inputs(),_CovarianceCache()
    programs=_TrackingErrorProgramCache(preparations)
    with _covariance_reuse_scope(preparations),_program_reuse_scope(programs):
        mt.wrapper_minimise_tracking_error(**inputs)
        def failed(*args,**kwargs):
            """Raise after the graph is populated with an earlier accepted solution."""
            raise error('intentional failure')
        monkeypatch.setattr(cvx.Problem,'solve',failed)
        if error is RuntimeError:
            with pytest.raises(RuntimeError,match='intentional failure'):
                mt.wrapper_minimise_tracking_error(**inputs)
        else:
            actual=mt.wrapper_minimise_tracking_error(**inputs)
            assert not actual[1].accepted and actual[1].status=='solver_error'
            np.testing.assert_array_equal(actual[1].weights,inputs['weights_0'].to_numpy())
    assert not programs.entries


def test_nested_program_scopes_restore_and_baseline_bypass_is_explicit():
    """Nested searches and bypass scopes cannot consume another owner's graph."""
    inputs,preparations=_inputs(),_CovarianceCache()
    outer,inner=_TrackingErrorProgramCache(preparations),_TrackingErrorProgramCache(preparations)
    with _covariance_reuse_scope(preparations),_program_reuse_scope(outer):
        mt.wrapper_minimise_tracking_error(**inputs)
        with pytest.raises(RuntimeError),_program_reuse_scope(inner):
            mt.wrapper_minimise_tracking_error(**inputs)
            raise RuntimeError('leave inner scope')
        mt.wrapper_minimise_tracking_error(**inputs)
        with _program_reuse_scope(None):
            mt.wrapper_minimise_tracking_error(**inputs)
    assert (outer.builds,outer.hits,inner.builds)==(1,1,1)
