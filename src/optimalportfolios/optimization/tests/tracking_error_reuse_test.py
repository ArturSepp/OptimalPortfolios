"""Shared minimum-tracking-error cache isolation and fresh-solve parity contracts."""
from copy import deepcopy

import cvxpy as cvx
import numpy as np
import pandas as pd
import pytest

from optimalportfolios.optimization.config import OptimiserConfig
from optimalportfolios.optimization.constraints import Constraints
from optimalportfolios.optimization import covar_factorization as f
from optimalportfolios.optimization._covariance_cache import (
    _CovarianceCache, _covariance_reuse_scope, _factorize_with_reuse,
)
from optimalportfolios.optimization._tracking_error_program import (
    _TrackingErrorProgramCache, _program_reuse_scope,
)
from optimalportfolios.optimization.general import minimum_tracking_error as mt


def _inputs():
    """Build a standalone four-asset optimizer request with no execution-layer dependency."""
    assets = pd.Index(['A', 'B', 'C', 'D'])
    return dict(
        pd_covar=pd.DataFrame(np.diag([.04, .03, .02, .01]), index=assets, columns=assets),
        benchmark_weights=pd.Series([.4, .3, .2, .1], index=assets),
        weights_0=pd.Series([.1, .2, .3, .4], index=assets),
        optimiser_config=OptimiserConfig(solver='CLARABEL'),
        constraints=Constraints(min_weights=pd.Series(0., index=assets),
                                max_weights=pd.Series(1., index=assets)),
        context='shared optimizer cache audit',
    )



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


@pytest.mark.parametrize('change', ['matrix', 'order', 'dimension', 'callable'])
def test_preparation_identity_checks_filtered_values_order_dimension_and_callable(change):
    """A covariance cache cannot reuse a different numerical geometry or implementation."""
    cache, values, assets = _CovarianceCache(), np.diag([.04, .02]), pd.Index(['A', 'B'])
    first = cache.factorize(f.factorize_covariance, values, assets)
    assert cache.factorize(f.factorize_covariance, values.copy(), assets.copy()) is first
    changed_values, changed_assets, function = values.copy(), assets.copy(), f.factorize_covariance
    if change == 'matrix':
        changed_values[0, 0] += .01
    elif change == 'order':
        changed_assets = assets[::-1]
    elif change == 'dimension':
        changed_values, changed_assets = values[:1, :1], assets[:1]
    else:
        def function(matrix):
            """Delegate through a distinct callable to exercise preparation identity."""
            return f.factorize_covariance(matrix)
    second = cache.factorize(function, changed_values, changed_assets)
    assert second is not first and (cache.factorizations, cache.hits) == (2, 1)
    assert cache.contains(first) and not cache.contains(None)


def test_shared_arrays_cannot_be_written_or_made_writeable():
    """Bytes-backed covariance/factor arrays protect every consumer of a shared result."""
    cache = _CovarianceCache()
    result = cache.factorize(f.factorize_covariance, np.diag([.04, .02]), pd.Index(['A', 'B']))
    for array in (result.covar, result.factor):
        with pytest.raises(ValueError):
            array[0, 0] = 999
        with pytest.raises(ValueError):
            array.setflags(write=True)
    np.testing.assert_array_equal(result.covar, f.factorize_covariance(np.diag([.04, .02])).covar)


def test_failed_preparation_is_retried_without_retained_identity():
    """An indefinite risk matrix remains a factorization error on every attempt."""
    cache = _CovarianceCache()
    for _ in range(2):
        with pytest.raises(ValueError, match='materially indefinite'):
            cache.factorize(f.factorize_covariance, np.diag([.04, -.02]), pd.Index(['A', 'B']))
    assert not cache.entries and (cache.factorizations, cache.hits) == (2, 0)


def test_unserializable_asset_metadata_bypasses_preparation_reuse():
    """Optional reuse preserves factorization when an asset key cannot be serialized."""
    class Unserializable:
        """Supply an intentionally nonserializable identity to exercise safe bypass."""
        def __reduce__(self):
            """Reject serialization without affecting numerical factorization."""
            raise TypeError('unsupported identity')
    cache = _CovarianceCache()
    for _ in range(2):
        cache.factorize(f.factorize_covariance, np.diag([.04, .02]), Unserializable())
    assert not cache.entries and cache.factorizations == 2


def test_preparation_scope_restores_after_exception_and_explicit_bypass():
    """Nested and bypass scopes cannot reuse another decision's preparation."""
    outer, inner, values, assets = (
        _CovarianceCache(), _CovarianceCache(), np.eye(2), pd.Index(['A', 'B']))
    with _covariance_reuse_scope(outer):
        first = _factorize_with_reuse(f.factorize_covariance, values, assets)
        with pytest.raises(RuntimeError), _covariance_reuse_scope(inner):
            _factorize_with_reuse(f.factorize_covariance, values, assets)
            raise RuntimeError('leave scope')
        assert _factorize_with_reuse(f.factorize_covariance, values, assets) is first
        with _covariance_reuse_scope(None):
            assert _factorize_with_reuse(f.factorize_covariance, values, assets) is not first
    assert _factorize_with_reuse(f.factorize_covariance, values, assets) is not first
    assert (outer.factorizations, outer.hits, inner.factorizations) == (1, 1, 1)
