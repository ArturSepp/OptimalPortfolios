"""Exact covariance identity, immutable sharing and retained key-storage regressions."""
from copy import deepcopy
from dataclasses import replace
import pickle

import numpy as np
import pandas as pd
import pytest

from optimalportfolios.execution import branches as b
from optimalportfolios.execution._projection_cache import _ProjectionCache
from optimalportfolios.execution.tests.projection_reuse_test import _inputs, _stub
from optimalportfolios.execution.tests.improvement_test import _problem
from optimalportfolios.optimization import covar_factorization as f
from optimalportfolios.optimization._covariance_cache import (
    _CovarianceCache, _covariance_reuse_scope, _factorize_with_reuse,
)
from optimalportfolios.optimization.general import minimum_tracking_error as mt


def test_key_blocks_preserve_complete_identity_and_store_common_payload_once():
    """Different requests retain full exact keys with deduplicated common matrix bytes."""
    inputs, cache = _inputs(), _ProjectionCache()
    labels = [f'asset_{i}' for i in range(150)]
    inputs['pd_covar'] = pd.DataFrame(np.eye(150), index=labels, columns=labels)
    for i in range(8):
        cache.project(_stub, dict(inputs, context=f'request {i}'))
    blocks = {id(block): block for key in cache.entries for block in key[1]}
    assert all(isinstance(block, bytes) for block in blocks.values())
    assert sum(map(len, blocks.values())) < 2 * len(pickle.dumps(inputs, protocol=5))
    for key in cache.entries:
        assert pickle.loads(b''.join(key[1]))['pd_covar'].equals(inputs['pd_covar'])
    cache.project(_stub, dict(inputs, context='request 0'))
    assert (cache.numerical_calls, cache.hits) == (8, 1)


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


def test_filtered_universes_and_changed_matrices_use_independent_preparations(monkeypatch):
    """The real wrapper prepares its filtered matrix, then audits new constraints per request."""
    inputs, cache, calls = _inputs(), _ProjectionCache(), []
    original = mt.factorize_covariance
    def counted(matrix):
        """Count real decompositions independently of cache telemetry."""
        calls.append(matrix.copy())
        return original(matrix)
    monkeypatch.setattr(mt, 'factorize_covariance', counted)
    inputs['constraints'] = replace(inputs['constraints'], min_weights=None, max_weights=None)
    first = cache.project(mt.wrapper_minimise_tracking_error, inputs)
    second = cache.project(mt.wrapper_minimise_tracking_error, dict(inputs, context='new request'))
    assert len(calls) == 1
    assert first[1].covar_factorization is second[1].covar_factorization
    filtered = dict(inputs, context='filtered', inclusion_indicators=pd.Series(
        [1., 1., 0., 1.], index=inputs['pd_covar'].index))
    cache.project(mt.wrapper_minimise_tracking_error, filtered)
    assert len(calls) == 2 and calls[-1].shape == (3, 3)
    changed = deepcopy(inputs)
    changed['pd_covar'].iloc[0, 0] += .001
    cache.project(mt.wrapper_minimise_tracking_error, changed)
    assert len(calls) == 3
    first[0].iloc[0] = -100
    hit = cache.project(mt.wrapper_minimise_tracking_error, inputs)
    assert hit[0].iloc[0] != -100 and hit[1].covar_factorization is second[1].covar_factorization


@pytest.mark.parametrize('enabled', [False, True])
def test_factorization_option_and_original_psd_telemetry_are_preserved(enabled):
    """Repair policy and metadata exactly reproduce fresh numerical preparation."""
    inputs, cache = _inputs(), _ProjectionCache()
    inputs['pd_covar'].iloc[:, :] = .04
    inputs['optimiser_config'] = replace(inputs['optimiser_config'], factorize_covar=enabled)
    reference = mt.wrapper_minimise_tracking_error(**inputs)
    actual = cache.project(mt.wrapper_minimise_tracking_error, inputs)
    pd.testing.assert_series_equal(actual[0], reference[0], check_exact=True)
    pd.testing.assert_frame_equal(
        actual[1].residuals_frame(), reference[1].residuals_frame(), check_exact=True)
    if enabled:
        left, right = actual[1].covar_factorization, reference[1].covar_factorization
        for name in left.__dataclass_fields__:
            np.testing.assert_equal(getattr(left, name), getattr(right, name))
        assert cache.preparations.factorizations == 1
    else:
        assert actual[1].covar_factorization is None and not cache.preparations.entries


def test_real_branch_search_preserves_all_paths_and_shares_only_immutable_preparation(monkeypatch):
    """Uncached and reused real searches match every checkpoint and logical decision."""
    problem, calls = _problem(max_trades=3), []
    config = b.ExecutionBranchConfig(allow_support_plateaus=True, preserve_guarded_path=True,
        max_sized_ticket_increase=0, max_turnover_increase_bp=0)
    original = mt.factorize_covariance
    def counted(matrix):
        """Independently count actual preparation work."""
        calls.append(True)
        return original(matrix)
    monkeypatch.setattr(mt, 'factorize_covariance', counted)
    reference = b.solve_branched_execution(problem, config)
    count_before = len(calls)
    calls.clear()
    actual = b.solve_branched_execution(problem, replace(config, reuse_projection_solves=True))
    assert len(calls) == 2 < count_before
    assert actual.summary['search_covariance_factorizations'] == 1
    assert (actual.summary['search_covariance_cache_hits']
            == actual.summary['search_numerical_projection_calls']-1)
    columns = reference.attempts.columns.drop('seconds')
    pd.testing.assert_frame_equal(
        actual.attempts[columns], reference.attempts[columns], check_exact=True)
    assert list(actual.checkpoints) == list(reference.checkpoints)
    for name, value in actual.checkpoints.items():
        expected = reference.checkpoints[name]
        pd.testing.assert_series_equal(value.weights, expected.weights, check_exact=True)
        pd.testing.assert_frame_equal(value.trade_table, expected.trade_table, check_exact=True)
        pd.testing.assert_frame_equal(
            value.outcome.residuals_frame(), expected.outcome.residuals_frame(), check_exact=True)


def test_projection_bypass_also_isolates_an_enclosing_preparation_scope():
    """An uncached baseline cannot consume a caller's active preparation cache."""
    from optimalportfolios.execution._projection_cache import _project, _projection_reuse_scope
    outer = _CovarianceCache()
    with _covariance_reuse_scope(outer), _projection_reuse_scope(None):
        _project(mt.wrapper_minimise_tracking_error, **_inputs())
    assert outer.factorizations == 0 and not outer.entries
