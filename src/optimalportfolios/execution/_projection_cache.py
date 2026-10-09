"""Reuse successful resolved projections within one bounded execution search."""
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass, field
import pickle

import numpy as np

from optimalportfolios.optimization._covariance_cache import (
    _CovarianceCache, _covariance_reuse_scope,
)
from optimalportfolios.optimization._tracking_error_program import (
    _TrackingErrorProgramCache, _program_reuse_scope,
)


@dataclass
class _ProjectionCache:
    """Keep exact resolved argument bytes and detached outcomes for this search only.

    Serialize the complete numerical-wrapper argument dictionary, including asset
    order, covariance, benchmark/current weights, resolved constraints, optimiser
    configuration and context. Keys are full bytes, not support masks or hashes.
    Equal fixed-size byte blocks share storage without dropping any key bytes.
    Serialization metadata may cause conservative misses. Never deserialize keys
    or persist the cache. Unserializable inputs and non-optimal outcomes bypass it.
    Filtered numerical covariance preparations share immutable arrays; all other
    outcome data stay detached. The common baseline remains outside this cache.
    """

    numerical_calls: int = 0
    hits: int = 0
    entries: dict = field(default_factory=dict)
    blocks: dict = field(default_factory=dict)
    preparations: _CovarianceCache = field(default_factory=_CovarianceCache)
    programs: _TrackingErrorProgramCache | None = None

    def enable_program_reuse(self):
        """Attach optional problem graphs to this cache's immutable preparation lifetime."""
        self.programs = _TrackingErrorProgramCache(self.preparations)

    def _copy_result(self, result):
        """Detach mutable results while retaining only verified immutable preparation."""
        factor = getattr(result[1], 'covar_factorization', None)
        memo = {id(factor): factor} if self.preparations.contains(factor) else None
        return deepcopy(result, memo)

    def project(self, projection, inputs):
        """Return a detached hit or execute and retain a successful fresh projection."""
        try:
            serialized = pickle.dumps(inputs, protocol=5)
            key = (projection, tuple(serialized[i:i+4096]
                                     for i in range(0, len(serialized), 4096)))
        except (pickle.PickleError, TypeError, AttributeError):
            key = None
        if key is not None and key in self.entries:
            self.hits += 1
            return self._copy_result(self.entries[key])
        self.numerical_calls += 1
        with _covariance_reuse_scope(self.preparations), _program_reuse_scope(self.programs):
            result = projection(**inputs)
        weights, outcome = result
        if (key is not None and outcome.accepted and outcome.compliant
                and outcome.status == 'optimal'
                and np.isfinite(weights.to_numpy()).all()
                and np.isfinite(outcome.weights).all()):
            key = (projection, tuple(self.blocks.setdefault(block, block) for block in key[1]))
            self.entries[key] = self._copy_result(result)
        return result


_ACTIVE_PROJECTION_CACHE = ContextVar('execution_projection_cache', default=None)


@contextmanager
def _projection_reuse_scope(cache):
    """Isolate one search's cache and restore enclosing state on every exit."""
    token = _ACTIVE_PROJECTION_CACHE.set(cache)
    try:
        with _covariance_reuse_scope(None), _program_reuse_scope(None):
            yield
    finally:
        _ACTIVE_PROJECTION_CACHE.reset(token)


def _project(projection, **inputs):
    """Apply optional reuse after constraint resolution, before numerical solving."""
    cache = _ACTIVE_PROJECTION_CACHE.get()
    if cache is None:
        return projection(**inputs)
    return cache.project(projection, inputs)
