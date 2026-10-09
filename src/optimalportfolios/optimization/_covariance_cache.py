"""Private exact covariance preparation reuse within an execution projection scope."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
import pickle

import numpy as np


def _immutable_array(values):
    """Keep array values in immutable bytes so writeability cannot be re-enabled."""
    return np.frombuffer(values.tobytes(order='C'), dtype=values.dtype).reshape(values.shape)


@dataclass
class _CovarianceCache:
    """Reuse exact filtered matrix/order/callable identities and immutable factors.

    The owning execution search determines lifetime. Default factorization and
    PSD-repair policy stay with the supplied numerical callable. Full byte keys
    retain equality checks; failed preparations and unserializable identities
    bypass reuse. No solver outcomes, constraints or mutable weights live here.
    """

    factorizations: int = 0
    hits: int = 0
    entries: dict = field(default_factory=dict)

    def factorize(self, function, values, assets):
        """Compute or retrieve an immutable factor for this exact numerical geometry."""
        try:
            key = (function, pickle.dumps(assets, protocol=5), values.shape,
                   values.dtype.str, values.tobytes(order='C'))
        except (pickle.PickleError, TypeError, AttributeError):
            key = None
        if key is not None and key in self.entries:
            self.hits += 1
            return self.entries[key]
        self.factorizations += 1
        result = function(values)
        if key is None:
            return result
        shared = replace(result, covar=_immutable_array(result.covar),
                         factor=_immutable_array(result.factor))
        self.entries[key] = shared
        return shared

    def contains(self, factor):
        """Allow sharing only a factor retained and made immutable by this cache."""
        return any(value is factor for value in self.entries.values())


_ACTIVE_COVARIANCE_CACHE = ContextVar('execution_covariance_cache', default=None)


@contextmanager
def _covariance_reuse_scope(cache):
    """Restore the enclosing preparation cache on ordinary and exceptional exits."""
    token = _ACTIVE_COVARIANCE_CACHE.set(cache)
    try:
        yield
    finally:
        _ACTIVE_COVARIANCE_CACHE.reset(token)


def _factorize_with_reuse(function, values, assets):
    """Apply optional reuse after wrapper filtering and low-level input validation."""
    cache = _ACTIVE_COVARIANCE_CACHE.get()
    if cache is None:
        return function(values)
    return cache.factorize(function, values, assets)
