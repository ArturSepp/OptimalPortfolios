"""Opt-in execution scope retaining supplied group bounds without endpoint rounding."""
from contextlib import contextmanager
from contextvars import ContextVar

from optimalportfolios.execution.schema import WEIGHT_ZERO_TOLERANCE


_ORIGINAL_GROUP_BOUNDS = ContextVar('execution_original_group_bounds', default=False)


def _retain_original_group_bounds():
    """Report the current call's explicit strict group-bound policy."""
    return _ORIGINAL_GROUP_BOUNDS.get()


def _linear_feasibility_tolerance():
    """Separate the strict caller's linear audit from its unchanged ticket threshold."""
    return 1e-7 if _ORIGINAL_GROUP_BOUNDS.get() else WEIGHT_ZERO_TOLERANCE


@contextmanager
def _original_group_bounds_scope():
    """Keep original bounds throughout nested sizing/search and restore on exit."""
    token = _ORIGINAL_GROUP_BOUNDS.set(True)
    try:
        yield
    finally:
        _ORIGINAL_GROUP_BOUNDS.reset(token)
