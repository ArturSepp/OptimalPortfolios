"""Context-local accounting of execution projection attempts, including rescue."""
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Optional


class _ProjectionBudgetExhausted(Exception):
    """Signal work exhaustion without classifying it as numerical infeasibility."""


@dataclass
class _ProjectionBudget:
    """Count attempted projections; the common production baseline is outside scope."""

    limit: Optional[int]
    used: int = 0

    @property
    def exhausted(self):
        """Report whether another projection would exceed the declared ceiling."""
        return self.limit is not None and self.used >= self.limit


_ACTIVE_BUDGETS = ContextVar('execution_projection_budgets', default=())


@contextmanager
def _projection_scope(budget):
    """Restore enclosing accounting even on exceptions; nested scopes share costs."""
    token = _ACTIVE_BUDGETS.set(_ACTIVE_BUDGETS.get()+(budget,))
    try:
        yield budget
    finally:
        _ACTIVE_BUDGETS.reset(token)


def _consume_projection():
    """Reserve one attempt before invoking the existing projection implementation."""
    budgets = _ACTIVE_BUDGETS.get()
    if any(budget.exhausted for budget in budgets):
        raise _ProjectionBudgetExhausted('post-baseline projection budget exhausted')
    for budget in budgets:
        budget.used += 1
