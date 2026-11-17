"""Call-local deadline checks shared by Ranked search and its QP retries."""

from contextvars import ContextVar
from time import perf_counter


_active = ContextVar("ranked_execution_deadline", default=None)


class _DeadlineReached(Exception):
    """Unwind the current search branch to its best audited delivery."""


class _SearchDeadline:
    """Retain the best audited portfolio independently of temporary search restarts."""

    def __init__(self, seconds, tolerance):
        """Start the method clock before ranking or compilation."""
        self.started = perf_counter()
        self.deadline = self.started + seconds
        self.tolerance = tolerance
        self.best = None
        self.value = float("inf")
        self.support = {}
        self.diagnostics = {}
        self.phase = "initialisation"

    def __enter__(self):
        """Install a context-local budget for this call only."""
        self.token = _active.set(self)
        return self

    def __exit__(self, *exc):
        """Restore the caller's budget after normal or interrupted completion."""
        _active.reset(self.token)

    def retain(self, weights, value, support):
        """Keep an already-audited candidate according to the existing objective tolerance."""
        if weights is not None and value < self.value - self.tolerance:
            self.best = weights.copy()
            self.value = value
            self.support = dict(support)


def checkpoint(phase):
    """Stop before new work when a finite Ranked deadline has elapsed."""
    budget = _active.get()
    if budget is not None:
        budget.phase = phase
        if perf_counter() >= budget.deadline:
            raise _DeadlineReached


def retain_audited(weights, value, support, diagnostics=None):
    """Expose audited progress to a finite-deadline call across search branches."""
    budget = _active.get()
    if budget is not None:
        if diagnostics is not None:
            budget.diagnostics = diagnostics
        budget.retain(weights, value, support)
