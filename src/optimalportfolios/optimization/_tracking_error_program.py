"""Private box-parameterized minimum-TRE programs for one bounded execution search."""
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from dataclasses import dataclass, field, fields
import pickle

import cvxpy as cvx
import numpy as np
import pandas as pd

from optimalportfolios.optimization.constraints import Constraints


class _ParameterizedBoxes:
    """Delegate the existing constraint compiler with parameterized box sides only."""

    def __init__(self, specification, lower, upper):
        """Own detached fixed policy data and substitute only the two compiler inputs."""
        self.specification = deepcopy(specification)
        self.min_weights = lower
        self.max_weights = upper

    def __getattr__(self, name):
        """Read every unchanged policy field from the detached specification."""
        return getattr(self.specification, name)

    def set_cvx_all_constraints(self, **kwargs):
        """Use the base compiler without constructing an invalid policy dataclass."""
        return Constraints.set_cvx_all_constraints(self, **kwargs)

    def set_cvx_exposure_constraints(self, **kwargs):
        """Keep box parameters visible when the complete compiler delegates exposure."""
        return Constraints.set_cvx_exposure_constraints(self, **kwargs)


@dataclass
class _Program:
    """Own a CVXPY graph, its two optional parameters and immutable risk geometry."""

    variable: object
    problem: object
    lower: object
    upper: object
    factor: object

    def update(self, specification):
        """Refresh both box sides and clear all preceding primal variable values."""
        for parameter, name in ((self.lower, 'min_weights'), (self.upper, 'max_weights')):
            if parameter is not None:
                parameter.value = getattr(specification, name).to_numpy(dtype=float, copy=True)
        for variable in self.problem.variables():
            variable.value = None


@dataclass
class _TrackingErrorProgramCache:
    """Reuse DPP graphs only for exact fixed policies and owned immutable geometry.

    Bounds may change; covariance, asset order, benchmark/current weights, other
    constraints, solver and builder identity must match. Supported requests use
    the original CLARABEL/MOSEK options with native warm-start disabled. The owning
    projection cache supplies lifetime and immutable factor provenance.
    """

    preparations: object
    builds: int = 0
    hits: int = 0
    bypasses: int = 0
    entries: dict = field(default_factory=dict)

    def acquire(self, builder, specification, covariance, factor, solver):
        """Return a refreshed DPP graph or conservatively construct a fresh problem."""
        supported = (type(specification) is Constraints and solver.upper() in ('CLARABEL', 'MOSEK')
                     and self.preparations.contains(factor))
        for name in ('min_weights', 'max_weights'):
            bound = getattr(specification, name)
            if bound is not None:
                supported = supported and (isinstance(bound, pd.Series)
                    and bound.index.equals(specification.benchmark_weights.index)
                    and np.isfinite(bound.to_numpy()).all())
        key = None
        if supported:
            fixed = {item.name: getattr(specification, item.name) for item in fields(specification)
                     if item.name not in ('min_weights', 'max_weights')}
            fixed['box_presence'] = (specification.min_weights is not None,
                                     specification.max_weights is not None)
            try:
                key = (builder, id(factor), solver.upper(), pickle.dumps(fixed, protocol=5))
            except (pickle.PickleError, TypeError, AttributeError):
                key = None
        if key is None:
            self.bypasses += 1
            self.builds += 1
            return builder(specification, covariance, factor), None
        if key in self.entries:
            self.hits += 1
            program = self.entries[key]
        else:
            n = covariance.shape[0]
            lower = cvx.Parameter(n) if specification.min_weights is not None else None
            upper = cvx.Parameter(n) if specification.max_weights is not None else None
            proxy = _ParameterizedBoxes(specification, lower, upper)
            variable, problem = builder(proxy, covariance, factor)
            self.builds += 1
            expected = {id(value) for value in (lower, upper) if value is not None}
            actual = {id(value) for value in problem.parameters()}
            if not problem.is_dpp() or not expected.issubset(actual):
                self.bypasses += 1
                self.builds += 1
                return builder(specification, covariance, factor), None
            program = _Program(variable, problem, lower, upper, factor)
            self.entries[key] = program
        program.update(specification)
        return (program.variable, program.problem), key

    def discard(self, key):
        """Evict a graph after a rejected solve or unexpected error."""
        self.entries.pop(key, None)


_ACTIVE_PROGRAM_CACHE = ContextVar('execution_tracking_error_programs', default=None)


@contextmanager
def _program_reuse_scope(cache):
    """Restore enclosing model ownership on successful and exceptional exits."""
    token = _ACTIVE_PROGRAM_CACHE.set(cache)
    try:
        yield
    finally:
        _ACTIVE_PROGRAM_CACHE.reset(token)


def _get_program(builder, specification, covariance, factor, solver):
    """Keep default construction unchanged outside an opt-in execution scope."""
    cache = _ACTIVE_PROGRAM_CACHE.get()
    if cache is None:
        return builder(specification, covariance, factor), None
    return cache.acquire(builder, specification, covariance, factor, solver)


def _discard_program(key):
    """Discard only a model owned by the currently active scope."""
    cache = _ACTIVE_PROGRAM_CACHE.get()
    if cache is not None:
        cache.discard(key)
