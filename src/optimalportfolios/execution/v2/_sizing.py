"""Reuse CVXPY's fixed-QP layout without changing numerical sizing or search."""

from collections import OrderedDict

import numpy as np
import scipy.sparse as sp

from .model import SCALE

_LAYOUT_LIMIT = 64


def _same_data(compiled, supplied):
    """Require identical canonical coefficients and sparse structure before reuse."""
    for name, expected in supplied.items():
        actual = compiled.get(name)
        if sp.issparse(expected):
            if not sp.issparse(actual) or actual.shape != expected.shape:
                return False
            actual = actual.tocsc()
            if any(not np.array_equal(getattr(actual, field), getattr(expected, field))
                   for field in ('indptr', 'indices', 'data')):
                return False
        elif actual is None or not np.array_equal(actual, expected):
            return False
    return True


class _CachedSizingModel:
    """Expose the narrow solve_qp contract using a verified CVXPY solving chain.

    Only primal weights and native status are consumed. The template's objective
    offset and dual labels belong to its first support and are never reported.
    Each call creates a fresh native solver, including the existing numerical retry.
    """

    def __init__(self, template, data, reference, direction):
        """Bind fresh coefficients to a cached inverse map of the same dimensions."""
        self.template, self.data = template, data
        self.reference, self.direction = reference, direction
        self.optimisation = self.weights = self
        self.selectors = {}
        self.qp_diagnostics = {}

    @property
    def status(self):
        """Return CVXPY's native status after its ordinary inverse mapping."""
        return self.template[0].optimisation.status

    @property
    def value(self):
        """Recover funded weights in the original compiler's evaluation order."""
        displacement = self.template[0].displacement_bp.value
        if displacement is None:
            return None
        return (SCALE * self.reference + self.direction @ displacement) / SCALE

    def solve(self, solver, warm_start=False, **solver_options):
        """Use CVXPY's solver interface with unchanged options and no native reuse."""
        if solver != 'CLARABEL':
            raise ValueError('Cached sizing requires CLARABEL')
        model, _, chain, inverse = self.template
        solution = chain.solve_via_data(
            model.optimisation, self.data, warm_start=False, solver_opts=solver_options)
        model.optimisation.unpack_results(solution, chain, inverse)


class _FixedStateSizer:
    """Compile exact conditional-QP data, retaining the original graph as reference.

    build_model remains the compiler for Exact, guidance and unsupported layouts.
    Its repeated graph construction dominates ranked sizing, so this private path
    reproduces its funded-coordinate algebra and verifies each new layout against
    it. A mismatch falls back to that graph; no approximate compatibility is allowed.
    The cache is bounded, local to one ranked call, and stores no search decisions.
    """

    def __init__(self, problem, states, builder, solver):
        """Retain immutable inputs and the existing compiler/solver selection."""
        self.problem, self.states = problem, states
        self.builder, self.solver = builder, solver
        self.layouts = OrderedDict()
        self.diagnostics = dict(layouts=0, reuses=0, fallbacks=0)

    def __call__(self, fixed_states):
        """Build one equivalent sizing model without changing its state assignment."""
        problem = self.problem
        if self.solver != 'CLARABEL':
            self.diagnostics['fallbacks'] += 1
            return self.builder(problem, fixed_states=fixed_states)
        lower, upper = problem.lower.copy(), problem.upper.copy()
        for i, j in fixed_states.items():
            lower[i], upper[i] = self.states[i][j].lower, self.states[i][j].upper
        noncash = np.arange(len(lower)) != problem.cash
        indices = np.flatnonzero((lower != upper) & noncash)
        if not len(indices):
            self.diagnostics['fallbacks'] += 1
            return self.builder(problem, fixed_states=fixed_states)
        reference = problem.base.copy()
        fixed = (lower == upper) & noncash
        reference[fixed] = lower[fixed]
        reference[problem.cash] = problem.investment - reference[noncash].sum()
        direction = np.zeros((len(lower), len(indices)))
        direction[indices, np.arange(len(indices))] = 1.0
        direction[problem.cash, :] = -1.0
        matrices, endpoints = [], []

        def interval_rows(matrix, constant, lo, hi, canonical=None):
            """Keep the same rows, with CVXPY's sparse constant-evaluation order."""
            keep = np.any(matrix != 0.0, axis=1) | (constant < lo) | (constant > hi)
            if keep.any():
                value = constant if canonical is None else canonical
                matrices.extend((-matrix[keep], matrix[keep]))
                endpoints.extend((value[keep] - lo[keep], hi[keep] - value[keep]))

        interval_rows(direction, SCALE * reference, SCALE * lower, SCALE * upper)
        if len(problem.group_lower):
            lo = np.nextafter(problem.group_lower - problem.rules.audit_tolerance,
                              problem.group_lower)
            hi = np.nextafter(problem.group_upper + problem.rules.audit_tolerance,
                              problem.group_upper)
            interval_rows(
                problem.groups @ direction, problem.groups @ (SCALE * reference),
                SCALE * lo, SCALE * hi, sp.csc_array(problem.groups) @ (SCALE * reference))
        tickets = sum(self.states[i][j].ticket for i, j in fixed_states.items())
        optional = sum(self.states[i][j].ticket for i, j in fixed_states.items()
                       if not problem.mandatory[i])
        for cap, count in ((problem.rules.ticket_cap, tickets),
                           (problem.rules.discretionary_allowance, optional)):
            if cap is not None:
                matrices.append(np.zeros((1, len(indices))))
                endpoints.append(np.array([float(cap - count)]))
        active = reference - problem.target
        gradient = direction.T @ problem.matvec(active)
        if problem.covariance is not None:
            funded = direction.T @ problem.covariance @ direction
        else:
            loadings = direction.T @ problem.factor_loadings
            funded = (loadings @ problem.factor_covariance @ loadings.T
                      + direction.T @ (problem.residual_variance[:, None] * direction))
        # Multiply by the reciprocal, as CVXPY does when extracting this objective.
        supplied = dict(P=sp.csc_array(funded * (2 / SCALE)), c=2 * gradient,
                        A=sp.csc_array(np.vstack(matrices)), b=np.concatenate(endpoints))
        key = (len(indices), len(supplied['b']))
        if key not in self.layouts:
            model = self.builder(problem, fixed_states=fixed_states)
            compiled, chain, inverse = model.optimisation.get_problem_data(
                'CLARABEL', solver_opts={})
            self.diagnostics['layouts'] += 1
            compatible = _same_data(compiled, supplied)
            self.layouts[key] = (model, compiled, chain, inverse) if compatible else None
            if len(self.layouts) > _LAYOUT_LIMIT:
                self.layouts.popitem(last=False)
            if not compatible:
                self.diagnostics['fallbacks'] += 1
                return model
        else:
            self.layouts.move_to_end(key)
            if self.layouts[key] is None:
                self.diagnostics['fallbacks'] += 1
                return self.builder(problem, fixed_states=fixed_states)
            self.diagnostics['reuses'] += 1
        template = self.layouts[key]
        return _CachedSizingModel(template, dict(template[1], **supplied), reference, direction)
