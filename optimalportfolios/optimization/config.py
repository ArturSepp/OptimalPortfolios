"""Solver configuration shared across all optimisation solvers.

Encapsulates backend-agnostic solver parameters: solver name (for CVXPY
solvers), verbosity, and constraint rescaling. Solver-specific parameters
(e.g., scipy ftol/maxiter) remain as direct arguments on the lowest-level
solver functions.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class OptimiserConfig:
    """Solver configuration shared across all optimisation solvers.

    Attributes:
        solver: CVXPY solver name. Ignored by scipy and pyrb solvers
            which use fixed backends (SLSQP and ADMM respectively).
        verbose: If True, print solver diagnostics (CVXPY output,
            scipy disp, pyrb constraint slack).
        apply_total_to_good_ratio: If True, rescale constraints and risk
            budgets proportionally when assets are excluded due to NaN or
            zero variance. This preserves the intended allocation across
            the valid asset subset.
    """
    solver: str = 'CLARABEL'
    verbose: bool = False
    apply_total_to_good_ratio: bool = False
