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
        use_drifted_weights_0: If True (default), every rolling optimiser
            drifts the previous-period weights to the current rebalance
            date using realised price returns before passing them as
            weights_0 to the next single-date optimisation. This makes
            turnover constraints and transaction-cost penalties act on
            the actual current holdings rather than on the stale target
            weights, matching the convention used in live optimisation.
            Set False to reproduce the legacy behaviour where the prior
            target is reused as-is (useful for ablation studies).
            Drift falls back silently to the legacy behaviour when prices
            are unavailable; see ``apply_drift_to_weights_0`` for the
            full set of gates.
    """
    solver: str = 'CLARABEL'
    verbose: bool = False
    apply_total_to_good_ratio: bool = False
    use_drifted_weights_0: bool = True
