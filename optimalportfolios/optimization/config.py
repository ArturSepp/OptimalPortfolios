"""Solver configuration shared across all optimisation solvers.

Encapsulates backend-agnostic solver parameters: solver name (for CVXPY
solvers), verbosity, and constraint rescaling. Solver-specific parameters
(e.g., scipy ftol/maxiter) remain as direct arguments on the lowest-level
solver functions.
"""
from dataclasses import dataclass
from typing import Optional


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
        diagnose_infeasibility: If True (default), when a solve is rejected
            (infeasible, or a numerical blow-up the solver mislabelled as
            optimal) the optimiser runs a second, cheap analysis and logs it
            on the same channel as the rejection. An infeasible solve runs an
            elastic minimum-violation LP that reports which box / group bounds
            must relax, and by how much, to make that rebalance solvable while
            holding full investment and long-only fixed; a numerical blow-up
            runs a covariance-conditioning report instead. This costs one extra
            LP per rejected rebalance (not per rebalance), so the overhead is
            confined to the dates that already failed. Set False to skip the
            diagnosis and keep only the one-line rejection notice.
        validate_inputs: If True (default), run a cheap pre-solve input contract
            at the wrapper entry — validates the covariance (finite, symmetric,
            right dimension), flags ill-conditioning, checks constraint
            self-consistency (box caps reach full investment, group bounds
            reachable, benchmark within bounds), and notes dropped assets — so a
            broken or structurally infeasible input is flagged before the solve
            rather than discovered as a failed solve. Set False to skip it.
        max_constraint_relaxation: If set, the frozen-overhang group-bound
            relaxation escalates to an ERROR log when a single relaxation exceeds
            this magnitude (e.g. 0.02), surfacing a large silent widening that a
            small drift would not cause. None (default) applies no magnitude
            bound; the relaxation is still logged (at INFO) and tallied.
    """
    solver: str = 'CLARABEL'
    verbose: bool = False
    apply_total_to_good_ratio: bool = False
    use_drifted_weights_0: bool = True
    diagnose_infeasibility: bool = True
    validate_inputs: bool = True
    max_constraint_relaxation: Optional[float] = None
