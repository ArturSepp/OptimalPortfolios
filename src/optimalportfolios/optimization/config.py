"""Solver configuration shared across all optimisation solvers.

Encapsulates backend-agnostic solver parameters: solver name, verbosity,
constraint rescaling, weights drift, input/failed-solve diagnostics, bounded
constraint relaxation, and covariance factorization. Solver-specific
parameters (for example SciPy ftol/maxiter) remain direct arguments on the
lowest-level solver functions.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class OptimiserConfig:
    """Solver configuration shared across all optimisation solvers.

    Attributes:
        solver: CVXPY solver name. Ignored by the scipy and risk-budgeting
            solvers which use fixed backends (SLSQP and CCD/ADMM respectively).
        verbose: If True, print solver diagnostics (CVXPY output,
            scipy disp, risk-budgeting constraint slack).
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
        diagnose_infeasibility: In the alpha-over-tracking-error wrapper, if True
            (default), a rejected solve runs a second diagnosis on the same log
            channel. An infeasible solve gets an elastic box/group-subset model;
            a numerical blow-up gets a covariance-conditioning report. Other
            solver wrappers currently retain this field for configuration
            compatibility but do not consume it.
        validate_inputs: In the alpha-over-tracking-error wrapper, if True
            (default), run a cheap pre-solve covariance, box/group reachability,
            and benchmark input contract. Other solver wrappers currently retain
            this field for configuration compatibility but do not consume it.
        max_constraint_relaxation: If set, the frozen-overhang group-bound
            relaxation escalates to an ERROR log when a single relaxation exceeds
            this magnitude (e.g. 0.02), surfacing a large silent widening that a
            small drift would not cause. None (default) applies no magnitude
            bound; the relaxation is still logged (at INFO) and tallied.
        factorize_covar: If True (default), compatible CVXPY solvers use one
            controlled eigendecomposition per solve and reuse the resulting
            covariance factor in objective and constraint risk expressions.
            Set False to use the legacy ``quad_form`` formulation. Scipy and
            dedicated risk-budgeting backends ignore this setting.
    """
    solver: str = 'CLARABEL'
    verbose: bool = False
    apply_total_to_good_ratio: bool = False
    use_drifted_weights_0: bool = True
    diagnose_infeasibility: bool = True
    validate_inputs: bool = True
    max_constraint_relaxation: Optional[float] = None
    factorize_covar: bool = True
