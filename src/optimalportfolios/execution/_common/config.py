"""Solver-independent Operational Exact configuration."""
from dataclasses import dataclass
from typing import Optional
import numpy as np

BASE_CONSTRAINT_SET = 'Min / Max'
EXECUTION_CONSTRAINT_SET = 'Min_execution / Max_execution'
CONSTRAINT_SETS = (BASE_CONSTRAINT_SET, EXECUTION_CONSTRAINT_SET)

@dataclass(frozen=True)
class ExactExecutionConfig:
    """Immutable controls for joint position-and-ticket execution selection.

    ``Exact Execution`` is the desk-facing name for the convex mixed-integer
    quadratic program (MIQP). Policy-level holding and ticket thresholds remain
    on :class:`ExecutionPolicyConfig`; this object contains only the additional
    position budget, retained-breach semantics and deterministic solver controls.

    Attributes:
        max_positions: Hard maximum number of terminal non-cash positions.
        repair_on_touch: Require a traded retained-breach row to finish inside
            its true product interval.
        alpha_tradeoff: Linear alpha coefficient in the exact objective.
            ``0.0001`` is the production default; zero gives pure minimum
            tracking error.
        solver: CVXPY mixed-integer solver name. Exact Execution currently
            requires MOSEK so its deterministic branch controls can be audited.
        relative_gap_tolerance: Relative optimality-gap termination tolerance.
        absolute_gap_tolerance: Absolute objective-gap termination tolerance.
        max_branches: Deterministic branch-and-bound work limit.
        max_solve_seconds: Optional MOSEK optimisation-time safety limit.
        accept_audited_incumbent: Accept an unproved integer incumbent only
            when every existing hard audit passes.
        seed: Deterministic MOSEK mixed-integer seed.
        sub_asset_class_penalty: Quadratic penalty on level-2 allocation
            shortfalls and excesses. None keeps every group bound hard.
        constraint_set: Level-2 workbook bounds used only by execution. The
            continuous optimiser always uses ``Min`` and ``Max``.
    """

    max_positions: int = 30
    repair_on_touch: bool = True
    alpha_tradeoff: float = 0.0001
    solver: str = 'MOSEK'
    relative_gap_tolerance: float = 1e-4
    absolute_gap_tolerance: float = 1e-12
    max_branches: int = 500_000
    max_solve_seconds: Optional[float] = None
    accept_audited_incumbent: bool = True
    seed: int = 0
    sub_asset_class_penalty: Optional[float] = None
    constraint_set: str = BASE_CONSTRAINT_SET

    def __post_init__(self) -> None:
        """Validate deterministic Operational Exact solver controls."""
        if not isinstance(self.solver, str):
            raise ValueError('solver must be a string')
        object.__setattr__(self, 'solver', self.solver.upper())
        if (isinstance(self.max_positions, (bool, np.bool_))
                or not isinstance(self.max_positions, (int, np.integer))
                or self.max_positions <= 0):
            raise ValueError('max_positions must be positive')
        if not np.isfinite(self.alpha_tradeoff) or self.alpha_tradeoff < 0.0:
            raise ValueError('alpha_tradeoff must be finite and non-negative')
        if self.solver != 'MOSEK':
            raise ValueError('Exact Execution currently requires solver="MOSEK"')
        if (not np.isfinite(self.relative_gap_tolerance)
                or self.relative_gap_tolerance < 0.0):
            raise ValueError(
                'relative_gap_tolerance must be finite and non-negative'
            )
        if (not np.isfinite(self.absolute_gap_tolerance)
                or self.absolute_gap_tolerance < 0.0):
            raise ValueError(
                'absolute_gap_tolerance must be finite and non-negative'
            )
        if (isinstance(self.max_branches, (bool, np.bool_))
                or not isinstance(self.max_branches, (int, np.integer))
                or self.max_branches <= 0):
            raise ValueError('max_branches must be positive')
        if (self.max_solve_seconds is not None
                and (not np.isfinite(self.max_solve_seconds)
                     or self.max_solve_seconds <= 0.0)):
            raise ValueError(
                'max_solve_seconds must be finite and positive or None'
            )
        if not isinstance(self.accept_audited_incumbent, (bool, np.bool_)):
            raise ValueError('accept_audited_incumbent must be boolean')
        if (isinstance(self.seed, (bool, np.bool_))
                or not isinstance(self.seed, (int, np.integer))
                or self.seed < 0):
            raise ValueError('seed must be non-negative')
        if (self.sub_asset_class_penalty is not None
                and (not np.isfinite(self.sub_asset_class_penalty)
                     or self.sub_asset_class_penalty <= 0.0)):
            raise ValueError(
                'sub_asset_class_penalty must be finite and positive or None'
            )
        if self.constraint_set not in CONSTRAINT_SETS:
            raise ValueError(
                f'constraint_set must be one of {list(CONSTRAINT_SETS)}, got '
                f'{self.constraint_set!r}'
            )
