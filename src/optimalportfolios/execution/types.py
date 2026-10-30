"""Immutable inputs for conditional, cash-funded execution optimisation."""

from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from types import MappingProxyType
from typing import Mapping

import numpy as np


def frozen_array(value, shape=None, dtype=float):
    """Copy into a bytes-backed array whose write flag cannot be re-enabled."""
    raw = np.asarray(value)
    if not np.isfinite(raw).all():
        raise ValueError("Nonfinite input")
    if dtype is bool and not np.isin(raw, [False, True]).all():
        raise ValueError("Masks must contain booleans")
    array = np.asarray(value, dtype=dtype)
    if shape is not None and array.shape != shape:
        raise ValueError(f"Expected shape {shape}, got {array.shape}")
    return np.frombuffer(array.tobytes(), dtype=array.dtype).reshape(array.shape)


def checked_cap(value):
    """Validate an integer capacity, with None denoting unlimited."""
    if value is not None and (
        isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0
    ):
        raise ValueError("Capacity must be a nonnegative integer or None")
    return None if value is None else int(value)


def immutable_metadata(value):
    """Deep-freeze JSON-compatible provenance, rejecting nonfinite numbers."""
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): immutable_metadata(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(immutable_metadata(v) for v in value)
    return json.loads(json.dumps(value, allow_nan=False))


@dataclass(frozen=True)
class ExecutionRules:
    """Hard ticket and holding rules in NAV fractions, with an explicit total cap.

    ``discretionary_allowance`` is additional to the frozen mandatory reservation.
    Exemptions remove only minimum ticket size, never counting or holding floors.
    """

    minimum_ticket: float = 0.0025
    holding_floor: object = 0.0
    ticket_cap: int | None = None
    mandatory_count_in_cap: bool = True
    exemption_mask: object = False
    discretionary_allowance: int | None = None
    zero_tolerance: float = 1e-8
    audit_tolerance: float = 1e-7

    def __post_init__(self):
        """Validate scalar conventions without inferring any exemptions."""
        checked_cap(self.ticket_cap)
        checked_cap(self.discretionary_allowance)
        if self.mandatory_count_in_cap is not True:
            raise ValueError("All mandatory tickets must count in the total cap")
        values = [self.minimum_ticket, self.zero_tolerance, self.audit_tolerance]
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError("Ticket size and tolerances must be positive")
        if self.minimum_ticket <= self.zero_tolerance:
            raise ValueError("Minimum ticket must exceed the zero convention")
        object.__setattr__(self, "holding_floor", frozen_array(self.holding_floor))
        object.__setattr__(self, "exemption_mask", frozen_array(self.exemption_mask, dtype=bool))


@dataclass(frozen=True)
class ExecutionProblem:
    """Consumer-resolved long-only domain, with original current holdings retained.

    Group rows multiply weights. Covariance is annualised in the declared units.
    Supply either dense covariance or an exact factor covariance and diagonal
    residual. Both inputs, when present, must agree. ``full_lower/upper`` are the
    eligible hard domain after retaining all mandatory and preparation pins.
    """

    names: tuple[str, ...]
    cash: int
    current: np.ndarray
    base: np.ndarray
    target: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    full_lower: np.ndarray
    full_upper: np.ndarray
    groups: np.ndarray
    group_lower: np.ndarray
    group_upper: np.ndarray
    mandatory: np.ndarray
    categories: tuple[str, ...]
    eligible: np.ndarray
    frozen: np.ndarray
    rules: ExecutionRules = field(default_factory=ExecutionRules)
    covariance: np.ndarray | None = None
    factor_loadings: np.ndarray | None = None
    factor_covariance: np.ndarray | None = None
    residual_variance: np.ndarray | None = None
    investment: float = 1.0
    annualisation: str = "annual"
    provenance: Mapping = field(default_factory=dict)
    preparation_seconds: float = 0.0

    def __post_init__(self):
        """Validate inputs without clipping weights, bounds or covariance."""
        names, n = tuple(self.names), len(self.names)
        if (
            n < 2
            or len(set(names)) != n
            or not all(isinstance(x, str) for x in names)
            or isinstance(self.cash, bool)
            or not isinstance(self.cash, (int, np.integer))
            or not 0 <= self.cash < n
            or len(self.categories) != n
        ):
            raise ValueError("Invalid instrument identities or cash coordinate")
        object.__setattr__(self, "names", names)
        object.__setattr__(self, "categories", tuple(self.categories))
        for name in ("current", "base", "target", "lower", "upper", "full_lower", "full_upper"):
            object.__setattr__(self, name, frozen_array(getattr(self, name), (n,)))
        for name in ("mandatory", "eligible", "frozen"):
            object.__setattr__(self, name, frozen_array(getattr(self, name), (n,), bool))
        g = len(self.group_lower)
        for name, shape in (("groups", (g, n)), ("group_lower", (g,)), ("group_upper", (g,))):
            object.__setattr__(self, name, frozen_array(getattr(self, name), shape))
        if (
            self.annualisation != "annual"
            or not np.isfinite(self.investment)
            or self.investment <= 0
            or not np.isfinite(self.preparation_seconds)
            or self.preparation_seconds < 0
        ):
            raise ValueError("Declare annual covariance, positive investment and preparation time")
        if (
            np.any(self.full_lower < 0)
            or np.any(self.full_lower > self.lower)
            or np.any(self.upper > self.full_upper)
            or np.any(self.lower > self.upper)
            or np.any(self.group_lower > self.group_upper)
        ):
            raise ValueError("Invalid or non-containing domains")
        required = np.abs(self.base - self.current) > self.rules.zero_tolerance
        required[self.cash] = False
        if not np.array_equal(required, self.mandatory):
            raise ValueError("Unclassified base-current move")
        floor = np.broadcast_to(self.rules.holding_floor, (n,))
        exempt = np.broadcast_to(self.rules.exemption_mask, (n,))
        if np.any(floor < 0) or exempt[self.cash] or np.any(exempt & ~self.mandatory):
            raise ValueError("Invalid floors or discretionary exemption")
        approved = {
            "model_entry",
            "model_exit",
            "small_cutoff",
            "product_min_buy",
            "product_max_sell",
            "instruction_min_buy",
            "instruction_max_sell",
            "forced_dust_exit",
            "compulsory_saa_funding_repair",
        }
        for i in np.flatnonzero(self.mandatory):
            if (
                self.categories[i] not in approved
                or self.full_lower[i] <= self.current[i] <= self.full_upper[i]
                or self.full_lower[i] != self.lower[i]
                or self.full_upper[i] != self.upper[i]
            ):
                raise ValueError("Mandatory interval or category is not preserved")
        restricted = (self.frozen | ~self.eligible) & ~self.mandatory
        restricted[self.cash] = False
        if np.any(restricted & ((self.full_lower != self.base) | (self.full_upper != self.base))):
            raise ValueError("Frozen or ineligible optional coordinate must be pinned")
        if abs(self.base.sum() - self.investment) > self.rules.audit_tolerance:
            raise ValueError("Base funding does not match declared investment")
        if self.rules.discretionary_allowance is not None:
            if self.rules.ticket_cap != self.mandatory_count + self.rules.discretionary_allowance:
                raise ValueError("Adjusted cap must equal M plus discretionary allowance")
        structured = (self.factor_loadings, self.factor_covariance, self.residual_variance)
        if any(x is not None for x in structured):
            if any(x is None for x in structured):
                raise ValueError("Incomplete exact structured covariance")
            k = np.asarray(self.factor_loadings).shape[1]
            for name, shape in (
                ("factor_loadings", (n, k)),
                ("factor_covariance", (k, k)),
                ("residual_variance", (n,)),
            ):
                object.__setattr__(self, name, frozen_array(getattr(self, name), shape))
            check_psd(self.factor_covariance)
            if np.any(self.residual_variance < 0):
                raise ValueError("Negative residual variance")
        if self.covariance is not None:
            object.__setattr__(self, "covariance", frozen_array(self.covariance, (n, n)))
            check_psd(self.covariance)
            if self.factor_loadings is not None:
                exact = self.factor_loadings @ self.factor_covariance @ self.factor_loadings.T
                exact += np.diag(self.residual_variance)
                if not np.allclose(exact, self.covariance, atol=1e-12, rtol=1e-10):
                    raise ValueError("Structured covariance is not an exact representation")
        elif self.factor_loadings is None:
            raise ValueError("Missing covariance")
        object.__setattr__(self, "provenance", immutable_metadata(dict(self.provenance)))

    @property
    def mandatory_count(self):
        """Return the frozen reservation, independent of returned numerical tickets."""
        return int(np.sum(self.mandatory))

    @property
    def exemption_hash(self):
        """Identify the explicit per-instrument minimum-ticket exemption mask."""
        mask = np.broadcast_to(self.rules.exemption_mask, self.current.shape)
        return hashlib.sha256(mask.tobytes()).hexdigest()

    def matvec(self, vector):
        """Apply authoritative covariance without a synthetic dense expansion."""
        if self.covariance is not None:
            return self.covariance @ vector
        b = self.factor_loadings
        return b @ (self.factor_covariance @ (b.T @ vector)) + self.residual_variance * vector

    def dense_covariance(self):
        """Materialise exact covariance only when a conic formulation requires it."""
        if self.covariance is not None:
            return self.covariance
        b = self.factor_loadings
        return b @ self.factor_covariance @ b.T + np.diag(self.residual_variance)


def check_psd(matrix):
    """Reject asymmetric or materially indefinite input without changing its bytes."""
    if not np.allclose(matrix, matrix.T, atol=1e-12, rtol=0):
        raise ValueError("Covariance must be symmetric")
    if np.linalg.eigvalsh(matrix).min(initial=0.0) < -1e-10:
        raise ValueError("Covariance must be positive semidefinite")


@dataclass(frozen=True)
class RankedConfig:
    """Explicit bounded state search, sizing and improvement controls.

    ``rescue_trials`` separately caps initial feasibility rescue and the coordinated
    original coordinated improvement and scarce-ticket extension support trials,
    each across at most ``improvement_passes`` incumbent anchors. The extension
    includes one perspective-guided seed and, per anchor, a complete one-for-one
    replacement sweep (at most B times the available ticket states additional
    supports). ``exchange`` enables these after the single-exchange pass. Each
    support retains the same ``mandatory_state_trials`` branch limit.

    Scarce-ticket diversification revisits the completed incumbent and the
    unrestricted guided seed, then excludes each of at most ``rescue_trials``
    incumbent names from a separate guidance solve. Each unique seed permits at
    most ``improvement_passes`` complete replacement sweeps. Sizing always uses
    the original problem; audited support results are cached within a call.

    A subsequent conditional phase includes/excludes the ticket states in the
    union of the completed incumbent and unrestricted guided seed. At most
    ``rescue_trials`` additional continuous guidance calls are made, in stable
    instrument/state order, inclusion before exclusion. Each unique proposal
    receives the same bounded replacement restart, followed by one final restart
    of the completed incumbent. These guidance decisions never restrict sizing
    or change the original acceptance domain. They also work from an empty
    discretionary incumbent whose tentative guided support is infeasible.

    A final scarce-ticket phase shares ``rescue_trials ** 2`` coordinated trials
    across at most ``improvement_passes`` anchors, with the same complete
    one-for-one sweeps. It skips cached supports and rejects only interval
    envelopes that cannot satisfy funding or a group row. All retained positions
    remain movable during unchanged joint sizing. This finite completion budget
    does not certify an exhaustive search or global optimality.

    Finally, at most ``rescue_trials`` cached audited alternatives, ordered by
    objective then instrument/state identifiers, seed separate bounded replacement
    restarts. The first branch improving the retained incumbent receives one more
    completion phase with the same squared trial budget, then search stops. An
    inferior intermediate seed may connect otherwise missed combinations; failed
    branches cannot displace the incumbent. No Exact support is supplied.
    """

    drop: bool = True
    exchange: bool = True
    compute_bound: bool = False
    rescue_trials: int = 24
    mandatory_state_trials: int = 32
    objective_tolerance: float = 1e-12
    qp_solver: str = "CLARABEL"
    improvement_passes: int = 3

    def __post_init__(self):
        """Reject unbounded or invalid numerical search controls."""
        for x in (self.rescue_trials, self.mandatory_state_trials, self.improvement_passes):
            if checked_cap(x) is None:
                raise ValueError("State search must be bounded")
        if self.mandatory_state_trials == 0 or self.objective_tolerance < 0:
            raise ValueError("Invalid state or improvement tolerance")


@dataclass(frozen=True)
class ExecutionResult:
    """Delivered audited weights and complete solver diagnostics, never repaired output."""

    weights: np.ndarray | None
    status: str
    diagnostics: dict

    def __post_init__(self):
        """Detach accepted weights from mutable solver buffers."""
        if self.weights is not None:
            object.__setattr__(self, "weights", frozen_array(self.weights))
