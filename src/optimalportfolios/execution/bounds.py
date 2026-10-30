"""Conservative supporting-plane certificates for matched continuous relaxations."""

from __future__ import annotations

from time import perf_counter
import cvxpy as cp
import numpy as np
from scipy.optimize import linprog

from .domain import domain_bounds
from .model import SCALE, build_model
from .types import ExecutionProblem


def supporting_certificate(problem, model):
    """Minimise a valid supporting plane using an LP dual and box residual correction.

    The conic primal objective is never returned as a lower bound. A tangent can
    be constructed away from feasibility, so tiny selector values are replaced
    only at the tangent point, not in the original problem or its constraints.
    """
    if model.displacement_bp is None:
        return max(0.0, model.constant - 1e-12), "constant objective with 1e-12 guard"
    x = np.asarray(model.displacement_bp.value) / SCALE
    remainder = model.funded_covariance - np.diag(model.split)
    z = np.array(
        [
            1.0 if problem.mandatory[i] else max(1e-8, float(model.occupancy[i].value))
            for i in model.indices
        ]
    )
    value = (
        model.constant
        + 2 * model.gradient @ x
        + x @ remainder @ x
        + np.sum(model.split * x * x / z)
    )
    gx = 2 * model.gradient + 2 * remainder @ x + 2 * model.split * x / z
    gz = -model.split * x * x / (z * z)
    offset = float(value - gx @ x - gz @ z)
    tangent = offset + gx @ model.displacement_bp / SCALE
    for j, i in enumerate(model.indices):
        tangent += gz[j] * (1.0 if problem.mandatory[i] else model.occupancy[i])
    lp_graph = cp.Problem(cp.Minimize(tangent), model.linear_constraints)
    data, _, inverse = lp_graph.get_problem_data("SCIPY")
    offsets = data["param_prob"].var_id_to_col
    bounds = np.full((len(data["c"]), 2), np.nan)
    lower, upper = domain_bounds(problem)
    index = offsets[model.displacement_bp.id]
    size = len(model.indices)
    bounds[index : index + size] = (
        SCALE
        * np.c_[
            lower[model.indices] - model.reference[model.indices],
            upper[model.indices] - model.reference[model.indices],
        ]
    )
    for i, selector in model.selectors.items():
        index = offsets[selector.id]
        bounds[index : index + selector.size] = [0.0, 1.0]
        index = offsets[model.contributions[i].id]
        bounds[index : index + selector.size] = [
            [min(0.0, s.lower * SCALE), max(0.0, s.upper * SCALE)] for s in model.states[i]
        ]
    if not np.isfinite(bounds).all():
        raise ValueError("Unmapped canonical variable in bound certificate")
    lp = linprog(
        data["c"],
        A_ub=data["G"],
        b_ub=data["h"],
        A_eq=data["A"],
        b_eq=data["b"],
        bounds=bounds,
        method="highs",
        options={"dual_feasibility_tolerance": 1e-9, "primal_feasibility_tolerance": 1e-9},
    )
    if not lp.success:
        raise ValueError(f"Certificate LP failed: {lp.message}")
    dual_ineq = np.minimum(lp.ineqlin.marginals, 0.0)
    dual_eq = lp.eqlin.marginals
    residual = data["c"] - data["G"].T @ dual_ineq - data["A"].T @ dual_eq
    constant = float(np.asarray(inverse[-1]["offset"]).item())
    corrected = (
        constant
        + data["h"] @ dual_ineq
        + data["b"] @ dual_eq
        + np.minimum(residual * bounds[:, 0], residual * bounds[:, 1]).sum()
    )
    guard = 1e-12 * (
        1
        + abs(constant)
        + np.sum(np.abs(data["h"] * dual_ineq))
        + np.sum(np.abs(data["b"] * dual_eq))
    )
    return max(
        0.0, float(corrected - guard)
    ), "supporting-plane LP dual, box correction, scaled 1e-12 guard"


def compute_perspective_lower_bound(problem: ExecutionProblem) -> dict:
    """Return plain and perspective primal diagnostics and conservative dual bounds.

    Both relax the identical CORRIDOR state problem. Unavailable certificates
    explicitly fall back to the nonnegative-variance bound of zero.
    """
    results = {}
    for perspective in (False, True):
        started = perf_counter()
        record = dict(
            primal=None,
            lower_bound=0.0,
            status="not_run",
            certificate="nonnegative variance only",
            residual=None,
        )
        try:
            model = build_model(problem, perspective=perspective, continuous=True)
            model.optimisation.solve(
                solver="CLARABEL", tol_gap_abs=1e-10, tol_gap_rel=1e-10, tol_feas=1e-10
            )
            record["status"] = model.optimisation.status
            if model.optimisation.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
                record["primal"] = float(model.optimisation.value / SCALE)
                record["residual"] = max(
                    float(np.max(row.violation(), initial=0.0))
                    for row in model.optimisation.constraints
                )
                bound, reason = supporting_certificate(problem, model)
                record.update(lower_bound=bound, certificate=reason, split=model.split.tolist())
        except Exception as error:
            record["certificate_error"] = repr(error)
        record["seconds"] = perf_counter() - started
        results["perspective" if perspective else "plain"] = record
    return results
