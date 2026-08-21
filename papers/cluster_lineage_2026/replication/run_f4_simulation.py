"""Run the fixed synthetic boundary-flip and Ward-verification experiment.

The simulation is the sole market-independent new computation in the manuscript roadmap.
Each DGP/estimator cell uses one frozen seed and common random paths across its four
hysteresis arms.  Per-cell checkpoints make the 144-cell run resumable without changing
random draws.  The flat assignment uses the known population blocks and mean correlation
distance; Ward uses the production one-minus-correlation distance, Ward linkage, cutoff
fraction 0.5, and the identical pairwise partition-distance bonus.  One proportionality
constant is fitted on the zero-delta arm of each DGP/estimator/method cell and then held
fixed across its other delta arms, as required by the absorbed-constant theory.

Because equal population blocks make every asset's population margin identical within a
cell, the requested within-cell margin-decile statistic has one tied bucket.  The output
records this explicitly instead of fabricating ten arbitrary bins; the three distinct
pooled margin levels are the separation values 0.10, 0.20, and 0.30.
"""

from __future__ import annotations

import hashlib
import io
import itertools
import json
import os
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import squareform
from scipy.special import ndtr


matplotlib.use("Agg")
import matplotlib.pyplot as plt


DIMENSIONS = (50, 100)
GROUP_COUNTS = (5, 10)
SEPARATIONS = (0.10, 0.20, 0.30)
GARCH_PARAMETERS = ((0.0, 0.0), (0.05, 0.90), (0.10, 0.85))
SPANS = (36, 156)
STEPS = (13.0 / 3.0, 13.0)
DELTA_LABELS = ("zero", "innovation", "level", "double_level")
REPLICATIONS = 500
ESTIMATION_DATES = 24
BURN_MULTIPLE = 5
RHO_BETWEEN = 0.20
CUTOFF_FRACTION = 0.5
BASE_SEED = 20260817
WORKERS = int(os.environ.get("CLUSTER_LINEAGE_F4_WORKERS", "4"))
BATCH_SIZE = 10
CACHE_VERSION = 1
TOLERANCE = 1e-12


def _output_root() -> Path:
    """Return the configured cluster-lineage output root."""
    value = os.environ.get("CLUSTER_LINEAGE_OUTPUT_DIR")
    if not value:
        raise RuntimeError("CLUSTER_LINEAGE_OUTPUT_DIR must be set")
    return Path(value).resolve()


def _root() -> Path:
    """Return the isolated F4 output directory."""
    root = _output_root() / "finalisation" / "f4"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _cell_root() -> Path:
    """Return the resumable per-cell checkpoint directory."""
    root = _root() / "cells"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _write(frame: pd.DataFrame, path: Path) -> None:
    """Write one deterministic high-precision CSV."""
    frame.to_csv(path, index=False, float_format="%.17g", lineterminator="\n")


def _sha256(path: Path) -> str:
    """Return one file's SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def garch_kappa(alpha: float, beta: float) -> float:
    """Return the manuscript's Gaussian-GARCH elliptical-kurtosis parameter."""
    denominator = 1.0 - (alpha + beta) ** 2 - 2.0 * alpha**2
    if denominator <= 0.0:
        raise ValueError("GARCH parameters do not have a finite fourth moment")
    return 2.0 * alpha**2 / denominator


def population_correlation(dimension: int, groups: int, separation: float) -> np.ndarray:
    """Return the equal-size G-block population correlation matrix."""
    if dimension % groups:
        raise ValueError("dimension must be divisible by groups")
    rho_within = RHO_BETWEEN + separation
    correlation = np.full((dimension, dimension), RHO_BETWEEN, dtype=float)
    size = dimension // groups
    for group in range(groups):
        start = group * size
        correlation[start : start + size, start : start + size] = rho_within
    np.fill_diagonal(correlation, 1.0)
    return correlation


def _true_groups(dimension: int, groups: int) -> np.ndarray:
    """Return fixed integer population-block labels."""
    return np.repeat(np.arange(groups), dimension // groups)


def _delta_grid(span: int, step: float, kappa: float, rho_within: float) -> dict[str, float]:
    """Return the fixed zero, innovation, level, and double-level thresholds."""
    decay = 1.0 - 2.0 / (span + 1.0)
    level = np.sqrt(1.0 + kappa) * (1.0 - rho_within**2) / np.sqrt(span)
    innovation = np.sqrt(2.0 * (1.0 - decay**step)) * level
    return {
        "zero": 0.0,
        "innovation": float(innovation),
        "level": float(level),
        "double_level": float(2.0 * level),
    }


def _step_pattern(step: float) -> np.ndarray:
    """Return 24 integer observation gaps with mean 13/3 or 13 exactly."""
    if np.isclose(step, 13.0):
        return np.full(ESTIMATION_DATES, 13, dtype=int)
    if np.isclose(step, 13.0 / 3.0):
        return np.resize(np.array([4, 4, 5], dtype=int), ESTIMATION_DATES)
    raise ValueError(f"unsupported simulation step {step!r}")


def _correlation(covariance: np.ndarray) -> np.ndarray:
    """Convert one positive covariance matrix to a clipped correlation matrix."""
    scale = np.sqrt(np.maximum(np.diag(covariance), np.finfo(float).tiny))
    correlation = covariance / np.outer(scale, scale)
    correlation = np.clip((correlation + correlation.T) / 2.0, -1.0, 1.0)
    np.fill_diagonal(correlation, 1.0)
    return correlation


def _distance_with_bonus(
    correlation: np.ndarray,
    previous: np.ndarray | None,
    delta: float,
) -> np.ndarray:
    """Apply the production partition bonus to one-minus-correlation distance."""
    distance = 1.0 - correlation
    if previous is not None and delta > 0.0:
        same = previous[:, None] == previous[None, :]
        distance[same] = np.maximum(distance[same] - delta, 0.0)
    distance = (distance + distance.T) / 2.0
    np.fill_diagonal(distance, 0.0)
    return distance


def _flat_partition(distance: np.ndarray, true_groups: np.ndarray) -> np.ndarray:
    """Assign every asset to its nearest known population block by mean distance."""
    group_ids = np.unique(true_groups)
    means = np.empty((len(true_groups), len(group_ids)), dtype=float)
    for column, group in enumerate(group_ids):
        members = true_groups == group
        sums = distance[:, members].sum(axis=1)
        counts = np.full(len(true_groups), members.sum(), dtype=float)
        own = members
        sums[own] -= np.diag(distance)[own]
        counts[own] -= 1.0
        means[:, column] = sums / counts
    return group_ids[np.argmin(means, axis=1)]


def _ward_partition(distance: np.ndarray) -> np.ndarray:
    """Apply the production Ward linkage and 0.5 maximum-distance cutoff."""
    condensed = squareform(distance, checks=False)
    tree = linkage(condensed, method="ward")
    cutoff = CUTOFF_FRACTION * float(np.max(condensed))
    return fcluster(tree, cutoff, criterion="distance").astype(int)


def _canonical(labels: np.ndarray) -> np.ndarray:
    """Relabel an initial partition by order of first appearance."""
    mapping: dict[int, int] = {}
    output = np.empty(len(labels), dtype=int)
    for index, value in enumerate(labels):
        mapping.setdefault(int(value), len(mapping))
        output[index] = mapping[int(value)]
    return output


def _align_to_previous(previous: np.ndarray, current: np.ndarray) -> tuple[np.ndarray, int]:
    """Maximise member overlap, return persistent labels and the asset flip count."""
    prior_ids = np.unique(previous)
    current_ids = np.unique(current)
    overlaps = np.zeros((len(prior_ids), len(current_ids)), dtype=int)
    for i, prior in enumerate(prior_ids):
        for j, present in enumerate(current_ids):
            overlaps[i, j] = int(np.sum((previous == prior) & (current == present)))
    left, right = linear_sum_assignment(-overlaps)
    mapping = {int(current_ids[j]): int(prior_ids[i]) for i, j in zip(left, right)}
    next_id = int(prior_ids.max()) + 1
    for value in current_ids:
        if int(value) not in mapping:
            mapping[int(value)] = next_id
            next_id += 1
    aligned = np.array([mapping[int(value)] for value in current], dtype=int)
    return aligned, int(np.sum(aligned != previous))


def _cell_grid() -> list[dict[str, object]]:
    """Return the deterministic 144-cell DGP/estimator grid."""
    rows = []
    for index, values in enumerate(
        itertools.product(
            DIMENSIONS,
            GROUP_COUNTS,
            SEPARATIONS,
            GARCH_PARAMETERS,
            SPANS,
            STEPS,
        )
    ):
        dimension, groups, separation, garch, span, step = values
        alpha, beta = garch
        rows.append(
            {
                "cell_index": index,
                "dimension": dimension,
                "groups": groups,
                "separation": separation,
                "alpha": alpha,
                "beta": beta,
                "span": span,
                "step": step,
                "seed": BASE_SEED + index,
            }
        )
    return rows


def _cell_fingerprint(cell: dict[str, object]) -> str:
    """Return a versioned fingerprint for one checkpoint."""
    payload = {
        **cell,
        "cache_version": CACHE_VERSION,
        "replications": REPLICATIONS,
        "estimation_dates": ESTIMATION_DATES,
        "burn_multiple": BURN_MULTIPLE,
        "rho_between": RHO_BETWEEN,
        "cutoff_fraction": CUTOFF_FRACTION,
        "batch_size": BATCH_SIZE,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _checkpoint_path(cell_index: int) -> Path:
    """Return one cell's checkpoint path."""
    return _cell_root() / f"cell_{cell_index:03d}.pkl"


def _update_garch_batch(
    rng: np.random.Generator,
    covariance: np.ndarray,
    variance: np.ndarray,
    correlation_cholesky: np.ndarray,
    alpha: float,
    beta: float,
    decay: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Advance one batched GARCH-return and EWMA-covariance observation."""
    innovations = rng.standard_normal(variance.shape)
    shocks = np.sqrt(variance) * innovations
    returns = shocks @ correlation_cholesky.T
    covariance *= decay
    covariance += (1.0 - decay) * np.einsum("bi,bj->bij", returns, returns)
    omega = 1.0 - alpha - beta
    variance = omega + alpha * shocks**2 + beta * variance
    return covariance, variance


def _simulate_cell(cell: dict[str, object], replications: int = REPLICATIONS) -> pd.DataFrame:
    """Simulate one DGP/estimator cell and aggregate all four hysteresis arms."""
    dimension = int(cell["dimension"])
    groups = int(cell["groups"])
    separation = float(cell["separation"])
    alpha = float(cell["alpha"])
    beta = float(cell["beta"])
    span = int(cell["span"])
    step = float(cell["step"])
    seed = int(cell["seed"])
    kappa = garch_kappa(alpha, beta)
    rho_within = RHO_BETWEEN + separation
    deltas = _delta_grid(span, step, kappa, rho_within)
    decay = 1.0 - 2.0 / (span + 1.0)
    sigma = (
        np.sqrt(2.0 * (1.0 - decay**step))
        * np.sqrt(1.0 + kappa)
        * (1.0 - rho_within**2)
        / np.sqrt(span)
    )
    predicted = {
        label: float(ndtr(-(separation + delta) / (np.sqrt(2.0) * sigma)))
        for label, delta in deltas.items()
    }
    population = population_correlation(dimension, groups, separation)
    cholesky = np.linalg.cholesky(population)
    true_groups = _true_groups(dimension, groups)
    step_pattern = _step_pattern(step)
    rng = np.random.default_rng(seed)
    counts = {
        method: {
            label: {
                "flips": 0,
                "pairs": 0,
                "cluster_count_sum": 0.0,
                "partition_dates": 0,
            }
            for label in DELTA_LABELS
        }
        for method in ("flat", "ward")
    }

    for batch_start in range(0, replications, BATCH_SIZE):
        batch = min(BATCH_SIZE, replications - batch_start)
        covariance = np.broadcast_to(np.eye(dimension), (batch, dimension, dimension)).copy()
        variance = np.ones((batch, dimension), dtype=float)
        for _ in range(BURN_MULTIPLE * span):
            covariance, variance = _update_garch_batch(
                rng, covariance, variance, cholesky, alpha, beta, decay
            )
        previous = {
            method: {label: [None] * batch for label in DELTA_LABELS}
            for method in ("flat", "ward")
        }
        for gap in step_pattern:
            for _ in range(int(gap)):
                covariance, variance = _update_garch_batch(
                    rng, covariance, variance, cholesky, alpha, beta, decay
                )
            for path in range(batch):
                corr = _correlation(covariance[path])
                for method in ("flat", "ward"):
                    for label in DELTA_LABELS:
                        prior = previous[method][label][path]
                        distance = _distance_with_bonus(corr, prior, deltas[label])
                        raw = (
                            _flat_partition(distance, true_groups)
                            if method == "flat"
                            else _ward_partition(distance)
                        )
                        if prior is None:
                            aligned = _canonical(raw)
                        else:
                            aligned, flips = _align_to_previous(prior, raw)
                            counts[method][label]["flips"] += flips
                            counts[method][label]["pairs"] += dimension
                        previous[method][label][path] = aligned
                        counts[method][label]["cluster_count_sum"] += len(
                            np.unique(aligned)
                        )
                        counts[method][label]["partition_dates"] += 1

    rows = []
    for method in ("flat", "ward"):
        for label in DELTA_LABELS:
            count = counts[method][label]
            realised = float(count["flips"] / count["pairs"])
            probability = predicted[label]
            fitted_c = realised / probability if probability > 0.0 else 0.0
            rows.append(
                {
                    **cell,
                    "cell_fingerprint": _cell_fingerprint(cell),
                    "method": method,
                    "delta_label": label,
                    "delta": deltas[label],
                    "rho_between": RHO_BETWEEN,
                    "rho_within": rho_within,
                    "kappa_theta": kappa,
                    "decay": decay,
                    "transition_sigma": sigma,
                    "population_margin": separation,
                    "within_cell_margin_deciles": 1,
                    "margin_decile_label": "all_assets_tied",
                    "replications": replications,
                    "estimation_dates": ESTIMATION_DATES,
                    "transition_asset_pairs": int(count["pairs"]),
                    "predicted_flip_probability": probability,
                    "realised_flip_probability": realised,
                    "absolute_prediction_error": abs(realised - probability),
                    "mean_absolute_prediction_error_by_margin_decile": abs(
                        realised - probability
                    ),
                    "fitted_proportionality_constant": fitted_c,
                    "expected_total_churn_per_transition": dimension * probability,
                    "realised_total_churn_per_transition": dimension * realised,
                    "mean_cluster_count": (
                        count["cluster_count_sum"] / count["partition_dates"]
                    ),
                    "step_pattern": "4|4|5" if step < 13.0 else "13",
                }
            )
    return pd.DataFrame(rows)


def _run_or_load_cell(cell: dict[str, object]) -> tuple[pd.DataFrame, str, float]:
    """Load a valid checkpoint or simulate and atomically persist one cell."""
    started = time.perf_counter()
    path = _checkpoint_path(int(cell["cell_index"]))
    fingerprint = _cell_fingerprint(cell)
    if path.exists():
        with path.open("rb") as stream:
            payload = pickle.load(stream)
        if payload.get("fingerprint") == fingerprint:
            return payload["frame"], "hit", time.perf_counter() - started
    frame = _simulate_cell(cell)
    payload = {"fingerprint": fingerprint, "frame": frame}
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)
    return frame, "miss_computed", time.perf_counter() - started


def _collect_cells() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute or load all cells with the frozen four-worker default."""
    cells = _cell_grid()
    frames = []
    runtime_rows = []
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(_run_or_load_cell, cell): cell for cell in cells}
        for completed, future in enumerate(as_completed(futures), 1):
            cell = futures[future]
            frame, status, seconds = future.result()
            frames.append(frame)
            runtime_rows.append(
                {
                    "cell_index": cell["cell_index"],
                    "cache_status": status,
                    "runtime_seconds": seconds,
                }
            )
            print(
                f"F4 cell {completed}/{len(cells)} index={cell['cell_index']} "
                f"status={status} seconds={seconds:.1f}",
                flush=True,
            )
    results = pd.concat(frames, ignore_index=True).sort_values(
        ["cell_index", "method", "delta"]
    ).reset_index(drop=True)
    runtime = pd.DataFrame(runtime_rows).sort_values("cell_index").reset_index(drop=True)
    return results, runtime


def _ward_verification(results: pd.DataFrame) -> pd.DataFrame:
    """Return every Ward cell with aggregate accuracy measures repeated for provenance."""
    ward = results.loc[results["method"].eq("ward")].copy()
    correlation = float(
        ward["predicted_flip_probability"].corr(ward["realised_flip_probability"])
    )
    mae = float(ward["absolute_prediction_error"].mean())
    ward["cross_cell_predicted_realised_correlation"] = correlation
    ward["cross_cell_mean_absolute_error"] = mae
    ward["acceptance_threshold"] = "DESCRIPTIVE_ONLY"
    return ward.reset_index(drop=True)


def _apply_cell_constants(results: pd.DataFrame) -> pd.DataFrame:
    """Fit one zero-arm churn multiplier per cell/method and hold it across deltas."""
    output = results.copy()
    output["unscaled_predicted_flip_probability"] = output[
        "predicted_flip_probability"
    ]
    keys = [
        "cell_index",
        "dimension",
        "groups",
        "separation",
        "alpha",
        "beta",
        "span",
        "step",
        "method",
    ]
    zero = output.loc[output["delta_label"].eq("zero")].copy()
    zero["cell_constant"] = zero["realised_flip_probability"].div(
        zero["unscaled_predicted_flip_probability"]
    )
    constants = zero.set_index(keys)["cell_constant"]
    row_index = pd.MultiIndex.from_frame(output[keys])
    output["fitted_proportionality_constant"] = constants.reindex(row_index).to_numpy()
    output["predicted_flip_probability"] = output[
        "unscaled_predicted_flip_probability"
    ].mul(output["fitted_proportionality_constant"])
    output["absolute_prediction_error"] = output[
        "realised_flip_probability"
    ].sub(output["predicted_flip_probability"]).abs()
    output["mean_absolute_prediction_error_by_margin_decile"] = output[
        "absolute_prediction_error"
    ]
    output["expected_total_churn_per_transition"] = output[
        "dimension"
    ].mul(output["predicted_flip_probability"])
    output["constant_calibration_arm"] = "zero"
    return output


def _flat_acceptance(results: pd.DataFrame) -> tuple[float, int, pd.DataFrame]:
    """Measure the two fixed flat-cut verification targets."""
    selected = results.loc[
        results["method"].eq("flat")
        & results["alpha"].eq(0.0)
        & results["beta"].eq(0.0)
        & results["separation"].ge(0.20)
    ]
    correlation = float(
        selected["predicted_flip_probability"].corr(
            selected["realised_flip_probability"]
        )
    )
    violations = []
    keys = [
        "cell_index",
        "dimension",
        "groups",
        "separation",
        "alpha",
        "beta",
        "span",
        "step",
    ]
    for values, frame in results.loc[results["method"].eq("flat")].groupby(keys):
        ordered = frame.sort_values(["delta", "delta_label"])
        realised = ordered["realised_flip_probability"].to_numpy(dtype=float)
        if np.any(np.diff(realised) > TOLERANCE):
            violations.append(
                {
                    **dict(zip(keys, values)),
                    "realised_by_delta": "|".join(f"{value:.17g}" for value in realised),
                }
            )
    columns = [*keys, "realised_by_delta"]
    return correlation, len(violations), pd.DataFrame(violations, columns=columns)


def _render(results: pd.DataFrame) -> bytes:
    """Render the predicted-versus-realised flip exhibit to deterministic PDF bytes."""
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 4.3), sharex=True, sharey=True)
    colors = {0.10: "#E45756", 0.20: "#4C78A8", 0.30: "#54A24B"}
    markers = {"zero": "o", "innovation": "s", "level": "^", "double_level": "D"}
    for axis, method in zip(axes, ("flat", "ward")):
        selected = results.loc[results["method"].eq(method)]
        for separation in SEPARATIONS:
            for label in DELTA_LABELS:
                points = selected.loc[
                    selected["separation"].eq(separation)
                    & selected["delta_label"].eq(label)
                ]
                axis.scatter(
                    points["predicted_flip_probability"],
                    points["realised_flip_probability"],
                    s=14,
                    alpha=0.55,
                    color=colors[separation],
                    marker=markers[label],
                    linewidths=0.0,
                )
        maximum = float(
            max(
                selected["predicted_flip_probability"].max(),
                selected["realised_flip_probability"].max(),
            )
        )
        axis.plot([0.0, maximum], [0.0, maximum], color="black", linewidth=0.8)
        axis.set_title("Known-block flat assignment" if method == "flat" else "Ward cutoff")
        axis.set_xlabel("Predicted flip probability")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Realised flip probability")
    fig.suptitle("Boundary-flip approximation: synthetic G-block panels")
    fig.tight_layout()
    metadata = {
        "Creator": "OptimalPortfolios cluster-lineage F4",
        "Producer": "matplotlib",
        "CreationDate": None,
        "ModDate": None,
    }
    buffer = io.BytesIO()
    fig.savefig(buffer, format="pdf", bbox_inches="tight", metadata=metadata)
    plt.close(fig)
    return buffer.getvalue()


def _run_parameters() -> pd.DataFrame:
    """Return the fixed simulation design as a machine-readable table."""
    return pd.DataFrame(
        [
            {
                "dimensions": "50|100",
                "groups": "5|10",
                "separations": "0.10|0.20|0.30",
                "rho_between": RHO_BETWEEN,
                "garch_parameters": "0,0|0.05,0.90|0.10,0.85",
                "spans": "36|156",
                "steps": "13/3|13",
                "delta_arms": "zero|innovation|level|double_level",
                "replications": REPLICATIONS,
                "burn_multiple": BURN_MULTIPLE,
                "estimation_dates": ESTIMATION_DATES,
                "base_seed": BASE_SEED,
                "workers": WORKERS,
                "cutoff_fraction": CUTOFF_FRACTION,
                "distance": "one_minus_correlation",
                "linkage": "ward",
                "fractional_step_pattern": "4|4|5",
                "within_cell_margin_deciles": "one tied bucket by equal-block design",
                "proportionality_constant_fit": (
                    "zero-delta arm per DGP/estimator/method cell"
                ),
                "proportionality_constant_application": (
                    "Gaussian probability multiplier held across delta arms"
                ),
            }
        ]
    )


def _artifact_hashes(names: list[str]) -> dict[str, str]:
    """Hash named F4 artifacts."""
    return {name: _sha256(_root() / name) for name in names}


def run() -> dict[str, pd.DataFrame]:
    """Execute F4, write artifacts, and enforce the proved-case targets."""
    results, runtime = _collect_cells()
    results = _apply_cell_constants(results)
    ward = _ward_verification(results)
    correlation, violations, violation_table = _flat_acceptance(results)
    expected_rows = len(_cell_grid()) * len(DELTA_LABELS) * 2
    acceptance = pd.DataFrame(
        [
            {
                "check": "DGP/estimator cells",
                "measured": results["cell_index"].nunique(),
                "tolerance": len(_cell_grid()),
                "status": (
                    "PASS"
                    if results["cell_index"].nunique() == len(_cell_grid())
                    else "FAIL"
                ),
            },
            {
                "check": "simulation rows",
                "measured": len(results),
                "tolerance": expected_rows,
                "status": "PASS" if len(results) == expected_rows else "FAIL",
            },
            {
                "check": "replications per cell",
                "measured": int(results["replications"].min()),
                "tolerance": REPLICATIONS,
                "status": "PASS" if results["replications"].eq(REPLICATIONS).all() else "FAIL",
            },
            {
                "check": "estimation dates per path",
                "measured": int(results["estimation_dates"].min()),
                "tolerance": ESTIMATION_DATES,
                "status": (
                    "PASS"
                    if results["estimation_dates"].eq(ESTIMATION_DATES).all()
                    else "FAIL"
                ),
            },
            {
                "check": "Gaussian flat predicted-realised correlation sep>=0.20",
                "measured": correlation,
                "tolerance": 0.9,
                "status": "PASS" if correlation >= 0.9 else "FAIL",
            },
            {
                "check": "flat realised-churn delta monotonicity violations",
                "measured": violations,
                "tolerance": 0,
                "status": "PASS" if violations == 0 else "FAIL",
            },
            {
                "check": "NaNs across simulation results",
                "measured": int(results.isna().sum().sum()),
                "tolerance": 0,
                "status": "PASS" if not results.isna().to_numpy().any() else "FAIL",
            },
        ]
    )
    _write(results, _root() / "simulation_results.csv")
    _write(ward, _root() / "ward_verification.csv")
    _write(runtime, _root() / "runtime.csv")
    _write(_run_parameters(), _root() / "run_parameters.csv")
    _write(violation_table, _root() / "monotonicity_violations.csv")
    _write(acceptance, _root() / "acceptance.csv")
    first_pdf = _render(results)
    second_pdf = _render(results)
    if first_pdf != second_pdf:
        raise AssertionError("F4 PDF rendering is not byte-identical")
    (_root() / "flip_approximation.pdf").write_bytes(first_pdf)
    deterministic_names = [
        "acceptance.csv",
        "flip_approximation.pdf",
        "monotonicity_violations.csv",
        "run_parameters.csv",
        "simulation_results.csv",
        "ward_verification.csv",
    ]
    first = _artifact_hashes(deterministic_names)
    _write(results, _root() / "simulation_results.csv")
    _write(ward, _root() / "ward_verification.csv")
    _write(_run_parameters(), _root() / "run_parameters.csv")
    _write(violation_table, _root() / "monotonicity_violations.csv")
    _write(acceptance, _root() / "acceptance.csv")
    (_root() / "flip_approximation.pdf").write_bytes(_render(results))
    second = _artifact_hashes(deterministic_names)
    determinism = pd.DataFrame(
        [
            {
                "artifact": name,
                "first_sha256": first[name],
                "second_sha256": second[name],
                "byte_identical": first[name] == second[name],
            }
            for name in deterministic_names
        ]
    )
    if not determinism["byte_identical"].all():
        raise AssertionError("F4 deterministic replay failed")
    _write(determinism, _root() / "determinism.csv")
    if not acceptance["status"].eq("PASS").all():
        raise AssertionError(acceptance.loc[~acceptance["status"].eq("PASS")])
    print(f"f4_root={_root()} rows={len(results)} flat_correlation={correlation:.6f}")
    return {
        "simulation_results.csv": results,
        "ward_verification.csv": ward,
        "runtime.csv": runtime,
        "acceptance.csv": acceptance,
        "determinism.csv": determinism,
    }


if __name__ == "__main__":
    run()
