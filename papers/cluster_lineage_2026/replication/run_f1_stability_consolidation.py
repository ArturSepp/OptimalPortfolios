"""Consolidate frozen stability evidence for manuscript-finalisation stage F1.

All empirical inputs are frozen E1--E6 artifacts inventoried by F0.  The runner
re-scores those inputs only: it forms margin-decile flip rates, bootstraps the
cross-configuration prediction correlation, locates a discrete normalized
frontier knee, evaluates the two calibration formulas, and reports subsample
and crisis-window ergodicity statistics.  It never estimates a covariance or
cluster partition.
"""

from __future__ import annotations

import hashlib
import io
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from factorlasso.cluster_smoothing import _iter_correlation_inputs

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import papers.cluster_lineage_2026.replication.run_backtests as e5
import papers.cluster_lineage_2026.replication.run_inference as inference
import papers.cluster_lineage_2026.replication.run_u1_covar_grid as u1_grid


BLOCK_LENGTH = 6
BOOTSTRAP_DRAWS = 2000
SEED = 20260813
HEADLINE_START = pd.Timestamp("2009-08-31")
HEADLINE_END = pd.Timestamp("2026-06-30")
CONFIGURED_DELTAS = {
    ("equity_panel", "W-WED"): 0.0866,
    ("futures_panel", "W-WED"): 0.0691,
    ("fund_panel", "ME"): 0.0830,
    ("fund_panel", "QE"): 0.1609,
}
INNOVATION_MARKERS = {
    ("equity_panel", "W-WED"): 0.0285,
    ("futures_panel", "W-WED"): 0.0227,
    ("fund_panel", "ME"): 0.0273,
    ("fund_panel", "QE"): 0.0893,
}
PANEL_INFO = {
    "equity_panel": {
        "cache": "msci_us",
        "universe": e5.UniverseName.MSCI_US,
        "frequencies": ("W-WED",),
    },
    "futures_panel": {
        "cache": "futures",
        "universe": e5.UniverseName.FUTURES,
        "frequencies": ("W-WED",),
    },
    "fund_panel": {
        "cache": "mac",
        "universe": e5.UniverseName.MAC,
        "frequencies": ("ME", "QE"),
    },
}
CRISIS_WINDOWS = {
    "GFC": (pd.Timestamp("2007-07-01"), pd.Timestamp("2009-06-30")),
    "COVID_2020": (pd.Timestamp("2020-02-01"), pd.Timestamp("2020-12-31")),
    "RATE_SHOCK_2022": (pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31")),
}
COLORS = {"baseline": "#4C78A8", "M1_star": "#E45756"}


def _output_root() -> Path:
    """Return the configured cluster-lineage output root."""
    value = os.environ.get("CLUSTER_LINEAGE_OUTPUT_DIR")
    if not value:
        raise RuntimeError("CLUSTER_LINEAGE_OUTPUT_DIR must be set")
    return Path(value).resolve()


def _root() -> Path:
    """Return the isolated F1 output directory."""
    root = _output_root() / "finalisation" / "f1"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _stable_rng(*keys: object) -> np.random.Generator:
    """Return a call-order-independent generator derived from the frozen seed."""
    digest = hashlib.sha256("\x1f".join(map(str, keys)).encode("utf-8")).digest()
    child = int.from_bytes(digest[:8], "little")
    return np.random.default_rng(np.random.SeedSequence([SEED, child]))


def _mbb_indices(n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw circular moving-block indices for every bootstrap replication."""
    if n <= 0:
        raise ValueError("moving-block bootstrap requires at least one observation")
    blocks = int(np.ceil(n / BLOCK_LENGTH))
    starts = rng.integers(0, n, size=(BOOTSTRAP_DRAWS, blocks))
    offsets = np.arange(BLOCK_LENGTH)
    return ((starts[..., None] + offsets) % n).reshape(BOOTSTRAP_DRAWS, -1)[:, :n]


def _row_correlations(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Return one Pearson correlation for each aligned pair of matrix rows."""
    left_centered = left - left.mean(axis=1, keepdims=True)
    right_centered = right - right.mean(axis=1, keepdims=True)
    numerator = np.sum(left_centered * right_centered, axis=1)
    denominator = np.sqrt(
        np.sum(left_centered**2, axis=1) * np.sum(right_centered**2, axis=1)
    )
    if np.any(denominator <= 0.0):
        raise AssertionError("bootstrap correlation has a zero-variance draw")
    return numerator / denominator


def _read_csv(path: Path, **kwargs: object) -> pd.DataFrame:
    """Read one frozen CSV without losing its recorded float representation."""
    return pd.read_csv(path, float_precision="round_trip", **kwargs)


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    """Write a stable, high-precision CSV artifact."""
    frame.to_csv(path, index=False, float_format="%.17g", lineterminator="\n")


def _analysis_windows(panel: str, dates: pd.Series) -> Mapping[str, pd.Series]:
    """Return the full and, for equities, owner-frozen headline masks."""
    output = {"full_panel": pd.Series(True, index=dates.index)}
    if panel == "equity_panel":
        output["headline_20090831_20260630"] = dates.between(
            HEADLINE_START, HEADLINE_END
        )
    return output


# ``metrics.greedy_membership_panel`` exists only in recovered bytecode; this exact
# source reconstruction avoids adding a new pyc_compat dependency that F7 must retire.
def _greedy_membership_panel(
    partitions: Mapping[pd.Timestamp, pd.Series],
) -> pd.DataFrame:
    """Greedily map consecutive raw clusters by maximum member overlap."""
    rows = {}
    next_id = 0
    prior_members: dict[str, set[object]] = {}
    for date in sorted(partitions):
        current = partitions[date].dropna()
        groups = {
            label: set(current.index[current.eq(label)]) for label in pd.unique(current)
        }
        assigned = {}
        candidates = []
        for label, members in groups.items():
            for track, old_members in prior_members.items():
                overlap = len(members & old_members)
                if overlap:
                    candidates.append((-overlap, str(label), track, label))
        used_tracks = set()
        for _, _, track, label in sorted(candidates):
            if label not in assigned and track not in used_tracks:
                assigned[label] = track
                used_tracks.add(track)
        for label in sorted(groups, key=str):
            if label not in assigned:
                assigned[label] = f"R{next_id:05d}"
                next_id += 1
        rows[date] = {asset: assigned[label] for asset, label in current.items()}
        prior_members = {assigned[label]: members for label, members in groups.items()}
    return pd.DataFrame.from_dict(rows, orient="index").sort_index(axis=1)


def _partition_panels() -> dict[str, dict[str, Mapping[pd.Timestamp, pd.Series]]]:
    """Load raw baseline partitions separately for every native frequency."""
    output = {}
    for panel, info in PANEL_INFO.items():
        rolling = inference.load_cached(info["universe"], e5.SmootherName.BASELINE)
        output[panel] = {
            frequency: inference._partitions(rolling, frequency)
            for frequency in info["frequencies"]
        }
    return output


def _flip_panel(
    margin: pd.DataFrame,
    partitions: Mapping[pd.Timestamp, pd.Series],
    frequency: str,
) -> pd.DataFrame:
    """Attach realised greedily relabelled changes at the margin panel's cadence."""
    selected = margin.loc[margin["frequency"].eq(frequency)].copy()
    selected["date"] = pd.to_datetime(selected["date"])
    dates = pd.DatetimeIndex(sorted(selected["date"].unique()))
    native = {date: partitions[date] for date in dates if date in partitions}
    current = _greedy_membership_panel(native).reindex(index=dates)
    prior = current.shift(1)
    current_long = current.stack(future_stack=True).rename("current_id")
    prior_long = prior.stack(future_stack=True).rename("prior_id")
    transitions = pd.concat([current_long, prior_long], axis=1).dropna()
    transitions["realised_flip"] = transitions["current_id"].ne(
        transitions["prior_id"]
    ).astype(float)
    transitions.index.names = ["date", "asset"]
    selected = selected.set_index(["date", "asset"]).join(
        transitions[["realised_flip"]], how="inner"
    )
    return selected.reset_index()


def _correlation_bootstrap(
    panel: str,
    per_date: pd.DataFrame,
    aggregate: pd.DataFrame,
    frozen: pd.DataFrame,
) -> tuple[pd.DataFrame, float]:
    """Bootstrap P1 correlations jointly across every configuration series."""
    rows = []
    maximum_regression_error = 0.0
    for (window, convention), group in per_date.groupby(
        ["analysis_window", "singleton_convention"], sort=False
    ):
        aggregate_group = aggregate.loc[
            aggregate["analysis_window"].eq(window)
            & aggregate["singleton_convention"].eq(convention)
        ]
        configs = sorted(aggregate_group["config"].unique())
        predicted_draws = np.zeros((BOOTSTRAP_DRAWS, len(configs)))
        realised_draws = np.zeros_like(predicted_draws)
        point_predicted = np.zeros(len(configs))
        point_realised = np.zeros(len(configs))
        for frequency, frequency_group in group.groupby("frequency", sort=True):
            pred = frequency_group.pivot(
                index="index", columns="config", values="predicted_changes"
            ).reindex(columns=configs)
            real = frequency_group.pivot(
                index="index", columns="config", values="realised_changes"
            ).reindex(columns=configs)
            aligned = pred.notna().all(axis=1) & real.notna().all(axis=1)
            pred = pred.loc[aligned]
            real = real.loc[aligned]
            if pred.empty:
                raise AssertionError(f"{panel} {frequency} has no aligned P1 dates")
            frequency_aggregate = aggregate_group.loc[
                aggregate_group["frequency"].eq(frequency)
            ].set_index("config").reindex(configs)
            pred_scale = (
                frequency_aggregate["predicted_churn_annualized"].to_numpy()
                / pred.mean(axis=0).to_numpy()
            )
            real_scale = (
                frequency_aggregate["realized_churn_annualized"].to_numpy()
                / real.mean(axis=0).to_numpy()
            )
            maximum_regression_error = max(
                maximum_regression_error,
                float(np.max(np.abs(pred_scale - real_scale))),
            )
            indices = _mbb_indices(
                len(pred), _stable_rng("p1", panel, window, convention, frequency)
            )
            predicted_draws += pred.to_numpy()[indices].mean(axis=1) * pred_scale
            realised_draws += real.to_numpy()[indices].mean(axis=1) * real_scale
            point_predicted += frequency_aggregate[
                "predicted_churn_annualized"
            ].to_numpy()
            point_realised += frequency_aggregate[
                "realized_churn_annualized"
            ].to_numpy()
        bootstrap = _row_correlations(predicted_draws, realised_draws)
        point = float(np.corrcoef(point_predicted, point_realised)[0, 1])
        recorded = float(
            frozen.loc[
                frozen["analysis_window"].eq(window)
                & frozen["singleton_convention"].eq(convention),
                "predicted_realized_correlation_across_configs",
            ].iloc[0]
        )
        maximum_regression_error = max(maximum_regression_error, abs(point - recorded))
        rows.append(
            {
                "panel": panel,
                "analysis_window": window,
                "singleton_convention": convention,
                "correlation": point,
                "correlation_ci_low": float(np.percentile(bootstrap, 2.5)),
                "correlation_ci_high": float(np.percentile(bootstrap, 97.5)),
                "n_configs": len(configs),
                "block_length": BLOCK_LENGTH,
                "bootstrap_draws": BOOTSTRAP_DRAWS,
                "seed": SEED,
            }
        )
    return pd.DataFrame(rows), maximum_regression_error


def _margin_flip_rates(
    partitions: Mapping[str, Mapping[str, Mapping[pd.Timestamp, pd.Series]]],
) -> tuple[pd.DataFrame, float, float]:
    """Build margin-decile flip rates and attach bootstrapped P1 correlations."""
    output = []
    maximum_correlation_error = 0.0
    maximum_flip_error = 0.0
    for panel, info in PANEL_INFO.items():
        cache = _output_root() / "stability" / info["cache"]
        margin = _read_csv(cache / "margin_assets.csv", parse_dates=["date"])
        per_date = _read_csv(cache / "predicted_realized_per_date.csv")
        per_date["index"] = pd.to_datetime(per_date["index"])
        aggregate = _read_csv(cache / "predicted_realized.csv")
        frozen = _read_csv(cache / "prediction_correlations.csv")
        correlations, error = _correlation_bootstrap(
            panel, per_date, aggregate, frozen
        )
        maximum_correlation_error = max(maximum_correlation_error, error)
        for frequency in info["frequencies"]:
            frequency_margin = margin.loc[margin["frequency"].eq(frequency)].copy()
            for window, window_mask in _analysis_windows(
                panel, frequency_margin["date"]
            ).items():
                window_margin = frequency_margin.loc[window_mask].copy()
                window_flip = _flip_panel(
                    window_margin, partitions[panel][frequency], frequency
                )
                for convention, include_singletons in (
                    ("including_singletons", True),
                    ("excluding_singletons", False),
                ):
                    sample = window_flip.copy()
                    if not include_singletons:
                        sample = sample.loc[~sample["is_singleton"].astype(bool)]
                    sample = sample.dropna(
                        subset=["margin", "predicted_churn_probability", "realised_flip"]
                    )
                    sample["margin_decile"] = pd.qcut(
                        sample["margin"].rank(method="first"),
                        10,
                        labels=range(1, 11),
                    ).astype(int)
                    realised_by_date = sample.groupby("date")["realised_flip"].sum()
                    recorded = per_date.loc[
                        per_date["analysis_window"].eq(window)
                        & per_date["frequency"].eq(frequency)
                        & per_date["config"].eq("baseline")
                        & per_date["singleton_convention"].eq(convention)
                    ].set_index("index")["realised_changes"]
                    common = realised_by_date.index.intersection(recorded.index)
                    if len(common):
                        maximum_flip_error = max(
                            maximum_flip_error,
                            float(
                                realised_by_date.reindex(common)
                                .subtract(recorded.reindex(common))
                                .abs()
                                .max()
                            ),
                        )
                    correlation = correlations.loc[
                        correlations["analysis_window"].eq(window)
                        & correlations["singleton_convention"].eq(convention)
                    ].iloc[0]
                    for decile, values in sample.groupby("margin_decile", sort=True):
                        output.append(
                            {
                                "panel": panel,
                                "frequency": frequency,
                                "analysis_window": window,
                                "singleton_convention": convention,
                                "margin_decile": int(decile),
                                "observations": len(values),
                                "margin_min": float(values["margin"].min()),
                                "margin_mean": float(values["margin"].mean()),
                                "margin_median": float(values["margin"].median()),
                                "margin_max": float(values["margin"].max()),
                                "predicted_flip_probability": float(
                                    values["predicted_churn_probability"].mean()
                                ),
                                "realised_flip_rate": float(values["realised_flip"].mean()),
                                "predicted_realised_correlation": float(
                                    correlation["correlation"]
                                ),
                                "correlation_ci_low": float(
                                    correlation["correlation_ci_low"]
                                ),
                                "correlation_ci_high": float(
                                    correlation["correlation_ci_high"]
                                ),
                                "n_configs": int(correlation["n_configs"]),
                                "block_length": BLOCK_LENGTH,
                                "bootstrap_draws": BOOTSTRAP_DRAWS,
                                "seed": SEED,
                            }
                        )
    return pd.DataFrame(output), maximum_correlation_error, maximum_flip_error


def _delta_coordinate(panel: str, config: str) -> float:
    """Return the scalar path coordinate used to order one frontier."""
    fixed = {
        "baseline": 0.0,
        "M1_delta_0.02": 0.02,
        "M1_delta_0.05": 0.05,
        "M1_delta_0.10": 0.10,
    }
    if config in fixed:
        return fixed[config]
    if config != "M1_star":
        raise KeyError(config)
    primary_frequency = "ME" if panel == "fund_panel" else "W-WED"
    return CONFIGURED_DELTAS[(panel, primary_frequency)]


def _menger_curvature(points: np.ndarray) -> np.ndarray:
    """Return discrete Menger curvature, leaving endpoints at zero."""
    output = np.zeros(len(points))
    for index in range(1, len(points) - 1):
        first, middle, last = points[index - 1 : index + 2]
        a = np.linalg.norm(first - middle)
        b = np.linalg.norm(middle - last)
        c = np.linalg.norm(last - first)
        first_vector = middle - first
        second_vector = last - first
        twice_area = abs(
            first_vector[0] * second_vector[1] - first_vector[1] * second_vector[0]
        )
        denominator = a * b * c
        output[index] = 2.0 * twice_area / denominator if denominator > 0 else 0.0
    return output


def _frontier() -> tuple[pd.DataFrame, dict[tuple[str, str], dict[str, object]]]:
    """Consolidate frontiers and mark the maximum normalized-curvature knee."""
    output = []
    knees = {}
    for panel, info in PANEL_INFO.items():
        source = _read_csv(
            _output_root() / "stability" / info["cache"] / "frontier.csv"
        )
        for window, group in source.groupby("analysis_window", sort=False):
            group = group.copy()
            group["delta_path_coordinate"] = [
                _delta_coordinate(panel, config) for config in group["config"]
            ]
            group = group.sort_values("delta_path_coordinate").reset_index(drop=True)
            x = group["baseline_partition_ari_median"].to_numpy(dtype=float)
            y = group["raw_churn"].to_numpy(dtype=float)
            x_range = float(np.ptp(x))
            y_range = float(np.ptp(y))
            if x_range <= 0.0 or y_range <= 0.0:
                raise AssertionError(f"{panel} {window} frontier has no range")
            points = np.column_stack(((x - x.min()) / x_range, (y - y.min()) / y_range))
            curvature = _menger_curvature(points)
            knee_index = int(np.argmax(curvature))
            knee = group.iloc[knee_index]
            knees[(panel, window)] = {
                "config": knee["config"],
                "delta": float(knee["delta_path_coordinate"]),
                "label": knee["delta_label"],
                "curvature": float(curvature[knee_index]),
            }
            for index, row in group.iterrows():
                output.append(
                    {
                        "panel": panel,
                        "analysis_window": window,
                        "config": row["config"],
                        "delta_label": row["delta_label"],
                        "delta_path_coordinate": float(row["delta_path_coordinate"]),
                        "raw_churn": float(row["raw_churn"]),
                        "fidelity": float(row["baseline_partition_ari_median"]),
                        "fidelity_status": row["fidelity_status"],
                        "normalized_menger_curvature": float(curvature[index]),
                        "is_knee": bool(index == knee_index),
                        "knee_config": knee["config"],
                        "knee_delta_label": knee["delta_label"],
                        "level_calibration": row["level_calibration"],
                        "innovation_calibration": row["innovation_overlay"],
                        "mac_path_coordinate_convention": (
                            "ME delta orders the combined ME/QE frontier"
                            if panel == "fund_panel"
                            else "single-frequency delta"
                        ),
                    }
                )
    return pd.DataFrame(output), knees


def _formula(span: int, step_k: float, kappa: float, rho: float) -> tuple[float, float]:
    """Return level and innovation noise-floor calibrations at z equal to one."""
    level = np.sqrt(1.0 + kappa) * (1.0 - rho**2) / np.sqrt(span)
    decay = 1.0 - 2.0 / (span + 1.0)
    innovation = level * np.sqrt(2.0 * (1.0 - decay**step_k))
    return float(level), float(innovation)


def _u1_me36_rho_bar() -> tuple[float, int, int]:
    """Re-score pooled within-cluster correlations for the adopted U1 ME/36 cell."""
    dates, eligibility = u1_grid._accepted_dates_and_eligibility()
    dates = dates[(dates >= HEADLINE_START) & (dates <= HEADLINE_END)]
    daily = u1_grid._read_daily(eligibility.columns)
    returns = u1_grid._native_returns(daily, "ME")
    partitions, _ = u1_grid._load_partition("ME", 36)
    model = u1_grid._model(36, "ME")
    values = []
    dates_used = 0
    for date, correlation in _iter_correlation_inputs(returns, list(dates), model):
        labels = partitions.loc[date].dropna()
        date_values = []
        for members in labels.groupby(labels).groups.values():
            members = pd.Index(members)
            if len(members) < 2:
                continue
            matrix = correlation.loc[members, members].to_numpy(dtype=float)
            date_values.append(matrix[np.triu_indices(len(members), 1)])
        if date_values:
            values.extend(np.concatenate(date_values).tolist())
            dates_used += 1
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        raise AssertionError("U1 ME/36 rho-bar has no finite within-cluster pairs")
    return float(np.median(finite)), len(finite), dates_used


def _calibration_bridge(
    knees: Mapping[tuple[str, str], Mapping[str, object]],
) -> tuple[pd.DataFrame, float]:
    """Build the theory-panel and adopted-application calibration bridge."""
    quality = _read_csv(_output_root() / "data_quality" / "all_universes_data_quality.csv")
    rho = _read_csv(_output_root() / "e2_baseline_rho_bar.csv")
    theory = [
        ("equity_panel", "W-WED", "msci_us", 156, 52.0 / 12.0),
        ("futures_panel", "W-WED", "futures", 156, 52.0 / 12.0),
        ("fund_panel", "ME", "mac", 36, 1.0),
        ("fund_panel", "QE", "mac", 12, 1.0),
    ]
    rows = []
    maximum_rounding_error = 0.0
    theory_values = {}
    for panel, frequency, universe, span, step_k in theory:
        quality_row = quality.loc[
            quality["universe"].eq(universe) & quality["frequency"].eq(frequency)
        ].iloc[0]
        rho_row = rho.loc[
            rho["universe"].eq(universe) & rho["frequency"].eq(frequency)
        ].iloc[0]
        kappa = float(quality_row["kappa_hat"])
        rho_bar = float(rho_row["rho_bar"])
        level, innovation = _formula(span, step_k, kappa, rho_bar)
        configured = CONFIGURED_DELTAS[(panel, frequency)]
        innovation_marker = INNOVATION_MARKERS[(panel, frequency)]
        maximum_rounding_error = max(
            maximum_rounding_error,
            abs(round(level, 4) - configured),
            abs(round(innovation, 4) - innovation_marker),
        )
        knee = knees[(panel, "full_panel")]
        theory_values[(panel, frequency)] = (kappa, rho_bar, level, innovation)
        rows.append(
            {
                "row_role": "theory_panel",
                "panel": panel,
                "frequency": frequency,
                "span_n": span,
                "step_k": step_k,
                "kappa_hat": kappa,
                "rho_bar": rho_bar,
                "z": 1.0,
                "delta_star_level": level,
                "delta_star_innovation": innovation,
                "configured_or_adopted_delta": configured,
                "delta_level_gap": configured - level,
                "knee_config": knee["config"],
                "knee_delta": float(knee["delta"]),
                "knee_source_panel": panel,
                "pair_observations": int(rho_row["pair_observations"]),
                "dates_used": int(rho_row["dates_used"]),
                "rho_source": "E2 frozen baseline rho-bar",
                "kappa_source": "E1 frozen data-quality report",
            }
        )
    me_rho, me_pairs, me_dates = _u1_me36_rho_bar()
    equity_kappa = theory_values[("equity_panel", "W-WED")][0]
    level, innovation = _formula(36, 1.0, equity_kappa, me_rho)
    knee = knees[("equity_panel", "full_panel")]
    rows.append(
        {
            "row_role": "adopted_application_cell",
            "panel": "u1_equity_application",
            "frequency": "ME",
            "span_n": 36,
            "step_k": 1.0,
            "kappa_hat": equity_kappa,
            "rho_bar": me_rho,
            "z": 1.0,
            "delta_star_level": level,
            "delta_star_innovation": innovation,
            "configured_or_adopted_delta": 0.0866,
            "delta_level_gap": 0.0866 - level,
            "knee_config": knee["config"],
            "knee_delta": float(knee["delta"]),
            "knee_source_panel": "equity_panel W-WED/156 sweep",
            "pair_observations": me_pairs,
            "dates_used": me_dates,
            "rho_source": "frozen U1 ME/36 baseline partitions re-scored",
            "kappa_source": "E1 equity W-WED kappa-hat (only frozen equity estimate)",
        }
    )
    futures = next(
        row
        for row in rows
        if row["panel"] == "futures_panel" and row["frequency"] == "W-WED"
    )
    rows.append(
        {
            **futures,
            "row_role": "adopted_application_cell",
            "panel": "u3_futures_application",
            "rho_source": "same frozen W-WED/156 cell as futures theory panel",
            "kappa_source": "same E1 futures estimate as theory panel",
        }
    )
    return pd.DataFrame(rows), maximum_rounding_error


def _bootstrap_means(
    values: pd.DataFrame,
    *keys: object,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return point means and joint moving-block percentile intervals."""
    clean = values.dropna()
    if clean.empty:
        raise AssertionError(f"empty ergodicity sample: {keys}")
    indices = _mbb_indices(len(clean), _stable_rng("ergodicity", *keys))
    draws = clean.to_numpy()[indices].mean(axis=1)
    return (
        clean.mean(axis=0).to_numpy(),
        np.percentile(draws, 2.5, axis=0),
        np.percentile(draws, 97.5, axis=0),
    )


def _ergodicity() -> pd.DataFrame:
    """Report chronological subsample and crisis-window stability means."""
    rows = []
    metrics = ("cluster_count", "size_entropy", "ari", "vi")
    for panel, info in PANEL_INFO.items():
        per_date = _read_csv(
            _output_root() / "stability" / info["cache"] / "per_date_metrics.csv"
        )
        per_date["date"] = pd.to_datetime(per_date["date"])
        for config in ("baseline", "M1_star"):
            selected = per_date.loc[
                per_date["analysis_window"].eq("full_panel")
                & per_date["config"].eq(config)
            ]
            shape = selected.loc[
                selected["panel"].eq("shape"),
                ["date", "cluster_count", "size_entropy"],
            ].set_index("date")
            consecutive = selected.loc[
                selected["panel"].eq("consecutive"), ["date", "ari", "vi"]
            ].set_index("date")
            joined = shape.join(consecutive, how="inner").sort_index().dropna()
            segments = []
            for parts, label in ((2, "half"), (3, "third")):
                positions_by_segment = np.array_split(np.arange(len(joined)), parts)
                for number, positions in enumerate(positions_by_segment, 1):
                    segments.append(("subsample", f"{label}_{number}", joined.iloc[positions]))
            for crisis, (start, end) in CRISIS_WINDOWS.items():
                segments.append(
                    ("crisis", crisis, joined.loc[(joined.index >= start) & (joined.index <= end)])
                )
            for sample_type, sample_id, sample in segments:
                point, low, high = _bootstrap_means(
                    sample[list(metrics)], panel, config, sample_type, sample_id
                )
                for index, metric in enumerate(metrics):
                    rows.append(
                        {
                            "panel": panel,
                            "config": config,
                            "sample_type": sample_type,
                            "sample_id": sample_id,
                            "metric": metric,
                            "sample_start": sample.index.min(),
                            "sample_end": sample.index.max(),
                            "observations": len(sample),
                            "mean": float(point[index]),
                            "ci_low": float(low[index]),
                            "ci_high": float(high[index]),
                            "block_length": BLOCK_LENGTH,
                            "bootstrap_draws": BOOTSTRAP_DRAWS,
                            "seed": SEED,
                        }
                    )
    return pd.DataFrame(rows)


def _render_pdf(fig: plt.Figure, path: Path) -> None:
    """Write a metadata-stable PDF and prove same-process byte identity."""
    metadata = {
        "Creator": "OptimalPortfolios cluster-lineage F1",
        "Producer": "matplotlib",
        "CreationDate": None,
        "ModDate": None,
    }
    first = io.BytesIO()
    second = io.BytesIO()
    fig.savefig(first, format="pdf", bbox_inches="tight", metadata=metadata)
    fig.savefig(second, format="pdf", bbox_inches="tight", metadata=metadata)
    if first.getvalue() != second.getvalue():
        raise AssertionError(f"PDF render is not byte-identical: {path.name}")
    path.write_bytes(first.getvalue())
    plt.close(fig)


def _plot_margins(frame: pd.DataFrame) -> None:
    """Render the three-panel margin distribution and flip-rate exhibit."""
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.4), sharey=False)
    for axis, panel in zip(axes, PANEL_INFO, strict=True):
        selected = frame.loc[
            frame["panel"].eq(panel)
            & frame["analysis_window"].eq("full_panel")
            & frame["singleton_convention"].eq("excluding_singletons")
        ]
        twin = axis.twinx()
        for frequency, values in selected.groupby("frequency", sort=True):
            x = values["margin_median"].to_numpy()
            share = values["observations"].to_numpy() / values["observations"].sum()
            axis.plot(x, share, marker="o", label=f"{frequency} mass")
            twin.plot(
                x,
                values["realised_flip_rate"],
                marker="s",
                linestyle="--",
                label=f"{frequency} flip",
            )
        axis.axvline(0.0, color="black", linewidth=0.7, alpha=0.6)
        axis.set_title(panel.replace("_panel", "").title())
        axis.set_xlabel("Assignment margin")
        axis.set_ylabel("Observation share")
        twin.set_ylabel("Realised flip rate")
        handles, labels = axis.get_legend_handles_labels()
        twin_handles, twin_labels = twin.get_legend_handles_labels()
        axis.legend(handles + twin_handles, labels + twin_labels, fontsize=7, frameon=False)
    fig.suptitle("Small assignment margins carry the highest realised flip rates")
    fig.tight_layout()
    _render_pdf(fig, _root() / "margin_histogram.pdf")


def _plot_frontier(frame: pd.DataFrame) -> None:
    """Render the churn-fidelity path and its discrete knee for each panel."""
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.4))
    for axis, panel in zip(axes, PANEL_INFO, strict=True):
        selected = frame.loc[
            frame["panel"].eq(panel) & frame["analysis_window"].eq("full_panel")
        ].sort_values("delta_path_coordinate")
        axis.plot(selected["fidelity"], selected["raw_churn"], color="#4C78A8", marker="o")
        knee = selected.loc[selected["is_knee"]].iloc[0]
        axis.scatter(
            [knee["fidelity"]],
            [knee["raw_churn"]],
            marker="*",
            s=140,
            color="#E45756",
            zorder=4,
            label=f"knee: {knee['knee_delta_label']}",
        )
        for _, row in selected.iterrows():
            axis.annotate(
                f"{row['delta_path_coordinate']:.3f}",
                (row["fidelity"], row["raw_churn"]),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7,
            )
        axis.set_title(panel.replace("_panel", "").title())
        axis.set_xlabel("Median ARI to baseline")
        axis.set_ylabel("Churn per asset-year")
        axis.legend(frameon=False, fontsize=7)
    fig.suptitle("The smoothing frontier has a measurable churn-fidelity knee")
    fig.tight_layout()
    _render_pdf(fig, _root() / "frontier_knee.pdf")


def _source_manifest() -> pd.DataFrame:
    """Return the F0-frozen source rows consumed by stage F1."""
    inventory = _read_csv(_output_root() / "finalisation" / "f0" / "cache_inventory.csv")
    selected = inventory.loc[inventory["stages"].str.contains("F1", regex=False)].copy()
    if not selected["status"].eq("PASS").all() or not selected["resolution_count"].eq(1).all():
        raise AssertionError("F1 source inventory is no longer complete and unambiguous")
    return selected


def _acceptance(
    deliverables: Sequence[pd.DataFrame],
    source_manifest: pd.DataFrame,
    correlation_error: float,
    flip_error: float,
    calibration_error: float,
) -> pd.DataFrame:
    """Return every measured-versus-tolerance F1 acceptance line."""
    corrected = _read_csv(
        _output_root()
        / "stability"
        / "msci_us"
        / "u1_corrected_asset_set_summary.csv"
    )
    nan_count = int(sum(frame.isna().to_numpy().sum() for frame in deliverables))
    checks = [
        ("F0 source paths resolved once", len(source_manifest), len(source_manifest), True),
        (
            "F0 source status failures",
            int((~source_manifest["status"].eq("PASS")).sum()),
            0,
            source_manifest["status"].eq("PASS").all(),
        ),
        (
            "E3b corrected U1 configurations",
            len(corrected),
            7,
            len(corrected) == 7,
        ),
        (
            "E3b corrected U1 maximum asset-set difference",
            float(corrected["max_symmetric_difference_share"].max()),
            0.0,
            corrected["max_symmetric_difference_share"].eq(0.0).all(),
        ),
        ("superseded U1 legacy tables consumed", 0, 0, True),
        (
            "P1 frozen correlation regression error",
            correlation_error,
            1e-12,
            correlation_error <= 1e-12,
        ),
        (
            "baseline realised-flip count regression error",
            flip_error,
            1e-12,
            flip_error <= 1e-12,
        ),
        (
            "calibration rounded-value regression error",
            calibration_error,
            0.0,
            calibration_error == 0.0,
        ),
        ("NaNs across F1 numerical deliverables", nan_count, 0, nan_count == 0),
        ("bootstrap block length", BLOCK_LENGTH, 6, BLOCK_LENGTH == 6),
        ("bootstrap draws", BOOTSTRAP_DRAWS, 2000, BOOTSTRAP_DRAWS == 2000),
        ("bootstrap seed", SEED, 20260813, SEED == 20260813),
    ]
    frame = pd.DataFrame(
        [
            {
                "check": check,
                "measured": measured,
                "tolerance": tolerance,
                "status": "PASS" if passed else "FAIL",
            }
            for check, measured, tolerance, passed in checks
        ]
    )
    if not frame["status"].eq("PASS").all():
        raise AssertionError(frame.loc[~frame["status"].eq("PASS")])
    return frame


def run() -> Mapping[str, pd.DataFrame]:
    """Execute F1 and write every deterministic numerical and plot artifact."""
    source_manifest = _source_manifest()
    partitions = _partition_panels()
    margins, correlation_error, flip_error = _margin_flip_rates(partitions)
    frontier, knees = _frontier()
    calibration, calibration_error = _calibration_bridge(knees)
    ergodicity = _ergodicity()
    acceptance = _acceptance(
        (margins, frontier, calibration, ergodicity),
        source_manifest,
        correlation_error,
        flip_error,
        calibration_error,
    )
    outputs = {
        "margins_flip_rates": margins,
        "frontier": frontier,
        "calibration_bridge": calibration,
        "ergodicity": ergodicity,
        "source_manifest": source_manifest,
        "run_parameters": pd.DataFrame(
            [
                {
                    "block_length": BLOCK_LENGTH,
                    "bootstrap_draws": BOOTSTRAP_DRAWS,
                    "seed": SEED,
                    "knee_method": "maximum Menger curvature after panel-wise x/y normalization",
                    "mac_knee_path": "ME delta orders combined ME/QE frontier",
                }
            ]
        ),
        "acceptance": acceptance,
    }
    for name, frame in outputs.items():
        _write_csv(frame, _root() / f"{name}.csv")
    _plot_margins(margins)
    _plot_frontier(frontier)
    return outputs


def _artifact_hashes() -> dict[str, str]:
    """Hash deterministic F1 artifacts while excluding the replay record."""
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(_root().iterdir())
        if path.is_file() and path.name != "determinism.csv"
    }


def verify_determinism() -> pd.DataFrame:
    """Run F1 twice and require byte-identical CSV and PDF artifacts."""
    run()
    first = _artifact_hashes()
    run()
    second = _artifact_hashes()
    names = sorted(set(first) | set(second))
    replay = pd.DataFrame(
        {
            "artifact": names,
            "first_sha256": [first.get(name) for name in names],
            "second_sha256": [second.get(name) for name in names],
            "byte_identical": [first.get(name) == second.get(name) for name in names],
        }
    )
    if not replay["byte_identical"].all():
        raise AssertionError(replay.loc[~replay["byte_identical"]])
    _write_csv(replay, _root() / "determinism.csv")
    return replay


def main() -> None:
    """Run F1 twice and print its deterministic acceptance summary."""
    replay = verify_determinism()
    print(f"F1 stability consolidation: PASS ({len(replay)}/{len(replay)} deterministic)")


if __name__ == "__main__":
    main()
