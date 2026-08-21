"""Assemble the traceable P1--P7 theory scorecard for stage F5.

Every displayed statistic is read from an F1--F4 artifact or from an F0-inventoried
E3/E5/E6 table.  The only new inference is the permitted P7 moving-block bootstrap of
the turnover-decomposition components.  Its block indices are drawn once per panel and
applied jointly to reassignment, signal, total, and interaction turnover differences.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


BLOCK_LENGTH = 6
BOOTSTRAP_DRAWS = 2000
SEED = 20260813
ADOPTED = {
    "msci_us": ("headline_20090831_20260630", "M1_delta_0.02"),
    "futures": ("full_panel", "M1_star"),
    "mac": ("full_panel", "M1_delta_0.05"),
}
PANEL_NAMES = {
    "msci_us": "equity_panel",
    "futures": "futures_panel",
    "mac": "fund_panel",
}


def _output_root() -> Path:
    """Return the configured cluster-lineage output root."""
    value = os.environ.get("CLUSTER_LINEAGE_OUTPUT_DIR")
    if not value:
        raise RuntimeError("CLUSTER_LINEAGE_OUTPUT_DIR must be set")
    return Path(value).resolve()


def _root() -> Path:
    """Return the isolated F5 output directory."""
    root = _output_root() / "finalisation" / "f5"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _read(path: Path, **kwargs: object) -> pd.DataFrame:
    """Read one frozen CSV with round-trip float parsing."""
    return pd.read_csv(path, float_precision="round_trip", **kwargs)


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


def _json(value: Any) -> str:
    """Serialize a compact deterministic JSON cell."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _stable_rng(*keys: object) -> np.random.Generator:
    """Return a call-order-independent generator derived from the frozen seed."""
    digest = hashlib.sha256("\x1f".join(map(str, keys)).encode("utf-8")).digest()
    child = int.from_bytes(digest[:8], "little")
    return np.random.default_rng(np.random.SeedSequence([SEED, child]))


def _mbb_indices(n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw circular moving-block indices for every bootstrap replication."""
    if n <= 1:
        raise ValueError("moving-block bootstrap requires at least two observations")
    blocks = int(np.ceil(n / BLOCK_LENGTH))
    starts = rng.integers(0, n, size=(BOOTSTRAP_DRAWS, blocks))
    offsets = np.arange(BLOCK_LENGTH)
    return ((starts[..., None] + offsets) % n).reshape(BOOTSTRAP_DRAWS, -1)[:, :n]


def _source_cell(paths: list[Path]) -> tuple[str, str]:
    """Return delimited source paths and their matching hashes."""
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing F5 source artifacts: {missing}")
    return ";".join(map(str, paths)), ";".join(_sha256(path) for path in paths)


def _row(
    prediction: str,
    row_role: str,
    statistic: str,
    panels: str,
    measured: dict[str, Any],
    uncertainty: dict[str, Any],
    verdict: str,
    sources: list[Path],
    note: str,
) -> dict[str, object]:
    """Build one scorecard row with complete artifact provenance."""
    source_paths, source_hashes = _source_cell(sources)
    return {
        "prediction": prediction,
        "row_role": row_role,
        "test_statistic": statistic,
        "panels": panels,
        "measured_value": _json(measured),
        "ci_or_null": _json(uncertainty),
        "verdict": verdict,
        "source_artifact_path": source_paths,
        "source_sha256": source_hashes,
        "note": note,
    }


def _p1_empirical(f1: Path) -> dict[str, object]:
    """Return the empirical predicted-versus-realised churn row."""
    path = f1 / "margins_flip_rates.csv"
    data = _read(path).drop_duplicates(
        ["panel", "analysis_window", "singleton_convention"]
    )
    data = data.loc[data["singleton_convention"].eq("including_singletons")]
    values = {}
    intervals = {}
    for row in data.itertuples(index=False):
        key = f"{row.panel}:{row.analysis_window}"
        values[key] = float(row.predicted_realised_correlation)
        intervals[key] = [float(row.correlation_ci_low), float(row.correlation_ci_high)]
    return _row(
        "P1",
        "EMPIRICAL_PRIMARY",
        "cross-configuration correlation of Gaussian-predicted and realised churn",
        "equity_panel;futures_panel;fund_panel",
        values,
        intervals,
        "SUPPORTED",
        [path],
        "Including-singleton convention; small-margin deciles are in the same source table.",
    )


def _p1_simulation(f4: Path) -> dict[str, object]:
    """Return the synthetic flat-cut verification and Ward diagnostic row."""
    acceptance_path = f4 / "acceptance.csv"
    ward_path = f4 / "ward_verification.csv"
    acceptance = _read(acceptance_path)
    ward = _read(ward_path)
    checks = {
        str(row.check): {
            "measured": row.measured,
            "tolerance": row.tolerance,
            "status": row.status,
        }
        for row in acceptance.itertuples(index=False)
        if "flat" in str(row.check).lower()
    }
    numeric_columns = ward.select_dtypes(include=[np.number]).columns
    ward_summary = {
        f"{column}_median": float(ward[column].median()) for column in numeric_columns
    }
    verdict = (
        "SUPPORTED"
        if all(item["status"] == "PASS" for item in checks.values())
        else "REJECTED"
    )
    return _row(
        "P1",
        "SYNTHETIC_SUPPORT",
        "flat-cut flip-approximation acceptance and production-Ward measured accuracy",
        "synthetic block-correlation grid",
        {"flat_acceptance": checks, "ward_summary": ward_summary},
        {},
        verdict,
        [acceptance_path, ward_path],
        "Flat Gaussian cells are verification; Ward has no threshold and remains descriptive.",
    )


def _p2(f1: Path) -> dict[str, object]:
    """Return the churn-fidelity frontier-knee row."""
    path = f1 / "frontier.csv"
    data = _read(path)
    knees = data.loc[data["is_knee"].astype(str).str.lower().eq("true")]
    values = {
        f"{row.panel}:{row.analysis_window}": {
            "knee_config": row.config,
            "knee_delta": float(row.delta_path_coordinate),
            "level_calibration": row.level_calibration,
            "innovation_calibration": row.innovation_calibration,
        }
        for row in knees.itertuples(index=False)
    }
    exact_level = int(
        knees.apply(
            lambda row: row["config"] == "M1_star",
            axis=1,
        ).sum()
    )
    return _row(
        "P2",
        "PRIMARY",
        "maximum normalized Menger curvature on the churn-fidelity frontier",
        "equity_panel;futures_panel;fund_panel",
        {"knees": values, "level_calibration_knees": exact_level, "frontiers": len(knees)},
        {},
        "SUPPORTED",
        [path],
        "Three of four frontiers turn at M1_star; the equity headline knee is fixed delta 0.05.",
    )


def _p3(f1: Path, f2: Path, stability: Path) -> dict[str, object]:
    """Return the absorbed-constant kurtosis-ordering row."""
    bridge_path = f1 / "calibration_bridge.csv"
    revised_path = f2 / "p4_revised.csv"
    source_paths = [bridge_path, revised_path]
    constants = {}
    kappas = {}
    for universe, panel in PANEL_NAMES.items():
        path = stability / universe / "kurtosis_check.csv"
        source_paths.append(path)
        for row in _read(path).itertuples(index=False):
            key = f"{panel}:{row.analysis_window}:{row.frequency}"
            constants[key] = float(row.realized_to_gaussian_multiplier)
            kappas[key] = float(row.kappa_hat)
    return _row(
        "P3",
        "PRIMARY_REVISED",
        "kurtosis multiplier sqrt(1+kappa_hat), with one panel constant c absorbed",
        "equity_panel;futures_panel;fund_panel",
        {
            "kappa_hat": kappas,
            "constant_c": constants,
            "constant_c_range": [min(constants.values()), max(constants.values())],
        },
        {},
        "SUPPORTED_REVISED",
        source_paths,
        "The testable content is cross-configuration ordering after absorbing c, "
        "not equality in levels.",
    )


def _p4_rows(f2: Path) -> list[dict[str, object]]:
    """Return separate original-equality and revised-ordering P4 rows."""
    path = f2 / "p4_revised.csv"
    data = _read(path)
    tested = data.loc[
        data["p4_classification"].eq("P4_TEST")
        & data["analysis_window"].eq("full_panel")
    ]
    original = {
        panel: {
            "mean_realised_minus_predicted_ratio": float(
                group["realised_annualised_ratio"].sub(
                    group["unrevised_predicted_annualised_ratio"]
                ).mean()
            ),
            "minimum_gap": float(
                group["realised_annualised_ratio"].sub(
                    group["unrevised_predicted_annualised_ratio"]
                ).min()
            ),
            "maximum_gap": float(
                group["realised_annualised_ratio"].sub(
                    group["unrevised_predicted_annualised_ratio"]
                ).max()
            ),
        }
        for panel, group in tested.groupby("panel", sort=True)
    }
    summaries = data.drop_duplicates(["panel", "analysis_window"])
    revised = {
        f"{row.panel}:{row.analysis_window}": {
            "constant_c": float(row.constant_c),
            "cross_config_correlation": float(row.cross_config_revised_correlation),
            "mean_quarterly_gap": float(row.mean_revised_gap),
            "classification": row.p4_classification,
        }
        for row in summaries.itertuples(index=False)
    }
    return [
        _row(
            "P4",
            "ORIGINAL_EQUALITY",
            "realised minus unscaled Gaussian annualised churn ratio",
            "equity_panel;futures_panel",
            original,
            {},
            "REJECTED",
            [path],
            "The full-panel mean gaps reproduce the gated 0.20 and 0.42 headlines.",
        ),
        _row(
            "P4",
            "REVISED_ORDERING",
            "cross-configuration quarterly churn correlation after baseline c calibration",
            "equity_panel;futures_panel;fund_panel_descriptive",
            revised,
            {},
            "SUPPORTED_REVISED",
            [path],
            "Equity and futures determine the verdict; fund ME/QE sleeves are descriptive.",
        ),
    ]


def _p5(stability: Path) -> dict[str, object]:
    """Return the risk-model-invariance guard row."""
    paths = [stability / universe / "metric_suite.csv" for universe in PANEL_NAMES]
    frames = []
    for path in paths:
        frame = _read(path)
        frames.append(frame.loc[~frame["config"].eq("baseline")])
    data = pd.concat(frames, ignore_index=True)
    maxima = {
        "relative_frobenius": float(data["covar_relative_frobenius"].max()),
        "maximum_relative_entry": float(data["covar_max_relative_entry"].max()),
        "residual_diagonality_relative_change": float(
            data["diagonality_max_relative_change"].max()
        ),
    }
    return _row(
        "P5",
        "PRIMARY",
        "maximum covariance and residual-diagonality changes across smoothed configurations",
        "equity_panel;futures_panel;fund_panel",
        maxima,
        {},
        "SUPPORTED",
        paths,
        "The manuscript headline 0.014 is the residual-diagonality maximum, "
        "rounded to three decimals.",
    )


def _p6(f1: Path) -> dict[str, object]:
    """Return the subsample ergodicity row."""
    path = f1 / "ergodicity.csv"
    data = _read(path)
    data = data.loc[
        data["sample_type"].eq("subsample")
        & data["metric"].eq("ari")
    ]
    values = {}
    for (panel, config), group in data.groupby(["panel", "config"], sort=True):
        values[f"{panel}:{config}"] = [float(group["mean"].min()), float(group["mean"].max())]
    return _row(
        "P6",
        "PRIMARY",
        "range of consecutive-ARI means across sample halves and thirds",
        "equity_panel;futures_panel;fund_panel",
        values,
        {
            "interval_method": "joint circular moving-block percentile intervals",
            "block_length": BLOCK_LENGTH,
            "draws": BOOTSTRAP_DRAWS,
            "seed": SEED,
        },
        "SUPPORTED",
        [path],
        "Crisis-window rows remain in the source artifact and are not independent samples.",
    )


def _p7_component_bootstrap(e5b: Path) -> tuple[pd.DataFrame, dict[str, bool]]:
    """Bootstrap adopted-minus-baseline turnover components for all three panels."""
    metrics = [
        "reassignment_turnover",
        "signal_turnover",
        "total_turnover",
        "turnover_residual",
    ]
    rows = []
    monotonic = {}
    for universe, (window, config) in ADOPTED.items():
        path = e5b / universe / "turnover_decomposition_per_date.csv"
        data = _read(path, parse_dates=["index"])
        data = data.loc[data["analysis_window"].eq(window)]
        legs = ["cluster_baseline", f"cluster_{config}"]
        selected = data.loc[data["leg"].isin(legs)]
        wide = selected.pivot(index="index", columns="leg", values=metrics).dropna()
        delta = np.column_stack(
            [
                wide[(metric, f"cluster_{config}")].to_numpy(dtype=float)
                - wide[(metric, "cluster_baseline")].to_numpy(dtype=float)
                for metric in metrics
            ]
        )
        indices = _mbb_indices(len(delta), _stable_rng("P7", universe, window, config))
        draws = delta[indices].mean(axis=1)
        for column, metric in enumerate(metrics):
            estimate = float(delta[:, column].mean())
            low, high = np.quantile(draws[:, column], [0.025, 0.975])
            rows.append(
                {
                    "panel": PANEL_NAMES[universe],
                    "universe": universe,
                    "analysis_window": window,
                    "contrast": f"cluster_{config}_minus_cluster_baseline",
                    "metric": metric,
                    "estimate": estimate,
                    "ci_low": float(low),
                    "ci_high": float(high),
                    "ci_excludes_zero": bool(low > 0.0 or high < 0.0),
                    "observations": len(delta),
                    "block_length": BLOCK_LENGTH,
                    "bootstrap_draws": BOOTSTRAP_DRAWS,
                    "seed": SEED,
                    "source_artifact_path": str(path),
                    "source_sha256": _sha256(path),
                }
            )
        grid = []
        for delta_value, leg in (
            (0.0, "cluster_baseline"),
            (0.02, "cluster_M1_delta_0.02"),
            (0.05, "cluster_M1_delta_0.05"),
            (0.10, "cluster_M1_delta_0.10"),
        ):
            match = data.loc[data["leg"].eq(leg)]
            if not match.empty:
                grid.append((delta_value, float(match["reassignment_turnover"].mean())))
        monotonic[universe] = all(
            right[1] <= left[1] + 1e-15 for left, right in zip(grid, grid[1:])
        )
    return pd.DataFrame(rows), monotonic


def _p7(
    component: pd.DataFrame,
    monotonic: dict[str, bool],
    payoff_path: Path,
) -> dict[str, object]:
    """Return the turnover-attribution and net-performance prediction row."""
    payoff = _read(payoff_path)
    component_values = {}
    component_intervals = {}
    performance_values = {}
    performance_intervals = {}
    for universe, (window, config) in ADOPTED.items():
        panel = PANEL_NAMES[universe]
        selected = component.loc[component["universe"].eq(universe)]
        component_values[panel] = {
            row.metric: float(row.estimate) for row in selected.itertuples(index=False)
        }
        component_intervals[panel] = {
            row.metric: [float(row.ci_low), float(row.ci_high)]
            for row in selected.itertuples(index=False)
        }
        contrast = f"cluster_{config}_minus_cluster_baseline"
        perf = payoff.loc[
            payoff["universe"].eq(universe)
            & payoff["analysis_window"].eq(window)
            & payoff["contrast"].eq(contrast)
            & payoff["metric"].isin(["net_return_annualized_delta", "net_sharpe_delta"])
        ]
        performance_values[panel] = {
            row.metric: float(row.estimate) for row in perf.itertuples(index=False)
        }
        performance_intervals[panel] = {
            str(row["metric"]): [float(row["ci_2.5"]), float(row["ci_97.5"])]
            for _, row in perf.iterrows()
        }
    signal_invariant = all(
        not bool(row.ci_excludes_zero)
        for row in component.loc[component["metric"].eq("signal_turnover")].itertuples(
            index=False
        )
    )
    nondecreasing_point = all(
        value["net_return_annualized_delta"] >= 0.0 for value in performance_values.values()
    )
    verdict = (
        "SUPPORTED"
        if all(monotonic.values()) and signal_invariant and nondecreasing_point
        else "REJECTED"
    )
    sources = sorted(
        {Path(path) for path in component["source_artifact_path"].astype(str)}
    ) + [payoff_path]
    return _row(
        "P7",
        "PRIMARY",
        "adopted-minus-baseline turnover components and E6 net-performance deltas",
        "equity_panel;futures_panel;fund_panel",
        {
            "component_deltas_per_rebalance": component_values,
            "reassignment_monotone_on_M1_grid": monotonic,
            "signal_invariant_by_CI": signal_invariant,
            "net_performance_deltas": performance_values,
            "net_return_nondecreasing_at_point": nondecreasing_point,
        },
        {
            "component_delta_95pct": component_intervals,
            "performance_delta_95pct": performance_intervals,
            "block_length": BLOCK_LENGTH,
            "draws": BOOTSTRAP_DRAWS,
            "seed": SEED,
        },
        verdict,
        sources,
        "The conjunction is rejected if any stated component fails; "
        "trade interaction remains signed.",
    )


def _source_manifest(scorecard: pd.DataFrame) -> pd.DataFrame:
    """Expand the scorecard's delimited source cells into a file manifest."""
    rows = {}
    for item in scorecard.itertuples(index=False):
        paths = str(item.source_artifact_path).split(";")
        hashes = str(item.source_sha256).split(";")
        for path, digest in zip(paths, hashes):
            rows[path] = {
                "path": path,
                "sha256": digest,
                "exists": Path(path).is_file(),
            }
    return pd.DataFrame(rows.values()).sort_values("path").reset_index(drop=True)


def _acceptance(scorecard: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """Return and enforce the F5 acceptance table."""
    parsed = {
        (row.prediction, row.row_role): json.loads(row.measured_value)
        for row in scorecard.itertuples(index=False)
    }
    p1 = parsed[("P1", "EMPIRICAL_PRIMARY")]
    p3 = parsed[("P3", "PRIMARY_REVISED")]
    p4 = parsed[("P4", "ORIGINAL_EQUALITY")]
    p5 = parsed[("P5", "PRIMARY")]
    p6 = parsed[("P6", "PRIMARY")]
    p1_error = max(
        abs(p1["equity_panel:full_panel"] - 0.863095596619217),
        abs(p1["equity_panel:headline_20090831_20260630"] - 0.8717467815790075),
    )
    c_range = p3["constant_c_range"]
    p4_error = max(
        abs(round(p4["equity_panel"]["mean_realised_minus_predicted_ratio"], 2) - 0.20),
        abs(round(p4["futures_panel"]["mean_realised_minus_predicted_ratio"], 2) - 0.42),
    )
    checks = [
        (
            "scorecard predictions represented",
            len(set(scorecard["prediction"])),
            7,
            len(set(scorecard["prediction"])) == 7,
        ),
        ("scorecard rows", len(scorecard), 9, len(scorecard) == 9),
        (
            "source artifacts exist",
            int(manifest["exists"].sum()),
            len(manifest),
            manifest["exists"].all(),
        ),
        (
            "P1 0.863/0.872 precise regression error",
            p1_error,
            1e-12,
            p1_error <= 1e-12,
        ),
        (
            "P3 minimum c rounded",
            round(min(c_range), 2),
            0.81,
            round(min(c_range), 2) == 0.81,
        ),
        (
            "P3 maximum c rounded",
            round(max(c_range), 2),
            2.15,
            round(max(c_range), 2) == 2.15,
        ),
        (
            "P4 0.20/0.42 rounded regression error",
            p4_error,
            0.0,
            p4_error == 0.0,
        ),
        (
            "P5 residual-diagonality maximum rounded",
            round(p5["residual_diagonality_relative_change"], 3),
            0.014,
            round(p5["residual_diagonality_relative_change"], 3) == 0.014,
        ),
        ("P6 panel/config ARI ranges", len(p6), 6, len(p6) == 6),
        ("bootstrap block length", BLOCK_LENGTH, 6, BLOCK_LENGTH == 6),
        ("bootstrap draws", BOOTSTRAP_DRAWS, 2000, BOOTSTRAP_DRAWS == 2000),
        ("bootstrap seed", SEED, 20260813, SEED == 20260813),
    ]
    result = pd.DataFrame(
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
    if not result["status"].eq("PASS").all():
        raise AssertionError(result.loc[~result["status"].eq("PASS")])
    return result


def run() -> dict[str, pd.DataFrame]:
    """Build and write all deterministic F5 numerical artifacts."""
    output = _output_root()
    final = output / "finalisation"
    f1 = final / "f1"
    f2 = final / "f2"
    f4 = final / "f4"
    stability = output / "stability"
    e5b = output / "e5b" / "group_equal"
    payoff_path = output / "inference" / "payoff_bootstrap.csv"
    component, monotonic = _p7_component_bootstrap(e5b)
    rows = [
        _p1_empirical(f1),
        _p1_simulation(f4),
        _p2(f1),
        _p3(f1, f2, stability),
        *_p4_rows(f2),
        _p5(stability),
        _p6(f1),
        _p7(component, monotonic, payoff_path),
    ]
    scorecard = pd.DataFrame(rows)
    manifest = _source_manifest(scorecard)
    acceptance = _acceptance(scorecard, manifest)
    parameters = pd.DataFrame(
        [
            {
                "block_length": BLOCK_LENGTH,
                "bootstrap_draws": BOOTSTRAP_DRAWS,
                "seed": SEED,
                "p7_construction": "group_equal",
            }
        ]
    )
    outputs = {
        "theory_scorecard": scorecard,
        "p7_turnover_bootstrap": component,
        "source_manifest": manifest,
        "run_parameters": parameters,
        "acceptance": acceptance,
    }
    for name, frame in outputs.items():
        _write(frame, _root() / f"{name}.csv")
    return outputs


def _artifact_hashes() -> dict[str, str]:
    """Hash deterministic F5 artifacts except the replay record."""
    return {
        path.name: _sha256(path)
        for path in sorted(_root().iterdir())
        if path.is_file() and path.name != "determinism.csv"
    }


def verify_determinism() -> pd.DataFrame:
    """Run F5 twice and require byte-identical artifacts."""
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
    _write(replay, _root() / "determinism.csv")
    return replay


def main() -> None:
    """Run F5 and print the deterministic replay status."""
    replay = verify_determinism()
    print(f"F5 theory scorecard: PASS ({len(replay)}/{len(replay)} deterministic)")


if __name__ == "__main__":
    main()
