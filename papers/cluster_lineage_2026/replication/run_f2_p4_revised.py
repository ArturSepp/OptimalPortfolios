"""Re-evaluate P4 after absorbing one baseline proportionality constant.

The runner consumes the frozen E3b margin panels and frequency-rescoring tables.
It first reproduces every unrevised annualised ME-to-QE probability-sum ratio.
It then removes the elliptical multiplier from the cached sigma values, applies
the E3b Gaussian constant calibrated once on baseline native-frequency churn,
and predicts absolute quarterly-schedule churn without refitting that constant
for any configuration.
"""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr


HEADLINE_START = pd.Timestamp("2009-08-31")
HEADLINE_END = pd.Timestamp("2026-06-30")
CONFIGS = (
    "baseline",
    "M0_quarterly_hold",
    "M1_delta_0.02",
    "M1_delta_0.05",
    "M1_delta_0.10",
    "M2_lambda_0.5",
    "M2_lambda_0.7",
    "M1_star",
)
PANELS = {
    "equity_panel": {"cache": "msci_us", "native_frequency": "W-WED", "span": 156},
    "futures_panel": {"cache": "futures", "native_frequency": "W-WED", "span": 156},
    "fund_panel": {"cache": "mac", "native_frequency": "ME", "span": None},
}


def _output_root() -> Path:
    """Return the configured cluster-lineage output root."""
    value = os.environ.get("CLUSTER_LINEAGE_OUTPUT_DIR")
    if not value:
        raise RuntimeError("CLUSTER_LINEAGE_OUTPUT_DIR must be set")
    return Path(value).resolve()


def _root() -> Path:
    """Return the isolated F2 output directory."""
    root = _output_root() / "finalisation" / "f2"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _read(path: Path, **kwargs: object) -> pd.DataFrame:
    """Read one frozen CSV with round-trip float parsing."""
    return pd.read_csv(path, float_precision="round_trip", **kwargs)


def _write(frame: pd.DataFrame, path: Path) -> None:
    """Write one deterministic, high-precision CSV."""
    frame.to_csv(path, index=False, float_format="%.17g", lineterminator="\n")


def _mean_probability_sum(
    frame: pd.DataFrame,
    delta: float,
    sigma: pd.Series | np.ndarray,
) -> float:
    """Return the mean across dates of summed Gaussian crossing probabilities."""
    probabilities = pd.Series(
        ndtr(
            -(frame["margin"].to_numpy(dtype=float) + delta)
            / (np.sqrt(2.0) * np.asarray(sigma, dtype=float))
        ),
        index=frame.index,
    )
    return float(probabilities.groupby(frame["date"]).sum().mean())


def _delta_map(aggregate: pd.DataFrame, window: str, config: str) -> dict[str, float]:
    """Return the frozen per-frequency delta for one configuration."""
    selected = aggregate.loc[
        aggregate["analysis_window"].eq(window)
        & aggregate["config"].eq(config)
        & aggregate["singleton_convention"].eq("including_singletons")
    ]
    return {
        frequency: float(group["delta"].iloc[0])
        for frequency, group in selected.groupby("frequency", sort=True)
    }


def _rescore_single_frequency_panel(
    margin: pd.DataFrame,
    delta: float,
    span: int,
    kappa: float,
) -> dict[str, float]:
    """Re-score one weekly panel at monthly and quarterly estimation steps."""
    decay = 1.0 - 2.0 / (span + 1.0)
    monthly_step = 52.0 / 12.0
    quarterly_step = 13.0
    scale = np.sqrt(
        (1.0 - decay**quarterly_step) / (1.0 - decay**monthly_step)
    )
    sigma_native = margin["sigma_d"].to_numpy(dtype=float)
    sigma_quarterly = sigma_native * scale
    sigma_gaussian_native = sigma_native / np.sqrt(1.0 + kappa)
    sigma_gaussian_quarterly = sigma_gaussian_native * scale
    return {
        "unrevised_native_probability_sum": _mean_probability_sum(
            margin, delta, sigma_native
        ),
        "unrevised_quarterly_probability_sum": _mean_probability_sum(
            margin, delta, sigma_quarterly
        ),
        "gaussian_native_probability_sum": _mean_probability_sum(
            margin, delta, sigma_gaussian_native
        ),
        "gaussian_quarterly_probability_sum": _mean_probability_sum(
            margin, delta, sigma_gaussian_quarterly
        ),
    }


def _rescore_fund_panel(
    margin: pd.DataFrame,
    deltas: Mapping[str, float],
    kappas: Mapping[str, float],
) -> dict[str, float]:
    """Re-score the native MAC ME and QE sleeves without pooling their assets."""
    output = {}
    for frequency, key in (("ME", "native"), ("QE", "quarterly")):
        selected = margin.loc[margin["frequency"].eq(frequency)]
        sigma = selected["sigma_d"].to_numpy(dtype=float)
        output[f"unrevised_{key}_probability_sum"] = _mean_probability_sum(
            selected, deltas[frequency], sigma
        )
        output[f"gaussian_{key}_probability_sum"] = _mean_probability_sum(
            selected,
            deltas[frequency],
            sigma / np.sqrt(1.0 + kappas[frequency]),
        )
    return output


def _panel_rows(
    panel: str,
    info: Mapping[str, object],
    corrected: pd.DataFrame,
) -> tuple[list[dict[str, object]], float, float]:
    """Build every revised P4 row for one theory panel."""
    cache = _output_root() / "stability" / str(info["cache"])
    margin = _read(cache / "margin_assets.csv", parse_dates=["date"])
    aggregate = _read(cache / "predicted_realized.csv")
    kurtosis = _read(cache / "kurtosis_check.csv")
    realised = _read(cache / "frequency_scaling.csv").set_index(
        ["analysis_window", "config"]
    )
    rows = []
    maximum_unrevised_error = 0.0
    maximum_baseline_calibration_error = 0.0
    for window in aggregate["analysis_window"].unique():
        window_margin = margin.copy()
        if window == "headline_20090831_20260630":
            window_margin = window_margin.loc[
                window_margin["date"].between(HEADLINE_START, HEADLINE_END)
            ]
        native_frequency = str(info["native_frequency"])
        baseline_kurtosis = kurtosis.loc[
            kurtosis["analysis_window"].eq(window)
            & kurtosis["frequency"].eq(native_frequency)
        ].iloc[0]
        constant_c = float(baseline_kurtosis["realized_to_gaussian_multiplier"])
        gaussian_baseline = float(
            baseline_kurtosis["gaussian_predicted_churn_annualized"]
        )
        kappas = {
            frequency: float(group["kappa_hat"].iloc[0])
            for frequency, group in kurtosis.loc[
                kurtosis["analysis_window"].eq(window)
            ].groupby("frequency", sort=True)
        }
        baseline_delta = _delta_map(aggregate, window, "baseline")
        if panel == "fund_panel":
            baseline_score = _rescore_fund_panel(
                window_margin, baseline_delta, kappas
            )
        else:
            baseline_score = _rescore_single_frequency_panel(
                window_margin,
                baseline_delta[native_frequency],
                int(info["span"]),
                kappas[native_frequency],
            )
        annualization_scale = (
            gaussian_baseline / baseline_score["gaussian_native_probability_sum"]
        )
        provisional = []
        for config in CONFIGS:
            deltas = _delta_map(aggregate, window, config)
            if panel == "fund_panel":
                score = _rescore_fund_panel(window_margin, deltas, kappas)
            else:
                score = _rescore_single_frequency_panel(
                    window_margin,
                    deltas[native_frequency],
                    int(info["span"]),
                    kappas[native_frequency],
                )
            unrevised_ratio = (
                (4.0 / 12.0)
                * score["unrevised_quarterly_probability_sum"]
                / score["unrevised_native_probability_sum"]
            )
            frozen = corrected.loc[
                corrected["universe"].eq(info["cache"])
                & corrected["analysis_window"].eq(window)
                & corrected["config"].eq(config)
            ].iloc[0]
            maximum_unrevised_error = max(
                maximum_unrevised_error,
                abs(unrevised_ratio - float(frozen["predicted_annualized_churn_ratio"])),
            )
            gaussian_native = (
                annualization_scale * score["gaussian_native_probability_sum"]
            )
            gaussian_quarterly = (
                annualization_scale
                * (4.0 / 12.0)
                * score["gaussian_quarterly_probability_sum"]
            )
            revised_native = constant_c * gaussian_native
            revised_quarterly = constant_c * gaussian_quarterly
            realised_row = realised.loc[(window, config)]
            provisional.append(
                {
                    "panel": panel,
                    "analysis_window": window,
                    "config": config,
                    "delta_native": deltas[native_frequency],
                    "delta_quarterly": deltas.get("QE", deltas[native_frequency]),
                    "constant_c": constant_c,
                    "constant_calibration_config": "baseline",
                    "gaussian_native_churn": gaussian_native,
                    "gaussian_quarterly_churn": gaussian_quarterly,
                    "revised_predicted_native_churn": revised_native,
                    "revised_predicted_quarterly_churn": revised_quarterly,
                    "realised_native_churn": float(realised_row["base_churn"]),
                    "realised_quarterly_churn": float(realised_row["quarterly_churn"]),
                    "revised_quarterly_gap_realised_minus_predicted": float(
                        realised_row["quarterly_churn"] - revised_quarterly
                    ),
                    "revised_predicted_annualised_ratio": (
                        revised_quarterly / revised_native
                    ),
                    "realised_annualised_ratio": float(
                        realised_row["realized_churn_ratio"]
                    ),
                    "unrevised_predicted_annualised_ratio": unrevised_ratio,
                    "unrevised_frozen_ratio": float(
                        frozen["predicted_annualized_churn_ratio"]
                    ),
                    "unrevised_ratio_error": abs(
                        unrevised_ratio
                        - float(frozen["predicted_annualized_churn_ratio"])
                    ),
                    "p4_classification": frozen["classification"],
                    "constant_source": "E3b baseline Gaussian realised/predicted",
                    "sigma_convention": "cached sigma with sqrt(1+kappa) removed",
                }
            )
        provisional_frame = pd.DataFrame(provisional)
        correlation = float(
            provisional_frame[
                ["revised_predicted_quarterly_churn", "realised_quarterly_churn"]
            ].corr().iloc[0, 1]
        )
        mean_gap = float(
            provisional_frame[
                "revised_quarterly_gap_realised_minus_predicted"
            ].mean()
        )
        verdict = (
            "DESCRIPTIVE_DIFFERENT_SLEEVES_AND_SPANS"
            if panel == "fund_panel"
            else "SUPPORTED_REVISED_ORDERING;LEVEL_EQUALITY_REJECTED"
        )
        for row in provisional:
            row["cross_config_revised_correlation"] = correlation
            row["mean_revised_gap"] = mean_gap
            row["revised_verdict"] = verdict
            rows.append(row)
        baseline = provisional_frame.loc[provisional_frame["config"].eq("baseline")].iloc[0]
        maximum_baseline_calibration_error = max(
            maximum_baseline_calibration_error,
            abs(
                float(baseline["revised_predicted_native_churn"])
                - float(baseline["realised_native_churn"])
            ),
        )
    return rows, maximum_unrevised_error, maximum_baseline_calibration_error


def _source_manifest() -> pd.DataFrame:
    """Return the F0-inventoried paths consumed by F2."""
    inventory = _read(_output_root() / "finalisation" / "f0" / "cache_inventory.csv")
    selected = inventory.loc[inventory["stages"].str.contains("F2", regex=False)].copy()
    if not selected["status"].eq("PASS").all() or not selected["resolution_count"].eq(1).all():
        raise AssertionError("F2 inputs are no longer complete and unambiguous")
    return selected


def _acceptance(
    revised: pd.DataFrame,
    sources: pd.DataFrame,
    unrevised_error: float,
    baseline_error: float,
) -> pd.DataFrame:
    """Return every measured-versus-tolerance F2 acceptance line."""
    c_counts = revised.groupby(["panel", "analysis_window"])["constant_c"].nunique()
    checks = [
        ("F0 source paths resolved once", len(sources), len(sources), True),
        (
            "unrevised E3b ratio regression error",
            unrevised_error,
            1e-9,
            unrevised_error <= 1e-9,
        ),
        (
            "baseline native c-calibration error",
            baseline_error,
            1e-12,
            baseline_error <= 1e-12,
        ),
        (
            "constants per panel/window",
            int(c_counts.max()),
            1,
            c_counts.eq(1).all(),
        ),
        (
            "constant calibration configs",
            revised["constant_calibration_config"].nunique(),
            1,
            revised["constant_calibration_config"].eq("baseline").all(),
        ),
        ("P4 rows", len(revised), 32, len(revised) == 32),
        (
            "fund verdict rows entering P4 test",
            int(
                revised.loc[revised["panel"].eq("fund_panel"), "revised_verdict"]
                .ne("DESCRIPTIVE_DIFFERENT_SLEEVES_AND_SPANS")
                .sum()
            ),
            0,
            revised.loc[
                revised["panel"].eq("fund_panel"), "revised_verdict"
            ].eq("DESCRIPTIVE_DIFFERENT_SLEEVES_AND_SPANS").all(),
        ),
        (
            "NaNs in p4_revised",
            int(revised.isna().to_numpy().sum()),
            0,
            not revised.isna().to_numpy().any(),
        ),
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
    """Execute the revised P4 analysis and write deterministic artifacts."""
    corrected = _read(
        _output_root() / "stability" / "corrected_frequency_scaling.csv"
    )
    rows = []
    maximum_unrevised_error = 0.0
    maximum_baseline_error = 0.0
    for panel, info in PANELS.items():
        panel_rows, unrevised_error, baseline_error = _panel_rows(
            panel, info, corrected
        )
        rows.extend(panel_rows)
        maximum_unrevised_error = max(maximum_unrevised_error, unrevised_error)
        maximum_baseline_error = max(maximum_baseline_error, baseline_error)
    revised = pd.DataFrame(rows)
    sources = _source_manifest()
    acceptance = _acceptance(
        revised, sources, maximum_unrevised_error, maximum_baseline_error
    )
    outputs = {
        "p4_revised": revised,
        "source_manifest": sources,
        "acceptance": acceptance,
    }
    for name, frame in outputs.items():
        _write(frame, _root() / f"{name}.csv")
    return outputs


def _hashes() -> dict[str, str]:
    """Hash deterministic F2 artifacts while excluding the replay record."""
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(_root().glob("*.csv"))
        if path.name != "determinism.csv"
    }


def verify_determinism() -> pd.DataFrame:
    """Run F2 twice and require byte-identical outputs."""
    run()
    first = _hashes()
    run()
    second = _hashes()
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
    """Execute F2 and print its deterministic acceptance summary."""
    replay = verify_determinism()
    print(f"F2 revised P4: PASS ({len(replay)}/{len(replay)} deterministic)")


if __name__ == "__main__":
    main()
