"""Bootstrap the seven frozen Part-B application comparisons for stage F6.

Each comparison reads the recorded net-NAV columns and its frozen performance table.  NAVs
are converted to month-end simple returns exactly as the accepted performance helper does.
Circular moving-block indices are drawn once per comparison and applied jointly to both
legs, preserving their cross-leg dependence.  The bootstrap changes no weight, cost,
eligibility rule, or sample.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


BLOCK_LENGTH = 6
BOOTSTRAP_DRAWS = 2000
SEED = 20260813
TOLERANCE = 1e-10
METRICS = (
    "net_return_annualized",
    "volatility_annualized",
    "sharpe_rf0",
)


@dataclass(frozen=True)
class Comparison:
    """Describe one frozen candidate-minus-benchmark application comparison."""

    table: str
    comparison: str
    nav_input_id: str
    output_input_id: str
    candidate_column: str
    benchmark_column: str
    candidate_selector: dict[str, object]
    benchmark_selector: dict[str, object]
    window_start: str | None = None
    window_end: str | None = None


COMPARISONS = (
    Comparison(
        "signal",
        "U1 cluster - global",
        "part_b_signal_navs__u1",
        "part_b_signal_grid__u1",
        "cluster_M1_star",
        "global",
        {"leg": "cluster_M1_star"},
        {"leg": "global"},
    ),
    Comparison(
        "signal",
        "U1 cluster - BICS sector",
        "part_b_signal_navs__u1",
        "part_b_signal_grid__u1",
        "cluster_M1_star",
        "bics_sector",
        {"leg": "cluster_M1_star"},
        {"leg": "bics_sector"},
    ),
    Comparison(
        "signal",
        "U2 cluster - global",
        "part_b_signal_navs__u2",
        "part_b_signal_grid__u2",
        "classic_12m_ex_1m__cluster",
        "classic_12m_ex_1m__global",
        {"signal_variant": "classic_12m_ex_1m", "method": "cluster"},
        {"signal_variant": "classic_12m_ex_1m", "method": "global"},
    ),
    Comparison(
        "signal",
        "U3 cluster - global",
        "part_b_signal_navs__u3",
        "part_b_signal_grid__u3",
        "short_3__cluster",
        "short_3__global",
        {"short_span": 3.0, "method": "cluster"},
        {"short_span": 3.0, "method": "global"},
    ),
    Comparison(
        "risk",
        "U1 Rolling-Ward HRP - flat ERC",
        "part_b_risk_navs__u1",
        "part_b_risk_output__u1",
        "ward_hrp",
        "flat_erc",
        {"method": "ward_hrp"},
        {"method": "flat_erc"},
    ),
    Comparison(
        "risk",
        "U1 Rolling-Ward HRP - single-link HRP",
        "part_b_risk_navs__u1",
        "part_b_risk_output__u1",
        "ward_hrp",
        "single_hrp",
        {"method": "ward_hrp"},
        {"method": "single_hrp"},
    ),
    Comparison(
        "risk",
        "U3 equal-cluster RB - flat ERC",
        "part_b_risk_navs__u3",
        "part_b_risk_output__u3",
        "cluster_rb_alpha_0",
        "flat_erc",
        {"method": "cluster_rb_alpha_0"},
        {"method": "flat_erc"},
        "2009-08-31",
        "2026-06-30",
    ),
)


def _output_root() -> Path:
    """Return the configured cluster-lineage output root."""
    value = os.environ.get("CLUSTER_LINEAGE_OUTPUT_DIR")
    if not value:
        raise RuntimeError("CLUSTER_LINEAGE_OUTPUT_DIR must be set")
    return Path(value).resolve()


def _root() -> Path:
    """Return the isolated F6 output directory."""
    root = _output_root() / "finalisation" / "f6"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _read(path: Path, **kwargs: object) -> pd.DataFrame:
    """Read one frozen CSV with round-trip float parsing."""
    return pd.read_csv(path, float_precision="round_trip", **kwargs)


def _write(frame: pd.DataFrame, path: Path) -> None:
    """Write one deterministic high-precision CSV."""
    frame.to_csv(path, index=False, float_format="%.17g", lineterminator="\n")


def _sha256(path: Path) -> str:
    """Return a file's SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _stable_rng(*keys: object) -> np.random.Generator:
    """Return a call-order-independent RNG derived from the frozen seed."""
    digest = hashlib.sha256("\x1f".join(map(str, keys)).encode("utf-8")).digest()
    child = int.from_bytes(digest[:8], "little")
    return np.random.default_rng(np.random.SeedSequence([SEED, child]))


def _mbb_indices(n: int, rng: np.random.Generator) -> np.ndarray:
    """Draw circular moving-block indices for all bootstrap replications."""
    if n <= 1:
        raise ValueError("moving-block bootstrap requires at least two observations")
    blocks = int(np.ceil(n / BLOCK_LENGTH))
    starts = rng.integers(0, n, size=(BOOTSTRAP_DRAWS, blocks))
    offsets = np.arange(BLOCK_LENGTH)
    return ((starts[..., None] + offsets) % n).reshape(BOOTSTRAP_DRAWS, -1)[:, :n]


def _nav_metrics(nav: pd.Series) -> tuple[np.ndarray, pd.Series]:
    """Return exact frozen performance metrics and its monthly return series."""
    nav = nav.dropna()
    monthly = nav.resample("ME").last().pct_change().dropna()
    years = (nav.index[-1] - nav.index[0]).days / 365.25
    annual_return = float((nav.iloc[-1] / nav.iloc[0]) ** (1.0 / years) - 1.0)
    volatility = float(monthly.std() * np.sqrt(12.0))
    sharpe = float(monthly.mean() / monthly.std() * np.sqrt(12.0))
    return np.array([annual_return, volatility, sharpe]), monthly


def _bootstrap_metrics(returns: np.ndarray) -> np.ndarray:
    """Return annual return, volatility, and RF-zero Sharpe for bootstrapped rows."""
    n = returns.shape[1]
    annual_return = np.expm1(12.0 * np.log1p(returns).sum(axis=1) / n)
    volatility = returns.std(axis=1, ddof=1) * np.sqrt(12.0)
    sharpe = returns.mean(axis=1) / returns.std(axis=1, ddof=1) * np.sqrt(12.0)
    return np.column_stack([annual_return, volatility, sharpe])


def _select_performance(frame: pd.DataFrame, selector: dict[str, object]) -> pd.Series:
    """Select exactly one frozen performance row, normalising label types only."""
    selected = frame.copy()
    for column, value in selector.items():
        selected = selected.loc[selected[column].astype(str).eq(str(value))]
    if len(selected) != 1:
        raise AssertionError(f"performance selector did not resolve once: {selector}")
    return selected.iloc[0]


def _f0_sources() -> pd.DataFrame:
    """Return the F0 provenance rows used by all F6 comparisons."""
    inventory = _read(_output_root() / "finalisation" / "f0" / "cache_inventory.csv")
    ids = {comparison.nav_input_id for comparison in COMPARISONS}
    ids.update(comparison.output_input_id for comparison in COMPARISONS)
    selected = inventory.loc[inventory["input_id"].isin(ids)].copy()
    if len(selected) != len(ids) or not selected["status"].eq("PASS").all():
        raise AssertionError("F6 inputs are not uniquely resolved and green in F0")
    return selected.loc[
        :, ["input_id", "path", "manifest_sha256", "status"]
    ].sort_values("input_id").reset_index(drop=True)


def _comparison_rows(
    comparison: Comparison,
    source_by_id: pd.DataFrame,
) -> tuple[list[dict[str, object]], float]:
    """Compute point-regression checks and joint MBB intervals for one comparison."""
    nav_path = Path(source_by_id.loc[comparison.nav_input_id, "path"])
    output_path = Path(source_by_id.loc[comparison.output_input_id, "path"])
    performance_path = output_path / "performance.csv"
    navs = _read(nav_path, index_col=0, parse_dates=True)
    pair = navs[[comparison.candidate_column, comparison.benchmark_column]].dropna()
    if comparison.window_start is not None:
        pair = pair.loc[pair.index >= pd.Timestamp(comparison.window_start)]
    if comparison.window_end is not None:
        pair = pair.loc[pair.index <= pd.Timestamp(comparison.window_end)]
    candidate_metrics, candidate_monthly = _nav_metrics(pair.iloc[:, 0])
    benchmark_metrics, benchmark_monthly = _nav_metrics(pair.iloc[:, 1])
    monthly = pd.concat([candidate_monthly, benchmark_monthly], axis=1).dropna()
    performance = _read(performance_path)
    candidate_frozen = _select_performance(performance, comparison.candidate_selector)
    benchmark_frozen = _select_performance(performance, comparison.benchmark_selector)
    frozen_delta = np.array(
        [
            float(candidate_frozen[metric]) - float(benchmark_frozen[metric])
            for metric in METRICS
        ]
    )
    recomputed_delta = candidate_metrics - benchmark_metrics
    regression_error = float(np.max(np.abs(recomputed_delta - frozen_delta)))

    indices = _mbb_indices(
        len(monthly), _stable_rng(comparison.table, comparison.comparison)
    )
    values = monthly.to_numpy(dtype=float)
    candidate_draws = _bootstrap_metrics(values[indices, 0])
    benchmark_draws = _bootstrap_metrics(values[indices, 1])
    delta_draws = candidate_draws - benchmark_draws
    lower = np.percentile(delta_draws, 2.5, axis=0)
    upper = np.percentile(delta_draws, 97.5, axis=0)
    rows = []
    for index, metric in enumerate(METRICS):
        rows.append(
            {
                "comparison": comparison.comparison,
                "metric": metric,
                "point_estimate": frozen_delta[index],
                "ci_low": float(lower[index]),
                "ci_high": float(upper[index]),
                "excludes_zero": bool(lower[index] > 0.0 or upper[index] < 0.0),
                "series_frequency": "ME",
                "sample_start": pair.index.min(),
                "sample_end": pair.index.max(),
                "monthly_observations": len(monthly),
                "block_length": BLOCK_LENGTH,
                "bootstrap_draws": BOOTSTRAP_DRAWS,
                "seed": SEED,
                "candidate_series": f"{nav_path}::{comparison.candidate_column}",
                "benchmark_series": f"{nav_path}::{comparison.benchmark_column}",
                "frozen_performance_path": str(performance_path),
                "point_recomputation_error": abs(
                    recomputed_delta[index] - frozen_delta[index]
                ),
            }
        )
    return rows, regression_error


def _artifacts() -> dict[str, pd.DataFrame]:
    """Build the two short F6 CI tables and their provenance manifest."""
    sources = _f0_sources()
    indexed = sources.set_index("input_id")
    rows = {"signal": [], "risk": []}
    errors = []
    for comparison in COMPARISONS:
        result, error = _comparison_rows(comparison, indexed)
        rows[comparison.table].extend(result)
        errors.append(
            {
                "comparison": comparison.comparison,
                "maximum_point_recomputation_error": error,
            }
        )
    return {
        "signal_cis.csv": pd.DataFrame(rows["signal"]),
        "risk_cis.csv": pd.DataFrame(rows["risk"]),
        "point_regression.csv": pd.DataFrame(errors),
        "source_manifest.csv": sources,
    }


def _write_artifacts(artifacts: dict[str, pd.DataFrame]) -> None:
    """Write all F6 data artifacts."""
    for name, frame in artifacts.items():
        _write(frame, _root() / name)


def _artifact_hashes(names: list[str]) -> dict[str, str]:
    """Hash named F6 artifacts."""
    return {name: _sha256(_root() / name) for name in names}


def run() -> dict[str, pd.DataFrame]:
    """Execute F6, assert its narrow scope, and prove deterministic replay."""
    artifacts = _artifacts()
    signal = artifacts["signal_cis.csv"]
    risk = artifacts["risk_cis.csv"]
    combined = pd.concat([signal, risk], ignore_index=True)
    maximum_error = float(
        artifacts["point_regression.csv"]["maximum_point_recomputation_error"].max()
    )
    acceptance = pd.DataFrame(
        [
            {
                "check": "F0 sources resolved once",
                "measured": len(artifacts["source_manifest.csv"]),
                "tolerance": 10,
                "status": (
                    "PASS" if len(artifacts["source_manifest.csv"]) == 10 else "FAIL"
                ),
            },
            {
                "check": "signal CI rows",
                "measured": len(signal),
                "tolerance": 12,
                "status": "PASS" if len(signal) == 12 else "FAIL",
            },
            {
                "check": "risk CI rows",
                "measured": len(risk),
                "tolerance": 9,
                "status": "PASS" if len(risk) == 9 else "FAIL",
            },
            {
                "check": "total CI rows",
                "measured": len(combined),
                "tolerance": 21,
                "status": "PASS" if len(combined) == 21 else "FAIL",
            },
            {
                "check": "maximum frozen-point regression error",
                "measured": maximum_error,
                "tolerance": TOLERANCE,
                "status": "PASS" if maximum_error <= TOLERANCE else "FAIL",
            },
            {
                "check": "block length",
                "measured": int(combined["block_length"].min()),
                "tolerance": BLOCK_LENGTH,
                "status": "PASS" if combined["block_length"].eq(BLOCK_LENGTH).all() else "FAIL",
            },
            {
                "check": "bootstrap draws",
                "measured": int(combined["bootstrap_draws"].min()),
                "tolerance": BOOTSTRAP_DRAWS,
                "status": (
                    "PASS"
                    if combined["bootstrap_draws"].eq(BOOTSTRAP_DRAWS).all()
                    else "FAIL"
                ),
            },
            {
                "check": "seed",
                "measured": int(combined["seed"].min()),
                "tolerance": SEED,
                "status": "PASS" if combined["seed"].eq(SEED).all() else "FAIL",
            },
            {
                "check": "NaNs across CI tables",
                "measured": int(combined.isna().sum().sum()),
                "tolerance": 0,
                "status": "PASS" if not combined.isna().to_numpy().any() else "FAIL",
            },
        ]
    )
    if not acceptance["status"].eq("PASS").all():
        raise AssertionError(acceptance.loc[~acceptance["status"].eq("PASS")])
    _write_artifacts(artifacts)
    _write(acceptance, _root() / "acceptance.csv")
    names = sorted([*artifacts, "acceptance.csv"])
    first = _artifact_hashes(names)
    replay = _artifacts()
    _write_artifacts(replay)
    _write(acceptance, _root() / "acceptance.csv")
    second = _artifact_hashes(names)
    determinism = pd.DataFrame(
        [
            {
                "artifact": name,
                "first_sha256": first[name],
                "second_sha256": second[name],
                "byte_identical": first[name] == second[name],
            }
            for name in names
        ]
    )
    if not determinism["byte_identical"].all():
        raise AssertionError("F6 deterministic replay failed")
    _write(determinism, _root() / "determinism.csv")
    print(f"f6_root={_root()} rows={len(combined)} max_error={maximum_error:.3g}")
    return {**artifacts, "acceptance.csv": acceptance, "determinism.csv": determinism}


if __name__ == "__main__":
    run()
