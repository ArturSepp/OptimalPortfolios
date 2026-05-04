"""
Reproducibility test for the MATF-CMA bootstrap exhibit.

Confirms that with seed=42, the published numbers reproduce within tolerance.
Tolerances are wide pending the 17-asset universe re-run (May 2026 update);
narrow once the new headline numbers stabilise.

Skipped automatically if the production data files are not present.

Run from the matf_cma_jpm_2026 directory:
    pytest tests/

or:
    python -m pytest tests/test_reproducibility.py -v
"""
from pathlib import Path

import numpy as np
import pytest

from run_bootstrap import load_real_data, run_real_bootstrap


ROOT = Path(__file__).resolve().parent.parent
XLSX = ROOT / "data" / "global_saa_universe_data_cmas_usd_2026q1_jpm.xlsx"
CSV  = ROOT / "data" / "futures_risk_factors.csv"


@pytest.fixture(scope="module")
def result():
    if not (XLSX.exists() and CSV.exists()):
        pytest.skip(f"Production data files not present: {XLSX.name}, {CSV.name}")
    data = load_real_data(str(XLSX), str(CSV), verbose=False)
    return run_real_bootstrap(
        data, n_boot=500, mean_block=12.0, min_block=3,
        n_vol_points=24, seed=42, verbose=False,
    )


def test_reduction_ratio_at_balanced_vol(result):
    """Reduction ratio at the Balanced-mandate vol target (~5.8%) should be
    around 2.1× on the 17-asset production data with backfill (seed=42)."""
    k = int(np.argmin(np.abs(result.vol_grid - 0.058)))
    raw = result.raw_returns[:, k]
    fac = result.factor_returns[:, k]
    raw = raw[np.isfinite(raw)]
    fac = fac[np.isfinite(fac)]
    raw_w = np.percentile(raw, 95) - np.percentile(raw, 5)
    fac_w = np.percentile(fac, 95) - np.percentile(fac, 5)
    ratio = raw_w / fac_w
    assert 1.8 <= ratio <= 2.5, f"Reduction ratio at 5.8% vol = {ratio:.2f}× (expected ≈ 2.1×)"


def test_gk_consistency_median(result):
    """GK median ‖Δ‖₂ ≈ 2.67% on the 17-asset production data with backfill (seed=42)."""
    median_norm = np.nanmedian(result.delta_norm_gk) * 100
    assert 2.2 <= median_norm <= 3.2, f"GK median ‖Δ‖₂ = {median_norm:.2f}% (expected ≈ 2.67%)"


def test_matf_consistency_zero(result):
    """MATF satisfies Δ ≡ 0 by construction for every draw."""
    assert np.all(result.delta_norm_matf == 0.0), \
        "MATF Δ should be identically zero across all bootstrap draws"


def test_vol_grid_covers_benchmarks(result):
    """Vol grid should span the eight benchmark mandates."""
    assert result.vol_grid[0] <= 0.05, f"Vol grid min = {result.vol_grid[0]:.3f}, should be ≤ 5%"
    assert result.vol_grid[-1] >= 0.13, f"Vol grid max = {result.vol_grid[-1]:.3f}, should be ≥ 13%"


def test_no_nan_baseline(result):
    """Baseline frontiers should have no NaN — vol grid is feasible by construction."""
    assert not np.any(np.isnan(result.baseline_raw)), "raw baseline frontier has NaN"
    assert not np.any(np.isnan(result.baseline_factor)), "MATF baseline frontier has NaN"
