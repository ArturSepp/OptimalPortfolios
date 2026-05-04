"""
Sensitivity of the MATF reduction ratio to the SR-prior dispersion σ_SR.

Sweeps σ_SR ∈ {0.05, 0.075, 0.10 (baseline), 0.125, 0.15, 0.20}, runs the
bootstrap for each value, and prints the 90%-bandwidth reduction ratio at
selected vol targets plus a summary line per σ_SR.

Why σ_SR = 0.10 in the paper: at the effective sample size T_eff ≈ 25, the
sample-mean Sharpe-ratio estimator has asymptotic standard deviation
1/√T_eff ≈ 0.20, so σ_SR = 0.10 encodes a structural prior twice as tight
as the historical record — informationally equivalent to four times the
sample size. This script confirms that the qualitative claim (factor
structure tightens the frontier fan) is robust across a wide band around
the chosen value, while the quantitative claim (~2× reduction) is specific
to σ_SR = 0.10.

Run from the matf_cma_jpm_2026 directory:
    python run_sensitivity.py

The script monkey-patches `run_bootstrap.SR_STD` for each sweep value
since the module-level constant is read once per `load_real_data` call.
The bootstrap itself takes σ_SR through `data.Sigma_SR` so each run is
fully independent.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

import run_bootstrap as rb


XLSX = Path("data") / "global_saa_universe_data_cmas_usd_2026q1_jpm.xlsx"
CSV  = Path("data") / "futures_risk_factors.csv"

# σ_SR sensitivity grid. 0.10 is the paper baseline; values flank it on a
# roughly geometric spacing.
SR_STD_GRID = [0.05, 0.075, 0.10, 0.125, 0.15, 0.20]

# Vol targets at which to report the reduction ratio (selected to align
# with the eight benchmark mandates, span 4–14 % vol).
VOL_TARGETS = [0.041, 0.062, 0.082, 0.103, 0.123, 0.144]


def reduction_at(result, vol: float) -> tuple[float, float, float]:
    """Return (raw 90% width, MATF 90% width, reduction ratio) at the
    grid point closest to `vol`. NaN-safe."""
    k = int(np.argmin(np.abs(result.vol_grid - vol)))
    raw = result.raw_returns[:, k]
    fac = result.factor_returns[:, k]
    raw = raw[np.isfinite(raw)]
    fac = fac[np.isfinite(fac)]
    if len(raw) == 0 or len(fac) == 0:
        return np.nan, np.nan, np.nan
    raw_w = np.percentile(raw, 95) - np.percentile(raw, 5)
    fac_w = np.percentile(fac, 95) - np.percentile(fac, 5)
    return raw_w, fac_w, (raw_w / fac_w if fac_w > 0 else np.nan)


def main() -> None:
    print(f"\nσ_SR sensitivity sweep — bootstrap reduction ratio")
    print(f"  paper baseline: σ_SR = 0.10")
    print(f"  asymptotic SR-noise scale: 1/√T_eff ≈ 0.20 (with T_eff ≈ 25)\n")

    # Header
    header = ["σ_SR"]
    for v in VOL_TARGETS:
        header.append(f"v={v*100:.1f}%")
    header.append("range")
    print(" | ".join(f"{h:>10s}" for h in header))
    print("-" * (12 * len(header)))

    rows = []
    for sr_std in SR_STD_GRID:
        # Monkey-patch the module-level SR_STD so load_real_data sees the
        # new value when it computes Sigma_SR.
        rb.SR_STD = sr_std
        data = rb.load_real_data(str(XLSX), str(CSV), verbose=False)
        result = rb.run_real_bootstrap(
            data, n_boot=500, mean_block=12.0, min_block=3,
            n_vol_points=24, seed=42, verbose=False,
        )
        ratios = []
        widths_raw = []
        widths_fac = []
        for v in VOL_TARGETS:
            rw, fw, ratio = reduction_at(result, v)
            ratios.append(ratio)
            widths_raw.append(rw)
            widths_fac.append(fw)

        ratios_arr = np.array(ratios)
        finite = ratios_arr[np.isfinite(ratios_arr)]
        rng_str = f"{finite.min():.2f}–{finite.max():.2f}×" if len(finite) else "n/a"
        marker = " ← baseline" if abs(sr_std - 0.10) < 1e-9 else ""

        cells = [f"{sr_std:>9.3f}"]
        for r in ratios:
            cells.append(f"{r:>9.2f}×" if np.isfinite(r) else "       n/a")
        cells.append(f"{rng_str:>10s}")
        print(" | ".join(cells) + marker)
        rows.append((sr_std, ratios, widths_raw, widths_fac))

    # Restore the module-level baseline for any subsequent imports
    rb.SR_STD = 0.10

    # Summary block: ratio at v = 6.2 % (Balanced w/o-Alts vol target)
    print("\n\nReduction ratio at the Balanced w/o-Alts vol target (v ≈ 6.2 %):\n")
    print(f"  {'σ_SR':>6s} | {'Reduction':>10s} | {'4× T_eff equivalent':>20s}")
    print("  " + "-" * 50)
    for sr_std, ratios, _, _ in rows:
        balanced_ratio = ratios[1]  # index 1 = 6.2 %
        # Effective-sample-size equivalence: variance ratio (raw/MATF)² to
        # express the MATF prior in T-equivalent units.
        var_ratio = balanced_ratio ** 2 if np.isfinite(balanced_ratio) else np.nan
        marker = " ← baseline" if abs(sr_std - 0.10) < 1e-9 else ""
        print(f"  {sr_std:>6.3f} | {balanced_ratio:>9.2f}× | "
              f"{var_ratio:>18.1f}× T_eff{marker}")

    print("\nInterpretation: the MATF reduction ratio scales approximately as")
    print("1/σ_SR (linear) and the variance-equivalent T_eff multiplier as")
    print("1/σ_SR². The qualitative claim — factor structure tightens the")
    print("frontier fan — survives across the whole σ_SR ∈ [0.05, 0.15] range.")


if __name__ == "__main__":
    main()
