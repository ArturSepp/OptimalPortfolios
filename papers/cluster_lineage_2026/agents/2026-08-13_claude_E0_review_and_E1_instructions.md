# E0 owner-side review and instructions for the next steps

**Date:** 2026-08-13
**Author:** Claude (owner-side review and design agent), for Artur Sepp
**Reviewed:** `2026-08-13_sol_E0_report.md` plus the six E0 deliverables read in full
(`local_path.py`, `configs.py`, `metrics.py`, `metrics_test.py`, `validate_e0.py`,
`validate_e0_independent.py`) and the four refactored runners.

---

## 1. Review verdict on E0

**Recommend APPROVE, with four amendments executed as a small E0b patch.** The work is
solid: all thirteen acceptance checks pass with honest measured-vs-tolerance reporting, the
red/green proof of the regression test was actually performed, the independent scikit-learn
ARI check agrees to 1e-16, the audit found and removed a fourth rosaa import beyond the three
the roadmap named, and the lineage path now calls canonical
`factorlasso.analyze_cluster_lineage` instead of the deprecation shim, at unchanged numbers
(3.2114693774 against the frozen 3.2115). The registry design is right: calibrated slots
raise rather than default.

None of the four amendments below changes a frozen regression number. Each keeps the current
behaviour as the default, so the E0 regression suite stays green.

**A1 — kurtosis multiplier missing from the margin noise scale (theory gap).**
`assignment_margins` implements `sigma_d = sqrt(2(1-lambda^k)) * (1-rho^2) / sqrt(span)`,
which is the Gaussian case only. The theory draft (Proposition 2(iii) in
`paper/cluster_distribution_theory.tex`) and Prediction P3 require the elliptical multiplier
`sqrt(1+kappa)`. Add an optional `kappa: float = 0.0` argument that multiplies `sigma` by
`sqrt(1+kappa)`. Default 0.0 preserves every existing output.

**A2 — `tracks_per_asset` does not match the programme's convention (definition defect).**
`lineage_metrics` computes `len(report.tracks) / n_assets` (about 0.4 on the S&P 500 panel).
The sweep convention behind "tracks per asset 18.5 → 14.5" is the mean number of DISTINCT
derived ids an asset passes through over the panel: per asset column of
`to_membership_panel()`, count `nunique(dropna=True)`, then average across assets with any
membership. Redefine `tracks_per_asset` to that quantity. If the ratio is worth keeping,
rename it `track_to_asset_ratio` so the two cannot be confused.

**A3 — `signal_rank_metrics` is weekly-only (blocks U3).** The 52/4-week window is
hardcoded. Parameterise as `lookback_periods`, `skip_periods`, and a period `freq` consumed
from `UniverseSpec.momentum_lookback/momentum_skip`, with defaults reproducing the current
weekly behaviour byte-for-byte. U3 then runs 12m-skip-1m (ME) and 4Q-skip-1Q (QE) through
the same function.

**A4 — unclassified flows are silent in `membership_flow_decomposition` (minor).** An asset
that stops being clustered for data reasons while remaining in the index produces no event.
Add an `unclassified` count column so the reconciliation `total transitions = sum of
categories` can be asserted in E3 rather than assumed.

One analysis convention to record now rather than discover in E3: under the average-distance
margin proxy, singletons get `rho_hat = 1`, `sigma = 0`, and predicted flip probability 0,
while real singletons do get absorbed. E3 therefore reports predicted-versus-realised churn
both including and excluding singleton assets. This is an analysis convention, not a code
change.

## 2. Rulings needed from the owner (gate E0)

1. **Approve E0 as frozen** subject to the E0b patch (A1–A4)? My recommendation: yes.
2. **The `.gitignore` question Sol raised** (line 185 ignores the whole
   `papers/cluster_lineage_2026` tree, so none of the E0 deliverables can be committed).
   This collides with the number-traceability gate: exhibit scripts must be committed and
   taggable. My recommendation: narrow the rule to the data payloads — ignore
   `papers/cluster_lineage_2026/data/` and `papers/cluster_lineage_2026/msci_us/` — and
   track `replication/`, `agents/`, and `paper/`. The licensing ruling on whether any data
   files may ever be committed stays open and is not forced by this change.
3. **Confirm the M1_star specification in section 3** so the slot can be filled from
   measured inputs, and confirm adding the grid point `M1_delta_0.02`.

## 3. Filling the calibrated slots

**M1_star (specify now).** The level-form noise-floor calibration at one standard error:

```
delta_star = z * sqrt(1 + kappa_hat) * (1 - rho_bar^2) / sqrt(span),   z = 1
```

computed once per universe and estimation frequency, from measured inputs:

- `rho_bar`: the median within-cluster pairwise correlation across all baseline estimation
  dates, read from the baseline caches (E2 output). Per frequency for U3 (ME and QE spans
  differ, so U3 carries a per-frequency delta via the smoother configuration).
- `kappa_hat = max(0, median_i(g_i) / 3)`, where `g_i` is asset i's sample excess kurtosis
  (Fisher) at the estimation frequency over the estimation sample. The cross-sectional
  median keeps single wild assets from setting the floor. Record `median_i(g_i)` in the E1
  data report so the input is visible before it is used.
- `span`: the universe's lasso span at that frequency (156 / 156 / 36 and 12).

Sequencing: baseline E2 runs produce `rho_bar`, then the E3 preparation step computes
`delta_star` per universe, writes the derivation into the stage report, and the owner
confirms the numbers before the `M1_star` estimation runs (E2b). The slot stays raising
until then, exactly as built.

**Grid amendment.** Add fixed configuration `M1_delta_0.02` to the grid. The innovation-form
calibration (`delta_star` scaled by `sqrt(2(1-lambda^k))`, about 0.02 at weekly span 156 and
monthly estimation) falls outside the current grid {0.05, 0.10}, and Prediction P4's
frontier needs coverage of that region. One extra config per universe is cheap for U2/U3
and acceptable for U1.

**M2_star stays empty.** The lambda calibration depends on the span-composition derivation
of the theory track (P1a), which is not yet validated. Do not fill it; the roadmap already
prevents silent selection.

## 4. Instructions for Sol — E0b patch, then Stage E1

**E0b (before or alongside E1 start; half a day).** Implement A1–A4 with defaults that
preserve current outputs. Re-run `metrics_test.py` and `validate_e0.py` and report both
green in a short `2026-08-XX_sol_E0b_report.md`. Add one regression assertion for A2's new
definition on the frozen S&P 500 panel (record the measured value as the new frozen
target; on that panel the expected magnitude is O(10), not O(0.4)). If the owner approves
ruling 2, apply the `.gitignore` narrowing in the same patch and list the files that become
trackable; commit nothing beyond what the owner authorises.

**E1 (data layer) proceeds per the roadmap, with these additions.**

1. The data-quality report gains one line per universe and frequency: `median_i(g_i)`
   (excess kurtosis) and the implied `kappa_hat`, feeding section 3.
2. FF6 fallback protocol as written: if network access fails, write
   `2026-08-XX_sol_escalation_ff6.md` and stop that sub-task only; U2/U3 preparation
   continues in parallel.
3. The 19 metadata-uncovered MAC columns: classify each from the metadata sources as
   universe member, benchmark series, or excluded; state the rule and the per-column
   outcome in the data report. Nothing is silently dropped.
4. Confirm in the report which market series U1 uses where: FF6 supplies the factor panel
   for estimation (Mkt-RF is the market factor), the MSCI US index file is reference only,
   and the EW basket is constructed at backtest time (E5), not in E1.
5. Register `M1_delta_0.02` in `configs.py` (ruling 3).

**Sequencing unchanged:** U2 end-to-end first, then U3, then the heavy U1. E2 may start per
universe as soon as that universe's E1 acceptance lines pass, without waiting for the other
universes' data reports.

## 5. What I could not verify from here

The frozen caches and workbooks live outside the connected folder tree
(`analytics/outputs/`), so I verified Sol's arithmetic against the recorded sweep numbers
and reviewed all six modules line by line, but did not re-execute the runners. The
independent scikit-learn check and the red/green proof in Sol's report are the compensating
evidence, and both are the kind of check that is hard to fake by accident. The E0b report
should paste the two-line pytest output verbatim as usual.
