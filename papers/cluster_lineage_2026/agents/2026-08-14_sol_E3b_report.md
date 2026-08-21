# Stage E3b execution report - corrected U1 harness and theory diagnostics

**Date:** 2026-08-14  
**Roadmap:** `papers/cluster_lineage_2026/agents/ROADMAP_cluster_lineage_empirics.md`  
**Owner dispatch:** `2026-08-14_owner_E3_gate.md`  
**Status:** COMPLETE

## Execution surface

Runners: `papers/cluster_lineage_2026/replication/estimate.py`,
`run_stability.py`, and `run_e3b.py`  
Independent validators: `validate_e3.py`, `validate_e3b.py`, and
`validate_e4_e5.py`  
Cache root: `C:\Users\artur\OneDrive\analytics\outputs\cluster_lineage_2026\msci_us\<config>\`  
Evidence root: `C:\Users\artur\OneDrive\analytics\outputs\cluster_lineage_2026\stability\`

The U1 smoother-universe diagnostic confirmed a harness defect. The legacy precompute passed
the full listed return panel to `compute_rolling_smoothed_clusters`, while the fitted estimator
used the point-in-time eligible member panel. This report replaces every U1 smoothed-config
result and the frequency-scaling conclusion in the original E3 report. The owner-accepted U2,
U3, predicted-versus-realised, and risk-model-invariance rulings remain unchanged.

## U1 universe-consistency diagnosis and correction

Before fitted-asset restriction, the legacy smoother's asset-set symmetric-difference share
against the baseline fitted partition was 0.550043 on average and 0.614875 at maximum. The
same mismatch applied to all seven smoothed configurations. This answers the owner's diagnostic
question: the legacy smoother clustered a superset, not the point-in-time member panel.

The corrected precompute applies the estimator's eligibility mask separately on every date,
then forms its correlation and smoothing state only on that eligible set. After correction,
all seven configs have mean and maximum asset-set symmetric-difference share exactly 0.0 over
all 238 dates. The comparison is explicitly labelled `BEFORE_FITTER_RESTRICTION` in
`u1_corrected_asset_set_per_date.csv`.

For the requested baseline-versus-M1-0.02 cluster-count series, baseline has mean/median/range
78.8445/83/[32, 122] clusters; corrected M1-0.02 has 81.2479/86/[35, 121]. The complete
238-date series is `u1_cluster_counts_baseline_vs_corrected_M1_002.csv`.

All seven smoothed U1 configurations were refitted; the baseline cache was not touched. Each
config produced 238 snapshots and matched its injected partition on 238/238 dates:

| Config | Precompute seconds | Fit seconds | Cache bytes | injected == fitted |
|---|---:|---:|---:|---:|
| M0 quarterly hold | 24.554 | 788.738 | 652,087,312 | 238/238 PASS |
| M1 0.02 | 686.741 | 699.853 | 651,956,695 | 238/238 PASS |
| M1 0.05 | 15.373 | 673.362 | 651,958,269 | 238/238 PASS |
| M1 0.10 | 12.836 | 650.512 | 651,960,059 | 238/238 PASS |
| M2 0.5 | 13.290 | 634.974 | 651,956,767 | 238/238 PASS |
| M2 0.7 | 13.841 | 648.317 | 651,956,653 | 238/238 PASS |
| M1 star | 13.936 | 637.146 | 651,959,576 | 238/238 PASS |

The first M1-0.02 precompute time includes construction of the reusable 238-date dynamic
correlation cache. All 238 baseline pickle modification dates remain 2026-08-13; zero baseline
pickles were modified during E3b.

## Corrected U1 fidelity verdicts

The symmetric fidelity band remains unchanged: absolute taxonomy-ARI changes must be at most
0.03 and absolute cluster-count change at most 0.15. Full-panel and headline rows remain
separate and are never pooled.

| Window | Config | Raw churn | Median same-date ARI | Sector dARI | Industry-group dARI | Industry dARI | Cluster-count change | Verdict |
|---|---|---:|---:|---:|---:|---:|---:|---|
| full | baseline | 3.545476 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | PASS |
| full | M0 hold | 1.440727 | 0.660116 | -0.001453 | +0.006367 | -0.000383 | -0.018072 | PASS |
| full | M1 0.02 | 1.675931 | 0.654005 | -0.008027 | -0.000112 | +0.012526 | +0.036145 | PASS |
| full | M1 0.05 | 0.856218 | 0.540457 | -0.021175 | -0.025255 | -0.004489 | +0.060241 | PASS |
| full | M1 0.10 | 0.454988 | 0.407710 | -0.042451 | -0.057693 | -0.049179 | +0.084337 | REJECTED |
| full | M2 0.5 | 3.102240 | 0.686992 | +0.002231 | +0.007927 | +0.006407 | +0.024096 | PASS |
| full | M2 0.7 | 2.859285 | 0.644727 | -0.002153 | +0.006428 | +0.006549 | +0.006024 | PASS |
| full | M1 star | 0.537168 | 0.424318 | -0.035687 | -0.045751 | -0.036336 | +0.096386 | REJECTED |
| headline | baseline | 3.212423 | 1.000000 | 0.000000 | 0.000000 | 0.000000 | 0.000000 | PASS |
| headline | M0 hold | 1.328019 | 0.676140 | -0.002720 | +0.001961 | -0.001581 | -0.011628 | PASS |
| headline | M1 0.02 | 1.265764 | 0.668512 | -0.010425 | -0.004917 | +0.009395 | +0.046512 | PASS |
| headline | M1 0.05 | 0.580072 | 0.555330 | -0.024502 | -0.031292 | -0.006015 | +0.069767 | REJECTED |
| headline | M1 0.10 | 0.296986 | 0.427154 | -0.045961 | -0.064351 | -0.048703 | +0.081395 | REJECTED |
| headline | M2 0.5 | 2.790436 | 0.702316 | -0.000688 | +0.003068 | +0.005363 | +0.011628 | PASS |
| headline | M2 0.7 | 2.555804 | 0.653918 | -0.003457 | +0.001719 | +0.006161 | 0.000000 | PASS |
| headline | M1 star | 0.355816 | 0.456046 | -0.038273 | -0.048818 | -0.037626 | +0.093023 | REJECTED |

Thus the corrected grid contains 11 PASS and 5 REJECTED rows. M1-star still fails, but the
valid reason is its taxonomy breach rather than the legacy universe mismatch. Headline
M1-star cuts raw churn by 88.9%, while all three taxonomy changes exceed the 0.03 band.
All smoothed rows pass the 0.05 residual-diagonality guard; the maximum is 0.007018.

Corrected predicted-versus-realised correlations across configurations are 0.863096/0.862986
including/excluding singletons on the full panel and 0.871747/0.871631 on the headline window.

## M2 correlation-state entries and exits

For U1 M2-0.5, `u1_m2_state_flows.csv` records 238 dates. The dynamic state contains a mean
611.04 assets (range 523-641). Entries have mean/median/max 6.038/1/633 and exits
3.840/2/43. The 633-entry maximum is the initial state construction; subsequent membership
entries and exits make the state discontinuous. This explains why the legacy static-universe
M2 anomaly could produce churn above baseline: its state included assets outside the fitted
point-in-time universe. Under the corrected state, M2-0.5 churn is below baseline in both
windows (3.102240 versus 3.545476 full; 2.790436 versus 3.212423 headline).

## Corrected annualised frequency scaling

For each config, the Gaussian prediction is recomputed from the frozen margin panel as

`(4/12) * sum_i Phi(-(margin_i + delta)/(sqrt(2)*sigma(k_QE))) /
sum_i Phi(-(margin_i + delta)/(sqrt(2)*sigma(k_ME)))`.

U1/U2 use the same cached weekly snapshots with `k_ME=52/12` and `k_QE=13`; no refits are
made. The linear-regime reference is 0.562 and the saturation bound is 0.333. Selected
endpoints are:

| Universe/window | Config | Predicted annualised ratio | Realised annualised ratio | Realised - predicted |
|---|---|---:|---:|---:|
| U2 full | baseline | 0.389885 | 0.574356 | +0.184471 |
| U2 full | M1 star | 0.461997 | 0.972572 | +0.510575 |
| U1 full | baseline | 0.367121 | 0.405165 | +0.038044 |
| U1 full | M1 star | 0.667591 | 0.841153 | +0.173562 |
| U1 headline | baseline | 0.368314 | 0.411850 | +0.043537 |
| U1 headline | M1 star | 0.688068 | 0.942567 | +0.254499 |

The corrected P4 equality is REJECTED across the configuration grid: realised exceeds predicted
for every U1/U2 row. Mean absolute gaps are 0.418458 for U2, 0.200858 for U1 full, and
0.247759 for U1 headline. The cross-config predicted/realised correlations are respectively
0.610102, 0.542104, and 0.624380. The corrected conclusion reverses the direction stated in
the original E3 report: the old 1.685 constant was not the requested annualised
probability-sum yardstick. U1 baseline is close in level, but the equality does not hold over
the grid.

U3's native ME-versus-QE rows are labelled
`DESCRIPTIVE_DIFFERENT_SLEEVES_AND_SPANS` and are excluded from the P4 verdict. Its baseline
predicted/realised pair is 0.017806/0.207850 and M1-star is 0.028483/0.323961.

## Kurtosis proportionality constant

The `Gaussian predicted` column is computed at baseline delta zero with kappa removed from
sigma. Asset-date crossing probabilities are summed, then scaled by the observed count of
transition pairs and panel years to the annualised churn convention. The explicit baseline
constant `c = realised / predicted` is:

| Universe/frequency/window | Gaussian predicted | Realised | c |
|---|---:|---:|---:|
| U2 W-WED full | 0.824198 | 0.666134 | 0.808221 |
| U3 ME full | 1.250197 | 1.651793 | 1.321226 |
| U3 QE full | 0.159423 | 0.343325 | 2.153557 |
| U1 W-WED full | 3.169685 | 3.545476 | 1.118558 |
| U1 W-WED headline | 3.109880 | 3.212423 | 1.032973 |

As specified by the owner, the theory absorbs `c`; the testable content remains
cross-configuration correlation and calibrated-delta placement.

## Workbook delivery

The owner's house-convention ruling superseded the prior artifact-tool escalation. The three
workbooks were emitted with `qis.save_df_to_excel`; `metric_suite` was opened and re-read from
each workbook and matched its source shape.

| Workbook | Sample shape | Bytes | Re-read |
|---|---:|---:|---|
| `msci_us_stability_20260813.xlsx` | 16 x 61 | 17,304,113 | PASS |
| `futures_stability_20260813.xlsx` | 8 x 59 | 4,352,040 | PASS |
| `mac_stability_20260813.xlsx` | 8 x 59 | 5,925,427 | PASS |

Each workbook lives in its universe directory under the E3 evidence root. Long sheet names
were deterministically sanitized by qis to Excel's 31-character limit.

## Acceptance and verification

Independent validation output:

```text
futures_metric_grid: PASS (8 rows; windows={'full_panel': 295})
futures_fidelity: PASS (8 PASS, 0 REJECTED; all marked)
futures_residual_guard: PASS (max smoothed relative change=0.006950759 <= 0.05)
mac_metric_grid: PASS (8 rows; windows={'full_panel': 284})
mac_fidelity: PASS (7 PASS, 1 REJECTED; all marked)
mac_residual_guard: PASS (max smoothed relative change=0.013975155 <= 0.05)
msci_us_metric_grid: PASS (16 rows; windows={'full_panel': 238,
  'headline_20090831_20260630': 203})
msci_us_fidelity: PASS (11 PASS, 5 REJECTED; all marked)
msci_us_residual_guard: PASS (max smoothed relative change=0.007017544 <= 0.05)
E3b: PASS (legacy superset diagnosed; corrected sets 0 difference;
  32 scaling rows; 3 workbooks reread)
E4: PASS (4 runs; vocabulary present; 3 coverage>=0.70 cases per universe)
E5 futures: PASS evidence; residual guard 7/8 rows PASS, max=0.118217
E5 mac: PASS evidence; residual guard 0/7 rows PASS, max=0.323679
```

Focused regression assertions are 20/20 green in aggregate (19 in the combined process plus
the frozen S&P regression independently), and selected E/F/W lint is green on all changed
replication files. The injected-versus-fitted second pass independently confirms 1,666/1,666
corrected U1 partitions. A full cache-first deterministic replay passed with all 20 U1
stability/E3b CSV files byte-identical. The initial refresh-versus-cache comparison differed
only in serialization: its pre-replay workbook comparison found identical content throughout,
apart from Excel boolean representation and a maximum 1.0e-11 Excel rounding difference in
`metric_suite`. No files were staged or pushed.

## GATE REQUEST

The owner must rule on:

1. The corrected U1 fidelity verdicts: M0 hold, M1-0.02, M2-0.5, and M2-0.7 in-band on both
   windows; M1-0.05 in-band only on the full panel; M1-0.10 and M1-star rejected.
2. The corrected P4 verdict: annualised equality is rejected across U1/U2 configs with
   realised ratios above predicted; U3 is descriptive and excluded.
