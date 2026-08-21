# F1 report — stability consolidation, margins, frontier, and calibration bridge

**Date:** 2026-08-20  
**Roadmap:** `agents/ROADMAP_manuscript_finalisation.md` v2  
**Status:** COMPLETE

## Execution and outputs

Runner: `papers/cluster_lineage_2026/replication/run_f1_stability_consolidation.py`.
Focused test: `replication/f1_stability_consolidation_test.py`.

Output directory:
`C:/Users/artur/OneDrive/analytics/outputs/cluster_lineage_2026/finalisation/f1/`.

Principal artifacts are `margins_flip_rates.csv`, `margin_histogram.pdf`, `frontier.csv`,
`frontier_knee.pdf`, `calibration_bridge.csv`, and `ergodicity.csv`. The directory also
contains `source_manifest.csv`, `run_parameters.csv`, `acceptance.csv`, and
`determinism.csv`.

All inputs trace to the F0 inventory. The equity rows assert the corrected E3b point-in-time
asset sets; the superseded pre-correction U1 outputs were not read. Bootstrap parameters are
block length 6, 2,000 draws, seed 20260813, circular moving blocks, and joint indices across
every configuration series within a frequency.

## P1 margins and predicted-versus-realised churn

The realised flip is the frozen P1 convention: raw clusters are greedily relabelled by
maximum consecutive member overlap separately by analysis window and native frequency. It
is not MCF lineage churn. Reconstructed per-date baseline flip counts match the frozen E3b
tables exactly.

| Panel/window | Singleton convention | Correlation | 95% block CI |
|---|---|---:|---:|
| Equity, full | including | 0.863096 | [0.850941, 0.871491] |
| Equity, full | excluding | 0.862986 | [0.850135, 0.871740] |
| Equity, headline | including | 0.871747 | [0.864663, 0.878447] |
| Equity, headline | excluding | 0.871631 | [0.865058, 0.878433] |
| Futures, full | including | 0.864768 | [0.814225, 0.903031] |
| Futures, full | excluding | 0.867473 | [0.820225, 0.907610] |
| Fund, full | including | 0.911033 | [0.885480, 0.931156] |
| Fund, full | excluding | 0.909712 | [0.883603, 0.928666] |

The margin CSV contains ten deciles for each panel/frequency/window/singleton convention,
with observed mass, margin bounds, Gaussian probability, and realised flip rate. This makes
the small-margin concentration directly reproducible from the table behind the figure.

## P2 frontier knees

Knees use maximum discrete Menger curvature after independently normalising frontier
fidelity and churn to [0, 1]. For the combined fund frontier the ME delta orders the path;
both ME and QE calibrated values remain visible in the labels.

| Panel/window | Knee | Curvature |
|---|---|---:|
| Equity, full | calibrated 0.0866 | 2.091297 |
| Equity, headline | fixed 0.05 | 1.810942 |
| Futures, full | calibrated 0.0691 | 1.585105 |
| Fund, full | calibrated ME 0.0830 / QE 0.1609 | 5.071886 |

The headline/full difference for equities is reported rather than resolved by selection.

## Calibration bridge

| Row | N | k | kappa-hat | rho-bar | delta-star level | delta-star innovation | configured/adopted |
|---|---:|---:|---:|---:|---:|---:|---:|
| Equity theory W-WED | 156 | 4.3333 | 2.124418 | 0.622741 | 0.086638 | 0.028483 | 0.0866 |
| Futures theory W-WED | 156 | 4.3333 | 1.612290 | 0.682899 | 0.069057 | 0.022703 | 0.0691 |
| Fund theory ME | 36 | 1 | 0.836854 | 0.795197 | 0.083049 | 0.027306 | 0.0830 |
| Fund theory QE | 12 | 1 | 1.287959 | 0.794647 | 0.160921 | 0.089263 | 0.1609 |
| U1 application ME/36 | 36 | 1 | 2.124418 | 0.636238 | 0.175346 | 0.057654 | 0.0866 |
| U3 application W-WED/156 | 156 | 4.3333 | 1.612290 | 0.682899 | 0.069057 | 0.022703 | 0.0691 |

The four theory rows reproduce every owner-frozen calibration marker after rounding to four
decimals. The application bridge makes the transfer explicit: U1's own ME/36 baseline has
954,620 within-cluster pair observations over 203 dates and implies a 0.1753 level threshold;
the adopted 0.0866 is the weekly-theory-cell calibration transferred to the selected ME/36
application, not an own-cell recalibration. The equity kappa-hat is the only frozen E1 equity
estimate and is therefore retained for this diagnostic row.

## P6 ergodicity evidence

`ergodicity.csv` reports halves, thirds, and GFC/COVID/rate-shock windows for cluster count,
size entropy, consecutive ARI, and VI, each with a joint moving-block interval. Across
halves/thirds, calibrated consecutive ARI ranges are 0.922--0.970 (equities), 0.977--0.992
(futures), and 0.964--0.979 (funds), versus baseline ranges 0.545--0.673,
0.891--0.951, and 0.791--0.831. The calibrated process is also more persistent in every
reported crisis window; these are subsample stability results, not independent samples.

## Acceptance checks

| Check | Measured | Tolerance | Result |
|---|---:|---:|---|
| F0 source paths resolved once | 36/36 | 36/36 | PASS |
| F0 source failures | 0 | 0 | PASS |
| Corrected U1 configurations | 7 | 7 | PASS |
| Corrected U1 maximum asset-set difference | 0 | 0 | PASS |
| Superseded U1 legacy tables consumed | 0 | 0 | PASS |
| P1 frozen-correlation regression error | `1.387779e-15` | `<=1e-12` | PASS |
| Baseline per-date flip-count regression error | 0 | `<=1e-12` | PASS |
| Rounded calibration regression error | 0 | 0 | PASS |
| NaNs across numerical deliverables | 0 | 0 | PASS |
| Bootstrap block length | 6 | 6 | PASS |
| Bootstrap draws | 2,000 | 2,000 | PASS |
| Bootstrap seed | 20260813 | 20260813 | PASS |
| Deterministic CSV/PDF artifacts | 9/9 byte-identical | 9/9 | PASS |
| Focused pytest | 3 passed | all pass | PASS |
| Isolated Ruff E/F/W | 0 findings | 0 | PASS |

No estimator was fit, no empirical search was performed, and no source cache was modified.
No git staging or push occurred.
