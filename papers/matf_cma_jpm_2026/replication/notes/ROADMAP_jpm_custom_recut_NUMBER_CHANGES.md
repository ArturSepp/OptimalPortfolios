# Number-change report — MATF-CMA JPM custom recut

2026-08-12. Roadmap `papers/cma_data/claude/ROADMAP_jpm_custom_recut.md`, Stage S6.
Old means the immutable `cma_data/snapshots/2026q2` cut; new means
`cma_data/snapshots/2026q2_custom`. The producing scripts are under
`papers/matf_cma_jpm_2026/replication/`.

The delta bundles the factor-model change with the August market-data revintage.
The governed priors and premia configuration are unchanged between the two snapshots;
the factor-model share is documented separately in
`rosaa/docs/implementation_notes/2026-08-12_matf_base_vs_custom_comparison.md`.
No delta below should be attributed to MATF_CUSTOM alone.

## Headline chain

| Manuscript quantity | Old | New | Producer |
|---|---:|---:|---|
| Balanced with Alts expected total return | 8.13% | **8.28%** | `run_optimisation.py` |
| Excess Sharpe, admission scale s = 0 | 0.31 | **0.33** | `run_admission_exhibits.py` |
| Excess Sharpe, production s = 1 | 0.369 | **0.385** | `run_admission_exhibits.py` |
| Frictionless factor ceiling | 0.614 | **0.529** | `governed_cma_projection.py` |
| Attainable systematic SR² | 0.238 | **0.205** | `governed_cma_projection.py` |
| FPIR | 38.7% | **38.8%** | `governed_cma_projection.py` |
| Raw admitted-alpha SR² | 1.398 | **1.444** | `governed_cma_projection.py` |
| GLS-projected admitted-alpha SR² | 0.625 | **0.481** | `governed_cma_projection.py` |
| Full-vector SR²_alpha | 0.456 | **0.402** | `governed_cma_projection.py` |
| Consensus SR²_alpha, 17 sleeves | 0.091 | **0.099** | `run_provider_exhibits.py` |
| Scenario total: base / 2022 stress / 2023 upside | 8.13 / 4.63 / 9.98% | **8.28 / 4.84 / 10.16%** | `run_admission_exhibits.py` |
| Scenario band width across the admission sweep | 5.24–5.39% | **5.22–5.32%** | `run_admission_exhibits.py` |
| Bootstrap 90% raw bandwidth at 9.3% vol | 6.68% | **6.49%** | `run_bootstrap_q2.py` |
| Bootstrap 90% MATF bandwidth at 9.3% vol | 3.34% | **2.85%** | `run_bootstrap_q2.py` |
| Bootstrap reduction at 9.3% vol | 2.00× | **2.28×** | `run_bootstrap_q2.py` |
| Bootstrap mean reduction across grid | 1.98× | **2.24×** | `run_bootstrap_q2.py` |
| SE triple: sample / GK / MATF | 3.24 / 3.77 / 1.53% | **3.25 / 3.77 / 1.52%** | `run_bootstrap_q2.py` |

The bootstrap remained B = 500, seed 42, Jul-2001–Jun-2026, mean/minimum blocks
12/3 months and sigma_SR = 0.10. Its prior expanded by family mapping to
`[0.40, 0.25, 0.40, 0.40, 0.25, 0.25, 0.15, 0.15, 0.60, 0.25, 0.00]`.

## Key takeaways, abstract and conclusion

| Current statement or chain | Old | New | Producer |
|---|---:|---:|---|
| “Production admission lifts Sharpe” | 0.31 → 0.37 | **0.33 → 0.39** | `run_admission_exhibits.py` |
| “One half beyond production” | 0.41 | **0.43** | `run_admission_exhibits.py` |
| Alternatives at the Balanced optimum | 37.3% | **37.0%** | `run_optimisation.py` |
| Unconstrained alternatives / PE / Sharpe | 81 / 58% / 0.53 | **48 / 18% / 0.58** | `run_optimisation.py` |
| Largest historical-mean residual | 6.44% | **5.60%** | `run_consistency_exhibits.py` |
| Bandwidth statement | 2.00× / 4.0× T_eff | **2.28× / 5.2× T_eff** | `run_bootstrap_q2.py` |
| Bootstrap residual medians / portfolio norm | 30–115 bp / 3.4% | **5–180 bp / 4.20%** | `run_bootstrap_q2.py` |

Provider figures attributed to named providers remain **[UNTRACED]** because
`providers.csv` has not landed. The manuscript’s 8.6%, 11.3%, provider-gap and
provider-frontier statements must not be updated from this recut.

## Section 2 — provider comparison

The Consensus-only path is traceable; provider A–D remains **[UNTRACED]**.

| Quantity | Old | New | Producer |
|---|---:|---:|---|
| MATF total return / vol / excess Sharpe | 8.13% / 10.7% / 0.37 | **8.28% / 10.6% / 0.39** | `run_provider_exhibits.py` |
| Consensus total return / vol / excess Sharpe | 8.42% / 10.7% / 0.40 | **8.43% / 10.6% / 0.40** | `run_provider_exhibits.py` |
| MATF implied Eq / Rates / Credit premia | 435 / 106 / 96 bp | **381 / 102 / 84 bp** | `run_provider_exhibits.py` |
| Consensus implied Eq / Rates / Credit premia | 373 / −7 / 323 bp | **368 / −11 / 156 bp** | `run_provider_exhibits.py` |
| MATF s²_h / s²_K / tangency | 0.359 / 0.238 / 0.77 | **0.347 / 0.205 / 0.74** | `run_provider_exhibits.py` |
| Consensus s²_h / s²_K / tangency | 0.091 / 0.238 / 0.57 | **0.099 / 0.205 / 0.55** | `run_provider_exhibits.py` |
| Largest Consensus sleeve residuals | Gold −75, RE +65, EM HC +58 bp | **Global IG −45, Global HY +27, EM HC +24 bp** | `run_provider_exhibits.py` |

Selected optimal-weight changes are MATF US 34.0→29.0%, Hedge Funds 2.8→2.5%,
and Consensus US 28.6→29.3%, Real Estate 2.9→2.3%. Benchmark weights and the
4.52/0.88% Asia ex-Japan / EM ex-Asia pair are unchanged.

## Section 3 — eleven-factor construction

`tab:nine_factors` retains its filename and label for O4 diff hygiene, but now has
eleven rows.

| Factor | Old premium / vol / ratio | New premium / vol / ratio | Producer |
|---|---:|---:|---|
| Equity | 3.98 / 15.4 / 0.26 | **3.82 / 15.5 / 0.25** | `run_snapshot_tables.py` |
| Rates | 1.01 / 5.6 / 0.18 | **1.00 / 5.6 / 0.18** | `run_snapshot_tables.py` |
| Credit | 1.42 / 4.7 / 0.30 | **1.42 / 4.6 / 0.31** | `run_snapshot_tables.py` |
| Credit EM | — | **1.64 / 4.6 / 0.36** | `run_snapshot_tables.py` |
| Carry (legacy) | 1.25 / 4.5 / 0.28 | retired | `run_snapshot_tables.py` |
| Carry G10 | — | **1.02 / 4.8 / 0.21** | `run_snapshot_tables.py` |
| Carry EM | — | **1.17 / 4.3 / 0.27** | `run_snapshot_tables.py` |
| Inflation | 0.75 / 4.7 / 0.16 | **0.67 / 4.5 / 0.15** | `run_snapshot_tables.py` |
| Commodities | 2.25 / 16.0 / 0.14 | **2.28 / 16.1 / 0.14** | `run_snapshot_tables.py` |
| Private Equity | 4.20 / 7.1 / 0.59 | **2.99 / 7.0 / 0.43** | `run_snapshot_tables.py` |
| Rates Vol | 1.25 / 5.2 / 0.24 | **1.13 / 5.2 / 0.22** | `run_snapshot_tables.py` |
| FX | 0.00 / 6.9 / 0.00 | **0.00 / 6.8 / 0.00** | `run_snapshot_tables.py` |

The static calibration table still reports family prior × configured target vol.
With beta priors ON, it no longer equals every published base premium; the maximum
gap is 121.49 bp. This distinction requires owner wording in the manuscript sweep.

## Section 4 — alpha admission and caps

| Sleeve | Old alpha / admitted / excess / share / IR | New alpha / admitted / excess / share / IR | Producer |
|---|---:|---:|---|
| Private Equity | 2.80 / 1.40 / 5.35% / 26% / 0.19 | **3.25 / 1.63 / 5.32% / 31% / 0.22** | `run_snapshot_tables.py` |
| Private Credit | 1.85 / 0.93 / 3.27% / 28% / 0.19 | **2.47 / 1.24 / 3.37% / 37% / 0.26** | `run_snapshot_tables.py` |
| Real Estate | −2.48 / 0.00 / 2.26% / 0% / 0.00 | **−1.63 / 0.00 / 2.67% / 0% / 0.00** | `run_snapshot_tables.py` |
| Insurance-Linked | 3.58 / 3.58 / 3.84% / 93% / 0.83 | **3.58 / 3.58 / 3.85% / 93% / 0.83** | `run_snapshot_tables.py` |
| Hedge Funds | 2.04 / 2.04 / 3.37% / 61% / 0.77 | **2.06 / 2.06 / 3.46% / 60% / 0.78** | `run_snapshot_tables.py` |
| Gold | 13.93 / 3.48 / 4.33% / 80% / 0.23 | **10.83 / 2.71 / 4.28% / 63% / 0.18** | `run_snapshot_tables.py` |

Cap 1 and Cap 2 verdicts are unchanged: ILS and Hedge Funds fail both; Gold fails
Cap 2 only. Cap 3 changes as follows:

| Policy | Old budget / theta / weights PE-PC-ILS-HF-Gold | New budget / theta / weights PE-PC-ILS-HF-Gold | Producer |
|---|---:|---:|---|
| production | — / 1.00 / .50-.50-1-1-.25 | **— / 1.00 / .50-.50-1-1-.25** | `exhibit_cap3_projection.py` |
| kappa 1.00 | .238 / .41 / .21-.21-.41-.41-.10 | **.205 / .38 / .19-.19-.38-.38-.09** | `exhibit_cap3_projection.py` |
| kappa 0.50 | .119 / .29 / .15-.15-.29-.29-.07 | **.103 / .27 / .13-.13-.27-.27-.07** | `exhibit_cap3_projection.py` |
| kappa 0.25 | .059 / .21 / .10-.10-.21-.21-.05 | **.051 / .19 / .09-.09-.19-.19-.05** | `exhibit_cap3_projection.py` |

Largest cuts become −223 / −263 / −291 bp from −210 / −254 / −284 bp. The
premium-like solo shares become PE 97%, PC 96%, ILS 79%, HF 37%, Gold 26%
from 84/77/61/29/29%. The implied-premia chart’s largest standardized deviation
is Rates Vol −0.72 (old −0.75); all remain inside one identification unit.

## Section 5 — scenarios and floors

| Factor | Old 2022 / 2023 annual return | New 2022 / 2023 annual return | Producer |
|---|---:|---:|---|
| Equity | −19.2 / 15.4% | **−19.2 / 15.4%** | `run_snapshot_tables.py` |
| Rates | −19.1 / 0.5% | **−19.1 / 0.5%** | `run_snapshot_tables.py` |
| Credit | −0.6 / 7.3% | **−0.6 / 7.3%** | `run_snapshot_tables.py` |
| Credit EM | — | **−10.8 / 2.9%** | `run_snapshot_tables.py` |
| Carry (legacy) | 1.4 / 2.1% | retired | `run_snapshot_tables.py` |
| Carry G10 | — | **−2.1 / 0.3%** | `run_snapshot_tables.py` |
| Carry EM | — | **13.1 / 4.4%** | `run_snapshot_tables.py` |
| Inflation | 3.6 / −0.8% | **8.1 / −1.7%** | `run_snapshot_tables.py` |
| Commodities | 13.5 / −10.0% | **13.5 / −10.0%** | `run_snapshot_tables.py` |
| Private Equity | −5.1 / 7.6% | **−5.1 / 7.6%** | `run_snapshot_tables.py` |
| Rates Vol | 5.4 / −1.3% | **5.4 / −1.3%** | `run_snapshot_tables.py` |
| FX | 11.8 / −0.6% | **11.7 / −0.6%** | `run_snapshot_tables.py` |

The recovered annual shocks (`factor_premia` scenario columns × 5) match the NAV
returns to 9.09e-13 bp; scenario CMA additivity holds to 8.33e-13 bp.

| Mandate | Old floor / stress production / headroom | New floor / stress production / headroom | Producer |
|---|---:|---:|---|
| Income w/o Alts | 1.25 / 2.85 / 1.61% | **1.36 / 3.19 / 1.83%** | `run_admission_exhibits.py` |
| Low w/o Alts | −0.46 / 3.19 / 3.65% | **−0.32 / 3.62 / 3.94%** | `run_admission_exhibits.py` |
| Balanced w/o Alts | −3.19 / 3.56 / 6.75% | **−3.03 / 3.93 / 6.96%** | `run_admission_exhibits.py` |
| Growth w/o Alts | −7.19 / 3.98 / 11.16% | **−7.01 / 4.28 / 11.28%** | `run_admission_exhibits.py` |
| Income with Alts | 1.48 / 3.37 / 1.89% | **1.58 / 3.71 / 2.12%** | `run_admission_exhibits.py` |
| Low with Alts | −0.06 / 4.07 / 4.13% | **0.10 / 4.42 / 4.31%** | `run_admission_exhibits.py` |
| Balanced with Alts | −1.83 / 4.63 / 6.46% | **−1.62 / 4.84 / 6.46%** | `run_admission_exhibits.py` |
| Growth with Alts | −3.77 / 4.92 / 8.69% | **−3.52 / 5.19 / 8.70%** | `run_admission_exhibits.py` |

All one-sigma floors remain slack under both admission policies.

## Section 6 — dials to weights

Production Sharpe moves 0.369→0.385. The largest sleeve Sharpe spans change:
Private Equity 0.059→0.070, Gold 0.050→0.039, ILS 0.007→0.006, Private Credit
0.006→0.006, Hedge Funds 0.003→0.003 and Real Estate −0.003→−0.002.
Cap-3 governed weights are the theta-scaled values in the cap table above; the
one-sigma floor remains 1.00/non-binding for every sleeve.

## Section 7 — consistency and sampling

| Quantity | Old | New | Producer |
|---|---:|---:|---|
| Median absolute historical residual | 0.39% | **0.63%** | `run_consistency_exhibits.py` |
| Median absolute MATF residual | 0.33% | **0.47%** | `run_consistency_exhibits.py` |
| Maximum historical residual | 6.44% | **5.60%** | `run_consistency_exhibits.py` |
| MATF reconciliation | 5.8e-17 | **1.2e-16** | `run_consistency_exhibits.py` |
| Idiosyncratic active-risk share, MATF / historical | 75 / 57% | **79 / 58%** | `run_consistency_exhibits.py` |
| Bootstrap residual median range | 30–115 bp | **5–180 bp** | `run_bootstrap_q2.py` |
| Median / p95 portfolio residual norm | 3.4% / not printed | **4.20 / 7.43%** | `run_bootstrap_q2.py` |

Bootstrap sensitivity moves from 3.88/2.68/2.00/1.62/1.37/1.04× to
**4.56/3.03/2.28/1.82/1.53/1.16×** over sigma_SR
0.050/0.075/0.100/0.125/0.150/0.200. T_eff equivalents become
**20.8/9.2/5.2/3.3/2.3/1.4×**.

## Appendix C factor-history quantities

Empirical Sharpe values on 2005–2026-Q2 change from legacy Carry 0.25,
Inflation 0.05, Commodities −0.10, PE 0.60 and Rates Vol 0.42 to Credit EM
**0.18**, Carry G10 **0.16**, Carry EM **0.51**, Inflation **−0.18**,
Commodities **−0.10**, PE **0.60**, Rates Vol **0.42**. Credit–Equity correlation
under the production W-WED/span-260 spec remains 0.86. The regional P-CAEY and
per-country rates exhibits remain **[UNTRACED]** and their four PNGs were not rebuilt.

## Per-asset snapshot comparison

`stat_alpha` and `factor_excess_cma` are percent per annum; `r2` is decimal.

| Sleeve | alpha old | alpha new | r2 old | r2 new | factor CMA old | factor CMA new |
|---|---:|---:|---:|---:|---:|---:|
| Global Government | −0.054 | −0.046 | 0.909 | 0.907 | 0.752 | 0.749 |
| Global IG Bonds | −0.476 | −1.094 | 0.832 | 0.846 | 1.182 | 1.622 |
| Global HY Bonds | −0.826 | −0.823 | 0.774 | 0.789 | 1.509 | 1.680 |
| EM HC Bonds | −0.218 | 0.144 | 0.736 | 0.870 | 1.507 | 2.082 |
| Global Inflation-Linked | −1.450 | −1.591 | 0.777 | 0.723 | 1.111 | 0.949 |
| US | 1.984 | 1.898 | 0.915 | 0.915 | 4.152 | 4.120 |
| Europe ex-UK | 4.932 | 2.946 | 0.797 | 0.815 | 4.384 | 5.465 |
| UK | 5.549 | 3.868 | 0.673 | 0.693 | 5.314 | 6.332 |
| Switzerland | 6.231 | 4.003 | 0.729 | 0.742 | 4.668 | 6.020 |
| Japan | 5.042 | 4.621 | 0.683 | 0.676 | 3.748 | 4.067 |
| Asia ex-Japan | 5.514 | 5.163 | 0.654 | 0.650 | 4.098 | 4.411 |
| EM ex-Asia | 0.935 | −2.230 | 0.628 | 0.674 | 4.521 | 5.863 |
| Private Equity | 2.799 | 3.253 | 0.627 | 0.644 | 3.953 | 3.690 |
| Private Credit | 1.854 | 2.473 | 0.645 | 0.662 | 2.343 | 2.134 |
| Real Estate | −2.479 | −1.629 | 0.569 | 0.602 | 2.262 | 2.669 |
| Insurance-Linked | 3.583 | 3.580 | 0.166 | 0.164 | 0.257 | 0.266 |
| Hedge Funds | 2.042 | 2.059 | 0.786 | 0.790 | 1.325 | 1.399 |
| Gold | 13.930 | 10.826 | 0.296 | 0.324 | 0.845 | 1.574 |

## Unchanged controls and verification

- Universe 18; rf 4.18%; PE paper admission 0.50.
- Optimizer `wrapper_maximise_alpha_over_tre`, ±50% box, 1.5% TE cap.
- B = 500, seed 42, 300 months, blocks 12/3, sigma_SR = 0.10.
- Waterfall maximum gap: 7.63e-17 decimal (7.63e-13 bp).
- Scenario de-compounding/additivity: 9.09e-13 / 8.33e-13 bp.
- All 30 shared-data and replication parity tests pass.

## Owner decisions still open

- **O1 — FAJ coordination:** FAJ remains on nine-factor `2026q2`; the companion
  0.57/0.24/1.40/0.63 chain is therefore on a different factor model.
- **O2 — Carry EM prior/citation:** both carry factors inherit 0.25 and cite
  Lustig et al. pending owner wording.
- **O3 — Credit EM calibration:** PD 2.0% / LGD 60% mirrors HY pending a governed
  sovereign calibration.
- **O4 — names:** `tab:nine_factors` and `exhibit_nine_factors.tex` now contain
  eleven rows.
- **O5 — providers.csv:** all provider A–D numbers and both provider frontiers
  remain **[UNTRACED]**.
- **O6 — regional extracts:** `equity_factor_cmas.PNG` and
  `rates_factor_cmas.PNG` remain blocked and binary-unchanged.
