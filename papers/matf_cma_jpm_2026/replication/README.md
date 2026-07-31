# Replication — MATF-CMA (JPM 2026)

Code that produces the exhibits the manuscript calls and the numbers it quotes.
The scripts write into `../paper/figures/`, and `paper/matf_cma_paper.tex` sets
`\graphicspath{{figures/}}`, so the exhibit build's output is what the compiled
manuscript reads. Named data artifacts the text quotes
(`bootstrap_headline_q2.json`, `sr_sensitivity_q2.csv`) land there too. Drop-in
LaTeX fragments for new and regenerated tables land in `figures/` beside the
scripts and are never spliced into the manuscript.

## Data: the shared layer

This package reads its inputs from the shared data layer at
`../../cma_data` (universe, benchmarks, loaders, the Consensus provider, and
immutable snapshots), resolved through `local_path.py` — optionally overridden by
a flat `settings.yaml` (key `CMA_DATA_PATH`), zero configuration needed on a
fresh clone. The pinned frozen cut is `SNAPSHOT = '2026q2'` in
`governed_cma_projection.py` and is imported from there, never re-pinned.
Loaders verify the snapshot's manifest hashes on every run. See
`../../cma_data/README.md` for the schema and freeze rules. The former per-paper
`data/` folder and its extractor are retired (superseded 2026-07-30).

```
replication/
  conftest.py                     pytest rootdir marker (no sys.path mutation)
  local_path.py                   path resolution + cma_data import
  exhibit_style.py                shared palette, save helpers, qis table grammar
  governed_cma_projection.py      Cap 3 / governed-set analysis (pins SNAPSHOT)
  run_optimisation.py             the single solve layer (moments, mandate, frontier)
  run_snapshot_tables.py          J1  snapshot-only LaTeX value tables
  run_factor_history_exhibits.py  J2  factor NAV exhibits
  run_universe_exhibits.py        J3  per-asset / benchmark / SR2 exhibits
  run_mandate_exhibits.py         J4b mandate frontier family
  run_admission_exhibits.py       J4c/e/f admission dial, scenarios, governed dial
  run_consistency_exhibits.py     J4d historical-mean comparator
  run_provider_exhibits.py        J4g Consensus now, A-D gated
  run_bootstrap_q2.py             J5  bootstrap, sensitivity, SE triple
  exhibit_cap3_projection.py      Cap 3 grid + implied-premia figure
  consensus_decomposition.py      Consensus GLS diagnostic
  excess_vs_total_optimisation.py excess-vs-total anchor-invariance proof
  tests/test_snapshot_parity.py   parity harness (16 tests)
  figures/                        tex fragments and script side outputs
  notes/                          dated per-stage records and the two reports
```

### What runs on a public checkout

The four config files of the snapshot ship publicly; the three return panels do
not (licensed index and factor histories — see `../../cma_data/README.md`).
**Eight of the twelve scripts run in full from a public clone.** The other four
stop with a message naming the missing file:

| script | needs |
|---|---|
| `run_factor_history_exhibits.py` | `factor_navs.csv` |
| `run_snapshot_tables.py` | `factor_navs.csv`, for `tab:factor_returns` only |
| `run_consistency_exhibits.py` | `asset_excess_logreturns.csv` |
| `run_bootstrap_q2.py` | both |

`pytest tests/` is green either way: 16 passed with the panels present, 15
passed and 1 skipped without them.

Run order for a full rebuild. Each script is standalone; only `tab:sharpe_cal`'s
empirical-SR column crosses scripts, so J2 runs before the final J1 pass.

```
python -m pytest tests/
python run_factor_history_exhibits.py      # writes figures/factor_empirical_sr.csv
python run_snapshot_tables.py              # joins that column into tab:sharpe_cal
python exhibit_cap3_projection.py
python run_universe_exhibits.py
python run_optimisation.py                 # acceptance report, no figures
python run_mandate_exhibits.py
python run_admission_exhibits.py
python run_consistency_exhibits.py
python run_provider_exhibits.py
python run_bootstrap_q2.py                 # ~6 minutes at B = 500
```

## Modules

| script | produces | depends on |
|---|---|---|
| `governed_cma_projection.py` | Sharpe accounting (ceiling, `SR2_MATFCMA`, FPIR), GLS decomposition, caps audit, Cap 3 grid; `figures/governed_projection_2026q2.xlsx` | numpy, pandas, cma_data |
| `run_optimisation.py` | moments, the mandate solve, the long-only frontier, the reporting layer; no figures | optimalportfolios, cvxpy |
| `exhibit_style.py` | palette, `save_figure`, `write_fragment`, `table_figure`; no mathematics | matplotlib, qis |
| `exhibit_cap3_projection.py` | `cap3_implied_premia.PNG`, `figures/exhibit_cap3_draft.tex` | matplotlib, governed_cma_projection |
| `consensus_decomposition.py` | Consensus GLS diagnostic report | cma_data.consensus |
| `excess_vs_total_optimisation.py` | anchor-invariance proof, two panels | cvxpy, optimalportfolios |
| `run_bootstrap_q2.py` | `bootstrap_frontier.PNG`, `bootstrap_headline_q2.json`, `sr_sensitivity_q2.csv`, `figures/exhibit_sr_sensitivity.tex` | cvxpy |

## Figure-to-script map

Left column is the name in `../paper/figures/` (case matters — the manuscript
calls it exactly). Middle column is the manuscript label. Right column is the
script that owns it.

| figure / artifact | label | script |
|---|---|---|
| `unintended_exposures.PNG` | `fig:unintended_exposures` | `run_consistency_exhibits.py` |
| `declared_delta.PNG` | `fig:declared_delta` | `run_consistency_exhibits.py` |
| `construction_waterfall.PNG` | `fig:construction_waterfall` | `run_universe_exhibits.py` |
| `cma_snapshot.png` | `tb:cma_snapshot` | `run_universe_exhibits.py` |
| `factor_attribution.png` | `tb:factor_attribution` | `run_universe_exhibits.py` |
| `benchmark_table.png` | `tab:benchmark_table` | `run_universe_exhibits.py` |
| `sr2_decomposition.PNG` | `fig:sr2_decomposition` | `run_universe_exhibits.py` |
| `risk_factors_perf.PNG` | `tb:risk_factors_perf` | `run_factor_history_exhibits.py` |
| `risk_factors_corr.PNG` | `tb:risk_factors_corr` | `run_factor_history_exhibits.py` |
| `risk_factors_annual.PNG` | `tb:risk_factors_annual` | `run_factor_history_exhibits.py` |
| `efficient_frontier.PNG` | `fig:illust_frontier` | `run_mandate_exhibits.py` |
| `factor_exposures.png` | `tab:factor_exposures` | `run_mandate_exhibits.py` |
| `admission_dial.PNG` | `fig:admission_dial` | `run_admission_exhibits.py` |
| `admission_dial_nobox.PNG` | `fig:admission_nobox` | `run_admission_exhibits.py` |
| `scenario_admission.PNG` | `fig:scenario_admission` | `run_admission_exhibits.py` |
| `dial_sweeps.PNG` | `fig:dial_sweeps` (new) | `run_admission_exhibits.py` |
| `sleeve_tornado.PNG` | `fig:sleeve_tornado` | `run_admission_exhibits.py` |
| `governed_dial.PNG` | `fig:governed_dial` (new, App E) | `run_admission_exhibits.py` |
| `cap3_implied_premia.PNG` | `fig:cap3_implied_premia` (new, App E) | `exhibit_cap3_projection.py` |
| `bootstrap_frontier.PNG` | `fig:bootstrap_frontier` | `run_bootstrap_q2.py` |
| `bootstrap_headline_q2.json` | §"How Much to Trust the Numbers" | `run_bootstrap_q2.py` |
| `sr_sensitivity_q2.csv` | `tab:sr_sensitivity` | `run_bootstrap_q2.py` |
| `provider_frontier_with_alts.PNG` | `fig:provider_frontier` | `run_provider_exhibits.py` — **BLOCKED on `providers.csv`** |
| `provider_frontier_wo_alts.PNG` | `fig:provider_frontier_wo` | `run_provider_exhibits.py` — **BLOCKED on `providers.csv`** |
| `equity_factor_cmas.PNG` | `tb:equity_factor_cmas` | **BLOCKED**, needs a regional P-CAEY extract |
| `rates_factor_cmas.PNG` | `tb:rates_factor_cmas` | **BLOCKED**, needs a per-country TP / roll-down extract |

## Tex fragments

Drop-in bodies and figure blocks in `figures/`, never spliced into the manuscript.

| fragment | target |
|---|---|
| `exhibit_nine_factors.tex` | `tab:nine_factors` body |
| `exhibit_admission_audit.tex` | `tab:admission_audit` body + Σ IR² line + Cap 3 row |
| `exhibit_sharpe_cal.tex` | `tab:sharpe_cal` body |
| `exhibit_factor_returns.tex` | `tab:factor_returns` body |
| `exhibit_cap3_draft.tex` | `tab:cap3_grid` table + `fig:cap3_implied_premia` figure |
| `exhibit_scenario_floors.tex` | §5.3 floor table (both admission policies) |
| `exhibit_governed_dial.tex` | `fig:governed_dial` figure block |
| `exhibit_provider_saa.tex` | `tab:provider_saa` Bench / MATF / Consensus columns |
| `exhibit_provider_decomposition.tex` | `tab:provider_decomposition` table (new) |
| `exhibit_sr_sensitivity.tex` | `tab:sr_sensitivity` body + `tab:se_comparison` values |
| `exhibit_factor_history_notes.tex` | caption changes for the three J2 exhibits (defect D7) |
| `exhibit_universe_notes.tex` | caption changes for the J3 exhibits (defect D3, ILS share) |
| `exhibit_mandate_notes.tex` | caption notes for the two J4b exhibits (defect D8) |
| `exhibit_consistency_notes.tex` | caption notes for the two J4d exhibits |

## Configuration of the frozen cut

- 2026-Q2 production cut, 18 assets (14 monthly, 4 quarterly), USD, `rf` 4.18%.
- Premia: the **July** production config (Equity 3.98%, Rates 1.01%, PE 4.20%,
  50/50 current-vs-equilibrium blends). This IS the R3 freeze (register K2).
- Optimizer: `wrapper_maximise_alpha_over_tre` with `FORCED_CONSTRAINTS`, ±50%
  box around the benchmark, tracking error capped at 1.5%.
- Benchmarks: `cma_data/benchmarks.py`, D8-correct (the R2 exhibit build
  transposed Asia ex-Japan against EM ex-Asia).
- Every optimization runs on **excess** CMAs; the reference cash rate enters only
  at the reporting layer.
- Bootstrap: B = 500, seed 42, window Jul 2001 – Jun 2026, T = 300 months,
  blocks mean 12 / min 3, **July** SR prior
  [0.40, 0.25, 0.40, 0.25, 0.15, 0.15, 0.60, 0.25, 0.00], σ_SR = 0.10, raw panel
  recentered, **no pre-inception backfill** (the proxies are not in the snapshot).
- Private equity admitted at w = 0.5. The **production policy** is `w_paper`;
  the **pre-recut workbook policy** is `w_workbook` (owner ruling O-J11 / B6).
- Package versions of the extraction run, from the manifest: `qis` 5.0.5,
  `optimalportfolios` 6.6.0, `factorlasso` 0.10.1.

## Reports

| file | contents |
|---|---|
| `notes/ROADMAP_jpm_exhibits_NUMBER_CHANGES.md` | every quantitative statement in the manuscript traced to this package, by R3 section |
| `notes/ROADMAP_jpm_exhibits_COMPLETION_REPORT.md` | per-stage outcomes, tex edits, deviations, blocked register, cross-paper flag |
| `notes/ROADMAP_jpm_exhibits_J0b_CORRECTION_2026-07-30.md` | the `equity_regional_addon` double-count fix and every number that moved |
| `notes/governed_projection_findings_2026q2.md` | the earlier governed-projection extraction findings |
| `notes/clm_ch6_application_note.tex` / `.pdf` | the CLM Ch. 6 application note |

## Open items

- **`providers.csv`** (owner item O-J7b) unblocks the A–D provider columns, rows
  and both frontier PNGs. Drop it into `../../cma_data/snapshots/2026q2/` against
  the schema in `run_provider_exhibits.PROVIDERS_SCHEMA` and re-run.
- **Regional P-CAEY and per-country rates extracts** unblock the two Appendix C
  exhibits. Minimal shapes are specified in the completion report's register.
- **Cross-paper bootstrap prior (O-J8)**: this package uses the July prior; the
  FAJ instruction file pins the old Q2 prior. Owner call.
- **Appendix B prose** needs two corrections the code forced: the recentering step
  is not stated, and the backfill sentence describes proxies the snapshot does not
  carry.
- **`[UNTRACED]` Grinold-Kroner SE**: the R2 3.64% does not reproduce from the
  printed components with ρ = 0.5, which give 3.77%.
- **Cross-document consistency with `achievable_sharpe_faj_2026`**: that package is
  the 2026-Q1 cut on 17 assets, this one the 2026-Q2 cut on 18. Both report a
  factor premium identification ratio near 42%; this cut gives **38.7%**. Confirm
  the ratio's sensitivity to the eighteenth asset or state the difference.
- **`stationary_block_indices`** now lives in `run_bootstrap_q2.py` here and in a
  copy in the FAJ package. Porting it to `qis` and importing from there in both
  papers remains the right end state (library change, OSS project).
