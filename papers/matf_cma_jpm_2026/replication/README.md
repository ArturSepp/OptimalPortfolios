# Replication — MATF-CMA (JPM 2026)

Code that produces the exhibits the manuscript calls and the numbers it quotes.
The scripts write into `../paper/figures/`, and `paper/matf_cma_paper.tex` sets
`\graphicspath{{figures/}}`, so the exhibit build's output is what the compiled
manuscript reads. Named data artifacts the text quotes
(`bootstrap_headline_q2.json`, `sr_sensitivity_q2.csv`) land there too.

## Status

[TODO: populated from `matf_revision_exhibits_2026q2_1.zip`, the frozen 2026-Q2
exhibit build. The zip is pending transfer.]

## Figure-to-script map

Read from the provenance comments in `../paper/matf_cma_paper.tex`. The right
column is the exhibit-build filename, the left is the name in
`../paper/figures/`.

| figure | exhibit build | script |
|---|---|---|
| `unintended_exposures.PNG` | `exhibit_a3_unintended_exposures.png` | [TODO] |
| `declared_delta.PNG` | `exhibit_a2_declared_delta.png` | [TODO] |
| `construction_waterfall.PNG` | `exhibit_a1_construction_waterfall.png` | [TODO] |
| `provider_frontier_with_alts.PNG` | `exhibit_b5_frontier_with_alts.png` | `exhibit_b45_peer_frontier.py` |
| `provider_frontier_wo_alts.PNG` | [TODO] | `exhibit_b45_peer_frontier.py` |
| `admission_dial.PNG` | `exhibit_c8_admission_dial.png` | [TODO] |
| `admission_dial_nobox.PNG` | `exhibit_c8b_admission_dial_nobox.png` | [TODO] |
| `sleeve_tornado.PNG` | `exhibit_c9_sleeve_tornado.png` | [TODO] |
| `sr2_decomposition.PNG` | `exhibit_c10_sr2_decomposition.png` | [TODO] |
| `scenario_admission.PNG` | `exhibit_c12_scenario_admission.png` | [TODO] |
| `bootstrap_frontier.PNG` | `bootstrap_frontier_q2.png` | `run_bootstrap_q2.py` |
| `risk_factors_perf.PNG` | [TODO] | [TODO] |
| `risk_factors_corr.PNG` | [TODO] | [TODO] |
| `risk_factors_annual.PNG` | [TODO] | [TODO] |
| `equity_factor_cmas.PNG` | [TODO] | [TODO] |
| `rates_factor_cmas.PNG` | [TODO] | [TODO] |
| `cma_snapshot.png` | [TODO] | [TODO] |
| `factor_attribution.png` | [TODO] | [TODO] |
| `factor_exposures.png` | [TODO] | [TODO] |
| `benchmark_table.png` | [TODO] | [TODO] |
| `efficient_frontier.PNG` | [TODO] | [TODO] |
| `consistency_violation.PNG` | [TODO] | [TODO] |

## Named artifacts the manuscript quotes

| artifact | consumed by | script |
|---|---|---|
| `bootstrap_headline_q2.json` | Section "How Much to Trust the Numbers" | `run_bootstrap_q2.py` |
| `sr_sensitivity_q2.csv` | Appendix B standard-error table | `run_sensitivity_q2.py` |

## Configuration of the frozen cut

- 2026-Q2 production cut, 18 assets (14 monthly, 4 quarterly), USD.
- Bootstrap: B = 500, seed 42, window Jul 2001 to Jun 2026, T = 300 months.
- Q2 Sharpe prior [0.35, 0.35, 0.25, 0.25, 0.15, 0.15, 0.50, 0.25, 0.00],
  sigma_SR = 0.10.
- Private equity admitted at w = 0.5.

## Open items

- [TODO] Appendix C figures are still on the prior cut. Regenerate the regional
  P-CAEY exhibits on the Q2 cut.
- [TODO] The Swiss sleeve worked example carries the prior cut's values.
- [TODO] State the `qis` and `optimalportfolios` versions each result depends on.
- [TODO] Cross-document consistency with `achievable_sharpe_faj_2026`. That
  package is the 2026-Q1 cut on 17 assets, this paper is the 2026-Q2 cut on 18
  assets, and each paper attributes its universe to the other. Both report a
  factor premium identification ratio near 42%. Confirm the ratio is insensitive
  to the eighteenth asset, or regenerate and state the difference.
- [TODO] The FAJ replication README names
  `matf_cma_jpm_2026/bootstrap_frontier_analytics.py` as the source of its
  `stationary_block_indices` copy. That file is not in this folder yet. Confirm
  it arrives with the exhibit build, then port the function to `qis` and have
  both papers import it from there.
