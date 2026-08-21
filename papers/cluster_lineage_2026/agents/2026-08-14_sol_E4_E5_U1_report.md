# E4/E5 U1 execution report — interpretability and momentum overlay

Date: 2026-08-14
Universe: U1 MSCI US
Configurations: `baseline`, `M1_delta_0.02`
Output roots: `$CLUSTER_LINEAGE_OUTPUT_DIR/interpretability/msci_us/` and
`$CLUSTER_LINEAGE_OUTPUT_DIR/backtests/msci_us/`

## Gate conventions applied

- U1 best in-band configuration is frozen as `M1_delta_0.02`.
- The headline window is 2009-08-31 through 2026-06-30; the full panel is reported
  separately and never pooled with it.
- Ranking yardsticks are only the global cross-sectional rank and GICS-sector taxonomy
  rank. EW-all is reported only as the alpha-profile base and beta/alpha market reference.
- The primary overlay uses quintiles and 10 bp costs. The `q=1/3` and volatility-adjusted
  variants are robustness rows.
- Metric 11 retains its frozen definition. Its signed residual is the trade-interaction
  term; the former 10% residual guard is retired and is not an acceptance criterion.

## Acceptance lines

| Acceptance line | Measured | Tolerance | Status |
|---|---:|---:|---|
| E4 required universe/config runs | 6 across all universes; U1 baseline and M1_0.02 present | 6; required U1 pair present | PASS |
| Futures adopted equity-beta cuts | 0.15 / 0.60 | exactly 0.15 / 0.60 | PASS |
| Interpretability taxonomy coverage | 3 cases per universe at or above 0.70 | at least 3 per universe | PASS |
| U1 payoff windows | headline and full panel, separately labelled | exactly 2; never pooled | PASS |
| U1 comparison yardsticks | global and taxonomy | exactly 2; EW excluded | PASS |
| EW role | reference-only block | no EW ranking contrast | PASS |
| U1 Metric-11 handling | signed interaction reported; guard retired | no residual-guard failure | PASS |
| Exact validator | `E4: PASS`; `E5 msci_us: PASS evidence` | both PASS | PASS |

The futures cut mapping is explicit in
`interpretability/futures_equity_beta_threshold_adopted.csv`: the empirical q25/q75 evidence
is approximately 0.0000/0.6687, while the adopted boundaries are the owner-approved round
numbers 0.15/0.60. The raw percentiles are not used as bucket cuts.

## U1 clarified payoff comparison

Headline primary rows:

| Ranking leg | Net annual return | Sharpe | Annual turnover | Cost drag bp/year | Return vs global | Return vs taxonomy | Sharpe vs global | Sharpe vs taxonomy |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| global | 5.8203% | 0.4405 | 2.7268 | 57.23 | — | — | — | — |
| taxonomy | 10.9829% | 0.7649 | 2.7090 | 59.66 | — | — | — | — |
| cluster baseline | 6.7838% | 0.5322 | 3.4894 | 73.99 | +0.9634% | -4.1991% | +0.0917 | -0.2327 |
| cluster M1_0.02 | 6.5015% | 0.5117 | 3.0095 | 63.61 | +0.6812% | -4.4813% | +0.0712 | -0.2532 |

Full-panel cluster M1_0.02 deltas are +1.9421% return and +0.1133 Sharpe versus global,
and -5.0695% return and -0.2647 Sharpe versus taxonomy. These are ranking-leg comparisons;
no performance conclusion is drawn against EW-all.

The separate EW reference block reports headline EW-all NAV statistics of 8.2077% annual
return, 15.1598% volatility and 0.5990 Sharpe solely to support the alpha/beta columns.

## Metric 11

For U1 M1_0.02, the signed trade-interaction term is -0.05752 in the headline window and
-0.06704 on the full panel. Absolute shares are 0.1939 and 0.2196, respectively, and are
reported descriptively under `RETIRED_NOT_AN_ACCEPTANCE_CRITERION`; no rerun or metric
redefinition was performed.

## Artifacts

- `backtests/msci_us/payoff_comparison.csv`: clarified ranking-leg comparison table.
- `backtests/msci_us/ew_reference.csv`: separate EW reference block.
- `backtests/msci_us/robustness.csv`: q=1/3 and volatility-adjusted rows by window.
- `backtests/msci_us/turnover_decomposition.csv`: frozen Metric-11 decomposition.
- `interpretability/msci_us/metric_set_12.csv`: baseline/M1_0.02 interpretability metrics.

No Git staging or push was performed.
