# E8b execution report — production-signal backtests and inference

Date: 2026-08-14
Runners: `run_e8b.py`, `build_e8_pdfs.py`, `validate_e8.py`
Output: `$CLUSTER_LINEAGE_OUTPUT_DIR/e8/u3m/e8b/`

**QE-frequency funds are EXCLUDED from the cluster-momentum arm.** The sentence appears
verbatim in every final CSV and inside every profile PDF. The 17 QE tickers have zero
intersection with emitted portfolio weights.

## Acceptance

| Acceptance line | Measured | Tolerance | Status |
|---|---:|---:|---|
| Required signal × ranking × variant profiles | 24 | 24 | PASS |
| S_prod Sub-Asset-Class diagnostic profiles | 2 | 2 | PASS |
| Sampled raw-score max difference across ranking legs | 0.0 | 0.0 | PASS |
| Global and taxonomy yardsticks present | 3/3 signals | 3/3 | PASS |
| EW reference blocks | 2 | 2 | PASS |
| Metric-11 maximum identity error | 5.55e-17 | ≤1e-14 | PASS |
| Inference rows | 8 | 8 | PASS |
| Bootstrap parameters | block 6 / 2,000 / 20260813 | frozen | PASS |
| QE assets in emitted weights | 0 | 0 | PASS |
| Policy-compliant CSVs | 13/13 | 100% | PASS |
| Deterministic final CSV/PDF artifacts | 16/16 | 100% | PASS |
| PDFs text-checked and visually rendered | 3/3 | 3/3 | PASS |
| Rosaa imports in replication AST scan | 0 | 0 | PASS |

The accepted U3 E5 convention has one labelled `full_panel` analysis window; E8 preserves
that convention rather than inventing a second date window. Both q=0.20 primary and q=1/3
headline-robustness portfolio variants are emitted and never pooled.

`S_prod` reproduces `MOMENTUM_CLUSTER` without importing rosaa: ME long span 12, volatility
span 13, no short leg, `MeanAdjType.NONE`, benchmark-relative risk adjustment, global
cross-sectional winsor/z-score behavior, and the `min_cluster_size=5` Global fallback.
The file-level provenance is in `configs.py`.

## Headline q=1/3 inference

| Contrast | Metric | Estimate | 95% CI | Excludes zero |
|---|---|---:|---:|---|
| S_prod − S_raw, best cluster | net Sharpe | +0.0535 | [-0.0308, +0.1267] | No |
| S_prod − S_raw, best cluster | annualized turnover | +0.4281 | [+0.2996, +0.5590] | Yes |
| S_raw best cluster − taxonomy | net Sharpe | -0.0430 | [-0.1102, +0.0344] | No |
| S_voladj best cluster − taxonomy | net Sharpe | -0.0202 | [-0.0930, +0.0554] | No |
| S_prod best cluster − taxonomy | net Sharpe | -0.0744 | [-0.1556, +0.0152] | No |
| S_prod best cluster − taxonomy | annualized turnover | +0.2954 | [+0.1671, +0.4268] | Yes |

## Gate interpretation

1. The production operating point does **not close the taxonomy gap** on U3M. Under
   `S_prod`, the best smoothed cluster leg has Sharpe 0.5115 versus 0.5859 for the Asset
   Class taxonomy leg; the -0.0744 delta is not statistically distinguishable from zero.
2. `S_prod` improves best-cluster Sharpe over `S_raw` by +0.0535, but the CI crosses zero.
   It also raises turnover by +0.4281 with a strictly positive CI.
3. The Sub-Asset-Class diagnostic is weaker still (Sharpe 0.4340).

EW-all is reported only as the base/reference NAV and the market benchmark for alpha/beta.
No performance conclusion compares a ranking leg with EW-all.

## OWNER GATE E8 request

Rule on (i) the taxonomy-gap result, (ii) the `S_prod` versus `S_raw` trade-off, and
(iii) whether to dispatch the U1 production-scale granularity confirmation as E9.
