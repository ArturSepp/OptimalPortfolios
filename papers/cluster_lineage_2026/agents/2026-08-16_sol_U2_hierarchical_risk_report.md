# U2 hierarchical-risk execution report

**Date:** 2026-08-16  
**Status:** COMPLETE — all runner and independent-validation checks pass  
**Scope:** Owner-frozen U2 BlackRock fund universe; standard well-known risk-allocation
methods and accepted long-short momentum construction. Ward-HERC and nested cluster risk
budgeting are excluded.

## Outcome and paper recommendation

The U2 result is asymmetric and useful:

1. **The Rolling-Ward cluster-short overlay beats the matched global long-short rank.** Net
   annual return improves by **96.33 bp**, realised volatility falls by **84.59 bp**, and net
   Sharpe improves by **0.127**. The overlay also lowers mean ex-ante volatility by **9.25%**
   and Ward-cluster net-exposure L1 by **19.31%**.
2. **Rolling-Ward HRP does not work as an unconstrained long-only allocation across this fund
   catalogue.** It returns -0.169% with 0.755% volatility and a -0.220 Sharpe, versus +0.936%,
   1.605%, and 0.589 for flat ERC. Canonical single-link HRP is also negative.
3. The long-only failure has a concrete mechanism. Rolling-Ward HRP allocates **98.91%** to
   Fixed Income and **0.15%** to Equity on average, with only **2.54 effective assets**. The
   inverse-variance recursive allocation compounds the attraction of near-duplicate,
   extremely low-volatility fixed-income fund blocks. Single-link HRP behaves similarly.
4. Cluster risk budgeting remains a valid mechanism diagnostic: moving from flat asset risk
   budgets through sqrt-size to equal-cluster budgets monotonically reduces ex-ante volatility
   and cluster-risk concentration. It also pushes capital further toward Fixed Income, so it
   is not a performance improvement.

For the paper, use the **U2 cluster-short overlay versus global rank** as the positive result.
Do not claim that Rolling-Ward HRP improves U2 long-only performance. The long-only table is a
useful robustness/limitation result: standard unconstrained HRP is not appropriate for a
heterogeneous fund catalogue without a separately imposed strategic sleeve budget. Adding
such a sleeve hierarchy would be a different, nested design and remains out of scope.

## Paper terminology and consolidated comparison

The snapshot method is **rolling EWMA-Ward correlation clustering**, shortened to
**Rolling-Ward**. The long-only application is **Rolling-Ward HRP**. The accepted U2 signal
application is more precisely **Rolling-Ward cluster-short overlay**, because its long side is
the global rank and its short side is cluster-relative. MCF remains the separate lineage
tracker and is not part of the clustering name.

| Panel | Paper label | Net return | Volatility | RF=0 Sharpe | One-way turnover |
|---|---|---:|---:|---:|---:|
| Long-only | Flat ERC | 0.936% | 1.605% | 0.589 | 0.357x |
| Long-only | Canonical HRP (single linkage) | -0.083% | 0.533% | -0.153 | 0.298x |
| Long-only | **Rolling-Ward HRP** | **-0.169%** | **0.755%** | **-0.220** | 0.522x |
| Long-short | Global momentum rank | -0.771% | 7.561% | -0.065 | 2.243x |
| Long-short | **Rolling-Ward cluster-short overlay** | **0.193%** | **6.716%** | **0.062** | 2.468x |

All rows cover 2009-08-31 through 2026-06-30 and are net of 20 bp one-way costs. No row is
compared with EW-all. EW-all is used only as the market reference for the separately reported
alpha/beta columns.

## Frozen design

- Universe: BlackRock funds passing the strict point-in-time rule that the latest 12 completed
  calendar month-end average Bloomberg AUM is greater than USD 100m.
- Breadth at the 102 rebalance dates: 118 minimum, 227 median, 370 final.
- Window/schedule: 203 monthly estimation dates from 2009-08-31 through 2026-06-30; the
  accepted every-two-month schedule produces 102 allocation/rebalance dates.
- Raw hierarchy: W-THU returns, EWMA span 156, demeaned Pearson correlation, `1-rho`
  distance, Ward linkage, cutoff 0.60.
- Common allocation covariance: the causal EWMA covariance immediately before FactorLasso
  correlation normalization, annualised by 52, with a common minimal eigenvalue floor that
  caps condition number at 1e6.
- Portfolio mechanics: long-only fully invested for allocation methods; one W-WED
  implementation-period lag; 20 bp one-way costs.
- Long-short construction: ROSAA production risk-adjusted momentum, q=0.25, Equity/Fixed
  Income/Rest gross sleeve budgets 50%/30%/20% per side, global-rank long and group-equal
  cluster-relative short.
- Benchmarks: flat ERC for long-only allocation; matched global rank for long-short momentum.
  EW-all is not a performance yardstick.

## Covariance reconstruction and conditioning

FactorLasso exposes the normalized rolling correlation used by clustering but not its
intermediate covariance. The runner therefore reuses FactorLasso's own point-in-time
demeaning/masking and EWMA update functions and asserts the normalization identity on every
rebalance date.

The raw covariance is positive definite but becomes severely ill-conditioned when
near-duplicate fund histories enter: the maximum raw condition number is **5.382e9**. This
caused the production CCD solver to miss equal-cluster risk targets despite valid weights.
The common allocation matrix is therefore conditioned with
`optimalportfolios.optimization.covar_factorization.factorize_covariance`:

- condition-number cap: **1.0e6** on every date;
- eigenvalues floored: 22.70 mean, 71 maximum;
- relative Frobenius adjustment: **2.57e-6 mean, 5.93e-6 maximum**;
- maximum risk-budget target error after conditioning: **1.12e-5**, tolerance 2e-5.

This is numerical conditioning, not a cluster or payoff parameter. It is applied identically
to all five long-only methods. Cluster discovery continues to use the raw correlation: the
reconstructed raw partition and Ward linkage match the accepted cache with exactly zero error
on 100% of dates.

## Long-only performance

| Method | Net return | Realised vol | Net Sharpe | One-way turnover | Cost drag/year |
|---|---:|---:|---:|---:|---:|
| Flat ERC | **0.936%** | 1.605% | **0.589** | 0.357x | 14.38 bp |
| Cluster RB sqrt(n) | 0.484% | 1.295% | 0.379 | 0.385x | 15.48 bp |
| Cluster RB equal | 0.210% | 1.119% | 0.193 | 0.441x | 17.69 bp |
| Rolling-Ward HRP | -0.169% | 0.755% | -0.220 | 0.522x | 20.89 bp |
| Canonical single-HRP | -0.083% | **0.533%** | -0.153 | **0.298x** | 11.90 bp |

Rolling-Ward HRP versus flat ERC: -110.5 bp annual net return, -84.9 bp realised volatility,
-0.809 Sharpe, and +0.166x one-way turnover. Versus canonical single-HRP it is also worse on
return, volatility, Sharpe, and turnover. These results reject Rolling-Ward HRP as the U2
long-only headline method.

## Ex-ante risk through Rolling-Ward clusters

| Method | Mean ex-ante vol | Effective risk clusters | Largest absolute cluster-risk share | Diversification ratio |
|---|---:|---:|---:|---:|
| Flat ERC | 1.390% | 5.93 | 32.49% | 1.966 |
| Cluster RB sqrt(n) | 0.873% | 10.12 | 19.46% | 2.164 |
| **Cluster RB equal** | **0.571%** | **13.98** | **8.22%** | **2.358** |
| Rolling-Ward HRP | 0.243% | 2.14 | 64.75% | 1.946 |
| Canonical single-HRP | 0.234% | 2.17 | 64.84% | 2.015 |

The cluster-risk-budget sequence behaves as designed and is the transparent mechanism check.
HRP's very low volatility is achieved by concentrating, not by distributing risk across the
discovered clusters.

## Broad-sleeve capital diagnostic

| Method | Equity mean | Fixed Income mean | Rest mean | Effective assets mean |
|---|---:|---:|---:|---:|
| Flat ERC | 7.52% | 91.18% | 1.30% | 4.82 |
| Cluster RB sqrt(n) | 4.31% | 94.40% | 1.29% | 3.96 |
| Cluster RB equal | 2.39% | 96.41% | 1.19% | 3.23 |
| Rolling-Ward HRP | **0.15%** | **98.91%** | 0.94% | **2.54** |
| Canonical single-HRP | 0.17% | 99.07% | 0.76% | 2.43 |

This diagnostic explains the negative excess returns of both HRP variants. It also shows why
their lower realised volatility is not evidence of superior diversification.

## Accepted long-short signal risk

| Diagnostic | Global rank | Rolling-Ward overlay | Overlay change |
|---|---:|---:|---:|
| Mean ex-ante volatility | 6.12% | **5.56%** | **-9.25%** |
| Effective Ward risk clusters | 4.46 | **4.87** | **+9.23%** |
| Largest absolute cluster-risk share | 40.05% | **38.76%** | **-3.20%** |
| Ward-cluster net-exposure L1 | 1.220 | **0.984** | **-19.31%** |

Unlike U1's pure cluster-relative long/short construction, U2 uses a global long side and only
the short side is cluster-relative. Therefore Ward-cluster net exposure is reduced rather
than eliminated. The lower risk is accompanied by the already accepted payoff improvement:
+96.33 bp net return, -84.59 bp realised volatility, and +0.127 Sharpe versus global rank.

## Acceptance — primary runner

All **20/20** lines pass:

| Check | Measured | Tolerance | Status |
|---|---:|---:|---|
| Scheduled allocation dates | 102 | 102 | PASS |
| Covariance/Ward asset-set match share | 1.000 | 1.000 | PASS |
| Reconstructed Ward partition match share | 1.000 | 1.000 | PASS |
| Frozen AUM100 partition match share | 1.000 | 1.000 | PASS |
| FactorLasso covariance/correlation max error | 0 | <=1e-12 | PASS |
| Frozen Ward linkage max error | 0 | <=1e-12 | PASS |
| Maximum conditioned covariance condition | 1.000e6 | <=1.000e6 | PASS |
| Alpha=1 versus flat ERC max weight error | 7.77e-16 | <=5e-10 | PASS |
| Maximum allocation weight-sum error | 8.52e-11 | <=5e-10 | PASS |
| Minimum hierarchical allocation weight | 1.65e-7 | >=0 | PASS |
| Weight outside point-in-time eligibility | 0 | <=5e-10 | PASS |
| AUM <= USD100m eligible observations | 0 | 0 | PASS |
| Maximum Euler risk reconciliation error | 5.55e-16 | <=5e-10 | PASS |
| Maximum risk-budget target error | 1.12e-5 | <=2e-5 | PASS |
| Risk rows | 510 | 510 | PASS |
| Maximum signal look-ahead days | 0 | <=0 | PASS |
| Maximum signal-weight construction error | 7.99e-15 | <=5e-10 | PASS |
| One-way transaction cost | 20 bp | 20 bp | PASS |
| Paper comparison rows | 5 | 5 | PASS |
| Ward-HERC paper rows | 0 | 0 | PASS |

Deterministic replay is **22/22 byte-identical CSV artifacts**.

## Independent validation

The separate validator reads the persisted files and does not call the experiment runner. It
reconstructs the EWMA covariance in one batch on the first, middle, and final dates, reapplies
the frozen conditioning rule, resolves flat ERC and equal-cluster risk budgeting directly,
and independently reclusters the raw covariance correlation.

All **15/15** checks pass, including:

- sampled raw-covariance Ward partition match share: 1.000;
- sampled flat-ERC solver error: 0, tolerance 2e-8;
- sampled equal-cluster solver error: 0, tolerance 2e-8;
- persisted Euler contribution reconciliation: 2.00e-15, tolerance 5e-10;
- paper long-short source-table error: 1.33e-15, tolerance 5e-10;
- EW-all performance comparisons: 0;
- Ward-HERC paper rows: 0.

The focused test suite passes **10/10**. The defect-first proof was observed before the U2
runner existed: the new test initially failed collection with `ModuleNotFoundError`; after
implementation, it additionally detected and pinned FactorLasso's exact floating diagonal
normalization.

## Runtime and footprint

- Final cold 102-date allocation-cache build: approximately 109 seconds by cache timestamp
  span; individual solves ranged from about 0.6 to 2.0 seconds.
- Allocation cache: 102 pickles, 2.43 MiB total.
- Complete cache-first run including five backtests, signal-risk decomposition, tables, and
  three exhibits: 27.2 seconds.

## Deliverables

Runner and validation code:

- `papers/cluster_lineage_2026/replication/run_u2_hierarchical_risk.py`
- `papers/cluster_lineage_2026/replication/validate_u2_hierarchical_risk.py`
- `papers/cluster_lineage_2026/replication/u2_hierarchical_risk_test.py`
- shared `papers/cluster_lineage_2026/replication/hierarchical_risk_allocations.py`

External artifact/cache root:

- `C:/Users/artur/OneDrive/analytics/outputs/cluster_lineage_2026/risk_allocation/u2_hierarchical_20260816/`

Primary tables:

- `paper_comparison.csv`
- `performance.csv`
- `risk_summary.csv`
- `signal_risk_summary.csv`
- `allocation_sleeve_summary.csv`
- `acceptance.csv`
- `independent_validation.csv`
- `determinism.csv`

Paper-ready exhibits:

- `u2_allocation_performance.PNG`
- `u2_allocation_risk_structure.PNG`
- `u2_signal_risk_structure.PNG`

All three exhibits were opened and visually checked after generation.

## Verification commands

```powershell
python -m pytest `
  papers/cluster_lineage_2026/replication/u2_hierarchical_risk_test.py `
  papers/cluster_lineage_2026/replication/hierarchical_risk_allocations_test.py -q
# .......... [100%]

ruff check --isolated --select E,F,W --line-length 100 `
  papers/cluster_lineage_2026/replication/run_u2_hierarchical_risk.py `
  papers/cluster_lineage_2026/replication/validate_u2_hierarchical_risk.py `
  papers/cluster_lineage_2026/replication/u2_hierarchical_risk_test.py
# All checks passed!

python -m papers.cluster_lineage_2026.replication.run_u2_hierarchical_risk
# U2 hierarchical risk allocation: PASS (22/22 deterministic)

python -m papers.cluster_lineage_2026.replication.validate_u2_hierarchical_risk
# 15/15 PASS
```

## Deviations and boundaries

- No cluster, momentum, AUM, sleeve-budget, cost, or holding-period parameter was reselected.
- The condition-number cap is a numerical allocation guard introduced after diagnosing the
  production risk-budget solver on near-duplicate fund histories. It is applied identically
  across allocation methods and changes no raw Ward partition or accepted long-short payoff.
- Ward-HERC, nested cluster allocation, new factor-model fitting, and statistical inference
  were not added.
- No git staging or push was performed.
