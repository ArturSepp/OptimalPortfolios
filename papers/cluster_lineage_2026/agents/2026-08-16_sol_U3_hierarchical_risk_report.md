# U3 hierarchical-risk execution report

**Date:** 2026-08-16  
**Status:** COMPLETE — all runner, replay, test, lint, and independent-validation checks pass  
**Scope:** Owner-frozen futures universe; standard long-only risk-allocation methods and the
accepted futures long-short momentum construction. Ward-HERC and nested cluster risk
budgeting are excluded.

## Outcome and paper recommendation

U3 produces three distinct findings:

1. **The transparent equal-cluster risk-budget allocation is the strongest long-only
   result.** It earns **1.229%** net with **1.816%** volatility and a **0.672** RF=0 Sharpe,
   versus 0.836%, 1.671%, and 0.497 for flat ERC. It raises Sharpe by **0.175** while reducing
   the largest M1-star cluster-risk share from 21.67% to **6.34%** and increasing effective
   risk clusters from 9.43 to **16.21**. This is a flat production risk-budget solve with
   equal group budgets and equal within-group asset budgets, not nested cluster allocation.
2. **Rolling-Ward HRP modestly dominates canonical single-link HRP but not flat ERC.** It
   has higher return (+1.58 bp), lower volatility (-0.90 bp), higher Sharpe (+0.0168), and
   lower turnover (-0.164x) than single-HRP. Against flat ERC it cuts volatility from 1.671%
   to **0.933%**, but net return falls from 0.836% to **0.026%**.
3. **The owner-frozen cluster-relative long-short book remains the stronger risk-isolation
   construction.** Versus matched global rank, net return improves by **4.60 bp**, volatility
   falls by **379.29 bp**, and one-way turnover falls by **0.408x**. Its M1-star cluster net
   exposure is zero to numerical precision. Sharpe is lower by 0.0117 because both net
   returns are economically near zero.

For the paper, use equal-cluster risk budgeting as the positive U3 long-only risk result and
the cluster-relative momentum construction as the signal-risk result. Rolling-Ward HRP is a
useful canonical-method comparison: it improves on the literature's single-link HRP control,
but its recursive inverse-variance allocation is too defensive to beat flat ERC on return or
Sharpe.

## Paper terminology and consolidated comparison

The clustering configuration is **M1-star Rolling-Ward**: W-WED returns, EWMA span 156,
Ward linkage, and the owner-confirmed smoother delta 0.0691. The long-only hierarchical
application is **M1-star Rolling-Ward HRP**. The signal application is **M1-star Rolling-Ward
cluster-relative momentum**. MCF remains the separate lineage tracker.

| Panel | Paper label | Net return | Volatility | RF=0 Sharpe | One-way turnover |
|---|---|---:|---:|---:|---:|
| Long-only | Flat ERC | 0.836% | 1.671% | 0.497 | 0.237x |
| Long-only | Canonical HRP (single linkage) | 0.010% | 0.942% | 0.021 | 0.412x |
| Long-only | **M1-star Rolling-Ward HRP** | **0.026%** | **0.933%** | **0.038** | **0.248x** |
| Long-short | Global momentum rank | -0.016% | 8.213% | 0.030 | 3.441x |
| Long-short | **M1-star cluster-relative momentum** | **0.030%** | **4.420%** | 0.018 | **3.033x** |

All rows cover 2009-08-31 through 2026-06-30 and are net of 10 bp one-way costs. The
long-only benchmark is flat ERC. The long-short benchmark is the matched global rank. No row
compares performance with EW-all.

## Frozen design

- Universe: point-in-time futures panel less the seven owner-frozen low-liquidity contracts:
  `BMR1 Curncy`, `CUA1 Comdty`, `IJ1 Comdty`, `KC1 Comdty`, `KM1 Index`, `MES1 Index`, and
  `RS1 Comdty`. Breadth is 83 assets initially and 88 at the end.
- Window: 203 monthly decision dates from 2009-08-31 through 2026-06-30.
- Dependence model: W-WED returns, EWMA span 156, M1-star smoother delta 0.0691, Ward linkage.
- Allocation covariance: owner-accepted M1-star HCGL covariance from the frozen E2b cache,
  restricted to the investable leaves and annualised by 52. No numerical conditioning was
  required; the minimum eligible covariance eigenvalue is 1.294e-4.
- Long-only methods: flat ERC, cluster-risk-budget sqrt-size, cluster-risk-budget equal,
  Rolling-Ward HRP, and canonical single-link HRP. All are long-only and fully invested.
- Long-short construction: q=0.25 ROSAA production risk-adjusted momentum, 12-month long
  span, no reversal span, volatility span 13, EWMA mean adjustment, and group fallback 5.
  Equity/Fixed Income/Commodities/FX budgets are 30%/30%/30%/10% on each side.
- Portfolio mechanics: one implementation period of lag and 10 bp one-way costs.

## Eligibility and hierarchy distinction

The accepted M1-star signal memberships were frozen before the seven liquidity exclusions;
the exclusions were then applied to eligibility and weights without performance-driven
refitting. Removing leaves changes a dendrogram cut, so a fresh eligible-only cut reproduces
the frozen memberships on **11.33%** of headline dates.

This is handled explicitly:

- the accepted long-short weights and all M1-star group-risk attribution preserve the frozen
  memberships after restricting them to eligible instruments;
- HRP requires a valid binary tree on exactly the investable leaves, so it uses the cached
  eligible-universe Ward tree from the same M1-star model and dates;
- asset sets and leaf order match exactly on 100% of dates.

The 11.33% value is a disclosed construction diagnostic, not a failed acceptance line and not
a silent redefinition of the frozen signal strategy.

## Full long-only performance

| Method | Net return | Realised vol | Net Sharpe | One-way turnover | Cost drag/year |
|---|---:|---:|---:|---:|---:|
| Flat ERC | 0.836% | 1.671% | 0.497 | **0.237x** | 4.77 bp |
| Cluster RB sqrt(n) | 0.988% | 1.702% | 0.577 | 0.283x | 5.71 bp |
| **Cluster RB equal** | **1.229%** | 1.816% | **0.672** | 0.358x | 7.25 bp |
| M1-star Rolling-Ward HRP | 0.026% | **0.933%** | 0.038 | 0.248x | 4.97 bp |
| Canonical single-HRP | 0.010% | 0.942% | 0.021 | 0.412x | 8.25 bp |

Equal-cluster risk budgeting versus flat ERC adds 39.27 bp of net return, 14.51 bp of
volatility, 0.175 of Sharpe, and 0.121x turnover. It is the only long-only cluster-aware row
that clearly improves the return/volatility trade-off over flat ERC.

## Ex-ante risk through M1-star groups

| Method | Mean ex-ante vol | Effective risk clusters | Largest absolute cluster-risk share | Diversification ratio |
|---|---:|---:|---:|---:|
| Flat ERC | 6.04% | 9.43 | 21.67% | 4.85 |
| Cluster RB sqrt(n) | 6.55% | 13.83 | 12.64% | **4.92** |
| **Cluster RB equal** | 7.56% | **16.21** | **6.34%** | 4.82 |
| M1-star Rolling-Ward HRP | **2.36%** | 1.96 | 61.43% | 3.26 |
| Canonical single-HRP | 2.38% | 1.99 | 60.63% | 3.28 |

The risk-budget progression works exactly as intended: moving from flat asset budgets to
equal cluster budgets spreads risk across the discovered groups. HRP reaches low volatility
through concentration rather than broad cluster-risk diversification.

## Asset-class capital diagnostic

| Method | STIR | Bonds | Equities | FX | Agriculture | Energy + Metals |
|---|---:|---:|---:|---:|---:|---:|
| Flat ERC | 47.65% | 38.62% | 3.53% | 4.84% | 4.08% | 1.28% |
| Cluster RB sqrt(n) | 47.33% | 37.08% | 3.23% | 4.92% | 5.78% | 1.66% |
| Cluster RB equal | 46.64% | 35.24% | 3.02% | 4.91% | 8.08% | 2.11% |
| M1-star Rolling-Ward HRP | **74.79%** | 24.36% | 0.15% | 0.48% | 0.16% | 0.05% |
| Canonical single-HRP | 74.89% | 24.16% | 0.13% | 0.51% | 0.24% | 0.07% |

Both HRP variants overwhelmingly select STIR and bond contracts. Rolling-Ward HRP averages
only 5.07 effective assets, versus 12.55 for flat ERC and 13.47 for equal-cluster risk
budgeting. This explains its low volatility and near-zero return.

## Frozen long-short signal risk

| Diagnostic | Global rank | M1-star cluster rank | Cluster change |
|---|---:|---:|---:|
| Mean ex-ante volatility | 37.72% | **33.52%** | **-11.13%** |
| Effective absolute-risk clusters | 6.77 | **6.97** | **+2.97%** |
| Largest absolute cluster-risk share | 29.02% | **28.64%** | **-1.34%** |
| M1-star cluster net-exposure L1 | 1.040 | **0.000** | **-100.00%** |
| Largest cluster net exposure | 19.18% | **0.000%** | **-100.00%** |

Within-cluster long and short budgets eliminate cluster net exposure to 3.05e-17 L1. That
is the direct risk-isolation mechanism: the strategy is not relying on directional exposure
to the discovered futures groups. It also reduces realised volatility and turnover, although
the two net returns are too small for the lower volatility to improve Sharpe in this sample.

## Acceptance — primary runner

All **20/20** lines pass:

| Check | Measured | Tolerance | Status |
|---|---:|---:|---|
| Headline allocation dates | 203 | 203 | PASS |
| Exact eligible asset-set share | 1.000 | 1.000 | PASS |
| Covariance contains eligible assets share | 1.000 | 1.000 | PASS |
| Ward leaf/membership order match share | 1.000 | 1.000 | PASS |
| Minimum covariance eigenvalue | 1.294e-4 | >0 | PASS |
| Alpha=1 versus flat ERC max weight error | 1.39e-16 | <=5e-10 | PASS |
| Maximum allocation weight-sum error | 1.34e-11 | <=5e-10 | PASS |
| Minimum hierarchical allocation weight | 1.12e-7 | >=0 | PASS |
| Weight outside point-in-time eligibility | 0 | <=5e-10 | PASS |
| Maximum Euler risk reconciliation error | 6.66e-16 | <=5e-10 | PASS |
| Maximum risk-budget target error | 1.69e-6 | <=2e-5 | PASS |
| Risk rows | 1,015 | 1,015 | PASS |
| Maximum signal look-ahead days | 0 | <=0 | PASS |
| Maximum signal construction error | 2.66e-15 | <=1e-12 | PASS |
| Maximum owner-excluded signal weight | 0 | 0 | PASS |
| Frozen signal performance error | 2.66e-15 | <=1e-12 | PASS |
| One-way transaction cost | 10 bp | 10 bp | PASS |
| Paper comparison rows | 5 | 5 | PASS |
| Ward-HERC paper rows | 0 | 0 | PASS |
| EW-all ranking-performance comparison rows | 0 | 0 | PASS |

Deterministic replay is **22/22 byte-identical CSV artifacts**.

## Independent validation

The separate validator reads the persisted artifacts without invoking the runner. On the
first, middle, and final dates it independently resolves flat ERC and equal-cluster risk
budgeting and separately implements HRP recursive bisection.

All **17/17** checks pass, including:

- sampled flat-ERC solver error: 0, tolerance 2e-8;
- sampled equal-cluster solver error: 0, tolerance 2e-8;
- sampled independent Ward-HRP recursion error: 2.22e-16, tolerance 5e-10;
- Euler cluster-risk contribution sum error: 1.67e-15, tolerance 5e-10;
- frozen long-short source-table error: 0, tolerance 1e-12;
- owner-excluded eligible observations: 0;
- EW-all performance yardstick rows: 0;
- Ward-HERC paper rows: 0.

The focused shared/U2/U3 allocation test set passes **13/13** and isolated E/F/W lint reports
`All checks passed!`. The defect-first proof was observed before implementation: collection
failed with `ModuleNotFoundError: ...run_u3_hierarchical_risk`; the same test is now green.

## Runtime, footprint, and visual verification

- Cold allocation cache: 203 pickles, 3,088,411 bytes (2.95 MiB), built in approximately
  68 seconds from the first to final cache timestamp.
- Complete cache-first run including five long-only backtests, exact accepted long-short
  reconstruction, risk diagnostics, tables, and four exhibits: **17.3 seconds**.
- All four PNG exhibits were opened after final generation. Labels, signs, units, and panel
  bounds are readable; the small signal returns use two decimal percentage labels.

## Deliverables

Code:

- `papers/cluster_lineage_2026/replication/run_u3_hierarchical_risk.py`
- `papers/cluster_lineage_2026/replication/validate_u3_hierarchical_risk.py`
- `papers/cluster_lineage_2026/replication/u3_hierarchical_risk_test.py`
- shared `papers/cluster_lineage_2026/replication/hierarchical_risk_allocations.py`

External artifact root:

- `C:/Users/artur/OneDrive/analytics/outputs/cluster_lineage_2026/risk_allocation/u3_hierarchical_20260816/`

Primary tables:

- `paper_comparison.csv`
- `performance.csv`
- `risk_summary.csv`
- `allocation_asset_class_summary.csv`
- `signal_performance.csv`
- `signal_risk_summary.csv`
- `acceptance.csv`
- `independent_validation.csv`
- `determinism.csv`

Paper-ready exhibits:

- `u3_allocation_performance.PNG`
- `u3_allocation_risk_structure.PNG`
- `u3_signal_comparison.PNG`
- `u3_signal_risk_structure.PNG`

## Verification outputs

```text
.............                                                            [100%]
All checks passed!
U3 hierarchical risk allocation: PASS (22/22 deterministic)
U3 hierarchical risk independent validation: PASS (17/17)
```

## Boundaries

- No cluster, signal, q, span, sleeve, cost, exclusion, or window parameter was reselected.
- No covariance or cluster model was refit; accepted M1-star and eligible-tree caches were
  consumed cache-first.
- Equal-cluster risk budgeting is a single flat risk-budget solve, not the excluded nested
  cluster-allocation design.
- Ward-HERC and EW-all performance comparison remain excluded.
- No git staging or push was performed.
