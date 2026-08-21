# Three-universe empirical evidence: peer-contained signal ranking and risk allocation

**Date:** 2026-08-16  
**Status:** COMPLETE — consolidated from accepted, independently validated stage outputs  
**Universes:** U1 equities, U2 BlackRock funds, U3 futures

## Empirical architecture

The evidence should be presented as two separate experiments.

### Experiment 1 — peer-contained signal ranking

- Portfolio: long-short for all three universes.
- Input: classic 12m-ex-1m momentum for U1; ROSAA risk-adjusted momentum for U2/U3.
- No covariance-based risk allocation.
- Weighting: equal notional budgets across available clusters, sectors, or fixed asset-class
  sleeves; equal weights among selected assets inside each group.
- Treatment: rank the signal inside correlation clusters.
- Controls: rank inside the full universe or the external classification.
- Question: with the signal, eligibility, and strategic budgets held fixed, does changing
  the ranking pool from broad peers to correlation-cluster peers improve the portfolio?

### Experiment 2 — risk allocation

- Portfolio: long-only only, fully invested.
- No alpha or momentum signal.
- Treatment: use the discovered Ward hierarchy or cluster membership in a standard risk
  allocator.
- Controls: flat ERC and canonical single-link HRP.
- Question: does the cluster structure improve diversification, risk concentration, or
  realised allocation efficiency relative to peer risk methods?

**No long-short allocation experiment is needed.** Long-short is used only for the signal
tests because it compares top and bottom ranks. The separate allocation experiment remains
long-only and signal-free.

## Experiment 1 — peer-contained signal ranking

### Frozen primary construction

| Universe | Signal | Cluster treatment | Ranking controls | Sleeve construction | Cost |
|---|---|---|---|---|---:|
| U1 equities | classic 12m-ex-1m | M1-star, delta 0.0866 | BICS sector and global | pure cluster/sector/global rank | 10 bp |
| U2 BlackRock funds | ROSAA risk-adjusted | W-THU/span 156, AUM > USD100m | 60/40 Equity/Fixed Income | pure cluster/global long-short rank | 20 bp |
| U3 futures | ROSAA risk-adjusted | M1-star, delta 0.0691 | 30/30/30/10 sleeve-global | pure cluster-relative rank | 10 bp |

All primary rows use q=25%, one implementation-period lag, point-in-time eligibility, and
the common 2009-08-31 through 2026-06-30 analysis window. All three signal experiments are
long-short. U2 rebalances every two months and includes only Equity and Fixed Income funds,
with 60/40 gross budgets on both signed sides. U1's classified comparison applies the same
BICS-coverage mask to cluster, sector, and global legs. U3 applies the seven frozen liquidity
exclusions. EW-all is not a ranking or payoff benchmark.

The U2 AUM rule is strict and point-in-time: the latest 12 completed month-end arithmetic
average Bloomberg AUM must be greater than USD100m. Missing AUM, an incomplete 12-month
history, and values at or below USD100m are ineligible. Commodity, Multi Asset, Digital
Assets, Real Estate, and Cash funds are removed before signal benchmarking, covariance
estimation, clustering, ranking, and backtesting. The identical mask is applied to cluster
and global legs.

### Primary performance

| Universe | Ranking leg | Net return | Volatility | RF=0 Sharpe | One-way turnover |
|---|---|---:|---:|---:|---:|
| U1 | **M1-star cluster rank** | **-2.682%** | **6.098%** | -0.415 | 2.608x |
| U1 | BICS sector rank | -3.822% | 9.363% | -0.368 | **2.499x** |
| U1 | Global rank | -4.000% | 12.771% | **-0.254** | 2.504x |
| U2 | Equity/FI-only 60/40 W-THU/156 cluster rank | -2.185% | **4.699%** | -0.446 | 2.581x |
| U2 | **Equity/FI-only 60/40 global-sleeve rank** | **-0.722%** | 6.666% | **-0.075** | **2.154x** |
| U3 | **M1-star cluster rank** | **0.030%** | **4.420%** | 0.018 | **3.033x** |
| U3 | 30/30/30/10 global sleeve rank | -0.016% | 8.213% | **0.030** | 3.441x |

The matched comparison within each universe is the relevant evidence. U2's pure cluster rank
materially lowers volatility but loses return and Sharpe to its matched global rank.

### Cluster-minus-control results

| Universe | Control | Net-return delta | Volatility delta | Sharpe delta | Turnover delta |
|---|---|---:|---:|---:|---:|
| U1 | BICS sectors | **+114.05 bp** | **-3.265 pp (-34.9%)** | -0.0466 | +0.109x |
| U1 | Global rank | **+131.84 bp** | **-6.673 pp (-52.2%)** | -0.1611 | +0.104x |
| U2 | 60/40 Equity/Fixed Income global | -146.30 bp | **-1.968 pp (-29.5%)** | -0.3708 | +0.427x |
| U3 | 30/30/30/10 global | **+4.60 bp** | **-3.793 pp (-46.2%)** | -0.0117 | **-0.408x** |

Directional scorecard across the four permitted comparisons:

- lower volatility: **4/4**;
- higher net return: **3/4**;
- lower turnover: **1/4**;
- higher Sharpe: **0/4**.

Because U1 contributes two controls, this is a comparison count rather than four independent
universe results. At the universe level, cluster ranks improve net returns in U1 and narrowly
in U3, but not in U2. Volatility is lower in all three universes.

### Plain-language interpretation

“Peer-contained signal ranking” is not a separate statistical model. It means that the same
momentum score is sorted using two different peer sets:

- global control: compare each eligible fund with all eligible funds in its Equity, Fixed
  Income, or Rest sleeve;
- cluster treatment: compare it only with funds in its correlation cluster inside that sleeve.

Everything else is held fixed. If the cluster portfolio wins, the narrower peer comparison
added value; if it loses, it did not. For U2, removing every non-Equity/non-Fixed-Income class
does not rescue the cluster book. The cluster-minus-global net gaps are -79.47 bp/year with no
AUM cutoff, -174.13 bp at USD50m, and -146.30 bp at USD100m. Cluster ranking lowers volatility
at every cutoff, but return, Sharpe, and turnover are worse. The no-AUM cell is the least-bad
cluster-relative result; the relationship with cutoff size is not monotone.

## Experiment 2 — long-only, signal-free risk allocation

### Methods

| Method | Role |
|---|---|
| Flat ERC | standard equal-asset-risk control |
| Canonical single-HRP | literature HRP control using single linkage |
| Rolling-Ward HRP | standard HRP recursion applied to the selected Ward hierarchy |
| Equal-cluster risk budgeting | one flat production risk-budget solve with equal cluster budgets and equal within-cluster asset budgets |

Equal-cluster risk budgeting is not nested allocation. Ward-HERC remains excluded.

### Performance across universes

| Universe | Method | Net return | Volatility | RF=0 Sharpe | One-way turnover |
|---|---|---:|---:|---:|---:|
| U1 | Flat ERC | 7.505% | 12.247% | 0.656 | **0.548x** |
| U1 | Equal-cluster RB | 6.033% | **11.605%** | 0.566 | 2.018x |
| U1 | **Rolling-Ward HRP** | **7.675%** | 11.852% | **0.687** | 1.329x |
| U1 | Canonical single-HRP | 7.578% | 11.969% | 0.674 | 1.146x |
| U2 | **Flat ERC** | **0.936%** | 1.605% | **0.589** | 0.357x |
| U2 | Equal-cluster RB | 0.210% | 1.119% | 0.193 | 0.441x |
| U2 | Rolling-Ward HRP | -0.169% | 0.755% | -0.220 | 0.522x |
| U2 | Canonical single-HRP | -0.083% | **0.533%** | -0.153 | **0.298x** |
| U3 | Flat ERC | 0.836% | 1.671% | 0.497 | **0.237x** |
| U3 | **Equal-cluster RB** | **1.229%** | 1.816% | **0.672** | 0.358x |
| U3 | Rolling-Ward HRP | 0.026% | **0.933%** | 0.038 | 0.248x |
| U3 | Canonical single-HRP | 0.010% | 0.942% | 0.021 | 0.412x |

Costs are 10 bp for U1/U3 and 20 bp for U2. The covariance matrix and estimation date are
held fixed across methods within each universe; only the allocation rule changes.

### Relative economic scorecard

| Fixed comparison | Higher return | Lower volatility | Higher Sharpe | Lower turnover |
|---|---:|---:|---:|---:|
| Equal-cluster RB versus flat ERC | 1/3 | 2/3 | 1/3 | 0/3 |
| Rolling-Ward HRP versus flat ERC | 1/3 | **3/3** | 1/3 | 0/3 |
| Rolling-Ward HRP versus canonical single-HRP | **2/3** | **2/3** | **2/3** | 1/3 |

There is no universal economic winner. The positive applications are:

- **U1:** Rolling-Ward HRP versus flat ERC adds 17.01 bp of net return, lowers volatility
  by 39.58 bp, and adds 0.031 Sharpe. It also beats canonical single-HRP on return,
  volatility, and Sharpe.
- **U3:** equal-cluster risk budgeting versus flat ERC adds 39.27 bp of net return and 0.175
  Sharpe, with 14.51 bp more volatility and 0.121x more turnover. Rolling-Ward HRP also
  modestly dominates single-HRP.
- **U2 limitation:** flat ERC remains superior. Both HRP variants concentrate almost all
  capital in very low-volatility fixed-income funds; distributing risk across clusters lowers
  portfolio volatility but does not preserve return.

### Cluster-risk distribution

Equal-cluster risk budgeting provides the most consistent allocation-mechanism evidence:

| Universe | Effective risk clusters: flat ERC | Equal-cluster RB | Largest cluster-risk share: flat ERC | Equal-cluster RB |
|---|---:|---:|---:|---:|
| U1 | 39.71 | **60.40 (+52%)** | 6.30% | **1.71%** |
| U2 | 5.93 | **13.98 (+136%)** | 32.49% | **8.22%** |
| U3 | 9.43 | **16.21 (+72%)** | 21.67% | **6.34%** |

The largest cluster-risk share falls by 4.59, 24.27, and 15.33 percentage points in U1, U2,
and U3 respectively. This proves that the allocation tool does what it is designed to do:
control risk concentration across discovered economic groups. It does not prove that this
mechanical diversification always raises realised Sharpe; the performance table shows that
it does so only in U3.

### Risk-allocation interpretation

The defensible empirical statement is:

> The cluster structure is an effective control layer for distributing portfolio risk across
> endogenous peer groups. Applied through standard HRP or a flat equal-cluster risk-budget
> solve, it can improve conventional allocation efficiency, but the payoff benefit depends on
> universe composition; low-volatility-dominated fund and futures trees can make unconstrained
> HRP excessively defensive.

The paper should therefore report U1 and U3 as positive applications and U2 as an informative
limitation, rather than choosing a different allocator ex post for each universe and claiming
three wins.

## Recommended empirical claims

1. **Primary signal claim:** peer-contained ranking improves net return and reduces volatility
   in the U1 and U3 long-short experiments, but the effect is not universal.
2. **U2 limitation:** after restricting the investable universe to Equity and Fixed Income
   and fixing 60/40 signed budgets, the pure cluster rank still loses to global at all three
   AUM cutoffs. The AUM50 cutoff is worst and no cutoff is least bad.
3. **Primary allocation claim:** equal-cluster risk budgets consistently reduce cluster-risk
   concentration across all three universes.
4. **Qualified efficiency claim:** Rolling-Ward HRP improves on canonical single-HRP in U1
   and U3, while cluster-aware allocation does not universally beat flat ERC.
5. **Design boundary:** long-short belongs to the signal-ranking experiment and must not be
   confused with the separate long-only, signal-free risk-allocation experiment.

## Reproducible consolidated outputs

Aggregation runner:

- `papers/cluster_lineage_2026/replication/summarize_three_universe_empirical_evidence.py`

External output directory:

- `C:/Users/artur/OneDrive/analytics/outputs/cluster_lineage_2026/evidence_summary/three_universe_two_role_20260816/`

Primary artifacts:

- `design.csv`
- `signal_isolation_performance.csv`
- `signal_isolation_comparison.csv`
- `allocation_performance_and_risk.csv`
- `allocation_comparison.csv`
- `evidence_scorecard.csv`
- `acceptance.csv`
- `determinism.csv`

Acceptance is **9/9 PASS** and deterministic replay is **7/7 byte-identical**. The U2
Equity/Fixed-Income 60/40 runner separately reports **9/9 acceptance PASS** and **8/8
deterministic artifacts** after fitting the class-restricted partitions. Isolated E/F/W lint
reports `All checks passed!`. Nothing was staged or pushed.
