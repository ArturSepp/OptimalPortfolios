# U1 ME/36 q=0.25 long-short curiosity report

**Date:** 2026-08-15  
**Status:** COMPLETE  
**Runner:** `papers/cluster_lineage_2026/replication/run_u1_me36_long_short.py`  
**Validator:** `papers/cluster_lineage_2026/replication/validate_u1_me36_long_short.py`

## Outcome

The matched dollar-neutral momentum test is negative for both ranking methods. In the U1
headline window, the ME/span-36 cluster spread returns -2.4406% per year net with a
-0.324483 Sharpe, while the global spread returns -4.0009% with a -0.251495 Sharpe.
Clustering therefore improves annual net return by 1.5603 percentage points, but not Sharpe:
its volatility is much lower and the negative return produces a 0.072988 more-negative
Sharpe.

The result is not caused only by costs. The annual pre-cost returns implied by the recorded
cost drag are -0.6360% for cluster and -2.9967% for global. The cluster spread has 4.5894
annual one-way turns versus 2.5960 for global, creating 180.45 versus 100.42 bp/year of
cost drag at 10 bp. Thus clustering improves the pre-cost spread by 2.3606 percentage
points, of which 0.8004 percentage point is surrendered to its higher trading cost.

## Construction

- U1 point-in-time membership and eligibility; headline dates 2009-08-31 through
  2026-06-30, with the full panel reported separately.
- Frozen 48-week momentum signal with four-week skip, ME decisions, q=0.25,
  implementation lag 1, and 10 bp trading cost.
- Cluster leg: baseline clusters formed from ME returns with EWMA span 36; rank within
  cluster; group-equal budgets applied separately to the top and bottom books.
- Global leg: identical score and q, ranked over the whole eligible universe, asset-equal
  within each side.
- Both portfolios target +1 long and -1 short (gross 2, net zero). Any asset selected on
  both sides because of a singleton/tiny tied group is removed from both books and each
  side is renormalised to one.

## Results

| window | leg | pre-cost return | net return | volatility | Sharpe | turnover | cost drag, bp/year | cumulative net return |
|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| headline | ME/36 cluster | -0.6360% | -2.4406% | 6.8764% | -0.324483 | 4.5894 | 180.45 | -39.0048% |
| headline | global | -2.9967% | -4.0009% | 12.8460% | -0.251495 | 2.5960 | 100.42 | -55.8226% |
| full panel | ME/36 cluster | -1.8881% | -3.9821% | 9.3487% | -0.386511 | 5.3963 | 209.39 | -55.6493% |
| full panel | global | -6.4434% | -7.5875% | 16.1774% | -0.401400 | 3.0557 | 114.42 | -79.3780% |

| cluster minus global | pre-cost return | net return | volatility | Sharpe | turnover | cost drag |
|:--|--:|--:|--:|--:|--:|--:|
| headline | +2.3606 pp | +1.5603 pp | -5.9696 pp | -0.072988 | +1.9934 | +80.04 bp/year |
| full panel | +4.5552 pp | +3.6054 pp | -6.8287 pp | +0.014888 | +2.3407 | +94.98 bp/year |

The positive long-only returns in the originating grid therefore do not translate into a
positive top-minus-bottom momentum premium. They contained substantial long equity-market
exposure. The cluster construction improves the spread return and materially reduces its
volatility, but the headline spread remains negative before and after costs.

## Acceptance

| check | measured | tolerance | verdict |
|:--|--:|--:|:--|
| long exposure | maximum error 6.106227e-15 | <= 1e-12 | PASS |
| short exposure | maximum error 6.106227e-15 | <= 1e-12 | PASS |
| net exposure | maximum absolute 1.038232e-15 | <= 1e-12 | PASS |
| gross exposure | maximum error 1.620926e-14 | <= 1e-12 | PASS |
| pre-net side weight sum | maximum error 2.220446e-16 | <= 1e-12 | PASS |
| pre-net group budget | maximum error 1.110223e-16 | <= 1e-15 | PASS |
| deterministic replay | 5/5 artifacts byte-identical | 5/5 | PASS |
| independent validator | PASS | PASS | PASS |

Focused verification:

```text
...                                                                      [100%]
All checks passed!
```

Independent validator output:

```text
U1 ME/36 q=0.25 long-short independent validation: PASS
cluster: return=-0.024406, Sharpe=-0.324483, turnover=4.589398
global: return=-0.040009, Sharpe=-0.251495, turnover=2.596003
cluster-minus-global: return=0.015603, Sharpe=-0.072988
determinism: 5/5 artifacts byte-identical
```

The runner, report, tests, and output artifacts remain local and ignored by git. Nothing was
staged or pushed.
