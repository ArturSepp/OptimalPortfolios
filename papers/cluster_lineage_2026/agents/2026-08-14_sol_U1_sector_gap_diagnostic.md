# U1 sector-rank gap diagnostic

**Date:** 2026-08-14  
**Status:** DEFECT CONFIRMED  
**Diagnostic runner:** `papers/cluster_lineage_2026/replication/diagnose_u1_sector_gap.py`  
**Output:** `C:/Users/artur/OneDrive/analytics/outputs/cluster_lineage_2026/e5b/diagnostics/u1_sector_gap/`

## Verdict

The reported U1 sector-rank advantage is not a valid like-for-like payoff result. The
taxonomy leg silently drops every eligible asset whose static metadata has no GICS sector,
while global and cluster legs continue to invest in those assets. Missing GICS coverage is
strongly concentrated in historical/delisted constituents, creating a survivorship-like
universe filter. The taxonomy itself is also a single static label per security, applied
retrospectively to every date.

This is a U1 taxonomy-harness defect. The current U1 sector payoff rows and their E6
contrasts should not be interpreted until all legs share the same investable asset set and
the classification timing convention is repaired or explicitly bounded.

## Evidence 1 — sector coverage is not point-in-time complete

- Metadata securities: 1,358.
- Missing `gics_sector`: 436 (32.1%).
- Of missing-sector securities, 99.54% have a last constituent date before 2026, versus
  39.26% among classified securities.
- In the headline window, the taxonomy leg excludes an average 93.6 otherwise eligible
  assets per date, or 15.31% of the eligible universe.
- The exclusion share is 30.33% on 2009-08-31, has a median of 14.53%, reaches 30.73%, and
  falls to 0% by 2026-06-30.

Average target weight allocated by the other legs to the securities dropped by taxonomy:

| construction | leg | mean | median | maximum | first date | last date |
|:--|:--|--:|--:|--:|--:|--:|
| group_equal | global | 0.1484 | 0.1360 | 0.3719 | 0.3306 | 0.0000 |
| group_equal | taxonomy | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| group_equal | cluster_baseline | 0.1555 | 0.1427 | 0.3875 | 0.3566 | 0.0000 |
| group_equal | cluster_M1_delta_0.02 | 0.1557 | 0.1328 | 0.3879 | 0.3583 | 0.0000 |
| asset_equal | global | 0.1484 | 0.1360 | 0.3719 | 0.3306 | 0.0000 |
| asset_equal | taxonomy | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| asset_equal | cluster_baseline | 0.1472 | 0.1397 | 0.3647 | 0.3155 | 0.0000 |
| asset_equal | cluster_M1_delta_0.02 | 0.1470 | 0.1326 | 0.3684 | 0.3216 | 0.0000 |

## Evidence 2 — matched-universe controls collapse the gap

All controls use q=0.20, the headline window, 10 bp costs, lag 1, identical scores, and
the group-equal construction where grouping applies.

| leg | net return | Sharpe | alpha vs EW | beta vs EW | turnover | cost drag bp/yr |
|:--|--:|--:|--:|--:|--:|--:|
| global_all | 0.058203 | 0.440462 | -0.013286 | 0.907319 | 2.726819 | 57.23 |
| global_classified_only | 0.103843 | 0.706982 | 0.028780 | 0.915406 | 2.718935 | 59.54 |
| sector_drop_unclassified (reported leg) | 0.106818 | 0.759496 | 0.030289 | 0.914014 | 2.705630 | 59.42 |
| sector_explicit_unclassified | 0.063184 | 0.488661 | -0.008344 | 0.891541 | 2.722397 | 57.42 |
| cluster_baseline_all | 0.056295 | 0.470865 | -0.013238 | 0.856468 | 4.070638 | 85.41 |
| cluster_baseline_classified_only | 0.099450 | 0.752704 | 0.025272 | 0.881478 | 3.932892 | 85.96 |
| cluster_M1_002_all | 0.051731 | 0.437578 | -0.018120 | 0.863013 | 3.330949 | 69.57 |
| cluster_M1_002_classified_only | 0.094434 | 0.718587 | 0.020077 | 0.888110 | 3.142805 | 68.34 |

The equal-weight reference confirms that this is not specific to momentum: restricting EW
to GICS-covered assets raises annual return from 8.21% to 10.34% and Sharpe from 0.599 to
0.730.

Attribution of the originally reported sector advantage to the inconsistent coverage filter:

| comparator | original return gap | matched return gap | gap explained | original Sharpe gap | matched Sharpe gap | gap explained |
|:--|--:|--:|--:|--:|--:|--:|
| global | 0.048615 | 0.002975 | 93.9% | 0.319034 | 0.052514 | 83.5% |
| cluster_baseline | 0.050523 | 0.007368 | 85.4% | 0.288631 | 0.006792 | 97.6% |
| cluster_M1_delta_0.02 | 0.055087 | 0.012384 | 77.5% | 0.321918 | 0.040909 | 87.3% |

The key comparison is baseline clusters on the same classified universe: sector Sharpe
0.759 versus cluster Sharpe 0.753. The supposed 0.289 Sharpe advantage becomes 0.0068.

## Evidence 3 — static taxonomy is retrospectively applied

`UniverseData.taxonomy` is a 1,358-row security cross-section, not a date-indexed history.
The backtest tiles its single `gics_sector` value over all dates. It reports the same 11
groups in 2009 and 2026 and applies present-style `Communication Services` and `Real Estate`
labels to 2009. Thus the taxonomy leg is not point-in-time GICS and contains classification
look-ahead. The residual matched-universe difference cannot be treated as a clean sector
signal estimate until this is fixed.

## Secondary construction effects

The cluster and sector legs also have unequal effective breadth under the frozen selection
rule. On the headline classified universe at q=0.20:

| config | mean groups | mean selected assets | effective holdings | singleton-group share | groups of size <=5 |
|:--|--:|--:|--:|--:|--:|
| baseline | 80.17 | 149.94 | 113.27 | 8.90% | 59.30% |
| M1_delta_0.02 | 82.76 | 151.96 | 116.80 | 8.79% | 60.10% |

Taxonomy selects about 110 assets. Because every tiny cluster must select at least one,
cluster q=0.20 behaves closer to an effective 29% selection fraction. This dilutes the
cluster momentum leg and raises turnover. It is a design effect, not the main source of the
original gap.

Group-equal weighting widens the gap but does not create it: under the accepted asset-equal
construction the sector-minus-M1 return and Sharpe gaps were already 4.48 percentage points
and 0.253. The group-equal gaps are 5.51 points and 0.322.

Price availability is also secondary. At lag 1, average target weight without a price is
0.09% for taxonomy, 0.55% for global, and about 0.59–0.62% for cluster legs. This is a real
coverage asymmetry but is far too small to explain a 5.5 percentage-point annual return gap.
The cost-drag difference is about 10 bp/year, also far too small.

## Required correction before interpretation

1. The taxonomy leg must never silently drop a currently eligible asset. Missing labels
   must be an explicit `Unclassified` group, or every comparison leg must be restricted to
   an identical predeclared universe. The latter remains survivorship-biased here and is
   diagnostic only, not the preferred production fix.
2. Preferred production input is a point-in-time GICS panel covering every eligible date.
   If that is unavailable, the static-label limitation must be explicit and the taxonomy
   leg demoted to robustness rather than used as a clean yardstick.
3. Add an invariant that the eligible asset set entering global, taxonomy, and cluster
   ranking is identical on every date. Report the symmetric difference and fail on any
   silent exclusion.
4. After correction, rerun U1 E5/E5b payoff rows, the U1 E6 payoff contrasts, and the q
   sweep. Clustering estimates and E2 caches do not need to be rerun.

No production or accepted result was changed by this diagnostic. Nothing was staged or
pushed; all diagnostic code, outputs, and this report remain local under the ignored paper
tree.
