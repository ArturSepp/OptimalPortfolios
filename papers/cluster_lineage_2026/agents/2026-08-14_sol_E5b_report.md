# E5b execution report — group-equal grouped ranking legs

**Date:** 2026-08-14  
**Status:** COMPLETE  
**Runner:** `papers/cluster_lineage_2026/replication/run_e5b.py`  
**Tests:** `papers/cluster_lineage_2026/replication/e5b_test.py`  
**Independent validator:** `papers/cluster_lineage_2026/replication/validate_e5b.py`

## Execution scope and construction

E5b was executed locally for U1 MSCI US, U2 futures, and U3 MAC. No clustering was
re-estimated and the E2 cache tree was untouched. The new primary construction is
`group_equal` for every taxonomy and cluster leg. At every date, groups with at least one
eligible asset and a valid score receive exactly `1/G`; selected assets within a group split
that budget equally. Groups with no valid score are excluded from `G`. The global leg and the
EW-all reference keep their accepted asset-equal construction. The accepted E5
`asset_equal` results remain unchanged and are retained as a labelled robustness variant.

Selection, implementation lag, rebalance dates, q=0.20 primary, q=1/3 and vol-adjusted
robustness, and 10/20/50 bp costs remain frozen. U1 uses its ME schedule. U3 QE selections
were independently checked over all 17 QE assets: **0 selection changes occurred on non-QE
dates**, so the QE sleeve enters at QE dates and holds between them.

Metric 11 uses the binding counterfactual: `w_tilde` is rebuilt with the same group-equal
formula under the prior-date partition and its prior group count. Consequently, group-count
changes enter the reassignment component. The signed residual remains the trade-interaction
term; the former 10% residual guard is retired and is not used as an acceptance criterion.

The comparison yardsticks are only global rank and taxonomy rank. EW-all appears only in a
separate reference block as the alpha-profile base and the market benchmark for beta/alpha.
No ranking-leg performance conclusion or acceptance line compares against EW-all.

## Execution and output locations

- E5b root: `C:/Users/artur/OneDrive/analytics/outputs/cluster_lineage_2026/e5b/`
- U1 group-equal outputs: `e5b/group_equal/msci_us/`
- U2 group-equal outputs: `e5b/group_equal/futures/`
- U3 group-equal outputs: `e5b/group_equal/mac/`
- E6 addendum: `e5b/e6_addendum/`
- Accepted asset-equal inputs: `backtests/<universe>/`
- Cluster caches, unchanged:
  `C:/Users/artur/OneDrive/analytics/outputs/cluster_lineage_2026/<msci_us|futures|mac>/<config>/YYYYMMDD.pkl`

U1 ran the full E5 profile in the isolated E5b directory. U2 and U3 reran only taxonomy and
cluster grouped legs; their accepted global and EW reference series were reused, not
backtested again. Complete CSV/PDF profiles, payoff tables, EW reference blocks, robustness
rows, crisis windows, weights, NAVs, returns, turnover decomposition, and group-count
diagnostics are in each universe directory.

## Acceptance checks

| Universe | Acceptance line | Measured | Tolerance | Verdict |
|:--|:--|--:|--:|:--|
| U1 MSCI US | maximum weight-sum absolute error | 2.220446e-16 | <= 1e-12 | PASS |
| U1 MSCI US | maximum per-group budget absolute error | 1.387779e-17 | <= 1e-15 | PASS |
| U1 MSCI US | EW performance-comparison violations | 0 | 0 | PASS |
| U2 futures | maximum weight-sum absolute error | 2.220446e-16 | <= 1e-12 | PASS |
| U2 futures | maximum per-group budget absolute error | 1.387779e-17 | <= 1e-15 | PASS |
| U2 futures | EW performance-comparison violations | 0 | 0 | PASS |
| U3 MAC | maximum weight-sum absolute error | 2.220446e-16 | <= 1e-12 | PASS |
| U3 MAC | maximum per-group budget absolute error | 1.387779e-17 | <= 1e-15 | PASS |
| U3 MAC | EW performance-comparison violations | 0 | 0 | PASS |
| all | deterministic replay | 46/46 CSV artifacts byte-identical | 46/46 | PASS |
| all | E6 group-equal bootstrap coverage | 36 rows | 36 | PASS |
| all | E6 combined construction coverage | 72 rows | 72 | PASS |
| U3 MAC | QE selection changes on non-QE dates | 0 across 17 QE assets | 0 | PASS |

Unit-test output:

```text
..                                                                       [100%]
```

Independent-validator output:

```text
E5b independent validation: PASS
acceptance lines: 9/9 PASS
determinism replay: 46/46 CSV artifacts byte-identical
E6 payoff bootstrap: 36 group_equal rows; 72 combined rows
U3 QE hold: 17 assets; 0 non-QE-date selection changes
```

## Primary group-equal payoff summary

Returns, Sharpe, and turnover below are ranking-leg quantities. Delta columns are only
against the two authorised yardsticks.

| universe | analysis window | leg | net return ann. | Sharpe RF=0 | one-way turnover ann. | return vs global | Sharpe vs global | return vs taxonomy | Sharpe vs taxonomy |
|:--|:--|:--|--:|--:|--:|--:|--:|--:|--:|
| U1 | headline | global | 0.058203 | 0.440462 | 2.726819 | — | — | — | — |
| U1 | headline | taxonomy | 0.106818 | 0.759496 | 2.705630 | — | — | — | — |
| U1 | headline | cluster_baseline | 0.056295 | 0.470865 | 4.070638 | -0.001908 | 0.030403 | -0.050523 | -0.288631 |
| U1 | headline | cluster_M1_delta_0.02 | 0.051731 | 0.437578 | 3.330949 | -0.006472 | -0.002884 | -0.055087 | -0.321918 |
| U1 | full panel | global | 0.019464 | 0.201717 | 3.159751 | — | — | — | — |
| U1 | full panel | taxonomy | 0.088510 | 0.579445 | 3.145205 | — | — | — | — |
| U1 | full panel | cluster_baseline | 0.025122 | 0.238585 | 4.884169 | 0.005659 | 0.036868 | -0.063387 | -0.340860 |
| U1 | full panel | cluster_M1_delta_0.02 | 0.020921 | 0.213133 | 4.074100 | 0.001457 | 0.011417 | -0.067589 | -0.366311 |
| U2 | full panel | global | 0.030639 | 0.393512 | 0.989010 | — | — | — | — |
| U2 | full panel | taxonomy | 0.024679 | 0.432213 | 1.038282 | — | — | — | — |
| U2 | full panel | cluster_baseline | 0.022615 | 0.402802 | 1.096378 | -0.008024 | 0.009291 | -0.002064 | -0.029411 |
| U2 | full panel | cluster_M1_star | 0.025152 | 0.434132 | 0.962328 | -0.005487 | 0.040620 | 0.000472 | 0.001919 |
| U3 | full panel | global | 0.054850 | 0.508487 | 2.255057 | — | — | — | — |
| U3 | full panel | taxonomy | 0.035364 | 0.551027 | 1.666525 | — | — | — | — |
| U3 | full panel | cluster_baseline | 0.033603 | 0.523880 | 2.947435 | -0.021247 | 0.015392 | -0.001761 | -0.027148 |
| U3 | full panel | cluster_M1_delta_0.05 | 0.040384 | 0.605699 | 2.411168 | -0.014466 | 0.097212 | 0.005020 | 0.054672 |

The full config grid, both construction labels, all cost levels, q=1/3, vol-adjusted rows,
and beta/alpha-vs-EW reference fields are in `payoff_comparison.csv`,
`construction_comparison.csv`, `robustness.csv`, and `ew_reference.csv` per universe.

## Available-group-count stability channel

The standard deviation is the per-date standard deviation requested by the owner.

| universe | window | leg | mean G | std G | min | max |
|:--|:--|:--|--:|--:|--:|--:|
| U1 | headline | taxonomy | 11.000000 | 0.000000 | 11 | 11 |
| U1 | headline | cluster_baseline | 81.773399 | 19.085993 | 32 | 122 |
| U1 | headline | cluster_M1_delta_0.02 | 84.261084 | 18.586320 | 35 | 121 |
| U1 | full panel | taxonomy | 11.000000 | 0.000000 | 11 | 11 |
| U1 | full panel | cluster_baseline | 78.844538 | 19.274527 | 32 | 122 |
| U1 | full panel | cluster_M1_delta_0.02 | 81.247899 | 19.001485 | 35 | 121 |
| U2 | full panel | taxonomy | 7.000000 | 0.000000 | 7 | 7 |
| U2 | full panel | cluster_baseline | 16.250847 | 3.092363 | 9 | 25 |
| U2 | full panel | cluster_M0_quarterly_hold | 16.318644 | 3.160151 | 11 | 25 |
| U2 | full panel | cluster_M1_delta_0.02 | 16.328814 | 2.901504 | 9 | 24 |
| U2 | full panel | cluster_M1_delta_0.05 | 16.271186 | 2.891640 | 9 | 24 |
| U2 | full panel | cluster_M1_delta_0.10 | 15.925424 | 2.542402 | 11 | 22 |
| U2 | full panel | cluster_M2_lambda_0.5 | 16.488136 | 3.043734 | 10 | 25 |
| U2 | full panel | cluster_M2_lambda_0.7 | 16.694915 | 3.076431 | 10 | 24 |
| U2 | full panel | cluster_M1_star | 16.016949 | 2.748478 | 11 | 23 |
| U3 | full panel | taxonomy | 4.000000 | 0.000000 | 4 | 4 |
| U3 | full panel | cluster_baseline | 14.848592 | 3.681915 | 8 | 23 |
| U3 | full panel | cluster_M0_quarterly_hold | 15.017606 | 3.671547 | 8 | 23 |
| U3 | full panel | cluster_M1_delta_0.02 | 14.943662 | 3.719445 | 9 | 23 |
| U3 | full panel | cluster_M1_delta_0.05 | 14.809859 | 3.679632 | 8 | 24 |
| U3 | full panel | cluster_M1_delta_0.10 | 14.697183 | 3.521744 | 9 | 23 |
| U3 | full panel | cluster_M2_lambda_0.5 | 15.288732 | 3.688548 | 9 | 25 |
| U3 | full panel | cluster_M2_lambda_0.7 | 15.542254 | 3.637502 | 9 | 23 |

The corresponding per-date panels contain 1,323 U1, 2,655 U2, and 2,272 U3 rows.

## Metric 11 decomposition

| universe | window | leg | reassignment | signal | total | signed trade interaction | absolute interaction share |
|:--|:--|:--|--:|--:|--:|--:|--:|
| U1 | headline | cluster_baseline | 0.282345 | 0.234931 | 0.401361 | -0.115915 | 0.288805 |
| U1 | headline | cluster_M1_delta_0.02 | 0.164755 | 0.233433 | 0.328485 | -0.069703 | 0.212196 |
| U1 | full panel | cluster_baseline | 0.299581 | 0.236498 | 0.412296 | -0.123783 | 0.300229 |
| U1 | full panel | cluster_M1_delta_0.02 | 0.190282 | 0.234420 | 0.344170 | -0.080531 | 0.233987 |
| U2 | full panel | cluster_baseline | 0.084662 | 0.195979 | 0.248570 | -0.032071 | 0.129024 |
| U2 | full panel | cluster_M0_quarterly_hold | 0.051892 | 0.196987 | 0.229789 | -0.019091 | 0.083079 |
| U2 | full panel | cluster_M1_delta_0.02 | 0.052300 | 0.197191 | 0.227495 | -0.021996 | 0.096688 |
| U2 | full panel | cluster_M1_delta_0.05 | 0.031448 | 0.197633 | 0.215190 | -0.013891 | 0.064551 |
| U2 | full panel | cluster_M1_delta_0.10 | 0.022136 | 0.199007 | 0.212550 | -0.008593 | 0.040427 |
| U2 | full panel | cluster_M2_lambda_0.5 | 0.057592 | 0.194726 | 0.228736 | -0.023582 | 0.103098 |
| U2 | full panel | cluster_M2_lambda_0.7 | 0.049424 | 0.190623 | 0.222102 | -0.017945 | 0.080796 |
| U2 | full panel | cluster_M1_star | 0.029322 | 0.201204 | 0.218035 | -0.012491 | 0.057290 |
| U3 | full panel | cluster_baseline | 0.203220 | 0.186368 | 0.286586 | -0.103001 | 0.359407 |
| U3 | full panel | cluster_M0_quarterly_hold | 0.118875 | 0.184300 | 0.230487 | -0.072688 | 0.315364 |
| U3 | full panel | cluster_M1_delta_0.02 | 0.146691 | 0.184488 | 0.255942 | -0.075237 | 0.293961 |
| U3 | full panel | cluster_M1_delta_0.05 | 0.115947 | 0.177516 | 0.233462 | -0.060001 | 0.257005 |
| U3 | full panel | cluster_M1_delta_0.10 | 0.086084 | 0.176085 | 0.217739 | -0.044430 | 0.204052 |
| U3 | full panel | cluster_M2_lambda_0.5 | 0.168936 | 0.177747 | 0.266517 | -0.080166 | 0.300793 |
| U3 | full panel | cluster_M2_lambda_0.7 | 0.150364 | 0.180247 | 0.255986 | -0.074624 | 0.291515 |

Every row is labelled `RETIRED_NOT_AN_ACCEPTANCE_CRITERION` in the artifact; the signed
interaction is reported, not rejected.

## E6 payoff-bootstrap addendum — all 72 rows

Frozen inference parameters: moving block length 6, 2,000 draws, seed 20260813. `sig=yes`
means the percentile 95% CI excludes zero. No CI is computed against EW-all.

| universe | window | construction | contrast | metric | estimate | 95% CI | sig |
|:--|:--|:--|:--|:--|--:|:--|:--|
| U1 | headline | asset_equal | M1_0.02 - global | net return | 0.006812 | [-0.018410, 0.030329] | no |
| U1 | headline | asset_equal | M1_0.02 - global | Sharpe | 0.071247 | [-0.062131, 0.244380] | no |
| U1 | headline | asset_equal | M1_0.02 - global | turnover | 0.282722 | [0.192211, 0.370475] | yes |
| U1 | headline | asset_equal | M1_0.02 - taxonomy | net return | -0.044813 | [-0.064541, -0.026238] | yes |
| U1 | headline | asset_equal | M1_0.02 - taxonomy | Sharpe | -0.253213 | [-0.388570, -0.137347] | yes |
| U1 | headline | asset_equal | M1_0.02 - taxonomy | turnover | 0.300560 | [0.214549, 0.384139] | yes |
| U1 | headline | asset_equal | M1_0.02 - baseline | net return | -0.002822 | [-0.010391, 0.003782] | no |
| U1 | headline | asset_equal | M1_0.02 - baseline | Sharpe | -0.020484 | [-0.067487, 0.020670] | no |
| U1 | headline | asset_equal | M1_0.02 - baseline | turnover | -0.479883 | [-0.541424, -0.424703] | yes |
| U1 | full panel | asset_equal | M1_0.02 - global | net return | 0.019421 | [-0.006522, 0.047563] | no |
| U1 | full panel | asset_equal | M1_0.02 - global | Sharpe | 0.113337 | [-0.019059, 0.264207] | no |
| U1 | full panel | asset_equal | M1_0.02 - global | turnover | 0.455435 | [0.324977, 0.600690] | yes |
| U1 | full panel | asset_equal | M1_0.02 - taxonomy | net return | -0.050695 | [-0.069530, -0.032314] | yes |
| U1 | full panel | asset_equal | M1_0.02 - taxonomy | Sharpe | -0.264712 | [-0.392710, -0.160349] | yes |
| U1 | full panel | asset_equal | M1_0.02 - taxonomy | turnover | 0.464066 | [0.337275, 0.592098] | yes |
| U1 | full panel | asset_equal | M1_0.02 - baseline | net return | -0.003093 | [-0.009693, 0.003669] | no |
| U1 | full panel | asset_equal | M1_0.02 - baseline | Sharpe | -0.018090 | [-0.056463, 0.021059] | no |
| U1 | full panel | asset_equal | M1_0.02 - baseline | turnover | -0.540980 | [-0.610797, -0.472912] | yes |
| U2 | full panel | asset_equal | M1_star - global | net return | -0.006607 | [-0.051855, 0.029362] | no |
| U2 | full panel | asset_equal | M1_star - global | Sharpe | 0.050254 | [-0.181939, 0.337623] | no |
| U2 | full panel | asset_equal | M1_star - global | turnover | 0.019488 | [-0.071959, 0.117143] | no |
| U2 | full panel | asset_equal | M1_star - taxonomy | net return | -0.000542 | [-0.019502, 0.017290] | no |
| U2 | full panel | asset_equal | M1_star - taxonomy | Sharpe | 0.012268 | [-0.155156, 0.207612] | no |
| U2 | full panel | asset_equal | M1_star - taxonomy | turnover | -0.068144 | [-0.126484, -0.006987] | yes |
| U2 | full panel | asset_equal | M1_star - baseline | net return | 0.000935 | [-0.005250, 0.007516] | no |
| U2 | full panel | asset_equal | M1_star - baseline | Sharpe | 0.013832 | [-0.051974, 0.090995] | no |
| U2 | full panel | asset_equal | M1_star - baseline | turnover | -0.072367 | [-0.110093, -0.036156] | yes |
| U3 | full panel | asset_equal | M1_0.05 - global | net return | -0.012976 | [-0.038882, 0.012198] | no |
| U3 | full panel | asset_equal | M1_0.05 - global | Sharpe | 0.001007 | [-0.213585, 0.241594] | no |
| U3 | full panel | asset_equal | M1_0.05 - global | turnover | 0.117981 | [-0.073708, 0.313792] | no |
| U3 | full panel | asset_equal | M1_0.05 - taxonomy | net return | -0.007113 | [-0.020527, 0.005279] | no |
| U3 | full panel | asset_equal | M1_0.05 - taxonomy | Sharpe | -0.017533 | [-0.120294, 0.107840] | no |
| U3 | full panel | asset_equal | M1_0.05 - taxonomy | turnover | 0.276513 | [0.137988, 0.427016] | yes |
| U3 | full panel | asset_equal | M1_0.05 - baseline | net return | 0.004442 | [-0.000224, 0.009254] | no |
| U3 | full panel | asset_equal | M1_0.05 - baseline | Sharpe | 0.050273 | [0.000619, 0.110762] | yes |
| U3 | full panel | asset_equal | M1_0.05 - baseline | turnover | -0.410557 | [-0.500552, -0.326195] | yes |
| U1 | headline | group_equal | M1_0.02 - global | net return | -0.006472 | [-0.035218, 0.021487] | no |
| U1 | headline | group_equal | M1_0.02 - global | Sharpe | -0.002884 | [-0.151443, 0.192597] | no |
| U1 | headline | group_equal | M1_0.02 - global | turnover | 0.604130 | [0.483186, 0.718533] | yes |
| U1 | headline | group_equal | M1_0.02 - taxonomy | net return | -0.055087 | [-0.078188, -0.031533] | yes |
| U1 | headline | group_equal | M1_0.02 - taxonomy | Sharpe | -0.321918 | [-0.498183, -0.167796] | yes |
| U1 | headline | group_equal | M1_0.02 - taxonomy | turnover | 0.625319 | [0.511793, 0.745741] | yes |
| U1 | headline | group_equal | M1_0.02 - baseline | net return | -0.004565 | [-0.015652, 0.005520] | no |
| U1 | headline | group_equal | M1_0.02 - baseline | Sharpe | -0.033287 | [-0.112804, 0.029249] | no |
| U1 | headline | group_equal | M1_0.02 - baseline | turnover | -0.739689 | [-0.810590, -0.673419] | yes |
| U1 | full panel | group_equal | M1_0.02 - global | net return | 0.001457 | [-0.029226, 0.033320] | no |
| U1 | full panel | group_equal | M1_0.02 - global | Sharpe | 0.011417 | [-0.149002, 0.191938] | no |
| U1 | full panel | group_equal | M1_0.02 - global | turnover | 0.914348 | [0.720056, 1.125986] | yes |
| U1 | full panel | group_equal | M1_0.02 - taxonomy | net return | -0.067589 | [-0.094354, -0.041153] | yes |
| U1 | full panel | group_equal | M1_0.02 - taxonomy | Sharpe | -0.366311 | [-0.558416, -0.214542] | yes |
| U1 | full panel | group_equal | M1_0.02 - taxonomy | turnover | 0.928895 | [0.742540, 1.129625] | yes |
| U1 | full panel | group_equal | M1_0.02 - baseline | net return | -0.004202 | [-0.014155, 0.005394] | no |
| U1 | full panel | group_equal | M1_0.02 - baseline | Sharpe | -0.025451 | [-0.092674, 0.031027] | no |
| U1 | full panel | group_equal | M1_0.02 - baseline | turnover | -0.810069 | [-0.896226, -0.723186] | yes |
| U2 | full panel | group_equal | M1_star - global | net return | -0.005487 | [-0.046626, 0.028356] | no |
| U2 | full panel | group_equal | M1_star - global | Sharpe | 0.040620 | [-0.177660, 0.310668] | no |
| U2 | full panel | group_equal | M1_star - global | turnover | -0.026682 | [-0.119325, 0.073531] | no |
| U2 | full panel | group_equal | M1_star - taxonomy | net return | 0.000472 | [-0.022016, 0.023344] | no |
| U2 | full panel | group_equal | M1_star - taxonomy | Sharpe | 0.001919 | [-0.190894, 0.226853] | no |
| U2 | full panel | group_equal | M1_star - taxonomy | turnover | -0.075954 | [-0.153157, -0.004228] | yes |
| U2 | full panel | group_equal | M1_star - baseline | net return | 0.002536 | [-0.006102, 0.011456] | no |
| U2 | full panel | group_equal | M1_star - baseline | Sharpe | 0.031330 | [-0.050078, 0.129715] | no |
| U2 | full panel | group_equal | M1_star - baseline | turnover | -0.134051 | [-0.180482, -0.087657] | yes |
| U3 | full panel | group_equal | M1_0.05 - global | net return | -0.014466 | [-0.044150, 0.012832] | no |
| U3 | full panel | group_equal | M1_0.05 - global | Sharpe | 0.097212 | [-0.121364, 0.362947] | no |
| U3 | full panel | group_equal | M1_0.05 - global | turnover | 0.156111 | [-0.057041, 0.379360] | no |
| U3 | full panel | group_equal | M1_0.05 - taxonomy | net return | 0.005020 | [-0.005460, 0.015754] | no |
| U3 | full panel | group_equal | M1_0.05 - taxonomy | Sharpe | 0.054672 | [-0.082870, 0.226989] | no |
| U3 | full panel | group_equal | M1_0.05 - taxonomy | turnover | 0.744642 | [0.586353, 0.908600] | yes |
| U3 | full panel | group_equal | M1_0.05 - baseline | net return | 0.006781 | [0.000155, 0.013757] | yes |
| U3 | full panel | group_equal | M1_0.05 - baseline | Sharpe | 0.081820 | [-0.003530, 0.181821] | no |
| U3 | full panel | group_equal | M1_0.05 - baseline | turnover | -0.536267 | [-0.652232, -0.421613] | yes |

## Result synopsis

- U1 headline group-equal M1_0.02 reduces turnover by 0.739689 versus the baseline cluster
  leg, with its 95% CI excluding zero; its return and Sharpe changes versus baseline do not.
- U2 group-equal M1_star reduces turnover by 0.134051 versus baseline and by 0.075954 versus
  taxonomy, both with CIs excluding zero. Its payoff deltas are not distinguishable from zero.
- U3 group-equal M1_0.05 reduces turnover by 0.536267 versus baseline and has a positive
  0.006781 net-return delta versus baseline, with both CIs excluding zero. Its return and
  Sharpe deltas versus the two ranking yardsticks do not exclude zero.

These statements compare ranking legs only. EW-all is not used as a payoff yardstick.

## Determinism, deviations, and repository state

The full E5b run was replayed and all 46 emitted CSV artifacts had identical SHA-256 hashes.
PDF files were regenerated but are not part of the byte-identity check. The independent
validator re-read every acceptance table, both U1 windows, group-count diagnostics, both
construction labels, frozen bootstrap parameters, and the U3 QE hold.

There were no deviations from the owner dispatch and no open items. Nothing was staged or
pushed. `papers/cluster_lineage_2026/` remains ignored, all implementation/report files are on
the local OneDrive-backed working folder, and `main` remains clean.
