# Stage E1 execution report - three-universe data layer

**Date:** 2026-08-13  
**Roadmap:** `papers/cluster_lineage_2026/agents/ROADMAP_cluster_lineage_empirics.md`  
**Owner dispatch:** `2026-08-13_claude_E0_review_and_E1_instructions.md`  
**Status:** COMPLETE; awaiting OWNER GATE E1

## Outcome

The common data layer is implemented and verified in the required U2 -> U3 -> U1 order.

- `replication/universes.py` supplies one common `UniverseData` container, a loader for each
  universe, point-in-time eligibility, taxonomy and role data, factor NAVs, and an
  eligibility-aware equal-weight benchmark constructor for later use in E5.
- `replication/fetch_ff6.py` fetched official Ken French daily FF5 and Momentum data and wrote
  the ignored, provenance-prefixed `data/ff6_factors_wwed.csv` artifact.
- `replication/run_data_quality.py` prints and writes the complete quality evidence.
- `replication/universes_test.py` independently checks return aggregation, U1 conventions,
  the 19 MAC classifications, the EW inclusion rule, and the kurtosis transform.
- No E2 run was started. `M1_star` and `M2_star` remain unset.

Runner scripts:

- `papers/cluster_lineage_2026/replication/fetch_ff6.py`
- `papers/cluster_lineage_2026/replication/run_data_quality.py`
- `papers/cluster_lineage_2026/replication/universes_test.py`

Input directory:

`papers/cluster_lineage_2026/data/`

Output directory:

`C:\Users\artur\OneDrive\analytics\outputs\cluster_lineage_2026\data_quality\`

## Acceptance results

| Universe | Frequency | Asset span | Shape | Missing share | Eligible min / median / max | Median asset Fisher excess kurtosis | `kappa_hat` | Factor span | Result |
|---|---|---|---:|---:|---:|---:|---:|---|---|
| U2 futures | W-WED | 1959-07-08 to 2026-08-12 | 3,502 x 95 | 0.351201 | 83 / 90 / 95 | 4.836869 | 1.612290 | 1999-01-06 to 2026-08-12, 11 MATF | PASS |
| U3 MAC | ME | 1998-12-31 to 2026-07-31 | 332 x 170 | 0.211623 | 97 / 150 / 170 | 2.510562 | 0.836854 | 1999-01-06 to 2026-08-12, 11 MATF | PASS |
| U3 MAC | QE | 1999-12-31 to 2026-06-30 | 107 x 17 | 0.100055 | 0 / 16 / 17 | 3.863878 | 1.287959 | 1999-01-06 to 2026-08-12, 11 MATF | PASS |
| U1 MSCI US | W-WED | 2006-08-02 to 2026-08-05 | 1,045 x 1,358 | 0.309629 | 0 / 619 / 641 | 6.373253 | 2.124418 | 1963-07-03 to 2026-07-01, FF6 | PASS with publication lag |

`kappa_hat` is `max(0, median_i(g_i) / 3)` in every row. The four values above are the
visible E1 inputs for the later `M1_star` calculation; no calibrated slot was filled.

U1 has zero eligible assets on the first two frozen estimation dates while the 12-week
warmup accumulates, then has a positive count on every remaining date. This yields 238
non-empty snapshots from the 240-date schedule and is within the E2 acceptance tolerance of
plus or minus two. U3 QE's minimum of zero is likewise a 12-native-observation warmup result,
not a dropped sleeve.

### Return and eligibility conventions

- U2: daily log returns are summed with
  `resample('W-WED').sum(min_count=1)`. All listed contracts become eligible after 12 valid
  weekly observations.
- U3: ME and QE log returns stay at their native frequencies; there is no resampling. Each
  sleeve becomes eligible after 12 valid native observations.
- U1: the source is owner-confirmed daily **excess log returns**. They are summed on W-WED;
  no risk-free rate is subtracted. Eligibility at an ME date requires the point-in-time
  inclusion flag and at least 12 weekly observations. An inclusion change between ME dates
  is therefore first visible at the next ME sampling date.

FF6 supplies U1's estimator factor panel, and `Mkt-RF` is its market factor. The MSCI US
total-return index file is reference only. The equal-weight performance/market benchmark is
constructed at backtest time in E5, not in E1.

### Weekly outlier scan

The exhaustive rows, including an `eligible` flag, are in the per-universe outlier CSVs.

- U2 has three `|log return| > 0.50` observations, all eligible:
  `XB1 Comdty` -0.554442470655202 on 2020-03-18; `CL1 Comdty`
  -0.685775443179168 on 2020-04-22; and `NG1 Comdty` 0.631991589225304 on
  2022-02-02.
- U3 has none at either ME or QE frequency.
- U1 has 1,201 such source observations, of which 257 occur while the security is eligible.
  All 1,201 are listed in `msci_us_outliers_abs_gt_050.csv`; they are retained, not winsorised
  or silently removed. The largest eligible values are associated primarily with delistings
  and distress events, including Signature Bank, Lehman Brothers, First Republic, Bear
  Stearns, Fannie Mae, Freddie Mac, and AIG.

### FF6 provenance and reconciliation

The fetch succeeded, so no FF6 escalation report was needed.

- Official sources: Ken French daily FF5 (2x3) and daily Momentum ZIP files.
- Vintage: 2026-08-13.
- Daily aligned source span: 1963-07-01 to 2026-06-30, satisfying the required start no later
  than 2003-01-01.
- Output: 3,288 W-WED rows and six columns: `Mkt-RF`, `SMB`, `HML`, `RMW`, `CMA`, `MOM`.
- Transformation: percent simple daily returns divided by 100, then `log1p`, then W-WED sum.
- Ten evenly spaced sampled weeks reconcile independently against summed daily log returns at
  absolute tolerance `1e-12`: PASS.
- The public vintage ends at W-WED 2026-07-01, 30 calendar days before the frozen U1 study
  end. The quality row marks this explicitly as `PASS_PUBLICATION_LAG`; it is not backfilled.

### MAC metadata gaps - exhaustive classification

The classification rule is source-based: a ticker found in the production replay
`saa_metadata.csv` is an investable `universe_member`; one found only in
`benchmark_metadata.csv` is a non-investable `benchmark_series`; otherwise it would be
`excluded`. Every uncovered return column is retained and classified: 17 universe members,
two benchmark series, zero excluded.

| Return column | Freq | Classification | Source rule | Asset Class / Sub Asset Class |
|---|---|---|---|---|
| LUATTRUU Index | ME | universe member | SAA metadata | Fixed Income / Government Bonds |
| LGCPTRUH Index | ME | universe member | SAA metadata | Fixed Income / Global IG Bonds |
| LG30TRUH Index | ME | universe member | SAA metadata | Fixed Income / Global HY Bonds |
| EMUSTRUU Index | ME | universe member | SAA metadata | Fixed Income / EM Bonds |
| LD19TRUU Index | ME | universe member | SAA metadata | Fixed Income / Other Fixed Income |
| NDDUUS Index | ME | universe member | SAA metadata | Equity / North America |
| MSDEE15N Index | ME | universe member | SAA metadata | Equity / Europe |
| NDDLJN Index | ME | universe member | SAA metadata | Equity / Japan |
| M1APJ Index | ME | universe member | SAA metadata | Equity / Asia ex-Japan |
| M1EFZ Index | ME | universe member | SAA metadata | Equity / EM ex-Asia |
| BCOMXPM Index | ME | universe member | SAA metadata | Alternatives / Commodities ex-Precious |
| BCOMPR Index | ME | universe member | SAA metadata | Alternatives / Commodities Precious |
| RUGL Index | ME | universe member | SAA metadata | Alternatives / REITs |
| LEGATRUH Index | ME | benchmark series | benchmark metadata only | Fixed Income / Credit |
| NDUEACWF Index | ME | benchmark series | benchmark metadata only | Equity / Equity |
| HFRXGL Index | QE | universe member | SAA metadata | Alternatives / Hedge Funds |
| MP503001 Index | QE | universe member | SAA metadata | Alternatives / Private Equity |
| MP503008 Index | QE | universe member | SAA metadata | Alternatives / Private Debt |
| EHFI804 Index | QE | universe member | SAA metadata | Alternatives / Insurance-Linked |

The two benchmark series remain in U3 for production covariance replication but are excluded
from the E5 equal-weight investable basket. The independent EW test verifies that exclusion.

## Deliverable files

External quality evidence (11 deterministic CSVs):

- `all_universes_data_quality.csv`
- `<universe>_data_quality.csv`, `<universe>_eligibility_counts.csv`, and
  `<universe>_outliers_abs_gt_050.csv` for futures, MAC, and MSCI US
- `mac_uncovered_classification.csv`

The generated FF6 artifact stays ignored under `papers/cluster_lineage_2026/data/`. The code,
tests, and agent reports are trackable after the E0b `.gitignore` narrowing. Nothing was
staged or pushed.

## Verification

E1 tests:

```text
.....                                                                    [100%]
```

Two complete executions of `run_data_quality.run_all()` produced the same 11 CSV files,
byte-for-byte: PASS (127,102 total bytes). The combined quality table has no NaNs. Focused
Ruff `E,F,W` result for all E0b/E1 Python files: `All checks passed!`

The owner-mandated E0b checks were rerun against
`C:\Users\artur\OneDrive\analytics\outputs\cluster_smoothing\sp500_baseline\baseline\`:

```text
....                                                                     [100%]
```

`validate_e0.py` then passed all six frozen checks, with deterministic status PASS and 923
identical serialized bytes. The E0b frozen numbers remain unchanged.

## Deviations and open items

- `risk_factors_custom.csv` contains strictly positive level series beginning at 100, not
  daily log returns despite the roadmap's data-description sentence. U2/U3 therefore consume
  it as the existing 11-factor MATF NAV panel and sample W-WED by last observation; treating
  the levels as returns would be numerically invalid. This convention requires owner sign-off.
- The latest public FF6 vintage has the explicit 30-day end lag described above. No values
  were synthesized.
- The standalone spreadsheet artifact runtime was unavailable in this environment. The CSV
  evidence was produced and independently verified with the repository's pandas-based harness;
  this did not alter formulas, formats, or deliverable locations.

## GATE REQUEST

Please rule on exactly these E1 data conventions before gated downstream interpretation:

1. Approve the four return/eligibility conventions and measured quality rows above, including
   the two U1 warmup-empty dates and the U3 QE warmup minimum of zero.
2. Approve consuming `risk_factors_custom.csv` as MATF factor NAV levels sampled W-WED by
   last observation, resolving the roadmap sentence that describes that file as log returns.
3. Approve the official FF6 vintage with its disclosed 30-day publication lag for U1.
4. Approve the exhaustive MAC classification: 17 universe members, two benchmark-only series,
   and zero excluded columns.
5. Confirm FF6/Mkt-RF for U1 estimation, the MSCI index as reference only, and construction of
   the EW benchmark in E5.
