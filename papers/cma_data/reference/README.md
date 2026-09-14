# reference/ — public external reference data

Public survey and reference tables consumed by the paper packages. Files here
are transcriptions of published sources, tracked because they are public and
the papers' numbers must trace to committed inputs. The source documents
themselves live in `papers/references/` (untracked).

## horizon_actuarial_2025_average_assumptions.csv

Horizon Actuarial Services, LLC, "Survey of Capital Market Assumptions:
2025 Edition", August 2025, Exhibit 17 (Average Survey Assumptions).
Transcribed 2026-07-30 from `papers/references/Rpt_CMA_Survey_2025_v0809.pdf`
(md5 93c9bc2188e94b4f2103e7ff86f2a694).

- 41 advisors, each assumption set equal-weighted; assumption vintages
  effective around January 2025.
- 10-year columns: all 41 advisors. 20-year columns: the 27-advisor subset
  providing long-term assumptions.
- Returns are nominal TOTAL returns for a USD investor, in decimals.
  `arith_*` = one-year average (arithmetic); `geom_*` = multi-year
  annualized (geometric). `stdev` applies to both bases.
- The `Inflation` row is the survey's expected-inflation line.
- The correlation matrix of Exhibit 17 is not transcribed; read it from the
  source PDF if needed.

## horizon_actuarial_2025_distributions_10y.csv

Same source, Exhibit 20: ranges of expected annual returns across the 41
advisors, 10-year horizon, GEOMETRIC basis, in decimals.
Columns: min, 25th, median, 75th, max as printed. The 20-year version
(Exhibit 21, 27 advisors) is not transcribed.

## Horizon 2026 reference tables (MATF-CMA paper)

Source: Horizon Actuarial Services, LLC, *Survey of Capital Market Assumptions:
2026 Edition*, August 2026. Local source `Rpt_CMA_Survey_2026_v0826.pdf`,
20 pages, SHA256
`3adcebd34d7ff3f9e3bb6d751594115baade0d0865a76fab4b02300ed37d7973`.
Transcribed September 11 and independently checked against the PDF on
September 14, 2026. All 180 printed values in these two CSVs match the source.

- `horizon_actuarial_2026_average_assumptions.csv`: Exhibit 17, PDF page 16;
  18 rows including inflation, decimal arithmetic/geometric 10/20-year returns
  and standard deviations. The 10-year group comprises 43 advisors; the
  20-year group is the 29-advisor subset supplying long-term assumptions.
- `horizon_actuarial_2026_distributions_10y.csv`: Exhibit 20, PDF page 19;
  geometric annual-return min/25th/median/75th/max across the nominal
  10-year advisor group, in decimals. These are cross-advisor forecast
  percentiles, not return outcomes or sampling confidence intervals.
- Assumptions are mostly effective around January 1, 2026; reported dates
  range from October 1, 2025 to March 31, 2026. The nominal 10-year group
  includes one advisor supplying only a seven-year horizon.
- The paper declares its sleeve/USD proxy mappings; the PDF does not explicitly
  standardize each asset category's currency-hedging convention. In particular,
  ex-US developed debt is a proxy for the USD-hedged global government sleeve,
  and broad commodities proxy Gold. These remain mapping judgments.
- The MATF-CMA comparison consumes the printed arithmetic averages directly
  and keeps the Q2 MATF covariance and 4.1811% reference cash anchor. Survey
  cash (3.48%) and survey volatilities are external reference assumptions.
- Shared 2025 inputs and API defaults remain unchanged. The JPM-specific
  `horizon_consensus_2026.py` selects these versioned 2026 files.
