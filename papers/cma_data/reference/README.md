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
