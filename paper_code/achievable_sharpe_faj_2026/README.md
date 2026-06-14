# Achievable Sharpe and Universe Selection — FAJ 2026

Replication package for:

> Sepp, A., and M.A. Kastenholz (2026), *Achievable Sharpe and Universe
> Selection: A Closed-Form Factor Decomposition*, working paper, under review
> at the Financial Analysts Journal.

Companion papers in this repository: `matf_cma_jpm_2026` (MATF-CMA, JPM) and
the ROSAA framework (Sepp, Ossa, Kastenholz, JPM 52(4) 2026).

## Layout

```
identity_analytics.py    closed-form analytics: MATF Identity, ceiling, FPIR,
                         per-factor attribution, identification deficit,
                         candidate-sleeve pricing (rank-one update), Eq.7 weight
                         blocks, GLS-orthogonal alpha, factor-Sharpe coordinates,
                         long-only tangency, TE overlay
paper_inputs.py          production workbook + factor-NAV loaders; paired
                         bootstrap panels (monthly factor log returns, native-
                         frequency residuals)
run_bootstrap.py         two-configuration identity bootstrap (paper Fig. 9,
                         Table 2): Config A ex-ante (structural SR prior),
                         Config B ex-post (realized premia). Sampler, draw
                         count (500), and seed (42) identical to
                         matf_cma_jpm_2026/run_bootstrap.py
run_exhibits.py          regenerates Figures 1-8 and 10 from cached inputs
tests/                   reproducibility suite asserting the published numbers
data/                    production workbook (2026Q1 vintage), factor NAVs,
                         universe snapshot, cached derived inputs (npz/pkl)
figures/                 paper figures (PNG, 160 dpi)
paper/                   LaTeX source and compiled PDF
```

## Reproduce

```bash
python run_exhibits.py            # Figures 1-9, 11
python run_bootstrap.py           # Figure 10 + Table 2 numbers
python -m pytest tests/ -v        # full suite incl. 500-draw bootstrap
python -m pytest tests/ -v -m "not slow"   # quick checks only
```

Dependencies: numpy, pandas, scipy, matplotlib, openpyxl, cvxpy (CLARABEL),
pytest. Python >= 3.10.

## Headline numbers (17 assets, USD, 31 Mar 2026)

| quantity | SR^2 | SR |
|---|---|---|
| frictionless ceiling | 0.61 | 0.78 |
| MATF Identity (FPIR 0.43) | 0.26 | 0.51 |
| + idiosyncratic (ILS, RA) | 0.37 | 0.61 |
| long-only tangency | 0.17 | 0.41 |

Bootstrap (500 draws, seed 42; panels Apr 2001-Mar 2026, T = 300 months,
identical window to the MATF-CMA bootstrap): Config A (ex ante) mean SR 0.54,
sd 0.08, P(SR > 0.40) = 98%; Config B (ex post) mean 0.75, sd 0.16.
All display values are rounded to two decimals (paper convention); the test
suite asserts the underlying full-precision values.

## Notes

- `stationary_block_indices` in `run_bootstrap.py` is a **verbatim copy** of
  `matf_cma_jpm_2026/bootstrap_frontier_analytics.py` (Politis-Romano 1994,
  min-block floor). It is slated for the `qis` package; once ported, both
  papers' replication code should import it from there and the local copies
  be removed.
- All inference is computed at native sampling frequency; annualized Sharpe
  values are for display.
- Data files are production vintages frozen at 31 March 2026; the cached
  `inputs17_v5.npz` / `allfig17.npz` pin the exact figure inputs.
