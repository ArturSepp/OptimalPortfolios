---
myst:
  html_meta:
    description: >-
      Run offline matched-moment optimization, two-asset accounting and time-series
      covariance forecast comparisons with exact provenance and measured resource use.
---

# Matched portfolio-stack benchmarks

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Maintained in [OptimalPortfolios](https://github.com/ArturSepp/OptimalPortfolios).
Software citation: [CITATION.cff](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).
Accounting and analytics use [qis](https://github.com/ArturSepp/QuantInvestStrats)
([citation](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff));
factor estimation uses [FactorLasso](https://github.com/ArturSepp/factorlasso)
([citation](https://github.com/ArturSepp/factorlasso/blob/main/CITATION.cff)).

The [benchmark harness](matched_benchmarks.py) compares tightly specified problems.
It reuses the [analytics infrastructure](docs_analytics/README.md), OP's seeded factor
simulator, QIS's existing independent cash ledger and FactorLasso's public covariance
container. It is repository-only maintainer tooling. No competitor is a runtime
package dependency, and no production algorithm is replaced.

## Workloads and conventions

| Workload | Matched inputs and reference | Qualification |
|---|---|---|
| Minimum variance and quadratic utility | Same supplied mean, SPD covariance, long-only budget of one, cap of 2/N, and CLARABEL tolerances. Independent augmented KKT equations check objectives and feasibility. | Bounds are inactive in the reviewed fixtures; strict convexity establishes uniqueness before weights are compared. This does not cover binding, infeasible, mixed-integer or non-convex problems. |
| Six two-asset accounting cases | Identical prices and dated targets; independent QIS cash ledger checks entry, drift, rebalance costs, one-observation lag and an unquoted-position freeze. Two paths also have explicit hand-calculated NAV values. | Native skfolio return evaluation is segmented explicitly; lag is supplied by the harness. It does not become a holdings/cash execution engine. |
| Time-series OLS factor covariance | Four common observed factor returns, 504 training rows and 252 untouched future rows; same intercept OLS problem. Independent least squares and residual sums reconstruct each forecast. | A restricted OLS problem is used, with no shrinkage, clustering or tuning. It is not a characteristics-based cross-sectional model or a reproduction of a 5,000-asset timing claim. |

Inputs are synthetic simple returns in per-observation units. The existing generator uses
business-day observations with a 1/260 simulation step. It is reused unchanged, and caller
NumPy RNG state is preserved. No covariance is annualised in this harness. FactorLasso's
public `CurrentFactorCovarData` assembles the fitted loadings, population factor covariance
and fitted residual variances directly in per-observation units. Forecast scores use only
the reserved future rows.

Both optimization adapters use variance, with quadratic utility expressed as expected return
minus gamma/2 times variance. The skfolio risk-aversion setting is therefore gamma/2.
The fixed-prior adapter supplies the same matrices without re-estimating them. Native
formulations can differ while solving this same problem. Benchmark-only instrumentation at
CVXPY's solve boundary supplies and records identical CLARABEL settings: absolute gap,
relative gap and feasibility tolerances of 1e-10, maximum 500 iterations. The OP case selects its existing `factorize_covar=False` route; these are not default-mode
timings. skfolio's objective and constraint scale factors are explicitly one. OP's production
signature and defaults are unchanged. Every accepted solve and independently checked result
is retained; a solver fallback cannot silently count as a successful comparison.

Native accounting differences are retained. QIS holds units and charges costs on actual
trade dates, sizing targets from pre-cost wealth and funding costs through its cash balance.
skfolio Portfolio deducts its configured costs per observation; changing a fee parameter's
periodicity would be a different contract. With weight drift enabled, no-cost fully quoted
segments match QIS. Its default fixed-weight evaluation differs when weights drift.
The unquoted-position case exposes the difference between valuing missing quotes and
executing frozen positions. A native mismatch is labelled and measured, not hidden by
changing prices, dates, costs or expected values.

The factor covariance uses denominator T for both factor estimators. FactorLasso's residual
variance uses T; the pinned skfolio time-series model uses T-1. Each matrix is verified against
its own denominator, and their difference is checked against the independently reconstructed
diagonal residual adjustment. Held-out relative squared Frobenius loss, Stein-type covariance divergence
and equal-weight realised/forecast variance calibration are reported. The loss and projected
variance formulas are each checked by a second calculation.

## Setup and execution

Use the repository's external interpreter and generated-state setup from AGENTS.md.
Keep all environments, caches, source exports and results outside OneDrive. The OP checkout
provides this tooling. QIS and FactorLasso may be installed distributions or source exports:
the ledger is imported from QIS's shipped tests and covariance assembly uses FactorLasso's
public container. Both remain owned references, without copied numerical implementations.

The command below uses recorded C-local OP, QIS and FactorLasso source exports. Set `$opSource`,
`$qisSource` and `$factorSource` to those directories, or omit the QIS and FactorLasso source
paths from `PYTHONPATH` when using installed distributions. Use the existing governance
source-export workflow; do not stage unrelated edits to obtain a check. Effective package
versions and installed distribution metadata are recorded separately because an editable
environment can report a version different from the exported source actually imported.

The optional comparator snapshot is skfolio 1.9.3. Its additional isolated tool-layer pins are
scikit-learn 1.9.1, joblib 1.6.0, threadpoolctl 3.7.0, plotly 7.1.0, narwhals 2.27.1 and
cloudpickle 3.1.2. Install them into a C-local target rather than replacing the core environment:

```powershell
# First run Enter-AgentRepo.ps1 for the owning OP checkout in this shell.
$py = 'C:\Python\OptimalPortfolios312\Scripts\python.exe'
$optional = Join-Path $env:AGENT_LOCAL_ROOT 'runs\b0-optional-tools'
$pins = @(
    'skfolio==1.9.3', 'scikit-learn==1.9.1', 'joblib==1.6.0',
    'threadpoolctl==3.7.0', 'plotly==7.1.0', 'narwhals==2.27.1',
    'cloudpickle==3.1.2'
)
& $py -m pip install --target $optional --no-deps @pins
$env:PYTHONPATH = @("$opSource\src", $opSource, "$qisSource\src", "$factorSource\src", $optional) -join ';'
$env:MPLBACKEND = 'Agg'
Set-Location -LiteralPath $opSource
$output = Join-Path $env:AGENT_LOCAL_ROOT 'runs\b0-new-comparison'
& $py -m tools.matched_benchmarks --output-root $output --repeats 5 --sizes 10 100
& $py -m tools.matched_benchmarks --verify $output
```

The registered core environment supplies the native scientific stack; the execution evidence
records its exact versions and actual imported source paths. The optional layer's installation
reports retain wheel hashes. Package installation may use the network; generation imports,
fits, scoring and reference checks run under the existing analytics offline guard. The guard
prevents accidental Python network use; it is not an OS sandbox for hostile native code.
Each imported dependency retains its own licence; no third-party source is copied here.

## Outputs and measurements

The new C-local output directory contains `bundle.json`, raw worker records, numeric input CSVs for optimization and factor estimation, `runtime_summary.csv`, `forecast_scores.csv` and `accounting.json`. Accounting prices and targets are fixed in `accounting_cases()`; their
hashes and the computed ledgers are retained. Minimum-variance means are identically zero. It refuses to
overwrite an existing directory. The manifest records source and helper hashes, imported paths,
source versions versus installed distribution versions, solver options/status, seed, CPU,
RAM, actual native threadpools, input fingerprints and output hashes. `--verify` rejects missing
or changed results and malformed CSVs. No plots or documentation previews are generated.

There is one fresh worker process per case. Imports, data preparation, first fit and at least
three warm fits are separate fields; the default uses five warm fits and reports their raw
samples, median, range and quartiles. Warm repetitions create fresh estimator/solver objects.
Fit time includes each native adapter and required covariance assembly but excludes the
independent reference calculations. Fresh-process wall time includes startup and all validation.
A fresh process does not imply a cold operating-system/file cache. Process-lifetime peak resident memory includes imports, preparation, native allocations and
reference validation; it is not an isolated fit allocation measurement. BLAS/OMP/Numba worker
threads are set to one and the native library settings are recorded. Run timings without
concurrent workloads when making performance comparisons; retain the recorded samples and
hardware instead of promoting one machine's duration to a universal threshold.

Only the reviewed 10- and 100-asset workloads are accepted. Expansion to 1,000 or 5,000 assets
requires an explicit data/model contract and resource review; a similarly named factor model
does not establish workload equivalence. These fixtures do not establish investment
outperformance, institutional adoption or a comprehensive library ranking.

## Verification

Run the named source tests with the core environment first; optional comparator checks clearly
skip when its tool layer is absent. Then expose the pinned optional layer and run the same suite:

```powershell
& $py -m pytest src/optimalportfolios/tests/matched_benchmarks_test.py
& $py -m ruff check tools/matched_benchmarks.py tools/matched_benchmark_references.py src/optimalportfolios/tests/matched_benchmarks_test.py
```

The implementation's verification record includes controlled objective-scaling, lag and
residual-denominator mutations that fail the relevant test, followed by restored green runs.
Existing QIS accounting and FactorLasso feature-parity checks remain with their owners.
If installed OP metadata describes another revision, build the actual frozen candidate wheel
on C: before CI and verify its Python payload against the source export. Use its genuine
metadata rather than editing version strings or skipping the version-contract test.
Run the registered OP CI documentation and full-test profiles for the recorded candidate;
a focused benchmark pass does not replace those gates. No wall-clock performance threshold
is imposed in CI. Store benchmark bundles and execution evidence on C:, copying back only
intentional durable evidence under agents/. Q1/Q3 human trials are separate acceptance evidence.

## Primary sources

- [skfolio MeanRisk](https://skfolio.org/generated/skfolio.optimization.MeanRisk.html): prior moments, objective scaling, solver and fee periodicity.
- [skfolio Portfolio](https://skfolio.org/generated/skfolio.portfolio.Portfolio.html): fixed weights, weight drift and return/cost evaluation.
- [Pinned skfolio release](https://pypi.org/project/skfolio/1.9.3/): tested peer version; current unversioned docs may evolve.
- [QIS accounting oracle](https://github.com/ArturSepp/QuantInvestStrats/blob/main/src/qis/portfolio/tests/backtester_missing_prices_test.py).
- [FactorLasso canonical assembly](https://github.com/ArturSepp/factorlasso/blob/main/examples/docs/quickstart.py).
- [FactorLasso feature parity](https://github.com/ArturSepp/factorlasso/blob/main/benchmarks/feature_parity.py).
