---
myst:
  html_meta:
    description: >-
      Authoring rules for OptimalPortfolios methodology and guides: article structure,
      attribution, portable mathematics, executable examples, and analytical provenance.
---

# Documentation standard

*[author / affiliation / date — placeholder]*

This standard applies to [OptimalPortfolios](https://github.com/ArturSepp/OptimalPortfolios).
Software citation: [CITATION.cff](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).

Methodology articles define a concept, state its assumptions, explain its calculation, and then
describe its implementation. Use neutral, encyclopedic prose and primary references. Distinguish
published methods from package conventions. This standard adapts the documentation approach
used for [qis](https://github.com/ArturSepp/QuantInvestStrats) to portfolio
construction and the existing OptimalPortfolios documentation tree.

## Article structure

A methodology article has one H1, the visible byline and software links, a short definition-led
introduction, and these H2 sections in order. Use descriptive H3 subsections within them.
Preserve existing section links with explicit anchors when reorganizing an article.

| Section | Required content |
|---|---|
| Overview | The question the method answers, its scope, and when to use it. |
| Inputs, notation, and assumptions | Symbols, dimensions, units, frequencies, data policies, and timing. |
| Methodology | Definitions and equations, followed by their interpretation. |
| Worked example | Fixed inputs, a small result, and what it establishes. |
| Implementation in optimalportfolios | Verified public entry points, input/output contract, runnable source, and verification context. |
| Interpretation and limitations | Assumptions, numerical qualifications, edge cases, and unsuitable uses. |
| See also | A small set of useful related methods and guides. |
| References | Verified primary method sources and the software citation. |

Installation, quickstart, navigation, architecture, package comparison, gallery, and contributor
guides use a **utility form**. They retain one H1, the byline, project/citation links, a useful page
description, and a logical heading hierarchy. They do not need empty methodology sections.

The human-maintained API entry remains `api.rst`; its autosummary inventory and generated pages
retain their own format. Build output and autosummary templates are not authoring locations.

## Copyable methodology template

Replace topic placeholders with actual content. Keep the author placeholder until the author
supplies the details. Record a tested version only after checking the imported package and source:
an installed release, checkout metadata, and uncommitted code can describe different states.

````markdown
---
myst:
  html_meta:
    description: >-
      [A factual description of the method and its OptimalPortfolios implementation.]
---

# [Method or analytical concept]

*[author / affiliation / date — placeholder]*

Implemented in [OptimalPortfolios](https://github.com/ArturSepp/OptimalPortfolios).
Software citation: [CITATION.cff](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).

[Define the concept and its scope.]

## Overview

[Purpose and appropriate uses.]

## Inputs, notation, and assumptions

[Symbols, units, dimensions, frequencies, missing-data policy, and timing.]

## Methodology

[Introduce each equation and explain its meaning.]

## Worked example

[Fixed inputs, result, and interpretation. Label synthetic data explicitly.]

## Implementation in optimalportfolios

[Verified entry points, ordinary source links, reproduction command, and source/version context.]

## Interpretation and limitations

[Assumptions, numerical qualifications, and edge cases.]

## See also

[Related methods and guides.]

## References

- [Verified author, year, title, venue, and DOI or primary-source link.]
- [OptimalPortfolios software citation](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).
````

For supplied details use `*Author: [name] / Affiliation: [affiliation] / Date: YYYY-MM-DD*`.
Author date, last substantive review date, data cutoff, and image generation time are separate
facts. A build must not silently claim that a method was reviewed again.

## Portable mathematics

Supported targets are GitHub Markdown, MyST/Sphinx, and VS Code's built-in Markdown preview.
Use `$...$` inline and standalone `$$` delimiters with blank lines around display blocks.
MyST's `dollarmath` extension is enabled in the site configuration. Basic CommonMark viewers may
show TeX source; retain ordinary links to the rendered site for readers using those viewers.

For example, let $b_i$ denote a nonnegative fractional risk budget for asset $i$ among $n$ assets:

$$
\sum_{i=1}^{n} b_i = 1, \qquad b_i \geq 0.
$$

Four equal fractional budgets are each $1/4$. These are target budgets; this identity alone does
not calculate portfolio weights or establish that realized risk contributions match the targets.

The source for the display equation is:

````markdown
$$
\sum_{i=1}^{n} b_i = 1, \qquad b_i \geq 0.
$$
````

- Keep semantic equations out of code spans and code fences. Fences are appropriate for showing
  source, as above. Do not use fenced `{math}` directives or `{math}`/`{eq}` roles in article prose.
- Use supported `aligned`, `cases`, or matrix environments inside a display block where needed.
  Split long expressions at meaningful equalities; avoid custom macros.
- Define symbols before use, with one meaning per symbol and explicit time subscripts.
- Link to ordinary section headings instead of renderer-specific equation labels.
- Keep complex formulas out of table cells. Use `\lvert`/`\rvert` and `\lVert`/`\rVert` for
  absolute values and norms where a raw pipe could be parsed as a table separator.
- Write currency as `USD 100` or `CHF 100` near math. Do not globally unescape source files.
- Check dimensions, signs, timing, covariance scale and normalization against the owning
  implementation. A delimiter repair must not alter mathematical meaning.

The underlying syntax is documented by
[GitHub](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions),
[MyST](https://myst-parser.readthedocs.io/en/latest/syntax/math.html), and
[VS Code](https://code.visualstudio.com/docs/languages/markdown#_math-formula-rendering).
Inspect each target independently; a successful Sphinx build does not establish GitHub or
VS Code rendering. Record unperformed viewer checks as pending.

## References and implementation ownership

Cite methodological claims where they appear and give full bibliographic entries under
References. Verify author names, dates, titles and DOIs against primary sources. A paper
citation does not identify the software used to produce an exhibit; a software citation does
not replace a mathematical source. Link to `CITATION.cff` rather than repeating release metadata.

Every article cites OptimalPortfolios. If calculations use
[qis](https://github.com/ArturSepp/QuantInvestStrats) or
[factorlasso](https://github.com/ArturSepp/FactorLasso), identify that delegation and include the
[qis citation](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff) or
[factorlasso citation](https://github.com/ArturSepp/FactorLasso/blob/main/CITATION.cff) as appropriate.

OptimalPortfolios owns construction and rolling orchestration. Generic sparse factor estimation
and HCGL belong to factorlasso; holdings simulation, reporting and analytics belong to qis.
Use the existing risk-model integration for tracking-error examples. Do not introduce another
backtester or statistics layer to make a chart.

The public interface is defined by re-exports and the existing public-API tests. Verify signatures
and enum members before writing examples; do not add a parallel export inventory. The complete
[constraints contract](constraints.md) stays authoritative. Source-adjacent READMEs describe
contributor/module workflows and link to public methodology instead of duplicating it.
Do not assume repository Markdown notes are included in installed wheels.

Keep runnable Python examples canonical and provide an ordinary source link beside Sphinx
includes. Preserve the quickstart's executed README and Python/notebook parity checks. Label
fragments with missing context as illustrative; do not present them as standalone commands.

## Analytical conventions and figures

State simple versus log returns, observation and estimation frequencies, covariance units,
annualization, decision and implementation dates, warmup, and missing-data policy where relevant.
Distinguish target weights from drifted holdings; target turnover from realized trades; gross
from net performance; and zero-rate from excess-return Sharpe conventions.

Explain constraints, solver acceptance, compliance, fallback use and tolerances where these
affect interpretation. Do not imply that substituting an objective leaves every input unchanged.
Keep synthetic teaching examples separate from historical empirical or paper-replication claims.

Each displayed analytical image needs an identifiable producer, fixed input/configuration
identity, sample period, actual source version and generation record. Captions explain the result;
alt text describes the comparison. Inspect labels, legends and tables at normal page width and
full resolution. Keep strategy colors and units consistent across comparable exhibits.

The six README previews are offline synthetic teaching exhibits. Their shared
[provenance record](../examples/figures/analytics_manifest.json) records the input/configuration
identity, actual software environment, generation timestamp and separate visual review. The
[analytics registry](../tools/docs_analytics/registry.json) now records the six stable preview
paths, their four producer families, and eight non-analytics badges. The
[analytics runner](../tools/docs_analytics/run.py) checks coverage without importing those scripts.
Generation, validation and publication tooling are available. All four producer families now
have offline implementations. Complete-bundle generation validates six PNGs and their supporting
CSVs; publication still requires the separate visual-review record.

<a id="analytics-registry-and-refresh-planning"></a>

### Analytics registry, planning and generation

After the repository environment setup, run from a C-local source export:

```console
python -m tools.docs_analytics.run --list
python -m tools.docs_analytics.run --plan
python -m tools.docs_analytics.run --plan --output-root <new-C-local-plan-directory>
python -m tools.docs_analytics.run --all --output-root <new-C-local-run>
python -m tools.docs_analytics.validate --run-root <existing-C-local-run>
```

Listing checks displayed-image coverage across the root README, human Markdown/RST pages and
source-adjacent READMEs. Every image needs a registered preview path or an explicit non-analytics
classification. Generated Sphinx trees and literal code examples are excluded. The parser covers
ordinary Markdown and reference images, HTML images, and MyST/RST image directives; it is a
source check, not a browser renderer.

A plan records effective source hashes, legacy image hashes, installed distribution metadata,
producer ownership, fixed fixtures and any outstanding generation work. The saved file is
`run_plan.json` with `status="planned"`. Its `generation_ready` flag reflects whether every
producer has an implemented entry point and a complete configuration; it does not prove execution.
Plans always have `publication_ready=false`. A legacy image hash identifies bytes without
recovering the historical inputs that produced them. Listing and planning use only the standard
library and do not import analytical packages.

The executor and [bundle validator](../tools/docs_analytics/validate.py) are implemented.
The current registry's `--all` command executes all four implemented families. If any family
is changed back to pending, generation exits with code 2 before creating output.
Every family has a fixed configuration and a registered `produce(spec)` function.
A `configuration: null` entry is an unresolved decision, not an implicit choice of defaults.

Use a new output directory below `AGENT_LOCAL_ROOT`, outside the source tree and OneDrive.
Generation refuses existing destinations, links and junctions. All producers run in one new
sibling staging directory. The executor finalizes it at the requested path only after the whole
bundle passes read-back validation. Failed staging directories retain `FAILED.json` and any
partial outputs for diagnosis; they have no completion manifest. Use a fresh destination to retry.

### Portfolio-report family preview

The [portfolio-report producer](../tools/docs_analytics/portfolio_reports.py) implements the
three report candidates. Generate and inspect this family independently during development:

```console
python -m tools.docs_analytics.portfolio_reports --output-root <new-C-local-preview>
```

This command writes three PNGs, twelve CSV tables and `family_manifest.json`. It applies the
same output boundary, offline guard, result checks, source/dependency fingerprints and PNG/CSV
read-back checks as the complete runner. Its record has
`kind="documentation_analytics_family_preview"` and `publication_ready=false`. It cannot be
passed to the complete-bundle publisher. Failed runs retain `FAILED.json` in staging.

The fixed teaching baseline selects US/European equity, Treasuries, investment-grade bonds,
gold and commodities from `qis.datasets.generate_synthetic_universe`. The generator is unchanged:
seed 20260725, business-day sample 4 January 2010 through 31 December 2025, clean mode
`apply_quirks=False`. Clean mode deliberately excludes the missing/stale/delisted-price scenarios
to focus these report panels on allocation and implementation. The CSV inputs freeze the exact
panel and the fixture's daily-rebalanced gross 60/40 benchmark; these are synthetic paths.

OptimalPortfolios estimates annualized covariance from weekly Wednesday log returns, with
trailing EWMA demeaning, span 52 and annualization factor 52. Fully invested, long-only
maximum-diversification targets have a 35% cap per asset. The production SLSQP wrapper uses
`ftol=1e-8` and `maxiter=500`; the producer verifies these effective defaults before solving.
Diagnostics retain every native status and hard-constraint residual. SLSQP uses the raw EWMA
matrix, so the record explicitly states that covariance factorization is inapplicable.

The warmup starts in 2010; decisions are requested from 2015 through 1 October 2025. With this
weekly grid, the quarterly schedule produces 43 decisions from 1 April 2015 through
1 October 2025. Quarter boundaries map to the next weekly Wednesday observation. Each target
trades one business-day observation later; the exact mapping is saved in `decision_schedule.csv`.
qis holds units between trades and charges 10 bp on gross traded notional, including entry.
Funding, carry and management fees are absent. The report ends on 31 December 2025.

| Candidate | Readable panels |
|---|---|
| `example_portfolio_factsheet1.PNG` | Net portfolio and gross synthetic benchmark growth of 100; drawdowns |
| `example_portfolio_factsheet2.PNG` | Last decided target weights; contributions to annualized volatility at that decision |
| `example_customised_report.PNG` | Target allocation through time; quarterly sums of daily cost divided by same-day NAV |

The risk snapshot describes the last decision, not the ending drifted holdings. The CSVs also
retain daily units, realized weights, costs, turnover, NAVs, drawdowns, performance and the last
covariance matrix. The performance table uses qis's `SharpeConvention.PA`, compounded annual
return divided by volatility annualized from weekly log returns, with a zero risk-free rate.
Quarterly cost sums are descriptive; they are not compounded performance drag. The comparison
with a gross benchmark is illustrative and does not establish historical strategy superiority.

Seven recorded checks cover finite inputs/results, weights, accepted solves, point-in-time
covariance at the first/middle/last decision, actual implementation dates, costs reconstructed
from unit changes, and the opening trade. Tests additionally perturb future prices, rerun
earlier targets, and compare risk contributions and the initial holding period with direct
arithmetic. Repeated runs compare every image and table hash in the same recorded environment.
This does not establish identical floating-point output across dependency versions or platforms.

### Span-sensitivity family preview

The [span-sensitivity producer](../tools/docs_analytics/span_sensitivity.py) compares the fixed
weekly EWMA spans **5, 13, 26, 52 and 104**. It uses the same six clean synthetic assets,
seed, sample, 35% target cap, quarterly schedule, one-business-day implementation lag and
10 bp trading costs as the [portfolio-report baseline](#portfolio-report-family-preview).

```console
python -m tools.docs_analytics.span_sensitivity --output-root <new-C-local-preview>
```

The command writes `max_diversification_span.PNG`, thirteen CSVs and `family_manifest.json`
through the shared preview writer. The record remains non-publishable until the complete
six-asset bundle is available and reviewed.

Only the declared EWMA span varies. Following the existing estimator, that span controls both
trailing return demeaning and covariance smoothing. A span is neither a hard lookback window
nor a half-life; labels use weekly observations rather than approximate month/year names.
Every case uses annualization factor 52 and the same 43 decision/implementation dates, for
215 solves overall. The 52-week case reproduces the portfolio-report baseline.

The two panels compare **net growth of 100** and **full-period trading costs**. Cost bars sum
daily costs divided by same-day NAV and convert the result to basis points. These sums include
entry costs and are descriptive, not compounded performance drag. All portfolios are net;
the synthetic 60/40 benchmark is retained in the common input snapshot but is not plotted here.

The CSVs retain shared prices, benchmark prices and the decision schedule; per-span targets,
realized weights, units, last covariances, volatility contributions and daily trading records;
and comparison NAVs, drawdowns, performance and cost totals. Per-span tables use a labelled
`span` index. Performance uses qis `SharpeConvention.PA` with a zero risk-free rate.

Ten comparison checks cover identical inputs, decision schedules and report samples; every
case's seven numerical checks; actual per-case configurations; the complete span/date solver
grid; distinct covariance and weight results; finite outputs; and cost totals. Provenance
retains all 215 native solver records, constraint residuals, and the configuration/checks
for each case. Missing solves, fallback results or silently reused spans fail generation.
Tests independently reconstruct all five covariance scales through weekly qis kernels,
reconstruct costs from traded units, and perturb future prices for the fastest and slowest spans.

Interpret the figure as a sensitivity analysis on one frozen synthetic path. No span is selected
using the plotted results, and the exhibit does not establish an optimal span or expected
market performance. Keeping sample dates fixed and saving input bytes makes subsequent
comparisons reviewable; changing the fixture or environment establishes a new baseline.

### Optimizer-comparison family preview

The [optimizer-comparison producer](../tools/docs_analytics/optimiser_comparison.py) compares
**minimum variance, maximum diversification and equal risk budgets** on the shared
[portfolio-report baseline](#portfolio-report-family-preview):

```console
python -m tools.docs_analytics.optimiser_comparison --output-root <new-C-local-preview>
```

This writes `multi_optimisers_backtest.PNG`, fourteen CSVs and `family_manifest.json`.
It is a C-local candidate with `publication_ready=false`; publication requires the complete
reviewed six-image bundle.

All objectives share six synthetic assets, 43 quarterly decisions, weekly Wednesday log
returns, EWMA span 52 for trailing demeaning and covariance smoothing, and annualization
factor 52. Constraints require full investment, long-only holdings and a 35% asset cap.
Implementation follows each decision by one business day, with 10 bp costs on traded
notional including entry. The report runs from 1 April 2015 to 31 December 2025; the final
target is decided on 1 October 2025.

| Objective | Native backend | Additional input and interpretation |
|---|---|---|
| Minimum variance | CVXPY / CLARABEL, default options | No expected-return estimate |
| Maximum diversification | SciPy SLSQP, `ftol=1e-8`, `maxiter=500` | Volatilities from the shared covariance |
| Equal risk budgets | Internal constrained ADMM-CCD, default options | Equal positive budgets of 1/6; caps can prevent equal realized risk contributions |

Explicit covariance factorization is disabled for this comparison. Every covariance must be
positive definite and identical across objectives. The risk-budget wrapper's variance floor
is checked to be inactive: every variance is at least `1e-6`. Solver options and dependency
source/version hashes are recorded. Risk-budget acceptance means the package validator
accepted its output; no iteration count or convergence flag is invented.

The manifest retains all **129 native solve records**, backend/status, decision date,
constraint residuals, fallback state and the explanation for unused factorization. Each case
retains the seven portfolio-baseline checks. Twelve comparison checks enforce common inputs,
covariances, report/trade schedules, configurations, a complete objective/date grid, distinct
allocations, finite outputs, cost totals, positive definiteness and an unused variance floor.
Rejected, noncompliant or fallback results stop generation.

CSVs preserve shared prices, benchmark prices, risk budgets, decision dates and the final
covariance; objective-labelled targets, realized holdings, units, volatility contributions
and trading; and comparison NAVs, drawdowns, performance and cost totals. Performance uses
qis `SharpeConvention.PA` with zero risk-free rate. The lower panel sums daily costs divided
by same-day NAV in basis points; this is not compounded performance drag.

This **covariance-only teaching comparison** has narrower coverage than the legacy
six-objective live-data example. Quadratic utility and maximum Sharpe also estimate expected
returns; CARA mixture fits its own rolling mixture distribution. They need a separate
estimation comparison. The initial fixed-sample trial of the legacy maximum-Sharpe path
produced eight infeasible solves, so its fallback holdings are not presented as successful
optimized allocations. No sample, seed, mean estimate or solver behavior was changed to make
those results appear successful.

Independent tests compare minimum variance to a separate SciPy quadratic formulation,
constrained risk budgets to a cone-constrained logarithmic objective, and costs to traded
units. Future-price perturbations and truncated reruns verify earlier targets. The chart is
illustrative; it does not select an optimizer or establish expected market performance.

### Covariance-comparison family preview

The [covariance-comparison producer](../tools/docs_analytics/covariance_comparison.py) compares
six estimators with the **same minimum-variance objective**:

```console
python -m tools.docs_analytics.covariance_comparison --output-root <new-C-local-preview>
```

The [existing factor-return simulator](../examples/covar_estimation/simulate_factor_returns.py)
is the explicit teaching-fixture exception to the QIS market-panel baseline. It is unchanged:
seed 42, four factors, eight assets, 783 business-day observations from 2 January 2023 through
31 December 2025, factor volatility 0.15, idiosyncratic-volatility range [0.01, 0.15],
beta range [-1.5, 1.5], generated factor correlation and a daily time step of 1/260.
The producer restores the generator's global NumPy random-state side effect after the call.

Gaussian increments are interpreted as **log returns**. QIS converts them to prices starting
at 100; the first increment is replaced by the initial price observation. Original increments,
loadings, idiosyncratic scales and known covariance are saved. Known loadings/covariance are
used for descriptive verification, never as inputs to estimated portfolios.

| Estimator | Volatility normalization | Regression configuration |
|---|---|---|
| EWMA | None | No factor regression |
| EWMA vol norm | Asset-return covariance | No factor regression |
| Lasso | None | LASSO penalty 1e-6 |
| Lasso factor vol norm | Factor covariance only | Same LASSO fit inputs and penalty |
| Group Lasso | None | Group penalty 1e-5; fixed asset pairs |
| Group Lasso factor vol norm | Factor covariance only | Same group fit inputs and penalty |

The four fixed groups are assets 1–2, 3–4, 5–6 and 7–8. They are declared teaching groups,
not clusters inferred from the known loadings. Group penalties use the normalized convention
with zero additional L1 weight. Factorlasso fits unconstrained signed coefficients with
trailing demeaning, span 52 and at least 52 weekly observations; sign constraints,
adaptive penalties and solver fallback chains are disabled.

Every case uses weekly Wednesday log returns, EWMA span 52 and covariance annualization
factor 52. Span is not a half-life. Factor models assemble the estimated asset covariance
from fitted loadings, factor covariance and the full diagonal residual variance. The known
daily covariance is annualized by 260; both estimates and truth therefore have annual units.
Volatility normalization restores the return scale in the resulting covariance.

The public rolling estimators use different date conventions. This comparison takes the
EWMA weekly schedule as a common clock and calls each estimator's public current-fit method
on data truncated to those same dates. There are eight decisions, from 3 January 2024 to
1 October 2025. Each target trades one business-day observation later, with full investment,
long-only weights, a 35% asset cap and 10 bp costs on traded notional including entry.
QIS holds units between trades; the report ends on 31 December 2025.

All allocation and factor-regression solves use CLARABEL through CVXPY, with native default
options. Explicit covariance factorization is disabled and every estimated covariance must
be positive definite. The manifest retains **48 allocation solves and 32 factor fits**,
native statuses, constraints or an explicit absence of hard regression constraints, fallback
state, and covariance-factorization explanations. Factor-model results do not expose solver
status themselves: scoped instrumentation observes the unchanged CVXPY solve and restores
the method even on failure. This producer runs those fits sequentially.

The command writes `MinVariance_multi_covar_estimator_backtest.PNG`, **23 CSVs** and a
non-publishable family manifest. CSVs include simulated inputs/truth/groups, every estimated
covariance, factor loadings/covariances/residual variances, decided and realized holdings,
units, NAVs, drawdowns, performance, trading, implementation dates and covariance errors.
Performance uses QIS `SharpeConvention.PA` with zero risk-free rate.

The upper panel shows net growth of 100. The lower panel shows relative Frobenius error
at the final decision: the square root of summed squared covariance errors divided by the
square root of summed squared true covariance entries. This weights the entire matrix by
squared entry size; it is not an out-of-sample risk forecast score or investment ranking.
The simulation has constant factor loadings and covariance, no return premium and one fixed
sample path. The chart makes no expected-performance claim.

The old example's two factor “VolNorm” pairs used identical settings. This producer explicitly
sets the normalization option and checks distinct estimates and targets. Fourteen checks cover
complete solve grids, shared dates, finite inputs/results, covariance definiteness,
implementation, costs, initialization, warmup and simulation/error identities. Tests independently
reconstruct covariance units/components and holdings costs, alter future observations and known
truth, and verify that solver instrumentation is restored.

With all four producer families implemented, the complete `run --all` command generates six
PNGs and 62 CSVs in one validated bundle. Two runs must agree in the recorded environment.
Family manifests remain insufficient for publication; use the complete-bundle review and
publication procedure below.

### Producer contract and provenance

The registry's `configuration` object requires these fields. No field is inferred from a
legacy script's defaults.

| Field | Required record |
|---|---|
| `fixture` | Factory description and source-relative `source_files` whose bytes are hashed |
| `parameters` | Explicit producer options, including estimator settings and any grid |
| `conventions` | Data kind, sample start/end, seed, universe, missing-data policy, return convention, observation and estimation frequencies, annualization, warmup, rebalance frequency, implementation lag, transaction costs, and Sharpe convention |
| `tables` / `input_tables` | Expected table names and the nonempty subset that snapshots the actual inputs |
| `checks` | Named numerical checks that the producer must calculate and pass |
| `rendering` | Explicit DPI and font family; execution records the resolved font file and hash |
| `solver` | Whether solves are required and the permitted backend names with their options |
| `dependencies` | Top-level import names mapped to installed distribution names |

Use ISO dates for the sample interval, `simple` or `log` for returns, and a nonnegative
observation count for implementation lag. Explicit `null` is allowed for a deterministic
fixture's seed and an inapplicable annualization factor; `not_applicable` is allowed for returns
when an exhibit does not use returns. Other conventions require explanatory strings.
Table/check identifiers are unique lowercase names such as `input_prices` and `weight_sum`.

`produce(spec)` returns a dictionary with exactly `figures`, `tables`, `configuration`,
`checks` and `diagnostics`. Figures map the producer's registered PNG basenames to Matplotlib
`Figure` objects. Tables map every declared name to a nonempty pandas `DataFrame`.
All index and column labels must be nonempty, distinct strings. Input tables must contain the
data actually used, and the echoed configuration must describe the settings actually applied.
The executor saves CSVs with 17 significant digits and explicit index labels, then reads them back.

Every named check must be the Boolean `True`. Producers must compute appropriate checks through
existing OptimalPortfolios/qis APIs and independent references where needed; a success flag alone
does not establish numerical correctness. The executor validates these records and their
consistency, rather than recomputing each producer's financial analysis.

`diagnostics.solves` records every actual solve, with backend, context, normalized status
(`optimal`, `optimal_inaccurate` or `success`), acceptance, constraint compliance, fallback source,
constraint residuals and covariance stabilization. Preserve native solver details as additional
fields. Fallback results and rejected/noncompliant solves fail validation. A residual records
`name`, `hard`, `passed`, nonnegative `violation` and `tolerance`; its pass flag must agree with
the comparison, and every hard constraint must pass. Covariance records state `factorized` and
the nonnegative `n_floored` count, or explain why factorization was inapplicable. Explain an
empty residual list with `constraints_not_applicable`. An empty solves list is allowed only when
`solver.required=false` and `diagnostics.not_applicable` explains why.

A completed candidate contains:

```text
images/<six registered PNG basenames>
tables/<producer>/<declared table>.csv
analytics_manifest.json
```

The manifest records generation time in UTC separately from the sample cutoff, the complete
registry/configurations, effective source hashes (including uncommitted edits), declared fixture
file hashes, actual input-table snapshots, imported dependency versions and origins, first-party
Python source identity, rendering/font identity, producer checks/diagnostics, and every output's
hash and dimensions or table shape. Matplotlib, NumPy, pandas and Pillow are always recorded;
producers declare their other dependencies, including OptimalPortfolios, qis, factorlasso and
solver backends as applicable. When OptimalPortfolios is declared, generation requires its import
to come from the selected source export.

Python socket/DNS entry points are blocked during imports, execution and validation. This catches
accidental network use by trusted producers; it is not an operating-system sandbox for arbitrary
subprocesses or native-library traffic. Producers must use offline fixtures and existing stack
APIs. Source, inputs and imported dependency content are checked again after execution.

The validator rejects missing or extra files/directories, linked paths, changed inputs/source,
environment or font drift, inconsistent diagnostics, modified output, malformed CSVs, and blank
or undersized PNGs. `status="complete"` means that integrity validation succeeded.
`review_status="pending"` and `publication_ready=false` remain explicit: visual inspection and
explicit publication are still required. Hashes detect drift against recorded provenance;
they do not authenticate a deliberately rewritten manifest or prove that an honest-looking check
was calculated correctly.

The utilities adapt the MIT-licensed
[qis documentation-analytics tooling](https://github.com/ArturSepp/QuantInvestStrats/tree/main/tools/docs_analytics)
locally. They do not import repository-only qis tools as installed functionality.
The [registry tests](../src/optimalportfolios/tests/documentation_analytics_registry_test.py)
cover inspection without analytical imports. The
[bundle tests](../src/optimalportfolios/tests/documentation_analytics_bundle_test.py) exercise
all six asset names with controlled test producers, repeated output, failure containment and
provenance rejection. Real-producer checks and the dated visual-review receipt provide
separate evidence for the financial exhibits in the [analytics gallery](analytics_gallery.md).

### Reviewed preview publication

The [publisher](../tools/docs_analytics/publish.py) accepts only the six existing preview paths
in `examples/figures/` plus `examples/figures/analytics_manifest.json`. It requires a complete,
validated bundle and a separate visual-review record. Real complete-bundle generation is verified;
publication tests also use controlled images in disposable repositories.

Run these commands from the C-local source export after the repository environment setup.
PowerShell line continuations keep the commands readable:

```powershell
python -m tools.docs_analytics.publish --repo-root <checkout> `
    --run-root <bundle> --prepare-review <new-C-local-review-directory>
python -m tools.docs_analytics.publish --repo-root <checkout> `
    --run-root <bundle> --review-file <review-directory>/review.json
python -m tools.docs_analytics.publish --verify --repo-root <checkout>
```

The first command validates the bundle and writes a **pending** `review.json` outside it.
It records the bundle's content identity and all six image hashes. It leaves the reviewer and
review date unset, with `full_resolution` and `article_width` false for every image.
Inspect each image in both views, checking labels, units, dates, legends, clipping and
readability. Then enter the actual `reviewer` and a UTC ISO `reviewed_at_utc` timestamp, set the checks to
true, and change `status` to `reviewed`. Preparation never attests that this inspection happened.
Do not edit the hashes or the immutable generation manifest to force a match: a changed bundle
needs a new review. This record is an explicit attestation, not cryptographic authentication
of the reviewer.

Publication revalidates the bundle against the destination checkout's current source, fixture
files, registry and execution environment. The source export and destination may have different
absolute paths, but their effective source bytes must match. It rejects incomplete or stale
review, widened asset paths, wrong-case destinations, links/junctions and directory collisions
before changing any destination. Only PNGs and the shared publication receipt are copied;
supporting CSVs, review working files, full factsheets and PDFs remain C-local.

The receipt wraps the unchanged generation record with the completed review, publication time
and exact preview paths. The candidate keeps `review_status="pending"` and
`publication_ready=false`; the separate receipt identifies the reviewed publication.
`--verify` checks displayed PNG bytes against this dated receipt and the current coverage
registry. It does not rerun analytics, read the retained CSVs, or certify that an older exhibit
matches today's source/environment. Full bundle validation is required before publishing again.

### Publication recovery

Before any replacement, publication prints the path of a C-local `publication-backup-...`
directory. It retains the previous bytes (including absence of an earlier manifest), proposed
bytes and a recovery journal. The publisher uses a same-machine operating-system lock and
same-volume replacement for each individual file. It checks for source and destination drift
again during publication and writes the shared manifest last.

Seven separate replacements are **not one filesystem transaction**. A reader or OneDrive sync
can observe a mixed set during copying. A normal exception triggers automatic rollback; a
process kill, power loss or failed restoration can require recovery. If rollback cannot finish,
the command reports failure and retains the backup. Do not delete the backup or assume the
displayed files form a complete publication.

Recover an interrupted copy, or explicitly undo a completed publication, with:

```powershell
python -m tools.docs_analytics.publish --repo-root <same-checkout> `
    --rollback <printed-C-local-backup>
```

Recovery checks all backup hashes, exact target paths and current destination bytes before
writing. It restores the images first and the old manifest (or its absence) last, then verifies
every original byte. Repeating recovery is safe when the previous state is already restored.
A later edit that matches neither the saved old nor proposed bytes blocks recovery; reconcile
that conflict before retrying. A corrupted backup also blocks recovery. The command does not
overwrite another session's conflicting changes.

The machine lock releases when its process exits; its reusable C-local lock file can remain.
It does not coordinate two computers using a OneDrive checkout. The existing prohibition on
concurrent cross-machine work still applies. Backups live on C and are not a guarantee against
disk loss; keep the durable repository and reviewed deliverables under the repository's
existing OneDrive policy.

The [publication tests](../src/optimalportfolios/tests/documentation_analytics_publication_test.py)
exercise exact publication scope, invalid review/bundles, failures before and after individual
replacements, rollback failure/retry, conflicting edits, and recovery in a fresh process after
a forced process exit. The [analytics gallery](analytics_gallery.md) displays the reviewed
real-producer bundle; its linked receipt records the actual generation and review dates.
Every replacement still requires a complete validated bundle and a separate visual review.

Use the frozen qis synthetic universe for new market-panel teaching examples where appropriate.
Preserve the established OptimalPortfolios multiasset fixture and existing simulation seeds.
A live-data refresh must be explicit and retain its input snapshot: a fixed end date alone does
not freeze a provider's adjusted history.

## Verification and migration

`tools/check_docs.py` checks article metadata, visible attribution/citation links, heading
structure, portable math delimiters and local file links without importing the numerical stack.
`tools/docs_inventory.json` explicitly classifies adopted pages, pending migrations and the API
entry. New reader-facing pages must enter this inventory; omitted pages are errors.

After the mandatory repository environment setup, use the prescribed external interpreter:

```console
python tools/check_docs.py
python tools/check_docs.py --files docs/documentation_standard.md
python tools/check_docs.py --source-all
python tools/check_docs.py --all
```

The default checks adopted pages and reports pending migrations. `--files` checks a selected
human-page batch regardless of its adoption status. `--source-all` validates every human-authored
source, including pending pages, without changing the inventory or claiming viewer acceptance.
It still rejects legacy human RST, malformed prose and missing local targets. `--all` is the
final migration gate and fails while pending pages remain. Legacy RST pages retain explicit
inventory entries until converted; do not leave both Markdown and RST sources with the same basename.

These checks do not prove mathematical correctness, external-link availability, bibliography
accuracy or rendering. Run relevant examples/tests and strict Sphinx HTML/link checks separately.
Use a C-local source export: autosummary writes generated source files, so redirecting only the
HTML output directory is insufficient for a OneDrive checkout.

When renaming a page, preserve its basename and HTML address, record existing anchors, and update
source/download links. Inspect rendered formulas after the math engine finishes. Keep numerical
changes and unsupported empirical claims out of cosmetic edits.

### Automated documentation checks

The [documentation workflow](../.github/workflows/docs.yml) runs on changes to articles, README,
examples and their previews, source/fixtures, documentation tooling, attribution, dependency
metadata or build configuration. It uses Python 3.12 and uv 0.12.13 with
`uv sync --locked --extra docs --group test`. The test group adds checker tests; the numerical
and Sphinx dependencies come from the reviewed `uv.lock`. Subsequent `uv run --no-sync` commands
use that environment without resolving again. The separate package CI retains public-API,
optional-dependency and installed-wheel tests.

The job checks all human sources, exercises checker regressions, validates displayed-image
registration and the dated preview receipt, generates and validates all four offline producers,
and runs strict Sphinx HTML and external-link builds. Link checks use one worker to reduce
concurrent requests to citation/source hosts; warnings and failed links remain fatal.
Environments, analytics candidates, caches
and HTML/linkcheck output use the ephemeral runner's temporary directory. Autosummary may write
generated sources in that disposable checkout. The workflow never publishes its analytics
candidate or marks a visual review complete.

These gates answer different questions:

| Gate | What a pass establishes |
|---|---|
| `check_docs.py --source-all` | Every inventoried human source meets the source standard; pending reviews stay explicit. |
| `tools.docs_analytics.run --list` | Every displayed image is registered, including its document consumers. |
| `tools.docs_analytics.publish --verify` | Committed previews match their dated review receipt and current coverage registry. |
| `tools.docs_analytics.run --all` | Current source and locked dependencies generate a complete, internally validated offline bundle. |
| Strict Sphinx HTML and linkcheck | The site builds without warnings and external links pass the configured link policy. |

Preview verification detects changed or missing image bytes relative to the receipt. It does
not assert that an older publication used today's source. Regeneration validates the current
candidate; it does not compare PNG bytes across operating systems, install that candidate or
replace mathematical/visual review. Follow the complete review/publication procedure above to
refresh the displayed bundle after source changes.

[Read the Docs configuration](../.readthedocs.yaml) uses the same Python minor version, uv version,
lockfile and `docs` extra, with `UV_PROJECT_ENVIRONMENT` pointing to its managed environment.
Its custom install uses `--locked` so inconsistent dependency metadata fails instead of silently
revising the lock. Source, image-coverage and dated-preview checks run before its strict Sphinx
build. Offline regeneration and the external-link gate run in GitHub Actions; hosting serves the
reviewed committed previews. Build-system bootstrap packages, operating-system libraries and
the Python patch version are not fully pinned by the application lockfile.

This setup follows the documented [Read the Docs build-job customization](https://docs.readthedocs.com/platform/stable/build-customization.html) and
[uv project environment and lock controls](https://docs.astral.sh/uv/concepts/projects/config/).
On this Windows host, continue to use the prescribed external interpreter and a C-local source
export after the mandatory repository setup; runner commands are not a replacement for that policy.

## See also

- [Documentation home](index.md)
- [Constraints and solver contracts](constraints.md)
- [Software design](software_design.md)
- [Contributor guidance](https://github.com/ArturSepp/OptimalPortfolios/blob/main/AGENTS.md)

## References

- [OptimalPortfolios software citation](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).
- [qis software citation](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).
- [factorlasso software citation](https://github.com/ArturSepp/FactorLasso/blob/main/CITATION.cff).
- [MyST: math and equations](https://myst-parser.readthedocs.io/en/latest/syntax/math.html).
- [GitHub: writing mathematical expressions](https://docs.github.com/en/get-started/writing-on-github/working-with-advanced-formatting/writing-mathematical-expressions).
- [VS Code: Markdown](https://code.visualstudio.com/docs/languages/markdown).
