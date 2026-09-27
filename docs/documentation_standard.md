---
myst:
  html_meta:
    description: >-
      Authoring rules for OptimalPortfolios documentation: page forms and the convention card,
      attribution, portable mathematics, the paper ledger, symbol and parameter ownership,
      executable examples, diagrams and analytical provenance.
---

# Documentation standard

*Author: [Artur Sepp](https://github.com/ArturSepp) / First recorded: [2026-09-14](https://github.com/ArturSepp/OptimalPortfolios/commit/bdcbc350a2699eb6cf8769f9af7c15aba38c545a)*

This standard applies to [OptimalPortfolios](https://github.com/ArturSepp/OptimalPortfolios).
Software citation: [CITATION.cff](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).

This is the OP supplement to the
[shared OSS documentation standard](https://github.com/ArturSepp/ArturSepp/blob/main/docs/documentation_standard.md).
The shared guide owns common authoring rules; this page retains OptimalPortfolios-specific
contracts, examples, analytics tooling, and verification. General changes belong in the
shared guide. Existing section headings remain available for incoming links.

## Article structure

Use the shared [article structure](https://github.com/ArturSepp/ArturSepp/blob/main/docs/documentation_standard.md#user-content-article-structure),
with the H2 `Implementation in optimalportfolios`. Three page forms exist, each recorded in the
[page inventory](../tools/docs_inventory.json):

- A **methodology article** has the eight H2 sections of the shared template, in order.
- A **case study** reports a study from one of the package's papers. Its H2 sections are, in
  order: Overview; Study design and data; Configuration; Results; What the study does and does
  not show; Reproduce; See also; References. Results are quoted from the paper by section, with
  the study design, and are never recomputed. The Python blocks build the study's configuration
  on a synthetic panel and assert the qualitative mechanism, never the paper's numbers.
- A **utility page** (home, installation, conventions, gallery, guides and reference pages) has a
  byline, the software-citation line and a metadata description, and no empty sections.

The inventory also lists **planned** pages with their form and title; they have no file until
they are written, and `check_docs.py --all` fails while any remains.

**Convention card.** The first table under *Inputs, notation, and assumptions* of a methodology
article has the header `| Convention | This article |` and exactly these rows, in order: Return
basis, Estimation grid, Rebalancing grid, Covariance units, Expected returns, Weight state,
Solver. The [conventions page](conventions.md) defines each row. The checker requires the card on
an adopted methodology article and on any article written to the canonical-script contract;
earlier articles gain it when they are revised.

**Callouts.** Write a practical insight or a common error as an ordinary blockquote whose first
word is bold, `> **Insight.** ...` or `> **Pitfall.** ...`. The site renders it as an admonition
through `docs/_ext/optimalportfolios_callouts.py`; GitHub and VS Code show a quotation.

**Notation.** Symbols reserved on the [conventions page](conventions.md#notation) keep one meaning
everywhere. A page declares any other symbol it uses and does not reuse a reserved one.

The API entry `api.rst` stays RST; `docs/conf.py` generates its reference body at build time (see
[References and implementation ownership](#references-and-implementation-ownership)). Build output,
generated sources and autosummary templates are not authoring locations.

## Copyable methodology template

Copy the [shared methodology template](https://github.com/ArturSepp/ArturSepp/blob/main/docs/documentation_standard.md#user-content-copyable-methodology-template),
replace `PACKAGE` with `optimalportfolios` and `REPOSITORY` with `OptimalPortfolios`, and supply
the topic. Keep the MyST description front matter. Follow the shared
[authorship and date rules](https://github.com/ArturSepp/ArturSepp/blob/main/docs/documentation_standard.md#user-content-authorship-and-dates);
a new uncommitted page uses the linked Artur Sepp byline without a date.

## Portable mathematics

Follow the shared [portable mathematics rules](https://github.com/ArturSepp/ArturSepp/blob/main/docs/documentation_standard.md#user-content-portable-mathematics).
MyST's `dollarmath` extension is enabled in the OP site configuration. Inspect Sphinx,
GitHub Markdown, and VS Code separately; keep unperformed viewer checks pending.

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

Check dimensions, signs, timing, covariance scale and normalization against the owning
implementation. A delimiter repair must not alter mathematical meaning.

Inside math, spell every command with letters. GitHub Markdown applies backslash escapes before
the math reaches its renderer, so a thin space written as backslash-comma displays as a comma, a
norm bar written as backslash-bar becomes a single bar, and an escaped brace loses its backslash.
Write `\lVert x \rVert`, `\lvert x \rvert`, `\lbrace` and `\rbrace`, or omit fine spacing. Inside
a display block, never start a line with `+`, `-`, `*`, `>`, `#` or a numbered-list marker;
GitHub reads it as a block element and ends the formula. Absolute values inside a Markdown table
use `\lvert` and `\rvert`, because a bare bar ends the table cell.

Three more GitHub faults concern inline math, and MyST and VS Code show none of them:

- GitHub escapes `<` and `>` in inline math twice, so the reader sees `&gt;`. Write `\lt` and
  `\gt`.
- GitHub opens inline math only after a space, `(` or `*`, and closes it only before a character
  that is not a letter or digit. Write `a matrix of size $n \times n$`, not `$n$-by-$n$`, and
  `asset $i$`, not `the $i$th asset`.
- An underscore after a closing brace, as in `\hat{\mu}_t`, can pair with a later letter
  subscript such as `w_{t^-}` in the same paragraph as emphasis, which breaks every formula in
  between. Attach the subscript to a letter: `\hat\mu_t`.

`tools/check_docs.py` rejects all of these. Before a page is adopted, it is also rendered through
the GitHub Markdown API and VS Code's KaTeX plugin; the stage audit records the result.

## References and implementation ownership

Apply the shared [reference and example rules](https://github.com/ArturSepp/ArturSepp/blob/main/docs/documentation_standard.md#user-content-references-and-executable-examples).
Every article cites OptimalPortfolios. If calculations use
[qis](https://github.com/ArturSepp/QuantInvestStrats) or
[factorlasso](https://github.com/ArturSepp/FactorLasso), identify that delegation and include the
[qis citation](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff) or
[factorlasso citation](https://github.com/ArturSepp/FactorLasso/blob/main/CITATION.cff) as appropriate.

OptimalPortfolios owns construction and rolling orchestration. Generic sparse factor estimation
and HCGL belong to factorlasso; holdings simulation, reporting and analytics belong to qis.
Use the existing risk-model integration for tracking-error examples. Do not introduce another
backtester or statistics layer to make a chart.

The public interface is defined by the package-root re-exports and the existing public-API tests;
there is no `__all__`. Verify signatures and enum members before writing examples. The complete
[constraints contract](constraints.md) stays authoritative. Source-adjacent READMEs describe
contributor/module workflows and link to public methodology instead of duplicating it.
Do not assume repository Markdown notes are included in installed wheels.

### Symbol and parameter ownership

The `symbols` section of the inventory assigns every public object of the package root to exactly
one page, which may be planned. `external_symbols` records the names re-exported from FactorLasso
and qis, with the owning documentation of the package that defines them. The `parameters` section
assigns every field of the mapped configuration dataclasses (`Constraints`, `OptimiserConfig`,
`EwmaCovarEstimator`, `FactorCovarEstimator` and `UniverseData`) to exactly one page. An adopted
page names, in inline code, every object and field it owns.

`tools/check_docs.py` reads the public surface and the dataclass fields from the package source
with `ast`, so a new export or a new field without an owner fails the checker without importing
the numerical stack; a test confirms that this reading equals the imported package. The API page
is generated when the site is built: `docs/conf.py` writes `docs/_generated/api_reference.rst`
(git-ignored) from the inventory, with one section per owning page in the order of the sidebar, the
re-exports, and one table of fields and defaults per mapped dataclass. Object pages keep their
published `generated/optimalportfolios.<name>.html` addresses.

### Papers and references

The package's own papers are recorded once, in the `papers` ledger of the inventory, with the
citation used everywhere on the site; titles used in earlier versions are listed as
`retired_titles`, and the checker fails when one appears in reader-facing text. The
[research papers page](research_papers.md) carries each ledger title. Only public papers are in
the ledger and on the site: a published article or a public working paper is cited by section and
equation, and its results are quoted with their study design. Research that has not been published
is neither cited, linked, quoted nor displayed; it may shape an example or a limitation, and a
fact it contains is stated with its published source or derived on the page.

Other literature is cited when it is already in a module docstring's references, in
[`paper.bib`](../paper.bib) or in a tracked manuscript's bibliography, or when it has been checked
against the publisher or DOI page and that check is recorded in the stage audit. For a method from
the literature, state what the implementation does not inherit.

### Executable examples

A page written from 27 September 2026 onwards has one canonical script,
`examples/docs/<page_basename>.py`, named by the `example` key of its inventory entry. The page
shows the code inline, so that it reads on GitHub and in an editor as well as on the site, and
links the script. Every fenced block tagged `python` must be a verbatim, contiguous excerpt of that
script, compared after removing common indentation; a block that is not meant to run carries the
comment `<!-- fragment -->` on the line before its fence. A script runs offline after
`pip install optimalportfolios`, builds its inputs from a fixed seed, and asserts every number the
page quotes against a reference computed a different way. A script that exercises a repository
example, such as the CSV risk-model page, needs a checkout instead and says so on its page.
`src/optimalportfolios/tests/documentation_examples_test.py` runs every script in the test suite,
and the examples workflow runs them again in their lane.

Every methodology page and the objective router follow this contract. Three utility pages keep
their own tests under `src/optimalportfolios/tests/`, because those check facts rather than
execute page blocks: `installation_documentation_test.py` checks the recipes against the
packaging metadata, `quickstart_documentation_test.py` the quickstart script's claims and output,
and `examples_documentation_test.py` the catalogue against the example classifier. Preserve the
quickstart's executed README and Python/notebook parity checks. Label fragments with missing
context as illustrative; do not present them as standalone commands.

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

A **teaching exhibit** is a synthetic figure drawn by the canonical script of the page that shows
it: a function of `examples/docs/<page>.py` draws the figure and returns its plotted table and the
numerical checks it illustrates. It is registered in
[`tools/docs_analytics/teaching.json`](../tools/docs_analytics/teaching.json) with the script's
constants that fix its inputs, the pages that display it, its question and its sample, and it is
published in `docs/images/` with a [manifest](images/analytics_manifest.json) of the image, table
and script hashes, the checks, the software versions and a review note:

```console
python -m tools.docs_analytics.teaching --all --output-root <new-C-local-bundle>
python -m tools.docs_analytics.teaching --publish <that-bundle> --review "<what was inspected>"
python -m tools.docs_analytics.run --verify
```

`--all` refuses a parameter that differs from the script or a failing check, and publication
refuses a stale or altered bundle. `run --verify` checks the README previews against their receipt
and every teaching exhibit against its manifest and the current script, so editing a canonical
script requires regenerating its exhibit. The README-preview registry is separate because its
publication receipt embeds that registry verbatim.

A **diagram** is Mermaid source in a fenced `mermaid` block. It renders natively on GitHub and,
through `sphinxcontrib-mermaid`, on the site; VS Code needs an extension, so each diagram is
followed by a sentence that states its content in words. A diagram carries no data, is reviewed as
source and is not registered as an image. Mermaid click links are disabled on GitHub, so a diagram
that stands for pages is accompanied by a table of links.

The six README previews are offline synthetic teaching exhibits. Their shared
[provenance record](../examples/figures/analytics_manifest.json) records the input/configuration
identity, actual software environment, generation timestamp and separate visual review. The
[analytics registry](../tools/docs_analytics/registry.json) now records the six stable preview
paths, their four producer families, and eight non-analytics badges. The
[analytics runner](../tools/docs_analytics/run.py) checks coverage without importing those scripts.
Generation, validation and publication tooling are available. All four producer families now
have offline implementations. Complete-bundle generation validates six PNGs and their supporting
CSVs; publication still requires the separate visual-review record.

The operator procedure for these previews, with every command, producer family, provenance
field and publication step, is kept with the tooling in
[`tools/docs_analytics/README.md`](https://github.com/ArturSepp/OptimalPortfolios/tree/main/tools/docs_analytics). The sections below summarise it and keep their
anchors for incoming links.

<a id="analytics-registry-and-refresh-planning"></a>

### Analytics registry, planning and generation

The [analytics registry](../tools/docs_analytics/registry.json) lists every displayed image with
its producer and the pages that show it. After the repository environment setup, run from a
C-local source export:

```console
python -m tools.docs_analytics.run --list
python -m tools.docs_analytics.run --all --output-root <new-C-local-run>
python -m tools.docs_analytics.validate --run-root <that-run>
```

`--list` checks that every displayed image is registered, `--all` generates a complete bundle into
a new directory and refuses an incomplete one, and `validate` reads the bundle back. Details are in the section *Analytics registry, planning and generation* of the [tooling README](https://github.com/ArturSepp/OptimalPortfolios/tree/main/tools/docs_analytics).

### Portfolio-report family preview

One producer draws the three report previews of a synthetic backtest, with their supporting CSV
tables and a family manifest. Details are in the section *Portfolio-report family preview* of the [tooling README](https://github.com/ArturSepp/OptimalPortfolios/tree/main/tools/docs_analytics).

### Span-sensitivity family preview

One preview compares fixed weekly EWMA spans of 5, 13, 26, 52 and 104 on the portfolio-report
baseline. Details are in the section *Span-sensitivity family preview* of the [tooling README](https://github.com/ArturSepp/OptimalPortfolios/tree/main/tools/docs_analytics).

### Optimizer-comparison family preview

One preview compares minimum variance, maximum diversification and equal risk budgets on the
same baseline. Details are in the section *Optimizer-comparison family preview* of the [tooling README](https://github.com/ArturSepp/OptimalPortfolios/tree/main/tools/docs_analytics).

### Covariance-comparison family preview

One preview compares six covariance estimators under the same minimum-variance objective on a
simulation with known factors. Details are in the section *Covariance-comparison family preview* of the [tooling README](https://github.com/ArturSepp/OptimalPortfolios/tree/main/tools/docs_analytics).

### Producer contract and provenance

The registry records each producer's configuration explicitly; no field is inferred from a legacy
script's defaults, and every output is hashed with its inputs, source and environment. Details are in the section *Producer contract and provenance* of the [tooling README](https://github.com/ArturSepp/OptimalPortfolios/tree/main/tools/docs_analytics).

### Reviewed preview publication

Only a complete, validated bundle with a separate visual-review record may replace the committed
previews. Details are in the section *Reviewed preview publication* of the [tooling README](https://github.com/ArturSepp/OptimalPortfolios/tree/main/tools/docs_analytics).

### Publication recovery

Publication keeps a C-local backup and a recovery journal, because replacing several files is not
one filesystem transaction. Details are in the section *Publication recovery* of the [tooling README](https://github.com/ArturSepp/OptimalPortfolios/tree/main/tools/docs_analytics).

## Verification and migration

`tools/check_docs.py` checks article metadata, visible attribution/citation links, heading
structure for each page form, portable math delimiters and local file links without importing the
numerical stack. On the pages that require them it also checks the convention card and that every
Python block is an excerpt of the page's canonical script. In every mode it checks the
repository-wide contracts: each public object and mapped dataclass field has one owning page,
each re-export has an external owner, the planned pages do not exist yet, and no retired paper
title appears in reader-facing text. `tools/docs_inventory.json` explicitly classifies adopted,
pending and planned pages and the API entry. New reader-facing pages must enter this inventory;
omitted pages are errors.

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
final migration gate and fails while pending or planned pages remain. Legacy RST pages retain explicit
inventory entries until converted; do not leave both Markdown and RST sources with the same basename.

These checks do not prove mathematical correctness, external-link availability, bibliography
accuracy or rendering. Run relevant examples/tests and strict Sphinx HTML/link checks separately.
Use a C-local source export: autosummary writes generated source files, so redirecting only the
HTML output directory is insufficient for a OneDrive checkout.

When renaming a page, preserve its basename and HTML address, record existing anchors, and update
source/download links. Inspect rendered formulas after the math engine finishes. Keep numerical
changes and unsupported empirical claims out of cosmetic edits.

Do not write version stamps such as "verified with OptimalPortfolios 7.6.0" in a page; they go
stale at the next release. `docs/conf.py` reads the versions of the build environment, which on
Read the Docs is `uv.lock`, and the footer of every page names them. What a page asserts is
checked by its canonical script or its test, in the locked environment.

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
| `check_docs.py --source-all` | Every inventoried human source meets the source standard; pending reviews stay explicit; every public object and mapped field has one owner; no retired paper title is used. |
| `readthedocs_config_test.py` | The hosted build's commands survive the Read the Docs shell wrapper and call uv by path. |
| `tools.docs_analytics.run --list` | Every displayed image is registered, including its document consumers. |
| `tools.docs_analytics.run --verify` | Committed README previews match their dated review receipt and current coverage registry, and every teaching exhibit matches its manifest and the current canonical script. |
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

The configuration installs uv into its own directory beside the virtualenv and calls that binary
by path in every step. Read the Docs does not put pip's uv entry point on the build `PATH`, and
once the project virtualenv exists it puts that environment's `bin` directory first, so neither a
bare `uv` nor `python -m uv` is reliable there. A bare `uv` failed every hosted build from
2026-09-14 to 2026-09-26 while `docs.yml` stayed green, since that workflow builds in its own
environment. Read the Docs also wraps each build-job command, unescaped, in
`/bin/sh -c '<command>'`, so a command must not contain a single quote.
`readthedocs_config_test.py` enforces these rules in the test matrix of every pull request.
The daily [external documentation health workflow](../.github/workflows/link-health.yml) now
fails when the newest finished build of `latest` did not succeed. The `stable` version rebuilds
only from a release tag.

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
