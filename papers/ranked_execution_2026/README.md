# Execution-Constrained Portfolio Rebalancing

*Policy Domains, Search and Numerical Reliability*

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Software: [OptimalPortfolios](https://github.com/ArturSepp/OptimalPortfolios).
Citation: [CITATION.cff](../../CITATION.cff).

## Article and scope

The [article](paper/main.pdf) and [TeX source](paper/main.tex) present one
conditional execution formulation, corridor-policy costs and numerical reliability,
with matched comparisons of Exact, Ranked and a Kernel Search adaptation.
The method names are **Exact Execution** (matched MIQP) and **Ranked Execution**
(historically V2). Historical protocols and results retain their original labels.
The primary policy attempts Exact first and invokes Ranked when it cannot deliver an
acceptable native result. Both use the same prepared domain and original-unit
audit. This is a working draft intended for *Computers & Operations Research*;
no submission, acceptance or permanent dataset deposit is claimed.

The public study uses released **OP 9.2.0**, not the earlier research builds that
shared 9.1.5 metadata. It contains:

- 219 monthly supplied-input decisions from the retrospective 302-ETF universe.
- The unchanged 24-case synthetic grid and two tiny enumeration controls.
- A fresh 64-case synthetic holdout and four further tiny controls.
- Nine separately labelled injected-primary-failure controls after preparation.

The [completed difficulty supplement](paper/supplement.pdf) adds 23 prospective
cases and 82 declared method attempts, with [frozen design/run identity](replication/study_run_protocol.json)
and [complete reference results](replication/study_expected_summary.json).
The main article integrates both cohorts, retaining failed deliveries, unresolved
gaps and results against each method. The supplement gives the full algorithms,
proofs and replication mechanics. No API migration is implied by display names.

The retained cohort's main arms are independent Exact, standalone Ranked and the
actual Exact-first policy with audited Ranked fallback. Secondary arms use explicit Ranked
starts; the existing synthetic grid also retains separate 60/600-second calls.
No comparator receives another method's result unless its name declares that
start, and all generation costs are included.

The public evidence measures conditional execution quality, timing and failure
handling. It does not establish realised returns, own-holdings rolling performance,
independent external validation or universal solver dominance. Private fund
observations, identifier mappings and raw vendor histories are excluded.

## Formulation and acceptance

The objective is full-covariance tracking variance to the supplied raw target.
Actual noncash movements consume tickets. Mandatory actions retain explicit
categories, intervals and any minimum-size exemption. Holding floors, cash and
group constraints remain hard rules. Common feasibility preparation is shared
and charged to each arm. A compiled constraint must correspond to an explicit
source declaration; an omitted membership vector is not a zero exposure.

The audit tolerance remains `1e-7` NAV and the numerical-zero threshold `1e-8`.
The descriptive equality band is `0.0001` annual tracking-error bp; it is not a
feasibility tolerance. The MIQP primary uses ten seconds, one thread and a
`1e-4` relative variance-gap target. The canonical objective offset is restored
before solving, so relative stopping uses the original variance origin.

The policy requires audited weights, an audited native/refined witness, no
native/callback failure and no detected bound contradiction. An acceptable
time-limited incumbent can be delivered. V2 is invoked lazily on rejection,
then audited against the same domain. It is not a best-of-two selection.
A native optimal label is not an independently verified global certificate.
Common preparation itself uses MOSEK; the fallback does not provide an entirely
MOSEK-independent pipeline. Injected faults occur after successful preparation.

## Frozen public inputs and holdout

The ETF archive supplies current, base and target weights, annual covariance,
corridor/full bounds, group rows, eligibility, mandatory categories, exemptions
and ticket rules. Stable `AssetNNN` identifiers and `Cash` replace instrument
labels. These are de-identified numerical inputs, not an assurance that public
instrument histories cannot be recognised. Model weights alone do not reproduce
the execution decisions. Readers need no private application or upstream refit.

Each ETF decision starts from the drifted previous supplied model allocation.
The three methods do not arrive with recursively different holdings.
The new holdout uses two fresh seeds, 72/144 optional instruments, 2/6 tickets,
disjoint/overlapping groups, tight/wide bands and diffuse/exit-rich targets.
Exit-rich variants set half the optional targets to zero and redistribute the
same target mass. Only target corridors change within each pair; hard limits,
current holdings, covariance, mandatory declarations and thresholds do not.
The four tiny controls retain positive targets for independent enumeration.
Factorial variants share random draws and are not independent market samples.

[matched_protocol.json](replication/matched_protocol.json) declares the design.
[matched_run_protocol.json](replication/matched_run_protocol.json) records the
pre-solve freeze, inputs, dependency versions and numerical source identities.
[matched_expected_summary.json](replication/matched_expected_summary.json)
contains the complete current summary, including failed-case denominators,
paired differences, timing tails and native-bound checks.

## Reproduction

Create a separate environment outside synced storage. On this Windows host use
`C:\Python\OPPublic920312`; do not create a virtual environment under OneDrive.
Install `optimalportfolios[exact]==9.2.0` and the numerical versions recorded in
the run protocol. The wheel's SHA-256 must be
`b26229b4a3e6772f741da3b03fbc6d5885e1f504175616b13da7063bf9a073d5`.
The runner checks both the wheel and every installed OP Python file against it.
A licensed MOSEK installation is required for the numerical comparisons.

From the repository root, with the separate numerical archive available:

```powershell
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
& C:\Python\OPPublic920312\Scripts\python.exe -m papers.ranked_execution_2026.replication.matched freeze --bundle C:\Research\execution\inputs\etf --wheel C:\Research\execution\software\optimalportfolios-9.2.0-py3-none-any.whl --output C:\Research\execution-run
& C:\Python\OPPublic920312\Scripts\python.exe -m papers.ranked_execution_2026.replication.matched solve --output C:\Research\execution-run
& C:\Python\OPPublic920312\Scripts\python.exe -m papers.ranked_execution_2026.replication.matched faults --output C:\Research\execution-run
& C:\Python\OPPublic920312\Scripts\python.exe -m papers.ranked_execution_2026.replication.summarize_matched --run C:\Research\execution-run --output C:\Research\execution-summary
```

`freeze` requires a fresh output directory. All raw inputs and source hashes are
written before the first preparation or solve. `solve` can resume completed
cases only against the same frozen sources and input bytes. It retains failed
preparation and failed arms. The summariser refuses incomplete experiments.
Run serially without other computational jobs when comparing timings. Requested
native allowances are not hard wall-clock deadlines.

The replication contract tests use public synthetic fixtures:

```powershell
& C:\Python\OPPublic920312\Scripts\python.exe -m pytest papers/ranked_execution_2026/replication/tests
```

## Prospective difficulty study

[study_protocol.json](replication/study_protocol.json) specifies a separate
23-case, 82-arm comparison: 18 sparse-book scaling cells, four OR-Library risk
adaptations and one analytic corridor counterexample. The eligible noncash
universes contain 300, 1,000 or 3,000 names; current holdings remain sparse.
The manifest reports effective executable candidates separately. Two bundled
synthetic profiles have one seed each, so their differences do not identify
individual causal effects or establish population-level superiority.

Main arms are independent Exact with a 60-second internal allowance, released
Ranked with a 180-second process ceiling, and a documented 60-second
[Kernel Search adaptation](replication/study_kernel.py). The three tight,
30-ticket cells additionally compare Exact10/600, the identical dense covariance
representation at ten seconds, and the full hard domain at 60 seconds. Full
versus Corridor is a policy-domain comparison; unfinished incumbents do not
establish its optimal restriction cost. No new primary/fallback policy arm is
inferred from these standalone calls.

Download `indtrack1.txt` and `indtrack6.txt` from the
[OR-Library owner](https://people.brunel.ac.uk/~mastjjb/jeb/orlib/indtrackinfo.html)
into the local references directory. The loader checks the frozen checksums.
It uses point-in-time QIS log-return EWMA risk, span 52, with explicitly assumed
weekly labels and 52-period annualisation. The source has no calendar labels.
All current weights, targets, cash and execution constraints are synthetic.
Raw owner files and derivative covariance arrays are excluded from the public
archive pending established reuse terms; the loader regenerates them locally.

After configuring the separate research environment and single-thread settings:

```powershell
& C:\Python\OPPublic920312\Scripts\python.exe -m papers.ranked_execution_2026.replication.study_protocol --references C:\Research\execution\references --wheel C:\Research\execution\software\optimalportfolios-9.2.0-py3-none-any.whl --output C:\Research\difficulty-run
& C:\Python\OPPublic920312\Scripts\python.exe -m papers.ranked_execution_2026.replication.study_run run --root C:\Research\difficulty-run
& C:\Python\OPPublic920312\Scripts\python.exe -m papers.ranked_execution_2026.replication.summarize_study --root C:\Research\difficulty-run
```

Freeze requires a new output directory and writes every input and source hash
before solving. Run serially without concurrent computational jobs. Timeouts
remain failed deliveries; the summary refuses an incomplete sample, reaudits
all returned weights and checks native bounds against every available witness
in the same domain. A smaller native gap may concern a rejected native vector;
the delivered-portfolio gap is recorded separately.

The hashes identify the evaluated inputs and implementation. They do not promise
identical time-limited search paths on different hardware or repeat runs: a
wall-clock cutoff can stop at a different incumbent. Reproduction must retain
the declared limits, all unsuccessful attempts and the original-input audits.
Historical protocol labels stay unchanged even when current tables use Exact
Execution and Ranked Execution.

[The supplement source](paper/supplement.tex) specifies the interval formulation,
Ranked search order, comparator and accounting. Separate historical numerical
controls illustrate objective-origin preservation, equivalent conic scaling
and removal of satisfied constant rows. They are not additional holdout cases:

```powershell
& C:\Python\OPPublic920312\Scripts\python.exe -m papers.ranked_execution_2026.replication.study_numerics --output C:\Research\numerical-mechanisms
& C:\Python\OPPublic920312\Scripts\python.exe -m papers.ranked_execution_2026.replication.summarize_study --root C:\Research\difficulty-run --numerics C:\Research\numerical-mechanisms
```

The completed scaling comparison returns audited portfolios in 18/18 calls for
Exact60 and Kernel60, and 2/18 for Ranked under its 180-second process ceiling.
Kernel has lower TE than Exact60 in 11 cases, ties in four and higher TE in three.
On the four external-risk cases, Exact and Kernel both deliver; Exact improves
the two larger cases and ties the two smaller ones. Ranked delivers only in the
two smaller cases. These are incumbent comparisons under the declared controls,
not a claim about globally optimal solutions or the best possible implementation
of each algorithm family. The supplement retains all missing deliveries and
unresolved longer-budget results.

The [separate numerical controls](replication/numerical_mechanisms_summary.json)
show small native-bound contradictions in the deliberately unbalanced cone
representation, but none above the comparison scale in the balanced one.
The historical constant-row sizing failure does not reproduce in this runtime:
both equivalent graphs solve the within-tolerance examples and preserve the
genuine infeasibility of all outside-tolerance examples.

## Retained research and publication boundary

Earlier ETF, sensitivity, scarce-ticket, breadth and search-ablation runners and
summaries remain as historical replication records. Their `protocol.json`,
`expected_summary.json` and associated version/hash records identify those older
cohorts; they are not relabelled as OP 9.2.0. The preceding 26-page manuscript and
its dependencies are preserved locally in the dated draft archive and in the
public replication history. The original internal research workspace is separate.

The current main article uses `historical_results.tex`, `study_results.tex` and
`study_controls.tex`, generated from immutable summaries by `summarize_study`,
and the difficulty figure. The supplement also uses `numerical_results.tex`.
Earlier `results.tex`, `joint_results.tex`, `robustness.tex`, `breadth.tex` and
figures remain preserved but are not included in the current main article.
Build `paper/main.tex` twice with pdfLaTeX and the standard Elsevier `elsarticle`
class in a C-local source copy. Build `paper/supplement.tex` twice in the same
copy. Its four generated table files and difficulty figure come from
`summarize_study`, using the completed mechanism directory for the numerical
table. [build_manifest.json](paper/build_manifest.json) records exact source,
result and PDF identities. Copy only reviewed durable deliverables back.

The exact Git allowlist exposes approved manuscript, public runners, tests and
reference summaries. Numerical inputs and complete outputs are supplied as a
separate archive. Its persistent identifier remains pending. This work changes
neither production behaviour nor a dependency lock and publishes no package.
Author declarations and final submission metadata remain to be completed.
