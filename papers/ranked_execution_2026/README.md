# Ranked Execution: Cash-Funded Search for Sparse Portfolio Rebalancing

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Software: [OptimalPortfolios](https://github.com/ArturSepp/OptimalPortfolios).
Citation: [CITATION.cff](../../CITATION.cff).

## Article and scope

The [article](paper/main.pdf) and [TeX source](paper/main.tex) describe Ranked
Execution, implemented in `optimalportfolios.execution.v2`. The full joint MIQP
formulation (historical replication identifier `Exact`) and search ablations use the same objective, prepared constraints,
ticket rules and return audit. This is a draft intended for *Computers &
Operations Research*, not an accepted publication.

The empirical benchmark uses 219 monthly execution decisions derived from the
existing retrospective 302-ETF study. Every decision starts from the drifted
previous supplied model portfolio. All arms receive the same current holdings,
raw model target and covariance. The benchmark does not simulate strategies
recursively following their own holdings and supports no realised-performance
claim.

The manuscript, portable runner, audit tests and frozen reference summaries are
the approved public source scope. The accompanying numerical input archive is
prepared locally; its persistent public deposit and DOI remain pending.
Private research, identifier mappings, extraction code, raw vendor histories and
production observations are excluded. No missing input is silently replaced with
live or synthetic data.

## Numerical inputs

The accompanying archive contains `inputs/manifest.json` and 219 compressed NumPy
files. Stable `AssetNNN` identifiers and `Cash` replace original labels. Supplied
arrays retain:

- Current holdings, the initial funded base and raw target weights.
- Annual covariance, eligible coordinates and frozen coordinates.
- Corridor and full bounds, group loadings and group limits.
- Mandatory categories, holding floors, minimum-ticket exemptions and ticket caps.

These numerical decision inputs, rather than model weights alone, reproduce
common preparation and execution. The manifest records dates, source vintage,
accounting conventions and file checksums. Opaque identifiers are
pseudonymisation: the numerical histories may still permit recognition of
publicly traded instruments. No claim of irreversible anonymity is made.

## Reproduction

Use the exact accompanying OP wheel in `software/current/`, or the matching source identified by
[protocol.json](replication/protocol.json). The protocol records dependency
versions and SHA-256 hashes of execution source files. The tested OP development
cohort is not assumed to be available from PyPI. A compatible licensed MOSEK
installation is needed for common mixed-integer preparation and both Exact arms.
The package version label alone does not identify this unreleased research build;
verify the accompanying artifact hash and source hashes.
CLARABEL supplies continuous sizing. QIS owns tracking-error reporting.

Create the environment outside synced storage. On Windows, use the repository's
prescribed external environment under `C:\Python`; never create a virtual
environment under OneDrive. Install the accompanying wheel and the versions
recorded in the protocol into a separate environment if preserving an existing
research environment. From the repository root, set native numerical libraries
to one thread and run:

```powershell
$env:OPENBLAS_NUM_THREADS = "1"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
& C:\Python\RankedReplication312\Scripts\python.exe -m papers.ranked_execution_2026.replication.run_benchmark --bundle C:\Research\ranked-replication\inputs --output C:\Research\ranked-run
& C:\Python\RankedReplication312\Scripts\python.exe -m papers.ranked_execution_2026.replication.summarize --run C:\Research\ranked-run --output C:\Research\ranked-summary
```

The paths are examples. `--output` must be fresh and outside OneDrive.
For a quick smoke check, add `--limit 3` to the runner and use another output
directory. The summariser refuses to use a smoke run for manuscript tables.
The offline contract tests need neither benchmark data nor a MOSEK licence:

```powershell
& C:\Python\RankedReplication312\Scripts\python.exe -m pytest papers/ranked_execution_2026/replication/tests
```

Run the complete sample serially without simultaneous computational workloads
when comparing timings. The five arms are full Ranked, Greedy, OneAnchor,
ExactBase and ExactRanked. Common preparation has a 30-second allowance;
each Exact call has a separate 10-second allowance and one native thread.
ExactRanked end-to-end cost includes Ranked incumbent generation. Defaults,
units, tolerances and timing boundaries are specified in the manuscript and
recorded before solving. Native wall times can exceed their requested allowance.

The output retains every attempted decision, failed preparation, full delivered
weights, independent audits, native status, incumbent attribution and elapsed
time. `complete.json` distinguishes a full run from a smoke sample.
Compare the regenerated `summary.json` with
[expected_summary.json](replication/expected_summary.json).
Solver timings and boundary choices can vary by machine and numerical build;
every returned portfolio must still satisfy the declared original-unit audit.
The descriptive TE equality band is 0.0001 annual basis points and is not a
portfolio feasibility tolerance or a global optimality certificate.
An additional published-9.1.5 control using the same dependency cohort produces
exactly identical Ranked weights and TE on all 219 non-scarce decisions. The
older archived ETF study used a different preparation implementation; changes
against that historical cohort must not be attributed to the new search alone.
The deposit includes the control records; its concurrent-run timings are excluded.

## Full joint MIQP comparison

The main article compares V2, a common-base joint MIQP, and a separate V2-started
joint MIQP. “Full joint” means simultaneous state selection and sizing in the
same prepared CORRIDOR domain; it does not widen the feasible domain. The
operational `execution/MIQP` policy permits different alpha, terminal-position
and repair-on-touch rules and is deliberately excluded from this matched test.

[joint_protocol.json](replication/joint_protocol.json) freezes 24 synthetic
cases: 24/72/144 nominal optional assets, 2/6 tickets, disjoint/overlapping groups,
tight/wide funding and exposure bands, and seed 111001. Two eight-asset controls
use seed 112001. The unchanged generator, qualified V2 source and all inputs were
frozen before solving. These cases then informed the MIQP numerical correction;
the current paired rerun is a regression study for that reference. V2 remains
unchanged. Each case receives
six independently started joint calls: common-base and V2 starts at 10, 60 and
600 seconds. Native calls use one thread; budget and start orders alternate.
No incumbent passes between budgets. Preparation is charged to every arm, and
V2 generation to each V2-started arm. These are upper allowances, not forced
run durations or equal total budgets.

```powershell
& C:\Python\RankedReplication312\Scripts\python.exe -m papers.ranked_execution_2026.replication.joint --output C:\Research\ranked-joint
& C:\Python\RankedReplication312\Scripts\python.exe -m papers.ranked_execution_2026.replication.summarize_joint --run C:\Research\ranked-joint --output C:\Research\ranked-joint-summary
```

[joint_run_protocol.json](replication/joint_run_protocol.json) identifies the
exact input and source hashes; [joint_expected_summary.json](replication/joint_expected_summary.json)
retains all denominators, timings, sources and bound checks. All 182 delivered
portfolios pass the unchanged audit. V2 ties all 24 independent broad deliveries
at every allowance within the 0.0001 bp descriptive band. The previous 3.645443 bp
advantage disappears after correcting the reference's native integer feasibility
setting and balancing its quadratic cone. The quadratic, linear constraints and
audit are unchanged. Independent MIQP is faster: median E2E at the 600-second
allowance is 0.165 seconds versus V2's 1.559 seconds. There are no native bounds
above an audited cross-arm objective by more than the existing 1e-12 variance
comparison scale. Raw diagnostics remain visible and are not independent
optimality certificates.
V2-started calls retain V2 on 12 broad cases and match its quality on all 24.
Increasing the allowance changes no delivered TE above the descriptive band.
The two tiny enumerations cover 74 supports with none unresolved; all arm
differences are below the band. Conservative relaxation bounds do not close
the larger cases. Effective optional dimensions are 20–41, not 144.

The refreshed ETF results use the same numerical correction. Both MIQP arms and
all three heuristic arms return audited portfolios on all 219 cases, with no
native-bound contradictions. V2 ties V2-started MIQP on all 219 decisions; against
common-base MIQP it ties 207 and improves 12 above the descriptive band. V2 weights
and TE remain exactly identical to the prior candidate on all 219 ETF and 26
synthetic cases. Fixed-state QPs and continuous relaxation graphs are unchanged.
The ETF and new joint tables report current serial timings; older development
timings below describe the archived implementation and are not pooled with them.
The accompanying qualified wheel has SHA-256
`7b3b1c9b0ebafd79019b610da280eee33ef9e67d727b3e944d7595a400117515`.
The shared 9.1.5 metadata does not identify the published PyPI artifact.
In the updated deposit, `reference/current/etf` and `reference/current/joint`
contain the refreshed records. `reference/before_miqp_bound_fix/` preserves the
previous manuscript, software and complete research records with their own
hashes. Its earlier timings must not be interpreted as current measurements.

## Controlled sensitivity and scarce tickets

The additional study uses the same public input bundle and a standalone seeded
generator. It contains 24 fixed public dates, 288 target/covariance perturbations
and 64 synthetic scarce-ticket development problems, followed by a 32-case
earlier holdout now replayed as regression data. No private mandate, holding, identifier mapping or production
replay is used. The 219-case study is rerun with the same corrected source.

[robustness_protocol.json](replication/robustness_protocol.json) fixes the grid,
seeds, perturbation definitions and unchanged tolerances. The original scarcity
results and the first holdout informed the search corrections, so all 64
previously examined cases are now development data.
[robustness_run_protocol.json](replication/robustness_run_protocol.json) records
the QIS 5.44.4 runtime and exact execution-source hashes. The corrected OP source
is unreleased: its 9.1.5 package metadata does **not** mean the PyPI 9.1.5 wheel
contains this fix. Use the accompanying source or local build identified by the
hashes in a separate external environment. The target perturbation
preserves zero targets and cash; rebuilding changes optional corridors only,
not lifecycle declarations or upstream estimates. These are conditional
experiments, not simulated investment performance.

```powershell
& C:\Python\RankedReplication312\Scripts\python.exe -m papers.ranked_execution_2026.replication.robustness --bundle C:\Research\ranked-replication\inputs --output C:\Research\ranked-robustness
& C:\Python\RankedReplication312\Scripts\python.exe -m papers.ranked_execution_2026.replication.robustness --bundle C:\Research\ranked-replication\inputs --output C:\Research\ranked-holdout --study scarcity --protocol papers/ranked_execution_2026/replication/holdout_protocol.json
& C:\Python\RankedReplication312\Scripts\python.exe -m papers.ranked_execution_2026.replication.summarize_robustness --run C:\Research\ranked-robustness --holdout C:\Research\ranked-holdout --output C:\Research\ranked-robustness-summary --compact
```

The summariser writes paired CSVs, `summary.json`, the manuscript extension
`robustness.tex` and `sensitivity.pdf`. `--compact` keeps only sensitivity in the
main article; omit it to regenerate the historical development appendix.
It rejects partial/smoke studies. Compare
with [robustness_expected_summary.json](replication/robustness_expected_summary.json).
The archived pre-fix outcomes remain in the numerical deposit; supply both old
run directories with repeated `--previous` arguments to regenerate the before/after
paragraph. Matching uses generator inputs, not ordinal case filenames. The earlier
correction addressed one remaining development miss and four first-holdout misses.
Fresh comparisons keep
independent ExactBase calls and Ranked-started Exact separate. The post-hoc
fixed-support check cannot change a benchmark output.

[holdout_protocol.json](replication/holdout_protocol.json) records seeds 61001
and 61002 from the earlier holdout. Subsequent algorithm development makes this
rerun a regression replay, not a new holdout.
[holdout_run_protocol.json](replication/holdout_run_protocol.json) and
[holdout_expected_summary.json](replication/holdout_expected_summary.json) retain
source hashes, every failure denominator, timings and delivered-source attribution.
The earlier correction recovered the remaining 0.114 bp miss and four first-holdout
misses. Those reference records remain in the deposit. Updated summaries report
the current algorithm on all 96 earlier cases and distinguish independent Exact
from Ranked-started Exact. Retained-start equality does not certify global optimality.
Complete replacement sweeps and diversified perspective guidance add work beyond the original
shortlist; the manuscript reports that cost. All runs are serial; uncontrolled
workstation activity remains a timing limitation.

## Broader ablation and stronger reference controls

[breadth_protocol.json](replication/breadth_protocol.json) defines 96 broader
cases with 24/72 nominal optional instruments, 2/4 tickets, disjoint/overlapping
fractional groups, tight/wider funding-exposure bands and seeds 71001/71002,
81001/81002 and 91001/91002. All three earlier grids are now development data.
Current holdings are heterogeneous; two forced exits exceed the unchanged
minimum ticket without exemptions. Twelve additional eight-optional cases use
seeds 72001/72002, 82001/82002 and 92001/92002 for independent full-weight QP enumeration.

Full Ranked and OneAnchor are fixed before solving. Independent Exact calls
receive 10 or 60 seconds; separate 60-second calls receive only the full-search
or OneAnchor incumbent. Each start's own generation time is charged. Arm order
alternates deterministically. Conservative supporting-plane LP dual bounds are
reported separately from native MIP diagnostics and relaxation primal values.
The runner freezes all generated inputs and hashes before the first solve.

The first run exposed four shared combination misses, up to 7.762 annual bp,
under tight overlapping rows. All four better supports were absent from the
heuristics' trial logs but passed unchanged conditional sizing and audit. Those
cases are now development data. V2 adds bounded include/exclude guidance around
both incumbent states and the tentative relaxed support, including empty optional
incumbents. Proposals receive the existing replacement sweeps and original-domain
sizing. No Exact support is supplied to V2.

The second grid exposed five further misses. The new correction adds a final
interval-screened pair search with up to 576 coordinated trials across three
anchors. The screen allows every retained trade to resize and rejects only
supports whose interval envelope cannot satisfy funding or a group row. Surviving
supports use unchanged joint sizing and audit. Both earlier grids and their
previous outcomes are retained for controlled before/after comparisons.

The third grid retained a 0.549 bp miss involving three replaced positions.
The present correction restarts from up to 24 audited alternatives already in
the sizing cache. Each branch can traverse an inferior seed while separately
retaining the incumbent. The first improving branch receives one further bounded
pair-completion phase, then search stops. All constraints and acceptance rules
remain unchanged; no Exact support is supplied to V2.

[breadth_holdout_protocol.json](replication/breadth_holdout_protocol.json) predeclares
new broad seeds 101001/101002 and tiny seeds 102001/102002. Controls otherwise remain
identical. The implementation is frozen before this holdout and is not retuned
on its outcomes. This remains a same-generator holdout, not external validation.

```powershell
& C:\Python\RankedReplication312\Scripts\python.exe -m papers.ranked_execution_2026.replication.breadth --output C:\Research\ranked-breadth
& C:\Python\RankedReplication312\Scripts\python.exe -m papers.ranked_execution_2026.replication.breadth --output C:\Research\ranked-breadth-holdout --protocol papers/ranked_execution_2026/replication/breadth_holdout_protocol.json
& C:\Python\RankedReplication312\Scripts\python.exe -m papers.ranked_execution_2026.replication.summarize_combinations --run C:\Research\ranked-breadth --holdout C:\Research\ranked-breadth-holdout --previous C:\Research\ranked-replication\reference\before_triple_search\current\breadth-development --previous C:\Research\ranked-replication\reference\before_triple_search\current\breadth-holdout --output C:\Research\ranked-breadth-summary
```

Use a fresh output directory. The independent enumeration constructs full-weight
QP rows directly, without OP's state compiler or sizing builder; an invertible
basis-point scaling preserves the objective and constraints. It retains rejected
vectors and unresolved supports, which prevent a completeness claim. The return
audit and solver tolerances remain unchanged. Native optimal status is not
presented as a separately validated global bound. The correction changes V2
search only; objective, constraints, tolerances, configuration defaults, V1,
MIQP, production lock and installed production package remain unchanged.

[breadth_run_protocol.json](replication/breadth_run_protocol.json) records the frozen
input and source hashes; [breadth_expected_summary.json](replication/breadth_expected_summary.json)
retains every arm, timing, bound and controlled before/after comparison.
[breadth_holdout_run_protocol.json](replication/breadth_holdout_run_protocol.json)
and [breadth_holdout_expected_summary.json](replication/breadth_holdout_expected_summary.json)
record the fresh cohort. The comparison rejects changed development inputs,
changed algorithm/runtime controls across the holdout, duplicated or omitted
archived cases, or reused seeds. Cases are matched by their generating parameters,
not ordinal filenames. Cells sharing size and seed share inputs; factorial cells are not
independent replications. Tiny-case numerical references do not certify larger problems.

Both variants recover the archived 0.549 bp three-position miss, reducing annual
tracking error from 171.989658 to 171.440647 bp without a reference start.
Default Ranked has seven lower-TE outcomes, 89 ties and no losses against rerun
independent Exact on the 96 development cases. OneAnchor still trails full search
by 5.983 bp in one earlier case and ties it in the other 95.

On the fresh grid, full Ranked has two lower-TE outcomes, 30 ties and no losses
against independent Exact within the unchanged descriptive band. OneAnchor ties
full search on all 32 fresh cases. No holdout outcome informed further tuning.
The development loss prevents simplifying the current default to one pass.
All 864 arm outputs across the two cohorts and their tiny controls pass the
original audit. These finite tests do not certify global optimality.
Full Ranked/OneAnchor/independent Exact median call times on holdout are
9.499/9.108/0.499 seconds. The wider search improves coverage at added cost;
these results establish neither universal superiority nor a speed advantage
over Exact, and do not qualify a production update.

## Building and release status

Build `paper/main.tex` twice with pdfLaTeX and the standard Elsevier `elsarticle`
class. `paper/results.tex` is regenerated from the refreshed ETF study;
`paper/robustness.tex` uses the extension summariser's compact sensitivity mode.
`paper/joint_results.tex` and `paper/figures/joint_comparison.pdf` come from
`summarize_joint`. `paper/breadth.tex` and complete development summaries remain
in the replication archive but are no longer included in the main article.
Build outside OneDrive; copy only the reviewed PDF back.

The exact Git allowlist exposes only the files listed above. The input and
reference-output archive is a separate deposit, not an automatic Git addition.
This workspace does not activate production policies or publish a package.
Author declarations, the permanent dataset identifier and final submission
metadata remain to be completed before journal submission.
