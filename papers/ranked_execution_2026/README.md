# Ranked Execution: Cash-Funded Search for Sparse Portfolio Rebalancing

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Software: [OptimalPortfolios](https://github.com/ArturSepp/OptimalPortfolios).
Citation: [CITATION.cff](../../CITATION.cff).

## Article and scope

The [article](paper/main.pdf) and [TeX source](paper/main.tex) describe Ranked
Execution, implemented in `optimalportfolios.execution.v2`. The matched Exact
formulation and search ablations use the same objective, prepared constraints,
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

Use the exact accompanying OP wheel, or the matching source identified by
[protocol.json](replication/protocol.json). The protocol records dependency
versions and SHA-256 hashes of execution source files. The tested OP development
cohort is not assumed to be available from PyPI. A compatible licensed MOSEK
installation is needed for common mixed-integer preparation and both Exact arms.
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

## Building and release status

Build `paper/main.tex` twice with pdfLaTeX and the standard Elsevier `elsarticle`
class. `paper/results.tex` is generated by the summariser from the retained
complete run. Build outside OneDrive; copy only the reviewed PDF back.

The exact Git allowlist exposes only the files listed above. The input and
reference-output archive is a separate deposit, not an automatic Git addition.
This workspace does not activate production policies or publish a package.
Author declarations, the permanent dataset identifier and final submission
metadata remain to be completed before journal submission.
