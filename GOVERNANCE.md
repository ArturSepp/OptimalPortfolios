# Governance, maintenance, and support

`optimalportfolios` is an open-source research-software project for portfolio construction and
rolling backtesting. This document states how decisions are made, how the project is maintained,
and what contributors and users can expect.

## Maintainer and decision model

Artur Sepp is the lead maintainer and currently has final responsibility for project scope,
technical decisions, releases, and repository administration. Development takes place through the
public GitHub repository. Contributors are encouraged to raise design questions in an issue before
starting a substantial change.

Decisions are normally discussed in issues and pull requests so that their rationale remains
public. The lead maintainer may accept, request changes to, defer, or decline a proposal. Decisions
are guided by the project scope and technical principles below, not by a vote or a fixed number of
approvals.

There is currently no steering committee or guaranteed maintainer succession process. A
contributor who demonstrates sustained technical judgement, constructive review, and commitment to
the project may be invited by the lead maintainer to take on review or maintenance responsibilities.
There is no automatic contribution threshold for such an invitation.

## Technical decision principles

Changes are evaluated against the following priorities:

1. Numerical correctness and the absence of look-ahead in backtests.
2. Preservation of published optimiser defaults, constraint semantics, and rebalancing
   conventions unless a documented correction is required.
3. Reuse of the surrounding open-source stack: `qis` owns performance analytics and backtesting
   primitives, while `factorlasso` owns generic factor and cluster-lineage estimation.
4. A core installation that remains testable offline and does not require proprietary data.
5. Backward compatibility of the public API, with explicit versioning and changelog treatment for
   public-signature or behavioural changes.
6. Cross-platform maintainability, reproducible tests, and documentation that a reviewer can run
   without access to the maintainer's environment.

The package uses CVXPY and `quadprog` as its optimisation backends and does not add parallel
analytics, plotting, or factor-estimation layers that belong in its declared dependencies. A
proposal outside these boundaries may be useful but will normally be directed to the appropriate
project rather than implemented here.

## Proposing and contributing changes

Small bug fixes and documentation improvements may go directly to a pull request. Open an issue
first for a new optimiser, constraint type, dependency, public API change, numerical convention, or
change spanning several subsystems.

The contribution process, development commands, testing expectations, and replication contract
are documented in [CONTRIBUTING.md](CONTRIBUTING.md). Pull requests are reviewed for scope,
correctness, tests, documentation, compatibility, and effects on published results. A contribution
may be declined when it duplicates `qis` or `factorlasso`, requires proprietary data for routine
verification, changes numerical behaviour without evidence, or creates a maintenance obligation
outside the project's scope.

## Releases and compatibility

Releases are versioned, recorded in [CHANGELOG.md](CHANGELOG.md), and distributed through
[PyPI](https://pypi.org/project/optimalportfolios/). Formal releases normally have matching package
metadata, a Git tag, and a GitHub Release. There is no fixed release cadence.

The supported Python versions and dependency floors are declared in `pyproject.toml` and tested in
continuous integration. Public API or default changes require explicit changelog and version
treatment. When practical, a replacement is introduced before an API is removed. The project does
not promise a fixed deprecation period: urgent correctness, compatibility, or security fixes may
require a faster change, which will be documented.

Published numerical results are not changed to make a test pass. If a correction affects a
published convention or replication, the difference must be explained and the relevant
replication checks rerun.

## Support

Use the public [issue tracker](https://github.com/ArturSepp/OptimalPortfolios/issues):

- use the bug-report form for reproducible defects;
- open a normal issue for usage or methodology questions;
- identify the relevant publication and section for questions about a paper;
- use generated or public data in reproductions rather than proprietary inputs.

Support is provided on a best-effort basis. The project has no service-level agreement, guaranteed
response time, or entitlement to individual portfolio advice. Clear, self-contained reports that
run on a supported Python version are the easiest to investigate. Questions that belong to `qis`
or `factorlasso` may be redirected to those repositories.

GitHub issues are public. Do not post credentials, licensed datasets, client information, or other
sensitive material. For a security or privacy issue that should not be disclosed publicly, contact
the maintainer at `artursepp@gmail.com` with “OptimalPortfolios” in the subject. This contact route
does not create a guaranteed response-time commitment.

## Conduct and disagreements

Be civil, assume good faith, and discuss the technical claim rather than the person making it.
Technical disagreement is welcome. If discussion reaches an impasse, the lead maintainer records
the decision and its scope; a declined proposal may be revisited when new evidence or a smaller
design is available.

Harassment, personal attacks, or publication of another person's private information are not
acceptable. Conduct concerns may be raised privately with the maintainer using the contact above.

## Changes to this policy

Governance changes use the same public issue and pull-request process as other project changes.
Material changes should explain why the current policy is insufficient and identify any new
responsibilities or promises they create.
