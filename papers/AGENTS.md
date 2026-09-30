# Paper workspaces

This is OP's paper-workspace contract. The repository AGENTS.md still controls the
external Python environment, numerical invariants and C-local runtime workspace.

## Layout and ownership

Keep each paper at `papers/<paper_id>/`, outside `src/optimalportfolios/`. Use the
following sections when needed; do not commit placeholders for local sections.

| Section | Purpose | Git policy |
|---|---|---|
| `paper/` | One current manuscript, reading PDF, bibliography/build dependencies and `figures/` | Local by default; exact approved files only |
| `drafts/<date_or_version>/` | Self-contained previous versions, including their own figures | Always ignored |
| `presentations/<date_event>/` | Slides and their own figures | Local by default; exact approved files only |
| `private/` | Editor correspondence, referee reports, replies and permission records | Always ignored |
| `replication/` | Code, `data/`, and automated `tests/` | Approved code and redistributable inputs tracked |
| `agents/` | Paper-specific roadmaps, execution reports and working notes | Always ignored |

The per-paper `agents/` rule is OP's explicit exception to the generated shared
core's root-only default. Repository-wide records remain in root `agents/`.
`AGENTS.md` is a tracked instruction file, not an agent working record.

Keep algorithms that belong to the reusable package in `src/optimalportfolios/`.
Replication imports the package; production modules must not import `papers.*`.
Use repository-root module entry points and `replication/tests/*_test.py`. Do not
rename a published module or change a numerical baseline as incidental cleanup.

`papers/cma_data/` is an existing shared data module, not a paper: preserve its
paths, snapshot bytes, hashes and consumers. Do not copy it into each paper.
`equity_factors/` is an existing local study, not a public paper release.

## Publication and preservation

- New paper workspaces are entirely ignored by default. The approved public roots
  are `cma_data/`, `crypto_allocation_risk_2023/`, `robust_optimisation_jpm_2026/`,
  and the limited JOIM companion described below.
  Adding another requires an explicit publication decision, an exact root exception
  in `.gitignore`, and an update to `PUBLIC_WORKSPACES` in the policy checker.
- Preserve existing full-workspace exclusions, including MATF-CMA.
  Moving folders does not authorise release of any manuscript or replication code.
- `smart_diversification_joim_2026/` permits only its README, .gitignore,
  `run_overlay_example.py` and
  `replication/tests/overlay_example_test.py`. These introduce the paper and run a
  synthetic illustration. The manuscript, empirical replication code, licensed
  inputs, outputs and private notes stay ignored. The checker enforces this exact
  allowlist even when another ignore rule tries to reopen the workspace. The accepted
  manuscript's bibliography cites the folder at
  `https://github.com/ArturSepp/OptimalPortfolios/tree/main/papers/smart_diversification_joim_2026`,
  and the OP documentation links its README and runnable file: do not rename or move the
  folder, `README.md` or `run_overlay_example.py`.
- Record public status in `papers/README.md` and the paper README: public bundle,
  public replication with local manuscript, or entirely local workspace. Keep
  confidential permission details in `private/`. SSRN permission and acceptance
  do not by themselves authorise GitHub redistribution.
- `paper/` and `presentations/` are denied by default. A per-paper `.gitignore`
  lists exact approved exceptions, including necessary bibliography/class/style
  files. Do not use broad `!*.tex`, `!*.pdf` or `!figures/**` exceptions.
- The crypto TeX is a retained existing exception, not a complete build bundle:
  its referenced `figs1/` assets are absent. Its PDF remains local. Do not invent
  figures or claim a successful manuscript build.
- Put restricted inputs in `replication/data/local/`. Track only reviewed static
  inputs; required small frozen reference caches are permitted with provenance.
  Keep new runs, reports and mutable caches in the C-local runtime workspace.
- Keep licensed-data extraction code local unless separately approved. Public
  loaders must describe missing inputs clearly, never silently substitute live or
  synthetic inputs for a published result.
- Archive drafts with their own figures. Preserve local files during migrations;
  ignored material needs private backup/version history outside Git.
- Keep historical local `update_2026/` and `outputs/` material intact during this
  adoption. New working records go in `agents/`; new manuscript versions go in
  `drafts/`. Migrate mixed legacy material only after identifying its role.

## Verification

From the repository root, after configuring the prescribed external environment:

```powershell
python .github/scripts/check_paper_policy.py --worktree
python .github/scripts/paper_policy_test.py
python -m pytest papers/cma_data/tests papers/crypto_allocation_risk_2023/replication/tests
```

Before committing, run `python .github/scripts/check_paper_policy.py` to check the
actual staged index. It reads indexed ignore rules, so unstaged policy changes
cannot make an unsafe staged file pass. The pre-commit preflight uses
`--source-export` to check the isolated staged files, including their ignore rules.
CI performs the same publication check and its regression suite. Never
force-add private material. Ignore rules do not remove existing indexed files.

Check built wheels and source archives with
`python .github/scripts/check_paper_policy.py --artifacts <build-directory>`.
Both distributions exclude `papers/` and root `agents/`.

For path moves, update imports, loaders, instructions and CI in the same change;
preserve data bytes and run the relevant offline replication tests. Separate
methodological examples, partial reproduction requiring licensed inputs, and
exact frozen reproduction. Document code revision, data vintage/hash, parameters
and seeds where known; do not infer missing historical environment versions.
