# F9 report — first scoped commit and exhibit-vintage tag

**Date:** 2026-08-21  
**Roadmap:** `agents/ROADMAP_manuscript_finalisation.md` v3  
**Status:** COMPLETE — awaiting OWNER GATE F9

## Scope and version-control actions

F9 ran on local branch `main`. The `.gitignore` cluster-lineage rule was narrowed so that
`papers/cluster_lineage_2026/agents/`, `paper/`, and `replication/` are trackable. The
following remain ignored: `data/`, `msci_us/`, Python bytecode and recovery bytecode,
compiled PDFs, and LaTeX build products.

The single local commit uses message:
`papers: execute cluster-lineage manuscript finalisation roadmap`.

The annotated exhibit-vintage tag is:
`cluster-lineage-exhibits-2026-08-21`.

No push, branch operation, release action, or staging outside the explicit F9 allowlist
occurred. Pre-existing unrelated changes on `main`, including the other `.gitignore`
hunks, were left unstaged and untouched.

## Staged-set audit

The final staged set immediately before the single commit was:

| Path group | Files |
|---|---:|
| `.gitignore` — cluster-lineage hunk only | 1 |
| `papers/cluster_lineage_2026/agents/` | 102 |
| `papers/cluster_lineage_2026/paper/` | 7 |
| `papers/cluster_lineage_2026/replication/` | 154 |
| **Total** | **264** |

Total staged working-tree bytes: `3,031,969` (`<25 MB`). The staged-path audit
found zero licensed data files, cache files, generated outputs, bytecode files, compiled
PDFs, LaTeX build products, office documents, email/message files, or editorial
correspondence. The `agents/` markdown files are the roadmaps, owner rulings, reviews,
execution reports, and escalation audit trail required by the reporting protocol.

## Acceptance checks

| Acceptance check | Measured | Tolerance | Result |
|---|---:|---:|---|
| Local branch | `main` | `main` | PASS |
| Commits created by F9 | 1 | 1 | PASS |
| Annotated tags created by F9 | 1 | 1 | PASS |
| Staged files | 264 | report; automatic escalation above 25 MB | PASS |
| Staged bytes | `3,031,969` | `<25 MB` | PASS |
| Disallowed staged paths | 0 | 0 | PASS |
| Data/cache/output/editorial files staged | 0 | 0 | PASS |
| Tracked source-set status after commit | clean, 0 changes | 0 | PASS |
| Tag target | the single F9 commit | must resolve | PASS |
| Pushes | 0 | 0 | PASS |

The repository-wide worktree remains dirty because unrelated owner/session edits predated
F9. That is outside the roadmap's scoped-cleanliness acceptance line; none was staged,
reverted, committed, or otherwise modified by F9.

## GATE REQUEST

OWNER GATE F9 is requested on the single local commit and annotated tag
`cluster-lineage-exhibits-2026-08-21`. Per the roadmap, executor work stops here; W3/W4/W5
manuscript integration and submission work remain on the owner/Claude track.
