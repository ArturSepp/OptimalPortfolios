# Workspace recovery escalation — cluster-lineage untracked source deletion

Date: 2026-08-14

During E8 execution, a concurrent workspace event twice restored the broad
`/papers/cluster_lineage_2026/` ignore rule and removed untracked files below `agents/` and
`replication/`. This included the canonical roadmap, prior E0–E6 reports, and their Python
source. External empirical caches and stage outputs were not affected; interpreter-generated
CPython 3.12 bytecode in `replication/__pycache__/` also survived.

Actions taken:

- repeated E8a and E8b determinism checks against the surviving exact executed modules;
- restored the E7/E8 source entry points and added a local bytecode recovery loader;
- copied the preserved modules to `replication/recovery_bytecode/`, preventing normal Python
  cache refreshes from overwriting the recovery payload;
- restored the narrow `.gitignore` rule so only `data/` and `msci_us/` remain ignored;
- rewrote the E7, E8a and E8b reports from the measured external outputs;
- staged nothing and pushed nothing during the empirical execution. A subsequent owner
  instruction on 2026-08-14 added the recovery files and reports to the `main` index; no push
  was performed.

The E7/E8 results are numerically complete and validated. Before an archival release, the
owner copy of the canonical roadmap and the full E0–E7 source/report tree should be restored
and reconciled with these E7/E8 additions. The recovery wrappers intentionally preserve the
exact executed Python 3.12 modules rather than pretending to reconstruct deleted source from
memory.
