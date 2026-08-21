# F10 — draft replication statement

**Date:** 2026-08-21  
**Roadmap:** `agents/ROADMAP_manuscript_finalisation.md` v2  
**Status:** COMPLETE; manuscript wording remains subject to owner sign-off under W2

## Manuscript-ready paragraph

The empirical analysis uses licensed MSCI constituent data, Bloomberg-derived fund and futures data, and factor returns from the Ken French Data Library; the Ken French series are the only freely obtainable data input. The licensed MSCI and Bloomberg inputs cannot be redistributed, and researchers seeking to reproduce the empirical results must obtain the corresponding data licences and construct the documented point-in-time panels. The `factorlasso` implementation of Rolling-Ward clustering, smoothing calibration, and cluster-lineage analytics is publicly available. The paper-specific replication harness is available from the authors on request and provides the versioned scripts, frozen specifications, provenance records, deterministic seeds, and exhibit-building instructions needed to reproduce the analysis once the licensed inputs have been supplied; the request package does not contain or purport to provide the proprietary source data.

## Request-package checklist

When the gated stages exist, the request package will contain:

- the complete F8 exhibit tree, including the manuscript CSV tables, vector EPS figures,
  PDF figure copies, and `exhibit_index.csv` provenance map;
- the tracked `papers/cluster_lineage_2026/replication/` folder exactly as recorded at the
  local F9 tag, including runners, tests, frozen specification registries, deterministic
  seeds, and environment/provenance instructions;
- a pointer to the public `factorlasso` release used by the tagged harness;
- retrieval instructions for the freely obtainable Ken French factor input;
- schema and construction instructions for the licensed point-in-time panels, without any
  redistributed MSCI or Bloomberg observations.

The requester must supply under their own licences:

- MSCI constituent data required for the point-in-time U1 equity universe;
- Bloomberg-derived fund data, including the return, classification, and assets-under-
  management fields required for U2;
- Bloomberg-derived futures data and associated metadata required for U3;
- any other licensed observations needed to instantiate the documented schemas.

## Acceptance record

Source documents: owner ruling 15 in `agents/ROADMAP_manuscript_finalisation.md`, the F0
inventory at
`C:/Users/artur/OneDrive/analytics/outputs/cluster_lineage_2026/finalisation/f0/cache_inventory.csv`,
and the live manuscript `paper/paper.tex`. No computational runner or cache was required for
this documentation stage.

| Acceptance check | Measured | Required | Result |
|---|---:|---:|---|
| Claims that licensed data ship publicly | 0 | 0 | PASS |
| Freely obtainable input families identified | 1: Ken French | exactly 1 | PASS |
| Proprietary source families identified | 2: MSCI, Bloomberg | 2 | PASS |
| Public implementation identified | `factorlasso` | `factorlasso` | PASS |
| Paper harness availability | authors on request | request-based | PASS |
| Required package components named | F8 tree + F9-tagged replication folder | both | PASS |
| Requester licensing obligations stated | MSCI + Bloomberg | both | PASS |
| New data, cache, or empirical output written | 0 | 0 | PASS |
| Git staging, commit, tag, or push | 0 | 0 | PASS |

This stage has no gate. The paragraph can replace the current manuscript data-availability
draft only after the owner signs off in W2 and the F8/F9 package references exist.
