# E7 execution report — exhibit assembly and traceability

Date: 2026-08-14
Runner: `papers/cluster_lineage_2026/replication/build_exhibits.py`
Output: `$CLUSTER_LINEAGE_OUTPUT_DIR/exhibits/`

## Outcome

E7 passes. The runner emitted one canonical workbook for each gated universe and a
12-row `exhibit_index.csv` (four claim families × three universes). E8 results are not
cited anywhere in the E7 workbooks or index, as required before OWNER GATE E8.

| Acceptance line | Measured | Tolerance | Status |
|---|---:|---:|---|
| Canonical universe workbooks | 3 | 3 | PASS |
| Claim-family summaries per workbook | 4 | 4 | PASS |
| Indexed prospective exhibits | 12 | 12 | PASS |
| Traceable index rows | 12 | 12 | PASS |
| Orphan/missing exhibits | 0 | 0 | PASS |
| Sampled-sheet reopen checks | 3 | 3 | PASS |
| E8 citations before gate | 0 | 0 | PASS |

The claim sheets contain only manuscript-quote numbers. Each row identifies its source
script and its source-data sheet within the canonical workbook. Source sheets retain the
gated E3–E6 rows used by the summary.

Workbook authoring used the owner-sanctioned repository convention
`qis.save_df_to_excel`; the artifact-tool runtime was unavailable in this session. Each
workbook's `C1` sheet was reopened programmatically after writing.

Artifacts: `msci_us_canonical_exhibits_20260814.xlsx`,
`futures_canonical_exhibits_20260814.xlsx`, `mac_canonical_exhibits_20260814.xlsx`, and
`exhibit_index.csv`.

## Gate request

OWNER GATE E7: approve the 12 indexed exhibits for manuscript hand-off and the adversarial
pass. E8 evidence remains outside these claim-family summaries pending OWNER GATE E8.
