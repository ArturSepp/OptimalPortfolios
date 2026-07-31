# cma_data — the shared data layer of the paper packages

One universe, one benchmark construction, one loader API, versioned
immutable snapshots. Both paper packages (`matf_cma_jpm_2026`,
`achievable_sharpe_faj_2026`) consume this layer; neither defines its own
universe or reads production files.

```
prod pipeline (private) --_local_ extractor--> snapshots/<tag>/  (frozen csv + MANIFEST.json)
                                                     |
                              loaders.load_snapshot(tag)  ->  PaperInputs
```

## Contents

| file | role |
|---|---|
| `universe.py` | the 18-asset paper universe, asset classes, paper admission policy, nine-factor panel, common bootstrap window (Jul 2001 – Jun 2026, 300 months) |
| `benchmarks.py` | the eight mandate benchmarks via the two-level construction of the JPM paper's Appendix F |
| `loaders.py` | `load_snapshot(tag)` with sha256 manifest verification; the `PaperInputs` container |
| `local_path.py` | path resolution via optional flat `settings.yaml` (key `SNAPSHOTS_PATH`); zero-config defaults |
| `_local_extract_from_prod.py` | UNTRACKED: production workbook + factor NAVs -> a new snapshot |
| `tests/` | universe counts, benchmark identities, manifest tampering detection, loader parity |
| `snapshots/<tag>/` | immutable frozen cuts (see schema below) |

## Snapshot schema (all decimal per annum unless stated)

`assets.csv` (index ticker): sleeve, name, asset_class, frequency, alpha
(raw Jensen alpha, EWMA residual mean), resid_vol, total_vol, r2,
w_workbook (production admission), w_paper (paper policy, PE recut 0.5),
factor_excess_cma (factor-implied incl. regional add-ons, excl. admitted
alpha), equity_regional_addon, rf_rate. `betas.csv` 18x9. `factor_covar.csv`
9x9 annualized. `factor_premia.csv` 9 x (base, stress, upside).
`asset_excess_logreturns.csv` / `asset_total_returns.csv`: return panels on
the bootstrap window (quarterly assets carry NaN off-quarter).
`factor_navs.csv`: daily factor NAVs, base 100. `MANIFEST.json`: source
workbook name and sha256, the production config_snapshot rows, package
versions, per-file sha256 (verified on every load).

Pending addition: `providers.csv` (provider CMA vectors under neutral labels
A–D plus Consensus; the name map stays untracked with the extractor).

## What ships publicly

`OptimalPortfolios` is a public repository, so the snapshot is committed
selectively.

| snapshot file | tracked | why |
|---|---|---|
| `assets.csv`, `betas.csv`, `factor_covar.csv`, `factor_premia.csv` | yes | the numbers the papers publish, and what a reader needs to verify them |
| `MANIFEST.json` | yes | provenance, config rows, per-file hashes |
| `asset_excess_logreturns.csv` | **no** | ~25 years of licensed index histories (MSCI, Bloomberg, ICE BofA, HFRI, Eurekahedge) |
| `asset_total_returns.csv` | **no** | same; no current consumer |
| `factor_navs.csv` | **no** | daily factor NAV histories; the same content was already untracked under its former path |
| `providers.csv` (pending) | **no** | licensed provider vectors, gated on the per-provider provenance confirmation |

`loaders.py` treats every panel as OPTIONAL: it loads when present, is `None`
when absent, and `PaperInputs.require_panel(name)` raises a message naming the
file and the scripts that need it. `verify_manifest` hashes every file that IS
present and returns the absent ones, so tampering is still caught on everything
shipped, while a missing CONFIG file stays fatal.

What that costs a public checkout, measured:

| needs | scripts |
|---|---|
| nothing beyond the config files | `governed_cma_projection`, `exhibit_cap3_projection`, `consensus_decomposition`, `excess_vs_total_optimisation`, `run_optimisation`, `run_mandate_exhibits`, `run_universe_exhibits`, `run_provider_exhibits` — **8 of 12 run in full** |
| `factor_navs` | `run_factor_history_exhibits` (all of J2), `run_snapshot_tables` (only `tab:factor_returns`) |
| `asset_excess_logreturns` | `run_consistency_exhibits` (J4d) |
| both | `run_bootstrap_q2` (J5) |

Test suites stay green either way: `cma_data/tests` 12 passed, the JPM parity
harness 16 passed with the panels and 15 passed + 1 skipped without them.

To run the four gated scripts, place the panel at `snapshots/<tag>/<name>.csv`
from the production extract; the manifest hash is verified on load, so a wrong
file fails loudly.

## Rules

- **Snapshots are immutable and append-only.** A new production cut is a NEW
  tag. Each paper pins its tag in one place and never reads a mutable
  "latest". Regenerating a cut moves no published number.
- **The extractor is the only file that touches production data**, and the
  committed snapshot carries only the 18 paper assets.
- **Anonymisation lives in the data layer**: provider data enters snapshots
  under neutral labels, never under names.
- **Consumers import by file location** through their own `local_path.py`
  (importlib with `submodule_search_locations`; no `sys.path` mutation).
  `settings.yaml` is optional and naturally untracked (`*.yaml` is ignored
  repo-wide); defaults resolve relative to the repository layout.

## Usage from a paper package

```python
from local_path import load_cma_data          # the paper's own local_path.py
cma_data = load_cma_data()
inputs = cma_data.load_snapshot(tag='2026q2') # manifest-verified PaperInputs
bench = cma_data.get_benchmark_weights(mandate='Balanced with Alts')
```
