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
| `universe.py` | the 18-asset paper universe, asset classes, paper admission policy, historical and current factor panels, common bootstrap window (Jul 2001 – Jun 2026, 300 months) |
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
alpha), equity_regional_addon, rf_rate. With M factors, `betas.csv` is 18 x M,
`factor_covar.csv` is M x M annualized, and `factor_premia.csv` is
M x (base, stress, upside). The frozen `2026q2` FAJ cut uses the legacy factor
panel; `2026q2_custom` remains the historical FAJ replication input.
JPM now pins `2026_matf_cma_revised_20260925`, extracted 25 September 2026.
FAJ retains `2026q2_custom_ig_hy_publication`, extracted 24 September 2026.
The historical eleven-factor, automatic-prior and fixed-IG-prior cuts remain
immutable. The current assets include `FactorPrior1`, `FactorPrior2`,
`pe_factor_exposure` and `long_only_betas` for the estimator adapter.
`asset_excess_logreturns.csv` / `asset_total_returns.csv` hold optional licensed
history panels; `factor_navs.csv` holds optional licensed factor NAVs. The
manifest stores source/input hashes, production configuration and per-file hashes.

The 25 September JPM revision uses the unchanged 30 June 2026 cutoff,
MATF_CUSTOM_IG_HY and the current production FCGL estimator. `FactorPrior1` and
`FactorPrior2` select single- or joint-factor weighted OLS priors; unmapped assets
use highest-R² selection. Nonzero prior signs override detected signs, with hard
economic and PE eligibility gates retained. IL selects Rates and Inflation jointly.
All 182 production indices are fitted before extracting the 18 paper sleeves.
The revised JPM fit uses per-response valid-EWMA-mass loss normalization, fixed
monthly/quarterly penalties and EWMA date-pooled signs. Long-run credit spreads
are IG 65 bp, HY 400 bp and EM 400 bp; net allowances are 21, 120 and 15 bp.
The archived FAJ snapshot retains its earlier settings.

PE alpha admission is 50%, its factor Sharpe anchor is 0.70, ILS admission is 100%,
and the discretionary EM equity haircut is zero. World and ACWI select `Equity`.
The global equity anchor is a fixed developed-market policy basket, not an ACWI
capitalization-weight identity. Credit premia apply verified outer MATF exposure
to both spread-minus-loss blend legs, with disclosed assumed tracker notional/NAV
of 1×. EM loss is the strategic assumption 1% PD × 50% LGD = 50 bp. The paper
shows ±20% inner-notional sensitivity; these inputs are not verified tracker
cash-flow identities or a benchmark-rating-based loss estimate.

All factor histories were rebuilt from the June input vintage. The owner
volatility-targeting covariance initialization depends on that supplied panel;
these are conditional descriptive histories, not a point-in-time backtest.
Earlier snapshots and originally distributed Q2 reports remain unchanged.
The FAJ draft adopted the publication snapshot on 25 September 2026.
This final rebuild is not a credit-split-only experiment.

Current manifest SHA-256: `863adc9693616913cfa4b77d6d1fd4117901f5722c6982ced07ad667cd4b9bb0`. See the
[JPM replication instructions](../matf_cma_jpm_2026/replication/README.md).

Local licensed input: `providers.csv` contains provider CMA vectors under neutral
labels A–D. It is ignored together with its extractor and name map; consumers
add the public Consensus vector in code.

## What ships publicly

`OptimalPortfolios` is a public repository, so the snapshot is committed
selectively.

| snapshot file | tracked | why |
|---|---|---|
| `assets.csv`, `betas.csv`, `factor_covar.csv`, `factor_premia.csv` | yes | the numbers the papers publish, and what a reader needs to verify them |
| `MANIFEST.json` | yes | provenance, config rows, per-file hashes |
| `asset_excess_logreturns.csv` | **no** | ~25 years of licensed index histories (MSCI, Bloomberg, ICE BofA, HFRI, Eurekahedge) |
| `asset_total_returns.csv` | **no** | same; retained for reporting and the production-export audit |
| `factor_navs.csv` | **no** | daily factor NAV histories; the same content was already untracked under its former path |
| `providers.csv` | **no** | licensed provider vectors under neutral labels A–D; local extractor and name map are ignored |

`loaders.py` treats every panel as OPTIONAL: it loads when present, is `None`
when absent, and `PaperInputs.require_panel(name)` raises a message naming the
file and the scripts that need it. `verify_manifest` hashes every file that IS
present and returns the absent ones, so tampering is still caught on everything
shipped, while a missing CONFIG file stays fatal.

Optional inputs required by the current and archived consumers:

| needs | scripts |
|---|---|
| nothing beyond the config files | factor/admission accounting, the public Consensus audit, optimisation and mandate/universe exhibits |
| `providers.csv` | complete provider A–D comparisons in `run_provider_exhibits` |
| `factor_navs` | `run_factor_history_exhibits` (all of J2), `run_snapshot_tables` (only `tab:factor_returns`) |
| `asset_excess_logreturns` | `run_consistency_exhibits` (J4d) |
| factor NAVs and asset excess returns | current `run_bootstrap_common_horizon`; archived `run_bootstrap_q2` |

The shared suite registers the publication tag in the existing manifest,
alignment and deliberate-tamper checks; the combined paper/data run passes 132 tests. Licensed panels
remain optional; their absence is reported rather than silently reconstructed.

To run a consumer requiring licensed histories, place the panel at `snapshots/<tag>/<name>.csv`
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
inputs = cma_data.load_snapshot(tag='2026q2_custom_ig_hy_publication') # current manifest-verified paper inputs
bench = cma_data.get_benchmark_weights(mandate='Balanced with Alts')
```

[Project](../../README.md) · [Software citation](../../CITATION.cff)

## 2026_matf_cma_equity_priors_20260925

Explicit Equity targets for all regional-equity sleeves. Same cutoff and approved economic assumptions as `2026_matf_cma_revised_20260925`. Earlier snapshots are preserved.
