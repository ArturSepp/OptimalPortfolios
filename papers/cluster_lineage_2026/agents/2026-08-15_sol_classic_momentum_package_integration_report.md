# Classic momentum package and article integration report

**Date:** 2026-08-15  
**Executor:** sol  
**Status:** COMPLETE  
**Package version:** 6.20.0  
**Repository state:** local working tree only; no staging or push

## Owner instruction and scope

The owner instructed that classic 12m-ex-1m momentum be added to the
`optimalportfolios` alpha layer following the existing momentum pattern and then used by
the cluster-lineage article analysis. This instruction explicitly supersedes, for this
work item only, the roadmap's earlier package-no-modification constraint.

The existing `compute_momentum_alpha` implementation and all of its defaults remain
unchanged. It continues to mean ROSAA's benchmark-relative, volatility-normalised,
EWMA-filtered risk-adjusted momentum. Classic fixed-window momentum is a separate public
signal rather than a mode flag on that function.

## Frozen signal definition

For a cadence-aligned log-return panel `r`, the new raw signal is

```text
classic_momentum(t) = r.shift(skip).rolling(lookback, min_periods=lookback).sum()(t)
```

The defaults are `lookback_periods=12` and `skip_periods=1`: exactly 12 completed
log-return observations enter the signal and the latest observation is excluded. There
is no benchmark subtraction, volatility normalisation, mean adjustment, short EWMA, or
long EWMA. The signal remains `NaN` until the complete fixed window is available.

The public return-panel primitive was necessary for the article. Reconstructing prices
from its already-frozen return panel changed missing-history masks for partially observed
stocks. Passing that return panel directly preserves its point-in-time `NaN` semantics
exactly.

## Package implementation

Primary implementation:

- `src/optimalportfolios/alphas/signals/classic_momentum.py`

Public API:

- `compute_classic_momentum_from_returns` -- exact fixed-window raw signal for an
  already sampled return panel;
- `compute_classic_momentum_alpha` -- global or fixed-group cross-sectional scores;
- `compute_classic_momentum_cluster_alpha` -- time-varying rolling-cluster scores;
- `profile_classic_momentum` and `ProfileSignal.CLASSIC_MOMENTUM`;
- `AlphaSignal.CLASSIC_MOMENTUM`, with lookback and skip forwarding through the alpha
  dispatcher and the single/multi backtest wrappers.

The price-based constructors accept either one return cadence or a per-asset cadence
Series. Lookback and skip settings may be scalars or cadence-keyed mappings. The mixed
cadence path computes each bucket independently, restores original column order, and
forward-fills only between that bucket's observation dates, matching the established
signal pattern.

The QIS stack remains authoritative for price-to-log-return conversion,
cross-sectional scoring, group splitting, and portfolio backtesting. The only local
calculation is the fixed rolling sum because QIS's risk-adjusted EWMA momentum primitive
cannot express a hard skipped-return window.

The two new optional dispatcher parameters were appended after every pre-existing public
parameter. A dedicated regression assertion locks the old positional argument order, so
the new feature does not silently reinterpret an existing positional call.

Public API documentation, mixed-frequency documentation, README examples, release notes,
and the three version locations were updated. The package, citation metadata, README
BibTeX entry, and editable-project entry in `uv.lock` now agree on version 6.20.0.

## Article integration

Central runner changed:

- `papers/cluster_lineage_2026/replication/run_u1_covar_grid_long_short_monthly.py`

Its `_classic_monthly_scores` helper now calls
`optimalportfolios.alphas.compute_classic_momentum_from_returns`; it no longer owns a
second vectorised implementation. These existing article runners reuse that helper and
therefore consume the package definition too:

- `papers/cluster_lineage_2026/replication/run_u1_bics_sector_comparison_classic.py`;
- `papers/cluster_lineage_2026/replication/run_u2_blackrock_signal_comparison.py`.

The U1 covariance grid was rerun from its existing 28 partition caches. There were no
covariance refits, cluster refits, eligibility changes, weight-construction changes, or
backtest-specification changes. The package migration produced 58 performance rows and
56 matched cluster-versus-global comparisons. All 58 construction acceptance rows remain
`PASS`.

The article's independent date-by-date history-slice reconstruction measures a maximum
raw-signal difference of `2.66453525910038e-15` against tolerance `1e-14`, with exact
`NaN`-mask agreement. All eight deterministic CSV artifacts are byte-identical to the
pre-migration outputs. Therefore every previously reported U1 grid performance number,
ranking, diagnostic, and acceptance verdict is frozen unchanged.

Output/cache directory consumed:

```text
C:\Users\artur\OneDrive\analytics\outputs\cluster_lineage_2026\e5b\covariance_frequency_span_grid\msci_us\long_short_grid_q_025_monthly_12m_skip1
```

The cached article pass took 368.81 seconds. The output directory remains outside the
repository.

## Verification

| acceptance line | measured | tolerance | verdict |
|:--|:--|:--|:--|
| independent classic formula | maximum error `2.665e-15` | `<= 1e-14` | PASS |
| classic signal missing-history mask | exact agreement | exact | PASS |
| hard-skip causality | perturbing only the skipped return leaves the formation signal unchanged | exact | PASS |
| mutation proof | replacing the hard shift by shift zero makes the defining test fail; correct code restored | failure required | PASS |
| standard/fixed-group/rolling-cluster raw panel | identical across score layers | exact | PASS |
| mixed-cadence coverage and original column order | all assets covered and ordered | exact | PASS |
| existing positional dispatcher parameters | unchanged prefix; new parameters appended | exact | PASS |
| focused package signal/dispatcher/backtest tests | 100% progress, exit code 0 | exit code 0 | PASS |
| new module line coverage | 100% | 100% | PASS |
| full installed-source package suite | 100% progress, exit code 0 | exit code 0 | PASS |
| full package line coverage | 99.03% | `>= 99.00%` | PASS |
| configured Ruff `F`, `TID251`, `TID253`, `ICN` gates | no findings | no findings | PASS |
| Interrogate docstring coverage | 100.0% | 100.0% | PASS |
| Sphinx strict build | succeeded with no warnings | zero warnings | PASS |
| version metadata tests | 4 passed | all pass | PASS |
| focused article integration tests | 10 passed | all pass | PASS |
| article grid coverage after migration | 28 cells, 58 performance rows | exact | PASS |
| article construction acceptance | 58/58 rows `PASS` | 100% | PASS |
| pre/post migration deterministic artifacts | 8/8 byte-identical | 8/8 | PASS |

The required second numerical pass used two independent references: explicit adjacent
log-price ratios in the package tests and explicit per-date return-history slicing in the
article validator. The mutation check temporarily removed the skip, observed the defining
test fail, and then restored the correct implementation before the final suite.

Final verification commands:

```text
python -m pytest --cov=optimalportfolios -q
ruff check --select F,TID251,TID253,ICN src/optimalportfolios
interrogate src/optimalportfolios
sphinx-build -W --keep-going -b html docs docs/_build/html
python -m pytest \
  papers/cluster_lineage_2026/replication/u1_covar_grid_long_short_monthly_test.py \
  papers/cluster_lineage_2026/replication/u1_bics_sector_comparison_classic_test.py \
  papers/cluster_lineage_2026/replication/u2_blackrock_signal_comparison_test.py -q
```

## Deviations and open items

The `uv` executable is not installed in this local environment. Because no dependency
changed, the only required lock edit was the editable project's own version from 6.19.0
to 6.20.0; it was updated directly. The editable install, version-metadata tests, full
suite, and strict documentation build all validate the resulting metadata.

There are no numerical deviations and no open implementation items. The ignored
`papers/cluster_lineage_2026/` tree remains local as instructed. Nothing was staged or
pushed.
