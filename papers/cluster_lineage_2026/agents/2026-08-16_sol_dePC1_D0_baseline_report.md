# de-PC1 D0 baseline and provenance report

**Date:** 2026-08-16  
**Executor:** sol  
**Status:** COMPLETE  
**Roadmap:** `agents/ROADMAP_factorlasso_depc1_clustering_and_backtest.md`  
**Repository actions:** no staging, commit, tag, push, publication, or baseline-cache write

## Outcome

The pre-change FactorLasso state and the three owner-frozen raw-correlation strategy
artifacts were frozen before the de-PC1 implementation. The local development checkout was
then placed first on `PYTHONPATH` for every implementation and empirical command.

## FactorLasso baseline provenance

| item | frozen value |
|---|---|
| checkout | `C:/Users/artur/OneDrive/analytics/my_github/FactorLasso` |
| branch | `main` |
| source commit | `8a5c152b97384b7491572e3c3832bfedc8bd8767` |
| package version before implementation | `0.14.0` |
| tracked-tree status before implementation | clean |
| preserved untracked local file | `.claude/settings.local.json` |
| activation | FactorLasso checkout first on `PYTHONPATH` |

Baseline Git blob identities:

| source | baseline Git blob |
|---|---|
| `factorlasso/cluster_utils.py` | `a9b94d2ce076ca885c0e1262a6d34e95395bead4` |
| `factorlasso/cluster_smoothing.py` | `371c37ed4cd49689eb2e0024a59997e524ab08da` |
| `factorlasso/lasso_estimator.py` | `b7f5f1640833abd70ff7a1caf776cd60436f6c37` |

The source-path assertion printed a path inside the checkout above, not `site-packages`.
The isolation import check showed none of `qis`, `optimalportfolios`, or `sklearn` loaded by
`import factorlasso`.

## Pre-change verification

Commands were run from the FactorLasso root with its existing environment:

```powershell
pytest tests/test_cluster_smoothing.py tests/test_dependence_measures.py -q
pytest -q
ruff check factorlasso/ tests/
python -c "import factorlasso,sys; print(factorlasso.__file__); print([m for m in ('qis','optimalportfolios','sklearn') if m in sys.modules])"
```

Measured results:

| check | measured | tolerance | status |
|---|---:|---:|---|
| clustering-focused baseline tests | 59 passed | all pass | PASS |
| full baseline suite | all passed; 9 expected skips | all pass | PASS |
| Ruff | 0 findings | 0 | PASS |
| local source-path assertion | 1 | 1 | PASS |
| banned modules loaded by base import | 0 | 0 | PASS |

## Frozen raw-strategy artifacts

These are accepted, pre-existing outputs. D0 read and hashed them; it did not refit or
rewrite them.

| universe | accepted raw operating row | net annual return | RF=0 Sharpe | artifact SHA-256 |
|---|---|---:|---:|---|
| U1 MSCI US | ME/span36, exact monthly U1 production signal, q=0.25 | -1.6204% | -0.2053 | `e8e0c6c7bb849a7926ffc8712ae76499472a1990f779cc277c43cc4f1e47e37c` |
| U2 BlackRock | AUM100 hybrid, W-THU/span156, q=0.25 | +0.1925% | +0.0620 | `ecc5ddaa5f20ac5f72b4b4773821504b096cd1c64d4ca5720d9c82199bc43dfa` |
| U3 futures | M1-star, 30/30/30/10, q=0.25 | +0.0297% | +0.0179 | `cc76c3d5afe0f1994db0405dc2da0cfecd64973b03a468e85690c208df6ed35c` |

Artifact paths:

- U1: `$CLUSTER_LINEAGE_OUTPUT_DIR/e5b/covariance_frequency_span_grid/msci_us/long_short_grid_q_025_prod_12m/performance.csv`
- U2: `$CLUSTER_LINEAGE_OUTPUT_DIR/e5b/covariance_frequency_span_grid/blackrock_us_etfs/aum50_filter_20260816/threshold_sensitivity/performance.csv`
- U3: `$CLUSTER_LINEAGE_OUTPUT_DIR/e5b/futures_prod_signal_grid_30_30_30_10_10bp_u1_window/best_relative_instrument_pnl_owner_exclusions_20260815/performance.csv`

U1's frozen machine-readable signal definition is
`replication/empirical_specs.py::U1_OPTIMAL_SPEC`: ME, long span 12, volatility span 13,
no short span, and `MeanAdjType.NONE`. This distinction is binding; the later exploratory
BICS transfer used EWMA and is not the U1 signal frozen for this experiment.

## Acceptance

| check | measured | tolerance | status |
|---|---:|---:|---|
| FactorLasso branch is `main` | 1 | 1 | PASS |
| source commit recorded | 1 | 1 | PASS |
| dirty pre-existing local file preserved | 1 | 1 | PASS |
| focused tests | 59 passed | all pass | PASS |
| full suite | all passed, 9 skipped | all pass | PASS |
| Ruff findings | 0 | 0 | PASS |
| local-source import | 1 | 1 | PASS |
| three accepted performance artifacts hashed | 3 | 3 | PASS |
| baseline caches written or changed by D0 | 0 | 0 | PASS |

## Deviations and open items

None. The de-PC1 experiment writes only below
`$CLUSTER_LINEAGE_OUTPUT_DIR/depc1/`; the three paths above remain read-only references.

## GATE REQUEST

None. D0 was an internal reproducibility gate and all acceptance lines passed.
