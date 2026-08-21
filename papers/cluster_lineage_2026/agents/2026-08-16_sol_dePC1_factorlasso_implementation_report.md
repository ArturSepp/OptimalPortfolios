# de-PC1 FactorLasso implementation report

**Date:** 2026-08-16  
**Executor:** sol  
**Status:** COMPLETE - adopted as a default-off diagnostic/robustness specification  
**Stages:** D1-D3  
**FactorLasso checkout:** `C:/Users/artur/OneDrive/analytics/my_github/FactorLasso`  
**Repository actions:** no staging, commit, tag, push, build publication, or release

## Outcome

FactorLasso now has an opt-in, default-off cluster-correlation diagnostic that removes the
dominant principal component after exact point-in-time asset restriction and before cluster
smoothing. The default `NONE` path bypasses the new numerical work and preserves existing
results. De-PC1 is documented as a robustness specification rather than an automatic production
replacement. All three source release-metadata locations are `0.15.0`; nothing was installed,
published, or released.

## Public API

Added and exported:

- `ClusterCorrelationTransform.NONE`
- `ClusterCorrelationTransform.REMOVE_PC1`
- `ClusterCorrelationTransformResult`
- `remove_first_principal_component(corr_matrix)`
- `apply_cluster_correlation_transform(corr_matrix, transform=NONE)`
- `LassoModel.cluster_correlation_transform`
- `compute_rolling_smoothed_clusters(..., eligibility=None)`

The implemented order is:

```text
causal return history through t
  -> signed dependence matrix
  -> exact eligible-asset subset at t
  -> optional PC1 deflation and residual restandardisation
  -> temporal smoother
  -> distance/linkage/cut
```

For `PARTITION_BONUS`, the fixed partition bonus is applied after residual-correlation
distance construction. For `SIMILARITY_EWMA`, the residual correlations themselves are
smoothed. External partitions still bypass discovery.

## Numerical contract

For the eligible correlation matrix `R`, the implementation computes

```text
Q       = R - lambda_1 v_1 v_1'
d_i     = Q_ii
R_dePC1 = diag(d)^(-1/2) Q diag(d)^(-1/2)
```

It symmetrises before `numpy.linalg.eigh`, neutral-fills unavailable off-diagonal pairs at
zero, restores a unit diagonal, preserves labels/order, and records the removed eigenvalue,
variance share, eigengap, dominant-eigenspace uniqueness, minimum residual variance,
missing-pair count, and isolated residual assets. Material negative residual variance or
material correlation-bound violations raise; no nearest-correlation projection or silent
clipping was added. One-asset and numerical-floor assets follow the roadmap's identity/
isolation conventions.

The docstrings and README cite Plerou et al. (2002) and MacMahon and Garlaschelli (2015),
and explicitly distinguish one-component common-mode removal from RMT bulk filtering.

The documentation formalisation also makes the direct-versus-indirect effect explicit: de-PC1
does not residualise response returns, fitted loadings, or the assembled covariance matrix, but
a changed diagnostic partition can alter fitted outputs indirectly when HCGL/FCGL penalties or
cluster-pooled signs consume it. The README shows both the pure audit helper and the opt-in model
configuration, explains every result field, and distinguishes fixed-cutoff from matched-count
partition comparisons. `COMPATIBILITY.md` records the additive v0.15.0 public surface.

## Files changed

- `factorlasso/cluster_utils.py`
- `factorlasso/cluster_smoothing.py`
- `factorlasso/lasso_estimator.py`
- `factorlasso/__init__.py`
- `tests/test_cluster_pc1_removal.py`
- `README.md`
- `CHANGELOG.md`
- `COMPATIBILITY.md`
- `ROADMAP_CLUSTER_CORRELATION_PC1_REMOVAL.md`
- `pyproject.toml`
- `CITATION.cff`

Current source SHA-256 values:

| source | SHA-256 |
|---|---|
| `factorlasso/cluster_utils.py` | `ec8e0aac16a0d2d8174713ed4cf6bb6a3477bb1c20f959757a76a516f21e9c1c` |
| `factorlasso/cluster_smoothing.py` | `dc184d0231f6c509f7cd6885d8352dd73e14e1604fa4efc21b6d725fb3b081e7` |
| `factorlasso/lasso_estimator.py` | `e96ca80ed246238cdadad5b4b0c81962d94922b3396d38ed0b8e9cfe8b56b64b` |
| `README.md` | `151e580d41e3de7252293ead5823ef9c2d30b01901cdd7c1cf242b58b6003347` |
| `CHANGELOG.md` | `f7bf9f148bf7750848776fa544323a48c27090a05870e79f0fb69ddf7b4f20d8` |
| `COMPATIBILITY.md` | `856abb7a562470dfe526d557edb0083c5805eaea8f2d39893b1615a81493f4a6` |

The unrelated untracked `.claude/settings.local.json` was preserved unchanged.

## Fail-before-pass checkpoint

The new test module was first collected before implementation. Collection failed with the
expected `ImportError` because `ClusterCorrelationTransform` did not yet exist. After the
implementation, the new module passed all 19 tests. This proves the new suite was capable of
detecting the missing feature.

## Independent numerical reference

The matrix-deflation result was independently reconstructed in observation space for both
uniformly weighted and EWMA-weighted complete panels: standardise observations, remove the
PC1 score, and recompute residual correlations. Maximum absolute disagreement with matrix
deflation was `<= 1e-12` in both cases.

The test suite also covers a hand-computed rank-one-plus-block matrix, eigenvector-sign
invariance, label preservation, one asset, identity, perfectly common assets, missing pairs,
near-singular residuals, malformed inputs, strong-common-factor block recovery, future-data
invariance, ineligible-column isolation, direct versus rolling identity, and both smoother
orderings.

## Verification commands and results

```powershell
pytest tests/test_cluster_pc1_removal.py tests/test_cluster_smoothing.py `
  tests/test_dependence_measures.py -v
pytest --cov=factorlasso --cov-report=term-missing -q
ruff check factorlasso/ tests/
python -c "import factorlasso,sys; print(factorlasso.__file__); print([m for m in ('qis','optimalportfolios','sklearn') if m in sys.modules])"
```

| check | measured | tolerance | status |
|---|---:|---:|---|
| new de-PC1 tests after implementation | 19 passed | all pass | PASS |
| focused D3 command | 65 passed | all pass | PASS |
| full suite | all passed; 9 expected skips | all pass | PASS |
| line coverage | 92.67% | >= 90% | PASS |
| Ruff findings | 0 | 0 | PASS |
| base-import banned modules | 0 | 0 | PASS |
| local-source path assertion | 1 | 1 | PASS |
| uniform observation-space reference max error | <= 1e-12 | <= 1e-12 | PASS |
| EWMA observation-space reference max error | <= 1e-12 | <= 1e-12 | PASS |
| default-`NONE` focused regressions | byte-identical | byte-identical | PASS |
| version metadata agreement | 3/3 at `0.15.0` | 3/3 | PASS |

The focused paper-harness tests added later also pass (`9 passed`) and isolated
`E,F,W` Ruff is green, but those belong to D4-D5 rather than the package gate.

## Deviations and open items

No numerical implementation deviation. The borrowed OptimalPortfolios virtual environment still
contains installed FactorLasso distribution metadata `0.14.0`, so `factorlasso.__version__`
reports that installed metadata even while `factorlasso.__file__` resolves to the local 0.15.0
source checkout on `PYTHONPATH`. This is environment state, not a disagreement among the three
source version records. An editable install, wheel build/install test, and any release action
remain deferred because the owner requested local source implementation, not an environment
replacement or FactorLasso release.

## GATE REQUEST

None. D1-D3 acceptance passed and the owner already authorised the downstream empirical run.
