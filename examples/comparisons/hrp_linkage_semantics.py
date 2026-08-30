"""Compare the two linkage conventions commonly called HRP.

The Lopez de Prado (2016) code snippet passes an ``N x N`` correlation-distance
matrix ``D`` directly to ``scipy.cluster.hierarchy.linkage``. SciPy interprets
the rows as observations, so the explicit equivalent is ``linkage(pdist(D))``:
the tree is built from distances between whole distance profiles.

The direct-distance convention condenses ``D`` first and clusters on its entries
``d_ij``. FactorLasso implements this convention with a public correlation-to-
distance transform; PyPortfolioOpt's HRP implementation uses the same geometry.
OptimalPortfolios deliberately accepts a caller-supplied linkage, so both trees
below use the same OP allocation routine and isolate the effect of tree construction.

The default example uses only OptimalPortfolios core dependencies and runs in CI.
The external PyPortfolioOpt replication is opt-in and imported inside its check:

    uv run --python 3.12 --isolated --with 'PyPortfolioOpt==1.6.0' \
        --with 'scipy<1.18' \
        python examples/comparisons/hrp_linkage_semantics.py --check-pypfopt

The isolated SciPy constraint is currently necessary because PyPortfolioOpt 1.6.0
accesses a private SciPy linkage attribute removed in SciPy 1.18. This is an
implementation comparison, not evidence about out-of-sample performance.
"""

from __future__ import annotations

import argparse
import warnings

import numpy as np
import pandas as pd
from scipy.cluster import hierarchy
from scipy.spatial import distance as spatial_distance

from factorlasso import DistanceTransform, compute_clusters_from_corr_matrix
from optimalportfolios import compute_hierarchical_risk_parity_weights


PYPFOPT_COMMAND = (
    "uv run --python 3.12 --isolated --with 'PyPortfolioOpt==1.6.0' "
    "--with 'scipy<1.18' "
    "python examples/comparisons/hrp_linkage_semantics.py --check-pypfopt"
)


def _fixture() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return a positive-definite four-asset correlation and covariance fixture."""
    labels = pd.Index(list("ABCD"), name="asset")
    corr = pd.DataFrame(
        [
            [1.000000, 0.186665, -0.566601, 0.116919],
            [0.186665, 1.000000, 0.221383, 0.191615],
            [-0.566601, 0.221383, 1.000000, 0.185620],
            [0.116919, 0.191615, 0.185620, 1.000000],
        ],
        index=labels,
        columns=labels,
    )
    vols = np.array([0.10, 0.14, 0.18, 0.23])
    covar = pd.DataFrame(
        np.outer(vols, vols) * corr.to_numpy(),
        index=labels,
        columns=labels,
    )
    return corr, covar


def _correlation_distance(corr: pd.DataFrame) -> np.ndarray:
    """Return the Lopez de Prado correlation distance with an exact zero diagonal."""
    values = corr.to_numpy(dtype=float)
    distance = np.sqrt(np.clip((1.0 - values) / 2.0, 0.0, None))
    np.fill_diagonal(distance, 0.0)
    return distance


def _profile_distance_linkage(distance: np.ndarray) -> np.ndarray:
    """Reproduce the 2016 square-input snippet through its explicit SciPy semantics."""
    profile_distances = spatial_distance.pdist(distance, metric="euclidean")
    return hierarchy.linkage(profile_distances, method="single")


def _direct_distance_linkage(corr: pd.DataFrame) -> np.ndarray:
    """Build the direct correlation-distance tree through FactorLasso's public API."""
    _, linkage, _ = compute_clusters_from_corr_matrix(
        corr_matrix=corr,
        linkage_method="single",
        distance_transform=DistanceTransform.CHORD,
    )
    return linkage


def _clades(linkage: np.ndarray, n_assets: int) -> frozenset[frozenset[int]]:
    """Return non-root descendant sets, a leaf-order-invariant tree representation."""
    descendants = {leaf: frozenset({leaf}) for leaf in range(n_assets)}
    clades: set[frozenset[int]] = set()
    for row, (left, right, _, _) in enumerate(linkage):
        merged = descendants[int(left)] | descendants[int(right)]
        descendants[n_assets + row] = merged
        if len(merged) < n_assets:
            clades.add(merged)
    return frozenset(clades)


def _format_clades(
        clades: frozenset[frozenset[int]],
        labels: pd.Index,
        ) -> str:
    """Format canonical clades with asset labels for readable console output."""
    labelled = [tuple(labels[i] for i in sorted(clade)) for clade in clades]
    labelled.sort(key=lambda clade: (len(clade), clade))
    return ", ".join("{" + ", ".join(clade) + "}" for clade in labelled)


def _verify_linkage_semantics(
        distance: np.ndarray,
        profile_linkage: np.ndarray,
        direct_linkage: np.ndarray,
        ) -> None:
    """Verify both explicit constructions against the square-input conventions."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", hierarchy.ClusterWarning)
        implicit_square_input = hierarchy.linkage(distance, method="single")
    np.testing.assert_allclose(implicit_square_input, profile_linkage)

    direct_condensed = spatial_distance.squareform(distance, checks=False)
    scipy_direct = hierarchy.linkage(direct_condensed, method="single")

    # FactorLasso's CHORD distance is exactly 2 * D. Single linkage therefore
    # preserves every merge and only doubles the linkage heights.
    np.testing.assert_allclose(direct_linkage[:, [0, 1, 3]], scipy_direct[:, [0, 1, 3]])
    np.testing.assert_allclose(direct_linkage[:, 2], 2.0 * scipy_direct[:, 2])


def _verify_weights(weights: pd.Series) -> None:
    """Require a finite, fully invested, long-only allocation."""
    values = weights.to_numpy(dtype=float)
    if not np.isfinite(values).all() or np.any(values < 0.0):
        raise AssertionError("HRP weights must be finite and non-negative")
    np.testing.assert_allclose(values.sum(), 1.0, atol=1e-12)


def _compare_with_pypfopt(
        covar: pd.DataFrame,
        direct_linkage: np.ndarray,
        direct_weights: pd.Series,
        ) -> None:
    """Optionally verify PyPortfolioOpt against the direct-distance OP allocation."""
    try:
        from pypfopt import HRPOpt
    except ModuleNotFoundError as exc:
        if exc.name != "pypfopt":
            raise
        raise RuntimeError(
            "PyPortfolioOpt is not installed. Run the isolated command:\n"
            f"  {PYPFOPT_COMMAND}"
        ) from exc

    optimiser = HRPOpt(cov_matrix=covar)
    try:
        raw_weights = optimiser.optimize(linkage_method="single")
    except AttributeError as exc:
        if "_LINKAGE_METHODS" not in str(exc):
            raise
        raise RuntimeError(
            "This PyPortfolioOpt version uses scipy.cluster.hierarchy._LINKAGE_METHODS, "
            "which SciPy 1.18 removed. Run the isolated command:\n"
            f"  {PYPFOPT_COMMAND}"
        ) from exc

    pypfopt_weights = pd.Series(raw_weights, dtype=float).reindex(covar.index)
    np.testing.assert_allclose(pypfopt_weights, direct_weights, atol=1e-12)
    if _clades(np.asarray(optimiser.clusters), len(covar)) != _clades(
            direct_linkage, len(covar)
            ):
        raise AssertionError("PyPortfolioOpt and FactorLasso built different direct trees")
    print("PyPortfolioOpt check: direct tree and weights match.")


def run_example(check_pypfopt: bool = False) -> None:
    """Compare both linkage conventions and optionally run the external replication."""
    corr, covar = _fixture()
    distance = _correlation_distance(corr)
    profile_linkage = _profile_distance_linkage(distance)
    direct_linkage = _direct_distance_linkage(corr)
    _verify_linkage_semantics(distance, profile_linkage, direct_linkage)

    profile_clades = _clades(profile_linkage, len(corr))
    direct_clades = _clades(direct_linkage, len(corr))
    if profile_clades == direct_clades:
        raise AssertionError("fixture must distinguish the two tree topologies")

    profile_weights = compute_hierarchical_risk_parity_weights(covar, profile_linkage)
    direct_weights = compute_hierarchical_risk_parity_weights(covar, direct_linkage)
    _verify_weights(profile_weights)
    _verify_weights(direct_weights)

    np.testing.assert_allclose(
        profile_weights.to_numpy(),
        [0.617773135213, 0.139769931308, 0.190670720745, 0.051786212734],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        direct_weights.to_numpy(),
        [0.520598106856, 0.237386172078, 0.143603980640, 0.098411740426],
        atol=1e-12,
    )

    comparison = pd.DataFrame(
        {
            "profile_distance": profile_weights,
            "direct_distance": direct_weights,
        }
    )
    l1_difference = float((profile_weights - direct_weights).abs().sum())

    print("HRP linkage semantics")
    print("  2016 square input : linkage(pdist(rows of D))")
    print("  direct distance   : FactorLasso CHORD, same tree as squareform(D)")
    print(f"  profile clades    : {_format_clades(profile_clades, corr.index)}")
    print(f"  direct clades     : {_format_clades(direct_clades, corr.index)}")
    print("\nWeights:")
    print(comparison.to_string(float_format=lambda value: f"{value:.6f}"))
    print(f"\nL1 weight difference: {l1_difference:.6f}")

    if check_pypfopt:
        _compare_with_pypfopt(covar, direct_linkage, direct_weights)


def main() -> None:
    """Parse the optional external-check flag and run the example."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-pypfopt",
        action="store_true",
        help="also compare against a locally installed PyPortfolioOpt",
    )
    args = parser.parse_args()
    run_example(check_pypfopt=args.check_pypfopt)


if __name__ == "__main__":
    main()
