"""Cluster-aware risk allocation methods used by the paper diagnostics.

The module keeps covariance estimation and cluster discovery fixed and changes only
portfolio allocation.  ``cluster_risk_budget`` maps a flat partition into the asset-level
budgets consumed by OptimalPortfolios' constrained risk-budgeting solver.  ``hrp_weights``
implements Lopez de Prado's inverse-variance recursive bisection.  The variance-HERC
implementation follows the actual dendrogram branches until the supplied flat partition,
then allocates inverse variance inside each terminal cluster.

These are paper-local research analytics.  Existing OptimalPortfolios risk-budgeting defaults
and public signatures are not changed.
"""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd
import qis
from scipy.cluster.hierarchy import leaves_list


def _validate_covar(covar: pd.DataFrame) -> None:
    """Require a finite, labelled, symmetric covariance with positive diagonal."""
    if not isinstance(covar, pd.DataFrame):
        raise TypeError("covar must be a pandas DataFrame")
    if covar.empty or covar.shape[0] != covar.shape[1]:
        raise ValueError(f"covar must be non-empty and square, got {covar.shape!r}")
    if not covar.index.equals(covar.columns) or not covar.index.is_unique:
        raise ValueError("covar index and columns must be identical unique asset labels")
    values = covar.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("covar must contain only finite values")
    if not np.allclose(values, values.T, atol=1e-12, rtol=1e-12):
        raise ValueError("covar must be symmetric")
    if np.any(np.diag(values) <= 0.0):
        raise ValueError("covar diagonal must be strictly positive")


def _validate_linkage(linkage: np.ndarray, n_assets: int) -> np.ndarray:
    """Return a validated SciPy linkage array for ``n_assets`` leaves."""
    values = np.asarray(linkage, dtype=float)
    expected = (max(n_assets - 1, 0), 4)
    if values.shape != expected:
        raise ValueError(f"linkage must have shape {expected}, got {values.shape!r}")
    if not np.isfinite(values).all():
        raise ValueError("linkage must contain only finite values")
    return values


def cluster_risk_budget(
    clusters: pd.Series,
    *,
    cluster_size_exponent: float,
) -> pd.Series:
    """Map a flat partition to asset budgets with cluster-size shrinkage.

    For cluster ``g`` with ``n_g`` valid assets, its total risk budget is
    ``B_g = n_g**alpha / sum_h(n_h**alpha)`` and each member receives ``B_g / n_g``.
    ``alpha=1`` is equal asset risk budgeting, ``alpha=0`` is equal cluster risk
    budgeting, and ``alpha=0.5`` is the square-root-size intermediate.
    """
    if not isinstance(clusters, pd.Series):
        raise TypeError("clusters must be a pandas Series")
    if not clusters.index.is_unique:
        raise ValueError("cluster asset labels must be unique")
    exponent = float(cluster_size_exponent)
    if not np.isfinite(exponent):
        raise ValueError("cluster_size_exponent must be finite")
    valid = clusters.dropna()
    if valid.empty:
        raise ValueError("at least one valid cluster membership is required")
    sizes = valid.value_counts(sort=False).astype(float)
    group_budgets = sizes.pow(exponent)
    group_budgets /= group_budgets.sum()
    budgets = valid.map(group_budgets / sizes).astype(float)
    budgets.name = "risk_budget"
    return budgets


def inverse_variance_weights(covar: pd.DataFrame) -> pd.Series:
    """Return unit-sum inverse-variance weights for one covariance block."""
    _validate_covar(covar)
    inverse = 1.0 / np.diag(covar.to_numpy(dtype=float))
    inverse /= inverse.sum()
    return pd.Series(inverse, index=covar.index, name="weight")


def _cluster_variance(covar: pd.DataFrame, assets: Iterable[object]) -> float:
    """Return the variance of the inverse-variance portfolio over ``assets``."""
    members = list(assets)
    block = covar.loc[members, members]
    weights = inverse_variance_weights(block).to_numpy()
    return float(weights @ block.to_numpy() @ weights)


def hrp_weights(covar: pd.DataFrame, linkage: np.ndarray) -> pd.Series:
    """Return canonical HRP weights for a supplied hierarchical linkage.

    Leaves are quasi-diagonalised using the linkage order.  Each ordered block is split
    in half recursively, and capital is assigned inversely to the two child-block
    variances.  Cluster variance uses the inverse-variance portfolio, matching the
    original volatility-HRP construction.
    """
    _validate_covar(covar)
    n_assets = len(covar)
    if n_assets == 1:
        return pd.Series(1.0, index=covar.index, name="weight")
    linkage = _validate_linkage(linkage, n_assets)
    ordered = covar.index[leaves_list(linkage)].tolist()
    weights = pd.Series(1.0, index=covar.index, name="weight")
    blocks: list[list[object]] = [ordered]
    while blocks:
        children: list[list[object]] = []
        for block in blocks:
            if len(block) <= 1:
                continue
            midpoint = len(block) // 2
            left = block[:midpoint]
            right = block[midpoint:]
            left_variance = _cluster_variance(covar, left)
            right_variance = _cluster_variance(covar, right)
            denominator = left_variance + right_variance
            if not np.isfinite(denominator) or denominator <= 0.0:
                raise ValueError("HRP child variances must have a positive finite sum")
            left_share = right_variance / denominator
            weights.loc[left] *= left_share
            weights.loc[right] *= 1.0 - left_share
            children.extend([left, right])
        blocks = children
    weights /= weights.sum()
    return weights


def _linkage_children(linkage: np.ndarray, n_assets: int) -> dict[int, tuple[int, int]]:
    """Return child-node identifiers for every non-leaf linkage node."""
    return {
        n_assets + row_number: (int(row[0]), int(row[1]))
        for row_number, row in enumerate(linkage)
    }


def _node_leaves(
    node: int,
    *,
    n_assets: int,
    children: dict[int, tuple[int, int]],
    memo: dict[int, tuple[int, ...]],
) -> tuple[int, ...]:
    """Return ordered leaf identifiers descending from one linkage node."""
    if node in memo:
        return memo[node]
    if node < n_assets:
        leaves = (node,)
    else:
        left, right = children[node]
        leaves = _node_leaves(
            left, n_assets=n_assets, children=children, memo=memo
        ) + _node_leaves(right, n_assets=n_assets, children=children, memo=memo)
    memo[node] = leaves
    return leaves


def herc_volatility_weights(
    covar: pd.DataFrame,
    linkage: np.ndarray,
    *,
    clusters: pd.Series,
) -> pd.Series:
    """Return variance-HERC weights using the supplied terminal partition.

    Terminal clusters use inverse-variance asset weights.  Their standalone variances
    are propagated up the pruned dendrogram by addition.  At every actual tree split,
    capital is divided inversely to the two child-subtree risks.  Recursion stops when
    all leaves below a node belong to one terminal cluster.
    """
    _validate_covar(covar)
    n_assets = len(covar)
    aligned = clusters.reindex(covar.index)
    if aligned.isna().any():
        missing = aligned.index[aligned.isna()].tolist()
        raise ValueError(f"clusters missing {len(missing)} covariance assets: {missing[:5]}")
    if n_assets == 1:
        return pd.Series(1.0, index=covar.index, name="weight")
    linkage = _validate_linkage(linkage, n_assets)
    children = _linkage_children(linkage, n_assets)
    leaf_memo: dict[int, tuple[int, ...]] = {}
    terminal_risk = {
        label: _cluster_variance(covar, members.index)
        for label, members in aligned.groupby(aligned, sort=False)
    }
    weights = pd.Series(0.0, index=covar.index, name="weight")

    def labels_below(node: int) -> tuple[object, ...]:
        """Return unique terminal labels below ``node`` in first-seen order."""
        leaves = _node_leaves(
            node, n_assets=n_assets, children=children, memo=leaf_memo
        )
        return tuple(pd.unique(aligned.iloc[list(leaves)]))

    def allocate(node: int, capital: float) -> None:
        """Allocate ``capital`` recursively through the pruned linkage tree."""
        labels = labels_below(node)
        leaves = _node_leaves(
            node, n_assets=n_assets, children=children, memo=leaf_memo
        )
        if len(labels) == 1:
            assets = covar.index[list(leaves)]
            inside = inverse_variance_weights(covar.loc[assets, assets])
            weights.loc[assets] += capital * inside
            return
        left, right = children[node]
        left_labels = labels_below(left)
        right_labels = labels_below(right)
        overlap = set(left_labels).intersection(right_labels)
        if overlap:
            raise ValueError(
                "terminal HERC cluster crosses a linkage split: "
                f"labels={sorted(map(str, overlap))[:5]}"
            )
        left_risk = float(sum(terminal_risk[label] for label in left_labels))
        right_risk = float(sum(terminal_risk[label] for label in right_labels))
        denominator = left_risk + right_risk
        if not np.isfinite(denominator) or denominator <= 0.0:
            raise ValueError("HERC child risks must have a positive finite sum")
        left_share = right_risk / denominator
        allocate(left, capital * left_share)
        allocate(right, capital * (1.0 - left_share))

    allocate(2 * n_assets - 2, 1.0)
    weights /= weights.sum()
    return weights


def risk_contribution_summary(
    weights: pd.Series,
    covar: pd.DataFrame,
    clusters: pd.Series,
) -> pd.DataFrame:
    """Aggregate Euler volatility contributions to the supplied clusters."""
    _validate_covar(covar)
    aligned_weights = weights.reindex(covar.index).fillna(0.0).astype(float)
    aligned_clusters = clusters.reindex(covar.index)
    if aligned_clusters.isna().any():
        raise ValueError("every covariance asset must have a cluster for risk aggregation")
    grouped_ratios = qis.compute_group_portfolio_risk_contribution_ratios(
        weights=aligned_weights,
        covar=covar,
        groups=aligned_clusters,
    )
    portfolio_volatility = float(np.sqrt(aligned_weights @ covar @ aligned_weights))
    grouped = grouped_ratios * portfolio_volatility
    absolute_total = float(grouped.abs().sum())
    if portfolio_volatility <= 0.0 or absolute_total <= 0.0:
        raise ValueError("portfolio must have positive volatility and aggregate risk")
    return pd.DataFrame(
        {
            "cluster": grouped.index,
            "risk_contribution": grouped.to_numpy(),
            "risk_share": grouped_ratios.to_numpy(),
            "absolute_risk_share": grouped.abs().to_numpy() / absolute_total,
            "portfolio_volatility": portfolio_volatility,
        }
    ).reset_index(drop=True)


def risk_concentration_metrics(summary: pd.DataFrame) -> dict[str, float]:
    """Summarise signed and absolute cluster-risk concentration."""
    absolute = summary["absolute_risk_share"].to_numpy(dtype=float)
    signed = summary["risk_share"].to_numpy(dtype=float)
    hhi = float(np.sum(np.square(absolute)))
    return {
        "portfolio_ex_ante_volatility": float(summary["portfolio_volatility"].iloc[0]),
        "cluster_risk_hhi_absolute": hhi,
        "effective_risk_clusters_absolute": 1.0 / hhi,
        "maximum_absolute_cluster_risk_share": float(np.max(absolute)),
        "negative_cluster_risk_share": float(-np.minimum(signed, 0.0).sum()),
        "cluster_risk_reconciliation_error": abs(float(signed.sum()) - 1.0),
    }
