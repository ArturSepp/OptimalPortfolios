---
myst:
  html_meta:
    description: >-
      Hierarchical risk parity and cluster risk budgets in optimalportfolios: recursive
      bisection over a supplied linkage, which distance the linkage is built on, the covariances
      HRP never reads, group budgets split equally within groups, why HRP takes no constraints,
      and a verified offline example against ERC.
---

# Hierarchical risk parity and cluster risk budgets

*Author: [Artur Sepp](https://github.com/ArturSepp)*

Hierarchical risk parity and group risk budgets are implemented in
[OptimalPortfolios](https://github.com/ArturSepp/OptimalPortfolios).
Software citation: [CITATION.cff](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).

## Overview

Two functions turn a clustering of the assets into an allocation. Hierarchical risk parity (HRP)
orders the assets by a cluster tree and splits the capital recursively, inversely to the variance
of each half. Group, or cluster, risk budgets give each group a share of the portfolio's risk and
split it equally among the group's members; the risk-budgeting solver then finds the weights. HRP
is a closed-form rule without constraints; the budget route is an optimisation that accepts a
`Constraints` object.

This page states what the HRP function computes and proves three properties: its weights are
positive and fully invested, it never reads the covariances between the two halves of its first
split, and with uncorrelated assets it is the inverse-variance portfolio. It explains which
distance the tree must be built on, because the same SciPy call supports two readings that build
different trees. The worked example compares HRP, equal risk contributions (ERC) and equal cluster
budgets on a universe of three correlated blocks.

Clustering itself belongs to [factorlasso](https://github.com/ArturSepp/FactorLasso), whose
[cluster discovery](https://factorlasso.readthedocs.io/en/latest/cluster_discovery.html) page
covers the dependence measure, the distance transform, the linkage and the cut.
[qis](https://github.com/ArturSepp/QuantInvestStrats) computes the risk contributions used to
compare the allocations.

## Inputs, notation, and assumptions

| Convention | This article |
|---|---|
| Return basis | None; neither function samples returns. HRP takes a covariance matrix and a linkage, and the group budgets take labels only |
| Estimation grid | None; the covariance, the correlations behind the linkage and the cluster labels are estimated outside these functions. The examples use fixed synthetic matrices |
| Rebalancing grid | One allocation per `compute_hierarchical_risk_parity_weights` call; there is no rolling HRP. A date-by-asset label panel gives one budget row per date, which `rolling_risk_budgeting` solves at its covariance dates |
| Covariance units | Any consistent units: HRP weights do not change when the covariance is multiplied by a positive constant, and no variance floor applies. Annual, fractional return squared in the examples |
| Expected returns | None |
| Weight state | HRP returns long-only, fully invested target weights, positive for every asset. The group function returns risk budgets, not weights: non-negative, summing to one, zero for an unclassified asset |
| Solver | None: HRP is a closed-form recursive bisection and the budgets are a formula. The budget-driven weights of the examples use the risk-budgeting solver (cyclical coordinate descent, or ADMM when a bound is set) |

The notation follows the [conventions page](conventions.md#notation). In addition:

| Symbol | Meaning |
|---|---|
| $\pi$ | Leaf order of the linkage: its assets from left to right in the dendrogram |
| $S$, $L$, $R$ | An ordered set of assets being split, and its left and right halves |
| $\tilde w^{S}$ | Inverse-variance portfolio of the set $S$ |
| $V(S)$ | Variance of $\tilde w^{S}$ |
| $\theta_L$, $\theta_R$ | Shares of the capital of $S$ given to its halves |
| $\rho_{ij}$ | Correlation of assets $i$ and $j$ |
| $d_{ij}$, $\tilde d_{ij}$ | Correlation distance, and the distance between rows of the matrix of $d_{ij}$ |
| $\phi_i$ | Risk share $\mathrm{RC}_i / \sigma(w)$ of asset $i$ |
| $G_g$, $n_g$ | Classified assets of group $g$ and their number |
| $B_g$, $\eta$ | Aggregate risk budget of group $g$, and the exponent `group_size_exponent` |

Supply a labelled covariance DataFrame with the same unique labels on both axes, finite entries,
symmetric and with positive variances, and a linkage built from the same assets, whose leaf $k$ is
the asset in row $k$ of the covariance. HRP checks the covariance and the linkage and raises
instead of filtering. Labels, like the covariance, must be available at the decision date.

## Methodology

### Recursive bisection

`compute_hierarchical_risk_parity_weights(covar, linkage)` implements the allocation stage of
hierarchical risk parity from [López de Prado (2016)](https://doi.org/10.3905/jpm.2016.42.4.059).
It reads the leaf order $\pi$ of the linkage (SciPy's `leaves_list`). Assets that the tree joins
early sit next to each other in this order, which the paper calls quasi-diagonalisation.

Starting from the whole ordered universe with capital one, every ordered set $S$ with at least two
assets is split at its midpoint: the left half $L$ holds the first
$\lfloor \lvert S \rvert / 2 \rfloor$ assets and the right half $R$ the rest. Each half is valued
by the variance of its inverse-variance portfolio,

$$
\tilde w^{S}_i = \frac{\sigma_i^{-2}}{\sum_{j \in S} \sigma_j^{-2}} \quad (i \in S),
\qquad
V(S) = \sum_{i, j \in S} \tilde w^{S}_i \tilde w^{S}_j \Sigma_{ij},
$$

and the capital of $S$ is divided inversely to the two variances:

$$
\theta_L = \frac{V(R)}{V(L) + V(R)},
\qquad
\theta_R = 1 - \theta_L = \frac{V(L)}{V(L) + V(R)}.
$$

The weight of an asset is the product of the shares of the halves that contain it, one for each
split on its path from the universe down to the asset.

**Proposition 1 (full investment).** The weights are positive and sum to one.

**Proof.** Every $V(S)$ is positive, since the function raises `ValueError` otherwise, so
$0 \lt \theta_L \lt 1$. A split passes the whole capital of $S$ to its halves because
$\theta_L + \theta_R = 1$, so by induction over the splits the weights of the assets in $S$ add up
to the capital of $S$, and the universe has capital one. The function divides by the sum at the
end, which removes only rounding. $\square$

**Proposition 2 (unread covariances).** Given the order $\pi$, the weights depend on $\Sigma$ only
through the variances $V(S)$ of the halves. A covariance between an asset on one side of the first
split and an asset on the other side enters none of them: the first split allocates as if its two
halves were uncorrelated. Every other covariance lies inside a half and can enter.

**Proof.** The universe is split but is never a half, so its variance is not computed. Every half
lies on one side of the first split, and $V(S)$ reads only entries of $\Sigma$ with both assets
in $S$. $\square$

**Proposition 3 (uncorrelated assets).** If $\Sigma$ is diagonal, HRP returns the
inverse-variance weights $w_i = \sigma_i^{-2} / \sum_j \sigma_j^{-2}$ for every linkage, and the
risk share of each asset equals its weight, $\phi_i = w_i$.

**Proof.** Write $P(S) = \sum_{j \in S} \sigma_j^{-2}$. With $\Sigma$ diagonal, $V(S) = 1 / P(S)$,
so $\theta_L = P(L) / (P(L) + P(R)) = P(L) / P(S)$. By induction, a set $S$ on the path receives
the capital $P(S) / P(U)$, where $U$ is the universe, and a single asset $i$ receives
$\sigma_i^{-2} / P(U)$. With $\Sigma$ diagonal the risk share is
$\phi_i = w_i^2 \sigma_i^2 / \sum_j w_j^2 \sigma_j^2$, and $w_i^2 \sigma_i^2$ is proportional to
$\sigma_i^{-2}$, hence to $w_i$, so $\phi_i = w_i$. $\square$

ERC with uncorrelated assets gives inverse-volatility weights instead, with equal risk shares
(the [diagonal case of risk budgeting](risk_budgeting.md#independent-diagonal-covariance-check)).
HRP is therefore not risk parity: in the diagonal case the lower-variance assets carry more of the
risk as well as more of the capital. Multiplying $\Sigma$ by a positive constant multiplies every
$V(S)$ by it and leaves every share, hence the weights, unchanged; the function applies no
variance floor.

The function reproduces the allocation stage of the 2016 method: the order from the tree, the
midpoint bisection, and the inverse-variance valuation and split. It does not build the tree:
the correlation distance, the linkage method and the covariance estimate are the caller's, and the
paper's distance is one of the choices below. It relies on none of the paper's out-of-sample
comparisons. Variants that split along the tree's own clusters, or value a half by another risk
measure, are not implemented.

### Linkage semantics

The function uses the linkage only through its leaf order $\pi$: the merge heights, and which
assets the tree joins, matter only through the order they produce. The leaves are the row
positions of `covar`, because a SciPy linkage carries no labels, so the linkage must be built
from the same assets in the order of `covar.index`. A mismatch is not detected and changes the
weights. The bisection cuts the ordered list at its midpoint, not at the tree's own clusters:
both trees of the worked example join B, C and D before A, yet their first split puts A with one
of the three.

Which distance the tree is built on matters. With the correlation distance

$$
d_{ij} = \sqrt{\tfrac{1}{2}(1 - \rho_{ij})},
$$

the 2016 procedure computes a second distance between the columns, or equally the rows, of the
symmetric matrix of $d_{ij}$,

$$
\tilde d_{ij} = \sqrt{\sum_{k=1}^{N} (d_{ki} - d_{kj})^2},
$$

and builds the single-linkage tree on $\tilde d$, which compares whole distance profiles. The
direct convention builds the tree on $d_{ij}$ itself. factorlasso's
`compute_clusters_from_corr_matrix` with `DistanceTransform.CHORD` uses
$\sqrt{2(1 - \rho_{ij})} = 2 d_{ij}$. Single linkage depends only on the ranking of the
distances, so $d_{ij}$, twice $d_{ij}$ and $1 - \rho_{ij}$ build the same tree, while $\tilde d$
is not a function of $d_{ij}$ alone and can build a different one. The
[distance transform](https://factorlasso.readthedocs.io/en/latest/cluster_discovery.html#step-2-the-distance-transform)
section of factorlasso's cluster discovery compares the transforms.

SciPy's `linkage` takes a condensed vector of distances or a matrix of observations. Given the
square matrix of $d_{ij}$, it reads each row as an observation and clusters on the Euclidean
distance between rows, which is $\tilde d$. The 2016 code passes the square matrix, and SciPy
issues a `ClusterWarning`. Pass `squareform` of the matrix to cluster on $d_{ij}$. OptimalPortfolios
takes no position: pass the linkage of the distance you mean. The
[linkage comparison](../examples/comparisons/hrp_linkage_semantics.py) contrasts the two
conventions on the same allocation routine.

### Group risk budgets

`compute_group_risk_budgets(groups, group_size_exponent=0.0)` turns one label per asset into
asset-level risk budgets $b$. Each group receives an aggregate budget, which its classified
members share equally:

$$
B_g = \frac{n_g^{\eta}}{\sum_h n_h^{\eta}},
\qquad
b_i = \frac{B_g}{n_g} = \frac{n_g^{\eta - 1}}{\sum_h n_h^{\eta}}
\quad \text{for } i \in G_g,
$$

where $h$ runs over the groups with at least one classified asset. An asset whose label is
missing receives a zero budget. The budgets are non-negative and sum to one.

| `group_size_exponent` | Allocation of target risk |
|---|---|
| `0`, the default | Equal aggregate budget for each group present |
| `1` | Equal budget for each classified asset |
| `0.5` | Aggregate budget proportional to the square root of the group's size |
| Negative | Smaller groups receive more than an equal share |

Within a group every member receives the same budget, whatever its volatility or correlations.
The budgets enter the risk-budgeting solver as `risk_budget`, which excludes zero budgets, so an
unclassified asset is not held. When no bound binds, each asset's risk share equals its budget,
so each group's risk share equals $B_g$ and its members contribute equally; the capital follows
from the covariance.

Any finite exponent is accepted. An infinite exponent, duplicate asset labels or an observation
without a classified asset raise `ValueError`. A date-by-asset DataFrame of labels is transformed
row by row, so a later row never changes an earlier budget, and the resulting panel is a valid
date-by-asset `risk_budget` for `rolling_risk_budgeting`, which needs a row for each covariance
date. Labels can be sectors, asset classes or statistical clusters, such as those of factorlasso's
cluster discovery or its [rolling clusters](https://factorlasso.readthedocs.io/en/latest/rolling_cluster_smoothing.html).

### HRP, constraints and cluster budgets

HRP is a rule, not an optimisation. Each split divides the capital of a set by a fixed formula, so
there is no objective and no feasible set for a `Constraints` object to restrict, and the
function's only inputs are the covariance and the linkage. Its weights are long-only and fully
invested by construction (Proposition 1). A box or group bound would need a second rule for where
the excess capital goes, which would change the split shares; the method defines none and the
package adds none. The function also has no pre-trade weights, freezing or fallback, and it raises
on a missing or non-positive variance where the risk-budgeting wrapper drops the asset.

Cluster budgets take the other route: the clusters set risk budgets, and `wrapper_risk_budgeting`
or `rolling_risk_budgeting` solves for the weights under a `Constraints` object with asset boxes
and loading-weighted group capital bounds, as described in [risk budgeting](risk_budgeting.md).

| | HRP | Cluster budgets with risk budgeting |
|---|---|---|
| Inputs | Covariance and a linkage | Covariance, labels and an optional `Constraints` |
| Target | None; a rule for splitting capital | Group risk shares $B_g$, equal within a group |
| Covariance read | All but the entries across the first split | The whole matrix |
| Bounds | None | Asset boxes and group capital bounds |
| Output | Positive, fully invested weights | Long-only, fully invested weights; zero where a budget is zero |

`qis.compute_group_portfolio_risk_contribution_ratios` aggregates the Euler risk shares of any
weights by group; the examples use it to compare the allocations.

## Worked example

The six Python blocks below run in order and need no download, data file or random seed. They are
excerpts of the canonical script
[`examples/docs/hierarchical_risk_parity_and_cluster_budgets.py`](../examples/docs/hierarchical_risk_parity_and_cluster_budgets.py),
which runs them and asserts every number and property on this page against a reference computed a
different way: a recursive bisection written with plain Python lists and its own walk of the
linkage, a naive single-linkage clustering of each distance, the inverse-variance closed form, the
group-budget formula and Euler risk shares recomputed from the covariance:

```console
python -m examples.docs.hierarchical_risk_parity_and_cluster_budgets
```

### Two linkages of one correlation matrix

The four-asset correlation matrix of the
[linkage comparison](../examples/comparisons/hrp_linkage_semantics.py) has one strongly negative
pair, A and C at -0.57. Build both trees and allocate with the same function:

```python
import warnings

import numpy as np
import pandas as pd
import qis
from factorlasso import DistanceTransform, compute_clusters_from_corr_matrix
from scipy.cluster import hierarchy
from scipy.spatial import distance as spatial_distance
import optimalportfolios as opt

labels = pd.Index(["A", "B", "C", "D"], name="asset")
corr = pd.DataFrame(
    [[1.000000, 0.186665, -0.566601, 0.116919],
     [0.186665, 1.000000, 0.221383, 0.191615],
     [-0.566601, 0.221383, 1.000000, 0.185620],
     [0.116919, 0.191615, 0.185620, 1.000000]],
    index=labels, columns=labels,
)
vols = np.array([0.10, 0.14, 0.18, 0.23])
covar = corr * np.outer(vols, vols)
distance = np.sqrt(np.clip((1.0 - corr.to_numpy()) / 2.0, 0.0, None))
np.fill_diagonal(distance, 0.0)
# The 2016 procedure: single linkage on distances between rows of the matrix.
profile_linkage = hierarchy.linkage(spatial_distance.pdist(distance), method="single")
# The direct convention: single linkage on d_ij itself, here through factorlasso.
_, direct_linkage, _ = compute_clusters_from_corr_matrix(
    corr_matrix=corr, linkage_method="single",
    distance_transform=DistanceTransform.CHORD,
)
weights = pd.DataFrame({
    "profile": opt.compute_hierarchical_risk_parity_weights(covar, profile_linkage),
    "direct": opt.compute_hierarchical_risk_parity_weights(covar, direct_linkage),
})
print(weights.round(3))
```

The result, rounded to three decimals, is:

| Asset | Profile tree | Direct tree |
|---|---:|---:|
| A | 0.618 | 0.521 |
| B | 0.140 | 0.237 |
| C | 0.191 | 0.144 |
| D | 0.052 | 0.098 |

Both weight vectors are positive, sum to one and equal the independent recursive bisection. The
profile tree joins B and D first and then C; the direct tree joins B and C first and then D. Their
leaf orders are A, C, B, D and A, D, B, C, so the first split is A and C against B and D under the
profile tree, and A and D against B and C under the direct tree. The profile split keeps the
hedged pair A and C together, whose inverse-variance portfolio has a low variance, and gives that
half 80.8% of the capital; the direct split gives A and D 61.9%. The absolute weight differences
of the two allocations add up to 0.288.

The same two trees come from SciPy directly, depending on what it is given:

```python
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    square_input = hierarchy.linkage(distance, method="single")
direct_scipy = hierarchy.linkage(
    spatial_distance.squareform(distance, checks=False), method="single",
)
print(np.array_equal(square_input, profile_linkage))  # True
print(caught[0].category.__name__)  # ClusterWarning
print(np.allclose(direct_linkage[:, 2], 2.0 * direct_scipy[:, 2]))  # True
```

The square matrix reproduces the profile tree exactly, with a warning. The condensed vector
reproduces the direct tree: factorlasso's chord distance doubles its merge heights and changes no
merge.

> **Pitfall.** `scipy.cluster.hierarchy.linkage` does not read a square matrix of distances as
> distances. It treats the rows as observations and clusters on the distances between them, as the
> 2016 code does, and only warns. In the example the two readings split the assets differently and
> the absolute weight differences add up to 0.288. Pass `squareform(distance)` to cluster on the
> distances themselves.

### What the first split ignores

Under the direct tree the halves of the first split are A and D against B and C. Set the four
correlations between the halves to zero and allocate again with the same linkage:

```python
print(labels[hierarchy.leaves_list(direct_linkage)].tolist())  # ['A', 'D', 'B', 'C']
decoupled = corr.copy()
decoupled.loc[["A", "D"], ["B", "C"]] = 0.0
decoupled.loc[["B", "C"], ["A", "D"]] = 0.0
decoupled_weights = opt.compute_hierarchical_risk_parity_weights(
    decoupled * np.outer(vols, vols), direct_linkage,
)
print((decoupled_weights - weights["direct"]).abs().max())  # 0.0
```

No weight changes, as Proposition 2 states. A change to the correlation of A and D, which share a
half, does move the weights.

> **Insight.** Given its tree, HRP never reads the covariances between the two halves of its first
> split. In the example, setting the four correlations between A and D and B and C to zero, the
> hedge of A and C at -0.57 among them, leaves every weight unchanged: correlation reaches that
> split only through the tree.

### Partially classified groups

With one Growth asset and two Defensive assets, equal group budgets assign `0.50` to Growth and
`0.25` to each Defensive member. An unclassified asset receives zero:

```python
memberships = pd.Series({
    "Equity": "Growth",
    "Bonds": "Defensive",
    "Diversifier": "Defensive",
    "Unclassified": None,
})
group_budgets = opt.compute_group_risk_budgets(
    groups=memberships, group_size_exponent=0.0,
)
print(group_budgets.tolist())  # [0.5, 0.25, 0.25, 0.0]
```

The script checks each exponent of the table against the formula; with `0.5` the two Defensive
budgets together are $\sqrt{2}$ times the Growth budget. It also passes a two-date label panel
through the function and on to `rolling_risk_budgeting`: at each date the solved risk shares equal
that date's budgets.

### HRP, ERC and cluster budgets on a block universe

Eight assets fall into three blocks with constant correlations: four government bond markets
correlated 0.6, two equity markets correlated 0.8, and commodities and gold correlated 0.4.
Equities and real assets correlate 0.3 with each other; bonds correlate -0.1 with equities and 0.1
with real assets. Volatilities range from 4% to 20%. factorlasso builds the single-linkage tree on
the chord distance and cuts it into three clusters, which are the three blocks. Compare HRP on
that tree, ERC and risk budgeting with equal cluster budgets:

```python
assets = ["US govt", "EU govt", "UK govt", "JP govt",
          "US equity", "EU equity", "Commodities", "Gold"]
blocks = pd.Series(["Bonds"] * 4 + ["Equities"] * 2 + ["Real assets"] * 2, index=assets)
block_corr = pd.DataFrame(
    [[0.60, -0.10, 0.10],
     [-0.10, 0.80, 0.30],
     [0.10, 0.30, 0.40]],
    index=["Bonds", "Equities", "Real assets"],
    columns=["Bonds", "Equities", "Real assets"],
)
universe_corr = pd.DataFrame(
    np.where(np.eye(len(assets)) == 1.0, 1.0, block_corr.loc[blocks, blocks].to_numpy()),
    index=assets, columns=assets,
)
universe_vols = np.array([0.05, 0.05, 0.06, 0.04, 0.16, 0.18, 0.20, 0.15])
universe_covar = universe_corr * np.outer(universe_vols, universe_vols)

clusters, tree, _ = compute_clusters_from_corr_matrix(
    corr_matrix=universe_corr, linkage_method="single",
    distance_transform=DistanceTransform.CHORD, n_clusters=3,
)
cluster_budgets = opt.compute_group_risk_budgets(groups=clusters)
constraints = opt.Constraints(is_long_only=True)
allocation = pd.DataFrame({
    "HRP": opt.compute_hierarchical_risk_parity_weights(universe_covar, tree),
    "ERC": opt.wrapper_risk_budgeting(pd_covar=universe_covar, constraints=constraints),
    "cluster budgets": opt.wrapper_risk_budgeting(
        pd_covar=universe_covar, constraints=constraints, risk_budget=cluster_budgets,
    ),
})
risk = allocation.apply(lambda w: qis.compute_group_portfolio_risk_contribution_ratios(
    weights=w, covar=universe_covar, groups=blocks))
print(allocation.groupby(blocks).sum().round(3))
print(risk.round(3))
```

Capital and risk shares by block, rounded to three decimals:

| Block | HRP capital | HRP risk | ERC capital | ERC risk | Cluster budgets capital | Cluster budgets risk |
|---|---:|---:|---:|---:|---:|---:|
| Bonds | 0.907 | 0.903 | 0.756 | 0.500 | 0.686 | 0.333 |
| Equities | 0.041 | 0.021 | 0.129 | 0.250 | 0.160 | 0.333 |
| Real assets | 0.053 | 0.076 | 0.115 | 0.250 | 0.154 | 0.333 |

Every column of weights sums to one, and the HRP weights are positive and equal the independent
recursive bisection. The bonds form one half of the first split, so HRP gives them
$V(R) / (V(L) + V(R))$ of the capital, with $L$ the bond block and $R$ the other four assets, and
divides the rest between equities and real assets by the next split; the script recomputes both
shares from the inverse-variance variances. The low variance of the bond half earns it 90.7% of
the capital and 90.3% of the risk: as in the diagonal case of Proposition 3, the lower-variance
half carries most of the risk as well as most of the capital. ERC gives the four bonds half of the
risk, one eighth per asset, and the equal cluster budgets give each block a third.

![Left: the single-linkage tree of the eight assets on the chord distance, with the four bonds,
the two equity markets and the two real assets merging within their blocks first, and a dashed
line where HRP's first split separates the bonds from the rest. Right: capital and risk shares by
block for HRP, ERC and equal cluster budgets. HRP holds 91% of capital and 90% of risk in bonds;
ERC holds 76% of capital in bonds for 50% of risk; equal cluster budgets hold 69% of capital in
bonds for a third of the risk.](images/hrp_vs_erc_weights.png)

*Figure: the cluster tree of the block universe and the capital and risk each allocation assigns
to the three blocks. Drawn by the `exhibit` function of the canonical script; the
[analytics gallery](analytics_gallery.md) lists its provenance.*

The four bonds merge at one height, so SciPy's tie-breaking sets their order in the tree. Another
order moves the bond weights by up to 0.4 percentage points and leaves the block totals unchanged,
because the splits that separate the blocks do not depend on the order within the bonds.

Only the budget route accepts a mandate. Cap the bond block at 60% of the capital with a group
bound and solve again with the cluster budgets:

```python
bond_cap = opt.Constraints(
    is_long_only=True,
    group_lower_upper_constraints=opt.GroupLowerUpperConstraints(
        group_loadings=pd.DataFrame({"Bonds": (blocks == "Bonds").astype(float)}),
        group_min_allocation=None,
        group_max_allocation=pd.Series({"Bonds": 0.60}),
    ),
)
capped = opt.wrapper_risk_budgeting(
    pd_covar=universe_covar, constraints=bond_cap, risk_budget=cluster_budgets,
)
capped_risk = qis.compute_group_portfolio_risk_contribution_ratios(
    weights=capped, covar=universe_covar, groups=blocks)
print(capped.groupby(blocks).sum().round(3).tolist())
print(capped_risk.round(3).tolist())
```

The bound binds: the bonds hold 60.0% of the capital, equities 19.9% and real assets 20.1%, and
the risk shares leave their budgets of a third for 19.5%, 40.2% and 40.3%. HRP has no such input;
its bonds hold 90.7% of the capital whatever the mandate.

## Implementation in optimalportfolios

| Entry point | Input and output contract |
|---|---|
| `compute_hierarchical_risk_parity_weights(covar, linkage)` | A labelled covariance DataFrame and a SciPy linkage over its rows; returns a Series named `weight`, positive and summing to one |
| `compute_group_risk_budgets(groups, group_size_exponent=0.0)` | A label Series, or a date-by-asset label DataFrame; returns budgets with the same shape and labels, a Series named `risk_budget` or a DataFrame |

`compute_hierarchical_risk_parity_weights` raises `TypeError` unless `covar` is a DataFrame, and
`ValueError` unless its two axes carry the same unique labels, its entries are finite, it is
symmetric and its variances are positive. The linkage must have one row per merge and four
columns, finite values, non-negative heights, integer children that each appear once, and
consistent cluster sizes. The tree is not estimated here: build it with factorlasso or SciPy.
Neither function is a `PortfolioObjective` member, so the rolling dispatcher does not route to
HRP; for a rolling HRP allocation, call the function at each date with that date's covariance and
linkage. Group budgets reach the rolling path through `rolling_risk_budgeting`, and single-date
solves through `wrapper_risk_budgeting`, both described in [risk budgeting](risk_budgeting.md).

The [risk-allocation sources](https://github.com/ArturSepp/OptimalPortfolios/tree/main/src/optimalportfolios/optimization/risk_allocation)
and the [API reference](api.rst) describe the public entry points. The
[canonical script](../examples/docs/hierarchical_risk_parity_and_cluster_budgets.py) runs the
worked example and checks it against the independent references, together with the
propositions, the linkage semantics and the validation rules above. The test suite runs it, and so
does the offline examples lane of CI.

## Interpretation and limitations

- **Cross-cluster hedges.** HRP ignores the covariances across its first split (Proposition 2).
  A hedge between the two halves, such as A and C under the direct tree, does not affect the
  weights.
- **Midpoint splits.** The bisection follows the order of the tree but not its clusters, so a
  split can cut through a cluster, as the first split does in the four-asset example. This is the
  2016 rule; splitting at the tree's own clusters is a different method.
- **Concentration in low variance.** Inverse-variance splits give the lower-variance half most of
  the capital and, with weak correlation between the halves, most of the risk: 90.7% of the
  capital and 90.3% of the risk in the block example. HRP is not risk parity.
- **Tree sensitivity.** The weights depend on the tree through its leaf order, which depends on
  the distance, the linkage method and, under tied distances, on the tie-breaking of SciPy. The
  four-asset example moves by 0.288 in total between two distances.
- **No mandate.** HRP accepts no bounds, pre-trade weights or turnover control. When a mandate
  applies, use cluster budgets with the risk-budgeting solver, or check the HRP weights against
  the mandate yourself.
- **Budgets are targets.** Group budgets are shares of risk, not of capital. The solved group
  shares equal them only when no bound binds: with the bond cap the bonds carry 19.5% of the risk
  against a budget of a third.
- **Estimation.** The covariance, the tree and the labels are inputs. Their estimation error, and
  the instability of clusters from date to date, pass into the weights; factorlasso's
  [rolling clusters](https://factorlasso.readthedocs.io/en/latest/rolling_cluster_smoothing.html)
  page covers causal smoothing of the labels.
- **No performance claim.** The examples are synthetic calculations. The page makes no claim about
  out-of-sample performance.

## See also

- [Risk budgeting](risk_budgeting.md)
- [Portfolio constraints](constraints.md)
- [Maximum diversification](maximum_diversification.md)
- [Conventions, notation and glossary](conventions.md)
- [HRP linkage semantics comparison](../examples/comparisons/hrp_linkage_semantics.py)
- [factorlasso: cluster discovery](https://factorlasso.readthedocs.io/en/latest/cluster_discovery.html)
- [factorlasso: causal smoothing of rolling clusters](https://factorlasso.readthedocs.io/en/latest/rolling_cluster_smoothing.html)

## References

- López de Prado, M. (2016). *Building Diversified Portfolios that Outperform Out of Sample*.
  The Journal of Portfolio Management, 42(4), 59–69.
  [DOI 10.3905/jpm.2016.42.4.059](https://doi.org/10.3905/jpm.2016.42.4.059). The allocation
  stage that the function implements; the tree construction is the caller's.
- Maillard, S., Roncalli, T. and Teïletche, J. (2010). *The Properties of Equally Weighted Risk
  Contribution Portfolios*. The Journal of Portfolio Management, 36(4), 60–70.
  [DOI 10.3905/jpm.2010.36.4.060](https://doi.org/10.3905/jpm.2010.36.4.060). Equal risk
  contributions, the comparison allocation of the example.
- [OptimalPortfolios software citation](https://github.com/ArturSepp/OptimalPortfolios/blob/main/CITATION.cff).
- [qis software citation](https://github.com/ArturSepp/QuantInvestStrats/blob/main/CITATION.cff).
- [factorlasso software citation](https://github.com/ArturSepp/FactorLasso/blob/main/CITATION.cff).
