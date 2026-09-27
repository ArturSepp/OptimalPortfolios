"""Canonical script of docs/hierarchical_risk_parity_and_cluster_budgets.md.

The page's Python blocks are excerpts of ``main`` and run here in the same order; every number
and property the page states is asserted after them against a reference computed a different
way: a recursive bisection written with plain Python lists and its own walk of the linkage, a
naive single-linkage clustering of each distance, the inverse-variance closed form of the
diagonal case, the group-budget formula and Euler risk shares recomputed as
``w * (Sigma w) / (w' Sigma w)``. The four-asset fixture and its two linkages copy the logic of
``examples/comparisons/hrp_linkage_semantics.py``. The script runs offline after
``pip install optimalportfolios`` and needs no data file or random seed:

    python -m examples.docs.hierarchical_risk_parity_and_cluster_budgets

``exhibit`` draws the page's figure; ``tools/docs_analytics/teaching.py`` calls it with the
constants below and records their values.
"""
import numpy as np
import pandas as pd

FIXTURE_ASSETS = ['A', 'B', 'C', 'D']
# Correlations and annual volatilities of the four-asset fixture of the linkage comparison.
FIXTURE_CORR = [[1.000000, 0.186665, -0.566601, 0.116919],
                [0.186665, 1.000000, 0.221383, 0.191615],
                [-0.566601, 0.221383, 1.000000, 0.185620],
                [0.116919, 0.191615, 0.185620, 1.000000]]
FIXTURE_VOLS = [0.10, 0.14, 0.18, 0.23]
# The block universe of the figure: constant correlations within and between three blocks, and
# annual volatilities.
ASSETS = ['US govt', 'EU govt', 'UK govt', 'JP govt', 'US equity', 'EU equity', 'Commodities',
          'Gold']
BLOCKS = ['Bonds'] * 4 + ['Equities'] * 2 + ['Real assets'] * 2
BLOCK_NAMES = ['Bonds', 'Equities', 'Real assets']
BLOCK_CORR = [[0.60, -0.10, 0.10],
              [-0.10, 0.80, 0.30],
              [0.10, 0.30, 0.40]]
VOLS = [0.05, 0.05, 0.06, 0.04, 0.16, 0.18, 0.20, 0.15]
N_CLUSTERS = 3
BOND_CAP = 0.60  # the maximum capital in the Bonds block of the constrained solve


def risk_shares(weights, covar) -> np.ndarray:
    """Euler risk shares w_i (Sigma w)_i / (w' Sigma w), computed without qis."""
    w = np.asarray(weights, dtype=float)
    sigma = np.asarray(covar, dtype=float)
    return w * (sigma @ w) / (w @ sigma @ w)


def leaves_under(linkage, node: int, n_assets: int) -> list:
    """Leaf positions under one node of a SciPy linkage, left child first."""
    if node < n_assets:
        return [node]
    left, right = (int(child) for child in linkage[node - n_assets][:2])
    return leaves_under(linkage, left, n_assets) + leaves_under(linkage, right, n_assets)


def leaf_order(linkage, n_assets: int) -> list:
    """Leaf positions of a SciPy linkage, walking down from the root, without SciPy."""
    return leaves_under(linkage, 2 * n_assets - 2, n_assets)


def chain_linkage(order: list) -> np.ndarray:
    """A valid SciPy linkage that adds one leaf at a time, so its leaf order is ``order``."""
    n_assets = len(order)
    rows = [[order[0], order[1], 1.0, 2.0]]
    for step, leaf in enumerate(order[2:], start=1):
        rows.append([n_assets + step - 1, leaf, 1.0 + step, 2.0 + step])
    return np.array(rows, dtype=float)


def ivp_variance(covar: list, members: list) -> float:
    """Variance of the inverse-variance portfolio of ``members``, summed pair by pair."""
    inverse = [1.0 / covar[i][i] for i in members]
    total = sum(inverse)
    return sum(inverse[a] * inverse[b] * covar[i][j]
               for a, i in enumerate(members) for b, j in enumerate(members)) / total ** 2


def recursive_bisection(covar: list, order: list) -> dict:
    """HRP by recursion: halve the ordered list, give each half V(other) / (V(left) + V(right))."""
    if len(order) == 1:
        return {order[0]: 1.0}
    middle = len(order) // 2
    left, right = order[:middle], order[middle:]
    v_left, v_right = ivp_variance(covar, left), ivp_variance(covar, right)
    weights = {i: w * v_right / (v_left + v_right)
               for i, w in recursive_bisection(covar, left).items()}
    weights.update({i: w * v_left / (v_left + v_right)
                    for i, w in recursive_bisection(covar, right).items()})
    return weights


def hrp_reference(covar: pd.DataFrame, linkage) -> np.ndarray:
    """Independent HRP weights in the order of ``covar``: own leaf walk and own bisection."""
    matrix = covar.to_numpy().tolist()
    weights = recursive_bisection(matrix, leaf_order(linkage, len(matrix)))
    return np.array([weights[i] for i in range(len(matrix))])


def linkage_clades(linkage, n_assets: int) -> set:
    """Non-root clusters of a SciPy linkage as sets of leaf positions."""
    descendants = {leaf: frozenset({leaf}) for leaf in range(n_assets)}
    clades = set()
    for row, (left, right, _, _) in enumerate(linkage):
        merged = descendants[int(left)] | descendants[int(right)]
        descendants[n_assets + row] = merged
        if len(merged) < n_assets:
            clades.add(merged)
    return clades


def naive_single_linkage(distance) -> set:
    """Non-root clusters of single linkage: merge the two closest clusters until two remain."""
    clusters = [frozenset({i}) for i in range(len(distance))]
    clades = set()
    while len(clusters) > 2:
        gaps = {(a, b): min(distance[i][j] for i in clusters[a] for j in clusters[b])
                for a in range(len(clusters)) for b in range(a + 1, len(clusters))}
        a, b = min(gaps, key=gaps.get)
        merged = clusters[a] | clusters[b]
        clusters = [c for k, c in enumerate(clusters) if k not in (a, b)] + [merged]
        clades.add(merged)
    return clades


def row_distance(distance) -> list:
    """Euclidean distance between the rows of a square matrix, with explicit sums."""
    n = len(distance)
    return [[sum((distance[i][k] - distance[j][k]) ** 2 for k in range(n)) ** 0.5
             for j in range(n)] for i in range(n)]


def group_reference(groups: dict, exponent: float) -> dict:
    """Group budget n_g**eta / sum n_h**eta, split equally among classified members."""
    sizes = {}
    for label in groups.values():
        if label is not None:
            sizes[label] = sizes.get(label, 0) + 1
    total = sum(size ** exponent for size in sizes.values())
    return {asset: 0.0 if label is None else sizes[label] ** exponent / total / sizes[label]
            for asset, label in groups.items()}


def block_universe() -> tuple:
    """Correlation and covariance of the block universe from the module constants."""
    index = [BLOCK_NAMES.index(block) for block in BLOCKS]
    corr = np.array([[1.0 if i == j else BLOCK_CORR[index[i]][index[j]]
                      for j in range(len(ASSETS))] for i in range(len(ASSETS))])
    vols = np.array(VOLS)
    return (pd.DataFrame(corr, index=ASSETS, columns=ASSETS),
            pd.DataFrame(np.outer(vols, vols) * corr, index=ASSETS, columns=ASSETS))


def allocations(corr: pd.DataFrame, covar: pd.DataFrame) -> tuple:
    """The block universe's tree, clusters and HRP, ERC and cluster-budget weights."""
    from factorlasso import DistanceTransform, compute_clusters_from_corr_matrix
    import optimalportfolios as opt

    clusters, tree, _ = compute_clusters_from_corr_matrix(
        corr_matrix=corr, linkage_method='single', distance_transform=DistanceTransform.CHORD,
        n_clusters=N_CLUSTERS)
    constraints = opt.Constraints(is_long_only=True)
    weights = pd.DataFrame({
        'HRP': opt.compute_hierarchical_risk_parity_weights(covar, tree),
        'ERC': opt.wrapper_risk_budgeting(pd_covar=covar, constraints=constraints),
        'Cluster budgets': opt.wrapper_risk_budgeting(
            pd_covar=covar, constraints=constraints,
            risk_budget=opt.compute_group_risk_budgets(groups=clusters, group_size_exponent=0.0)),
    })
    return tree, clusters, weights


def assert_raises(error: type, function, *arguments, **keywords) -> None:
    """Fail unless ``function(*arguments, **keywords)`` raises ``error``."""
    try:
        function(*arguments, **keywords)
    except error:
        return
    raise AssertionError(f'expected {error.__name__}')


def main() -> None:
    """Run the page's blocks in order and assert every number and property it states."""
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

    # The block's inputs are the module constants; the matrix is positive definite.
    assert labels.tolist() == FIXTURE_ASSETS and corr.to_numpy().tolist() == FIXTURE_CORR
    assert vols.tolist() == FIXTURE_VOLS and np.linalg.eigvalsh(corr.to_numpy()).min() > 0.0
    assert np.allclose(np.diag(covar), vols ** 2, rtol=0.0, atol=1e-15)
    # d_ij = sqrt((1 - rho_ij) / 2); both trees agree with a naive single-linkage clustering, of
    # the explicit row distance for the profile tree and of d_ij for the direct tree.
    names = labels.tolist()
    square = distance.tolist()
    assert np.array_equal(distance, distance.T)  # rows and columns give the same profile distance
    assert np.isclose(distance[0, 2], np.sqrt((1 + 0.566601) / 2), rtol=0.0, atol=1e-15)
    profile_clades = linkage_clades(profile_linkage, 4)
    direct_clades = linkage_clades(direct_linkage, 4)
    assert profile_clades == naive_single_linkage(row_distance(square))
    assert direct_clades == naive_single_linkage(square)
    # The page: the profile tree joins B and D, then C; the direct tree joins B and C, then D.
    assert profile_clades == {frozenset({1, 3}), frozenset({1, 2, 3})}
    assert direct_clades == {frozenset({1, 2}), frozenset({1, 2, 3})}
    # Leaf orders A, C, B, D and A, D, B, C; the first split is {A, C} | {B, D} and
    # {A, D} | {B, C}, which cuts through the tree's cluster {B, C, D} both times.
    profile_order = [names[i] for i in leaf_order(profile_linkage, 4)]
    direct_order = [names[i] for i in leaf_order(direct_linkage, 4)]
    assert profile_order == ['A', 'C', 'B', 'D'] and direct_order == ['A', 'D', 'B', 'C']
    assert profile_order == labels[hierarchy.leaves_list(profile_linkage)].tolist()
    assert direct_order == labels[hierarchy.leaves_list(direct_linkage)].tolist()
    # The weights equal the independent recursive bisection, are positive and sum to one.
    for column, linkage in (("profile", profile_linkage), ("direct", direct_linkage)):
        np.testing.assert_allclose(weights[column], hrp_reference(covar, linkage),
                                   rtol=0.0, atol=1e-15)
        assert (weights[column] > 0.0).all() and abs(weights[column].sum() - 1.0) <= 1e-15
    # The numbers of examples/comparisons/hrp_linkage_semantics.py, the page's table and the
    # L1 difference 0.288.
    np.testing.assert_allclose(weights["profile"], [0.617773135213, 0.139769931308,
                                                    0.190670720745, 0.051786212734], atol=1e-12)
    np.testing.assert_allclose(weights["direct"], [0.520598106856, 0.237386172078,
                                                   0.143603980640, 0.098411740426], atol=1e-12)
    assert weights["profile"].round(3).tolist() == [0.618, 0.140, 0.191, 0.052]
    assert weights["direct"].round(3).tolist() == [0.521, 0.237, 0.144, 0.098]
    assert round((weights["profile"] - weights["direct"]).abs().sum(), 3) == 0.288
    # The first split: the profile tree's half {A, C} holds the hedged pair (rho = -0.57).
    assert round(weights["profile"][["A", "C"]].sum(), 3) == 0.808
    assert round(weights["direct"][["A", "D"]].sum(), 3) == 0.619
    # Only the leaf order is read: a linkage of another topology with the same leaf order gives
    # the same weights, and so does the direct tree built by SciPy on d_ij (half the heights).
    chain = chain_linkage(leaf_order(direct_linkage, 4))
    assert leaf_order(chain, 4) == leaf_order(direct_linkage, 4) == [0, 3, 1, 2]
    assert linkage_clades(chain, 4) != direct_clades
    np.testing.assert_allclose(opt.compute_hierarchical_risk_parity_weights(covar, chain),
                               weights["direct"], rtol=0.0, atol=0.0)
    # Proposition 3: with a diagonal covariance, HRP is inverse variance whatever the tree, and
    # each risk share equals the weight.
    diagonal = pd.DataFrame(np.diag(vols ** 2), index=labels, columns=labels)
    inverse_variance = (1.0 / vols ** 2) / np.sum(1.0 / vols ** 2)
    for linkage in (profile_linkage, direct_linkage, chain):
        diagonal_weights = opt.compute_hierarchical_risk_parity_weights(diagonal, linkage)
        np.testing.assert_allclose(diagonal_weights, inverse_variance, rtol=0.0, atol=1e-15)
        np.testing.assert_allclose(risk_shares(diagonal_weights, diagonal), diagonal_weights,
                                   rtol=0.0, atol=1e-15)
    # ERC between two uncorrelated assets gives inverse volatility, not inverse variance.
    pair = diagonal.iloc[:2, :2]
    erc_pair = opt.wrapper_risk_budgeting(pd_covar=pair,
                                          constraints=opt.Constraints(is_long_only=True))
    np.testing.assert_allclose(erc_pair, (1 / vols[:2]) / np.sum(1 / vols[:2]), atol=1e-6)
    # Scale invariance: HRP applies no variance floor, so any positive multiple of the
    # covariance, even a tiny one, gives the same weights.
    for scale in (1e-9, 52.0):
        np.testing.assert_allclose(
            opt.compute_hierarchical_risk_parity_weights(covar * scale, direct_linkage),
            weights["direct"], rtol=0.0, atol=1e-14)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        square_input = hierarchy.linkage(distance, method="single")
    direct_scipy = hierarchy.linkage(
        spatial_distance.squareform(distance, checks=False), method="single",
    )
    print(np.array_equal(square_input, profile_linkage))  # True
    print(caught[0].category.__name__)  # ClusterWarning
    print(np.allclose(direct_linkage[:, 2], 2.0 * direct_scipy[:, 2]))  # True

    # Pitfall: a square matrix is read as observations, so the call reproduces the profile
    # tree; SciPy warns. Condensed d_ij gives the direct tree, whose heights factorlasso's
    # chord distance sqrt(2 (1 - rho)) = 2 d_ij doubles, with the same merges.
    assert np.array_equal(square_input, profile_linkage)
    assert issubclass(caught[0].category, hierarchy.ClusterWarning)
    np.testing.assert_allclose(direct_linkage[:, [0, 1, 3]], direct_scipy[:, [0, 1, 3]],
                               rtol=0.0, atol=0.0)
    np.testing.assert_allclose(direct_linkage[:, 2], 2.0 * direct_scipy[:, 2], rtol=1e-14)
    np.testing.assert_allclose(np.sqrt(2.0 * (1.0 - corr.to_numpy())), 2.0 * distance,
                               rtol=0.0, atol=1e-15)
    # A monotone transform of d_ij leaves single linkage unchanged: 1 - rho builds the same tree.
    one_minus_rho = hierarchy.linkage(
        spatial_distance.squareform(1.0 - corr.to_numpy(), checks=False), method="single")
    np.testing.assert_allclose(one_minus_rho[:, [0, 1, 3]], direct_scipy[:, [0, 1, 3]], atol=0.0)
    # The linkage carries no labels: its leaves are the row positions of covar, so a covariance
    # in another asset order with the same linkage silently changes the allocation.
    reordered = covar.loc[["B", "A", "C", "D"], ["B", "A", "C", "D"]]
    shuffled = opt.compute_hierarchical_risk_parity_weights(reordered, direct_linkage)
    assert (shuffled.reindex(labels) - weights["direct"]).abs().max() > 0.05

    print(labels[hierarchy.leaves_list(direct_linkage)].tolist())  # ['A', 'D', 'B', 'C']
    decoupled = corr.copy()
    decoupled.loc[["A", "D"], ["B", "C"]] = 0.0
    decoupled.loc[["B", "C"], ["A", "D"]] = 0.0
    decoupled_weights = opt.compute_hierarchical_risk_parity_weights(
        decoupled * np.outer(vols, vols), direct_linkage,
    )
    print((decoupled_weights - weights["direct"]).abs().max())  # 0.0

    # Insight: the covariance between the halves of the first split is never read; the
    # decoupled matrix is a valid correlation matrix and the weights are identical.
    assert np.linalg.eigvalsh(decoupled.to_numpy()).min() > 0.0
    assert (decoupled_weights - weights["direct"]).abs().max() == 0.0
    # Under the direct tree the hedge rho(A, C) = -0.57 is one of the unread entries; every
    # entry within a half is read: changing rho(A, D) changes the weights.
    assert decoupled.loc["A", "C"] == 0.0 and corr.loc["A", "C"] < -0.5
    within = corr.copy()
    within.loc["A", "D"] = within.loc["D", "A"] = 0.5
    assert np.abs(opt.compute_hierarchical_risk_parity_weights(
        within * np.outer(vols, vols), direct_linkage) - weights["direct"]).max() > 0.01
    # HRP validates rather than filters: a missing variance or a linkage of the wrong size
    # raises ValueError, where the risk-budgeting wrapper would drop the asset.
    missing = covar.copy()
    missing.loc["D", "D"] = np.nan
    assert_raises(ValueError, opt.compute_hierarchical_risk_parity_weights, missing,
                  direct_linkage)
    dropped = opt.wrapper_risk_budgeting(pd_covar=missing,
                                         constraints=opt.Constraints(is_long_only=True))
    assert dropped["D"] == 0.0 and abs(dropped.sum() - 1.0) <= 1e-8
    assert_raises(ValueError, opt.compute_hierarchical_risk_parity_weights, covar,
                  direct_linkage[:2])
    # The other validation rules of the Implementation section: a DataFrame with the same labels
    # on both axes, symmetric, with positive variances; a linkage that uses each child once.
    assert_raises(TypeError, opt.compute_hierarchical_risk_parity_weights, covar.to_numpy(),
                  direct_linkage)
    assert_raises(ValueError, opt.compute_hierarchical_risk_parity_weights,
                  covar.rename(columns={"D": "E"}), direct_linkage)
    skewed = covar.copy()
    skewed.loc["A", "B"] += 0.001
    assert_raises(ValueError, opt.compute_hierarchical_risk_parity_weights, skewed,
                  direct_linkage)
    flat = covar.copy()
    flat.loc["B", "B"] = 0.0
    assert_raises(ValueError, opt.compute_hierarchical_risk_parity_weights, flat, direct_linkage)
    reused, negative, fractional, infinite, miscounted = (direct_linkage.copy() for _ in range(5))
    reused[1, :2] = [1.0, 2.0]  # child 2 twice, child 3 never
    negative[0, 2] = -0.1
    fractional[0, 1] = 2.5
    infinite[2, 2] = np.inf
    miscounted[0, 3] = 3.0
    for broken in (reused, negative, fractional, infinite, miscounted):
        assert_raises(ValueError, opt.compute_hierarchical_risk_parity_weights, covar, broken)
    assert opt.compute_hierarchical_risk_parity_weights(covar, direct_linkage).name == "weight"
    # No rolling HRP: no PortfolioObjective member selects it.
    assert not any(name.startswith(("HIER", "HRP")) for name in opt.PortfolioObjective.__members__)

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

    # The group formula: equal group budgets 0.50 and 2 x 0.25; the unclassified asset gets 0.
    labelled = {asset: label if isinstance(label, str) else None
                for asset, label in memberships.items()}
    expected = group_reference(labelled, 0.0)
    np.testing.assert_allclose(group_budgets, list(expected.values()), atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(group_budgets, [0.5, 0.25, 0.25, 0.0], atol=1e-12, rtol=0.0)
    assert abs(group_budgets.sum() - 1.0) <= 1e-12 and group_budgets["Unclassified"] == 0.0
    # The exponent table: 1 gives equal asset budgets; 0.5 makes each group's aggregate budget
    # proportional to the square root of its size; a negative exponent favours small groups.
    for exponent in (-1.0, 0.5, 1.0, 2.0):
        result = opt.compute_group_risk_budgets(groups=memberships,
                                                group_size_exponent=exponent)
        np.testing.assert_allclose(result, list(group_reference(labelled, exponent).values()),
                                   atol=1e-12, rtol=0.0)
        assert abs(result.sum() - 1.0) <= 1e-12
    np.testing.assert_allclose(opt.compute_group_risk_budgets(
        groups=memberships, group_size_exponent=1.0).iloc[:3], 1 / 3, atol=1e-12)
    root = opt.compute_group_risk_budgets(groups=memberships, group_size_exponent=0.5)
    assert np.isclose((root["Bonds"] + root["Diversifier"]) / root["Equity"], np.sqrt(2))
    negative = opt.compute_group_risk_budgets(groups=memberships, group_size_exponent=-1.0)
    assert negative["Equity"] > 0.5
    # Within a group every member gets the same budget, whatever its volatility.
    assert group_budgets["Bonds"] == group_budgets["Diversifier"]
    # No classified asset or a non-finite exponent raises; a membership panel is transformed row
    # by row, so a later row cannot change an earlier budget.
    assert_raises(ValueError, opt.compute_group_risk_budgets,
                  groups=pd.Series({"A": None, "B": None}))
    assert_raises(ValueError, opt.compute_group_risk_budgets, groups=memberships,
                  group_size_exponent=np.inf)
    assert_raises(ValueError, opt.compute_group_risk_budgets,
                  groups=pd.Series(["Growth", "Defensive"], index=["Equity", "Equity"]))
    assert group_budgets.name == "risk_budget"
    dates = pd.to_datetime(["2024-03-29", "2024-06-28"])
    panel = pd.DataFrame([memberships.tolist(), ["Growth", "Growth", "Defensive", None]],
                         index=dates, columns=memberships.index)
    by_row = opt.compute_group_risk_budgets(groups=panel)
    np.testing.assert_allclose(by_row.iloc[0], group_budgets, atol=1e-12)
    np.testing.assert_allclose(by_row.iloc[1], [0.25, 0.25, 0.5, 0.0], atol=1e-12)
    np.testing.assert_allclose(opt.compute_group_risk_budgets(groups=panel.iloc[:1]).iloc[0],
                               by_row.iloc[0], atol=0.0)
    # The budget panel feeds rolling_risk_budgeting: at each date the group risk shares of the
    # solve equal that date's group budgets.
    three = ["Equity", "Bonds", "Diversifier"]
    three_covar = pd.DataFrame([[0.040, 0.004, 0.002], [0.004, 0.010, 0.001],
                                [0.002, 0.001, 0.022]], index=three, columns=three)
    prices = pd.DataFrame(100.0, index=pd.bdate_range(dates[0], dates[-1]), columns=three)
    rolling = opt.rolling_risk_budgeting(prices=prices,
                                         constraints=opt.Constraints(is_long_only=True),
                                         risk_budget=by_row[three],
                                         covar_dict={date: three_covar for date in dates})
    for date in dates:
        shares = risk_shares(rolling.loc[date, three], three_covar)
        np.testing.assert_allclose(shares, by_row.loc[date, three], atol=1e-6)
    # An unclassified asset's zero budget leaves it out of the risk-budgeting solve.
    partial = opt.compute_group_risk_budgets(
        groups=pd.Series({"Equity": "Growth", "Bonds": "Defensive", "Diversifier": None}))
    excluded = opt.wrapper_risk_budgeting(pd_covar=three_covar,
                                          constraints=opt.Constraints(is_long_only=True),
                                          risk_budget=partial)
    assert partial["Diversifier"] == 0.0 and excluded["Diversifier"] == 0.0
    assert abs(excluded.sum() - 1.0) <= 1e-8

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

    # The block's inputs are the module constants; the matrix is positive definite.
    reference_corr, reference_covar = block_universe()
    assert assets == ASSETS and blocks.tolist() == BLOCKS
    assert block_corr.to_numpy().tolist() == BLOCK_CORR and universe_vols.tolist() == VOLS
    assert (universe_vols.min(), universe_vols.max()) == (0.04, 0.20)
    np.testing.assert_allclose(universe_covar, reference_covar, rtol=0.0, atol=0.0)
    assert np.linalg.eigvalsh(universe_corr.to_numpy()).min() > 0.0
    # factorlasso's three clusters are the three blocks, and each budget is 1/3 split equally.
    assert clusters.groupby(blocks).nunique().eq(1).all() and clusters.nunique() == N_CLUSTERS
    np.testing.assert_allclose(cluster_budgets, [1 / 12] * 4 + [1 / 6] * 4, atol=1e-15)
    np.testing.assert_allclose(cluster_budgets, list(group_reference(
        dict(zip(assets, clusters.astype(str))), 0.0).values()), atol=1e-15)
    # HRP equals the independent bisection; every column sums to one and is positive.
    np.testing.assert_allclose(allocation["HRP"], hrp_reference(universe_covar, tree),
                               rtol=0.0, atol=1e-15)
    assert np.allclose(allocation.sum(), 1.0, atol=1e-8) and (allocation > 0.0).all().all()
    # The Bonds block is one half of the first split, so HRP gives it V_other / (V_B + V_other)
    # of the capital, with the inverse-variance variances of the halves; the other half splits
    # between Equities and Real assets the same way.
    order = [assets[i] for i in leaf_order(tree, 8)]
    assert set(order[:4]) == set(blocks.index[blocks == "Bonds"])
    matrix = universe_covar.to_numpy().tolist()
    v_bonds, v_rest = ivp_variance(matrix, [0, 1, 2, 3]), ivp_variance(matrix, [4, 5, 6, 7])
    v_equities, v_real = ivp_variance(matrix, [4, 5]), ivp_variance(matrix, [6, 7])
    capital = allocation.groupby(blocks).sum()
    assert np.isclose(capital.loc["Bonds", "HRP"], v_rest / (v_bonds + v_rest), atol=1e-15)
    assert np.isclose(capital.loc["Equities", "HRP"],
                      v_bonds / (v_bonds + v_rest) * v_real / (v_equities + v_real), atol=1e-15)
    # With no bound binding, each asset's risk share equals its cluster budget.
    np.testing.assert_allclose(risk_shares(allocation["cluster budgets"], universe_covar),
                               cluster_budgets, atol=1e-6)
    # The page: HRP puts 90.7% of capital and 90.3% of risk in Bonds; ERC 75.6% and 50%; the
    # equal cluster budgets 68.6% and 33.3%.
    np.testing.assert_allclose(risk, allocation.apply(
        lambda w: pd.Series(risk_shares(w, universe_covar), index=assets).groupby(blocks).sum()),
        atol=1e-14)
    assert capital["HRP"].round(3).tolist() == [0.907, 0.041, 0.053]
    assert capital["ERC"].round(3).tolist() == [0.756, 0.129, 0.115]
    assert capital["cluster budgets"].round(3).tolist() == [0.686, 0.160, 0.154]
    assert risk["HRP"].round(3).tolist() == [0.903, 0.021, 0.076]
    np.testing.assert_allclose(risk["ERC"], [0.5, 0.25, 0.25], atol=1e-6)
    np.testing.assert_allclose(risk["cluster budgets"], [1 / 3] * 3, atol=1e-6)
    np.testing.assert_allclose(risk_shares(allocation["ERC"], universe_covar), 1 / 8, atol=1e-6)
    # The ERC and cluster-budget weights match the exhibit's own computation.
    _, _, exhibit_weights = allocations(reference_corr, reference_covar)
    np.testing.assert_allclose(exhibit_weights.to_numpy(), allocation.to_numpy(), rtol=0.0,
                               atol=0.0)
    # HRP never reads the correlations between Bonds and the rest: setting them to zero leaves
    # the weights unchanged.
    decoupled_universe = universe_corr.where(
        (blocks.to_numpy()[:, None] == "Bonds") == (blocks.to_numpy()[None, :] == "Bonds"), 0.0)
    np.testing.assert_allclose(opt.compute_hierarchical_risk_parity_weights(
        decoupled_universe * np.outer(universe_vols, universe_vols), tree),
        allocation["HRP"], rtol=0.0, atol=0.0)
    # The four bonds merge at one height, so SciPy's tie-breaking sets their order. Another order
    # moves weight between bonds but keeps the block totals, which the first two splits fix.
    bond_merges = [row[2] for node, row in enumerate(tree, start=8)
                   if set(leaves_under(tree, node, 8)) <= {0, 1, 2, 3}]
    assert len(bond_merges) == 3 and np.ptp(bond_merges) == 0.0 and order[:4] != assets[:4]
    assert np.isclose(bond_merges[0], np.sqrt(2.0 * (1.0 - 0.60)), rtol=1e-14)
    reordered_bonds = opt.compute_hierarchical_risk_parity_weights(
        universe_covar, chain_linkage([0, 2, 1, 3, 4, 5, 6, 7]))
    assert round((reordered_bonds - allocation["HRP"]).abs().max(), 3) == 0.004
    np.testing.assert_allclose(reordered_bonds.groupby(blocks).sum(), capital["HRP"], atol=1e-15)

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

    # The cap binds: Bonds hold 60% of capital, and the group risk shares leave their budgets.
    assert bond_cap.group_lower_upper_constraints.group_max_allocation["Bonds"] == BOND_CAP
    assert np.isclose(capped[blocks == "Bonds"].sum(), BOND_CAP, atol=1e-8)
    assert abs(capped.sum() - 1.0) <= 1e-8 and (capped >= 0.0).all()
    np.testing.assert_allclose(capped_risk, pd.Series(
        risk_shares(capped, universe_covar), index=assets).groupby(blocks).sum(), atol=1e-14)
    assert capped_risk["Bonds"] < 1 / 3 - 0.02
    assert capped.groupby(blocks).sum().round(3).tolist() == [0.600, 0.199, 0.201]
    assert capped_risk.round(3).tolist() == [0.195, 0.402, 0.403]
    print("hierarchical_risk_parity_and_cluster_budgets: all page statements verified.")


def exhibit(path) -> dict:
    """Draw the page's figure: the block universe's tree, and capital and risk by block.

    Args:
        path: PNG file to write.

    Returns:
        The plotted table and the checks the figure illustrates.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.transforms import blended_transform_factory
    from scipy.cluster import hierarchy

    corr, covar = block_universe()
    tree, clusters, weights = allocations(corr, covar)
    n_assets = len(ASSETS)
    blocks = pd.Series(BLOCKS, index=ASSETS)
    shares = weights.apply(lambda w: pd.Series(risk_shares(w, covar), index=ASSETS))
    table = weights.add_suffix(': weight').join(shares.add_suffix(': risk share'))
    table.insert(0, 'block', BLOCKS)
    capital = weights.groupby(blocks).sum().reindex(BLOCK_NAMES)
    risk = shares.groupby(blocks).sum().reindex(BLOCK_NAMES)

    ink, muted, grid, surface = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
    colours = dict(zip(BLOCK_NAMES, ('#2a78d6', '#eb6834', '#1baf7a')))
    context = '#b9b8b3'
    plt.rcParams.update({'font.size': 11, 'axes.edgecolor': grid, 'axes.labelcolor': ink,
                         'xtick.color': muted, 'ytick.color': muted,
                         'xtick.labelcolor': ink, 'ytick.labelcolor': ink})
    fig, (left, right) = plt.subplots(1, 2, figsize=(10.0, 4.4), facecolor=surface,
                                      gridspec_kw={'width_ratios': [1.0, 1.25]})

    def link_colour(node: int) -> str:
        """Colour a link by its block when all its leaves share one, grey otherwise."""
        members = {BLOCKS[i] for i in leaves_under(tree, node, n_assets)}
        return colours[members.pop()] if len(members) == 1 else context

    hierarchy.dendrogram(tree, labels=ASSETS, orientation='right', ax=left,
                         link_color_func=link_colour, leaf_font_size=10)
    # SciPy draws leaf k at height 10 k + 5; the first bisection falls between leaves 3 and 4.
    split = 10.0 * (n_assets // 2)
    left.axhline(split, color=muted, linestyle='--', linewidth=1.0)
    left.set_xlim(0.0, 1.75)
    left.set_xticks(np.arange(0.0, 1.51, 0.25))
    left.text(1.74, split + 1.0, 'first HRP split', ha='right', va='bottom', fontsize=9,
              color=ink)
    left.set_title('Single-linkage tree of the correlations', loc='left', color=ink)
    left.set_xlabel(r'Merge height $\sqrt{2(1-\rho)}$')
    left.tick_params(axis='y', length=0)

    width = 0.36
    positions = []
    for group, method in enumerate(weights.columns):
        for offset, (kind, frame) in zip((-0.2, 0.2), (('Capital', capital), ('Risk', risk))):
            x = group + offset
            positions.append((x, kind))
            bottom = 0.0
            for block in BLOCK_NAMES:
                value = frame.loc[block, method]
                right.bar(x, value, width, bottom=bottom, color=colours[block],
                          label=block if group == 0 and kind == 'Capital' else None)
                if value >= 0.07:
                    right.text(x, bottom + value / 2, f'{value:.0%}', ha='center',
                               va='center', fontsize=9, color=ink)
                bottom += value
    right.set_xticks([x for x, _ in positions], [kind for _, kind in positions], fontsize=9)
    below = blended_transform_factory(right.transData, right.transAxes)
    for group, name in enumerate(('HRP', 'ERC', 'Cluster budgets')):
        right.text(group, -0.13, name, transform=below, ha='center', va='top', color=ink)
    right.set_ylim(0.0, 1.16)
    right.set_yticks(np.linspace(0.0, 1.0, 6))
    right.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    right.set_title('Capital and risk by block', loc='left', color=ink)
    right.legend(frameon=False, loc='upper center', ncol=3, fontsize=9, labelcolor=ink)
    right.grid(axis='y', color=grid, linewidth=0.8)
    right.set_axisbelow(True)
    for axis in (left, right):
        axis.set_facecolor(surface)
        for side in ('top', 'right'):
            axis.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=surface)
    plt.close(fig)

    matrix = covar.to_numpy().tolist()
    order = leaf_order(tree, n_assets)
    bonds = [i for i, block in enumerate(BLOCKS) if block == 'Bonds']
    rest = [i for i in range(n_assets) if i not in bonds]
    v_bonds, v_rest = ivp_variance(matrix, bonds), ivp_variance(matrix, rest)
    equities = [i for i, block in enumerate(BLOCKS) if block == 'Equities']
    real = [i for i, block in enumerate(BLOCKS) if block == 'Real assets']
    v_equities, v_real = ivp_variance(matrix, equities), ivp_variance(matrix, real)
    prescribed = {'Bonds': v_rest / (v_bonds + v_rest),
                  'Equities': v_bonds / (v_bonds + v_rest) * v_real / (v_equities + v_real),
                  'Real assets': v_bonds / (v_bonds + v_rest) * v_equities / (v_equities + v_real)}
    checks = {
        'weights_sum_to_one': bool(np.allclose(weights.sum(), 1.0, atol=1e-8)),
        'hrp_weights_positive': bool((weights['HRP'] > 0.0).all()),
        'first_split_separates_bonds': bool(set(order[:n_assets // 2]) == set(bonds)
                                            or set(order[n_assets // 2:]) == set(bonds)),
        'hrp_blocks_follow_bisection': bool(all(
            np.isclose(capital.loc[block, 'HRP'], value, atol=1e-14)
            for block, value in prescribed.items())),
        'hrp_matches_independent_bisection': bool(np.allclose(
            weights['HRP'], hrp_reference(covar, tree), atol=1e-15)),
        'erc_equal_risk_shares': bool(np.allclose(shares['ERC'], 1 / n_assets, atol=1e-6)),
        'cluster_budgets_met': bool(np.allclose(risk['Cluster budgets'], 1 / 3, atol=1e-6)),
        'clusters_are_blocks': bool(clusters.groupby(blocks).nunique().eq(1).all()
                                    and clusters.nunique() == N_CLUSTERS),
    }
    return {'table': table, 'checks': checks}


if __name__ == '__main__':
    main()
