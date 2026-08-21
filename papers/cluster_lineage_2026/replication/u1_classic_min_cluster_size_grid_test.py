"""Regression tests for the U1 classic cluster-size fallback grid."""
import numpy as np
import pandas as pd

import papers.cluster_lineage_2026.replication.run_u1_classic_min_cluster_size_grid as run


EXPECTED_CLUSTER_NET_RETURNS = {
    5: -0.0365261198057413,
    10: -0.0328467057636845,
    15: -0.034465861151598,
    20: -0.033071552321988,
}


def test_grid_outputs_are_accepted_and_deterministic() -> None:
    """Require all validation rows and replay hashes to pass."""
    acceptance = pd.read_csv(run._root() / "acceptance.csv")
    replay = pd.read_csv(run._root() / "determinism.csv")

    assert acceptance["status"].eq("PASS").all()
    assert replay["byte_identical"].all()


def test_cluster_net_returns_are_frozen() -> None:
    """Freeze every cluster-leg grid result independently of table ordering."""
    performance = pd.read_csv(
        run._root() / "performance.csv", float_precision="round_trip"
    )
    cluster = performance.loc[performance["leg"].eq("cluster_M1_star")].set_index(
        "min_cluster_size"
    )
    assert set(cluster.index) == set(run.MIN_CLUSTER_SIZE_GRID)
    for threshold, expected in EXPECTED_CLUSTER_NET_RETURNS.items():
        assert np.isclose(
            cluster.loc[threshold, "net_return_annualized"],
            expected,
            atol=1e-12,
            rtol=0.0,
        )


def test_sector_and_global_yardsticks_are_threshold_invariant() -> None:
    """The fixed yardsticks must not drift as the M1 fallback changes."""
    performance = pd.read_csv(run._root() / "performance.csv")
    for leg in ("bics_sector", "global"):
        rows = performance.loc[performance["leg"].eq(leg)]
        assert rows["net_return_annualized"].nunique() == 1
        assert rows["volatility_annualized"].nunique() == 1
