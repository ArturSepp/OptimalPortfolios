"""Regressions for the 10 bp global-rank 30/30/30/10 futures book."""
import numpy as np

import papers.cluster_lineage_2026.replication.run_futures_global_30303010_10bp as run


def test_design_uses_owner_weights_and_ten_basis_points() -> None:
    """Pin the requested strategic budgets and one-way transaction-cost rate."""
    assert run.TARGET == {
        "Equity": 0.30,
        "Fixed Income": 0.30,
        "Commodities": 0.30,
        "FX": 0.10,
    }
    assert run.COST_BPS == 10.0
    assert run.QUANTILES == (0.20, 0.25)
    assert np.isclose(sum(run.TARGET.values()), 1.0, atol=1e-15)


def test_persisted_run_is_exact_and_deterministic() -> None:
    """Require exact sleeve exposures, CUA exclusion, and replay stability."""
    performance = run.pd.read_csv(
        run._root() / "performance.csv", float_precision="round_trip"
    )
    acceptance = run.pd.read_csv(
        run._root() / "acceptance.csv", float_precision="round_trip"
    )
    replay = run.pd.read_csv(run._root() / "determinism.csv")
    reconstruction = run.pd.read_csv(
        run._root() / "standalone_weight_reconstruction.csv",
        float_precision="round_trip",
    )
    assert len(performance) == 2
    assert set(performance["q"]) == {0.20, 0.25}
    assert acceptance["status"].eq("PASS").all()
    assert acceptance["max_owner_excluded_weight_abs"].eq(0.0).all()
    assert reconstruction["status"].eq("PASS").all()
    assert reconstruction["max_weight_abs_error"].le(1e-12).all()
    assert replay["byte_identical"].all()

    primary = performance.set_index("q")
    assert np.isclose(
        primary.loc[0.20, "net_return_annualized"],
        0.0175476754899284,
        atol=1e-15,
    )
    assert np.isclose(
        primary.loc[0.20, "sharpe_rf0"],
        0.228467085267246,
        atol=1e-15,
    )
    assert np.isclose(
        primary.loc[0.25, "net_return_annualized"],
        0.022993637937647,
        atol=1e-15,
    )


def test_ten_basis_points_improves_the_matched_twenty_bp_path() -> None:
    """Keep cost sensitivity on identical decisions and require the expected ordering."""
    sensitivity = run.pd.read_csv(
        run._root() / "cost_sensitivity.csv", float_precision="round_trip"
    )
    assert sensitivity["net_return_improvement_10bp_vs_20bp"].gt(0.0).all()
    assert sensitivity["cost_drag_bp_per_year_10bp"].lt(
        sensitivity["cost_drag_bp_per_year_20bp"]
    ).all()
