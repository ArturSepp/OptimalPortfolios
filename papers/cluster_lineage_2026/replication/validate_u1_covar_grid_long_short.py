"""Independently validate the persisted U1 q=0.25 covariance long-short grid."""
from __future__ import annotations

import pandas as pd

from papers.cluster_lineage_2026.replication.run_u1_covar_grid import FREQUENCY_SPANS
from papers.cluster_lineage_2026.replication.run_u1_covar_grid_long_short import (
    Q,
    _root,
)


EXPECTED_CELLS = sum(len(spans) for spans in FREQUENCY_SPANS.values())
EXPECTED_CLUSTER_ROWS = EXPECTED_CELLS * 2


def _read(name: str) -> pd.DataFrame:
    """Read one deterministic long-short grid artifact."""
    return pd.read_csv(_root() / f"{name}.csv", float_precision="round_trip")


def validate() -> None:
    """Assert coverage, exact exposures, regression, rankings, and determinism."""
    performance = _read("performance")
    comparison = _read("comparison_vs_global")
    rankings = _read("rankings")
    risk = _read("risk_diagnostics")
    acceptance = _read("acceptance")
    regression = _read("regression")
    determinism = _read("determinism")

    assert EXPECTED_CELLS == 28
    assert len(performance) == EXPECTED_CLUSTER_ROWS + 2
    assert len(comparison) == EXPECTED_CLUSTER_ROWS
    assert len(rankings) == EXPECTED_CLUSTER_ROWS
    assert len(risk) == EXPECTED_CLUSTER_ROWS
    assert len(acceptance) == EXPECTED_CLUSTER_ROWS + 2
    assert performance["q"].eq(Q).all()
    assert comparison["benchmark_leg"].eq("global").all()
    assert acceptance["status"].eq("PASS").all()
    assert regression["status"].eq("PASS").all()
    assert determinism["byte_identical"].astype(bool).all()
    assert risk["cluster_max_group_l1_net_exposure"].max() <= 1e-12
    assert not comparison.columns.str.contains("delta_vs_ew_all", case=False).any()

    headline = comparison.loc[
        comparison["analysis_window"].eq("headline_20090831_20260630")
    ]
    winner = headline.sort_values(
        ["delta_net_return_annualized", "cluster_volatility_annualized"],
        ascending=[False, True],
    ).iloc[0]
    return_wins = int(headline["beats_global_net_return"].astype(bool).sum())
    mv_wins = int(headline["mean_variance_dominates_global"].astype(bool).sum())
    return_sharpe_wins = int(
        headline["beats_global_return_and_sharpe"].astype(bool).sum()
    )
    max_exposure_error = float(
        acceptance[
            [
                "max_long_exposure_error",
                "max_short_exposure_error",
                "max_net_exposure_error",
                "max_gross_exposure_error",
            ]
        ].to_numpy().max()
    )

    print("U1 q=0.25 covariance long-short grid independent validation: PASS")
    print(f"grid: {EXPECTED_CELLS} cells x 2 windows = {len(comparison)} comparisons")
    print(
        f"exposures: {len(acceptance)}/{len(acceptance)} PASS; "
        f"max error={max_exposure_error:.3e}"
    )
    print(
        f"headline winner: {winner['frequency']} span {int(winner['span'])}; "
        f"net return delta={float(winner['delta_net_return_annualized']):+.6f}; "
        f"volatility delta={float(winner['delta_volatility_annualized']):+.6f}"
    )
    print(
        f"headline cells: return wins={return_wins}/{EXPECTED_CELLS}; "
        f"mean-variance wins={mv_wins}/{EXPECTED_CELLS}; "
        f"return-and-Sharpe wins={return_sharpe_wins}/{EXPECTED_CELLS}"
    )
    print(
        f"determinism: {int(determinism['byte_identical'].astype(bool).sum())}/"
        f"{len(determinism)} artifacts byte-identical"
    )


if __name__ == "__main__":
    validate()
