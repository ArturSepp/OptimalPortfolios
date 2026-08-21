"""Independently validate the persisted U1 covariance frequency/span grid."""
from __future__ import annotations

import pandas as pd

from papers.cluster_lineage_2026.replication.run_u1_covar_grid import (
    FREQUENCY_SPANS,
    _root,
)


EXPECTED_CELLS = sum(len(spans) for spans in FREQUENCY_SPANS.values())
EXPECTED_CLUSTER_ROWS = EXPECTED_CELLS * 5 * 2


def _read(name: str) -> pd.DataFrame:
    """Read one deterministic grid artifact."""
    return pd.read_csv(_root() / f"{name}.csv", float_precision="round_trip")


def validate() -> None:
    """Assert coverage, numerical acceptance, regressions, and replay hashes."""
    performance = _read("performance")
    comparison = _read("comparison_vs_global")
    acceptance = _read("acceptance")
    regression = _read("regression")
    determinism = _read("determinism")
    summary = _read("cell_summary")

    assert EXPECTED_CELLS == 28
    assert len(comparison) == EXPECTED_CLUSTER_ROWS
    assert len(performance) == EXPECTED_CLUSTER_ROWS + 10
    assert len(acceptance) == EXPECTED_CLUSTER_ROWS
    assert acceptance["status"].eq("PASS").all()
    assert regression["status"].eq("PASS").all()
    assert determinism["byte_identical"].astype(bool).all()
    assert len(summary) == EXPECTED_CELLS * 2
    assert not comparison.columns.str.contains("taxonomy", case=False).any()
    assert not comparison["leg"].str.contains("taxonomy", case=False).any()
    assert not comparison.columns.str.contains("delta_vs_ew", case=False).any()

    headline = comparison.loc[
        comparison["analysis_window"].eq("headline_20090831_20260630")
    ]
    raw = headline.sort_values(
        ["sharpe_rf0", "net_return_annualized", "one_way_turnover_annualized"],
        ascending=[False, False, True],
    ).iloc[0]
    same_q_winners = int(headline["beats_same_q_global_both"].astype(bool).sum())
    absolute_winners = int(headline["beats_best_global_both"].astype(bool).sum())
    max_weight_error = float(acceptance["weight_sum_error"].max())
    max_budget_error = float(acceptance["group_budget_error"].max())

    print("U1 covariance frequency/span grid independent validation: PASS")
    print(f"grid: {EXPECTED_CELLS} cells x 5 q x 2 windows = {len(comparison)} rows")
    print(
        f"construction: {len(acceptance)}/{len(acceptance)} PASS; "
        f"max weight error={max_weight_error:.3e}; "
        f"max group-budget error={max_budget_error:.3e}"
    )
    print(
        f"headline raw winner: {raw['frequency']} span {int(raw['span'])} "
        f"at q={float(raw['q']):.2f}; Sharpe={float(raw['sharpe_rf0']):.6f}"
    )
    print(f"headline rows beating same-q global on return and Sharpe: {same_q_winners}")
    print(f"headline rows beating best global on return and Sharpe: {absolute_winners}")
    print(
        f"determinism: {int(determinism['byte_identical'].astype(bool).sum())}/"
        f"{len(determinism)} numerical artifacts byte-identical"
    )


if __name__ == "__main__":
    validate()
