"""Focused checks for the U3 Rolling-Ward risk-allocation experiment."""

from __future__ import annotations

import pandas as pd

import papers.cluster_lineage_2026.replication.run_futures_best_relative_pnl_scatter as best
import papers.cluster_lineage_2026.replication.run_u3_hierarchical_risk as run


def test_u3_paper_method_set_excludes_herc() -> None:
    """The owner-excluded HERC variant must not be computed or reported for U3."""
    assert "ward_herc" not in run.METHODS
    assert tuple(run.PAPER_LONG_ONLY_METHODS) == (
        "flat_erc",
        "single_hrp",
        "ward_hrp",
    )


def test_u3_signal_spec_is_the_owner_frozen_futures_cell() -> None:
    """The U3 runner must bind to the selected futures signal without a new search."""
    assert run.COST_BPS == 10.0
    assert run.Q == 0.25
    assert run.SIGNAL_SPEC == best.SPEC
    assert run.SIGNAL_CLUSTER_METHOD == "sleeve_cluster_M1_star"
    assert run.WINDOW_START == pd.Timestamp("2009-08-31")
    assert run.WINDOW_END == pd.Timestamp("2026-06-30")


def test_u3_exclusions_are_inherited_from_the_frozen_investability_rule() -> None:
    """The paper runner must not carry a second divergent futures exclusion list."""
    assert run.OWNER_EXCLUSIONS is run.e5.FUTURES_INVESTABILITY_EXCLUSIONS
    assert len(run.OWNER_EXCLUSIONS) == 7
    assert {
        "BMR1 Curncy",
        "CUA1 Comdty",
        "IJ1 Comdty",
        "KC1 Comdty",
        "KM1 Index",
        "MES1 Index",
        "RS1 Comdty",
    } == set(run.OWNER_EXCLUSIONS)
