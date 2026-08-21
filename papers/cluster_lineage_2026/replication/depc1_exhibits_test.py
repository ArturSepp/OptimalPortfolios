"""Focused tests for the traced de-PC1 exhibit builder."""

import pandas as pd

import papers.cluster_lineage_2026.replication.build_depc1_exhibits as exhibits


def test_pnl_comparison_is_one_row_per_asset_with_signed_delta():
    """Contribution data must compare the two cluster arms directly."""
    data = pd.DataFrame(
        {
            "asset": ["A", "A", "B", "B"],
            "leg": ["cluster_raw", "cluster_depc1"] * 2,
            "net_pnl_pct_of_start": [1.0, 1.5, -0.5, -0.75],
        }
    )
    output = exhibits._pnl_comparison(data).set_index("asset")
    assert output.at["A", "depc1_minus_raw_pnl_pct"] == 0.5
    assert output.at["B", "depc1_minus_raw_pnl_pct"] == -0.25


def test_taxonomy_series_pairs_raw_and_depc1_columns():
    """Topology exhibit data must pair both arms for every taxonomy."""
    data = pd.DataFrame(
        {
            "raw_taxonomy_ari_sector": [0.4],
            "depc1_taxonomy_ari_sector": [0.3],
            "unrelated": [1.0],
        }
    )
    output = exhibits._taxonomy_series(data)
    assert list(output) == ["raw sector", "de-PC1 sector"]


def test_required_exhibit_contract_has_five_unique_files():
    """D6 must retain each of the five roadmap exhibits."""
    assert len(exhibits.REQUIRED_EXHIBITS) == 5
    assert len(set(exhibits.REQUIRED_EXHIBITS)) == 5
