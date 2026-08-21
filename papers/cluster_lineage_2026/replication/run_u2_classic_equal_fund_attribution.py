"""Run all eligible U2 funds in one equal-weight classic-momentum cross-section.

This is the classic 12-month excluding one-month momentum counterpart to the
otherwise identical ROSAA experiment. It uses the public OptimalPortfolios
classic global and cluster signal functions, point-in-time AUM100 eligibility,
cluster minimum size 10, equal fund weights on both sides, and 20 bp costs.
Official asset classes enter only in the exact ex-post P&L attribution.
"""
from pathlib import Path

import papers.cluster_lineage_2026.replication.run_u2_rosaa_short3_equal_fund_attribution as base


base.SIGNAL_ID = "classic_12m_ex_1m"
base.SHORT_SPAN = 1
base.RUNNER = (
    "papers/cluster_lineage_2026/replication/"
    "run_u2_classic_equal_fund_attribution.py"
)


def _root() -> Path:
    """Return the gitignored classic equal-fund attribution directory."""
    root = (
        Path(__file__).resolve().parents[1]
        / "local_outputs"
        / "e5b"
        / "u2_classic_12m1m_min10_equal_fund_attribution_20260816"
    )
    root.mkdir(parents=True, exist_ok=True)
    return root


base._root = _root


if __name__ == "__main__":
    base.main()
