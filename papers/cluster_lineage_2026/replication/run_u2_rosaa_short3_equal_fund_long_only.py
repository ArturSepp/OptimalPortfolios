"""Run the U2 ROSAA-momentum equal-fund experiment as long-only.

This is the risk-adjusted-momentum counterpart to the classic long-only run.
It retains the fixed ROSAA 12-month/3-month signal, 13-month volatility span,
EWMA mean adjustment, cluster minimum size 10, point-in-time AUM100 universe,
and equal weights across the selected top-quartile funds.
"""
from pathlib import Path

import papers.cluster_lineage_2026.replication.run_u2_classic_equal_fund_long_only as long_only


base = long_only.base
base.SIGNAL_ID = "rosaa_risk_adjusted_momentum"
base.SHORT_SPAN = 3
base.BOOK = "equal_fund_single_cross_section_long_only"
base.RUNNER = (
    "papers/cluster_lineage_2026/replication/"
    "run_u2_rosaa_short3_equal_fund_long_only.py"
)


def _root() -> Path:
    """Return the gitignored ROSAA long-only attribution directory."""
    root = (
        Path(__file__).resolve().parents[1]
        / "local_outputs"
        / "e5b"
        / "u2_rosaa_short3_min10_equal_fund_long_only_20260816"
    )
    root.mkdir(parents=True, exist_ok=True)
    return root


base._root = _root
base._equal_fund_weights = long_only._equal_fund_weights


if __name__ == "__main__":
    base.main()
