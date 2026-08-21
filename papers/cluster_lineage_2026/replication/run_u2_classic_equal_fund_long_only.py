"""Run the U2 classic-momentum equal-fund experiment as long-only.

The experiment matches the equal-fund long-short comparison except that only
the top classic 12-month-ex-one-month momentum quartile is held. The public
OptimalPortfolios quantile-weight function supplies equal weights across all
selected point-in-time AUM100-eligible funds. Clusters change signal
standardisation only; official asset classes are used solely for attribution.
"""
from collections.abc import Mapping
from pathlib import Path

import pandas as pd
from optimalportfolios.alphas import compute_top_quantile_equal_weights

import papers.cluster_lineage_2026.replication.run_u2_rosaa_short3_equal_fund_attribution as base


base.SIGNAL_ID = "classic_12m_ex_1m"
base.SHORT_SPAN = 1
base.BOOK = "equal_fund_single_cross_section_long_only"
base.RUNNER = (
    "papers/cluster_lineage_2026/replication/"
    "run_u2_classic_equal_fund_long_only.py"
)


def _root() -> Path:
    """Return the gitignored classic long-only attribution directory."""
    root = (
        Path(__file__).resolve().parents[1]
        / "local_outputs"
        / "e5b"
        / "u2_classic_12m1m_min10_equal_fund_long_only_20260816"
    )
    root.mkdir(parents=True, exist_ok=True)
    return root


def _equal_fund_weights(
    scores: pd.DataFrame,
    context: Mapping[str, object],
) -> tuple[pd.DataFrame, Mapping[str, float]]:
    """Select the top quartile and weight every selected fund equally."""
    eligibility = context["eligibility"].reindex_like(scores).fillna(False).astype(bool)
    prices = context["rank_prices"].reindex_like(scores)
    weights = compute_top_quantile_equal_weights(
        alpha_scores=scores.where(eligibility),
        prices=prices.where(eligibility),
        quantile=context["q"],
    )
    selected = weights.gt(0.0)
    selected_weights = weights.where(selected)
    weight_range = selected_weights.max(axis=1).subtract(selected_weights.min(axis=1))
    diagnostics = {
        "weight_sum_abs_error": float(weights.sum(axis=1).sub(1.0).abs().max()),
        "weight_outside_eligibility_abs_error": float(
            weights.where(~eligibility, 0.0).abs().to_numpy().max()
        ),
        "max_selected_fund_weight_range": float(weight_range.fillna(0.0).max()),
        "min_selected_funds": int(selected.sum(axis=1).min()),
        "max_selected_funds": int(selected.sum(axis=1).max()),
    }
    return weights, diagnostics


base._root = _root
base._equal_fund_weights = _equal_fund_weights


if __name__ == "__main__":
    base.main()
