"""Run U2 ROSAA long-only with 55/35/10 broad-sleeve budgets.

The point-in-time AUM50 universe is split into Equity, Fixed Income, and Rest.
Each sleeve receives 55%, 35%, and 10%, respectively. Within every sleeve the
top ROSAA-momentum quartile is selected and its funds are equally weighted.
Clusters change score standardisation only and receive no capital budgets.
"""
from collections.abc import Mapping
from pathlib import Path

import pandas as pd
from optimalportfolios.alphas import compute_top_quantile_equal_weights

import papers.cluster_lineage_2026.replication.run_u2_blackrock_aum_filter as aum
import papers.cluster_lineage_2026.replication.run_u2_blackrock_aum_sensitivity as sensitivity
import papers.cluster_lineage_2026.replication.run_u2_blackrock_etf_grid as funds
import papers.cluster_lineage_2026.replication.run_u2_blackrock_sleeve_grid as sleeves
import papers.cluster_lineage_2026.replication.run_u2_rosaa_long_only_aum_sensitivity as selected
import papers.cluster_lineage_2026.replication.run_u2_u3_min_cluster10_signal_comparison as comparison


TARGET = {"Equity": 0.55, "Fixed Income": 0.35, "Rest": 0.10}
FILTER_ID = "aum_50m"
RUNNER = (
    "papers/cluster_lineage_2026/replication/"
    "run_u2_rosaa_long_only_55_35_10_aum50.py"
)
base = selected.base


def _root() -> Path:
    """Return the gitignored 55/35/10 output directory."""
    root = (
        Path(__file__).resolve().parents[1]
        / "local_outputs"
        / "e5b"
        / "u2_rosaa_short3_min10_long_only_aum50_E55_F35_R10_20260816"
    )
    root.mkdir(parents=True, exist_ok=True)
    return root


def _weights(
    scores: pd.DataFrame,
    context: Mapping[str, object],
) -> tuple[pd.DataFrame, Mapping[str, float]]:
    """Select and equal-weight the top quartile inside each broad sleeve."""
    eligibility = context["eligibility"].reindex_like(scores).fillna(False).astype(bool)
    prices = context["rank_prices"].reindex_like(scores)
    sleeve_panel = context["sleeve_panel"].reindex_like(scores)
    output = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)
    diagnostics: dict[str, float] = {}
    all_counts = pd.DataFrame(index=scores.index)
    for sleeve in sleeves.SLEEVES:
        available = eligibility & sleeve_panel.eq(sleeve)
        unit = compute_top_quantile_equal_weights(
            alpha_scores=scores.where(available),
            prices=prices.where(available),
            quantile=context["q"],
        )
        if unit.sum(axis=1).le(0.0).any():
            raise AssertionError(f"{sleeve} has an empty top-quartile selection")
        selected_weights = unit.where(unit.gt(0.0))
        within_range = selected_weights.max(axis=1).subtract(
            selected_weights.min(axis=1)
        )
        sleeve_weights = unit.mul(TARGET[sleeve])
        output = output.add(sleeve_weights, fill_value=0.0)
        diagnostics[f"{sleeve}_budget_abs_error"] = float(
            sleeve_weights.sum(axis=1).sub(TARGET[sleeve]).abs().max()
        )
        diagnostics[f"{sleeve}_equal_weight_abs_error"] = float(
            within_range.fillna(0.0).max()
        )
        all_counts[sleeve] = unit.gt(0.0).sum(axis=1)
    diagnostics.update(
        {
            "weight_sum_abs_error": float(
                output.sum(axis=1).sub(1.0).abs().max()
            ),
            "weight_outside_eligibility_abs_error": float(
                output.where(~eligibility, 0.0).abs().to_numpy().max()
            ),
            "min_selected_funds": int(all_counts.sum(axis=1).min()),
            "max_selected_funds": int(all_counts.sum(axis=1).max()),
        }
    )
    return output, diagnostics


def _context() -> dict[str, object]:
    """Load the cached point-in-time AUM50 context and broad sleeves."""
    daily = funds._read_daily()
    dates = funds._dates()
    headline_dates = dates[
        (dates >= funds.HEADLINE_START) & (dates <= funds.HEADLINE_END)
    ]
    rolling_aum = aum._rolling_aum()
    eligibility_all = sensitivity._eligibilities(daily, dates, rolling_aum)
    monthly_returns = funds._native_returns(daily, "ME")
    monthly_eligibility = sensitivity._eligibilities(
        daily, monthly_returns.index, rolling_aum
    )
    partitions, _, cache_status = sensitivity._build_partitions(eligibility_all)
    if cache_status != "hit":
        raise AssertionError("55/35/10 run must consume the completed partition cache")
    context = selected._context(
        filter_id=FILTER_ID,
        daily=daily,
        headline_dates=headline_dates,
        monthly_returns=monthly_returns,
        eligibility_all=eligibility_all,
        monthly_eligibility=monthly_eligibility,
        partitions=partitions,
        performance_prices=funds._performance_prices(daily),
    )
    sleeve_map = sleeves._broad_sleeves(context["eligibility"].columns)
    context["sleeve_panel"] = sleeves._sleeve_panel(headline_dates, sleeve_map)
    return context


def main() -> None:
    """Run and deterministically replay the 55/35/10 comparison."""
    context = _context()
    comparison._u2_context = lambda: context
    base._root = _root
    base._equal_fund_weights = _weights
    base.SIGNAL_ID = "rosaa_risk_adjusted_momentum"
    base.SHORT_SPAN = 3
    base.BOOK = "broad_sleeve_equal_fund_long_only"
    base.ELIGIBILITY_LABEL = "all BlackRock funds passing point-in-time AUM50"
    base.RANK_SCOPE = "within Equity, Fixed Income, and Rest"
    base.ASSET_CLASS_BUDGETS = "Equity 0.55; Fixed Income 0.35; Rest 0.10"
    base.RUNNER = RUNNER
    base.main()


if __name__ == "__main__":
    main()
