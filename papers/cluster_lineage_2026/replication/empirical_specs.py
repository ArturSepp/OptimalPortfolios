"""Frozen operating specifications selected by the empirical research programme.

The U1 operating point was selected from the 2026-08-15 covariance and signal grids.
It is recorded here before transfer to the BlackRock ETF universe so the U2 grid can
distinguish the ex-ante U1 specification from its exploratory U2 winner.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd


@dataclass(frozen=True)
class MomentumClusterSpec:
    """Define one complete cluster-relative momentum operating point."""

    covariance_frequency: str
    covariance_span: int
    covariance_cutoff_fraction: float
    covariance_linkage_method: str
    covariance_distance_transform: str
    covariance_dependence_measure: str
    covariance_demean: bool
    quantile: float
    signal_name: str
    signal_frequency: str
    momentum_long_span: int
    momentum_vol_span: int
    momentum_short_span: int | None
    momentum_mean_adj_type: str
    momentum_min_cluster_size: int
    membership_warmup_frequency: str
    membership_warmup_observations: int
    performance_frequency: str
    rebalance_frequency: str
    implementation_lag: int
    cost_bps: float
    cluster_construction: str
    global_construction: str
    long_only_gross_exposure: float
    long_short_long_exposure: float
    long_short_short_exposure: float

    def to_frame(self, *, name: str) -> pd.DataFrame:
        """Return the specification as an auditable two-column table."""
        values = asdict(self)
        return pd.DataFrame(
            {
                "parameter": list(values),
                "value": [values[key] for key in values],
                "specification": name,
            }
        )


@dataclass(frozen=True)
class FundAumEligibilitySpec:
    """Define a point-in-time fund-AUM eligibility rule."""

    data_field: str
    currency: str
    rolling_months: int
    threshold_usd_millions: float
    threshold_operator: str
    observation_timing: str
    missing_or_incomplete_history: str

    def to_frame(self, *, name: str) -> pd.DataFrame:
        """Return the eligibility specification as an auditable table."""
        values = asdict(self)
        return pd.DataFrame(
            {
                "parameter": list(values),
                "value": [values[key] for key in values],
                "specification": name,
            }
        )


U1_OPTIMAL_SPEC = MomentumClusterSpec(
    covariance_frequency="ME",
    covariance_span=36,
    covariance_cutoff_fraction=0.6,
    covariance_linkage_method="ward",
    covariance_distance_transform="one_minus_rho",
    covariance_dependence_measure="pearson",
    covariance_demean=True,
    quantile=0.25,
    signal_name="ROSAA_production_exact_monthly_12m",
    signal_frequency="ME",
    momentum_long_span=12,
    momentum_vol_span=13,
    momentum_short_span=None,
    momentum_mean_adj_type="NONE",
    momentum_min_cluster_size=5,
    membership_warmup_frequency="W-WED",
    membership_warmup_observations=12,
    performance_frequency="W-WED",
    rebalance_frequency="ME",
    implementation_lag=1,
    cost_bps=10.0,
    cluster_construction="group_equal",
    global_construction="asset_equal",
    long_only_gross_exposure=1.0,
    long_short_long_exposure=1.0,
    long_short_short_exposure=-1.0,
)


U2_BLACKROCK_PRIMARY_AUM_SPEC = FundAumEligibilitySpec(
    data_field="FUND_TOTAL_ASSETS",
    currency="USD",
    rolling_months=12,
    threshold_usd_millions=100.0,
    threshold_operator="strictly_greater_than",
    observation_timing="latest_completed_month_end_before_decision",
    missing_or_incomplete_history="ineligible",
)


U2_SIGNAL_PRIMARY_AUM_SPEC = FundAumEligibilitySpec(
    data_field="FUND_TOTAL_ASSETS",
    currency="USD",
    rolling_months=12,
    threshold_usd_millions=50.0,
    threshold_operator="strictly_greater_than",
    observation_timing="latest_completed_month_end_before_decision",
    missing_or_incomplete_history="ineligible",
)
