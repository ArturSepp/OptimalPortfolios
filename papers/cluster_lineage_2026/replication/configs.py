"""Frozen universe and smoothing configuration registry for the empirical study.

Production ``MOMENTUM_CLUSTER`` provenance was pinned from the read-only sources
``rosaa/alphas/alpha_aggregator.py``,
``rosaa/products/funds/pipeline/product_config.py``, and
``outputs/inputs_store/mac_20260630/config_snapshot.csv``. Replication code does not import
rosaa. The executed registry fixes ME long span 12, volatility span 13, no short leg,
``MeanAdjType.NONE``, minimum cluster size 5, and excludes QE-frequency funds from the arm.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Tuple

from factorlasso import ClusterSmootherType


class UniverseName(str, Enum):
    """Stable identifiers for the three empirical universes."""

    MSCI_US = "msci_us"
    FUTURES = "futures"
    MAC = "mac"


class SmootherName(str, Enum):
    """Stable identifiers for baseline, fixed-grid, and calibrated configurations."""

    BASELINE = "baseline"
    M0_QUARTERLY_HOLD = "M0_quarterly_hold"
    M1_DELTA_002 = "M1_delta_0.02"
    M1_DELTA_005 = "M1_delta_0.05"
    M1_DELTA_010 = "M1_delta_0.10"
    M2_LAMBDA_05 = "M2_lambda_0.5"
    M2_LAMBDA_07 = "M2_lambda_0.7"
    M1_STAR = "M1_star"
    M2_STAR = "M2_star"


@dataclass(frozen=True)
class UniverseSpec:
    """Point-in-time estimation and backtest conventions for one universe."""

    name: UniverseName
    factor_model: str
    asset_frequencies: Tuple[str, ...]
    span_freq_dict: Mapping[str, int]
    estimation_start: str
    estimation_end: str
    cost_bps: float
    ranking_taxonomy: str
    interpretability_taxonomies: Tuple[str, ...]
    momentum_lookback: Mapping[str, int]
    momentum_skip: Mapping[str, int]
    returns_are_excess: bool = False


@dataclass(frozen=True)
class SmootherSpec:
    """Declarative FactorLasso smoother configuration; ``None`` marks an owner slot."""

    name: SmootherName
    smoother_type: ClusterSmootherType
    parameter: Optional[float] = None
    recluster_freq: Optional[str] = None

    @property
    def is_calibrated(self) -> bool:
        """Return whether this slot requires an owner-supplied calibrated value."""
        return self.name in (SmootherName.M1_STAR, SmootherName.M2_STAR)

    def as_lasso_kwargs(self) -> Dict[str, Any]:
        """Return keyword arguments accepted by ``factorlasso.LassoModel``."""
        if self.is_calibrated and self.parameter is None:
            raise ValueError(f"{self.name.value} awaits the owner-calibrated value")
        output: Dict[str, Any] = {"cluster_smoother_type": self.smoother_type}
        if self.recluster_freq is not None:
            output["recluster_freq"] = self.recluster_freq
        if self.smoother_type == ClusterSmootherType.PARTITION_BONUS:
            output["smoother_delta"] = self.parameter
        elif self.smoother_type == ClusterSmootherType.SIMILARITY_EWMA:
            output["smoother_lambda"] = self.parameter
        return output


UNIVERSE_SPECS: Mapping[UniverseName, UniverseSpec] = {
    UniverseName.MSCI_US: UniverseSpec(
        name=UniverseName.MSCI_US,
        factor_model="FF6",
        asset_frequencies=("W-WED",),
        span_freq_dict={"W-WED": 156},
        estimation_start="2006-08-31",
        estimation_end="2026-07-31",
        cost_bps=10.0,
        ranking_taxonomy="gics_sector",
        interpretability_taxonomies=(
            "gics_sector",
            "gics_industry_group",
            "gics_industry",
        ),
        momentum_lookback={"W-WED": 48},
        momentum_skip={"W-WED": 4},
        returns_are_excess=True,
    ),
    UniverseName.FUTURES: UniverseSpec(
        name=UniverseName.FUTURES,
        factor_model="MATF11",
        asset_frequencies=("W-WED",),
        span_freq_dict={"W-WED": 156},
        estimation_start="2002-01-31",
        estimation_end="2026-07-31",
        cost_bps=20.0,
        ranking_taxonomy="asset_class",
        interpretability_taxonomies=("asset_class", "ac_geography"),
        momentum_lookback={"W-WED": 48},
        momentum_skip={"W-WED": 4},
    ),
    UniverseName.MAC: UniverseSpec(
        name=UniverseName.MAC,
        factor_model="MATF11",
        asset_frequencies=("ME", "QE"),
        span_freq_dict={"ME": 36, "QE": 12},
        estimation_start="2002-12-31",
        estimation_end="2026-07-31",
        cost_bps=50.0,
        ranking_taxonomy="Asset Class",
        interpretability_taxonomies=("Asset Class", "Sub Asset Class"),
        momentum_lookback={"ME": 12, "QE": 4},
        momentum_skip={"ME": 1, "QE": 1},
    ),
}


SMOOTHER_SPECS: Mapping[SmootherName, SmootherSpec] = {
    SmootherName.BASELINE: SmootherSpec(SmootherName.BASELINE, ClusterSmootherType.NONE),
    SmootherName.M0_QUARTERLY_HOLD: SmootherSpec(
        SmootherName.M0_QUARTERLY_HOLD,
        ClusterSmootherType.HOLD,
        recluster_freq="QE",
    ),
    SmootherName.M1_DELTA_002: SmootherSpec(
        SmootherName.M1_DELTA_002, ClusterSmootherType.PARTITION_BONUS, 0.02
    ),
    SmootherName.M1_DELTA_005: SmootherSpec(
        SmootherName.M1_DELTA_005, ClusterSmootherType.PARTITION_BONUS, 0.05
    ),
    SmootherName.M1_DELTA_010: SmootherSpec(
        SmootherName.M1_DELTA_010, ClusterSmootherType.PARTITION_BONUS, 0.10
    ),
    SmootherName.M2_LAMBDA_05: SmootherSpec(
        SmootherName.M2_LAMBDA_05, ClusterSmootherType.SIMILARITY_EWMA, 0.5
    ),
    SmootherName.M2_LAMBDA_07: SmootherSpec(
        SmootherName.M2_LAMBDA_07, ClusterSmootherType.SIMILARITY_EWMA, 0.7
    ),
    SmootherName.M1_STAR: SmootherSpec(
        SmootherName.M1_STAR, ClusterSmootherType.PARTITION_BONUS
    ),
    SmootherName.M2_STAR: SmootherSpec(
        SmootherName.M2_STAR, ClusterSmootherType.SIMILARITY_EWMA
    ),
}


M1_STAR_DELTAS: Mapping[UniverseName, Mapping[str, float]] = {
    UniverseName.FUTURES: {"W-WED": 0.0691},
    UniverseName.MAC: {"ME": 0.0830, "QE": 0.1609},
    UniverseName.MSCI_US: {"W-WED": 0.0866},
}


PRODUCTION_MOMENTUM_CLUSTER: Mapping[str, Any] = {
    "returns_freq": "ME",
    "long_span": 12,
    "short_span": None,
    "vol_span": 13,
    "mean_adj_type": "NONE",
    "min_cluster_size": 5,
    "benchmark": "EW_all",
    "qe_policy": "QE-frequency funds are EXCLUDED from the cluster-momentum arm.",
}


def get_universe_spec(name: UniverseName | str) -> UniverseSpec:
    """Return one frozen universe specification."""
    return UNIVERSE_SPECS[UniverseName(name)]


def get_smoother_spec(name: SmootherName | str) -> SmootherSpec:
    """Return one frozen smoother specification."""
    return SMOOTHER_SPECS[SmootherName(name)]


def get_m1_star_delta(universe: UniverseName | str, frequency: str) -> float:
    """Return the owner-confirmed M1-star delta for one universe-frequency panel."""
    universe = UniverseName(universe)
    try:
        return M1_STAR_DELTAS[universe][frequency]
    except KeyError as error:
        raise ValueError(f"no M1_star delta for {universe.value}/{frequency}") from error
