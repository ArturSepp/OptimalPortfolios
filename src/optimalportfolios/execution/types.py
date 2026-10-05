"""Typed, copied inputs for one resolved execution decision.

Callers own lifecycle, cadence and instruction resolution. These contracts keep
the raw model, post-mandatory base and current portfolio distinct, and retain the
column vocabulary needed to compare engines without changing policy semantics.
"""
from copy import deepcopy
from dataclasses import dataclass, field
from typing import ClassVar, Optional

import numpy as np
import pandas as pd

from optimalportfolios.optimization.config import OptimiserConfig
from optimalportfolios.optimization.constraints import Constraints
from optimalportfolios.execution import schema as s


@dataclass(frozen=True)
class ExecutionRankingConfig:
    """Legacy funded alpha/TRE selection settings, independent of product labels.

    Args:
        tre_weight_by_asset_class: Nonnegative coefficient for every caller-defined class.
        max_trades: Requested non-cash ticket allowance; None means unrestricted.
        add_mandatory_trades_to_max_trades: Treat the allowance as discretionary when true.
        minimum_trade_score: Optional minimum discretionary selection score.
        sequential_greedy_selection: Recompute scores after each selected funded trade.

    The allowance controls requested selection, not a hard cap on feasibility rescue.
    Missing alphas retain the legacy zero substitution in the ranking function.
    """

    tre_weight_by_asset_class: dict[str, float]
    max_trades: Optional[int] = None
    add_mandatory_trades_to_max_trades: bool = False
    minimum_trade_score: Optional[float] = None
    sequential_greedy_selection: bool = False

    def __post_init__(self) -> None:
        """Validate settings and detach the caller's coefficient mapping."""
        weights = dict(self.tre_weight_by_asset_class)
        if not weights or any(not isinstance(key, str) or not key for key in weights):
            raise ValueError('TRE weights require non-empty asset-class names')
        if any(not np.isfinite(value) or value < 0.0 for value in weights.values()):
            raise ValueError('TRE weights must be finite and non-negative')
        if self.max_trades is not None and (
                isinstance(self.max_trades, (bool, np.bool_))
                or not isinstance(self.max_trades, (int, np.integer))
                or self.max_trades <= 0):
            raise ValueError('max_trades must be a positive integer or None')
        for name in ('add_mandatory_trades_to_max_trades', 'sequential_greedy_selection'):
            if not isinstance(getattr(self, name), (bool, np.bool_)):
                raise ValueError(f'{name} must be boolean')
        if self.minimum_trade_score is not None and not np.isfinite(self.minimum_trade_score):
            raise ValueError('minimum_trade_score must be finite or None')
        object.__setattr__(self, 'tre_weight_by_asset_class', weights)


@dataclass(frozen=True)
class ResolvedExecutionProblem:
    """One dated numerical decision after the consumer resolves product policy.

    Args:
        target: Instrument-indexed resolved table using ``execution.schema`` columns.
        covariance: Finite symmetric covariance, in the units used for TRE coefficients.
        alphas: Instrument alphas; missing values retain the legacy zero substitution.
        asset_classes: Caller-defined class per instrument, selecting ranking coefficients.
        constraints: Existing OP hard constraints; legacy execution filters inherited
            turnover/TRE controls when constructing selected execution constraints.
        ranking_config: Effective ranking settings, not a named product preset.
        optimiser_config: Existing OP numerical settings; select CLARABEL for portable runs.
        asset_class_tre_weights: Optional nonnegative product multipliers per configured class.
        partition_groups: Names of a disjoint exhaustive membership partition in the
            group constraints, enabling the legacy full-investment bridge tightening.
        expand_for_feasibility: Admit additional eligible trades to repair hard constraints.
        sign_directed_rescue: Apply the legacy signed residual filter when needed.
        allow_corridor_relaxation: Permit and audit the final selected-corridor retry.
        context: Decision label passed to solver diagnostics.

    Inputs are copied and aligned at construction. Pandas objects remain editable;
    ``solve_ranked_execution`` takes and validates a fresh copy before each solve.
    No covariance estimation, resampling or annualisation is performed here. PSD
    handling remains that of the existing OP covariance factorization and solver.
    """

    contract_version: ClassVar[str] = '1.0'
    ranking_objective: ClassVar[str] = 'legacy_funded_alpha_tre'
    projection_objective: ClassVar[str] = 'minimum_raw_model_tracking_variance'

    target: pd.DataFrame
    covariance: pd.DataFrame
    alphas: pd.Series
    asset_classes: pd.Series
    constraints: Constraints
    ranking_config: ExecutionRankingConfig
    optimiser_config: OptimiserConfig = field(default_factory=OptimiserConfig)
    asset_class_tre_weights: Optional[pd.Series] = None
    partition_groups: tuple[str, ...] = ()
    expand_for_feasibility: bool = False
    sign_directed_rescue: bool = False
    allow_corridor_relaxation: bool = False
    context: str = ''

    def __post_init__(self) -> None:
        """Validate the numerical seam and detach all mutable caller-owned inputs."""
        target = self.target.copy(deep=True)
        if target.empty or not target.index.is_unique or not target.columns.is_unique:
            raise ValueError('target must be non-empty with unique instrument and column labels')
        numeric = (
            s.CURRENT_WEIGHT, s.RAW_MODEL_WEIGHT, s.EFFECTIVE_MODEL_WEIGHT,
            s.BASE_WEIGHT, s.DESIRED_REWEIGHT, s.MANDATORY_FUNDING,
            s.POLICY_MIN_WEIGHT, s.POLICY_MAX_WEIGHT,
        )
        flags = (
            s.SETTLEMENT_CASH, s.MANDATORY_TRADE, s.RULE4_TRADE,
            s.REBALANCE_CADENCE_ELIGIBLE, s.TRADE_CANDIDATE, s.MATERIAL_TRADE,
        )
        missing = set(numeric + flags).difference(target.columns)
        if missing:
            raise ValueError(f'target is missing columns {sorted(missing)}')
        for name in numeric:
            target[name] = target[name].astype(float)
            if not np.isfinite(target[name].to_numpy()).all():
                raise ValueError(f'{name} must be finite')
        for name in flags:
            if target[name].isna().any() or not target[name].isin([False, True]).all():
                raise ValueError(f'{name} must contain boolean values')
            target[name] = target[name].astype(bool)
        if int(target[s.SETTLEMENT_CASH].sum()) != 1:
            raise ValueError('target requires exactly one settlement-cash asset')
        if (target[s.SETTLEMENT_CASH] & target[s.MANDATORY_TRADE]).any():
            raise ValueError('settlement cash funding is not a mandatory non-cash ticket')
        if (target[s.POLICY_MIN_WEIGHT] > target[s.POLICY_MAX_WEIGHT]).any():
            raise ValueError('policy lower bound exceeds upper bound')
        candidates = target[s.TRADE_CANDIDATE]
        invalid = (target[s.SETTLEMENT_CASH] | target[s.MANDATORY_TRADE]
                   | ~target[s.RULE4_TRADE] | ~target[s.REBALANCE_CADENCE_ELIGIBLE]
                   | ~target[s.MATERIAL_TRADE]
                   | target[s.DESIRED_REWEIGHT].abs().le(s.WEIGHT_ZERO_TOLERANCE))
        if (candidates & invalid).any():
            raise ValueError(
                'trade candidates must be material, eligible discretionary non-cash rows')
        covariance = self.covariance
        if not covariance.index.is_unique or not covariance.columns.is_unique:
            raise ValueError('covariance axes must be unique')
        if (not target.index.isin(covariance.index).all()
                or not target.index.isin(covariance.columns).all()):
            raise ValueError('covariance is missing execution instruments')
        covariance = covariance.loc[target.index, target.index].astype(float).copy()
        array = covariance.to_numpy()
        if not np.isfinite(array).all():
            raise ValueError('execution covariance must be finite')
        if not np.allclose(array, array.T, atol=1e-10):
            raise ValueError('execution covariance must be symmetric')
        if not self.alphas.index.is_unique or not self.asset_classes.index.is_unique:
            raise ValueError('alpha and asset-class indices must be unique')
        alphas = self.alphas.reindex(target.index).astype(float).copy()
        if not np.isfinite(alphas.fillna(0.0).to_numpy()).all():
            raise ValueError('execution alphas must be finite or missing')
        groups = self.asset_classes.reindex(target.index).copy()
        config = deepcopy(self.ranking_config)
        config.__post_init__()
        if groups.isna().any() or not groups.isin(config.tre_weight_by_asset_class).all():
            raise ValueError('asset classes must be complete and have configured TRE coefficients')
        product_weights = self.asset_class_tre_weights
        if product_weights is not None:
            if not product_weights.index.is_unique:
                raise ValueError('product TRE weight index must be unique')
            product_weights = product_weights.reindex(
                config.tre_weight_by_asset_class).astype(float)
            if (not np.isfinite(product_weights.to_numpy()).all()
                    or product_weights.lt(0.0).any()):
                raise ValueError('product TRE weights must be finite, complete and non-negative')
        for name in ('expand_for_feasibility', 'sign_directed_rescue', 'allow_corridor_relaxation'):
            if not isinstance(getattr(self, name), (bool, np.bool_)):
                raise ValueError(f'{name} must be boolean')
        partition = tuple(self.partition_groups)
        if len(set(partition)) != len(partition):
            raise ValueError('partition group names must be unique')
        if partition:
            group_constraints = self.constraints.group_lower_upper_constraints
            if (group_constraints is None or not set(partition).issubset(
                    group_constraints.group_loadings.columns)):
                raise ValueError('partition groups must name existing group constraints')
            loadings = group_constraints.group_loadings.reindex(target.index).loc[
                :, list(partition)]
            if (not loadings.isin([0.0, 1.0]).all().all()
                    or not loadings.sum(axis=1).eq(1.0).all()):
                raise ValueError(
                    'partition groups must be an exhaustive disjoint membership partition')
        for name, value in (
                ('target', target), ('covariance', covariance), ('alphas', alphas),
                ('asset_classes', groups), ('ranking_config', config),
                ('asset_class_tre_weights', product_weights), ('partition_groups', partition),
                ('constraints', deepcopy(self.constraints)),
                ('optimiser_config', deepcopy(self.optimiser_config))):
            object.__setattr__(self, name, value)
