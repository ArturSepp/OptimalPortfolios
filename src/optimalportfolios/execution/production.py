"""Resolve workbook policy into the shared declared execution domain.

Lifecycle resolution remains in the shared target builder. This bridge reuses
its instrument-bound compiler, preserves the original group rows, performs the
common risk-neutral preparation and returns the standard production audit table.
"""
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from optimalportfolios.execution import schema as s
from optimalportfolios.execution import instrument_bounds as shared
from optimalportfolios.execution.lifecycle import ExecutionLifecycleConfig
from optimalportfolios.optimization import ConstraintEnforcementType, OptimizationOutcome
from optimalportfolios.optimization.constraints.analytics import (
    ConstraintResidual, evaluate_constraint_residuals,
)
from .types import ExecutionProblem, ExecutionRules, RankedConfig
from .domain import audit_execution
from .preparation import prepare_common_execution
from .ranked_execution import wrapper_ranked_execution


@dataclass(frozen=True, kw_only=True)
class RankedExecutionPolicyConfig(ExecutionLifecycleConfig):
    """Ranked policy with actual-ticket limits and common feasibility preparation.

    The inherited cutoff is a hard held-position floor. The inherited materiality
    threshold is a hard actual non-exempt trade minimum. Neither is relaxed by
    sizing. Preparation may declare pinned compulsory corridor exceptions; its
    repairs reserve tickets before the discretionary allowance is applied.
    """

    ranked_config: RankedConfig = field(default_factory=RankedConfig)
    preparation_time_limit_s: float = 30.
    preparation_solver: str = 'MOSEK'
    expand_for_feasibility: bool = True

    @property
    def execution_method(self):
        """Distinguish the Ranked policy from legacy ranking and operational MIQP."""
        return 'ranked_execution'

    def __post_init__(self):
        """Reject ambiguous mixtures of V1 score controls and Ranked state search."""
        super().__post_init__()
        if self.exact_execution is not None:
            raise ValueError('Ranked policy cannot also select operational MIQP')
        if (self.minimum_trade_score is not None or self.sequential_greedy_selection
                or self.sign_directed_rescue or self.allow_corridor_relaxation
                or any(x != 1. for x in self.tre_weight_by_asset_class.values())):
            raise ValueError('Ranked uses unweighted full-covariance risk and its own state search')
        if not self.expand_for_feasibility:
            raise ValueError('Ranked requires common feasibility preparation')
        if (not np.isfinite(self.preparation_time_limit_s)
                or self.preparation_time_limit_s <= 0):
            raise ValueError('Preparation budget must be finite and positive')


def _execution_constraints(base):
    """Apply the existing ranked execution scope and reject unsupported hard rules."""
    # V1 explicitly excludes these inherited model optimisation controls from
    # execution. Preserve that existing scope, with original group bounds intact.
    scoped = replace(
        base, tracking_err_vol_constraint=None, turnover_constraint=None,
        turnover_costs=None, group_tracking_error_constraint=None,
        group_turnover_constraint=None, tre_utility_weight=None,
        turnover_utility_weight=None,
        constraint_enforcement_type=ConstraintEnforcementType.FORCED_CONSTRAINTS,
    )
    if not scoped.is_long_only or scoped.min_exposure != scoped.max_exposure:
        raise ValueError('Ranked requires long-only fixed investment')
    for name in ('max_target_portfolio_vol_an', 'target_return', 'benchmark_beta_constraint',
                 'sector_deviation_constraints', 'style_deviation_constraints',
                 'linear_constraints'):
        if getattr(scoped, name, None) is not None:
            raise ValueError(f'Unsupported extra hard rule: {name}')
    return scoped


def resolve_production_problem(trade_table, base_constraints, covariance, config):
    """Compile calendar-resolved lifecycle bounds without applying V1 score selection.

    Return the raw Ranked problem and the scoped original execution constraints.
    Common preparation is a separate step so every comparator receives identical
    prepared data and repair declarations.
    """
    constraints = _execution_constraints(base_constraints)
    table = trade_table.copy()
    table[s.SELECTED_TRADE] = (
        table[s.RULE4_TRADE].astype(bool)
        & table[s.REBALANCE_CADENCE_ELIGIBLE].astype(bool)
        & ~table[s.SETTLEMENT_CASH].astype(bool)
    )
    lo, hi = shared._build_selected_execution_bound_series(constraints, table, False)
    full_lo, full_hi = shared._build_selected_execution_bound_series(constraints, table, True)
    cash = int(np.flatnonzero(table[s.SETTLEMENT_CASH])[0])
    current = table[s.CURRENT_WEIGHT].to_numpy(dtype=float)
    base = table[s.BASE_WEIGHT].to_numpy(dtype=float)
    mandatory = table[s.MANDATORY_TRADE].to_numpy(dtype=bool)
    categories = table[s.MANDATORY_CATEGORY].astype(str).tolist()
    for i in np.flatnonzero(mandatory):
        if categories[i] == 'fixed_instruction':
            categories[i] = (
                'instruction_min_buy' if base[i] > current[i] else 'instruction_max_sell')
    categories[cash] = ''
    floor = table[s.EFFECTIVE_SMALL_TRADE_CUTOFF].to_numpy(dtype=float).copy()
    floor[cash] = 0.
    allowance = config.max_trades if config.add_mandatory_trades_to_max_trades else None
    cap = config.max_trades if allowance is None else int(mandatory.sum())+allowance
    rules = ExecutionRules(
        minimum_ticket=config.material_trade_threshold, holding_floor=floor,
        ticket_cap=cap, discretionary_allowance=allowance, exemption_mask=mandatory,
    )
    group = constraints.group_lower_upper_constraints
    if group is None:
        groups, group_lo, group_hi = np.zeros((0, len(table))), np.zeros(0), np.zeros(0)
    else:
        labels = group.group_loadings.columns
        groups = group.group_loadings.loc[table.index, labels].to_numpy(dtype=float).T
        group_lo = group.group_min_allocation.loc[labels].to_numpy(dtype=float)
        group_hi = group.group_max_allocation.loc[labels].to_numpy(dtype=float)
    fixed = (full_lo == full_hi).to_numpy(copy=True)
    fixed[cash] = False
    problem = ExecutionProblem(
        names=tuple(table.index), cash=cash, current=current, base=base,
        target=table[s.RAW_MODEL_WEIGHT].to_numpy(dtype=float),
        lower=lo.to_numpy(), upper=hi.to_numpy(), full_lower=full_lo.to_numpy(),
        full_upper=full_hi.to_numpy(), groups=groups, group_lower=group_lo,
        group_upper=group_hi, mandatory=mandatory, categories=tuple(categories),
        eligible=~fixed, frozen=fixed, rules=rules,
        covariance=covariance.loc[table.index, table.index].to_numpy(dtype=float),
        investment=constraints.min_exposure,
        provenance={'scope': 'ranked_execution', 'objective': 'full_covariance_raw_model',
                    'holding_floor': 'effective_asset_class_cutoff'},
    )
    return problem, constraints


def _production_result(problem, weights, table, constraints, engine, diagnostics, context):
    """Return production fields with both convex residuals and the strict Ranked gate."""
    index = table.index
    weights = pd.Series(weights, index=index, name='execution')
    constraints = replace(constraints, min_weights=pd.Series(problem.lower, index=index),
                          max_weights=pd.Series(problem.upper, index=index),
                          weights_0=table[s.CURRENT_WEIGHT],
                          benchmark_weights=table[s.RAW_MODEL_WEIGHT])
    audit = audit_execution(problem, weights.to_numpy())
    residuals = evaluate_constraint_residuals(
        weights.to_numpy(), constraints, tolerance=problem.rules.audit_tolerance)
    gate = ConstraintResidual('execution_policy', 'Execution independent audit',
                              float(not audit['passed']), 0., 0., float(not audit['passed']),
                              0., True, audit['passed'])
    accepted = audit['passed'] and all(x.passed for x in residuals if x.hard)
    outcome = OptimizationOutcome(
        weights.to_numpy(), accepted, 'OP Execution / '+engine, 'audited_incumbent', context,
        fallback_source=engine if engine.endswith('_fallback') else None,
        constraint_residuals=residuals+(gate,), constraints=constraints,
    )
    result = table.copy()
    result['original_base_weight'] = result[s.BASE_WEIGHT]
    result[s.BASE_WEIGHT] = problem.base
    result[s.MANDATORY_TRADE] = problem.mandatory
    result[s.MANDATORY_CATEGORY] = problem.categories
    result[shared.PROPOSED_WEIGHT] = weights
    result[shared.EXECUTED_TRADE] = weights-result[s.CURRENT_WEIGHT]
    noncash = ~result[s.SETTLEMENT_CASH].astype(bool)
    mandatory = result[s.MANDATORY_TRADE].astype(bool)
    traded = result[shared.EXECUTED_TRADE].abs() > problem.rules.zero_tolerance
    result[s.SELECTED_TRADE] = noncash & ~mandatory & traded
    result[s.REQUESTED_TRADE] = result[s.SELECTED_TRADE]
    result[s.FEASIBILITY_RESCUE_TRADE] = (
        result[s.MANDATORY_CATEGORY] == 'compulsory_saa_funding_repair')
    result[shared.MANDATORY_EXECUTED_TRADE] = (weights-result[s.CURRENT_WEIGHT]).where(mandatory, 0.)
    result[shared.DISCRETIONARY_EXECUTED_TRADE] = (weights-result[s.BASE_WEIGHT]).where(~mandatory, 0.)
    result[s.MANDATORY_FUNDING] = result[s.BASE_WEIGHT]-result[s.CURRENT_WEIGHT]
    funded = result[s.MANDATORY_FUNDING].where(noncash, 0.)
    category = result[s.MANDATORY_CATEGORY]
    result[s.ENTRY_FUNDING] = funded.where(category.eq('model_entry'), 0.)
    result[s.EXIT_FUNDING] = funded.where(category.isin(['model_exit', 'small_cutoff']), 0.)
    result[s.DESK_FUNDING] = funded.where(category.isin(
        ['instruction_min_buy', 'instruction_max_sell']), 0.)
    result[s.OTHER_FUNDING] = (
        funded-result[s.ENTRY_FUNDING]-result[s.EXIT_FUNDING]-result[s.DESK_FUNDING])
    result[shared.TARGET_GAP] = weights-result[s.EFFECTIVE_MODEL_WEIGHT]
    result[shared.STRICT_CORRIDOR_MIN] = problem.lower
    result[shared.STRICT_CORRIDOR_MAX] = problem.upper
    result[shared.CORRIDOR_RELAXATION] = 0.
    result[shared.RELAXED_CORRIDOR_SOLVE] = False
    result[shared.REWEIGHT_FUNDING] = 0.
    result.loc[~noncash, shared.REWEIGHT_FUNDING] = float(
        (weights-result[s.BASE_WEIGHT])[noncash].sum())
    result[shared.EXPECTED_CASH_WEIGHT] = weights.where(~noncash)
    result['group_mechanism_abandoned'] = False
    result['group_mechanism_residual_bridge'] = 0.
    result['group_mechanism_residual_bridge_bp'] = 0.
    result['group_mechanism_failure_diagnostic'] = ''
    result['group_mechanism_fallback_diagnostic'] = ''
    result[shared.DIRECTED_RESCUE_CANDIDATE] = False
    result[shared.RESCUE_REASON] = result[s.MANDATORY_CATEGORY].where(
        result[s.FEASIBILITY_RESCUE_TRADE], '')
    result['execution_engine'] = engine
    result.attrs['execution'] = dict(diagnostics, audit=audit)
    return shared.ExecutionOptimizationResult(weights=weights, outcome=outcome, trade_table=result)


def solve_ranked_execution_portfolio(trade_table, base_constraints, covariance, config, *,
                                 context=''):
    """Prepare and solve Ranked Execution, preserving explicit failure on abstention.

    The retired V1 fallback is not part of this interface. Failure retains
    the original trade table for dated no-trade reporting, never an executable
    model-display fallback.
    """
    diagnostics = {}
    try:
        raw, constraints = resolve_production_problem(
            trade_table, base_constraints, covariance, config)
        problem, preparation = prepare_common_execution(
            raw, time_limit_s=config.preparation_time_limit_s, solver=config.preparation_solver)
        diagnostics['preparation'] = preparation
        if problem is None:
            raise RuntimeError('Ranked common preparation failed')
        solved = wrapper_ranked_execution(problem, config.ranked_config)
        diagnostics.update(solved.diagnostics)
        diagnostics['ranked_status'] = solved.status
        weights, engine = solved.weights, 'ranked_execution'
        if weights is None:
            raise RuntimeError('Ranked bounded abstention without an audited delivery')
        return _production_result(
            problem, weights, trade_table, constraints, engine, diagnostics, context)
    except (ValueError, RuntimeError) as error:
        error.trade_table = trade_table.copy()
        error.diagnostics = diagnostics
        raise
