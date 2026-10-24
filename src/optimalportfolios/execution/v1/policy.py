"""
Execution-policy specification, target preparation and trade prioritisation.

This module is deliberately independent of the model backtest. It consumes
one stored model cross-section plus the previously implemented portfolio and
produces a deterministic ranked trade table. The later execution optimiser
will consume only the selected rows and enforce the product constraints.

Policy mechanics implemented here are:

* fix zero-model holdings, model entries/exits and dated instructions first;
* expose held sub-cutoff positions through sell-down corridors when their
  capacity is needed by a hard group minimum;
* fund those mandatory non-cash trades through one settlement-cash asset;
* honour each instrument's workbook rebalancing frequency;
* drift the prior implementation weights using realised prices;
* rank only the remaining current-to-model trades using cash-aware marginal
  alpha and marginal tracking error from the full covariance matrix;
* optionally re-score the remaining candidates after each selected trade;
* select a configurable number of trades under either a total-ticket cap or
  an optional discretionary allowance that adds active mandatory tickets.

No optimisation, workbook I/O or reporting lives here.
"""
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
import qis as qis

from optimalportfolios.utils.weights_drift import apply_drift_to_weights_0
from optimalportfolios.execution.v1.feasibility import compute_group_bound_bridges
from optimalportfolios.execution._common.config import ExactExecutionConfig
from optimalportfolios.execution.v1.ranking import (
    resolve_effective_max_trades as resolve_effective_max_trades,
)
from optimalportfolios.execution.v1.ranking import (
    _incremental_funded_tre as _incremental_funded_tre,
    score_execution_trades as score_execution_trades,
)


from optimalportfolios.execution.v1.schema import WEIGHT_ZERO_TOLERANCE as WEIGHT_ZERO_TOLERANCE
# Model solver residuals up to 0.01 bp are numerical zero for discrete rules.
# Keep the tighter holding/trade/bound tolerance separate; raw model weights
# remain unchanged for risk objectives and diagnostics.
from optimalportfolios.execution._common.model import (
    MODEL_WEIGHT_ZERO_TOLERANCE as MODEL_WEIGHT_ZERO_TOLERANCE, is_model_zero,
)



# Target-band risk-budget-ranked execution diagnostics.  These are separate
# from the legacy marginal-alpha/TRE score so the two policies cannot be
# mistaken for one another in downstream reports.


from optimalportfolios.execution.v1.schema import (
    ALLOW_CURRENT_PIN as ALLOW_CURRENT_PIN,
    MODEL_MAX_REPAIR as MODEL_MAX_REPAIR,
    CURRENT_WEIGHT as CURRENT_WEIGHT,
    PREVIOUS_MODEL_WEIGHT as PREVIOUS_MODEL_WEIGHT,
    RAW_MODEL_WEIGHT as RAW_MODEL_WEIGHT,
    POLICY_TARGET_WEIGHT as POLICY_TARGET_WEIGHT,
    EFFECTIVE_MODEL_WEIGHT as EFFECTIVE_MODEL_WEIGHT,
    BASE_WEIGHT as BASE_WEIGHT,
    TRADE_WEIGHT as TRADE_WEIGHT,
    LOW_WEIGHT_EXCLUDED as LOW_WEIGHT_EXCLUDED,
    CUTOFF_SELL_DOWN as CUTOFF_SELL_DOWN,
    CUTOFF_SELL_DOWN_REASON as CUTOFF_SELL_DOWN_REASON,
    CUTOFF_SELL_DOWN_RETENTION as CUTOFF_SELL_DOWN_RETENTION,
    CUTOFF_SELL_DOWN_RETAINED_FRACTION as CUTOFF_SELL_DOWN_RETAINED_FRACTION,
    CUTOFF_SELL_DOWN_DUST_ROUNDED as CUTOFF_SELL_DOWN_DUST_ROUNDED,
    SELL_DOWN_DUST_THRESHOLD as SELL_DOWN_DUST_THRESHOLD,
    EFFECTIVE_SMALL_TRADE_CUTOFF as EFFECTIVE_SMALL_TRADE_CUTOFF,
    CUTOFF_RESIDUAL_BRIDGE as CUTOFF_RESIDUAL_BRIDGE,
    CUTOFF_RESIDUAL_BRIDGE_BP as CUTOFF_RESIDUAL_BRIDGE_BP,
    CUTOFF_RESIDUAL_BRIDGE_GROUP as CUTOFF_RESIDUAL_BRIDGE_GROUP,
    MODEL_ENTRY as MODEL_ENTRY,
    MODEL_EXIT as MODEL_EXIT,
    CUTOFF_EXIT as CUTOFF_EXIT,
    INSTRUCTION_MIN_BUY as INSTRUCTION_MIN_BUY,
    INSTRUCTION_MAX_SELL as INSTRUCTION_MAX_SELL,
    INSTRUCTION_MIN_WEIGHT as INSTRUCTION_MIN_WEIGHT,
    INSTRUCTION_MAX_WEIGHT as INSTRUCTION_MAX_WEIGHT,
    EXECUTION_INSTRUCTION as EXECUTION_INSTRUCTION,
    INSTRUCTION_COMMENT as INSTRUCTION_COMMENT,
    INSTRUCTION_SOURCE as INSTRUCTION_SOURCE,
    INSTRUCTION_WITHIN_TOLERANCE as INSTRUCTION_WITHIN_TOLERANCE,
    INSTRUCTION_RETAINED_BREACH as INSTRUCTION_RETAINED_BREACH,
    FIXED_INSTRUCTION as FIXED_INSTRUCTION,
    PINNED_INSTRUCTION as PINNED_INSTRUCTION,
    MANDATORY_REASON as MANDATORY_REASON,
    MANDATORY_CATEGORY as MANDATORY_CATEGORY,
    MANDATORY_FUNDING as MANDATORY_FUNDING,
    ENTRY_FUNDING as ENTRY_FUNDING,
    EXIT_FUNDING as EXIT_FUNDING,
    DESK_FUNDING as DESK_FUNDING,
    OTHER_FUNDING as OTHER_FUNDING,
    SETTLEMENT_CASH as SETTLEMENT_CASH,
    RULE4_TRADE as RULE4_TRADE,
    DESIRED_REWEIGHT as DESIRED_REWEIGHT,
    MANDATORY_TRADE as MANDATORY_TRADE,
    REBALANCE_CADENCE_ELIGIBLE as REBALANCE_CADENCE_ELIGIBLE,
    MATERIAL_TRADE as MATERIAL_TRADE,
    MATERIAL_TRADE_THRESHOLD as MATERIAL_TRADE_THRESHOLD,
    TRADE_CANDIDATE as TRADE_CANDIDATE,
    POLICY_MIN_WEIGHT as POLICY_MIN_WEIGHT,
    POLICY_MAX_WEIGHT as POLICY_MAX_WEIGHT,
    PRODUCT_MIN_WEIGHT as PRODUCT_MIN_WEIGHT,
    PRODUCT_MAX_WEIGHT as PRODUCT_MAX_WEIGHT,
    PRODUCT_BOUND_OVERRIDE as PRODUCT_BOUND_OVERRIDE,
    PRODUCT_BOUND_OVERRIDE_BREACH as PRODUCT_BOUND_OVERRIDE_BREACH,
    PRODUCT_BOUND_OVERRIDE_REASON as PRODUCT_BOUND_OVERRIDE_REASON,
    ASSET_CLASS as ASSET_CLASS,
    ALPHA as ALPHA,
    MARGINAL_ALPHA as MARGINAL_ALPHA,
    COVAR_ACTIVE as COVAR_ACTIVE,
    MARGINAL_TRE as MARGINAL_TRE,
    TRE_WEIGHT as TRE_WEIGHT,
    TRADE_SCORE as TRADE_SCORE,
    SCORE_PER_TURNOVER as SCORE_PER_TURNOVER,
    TRADE_RANK as TRADE_RANK,
    SELECTED_TRADE as SELECTED_TRADE,
    REQUESTED_TRADE as REQUESTED_TRADE,
    FEASIBILITY_RESCUE_TRADE as FEASIBILITY_RESCUE_TRADE,
    SELECTION_SCORE as SELECTION_SCORE,
    SELECTION_MARGINAL_TRE as SELECTION_MARGINAL_TRE,
    SEQUENTIAL_SELECTION_ORDER as SEQUENTIAL_SELECTION_ORDER,
    ALPHA_MISSING as ALPHA_MISSING,
    ALPHA_STRENGTH as ALPHA_STRENGTH,
    TARGET_BAND_WIDTH as TARGET_BAND_WIDTH,
    TARGET_BAND_LOWER as TARGET_BAND_LOWER,
    TARGET_BAND_UPPER as TARGET_BAND_UPPER,
    MODEL_GAP as MODEL_GAP,
    DISTANCE_TO_BAND as DISTANCE_TO_BAND,
    ABS_DISTANCE_TO_BAND as ABS_DISTANCE_TO_BAND,
    GAP_BAND_SCORE as GAP_BAND_SCORE,
    ASSET_CLASS_RANK as ASSET_CLASS_RANK,
    RANKING_ELIGIBLE as RANKING_ELIGIBLE,
    RISK_BUDGET_SELECTED as RISK_BUDGET_SELECTED,
    GROUP_TRE_WEIGHT_RAW as GROUP_TRE_WEIGHT_RAW,
    GROUP_TRE_WEIGHT_NORMALIZED as GROUP_TRE_WEIGHT_NORMALIZED,
    GROUP_TRE_BUDGET as GROUP_TRE_BUDGET,
    GROUP_TRE_INITIAL as GROUP_TRE_INITIAL,
    GROUP_TRE_BEFORE as GROUP_TRE_BEFORE,
    GROUP_TRE_AFTER as GROUP_TRE_AFTER,
    GROUP_TRE_PREFIX_CURVE as GROUP_TRE_PREFIX_CURVE,
    GROUP_K_RISK as GROUP_K_RISK,
    GROUP_K_FINAL as GROUP_K_FINAL,
    GROUP_TRE_REALIZED as GROUP_TRE_REALIZED,
    GROUP_TRE_BUDGET_DIFFERENCE as GROUP_TRE_BUDGET_DIFFERENCE,
    GROUP_TRE_BUDGET_MET as GROUP_TRE_BUDGET_MET,
    GROUP_SELECTION_STATUS as GROUP_SELECTION_STATUS,
    BASE_CONSTRAINT_SET as BASE_CONSTRAINT_SET,
    EXECUTION_CONSTRAINT_SET as EXECUTION_CONSTRAINT_SET,
    CONSTRAINT_SETS as CONSTRAINT_SETS,
)

CURRENT_PIN_ROUNDING_TOLERANCE: float = 0.5 / 10_000.0
DEFAULT_SMALL_TRADE_CUTOFF: float = 0.0049
DEFAULT_SELL_DOWN_DUST_THRESHOLD: float = 1e-4
DEFAULT_MATERIAL_TRADE_THRESHOLD: float = 0.0020


def _default_tre_weights() -> Dict[str, float]:
    """Return an empty map; consumers supply class coefficients."""
    return {}


def _default_small_trade_cutoffs() -> Dict[str, float]:
    """Return no class overrides; consumers supply their holding policy."""
    return {}


@dataclass(frozen=True)
class MandatoryTrades:
    """Optional live instructions that override the model target.

    Attributes:
        hard_sell: Instruments whose effective target must be zero.
        hard_buy_min: Instrument-to-minimum-weight instructions. The effective
            target is raised to this weight when it is lower.
        hard_max: Instrument-to-maximum-weight instructions.
        hard_target: Instrument-to-fixed-target instructions.
    """
    hard_sell: Tuple[str, ...] = ()
    hard_buy_min: Dict[str, float] = field(default_factory=dict)
    hard_max: Dict[str, float] = field(default_factory=dict)
    hard_target: Dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate the supplied policy fields."""
        if len(self.hard_sell) != len(set(self.hard_sell)):
            raise ValueError("hard_sell contains duplicate instruments")
        sell = set(self.hard_sell)
        overlap = sell.intersection(
            set(self.hard_buy_min).union(self.hard_max).union(self.hard_target)
        )
        if overlap:
            raise ValueError(
                'instruments cannot be both hard sells and hard buys or another '
                'instruction: '
                f'{sorted(overlap)}'
            )
        target_overlap = set(self.hard_target).intersection(
            set(self.hard_buy_min).union(self.hard_max)
        )
        if target_overlap:
            raise ValueError(
                'hard_target cannot be combined with hard_buy_min or hard_max: '
                f'{sorted(target_overlap)}'
            )
        all_weights = {
            **self.hard_buy_min,
            **self.hard_max,
            **self.hard_target,
        }
        invalid = {
            instrument: weight for instrument, weight in all_weights.items()
            if not np.isfinite(weight) or weight < 0.0 or weight > 1.0
        }
        if invalid:
            raise ValueError(f"mandatory instruction weights must be in [0, 1]: {invalid}")
        incompatible = {
            instrument: (self.hard_buy_min[instrument], self.hard_max[instrument])
            for instrument in set(self.hard_buy_min).intersection(self.hard_max)
            if self.hard_buy_min[instrument] > self.hard_max[instrument]
        }
        if incompatible:
            raise ValueError(
                f'hard-buy minimum exceeds hard maximum: {incompatible}'
            )


def merge_mandatory_trades(*instructions: MandatoryTrades) -> MandatoryTrades:
    """Merge non-conflicting instruction sets without silent precedence."""
    hard_sell = []
    hard_buy_min: Dict[str, float] = {}
    hard_max: Dict[str, float] = {}
    hard_target: Dict[str, float] = {}
    for instruction in instructions:
        hard_sell.extend(instruction.hard_sell)
        for instrument, weight in instruction.hard_buy_min.items():
            hard_buy_min[instrument] = max(hard_buy_min.get(instrument, 0.0), weight)
        for instrument, weight in instruction.hard_max.items():
            hard_max[instrument] = min(hard_max.get(instrument, 1.0), weight)
        for instrument, weight in instruction.hard_target.items():
            if instrument in hard_target and not np.isclose(
                    hard_target[instrument], weight,
                    atol=WEIGHT_ZERO_TOLERANCE, rtol=0.0):
                raise ValueError(
                    f'conflicting hard targets for {instrument!r}: '
                    f'{hard_target[instrument]} and {weight}'
                )
            hard_target[instrument] = weight
    return MandatoryTrades(
        hard_sell=tuple(dict.fromkeys(hard_sell)),
        hard_buy_min=hard_buy_min,
        hard_max=hard_max,
        hard_target=hard_target,
    )






@dataclass(frozen=True)
class ExecutionPolicyConfig:
    """Sweepable parameters for one model-execution policy.

    Attributes:
        small_trade_cutoff: Raw model weights at or below this size are fixed
            to zero unless an overriding hard minimum requires a holding.
        small_trade_cutoff_by_asset_class: Optional level-0 overrides for the
            uniform cutoff. The default is zero for Liquidity and Fixed Income
            and 0.49% for Equity and Alternatives, as specified in Exhibit 2.
            Explicit ``None`` applies ``small_trade_cutoff`` uniformly.
        sell_down_dust_threshold: Accepted sell-down weights below this size
            may be rounded to zero after a hard-compliance audit.
        material_trade_threshold: Minimum absolute trade weight eligible for
            ranking and implementation.
        max_trades: Configured non-cash ticket allowance. It is a hard total
            cap by default. When mandatory-ticket expansion is enabled, it is
            the discretionary allowance and the effective total cap varies by
            decision.
        add_mandatory_trades_to_max_trades: When true, ``max_trades`` is the
            discretionary ticket allowance and active mandatory non-cash
            tickets are added to the effective total cap for each decision.
            False preserves the historical total-ticket-cap semantics.
        expand_for_feasibility: Add the smallest score-ranked prefix of
            otherwise excluded trades when the requested selection is
            incompatible with hard product constraints.
        minimum_trade_score: Optional lower score cutoff for discretionary
            trades.
        sequential_greedy_selection: Re-score remaining candidates after each
            selected cash-funded trade. False keeps one-shot ranking.
        sign_directed_rescue: Restrict feasibility rescue to trades whose
            direction reduces a breached cash or group-bound residual.
        tre_weight_by_asset_class: Linear marginal-TRE coefficient selected by
            each instrument's level-0 asset class. Scoring multiplies this
            policy coefficient by the product ``TRE weight`` loaded from
            ``saa_asset_class``. Marginal TRE itself uses the complete
            covariance matrix and the instrument-versus-cash trade.
        use_weight_drift: Drift the previous implementation target to the next
            decision date using realised prices.
        allow_corridor_relaxation: Permit the final, explicitly reported
            feasibility stage to relax selected base-to-model corridors.
        mandatory_trades: Optional explicit live hard sells, minimum buys,
            maximums and fixed targets.
        output_by_ticker: Format policy output with workbook tickers as index
            instead of internal instrument names.
        instruction_correction_tolerance: Dead-band for corrections caused by
            desk minimum, maximum and fixed-target instructions. A breach no
            larger than this value is retained at the current weight. Product
            bounds and lifecycle trades remain exact. Zero preserves the
            pre-dead-band behaviour.
        exact_execution: Optional Exact Execution solver configuration. None
            retains the ranked-selection and continuous-projection policy.
    """
    small_trade_cutoff: float = DEFAULT_SMALL_TRADE_CUTOFF
    small_trade_cutoff_by_asset_class: Optional[Dict[str, float]] = field(
        default_factory=_default_small_trade_cutoffs
    )
    sell_down_dust_threshold: float = DEFAULT_SELL_DOWN_DUST_THRESHOLD
    material_trade_threshold: float = DEFAULT_MATERIAL_TRADE_THRESHOLD
    max_trades: Optional[int] = None
    expand_for_feasibility: bool = False
    minimum_trade_score: Optional[float] = None
    sequential_greedy_selection: bool = False
    sign_directed_rescue: bool = False
    allow_corridor_relaxation: bool = False
    tre_weight_by_asset_class: Dict[str, float] = field(
        default_factory=_default_tre_weights
    )
    use_weight_drift: bool = True
    mandatory_trades: MandatoryTrades = field(default_factory=MandatoryTrades)
    output_by_ticker: bool = False
    instruction_correction_tolerance: float = 0.0
    exact_execution: Optional[ExactExecutionConfig] = None
    add_mandatory_trades_to_max_trades: bool = False

    @property
    def execution_method(self) -> str:
        """Return the desk-facing ranked, exact, or exact-soft method name."""
        if self.exact_execution is None:
            return 'ranked'
        if self.exact_execution.sub_asset_class_penalty is not None:
            return 'exact_soft'
        return 'exact'

    def __post_init__(self) -> None:
        """Validate the supplied policy fields."""
        if not np.isfinite(self.small_trade_cutoff) or self.small_trade_cutoff < 0.0:
            raise ValueError(
                "small_trade_cutoff must be finite and non-negative, got "
                f"{self.small_trade_cutoff}"
            )
        if self.small_trade_cutoff_by_asset_class is not None:
            unsupported = set(self.small_trade_cutoff_by_asset_class).difference(
                tuple(self.tre_weight_by_asset_class)
            )
            if unsupported:
                raise ValueError(
                    'small_trade_cutoff_by_asset_class contains unsupported '
                    f'level-0 classes {sorted(unsupported)}'
                )
            invalid_cutoffs = {
                key: value
                for key, value in self.small_trade_cutoff_by_asset_class.items()
                if not np.isfinite(value) or value < 0.0
            }
            if invalid_cutoffs:
                raise ValueError(
                    'asset-class cutoffs must be finite and non-negative: '
                    f'{invalid_cutoffs}'
                )
        if (not np.isfinite(self.sell_down_dust_threshold)
                or self.sell_down_dust_threshold < 0.0):
            raise ValueError(
                'sell_down_dust_threshold must be finite and non-negative, got '
                f'{self.sell_down_dust_threshold}'
            )
        if (not np.isfinite(self.material_trade_threshold)
                or self.material_trade_threshold <= 0.0):
            raise ValueError(
                "material_trade_threshold must be finite and positive, got "
                f"{self.material_trade_threshold}"
            )
        if (not np.isfinite(self.instruction_correction_tolerance)
                or self.instruction_correction_tolerance < 0.0):
            raise ValueError(
                'instruction_correction_tolerance must be finite and '
                'non-negative, got '
                f'{self.instruction_correction_tolerance}'
            )
        if self.max_trades is not None and self.max_trades <= 0:
            raise ValueError("max_trades must be positive or None")
        if not isinstance(
                self.add_mandatory_trades_to_max_trades, (bool, np.bool_)):
            raise ValueError(
                'add_mandatory_trades_to_max_trades must be boolean'
            )
        if self.exact_execution is not None:
            if self.max_trades is None:
                raise ValueError('Exact Execution requires a finite max_trades')
            if (isinstance(self.max_trades, (bool, np.bool_))
                    or not isinstance(self.max_trades, (int, np.integer))):
                raise ValueError('Exact Execution max_trades must be an integer')
            if self.expand_for_feasibility:
                raise ValueError(
                    'Exact Execution does not use expand_for_feasibility'
                )
            if self.minimum_trade_score is not None:
                raise ValueError('Exact Execution does not use minimum_trade_score')
            if self.sequential_greedy_selection:
                raise ValueError('Exact Execution does not use sequential selection')
            if self.sign_directed_rescue:
                raise ValueError('Exact Execution does not use sign-directed rescue')
            if self.allow_corridor_relaxation:
                raise ValueError('Exact Execution does not use corridor relaxation')
        if (self.minimum_trade_score is not None
                and not np.isfinite(self.minimum_trade_score)):
            raise ValueError('minimum_trade_score must be finite or None')
        if not self.tre_weight_by_asset_class:
            raise ValueError(
                "tre_weight_by_asset_class must define at least one class"
            )
        invalid_tre = {
            key: value
            for key, value in self.tre_weight_by_asset_class.items()
            if not np.isfinite(value) or value < 0.0
        }
        if invalid_tre:
            raise ValueError(f"TRE weights must be finite and non-negative: {invalid_tre}")


@dataclass(frozen=True, kw_only=True)
class RankedExecutionPolicyConfig(ExecutionPolicyConfig):
    """Workbook-ranked policy carrying an explicit level-2 constraint set."""

    constraint_set: str = BASE_CONSTRAINT_SET

    def __post_init__(self) -> None:
        """Validate the ranked policy and its level-2 constraint selector."""
        super().__post_init__()
        if self.constraint_set not in CONSTRAINT_SETS:
            raise ValueError(
                f'constraint_set must be one of {list(CONSTRAINT_SETS)}, got '
                f'{self.constraint_set!r}'
            )


@dataclass(frozen=True, kw_only=True)
class RiskBudgetRankedExecutionPolicyConfig(ExecutionPolicyConfig):
    """Current/model target-band policy with risk-derived class ticket counts.

    This is a distinct policy type so legacy ranked and Exact configurations
    retain their published constructor and numerical behaviour.

    Attributes:
        execution_tre_budget: Total annualized implementation-TRE selection
            budget as a volatility fraction, for example ``0.005`` for 50 bp.
        target_band_tight_by_asset_class: Half-width used at unit alpha
            strength for every level-0 asset class.
        target_band_wide_by_asset_class: Half-width used at zero alpha
            strength for every level-0 asset class.
        constraint_set: Level-2 bounds used by the execution solve.
    """

    execution_tre_budget: float
    target_band_tight_by_asset_class: Dict[str, float]
    target_band_wide_by_asset_class: Dict[str, float]
    constraint_set: str = EXECUTION_CONSTRAINT_SET
    instruction_correction_tolerance: float = 0.0
    sell_down_dust_threshold: float = 0.0

    @property
    def execution_method(self) -> str:
        """Return the desk-facing method selector."""
        return 'risk_budget_ranked'

    def __post_init__(self) -> None:
        """Validate the risk budget and complete per-class target bands."""
        super().__post_init__()
        if self.instruction_correction_tolerance != 0.0:
            raise ValueError(
                'risk-budget-ranked execution does not use Instruction Tolerance'
            )
        if self.sell_down_dust_threshold != 0.0:
            raise ValueError(
                'risk-budget-ranked execution does not use Sell Down Dust Threshold'
            )
        if self.exact_execution is not None:
            raise ValueError(
                'risk-budget-ranked execution cannot also configure Exact Execution'
            )
        if (not np.isfinite(self.execution_tre_budget)
                or self.execution_tre_budget <= 0.0):
            raise ValueError('execution_tre_budget must be finite and positive')
        for name, values in (
                ('target_band_tight_by_asset_class',
                 self.target_band_tight_by_asset_class),
                ('target_band_wide_by_asset_class',
                 self.target_band_wide_by_asset_class)):
            if set(values) != set(tuple(self.tre_weight_by_asset_class)):
                raise ValueError(
                    f'{name} must define exactly {list(tuple(self.tre_weight_by_asset_class))}')
            invalid = {
                key: value for key, value in values.items()
                if not np.isfinite(value) or value < 0.0
            }
            if invalid:
                raise ValueError(f'{name} must be finite and non-negative: {invalid}')
        inverted = {
            asset_class: (
                self.target_band_tight_by_asset_class[asset_class],
                self.target_band_wide_by_asset_class[asset_class],
            )
            for asset_class in tuple(self.tre_weight_by_asset_class)
            if self.target_band_tight_by_asset_class[asset_class]
            > self.target_band_wide_by_asset_class[asset_class]
        }
        if inverted:
            raise ValueError(f'target tight band exceeds wide band: {inverted}')
        if self.constraint_set not in CONSTRAINT_SETS:
            raise ValueError(
                f'constraint_set must be one of {list(CONSTRAINT_SETS)}, got '
                f'{self.constraint_set!r}'
            )




def build_instrument_rebalancing_indicators(
        decision_dates: pd.DatetimeIndex,
        instrument_frequencies: pd.Series,
) -> pd.DataFrame:
    """Build the policy calendar from each instrument's workbook frequency.

    The model covariance dates are the decision grid. Frequency schedules are
    generated with the same qis helper used by the model backtest and then
    matched exactly to that grid. Every instrument is enabled on the first
    decision date to initialise a historical execution portfolio.

    Args:
        decision_dates: Sorted model covariance/decision dates.
        instrument_frequencies: Frequency strings indexed by instrument.

    Returns:
        Boolean DataFrame indexed by decision date and instrument.
    """
    dates = pd.DatetimeIndex(decision_dates)
    if dates.empty or not dates.is_unique or not dates.is_monotonic_increasing:
        raise ValueError("decision_dates must be non-empty, unique and increasing")
    if instrument_frequencies.empty or not instrument_frequencies.index.is_unique:
        raise ValueError("instrument frequencies must be non-empty with a unique index")
    if instrument_frequencies.isna().any():
        missing = instrument_frequencies.index[instrument_frequencies.isna()].tolist()
        raise ValueError(f"missing instrument rebalancing frequencies for {missing}")

    schedule = qis.create_rebalancing_indicators_from_freqs(
        rebalancing_freqs=instrument_frequencies.astype(str),
        time_period=qis.TimePeriod(dates[0], dates[-1]),
        include_start_date=True,
    )
    indicators = schedule.reindex(index=dates, columns=instrument_frequencies.index)
    indicators = indicators.fillna(0.0).astype(bool)
    indicators.iloc[0, :] = True
    return indicators


def drift_execution_weights(
        previous_target: pd.Series,
        prices: pd.DataFrame,
        previous_date: Optional[pd.Timestamp],
        decision_date: pd.Timestamp,
        use_weight_drift: bool = True,
) -> pd.Series:
    """Return pre-trade implementation weights at one decision date.

    Args:
        previous_target: Implemented target weights at the prior decision.
        prices: Reference-currency total-return price panel.
        previous_date: Date on which ``previous_target`` was implemented.
        decision_date: Current execution decision date.
        use_weight_drift: False retains the prior target unchanged.

    Returns:
        Drifted weights with the same index as ``previous_target``.
    """
    drifted = apply_drift_to_weights_0(
        weights_0=previous_target,
        prices=prices,
        prev_date=previous_date,
        date=decision_date,
        use_drifted_weights_0=use_weight_drift,
    )
    if drifted is None:
        raise ValueError("previous_target must not be None")
    return drifted


def build_execution_target(
        current_weights: pd.Series,
        model_weights: pd.Series,
        config: ExecutionPolicyConfig,
        previous_model_weights: Optional[pd.Series] = None,
        settlement_cash_asset: Optional[str] = None,
        rebalancing_allowed: Optional[pd.Series] = None,
        required_min_weights: Optional[pd.Series] = None,
        required_max_weights: Optional[pd.Series] = None,
        mandatory_trades: Optional[MandatoryTrades] = None,
        instruction_audit: Optional[pd.DataFrame] = None,
        asset_classes: Optional[pd.Series] = None,
        group_loadings: Optional[pd.DataFrame] = None,
        group_min: Optional[pd.Series] = None,
        group_max: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """Build lifecycle targets, mandatory trades, funding, and Rule 4 trades.

    Workbook Current Min/Max or explicit hard instructions are intersected
    with ordinary product bounds. The sole exception is a desk-owned exact
    Current Min/Max pin matching the live current weight within 0.5 bp: it may
    grandfather that existing product-bound breach at zero trade. Small model
    weights, model entries, and model exits are then consolidated into one
    mandatory target per instrument. Settlement cash absorbs the complete
    non-cash mandatory funding leg. Only the remaining current-to-model moves
    are Rule 4 candidates for ranking.

    Args:
        current_weights: Drifted pre-trade implementation weights.
        model_weights: Model-backtest weights at the decision date. Absolute
            weights up to 1e-6 (0.01 bp) are zero for execution eligibility;
            the raw weights remain available for diagnostics and risk scoring.
        config: Execution policy parameters.
        previous_model_weights: Model weights at each instrument's previous
            eligible rebalance. None treats the current model as previous and
            therefore creates no model-entry/exit transition labels.
        settlement_cash_asset: Designated cash instrument. None auto-detects a
            unique index label containing ``cash`` when available.
        rebalancing_allowed: Instrument-frequency indicator for the date.
        required_min_weights: Ordinary product minimum instrument weights.
        required_max_weights: Ordinary product maximum instrument weights.
        mandatory_trades: Resolved rebalance-date instructions. None uses
            ``config.mandatory_trades``.
        instruction_audit: Optional resolved instruction table carrying
            ``Instruction``, ``Comment`` and ``Source`` columns.
        asset_classes: Level-0 class per instrument. Owner-default class
            cutoffs fall back to the scalar cutoff when metadata is omitted
            by a lightweight caller; custom class overrides require it.
        group_loadings: Optional instrument-by-group allocation loadings used
            to preserve group-minimum capacity before cutoff sells.
        group_min: Hard group minimum allocations aligned to loading columns.
        group_max: Hard group maximum allocations aligned to loading columns.

    Returns:
        Instrument-indexed target and eligibility table.
    """
    index = current_weights.index
    if not index.is_unique:
        raise ValueError("current_weights index must be unique")
    current = current_weights.astype(float)
    model = model_weights.reindex(index).fillna(0.0).astype(float)
    previous_model = (
        model.copy() if previous_model_weights is None
        else previous_model_weights.reindex(index).fillna(0.0).astype(float)
    )
    if (not np.isfinite(current.to_numpy()).all()
            or not np.isfinite(model.to_numpy()).all()
            or not np.isfinite(previous_model.to_numpy()).all()):
        raise ValueError("current and model weights must be finite")
    model_for_rules = model.mask(is_model_zero(model), 0.0)
    previous_model_for_rules = previous_model.mask(
        is_model_zero(previous_model), 0.0
    )
    current_for_rules = current.mask(
        current.abs().le(WEIGHT_ZERO_TOLERANCE), 0.0
    )

    cash_asset = settlement_cash_asset
    if cash_asset is None:
        cash_candidates = index[index.astype(str).str.contains('cash', case=False)]
        if len(cash_candidates) == 1:
            cash_asset = str(cash_candidates[0])
    if cash_asset is not None and cash_asset not in index:
        raise ValueError(f"settlement cash asset {cash_asset!r} is not in the universe")
    is_cash = pd.Series(index == cash_asset, index=index)

    effective_cutoff = pd.Series(config.small_trade_cutoff, index=index)
    if config.small_trade_cutoff_by_asset_class is not None:
        if asset_classes is None:
            if (
                config.small_trade_cutoff_by_asset_class
                != _default_small_trade_cutoffs()
            ):
                raise ValueError(
                    'asset_classes are required for custom '
                    'small_trade_cutoff_by_asset_class'
                )
        else:
            aligned_classes = asset_classes.reindex(index)
            if aligned_classes.isna().any():
                raise ValueError(
                    'missing asset classes for cutoff overrides: '
                    f'{aligned_classes.index[aligned_classes.isna()].tolist()}'
                )
            unsupported = set(aligned_classes).difference(tuple(config.tre_weight_by_asset_class))
            if unsupported:
                raise ValueError(
                    f'unsupported level-0 asset classes {sorted(unsupported)}'
                )
            overrides = aligned_classes.map(config.small_trade_cutoff_by_asset_class)
            effective_cutoff = overrides.fillna(effective_cutoff).astype(float)

    allowed = (
        pd.Series(True, index=index) if rebalancing_allowed is None
        else rebalancing_allowed.reindex(index).fillna(False).astype(bool)
    )
    low_weight = model_for_rules.le(effective_cutoff) & ~is_cash
    policy_target = model_for_rules.mask(low_weight, other=0.0)

    required_min = (
        required_min_weights.reindex(index).fillna(0.0).astype(float)
        if required_min_weights is not None else pd.Series(0.0, index=index)
    )
    required_max = (
        required_max_weights.reindex(index).fillna(1.0).astype(float)
        if required_max_weights is not None else pd.Series(1.0, index=index)
    )
    if (not np.isfinite(required_min.to_numpy()).all()
            or not np.isfinite(required_max.to_numpy()).all()):
        raise ValueError("required product min/max weights must be finite")
    invalid_required = required_min > required_max
    if invalid_required.any():
        raise ValueError(
            "required product Min exceeds Max for "
            f"{invalid_required.index[invalid_required].tolist()}"
        )
    instructions = mandatory_trades or config.mandatory_trades
    instructed_assets = (
        set(instructions.hard_sell)
        .union(instructions.hard_buy_min)
        .union(instructions.hard_max)
        .union(instructions.hard_target)
    )
    unknown = instructed_assets.difference(index)
    if unknown:
        raise ValueError(f"mandatory instructions reference unknown instruments {sorted(unknown)}")

    instruction_min = pd.Series(0.0, index=index)
    instruction_max = pd.Series(1.0, index=index)
    instruction_type = pd.Series('', index=index, dtype=object)
    instruction_comment = pd.Series('', index=index, dtype=object)
    instruction_source = pd.Series('', index=index, dtype=object)
    if instruction_audit is not None and not instruction_audit.empty:
        audit = instruction_audit.reindex(index)
        for column, target in (
                ('Instruction', instruction_type),
                ('Comment', instruction_comment),
                ('Source', instruction_source)):
            if column in audit.columns:
                target.loc[:] = audit[column].fillna('').astype(str)

    if instructions.hard_sell:
        assets = list(instructions.hard_sell)
        instruction_min.loc[assets] = 0.0
        instruction_max.loc[assets] = 0.0
        instruction_type.loc[assets] = instruction_type.loc[assets].mask(
            instruction_type.loc[assets].eq(''), 'hard_sell'
        )
    paired_bounds = set(instructions.hard_buy_min).intersection(instructions.hard_max)
    for instrument in paired_bounds:
        minimum = instructions.hard_buy_min[instrument]
        maximum = instructions.hard_max[instrument]
        instruction_min.loc[instrument] = minimum
        instruction_max.loc[instrument] = maximum
        instruction_type.loc[instrument] = (
            instruction_type.loc[instrument] or 'hard_range'
        )
    for instrument, minimum in instructions.hard_buy_min.items():
        if instrument in paired_bounds:
            continue
        instruction_min.loc[instrument] = minimum
        instruction_type.loc[instrument] = instruction_type.loc[instrument] or 'hard_buy_min'
    for instrument, maximum in instructions.hard_max.items():
        if instrument in paired_bounds:
            continue
        instruction_max.loc[instrument] = maximum
        instruction_type.loc[instrument] = instruction_type.loc[instrument] or 'hard_max'
    for instrument, target in instructions.hard_target.items():
        instruction_min.loc[instrument] = target
        instruction_max.loc[instrument] = target
        instruction_type.loc[instrument] = instruction_type.loc[instrument] or 'hard_target'

    explicit_fixed = index.isin(
        tuple(instructions.hard_sell) + tuple(instructions.hard_target)
    )
    instruction_flags = pd.DataFrame(
        False, index=index, columns=[ALLOW_CURRENT_PIN, MODEL_MAX_REPAIR])
    if instruction_audit is not None:
        for flag in instruction_flags:
            if flag in instruction_audit:
                instruction_flags[flag] = (
                    instruction_audit[flag].reindex(index).fillna(False).astype(bool))
    fixed_instruction = pd.Series(explicit_fixed, index=index)
    tolerance = float(config.instruction_correction_tolerance)
    fixed_breach = instruction_min.sub(current).abs()
    lower_breach = instruction_min.sub(current).clip(lower=0.0)
    upper_breach = current.sub(instruction_max).clip(lower=0.0)
    deadband_active = tolerance > 0.0
    # Dead-band retention is only valid where the retained current weight is
    # itself product-feasible: product bounds are exact and outside the
    # dead-band, so a current weight breaching them forces the full
    # correction regardless of the instruction breach size. Without this
    # guard a within-tolerance pin at a product-infeasible current weight
    # builds an empty executable interval (2026-08-09 live failure: product
    # Min=Max=2% with a fixed target pinned at current 1.8642%).
    product_feasible_current = (
        current.ge(required_min - WEIGHT_ZERO_TOLERANCE)
        & current.le(required_max + WEIGHT_ZERO_TOLERANCE)
    )
    rounding_tolerance = CURRENT_PIN_ROUNDING_TOLERANCE
    exact_current_pin = (
        fixed_instruction
        & instruction_flags[ALLOW_CURRENT_PIN]
        & ~is_cash
        & instruction_min.sub(instruction_max).abs().le(rounding_tolerance)
        & instruction_min.sub(current).abs().le(rounding_tolerance)
        & instruction_max.sub(current).abs().le(rounding_tolerance)
    )
    product_bound_override = exact_current_pin & ~product_feasible_current
    product_bound_override_breach = pd.concat(
        [
            required_min.sub(current).clip(lower=0.0),
            current.sub(required_max).clip(lower=0.0),
        ],
        axis=1, sort=False).max(axis=1).where(product_bound_override, 0.0)
    product_bound_override_reason = pd.Series('', index=index, dtype=object)
    product_bound_override_reason.loc[product_bound_override] = (
        'exact Current Min/Max pin matches Current Weight within 0.5 bp'
    )
    fixed_within_tolerance = (
        fixed_instruction
        & ~index.isin(instructions.hard_sell)
        & product_feasible_current
        & fixed_breach.gt(WEIGHT_ZERO_TOLERANCE)
        & fixed_breach.le(tolerance)
    ) if deadband_active else pd.Series(False, index=index)
    min_bound_assets = index.isin(tuple(instructions.hard_buy_min))
    max_bound_assets = index.isin(tuple(instructions.hard_max))
    min_within_tolerance = (
        pd.Series(min_bound_assets, index=index)
        & ~fixed_instruction
        & product_feasible_current
        & lower_breach.gt(WEIGHT_ZERO_TOLERANCE)
        & lower_breach.le(tolerance)
    ) if deadband_active else pd.Series(False, index=index)
    max_within_tolerance = (
        pd.Series(max_bound_assets, index=index)
        & ~fixed_instruction
        & product_feasible_current
        & upper_breach.gt(WEIGHT_ZERO_TOLERANCE)
        & upper_breach.le(tolerance)
    ) if deadband_active else pd.Series(False, index=index)
    instruction_within_tolerance = (
        fixed_within_tolerance
        | min_within_tolerance
        | max_within_tolerance
    )
    retained_breach = pd.Series(0.0, index=index)
    retained_breach.loc[fixed_within_tolerance] = fixed_breach.loc[
        fixed_within_tolerance
    ]
    retained_breach.loc[min_within_tolerance] = lower_breach.loc[
        min_within_tolerance
    ]
    retained_breach.loc[max_within_tolerance] = upper_breach.loc[
        max_within_tolerance
    ]

    executable_instruction_min = instruction_min.copy()
    executable_instruction_max = instruction_max.copy()
    executable_instruction_min.loc[fixed_within_tolerance] = current.loc[
        fixed_within_tolerance
    ]
    executable_instruction_max.loc[fixed_within_tolerance] = current.loc[
        fixed_within_tolerance
    ]
    executable_instruction_min.loc[min_within_tolerance] = current.loc[
        min_within_tolerance
    ]
    executable_instruction_max.loc[max_within_tolerance] = current.loc[
        max_within_tolerance
    ]
    executable_instruction_min.loc[product_bound_override] = current.loc[
        product_bound_override
    ]
    executable_instruction_max.loc[product_bound_override] = current.loc[
        product_bound_override
    ]

    executable_product_min = required_min.copy()
    executable_product_max = required_max.copy()
    executable_product_min.loc[product_bound_override] = current.loc[
        product_bound_override
    ]
    executable_product_max.loc[product_bound_override] = current.loc[
        product_bound_override
    ]

    policy_min = pd.concat(
        [executable_product_min, executable_instruction_min], axis=1
    , sort=False).max(axis=1)
    policy_max = pd.concat(
        [executable_product_max, executable_instruction_max], axis=1
    , sort=False).min(axis=1)
    invalid_bounds = policy_min > policy_max + WEIGHT_ZERO_TOLERANCE
    if invalid_bounds.any():
        details = pd.DataFrame({
            'product_min': required_min,
            'product_max': required_max,
            'instruction_min': instruction_min,
            'instruction_max': instruction_max,
            'effective_min': policy_min,
            'effective_max': policy_max,
        }).loc[invalid_bounds]
        raise ValueError(
            'execution instructions do not intersect product bounds:\n'
            f'{details.to_string()}'
        )

    effective = policy_target.clip(lower=policy_min, upper=policy_max)
    pinned_instruction = (
        fixed_instruction
        & (
            (
                instruction_min.sub(current).abs().le(WEIGHT_ZERO_TOLERANCE)
                & instruction_max.sub(current).abs().le(WEIGHT_ZERO_TOLERANCE)
            )
            | fixed_within_tolerance
            | product_bound_override
        )
    )
    instruction_min_buy = (
        current.lt(instruction_min - WEIGHT_ZERO_TOLERANCE)
        & index.isin(tuple(instructions.hard_buy_min) + tuple(instructions.hard_target))
        & ~instruction_within_tolerance
    )
    instruction_max_sell = (
        current.gt(instruction_max + WEIGHT_ZERO_TOLERANCE)
        & index.isin(
            tuple(instructions.hard_sell)
            + tuple(instructions.hard_max)
            + tuple(instructions.hard_target)
        )
        & ~instruction_within_tolerance
    )
    product_min_buy = current.lt(
        executable_product_min - WEIGHT_ZERO_TOLERANCE
    )
    product_max_sell = current.gt(
        executable_product_max + WEIGHT_ZERO_TOLERANCE
    )

    model_entry = (
        model_for_rules.gt(effective_cutoff)
        & previous_model_for_rules.le(effective_cutoff)
        & current_for_rules.eq(0.0)
        & ~pinned_instruction
        & ~is_cash
    )
    model_exit = (
        previous_model_for_rules.gt(effective_cutoff)
        & model_for_rules.le(effective_cutoff)
        & current_for_rules.gt(0.0)
        & ~pinned_instruction
        & ~is_cash
    )
    cutoff_candidate = (
        low_weight
        & current_for_rules.gt(0.0)
        & model_for_rules.gt(WEIGHT_ZERO_TOLERANCE)
        & ~pinned_instruction
    )

    cutoff_sell_down = pd.Series(False, index=index)
    sell_down_reasons = {instrument: set() for instrument in index}
    residual_bridge = pd.Series(0.0, index=index)
    residual_bridge_groups = {instrument: [] for instrument in index}
    group_bridge_diagnostic = pd.DataFrame()
    if group_loadings is not None:
        loadings = group_loadings.reindex(index=index).fillna(0.0).astype(float)
        minimum = (
            group_min.reindex(loadings.columns).astype(float)
            if group_min is not None
            else pd.Series(np.nan, index=loadings.columns)
        )
        maximum = (
            group_max.reindex(loadings.columns).astype(float)
            if group_max is not None
            else pd.Series(np.nan, index=loadings.columns)
        )

        provisional = current.copy()

        def apply_provisional(mask: pd.Series, values: pd.Series) -> None:
            """Apply a provisional cutoff target before capacity diagnosis."""
            active = mask.astype(bool) & ~is_cash
            provisional.loc[active] = values.loc[active]

        apply_provisional(product_min_buy, required_min)
        apply_provisional(product_max_sell, required_max)
        apply_provisional(
            low_weight & policy_min.le(WEIGHT_ZERO_TOLERANCE),
            pd.Series(0.0, index=index),
        )
        apply_provisional(
            model_exit & policy_min.le(WEIGHT_ZERO_TOLERANCE),
            pd.Series(0.0, index=index),
        )
        apply_provisional(model_entry, effective)
        apply_provisional(instruction_min_buy, policy_min)
        apply_provisional(instruction_max_sell, policy_max)
        apply_provisional(fixed_instruction, policy_min)
        provisional_mandatory = provisional.sub(current).abs().gt(
            WEIGHT_ZERO_TOLERANCE
        ) & ~is_cash
        sell_down_eligible = (
            cutoff_candidate
            & policy_min.le(WEIGHT_ZERO_TOLERANCE)
            & ~product_max_sell
            & ~instruction_max_sell
            & ~fixed_instruction
        )

        def diagnose_cutoff_capacity(
                retain_sell_down: bool = False,
        ) -> pd.DataFrame:
            """Measure cutoff capacity against the unchanged group constraints."""
            upper = current.copy()
            upper.loc[provisional_mandatory] = provisional.loc[provisional_mandatory]
            excluded = (
                low_weight
                & policy_min.le(WEIGHT_ZERO_TOLERANCE)
            )
            if retain_sell_down:
                excluded &= ~cutoff_sell_down
            upper.loc[excluded] = 0.0
            return compute_group_bound_bridges(
                lower_bounds=pd.Series(0.0, index=index),
                upper_bounds=upper,
                group_loadings=loadings,
                group_min=minimum,
                group_max=maximum,
            )

        initial_bridges = diagnose_cutoff_capacity()
        breached_min_groups = initial_bridges.index[
            initial_bridges['bridge_min'].gt(WEIGHT_ZERO_TOLERANCE)
        ]
        for group in breached_min_groups:
            affected = (
                sell_down_eligible
                & (loadings[group] * current).gt(WEIGHT_ZERO_TOLERANCE)
            )
            cutoff_sell_down |= affected
            for instrument in index[affected]:
                sell_down_reasons[instrument].add(f'{group}:min')

        group_bridge_diagnostic = diagnose_cutoff_capacity(
            retain_sell_down=True
        )
        remaining = group_bridge_diagnostic.loc[
            group_bridge_diagnostic['bridge_min'].gt(WEIGHT_ZERO_TOLERANCE)
        ]
        for group, row in remaining.iterrows():
            affected = loadings[group].gt(0.0)
            residual_bridge.loc[affected] = np.maximum(
                residual_bridge.loc[affected], float(row['bridge_min'])
            )
            for instrument in index[affected]:
                residual_bridge_groups[instrument].append(f'{group}:min')

    low_weight_excluded = (
        low_weight
        & policy_min.le(WEIGHT_ZERO_TOLERANCE)
        & ~cutoff_sell_down
        & ~pinned_instruction
    )
    cutoff_exit = cutoff_candidate & ~cutoff_sell_down
    model_exit = model_exit & ~cutoff_sell_down

    base = current.copy()
    reasons = pd.Series('', index=index, dtype=object)
    categories = pd.Series('', index=index, dtype=object)

    def apply_base(mask: pd.Series, values: pd.Series, reason: str) -> None:
        """Apply a mandatory target and retain its policy reason."""
        active = mask.astype(bool) & ~is_cash
        base.loc[active] = values.loc[active]
        reasons.loc[active] = reasons.loc[active].where(
            reasons.loc[active].eq(''), reasons.loc[active] + '|'
        ) + reason
        categories.loc[active] = reason

    # Ordinary product-bound repairs establish a compliant starting point.
    # Model lifecycle rules are applied afterwards because a valid model exit
    # to zero or model entry inside the product bounds must not be overwritten
    # by the current portfolio's one-sided boundary repair. Explicit desk and
    # workbook instructions remain last and therefore retain final precedence.
    apply_base(product_min_buy, required_min, 'product_min_buy')
    apply_base(product_max_sell, required_max, 'product_max_sell')
    apply_base(low_weight_excluded,
               pd.Series(0.0, index=index), 'small_cutoff')
    apply_base(model_exit & policy_min.le(WEIGHT_ZERO_TOLERANCE),
               pd.Series(0.0, index=index), 'model_exit')
    apply_base(model_entry, effective, 'model_entry')
    apply_base(instruction_min_buy, policy_min, 'instruction_min_buy')
    instruction_max_target = policy_max.copy()
    desk_current_max_sell = (
        instruction_max_sell & instruction_flags[MODEL_MAX_REPAIR]
    )
    # Current Max breaches go to the model or cap; sub-cutoff model weights go to zero.
    instruction_max_target.loc[desk_current_max_sell] = policy_target.clip(
        lower=policy_min, upper=policy_max
    ).loc[desk_current_max_sell]
    apply_base(
        instruction_max_sell, instruction_max_target, 'instruction_max_sell'
    )
    apply_base(fixed_instruction & ~pinned_instruction,
               policy_min, 'fixed_instruction')

    mandatory = (
        base.sub(current).abs().gt(WEIGHT_ZERO_TOLERANCE) & ~is_cash
    )
    noncash_funding = base.sub(current).where(~is_cash, 0.0)
    mandatory_funding = float(noncash_funding.sum())
    if cash_asset is None and abs(mandatory_funding) > WEIGHT_ZERO_TOLERANCE:
        raise ValueError(
            'mandatory execution funding requires one settlement-cash asset; '
            'pass settlement_cash_asset or provide one unique cash-labelled instrument'
        )
    if cash_asset is not None:
        base.loc[cash_asset] = current.loc[cash_asset] - mandatory_funding
        reasons.loc[cash_asset] = (
            'mandatory_cash_funding'
            if abs(mandatory_funding) > WEIGHT_ZERO_TOLERANCE else ''
        )
        categories.loc[cash_asset] = reasons.loc[cash_asset]
    if not np.isclose(base.sum(), current.sum(), atol=WEIGHT_ZERO_TOLERANCE, rtol=0.0):
        raise RuntimeError(
            'post-mandatory base is not self-financing: '
            f'base_sum={base.sum():.12f}, current_sum={current.sum():.12f}'
        )

    # The effective target is the fully funded policy destination, not merely
    # the independently clipped model vector. Mandatory coordinates remain at
    # their consolidated base targets and settlement cash funds every non-cash
    # coordinate. This makes the target a proper portfolio and supports the
    # execution-vs-target-vs-model implementation-risk decomposition.
    effective.loc[mandatory | cutoff_sell_down] = base.loc[
        mandatory | cutoff_sell_down
    ]
    if cash_asset is not None:
        effective.loc[cash_asset] = current.sum() - effective.loc[~is_cash].sum()
    if not np.isclose(effective.sum(), current.sum(), atol=WEIGHT_ZERO_TOLERANCE, rtol=0.0):
        raise RuntimeError(
            'effective execution target is not self-financing: '
            f'target_sum={effective.sum():.12f}, current_sum={current.sum():.12f}'
        )

    rule4 = (
        model_for_rules.gt(effective_cutoff)
        & ~mandatory
        & ~cutoff_sell_down
        & ~pinned_instruction
        & ~is_cash
    )
    desired_trade = effective - base
    total_trade = desired_trade.where(~mandatory, base - current)
    if cash_asset is not None:
        total_trade.loc[cash_asset] = base.loc[cash_asset] - current.loc[cash_asset]
    material = desired_trade.abs() >= config.material_trade_threshold
    candidate = rule4 & allowed & material
    entry_funding = noncash_funding.where(categories.eq('model_entry'), 0.0)
    exit_funding = noncash_funding.where(
        categories.isin(['small_cutoff', 'model_exit']), 0.0
    )
    desk_funding = noncash_funding.where(
        categories.isin([
            'instruction_min_buy', 'instruction_max_sell', 'fixed_instruction'
        ]), 0.0
    )
    other_funding = noncash_funding - entry_funding - exit_funding - desk_funding

    return pd.DataFrame({
        ALLOW_CURRENT_PIN: instruction_flags[ALLOW_CURRENT_PIN],
        MODEL_MAX_REPAIR: instruction_flags[MODEL_MAX_REPAIR],
        CURRENT_WEIGHT: current,
        PREVIOUS_MODEL_WEIGHT: previous_model,
        RAW_MODEL_WEIGHT: model,
        POLICY_TARGET_WEIGHT: policy_target,
        EFFECTIVE_SMALL_TRADE_CUTOFF: effective_cutoff,
        EFFECTIVE_MODEL_WEIGHT: effective,
        BASE_WEIGHT: base,
        TRADE_WEIGHT: total_trade,
        LOW_WEIGHT_EXCLUDED: low_weight_excluded,
        CUTOFF_SELL_DOWN: cutoff_sell_down,
        CUTOFF_SELL_DOWN_REASON: pd.Series({
            instrument: '|'.join(sorted(sell_down_reasons[instrument]))
            for instrument in index
        }),
        CUTOFF_SELL_DOWN_RETENTION: np.nan,
        CUTOFF_SELL_DOWN_RETAINED_FRACTION: np.nan,
        CUTOFF_SELL_DOWN_DUST_ROUNDED: False,
        SELL_DOWN_DUST_THRESHOLD: config.sell_down_dust_threshold,
        CUTOFF_RESIDUAL_BRIDGE: residual_bridge,
        CUTOFF_RESIDUAL_BRIDGE_BP: 10_000.0 * residual_bridge,
        CUTOFF_RESIDUAL_BRIDGE_GROUP: pd.Series({
            instrument: '|'.join(sorted(set(residual_bridge_groups[instrument])))
            for instrument in index
        }),
        MODEL_ENTRY: model_entry,
        MODEL_EXIT: model_exit,
        CUTOFF_EXIT: cutoff_exit,
        INSTRUCTION_MIN_BUY: instruction_min_buy,
        INSTRUCTION_MAX_SELL: instruction_max_sell,
        EXECUTION_INSTRUCTION: instruction_type,
        INSTRUCTION_COMMENT: instruction_comment,
        INSTRUCTION_SOURCE: instruction_source,
        INSTRUCTION_WITHIN_TOLERANCE: instruction_within_tolerance,
        INSTRUCTION_RETAINED_BREACH: retained_breach,
        INSTRUCTION_MIN_WEIGHT: instruction_min,
        INSTRUCTION_MAX_WEIGHT: instruction_max,
        FIXED_INSTRUCTION: fixed_instruction,
        PINNED_INSTRUCTION: pinned_instruction,
        MANDATORY_REASON: reasons,
        MANDATORY_CATEGORY: categories,
        MANDATORY_FUNDING: noncash_funding.where(~is_cash, -mandatory_funding),
        ENTRY_FUNDING: entry_funding,
        EXIT_FUNDING: exit_funding,
        DESK_FUNDING: desk_funding,
        OTHER_FUNDING: other_funding,
        SETTLEMENT_CASH: is_cash,
        RULE4_TRADE: rule4,
        DESIRED_REWEIGHT: desired_trade.where(rule4, 0.0),
        MANDATORY_TRADE: mandatory,
        REBALANCE_CADENCE_ELIGIBLE: allowed,
        MATERIAL_TRADE: material,
        TRADE_CANDIDATE: candidate,
        POLICY_MIN_WEIGHT: policy_min,
        POLICY_MAX_WEIGHT: policy_max,
        PRODUCT_MIN_WEIGHT: required_min,
        PRODUCT_MAX_WEIGHT: required_max,
        PRODUCT_BOUND_OVERRIDE: product_bound_override,
        PRODUCT_BOUND_OVERRIDE_BREACH: product_bound_override_breach,
        PRODUCT_BOUND_OVERRIDE_REASON: product_bound_override_reason,
    })
