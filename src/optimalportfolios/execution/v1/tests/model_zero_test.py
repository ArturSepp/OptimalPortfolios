"""Numerical model dust must not change discrete execution eligibility."""
import numpy as np
import pandas as pd
import pytest

from optimalportfolios import Constraints
from optimalportfolios.execution.v1.solver import (
    _audit_exact_cash_bootstrap_model,
)

from optimalportfolios.execution.v1.policy import (
    RiskBudgetRankedExecutionPolicyConfig,
)
from optimalportfolios.execution.MIQP.solver import (
    _prepare_exact_problem,
)
from optimalportfolios.execution.v1.policy import (
    BASE_WEIGHT, CUTOFF_SELL_DOWN, CURRENT_WEIGHT,
    MANDATORY_TRADE, MODEL_ENTRY, MODEL_EXIT, POLICY_MIN_WEIGHT,
    POLICY_MAX_WEIGHT, POLICY_TARGET_WEIGHT, RAW_MODEL_WEIGHT, RULE4_TRADE,
    SELECTED_TRADE, ExactExecutionConfig, ExecutionPolicyConfig,
    MandatoryTrades, build_execution_target, score_execution_trades,
)
from optimalportfolios.execution.v1.risk_budget_ranker import (
    build_risk_budget_ranked_trade_table,
)


ASSET_CLASSES = ('Liquidity', 'Fixed Income', 'Equity', 'Alternatives')


RESIDUALS = [9.998639980515303e-9, 1.0648363724444109e-8,
             2.579430410724267e-8, 4e-7, 1e-6, -4e-7]
ASSETS = pd.Index(['Dust', 'Increase', 'Cash'])


def _target(residual, asset_class='Fixed Income', current_dust=0.006, **kwargs):
    """Build a held exit and an ordinary ranked increase."""
    config = kwargs.pop('config', ExecutionPolicyConfig(
        tre_weight_by_asset_class={key: 1.0 for key in ASSET_CLASSES},
        small_trade_cutoff_by_asset_class={
            key: 0.0049 if key in ('Equity', 'Alternatives') else 0.0
            for key in ASSET_CLASSES},
        max_trades=1, add_mandatory_trades_to_max_trades=True,
    ))
    current = pd.Series([current_dust, 0.30, 0.70 - current_dust], index=ASSETS)
    model = pd.Series([residual, 0.40, 0.60], index=ASSETS)
    previous = kwargs.pop('previous_model_weights', current)
    classes = pd.Series([asset_class, 'Equity', 'Liquidity'], index=ASSETS)
    target = build_execution_target(
        current_weights=current, model_weights=model,
        previous_model_weights=previous, config=config,
        settlement_cash_asset='Cash', asset_classes=classes, **kwargs,
    )
    return target, classes, config


def _score(target, classes, config):
    """Run the real ranking with a deterministic diagonal covariance."""
    return score_execution_trades(
        target=target, alphas=pd.Series(0.0, index=ASSETS),
        covariance=pd.DataFrame(np.eye(3) * 0.04, index=ASSETS, columns=ASSETS),
        asset_classes=classes, config=config,
    )


@pytest.mark.parametrize('residual', RESIDUALS)
def test_model_dust_exits_without_consuming_discretionary_ticket(residual):
    """Observed IMBS residuals have the same exit and selection as exact zero."""
    target, classes, config = _target(residual)
    scored = _score(target, classes, config)
    assert target.at['Dust', POLICY_TARGET_WEIGHT] == 0.0
    assert target.at['Dust', MODEL_EXIT]
    assert target.at['Dust', MANDATORY_TRADE]
    assert not target.at['Dust', RULE4_TRADE]
    # Independent reference: liquidate the held 0.6%, funded into cash.
    np.testing.assert_allclose(target[BASE_WEIGHT], [0.0, 0.30, 0.70],
                               atol=1e-15, rtol=0.0)
    assert scored.index[scored[SELECTED_TRADE]].tolist() == ['Increase']
    assert target.at['Dust', RAW_MODEL_WEIGHT] == residual


@pytest.mark.parametrize('residual', RESIDUALS)
def test_model_dust_cannot_become_group_minimum_sell_down_capacity(residual):
    """EEM-like residuals cannot reopen a zero-model sell-down corridor."""
    target, _, _ = _target(
        residual, asset_class='Equity',
        group_loadings=pd.DataFrame({'Asia': [1.0, 0.0, 0.0]}, index=ASSETS),
        group_min=pd.Series({'Asia': 0.003}),
    )
    assert not target.at['Dust', CUTOFF_SELL_DOWN]
    assert target.at['Dust', BASE_WEIGHT] == 0.0
    assert target.at['Dust', MANDATORY_TRADE]


@pytest.mark.parametrize('residual', RESIDUALS)
def test_previous_model_dust_is_zero_for_a_genuine_entry(residual):
    """Previous solver noise must not suppress a new fixed-income entry."""
    previous = pd.Series([residual, 0.30, 0.70], index=ASSETS)
    target, _, _ = _target(0.02, current_dust=0.0,
                           previous_model_weights=previous)
    assert target.at['Dust', MODEL_ENTRY]
    assert target.at['Dust', BASE_WEIGHT] == 0.02


def test_real_small_model_allocation_and_explicit_small_instruction_survive():
    """The numerical floor leaves real model targets and desk bounds intact."""
    target, _, _ = _target(2e-6, current_dust=0.0)
    assert target.at['Dust', POLICY_TARGET_WEIGHT] == 2e-6
    assert target.at['Dust', MODEL_ENTRY]
    instructed, _, _ = _target(
        4e-7, current_dust=2e-7,
        mandatory_trades=MandatoryTrades(hard_target={'Dust': 5e-7}),
    )
    assert instructed.at['Dust', CURRENT_WEIGHT] == 2e-7
    assert instructed.at['Dust', POLICY_MIN_WEIGHT] == 5e-7
    assert instructed.at['Dust', POLICY_MAX_WEIGHT] == 5e-7
    assert instructed.at['Dust', BASE_WEIGHT] == 5e-7
    assert instructed.at['Dust', MANDATORY_TRADE]


def test_risk_budget_policy_uses_the_same_model_zero_convention():
    """The alternate ranker must not reconstruct presence from raw dust."""
    config = RiskBudgetRankedExecutionPolicyConfig(
        tre_weight_by_asset_class={key: 1.0 for key in ASSET_CLASSES},
        small_trade_cutoff_by_asset_class={
            key: 0.0049 if key in ('Equity', 'Alternatives') else 0.0
            for key in ASSET_CLASSES},
        execution_tre_budget=0.005,
        target_band_tight_by_asset_class={key: 0.005 for key in ASSET_CLASSES},
        target_band_wide_by_asset_class={key: 0.02 for key in ASSET_CLASSES},
    )
    target, classes, _ = _target(4e-7, config=config)
    table = build_risk_budget_ranked_trade_table(
        scaffold=target, alphas=pd.Series(0.0, index=ASSETS),
        covariance=pd.DataFrame(np.eye(3) * 0.04, index=ASSETS, columns=ASSETS),
        asset_classes=classes, asset_class_tre_weights=pd.Series(1.0, index=ASSET_CLASSES),
        config=config,
    )
    assert table.at['Dust', MODEL_EXIT]
    assert table.at['Dust', BASE_WEIGHT] == 0.0
    assert table.at['Dust', RAW_MODEL_WEIGHT] == 4e-7


def test_exact_policy_does_not_admit_unheld_model_dust():
    """Exact support excludes dust but retains a tiny actual holding."""
    config = ExecutionPolicyConfig(
        tre_weight_by_asset_class={key: 1.0 for key in ASSET_CLASSES},
        small_trade_cutoff_by_asset_class={
            key: 0.0049 if key in ('Equity', 'Alternatives') else 0.0
            for key in ASSET_CLASSES},
        max_trades=2, exact_execution=ExactExecutionConfig(max_positions=2),
    )
    for current_dust, expected_frozen in [(0.0, True), (4e-7, False)]:
        target, classes, _ = _target(4e-7, current_dust=current_dust, config=config)
        table = _score(target, classes, config)
        problem = _prepare_exact_problem(table, config)
        assert ('Dust' in problem.index[problem.frozen]) == expected_frozen


def test_cash_bootstrap_audits_model_dust_without_relaxing_product_bounds():
    """A pinned zero may absorb model dust only when original bounds allow it."""
    target, _, _ = _target(4e-7, current_dust=0.0)
    target[CURRENT_WEIGHT] = [0.0, 0.0, 1.0]
    target[RAW_MODEL_WEIGHT] = [4e-7, 0.4, 0.5999996]
    product = Constraints(
        min_weights=pd.Series(0.0, index=ASSETS),
        max_weights=pd.Series(1.0, index=ASSETS),
    )
    selected = product.copy(
        min_weights=pd.Series([0.0, 0.4, 0.0], index=ASSETS),
        max_weights=pd.Series([0.0, 0.4, 1.0], index=ASSETS),
    )
    covariance = pd.DataFrame(np.eye(3), index=ASSETS, columns=ASSETS)
    outcome = _audit_exact_cash_bootstrap_model(
        target, product, selected, covariance, 'model-dust-regression',
    )
    assert outcome is not None and outcome.accepted
    np.testing.assert_allclose(outcome.weights, [0.0, 0.4, 0.6], atol=1e-15, rtol=0.0)
    strict_minimum = product.min_weights.copy()
    strict_minimum['Dust'] = 2e-6
    rejected = _audit_exact_cash_bootstrap_model(
        target, product.copy(min_weights=strict_minimum), selected,
        covariance, 'model-dust-product-bound-regression',
    )
    assert rejected is None
