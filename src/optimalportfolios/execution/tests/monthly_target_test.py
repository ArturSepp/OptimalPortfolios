"""Independent small-book checks for the instruction-free monthly lifecycle."""
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from optimalportfolios import Constraints, GroupLowerUpperConstraints
from optimalportfolios.execution import schema as s
from optimalportfolios.execution._monthly_target import _build_monthly_target


def _case():
    """Return a hand-checkable book with one buy, one cutoff sale and one reweight."""
    idx = pd.Index(['cash', 'entry', 'exit', 'held'])
    current = pd.Series([.2, 0., .1, .7], index=idx)
    model = pd.Series([.2, .1, .004, .696], index=idx)
    previous = pd.Series([.2, 0., .1, .7], index=idx)
    constraints = Constraints(min_weights=pd.Series(0., index=idx),
                              max_weights=pd.Series(1., index=idx))
    cutoffs = pd.Series([0., .005, .005, .005], index=idx)
    return dict(current=current, model=model, previous_model=previous,
                constraints=constraints, cutoffs=cutoffs, cash_asset='cash')


def test_entry_exit_cash_funding_and_material_reweight():
    """Lifecycle trades net to zero cash; effective cash also funds the reweight."""
    target = _build_monthly_target(**_case())
    np.testing.assert_allclose(target[s.BASE_WEIGHT], [.2, .1, 0., .7], atol=1e-15)
    np.testing.assert_allclose(target[s.EFFECTIVE_MODEL_WEIGHT], [.204, .1, 0., .696])
    assert target[s.MANDATORY_TRADE].tolist() == [False, True, True, False]
    assert target[s.TRADE_CANDIDATE].tolist() == [False, False, False, True]
    assert target.loc['exit', s.MANDATORY_REASON] == 'small_cutoff|model_exit'
    assert target.loc['entry', s.ENTRY_FUNDING] == .1
    assert target.loc['exit', s.EXIT_FUNDING] == -.1
    assert target[s.MANDATORY_FUNDING].sum() == 0.


def test_fractional_group_retention_and_disabled_fallback():
    """A fractional class floor preserves cutoff capacity only in the first attempt."""
    case = _case()
    loading = pd.DataFrame({'Alternatives': [0., 0., .6, 0.]}, index=case['current'].index)
    group = GroupLowerUpperConstraints(group_loadings=loading,
        group_min_allocation=pd.Series({'Alternatives': .05}),
        group_max_allocation=pd.Series({'Alternatives': .15}))
    case['constraints'] = replace(case['constraints'], group_lower_upper_constraints=group)
    target = _build_monthly_target(**case)
    assert target.loc['exit', s.CUTOFF_SELL_DOWN]
    assert not target.loc['exit', s.MANDATORY_TRADE]
    assert not target.loc['exit', s.TRADE_CANDIDATE]
    assert target.loc['exit', s.BASE_WEIGHT] == .1
    assert target.loc['cash', s.BASE_WEIGHT] == .1
    assert target.loc['exit', s.CUTOFF_SELL_DOWN_REASON] == 'Alternatives:min'
    # The legacy diagnostic keeps the provisional exit pinned at zero. Retention
    # changes the solver corridor, not this recorded pre-solve capacity bridge.
    assert target.loc['exit', s.CUTOFF_RESIDUAL_BRIDGE] == .05
    disabled = _build_monthly_target(**case, group_enabled=False)
    assert not disabled[s.CUTOFF_SELL_DOWN].any()
    assert disabled.loc['exit', s.BASE_WEIGHT] == 0.
    # Changing the floor changes that provisional diagnostic, not the retention pin.
    group.group_min_allocation.loc['Alternatives'] = .07
    residual = _build_monthly_target(**case)
    assert residual.loc['exit', s.CUTOFF_RESIDUAL_BRIDGE] == pytest.approx(.07)
    assert residual.loc['exit', s.CUTOFF_RESIDUAL_BRIDGE_GROUP] == 'Alternatives:min'


def test_product_cap_and_zero_model_tolerance():
    """Dated zero caps force unavailable holdings out; tolerance suppresses fake entries."""
    case = _case()
    case['constraints'].max_weights.loc['exit'] = 0.
    case['model'].loc['exit'] = 1e-9
    case['cutoffs'].loc['exit'] = 0.
    target = _build_monthly_target(**case)
    assert target.loc['exit', s.BASE_WEIGHT] == 0.
    assert target.loc['exit', s.POLICY_MAX_WEIGHT] == 0.
    assert not target.loc['exit', s.RULE4_TRADE]
    case['constraints'] = Constraints()
    default = _build_monthly_target(**case)
    assert default[s.PRODUCT_MIN_WEIGHT].eq(0.).all()
    assert default[s.PRODUCT_MAX_WEIGHT].eq(1.).all()


def test_funding_audit_is_enforced(monkeypatch):
    """A failed conservation check cannot return a target to the solver."""
    monkeypatch.setattr(np, 'isclose', lambda *args, **kwargs: False)
    with pytest.raises(RuntimeError, match='not self-financing'):
        _build_monthly_target(**_case())


@pytest.mark.parametrize('fault', ['cash', 'duplicate', 'nan', 'cutoff', 'bounds', 'long_only'])
def test_invalid_lifecycle_inputs_fail_closed(fault):
    """Reject malformed or unsupported numeric state before constructing trades."""
    case = _case()
    if fault == 'cash':
        case['cash_asset'] = 'absent'
    elif fault == 'duplicate':
        case['current'].index = ['cash', 'entry', 'exit', 'exit']
    elif fault == 'nan':
        case['model'].iloc[1] = np.nan
    elif fault == 'cutoff':
        case['cutoffs'].iloc[1] = -.1
    elif fault == 'bounds':
        case['constraints'].min_weights.iloc[1] = 2.
    else:
        case['constraints'].min_weights.iloc[1] = 1.1
        case['constraints'].max_weights.iloc[1] = 1.2
    with pytest.raises(ValueError):
        _build_monthly_target(**case)
