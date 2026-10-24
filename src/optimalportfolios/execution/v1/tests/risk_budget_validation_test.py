"""Target-band rejection and persisted-diagnostic reconstruction contracts."""

from dataclasses import replace
import numpy as np
import pandas as pd
import pytest
from optimalportfolios.execution.v1 import risk_budget_ranker as r
from optimalportfolios.execution.v1.tests.risk_budget_contract_test import (
    _trade_table,
    _config,
    ASSET_CLASSES,
)


def inputs():
    """Return a complete scaffold and externally supplied class budgets."""
    table, covariance, groups = _trade_table()
    return dict(
        scaffold=table,
        alphas=pd.Series(0.0, index=table.index),
        covariance=covariance,
        asset_classes=groups,
        asset_class_tre_weights=pd.Series(1.0, index=ASSET_CLASSES),
        config=_config(),
    )


@pytest.mark.parametrize(
    "case,match",
    [
        ("column", "missing columns"),
        ("duplicate", "unique"),
        ("covar_missing", "missing instruments"),
        ("covar_nan", "finite"),
        ("covar_asymmetric", "symmetric"),
        ("class_missing", "missing asset classes"),
        ("class_unknown", "unsupported"),
        ("cash", "exactly one"),
        ("nonfinite", "finite"),
        ("bounds", "do not intersect"),
        ("alpha_inf", "finite or missing"),
        ("alpha_large", "alpha in"),
        ("missing_budgets", "requires"),
        ("bad_budgets", "sum positive"),
        ("corrupt_band", "target band is empty"),
        ("bad_funding", "self-financing"),
    ],
)
def test_invalid_target_band_inputs(case, match):
    """Reject malformed supplied arrays, corrupted bands and unauditable funding."""
    a = inputs()
    t = a["scaffold"]
    first = t.index[0]
    if case == "column":
        a["scaffold"] = t.drop(columns=[r.CURRENT_WEIGHT])
    elif case == "duplicate":
        a["scaffold"] = pd.concat([t, t.iloc[:1]])
    elif case == "covar_missing":
        a["covariance"] = a["covariance"].drop(index=first)
    elif case == "covar_nan":
        a["covariance"].iloc[0, 0] = np.nan
    elif case == "covar_asymmetric":
        a["covariance"].iloc[0, 1] = 1
    elif case == "class_missing":
        a["asset_classes"] = a["asset_classes"].drop(index=first)
    elif case == "class_unknown":
        a["asset_classes"].iloc[0] = "Unknown"
    elif case == "cash":
        t[r.SETTLEMENT_CASH] = False
    elif case == "nonfinite":
        t.loc[first, r.CURRENT_WEIGHT] = np.inf
    elif case == "bounds":
        t.loc[first, r.INSTRUCTION_MIN_WEIGHT] = 1.1
    elif case == "alpha_inf":
        a["alphas"].iloc[0] = np.inf
    elif case == "alpha_large":
        a["alphas"].iloc[0] = 2
    elif case == "missing_budgets":
        a["asset_class_tre_weights"] = None
    elif case == "bad_budgets":
        a["asset_class_tre_weights"] *= 0
    elif case == "corrupt_band":
        object.__setattr__(
            a["config"], "target_band_wide_by_asset_class", {k: -0.1 for k in ASSET_CLASSES}
        )
    else:
        t.loc[first, r.CURRENT_WEIGHT] = 1e16
        t.loc["Cash", r.CURRENT_WEIGHT] = -1e16
    with pytest.raises((ValueError, RuntimeError), match=match):
        r.build_risk_budget_ranked_trade_table(**a)


def test_negative_cash_funded_variance_is_rejected():
    """A covariance with a negative variance cannot produce a zero-risk diagnostic."""
    with pytest.raises(ValueError, match="negative"):
        r._cash_funded_group_tre(
            pd.Series([0.5, 0.5]), pd.Series([0.1, 0.9]), -np.eye(2), np.array([True, False]), 1
        )


def test_ticket_budget_truncates_global_prefixes():
    """The global ticket cap keeps the highest-ranked eligible trade across classes."""
    a = inputs()
    a["config"] = replace(
        a["config"],
        max_trades=1,
        execution_tre_budget=1e-8,
        add_mandatory_trades_to_max_trades=True,
    )
    table = r.build_risk_budget_ranked_trade_table(**a)
    assert table[r.RISK_BUDGET_SELECTED].sum() == 1
    assert table[r.GROUP_SELECTION_STATUS].str.contains("ticket cap").any()


def test_summary_reconstruction_and_declared_class_retention():
    """Persisted diagnostics retain empty declared classes and rebuild surviving rows."""
    table, covar, _ = _trade_table()
    expected = r.build_risk_budget_group_summary(table)
    historical = table.copy()
    historical.attrs.pop("execution_asset_classes")
    assert r._declared_classes(historical) == tuple(expected.index)
    without_attrs = table.copy()
    without_attrs.attrs = {}
    rebuilt = r.build_risk_budget_group_summary(without_attrs)
    assert set(rebuilt.index) == set(table[r.ASSET_CLASS])
    assert r._declared_classes(without_attrs) == tuple(dict.fromkeys(table[r.ASSET_CLASS]))
    without_attrs.attrs["execution_asset_classes"] = tuple(ASSET_CLASSES) + ("Empty",)
    assert "Empty" not in r.build_risk_budget_group_summary(without_attrs).index
    bad = table.copy()
    bad.attrs["execution_asset_classes"] = tuple(ASSET_CLASSES) + ("Missing",)
    with pytest.raises(ValueError, match="missing asset classes"):
        r.build_risk_budget_group_summary(bad)
    with pytest.raises(ValueError, match="missing asset classes"):
        r.build_risk_budget_prefix_curve(bad)
    missing = table.copy()
    missing.attrs = {}
    with pytest.raises(ValueError, match="unavailable"):
        r.build_risk_budget_prefix_curve(missing)
    with pytest.raises(ValueError, match="missing columns"):
        r.build_risk_budget_group_summary(missing.drop(columns=[r.TRADE_RANK]))
    rescued = table.copy()
    rescued.loc[rescued.index[0], r.FEASIBILITY_RESCUE_TRADE] = True
    final = r.finalize_risk_budget_diagnostics(rescued, rescued[r.BASE_WEIGHT], covar)
    assert final[r.GROUP_SELECTION_STATUS].str.contains("hard-feasibility expansion").any()
    for has_stored in (True, False):
        cut = table.loc[~table[r.ASSET_CLASS].eq("Alternatives")].copy()
        if not has_stored:
            cut.attrs.pop(r.RISK_BUDGET_GROUP_SUMMARY_ATTR)
        r.finalize_risk_budget_diagnostics(cut, cut[r.BASE_WEIGHT], covar)
    broken = table.copy()
    broken[r.SETTLEMENT_CASH] = False
    with pytest.raises(ValueError, match="one settlement-cash"):
        r.finalize_risk_budget_diagnostics(broken, broken[r.BASE_WEIGHT], covar)
    missing = table.drop(columns=[r.TRADE_RANK]).copy()
    missing.attrs.pop(r.RISK_BUDGET_GROUP_SUMMARY_ATTR)
    r.finalize_risk_budget_diagnostics(missing, missing[r.BASE_WEIGHT], covar)
