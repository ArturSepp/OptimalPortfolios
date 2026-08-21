"""Independently validate the persisted U1 production-momentum covariance grid."""
from __future__ import annotations

import pandas as pd

from papers.cluster_lineage_2026.replication.run_u1_covar_grid_long_short_prod import (
    AVAILABLE_WINDOW,
    EXACT_VARIANT,
    HEADLINE_WINDOW,
    LONG_SPANS,
    MIN_CLUSTER_SIZE,
    SCALED_VARIANT,
    VARIANTS,
    _cells,
    _root,
)


EXPECTED_CELLS = 28
EXPECTED_COMPARISONS = EXPECTED_CELLS * len(VARIANTS) * 2
EXPECTED_GLOBAL_ROWS = (1 + len(LONG_SPANS)) * 2
EXPECTED_PERFORMANCE_ROWS = EXPECTED_COMPARISONS + EXPECTED_GLOBAL_ROWS
EXPOSURE_TOLERANCE = 1e-12


def _read(name: str) -> pd.DataFrame:
    """Read one persisted round-trip CSV artifact."""
    return pd.read_csv(_root() / f"{name}.csv", float_precision="round_trip")


def validate() -> None:
    """Assert construction, horizon, matching, regression, and replay contracts."""
    parameters = _read("signal_parameters")
    performance = _read("performance")
    comparison = _read("comparison_vs_global")
    other = _read("comparison_vs_other_signals")
    breadth = _read("signal_breadth_summary")
    risk = _read("risk_diagnostics")
    scores = _read("score_diagnostics")
    regression = _read("regression")
    acceptance = _read("acceptance")
    determinism = _read("determinism")

    assert len(parameters) == len(VARIANTS) * len(LONG_SPANS)
    assert len(performance) == EXPECTED_PERFORMANCE_ROWS
    assert len(comparison) == EXPECTED_COMPARISONS
    assert len(other) == EXPECTED_CELLS
    assert len(breadth) == 4
    assert len(risk) == EXPECTED_COMPARISONS
    assert len(scores) == EXPECTED_GLOBAL_ROWS
    assert len(acceptance) == EXPECTED_PERFORMANCE_ROWS
    assert set(comparison["analysis_window"]) == {HEADLINE_WINDOW, AVAILABLE_WINDOW}
    assert set(comparison["signal_variant"]) == set(VARIANTS)
    assert set(zip(other["frequency"], other["span"])) == set(_cells())

    scaled = parameters.loc[parameters["signal_variant"].eq(SCALED_VARIANT)]
    exact = parameters.loc[parameters["signal_variant"].eq(EXACT_VARIANT)]
    assert scaled["long_horizon_months"].eq(12.0).all()
    assert exact["signal_frequency"].eq("ME").all()
    assert exact["long_span"].eq(12).all()
    assert exact["vol_span"].eq(13).all()
    assert parameters["min_cluster_size"].eq(MIN_CLUSTER_SIZE).all()
    assert parameters["short_span"].isna().all()
    assert parameters["mean_adj_type"].eq("NONE").all()

    assert regression["status"].eq("PASS").all()
    signal_checks = regression.loc[
        regression["check"].eq("production_signal_preflight")
    ]
    assert len(signal_checks) == len(LONG_SPANS)
    assert signal_checks["long_horizon_months"].eq(12.0).all()
    assert signal_checks["max_signal_lookahead_days"].le(0.0).all()
    assert (
        signal_checks["return_roundtrip_max_abs_error"]
        <= signal_checks["return_roundtrip_tolerance"]
    ).all()
    me_check = regression.loc[
        regression["check"].eq("exact_monthly_equals_scaled_ME")
    ].iloc[0]
    assert float(me_check["measured"]) <= float(me_check["tolerance"])

    assert acceptance["status"].eq("PASS").all()
    error_columns = [
        "max_long_exposure_error",
        "max_short_exposure_error",
        "max_net_exposure_error",
        "max_gross_exposure_error",
        "max_post_net_group_l1_exposure",
    ]
    max_exposure_error = float(
        acceptance[error_columns].to_numpy(dtype=float).max()
    )
    assert max_exposure_error <= EXPOSURE_TOLERANCE
    assert determinism["byte_identical"].astype(bool).all()

    globals_frame = performance.loc[performance["leg"].str.startswith("global_")]
    global_keys = globals_frame[
        ["analysis_window", "signal_variant", "signal_frequency", "leg"]
    ].drop_duplicates()
    joined = comparison.merge(
        global_keys,
        on=["analysis_window", "signal_variant", "signal_frequency"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_expected"),
    )
    assert joined["leg"].notna().all()
    assert joined["benchmark_leg"].eq(joined["leg"]).all()
    assert comparison["benchmark_leg"].str.startswith("global_").all()

    headline = comparison.loc[comparison["analysis_window"].eq(HEADLINE_WINDOW)]
    calculated = (
        headline.groupby("signal_variant", sort=False)
        .agg(
            return_wins=("beats_global_net_return", "sum"),
            mean_variance_wins=("mean_variance_dominates_global", "sum"),
        )
        .astype(int)
    )
    summary_index = breadth.set_index("signal")
    for variant, prefix in (
        (EXACT_VARIANT, "prod_exact"),
        (SCALED_VARIANT, "prod_scaled"),
    ):
        assert int(calculated.loc[variant, "return_wins"]) == int(
            summary_index.loc[prefix, "return_wins"]
        )
        assert int(calculated.loc[variant, "mean_variance_wins"]) == int(
            summary_index.loc[prefix, "mean_variance_wins"]
        )

    winners = headline.sort_values(
        ["signal_variant", "delta_net_return_annualized"],
        ascending=[True, False],
    ).groupby("signal_variant", sort=False).head(1)
    exact_wins = int(calculated.loc[EXACT_VARIANT, "return_wins"])
    scaled_wins = int(calculated.loc[SCALED_VARIANT, "return_wins"])
    maximum_roundtrip = float(
        signal_checks["return_roundtrip_max_abs_error"].max()
    )
    print("U1 production-momentum covariance long-short grid validation: PASS")
    print(
        f"grid: {EXPECTED_CELLS} cells x {len(VARIANTS)} production variants "
        f"x 2 windows = {EXPECTED_COMPARISONS} comparisons"
    )
    print(
        "horizons: scaled B=252, weekly=52, ME=12; "
        "exact control=ME 12/13; all long horizons=12 months"
    )
    print(
        f"causality: maximum lookahead=0 days; "
        f"return roundtrip max error={maximum_roundtrip:.3e}"
    )
    print(
        f"exposures: {len(acceptance)}/{len(acceptance)} PASS; "
        f"max error={max_exposure_error:.3e}"
    )
    print(
        f"headline global-relative return wins: exact={exact_wins}/"
        f"{EXPECTED_CELLS}; scaled={scaled_wins}/{EXPECTED_CELLS}"
    )
    for _, winner in winners.iterrows():
        print(
            f"winner {winner['signal_variant']}: {winner['frequency']} "
            f"span {int(winner['span'])}; "
            f"delta={float(winner['delta_net_return_annualized']):+.6f}"
        )
    print(
        f"determinism: {len(determinism)}/{len(determinism)} "
        "artifacts byte-identical"
    )


if __name__ == "__main__":
    validate()
