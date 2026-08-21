"""Validate the persisted U1 monthly 12-minus-1 covariance long-short grid."""
from __future__ import annotations

import pandas as pd

from papers.cluster_lineage_2026.replication.run_u1_covar_grid import FREQUENCY_SPANS
from papers.cluster_lineage_2026.replication.run_u1_covar_grid_long_short_monthly import (
    AVAILABLE_WINDOW,
    HEADLINE_WINDOW,
    LOOKBACK_MONTHS,
    Q,
    SIGNAL_VARIANT,
    SKIP_MONTHS,
    _root,
)


EXPECTED_CELLS = sum(len(spans) for spans in FREQUENCY_SPANS.values())
EXPECTED_CLUSTER_ROWS = EXPECTED_CELLS * 2


def _read(name: str) -> pd.DataFrame:
    """Read one deterministic monthly-signal artifact."""
    return pd.read_csv(_root() / f"{name}.csv", float_precision="round_trip")


def validate() -> None:
    """Assert signal, coverage, exposure, comparison, and replay acceptance."""
    performance = _read("performance")
    comparison = _read("comparison_vs_global")
    weekly_comparison = _read("comparison_vs_weekly_signal")
    rankings = _read("rankings")
    risk = _read("risk_diagnostics")
    acceptance = _read("acceptance")
    signal_regression = _read("signal_regression")
    determinism = _read("determinism")

    assert EXPECTED_CELLS == 28
    assert len(performance) == EXPECTED_CLUSTER_ROWS + 2
    assert len(comparison) == EXPECTED_CLUSTER_ROWS
    assert len(weekly_comparison) == EXPECTED_CELLS
    assert len(rankings) == EXPECTED_CLUSTER_ROWS
    assert len(risk) == EXPECTED_CLUSTER_ROWS
    assert len(acceptance) == EXPECTED_CLUSTER_ROWS + 2
    assert performance["q"].eq(Q).all()
    assert performance["signal_variant"].eq(SIGNAL_VARIANT).all()
    assert performance["momentum_lookback_periods"].eq(LOOKBACK_MONTHS).all()
    assert performance["momentum_skip_periods"].eq(SKIP_MONTHS).all()
    assert comparison["benchmark_leg"].eq("global").all()
    assert acceptance["status"].eq("PASS").all()
    assert signal_regression["status"].eq("PASS").all()
    assert signal_regression["nan_mask_match"].astype(bool).all()
    assert determinism["byte_identical"].astype(bool).all()
    assert risk["cluster_max_group_l1_net_exposure"].max() <= 1e-12
    assert set(performance["analysis_window"]) == {HEADLINE_WINDOW, AVAILABLE_WINDOW}
    assert signal_regression["warmup_empty_dates"].eq(10).all()
    assert pd.to_datetime(signal_regression["first_available_date"]).eq(
        pd.Timestamp("2007-08-31")
    ).all()

    headline = comparison.loc[
        comparison["analysis_window"].eq(HEADLINE_WINDOW)
    ]
    winner = headline.sort_values(
        ["delta_net_return_annualized", "cluster_volatility_annualized"],
        ascending=[False, True],
    ).iloc[0]
    return_wins = int(headline["beats_global_net_return"].astype(bool).sum())
    mv_wins = int(headline["mean_variance_dominates_global"].astype(bool).sum())
    weekly_headline = weekly_comparison.loc[
        weekly_comparison["analysis_window"].eq(HEADLINE_WINDOW)
    ]
    monthly_better_than_weekly = int(
        (
            weekly_headline[
                "monthly_minus_weekly_delta_net_return_annualized"
            ]
            > 0.0
        ).sum()
    )
    max_exposure_error = float(
        acceptance[
            [
                "max_long_exposure_error",
                "max_short_exposure_error",
                "max_net_exposure_error",
                "max_gross_exposure_error",
            ]
        ].to_numpy().max()
    )

    print("U1 monthly 12m-skip-1 covariance long-short grid validation: PASS")
    print(f"grid: {EXPECTED_CELLS} cells x 2 windows = {len(comparison)} comparisons")
    print(
        f"signal: {LOOKBACK_MONTHS} included ME returns, skip {SKIP_MONTHS}; "
        f"independent max error={float(signal_regression['max_abs_error'].max()):.3e}"
    )
    print(
        f"exposures: {len(acceptance)}/{len(acceptance)} PASS; "
        f"max error={max_exposure_error:.3e}"
    )
    print(
        f"headline winner: {winner['frequency']} span {int(winner['span'])}; "
        f"net return delta={float(winner['delta_net_return_annualized']):+.6f}; "
        f"volatility delta={float(winner['delta_volatility_annualized']):+.6f}"
    )
    print(
        f"headline cells: global-relative return wins={return_wins}/{EXPECTED_CELLS}; "
        f"mean-variance wins={mv_wins}/{EXPECTED_CELLS}; "
        f"monthly delta beats weekly delta={monthly_better_than_weekly}/{EXPECTED_CELLS}"
    )
    print(
        f"determinism: {int(determinism['byte_identical'].astype(bool).sum())}/"
        f"{len(determinism)} artifacts byte-identical"
    )


if __name__ == "__main__":
    validate()
