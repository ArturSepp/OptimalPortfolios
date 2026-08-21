"""Run the futures 30/30/30/10 global-rank long-short book at 10 bp.

Each signed side allocates 30% to Equity, 30% to Fixed Income, 30% to Commodities,
and 10% to FX.  Momentum is ranked globally within each broad sleeve on the changing
point-in-time eligible cross-section.  The strategy is +1 long / -1 short, uses the
paper raw-weekly 48-week-minus-4-week score, monthly decisions, one W-WED implementation
lag, and the corrected U1 calendar window.  CUA1 Comdty is owner-excluded throughout.

Ten basis points per one-way traded notional is the primary cost assumption.  A 20 bp
replay using identical decisions is retained only as a cost-sensitivity reference.
"""
from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

import papers.cluster_lineage_2026.replication.run_backtests as e5
import papers.cluster_lineage_2026.replication.run_futures_asset_class_long_short as asset
import papers.cluster_lineage_2026.replication.run_futures_sleeve_grid as equal
import papers.cluster_lineage_2026.replication.run_futures_weight_30303010 as construction
import papers.cluster_lineage_2026.replication.run_futures_weight_30303010_u1_window as matched


TARGET = dict(construction.TARGET)
COST_BPS = 10.0
REFERENCE_COST_BPS = 20.0
QUANTILES = tuple(equal.QUANTILES)
METHOD = "sleeve_global"
RUNNER = (
    "papers/cluster_lineage_2026/replication/"
    "run_futures_global_30303010_10bp.py"
)
TOLERANCE = 1e-12


def _root() -> Path:
    """Return and create the external 10 bp combined-global output directory."""
    return e5.get_output_path(
        "e5b", "futures_global_30_30_30_10_10bp_u1_window", create=True
    )


def _design(
    dates: pd.DatetimeIndex,
    eligibility: pd.DataFrame,
    sleeves: pd.Series,
) -> pd.DataFrame:
    """Return the frozen cost, weight, signal, and availability specification."""
    class_counts = {
        sleeve: eligibility.loc[:, sleeves.eq(sleeve)].sum(axis=1)
        for sleeve in equal.SLEEVES
    }
    return pd.DataFrame(
        [
            {
                "universe": equal.UNIVERSE.value,
                "analysis_window": matched.WINDOW,
                "strategy": "long_short",
                "method": METHOD,
                "primary_cost_bps_one_way": COST_BPS,
                "reference_cost_bps_one_way": REFERENCE_COST_BPS,
                "q_values": "|".join(f"{q:.2f}" for q in QUANTILES),
                "decision_dates": len(dates),
                "decision_start": dates.min(),
                "decision_end": dates.max(),
                "signal": "48-week log-return sum excluding latest 4 weeks",
                "signal_sum_min_count": 1,
                "full_lookback_required": False,
                "implementation_lag_observations": 1,
                "equity_budget_per_side": TARGET["Equity"],
                "fixed_income_budget_per_side": TARGET["Fixed Income"],
                "commodities_budget_per_side": TARGET["Commodities"],
                "fx_budget_per_side": TARGET["FX"],
                "eligible_futures_min": int(eligibility.sum(axis=1).min()),
                "eligible_futures_max": int(eligibility.sum(axis=1).max()),
                **{
                    f"{sleeve.lower().replace(' ', '_')}_eligible_min": int(
                        counts.min()
                    )
                    for sleeve, counts in class_counts.items()
                },
                **{
                    f"{sleeve.lower().replace(' ', '_')}_eligible_max": int(
                        counts.max()
                    )
                    for sleeve, counts in class_counts.items()
                },
                "owner_exclusions": "|".join(
                    sorted(e5.FUTURES_INVESTABILITY_EXCLUSIONS)
                ),
                "runner": RUNNER,
            }
        ]
    )


def _availability_rows(
    q: float,
    eligibility: pd.DataFrame,
    sleeves: pd.Series,
    weights: pd.DataFrame,
) -> list[dict]:
    """Return date-level eligible and selected counts by strategic sleeve."""
    rows = []
    for date in eligibility.index:
        for sleeve in equal.SLEEVES:
            members = sleeves.eq(sleeve)
            rows.append(
                {
                    "date": date,
                    "q": q,
                    "sleeve": sleeve,
                    "eligible_contracts": int(
                        eligibility.loc[date, members].sum()
                    ),
                    "selected_long_contracts": int(
                        weights.loc[date, members].gt(0.0).sum()
                    ),
                    "selected_short_contracts": int(
                        weights.loc[date, members].lt(0.0).sum()
                    ),
                    "long_exposure": float(
                        weights.loc[date, members].clip(lower=0.0).sum()
                    ),
                    "short_exposure_abs": float(
                        -weights.loc[date, members].clip(upper=0.0).sum()
                    ),
                }
            )
    return rows


def _run_one_cost(
    *,
    q: float,
    cost_bps: float,
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    diagnostics: Mapping[str, float],
    sleeve_panel: pd.DataFrame,
    ew_nav: pd.Series,
) -> tuple[dict, dict, list[dict], dict]:
    """Backtest identical global decisions at one proportional cost rate."""
    records = matched._run_leg(
        strategy="long_short",
        method=METHOD,
        q=q,
        prices=prices,
        weights=weights,
        diagnostics=diagnostics,
        sleeve_panel=sleeve_panel,
        ew_nav=ew_nav,
        costs=cost_bps / 10000.0,
        target=TARGET,
    )
    performance, acceptance, allocation, horizon = records
    performance["cost_bps_one_way"] = cost_bps
    performance["runner"] = RUNNER
    acceptance["cost_bps_one_way"] = cost_bps
    horizon["cost_bps_one_way"] = cost_bps
    return performance, acceptance, allocation, horizon


def run() -> Mapping[str, pd.DataFrame]:
    """Execute q=20% and q=25% global-rank books at 10 bp and 20 bp."""
    started = time.perf_counter()
    construction._validate_target()
    data = e5.load_universe(equal.UNIVERSE)
    dates = matched._window_dates(
        e5.load_cached(equal.UNIVERSE, e5.SmootherName.BASELINE).dates
    )
    eligibility = e5._investable_eligibility(data, dates)
    columns = eligibility.columns
    scores = e5._raw_momentum_scores(data, dates, vol_adjusted=False)
    scores = scores.reindex(columns=columns).where(eligibility)
    prices = matched._prices_with_context(e5._prices(data).reindex(columns=columns))
    sleeves = equal._broad_sleeves(data.taxonomy, columns)
    sleeve_panel = equal._sleeve_panel(dates, sleeves)
    accepted_navs = pd.read_csv(
        equal._accepted_root() / "navs.csv",
        parse_dates=["date"],
        float_precision="round_trip",
    ).set_index("date")
    ew_nav = matched._bounded_panel(accepted_navs["EW_all"])
    if not isinstance(ew_nav, pd.Series):
        raise AssertionError("bounded EW reference is not a Series")

    primary_rows = []
    sensitivity_rows = []
    acceptance_rows = []
    allocation_rows = []
    horizon_rows = []
    availability_rows = []
    reconstruction_rows = []
    for q in QUANTILES:
        weights, diagnostics = construction._build_constrained_weights(
            "long_short",
            scores,
            eligibility,
            sleeve_panel,
            sleeve_panel,
            q,
        )
        reconstructed = sum(
            asset._standalone_weights(
                scores,
                eligibility,
                sleeve_panel,
                sleeve_panel,
                sleeve,
                q,
            )[0].mul(TARGET[sleeve])
            for sleeve in equal.SLEEVES
        )
        reconstruction_error = float(
            reconstructed.subtract(weights).abs().to_numpy().max()
        )
        reconstruction_rows.append(
            {
                "q": q,
                "max_weight_abs_error": reconstruction_error,
                "tolerance": TOLERANCE,
                "status": "PASS"
                if reconstruction_error <= TOLERANCE
                else "FAIL",
            }
        )
        excluded = weights.columns.intersection(
            sorted(e5.FUTURES_INVESTABILITY_EXCLUSIONS)
        )
        exclusion_error = float(
            weights.loc[:, excluded].abs().to_numpy().max()
        )
        availability_rows.extend(
            _availability_rows(q, eligibility, sleeves, weights)
        )
        cost_records = {}
        for cost_bps in (COST_BPS, REFERENCE_COST_BPS):
            records = _run_one_cost(
                q=q,
                cost_bps=cost_bps,
                prices=prices,
                weights=weights,
                diagnostics=diagnostics,
                sleeve_panel=sleeve_panel,
                ew_nav=ew_nav,
            )
            performance, acceptance, allocation, horizon = records
            acceptance["max_owner_excluded_weight_abs"] = exclusion_error
            acceptance["status"] = (
                "PASS"
                if acceptance["status"] == "PASS"
                and exclusion_error <= TOLERANCE
                else "FAIL"
            )
            cost_records[cost_bps] = performance
            if cost_bps == COST_BPS:
                primary_rows.append(performance)
                acceptance_rows.append(acceptance)
                allocation_rows.extend(allocation)
                horizon_rows.append(horizon)

        primary = cost_records[COST_BPS]
        reference = cost_records[REFERENCE_COST_BPS]
        sensitivity_rows.append(
            {
                "q": q,
                "primary_cost_bps_one_way": COST_BPS,
                "reference_cost_bps_one_way": REFERENCE_COST_BPS,
                "gross_return_annualized": primary["gross_return_annualized"],
                "net_return_annualized_10bp": primary["net_return_annualized"],
                "net_return_annualized_20bp": reference["net_return_annualized"],
                "net_return_improvement_10bp_vs_20bp": (
                    primary["net_return_annualized"]
                    - reference["net_return_annualized"]
                ),
                "sharpe_rf0_10bp": primary["sharpe_rf0"],
                "sharpe_rf0_20bp": reference["sharpe_rf0"],
                "cost_drag_bp_per_year_10bp": primary["cost_drag_bp_per_year"],
                "cost_drag_bp_per_year_20bp": reference["cost_drag_bp_per_year"],
            }
        )

    performance = pd.DataFrame(primary_rows).sort_values("q").reset_index(drop=True)
    acceptance = pd.DataFrame(acceptance_rows).sort_values("q").reset_index(drop=True)
    if not acceptance["status"].eq("PASS").all():
        raise AssertionError(acceptance.loc[~acceptance["status"].eq("PASS")])
    reconstruction = pd.DataFrame(reconstruction_rows).sort_values("q")
    if not reconstruction["status"].eq("PASS").all():
        raise AssertionError(reconstruction.loc[~reconstruction["status"].eq("PASS")])
    outputs = {
        "design": _design(dates, eligibility, sleeves),
        "performance": performance,
        "cost_sensitivity": pd.DataFrame(sensitivity_rows).sort_values("q"),
        "acceptance": acceptance,
        "allocation_diagnostics": pd.DataFrame(allocation_rows),
        "availability_by_date": pd.DataFrame(availability_rows),
        "horizon_diagnostic": pd.DataFrame(horizon_rows),
        "standalone_weight_reconstruction": reconstruction,
    }
    for name, frame in outputs.items():
        e5._write(frame, _root() / f"{name}.csv")
    e5._write(
        pd.DataFrame([{"runtime_seconds": time.perf_counter() - started}]),
        _root() / "runtime.csv",
    )
    return outputs


def _hash_outputs() -> dict[str, str]:
    """Hash deterministic outputs while excluding runtime and replay records."""
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(_root().glob("*.csv"))
        if path.name not in {"runtime.csv", "determinism.csv"}
    }


def verify_determinism() -> pd.DataFrame:
    """Replay the complete 10/20 bp sensitivity and require identical outputs."""
    run()
    first = _hash_outputs()
    run()
    second = _hash_outputs()
    names = sorted(set(first) | set(second))
    replay = pd.DataFrame(
        {
            "artifact": names,
            "first_sha256": [first.get(name) for name in names],
            "second_sha256": [second.get(name) for name in names],
            "byte_identical": [first.get(name) == second.get(name) for name in names],
        }
    )
    e5._write(replay, _root() / "determinism.csv")
    if not replay["byte_identical"].all():
        raise AssertionError(replay.loc[~replay["byte_identical"]])
    return replay


def main() -> None:
    """Run, replay, and print primary and cost-sensitivity results."""
    replay = verify_determinism()
    performance = pd.read_csv(
        _root() / "performance.csv", float_precision="round_trip"
    )
    print(
        performance[
            [
                "q",
                "gross_return_annualized",
                "net_return_annualized",
                "volatility_annualized",
                "sharpe_rf0",
                "one_way_turnover_annualized",
                "cost_drag_bp_per_year",
            ]
        ].to_string(index=False)
    )
    print(
        f"Futures global 30/30/30/10 at 10 bp: PASS "
        f"({len(replay)}/{len(replay)} deterministic)",
        flush=True,
    )


if __name__ == "__main__":
    main()
