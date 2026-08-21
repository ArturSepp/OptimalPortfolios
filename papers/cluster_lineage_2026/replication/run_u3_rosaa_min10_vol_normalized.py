"""Run the primary U3 ROSAA cross-section with volatility-normalized sizing.

Selection ranks all eligible futures together. Each selected contract is sized
with TrendFollowingSystems ``compute_vol_target_weight``: 15% divided by its
point-in-time annualized EWMA volatility, estimated from daily log returns with
span 33 and annualization factor 260, capped at 5 as in the European-system
backtest. Scaled contract weights are averaged within the selected long and
short quartiles. The filtered U3 universe, ROSAA signal, M1-star cluster score
standardisation with minimum cluster size 10, monthly schedule, implementation
lag, and 10 bp one-way costs are unchanged. QIS holds units between trades.
"""
from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd
from trendfollowing.systems.backtest_utils import compute_vol_target_weight

import papers.cluster_lineage_2026.replication.run_backtests as e5
import papers.cluster_lineage_2026.replication.run_futures_best_relative_pnl_scatter as prior
import papers.cluster_lineage_2026.replication.run_futures_prod_cluster_30303010_10bp as u3_source
import papers.cluster_lineage_2026.replication.run_futures_sleeve_grid as u3_equal
import papers.cluster_lineage_2026.replication.run_u2_u3_min_cluster10_signal_comparison as comparison
import papers.cluster_lineage_2026.replication.run_u3_rosaa_min10_equal_contract as equal_contract


SIGNAL_ID = "rosaa_risk_adjusted_momentum"
BOOK = "vol_normalized_single_cross_section"
VOL_SPAN_DAYS = 33
INSTRUMENT_VOL_TARGET = 0.15
ANNUALIZATION_FACTOR = 260.0
INSTRUMENT_WEIGHT_CAP = 5.0
ACCOUNTING_TOLERANCE = 1e-10
RUNNER = (
    "papers/cluster_lineage_2026/replication/"
    "run_u3_rosaa_min10_vol_normalized.py"
)


def _root() -> Path:
    """Return the gitignored volatility-normalized output directory."""
    root = (
        Path(__file__).resolve().parents[1]
        / "local_outputs"
        / "e5b"
        / "u3_rosaa_ra_min10_vol_normalized_20260816"
    )
    root.mkdir(parents=True, exist_ok=True)
    return root


def _point_in_time_vol_scalers(
    dates: pd.DatetimeIndex,
    columns: pd.Index,
    vol_span_days: int = VOL_SPAN_DAYS,
) -> tuple[pd.DataFrame, pd.DataFrame, Mapping[str, float]]:
    """Compute exact TrendFollowingSystems scalers using only data known by date."""
    daily_log_returns = u3_source._read_daily(columns).sort_index()
    scalers = pd.DataFrame(np.nan, index=dates, columns=columns)
    annual_vols = pd.DataFrame(np.nan, index=dates, columns=columns)
    latest_source_dates = []
    for date in dates:
        history = daily_log_returns.loc[:date]
        if history.empty:
            continue
        raw_scaler, vols = compute_vol_target_weight(
            returns=history.to_numpy(),
            vol_span=vol_span_days,
            vol_target=INSTRUMENT_VOL_TARGET,
            annualization_factor=ANNUALIZATION_FACTOR,
        )
        scalers.loc[date] = raw_scaler[-1]
        annual_vols.loc[date] = vols[-1]
        latest_source_dates.append(history.index[-1])
    uncapped = scalers.copy()
    scalers = scalers.clip(lower=0.0, upper=INSTRUMENT_WEIGHT_CAP)
    finite = np.isfinite(scalers.to_numpy()) & np.isfinite(annual_vols.to_numpy())
    identity = scalers.multiply(annual_vols).where(
        uncapped.le(INSTRUMENT_WEIGHT_CAP) & finite
    )
    identity_error = identity.subtract(INSTRUMENT_VOL_TARGET).abs().to_numpy()
    identity_error = identity_error[np.isfinite(identity_error)]
    lookahead_days = [
        (source_date - decision_date).days
        for source_date, decision_date in zip(latest_source_dates, dates)
    ]
    diagnostics = {
        "vol_span_days": vol_span_days,
        "instrument_vol_target": INSTRUMENT_VOL_TARGET,
        "annualization_factor": ANNUALIZATION_FACTOR,
        "instrument_weight_cap": INSTRUMENT_WEIGHT_CAP,
        "max_vol_source_lookahead_days": max(lookahead_days, default=0),
        "uncapped_scaler_identity_max_abs_error": (
            float(identity_error.max()) if identity_error.size else 0.0
        ),
        "finite_scaler_observations": int(finite.sum()),
        "cap_hits": int(uncapped.ge(INSTRUMENT_WEIGHT_CAP).to_numpy().sum()),
    }
    if diagnostics["max_vol_source_lookahead_days"] > 0:
        raise AssertionError("volatility scaler uses future data")
    return scalers, annual_vols, diagnostics


def _vol_normalized_weights(
    base_weights: pd.DataFrame,
    scalers: pd.DataFrame,
    annual_vols: pd.DataFrame,
    eligibility: pd.DataFrame,
) -> tuple[pd.DataFrame, Mapping[str, float]]:
    """Apply capped inverse-volatility sizing and average within each side."""
    scalers = scalers.reindex_like(base_weights)
    annual_vols = annual_vols.reindex_like(base_weights)
    eligibility = eligibility.reindex_like(base_weights).fillna(False).astype(bool)
    valid = eligibility & scalers.notna() & scalers.gt(0.0)
    long_selected = base_weights.gt(0.0) & valid
    short_selected = base_weights.lt(0.0) & valid
    long_counts = long_selected.sum(axis=1)
    short_counts = short_selected.sum(axis=1)
    if long_counts.le(0).any() or short_counts.le(0).any():
        raise AssertionError("volatility sizing produced an empty selected side")
    long_book = scalers.where(long_selected, 0.0).div(long_counts, axis=0)
    short_book = scalers.where(short_selected, 0.0).div(short_counts, axis=0)
    weights = long_book - short_book
    risk_long = long_book.multiply(annual_vols).sum(axis=1)
    risk_short = short_book.multiply(annual_vols).sum(axis=1)
    diagnostics = {
        "min_long_contracts": int(long_counts.min()),
        "max_long_contracts": int(long_counts.max()),
        "min_short_contracts": int(short_counts.min()),
        "max_short_contracts": int(short_counts.max()),
        "max_weight_outside_eligibility": float(
            weights.where(~eligibility, 0.0).abs().to_numpy().max()
        ),
        "max_overlap_assets": int((long_selected & short_selected).to_numpy().sum()),
        "long_capital_gross_min": float(long_book.sum(axis=1).min()),
        "long_capital_gross_max": float(long_book.sum(axis=1).max()),
        "short_capital_gross_min": float(short_book.sum(axis=1).min()),
        "short_capital_gross_max": float(short_book.sum(axis=1).max()),
        "long_average_target_risk_min": float(risk_long.min()),
        "long_average_target_risk_max": float(risk_long.max()),
        "short_average_target_risk_min": float(risk_short.min()),
        "short_average_target_risk_max": float(risk_short.max()),
    }
    return weights, diagnostics


def run() -> Mapping[str, pd.DataFrame]:
    """Run, reconcile, and save the volatility-normalized comparison."""
    started = time.perf_counter()
    context = comparison._u3_context()
    global_scores, cluster_scores, signal_diagnostics = comparison._signal_pair(
        signal_id=SIGNAL_ID,
        prices=context["signal_prices"],
        benchmark=context["benchmark"],
        groups=context["groups"],
        dates=context["dates"],
        eligibility=context["eligibility"],
    )
    scalers, annual_vols, vol_diagnostics = _point_in_time_vol_scalers(
        context["dates"], context["eligibility"].columns
    )
    portfolios = {}
    weights = {}
    weight_rows = []
    for method, scores in (("global", global_scores), ("cluster", cluster_scores)):
        base_weights, base_diagnostics = equal_contract._equal_contract_weights(
            scores, context
        )
        method_weights, sizing_diagnostics = _vol_normalized_weights(
            base_weights,
            scalers,
            annual_vols,
            context["eligibility"],
        )
        passed = (
            sizing_diagnostics["max_weight_outside_eligibility"]
            <= comparison.TOLERANCE
            and sizing_diagnostics["max_overlap_assets"] == 0
        )
        weight_rows.append(
            {
                "method": method,
                **base_diagnostics,
                **sizing_diagnostics,
                "status": "PASS" if passed else "FAIL",
            }
        )
        net, gross = u3_equal._backtest(
            context["performance_prices"],
            method_weights.reindex(index=context["scheduled_dates"]),
            context["cost_bps"] / 10000.0,
            f"U3_{BOOK}_{SIGNAL_ID}_{method}_min10",
        )
        portfolios[method] = net
        portfolios[f"{method}_gross"] = gross
        weights[method] = method_weights
    weight_diagnostics = pd.DataFrame(weight_rows)
    if not weight_diagnostics["status"].eq("PASS").all():
        raise AssertionError(weight_diagnostics)

    performance_rows = []
    pnl_by_method = {}
    pnl_diagnostics = {}
    for method in ("global", "cluster"):
        performance_rows.append(
            {
                "method": method,
                **u3_equal._performance_payload(
                    portfolios[method],
                    portfolios[f"{method}_gross"],
                    context["ew_nav"],
                ),
            }
        )
        pnl_by_method[method], pnl_diagnostics[method] = prior._net_attribution(
            portfolios[method]
        )
    performance = pd.DataFrame(performance_rows)
    eligibility = context["eligibility"]
    data = e5.load_universe(u3_equal.UNIVERSE)
    tickers = eligibility.columns[eligibility.any(axis=0)]
    sleeves = u3_equal._broad_sleeves(data.taxonomy, eligibility.columns)
    instrument = prior._instrument_table(
        tickers=tickers,
        cluster_net_pnl=pnl_by_method["cluster"].sum(axis=0),
        global_net_pnl=pnl_by_method["global"].sum(axis=0),
        cluster_beginning_nav=float(pnl_diagnostics["cluster"]["beginning_nav"]),
        global_beginning_nav=float(pnl_diagnostics["global"]["beginning_nav"]),
        taxonomy=data.taxonomy,
        sleeves=sleeves,
        eligibility=eligibility,
        cluster_weights=weights["cluster"],
        global_weights=weights["global"],
    )
    cluster_total = float(instrument["cluster_net_pnl_pct_of_start"].sum())
    global_total = float(instrument["global_net_pnl_pct_of_start"].sum())
    accounting_error = max(
        float(pnl_diagnostics[method][field])
        for method in ("global", "cluster")
        for field in (
            "max_step_reconciliation_abs_error",
            "cumulative_reconciliation_abs_error",
        )
    )
    accounting_error = max(
        accounting_error,
        abs(
            cluster_total
            - 100.0
            * float(pnl_diagnostics["cluster"]["attributed_net_total_return"])
        ),
        abs(
            global_total
            - 100.0
            * float(pnl_diagnostics["global"]["attributed_net_total_return"])
        ),
    )
    excluded_rows = int(
        instrument["ticker"].isin(e5.FUTURES_INVESTABILITY_EXCLUSIONS).sum()
    )
    passed = (
        accounting_error <= ACCOUNTING_TOLERANCE
        and excluded_rows == 0
        and vol_diagnostics["max_vol_source_lookahead_days"] <= 0
        and vol_diagnostics["uncapped_scaler_identity_max_abs_error"]
        <= comparison.TOLERANCE
    )
    reconciliation = pd.DataFrame(
        [
            {
                "universe": "U3_futures",
                "construction": BOOK,
                "signal_id": SIGNAL_ID,
                "analysis_window": context["window"],
                "min_cluster_size": comparison.MIN_CLUSTER_SIZE,
                "q": context["q"],
                "cost_bps_one_way": context["cost_bps"],
                "eligible_instruments": len(instrument),
                "cluster_total_net_pnl_pct": cluster_total,
                "global_total_net_pnl_pct": global_total,
                "cluster_minus_global_pct": cluster_total - global_total,
                "maximum_accounting_error": accounting_error,
                "owner_excluded_rows": excluded_rows,
                "status": "PASS" if passed else "FAIL",
                "runner": RUNNER,
            }
        ]
    )
    if not passed:
        raise AssertionError(reconciliation.to_dict(orient="records")[0])
    outputs = {
        "performance": performance,
        "weight_diagnostics": weight_diagnostics,
        "volatility_diagnostics": pd.DataFrame([vol_diagnostics]),
        "signal_diagnostics": pd.DataFrame([signal_diagnostics]),
        "instrument_pnl": instrument,
        "reconciliation": reconciliation,
        "design": pd.DataFrame(
            [
                {
                    "construction": BOOK,
                    "selection": "top/bottom q across all eligible futures",
                    "sizing": "mean per side of capped TrendFollowingSystems vol-target weights",
                    "vol_span_days": VOL_SPAN_DAYS,
                    "instrument_vol_target": INSTRUMENT_VOL_TARGET,
                    "annualization_factor": ANNUALIZATION_FACTOR,
                    "instrument_weight_cap": INSTRUMENT_WEIGHT_CAP,
                    "qis_backtest": True,
                    "signal": SIGNAL_ID,
                    "min_cluster_size": comparison.MIN_CLUSTER_SIZE,
                    "runner": RUNNER,
                }
            ]
        ),
    }
    for name, frame in outputs.items():
        e5._write(frame, _root() / f"{name}.csv")
    e5._write(
        pd.DataFrame([{"runtime_seconds": time.perf_counter() - started}]),
        _root() / "runtime.csv",
    )
    return outputs


def _hash_outputs() -> dict[str, str]:
    """Hash deterministic artifacts while excluding runtime and replay."""
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(_root().glob("*.csv"))
        if path.name not in {"runtime.csv", "determinism.csv"}
    }


def verify_determinism() -> pd.DataFrame:
    """Replay the volatility-normalized design and require identical outputs."""
    run()
    first = _hash_outputs()
    run()
    second = _hash_outputs()
    names = sorted(first)
    replay = pd.DataFrame(
        {
            "artifact": names,
            "first_sha256": [first[name] for name in names],
            "second_sha256": [second[name] for name in names],
            "byte_identical": [first[name] == second[name] for name in names],
        }
    )
    e5._write(replay, _root() / "determinism.csv")
    if not replay["byte_identical"].all():
        raise AssertionError(replay.loc[~replay["byte_identical"]])
    return replay


def main() -> None:
    """Run, replay, and print volatility-normalized U3 performance."""
    replay = verify_determinism()
    performance = pd.read_csv(_root() / "performance.csv", float_precision="round_trip")
    print(performance.to_string(index=False))
    print(f"U3 ROSAA volatility-normalized analysis: PASS ({len(replay)}/{len(replay)})")


if __name__ == "__main__":
    main()
