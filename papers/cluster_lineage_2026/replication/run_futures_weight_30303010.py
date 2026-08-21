"""Run the futures 30/30/30/10 sleeve-budget performance experiment.

The strategic long-side budgets are Equity 30%, Fixed Income 30%, Commodities 30%,
and FX 10%.  The same budgets are imposed on the global-within-sleeve control and on
the baseline and M1-star cluster legs, so cluster comparisons remain like-for-like.
For long-short portfolios the target is applied independently to both signed sides,
giving +1/-1 exposure.  The accepted unconstrained global leg is retained unchanged as
an external reference, and the previously accepted 25/25/25/25 experiment is reported
as a separate budget-sensitivity comparison.

All other frozen futures conventions are unchanged: 48-week production momentum
excluding four weeks, monthly decisions, one-observation implementation lag, W-WED
returns, 20 bp costs, q=0.20 primary, and q=0.25 robustness.  EW-all is used only by
the accepted performance routine for beta and alpha columns; it is never a ranking
yardstick.
"""
from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

import papers.cluster_lineage_2026.replication.run_backtests as e5
import papers.cluster_lineage_2026.replication.run_futures_sleeve_grid as equal


TARGET = {
    "Equity": 0.30,
    "Fixed Income": 0.30,
    "Commodities": 0.30,
    "FX": 0.10,
}
RUNNER = "papers/cluster_lineage_2026/replication/run_futures_weight_30303010.py"


def _root() -> Path:
    """Return and create the external 30/30/30/10 output directory."""
    return e5.get_output_path("e5b", "futures_weight_30_30_30_10", create=True)


def _validate_target() -> None:
    """Require a complete, positive, unit-sum strategic sleeve target."""
    if tuple(TARGET) != equal.SLEEVES:
        raise AssertionError(f"target sleeve order differs from {equal.SLEEVES}: {TARGET}")
    if any(weight <= 0.0 for weight in TARGET.values()):
        raise AssertionError(f"all strategic sleeve targets must be positive: {TARGET}")
    if not np.isclose(sum(TARGET.values()), 1.0, atol=1e-15):
        raise AssertionError(f"strategic sleeve targets do not sum to one: {TARGET}")


def _design(dates: pd.DatetimeIndex, sleeves: pd.Series) -> pd.DataFrame:
    """Return the frozen machine-readable scenario design."""
    spec = e5.get_universe_spec(equal.UNIVERSE)
    equal_performance = equal._root() / "performance.csv"
    return pd.DataFrame(
        [
            {
                "universe": equal.UNIVERSE.value,
                "contracts": len(sleeves),
                "decision_dates": len(dates),
                "decision_start": dates.min(),
                "decision_end": dates.max(),
                "signal": "48-week log-return sum excluding latest 4 weeks",
                "primary_q": equal.PRIMARY_Q,
                "robustness_q": equal.QUANTILES[1],
                "cost_bps": spec.cost_bps,
                "implementation_lag": 1,
                "equity_target": TARGET["Equity"],
                "fixed_income_target": TARGET["Fixed Income"],
                "commodities_target": TARGET["Commodities"],
                "fx_target": TARGET["FX"],
                "configs": "baseline|M1_star",
                "returns_convention": equal.data_convention(spec),
                "equal_sleeve_performance_sha256": hashlib.sha256(
                    equal_performance.read_bytes()
                ).hexdigest(),
                "runner": RUNNER,
            }
        ]
    )


def _comparison_vs_equal_sleeves(performance: pd.DataFrame) -> pd.DataFrame:
    """Compare each 30/30/30/10 leg with its 25/25/25/25 counterpart."""
    benchmark = pd.read_csv(
        equal._root() / "performance.csv", float_precision="round_trip"
    ).set_index(["strategy", "q", "method"])
    if benchmark.index.has_duplicates:
        raise AssertionError("equal-sleeve performance keys are not unique")
    rows = []
    for _, current in performance.iterrows():
        key = (current["strategy"], current["q"], current["method"])
        if key not in benchmark.index:
            raise AssertionError(f"equal-sleeve benchmark is missing {key}")
        reference = benchmark.loc[key]
        row = current.to_dict()
        row["equal_sleeve_method"] = current["method"]
        for metric in equal.COMPARISON_METRICS:
            row[f"equal_sleeve_{metric}"] = reference[metric]
            row[f"delta_vs_equal_sleeves_{metric}"] = (
                current[metric] - reference[metric]
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["strategy", "q", "method"])


def _build_constrained_weights(
    strategy: str,
    scores: pd.DataFrame,
    eligibility: pd.DataFrame,
    sleeve_panel: pd.DataFrame,
    groups: pd.DataFrame,
    q: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Construct one target-budget global or cluster portfolio."""
    if strategy == "long_only":
        return equal._long_only_sleeve_weights(
            scores, eligibility, sleeve_panel, groups, q, TARGET
        )
    return equal._long_short_sleeve_weights(
        scores, eligibility, sleeve_panel, groups, q, TARGET
    )


def run() -> Mapping[str, pd.DataFrame]:
    """Execute the 30/30/30/10 global-versus-cluster experiment once."""
    started = time.perf_counter()
    _validate_target()
    data = e5.load_universe(equal.UNIVERSE)
    dates = e5.load_cached(equal.UNIVERSE, e5.SmootherName.BASELINE).dates
    eligibility = e5._investable_eligibility(data, dates)
    columns = eligibility.columns
    scores = e5._raw_momentum_scores(
        data, dates, vol_adjusted=False
    ).reindex(columns=columns).where(eligibility)
    prices = e5._prices(data).reindex(columns=columns)
    sleeves = equal._broad_sleeves(data.taxonomy, columns)
    sleeve_panel = equal._sleeve_panel(dates, sleeves)
    accepted_navs = pd.read_csv(
        equal._accepted_root() / "navs.csv",
        parse_dates=["date"],
        float_precision="round_trip",
    ).set_index("date")
    ew_nav = accepted_navs["EW_all"]
    costs = e5.get_universe_spec(equal.UNIVERSE).cost_bps / 10000.0
    cluster_groups = {
        config: equal._hierarchical_groups(
            e5._cluster_groups(equal.UNIVERSE, config).reindex(
                index=dates, columns=columns
            ),
            sleeve_panel,
        )
        for config in equal.CONFIGS
    }

    performance_rows = []
    acceptance_rows = []
    allocation_rows = []
    primary_global_weights = None
    constrained_groups = {
        "sleeve_global": sleeve_panel,
        **{
            f"sleeve_cluster_{config.value}": groups
            for config, groups in cluster_groups.items()
        },
    }
    for q in equal.QUANTILES:
        for strategy in ("long_only", "long_short"):
            original_weights, diagnostics = equal._original_global_weights(
                strategy, scores, eligibility, q
            )
            if strategy == "long_only" and q == equal.PRIMARY_Q:
                primary_global_weights = original_weights
            performance, acceptance, allocation = equal._run_leg(
                strategy=strategy,
                method="original_global",
                q=q,
                prices=prices,
                weights=original_weights,
                diagnostics=diagnostics,
                sleeve_panel=sleeve_panel,
                ew_nav=ew_nav,
                costs=costs,
                target=None,
                runner=RUNNER,
            )
            performance_rows.append(performance)
            acceptance_rows.append(acceptance)
            allocation_rows.extend(allocation)

            for method, groups in constrained_groups.items():
                weights, diagnostics = _build_constrained_weights(
                    strategy,
                    scores,
                    eligibility,
                    sleeve_panel,
                    groups,
                    q,
                )
                performance, acceptance, allocation = equal._run_leg(
                    strategy=strategy,
                    method=method,
                    q=q,
                    prices=prices,
                    weights=weights,
                    diagnostics=diagnostics,
                    sleeve_panel=sleeve_panel,
                    ew_nav=ew_nav,
                    costs=costs,
                    target=TARGET,
                    runner=RUNNER,
                )
                performance_rows.append(performance)
                acceptance_rows.append(acceptance)
                allocation_rows.extend(allocation)

    if primary_global_weights is None:
        raise AssertionError("primary accepted global weights were not constructed")
    performance = pd.DataFrame(performance_rows).sort_values(
        ["strategy", "q", "method"]
    ).reset_index(drop=True)
    acceptance = pd.DataFrame(acceptance_rows).sort_values(
        ["strategy", "q", "method"]
    ).reset_index(drop=True)
    if not acceptance["status"].eq("PASS").all():
        raise AssertionError(acceptance.loc[~acceptance["status"].eq("PASS")])
    regression = equal._global_regression(primary_global_weights, performance)
    if not regression["status"].eq("PASS").all():
        raise AssertionError(regression)
    outputs = {
        "design": _design(dates, sleeves),
        "performance": performance,
        "comparison": equal._comparison(performance),
        "comparison_vs_equal_sleeves": _comparison_vs_equal_sleeves(performance),
        "allocation_diagnostics": pd.DataFrame(allocation_rows),
        "acceptance": acceptance,
        "global_regression": regression,
    }
    for name, frame in outputs.items():
        e5._write(frame, _root() / f"{name}.csv")
    e5._write(
        pd.DataFrame(
            [
                {
                    "portfolios": len(performance),
                    "runtime_seconds": time.perf_counter() - started,
                }
            ]
        ),
        _root() / "runtime.csv",
    )
    return outputs


def _hash_outputs() -> dict[str, str]:
    """Hash numerical outputs while excluding timing and replay records."""
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(_root().glob("*.csv"))
        if path.name not in {"runtime.csv", "determinism.csv"}
    }


def verify_determinism() -> pd.DataFrame:
    """Replay the full target-budget experiment and require identical CSV bytes."""
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
    """Run and replay the 30/30/30/10 futures experiment."""
    replay = verify_determinism()
    print(
        "Futures 30/30/30/10 grid: PASS "
        f"({len(replay)}/{len(replay)} deterministic)",
        flush=True,
    )


if __name__ == "__main__":
    main()
