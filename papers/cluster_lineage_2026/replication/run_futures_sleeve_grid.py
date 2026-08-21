"""Run fair four-sleeve global-versus-cluster futures momentum portfolios.

The accepted futures global rank is first retained unchanged.  Its decision-date exposure
is then diagnosed over four owner-specified sleeves: Equity, Fixed Income, Commodities,
and FX.  Because that diagnostic breaches the predeclared ten-percentage-point mean-budget
guard and frequently leaves sleeves empty, the follow-up gives each sleeve exactly 25%.

The equal-sleeve global control ranks within each broad sleeve.  Cluster treatments rank
within correlation clusters split by broad sleeve and allocate each sleeve budget equally
across its available clusters.  Thus global and cluster legs have identical strategic
budgets.  Long-short portfolios apply 25% independently to each sleeve on both signed
sides, producing +1/-1 exposure and zero net exposure.  The accepted 48-week momentum
excluding four weeks, monthly decisions, one-observation implementation lag, and 20 bp
futures costs are unchanged.  q=0.20 is primary and q=0.25 is labelled robustness.
"""
from __future__ import annotations

import hashlib
import time
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

import papers.cluster_lineage_2026.replication.run_backtests as e5
from papers.cluster_lineage_2026.replication.run_e5b import (
    GROUP_BUDGET_TOLERANCE,
    WEIGHT_TOLERANCE,
    _group_equal_from_ranks,
)
from papers.cluster_lineage_2026.replication.run_u1_global_grid import _backtest


UNIVERSE = e5.UniverseName.FUTURES
SLEEVES = ("Equity", "Fixed Income", "Commodities", "FX")
TARGET = {sleeve: 0.25 for sleeve in SLEEVES}
PRIMARY_Q = 0.20
QUANTILES = (PRIMARY_Q, 0.25)
CONFIGS = (e5.SmootherName.BASELINE, e5.SmootherName.M1_STAR)
EXPOSURE_TRIGGER_TOLERANCE = 0.10
EXPOSURE_TOLERANCE = 1e-12
COMPARISON_METRICS = (
    "net_total_return",
    "net_return_annualized",
    "volatility_annualized",
    "sharpe_rf0",
    "alpha_vs_ew_annualized",
    "beta_vs_ew",
    "one_way_turnover_annualized",
    "cost_drag_bp_per_year",
    "gross_return_annualized",
)
RUNNER = "papers/cluster_lineage_2026/replication/run_futures_sleeve_grid.py"


def _root() -> Path:
    """Return and create the external futures four-sleeve output directory."""
    return e5.get_output_path("e5b", "futures_four_sleeve", create=True)


def _accepted_root() -> Path:
    """Return the accepted futures E5 output directory."""
    return e5.get_output_path("backtests", UNIVERSE.value)


def _broad_sleeves(taxonomy: pd.DataFrame, columns: pd.Index) -> pd.Series:
    """Map the seven accepted asset classes into four complete broad sleeves."""
    asset_class = taxonomy["asset_class"].reindex(columns)
    mapping = {
        "Equities": "Equity",
        "Bonds": "Fixed Income",
        "STIR": "Fixed Income",
        "Agriculture": "Commodities",
        "Energy": "Commodities",
        "Metals": "Commodities",
        "FX": "FX",
    }
    broad = asset_class.map(mapping)
    if broad.isna().any():
        missing = asset_class.loc[broad.isna()].value_counts(dropna=False).to_dict()
        raise AssertionError(f"unclassified futures asset classes: {missing}")
    broad.name = "sleeve"
    return broad


def _sleeve_panel(index: pd.DatetimeIndex, sleeves: pd.Series) -> pd.DataFrame:
    """Broadcast the static four-sleeve map over monthly decision dates."""
    return pd.DataFrame(
        np.tile(sleeves.to_numpy(), (len(index), 1)),
        index=index,
        columns=sleeves.index,
    )


def _hierarchical_groups(
    clusters: pd.DataFrame, sleeve_panel: pd.DataFrame
) -> pd.DataFrame:
    """Split every estimated correlation cluster by its broad futures sleeve."""
    cluster_text = clusters.astype("string")
    groups = sleeve_panel.astype("string") + "|" + cluster_text
    return groups.where(clusters.notna())


def _sleeve_budget_error(
    side: pd.DataFrame,
    sleeve_panel: pd.DataFrame,
    target: Mapping[str, float] = TARGET,
) -> float:
    """Return the maximum absolute error against the four strategic budgets."""
    errors = []
    for sleeve in SLEEVES:
        measured = side.where(sleeve_panel.eq(sleeve), 0.0).sum(axis=1)
        errors.append(float(measured.sub(target[sleeve]).abs().max()))
    return max(errors)


def _weighted_side(
    scores: pd.DataFrame,
    eligibility: pd.DataFrame,
    sleeve_panel: pd.DataFrame,
    groups: pd.DataFrame,
    q: float,
    target: Mapping[str, float] = TARGET,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Allocate one unit side over sleeves and available within-sleeve groups."""
    ranks = e5._rank_panel(scores, groups)
    output = pd.DataFrame(0.0, index=scores.index, columns=scores.columns)
    weight_errors = []
    group_errors = []
    minimum_groups = []
    for sleeve in SLEEVES:
        sleeve_eligibility = eligibility & sleeve_panel.eq(sleeve)
        weights, available, validation = _group_equal_from_ranks(
            ranks,
            sleeve_eligibility,
            groups,
            q,
            UNIVERSE,
        )
        if available.le(0).any():
            raise AssertionError(f"{sleeve} has no selected group on a decision date")
        output = output.add(weights.mul(target[sleeve]), fill_value=0.0)
        weight_errors.append(float(validation["weight_sum_abs_error"].max()))
        group_errors.append(float(validation["max_group_budget_abs_error"].max()))
        minimum_groups.append(int(available.min()))
    return output, {
        "max_pre_scale_weight_sum_abs_error": max(weight_errors),
        "max_within_sleeve_group_budget_abs_error": max(group_errors),
        "minimum_available_groups_in_sleeve": min(minimum_groups),
    }


def _renormalize_side_by_sleeve(
    side: pd.DataFrame,
    sleeve_panel: pd.DataFrame,
    target: Mapping[str, float] = TARGET,
) -> pd.DataFrame:
    """Restore exact sleeve budgets after removing long-short overlap."""
    output = pd.DataFrame(0.0, index=side.index, columns=side.columns)
    for sleeve in SLEEVES:
        sleeve_side = side.where(sleeve_panel.eq(sleeve), 0.0)
        total = sleeve_side.sum(axis=1)
        if total.le(0.0).any():
            raise AssertionError(f"{sleeve} has an empty side after overlap removal")
        output = output.add(
            sleeve_side.div(total, axis=0).mul(target[sleeve]), fill_value=0.0
        )
    return output


def _long_only_sleeve_weights(
    scores: pd.DataFrame,
    eligibility: pd.DataFrame,
    sleeve_panel: pd.DataFrame,
    groups: pd.DataFrame,
    q: float,
    target: Mapping[str, float] = TARGET,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Build a +1 four-sleeve long-only portfolio and exact diagnostics."""
    weights, pre_scale = _weighted_side(
        scores, eligibility, sleeve_panel, groups, q, target
    )
    total = weights.sum(axis=1)
    diagnostics = {
        **pre_scale,
        "max_weight_sum_abs_error": float(total.sub(1.0).abs().max()),
        "max_top_level_sleeve_budget_abs_error": _sleeve_budget_error(
            weights, sleeve_panel, target
        ),
        "max_long_exposure_abs_error": float(total.sub(1.0).abs().max()),
        "max_short_exposure_abs_error": 0.0,
        "max_net_exposure_abs_error": float(total.sub(1.0).abs().max()),
        "max_gross_exposure_abs_error": float(
            weights.abs().sum(axis=1).sub(1.0).abs().max()
        ),
        "max_overlap_assets_removed": 0,
    }
    return weights, diagnostics


def _long_short_sleeve_weights(
    scores: pd.DataFrame,
    eligibility: pd.DataFrame,
    sleeve_panel: pd.DataFrame,
    groups: pd.DataFrame,
    q: float,
    target: Mapping[str, float] = TARGET,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Build a sleeve-neutral +1/-1 portfolio and exact diagnostics."""
    long_raw, long_pre = _weighted_side(
        scores, eligibility, sleeve_panel, groups, q, target
    )
    short_raw, short_pre = _weighted_side(
        -scores, eligibility, sleeve_panel, groups, q, target
    )
    overlap = long_raw.gt(0.0) & short_raw.gt(0.0)
    long_book = _renormalize_side_by_sleeve(
        long_raw.mask(overlap, 0.0), sleeve_panel, target
    )
    short_book = _renormalize_side_by_sleeve(
        short_raw.mask(overlap, 0.0), sleeve_panel, target
    )
    weights = long_book - short_book
    diagnostics = {
        "max_pre_scale_weight_sum_abs_error": max(
            long_pre["max_pre_scale_weight_sum_abs_error"],
            short_pre["max_pre_scale_weight_sum_abs_error"],
        ),
        "max_within_sleeve_group_budget_abs_error": max(
            long_pre["max_within_sleeve_group_budget_abs_error"],
            short_pre["max_within_sleeve_group_budget_abs_error"],
        ),
        "minimum_available_groups_in_sleeve": min(
            long_pre["minimum_available_groups_in_sleeve"],
            short_pre["minimum_available_groups_in_sleeve"],
        ),
        "max_weight_sum_abs_error": float(weights.sum(axis=1).abs().max()),
        "max_top_level_sleeve_budget_abs_error": max(
            _sleeve_budget_error(long_book, sleeve_panel, target),
            _sleeve_budget_error(short_book, sleeve_panel, target),
        ),
        "max_long_exposure_abs_error": float(
            long_book.sum(axis=1).sub(1.0).abs().max()
        ),
        "max_short_exposure_abs_error": float(
            short_book.sum(axis=1).sub(1.0).abs().max()
        ),
        "max_net_exposure_abs_error": float(weights.sum(axis=1).abs().max()),
        "max_gross_exposure_abs_error": float(
            weights.abs().sum(axis=1).sub(2.0).abs().max()
        ),
        "max_overlap_assets_removed": int(overlap.sum(axis=1).max()),
    }
    return weights, diagnostics


def _original_global_weights(
    strategy: str,
    scores: pd.DataFrame,
    eligibility: pd.DataFrame,
    q: float,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Build the accepted unconstrained whole-universe rank construction."""
    groups = pd.DataFrame("global", index=scores.index, columns=scores.columns)

    def side(source: pd.DataFrame) -> pd.DataFrame:
        """Build one accepted asset-equal signed side."""
        ranks = e5._rank_panel(source, groups)
        return e5._weights_from_ranks(ranks, eligibility, q, UNIVERSE)

    long_book = side(scores)
    if strategy == "long_only":
        total = long_book.sum(axis=1)
        error = float(total.sub(1.0).abs().max())
        return long_book, {
            "max_pre_scale_weight_sum_abs_error": error,
            "max_within_sleeve_group_budget_abs_error": 0.0,
            "minimum_available_groups_in_sleeve": np.nan,
            "max_weight_sum_abs_error": error,
            "max_top_level_sleeve_budget_abs_error": 0.0,
            "max_long_exposure_abs_error": error,
            "max_short_exposure_abs_error": 0.0,
            "max_net_exposure_abs_error": error,
            "max_gross_exposure_abs_error": float(
                long_book.abs().sum(axis=1).sub(1.0).abs().max()
            ),
            "max_overlap_assets_removed": 0,
        }

    short_book = side(-scores)
    overlap = long_book.gt(0.0) & short_book.gt(0.0)
    long_book = long_book.mask(overlap, 0.0)
    short_book = short_book.mask(overlap, 0.0)
    long_book = long_book.div(long_book.sum(axis=1), axis=0)
    short_book = short_book.div(short_book.sum(axis=1), axis=0)
    weights = long_book - short_book
    return weights, {
        "max_pre_scale_weight_sum_abs_error": 0.0,
        "max_within_sleeve_group_budget_abs_error": 0.0,
        "minimum_available_groups_in_sleeve": np.nan,
        "max_weight_sum_abs_error": float(weights.sum(axis=1).abs().max()),
        "max_top_level_sleeve_budget_abs_error": 0.0,
        "max_long_exposure_abs_error": float(
            long_book.sum(axis=1).sub(1.0).abs().max()
        ),
        "max_short_exposure_abs_error": float(
            short_book.sum(axis=1).sub(1.0).abs().max()
        ),
        "max_net_exposure_abs_error": float(weights.sum(axis=1).abs().max()),
        "max_gross_exposure_abs_error": float(
            weights.abs().sum(axis=1).sub(2.0).abs().max()
        ),
        "max_overlap_assets_removed": int(overlap.sum(axis=1).max()),
    }


def _performance_payload(net, gross, ew_nav: pd.Series) -> dict[str, float]:
    """Return accepted performance metrics plus explicit pre-cost annual return."""
    payload = e5._performance_row(net, gross, ew_nav)
    payload["gross_return_annualized"] = (
        payload["net_return_annualized"]
        + payload["cost_drag_bp_per_year"] / 10000.0
    )
    return payload


def _allocation_rows(
    strategy: str,
    method: str,
    q: float,
    weights: pd.DataFrame,
    sleeve_panel: pd.DataFrame,
    target: Mapping[str, float] | None = TARGET,
) -> list[dict]:
    """Summarize signed decision-date exposure by broad sleeve."""
    rows = []
    for sleeve in SLEEVES:
        long_exposure = weights.clip(lower=0.0).where(
            sleeve_panel.eq(sleeve), 0.0
        ).sum(axis=1)
        short_exposure = (-weights.clip(upper=0.0)).where(
            sleeve_panel.eq(sleeve), 0.0
        ).sum(axis=1)
        rows.append(
            {
                "strategy": strategy,
                "method": method,
                "q": q,
                "sleeve": sleeve,
                "target_budget": np.nan if target is None else target[sleeve],
                "mean_long_exposure": float(long_exposure.mean()),
                "mean_short_exposure_abs": float(short_exposure.mean()),
                "mean_net_exposure": float((long_exposure - short_exposure).mean()),
                "std_net_exposure": float((long_exposure - short_exposure).std()),
                "min_net_exposure": float((long_exposure - short_exposure).min()),
                "max_net_exposure": float((long_exposure - short_exposure).max()),
            }
        )
    return rows


def _run_leg(
    *,
    strategy: str,
    method: str,
    q: float,
    prices: pd.DataFrame,
    weights: pd.DataFrame,
    diagnostics: Mapping[str, float],
    sleeve_panel: pd.DataFrame,
    ew_nav: pd.Series,
    costs: float,
    target: Mapping[str, float] | None = TARGET,
    runner: str = RUNNER,
) -> tuple[dict, dict, list[dict]]:
    """Backtest one portfolio and return performance, acceptance, and exposures."""
    net, gross = _backtest(
        prices, weights, costs, f"futures_{strategy}_{method}_q_{q:.2f}"
    )
    performance = {
        "universe": UNIVERSE.value,
        "analysis_window": "full_panel",
        "strategy": strategy,
        "method": method,
        "q": q,
        **_performance_payload(net, gross, ew_nav),
        "runner": runner,
    }
    ordinary_errors = [
        abs(float(value))
        for key, value in diagnostics.items()
        if key.startswith("max_")
        and key.endswith("error")
        and "group_budget" not in key
    ]
    group_error = float(
        diagnostics["max_within_sleeve_group_budget_abs_error"]
    )
    passed = (
        max(ordinary_errors) <= EXPOSURE_TOLERANCE
        and group_error <= GROUP_BUDGET_TOLERANCE
    )
    acceptance = {
        "strategy": strategy,
        "method": method,
        "q": q,
        **diagnostics,
        "exposure_tolerance": EXPOSURE_TOLERANCE,
        "weight_tolerance": WEIGHT_TOLERANCE,
        "group_budget_tolerance": GROUP_BUDGET_TOLERANCE,
        "status": "PASS" if passed else "FAIL",
    }
    return (
        performance,
        acceptance,
        _allocation_rows(strategy, method, q, weights, sleeve_panel, target),
    )


def _comparison(performance: pd.DataFrame) -> pd.DataFrame:
    """Compare cluster legs with same-budget and original global ranks."""
    sleeve_global = performance.loc[
        performance["method"].eq("sleeve_global")
    ].set_index(["strategy", "q"])
    original = performance.loc[
        performance["method"].eq("original_global")
    ].set_index(["strategy", "q"])
    rows = []
    clusters = performance.loc[
        performance["method"].str.startswith("sleeve_cluster")
    ]
    for _, cluster in clusters.iterrows():
        key = (cluster["strategy"], cluster["q"])
        fair = sleeve_global.loc[key]
        legacy = original.loc[key]
        row = cluster.to_dict()
        for metric in COMPARISON_METRICS:
            row[f"sleeve_global_{metric}"] = fair[metric]
            row[f"delta_vs_sleeve_global_{metric}"] = cluster[metric] - fair[metric]
            row[f"original_global_{metric}"] = legacy[metric]
            row[f"delta_vs_original_global_{metric}"] = cluster[metric] - legacy[metric]
        row["beats_sleeve_global_net_return"] = (
            row["delta_vs_sleeve_global_net_return_annualized"] > 0.0
        )
        row["beats_original_global_net_return"] = (
            row["delta_vs_original_global_net_return_annualized"] > 0.0
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["strategy", "q", "method"])


def _global_exposure_diagnostic(
    weights: pd.DataFrame, sleeves: pd.Series
) -> pd.DataFrame:
    """Measure the accepted q=20% global rank's strategic concentration."""
    exposure = weights.T.groupby(sleeves).sum().T.reindex(columns=SLEEVES)
    selected = weights.gt(0.0).T.groupby(sleeves).sum().T.reindex(columns=SLEEVES)
    trigger = bool(
        exposure.mean().sub(0.25).abs().max() > EXPOSURE_TRIGGER_TOLERANCE
        or exposure.eq(0.0).any().any()
    )
    rows = []
    counts = sleeves.value_counts()
    for sleeve in SLEEVES:
        rows.append(
            {
                "sleeve": sleeve,
                "universe_contracts": int(counts[sleeve]),
                "mean_weight": float(exposure[sleeve].mean()),
                "std_weight": float(exposure[sleeve].std()),
                "min_weight": float(exposure[sleeve].min()),
                "max_weight": float(exposure[sleeve].max()),
                "mean_selected_assets": float(selected[sleeve].mean()),
                "min_selected_assets": int(selected[sleeve].min()),
                "max_selected_assets": int(selected[sleeve].max()),
                "empty_dates": int(exposure[sleeve].eq(0.0).sum()),
                "mean_deviation_from_equal": float(
                    abs(exposure[sleeve].mean() - 0.25)
                ),
                "trigger_tolerance": EXPOSURE_TRIGGER_TOLERANCE,
                "equal_sleeve_trigger": trigger,
            }
        )
    return pd.DataFrame(rows)


def _global_regression(
    weights: pd.DataFrame, performance: pd.DataFrame
) -> pd.DataFrame:
    """Require primary long-only global weights and payoffs to match accepted E5."""
    accepted_weights = pd.read_csv(
        _accepted_root() / "weights.csv",
        parse_dates=["index"],
        float_precision="round_trip",
    )
    accepted_weights = accepted_weights.loc[
        accepted_weights["leg"].eq("global")
    ].set_index("index").drop(columns="leg")
    accepted_weights = accepted_weights.reindex_like(weights)
    weight_error = float((weights - accepted_weights).abs().to_numpy().max())
    accepted_performance = pd.read_csv(
        _accepted_root() / "performance.csv", float_precision="round_trip"
    ).set_index("leg").loc["global"]
    current = performance.loc[
        performance["strategy"].eq("long_only")
        & performance["method"].eq("original_global")
        & performance["q"].eq(PRIMARY_Q)
    ].iloc[0]
    performance_error = max(
        abs(float(current[metric]) - float(accepted_performance[metric]))
        for metric in COMPARISON_METRICS
        if metric != "gross_return_annualized"
    )
    maximum = max(weight_error, performance_error)
    return pd.DataFrame(
        [
            {
                "check": "accepted q=0.20 futures global long-only regression",
                "max_weight_abs_error": weight_error,
                "max_performance_abs_error": performance_error,
                "measured_max_abs_error": maximum,
                "tolerance": 1e-12,
                "status": "PASS" if maximum <= 1e-12 else "FAIL",
            }
        ]
    )


def _design(dates: pd.DatetimeIndex, sleeves: pd.Series) -> pd.DataFrame:
    """Return the frozen machine-readable experiment design."""
    spec = e5.get_universe_spec(UNIVERSE)
    return pd.DataFrame(
        [
            {
                "universe": UNIVERSE.value,
                "contracts": len(sleeves),
                "decision_dates": len(dates),
                "decision_start": dates.min(),
                "decision_end": dates.max(),
                "signal": "48-week log-return sum excluding latest 4 weeks",
                "primary_q": PRIMARY_Q,
                "robustness_q": QUANTILES[1],
                "cost_bps": spec.cost_bps,
                "implementation_lag": 1,
                "target_per_sleeve": 0.25,
                "configs": "baseline|M1_star",
                "returns_convention": data_convention(spec),
                "runner": RUNNER,
            }
        ]
    )


def data_convention(spec) -> str:
    """Return the frozen futures return convention as a compact label."""
    return "simple NAV returns built from W-WED log returns; non-excess"


def run() -> Mapping[str, pd.DataFrame]:
    """Execute the accepted-global diagnostic and fair four-sleeve comparison."""
    started = time.perf_counter()
    data = e5.load_universe(UNIVERSE)
    dates = e5.load_cached(UNIVERSE, e5.SmootherName.BASELINE).dates
    eligibility = e5._investable_eligibility(data, dates)
    columns = eligibility.columns
    scores = e5._raw_momentum_scores(
        data, dates, vol_adjusted=False
    ).reindex(columns=columns).where(eligibility)
    prices = e5._prices(data).reindex(columns=columns)
    sleeves = _broad_sleeves(data.taxonomy, columns)
    sleeve_panel = _sleeve_panel(dates, sleeves)
    accepted_navs = pd.read_csv(
        _accepted_root() / "navs.csv",
        parse_dates=["date"],
        float_precision="round_trip",
    ).set_index("date")
    ew_nav = accepted_navs["EW_all"]
    costs = e5.get_universe_spec(UNIVERSE).cost_bps / 10000.0
    cluster_groups = {
        config: _hierarchical_groups(
            e5._cluster_groups(UNIVERSE, config).reindex(
                index=dates, columns=columns
            ),
            sleeve_panel,
        )
        for config in CONFIGS
    }

    performance_rows = []
    acceptance_rows = []
    allocation_rows = []
    primary_global_weights = None
    for q in QUANTILES:
        for strategy in ("long_only", "long_short"):
            original_weights, diagnostics = _original_global_weights(
                strategy, scores, eligibility, q
            )
            if strategy == "long_only" and q == PRIMARY_Q:
                primary_global_weights = original_weights
            performance, acceptance, allocation = _run_leg(
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
            )
            performance_rows.append(performance)
            acceptance_rows.append(acceptance)
            allocation_rows.extend(allocation)

            if strategy == "long_only":
                sleeve_weights, diagnostics = _long_only_sleeve_weights(
                    scores, eligibility, sleeve_panel, sleeve_panel, q
                )
            else:
                sleeve_weights, diagnostics = _long_short_sleeve_weights(
                    scores, eligibility, sleeve_panel, sleeve_panel, q
                )
            performance, acceptance, allocation = _run_leg(
                strategy=strategy,
                method="sleeve_global",
                q=q,
                prices=prices,
                weights=sleeve_weights,
                diagnostics=diagnostics,
                sleeve_panel=sleeve_panel,
                ew_nav=ew_nav,
                costs=costs,
            )
            performance_rows.append(performance)
            acceptance_rows.append(acceptance)
            allocation_rows.extend(allocation)

            for config in CONFIGS:
                if strategy == "long_only":
                    weights, diagnostics = _long_only_sleeve_weights(
                        scores,
                        eligibility,
                        sleeve_panel,
                        cluster_groups[config],
                        q,
                    )
                else:
                    weights, diagnostics = _long_short_sleeve_weights(
                        scores,
                        eligibility,
                        sleeve_panel,
                        cluster_groups[config],
                        q,
                    )
                method = f"sleeve_cluster_{config.value}"
                performance, acceptance, allocation = _run_leg(
                    strategy=strategy,
                    method=method,
                    q=q,
                    prices=prices,
                    weights=weights,
                    diagnostics=diagnostics,
                    sleeve_panel=sleeve_panel,
                    ew_nav=ew_nav,
                    costs=costs,
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
    regression = _global_regression(primary_global_weights, performance)
    if not regression["status"].eq("PASS").all():
        raise AssertionError(regression)
    outputs = {
        "design": _design(dates, sleeves),
        "global_exposure_diagnostic": _global_exposure_diagnostic(
            primary_global_weights, sleeves
        ),
        "performance": performance,
        "comparison": _comparison(performance),
        "allocation_diagnostics": pd.DataFrame(allocation_rows),
        "acceptance": acceptance,
        "global_regression": regression,
    }
    for name, frame in outputs.items():
        e5._write(frame, _root() / f"{name}.csv")
    runtime = pd.DataFrame(
        [
            {
                "portfolios": len(performance),
                "runtime_seconds": time.perf_counter() - started,
            }
        ]
    )
    e5._write(runtime, _root() / "runtime.csv")
    return outputs


def _hash_outputs() -> dict[str, str]:
    """Hash numerical outputs while excluding timing and replay records."""
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(_root().glob("*.csv"))
        if path.name not in {"runtime.csv", "determinism.csv"}
    }


def verify_determinism() -> pd.DataFrame:
    """Replay the full four-sleeve experiment and require identical CSV bytes."""
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
    """Run and replay the futures four-sleeve experiment."""
    replay = verify_determinism()
    print(
        f"Futures four-sleeve grid: PASS ({len(replay)}/{len(replay)} deterministic)",
        flush=True,
    )


if __name__ == "__main__":
    main()
