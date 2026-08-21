"""Run the E5b group-equal grouped-ranking construction and E6 payoff addendum.

Every taxonomy or cluster group that has at least one eligible asset with a valid score
receives budget ``1 / G``.  Selected assets within that group split the budget equally.
The global leg and EW-all reference retain their accepted construction.  For Metric 11,
``w_tilde`` is built with this same group-equal construction under the prior-date
partition, including the prior partition's available-group count; group-count changes
therefore enter the reassignment component.  The accepted asset-equal outputs are retained
unchanged and are emitted alongside group-equal results as a labelled robustness variant.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd

import qis

import papers.cluster_lineage_2026.replication.run_backtests as e5
import papers.cluster_lineage_2026.replication.run_inference as e6


CONSTRUCTION = "group_equal"
ASSET_EQUAL = "asset_equal"
RUNNER = "papers/cluster_lineage_2026/replication/run_e5b.py"
PRIMARY_Q = 0.20
ROBUSTNESS = (("momentum_q_1_3", 1.0 / 3.0), ("momentum_vol_adj", 0.20))
WEIGHT_TOLERANCE = 1e-12
GROUP_BUDGET_TOLERANCE = 1e-15


def _root() -> Path:
    """Return the local E5b output root."""
    return e5.get_output_path("e5b", create=True)


def _universe_root(universe: e5.UniverseName | str) -> Path:
    """Return and create one universe's group-equal output directory."""
    universe = e5.UniverseName(universe)
    root = _root() / CONSTRUCTION / universe.value
    root.mkdir(parents=True, exist_ok=True)
    return root


def _accepted_root(universe: e5.UniverseName | str) -> Path:
    """Return the accepted asset-equal E5 output directory."""
    universe = e5.UniverseName(universe)
    return e5.get_output_path("backtests", universe.value)


def _write(frame: pd.DataFrame, path: Path) -> None:
    """Write a deterministic CSV using the accepted E5 writer."""
    e5._write(frame, path)


def _date_key(dates: pd.DatetimeIndex) -> tuple[int, pd.Timestamp, pd.Timestamp]:
    """Return a stable identity for an analysis-window date index."""
    return len(dates), pd.Timestamp(dates[0]), pd.Timestamp(dates[-1])


def _group_equal_from_ranks(
    ranks: pd.DataFrame,
    eligibility: pd.DataFrame,
    groups: pd.DataFrame,
    q: float,
    universe: e5.UniverseName,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Convert accepted within-group selections to exact group-equal asset weights."""
    ranks = ranks.copy()
    groups = groups.reindex(index=ranks.index, columns=ranks.columns)
    eligibility = eligibility.reindex_like(ranks).fillna(False).astype(bool)

    # This call freezes the accepted selection rule, including the U3 QE quarterly
    # reselection/hold convention.  Only the budget allocation below changes.
    asset_equal = e5._weights_from_ranks(ranks, eligibility, q, universe)
    selected = asset_equal.gt(0.0)
    weights = pd.DataFrame(0.0, index=ranks.index, columns=ranks.columns)
    group_counts = pd.Series(0, index=ranks.index, dtype=int, name="available_group_count")
    validation_rows: list[dict] = []

    for date in ranks.index:
        labels = groups.loc[date]
        mask = selected.loc[date] & labels.notna()
        selected_labels = labels.loc[mask]
        counts = selected_labels.value_counts(dropna=True, sort=False)
        group_count = int(len(counts))
        group_counts.loc[date] = group_count
        if group_count:
            row = weights.loc[date]
            for label, count in counts.items():
                assets = selected_labels.index[selected_labels.eq(label)]
                row.loc[assets] = 1.0 / (group_count * int(count))
            weights.loc[date] = row
            budgets = weights.loc[date].groupby(labels, dropna=True).sum()
            budgets = budgets.loc[budgets.gt(0.0)]
            budget_error = float((budgets - 1.0 / group_count).abs().max())
        else:
            budget_error = 0.0
        weight_sum = float(weights.loc[date].sum())
        validation_rows.append(
            {
                "date": date,
                "available_group_count": group_count,
                "weight_sum": weight_sum,
                "weight_sum_abs_error": abs(weight_sum - 1.0),
                "max_group_budget_abs_error": budget_error,
                "weight_status": "PASS"
                if abs(weight_sum - 1.0) <= WEIGHT_TOLERANCE
                else "FAIL",
                "group_budget_status": "PASS"
                if budget_error <= GROUP_BUDGET_TOLERANCE
                else "FAIL",
            }
        )

    validation = pd.DataFrame(validation_rows)
    return weights, group_counts, validation


def _taxonomy_panel(data, dates: pd.DatetimeIndex, columns: pd.Index) -> pd.DataFrame:
    """Return the date-tiled accepted taxonomy panel."""
    taxonomy = e5._taxonomy_groups(data, columns)
    return pd.DataFrame(
        np.tile(taxonomy.to_numpy(), (len(dates), 1)),
        index=dates,
        columns=columns,
    )


def _grouped_weights(
    data,
    dates: pd.DatetimeIndex,
    scores: pd.DataFrame,
    eligibility: pd.DataFrame,
    q: float,
    configs: Tuple[e5.SmootherName, ...],
) -> tuple[
    Dict[str, pd.DataFrame],
    Dict[str, pd.DataFrame],
    Dict[str, pd.Series],
    Dict[str, pd.DataFrame],
]:
    """Build group-equal taxonomy/cluster weights and prior-partition counterfactuals."""
    weights: Dict[str, pd.DataFrame] = {}
    counterfactuals: Dict[str, pd.DataFrame] = {}
    counts: Dict[str, pd.Series] = {}
    validations: Dict[str, pd.DataFrame] = {}

    taxonomy_groups = _taxonomy_panel(data, dates, scores.columns)
    taxonomy_ranks = e5._rank_panel(scores, taxonomy_groups)
    taxonomy_weights, taxonomy_counts, taxonomy_validation = _group_equal_from_ranks(
        taxonomy_ranks, eligibility, taxonomy_groups, q, data.name
    )
    weights["taxonomy"] = taxonomy_weights
    counts["taxonomy"] = taxonomy_counts
    validations["taxonomy"] = taxonomy_validation

    for config in configs:
        groups = e5._cluster_groups(data.name, config).reindex(
            index=dates, columns=scores.columns
        )
        name = f"cluster_{config.value}"
        ranks = e5._rank_panel(scores, groups)
        current, group_counts, validation = _group_equal_from_ranks(
            ranks, eligibility, groups, q, data.name
        )
        weights[name] = current
        counts[name] = group_counts
        validations[name] = validation

        prior_groups = groups.shift(1)
        prior_ranks = e5._rank_panel(scores, prior_groups)
        prior, _, _ = _group_equal_from_ranks(
            prior_ranks, eligibility, prior_groups, q, data.name
        )
        counterfactuals[name] = prior

    return weights, counterfactuals, counts, validations


def _builder_with_collector(
    window_names: Mapping[tuple[int, pd.Timestamp, pd.Timestamp], str],
    collector: dict[str, tuple[Dict[str, pd.Series], Dict[str, pd.DataFrame]]],
):
    """Return an accepted-runner-compatible builder with primary diagnostics capture."""

    def build(data, dates, scores, eligibility, q, configs):
        global_groups = pd.DataFrame("global", index=dates, columns=scores.columns)
        global_weights = e5._weights_from_ranks(
            e5._rank_panel(scores, global_groups), eligibility, q, data.name
        )
        grouped, counterfactuals, counts, validations = _grouped_weights(
            data, dates, scores, eligibility, q, configs
        )
        weights = {"global": global_weights, **grouped}
        ew_all = eligibility.astype(float).div(
            eligibility.sum(axis=1).replace(0, np.nan), axis=0
        ).fillna(0.0)
        weights["EW_all"] = ew_all
        key = _date_key(dates)
        window = window_names.get(key, "full_panel")
        collector.setdefault(window, (counts, validations))
        return weights, counterfactuals

    return build


def _diagnostic_tables(
    universe: e5.UniverseName,
    captured: Mapping[str, tuple[Dict[str, pd.Series], Dict[str, pd.DataFrame]]],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Assemble per-date group counts, standard deviations, and weight validations."""
    count_rows: list[pd.DataFrame] = []
    validation_rows: list[pd.DataFrame] = []
    for window, (counts, validations) in captured.items():
        for leg, series in counts.items():
            count_rows.append(
                pd.DataFrame(
                    {
                        "universe": universe.value,
                        "analysis_window": window,
                        "date": series.index,
                        "leg": leg,
                        "available_group_count": series.to_numpy(),
                        "construction": CONSTRUCTION,
                    }
                )
            )
            frame = validations[leg].copy()
            frame.insert(0, "leg", leg)
            frame.insert(0, "analysis_window", window)
            frame.insert(0, "universe", universe.value)
            frame["construction"] = CONSTRUCTION
            validation_rows.append(frame)
    per_date = pd.concat(count_rows, ignore_index=True)
    summary = (
        per_date.groupby(["universe", "analysis_window", "leg"], sort=False)[
            "available_group_count"
        ]
        .agg(["count", "mean", "std", "min", "max"])
        .reset_index()
        .rename(columns={"std": "available_group_count_std"})
    )
    summary["construction"] = CONSTRUCTION
    validation = pd.concat(validation_rows, ignore_index=True)
    return per_date, summary, validation


def _label_construction(frame: pd.DataFrame) -> pd.DataFrame:
    """Label a table as group-equal without changing index-like numerical panels."""
    frame = frame.copy()
    if "construction" not in frame.columns:
        position = 2 if "analysis_window" in frame.columns else 1
        frame.insert(min(position, len(frame.columns)), "construction", CONSTRUCTION)
    return frame


def _construction_comparison(
    universe: e5.UniverseName, group_payoff: pd.DataFrame
) -> pd.DataFrame:
    """Combine accepted asset-equal and new group-equal clarified payoff rows."""
    accepted = pd.read_csv(
        _accepted_root(universe) / "payoff_comparison.csv", float_precision="round_trip"
    )
    accepted = accepted.copy()
    accepted.insert(
        min(2, len(accepted.columns)), "construction", ASSET_EQUAL
    )
    group_payoff = _label_construction(group_payoff)
    return pd.concat([accepted, group_payoff], ignore_index=True, sort=False)


def _run_u1() -> Mapping[str, pd.DataFrame]:
    """Run the full accepted U1 E5 workflow with group-equal grouped legs."""
    universe = e5.UniverseName.MSCI_US
    dates = e5.load_cached(universe, e5.SmootherName.BASELINE).dates
    windows = e5._analysis_windows(universe, dates)
    window_names = {_date_key(index): name for name, index in windows.items()}
    captured: dict[str, tuple[Dict[str, pd.Series], Dict[str, pd.DataFrame]]] = {}

    module = e5._executed
    original_builder = module._build_leg_weights
    original_output_dir = module._output_dir
    original_runner = module.RUNNER
    module._build_leg_weights = _builder_with_collector(window_names, captured)
    module._output_dir = lambda _: _universe_root(universe)
    module.RUNNER = RUNNER
    try:
        output = module.run_universe(universe)
    finally:
        module._build_leg_weights = original_builder
        module._output_dir = original_output_dir
        module.RUNNER = original_runner

    root = _universe_root(universe)
    labelled_names = {
        "performance",
        "alpha_rank_analysis",
        "turnover_decomposition",
        "turnover_decomposition_per_date",
        "crisis_windows",
        "robustness",
        "score_identity",
        "payoff_comparison",
        "ew_reference",
    }
    for name in labelled_names.intersection(output):
        output[name] = _label_construction(output[name])
        _write(output[name], root / f"{name}.csv")

    per_date, summary, validation = _diagnostic_tables(universe, captured)
    output["group_count_per_date"] = per_date
    output["group_count_summary"] = summary
    output["weight_validation"] = validation
    output["construction_comparison"] = _construction_comparison(
        universe, output["payoff_comparison"]
    )
    for name in (
        "group_count_per_date",
        "group_count_summary",
        "weight_validation",
        "construction_comparison",
    ):
        _write(output[name], root / f"{name}.csv")
    return output


def _read_accepted_weights(universe: e5.UniverseName, leg: str) -> pd.DataFrame:
    """Read one accepted weight panel without rerunning its portfolio."""
    frame = pd.read_csv(_accepted_root(universe) / "weights.csv")
    date_column = "index" if "index" in frame.columns else "date"
    frame[date_column] = pd.to_datetime(frame[date_column])
    selected = frame.loc[frame["leg"].eq(leg)].set_index(date_column)
    drop = [column for column in ("leg", "analysis_window") if column in selected.columns]
    return selected.drop(columns=drop).sort_index()


def _accepted_navs(universe: e5.UniverseName) -> pd.DataFrame:
    """Read accepted global/EW NAVs for reuse without rerunning those legs."""
    frame = pd.read_csv(_accepted_root(universe) / "navs.csv", float_precision="round_trip")
    frame["date"] = pd.to_datetime(frame["date"])
    return frame.set_index("date").sort_index()


def _accepted_rows(universe: e5.UniverseName, name: str) -> pd.DataFrame:
    """Read an accepted E5 table and normalize its window label."""
    frame = pd.read_csv(_accepted_root(universe) / f"{name}.csv", float_precision="round_trip")
    if "analysis_window" not in frame.columns:
        frame.insert(1 if "universe" in frame.columns else 0, "analysis_window", "full_panel")
    return frame


def _backtest_grouped(
    universe: e5.UniverseName,
    prices: pd.DataFrame,
    weights: Mapping[str, pd.DataFrame],
    costs: float,
) -> tuple[dict, dict]:
    """Backtest only grouped legs, leaving global and EW accepted artifacts untouched."""
    net = {}
    gross = {}
    for leg, frame in weights.items():
        net[leg] = qis.backtest_model_portfolio(
            prices=prices,
            weights=frame,
            rebalancing_freq=None,
            rebalancing_costs=costs,
            weight_implementation_lag=1,
            ticker=leg,
        )
        gross[leg] = qis.backtest_model_portfolio(
            prices=prices,
            weights=frame,
            rebalancing_freq=None,
            rebalancing_costs=None,
            weight_implementation_lag=1,
            ticker=f"{leg}_gross",
        )
    return net, gross


def _run_u2_u3(universe: e5.UniverseName) -> Mapping[str, pd.DataFrame]:
    """Rerun only U2/U3 grouped legs and reuse accepted global/EW artifacts."""
    dates = e5.load_cached(universe, e5.SmootherName.BASELINE).dates
    data = e5.load_universe(universe)
    prices = e5._prices(data)
    eligibility = e5._investable_eligibility(data, dates)
    prices = prices.reindex(columns=eligibility.columns)
    scores = e5._raw_momentum_scores(data, dates, vol_adjusted=False).reindex(
        columns=eligibility.columns
    ).where(eligibility)
    configs = e5.IN_BAND[universe]
    weights, counterfactuals, counts, validations = _grouped_weights(
        data, dates, scores, eligibility, PRIMARY_Q, configs
    )
    costs = e5.get_universe_spec(universe).cost_bps / 10000.0
    net, gross = _backtest_grouped(universe, prices, weights, costs)

    accepted_performance = _accepted_rows(universe, "performance")
    accepted_nvs = _accepted_navs(universe)
    benchmark_nav = accepted_nvs["EW_all"]
    rows = []
    for leg, portfolio in net.items():
        rows.append(
            {
                "universe": universe.value,
                "analysis_window": "full_panel",
                "leg": leg,
                **e5._performance_row(portfolio, gross[leg], benchmark_nav),
                "runner": RUNNER,
                "construction": CONSTRUCTION,
            }
        )
    grouped_performance = pd.DataFrame(rows)
    references = accepted_performance.loc[
        accepted_performance["leg"].isin(["global", "EW_all"])
    ].copy()
    references["construction"] = ASSET_EQUAL
    performance = pd.concat([references, grouped_performance], ignore_index=True, sort=False)

    decomposition_rows = []
    decomposition_panels = []
    for leg, prior_weights in counterfactuals.items():
        targets = weights[leg]
        drifted = e5._drifted_prior_weights(targets, prices)
        summary, panel = e5.turnover_decomposition(targets, prior_weights, drifted)
        residual_share = abs(summary["turnover_residual"]) / summary["total_turnover"]
        decomposition_rows.append(
            {
                "universe": universe.value,
                "analysis_window": "full_panel",
                "construction": CONSTRUCTION,
                "leg": leg,
                **summary,
                "trade_interaction_turnover": summary["turnover_residual"],
                "absolute_residual_share": residual_share,
                "residual_guard_status": "RETIRED_NOT_AN_ACCEPTANCE_CRITERION",
            }
        )
        panel = panel.reset_index()
        panel.insert(0, "leg", leg)
        panel.insert(0, "construction", CONSTRUCTION)
        panel.insert(0, "analysis_window", "full_panel")
        panel.insert(0, "universe", universe.value)
        decomposition_panels.append(panel)

    accepted_robustness = _accepted_rows(universe, "robustness")
    robustness_rows = []
    global_robustness = accepted_robustness.loc[
        accepted_robustness["leg"].eq("global")
    ].copy()
    global_robustness["construction"] = ASSET_EQUAL
    robustness_rows.extend(global_robustness.to_dict("records"))
    headline = (
        "taxonomy",
        "cluster_baseline",
        f"cluster_{e5.BEST[universe].value}",
    )
    for variant, q in ROBUSTNESS:
        variant_scores = e5._raw_momentum_scores(
            data, dates, vol_adjusted=variant == "momentum_vol_adj"
        ).reindex(columns=eligibility.columns).where(eligibility)
        variant_weights, _, _, _ = _grouped_weights(
            data,
            dates,
            variant_scores,
            eligibility,
            q,
            (e5.SmootherName.BASELINE, e5.BEST[universe]),
        )
        selected = {leg: variant_weights[leg] for leg in headline}
        variant_net, variant_gross = _backtest_grouped(universe, prices, selected, costs)
        for leg in headline:
            robustness_rows.append(
                {
                    "universe": universe.value,
                    "analysis_window": "full_panel",
                    "construction": CONSTRUCTION,
                    "variant": variant,
                    "q": q,
                    "leg": leg,
                    **e5._performance_row(
                        variant_net[leg], variant_gross[leg], benchmark_nav
                    ),
                }
            )

    navs = pd.concat(
        [
            accepted_nvs[["global", "EW_all"]],
            *[
                portfolio.get_portfolio_nav().rename(leg)
                for leg, portfolio in net.items()
            ],
        ],
        axis=1,
    ).reset_index()

    accepted_global_weights = _read_accepted_weights(universe, "global")
    weight_columns = accepted_global_weights.columns.union(eligibility.columns, sort=False)
    weight_frames = [
        accepted_global_weights.reindex(columns=weight_columns, fill_value=0.0)
        .assign(leg="global")
        .reset_index()
    ]
    for leg, frame in weights.items():
        weight_frames.append(
            frame.reindex(columns=weight_columns, fill_value=0.0)
            .assign(leg=leg)
            .reset_index()
        )
    weights_long = pd.concat(weight_frames, ignore_index=True, sort=False)

    captured = {"full_panel": (counts, validations)}
    group_count, group_summary, weight_validation = _diagnostic_tables(universe, captured)
    performance = performance.sort_values(
        "leg", key=lambda values: values.map(
            {leg: index for index, leg in enumerate(["global", "taxonomy", *[f"cluster_{c.value}" for c in configs], "EW_all"])}
        )
    ).reset_index(drop=True)
    clarified = e5._clarified_payoff_tables(performance)
    payoff = _label_construction(clarified["payoff_comparison"])
    ew_reference = _label_construction(clarified["ew_reference"])

    output = {
        "performance": performance,
        "turnover_decomposition": pd.DataFrame(decomposition_rows),
        "turnover_decomposition_per_date": pd.concat(
            decomposition_panels, ignore_index=True
        ),
        "robustness": pd.DataFrame(robustness_rows),
        "weights": weights_long,
        "navs": navs,
        "crisis_windows": _label_construction(e5._crisis_rows(navs.set_index("date"))),
        "payoff_comparison": payoff,
        "ew_reference": ew_reference,
        "group_count_per_date": group_count,
        "group_count_summary": group_summary,
        "weight_validation": weight_validation,
        "construction_comparison": _construction_comparison(universe, payoff),
    }
    root = _universe_root(universe)
    for name, frame in output.items():
        _write(frame, root / f"{name}.csv")
    return output


def run_universe(universe: e5.UniverseName | str) -> Mapping[str, pd.DataFrame]:
    """Run E5b for one universe without touching accepted asset-equal artifacts."""
    universe = e5.UniverseName(universe)
    if universe == e5.UniverseName.MSCI_US:
        return _run_u1()
    return _run_u2_u3(universe)


def _group_equal_performance_series(universe: e5.UniverseName):
    """Feed the exact E6 bootstrap engine from the E5b group-equal output root."""
    module = e6._executed
    original_get_output_path = module.get_output_path

    def redirected(*parts, **kwargs):
        if parts and parts[0] == "backtests":
            return _universe_root(parts[1])
        return original_get_output_path(*parts, **kwargs)

    module.get_output_path = redirected
    try:
        return e6._performance_series(universe)
    finally:
        module.get_output_path = original_get_output_path


def run_e6_addendum() -> Mapping[str, pd.DataFrame]:
    """Recompute all frozen payoff-bootstrap contrasts for group-equal grouped legs."""
    module = e6._executed
    original_series = module._performance_series
    module._performance_series = _group_equal_performance_series
    try:
        group_equal = pd.concat(
            [module._e5_bootstrap(universe) for universe in e5.UniverseName],
            ignore_index=True,
        )
    finally:
        module._performance_series = original_series
    group_equal.insert(2, "construction", CONSTRUCTION)

    asset_equal = pd.read_csv(
        e5.get_output_path("inference", "payoff_bootstrap.csv"),
        float_precision="round_trip",
    )
    asset_equal.insert(2, "construction", ASSET_EQUAL)
    combined = pd.concat([asset_equal, group_equal], ignore_index=True, sort=False)

    root = _root() / "e6_addendum"
    root.mkdir(parents=True, exist_ok=True)
    _write(group_equal, root / "payoff_bootstrap_group_equal.csv")
    _write(combined, root / "payoff_bootstrap_all_constructions.csv")
    return {
        "payoff_bootstrap_group_equal": group_equal,
        "payoff_bootstrap_all_constructions": combined,
    }


def _validate_outputs() -> pd.DataFrame:
    """Validate the binding E5b construction and yardstick acceptance conditions."""
    rows = []
    for universe in e5.UniverseName:
        root = _universe_root(universe)
        validation = pd.read_csv(root / "weight_validation.csv")
        payoff = pd.read_csv(root / "payoff_comparison.csv")
        max_sum_error = float(validation["weight_sum_abs_error"].max())
        max_budget_error = float(validation["max_group_budget_abs_error"].max())
        ew_delta_columns = [
            column for column in payoff.columns if "delta_vs_ew" in column.lower()
        ]
        ew_legs = int(payoff["leg"].astype(str).str.lower().eq("ew_all").sum())
        rows.extend(
            [
                {
                    "universe": universe.value,
                    "acceptance_line": "weights_sum_to_one",
                    "measured": max_sum_error,
                    "tolerance": WEIGHT_TOLERANCE,
                    "status": "PASS" if max_sum_error <= WEIGHT_TOLERANCE else "FAIL",
                },
                {
                    "universe": universe.value,
                    "acceptance_line": "equal_group_budget",
                    "measured": max_budget_error,
                    "tolerance": GROUP_BUDGET_TOLERANCE,
                    "status": "PASS"
                    if max_budget_error <= GROUP_BUDGET_TOLERANCE
                    else "FAIL",
                },
                {
                    "universe": universe.value,
                    "acceptance_line": "no_ew_performance_comparison",
                    "measured": len(ew_delta_columns) + ew_legs,
                    "tolerance": 0,
                    "status": "PASS"
                    if not ew_delta_columns and ew_legs == 0
                    else "FAIL",
                },
            ]
        )
    result = pd.DataFrame(rows)
    _write(result, _root() / "acceptance.csv")
    if not result["status"].eq("PASS").all():
        raise AssertionError(result.loc[~result["status"].eq("PASS")])
    return result


def _csv_hashes() -> dict[str, str]:
    """Hash every E5b CSV except the replay record itself."""
    hashes = {}
    for path in sorted(_root().rglob("*.csv")):
        if path.name == "determinism.csv":
            continue
        hashes[str(path.relative_to(_root()))] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def run_all() -> Mapping[str, pd.DataFrame]:
    """Run all E5b universes and the frozen E6 group-equal payoff addendum."""
    output = {}
    for universe in (
        e5.UniverseName.MSCI_US,
        e5.UniverseName.FUTURES,
        e5.UniverseName.MAC,
    ):
        output[universe.value] = run_universe(universe)
    output["e6_addendum"] = run_e6_addendum()
    output["acceptance"] = _validate_outputs()
    return output


def verify_determinism() -> pd.DataFrame:
    """Rerun E5b and assert byte-identical CSV outputs."""
    before = _csv_hashes()
    run_all()
    after = _csv_hashes()
    names = sorted(set(before) | set(after))
    result = pd.DataFrame(
        [
            {
                "artifact": name,
                "first_sha256": before.get(name),
                "second_sha256": after.get(name),
                "byte_identical": before.get(name) == after.get(name),
            }
            for name in names
        ]
    )
    _write(result, _root() / "determinism.csv")
    if not result["byte_identical"].all():
        raise AssertionError(result.loc[~result["byte_identical"]])
    return result


if __name__ == "__main__":
    result = run_all()
    print(result["acceptance"].to_string(index=False))
