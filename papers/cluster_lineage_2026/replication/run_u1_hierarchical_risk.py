"""Evaluate U1 cluster structure in standard risk-allocation methods.

The experiment freezes the U1 headline universe, baseline HCGL covariance snapshots,
ME/span-36 Ward hierarchy, monthly schedule, one-period implementation lag, and 10 bp
one-way costs.  It compares flat ERC, cluster-aware risk budgeting, Ward-HRP,
variance-HERC on the Ward tree, and canonical single-link HRP.  No covariance or factor
model is refitted.

The same Ward clusters also decompose ex-ante risk in the accepted global and cluster
long-short momentum strategies.  EW-all is retained only as the market reference for
beta and alpha; it is never a performance yardstick.
"""
from __future__ import annotations

import hashlib
import os
import pickle
import time
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from factorlasso import compute_clusters_from_corr_matrix
from factorlasso.cluster_smoothing import _iter_correlation_inputs
from optimalportfolios import Constraints
from optimalportfolios import wrapper_risk_budgeting
from optimalportfolios.utils.portfolio_funcs import calculate_diversification_ratio

import papers.cluster_lineage_2026.replication.run_backtests as e5
import papers.cluster_lineage_2026.replication.run_depc1_cluster_comparison as d4
import papers.cluster_lineage_2026.replication.run_depc1_strategy_backtests as d5
import papers.cluster_lineage_2026.replication.run_u1_bics_sector_comparison as u1_bics
import papers.cluster_lineage_2026.replication.run_u1_covar_grid_long_short as u1_ls
from papers.cluster_lineage_2026.replication.hierarchical_risk_allocations import (
    cluster_risk_budget,
    herc_volatility_weights,
    hrp_weights,
    risk_concentration_metrics,
    risk_contribution_summary,
)


RUNNER = "papers/cluster_lineage_2026/replication/run_u1_hierarchical_risk.py"
ALLOCATION_CACHE_VERSION = 1
WINDOW = "headline_20090831_20260630"
METHODS = (
    "flat_erc",
    "cluster_rb_alpha_0_5",
    "cluster_rb_alpha_0",
    "ward_hrp",
    "ward_herc",
    "single_hrp",
)
RISK_BUDGET_EXPONENTS = {
    "flat_erc": 1.0,
    "cluster_rb_alpha_0_5": 0.5,
    "cluster_rb_alpha_0": 0.0,
}
WEIGHT_TOLERANCE = 1e-10
RISK_TOLERANCE = 2e-5
PARTITION_TOLERANCE = 1.0
RISK_COVAR_ROOT = Path(
    os.environ.get(
        "CLUSTER_LINEAGE_OUTPUT_DIR",
        r"C:\Users\artur\OneDrive\analytics\outputs\cluster_lineage_2026",
    )
) / "msci_us" / "baseline"
DEPC1_SIGNAL_PERFORMANCE_PATH = (
    RISK_COVAR_ROOT.parents[1] / "depc1" / "msci_us" / "performance.csv"
)


def _root() -> Path:
    """Return the isolated external U1 hierarchical-risk output root."""
    return e5.get_output_path("risk_allocation", "u1_hierarchical_20260816", create=True)


def _allocation_cache_root() -> Path:
    """Return the cache directory containing one allocation payload per date."""
    root = _root() / "allocation_cache"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _allocation_cache_path(date: pd.Timestamp) -> Path:
    """Return one date's allocation cache path."""
    return _allocation_cache_root() / f"{pd.Timestamp(date):%Y%m%d}.pkl"


def _risk_covar_path(date: pd.Timestamp) -> Path:
    """Return the accepted baseline HCGL covariance snapshot path."""
    return RISK_COVAR_ROOT / f"{pd.Timestamp(date):%Y%m%d}.pkl"


def _sha256(path: Path) -> str:
    """Return a file's SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _same_partition(left: pd.Series, right: pd.Series) -> bool:
    """Return whether two assignments induce identical pairwise memberships."""
    frame = pd.concat([left.rename("left"), right.rename("right")], axis=1).dropna()
    if len(frame) != left.notna().sum() or len(frame) != right.notna().sum():
        return False
    first = frame["left"].to_numpy()
    second = frame["right"].to_numpy()
    return bool(
        np.array_equal(
            first[:, None] == first[None, :],
            second[:, None] == second[None, :],
        )
    )


def _load_date_inputs(
    date: pd.Timestamp,
    inputs: d4.UniverseInputs,
) -> tuple[pd.DataFrame, pd.Series, np.ndarray, dict[str, object]]:
    """Load one frozen covariance and Ward hierarchy with exact-set diagnostics."""
    ward_path = d4._cache_dir(inputs, "raw") / f"{pd.Timestamp(date):%Y%m%d}.pkl"
    with ward_path.open("rb") as stream:
        ward = pickle.load(stream)
    with _risk_covar_path(date).open("rb") as stream:
        snapshot = pickle.load(stream)
    clusters = ward["clusters"].dropna()
    raw_covar = snapshot.get_y_covar()
    asset_set_match = set(clusters.index) == set(raw_covar.index)
    if not asset_set_match:
        missing_cluster = sorted(set(raw_covar.index) - set(clusters.index))
        missing_covar = sorted(set(clusters.index) - set(raw_covar.index))
        raise AssertionError(
            f"{date:%Y-%m-%d} covariance/cluster asset mismatch: "
            f"cluster_missing={missing_cluster[:5]}, covar_missing={missing_covar[:5]}"
        )
    covar = raw_covar.reindex(index=clusters.index, columns=clusters.index)
    frozen = inputs.frozen_panel.loc[date].dropna().reindex(clusters.index)
    partition_match = _same_partition(clusters, frozen)
    if not partition_match:
        raise AssertionError(f"{date:%Y-%m-%d} Ward cache does not match frozen ME/36")
    diagnostics = {
        "asset_set_match": asset_set_match,
        "partition_match": partition_match,
        "assets": len(clusters),
        "clusters": int(clusters.nunique()),
        "ward_path": str(ward_path),
        "risk_covar_path": str(_risk_covar_path(date)),
    }
    return covar, clusters, np.asarray(ward["linkage"], dtype=float), diagnostics


def _risk_budget_weights(
    covar: pd.DataFrame,
    clusters: pd.Series,
    exponent: float,
) -> tuple[pd.Series, pd.Series]:
    """Solve production constrained risk budgeting for one cluster exponent."""
    budget = cluster_risk_budget(clusters, cluster_size_exponent=exponent)
    weights = wrapper_risk_budgeting(
        pd_covar=covar,
        constraints=Constraints(is_long_only=True),
        risk_budget=budget,
    )
    return weights, budget


def _date_risk_rows(
    date: pd.Timestamp,
    methods: Mapping[str, pd.Series],
    covar: pd.DataFrame,
    clusters: pd.Series,
    budgets: Mapping[str, pd.Series],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return per-method risk metrics and full cluster contributions for one date."""
    metric_rows = []
    contribution_rows = []
    values = covar.to_numpy()
    for method, weights in methods.items():
        contributions = risk_contribution_summary(weights, covar, clusters)
        contributions.insert(0, "method", method)
        contributions.insert(0, "date", date)
        contribution_rows.append(contributions)
        metrics = risk_concentration_metrics(contributions)
        grouped_capital = weights.groupby(clusters).sum()
        method_budget = budgets.get(method)
        if method_budget is None:
            target_error = np.nan
        else:
            targets = method_budget.groupby(clusters).sum()
            realised = contributions.set_index("cluster")["risk_share"].reindex(targets.index)
            target_error = float(realised.subtract(targets).abs().max())
        metric_rows.append(
            {
                "date": date,
                "method": method,
                "assets": len(weights),
                "clusters": int(clusters.nunique()),
                "weight_sum": float(weights.sum()),
                "minimum_weight": float(weights.min()),
                "maximum_weight": float(weights.max()),
                "effective_assets": float(1.0 / weights.pow(2.0).sum()),
                "cluster_capital_hhi": float(grouped_capital.pow(2.0).sum()),
                "diversification_ratio": float(
                    calculate_diversification_ratio(weights.to_numpy(), values)
                ),
                "maximum_cluster_risk_target_error": target_error,
                **metrics,
            }
        )
    return pd.DataFrame(metric_rows), pd.concat(contribution_rows, ignore_index=True)


def _compute_date_payload(
    date: pd.Timestamp,
    corr: pd.DataFrame,
    inputs: d4.UniverseInputs,
) -> dict[str, object]:
    """Compute all six allocation methods for one date and persist diagnostics."""
    covar, clusters, ward_linkage, input_diagnostics = _load_date_inputs(date, inputs)
    corr = corr.reindex(index=clusters.index, columns=clusters.index)
    _, single_linkage, _ = compute_clusters_from_corr_matrix(
        corr,
        cutoff_fraction=inputs.model.cutoff_fraction,
        linkage_method="single",
        distance_transform=inputs.model.distance_transform,
        n_clusters=None,
    )
    flat_erc = wrapper_risk_budgeting(
        pd_covar=covar,
        constraints=Constraints(is_long_only=True),
        risk_budget=None,
    )
    alpha_one, flat_budget = _risk_budget_weights(covar, clusters, 1.0)
    square_root, square_root_budget = _risk_budget_weights(covar, clusters, 0.5)
    cluster_equal, cluster_equal_budget = _risk_budget_weights(covar, clusters, 0.0)
    methods = {
        "flat_erc": flat_erc,
        "cluster_rb_alpha_0_5": square_root,
        "cluster_rb_alpha_0": cluster_equal,
        "ward_hrp": hrp_weights(covar, ward_linkage),
        "ward_herc": herc_volatility_weights(
            covar, ward_linkage, clusters=clusters
        ),
        "single_hrp": hrp_weights(covar, single_linkage),
    }
    budgets = {
        "flat_erc": flat_budget,
        "cluster_rb_alpha_0_5": square_root_budget,
        "cluster_rb_alpha_0": cluster_equal_budget,
    }
    risk_metrics, contributions = _date_risk_rows(
        date, methods, covar, clusters, budgets
    )
    diagnostics = {
        "date": date,
        **input_diagnostics,
        "flat_erc_vs_alpha_one_max_abs_weight_error": float(
            flat_erc.subtract(alpha_one).abs().max()
        ),
        "maximum_weight_sum_error": max(
            abs(float(weights.sum()) - 1.0) for weights in methods.values()
        ),
        "minimum_method_weight": min(
            float(weights.min()) for weights in methods.values()
        ),
        "maximum_risk_reconciliation_error": float(
            risk_metrics["cluster_risk_reconciliation_error"].max()
        ),
        "maximum_rb_target_error": float(
            risk_metrics["maximum_cluster_risk_target_error"].dropna().max()
        ),
    }
    payload = {
        "version": ALLOCATION_CACHE_VERSION,
        "date": pd.Timestamp(date),
        "methods": methods,
        "risk_metrics": risk_metrics,
        "contributions": contributions,
        "diagnostics": diagnostics,
    }
    path = _allocation_cache_path(date)
    temporary = path.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        pickle.dump(payload, stream, protocol=pickle.HIGHEST_PROTOCOL)
    temporary.replace(path)
    return payload


def _valid_allocation_cache(date: pd.Timestamp) -> bool:
    """Return whether one date has a complete versioned allocation cache."""
    path = _allocation_cache_path(date)
    if not path.exists():
        return False
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    return (
        payload.get("version") == ALLOCATION_CACHE_VERSION
        and pd.Timestamp(payload.get("date")) == pd.Timestamp(date)
        and tuple(payload.get("methods", {}).keys()) == METHODS
    )


def _load_allocation_cache(date: pd.Timestamp) -> dict[str, object]:
    """Load one already validated allocation cache."""
    with _allocation_cache_path(date).open("rb") as stream:
        return pickle.load(stream)


def _build_allocations(
    inputs: d4.UniverseInputs,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute or load all rolling methods and assemble labelled panels."""
    dates = inputs.dates
    if not all(_valid_allocation_cache(date) for date in dates):
        iterator = _iter_correlation_inputs(
            inputs.returns, list(dates), inputs.model
        )
        produced = set()
        for date, full_corr in iterator:
            date = pd.Timestamp(date)
            if date not in dates:
                continue
            started = time.perf_counter()
            payload = _compute_date_payload(date, full_corr, inputs)
            produced.add(date)
            print(
                f"U1 hierarchical risk {date:%Y-%m-%d}: "
                f"{payload['diagnostics']['assets']} assets in "
                f"{time.perf_counter() - started:.2f}s",
                flush=True,
            )
        missing = dates.difference(pd.DatetimeIndex(produced))
        if len(missing):
            raise AssertionError(f"correlation iterator missed dates: {missing.tolist()}")

    weights = {
        method: pd.DataFrame(0.0, index=dates, columns=inputs.eligibility.columns)
        for method in METHODS
    }
    risk_frames = []
    contribution_frames = []
    diagnostic_rows = []
    for date in dates:
        payload = _load_allocation_cache(date)
        for method, series in payload["methods"].items():
            weights[method].loc[date, series.index] = series.to_numpy()
        risk_frames.append(payload["risk_metrics"])
        contribution_frames.append(payload["contributions"])
        diagnostic_rows.append(payload["diagnostics"])
    return (
        weights,
        pd.concat(risk_frames, ignore_index=True),
        pd.concat(contribution_frames, ignore_index=True),
        pd.DataFrame(diagnostic_rows),
    )


def _performance(
    weights: Mapping[str, pd.DataFrame],
    inputs: d4.UniverseInputs,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Backtest the six fully invested risk portfolios with frozen mechanics."""
    data = e5.load_universe(e5.UniverseName.MSCI_US)
    prices = e5._prices(data).reindex(columns=inputs.eligibility.columns)
    ew_nav = u1_bics._ew_navs()[WINDOW]
    rows = []
    portfolios = {}
    for method in METHODS:
        net, gross = u1_bics._backtest(
            prices,
            weights[method],
            u1_bics.COST_BPS / 10000.0,
            f"u1_hierarchical_risk_{method}",
        )
        portfolios[method] = net
        rows.append(
            {
                "universe": "msci_us",
                "analysis_window": WINDOW,
                "method": method,
                "cost_bps_one_way": u1_bics.COST_BPS,
                **u1_ls._performance_payload(net, gross, ew_nav),
            }
        )
    return pd.DataFrame(rows), portfolios


def _method_comparison(
    performance: pd.DataFrame,
    risk_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Compare every cluster-aware method with flat ERC, never with EW-all."""
    base_performance = performance.set_index("method").loc["flat_erc"]
    base_risk = risk_summary.set_index("method").loc["flat_erc"]
    rows = []
    performance_metrics = (
        "net_return_annualized",
        "volatility_annualized",
        "sharpe_rf0",
        "one_way_turnover_annualized",
        "cost_drag_bp_per_year",
        "maximum_drawdown",
    )
    risk_metrics = (
        "portfolio_ex_ante_volatility_mean",
        "effective_risk_clusters_absolute_mean",
        "maximum_absolute_cluster_risk_share_mean",
        "cluster_risk_hhi_absolute_mean",
        "effective_assets_mean",
        "diversification_ratio_mean",
    )
    for method in METHODS:
        if method == "flat_erc":
            continue
        candidate_performance = performance.set_index("method").loc[method]
        candidate_risk = risk_summary.set_index("method").loc[method]
        row = {"method": method, "benchmark_method": "flat_erc"}
        for metric in performance_metrics:
            if metric in performance.columns:
                row[metric] = candidate_performance[metric]
                row[f"delta_{metric}"] = (
                    candidate_performance[metric] - base_performance[metric]
                )
        for metric in risk_metrics:
            row[metric] = candidate_risk[metric]
            row[f"delta_{metric}"] = candidate_risk[metric] - base_risk[metric]
        rows.append(row)
    return pd.DataFrame(rows)


def _risk_method_summary(risk_per_date: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-date ex-ante risk diagnostics by allocation method."""
    metrics = (
        "portfolio_ex_ante_volatility",
        "cluster_risk_hhi_absolute",
        "effective_risk_clusters_absolute",
        "maximum_absolute_cluster_risk_share",
        "negative_cluster_risk_share",
        "effective_assets",
        "cluster_capital_hhi",
        "diversification_ratio",
        "maximum_weight",
        "maximum_cluster_risk_target_error",
    )
    rows = []
    for method, panel in risk_per_date.groupby("method", sort=False):
        row = {"method": method, "dates": len(panel)}
        for metric in metrics:
            row[f"{metric}_mean"] = float(panel[metric].mean())
            row[f"{metric}_median"] = float(panel[metric].median())
            row[f"{metric}_max"] = float(panel[metric].max())
        rows.append(row)
    return pd.DataFrame(rows)


def _signal_risk_diagnostic(
    inputs: d4.UniverseInputs,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Decompose accepted global and cluster long-short risk by Ward cluster."""
    dates = inputs.dates
    eligibility = inputs.eligibility.astype(bool)
    columns = eligibility.columns
    data = e5.load_universe(e5.UniverseName.MSCI_US)
    bics = inputs.taxonomy["bbg_bics_sector"].replace("", np.nan)
    primary_eligibility = eligibility & bics.notna().to_numpy()
    global_scores, raw_decision, raw_source, preflight = d5._u1_signal_inputs(
        data, dates, columns
    )
    groups = inputs.frozen_panel.reindex(index=dates, columns=columns)
    cluster_scores, cluster_timestamps = d5._cluster_scores(
        raw_source,
        groups,
        dates,
        primary_eligibility,
        raw_decision.reindex(index=dates, columns=columns),
    )
    global_groups = pd.DataFrame("global", index=dates, columns=columns)
    signal_weights = {
        "cluster_long_short": u1_bics._long_short_weights(
            cluster_scores, primary_eligibility, groups
        )[0],
        "global_long_short": u1_bics._long_short_weights(
            global_scores.reindex(index=dates, columns=columns).where(primary_eligibility),
            primary_eligibility,
            global_groups,
        )[0],
    }
    rows = []
    contribution_rows = []
    for date in dates:
        covar, clusters, _, _ = _load_date_inputs(date, inputs)
        for leg, panel in signal_weights.items():
            weights = panel.loc[date].reindex(covar.index).fillna(0.0)
            contributions = risk_contribution_summary(weights, covar, clusters)
            contributions.insert(0, "leg", leg)
            contributions.insert(0, "date", date)
            contribution_rows.append(contributions)
            metrics = risk_concentration_metrics(contributions)
            grouped_exposure = weights.groupby(clusters).sum()
            long_side = weights.clip(lower=0.0)
            short_side = -weights.clip(upper=0.0)
            rows.append(
                {
                    "date": date,
                    "leg": leg,
                    "cluster_net_exposure_l1": float(grouped_exposure.abs().sum()),
                    "maximum_absolute_cluster_net_exposure": float(
                        grouped_exposure.abs().max()
                    ),
                    "effective_long_assets": float(
                        long_side.sum() ** 2 / long_side.pow(2.0).sum()
                    ),
                    "effective_short_assets": float(
                        short_side.sum() ** 2 / short_side.pow(2.0).sum()
                    ),
                    **metrics,
                }
            )
    preflight = preflight.copy()
    preflight["max_cluster_score_lookahead_days"] = float(
        cluster_timestamps.sub(cluster_timestamps.index).dt.days.max()
    )
    return (
        pd.DataFrame(rows),
        pd.concat(contribution_rows, ignore_index=True),
        preflight,
    )


def _signal_risk_summary(signal_risk: pd.DataFrame) -> pd.DataFrame:
    """Aggregate long-short risk decomposition for the two ranking strategies."""
    metrics = [
        column
        for column in signal_risk.columns
        if column not in {"date", "leg"}
    ]
    rows = []
    for leg, panel in signal_risk.groupby("leg", sort=False):
        row = {"leg": leg, "dates": len(panel)}
        for metric in metrics:
            row[f"{metric}_mean"] = float(panel[metric].mean())
            row[f"{metric}_median"] = float(panel[metric].median())
            row[f"{metric}_max"] = float(panel[metric].max())
        rows.append(row)
    return pd.DataFrame(rows)


def _acceptance(
    diagnostics: pd.DataFrame,
    risk_per_date: pd.DataFrame,
    weights: Mapping[str, pd.DataFrame],
    inputs: d4.UniverseInputs,
    signal_preflight: pd.DataFrame,
    paper_comparison: pd.DataFrame,
) -> pd.DataFrame:
    """Return every measured-versus-tolerance numerical acceptance line."""
    outside = max(
        float(frame.where(~inputs.eligibility, 0.0).abs().to_numpy().max())
        for frame in weights.values()
    )
    checks = [
        (
            "headline allocation dates",
            float(len(diagnostics)),
            float(len(inputs.dates)),
            len(diagnostics) == len(inputs.dates),
        ),
        (
            "covariance and Ward asset-set match share",
            float(diagnostics["asset_set_match"].mean()),
            PARTITION_TOLERANCE,
            diagnostics["asset_set_match"].all(),
        ),
        (
            "Ward cache and frozen ME36 partition match share",
            float(diagnostics["partition_match"].mean()),
            PARTITION_TOLERANCE,
            diagnostics["partition_match"].all(),
        ),
        (
            "alpha one versus flat ERC maximum weight error",
            float(diagnostics["flat_erc_vs_alpha_one_max_abs_weight_error"].max()),
            WEIGHT_TOLERANCE,
            float(diagnostics["flat_erc_vs_alpha_one_max_abs_weight_error"].max())
            <= WEIGHT_TOLERANCE,
        ),
        (
            "maximum allocation weight-sum error",
            float(diagnostics["maximum_weight_sum_error"].max()),
            WEIGHT_TOLERANCE,
            float(diagnostics["maximum_weight_sum_error"].max()) <= WEIGHT_TOLERANCE,
        ),
        (
            "minimum hierarchical allocation weight",
            float(diagnostics["minimum_method_weight"].min()),
            0.0,
            float(diagnostics["minimum_method_weight"].min()) >= 0.0,
        ),
        (
            "weight outside point-in-time eligibility",
            outside,
            WEIGHT_TOLERANCE,
            outside <= WEIGHT_TOLERANCE,
        ),
        (
            "maximum Euler risk reconciliation error",
            float(diagnostics["maximum_risk_reconciliation_error"].max()),
            WEIGHT_TOLERANCE,
            float(diagnostics["maximum_risk_reconciliation_error"].max())
            <= WEIGHT_TOLERANCE,
        ),
        (
            "maximum risk-budget target error",
            float(diagnostics["maximum_rb_target_error"].max()),
            RISK_TOLERANCE,
            float(diagnostics["maximum_rb_target_error"].max()) <= RISK_TOLERANCE,
        ),
        (
            "risk rows",
            float(len(risk_per_date)),
            float(len(inputs.dates) * len(METHODS)),
            len(risk_per_date) == len(inputs.dates) * len(METHODS),
        ),
        (
            "maximum signal lookahead days",
            float(
                max(
                    signal_preflight["max_signal_lookahead_days"].max(),
                    signal_preflight["max_cluster_score_lookahead_days"].max(),
                )
            ),
            0.0,
            max(
                signal_preflight["max_signal_lookahead_days"].max(),
                signal_preflight["max_cluster_score_lookahead_days"].max(),
            )
            <= 0.0,
        ),
        (
            "paper comparison rows",
            float(len(paper_comparison)),
            5.0,
            len(paper_comparison) == 5,
        ),
        (
            "Ward-HERC paper rows",
            float(paper_comparison["method_id"].eq("ward_herc").sum()),
            0.0,
            not paper_comparison["method_id"].eq("ward_herc").any(),
        ),
    ]
    frame = pd.DataFrame(
        [
            {
                "check": check,
                "measured": measured,
                "tolerance": tolerance,
                "status": "PASS" if passed else "FAIL",
            }
            for check, measured, tolerance, passed in checks
        ]
    )
    if not frame["status"].eq("PASS").all():
        raise AssertionError(frame.loc[~frame["status"].eq("PASS")])
    return frame


def _design() -> pd.DataFrame:
    """Return the frozen experimental design and interpretation boundaries."""
    return pd.DataFrame(
        [
            {
                "universe": "U1 MSCI US point-in-time members",
                "analysis_window": "2009-08-31 through 2026-06-30",
                "risk_covariance": "baseline HCGL/FF6 snapshot, frozen E2 cache",
                "hierarchy": "ME returns, EWMA span 36, Pearson, 1-rho, cutoff 0.60",
                "ward_role": "paper cluster hierarchy",
                "single_role": "canonical HRP literature comparator",
                "risk_budget_exponents": "1.0 flat|0.5 sqrt-size|0.0 equal-cluster",
                "constraints": "long-only, fully invested",
                "rebalance": "ME",
                "implementation_lag": 1,
                "cost_bps_one_way": 10.0,
                "performance_benchmark": "flat ERC only",
                "ew_role": "market reference for beta/alpha only",
                "nested_cluster_risk_budgeting": "excluded",
                "snapshot_method_name": "rolling EWMA-Ward correlation clustering",
                "risk_strategy_name": "Rolling-Ward HRP",
                "signal_strategy_name": "Rolling-Ward cluster-relative momentum",
            }
        ]
    )


def _paper_comparison(performance: pd.DataFrame) -> pd.DataFrame:
    """Return the frozen long-only and long-short paper comparison rows."""
    long_labels = {
        "flat_erc": ("Flat ERC", "control", "flat ERC"),
        "single_hrp": (
            "Canonical HRP (single linkage)",
            "control",
            "canonical single-link HRP",
        ),
        "ward_hrp": (
            "Rolling-Ward HRP",
            "proposed clustering",
            "flat ERC and canonical single-link HRP",
        ),
    }
    rows = []
    for method_id, (label, role, benchmark) in long_labels.items():
        source = performance.set_index("method").loc[method_id]
        rows.append(
            {
                "panel": "long_only_risk_allocation",
                "method_id": method_id,
                "paper_label": label,
                "role": role,
                "comparison_benchmark": benchmark,
                "net_return_annualized": source["net_return_annualized"],
                "volatility_annualized": source["volatility_annualized"],
                "sharpe_rf0": source["sharpe_rf0"],
                "one_way_turnover_annualized": source[
                    "one_way_turnover_annualized"
                ],
                "cost_bps_one_way": source["cost_bps_one_way"],
                "source": "risk_allocation/performance.csv",
            }
        )
    signal = pd.read_csv(DEPC1_SIGNAL_PERFORMANCE_PATH, float_precision="round_trip")
    signal_labels = {
        "global": ("Global momentum rank", "control", "global rank"),
        "cluster_raw": (
            "Rolling-Ward cluster-relative momentum",
            "proposed clustering",
            "global rank",
        ),
    }
    signal = signal.set_index("leg")
    for method_id, (label, role, benchmark) in signal_labels.items():
        source = signal.loc[method_id]
        rows.append(
            {
                "panel": "long_short_momentum",
                "method_id": method_id,
                "paper_label": label,
                "role": role,
                "comparison_benchmark": benchmark,
                "net_return_annualized": source["net_return_annualized"],
                "volatility_annualized": source["volatility_annualized"],
                "sharpe_rf0": source["sharpe_rf0"],
                "one_way_turnover_annualized": source[
                    "one_way_turnover_annualized"
                ],
                "cost_bps_one_way": source["cost_bps_one_way"],
                "source": str(DEPC1_SIGNAL_PERFORMANCE_PATH),
            }
        )
    return pd.DataFrame(rows)


def _source_manifest(inputs: d4.UniverseInputs) -> pd.DataFrame:
    """Record runner, allocation module, and frozen input roots."""
    runner = Path(__file__)
    allocation_module = runner.with_name("hierarchical_risk_allocations.py")
    sample_date = inputs.dates[-1]
    sample_ward = d4._cache_dir(inputs, "raw") / f"{sample_date:%Y%m%d}.pkl"
    sample_covar = _risk_covar_path(sample_date)
    return pd.DataFrame(
        [
            {"kind": "runner", "path": str(runner), "sha256": _sha256(runner)},
            {
                "kind": "allocation_module",
                "path": str(allocation_module),
                "sha256": _sha256(allocation_module),
            },
            {
                "kind": "sample_ward_cache",
                "path": str(sample_ward),
                "sha256": _sha256(sample_ward),
            },
            {
                "kind": "sample_risk_covar_cache",
                "path": str(sample_covar),
                "sha256": _sha256(sample_covar),
            },
        ]
    )


def _save_exhibit(fig, file_name: str) -> Path:
    """Save one paper exhibit through the repository's qis reporting layer."""
    import matplotlib.pyplot as plt
    import qis

    path = Path(
        qis.save_fig(
            fig=fig,
            file_name=Path(file_name).stem,
            local_path=str(_root()),
            dpi=180,
            add_current_date=False,
        )
    )
    plt.close(fig)
    return path


def _four_panel_bars(
    table: pd.DataFrame,
    *,
    specifications: tuple[tuple[str, str, str], ...],
    title: str,
):
    """Return a four-panel qis bar exhibit for named table columns."""
    import matplotlib.pyplot as plt
    import qis

    fig, axes = plt.subplots(2, 2, figsize=(15, 8.5))
    for axis, (column, panel_title, value_format) in zip(
        axes.flat, specifications, strict=True
    ):
        qis.plot_bars(
            table[column],
            stacked=False,
            title=panel_title,
            add_bar_values=True,
            xvar_format=value_format,
            yvar_format=value_format,
            x_rotation=25,
            legend_loc=None,
            series_color="steelblue",
            ax=axis,
        )
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    return fig


def _build_exhibits(
    performance: pd.DataFrame,
    risk_summary: pd.DataFrame,
    signal_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Build the paper-ready allocation and signal-risk comparison exhibits."""
    labels = {
        "flat_erc": "Flat ERC",
        "cluster_rb_alpha_0_5": "Cluster RB sqrt(n)",
        "cluster_rb_alpha_0": "Cluster RB equal",
        "ward_hrp": "Ward-HRP",
        "ward_herc": "Ward-HERC",
        "single_hrp": "Single-HRP",
        "cluster_long_short": "Cluster rank",
        "global_long_short": "Global rank",
    }
    perf = performance.set_index("method").rename(index=labels)
    risk = risk_summary.set_index("method").rename(index=labels)
    perf = perf.drop(index="Ward-HERC")
    risk = risk.drop(index="Ward-HERC")
    signal = signal_summary.set_index("leg").rename(index=labels)
    figures = {
        "u1_allocation_performance.png": _four_panel_bars(
            perf,
            specifications=(
                ("net_return_annualized", "Net annualized return", "{:.1%}"),
                ("volatility_annualized", "Realized annualized volatility", "{:.1%}"),
                ("sharpe_rf0", "Net Sharpe (rf=0)", "{:.2f}"),
                (
                    "one_way_turnover_annualized",
                    "Annualized one-way turnover",
                    "{:.1f}x",
                ),
            ),
            title="U1: standard risk-allocation methods",
        ),
        "u1_allocation_risk_structure.png": _four_panel_bars(
            risk,
            specifications=(
                (
                    "portfolio_ex_ante_volatility_mean",
                    "Mean ex-ante volatility",
                    "{:.1%}",
                ),
                (
                    "effective_risk_clusters_absolute_mean",
                    "Effective Ward risk clusters",
                    "{:.1f}",
                ),
                (
                    "maximum_absolute_cluster_risk_share_mean",
                    "Largest absolute cluster-risk share",
                    "{:.1%}",
                ),
                ("diversification_ratio_mean", "Diversification ratio", "{:.2f}"),
            ),
            title="U1: ex-ante risk through the selected Ward structure",
        ),
        "u1_signal_risk_structure.png": _four_panel_bars(
            signal,
            specifications=(
                (
                    "portfolio_ex_ante_volatility_mean",
                    "Mean ex-ante volatility",
                    "{:.1%}",
                ),
                (
                    "effective_risk_clusters_absolute_mean",
                    "Effective Ward risk clusters",
                    "{:.1f}",
                ),
                (
                    "maximum_absolute_cluster_risk_share_mean",
                    "Largest absolute cluster-risk share",
                    "{:.1%}",
                ),
                (
                    "cluster_net_exposure_l1_mean",
                    "Ward-cluster net-exposure L1",
                    "{:.2f}",
                ),
            ),
            title="U1 long-short momentum: cluster rank versus global rank",
        ),
    }
    rows = []
    for file_name, figure in figures.items():
        path = _save_exhibit(figure, file_name)
        rows.append(
            {
                "exhibit": file_name,
                "path": str(path),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
                "status": "PASS" if path.stat().st_size > 0 else "FAIL",
            }
        )
    return pd.DataFrame(rows)


def run() -> Mapping[str, pd.DataFrame]:
    """Execute the complete U1 hierarchy/risk experiment from cache-first inputs."""
    started = time.perf_counter()
    inputs = d4._u1_inputs()
    weights, risk_per_date, contributions, diagnostics = _build_allocations(inputs)
    performance, portfolios = _performance(weights, inputs)
    risk_summary = _risk_method_summary(risk_per_date)
    comparison = _method_comparison(performance, risk_summary)
    signal_risk, signal_contributions, signal_preflight = _signal_risk_diagnostic(inputs)
    signal_summary = _signal_risk_summary(signal_risk)
    paper_comparison = _paper_comparison(performance)
    acceptance = _acceptance(
        diagnostics,
        risk_per_date,
        weights,
        inputs,
        signal_preflight,
        paper_comparison,
    )
    output = {
        "navs": pd.concat(
            {
                method: portfolio.get_portfolio_nav()
                for method, portfolio in portfolios.items()
            },
            axis=1,
        ).rename_axis("date").reset_index(),
        "design": _design(),
        "performance": performance,
        "comparison_vs_flat_erc": comparison,
        "risk_per_date": risk_per_date,
        "risk_summary": risk_summary,
        "cluster_risk_contributions": contributions,
        "signal_risk_per_date": signal_risk,
        "signal_risk_summary": signal_summary,
        "paper_comparison": paper_comparison,
        "signal_cluster_risk_contributions": signal_contributions,
        "signal_preflight": signal_preflight,
        "allocation_diagnostics": diagnostics,
        "acceptance": acceptance,
        "source_manifest": _source_manifest(inputs),
        "exhibit_manifest": _build_exhibits(performance, risk_summary, signal_summary),
        "runtime": pd.DataFrame(
            [{"runtime_seconds": time.perf_counter() - started, "runner": RUNNER}]
        ),
    }
    for method, frame in weights.items():
        output[f"weights_{method}"] = frame.reset_index(names="date")
    for name, frame in output.items():
        e5._write(frame, _root() / f"{name}.csv")
    return output


def _hash_outputs() -> dict[str, str]:
    """Hash deterministic CSV artifacts while excluding timing and replay rows."""
    return {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(_root().glob("*.csv"))
        if path.name not in {"runtime.csv", "determinism.csv"}
    }


def verify_determinism() -> pd.DataFrame:
    """Replay the cache-first run and require byte-identical numerical artifacts."""
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
    """Run U1 hierarchical risk allocation and deterministic replay."""
    replay = verify_determinism()
    print(
        f"U1 hierarchical risk allocation: PASS "
        f"({len(replay)}/{len(replay)} deterministic)",
        flush=True,
    )


if __name__ == "__main__":
    main()
