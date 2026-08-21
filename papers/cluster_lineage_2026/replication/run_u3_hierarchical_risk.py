"""Evaluate frozen U3 futures clusters in standard risk-allocation methods.

The long-only panel compares flat ERC, canonical single-link HRP, and the
owner-selected rolling Ward hierarchy.  Two transparent cluster-risk-budget
allocations are retained as mechanism diagnostics.  Ward-HERC and nested cluster
allocation are excluded from computation and from the paper table.

The long-short panel is not re-selected: it is the frozen M1-star cluster-relative
ROSAA momentum strategy and its matched global-rank comparator.  Both use q=25%,
30/30/30/10 Equity/Fixed Income/Commodities/FX budgets on each side, one-period
implementation lag, 10 bp one-way costs, and the seven owner-frozen liquidity
exclusions.  EW-all is used by the accepted performance machinery only as the alpha
reference and is never a ranking-performance yardstick.
"""

from __future__ import annotations

import hashlib
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
from scipy.cluster.hierarchy import fcluster

import papers.cluster_lineage_2026.replication.run_backtests as e5
import papers.cluster_lineage_2026.replication.run_depc1_cluster_comparison as d4
import papers.cluster_lineage_2026.replication.run_futures_best_relative_pnl_scatter as best
import papers.cluster_lineage_2026.replication.run_futures_sleeve_grid as futures
import papers.cluster_lineage_2026.replication.run_futures_weight_30303010_u1_window as windowed
from papers.cluster_lineage_2026.replication.hierarchical_risk_allocations import (
    cluster_risk_budget,
    hrp_weights,
    risk_concentration_metrics,
    risk_contribution_summary,
)


RUNNER = "papers/cluster_lineage_2026/replication/run_u3_hierarchical_risk.py"
ALLOCATION_CACHE_VERSION = 1
WINDOW_START = pd.Timestamp("2009-08-31")
WINDOW_END = pd.Timestamp("2026-06-30")
WINDOW = windowed.WINDOW
COST_BPS = best.COST_BPS
Q = best.Q
SIGNAL_SPEC = best.SPEC
SIGNAL_CLUSTER_METHOD = best.CLUSTER_METHOD
OWNER_EXCLUSIONS = e5.FUTURES_INVESTABILITY_EXCLUSIONS
ANNUALIZATION = 52.0
METHODS = (
    "flat_erc",
    "cluster_rb_alpha_0_5",
    "cluster_rb_alpha_0",
    "ward_hrp",
    "single_hrp",
)
PAPER_LONG_ONLY_METHODS = ("flat_erc", "single_hrp", "ward_hrp")
RISK_BUDGET_EXPONENTS = {
    "flat_erc": 1.0,
    "cluster_rb_alpha_0_5": 0.5,
    "cluster_rb_alpha_0": 0.0,
}
WEIGHT_TOLERANCE = 5e-10
RISK_TOLERANCE = 2e-5
PARTITION_TOLERANCE = 1.0


def _root() -> Path:
    """Return the isolated external U3 hierarchical-risk output root."""
    return e5.get_output_path("risk_allocation", "u3_hierarchical_20260816", create=True)


def _allocation_cache_root() -> Path:
    """Return the directory containing one allocation payload per decision date."""
    root = _root() / "allocation_cache"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _allocation_cache_path(date: pd.Timestamp) -> Path:
    """Return one decision date's allocation-cache path."""
    return _allocation_cache_root() / f"{pd.Timestamp(date):%Y%m%d}.pkl"


def _risk_covar_path(date: pd.Timestamp) -> Path:
    """Return the accepted M1-star HCGL covariance snapshot path."""
    return e5.get_output_path(
        e5.UniverseName.FUTURES.value,
        e5.SmootherName.M1_STAR.value,
        f"{pd.Timestamp(date):%Y%m%d}.pkl",
    )


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


def _headline_inputs() -> d4.UniverseInputs:
    """Return U3 inputs restricted to the frozen U1 headline calendar."""
    source = d4._u3_inputs()
    dates = source.dates[(source.dates >= WINDOW_START) & (source.dates <= WINDOW_END)]
    return d4.UniverseInputs(
        universe=source.universe,
        returns=source.returns,
        dates=dates,
        eligibility=source.eligibility.reindex(index=dates),
        model=source.model,
        taxonomy=source.taxonomy,
        frozen_panel=source.frozen_panel.reindex(index=dates),
        config_id=source.config_id,
        input_paths=source.input_paths,
    )


def _covariance_to_correlation(covar: pd.DataFrame) -> pd.DataFrame:
    """Normalize a finite positive-diagonal covariance matrix to correlation."""
    values = covar.to_numpy(dtype=float)
    diagonal = np.diag(values)
    if np.any(diagonal <= 0.0) or not np.isfinite(values).all():
        raise ValueError("eligible covariance must be finite with positive diagonal")
    inverse = 1.0 / np.sqrt(diagonal)
    correlation = values * np.outer(inverse, inverse)
    return pd.DataFrame(correlation, index=covar.index, columns=covar.columns)


def _load_date_inputs(
    date: pd.Timestamp,
    full_corr: pd.DataFrame,
    inputs: d4.UniverseInputs,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, np.ndarray, dict[str, object]]:
    """Load one frozen covariance, membership vector, and eligible Ward tree."""
    ward_path = d4._cache_dir(inputs, "raw") / f"{pd.Timestamp(date):%Y%m%d}.pkl"
    with ward_path.open("rb") as stream:
        ward = pickle.load(stream)
    risk_path = _risk_covar_path(date)
    with risk_path.open("rb") as stream:
        snapshot = pickle.load(stream)

    clusters = inputs.frozen_panel.loc[date].dropna()
    eligible = inputs.eligibility.columns[inputs.eligibility.loc[date].astype(bool)]
    ward_clusters = ward["clusters"].dropna()
    exact_asset_set = set(clusters.index) == set(eligible) == set(ward_clusters.index)
    if not exact_asset_set or not clusters.index.equals(ward_clusters.index):
        raise AssertionError(f"{date:%Y-%m-%d} U3 hierarchy asset/order mismatch")

    raw_covar = snapshot.get_y_covar()
    covariance_superset = set(clusters.index).issubset(raw_covar.index)
    if not covariance_superset:
        raise AssertionError(f"{date:%Y-%m-%d} U3 covariance misses eligible assets")
    covar = raw_covar.reindex(index=clusters.index, columns=clusters.index) * ANNUALIZATION
    corr = full_corr.reindex(index=clusters.index, columns=clusters.index)
    linkage = np.asarray(ward["linkage"], dtype=float)
    if linkage.shape != (len(clusters) - 1, 4):
        raise AssertionError(f"{date:%Y-%m-%d} invalid U3 Ward linkage shape")

    refit = pd.Series(
        fcluster(linkage, t=float(ward["cutoff"]), criterion="distance"),
        index=clusters.index,
    )
    diagnostics = {
        "date": date,
        "assets": len(clusters),
        "clusters": int(clusters.nunique()),
        "exact_eligible_asset_set": exact_asset_set,
        "covariance_contains_eligible_assets": covariance_superset,
        "ward_leaf_order_matches_membership_order": clusters.index.equals(ward_clusters.index),
        "eligible_refit_cut_matches_frozen_membership": _same_partition(refit, clusters),
        "minimum_covariance_eigenvalue": float(np.linalg.eigvalsh(covar).min()),
        "covariance_condition_number": float(np.linalg.cond(covar)),
        "ward_path": str(ward_path),
        "risk_covar_path": str(risk_path),
    }
    return covar, corr, clusters, linkage, diagnostics


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
    """Return method-level risk metrics and full cluster contributions."""
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
    full_corr: pd.DataFrame,
    inputs: d4.UniverseInputs,
) -> dict[str, object]:
    """Compute all U3 allocation methods for one date and persist diagnostics."""
    covar, corr, clusters, ward_linkage, input_diagnostics = _load_date_inputs(
        date, full_corr, inputs
    )
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
        "single_hrp": hrp_weights(covar, single_linkage),
    }
    budgets = {
        "flat_erc": flat_budget,
        "cluster_rb_alpha_0_5": square_root_budget,
        "cluster_rb_alpha_0": cluster_equal_budget,
    }
    risk_metrics, contributions = _date_risk_rows(date, methods, covar, clusters, budgets)
    diagnostics = {
        **input_diagnostics,
        "flat_erc_vs_alpha_one_max_abs_weight_error": float(
            flat_erc.subtract(alpha_one).abs().max()
        ),
        "maximum_weight_sum_error": max(
            abs(float(weights.sum()) - 1.0) for weights in methods.values()
        ),
        "minimum_method_weight": min(float(weights.min()) for weights in methods.values()),
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
    """Load one validated allocation cache."""
    with _allocation_cache_path(date).open("rb") as stream:
        return pickle.load(stream)


def _build_allocations(
    inputs: d4.UniverseInputs,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Compute or load rolling U3 allocations and assemble labelled panels."""
    dates = inputs.dates
    if not all(_valid_allocation_cache(date) for date in dates):
        produced = set()
        for date, full_corr in _iter_correlation_inputs(inputs.returns, list(dates), inputs.model):
            date = pd.Timestamp(date)
            if date not in dates:
                continue
            started = time.perf_counter()
            payload = _compute_date_payload(date, full_corr, inputs)
            produced.add(date)
            print(
                f"U3 hierarchical risk {date:%Y-%m-%d}: "
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
    diagnostics = []
    for date in dates:
        payload = _load_allocation_cache(date)
        for method, series in payload["methods"].items():
            weights[method].loc[date, series.index] = series.to_numpy()
        risk_frames.append(payload["risk_metrics"])
        contribution_frames.append(payload["contributions"])
        diagnostics.append(payload["diagnostics"])
    return (
        weights,
        pd.concat(risk_frames, ignore_index=True),
        pd.concat(contribution_frames, ignore_index=True),
        pd.DataFrame(diagnostics),
    )


def _long_only_performance(
    weights: Mapping[str, pd.DataFrame],
    context: Mapping[str, object],
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Backtest every fully invested U3 risk allocation with frozen mechanics."""
    rows = []
    portfolios = {}
    for method in METHODS:
        net, gross = futures._backtest(
            context["performance_prices"],
            weights[method],
            COST_BPS / 10000.0,
            f"u3_hierarchical_risk_{method}",
        )
        portfolios[method] = net
        payload = futures._performance_payload(
            windowed._WindowedPortfolio(net),
            windowed._WindowedPortfolio(gross),
            context["ew_nav"],
        )
        rows.append(
            {
                "universe": "futures",
                "analysis_window": WINDOW,
                "method": method,
                "cost_bps_one_way": COST_BPS,
                **payload,
            }
        )
    return pd.DataFrame(rows), portfolios


def rebuild_navs_from_frozen_weights() -> pd.DataFrame:
    """Rebuild missing NAVs from the accepted seven-exclusion weight panels.

    The later U3 signal study added four liquidity exclusions after this risk
    experiment was accepted.  The accepted risk weights therefore define the
    experiment vintage; the current context is used only for its unchanged
    performance-price panel and EW alpha reference.  Core performance metrics
    must reproduce the frozen table before the recovered NAVs are written.
    """
    root = _root()
    weights = {
        method: pd.read_csv(
            root / f"weights_{method}.csv",
            float_precision="round_trip",
            parse_dates=["date"],
        ).set_index("date")
        for method in METHODS
    }
    context = best.grid._build_context()
    performance, portfolios = _long_only_performance(weights, context)
    frozen = pd.read_csv(
        root / "performance.csv", float_precision="round_trip"
    ).set_index("method")
    measured = performance.set_index("method")
    metrics = (
        "net_total_return",
        "net_return_annualized",
        "volatility_annualized",
        "sharpe_rf0",
        "one_way_turnover_annualized",
        "gross_return_annualized",
    )
    errors = measured.loc[frozen.index, metrics].subtract(frozen[list(metrics)]).abs()
    maximum_error = float(errors.to_numpy().max())
    if maximum_error > 1e-12:
        raise AssertionError(
            "recovered U3 risk NAVs do not reproduce frozen performance: "
            f"{maximum_error:.3e}"
        )
    navs = pd.concat(
        {
            method: portfolio.get_portfolio_nav()
            for method, portfolio in portfolios.items()
        },
        axis=1,
    ).rename_axis("date").reset_index()
    acceptance = pd.DataFrame(
        [
            {
                "check": "recovered NAV performance versus frozen table",
                "measured": maximum_error,
                "tolerance": 1e-12,
                "status": "PASS",
                "risk_vintage": "seven owner exclusions",
                "price_vintage": "unchanged U3 performance-price panel",
            }
        ]
    )
    e5._write(navs, root / "navs.csv")
    e5._write(acceptance, root / "nav_rebuild_acceptance.csv")
    return acceptance


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


def _allocation_asset_class_summary(
    weights: Mapping[str, pd.DataFrame],
    inputs: d4.UniverseInputs,
) -> pd.DataFrame:
    """Summarize mean long-only capital allocations by source asset class."""
    taxonomy = inputs.taxonomy["asset_class"].reindex(inputs.eligibility.columns)
    rows = []
    for method, panel in weights.items():
        grouped = panel.mean(axis=0).groupby(taxonomy).sum()
        for asset_class, mean_weight in grouped.items():
            rows.append(
                {
                    "method": method,
                    "asset_class": asset_class,
                    "mean_capital_weight": float(mean_weight),
                }
            )
    return pd.DataFrame(rows)


def _method_comparison(
    performance: pd.DataFrame,
    risk_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Compare cluster-aware long-only methods with flat ERC only."""
    base_performance = performance.set_index("method").loc["flat_erc"]
    base_risk = risk_summary.set_index("method").loc["flat_erc"]
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
    rows = []
    for method in METHODS:
        if method == "flat_erc":
            continue
        candidate_performance = performance.set_index("method").loc[method]
        candidate_risk = risk_summary.set_index("method").loc[method]
        row = {"method": method, "benchmark_method": "flat_erc"}
        for metric in performance_metrics:
            if metric in performance.columns:
                row[metric] = candidate_performance[metric]
                row[f"delta_{metric}"] = candidate_performance[metric] - base_performance[metric]
        for metric in risk_metrics:
            row[metric] = candidate_risk[metric]
            row[f"delta_{metric}"] = candidate_risk[metric] - base_risk[metric]
        rows.append(row)
    return pd.DataFrame(rows)


def _signal_performance(portfolios: Mapping[str, object], ew_nav: pd.Series) -> pd.DataFrame:
    """Return the exact accepted U3 cluster/global long-short payoff rows."""
    frame = best._performance_table(portfolios, ew_nav).copy()
    return frame.assign(
        leg=frame["method"].map({SIGNAL_CLUSTER_METHOD: "cluster", best.GLOBAL_METHOD: "global"})
    )


def _signal_risk_diagnostic(
    signal_weights: Mapping[str, pd.DataFrame],
    inputs: d4.UniverseInputs,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Decompose accepted global and cluster long-short risk by M1-star group."""
    rows = []
    contribution_rows = []
    for date in inputs.dates:
        with _risk_covar_path(date).open("rb") as stream:
            snapshot = pickle.load(stream)
        clusters = inputs.frozen_panel.loc[date].dropna()
        covar = (
            snapshot.get_y_covar().reindex(index=clusters.index, columns=clusters.index)
            * ANNUALIZATION
        )
        for leg in ("cluster", "global"):
            weights = signal_weights[leg].loc[date].reindex(clusters.index).fillna(0.0)
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
                    "maximum_absolute_cluster_net_exposure": float(grouped_exposure.abs().max()),
                    "effective_long_assets": float(long_side.sum() ** 2 / long_side.pow(2.0).sum()),
                    "effective_short_assets": float(
                        short_side.sum() ** 2 / short_side.pow(2.0).sum()
                    ),
                    **metrics,
                }
            )
    return pd.DataFrame(rows), pd.concat(contribution_rows, ignore_index=True)


def _signal_risk_summary(signal_risk: pd.DataFrame) -> pd.DataFrame:
    """Aggregate long-short risk diagnostics for the two ranking strategies."""
    metrics = [column for column in signal_risk if column not in {"date", "leg"}]
    rows = []
    for leg, panel in signal_risk.groupby("leg", sort=False):
        row = {"leg": leg, "dates": len(panel)}
        for metric in metrics:
            row[f"{metric}_mean"] = float(panel[metric].mean())
            row[f"{metric}_median"] = float(panel[metric].median())
            row[f"{metric}_max"] = float(panel[metric].max())
        rows.append(row)
    return pd.DataFrame(rows)


def _paper_comparison(
    performance: pd.DataFrame,
    signal_performance: pd.DataFrame,
) -> pd.DataFrame:
    """Return the frozen five-row U3 paper comparison."""
    labels = {
        "flat_erc": ("Flat ERC", "control", "flat ERC"),
        "single_hrp": (
            "Canonical HRP (single linkage)",
            "control",
            "canonical single-link HRP",
        ),
        "ward_hrp": (
            "M1-star Rolling-Ward HRP",
            "proposed clustering",
            "flat ERC and canonical single-link HRP",
        ),
    }
    rows = []
    indexed = performance.set_index("method")
    for method, (label, role, benchmark) in labels.items():
        source = indexed.loc[method]
        rows.append(
            {
                "panel": "long_only_risk_allocation",
                "method_id": method,
                "paper_label": label,
                "role": role,
                "comparison_benchmark": benchmark,
                "net_return_annualized": source["net_return_annualized"],
                "volatility_annualized": source["volatility_annualized"],
                "sharpe_rf0": source["sharpe_rf0"],
                "one_way_turnover_annualized": source["one_way_turnover_annualized"],
                "cost_bps_one_way": source["cost_bps_one_way"],
                "source": "risk_allocation/performance.csv",
            }
        )
    signal_labels = {
        "global": ("Global momentum rank", "control", "global rank"),
        "cluster": (
            "M1-star Rolling-Ward cluster-relative momentum",
            "proposed clustering",
            "global rank",
        ),
    }
    signal = signal_performance.set_index("leg")
    for leg, (label, role, benchmark) in signal_labels.items():
        source = signal.loc[leg]
        rows.append(
            {
                "panel": "long_short_momentum",
                "method_id": leg,
                "paper_label": label,
                "role": role,
                "comparison_benchmark": benchmark,
                "net_return_annualized": source["net_return_annualized"],
                "volatility_annualized": source["volatility_annualized"],
                "sharpe_rf0": source["sharpe_rf0"],
                "one_way_turnover_annualized": source["one_way_turnover_annualized"],
                "cost_bps_one_way": source["cost_bps_one_way"],
                "source": "owner-frozen futures best-relative reconstruction",
            }
        )
    return pd.DataFrame(rows)


def _acceptance(
    diagnostics: pd.DataFrame,
    risk_per_date: pd.DataFrame,
    weights: Mapping[str, pd.DataFrame],
    inputs: d4.UniverseInputs,
    signal_weights: Mapping[str, pd.DataFrame],
    signal_diagnostics: Mapping[str, object],
    signal_performance: pd.DataFrame,
    paper_comparison: pd.DataFrame,
) -> pd.DataFrame:
    """Return every measured-versus-tolerance U3 acceptance line."""
    outside = max(
        float(frame.where(~inputs.eligibility, 0.0).abs().to_numpy().max())
        for frame in weights.values()
    )
    excluded = inputs.eligibility.columns.intersection(OWNER_EXCLUSIONS)
    excluded_weights = max(
        float(frame.reindex(columns=excluded).fillna(0.0).abs().to_numpy().max())
        for frame in signal_weights.values()
    )
    source_path = best._root() / "performance.csv"
    source = pd.read_csv(source_path, float_precision="round_trip").set_index("method")
    current = signal_performance.set_index("method")
    metrics = [
        "net_return_annualized",
        "volatility_annualized",
        "sharpe_rf0",
        "one_way_turnover_annualized",
    ]
    source_error = float(
        current.loc[source.index, metrics].subtract(source[metrics]).abs().to_numpy().max()
    )
    signal_error = max(
        abs(float(value))
        for name in ("cluster_weights", "global_weights")
        for key, value in signal_diagnostics[name].items()
        if key.startswith("max_") and key.endswith("error")
    )
    checks = [
        ("headline allocation dates", len(diagnostics), 203, len(diagnostics) == 203),
        (
            "exact eligible asset-set share",
            diagnostics["exact_eligible_asset_set"].mean(),
            PARTITION_TOLERANCE,
            diagnostics["exact_eligible_asset_set"].all(),
        ),
        (
            "covariance contains eligible assets share",
            diagnostics["covariance_contains_eligible_assets"].mean(),
            PARTITION_TOLERANCE,
            diagnostics["covariance_contains_eligible_assets"].all(),
        ),
        (
            "Ward leaf/membership order match share",
            diagnostics["ward_leaf_order_matches_membership_order"].mean(),
            PARTITION_TOLERANCE,
            diagnostics["ward_leaf_order_matches_membership_order"].all(),
        ),
        (
            "minimum covariance eigenvalue",
            diagnostics["minimum_covariance_eigenvalue"].min(),
            0.0,
            diagnostics["minimum_covariance_eigenvalue"].min() > 0.0,
        ),
        (
            "alpha one versus flat ERC maximum weight error",
            diagnostics["flat_erc_vs_alpha_one_max_abs_weight_error"].max(),
            WEIGHT_TOLERANCE,
            diagnostics["flat_erc_vs_alpha_one_max_abs_weight_error"].max() <= WEIGHT_TOLERANCE,
        ),
        (
            "maximum allocation weight-sum error",
            diagnostics["maximum_weight_sum_error"].max(),
            WEIGHT_TOLERANCE,
            diagnostics["maximum_weight_sum_error"].max() <= WEIGHT_TOLERANCE,
        ),
        (
            "minimum hierarchical allocation weight",
            diagnostics["minimum_method_weight"].min(),
            0.0,
            diagnostics["minimum_method_weight"].min() >= 0.0,
        ),
        (
            "weight outside point-in-time eligibility",
            outside,
            WEIGHT_TOLERANCE,
            outside <= WEIGHT_TOLERANCE,
        ),
        (
            "maximum Euler risk reconciliation error",
            diagnostics["maximum_risk_reconciliation_error"].max(),
            WEIGHT_TOLERANCE,
            diagnostics["maximum_risk_reconciliation_error"].max() <= WEIGHT_TOLERANCE,
        ),
        (
            "maximum risk-budget target error",
            diagnostics["maximum_rb_target_error"].max(),
            RISK_TOLERANCE,
            diagnostics["maximum_rb_target_error"].max() <= RISK_TOLERANCE,
        ),
        (
            "risk rows",
            len(risk_per_date),
            len(inputs.dates) * len(METHODS),
            len(risk_per_date) == len(inputs.dates) * len(METHODS),
        ),
        (
            "maximum signal lookahead days",
            signal_diagnostics["signal"]["max_signal_lookahead_days"],
            0.0,
            signal_diagnostics["signal"]["max_signal_lookahead_days"] <= 0.0,
        ),
        ("maximum signal construction error", signal_error, 1e-12, signal_error <= 1e-12),
        ("maximum owner-excluded signal weight", excluded_weights, 0.0, excluded_weights == 0.0),
        ("frozen signal performance error", source_error, 1e-12, source_error <= 1e-12),
        ("one-way transaction cost bps", COST_BPS, 10.0, COST_BPS == 10.0),
        ("paper comparison rows", len(paper_comparison), 5, len(paper_comparison) == 5),
        (
            "Ward-HERC paper rows",
            paper_comparison["method_id"].eq("ward_herc").sum(),
            0,
            not paper_comparison["method_id"].eq("ward_herc").any(),
        ),
        (
            "EW-all ranking-performance comparison rows",
            paper_comparison["comparison_benchmark"].str.contains("EW", case=False).sum(),
            0,
            not paper_comparison["comparison_benchmark"].str.contains("EW", case=False).any(),
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


def _design(diagnostics: pd.DataFrame) -> pd.DataFrame:
    """Return the frozen U3 design and the hierarchy-cut diagnostic."""
    return pd.DataFrame(
        [
            {
                "universe": "U3 owner-frozen futures",
                "analysis_window": "2009-08-31 through 2026-06-30",
                "risk_covariance": "M1-star HCGL snapshot, frozen E2b cache",
                "hierarchy": "W-WED returns, EWMA span 156, M1-star delta 0.0691, Ward",
                "eligible_refit_cut_matches_frozen_share": diagnostics[
                    "eligible_refit_cut_matches_frozen_membership"
                ].mean(),
                "long_only_constraints": "long-only, fully invested",
                "long_short_signal": SIGNAL_SPEC.signal_id,
                "long_short_q": Q,
                "sleeve_budgets_per_side": "Equity 30%|Fixed Income 30%|Commodities 30%|FX 10%",
                "rebalance": "ME decisions on the frozen W-WED information set",
                "implementation_lag": 1,
                "cost_bps_one_way": COST_BPS,
                "owner_exclusions": "|".join(sorted(OWNER_EXCLUSIONS)),
                "performance_benchmark": "flat ERC long-only|matched global rank long-short",
                "ew_role": "alpha/market reference only",
                "ward_herc": "excluded",
                "nested_cluster_risk_budgeting": "excluded",
                "risk_strategy_name": "M1-star Rolling-Ward HRP",
                "signal_strategy_name": "M1-star Rolling-Ward cluster-relative momentum",
            }
        ]
    )


def _source_manifest(inputs: d4.UniverseInputs) -> pd.DataFrame:
    """Record runner, shared allocation code, and frozen inputs."""
    runner = Path(__file__)
    allocation_module = runner.with_name("hierarchical_risk_allocations.py")
    date = inputs.dates[-1]
    paths = [
        ("runner", runner),
        ("allocation_module", allocation_module),
        ("sample_ward_cache", d4._cache_dir(inputs, "raw") / f"{date:%Y%m%d}.pkl"),
        ("sample_risk_covar_cache", _risk_covar_path(date)),
        ("accepted_signal_performance", best._root() / "performance.csv"),
    ]
    return pd.DataFrame(
        [{"kind": kind, "path": str(path), "sha256": _sha256(path)} for kind, path in paths]
    )


def _save_exhibit(fig, file_name: str) -> Path:
    """Save one exhibit through the repository's qis reporting layer."""
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
    specifications: tuple[tuple[str, str, str], ...],
    title: str,
):
    """Return a compact four-panel qis bar exhibit."""
    import matplotlib.pyplot as plt
    import qis

    fig, axes = plt.subplots(2, 2, figsize=(15, 8.5))
    for axis, (column, panel_title, value_format) in zip(axes.flat, specifications, strict=True):
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
    signal_performance: pd.DataFrame,
    signal_risk_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Build paper-ready U3 allocation and signal exhibits."""
    labels = {
        "flat_erc": "Flat ERC",
        "cluster_rb_alpha_0_5": "Cluster RB sqrt(n)",
        "cluster_rb_alpha_0": "Cluster RB equal",
        "ward_hrp": "M1-star Ward-HRP",
        "single_hrp": "Single-HRP",
    }
    perf = performance.set_index("method").rename(index=labels)
    risk = risk_summary.set_index("method").rename(index=labels)
    signal = signal_performance.set_index("leg").rename(
        index={"cluster": "M1-star cluster rank", "global": "Global rank"}
    )
    signal_risk = signal_risk_summary.set_index("leg").rename(
        index={"cluster": "M1-star cluster rank", "global": "Global rank"}
    )
    figures = {
        "u3_allocation_performance.png": _four_panel_bars(
            perf,
            (
                ("net_return_annualized", "Net annualized return", "{:.1%}"),
                ("volatility_annualized", "Realized annualized volatility", "{:.1%}"),
                ("sharpe_rf0", "Net Sharpe (rf=0)", "{:.2f}"),
                ("one_way_turnover_annualized", "Annualized one-way turnover", "{:.1f}x"),
            ),
            "U3 futures: standard risk-allocation methods",
        ),
        "u3_allocation_risk_structure.png": _four_panel_bars(
            risk,
            (
                ("portfolio_ex_ante_volatility_mean", "Mean ex-ante volatility", "{:.1%}"),
                (
                    "effective_risk_clusters_absolute_mean",
                    "Effective M1-star risk clusters",
                    "{:.1f}",
                ),
                (
                    "maximum_absolute_cluster_risk_share_mean",
                    "Largest absolute cluster-risk share",
                    "{:.1%}",
                ),
                ("diversification_ratio_mean", "Diversification ratio", "{:.2f}"),
            ),
            "U3 futures: ex-ante risk through M1-star groups",
        ),
        "u3_signal_comparison.png": _four_panel_bars(
            signal,
            (
                ("net_return_annualized", "Net annualized return", "{:.2%}"),
                ("volatility_annualized", "Realized annualized volatility", "{:.1%}"),
                ("sharpe_rf0", "Net Sharpe (rf=0)", "{:.2f}"),
                ("one_way_turnover_annualized", "Annualized one-way turnover", "{:.1f}x"),
            ),
            "U3 futures: cluster-relative versus matched global momentum",
        ),
        "u3_signal_risk_structure.png": _four_panel_bars(
            signal_risk,
            (
                ("portfolio_ex_ante_volatility_mean", "Mean ex-ante volatility", "{:.1%}"),
                ("cluster_net_exposure_l1_mean", "M1-star cluster net-exposure L1", "{:.2f}"),
                (
                    "maximum_absolute_cluster_net_exposure_mean",
                    "Largest cluster net exposure",
                    "{:.1%}",
                ),
                (
                    "effective_risk_clusters_absolute_mean",
                    "Effective absolute-risk clusters",
                    "{:.1f}",
                ),
            ),
            "U3 futures: signal risk through M1-star groups",
        ),
    }
    rows = []
    for file_name, fig in figures.items():
        path = _save_exhibit(fig, file_name)
        rows.append(
            {
                "artifact": file_name,
                "path": str(path),
                "sha256": _sha256(path),
                "bytes": path.stat().st_size,
                "status": "PASS" if path.stat().st_size > 0 else "FAIL",
            }
        )
    return pd.DataFrame(rows)


def run() -> Mapping[str, pd.DataFrame]:
    """Execute the complete cache-first U3 hierarchy/risk experiment."""
    started = time.perf_counter()
    inputs = _headline_inputs()
    weights, risk_per_date, contributions, diagnostics = _build_allocations(inputs)
    signal_portfolios, signal_weights, signal_details = best._build_weights_and_portfolios()
    context = signal_details["context"]
    if not pd.DatetimeIndex(context["dates"]).equals(inputs.dates):
        raise AssertionError("U3 risk and signal decision calendars differ")
    performance, allocation_portfolios = _long_only_performance(weights, context)
    risk_summary = _risk_method_summary(risk_per_date)
    allocation_asset_classes = _allocation_asset_class_summary(weights, inputs)
    comparison = _method_comparison(performance, risk_summary)
    signal_performance = _signal_performance(signal_portfolios, context["ew_nav"])
    signal_risk, signal_contributions = _signal_risk_diagnostic(signal_weights, inputs)
    signal_risk_summary = _signal_risk_summary(signal_risk)
    paper_comparison = _paper_comparison(performance, signal_performance)
    acceptance = _acceptance(
        diagnostics,
        risk_per_date,
        weights,
        inputs,
        signal_weights,
        signal_details,
        signal_performance,
        paper_comparison,
    )
    output = {
        "navs": pd.concat(
            {
                method: portfolio.get_portfolio_nav()
                for method, portfolio in allocation_portfolios.items()
            },
            axis=1,
        ).rename_axis("date").reset_index(),
        "design": _design(diagnostics),
        "performance": performance,
        "comparison_vs_flat_erc": comparison,
        "risk_per_date": risk_per_date,
        "risk_summary": risk_summary,
        "allocation_asset_class_summary": allocation_asset_classes,
        "cluster_risk_contributions": contributions,
        "signal_performance": signal_performance,
        "signal_risk_per_date": signal_risk,
        "signal_risk_summary": signal_risk_summary,
        "paper_comparison": paper_comparison,
        "signal_cluster_risk_contributions": signal_contributions,
        "allocation_diagnostics": diagnostics,
        "acceptance": acceptance,
        "source_manifest": _source_manifest(inputs),
        "exhibit_manifest": _build_exhibits(
            performance, risk_summary, signal_performance, signal_risk_summary
        ),
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
    """Replay cache-first and require byte-identical numerical artifacts."""
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
    """Run U3 hierarchical risk allocation and deterministic replay."""
    replay = verify_determinism()
    print(
        f"U3 hierarchical risk allocation: PASS ({len(replay)}/{len(replay)} deterministic)",
        flush=True,
    )


if __name__ == "__main__":
    main()
