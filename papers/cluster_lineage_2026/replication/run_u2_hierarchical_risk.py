"""Evaluate U2 Rolling-Ward clusters in standard risk-allocation methods.

The experiment freezes the owner-selected BlackRock-fund universe, strict rolling
12-month average AUM above USD 100m, W-THU/span-156 EWMA covariance, Ward hierarchy,
every-two-month schedule, one-period implementation lag, and 20 bp one-way costs.  It
compares flat ERC, two transparent cluster-risk-budget variants, Rolling-Ward HRP, and
canonical single-link HRP.  Ward-HERC and nested cluster allocation are excluded.

The same Ward structure decomposes ex-ante risk in the accepted U2 global-rank and
global-long/cluster-short momentum portfolios.  EW-all is retained only as the market
reference used by the frozen performance source; it is never a performance yardstick.
"""

from __future__ import annotations

import hashlib
import pickle
import time
from pathlib import Path
from typing import Iterator, Mapping

import numpy as np
import pandas as pd
from factorlasso import compute_clusters_from_corr_matrix, compute_ewm_covar
from factorlasso.cluster_smoothing import _iter_correlation_inputs
from factorlasso.lasso_estimator import get_x_y_np
from optimalportfolios import Constraints
from optimalportfolios.optimization.covar_factorization import factorize_covariance
from optimalportfolios import wrapper_risk_budgeting
from optimalportfolios.utils.portfolio_funcs import calculate_diversification_ratio

import papers.cluster_lineage_2026.replication.run_backtests as e5
import papers.cluster_lineage_2026.replication.run_depc1_cluster_comparison as d4
import papers.cluster_lineage_2026.replication.run_u2_blackrock_aum_filter as u2_aum
import papers.cluster_lineage_2026.replication.run_u2_blackrock_aum_sensitivity as u2_sensitivity
import papers.cluster_lineage_2026.replication.run_u2_blackrock_etf_grid as funds
import papers.cluster_lineage_2026.replication.run_u2_blackrock_long_short_search as u2_search
import papers.cluster_lineage_2026.replication.run_u2_blackrock_sleeve_grid as u2_sleeves
from papers.cluster_lineage_2026.replication.hierarchical_risk_allocations import (
    cluster_risk_budget,
    hrp_weights,
    risk_concentration_metrics,
    risk_contribution_summary,
)


RUNNER = "papers/cluster_lineage_2026/replication/run_u2_hierarchical_risk.py"
ALLOCATION_CACHE_VERSION = 2
WINDOW = u2_search.FULL_WINDOW
SCHEDULE = u2_sensitivity.SCHEDULE
COST_BPS = u2_sensitivity.COST_BPS
ANNUALIZATION = 52.0
MAX_CONDITION_NUMBER = 1e6
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
CORRELATION_TOLERANCE = 1e-12


def _root() -> Path:
    """Return the isolated external U2 hierarchical-risk output root."""
    return e5.get_output_path("risk_allocation", "u2_hierarchical_20260816", create=True)


def _allocation_cache_root() -> Path:
    """Return the directory containing one allocation payload per rebalance date."""
    root = _root() / "allocation_cache"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _allocation_cache_path(date: pd.Timestamp) -> Path:
    """Return one rebalance date's allocation-cache path."""
    return _allocation_cache_root() / f"{pd.Timestamp(date):%Y%m%d}.pkl"


def _sha256(path: Path) -> str:
    """Return a file's SHA-256 digest."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _frame_hash(frame: pd.DataFrame) -> str:
    """Hash one labelled frame independently of pickle serialization."""
    values = pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()
    columns = "\x1f".join(map(str, frame.columns)).encode("utf-8")
    return hashlib.sha256(values + columns).hexdigest()


def _input_fingerprint(inputs: d4.UniverseInputs) -> str:
    """Return a stable digest of the frozen U2 data, model, and experiment."""
    payload = "|".join(
        [
            *(f"{path}:{_sha256(path)}" for path in inputs.input_paths),
            f"returns:{_frame_hash(inputs.returns)}",
            f"eligibility:{_frame_hash(inputs.eligibility)}",
            f"partition:{_frame_hash(inputs.frozen_panel)}",
            f"model:{d4._model_payload(inputs.model)!r}",
            f"methods:{METHODS!r}",
            f"schedule:{SCHEDULE}",
            f"cost:{COST_BPS}",
            f"annualization:{ANNUALIZATION}",
            f"maximum_condition_number:{MAX_CONDITION_NUMBER}",
        ]
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _headline_dates(inputs: d4.UniverseInputs) -> pd.DatetimeIndex:
    """Return the frozen 203-date U2 headline estimation window."""
    return inputs.dates[
        (inputs.dates >= funds.HEADLINE_START) & (inputs.dates <= funds.HEADLINE_END)
    ]


def _rebalance_dates(inputs: d4.UniverseInputs) -> pd.DatetimeIndex:
    """Return the every-two-month subset used by the accepted U2 strategy."""
    return u2_search._rebalance_dates(_headline_dates(inputs), SCHEDULE)


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


def _covariance_to_correlation(covar: pd.DataFrame) -> pd.DataFrame:
    """Normalize a positive-diagonal covariance matrix to correlation."""
    values = covar.to_numpy(dtype=float)
    diagonal = np.diag(values)
    if np.any(diagonal <= 0.0) or not np.isfinite(values).all():
        raise ValueError("eligible covariance must be finite with positive diagonal")
    inverse = 1.0 / np.sqrt(diagonal)
    correlation = values * np.outer(inverse, inverse)
    return pd.DataFrame(correlation, index=covar.index, columns=covar.columns)


def _condition_covariance(covar: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float]]:
    """Apply the smallest eigenvalue floor that caps condition at ``1e6``.

    The raw covariance is used for cluster discovery.  The conditioned matrix is held
    fixed across every allocation method, avoiding solver failure on near-duplicate fund
    histories without changing the Rolling-Ward hierarchy.
    """
    values = covar.to_numpy(dtype=float)
    largest = float(np.linalg.eigvalsh(values).max())
    eigenvalue_floor = largest / MAX_CONDITION_NUMBER
    factorization = factorize_covariance(values, eigenvalue_floor=eigenvalue_floor)
    conditioned = pd.DataFrame(factorization.covar, index=covar.index, columns=covar.columns)
    denominator = max(float(np.linalg.norm(values, ord="fro")), np.finfo(float).eps)
    diagnostics = {
        "raw_minimum_eigenvalue": factorization.raw_min_eigenvalue,
        "raw_condition_number": factorization.raw_condition_number,
        "conditioned_minimum_eigenvalue": factorization.stabilized_min_eigenvalue,
        "conditioned_condition_number": factorization.stabilized_condition_number,
        "conditioned_eigenvalues_floored": factorization.n_eigenvalues_floored,
        "conditioning_max_eigenvalue_adjustment": (factorization.max_eigenvalue_adjustment),
        "conditioning_relative_frobenius_adjustment": float(
            np.linalg.norm(conditioned.to_numpy() - values, ord="fro") / denominator
        ),
    }
    return conditioned, diagnostics


def _iter_covariance_inputs(
    returns: pd.DataFrame,
    dates: pd.DatetimeIndex,
    model: object,
) -> Iterator[tuple[pd.Timestamp, pd.DataFrame]]:
    """Yield the causal covariance immediately before FactorLasso normalizes it.

    FactorLasso's ``_iter_correlation_inputs`` is the frozen clustering path but exposes
    only normalized correlations.  No stack symbol exports its intermediate rolling
    covariance.  This iterator therefore reuses FactorLasso's own ``get_x_y_np``
    preprocessing and public ``compute_ewm_covar`` recursion one observation at a time.
    The focused regression requires exact equality after correlation normalization.
    """
    dates = pd.DatetimeIndex(dates)
    if len(dates) == 0:
        return
    if str(model.dependence_measure.value) != "pearson" or model.span is None:
        raise ValueError("the frozen U2 covariance path requires Pearson EWMA")
    limited = returns.loc[: dates[-1]]
    dummy = pd.DataFrame(0.0, index=limited.index, columns=["__cluster_dummy__"])
    _, values, valid_mask = get_x_y_np(
        x=dummy,
        y=limited,
        span=model.span,
        demean=model.demean,
    )
    observations = np.where(valid_mask > 0, values, np.nan)
    observation_index = limited.index[1:] if model.demean else limited.index
    covariance = np.zeros((len(returns.columns), len(returns.columns)))
    position = 0
    for date in dates:
        while position < len(observation_index) and observation_index[position] <= date:
            covariance = compute_ewm_covar(
                observations[position],
                span=model.span,
                covar0=covariance,
                is_corr=False,
            )
            position += 1
        yield (
            pd.Timestamp(date),
            pd.DataFrame(covariance.copy(), index=returns.columns, columns=returns.columns),
        )


def _load_date_inputs(
    date: pd.Timestamp,
    full_covar: pd.DataFrame,
    full_corr: pd.DataFrame,
    inputs: d4.UniverseInputs,
) -> tuple[pd.DataFrame, pd.Series, np.ndarray, np.ndarray, dict[str, object]]:
    """Load one frozen Ward hierarchy and validate the reconstructed covariance."""
    ward_path = d4._cache_dir(inputs, "raw") / f"{pd.Timestamp(date):%Y%m%d}.pkl"
    with ward_path.open("rb") as stream:
        ward = pickle.load(stream)
    clusters = ward["clusters"].dropna()
    eligible = inputs.eligibility.columns[inputs.eligibility.loc[date].astype(bool)]
    frozen = inputs.frozen_panel.loc[date].dropna()
    asset_set_match = set(clusters.index) == set(eligible) == set(frozen.index)
    if not asset_set_match:
        raise AssertionError(f"{date:%Y-%m-%d} U2 covariance/cluster asset mismatch")
    covar = full_covar.reindex(index=clusters.index, columns=clusters.index)
    corr = _covariance_to_correlation(covar)
    reference = full_corr.reindex(index=clusters.index, columns=clusters.index)
    correlation_error = float(corr.subtract(reference).abs().to_numpy().max())
    labels, reproduced_linkage, _ = compute_clusters_from_corr_matrix(
        corr,
        cutoff_fraction=inputs.model.cutoff_fraction,
        linkage_method=inputs.model.linkage_method,
        distance_transform=inputs.model.distance_transform,
        n_clusters=inputs.model.n_clusters,
    )
    partition_match = _same_partition(clusters, labels)
    frozen_match = _same_partition(clusters, frozen.reindex(clusters.index))
    if not partition_match or not frozen_match:
        raise AssertionError(f"{date:%Y-%m-%d} reconstructed U2 partition mismatch")
    ward_linkage = np.asarray(ward["linkage"], dtype=float)
    linkage_error = float(np.max(np.abs(ward_linkage - reproduced_linkage)))
    _, single_linkage, _ = compute_clusters_from_corr_matrix(
        corr,
        cutoff_fraction=inputs.model.cutoff_fraction,
        linkage_method="single",
        distance_transform=inputs.model.distance_transform,
        n_clusters=None,
    )
    conditioned, conditioning_diagnostics = _condition_covariance(covar * ANNUALIZATION)
    diagnostics = {
        "asset_set_match": asset_set_match,
        "partition_match": partition_match,
        "frozen_partition_match": frozen_match,
        "correlation_input_max_abs_error": correlation_error,
        "ward_linkage_max_abs_error": linkage_error,
        "assets": len(clusters),
        "clusters": int(clusters.nunique()),
        "ward_path": str(ward_path),
        **conditioning_diagnostics,
    }
    return conditioned, clusters, ward_linkage, single_linkage, diagnostics


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
    """Return per-method risk metrics and full Ward-cluster contributions."""
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
    full_covar: pd.DataFrame,
    full_corr: pd.DataFrame,
    inputs: d4.UniverseInputs,
    fingerprint: str | None = None,
) -> dict[str, object]:
    """Compute the five U2 allocation methods for one rebalance date."""
    covar, clusters, ward_linkage, single_linkage, input_diagnostics = _load_date_inputs(
        date, full_covar, full_corr, inputs
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
        "date": date,
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
        "fingerprint": fingerprint or _input_fingerprint(inputs),
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


def _valid_allocation_cache(date: pd.Timestamp, fingerprint: str) -> bool:
    """Return whether one date has a complete versioned allocation cache."""
    path = _allocation_cache_path(date)
    if not path.exists():
        return False
    with path.open("rb") as stream:
        payload = pickle.load(stream)
    return (
        payload.get("version") == ALLOCATION_CACHE_VERSION
        and payload.get("fingerprint") == fingerprint
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
    """Compute or load every rolling method and assemble labelled panels."""
    dates = _rebalance_dates(inputs)
    fingerprint = _input_fingerprint(inputs)
    if not all(_valid_allocation_cache(date, fingerprint) for date in dates):
        covariance_iterator = _iter_covariance_inputs(inputs.returns, dates, inputs.model)
        correlation_iterator = _iter_correlation_inputs(inputs.returns, list(dates), inputs.model)
        produced = set()
        for (date, full_covar), (corr_date, full_corr) in zip(
            covariance_iterator, correlation_iterator, strict=True
        ):
            if pd.Timestamp(date) != pd.Timestamp(corr_date):
                raise AssertionError("covariance and correlation iterators differ in date")
            started = time.perf_counter()
            payload = _compute_date_payload(
                date, full_covar, full_corr, inputs, fingerprint=fingerprint
            )
            produced.add(pd.Timestamp(date))
            print(
                f"U2 hierarchical risk {date:%Y-%m-%d}: "
                f"{payload['diagnostics']['assets']} assets in "
                f"{time.perf_counter() - started:.2f}s",
                flush=True,
            )
        missing = dates.difference(pd.DatetimeIndex(produced))
        if len(missing):
            raise AssertionError(f"covariance iterator missed dates: {missing.tolist()}")

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


def _performance(
    weights: Mapping[str, pd.DataFrame],
    inputs: d4.UniverseInputs,
) -> pd.DataFrame:
    """Backtest all fully invested risk portfolios with frozen U2 mechanics."""
    daily = funds._read_daily()
    rolling_aum = u2_aum._rolling_aum()
    eligibility_all = u2_sensitivity._eligibilities(daily, inputs.dates, rolling_aum)["aum_100m"]
    headline = _headline_dates(inputs)
    window = u2_sensitivity._window(
        funds._performance_prices(daily), eligibility_all, WINDOW, headline
    )
    rows = []
    for method in METHODS:
        net, gross = funds._backtest(
            window["prices"],
            weights[method],
            COST_BPS / 10000.0,
            f"u2_hierarchical_risk_{method}",
        )
        rows.append(
            {
                "universe": "blackrock_funds",
                "analysis_window": WINDOW,
                "method": method,
                "cost_bps_one_way": COST_BPS,
                **u2_sleeves._performance_payload(net, gross, window["ew_nav"]),
            }
        )
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


def _method_comparison(performance: pd.DataFrame, risk_summary: pd.DataFrame) -> pd.DataFrame:
    """Compare every cluster-aware method with flat ERC, never EW-all."""
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


def _allocation_sleeve_summary(
    weights: Mapping[str, pd.DataFrame],
) -> pd.DataFrame:
    """Summarize capital allocation to the frozen U2 broad sleeves."""
    rows = []
    for method, panel in weights.items():
        mapping = u2_sleeves._broad_sleeves(panel.columns)
        if mapping.isna().any():
            raise AssertionError("a U2 allocation column lacks a broad sleeve")
        grouped = panel.T.groupby(mapping).sum().T
        for sleeve in u2_sleeves.SLEEVES:
            series = grouped[sleeve]
            rows.append(
                {
                    "method": method,
                    "sleeve": sleeve,
                    "capital_share_mean": float(series.mean()),
                    "capital_share_median": float(series.median()),
                    "capital_share_min": float(series.min()),
                    "capital_share_max": float(series.max()),
                }
            )
    return pd.DataFrame(rows)


def _signal_weights(
    inputs: d4.UniverseInputs,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """Rebuild the accepted U2 global and cluster-short-overlay decision weights."""
    daily = funds._read_daily()
    headline = _headline_dates(inputs)
    eligibility = inputs.eligibility.reindex(index=headline).astype(bool)
    rolling_aum = u2_aum._rolling_aum()
    monthly_dates = funds._native_returns(daily, "ME").index
    monthly_eligibility = u2_sensitivity._eligibilities(daily, monthly_dates, rolling_aum)[
        "aum_100m"
    ]
    clusters = inputs.frozen_panel.reindex(index=headline, columns=eligibility.columns)
    global_scores, cluster_scores, signal_diagnostics = u2_sensitivity._signal_panels(
        daily, headline, eligibility, monthly_eligibility, clusters
    )
    sleeve_map = u2_sleeves._broad_sleeves(eligibility.columns)
    sleeve_panel = u2_sleeves._sleeve_panel(headline, sleeve_map)
    rebuilt, weight_diagnostics = u2_sensitivity._weights(
        global_scores, cluster_scores, eligibility, clusters, sleeve_panel
    )
    dates = _rebalance_dates(inputs)
    weights = {
        "global_long_short": rebuilt["global"].reindex(index=dates),
        "rolling_ward_cluster_short_overlay": rebuilt["hybrid"].reindex(index=dates),
    }
    diagnostics = pd.DataFrame(
        [
            {
                **signal_diagnostics,
                **weight_diagnostics,
                "scheduled_dates": len(dates),
            }
        ]
    )
    return weights, diagnostics


def _signal_risk_diagnostic(
    inputs: d4.UniverseInputs,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Decompose accepted U2 long-short risk by frozen Ward cluster."""
    signal_weights, preflight = _signal_weights(inputs)
    dates = _rebalance_dates(inputs)
    rows = []
    contribution_rows = []
    for date, full_covar in _iter_covariance_inputs(inputs.returns, dates, inputs.model):
        ward_path = d4._cache_dir(inputs, "raw") / f"{date:%Y%m%d}.pkl"
        with ward_path.open("rb") as stream:
            clusters = pickle.load(stream)["clusters"].dropna()
        covar = _condition_covariance(
            full_covar.reindex(index=clusters.index, columns=clusters.index) * ANNUALIZATION
        )[0]
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
                    "maximum_absolute_cluster_net_exposure": float(grouped_exposure.abs().max()),
                    "effective_long_assets": float(long_side.sum() ** 2 / long_side.pow(2.0).sum()),
                    "effective_short_assets": float(
                        short_side.sum() ** 2 / short_side.pow(2.0).sum()
                    ),
                    **metrics,
                }
            )
    return (
        pd.DataFrame(rows),
        pd.concat(contribution_rows, ignore_index=True),
        preflight,
    )


def _signal_risk_summary(signal_risk: pd.DataFrame) -> pd.DataFrame:
    """Aggregate long-short risk decomposition for the two accepted strategies."""
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


def _paper_comparison(performance: pd.DataFrame) -> pd.DataFrame:
    """Return the frozen U2 long-only and long-short paper comparison rows."""
    labels = {
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
    indexed = performance.set_index("method")
    for method_id in PAPER_LONG_ONLY_METHODS:
        label, role, benchmark = labels[method_id]
        source = indexed.loc[method_id]
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
                "one_way_turnover_annualized": source["one_way_turnover_annualized"],
                "cost_bps_one_way": source["cost_bps_one_way"],
                "source": "risk_allocation/performance.csv",
            }
        )
    signal_path = d4._universe_root("blackrock_funds") / "performance.csv"
    signal = pd.read_csv(signal_path, float_precision="round_trip")
    signal = signal.loc[signal["is_primary_window"].astype(bool)].set_index("leg")
    signal_labels = {
        "global": ("Global momentum rank", "control", "global rank"),
        "cluster_raw": (
            "Rolling-Ward cluster-short overlay",
            "proposed clustering",
            "global rank",
        ),
    }
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
                "one_way_turnover_annualized": source["one_way_turnover_annualized"],
                "cost_bps_one_way": source["cost_bps_one_way"],
                "source": str(signal_path),
            }
        )
    return pd.DataFrame(rows)


def _acceptance(
    diagnostics: pd.DataFrame,
    risk_per_date: pd.DataFrame,
    weights: Mapping[str, pd.DataFrame],
    inputs: d4.UniverseInputs,
    signal_preflight: pd.DataFrame,
    paper_comparison: pd.DataFrame,
) -> pd.DataFrame:
    """Return every U2 measured-versus-tolerance acceptance line."""
    dates = _rebalance_dates(inputs)
    eligibility = inputs.eligibility.reindex(index=dates).astype(bool)
    outside = max(
        float(frame.where(~eligibility, 0.0).abs().to_numpy().max()) for frame in weights.values()
    )
    rolling_aum = u2_aum._rolling_aum()
    aum_at_dates = u2_aum._aum_for_dates(dates, rolling_aum).reindex(columns=eligibility.columns)
    aum_violations = int((eligibility & ~aum_at_dates.gt(100.0)).sum().sum())
    signal_error = max(
        abs(float(value)) for column, value in signal_preflight.iloc[0].items() if "error" in column
    )
    checks = [
        (
            "scheduled allocation dates",
            float(len(diagnostics)),
            float(len(dates)),
            len(diagnostics) == len(dates),
        ),
        (
            "covariance and Ward asset-set match share",
            float(diagnostics["asset_set_match"].mean()),
            PARTITION_TOLERANCE,
            diagnostics["asset_set_match"].all(),
        ),
        (
            "reconstructed Ward partition match share",
            float(diagnostics["partition_match"].mean()),
            PARTITION_TOLERANCE,
            diagnostics["partition_match"].all(),
        ),
        (
            "frozen AUM100 partition match share",
            float(diagnostics["frozen_partition_match"].mean()),
            PARTITION_TOLERANCE,
            diagnostics["frozen_partition_match"].all(),
        ),
        (
            "FactorLasso covariance-to-correlation maximum error",
            float(diagnostics["correlation_input_max_abs_error"].max()),
            CORRELATION_TOLERANCE,
            float(diagnostics["correlation_input_max_abs_error"].max()) <= CORRELATION_TOLERANCE,
        ),
        (
            "frozen Ward linkage maximum error",
            float(diagnostics["ward_linkage_max_abs_error"].max()),
            CORRELATION_TOLERANCE,
            float(diagnostics["ward_linkage_max_abs_error"].max()) <= CORRELATION_TOLERANCE,
        ),
        (
            "maximum conditioned covariance condition number",
            float(diagnostics["conditioned_condition_number"].max()),
            MAX_CONDITION_NUMBER * (1.0 + CORRELATION_TOLERANCE),
            float(diagnostics["conditioned_condition_number"].max())
            <= MAX_CONDITION_NUMBER * (1.0 + CORRELATION_TOLERANCE),
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
        ("AUM <= USD100m eligible observations", float(aum_violations), 0.0, aum_violations == 0),
        (
            "maximum Euler risk reconciliation error",
            float(diagnostics["maximum_risk_reconciliation_error"].max()),
            WEIGHT_TOLERANCE,
            float(diagnostics["maximum_risk_reconciliation_error"].max()) <= WEIGHT_TOLERANCE,
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
            float(len(dates) * len(METHODS)),
            len(risk_per_date) == len(dates) * len(METHODS),
        ),
        (
            "maximum signal lookahead days",
            float(
                max(
                    signal_preflight["max_global_lookahead_days"].max(),
                    signal_preflight["max_cluster_lookahead_days"].max(),
                )
            ),
            0.0,
            max(
                signal_preflight["max_global_lookahead_days"].max(),
                signal_preflight["max_cluster_lookahead_days"].max(),
            )
            <= 0.0,
        ),
        (
            "maximum signal-weight construction error",
            signal_error,
            WEIGHT_TOLERANCE,
            signal_error <= WEIGHT_TOLERANCE,
        ),
        ("one-way transaction cost bps", COST_BPS, 20.0, COST_BPS == 20.0),
        ("paper comparison rows", float(len(paper_comparison)), 5.0, len(paper_comparison) == 5),
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
    """Return the frozen U2 experimental design and interpretation boundaries."""
    return pd.DataFrame(
        [
            {
                "universe": "U2 BlackRock funds; rolling average AUM > USD100m",
                "analysis_window": "2009-08-31 through 2026-06-30",
                "risk_covariance": (
                    "causal W-THU EWMA span 156; annualized by 52; "
                    "minimal eigenvalue floor to condition <= 1e6"
                ),
                "hierarchy": "W-THU returns, EWMA span 156, Pearson, 1-rho, cutoff 0.60",
                "ward_role": "paper cluster hierarchy",
                "single_role": "canonical HRP literature comparator",
                "risk_budget_exponents": "1.0 flat|0.5 sqrt-size|0.0 equal-cluster",
                "constraints": "long-only, fully invested",
                "rebalance": SCHEDULE,
                "implementation_lag": 1,
                "cost_bps_one_way": COST_BPS,
                "performance_benchmark": "flat ERC only",
                "long_short_benchmark": "matched global momentum rank",
                "ew_role": "market reference for beta/alpha only",
                "ward_herc": "excluded by owner ruling",
                "nested_cluster_risk_budgeting": "excluded",
                "snapshot_method_name": "rolling EWMA-Ward correlation clustering",
                "risk_strategy_name": "Rolling-Ward HRP",
                "signal_strategy_name": "Rolling-Ward cluster-short overlay",
                "signal_strategy_definition": "global-rank long / cluster-relative short",
            }
        ]
    )


def _source_manifest(inputs: d4.UniverseInputs) -> pd.DataFrame:
    """Record runner, shared allocator, frozen sources, and sample Ward cache."""
    runner = Path(__file__)
    allocation_module = runner.with_name("hierarchical_risk_allocations.py")
    sample_date = _rebalance_dates(inputs)[-1]
    sample_ward = d4._cache_dir(inputs, "raw") / f"{sample_date:%Y%m%d}.pkl"
    signal_path = d4._universe_root("blackrock_funds") / "performance.csv"
    paths = [runner, allocation_module, sample_ward, signal_path, *inputs.input_paths]
    return pd.DataFrame(
        [{"kind": "source", "path": str(path), "sha256": _sha256(path)} for path in paths]
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
    signal_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Build U2 allocation and signal-risk comparison exhibits."""
    labels = {
        "flat_erc": "Flat ERC",
        "cluster_rb_alpha_0_5": "Cluster RB sqrt(n)",
        "cluster_rb_alpha_0": "Cluster RB equal",
        "ward_hrp": "Rolling-Ward HRP",
        "single_hrp": "Single-HRP",
        "global_long_short": "Global rank",
        "rolling_ward_cluster_short_overlay": "Rolling-Ward overlay",
    }
    perf = performance.set_index("method").rename(index=labels)
    risk = risk_summary.set_index("method").rename(index=labels)
    signal = signal_summary.set_index("leg").rename(index=labels)
    figures = {
        "u2_allocation_performance.png": _four_panel_bars(
            perf,
            specifications=(
                ("net_return_annualized", "Net annualized return", "{:.1%}"),
                ("volatility_annualized", "Realized annualized volatility", "{:.1%}"),
                ("sharpe_rf0", "Net Sharpe (rf=0)", "{:.2f}"),
                ("one_way_turnover_annualized", "Annualized one-way turnover", "{:.1f}x"),
            ),
            title="U2: standard risk-allocation methods",
        ),
        "u2_allocation_risk_structure.png": _four_panel_bars(
            risk,
            specifications=(
                ("portfolio_ex_ante_volatility_mean", "Mean ex-ante volatility", "{:.1%}"),
                ("effective_risk_clusters_absolute_mean", "Effective Ward risk clusters", "{:.1f}"),
                (
                    "maximum_absolute_cluster_risk_share_mean",
                    "Largest absolute cluster-risk share",
                    "{:.1%}",
                ),
                ("diversification_ratio_mean", "Diversification ratio", "{:.2f}"),
            ),
            title="U2: ex-ante risk through Rolling-Ward clusters",
        ),
        "u2_signal_risk_structure.png": _four_panel_bars(
            signal,
            specifications=(
                ("portfolio_ex_ante_volatility_mean", "Mean ex-ante volatility", "{:.1%}"),
                ("effective_risk_clusters_absolute_mean", "Effective Ward risk clusters", "{:.1f}"),
                (
                    "maximum_absolute_cluster_risk_share_mean",
                    "Largest absolute cluster-risk share",
                    "{:.1%}",
                ),
                ("cluster_net_exposure_l1_mean", "Ward-cluster net-exposure L1", "{:.2f}"),
            ),
            title="U2 long-short momentum: overlay versus global rank",
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
    """Execute the complete cache-first U2 hierarchy/risk experiment."""
    started = time.perf_counter()
    inputs = d4._u2_inputs()
    weights, risk_per_date, contributions, diagnostics = _build_allocations(inputs)
    performance = _performance(weights, inputs)
    risk_summary = _risk_method_summary(risk_per_date)
    comparison = _method_comparison(performance, risk_summary)
    sleeve_summary = _allocation_sleeve_summary(weights)
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
        "design": _design(),
        "performance": performance,
        "comparison_vs_flat_erc": comparison,
        "allocation_sleeve_summary": sleeve_summary,
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
    """Run U2 hierarchical risk allocation and deterministic replay."""
    replay = verify_determinism()
    print(
        f"U2 hierarchical risk allocation: PASS ({len(replay)}/{len(replay)} deterministic)",
        flush=True,
    )


if __name__ == "__main__":
    main()
